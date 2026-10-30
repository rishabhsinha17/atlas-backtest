"""Red-team conformance probes: cross-sectional factor strategies S085-S100 (atlasbt/strategies/factors.py).

SYNTHETIC daily data, mechanics only. Each test builds the exact sheet scenario and asserts the
engine-level behaviour (signal bar, fill bar, basket membership, PIT fundamentals).
"""
import numpy as np
import pandas as pd
import pytest

from atlasbt.core.types import RunStatus
from atlasbt.strategies import factors as F

from rtc_fs_helpers import CAL, close_ts, fund, nxt, open_ts, orders, path, portal, run, sess, sides_at

SIGNAL = pd.Timestamp("2023-06-30")
SIG_CLOSE = close_ts(SIGNAL)
START, END = "2023-06-26", "2023-07-07"
ME = F.month_end_sessions(CAL, SIGNAL, 14)          # ME[0] = signal month end, ME[k] = k months earlier
DATES = sess("2021-01-04", "2023-07-31")
STOCKS = [f"S{i:02d}" for i in range(20)]
SECT = ["Information Technology", "Financials", "Health Care", "Energy", "Industrials"]


def master(symbols, sectors=None, sub=None, real_columns=False):
    n = len(symbols)
    sec = sectors if sectors is not None else [SECT[i % 5] for i in range(n)]
    sb = sub if sub is not None else [f"SUB{i % 10}" for i in range(n)]
    if real_columns:   # column names of atlasbt.data.portal.Portal.security_master (checked 2026-10)
        return pd.DataFrame({"symbol": list(symbols), "sector": sec, "exchange": ["NYSE"] * n,
                             "gics_sector_current": sec, "gics_sub_industry_current": sb,
                             "sic": [3570 + i for i in range(n)]})
    return pd.DataFrame({"symbol": list(symbols), "gics_sector": sec, "gics_sub_industry": sb,
                         "exchange": ["NYSE"] * n, "sic": [3570 + i for i in range(n)]})


def jump_paths(jumps: dict, drift_step=1e-6):
    """Flat paths with tiny distinct drifts (no ties) plus single-day jumps {sym: [(session, ret), ...]}."""
    di = pd.DatetimeIndex(DATES)
    out = {}
    for i, s in enumerate(STOCKS):
        r = np.full(len(di), drift_step * i)
        for d, x in jumps.get(s, []):
            r[di.get_loc(pd.Timestamp(d))] += x
        out[s] = path(di, r, 50.0)
    return out


def after(d):
    return nxt(d)


# ------------------------------------------------------------------------------------- S085
def test_s085_formation_window_is_exactly_M12_to_M1_and_fills_next_open():
    """Sheet: 'rank cumulative total return t-12 to t-2, skipping most recent 1m ... enter next first-month RTH
    open'. Jumps ON the M12 session or AFTER M1 must not count; jumps right after M12 and ON M1 must."""
    j = {"S01": [(after(ME[12]), 0.40)], "S02": [(ME[1], 0.35)],          # counted -> long
         "S03": [(ME[12], 0.60)], "S04": [(after(ME[1]), 0.60)],          # excluded
         "S05": [(after(ME[12]), -0.40)], "S06": [(ME[1], -0.35)],        # counted -> short
         "S07": [(ME[12], -0.60)], "S08": [(after(ME[1]), -0.60)]}        # excluded
    res = run(F.S085(), portal(jump_paths(j), master=master(STOCKS)), START, END)
    assert res.status == RunStatus.OK
    lo, sh = sides_at(res, SIG_CLOSE)
    assert lo == {"S01", "S02"} and sh == {"S05", "S06"}
    f = res.fills
    assert set(pd.DatetimeIndex(f["ts"])) == {open_ts(nxt(SIGNAL))}


def test_s085_industry_neutral_unclassified_loser_is_not_bought():
    """Optional 'industry neutral' sort: a name without a sector forms a one-name group; quantile_bucket puts a
    single name in the TOP decile, so the worst loser is bought."""
    syms = STOCKS + ["S20"]
    j = {"S20": [(after(ME[12]), -0.50)]}
    px = jump_paths(j)
    px["S20"] = path(DATES, np.where(pd.DatetimeIndex(DATES) == after(ME[12]), -0.50, 0.0), 50.0)
    m = master(STOCKS, sectors=["Information Technology"] * 10 + ["Financials"] * 10)   # S20 not in master
    st = F.S085(industry_neutral=True)
    res = run(st, portal(px, master=m, universe=syms), START, END)
    lo, sh = sides_at(res, SIG_CLOSE)
    assert "S20" not in lo, "worst 12-2 loser (no sector) ends up in the LONG basket"


# ------------------------------------------------------------------------------------- S086
def test_s086_window_is_M6_to_M0():
    """Sheet: 'rank total return over months t-6 to t-1' (t = holding month): signal-month return counts,
    the M6 session itself does not."""
    j = {"S01": [(ME[0], 0.40)], "S02": [(after(ME[6]), 0.35)], "S03": [(ME[6], 0.60)],
         "S05": [(ME[0], -0.40)], "S06": [(after(ME[6]), -0.35)], "S07": [(ME[6], -0.60)]}
    res = run(F.S086(), portal(jump_paths(j), master=master(STOCKS)), START, END)
    lo, sh = sides_at(res, SIG_CLOSE)
    assert lo == {"S01", "S02"} and sh == {"S05", "S06"}


# ------------------------------------------------------------------------------------- S087
def _s087_portal(real_columns):
    di = pd.DatetimeIndex(DATES)
    px, rows = {}, []
    for i, s in enumerate(STOCKS):
        k = i % 10
        r = np.where((di > ME[7]) & (di <= ME[1]), 0.0005 * (k - 4.5), 0.0) + 1e-6 * i
        px[s] = path(di, r, 30.0 + i)
        rows.append(fund(s, "shares_outstanding", 1e8 * (1 + i), "2023-03-31", "2023-05-01 10:00"))
    return portal(px, master=master(STOCKS, real_columns=real_columns), fundamentals=rows)


def test_s087_control_trades_with_author_column_names():
    res = run(F.S087(), _s087_portal(False), START, END)
    lo, sh = sides_at(res, SIG_CLOSE)
    assert lo and sh


def test_s087_trades_with_real_portal_security_master_columns():
    """Real Portal.security_master exposes gics_sub_industry_CURRENT (no 'gics_sub_industry'/'industry'
    column): SecMaster.industry() returns None for every name -> baseline GICS mode never trades (silent)."""
    res = run(F.S087(), _s087_portal(True), START, END)
    lo, sh = sides_at(res, SIG_CLOSE)
    assert lo and sh, "S087 (industry_map=GICS) silently produces no trades on the real security master"


# ------------------------------------------------------------------------------------- S088
ETF9 = list(F.SECTOR_SPDR_9)


def _s088_world(n_it, others, rel_stock=False, seed=3, start="2014-01-02", signal=SIGNAL):
    dates = sess(start, (pd.Timestamp(signal) + pd.Timedelta(days=12)).strftime("%Y-%m-%d"))
    di = pd.DatetimeIndex(dates)
    me = F.month_end_sessions(CAL, signal, 13)
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.0003, 0.01, len(di))
    px = {"SPY": path(di, mkt, 200.0)}
    for e in ETF9:
        px[e] = path(di, mkt + rng.normal(0, 0.004, len(di)), 50.0)
    fw = (di > me[12]) & (di <= me[1])
    syms, secs = [], []
    for i in range(n_it):
        s = f"T{i:02d}"
        px[s] = path(di, mkt + rng.normal(0, 0.003, len(di)) + np.where(fw, 0.0004 * (i - n_it / 2), 0.0), 40.0)
        syms.append(s)
        secs.append("Information Technology")
    for i, sec in enumerate(others):
        s = f"O{i:02d}"
        px[s] = path(di, mkt + rng.normal(0, 0.003, len(di)) - np.where(fw, 0.0003 * (i + 1), 0.0), 40.0)
        syms.append(s)
        secs.append(sec)
    if rel_stock:
        xlre_d = di[di >= pd.Timestamp("2015-10-08")]
        px["XLRE"] = path(xlre_d, mkt[di >= pd.Timestamp("2015-10-08")], 30.0)
        px["R00"] = path(di, mkt + rng.normal(0, 0.003, len(di)) + np.where(fw, 0.004, 0.0), 40.0)
        syms.append("R00")
        secs.append("Real Estate")
    return px, syms, secs


def test_s088_real_estate_name_ranked_after_xlre_inception():
    """Deviation: 'Sector SPDR before its inception ... market-only regression'. After XLRE's inception
    (2015-10-08) but before it has reg_window sessions, the code regresses on XLRE anyway -> NaN residuals ->
    the strongest residual-momentum name is silently dropped from the ranking."""
    sig = pd.Timestamp("2016-06-30")
    px, syms, secs = _s088_world(19, [], rel_stock=True, signal=sig)
    st = F.S088()
    run(st, portal(px, master=master(syms, sectors=secs), universe=syms), "2016-06-27", "2016-07-08")
    assert "R00" in st.book.get("long", []), "top residual-momentum REIT excluded (undeclared)"


def test_s088_control_same_stock_ranked_when_classified_it():
    sig = pd.Timestamp("2016-06-30")
    px, syms, secs = _s088_world(19, [], rel_stock=True, signal=sig)
    secs = ["Information Technology"] * len(secs)
    st = F.S088()
    run(st, portal(px, master=master(syms, sectors=secs), universe=syms), "2016-06-27", "2016-07-08")
    assert "R00" in st.book.get("long", [])


def test_s088_stock_book_dollar_neutral_when_sector_cap_binds():
    """Sheet: 'preserve dollar and estimated beta neutrality'; deviation claims 'equal-weight stock book is dollar
    neutral'. When the long decile is concentrated in one sector the 10% sector cap removes long weight only."""
    others = [s for s in SECT[1:] for _ in range(6)] + ["Utilities"] * 6
    px, syms, secs = _s088_world(30, others, start="2020-06-01")
    st = F.S088()
    run(st, portal(px, master=master(syms, sectors=secs), universe=syms), START, END)
    w = {s: v for s, v in st.last_weights.items() if s != "SPY"}
    assert len([v for v in w.values() if v > 0]) >= 6
    assert abs(sum(w.values())) < 1e-9, f"stock book net = {sum(w.values()):+.4f} NAV"


# ------------------------------------------------------------------------------------- S091
def _s091():
    di = pd.DatetimeIndex(DATES)
    px, rows = {}, []
    for i, s in enumerate(STOCKS):
        c = pd.Series(100.0, index=di)
        if s == "S03":           # December ME low -> high B/M on Dec ME (June price is back to 100)
            c[(di >= pd.Timestamp("2022-12-01")) & (di <= pd.Timestamp("2022-12-31"))] = 10.0
        px[s] = c
        sh = 1e6 * (1 + i)
        rows += [fund(s, "shares_outstanding", sh, "2022-09-30", "2022-11-01 10:00"),
                 fund(s, "shares_outstanding", sh, "2023-03-31", "2023-05-01 10:00")]
        if s == "S05":           # FY ends March: FF uses FY ending in t-1 (Mar-2022), not Mar-2023
            rows += [fund(s, "net_income_fy", 1, "2022-03-31", "2022-05-20 10:00"),
                     fund(s, "stockholders_equity", 1e5, "2022-03-31", "2022-05-20 10:00"),
                     fund(s, "net_income_fy", 1, "2023-03-31", "2023-05-20 10:00"),
                     fund(s, "stockholders_equity", 1e10, "2023-03-31", "2023-05-20 10:00")]
            continue
        se = 1e10 if s == "S12" else 1e8 * (1 + 0.01 * i)
        rows += [fund(s, "net_income_fy", 1, "2022-12-31", "2023-02-20 10:00"),
                 fund(s, "stockholders_equity", se, "2022-12-31", "2023-02-20 10:00")]
        if s == "S12":           # FF BE = SE + DT - PS: preferred 99.9% of SE -> lowest B/M in its size group
            rows.append(fund(s, "preferred_stock", 0.999e10, "2022-12-31", "2023-02-20 10:00"))
    st = F.S091()
    res = run(st, portal(px, master=master(STOCKS), fundamentals=rows), START, END)
    return res, st


_S091 = {}


def s091():
    if not _S091:
        _S091["r"] = _s091()
    return _S091["r"]


def test_s091_december_me_and_fiscal_year_ending_in_t_minus_1():
    res, st = s091()
    lo, sh = sides_at(res, SIG_CLOSE)
    assert "S03" in lo, "B/M must use December t-1 market equity"
    assert "S05" in sh, "B/M must use the fiscal year ending in t-1 (Mar-2022), not Mar-2023"
    assert all(pd.DatetimeIndex(res.fills["ts"]) == open_ts(nxt(SIGNAL))), "rebalance at first July open"


def test_s091_book_equity_subtracts_preferred_stock():
    """Deviation claims 'the data layer's XBRL map only has StockholdersEquity'; docs/DATA.md §4.4 lists
    preferred_stock and deferred_taxes and the FF BE formula. A firm whose equity is almost all preferred is
    ranked as the CHEAPEST value stock."""
    res, st = s091()
    lo, sh = sides_at(res, SIG_CLOSE)
    assert "S12" not in lo and "S12" in sh


# ------------------------------------------------------------------------------------- S092
def test_s092_ttm_and_filings_public_before_signal():
    rows = []
    for i, s in enumerate(STOCKS):
        rows += [fund(s, "shares_outstanding", 1e6, "2023-03-31", "2023-05-01 10:00"),
                 fund(s, "net_income_fy", 90 + i, "2021-12-31", "2022-02-20 10:00"),
                 fund(s, "net_income_fy", 100 + i, "2022-12-31", "2023-02-20 10:00"),
                 fund(s, "net_income_q", 20, "2022-03-31", "2022-05-05 10:00"),
                 fund(s, "net_income_q", 40, "2023-03-31", "2023-05-05 10:00")]
    rows.append(fund("S19", "net_income_q", 5000, "2023-06-30", "2023-06-30 16:30"))  # accepted after close
    px = {s: path(DATES, np.full(len(DATES), 1e-6 * i), 50.0) for i, s in enumerate(STOCKS)}
    st = F.S092()
    res = run(st, portal(px, master=master(STOCKS), fundamentals=rows), START, END)
    assert st.fh.ttm("S00", "net_income", SIG_CLOSE) == pytest.approx(100 + 40 - 20)
    assert st.fh.ttm("S19", "net_income", SIG_CLOSE) == pytest.approx(119 + 40 - 20)
    lo, sh = sides_at(res, SIG_CLOSE)
    assert len(lo) == 4 and len(sh) == 4          # quintiles of 20


# ------------------------------------------------------------------------------------- S099
def test_s099_asset_growth_uses_fiscal_year_ending_in_t_minus_1():
    """Fama-French CMA at June t: asset growth of the fiscal year ending in calendar t-1 (as S091 does with
    not_after). S099 takes whatever FY is latest at June end, so a March-FY firm uses FY ending Mar t."""
    rows = []
    for i, s in enumerate(STOCKS):
        if s == "S07":
            for pe, kn, a in (("2021-03-31", "2021-05-20", 100), ("2022-03-31", "2022-05-20", 101),
                              ("2023-03-31", "2023-05-20", 300)):
                rows += [fund(s, "net_income_fy", 1, pe, kn + " 10:00"), fund(s, "total_assets", a, pe, kn + " 10:00")]
            continue
        g = 1.10 + 0.01 * i
        for pe, kn, a in (("2021-12-31", "2022-02-20", 100), ("2022-12-31", "2023-02-20", 100 * g)):
            rows += [fund(s, "net_income_fy", 1, pe, kn + " 10:00"), fund(s, "total_assets", a, pe, kn + " 10:00")]
    px = {s: path(DATES, np.zeros(len(DATES)), 50.0) for s in STOCKS}
    res = run(F.S099(), portal(px, master=master(STOCKS), fundamentals=rows), START, END)
    lo, sh = sides_at(res, SIG_CLOSE)
    assert "S07" in lo, "FY Mar-2022 growth (1%) is the lowest -> long; code uses FY Mar-2023 (+197%) -> short"
