"""Red-team conformance probes: stat-arb / relative value S106-S115, S123 (atlasbt/strategies/statarb.py).

SYNTHETIC daily data, mechanics only.
"""
import numpy as np
import pandas as pd
import pytest

from atlasbt.core.types import RunStatus
from atlasbt.strategies import factors as F
from atlasbt.strategies import statarb as SA

from rtc_fs_helpers import (CAL, NAV0, close_ts, nxt, open_ts, orders, path, portal, run, sess, sides_at)


def master(symbols, sectors, subs, real_columns=False):
    n = len(symbols)
    # the real Portal security master always carries 'sic'; S107/S113 default to 3-digit SIC groups
    # (integrator decision 2026-10-29): one SIC per sub-industry label here
    sic = [6021 if s_ == "Banks" else 1311 if s_ == "Oil" else 7372 for s_ in subs]
    if real_columns:   # atlasbt.data.portal.Portal.security_master column names
        return pd.DataFrame({"symbol": symbols, "sector": sectors, "exchange": ["NYSE"] * n,
                             "gics_sector_current": sectors, "gics_sub_industry_current": subs, "sic": sic})
    # 'sector' (SIC->GICS-like) is always present on the real Portal; strategies read only that column
    return pd.DataFrame({"symbol": symbols, "sector": sectors, "gics_sector": sectors, "gics_sub_industry": subs,
                         "exchange": ["NYSE"] * n, "sic": sic})


# ------------------------------------------------------------------------------------- S107 / S113 industry
def _coint_world():
    dates = sess("2022-01-03", "2023-07-31")
    rng = np.random.default_rng(11)
    x = np.cumsum(rng.normal(0, 0.01, len(dates)))
    e = np.zeros(len(dates))
    for t in range(1, len(dates)):
        e[t] = 0.5 * e[t - 1] + rng.normal(0, 0.004)
    di = pd.DatetimeIndex(dates)
    px = {"AAA": pd.Series(50 * np.exp(x + e), index=di), "BBB": pd.Series(40 * np.exp(x), index=di),
          "CCC": pd.Series(30 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))), index=di)}
    return px


@pytest.mark.parametrize("real", [False, True], ids=["author_columns", "real_portal_columns"])
def test_s107_industry_pairs_found(real):
    """S107 'industry pairs' = same GICS sub-industry via SecMaster.INDUSTRY_COLS
    ('gics_sub_industry','gics_industry','industry'); the real Portal exposes gics_sub_industry_current."""
    px = _coint_world()
    syms = list(px)
    m = master(syms, ["Financials"] * 3, ["Banks"] * 3, real_columns=real)
    st = SA.S107()
    run(st, portal(px, master=m), "2023-06-26", "2023-07-07")
    assert st.active, "no industry pairs formed -> strategy silently never trades"


def _mom_pair_world():
    dates = sess("2022-06-01", "2023-07-31")
    di = pd.DatetimeIndex(dates)
    rng = np.random.default_rng(5)
    mkt = rng.normal(0.0002, 0.01, len(di))
    me = F.month_end_sessions(CAL, pd.Timestamp("2023-06-30"), 3)
    win = (di > me[3]) & (di <= me[0])
    px = {"SPY": path(di, mkt, 300.0),
          "WIN": path(di, mkt + rng.normal(0, 0.002, len(di)) + np.where(win, 0.004, 0.0), 50.0),
          "LOS": path(di, mkt + rng.normal(0, 0.002, len(di)) - np.where(win, 0.0, 0.0), 50.0)}
    return px


@pytest.mark.parametrize("real", [False, True], ids=["author_columns", "real_portal_columns"])
def test_s113_matching_industry_pair_traded(real):
    px = _mom_pair_world()
    m = master(["WIN", "LOS"], ["Energy"] * 2, ["Oil"] * 2, real_columns=real)
    res = run(SA.S113(), portal(px, master=m, universe=["WIN", "LOS"]), "2023-06-26", "2023-07-07")
    lo, sh = sides_at(res, close_ts("2023-06-30"))
    assert lo == {"WIN"} and sh == {"LOS"}


# ------------------------------------------------------------------------------------- S106
def test_s106_short_rich_long_cheap_equal_dollars_and_converged_exit():
    dates = sess("2022-01-03", "2023-03-31")
    di = pd.DatetimeIndex(dates)
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.01, len(di))
    ra = base + rng.normal(0, 0.001, len(di))
    rb = base + rng.normal(0, 0.001, len(di))
    start = pd.Timestamp("2023-01-03")
    k = di.get_loc(pd.Timestamp("2023-01-10"))
    ra[k] += 0.03                       # A becomes rich vs B: z ~ 2.5 (formation spread sd ~1.7%)
    ra[k + 5] -= 0.04                   # reverts below the formation mean -> z < 0.5
    px = {"A": path(di, ra, 100.0), "B": path(di, rb, 100.0)}
    m = master(["A", "B"], ["Energy"] * 2, ["Oil"] * 2)
    st = SA.S106()
    res = run(st, portal(px, master=m), start, "2023-02-15")
    lo, sh = sides_at(res, close_ts(di[k]))
    assert lo == {"B"} and sh == {"A"}
    f = res.fills
    ent = f[pd.DatetimeIndex(f["ts"]) == open_ts(di[k + 1])]
    notional = (ent["quantity"].abs() * ent["price"]).to_numpy()
    assert len(ent) == 2 and np.allclose(notional, 0.05 * NAV0, rtol=0.01)
    assert any(ev[2] == "exit" and ev[3] == "converged" for ev in st.events)


# ------------------------------------------------------------------------------------- S110
def test_s110_not_testable_without_inav():
    dates = sess("2023-01-03", "2023-02-28")
    px = {s: path(dates, np.zeros(len(dates)), 100.0) for s in ("SPY", "QQQ", "IWM", "DIA")}
    from atlasbt.engine.backtester import run_backtest
    from rtc_fs_helpers import zero_cost
    r = run_backtest(SA.S110(), portal(px), "2023-01-10", "2023-02-10", zero_cost())
    assert r.status == RunStatus.NOT_TESTABLE


# ------------------------------------------------------------------------------------- S114
def test_s114_shock_entry_next_open_exit_next_close():
    dates = sess("2022-01-03", "2023-03-31")
    di = pd.DatetimeIndex(dates)
    rng = np.random.default_rng(9)
    mkt = rng.normal(0, 0.005, len(di))
    shock = pd.Timestamp("2023-03-01")
    k = di.get_loc(shock)
    px = {"SPY": path(di, mkt, 300.0)}
    for e in F.SECTOR_SPDR_9:
        r = mkt + rng.normal(0, 0.003, len(di))
        if e == "XLK":
            r[k] = 0.05
        px[e] = path(di, r, 50.0)
    names = [f"K{i}" for i in range(10)]
    for i, s in enumerate(names):
        r = mkt + rng.normal(0, 0.002, len(di))
        if True:
            r[k] = 0.05 + 0.004 * (i - 4.5)        # beta-residual ranks by i
        px[s] = path(di, r, 40.0)
    m = master(names, ["Information Technology"] * 10, ["SW"] * 10)
    res = run(SA.S114(), portal(px, master=m, universe=names), "2023-02-27", "2023-03-08")
    o = orders(res)
    o = o[(pd.DatetimeIndex(o["submitted_at"]) == close_ts(shock)) & o["tag"].str.startswith("entry")]
    assert set(o.loc[o["quantity"] > 0, "symbol"]) == {"K0", "K1"}
    assert set(o.loc[o["quantity"] < 0, "symbol"]) == {"K8", "K9"}
    f = res.fills
    ts = set(pd.DatetimeIndex(f["ts"]))
    assert ts == {open_ts(nxt(shock)), close_ts(nxt(shock))}


# ------------------------------------------------------------------------------------- S123
def test_s123_hysteresis_and_equal_notional():
    dates = sess("2022-06-01", "2023-10-31")
    di = pd.DatetimeIndex(dates)
    spy = np.full(len(di), 0.0002)
    rsp = spy.copy()
    me = [pd.Timestamp(x) for x in ("2023-06-30", "2023-07-31", "2023-08-31")]
    targets = {me[0]: 0.025, me[1]: 0.010, me[2]: 0.003}   # enter; hold (above exit 0.5%); exit
    for d, ex in targets.items():
        i = di.get_loc(d)
        w0 = i - 63
        cur = np.prod(1 + rsp[w0 + 1:i + 1]) / np.prod(1 + spy[w0 + 1:i + 1]) - 1
        rsp[i] += (1 + ex) / (1 + cur) - 1 + 0  # approximate: set the window excess with one day's return
    px = {"RSP": path(di, rsp, 150.0), "SPY": path(di, spy, 400.0)}
    st = SA.S123()
    res = run(st, portal(px), "2023-06-26", "2023-09-08")
    states = [e[3].split(" ")[0] for e in st.events]
    assert states[:3] == ["0->1", "1->1", "1->0"], st.events
    f = res.fills
    ent = f[pd.DatetimeIndex(f["ts"]) == open_ts(nxt(me[0]))]
    assert set(ent["symbol"]) == {"RSP", "SPY"}
    assert np.allclose((ent["quantity"].abs() * ent["price"]).to_numpy(), 0.05 * NAV0, rtol=0.01)
