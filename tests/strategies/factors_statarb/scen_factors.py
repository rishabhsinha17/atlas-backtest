"""SYNTHETIC scenarios (mechanics only) for the cross-sectional factor strategies S085-S100.

Each builder returns a dict:
  cls, params, build(cut=None) -> portal, signal (session), start, end, longs, shorts,
  weights (optional {sym: signed weight}), near_miss (symbols that must not trade), ignore (overlay symbols)
Expected baskets are computed here by straightforward reference code on the synthetic inputs.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from atlasbt.core.types import ActionType, CorporateAction
from atlasbt.strategies import factors as F

from fsa_fixtures import (CAL, build_portal, close_ts, closes_from_returns, fund_row, master, sessions)

SIGNAL = pd.Timestamp("2023-06-30")
START, END = "2023-06-26", "2023-07-07"
STOCKS = [f"S{i:02d}" for i in range(20)]


def month_end(d):
    return F.month_end_sessions(CAL, SIGNAL, 13)[d]


def top_bottom(score: pd.Series, k_top: int, k_bot: int | None = None):
    s = score.dropna().sort_values(kind="mergesort")
    k_bot = k_top if k_bot is None else k_bot
    return list(s.index[-k_top:]), list(s.index[:k_bot])


def bucket_sets(score: pd.Series, q: int):
    """Documented quantile_bucket rule: average ranks, bucket = ceil(rank*q/n)."""
    s = score.dropna()
    r = s.rank(method="average")
    b = np.ceil(r * q / len(s) - 1e-9).clip(1, q)
    return list(s.index[b == q]), list(s.index[b == 1])


def shares_rows(syms, shares, period="2022-10-20", known="2022-10-28 21:00"):
    return [fund_row(s, "shares_outstanding", shares[s], period, known) for s in syms]


# --------------------------------------------------------------------------- momentum
def _mom_closes(seed=0, reversal=0.004):
    dates = sessions("2021-01-04", "2023-08-31")
    m12, m1, m0 = month_end(12), month_end(1), month_end(0)
    out = {}
    for i, s in enumerate(STOCKS):
        r = np.random.default_rng(seed + i).normal(0, 0.002, len(dates))
        di = pd.DatetimeIndex(dates)
        r = r + np.where((di > m12) & (di <= m1), 0.0004 * (i - 9.5), 0.0)
        r = r + np.where((di > m1) & (di <= m0), -reversal * (i - 9.5), 0.0)
        out[s] = closes_from_returns(dates, r, 40 + i)
    return out


def s085():
    c = _mom_closes()
    m = master(STOCKS)
    px = pd.DataFrame(c)
    score = px.loc[month_end(1)] / px.loc[month_end(12)] - 1
    longs, shorts = top_bottom(score, 2)
    noskip = px.loc[month_end(0)] / px.loc[month_end(12)] - 1
    assert set(top_bottom(noskip, 2)[0]) != set(longs)       # the skip month matters in this design
    s_sorted = score.sort_values()
    return dict(cls=F.S085, params={}, build=lambda cut=None: build_portal(c, master_df=m, cut=cut),
                longs=longs, shorts=shorts, near_miss=[s_sorted.index[-3], s_sorted.index[2]])


def s086():
    c = _mom_closes()
    m = master(STOCKS)
    px = pd.DataFrame(c)
    score = px.loc[month_end(0)] / px.loc[month_end(6)] - 1
    longs, shorts = top_bottom(score, 2)
    s_sorted = score.sort_values()
    return dict(cls=F.S086, params={}, build=lambda cut=None: build_portal(c, master_df=m, cut=cut),
                longs=longs, shorts=shorts, near_miss=[s_sorted.index[-3], s_sorted.index[2]])


def s087():
    dates = sessions("2021-01-04", "2023-08-31")
    di = pd.DatetimeIndex(dates)
    m7, m1, m0 = month_end(7), month_end(1), month_end(0)
    c, shares = {}, {}
    for i, s in enumerate(STOCKS):
        k = i % 10
        r = np.random.default_rng(100 + i).normal(0, 0.002, len(dates))
        r = r + np.where((di > m7) & (di <= m1), 0.0005 * (k - 4.5) + 0.0001 * (i // 10), 0.0)
        r = r + np.where((di > m1) & (di <= m0), -0.004 * (k - 4.5), 0.0)
        c[s] = closes_from_returns(dates, r, 30 + i)
        shares[s] = 1e8 * (1 + i)
    m = master(STOCKS)
    px = pd.DataFrame(c)
    ret = px.loc[m1] / px.loc[m7] - 1
    w = pd.Series(shares) * px.loc[m7]
    ind = pd.Series({s: f"SUB{i % 10}" for i, s in enumerate(STOCKS)})
    iret = (ret * w).groupby(ind).sum() / w.groupby(ind).sum()
    top_i, bot_i = top_bottom(iret, 2)
    longs = [s for s in STOCKS if ind[s] in top_i]
    shorts = [s for s in STOCKS if ind[s] in bot_i]
    third = iret.sort_values().index[-3]
    fund = pd.DataFrame(shares_rows(STOCKS, shares))
    return dict(cls=F.S087, params={}, build=lambda cut=None: build_portal(c, master_df=m, fundamentals=fund, cut=cut),
                longs=longs, shorts=shorts, near_miss=[s for s in STOCKS if ind[s] == third])


SECTOR_OF = {s: ["Information Technology", "Financials", "Health Care", "Energy", "Industrials"][i % 5]
             for i, s in enumerate(STOCKS)}
ETF_OF_SECTOR = {"Information Technology": "XLK", "Financials": "XLF", "Health Care": "XLV", "Energy": "XLE",
                 "Industrials": "XLI"}


def _factor_market(dates, seed=7):
    rng = np.random.default_rng(seed)
    spy = rng.normal(0.0002, 0.008, len(dates))
    etf = {e: rng.normal(0, 0.006, len(dates)) for e in ETF_OF_SECTOR.values()}
    return spy, etf


def s088():
    dates = sessions("2020-06-01", "2023-08-31")
    di = pd.DatetimeIndex(dates)
    spy, etf = _factor_market(dates)
    jump_day = pd.Timestamp("2023-01-17")
    jd = di == jump_day          # factor returns zero on the jump day: the jump is purely idiosyncratic
    spy[jd] = 0.0
    for e in etf:
        etf[e][jd] = 0.0
    c = {"SPY": closes_from_returns(dates, spy, 380)}
    for j, e in enumerate(F.SECTOR_SPDR_9):
        c[e] = closes_from_returns(dates, etf.get(e, np.random.default_rng(250 + j).normal(0, 0.006, len(dates))), 70)
    for i, s in enumerate(STOCKS):
        e = ETF_OF_SECTOR[SECTOR_OF[s]]
        r = 1.0 * spy + 0.5 * etf[e] + np.random.default_rng(200 + i).normal(0, 0.0005, len(dates))
        r = r + np.where(di == jump_day, 0.02 * (i - 9.5), 0.0)
        if s == "S05":       # huge idiosyncratic move inside the SKIP month: must not matter
            r = r + np.where(di == pd.Timestamp("2023-06-15"), 0.30, 0.0)
        c[s] = closes_from_returns(dates, r, 50 + i)
    m = master(STOCKS)
    return dict(cls=F.S088, params={}, build=lambda cut=None: build_portal(c, universe=STOCKS, master_df=m, cut=cut),
                longs=["S18", "S19"], shorts=["S00", "S01"], near_miss=["S05", "S17", "S02"], ignore={"SPY"})


# --------------------------------------------------------------------------- fundamentals
def _flat_closes(dates, seed, p0s):
    return {s: closes_from_returns(dates, np.random.default_rng(seed + i).normal(0, 0.001, len(dates)), p0s[i])
            for i, s in enumerate(STOCKS)}


def _size_groups(mcap: pd.Series):
    s = mcap.sort_values()
    return list(s.index[:len(s) // 2]), list(s.index[len(s) // 2:])


def s091():
    dates = sessions("2021-01-04", "2023-08-31")
    p0 = [50.0] * 20
    c = _flat_closes(dates, 300, p0)
    shares = {s: 1e8 * (1 + 0.1 * i) for i, s in enumerate(STOCKS)}
    px = pd.DataFrame(c)
    dec = pd.Timestamp("2022-12-30")
    me_dec = px.loc[dec] * pd.Series(shares)
    bm = pd.Series(np.random.default_rng(5).permutation(np.linspace(0.2, 2.0, 20)), index=STOCKS)
    rows = shares_rows(STOCKS, shares)
    for i, s in enumerate(STOCKS):
        be = bm[s] * me_dec[s]
        rows += [fund_row(s, "stockholders_equity", be, "2022-12-31", "2023-02-21 21:00"),
                 fund_row(s, "net_income_fy", 0.1 * be, "2022-12-31", "2023-02-21 21:00"),
                 # later 10-Q equity in reverse order: must NOT be used (FY book equity rule)
                 fund_row(s, "stockholders_equity", me_dec[s] * (2.2 - bm[s]), "2023-03-31", "2023-05-05 21:00")]
    # S20: 10-K accepted one second AFTER the signal close -> not usable
    c["S20"] = closes_from_returns(dates, np.zeros(len(dates)), 50.0)
    rows += [fund_row("S20", "stockholders_equity", 9e10, "2022-12-31", close_ts(SIGNAL) + pd.Timedelta(seconds=1)),
             fund_row("S20", "net_income_fy", 1e9, "2022-12-31", close_ts(SIGNAL) + pd.Timedelta(seconds=1)),
             fund_row("S20", "shares_outstanding", 1e8, "2022-10-20", "2022-10-28 21:00")]
    fund = pd.DataFrame(rows)
    uni = STOCKS + ["S20"]
    m = master(uni)
    size = px.loc[SIGNAL] * pd.Series(shares)
    small, big = _size_groups(size)
    longs = [bm[small].idxmax(), bm[big].idxmax()]
    shorts = [bm[small].idxmin(), bm[big].idxmin()]
    return dict(cls=F.S091, params={}, build=lambda cut=None: build_portal(c, universe=uni, master_df=m,
                                                                          fundamentals=fund, cut=cut),
                longs=longs, shorts=shorts, near_miss=["S20"])


def _earn_rows(s, field, fy21, q1_22, fy22, q1_23, q1_23_known="2023-05-05 21:00"):
    return [fund_row(s, f"{field}_fy", fy21, "2021-12-31", "2022-02-22 21:00"),
            fund_row(s, f"{field}_q", q1_22, "2022-03-31", "2022-05-05 21:00"),
            fund_row(s, f"{field}_fy", fy22, "2022-12-31", "2023-02-21 21:00"),
            fund_row(s, f"{field}_q", q1_23, "2023-03-31", q1_23_known)]


def s092(exclude_negative=True):
    dates = sessions("2021-01-04", "2023-08-31")
    p0 = [20.0 + 3 * i for i in range(20)]
    c = _flat_closes(dates, 400, p0)
    shares = {s: 1e8 for s in STOCKS}
    rng = np.random.default_rng(9)
    fy22 = pd.Series(rng.uniform(1e8, 1e9, 20), index=STOCKS)
    q1_22 = fy22 / 4
    q1_23 = pd.Series(rng.uniform(0.0, 6e8, 20), index=STOCKS)
    q1_23["S07"] = -2e9                            # FY positive, TTM negative
    rows = shares_rows(STOCKS, shares)
    for s in STOCKS:
        rows += _earn_rows(s, "net_income", fy22[s] * 0.9, q1_22[s], fy22[s], q1_23[s])
    ttm = fy22 + q1_23 - q1_22
    mcap = pd.DataFrame(c).loc[SIGNAL] * 1e8
    ey = ttm / mcap
    if exclude_negative:
        ey = ey[ttm > 0]
    longs, shorts = bucket_sets(ey, 5)
    fyrank = bucket_sets(fy22 / mcap, 5)
    assert set(fyrank[0]) != set(longs) or set(fyrank[1]) != set(shorts)    # TTM matters in this design
    fund = pd.DataFrame(rows)
    m = master(STOCKS)
    return dict(cls=F.S092, params={"exclude_negative": exclude_negative},
                build=lambda cut=None: build_portal(c, master_df=m, fundamentals=fund, cut=cut),
                longs=longs, shorts=shorts, near_miss=["S07"] if exclude_negative else [])


def s093(lag="exact"):
    dates = sessions("2021-01-04", "2023-08-31")
    p0 = [20.0 + 2 * i for i in range(20)]
    c = _flat_closes(dates, 500, p0)
    rng = np.random.default_rng(11)
    ocf = pd.Series(rng.uniform(2e8, 1e9, 20), index=STOCKS)
    q1_23 = pd.Series(rng.uniform(0, 5e8, 20), index=STOCKS)
    capex = pd.Series(rng.uniform(5e7, 2e8, 20), index=STOCKS)
    debt = pd.Series(rng.uniform(0, 2e9, 20), index=STOCKS)
    cash = pd.Series(rng.uniform(0, 5e8, 20), index=STOCKS)
    cash["S03"] = 1e11                              # EV <= 0 -> excluded although best FCF
    ocf["S03"] = 5e9
    late = "S05"                                     # its Q1-23 filing is only 15 days old at the signal
    q1_23[late] = 3e9
    ocf[late] = 2.1e8
    rows = shares_rows(STOCKS, {s: 1e8 for s in STOCKS})
    for s in STOCKS:
        kn = "2023-06-15 21:00" if s == late else "2023-05-05 21:00"
        rows += _earn_rows(s, "operating_cash_flow", ocf[s], ocf[s] / 4, ocf[s], q1_23[s], kn)
        rows += _earn_rows(s, "capex", capex[s], capex[s] / 4, capex[s], capex[s] / 4, kn)
        rows += [fund_row(s, "long_term_debt", debt[s], "2022-12-31", "2023-02-21 21:00"),
                 fund_row(s, "cash", cash[s], "2022-12-31", "2023-02-21 21:00")]
    ttm_ocf = ocf.copy()
    if lag == "exact":
        ttm_ocf = ocf + q1_23 - ocf / 4
    else:
        ttm_ocf[[s for s in STOCKS if s != late]] = (ocf + q1_23 - ocf / 4)[[s for s in STOCKS if s != late]]
    fcf = ttm_ocf - capex
    mcap = pd.DataFrame(c).loc[SIGNAL] * 1e8
    ev = mcap + debt - cash
    y = (fcf / ev)[ev > 0]
    longs, shorts = bucket_sets(y, 5)
    fund = pd.DataFrame(rows)
    m = master(STOCKS)
    return dict(cls=F.S093, params={"lag": lag}, build=lambda cut=None: build_portal(c, master_df=m, fundamentals=fund,
                                                                                    cut=cut),
                longs=longs, shorts=shorts, near_miss=["S03"])


def s094():
    dates = sessions("2021-01-04", "2023-08-31")
    c = _flat_closes(dates, 600, [50.0] * 20)
    shares = {s: 1e8 * (1 + 0.07 * i) for i, s in enumerate(STOCKS)}
    rng = np.random.default_rng(13)
    gpa = pd.Series(rng.permutation(np.linspace(0.05, 0.6, 20)), index=STOCKS)
    rows = shares_rows(STOCKS, shares)
    for s in STOCKS:
        ta = 1e10
        rows += [fund_row(s, "total_assets", ta, "2022-12-31", "2023-02-21 21:00"),
                 fund_row(s, "gross_profit_fy", gpa[s] * ta, "2022-12-31", "2023-02-21 21:00"),
                 fund_row(s, "total_assets", ta * (1 + 5 * gpa[s]), "2023-03-31", "2023-05-05 21:00")]
    size = pd.DataFrame(c).loc[SIGNAL] * pd.Series(shares)
    small, big = _size_groups(size)
    longs = list(gpa[small].nlargest(2).index) + list(gpa[big].nlargest(2).index)
    shorts = list(gpa[small].nsmallest(2).index) + list(gpa[big].nsmallest(2).index)
    fund = pd.DataFrame(rows)
    m = master(STOCKS)
    return dict(cls=F.S094, params={}, build=lambda cut=None: build_portal(c, master_df=m, fundamentals=fund, cut=cut),
                longs=longs, shorts=shorts, near_miss=[])


SECT2 = ["Information Technology"] * 10 + ["Health Care"] * 10


def s095():
    dates = sessions("2021-01-04", "2023-08-31")
    c = _flat_closes(dates, 700, [40.0 + i for i in range(20)])
    rows = []
    for i, s in enumerate(STOCKS):
        q = i % 10
        g, roe, gpa, lev = 0.02 + 0.01 * q, 0.05 + 0.01 * q, 0.10 + 0.02 * q, 0.5 - 0.04 * q
        ta = 1e10
        for y in range(2015, 2023):
            se = 4e9 * (1.03 ** (y - 2015))
            pe, kn = f"{y}-12-31", f"{y + 1}-02-21 21:00"
            amp = 0.1 * (9 - q) / 9
            ni = se * roe * (1 + amp * (-1) ** y)
            rows += [fund_row(s, "revenue_fy", 1e9 * (1 + g) ** (y - 2015), pe, kn),
                     fund_row(s, "net_income_fy", ni, pe, kn),
                     fund_row(s, "stockholders_equity", se, pe, kn),
                     fund_row(s, "gross_profit_fy", gpa * ta, pe, kn),
                     fund_row(s, "total_assets", ta, pe, kn),
                     fund_row(s, "long_term_debt", lev * ta, pe, kn)]
            rows += [fund_row(s, "net_income_q", ni / 4, f"{y}-03-31", f"{y}-05-05 21:00"),
                     fund_row(s, "gross_profit_q", gpa * ta / 4, f"{y}-03-31", f"{y}-05-05 21:00")]
        rows += [fund_row(s, "net_income_q", se * roe * 1.1 / 4, "2023-03-31", "2023-05-05 21:00"),
                 fund_row(s, "gross_profit_q", gpa * ta / 4, "2023-03-31", "2023-05-05 21:00")]
    fund = pd.DataFrame(rows)
    m = master(STOCKS, sectors=SECT2)
    return dict(cls=F.S095, params={}, build=lambda cut=None: build_portal(c, master_df=m, fundamentals=fund, cut=cut),
                longs=["S09", "S19"], shorts=["S00", "S10"], near_miss=["S08", "S18", "S01", "S11"])


def s096():
    dates = sessions("2021-06-01", "2023-08-31")
    c = {s: closes_from_returns(dates, np.random.default_rng(800 + i).normal(0, 0.004 + 0.0007 * i, len(dates)),
                                60.0) for i, s in enumerate(STOCKS)}
    shares = {s: 1e8 * (1 + 0.05 * ((i * 7) % 20)) for i, s in enumerate(STOCKS)}
    px = pd.DataFrame(c)
    vol = (px / px.shift(1) - 1).loc[:SIGNAL].iloc[-252:].std(ddof=1)
    size = px.loc[SIGNAL] * pd.Series(shares)
    small, big = _size_groups(size)
    longs = [vol[small].idxmin(), vol[big].idxmin()]
    shorts = [vol[small].idxmax(), vol[big].idxmax()]
    fund = pd.DataFrame(shares_rows(STOCKS, shares))
    m = master(STOCKS)
    return dict(cls=F.S096, params={}, build=lambda cut=None: build_portal(c, master_df=m, fundamentals=fund, cut=cut),
                longs=longs, shorts=shorts, near_miss=[])


def s097():
    dates = sessions("2021-06-01", "2023-08-31")
    spy = np.random.default_rng(900).normal(0.0003, 0.01, len(dates))
    c = {"SPY": closes_from_returns(dates, spy, 400.0)}
    betas = {s: 0.4 + 0.08 * i for i, s in enumerate(STOCKS)}
    for i, s in enumerate(STOCKS):
        c[s] = closes_from_returns(dates, betas[s] * spy + np.random.default_rng(910 + i).normal(0, 0.004, len(dates)),
                                   30.0 + i)
    px = pd.DataFrame(c)
    r = (px / px.shift(1) - 1).loc[:SIGNAL].iloc[-252:]
    b = pd.Series({s: np.polyfit(r["SPY"], r[s], 1)[0] for s in STOCKS})
    longs, shorts = list(b.nsmallest(4).index), list(b.nlargest(4).index)
    bl, bs = 0.02 * b[longs].sum(), 0.02 * b[shorts].sum()
    w = {s: 0.02 for s in longs}
    w.update({s: -0.02 * bl / bs for s in shorts})
    m = master(STOCKS)
    return dict(cls=F.S097, params={}, build=lambda cut=None: build_portal(c, universe=STOCKS, master_df=m, cut=cut),
                longs=longs, shorts=shorts, weights=w, near_miss=[])


def s098():
    dates = sessions("2021-01-04", "2023-08-31")
    c = _flat_closes(dates, 1000, [30.0 + i for i in range(20)])
    split_day = pd.Timestamp("2023-03-01")
    c["S04"] = c["S04"].where(c["S04"].index < split_day, c["S04"] / 2.0)
    actions = [CorporateAction("S04", ActionType.SPLIT, split_day, pd.Timestamp("2023-02-01 21:00", tz="UTC"), ratio=2.0)]
    rng = np.random.default_rng(15)
    div = pd.Series(rng.uniform(0, 1.2e8, 20), index=STOCKS)
    sh_old = pd.Series(1e8, index=STOCKS)
    chg = pd.Series(rng.uniform(-0.08, 0.06, 20), index=STOCKS)
    chg["S04"] = -0.10                                       # best real buyback, but 2:1 split doubles the count
    sh_new = sh_old * (1 + chg)
    sh_new["S04"] = sh_old["S04"] * (1 + chg["S04"]) * 2.0
    rows = []
    for s in STOCKS:
        rows += [fund_row(s, "shares_outstanding", sh_old[s], "2022-04-20", "2022-05-05 21:00"),
                 fund_row(s, "shares_outstanding", sh_new[s], "2023-04-20", "2023-05-05 21:00")]
        rows += _earn_rows(s, "dividends_paid", div[s], div[s] / 4, div[s], div[s] / 4)
    mcap = pd.DataFrame(c).loc[SIGNAL] * sh_new
    split = pd.Series(1.0, index=STOCKS)
    split["S04"] = 2.0
    y = div / mcap - (sh_new / (sh_old * split) - 1)
    longs, shorts = bucket_sets(y, 5)
    assert "S04" in longs
    fund = pd.DataFrame(rows)
    m = master(STOCKS)
    return dict(cls=F.S098, params={}, build=lambda cut=None: build_portal(c, master_df=m, fundamentals=fund,
                                                                          actions=actions, cut=cut),
                longs=longs, shorts=shorts, near_miss=[])


def s099():
    dates = sessions("2021-01-04", "2023-08-31")
    c = _flat_closes(dates, 1100, [30.0 + i for i in range(20)])
    rng = np.random.default_rng(17)
    g = pd.Series(rng.permutation(np.linspace(-0.1, 0.5, 20)), index=STOCKS)
    rows = []
    for s in STOCKS:
        a21 = 1e10
        rows += [fund_row(s, "total_assets", a21, "2021-12-31", "2022-02-22 21:00"),
                 fund_row(s, "revenue_fy", 1e9, "2021-12-31", "2022-02-22 21:00"),
                 fund_row(s, "total_assets", a21 * (1 + g[s]), "2022-12-31", "2023-02-21 21:00"),
                 fund_row(s, "revenue_fy", 1e9, "2022-12-31", "2023-02-21 21:00"),
                 fund_row(s, "total_assets", a21 * (1.4 - g[s]), "2023-03-31", "2023-05-05 21:00")]
    fund = pd.DataFrame(rows)
    shorts, longs = bucket_sets(g, 5)
    m = master(STOCKS)
    return dict(cls=F.S099, params={}, build=lambda cut=None: build_portal(c, master_df=m, fundamentals=fund, cut=cut),
                longs=longs, shorts=shorts, near_miss=[])


def s100():
    dates = sessions("2021-01-04", "2023-08-31")
    di = pd.DatetimeIndex(dates)
    m12, m1 = month_end(12), month_end(1)
    c = {}
    for i, s in enumerate(STOCKS):
        r = np.random.default_rng(1200 + i).normal(0, 0.001, len(dates))
        r = r + np.where((di > m12) & (di <= m1), 0.0002 * (i % 10) + 0.00003 * i, 0.0)
        c[s] = closes_from_returns(dates, r, 50.0)
    rows = shares_rows(STOCKS, {s: 1e8 for s in STOCKS})
    for i, s in enumerate(STOCKS):
        q = i % 10
        se, ta = 1e9 * (1 + 0.1 * q + 0.01 * i), 1e10
        ni = ta * (0.02 + 0.004 * q + 0.0003 * i)
        rows += [fund_row(s, "stockholders_equity", se, "2022-12-31", "2023-02-21 21:00"),
                 fund_row(s, "total_assets", ta, "2022-12-31", "2023-02-21 21:00")]
        rows += _earn_rows(s, "net_income", ni, ni / 4, ni, ni / 4)
    px = pd.DataFrame(c)
    mcap = px.loc[SIGNAL] * 1e8
    bm = pd.Series({s: (1e9 * (1 + 0.1 * (i % 10) + 0.01 * i)) for i, s in enumerate(STOCKS)}) / mcap
    roa = pd.Series({s: 0.02 + 0.004 * (i % 10) + 0.0003 * i for i, s in enumerate(STOCKS)})
    mom = px.loc[m1] / px.loc[m12] - 1
    sec = pd.Series(dict(zip(STOCKS, SECT2)))

    def gz(x):
        return x.groupby(sec).transform(lambda v: (v - v.mean()) / v.std(ddof=1))

    comp = (gz(bm) + gz(roa) + gz(mom)) / 3
    longs, shorts = bucket_sets(comp, 10)
    fund = pd.DataFrame(rows)
    m = master(STOCKS, sectors=SECT2)
    return dict(cls=F.S100, params={}, build=lambda cut=None: build_portal(c, master_df=m, fundamentals=fund, cut=cut),
                longs=longs, shorts=shorts, near_miss=[])


SCENARIOS = {"S085": s085, "S086": s086, "S087": s087, "S088": s088, "S091": s091, "S092": s092, "S093": s093,
             "S094": s094, "S095": s095, "S096": s096, "S097": s097, "S098": s098, "S099": s099, "S100": s100}
