"""Synthetic tests for S054 / S074 / S082 (CROSS_SECTION), S055 (all-time-high, SP500 universe)
and S083 (VIX shock, Cboe VIX known at 16:15 ET).

SYNTHETIC data, mechanics only.
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from swing_fixtures import (INDEX_INSTRUMENTS, CAL, SectorPortal, arrays, cent_rows, close_ts, entries,  # noqa: E402
                            fills_of, open_ts, orders, portal_for, rows_from_closes, run, sessions, stop_in_force,
                            to_df, trend_closes, with_universe, zero_cost)

from atlasbt.core.types import NotTestableError, RunStatus  # noqa: E402
from atlasbt.engine.backtester import run_backtest  # noqa: E402
from atlasbt.engine.strategy import Requirement  # noqa: E402
from atlasbt.indicators import atr, percentile_rank, rsi, sma  # noqa: E402
from atlasbt.strategies import swing as sw  # noqa: E402

N = 300


# ===================================================================== S055
def _ath_rows(vol=1_000_000.0):
    c = list(trend_closes(N, 30.0, 0.002, amp=0.004))
    for r in [-0.003] * 20 + [0.006] * 15:
        c.append(c[-1] * (1 + r))
    return cent_rows(rows_from_closes(c, rng=0.004, vol=vol))


def _first_ath(h, c, start):
    for t in range(start, len(c)):
        if c[t] > np.max(h[:t]):
            return t
    return None


def _run_one(cls, rows, s, params=None, run_from=2, extra_after=4):
    ss = sessions(len(rows))
    T = with_universe(cls, ["XYZ"])
    res = run(T, portal_for({"XYZ": to_df(rows)}), ss[s - run_from], ss[min(s + extra_after, len(rows) - 1)], params)
    return res, ss


def test_synthetic_s055_new_all_time_high_entry_and_3atr_stop():
    rows = _ath_rows(vol=1_000_000.0)                      # ~ $60m daily dollar volume
    o, h, l, c, v = arrays(rows)
    s = _first_ath(h, c, N + 1)
    assert s is not None and s > N + 20
    res, ss = _run_one(sw.S055, rows, s, run_from=15)       # flat-to-down days before: ATH tracked incrementally
    e = entries(res).sort_values("order_id").iloc[0]
    assert e["submitted_at"] == close_ts(ss[s])
    f = fills_of(res, e["order_id"])
    assert f["ts"].iloc[0] == open_ts(ss[s + 1]) and f["price"].iloc[0] == pytest.approx(rows[s + 1][0])
    a = np.asarray(atr(h[: s + 1], l[: s + 1], c[: s + 1], 14))[-1]
    assert stop_in_force(res, close_ts(ss[s + 1])) == pytest.approx(f["price"].iloc[0] - 3 * a, abs=0.011)


def test_synthetic_s055_near_miss_dollar_volume_below_20m():
    rows = _ath_rows(vol=300_000.0)                         # ~ $18m
    o, h, l, c, v = arrays(rows)
    s = _first_ath(h, c, N + 1)
    assert np.mean(c[s - 19:s + 1] * v[s - 19:s + 1]) < 20e6
    res, ss = _run_one(sw.S055, rows, s, run_from=2, extra_after=0)
    assert len(entries(res)) == 0


def test_synthetic_s055_near_miss_history_shorter_than_min():
    rows = _ath_rows()[N - 200:]                            # only ~235 bars of history
    o, h, l, c, v = arrays(rows)
    s = _first_ath(h, c, 201)
    res, ss = _run_one(sw.S055, rows, s, run_from=1, extra_after=0)
    assert len(entries(res)) == 0
    res2, _ = _run_one(sw.S055, rows, s, run_from=1, extra_after=1, params={"min_history": 126})
    assert len(entries(res2)) == 1


# ===================================================================== cross-section helpers
STK = [f"K{i:02d}" for i in range(20)]
SECTORS = {s: ("Information Technology" if i % 2 == 0 else "Financials") for i, s in enumerate(STK)}


def _xs_portal(dfs, sectors=SECTORS, **kw):
    bars = {(s, "1d"): df for s, df in dfs.items()}
    return SectorPortal(bars=bars, sectors=sectors, universes={"SP500": list(STK)}, **kw)


def _xs_class(cls, **attrs):
    d = {"requires": [Requirement("bars", "1d", symbols=("K00",))]}
    d.update(attrs)
    return type(f"{cls.__name__}_X", (cls,), d)


def _month_end_index(ss, after):
    for i in range(after, len(ss)):
        if CAL.is_last_session_of_month(ss[i]):
            return i
    raise AssertionError


# ===================================================================== S054
def test_synthetic_s054_month_end_52w_high_ranking_and_next_open_rebalance():
    n = N + 60
    ss = sessions(n)
    me = _month_end_index(ss, N + 5)
    dfs, ratio = {}, {}
    for i, s in enumerate(STK):
        c = list(trend_closes(me - 19, 50.0 + i, 0.002, amp=0.003))
        dd = 0.01 * (i + 1)                                   # K00 nearest its high ... K19 furthest
        for k in range(1, 20):
            c.append(c[me - 19 - 1] * (1 - dd * k / 19))
        c += [c[-1]] * (n - len(c))
        rows = cent_rows(rows_from_closes(c, rng=0.002))
        dfs[s] = to_df(rows)
        o, h, l, cc, v = arrays(rows)
        ratio[s] = cc[me] / np.max(h[me - 251:me + 1])
    order = sorted(STK, key=lambda s: ratio[s])
    res = run(_xs_class(sw.S054), _xs_portal(dfs), ss[me - 3], ss[me + 2])
    o = orders(res)
    assert len(o) and set(o["submitted_at"]) == {close_ts(ss[me])}        # only at the month-end close
    longs = set(o[o["quantity"] > 0]["symbol"])
    shorts = set(o[o["quantity"] < 0]["symbol"])
    assert longs == set(order[-2:]) and shorts == set(order[:2])          # top / bottom decile of 20 names
    f = res.fills
    assert set(f["ts"]) == {open_ts(ss[me + 1])}
    nav = 1_000_000.0
    for _, r in o.iterrows():                                              # 2% single-name cap binds (25% > 2%)
        px = dfs[r["symbol"]]["close"].iloc[me]
        assert abs(r["quantity"]) == math.trunc(0.02 * nav / px + 1e-9)


def test_synthetic_s054_lag5_executes_five_sessions_later():
    n = N + 60
    ss = sessions(n)
    me = _month_end_index(ss, N + 5)
    dfs = {s: to_df(cent_rows(rows_from_closes(trend_closes(n, 50.0 + i, 0.001 * (i - 10), amp=0.01),
                                               rng=0.003))) for i, s in enumerate(STK)}
    res = run(_xs_class(sw.S054), _xs_portal(dfs), ss[me - 2], ss[me + 6], {"lag": 5})
    o = orders(res)
    assert len(o) and set(o["submitted_at"]) == {close_ts(ss[me + 4])}
    assert set(res.fills["ts"]) == {open_ts(ss[me + 5])}


# ===================================================================== S074
def _weekly_fixture(n=N + 40):
    ss = sessions(n)
    fri = next(i for i in range(N + 5, n) if CAL.next_session(ss[i]).isocalendar()[1] != ss[i].isocalendar()[1])
    dfs, r5 = {}, {}
    for i, s in enumerate(STK):
        c = list(trend_closes(fri - 4, 40.0 + i, 0.0005, amp=0.004))
        step = (i - 9.5) * 0.002                               # 5-day return rises with i
        for k in range(5):
            c.append(c[-1] * (1 + step))
        c += [c[-1]] * (n - len(c))
        rows = cent_rows(rows_from_closes(c, rng=0.003))
        dfs[s] = to_df(rows)
        cc = arrays(rows)[3]
        r5[s] = cc[fri] / cc[fri - 5] - 1
    return ss, fri, dfs, r5


def test_synthetic_s074_weekly_sector_neutral_reversal():
    ss, fri, dfs, r5 = _weekly_fixture()
    res = run(_xs_class(sw.S074), _xs_portal(dfs), ss[fri - 3], ss[fri + 1])
    o = orders(res)
    assert set(o["submitted_at"]) == {close_ts(ss[fri])}                  # last session of the week only
    assert set(res.fills["ts"]) == {open_ts(ss[fri + 1])}                 # next session (Monday) open
    exp_long, exp_short = set(), set()
    for sec in set(SECTORS.values()):
        names = sorted((s for s in STK if SECTORS[s] == sec), key=lambda s: r5[s])
        exp_long.add(names[0])                                             # bottom decile of 10 names = 1 name
        exp_short.add(names[-1])
    assert set(o[o["quantity"] > 0]["symbol"]) == exp_long
    assert set(o[o["quantity"] < 0]["symbol"]) == exp_short


def test_synthetic_s074_not_sector_neutral_uses_global_deciles():
    ss, fri, dfs, r5 = _weekly_fixture()
    res = run(_xs_class(sw.S074), _xs_portal(dfs), ss[fri - 3], ss[fri + 1], {"sector_neutral": False})
    o = orders(res)
    srt = sorted(STK, key=lambda s: r5[s])
    assert set(o[o["quantity"] > 0]["symbol"]) == set(srt[:2])
    assert set(o[o["quantity"] < 0]["symbol"]) == set(srt[-2:])


def test_synthetic_s074_sector_neutral_without_sector_data_is_not_faked():
    ss, fri, dfs, r5 = _weekly_fixture()
    portal = portal_for(dfs, universes={"SP500": list(STK)})               # security master without sectors
    res = run_backtest(_xs_class(sw.S074)(), portal, ss[fri - 1], ss[fri + 1], zero_cost(), raise_errors=False)
    assert res.status == RunStatus.NOT_TESTABLE and "sector" in res.not_testable_reason
    # sectors only feed the caps: still NOT_TESTABLE by default, opt-in no_sector_caps -> runs, flagged T2
    res2 = run_backtest(_xs_class(sw.S074)(sector_neutral=False), portal, ss[fri - 1], ss[fri + 1], zero_cost(),
                        raise_errors=False)
    assert res2.status == RunStatus.NOT_TESTABLE and "sector" in res2.not_testable_reason
    res3 = run_backtest(_xs_class(sw.S074)(sector_neutral=False, no_sector_caps=True), portal, ss[fri - 1],
                        ss[fri + 1], zero_cost(), raise_errors=True)
    assert res3.status == RunStatus.OK and len(res3.fills) > 0
    assert any("sector caps not applied" in r["reason"] for r in res3.fidelity["reasons"])


# ===================================================================== S082
def test_synthetic_s082_residual_dip_long_and_rip_short():
    n = N + 40
    ss = sessions(n)
    rb = next(i for i in range(N, n) if CAL.session_index(ss[i]) % 5 == 0)
    rng = np.random.default_rng(3)
    rm = rng.normal(0.0004, 0.01, n)
    rk = rng.normal(0.0, 0.008, n)
    rf = rng.normal(0.0, 0.008, n)
    to_px = lambda r, p0: p0 * np.cumprod(1 + r)
    dfs = {"SPY": to_df(cent_rows(rows_from_closes(to_px(rm, 300.0)))),
           "XLK": to_df(cent_rows(rows_from_closes(to_px(rm + rk, 80.0)))),
           "XLF": to_df(cent_rows(rows_from_closes(to_px(rm + rf, 30.0))))}
    shocks = {"K03": -0.07, "K08": -0.06, "K05": 0.07, "K12": 0.06}
    for i, s in enumerate(STK):
        sec = rk if SECTORS[s] == "Information Technology" else rf
        r = 0.9 * rm + 0.8 * sec + rng.normal(0, 0.004, n)
        if s in shocks:
            r[rb - 4: rb + 1] += shocks[s] / 5
        dfs[s] = to_df(cent_rows(rows_from_closes(to_px(r, 50.0 + i))))
    T = _xs_class(sw.S082, requires=[Requirement("bars", "1d", symbols=("SPY",))])
    res = run(T, _xs_portal(dfs), ss[rb - 2], ss[rb + 1])
    o = orders(res)
    assert set(o["submitted_at"]) == {close_ts(ss[rb])}
    assert set(o[o["quantity"] > 0]["symbol"]) == {"K03", "K08"}
    assert set(o[o["quantity"] < 0]["symbol"]) == {"K05", "K12"}
    assert set(res.fills["ts"]) == {open_ts(ss[rb + 1])}


# ===================================================================== S083
def _vix_fixture(spike=True):
    n = N + 20
    ss = sessions(n)
    c = list(trend_closes(N, 300.0, 0.0005, amp=0.004))
    c += [c[-1] * 0.98, c[-1] * 0.98 * 0.98] + [c[-1] * 0.97] * 3 + [c[-1] * 1.0] * (n - N - 5)
    spy = cent_rows(rows_from_closes(c, rng=0.003))
    i = np.arange(n)
    lvl = 14 + 2 * np.sin(i / 5.0)
    if spike:
        lvl[N + 1:N + 4] = 35.0
    else:
        lvl[:] = 14 + 2 * np.sin(i / 5.0) + np.where(i > N - 260, 22.0, 0.0)   # high for the whole year: no shock
    vix = to_df([(x, x * 1.02, x * 0.98, x, float("nan")) for x in lvl])
    vix["known_at"] = vix["known_at"] + pd.Timedelta(minutes=15)           # Cboe official close at 16:15 ET
    return ss, spy, vix


def test_synthetic_s083_vix_shock_decided_after_1615_executes_next_open():
    ss, spy, vix = _vix_fixture()
    o, h, l, c, v = arrays(spy)
    s = N + 1                                                              # SPY RSI2 < 5 and VIX spike close
    assert np.asarray(rsi(c, 2))[s] < 5 and c[s] > 0
    assert np.asarray(percentile_rank(vix["close"].to_numpy(), 252, include_current=False))[s] > 0.9
    portal = portal_for({"SPY": to_df(spy), "VIX": vix}, instruments=INDEX_INSTRUMENTS)
    res = run(sw.S083, portal, ss[s - 2], ss[s + 6])
    e = entries(res)
    assert len(e) == 1
    assert e["submitted_at"].iloc[0] == open_ts(ss[s + 1])               # VIX close known 16:15 -> pre-open decision
    f = fills_of(res, e["order_id"].iloc[0])
    assert f["ts"].iloc[0] == open_ts(ss[s + 1]) and f["price"].iloc[0] == pytest.approx(spy[s + 1][0])
    assert e["quantity"].iloc[0] == math.trunc(0.10 * 1_000_000.0 / c[s])
    # exit: first SPY close > SMA5 or VIX < its 20d median, executed at the following open
    s5 = np.asarray(sma(c, 5))
    vv = vix["close"].to_numpy()
    x = next(t for t in range(s + 1, len(c)) if c[t] > s5[t] or vv[t] < np.median(vv[t - 19:t + 1]))
    ex = orders(res)
    ex = ex[ex["tag"].isin(["spy_above_sma5", "vix_below_median", "time_exit"])]
    assert ex["submitted_at"].iloc[0] == open_ts(ss[x + 1])


def test_synthetic_s083_near_miss_vix_not_in_trailing_top_decile():
    ss, spy, vix = _vix_fixture(spike=False)
    s = N + 1
    assert np.asarray(percentile_rank(vix["close"].to_numpy(), 252, include_current=False))[s] <= 0.9
    portal = portal_for({"SPY": to_df(spy), "VIX": vix}, instruments=INDEX_INSTRUMENTS)
    res = run(sw.S083, portal, ss[s - 2], ss[s + 4])
    assert len(entries(res)) == 0


def test_s083_vix_requirement_makes_run_not_testable():
    ss, spy, vix = _vix_fixture()
    portal = portal_for({"SPY": to_df(spy)})
    res = run_backtest(sw.S083(), portal, ss[N - 2], ss[N + 2], zero_cost())
    assert res.status == RunStatus.NOT_TESTABLE and "VIX" in res.not_testable_reason
