"""Synthetic-pattern trigger / near-miss tests for the swing mean-reversion strategies
S068-S084 (single-name ones; S074/S082 cross-section and S083 VIX are in
test_swing_xs_vix.py).

SYNTHETIC data, mechanics only. Every case is built twice from the same dip
pattern: on a rising base (filters pass -> exact entry bar/fill/stop asserted)
and on a falling base or with a minimally changed input (must NOT trigger).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from swing_fixtures import (arrays, cent_rows, close_ts, entries, fills_of, open_ts, orders, portal_for,  # noqa: E402
                            rows_from_closes, run, sessions, stop_in_force, to_df, trend_closes, with_universe)

from atlasbt.indicators import (atr, bollinger, cci, donchian, down_streak, ema, ibs, rsi, sma, stochastic,  # noqa: E402
                                williams_r, zscore)
from atlasbt.strategies import swing as sw  # noqa: E402

SYM = "XYZ"
N_BASE = 300


def _rows(g, dip, up, n=N_BASE, p0=40.0, rng=0.004, vol=1_000_000.0, extra=None):
    """Base trend of n bars, then `dip` = list of daily returns, then `up` returns, then 5 flat-ish bars."""
    c = list(trend_closes(n, p0, g, amp=0.002))
    for r in list(dip) + list(up) + [0.0005, -0.0005] * 4:
        c.append(c[-1] * (1 + r))
    rows = rows_from_closes(c, rng=rng, vol=vol)
    if extra:
        extra(rows)
    return cent_rows(rows)


def _run(cls, rows, s, params=None, run_from=2, extra_after=6, **attrs):
    ss = sessions(len(rows))
    portal = portal_for({SYM: to_df(rows)})
    T = with_universe(cls, [SYM], **attrs)
    res = run(T, portal, ss[s - run_from], ss[min(s + extra_after, len(rows) - 1)], params)
    return res, ss


def _first(mask, start=N_BASE - 5):
    idx = np.flatnonzero(np.asarray(mask, bool)[start:])
    return int(start + idx[0]) if idx.size else None


def _cross_up(x, level):
    x = np.asarray(x, float)
    out = np.zeros(len(x), bool)
    out[1:] = (x[:-1] <= level) & (x[1:] > level)
    return out


def _check_entry(res, ss, rows, s):
    e = entries(res).sort_values("order_id")
    assert len(e), "expected an entry"
    e = e.iloc[0]
    assert e["submitted_at"] == close_ts(ss[s])
    f = fills_of(res, e["order_id"])
    assert f["ts"].iloc[0] == open_ts(ss[s + 1]) and f["price"].iloc[0] == pytest.approx(rows[s + 1][0])
    return float(f["price"].iloc[0])


def _sma_up(c, n=200, lag=20):
    s = np.asarray(sma(c, n))
    out = np.zeros(len(c), bool)
    out[lag:] = s[lag:] > s[:-lag]
    return out


# ------------------------------------------------------------------ conditions (independent restatements)
def c068(o, h, l, c, v):
    return (c > np.asarray(sma(c, 200))) & (np.asarray(rsi(c, 2)) < 5)


def c069(o, h, l, c, v):
    return (c > np.asarray(sma(c, 200))) & _cross_up(rsi(c, 14), 30)


def c070(o, h, l, c, v):
    bb = bollinger(c, 20, 2.0)
    lo, up = np.asarray(bb["lower"]), np.asarray(bb["upper"])
    prev_below = np.zeros(len(c), bool)
    prev_below[1:] = c[:-1] < lo[:-1]
    return prev_below & (c >= lo) & (c <= up) & _sma_up(c)


def c071(o, h, l, c, v):
    return (c < np.asarray(bollinger(c, 20, 2.0)["lower"])) & (c > np.asarray(sma(c, 200)))


def c072(o, h, l, c, v):
    return (np.asarray(zscore(c, 20)) < -2) & _sma_up(c)


def c073(o, h, l, c, v):
    return (np.asarray(down_streak(c)) >= 3) & (c > np.asarray(sma(c, 200)))


def c075(o, h, l, c, v):
    return (np.nan_to_num(np.asarray(ibs(h, l, c)), nan=1.0) < 0.2) & (c > np.asarray(sma(c, 200)))


def c076(o, h, l, c, v):
    wr = np.asarray(williams_r(h, l, c, 14))
    s200 = np.asarray(sma(c, 200))
    out = np.zeros(len(c), bool)
    for t in range(2, len(c)):
        if wr[t - 1] <= -80 < wr[t] and c[t] > s200[t]:
            j = t - 1
            while j >= 0 and wr[j] <= -80:
                if wr[j] <= -90 and c[j] > s200[j]:
                    out[t] = True
                    break
                j -= 1
    return out


def c077(o, h, l, c, v):
    st = stochastic(h, l, c, 14, 3, 3)
    k, d = np.asarray(st["slow_k"]), np.asarray(st["slow_d"])
    x = np.zeros(len(c), bool)
    x[1:] = (k[:-1] <= d[:-1]) & (k[1:] > d[1:])
    return x & (k < 20) & (d < 20) & (c > np.asarray(sma(c, 200)))


def c078(o, h, l, c, v):
    return _cross_up(cci(h, l, c, 20), -100) & (c > np.asarray(sma(c, 200)))


def c079(o, h, l, c, v):
    a = np.asarray(atr(h, l, c, 14))
    out = np.zeros(len(c), bool)
    for t in range(21, len(c)):
        out[t] = c[t] <= c[t - 1] - 2 * a[t - 1] and v[t] >= 1.5 * np.mean(v[t - 20:t])
    return out & _sma_up(c)


def c080(o, h, l, c, v):
    s = np.asarray(sma(c, 200))
    return _sma_up(c) & (c >= s) & (c <= 1.01 * s) & (np.asarray(rsi(c, 14)) < 40)


def c081(o, h, l, c, v):
    e, a, e200 = np.asarray(ema(c, 20)), np.asarray(atr(h, l, c, 20)), np.asarray(ema(c, 200))
    up = np.zeros(len(c), bool)
    up[20:] = e200[20:] > e200[:-20]
    return (c < e - 2 * a) & (c > e - 3 * a) & up


def c084(o, h, l, c, v):
    ll = np.asarray(donchian(h, l, 252)["lower"])
    near = np.abs(c / ll - 1) <= 0.03
    dv = np.full(len(c), np.nan)
    for t in range(19, len(c)):
        dv[t] = np.mean(c[t - 19:t + 1] * v[t - 19:t + 1])
    return near & _cross_up(rsi(c, 14), 30) & (dv >= 20e6)


D3 = [-0.015] * 3
CASES = {
    #        cls, g_up, dip, up, cond, extra kwargs for _rows
    "S068": (sw.S068, 0.002, D3, [0.015] * 4, c068, {}),
    "S069": (sw.S069, 0.004, [-0.03] * 7, [0.025] * 6, c069, {}),
    "S070": (sw.S070, 0.002, [-0.025] * 4, [0.02, 0.02], c070, {}),
    "S071": (sw.S071, 0.002, [-0.025] * 3, [0.02] * 3, c071, {}),
    "S072": (sw.S072, 0.002, [-0.025] * 3, [0.02] * 3, c072, {}),
    "S073": (sw.S073, 0.002, [-0.004] * 3, [0.01] * 3, c073, {}),
    "S075": (sw.S075, 0.002, [-0.02], [0.01] * 3, c075, {"rng": 0.002}),
    "S076": (sw.S076, 0.002, [-0.02] * 4, [0.03, 0.02], c076, {}),
    "S077": (sw.S077, 0.002, [-0.02] * 5, [0.003, 0.02, 0.02], c077, {}),
    "S078": (sw.S078, 0.002, [-0.02] * 4, [0.02] * 3, c078, {}),
    "S080": (sw.S080, 0.001, [-0.01] * 9, [0.01] * 3, c080, {}),
    "S081": (sw.S081, 0.002, [-0.03] * 3, [0.02] * 3, c081, {}),
}


@pytest.mark.parametrize("sid", sorted(CASES))
def test_synthetic_mr_entry_bar_and_fill(sid):
    cls, g, dip, up, cond, kw = CASES[sid]
    rows = _rows(g, dip, up, **kw)
    o, h, l, c, v = arrays(rows)
    s = _first(cond(o, h, l, c, v))
    assert s is not None and s >= N_BASE, f"{sid}: fixture does not produce the pattern"
    res, ss = _run(cls, rows, s)
    _check_entry(res, ss, rows, s)


@pytest.mark.parametrize("sid", sorted(CASES))
def test_synthetic_mr_near_miss_on_falling_base(sid):
    cls, g, dip, up, cond, kw = CASES[sid]
    rows = _rows(-abs(g), dip, up, **kw)
    o, h, l, c, v = arrays(rows)
    assert not np.asarray(cond(o, h, l, c, v))[N_BASE - 5:].any()
    res, ss = _run(cls, rows, N_BASE - 3, run_from=2, extra_after=len(rows) - N_BASE + 2)
    assert len(entries(res)) == 0


# ------------------------------------------------------------------ stops / exits for selected cases
def test_synthetic_s070_disaster_stop_3atr_and_mid_band_exit():
    cls, g, dip, up, cond, kw = CASES["S070"]
    rows = _rows(g, dip, up + [0.02] * 3)
    o, h, l, c, v = arrays(rows)
    s = _first(cond(o, h, l, c, v))
    res, ss = _run(cls, rows, s, extra_after=8)
    fill = _check_entry(res, ss, rows, s)
    a = np.asarray(atr(h[: s + 1], l[: s + 1], c[: s + 1], 14))[-1]
    assert stop_in_force(res, close_ts(ss[s + 1])) == pytest.approx(fill - 3 * a, abs=0.011)
    mid = np.asarray(sma(c, 20))
    x = next(t for t in range(s + 1, len(c)) if c[t] >= mid[t])
    ex = orders(res)
    ex = ex[ex["tag"] == "close_at_mid"]
    assert ex["submitted_at"].iloc[0] == close_ts(ss[x])
    fx = fills_of(res, ex["order_id"].iloc[0])
    assert fx["ts"].iloc[0] == open_ts(ss[x + 1])


def test_synthetic_s068_exit_first_close_above_sma5_no_stop():
    cls, g, dip, up, cond, kw = CASES["S068"]
    rows = _rows(g, dip, up)
    o, h, l, c, v = arrays(rows)
    s = _first(cond(o, h, l, c, v))
    res, ss = _run(cls, rows, s, extra_after=8)
    _check_entry(res, ss, rows, s)
    assert stop_in_force(res, close_ts(ss[s + 3])) is None             # published baseline: no stop
    s5 = np.asarray(sma(c, 5))
    x = next(t for t in range(s + 1, len(c)) if c[t] > s5[t])
    ex = orders(res)
    ex = ex[ex["tag"] == "close_above_sma"]
    assert ex["submitted_at"].iloc[0] == close_ts(ss[x])


def test_synthetic_s072_z_stop_exits_next_open():
    cls, g, dip, up, cond, kw = CASES["S072"]
    rows = _rows(g, dip, [-0.05, -0.05, 0.0])
    o, h, l, c, v = arrays(rows)
    s = _first(cond(o, h, l, c, v))
    z = np.asarray(zscore(c, 20))
    x = next(t for t in range(s + 1, len(c)) if z[t] < -3 or z[t] >= 0)
    assert z[x] < -3
    res, ss = _run(cls, rows, s, extra_after=6)
    _check_entry(res, ss, rows, s)
    ex = orders(res)
    ex = ex[ex["tag"] == "z_stop"]
    assert len(ex) >= 1 and ex["submitted_at"].iloc[0] == close_ts(ss[x])
    assert fills_of(res, ex["order_id"].iloc[0])["ts"].iloc[0] == open_ts(ss[x + 1])


def test_synthetic_s073_time_exit_after_5_sessions():
    cls, g, dip, up, cond, kw = CASES["S073"]
    rows = _rows(g, dip, [-0.001] * 8)                                 # never closes above SMA5
    o, h, l, c, v = arrays(rows)
    s = _first(cond(o, h, l, c, v))
    res, ss = _run(cls, rows, s, extra_after=8, params={"n_down": 3})
    _check_entry(res, ss, rows, s)
    te = orders(res)
    te = te[te["tag"] == "time_exit"]
    assert len(te) == 1 and te["submitted_at"].iloc[0] == close_ts(ss[s + 5])
    assert fills_of(res, te["order_id"].iloc[0])["ts"].iloc[0] == open_ts(ss[s + 6])


def test_synthetic_s081_moving_stop_ema20_minus_3atr():
    cls, g, dip, up, cond, kw = CASES["S081"]
    rows = _rows(g, dip, [0.004, 0.004, 0.004])
    o, h, l, c, v = arrays(rows)
    s = _first(cond(o, h, l, c, v))
    res, ss = _run(cls, rows, s, extra_after=4)
    _check_entry(res, ss, rows, s)
    for d in (s + 1, s + 2):
        e = np.asarray(ema(c[: d + 1], 20))[-1]
        a = np.asarray(atr(h[: d + 1], l[: d + 1], c[: d + 1], 20))[-1]
        assert stop_in_force(res, close_ts(ss[d])) == pytest.approx(e - 3 * a, abs=0.011)


# ------------------------------------------------------------------ S079 (volume-confirmed flash selloff)
def _s079_rows(vol_mult=3.0, g=0.002):
    def extra(rows):
        o, h, l, c, v = rows[N_BASE]
        rows[N_BASE] = (o, h, l, c, 1_000_000.0 * vol_mult)
    return _rows(g, [-0.07], [0.01] * 3, extra=extra)


def test_synthetic_s079_flash_selloff_entry_stop_3atr():
    rows = _s079_rows()
    o, h, l, c, v = arrays(rows)
    s = _first(c079(o, h, l, c, v))
    assert s == N_BASE
    res, ss = _run(sw.S079, rows, s)
    fill = _check_entry(res, ss, rows, s)
    a = np.asarray(atr(h[: s + 1], l[: s + 1], c[: s + 1], 14))[-1]
    assert stop_in_force(res, close_ts(ss[s + 1])) == pytest.approx(fill - 3 * a, abs=0.011)


def test_synthetic_s079_near_miss_volume_below_1_5x():
    rows = _s079_rows(vol_mult=1.45)
    o, h, l, c, v = arrays(rows)
    assert _first(c079(o, h, l, c, v)) is None
    res, ss = _run(sw.S079, rows, N_BASE, run_from=1, extra_after=3)
    assert len(entries(res)) == 0


# ------------------------------------------------------------------ S084 (52-week-low bounce, SP500 universe)
def _s084_rows(vol=2_000_000.0):
    c = list(trend_closes(N_BASE, 60.0, -0.001, amp=0.01))
    for r in [-0.02] * 3 + [0.02, 0.0, 0.0, 0.0, 0.0, 0.0]:
        c.append(c[-1] * (1 + r))
    return cent_rows(rows_from_closes(c, rng=0.004, vol=vol))


def test_synthetic_s084_low_bounce_entry_stop_below_252d_low():
    rows = _s084_rows()
    o, h, l, c, v = arrays(rows)
    s = _first(c084(o, h, l, c, v))
    assert s is not None
    res, ss = _run(sw.S084, rows, s)
    _check_entry(res, ss, rows, s)
    ll = np.asarray(donchian(h, l, 252)["lower"])[s]
    assert stop_in_force(res, close_ts(ss[s + 1])) == pytest.approx(ll - 0.01, abs=0.011)


def test_synthetic_s084_near_miss_adv_below_20m():
    rows = _s084_rows(vol=200_000.0)
    o, h, l, c, v = arrays(rows)
    assert _first(c084(o, h, l, c, v)) is None
    res, ss = _run(sw.S084, rows, N_BASE, run_from=2, extra_after=10)
    assert len(entries(res)) == 0


def test_s084_records_survivorship_caveat():
    assert any("SURVIVORSHIP" in d for d in sw.S084.deviations)
