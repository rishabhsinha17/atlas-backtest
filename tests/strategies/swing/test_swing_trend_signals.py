"""Synthetic-pattern trigger / near-miss tests for the swing trend strategies S046-S067.

SYNTHETIC data, mechanics only: exact signal bar, fill bar/price, stop and
target levels; a minimally changed pattern must not trigger.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from swing_fixtures import (CAL, arrays, cent_rows, close_ts, entries, fills_of, flat_rows, limit_in_force, open_ts, orders,  # noqa: E402
                            portal_for, rows_from_closes, run, sessions, stop_in_force, with_universe, to_df,
                            trend_closes)

from atlasbt.indicators import (atr, bandwidth_percentile, bollinger, bullish_engulfing, darvas_box, dmi, donchian,  # noqa: E402
                                ema, head_and_shoulders, ichimoku, inside_bar, macd, nr_n, parabolic_sar,
                                percentile_rank, realized_vol, rsi, sma, squeeze_on, supertrend,
                                volatility_contraction)
from atlasbt.strategies import swing as sw  # noqa: E402
from atlasbt.strategies.swing_trend import hs_neckline_bars  # noqa: E402

SYM = "XYZ"


def _run(cls, rows, s, params=None, extra_after=3, run_from=2, sym=SYM, **attrs):
    rows[:] = cent_rows(rows)                    # prices on the $0.01 grid (engine fills are tick-rounded)
    ss = sessions(len(rows))
    portal = portal_for({sym: to_df(rows)})
    T = with_universe(cls, [sym], **attrs)
    end = ss[min(s + extra_after, len(rows) - 1)]
    res = run(T, portal, ss[s - run_from], end, params)
    return res, ss


def _first_entry(res):
    e = entries(res)
    assert len(e), "expected an entry order"
    return e.sort_values("order_id").iloc[0]


def _assert_market_entry(res, ss, rows, s, side=1):
    e = _first_entry(res)
    assert e["submitted_at"] == close_ts(ss[s])                 # decided at the signal close
    assert np.sign(e["quantity"]) == side
    f = fills_of(res, e["order_id"])
    assert len(f) and f["ts"].iloc[0] == open_ts(ss[s + 1])     # filled at the NEXT open
    assert f["price"].iloc[0] == pytest.approx(rows[s + 1][0])
    return float(f["price"].iloc[0])


def _no_entry(res):
    assert len(entries(res)) == 0


def _base_flat(n_trend=260, n_flat=25, p_end=None):
    c = trend_closes(n_trend, 40.0, 0.002)
    rows = rows_from_closes(c)
    p = p_end or float(c[-1])
    rows += flat_rows(n_flat, p)
    return rows, p


# ===================================================================== S046
@pytest.mark.parametrize("side", [1, -1])
def test_synthetic_s046_breakout_entry_and_atr_stop(side):
    rows, p = _base_flat()
    s = len(rows)
    sig = (p, p * 1.03, p * 0.995, p * 1.025) if side > 0 else (p, p * 1.005, p * 0.97, p * 0.975)
    rows += [sig + (1e6,), (sig[3] * 1.001, sig[3] * 1.01, sig[3] * 0.99, sig[3], 1e6)] + \
        [(sig[3], sig[3] * 1.004, sig[3] * 0.996, sig[3], 1e6)] * 3
    res, ss = _run(sw.S046, rows, s)
    fill = _assert_market_entry(res, ss, rows, s, side)
    o, h, l, c, v = arrays(rows)
    a20 = np.asarray(atr(h[: s + 1], l[: s + 1], c[: s + 1], 20))[-1]
    exp_stop = fill - side * 2.0 * a20
    assert stop_in_force(res, close_ts(ss[s + 1])) == pytest.approx(exp_stop, abs=0.011)


def test_synthetic_s046_near_miss_close_inside_channel():
    rows, p = _base_flat()
    s = len(rows)
    rows += [(p, p * 1.03, p * 0.995, p * 1.004, 1e6)] + flat_rows(4, p * 1.004)   # high pokes out, close does not
    res, ss = _run(sw.S046, rows, s)
    _no_entry(res)


def test_synthetic_s046_channel_exit_next_open():
    rows, p = _base_flat()
    s = len(rows)
    rows += [(p, p * 1.03, p * 0.995, p * 1.025, 1e6)]
    rows += flat_rows(3, p * 1.025, 0.002)
    x = len(rows)                                         # close below prior 10d low
    rows += [(p * 1.025, p * 1.026, p * 0.98, p * 0.985, 1e6), (p * 0.99, p * 0.995, p * 0.985, p * 0.99, 1e6),
             (p * 0.99, p * 0.995, p * 0.985, p * 0.99, 1e6)]
    res, ss = _run(sw.S046, rows, s, extra_after=x - s + 2, params={"stop_atr": 10.0, "short": False})
    o = orders(res)
    ex = o[o["tag"] == "channel_exit"]
    assert len(ex) == 1 and ex["submitted_at"].iloc[0] == close_ts(ss[x])
    f = fills_of(res, ex["order_id"].iloc[0])
    assert f["ts"].iloc[0] == open_ts(ss[x + 1]) and f["price"].iloc[0] == pytest.approx(rows[x + 1][0])


# ===================================================================== S047
def test_synthetic_s047_turtle55_entry_trailing_stop_and_pyramid_add():
    c = trend_closes(260, 40.0, 0.002)
    rows = rows_from_closes(c)
    p = float(c[-1])
    rows += flat_rows(60, p, 0.04)                       # wide range: N ~ 8% so one unit < 10% notional
    s = len(rows)
    rows += [(p, p * 1.07, p * 0.99, p * 1.06, 1e6)]
    q = p * 1.06
    rows += [(q, q * 1.01, q * 0.99, q, 1e6)]                          # entry day (fill at open = q)
    rows += [(q, q * 1.10, q * 0.995, q * 1.08, 1e6)] + flat_rows(3, q * 1.08, 0.003)
    res, ss = _run(sw.S047, rows, s, params={"adds": 1}, extra_after=5)
    fill = _assert_market_entry(res, ss, rows, s)
    o, h, l, c, v = arrays(rows)
    n_sig = np.asarray(atr(h[: s + 1], l[: s + 1], c[: s + 1], 20))[-1]
    a_now = np.asarray(atr(h[: s + 2], l[: s + 2], c[: s + 2], 20))[-1]
    exp = max(fill - 2 * n_sig, c[s + 1] - 2 * a_now)                 # ratchet at the entry-day close
    assert stop_in_force(res, close_ts(ss[s + 1])) == pytest.approx(exp, abs=0.011)
    add = orders(res)
    add = add[add["tag"] == "add"].sort_values("order_id")
    assert len(add) >= 1 and add["submitted_at"].iloc[0] == close_ts(ss[s + 1])
    assert add["stop_price"].iloc[0] == pytest.approx(fill + 0.5 * n_sig, abs=0.011)
    fa = fills_of(res, add["order_id"].iloc[0])
    assert len(fa) == 1 and open_ts(ss[s + 2]) <= fa["ts"].iloc[0] < close_ts(ss[s + 2])
    assert fa["price"].iloc[0] == pytest.approx(max(add["stop_price"].iloc[0], rows[s + 2][0]), abs=1e-6)
    # after the add, the protective stop covers both units at >= add fill - 2N
    st = orders(res)
    st = st[(st["tag"] == "stop") & (st["submitted_at"] == close_ts(ss[s + 2]))]
    assert len(st) == 1 and st["quantity"].iloc[0] == -(fa["quantity"].iloc[0] + fills_of(res, _first_entry(res)["order_id"])["quantity"].sum())
    assert st["stop_price"].iloc[0] >= fa["price"].iloc[0] - 2 * n_sig - 0.011


def test_synthetic_s047_near_miss_54_day_high_only():
    rows, p = _base_flat(n_flat=60)
    rows[-50] = (p, p * 1.04, p * 0.995, p, 1e6)          # a higher high 50 bars back
    s = len(rows)
    rows += [(p, p * 1.035, p * 0.995, p * 1.03, 1e6)] + flat_rows(3, p * 1.03)
    res, ss = _run(sw.S047, rows, s)
    _no_entry(res)


# ===================================================================== S048
def _s048_rows(second_breakout=True, winner=True):
    p = 60.0
    rows = flat_rows(285, p)                         # no earlier breakouts: no prior hypothetical S1 trade
    a = len(rows)
    rows += [(p, p * 1.03, p * 0.995, p * 1.025, 1e6)]                  # breakout A
    q = p * 1.025
    for k in range(1, 13):                                               # strong run: hypothetical winner
        cc = q * 1.02 ** k
        rows.append((cc / 1.02, cc * 1.003, cc / 1.02 * 0.998, cc, 1e6))
    top = rows[-1][3]
    lvl = q * 1.02 if winner else q * 0.99                               # exit at next open: win (> q) / loss
    rows += [(top, top * 1.002, lvl * 0.999, lvl, 1e6)]                  # close below 10d low -> exit next open
    rows += flat_rows(25, lvl, 0.004)
    b = len(rows)
    if second_breakout:
        rows += [(lvl, lvl * 1.03, lvl * 0.995, lvl * 1.025, 1e6)]      # breakout B
    else:
        rows += [(lvl, lvl * 1.005, lvl * 0.995, lvl, 1e6)]
    rows += flat_rows(3, rows[-1][3])
    return rows, a, b


def test_synthetic_s048_skip_rule_skips_after_winning_s1_breakout():
    rows, a, b = _s048_rows()
    res, ss = _run(sw.S048, rows, a, extra_after=b - a + 3)
    e = entries(res).sort_values("order_id")
    assert list(e["submitted_at"]) == [close_ts(ss[a])]                  # A taken, B skipped
    o, h, l, c, v = arrays(rows)
    entry_a = rows[a + 1][0]
    exit_bar = a + 13
    assert c[exit_bar] < np.asarray(donchian(h, l, 10)["lower"])[exit_bar]
    assert rows[exit_bar + 1][0] > entry_a                               # the hypothetical/real A trade won


def test_synthetic_s048_skip_rule_takes_breakout_after_losing_s1():
    rows, a, b = _s048_rows(winner=False)
    res, ss = _run(sw.S048, rows, a, extra_after=b - a + 3)
    e = entries(res).sort_values("order_id")
    assert list(e["submitted_at"]) == [close_ts(ss[a]), close_ts(ss[b])]
    assert rows[a + 14][0] < rows[a + 1][0]                              # A lost


def test_synthetic_s048_skip_off_takes_both_breakouts():
    rows, a, b = _s048_rows()
    res, ss = _run(sw.S048, rows, a, extra_after=b - a + 3, params={"skip": False})
    e = entries(res).sort_values("order_id")
    assert list(e["submitted_at"]) == [close_ts(ss[a]), close_ts(ss[b])]


def test_synthetic_s048_hypothetical_tracked_when_run_starts_after_first_breakout():
    """The skip decision uses the prior breakout even if it happened before the run (tracked from history)."""
    rows, a, b = _s048_rows()
    res, ss = _run(sw.S048, rows, b, run_from=2)
    _no_entry(res)
    res2, _ = _run(sw.S048, rows, b, run_from=2, params={"skip": False})
    assert len(entries(res2)) == 1


# ===================================================================== S049 / S050 (cross-based)
def _cross_rows(n_up=260, n_dn=45, n_rally=60, down_base=False):
    base = trend_closes(n_up, 40.0, -0.002 if down_base else 0.002, amp=0.004)
    last = base[-1]
    dn = last * np.exp(-0.004 * np.arange(1, n_dn + 1))
    up = dn[-1] * np.exp(0.008 * np.arange(1, n_rally + 1))
    return rows_from_closes(np.concatenate([base, dn, up]), rng=0.003)


def _first_true(mask, start):
    idx = np.flatnonzero(np.asarray(mask, bool)[start:])
    return int(start + idx[0]) if idx.size else None


def test_synthetic_s049_sma_cross_with_trend_filter_and_trailing_stop():
    rows = _cross_rows()
    o, h, l, c, v = arrays(rows)
    f, sl, t = np.asarray(sma(c, 20)), np.asarray(sma(c, 50)), np.asarray(sma(c, 200))
    cross = np.zeros(len(c), bool)
    cross[1:] = (f[:-1] <= sl[:-1]) & (f[1:] > sl[1:]) & (c[1:] > t[1:])
    s = _first_true(cross, 300)
    assert s is not None
    res, ss = _run(sw.S049, rows, s, run_from=5)
    fill = _assert_market_entry(res, ss, rows, s)
    a = np.asarray(atr(h[: s + 1], l[: s + 1], c[: s + 1], 14))[-1]
    a1 = np.asarray(atr(h[: s + 2], l[: s + 2], c[: s + 2], 14))[-1]
    exp = max(fill - 2.5 * a, c[s + 1] - 2.5 * a1)
    assert stop_in_force(res, close_ts(ss[s + 1])) == pytest.approx(exp, abs=0.011)


def test_synthetic_s049_near_miss_below_sma200():
    rows = _cross_rows(down_base=True)
    o, h, l, c, v = arrays(rows)
    f, sl, t = np.asarray(sma(c, 20)), np.asarray(sma(c, 50)), np.asarray(sma(c, 200))
    raw = np.zeros(len(c), bool)
    raw[1:] = (f[:-1] <= sl[:-1]) & (f[1:] > sl[1:])
    s = _first_true(raw, 300)
    assert s is not None and c[s] <= t[s]                                # cross happens, filter blocks
    res, ss = _run(sw.S049, rows, s, run_from=5)
    _no_entry(res)
    res2, _ = _run(sw.S049, rows, s, run_from=5, params={"trend_filter": False})
    _assert_market_entry(res2, ss, rows, s)


def test_synthetic_s050_golden_cross():
    c0 = np.concatenate([trend_closes(300, 60.0, -0.0015, amp=0.003),
                         trend_closes(200, 60.0 * np.exp(-0.0015 * 300), 0.004, amp=0.003)])
    rows = rows_from_closes(c0, rng=0.003)
    o, h, l, c, v = arrays(rows)
    f, sl = np.asarray(sma(c, 50)), np.asarray(sma(c, 200))
    cross = np.zeros(len(c), bool)
    cross[1:] = (f[:-1] <= sl[:-1]) & (f[1:] > sl[1:])
    s = _first_true(cross, 300)
    assert s is not None
    res, ss = _run(sw.S050, rows, s, run_from=5)
    _assert_market_entry(res, ss, rows, s)
    assert stop_in_force(res, close_ts(ss[s + 2])) is None              # no price stop by default
    # near miss: data ends one bar before the cross; nothing may be entered
    res2, ss2 = _run(sw.S050, rows[:s], s - 1, run_from=5, extra_after=0)
    _no_entry(res2)


# ===================================================================== S051
def test_synthetic_s051_pullback_entry_and_5day_low_stop():
    c0 = trend_closes(300, 40.0, 0.003, amp=0.002)
    rows = rows_from_closes(c0, rng=0.003)
    p = rows[-1][3]
    e9 = np.asarray(ema(arrays(rows)[3], 9))[-1]
    # pullback day: low tags EMA9, then a close above the prior high
    rows.append((p, p * 1.001, e9 * 0.999, p * 0.998, 1e6))
    s = len(rows)
    rows.append((p * 0.999, p * 1.02, p * 0.997, p * 1.015, 1e6))
    rows += flat_rows(3, p * 1.015, 0.002)
    o, h, l, c, v = arrays(rows)
    assert c[s] > np.asarray(ema(c, 21))[s] > np.asarray(ema(c, 50))[s] and l[s - 1] <= np.asarray(ema(c, 9))[s - 1]
    res, ss = _run(sw.S051, rows, s)
    _assert_market_entry(res, ss, rows, s)
    exp = float(np.min(l[s - 4: s + 1])) - 0.01
    assert stop_in_force(res, close_ts(ss[s + 1])) == pytest.approx(exp, abs=0.011)


def test_synthetic_s051_near_miss_close_not_above_prior_high():
    c0 = trend_closes(300, 40.0, 0.003, amp=0.002)
    rows = rows_from_closes(c0, rng=0.003)
    p = rows[-1][3]
    e9 = np.asarray(ema(arrays(rows)[3], 9))[-1]
    rows.append((p, p * 1.001, e9 * 0.999, p * 0.998, 1e6))
    s = len(rows)
    rows.append((p * 0.999, p * 1.0009, p * 0.997, p * 1.0008, 1e6))   # close < prior high (p*1.001)
    rows += flat_rows(3, p * 0.999, 0.0005)
    res, ss = _run(sw.S051, rows, s)
    _no_entry(res)


# ===================================================================== S052 / S053 / S056 / S057 / S062 / S063 / S064 / S065
def _cond_test(cls, rows, cond, params=None, start=280, side=1):
    s = _first_true(cond, start)
    assert s is not None, "fixture does not produce the pattern"
    res, ss = _run(cls, rows, s, params=params, run_from=3)
    fill = _assert_market_entry(res, ss, rows, s, side)
    return res, ss, s, fill


def _osc_rows(n=420, amp=0.06, per=45.0, g=0.0008):
    return rows_from_closes(trend_closes(n, 40.0, g, amp=amp, per=per), rng=0.004)


def _macd_cond(rows):
    o, h, l, c, v = arrays(cent_rows(rows))
    m = macd(c, 12, 26, 9)
    hist, line, t = np.asarray(m["hist"]), np.asarray(m["macd"]), np.asarray(sma(c, 200))
    cross = np.zeros(len(c), bool)
    cross[1:] = (hist[:-1] <= 0) & (hist[1:] > 0)
    return cross, cross & (line > 0) & (c > t)


def test_synthetic_s052_macd_histogram_cross():
    rows = rows_from_closes(trend_closes(420, 40.0, 0.002, amp=0.02, per=25), rng=0.004)
    cross, cond = _macd_cond(rows)
    res, ss, s, fill = _cond_test(sw.S052, rows, cond)
    o, h, l, c, v = arrays(rows)
    a = np.asarray(atr(h[: s + 1], l[: s + 1], c[: s + 1], 14))[-1]
    a1 = np.asarray(atr(h[: s + 2], l[: s + 2], c[: s + 2], 14))[-1]
    assert stop_in_force(res, close_ts(ss[s + 1])) == pytest.approx(max(fill - 2 * a, c[s + 1] - 2 * a1), abs=0.011)


def test_synthetic_s052_near_miss_histogram_cross_below_zero_line():
    rows = rows_from_closes(trend_closes(420, 40.0, -0.002, amp=0.02, per=25), rng=0.004)
    cross, cond = _macd_cond(rows)
    assert cross[300:400].any() and not cond[300:400].any()
    res, ss = _run(sw.S052, rows, 302, run_from=2, extra_after=95)
    _no_entry(res)


def test_synthetic_s053_adx_breakout():
    c0 = np.concatenate([trend_closes(260, 40.0, 0.0, amp=0.01, per=11), trend_closes(60, 40.0, 0.006, amp=0.002)])
    rows = rows_from_closes(c0, rng=0.004)
    o, h, l, c, v = arrays(rows)
    d = dmi(h, l, c, 14)
    up = np.asarray(donchian(h, l, 20)["upper"])
    cond = (c > up) & (np.asarray(d["adx"]) > 25) & (np.asarray(d["plus_di"]) > np.asarray(d["minus_di"]))
    res, ss, s, fill = _cond_test(sw.S053, rows, cond, start=262)
    # near miss: same breakout but ADX threshold 99 -> no entry at s
    res2, _ = _run(sw.S053, rows, s, params={"adx_min": 99}, run_from=3)
    _no_entry(res2)


def test_synthetic_s056_squeeze_then_upper_band_cross():
    c0 = list(trend_closes(300, 40.0, 0.0005, amp=0.03, per=17))
    c0 += list(np.full(30, c0[-1]) * (1 + 0.0015 * np.sin(np.arange(30))))    # squeeze
    p = c0[-1]
    c0 += [p * 1.03, p * 1.035, p * 1.04, p * 1.04]
    rows = rows_from_closes(c0, rng=0.002)
    o, h, l, c, v = arrays(rows)
    pct = np.asarray(bandwidth_percentile(c, 20, 2.0, 252))
    up = np.asarray(bollinger(c, 20, 2.0)["upper"])
    cond = np.zeros(len(c), bool)
    cond[1:] = (pct[:-1] <= 0.10) & (c[:-1] <= up[:-1]) & (c[1:] > up[1:])
    res, ss, s, fill = _cond_test(sw.S056, rows, cond, start=310)
    assert s == 330
    res2, _ = _run(sw.S056, rows, s, params={"bw_pct": 0.0}, run_from=3)    # never in the bottom 0%
    _no_entry(res2)


def test_synthetic_s057_keltner_squeeze_release():
    c0 = list(trend_closes(300, 40.0, 0.0005, amp=0.03, per=17))
    c0 += list(np.full(30, c0[-1]) * (1 + 0.0015 * np.sin(np.arange(30))))
    p = c0[-1]
    c0 += [p * 1.03, p * 1.035, p * 1.04, p * 1.04]
    rows = rows_from_closes(c0, rng=0.004)
    o, h, l, c, v = arrays(rows)
    sq = np.asarray(squeeze_on(h, l, c, 20, 2.0, 1.5, 20))
    up = np.asarray(bollinger(c, 20, 2.0)["upper"])
    cond = np.zeros(len(c), bool)
    cond[1:] = sq[:-1] & ~sq[1:] & (c[1:] > up[1:])
    res, ss, s, fill = _cond_test(sw.S057, rows, cond, start=310)
    a = np.asarray(atr(h[: s + 1], l[: s + 1], c[: s + 1], 14))[-1]
    assert stop_in_force(res, close_ts(ss[s + 1])) == pytest.approx(fill - 3 * a, abs=0.011)
    res2, _ = _run(sw.S057, rows, s, params={"kc_k": 0.1}, run_from=3)       # no squeeze with a tiny KC
    _no_entry(res2)


# ===================================================================== S058 / S059 (next-day stop entries)
def _nr_rows(narrow=True):
    rows, p = _base_flat(n_trend=100, n_flat=0)
    rows += [(p, p * (1.02 + 0.002 * k), p * (0.98 - 0.002 * k), p, 1e6) for k in range(8)]   # widening ranges
    s = len(rows)
    rows.append((p, p * 1.004, p * 0.996, p * 1.001, 1e6) if narrow else (p, p * 1.03, p * 0.97, p, 1e6))
    return rows, s, p


def test_synthetic_s058_nr7_buy_stop_one_tick_above_high_with_2r_target():
    rows, s, p = _nr_rows()
    hi, lo = rows[s][1], rows[s][2]
    rows.append((p * 1.002, p * 1.008, p * 1.0, p * 1.006, 1e6))    # trades through the stop
    rows += flat_rows(8, p * 1.005, 0.001)                          # neither stop nor 2R target
    res, ss = _run(sw.S058, rows, s, extra_after=8)
    e = _first_entry(res)
    assert e["submitted_at"] == close_ts(ss[s]) and e["type"] == "STP"
    assert e["stop_price"] == pytest.approx(round(hi + 0.01, 2))
    f = fills_of(res, e["order_id"])
    assert len(f) and f["ts"].iloc[0] >= open_ts(ss[s + 1]) and f["ts"].iloc[0] < close_ts(ss[s + 1])
    fill = float(f["price"].iloc[0])
    assert fill == pytest.approx(max(round(hi + 0.01, 2), rows[s + 1][0]))
    assert stop_in_force(res, close_ts(ss[s + 1])) == pytest.approx(lo, abs=0.011)
    assert limit_in_force(res, close_ts(ss[s + 1])) == pytest.approx(fill + 2 * (fill - lo), abs=0.011)
    o = orders(res)
    te = o[o["tag"] == "time_exit"]
    assert len(te) == 1 and te["submitted_at"].iloc[0] == close_ts(ss[s + 5])   # entry session = 1 -> 5th close


def test_synthetic_s058_near_miss_and_unfilled_stop_expires():
    rows, s, p = _nr_rows(narrow=False)
    rows += flat_rows(3, p)
    res, ss = _run(sw.S058, rows, s, extra_after=0)
    _no_entry(res)
    rows, s, p = _nr_rows()
    rows += flat_rows(3, p, 0.001)                                  # never reaches the stop
    res, ss = _run(sw.S058, rows, s, extra_after=1)
    e = _first_entry(res)
    assert e["status"] == "EXPIRED" and len(fills_of(res, e["order_id"])) == 0


@pytest.mark.parametrize("nested", [1, 2])
def test_synthetic_s059_inside_day_breakout(nested):
    rows, p = _base_flat(n_trend=100, n_flat=0)
    rows.append((p, p * 1.05, p * 0.95, p, 1e6))
    rows.append((p, p * 1.03, p * 0.97, p, 1e6))
    if nested == 2:
        rows.append((p, p * 1.02, p * 0.98, p, 1e6))
    s = len(rows) - 1
    hi, lo = rows[s][1], rows[s][2]
    rows.append((p * 1.01, p * 1.06, p * 1.0, p * 1.05, 1e6))
    rows += flat_rows(6, p * 1.05, 0.001)
    res, ss = _run(sw.S059, rows, s, params={"nested": nested}, extra_after=6)
    e = _first_entry(res)
    assert e["submitted_at"] == close_ts(ss[s]) and e["stop_price"] == pytest.approx(round(hi + 0.01, 2))
    fill = float(fills_of(res, e["order_id"])["price"].iloc[0])
    assert stop_in_force(res, close_ts(ss[s + 1])) == pytest.approx(lo, abs=0.011)
    assert limit_in_force(res, close_ts(ss[s + 1])) == pytest.approx(fill + 2.0 * (fill - lo), abs=0.011)


def test_synthetic_s059_near_miss_equal_high_is_not_inside():
    rows, p = _base_flat(n_trend=100, n_flat=0)
    rows.append((p, p * 1.05, p * 0.95, p, 1e6))
    rows.append((p, p * 1.05, p * 0.97, p, 1e6))                    # equal high -> not strictly inside
    s = len(rows) - 1
    rows += flat_rows(3, p)
    res, ss = _run(sw.S059, rows, s, extra_after=0)
    _no_entry(res)


# ===================================================================== S060
def test_synthetic_s060_darvas_box_breakout():
    rows, p = _base_flat(n_trend=260, n_flat=0)
    rows.append((p, p * 1.06, p * 0.99, p * 1.05, 1e6))             # resistance bar
    rows += flat_rows(19, p * 1.05, 0.004)                          # box under the resistance
    s = len(rows)
    rows.append((p * 1.05, p * 1.08, p * 1.05, p * 1.07, 1e6))
    rows += flat_rows(3, p * 1.07)
    o, h, l, c, v = arrays(rows)
    d = darvas_box(h, l, c, 20, 5, 2.0, 20)
    assert bool(np.asarray(d["signal"])[s]) and not np.asarray(d["signal"])[s - 5:s].any()
    res, ss = _run(sw.S060, rows, s)
    _assert_market_entry(res, ss, rows, s)
    floor = np.asarray(d["floor"])[s]
    assert stop_in_force(res, close_ts(ss[s + 1])) == pytest.approx(floor - 0.01, abs=0.011)
    res2, _ = _run(sw.S060, rows, s, params={"depth_atr": 0.1})     # floor too deep for a 0.1 ATR box
    _no_entry(res2)


# ===================================================================== S061
def test_synthetic_s061_vcp_breakout_initial_stop_prior_10d_low():
    c0 = list(trend_closes(240, 30.0, 0.003, amp=0.002))
    p = c0[-1]
    rows = rows_from_closes(c0, rng=0.004)
    for k, r in enumerate([0.03, 0.02, 0.012]):                      # three contracting 10-bar ranges
        rows += [(p, p * (1 + r), p * (1 - r), p, 1e6)] * 10
    s = len(rows)
    rows.append((p, p * 1.05, p * 0.999, p * 1.04, 1e6))
    rows += flat_rows(3, p * 1.04, 0.002)
    o, h, l, c, v = arrays(rows)
    assert bool(np.asarray(volatility_contraction(h, l, c, 10, 3))[s])
    res, ss = _run(sw.S061, rows, s, universe=[SYM])
    _assert_market_entry(res, ss, rows, s)
    lo10 = float(np.min(l[s - 10: s]))
    a1 = np.asarray(atr(h[: s + 2], l[: s + 2], c[: s + 2], 14))[-1]
    assert stop_in_force(res, close_ts(ss[s + 1])) == pytest.approx(max(lo10, c[s + 1] - 3 * a1), abs=0.011)
    res2, _ = _run(sw.S061, rows, s, params={"contractions": 4})    # only 3 contractions in the fixture
    _no_entry(res2)


# ===================================================================== S062
def test_synthetic_s062_tenkan_kijun_cross_above_known_cloud():
    rows = _osc_rows(n=420, amp=0.05, per=40.0, g=0.0015)
    o, h, l, c, v = arrays(rows)
    ic = ichimoku(h, l, c, 9, 26, 52, 26)
    tk, kj, top = np.asarray(ic["tenkan"]), np.asarray(ic["kijun"]), np.asarray(ic["cloud_top"])
    cross = np.zeros(len(c), bool)
    cross[1:] = (tk[:-1] <= kj[:-1]) & (tk[1:] > kj[1:])
    cond = cross & (c > top)
    res, ss, s, fill = _cond_test(sw.S062, rows, cond)
    a = np.asarray(atr(h[: s + 1], l[: s + 1], c[: s + 1], 14))[-1]
    assert stop_in_force(res, close_ts(ss[s + 1])) == pytest.approx(fill - 2 * a, abs=0.011)



def test_synthetic_s062_near_miss_cross_below_cloud():
    rows = _osc_rows(n=420, amp=0.05, per=40.0, g=-0.0015)
    o, h, l, c, v = arrays(rows)
    ic = ichimoku(h, l, c, 9, 26, 52, 26)
    tk, kj, top = np.asarray(ic["tenkan"]), np.asarray(ic["kijun"]), np.asarray(ic["cloud_top"])
    cross = np.zeros(len(c), bool)
    cross[1:] = (tk[:-1] <= kj[:-1]) & (tk[1:] > kj[1:])
    s2 = _first_true(cross, 300)
    assert s2 is not None and not (cross & (c > top))[300:].any()
    res, ss = _run(sw.S062, rows, 302, run_from=2, extra_after=110)
    _no_entry(res)
    res2, _ = _run(sw.S062, rows, s2, run_from=1, extra_after=1, params={"cloud_filter": False})
    _assert_market_entry(res2, ss, rows, s2)


# ===================================================================== S063
def test_synthetic_s063_sar_flip_entry_and_resting_sar_stop():
    rows = _osc_rows(n=420, amp=0.05, per=30.0, g=0.0015)
    o, h, l, c, v = arrays(rows)
    ps = parabolic_sar(h, l, 0.02, 0.02, 0.2)
    s200 = np.asarray(sma(c, 200))
    up = np.zeros(len(c), bool)
    up[20:] = s200[20:] > s200[:-20]
    cond = (np.asarray(ps["flip"]) == 1) & up
    res, ss, s, fill = _cond_test(sw.S063, rows, cond)
    nxt = np.asarray(parabolic_sar(h[: s + 2], l[: s + 2], 0.02, 0.02, 0.2)["next_sar"])[-1]
    assert stop_in_force(res, close_ts(ss[s + 1])) == pytest.approx(nxt, abs=0.011)


def test_synthetic_s063_near_miss_flip_with_falling_sma200():
    rows = _osc_rows(n=420, amp=0.05, per=30.0, g=-0.0015)
    o, h, l, c, v = arrays(rows)
    flips = np.asarray(parabolic_sar(h, l, 0.02, 0.02, 0.2)["flip"]) == 1
    assert flips[300:410].sum() >= 2
    res, ss = _run(sw.S063, rows, 302, run_from=2, extra_after=105)
    _no_entry(res)


# ===================================================================== S064
def test_synthetic_s064_supertrend_flip_and_band_stop():
    rows = _osc_rows(n=420, amp=0.06, per=35.0)
    o, h, l, c, v = arrays(rows)
    st = supertrend(h, l, c, 10, 3.0)
    cond = np.asarray(st["flip"]) == 1
    res, ss, s, fill = _cond_test(sw.S064, rows, cond)
    st1 = supertrend(h[: s + 2], l[: s + 2], c[: s + 2], 10, 3.0)
    assert np.asarray(st1["direction"])[-1] == 1
    assert stop_in_force(res, close_ts(ss[s + 1])) == pytest.approx(np.asarray(st1["supertrend"])[-1], abs=0.011)
    res2, _ = _run(sw.S064, rows, s, params={"mult": 4.0}, run_from=0, extra_after=0)
    if np.asarray(supertrend(h, l, c, 10, 4.0)["flip"])[s] != 1:
        _no_entry(res2)


# ===================================================================== S065
def test_synthetic_s065_breakout_with_low_vol_percentile():
    rows, p = _base_flat(n_trend=300, n_flat=25)
    s = len(rows)
    rows += [(p, p * 1.03, p * 0.995, p * 1.025, 1e6)] + flat_rows(3, p * 1.025)
    o, h, l, c, v = arrays(rows)
    pct = np.asarray(percentile_rank(realized_vol(c, 20, annualize=None), 252, include_current=True))[s]
    assert pct < 0.8
    res, ss = _run(sw.S065, rows, s)
    fill = _assert_market_entry(res, ss, rows, s)
    a = np.asarray(atr(h[: s + 1], l[: s + 1], c[: s + 1], 20))[-1]
    a1 = np.asarray(atr(h[: s + 2], l[: s + 2], c[: s + 2], 20))[-1]
    assert stop_in_force(res, close_ts(ss[s + 1])) == pytest.approx(max(fill - 2 * a, c[s + 1] - 2 * a1), abs=0.011)
    res2, _ = _run(sw.S065, rows, s, params={"vol_pct": pct - 1e-9}, run_from=0, extra_after=0)
    _no_entry(res2)


# ===================================================================== S066
def test_synthetic_s066_bullish_engulfing_2r_and_time_exit():
    c0 = trend_closes(260, 40.0, 0.002, amp=0.002)
    rows = rows_from_closes(c0, rng=0.003)
    p = rows[-1][3]
    rows += [(p * (1 - 0.02 * k), p * (1 - 0.02 * k) * 1.002, p * (0.98 - 0.02 * k) * 0.998, p * (0.98 - 0.02 * k),
              1e6) for k in range(4)]                                   # pull back (RSI < 60)
    p = rows[-1][3] / 0.975
    rows.append((p * 0.985, p * 0.99, p * 0.96, p * 0.965, 1e6))      # prior: C < O
    s = len(rows)
    rows.append((p * 0.96, p * 0.995, p * 0.955, p * 0.99, 1e6))      # O < prior C, C > prior O
    rows += flat_rows(6, p * 0.99, 0.002)
    o, h, l, c, v = arrays(rows)
    assert bool(np.asarray(bullish_engulfing(o, h, l, c))[s]) and np.asarray(rsi(c, 14))[s] < 60
    res, ss = _run(sw.S066, rows, s, extra_after=5)
    fill = _assert_market_entry(res, ss, rows, s)
    stop = min(l[s], l[s - 1])
    assert stop_in_force(res, close_ts(ss[s + 1])) == pytest.approx(stop, abs=0.011)
    assert limit_in_force(res, close_ts(ss[s + 1])) == pytest.approx(fill + 2 * (fill - stop), abs=0.011)
    te = orders(res)
    te = te[te["tag"] == "time_exit"]
    assert len(te) == 1 and te["submitted_at"].iloc[0] == close_ts(ss[s + 3])
    rows2 = list(rows)
    rows2[s] = (p * 0.96, p * 0.995, p * 0.955, p * 0.984, 1e6)        # close below prior open: not engulfing
    res2, _ = _run(sw.S066, rows2, s)
    _no_entry(res2)


# ===================================================================== S067
def _hs_rows(rs_high=1.06):
    base = list(trend_closes(260, 40.0, 0.001, amp=0.002))
    p = base[-1]
    rows = rows_from_closes(base, rng=0.003)

    def leg(a, b, n):
        return list(np.linspace(a, b, n + 1)[1:])
    path = leg(p, p * 1.06, 6) + leg(p * 1.06, p * 1.0, 6) + leg(p * 1.0, p * 1.12, 6) + \
        leg(p * 1.12, p * 1.01, 6) + leg(p * 1.01, p * rs_high, 6) + leg(p * rs_high, p * 1.0, 5) + \
        [p * 0.985, p * 0.98, p * 0.975, p * 0.97, p * 0.97, p * 0.97]
    rows += rows_from_closes([p] + path, rng=0.002)[1:]
    return rows


def test_synthetic_s067_head_and_shoulders_short_neckline_and_3r():
    rows = cent_rows(_hs_rows())
    o, h, l, c, v = arrays(rows)
    hs = head_and_shoulders(h, l, c, 3, 14, 1.0, 1.0)
    s = _first_true(np.asarray(hs["signal"]), 262)
    assert s is not None
    res, ss = _run(sw.S067, rows, s, extra_after=3)
    fill = _assert_market_entry(res, ss, rows, s, side=-1)
    rs = np.asarray(hs["rs_high"])[s]
    assert stop_in_force(res, close_ts(ss[s + 1])) == pytest.approx(rs, abs=0.011)
    assert limit_in_force(res, close_ts(ss[s + 1])) == pytest.approx(fill - 3 * (rs - fill), abs=0.011)
    b1, b2 = hs_neckline_bars(h, l, 3, np.asarray(hs["ls_high"])[s], np.asarray(hs["head_high"])[s], rs, s)
    neck = l[b1] + (l[b2] - l[b1]) / (b2 - b1) * (s - b1)
    assert neck == pytest.approx(np.asarray(hs["neckline"])[s])
    # near miss: shoulders too asymmetric for sym_atr = 1 (RS much lower than LS)
    rows2 = cent_rows(_hs_rows(rs_high=1.02))
    o2, h2, l2, c2, v2 = arrays(rows2)
    assert not np.asarray(head_and_shoulders(h2, l2, c2, 3, 14, 1.0, 1.0)["signal"])[262:].any()
    res2, _ = _run(sw.S067, rows2, len(rows2) - 4, extra_after=3)
    _no_entry(res2)
