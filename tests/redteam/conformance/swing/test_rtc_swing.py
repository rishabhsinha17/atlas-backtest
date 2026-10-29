"""RED-TEAM spec-conformance probes for the swing family (S046-S084).

SYNTHETIC patterns (mechanics only). Each test builds an exact sheet scenario
(docs/strategies.json) and asserts engine-level behaviour.

* ``test_finding_*`` encode the SHEET behaviour and are expected to FAIL against
  the current implementation (each documents a finding in the red-team report).
* ``test_conf_*`` are conformance probes that pass (clauses checked and found
  conformant).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "strategies", "swing"))
from swing_fixtures import (SectorPortal, arrays, cent_rows, close_ts, entries, fills_of, flat_rows,  # noqa: E402
                            limit_in_force, open_ts, orders, portal_for, rows_from_closes, run, sessions,
                            stop_in_force, to_df, trend_closes, with_universe, zero_cost)

from atlasbt.core.types import RunStatus  # noqa: E402
from atlasbt.engine.backtester import run_backtest  # noqa: E402
from atlasbt.engine.strategy import Requirement  # noqa: E402
from atlasbt.indicators import atr, nr_n, rsi, sma  # noqa: E402
from atlasbt.strategies import swing as sw  # noqa: E402

SYM = "XYZ"


def _run(cls, rows, s, params=None, extra_after=3, run_from=2, sym=SYM, **attrs):
    rows[:] = cent_rows(rows)
    ss = sessions(len(rows))
    portal = portal_for({sym: to_df(rows)})
    T = with_universe(cls, [sym], **attrs)
    end = ss[min(s + extra_after, len(rows) - 1)]
    return run(T, portal, ss[s - run_from], end, params), ss


def _base_flat(n_trend=260, n_flat=25, half_range=0.005):
    c = trend_closes(n_trend, 40.0, 0.002)
    rows = rows_from_closes(c)
    p = float(c[-1])
    rows += flat_rows(n_flat, p, half_range)
    return rows, p


def _entry(res):
    e = entries(res)
    assert len(e), "expected an entry order"
    return e.sort_values("order_id").iloc[0]


def _fills_tagged(res, tag):
    f = res.fills
    return f[f["tag"] == tag]


# ============================================================================ entry-day fill anchoring
def _s046_gap_rows():
    """Signal close C, entry open gaps up by 4*ATR; the entry-day low trades below fill-2*ATR20 but stays
    above C-2*ATR20 (the planned level the bracket uses)."""
    rows, p = _base_flat()
    s = len(rows)
    rows.append((p, p * 1.03, p * 0.995, p * 1.025, 1e6))
    rows[:] = cent_rows(rows)
    o, h, l, c, v = arrays(rows)
    a20 = float(np.asarray(atr(h, l, c, 20))[-1])
    C = c[s]
    op = C + 4.0 * a20
    low = C + 1.0 * a20                           # below op - 2*a20 = C + 2*a20, above C - 2*a20
    rows.append((op, op + 0.2 * a20, low, op - 0.5 * a20, 1e6))
    rows += flat_rows(4, op - 0.5 * a20, 0.001)
    return rows, s, a20


def test_finding_s046_entry_day_stop_is_anchored_to_signal_close_not_fill():
    """S046 sheet: 'Stop 2*ATR20 from fill, execute on next available trade/gap'. On a gap-up fill the stop in
    force on the entry day must be fill - 2*ATR20; the bracket child uses signal close - 2*ATR20 (4 ATR looser)."""
    rows, s, a20 = _s046_gap_rows()
    res, ss = _run(sw.S046, rows, s, params={"short": False}, extra_after=3)
    e = _entry(res)
    fill = float(fills_of(res, e["order_id"])["price"].iloc[0])
    o = orders(res)
    child = o[(o["tag"] == "stop") & (o["submitted_at"] < close_ts(ss[s + 1]))]   # bracket child (entry day)
    assert len(child) == 1
    assert float(child["stop_price"].iloc[0]) == pytest.approx(fill - 2.0 * a20, abs=0.011)


def test_finding_s046_gap_up_entry_day_trade_through_fill_stop_must_exit_same_day():
    """Consequence of the anchoring: the entry-day low trades through fill-2*ATR20 -> the sheet trade is
    stopped on the ENTRY day at ~fill-2*ATR20. The implementation keeps the position overnight (optimistic
    here: the loss is only realised later, at the next open, if at all)."""
    rows, s, a20 = _s046_gap_rows()
    res, ss = _run(sw.S046, rows, s, params={"short": False}, extra_after=3)
    st = _fills_tagged(res, "stop")
    assert len(st) == 1, "no stop fill at all"
    assert open_ts(ss[s + 1]) <= st["ts"].iloc[0] < close_ts(ss[s + 1])


def _nr7_gap_rows(gap=1.05):
    rows, p = _base_flat(n_trend=100, n_flat=0)
    rows += [(p, p * (1.02 + 0.002 * k), p * (0.98 - 0.002 * k), p, 1e6) for k in range(8)]
    s = len(rows)
    rows.append((p, p * 1.004, p * 0.996, p * 1.001, 1e6))                  # NR7 signal day
    q = p * gap                                                              # entry open far above the 2R level
    rows.append((q, q * 1.01, q * 0.99, q * 1.005, 1e6))
    rows += flat_rows(4, q * 1.005, 0.001)
    return rows, s, p


def test_finding_s058_gap_above_planned_2r_target_exits_on_entry_day():
    """S058 sheet: 'Initial stop signal low; exit after 5 sessions or at 2R' with R from the actual fill
    (guide rule 4). Entry opens 5% above the signal high: fill-based 2R target = fill + 2*(fill - signal low)
    ~ +16%, never reached in the scenario. The bracket target uses the PLANNED entry (stop price) -> ~+2.4%,
    which is below the entry fill, so the position is liquidated on the entry day."""
    rows, s, p = _nr7_gap_rows()
    res, ss = _run(sw.S058, rows, s, extra_after=4)
    e = _entry(res)
    f = fills_of(res, e["order_id"])
    assert len(f) == 1
    tg = _fills_tagged(res, "target")
    assert len(tg) == 0, f"target filled at {tg['price'].tolist()} on {tg['ts'].tolist()} (fill {f['price'].iloc[0]})"


def test_finding_s066_market_entry_gap_planned_2r_target_below_fill():
    """S066 sheet: 'Stop min(signal low, prior low); exit 3d or 2R'. With a gap-up open above the planned
    2R level (signal close + 2*(signal close - stop)) the entry-day target is already below the fill."""
    c0 = trend_closes(260, 40.0, 0.002, amp=0.002)
    rows = rows_from_closes(c0, rng=0.003)
    p = rows[-1][3]
    rows += [(p * (1 - 0.02 * k), p * (1 - 0.02 * k) * 1.002, p * (0.98 - 0.02 * k) * 0.998, p * (0.98 - 0.02 * k),
              1e6) for k in range(4)]
    p = rows[-1][3] / 0.975
    rows.append((p * 0.985, p * 0.99, p * 0.96, p * 0.965, 1e6))
    s = len(rows)
    rows.append((p * 0.96, p * 0.995, p * 0.955, p * 0.99, 1e6))           # bullish engulfing
    stop = min(p * 0.955, p * 0.96)
    planned_tgt = p * 0.99 + 2 * (p * 0.99 - stop)                          # ~ +7% above the signal close
    q = planned_tgt * 1.02
    rows.append((q, q * 1.005, q * 0.995, q, 1e6))
    rows += flat_rows(3, q, 0.001)
    res, ss = _run(sw.S066, rows, s, extra_after=3)
    e = _entry(res)
    assert len(fills_of(res, e["order_id"])) == 1
    assert len(_fills_tagged(res, "target")) == 0


# ============================================================================ Turtle pyramiding vs caps
def test_finding_s047_pyramid_adds_blocked_by_10pct_notional_cap_at_typical_vol():
    """S047 grid 'pyramid 0/1/2 adds' under SWING_RISK. With a typical ETF N (ATR20 ~1.2% of price) one unit
    (0.5% NAV at a 2N stop = ~20% notional) is already clipped to the 10% notional cap. Resolution (verified):
    the clipping is faithful to the sizing profile and DECLARED; no dust adds (a few shares) may be placed."""
    rows, p = _base_flat(n_flat=60, half_range=0.006)
    s = len(rows)
    rows.append((p, p * 1.03, p * 0.995, p * 1.025, 1e6))
    q = p * 1.025
    for k in range(1, 8):                                                   # steady rise: +0.5N levels traded
        cc = q * (1 + 0.012 * k)
        rows.append((cc / 1.012, cc * 1.002, cc / 1.012 * 0.999, cc, 1e6))
    res, ss = _run(sw.S047, rows, s, params={"adds": 2}, extra_after=7)
    e = _entry(res)
    f = fills_of(res, e["order_id"])
    unit = float(f["quantity"].iloc[0])
    notional = float(f["price"].iloc[0] * unit)
    assert notional == pytest.approx(0.10 * 1e6, rel=0.01)                  # first unit already at the cap
    adds = _fills_tagged(res, "add")
    o = orders(res)
    add_orders = o[o["tag"] == "add"]
    assert (add_orders["quantity"].abs() >= 0.25 * unit).all(), add_orders[["quantity", "stop_price"]].to_string()
    assert (adds["quantity"].abs() >= 0.25 * unit).all()                    # a 1-share add is not a unit
    assert any("10% NAV" in d and "clipped" in d for d in sw.S047.deviations)
    assert any("10% NAV" in d and "clipped" in d for d in sw.S048.deviations)
    assert any("clipped" in str(lg.get("msg", "")) for lg in res.diagnostics["logs"])


# ============================================================================ aggregate gross cap
def _dip_rows(n, dip_at):
    c = list(trend_closes(n, 50.0, 0.002, amp=0.0))
    out = []
    for i in range(len(c)):
        out.append(c[i])
    # two -2% closes ending at dip_at, flat afterwards
    base = out[dip_at - 2]
    out[dip_at - 1] = base * 0.98
    out[dip_at] = base * 0.98 * 0.98
    for i in range(dip_at + 1, n):
        out[i] = out[dip_at]
    return cent_rows(rows_from_closes(out, rng=0.003, vol=5e6))


def test_finding_s068_filled_pending_entry_double_counted_in_gross_cap():
    """SWING_NOTIONAL 'Long notional 10% of NAV per instrument ... no implicit leverage' (gross <= 100%).
    ZZZ enters (10%) the day before nine names signal. At that close the true gross is 10% -> all nine fit
    (100%). SwingBase adds the already-FILLED ZZZ entry twice (position + stale pending until ZZZ is processed,
    alphabetically last), so the ninth name is cut to ~0."""
    n = 280
    d0 = 262
    names = [f"A{i:02d}" for i in range(9)]
    dfs = {"ZZZ": to_df(_dip_rows(n, d0))}
    for nm in names:
        dfs[nm] = to_df(_dip_rows(n, d0 + 1))
    o_, h, l, c, v = arrays(_dip_rows(n, d0 + 1))
    assert np.asarray(rsi(c, 2))[d0 + 1] < 5 and c[d0 + 1] > np.asarray(sma(c, 200))[d0 + 1]
    ss = sessions(n)
    T = with_universe(sw.S068, names + ["ZZZ"])
    res = run(T, portal_for(dfs), ss[d0 - 1], ss[d0 + 3])
    e = entries(res)
    late = e[e["submitted_at"] == close_ts(ss[d0 + 1])]
    assert set(late["symbol"]) == set(names), f"entries submitted: {sorted(late['symbol'])}"
    px = c[d0 + 1]
    qmin = np.floor(0.10 * 1e6 * 0.98 / px)
    assert (late["quantity"] >= qmin).all(), late[["symbol", "quantity"]].to_string()


def test_conf_s068_gross_cap_alphabetical_priority_when_cap_binds():
    """Declared convention: competing entries on one close are processed alphabetically and the 100% gross
    cap binds. 12 names signal together -> the first 10 alphabetical names get ~10% each, the last get 0."""
    n = 280
    d0 = 262
    names = [f"B{i:02d}" for i in range(12)]
    dfs = {nm: to_df(_dip_rows(n, d0)) for nm in names}
    ss = sessions(n)
    res = run(with_universe(sw.S068, names), portal_for(dfs), ss[d0 - 1], ss[d0 + 2])
    e = entries(res)
    got = sorted(e[e["quantity"] >= 1]["symbol"])
    assert got[:9] == names[:9] and "B11" not in got
    f = res.fills
    gross = (f[f["tag"] == "entry"]["price"] * f[f["tag"] == "entry"]["quantity"]).sum()
    assert gross <= 1.0 * 1e6 + 1.0


# ============================================================================ requirements
def test_finding_s055_missing_pit_universe_is_silent_no_trade_not_not_testable():
    """S055/S061/S084 (and the cross-sections S054/S074/S082) trade portal universe 'SP500'. A portal without
    that universe must give NOT_TESTABLE (guide §1 'Data the strategy cannot get: declare a critical
    Requirement'), not a silent OK run with zero trades."""
    rows = cent_rows(rows_from_closes(trend_closes(400, 30.0, 0.002)))
    portal = portal_for({"SPY": to_df(rows), "K00": to_df(rows)})          # no universes={'SP500': ...}
    ss = sessions(400)
    for cls in (sw.S055, sw.S061, sw.S084, sw.S054, sw.S074, sw.S082):
        res = run_backtest(cls(), portal, ss[300], ss[399], zero_cost(), raise_errors=False)
        assert res.status == RunStatus.NOT_TESTABLE, (cls.id, res.status, len(res.fills))
        assert "SP500" in res.not_testable_reason or cls is sw.S082     # S082 also lacks its sector ETFs here


def test_finding_s055_universe_data_unavailable_error_is_not_testable():
    """The real Portal raises DataUnavailableError when S&P 500 membership is not in the store."""
    from atlasbt.core.types import DataUnavailableError

    class NoMembership(SectorPortal):
        def universe(self, name, as_of):
            raise DataUnavailableError("S&P 500 membership not in store")

    rows = cent_rows(rows_from_closes(trend_closes(400, 30.0, 0.002)))
    portal = NoMembership(bars={("SPY", "1d"): to_df(rows), ("K00", "1d"): to_df(rows)})
    ss = sessions(400)
    res = run_backtest(sw.S055(), portal, ss[300], ss[399], zero_cost(), raise_errors=False)
    assert res.status == RunStatus.NOT_TESTABLE and "membership" in res.not_testable_reason


# ============================================================================ cross-section bucket symmetry
def test_finding_s074_sector_bucket_long_short_name_counts_asymmetric():
    """S074 sheet: 'long bottom decile and short top decile in sector-neutral groups'. In a 15-name sector the
    ceil-based quantile_bucket puts 1 name in decile 1 and 2 names in decile 10 (bucket 1 = floor(n/q) names,
    bucket q = ceil(n/q)): the short leg systematically holds more names than the long leg."""
    names = [f"K{i:02d}" for i in range(15)]
    n = 60
    ss = sessions(n)
    fri = next(i for i in range(40, n - 3) if
               pd.Timestamp(ss[i + 1]).isocalendar()[1] != pd.Timestamp(ss[i]).isocalendar()[1])
    dfs = {}
    for i, s in enumerate(names):
        c = [40.0 + i] * (fri - 4)
        for _ in range(5):
            c.append(c[-1] * (1 + (i - 7) * 0.002))
        c += [c[-1]] * (n - len(c))
        dfs[s] = to_df(cent_rows(rows_from_closes(c, rng=0.003)))
    bars = {(s, "1d"): df for s, df in dfs.items()}
    portal = SectorPortal(bars=bars, sectors={s: "Financials" for s in names}, universes={"SP500": names})
    T = type("S074_X", (sw.S074,), {"requires": [Requirement("bars", "1d", symbols=("K00",))]})
    res = run(T, portal, ss[fri - 2], ss[fri + 1])
    o = orders(res)
    nl, ns = int((o["quantity"] > 0).sum()), int((o["quantity"] < 0).sum())
    assert nl == ns, f"longs {nl} vs shorts {ns}"


# ============================================================================ conformance probes (pass)
def test_conf_s046_breakout_equal_to_prior_high_does_not_trigger():
    """'C_t > max(H_{t-20:t-1})' strict and excluding today: a close equal to the prior 20d high is no signal,
    while today's own high above it is irrelevant."""
    rows, p = _base_flat()
    rows[:] = cent_rows(rows)
    s = len(rows)
    hi20 = max(r[1] for r in rows[-20:])
    rows.append((p, hi20 * 1.02, p * 0.995, hi20, 1e6))
    rows += flat_rows(3, hi20, 0.001)
    res, ss = _run(sw.S046, rows, s, params={"short": False})
    assert len(entries(res)) == 0


def test_conf_s058_nr7_includes_today_and_entry_is_next_session_buy_stop():
    """NR7 'smallest of previous 7 completed days (including today)'; buy stop = signal high + 1 tick valid only
    for the next session; initial stop = signal low."""
    rows, s, p = _nr7_gap_rows(gap=1.0)
    rows[s + 1] = (p * 1.0, p * 1.008, p * 0.999, p * 1.006, 1e6)
    rows[:] = cent_rows(rows)
    o, h, l, c, v = arrays(rows)
    assert bool(np.asarray(nr_n(h, l, 7))[s])
    res, ss = _run(sw.S058, rows, s, extra_after=3)
    e = _entry(res)
    assert e["type"] == "STP" and e["stop_price"] == pytest.approx(round(h[s] + 0.01, 2))
    assert e["tif"] == "DAY"
    assert stop_in_force(res, close_ts(ss[s + 1])) == pytest.approx(l[s], abs=0.011)


def _s084_rows(bounce):
    c = list(np.linspace(80.0, 40.0, 300))                                  # steady decline: RSI14 pinned low
    c.append(c[-1] * (1 + bounce))
    c += [c[-1]] * 4
    return cent_rows(rows_from_closes(c, rng=0.002, vol=1e6))


@pytest.mark.parametrize("bounce,expect", [(0.020, True), (0.035, False)])
def test_conf_s084_within_3pct_of_prior_252d_low_threshold(bounce, expect):
    """S084 'C within 3% of prior 252d lowest low, RSI14 crosses above 30, ADV>=$20m; buy next open'.
    The bounce day crosses RSI14 above 30 in both cases; only the 2% bounce is within 3% of the prior low."""
    rows = _s084_rows(bounce)
    o, h, l, c, v = arrays(rows)
    s = 300
    r = np.asarray(rsi(c, 14))
    assert r[s - 1] <= 30 < r[s]
    dist = c[s] / l[s - 252:s].min() - 1
    assert (dist <= 0.03) == expect
    res, ss = _run(sw.S084, rows, s, extra_after=2, universe_name=None)
    e = entries(res)
    assert (len(e) == 1) == expect
    if expect:
        assert e["submitted_at"].iloc[0] == close_ts(ss[s])
        assert stop_in_force(res, close_ts(ss[s + 1])) == pytest.approx(round(l[s - 252:s].min() - 0.01, 2),
                                                                         abs=0.011)


def _me_indices(ss, after):
    from swing_fixtures import CAL
    return [i for i in range(after, len(ss)) if CAL.is_last_session_of_month(ss[i])]


def test_conf_s054_quintile_exit_hysteresis_at_rebalance():
    """S054 'long top decile ratio ... Hold one month; close if moves out of quintile at rebalance'.
    20 names: decile = 2 names, quintile = 4. At month-end 2 the former #1 (K19) is ranked 17/20 (still top
    quintile -> kept), the former #2 (K18) is ranked 10/20 (out of the quintile -> closed)."""
    names = [f"K{i:02d}" for i in range(20)]
    n = 340
    ss = sessions(n)
    m1, m2 = _me_indices(ss, 262)[:2]
    order2 = {s: i for i, s in enumerate(names)}
    order2.update({"K19": 16, "K16": 19, "K18": 9, "K09": 18})            # new ranks (0-based) at m2
    dfs = {}
    for i, s in enumerate(names):
        rows = [(100.0, 101.0, 99.0, 100.0, 1e6)] * m1
        c1 = 90.0 + 0.5 * i
        rows += [(c1, c1, c1, c1, 1e6)] * (m2 - m1)
        c2 = 90.0 + 0.5 * order2[s]
        rows += [(c2, c2, c2, c2, 1e6)] * (n - m2)
        dfs[s] = to_df(rows)
    bars = {(s, "1d"): df for s, df in dfs.items()}
    portal = SectorPortal(bars=bars, sectors={s: f"S{i % 11}" for i, s in enumerate(names)},
                          universes={"SP500": names})
    T = type("S054_X", (sw.S054,), {"requires": [Requirement("bars", "1d", symbols=("K00",))]})
    res = run(T, portal, ss[m1 - 1], ss[m2 + 2])
    o = orders(res)
    first = o[o["submitted_at"] == close_ts(ss[m1])]
    assert set(first[first["quantity"] > 0]["symbol"]) == {"K18", "K19"}
    second = o[o["submitted_at"] == close_ts(ss[m2])]
    assert "K18" in set(second[second["quantity"] < 0]["symbol"])          # out of the quintile -> closed
    longs_now = {s for s in names if o[(o["symbol"] == s) & (o["status"] == "FILLED")]["quantity"].sum() > 0}
    assert longs_now == {"K16", "K09", "K19"}, longs_now                   # new top decile + K19 kept
