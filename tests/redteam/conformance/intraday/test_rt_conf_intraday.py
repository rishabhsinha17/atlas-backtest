"""RED-TEAM spec-conformance probes for the intraday family (S001-S030, S116).

SYNTHETIC patterns (mechanics only). Each test builds the exact sheet scenario
(docs/strategies.json) and asserts engine-level behaviour. Tests named
``test_finding_*`` encode the SHEET behaviour and are expected to FAIL against
the current implementation (they document a finding); ``test_conf_*`` are
conformance probes that pass.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.clock import et
from atlasbt.core.types import RunStatus
from atlasbt.execution.constraints import round_stop_for_side
from atlasbt.strategies import registry
import atlasbt.strategies.intraday  # noqa: F401  (registers the classes)

from tests.strategies.intraday._synth import (
    CAL, DAY, SYM, V1, V5, Scenario, assert_status_ok, day_frame, flat, frame, history, orders, path,
    prior_sessions, run, starts,
)
from tests.strategies.intraday._scenarios import HIST, _one, _quote, s005, s006, s019


def _entries(r):
    return orders(r, "entry")


def _first_entry_at(r, hhmm, day=DAY):
    e = _entries(r)
    assert len(e) >= 1, "no entry order"
    assert e.iloc[0].submitted_at == et(day, hhmm), (e.iloc[0].submitted_at, et(day, hhmm))
    f = r.fills[r.fills.tag == "entry"]
    assert len(f) >= 1 and f.iloc[0].ts == et(day, hhmm)      # fills at the NEXT bar's open
    return e.iloc[0]


# ============================================================================ FINDINGS (expected to fail)
def _s017_rows(confirm_high):
    rows = flat(100.0, 30, rng=0.1)                                   # 09:30-09:59, HOD 100.1, LOD 99.9
    rows += [(100.0, 100.6, 99.95, 100.55, V1),                       # 10:00 first break of HOD
             (100.55, 100.8, 100.5, 100.75, V1),                      # 10:01 peak 100.80
             (100.75, 100.78, 100.65, 100.7, V1),                     # 10:02 pullback bar 1
             (100.7, 100.74, 100.62, 100.66, V1),                     # 10:03 pullback bar 2
             (100.66, 100.72, 100.6, 100.68, V1),                     # 10:04 pullback bar 3
             (100.68, 100.75, 100.63, 100.7, V1),                     # 10:05 pullback bar 4 (pb high 100.78)
             (100.7, confirm_high, 100.69, 100.79, V1),               # 10:06 close 100.79 > pullback high
             (100.79, 100.8, 100.78, 100.79, V1)]
    return rows


def test_finding_s017_confirmation_bar_with_new_high_must_enter():
    """S017 sheet: 'wait first 3-10 1m bars with <0.5*impulse retracement; buy next 1m close >pullback high'.
    Pullback = 4 bars, retracement (100.8-100.6)/(100.8-99.9)=22% < 50%; the 10:06 bar closes 100.79 > pullback
    high 100.78 -> buy at the 10:07 open. Its HIGH (100.85) prints above the prior peak (100.80); the code treats
    that as a new peak and restarts the pullback (intraday_breakout.py:868-870 run before the entry test at :874),
    so the sheet's entry never happens."""
    ok = Scenario("S017", _one("1m", _s017_rows(100.8)))
    r_ok = run(registry.get("S017"), ok)
    _first_entry_at(r_ok, "10:07")                                    # control: same bar, high == peak -> entry
    sc = Scenario("S017", _one("1m", _s017_rows(100.85)))
    r = run(registry.get("S017"), sc)
    assert_status_ok(r)
    e = _first_entry_at(r, "10:07")
    st = orders(r, "stop")
    assert st.iloc[0].stop_price == pytest.approx(round_stop_for_side(100.6, 0.01, True))
    assert e.quantity > 0


def _s018_rows(T):
    # all highs <= T, open 100.0; the 09:50 bar closes exactly at open*(1+1%) (T) and above VWAP
    return [(100.0, 100.25, 99.95, 100.2, V5), (100.2, 100.45, 100.15, 100.4, V5),
            (100.4, 100.65, 100.35, 100.6, V5), (100.6, 100.85, 100.55, 100.8, V5),
            (100.8, T, 100.75, T, V5),                                 # 09:50 close == open x 1.01
            (T, T, 100.9, 100.95, V5),                                 # 09:55
            (100.95, T, 100.3, 100.9, V5),                             # 10:00 low <= EMA20
            (100.9, 100.98, 100.85, 100.97, V5)]                       # 10:05 close > EMA20 -> signal 10:10


def test_finding_s018_trend_gate_is_strict_greater():
    """S018 sheet: 'By 11:00 price>open by 1%'. A day whose price reaches EXACTLY open x 1.01 (never above) is
    not a trend day; code uses `>=` (intraday_breakout.py:929) and trades it."""
    T_strict = 100.0 * (1 + 0.01) + 0.01
    r_ok = run(registry.get("S018"), Scenario("S018", _one("5m", _s018_rows(T_strict))))
    _first_entry_at(r_ok, "10:10")                                    # control: strictly above -> entry
    T = 100.0 * (1 + 0.01)
    r = run(registry.get("S018"), Scenario("S018", _one("5m", _s018_rows(T))))
    assert_status_ok(r)
    assert len(_entries(r)) == 0, "price == open x 1.01 is not 'price > open by 1%'"


@pytest.mark.parametrize("fn,sid", [(s005, "S005"), (s006, "S006")])
def test_finding_s005_s006_premarket_quote_requirement(fn, sid):
    """S005/S006 sheet: 'Premarket 09:29 midpoint'. Quotes exist only in RTH (10:00) -> the 09:29 midpoint is
    unavailable. Guide §1: missing critical data must give NOT_TESTABLE (or an opt-in flagged proxy), never a
    silent no-trade. Code declares Requirement('quotes', sessions=('RTH',)) (intraday_base.py:645), so the run is
    OK/grade B with zero trades."""
    trig, _ = fn()
    px = 103.0 if sid == "S005" else 97.0
    sc = replace(trig, quotes={SYM: _quote("10:00:30", px - 0.1, px + 0.1)})
    r = run(registry.get(sid), sc)
    assert r.status == RunStatus.NOT_TESTABLE or any("proxied" in x["reason"] for x in r.fidelity["reasons"]), \
        (r.status, len(_entries(r)), r.fidelity.get("grade"))


# ============================================================================ CONFORMANCE PROBES (pass)
def _s003_rows(retest_at):
    from tests.strategies.intraday._scenarios import _or_rows
    rows = _or_rows()                                                  # OR 09:30-09:44: H=100.5, L=99.5
    rows += [(100.0, 100.7, 99.95, 100.6, V1)]                         # b = 09:45 first close > H
    rows += [(100.6, 100.65, 100.55, 100.6, V1)] * (retest_at - 1)     # stays above H (lows > H)
    rows += [(100.6, 100.62, 100.45, 100.5, V1)]                       # retest bar b+retest_at: low <= H
    rows += [(100.5, 100.72, 100.48, 100.7, V1), (100.7, 100.72, 100.68, 100.7, V1)]  # close > H x 1.0005
    return rows


def test_conf_s003_retest_window_is_next_15_bars():
    """'low of next 15 bars <=OR high': a retest on bar b+15 counts, on b+16 it does not."""
    r = run(registry.get("S003"), Scenario("S003", _one("1m", _s003_rows(15))))
    e = _first_entry_at(r, "10:02")
    assert orders(r, "stop").iloc[0].stop_price == pytest.approx(100.0)   # min(retest swing low 100.45, OR mid 100)
    r2 = run(registry.get("S003"), Scenario("S003", _one("1m", _s003_rows(16))))
    assert len(_entries(r2)) == 0


def test_conf_s022_mirror_false_breakdown_long():
    rows = [(99.6, 99.65, 99.5, 99.55, V1)] * 15
    rows[8] = (99.6, 100.5, 99.55, 99.6, V1)                           # OR: H=100.5, L=99.5
    rows += [(99.55, 99.6, 99.3, 99.4, V1),                            # 09:45 first close < L
             (99.4, 99.7, 99.38, 99.65, V1),                           # 09:46 first close back > L, > VWAP
             (99.65, 99.66, 99.64, 99.65, V1)]
    r = run(registry.get("S022"), Scenario("S022", _one("1m", rows)))
    e = _first_entry_at(r, "09:47")
    assert e.quantity > 0
    assert orders(r, "stop").iloc[0].stop_price == pytest.approx(99.3)
    assert orders(r, "target").iloc[0].limit_price == pytest.approx(100.5)


def test_conf_s021_mirror_long_on_gap_down():
    days = HIST
    h = [history(days[:-1], 5, price=94.0)]
    last = flat(94.0, 58, v=V5) + path(94.0, [94.0 + 0.3 * (k + 1) for k in range(20)], rng=0.05, v=V5)
    h.append(frame(last, starts(days[-1], 5), 5))                      # prior close 100.0
    o = 98.0                                                           # gap -2%
    rows = [(o, o + 0.3, o - 0.3, o + 0.3, V5), (o + 0.3, o + 0.5, o + 0.25, o + 0.4, V5)]
    rows += path(o + 0.4, list(np.linspace(o + 0.5, 99.8, 8)), rng=0.03, v=V5)
    rows += [(99.8, 100.1, 99.75, 100.05, V5)]
    r = run(registry.get("S021"), Scenario("S021", _one("5m", rows, hist=pd.concat(h))))
    e = _first_entry_at(r, "09:35")
    assert e.quantity > 0
    assert orders(r, "stop").iloc[0].stop_price == pytest.approx(97.7)
    assert orders(r, "target").iloc[0].limit_price == pytest.approx(100.0)
    t = r.fills[r.fills.tag == "target"]
    assert len(t) == 1 and t.iloc[0].price == 100.0


def test_conf_s028_mirror_long():
    rows = flat(100, 6, rng=0.05, v=V5) + path(100, [99.7, 99.4, 99.1, 98.8], 0.05, V5)
    rows += path(98.8, [99.3], 0.05, V5) + [(99.3,) * 4 + (V5,)]
    lod = min(rw[2] for rw in rows[:11])
    r = run(registry.get("S028"), Scenario("S028", _one("5m", rows)))
    e = _first_entry_at(r, "10:25")
    assert e.quantity > 0
    assert orders(r, "stop").iloc[0].stop_price == pytest.approx(round_stop_for_side(lod, 0.01, True))


def _s023_base():
    from tests.strategies.intraday._scenarios import _daily_extra, _vsig_rows
    return _vsig_rows(1), _daily_extra(10.0)


def test_conf_s023_time_exit_after_12_bars():
    rows, dx = _s023_base()
    rows = rows + [(99.3, 99.35, 99.25, 99.3, V5)]                     # then flat 99.3: below VWAP, z > -3
    r = run(registry.get("S023"), Scenario("S023", _one("5m", rows), daily_extra=dx))
    _first_entry_at(r, "10:05")
    tx = r.fills[r.fills.tag == "time_exit"]
    assert len(tx) == 1 and tx.iloc[0].ts == et(DAY, "11:05")        # entry bar 10:05 + 12 x 5m


def test_conf_s023_z_stop_is_next_bar_market():
    rows, dx = _s023_base()
    rows = rows + [(99.3, 99.3, 97.5, 97.6, V5), (97.6, 97.7, 97.5, 97.55, V5)]
    r = run(registry.get("S023"), Scenario("S023", _one("5m", rows), daily_extra=dx))
    _first_entry_at(r, "10:05")
    ex = r.fills[r.fills.tag == "stop_z"]
    assert len(ex) == 1 and ex.iloc[0].ts == et(DAY, "10:10") and ex.iloc[0].price == pytest.approx(97.6)


def test_conf_s009_r_target_cancelled_by_time_exit():
    rows = [(100.0, 100.2, 99.8, 100.0, V5)] * 7 + [(100.0, 100.05, 99.95, 100.0, V5),
                                                  (100.0, 100.25, 99.98, 100.2, V5)]
    rows += [(100.2, 100.22, 100.18, 100.2, V5)] * 6 + [(100.2, 100.21, 100.19, 100.2, V5)] * 3
    rows += [(100.2, 101.5, 100.19, 101.0, V5)]                        # rallies through the 2R level after exit
    r = run(registry.get("S009"), Scenario("S009", _one("5m", rows)))
    _first_entry_at(r, "10:15")
    assert set(r.fills.tag) <= {"entry", "time_exit", "flat_by"}
    assert r.fills.quantity.sum() == 0


def test_conf_early_close_exit_and_no_flip():
    """'exit 15:55' on a 13:00 close = 12:55 (guide §1); one exit fill, flat, no double flatten."""
    ed = pd.Timestamp("2024-07-03")
    hist = history(prior_sessions(ed, 22), 1)
    o = 101.0
    closes = [o - 0.1, o - 0.15, o - 0.2, o - 0.25, o - 0.3, o - 0.35, o - 0.4, o - 0.35, o - 0.4, o - 0.4]
    rows = path(o, closes, rng=0.05) + [(o - 0.4,) * 4 + (V1,)]
    sc = Scenario("S116", {("SPY", "1m"): pd.concat([hist, day_frame(ed, rows, 1)])}, universe=("SPY",),
                  symbol="SPY", day=ed)
    r = run(registry.get("S116"), sc)
    assert_status_ok(r)
    _first_entry_at(r, "09:40", day=ed)
    ex = r.fills[r.fills.tag != "entry"]
    assert len(ex) == 1 and ex.iloc[0].ts == et(ed, "12:55")
    assert r.fills.quantity.sum() == 0


def test_conf_s019_benchmark_is_critical():
    sc = s019()
    sc = replace(sc, intraday={k: v for k, v in sc.intraday.items() if k[0] != "SPY"})
    r = run(registry.get("S019"), sc)
    assert r.status == RunStatus.NOT_TESTABLE and "SPY" in r.not_testable_reason


def test_finding_s023_signal_beyond_stop_z_is_skipped():
    """S023 sheet: entry 'z<=-2' (grid -1.5/-2/-2.5), exit 'stop at z=-3 (next-bar stops conservative)' (grid
    -2.5/-3/-3.5). The z-stop is evaluated at 5m closes after the entry (dynamic, not a fixed price) and the
    planned stop for DAY_RISK sizing / R is VWAP + stop_z x sd at the signal bar. A signal whose z is already at or
    beyond stop_z has no positive risk distance and is not traded: this is a DECLARED deviation, and the grid cell
    z=-2.5/stop_z=-2.5 (which can never trade) must be surfaced (fidelity flag + log), not read as a plain 0-trade
    OK. Same rule for S024 (z>=+3 signals)."""
    from tests.strategies.intraday._scenarios import s023
    cls = registry.get("S023")
    assert any("at/beyond stop_z are not traded" in d for d in cls.deviations)
    assert any("at/beyond stop_z are not traded" in d for d in registry.get("S024").deviations)
    sc = replace(s023(), check=None)
    r = run(cls, sc, z=-2.5, stop_z=-2.5)
    assert_status_ok(r)
    assert len(_entries(r)) == 0
    flags = [x for x in r.fidelity["reasons"] if "degenerate grid cell" in x["reason"]]
    assert flags and flags[0]["tier"] == "NA"
    assert r.fidelity["grade"] != "B"
    # a non-degenerate cell still trades the same pattern
    _first_entry_at(run(cls, sc), "10:05")
