"""Per-strategy SYNTHETIC pattern tests (mechanics only) for S001-S030, S116:

* trigger: exact bar of the signal (entry order submitted at that bar's close),
  exact fill bar (next bar open, zero-cost config), stop and target levels;
* near miss: the same pattern with one condition broken -> no entry order;
* lookahead: data truncated just after the signal vs full data -> identical
  orders up to the cut.
"""
from __future__ import annotations

import pandas as pd
import pytest

from atlasbt.core.clock import et
from atlasbt.engine.config import BacktestConfig
from atlasbt.execution.constraints import round_limit_for_side, round_stop_for_side
from atlasbt.strategies import registry
import atlasbt.strategies.intraday as fam

from ._scenarios import PAIRS, s004_vwap_exit
from ._synth import DAY, assert_status_ok, bar_open_at, orders, run

IDS = list(fam.IDS)


def _pair(sid):
    p = PAIRS[sid]()
    return p


def _tick_round_stop(px, side):
    return round_stop_for_side(px, 0.01, -side > 0)       # exit stop: sell stop for a long


def _tick_round_limit(px, side):
    return round_limit_for_side(px, 0.01, -side > 0)      # exit limit: sell (long) rounds up, buy rounds down


def _check_trigger(sid, sc):
    cls = registry.get(sid)
    r = run(cls, sc)
    assert_status_ok(r)
    e = orders(r, "entry")
    assert len(e) >= 1, f"{sid}: no entry; logs={r.diagnostics.get('logs', [])[-5:]}"
    e0 = e.iloc[0]
    sig = et(DAY, sc.signal)
    assert e0.symbol == sc.symbol
    assert e0.submitted_at == sig, (sid, e0.submitted_at, sig)
    assert (e0.quantity > 0) == (sc.side > 0)
    f = r.fills[r.fills.tag == "entry"]
    assert len(f) >= 1
    f0 = f.iloc[0]
    assert f0.ts == sig                                 # next bar (starting at the signal bar's end) ...
    assert f0.price == pytest.approx(bar_open_at(sc, sc.symbol, sig))   # ... at its open, never the signal close
    if sc.stop is not None:
        st = orders(r, "stop")
        assert len(st) >= 1, f"{sid}: no protective stop"
        assert st.iloc[0].stop_price == pytest.approx(_tick_round_stop(sc.stop, sc.side), abs=1e-9), \
            (sid, st.iloc[0].stop_price, sc.stop)
    if sc.target is not None:
        t = orders(r, "target")
        assert len(t) >= 1 and t.iloc[0].limit_price == pytest.approx(_tick_round_limit(sc.target, sc.side))
    if sc.target_r is not None:
        t = orders(r, "target")
        assert len(t) >= 1, f"{sid}: no R target"
        fill = f0.price
        exp = fill + sc.target_r * (fill - sc.stop)
        assert t.iloc[0].limit_price == pytest.approx(_tick_round_limit(exp, sc.side), abs=1e-9), \
            (sid, t.iloc[0].limit_price, exp)
        assert t.iloc[0].oco_group == orders(r, "stop").iloc[0].oco_group
    if sc.check is not None:
        sc.check(r, sc)
    # one entry per symbol at a time / day rules: nothing else was entered
    assert len(e) == 1, f"{sid}: unexpected extra entries {e[['submitted_at', 'quantity']].values.tolist()}"
    return r


@pytest.mark.parametrize("sid", IDS)
def test_synthetic_trigger(sid):
    trig, _ = _pair(sid)
    _check_trigger(sid, trig)


@pytest.mark.parametrize("sid", IDS)
def test_synthetic_near_miss(sid):
    _, near = _pair(sid)
    r = run(registry.get(sid), near)
    assert_status_ok(r)
    assert len(orders(r, "entry")) == 0, f"{sid}: near miss triggered"


@pytest.mark.parametrize("sid", IDS)
def test_lookahead_truncation(sid):
    trig, _ = _pair(sid)
    cls = registry.get(sid)
    cut = et(DAY, trig.signal) + pd.Timedelta(minutes=10)
    cfg = BacktestConfig(use_quotes="never")
    full = run(cls, trig, config=cfg)
    part = run(cls, trig, cut=cut, config=cfg)
    assert_status_ok(full)
    assert_status_ok(part)
    # OCO members' quantities are mutated later by reduce-on-fill -> compare quantities of stand-alone orders only
    cols = ["submitted_at", "symbol", "type", "limit_price", "stop_price", "tag", "parent_qty"]
    of, op = orders(full).copy(), orders(part).copy()
    for o in (of, op):
        o["parent_qty"] = o["quantity"].where(o["parent_id"].isna() & o["oco_group"].isna(), 0.0)
    a = of[of.submitted_at <= cut][cols].reset_index(drop=True)
    b = op[op.submitted_at <= cut][cols].reset_index(drop=True)
    assert len(a) >= 1
    pd.testing.assert_frame_equal(a, b)


# ----------------------------------------------------------------- extra mechanics
def test_s004_two_closes_below_vwap_exit():
    _check_trigger("S004", s004_vwap_exit())


def test_s001_variants_atr_stop_and_r_target():
    from ._scenarios import _atr_expected, s001
    trig, _ = s001()
    sc = trig
    df = sc.intraday[("XYZ", "1m")]
    a = _atr_expected(df, et(DAY, "09:34"), k_min=5)      # complete 5m buckets at 09:37: 09:30-09:34 only
    from dataclasses import replace
    _check_trigger("S001", replace(sc, params={"stop": "atr5m", "target_r": 2.0}, stop=100.6 - a, target_r=2.0))


def test_s005_s006_quotes_are_critical_unless_proxy():
    from dataclasses import replace
    from atlasbt.core.types import RunStatus
    from ._scenarios import s005, s006
    for fn, sid in ((s005, "S005"), (s006, "S006")):
        trig, _ = fn()
        noq = replace(trig, quotes={})
        r = run(registry.get(sid), noq)
        assert r.status == RunStatus.NOT_TESTABLE and "quotes" in r.not_testable_reason
        r2 = _check_trigger(sid, replace(noq, params={"pm_mid_proxy": True}))
        assert any("proxied" in x["reason"] for x in r2.fidelity["reasons"])


def test_s015_short_reversal_and_quote_requirement():
    from dataclasses import replace
    from atlasbt.core.types import RunStatus
    from ._scenarios import s015
    _check_trigger("S015", s015(side=-1))
    noq = s015(quotes=False)
    assert run(registry.get("S015"), noq).status == RunStatus.NOT_TESTABLE
    r = _check_trigger("S015", replace(noq, params={"mid_proxy": True}))
    assert any("proxied" in x["reason"] for x in r.fidelity["reasons"])


def test_s030_no_entry_while_halted():
    from ._scenarios import s030
    r = run(registry.get("S030"), s030(halt=True))
    assert_status_ok(r)
    assert len(orders(r, "entry")) == 0


def test_s019_requires_benchmark_and_ranks_cross_section():
    from ._scenarios import s019
    r = _check_trigger("S019", s019())
    assert set(orders(r, "entry").symbol) == {"N0"}
