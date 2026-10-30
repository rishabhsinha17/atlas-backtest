"""S161-S164 (SPX 0DTE) mechanics on SYNTHETIC minute data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.clock import et
from atlasbt.core.types import Fidelity, RunStatus
from atlasbt.strategies.options_events import S161, S162, S163, S164

from .oe_fix import (ev, fills, index_instrument, intraday_bars, make_portal, minute_snaps, option_book,
                     order_key, orders, run, sess)

DAY = pd.Timestamp("2024-03-13")            # Wednesday, SPXW PM expiry listed
PREV = pd.Timestamp("2024-03-12")
SIG_T0 = 0.15 * np.sqrt(6.0 / (365 * 24))   # sigma*sqrt(T) at 10:00 for 15% vol


def T(hhmm, day=DAY):
    return et(day, hhmm)


def spx_spec(spot_fn, vol_scale=lambda t: 1.0, expiry=DAY, settle="CASH_PM", root="SPXW", spy_fn=None,
             days=(PREV, DAY), price_override=None, fid=0):
    """SPX 1m bars + 0DTE SPXW chain whose prices keep sigma*sqrt(T) constant (x vol_scale(t))."""
    bars = {("SPX", "1m"): intraday_bars(days, spot_fn)}
    spy = spy_fn or (lambda t: spot_fn(t) / 10.0)
    bars[("SPY", "1m")] = intraday_bars(days, spy)
    snaps = minute_snaps([DAY])
    vol = lambda t, K, e, Ty: SIG_T0 * vol_scale(t) / np.sqrt(Ty)
    optq, insts = option_book("SPX", [(expiry, settle, "EUROPEAN", root)], np.arange(3900, 4101, 5.0), snaps,
                              spot_fn, vol, hs=lambda p: 0.05, price_override=price_override, fidelity=fid)
    insts["SPX"] = index_instrument("SPX")
    return {"bars": bars, "option_quotes": optq, "instruments": insts}


def flat(level=4000.0, jump_at=None, jump_to=None, day=DAY):
    def f(t):
        if jump_at is not None and t >= et(day, jump_at):
            return jump_to
        return level
    return f


def combo_orders(res, prefix):
    o = orders(res)
    return o[o["tag"].astype(str).str.startswith(prefix)]


def legs_of(res, tag_prefix):
    f = fills(res, tag_prefix)
    return sorted(zip(f["symbol"].str[12:13], (f["symbol"].str[13:].astype(int) / 1000.0), f["quantity"]))


# ---------------------------------------------------------------------------- S161
def test_synthetic_S161_entry_strikes_and_time_exit():
    spec = spx_spec(flat())
    r = run(S161(), spec, DAY, DAY)
    assert r.status == RunStatus.OK
    op = combo_orders(r, "S161:open")
    assert len(op) == 1 and pd.Timestamp(op["submitted_at"].iloc[0]) == T("10:00")
    f = fills(r, "S161:open")
    assert set(f["ts"]) == {T("10:01")}                      # first snapshot after the 10:00 decision
    legs = legs_of(r, "S161:open")
    n = int(abs(f["quantity"]).iloc[0])
    assert [(rt, k) for rt, k, _ in legs] == [("C", 4015.0), ("C", 4025.0), ("P", 3975.0), ("P", 3985.0)]
    assert {(rt, k): np.sign(q) for rt, k, q in legs} == {("C", 4015.0): -1, ("C", 4025.0): 1,
                                                         ("P", 3975.0): 1, ("P", 3985.0): -1}
    # OPT_CREDIT: n * max loss <= 0.5% NAV and n+1 would exceed it
    credit = -(f["quantity"] * f["price"]).sum() / n
    ml = (10.0 - credit) * 100.0
    assert n * ml <= 0.005 * 1e6 < (n + 1) * ml
    cl = combo_orders(r, "S161:close")
    assert list(cl["tag"]) == ["S161:close:time_exit"]
    assert pd.Timestamp(cl["submitted_at"].iloc[0]) == T("15:29")
    assert set(fills(r, "S161:close")["ts"]) == {T("15:30")}


@pytest.mark.parametrize("scenario,when,reason", [
    ("stop", "13:00", "stop"),                 # SPX jumps through the short call -> cost to close >= 2x credit
    ("tp", "12:00", "take_profit"),            # vol collapses -> >= 50% of the credit captured
])
def test_synthetic_S161_stop_and_take_profit(scenario, when, reason):
    if scenario == "stop":
        spec = spx_spec(flat(jump_at="13:00", jump_to=4030.0))
    else:
        spec = spx_spec(flat(), vol_scale=lambda t: 0.4 if t >= T("12:00") else 1.0)
    r = run(S161(), spec, DAY, DAY)
    cl = combo_orders(r, "S161:close")
    assert list(cl["tag"]) == [f"S161:close:{reason}"]
    assert pd.Timestamp(cl["submitted_at"].iloc[0]) == T(when)
    assert set(fills(r, "S161:close")["ts"]) == {T(when) + pd.Timedelta(minutes=1)}


def test_synthetic_S161_near_miss_no_0dte_expiry():
    spec = spx_spec(flat(), expiry=pd.Timestamp("2024-03-14"))      # next-day expiry only
    r = run(S161(), spec, DAY, DAY)
    assert r.status == RunStatus.OK and len(combo_orders(r, "S161")) == 0
    spec = spx_spec(flat(), settle="CASH_AM", root="SPX")           # AM-settled: not a PM 0DTE
    r = run(S161(), spec, DAY, DAY)
    assert len(combo_orders(r, "S161")) == 0


def test_S161_fidelity_gates():
    spec = spx_spec(flat())
    r = run(S161(), spec, DAY, DAY, option_fidelity=Fidelity.T2_PROXY)      # EOD-grade option data
    assert r.status == RunStatus.NOT_TESTABLE and "options" in r.not_testable_reason
    r = run(S161(allow_synthetic=True), spec, DAY, DAY, option_fidelity=Fidelity.T2_PROXY)
    assert r.status == RunStatus.OK and len(combo_orders(r, "S161:open")) == 1
    spec3 = spx_spec(flat(), fid=3)                                          # synthetic chain rows
    r = run(S161(), spec3, DAY, DAY)
    assert len(combo_orders(r, "S161")) == 0
    r = run(S161(allow_synthetic=True), spec3, DAY, DAY)
    assert len(combo_orders(r, "S161:open")) == 1
    assert any(x["tier"] == int(Fidelity.T3_MODELED) or x["tier"] == "T3_MODELED"
               for x in r.fidelity["reasons"])


# ---------------------------------------------------------------------------- S162
def test_synthetic_S162_trigger_and_near_miss():
    # SPX gaps up (open 4000 > prior close 3990) and SPY trades above its VWAP at 10:00
    def spx(t):
        return 3990.0 if t.tz_convert("America/New_York").normalize().tz_localize(None) == PREV else 4000.0

    def spy_up(t):
        if t.tz_convert("America/New_York").tz_localize(None).normalize() == PREV:
            return 399.0
        return 399.0 + (t - T("09:30")).total_seconds() / 3600.0      # rising -> close above VWAP
    spec = spx_spec(spx, spy_fn=spy_up)
    r = run(S162(), spec, DAY, DAY)
    op = combo_orders(r, "S162:open")
    assert len(op) == 1 and pd.Timestamp(op["submitted_at"].iloc[0]) == T("10:00")
    legs = legs_of(r, "S162:open")
    assert [(rt, k) for rt, k, _ in legs] == [("P", 3975.0), ("P", 3985.0)]
    assert set(fills(r, "S162:open")["ts"]) == {T("10:01")}
    cl = combo_orders(r, "S162:close")
    assert pd.Timestamp(cl["submitted_at"].iloc[0]) == T("15:30") and cl["tag"].iloc[0] == "S162:close:time_exit"
    # near miss 1: SPY below VWAP (falling)
    spy_dn = lambda t: 2 * 399.0 - spy_up(t) if t >= T("09:30") else 399.0
    r = run(S162(), spx_spec(spx, spy_fn=spy_dn), DAY, DAY)
    assert len(combo_orders(r, "S162")) == 0
    # near miss 2: no gap up (open == prior close)
    r = run(S162(), spx_spec(flat(), spy_fn=spy_up), DAY, DAY)
    assert len(combo_orders(r, "S162")) == 0


# ---------------------------------------------------------------------------- S163
def _breakout_spx(break_at="10:25", level=4000.0, to=4012.0):
    def f(t):
        # opening range 09:30-10:00 high = level + 2 (one spike), breakout above it at break_at
        if T("09:40") <= t < T("09:41"):
            return level + 2.0
        if t >= T(break_at):
            return to
        return level
    return f


def test_synthetic_S163_breakout_entry_and_vwap_exit():
    spy = lambda t: 400.0 + (0.5 if t >= T("10:25") else 0.0) - (2.0 if t >= T("13:05") else 0.0)
    spec = spx_spec(_breakout_spx(), spy_fn=spy)
    r = run(S163(), spec, DAY, DAY)
    op = combo_orders(r, "S163:open")
    assert len(op) == 1 and pd.Timestamp(op["submitted_at"].iloc[0]) == T("10:25")   # 5m bar 10:20-10:25 close
    legs = legs_of(r, "S163:open")
    assert [rt for rt, _, _ in legs] == ["C", "C"]
    (k_lo, q_lo), (k_hi, q_hi) = [(k, q) for _, k, q in legs]
    assert q_lo > 0 > q_hi and k_lo < k_hi                    # long .55-delta call, short .25-delta call
    assert set(fills(r, "S163:open")["ts"]) == {T("10:26")}
    cl = combo_orders(r, "S163:close")
    assert cl["tag"].iloc[0] == "S163:close:below_vwap"
    assert pd.Timestamp(cl["submitted_at"].iloc[0]) == T("13:05")
    # near miss: price stays inside the opening range
    r = run(S163(), spx_spec(_breakout_spx(to=4001.5), spy_fn=spy), DAY, DAY)
    assert len(combo_orders(r, "S163")) == 0
    # outside the 10:00-12:00 window
    r = run(S163(), spx_spec(_breakout_spx(break_at="12:10"), spy_fn=spy), DAY, DAY)
    assert len(combo_orders(r, "S163")) == 0


# ---------------------------------------------------------------------------- S164
def _fomc_spec(actual="14:00", scheduled="14:00", move=20.0):
    spot = lambda t: 4000.0 + (move if t >= T(actual) else 0.0)
    spec = spx_spec(spot)
    spec["events"] = [ev("FOMC_SCHEDULE", None, "2024-01-01 12:00", T(scheduled), type="STATEMENT"),
                      ev("FOMC", None, T(actual), T(actual), type="STATEMENT", scheduled=True)]
    return spec


def test_synthetic_S164_straddle_around_statement():
    r = run(S164(), _fomc_spec(), DAY, DAY)
    op = combo_orders(r, "S164:open")
    assert len(op) == 1 and pd.Timestamp(op["submitted_at"].iloc[0]) == T("13:55")
    legs = legs_of(r, "S164:open")
    assert [(rt, k, np.sign(q)) for rt, k, q in legs] == [("C", 4000.0, 1), ("P", 4000.0, 1)]
    assert set(fills(r, "S164:open")["ts"]) == {T("13:56")}
    cl = combo_orders(r, "S164:close")
    assert pd.Timestamp(cl["submitted_at"].iloc[0]) == T("14:30")
    assert set(fills(r, "S164:close")["ts"]) == {T("14:31")}
    # delayed statement: exit is timed from the ACTUAL release
    r = run(S164(exit_offset=15), _fomc_spec(actual="14:07"), DAY, DAY)
    cl = combo_orders(r, "S164:close")
    assert pd.Timestamp(cl["submitted_at"].iloc[0]) == T("14:22")


def test_synthetic_S164_near_miss_no_schedule_or_press_conference():
    spec = _fomc_spec()
    spec["events"] = [spec["events"][1]]                         # only the release itself (not known before)
    r = run(S164(), spec, DAY, DAY)
    assert r.status == RunStatus.NOT_TESTABLE                    # FOMC_SCHEDULE requirement
    spec = _fomc_spec()
    spec["events"][0] = ev("FOMC_SCHEDULE", None, "2024-01-01 12:00", T("14:30"), type="PRESS_CONFERENCE")
    spec["events"].append(ev("FOMC_SCHEDULE", None, "2024-01-01 12:00", T("14:00"), type="MINUTES"))
    r = run(S164(), spec, DAY, DAY)
    assert len(combo_orders(r, "S164")) == 0


# ---------------------------------------------------------------------------- lookahead + smoke
@pytest.mark.parametrize("cls,spec_fn,cut", [
    (S161, lambda: spx_spec(flat(jump_at="13:00", jump_to=4030.0)), "12:30"),
    (S162, lambda: spx_spec(lambda t: 3990.0 if t < T("09:30") else 4000.0,
                            spy_fn=lambda t: 399.0 + max((t - T("09:30")).total_seconds(), 0) / 3600.0), "11:00"),
    (S163, lambda: spx_spec(_breakout_spx(), spy_fn=lambda t: 400.0 - (2.0 if t >= T("13:05") else 0.0)), "11:00"),
    (S164, lambda: _fomc_spec(), "14:10"),
])
def test_lookahead_0dte(cls, spec_fn, cut):
    spec = spec_fn()
    full = run(cls(), spec, PREV, DAY)
    trunc = run(cls(), spec, PREV, DAY, cut=T(cut))
    a, b = order_key(orders(full), T(cut)), order_key(orders(trunc), T(cut))
    assert a == b and len(a) >= 1


@pytest.mark.parametrize("cls", [S161, S162, S163, S164])
def test_smoke_0dte(cls):
    spec = _fomc_spec()
    r = run(cls(), spec, PREV, DAY)
    assert r.status == RunStatus.OK
    assert r.diagnostics.get("lookahead_guard_violations", 0) == 0
