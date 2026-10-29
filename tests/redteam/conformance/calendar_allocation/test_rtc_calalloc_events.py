"""Red-team conformance probes: S038, S043, S044, S045 (event timing, PIT inputs, sizing caps).

SYNTHETIC data, mechanics only. Payload fields mirror what the project's data layer emits
(atlasbt/data/events.py, providers/sp500.py, docs/DATA.md).
"""
import numpy as np
import pandas as pd
import pytest

from atlasbt.core.types import Fidelity, RunStatus
from atlasbt.strategies import calendar_events as CE

from rtc_calalloc_helpers import (CAL, EventPortal, combine_daily, daily_frame, et_ts, ev, fills, flag_reasons,
                                  intraday_frame, orders, run, sess, sessions_before)


def factory(bars, events=(), universes=None, borrow=None):
    return EventPortal(bars=bars, events=list(events), universes=universes or {}, borrow=borrow)


# ======================================================================= S038
def _s038_bars(syms, a="2024-01-02", b="2024-04-30", px=50.0):
    ss = sess(a, b)
    return {(s, "1d"): daily_frame(ss, np.full(len(ss), px)) for s in syms}


def test_s038_post_hoc_consensus_is_not_used_silently():
    """FINDING: data layer (atlasbt/data/events.py::_attach_estimates) falls back to DoltHub eps_history
    'consensus recorded post hoc; revisions not PIT' with estimate_known_at = release - 1 min and
    estimate_fidelity = T2. S038._surprise only checks estimate_known_at < known_at, so this non-PIT
    consensus passes; the release itself is T1 (ZipLime 8-K acceptance time), so no T2 flag is raised.
    Sheet caveat: 'Point-in-time consensus revisions and exact release time'. Expected: skipped, or the run
    flagged T2 for the consensus."""
    syms = [f"E{i:02d}" for i in range(20)]
    bars = _s038_bars(syms)
    rel = "2024-03-05 08:00"
    k = pd.Timestamp(rel, tz="America/New_York").tz_convert("UTC")
    evs = [ev("EARNINGS", s, rel, fidelity=Fidelity.T1_BAR, eps_actual=1.0 + (i - 9.5) / 10.0, eps_estimate=1.0,
              estimate_known_at=k - pd.Timedelta(minutes=1), estimate_source="dolthub_eps_history",
              estimate_fidelity=int(Fidelity.T2_PROXY), timing="BMO")
           for i, s in enumerate(syms)]
    res = run(CE.S038(min_pool=10), factory(bars, evs, {"SP500": syms}), "2024-03-04", "2024-03-08")
    traded = (orders(res)["tag"] == "entry").any() if len(orders(res)) else False
    flagged = any("consensus" in r.lower() or "eps_estimate" in r.lower() for r in flag_reasons(res))
    assert (not traded) or flagged, f"non-PIT consensus traded without a fidelity flag: {flag_reasons(res)}"


def test_s038_release_0915_trades_same_open_0916_next_open():
    """'enter first RTH open >= 15m after disseminated release' (inclusive at 15 min)."""
    syms = [f"F{i:02d}" for i in range(20)]
    bars = _s038_bars(syms)
    est = pd.Timestamp("2024-02-20", tz="UTC")
    evs = []
    for i, s in enumerate(syms):
        t = "2024-03-05 09:15" if s == "F19" else ("2024-03-05 09:16" if s == "F00" else "2024-03-04 18:00")
        evs.append(ev("EARNINGS", s, t, eps_actual=1.0 + (i - 9.5) / 10.0, eps_estimate=1.0,
                      estimate_known_at=est, timing="BMO"))
    res = run(CE.S038(min_pool=10), factory(bars, evs, {"SP500": syms}), "2024-03-04", "2024-03-08")
    o = orders(res)
    e = o[o["tag"] == "entry"].set_index("symbol")
    f = fills(res).set_index("order_id")
    open5, open6 = CAL.open_close(pd.Timestamp("2024-03-05"))[0], CAL.open_close(pd.Timestamp("2024-03-06"))[0]
    assert pd.Timestamp(f.loc[e.loc["F19", "order_id"], "ts"]) == open5      # 09:15 -> 09:30 same day
    assert pd.Timestamp(f.loc[e.loc["F00", "order_id"], "ts"]) == open6      # 09:16 -> next open


def test_s038_cross_section_long_basket_capped_at_50pct_nav():
    """FINDING (sizing): CROSS_SECTION = long basket +50% NAV. S038._enter caps each new name with
    room = 50% NAV - FILLED same-side notional; entries submitted in the same step are pending, not
    filled, so a large same-morning batch overshoots the 50% basket."""
    syms = [f"G{i:03d}" for i in range(200)]
    bars = _s038_bars(syms)
    est = pd.Timestamp("2024-02-20", tz="UTC")
    evs = [ev("EARNINGS", s, "2024-03-05 08:00", eps_actual=1.0 + (i - 99.5) / 100.0, eps_estimate=1.0,
              estimate_known_at=est, timing="BMO") for i, s in enumerate(syms)]
    res = run(CE.S038(min_pool=20, pct=0.20), factory(bars, evs, {"SP500": syms}), "2024-03-05", "2024-03-06")
    f = fills(res)
    nav0 = 1_000_000.0
    long_notional = float((f["quantity"].clip(lower=0) * f["price"]).sum())
    assert long_notional <= 0.50 * nav0 * 1.001, f"long basket {long_notional / nav0:.1%} of NAV > 50%"


# ======================================================================= S043 / S044
def _index_bars(sym, win, base=50.0, beta=1.0, seed=3):
    hs = sessions_before(win[0], 300)
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, len(hs))
    spy = 400.0 * np.cumprod(1 + r)
    stk = base * np.cumprod(1 + beta * r)
    spy, stk = spy / spy[-1] * 400.0, stk / stk[-1] * base
    i_s = intraday_frame({s: (base, {}) for s in win}, "5m")
    i_spy = intraday_frame({s: (400.0, {}) for s in win}, "5m")
    return {(sym, "5m"): i_s, (sym, "1d"): combine_daily(daily_frame(hs, stk), i_s),
            ("SPY", "5m"): i_spy, ("SPY", "1d"): combine_daily(daily_frame(hs, spy), i_spy)}


@pytest.mark.parametrize("sid,kind", [("S043", "INDEX_ADD"), ("S044", "INDEX_DEL")])
def test_s043_s044_skip_events_without_a_real_announcement_timestamp(sid, kind):
    """FINDING: providers/sp500.py / docs/DATA.md §4.5: only 623 of 1,544 S&P events have a parsed
    announcement date; the rest carry announce_source='fallback_prev_session', known_at = 17:15 ET of the
    session BEFORE the effective date ('announcement-drift strategies are only testable on the 623').
    S043/S044 ignore announce_source and trade the fabricated timestamp: entry at the effective-date open,
    exit 15:50 the same day - an effective-day trade, not the sheet's announcement-drift trade."""
    win = sess("2024-03-15", "2024-03-18")
    eff = pd.Timestamp("2024-03-18")
    sym = "NEWCO" if kind == "INDEX_ADD" else "OLDCO"
    bars = _index_bars(sym, win)
    e = ev(kind, sym, "2024-03-15 17:15", eff=CAL.open_close(eff)[0], index="SP500", effective_date=eff,
           announce_source="fallback_prev_session")
    cls = CE.S043 if sid == "S043" else CE.S044
    res = run(cls(), factory(bars, [e]), win[0], win[-1])
    o = orders(res)
    assert not (len(o) and (o["tag"] == "entry").any()), \
        f"{sid} traded an event whose announcement time is a fallback (effective-day trade)"


def test_s043_batch_of_additions_respects_100pct_gross():
    """FINDING (sizing, MINOR): S043 hand-rolls SWING_NOTIONAL (not atlasbt.risk.sizing) and its declared
    'Gross (stock + hedge) <= 100% NAV' uses equity_gross(ctx) = FILLED positions only; six additions
    announced the same evening are all sized at 10% + 10% beta hedge -> 120% gross."""
    win = sess("2024-03-04", "2024-03-06")
    eff = pd.Timestamp("2024-03-06")
    syms = [f"AD{i}" for i in range(6)]
    bars, evs = {}, []
    for s in syms:
        b = _index_bars(s, win, base=50.0, beta=1.0)
        bars.update(b)
        evs.append(ev("INDEX_ADD", s, "2024-03-01 17:15", eff=CAL.open_close(eff)[0], index="SP500",
                      effective_date=eff, announce_source="sp_press_release_citation"))
    res = run(CE.S043(), factory(bars, evs), win[0], win[-1])
    f = fills(res)
    t_open = CAL.open_close(win[0])[0]
    f0 = f[pd.DatetimeIndex(f["ts"]) == t_open]
    gross = float((f0["quantity"].abs() * f0["price"]).sum())
    assert gross <= 1.0 * 1_000_000.0 * 1.001, f"gross at entry {gross / 1e6:.0%} of NAV"


# ======================================================================= S045
def _s045_portal():
    ss = sess("2023-12-01", "2024-04-05")
    closes = np.full(len(ss), 40.0)
    opens = closes.copy()
    highs, lows = closes + 1.0, closes - 1.0
    bars = {("CCC", "1d"): daily_frame(ss, closes, opens=opens, highs=highs, lows=lows)}
    evs = [ev("BUYBACK", "CCC", "2024-03-04 08:30", amount_usd=1e9, pct_mcap=0.06, with_earnings=False)]
    return factory(bars, evs, {"SP500": ["CCC"]})


def test_s045_exit_after_10_trading_sessions():
    """FINDING (MINOR): 'Exit after 10 trading sessions'. Entry fills at the OPEN of E (a full session);
    S045 sells MOC at the close of E+10, i.e. 11 sessions held (E..E+10). S038 (same module) counts
    'after 20 trading sessions' as exit at the open of E+20 (exactly 20). Expected exit no later than the
    open of E+10."""
    res = run(CE.S045(), _s045_portal(), "2024-03-01", "2024-04-05")
    o = orders(res)
    f = fills(res).set_index("order_id")
    ent = o[o["tag"] == "entry"].iloc[0]
    ex = o[o["tag"].isin(["time_exit"])].iloc[0]
    e_ts = pd.Timestamp(f.loc[ent["order_id"], "ts"])
    x_ts = pd.Timestamp(f.loc[ex["order_id"], "ts"])
    e_sess = CAL.sessions(e_ts.tz_convert("America/New_York").tz_localize(None).normalize(),
                          x_ts.tz_convert("America/New_York").tz_localize(None).normalize())
    held_sessions = len(e_sess) if x_ts == CAL.open_close(e_sess[-1])[1] else len(e_sess) - 1
    assert held_sessions <= 10, f"held {held_sessions} sessions"
