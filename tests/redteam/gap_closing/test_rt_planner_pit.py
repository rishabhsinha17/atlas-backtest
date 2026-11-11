"""Red team (gap closing): planner (atlasbt/run/plan.py, 82bd563) vs the engine (Backtester).

Each test builds a tiny synthetic portal, asks the planner for a window, then runs the REAL engine checks
(Backtester._check_requirements + the per-session universe coverage rule _coverage_check, which the
strategy triggers every session it asks for the universe) on exactly that window. Every test asserts the
correct contract: "planner says testable at tier X" implies "the engine accepts the run at tier X".
A failure is a finding.
"""
from __future__ import annotations

import pandas as pd

from atlasbt.core.clock import nyse
from atlasbt.core.types import Fidelity
from atlasbt.engine import Backtester, BacktestConfig
from atlasbt.engine.strategy import Requirement, Strategy
from atlasbt.run.plan import PIT_STEP_SESSIONS, PlanContext, _plan_instance

from tests.run.test_plan_intraday import ALL, END, ET, TOP, _IInst, _IPortal, _record_windows, _sessions
from tests.run.test_plan_star import _row

CAL = nyse()


class _EPortal(_IPortal):
    """_IPortal + the windowed has() the engine uses for member verification, quote fidelity by source
    (Portal.QUOTE_FIDELITY: alpaca T0, databento/iex T2) and a quotes() reader for the engine's probe."""

    def has(self, sym, kind, freq=None, sessions=None, *, start=None, end=None, as_of=None, **kw):
        if kind == "bars" and sym != "*" and (start is not None or end is not None):
            days = self._days(sym, freq)
            s = pd.Timestamp(start).tz_convert(ET).tz_localize(None).normalize()
            e = pd.Timestamp(end).tz_convert(ET).tz_localize(None).normalize()
            return bool(((days >= s) & (days <= e)).any())
        return super().has(sym, kind, freq, sessions)

    def fidelity_of(self, sym, kind, freq=None, sessions=None):
        if kind == "quotes":
            if self.store.exists("quotes", "alpaca", sym):
                return Fidelity.T0_QUOTE
            if self.store.exists("quotes", "databento", sym):
                return Fidelity.T2_PROXY
            return Fidelity.NA
        return super().fidelity_of(sym, kind, freq, sessions)

    def quotes(self, sym, start, end=None, as_of=None):
        for src in ("alpaca", "databento"):
            if self.store.exists("quotes", src, sym):
                q = self.store.read("quotes", src, sym)
                idx = pd.DatetimeIndex(q.index)
                idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
                q.index = idx
                return q[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
        return pd.DataFrame()


def _engine(port, requires, start, end, universe_name=None, universe=(), base="1m", th=0.95):
    """(accepted, reason, t2_flagged) of the real engine checks on [start, end]: requirements at the first
    session, then the universe coverage rule at every session (intraday strategies query the universe each
    session: IntradayStrategy.symbols)."""
    S = type("RT", (Strategy,), {"id": "RT", "bar_freq": base, "universe": list(universe),
                                 "universe_name": universe_name, "requires": list(requires)})
    bt = Backtester(S(), port, BacktestConfig(min_universe_coverage=th))
    sess = CAL.sessions(pd.Timestamp(start), pd.Timestamp(end))
    bt._run_window = (CAL.open_close(sess[0])[0], CAL.open_close(sess[-1])[1])
    ok, why, _f = bt._check_requirements(sess)
    if not ok:
        return False, why, bool(bt._cov_flagged)
    if universe_name:
        for d in sess:
            o = CAL.open_close(d)[0]
            bt._coverage_check(universe_name, o, port.universe(universe_name, o), raise_=False)
            if bt._coverage_fail:
                return False, bt._coverage_fail, True
    return True, "", bool(bt._cov_flagged)


def _plan(port, requires, tmp_path, universe_name=None, universe=(), th=0.95, tag="r"):
    inst = _IInst(requires, universe=list(universe), universe_name=universe_name)
    return _plan_instance("T", inst, "1m", PlanContext(port, tmp_path / tag, min_coverage=th), _row())


# ------------------------------------------------------------------------------------------------ 1
def test_rt_pit_dip_between_samples_planner_testable_engine_refuses(tmp_path):
    """Three sessions strictly between two 10-session samples have a member (X) without any 1m bars:
    the engine's coverage there is 2/3 < 95 % and it refuses; the planner never looks at them."""
    sess = _sessions("2016-01-01", END)
    base = 1000                                 # sampled index (multiple of PIT_STEP_SESSIONS)
    assert base % PIT_STEP_SESSIONS == 0
    dip = set(sess[base + 3: base + 6])

    def members(d):
        return ["A", "B", "X"] if d in dip else ["A", "B", "C"]

    port = _EPortal(tmp_path, spans={s: ALL for s in "ABC"}, daily={s: "2010-01-04" for s in "ABCX"},
                    members=members)
    reqs = [Requirement("bars", "1m")]
    r = _plan(port, reqs, tmp_path, universe_name=TOP)
    if r["testable"] != "Y":
        return                                  # planner refuses: consistent with the engine
    ok, why, _ = _engine(port, reqs, r["start"], r["end"], universe_name=TOP)
    assert ok, (f"planner: testable {r['start']}..{r['end']}; engine refuses: {why}")


# ------------------------------------------------------------------------------------------------ 2
def _late_member_spans(m24_spans):
    mem = [f"M{i:02d}" for i in range(25)]
    spans = {m: ALL for m in mem[:24]}
    spans["M24"] = m24_spans
    spans["SPY"] = [("2020-01-02", END)]        # an explicit-symbol input that starts later
    return mem, spans


def test_rt_pit_probe_checked_at_pit_start_not_at_final_start(tmp_path):
    """pit_window verifies the engine's PRE probe at ITS start (2016-01-04). Another requirement (SPY 1m
    from 2020-01-02) moves the run start; at that start M24 is kept (it has bars later in the window) but
    has no bars in the first 10 sessions, so the engine's probe refuses the run the planner called testable."""
    mem, spans = _late_member_spans([("2016-01-04", "2016-06-30"), ("2020-02-18", END)])
    port = _EPortal(tmp_path, spans=spans, daily={m: "2010-01-04" for m in mem + ["SPY"]}, members=lambda d: mem)
    reqs = [Requirement("bars", "1m", sessions=("PRE", "RTH")), Requirement("bars", "1m", symbols=("SPY",))]
    r = _plan(port, reqs, tmp_path, universe_name=TOP)
    # (verifier) the start may move past 2020-01-02: no start before M24's bars resume passes the probe
    assert r["testable"] == "Y" and r["start"] >= "2020-01-02", r
    ok, why, _ = _engine(port, reqs, r["start"], r["end"], universe_name=TOP)
    assert ok, f"planner: testable {r['start']}..{r['end']}; engine refuses: {why}"
    # and the start is not later than needed: the engine refuses one session earlier
    if r["start"] > "2020-01-02":
        prev = str(CAL.sessions(pd.Timestamp("2020-01-02"), pd.Timestamp(r["start"]))[-2].date())
        ok_prev, why_prev, _ = _engine(port, reqs, prev, r["end"], universe_name=TOP)
        assert not ok_prev and "PRE" in why_prev, (prev, why_prev)


def test_rt_pit_engine_coverage_tier_uses_pit_window_not_final_window(tmp_path):
    """engine_cov (T2 tiering) is computed on the PIT window [2016-01-04, END]; the final window starts
    2020-01-02 (SPY). M24 has 1m bars only in 2016, so the engine drops it in the final window and flags the
    run T2_PROXY (coverage 96 %), while the planner promises T1_BAR."""
    mem, spans = _late_member_spans([("2016-01-04", "2016-06-30")])
    port = _EPortal(tmp_path, spans=spans, daily={m: "2010-01-04" for m in mem + ["SPY"]}, members=lambda d: mem)
    reqs = [Requirement("bars", "1m"), Requirement("bars", "1m", symbols=("SPY",))]
    r = _plan(port, reqs, tmp_path, universe_name=TOP)
    assert r["testable"] == "Y" and r["start"] == "2020-01-02", r
    ok, why, t2 = _engine(port, reqs, r["start"], r["end"], universe_name=TOP)
    assert ok, why
    assert t2, "engine accepted without a coverage flag"         # the engine DOES flag T2_PROXY
    assert r["expected_fidelity"] == "T2_PROXY", (
        f"planner expected {r['expected_fidelity']} but the engine flags the run T2_PROXY "
        f"(universe coverage < 100 % in {r['start']}..{r['end']})")


# ------------------------------------------------------------------------------------------------ 3
def test_rt_quotes_min_fidelity_ignored_planner_testable_engine_refuses(tmp_path):
    """S171 shape: static universe + Requirement('quotes', min_fidelity=T0_QUOTE). The only stock quotes are
    Databento equity quotes (Portal.QUOTE_FIDELITY T2_PROXY). The engine refuses ('fidelity T2_PROXY, need
    T0_QUOTE'); the new quotes branch of the planner never compares the tier with min_fidelity."""
    port = _EPortal(tmp_path, spans={"SPY": ALL, "QQQ": ALL})
    days = _sessions("2024-01-02", "2024-03-28")
    for s in ("SPY", "QQQ"):
        q = pd.DataFrame({"bid": 10.0, "ask": 10.01, "bid_size": 1.0, "ask_size": 1.0},
                         index=pd.DatetimeIndex([pd.Timestamp(f"{d.date()} 12:00", tz=ET) for d in days],
                                                name="ts").tz_convert("UTC"))
        port.store.write("quotes", "databento", s, q, partition="year")
    reqs = [Requirement("bars", "1m"), Requirement("quotes", min_fidelity=Fidelity.T0_QUOTE)]
    r = _plan(port, reqs, tmp_path, universe=("SPY", "QQQ"))
    ok, why, _ = _engine(port, reqs, r["start"] or "2024-01-02", r["end"] or "2024-03-28", universe=("SPY", "QQQ"))
    assert not ok and "T0_QUOTE" in why                         # the engine's verdict
    assert r["testable"] == "N", (f"planner: testable {r['start']}..{r['end']} at {r.get('expected_fidelity')}; "
                                  f"engine: {why}")


# ------------------------------------------------------------------------------------------------ 4
def test_rt_pit_pre_quote_probe_accepts_rth_only_quote_sessions(tmp_path):
    """PIT universe with Requirement('quotes', sessions=('PRE',)): the members' recorded Alpaca windows are
    close_m10 (15:50 ET, RTH) only. The planner's probe counts any quote session; the engine's probe
    classifies the quote stamps and finds no PRE quote, so it refuses."""
    mem = ["A", "B", "C"]
    port = _EPortal(tmp_path, spans={s: ALL for s in mem}, daily={s: "2010-01-04" for s in mem},
                    members=lambda d: mem)
    for s in mem:
        _record_windows(port.store, s, _sessions("2024-01-02", "2024-06-28"), window="close_m10")
    reqs = [Requirement("bars", "1m"), Requirement("quotes", None, sessions=("PRE",))]
    r = _plan(port, reqs, tmp_path, universe_name=TOP)
    # (verifier) no PRE quote exists anywhere, so the correct planner answer is NOT_TESTABLE; the original
    # premise `testable == "Y"` encoded the defect. Contract in both directions:
    if r["testable"] == "Y":
        ok, why, _ = _engine(port, reqs, r["start"], r["end"], universe_name=TOP)
        assert ok, f"planner: testable {r['start']}..{r['end']}; engine refuses: {why}"
    else:
        ok, why, _ = _engine(port, reqs, "2024-01-02", "2024-06-28", universe_name=TOP)
        assert not ok and "PRE" in why, why
    # positive control: recorded pre-open windows -> the planner says testable and the engine accepts
    port2 = _EPortal(tmp_path / "pre", spans={s: ALL for s in mem}, daily={s: "2010-01-04" for s in mem},
                     members=lambda d: mem)
    for s in mem:
        _record_windows(port2.store, s, _sessions("2024-01-02", "2024-06-28"), window="pre_0920")
    r2 = _plan(port2, reqs, tmp_path, universe_name=TOP, tag="pre")
    assert r2["testable"] == "Y", r2["reason"]
    ok2, why2, _ = _engine(port2, reqs, r2["start"], r2["end"], universe_name=TOP)
    assert ok2, f"planner: testable {r2['start']}..{r2['end']}; engine refuses: {why2}"
