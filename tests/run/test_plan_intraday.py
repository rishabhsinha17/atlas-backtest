"""Planner changes for the Alpaca intraday data (docs/DATA.md §4.15, "Planner changes needed for intraday"):

1. intraday bar ranges reach the Alpaca history start (2016-01-04), not ``now - 800 days``;
2. quote requirements with enough quote sessions give the window (first/last quote session) instead of
   always NOT_TESTABLE; a PIT universe uses the UNION of its members' quote sessions;
3. Alpaca quote sessions come from ``reference/alpaca/quote_windows``, not the members' quotes parquet;
4. PIT intraday universes are dated by the per-session coverage of THAT session's members (the engine's
   coverage rule at every session it queries the universe), plus the engine's first-10-session PRE probe;
5. memoised intraday facts in plan_cache.json are dropped when intraday bars / quotes change.
"""
import pandas as pd

from atlasbt.core.clock import nyse
from atlasbt.core.types import Fidelity
from atlasbt.data.store import ParquetStore
from atlasbt.engine.strategy import Requirement
from atlasbt.run.plan import PlanContext, _plan_instance

from .test_plan_star import _row

END = "2026-10-30"
ET = "America/New_York"
CAL = nyse()
TOP = "SP500_TOP3_DV"


def _sessions(a, b) -> pd.DatetimeIndex:
    return CAL.sessions(pd.Timestamp(a), pd.Timestamp(b))


def _iso(idx) -> list:
    return [str(d.date()) for d in idx]


class _IPortal:
    """Synthetic portal: daily bars from ``daily[sym]`` to END; intraday bars ONLY on the session spans
    ``spans[sym]`` (Alpaca stores a stock's 1m bars around its top-60 sessions, not its whole history);
    PIT universes ``SP500_TOP3_DV`` / ``SP500`` = ``members(date)``; quotes in a real ParquetStore."""

    def __init__(self, root, spans=None, daily=None, members=None, static=None):
        self.store = ParquetStore(root / "store")
        self.spans = {k: [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in v] for k, v in (spans or {}).items()}
        self.daily = {"SPY": pd.Timestamp("1993-01-29")}
        self.daily.update({k: pd.Timestamp(v) for k, v in (daily or {}).items()})
        self.members = members or (lambda d: [])
        self.static = static or {}

    def _days(self, sym, freq) -> pd.DatetimeIndex:
        if (freq or "1d") == "1d":
            f = self.daily.get(sym)
            return _sessions(f, END) if f is not None else pd.DatetimeIndex([])
        parts = [_sessions(a, b) for a, b in self.spans.get(sym, [])]
        return parts[0].append(parts[1:]) if parts else pd.DatetimeIndex([])

    def universe(self, name, as_of, require_bars=False):
        n = name.upper()
        if n in (TOP, "SP500"):
            d = pd.Timestamp(as_of).tz_convert(ET).tz_localize(None).normalize()
            return list(self.members(d))
        return list(self.static.get(n, []))

    def universe_survivorship_bias(self, name, as_of):
        m = self.universe(name, as_of)
        return {"members": len(m), "with_data": len(m), "coverage": 1.0, "missing": []}

    def has(self, sym, kind, freq=None, sessions=None, **kw):
        if kind == "bars":
            return True if sym == "*" else len(self._days(sym, freq)) > 0
        if kind == "quotes":
            return any(self.store.exists("quotes", s, sym) for s in ("alpaca", "databento"))
        return False

    def fidelity_of(self, sym, kind, freq=None, sessions=None):
        if not self.has(sym, kind, freq):
            return Fidelity.NA
        return Fidelity.T0_QUOTE if kind == "quotes" else Fidelity.T1_BAR

    def bars(self, sym, freq, start, end, as_of, adjusted=False, sessions=("RTH",)):
        days = self._days(sym, freq)
        off = pd.Timedelta(hours=16) if (freq or "1d") == "1d" else pd.Timedelta(hours=9, minutes=30)
        ts = (days + off).tz_localize(ET).tz_convert("UTC")
        ts = ts[(ts >= pd.Timestamp(start)) & (ts < pd.Timestamp(end))]
        return pd.DataFrame({"close": 1.0}, index=ts)

    def bars_coverage(self, sym, start, end, as_of, freq="1d"):
        b = self.bars(sym, freq, start, end, as_of)
        if not len(b):
            return {"available": False}
        return {"available": True, "first": b.index[0], "last": b.index[-1]}


class _IInst:
    bar_freq = "1m"
    sizing = None
    asset = "STK"
    flat_by = None
    runtime_universes = ()

    def __init__(self, requires, universe=(), universe_name=None):
        self.requires = list(requires)
        self.universe = list(universe)
        self.universe_name = universe_name


def _record_windows(store, sym, days, window="close_m10", n=1, write_quotes=True):
    """Record Alpaca quote windows [15:50, 15:51] ET (pre_0920: [09:20, 09:30]) of ``sym`` on ``days``."""
    a, b = ("09:20", "09:30") if window == "pre_0920" else ("15:50", "15:51")
    w0 = pd.DatetimeIndex([pd.Timestamp(f"{d.date()} {a}", tz=ET) for d in days]).tz_convert("UTC")
    w1 = pd.DatetimeIndex([pd.Timestamp(f"{d.date()} {b}", tz=ET) for d in days]).tz_convert("UTC")
    store.write("reference", "alpaca", "quote_windows",
                pd.DataFrame({"symbol": sym, "w0": w0, "w1": w1, "window": window, "n": n}), key=["symbol", "w0"])
    if write_quotes and n:
        q = pd.DataFrame({"bid": 10.0, "ask": 10.01, "bid_size": 100.0, "ask_size": 100.0},
                         index=pd.DatetimeIndex(w0 + pd.Timedelta(seconds=30), name="ts"))
        store.write("quotes", "alpaca", sym, q, partition="year")


ALL = [("2016-01-04", END)]


# ------------------------------------------------------------------------------------------- item 1
def test_intraday_bar_range_reaches_the_alpaca_history_start(tmp_path):
    port = _IPortal(tmp_path, spans={"SPY": ALL})
    pc = PlanContext(port, tmp_path / "res")
    assert pc.bar_range("SPY", "1m") == ["2016-01-04", END]
    r = _plan_instance("T", _IInst([Requirement("bars", "1m")], universe=["SPY"]), "1m", pc, _row())
    assert r["testable"] == "Y" and (r["start"], r["end"]) == ("2016-01-04", END), r


# ------------------------------------------------------------------------------------------- item 2
def test_quotes_window_from_common_quote_sessions(tmp_path):
    port = _IPortal(tmp_path, spans={"SPY": ALL, "QQQ": ALL})
    _record_windows(port.store, "SPY", _sessions("2024-01-02", "2024-06-28"))
    _record_windows(port.store, "QQQ", _sessions("2024-02-01", "2024-09-30"))
    reqs = [Requirement("bars", "1m"), Requirement("quotes", None, note="quote midpoint")]
    r = _plan_instance("T", _IInst(reqs, universe=["SPY", "QQQ"]), "1m", PlanContext(port, tmp_path / "r"), _row())
    assert r["testable"] == "Y", r["reason"]
    assert (r["start"], r["end"]) == ("2024-02-01", "2024-06-28"), r
    assert "quote" in r["binding_constraint"]


def test_quotes_below_minimum_stay_not_testable(tmp_path):
    port = _IPortal(tmp_path, spans={"SPY": ALL, "QQQ": ALL})
    _record_windows(port.store, "SPY", _sessions("2024-01-02", "2024-06-28"))
    _record_windows(port.store, "QQQ", _sessions("2024-06-24", "2024-06-28"))
    reqs = [Requirement("bars", "1m"), Requirement("quotes", None)]
    r = _plan_instance("T", _IInst(reqs, universe=["SPY", "QQQ"]), "1m", PlanContext(port, tmp_path / "r"), _row())
    assert r["testable"] == "N" and "5 common session" in r["reason"], r


def _rotating(d):
    return ["A", "B", "C"] if d < pd.Timestamp("2024-04-01") else ["D", "E", "F"]


def test_pit_universe_uses_the_union_of_member_quote_sessions(tmp_path):
    # pre-open windows exist only while a stock is in the top-N: no session is common to all members
    spans = {s: [("2024-01-02", "2024-04-08")] for s in "ABC"}
    spans.update({s: [("2024-02-15", END)] for s in "DEF"})
    port = _IPortal(tmp_path, spans=spans, daily={s: "2010-01-04" for s in "ABCDEF"}, members=_rotating)
    for s in "ABC":
        _record_windows(port.store, s, _sessions("2024-01-02", "2024-03-28"), window="pre_0920")
    for s in "DEF":
        _record_windows(port.store, s, _sessions("2024-04-01", "2024-09-30"), window="pre_0920")
    reqs = [Requirement("bars", "1m", sessions=("PRE", "RTH")), Requirement("bars", "1d"),
            Requirement("quotes", None, sessions=("PRE",))]
    pc = PlanContext(port, tmp_path / "r", min_coverage=0.95)
    r = _plan_instance("T", _IInst(reqs, universe_name=TOP), "1m", pc, _row())
    assert r["testable"] == "Y", r["reason"]
    assert (r["start"], r["end"]) == ("2024-01-02", "2024-09-30"), r


# ------------------------------------------------------------------------------------------- item 3
def test_quote_range_reads_recorded_alpaca_windows_not_the_quotes_parquet(tmp_path, monkeypatch):
    port = _IPortal(tmp_path)
    st = port.store
    days = _sessions("2024-01-02", "2024-01-05")
    _record_windows(st, "SPY", days[:3])
    _record_windows(st, "SPY", days[3:], n=0)              # fetched, but the window had no NBBO update
    extra = pd.DataFrame({"bid": [1.0], "ask": [1.1], "bid_size": [1.0], "ask_size": [1.0]},
                         index=pd.DatetimeIndex([pd.Timestamp("2024-01-08 12:00", tz=ET)], name="ts"))
    st.write("quotes", "alpaca", "SPY", extra, partition="year")      # outside every window: never served
    dq = pd.DataFrame({"bid": 1.0, "ask": 1.1, "bid_size": 1.0, "ask_size": 1.0},
                      index=pd.DatetimeIndex([pd.Timestamp(f"{d} 12:00", tz=ET) for d in ("2024-01-03", "2024-01-04")],
                                             name="ts"))
    st.write("quotes", "databento", "QQQ", dq, partition="year")
    orig = st.read

    def guard(kind, source, symbol, *a, **k):
        assert not (kind == "quotes" and source == "alpaca"), "read a full Alpaca quotes parquet"
        return orig(kind, source, symbol, *a, **k)

    monkeypatch.setattr(st, "read", guard)
    pc = PlanContext(port, tmp_path / "r")
    assert pc.quote_range("SPY") == _iso(days[:3])
    assert pc.quote_range("QQQ") == ["2024-01-03", "2024-01-04"]
    assert pc.quote_range("IWM") is None


# ------------------------------------------------------------------------------------------- item 4
def test_pit_intraday_universe_dated_by_each_sessions_members(tmp_path):
    # members rotate on 2020-01-02; each stock has 1m bars only around its own member sessions. Today's
    # members' first bars (2019-11-15) would cut four years of fully covered sessions.
    spans = {s: [("2016-01-04", "2020-01-09")] for s in "ABC"}
    spans.update({s: [("2019-11-15", END)] for s in "DEF"})
    port = _IPortal(tmp_path, spans=spans, daily={s: "2010-01-04" for s in "ABCDEF"},
                    members=lambda d: ["A", "B", "C"] if d < pd.Timestamp("2020-01-02") else ["D", "E", "F"])
    pc = PlanContext(port, tmp_path / "r", min_coverage=0.95)
    r = _plan_instance("T", _IInst([Requirement("bars", "1m"), Requirement("bars", "1d")], universe_name=TOP),
                       "1m", pc, _row())
    assert r["testable"] == "Y", r["reason"]
    assert (r["start"], r["end"]) == ("2016-01-04", END), r
    assert "coverage" in r["binding_constraint"] and r["expected_fidelity"] == "T1_BAR"


def test_pit_intraday_partial_coverage_tiers_t2(tmp_path):
    # M24 never has 1m bars: 96 % >= 95 % is accepted, but the engine flags coverage < 100 % as T2_PROXY
    mem = [f"M{i:02d}" for i in range(25)]
    port = _IPortal(tmp_path, spans={m: ALL for m in mem[:24]}, daily={m: "2010-01-04" for m in mem},
                    members=lambda d: mem)
    r = _plan_instance("T", _IInst([Requirement("bars", "1m")], universe_name=TOP), "1m",
                       PlanContext(port, tmp_path / "r", min_coverage=0.95), _row())
    assert r["testable"] == "Y" and r["expected_fidelity"] == "T2_PROXY", r
    assert "96.0%" in r["fidelity_notes"]
    strict = _plan_instance("T", _IInst([Requirement("bars", "1m")], universe_name=TOP), "1m",
                            PlanContext(port, tmp_path / "s", min_coverage=1.0), _row())
    assert strict["testable"] == "N" and "coverage never reaches 100%" in strict["reason"], strict


def test_pit_intraday_start_waits_for_past_members_coverage(tmp_path):
    # C (a member until 2017) never got intraday bars: the engine's coverage at those sessions is 2/3 and it
    # refuses the run; today's members (A, B, D) all have bars from 2017-11-15
    spans = {"A": ALL, "B": ALL, "D": [("2017-11-15", END)]}
    port = _IPortal(tmp_path, spans=spans, daily={s: "2010-01-04" for s in "ABCD"},
                    members=lambda d: ["A", "B", "C"] if d < pd.Timestamp("2018-01-02") else ["A", "B", "D"])
    pc = PlanContext(port, tmp_path / "r", min_coverage=0.95)
    r = _plan_instance("T", _IInst([Requirement("bars", "1m")], universe_name=TOP), "1m", pc, _row())
    assert r["testable"] == "Y", r["reason"]
    assert r["start"] == "2018-01-02", r


def test_pit_intraday_pre_probe_of_kept_members(tmp_path):
    # 24 of 25 members have bars from 2016-01-04 (96 % >= 95 %); M24 only from 2016-03-01. RTH-only: the run
    # starts 2016-01-04. With PRE bars the engine probes every kept member over the first 10 sessions.
    mem = [f"M{i:02d}" for i in range(25)]
    spans = {m: ALL for m in mem[:24]}
    spans["M24"] = [("2016-03-01", END)]
    port = _IPortal(tmp_path, spans=spans, daily={m: "2010-01-04" for m in mem}, members=lambda d: mem)
    pc = PlanContext(port, tmp_path / "r", min_coverage=0.95)
    rth = _plan_instance("T", _IInst([Requirement("bars", "1m")], universe_name=TOP), "1m", pc, _row())
    assert rth["testable"] == "Y" and rth["start"] == "2016-01-04", rth
    pre = _plan_instance("T", _IInst([Requirement("bars", "1m", sessions=("PRE", "RTH"))], universe_name=TOP),
                         "1m", pc, _row())
    assert pre["testable"] == "Y", pre["reason"]
    assert pre["start"] == str(_sessions("2016-01-04", "2016-03-01")[-10].date()), pre


def test_sp500_intraday_members_checked_per_session(tmp_path):
    # S039/S040-like: 5m bars exist only from 2017-11 for the members of those sessions; C left in 2018
    spans = {"A": [("2017-11-01", END)], "B": [("2017-11-01", END)], "D": [("2018-01-02", END)]}
    port = _IPortal(tmp_path, spans=spans, daily={s: "1995-01-03" for s in "ABCD"},
                    members=lambda d: ["A", "B", "C"] if d < pd.Timestamp("2018-02-01") else ["A", "B", "D"])
    pc = PlanContext(port, tmp_path / "r", min_coverage=0.95)
    inst = _IInst([Requirement("bars", "5m"), Requirement("bars", "1d")], universe_name="SP500")
    inst.bar_freq = "5m"
    r = _plan_instance("T", inst, "5m", pc, _row())
    assert r["testable"] == "Y", r["reason"]
    assert r["start"] == "2018-02-01", r


# ------------------------------------------------------------------------------------------- item 5
def test_plan_cache_drops_intraday_memo_when_intraday_data_changes(tmp_path):
    port = _IPortal(tmp_path)
    res = tmp_path / "res"
    pc = PlanContext(port, res)
    assert pc.bar_range("SPY", "1m") is None and pc.quote_range("SPY") is None
    assert pc.bar_range("SPY", "1d") is not None
    pc.save()
    assert "bars|SPY|1m" in PlanContext(port, res).cache            # unchanged data: memo kept
    # an intraday fetch lands (1m bars + recorded quote windows)
    days = _sessions("2024-01-02", "2024-01-31")
    port.spans["SPY"] = [(days[0], days[-1])]
    b = port.bars("SPY", "1m", days[0].tz_localize(ET), days[-1].tz_localize(ET) + pd.Timedelta(days=1), None)
    port.store.write("bars_1m", "alpaca", "SPY", b.rename_axis("start"), partition="year")
    _record_windows(port.store, "SPY", days)
    pc = PlanContext(port, res)
    assert "bars|SPY|1m" not in pc.cache and "quotes|SPY" not in pc.cache
    assert "bars|SPY|1d" in pc.cache                                # daily facts are not intraday-dependent
    assert pc.bar_range("SPY", "1m") == ["2024-01-02", "2024-01-31"]
    assert pc.quote_range("SPY") == _iso(days)


def test_pit_members_every_session_persist_as_diffs(tmp_path):
    # the per-session membership (every session, not samples) is stored as changes and replayed exactly
    calls = []
    rot = lambda d: (calls.append(d), ["A", "B", "C"] if d < pd.Timestamp("2020-01-02") else ["A", "B", "D"])[1]
    port = _IPortal(tmp_path, spans={s: ALL for s in "ABCD"}, daily={s: "2010-01-04" for s in "ABCD"}, members=rot)
    pc = PlanContext(port, tmp_path / "r", min_coverage=0.95)
    first = pc._members_every_session(TOP)
    n_calls = len(calls)
    assert n_calls == len(first) and first[0] == ["A", "B", "C"] and first[-1] == ["A", "B", "D"]
    rec = pc.cache[f"universe|{TOP}|every_session"]
    assert len(rec["diffs"]) == 2                       # the first session + the 2020-01-02 swap
    pc.save()
    again = PlanContext(port, tmp_path / "r", min_coverage=0.95)._members_every_session(TOP)
    assert again == first and len(calls) == n_calls     # replayed from plan_cache.json, no portal call
