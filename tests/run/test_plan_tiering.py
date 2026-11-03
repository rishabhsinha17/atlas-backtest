"""Planner: non-critical requirements tier the expected fidelity (S042 DIVIDEND_DECL -> T2) and universes a
strategy queries at run time (S042: 'SP500_TOP50_DV') must pass the engine's coverage rule."""
import pandas as pd

from atlasbt.core.types import Fidelity
from atlasbt.engine.strategy import Requirement
from atlasbt.run.plan import PlanContext, _plan_instance, _runtime_universes

from .test_plan_star import _Inst, _Portal, _row


class _EvPortal(_Portal):
    def __init__(self, first, static=None, events=None):
        super().__init__(first, static)
        self.events = events or {}          # kind -> Fidelity
        self.EVENT_SOURCES = {}

    def has(self, sym, kind, freq=None, **kw):
        if kind == "events":
            return (freq or "").upper() in self.events
        return super().has(sym, kind, freq, **kw)

    def fidelity_of(self, sym, kind, freq=None):
        if kind == "events":
            return self.events.get((freq or "").upper(), Fidelity.NA)
        return super().fidelity_of(sym, kind, freq)


def _etf(tmp_path, events, static=None, first=None):
    first = first or {"SPY": "2005-01-03", "QQQ": "2005-01-03"}
    return PlanContext(_EvPortal(first, static, events), tmp_path, min_coverage=0.95)


def test_noncritical_event_tiers_the_run(tmp_path):
    reqs = [Requirement("bars", "1d"),
            Requirement("events", "DIVIDEND_DECL", symbols=("*",), critical=False, min_fidelity=Fidelity.T2_PROXY)]
    r = _plan_instance("T", _Inst(reqs, universe=["SPY", "QQQ"]), "1d",
                       _etf(tmp_path, {"DIVIDEND_DECL": Fidelity.T2_PROXY}), _row())
    assert r["testable"] == "Y" and r["expected_fidelity"] == "T2_PROXY", r
    assert "DIVIDEND_DECL" in r["fidelity_notes"]


def test_noncritical_missing_never_blocks(tmp_path):
    reqs = [Requirement("bars", "1d"), Requirement("events", "DIVIDEND_DECL", symbols=("*",), critical=False)]
    r = _plan_instance("T", _Inst(reqs, universe=["SPY", "QQQ"]), "1d", _etf(tmp_path / "b", {}), _row())
    assert r["testable"] == "Y" and r["expected_fidelity"] == "T1_BAR"
    assert "unavailable" in r["fidelity_notes"]


def test_runtime_universe_coverage_rule(tmp_path):
    # the strategy trades SPY/QQQ but also asks the portal for 'TOPX' at run time: 2 of 4 members lack bars
    first = {"SPY": "2005-01-03", "QQQ": "2005-01-03", "AAA": "2005-01-03", "BBB": "2005-01-03",
             "CCC": "2099-12-31", "DDD": "2099-12-31"}
    inst = _Inst([Requirement("bars", "1d")], universe=["SPY", "QQQ"])
    inst.runtime_universes = ("TOPX",)
    assert _runtime_universes(inst, None) == ["TOPX"]
    pc = _etf(tmp_path, {}, static={"TOPX": ["AAA", "BBB", "CCC", "DDD"]}, first=first)
    r = _plan_instance("T", inst, "1d", pc, _row())
    assert r["testable"] == "N" and "TOPX" in r["reason"] and "50%" in r["reason"], r
    first.update({"CCC": "2005-01-03", "DDD": "2005-01-03"})
    pc = _etf(tmp_path / "c", {}, static={"TOPX": ["AAA", "BBB", "CCC", "DDD"]}, first=first)
    assert _plan_instance("T", inst, "1d", pc, _row())["testable"] == "Y"


def test_s042_declares_runtime_top50():
    from atlasbt.strategies import registry as R
    from atlasbt.run import variants as V
    R.load_all()
    assert _runtime_universes(V.strategy_class("S042")(), None) == ["SP500_TOP50_DV"]


class _FundPortal(_EvPortal):
    """fundamentals: 'xbrl_f' known from 2009-08-14 (T1), 'cons_ntm' from 2017-10-27 (T2)."""
    FIRST = {"xbrl_f": pd.Timestamp("2009-08-14", tz="UTC"), "cons_ntm": pd.Timestamp("2017-10-27", tz="UTC")}
    FID = {"xbrl_f": Fidelity.T1_BAR, "cons_ntm": Fidelity.T2_PROXY}

    def has(self, sym, kind, freq=None, **kw):
        if kind == "fundamentals":
            return freq in self.FIRST
        return super().has(sym, kind, freq, **kw)

    def fidelity_of(self, sym, kind, freq=None):
        if kind == "fundamentals":
            return self.FID.get(freq, Fidelity.NA)
        return super().fidelity_of(sym, kind, freq)

    def fundamentals(self, symbols, fields, as_of):
        rows = [{"symbol": s, "field": f, "value": 1.0} for s in symbols for f in fields
                if f in self.FIRST and pd.Timestamp(as_of) >= self.FIRST[f]]
        return pd.DataFrame(rows, columns=["symbol", "field", "value"])


def test_fundamentals_field_start_and_tier_from_portal(tmp_path):
    first = {"SPY": "2000-01-03", "QQQ": "2000-01-03"}
    pc = PlanContext(_FundPortal(first), tmp_path, min_coverage=0.95)
    assert pc.field_start("cons_ntm") == "2017-11-30" and pc.field_start("xbrl_f") == "2009-09-30"
    reqs = [Requirement("bars", "1d"), Requirement("fundamentals", "cons_ntm", symbols=("*",),
                                                   min_fidelity=Fidelity.T2_PROXY)]
    r = _plan_instance("T", _Inst(reqs, universe=["SPY", "QQQ"]), "1d", pc, _row())
    assert r["testable"] == "Y" and r["start"] == "2017-11-30" and r["expected_fidelity"] == "T2_PROXY", r
    # a T2 field under a T1 minimum is refused, as the engine would
    reqs[1] = Requirement("fundamentals", "cons_ntm", symbols=("*",), min_fidelity=Fidelity.T1_BAR)
    r = _plan_instance("T", _Inst(reqs, universe=["SPY", "QQQ"]), "1d", pc, _row())
    assert r["testable"] == "N" and "cons_ntm" in r["reason"]
