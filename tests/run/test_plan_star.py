"""Planner: bars requirements declared for the whole universe (``symbols=("*",)``) are verified by the
same universe coverage rule as bars requirements without symbols (they used to be skipped)."""
import pandas as pd

from atlasbt.core.types import Fidelity
from atlasbt.engine.strategy import Requirement
from atlasbt.run.plan import PlanContext, _plan_instance


class _Portal:
    """Synthetic portal: SP500 = 20 members; `covered(d)` members have bars from `first[m]` on."""

    def __init__(self, first: dict, static=None):
        self.first = {k: pd.Timestamp(v) for k, v in first.items()}
        self.static = static or {}

    # universe / coverage -------------------------------------------------------------------
    def universe(self, name, as_of, require_bars=False):
        mem = sorted(self.first) if name.upper() == "SP500" else list(self.static.get(name.upper(), []))
        if not require_bars:
            return mem
        d = pd.Timestamp(as_of).tz_convert("America/New_York").tz_localize(None).normalize()
        return [m for m in mem if self.first.get(m, pd.Timestamp.max) <= d]

    def universe_survivorship_bias(self, name, as_of):
        allm, kept = self.universe(name, as_of), self.universe(name, as_of, True)
        return {"members": len(allm), "with_data": len(kept), "coverage": len(kept) / len(allm), "missing": []}

    # bars ----------------------------------------------------------------------------------
    def has(self, sym, kind, freq=None, **kw):
        if sym == "*":
            return kind == "bars" and bool(self.first)
        return kind == "bars" and sym in self.first and self.first[sym] < pd.Timestamp("2099-01-01")

    def fidelity_of(self, sym, kind, freq=None):
        return Fidelity.T1_BAR if self.has(sym, kind, freq) else Fidelity.NA

    def bars_coverage(self, sym, start, end, as_of, freq="1d"):
        f = self.first.get(sym)
        if f is None or f >= pd.Timestamp("2099-01-01"):
            return {"available": False}
        return {"available": True, "first": pd.Timestamp(f, tz="America/New_York"),
                "last": pd.Timestamp("2026-10-27 16:00", tz="America/New_York")}


class _Inst:
    bar_freq = "1d"
    sizing = None
    asset = "STK"
    flat_by = None

    def __init__(self, requires, universe=(), universe_name=None):
        self.requires = list(requires)
        self.universe = list(universe)
        self.universe_name = universe_name


def _row():
    return {"id": "T", "testable": "N", "reason": "", "start": "", "end": "", "fidelity_notes": "", "n_grid": 1}


STAR = Requirement("bars", "1d", symbols=("*",))


def test_star_bars_sp500_uses_pit_coverage_start(tmp_path):
    # 20 members: 18 have bars from 2005, one from 2012-03-01, one never -> 95 % first reached in 2012-03
    first = {f"M{i:02d}": "2005-01-03" for i in range(18)}
    first["LATE"] = "2012-03-01"
    first["DEAD"] = "2099-12-31"
    pc = PlanContext(_Portal(first), tmp_path, min_coverage=0.95)
    r = _plan_instance("T", _Inst([STAR], universe_name="SP500"), "1d", pc, _row())
    assert r["testable"] == "Y", r["reason"]
    assert r["start"].startswith("2012-03"), r["start"]
    assert "coverage" in r["binding_constraint"]
    assert "DEAD" in r["fidelity_notes"]           # dropped by the coverage filter, as for symbols=None


def test_star_bars_sp500_strict_rule_fails_like_no_symbols(tmp_path):
    first = {f"M{i:02d}": "2005-01-03" for i in range(19)}
    first["DEAD"] = "2099-12-31"
    pc = PlanContext(_Portal(first), tmp_path, min_coverage=1.0)
    star = _plan_instance("T", _Inst([STAR], universe_name="SP500"), "1d", pc, _row())
    plain = _plan_instance("T", _Inst([Requirement("bars", "1d")], universe_name="SP500"), "1d", pc, _row())
    assert star["testable"] == plain["testable"] == "N"
    assert star["reason"] == plain["reason"]


def test_star_bars_static_universe_checks_members(tmp_path):
    # a static universe member without bars makes the requirement fail (it used to be skipped)
    port = _Portal({"AAA": "2010-01-04", "BBB": "2099-12-31"})
    pc = PlanContext(port, tmp_path, min_coverage=0.95)
    r = _plan_instance("T", _Inst([STAR], universe=["AAA", "BBB"]), "1d", pc, _row())
    assert r["testable"] == "N" and "BBB" in r["reason"]
    # all members present: the window starts at the latest listing (+ warm-up), like symbols=None
    port = _Portal({"AAA": "2010-01-04", "BBB": "2015-06-01"})
    pc = PlanContext(port, tmp_path / "b", min_coverage=0.95)
    r = _plan_instance("T", _Inst([STAR], universe=["AAA", "BBB"]), "1d", pc, _row())
    assert r["testable"] == "Y" and r["start"].startswith("2016-0"), r
    assert "BBB" in r["binding_constraint"]


def test_star_bars_without_universe_is_portal_wide(tmp_path):
    pc = PlanContext(_Portal({}), tmp_path, min_coverage=0.95)
    r = _plan_instance("T", _Inst([STAR]), "1d", pc, _row())
    assert r["testable"] == "N" and "'*'" in r["reason"]
