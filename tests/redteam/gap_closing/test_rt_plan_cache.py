"""Red team (gap closing): plan_cache.json invalidation (82bd563 _intraday_version) vs the OPRA store (339d0f0).

The PIT intraday window memo (key 'pit|window|...') depends on the engine's member checks, including
portal.has(member, 'options', '1m'), which the OPRA store (option_quotes/databento) answers. The memo is
dropped only when bars_<intraday>, quotes or reference/alpaca change, so an OPRA fetch leaves a stale
NOT_TESTABLE (or a stale window) in results/plan_cache.json. A failure is a finding.
"""
from __future__ import annotations

from atlasbt.core.types import Fidelity
from atlasbt.engine.strategy import Requirement
from atlasbt.run.plan import PlanContext, _plan_instance

from tests.redteam.gap_closing.test_rt_opra import _OPortal, _write, cache  # noqa: F401  (fixture)
from tests.run.test_plan_intraday import ALL, TOP, _IInst
from tests.run.test_plan_star import _row

MEM = ["AAPL", "MSFT", "NVDA"]


def _plan(port, res):
    reqs = [Requirement("bars", "1m"), Requirement("options", "1m", min_fidelity=Fidelity.T1_BAR)]
    pc = PlanContext(port, res, min_coverage=0.95)
    r = _plan_instance("T", _IInst(reqs, universe_name=TOP), "1m", pc, _row())
    pc.save()
    return r


def test_rt_pit_window_memo_survives_an_opra_fetch(cache):
    port = _OPortal(cache, spans={s: ALL for s in MEM}, daily={s: "2010-01-04" for s in MEM},
                    members=lambda d: MEM)
    res = cache / "res"
    before = _plan(port, res)
    assert before["testable"] == "N", before            # no option data yet: correctly refused
    # databento_options fetch (earnings plan) lands OPRA samples for every member
    _write(cache / "store", [("2024-01-10T15:00:00Z", "2024-01-10T14:59:00Z", f"{s:<6}240119C00190000", 1.0, 1.1)
                             for s in MEM])
    port = _OPortal(cache, spans={s: ALL for s in MEM}, daily={s: "2010-01-04" for s in MEM},
                    members=lambda d: MEM)
    fresh = _plan(port, cache / "fresh")                 # what the planner says on the new data
    after = _plan(port, res)                             # same results dir: plan_cache.json reused
    assert fresh["testable"] == "Y", fresh["reason"]
    assert (after["testable"], after["reason"]) == (fresh["testable"], fresh["reason"]), (
        f"stale memo after the OPRA fetch: {after['testable']} ({after['reason']}); fresh plan: {fresh['testable']}")
