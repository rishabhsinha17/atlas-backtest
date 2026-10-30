import pytest
import pandas as pd

from atlasbt.run.holds import held, stock_symbols
from atlasbt.run.plan import PlanContext, plan_strategy

from .conftest import needs_cache


def test_stock_hold_classification(monkeypatch):
    monkeypatch.delenv("ATLASBT_ALLOW_STOCK", raising=False)
    monkeypatch.setenv("ATLASBT_STOCK_HOLD", "0")
    assert not held("S085")                      # hold lifted
    monkeypatch.setenv("ATLASBT_STOCK_HOLD", "1")
    assert held("S085") and held("S001") and held("S043")
    assert not held("S050") and not held("S105") and not held("S129")
    assert not held("S085", allow=True)
    assert "universe:SP500" in stock_symbols("S085")


@needs_cache
@pytest.mark.slow
def test_plan_windows_and_reasons(portal, tmp_path):
    pc = PlanContext(portal, tmp_path)
    r = plan_strategy("S105", pc)              # SPY + IEF, IEF listed 2002-07-26 -> + 252-session warm-up
    assert r["testable"] == "Y" and r["start"].startswith("2003-07") and r["n_grid"] == 6
    assert pd.Timestamp(r["end"]) >= pd.Timestamp("2026-10-01")
    r = plan_strategy("S015", pc)              # needs quotes for the ETF core: only sample days exist
    assert r["testable"] == "N" and "quotes" in r["reason"]
    r = plan_strategy("S110", pc)              # iNAV has no source
    assert r["testable"] == "N"
    r = plan_strategy("S158", pc)              # intraday option quotes do not exist (portal.fidelity_of NA)
    assert r["testable"] == "N" and "options 1m" in r["reason"]
