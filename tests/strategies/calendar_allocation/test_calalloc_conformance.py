"""Every strategy of this module pair is registered with the sheet's metadata:
verbatim spec, sizing profile, grid keys that are params, declared deviations."""
import json
from pathlib import Path

import pytest

from atlasbt.strategies.registry import get
from atlasbt.engine.strategy import Strategy

ROOT = Path(__file__).resolve().parents[3]
SHEET = {r["ID"]: r for r in json.loads((ROOT / "docs" / "strategies.json").read_text())}
IDS = [f"S{i:03d}" for i in range(31, 46)] + ["S089", "S090", "S101", "S102", "S103", "S104", "S105", "S117",
                                               "S118", "S119", "S120", "S121", "S122", "S124", "S125"]


@pytest.mark.parametrize("sid", IDS)
def test_registered_with_sheet_metadata(sid):
    cls = get(sid)
    row = SHEET[sid]
    assert issubclass(cls, Strategy) and cls.id == sid
    assert cls.name == row["Strategy"]
    assert cls.spec == row["Entry rule"] + " || " + row["Exit / risk rule"]
    assert cls.sizing == row["Sizing"]
    assert cls.asset == row["Assets"]
    assert set(cls.grid) <= set(cls.params)
    assert all(len(v) >= 1 for v in cls.grid.values())
    assert len(cls.deviations) >= 1 and cls.requires
    inst = cls()                                      # baseline params construct
    assert inst.params == cls.params
