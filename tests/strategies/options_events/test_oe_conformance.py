"""Conformance of S158-S172 to the sheet: verbatim spec, sizing, grid keys are params, registry,
fidelity gates (requirements) and the allow_synthetic opt-in."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlasbt.core.types import Fidelity
from atlasbt.strategies import registry

ROWS = {r["ID"]: r for r in json.loads((Path(__file__).resolve().parents[3] / "docs" / "strategies.json").read_text())}
IDS = [f"S{i}" for i in range(158, 173)]
#: docs/OPTIONS.md §13: not testable with free data (need intraday / SPX real quotes)
NT_BY_DESIGN = {"S158", "S159", "S161", "S162", "S163", "S164", "S171", "S172"}


@pytest.mark.parametrize("sid", IDS)
def test_conformance(sid):
    cls = registry.get(sid)
    row = ROWS[sid]
    assert cls.spec == row["Entry rule"] + " || " + row["Exit / risk rule"]
    assert cls.sizing == row["Sizing"]
    assert set(cls.grid) <= set(cls.params)
    assert cls.params.get("allow_synthetic") is False
    assert len(cls.deviations) >= 3
    s = cls()
    opt = [r for r in s.requires if r.kind == "options"]
    assert opt and all(r.critical for r in opt)
    if sid in NT_BY_DESIGN:
        # EOD (T2) chains are not enough: need timestamped quotes T0/T1
        assert all(r.min_fidelity <= Fidelity.T1_BAR for r in opt)
    s2 = cls(allow_synthetic=True)
    assert all(not r.critical and r.min_fidelity == Fidelity.T3_MODELED for r in s2.requires if r.kind == "options")
