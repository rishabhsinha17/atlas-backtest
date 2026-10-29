"""Regression: synthetic quoting before VIX history must not crash (empty VIX family)."""
import pandas as pd

from atlasbt.options.synthetic import _vix30_or_default


def test_vix30_default_on_empty_family():
    assert _vix30_or_default(pd.DataFrame()) == 0.20
    assert _vix30_or_default(pd.DataFrame(columns=["index", "level"])) == 0.20
    fam = pd.DataFrame([{"index": "VIX9D", "level": 0.15}])
    assert _vix30_or_default(fam) == 0.20
    fam = pd.DataFrame([{"index": "VIX", "level": 0.31}])
    assert _vix30_or_default(fam) == 0.31
