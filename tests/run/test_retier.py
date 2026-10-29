from types import SimpleNamespace

import pandas as pd

from atlasbt.run.retier import retier_result


def test_synthetic_retier_calibrated_bar_fills_to_t1_and_keeps_options():
    f = pd.DataFrame({
        "ts": pd.to_datetime(["2020-01-02 14:30", "2020-01-03 14:30", "2020-01-06 21:00"], utc=True),
        "fidelity": [2, 2, 3],
        "fidelity_reason": ["spread modelled (calibrated); daily bar: next-open fill", "spread modelled (calibrated)",
                            "option quote snapshot"],
        "liquidity": ["TAKER", "TAKER", "TAKER"], "asset_class": ["ETF", "ETF", "OPT"],
        "notional": [1000.0, 2000.0, 500.0], "quantity": [1, -1, 1], "price": [1.0, 1.0, 1.0]})
    res = SimpleNamespace(fills=f, fidelity={"tiers": {"T2_PROXY": {"fills": 2, "notional": 3000.0}}, "reasons": [],
                                             "grade": "C"}, meta={})
    ch = retier_result(res, "dd5e315")
    assert res.meta["retiered_with"] == "dd5e315"
    assert list(res.fills["fidelity"])[2] == 3                      # option fill keeps its data tier
    assert res.fidelity["grade_before_retier"] == "C" and res.fidelity["grade"] is None
    assert sum(v["fills"] for v in res.fidelity["tiers"].values()) == 3
    assert isinstance(ch, dict)
