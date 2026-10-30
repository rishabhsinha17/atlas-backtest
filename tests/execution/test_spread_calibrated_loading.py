"""The shipped calibration JSON loads (schema v1 and v2 accepted, clear error otherwise) and a default
BacktestConfig builds and runs a Backtester with the calibrated spread model."""
import json

import pandas as pd
import pytest

from atlasbt.core.clock import nyse
from atlasbt.engine import BacktestConfig, Backtester, Strategy
from atlasbt.engine.testing import InMemoryPortal, synthetic_bars_from_rows
from atlasbt.execution.spread_calibrated import (CALIBRATION_PATH, CalibratedSpreadModel,
                                                 normalize_calibration)


def test_shipped_json_loads_and_prices():
    raw = json.loads(CALIBRATION_PATH.read_text())
    m = CalibratedSpreadModel()
    h, why = m.half_spread("SPY", "2026-07-01", 746.77, 4.714e10, 0.01143, "ETF")
    assert 0 < h < 1e-4 and why.startswith("calibrated spread model (T2)")
    assert ("era" in raw) or ("era_anchors" in raw)


def test_v1_schema_still_accepted_and_bad_schema_rejected():
    raw = json.loads(CALIBRATION_PATH.read_text())
    v1 = {k: v for k, v in raw.items() if k not in ("era", "sigma_prior")}
    v1["era_anchors"] = [[1990.0, 0.0, 0.0], [2100.0, 0.0, 0.0]]
    v1["tod_profile"] = {k: v for k, v in raw["tod_profile"].items() if k != "liquid_etf"}
    m = CalibratedSpreadModel(calibration=v1)
    assert m.half_spread(None, "2024-01-02", 30.0, 5e7, 0.02, "STK")[0] > 0
    assert m.tod_multiplier(0, 390, 1e10, "ETF") == m.tod_multiplier(0, 390, 1e10, "STK")   # no ETF profile in v1
    with pytest.raises(ValueError, match="unsupported schema"):
        normalize_calibration({"beta": {}})


class _Buy(Strategy):
    id = "SPREADLOAD"
    bar_freq = "1d"
    universe = ["XYZ"]

    def on_bar(self, ctx):
        if not ctx.position("XYZ"):
            ctx.order("XYZ", 100)


def test_default_backtest_config_builds_and_runs_with_calibrated_spreads():
    cfg = BacktestConfig()
    assert cfg.resolved_spread_model() == "calibrated"
    ss = list(nyse().sessions(pd.Timestamp("2024-03-04"), pd.Timestamp("2024-05-31")))[:40]
    rows = [(100 + 0.1 * i, 101 + 0.1 * i, 99 + 0.1 * i, 100 + 0.1 * i, 2e6) for i in range(len(ss))]
    df = synthetic_bars_from_rows(rows, "1d", session_dates=ss)
    bt = Backtester(_Buy(), InMemoryPortal(bars={("XYZ", "1d"): df}), cfg)
    assert isinstance(bt.spread_model, CalibratedSpreadModel)
    r = bt.run(ss[25], ss[-1], raise_errors=True)
    assert len(r.fills) >= 1
    assert all(r.fills.spread_cost.to_numpy() >= 0)
