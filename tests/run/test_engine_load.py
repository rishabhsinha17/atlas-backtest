"""Load test: the engine must construct and run with its DEFAULT config (catches e.g. a calibration file
out of sync with the spread-model code, which broke every backtest on 2026-10-29: KeyError 'era')."""
from atlasbt.engine import BacktestConfig, Backtester
from atlasbt.engine.testing import InMemoryPortal, synthetic_daily_bars
from atlasbt.run.replicate import BuyHold


def test_default_config_constructs_and_runs_on_synthetic_portal():
    bars = synthetic_daily_bars("2020-01-02", "2020-06-30", seed=3)          # synthetic: mechanics only
    portal = InMemoryPortal(bars={("SPY", "1d"): bars})
    res = Backtester(BuyHold(), portal, BacktestConfig()).run("2020-01-02", "2020-06-30")
    assert res.status.value == "OK", res.not_testable_reason
    assert len(res.fills) >= 1 and len(res.equity) > 100
