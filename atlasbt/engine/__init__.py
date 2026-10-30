"""Backtest engine: event loop, PIT Context, order manager, results.

Quick start::

    from atlasbt.engine import Backtester, BacktestConfig, Strategy, Requirement
    result = Backtester(MyStrategy(), portal, BacktestConfig()).run("2015-01-01", "2024-12-31")
"""
from .backtester import Backtester, run_backtest  # noqa: F401
from .config import BacktestConfig  # noqa: F401
from .strategy import PROFILE_SESSION_RULES, Requirement, Strategy  # noqa: F401
