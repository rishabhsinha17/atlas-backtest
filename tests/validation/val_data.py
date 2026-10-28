"""Synthetic BacktestResult builders for validation tests (mechanics only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from atlasbt.core.types import BacktestResult, RunStatus


def synthetic_result(returns, fills=None, trades=None, nav0=100_000.0, sid="SYN", start="2020-01-02", meta=None):
    r = np.asarray(returns, float)
    idx = pd.bdate_range(start, periods=len(r), tz="UTC") + pd.Timedelta(hours=21)
    eq = pd.Series(nav0 * np.cumprod(1 + r), index=idx)
    m = {"initial_capital": nav0}
    m.update(meta or {})
    return BacktestResult(strategy_id=sid, params={}, status=RunStatus.OK, equity=eq, returns=pd.Series(r, index=idx),
                          fills=fills if fills is not None else pd.DataFrame(),
                          trades=trades if trades is not None else pd.DataFrame(), positions=pd.DataFrame(),
                          ledger=pd.DataFrame(), fidelity={}, meta=m)


def fills_frame(ts, qty, price, spread_cost=0.0, commission=0.0, multiplier=1.0, symbol="SPY"):
    n = len(ts)
    return pd.DataFrame({"ts": ts, "symbol": [symbol] * n, "quantity": qty, "price": price,
                         "multiplier": multiplier, "commission": commission, "exchange_fees": 0.0,
                         "regulatory_fees": 0.0, "spread_cost": spread_cost, "impact_cost": 0.0,
                         "fidelity": 1})
