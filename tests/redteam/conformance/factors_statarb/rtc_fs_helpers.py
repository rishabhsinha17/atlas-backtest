"""Red-team conformance helpers (factors S085-S100 + stat-arb S106-S115, S123).

SYNTHETIC data, mechanics only. Self-contained so the red-team suite does not depend on the
authors' test helpers.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from atlasbt.core.clock import nyse
from atlasbt.engine import BacktestConfig
from atlasbt.engine.backtester import run_backtest
from atlasbt.engine.testing import InMemoryPortal
from atlasbt.execution.impact import SquareRootImpact
from atlasbt.execution.spread import SpreadConfig

CAL = nyse()
NAV0 = 1_000_000.0


def zero_cost(**kw) -> BacktestConfig:
    base = dict(fees="NO_FEES",
                spread=SpreadConfig(bucket_floors=((math.inf, 0.0),), default_spread=1e-12, tod_open=(),
                                    tod_close=(), tick_floor=False, max_spread=1e-12),
                impact=SquareRootImpact(k=0.0), debug=False)
    base.update(kw)
    return BacktestConfig(**base)


def sess(a, b) -> list:
    return list(CAL.sessions(pd.Timestamp(a), pd.Timestamp(b)))


def close_ts(s) -> pd.Timestamp:
    return CAL.open_close(pd.Timestamp(s))[1]


def open_ts(s) -> pd.Timestamp:
    return CAL.open_close(pd.Timestamp(s))[0]


def nxt(s) -> pd.Timestamp:
    return CAL.next_session(pd.Timestamp(s))


def bars(closes: pd.Series, volume: float = 1e8) -> pd.DataFrame:
    """Daily bars; open = previous close, high/low +-0.1% around open/close."""
    c = closes.astype(float)
    o = c.shift(1).fillna(c.iloc[0])
    oc = [CAL.open_close(s) for s in c.index]
    return pd.DataFrame({"open": o.to_numpy(), "high": (np.maximum(o, c) * 1.001).to_numpy(),
                         "low": (np.minimum(o, c) * 0.999).to_numpy(), "close": c.to_numpy(),
                         "volume": float(volume), "session": "DAY",
                         "known_at": pd.DatetimeIndex([b for _, b in oc])},
                        index=pd.DatetimeIndex([a for a, _ in oc], name="start"))


def path(dates, rets, p0=100.0) -> pd.Series:
    return pd.Series(p0 * np.cumprod(1.0 + np.asarray(rets, float)), index=pd.DatetimeIndex(dates))


class SMPortal(InMemoryPortal):
    def __init__(self, *a, master: Optional[pd.DataFrame] = None, **kw):
        super().__init__(*a, **kw)
        self._master = master

    def security_master(self, as_of):
        if self._master is None:
            return super().security_master(as_of)
        return self._master.copy()


def fund(sym, field, value, period_end, known_at) -> dict:
    k = pd.Timestamp(known_at)
    k = k.tz_localize("America/New_York").tz_convert("UTC") if k.tzinfo is None else k.tz_convert("UTC")
    return {"symbol": sym, "field": field, "value": float(value), "period_end": pd.Timestamp(period_end),
            "known_at": k, "source": "synthetic"}


def portal(closes: dict, *, master=None, fundamentals=None, universe=None, borrow=None,
           universe_name="SP500") -> SMPortal:
    b = {(s, "1d"): bars(c) for s, c in closes.items()}
    f = None
    if fundamentals:
        f = pd.DataFrame(fundamentals)
        f["known_at"] = pd.to_datetime(f["known_at"], utc=True)
    uni = {universe_name: list(universe if universe is not None else closes)}
    return SMPortal(bars=b, fundamentals=f, universes=uni, master=master, borrow=borrow)


def run(strategy, p, start, end, **cfg):
    return run_backtest(strategy, p, start, end, zero_cost(**cfg), raise_errors=True)


def orders(res) -> pd.DataFrame:
    o = res.diagnostics["orders"]
    if len(o) == 0:
        return pd.DataFrame(columns=["symbol", "quantity", "type", "tag", "status", "submitted_at"])
    return o[o["status"] != "REJECTED"]


def sides_at(res, ts) -> tuple[set, set]:
    o = orders(res)
    o = o[pd.DatetimeIndex(o["submitted_at"]) == ts]
    return set(o.loc[o["quantity"] > 0, "symbol"]), set(o.loc[o["quantity"] < 0, "symbol"])
