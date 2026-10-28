"""Shared helpers for the engine/execution red-team tests (SYNTHETIC data, mechanics only)."""
import math

import pandas as pd

from atlasbt.core.clock import nyse
from atlasbt.engine import BacktestConfig, Strategy
from atlasbt.engine.testing import InMemoryPortal, synthetic_bars_from_rows
from atlasbt.execution.impact import SquareRootImpact
from atlasbt.execution.spread import SpreadConfig

CAL = nyse()


def zero_cost(**kw) -> BacktestConfig:
    base = dict(fees="NO_FEES",
                spread=SpreadConfig(bucket_floors=((math.inf, 0.0),), default_spread=1e-12, tod_open=(),
                                    tod_close=(), tick_floor=False),
                impact=SquareRootImpact(k=0.0), debug=True)
    base.update(kw)
    return BacktestConfig(**base)


class Scripted(Strategy):
    id = "RT"
    bar_freq = "1d"
    universe = ["XYZ"]

    def __init__(self, script=None, on_start=None, on_end=None, **kw):
        super().__init__()
        self.script, self.on_start, self.on_end = script, on_start, on_end
        self.seen, self.fills = [], []

    def on_bar(self, ctx):
        if self.script:
            self.script(ctx, self)

    def on_session_start(self, ctx):
        if self.on_start:
            self.on_start(ctx, self)

    def on_session_end(self, ctx):
        if self.on_end:
            self.on_end(ctx, self)

    def on_fill(self, ctx, fill):
        self.fills.append(fill)


def make(freq="1d", **attrs):
    return type("S", (Scripted,), {"bar_freq": freq, **attrs})


def sessions(start, n):
    return list(CAL.sessions(pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=3 * n + 7)))[:n]


def daily_portal(rows, start="2024-03-04", sym="XYZ", **kw):
    ss = sessions(start, len(rows))
    df = synthetic_bars_from_rows(rows, "1d", session_dates=ss)
    return InMemoryPortal(bars={(sym, "1d"): df}, **kw), ss


def intraday_frame(session, freq="1m", price=100.0, volume=10_000.0, rows=None):
    """Flat (or explicit) RTH bars for one session."""
    o, c = CAL.open_close(session)
    td = pd.Timedelta(freq.replace("m", "min"))
    starts = pd.date_range(o, c, freq=td, inclusive="left")
    if rows is None:
        rows = [(price, price + 0.05, price - 0.05, price, volume)] * len(starts)
    df = synthetic_bars_from_rows(rows, freq, start_times=starts)
    known = (pd.DatetimeIndex(df.index) + td)
    df["known_at"] = known.where(known <= c, c)
    return df
