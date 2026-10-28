"""Helpers for the red-team portfolio/risk/results tests (SYNTHETIC data only)."""
from __future__ import annotations

import math

import pandas as pd

from atlasbt.core.clock import nyse
from atlasbt.engine import BacktestConfig, Backtester, Strategy
from atlasbt.engine.testing import InMemoryPortal, synthetic_bars_from_rows
from atlasbt.execution.impact import SquareRootImpact
from atlasbt.execution.spread import SpreadConfig

CAL = nyse()
KNOWN = pd.Timestamp("2024-01-02", tz="UTC")


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

    def __init__(self, script=None, on_start=None, **kw):
        super().__init__()
        self.script = script
        self.on_start = on_start
        self.seen = []

    def on_bar(self, ctx):
        if self.script:
            self.script(ctx, self)

    def on_session_start(self, ctx):
        if self.on_start:
            self.on_start(ctx, self)


def make(freq="1d", **attrs):
    return type("S", (Scripted,), {"bar_freq": freq, **attrs})


def sessions(start, n):
    return list(CAL.sessions(pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=3 * n + 7)))[:n]


def daily_portal(rows_by_sym, ss, **kw):
    bars = {(sym, "1d"): synthetic_bars_from_rows(rows, "1d", session_dates=ss[:len(rows)])
            for sym, rows in rows_by_sym.items()}
    return InMemoryPortal(bars=bars, **kw)


def run(script, portal, ss, cfg=None, freq="1d", on_start=None, **attrs):
    st = make(freq, **attrs)(script, on_start=on_start)
    return Backtester(st, portal, cfg or zero_cost()).run(ss[0], ss[-1], raise_errors=True), st
