"""Helpers for the swing-family tests (S046-S084).

All price data built here is SYNTHETIC and only exercises mechanics
(signal timing, order/fill timing, stop/target levels, causality).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from atlasbt.core.clock import nyse
from atlasbt.core.types import AssetClass, Instrument
from atlasbt.engine import BacktestConfig
from atlasbt.engine.backtester import run_backtest
from atlasbt.engine.strategy import Requirement
from atlasbt.engine.testing import InMemoryPortal, synthetic_bars_from_rows, synthetic_daily_bars
from atlasbt.execution.impact import SquareRootImpact
from atlasbt.execution.spread import SpreadConfig

CAL = nyse()
START = "2019-01-02"


def zero_cost(**kw) -> BacktestConfig:
    base = dict(fees="NO_FEES",
                spread=SpreadConfig(bucket_floors=((math.inf, 0.0),), default_spread=1e-12, max_spread=1e-12, tod_open=(),
                                    tod_close=(), tick_floor=False),
                impact=SquareRootImpact(k=0.0), debug=True)
    base.update(kw)
    return BacktestConfig(**base)


def sessions(n: int, start: str = START) -> list[pd.Timestamp]:
    s = pd.Timestamp(start)
    return list(CAL.sessions(s, s + pd.Timedelta(days=int(n * 1.6) + 30)))[:n]


def close_ts(d) -> pd.Timestamp:
    return CAL.open_close(d)[1]


def open_ts(d) -> pd.Timestamp:
    return CAL.open_close(d)[0]


# ------------------------------------------------------------------ synthetic rows
def rows_from_closes(closes, rng: float = 0.004, vol: float = 1_000_000.0, opens=None):
    rows = []
    prev = float(closes[0])
    for i, c in enumerate(closes):
        c = float(c)
        o = float(opens[i]) if opens is not None else prev
        rows.append((o, max(o, c) * (1 + rng), min(o, c) * (1 - rng), c, vol))
        prev = c
    return rows


def trend_closes(n: int, p0: float = 50.0, g: float = 0.002, amp: float = 0.01, per: float = 7.0, phase: float = 0.0):
    i = np.arange(n)
    return p0 * np.exp(g * i) * (1 + amp * np.sin(2 * np.pi * i / per + phase))


def flat_rows(n: int, p: float, half_range: float = 0.005, vol: float = 1_000_000.0):
    return [(p, p * (1 + half_range), p * (1 - half_range), p, vol) for _ in range(n)]


def cent_rows(rows):
    """Round OHLC to the cent grid (keeps H >= max(O, C) and L <= min(O, C))."""
    out = []
    for r in rows:
        o, c = round(float(r[0]), 2), round(float(r[3]), 2)
        h = max(round(float(r[1]), 2), o, c)
        l = min(round(float(r[2]), 2), o, c)
        out.append((o, h, l, c) + tuple(r[4:]))
    return out


def to_df(rows, start: str = START) -> pd.DataFrame:
    return synthetic_bars_from_rows(rows, "1d", session_dates=sessions(len(rows), start))


def arrays(rows):
    a = np.array([r[:5] for r in cent_rows(rows)], dtype=float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]


class SectorPortal(InMemoryPortal):
    """InMemoryPortal whose security master carries a 'sector' column."""

    def __init__(self, *a, sectors=None, **kw):
        super().__init__(*a, **kw)
        self._sectors = dict(sectors or {})

    def security_master(self, as_of):
        df = super().security_master(as_of)
        df["sector"] = [self._sectors.get(s) for s in df["symbol"]]
        return df


def portal_for(dfs: dict, **kw) -> InMemoryPortal:
    sectors = kw.pop("sectors", None)
    bars = {(s, "1d"): df for s, df in dfs.items()}
    if sectors is not None:
        return SectorPortal(bars=bars, sectors=sectors, **kw)
    return InMemoryPortal(bars=bars, **kw)


def with_universe(cls, syms, **attrs):
    """Same strategy logic on a test universe (the registered class is untouched)."""
    d = {"universe": list(syms), "universe_name": None, "requires": [Requirement("bars", "1d")]}
    d.update(attrs)
    return type(f"{cls.__name__}_T", (cls,), d)


def run(cls, portal, start, end, params=None, cfg=None):
    return run_backtest(cls(**(params or {})), portal, start, end, cfg or zero_cost(), raise_errors=True)


def orders(res) -> pd.DataFrame:
    return res.diagnostics["orders"]


def entries(res) -> pd.DataFrame:
    o = orders(res)
    return o[o["tag"] == "entry"] if len(o) else o


def fills_of(res, order_id) -> pd.DataFrame:
    f = res.fills
    return f[f["order_id"] == order_id]


def stop_in_force(res, after: pd.Timestamp, tag: str = "stop"):
    """Stop price of the latest protective order with ``tag`` submitted at or before ``after``."""
    o = orders(res)
    o = o[(o["tag"] == tag) & (o["submitted_at"] <= after)]
    if not len(o):
        return None
    return float(o.sort_values("order_id").iloc[-1]["stop_price"])


def limit_in_force(res, after: pd.Timestamp, tag: str = "target"):
    o = orders(res)
    o = o[(o["tag"] == tag) & (o["submitted_at"] <= after)]
    if not len(o):
        return None
    return float(o.sort_values("order_id").iloc[-1]["limit_price"])


def random_portal(syms, start="2016-01-04", end="2019-06-28", vol=0.018, sectors=None, extra=None):
    dfs = {}
    for i, s in enumerate(syms):
        dfs[s] = synthetic_daily_bars(start, end, start_price=40 + 7 * i, daily_vol=vol + 0.002 * (i % 4),
                                      seed=100 + i, drift=0.0004 * ((i % 3) - 1), volume=3_000_000.0)
    if extra:
        dfs.update(extra)
    return dfs


def vix_df(start="2016-01-04", end="2019-06-28", seed=7):
    """Synthetic VIX-like index bars known 15 minutes after the close (Cboe dissemination)."""
    d = synthetic_daily_bars(start, end, start_price=15.0, daily_vol=0.06, seed=seed)
    lvl = 12 + 25 * (np.sin(np.arange(len(d)) / 23.0) ** 8) + 3 * np.sin(np.arange(len(d)) / 3.1)
    d["close"] = lvl
    d["open"] = lvl
    d["high"] = lvl * 1.03
    d["low"] = lvl * 0.97
    d["volume"] = np.nan
    d["known_at"] = d["known_at"] + pd.Timedelta(minutes=15)
    return d


INDEX_INSTRUMENTS = {"VIX": Instrument("VIX", AssetClass.INDEX)}
