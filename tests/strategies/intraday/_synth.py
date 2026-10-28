"""SYNTHETIC fixtures for the intraday strategy tests (mechanics only - never
use these to evaluate a strategy).

A scenario = a few flat history sessions (constant volume -> known
time-of-day medians, ADV and prior close) + one hand-built pattern day whose
first bars are written explicitly and the rest is flat filler.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from atlasbt.core.clock import et, nyse
from atlasbt.core.types import RunStatus
from atlasbt.engine.backtester import run_backtest
from atlasbt.engine.config import BacktestConfig
from atlasbt.engine.testing import InMemoryPortal, daily_from_intraday
from atlasbt.execution.impact import SquareRootImpact
from atlasbt.execution.spread import SpreadConfig

CAL = nyse()
DAY = pd.Timestamp("2024-04-17")        # Wednesday, regular session
SYM = "XYZ"
V1 = 100_000.0                          # volume per 1m bar in history
V5 = 500_000.0                          # volume per 5m bar in history


def prior_sessions(day: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    ss = CAL.sessions(day - pd.Timedelta(days=3 * n + 10), day)
    ss = [s for s in ss if s < day]
    return ss[-n:]


def starts(session, freq_min: int, t0: str = "09:30", t1: Optional[str] = None) -> pd.DatetimeIndex:
    o, c = CAL.open_close(session)
    a = et(session, t0)
    b = et(session, t1) if t1 else c
    return pd.date_range(a, b, freq=f"{freq_min}min", inclusive="left")


def frame(rows, idx: pd.DatetimeIndex, freq_min: int, session: str = "RTH") -> pd.DataFrame:
    rows = list(rows)
    assert len(rows) == len(idx), (len(rows), len(idx))
    a = np.array([r[:4] for r in rows], float)
    v = np.array([r[4] for r in rows], float)
    return pd.DataFrame({"open": a[:, 0], "high": a[:, 1], "low": a[:, 2], "close": a[:, 3], "volume": v,
                         "session": session, "known_at": idx + pd.Timedelta(minutes=freq_min)},
                        index=pd.DatetimeIndex(idx, name="start"))


def bar(o, c, rng=0.05, v=V1, h=None, l=None):
    return (o, max(o, c) + rng if h is None else h, min(o, c) - rng if l is None else l, c, v)


def path(o0, closes, rng=0.05, v=V1):
    out, o = [], o0
    for c in closes:
        out.append(bar(o, c, rng, v))
        o = c
    return out


def flat(p, n, rng=0.05, v=V1, alt=0.0):
    """n bars around p; alt != 0 alternates closes p+alt / p-alt (non-zero returns)."""
    out = []
    prev = p
    for i in range(n):
        c = p + (alt if i % 2 == 0 else -alt)
        out.append((prev, max(prev, c) + rng, min(prev, c) - rng, c, v))
        prev = c
    return out


def history(sym_days, freq_min: int, price=100.0, v=None, alt=0.0, rng=0.05, prices=None) -> pd.DataFrame:
    """Flat RTH sessions (one price per session if `prices` given)."""
    v = v if v is not None else (V1 if freq_min == 1 else V5)
    fr = []
    for k, s in enumerate(sym_days):
        idx = starts(s, freq_min)
        p = prices[k] if prices is not None else price
        fr.append(frame(flat(p, len(idx), rng, v, alt), idx, freq_min))
    return pd.concat(fr)


def day_frame(session, rows, freq_min: int, fill=None, fill_rng=0.0, fill_v=None, t0="09:30") -> pd.DataFrame:
    """Explicit `rows` from t0, then flat filler bars (o=h=l=c=fill or last close) to the close."""
    idx = starts(session, freq_min, t0)
    rows = list(rows)
    last = rows[-1][3] if rows else fill
    p = last if fill is None else fill
    fv = fill_v if fill_v is not None else (V1 if freq_min == 1 else V5)
    filler = [(p, p + fill_rng, p - fill_rng, p, fv)] * (len(idx) - len(rows))
    return frame(rows + filler, idx, freq_min)


def pre_frame(session, rows_or_price, n=None, v=30_000.0, t0="04:00", t1="09:30") -> pd.DataFrame:
    idx = starts(session, 1, t0, t1)
    if isinstance(rows_or_price, (int, float)):
        rows = [(rows_or_price, rows_or_price + 0.05, rows_or_price - 0.05, rows_or_price, v)] * len(idx)
    else:
        rows = rows_or_price
    return frame(rows, idx, 1, session="PRE")


def daily_ramp(day_list, closes, rng=0.3, v=39_000_000.0, open_noise=None) -> pd.DataFrame:
    rows = []
    prev = closes[0]
    for k, c in enumerate(closes):
        o = prev if open_noise is None else prev * (1 + open_noise[k])
        rows.append((o, max(o, c) + rng, min(o, c) - rng, c, v))
        prev = c
    oc = [CAL.open_close(s) for s in day_list]
    a = np.array([r[:4] for r in rows])
    return pd.DataFrame({"open": a[:, 0], "high": a[:, 1], "low": a[:, 2], "close": a[:, 3],
                         "volume": [r[4] for r in rows], "session": "DAY",
                         "known_at": pd.DatetimeIndex([c for _, c in oc])},
                        index=pd.DatetimeIndex([o for o, _ in oc], name="start"))


# ======================================================================= portal / run
@dataclass
class Scenario:
    sid: str
    intraday: dict                       # (sym, freq) -> full intraday frame (history + pattern day)
    params: dict = field(default_factory=dict)
    daily_extra: dict = field(default_factory=dict)   # sym -> daily-only rows BEFORE the intraday history
    quotes: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    universe: tuple = (SYM,)
    universe_name: Optional[str] = None
    universes: dict = field(default_factory=dict)
    day: pd.Timestamp = DAY
    # expectations (None = do not check)
    signal: Optional[str] = None         # ET HH:MM of the entry order (= entry fill bar start)
    side: int = 1
    stop: Optional[float] = None
    target: Optional[float] = None       # fixed target
    target_r: Optional[float] = None     # target = fill + R*(fill - stop)
    symbol: str = SYM
    check: Optional[Callable] = None     # extra assertions check(result, scenario)


def build_portal(sc: Scenario, cut: Optional[pd.Timestamp] = None) -> InMemoryPortal:
    bars = {}
    for (sym, fq), df in sc.intraday.items():
        d = df if cut is None else df[df.index < cut]
        bars[(sym, fq)] = d
    daily_src = {}
    for (sym, fq), df in bars.items():
        if sym not in daily_src or fq == "1m":
            daily_src[sym] = df
    for sym, df in daily_src.items():
        d = daily_from_intraday(df)
        if sym in sc.daily_extra:
            d = pd.concat([sc.daily_extra[sym], d]).sort_index()
        bars[(sym, "1d")] = d
    quotes = {s: (q if cut is None else q[q.index < cut]) for s, q in sc.quotes.items()}
    return InMemoryPortal(bars=bars, quotes=quotes, events=sc.events, universes=sc.universes)


def zero_cost(**kw) -> BacktestConfig:
    base = dict(fees="NO_FEES",
                spread=SpreadConfig(bucket_floors=((math.inf, 0.0),), default_spread=1e-12, tod_open=(),
                                    tod_close=(), tick_floor=False, max_spread=0.0),
                impact=SquareRootImpact(k=0.0), debug=True, use_quotes="never")
    base.update(kw)
    return BacktestConfig(**base)


def strategy_for(cls, sc: Scenario, **params):
    T = type(cls.__name__ + "Test", (cls,), {"universe": tuple(sc.universe), "universe_name": sc.universe_name})
    p = dict(sc.params)
    p.update(params)
    return T(**p)


def run(cls, sc: Scenario, cut=None, config=None, **params):
    st = strategy_for(cls, sc, **params)
    return run_backtest(st, build_portal(sc, cut), sc.day, sc.day, config or zero_cost(), raise_errors=True)


def orders(r, tag=None) -> pd.DataFrame:
    o = r.diagnostics["orders"]
    if o is None or len(o) == 0:
        return pd.DataFrame(columns=["tag", "symbol", "submitted_at", "status"])
    return o if tag is None else o[o["tag"] == tag]


def bar_open_at(sc: Scenario, sym: str, ts: pd.Timestamp) -> float:
    for (s, fq), df in sc.intraday.items():
        if s == sym and ts in df.index:
            return float(df.loc[ts, "open"])
    raise KeyError(ts)


def assert_status_ok(r):
    assert r.status == RunStatus.OK, (r.status, r.not_testable_reason, r.diagnostics.get("traceback"))
    assert r.diagnostics.get("lookahead_guard_violations", 0) == 0
