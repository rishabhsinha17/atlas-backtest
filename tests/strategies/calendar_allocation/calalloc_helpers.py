"""Shared SYNTHETIC fixtures for the calendar/event and allocation strategy tests.

Everything here is fake data for testing MECHANICS only (timing, prices, sizing).
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import pandas as pd

from atlasbt.core.clock import et, nyse
from atlasbt.core.types import Fidelity
from atlasbt.engine import BacktestConfig
from atlasbt.engine.backtester import run_backtest
from atlasbt.engine.testing import InMemoryPortal, daily_from_intraday
from atlasbt.engine.timeutil import freq_to_timedelta
from atlasbt.execution.impact import SquareRootImpact
from atlasbt.execution.spread import SpreadConfig

CAL = nyse()
BIG_VOL = 1e8


def zero_cost(**kw) -> BacktestConfig:
    base = dict(fees="NO_FEES",
                spread=SpreadConfig(bucket_floors=((math.inf, 0.0),), default_spread=1e-12, tod_open=(),
                                    tod_close=(), tick_floor=False, max_spread=1e-12),
                impact=SquareRootImpact(k=0.0), debug=True)
    base.update(kw)
    return BacktestConfig(**base)


def sess(a, b) -> list:
    return list(CAL.sessions(pd.Timestamp(a), pd.Timestamp(b)))


def sessions_before(day, n) -> list:
    d = pd.Timestamp(day)
    i = CAL.session_index(d) if CAL.is_session(d) else CAL.session_index(CAL.next_session(d))
    return list(CAL.all_sessions[i - n:i])


def daily_frame(sessions, closes, spread: float = 1.0, opens=None, highs=None, lows=None,
                volume: float = BIG_VOL) -> pd.DataFrame:
    """Daily bars: open = previous close (or given), high/low = max/min(o,c) +/- spread."""
    closes = np.asarray(closes, float)
    o = np.asarray(opens, float) if opens is not None else np.concatenate([[closes[0]], closes[:-1]])
    h = np.asarray(highs, float) if highs is not None else np.maximum(o, closes) + spread
    l = np.asarray(lows, float) if lows is not None else np.minimum(o, closes) - spread
    oc = [CAL.open_close(s) for s in sessions]
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": closes, "volume": float(volume),
                         "session": "DAY", "known_at": pd.DatetimeIndex([c for _, c in oc])},
                        index=pd.DatetimeIndex([x for x, _ in oc], name="start"))


def intraday_session(session, base: float, freq: str = "1m", overrides: Optional[dict] = None,
                     volume: float = 1e6) -> pd.DataFrame:
    """Flat RTH bars at ``base`` (high/low = base +/- 0.01) with per-bar overrides
    {"HH:MM" (bar start ET): (o, h, l, c[, v])}."""
    o_, c_ = CAL.open_close(session)
    td = freq_to_timedelta(freq)
    starts = pd.date_range(o_, c_, freq=td, inclusive="left")
    n = len(starts)
    o = np.full(n, base)
    c = np.full(n, base)
    h = np.full(n, base + 0.01)
    l = np.full(n, base - 0.01)
    v = np.full(n, float(volume))
    pos = {t: i for i, t in enumerate(starts)}
    for hhmm, row in (overrides or {}).items():
        i = pos[et(session, hhmm)]
        o[i], h[i], l[i], c[i] = row[:4]
        if len(row) > 4:
            v[i] = row[4]
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v, "session": "RTH",
                         "known_at": starts + td}, index=pd.DatetimeIndex(starts, name="start"))


def intraday_frame(day_specs: dict, freq: str = "1m") -> pd.DataFrame:
    """{session: (base, overrides[, volume])} -> concatenated RTH bars."""
    parts = []
    for s, spec in sorted(day_specs.items()):
        base, ov = spec[0], spec[1]
        vol = spec[2] if len(spec) > 2 else 1e6
        parts.append(intraday_session(s, base, freq, ov, vol))
    return pd.concat(parts)


def combine_daily(hist: pd.DataFrame, intraday: pd.DataFrame) -> pd.DataFrame:
    """Daily bars = given history + daily aggregation of the intraday window."""
    agg = daily_from_intraday(intraday)
    agg["volume"] = BIG_VOL
    return pd.concat([hist[~hist.index.isin(agg.index)], agg]).sort_index()


class EventPortal(InMemoryPortal):
    """InMemoryPortal that also answers ``has('*', 'events', KIND)`` and
    ``has('*', 'fundamentals', FIELD)`` (test double for the requirement check)."""

    def has(self, symbol, kind, freq=None):
        if kind == "events":
            return any(e.kind == freq for e in self._events)
        if kind == "fundamentals":
            return self._fund is not None and freq in set(self._fund["field"])
        return super().has(symbol, kind, freq)

    def fidelity_of(self, symbol, kind, freq=None):
        if kind in ("events", "fundamentals"):
            return Fidelity.T2_PROXY if self.has(symbol, kind, freq) else Fidelity.NA
        return super().fidelity_of(symbol, kind, freq)


def truncate(bars: dict, cut: pd.Timestamp) -> dict:
    """Drop every bar not known at ``cut`` (known_at > cut)."""
    out = {}
    for k, df in bars.items():
        ka = pd.DatetimeIndex(df["known_at"])
        out[k] = df[ka <= cut]
    return out


def orders(res) -> pd.DataFrame:
    df = res.diagnostics["orders"]
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["symbol", "type", "quantity", "tag", "submitted_at", "stop_price",
                                     "limit_price", "order_id"])
    return df


def order_sig(res, cut) -> list:
    df = orders(res)
    if len(df) == 0:
        return []
    df = df[pd.DatetimeIndex(df["submitted_at"]) <= cut]
    # OCO/bracket children have their quantity reduced later by sibling fills: not compared
    return [(r.symbol, r.type,
             None if (isinstance(r.oco_group, str) or r.parent_id == r.parent_id and
                                            r.parent_id is not None) else float(r.quantity),
             r.tag, pd.Timestamp(r.submitted_at),
             None if r.stop_price is None or r.stop_price != r.stop_price else round(float(r.stop_price), 6),
             None if r.limit_price is None or r.limit_price != r.limit_price else round(float(r.limit_price), 6))
            for r in df.itertuples()]


def assert_no_lookahead(make_strategy: Callable, make_portal: Callable, start, end, cut,
                        config: Optional[BacktestConfig] = None):
    """Run on full data and on data truncated at ``cut``: identical orders up to ``cut``."""
    cfg = config or zero_cost(debug=False)
    full = run_backtest(make_strategy(), make_portal(None), start, end, cfg, raise_errors=True)
    trunc = run_backtest(make_strategy(), make_portal(cut), start, end, cfg, raise_errors=True)
    a, b = order_sig(full, cut), order_sig(trunc, cut)
    assert a == b
    assert full.diagnostics["lookahead_guard_violations"] == 0
    return full, a


def et_ts(day, hhmm) -> pd.Timestamp:
    return et(pd.Timestamp(day), hhmm)
