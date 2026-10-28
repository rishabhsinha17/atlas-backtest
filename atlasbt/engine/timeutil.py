"""Small time helpers shared by engine modules (int64 ns fast paths)."""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from ..core.clock import ET

_FREQ_RE = re.compile(r"^(\d+)\s*(m|min|t|h|d|s)$", re.I)

INTRADAY_FREQS = ("1m", "2m", "5m", "10m", "15m", "30m", "60m", "1h")
#: base frequencies that run on the daily timeline (monthly/event strategies
#: resample daily data causally; ARCHITECTURE §3)
DAILY_ALIASES = {"1d", "d", "day", "daily", "1mo", "mo", "monthly", "1w", "w", "weekly", "event"}


@lru_cache(maxsize=256)
def freq_to_timedelta(freq: str) -> pd.Timedelta:
    f = freq.strip().lower()
    if f in DAILY_ALIASES:
        return pd.Timedelta(days=1)
    m = _FREQ_RE.match(f)
    if not m:
        raise ValueError(f"unsupported frequency {freq!r}")
    n, u = int(m.group(1)), m.group(2).lower()
    if u in ("m", "min", "t"):
        return pd.Timedelta(minutes=n)
    if u == "h":
        return pd.Timedelta(hours=n)
    if u == "s":
        return pd.Timedelta(seconds=n)
    return pd.Timedelta(days=n)


@lru_cache(maxsize=256)
def is_daily(freq: str) -> bool:
    return freq.strip().lower() in DAILY_ALIASES


@lru_cache(maxsize=256)
def canonical_freq(freq: str) -> str:
    f = freq.strip().lower()
    if f in DAILY_ALIASES:
        return "1d"
    td = freq_to_timedelta(f)
    mins = int(td / pd.Timedelta(minutes=1))
    return f"{mins}m"


def ns(ts) -> int:
    return int(pd.Timestamp(ts).value)


def ts_utc(v: int) -> pd.Timestamp:
    return pd.Timestamp(v, tz="UTC")


def session_label(session) -> pd.Timestamp:
    t = pd.Timestamp(session)
    if t.tzinfo is not None:
        t = t.tz_convert(ET).tz_localize(None)
    return t.normalize()


def utc_index(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx.as_unit("ns")


def as_ns_array(idx) -> np.ndarray:
    """int64 nanoseconds since epoch (pandas 3 indexes may be in 'us')."""
    return utc_index(idx).asi8.astype(np.int64)
