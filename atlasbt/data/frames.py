"""Canonical frame construction & validation (docs/ARCHITECTURE.md §3).

bars:   index `start` (UTC tz-aware), columns open, high, low, close, volume (float64),
        optional vwap, trades, `session` ('PRE'|'RTH'|'POST'|'DAY'), `known_at` (UTC).
        Daily: start = session open (UTC), known_at = actual session close (UTC; 13:00 ET
        on early closes).  Intraday: known_at = start + bar length.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

from atlasbt.core.clock import ET, NYSECalendar

BAR_COLS = ["open", "high", "low", "close", "volume"]
CHAIN_COLS = ["ts", "underlying", "symbol", "expiry", "strike", "right", "bid", "ask", "mid", "iv",
              "delta", "gamma", "vega", "theta", "underlying_price", "volume", "open_interest",
              "exercise", "settlement", "multiplier", "fidelity"]


@lru_cache(maxsize=1)
def calendar() -> NYSECalendar:
    """NYSE calendar spanning all history we may load (Yahoo goes back to 1962)."""
    return NYSECalendar(start="1960-01-01")


@lru_cache(maxsize=1)
def schedule() -> pd.DataFrame:
    """DataFrame indexed by naive session date with UTC `open`, `close`, `early`."""
    cal = calendar()
    sess = cal.all_sessions
    opens, closes = zip(*(cal.open_close(s) for s in sess))
    df = pd.DataFrame({"open": pd.DatetimeIndex(opens), "close": pd.DatetimeIndex(closes)}, index=sess)
    df["early"] = [cal.is_early_close(s) for s in sess]
    df.index.name = "date"
    return df


def freq_to_timedelta(freq: str) -> pd.Timedelta:
    m = re.fullmatch(r"(\d+)(m|min|h|d)", freq)
    if not m:
        raise ValueError(f"unsupported freq {freq!r}")
    n, u = int(m.group(1)), m.group(2)
    return pd.Timedelta(minutes=n) if u in ("m", "min") else pd.Timedelta(hours=n) if u == "h" else pd.Timedelta(days=n)


def to_naive_date(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_convert(ET).tz_localize(None)
    return idx.normalize()


def daily_bars(df: pd.DataFrame, extra_cols: tuple[str, ...] = ()) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Convert a date-indexed OHLCV frame to canonical daily bars.

    Returns (bars, dropped_dates) where dropped_dates are input dates that are
    not NYSE sessions (data errors / foreign holidays) and were removed.
    """
    dates = to_naive_date(df.index)
    sch = schedule()
    ok = dates.isin(sch.index)
    dropped = dates[~ok]
    src = df[ok]
    d = dates[ok]
    out = pd.DataFrame({c: pd.to_numeric(src[c], errors="coerce").astype("float64").to_numpy()
                        for c in BAR_COLS if c in src.columns})
    for c in extra_cols:
        if c in src.columns:
            out[c] = src[c].to_numpy()
    out["session"] = "DAY"
    out["known_at"] = sch.loc[d, "close"].to_numpy()
    out.index = pd.DatetimeIndex(sch.loc[d, "open"].to_numpy(), name="start")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out, dropped


def tag_sessions(start_utc: pd.DatetimeIndex) -> np.ndarray:
    """'PRE' / 'RTH' / 'POST' for intraday bar start times (UTC); '' if not a session day."""
    st = pd.DatetimeIndex(start_utc)
    d = to_naive_date(st)
    sch = schedule().reindex(d)
    o = pd.DatetimeIndex(sch["open"].to_numpy())
    c = pd.DatetimeIndex(sch["close"].to_numpy())
    st_v = st.tz_convert("UTC")
    tag = np.where(st_v < o, "PRE", np.where(st_v < c, "RTH", "POST"))
    tag = np.where(sch["open"].isna().to_numpy(), "", tag)
    return tag.astype(object)


def intraday_bars(df: pd.DataFrame, freq: str, extra_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    """Canonicalize an intraday OHLCV frame whose index is bar START (any tz)."""
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize(ET) if idx.tz is None else idx
    idx = idx.tz_convert("UTC")
    out = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce").astype("float64").to_numpy()
                        for c in BAR_COLS if c in df.columns}, index=pd.DatetimeIndex(idx, name="start"))
    for c in ("vwap", "trades") + tuple(extra_cols):
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce").to_numpy()
    out["session"] = tag_sessions(out.index)
    out = out[out["session"] != ""]
    out["known_at"] = out.index + freq_to_timedelta(freq)
    # an RTH bar cannot be known after the session close on early-close days etc.
    return out[~out.index.duplicated(keep="last")].sort_index()


def empty_bars() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in BAR_COLS})
    df["session"] = pd.Series(dtype=object)
    df["known_at"] = pd.Series(dtype="datetime64[ns, UTC]")
    df.index = pd.DatetimeIndex([], tz="UTC", name="start")
    return df


def empty_chain() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in CHAIN_COLS})


def validate_bars(df: pd.DataFrame) -> list[str]:
    """Structural checks of the canonical contract; returns list of problems."""
    probs = []
    if df.index.name != "start":
        probs.append("index not named 'start'")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        probs.append("index not tz-aware DatetimeIndex")
    for c in BAR_COLS + ["session", "known_at"]:
        if c not in df.columns:
            probs.append(f"missing column {c}")
    if "known_at" in df.columns and len(df) and (df["known_at"] <= df.index.to_series(index=df.index)).any():
        probs.append("known_at <= start for some rows")
    if not df.index.is_monotonic_increasing:
        probs.append("index not sorted")
    return probs
