"""Intraday (session-aware) indicators on canonical bar frames.

Input: a bars DataFrame as served by the DataPortal (index = bar START, UTC,
tz-aware; columns open, high, low, close, volume, optional vwap, session).
A row's value is known at the bar's END (start + bar length), so every
function here returns, at row t, a value computed from rows <= t only.

Session key = ET calendar date of the bar start. If a ``session`` column is
present it is used to select 'PRE'/'RTH'/'POST' bars; otherwise wall-clock ET
windows are used (RTH = [09:30, 16:00)).

References: VWAP - Berkowitz, Logue & Noser (1988) "The Total Cost of
Transactions on the NYSE", J. Finance 43(1); intraday volume seasonality /
relative volume - Jain & Joh (1988); the RVOL used here compares to the
median of the same time-of-day slot over the previous N sessions only.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

ET = "America/New_York"


# ---------------------------------------------------------------- helpers
def _check(bars: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError("bars must have a tz-aware DatetimeIndex (bar start, UTC)")
    if not bars.index.is_monotonic_increasing:
        raise ValueError("bars index must be sorted ascending")
    return bars


def et_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_convert(ET)


def session_key(idx: pd.DatetimeIndex) -> np.ndarray:
    """ET date (naive midnight Timestamp values) per bar."""
    return et_index(idx).tz_localize(None).normalize().to_numpy()


def minute_of_day(idx: pd.DatetimeIndex) -> np.ndarray:
    e = et_index(idx)
    return (e.hour * 60 + e.minute).to_numpy()


def _hhmm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def infer_bar_length(idx: pd.DatetimeIndex) -> pd.Timedelta:
    d = np.diff(pd.DatetimeIndex(idx).as_unit("ns").asi8)
    d = d[d > 0]
    if d.size == 0:
        raise ValueError("cannot infer bar length from fewer than two bars")
    return pd.Timedelta(int(np.median(d)), unit="ns")


def session_mask(bars: pd.DataFrame, sessions: Sequence[str] = ("RTH",), start: str = "09:30",
                 end: str = "16:00") -> np.ndarray:
    if "session" in bars.columns:
        return bars["session"].isin(list(sessions)).to_numpy()
    mod = minute_of_day(bars.index)
    if tuple(sessions) == ("RTH",):
        return (mod >= _hhmm(start)) & (mod < _hhmm(end))
    raise ValueError("bars has no 'session' column; only RTH can be inferred from clock times")


def _price(bars: pd.DataFrame, price: str) -> np.ndarray:
    h, l, c = (bars[k].to_numpy(float) for k in ("high", "low", "close"))
    if price == "typical":
        return (h + l + c) / 3.0
    if price == "close":
        return c
    if price == "ohlc4":
        return (bars["open"].to_numpy(float) + h + l + c) / 4.0
    if price == "bar_vwap":
        tp = (h + l + c) / 3.0
        if "vwap" not in bars.columns:
            return tp
        v = bars["vwap"].to_numpy(float)
        return np.where(np.isnan(v), tp, v)
    raise ValueError("price must be typical|close|ohlc4|bar_vwap")


# ---------------------------------------------------------------- VWAP family
def session_vwap(bars: pd.DataFrame, price: str = "typical", sessions: Sequence[str] = ("RTH",)) -> pd.DataFrame:
    """Session-anchored VWAP with volume-weighted standard deviation.

    vwap_t = sum(v_i p_i)/sum(v_i), i over this session's selected bars <= t
    sd_t   = sqrt(sum(v_i p_i^2)/sum(v_i) - vwap_t^2)  (volume-weighted, population)
    z_t    = (close_t - vwap_t)/sd_t
    ``price``: 'typical' (H+L+C)/3 (default), 'close', 'ohlc4', or 'bar_vwap'
    (the bar's own VWAP when the feed provides it - most faithful).
    Bars outside ``sessions`` get NaN. Resets every session.
    """
    _check(bars)
    sel = session_mask(bars, sessions)
    p = _price(bars, price)
    v = bars["volume"].to_numpy(float)
    v = np.where(sel & ~np.isnan(p), np.nan_to_num(v), 0.0)
    key = session_key(bars.index)
    # centre prices on the session's first selected price to avoid catastrophic
    # cancellation in E[p^2] - E[p]^2 (variance is shift-invariant)
    p0 = pd.Series(np.where(sel, p, np.nan)).groupby(key).transform("first").to_numpy()
    pc = np.nan_to_num(p - p0)
    df = pd.DataFrame({"k": key, "pv": v * pc, "pv2": v * pc ** 2, "v": v})
    cs = df.groupby("k", sort=False)[["pv", "pv2", "v"]].cumsum()
    with np.errstate(divide="ignore", invalid="ignore"):
        mc = cs["pv"].to_numpy() / cs["v"].to_numpy()
        var = cs["pv2"].to_numpy() / cs["v"].to_numpy() - mc ** 2
        vw = mc + p0
    sd = np.sqrt(np.clip(var, 0.0, None))
    vw = np.where(sel & (cs["v"].to_numpy() > 0), vw, np.nan)
    sd = np.where(np.isnan(vw), np.nan, sd)
    c = bars["close"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, (c - vw) / sd, np.nan)
    return pd.DataFrame({"vwap": vw, "sd": sd, "z": z, "cum_volume": np.where(sel, cs["v"].to_numpy(), np.nan)},
                        index=bars.index)


def vwap_bands(bars: pd.DataFrame, k: float = 2.0, price: str = "typical", sessions=("RTH",)) -> pd.DataFrame:
    out = session_vwap(bars, price, sessions)
    out["upper"] = out["vwap"] + k * out["sd"]
    out["lower"] = out["vwap"] - k * out["sd"]
    return out


def window_stats(bars: pd.DataFrame, start: str, end: str, price: str = "typical",
                 sessions: Sequence[str] | None = None) -> pd.DataFrame:
    """Per-session statistics of bars whose START lies in [start, end) ET.

    Returns a frame indexed by session date with columns high, low, vwap,
    volume, first_open, last_close, n_bars, known_at (UTC end of the last
    bar in the window - nothing here may be used before known_at).
    Typical uses: premarket high / VWAP / volume (07:00-09:30 or 04:00-09:30).
    """
    _check(bars)
    mod = minute_of_day(bars.index)
    m = (mod >= _hhmm(start)) & (mod < _hhmm(end))
    if sessions is not None and "session" in bars.columns:
        m &= bars["session"].isin(list(sessions)).to_numpy()
    b = bars.loc[m]
    if b.empty:
        return pd.DataFrame(columns=["high", "low", "vwap", "volume", "first_open", "last_close", "n_bars", "known_at"])
    bl = infer_bar_length(bars.index)
    p = _price(b, price)
    v = b["volume"].to_numpy(float)
    key = session_key(b.index)
    df = pd.DataFrame({"k": key, "high": b["high"].to_numpy(float), "low": b["low"].to_numpy(float),
                       "pv": p * v, "v": v, "open": b["open"].to_numpy(float), "close": b["close"].to_numpy(float),
                       "end": b.index + bl})
    g = df.groupby("k", sort=True)
    out = pd.DataFrame({"high": g["high"].max(), "low": g["low"].min(), "volume": g["v"].sum(),
                        "first_open": g["open"].first(), "last_close": g["close"].last(), "n_bars": g["v"].size(),
                        "known_at": g["end"].max()})
    with np.errstate(divide="ignore", invalid="ignore"):
        out["vwap"] = g["pv"].sum() / out["volume"].where(out["volume"] > 0)
    out.index.name = "session"
    return out


def premarket(bars: pd.DataFrame, start: str = "04:00", end: str = "09:30", price: str = "typical") -> pd.DataFrame:
    """Premarket high/low/VWAP/volume per session (needs PRE bars)."""
    return window_stats(bars, start, end, price, sessions=("PRE",) if "session" in bars.columns else None)


def opening_range(bars: pd.DataFrame, minutes: int = 5, open_time: str = "09:30",
                  bar_length: pd.Timedelta | None = None) -> pd.DataFrame:
    """Opening-range high/low over [open, open+minutes) broadcast to bars.

    Defined on a row only once the range is complete *as of that row's known
    time*: bar end (start + bar_length) >= open + minutes. Earlier rows: NaN.
    Columns: or_high, or_low, or_mid.
    """
    _check(bars)
    bl = bar_length or infer_bar_length(bars.index)
    mod = minute_of_day(bars.index)
    o = _hhmm(open_time)
    inwin = (mod >= o) & (mod < o + minutes)
    key = session_key(bars.index)
    df = pd.DataFrame({"k": key, "h": np.where(inwin, bars["high"].to_numpy(float), np.nan),
                       "l": np.where(inwin, bars["low"].to_numpy(float), np.nan)})
    hi = df.groupby("k")["h"].transform("max").to_numpy()
    lo = df.groupby("k")["l"].transform("min").to_numpy()
    end_min = mod + bl / pd.Timedelta(minutes=1)
    ready = end_min >= o + minutes
    hi = np.where(ready, hi, np.nan)
    lo = np.where(ready, lo, np.nan)
    return pd.DataFrame({"or_high": hi, "or_low": lo, "or_mid": 0.5 * (hi + lo)}, index=bars.index)


def session_extremes(bars: pd.DataFrame, sessions: Sequence[str] = ("RTH",)) -> pd.DataFrame:
    """Running session high/low including the current bar (hod/lod) and through
    the PRIOR bar (hod_prior/lod_prior, for "close > high of day so far")."""
    _check(bars)
    sel = session_mask(bars, sessions)
    key = session_key(bars.index)
    df = pd.DataFrame({"k": key, "h": np.where(sel, bars["high"].to_numpy(float), np.nan),
                       "l": np.where(sel, bars["low"].to_numpy(float), np.nan)})
    hod = df.groupby("k")["h"].cummax()
    lod = df.groupby("k")["l"].cummin()
    hp = pd.DataFrame({"k": key, "x": hod}).groupby("k")["x"].shift(1)
    lp = pd.DataFrame({"k": key, "x": lod}).groupby("k")["x"].shift(1)
    out = pd.DataFrame({"hod": hod.to_numpy(), "lod": lod.to_numpy(), "hod_prior": hp.to_numpy(),
                        "lod_prior": lp.to_numpy()}, index=bars.index)
    out.loc[~sel] = np.nan
    return out


# ---------------------------------------------------------------- relative volume
def time_of_day_profile(bars: pd.DataFrame, lookback: int = 20, cumulative: bool = False, stat: str = "median",
                        min_sessions: int | None = None, sessions: Sequence[str] = ("RTH",)) -> pd.Series:
    """Expected volume for each bar from the SAME time-of-day slot over the
    previous ``lookback`` sessions (current session excluded).

    cumulative=False: per-bar volume in that slot.
    cumulative=True : session-to-date cumulative volume up to the end of that
                      slot (S016 "cumulative volume vs same-time median").
    stat: 'median' (default) or 'mean', taken over the sessions in the window
    that actually have the slot. min_sessions defaults to ceil(0.75*lookback).
    A missing bar inside a session counts as 0 volume; slots after a session's
    last bar (early closes) are excluded rather than counted.
    """
    _check(bars)
    sel = session_mask(bars, sessions)
    b = bars.loc[sel]
    key = session_key(b.index)
    slot = minute_of_day(b.index)
    v = b["volume"].to_numpy(float)
    df = pd.DataFrame({"k": key, "s": slot, "v": v})
    piv = df.pivot_table(index="k", columns="s", values="v", aggfunc="sum").sort_index()
    # A slot with no bar is a no-trade interval (0 volume) when the session has
    # LATER bars; slots after a session's last bar (early close / halt to the
    # close) do not exist for that session and are left NaN so they are simply
    # excluded from the median, never blanking later sessions.
    raw_valid = piv.notna()
    later_valid = piv.bfill(axis=1).notna()
    piv = piv.fillna(0.0).where(raw_valid | later_valid)
    if cumulative:
        piv = piv.fillna(0.0).cumsum(axis=1).where(raw_valid | later_valid)
    mp = int(np.ceil(0.75 * lookback)) if min_sessions is None else int(min_sessions)
    roll = piv.rolling(lookback, min_periods=mp)
    prof = (roll.median() if stat == "median" else roll.mean()).shift(1)
    st = prof.stack()
    st.index.names = ["k", "s"]
    exp = pd.Series(st, name="expected")
    look = pd.MultiIndex.from_arrays([key, slot], names=["k", "s"])
    vals = exp.reindex(look).to_numpy()
    out = pd.Series(np.nan, index=bars.index, name="expected_volume")
    out.loc[sel] = vals
    return out


def relative_volume(bars: pd.DataFrame, lookback: int = 20, cumulative: bool = False, stat: str = "median",
                    min_sessions: int | None = None, sessions: Sequence[str] = ("RTH",)) -> pd.DataFrame:
    """RVOL = actual / expected (same slot, prior sessions only).

    cumulative=True compares session-to-date cumulative volume with the
    historical median cumulative volume at the same time (e.g. S001's
    "09:35 RVOL(5m)" = cumulative volume through the 09:34 bar vs the median of
    the same quantity over the previous 20 sessions).
    """
    sel = session_mask(bars, sessions)
    exp = time_of_day_profile(bars, lookback, cumulative, stat, min_sessions, sessions)
    v = bars["volume"].astype(float).where(sel)
    if cumulative:
        key = session_key(bars.index)
        act = v.fillna(0.0).groupby(key).cumsum().where(sel)
    else:
        act = v
    with np.errstate(divide="ignore", invalid="ignore"):
        rv = act / exp.where(exp > 0)
    return pd.DataFrame({"actual": act, "expected": exp, "rvol": rv}, index=bars.index)


# ---------------------------------------------------------------- daily liquidity
def adv(volume, n: int = 20, lag: int = 0):
    """Average daily volume (shares) over n sessions ending at t-lag.
    Use lag=1 inside a session (today's volume is incomplete)."""
    s = volume if isinstance(volume, pd.Series) else pd.Series(np.asarray(volume, float))
    out = s.astype(float).rolling(n, min_periods=n).mean()
    return out.shift(lag) if lag else out


def dollar_volume(close, volume, n: int = 20, lag: int = 0):
    """Average daily dollar volume (close*volume) over n sessions ending at t-lag."""
    c = close if isinstance(close, pd.Series) else pd.Series(np.asarray(close, float))
    v = volume if isinstance(volume, pd.Series) else pd.Series(np.asarray(volume, float), index=c.index)
    out = (c.astype(float) * v.astype(float)).rolling(n, min_periods=n).mean()
    return out.shift(lag) if lag else out


def minutes_since_open(bars: pd.DataFrame, open_time: str = "09:30", bar_length: pd.Timedelta | None = None):
    """Minutes from the session open to each bar's END (its known time)."""
    bl = bar_length or infer_bar_length(bars.index)
    return pd.Series(minute_of_day(bars.index) + bl / pd.Timedelta(minutes=1) - _hhmm(open_time),
                     index=bars.index, name="minutes_since_open")
