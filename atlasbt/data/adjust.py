"""Corporate-action arithmetic.

* `unadjust_yahoo`: Yahoo's `auto_adjust=False` OHLC are *split-adjusted to
  today* (not dividend-adjusted) and volume / dividends are split-adjusted too.
  We restore as-traded prices: raw(t) = yahoo(t) * prod(ratio_s for splits s
  with ex_date > t); raw volume = yahoo volume / same factor; raw dividend =
  yahoo dividend * prod(ratio_s for splits with ex_date > dividend ex_date).

* `adjustment_factors`: point-in-time total-return back-adjustment.  Given raw
  daily bars and the actions *visible at as_of* (ex_date <= as_of session and
  known_at <= as_of), returns per-bar price multipliers so that adjusted series
  have continuous total returns and are expressed in the share basis of as_of:
      split  (ratio r, ex e): bars with date < e:  price /= r, volume *= r
      cash dividend (D, ex e): bars with date < e:  price *= (1 - D / C_prev)
          where C_prev is the RAW close of the session before e (CRSP method).
  Actions with ex_date after as_of never touch history -> no leakage.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from atlasbt.core.types import ActionType, CorporateAction

PRICE_COLS = ("open", "high", "low", "close", "vwap")
DIVIDEND_TYPES = (ActionType.CASH_DIVIDEND, ActionType.SPECIAL_DIVIDEND, ActionType.SPINOFF)   # SPINOFF: amount = value spun off per share


def split_factor_after(dates: pd.DatetimeIndex, split_dates: pd.DatetimeIndex, ratios: np.ndarray) -> np.ndarray:
    """For each date d: prod(ratios[split_date > d]). Dates naive (ET dates)."""
    dates = pd.DatetimeIndex(dates)
    if len(split_dates) == 0:
        return np.ones(len(dates))
    order = np.argsort(split_dates.values)
    sd = split_dates.values[order]
    r = np.asarray(ratios, dtype=float)[order]
    # suffix products: cum[i] = prod(r[i:])
    suffix = np.append(np.cumprod(r[::-1])[::-1], 1.0)
    pos = np.searchsorted(sd, dates.values, side="right")   # first split strictly after d
    return suffix[pos]


def unadjust_yahoo(df: pd.DataFrame, dates: pd.DatetimeIndex, splits: pd.Series) -> pd.DataFrame:
    """df: yahoo frame (open/high/low/close/volume[, adjclose]) aligned with naive `dates`.
    splits: Series ratio indexed by naive ex_date."""
    splits = splits[splits > 0]
    f = split_factor_after(dates, pd.DatetimeIndex(splits.index), splits.to_numpy())
    out = df.copy()
    for c in ("open", "high", "low", "close"):
        if c in out:
            out[c] = out[c].to_numpy() * f
    if "volume" in out:
        out["volume"] = out["volume"].to_numpy() / f
    return out


def actions_to_frame(actions: Iterable[CorporateAction]) -> pd.DataFrame:
    rows = [{"symbol": a.symbol, "type": a.type.value, "ex_date": a.ex_date, "known_at": a.known_at,
             "ratio": a.ratio, "amount": a.amount, "pay_date": a.pay_date, "new_symbol": a.new_symbol,
             "fidelity": int(a.fidelity)} for a in actions]
    return pd.DataFrame(rows)


def _naive(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return (t.tz_convert("America/New_York").tz_localize(None) if t.tzinfo else t).normalize()


def _known_on_or_after(known_at, ex) -> bool:
    """True when the action became known at/after the ex-date's close (value-based spin-offs);
    earlier-known amounts use the previous close."""
    if known_at is None or pd.isna(known_at):
        return False
    k = pd.Timestamp(known_at)
    k = k.tz_convert("America/New_York").tz_localize(None) if k.tzinfo else k
    return k >= pd.Timestamp(ex).normalize() + pd.Timedelta(hours=12, minutes=59)


def adjustment_factors(dates: pd.DatetimeIndex, raw_close: pd.Series,
                       actions: Iterable[CorporateAction]) -> tuple[np.ndarray, np.ndarray]:
    """Return (price_factor, volume_factor) arrays aligned with `dates` (naive session dates).

    `raw_close` indexed by the same naive dates (used for dividend factors).
    `actions` must already be filtered to those visible at as_of.
    """
    dates = pd.DatetimeIndex(dates)
    actions = list(actions)
    n = len(dates)
    pf = np.ones(n)
    vf = np.ones(n)
    closes = raw_close.to_numpy(dtype=float)
    for a in actions:
        ex = pd.Timestamp(a.ex_date)
        ex = (ex.tz_convert("America/New_York").tz_localize(None) if ex.tzinfo else ex).normalize()
        k = np.searchsorted(dates.values, ex.to_datetime64(), side="left")   # bars [0, k) are before ex
        if k == 0:
            continue
        if a.type == ActionType.SPLIT and a.ratio and a.ratio > 0 and a.ratio != 1.0:
            pf[:k] /= a.ratio
            vf[:k] *= a.ratio
        elif a.type == ActionType.SPINOFF and a.amount > 0 and _known_on_or_after(a.known_at, ex) and \
                k < n and dates[k] == ex and np.isfinite(closes[k]):
            # value-based spin-off (amount = children's first ex-date close x ratio, known at that
            # close): factor = S / (S + D), S = parent ex-date close per PRE-ex share (same-day
            # reverse split folded in). Using the previous close instead (1 - D/P_prev) inflates
            # the ex-date return whenever the market moved that day (DD 2019-06-03, HPQ/HPE).
            r_same = np.prod([b.ratio for b in actions if b.type == ActionType.SPLIT and b.ratio and
                              _naive(b.ex_date) == ex])
            S = closes[k] * (r_same if r_same else 1.0)
            if S > 0:
                pf[:k] *= S / (S + a.amount)
        elif a.type in DIVIDEND_TYPES and a.amount > 0:
            prev_close = closes[k - 1]
            if np.isfinite(prev_close) and prev_close > a.amount:
                pf[:k] *= 1.0 - a.amount / prev_close
    return pf, vf


def apply_factors(bars: pd.DataFrame, pf: np.ndarray, vf: np.ndarray) -> pd.DataFrame:
    out = bars.copy()
    for c in PRICE_COLS:
        if c in out.columns:
            out[c] = out[c].to_numpy(dtype=float) * pf
    if "volume" in out.columns:
        out["volume"] = out["volume"].to_numpy(dtype=float) * vf
    return out
