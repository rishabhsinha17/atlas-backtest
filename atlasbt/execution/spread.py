"""Bid-ask spread estimators from daily OHLC + floors + time-of-day profile.

Implemented from the papers:

* Corwin, S. & Schultz, P. (2012) "A Simple Way to Estimate Bid-Ask Spreads
  from Daily High and Low Prices", Journal of Finance 67(2), eqs. (14), (18)
  and the overnight-return adjustment of their Section II.B: if the day t+1
  low is above the day t close, both H_{t+1} and L_{t+1} are lowered by the
  gap; if the day t+1 high is below the day t close both are raised. Negative
  two-day estimates are set to zero before averaging (their main results).
* Abdi, F. & Ranaldo, A. (2017) "A Simple Estimation of Bid-Ask Spreads from
  Daily Close, High, and Low Prices", Review of Financial Studies 30(12):
  eta_t = (ln H_t + ln L_t)/2, c_t = ln C_t,
  S^2 = 4 E[(c_t - eta_t)(c_t - eta_{t+1})]; monthly estimator
  S = sqrt(max(S^2, 0)); two-day corrected version averages
  sqrt(max(s_t^2, 0)).
* Roll, R. (1984) "A Simple Implicit Measure of the Effective Bid-Ask Spread",
  Journal of Finance 39(4): S = 2 sqrt(-cov(dp_t, dp_{t-1})) (0 if cov >= 0).

All estimators return *relative* (proportional) full spreads.
PIT: :class:`SpreadModel` returns for session i an estimate computed only from
sessions <= i-1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

_K = 3.0 - 2.0 * math.sqrt(2.0)


# --------------------------------------------------------------------------
# Corwin-Schultz
# --------------------------------------------------------------------------
def corwin_schultz_pair(h0: float, l0: float, h1: float, l1: float,
                        c0: Optional[float] = None, overnight_adjust: bool = True) -> float:
    """Two-day CS spread for days (t, t+1). Returns raw (possibly negative)."""
    if overnight_adjust and c0 is not None and np.isfinite(c0):
        if l1 > c0:
            g = l1 - c0
            h1, l1 = h1 - g, l1 - g
        elif h1 < c0:
            g = c0 - h1
            h1, l1 = h1 + g, l1 + g
    beta = math.log(h0 / l0) ** 2 + math.log(h1 / l1) ** 2
    gamma = math.log(max(h0, h1) / min(l0, l1)) ** 2
    alpha = (math.sqrt(2.0 * beta) - math.sqrt(beta)) / _K - math.sqrt(gamma / _K)
    return 2.0 * (math.exp(alpha) - 1.0) / (1.0 + math.exp(alpha))


def corwin_schultz(high: np.ndarray, low: np.ndarray, close: Optional[np.ndarray] = None,
                   overnight_adjust: bool = True) -> np.ndarray:
    """Vectorised two-day CS estimates. Element t is the estimate for the pair
    (t-1, t); element 0 is NaN. Values are raw (may be negative)."""
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    n = len(h)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    h0, l0 = h[:-1], l[:-1]
    h1, l1 = h[1:].copy(), l[1:].copy()
    if overnight_adjust and close is not None:
        c0 = np.asarray(close, dtype=float)[:-1]
        up = l1 > c0
        g = np.where(up, l1 - c0, 0.0)
        h1, l1 = h1 - g, l1 - g
        dn = h1 < c0
        g = np.where(dn, c0 - h1, 0.0)
        h1, l1 = h1 + g, l1 + g
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.log(h0 / l0) ** 2 + np.log(h1 / l1) ** 2
        gamma = np.log(np.maximum(h0, h1) / np.minimum(l0, l1)) ** 2
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
        s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    out[1:] = s
    return out


def _rolling_nanmean(x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Trailing mean over `window` ending at t (inclusive), ignoring NaN."""
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    xv = np.where(valid, x, 0.0)
    cs = np.concatenate([[0.0], np.cumsum(xv)])
    cn = np.concatenate([[0], np.cumsum(valid.astype(np.int64))])
    n = len(x)
    idx = np.arange(1, n + 1)
    lo = np.maximum(idx - window, 0)
    s = cs[idx] - cs[lo]
    c = cn[idx] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / c
    m[c < min_periods] = np.nan
    return m


def corwin_schultz_rolling(high, low, close=None, window: int = 21, min_periods: int = 5,
                           overnight_adjust: bool = True, clip_negative: bool = True) -> np.ndarray:
    s = corwin_schultz(high, low, close, overnight_adjust)
    if clip_negative:
        s = np.where(np.isfinite(s), np.maximum(s, 0.0), np.nan)
    return _rolling_nanmean(s, window, min_periods)


# --------------------------------------------------------------------------
# Abdi-Ranaldo
# --------------------------------------------------------------------------
def abdi_ranaldo_products(high, low, close) -> np.ndarray:
    """Element t = 4 (c_{t-1} - eta_{t-1}) (c_{t-1} - eta_t); element 0 NaN."""
    h = np.asarray(high, float); l = np.asarray(low, float); c = np.asarray(close, float)
    n = len(h)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = 0.5 * (np.log(h) + np.log(l))
        lc = np.log(c)
    out[1:] = 4.0 * (lc[:-1] - eta[:-1]) * (lc[:-1] - eta[1:])
    return out


def abdi_ranaldo_rolling(high, low, close, window: int = 21, min_periods: int = 5,
                         method: str = "monthly") -> np.ndarray:
    p = abdi_ranaldo_products(high, low, close)
    if method == "monthly":
        m = _rolling_nanmean(p, window, min_periods)
        return np.sqrt(np.where(np.isfinite(m), np.maximum(m, 0.0), np.nan))
    if method == "two_day":
        s = np.sqrt(np.where(np.isfinite(p), np.maximum(p, 0.0), np.nan))
        return _rolling_nanmean(s, window, min_periods)
    raise ValueError(method)


# --------------------------------------------------------------------------
# Roll
# --------------------------------------------------------------------------
def roll_spread(close, window: int = 21, min_periods: int = 10) -> np.ndarray:
    """Rolling Roll (1984) estimate on log price changes (relative spread).

    Element t uses the price changes ending at t (``window`` changes, i.e.
    ``window - 1`` lag pairs); sample covariance (ddof=1)."""
    c = np.asarray(close, float)
    n = len(c)
    out = np.full(n, np.nan)
    if n < 4:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        dp = np.diff(np.log(c))
    a = pd.Series(dp[1:])           # change into t = j + 2
    b = pd.Series(dp[:-1])
    mp = max(min(max(min_periods, 3), window) - 1, 2)
    cov = a.rolling(window - 1, min_periods=mp).cov(b).to_numpy()
    est = np.where(np.isfinite(cov), np.where(cov < 0, 2.0 * np.sqrt(np.abs(cov)), 0.0), np.nan)
    out[2:] = est
    return out


# --------------------------------------------------------------------------
# Floors and time-of-day profile
# --------------------------------------------------------------------------
def tick_size_for_price(price: float) -> float:
    """Reg NMS Rule 612: $0.01 quoting increment at/above $1, $0.0001 below."""
    return 0.01 if price >= 1.0 else 0.0001


#: (upper price bound, minimum relative full spread). Engineering defaults,
#: conservative for low-priced stocks; for liquid large caps the 1-tick floor
#: and the estimator dominate. Configurable.
DEFAULT_PRICE_BUCKET_FLOORS: tuple[tuple[float, float], ...] = (
    (1.0, 0.0100),     # < $1: 100 bp
    (5.0, 0.0030),     # $1-5: 30 bp
    (10.0, 0.0010),    # $5-10: 10 bp
    (20.0, 0.0004),    # $10-20: 4 bp
    (math.inf, 0.0001),  # >= $20: 1 bp
)


def price_bucket_floor(price: float, buckets=DEFAULT_PRICE_BUCKET_FLOORS) -> float:
    for ub, fl in buckets:
        if price < ub:
            return fl
    return buckets[-1][1]


#: (minutes since RTH open upper bound, multiplier) and (minutes to close upper
#: bound, multiplier). Spreads are U/L-shaped intraday: widest right after the
#: open, narrowing through the morning, slightly wider into the close
#: (McInish & Wood 1992 JF; Chung, Van Ness & Van Ness 1999 JFE). The
#: multipliers are engineering defaults relative to the daily average.
# Measured NBBO time-of-day profile (5-minute buckets relative to the RTH time-weighted
# average; docs/SPREADS.md, calibration json ``tod_profile``), coarsened for the legacy
# model, which has no ADV split: the 'illiquid' curve (first 5 min x2.9; liquid names
# x4.3 in the calibrated model). Spreads are NARROWER into the close, not wider.
DEFAULT_TOD_OPEN = ((5.0, 2.9), (10.0, 2.0), (15.0, 1.7), (30.0, 1.5), (60.0, 1.15))
DEFAULT_TOD_CLOSE = ((5.0, 0.8), (30.0, 0.83), (120.0, 0.87))


def tod_multiplier(minutes_since_open: Optional[float], minutes_to_close: Optional[float],
                   open_profile=DEFAULT_TOD_OPEN, close_profile=DEFAULT_TOD_CLOSE,
                   extended_hours_mult: float = 3.0) -> float:
    if minutes_since_open is None:
        return 1.0
    if minutes_since_open < 0 or (minutes_to_close is not None and minutes_to_close < 0):
        return extended_hours_mult
    for ub, m in open_profile:
        if minutes_since_open < ub:
            return m
    if minutes_to_close is not None:
        for ub, m in close_profile:
            if minutes_to_close <= ub:
                return m
    return 1.0


# --------------------------------------------------------------------------
# Spread model (per symbol, PIT)
# --------------------------------------------------------------------------
@dataclass
class SpreadConfig:
    method: str = "cs"                 # 'cs' | 'ar' | 'roll'
    window: int = 21
    min_periods: int = 5
    overnight_adjust: bool = True
    fallbacks: tuple[str, ...] = ("ar", "roll")
    bucket_floors: tuple = DEFAULT_PRICE_BUCKET_FLOORS
    tod_open: tuple = DEFAULT_TOD_OPEN
    tod_close: tuple = DEFAULT_TOD_CLOSE
    extended_hours_mult: float = 3.0
    default_spread: float = 0.0020     # used when no estimate is computable (flagged)
    max_spread: float = 0.10           # cap absurd estimates
    tick_floor: bool = True            # full spread >= 1 tick (disable only in mechanics tests)


_NAMES = {"cs": "Corwin-Schultz", "ar": "Abdi-Ranaldo", "roll": "Roll"}


class SpreadSeries:
    """PIT spread estimate per session index for one symbol."""

    def __init__(self, est: np.ndarray, src: np.ndarray):
        self.est = est      # relative full spread for session i (uses <= i-1)
        self.src = src      # object array of source names

    def at(self, i: int) -> tuple[float, str]:
        if i < 0 or i >= len(self.est):
            return float("nan"), "none"
        return float(self.est[i]), str(self.src[i])


class SpreadModel:
    def __init__(self, cfg: Optional[SpreadConfig] = None):
        self.cfg = cfg or SpreadConfig()

    def _estimate(self, method: str, h, l, c) -> np.ndarray:
        cfg = self.cfg
        if method == "cs":
            return corwin_schultz_rolling(h, l, c, cfg.window, cfg.min_periods, cfg.overnight_adjust)
        if method == "ar":
            return abdi_ranaldo_rolling(h, l, c, cfg.window, cfg.min_periods)
        if method == "roll":
            return roll_spread(c, cfg.window, max(cfg.min_periods, 10))
        raise ValueError(method)

    def series(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> SpreadSeries:
        """Inputs must be split-consistent (back-adjusted for splits; the
        estimators are scale invariant within each pair so this does not leak
        future information). Output index i uses sessions <= i-1."""
        n = len(high)
        est = np.full(n, np.nan)
        src = np.empty(n, dtype=object)
        src[:] = "default"
        order = (self.cfg.method,) + tuple(m for m in self.cfg.fallbacks if m != self.cfg.method)
        for m in order:
            if m != order[0] and np.isfinite(est[1:]).all():
                break                     # fallbacks only where the primary is missing
            e = self._estimate(m, high, low, close)
            # shift by one: estimate for session i from data through i-1
            sh = np.full(n, np.nan)
            sh[1:] = e[:-1]
            take = ~np.isfinite(est) & np.isfinite(sh) & (sh > 0)
            est[take] = np.minimum(sh[take], self.cfg.max_spread)
            src[take] = _NAMES[m]
        return SpreadSeries(est, src)

    def full_spread(self, estimate: float, price: float, tick: Optional[float] = None,
                    minutes_since_open: Optional[float] = None,
                    minutes_to_close: Optional[float] = None) -> float:
        """Relative full spread to apply at a given price/time, with floors."""
        cfg = self.cfg
        s = estimate if (estimate is not None and np.isfinite(estimate) and estimate > 0) else cfg.default_spread
        s = max(s, price_bucket_floor(price, cfg.bucket_floors))
        s *= tod_multiplier(minutes_since_open, minutes_to_close, cfg.tod_open, cfg.tod_close,
                            cfg.extended_hours_mult)
        tk = tick if tick is not None else tick_size_for_price(price)
        if price > 0 and cfg.tick_floor:
            s = max(s, tk / price)
        return min(s, cfg.max_spread * 3)
