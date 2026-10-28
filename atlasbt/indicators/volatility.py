"""Volatility / band indicators.

* ``true_range``: TR_t = max(H_t, C_{t-1}) - min(L_t, C_{t-1}); TR_0 = H_0 - L_0
  (no prior close). Wilder (1978) ch. III.
* ``atr``: Wilder ATR (default): first value = mean(TR_0..TR_{n-1}) at index
  n-1, then ATR_t = (ATR_{t-1}*(n-1) + TR_t)/n. This matches the StockCharts
  worked spreadsheet cs-atr.xls exactly. ``method='sma'`` gives the plain
  n-bar mean of TR (used by S061 "10-day true-range means").
* ``bollinger``: middle = SMA_n(close); sigma = **population** standard
  deviation (ddof=0) of the same n closes; bands = middle +/- k*sigma.
  Bollinger ("Bollinger on Bollinger Bands", 2001, ch. 7) specifies the
  population formula; StockCharts cs-bbands.xls uses it too (tested).
  bandwidth = (upper - lower)/middle (a fraction; StockCharts shows x100);
  %b = (close - lower)/(upper - lower).
* ``keltner``: middle = EMA_n(close) (SMA seeded), bands = middle +/- k*ATR_m
  (Wilder ATR, m defaults to n; Chester Keltner 1960 / Linda Raschke modern form).
* ``donchian``: channel of the prior n bars **excluding** the current bar by
  default (the spec's "prior N-day high", e.g. S046 C_t > max(H_{t-20:t-1})).
* ``realized_vol``: rolling sample stdev (ddof=1) of simple (or log) returns,
  optionally annualised by sqrt(periods); ``lag=1`` gives the "prior completed
  days" version.
* ``percentile_rank``: share of a trailing reference window strictly below the
  current value plus half the ties (mid-rank), in [0, 1].
* ``zscore``: (x - SMA_n) / stdev_n with population stdev by default so that
  z <= -k is identical to "close below the lower Bollinger(n, k) band".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._util import (as_float_array, check_period, index_of, rolling_max, rolling_min, shift,
                    wrap, wrap_frame)
from .moving import _ewm_seeded, ema, sma


def true_range(high, low, close):
    h, l, c = as_float_array(high), as_float_array(low), as_float_array(close)
    pc = shift(c, 1)
    hi = np.where(np.isnan(pc), h, np.maximum(h, pc))
    lo = np.where(np.isnan(pc), l, np.minimum(l, pc))
    return wrap(hi - lo, index_of(high, low, close), "tr")


def atr(high, low, close, n: int = 14, method: str = "wilder"):
    """Average True Range. method in {'wilder', 'sma', 'ema'}."""
    n = check_period(n)
    tr = as_float_array(true_range(high, low, close))
    if method == "wilder":
        out = _ewm_seeded(tr, 1.0 / n, n, True)
    elif method == "sma":
        out = pd.Series(tr).rolling(n, min_periods=n).mean().to_numpy()
    elif method == "ema":
        out = as_float_array(ema(tr, n))
    else:
        raise ValueError("method must be wilder|sma|ema")
    return wrap(out, index_of(high, low, close), f"atr{n}")


def rolling_std(x, n: int, ddof: int = 0):
    a = as_float_array(x)
    out = pd.Series(a).rolling(n, min_periods=n).std(ddof=ddof).to_numpy()
    return wrap(out, index_of(x), f"std{n}")


def bollinger(close, n: int = 20, k: float = 2.0, ddof: int = 0):
    """Bollinger Bands; returns mid, std, upper, lower, bandwidth, pct_b."""
    n = check_period(n)
    c = as_float_array(close)
    mid = as_float_array(sma(c, n))
    sd = as_float_array(rolling_std(c, n, ddof))
    up, lo = mid + k * sd, mid - k * sd
    with np.errstate(divide="ignore", invalid="ignore"):
        bw = (up - lo) / mid
        pb = np.where(up > lo, (c - lo) / (up - lo), np.nan)
    return wrap_frame({"mid": mid, "std": sd, "upper": up, "lower": lo, "bandwidth": bw, "pct_b": pb},
                      index_of(close))


def keltner(high, low, close, n: int = 20, k: float = 1.5, atr_n: int | None = None):
    c = as_float_array(close)
    mid = as_float_array(ema(c, n))
    a = as_float_array(atr(high, low, close, atr_n or n))
    return wrap_frame({"mid": mid, "atr": a, "upper": mid + k * a, "lower": mid - k * a},
                      index_of(high, low, close))


def squeeze_on(high, low, close, n: int = 20, bb_k: float = 2.0, kc_k: float = 1.5, atr_n: int | None = None):
    """True when the Bollinger(n, bb_k) bands lie fully inside Keltner(n, kc_k)."""
    bb = bollinger(close, n, bb_k)
    kc = keltner(high, low, close, n, kc_k, atr_n)
    bu, bl = as_float_array(bb["upper"]), as_float_array(bb["lower"])
    ku, kl = as_float_array(kc["upper"]), as_float_array(kc["lower"])
    on = (bu < ku) & (bl > kl)
    on = np.where(np.isnan(bu) | np.isnan(ku), False, on)
    return wrap(on, index_of(high, low, close), "squeeze_on")


def donchian(high, low, n: int = 20, include_current: bool = False):
    """Donchian channel (upper = highest high, lower = lowest low).

    include_current=False (default): uses bars t-n .. t-1 ("prior n bars").
    """
    n = check_period(n)
    h, l = as_float_array(high), as_float_array(low)
    up, lo = rolling_max(h, n), rolling_min(l, n)
    if not include_current:
        up, lo = shift(up, 1), shift(lo, 1)
    return wrap_frame({"upper": up, "lower": lo, "mid": 0.5 * (up + lo)}, index_of(high, low))


def returns(close, log: bool = False):
    c = as_float_array(close)
    pc = shift(c, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(c / pc) if log else c / pc - 1.0
    return wrap(r, index_of(close), "ret")


def realized_vol(close=None, n: int = 20, annualize: int | None = 252, log: bool = False, ddof: int = 1,
                 lag: int = 0, rets=None):
    """Rolling realised volatility of daily returns.

    Pass either prices (``close``) or precomputed returns (``rets``).
    annualize=None returns the per-period sigma. lag=1 -> uses returns up to
    t-1 only ("prior completed sessions", S032/S033).
    """
    n = check_period(n)
    r = as_float_array(rets) if rets is not None else as_float_array(returns(close, log))
    sd = pd.Series(r).rolling(n, min_periods=n).std(ddof=ddof).to_numpy()
    if annualize:
        sd = sd * np.sqrt(annualize)
    if lag:
        sd = shift(sd, lag)
    return wrap(sd, index_of(rets if rets is not None else close), f"rv{n}")


def vol_scaled_to_time(sigma_daily, minutes_elapsed, session_minutes: float = 390.0,
                       overnight_share: float = 0.0):
    """Scale a daily sigma to the horizon elapsed so far in a session.

    sigma_h = sigma_daily * sqrt(overnight_share + (1-overnight_share) * f),
    f = clip(minutes_elapsed / session_minutes, 0, 1). With overnight_share=0
    (default) this is the plain square-root-of-time rule from the open. When
    the move is measured from the *prior close* (S026), set overnight_share to
    the share of daily variance realised overnight (estimate it causally, e.g.
    with ``overnight_variance_share``).
    """
    f = np.clip(np.asarray(minutes_elapsed, dtype=float) / session_minutes, 0.0, 1.0)
    return np.asarray(sigma_daily, dtype=float) * np.sqrt(overnight_share + (1.0 - overnight_share) * f)


def overnight_variance_share(open_, close, n: int = 252):
    """Causal rolling estimate of var(close->open) / var(close->close)."""
    o, c = as_float_array(open_), as_float_array(close)
    pc = shift(c, 1)
    on = np.log(o / pc)
    cc = np.log(c / pc)
    v_on = pd.Series(on).rolling(n, min_periods=n).var().to_numpy()
    v_cc = pd.Series(cc).rolling(n, min_periods=n).var().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.clip(v_on / v_cc, 0.0, 1.0)
    return wrap(out, index_of(open_, close), "overnight_share")


def percentile_rank(x, window: int, include_current: bool = True, min_periods: int | None = None):
    """Mid-rank percentile of x_t within a trailing window.

    include_current=True : reference set = x_{t-window+1..t} (contains x_t,
                           which counts as half a tie).
    include_current=False: reference set = x_{t-window..t-1} ("previous N values").
    Result in [0, 1]; NaN if x_t is NaN or fewer than min_periods valid refs.
    """
    window = check_period(window, "window")
    a = as_float_array(x)
    m = a.shape[0]
    mp = window if min_periods is None else int(min_periods)
    out = np.full(m, np.nan)
    if include_current:
        if m >= window:
            w = np.lib.stride_tricks.sliding_window_view(a, window)  # rows end at t=window-1..m-1
            cur = a[window - 1:]
            valid = ~np.isnan(w)
            below = np.sum((w < cur[:, None]) & valid, axis=1)
            ties = np.sum((w == cur[:, None]) & valid, axis=1)
            cnt = valid.sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = (below + 0.5 * ties) / cnt
            r = np.where((cnt >= mp) & ~np.isnan(cur), r, np.nan)
            out[window - 1:] = r
    else:
        if m > window:
            w = np.lib.stride_tricks.sliding_window_view(a[:-1], window)  # rows end at t-1 for t=window..m-1
            cur = a[window:]
            valid = ~np.isnan(w)
            below = np.sum((w < cur[:, None]) & valid, axis=1)
            ties = np.sum((w == cur[:, None]) & valid, axis=1)
            cnt = valid.sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = (below + 0.5 * ties) / cnt
            r = np.where((cnt >= mp) & ~np.isnan(cur), r, np.nan)
            out[window:] = r
    return wrap(out, index_of(x), "pct_rank")


def bandwidth_percentile(close, n: int = 20, k: float = 2.0, lookback: int = 252):
    """Percentile of today's Bollinger bandwidth vs the *previous* `lookback`
    bandwidth values (S056 "bottom 10% of previous 252 values")."""
    bw = bollinger(close, n, k)["bandwidth"]
    return percentile_rank(bw, lookback, include_current=False)


def zscore(x, n: int = 20, ddof: int = 0):
    """(x_t - SMA_n) / stdev_n over the window ending at t (population stdev by default)."""
    a = as_float_array(x)
    mu = as_float_array(sma(a, n))
    sd = as_float_array(rolling_std(a, n, ddof))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, (a - mu) / sd, np.nan)
    return wrap(z, index_of(x), f"z{n}")
