"""Moving averages: SMA, EMA (span/alpha conventions, SMA seeding), Wilder smoothing.

Conventions (see docs/ANALYTICS.md section "Moving averages"):

* ``sma(x, n)``: arithmetic mean of the n values ending at t (inclusive).
  NaN for t < first_valid + n - 1 and for any window containing a NaN.
* ``ema(x, n)``: alpha = 2/(n+1) ("span" convention, as in StockCharts and
  pandas ``ewm(span=n, adjust=False)``). Alternatively pass ``alpha``.
  **Seeding** (default ``seed='sma'``): the first EMA value is the SMA of the
  first n valid observations, placed at index first_valid+n-1; thereafter
  EMA_t = EMA_{t-1} + alpha*(x_t - EMA_{t-1}). ``seed='first'`` seeds with the
  first observation (pandas ``adjust=False``) and emits from the first valid
  value. Reference: StockCharts ChartSchool "Moving Averages - Simple and
  Exponential" worked spreadsheet (cs-movavg.xls); tested to 1e-10.
* ``wilder(x, n)``: Wilder's smoothing (a.k.a. RMA/SMMA), alpha = 1/n, seeded
  with the SMA of the first n values (Wilder 1978, "New Concepts in Technical
  Trading Systems", pp. 23-24, 38).
* Internal NaNs after warm-up: the output is NaN at that bar and the recursive
  state is carried unchanged (the NaN observation is skipped, never imputed).
"""
from __future__ import annotations

import numpy as np

from ._util import as_float_array, check_period, index_of, njit, wrap


@njit
def _ewm_seeded(a, alpha, n, seed_sma):
    out = np.full(a.shape[0], np.nan)
    m = a.shape[0]
    # locate first valid
    i0 = 0
    while i0 < m and np.isnan(a[i0]):
        i0 += 1
    if i0 >= m:
        return out
    if seed_sma:
        # need n consecutive valid values for the seed
        s = 0.0
        cnt = 0
        i = i0
        while i < m and cnt < n:
            if np.isnan(a[i]):
                # restart the seed window after an internal gap
                s = 0.0
                cnt = 0
            else:
                s += a[i]
                cnt += 1
            i += 1
        if cnt < n:
            return out
        prev = s / n
        out[i - 1] = prev
        start = i
    else:
        prev = a[i0]
        out[i0] = prev
        start = i0 + 1
    for t in range(start, m):
        x = a[t]
        if np.isnan(x):
            continue
        prev = prev + alpha * (x - prev)
        out[t] = prev
    return out


def sma(x, n: int, name: str | None = None):
    """Simple moving average over the trailing n observations (inclusive of t)."""
    n = check_period(n)
    a = as_float_array(x)
    import pandas as pd

    out = pd.Series(a).rolling(n, min_periods=n).mean().to_numpy()
    return wrap(out, index_of(x), name or f"sma{n}")


def ema(x, n: int | None = None, alpha: float | None = None, seed: str = "sma", name: str | None = None):
    """Exponential moving average.

    n      : span; alpha = 2/(n+1) unless ``alpha`` is given explicitly.
    seed   : 'sma' (default; first value = SMA of first n) or 'first'.
    """
    if alpha is None:
        if n is None:
            raise ValueError("give n (span) or alpha")
        n = check_period(n)
        alpha = 2.0 / (n + 1.0)
    else:
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        if n is None:
            n = max(1, int(round(2.0 / alpha - 1.0)))
    if seed not in ("sma", "first"):
        raise ValueError("seed must be 'sma' or 'first'")
    a = as_float_array(x)
    out = _ewm_seeded(a, float(alpha), int(n), seed == "sma")
    return wrap(out, index_of(x), name or f"ema{n}")


def wilder(x, n: int, name: str | None = None):
    """Wilder smoothing: alpha = 1/n, SMA seed (RMA / SMMA)."""
    n = check_period(n)
    a = as_float_array(x)
    out = _ewm_seeded(a, 1.0 / n, n, True)
    return wrap(out, index_of(x), name or f"wilder{n}")


@njit
def _wilder_sum(a, n, start):
    """Wilder's *running-sum* form used in the ADX worksheet:
    S_first = sum(a[start:start+n]) at index start+n-1;
    S_t = S_{t-1} - S_{t-1}/n + a_t.  (S = n * RMA.)"""
    m = a.shape[0]
    out = np.full(m, np.nan)
    if start + n > m:
        return out
    s = 0.0
    for i in range(start, start + n):
        s += a[i]
    out[start + n - 1] = s
    for t in range(start + n, m):
        if np.isnan(a[t]):
            continue
        s = s - s / n + a[t]
        out[t] = s
    return out
