"""Momentum oscillators: RSI (Wilder / Cutler), MACD, Stochastics, Williams %R, CCI.

All functions are causal: the value at t uses observations <= t only.
References (worked spreadsheets stored as test fixtures in
tests/indicators/fixtures/stockcharts/):

* RSI - J. Welles Wilder Jr. (1978), *New Concepts in Technical Trading
  Systems*, ch. VI; StockCharts ChartSchool "Relative Strength Index"
  (cs-rsi.xls). Cutler's variant replaces Wilder smoothing by SMAs of gains
  and losses (no dependence on the start of the series).
* MACD - Gerald Appel; EMA(12) - EMA(26), signal EMA(9) of MACD, SMA-seeded
  EMAs (StockCharts convention).
* Stochastics - George Lane; fast %K = 100*(C-LL_n)/(HH_n-LL_n); slow %K =
  SMA_3(fast %K); %D = SMA_3(slow %K) (ChartSchool cs-soo.xls).
* Williams %R - Larry Williams; -100*(HH_n - C)/(HH_n - LL_n) (cs-percentr.xls).
* CCI - Donald Lambert (1980), *Commodities* magazine; TP=(H+L+C)/3,
  CCI = (TP - SMA_n(TP)) / (0.015 * MD_n), MD = mean absolute deviation of
  the n TPs around the *current* SMA (cs-cci.xls).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._util import as_float_array, check_period, index_of, rolling_apply_1d, rolling_max, rolling_min, wrap, wrap_frame
from .moving import _ewm_seeded, ema, sma


def _rsi_from_avgs(ag: np.ndarray, al: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = ag / al
        out = 100.0 - 100.0 / (1.0 + rs)
    # Edge cases (Wilder/StockCharts): no losses -> 100; no gains -> 0;
    # no movement at all -> 50 (neutral, documented choice).
    zero_loss = (al == 0) & ~np.isnan(ag)
    out = np.where(zero_loss & (ag > 0), 100.0, out)
    out = np.where(zero_loss & (ag == 0), 50.0, out)
    return out


def rsi(close, n: int = 14, method: str = "wilder", return_components: bool = False):
    """Relative Strength Index.

    method='wilder' (default): average gain/loss seeded with the simple mean
    of the first n changes (first value at index n), then Wilder smoothing.
    method='cutler': SMA_n of gains and losses.
    """
    n = check_period(n)
    c = as_float_array(close)
    d = np.full_like(c, np.nan)
    d[1:] = c[1:] - c[:-1]
    gain = np.where(np.isnan(d), np.nan, np.maximum(d, 0.0))
    loss = np.where(np.isnan(d), np.nan, np.maximum(-d, 0.0))
    if method == "wilder":
        ag = _ewm_seeded(gain, 1.0 / n, n, True)
        al = _ewm_seeded(loss, 1.0 / n, n, True)
    elif method == "cutler":
        ag = pd.Series(gain).rolling(n, min_periods=n).mean().to_numpy()
        al = pd.Series(loss).rolling(n, min_periods=n).mean().to_numpy()
    else:
        raise ValueError("method must be 'wilder' or 'cutler'")
    out = _rsi_from_avgs(ag, al)
    idx = index_of(close)
    if return_components:
        return wrap_frame({"avg_gain": ag, "avg_loss": al, "rsi": out}, idx)
    return wrap(out, idx, f"rsi{n}")


def macd(close, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line = EMA_fast - EMA_slow; signal = EMA_signal(MACD) (SMA-seeded on
    the first `signal` valid MACD values); hist = MACD - signal."""
    if fast >= slow:
        raise ValueError("fast must be < slow")
    c = as_float_array(close)
    m = ema(c, fast) - ema(c, slow)
    s = ema(m, signal)
    return wrap_frame({"macd": m, "signal": s, "hist": m - s}, index_of(close))


def stochastic(high, low, close, k: int = 14, smooth_k: int = 3, d: int = 3):
    """Stochastic oscillator (Lane).

    Returns columns: fast_k (raw %K), slow_k (= SMA_smooth_k(fast_k), i.e. the
    "%K" of the slow stochastic), slow_d (= SMA_d(slow_k)), fast_d (= SMA_d(fast_k)).
    If HH == LL over the window the raw %K is undefined -> NaN (documented).
    """
    k = check_period(k)
    h, l, c = as_float_array(high), as_float_array(low), as_float_array(close)
    hh, ll = rolling_max(h, k), rolling_min(l, k)
    rng = hh - ll
    with np.errstate(divide="ignore", invalid="ignore"):
        fk = np.where(rng > 0, 100.0 * (c - ll) / rng, np.nan)
    sk = sma(fk, smooth_k) if smooth_k > 1 else fk
    sd = sma(sk, d)
    fd = sma(fk, d)
    return wrap_frame({"fast_k": fk, "fast_d": fd, "slow_k": sk, "slow_d": sd}, index_of(high, low, close))


def williams_r(high, low, close, n: int = 14):
    """Williams %R in [-100, 0]."""
    n = check_period(n)
    h, l, c = as_float_array(high), as_float_array(low), as_float_array(close)
    hh, ll = rolling_max(h, n), rolling_min(l, n)
    rng = hh - ll
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(rng > 0, -100.0 * (hh - c) / rng, np.nan)
    return wrap(out, index_of(high, low, close), f"williams_r{n}")


def cci(high, low, close, n: int = 20, constant: float = 0.015, return_components: bool = False):
    """Commodity Channel Index (Lambert 1980) with mean absolute deviation."""
    n = check_period(n)
    tp = (as_float_array(high) + as_float_array(low) + as_float_array(close)) / 3.0
    ma = rolling_apply_1d(tp, n, lambda w: w.mean(axis=1))
    md = rolling_apply_1d(tp, n, lambda w: np.abs(w - w.mean(axis=1, keepdims=True)).mean(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(md > 0, (tp - ma) / (constant * md), np.nan)
    idx = index_of(high, low, close)
    if return_components:
        return wrap_frame({"tp": tp, "sma_tp": ma, "md": md, "cci": out}, idx)
    return wrap(out, idx, f"cci{n}")
