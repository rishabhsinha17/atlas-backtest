"""Realized-volatility estimators and forecasts (S144-S147, S169).

All estimators are CAUSAL: the value at index t uses bars <= t only (it is
known at bar t's close).  Inputs: daily bars frame with open/high/low/close
(raw or adjusted consistently — use split/dividend-adjusted history so a split
is not a 50% "return").  Outputs are annualized vols (decimal), 252 sessions/yr.

Estimators: close-to-close, Parkinson (1980), Garman-Klass (1980),
Rogers-Satchell (1991), Yang-Zhang (2000).
Forecasts: EWMA (RiskMetrics, lambda=0.94) and HAR-RV (Corsi 2009) with
walk-forward OLS fitting (only targets fully realized by the fit date).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252.0


def _lr(a, b):
    return np.log(np.asarray(a, float) / np.asarray(b, float))


def close_to_close(close: pd.Series, window: int = 20, demean: bool = True) -> pd.Series:
    r = np.log(close).diff()
    if demean:
        v = r.rolling(window).var(ddof=1)
    else:
        v = (r * r).rolling(window).mean()
    return np.sqrt(v * ANN)


def parkinson(bars: pd.DataFrame, window: int = 20) -> pd.Series:
    hl = np.log(bars["high"] / bars["low"]) ** 2
    return np.sqrt(hl.rolling(window).mean() / (4 * np.log(2)) * ANN)


def garman_klass(bars: pd.DataFrame, window: int = 20) -> pd.Series:
    hl = np.log(bars["high"] / bars["low"]) ** 2
    co = np.log(bars["close"] / bars["open"]) ** 2
    v = 0.5 * hl - (2 * np.log(2) - 1) * co
    return np.sqrt(v.rolling(window).mean() * ANN)


def rogers_satchell(bars: pd.DataFrame, window: int = 20) -> pd.Series:
    h, l, o, c = (np.log(bars[x]) for x in ("high", "low", "open", "close"))
    v = (h - c) * (h - o) + (l - c) * (l - o)
    return np.sqrt(v.rolling(window).mean() * ANN)


def yang_zhang(bars: pd.DataFrame, window: int = 20) -> pd.Series:
    o, c = bars["open"], bars["close"]
    ro = np.log(o / c.shift(1))          # overnight
    rc = np.log(c / o)                   # open-to-close
    h, l = np.log(bars["high"]), np.log(bars["low"])
    lo_, lc = np.log(o), np.log(c)
    rs = (h - lc) * (h - lo_) + (l - lc) * (l - lo_)
    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    v = ro.rolling(n).var(ddof=1) + k * rc.rolling(n).var(ddof=1) + (1 - k) * rs.rolling(n).mean()
    return np.sqrt(v * ANN)


ESTIMATORS = {"cc": close_to_close, "parkinson": parkinson, "gk": garman_klass,
              "rs": rogers_satchell, "yz": yang_zhang}


def realized_vol(bars: pd.DataFrame, window: int = 20, method: str = "cc") -> pd.Series:
    if method == "cc":
        return close_to_close(bars["close"], window)
    return ESTIMATORS[method](bars, window)


def daily_variance(bars: pd.DataFrame, method: str = "gk") -> pd.Series:
    """One-day variance proxy (not annualized): 'cc' squared log return,
    'gk' Garman-Klass + overnight squared gap (so it covers close-to-close)."""
    if method == "cc":
        return np.log(bars["close"]).diff() ** 2
    hl = np.log(bars["high"] / bars["low"]) ** 2
    co = np.log(bars["close"] / bars["open"]) ** 2
    gk = 0.5 * hl - (2 * np.log(2) - 1) * co
    on = np.log(bars["open"] / bars["close"].shift(1)) ** 2
    return gk + on


def ewma_vol(returns: pd.Series, lam: float = 0.94, seed_window: int = 20) -> pd.Series:
    """RiskMetrics EWMA annualized vol; value at t uses returns <= t."""
    r = returns.astype(float)
    out = np.full(len(r), np.nan)
    v = np.nan
    vals = r.values
    for i in range(len(vals)):
        x = vals[i]
        if not np.isfinite(x):
            out[i] = np.sqrt(v * ANN) if np.isfinite(v) else np.nan
            continue
        if not np.isfinite(v):
            if i + 1 >= seed_window:
                w = vals[i + 1 - seed_window:i + 1]
                w = w[np.isfinite(w)]
                v = float(np.mean(w * w)) if len(w) else np.nan
        else:
            v = lam * v + (1 - lam) * x * x
        out[i] = np.sqrt(v * ANN) if np.isfinite(v) else np.nan
    return pd.Series(out, index=r.index)


def har_features(rv: pd.Series) -> pd.DataFrame:
    """HAR regressors at t: daily, weekly (5) and monthly (22) mean variance."""
    return pd.DataFrame({"d": rv, "w": rv.rolling(5).mean(), "m": rv.rolling(22).mean()})


def har_rv_forecast(rv: pd.Series, horizon: int = 21, min_train: int = 252, refit_every: int = 21,
                    window: int | None = None, log: bool = False) -> pd.DataFrame:
    """Walk-forward HAR-RV forecast of the average daily variance over the next
    ``horizon`` sessions, made at each t (known at t's close).

    Training at fit date t uses rows i with i + horizon <= t (targets entirely
    realized by t) -> no lookahead.  ``window`` = rolling training length (None
    = expanding).  Returns frame with columns: var_fcst (daily variance),
    vol_fcst (annualized), fit_date.
    """
    rv = rv.astype(float)
    X = har_features(rv)
    y = rv[::-1].rolling(horizon).mean()[::-1].shift(-1)  # mean of rv[t+1 .. t+h]
    if log:
        Xf, yf = np.log(X.clip(lower=1e-12)), np.log(y.clip(lower=1e-12))
    else:
        Xf, yf = X, y
    n = len(rv)
    pos = np.arange(n)
    out_var = np.full(n, np.nan)
    fit_date = np.full(n, None, dtype=object)
    beta = None
    last_fit = -10**9
    Xv = Xf.values
    yv = yf.values
    for t in range(n):
        if t - last_fit >= refit_every or beta is None:
            hi = t - horizon  # last usable training row index
            lo = 0 if window is None else max(0, hi - window + 1)
            if hi - lo + 1 >= min_train:
                rows = pos[lo:hi + 1]
                Xt, yt = Xv[rows], yv[rows]
                ok = np.all(np.isfinite(Xt), axis=1) & np.isfinite(yt)
                if ok.sum() >= min_train // 2:
                    A = np.c_[np.ones(ok.sum()), Xt[ok]]
                    beta, *_ = np.linalg.lstsq(A, yt[ok], rcond=None)
                    last_fit = t
        if beta is not None and np.all(np.isfinite(Xv[t])):
            pred = beta[0] + Xv[t] @ beta[1:]
            out_var[t] = np.exp(pred) if log else max(pred, 1e-12)
            fit_date[t] = rv.index[last_fit]
    return pd.DataFrame({"var_fcst": out_var, "vol_fcst": np.sqrt(out_var * ANN), "fit_date": fit_date},
                        index=rv.index)
