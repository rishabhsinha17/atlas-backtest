"""Generic causality harness (red team).

For every public indicator: value(t) computed on the full sample must equal
value(t) computed (a) on the prefix [0..t] and (b) on a copy whose future
(t+1..end) is perturbed WILDLY (x0.2..x5 multiplicative noise, injected NaNs,
volume x0.01..x100). 40 random cut points per function, on data with internal
NaN gaps. These tests are expected to PASS - they document what was checked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import atlasbt.indicators as ind
from atlasbt.indicators import streaming as st
from rt_data import daily_ohlcv, intraday_bars


def _frame(out) -> pd.DataFrame:
    if isinstance(out, pd.DataFrame):
        return out.reset_index(drop=True).astype(float)
    if isinstance(out, pd.Series):
        return out.reset_index(drop=True).astype(float).to_frame()
    if isinstance(out, dict):
        return pd.DataFrame({k: np.asarray(v, float) for k, v in out.items()})
    return pd.DataFrame({"v": np.asarray(out, float)})


def _wild(df: pd.DataFrame, t: int, rng) -> pd.DataFrame:
    p = df.copy()
    n = len(p) - t - 1
    if n <= 0:
        return p
    f = rng.uniform(0.2, 5.0, n)
    for c in ("open", "high", "low", "close"):
        if c in p:
            p.iloc[t + 1:, p.columns.get_loc(c)] = p[c].to_numpy()[t + 1:] * f * rng.uniform(0.97, 1.03, n)
    if {"open", "high", "low", "close"} <= set(p.columns):
        o, c = p["open"].to_numpy(), p["close"].to_numpy()
        p["high"] = np.fmax(p["high"].to_numpy(), np.fmax(o, c))
        p["low"] = np.fmin(p["low"].to_numpy(), np.fmin(o, c))
    if "volume" in p:
        p.iloc[t + 1:, p.columns.get_loc("volume")] = p["volume"].to_numpy()[t + 1:] * rng.uniform(0.01, 100, n)
    # random future NaN rows
    k = rng.integers(0, max(1, n // 10) + 1)
    if k:
        rows = t + 1 + rng.choice(n, size=min(k, n), replace=False)
        cols = [c for c in ("open", "high", "low", "close", "volume") if c in p]
        p.iloc[rows, [p.columns.get_loc(c) for c in cols]] = np.nan
    return p


def check_causal(fn, df, cuts, seed=0):
    base = _frame(fn(df))
    rng = np.random.default_rng(seed)
    for t in cuts:
        alt = _frame(fn(_wild(df, t, rng)))
        pd.testing.assert_frame_equal(base.iloc[:t + 1], alt.iloc[:t + 1], check_exact=False, rtol=1e-9, atol=1e-9)
        tr = _frame(fn(df.iloc[:t + 1]))
        pd.testing.assert_frame_equal(base.iloc[:t + 1], tr, check_exact=False, rtol=1e-9, atol=1e-9)


D = daily_ohlcv(420, seed=11, nan_at=(150, 151, 300))
CUTS = sorted(np.random.default_rng(1).choice(np.arange(5, 419), 40, replace=False))

DAILY = {
    "sma": lambda d: ind.sma(d.close, 20), "ema": lambda d: ind.ema(d.close, 12),
    "ema_alpha": lambda d: ind.ema(d.close, alpha=0.1), "wilder": lambda d: ind.wilder(d.close, 14),
    "rsi": lambda d: ind.rsi(d.close, 14, return_components=True), "rsi2c": lambda d: ind.rsi(d.close, 2, "cutler"),
    "macd": lambda d: ind.macd(d.close), "stoch": lambda d: ind.stochastic(d.high, d.low, d.close),
    "willr": lambda d: ind.williams_r(d.high, d.low, d.close), "cci": lambda d: ind.cci(d.high, d.low, d.close),
    "atr": lambda d: ind.atr(d.high, d.low, d.close), "bb": lambda d: ind.bollinger(d.close),
    "bwp": lambda d: ind.bandwidth_percentile(d.close, 20, 2, 60),
    "keltner": lambda d: ind.keltner(d.high, d.low, d.close), "squeeze": lambda d: ind.squeeze_on(d.high, d.low, d.close),
    "donchian": lambda d: ind.donchian(d.high, d.low, 20), "donch_inc": lambda d: ind.donchian(d.high, d.low, 20, True),
    "rv": lambda d: ind.realized_vol(d.close, 20, lag=1), "ovs": lambda d: ind.overnight_variance_share(d.open, d.close, 40),
    "pr": lambda d: ind.percentile_rank(d.close, 50), "pr_prev": lambda d: ind.percentile_rank(d.close, 50, False),
    "z": lambda d: ind.zscore(d.close, 20), "dmi": lambda d: ind.dmi(d.high, d.low, d.close),
    "psar": lambda d: ind.parabolic_sar(d.high, d.low), "st": lambda d: ind.supertrend(d.high, d.low, d.close),
    "ichi": lambda d: ind.ichimoku(d.high, d.low, d.close), "slope": lambda d: ind.linreg_slope(d.close, 20),
    "hr": lambda d: ind.high_ratio(d.close, d.high, 100), "ath": lambda d: ind.prior_all_time_high(d.high, 50),
    "nr7": lambda d: ind.nr_n(d.high, d.low, 7), "inside": lambda d: ind.inside_bar(d.high, d.low, 2),
    "ibs": lambda d: ind.ibs(d.high, d.low, d.close), "eng": lambda d: ind.bullish_engulfing(d.open, d.high, d.low, d.close),
    "streak": lambda d: ind.down_streak(d.close), "piv": lambda d: ind.pivots(d.high, d.low, 3, 5),
    "db": lambda d: ind.double_bottom(d.high, d.low, d.close, 2, 0.03, 3, 30),
    "hs": lambda d: ind.head_and_shoulders(d.high, d.low, d.close, 3, 14, 0.1, 3.0),
    "darvas": lambda d: ind.darvas_box(d.high, d.low, d.close), "vcp": lambda d: ind.volatility_contraction(d.high, d.low, d.close),
    "adv": lambda d: ind.adv(d.volume, 20, 1), "dv": lambda d: ind.dollar_volume(d.close, d.volume, 20),
    "ols": lambda d: ind.rolling_ols(d.close.pct_change(), d.open.pct_change(), 60),
    "kalman": lambda d: ind.kalman_hedge(d.close, d.open, 1e-3, None, True, 60),
}


@pytest.mark.parametrize("name", sorted(DAILY))
def test_rt_daily_causal_wild_perturbation(name):
    check_causal(DAILY[name], D, CUTS, seed=sum(map(ord, name)))


I = intraday_bars(24, seed=3, pre=True, freq_min=5)
INTRA = {
    "vwap": lambda b: ind.session_vwap(b), "vwap_ohlc4": lambda b: ind.session_vwap(b, "ohlc4"),
    "or30": lambda b: ind.opening_range(b, 30), "ext": lambda b: ind.session_extremes(b),
    "rvol": lambda b: ind.relative_volume(b, 10), "rvol_cum": lambda b: ind.relative_volume(b, 10, cumulative=True),
    "tod_med": lambda b: ind.time_of_day_profile(b, 5),
}


@pytest.mark.parametrize("name", sorted(INTRA))
def test_rt_intraday_causal_wild_perturbation(name):
    n = len(I)
    cuts = sorted(np.random.default_rng(2).choice(np.arange(200, n - 1), 12, replace=False))
    check_causal(INTRA[name], I, cuts, seed=7)


def test_rt_pca_residuals_uses_only_past_window():
    rng = np.random.default_rng(3)
    R = pd.DataFrame(rng.normal(0, 0.01, (260, 10)) + rng.normal(0, 0.01, (260, 1)))
    base = ind.pca_residuals(R, 120, 2)
    for t in (130, 200, 259):
        P = R.copy()
        if t + 1 < len(P):
            P.iloc[t + 1:] *= 50
        pd.testing.assert_frame_equal(base.iloc[:t + 1], ind.pca_residuals(P, 120, 2).iloc[:t + 1])
        pd.testing.assert_frame_equal(base.iloc[:t + 1], ind.pca_residuals(R.iloc[:t + 1], 120, 2))
