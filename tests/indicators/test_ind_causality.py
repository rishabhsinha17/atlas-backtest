"""Causality tests for EVERY public indicator: perturbing (or appending) data
after bar t must leave every output at positions <= t unchanged."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import atlasbt.indicators as ind
from atlasbt.indicators import cross_section as xs
from atlasbt.indicators import streaming as st
from ind_data import synthetic_intraday, synthetic_ohlcv

CUTS = (120, 260, 399)


def _as_frame(out) -> pd.DataFrame:
    if isinstance(out, pd.DataFrame):
        return out.reset_index(drop=True).astype(float)
    if isinstance(out, pd.Series):
        return out.reset_index(drop=True).astype(float).to_frame()
    if isinstance(out, dict):
        return pd.DataFrame({k: np.asarray(v, float) for k, v in out.items()})
    return pd.DataFrame({"v": np.asarray(out, float)})


def _perturb(df: pd.DataFrame, t: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = df.copy()
    n = len(df) - t - 1
    if n <= 0:
        return p
    for c in ("open", "high", "low", "close"):
        if c in p:
            p.iloc[t + 1:, p.columns.get_loc(c)] = p[c].to_numpy()[t + 1:] * np.exp(rng.normal(0, 0.05, n))
    if {"high", "low", "open", "close"} <= set(p.columns):
        o, c = p["open"].to_numpy(), p["close"].to_numpy()
        p["high"] = np.maximum(p["high"].to_numpy(), np.maximum(o, c))
        p["low"] = np.minimum(p["low"].to_numpy(), np.minimum(o, c))
    if "volume" in p:
        p.iloc[t + 1:, p.columns.get_loc("volume")] = p["volume"].to_numpy()[t + 1:] * rng.uniform(0.1, 10, n)
    return p


def assert_causal(fn, df: pd.DataFrame, cuts=CUTS, also_truncate=True):
    base = _as_frame(fn(df))
    for k, t in enumerate(cuts):
        t = min(t, len(df) - 1)
        alt = _as_frame(fn(_perturb(df, t, 100 + k)))
        pd.testing.assert_frame_equal(base.iloc[:t + 1], alt.iloc[:t + 1], check_exact=False, rtol=1e-12, atol=1e-12)
        if also_truncate and t < len(df) - 1:
            tr = _as_frame(fn(df.iloc[:t + 1]))
            pd.testing.assert_frame_equal(base.iloc[:t + 1], tr, check_exact=False, rtol=1e-10, atol=1e-10)


D = synthetic_ohlcv(400, seed=7)

DAILY = {
    "sma": lambda d: ind.sma(d.close, 20),
    "ema": lambda d: ind.ema(d.close, 20),
    "ema_first": lambda d: ind.ema(d.close, 20, seed="first"),
    "wilder": lambda d: ind.wilder(d.close, 14),
    "rsi": lambda d: ind.rsi(d.close, 14, return_components=True),
    "rsi_cutler": lambda d: ind.rsi(d.close, 2, method="cutler"),
    "macd": lambda d: ind.macd(d.close),
    "stochastic": lambda d: ind.stochastic(d.high, d.low, d.close),
    "williams_r": lambda d: ind.williams_r(d.high, d.low, d.close),
    "cci": lambda d: ind.cci(d.high, d.low, d.close, return_components=True),
    "true_range": lambda d: ind.true_range(d.high, d.low, d.close),
    "atr": lambda d: ind.atr(d.high, d.low, d.close, 14),
    "atr_sma": lambda d: ind.atr(d.high, d.low, d.close, 10, method="sma"),
    "atr_ema": lambda d: ind.atr(d.high, d.low, d.close, 10, method="ema"),
    "bollinger": lambda d: ind.bollinger(d.close),
    "bandwidth_pct": lambda d: ind.bandwidth_percentile(d.close, 20, 2, 100),
    "keltner": lambda d: ind.keltner(d.high, d.low, d.close),
    "squeeze": lambda d: ind.squeeze_on(d.high, d.low, d.close),
    "donchian": lambda d: ind.donchian(d.high, d.low, 20),
    "returns": lambda d: ind.returns(d.close, log=True),
    "realized_vol": lambda d: ind.realized_vol(d.close, 20, lag=1),
    "overnight_share": lambda d: ind.overnight_variance_share(d.open, d.close, 60),
    "percentile_rank": lambda d: ind.percentile_rank(d.close, 100),
    "percentile_rank_prev": lambda d: ind.percentile_rank(d.close, 100, include_current=False),
    "zscore": lambda d: ind.zscore(d.close, 20),
    "dmi": lambda d: ind.dmi(d.high, d.low, d.close),
    "psar": lambda d: ind.parabolic_sar(d.high, d.low),
    "supertrend": lambda d: ind.supertrend(d.high, d.low, d.close),
    "ichimoku": lambda d: ind.ichimoku(d.high, d.low, d.close),
    "linreg_slope": lambda d: ind.linreg_slope(d.close, 20, normalize=True),
    "high_ratio": lambda d: ind.high_ratio(d.close, d.high, 252),
    "prior_ath": lambda d: ind.prior_all_time_high(d.high, 100),
    "nr7": lambda d: ind.nr_n(d.high, d.low, 7),
    "inside": lambda d: ind.inside_bar(d.high, d.low, 2),
    "ibs": lambda d: ind.ibs(d.high, d.low, d.close),
    "engulfing": lambda d: ind.bullish_engulfing(d.open, d.high, d.low, d.close),
    "down_streak": lambda d: ind.down_streak(d.close),
    "pivots": lambda d: ind.pivots(d.high, d.low, 3, 3),
    "double_bottom": lambda d: ind.double_bottom(d.high, d.low, d.close, 2, 0.02, 3, 30),
    "head_shoulders": lambda d: ind.head_and_shoulders(d.high, d.low, d.close, 3, 14, 0.2, 2.0),
    "darvas": lambda d: ind.darvas_box(d.high, d.low, d.close),
    "vcp": lambda d: ind.volatility_contraction(d.high, d.low, d.close),
    "adv": lambda d: ind.adv(d.volume, 20, lag=1),
    "dollar_volume": lambda d: ind.dollar_volume(d.close, d.volume, 20),
    "rolling_ols": lambda d: ind.rolling_ols(d.close.pct_change(), d.open.pct_change(), 60),
    "rolling_beta": lambda d: ind.rolling_beta(d.close.pct_change(), d.open.pct_change(), 60),
    "kalman": lambda d: ind.kalman_hedge(d.close, d.open, 1e-3, None, True, 60),
    "kalman_fixedR": lambda d: ind.kalman_hedge(d.close, d.open, 1e-2, 0.5, True),
}


@pytest.mark.parametrize("name", sorted(DAILY))
def test_daily_indicator_is_causal(name):
    assert_causal(DAILY[name], D)


def test_pca_residuals_causal():
    rng = np.random.default_rng(3)
    f = rng.normal(0, 0.01, (300, 2))
    R = pd.DataFrame(f @ rng.normal(1, 0.3, (2, 12)) + rng.normal(0, 0.005, (300, 12)))
    fn = lambda d: ind.pca_residuals(d, window=100, n_components=2, refit_every=5)
    base = fn(R)
    for t in (150, 220):
        P = R.copy()
        P.iloc[t + 1:] = rng.normal(0, 0.05, P.iloc[t + 1:].shape)
        pd.testing.assert_frame_equal(base.iloc[:t + 1], fn(P).iloc[:t + 1])
    assert base.iloc[:100].isna().all().all() and base.iloc[100:].notna().all().all()


def test_pair_spread_z_causal_after_formation():
    a, b = D.close, D.open * 1.01
    fe = D.index[200]
    base = ind.pair_spread_z(a, b, D.index[0], fe)
    t = 260
    a2 = a.copy(); a2.iloc[t + 1:] *= 2
    alt = ind.pair_spread_z(a2, b, D.index[0], fe)
    pd.testing.assert_frame_equal(base.iloc[:t + 1], alt.iloc[:t + 1])


def test_cross_section_panel_rowwise_causal():
    rng = np.random.default_rng(0)
    P = pd.DataFrame(rng.normal(size=(30, 40)))
    b0 = xs.quantile_bucket_panel(P, 10)
    P2 = P.copy(); P2.iloc[20:] = rng.normal(size=(10, 40))
    pd.testing.assert_frame_equal(b0.iloc[:20], xs.quantile_bucket_panel(P2, 10).iloc[:20])


# ---------------------------------------------------------------- intraday
I = synthetic_intraday(22, seed=5, pre=True)
INTRA = {
    "session_vwap": lambda b: ind.session_vwap(b),
    "session_vwap_close": lambda b: ind.session_vwap(b, price="close"),
    "vwap_bands": lambda b: ind.vwap_bands(b, 2.0),
    "opening_range": lambda b: ind.opening_range(b, 15),
    "session_extremes": lambda b: ind.session_extremes(b),
    "rvol": lambda b: ind.relative_volume(b, 10),
    "rvol_cum": lambda b: ind.relative_volume(b, 10, cumulative=True),
    "tod_profile": lambda b: ind.time_of_day_profile(b, 10, stat="mean"),
    "minutes_since_open": lambda b: ind.minutes_since_open(b),
}


@pytest.mark.parametrize("name", sorted(INTRA))
def test_intraday_indicator_is_causal(name):
    n = len(I)
    assert_causal(INTRA[name], I, cuts=(n // 3 + 17, n // 2 + 101, n - 50))


def test_window_stats_known_only_after_window():
    ws = ind.premarket(I, "07:00", "09:30")
    # known_at is the end of the 09:29 bar = 09:30 ET
    k = ws["known_at"].iloc[0].tz_convert("America/New_York")
    assert (k.hour, k.minute) == (9, 30)
    # perturbing RTH bars does not change premarket stats
    I2 = I.copy()
    I2.loc[I2.session == "RTH", "high"] *= 3
    pd.testing.assert_frame_equal(ws, ind.premarket(I2, "07:00", "09:30"))


def test_flag_pattern_causal():
    b = synthetic_intraday(6, seed=11)
    key = ind.intraday.session_key(b.index)
    fn = lambda d: ind.flag_pattern(d.open, d.high, d.low, d.close, d.volume, ind.intraday.session_key(d.index),
                                    impulse_pct=0.001, impulse_bars=45, retrace=(0.0, 1.0), flag_bars=(2, 30))
    assert_causal(fn, b, cuts=(500, 1200, 2000))
    assert key.shape[0] == len(b)


# ---------------------------------------------------------------- streaming == batch
def test_streaming_matches_batch():
    d = D
    np.testing.assert_allclose(st.run_streaming(st.StreamingSMA(20), d.close), ind.sma(d.close, 20), equal_nan=True)
    np.testing.assert_allclose(st.run_streaming(st.StreamingEMA(20), d.close), ind.ema(d.close, 20), equal_nan=True,
                               rtol=1e-12)
    np.testing.assert_allclose(st.run_streaming(st.StreamingRSI(14), d.close), ind.rsi(d.close, 14), equal_nan=True,
                               rtol=1e-12)
    np.testing.assert_allclose(st.run_streaming(st.StreamingATR(14), d.high, d.low, d.close),
                               ind.atr(d.high, d.low, d.close, 14), equal_nan=True, rtol=1e-12)
    np.testing.assert_allclose(st.run_streaming(st.RollingExtreme(20, "max"), d.high),
                               d.high.rolling(20).max(), equal_nan=True)
    np.testing.assert_allclose(st.run_streaming(st.RollingExtreme(20, "min"), d.low),
                               d.low.rolling(20).min(), equal_nan=True)


def test_streaming_vwap_and_opening_range_match_batch():
    b = I[I.session == "RTH"]
    key = ind.intraday.session_key(b.index)
    sv = st.StreamingSessionVWAP("typical")
    vw, sd = [], []
    for k, (h, l, c, v) in zip(key, b[["high", "low", "close", "volume"]].to_numpy()):
        vw.append(sv.update(k, h, l, c, v)); sd.append(sv.sd)
    ref = ind.session_vwap(b)
    np.testing.assert_allclose(vw, ref.vwap, rtol=1e-10)
    np.testing.assert_allclose(sd, ref.sd, rtol=1e-6, atol=1e-9)
    so = st.StreamingOpeningRange(15)
    mso = ind.minutes_since_open(b).to_numpy()
    outs = np.array([so.update(k, m, h, l) for k, m, h, l in zip(key, mso, b.high, b.low)])
    ref = ind.opening_range(b, 15)
    np.testing.assert_allclose(outs[:, 0], ref.or_high, equal_nan=True)
    np.testing.assert_allclose(outs[:, 1], ref.or_low, equal_nan=True)
