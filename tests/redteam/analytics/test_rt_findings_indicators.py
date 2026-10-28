"""Red-team findings in atlasbt.indicators. Each test asserts the CORRECT /
documented behaviour and FAILS on the current code (one test per finding)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import atlasbt.indicators as ind
from atlasbt.indicators import streaming as st
from rt_data import daily_ohlcv, et_minute, intraday_bars


# ---------------------------------------------------------------- F: ADX / DMI dies on leading NaNs
def test_dmi_adx_with_leading_nans_matches_trimmed_series():
    """A symbol that lists 5 bars after the panel start (leading NaNs) must get
    the same ADX as when the NaNs are trimmed. _wilder_sum seeds at fixed index
    1, so a NaN anywhere in the first n+1 bars makes TR_n/+DM_n/-DM_n/DI/ADX NaN
    FOREVER (all 300 values)."""
    d = daily_ohlcv(300, seed=3)
    h, l, c = (d[k].to_numpy().copy() for k in ("high", "low", "close"))
    H, L, C = h.copy(), l.copy(), c.copy()
    H[:5] = L[:5] = C[:5] = np.nan
    full = pd.DataFrame(ind.dmi(H, L, C, 14))
    ref = pd.DataFrame(ind.dmi(h[5:], l[5:], c[5:], 14))
    assert full["adx"].notna().sum() == ref["adx"].notna().sum() == 268  # actual: 0
    np.testing.assert_allclose(full["adx"].to_numpy()[5:], ref["adx"].to_numpy(), equal_nan=True)


def test_dmi_single_nan_inside_seed_window_does_not_kill_series():
    d = daily_ohlcv(300, seed=3)
    h, l, c = (d[k].to_numpy().copy() for k in ("high", "low", "close"))
    h[10] = np.nan
    adx = np.asarray(ind.adx(h, l, c, 14))
    assert np.isfinite(adx[100:]).all()  # actual: all NaN


# ---------------------------------------------------------------- F: Parabolic SAR dies on leading NaNs
def test_parabolic_sar_with_leading_nans_matches_trimmed_series():
    d = daily_ohlcv(300, seed=3)
    h, l = d.high.to_numpy().copy(), d.low.to_numpy().copy()
    H, L = h.copy(), l.copy()
    H[:3] = L[:3] = np.nan
    full = pd.DataFrame(ind.parabolic_sar(H, L))
    ref = pd.DataFrame(ind.parabolic_sar(h[3:], l[3:]))
    assert full["sar"].notna().sum() == ref["sar"].notna().sum()  # actual 0 vs 296
    np.testing.assert_allclose(full["sar"].to_numpy()[3:], ref["sar"].to_numpy(), equal_nan=True)


def test_parabolic_sar_internal_nan_bar_is_nan_and_not_advanced():
    """ANALYTICS.md s.1: recursive indicators emit NaN on a NaN bar and carry the
    state unchanged. _psar emits a SAR for the missing bar and advances the
    SAR/AF one extra step for it."""
    d = daily_ohlcv(300, seed=3)
    h, l = d.high.to_numpy().copy(), d.low.to_numpy().copy()
    H, L = h.copy(), l.copy()
    H[100] = L[100] = np.nan
    p = pd.DataFrame(ind.parabolic_sar(H, L))
    assert np.isnan(p["sar"].iloc[100])  # actual: a finite SAR value
    # equals the series computed with that bar deleted (state and prior-bar extremes not advanced)
    ref = pd.DataFrame(ind.parabolic_sar(np.delete(h, 100), np.delete(l, 100)))
    for col in ("sar", "trend", "next_sar", "af", "ep"):
        np.testing.assert_allclose(np.delete(p[col].to_numpy(), 100), ref[col].to_numpy(), equal_nan=True)


# ---------------------------------------------------------------- F: RVOL blanked for 20 sessions after an early close
def _early_close_bars():
    b = intraday_bars(30, seed=1, freq_min=5)
    key = ind.intraday.session_key(b.index)
    days = pd.unique(key)
    early = (key == days[5]) & (et_minute(b.index) >= 13 * 60)  # 13:00 ET early close on session 5
    return b.loc[~early], days


def test_rvol_afternoon_defined_after_an_early_close_session():
    """S016 trades 13:30-15:00 on cumulative RVOL vs the same-time median of the
    previous 20 sessions. One early close (13:00) inside the lookback makes the
    afternoon slots NaN for the next 20 sessions (rolling min_periods=lookback,
    and the half-day row is NaN after 13:00). With ~3 early closes/yr this blanks
    afternoon RVOL on roughly a quarter of all sessions. Expected: the median over
    the sessions that HAVE the slot (or treat the half day as ended)."""
    b2, days = _early_close_bars()
    for cum in (False, True):
        rv = ind.relative_volume(b2, 20, cumulative=cum)
        k2 = ind.intraday.session_key(b2.index)
        pm = et_minute(b2.index) >= 13 * 60
        s = pd.Series(rv["rvol"].notna().to_numpy()[pm], index=k2[pm]).groupby(level=0).mean()
        # session index 21 (2024-04-01) has 20 prior sessions, all with afternoon data except session 5
        assert s.loc[days[21]] == 1.0, f"cumulative={cum}: afternoon RVOL NaN on {days[21]}"  # actual 0.0


def test_rvol_noncumulative_treats_missing_bar_as_zero_volume():
    """Feeds omit no-trade minute bars. One missing 10:00 bar in a prior session
    makes the 10:00 expected volume NaN for the next `lookback` sessions in
    non-cumulative mode (the cumulative mode already treats it as 0)."""
    b = intraday_bars(25, seed=4, freq_min=5)
    key = ind.intraday.session_key(b.index)
    days = pd.unique(key)
    drop = (key == days[10]) & (et_minute(b.index) == 10 * 60)
    b2 = b.loc[~drop]
    rv = ind.relative_volume(b2, 10)
    k2 = ind.intraday.session_key(b2.index)
    m = (k2 == days[15]) & (et_minute(b2.index) == 10 * 60)
    assert np.isfinite(rv["expected"].to_numpy()[m]).all()  # actual NaN


# ---------------------------------------------------------------- F: StreamingOpeningRange includes premarket bars
def test_streaming_opening_range_matches_batch_when_fed_premarket_bars():
    """streaming.py claims equality with the batch functions. update() accepts
    every bar with end_minute <= minutes, i.e. ALL premarket bars (negative
    minutes since open) and the 09:29 bar (end minute 0). A premarket spike then
    becomes the 'opening range high' (S001/S002/S003 breakout levels)."""
    b = intraday_bars(3, seed=2, pre=True)
    et = b.index.tz_convert("America/New_York")
    b = b.copy()
    b.loc[(et.hour == 8) & (et.minute == 0), "high"] *= 1.05  # premarket spike
    key = ind.intraday.session_key(b.index)
    mso = ind.minutes_since_open(b).to_numpy()
    so = st.StreamingOpeningRange(5)
    out = np.array([so.update(k, m, h, l) for k, m, h, l in zip(key, mso, b.high, b.low)])
    batch = ind.opening_range(b, 5)
    np.testing.assert_allclose(out[:, 0], batch["or_high"].to_numpy(), equal_nan=True)  # 105.46 vs 100.53
    np.testing.assert_allclose(out[:, 1], batch["or_low"].to_numpy(), equal_nan=True)


# ---------------------------------------------------------------- F: streaming != batch on NaN inputs
def test_streaming_rsi_and_rolling_extreme_match_batch_with_nan_inputs():
    d = daily_ohlcv(300, seed=3)
    c, h = d.close.to_numpy().copy(), d.high.to_numpy().copy()
    c[[60, 61, 150]] = np.nan
    h[[60, 61, 150]] = np.nan
    np.testing.assert_allclose(st.run_streaming(st.StreamingRSI(14), c), np.asarray(ind.rsi(c, 14)),
                               equal_nan=True, rtol=1e-12)  # streaming returns a stale RSI on NaN bars
    np.testing.assert_allclose(st.run_streaming(st.RollingExtreme(20, "max"), h),
                               pd.Series(h).rolling(20).max().to_numpy(), equal_nan=True)
