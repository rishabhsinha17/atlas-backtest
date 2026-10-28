"""Known-answer tests against StockCharts ChartSchool worked spreadsheets
(fixtures/stockcharts/*.csv, extracted verbatim from the .xls files with full
precision) and against hand computations written out in the tests."""
from __future__ import annotations

import numpy as np
import pytest

import atlasbt.indicators as ind
from ind_data import load_fixture


def close_where(actual, expected, tol=1e-9):
    a, e = np.asarray(actual, float), np.asarray(expected, float)
    m = ~np.isnan(e)
    assert m.sum() > 0
    np.testing.assert_allclose(a[m], e[m], rtol=0, atol=tol)
    return m


# ---------------------------------------------------------------- StockCharts spreadsheets
def test_sma_ema_stockcharts():
    d = load_fixture("cs_movavg.csv")
    close_where(ind.sma(d.close, 10), d.sma10, 1e-10)
    m = close_where(ind.ema(d.close, 10), d.ema10, 1e-10)
    # warm-up: exactly 9 NaNs, first EMA is the SMA seed
    assert np.isnan(ind.ema(d.close, 10)[:9]).all() and m.sum() == 21


def test_rsi_stockcharts():
    d = load_fixture("cs_rsi.csv")
    out = ind.rsi(d.close, 14, return_components=True)
    close_where(out.avg_gain, d.avg_gain, 1e-12)
    close_where(out.avg_loss, d.avg_loss, 1e-12)
    close_where(out.rsi, d.rsi14, 1e-9)
    assert np.isnan(out.rsi.to_numpy()[:14]).all() and not np.isnan(out.rsi.iloc[14])
    assert d.rsi14.iloc[14] == pytest.approx(70.53278948369497)


def test_atr_stockcharts():
    d = load_fixture("cs_atr.csv")
    close_where(ind.true_range(d.high, d.low, d.close), d.tr, 1e-12)
    close_where(ind.atr(d.high, d.low, d.close, 14), d.atr14, 1e-12)


def test_adx_stockcharts_qqqq_504_bars():
    d = load_fixture("cs_adx.csv")
    x = ind.dmi(d.high, d.low, d.close, 14)
    for ours, theirs in [("tr", "tr"), ("plus_dm", "pdm1"), ("minus_dm", "mdm1"), ("tr_n", "tr14"),
                         ("plus_dm_n", "pdm14"), ("minus_dm_n", "mdm14"), ("plus_di", "pdi14"),
                         ("minus_di", "mdi14"), ("dx", "dx"), ("adx", "adx14")]:
        close_where(x[ours], d[theirs], 1e-9)
    assert x.adx.first_valid_index() == 27  # 2n-1
    assert (~np.isnan(d.adx14)).sum() > 470


def test_bollinger_stockcharts_population_sigma():
    d = load_fixture("cs_bbands.csv")
    b = ind.bollinger(d.close, 20, 2.0)
    close_where(b["mid"], d.mid20, 1e-10)
    close_where(b["std"], d.std20, 1e-10)
    close_where(b["upper"], d.upper, 1e-10)
    close_where(b["lower"], d.lower, 1e-10)
    # the sheet's "BandWidth" column is upper - lower (absolute)
    close_where(b["upper"] - b["lower"], d.bandwidth_pct, 1e-10)
    # sample sigma (ddof=1) would NOT match: guards the population convention
    bs = ind.bollinger(d.close, 20, 2.0, ddof=1)
    assert np.nanmax(np.abs(bs["std"] - d.std20)) > 1e-3


@pytest.mark.parametrize("fname", ["cs_cci.csv", "cs_cci_spx.csv"])
def test_cci_stockcharts(fname):
    d = load_fixture(fname)
    c = ind.cci(d.high, d.low, d.close, 20, return_components=True)
    close_where(c.sma_tp, d.sma_tp20, 1e-8)
    close_where(c.md, d.md20, 1e-8)
    close_where(c.cci, d.cci20, 1e-8)


def test_stochastic_stockcharts():
    d = load_fixture("cs_soo.csv")
    s = ind.stochastic(d.high, d.low, d.close, 14, 3, 3)
    close_where(s.fast_k, d.stoch_k14, 1e-9)
    # slow %K = SMA3 of the published fast %K; slow %D = SMA3 of that
    ref_slow = d.stoch_k14.rolling(3).mean()
    close_where(s.slow_k, ref_slow, 1e-9)
    close_where(s.slow_d, ref_slow.rolling(3).mean(), 1e-9)


def test_williams_r_stockcharts():
    d = load_fixture("cs_percentr.csv")
    close_where(ind.williams_r(d.high, d.low, d.close, 14), d.williams_r14, 1e-9)


# ---------------------------------------------------------------- hand computations
def test_rsi_wilder_vs_cutler_hand():
    c = np.array([10, 11, 10, 12, 11, 13], float)  # changes: +1 -1 +2 -1 +2
    w = ind.rsi(c, 3)
    # seed at idx3: gains (1,0,2)/3=1, losses (0,1,0)/3=1/3 -> RS=3 -> 75
    assert w[3] == pytest.approx(75.0)
    # idx4: change -1: ag=(1*2+0)/3=2/3, al=(1/3*2+1)/3=5/9 -> RS=1.2 -> 54.5454..
    assert w[4] == pytest.approx(100 - 100 / 2.2)
    cu = ind.rsi(c, 3, method="cutler")
    # idx4 window changes (-1,+2,-1): ag=2/3, al=2/3 -> 50
    assert cu[4] == pytest.approx(50.0)
    assert ind.rsi(np.array([1, 2, 3, 4, 5.0]), 3)[4] == 100.0
    assert ind.rsi(np.ones(6), 3)[4] == 50.0


def test_macd_hand():
    c = np.array([1, 2, 3, 4, 5, 6.0])
    m = ind.macd(c, fast=2, slow=3, signal=2)
    # EMA2 (alpha 2/3) seed idx1 = 1.5; idx2 = 1.5+2/3*1.5 = 2.5; idx3 = 3.5 ...
    # EMA3 (alpha 1/2) seed idx2 = 2; idx3 = 3; idx4 = 4 ; EMA2 idx4 = 4.5
    assert m["macd"][2] == pytest.approx(0.5) and m["macd"][4] == pytest.approx(0.5)
    assert np.isnan(m["signal"][2]) and m["signal"][3] == pytest.approx(0.5)
    assert m["hist"][5] == pytest.approx(0.0)


def test_cci_constant_015():
    h = np.array([3, 4, 5, 6.0]); l = h - 2; c = h - 1  # TP = h-1 : 2,3,4,5
    out = ind.cci(h, l, c, 4)
    # SMA=3.5, MD=(1.5+.5+.5+1.5)/4=1 -> (5-3.5)/(0.015*1) = 100
    assert out[3] == pytest.approx(100.0)


def test_parabolic_sar_hand_trace():
    H = np.array([10, 11, 12, 13, 12.5, 11, 10.0])
    L = np.array([9, 10, 11, 12, 11.5, 9.5, 9.0])
    p = ind.parabolic_sar(H, L, 0.02, 0.02, 0.2)
    exp = [np.nan, 9.0, 9.0, 9.12, 9.3528, 13.0, 12.93]
    np.testing.assert_allclose(p["sar"], exp, atol=1e-12)
    assert list(p["trend"]) == [0, 1, 1, 1, 1, -1, -1]
    assert list(p["flip"]) == [0, 0, 0, 0, 0, -1, 0]
    # SAR for bar 2 would be 9.04 but may not exceed the lows of bars 0/1 (Wilder rule) -> 9.0
    assert p["next_sar"][1] == pytest.approx(9.0)
    assert p["next_sar"][6] == pytest.approx(12.7728)
    assert p["af"][3] == pytest.approx(0.06) and p["af"][6] == pytest.approx(0.04)


def test_supertrend_hand_trace():
    m = np.array([10, 10, 10, 11, 12, 13, 12, 11, 10, 9.0])
    st = ind.supertrend(m + 1, m - 1, m, atr_n=3, mult=1.0)
    np.testing.assert_allclose(st["supertrend"], [np.nan, np.nan, 12, 12, 12, 11, 11, 11, 12, 11], atol=1e-12)
    assert list(st["direction"]) == [0, 0, -1, -1, -1, 1, 1, 1, -1, -1]
    assert list(st["flip"]) == [0, 0, 0, 0, 0, 1, 0, 0, -1, 0]


def test_ichimoku_information_time(ohlcv):
    d = ohlcv
    ich = ind.ichimoku(d.high, d.low, d.close)
    tk = (d.high.rolling(9).max() + d.low.rolling(9).min()) / 2
    kj = (d.high.rolling(26).max() + d.low.rolling(26).min()) / 2
    sb = (d.high.rolling(52).max() + d.low.rolling(52).min()) / 2
    np.testing.assert_allclose(ich.cloud_a, ((tk + kj) / 2).shift(26), equal_nan=True)
    np.testing.assert_allclose(ich.cloud_b, sb.shift(26), equal_nan=True)
    np.testing.assert_allclose(ich.chikou_ref, d.close.shift(26), equal_nan=True)
    assert ich.cloud_top.first_valid_index() == d.index[51 + 26]


def test_keltner_donchian_hand():
    h = np.array([5, 6, 7, 8, 9.0]); l = h - 2; c = h - 1
    dc = ind.donchian(h, l, 3)
    assert np.isnan(dc["upper"][2]) and dc["upper"][3] == 7 and dc["lower"][4] == 4
    dc2 = ind.donchian(h, l, 3, include_current=True)
    assert dc2["upper"][2] == 7
    kc = ind.keltner(h, l, c, n=3, k=2.0)
    # EMA3 seed idx2 = mean(4,5,6)=5; ATR3: TR=(2,2,2,..) -> 2 ; upper = 5 + 4
    assert kc["upper"][2] == pytest.approx(9.0) and kc["lower"][2] == pytest.approx(1.0)


def test_linreg_zscore_percentile_hand():
    x = np.array([1, 3, 5, 7, 9.0])
    assert ind.linreg_slope(x, 3)[4] == pytest.approx(2.0)
    z = ind.zscore(np.array([1, 2, 3.0]), 3)
    assert z[2] == pytest.approx(1 / np.std([1, 2, 3]))
    pr = ind.percentile_rank(np.array([1, 2, 3, 4, 2.0]), 4, include_current=False)
    # ref set (1,2,3,4); value 2: below=1, ties=1 -> 1.5/4
    assert pr[4] == pytest.approx(0.375)
    pr2 = ind.percentile_rank(np.array([1, 2, 3, 4.0]), 4, include_current=True)
    assert pr2[3] == pytest.approx(3.5 / 4)


def test_realized_vol_and_scaling():
    c = np.array([100, 101, 100, 101, 100.0])
    rv = ind.realized_vol(c, 4, annualize=None)
    r = np.diff(c) / c[:-1]
    assert rv[4] == pytest.approx(np.std(r, ddof=1))
    assert ind.realized_vol(c, 4, annualize=252)[4] == pytest.approx(np.std(r, ddof=1) * np.sqrt(252))
    assert np.isnan(ind.realized_vol(c, 3, annualize=None, lag=1)[3])
    assert ind.vol_scaled_to_time(0.02, 97.5) == pytest.approx(0.01)
    assert ind.vol_scaled_to_time(0.02, 0, overnight_share=0.25) == pytest.approx(0.01)


def test_high_ratio_and_ath():
    h = np.array([10, 12, 11, 13, 9.0]); c = h - 1
    hr = ind.high_ratio(c, h, 3)
    assert hr[4] == pytest.approx(8 / 13)
    ath = ind.prior_all_time_high(h, min_history=2)
    assert np.isnan(ath[1]) and ath[2] == 12 and ath[4] == 13
