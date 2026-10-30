import numpy as np
import pandas as pd
import pytest

from atlasbt.pricing.realized import (close_to_close, daily_variance, ewma_vol, garman_klass, har_rv_forecast,
                                      parkinson, realized_vol, yang_zhang)


def synthetic_gbm_bars(n=1500, sigma=0.2, seed=7, steps=390):
    """synthetic_* fixture: intraday GBM aggregated to daily OHLC (no drift, no gaps)."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252 / steps
    logp = np.cumsum(rng.normal(0, sigma * np.sqrt(dt), n * steps))
    p = 100 * np.exp(logp).reshape(n, steps)
    idx = pd.bdate_range("2010-01-01", periods=n)
    o = np.r_[100.0, p[:-1, -1]]
    return pd.DataFrame({"open": o, "high": np.maximum(p.max(1), o), "low": np.minimum(p.min(1), o), "close": p[:, -1]},
                        index=idx)


def test_estimators_unbiased_on_gbm():
    b = synthetic_gbm_bars()
    for f in (lambda: close_to_close(b["close"], 1000), lambda: parkinson(b, 1000), lambda: garman_klass(b, 1000),
              lambda: yang_zhang(b, 1000), lambda: realized_vol(b, 1000, "rs")):
        v = f().iloc[-1]
        assert v == pytest.approx(0.2, rel=0.07)  # discrete-path range estimators are slightly biased low


def test_estimators_are_causal():
    b = synthetic_gbm_bars(300)
    v1 = yang_zhang(b, 20)
    b2 = b.copy()
    b2.iloc[200:] *= 1.5  # perturb the future
    v2 = yang_zhang(b2, 20)
    pd.testing.assert_series_equal(v1.iloc[:200], v2.iloc[:200])


def test_ewma():
    r = pd.Series(np.r_[np.full(30, 0.01), np.full(30, 0.02)])
    v = ewma_vol(r)
    assert v.iloc[29] == pytest.approx(0.01 * np.sqrt(252))
    assert v.iloc[59] > v.iloc[30] > v.iloc[29]


def test_har_walk_forward_no_lookahead_and_accuracy():
    b = synthetic_gbm_bars(1200)
    rv = daily_variance(b, "gk")
    f1 = har_rv_forecast(rv, horizon=21, min_train=250)
    rv2 = rv.copy()
    rv2.iloc[800:] *= 10.0
    f2 = har_rv_forecast(rv2, horizon=21, min_train=250)
    # forecasts made at t < 800 - 21 use only targets realised before 800
    pd.testing.assert_series_equal(f1["var_fcst"].iloc[:779], f2["var_fcst"].iloc[:779])
    assert f1["vol_fcst"].dropna().iloc[-200:].mean() == pytest.approx(0.2, rel=0.1)
    # fit dates never after the forecast date
    fd = f1["fit_date"].dropna()
    assert all(pd.Timestamp(d) <= i for i, d in fd.items())
