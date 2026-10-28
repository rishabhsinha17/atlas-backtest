"""Stat-arb helpers and cross-sectional utilities: known answers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import atlasbt.indicators as ind
from atlasbt.indicators import cross_section as xs


def test_rolling_ols_matches_lstsq():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"m": rng.normal(size=n), "s": rng.normal(size=n)})
    y = pd.Series(0.1 + 1.5 * X.m - 0.7 * X.s + rng.normal(0, 0.1, n))
    y.iloc[50] = np.nan
    out = ind.rolling_ols(y, X, 60, min_periods=50)
    assert ind.rolling_ols(y, X, 60).iloc[59:110, 0].isna().all()  # default: full window of valid rows
    for t in (59, 100, 199):
        w = slice(t - 59, t + 1)
        ok = y.iloc[w].notna()
        A = np.column_stack([np.ones(ok.sum()), X.iloc[w][ok].to_numpy()])
        b, *_ = np.linalg.lstsq(A, y.iloc[w][ok].to_numpy(), rcond=None)
        np.testing.assert_allclose(out.iloc[t, :3].to_numpy(), b, rtol=1e-8)
        assert out.resid.iloc[t] == pytest.approx(y.iloc[t] - A[-1] @ b if ok.iloc[-1] else np.nan, nan_ok=True)
    # out-of-sample residual uses beta fit through t-1
    b_prev = out.iloc[149, :3].to_numpy()
    assert out.resid_oos.iloc[150] == pytest.approx(y.iloc[150] - np.r_[1, X.iloc[150]] @ b_prev)
    assert out.iloc[:49, 0].isna().all()


def test_kalman_one_step_ahead_and_rls_limit():
    rng = np.random.default_rng(1)
    n = 500
    x = np.cumsum(rng.normal(size=n)) + 50
    y = 2.0 + 1.3 * x + rng.normal(0, 0.5, n)
    k = ind.kalman_hedge(y, x, q_ratio=0.0, obs_var=0.25, p0=1e6)
    # innovation uses the PRIOR state (data <= t-1)
    t = 300
    assert k.innovation[t] == pytest.approx(y[t] - (k.alpha_prior[t] + k.beta_prior[t] * x[t]))
    assert k.beta_prior[t] == pytest.approx(k.beta[t - 1])
    # with no state noise the filter is recursive least squares -> full-sample OLS
    b = np.polyfit(x, y, 1)
    assert k.beta[n - 1] == pytest.approx(b[0], rel=1e-4) and k.alpha[n - 1] == pytest.approx(b[1], rel=1e-3)
    kw = ind.kalman_hedge(pd.Series(y), pd.Series(x), q_ratio=1e-3, obs_var=None, warmup=60)
    assert kw.z.iloc[:60].isna().all() and kw.z.iloc[60:].notna().all()
    assert abs(np.nanmean(kw.z.iloc[100:])) < 0.2 and 0.6 < np.nanstd(kw.z.iloc[100:]) < 1.6


def test_engle_granger_and_adf():
    rng = np.random.default_rng(2)
    n = 500
    x = np.cumsum(rng.normal(size=n)) + 100
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.5 * e[t - 1] + rng.normal()
    y = 3 + 2 * x + e
    r = ind.engle_granger(pd.Series(y), pd.Series(x))
    assert r.pvalue < 0.01 and r.beta == pytest.approx(2.0, abs=0.05)
    z = np.cumsum(rng.normal(size=n))
    assert ind.engle_granger(pd.Series(z), pd.Series(x)).pvalue > 0.05
    assert ind.adf(e)["pvalue"] < 0.01 and ind.adf(x)["pvalue"] > 0.05


def test_pca_residuals_remove_factor_structure():
    rng = np.random.default_rng(3)
    T, N = 400, 15
    F = rng.normal(0, 0.01, (T, 2))
    B = rng.normal(1, 0.5, (2, N))
    R = pd.DataFrame(F @ B + rng.normal(0, 0.0005, (T, N)))
    r2 = ind.pca_residuals(R, window=150, n_components=2)
    r1 = ind.pca_residuals(R, window=150, n_components=1)
    tail = slice(150, None)
    assert (r2.iloc[tail].abs().mean().mean()) < 0.15 * R.iloc[tail].abs().mean().mean()
    assert r1.iloc[tail].abs().mean().mean() > 3 * r2.iloc[tail].abs().mean().mean()


def test_ssd_pairs_selection():
    rng = np.random.default_rng(4)
    base = np.cumsum(rng.normal(0, 0.01, 252))
    P = pd.DataFrame({"A": 100 * np.exp(base), "B": 50 * np.exp(base + rng.normal(0, 0.001, 252)),
                      "C": 30 * np.exp(np.cumsum(rng.normal(0, 0.01, 252))),
                      "D": 20 * np.exp(base + rng.normal(0, 0.003, 252))})
    s = ind.ssd_pairs(P)
    assert tuple(s.iloc[0][["a", "b"]]) == ("A", "B")
    sel = ind.select_pairs(s, n=5)
    assert len(sel) == 2 and set(sel.iloc[0][["a", "b"]]) == {"A", "B"}
    g = ind.ssd_pairs(P, groups={"A": 1, "B": 2, "C": 1, "D": 1})
    assert ("A", "B") not in set(zip(g.a, g.b))
    z = ind.pair_spread_z(P.A, P.B, 0, 125)
    f = z.spread.iloc[:126]
    assert z.z.iloc[200] == pytest.approx((z.spread.iloc[200] - f.mean()) / f.std(ddof=1))


def test_quantile_buckets_ties_and_breakpoints():
    s = pd.Series(np.arange(1, 11, dtype=float), index=list("abcdefghij"))
    assert list(xs.quantile_bucket(s, 10)) == list(range(1, 11))
    assert list(xs.quantile_bucket(s, 5)) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    t = pd.Series([1, 1, 1, 2, np.nan])
    b = xs.quantile_bucket(t, 2)
    assert b.iloc[0] == b.iloc[1] == b.iloc[2] and np.isnan(b.iloc[4])
    # NYSE breakpoints: universe (1..4) median 2.5; value equal to a breakpoint -> lower bucket
    m = pd.Series([1, 2, 3, 4, 2.5, 10.0])
    ny = pd.Series([True, True, True, True, False, False])
    assert list(xs.size_buckets(m, ny, 2)) == [1, 1, 2, 2, 1, 2]
    np.testing.assert_allclose(xs.breakpoints(m[ny], 4), [1.75, 2.5, 3.25])
    ls = xs.long_short(xs.quantile_bucket(s, 5), 5)
    assert ls["j"] == 1 and ls["a"] == -1 and ls["e"] == 0


def test_group_ops_and_winsorize():
    x = pd.Series([1, 2, 3, 10, 20, 30.0], index=list("abcdef"))
    g = pd.Series(["t", "t", "t", "f", "f", "f"], index=x.index)
    z = xs.group_zscore(x, g)
    assert z["a"] == pytest.approx(-1.0) and z["f"] == pytest.approx(1.0)
    assert xs.group_demean(x, g)["d"] == pytest.approx(-10)
    assert list(xs.group_rank(x, g)) == pytest.approx([1 / 3, 2 / 3, 1, 1 / 3, 2 / 3, 1])
    assert list(xs.group_quantile_bucket(x, g, 3)) == [1, 2, 3, 1, 2, 3]
    w = xs.winsorize(pd.Series(np.arange(101.0)), 0.05, 0.95)
    assert w.min() == 5 and w.max() == 95
    assert xs.group_zscore(pd.Series([1.0, 2.0, 3.0]), pd.Series(["x", "y", "y"])).isna().iloc[0]
    vw = xs.value_weights(pd.Series([1.0, 3.0]))
    assert list(vw) == [0.25, 0.75]
