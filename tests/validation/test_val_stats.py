"""PSR/DSR (paper example), trial accounting, PBO/CSCV constructed cases,
bootstrap block length and coverage, SPA/RC vs arch, BH vs statsmodels."""
from __future__ import annotations

import math

import numpy as np
import pytest

from atlasbt.validation import bootstrap as bs
from atlasbt.validation import multiple_testing as mt
from atlasbt.validation import pbo as pbo_mod
from atlasbt.validation import sharpe_tests as st


# ---------------------------------------------------------------- PSR / DSR
def test_dsr_paper_numerical_example():
    """Bailey & Lopez de Prado (2014), JPM 40(5), numerical example:
    SR_ann = 2.5, T = 1250 daily obs (250/yr), skew = -3, kurt = 10, N = 100,
    V[SR_n] (annualised) = 0.5  ->  SR0 = 0.1132 (per period), DSR = 0.9004."""
    sr = 2.5 / math.sqrt(250)
    v = 0.5 / 250
    assert st.expected_max_sharpe(100, v) == pytest.approx(0.1132, abs=5e-5)
    assert st.dsr(sr, 1250, -3.0, 10.0, 100, v) == pytest.approx(0.9004, abs=5e-5)


def test_psr_properties_and_min_trl():
    assert st.psr(0.1, 0.1, 500) == pytest.approx(0.5)
    # normal returns: PSR = Phi(SR sqrt(T-1)/sqrt(1 + SR^2/2))
    z = 0.1 * math.sqrt(999) / math.sqrt(1 + 0.5 * 0.01)
    from scipy.stats import norm
    assert st.psr(0.1, 0.0, 1000, 0.0, 3.0) == pytest.approx(norm.cdf(z))
    # negative skew & fat tails lower PSR
    assert st.psr(0.1, 0.0, 1000, -2.0, 9.0) < st.psr(0.1, 0.0, 1000)
    n = st.min_track_record_length(0.1, 0.0, -1.0, 6.0, 0.95)
    assert st.psr(0.1, 0.0, int(math.ceil(n)), -1.0, 6.0) >= 0.95 > st.psr(0.1, 0.0, int(math.floor(n)) - 1, -1.0, 6.0)
    assert st.expected_max_sharpe(1, 0.01) == 0.0
    assert st.effective_trials(100, 1.0) == 1.0 and st.effective_trials(100, 0.0) == 100


def test_trial_accounting_whole_atlas():
    assert st.parse_grid_size("opening 5/15/30m; RVOL 1/1.5/2; target none/1.5R/2R; stop OR low/1 ATR(5m)") == 54
    assert st.parse_grid_size("5 metrics equal vs weighted; top 10/20%; trim winsorize 1/5%") == 8
    assert st.parse_grid_size("ratio 1:2; short .45/.5; long .2/.25") == 4
    t = st.atlas_trials()
    assert len(t) == 172 and t.set_index("strategy_id").loc["S001", "n_trials"] == 54
    tot = st.total_atlas_trials()
    assert tot == t.n_trials.sum() and tot > 1000
    t2 = st.atlas_trials({"S001": 12})
    assert t2.set_index("strategy_id").loc["S001", "n_trials"] == 12 and t2.n_trials.sum() == tot - 42


# ---------------------------------------------------------------- PBO / CSCV
def _block_pattern(good: bool, bs_: int):
    base = np.tile([0.02, 0.0], bs_ // 2) if good else np.tile([0.005, -0.005], bs_ // 2)
    return base


def test_pbo_constructed_always_overfit():
    """Strategy i is good only in block i: whatever looks best in-sample has no
    good block out-of-sample -> ranks below the median every time -> PBO = 1."""
    S, bsz = 4, 50
    M = np.column_stack([np.concatenate([_block_pattern(b == i, bsz) for b in range(S)]) for i in range(S)])
    r = pbo_mod.pbo_cscv(M, S=S)
    assert r.n_combinations == 6
    assert r.pbo == 1.0
    assert np.all(r.logits < 0)


def test_pbo_true_skill_near_zero_and_noise_near_half():
    rng = np.random.default_rng(0)
    T, N = 2000, 20
    M = rng.normal(0, 0.01, (T, N))
    M[:, 7] += 0.002
    assert pbo_mod.pbo_cscv(M, S=16).pbo < 0.02
    noise = pbo_mod.pbo_cscv(rng.normal(0, 0.01, (T, N)), S=16)
    assert noise.n_combinations == 12870
    # under pure noise PBO is ~0.5 on average (single draws vary a lot: combinations are dependent)
    draws = [pbo_mod.pbo_cscv(rng.normal(0, 0.01, (600, N)), S=10).pbo for _ in range(30)]
    assert 0.4 < float(np.mean(draws)) < 0.6
    sub = pbo_mod.pbo_cscv(rng.normal(0, 0.01, (T, N)), S=16, max_combinations=500)
    assert sub.n_combinations == 500


# ---------------------------------------------------------------- bootstrap
def _ar1(n, phi, rng, mu=0.0, sd=1.0):
    e = rng.normal(0, sd, n)
    x = np.empty(n)
    x[0] = e[0] / math.sqrt(1 - phi ** 2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x + mu


def test_optimal_block_length_ar1_theory_and_arch():
    from arch.bootstrap import optimal_block_length as arch_obl

    rng = np.random.default_rng(1)
    n, phi = 10_000, 0.5
    x = _ar1(n, phi, rng)
    theory = (2 * phi / (1 - phi ** 2)) ** (2 / 3) * n ** (1 / 3)  # = 26.1
    b = bs.optimal_block_length(x)
    assert b["stationary"] == pytest.approx(theory, rel=0.3)
    assert b["circular"] == pytest.approx(b["stationary"] * (2 / (4 / 3)) ** (1 / 3), rel=1e-9)
    a = arch_obl(x)
    assert b["stationary"] == pytest.approx(float(a["stationary"].iloc[0]), rel=0.15)
    assert bs.optimal_block_length(rng.normal(size=5000))["stationary"] < 5


def test_bootstrap_index_generators():
    rng = np.random.default_rng(2)
    idx = bs.stationary_indices(2000, 10.0, 50, rng)
    starts = np.mean(np.diff(idx, axis=1) != 1) + 0.0
    assert 1 / starts == pytest.approx(10.0, rel=0.15)  # mean block length ~ b (wrap-arounds negligible)
    c = bs.circular_indices(103, 10, 5, rng)
    assert c.shape == (5, 103) and np.all((np.diff(c[:, :10], axis=1) % 103) == 1)


def test_metric_functions_known_answers():
    r = np.array([0.1, -0.5, 0.2])
    assert bs.max_drawdown(r) == pytest.approx(-0.5)
    assert bs.cagr(np.full(252, 0.001), 252) == pytest.approx(1.001 ** 252 - 1)
    ci = bs.bootstrap_ci(np.random.default_rng(3).normal(0.001, 0.01, 500), "sharpe", reps=200, block=5)
    assert ci.lo < ci.point < ci.hi and ci.block == 5


def test_bootstrap_coverage_sanity_dependent_data():
    """95% CIs for the mean of AR(1) data: stationary bootstrap covers ~nominal,
    an iid bootstrap under-covers (it ignores autocorrelation)."""
    rng = np.random.default_rng(4)
    mu, phi, n, reps, sims = 0.05, 0.5, 400, 300, 150
    mean_fn = lambda a: a.mean(axis=-1)
    hit_sb = hit_iid = 0
    for s in range(sims):
        x = _ar1(n, phi, rng, mu)
        a = bs.bootstrap_ci(x, mean_fn, reps=reps, method="stationary", block=8, seed=s)
        b = bs.bootstrap_ci(x, mean_fn, reps=reps, method="iid", block=1, seed=s)
        hit_sb += a.lo <= mu <= a.hi
        hit_iid += b.lo <= mu <= b.hi
    assert hit_sb / sims >= 0.85
    assert hit_iid / sims < hit_sb / sims - 0.08


# ---------------------------------------------------------------- SPA / RC / FDR
def test_spa_and_reality_check_vs_arch():
    from arch.bootstrap import SPA

    rng = np.random.default_rng(5)
    n, k = 1000, 15
    d = rng.normal(0, 0.01, (n, k))
    d[:, 0] += 0.0005
    ours = mt.hansen_spa(d, reps=3000, block=10, studentize=False, seed=1)
    a = SPA(np.zeros(n), -d, block_size=10, reps=3000, seed=2, studentize=False)
    a.compute()
    for key in ("lower", "consistent", "upper"):
        assert ours[key] == pytest.approx(float(a.pvalues[key]), abs=0.04)
    assert mt.white_reality_check(d, reps=1000, block=10)["pvalue"] == pytest.approx(ours["upper"], abs=0.05)


def test_spa_detects_real_edge_and_not_noise():
    rng = np.random.default_rng(6)
    d = rng.normal(0, 0.01, (1500, 30))
    p_null = mt.hansen_spa(d, reps=1000, seed=0)
    assert p_null["consistent"] > 0.1
    d[:, 3] += 0.003
    p_alt = mt.hansen_spa(d, reps=1000, seed=0)
    assert p_alt["consistent"] < 0.01 and p_alt["lower"] <= p_alt["consistent"] <= p_alt["upper"]


def test_kernel_variance_matches_definition():
    rng = np.random.default_rng(7)
    d = rng.normal(size=(300, 3))
    n, q = 300, 0.1
    e = d - d.mean(0)
    v = (e * e).sum(0) / n
    for i in range(1, n):
        kap = (1 - i / n) * (1 - q) ** i + (i / n) * (1 - q) ** (n - i)
        v += 2 * kap * (e[:n - i] * e[i:]).sum(0) / n
    np.testing.assert_allclose(mt.kernel_variance(d, 10), v, rtol=1e-10)


def test_bh_and_by_match_statsmodels():
    from statsmodels.stats.multitest import multipletests

    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    rej, q = mt.benjamini_hochberg(p, 0.05)
    r2, q2, *_ = multipletests(p, 0.05, method="fdr_bh")
    np.testing.assert_allclose(q, q2) and np.testing.assert_array_equal(rej, r2)
    rej, q = mt.benjamini_hochberg(p, 0.05, method="by")
    r2, q2, *_ = multipletests(p, 0.05, method="fdr_by")
    np.testing.assert_allclose(q, q2)
    np.testing.assert_array_equal(rej, r2)
