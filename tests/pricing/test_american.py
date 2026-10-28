import numpy as np
import pytest
from scipy.stats import multivariate_normal

from atlasbt.pricing import bs1993_call, bs2002_price, bsm_price, cbnd, crr_price, american_greeks
from atlasbt.pricing.american import _HAVE_NUMBA


def test_cbnd_vs_scipy():
    rng = np.random.default_rng(0)
    a, b, r = rng.normal(size=60) * 2, rng.normal(size=60) * 2, rng.uniform(-0.99, 0.99, 60)
    ref = [multivariate_normal(mean=[0, 0], cov=[[1, ri], [ri, 1]]).cdf([ai, bi]) for ai, bi, ri in zip(a, b, r)]
    np.testing.assert_allclose(cbnd(a, b, r), ref, atol=1e-7)


def test_hull_binomial_american_put():
    # Hull Example 21.1: S=50,K=50,r=10%,sigma=40%,T=5 months, 5 steps -> 4.49
    assert crr_price(50, 50, 5 / 12, 0.10, 0.40, "P", steps=5, smooth=False) == pytest.approx(4.49, abs=0.005)


def test_crr_converges_to_bsm_for_european():
    for cp in ("C", "P"):
        eu = bsm_price(100, 105, 0.75, 0.04, 0.02, 0.3, cp)
        tree = crr_price(100, 105, 0.75, 0.04, 0.3, cp, american=False, steps=1000, q=0.02)
        assert tree == pytest.approx(eu, abs=2e-3)


def test_crr_discrete_dividends_european_matches_escrowed_bsm():
    from atlasbt.pricing import bsm_price_discrete_div
    eu = bsm_price_discrete_div(40, 40, 0.5, 0.09, 0.30, "C", [2 / 12, 5 / 12], [0.5, 0.5])
    tree = crr_price(40, 40, 0.5, 0.09, 0.30, "C", american=False, steps=1000,
                     div_times=[2 / 12, 5 / 12], div_amounts=[0.5, 0.5])
    assert tree == pytest.approx(eu, abs=2e-3)


def test_american_call_with_dividend_early_exercise_premium():
    # big dividend just before expiry makes early exercise of a deep ITM call valuable
    am = crr_price(100, 80, 0.25, 0.03, 0.2, "C", steps=500, div_times=[0.2], div_amounts=[5.0])
    eu = crr_price(100, 80, 0.25, 0.03, 0.2, "C", american=False, steps=500, div_times=[0.2], div_amounts=[5.0])
    assert am > eu + 0.5
    assert am >= 100 - 80 - 1e-9


def test_bs1993_haug_known_answer():
    # Haug (2007) Bjerksund-Stensland 1993: S=42,X=40,T=.75,r=.04,b=-.04,v=.35 -> 5.2704
    assert bs1993_call(42, 40, 0.75, 0.04, -0.04, 0.35) == pytest.approx(5.2704, abs=1e-4)


def test_bs2002_close_to_binomial_and_between_bounds():
    S, K, T, r, q, v = 42, 40, 0.75, 0.04, 0.08, 0.35
    bs = bs2002_price(S, K, T, r, q, v, "C")
    tree = crr_price(S, K, T, r, v, "C", steps=2000, q=q)
    assert bs1993_call(S, K, T, r, r - q, v) <= bs + 1e-9  # 2002 improves on (is above) the 1993 lower bound
    assert bs == pytest.approx(tree, rel=0.006)
    assert bs >= bsm_price(S, K, T, r, q, v, "C")
    for cp, k in (("P", 40), ("P", 45), ("C", 35)):
        a = bs2002_price(S, k, T, r, 0.01, v, cp)
        b = crr_price(S, k, T, r, v, cp, steps=1500, q=0.01)
        assert a == pytest.approx(b, rel=0.006)


def test_bs2002_no_early_exercise_call_equals_european():
    assert bs2002_price(100, 100, 1, 0.05, 0.0, 0.2, "C") == pytest.approx(bsm_price(100, 100, 1, 0.05, 0, 0.2, "C"))


def test_numba_and_python_kernels_agree():
    if not _HAVE_NUMBA:
        pytest.skip("numba not installed")
    a = crr_price(95, 100, 0.4, 0.05, 0.3, "P", steps=200, use_numba=True, div_times=[0.1], div_amounts=[1.0])
    b = crr_price(95, 100, 0.4, 0.05, 0.3, "P", steps=200, use_numba=False, div_times=[0.1], div_amounts=[1.0])
    assert a == pytest.approx(b, abs=1e-10)


def test_american_greeks_reasonable():
    g = american_greeks(100, 100, 0.5, 0.05, 0.0, 0.25, "P")
    from atlasbt.pricing import bsm_greeks
    e = bsm_greeks(100, 100, 0.5, 0.05, 0.0, 0.25, "P")
    assert g.delta < 0 and abs(g.delta - e.delta) < 0.06
    assert g.gamma > 0 and g.vega > 0 and g.theta < 0
