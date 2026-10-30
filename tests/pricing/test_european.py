"""Known-answer tests: Hull (OFOD) and Haug (2007) published values."""
import numpy as np
import pytest

from atlasbt.pricing import (black76_greeks, black76_price, bsm_greeks, bsm_price, bsm_price_discrete_div,
                             strike_from_delta, bsm_delta, pv_dividends)


def test_hull_bsm_example():
    # Hull, Example 15.6 (8th ed.): S=42,K=40,r=10%,sigma=20%,T=0.5 -> c=4.76, p=0.81
    assert bsm_price(42, 40, 0.5, 0.10, 0.0, 0.20, "C") == pytest.approx(4.76, abs=0.005)
    assert bsm_price(42, 40, 0.5, 0.10, 0.0, 0.20, "P") == pytest.approx(0.81, abs=0.005)


def test_haug_generalized_bsm_examples():
    # Haug (2007) 1.1.1: S=60,K=65,T=.25,r=.08,b=.08,v=.30 -> call 2.1334
    assert bsm_price(60, 65, 0.25, 0.08, 0.0, 0.30, "C") == pytest.approx(2.1334, abs=1e-4)
    # Haug Merton (1973): S=100,K=95,T=.5,r=.10,q=.05,v=.20 -> put 2.4648
    assert bsm_price(100, 95, 0.5, 0.10, 0.05, 0.20, "P") == pytest.approx(2.4648, abs=1e-4)


def test_black76_known_answers():
    # Haug (2007) 1.1.3 Black-76: F=19,K=19,T=.75,r=.10,v=.28 -> call = put = 1.7011
    assert black76_price(19, 19, 0.75, 0.10, 0.28, "C") == pytest.approx(1.7011, abs=1e-4)
    assert black76_price(19, 19, 0.75, 0.10, 0.28, "P") == pytest.approx(1.7011, abs=1e-4)
    # Hull Example 18.x: futures put F=20,K=20,r=9%,T=4/12,sigma=25% -> 1.12
    assert black76_price(20, 20, 4 / 12, 0.09, 0.25, "P") == pytest.approx(1.12, abs=0.005)


def test_hull_discrete_dividend_example():
    # Hull Example 15.10: S=40,K=40,r=9%,sigma=30%,T=0.5, dividends 0.50 at 2 and 5 months -> c=3.67
    c = bsm_price_discrete_div(40, 40, 0.5, 0.09, 0.30, "C", [2 / 12, 5 / 12], [0.5, 0.5])
    assert c == pytest.approx(3.67, abs=0.005)
    assert 40 - pv_dividends([2 / 12, 5 / 12], [0.5, 0.5], 0.5, 0.09) == pytest.approx(39.0259, abs=1e-4)


def test_hull_greeks_example():
    # Hull Tables 19.x: S=49,K=50,r=5%,sigma=20%,T=20/52 -> delta .522, gamma .066, vega 12.1 (per 1.00),
    # theta -4.31/yr, rho 8.91 (per 1.00)
    g = bsm_greeks(49, 50, 20 / 52, 0.05, 0.0, 0.20, "C")
    assert g.price == pytest.approx(2.40, abs=0.005)
    assert g.delta == pytest.approx(0.522, abs=5e-4)
    assert g.gamma == pytest.approx(0.066, abs=5e-4)
    assert g.vega * 100 == pytest.approx(12.1, abs=0.05)
    assert g.theta * 365 == pytest.approx(-4.31, abs=0.01)
    assert g.rho * 100 == pytest.approx(8.91, abs=0.01)
    pc = g.per_contract(100)
    assert pc.delta == pytest.approx(52.2, abs=0.05)


@pytest.mark.parametrize("cp", ["C", "P"])
def test_greeks_vs_finite_differences(cp):
    S, K, T, r, q, v = 101.0, 95.0, 0.37, 0.03, 0.015, 0.27
    g = bsm_greeks(S, K, T, r, q, v, cp)
    h = 1e-3
    f = lambda **kw: bsm_price(kw.get("S", S), K, kw.get("T", T), kw.get("r", r), q, kw.get("v", v), cp)
    assert g.delta == pytest.approx((f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-6)
    assert g.gamma == pytest.approx((f(S=S + h) - 2 * f() + f(S=S - h)) / h ** 2, rel=1e-4)
    assert g.vega == pytest.approx((f(v=v + 1e-5) - f(v=v - 1e-5)) / 2e-5 / 100, rel=1e-6)
    assert g.theta == pytest.approx(-(f(T=T + 1e-6) - f(T=T - 1e-6)) / 2e-6 / 365, rel=1e-5)
    assert g.rho == pytest.approx((f(r=r + 1e-6) - f(r=r - 1e-6)) / 2e-6 / 100, rel=1e-5)
    gb = black76_greeks(S, K, T, r, v, cp)
    fb = lambda F: black76_price(F, K, T, r, v, cp)
    assert gb.delta == pytest.approx((fb(S + h) - fb(S - h)) / (2 * h), rel=1e-6)


def test_put_call_parity_identity():
    rng = np.random.default_rng(1)
    S, K = rng.uniform(50, 150, 50), rng.uniform(50, 150, 50)
    T, r, q, v = rng.uniform(0.01, 2, 50), rng.uniform(0, 0.08, 50), rng.uniform(0, 0.05, 50), rng.uniform(0.05, 0.9, 50)
    c, p = bsm_price(S, K, T, r, q, v, "C"), bsm_price(S, K, T, r, q, v, "P")
    np.testing.assert_allclose(c - p, S * np.exp(-q * T) - K * np.exp(-r * T), atol=1e-10)


def test_vectorized_rights_and_expired():
    out = bsm_price([100, 100], [90, 110], [0.0, 0.0], 0.05, 0.0, 0.2, np.array(["C", "P"]))
    np.testing.assert_allclose(out, [10.0, 10.0])


def test_strike_from_delta_roundtrip():
    for cp, d in (("C", 0.25), ("P", -0.25), ("C", 0.5), ("P", -0.10)):
        K = strike_from_delta(d, 100, 0.1, 0.03, 0.01, 0.2, cp)
        assert bsm_delta(100, K, 0.1, 0.03, 0.01, 0.2, cp) == pytest.approx(d, abs=1e-10)
