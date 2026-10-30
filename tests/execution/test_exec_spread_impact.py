"""Spread estimators (known answers computed by hand from the papers'
formulas), floors, time-of-day profile, PIT shift, square-root impact."""
import math

import numpy as np
import pytest

from atlasbt.execution.impact import SquareRootImpact
from atlasbt.execution.spread import (
    SpreadConfig, SpreadModel, abdi_ranaldo_products, abdi_ranaldo_rolling, corwin_schultz,
    corwin_schultz_pair, corwin_schultz_rolling, price_bucket_floor, roll_spread, tick_size_for_price,
    tod_multiplier,
)


def test_corwin_schultz_hand_calculation():
    """H=(101, 101.2), L=(99, 99.2), no overnight gap.
    beta = ln(101/99)^2 + ln(101.2/99.2)^2 = 0.00040003 + 0.00039842 = 0.00079845
    gamma = ln(101.2/99)^2 = 0.00048307
    alpha = (sqrt(2b) - sqrt(b))/(3-2sqrt2) - sqrt(g/(3-2sqrt2))
          = 0.0117044/0.1715729 - 0.0530613 = 0.0151574
    S = 2(e^a - 1)/(1 + e^a) = 0.015157  (hand, 5 s.f.)"""
    s = corwin_schultz_pair(101, 99, 101.2, 99.2, c0=100.0)
    assert s == pytest.approx(0.015157, abs=2e-6)
    v = corwin_schultz(np.array([101, 101.2]), np.array([99, 99.2]), np.array([100.0, 100.0]))
    assert np.isnan(v[0]) and v[1] == pytest.approx(s)


def test_corwin_schultz_overnight_adjustment():
    # day 2 entirely above day-1 close (gap up 5): CS shifts day-2 H/L down by L2 - C1
    adj = corwin_schultz_pair(101, 99, 107, 105, c0=100.0, overnight_adjust=True)
    manual = corwin_schultz_pair(101, 99, 102, 100, c0=None, overnight_adjust=False)
    assert adj == pytest.approx(manual)
    raw = corwin_schultz_pair(101, 99, 107, 105, c0=100.0, overnight_adjust=False)
    assert raw < adj        # unadjusted 2-day range inflates gamma -> lower (negative) estimate
    # gap down
    dn = corwin_schultz_pair(101, 99, 95, 93, c0=100.0)
    assert dn == pytest.approx(corwin_schultz_pair(101, 99, 100, 98, c0=None, overnight_adjust=False))
    # vectorised matches scalar
    v = corwin_schultz(np.array([101, 107.0]), np.array([99, 105.0]), np.array([100, 106.0]))
    assert v[1] == pytest.approx(adj)


def test_corwin_schultz_rolling_clips_negatives():
    h = np.array([101, 101.2, 101.0, 101.2, 103.0])
    l = np.array([99, 99.2, 99.0, 99.2, 97.0])
    c = np.array([100, 100, 100, 100, 100.0])
    raw = corwin_schultz(h, l, c)
    r = corwin_schultz_rolling(h, l, c, window=4, min_periods=1)
    exp = np.mean(np.maximum(raw[1:5], 0.0))
    assert r[4] == pytest.approx(exp)


def test_abdi_ranaldo_hand_calculation():
    """eta0 = (ln101 + ln99)/2, eta1 = (ln101 + ln98)/2, c0 = ln 100.9
    c0-eta0 = 0.0090097, c0-eta1 = 0.0140542 -> 4*prod = 0.00050649 -> S = 0.022531"""
    p = abdi_ranaldo_products([101, 101], [99, 98], [100.9, 100.0])
    assert math.sqrt(p[1]) == pytest.approx(0.022531, abs=5e-6)
    s = abdi_ranaldo_rolling([101, 101], [99, 98], [100.9, 100.0], window=2, min_periods=1)
    assert s[1] == pytest.approx(math.sqrt(p[1]))


def test_roll_recovers_bounce_spread():
    """Roll (1984) model: trades at bid or ask with prob 1/2 around a constant
    value -> cov(dp_t, dp_t-1) = -S^2/4 -> estimator = S."""
    rng = np.random.default_rng(0)
    S = 0.002
    side = rng.choice([-1, 1], 5000)
    price = 100 * np.exp(side * S / 2)
    est = roll_spread(price, window=5000, min_periods=10)
    assert est[-1] == pytest.approx(S, rel=0.08)


def test_floors_and_ticks():
    assert tick_size_for_price(0.5) == 0.0001 and tick_size_for_price(1.0) == 0.01
    assert price_bucket_floor(0.5) == 0.01 and price_bucket_floor(50) == 0.0001
    m = SpreadModel()
    # tiny estimate on a $10 stock -> floored at max(4bp bucket, 1 tick = 10bp)
    assert m.full_spread(1e-6, 10.0) == pytest.approx(0.001)
    # SPY-like $500: 1bp bucket floor dominates a zero estimate (default 20bp used when NaN)
    assert m.full_spread(1e-6, 500.0) == pytest.approx(0.0001)
    assert m.full_spread(float("nan"), 500.0) == pytest.approx(SpreadConfig().default_spread)


def test_time_of_day_multiplier():
    # measured NBBO profile (illiquid curve): wide open, NARROWER into the close
    assert tod_multiplier(0, 389) == 2.9
    assert tod_multiplier(10, 370) == 1.7
    assert tod_multiplier(20, 360) == 1.5
    assert tod_multiplier(150, 240) == 1.0
    assert tod_multiplier(387, 2) == 0.8 and tod_multiplier(370, 20) == 0.83
    assert tod_multiplier(-30, 420) == 3.0          # pre-market
    assert tod_multiplier(None, None) == 1.0        # daily bar
    m = SpreadModel()
    assert m.full_spread(0.001, 100, minutes_since_open=0, minutes_to_close=389) == pytest.approx(0.0029)


def test_spread_model_is_point_in_time():
    rng = np.random.default_rng(1)
    n = 60
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    h = c * 1.01
    l = c * 0.99
    s1 = SpreadModel().series(h, l, c)
    h2, l2, c2 = h.copy(), l.copy(), c.copy()
    h2[40:] *= 1.5
    l2[40:] *= 0.7       # perturb sessions >= 40
    s2 = SpreadModel().series(h2, l2, c2)
    np.testing.assert_allclose(s1.est[:41], s2.est[:41])   # estimate for session 40 uses <= 39
    assert not np.allclose(s1.est[41:], s2.est[41:])


def test_square_root_impact():
    imp = SquareRootImpact(k=0.1)
    # 1% of ADV, sigma 2% -> 0.1 * 0.02 * 0.1 = 2bp
    assert imp.impact_fraction(1e6, 1e8, 0.02) == pytest.approx(0.0002)
    assert imp.impact_fraction(1e6, float("nan"), 0.02) == 0.0
    assert SquareRootImpact(k=0.1, max_impact=0.01).impact_fraction(1e9, 1e6, 0.5) == 0.01
