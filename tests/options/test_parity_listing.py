import numpy as np
import pandas as pd
import pytest

from atlasbt.options.listing import has_0dte, listed_expiries, monthly_expiration_day, strike_grid, tick_size
from atlasbt.options.parity import (american_parity_bounds, box_implied_rate, box_vs_tbill, conversion_edge,
                                    european_parity, parity_screen, reversal_edge)
from atlasbt.pricing import bsm_price, tbill_discount_to_cc


def test_spxw_daily_listing_history():
    assert has_0dte("SPX", "2023-06-13")          # Tuesday 2023: dailies exist
    assert not has_0dte("SPX", "2015-06-09")      # Tuesday 2015: no
    assert has_0dte("SPX", "2015-06-12")          # Friday weekly PM
    assert has_0dte("SPX", "2016-03-02")          # first Wednesday SPXW
    assert not has_0dte("SPX", "2016-02-24")
    assert has_0dte("SPX", "2016-08-22") and not has_0dte("SPX", "2016-08-15")  # first Monday weekly
    assert not has_0dte("SPX", "2022-04-19") and has_0dte("SPX", "2022-04-26")
    assert has_0dte("SPX", "2022-05-19") and not has_0dte("SPX", "2022-05-12")
    assert not has_0dte("SPX", "2008-03-14")      # 2008 weeklies were AM-settled (no PM 0DTE)


def test_spy_listing_history():
    assert has_0dte("SPY", "2022-11-15") and not has_0dte("SPY", "2022-11-08")
    assert has_0dte("SPY", "2010-06-11") and not has_0dte("SPY", "2010-06-04")
    assert not has_0dte("SPY", "2009-06-12")
    assert has_0dte("SPY", "2018-02-26")


def test_holiday_shifted_monthly_and_spx_am():
    # Good Friday 2014-04-18 -> equity monthly last trading day Thursday 2014-04-17
    assert monthly_expiration_day(2014, 4) == pd.Timestamp("2014-04-17")
    df = listed_expiries("SPX", "2014-04-01", horizon_days=30)
    am = df[(df["root"] == "SPX") & (df["settlement"] == "CASH_AM")].iloc[0]
    assert am["settlement_date"] == pd.Timestamp("2014-04-17") and am["expiry"] == pd.Timestamp("2014-04-16")


def test_spx_no_pm_weekly_in_standard_week_before_2017():
    df = listed_expiries("SPX", "2015-06-01", horizon_days=30)
    assert not ((df["expiry"] == pd.Timestamp("2015-06-19")) & (df["root"] == "SPXW")).any()
    df2 = listed_expiries("SPX", "2018-06-01", horizon_days=30)
    assert ((df2["expiry"] == pd.Timestamp("2018-06-15")) & (df2["root"] == "SPXW")).any()


def test_strike_grid_and_ticks():
    g = strike_grid("SPX", 4312.3, "2023-06-13", 30, 0.15)
    assert np.all(np.mod(g, 5) == 0) and g.min() < 4312 < g.max()
    assert np.all(np.diff(g[(g > 4312 * 0.86) & (g < 4312 * 1.14)]) == 5)
    assert tick_size("SPY", 5.0, "2020-01-02") == 0.01
    assert tick_size("XYZ", 5.0, "2020-01-02") == 0.05
    assert tick_size("XYZ", 1.0, "2005-01-03") == 0.05


def test_european_parity_and_american_bounds():
    S, K, T, r = 100.0, 95.0, 0.5, 0.05
    divs = [(0.25, 1.0)]
    c = bsm_price(S - 1.0 * np.exp(-r * 0.25), K, T, r, 0, 0.3, "C")
    p = bsm_price(S - 1.0 * np.exp(-r * 0.25), K, T, r, 0, 0.3, "P")
    assert c - p == pytest.approx(european_parity(S, K, T, r, divs))
    lo, hi = american_parity_bounds(S, K, T, r, divs)
    assert lo <= c - p <= hi


def test_conversion_reversal_edges_fair_market_zero():
    S, K, T, r = 100.0, 100.0, 0.25, 0.04
    c = float(bsm_price(S, K, T, r, 0, 0.25, "C"))
    p = float(bsm_price(S, K, T, r, 0, 0.25, "P"))
    conv = conversion_edge(S, c, p, K, T, r, american=False)
    rev = reversal_edge(S, c, p, K, T, r, american=False)
    assert conv.hold_to_expiry == pytest.approx(0, abs=1e-10) and rev.hold_to_expiry == pytest.approx(0, abs=1e-10)
    # mispriced call (too rich) -> conversion profitable; spreads/fees kill small edges
    s = parity_screen(99.99, 100.01, c + 0.5, c + 0.55, p - 0.05, p, K, T, r, min_edge_bp=10)
    assert s["signal"] == "conversion"
    s2 = parity_screen(99.95, 100.05, c - 0.05, c + 0.05, p - 0.05, p + 0.05, K, T, r, min_edge_bp=10)
    assert s2["signal"] is None
    # borrow fee hurts reversals
    r0 = reversal_edge(100, c - 0.3, p, K, T, r, borrow_fee=0.0, american=False)
    r20 = reversal_edge(100, c - 0.3, p, K, T, r, borrow_fee=0.20, american=False)
    assert r20.hold_to_expiry < r0.hold_to_expiry


def test_conversion_dividend_early_assignment_worst_case():
    S, K, T, r = 100.0, 80.0, 0.25, 0.03
    divs = [(0.1, 2.0)]
    e = conversion_edge(S, 20.5, 0.2, K, T, r, divs)
    assert e.worst_case < e.hold_to_expiry   # losing the dividend to early assignment


def test_box_rate():
    width, T = 100.0, 0.5
    price = width * np.exp(-0.05 * T)
    assert box_implied_rate(price, width, T)["cc"] == pytest.approx(0.05)
    q = dict(c1_bid=30.0, c1_ask=30.2, c2_bid=5.0, c2_ask=5.2, p1_bid=1.0, p1_ask=1.2, p2_bid=72.4, p2_ask=72.6)
    out = box_vs_tbill(q, 4000, 4100, T, tbill_cc=float(tbill_discount_to_cc(0.05, 182)), fee_per_contract=0.5)
    assert out["lend_rate_cc"] < out["mid_rate_cc"] < out["borrow_rate_cc"]
    assert out["terminal_value_usd"] == 10000.0
