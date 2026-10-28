"""Known-answer tests: Cboe Margin Manual (Nov 30, 2021) worked examples.
Each asserts the manual's 'Margin Requirement' and 'SMA Debit or Margin Call'."""
import pandas as pd
import pytest

from atlasbt.options.margin import OptionPosition as P, naked_requirement, strategy_margin, stress_loss

E1 = pd.Timestamp("2025-11-21")
E2 = pd.Timestamp("2025-12-19")


def m(positions, S, **kw):
    return strategy_margin(positions, S, **kw)


def test_naked_put_equity_minimum_applies():
    # Short 1 Sep 80 put at 2, underlying 95 -> requirement 1,000, call 800
    r = m([P(80, "P", E1, -1, 2.0)], 95)
    assert r.requirement == pytest.approx(1000.0) and r.net_requirement == pytest.approx(800.0)


def test_naked_put_equity_itm_basic():
    r = m([P(20, "P", E1, -1, 1.50)], 19.50)
    assert r.requirement == pytest.approx(540.0) and r.net_requirement == pytest.approx(390.0)


def test_naked_put_leveraged_etf():
    r = m([P(725, "P", E1, -1, 3.0)], 970.0, leverage=2.0)
    assert r.requirement == pytest.approx(14800.0)
    r = m([P(400, "P", E1, -1, 15.50)], 390.70, leverage=3.0)
    assert r.requirement == pytest.approx(24992.0)


def test_naked_put_broad_index():
    assert m([P(410, "P", E1, -1, 0.10)], 445.35, underlying_class="broad_index").requirement == pytest.approx(4110.0)
    r = m([P(430, "P", E1, -1, 7.80)], 433.35, underlying_class="broad_index")
    assert r.requirement == pytest.approx(6945.25) and r.net_requirement == pytest.approx(6165.25)


def test_naked_calls():
    r = m([P(120, "C", E1, -1, 8.40)], 128.50)
    assert r.requirement == pytest.approx(3410.0) and r.net_requirement == pytest.approx(2570.0)
    assert m([P(30, "C", E1, -1, 0.05)], 26.38).requirement == pytest.approx(268.80)
    assert m([P(430, "C", E1, -1, 8.70)], 433.35, underlying_class="broad_etf").requirement == pytest.approx(7370.25)
    assert m([P(1250, "C", E1, -1, 2.0)], 1050.30, leverage=1.5).requirement == pytest.approx(15954.50)
    assert naked_requirement("C", 45, 1.35, 43.34, "broad_index") == pytest.approx(619.10)


def test_debit_put_spread_no_loss():
    r = m([P(250, "P", E1, 1, 3.0), P(240, "P", E1, -1, 0.95)], 255)
    assert r.requirement == pytest.approx(300.0) and r.net_requirement == pytest.approx(205.0)


def test_credit_put_spread_index():
    r = m([P(425, "P", E1, 1, 6.40), P(430, "P", E1, -1, 7.80)], 433.35, underlying_class="broad_index")
    assert r.requirement == pytest.approx(1140.0) and r.net_requirement == pytest.approx(360.0)


def test_credit_call_spread_equity():
    r = m([P(125, "C", E1, 1, 3.80), P(120, "C", E1, -1, 8.40)], 128.50)
    assert r.requirement == pytest.approx(880.0) and r.net_requirement == pytest.approx(40.0)


def test_calendar_long_expires_before_short_is_not_a_spread():
    r = m([P(70, "C", pd.Timestamp("2025-03-21"), 1, 5.0), P(70, "C", pd.Timestamp("2025-06-20"), -1, 8.0)], 75)
    assert r.requirement == pytest.approx(2800.0) and r.net_requirement == pytest.approx(2000.0)
    r = m([P(425, "C", E1, 1, 13.10), P(430, "C", E2, -1, 12.20)], 433.35, underlying_class="broad_index")
    assert r.requirement == pytest.approx(9030.25)


def test_calendar_long_expires_after_short_is_a_spread():
    # long later-dated same-strike call covers the short: requirement = long premium only
    r = m([P(100, "C", E1, -1, 2.0), P(100, "C", E2, 1, 3.5)], 100)
    assert r.requirement == pytest.approx(350.0) and r.net_requirement == pytest.approx(150.0)
    # diagonal: long strike above short -> loss = strike difference
    r = m([P(100, "C", E1, -1, 2.0), P(105, "C", E2, 1, 1.5)], 100)
    assert r.requirement == pytest.approx(500.0 + 150.0)


def test_short_straddles():
    r = m([P(90, "C", E1, -1, 7.0), P(90, "P", E1, -1, 3.70)], 92.63)
    assert r.requirement == pytest.approx(2922.60) and r.net_requirement == pytest.approx(1852.60)
    r = m([P(435, "P", E1, -1, 7.20), P(435, "C", E1, -1, 5.50)], 433.35, underlying_class="broad_index")
    assert r.requirement == pytest.approx(7770.25) and r.net_requirement == pytest.approx(6500.25)
    r = m([P(45, "P", E1, -1, 2.85), P(45, "C", E1, -1, 1.35)], 43.34, underlying_class="broad_index")
    assert r.requirement == pytest.approx(1070.10)


def test_covered_call_proceeds_applied():
    # Long 100 at 92.38, short Dec 90 call at 7: stock 4,619 (Reg T, portfolio), option net -700 -> call 3,919
    r = m([P(90, "C", E1, -1, 7.0)], 92.38, stock_shares=100)
    assert r.requirement == pytest.approx(0.0) and r.net_requirement == pytest.approx(-700.0)
    assert 0.5 * 92.38 * 100 + r.net_requirement == pytest.approx(3919.0)


def test_covered_put_short_stock():
    r = m([P(250, "P", E1, -1, 3.0)], 255, stock_shares=-100)
    assert r.requirement == pytest.approx(0.0)
    assert 1.5 * 255 * 100 - 255 * 100 + r.net_requirement == pytest.approx(12450.0)


def test_boxes():
    long_box = [P(535, "C", E1, 1, 19.40), P(545, "C", E1, -1, 12.20), P(545, "P", E1, 1, 5.30), P(535, "P", E1, -1, 3.0)]
    # loan value not utilised (as in the manual's second example): longs paid in full
    r = m([P(40, "C", E1, 1, 15.40), P(50, "C", E1, -1, 7.20), P(50, "P", E1, 1, 1.75), P(40, "P", E1, -1, 0.40)], 50)
    assert r.requirement == pytest.approx(1715.0) and r.net_requirement == pytest.approx(955.0)
    short_box = [P(50, "C", E1, -1, 17.30), P(60, "C", E1, 1, 8.40), P(60, "P", E1, -1, 1.25), P(50, "P", E1, 1, 0.30)]
    r = m(short_box, 66)
    assert r.requirement == pytest.approx(1870.0) and r.net_requirement == pytest.approx(15.0)
    assert m(long_box, 550).requirement == pytest.approx(1940 + 530)


def test_butterflies_condors_iron():
    r = m([P(540, "P", E1, 1, 5.60), P(550, "P", E1, -2, 7.20), P(555, "P", E1, 1, 9.80)], 550)
    assert r.requirement == pytest.approx(2040.0) and r.net_requirement == pytest.approx(600.0)
    r = m([P(545, "C", E1, 1, 12.40), P(550, "C", E1, -2, 8.80), P(565, "C", E1, 1, 2.0)], 550)
    assert r.requirement == pytest.approx(2440.0) and r.net_requirement == pytest.approx(680.0)
    r = m([P(1050, "P", E1, 1, 47.10), P(1075, "P", E1, -1, 55.70), P(1100, "P", E1, -1, 66.30), P(1125, "P", E1, 1, 85.40)], 1160)
    assert r.requirement == pytest.approx(13250.0) and r.net_requirement == pytest.approx(1050.0)
    r = m([P(22.5, "C", E1, 1, .45), P(25, "C", E1, -1, .75), P(30, "C", E1, -1, 2.75), P(32.5, "C", E1, 1, 4.30)], 26.75)
    assert r.requirement == pytest.approx(475.0) and r.net_requirement == pytest.approx(125.0)
    r = m([P(16, "P", E1, 1, .10), P(20, "P", E1, -1, .20), P(20, "C", E1, -1, 7.0), P(24, "C", E1, 1, 4.0)], 26.75)
    assert r.requirement == pytest.approx(810.0) and r.net_requirement == pytest.approx(90.0)
    r = m([P(1000, "P", E1, 1, 32.0), P(1025, "P", E1, -1, 35.10), P(1150, "C", E1, -1, 9.40), P(1175, "C", E1, 1, 6.30)], 1060)
    assert r.requirement == pytest.approx(6330.0) and r.net_requirement == pytest.approx(1880.0)


def test_nonsymmetric_iron_condor_max_loss_table():
    # Manual p.37: long 5x60P, short 5x70P, short 10x85C, long 10x90C -> $5,000 loss computed
    r = m([P(60, "P", E1, 5, 0.0), P(70, "P", E1, -5, 0.0), P(85, "C", E1, -10, 0.0), P(90, "C", E1, 10, 0.0)], 78)
    assert r.requirement == pytest.approx(5000.0)


def test_ratio_backspread_and_unbalanced_ratio():
    # 1x2 call backspread: bounded, no loss above; loss between strikes counted
    r = m([P(100, "C", E1, -1, 5.0), P(110, "C", E1, 2, 1.5)], 100)
    assert r.requirement == pytest.approx(1000.0 + 300.0)
    # 1x2 front ratio (short 2): one short is naked
    r = m([P(100, "C", E1, 1, 5.0), P(110, "C", E1, -2, 1.5)], 100)
    assert r.requirement == pytest.approx(500.0 + naked_requirement("C", 110, 1.5, 100))


def test_jade_lizard_and_risk_reversal():
    # short put naked + call credit spread: the call spread loss is a group; put naked
    r = m([P(95, "P", E1, -1, 2.0), P(105, "C", E1, -1, 2.0), P(110, "C", E1, 1, 0.8)], 100)
    assert r.requirement == pytest.approx(500.0 + 80.0 + naked_requirement("P", 95, 2.0, 100))
    r = m([P(95, "P", E1, -1, 2.0), P(105, "C", E1, 1, 2.0)], 100)
    assert r.requirement == pytest.approx(200.0 + naked_requirement("P", 95, 2.0, 100))


def test_cash_account():
    r = m([P(80, "P", E1, -1, 2.0)], 95, account="cash")
    assert r.net_requirement == pytest.approx(8000.0) and r.cash_account_ok
    r2 = m([P(80, "P", E1, -1, 2.0)], 95, account="cash", apply_proceeds_in_cash_account=True)
    assert r2.net_requirement == pytest.approx(7800.0)
    assert m([P(100, "C", E1, -1, 2.0)], 95, account="cash").cash_account_ok is False
    assert m([P(100, "C", E1, -1, 2.0)], 95, account="cash", stock_shares=100).cash_account_ok


def test_long_option_over_9_months():
    r = m([P(80, "C", pd.Timestamp("2027-06-18"), 1, 12.0)], 78, now=pd.Timestamp("2025-12-01"))
    assert r.requirement == pytest.approx(900.0)
    r = m([P(80, "C", pd.Timestamp("2026-03-20"), 1, 12.0)], 78, now=pd.Timestamp("2025-12-01"))
    assert r.requirement == pytest.approx(1200.0)


def test_stress_loss_short_strangle_vs_condor():
    now = pd.Timestamp("2025-10-20")
    strangle = [P(90, "P", E1, -1, 1.0, iv=0.25), P(110, "C", E1, -1, 1.0, iv=0.25)]
    condor = strangle + [P(80, "P", E1, 1, 0.3, iv=0.3), P(120, "C", E1, 1, 0.3, iv=0.22)]
    s1 = stress_loss(strangle, 100, now)
    s2 = stress_loss(condor, 100, now)
    assert s1["worst_loss"] > s2["worst_loss"] > 0
    assert s2["worst_loss"] <= 1000.0 + 1e-6 + 100  # bounded by width (+ small time value effects)
    assert len(s1["table"]) == 8
