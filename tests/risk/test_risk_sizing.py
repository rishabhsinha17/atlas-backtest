"""The 13 sizing profiles (Atlas sheet 'Sizing Profiles')."""
import math

import pandas as pd
import pytest

from atlasbt.core.types import AssetClass, Instrument, Right
from atlasbt.risk import sizing as S


def test_profile_text_is_verbatim_and_complete():
    assert len(S.PROFILE_TEXT) == 13 and set(S.PROFILE_TEXT) == set(S.PROFILES)
    assert S.PROFILE_TEXT["DAY_RISK"].startswith("Risk at initial planned stop = 0.25% of prior-day NAV")


def test_day_risk():
    # 0.25% of 1,000,000 = 2,500 risk; stop distance 0.50 -> 5,000 shares; cap 10% NAV @ $50 = 2,000
    assert S.size_day_risk(1_000_000, 50.0, 49.5) == 2000
    assert S.size_day_risk(1_000_000, 50.0, 48.0) == 1250          # risk-bound
    assert S.size_day_risk(1_000_000, 50.0, 52.0) == -1250         # short when stop above entry
    assert S.size_day_risk(1_000_000, 50.0, 49.5, existing_qty=1500) == 500
    t = pd.Timestamp("2024-03-04 15:16", tz="America/New_York")
    assert S.size_day_risk(1_000_000, 50.0, 48.0, now_et=t) == 0      # no new entries after 15:15
    assert S.size_day_risk(1_000_000, 50.0, 48.0, now_et=t.replace(minute=15)) == 1250
    assert S.size_day_risk(1_000_000, 50.0, 50.0) == 0


def test_day_notional():
    assert S.size_day_notional(1_000_000, 40.0) == 2500
    assert S.size_day_notional(1_000_000, 40.0, side=-1) == -2500
    assert S.size_day_notional(1_000_000, 40.0, existing_qty=10) == 0     # one position per instrument


def test_swing_risk_caps():
    assert S.size_swing_risk(1_000_000, 100.0, 95.0) == 1000             # 5,000 risk / 5 = 1000; 100k notional
    assert S.size_swing_risk(1_000_000, 100.0, 99.0) == 1000             # notional cap binds (5,000 sh -> 1,000)
    assert S.size_swing_risk(1_000_000, 100.0, 95.0, current_gross=950_000) == 500   # gross cap 100%
    assert S.size_swing_risk(1_000_000, 100.0, 105.0) == -1000


def test_swing_notional():
    assert S.size_swing_notional(1_000_000, 50.0) == 2000
    assert S.size_swing_notional(1_000_000, 50.0, side=-1) == -1000
    assert S.size_swing_notional(1_000_000, 50.0, current_gross=980_000) == 400   # no implicit leverage


def test_cross_section_weights():
    longs = [f"L{i}" for i in range(30)]
    shorts = [f"S{i}" for i in range(20)]
    w = S.cross_section_weights(longs, shorts)
    assert w["L0"] == pytest.approx(0.50 / 30) and w["S0"] == pytest.approx(-0.02)   # 2.5% capped to 2%
    sectors = {s: ("TECH" if i < 8 else f"X{i}") for i, s in enumerate(longs)}
    w = S.cross_section_weights(longs, [], sectors)
    tech = sum(v for k, v in w.items() if sectors[k] == "TECH")
    assert tech == pytest.approx(0.10)
    assert w["L10"] == pytest.approx(0.50 / 30)
    net = S.cross_section_weights(["A", "B"], ["C"], {"A": "T", "B": "T", "C": "T"}, long_gross=0.5,
                                  short_gross=0.5, name_cap=0.2, sector_cap=0.1, sector_cap_mode="net")
    assert sum(net.values()) == pytest.approx(0.1)
    shares = S.weights_to_shares({"A": 0.02, "B": -0.02}, 1_000_000, {"A": 33.0, "B": 7.0})
    assert shares == {"A": 606, "B": -2857}


def test_portfolio_weights_residual_cash():
    w, resid = S.portfolio_weights({"SPY": 0.6, "TLT": 0.6, "GLD": -0.1})
    assert sum(w.values()) == pytest.approx(1.0) and resid == pytest.approx(0.0) and w["GLD"] == 0
    w, resid = S.portfolio_weights({"SPY": 0.3, "TLT": 0.3})
    assert resid == pytest.approx(0.4)
    assert S.rebalance_deltas({"A": 100, "B": 0}, {"A": 60, "B": 10, "C": 5}) == {"A": 40, "B": -10, "C": -5}


def test_pairs_dollar_neutral():
    l, s = S.size_pairs(1_000_000, 50.0, 25.0)
    assert l == 1000 and s == -2000


def _opt(k, right, exp="2024-07-19", mult=100.0):
    i = Instrument.option("XYZ", pd.Timestamp(exp), k, right)
    if mult != 100:
        from dataclasses import replace
        i = replace(i, multiplier=mult, deliverable_shares=mult)
    return i


def test_opt_debit():
    assert S.size_opt_debit(1_000_000, 350.0) == 14             # 5,000 / 350
    assert S.size_opt_debit(1_000_000, 350.0, existing_option_debit=48_000) == 5
    assert S.verify_contract(_opt(100, Right.CALL))[0]
    assert not S.verify_contract(_opt(100, Right.CALL, mult=150))[0]


def test_opt_credit_defined_risk_and_max_loss():
    legs = [S.OptLeg(_opt(95, Right.PUT), -1, 2.0), S.OptLeg(_opt(90, Right.PUT), 1, 0.8)]
    assert S.is_defined_risk(legs)[0]
    ml = S.max_loss_vertical(legs)                     # width 5 - credit 1.2 = 3.8 * 100
    assert ml == pytest.approx(380.0)
    n, coll = S.size_opt_credit(1_000_000, ml, legs=legs)
    assert n == 13 and coll == pytest.approx(13 * 380)
    naked = [S.OptLeg(_opt(95, Right.PUT), -2, 2.0), S.OptLeg(_opt(90, Right.PUT), 1, 0.8)]
    assert not S.is_defined_risk(naked)[0] and S.size_opt_credit(1_000_000, 380, legs=naked) == (0, 0.0)
    condor = [S.OptLeg(_opt(90, Right.PUT), 1, 0.5), S.OptLeg(_opt(95, Right.PUT), -1, 1.5),
              S.OptLeg(_opt(105, Right.CALL), -1, 1.5), S.OptLeg(_opt(110, Right.CALL), 1, 0.5)]
    assert S.max_loss_vertical(condor) == pytest.approx((5 - 2.0) * 100)
    assert math.isinf(S.max_loss_vertical([S.OptLeg(_opt(105, Right.CALL), -1, 1.0)]))


def test_opt_naked_never_zero_margin():
    put = _opt(95, Right.PUT)
    loss = S.stress_loss_short_option(put, 100.0, 1.0)     # S -20% -> 80: intrinsic 15 + extr 2 - 1 = 16
    assert loss == pytest.approx(1600.0)
    assert S.size_opt_naked(1_000_000, loss, margin_per_contract=1900.0) == 6
    with pytest.raises(ValueError):
        S.size_opt_naked(1_000_000, loss, margin_per_contract=0.0)


def test_opt_covered():
    buy, calls = S.size_opt_covered(1_000_000, 45.0)        # cap 100k/45 = 2222 -> 22 lots
    assert buy == 2200 and calls == 22
    buy, calls = S.size_opt_covered(1_000_000, 45.0, existing_shares=1000)
    assert buy == 1200 and calls == 22


def test_opt_cashsec_no_double_counting():
    book = S.CollateralBook()
    n = S.size_opt_cashsec(100_000, 40.0, collateral=book)
    assert n == 25
    book.reserve("pkg1", 25 * 4000)
    assert S.size_opt_cashsec(100_000, 40.0, collateral=book) == 0
    assert S.size_opt_cashsec(1_000_000, 40.0, nav=1_000_000) == 25      # 10% name cap


def test_opt_arbitrage_gate():
    ok = S.arbitrage_edge(105.0, 100.0, fees=2.0, borrow_cost=1.0, funding_cost=0.5, margin_cost=0.5)
    assert ok.ok and ok.edge == pytest.approx(1.0)
    assert not S.arbitrage_edge(105.0, 100.0, fees=6.0).ok
    assert not S.arbitrage_edge(105.0, 100.0, shorts_located=False).ok
