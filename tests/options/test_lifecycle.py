import numpy as np
import pandas as pd
import pytest

from atlasbt.core.types import ExerciseStyle, Instrument, Right, SettlementType
from atlasbt.options.lifecycle import (ExpiryPolicy, adjust_for_special_dividend, adjust_for_split,
                                       early_assignment_events, early_exercise_rational, expiration_events,
                                       settlement_due)
from atlasbt.pricing.time import settlement_instant, year_fraction


def opt(K, right, expiry="2024-03-15", st=SettlementType.PHYSICAL, ex=ExerciseStyle.AMERICAN, und="XYZ", root=None):
    return Instrument.option(und, pd.Timestamp(expiry), K, Right(right), ex, st, root=root)


def ref(prices):
    return lambda u, s, kind: prices[kind]


def test_physical_exercise_by_exception_threshold():
    c_itm, c_penny, c_otm = opt(100, "C"), opt(100.99, "C"), opt(101.01, "C")
    pos = {c_itm: 2, c_penny: -1, c_otm: 3}
    ev = {e.instrument.strike: e for e in expiration_events(pos, "2024-03-15", ref({"close": 101.0}))}
    assert ev[100].kind == "EXERCISE" and ev[100].stock_qty == 200 and ev[100].cash == -200 * 100
    assert ev[100.99].kind == "ASSIGNMENT" and ev[100.99].stock_qty == -100      # ITM by exactly $0.01
    assert ev[101.01].kind == "EXPIRE_WORTHLESS" and ev[101.01].option_qty == -3
    fills = ev[100].to_fills()
    assert fills[0].quantity == -2 and fills[0].price == 0 and fills[1].quantity == 200 and fills[1].price == 100
    assert fills[1].liquidity == "EXERCISE"


def test_puts_physical():
    p = opt(100, "P")
    ev = expiration_events({p: -1}, "2024-03-15", ref({"close": 95.0}))[0]
    assert ev.kind == "ASSIGNMENT" and ev.stock_qty == 100 and ev.cash == -10000


def test_not_due_on_other_days():
    assert expiration_events({opt(100, "C"): 1}, "2024-03-14", ref({"close": 101.0})) == []


def test_spx_am_settlement_uses_soq_next_session():
    # SPX March 2024 AM: last trading day Thu 2024-03-14, settles at SOQ Fri 2024-03-15
    c = opt(5100, "C", "2024-03-14", SettlementType.CASH_AM, ExerciseStyle.EUROPEAN, "SPX")
    assert not settlement_due(c, "2024-03-14") and settlement_due(c, "2024-03-15")
    ev = expiration_events({c: -1}, "2024-03-15", ref({"soq": 5130.0, "close": 5000.0}))[0]
    assert ev.kind == "CASH_SETTLEMENT" and ev.cash == pytest.approx(-30.0 * 100)
    assert ev.reference_price == 5130.0 and ev.stock_qty == 0
    f = ev.to_fills()[0]
    assert f.price == pytest.approx(30.0) and f.quantity == 1 and f.liquidity == "SETTLEMENT"
    assert settlement_instant(c.expiry, c.settlement) == pd.Timestamp("2024-03-15 13:30", tz="UTC")


def test_spxw_pm_settlement_close_and_0dte_time():
    p = opt(5000, "P", "2024-03-12", SettlementType.CASH_PM, ExerciseStyle.EUROPEAN, "SPX", root="SPXW")
    ev = expiration_events({p: 2}, "2024-03-12", ref({"close": 4990.0, "soq": 0.0}))[0]
    assert ev.cash == pytest.approx(2 * 10 * 100)
    T = year_fraction(pd.Timestamp("2024-03-12 14:00", tz="UTC"), p.expiry, p.settlement)  # 10:00 ET
    assert T == pytest.approx(6 / (365 * 24))


def test_early_close_settlement_time():
    # 2024-11-29 (day after Thanksgiving) closes 13:00 ET
    assert settlement_instant(pd.Timestamp("2024-11-29"), "PHYSICAL") == pd.Timestamp("2024-11-29 18:00", tz="UTC")


def test_pin_risk_adverse_policy():
    c = opt(100, "C")
    occ = expiration_events({c: -1}, "2024-03-15", ref({"close": 99.97}))[0]
    adv = expiration_events({c: -1}, "2024-03-15", ref({"close": 99.97}), ExpiryPolicy(pin_band=0.05, pin_policy="adverse"))[0]
    assert occ.kind == "EXPIRE_WORTHLESS" and adv.kind == "ASSIGNMENT"


def test_early_assignment_call_before_ex_dividend():
    c = opt(90, "C", "2024-04-19")
    # 2024-03-07 ex-date tomorrow (session 2024-03-06), dividend 1.00, call extrinsic 0.20 -> assigned
    ok, why = early_exercise_rational(c, 100.0, 10.20, "2024-03-06", 0.05, [(pd.Timestamp("2024-03-07"), 1.0)])
    assert ok and "ex-div" in why
    ok2, _ = early_exercise_rational(c, 100.0, 11.50, "2024-03-06", 0.05, [(pd.Timestamp("2024-03-07"), 1.0)])
    assert not ok2
    ev = early_assignment_events({c: -3}, "2024-03-06", {"XYZ": 100.0}, {c.symbol: 10.2}, 0.05,
                                 {"XYZ": [(pd.Timestamp("2024-03-07"), 1.0)]})
    assert len(ev) == 1 and ev[0].stock_qty == -300 and ev[0].kind == "EARLY_ASSIGNMENT"


def test_early_assignment_deep_itm_put():
    p = opt(150, "P", "2024-09-20")
    ok, _ = early_exercise_rational(p, 100.0, 50.05, "2024-03-06", 0.05)   # extrinsic .05 < carry ~3.9
    assert ok
    ok2, _ = early_exercise_rational(p, 140.0, 13.0, "2024-03-06", 0.05)   # extrinsic 3 vs carry ~3.9 -> exercise
    assert ok2
    ok3, _ = early_exercise_rational(p, 145.0, 10.0, "2024-03-06", 0.05)   # extrinsic 5 > carry
    assert not ok3
    eu = opt(150, "P", "2024-09-20", SettlementType.CASH_PM, ExerciseStyle.EUROPEAN)
    assert not early_exercise_rational(eu, 100.0, 50.0, "2024-03-06", 0.05)[0]


def test_split_and_special_dividend_adjustments():
    c = opt(300, "C", "2024-06-21")
    n, q = adjust_for_split(c, 2, 4.0)
    assert n.strike == 75 and q == 8 and n.deliverable_shares == 100
    n2, q2 = adjust_for_split(c, 2, 1.5)
    assert n2.strike == 300 and q2 == 2 and n2.deliverable_shares == 150 and n2.symbol.startswith("XYZ1")
    n3, _ = adjust_for_split(c, 1, 0.1)    # 1:10 reverse split
    assert n3.deliverable_shares == pytest.approx(10)
    ev = expiration_events({n2: 1}, "2024-06-21", ref({"close": 310.0}))[0]
    assert ev.stock_qty == 150
    assert adjust_for_special_dividend(c, 0.10).strike == 300      # $10/contract < $12.50
    assert adjust_for_special_dividend(c, 5.00).strike == 295


def test_early_exercise_never_treats_missing_mark_as_zero_extrinsic():
    from atlasbt.options.lifecycle import OptionMark
    p = opt(150, "P", "2024-09-20")
    assert not early_exercise_rational(p, 100.0, float("nan"), "2024-03-06", 0.05)[0]
    stale = OptionMark(50.05, pd.Timestamp("2024-03-05 21:00", tz="UTC"))
    assert not early_exercise_rational(p, 100.0, stale, "2024-03-06", 0.05)[0]
    same = OptionMark(50.05, pd.Timestamp("2024-03-06 20:00", tz="UTC"))
    assert early_exercise_rational(p, 100.0, same, "2024-03-06", 0.05)[0]
    model = OptionMark(float("nan"), None, model_value=50.05)
    ok, why = early_exercise_rational(p, 100.0, model, "2024-03-06", 0.05)
    assert ok and "PIT model value" in why
