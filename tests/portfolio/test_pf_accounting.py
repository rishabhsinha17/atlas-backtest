"""Accounting, corporate actions, financing, margin and option lifecycle."""
import math

import pandas as pd
import pytest

from atlasbt.core.types import (
    ActionType, AssetClass, CorporateAction, CostBreakdown, ExerciseStyle, Fidelity, Fill, Instrument,
    Right, SettlementType,
)
from atlasbt.portfolio.account import Account, AccountType
from atlasbt.portfolio.corporate_actions import CorporateActionConfig, CorporateActionProcessor, as_session_date
from atlasbt.portfolio.financing import FinancingConfig, accrue_financing
from atlasbt.portfolio.hooks import DefaultOptionLifecycle
from atlasbt.portfolio.ledger import Reason
from atlasbt.portfolio.margin import ConservativeOptionMargin, RegTMarginModel

STK = Instrument("XYZ", AssetClass.STOCK)
TS = pd.Timestamp("2024-03-01 15:00", tz="UTC")


def fill(q, px, inst=STK, comm=0.0, ts=TS):
    return Fill(1, inst, ts, q, px, CostBreakdown(commission=comm), Fidelity.T1_BAR)


def acct(cash=100_000.0, kind=AccountType.REG_T_MARGIN):
    return Account(cash, kind, start_ts=TS)


# ------------------------------------------------------------------ account
def test_average_cost_realized_pnl_and_flip():
    a = acct()
    a.apply_fill(fill(100, 10.0, comm=1.0))
    a.apply_fill(fill(100, 12.0))
    assert a.get("XYZ").avg_price == pytest.approx(11.0)
    r = a.apply_fill(fill(-150, 13.0))
    assert r == pytest.approx(150 * 2.0)
    r = a.apply_fill(fill(-100, 9.0))            # closes 50 (+ -100 pnl), opens short 50 @ 9
    assert r == pytest.approx(50 * (9.0 - 11.0))
    p = a.get("XYZ")
    assert p.qty == -50 and p.avg_price == 9.0
    a.mark("XYZ", 8.0)
    assert a.nav == pytest.approx(a.cash + (-50 * 8.0))
    assert a.check_invariants() == []
    assert a.cash == pytest.approx(a.ledger.total)
    by = a.ledger.by_reason()
    assert by["COMMISSION"] == -1.0 and by["DEPOSIT"] == 100_000


def test_option_multiplier_in_cash_and_nav():
    o = Instrument.option("XYZ", pd.Timestamp("2024-06-21"), 50, Right.CALL)
    a = acct()
    a.apply_fill(fill(2, 3.0, inst=o))
    assert a.cash == pytest.approx(100_000 - 600)
    a.mark(o.symbol, 4.0)
    assert a.nav == pytest.approx(100_000 + 200)


# ------------------------------------------------------------------ corporate actions
def ca(t, ratio=1.0, amount=0.0, pay=None, new=None, ex="2024-03-04"):
    return CorporateAction("XYZ", t, pd.Timestamp(ex), pd.Timestamp("2024-02-01", tz="UTC"),
                           ratio=ratio, amount=amount, pay_date=pd.Timestamp(pay) if pay else None, new_symbol=new)


def test_split_conserves_nav():
    a = acct()
    a.apply_fill(fill(100, 50.0))
    a.mark("XYZ", 60.0)
    nav0 = a.nav
    CorporateActionProcessor().apply(ca(ActionType.SPLIT, ratio=2.0), a, TS, pd.Timestamp("2024-03-04"))
    p = a.get("XYZ")
    assert p.qty == 200 and p.avg_price == 25.0 and p.mark == 30.0
    assert a.nav == pytest.approx(nav0)


def test_reverse_split_cash_in_lieu():
    a = acct()
    a.apply_fill(fill(105, 2.0))
    a.mark("XYZ", 2.0)
    nav0 = a.nav
    CorporateActionProcessor().apply(ca(ActionType.SPLIT, ratio=0.1), a, TS, pd.Timestamp("2024-03-04"))
    p = a.get("XYZ")
    assert p.qty == 10 and p.mark == pytest.approx(20.0)
    assert a.ledger.by_reason()["SPLIT_CASH_IN_LIEU"] == pytest.approx(0.5 * 20.0)
    assert a.nav == pytest.approx(nav0)


def test_fractional_split_allowed():
    a = acct()
    a.apply_fill(fill(105, 2.0))
    CorporateActionProcessor(CorporateActionConfig(fractional_shares=True)).apply(
        ca(ActionType.SPLIT, ratio=0.1), a, TS, pd.Timestamp("2024-03-04"))
    assert a.get("XYZ").qty == pytest.approx(10.5)


def test_dividend_entitlement_at_ex_cash_at_pay_and_withholding():
    a = acct()
    a.apply_fill(fill(100, 50.0))
    a.mark("XYZ", 50.0)
    nav0 = a.nav
    proc = CorporateActionProcessor(CorporateActionConfig(withholding_rate=0.15))
    proc.apply(ca(ActionType.CASH_DIVIDEND, amount=0.50, pay="2024-03-20"), a, TS, pd.Timestamp("2024-03-04"))
    a.mark("XYZ", 49.50)                   # price drops by the dividend on the ex-date
    assert a.cash == pytest.approx(100_000 - 5000)          # nothing paid yet
    assert a.nav == pytest.approx(nav0 - 50 * 0 - 7.5)      # receivable 42.5 offsets the 50 drop, minus tax
    assert proc.pay_due(a, pd.Timestamp("2024-03-19"), TS) == []
    proc.pay_due(a, pd.Timestamp("2024-03-20"), TS)
    by = a.ledger.by_reason()
    assert by["DIVIDEND"] == pytest.approx(50.0) and by["WITHHOLDING_TAX"] == pytest.approx(-7.5)
    assert a.accruals == [] and a.check_invariants() == []


def test_short_pays_dividend_in_lieu():
    a = acct()
    a.apply_fill(fill(-100, 50.0))
    a.mark("XYZ", 50.0)
    proc = CorporateActionProcessor()
    proc.apply(ca(ActionType.CASH_DIVIDEND, amount=1.0, pay="2024-03-04"), a, TS, pd.Timestamp("2024-03-04"))
    proc.pay_due(a, pd.Timestamp("2024-03-04"), TS)
    assert a.ledger.by_reason()["DIVIDEND_IN_LIEU"] == pytest.approx(-100.0)


def test_delisting_and_symbol_change():
    a = acct()
    a.apply_fill(fill(100, 10.0))
    res = CorporateActionProcessor().apply(ca(ActionType.DELIST, amount=0.37), a, TS, pd.Timestamp("2024-03-04"))
    assert a.get("XYZ") is None and res.closed_qty == -100
    assert a.ledger.by_reason()["DELIST_CASHOUT"] == pytest.approx(37.0)
    assert a.realized_total == pytest.approx(100 * (0.37 - 10))
    b = acct()
    b.apply_fill(fill(100, 10.0))
    CorporateActionProcessor().apply(ca(ActionType.SYMBOL_CHANGE, new="XYZN"), b, TS, pd.Timestamp("2024-03-04"))
    assert b.position("XYZN") == 100 and b.get("XYZN").instrument.symbol == "XYZN" and b.position("XYZ") == 0
    c = acct()
    c.apply_fill(fill(100, 10.0))
    c.mark("XYZ", 10.0)
    res = CorporateActionProcessor().apply(ca(ActionType.DELIST, amount=float("nan")), c, TS, pd.Timestamp("2024-03-04"))
    assert res.fidelity == Fidelity.T2_PROXY


def test_as_session_date_conventions():
    assert as_session_date(pd.Timestamp("2024-03-04")) == pd.Timestamp("2024-03-04")
    assert as_session_date(pd.Timestamp("2024-03-04", tz="UTC")) == pd.Timestamp("2024-03-04")
    assert as_session_date(pd.Timestamp("2024-03-04 21:00", tz="UTC")) == pd.Timestamp("2024-03-04")


# ------------------------------------------------------------------ financing
def test_weekend_financing_accrues_three_days():
    a = acct(cash=1_000_000)
    res = accrue_financing(a, TS, days=3, rf=0.05, borrow_fee=lambda s: 0.0, cfg=FinancingConfig())
    assert res.credit == pytest.approx(1_000_000 * 0.045 * 3 / 360)
    assert a.ledger.by_reason()["CASH_INTEREST"] == pytest.approx(375.0)


def test_debit_interest_and_borrow_fee():
    a = acct(cash=10_000)
    a.apply_fill(fill(300, 50.0))                 # cash -5,000 -> margin loan
    a.mark("XYZ", 50.0)
    res = accrue_financing(a, TS, 1, 0.04, lambda s: 0.0, FinancingConfig())
    assert res.debit == pytest.approx(-5000 * 0.055 / 360)
    b = acct(cash=100_000)
    b.apply_fill(fill(-1000, 20.0))
    b.mark("XYZ", 20.0)
    r = accrue_financing(b, TS, 3, 0.0, lambda s: 0.02, FinancingConfig())
    assert r.borrow == pytest.approx(-20_000 * 0.02 * 3 / 360)
    assert r.credit == 0.0                     # rf - spread floored at 0
    assert r.by_symbol["XYZ"] == pytest.approx(r.borrow)
    # short proceeds earn no credit interest (retail): base = cash - |short MV| = 100k
    c = acct(cash=100_000)
    c.apply_fill(fill(-1000, 20.0))
    c.mark("XYZ", 20.0)
    r = accrue_financing(c, TS, 1, 0.05, lambda s: 0.0, FinancingConfig())
    assert r.credit == pytest.approx(100_000 * 0.045 / 360)


# ------------------------------------------------------------------ margin
def test_reg_t_initial_and_maintenance():
    m = RegTMarginModel()
    a = acct(cash=100_000)
    ok, _ = m.check_order(a, STK, 3000, 50.0, lambda s: 50.0)     # $150k long: 75k initial <= 100k
    assert ok
    ok, why = m.check_order(a, STK, 5000, 50.0, lambda s: 50.0)   # $250k: 125k > 100k
    assert not ok and "Reg-T" in why
    ok, _ = m.check_order(a, STK, -3000, 50.0, lambda s: 50.0)    # short 150k -> 75k initial
    assert ok
    a.apply_fill(fill(4000, 50.0))
    a.mark("XYZ", 50.0)
    s = m.snapshot(a, lambda s: 50.0)
    assert s.initial == pytest.approx(100_000) and s.maintenance == pytest.approx(50_000)
    a.mark("XYZ", 33.0)                                            # equity 32k, maint 33k
    s = m.snapshot(a, lambda s: 33.0)
    assert s.is_call
    plan = m.liquidation_plan(a, s)
    (p, dq), = plan
    assert dq < 0 and -dq <= 4000
    # after selling, maintenance <= equity / 1.1
    remaining = 4000 + dq
    assert remaining * 33 * 0.25 <= s.equity / 1.1 + 33 * 0.25


def test_short_maintenance_low_priced():
    m = RegTMarginModel()
    a = acct()
    a.apply_fill(fill(-1000, 4.0))
    a.mark("XYZ", 4.0)
    assert m.snapshot(a, lambda s: 4.0).maintenance == pytest.approx(max(2.5 * 1000, 4000))
    b = acct()
    b.apply_fill(fill(-1000, 10.0))
    b.mark("XYZ", 10.0)
    assert m.snapshot(b, lambda s: 10.0).maintenance == pytest.approx(5000)   # $5/share > 30%


def test_cash_account_rules():
    m = RegTMarginModel()
    a = acct(cash=10_000, kind=AccountType.CASH)
    assert not m.check_order(a, STK, -10, 50.0, lambda s: 50.0)[0]
    assert not m.check_order(a, STK, 300, 50.0, lambda s: 50.0)[0]
    assert m.check_order(a, STK, 200, 50.0, lambda s: 50.0)[0]
    put = Instrument.option("XYZ", pd.Timestamp("2024-06-21"), 40, Right.PUT)
    assert m.check_order(a, put, -2, 1.0, lambda s: 50.0)[0]        # 8,000 secured by 10,002 cash
    assert not m.check_order(a, put, -3, 1.0, lambda s: 50.0)[0]
    call = Instrument.option("XYZ", pd.Timestamp("2024-06-21"), 60, Right.CALL)
    assert not m.check_order(a, call, -1, 1.0, lambda s: 50.0)[0]  # naked call


def test_option_margin_naked_spread_covered():
    om = ConservativeOptionMargin()
    put = Instrument.option("XYZ", pd.Timestamp("2024-06-21"), 45, Right.PUT)
    a = acct()
    a.apply_fill(fill(-1, 2.0, inst=put))
    a.mark(put.symbol, 2.0)
    # max(2 + 0.2*50 - 5, 2 + 0.1*45) * 100 = max(7, 6.5) * 100
    assert om.requirement(a, lambda s: 50.0) == pytest.approx(700.0)
    long_put = Instrument.option("XYZ", pd.Timestamp("2024-06-21"), 40, Right.PUT)
    a.apply_fill(fill(1, 1.0, inst=long_put))
    a.mark(long_put.symbol, 1.0)
    assert om.requirement(a, lambda s: 50.0) == pytest.approx(500.0)     # put credit spread width
    call = Instrument.option("XYZ", pd.Timestamp("2024-06-21"), 55, Right.CALL)
    b = acct()
    b.apply_fill(fill(100, 50.0))
    b.apply_fill(fill(-1, 1.0, inst=call))
    b.mark(call.symbol, 1.0)
    assert om.requirement(b, lambda s: 50.0) == 0.0                      # covered call
    assert RegTMarginModel().snapshot(b, lambda s: 50.0).maintenance > 0  # stock still margined


# ------------------------------------------------------------------ option lifecycle
class _P:
    def __init__(self, close=None, open_=None):
        self._c, self._o = close, open_

    def close(self, s, d):
        return self._c

    def open(self, s, d):
        return self._o


def test_default_lifecycle_physical_and_cash():
    lc = DefaultOptionLifecycle()
    exp = pd.Timestamp("2024-06-21")
    call = Instrument.option("XYZ", exp, 50, Right.CALL)
    a = acct()
    a.apply_fill(fill(-2, 1.0, inst=call))
    ev = lc.at_session_end(exp, a.option_positions(), _P(close=55.0))
    assert ev[0].kind == "ASSIGNMENT" and ev[0].underlying_qty == -200 and ev[0].underlying_price == 50
    ev = lc.at_session_end(exp, a.option_positions(), _P(close=50.005))     # ITM by < $0.01
    assert ev[0].kind == "EXPIRE_WORTHLESS"
    assert lc.at_session_end(exp - pd.Timedelta(days=1), a.option_positions(), _P(close=60.0)) == []
    spx = Instrument.option("SPX", pd.Timestamp("2024-06-20"), 5000, Right.PUT, ExerciseStyle.EUROPEAN,
                            SettlementType.CASH_AM)
    b = acct()
    b.apply_fill(fill(1, 10.0, inst=spx))
    assert lc.at_session_end(pd.Timestamp("2024-06-20"), b.option_positions(), _P(close=4900.0)) == []
    ev = lc.at_session_start(pd.Timestamp("2024-06-21"), pd.Timestamp("2024-06-20"), b.option_positions(),
                             _P(open_=4950.0))
    assert ev[0].kind == "CASH_SETTLE" and ev[0].option_price == pytest.approx(50.0)
    assert ev[0].fidelity == Fidelity.T2_PROXY


def test_non_standard_deliverable_exercise_and_missing_price_fallbacks():
    from dataclasses import replace
    lc = DefaultOptionLifecycle()
    exp = pd.Timestamp("2024-06-21")
    nonstd = replace(Instrument.option("XYZ", exp, 100.0, Right.CALL, root="XYZ1"), deliverable_shares=150.0)
    a = acct()
    a.apply_fill(fill(1, 5.0, inst=nonstd))
    ev = lc.at_session_end(exp, a.option_positions(), _P(close=80.0))       # 150 sh x 80 = 12,000 > 10,000
    assert ev[0].kind == "EXERCISE" and ev[0].underlying_qty == 150
    assert ev[0].underlying_price == pytest.approx(100.0 * 100 / 150)       # aggregate strike 10,000 / 150 sh

    class _Last(_P):
        def last_close(self, s):
            return 120.0
    std = Instrument.option("XYZ", exp, 100.0, Right.CALL)
    b = acct()
    b.apply_fill(fill(-1, 5.0, inst=std))
    ev = lc.at_session_end(exp, b.option_positions(), _Last(close=float("nan")))
    assert ev[0].kind == "ASSIGNMENT" and ev[0].fidelity == Fidelity.T2_PROXY
    b.mark(std.symbol, 7.0)
    ev = lc.at_session_end(exp, b.option_positions(), _P(close=float("nan")))
    assert ev[0].kind == "SETTLE_AT_MARK" and ev[0].option_price == 7.0


def test_long_options_have_no_loan_value_and_long_dated_75pct():
    m = RegTMarginModel()
    a = acct()
    near = Instrument.option("XYZ", pd.Timestamp("2024-06-21"), 50, Right.CALL)
    far = Instrument.option("XYZ", pd.Timestamp("2025-06-20"), 50, Right.CALL)
    a.apply_fill(fill(1, 10.0, inst=near))
    a.apply_fill(fill(1, 10.0, inst=far))
    a.mark(near.symbol, 10.0)
    a.mark(far.symbol, 10.0)
    snap = m.snapshot(a, lambda s: 50.0, as_of=TS)
    assert snap.initial == pytest.approx(1000 + 750)


def test_dividend_without_pay_date_assumes_ex_plus_7_rolled():
    from atlasbt.core.clock import nyse
    cal = nyse()
    a = acct()
    a.apply_fill(fill(100, 50.0))
    proc = CorporateActionProcessor(roll_to_session=lambda d: d if cal.is_session(d) else cal.next_session(d))
    res = proc.apply(ca(ActionType.CASH_DIVIDEND, amount=0.5, ex="2024-03-01"), a, TS, pd.Timestamp("2024-03-01"))
    assert res.fidelity == Fidelity.T2_PROXY and "assumed ex+7d" in res.note
    assert a.accruals[0].pay_date == pd.Timestamp("2024-03-08")
    res = proc.apply(ca(ActionType.CASH_DIVIDEND, amount=0.5, ex="2024-03-02"), a, TS, pd.Timestamp("2024-03-02"))
    assert a.accruals[1].pay_date == pd.Timestamp("2024-03-11")          # Sat 03-09 rolled to Monday
