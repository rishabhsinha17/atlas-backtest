"""IBKR Pro dated, tiered financing schedule (FinancingConfig(schedule='ibkr_pro')).

Known answers hand-computed from the dated tables in atlasbt/execution/rates.py
(IBG annual reports 2012-2021, Attachment C July 2017, live IBKR pages 2026-10-30)."""
import math

import pandas as pd
import pytest

from atlasbt.core.types import AssetClass, CostBreakdown, Fidelity, Fill, Instrument
from atlasbt.engine import BacktestConfig, Backtester, Strategy
from atlasbt.engine.testing import InMemoryPortal, synthetic_daily_bars
from atlasbt.execution.impact import SquareRootImpact
from atlasbt.execution.spread import SpreadConfig
from atlasbt.portfolio.account import Account, AccountType
from atlasbt.portfolio.financing import FinancingConfig, accrue_financing, credit_eligibility

STK = Instrument("XYZ", AssetClass.STOCK)
IB = FinancingConfig(schedule="ibkr_pro")


def ts(d):
    return pd.Timestamp(d + " 21:00", tz="UTC")


def acct(cash, t):
    return Account(cash, AccountType.REG_T_MARGIN, start_ts=t)


def fill(q, px, t):
    return Fill(1, STK, t, q, px, CostBreakdown(), Fidelity.T1_BAR)


def test_credit_nothing_on_first_10k_and_bm_minus_half():
    t = ts("2024-06-03")
    a = acct(1_000_000, t)
    r = accrue_financing(a, t, 1, rf=0.05, borrow_fee=lambda s: 0.0, cfg=IB, benchmark=0.0533)
    assert r.credit == pytest.approx(990_000 * (0.0533 - 0.005) / 360)
    assert r.rf == 0.0533


def test_credit_floor_zero_when_rates_near_zero():
    t = ts("2014-06-02")
    a = acct(1_000_000, t)
    r = accrue_financing(a, t, 3, rf=0.0003, borrow_fee=lambda s: 0.0, cfg=IB, benchmark=0.0009)
    assert r.credit == 0.0 and "CASH_INTEREST" not in a.ledger.by_reason()


def test_nav_eligibility_cutoff_then_pro_rata():
    assert credit_eligibility(50_000, ts("2017-07-14")) == 0.0          # July 2017: NAV must exceed 100K
    assert credit_eligibility(50_000, ts("2020-02-10")) == pytest.approx(0.5)   # pro rata from 2020-02-10
    assert credit_eligibility(150_000, ts("2010-01-04")) == 1.0
    t = ts("2024-06-03")
    a = acct(50_000, t)
    r = accrue_financing(a, t, 1, rf=0.05, borrow_fee=lambda s: 0.0, cfg=IB, benchmark=0.05)
    assert r.credit == pytest.approx(40_000 * 0.045 * 0.5 / 360)


def test_debit_blended_tiers_by_era():
    for d, expect in (("2016-06-01", 100_000 * 0.015 + 900_000 * 0.010 + 500_000 * 0.005),
                      ("2024-06-03", 100_000 * 0.015 + 900_000 * 0.010 + 500_000 * 0.0075)):
        t = ts(d)
        a = acct(500_000, t)
        a.apply_fill(fill(40_000, 50.0, t))                  # $2M stock, $1.5M loan
        a.mark("XYZ", 50.0)
        r = accrue_financing(a, t, 1, rf=0.0, borrow_fee=lambda s: 0.0, cfg=IB, benchmark=0.02)
        assert r.debit == pytest.approx(-(1.5e6 * 0.02 + expect) / 360)


def test_short_proceeds_tiers_and_borrow_fee():
    t = ts("2024-06-03")
    a = acct(1_000_000, t)
    a.apply_fill(fill(-40_000, 50.0, t))                     # $2M short
    a.mark("XYZ", 50.0)
    r = accrue_financing(a, t, 1, rf=0.05, borrow_fee=lambda s: 0.003, cfg=IB, benchmark=0.05)
    # cash interest on cash net of proceeds: 1M - 10K at BM-0.5%
    assert r.credit == pytest.approx(990_000 * 0.045 / 360)
    # proceeds: 0 on 100K, 900K at 3.75%, 1M at 4.5%
    assert r.rebate == pytest.approx((900_000 * 0.0375 + 1_000_000 * 0.045) / 360)
    assert r.borrow == pytest.approx(-2e6 * 0.003 / 360)
    assert r.by_symbol["XYZ"] == pytest.approx(r.rebate + r.borrow)
    # flat (default) schedule: rebate 0, unchanged behaviour
    b = acct(1_000_000, t)
    b.apply_fill(fill(-40_000, 50.0, t))
    b.mark("XYZ", 50.0)
    assert accrue_financing(b, t, 1, 0.05, lambda s: 0.003, FinancingConfig()).rebate == 0.0


def test_unknown_schedule_raises():
    t = ts("2024-06-03")
    with pytest.raises(ValueError):
        accrue_financing(acct(1e6, t), t, 1, 0.05, lambda s: 0.0, FinancingConfig(schedule="x"))


class _TenorPortal(InMemoryPortal):
    """Fed Funds ('ON') 2%, every other tenor 5%: proves the IBKR schedule reads the ON benchmark."""

    def risk_free_rate(self, as_of, tenor="3M"):
        return 0.02 if tenor == "ON" else 0.05


class Idle(Strategy):
    id = "IDLE"
    bar_freq = "1d"
    universe = ["XYZ"]

    def on_bar(self, ctx):
        pass


def _run(fin, start="2024-03-04", end="2024-03-08"):
    df = synthetic_daily_bars(start, end, seed=1)
    p = _TenorPortal(bars={("XYZ", "1d"): df})
    cfg = BacktestConfig(fees="NO_FEES", financing=fin, impact=SquareRootImpact(k=0.0),
                         spread=SpreadConfig(bucket_floors=((math.inf, 0.0),), default_spread=1e-12,
                                             tod_open=(), tod_close=(), tick_floor=False))
    return Backtester(Idle(), p, cfg).run(start, end, raise_errors=True)


def test_engine_uses_fed_funds_benchmark_and_keeps_rf_meta():
    r = _run(IB)
    led = r.ledger[r.ledger.reason == "CASH_INTEREST"]
    assert led.amount.iloc[0] == pytest.approx(990_000 * 0.015 / 360)
    assert r.meta["rf_daily"].iloc[0] == pytest.approx(0.05 / 360)        # Sharpe rf still rf_tenor
    assert r.meta["config"]["financing_schedule"] == "ibkr_pro"
    flat = _run(FinancingConfig())
    assert flat.ledger[flat.ledger.reason == "CASH_INTEREST"].amount.iloc[0] == pytest.approx(1e6 * 0.045 / 360)
    assert "financing_schedule" not in flat.meta["config"]


def test_libor_era_flagged():
    r = _run(IB, "2005-03-01", "2005-03-04")
    reasons = [x["reason"] for x in r.fidelity["reasons"]]
    assert any("LIBOR era" in x for x in reasons)
