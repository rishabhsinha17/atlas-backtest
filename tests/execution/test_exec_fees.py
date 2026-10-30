"""Known-answer tests for fee schedules (IBKR published examples, SEC s.31
advisories, FINRA Schedule A)."""
import math

import pandas as pd
import pytest

from atlasbt.core.types import AssetClass, ExerciseStyle, Instrument, Right, SettlementType
from atlasbt.execution import rates
from atlasbt.execution.fees import (
    IBKRProFixed, IBKRProTiered, NoFees, ZeroCommissionRetail, cboe_spx_fee, finra_taf_equity,
    finra_taf_options, ibkr_option_rate, is_cboe_spx, make_fee_schedule, sec_fee,
)

STK = Instrument("XYZ", AssetClass.STOCK)
TS = pd.Timestamp("2024-06-03 15:00", tz="UTC")


def opt(premium_root="XYZ", right=Right.CALL, settlement=SettlementType.PHYSICAL, und="XYZ"):
    return Instrument.option(und, pd.Timestamp("2024-07-19"), 100.0, right, settlement=settlement,
                             root=premium_root)


# ---------------------------------------------------------------- SEC s.31
@pytest.mark.parametrize("date,rate", [
    ("2001-06-01", 1e6 / 300 / 100),
    ("2002-01-15", 15.00),
    ("2008-05-01", 5.60),
    ("2011-01-21", 19.20),
    ("2013-05-24", 22.40), ("2013-05-25", 17.40),
    ("2021-02-24", 22.10), ("2021-02-25", 5.10),
    ("2023-02-27", 8.00),
    ("2024-05-21", 8.00), ("2024-05-22", 27.80),
    ("2025-05-13", 27.80), ("2025-05-14", 0.00),
    ("2026-04-03", 0.00), ("2026-04-04", 20.60),
])
def test_sec31_rate_table_effective_dates(date, rate):
    assert rates.SEC31.at(pd.Timestamp(date), "per_million") == pytest.approx(rate)


def test_sec_fee_known_answers():
    # $100,000 sold on 2024-06-03 at $27.80 per $1M = $2.78
    assert sec_fee(100_000, "2024-06-03") == pytest.approx(2.78)
    # $10,000 sold on 2023-06-01 at $8.00/M = $0.08
    assert sec_fee(10_000, "2023-06-01") == pytest.approx(0.08)
    # $12,345 at $27.80/M = 0.343191 -> rounded UP to $0.35
    assert sec_fee(12_345, "2024-06-03") == pytest.approx(0.35)
    assert sec_fee(12_345, "2024-06-03", round_up_to_cent=False) == pytest.approx(0.343191)
    # FY2025 zero rate
    assert sec_fee(1_000_000, "2025-06-02") == 0.0
    # tz-aware timestamps are interpreted in ET (2024-05-22 00:30 UTC is still 05-21 in New York)
    assert sec_fee(1e6, pd.Timestamp("2024-05-22 00:30", tz="UTC"), False) == pytest.approx(8.00)


def test_finra_taf():
    # 2024: $0.000166/share, max $8.30
    assert finra_taf_equity(1000, "2024-06-03") == pytest.approx(0.17)          # 0.166 rounded up
    assert finra_taf_equity(100_000, "2024-06-03") == pytest.approx(8.30)       # capped
    assert finra_taf_equity(1000, "2016-06-01", round_up_to_cent=False) == pytest.approx(0.119)
    assert finra_taf_equity(1000, "2026-11-02") == 0.0                          # Q4-2026 fee holiday
    assert finra_taf_options(10, "2024-06-03", round_up_to_cent=False) == pytest.approx(0.0279)
    # price below per-share rate -> no fee
    assert finra_taf_equity(1000, "2024-06-03", price=0.0001) == 0.0


# ---------------------------------------------------------------- IBKR fixed (published examples)
@pytest.mark.parametrize("qty,price,expected", [
    (100, 25.0, 1.00), (1000, 25.0, 5.00), (1000, 0.25, 2.50), (10, 0.20, 0.02),
])
def test_ibkr_fixed_published_examples(qty, price, expected):
    f = IBKRProFixed()
    c = f.explicit_costs(STK, qty, price, TS, "TAKER", order_id=1)
    assert c.commission == pytest.approx(expected)
    assert c.exchange_fees == 0.0           # included in Fixed
    assert c.regulatory_fees == 0.0         # buys pay no SEC/TAF


def test_ibkr_fixed_sale_passes_through_regulatory():
    f = IBKRProFixed()
    c = f.explicit_costs(STK, -1000, 50.0, TS, "TAKER", order_id=1)
    assert c.commission == pytest.approx(5.00)
    assert c.regulatory_fees == pytest.approx(sec_fee(50_000, TS) + finra_taf_equity(1000, TS))


def test_ibkr_minimum_applies_once_per_order_across_partial_fills():
    f = IBKRProFixed()
    c1 = f.explicit_costs(STK, 50, 20.0, TS, "TAKER", order_id=7)
    c2 = f.explicit_costs(STK, 50, 20.0, TS, "TAKER", order_id=7)
    assert c1.commission + c2.commission == pytest.approx(1.00)    # 100 sh * 0.005 = 0.5 -> min 1.00 total
    c3 = f.explicit_costs(STK, 300, 20.0, TS, "TAKER", order_id=7)
    assert c1.commission + c2.commission + c3.commission == pytest.approx(2.00)


# ---------------------------------------------------------------- IBKR tiered
def test_ibkr_tiered_marginal_monthly_example():
    """IBKR: 'If you execute 500,000 US shares in a month ... 300,000 shares
    at USD 0.0035, 200,000 shares at USD 0.002'."""
    f = IBKRProTiered()
    total = 0.0
    for i in range(50):    # 50 orders of 10,000 shares @ $50 in June 2024
        total += f.explicit_costs(STK, 10_000, 50.0, TS, "TAKER", order_id=i).commission
    assert total == pytest.approx(300_000 * 0.0035 + 200_000 * 0.0020)


def test_ibkr_tiered_min_and_third_party_fees():
    f = IBKRProTiered()
    c = f.explicit_costs(STK, 10, 50.0, TS, "TAKER", order_id=1)
    assert c.commission == pytest.approx(0.35)          # min per order
    # taker access fee 0.003 + clearing 0.0002 + pass-through on commission + CAT (approx row)
    cat = 10 * rates.CAT_FEE.at(TS, "per_share")
    exp = 10 * 0.003 + 10 * 0.0002 + 0.35 * (0.000175 + 0.00056) + cat
    assert c.exchange_fees == pytest.approx(exp)
    m = f.explicit_costs(STK, 10, 50.0, TS, "MAKER", order_id=2)
    assert m.exchange_fees < c.exchange_fees


def test_ibkr_tiered_month_rollover_resets_volume():
    f = IBKRProTiered()
    f.explicit_costs(STK, 400_000, 10.0, pd.Timestamp("2024-05-30 15:00", tz="UTC"), order_id=1)
    c = f.explicit_costs(STK, 1000, 10.0, pd.Timestamp("2024-06-03 15:00", tz="UTC"), order_id=2)
    assert c.commission == pytest.approx(3.50)          # back to 0.0035 tier


# ---------------------------------------------------------------- options (IBKR published examples)
@pytest.mark.parametrize("mtd,contracts,premium,expected", [
    (0, 1, 2.0, 1.00), (0, 2, 5.0, 1.30), (0, 3, 0.075, 1.50), (0, 5, 0.03, 1.25),
    (20_000, 2, 5.0, 1.00), (20_000, 3, 0.075, 1.50), (20_000, 5, 0.03, 1.25),
    (60_000, 5, 2.0, 1.25), (200_000, 7, 2.0, 1.05),
])
def test_ibkr_option_commission_published_examples(mtd, contracts, premium, expected):
    f = IBKRProTiered()
    f._roll_month(rates._as_naive_date(TS))
    f._mtd_contracts = mtd
    c = f.explicit_costs(opt(), contracts, premium, TS, "TAKER", order_id=1)
    assert c.commission == pytest.approx(expected)


def test_option_exchange_regulatory_fees():
    f = IBKRProTiered()
    c = f.explicit_costs(opt(), -10, 2.0, TS, "TAKER", order_id=1)
    occ = 10 * 0.02          # OCC $0.02 (2021-06-01 .. 2024-12-31)
    orf = 10 * rates.ORF_AGGREGATE.at(TS, "per_contract")
    assert c.exchange_fees == pytest.approx(occ + orf + 10 * f.option_exchange_fee_per_contract)
    exp_reg = sec_fee(10 * 2.0 * 100, TS) + finra_taf_options(10, TS, 2.0) + 10 * rates.CAT_FEE.at(TS, "per_contract")
    assert c.regulatory_fees == pytest.approx(exp_reg)
    assert rates.OCC_CLEARING.at("2025-02-03", "per_contract") == 0.025
    assert rates.OCC_CLEARING.at("2019-05-01", "per_contract") == 0.055


def test_cboe_spx_fees():
    spx = Instrument.option("SPX", pd.Timestamp("2026-12-17"), 6000, Right.PUT, ExerciseStyle.EUROPEAN,
                            SettlementType.CASH_AM)
    spxw = Instrument.option("SPX", pd.Timestamp("2026-12-18"), 6000, Right.PUT, ExerciseStyle.EUROPEAN,
                             SettlementType.CASH_PM, root="SPXW")
    assert is_cboe_spx(spx) == "SPX" and is_cboe_spx(spxw) == "SPXW"
    assert is_cboe_spx(opt()) is None
    d = "2026-10-15"
    assert cboe_spx_fee(spx, 2, 5.0, d) == pytest.approx(2 * (0.45 + 0.21))
    assert cboe_spx_fee(spxw, 2, 0.50, d) == pytest.approx(2 * (0.36 + 0.14))


def test_exercise_and_assignment_have_no_commission():
    f = IBKRProTiered()
    c = f.explicit_costs(STK, -100, 50.0, TS, "ASSIGNMENT")
    assert c.commission == 0.0 and c.regulatory_fees == pytest.approx(sec_fee(5000, TS))
    assert f.explicit_costs(opt(), -1, 0.0, TS, "EXERCISE").explicit == 0.0


def test_zero_commission_retail_eras():
    z = ZeroCommissionRetail()
    old = z.explicit_costs(STK, 100, 50.0, pd.Timestamp("2018-03-01 15:00", tz="UTC"), order_id=1)
    new = z.explicit_costs(STK, 100, 50.0, pd.Timestamp("2020-03-02 15:00", tz="UTC"), order_id=2)
    assert old.commission == pytest.approx(4.95)
    assert new.commission == 0.0
    sale = z.explicit_costs(STK, -100, 50.0, pd.Timestamp("2020-03-02 15:00", tz="UTC"), order_id=3)
    assert sale.regulatory_fees > 0      # SEC + TAF still passed through
    assert ZeroCommissionRetail(historical=False).explicit_costs(
        STK, 100, 50.0, pd.Timestamp("2018-03-01 15:00", tz="UTC")).commission == 0.0


def test_factory_and_nofees():
    assert make_fee_schedule("ibkr_pro_fixed").name == "IBKR_PRO_FIXED"
    with pytest.raises(ValueError):
        make_fee_schedule("nope")
    assert NoFees().explicit_costs(STK, -100, 10, TS).explicit == 0.0


def test_ibkr_option_rate_table():
    assert ibkr_option_rate(0.10, 0) == 0.65
    assert ibkr_option_rate(0.09, 0) == 0.50
    assert ibkr_option_rate(0.04, 0) == 0.25
    assert ibkr_option_rate(1.0, 150_000) == 0.15


def test_fee_estimate_does_not_mutate_state():
    f = IBKRProTiered()
    f.explicit_costs(STK, 299_000, 10.0, TS, order_id=1)
    e1 = f.estimate(STK, 5_000, 10.0, TS)
    e2 = f.estimate(STK, 5_000, 10.0, TS)
    assert e1.commission == e2.commission == pytest.approx(1000 * 0.0035 + 4000 * 0.0020)
    assert f._mtd_shares == 299_000
