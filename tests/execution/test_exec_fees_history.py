"""Point-in-time IBKR Pro commission history (opt-in) and dated IBKR interest / PDT / settlement
tables. Known answers are hand-computed from the dated sources cited in execution/rates.py."""
import pandas as pd
import pytest

from atlasbt.core.types import AssetClass, Instrument, Right
from atlasbt.execution import rates
from atlasbt.execution.fees import IBKRProFixed, IBKRProTiered, ibkr_option_rate, make_fee_schedule

STK = Instrument("XYZ", AssetClass.STOCK)
OPT = Instrument.option("XYZ", pd.Timestamp("2016-07-15"), 100.0, Right.CALL, root="XYZ")


def ts(d):
    return pd.Timestamp(d + " 15:00", tz="UTC")


def test_history_table_is_point_in_time():
    row = rates.IBKR_PRO_HISTORY.row_at
    assert row("2010-01-04").approx                       # before the first verified date
    assert row("2014-02-03").values["opt_top"] == 0.70 and not row("2014-02-03").approx
    assert row("2019-09-25").values["opt_top"] == 0.70
    assert row("2019-09-26").values["opt_top"] == 0.65
    assert row("2017-03-02").values["tiered_min"] == 0.70
    assert row("2026-10-30").values["tiered_min"] == 0.35


def test_default_schedule_unchanged_on_old_dates():
    """Default (historical=False) keeps the current schedule for every date (golden runs)."""
    t = IBKRProTiered()
    assert t.name == "IBKR_PRO_TIERED"
    # 10 sh @ $50 in 2016: 10*0.0035 = 0.035 -> min 0.35
    assert t.explicit_costs(STK, 10, 50.0, ts("2016-06-01"), order_id=1).commission == pytest.approx(0.35)
    o = IBKRProTiered()
    assert o.explicit_costs(OPT, 10, 1.0, ts("2016-06-01"), order_id=2).commission == pytest.approx(6.50)


def test_historical_tiered_stock_minimum_and_option_rate():
    t = make_fee_schedule("IBKR_PRO_TIERED_HISTORICAL")
    assert isinstance(t, IBKRProTiered) and t.historical and t.name == "IBKR_PRO_TIERED_HISTORICAL"
    # 2016: USD 0.70 minimum per order below 100M shares/month (IBG AR 2013/2014)
    assert t.explicit_costs(STK, 10, 50.0, ts("2016-06-01"), order_id=1).commission == pytest.approx(0.70)
    # 1,000 sh: 3.50 > min -> unaffected by the era
    assert t.explicit_costs(STK, 1000, 50.0, ts("2016-06-01"), order_id=2).commission == pytest.approx(3.50)
    # 2016 options: $0.70/contract (IBG AR 2016: "$0.00 base, $0.70 per contract")
    assert t.explicit_costs(OPT, 10, 1.0, ts("2016-06-01"), order_id=3).commission == pytest.approx(7.00)
    # low-premium tiers keep the 0.50 / 0.25 structure; min USD 1.00 per order
    assert t.explicit_costs(OPT, 1, 0.07, ts("2016-06-01"), order_id=4).commission == pytest.approx(1.00)
    # after the repricing the current schedule applies
    assert t.explicit_costs(STK, 10, 50.0, ts("2024-06-03"), order_id=5).commission == pytest.approx(0.35)
    assert t.explicit_costs(OPT, 10, 1.0, ts("2024-06-03"), order_id=6).commission == pytest.approx(6.50)


def test_historical_fixed_schedule():
    f = make_fee_schedule("IBKR_PRO_FIXED_HISTORICAL")
    assert isinstance(f, IBKRProFixed) and f.name == "IBKR_PRO_FIXED_HISTORICAL"
    assert f.explicit_costs(STK, 100, 25.0, ts("2015-06-01"), order_id=1).commission == pytest.approx(1.00)
    assert f.explicit_costs(OPT, 2, 2.0, ts("2015-06-01"), order_id=2).commission == pytest.approx(1.40)


def test_option_rate_top_override():
    assert ibkr_option_rate(0.50, 0, top=0.70) == 0.70
    assert ibkr_option_rate(0.07, 0, top=0.70) == 0.50
    assert ibkr_option_rate(0.50, 20_000, top=0.70) == 0.50


def test_ibkr_interest_tiers_known_answers():
    tb = rates.tiered_blend
    rows = rates.IBKR_PRO_INTEREST.row_at
    old, new = rows("2016-06-01").values, rows("2024-06-03").values
    # 2013-2020 tiers: $1.5M debit at BM 0.14% -> blended APR 0.14% + (150k+900k+250k... ) / 1.5M
    # = 0.14% + (100k*1.5% + 900k*1.0% + 500k*0.5%) / 1.5M = 0.14% + 0.8667% (IBG AR 2012: 0.99%)
    assert tb(1.5e6, old["debit"], 0.0014, +1) / 1.5e6 == pytest.approx(0.0014 + 0.0130 / 1.5, rel=1e-9)
    assert tb(1.5e6, old["debit"], 0.0014, +1) / 1.5e6 == pytest.approx(0.0099, abs=2e-4)
    # $3.5M at 2013 tiers: 0.14% + 2.175% / 3.5 = 0.761% (IBG AR 2012: 0.76%)
    assert tb(3.5e6, old["debit"], 0.0014, +1) / 3.5e6 == pytest.approx(0.0076, abs=2e-4)
    # current tiers at BM 0.08% (IBG AR 2021, 2022-01): $1.5M 1.03%, $3.5M 0.91%
    assert tb(1.5e6, new["debit"], 0.0008, +1) / 1.5e6 == pytest.approx(0.0103, abs=2e-4)
    assert tb(3.5e6, new["debit"], 0.0008, +1) / 3.5e6 == pytest.approx(0.0091, abs=2e-4)
    # short-proceeds tiers (Attachment C July 2017, BM 1.16%): 0 / BM-1.25% (floored) / BM-0.5% / BM-0.25%
    assert tb(5e5, old["short_proceeds"], 0.0116, -1) == 0.0
    assert tb(2e6, old["short_proceeds"], 0.0116, -1) == pytest.approx(1e6 * 0.0066)
    assert old["credit_threshold"] == 10_000 and old["credit_spread"] == 0.005
    assert not old["nav_proration"] and new["nav_proration"]


def test_pdt_and_settlement_constants():
    assert rates.PDT_EFFECTIVE == pd.Timestamp("2001-09-28") and rates.PDT_MIN_EQUITY == 25_000
    assert rates.settlement_days("1995-06-06") == 5
    assert rates.settlement_days("2017-09-01") == 3
    assert rates.settlement_days("2017-09-05") == 2
    assert rates.settlement_days("2024-05-27") == 2
    assert rates.settlement_days(pd.Timestamp("2024-05-28 15:00", tz="UTC")) == 1
