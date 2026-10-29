"""DTB3 discount-basis conversion (known answers) and FRED PIT publication rule."""
import pandas as pd
import pytest

from atlasbt.data import rates
from atlasbt.data.providers import fred

from .conftest import FIX, utc


def test_discount_conversions_known_answer():
    # textbook: 91-day bill at 5.00% discount -> price 98.7361, MMY 5.0640%, BEY 5.1343%
    assert rates.discount_to_price(0.05, 91) == pytest.approx(0.9873611, abs=1e-7)
    assert rates.discount_to_money_market(0.05, 91) == pytest.approx(0.050640, abs=1e-6)
    assert rates.discount_to_bond_equivalent(0.05, 91) == pytest.approx(0.051343, abs=1e-6)
    assert rates.discount_to_continuous(0.05, 91) == pytest.approx(0.051018, abs=1e-6)
    # money-market yield is always above the discount rate
    assert rates.discount_to_money_market(0.02, 91) > 0.02


def test_fred_parse_and_known_at():
    df = fred.parse_csv("DTB3", (FIX / "fred_DTB3.csv").read_bytes())
    assert pd.Timestamp("2023-12-25") not in df.index          # holiday blanks dropped
    # value for Fri 2023-12-29 is published next business day (Tue 2024-01-02) ~16:15 ET
    assert df.loc[pd.Timestamp("2023-12-29"), "known_at"] == utc("2024-01-02 21:30")
    assert df.loc[pd.Timestamp("2024-01-02"), "known_at"] == utc("2024-01-03 21:30")
    assert df.loc[pd.Timestamp("2024-01-02"), "value"] == 5.22


def test_portal_risk_free_rate_pit(store, portal):
    df = fred.parse_csv("DTB3", (FIX / "fred_DTB3.csv").read_bytes())
    store.write("rates", "fred", "DTB3", df)
    r = portal.risk_free_rate(utc("2024-01-03 21:29"), "3M")       # only the 2023-12-29 print is public
    assert r == pytest.approx(rates.discount_to_money_market(0.0520, 91), abs=1e-9)
    assert r == pytest.approx(0.052693, abs=1e-6)
    r = portal.risk_free_rate(utc("2024-01-03 21:30"), "3M")
    assert r == pytest.approx(rates.discount_to_money_market(0.0522, 91), abs=1e-9)
    # 1M tenor falls back to 3M when DTB4WK is absent
    assert portal.risk_free_rate(utc("2024-01-03 21:30"), "1M") == pytest.approx(r)


def test_risk_free_rate_info_fallback_flagged(store, portal):
    """3M from the real DTB3 fixture; synthetic_ DTB4WK starting 2024-01-02: before it exists the
    1M request falls back to 3M and is flagged T2; afterwards the native series is T1."""
    from atlasbt.core.types import Fidelity
    store.write("rates", "fred", "DTB3", fred.parse_csv("DTB3", (FIX / "fred_DTB3.csv").read_bytes()))
    wk = pd.DataFrame({"value": [5.27, 5.28]}, index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="date"))
    wk["known_at"] = fred.known_at_for(wk.index)
    store.write("rates", "fred", "DTB4WK", wk)
    r, sid, fid, obs = portal.risk_free_rate_info(utc("2024-01-02 22:00"), "1M")
    assert sid == "DTB3" and fid == Fidelity.T2_PROXY
    r, sid, fid, obs = portal.risk_free_rate_info(utc("2024-01-03 21:30"), "1M")
    assert sid == "DTB4WK" and fid == Fidelity.T1_BAR and obs == pd.Timestamp("2024-01-02")
    assert r == pytest.approx(rates.discount_to_money_market(0.0527, 28))
    assert portal.risk_free_rate(utc("2024-01-03 21:30"), "1M") == r
    # every series stale (> 10 days): freshest value returned, flagged T2
    r, sid, fid, obs = portal.risk_free_rate_info(utc("2024-02-15"), "1M")
    assert fid == Fidelity.T2_PROXY and obs == pd.Timestamp("2024-01-09") and sid == "DTB3"
