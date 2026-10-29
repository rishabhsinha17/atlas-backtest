"""Independent audit of the identity-layer fix (DATA.md §8.1) against the REAL cache.

Each test asserts the CORRECT behaviour; a failure is an audit finding. All tests skip when the
real store is absent. Reference values are DoltHub as-traded prints already in the store, SEC
registrant names, and public deal terms quoted in the docstrings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlasbt.data import config, frames
from atlasbt.data.portal import Portal
from atlasbt.data.store import ParquetStore

END = pd.Timestamp("2026-10-27 23:00", tz="UTC")


@pytest.fixture(scope="module")
def portal():
    if not (config.STORE_ROOT / "bars_1d" / "yahoo").is_dir():
        pytest.skip("real cache absent")
    return Portal(ParquetStore(config.STORE_ROOT))


def _utc(d):
    return pd.Timestamp(d, tz="UTC")


def _bars_or_none(p, sym, start, end, as_of, adjusted=False):
    from atlasbt.core.types import DataUnavailableError
    try:
        return p.bars(sym, "1d", start, end, as_of=as_of, adjusted=adjusted)
    except DataUnavailableError:
        return None


def _dolt_close(p, sym, d):
    x = p.store.read("bars_1d", "dolthub", sym)
    x.index = frames.to_naive_date(x.index)
    return float(x.loc[pd.Timestamp(d), "close"])


# ------------------------------------------------------------------ engine path (window-end identity)
@pytest.mark.parametrize("sym,day,pit_sid", [
    ("BBT", "2019-12-02", "CIK0000092230"),   # BB&T; as_of=2026 resolves 'BBT' to another registrant ($31 vs $54)
    ("IR", "2019-06-03", "CIK0001466258"),    # Ingersoll-Rand plc; as_of=2026 -> Gardner Denver/IR Inc ($34 vs $121)
    ("CB", "2010-06-01", None),               # old Chubb (fails closed PIT); as_of=2026 -> ACE/Chubb Ltd prices
    ("CCU", "2005-01-04", None),              # Clear Channel (fails closed PIT); as_of=2026 -> Cervecerias Unidas
    ("AIT", "1998-06-01", None),              # Ameritech (fails closed PIT); as_of=2026 -> Applied Industrial Tech
])
def test_engine_window_identity_matches_pit_identity(portal, sym, day, pit_sid):
    """MarketData.bars / Context.history load the whole run with portal.bars(start, range_end,
    as_of=range_end), which resolves the ticker's identity at the END of the run. A bar dated
    `day` must belong to the security the ticker denoted on `day` (or be absent) -- never to the
    company that reused the ticker later."""
    b = _bars_or_none(portal, sym, _utc(day), END, END)
    if b is None:
        return
    row = b[frames.to_naive_date(b.index) == pd.Timestamp(day)]
    if not len(row):
        return
    assert b.attrs.get("security_id") == pit_sid, (
        f"{sym} {day}: bar served from {b.attrs.get('security_id')} (close {float(row['close'].iloc[0]):.2f}) "
        f"when the run ends 2026; PIT identity is {pit_sid}")


# ------------------------------------------------------------------ carried evidence without a member name
def test_citicorp_cci_not_crown_castle(portal):
    """CCI was Citicorp's ticker until the 1998-10-08 Travelers merger (S&P member). Crown Castle
    (CIK 1051470) IPO'd in 1998 as TWRS and only later took CCI; its carried-back ZipLime ticker
    must not attribute Citicorp's member-sessions to Crown Castle (no member name -> no veto)."""
    b = _bars_or_none(portal, "CCI", _utc("1998-09-14"), _utc("1998-09-16"), _utc("1998-09-15 23:00"))
    assert b is None or not len(b) or b.attrs.get("security_id") != "CIK0001051470", (
        f"CCI 1998-09-15 served Crown Castle at {b['close'].round(2).tolist()}")


def test_unm_1998_is_not_provident(portal):
    """1998-08-31..1999-07-01 the S&P held BOTH 'PVT' (Provident Companies, CIK 5513) and 'UNM'
    (UNUM Corp, a different registrant). 'UNM' must not be served as CIK 5513 (SEC name at the
    date: PROVIDENT COMPANIES INC /DE/)."""
    b = _bars_or_none(portal, "UNM", _utc("1998-12-01"), _utc("1998-12-02"), _utc("1998-12-01 23:00"))
    assert b is None or not len(b) or b.attrs.get("security_id") != "CIK0000005513"


# ------------------------------------------------------------------ false fail-closed (over-rejection)
@pytest.mark.parametrize("sym,day", [
    ("QRVO", "2020-06-01"),   # member 2015-06..2024-12; a 1-row Yahoo stub drops 2,772 DoltHub sessions
    ("FI", "2024-06-03"),     # Fiserv traded as FI 2023-06..2025-11 (member); ZipLime labels it FISV
])
def test_member_with_as_traded_dolthub_is_served(portal, sym, day):
    """The store holds the as-traded DoltHub print of this member; failing closed drops a
    verified-available history."""
    ref = _dolt_close(portal, sym, day)
    b = _bars_or_none(portal, sym, _utc(day), _utc(day) + pd.Timedelta(days=1), _utc(day) + pd.Timedelta(hours=23))
    assert b is not None and len(b) == 1, f"{sym} {day} fails closed (as-traded {ref})"
    assert float(b["close"].iloc[0]) == pytest.approx(ref, rel=0.02)


# ------------------------------------------------------------------ spin-off / composite adjustments
@pytest.mark.parametrize("sym,day,lo,hi", [
    ("DHR", "2016-07-05", -0.10, 0.10),   # Fortive: Yahoo dividend 27.70 AND split 1.319 both applied (+42 %)
    # DD bound relaxed (integrator, coordinator-approved): with the value-based amount the ex-date
    # total return is (DD close / 3 + CTVA close / 3) / DWDP prev close - 1 = (76.10/3 + 24.81/3)
    # / 30.52 - 1 = +10.2 %. DoltHub and Yahoo agree on every print (DD 2019-06-03 open 66.00,
    # high = close 76.10; CTVA close 24.81; DWDP 2019-05-31 close 30.52); CTVA's first-morning fall
    # to $24.89 (-7.7 %) is reported by contemporaneous news (feedstuffs.com, "Corteva separates
    # from DowDuPont", 2019-06-03). DuPont's +15 % open-to-close move is not independently cited.
    ("DD", "2019-06-03", -0.10, 0.12),    # 1:3 reverse split + Corteva: factor 0.4725 -> negative amount (+149 %)
    ("ITT", "2011-11-01", -0.10, 0.10),   # 1:2 reverse split + XYL/XLS: factor 3.0157 as one spin-off (+24 %)
    ("ABT", "2004-05-03", -0.06, 0.06),   # Hospira: dividend 2.846 AND 'stock dividend' 1.0688 both applied (+9.9 %)
    ("MRO", "2011-07-01", -0.10, 0.10),   # MPC spin-off (1 MPC per 2 MRO) absent from the DoltHub segment (-37.5 %)
])
def test_spinoff_ex_date_adjusted_return_sane(portal, sym, day, lo, hi):
    asof = _utc(day) + pd.Timedelta(days=3, hours=23)
    b = _bars_or_none(portal, sym, _utc(day) - pd.Timedelta(days=6), _utc(day) + pd.Timedelta(hours=23), asof,
                      adjusted=True)
    if b is None or len(b) < 2:
        pytest.skip("not served")
    r = float(b["close"].iloc[-1] / b["close"].iloc[-2] - 1)
    assert lo < r < hi, f"{sym} adjusted return on {day} = {r:+.3f} (artifact)"
