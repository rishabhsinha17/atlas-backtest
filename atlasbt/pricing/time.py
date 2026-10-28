"""Time-to-expiry and rate conventions.

* Year fraction: ACT/365F on exact wall-clock time to the settlement
  instant (``settlement_instant``): PHYSICAL and CASH_PM settle at the official
  close of the expiry session (16:00 ET, 13:00 ET on early closes);
  CASH_AM settles at the special opening quotation (SOQ) on the settlement
  session, modeled at the 09:30 ET open of that session.  So a 0DTE PM option
  at 10:00 ET has T = 6h/(365*24h).
* Rates: pricing functions take continuously-compounded r.  The DataPortal's
  ``risk_free_rate`` is an annualized simple rate; FRED DTB3 is a T-bill
  *discount* yield (ACT/360).  Convert with the helpers below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from atlasbt.core import clock
from atlasbt.core.types import SettlementType

MIN_T = 1.0 / (365.0 * 24 * 60)  # one minute


def settlement_session(expiry: pd.Timestamp, settlement: SettlementType | str | None) -> pd.Timestamp:
    """Session whose open (AM) / close (PM, physical) fixes the settlement value.
    ``expiry`` is the LAST TRADING DAY (core.types.Instrument convention)."""
    exp = pd.Timestamp(expiry).normalize()
    if exp.tzinfo is not None:
        exp = exp.tz_localize(None)
    st = SettlementType(settlement) if settlement is not None else SettlementType.PHYSICAL
    if st == SettlementType.CASH_AM:
        return clock.nyse().next_session(exp)
    return exp


def settlement_instant(expiry: pd.Timestamp, settlement: SettlementType | str | None) -> pd.Timestamp:
    """UTC instant at which the option's value is fixed."""
    st = SettlementType(settlement) if settlement is not None else SettlementType.PHYSICAL
    sess = settlement_session(expiry, st)
    cal = clock.nyse()
    if cal.is_session(sess):
        o, c = cal.open_close(sess)
    else:  # outside calendar coverage: regular hours
        o, c = clock.et(sess, "09:30"), clock.et(sess, "16:00")
    return o if st == SettlementType.CASH_AM else c


def year_fraction(now: pd.Timestamp, expiry: pd.Timestamp, settlement=None, min_t: float = 0.0) -> float:
    """ACT/365F year fraction from ``now`` (tz-aware UTC) to settlement."""
    now = clock.to_utc(now)
    secs = (settlement_instant(expiry, settlement) - now).total_seconds()
    return max(secs / (365.0 * 86400.0), min_t)


def year_fractions(now: pd.Timestamp, expiries, settlements=None) -> np.ndarray:
    """Vectorized ``year_fraction`` over arrays of expiries/settlements (cached per unique pair)."""
    exps = pd.to_datetime(pd.Series(expiries)).dt.normalize()
    sts = pd.Series(settlements if settlements is not None else ["PHYSICAL"] * len(exps)).fillna("PHYSICAL").astype(str)
    out = np.empty(len(exps))
    cache: dict = {}
    for i, (e, s) in enumerate(zip(exps, sts)):
        key = (e, s)
        if key not in cache:
            cache[key] = year_fraction(now, e, s)
        out[i] = cache[key]
    return out


# ---- rates -----------------------------------------------------------------
def simple_to_cc(r_simple, T):
    """Annualized simple rate over horizon T (years) -> continuous rate."""
    r_simple, T = np.asarray(r_simple, float), np.asarray(T, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(T > 1e-9, np.log1p(r_simple * T) / np.where(T > 1e-9, T, 1.0), r_simple)
    return out[()] if out.ndim == 0 else out


def cc_to_simple(r_cc, T):
    r_cc, T = np.asarray(r_cc, float), np.asarray(T, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(T > 1e-9, np.expm1(r_cc * T) / np.where(T > 1e-9, T, 1.0), r_cc)
    return out[()] if out.ndim == 0 else out


def tbill_discount_to_cc(discount_yield, days: float):
    """T-bill bank-discount yield (decimal, ACT/360, e.g. FRED DTB3/100) for a
    bill with ``days`` to maturity -> continuously-compounded ACT/365 rate."""
    d = np.asarray(discount_yield, float)
    price = 1.0 - d * days / 360.0
    return -np.log(price) / (days / 365.0)


def tbill_discount_to_bey(discount_yield, days: float):
    """Bond-equivalent (investment) yield for bills <= 182 days."""
    d = np.asarray(discount_yield, float)
    return 365.0 * d / (360.0 - d * days)
