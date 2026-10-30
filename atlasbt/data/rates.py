"""Rate conventions.

The engine accrues interest ACT/360 (docs/ARCHITECTURE.md §4), so `Portal.risk_free_rate`
returns a SIMPLE annualized MONEY-MARKET yield on an ACT/360 basis, in decimals.

T-bill rates on FRED (DTB3, DTB4WK, DTB6, DTB1YR) are quoted on a BANK DISCOUNT basis d
(percent, 360-day year): price P = 1 - d * t / 360 for t days to maturity. Converting:
    money-market yield  r_mm = d / (1 - d * t / 360)                   (ACT/360 simple)
    bond-equivalent     r_be = 365 * d / (360 - d * t)                 (t <= 182 days)
    continuous (ACT/365) r_c = -ln(P) * 365 / t
Known answer: d = 5.00 %, t = 91  ->  P = 0.987361, r_mm = 5.0640 %, r_be = 5.1343 %,
r_c = 5.1018 %.
Constant-maturity yields (DGSx) are bond-equivalent; EFFR/DFF/SOFR are already ACT/360.
"""
from __future__ import annotations

import math

TENOR_DAYS = {"1M": 28, "3M": 91, "6M": 182, "1Y": 364}


def discount_to_price(d: float, days: int) -> float:
    return 1.0 - d * days / 360.0


def discount_to_money_market(d: float, days: int = 91) -> float:
    """d and result in decimals."""
    return d / (1.0 - d * days / 360.0)


def discount_to_bond_equivalent(d: float, days: int = 91) -> float:
    return 365.0 * d / (360.0 - d * days)


def discount_to_continuous(d: float, days: int = 91) -> float:
    return -math.log(discount_to_price(d, days)) * 365.0 / days


def bond_equivalent_to_money_market(r_be: float) -> float:
    """Approximate re-basing of a simple 365-day yield to a 360-day basis."""
    return r_be * 360.0 / 365.0


def money_market_to_continuous(r_mm: float, days: int = 91) -> float:
    return math.log(1.0 + r_mm * days / 360.0) * 365.0 / days
