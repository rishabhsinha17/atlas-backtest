"""Market constraints: tick sizes, Reg SHO short-sale price tests, locates,
halts and trading-hours permissions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import rates
from .spread import tick_size_for_price


# --------------------------------------------------------------------------
# Ticks (Reg NMS Rule 612 sub-penny rule)
# --------------------------------------------------------------------------
def valid_tick(price: float, instrument=None) -> float:
    """Minimum quoting increment. Stocks/ETFs: $0.01 at/above $1, $0.0001
    below (Rule 612). Options: the instrument's ``tick_size`` (penny program /
    $0.05 / $0.10 classes are the OPTIONS agent's responsibility)."""
    if instrument is not None and getattr(instrument, "is_option", False):
        return float(instrument.tick_size)
    return tick_size_for_price(price)


def round_to_tick(price: float, tick: float, mode: str = "nearest") -> float:
    """mode: 'nearest' | 'up' | 'down'. Robust to float noise."""
    if not (price == price) or tick <= 0:
        return price
    q = price / tick
    if mode == "up":
        n = math.ceil(q - 1e-9)
    elif mode == "down":
        n = math.floor(q + 1e-9)
    else:
        n = math.floor(q + 0.5)
    return round(n * tick, 10)


def round_limit_for_side(price: float, tick: float, is_buy: bool) -> float:
    """Conservative rounding of a LIMIT price onto the tick grid: buy limits
    round down, sell limits round up (never more aggressive than asked)."""
    return round_to_tick(price, tick, "down" if is_buy else "up")


def round_stop_for_side(price: float, tick: float, is_buy: bool) -> float:
    """Stops round away from the market: buy stops up, sell stops down."""
    return round_to_tick(price, tick, "up" if is_buy else "down")


# --------------------------------------------------------------------------
# Reg SHO short-sale price tests
# --------------------------------------------------------------------------
@dataclass
class ShortSaleRules:
    """Era-aware short-sale price test.

    * Before ``uptick_rule_end`` (2007-07-06): Rule 10a-1 tick test / NASD bid
      test applied to every short sale -> treated as permanently restricted
      (short sales may not hit the bid).
    * ``uptick_rule_end`` .. ``rule201_start``: no price test.
    * From ``rule201_start`` (2011-02-28): Rule 201 circuit breaker. Triggered
      when a covered security's price declines by >= 10% from the prior day's
      closing price; restriction applies for the remainder of that day and the
      whole following day.
    """

    uptick_rule_end: pd.Timestamp = rates.UPTICK_RULE_END
    rule201_start: pd.Timestamp = rates.RULE_201_START
    trigger_decline: float = 0.10
    model_pre_2007_tick_test: bool = True

    def era(self, session: pd.Timestamp) -> str:
        s = pd.Timestamp(session)
        if s < self.uptick_rule_end:
            return "TICK_TEST" if self.model_pre_2007_tick_test else "NONE"
        if s < self.rule201_start:
            return "NONE"
        return "RULE201"

    def triggered(self, prior_close: float, low_so_far: float) -> bool:
        if not (prior_close > 0 and low_so_far == low_so_far):
            return False
        return low_so_far <= prior_close * (1.0 - self.trigger_decline) + 1e-12


class SSRStateMachine:
    """Explicit Rule 201 state machine (useful for event-driven callers and
    tests). States per symbol: ``inactive`` -> ``active_today`` (on trigger)
    -> ``active_next_day`` (next session) -> ``inactive``.
    """

    def __init__(self, rules: Optional[ShortSaleRules] = None):
        self.rules = rules or ShortSaleRules()
        self._trigger_session: dict[str, pd.Timestamp] = {}
        self._trigger_ts: dict[str, pd.Timestamp] = {}

    def on_price(self, symbol: str, session: pd.Timestamp, ts: pd.Timestamp,
                 prior_close: float, price_low: float) -> bool:
        """Feed an observed (low) price; returns True if this triggers SSR."""
        if self.rules.era(session) != "RULE201":
            return False
        if self._trigger_session.get(symbol) == session:
            return False
        if self.rules.triggered(prior_close, price_low):
            self._trigger_session[symbol] = pd.Timestamp(session)
            self._trigger_ts[symbol] = ts
            return True
        return False

    def is_active(self, symbol: str, session: pd.Timestamp, ts: Optional[pd.Timestamp] = None,
                  prev_session: Optional[pd.Timestamp] = None) -> bool:
        era = self.rules.era(session)
        if era == "TICK_TEST":
            return True
        if era == "NONE":
            return False
        trig = self._trigger_session.get(symbol)
        if trig is None:
            return False
        session = pd.Timestamp(session)
        if trig == session:
            return ts is None or ts >= self._trigger_ts[symbol]
        if prev_session is not None and trig == pd.Timestamp(prev_session):
            return True
        return False

    def state(self, symbol: str, session, prev_session=None) -> str:
        trig = self._trigger_session.get(symbol)
        if trig is None:
            return "inactive"
        if trig == pd.Timestamp(session):
            return "active_today"
        if prev_session is not None and trig == pd.Timestamp(prev_session):
            return "active_next_day"
        return "inactive"


# --------------------------------------------------------------------------
# Locates / borrow
# --------------------------------------------------------------------------
@dataclass
class BorrowConfig:
    """Default when the portal has no borrow data: easy-to-borrow at
    ``default_fee`` p.a. (flagged T2). If ``require_borrow_data`` is True,
    shorts without borrow data are rejected instead."""

    default_shortable: bool = True
    default_fee: float = 0.0030          # 0.30% p.a. GC-like (ETF/large caps 0.25-0.5%)
    require_borrow_data: bool = False


# --------------------------------------------------------------------------
# Halts
# --------------------------------------------------------------------------
def bar_is_halted(volume: float, session_code: str, open_: float) -> bool:
    """Zero-volume or NaN-price RTH/DAY bar -> treat as halted (no fills)."""
    if not (open_ == open_):
        return True
    if session_code in ("RTH", "DAY") and (not (volume == volume) or volume <= 0):
        return True
    return False


# --------------------------------------------------------------------------
# Trading hours
# --------------------------------------------------------------------------
def session_allows(session_code: str, outside_rth: bool) -> bool:
    if session_code in ("RTH", "DAY"):
        return True
    return bool(outside_rth) and session_code in ("PRE", "POST")
