"""Positions, cash and NAV with average-cost realized PnL.

NAV = cash + sum(qty * multiplier * mark) + accrued dividend receivables
      - accrued dividend payables.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

import pandas as pd

from ..core.types import AssetClass, Fill, Instrument
from .ledger import Ledger, Reason

QTY_EPS = 1e-9


class AccountType(str, enum.Enum):
    CASH = "CASH"
    REG_T_MARGIN = "REG_T_MARGIN"


@dataclass
class Position:
    instrument: Instrument
    qty: float = 0.0
    avg_price: float = 0.0          # average entry price per unit (share / option premium)
    mark: float = float("nan")
    mark_ts: Optional[pd.Timestamp] = None
    realized: float = 0.0           # realized PnL (gross of explicit fees) since open
    opened_at: Optional[pd.Timestamp] = None

    @property
    def symbol(self) -> str:
        return self.instrument.symbol

    @property
    def multiplier(self) -> float:
        return self.instrument.multiplier

    @property
    def market_value(self) -> float:
        m = self.mark if self.mark == self.mark else self.avg_price
        return self.qty * self.multiplier * m

    @property
    def unrealized(self) -> float:
        m = self.mark if self.mark == self.mark else self.avg_price
        return self.qty * self.multiplier * (m - self.avg_price)


@dataclass
class Accrual:
    """Dividend entitlement (receivable > 0) or obligation (payable < 0)."""

    symbol: str
    amount: float                   # net cash to be posted at pay_date
    gross: float                    # gross dividend (before withholding)
    withholding: float
    pay_date: pd.Timestamp          # session date (naive)
    ex_date: pd.Timestamp
    kind: str                       # 'DIVIDEND' | 'DIVIDEND_IN_LIEU'


class Account:
    def __init__(self, initial_cash: float, account_type: AccountType = AccountType.REG_T_MARGIN,
                 start_ts: Optional[pd.Timestamp] = None):
        self.initial_cash = float(initial_cash)
        self.account_type = AccountType(account_type)
        self.ledger = Ledger()
        self.cash = 0.0
        self.positions: dict[str, Position] = {}
        self.accruals: list[Accrual] = []
        self.realized_total = 0.0
        self.fees_total = 0.0
        if initial_cash:
            self._cash(start_ts, Reason.DEPOSIT, initial_cash, note="initial deposit")

    # ------------------------------------------------------------------ cash
    def _cash(self, ts, reason: Reason, amount: float, symbol: str = "", note: str = "",
              ref: Optional[int] = None) -> None:
        if amount == 0.0:
            return
        self.cash += amount
        self.ledger.post(ts, reason, amount, symbol, note, ref)

    def post_cash(self, ts, reason: Reason, amount: float, symbol: str = "", note: str = "",
                  ref: Optional[int] = None) -> None:
        self._cash(ts, reason, amount, symbol, note, ref)

    # ------------------------------------------------------------- positions
    def position(self, symbol: str) -> float:
        p = self.positions.get(symbol)
        return p.qty if p is not None else 0.0

    def get(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def qty_map(self) -> dict[str, float]:
        return {s: p.qty for s, p in sorted(self.positions.items())}

    def apply_fill(self, fill: Fill, cash_reason: Reason = Reason.TRADE) -> float:
        """Book a fill: principal + explicit fees to cash; position + realized
        PnL by average cost. Returns realized PnL (gross of fees)."""
        inst = fill.instrument
        sym = inst.symbol
        q = float(fill.quantity)
        px = float(fill.price)
        mult = inst.multiplier
        self._cash(fill.ts, cash_reason, -q * px * mult, sym, f"{fill.liquidity} {q:+g}@{px:.6g}", fill.order_id)
        c = fill.costs
        self._cash(fill.ts, Reason.COMMISSION, -c.commission, sym, "", fill.order_id)
        self._cash(fill.ts, Reason.EXCHANGE_FEES, -c.exchange_fees, sym, "", fill.order_id)
        self._cash(fill.ts, Reason.REGULATORY_FEES, -c.regulatory_fees, sym, "", fill.order_id)
        self.fees_total += c.explicit
        return self.change_position(inst, q, px, fill.ts)

    def change_position(self, inst: Instrument, q: float, px: float, ts) -> float:
        """Update position quantity at price px without touching cash."""
        sym = inst.symbol
        pos = self.positions.get(sym)
        if pos is None:
            pos = Position(instrument=inst, opened_at=ts)
            self.positions[sym] = pos
        realized = 0.0
        old = pos.qty
        mult = inst.multiplier
        if old == 0.0 or (old > 0) == (q > 0):
            new = old + q
            pos.avg_price = (old * pos.avg_price + q * px) / new if abs(new) > QTY_EPS else 0.0
            pos.qty = new
            if old == 0.0:
                pos.opened_at = ts
                pos.realized = 0.0
        else:
            closing = min(abs(q), abs(old))
            sign_old = 1.0 if old > 0 else -1.0
            realized = closing * (px - pos.avg_price) * sign_old * mult
            new = old + q
            if abs(new) <= QTY_EPS:
                new = 0.0
            if new != 0.0 and (new > 0) != (old > 0):
                # flipped: remainder opens at px
                pos.avg_price = px
                pos.opened_at = ts
            pos.qty = new
            pos.realized += realized
            self.realized_total += realized
        if pos.mark != pos.mark:
            pos.mark = px
            pos.mark_ts = ts
        if pos.qty == 0.0:
            del self.positions[sym]
        return realized

    def mark(self, symbol: str, price: float, ts=None) -> None:
        p = self.positions.get(symbol)
        if p is not None and price == price:
            p.mark = float(price)
            p.mark_ts = ts

    # ----------------------------------------------------------------- views
    @property
    def accrued(self) -> float:
        return sum(a.amount for a in self.accruals)

    @property
    def nav(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values()) + self.accrued

    def long_market_value(self, stock_only: bool = False) -> float:
        return sum(p.market_value for p in self.positions.values()
                   if p.qty > 0 and (not stock_only or not p.instrument.is_option))

    def short_market_value(self, stock_only: bool = False) -> float:
        """Negative number (market value of short positions)."""
        return sum(p.market_value for p in self.positions.values()
                   if p.qty < 0 and (not stock_only or not p.instrument.is_option))

    def gross_exposure(self, stock_only: bool = True) -> float:
        return sum(abs(p.market_value) for p in self.positions.values()
                   if not stock_only or not p.instrument.is_option)

    def option_positions(self) -> list[Position]:
        return [p for _, p in sorted(self.positions.items()) if p.instrument.is_option]

    def stock_positions(self) -> list[Position]:
        return [p for _, p in sorted(self.positions.items()) if not p.instrument.is_option]

    # ---------------------------------------------------------- invariants
    def check_invariants(self, tol: float = 1e-6) -> list[str]:
        errs = []
        scale = max(1.0, abs(self.initial_cash))
        led = self.ledger.total
        if abs(self.cash - led) > tol * scale:   # deposit is itself a ledger entry
            errs.append(f"ledger mismatch: cash={self.cash} ledger_total={led}")
        if abs(self.ledger.recomputed_total() - led) > tol * scale:
            errs.append("ledger running total drift")
        nav_direct = self.cash + sum(p.qty * p.multiplier * (p.mark if p.mark == p.mark else p.avg_price)
                                     for p in self.positions.values()) + self.accrued
        if abs(nav_direct - self.nav) > tol * scale:
            errs.append(f"NAV identity broken: {nav_direct} vs {self.nav}")
        for s, p in self.positions.items():
            if not math.isfinite(p.qty) or p.qty == 0:
                errs.append(f"bad qty for {s}: {p.qty}")
            if p.mark != p.mark:
                errs.append(f"unmarked position {s}")
        if not math.isfinite(self.cash):
            errs.append("cash not finite")
        return errs

    def snapshot(self, ts) -> list[dict]:
        return [
            {"ts": ts, "symbol": s, "quantity": p.qty, "mark": p.mark, "market_value": p.market_value,
             "avg_price": p.avg_price, "unrealized": p.unrealized, "multiplier": p.multiplier,
             "asset_class": p.instrument.asset_class.value}
            for s, p in sorted(self.positions.items())
        ]
