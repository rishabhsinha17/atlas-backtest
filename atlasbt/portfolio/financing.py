"""Daily financing accrual (posted to cash at each session end).

* Cash interest: interest-bearing balance = cash - |short stock market value|
  (short-sale proceeds are collateral and earn the short rebate, not credit
  interest; retail rebate default 0). Positive balance earns
  max(rf - credit_spread, 0); negative balance (margin loan) pays
  rf + debit_spread. Day count actual/360 (USD money-market convention, IBKR
  uses 360 for USD), applied to the CALENDAR days from this session to the
  next one (Friday -> Monday = 3 days).
* Short borrow fee: |short market value| * fee_rate * days / 360, fee from
  ``portal.borrow`` (or the default, flagged T2).
* Short rebate: |short market value| * rebate_rate * days / 360 (default 0).

Defaults: credit spread 0.50%, debit spread 1.50% (IBKR Pro benchmark +/-
spreads for small balances; interactivebrokers.com interest-rates pages).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from .account import Account, AccountType
from .ledger import Reason


@dataclass
class FinancingConfig:
    credit_spread: float = 0.005
    debit_spread: float = 0.015
    short_rebate: float = 0.0
    day_count: float = 360.0
    floor_credit_at_zero: bool = True
    enabled: bool = True


@dataclass
class FinancingResult:
    credit: float = 0.0
    debit: float = 0.0
    borrow: float = 0.0
    rebate: float = 0.0
    days: int = 0
    rf: float = 0.0
    by_symbol: dict = field(default_factory=dict)   # symbol -> signed borrow/rebate cash


def accrue_financing(account: Account, ts: pd.Timestamp, days: int, rf: float,
                     borrow_fee: Callable[[str], float], cfg: FinancingConfig) -> FinancingResult:
    res = FinancingResult(days=days, rf=rf)
    if not cfg.enabled or days <= 0:
        return res
    rf = float(rf) if rf == rf else 0.0
    yf = days / cfg.day_count
    short_stock_mv = -account.short_market_value(stock_only=True)   # positive
    base = account.cash - (short_stock_mv if account.account_type == AccountType.REG_T_MARGIN else 0.0)
    if base > 0:
        rate = rf - cfg.credit_spread
        if cfg.floor_credit_at_zero:
            rate = max(rate, 0.0)
        res.credit = base * rate * yf
        account.post_cash(ts, Reason.INTEREST_CREDIT, res.credit, "", f"{days}d @ {rate:.4%}")
    elif base < 0:
        rate = max(rf, 0.0) + cfg.debit_spread
        res.debit = base * rate * yf     # negative
        account.post_cash(ts, Reason.INTEREST_DEBIT, res.debit, "", f"{days}d @ {rate:.4%}")
    for p in account.stock_positions():
        if p.qty >= 0:
            continue
        mv = abs(p.market_value)
        fee = borrow_fee(p.symbol)
        if fee and fee > 0:
            amt = -mv * fee * yf
            res.borrow += amt
            res.by_symbol[p.symbol] = res.by_symbol.get(p.symbol, 0.0) + amt
            account.post_cash(ts, Reason.BORROW_FEE, amt, p.symbol, f"{days}d @ {fee:.4%}")
        if cfg.short_rebate:
            amt = mv * cfg.short_rebate * yf
            res.rebate += amt
            res.by_symbol[p.symbol] = res.by_symbol.get(p.symbol, 0.0) + amt
            account.post_cash(ts, Reason.SHORT_REBATE, amt, p.symbol)
    return res
