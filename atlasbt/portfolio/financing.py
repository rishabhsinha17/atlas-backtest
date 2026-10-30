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

``schedule="ibkr_pro"`` (opt-in) replaces the flat spreads with IBKR Pro's dated, tiered USD
schedule (``execution.rates.IBKR_PRO_INTEREST``), point-in-time on the accrual date:
benchmark BM = Fed Funds Effective (passed in as ``benchmark``); debit balance pays the
blended BM + tier spread; cash above USD 10,000 earns max(BM - 0.5%, 0) (nothing on the first
10,000); short-sale proceeds earn the blended short-collateral tiers (0% to 100K, BM-1.25% to
1M, BM-0.5% to 3M, BM-0.25% above, floored at 0) on top of the separately charged borrow fee;
credit (cash and short proceeds) needs NAV > USD 100,000 before 2020-02-10 and is paid pro rata
NAV/100,000 below that from then on.
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
    #: 'flat' (default: the spreads above on rf) | 'ibkr_pro' (dated tiered IBKR Pro schedule on
    #: the Fed Funds benchmark; short_rebate is then ignored in favour of the short-proceeds tiers)
    schedule: str = "flat"


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
                     borrow_fee: Callable[[str], float], cfg: FinancingConfig,
                     benchmark: Optional[float] = None) -> FinancingResult:
    res = FinancingResult(days=days, rf=rf)
    if not cfg.enabled or days <= 0:
        return res
    if cfg.schedule == "ibkr_pro":
        return _accrue_ibkr(account, ts, days, rf if benchmark is None else benchmark, borrow_fee, cfg, res)
    if cfg.schedule != "flat":
        raise ValueError(f"FinancingConfig.schedule must be 'flat' or 'ibkr_pro', not {cfg.schedule!r}")
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


def credit_eligibility(nav: float, date) -> float:
    """Fraction of the IBKR credit rate paid at this NAV (1 above USD 100K; below it 0 before
    the pro-rata rule, NAV/100K after; see rates.IBKR_PRO_INTEREST)."""
    from ..execution import rates as R
    if nav > R.IBKR_NAV_ELIGIBILITY:
        return 1.0
    if R.IBKR_PRO_INTEREST.at(date, "nav_proration"):
        return max(nav, 0.0) / R.IBKR_NAV_ELIGIBILITY
    return 0.0


def _accrue_ibkr(account: Account, ts, days: int, bm: float, borrow_fee, cfg: FinancingConfig,
                 res: FinancingResult) -> FinancingResult:
    from ..execution import rates as R
    bm = float(bm) if bm == bm else 0.0
    res.rf = bm
    yf = days / cfg.day_count
    row = R.IBKR_PRO_INTEREST.row_at(ts).values
    nav = account.nav
    elig = credit_eligibility(nav, ts)
    margin = account.account_type == AccountType.REG_T_MARGIN
    shorts = [(p, abs(p.market_value)) for p in account.stock_positions() if p.qty < 0]
    short_mv = sum(mv for _, mv in shorts)
    base = account.cash - (short_mv if margin else 0.0)
    if base > 0:
        bal = max(base - row["credit_threshold"], 0.0)
        annual = bal * max(bm - row["credit_spread"], 0.0) * elig
        res.credit = annual * yf
        if res.credit:
            account.post_cash(ts, Reason.INTEREST_CREDIT, res.credit, "",
                              f"{days}d IBKR Pro: BM {bm:.4%} - {row['credit_spread']:.2%} above "
                              f"{row['credit_threshold']:,.0f}, eligibility {elig:.3f}")
    elif base < 0:
        annual = R.tiered_blend(-base, row["debit"], max(bm, 0.0), +1)
        res.debit = -annual * yf
        account.post_cash(ts, Reason.INTEREST_DEBIT, res.debit, "",
                          f"{days}d IBKR Pro blended BM {bm:.4%} + tier spreads on {-base:,.0f}")
    rebate_annual = R.tiered_blend(short_mv, row["short_proceeds"], bm, -1) * elig if margin and short_mv > 0 else 0.0
    for p, mv in shorts:
        fee = borrow_fee(p.symbol)
        if fee and fee > 0:
            amt = -mv * fee * yf
            res.borrow += amt
            res.by_symbol[p.symbol] = res.by_symbol.get(p.symbol, 0.0) + amt
            account.post_cash(ts, Reason.BORROW_FEE, amt, p.symbol, f"{days}d @ {fee:.4%}")
        if rebate_annual > 0:
            amt = rebate_annual * yf * mv / short_mv          # pro rata by short market value
            res.rebate += amt
            res.by_symbol[p.symbol] = res.by_symbol.get(p.symbol, 0.0) + amt
            account.post_cash(ts, Reason.SHORT_REBATE, amt, p.symbol,
                              f"{days}d IBKR Pro short-proceeds tiers, BM {bm:.4%}")
    return res
