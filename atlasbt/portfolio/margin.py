"""Reg-T margin for stocks, a pluggable option-margin model, buying-power
checks and margin-call liquidation planning.

Stock rules (Federal Reserve Reg T 12 CFR 220.12; FINRA Rule 4210(c)):
* Initial: 50% of long market value; short sales 150% of short value
  (i.e. 50% equity on top of the proceeds).
* Maintenance: long 25% of market value; short: price >= $5 -> max($5/share,
  30% of MV); price < $5 -> max($2.50/share, 100% of MV).
* CASH accounts: no borrowing (cash must cover purchases), no short stock, no
  naked options (covered calls and cash-secured puts allowed).

Options: the OPTIONS agent owns the authoritative schedule
(``atlasbt.options.margin``). Anything implementing :class:`OptionMarginModel`
can be plugged into :class:`RegTMarginModel`. The default
:class:`ConservativeOptionMargin` implements the Cboe/Reg-T strategy-based
basics: long options paid in full (no loan value), covered calls / puts,
vertical spreads at max loss, naked: max(premium + 20% (15% broad index) of
underlying - OTM, premium + 10% of underlying (calls) / strike (puts)), cash
account short puts must be fully cash secured (strike x multiplier).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Protocol, runtime_checkable

import pandas as pd

from ..core.types import Instrument, Right
from .account import Account, AccountType, Position

BROAD_INDEX_UNDERLYINGS = {"SPX", "XSP", "NDX", "RUT", "DJX", "OEX", "XEO", "VIX", "MRUT", "NANOS"}


@runtime_checkable
class OptionMarginModel(Protocol):
    """Plug-in interface for option margin (OPTIONS agent implements).

    ``requirement`` returns the TOTAL margin requirement (USD, >= 0) for all
    option positions in the account (including any stock they are paired
    with as cover), given an underlying price function. It must never return
    0 for a naked short option."""

    def requirement(self, account: Account, underlying_price: Callable[[str], float]) -> float: ...

    # Optional attribute ``includes_long_premium: bool``: True if the requirement
    # already charges long options in full (no loan value); otherwise the Reg-T
    # model adds 100% (75% if > 9 months) of long-option market value itself.


@dataclass
class ConservativeOptionMargin:
    includes_long_premium = False
    naked_pct: float = 0.20
    naked_pct_index: float = 0.15
    min_pct: float = 0.10
    min_pct_index: float = 0.10

    def requirement(self, account: Account, underlying_price: Callable[[str], float]) -> float:
        opts = account.option_positions()
        if not opts:
            return 0.0
        req = 0.0
        # group by (underlying, right, expiry)
        shorts: list[list] = []   # [Position, remaining_contracts]
        longs: list[list] = []
        for p in opts:
            (shorts if p.qty < 0 else longs).append([p, abs(p.qty)])
        # stock cover
        stock_avail = {p.symbol: p.qty for p in account.stock_positions()}
        for item in shorts:
            p, n = item
            inst = p.instrument
            und = (inst.underlying or "").upper()
            per = inst.deliverable_shares
            if inst.right == Right.CALL and stock_avail.get(und, 0.0) > 0:
                cov = min(n, math.floor(stock_avail[und] / per + 1e-9))
                stock_avail[und] -= cov * per
                item[1] -= cov
            elif inst.right == Right.PUT and stock_avail.get(und, 0.0) < 0:
                cov = min(n, math.floor(-stock_avail[und] / per + 1e-9))
                stock_avail[und] += cov * per
                item[1] -= cov
        # vertical / calendar spreads: pair shorts with longs of same underlying & right,
        # long expiry >= short expiry; requirement = max loss for credit spreads
        for item in shorts:
            p, n = item
            if n <= 0:
                continue
            si = p.instrument
            for litem in sorted(longs, key=lambda li: abs((li[0].instrument.strike or 0) - (si.strike or 0))):
                lp, ln = litem
                li = lp.instrument
                if ln <= 0 or li.underlying != si.underlying or li.right != si.right:
                    continue
                if li.expiry is None or si.expiry is None or li.expiry < si.expiry:
                    continue
                k = min(n, ln)
                width = 0.0
                if si.right == Right.CALL:
                    width = max((li.strike or 0) - (si.strike or 0), 0.0)
                else:
                    width = max((si.strike or 0) - (li.strike or 0), 0.0)
                req += k * width * si.multiplier * (si.deliverable_shares / 100.0)
                n -= k
                litem[1] -= k
                if n <= 0:
                    break
            item[1] = n
        # remaining naked shorts
        for p, n in shorts:
            if n <= 0:
                continue
            inst = p.instrument
            und = (inst.underlying or "").upper().lstrip("^$")
            s = underlying_price(inst.underlying or "")
            prem = p.mark if p.mark == p.mark else p.avg_price
            k = inst.strike or 0.0
            mult = inst.multiplier
            if not (s == s and s > 0):
                # unknown underlying price: be conservative (strike-based)
                s = k
            if account.account_type == AccountType.CASH:
                if inst.right == Right.PUT:
                    req += n * k * inst.deliverable_shares   # cash secured: strike x deliverable
                else:
                    req += n * 10 * s * mult   # naked call not allowed; huge requirement
                continue
            idx = und in BROAD_INDEX_UNDERLYINGS
            pct = self.naked_pct_index if idx else self.naked_pct
            mn = self.min_pct_index if idx else self.min_pct
            if inst.right == Right.CALL:
                otm = max(k - s, 0.0)
                per = max(prem + pct * s - otm, prem + mn * s)
            else:
                otm = max(s - k, 0.0)
                per = max(prem + pct * s - otm, prem + mn * k)
            req += n * per * mult
        return req


@dataclass
class MarginSnapshot:
    equity: float
    initial: float
    maintenance: float
    long_mv: float
    short_mv: float
    option_req: float

    @property
    def excess(self) -> float:
        return self.equity - self.maintenance

    @property
    def is_call(self) -> bool:
        return self.equity < self.maintenance - 1e-6


@dataclass
class RegTMarginModel:
    initial_long: float = 0.50
    initial_short: float = 0.50
    maint_long: float = 0.25
    maint_short: float = 0.30
    short_low_price: float = 5.0
    option_model: Optional[OptionMarginModel] = None
    liquidation_buffer: float = 0.10

    def __post_init__(self):
        if self.option_model is None:
            self.option_model = ConservativeOptionMargin()

    # -- per-position requirements
    def _short_maint(self, price: float, qty_abs: float) -> float:
        if price >= self.short_low_price:
            return max(5.0 * qty_abs, self.maint_short * price * qty_abs)
        return max(2.5 * qty_abs, 1.0 * price * qty_abs)

    long_option_loan_value_months: float = 9.0
    long_option_long_dated_req: float = 0.75

    def _long_option_req(self, account: Account, as_of=None) -> float:
        """Long options have no loan value (Reg T / FINRA 4210(f)(2)(C)):
        100% of market value; listed options with > 9 months to expiry may be
        financed at 25% loan value -> 75% requirement."""
        req = 0.0
        for p in account.option_positions():
            if p.qty <= 0:
                continue
            mv = p.market_value
            frac = 1.0
            ref = as_of if as_of is not None else p.mark_ts
            if p.instrument.expiry is not None and ref is not None:
                r = pd.Timestamp(ref)
                r = r.tz_convert("America/New_York").tz_localize(None) if r.tzinfo is not None else r
                months = (pd.Timestamp(p.instrument.expiry) - r.normalize()).days / 30.4375
                if months > self.long_option_loan_value_months:
                    frac = self.long_option_long_dated_req
            req += frac * mv
        return req

    def snapshot(self, account: Account, underlying_price: Callable[[str], float], as_of=None) -> MarginSnapshot:
        long_mv = 0.0
        short_mv = 0.0
        init = 0.0
        maint = 0.0
        for p in account.stock_positions():
            px = p.mark if p.mark == p.mark else p.avg_price
            mv = p.qty * px * p.multiplier
            if p.qty > 0:
                long_mv += mv
                init += self.initial_long * mv
                maint += self.maint_long * mv
            else:
                short_mv += mv
                init += self.initial_short * -mv
                maint += self._short_maint(px, -p.qty * p.multiplier)
        opt = self.option_model.requirement(account, underlying_price) if self.option_model else 0.0
        if account.account_type == AccountType.CASH:
            # cash account: everything is paid in full; requirement = option collateral only
            init = opt
            maint = opt
        else:
            includes = getattr(self.option_model, "includes_long_premium", None)
            if includes is None:   # OPTIONS agent's CboeStrategyMargin charges longs in full itself
                includes = type(self.option_model).__name__ == "CboeStrategyMargin"
            lo = 0.0 if includes else self._long_option_req(account, as_of)
            init += opt + lo
            maint += opt + lo
            opt += lo
        return MarginSnapshot(account.nav, init, maint, long_mv, short_mv, opt)

    # -- pre-trade check
    def check_order(self, account: Account, instrument: Instrument, delta_qty: float, price: float,
                    underlying_price: Callable[[str], float],
                    pending: Iterable[tuple[Instrument, float, float]] = (), est_cost: float = 0.0,
                    as_of=None) -> tuple[bool, str]:
        """Would the account still satisfy *initial* margin (margin account) /
        have enough unreserved cash (cash account) after this order and all
        other pending EXPOSURE-INCREASING orders fill? ``price`` and pending
        prices must already be conservative (the caller prices orders at the
        worse of order and reference price); ``est_cost`` = estimated explicit
        fees. Pending orders that reduce exposure get NO credit."""
        if not (price == price and price > 0):
            return True, ""
        mult = instrument.multiplier
        # keep only pending orders that increase |position| (cumulatively per symbol)
        proj: dict[str, float] = {}
        inc: list[tuple[Instrument, float, float]] = []
        for inst, q, px in pending:
            cur_p = proj.get(inst.symbol, account.position(inst.symbol))
            if abs(cur_p + q) > abs(cur_p) + 1e-12 and px == px and px > 0:
                inc.append((inst, q, px))
                proj[inst.symbol] = cur_p + q
        if account.account_type == AccountType.CASH:
            cur = account.position(instrument.symbol)
            new = cur + delta_qty
            if not instrument.is_option and new < -1e-9:
                return False, "cash account: short selling not permitted"
            cash_after = account.cash - delta_qty * price * mult - max(est_cost, 0.0)
            reserve = 0.0
            for p in account.option_positions():
                if p.qty < 0 and p.instrument.right == Right.PUT and p.symbol != instrument.symbol:
                    reserve += -p.qty * (p.instrument.strike or 0.0) * p.instrument.deliverable_shares
            for inst, q, px in inc:
                cash_after -= q * px * inst.multiplier
                if inst.is_option and q < 0 and inst.right == Right.PUT:
                    reserve += -q * (inst.strike or 0.0) * inst.deliverable_shares
            if instrument.is_option and new < 0 and instrument.right == Right.PUT:
                reserve += -new * (instrument.strike or 0.0) * instrument.deliverable_shares
            if instrument.is_option and new < 0:
                # short option in cash account: only covered call / cash-secured put
                if instrument.right == Right.PUT:
                    if cash_after < reserve - 1e-6:
                        return False, "cash account: put not fully cash-secured"
                else:
                    shares = account.position(instrument.underlying or "")
                    if shares < -new * instrument.deliverable_shares - 1e-9:
                        return False, "cash account: naked call not permitted"
            if delta_qty > 0 and cash_after - reserve < -1e-6:
                return False, (f"cash account: insufficient unreserved cash ({cash_after:.2f} after order, "
                               f"{reserve:.2f} reserved for short puts)")
            return True, ""
        # margin account: simulate post-trade initial requirement
        tmp = _ShadowAccount(account)
        for inst, q, px in inc:
            tmp.add(inst, q, px)
        tmp.add(instrument, delta_qty, price)
        tmp.account.cash -= max(est_cost, 0.0)
        snap = self.snapshot(tmp.account, underlying_price, as_of)
        if snap.initial > snap.equity + 1e-6 and _increases_exposure(account, instrument, delta_qty):
            return False, (f"Reg-T initial margin: requirement {snap.initial:.2f} > equity {snap.equity:.2f} "
                           f"(pending exposure-reducing orders, e.g. sells in a rotation, get no buying-power "
                           f"credit until they fill)")
        return True, ""

    # -- margin call liquidation plan
    def liquidation_plan(self, account: Account, snap: MarginSnapshot) -> list[tuple[Position, float]]:
        """Proportional reduction of every requirement-carrying position so
        that maintenance <= equity / (1 + buffer). Returns (position, delta_qty)."""
        if not snap.is_call:
            return []
        eq = snap.equity
        target = eq / (1.0 + self.liquidation_buffer)
        if eq <= 0 or snap.maintenance <= 0:
            frac = 1.0
        else:
            frac = min(max(1.0 - target / snap.maintenance, 0.0), 1.0)
        plan = []
        for p in sorted(account.positions.values(), key=lambda x: x.symbol):
            carries = (not p.instrument.is_option) or p.qty < 0
            if not carries:
                continue
            n = min(math.ceil(abs(p.qty) * frac - 1e-9), abs(p.qty))
            if n > 0:
                plan.append((p, -math.copysign(n, p.qty)))
        return plan


def _increases_exposure(account: Account, inst: Instrument, dq: float) -> bool:
    cur = account.position(inst.symbol)
    return abs(cur + dq) > abs(cur) + 1e-12


class _ShadowAccount:
    """Copy-light hypothetical account for pre-trade checks."""

    def __init__(self, base: Account):
        acct = Account(0.0, base.account_type)
        acct.cash = base.cash
        acct.accruals = list(base.accruals)
        acct.positions = {s: Position(instrument=p.instrument, qty=p.qty, avg_price=p.avg_price,
                                      mark=p.mark, mark_ts=p.mark_ts)
                          for s, p in base.positions.items()}
        self.account = acct

    def add(self, inst: Instrument, q: float, px: float) -> None:
        a = self.account
        a.cash -= q * px * inst.multiplier
        p = a.positions.get(inst.symbol)
        if p is None:
            a.positions[inst.symbol] = Position(instrument=inst, qty=q, avg_price=px, mark=px)
        else:
            p.qty += q
            if p.mark != p.mark:
                p.mark = px
            if abs(p.qty) < 1e-12:
                del a.positions[inst.symbol]
