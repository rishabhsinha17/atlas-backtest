"""Application of corporate actions to the account (raw-price accounting).

Conventions (docs/ARCHITECTURE.md §1/§6):
* Splits (ratio = new shares per old share): qty *= ratio, avg_price /=
  ratio, mark /= ratio. Fractional shares are cashed out ("cash in lieu") at
  the post-split equivalent of the prior close unless fractional shares are
  allowed.
* Cash dividends: entitlement on the ex-date for the position held at the
  prior close (i.e. at session start, before the ex-date's trades); cash is
  posted on pay_date. Shorts owe a payment in lieu. Optional withholding on
  long receipts.
* Delisting: position closed at the delisting value (``amount``); if the value
  is missing (NaN) the last mark is used and the event is flagged T2.
* Symbol change: position (and its marks) moved to ``new_symbol``.
* Spin-off: ``ratio`` new shares per parent share; ``amount`` = fraction of the
  parent's cost basis allocated to the spin-off (0..1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import pandas as pd

from ..core.types import ActionType, CorporateAction, Fidelity, Instrument
from .account import Account, Accrual, Position
from .ledger import Reason


def as_session_date(x) -> pd.Timestamp:
    """Normalise an action date to a naive session date.

    Accepts naive dates, and tz-aware timestamps. A tz-aware timestamp at
    exactly 00:00 UTC is read as the *date label* (core/types.py convention:
    "UTC midnight of ET date"); any other tz-aware time is converted to ET."""
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        return t.normalize()
    u = t.tz_convert("UTC")
    if u.hour == 0 and u.minute == 0 and u.second == 0 and u.microsecond == 0:
        return u.tz_localize(None)
    return t.tz_convert("America/New_York").tz_localize(None).normalize()


@dataclass
class CorporateActionConfig:
    withholding_rate: float = 0.0          # e.g. 0.30 for non-resident alien
    fractional_shares: bool = False
    short_pays_in_lieu: bool = True
    #: dividends without a pay_date (e.g. Yahoo): pay at ex_date + lag calendar days,
    #: rolled to the next session (flagged T2). NAV is unaffected (receivable).
    pay_date_lag_days: int = 7


@dataclass
class ActionResult:
    action: CorporateAction
    applied: bool
    note: str = ""
    fidelity: Fidelity = Fidelity.T1_BAR
    cancel_orders_for: tuple[str, ...] = ()
    closed_qty: float = 0.0
    close_price: float = float("nan")
    renamed_to: Optional[str] = None
    split_ratio: float = 1.0                 # applied split ratio (1.0 = none)
    cil_qty: float = 0.0                     # fractional shares (post-split units) cashed out, signed
    cil_price: float = float("nan")
    accrual: Optional[Accrual] = None        # dividend accrual created by THIS action
    spinoff_symbol: Optional[str] = None
    spinoff_qty: float = 0.0
    spinoff_basis: float = 0.0               # USD cost basis moved from parent to child


class CorporateActionProcessor:
    def __init__(self, cfg: Optional[CorporateActionConfig] = None,
                 roll_to_session: Optional[Callable[[pd.Timestamp], pd.Timestamp]] = None):
        self.cfg = cfg or CorporateActionConfig()
        #: maps a calendar date to the same-or-next trading session (engine supplies the NYSE calendar)
        self.roll_to_session = roll_to_session

    # ---------------------------------------------------------------- entry
    def apply(self, action: CorporateAction, account: Account, ts: pd.Timestamp,
              session: pd.Timestamp,
              instrument_lookup: Optional[Callable[[str], Instrument]] = None,
              price_lookup: Optional[Callable[[str], float]] = None) -> ActionResult:
        t = action.type
        if t == ActionType.SPLIT:
            return self._split(action, account, ts)
        if t in (ActionType.CASH_DIVIDEND, ActionType.SPECIAL_DIVIDEND):
            return self._dividend(action, account, ts, session)
        if t == ActionType.DELIST:
            return self._delist(action, account, ts)
        if t == ActionType.SYMBOL_CHANGE:
            return self._rename(action, account, instrument_lookup)
        if t == ActionType.SPINOFF:
            return self._spinoff(action, account, ts, instrument_lookup, price_lookup)
        return ActionResult(action, False, f"unsupported action {t}")

    def pay_due(self, account: Account, session: pd.Timestamp, ts: pd.Timestamp) -> list[Accrual]:
        """Post dividend accruals whose pay_date <= session."""
        due = [a for a in account.accruals if a.pay_date <= session]
        if not due:
            return []
        account.accruals = [a for a in account.accruals if a.pay_date > session]
        for a in due:
            if a.kind == "DIVIDEND":
                account.post_cash(ts, Reason.DIVIDEND, a.gross, a.symbol, f"ex {a.ex_date.date()}")
                account.post_cash(ts, Reason.WITHHOLDING, -a.withholding, a.symbol)
            else:
                account.post_cash(ts, Reason.DIVIDEND_IN_LIEU, a.amount, a.symbol, f"ex {a.ex_date.date()}")
        return due

    # ------------------------------------------------------------ handlers
    def _split(self, a: CorporateAction, acct: Account, ts) -> ActionResult:
        pos = acct.get(a.symbol)
        r = float(a.ratio)
        if not (r > 0):
            return ActionResult(a, False, "invalid split ratio")
        if pos is None:
            return ActionResult(a, True, "no position", cancel_orders_for=(a.symbol,), split_ratio=r)
        old_qty = pos.qty
        new_qty = old_qty * r
        pos.avg_price /= r
        if pos.mark == pos.mark:
            pos.mark /= r
        # the adjusted mark supersedes pre-ex-date bars until a post-split bar prints
        pos.mark_ts = ts
        cil_qty, cil_px = 0.0, float("nan")
        note = f"split {r:g}: {old_qty:g} -> {new_qty:g}"
        if not self.cfg.fractional_shares and not pos.instrument.is_option:
            whole = math.trunc(round(new_qty, 9))
            frac = new_qty - whole
            if abs(frac) > 1e-9:
                cil_px = pos.mark if pos.mark == pos.mark else pos.avg_price
                acct.post_cash(ts, Reason.CASH_IN_LIEU, frac * cil_px, a.symbol, f"{frac:.6f} sh @ {cil_px:.6g}")
                cil_qty = frac
                # realised PnL on the fractional piece
                pnl = frac * (cil_px - pos.avg_price) * pos.multiplier
                pos.realized += pnl
                acct.realized_total += pnl
                note += f"; cash in lieu {frac:.6f} sh"
            new_qty = float(whole)
        pos.qty = new_qty
        if pos.qty == 0:
            del acct.positions[a.symbol]
        return ActionResult(a, True, note, cancel_orders_for=(a.symbol,), split_ratio=r,
                            cil_qty=cil_qty, cil_price=cil_px)

    def _dividend(self, a: CorporateAction, acct: Account, ts, session) -> ActionResult:
        pos = acct.get(a.symbol)
        if pos is None or pos.instrument.is_option:
            return ActionResult(a, True, "no position")
        amt = float(a.amount)
        fid = a.fidelity
        assumed = ""
        if a.pay_date is not None:
            pay = as_session_date(a.pay_date)
        else:
            pay = session + pd.Timedelta(days=self.cfg.pay_date_lag_days)
            if self.roll_to_session is not None:
                pay = self.roll_to_session(pay)
            fid = Fidelity.T2_PROXY
            assumed = f"pay_date unknown, assumed ex+{self.cfg.pay_date_lag_days}d"
        gross = pos.qty * amt * pos.multiplier
        if pos.qty > 0:
            wh = gross * self.cfg.withholding_rate
            acc = Accrual(a.symbol, gross - wh, gross, wh, pay, session, "DIVIDEND")
            note = f"dividend entitlement {gross:.2f} (withholding {wh:.2f}) pay {pay.date()}"
        else:
            if not self.cfg.short_pays_in_lieu:
                return ActionResult(a, True, "short: in-lieu disabled")
            acc = Accrual(a.symbol, gross, gross, 0.0, pay, session, "DIVIDEND_IN_LIEU")
            note = f"short owes in lieu {gross:.2f} pay {pay.date()}"
        acct.accruals.append(acc)
        # Dividends paid at/before the ex-date session start (pay_date <= session)
        # are posted immediately by the caller via pay_due.
        if assumed:
            note = f"{assumed}; {note}"
        return ActionResult(a, True, note, accrual=acc, fidelity=fid)

    def _delist(self, a: CorporateAction, acct: Account, ts) -> ActionResult:
        pos = acct.get(a.symbol)
        if pos is None:
            return ActionResult(a, True, "no position", cancel_orders_for=(a.symbol,))
        val = float(a.amount) if a.amount is not None else float("nan")
        fid = a.fidelity
        note = ""
        if not (val == val) or val < 0:
            val = pos.mark if pos.mark == pos.mark else pos.avg_price
            fid = Fidelity.T2_PROXY
            note = "delisting value missing: last mark used"
        qty = pos.qty
        acct.post_cash(ts, Reason.DELISTING, qty * val * pos.multiplier, a.symbol, f"delist {qty:g} @ {val:.6g}")
        acct.change_position(pos.instrument, -qty, val, ts)
        return ActionResult(a, True, note or f"delisted at {val:.6g}", fidelity=fid,
                            cancel_orders_for=(a.symbol,), closed_qty=-qty, close_price=val)

    def _rename(self, a: CorporateAction, acct: Account, instrument_lookup) -> ActionResult:
        new = a.new_symbol
        if not new:
            return ActionResult(a, False, "symbol change without new_symbol")
        pos = acct.positions.pop(a.symbol, None)
        if pos is None:
            return ActionResult(a, True, "no position", cancel_orders_for=(a.symbol,), renamed_to=new)
        inst = None
        if instrument_lookup is not None:
            try:
                inst = instrument_lookup(new)
            except Exception:
                inst = None
        if inst is None:
            inst = replace(pos.instrument, symbol=new)
        pos.instrument = inst
        if new in acct.positions:   # merge (rare)
            other = acct.positions[new]
            tot = other.qty + pos.qty
            other.avg_price = (other.qty * other.avg_price + pos.qty * pos.avg_price) / tot if tot else 0.0
            other.qty = tot
        else:
            acct.positions[new] = pos
        return ActionResult(a, True, f"{a.symbol} -> {new}", cancel_orders_for=(a.symbol,), renamed_to=new)

    def _spinoff(self, a: CorporateAction, acct: Account, ts, instrument_lookup, price_lookup) -> ActionResult:
        pos = acct.get(a.symbol)
        if pos is None or not a.new_symbol:
            return ActionResult(a, True, "no position")
        frac = float(a.amount) if 0.0 <= float(a.amount) < 1.0 else 0.0
        new_qty = pos.qty * float(a.ratio)
        if not self.cfg.fractional_shares:
            new_qty = float(math.trunc(round(new_qty, 9)))
        if new_qty == 0:
            return ActionResult(a, True, "spin-off below one share")
        cost_parent = pos.avg_price * pos.qty
        alloc = cost_parent * frac
        pos.avg_price = (cost_parent - alloc) / pos.qty
        inst = None
        if instrument_lookup is not None:
            try:
                inst = instrument_lookup(a.new_symbol)
            except Exception:
                inst = None
        if inst is None:
            inst = replace(pos.instrument, symbol=a.new_symbol)
        px = float("nan")
        if price_lookup is not None:
            try:
                px = float(price_lookup(a.new_symbol))
            except Exception:
                px = float("nan")
        basis_px = alloc / new_qty if new_qty else 0.0
        acct.change_position(inst, new_qty, basis_px, ts)
        np_ = acct.get(a.new_symbol)
        fid = a.fidelity
        if np_ is not None:
            np_.mark = px if px == px else basis_px
            if px != px:
                fid = Fidelity.T2_PROXY
        # NOTE: parent price drop on ex-date is reflected by the parent's next mark.
        return ActionResult(a, True, f"spin-off {new_qty:g} {a.new_symbol}", fidelity=fid,
                            cancel_orders_for=(a.symbol,), spinoff_symbol=a.new_symbol,
                            spinoff_qty=new_qty, spinoff_basis=alloc)
