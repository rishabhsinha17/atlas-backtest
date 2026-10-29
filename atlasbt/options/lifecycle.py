"""Option lifecycle: expiration, exercise-by-exception, cash settlement (AM SOQ /
PM close), physical delivery, early-assignment model, pin-risk policy and
non-standard deliverables after splits / special dividends.

Engine integration (pure functions; the portfolio books the returned events):

* ``settlement_due(instr, session)`` -> True on the session whose value fixes
  settlement: the expiry (last trading) day for PHYSICAL / CASH_PM, the next
  session (SOQ) for CASH_AM.
* ``expiration_events(positions, session, ref_price, policy)`` at the END of
  that session.  ``ref_price(underlying, session, kind)`` must return the
  official close (kind='close') or the special opening quotation
  (kind='soq'; the engine may proxy it with the session's opening print of the
  index and flag the fill T2_PROXY).
* ``early_assignment_events(...)`` at the end of every session for short
  American options (deterministic, conservative model; see function doc).
* ``LifecycleEvent.to_fills()`` converts an event to ``core.types.Fill``s
  (liquidity 'EXERCISE' / 'ASSIGNMENT' / 'SETTLEMENT'); fees for
  exercise/assignment are left to the execution cost model.

OCC exercise-by-exception: an expiring equity/index option in-the-money by
>= $0.01 is exercised automatically (OCC threshold for customer and firm
accounts since 2008; earlier eras used higher customer thresholds, which this
model does NOT reproduce — configurable via ``ExpiryPolicy.threshold``).
Short positions are assumed assigned in full whenever the option is
auto-exercised (aggregate holder behaviour; conservative).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping, Optional

import numpy as np
import pandas as pd

from atlasbt.core.types import AssetClass, CostBreakdown, Fidelity, Fill, Instrument, SettlementType

from atlasbt.pricing.time import settlement_session


@dataclass(frozen=True)
class ExpiryPolicy:
    threshold: float = 0.01          # OCC ex-by-ex ($ ITM)
    pin_band: float = 0.0            # $ band around strike for pin-risk handling (physical only)
    pin_policy: str = "occ"          # 'occ' | 'adverse'
    # 'adverse': short physical options with |S-K| <= pin_band are treated as assigned
    #            even if marginally OTM (after-hours move / contrarian exercise risk).


@dataclass(frozen=True)
class LifecycleEvent:
    kind: str                       # EXPIRE_WORTHLESS | EXERCISE | ASSIGNMENT | CASH_SETTLEMENT | EARLY_ASSIGNMENT
    instrument: Instrument
    option_qty: float               # change in option position (closes it): -qty_held
    stock_symbol: Optional[str]
    stock_qty: float                # shares received (+) / delivered (-); 0 if cash settled
    stock_price: float              # strike (delivery price per share)
    cash: float                     # USD cash flow from settlement (+ receive); physical: -stock_qty*strike
    reference_price: float          # underlying close / SOQ used
    session: pd.Timestamp
    ts: pd.Timestamp                # UTC time the event takes effect
    fidelity: Fidelity
    reason: str = ""

    def to_fills(self) -> list[Fill]:
        liq = {"EXPIRE_WORTHLESS": "SETTLEMENT", "CASH_SETTLEMENT": "SETTLEMENT", "EXERCISE": "EXERCISE",
               "ASSIGNMENT": "ASSIGNMENT", "EARLY_ASSIGNMENT": "ASSIGNMENT"}[self.kind]
        fills = []
        opt_px = 0.0
        if self.kind == "CASH_SETTLEMENT":
            # settle the option at its intrinsic value: cash = -option_qty * mult * px
            opt_px = abs(self.cash) / (abs(self.option_qty) * self.instrument.multiplier) if self.option_qty else 0.0
        fills.append(Fill(order_id=-1, instrument=self.instrument, ts=self.ts, quantity=self.option_qty,
                          price=opt_px, costs=CostBreakdown(), fidelity=self.fidelity, liquidity=liq,
                          note=f"{self.kind}: {self.reason}"))
        if self.stock_qty:
            stk = Instrument(symbol=self.stock_symbol, asset_class=AssetClass.STOCK)
            fills.append(Fill(order_id=-1, instrument=stk, ts=self.ts, quantity=self.stock_qty, price=self.stock_price,
                              costs=CostBreakdown(), fidelity=self.fidelity, liquidity=liq,
                              note=f"{self.kind} delivery {self.instrument.symbol}"))
        return fills


def effective_underlying(instr: Instrument, S: float) -> float:
    """Underlying value per unit of the option's price quote: deliverable*S/multiplier
    (== S for standard 100-share contracts; 150*S/100 after a 3:2 split, 10*S/100 after
    a 1:10 reverse split)."""
    return float(S) * instr.deliverable_shares / instr.multiplier


def intrinsic(instr: Instrument, S: float) -> float:
    """Intrinsic value per price unit (x multiplier = per contract):
    per contract = deliverable*S - K*multiplier (calls), reverse for puts."""
    se = effective_underlying(instr, S)
    return max(se - instr.strike, 0.0) if instr.right.value == "C" else max(instr.strike - se, 0.0)


def delivery_price(instr: Instrument) -> float:
    """Price per delivered share: aggregate exercise price K*multiplier spread over the deliverable."""
    return instr.strike * instr.multiplier / instr.deliverable_shares


def settlement_due(instr: Instrument, session) -> bool:
    s = pd.Timestamp(session).normalize()
    return settlement_session(instr.expiry, instr.settlement) == s


def _close_ts(session):
    from atlasbt.core import clock
    cal = clock.nyse()
    return cal.open_close(session)[1] if cal.is_session(session) else clock.et(session, "16:00")


def expiration_events(positions: Mapping[Instrument, float], session, ref_price: Callable[[str, pd.Timestamp, str], float],
                      policy: ExpiryPolicy = ExpiryPolicy()) -> list[LifecycleEvent]:
    """Process every option position whose settlement is fixed on ``session``."""
    session = pd.Timestamp(session).normalize()
    out: list[LifecycleEvent] = []
    ts = _close_ts(session)
    for instr, qty in positions.items():
        if not instr.is_option or qty == 0 or not settlement_due(instr, session):
            continue
        st = instr.settlement or SettlementType.PHYSICAL
        kind_px = "soq" if st == SettlementType.CASH_AM else "close"
        S = float(ref_price(instr.underlying, session, kind_px))
        fid = Fidelity.T2_PROXY if st == SettlementType.CASH_AM else Fidelity.T1_BAR
        itm = intrinsic(instr, S)
        exercised = itm >= policy.threshold - 1e-12
        pin = abs(effective_underlying(instr, S) - instr.strike) <= policy.pin_band
        if st != SettlementType.PHYSICAL:
            cash = itm * instr.multiplier * qty if exercised else 0.0
            out.append(LifecycleEvent("CASH_SETTLEMENT" if exercised else "EXPIRE_WORTHLESS", instr, -qty, None, 0.0, 0.0,
                                      cash, S, session, ts, fid,
                                      f"{kind_px}={S:.4f} intrinsic={itm:.4f}"))
            continue
        if qty < 0 and policy.pin_policy == "adverse" and pin and not exercised:
            exercised = True
            why = f"pin-risk adverse assignment |S_eff-K|={abs(effective_underlying(instr, S) - instr.strike):.4f}<= {policy.pin_band}"
        else:
            why = f"close={S:.4f} intrinsic={itm:.4f}" + (" PIN" if pin and policy.pin_band > 0 else "")
        if not exercised:
            out.append(LifecycleEvent("EXPIRE_WORTHLESS", instr, -qty, None, 0.0, 0.0, 0.0, S, session, ts, fid, why))
            continue
        shares = instr.deliverable_shares * abs(qty)
        is_call = instr.right.value == "C"
        if qty > 0:   # long exercises: call buys shares, put sells shares
            sq = shares if is_call else -shares
            kind = "EXERCISE"
        else:         # short assigned: call delivers shares, put receives shares
            sq = -shares if is_call else shares
            kind = "ASSIGNMENT"
        px = delivery_price(instr)
        out.append(LifecycleEvent(kind, instr, -qty, instr.underlying, sq, px, -sq * px, S,
                                  session, ts, fid, why))
    return out


# --------------------------------------------------------------------------
# Early exercise / assignment (American)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class OptionMark:
    """A mark supplied by the engine for the early-assignment test."""
    mid: float
    ts: Optional[pd.Timestamp] = None        # when the mark was observed (UTC)
    fidelity: Optional[Fidelity] = None
    model_value: float = float("nan")         # optional PIT model value (fallback)


def _same_session_value(option_mid, session) -> tuple[float, str]:
    """Value usable as the option price at ``session``: a finite mid observed in the
    SAME session (or a bare float, which the caller asserts is same-session), else
    the PIT model value if supplied, else NaN (-> no assignment)."""
    from atlasbt.core import clock
    if isinstance(option_mid, OptionMark):
        mid, ts = option_mid.mid, option_mid.ts
        same = ts is None or clock.session_date_of(clock.to_utc(ts)) == session
        if np.isfinite(mid) and same:
            return float(mid), "same-session mark"
        if np.isfinite(option_mid.model_value):
            return float(option_mid.model_value), "PIT model value (no same-session mark)"
        return float("nan"), "no same-session mark"
    v = float(option_mid) if option_mid is not None else float("nan")
    return (v, "mark") if np.isfinite(v) else (float("nan"), "no mark")


def early_exercise_rational(instr: Instrument, S: float, option_mid, session, r: float,
                            dividends: Optional[list] = None) -> tuple[bool, str]:
    """Holder's early-exercise decision (deterministic rule of thumb).

    ``option_mid``: an ``OptionMark`` (mid + observation ts [+ PIT model value]) or a
    float the caller guarantees is a same-session mark.  Without a valid same-session
    mark (or model fallback) NO assignment is generated — a missing price is never
    treated as zero extrinsic value.""" + """


    * call: tomorrow is the ex-date of a cash dividend D, the call is ITM and
      its extrinsic value (mid - intrinsic) < D (exercise captures D).
    * put : ITM and extrinsic < interest on the strike to expiry minus PV of
      dividends before expiry, K(1-e^{-rT}) - PV(D) (exercise earns interest).
    ``dividends``: list of (ex_date, amount) known at ``session``.
    """
    if instr.exercise is None or instr.exercise.value != "AMERICAN":
        return False, "european"
    itm = intrinsic(instr, S)
    if itm <= 0:
        return False, "otm"
    session = pd.Timestamp(session).normalize()
    value, src = _same_session_value(option_mid, session)
    if not np.isfinite(value):
        return False, f"{src}: early-assignment test skipped"
    extrinsic = max(value - itm, 0.0)
    from atlasbt.core import clock
    nxt = clock.nyse().next_session(session)
    divs = dividends or []
    if instr.right.value == "C":
        for ex, amt in divs:
            if pd.Timestamp(ex).normalize() == nxt and pd.Timestamp(ex) <= instr.expiry + pd.Timedelta(days=1):
                amt_u = amt * instr.deliverable_shares / instr.multiplier   # dividend per price unit
                if extrinsic < amt_u:
                    return True, f"call ex-div tomorrow: extrinsic {extrinsic:.4f} < dividend {amt_u:.4f} ({src})"
        return False, "call: no dividend capture"
    T = max((pd.Timestamp(instr.expiry) - session).days, 0) / 365.0
    pv_d = instr.deliverable_shares / instr.multiplier * sum(a * np.exp(-r * max((pd.Timestamp(e) - session).days, 0) / 365.0)
               for e, a in divs if session < pd.Timestamp(e) <= pd.Timestamp(instr.expiry))
    carry = instr.strike * (1 - np.exp(-r * T)) - pv_d
    if carry > 0 and extrinsic < carry:
        return True, f"deep ITM put: extrinsic {extrinsic:.4f} < carry {carry:.4f} ({src})"
    return False, "put: extrinsic >= carry"


def early_assignment_events(positions: Mapping[Instrument, float], session, spot: Mapping[str, float] | Callable,
                            option_mid: Mapping | Callable, r: float,
                            dividends: Mapping[str, list] | None = None) -> list[LifecycleEvent]:
    """Short American options assigned at the end of ``session`` when the
    holder rule in ``early_exercise_rational`` says exercise (whole position,
    conservative).  Expiring-today options are left to ``expiration_events``.
    ``option_mid`` maps symbol -> OptionMark (preferred) or same-session float mid."""
    session = pd.Timestamp(session).normalize()
    ts = _close_ts(session)
    out = []
    for instr, qty in positions.items():
        if not instr.is_option or qty >= 0 or settlement_due(instr, session):
            continue
        S = spot(instr.underlying) if callable(spot) else spot.get(instr.underlying, np.nan)
        m = option_mid(instr) if callable(option_mid) else option_mid.get(instr.symbol, np.nan)
        if not np.isfinite(S):
            continue
        ex, why = early_exercise_rational(instr, S, m, session, r, (dividends or {}).get(instr.underlying))
        if not ex:
            continue
        shares = instr.deliverable_shares * abs(qty)
        sq = -shares if instr.right.value == "C" else shares
        px = delivery_price(instr)
        out.append(LifecycleEvent("EARLY_ASSIGNMENT", instr, -qty, instr.underlying, sq, px, -sq * px,
                                  S, session, ts, Fidelity.T2_PROXY, why))
    return out


# --------------------------------------------------------------------------
# Corporate-action adjustments (OCC)
# --------------------------------------------------------------------------
def adjust_for_split(instr: Instrument, qty: float, ratio: float) -> tuple[Instrument, float]:
    """OCC contract adjustment for a split of ``ratio`` new shares per old.

    Whole-number forward splits (2:1, 3:1 ...): contracts x ratio, strike / ratio,
    deliverable stays 100 (standard).  Otherwise (3:2, reverse splits):
    contracts and strike unchanged, deliverable = 100 * ratio shares,
    non-standard root (underlying + '1').  The cash multiplier (100) is unchanged
    in both cases (price per share x 100 still defines premium dollars for
    standard contracts; for non-standard contracts the option price quotes per
    original contract unit)."""
    if not instr.is_option:
        raise ValueError("option instrument required")
    if ratio >= 1 and np.isclose(ratio, round(ratio)):
        r = int(round(ratio))
        new = Instrument.option(instr.underlying, instr.expiry, instr.strike / r, instr.right, instr.exercise,
                                instr.settlement, instr.multiplier, root=instr.symbol[:6].strip())
        new = replace(new, deliverable_shares=instr.deliverable_shares)
        return new, qty * r
    root = (instr.underlying + "1")[:6]
    new = Instrument.option(instr.underlying, instr.expiry, instr.strike, instr.right, instr.exercise,
                            instr.settlement, instr.multiplier, root=root)
    new = replace(new, deliverable_shares=instr.deliverable_shares * ratio)
    return new, qty


SPECIAL_DIV_THRESHOLD_PER_CONTRACT = 12.50


def adjust_for_special_dividend(instr: Instrument, amount_per_share: float) -> Instrument:
    """OCC: special (non-ordinary) cash dividends of at least $12.50 per contract
    reduce the strike by the dividend amount; smaller ones and ordinary
    dividends are not adjusted."""
    if amount_per_share * instr.deliverable_shares < SPECIAL_DIV_THRESHOLD_PER_CONTRACT:
        return instr
    new = Instrument.option(instr.underlying, instr.expiry, instr.strike - amount_per_share, instr.right,
                            instr.exercise, instr.settlement, instr.multiplier, root=instr.symbol[:6].strip())
    return replace(new, deliverable_shares=instr.deliverable_shares)
