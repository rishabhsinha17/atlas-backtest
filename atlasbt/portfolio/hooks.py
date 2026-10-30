"""Option lifecycle plug-in interface (expiration, exercise, assignment,
cash settlement).

The OPTIONS agent owns the authoritative implementation (OCC ex-by-ex,
early-assignment model, pin risk, AM/PM settlement values). The engine only
needs an object implementing :class:`OptionLifecycle`; it applies the returned
:class:`OptionEvent` list to the account (so accounting stays in this
package). :class:`DefaultOptionLifecycle` is a conservative stand-in:

* PHYSICAL and CASH_PM options expire at the END of their ``expiry`` session
  using the underlying's official close of that session.
* CASH_AM options (e.g. SPX monthlies) are settled at the START of the first
  session after ``expiry`` (``Instrument.expiry`` is the last trading day) at
  that session's underlying OPENING price, a proxy for the Special Opening
  Quotation (flagged T2).
* Automatic exercise if in the money by >= $0.01 (OCC exercise-by-exception
  threshold); long ITM -> exercise, short ITM -> assignment; OTM -> expire
  worthless. No early exercise/assignment (flagged in docs as a limitation).
* Missing settlement price: the last known underlying close is used (T2);
  if none exists the option is closed at its last mark (T2) - never at 0.
* Non-standard deliverables (after OCC split adjustments) are honoured:
  intrinsic = deliverable*S/multiplier - strike; delivery price per share =
  strike*multiplier/deliverable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

import pandas as pd

from ..core.types import Fidelity, Instrument, Right, SettlementType
from .account import Position


@dataclass(frozen=True)
class OptionEvent:
    kind: str                         # 'EXPIRE_WORTHLESS' | 'EXERCISE' | 'ASSIGNMENT' | 'CASH_SETTLE' | 'SETTLE_AT_MARK'
    option: Instrument
    option_qty: float                 # signed change to the option position (closes it)
    option_price: float               # per-unit price at which the option leg is closed
    underlying: Optional[str] = None  # physical delivery: stock symbol
    underlying_qty: float = 0.0       # signed shares delivered (+ received)
    underlying_price: float = 0.0     # strike (price per share of the delivery)
    settlement_value: float = float("nan")
    fidelity: Fidelity = Fidelity.T1_BAR
    note: str = ""


class UnderlyingPrices(Protocol):
    def close(self, symbol: str, session: pd.Timestamp) -> float: ...
    def open(self, symbol: str, session: pd.Timestamp) -> float: ...
    # optional: def last_close(self, symbol: str) -> float  (fallback when close/open is missing)


@runtime_checkable
class OptionLifecycle(Protocol):
    def at_session_start(self, session: pd.Timestamp, prev_session: Optional[pd.Timestamp],
                         options: Sequence[Position], prices: UnderlyingPrices) -> list[OptionEvent]: ...

    def at_session_end(self, session: pd.Timestamp, options: Sequence[Position],
                       prices: UnderlyingPrices) -> list[OptionEvent]: ...


def intrinsic(inst: Instrument, s: float) -> float:
    """Intrinsic value per quoted unit (premium basis). Handles non-standard
    deliverables: value per contract = deliverable*S - strike*multiplier."""
    k = inst.strike or 0.0
    mult = inst.multiplier or 100.0
    sd = s * inst.deliverable_shares / mult
    if inst.right == Right.CALL:
        return max(sd - k, 0.0)
    return max(k - sd, 0.0)


@dataclass
class DefaultOptionLifecycle:
    itm_threshold: float = 0.01

    def _settle(self, p: Position, s: float, fidelity: Fidelity, note: str, prices=None) -> list[OptionEvent]:
        inst = p.instrument
        q = p.qty
        if not (s == s) and prices is not None and hasattr(prices, "last_close"):
            try:
                s = float(prices.last_close(inst.underlying or ""))
            except Exception:
                s = float("nan")
            if s == s:
                fidelity = Fidelity.T2_PROXY
                note = "settlement price missing: last known underlying close used"
        if not (s == s):
            mk = p.mark if p.mark == p.mark else p.avg_price
            return [OptionEvent("SETTLE_AT_MARK", inst, -q, float(mk), fidelity=Fidelity.T2_PROXY,
                                note="no underlying price at expiry: closed at last option mark (flagged)")]
        iv = intrinsic(inst, s)
        if inst.settlement in (SettlementType.CASH_AM, SettlementType.CASH_PM):
            if iv <= 0:
                return [OptionEvent("EXPIRE_WORTHLESS", inst, -q, 0.0, settlement_value=s, fidelity=fidelity, note=note)]
            return [OptionEvent("CASH_SETTLE", inst, -q, iv, settlement_value=s, fidelity=fidelity, note=note)]
        # physical
        if iv < self.itm_threshold:
            return [OptionEvent("EXPIRE_WORTHLESS", inst, -q, 0.0, settlement_value=s, fidelity=fidelity, note=note)]
        shares = abs(q) * inst.deliverable_shares
        # long call / short put -> receive shares; long put / short call -> deliver shares
        receive = (inst.right == Right.CALL) == (q > 0)
        kind = "EXERCISE" if q > 0 else "ASSIGNMENT"
        # aggregate exercise price per contract = strike x multiplier, spread over the deliverable
        px = (inst.strike or 0.0) * inst.multiplier / inst.deliverable_shares
        return [OptionEvent(kind, inst, -q, 0.0, underlying=(inst.underlying or "").upper(),
                            underlying_qty=shares if receive else -shares,
                            underlying_price=px, settlement_value=s,
                            fidelity=fidelity, note=note)]

    def at_session_start(self, session, prev_session, options, prices) -> list[OptionEvent]:
        out: list[OptionEvent] = []
        for p in options:
            inst = p.instrument
            if inst.settlement != SettlementType.CASH_AM or inst.expiry is None:
                continue
            if pd.Timestamp(inst.expiry) < pd.Timestamp(session):
                s = prices.open(inst.underlying or "", session)
                out += self._settle(p, s, Fidelity.T2_PROXY, "AM settlement: SOQ proxied by session open", prices)
        return out

    def at_session_end(self, session, options, prices) -> list[OptionEvent]:
        out: list[OptionEvent] = []
        for p in options:
            inst = p.instrument
            if inst.expiry is None or inst.settlement == SettlementType.CASH_AM:
                continue
            if pd.Timestamp(inst.expiry) <= pd.Timestamp(session):
                s = prices.close(inst.underlying or "", session)
                out += self._settle(p, s, Fidelity.T1_BAR, "PM/physical expiry at official close", prices)
        return out
