"""Order manager: ids, latency, TIF expiry, OCO groups, brackets.

OCO semantics are *reduce-on-fill* (IBKR "OCA with reduce"): when an order in
a group fills ``q`` units, every other active order in the group has its
remaining quantity reduced by ``q`` (cancelled when it reaches zero). For a
complete fill this is classic one-cancels-other; for partial fills it keeps
bracket exits consistent with the position.

Brackets: a parent (entry) order plus child templates (protective stop,
profit target, time exit). Children are created when — and in proportion to
how much — the parent fills. Children share an OCO group.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Optional

import pandas as pd

from ..core.types import Instrument, Order, OrderStatus, OrderType, TimeInForce

ACTIVE = (OrderStatus.PENDING, OrderStatus.WORKING, OrderStatus.PARTIAL)
DONE = (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED, OrderStatus.REJECTED)


@dataclass
class OrderExtras:
    outside_rth: bool = False
    initial_stop: Optional[float] = None     # for R-multiple accounting
    activate_at_ns: Optional[int] = None     # earliest bar start this order may use (time exits)
    parent_id: Optional[int] = None
    valid_session_ns: Optional[int] = None   # DAY orders: session (naive date ns) they are valid for
    target_session_ns: Optional[int] = None  # MOO/MOC: session whose auction they join
    evaluated: bool = False                  # IOC: has been offered data at least once
    in_bar_from: Optional[float] = None      # activated inside the current bar at path position
    is_liquidation: bool = False
    is_flatten: bool = False
    submitted_ns: int = 0
    eligible_ns: int = 0
    fills: int = 0


@dataclass
class ChildTemplate:
    type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    tag: str = ""
    tif: TimeInForce = TimeInForce.GTC
    activate_at: Optional[pd.Timestamp] = None
    outside_rth: bool = False


@dataclass
class BracketSpec:
    parent_id: int
    templates: list[ChildTemplate]
    oco_group: str
    child_ids: dict[int, int] = field(default_factory=dict)   # template index -> order id


@dataclass
class OrderEvent:
    ts: pd.Timestamp
    order_id: int
    event: str
    detail: str = ""


class OrderManager:
    def __init__(self):
        self.orders: dict[int, Order] = {}
        self.extras: dict[int, OrderExtras] = {}
        self._active_by_symbol: dict[str, list[int]] = {}
        self._brackets: dict[int, BracketSpec] = {}
        self._next_id = 1
        self.events: list[OrderEvent] = []
        self.rejected: list[dict] = []

    # ------------------------------------------------------------- ids
    def next_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    @staticmethod
    def symbol_of(order: Order) -> str:
        if order.instrument is not None:
            return order.instrument.symbol
        return "COMBO:" + "|".join(f"{l.ratio:+d}{l.instrument.symbol}" for l in order.legs)

    # ------------------------------------------------------------- lifecycle
    def add(self, order: Order, extras: OrderExtras, ts: pd.Timestamp) -> Order:
        if order.id < 0:
            order.id = self.next_id()
        order.status = OrderStatus.WORKING
        self.orders[order.id] = order
        self.extras[order.id] = extras
        self._active_by_symbol.setdefault(self.symbol_of(order), []).append(order.id)
        self.events.append(OrderEvent(ts, order.id, "SUBMIT", f"{order.type.value} {order.quantity:+g} {order.tag}"))
        return order

    def reject(self, order: Order, reason: str, ts: pd.Timestamp) -> Order:
        if order.id < 0:
            order.id = self.next_id()
        order.status = OrderStatus.REJECTED
        order.reject_reason = reason
        self.orders[order.id] = order
        self.extras.setdefault(order.id, OrderExtras())
        self.rejected.append({"ts": ts, "order_id": order.id, "symbol": self.symbol_of(order),
                              "quantity": order.quantity, "type": order.type.value, "tag": order.tag,
                              "reason": reason})
        self.events.append(OrderEvent(ts, order.id, "REJECT", reason))
        return order

    def _deactivate(self, order: Order) -> None:
        lst = self._active_by_symbol.get(self.symbol_of(order))
        if lst and order.id in lst:
            lst.remove(order.id)

    def cancel(self, order_id: int, ts: pd.Timestamp, reason: str = "cancelled",
               status: OrderStatus = OrderStatus.CANCELLED) -> bool:
        o = self.orders.get(order_id)
        if o is None or o.status not in ACTIVE:
            return False
        o.status = status
        if reason and not o.reject_reason:
            o.reject_reason = reason
        self._deactivate(o)
        self.events.append(OrderEvent(ts, order_id, status.value, reason))
        # cancelling an unfilled bracket parent cancels its pending children templates
        return True

    def cancel_all(self, ts: pd.Timestamp, symbol: Optional[str] = None, reason: str = "cancel_all") -> int:
        n = 0
        for oid in list(self.active_ids(symbol)):
            n += self.cancel(oid, ts, reason)
        return n

    # ------------------------------------------------------------- queries
    def active_ids(self, symbol: Optional[str] = None) -> list[int]:
        if symbol is None:
            ids = [i for lst in self._active_by_symbol.values() for i in lst]
            return sorted(ids)
        return sorted(self._active_by_symbol.get(symbol, []))

    def active(self, symbol: Optional[str] = None) -> list[Order]:
        return [self.orders[i] for i in self.active_ids(symbol)]

    def active_symbols(self) -> list[str]:
        return sorted(s for s, lst in self._active_by_symbol.items() if lst)

    # ------------------------------------------------------------- brackets
    def register_bracket(self, parent: Order, templates: list[ChildTemplate], oco_group: str) -> BracketSpec:
        spec = BracketSpec(parent.id, templates, oco_group)
        self._brackets[parent.id] = spec
        return spec

    def bracket_of(self, parent_id: int) -> Optional[BracketSpec]:
        return self._brackets.get(parent_id)

    # ------------------------------------------------------------- fills
    def on_fill(self, order: Order, qty: float, ts: pd.Timestamp) -> tuple[list[Order], list[ChildTemplate], list[tuple[int, float]]]:
        """Record a fill of ``qty`` (signed) on ``order``.

        Returns (orders cancelled/reduced by OCO, child templates to create
        with their quantity in the returned list of (template_index, qty)).
        The engine creates/updates children (needs submit-time validation)."""
        order.filled_qty += qty
        ex = self.extras[order.id]
        ex.fills += 1
        if abs(order.remaining) <= 1e-9:
            order.status = OrderStatus.FILLED
            self._deactivate(order)
        else:
            order.status = OrderStatus.PARTIAL
        self.events.append(OrderEvent(ts, order.id, "FILL", f"{qty:+g}"))
        touched: list[Order] = []
        if order.oco_group:
            for oid in self.active_ids(None):
                other = self.orders[oid]
                if other.id == order.id or other.oco_group != order.oco_group:
                    continue
                # reduce other's remaining by |qty| (same units)
                rem = other.remaining
                red = min(abs(qty), abs(rem))
                new_qty = other.quantity - red * (1 if other.quantity > 0 else -1)
                other.quantity = new_qty
                if abs(other.remaining) <= 1e-9:
                    if abs(other.filled_qty) > 1e-9:
                        other.status = OrderStatus.FILLED
                        self._deactivate(other)
                        self.events.append(OrderEvent(ts, other.id, "OCO_DONE", f"by {order.id}"))
                    else:
                        self.cancel(other.id, ts, f"OCO with {order.id}")
                touched.append(other)
        child_updates: list[tuple[int, float]] = []
        spec = self._brackets.get(order.id)
        if spec is not None:
            for k, _t in enumerate(spec.templates):
                child_updates.append((k, -qty))
        return touched, (spec.templates if spec else []), child_updates

    # ------------------------------------------------------------- expiry
    def expire_session(self, session_ns: int, ts: pd.Timestamp) -> list[int]:
        """Expire DAY orders (and MOO/MOC) valid for ``session_ns``."""
        out = []
        for oid in self.active_ids(None):
            o = self.orders[oid]
            ex = self.extras[oid]
            if o.tif == TimeInForce.DAY and ex.valid_session_ns is not None and ex.valid_session_ns <= session_ns:
                self.cancel(oid, ts, "DAY order expired", OrderStatus.EXPIRED)
                out.append(oid)
            elif o.type in (OrderType.MOO, OrderType.MOC) and ex.target_session_ns is not None \
                    and ex.target_session_ns <= session_ns:
                self.cancel(oid, ts, "auction order not filled", OrderStatus.EXPIRED)
                out.append(oid)
        return out

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for oid in sorted(self.orders):
            o = self.orders[oid]
            ex = self.extras.get(oid, OrderExtras())
            rows.append({"order_id": oid, "symbol": self.symbol_of(o), "type": o.type.value,
                         "quantity": o.quantity, "filled_qty": o.filled_qty, "limit_price": o.limit_price,
                         "stop_price": o.stop_price, "tif": o.tif.value, "tag": o.tag, "oco_group": o.oco_group,
                         "status": o.status.value, "submitted_at": o.submitted_at, "eligible_at": o.eligible_at,
                         "reason": o.reject_reason, "parent_id": ex.parent_id, "short_sale": o.is_short_sale})
        return pd.DataFrame(rows)
