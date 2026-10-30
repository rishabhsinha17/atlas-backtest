"""Adapters implementing the ENGINE agent's plug-in protocols
(``atlasbt.portfolio.hooks.OptionLifecycle`` and
``atlasbt.portfolio.margin.OptionMarginModel``) on top of
``atlasbt.options.lifecycle`` and ``atlasbt.options.margin``.

Use in an engine config::

    from atlasbt.options.plugins import OccOptionLifecycle, CboeStrategyMargin
    cfg = EngineConfig(..., option_lifecycle=OccOptionLifecycle(rate=lambda s: 0.04,
                       dividends=lambda und, session: [(ex_date, amount), ...]),
                       option_margin=CboeStrategyMargin())
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from atlasbt.core.types import SettlementType

from atlasbt.pricing.time import settlement_session

from .lifecycle import ExpiryPolicy, LifecycleEvent, early_assignment_events, expiration_events, settlement_due
from .margin import OptionPosition, strategy_margin

BROAD_INDEXES = {"SPX", "SPXW", "XSP", "NDX", "RUT", "DJX", "OEX", "XEO", "MRUT", "NANOS"}
BROAD_ETFS = {"SPY", "QQQ", "IWM", "DIA", "VOO", "IVV", "VTI", "MDY", "EFA", "EEM", "RSP"}
LEVERAGED_ETFS = {"SSO": 2.0, "UPRO": 3.0, "SPXL": 3.0, "QLD": 2.0, "TQQQ": 3.0, "SDS": 2.0, "SPXU": 3.0,
                  "SPXS": 3.0, "SQQQ": 3.0, "QID": 2.0, "TNA": 3.0, "TZA": 3.0, "UWM": 2.0, "SOXL": 3.0,
                  "SOXS": 3.0, "TMF": 3.0, "TMV": 3.0, "UVXY": 1.5, "SVXY": 0.5}
LEVERAGED_BROAD = {"SSO", "UPRO", "SPXL", "QLD", "TQQQ", "SDS", "SPXU", "SPXS", "SQQQ", "QID", "TNA", "TZA", "UWM"}


def underlying_class(underlying: str) -> tuple[str, float]:
    """(Cboe class, leverage factor) for margin.  Sector SPDRs are narrow-based
    (20%); broad-based index ETFs and indexes 15%."""
    u = (underlying or "").upper().lstrip("^$")
    if u in BROAD_INDEXES:
        return "broad_index", 1.0
    if u in LEVERAGED_ETFS:
        return ("broad_etf" if u in LEVERAGED_BROAD else "equity"), max(LEVERAGED_ETFS[u], 1.0)
    if u in BROAD_ETFS:
        return "broad_etf", 1.0
    return "equity", 1.0


def _to_hook_event(ev: LifecycleEvent):
    from atlasbt.portfolio.hooks import OptionEvent
    kind = {"CASH_SETTLEMENT": "CASH_SETTLE", "EARLY_ASSIGNMENT": "ASSIGNMENT"}.get(ev.kind, ev.kind)
    opt_px = 0.0
    if ev.kind == "CASH_SETTLEMENT" and ev.option_qty:
        opt_px = abs(ev.cash) / (abs(ev.option_qty) * ev.instrument.multiplier)
    return OptionEvent(kind=kind, option=ev.instrument, option_qty=ev.option_qty, option_price=opt_px,
                       underlying=(ev.stock_symbol or None), underlying_qty=ev.stock_qty,
                       underlying_price=ev.stock_price, settlement_value=ev.reference_price,
                       fidelity=ev.fidelity, note=f"{ev.kind}: {ev.reason}")


@dataclass
class OccOptionLifecycle:
    """OCC ex-by-ex expiration + AM/PM settlement + early-assignment model.

    ``rate(session) -> continuous r`` and ``dividends(underlying, session) ->
    [(ex_date, amount), ...]`` (dividends declared/known at ``session``) enable
    the early-assignment model; without ``dividends`` only the deep-ITM-put
    rule runs (call/ex-dividend assignment then is NOT modeled — flagged)."""

    policy: ExpiryPolicy = field(default_factory=ExpiryPolicy)
    rate: Optional[Callable[[pd.Timestamp], float]] = None
    dividends: Optional[Callable[[str, pd.Timestamp], list]] = None
    early_assignment: bool = True
    model_value: Optional[Callable[[object, pd.Timestamp], float]] = None   # PIT model price fallback

    def at_session_start(self, session, prev_session, options: Sequence, prices) -> list:
        """CASH_AM options settle at the SOQ of their settlement session; the
        engine's opening print of the index proxies the SOQ (flagged T2).
        Overdue AM options (settlement session skipped) settle at this open."""
        session = pd.Timestamp(session).normalize()
        due, overdue = {}, {}
        for p in options:
            i = p.instrument
            if i.settlement != SettlementType.CASH_AM or i.expiry is None:
                continue
            if settlement_due(i, session):
                due[i] = p.qty
            elif pd.Timestamp(i.expiry) < session:
                overdue[i] = p.qty
        ref = lambda u, s, kind: prices.open(u, session)  # noqa: E731
        evs = expiration_events(due, session, ref, self.policy)
        for i, q in overdue.items():
            S = float(prices.open(i.underlying, session))
            evs += [replace(e, reason=e.reason + " (overdue AM settlement at later open)") for e in
                    expiration_events({i: q}, settlement_session(i.expiry, i.settlement),
                                      lambda u, s, k: S, self.policy)]
        return [_to_hook_event(e) for e in evs]

    def at_session_end(self, session, options: Sequence, prices) -> list:
        session = pd.Timestamp(session).normalize()
        pos = {}
        for p in options:
            i = p.instrument
            if i.expiry is None or i.settlement == SettlementType.CASH_AM:
                continue
            if settlement_due(i, session) or pd.Timestamp(i.expiry) < session:
                pos[i] = p.qty
        ref = lambda u, s, kind: prices.close(u, session)  # noqa: E731
        out = [_to_hook_event(e) for e in expiration_events(
            {i: q for i, q in pos.items() if settlement_due(i, session)}, session, ref, self.policy)]
        # overdue physical/PM (session skipped): settle at this close, flagged by note
        for i, q in pos.items():
            if not settlement_due(i, session):
                evs = expiration_events({i: q}, pd.Timestamp(i.expiry), lambda u, s, k: prices.close(u, session), self.policy)
                out += [_to_hook_event(e) for e in evs]
        if self.early_assignment:
            shorts = {p.instrument: p.qty for p in options if p.qty < 0 and p.instrument.expiry is not None
                      and p.instrument.exercise is not None and p.instrument.exercise.value == "AMERICAN"
                      and p.instrument not in pos}
            if shorts:
                r = float(self.rate(session)) if self.rate else 0.0
                from .lifecycle import OptionMark
                # only a mark observed in THIS session counts (Position.mark_ts); optional
                # engine-supplied PIT model value via ``model_value(instrument, session)``
                marks = {}
                for p in options:
                    mv = float(self.model_value(p.instrument, session)) if self.model_value else float("nan")
                    marks[p.instrument.symbol] = OptionMark(
                        mid=p.mark if p.mark == p.mark else float("nan"),
                        ts=getattr(p, "mark_ts", None) if getattr(p, "mark_ts", None) is not None else pd.Timestamp("1900-01-01", tz="UTC"),
                        fidelity=getattr(p, "mark_fidelity", None), model_value=mv)
                spot = lambda u: prices.close(u, session)  # noqa: E731
                divs = {}
                if self.dividends:
                    for i in shorts:
                        divs.setdefault(i.underlying, self.dividends(i.underlying, session))
                evs = early_assignment_events(shorts, session, spot, marks, r, divs)
                out += [_to_hook_event(e) for e in evs]
        return out


@dataclass
class CboeStrategyMargin:
    """``OptionMarginModel``: Cboe strategy-based requirement per underlying
    (``atlasbt.options.margin.strategy_margin``) using current option marks
    (maintenance).  Returns sum over underlyings of max(0, requirement - short
    option market value), i.e. the amount NAV must exceed on top of the stock
    Reg-T requirement (credits from covered calls are not used to reduce the
    stock requirement -> slightly conservative)."""

    # The returned figure already includes 100% of long option premium (longs have no
    # loan value), so the engine must not add long option value separately.
    includes_long_premium: bool = True
    # date whose rules apply (engine may set it each step); None -> latest position mark_ts
    as_of: Optional[pd.Timestamp] = None

    def requirement(self, account, underlying_price: Callable[[str], float]) -> float:
        from atlasbt.portfolio.account import AccountType
        opts = account.option_positions()
        if not opts:
            return 0.0
        stock = {p.symbol.upper(): p.qty for p in account.stock_positions()}
        by_und: dict = {}
        for p in opts:
            by_und.setdefault((p.instrument.underlying or "").upper(), []).append(p)
        acct = "cash" if getattr(account, "account_type", None) == AccountType.CASH else "margin"
        total = 0.0
        for und, ps in by_und.items():
            S = underlying_price(und)
            if not (S == S and S > 0):
                S = max(p.instrument.strike for p in ps)   # unknown price: strike-based (conservative)
            cls, lev = underlying_class(und)
            pos = [OptionPosition(p.instrument.strike, p.instrument.right.value, pd.Timestamp(p.instrument.expiry), p.qty,
                                  (p.mark if p.mark == p.mark else p.avg_price), p.instrument.multiplier,
                                  p.instrument.exercise.value if p.instrument.exercise else "AMERICAN",
                                  p.instrument.settlement.value if p.instrument.settlement else "PHYSICAL") for p in ps]
            when = self.as_of
            if when is None:
                ts = [getattr(p, "mark_ts", None) for p in ps]
                ts = [t for t in ts if t is not None]
                when = max(ts) if ts else None
            res = strategy_margin(pos, S, stock.get(und, 0.0) * 1.0, cls, lev, account=acct, as_of=when, now=when)
            total += max(0.0, res.net_requirement)
        return float(total)
