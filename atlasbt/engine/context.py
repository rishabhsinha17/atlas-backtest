"""Strategy Context implementation (core.protocols.Context).

All data reads are point-in-time at ``ctx.now`` through :class:`PITPortal`;
``history`` never includes the in-progress bar. Orders go through the
backtester (validation, latency, constraints).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..core.clock import ET, to_et
from ..core.types import (
    Fidelity, Instrument, LookaheadError, Order, OrderType, Quote, TimeInForce,
)
from .orders import ChildTemplate
from .pit import PITPortal
from .timeutil import canonical_freq, freq_to_timedelta, is_daily

if TYPE_CHECKING:  # pragma: no cover
    from .backtester import Backtester


def engine_of(ctx: "EngineContext") -> "Backtester":
    """ENGINE-INTERNAL accessor (tests/diagnostics only; strategies must not use it -
    enforced by tests/engine/test_eng_strategy_lint.py)."""
    return ctx._EngineContext__bt


class EngineContext:
    __slots__ = ("__bt", "__pit", "strategy_id", "state", "__weakref__")

    def __init__(self, bt: "Backtester"):
        self.__bt = bt
        self.strategy_id = bt.strategy.id
        self.__pit = PITPortal(bt.portal, lambda: self.now, on_access=bt._note_data_access,
                               universe_filter=bt._filter_universe)
        self.state: dict[str, Any] = {}        # free scratch space for strategies

    # ------------------------------------------------------------------ time
    @property
    def now(self) -> pd.Timestamp:
        return self.__bt.now

    def now_et(self) -> pd.Timestamp:
        return to_et(self.now)

    def session_date(self) -> pd.Timestamp:
        return self.__bt.session

    def minutes_since_open(self) -> float:
        o, _ = self.__bt.session_open_close
        return (self.now - o).total_seconds() / 60.0

    def minutes_to_close(self) -> float:
        _, c = self.__bt.session_open_close
        return (c - self.now).total_seconds() / 60.0

    def is_session_close_bar(self) -> bool:
        _, c = self.__bt.session_open_close
        return self.now >= c

    def calendar(self):
        return self.__bt.cal

    # ------------------------------------------------------------------ account
    @property
    def nav(self) -> float:
        return self.__bt.account.nav

    @property
    def prior_nav(self) -> float:
        return self.__bt.prior_nav

    @property
    def cash(self) -> float:
        return self.__bt.account.cash

    def position(self, symbol: str) -> float:
        return self.__bt.account.position(symbol)

    def positions(self) -> dict[str, float]:
        return self.__bt.account.qty_map()

    def open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        return self.__bt.om.active(symbol)

    def mark(self, symbol: str) -> float:
        """Latest known price (last completed bar close / position mark)."""
        return self.__bt.last_price(symbol)

    def account(self):
        """Read-only view of the account (positions with avg price/marks)."""
        return self.__bt.account

    # ------------------------------------------------------------------ data
    def portal(self) -> PITPortal:
        return self.__pit

    def instrument(self, symbol: str) -> Instrument:
        return self.__bt.instrument(symbol)

    def history(self, symbol: str, n: int, freq: Optional[str] = None, adjusted: bool = True,
                sessions: Sequence[str] = ("RTH",)) -> pd.DataFrame:
        """Last ``n`` COMPLETED bars (known_at <= now). Daily bars of the
        current session are never included before the close."""
        freq = canonical_freq(freq or self.__bt.base_freq)
        now = self.now
        if not adjusted and freq == self.__bt.base_freq and tuple(sessions) == tuple(self.__bt.data_sessions):
            b = self.__bt.md.bars(symbol, freq)
            i1 = b.last_index_at(int(now.value)) + 1
            # cache holds enough history (or all that exists); else fall through to the PIT portal
            if i1 >= n or not self.__bt.md.may_be_truncated(b):
                return b.frame(max(i1 - n, 0), i1)
        cal = self.__bt.cal
        if is_daily(freq):
            sess = self.__bt.session
            back = cal.prev_session(sess, min(n + 5, max(cal.session_index(sess), 1)))
            start = cal.open_close(back)[0] - pd.Timedelta(days=1)
        else:
            td = freq_to_timedelta(freq)
            bars_per_day = max(int(6.5 * 3600 / td.total_seconds()), 1)
            ext = 16 * 3600 / td.total_seconds() if any(s != "RTH" for s in sessions) else bars_per_day
            days = int(np.ceil(n / max(ext, 1))) + 3
            start = now - pd.Timedelta(days=int(days * 7 / 5) + 4)
        df = self.__pit.bars(symbol, freq, start, now + pd.Timedelta(1, "ns"), as_of=now, adjusted=adjusted,
                            sessions=sessions)
        if df is None:
            return pd.DataFrame()
        return df.iloc[-n:] if n is not None else df

    def history_arrays(self, symbol: str, n: int, freq: Optional[str] = None, adjusted: bool = False,
                       sessions: Sequence[str] = ("RTH",)) -> dict:
        """Fast PIT history: dict of READ-ONLY numpy arrays (start, end [int64 ns],
        open, high, low, close, volume, session [int8: 0 RTH, 1 PRE, 2 POST, 3 DAY])
        for the last ``n`` completed bars. Raw base-frequency bars are served as
        views on the engine cache (no DataFrame); anything else falls back to
        ``history`` and converts."""
        freq = canonical_freq(freq or self.__bt.base_freq)
        bt = self.__bt
        if not adjusted and freq == bt.base_freq and tuple(sessions) == tuple(bt.data_sessions):
            b = bt.md.bars(symbol, freq)
            i1 = b.last_index_at(int(self.now.value)) + 1
            i0 = i1 - n
            if i0 < 0 and not bt.md.may_be_truncated(b):
                i0 = 0
        if not adjusted and freq == bt.base_freq and tuple(sessions) == tuple(bt.data_sessions) and i0 >= 0:
            out = {"start": b.start[i0:i1], "end": b.end[i0:i1], "open": b.o[i0:i1], "high": b.h[i0:i1],
                   "low": b.l[i0:i1], "close": b.c[i0:i1], "volume": b.v[i0:i1], "session": b.sess[i0:i1]}
            for v in out.values():
                v.setflags(write=False)
            return out
        df = self.history(symbol, n, freq, adjusted=adjusted, sessions=sessions)
        from .timeutil import as_ns_array
        sess_map = {"RTH": 0, "PRE": 1, "POST": 2, "DAY": 3}
        out = {"start": as_ns_array(df.index) if len(df) else np.zeros(0, np.int64),
               "end": as_ns_array(df["known_at"]) if len(df) and "known_at" in df else np.zeros(0, np.int64)}
        for c in ("open", "high", "low", "close", "volume"):
            out[c] = df[c].to_numpy(float) if c in df else np.zeros(0)
        out["session"] = np.array([sess_map.get(str(x), 0) for x in df["session"]], np.int8) \
            if len(df) and "session" in df else np.zeros(len(df), np.int8)
        for v in out.values():
            v.setflags(write=False)
        return out

    def session_bars(self, symbol: str, freq: Optional[str] = None,
                     sessions: Sequence[str] = ("RTH",)) -> pd.DataFrame:
        freq = canonical_freq(freq or self.__bt.base_freq)
        now = self.now
        o, _ = self.__bt.session_open_close
        start = o
        if "PRE" in sessions:
            start = o - pd.Timedelta(hours=5, minutes=30)
        if freq == self.__bt.base_freq and tuple(sessions) == tuple(self.__bt.data_sessions):
            b = self.__bt.md.bars(symbol, freq)
            i1 = b.last_index_at(int(now.value)) + 1
            i0 = int(np.searchsorted(b.start, int(start.value), side="left"))
            return b.frame(i0, max(i0, i1))
        return self.__pit.bars(symbol, freq, start, now + pd.Timedelta(1, "ns"), as_of=now, adjusted=False,
                              sessions=sessions)

    def last_quote(self, symbol: str) -> Optional[Quote]:
        now = self.now
        df = self.__pit.quotes(symbol, now - pd.Timedelta(days=4), now + pd.Timedelta(1, "ns"), as_of=now)
        if df is None or len(df) == 0:
            return None
        r = df.iloc[-1]
        return Quote(symbol, df.index[-1], float(r["bid"]), float(r["ask"]),
                     float(r.get("bid_size", np.nan)), float(r.get("ask_size", np.nan)))

    # ------------------------------------------------------------------ orders
    def submit(self, order: Order, *, outside_rth: bool = False, initial_stop: Optional[float] = None,
               activate_at: Optional[pd.Timestamp] = None, planned_entry: Optional[float] = None) -> Order:
        return self.__bt.submit(order, outside_rth=outside_rth, initial_stop=initial_stop,
                                activate_at=activate_at, planned_entry=planned_entry)

    def order(self, symbol_or_inst: Union[str, Instrument], quantity: float, type: OrderType = OrderType.MARKET,
              limit_price: Optional[float] = None, stop_price: Optional[float] = None,
              tif: TimeInForce = TimeInForce.DAY, tag: str = "", oco_group: Optional[str] = None,
              **kw) -> Order:
        inst = symbol_or_inst if isinstance(symbol_or_inst, Instrument) else self.instrument(symbol_or_inst)
        return self.submit(Order(inst, quantity, type, limit_price, stop_price, tif, tag=tag, oco_group=oco_group), **kw)

    def bracket(self, symbol_or_inst: Union[str, Instrument], quantity: float, *,
                entry_type: OrderType = OrderType.MARKET, entry_limit: Optional[float] = None,
                entry_stop: Optional[float] = None, stop: Optional[float] = None,
                target: Optional[float] = None, time_exit: Optional[pd.Timestamp] = None,
                tif: TimeInForce = TimeInForce.DAY, child_tif: TimeInForce = TimeInForce.GTC,
                tag: str = "entry", outside_rth: bool = False,
                planned_entry: Optional[float] = None,
                stop_offset: Optional[float] = None, target_offset: Optional[float] = None,
                stop_fn=None, target_fn=None, time_exit_moc=None) -> Order:
        """Entry + protective stop + profit target + time exit (OCO children
        created as the entry fills). ``stop`` is also recorded as the initial
        risk for R-multiples.

        Fill-relative children: ``stop_offset`` / ``target_offset`` (price
        distances) or ``stop_fn`` / ``target_fn`` (callables ``f(fill_price,
        side) -> price``, side +1 long / -1 short) anchor the levels to the
        ACTUAL entry fill price; they become active right after the entry fill
        (rest of the entry bar under the adverse-first rule, or later bars), and
        the computed stop becomes the initial stop for R.

        ``time_exit_moc``: exit in the CLOSING AUCTION of a session (MOC child,
        reduce-only, OCO with the other children): pass a session date, or
        ``True`` to use the session of ``time_exit``. Use this on daily bars,
        where a market time exit could only fill at a later open."""
        inst = symbol_or_inst if isinstance(symbol_or_inst, Instrument) else self.instrument(symbol_or_inst)
        templates = []
        if stop_offset is not None and stop_fn is None:
            off = float(stop_offset)
            stop_fn = lambda px, side, _o=off: px - side * _o            # noqa: E731
        if target_offset is not None and target_fn is None:
            offt = float(target_offset)
            target_fn = lambda px, side, _o=offt: px + side * _o         # noqa: E731
        if stop is not None or stop_fn is not None:
            templates.append(ChildTemplate(OrderType.STOP, stop_price=stop, tag="stop", tif=child_tif,
                                           rel_fn=stop_fn))
        if target is not None or target_fn is not None:
            templates.append(ChildTemplate(OrderType.LIMIT, limit_price=target, tag="target", tif=child_tif,
                                           rel_fn=target_fn))
        if time_exit_moc is not None and time_exit_moc is not False:
            if time_exit_moc is True:
                if time_exit is None:
                    raise ValueError("time_exit_moc=True needs time_exit (or pass the session date)")
                from ..portfolio.corporate_actions import as_session_date
                sess = as_session_date(pd.Timestamp(time_exit))
            else:
                from ..portfolio.corporate_actions import as_session_date
                sess = as_session_date(pd.Timestamp(time_exit_moc))
            templates.append(ChildTemplate(OrderType.MOC, tag="time_exit", tif=TimeInForce.GTC,
                                           target_session=sess))
        elif time_exit is not None:
            templates.append(ChildTemplate(OrderType.MARKET, tag="time_exit", tif=TimeInForce.GTC,
                                           activate_at=pd.Timestamp(time_exit)))
        parent = Order(inst, quantity, entry_type, entry_limit, entry_stop, tif, tag=tag)
        return self.__bt.submit(parent, outside_rth=outside_rth, initial_stop=stop, children=templates,
                                planned_entry=planned_entry)

    def bracket_group(self, parent: Union[Order, int]) -> Optional[str]:
        """Public OCO group name of a bracket (children carry it as ``oco_group``)."""
        pid = parent.id if isinstance(parent, Order) else int(parent)
        spec = self.__bt.om.bracket_of(pid)
        return spec.oco_group if spec is not None else None

    def amend_bracket(self, parent_or_group: Union[Order, int, str], *, stop: Optional[float] = None,
                      target: Optional[float] = None, time_exit: Optional[pd.Timestamp] = None) -> list[Order]:
        """Add or replace the stop / target / time exit of an existing bracket.
        Working children are re-priced (cancel/replace semantics: effective from
        the next bar); missing legs are added (sized to the open exit
        quantity, reduce-only). Returns the affected child orders."""
        return self.__bt.amend_bracket(parent_or_group, stop=stop, target=target, time_exit=time_exit)

    def cancel(self, order_id: int) -> None:
        """Cancel a strategy order (engine-internal flat_by / margin-liquidation
        orders cannot be cancelled by strategy code)."""
        self.__bt.strategy_cancel(order_id)

    def cancel_all(self, symbol: Optional[str] = None) -> None:
        self.__bt.strategy_cancel_all(symbol)

    def flatten(self, symbol: Optional[str] = None, tag: str = "flatten") -> list[Order]:
        """Cancel open orders and submit market orders closing positions."""
        out = []
        syms = [symbol] if symbol else list(self.positions())
        for s in syms:
            self.cancel_all(s)
            q = self.position(s)
            if q:
                out.append(self.order(s, -q, tag=tag))
        return out

    # ------------------------------------------------------------------ bookkeeping
    def log(self, msg: str, **kv) -> None:
        self.__bt.logs.append({"ts": self.now, "msg": msg, **kv})

    def flag_fidelity(self, tier: Fidelity, reason: str) -> None:
        self.__bt.flag_fidelity(tier, reason)
