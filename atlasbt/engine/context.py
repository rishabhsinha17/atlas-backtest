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


class EngineContext:
    def __init__(self, bt: "Backtester"):
        self._bt = bt
        self.strategy_id = bt.strategy.id
        self._pit = PITPortal(bt.portal, lambda: self.now, on_access=bt._note_data_access)
        self.state: dict[str, Any] = {}        # free scratch space for strategies

    # ------------------------------------------------------------------ time
    @property
    def now(self) -> pd.Timestamp:
        return self._bt.now

    def now_et(self) -> pd.Timestamp:
        return to_et(self.now)

    def session_date(self) -> pd.Timestamp:
        return self._bt.session

    def minutes_since_open(self) -> float:
        o, _ = self._bt.session_open_close
        return (self.now - o).total_seconds() / 60.0

    def minutes_to_close(self) -> float:
        _, c = self._bt.session_open_close
        return (c - self.now).total_seconds() / 60.0

    def is_session_close_bar(self) -> bool:
        _, c = self._bt.session_open_close
        return self.now >= c

    def calendar(self):
        return self._bt.cal

    # ------------------------------------------------------------------ account
    @property
    def nav(self) -> float:
        return self._bt.account.nav

    @property
    def prior_nav(self) -> float:
        return self._bt.prior_nav

    @property
    def cash(self) -> float:
        return self._bt.account.cash

    def position(self, symbol: str) -> float:
        return self._bt.account.position(symbol)

    def positions(self) -> dict[str, float]:
        return self._bt.account.qty_map()

    def open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        return self._bt.om.active(symbol)

    def mark(self, symbol: str) -> float:
        """Latest known price (last completed bar close / position mark)."""
        return self._bt.last_price(symbol)

    def account(self):
        """Read-only view of the account (positions with avg price/marks)."""
        return self._bt.account

    # ------------------------------------------------------------------ data
    def portal(self) -> PITPortal:
        return self._pit

    def instrument(self, symbol: str) -> Instrument:
        return self._bt.instrument(symbol)

    def history(self, symbol: str, n: int, freq: Optional[str] = None, adjusted: bool = True,
                sessions: Sequence[str] = ("RTH",)) -> pd.DataFrame:
        """Last ``n`` COMPLETED bars (known_at <= now). Daily bars of the
        current session are never included before the close."""
        freq = canonical_freq(freq or self._bt.base_freq)
        now = self.now
        if not adjusted and freq == self._bt.base_freq and tuple(sessions) == tuple(self._bt.data_sessions):
            b = self._bt.md.bars(symbol, freq)
            i1 = b.last_index_at(int(now.value)) + 1
            i0 = max(i1 - n, 0)
            return b.frame(i0, i1)
        cal = self._bt.cal
        if is_daily(freq):
            sess = self._bt.session
            back = cal.prev_session(sess, min(n + 5, max(cal.session_index(sess), 1)))
            start = cal.open_close(back)[0] - pd.Timedelta(days=1)
        else:
            td = freq_to_timedelta(freq)
            bars_per_day = max(int(6.5 * 3600 / td.total_seconds()), 1)
            ext = 16 * 3600 / td.total_seconds() if any(s != "RTH" for s in sessions) else bars_per_day
            days = int(np.ceil(n / max(ext, 1))) + 3
            start = now - pd.Timedelta(days=int(days * 7 / 5) + 4)
        df = self._pit.bars(symbol, freq, start, now + pd.Timedelta(1, "ns"), as_of=now, adjusted=adjusted,
                            sessions=sessions)
        if df is None:
            return pd.DataFrame()
        return df.iloc[-n:] if n is not None else df

    def session_bars(self, symbol: str, freq: Optional[str] = None,
                     sessions: Sequence[str] = ("RTH",)) -> pd.DataFrame:
        freq = canonical_freq(freq or self._bt.base_freq)
        now = self.now
        o, _ = self._bt.session_open_close
        start = o
        if "PRE" in sessions:
            start = o - pd.Timedelta(hours=5, minutes=30)
        if freq == self._bt.base_freq and tuple(sessions) == tuple(self._bt.data_sessions):
            b = self._bt.md.bars(symbol, freq)
            i1 = b.last_index_at(int(now.value)) + 1
            i0 = int(np.searchsorted(b.start, int(start.value), side="left"))
            return b.frame(i0, max(i0, i1))
        return self._pit.bars(symbol, freq, start, now + pd.Timedelta(1, "ns"), as_of=now, adjusted=False,
                              sessions=sessions)

    def last_quote(self, symbol: str) -> Optional[Quote]:
        now = self.now
        df = self._pit.quotes(symbol, now - pd.Timedelta(days=4), now + pd.Timedelta(1, "ns"), as_of=now)
        if df is None or len(df) == 0:
            return None
        r = df.iloc[-1]
        return Quote(symbol, df.index[-1], float(r["bid"]), float(r["ask"]),
                     float(r.get("bid_size", np.nan)), float(r.get("ask_size", np.nan)))

    # ------------------------------------------------------------------ orders
    def submit(self, order: Order, *, outside_rth: bool = False, initial_stop: Optional[float] = None,
               activate_at: Optional[pd.Timestamp] = None) -> Order:
        return self._bt.submit(order, outside_rth=outside_rth, initial_stop=initial_stop,
                               activate_at=activate_at)

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
                tag: str = "entry", outside_rth: bool = False) -> Order:
        """Entry + protective stop + profit target + time exit (OCO children
        created as the entry fills). ``stop`` is also recorded as the initial
        risk for R-multiples."""
        inst = symbol_or_inst if isinstance(symbol_or_inst, Instrument) else self.instrument(symbol_or_inst)
        templates = []
        if stop is not None:
            templates.append(ChildTemplate(OrderType.STOP, stop_price=stop, tag="stop", tif=child_tif))
        if target is not None:
            templates.append(ChildTemplate(OrderType.LIMIT, limit_price=target, tag="target", tif=child_tif))
        if time_exit is not None:
            templates.append(ChildTemplate(OrderType.MARKET, tag="time_exit", tif=TimeInForce.GTC,
                                           activate_at=pd.Timestamp(time_exit)))
        parent = Order(inst, quantity, entry_type, entry_limit, entry_stop, tif, tag=tag)
        return self._bt.submit(parent, outside_rth=outside_rth, initial_stop=stop, children=templates)

    def cancel(self, order_id: int) -> None:
        self._bt.om.cancel(order_id, self.now, "cancelled by strategy")

    def cancel_all(self, symbol: Optional[str] = None) -> None:
        self._bt.om.cancel_all(self.now, symbol, "cancel_all by strategy")

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
        self._bt.logs.append({"ts": self.now, "msg": msg, **kv})

    def flag_fidelity(self, tier: Fidelity, reason: str) -> None:
        self._bt.flag_fidelity(tier, reason)
