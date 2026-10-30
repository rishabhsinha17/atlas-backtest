"""Interfaces between subsystems. Implementations live in their owning packages
(see docs/ARCHITECTURE.md "Ownership"). Keep these signatures stable.
"""
from __future__ import annotations

from typing import Iterable, Optional, Protocol, Sequence, runtime_checkable

import pandas as pd

from .types import (
    Bar, CorporateAction, Fidelity, Fill, Instrument, MarketEvent, OptionQuote,
    Order, Quote,
)


# --------------------------------------------------------------------------
# Data layer  (owner: atlasbt.data)
# --------------------------------------------------------------------------
@runtime_checkable
class DataPortal(Protocol):
    """Point-in-time access to everything. Every method takes `as_of` and must
    never return information with known_at > as_of. The strategy context wraps
    this with as_of=ctx.now; the engine uses it for fills and accounting.

    Frames follow docs/ARCHITECTURE.md "Canonical frames".
    """

    # ---- reference
    def instrument(self, symbol: str) -> Instrument: ...
    def security_master(self, as_of: pd.Timestamp) -> pd.DataFrame: ...
    def universe(self, name: str, as_of: pd.Timestamp) -> list[str]:
        """PIT members as the ticker used AT as_of (e.g. 'FISV', 'UTX')."""
        ...
    def security_id(self, symbol: str, as_of: pd.Timestamp) -> Optional[str]:
        """Stable identity (CIK-based) of the security that `symbol` denoted at as_of."""
        ...

    # ---- prices
    def bars(self, symbol: str, freq: str, start: pd.Timestamp, end: pd.Timestamp,
             as_of: pd.Timestamp, adjusted: bool = False,
             sessions: Sequence[str] = ("RTH",)) -> pd.DataFrame:
        """Bars with start in [start, end) AND known_at <= as_of.
        adjusted=True -> split & dividend adjusted using only actions with
        ex_date <= as_of (total-return style back-adjustment). Raw otherwise.
        FAIL-CLOSED: raises DataUnavailableError for sessions whose security identity
        cannot be verified; attrs['identity_log'] documents accepted/rejected segments."""
        ...

    def quotes(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp,
               as_of: pd.Timestamp) -> pd.DataFrame: ...

    def has(self, symbol: str, kind: str, freq: Optional[str] = None) -> bool:
        """kind in {'bars','quotes','options','actions','borrow','events','fundamentals'}.
        Portal-wide kinds use symbol='*': has('*','events','EARNINGS'),
        has('*','fundamentals','Assets') (freq carries the event kind / field)."""
        ...

    def fidelity_of(self, symbol: str, kind: str, freq: Optional[str] = None) -> Fidelity: ...

    # ---- corporate actions & events
    def corporate_actions(self, symbols: Optional[Iterable[str]], start: pd.Timestamp,
                          end: pd.Timestamp, as_of: pd.Timestamp) -> list[CorporateAction]: ...
    def events(self, kind: str, symbols: Optional[Iterable[str]], start: pd.Timestamp,
               end: pd.Timestamp, as_of: pd.Timestamp) -> list[MarketEvent]: ...

    # ---- options
    def option_chain(self, underlying: str, as_of: pd.Timestamp,
                     min_dte: int = 0, max_dte: int = 400) -> pd.DataFrame:
        """Latest available chain snapshot with ts <= as_of. Columns per
        "Canonical frames/option_chain". Includes a 'fidelity' column."""
        ...

    def option_quote(self, instrument: Instrument, as_of: pd.Timestamp) -> Optional[OptionQuote]: ...

    def option_quotes(self, instrument: Instrument, start: pd.Timestamp, end: pd.Timestamp,
                      as_of: pd.Timestamp) -> pd.DataFrame:
        """All snapshots with ts in (start, end] and ts <= as_of, ascending; columns as
        option_chain (bid, ask, iv, delta, underlying_price, fidelity...), index ts.
        Used by fill models to take the FIRST snapshot after eligible_at."""
        ...

    # ---- rates, borrow, fundamentals
    def risk_free_rate(self, as_of: pd.Timestamp, tenor: str = "3M") -> float:
        """Annualized simple rate (decimal) known at as_of."""
        ...

    def borrow(self, symbol: str, as_of: pd.Timestamp) -> tuple[bool, float, Fidelity]:
        """(shortable/locate available, annualized borrow fee decimal, fidelity)."""
        ...

    def risk_free_rate_info(self, as_of: pd.Timestamp, tenor: str = "3M") -> tuple:
        """(rate, series_used, fidelity, observation_date); fidelity T2 when a fallback series is used."""
        ...

    def fundamentals_history(self, symbols: Optional[Iterable[str]], fields: Sequence[str],
                             as_of: pd.Timestamp, start: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Every filed value (incl. restatements/comparatives) with known_at <= as_of."""
        ...

    def fundamentals(self, symbols: Optional[Iterable[str]], fields: Sequence[str],
                     as_of: pd.Timestamp) -> pd.DataFrame:
        """Latest values per (symbol, field) whose filing/known_at <= as_of."""
        ...


# --------------------------------------------------------------------------
# Execution  (owner: atlasbt.execution)
# --------------------------------------------------------------------------
@runtime_checkable
class FillModel(Protocol):
    """Decides whether/how a WORKING order fills against *new* market data
    that arrived strictly after order.eligible_at. Must never use data
    timestamped before eligible_at (no filling at a past price) and never data
    after the current engine time."""

    def try_fill(self, order: Order, portal: DataPortal, window_start: pd.Timestamp,
                 window_end: pd.Timestamp, state: "ExecutionState") -> list[Fill]: ...


class ExecutionState(Protocol):
    """What fill/cost models may know about the account at fill time."""

    def position(self, symbol: str) -> float: ...
    def is_ssr_active(self, symbol: str, ts: pd.Timestamp) -> bool: ...
    def adv(self, symbol: str, as_of: pd.Timestamp) -> float: ...           # shares, 20d
    def daily_vol(self, symbol: str, as_of: pd.Timestamp) -> float: ...     # 20d stdev of returns


@runtime_checkable
class CostModel(Protocol):
    def explicit_costs(self, instrument: Instrument, quantity: float, price: float,
                       ts: pd.Timestamp, liquidity: str,
                       order_id: Optional[int] = None) -> "CostBreakdownLike": ...


CostBreakdownLike = object  # atlasbt.core.types.CostBreakdown


# --------------------------------------------------------------------------
# Strategy  (owner: atlasbt.engine defines Context; atlasbt.strategies implements)
# --------------------------------------------------------------------------
class Context(Protocol):
    """Everything a strategy may see/do. All reads are PIT at self.now."""

    now: pd.Timestamp                 # UTC; the END of the bar that just completed
    strategy_id: str

    # time
    def now_et(self) -> pd.Timestamp: ...
    def session_date(self) -> pd.Timestamp: ...          # ET session date (naive midnight)
    def minutes_since_open(self) -> float: ...
    def minutes_to_close(self) -> float: ...
    def is_session_close_bar(self) -> bool: ...
    def calendar(self) -> "TradingCalendar": ...

    # account
    @property
    def nav(self) -> float: ...
    @property
    def prior_nav(self) -> float: ...                    # NAV at previous session close
    @property
    def cash(self) -> float: ...
    def position(self, symbol: str) -> float: ...
    def positions(self) -> dict[str, float]: ...
    def mark(self, symbol: str) -> float: ...            # last mark known at now
    def instrument(self, symbol: str) -> Instrument: ...
    def open_orders(self, symbol: Optional[str] = None) -> list[Order]: ...

    # data (PIT; raise LookaheadError on violation)
    def history(self, symbol: str, n: int, freq: str, adjusted: bool = True,
                sessions: Sequence[str] = ("RTH",)) -> pd.DataFrame: ...
    def session_bars(self, symbol: str, freq: str, sessions: Sequence[str] = ("RTH",)) -> pd.DataFrame:
        """Today's bars so far (completed only)."""
        ...
    def last_quote(self, symbol: str) -> Optional[Quote]: ...
    def portal(self) -> DataPortal: ...                  # PIT-wrapped portal (as_of=now)

    # orders
    def submit(self, order: Order) -> Order: ...
    def bracket(self, entry: Order, stop: Optional[float] = None, target: Optional[float] = None,
                time_exit: Optional[pd.Timestamp] = None, tag: str = "") -> list[Order]: ...
    def cancel(self, order_id: int) -> None: ...
    def cancel_all(self, symbol: Optional[str] = None) -> None: ...

    # bookkeeping for reports
    def log(self, msg: str, **kv) -> None: ...
    def flag_fidelity(self, tier: Fidelity, reason: str) -> None: ...


class TradingCalendar(Protocol):
    def sessions(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex: ...   # naive ET dates
    def open_close(self, session: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]: ...  # UTC
    def is_session(self, date: pd.Timestamp) -> bool: ...
    def next_session(self, date: pd.Timestamp, n: int = 1) -> pd.Timestamp: ...
    def prev_session(self, date: pd.Timestamp, n: int = 1) -> pd.Timestamp: ...
    def is_early_close(self, session: pd.Timestamp) -> bool: ...
    def is_last_session_of_month(self, session: pd.Timestamp) -> bool: ...
    def sessions_in_month(self, session: pd.Timestamp) -> pd.DatetimeIndex: ...
    def holidays(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex: ...   # full-day closures on weekdays
