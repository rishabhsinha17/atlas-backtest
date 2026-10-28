"""Shared contracts for atlasbt.

Every module in the package builds against the types in this file. Change them
only via the integrator (see docs/ARCHITECTURE.md, "Contract changes").

Conventions (non-negotiable, they are what makes the engine faithful):
  * All timestamps are tz-aware pandas Timestamps in UTC. Use
    atlasbt.core.clock helpers to convert to/from America/New_York.
  * Bars are labelled by their START time; a bar's information becomes
    usable only at its END time (`known_at = start + freq`). Daily bars are
    known at the session's actual close (16:00 ET, 13:00 ET on early closes).
  * Every piece of information carries a knowledge time (`known_at`). The
    strategy context refuses to return anything whose known_at > ctx.now
    (raises LookaheadError). There is no "peek" escape hatch.
  * Prices inside the engine are RAW (as traded). Splits, dividends,
    delistings etc. are separate CorporateAction events applied by the
    portfolio. Indicator history is served adjusted *as of now* (only actions
    with ex_date <= now applied), so it never leaks future actions.
  * Quantities are signed: +long, -short. Option quantities are contracts.
  * Money is USD floats. Basis point = 1e-4.
  * Every fill and every data access carries a Fidelity tier so reports can
    say exactly how close to real life each result is.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class AtlasError(Exception):
    pass


class LookaheadError(AtlasError):
    """Raised when code asks for information not yet known at ctx.now."""


class DataUnavailableError(AtlasError):
    """Required data missing. Strategies must NOT silently substitute."""


class NotTestableError(AtlasError):
    """Strategy cannot be tested faithfully with the available data."""


# --------------------------------------------------------------------------
# Fidelity
# --------------------------------------------------------------------------
class Fidelity(enum.IntEnum):
    """Lower is better. Aggregated per backtest into a fidelity scorecard."""

    T0_QUOTE = 0      # executable NBBO/quote (or tick) at the true decision time
    T1_BAR = 1        # real bars at strategy resolution; next-bar fill + modeled spread
    T2_PROXY = 2      # real data but proxy timing/estimated input (e.g. close for 15:50
                      # quote, estimated spread, default borrow, approximated event time)
    T3_MODELED = 3    # synthetic/modeled data (e.g. options priced off a vol surface
                      # model, not real chain quotes)
    NA = 9            # cannot be produced -> strategy NOT_TESTABLE for that leg


# --------------------------------------------------------------------------
# Instruments
# --------------------------------------------------------------------------
class AssetClass(str, enum.Enum):
    STOCK = "STK"
    ETF = "ETF"
    INDEX = "IDX"        # not directly tradable (SPX, VIX); used for signals/settlement
    OPTION = "OPT"
    CASH = "CASH"


class Right(str, enum.Enum):
    CALL = "C"
    PUT = "P"


class ExerciseStyle(str, enum.Enum):
    AMERICAN = "AMERICAN"
    EUROPEAN = "EUROPEAN"


class SettlementType(str, enum.Enum):
    PHYSICAL = "PHYSICAL"   # equity/ETF options deliver shares
    CASH_AM = "CASH_AM"     # e.g. SPX standard monthly: special opening quotation (SOQ)
    CASH_PM = "CASH_PM"     # e.g. SPXW: official close


@dataclass(frozen=True)
class Instrument:
    """Immutable identity of a tradable/observable thing.

    `symbol` is the engine-wide key. For options use Instrument.option(...),
    which builds an OCC-style symbol, e.g. 'SPY   240119C00470000'.
    """

    symbol: str
    asset_class: AssetClass
    multiplier: float = 1.0
    currency: str = "USD"
    tick_size: float = 0.01                # min price increment (Reg NMS: $0.0001 below $1)
    # option-only fields
    underlying: Optional[str] = None
    expiry: Optional[pd.Timestamp] = None  # last trading day (date, ET); settlement per `settlement`
    strike: Optional[float] = None
    right: Optional[Right] = None
    exercise: Optional[ExerciseStyle] = None
    settlement: Optional[SettlementType] = None
    deliverable_shares: float = 100.0      # adjusted after splits/specials (non-standard deliverables)

    @property
    def is_option(self) -> bool:
        return self.asset_class == AssetClass.OPTION

    @staticmethod
    def option(underlying: str, expiry: pd.Timestamp, strike: float, right: Right,
               exercise: ExerciseStyle = ExerciseStyle.AMERICAN,
               settlement: SettlementType = SettlementType.PHYSICAL,
               multiplier: float = 100.0, root: Optional[str] = None,
               tick_size: float = 0.01) -> "Instrument":
        root = (root or underlying).upper()
        exp = pd.Timestamp(expiry).normalize()
        if exp.tzinfo is not None:
            exp = exp.tz_localize(None)
        sym = f"{root:<6}{exp:%y%m%d}{right.value}{int(round(strike * 1000)):08d}"
        return Instrument(symbol=sym, asset_class=AssetClass.OPTION, multiplier=multiplier,
                          underlying=underlying.upper(), expiry=exp, strike=float(strike),
                          right=right, exercise=exercise, settlement=settlement,
                          tick_size=tick_size)


# --------------------------------------------------------------------------
# Market data records (row-level views; bulk data lives in DataFrames, see
# docs/ARCHITECTURE.md "Canonical frames")
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Bar:
    symbol: str
    start: pd.Timestamp
    end: pd.Timestamp          # == known_at
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    trades: Optional[int] = None
    session: str = "RTH"       # 'PRE' | 'RTH' | 'POST' | 'DAY'
    fidelity: Fidelity = Fidelity.T1_BAR


@dataclass(frozen=True)
class Quote:
    symbol: str
    ts: pd.Timestamp           # == known_at
    bid: float
    ask: float
    bid_size: float = float("nan")
    ask_size: float = float("nan")
    fidelity: Fidelity = Fidelity.T0_QUOTE

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def is_locked_or_crossed(self) -> bool:
        return self.ask <= self.bid


@dataclass(frozen=True)
class OptionQuote:
    instrument: Instrument
    ts: pd.Timestamp
    bid: float
    ask: float
    underlying_price: float = float("nan")
    iv: float = float("nan")
    delta: float = float("nan")
    gamma: float = float("nan")
    vega: float = float("nan")
    theta: float = float("nan")
    volume: float = float("nan")
    open_interest: float = float("nan")
    bid_size: float = float("nan")
    ask_size: float = float("nan")
    fidelity: Fidelity = Fidelity.T0_QUOTE

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)


class ActionType(str, enum.Enum):
    SPLIT = "SPLIT"                    # ratio = new shares per old share (2-for-1 -> 2.0)
    CASH_DIVIDEND = "CASH_DIVIDEND"    # amount per share, pay_date
    SPECIAL_DIVIDEND = "SPECIAL_DIVIDEND"
    DELIST = "DELIST"                  # position closed at `amount` per share (delisting value)
    SYMBOL_CHANGE = "SYMBOL_CHANGE"    # new_symbol
    SPINOFF = "SPINOFF"                # new_symbol, ratio, amount (cost basis allocation)


@dataclass(frozen=True)
class CorporateAction:
    symbol: str
    type: ActionType
    ex_date: pd.Timestamp              # session date: NAIVE midnight of the ET date the action takes effect
    known_at: pd.Timestamp             # declaration/announcement time (PIT)
    ratio: float = 1.0
    amount: float = 0.0
    pay_date: Optional[pd.Timestamp] = None
    new_symbol: Optional[str] = None
    fidelity: Fidelity = Fidelity.T1_BAR


@dataclass(frozen=True)
class MarketEvent:
    """Generic timestamped non-price information: earnings, index changes,
    FOMC statements, buybacks, halts... `payload` keys are documented per kind in
    docs/ARCHITECTURE.md."""

    kind: str                          # 'EARNINGS' | 'INDEX_ADD' | 'INDEX_DEL' | 'FOMC' | 'BUYBACK' | 'HALT' | ...
    symbol: Optional[str]
    known_at: pd.Timestamp             # when the market could know it (release timestamp)
    effective_at: Optional[pd.Timestamp] = None
    payload: dict = field(default_factory=dict)
    fidelity: Fidelity = Fidelity.T1_BAR


# --------------------------------------------------------------------------
# Orders & fills
# --------------------------------------------------------------------------
class Side(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, enum.Enum):
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"                # stop-market; gap-aware
    STOP_LIMIT = "STP_LMT"
    MOO = "MOO"                 # market-on-open (opening auction)
    MOC = "MOC"                 # market-on-close (closing auction)
    COMBO = "COMBO"             # multi-leg (options/stock) with net limit or market


class TimeInForce(str, enum.Enum):
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"


class OrderStatus(str, enum.Enum):
    PENDING = "PENDING"         # submitted, not yet eligible (latency)
    WORKING = "WORKING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class Leg:
    instrument: Instrument
    ratio: int                  # signed: +1 buy, -1 sell, -2 sell two ... per 1 combo unit


@dataclass
class Order:
    instrument: Optional[Instrument]   # None for COMBO (see legs)
    quantity: float                    # signed (+ buy / - sell); for COMBO: combo units (+ = buy the package)
    type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None    # COMBO: net price per unit (+ debit / - credit)
    stop_price: Optional[float] = None
    tif: TimeInForce = TimeInForce.DAY
    legs: tuple[Leg, ...] = ()
    tag: str = ""                      # free text: 'entry', 'stop', 'target', 'time_exit'...
    oco_group: Optional[str] = None    # orders sharing a group cancel each other on fill
    is_short_sale: bool = False        # set by engine: sell that opens/increases a short
    outside_rth: bool = False          # may fill in PRE/POST sessions (needs extended-hours data)
    meta: dict = field(default_factory=dict)   # e.g. initial_stop, activate_at, sizing info
    # engine-managed
    id: int = -1
    strategy_id: str = ""
    submitted_at: Optional[pd.Timestamp] = None
    eligible_at: Optional[pd.Timestamp] = None   # submitted_at + latency
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    reject_reason: str = ""

    @property
    def side(self) -> Side:
        return Side.BUY if self.quantity > 0 else Side.SELL

    @property
    def remaining(self) -> float:
        return self.quantity - self.filled_qty


@dataclass(frozen=True)
class CostBreakdown:
    """All in USD, all >= 0 (costs), per fill."""

    commission: float = 0.0
    exchange_fees: float = 0.0        # exchange/clearing (incl. OCC, ORF for options)
    regulatory_fees: float = 0.0      # SEC Section 31, FINRA TAF
    spread_cost: float = 0.0          # vs mid, informational (already in price)
    impact_cost: float = 0.0          # vs mid-after-spread, informational (already in price)

    @property
    def explicit(self) -> float:
        """Fees actually debited from cash on top of the fill price."""
        return self.commission + self.exchange_fees + self.regulatory_fees

    @property
    def total(self) -> float:
        return self.explicit + self.spread_cost + self.impact_cost


@dataclass(frozen=True)
class Fill:
    order_id: int
    instrument: Instrument
    ts: pd.Timestamp
    quantity: float                   # signed
    price: float                      # execution price per unit (per share / per option premium)
    costs: CostBreakdown
    fidelity: Fidelity
    reference_mid: float = float("nan")   # mid at fill time if known, for slippage analytics
    liquidity: str = "TAKER"          # 'TAKER' | 'MAKER' | 'AUCTION' | 'EXERCISE' | 'ASSIGNMENT' | 'SETTLEMENT' | 'CORP_ACTION'
    note: str = ""


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
class RunStatus(str, enum.Enum):
    OK = "OK"
    NOT_TESTABLE = "NOT_TESTABLE"
    FAILED = "FAILED"


@dataclass
class BacktestResult:
    strategy_id: str
    params: dict
    status: RunStatus
    equity: pd.Series                 # NAV indexed by session close timestamps (UTC)
    returns: pd.Series                # daily simple returns of NAV
    fills: pd.DataFrame               # one row per Fill
    trades: pd.DataFrame              # round-trip trades (entry/exit, pnl, R multiple, mae/mfe)
    positions: pd.DataFrame           # end-of-day positions snapshot (long format)
    ledger: pd.DataFrame              # cash ledger: every cash movement with reason
    fidelity: dict                    # scorecard: counts/notional by Fidelity tier & reasons
    diagnostics: dict = field(default_factory=dict)   # warnings, rejected orders, margin calls...
    not_testable_reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
