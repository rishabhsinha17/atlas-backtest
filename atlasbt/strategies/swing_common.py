"""Shared machinery for the swing family (S046-S084, daily bars).

Contents
--------
* Default universes (STRATEGY_GUIDE §3) used by the swing strategies.
* :class:`BarCache` - per-symbol, point-in-time, split/dividend-adjusted daily
  history kept incrementally (full refetch only when a corporate action with an
  ex-date after the cached data becomes effective), so that S&P 500-wide daily
  strategies do not re-query 300+ bars per name per day. Every read goes
  through ``ctx.history`` (PIT-guarded); nothing here can see a bar before its
  ``known_at``.
* :class:`SwingBase` - the position engine shared by every single-name swing
  strategy: signal at the daily close -> entry order for the next session
  (market at the open, or a resting buy/sell stop), protective price-level
  stops as resting STOP orders re-placed at each close (gap-aware fills by the
  engine), close-based exits as market orders filling at the next open, R
  targets as resting limits, time exits, optional Turtle-style pyramiding.
* :class:`CrossSectionBase` - periodic long/short rebalancing to
  ``risk.sizing.cross_section_weights`` targets.

Conventions that apply to every swing strategy are listed in
``COMMON_DEVIATIONS`` and copied into each class's ``deviations``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..core.types import ActionType, Fidelity, NotTestableError, OrderStatus, OrderType, TimeInForce
from ..engine.strategy import Requirement, Strategy
from ..indicators import atr as _atr
from ..risk import sizing as _sz

# ============================================================== universes (guide §3)
ETF_CORE = ["SPY", "QQQ", "IWM", "DIA"]
SECTOR_SPDR = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]
SWING_ETF = ETF_CORE + SECTOR_SPDR + ["TLT", "GLD", "EFA", "EEM"]
SP500_PIT = "SP500"          # portal universe name (point in time)

# Sector -> Select Sector SPDR with inception dates (XLRE 2015-10-08, XLC 2018-06-19)
SECTOR_ETF = {
    "Materials": ("XLB", "1998-12-22"), "Energy": ("XLE", "1998-12-22"), "Financials": ("XLF", "1998-12-22"),
    "Industrials": ("XLI", "1998-12-22"), "Information Technology": ("XLK", "1998-12-22"),
    "Consumer Staples": ("XLP", "1998-12-22"), "Utilities": ("XLU", "1998-12-22"),
    "Health Care": ("XLV", "1998-12-22"), "Consumer Discretionary": ("XLY", "1998-12-22"),
    "Real Estate": ("XLRE", "2015-10-08"), "Communication Services": ("XLC", "2018-06-19"),
}
_SECTOR_ALIASES = {"Technology": "Information Technology", "Telecommunication Services": "Communication Services",
                   "Telecommunications": "Communication Services", "Healthcare": "Health Care",
                   "Consumer Cyclical": "Consumer Discretionary", "Consumer Defensive": "Consumer Staples",
                   "Basic Materials": "Materials", "Financial Services": "Financials"}
SECTOR_COLUMNS = ("sector", "gics_sector", "sector_sic", "sector_gics")

COMMON_DEVIATIONS = [
    "Universe (guide §3): 'STK/ETF' and 'ETF' swing rows trade SWING_ETF (SPY QQQ IWM DIA + 11 sector SPDRs + "
    "TLT GLD EFA EEM; each symbol only once it has bars), as in the guide's S046 example; 'STK' rows trade the "
    "point-in-time portal universe 'SP500'.",
    "Signals use split+dividend adjusted daily history as of the decision close; resting order prices are raw "
    "(current) prices. Resting stops/targets cancelled by the engine on a split are re-placed at the next session "
    "start with split-adjusted prices/quantities; cash dividends do not adjust resting stops.",
    "Entry-day protection: the engine's bracket children have fixed prices, so stops/targets defined relative to "
    "the actual fill (e.g. 'k*ATR from fill', 'nR') are placed with the bracket at levels computed from the planned "
    "entry (signal close, or the entry-stop price) and replaced by the exact fill-based levels at the entry-day "
    "close; from then on they are resting orders re-placed at every close.",
    "Close-based exits ('exit close<X', 'exit after N sessions') are market orders submitted at the signal close and "
    "filled at the next open; they share an OCO group with the resting protective stop so only one can fill.",
    "Holding period: the entry session counts as session 1; 'exit after N sessions'/'max hold N' submits the exit "
    "at the close of session N (fill at the next open).",
    "'k*ATR' with no stated period uses Wilder ATR14; 'trailing k*ATR stop' = max over closes since entry of "
    "(C_t - k*ATR_t) for longs (min of C_t + k*ATR_t for shorts), ratcheting only, initial level = fill - k*ATR "
    "at the signal.",
    "'SMA200 positive/up/rising' = SMA200_t > SMA200_{t-20} (guide rule 9); 'X crosses above Y' = "
    "X_{t-1} <= Y_{t-1} and X_t > Y_t (the S050 sheet form; thresholds treated as constant Y).",
    "'prior N-day high/low' = max(H)/min(L) over t-N..t-1 (indicators.donchian default).",
    "Recursive indicators (EMA, Wilder RSI/ATR/ADX, SAR, Supertrend) are evaluated on a trailing window of at "
    "least 400 sessions (seed influence < 1e-6); path-dependent state is identical once a flip occurred inside "
    "the window.",
    "When several entries compete for the aggregate gross cap on the same close they are processed in alphabetical "
    "symbol order; pending entry orders count toward the cap.",
]

SECTOR_CAVEAT = ("Sectors come from the portal security master (column 'sector'/'gics_sector'), re-read every 21 "
                 "sessions; the portal documents them as current (SIC-derived or current GICS) classifications, "
                 "not point-in-time - a data caveat for sector caps/neutrality.")

_PRICE_ACTIONS = (ActionType.SPLIT, ActionType.CASH_DIVIDEND, ActionType.SPECIAL_DIVIDEND, ActionType.SPINOFF)


def tick_for(price: float) -> float:
    """Rule 612 minimum increment."""
    return 0.01 if price >= 1.0 else 0.0001


def crossed_above(x_prev: float, x: float, y_prev: float, y: float) -> bool:
    return bool(np.isfinite(x_prev) and np.isfinite(x) and np.isfinite(y_prev) and np.isfinite(y)
                and x_prev <= y_prev and x > y)


def crossed_below(x_prev: float, x: float, y_prev: float, y: float) -> bool:
    return bool(np.isfinite(x_prev) and np.isfinite(x) and np.isfinite(y_prev) and np.isfinite(y)
                and x_prev >= y_prev and x < y)


def finite(*xs) -> bool:
    return all(x is not None and np.isfinite(x) for x in xs)


def normalize_sector(s) -> Optional[str]:
    if s is None or (isinstance(s, float) and not np.isfinite(s)):
        return None
    s = str(s).strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    return _SECTOR_ALIASES.get(s, s)


def sector_etf_for(sector: Optional[str], session) -> Optional[str]:
    """Sector SPDR for ``sector`` if it was trading on ``session`` (else None)."""
    sector = normalize_sector(sector)
    if sector not in SECTOR_ETF:
        return None
    etf, first = SECTOR_ETF[sector]
    d = pd.Timestamp(session)
    if d.tzinfo is not None:
        d = d.tz_convert("America/New_York").tz_localize(None)
    return etf if d.normalize() >= pd.Timestamp(first) else None


def sectors_from_master(ctx, symbols) -> dict[str, str]:
    """{symbol: sector} from the point-in-time security master ({} if it has no sector column)."""
    try:
        sm = ctx.portal().security_master()
    except Exception:
        return {}
    if sm is None or len(sm) == 0:
        return {}
    col = next((c for c in SECTOR_COLUMNS if c in sm.columns), None)
    if col is None or "symbol" not in sm.columns:
        return {}
    want = None if symbols is None else set(symbols)
    out = {}
    for s, v in zip(sm["symbol"].to_numpy(), sm[col].to_numpy()):
        if want is None or s in want:
            v2 = normalize_sector(v)
            if v2 is not None:
                out[s] = v2
    return out


# ============================================================== bar cache
class Bars:
    """Numpy view of daily bars (adjusted as of the decision time)."""

    __slots__ = ("o", "h", "l", "c", "v", "start", "end")

    def __init__(self, o, h, l, c, v, start, end):
        self.o, self.h, self.l, self.c, self.v, self.start, self.end = o, h, l, c, v, start, end

    def __len__(self) -> int:
        return self.c.shape[0]

    def tail(self, n: Optional[int]) -> "Bars":
        if n is None or n >= len(self):
            return self
        s = slice(len(self) - n, None)
        return Bars(self.o[s], self.h[s], self.l[s], self.c[s], self.v[s], self.start[s], self.end[s])

    def series(self, name: str) -> pd.Series:
        return pd.Series(getattr(self, name), index=pd.DatetimeIndex(self.start, tz="UTC"))


def _frame_arrays(df: pd.DataFrame):
    start = np.asarray(pd.DatetimeIndex(df.index).as_unit("ns").asi8, dtype=np.int64)
    if "known_at" in df.columns:
        ka = df["known_at"].array
        end = np.asarray(ka.as_unit("ns").asi8 if hasattr(ka, "asi8") else
                         pd.DatetimeIndex(df["known_at"]).as_unit("ns").asi8, dtype=np.int64)
    else:
        end = start.copy()
    o, h, l, c = (np.array(df[k].to_numpy(), dtype=float) for k in ("open", "high", "low", "close"))
    v = np.array(df["volume"].to_numpy(), dtype=float) if "volume" in df.columns else np.full(len(df), np.nan)
    return start, end, [o, h, l, c], v


class _Entry:
    __slots__ = ("start", "end", "o", "h", "l", "c", "v", "ver")


class BarCache:
    """Incremental point-in-time adjusted daily history per symbol.

    ``get(ctx, sym, n)`` returns the last ``n`` completed daily bars adjusted
    as of ``ctx.now``. The first call fetches ``max(n, fetch)`` adjusted bars;
    later calls append the newest raw bars (identical to adjusted values for
    the latest bars since no action with a later ex-date is applied) unless a
    split/dividend with an ex-date after the cached data has become effective,
    in which case the symbol is refetched."""

    def __init__(self, keep: Optional[int] = 1000, fetch: Optional[int] = None):
        self.keep = keep
        self.fetch = fetch
        self._d: dict[str, _Entry] = {}
        self._ex: dict[str, int] = {}
        self._seen: set = set()
        self._checked: Optional[pd.Timestamp] = None
        self._ver = 0

    def stale(self, sym: str) -> bool:
        """True if an adjustment with an ex-date after the cached data has become known."""
        e = self._d.get(sym)
        return e is None or self._ex.get(sym, -(2 ** 62)) > e.start[-1]

    def version(self, sym: str) -> int:
        """Changes whenever the symbol's history was (re)fetched in full (adjustment change)."""
        e = self._d.get(sym)
        return -1 if e is None else e.ver

    def refresh_actions(self, ctx) -> None:
        today = ctx.session_date()
        if self._checked is not None and self._checked == today:
            return
        start = today - pd.Timedelta(days=10)
        syms = sorted(self._d)                 # only cached histories can become stale
        if not syms:
            return
        acts = ctx.portal().corporate_actions(syms, start, today)
        for a in acts or []:
            if a.type not in _PRICE_ACTIONS:
                continue
            ex = pd.Timestamp(a.ex_date)
            if ex.tzinfo is not None:
                ex = ex.tz_convert("America/New_York").tz_localize(None)
            ex = ex.normalize()
            key = (a.symbol, a.type, ex, a.ratio, a.amount)
            if key in self._seen:
                continue
            self._seen.add(key)
            # bars that START before the ex-date session change on adjustment
            ex_open = ctx.calendar().open_close(ex)[0] if ctx.calendar().is_session(ex) else \
                pd.Timestamp(ex).tz_localize("America/New_York").tz_convert("UTC")
            v = int(ex_open.value)
            if v > self._ex.get(a.symbol, -(2 ** 62)):
                self._ex[a.symbol] = v
        self._checked = today

    def _full(self, ctx, sym: str, n: int) -> Optional[_Entry]:
        k = max(n, self.fetch or 0)
        df = ctx.history(sym, k, "1d", adjusted=True)
        if df is None or len(df) == 0:
            self._d.pop(sym, None)
            return None
        e = _Entry()
        e.start, e.end, (e.o, e.h, e.l, e.c), e.v = _frame_arrays(df)
        self._ver += 1
        e.ver = self._ver
        self._d[sym] = e
        return e

    def get(self, ctx, sym: str, n: Optional[int]) -> Optional[Bars]:
        need = n if n is not None else 100_000
        e = self._d.get(sym)
        if e is not None and self._ex.get(sym, -(2 ** 62)) > e.start[-1]:
            e = None                       # an adjustment now applies to cached bars
        if e is None:
            e = self._full(ctx, sym, need)
            if e is None:
                return None
        else:
            now_ns = int(ctx.now.value)
            if e.end[-1] < now_ns:
                df = ctx.history(sym, 3, "1d", adjusted=False)
                if df is not None and len(df):
                    st, en, (o, h, l, c), v = _frame_arrays(df)
                    m = st > e.start[-1]
                    if m.all() and len(st) >= 3:
                        e = self._full(ctx, sym, need)       # gap: refetch
                        if e is None:
                            return None
                    elif m.any():
                        e.start = np.concatenate([e.start, st[m]]); e.end = np.concatenate([e.end, en[m]])
                        e.o = np.concatenate([e.o, o[m]]); e.h = np.concatenate([e.h, h[m]])
                        e.l = np.concatenate([e.l, l[m]]); e.c = np.concatenate([e.c, c[m]])
                        e.v = np.concatenate([e.v, v[m]])
                        if self.keep is not None and len(e.start) > self.keep + 200:
                            k = self.keep
                            e.start, e.end, e.o, e.h, e.l, e.c, e.v = (x[-k:] for x in
                                                                       (e.start, e.end, e.o, e.h, e.l, e.c, e.v))
        b = Bars(e.o, e.h, e.l, e.c, e.v, e.start, e.end)
        return b.tail(n)


# ============================================================== swing position engine
@dataclass
class Plan:
    """An entry decided at a close."""

    side: int                                  # +1 long / -1 short
    ref_price: float                           # planned entry (close, or the entry-stop price)
    entry_type: str = "MKT"                    # 'MKT' (next open) | 'STP' (resting stop, next session only)
    entry_stop: Optional[float] = None
    stop_mode: Optional[str] = None            # None | 'atr_fill' (k*atr from fill) | 'level'
    stop_k: float = float("nan")
    atr: float = float("nan")
    stop_level: float = float("nan")
    trail_k: Optional[float] = None            # trailing k*ATR stop (ratchet on closes); use with
                                               # stop_mode='atr_fill', stop_k=trail_k, atr=ATR(trail_atr_n)
    trail_atr_n: int = 14
    target_r: Optional[float] = None
    max_hold: Optional[int] = None
    sizing_stop: Optional[float] = None        # stop used for SWING_RISK sizing if different
    max_adds: int = 0                          # pyramiding (Turtle)
    add_atr: Optional[float] = None
    tag: str = "entry"
    data: dict = field(default_factory=dict)

    def planned_stop(self) -> Optional[float]:
        if self.stop_mode == "atr_fill":
            return self.ref_price - self.side * self.stop_k * self.atr
        if self.stop_mode == "level":
            return self.stop_level
        return None

    def scale(self, r: float) -> None:
        """Split by ratio r (new shares per old): prices / r."""
        self.ref_price /= r
        if self.entry_stop is not None:
            self.entry_stop /= r
        self.atr /= r
        self.stop_level /= r
        if self.sizing_stop is not None:
            self.sizing_stop /= r
        for k in list(self.data):
            if k.endswith("_px"):
                self.data[k] = self.data[k] / r


@dataclass
class PosState:
    side: int
    plan: Plan
    entry_session: pd.Timestamp
    entry_px: float = float("nan")
    entry_qty: float = 0.0
    fresh: bool = True
    stop: Optional[float] = None
    target: Optional[float] = None
    init_stop: Optional[float] = None
    units: int = 1
    last_fill: float = float("nan")
    exiting: Optional[str] = None
    oco: str = ""
    data: dict = field(default_factory=dict)

    def scale(self, r: float) -> None:
        self.plan.scale(r)
        for a in ("entry_px", "last_fill"):
            setattr(self, a, getattr(self, a) / r)
        for a in ("stop", "target", "init_stop"):
            v = getattr(self, a)
            if v is not None:
                setattr(self, a, v / r)
        self.entry_qty *= r
        for k in list(self.data):
            if k.endswith("_px"):
                self.data[k] = self.data[k] / r


@dataclass
class Pending:
    order_id: int
    plan: Plan
    qty: float
    session: pd.Timestamp


class SwingBase(Strategy):
    """Single-name daily swing engine. Subclasses implement ``entry`` and
    optionally ``exit_signal`` / ``update_levels`` / ``window``."""

    bar_freq = "1d"
    asset = "STK/ETF"
    universe = SWING_ETF
    requires = [Requirement("bars", "1d")]
    sizing = "SWING_RISK"
    min_history: int = 30
    allow_reversal: bool = False

    # ------------------------------------------------------------ hooks
    def window(self) -> Optional[int]:
        """Bars needed for the signal (None = full history)."""
        return 400

    def entry(self, ctx, sym: str, b: Bars) -> Optional[Plan]:          # pragma: no cover - abstract
        return None

    def exit_signal(self, ctx, sym: str, b: Bars, ps: PosState) -> Optional[str]:
        return None

    def update_levels(self, ctx, sym: str, b: Bars, ps: PosState) -> None:
        """Default: ratchet a trailing k*ATR stop if the plan has one."""
        p = ps.plan
        if p.trail_k:
            a = float(np.asarray(_atr(b.h, b.l, b.c, p.trail_atr_n))[-1])
            if np.isfinite(a):
                cand = b.c[-1] - ps.side * p.trail_k * a
                ps.stop = cand if ps.stop is None else (max(ps.stop, cand) if ps.side > 0 else min(ps.stop, cand))

    def prefilter(self, ctx, sym: str) -> bool:
        """Cheap necessary condition for an entry (False = skip loading bars today for a flat name)."""
        return True

    def on_entry_filled(self, ctx, sym: str, b: Bars, ps: PosState) -> None:
        """Fill-based initial levels (called at the entry-day close)."""
        p = ps.plan
        fill = ps.entry_px
        if p.stop_mode == "atr_fill":
            ps.stop = fill - ps.side * p.stop_k * p.atr
        elif p.stop_mode == "level":
            ps.stop = p.stop_level
        ps.init_stop = ps.stop
        if p.target_r and ps.init_stop is not None:
            risk = abs(fill - ps.init_stop)
            ps.target = fill + ps.side * p.target_r * risk if risk > 0 else None
        ps.last_fill = fill

    def add_order(self, ctx, sym: str, b: Bars, ps: PosState) -> Optional[tuple[float, float]]:
        """Pyramiding: (stop price, signed qty) for the next add, or None."""
        p = ps.plan
        if not p.max_adds or p.add_atr is None or ps.units > p.max_adds or not np.isfinite(ps.last_fill):
            return None
        n = p.atr
        px = ps.last_fill + ps.side * p.add_atr * n
        stop = px - ps.side * p.stop_k * n
        q = self._size(ctx, sym, ps.side, px, stop, extra_gross=0.0, existing_qty=ctx.position(sym))
        if q == 0:
            return None
        return px, q

    def on_add_filled(self, ctx, ps: PosState, px: float) -> None:
        """Turtle: after an add, all units' stop = last add fill - k*N (never loosened)."""
        p = ps.plan
        cand = px - ps.side * p.stop_k * p.atr
        if ps.stop is None:
            ps.stop = cand
        else:
            ps.stop = max(ps.stop, cand) if ps.side > 0 else min(ps.stop, cand)

    # ------------------------------------------------------------ engine callbacks
    def init(self, ctx) -> None:
        w = self.window()
        self._cache = BarCache(keep=None if w is None else max(w, 1000), fetch=None if w is None else w)
        self._pos: dict[str, PosState] = {}
        self._pending: dict[str, Pending] = {}
        self._roles: dict[int, tuple[str, str]] = {}
        self._oco_n = 0
        self._oid_map: dict[int, Any] = {}
        self._nodata: dict[str, int] = {}
        self._gross = 0.0

    def universe_today(self, ctx) -> list[str]:
        return self.symbols(ctx)

    def on_fill(self, ctx, fill) -> None:
        role = self._roles.get(fill.order_id)
        if role is None:
            return
        kind, sym = role
        if kind == "entry":
            pend = self._pending.get(sym)
            ps = self._pos.get(sym)
            if ps is not None and ps.exiting and pend is not None and pend.plan.side != ps.side:
                ps = None                      # reversal: the old state is being closed
            if ps is None and pend is not None:
                ps = PosState(side=pend.plan.side, plan=pend.plan, entry_session=ctx.session_date())
                self._pos[sym] = ps
            if ps is not None:
                tot = ps.entry_qty + fill.quantity
                ps.entry_px = fill.price if ps.entry_qty == 0 else \
                    (ps.entry_px * ps.entry_qty + fill.price * fill.quantity) / tot
                ps.entry_qty = tot
        elif kind == "add":
            ps = self._pos.get(sym)
            if ps is not None:
                ps.units += 1
                ps.last_fill = fill.price
                self.on_add_filled(ctx, ps, fill.price)
                ps.data["add_pending_resync"] = True

    def on_session_start(self, ctx) -> None:
        syms = sorted(set(self._pos) | set(self._pending))
        if not syms:
            return
        today = ctx.session_date()
        try:
            acts = ctx.portal().corporate_actions(syms, today, today)
        except Exception:
            acts = []
        for a in acts:
            ex = pd.Timestamp(a.ex_date)
            if ex.tzinfo is not None:
                ex = ex.tz_convert("America/New_York").tz_localize(None)
            if ex.normalize() != today:
                continue
            if a.type == ActionType.SPLIT and a.ratio and a.ratio > 0:
                r = float(a.ratio)
                ps = self._pos.get(a.symbol)
                if ps is not None:
                    ps.scale(r)
                    self._sync_orders(ctx, a.symbol, ps, force=True)
                pend = self._pending.get(a.symbol)
                if pend is not None:
                    # the engine either adjusts or cancels working orders on a split: keep the plan in
                    # post-split prices and re-submit the entry only if it was cancelled
                    pend.plan.scale(r)
                    pend.qty = float(int(pend.qty * r))
                    # (bracket child templates may not be adjusted by the engine, so an adjusted entry is
                    # cancelled and re-sent with split-adjusted prices)
                    o = self._order(ctx, pend.order_id)
                    if o is not None and abs(o.filled_qty) < 1e-9:
                        if o.status in (OrderStatus.PENDING, OrderStatus.WORKING):
                            ctx.cancel(o.id)
                        self._pending.pop(a.symbol, None)
                        if pend.qty:
                            self._submit_entry(ctx, a.symbol, pend.plan, pend.qty)
            elif a.type == ActionType.SYMBOL_CHANGE and a.new_symbol:
                if a.symbol in self._pos:
                    self._pos[a.new_symbol] = self._pos.pop(a.symbol)
                    self._sync_orders(ctx, a.new_symbol, self._pos[a.new_symbol], force=True)
                self._pending.pop(a.symbol, None)

    def on_bar(self, ctx) -> None:
        self._cache.refresh_actions(ctx)
        univ = set(self.universe_today(ctx))
        syms = sorted(univ | set(self._pos) | set(self._pending) | {s for s, q in ctx.positions().items() if q})
        self._gross = _sz.equity_gross(ctx) + sum(abs(p.qty) * p.plan.ref_price for p in self._pending.values())
        self.on_day_start(ctx, syms)
        for sym in syms:
            self._process(ctx, sym, sym in univ)

    def on_day_start(self, ctx, syms) -> None:
        pass

    # ------------------------------------------------------------ core
    def _order(self, ctx, oid: int):
        return self._oid_map.get(oid)

    def _bars(self, ctx, sym: str) -> Optional[Bars]:
        cal = ctx.calendar()
        i = cal.session_index(ctx.session_date())
        last = self._nodata.get(sym)
        if last is not None and i - last < 20 and sym not in self._pos:
            return None
        b = self._cache.get(ctx, sym, self.window())
        if b is None or len(b) == 0:
            self._nodata[sym] = i
            return None
        self._nodata.pop(sym, None)
        return b

    def _process(self, ctx, sym: str, in_universe: bool) -> None:
        q = ctx.position(sym)
        pend = self._pending.get(sym)
        if pend is not None:
            o = self._order(ctx, pend.order_id)
            if o is None or o.status not in (OrderStatus.PENDING, OrderStatus.WORKING, OrderStatus.PARTIAL):
                self._pending.pop(sym, None)
                self._gross -= abs(pend.qty) * pend.plan.ref_price
                pend = None
        ps = self._pos.get(sym)
        if q == 0 and ps is not None and (pend is None):
            self._pos.pop(sym, None)       # closed by stop / target / exit / delisting
            self._cancel_sym(ctx, sym)
            ps = None
        if q == 0 and ps is None and pend is None and in_universe and not self.prefilter(ctx, sym):
            return
        if q != 0 and ps is None:
            # position without state (should not happen): adopt with no protective levels
            acct = ctx.account().get(sym)
            px = acct.avg_price if acct is not None else ctx.mark(sym)
            ps = PosState(side=1 if q > 0 else -1, plan=Plan(side=1 if q > 0 else -1, ref_price=px),
                          entry_session=ctx.session_date(), entry_px=px, entry_qty=q)
            self._pos[sym] = ps
        b = self._bars(ctx, sym)
        fresh_bar = b is not None and len(b) > 0 and b.end[-1] == int(ctx.now.value)
        if ps is not None and q != 0:
            was_exiting = ps.exiting
            self._manage(ctx, sym, b if fresh_bar else None, ps)
            # stop-and-reverse: an exit sent at this close may be paired with an opposite-side entry
            if ps.exiting and not was_exiting and self.allow_reversal and pend is None and in_universe \
                    and fresh_bar and len(b) >= self.min_history:
                plan = self.entry(ctx, sym, b)
                if plan is not None and plan.side != ps.side:
                    self._enter(ctx, sym, plan)
            return
        if pend is not None or not in_universe or not fresh_bar or len(b) < self.min_history:
            return
        plan = self.entry(ctx, sym, b)
        if plan is not None:
            self._enter(ctx, sym, plan)

    def held_sessions(self, ctx, ps: PosState) -> int:
        cal = ctx.calendar()
        return cal.session_index(ctx.session_date()) - cal.session_index(ps.entry_session) + 1

    def _manage(self, ctx, sym: str, b: Optional[Bars], ps: PosState) -> None:
        if ps.exiting:
            # an exit order was sent; re-send if it expired unfilled
            if not any(o.tag == ps.exiting and o.type == OrderType.MARKET for o in ctx.open_orders(sym)):
                self._send_exit(ctx, sym, ps, ps.exiting)
            return
        if b is None:
            return
        if ps.fresh:
            if not np.isfinite(ps.entry_px):
                acct = ctx.account().get(sym)
                ps.entry_px = acct.avg_price if acct is not None else b.c[-1]
            self.on_entry_filled(ctx, sym, b, ps)
            ps.fresh = False
        reason = self.exit_signal(ctx, sym, b, ps)
        if reason is None and ps.plan.max_hold and self.held_sessions(ctx, ps) >= ps.plan.max_hold:
            reason = "time_exit"
        if reason:
            self._send_exit(ctx, sym, ps, reason)
            return
        self.update_levels(ctx, sym, b, ps)
        self._sync_orders(ctx, sym, ps, b=b)

    # ------------------------------------------------------------ orders
    def _track(self, o, kind: str, sym: str):
        self._oid_map[o.id] = o
        self._roles[o.id] = (kind, sym)
        return o

    def _cancel_sym(self, ctx, sym: str) -> None:
        for o in ctx.open_orders(sym):
            ctx.cancel(o.id)

    def _new_oco(self, sym: str) -> str:
        self._oco_n += 1
        return f"{self.id}-{sym}-{self._oco_n}"

    def _send_exit(self, ctx, sym: str, ps: PosState, reason: str) -> None:
        q = ctx.position(sym)
        self._cancel_sym(ctx, sym)
        ps.exiting = reason
        if q == 0:
            return
        ps.oco = self._new_oco(sym)
        if ps.stop is not None and np.isfinite(ps.stop):
            o = ctx.order(sym, -q, OrderType.STOP, stop_price=float(ps.stop), tif=TimeInForce.GTC, tag="stop",
                          oco_group=ps.oco)
            self._track(o, "stop", sym)
        o = ctx.order(sym, -q, OrderType.MARKET, tif=TimeInForce.DAY, tag=reason, oco_group=ps.oco)
        self._track(o, "exit", sym)

    def _sync_orders(self, ctx, sym: str, ps: PosState, force: bool = False, b: Optional[Bars] = None) -> None:
        q = ctx.position(sym)
        if q == 0:
            return
        want = []
        if ps.stop is not None and np.isfinite(ps.stop):
            want.append(("stop", OrderType.STOP, -q, float(ps.stop)))
        if ps.target is not None and np.isfinite(ps.target):
            want.append(("target", OrderType.LIMIT, -q, float(ps.target)))
        add = self.add_order(ctx, sym, b, ps) if b is not None else None
        if add is not None:
            want.append(("add", OrderType.STOP, add[1], float(add[0])))
        have = [o for o in ctx.open_orders(sym)]
        if not force and not ps.data.pop("add_pending_resync", False) and len(have) == len(want):
            ok = True
            for (tag, typ, qty, px), o in zip(sorted(want), sorted(have, key=lambda o: o.tag)):
                opx = o.stop_price if typ == OrderType.STOP else o.limit_price
                tick = tick_for(px)
                if o.tag != tag or o.type != typ or abs(o.remaining - qty) > 1e-9 or opx is None \
                        or abs(opx - px) >= tick or o.oco_group is None and tag != "add":
                    ok = False
                    break
            if ok:
                return
        self._cancel_sym(ctx, sym)
        ps.oco = self._new_oco(sym)
        for tag, typ, qty, px in want:
            if typ == OrderType.STOP:
                o = ctx.order(sym, qty, typ, stop_price=px, tif=TimeInForce.GTC, tag=tag,
                              oco_group=None if tag == "add" else ps.oco)
            else:
                o = ctx.order(sym, qty, typ, limit_price=px, tif=TimeInForce.GTC, tag=tag, oco_group=ps.oco)
            self._track(o, "add" if tag == "add" else tag, sym)

    def _size(self, ctx, sym: str, side: int, px: float, stop: Optional[float], extra_gross: float = 0.0,
              existing_qty: float = 0.0) -> float:
        nav = ctx.nav
        gross = self._gross + extra_gross
        if self.sizing == "SWING_RISK":
            if stop is None or not np.isfinite(stop) or (stop - px) * side >= 0:
                return 0.0
            return _sz.size_swing_risk(nav, px, stop, current_gross=gross, existing_notional=abs(existing_qty) * px)
        if self.sizing == "SWING_NOTIONAL":
            return _sz.size_swing_notional(nav, px, side, current_gross=gross)
        raise ValueError(f"{self.id}: unsupported sizing {self.sizing}")

    def _enter(self, ctx, sym: str, plan: Plan) -> None:
        sstop = plan.sizing_stop if plan.sizing_stop is not None else plan.planned_stop()
        q = self._size(ctx, sym, plan.side, plan.ref_price, sstop)
        if q == 0 or (q > 0) != (plan.side > 0):
            return
        self._submit_entry(ctx, sym, plan, q)

    def _submit_entry(self, ctx, sym: str, plan: Plan, q: float) -> None:
        stop = plan.planned_stop()
        tgt = None
        if plan.target_r and stop is not None:
            risk = abs(plan.ref_price - stop)
            tgt = plan.ref_price + plan.side * plan.target_r * risk if risk > 0 else None
        et_ = OrderType.STOP if plan.entry_type == "STP" else OrderType.MARKET
        if stop is not None or tgt is not None:
            o = ctx.bracket(sym, q, entry_type=et_, entry_stop=plan.entry_stop, stop=stop, target=tgt,
                            tif=TimeInForce.DAY, child_tif=TimeInForce.GTC, tag=plan.tag)
        else:
            o = ctx.order(sym, q, et_, stop_price=plan.entry_stop, tif=TimeInForce.DAY, tag=plan.tag)
        if o.status == OrderStatus.REJECTED:
            return
        self._track(o, "entry", sym)
        self._pending[sym] = Pending(o.id, plan, q, ctx.session_date())
        self._gross += abs(q) * plan.ref_price


# ============================================================== cross-section base
class CrossSectionBase(Strategy):
    """Periodic long/short rebalancing (CROSS_SECTION sizing profile)."""

    bar_freq = "1d"
    asset = "STK"
    sizing = "CROSS_SECTION"
    universe = ()
    universe_name = SP500_PIT
    requires = [Requirement("bars", "1d", symbols=("SPY",), critical=False,
                            note="universe members are resolved PIT through portal.universe('SP500')")]
    need_sectors: bool = False

    def window(self) -> int:
        return 300

    def init(self, ctx) -> None:
        self._cache = BarCache(keep=max(self.window(), 400) + 50, fetch=self.window())
        self._queue: Optional[tuple[int, dict]] = None    # (sessions to wait, target weights)
        self._sector_flagged = False
        self._sec_cache = None

    # hooks
    def is_signal_day(self, ctx) -> bool:  # pragma: no cover - abstract
        return False

    def targets(self, ctx, members: list[str]) -> Optional[dict[str, float]]:  # pragma: no cover
        return None

    def exec_lag(self) -> int:
        """Sessions between the signal close and the open at which the trade executes (1 = next open)."""
        return 1

    def members(self, ctx) -> list[str]:
        return sorted(set(ctx.portal().universe(self.universe_name)))

    def sectors(self, ctx, syms) -> dict[str, str]:
        # the security master is re-read at most every 21 sessions (classifications change rarely and the
        # portal's master is expensive to build)
        i = ctx.calendar().session_index(ctx.session_date())
        cached = getattr(self, "_sec_cache", None)
        if cached is None or i - cached[0] >= 21:
            allsec = sectors_from_master(ctx, None)
            self._sec_cache = cached = (i, allsec)
        sec = {s: cached[1][s] for s in syms if s in cached[1]}
        if not sec:
            if self.need_sectors:
                raise NotTestableError(f"{self.id}: security master has no sector column ({SECTOR_COLUMNS}); "
                                       "sector-neutral construction cannot be done faithfully")
            if not self._sector_flagged:
                ctx.flag_fidelity(Fidelity.T2_PROXY, f"{self.id}: no sector data in security master; "
                                                     "CROSS_SECTION sector caps not applied")
                self._sector_flagged = True
        return sec

    def on_bar(self, ctx) -> None:
        self._cache.refresh_actions(ctx)
        if self._queue is not None:
            wait, w = self._queue
            wait -= 1
            if wait <= 0:
                self._queue = None
                self._rebalance(ctx, w)
            else:
                self._queue = (wait, w)
        if self.is_signal_day(ctx):
            w = self.targets(ctx, self.members(ctx))
            if w is not None:
                lag = self.exec_lag()
                if lag <= 1:
                    self._rebalance(ctx, w)
                else:
                    self._queue = (lag - 1, w)

    def _rebalance(self, ctx, weights: dict[str, float]) -> None:
        ctx.cancel_all()
        nav = ctx.nav
        cur = {s: q for s, q in ctx.positions().items() if q}
        prices = {}
        for s in set(weights) | set(cur):
            try:
                prices[s] = float(ctx.mark(s))
            except Exception:
                prices[s] = float("nan")
        tgt = _sz.weights_to_shares(weights, nav, prices)
        deltas = _sz.rebalance_deltas(tgt, cur)
        # reductions first, then increases (buying power)
        def key(item):
            s, d = item
            c = cur.get(s, 0.0)
            return (0 if abs(c + d) < abs(c) else 1, s)
        for s, d in sorted(deltas.items(), key=key):
            ctx.order(s, d, tag="exit_rebalance" if tgt.get(s, 0.0) == 0 else "rebalance")
