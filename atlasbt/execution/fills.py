"""Fill models.

* :class:`BarFillModel` - bar data only (T1). Next-bar fills, gap-aware stop
  markets, limit "trade-through by 1 tick" rule, stop-limits, participation
  cap with partial fills, adverse-first intrabar path, modelled half-spread +
  square-root impact on taker fills, SSR handling for short sales.
* :class:`QuoteFillModel` - NBBO quotes (T0): buy at ask / sell at bid.
* :class:`AuctionFillModel` - MOO/MOC at the official open/close +/- impact.
* :class:`ComboFillModel` - single options and multi-leg packages from ONE
  quote snapshot per leg set (all legs same snapshot or reject), at natural
  with an optional price-improvement fraction toward mid.

The engine owns the event loop and calls :meth:`BarFillModel.simulate` /
:meth:`QuoteFillModel.simulate` per symbol and step with the orders that are
eligible. ``try_fill`` methods implement ``core.protocols.FillModel`` for
stand-alone single-order use (they read the portal directly).

Intrabar path (ARCHITECTURE §4): within one bar the price is assumed to move
open -> adverse extreme -> favourable extreme -> close for the position being
managed (long: low first; short: high first; flat: adverse for the side of the
pending orders). Every fill gets a ``pos`` on this path (0 = open, 3 = close)
so several orders on the same symbol are processed in path order; OCO
reductions and bracket child activations happen in that order.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.types import (
    CostBreakdown, Fidelity, Fill, Instrument, Leg, Order, OrderType, Side,
)
from .constraints import bar_is_halted, round_to_tick, session_allows, valid_tick
from .impact import SquareRootImpact
from .spread import SpreadModel

INF = float("inf")


# --------------------------------------------------------------------------
# Data views
# --------------------------------------------------------------------------
@dataclass
class BarView:
    start: int
    end: int
    o: float
    h: float
    l: float
    c: float
    v: float
    sess: str = "RTH"
    mso: Optional[float] = None      # minutes since RTH open at bar start (None: daily bar)
    mtc: Optional[float] = None      # minutes to RTH close at bar end


class Path:
    """Piecewise-linear intrabar path through 4 points."""

    __slots__ = ("pts",)

    def __init__(self, pts: Sequence[float]):
        self.pts = list(pts)

    def price_at(self, s: float) -> float:
        k = int(math.floor(s))
        if k >= 3:
            return self.pts[3]
        f = s - k
        a, b = self.pts[k], self.pts[k + 1]
        return a + (b - a) * f

    def first_reach(self, s: float, level: float, up: bool) -> Optional[tuple[float, float, bool]]:
        """First position >= s where price >= level (up) / <= level (down).
        Returns (pos, price, gapped). ``gapped`` means the price at ``s`` is
        already beyond the level (fill at that price, not the level)."""
        p0 = self.price_at(s)
        if (up and p0 >= level) or ((not up) and p0 <= level):
            return s, p0, True
        k = int(math.floor(s))
        for seg in range(k, 3):
            a = p0 if seg == k else self.pts[seg]
            b = self.pts[seg + 1]
            st = s if seg == k else float(seg)
            if (up and b >= level) or ((not up) and b <= level):
                f = 0.0 if b == a else (level - a) / (b - a)
                f = min(max(f, 0.0), 1.0)
                return st + f * (seg + 1 - st), level, False
        return None


@dataclass
class Trigger:
    pos: float
    base: float                    # reference (mid-proxy) price where a taker executes
    liquidity: str                 # 'TAKER' | 'MAKER'
    cap: Optional[float] = None    # taker: limit cap; maker: exact fill price
    reasons: tuple[str, ...] = ()
    reject_reason: str = ""

    @property
    def ssr_reject(self) -> bool:
        return bool(self.reject_reason)


@dataclass
class FillDecision:
    order: Order
    pos: float
    ts_ns: int
    qty: float
    price: float
    base: float
    liquidity: str
    spread_cost: float
    impact_cost: float
    fidelity: Fidelity
    reasons: tuple[str, ...]
    partial: bool = False


@dataclass
class SymbolMarket:
    """Per-symbol, per-step pricing context supplied by the engine."""

    instrument: Instrument
    spread_estimate: float = float("nan")      # relative full spread (PIT daily estimate)
    spread_source: str = "default"
    adv_dollar: float = float("nan")
    sigma: float = float("nan")
    ssr_active: Callable[[BarView], bool] = lambda b: False
    position: Callable[[], float] = lambda: 0.0
    daily_mode: bool = False
    daily_open_spread_mult: float = 1.0
    #: fill-time short-sale classification: order -> (is_short_sale, locate_available)
    short_sale_state: Optional[Callable[[Order], tuple[bool, bool]]] = None


@dataclass
class BarFillConfig:
    participation_cap: float = 0.10
    allow_fractional: bool = False
    ssr_mode: str = "ask_limit"          # 'ask_limit' | 'reject' | 'ignore'
    same_bar_exit: bool = True
    limit_through_ticks: int = 1


@dataclass
class SimStats:
    halted_bars: int = 0
    nan_volume_bars: int = 0
    ambiguous_bars: int = 0
    partial_fills: int = 0
    ssr_blocked: int = 0


# --------------------------------------------------------------------------
# Bar fill model
# --------------------------------------------------------------------------
class BarFillModel:
    def __init__(self, cfg: Optional[BarFillConfig] = None, spread: Optional[SpreadModel] = None,
                 impact: Optional[SquareRootImpact] = None):
        self.cfg = cfg or BarFillConfig()
        self.spread = spread or SpreadModel()
        self.impact = impact or SquareRootImpact()

    # ---------------------------------------------------------------- path
    @staticmethod
    def adverse_path(bar: BarView, low_first: bool, start_price: Optional[float] = None) -> Path:
        o = bar.o if start_price is None else start_price
        return Path([o, bar.l, bar.h, bar.c] if low_first else [o, bar.h, bar.l, bar.c])

    @staticmethod
    def low_first_for(position: float, orders: Iterable[Order], bar: BarView) -> bool:
        if position > 0:
            return True
        if position < 0:
            return False
        net = 0
        for o in orders:
            net += 1 if o.quantity > 0 else -1
        if net > 0:
            return False      # buyers: adverse = up first
        if net < 0:
            return True
        return abs(bar.o - bar.l) <= abs(bar.h - bar.o)   # nearer extreme first

    # ---------------------------------------------------------------- pricing
    def half_spread(self, mkt: SymbolMarket, price: float, bar: BarView) -> float:
        tick = valid_tick(price, mkt.instrument)
        if getattr(self.spread, "method", None) == "calibrated":
            full = self.spread.full_spread(mkt.spread_estimate, price, tick, bar.mso, bar.mtc,
                                           adv_dollars=mkt.adv_dollar, asset_class=mkt.instrument.asset_class)
        else:
            full = self.spread.full_spread(mkt.spread_estimate, price, tick, bar.mso, bar.mtc)
        if bar.mso is None:
            full *= mkt.daily_open_spread_mult
        return 0.5 * full

    def trigger(self, order: Order, path: Path, s: float, mkt: SymbolMarket, bar: BarView,
                passive_only: bool = False) -> Optional[Trigger]:
        buy = order.quantity > 0
        t = order.type
        p0 = path.price_at(s)
        tick = valid_tick(p0, mkt.instrument)
        through = self.cfg.limit_through_ticks * tick
        if t == OrderType.MARKET:
            trig = Trigger(s, p0, "TAKER", None, ("market",))
        elif t == OrderType.LIMIT:
            trig = self._limit(order.limit_price, buy, path, s, mkt, bar, through, passive_only=passive_only)
        elif t == OrderType.STOP:
            r = path.first_reach(s, order.stop_price, up=buy)
            if r is None:
                return None
            pos, px, gapped = r
            trig = Trigger(pos, px, "TAKER", None, ("stop gapped: filled at first price after trigger",) if gapped else ("stop",))
        elif t == OrderType.STOP_LIMIT:
            r = path.first_reach(s, order.stop_price, up=buy)
            if r is None:
                return None
            pos, px, gapped = r
            trig = self._limit(order.limit_price, buy, path, pos, mkt, bar, through, base_override=px,
                               passive_only=passive_only)
            if trig is not None:
                trig.reasons = ("stop-limit triggered",) + trig.reasons
        else:
            return None
        if trig is None:
            return None
        # short-sale marking decided at fill time (after earlier fills in this bar);
        # a mark set at submission is never cleared
        if (not buy) and mkt.short_sale_state is not None and not order.is_short_sale:
            is_short, located = mkt.short_sale_state(order)
            if is_short:
                order.is_short_sale = True
                if not located:
                    return Trigger(trig.pos, trig.base, "TAKER", None, ("no locate",),
                                   reject_reason="short sale rejected at fill time: no locate / not shortable")
        # Reg SHO price test for short sales hitting the bid
        if (not buy) and order.is_short_sale and trig.liquidity == "TAKER" and self.cfg.ssr_mode != "ignore" \
                and mkt.ssr_active(bar):
            if self.cfg.ssr_mode == "reject":
                return Trigger(trig.pos, trig.base, "TAKER", None, ("SSR active",),
                               reject_reason="short sale rejected: SSR (Rule 201) active")
            hs = self.half_spread(mkt, trig.base, bar)
            ask = round_to_tick(trig.base * (1.0 + hs), tick, "up")
            if order.limit_price is not None:
                ask = max(ask, order.limit_price)
            r = path.first_reach(trig.pos, ask + tick, up=True)
            if r is None:
                return None
            return Trigger(r[0], ask, "MAKER", ask, ("SSR: short sale posted at ask (cannot hit bid)",))
        return trig

    def _limit(self, L: float, buy: bool, path: Path, s: float, mkt: SymbolMarket, bar: BarView,
               through: float, base_override: Optional[float] = None,
               passive_only: bool = False) -> Optional[Trigger]:
        p0 = path.price_at(s) if base_override is None else base_override
        hs = self.half_spread(mkt, p0, bar)
        if buy:
            if not passive_only and p0 * (1.0 + hs) <= L + 1e-12:
                return Trigger(s, p0, "TAKER", L, ("marketable limit",))
            r = path.first_reach(s, L - through, up=False)
        else:
            if not passive_only and p0 * (1.0 - hs) >= L - 1e-12:
                return Trigger(s, p0, "TAKER", L, ("marketable limit",))
            r = path.first_reach(s, L + through, up=True)
        if r is None:
            return None
        return Trigger(r[0], L, "MAKER", L, ("limit traded through by >=1 tick",))

    def max_qty_within_limit(self, base: float, hs: float, limit: float, buy: bool, mkt: SymbolMarket) -> float:
        """Largest quantity whose modelled price base*(1 +/- hs +/- imp(q)) stays
        within ``limit`` (square-root impact inverted). inf if impact is not
        modelled."""
        room = (limit / base - 1.0 - hs) if buy else (1.0 - hs - limit / base)
        if room < -1e-12:
            return 0.0
        room = max(room, 0.0)
        coef = self.impact.k * mkt.sigma if mkt.sigma == mkt.sigma else float("nan")
        if not (coef == coef and coef > 0 and mkt.adv_dollar == mkt.adv_dollar and mkt.adv_dollar > 0):
            return INF
        x = min(room, self.impact.max_impact) / coef
        if room >= self.impact.max_impact:
            return INF
        return mkt.adv_dollar * x * x / base

    def price(self, order: Order, trig: Trigger, qty_abs: float, mkt: SymbolMarket, bar: BarView
              ) -> tuple[float, float, float, list[str]]:
        """Returns (price, spread_cost_usd, impact_cost_usd, reasons). Taker
        prices are rounded to the tick AGAINST the trader (buys up, sells down)
        and capped at any limit."""
        mult = mkt.instrument.multiplier
        reasons: list[str] = []
        if trig.liquidity == "MAKER":
            return float(trig.cap), 0.0, 0.0, reasons
        buy = order.quantity > 0
        base = trig.base
        hs = self.half_spread(mkt, base, bar)
        imp = self.impact.impact_fraction(qty_abs * base, mkt.adv_dollar, mkt.sigma)
        if not (mkt.adv_dollar == mkt.adv_dollar and mkt.sigma == mkt.sigma):
            reasons.append("impact not modelled: ADV/vol unavailable")
        if buy:
            raw = base * (1.0 + hs + imp)
            px = round_to_tick(raw, valid_tick(raw, mkt.instrument), "up")
            if trig.cap is not None:
                px = min(px, trig.cap)
        else:
            raw = base * (1.0 - hs - imp)
            px = round_to_tick(raw, valid_tick(raw, mkt.instrument), "down")
            if trig.cap is not None:
                px = max(px, trig.cap)
        spread_cost = abs(min(abs(px - base), base * hs)) * qty_abs * mult
        impact_cost = max(abs(px - base) * qty_abs * mult - spread_cost, 0.0)
        return px, spread_cost, impact_cost, reasons

    # ---------------------------------------------------------------- simulation
    @staticmethod
    def _restart_path(path: Path, bar: BarView, fill_pos: float, fill_base: float, long: bool) -> Path:
        """Remainder of the bar after an entry at path position ``fill_pos``.

        The adverse extreme for the new position is assumed still reachable;
        the favourable extreme only if the ORIGINAL path reaches it at or after
        the entry (otherwise it already happened before the entry)."""
        adverse = bar.l if long else bar.h
        fav = bar.h if long else bar.l
        fav_ok = fill_pos <= 0.0 or any(path.pts[k] == fav and k >= fill_pos for k in (1, 2))
        if fav_ok:
            return Path([fill_base, adverse, fav, bar.c])
        return Path([fill_base, adverse, adverse, bar.c])

    def simulate(self, mkt: SymbolMarket, bars: Sequence[BarView],
                 get_orders: Callable[[], list[tuple[Order, object]]],
                 on_fill: Callable[[FillDecision], None],
                 on_reject: Callable[[Order, str], None],
                 resolved_with: Optional[str] = None) -> SimStats:
        stats = SimStats()
        cfg = self.cfg
        for bar in bars:
            if not (bar.v == bar.v):
                stats.nan_volume_bars += 1          # unknown volume = no available liquidity
                continue
            if bar_is_halted(bar.v, bar.sess, bar.o):
                stats.halted_bars += 1
                continue
            cap_left = cfg.participation_cap * bar.v
            orders0 = [o for o, _ in get_orders()]
            if not orders0:
                break
            pos_before = mkt.position()
            path = self.adverse_path(bar, self.low_first_for(pos_before, orders0, bar))
            s = 0.0
            t0 = bar.start
            restarted = False
            nfills = 0
            for _guard in range(64):
                cands = []
                for o, ex in get_orders():
                    if o.type in (OrderType.MOO, OrderType.MOC, OrderType.COMBO):
                        continue
                    if not session_allows(bar.sess, getattr(ex, "outside_rth", False)):
                        continue
                    if getattr(ex, "activate_at_ns", None) is not None and bar.start < ex.activate_at_ns:
                        continue
                    ib = getattr(ex, "in_bar_from", None)
                    if ib is not None:
                        start = max(s, ib)
                    elif ex.eligible_ns > bar.start:
                        continue
                    else:
                        start = s
                    ex.evaluated = True
                    passive = getattr(ex, "taker_bar", None) == bar.start
                    tr = self.trigger(o, path, start, mkt, bar, passive_only=passive)
                    if tr is not None:
                        cands.append((tr.pos, o.id, o, ex, tr))
                if not cands:
                    break
                cands.sort(key=lambda x: (x[0], x[1]))
                pos, _, o, ex, tr = cands[0]
                if tr.reject_reason:
                    if "SSR" in tr.reject_reason:
                        stats.ssr_blocked += 1
                    on_reject(o, tr.reject_reason)
                    continue
                rem = abs(o.remaining)
                q = min(rem, cap_left)
                limit_bound = False
                if tr.liquidity == "TAKER" and tr.cap is not None:
                    hs = self.half_spread(mkt, tr.base, bar)
                    qmax = self.max_qty_within_limit(tr.base, hs, tr.cap, o.quantity > 0, mkt)
                    if qmax < q:
                        q = qmax
                        limit_bound = True
                if not cfg.allow_fractional or mkt.instrument.is_option:
                    q = math.floor(q + 1e-9)
                if q <= 0:
                    if limit_bound:
                        ex.taker_bar = bar.start     # rest passively for the remainder of the bar
                        continue
                    break
                if limit_bound:
                    ex.taker_bar = bar.start
                price, sc, ic, preasons = self.price(o, tr, q, mkt, bar)
                reasons = list(tr.reasons) + preasons
                if limit_bound:
                    reasons.append("marketable limit: only size whose modelled cost is within the limit taken")
                fid = Fidelity.T1_BAR
                if mkt.spread_source == "default" and tr.liquidity == "TAKER":
                    reasons.append("spread default (no estimate)")
                    fid = Fidelity.T2_PROXY
                elif tr.liquidity == "TAKER" and str(mkt.spread_source).startswith(
                        ("calibrated", "Corwin-Schultz fallback")):
                    reasons.append("spread modelled (calibrated)")
                    head, *quals = [q.strip() for q in str(mkt.spread_source).split(",")]
                    if head != "calibrated":
                        reasons.append(f"spread: {head}")
                    # model qualifiers: era-extrapolated, listing assumed, stress regime,
                    # asset class inferred, ADV below calibration range, sigma prior, ...
                    reasons.extend(f"spread: {q}" for q in quals if q)
                    fid = max(fid, Fidelity.T2_PROXY)
                elif tr.liquidity == "TAKER":
                    reasons.append(f"spread estimated ({mkt.spread_source})")
                if bar.mso is None and tr.liquidity == "TAKER":
                    reasons.append("daily bar: next-open fill")
                if resolved_with:
                    reasons.append(f"intrabar path resolved with {resolved_with} bars")
                elif (len(cands) > 1 and nfills == 0 and cands[1][0] < 3.0) or restarted or nfills > 0:
                    reasons.append("intrabar order assumed adverse-first (bar data)")
                    fid = max(fid, Fidelity.T2_PROXY)
                    stats.ambiguous_bars += 1
                if "SSR" in " ".join(tr.reasons):
                    stats.ssr_blocked += 1
                # approximate intrabar time from the path position (floored to the second),
                # mapped onto [t0, bar.end] (t0 = entry time after a same-bar entry)
                frac = min(pos, 3.0) / 3.0
                ts_ns = int(t0 + int((bar.end - t0) * frac) // 10**9 * 10**9)
                ts_ns = max(ts_ns, t0)
                partial = q < rem - 1e-9
                if partial:
                    stats.partial_fills += 1
                    if not limit_bound:
                        reasons.append(f"participation cap {cfg.participation_cap:.0%} of bar volume")
                dec = FillDecision(o, pos, ts_ns, math.copysign(q, o.quantity), price, tr.base, tr.liquidity,
                                   sc, ic, fid, tuple(reasons), partial)
                on_fill(dec)
                nfills += 1
                cap_left -= q
                s = pos
                pos_after = mkt.position()
                if cfg.same_bar_exit and pos_before == 0 and pos_after != 0 and not restarted:
                    fill_base = tr.base if tr.liquidity == "TAKER" else price
                    path = self._restart_path(path, bar, pos, fill_base, pos_after > 0)
                    s = 0.0
                    t0 = ts_ns
                    restarted = True
                    for _o, _ex in get_orders():
                        if getattr(_ex, "in_bar_from", None) is not None:
                            _ex.in_bar_from = 0.0
                if cap_left <= 0:
                    break
            for _o, _ex in get_orders():
                if getattr(_ex, "in_bar_from", None) is not None:
                    _ex.in_bar_from = None
        return stats

    # ---------------------------------------------------------------- protocol
    def try_fill(self, order: Order, portal, window_start: pd.Timestamp, window_end: pd.Timestamp,
                 state, freq: str = "1m", spread_estimate: float = float("nan")) -> list[Fill]:
        """Stand-alone single-order evaluation (core.protocols.FillModel)."""
        inst = order.instrument
        df = portal.bars(inst.symbol, freq, window_start, window_end + pd.Timedelta(1, "ns"),
                         as_of=window_end, adjusted=False, sessions=("RTH",))
        elig = pd.Timestamp(order.eligible_at or window_start)
        df = df[df.index >= elig]
        out: list[Fill] = []

        class _Ex:
            outside_rth = False
            activate_at_ns = None
            in_bar_from = None
            eligible_ns = int(elig.value)
            evaluated = False

        ex = _Ex()
        pos_holder = {"q": state.position(inst.symbol) if state is not None else 0.0}
        adv = state.adv(inst.symbol, window_end) if state is not None else float("nan")
        sig = state.daily_vol(inst.symbol, window_end) if state is not None else float("nan")
        mkt = SymbolMarket(inst, spread_estimate, "default" if spread_estimate != spread_estimate else "given",
                           adv * float(df["close"].iloc[0]) if len(df) else float("nan"), sig,
                           position=lambda: pos_holder["q"])
        bars = [BarView(int(t.value), int(pd.Timestamp(r["known_at"]).value) if "known_at" in r else int(t.value),
                        r["open"], r["high"], r["low"], r["close"], r["volume"], str(r.get("session", "RTH")))
                for t, r in df.iterrows()]

        def _orders():
            return [(order, ex)] if abs(order.remaining) > 1e-9 else []

        def _on_fill(d: FillDecision):
            order.filled_qty += d.qty
            pos_holder["q"] += d.qty
            out.append(Fill(order.id, inst, pd.Timestamp(d.ts_ns, tz="UTC"), d.qty, d.price,
                            CostBreakdown(spread_cost=d.spread_cost, impact_cost=d.impact_cost),
                            d.fidelity, d.base, d.liquidity, "; ".join(d.reasons)))

        self.simulate(mkt, bars, _orders, _on_fill, lambda o, r: None)
        return out


# --------------------------------------------------------------------------
# Quote fill model (T0)
# --------------------------------------------------------------------------
@dataclass
class QuoteView:
    ts: int
    bid: float
    ask: float
    bid_size: float = float("nan")
    ask_size: float = float("nan")
    sess: str = "RTH"              # 'PRE' | 'RTH' | 'POST' (classified from the calendar)
    session_ns: int = 0            # session date (naive ns) the quote belongs to (0 = unknown)


class QuoteFillModel:
    """Fills against NBBO quotes strictly after ``eligible_at``.

    MARKET: buy at ask, sell at bid. LIMIT: only when marketable against a
    quote (passive queue position unknown -> resting limits are NOT filled on
    quotes alone; conservative). STOP: triggers when the touch crosses the stop
    (buy: ask >= stop; sell: bid <= stop) and executes at that quote. Displayed
    size caps each fill when available (remainder waits for later quotes);
    additional square-root impact is applied only when size is unknown.
    Locked/crossed quotes are skipped."""

    def __init__(self, impact: Optional[SquareRootImpact] = None, ssr_mode: str = "ask_limit",
                 allow_fractional: bool = False):
        self.impact = impact or SquareRootImpact()
        self.ssr_mode = ssr_mode
        self.allow_fractional = allow_fractional

    def simulate(self, mkt: SymbolMarket, quotes: Sequence[QuoteView],
                 get_orders: Callable[[], list[tuple[Order, object]]],
                 on_fill: Callable[[FillDecision], None],
                 on_reject: Callable[[Order, str], None]) -> SimStats:
        stats = SimStats()
        triggered: set[int] = set()
        for q in quotes:
            if not (q.bid > 0 and q.ask > 0) or q.ask <= q.bid:
                continue
            # displayed size is consumed across orders in priority (order id) order
            left = {True: q.ask_size, False: q.bid_size}
            for o, ex in get_orders():
                if max(ex.eligible_ns, getattr(ex, "quote_eligible_ns", 0) or 0) >= q.ts:
                    continue
                if not session_allows(q.sess, getattr(ex, "outside_rth", False)):
                    continue
                vs = getattr(ex, "valid_session_ns", None)
                if vs is not None and q.session_ns and q.session_ns < vs:
                    continue
                if o.type in (OrderType.MOO, OrderType.MOC, OrderType.COMBO):
                    continue
                if getattr(ex, "activate_at_ns", None) is not None and q.ts < ex.activate_at_ns:
                    continue
                ex.evaluated = True
                buy = o.quantity > 0
                touch = q.ask if buy else q.bid
                size = left[buy]
                if size == size and size <= 0:
                    continue
                ok = False
                cap_px = None
                if o.type == OrderType.MARKET:
                    ok = True
                elif o.type == OrderType.LIMIT:
                    ok = touch <= o.limit_price if buy else touch >= o.limit_price
                elif o.type in (OrderType.STOP, OrderType.STOP_LIMIT):
                    if o.id not in triggered:
                        hit = (q.ask >= o.stop_price) if buy else (q.bid <= o.stop_price)
                        if hit:
                            triggered.add(o.id)
                    if o.id in triggered:
                        if o.type == OrderType.STOP:
                            ok = True
                        else:
                            ok = touch <= o.limit_price if buy else touch >= o.limit_price
                if not ok:
                    continue
                reasons = ["NBBO quote fill", "fill source: quotes"]
                liq = "TAKER"
                px = touch
                if (not buy) and mkt.short_sale_state is not None and not o.is_short_sale:
                    is_short, located = mkt.short_sale_state(o)
                    if is_short:
                        o.is_short_sale = True
                        if not located:
                            on_reject(o, "short sale rejected at fill time: no locate / not shortable")
                            continue
                if (not buy) and o.is_short_sale and mkt.ssr_active(BarView(q.ts, q.ts, q.bid, q.ask, q.bid, q.bid, 0.0)) \
                        and self.ssr_mode != "ignore":
                    if self.ssr_mode == "reject":
                        on_reject(o, "short sale rejected: SSR (Rule 201) active")
                        stats.ssr_blocked += 1
                        continue
                    # cannot sell at/below NBB: rest at ask, fill only if a later bid reaches it
                    stats.ssr_blocked += 1
                    continue
                rem = abs(o.remaining)
                qty = rem if not (size == size and size > 0) else min(rem, size)
                if not self.allow_fractional or mkt.instrument.is_option:
                    qty = math.floor(qty + 1e-9)
                if qty <= 0:
                    continue
                mid = 0.5 * (q.bid + q.ask)
                imp = 0.0
                if not (size == size):
                    imp = self.impact.impact_fraction(qty * mid, mkt.adv_dollar, mkt.sigma)
                    reasons.append("displayed size unknown: sqrt impact added")
                px = px * (1 + imp) if buy else px * (1 - imp)
                if o.type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
                    px = min(px, o.limit_price) if buy else max(px, o.limit_price)
                mult = mkt.instrument.multiplier
                sc = abs(touch - mid) * qty * mult
                ic = abs(px - touch) * qty * mult
                partial = qty < rem - 1e-9
                if partial:
                    stats.partial_fills += 1
                if size == size:
                    left[buy] = size - qty
                on_fill(FillDecision(o, 0.0, q.ts, math.copysign(qty, o.quantity), px, mid, liq, sc, ic,
                                     Fidelity.T0_QUOTE, tuple(reasons), partial))
        return stats

    def try_fill(self, order, portal, window_start, window_end, state) -> list[Fill]:
        inst = order.instrument
        df = portal.quotes(inst.symbol, window_start, window_end + pd.Timedelta(1, "ns"), as_of=window_end)
        out: list[Fill] = []
        elig = int(pd.Timestamp(order.eligible_at or window_start).value)

        class _Ex:
            eligible_ns = elig
            activate_at_ns = None
            evaluated = False

        ex = _Ex()
        mkt = SymbolMarket(inst)
        qs = [QuoteView(int(t.value), float(r["bid"]), float(r["ask"]), float(r.get("bid_size", np.nan)),
                        float(r.get("ask_size", np.nan))) for t, r in df.iterrows()]

        def _orders():
            return [(order, ex)] if abs(order.remaining) > 1e-9 else []

        def _on_fill(d):
            order.filled_qty += d.qty
            out.append(Fill(order.id, inst, pd.Timestamp(d.ts_ns, tz="UTC"), d.qty, d.price,
                            CostBreakdown(spread_cost=d.spread_cost, impact_cost=d.impact_cost),
                            d.fidelity, d.base, d.liquidity, "; ".join(d.reasons)))

        self.simulate(mkt, qs, _orders, _on_fill, lambda o, r: None)
        return out


# --------------------------------------------------------------------------
# Auction fill model
# --------------------------------------------------------------------------
@dataclass
class AuctionConfig:
    moo_cutoff: pd.Timedelta = pd.Timedelta(minutes=2)     # NYSE/Nasdaq MOO entry cutoff 09:28 ET
    moc_cutoff: pd.Timedelta = pd.Timedelta(minutes=10)    # NYSE MOC entry cutoff 15:50 ET
    participation_cap: float = 0.10
    open_auction_share: float = 0.02     # opening cross ~1-3% of daily volume (daily-bar mode)
    close_auction_share: float = 0.08    # closing cross ~7-10% of daily volume (daily-bar mode)


class AuctionFillModel:
    def __init__(self, cfg: Optional[AuctionConfig] = None, impact: Optional[SquareRootImpact] = None,
                 allow_fractional: bool = False):
        self.cfg = cfg or AuctionConfig()
        self.impact = impact or SquareRootImpact()
        self.allow_fractional = allow_fractional

    def fill(self, order: Order, price: float, auction_volume: float, ts_ns: int, mkt: SymbolMarket,
             is_open: bool, daily_bar: bool) -> Optional[FillDecision]:
        if not (price == price and price > 0):
            return None
        rem = abs(order.remaining)
        vol = auction_volume * ((self.cfg.open_auction_share if is_open else self.cfg.close_auction_share)
                                if daily_bar else 1.0)
        cap = self.cfg.participation_cap * vol if vol == vol and vol > 0 else INF
        q = min(rem, cap)
        if not self.allow_fractional:
            q = math.floor(q + 1e-9)
        if q <= 0:
            return None
        buy = order.quantity > 0
        imp = self.impact.impact_fraction(q * price, mkt.adv_dollar, mkt.sigma)
        if order.limit_price is not None:   # LOO/LOC-style protection if provided
            if buy and price > order.limit_price or (not buy) and price < order.limit_price:
                return None
        px = price * (1 + imp) if buy else price * (1 - imp)
        if imp > 0:
            px = round_to_tick(px, valid_tick(px, mkt.instrument), "up" if buy else "down")
        mult = mkt.instrument.multiplier
        reasons = ["opening auction" if is_open else "closing auction"]
        if daily_bar:
            reasons.append("auction volume estimated as share of daily volume")
        partial = q < rem - 1e-9
        if partial:
            reasons.append("auction participation cap: remainder cancelled")
        return FillDecision(order, 0.0 if is_open else 3.0, ts_ns, math.copysign(q, order.quantity), px, price,
                            "AUCTION", 0.0, abs(px - price) * q * mult, Fidelity.T1_BAR, tuple(reasons), partial)


# --------------------------------------------------------------------------
# Options / combos from quote snapshots
# --------------------------------------------------------------------------
@dataclass
class LegQuote:
    instrument: Instrument
    ts: int
    bid: float
    ask: float
    fidelity: Fidelity
    note: str = ""


@dataclass
class ComboConfig:
    price_improvement: float = 0.0             # fraction of (natural -> mid) captured, 0 = natural
    snapshot_tolerance: pd.Timedelta = pd.Timedelta(0)
    max_spread_frac: float = 2.0               # skip quotes with spread > 200% of mid (bad prints)
    reject_locked: bool = True                 # locked/crossed (ask <= bid) snapshots: wait for the next valid one
    stock_quote_max_age: pd.Timedelta = pd.Timedelta(seconds=5)   # stock leg quote sync window


class ComboFillModel:
    """Multi-leg (and single-leg option) fills from one quote snapshot.

    Per leg natural price: buy at ask, sell at bid. With
    ``price_improvement=f`` each leg moves a fraction f from natural toward
    mid. Package net per unit (ratios applied, +debit/-credit as defined by
    the legs) must satisfy the order limit: buying the package (qty>0) fills
    when net <= limit; selling (qty<0) when net >= limit. All legs must come
    from the same snapshot (identical ts within tolerance) or the order is
    rejected."""

    def __init__(self, cfg: Optional[ComboConfig] = None):
        self.cfg = cfg or ComboConfig()

    def evaluate(self, order: Order, legs: Sequence[tuple[Leg, Optional[LegQuote]]], eligible_ns: int
                 ) -> tuple[str, list[tuple[Instrument, float, float, float, Fidelity]], float, str]:
        """Returns (status, leg_fills, net, reason). status in
        {'FILL','WAIT','REJECT'}; leg_fills = (instrument, signed qty, price,
        mid, fidelity)."""
        if any(q is None for _, q in legs):
            return "WAIT", [], float("nan"), "missing leg quote"
        ts = [q.ts for _, q in legs]
        if max(ts) <= eligible_ns:
            return "WAIT", [], float("nan"), "no snapshot after order became eligible"
        if min(ts) <= eligible_ns:
            return "WAIT", [], float("nan"), "stale leg quote"
        if max(ts) - min(ts) > self.cfg.snapshot_tolerance.value:
            return "REJECT", [], float("nan"), "legs not from the same snapshot"
        units = abs(order.quantity - order.filled_qty) if order.legs else abs(order.remaining)
        units = math.floor(units + 1e-9)
        if units <= 0:
            return "WAIT", [], float("nan"), "nothing to fill"
        sgn = 1 if order.quantity > 0 else -1
        f = self.cfg.price_improvement
        out = []
        net = 0.0
        for leg, q in legs:
            if not (q.bid >= 0 and q.ask > 0 and q.ask >= q.bid):
                return "WAIT", [], float("nan"), f"invalid quote {leg.instrument.symbol}"
            if self.cfg.reject_locked and leg.instrument.is_option and q.bid > 0 and q.ask <= q.bid:
                return "WAIT", [], float("nan"), f"locked/crossed quote {leg.instrument.symbol}: waiting"
            mid = 0.5 * (q.bid + q.ask)
            if mid > 0 and (q.ask - q.bid) / mid > self.cfg.max_spread_frac:
                return "WAIT", [], float("nan"), f"quote too wide {leg.instrument.symbol}"
            eff_ratio = leg.ratio * sgn
            natural = q.ask if eff_ratio > 0 else q.bid
            px = natural + f * (mid - natural)
            net += leg.ratio * px
            out.append((leg.instrument, eff_ratio * units, px, mid, q.fidelity))
        lim = order.limit_price
        if lim is not None:
            if sgn > 0 and net > lim + 1e-12:
                return "WAIT", [], net, "limit not reached"
            if sgn < 0 and net < lim - 1e-12:
                return "WAIT", [], net, "limit not reached"
        return "FILL", out, net, "package filled at " + ("natural" if f == 0 else f"natural+{f:.0%} to mid")
