"""Shared machinery for the options directional / income family (S126-S157).

Everything option-specific goes through the OPTIONS agent's public API
(docs/OPTIONS.md): ``options.chain`` (prepare / quality / expiry & delta
selection), ``options.structures`` (package builders, natural prices, max P/L,
COMBO orders), ``options.marking`` (marks for contracts missing from the
snapshot), ``options.metrics`` (constant-maturity ATM IV), ``options.margin``
(Cboe requirement for OPT_NAKED sizing) and ``pricing.realized`` (realized
vol, HAR-RV forecasts). Sizing goes through ``atlasbt.risk.sizing``.

Mechanics shared by every strategy of the family
------------------------------------------------
* Daily timeline. Signals are evaluated at the session close on completed,
  split/dividend adjusted daily bars; option selection uses the latest chain
  snapshot with ts <= now (``ctx.portal().option_chain``: real DoltHub EOD
  quotes (T2) where they exist, else the synthetic T3 chain).
* Every option order is ONE ``COMBO`` order (all legs, market, GTC). The engine
  fills it at natural prices (buy ask / sell bid) from the FIRST snapshot after
  submission ("next valid quote timestamp"); all legs from the same snapshot.
  Closing a package is one combo that reverses every remaining leg.
* Open/close state is tracked from the strategy's own combo fills (``on_fill``)
  and reconciled every bar against ``ctx.positions()`` so that expirations,
  exercises and (early) assignments booked by the engine's option lifecycle are
  seen even if no ``on_fill`` is delivered for them.
* Marks for P&L based exits are snapshot mids (contract quote; contracts
  missing from the snapshot are marked from the same snapshot's surface with
  ``options.marking.mark_contract`` and flagged T2/T3).
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..core.clock import session_date_of
from ..core.types import Fidelity, Instrument, Leg, Order, OrderStatus, OrderType, Right, TimeInForce
from ..engine.strategy import Requirement, Strategy
from ..indicators import donchian, ema, percentile_rank, sma
from ..options import structures as st
from ..options.chain import SelectionError, choose_expiry, prepare_chain, quality_mask, slice_expiry
from ..options.listing import monthly_expiration_day
from ..options.margin import package_margin
from ..options.marking import mark_contract
from ..options.metrics import chain_metrics
from ..options.plugins import underlying_class
from ..options.synthetic import INDEX_MAP, SPX_LIKE
from ..pricing.realized import close_to_close, daily_variance, har_rv_forecast
from ..pricing.time import simple_to_cc
from ..risk import sizing as sz

# ------------------------------------------------------------------ universes
#: Default underlying: SPY is the only liquid index ETF with free real chains (DoltHub, 2020+).
DEFAULT_UNDERLYING = ("SPY",)
#: Declared static mega-cap list for the single-stock income strategies (opt-in via
#: ``use_megacap=True``). Names with DoltHub chains; static list => survivorship bias
#: possible (recorded in the result meta by the engine).
MEGACAP = ("AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "XOM", "JNJ", "KO")
ETF_SYMBOLS = {"SPY", "QQQ", "IWM", "DIA", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
               "XLRE", "XLC", "TLT", "GLD", "EFA", "EEM", "VOO", "IVV", "RSP"}
INDEX_UNDERLYINGS = {"SPX", "SPXW", "XSP", "NDX", "RUT", "DJX"}
STANDARD_KINDS = ("MONTHLY", "QUARTERLY", "LEAPS")

COMMON_DEVIATIONS = [
    "Daily timeline: signals at the session close on completed split/dividend-adjusted daily bars; option "
    "orders are single COMBO market orders (GTC) filled by the engine at natural prices from the FIRST chain "
    "snapshot after submission (real DoltHub EOD snapshots are stamped at the close and arrive ~3x/week, so the "
    "fill can be 1-3 sessions after the signal). Stock legs (covered/hedged strategies) are separate market orders "
    "filled at the next open unless stated otherwise.",
    "Expiry choice: options.chain.choose_expiry (min |DTE-target| inside the DTE window, earlier expiry on ties, "
    "standard monthly before weekly). Grid DTE values shift the sheet's baseline window by the same offset "
    "(e.g. baseline 25-35 around 30 -> 21 uses 16-26, 45 uses 40-50). DTE = calendar days from the ET session date.",
    "Strike choice: options.chain.select_by_delta / nearest_atm / wing_by_points on quality-filtered quotes "
    "(two-sided, nonzero bid, rel. spread <= 50%); deltas are the chain's quoted deltas, else BSM from the quoted "
    "IV, else from the mid IV. No listed contract satisfying the rule -> no trade (SelectionError, logged).",
    "P&L / credit-captured exits are evaluated at the daily close on snapshot MID marks of every leg and executed "
    "at the next snapshot at natural prices. Contracts absent from the snapshot are marked from the same snapshot's "
    "surface (options.marking: T2 inside the quoted range, T3 extrapolated) and flagged.",
    "Unfilled opening combos are cancelled after 5 sessions; closing combos stay working (GTC) until filled or the "
    "position is settled by the engine's option lifecycle (expiry / exercise / assignment), which is reconciled "
    "every bar from ctx.positions().",
    "One open package per underlying at a time; holding periods count sessions after the fill session; "
    "'exit after N days' submits the closing combo at the close of the N-th session after the fill session.",
    "Implied-vol signals use REAL chain quotes only (options.metrics.chain_metrics, constant-maturity ATM IV on the "
    "parity forward); synthetic chain smiles are never used as signals. iv_source='auto' falls back to the Cboe "
    "vol index of the underlying (VIX for SPX/SPY, VXN/RVX/VXD... via options.synthetic.INDEX_MAP) when the "
    "current snapshot is synthetic (flagged T2_PROXY); percentiles are computed within one source only.",
    "Realized-vol forecasts ('predeclared model') = walk-forward HAR-RV (pricing.realized.har_rv_forecast, Corsi "
    "2009) on Garman-Klass + overnight-gap daily variance of adjusted bars, horizon 20 sessions, 252-session "
    "minimum training, refit every 21 sessions; 'trailing 20d realized vol' = close-to-close 20d (annualized).",
]


def finite(*xs) -> bool:
    return all(x is not None and isinstance(x, (int, float, np.floating, np.integer)) and np.isfinite(x) for x in xs)


def dte_window(target: float, offsets: tuple[int, int]) -> tuple[int, int]:
    """Window [target+lo, target+hi] (offsets from the sheet baseline)."""
    return int(target + offsets[0]), int(target + offsets[1])


def crossed_above(a_prev, a, b_prev, b) -> bool:
    return finite(a_prev, a, b_prev, b) and a_prev <= b_prev and a > b


def crossed_below(a_prev, a, b_prev, b) -> bool:
    return finite(a_prev, a, b_prev, b) and a_prev >= b_prev and a < b


def vol_index_for(und: str) -> Optional[str]:
    """Cboe 30-day vol index of an underlying (VIX for SPX-like, mapped index otherwise)."""
    u = und.upper()
    if u in SPX_LIKE:
        return "VIX"
    return INDEX_MAP.get(u)


def package_stress_loss(pkg: st.Package, spot: float, scenario: sz.StressScenario = sz.DEFAULT_STRESS) -> float:
    """Worst loss (USD per package unit) under risk.sizing's predefined OPT_NAKED scenario applied to ALL legs
    at once: underlying moves by each scenario move, every leg's extrinsic value (natural entry price minus
    intrinsic) is multiplied by ``vol_up_extrinsic_mult``. For a single short option this equals
    ``sizing.stress_loss_short_option`` exactly (tested)."""
    worst = 0.0
    for m in scenario.moves:
        s1 = spot * (1 + m)
        pnl = 0.0
        for leg in pkg.legs:
            p0 = leg.ask if leg.ratio > 0 else leg.bid
            if not finite(p0):
                p0 = leg.mid
            i0 = float(leg.intrinsic(spot))
            extr = max(p0 - i0, 0.0)
            val = float(leg.intrinsic(s1)) + extr * scenario.vol_up_extrinsic_mult
            pnl += leg.ratio * (val - p0)
        for s in pkg.stock:
            pnl += s.shares * (s1 - spot) / pkg.multiplier
        worst = max(worst, -pnl)
    return worst * pkg.multiplier


# ------------------------------------------------------------------ chain view
@dataclass
class ChainView:
    und: str
    raw: pd.DataFrame          # canonical chain (underlying_price filled)
    x: pd.DataFrame            # prepare_chain output
    usable: pd.Series          # quality mask aligned with x
    spot: float
    r: float
    synthetic: bool
    ts: pd.Timestamp
    fidelity: Fidelity

    def expiry_slice(self, expiry) -> pd.DataFrame:
        return slice_expiry(self.x, expiry)


@dataclass
class OptPos:
    """One option package held (or being opened/closed) by the strategy."""

    key: str
    und: str
    name: str
    legs: list                          # [(Instrument, ratio per unit)]
    units: int
    status: str = "OPENING"             # OPENING | OPEN | CLOSING | CLOSED
    open_oid: int = -1
    close_oid: Optional[int] = None
    submitted_session: Optional[pd.Timestamp] = None
    entry_session: Optional[pd.Timestamp] = None
    entry_net: float = float("nan")     # per unit, + debit / - credit (option-price units)
    planned_net: float = float("nan")
    filled: dict = field(default_factory=dict)       # symbol -> filled qty (open)
    cost: float = 0.0                   # sum qty*price of opening fills
    close_reason: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def multiplier(self) -> float:
        return float(self.legs[0][0].multiplier) if self.legs else 100.0

    @property
    def expiry(self) -> pd.Timestamp:
        """Nearest expiry among the option legs."""
        return min(pd.Timestamp(i.expiry) for i, _ in self.legs if i.is_option)

    def option_legs(self) -> list:
        return [(i, r) for i, r in self.legs if i.is_option]

    @property
    def credit(self) -> float:
        """Credit received per unit (> 0 for credit packages)."""
        return -self.entry_net

    def symbols(self) -> list[str]:
        return [i.symbol for i, _ in self.legs if i.is_option]


# ------------------------------------------------------------------ base class
class OptionBase(Strategy):
    """Base for S126-S157. Subclasses implement :meth:`step`."""

    bar_freq = "1d"
    asset = "OPT"
    family_deviations: tuple = ()
    #: underlying param name holding the traded underlyings
    HIST_N = 400
    OPEN_TIMEOUT_SESSIONS = 5
    #: True for strategies that hold the underlying by design (covered / cash-secured / hedged). Others flatten
    #: any shares delivered by exercise/assignment at the next session open.
    HOLDS_STOCK = False

    def __init__(self, **params: Any):
        super().__init__(**params)
        p = self.params
        if p.get("use_megacap"):
            unds = list(MEGACAP)
        else:
            u = p.get("underlyings", p.get("underlying", DEFAULT_UNDERLYING))
            unds = [u] if isinstance(u, str) else list(u)
        self.universe = [s.upper() for s in unds]
        self.requires = self.build_requires()
        self.reset_state()

    # ------------------------------------------------------------ requirements
    def build_requires(self) -> list:
        req = [Requirement("bars", "1d", symbols=tuple(self.universe))]
        real = tuple(u for u in self.universe if u not in INDEX_UNDERLYINGS)
        if real:
            req.append(Requirement("options", symbols=real, min_fidelity=Fidelity.T3_MODELED,
                                   note="option chain snapshots (DoltHub EOD T2; synthetic T3 fallback)"))
        idx = tuple(u for u in self.universe if u in INDEX_UNDERLYINGS)
        if idx:
            req.append(Requirement("bars", "1d", symbols=("VIX",),
                                   note="index options only via the synthetic T3 chain (VIX family term structure)"))
        return req + list(self.extra_requires())

    def extra_requires(self) -> list:
        return []

    # ------------------------------------------------------------ state
    def reset_state(self) -> None:
        self.positions: dict[str, list[OptPos]] = defaultdict(list)
        self.oid_map: dict[int, tuple[OptPos, str]] = {}
        self.collateral = sz.CollateralBook()
        self.seq = 0
        self.cv_cache: dict = {}
        self.events_seen: list = []
        self.store: dict = {}

    def init(self, ctx) -> None:
        self.reset_state()

    def live(self, und: str) -> list[OptPos]:
        return [p for p in self.positions.get(und, []) if p.status != "CLOSED"]

    def first_live(self, und: str) -> Optional[OptPos]:
        lv = self.live(und)
        return lv[0] if lv else None

    # ------------------------------------------------------------ callbacks
    def on_bar(self, ctx) -> None:
        for und in self.universe:
            self.reconcile(ctx, und)
            try:
                self.step(ctx, und)
            except SelectionError as e:
                self.activity(ctx, f"{und}: no trade (selection: {e})", "selection_failure")

    ACTIVITY_CATEGORIES = ("selection_failure", "combo_cancelled", "order_rejected", "no_signal", "data_missing",
                           "skip")

    def activity(self, ctx, msg: str, category: str = "skip") -> None:
        """Surface why a strategy did not trade: ctx.log(msg, activity=<category>) (runner activity interface)
        plus an aggregated T1 fidelity note 'activity <SID>: ...' (numbers stripped so counts aggregate)."""
        if category not in self.ACTIVITY_CATEGORIES:
            category = "skip"
        ctx.log(f"{self.id} {msg}", activity=category)
        generic = re.sub(r"\d+(\.\d+)?", "#", msg)
        generic = re.sub(r"\b[A-Z]{1,6}\s+#[CP]#", "<contract>", generic)
        ctx.flag_fidelity(Fidelity.T1_BAR, f"activity {self.id}: {generic[:160]}")

    # ------------------------------------------------------------ delivered shares (option-only strategies)
    def on_session_start(self, ctx) -> None:
        if self.HOLDS_STOCK:
            return
        for und in self.universe:
            self.reconcile(ctx, und)
            for pos in self.live(und):
                if pos.status == "OPEN" and pos.meta.get("partial"):
                    self.close_package(ctx, pos, "leg_settled")
            self.close_delivered(ctx, und)

    def pending_stock(self, ctx, und: str) -> float:
        return sum(o.remaining for o in ctx.open_orders(und) if o.instrument is not None and not o.instrument.is_option)

    def close_delivered(self, ctx, und: str) -> None:
        """Option-only strategies never hold the underlying: shares delivered by an (early) assignment or an
        exercise are sold/bought back with a market order at the session open; the stock P&L is booked to the
        package that delivered them (meta['delivered'], ['delivery_cost'], ['delivery_proceeds'])."""
        q = ctx.position(und)
        pend = self.pending_stock(ctx, und)
        if abs(q + pend) < 1e-9:
            return
        o = ctx.order(und, -(q + pend), tag="close_delivered")
        if o.status == OrderStatus.REJECTED:
            self.activity(ctx, f"{und}: close of delivered shares rejected ({o.reject_reason})", "order_rejected")
            return
        pos = self.delivery_owner(und)
        if pos is not None:
            self.oid_map[o.id] = (pos, "deliv")

    def delivery_owner(self, und: str) -> Optional["OptPos"]:
        lst = [p for p in self.positions.get(und, []) if p.meta.get("delivered")] or self.positions.get(und, [])
        return lst[-1] if lst else None

    def step(self, ctx, und: str) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def on_fill(self, ctx, fill) -> None:
        liq = fill.liquidity
        if liq in ("ASSIGNMENT", "EXERCISE", "SETTLEMENT"):
            self.events_seen.append((fill.ts, liq, fill.instrument.symbol, fill.quantity))
            if not fill.instrument.is_option:
                und = fill.instrument.symbol
                owner = None
                for p in reversed(self.positions.get(und, [])):
                    if p.status != "OPENING":
                        owner = p
                        break
                if owner is not None:
                    owner.meta["delivered"] = owner.meta.get("delivered", 0.0) + fill.quantity
                    owner.meta["delivery_cost"] = owner.meta.get("delivery_cost", 0.0) + fill.quantity * fill.price
            self.on_lifecycle_fill(ctx, fill)
            return
        hit = self.oid_map.get(fill.order_id)
        if hit is None:
            return
        pos, side = hit
        sym = fill.instrument.symbol
        if side == "open":
            pos.filled[sym] = pos.filled.get(sym, 0.0) + fill.quantity
            if fill.instrument.is_option:
                pos.cost += fill.quantity * fill.price
            if all(abs(pos.filled.get(i.symbol, 0.0)) >= abs(r * pos.units) - 1e-9 for i, r in pos.legs):
                pos.status = "OPEN"
                pos.entry_net = pos.cost / pos.units
                pos.entry_session = ctx.session_date()
                self.on_opened(ctx, pos)
        elif side == "deliv":
            pos.meta["delivery_proceeds"] = pos.meta.get("delivery_proceeds", 0.0) - fill.quantity * fill.price
            pos.meta["delivery_pnl"] = pos.meta["delivery_proceeds"] - pos.meta.get("delivery_cost", 0.0)
        else:
            pos.meta["close_filled"] = pos.meta.get("close_filled", 0.0) + abs(fill.quantity)

    def on_lifecycle_fill(self, ctx, fill) -> None:
        """Hook: engine lifecycle fill (expiry settlement / exercise / assignment)."""

    def on_opened(self, ctx, pos: OptPos) -> None:
        """Hook called once all opening legs are filled."""

    def on_closed(self, ctx, pos: OptPos) -> None:
        """Hook called when a package is fully gone (closed by us or settled)."""

    # ------------------------------------------------------------ reconciliation
    def reconcile(self, ctx, und: str) -> None:
        sess = ctx.session_date()
        for pos in self.positions.get(und, []):
            if pos.status == "CLOSED":
                continue
            held = {i.symbol: ctx.position(i.symbol) for i, _ in pos.option_legs()}
            if pos.status == "OPENING":
                o_live = any(o.id == pos.open_oid for o in ctx.open_orders())
                if not o_live and all(abs(q) < 1e-9 for q in held.values()):
                    self.finish(ctx, pos, "open_not_filled")
                    continue
                if o_live and pos.submitted_session is not None:
                    n = len(ctx.calendar().sessions(pos.submitted_session, sess)) - 1
                    if n >= self.OPEN_TIMEOUT_SESSIONS:
                        ctx.cancel(pos.open_oid)
                        if all(abs(q) < 1e-9 for q in held.values()):
                            self.activity(ctx, f"{und}: opening combo cancelled unfilled after "
                                               f"{self.OPEN_TIMEOUT_SESSIONS} sessions (contract not quoted / limit)",
                                          "combo_cancelled")
                            self.finish(ctx, pos, "open_timeout")
                continue
            gone = [abs(q) < 1e-9 for q in held.values()]
            if not all(gone) and any(gone):
                pos.meta["partial"] = True          # one leg expired / assigned / exercised early
                if pos.status == "CLOSING" and pos.close_oid is not None:
                    # the working closing combo still references the settled leg -> replace it
                    if any(o.id == pos.close_oid for o in ctx.open_orders()):
                        ctx.cancel(pos.close_oid)
                    pos.status, pos.close_oid = "OPEN", None
            if all(gone):
                if pos.close_oid is not None and any(o.id == pos.close_oid for o in ctx.open_orders()):
                    ctx.cancel(pos.close_oid)
                why = pos.close_reason or "settled"
                if pos.status != "CLOSING":
                    why = "expired_or_assigned"
                self.finish(ctx, pos, why)
            elif pos.status == "CLOSING" and pos.close_oid is not None and \
                    not any(o.id == pos.close_oid for o in ctx.open_orders()):
                # closing combo cancelled/rejected while legs still held -> retry next bar
                pos.status = "OPEN"
                pos.close_oid = None

    def finish(self, ctx, pos: OptPos, reason: str) -> None:
        pos.status = "CLOSED"
        if not pos.close_reason:
            pos.close_reason = reason
        pos.meta["closed_session"] = ctx.session_date()
        self.collateral.release(pos.key)
        self.on_closed(ctx, pos)

    # ------------------------------------------------------------ data helpers
    def hist(self, ctx, und: str, n: Optional[int] = None) -> pd.DataFrame:
        return ctx.history(und, n or self.HIST_N, "1d", adjusted=True)

    def last_raw_close(self, ctx, und: str) -> float:
        h = ctx.history(und, 1, "1d", adjusted=False)
        return float(h["close"].iloc[-1]) if h is not None and len(h) else float("nan")

    def rate(self, ctx) -> float:
        try:
            r = float(ctx.portal().risk_free_rate())
        except Exception:
            r = float("nan")
        return float(simple_to_cc(r, 0.25)) if np.isfinite(r) else 0.0

    def chain(self, ctx, und: str) -> Optional[ChainView]:
        key = (und, ctx.now)
        if key in self.cv_cache:
            return self.cv_cache[key]
        self.cv_cache = {k: v for k, v in self.cv_cache.items() if k[1] == ctx.now}
        cv = None
        try:
            ch = ctx.portal().option_chain(und, min_dte=0, max_dte=800)
        except Exception as e:  # data layer failure -> no trade, logged
            self.activity(ctx, f"{und}: option_chain failed: {e!r}", "data_missing")
            ch = None
        if ch is not None and len(ch):
            raw = ch.copy()
            spot = float("nan")
            if "underlying_price" in raw.columns:
                up = pd.to_numeric(raw["underlying_price"], errors="coerce")
                if up.notna().any():
                    spot = float(np.nanmedian(up))
            if not np.isfinite(spot):
                spot = self.last_raw_close(ctx, und)
            if np.isfinite(spot) and spot > 0:
                if "underlying_price" not in raw.columns:
                    raw["underlying_price"] = spot
                raw["underlying_price"] = pd.to_numeric(raw["underlying_price"], errors="coerce").fillna(spot)
                r = self.rate(ctx)
                fid_col = pd.to_numeric(raw["fidelity"], errors="coerce") if "fidelity" in raw.columns else None
                fmax = int(fid_col.max()) if fid_col is not None and fid_col.notna().any() else int(Fidelity.T2_PROXY)
                synthetic = bool(ch.attrs.get("synthetic", False)) or fmax >= int(Fidelity.T3_MODELED)
                x = prepare_chain(raw, ctx.now, r=r)
                usable = quality_mask(x)
                ts = pd.Timestamp(pd.to_datetime(raw["ts"]).max()) if "ts" in raw.columns else ctx.now
                cv = ChainView(und, raw, x, usable, spot, r, synthetic, ts, Fidelity(min(max(fmax, 0), 3)))
        self.cv_cache[key] = cv
        return cv

    def note_chain_use(self, ctx, cv: ChainView, what: str) -> None:
        if cv.synthetic:
            ctx.flag_fidelity(Fidelity.T3_MODELED, f"{what}: synthetic {cv.und} chain (T3_MODELED, options.synthetic)")

    def pick_expiry(self, cv: ChainView, target: float, lo: int, hi: int, standard_only: bool = False,
                    root: Optional[str] = None) -> pd.Timestamp:
        x = cv.x[cv.usable]
        if standard_only and "kind" in x.columns:
            x = x[x["kind"].astype(str).isin(STANDARD_KINDS)]
        if root is not None and "root" in x.columns:
            x = x[x["root"].astype(str) == root]
        return choose_expiry(x, target, lo, hi)

    def series_of(self, cv: ChainView, expiry, root: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
        """(root, settlement) of the expiry SERIES to trade on ``expiry``: the requested root if given, else the
        standard (monthly) series if several share the date (e.g. SPX AM vs SPXW PM), else the first in
        alphabetical order. Always passed explicitly to slice_expiry (never mixes AM/PM legs)."""
        x = cv.x[cv.usable]
        s = x[x["expiry"] == pd.Timestamp(expiry)]
        if root is not None and "root" in s.columns:
            s = s[s["root"].astype(str) == root]
        r = root if "root" in cv.x.columns else None
        st_ = None
        keys = [c for c in ("root", "settlement") if c in s.columns]
        if keys and len(s):
            combos = s[keys + (["kind"] if "kind" in s.columns else [])].astype(str).drop_duplicates()
            if "kind" in combos.columns and len(combos) > 1:
                std = combos[combos["kind"].isin(STANDARD_KINDS)]
                combos = std if len(std) else combos
            combos = combos.sort_values(keys)
            first = combos.iloc[0]
            r = str(first["root"]) if "root" in keys else r
            st_ = str(first["settlement"]) if "settlement" in keys else None
        return r, st_

    def slice(self, cv: ChainView, expiry, root: Optional[str] = None) -> tuple[pd.DataFrame, pd.Series]:
        r, st_ = self.series_of(cv, expiry, root)
        xe = slice_expiry(cv.x, expiry, root=r, settlement=st_)
        return xe, cv.usable.reindex(xe.index).fillna(False).astype(bool)

    # ------------------------------------------------------------ marks
    def leg_row(self, cv: ChainView, inst: Instrument) -> Optional[pd.Series]:
        x = cv.x
        m = (x["expiry"] == pd.Timestamp(inst.expiry).normalize()) & np.isclose(x["strike"].astype(float), inst.strike) \
            & (x["right"] == inst.right.value)
        if "root" in x.columns:
            root = inst.symbol[:6].strip()
            mr = m & (x["root"].astype(str) == root)
            if mr.any():
                m = mr
        d = x[m]
        return d.iloc[0] if len(d) else None

    def leg_mark(self, ctx, cv: ChainView, inst: Instrument) -> tuple[float, float, float, float]:
        """(mid, bid, ask, delta) of a held contract from the snapshot (surface mark if not quoted)."""
        row = self.leg_row(cv, inst)
        if row is not None and finite(float(row["ask"])) and float(row["ask"]) > 0 and float(row["bid"]) >= 0:
            return float(row["mid"]), float(row["bid"]), float(row["ask"]), float(row["delta_used"])
        mk = mark_contract(cv.raw, ctx.now, cv.r, inst.expiry, float(inst.strike), inst.right.value,
                           exercise=inst.exercise.value if inst.exercise else "AMERICAN",
                           settlement=inst.settlement.value if inst.settlement else "PHYSICAL", underlying=cv.und)
        if mk.fidelity == Fidelity.NA or not finite(mk.mid):
            return float("nan"), float("nan"), float("nan"), float("nan")
        ctx.flag_fidelity(mk.fidelity, f"mark of {inst.symbol} not quoted in snapshot: {mk.reason}")
        return mk.mid, mk.bid, mk.ask, mk.delta

    def package_value(self, ctx, cv: Optional[ChainView], pos: OptPos, held_only: bool = True) -> float:
        """Mid value per unit (sum ratio*mid over legs still held)."""
        if cv is None:
            return float("nan")
        tot = 0.0
        for inst, ratio in pos.option_legs():
            if held_only and abs(ctx.position(inst.symbol)) < 1e-9:
                continue
            mid, *_ = self.leg_mark(ctx, cv, inst)
            if not finite(mid):
                return float("nan")
            tot += ratio * mid
        return tot

    def package_delta_shares(self, ctx, cv: Optional[ChainView], pos: OptPos) -> float:
        """Share-equivalent delta of the option legs currently held."""
        if cv is None:
            return float("nan")
        d = 0.0
        for inst, _ in pos.option_legs():
            q = ctx.position(inst.symbol)
            if abs(q) < 1e-9:
                continue
            *_, delta = self.leg_mark(ctx, cv, inst)
            if not finite(delta):
                return float("nan")
            d += q * delta * inst.multiplier
        return d

    def dte(self, ctx, expiry) -> int:
        return int((pd.Timestamp(expiry).normalize() - ctx.session_date()).days)

    def sessions_since(self, ctx, session) -> int:
        if session is None:
            return 0
        return len(ctx.calendar().sessions(pd.Timestamp(session), ctx.session_date())) - 1

    # ------------------------------------------------------------ orders
    def open_package(self, ctx, und: str, pkg: st.Package, units: int, cv: ChainView, tag: str = "",
                     limit: Optional[float] = None, extra_legs: tuple = (), meta: Optional[dict] = None,
                     collateral: float = 0.0) -> Optional[OptPos]:
        units = int(units)
        if units <= 0:
            self.activity(ctx, f"{und}: size 0 for {pkg.name} ({self.sizing})")
            return None
        for leg in pkg.legs:
            ok, why = sz.verify_contract(leg.instrument())
            if not ok:
                self.activity(ctx, f"{und}: contract check failed ({why}) -> no trade")
                return None
        self.note_chain_use(ctx, cv, f"{self.id} entry")
        legs = list(pkg.core_legs()) + list(extra_legs)
        order = Order(None, float(units), OrderType.COMBO, limit_price=limit, tif=TimeInForce.GTC,
                      legs=tuple(legs), tag=tag or f"open_{pkg.name}")
        o = ctx.submit(order)
        if o.status == OrderStatus.REJECTED:
            self.activity(ctx, f"{und}: open rejected: {o.reject_reason}", "order_rejected")
            return None
        self.seq += 1
        pos = OptPos(key=f"{und}-{self.seq}", und=und, name=pkg.name,
                     legs=[(l.instrument, int(l.ratio)) for l in legs], units=units, open_oid=o.id,
                     submitted_session=ctx.session_date(), planned_net=pkg.natural_open(), meta=dict(meta or {}))
        pos.meta.setdefault("pkg", pkg)
        self.positions.setdefault(und, []).append(pos)
        self.oid_map[o.id] = (pos, "open")
        if collateral > 0:
            self.collateral.reserve(pos.key, collateral)
        return pos

    def close_package(self, ctx, pos: OptPos, reason: str) -> Optional[Order]:
        if pos.status != "OPEN":
            return None
        held = []
        for inst, _ in pos.legs:
            q = int(round(ctx.position(inst.symbol)))
            if q != 0 and inst.is_option:
                held.append((inst, q))
        if not held:
            return None
        units = math.gcd(*[abs(q) for _, q in held]) or 1
        legs = [Leg(inst, q // units) for inst, q in held]
        # selling ``units`` packages of the HELD ratios reverses every remaining leg in one combo
        order = Order(None, float(-units), OrderType.COMBO, tif=TimeInForce.GTC, legs=tuple(legs),
                      tag=f"exit_{reason}")
        o = ctx.submit(order)
        if o.status == OrderStatus.REJECTED:
            self.activity(ctx, f"{pos.und}: close rejected: {o.reject_reason}", "order_rejected")
            return None
        pos.status = "CLOSING"
        pos.close_oid = o.id
        pos.close_reason = reason
        self.oid_map[o.id] = (pos, "close")
        return o

    # ------------------------------------------------------------ sizing
    def open_debit_usd(self) -> float:
        tot = 0.0
        for lst in self.positions.values():
            for p in lst:
                if p.status == "CLOSED":
                    continue
                net = p.entry_net if finite(p.entry_net) else p.planned_net
                if finite(net) and net > 0:
                    tot += net * p.units * p.multiplier
        return tot

    def size_debit(self, ctx, pkg: st.Package) -> int:
        """OPT_DEBIT: <= 0.50% NAV per package, <= 5% NAV total option debit. The per-package amount is the
        natural debit, or the max theoretical loss when that is larger (ratio spreads / net-credit packages)."""
        debit = pkg.natural_open() * pkg.multiplier
        risk = debit
        if pkg.same_expiry:
            ml = pkg.max_profit_loss(pkg.natural_open())["max_loss_usd"]
            if finite(ml):
                risk = max(debit, ml)
        if not (finite(risk) and risk > 0):
            return 0
        return sz.size_opt_debit(ctx.nav, risk, existing_option_debit=self.open_debit_usd())

    def size_credit(self, ctx, pkg: st.Package) -> tuple[int, float]:
        """OPT_CREDIT: max theoretical defined loss <= 0.50% NAV; collateral (n x max loss) reserved and capped by
        cash not already reserved; packages with naked leg exposure are refused."""
        ml = pkg.max_profit_loss(pkg.natural_open())["max_loss_usd"]
        legs = [sz.OptLeg(l.instrument(), int(l.ratio), (l.ask if l.ratio > 0 else l.bid)) for l in pkg.legs]
        n, coll = sz.size_opt_credit(ctx.nav, ml, legs=legs)
        if n > 0 and finite(ml) and ml > 0:
            n = min(n, int(math.floor(max(self.collateral.available(ctx.cash), 0.0) / ml + 1e-9)))
            coll = n * ml
        return n, coll

    def size_naked(self, ctx, pkg: st.Package, spot: float, und: str) -> int:
        """OPT_NAKED: predefined stress loss (risk.sizing scenario, all legs) <= 1% NAV; Cboe strategy margin
        (options.margin) must be > 0 (never assume zero margin)."""
        stress = package_stress_loss(pkg, spot)
        cls, lev = underlying_class(und)
        margin = float(package_margin(pkg, 1, spot, underlying_class=cls, leverage=lev).requirement)
        try:
            return sz.size_opt_naked(ctx.nav, stress, margin)
        except ValueError as e:
            self.activity(ctx, f"{und}: {e}", "skip")
            return 0

    def size_cashsec(self, ctx, strike: float, und: str, spot: float, mult: float = 100.0,
                     deliverable: float = 100.0, name_cap: Optional[float] = None) -> int:
        """OPT_CASHSEC: strike x deliverable reserved per contract from unreserved cash; assignment notional +
        existing shares <= 10% NAV for single names / ETFs; broad indexes (and an explicit ``name_cap``, e.g. the
        S129 index-ETF putwrite variant) 100%. A zero size caused by the name cap is surfaced as a diagnostic."""
        cap = name_cap if name_cap is not None else (1.0 if und in INDEX_UNDERLYINGS else 0.10)
        shares = ctx.position(und) if und not in INDEX_UNDERLYINGS else 0.0
        existing = abs(shares) * spot if finite(spot) else 0.0
        n = sz.size_opt_cashsec(ctx.cash, strike, nav=ctx.nav, collateral=self.collateral, multiplier=mult,
                                deliverable=deliverable, name_cap=cap, existing_underlying_notional=existing)
        if n <= 0:
            n_cash = sz.size_opt_cashsec(ctx.cash, strike, collateral=self.collateral, multiplier=mult,
                                         deliverable=deliverable)
            if n_cash > 0:
                self.activity(ctx, f"{und}: name cap reached ({cap:.0%} NAV incl. {abs(shares):.0f} shares held) "
                                   f"-> no new cash-secured put")
        return n


    # ------------------------------------------------------------ implied-vol signals
    IV_MIN_REFS = 100        # min. real-chain snapshots in the percentile window (DoltHub ~3/week)

    def metrics(self, ctx, cv: ChainView) -> dict:
        """options.metrics.chain_metrics for a REAL snapshot (cached per snapshot ts); {} for synthetic."""
        if cv is None or cv.synthetic:
            return {}
        cache = self.store.setdefault("metrics", {})
        key = (cv.und, cv.ts)
        if key not in cache:
            try:
                cache[key] = chain_metrics(self.metrics_rows(cv.raw, ctx.now, cv.spot), ctx.now, cv.r, spot=cv.spot)
            except Exception as e:      # degenerate snapshot -> no signal
                self.activity(ctx, f"{cv.und}: chain_metrics failed: {e!r}", "data_missing")
                cache[key] = {}
        return cache[key]

    @staticmethod
    def metrics_rows(raw: pd.DataFrame, now, spot: float, days: float = 30.0, band: float = 0.20) -> pd.DataFrame:
        """Rows that determine the 30D ATM IV: the expiries bracketing ``days`` (nearest with DTE <= days and
        nearest with DTE >= days, DTE >= 1) and strikes within +-band of spot. Pure performance filter: the
        constant-maturity ATM IV (total-variance interpolation at k=0) only uses those rows."""
        if raw is None or not len(raw):
            return raw
        sess = session_date_of(now)
        exp = pd.to_datetime(raw["expiry"])
        if getattr(exp.dt, "tz", None) is not None:
            exp = exp.dt.tz_localize(None)
        dte = (exp.dt.normalize() - sess).dt.days
        ok = dte >= 1
        below = dte[ok & (dte <= days)]
        above = dte[ok & (dte >= days)]
        keep = set()
        if len(below):
            keep.add(int(below.max()))
        if len(above):
            keep.add(int(above.min()))
        m = dte.isin(list(keep))
        if finite(spot) and spot > 0:
            m &= (raw["strike"].astype(float) >= spot * (1 - band)) & (raw["strike"].astype(float) <= spot * (1 + band))
        return raw[m.to_numpy()]

    def vol_index_series(self, ctx, und: str, n: int) -> pd.Series:
        idx = vol_index_for(und)
        if idx is None:
            return pd.Series(dtype=float)
        h = ctx.history(idx, n, "1d", adjusted=False)
        if h is None or not len(h):
            return pd.Series(dtype=float)
        return h["close"].astype(float) / 100.0

    def iv_source_today(self, ctx, und: str) -> str:
        """'chain' (real snapshot), 'index' (Cboe vol-index fallback) or '' (nothing usable)."""
        mode = self.params.get("iv_source", "auto")
        cv = self.chain(ctx, und)
        real = cv is not None and not cv.synthetic
        if mode == "chain":
            return "chain" if real else ""
        if mode == "index":
            return "index" if vol_index_for(und) else ""
        return "chain" if real else ("index" if vol_index_for(und) else "")

    def atm_iv30(self, ctx, und: str) -> tuple[float, str]:
        src = self.iv_source_today(ctx, und)
        if src == "chain":
            v = self.metrics(ctx, self.chain(ctx, und)).get("atm_iv_30", float("nan"))
            return (float(v) if finite(v) else float("nan")), src
        if src == "index":
            s = self.vol_index_series(ctx, und, 1)
            if len(s):
                ctx.flag_fidelity(Fidelity.T2_PROXY, f"{und} ATM IV proxied by Cboe vol index {vol_index_for(und)} "
                                                     f"(no real chain snapshot)")
                return float(s.iloc[-1]), src
        return float("nan"), ""

    def iv_percentile(self, ctx, und: str, lookback: int = 252) -> tuple[float, float, str]:
        """(percentile in [0,1] of today's 30D ATM IV vs the PRIOR ``lookback`` sessions, today's IV, source).
        Chain source: real snapshots only (backfilled once from the PIT portal); index source: vol-index closes."""
        src = self.iv_source_today(ctx, und)
        if src == "index":
            s = self.vol_index_series(ctx, und, lookback + 1)
            if len(s) < lookback + 1:
                return float("nan"), float("nan"), src
            ctx.flag_fidelity(Fidelity.T2_PROXY, f"{und} IV percentile from Cboe vol index {vol_index_for(und)}")
            pr = percentile_rank(s.to_numpy(), lookback, include_current=False)
            return float(pr[-1]), float(s.iloc[-1]), src
        if src != "chain":
            return float("nan"), float("nan"), ""
        hist = self.chain_iv_history(ctx, und, lookback)
        cv = self.chain(ctx, und)
        today = self.metrics(ctx, cv).get("atm_iv_30", float("nan"))
        if not finite(today):
            return float("nan"), float("nan"), src
        first = ctx.calendar().prev_session(ctx.session_date(), lookback)
        refs = [v for (sess, ts), v in sorted(hist.items()) if first <= sess < ctx.session_date() and ts != cv.ts
                and finite(v)]
        need = min(self.IV_MIN_REFS, int(0.4 * lookback))
        if len(refs) < max(need, 1):
            return float("nan"), float(today), src
        arr = np.asarray(refs + [today], float)
        pr = percentile_rank(arr, len(refs), include_current=False, min_periods=max(need, 1))
        return float(pr[-1]), float(today), src

    def chain_iv_history(self, ctx, und: str, lookback: int) -> dict:
        """{(snapshot session, snapshot ts): atm_iv_30} from REAL snapshots with ts <= now. Backfilled once by
        querying the PIT portal at each prior session close (as_of <= now: no lookahead), then appended daily."""
        hist = self.store.setdefault("iv_hist", {}).setdefault(und, {})
        done = self.store.setdefault("iv_backfilled", set())
        cal = ctx.calendar()
        if und not in done:
            done.add(und)
            sess = cal.sessions(cal.prev_session(ctx.session_date(), lookback), cal.prev_session(ctx.session_date()))
            seen = set()
            for s in sess:
                as_of = cal.open_close(s)[1]
                try:
                    ch = ctx.portal().option_chain(und, as_of=as_of, min_dte=0, max_dte=800)
                except Exception:
                    continue
                if ch is None or not len(ch) or bool(ch.attrs.get("synthetic", False)):
                    continue
                if "fidelity" in ch.columns and pd.to_numeric(ch["fidelity"], errors="coerce").max() >= 3:
                    continue
                ts = pd.Timestamp(pd.to_datetime(ch["ts"]).max())
                if ts in seen:
                    continue
                seen.add(ts)
                raw = ch.copy()
                spot = float("nan")
                if "underlying_price" in raw.columns and pd.to_numeric(raw["underlying_price"], errors="coerce").notna().any():
                    spot = float(np.nanmedian(pd.to_numeric(raw["underlying_price"], errors="coerce")))
                if not np.isfinite(spot):
                    hh = ctx.portal().bars(und, "1d", ts - pd.Timedelta(days=10), ts + pd.Timedelta(seconds=1),
                                           as_of=ts, adjusted=False)
                    spot = float(hh["close"].iloc[-1]) if hh is not None and len(hh) else float("nan")
                if not np.isfinite(spot):
                    continue
                raw["underlying_price"] = pd.to_numeric(raw.get("underlying_price", spot), errors="coerce")
                raw["underlying_price"] = raw["underlying_price"].fillna(spot)
                try:
                    m = chain_metrics(self.metrics_rows(raw, ts, spot), ts, self.rate(ctx), spot=spot)
                except Exception:
                    continue
                hist[(session_date_of(ts), ts)] = float(m.get("atm_iv_30", float("nan")))
        cv = self.chain(ctx, und)
        if cv is not None and not cv.synthetic:
            v = self.metrics(ctx, cv).get("atm_iv_30", float("nan"))
            hist[(session_date_of(cv.ts), cv.ts)] = float(v) if finite(v) else float("nan")
        return hist


# ------------------------------------------------------------------ indicator snapshot
def trend_state(h: pd.DataFrame) -> dict:
    """Last/previous values of the trend indicators used by the directional rules."""
    c = h["close"]
    out = {"n": len(h), "close": float(c.iloc[-1]) if len(c) else float("nan")}
    if len(c) < 2:
        return out
    e20, e50 = ema(c, 20), ema(c, 50)
    s200, s50 = sma(c, 200), sma(c, 50)
    out.update(ema20=float(e20.iloc[-1]), ema20_prev=float(e20.iloc[-2]), ema50=float(e50.iloc[-1]),
               ema50_prev=float(e50.iloc[-2]), sma200=float(s200.iloc[-1]), sma50=float(s50.iloc[-1]),
               close_prev=float(c.iloc[-2]))
    dc = donchian(h["high"], h["low"], 20)
    out.update(dc_high=float(dc["upper"].iloc[-1]), dc_low=float(dc["lower"].iloc[-1]),
               dc_high_prev=float(dc["upper"].iloc[-2]), dc_low_prev=float(dc["lower"].iloc[-2]))
    out["ret20"] = float(c.iloc[-1] / c.iloc[-21] - 1.0) if len(c) > 20 else float("nan")
    return out


def har_forecast_vol(h: pd.DataFrame, horizon: int = 20) -> float:
    """Annualized HAR-RV forecast of the average vol over the next ``horizon`` sessions (NaN in warm-up)."""
    if len(h) < 300:
        return float("nan")
    rv = daily_variance(h, "gk")
    f = har_rv_forecast(rv, horizon=horizon, min_train=252, refit_every=21)
    v = f["vol_fcst"].iloc[-1]
    return float(v) if np.isfinite(v) else float("nan")


def trailing_rv(h: pd.DataFrame, n: int = 20) -> float:
    if len(h) < n + 1:
        return float("nan")
    v = close_to_close(h["close"], n).iloc[-1]
    return float(v) if np.isfinite(v) else float("nan")


def third_friday_session(cal, year: int, month: int) -> pd.Timestamp:
    """Standard monthly expiration session (3rd Friday, holiday -> prior session)."""
    d = pd.Timestamp(monthly_expiration_day(year, month)).normalize()
    if not cal.is_session(d):
        d = cal.prev_session(d)
    return d


__all__ = [n for n in dir() if not n.startswith("_")]
