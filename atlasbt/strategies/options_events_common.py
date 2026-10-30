"""Shared machinery for the options event / skew / short-dated strategies S158-S172
(``atlasbt.strategies.options_events``).

Module-wide conventions (declared once, apply to every class of the family)
---------------------------------------------------------------------------
* Option prices come ONLY from the portal's chain snapshots / option quote
  snapshots (``DataPortal.option_chain`` / ``option_quote``), never from a model,
  unless the run opts in with ``allow_synthetic=True`` (see below). Option orders
  are COMBO orders (all legs from the SAME snapshot, natural prices: buy at ask /
  sell at bid); the engine fills them at the FIRST snapshot strictly after the
  order became eligible - a decision at time t never fills at t's quotes.
* Chain fidelity gate: each strategy declares the worst chain fidelity it accepts
  (``CHAIN_MAX_FID``): intraday / event strategies need timestamped executable
  quotes (T0/T1: e.g. a keyed Databento OPRA provider); EOD-signal strategies
  accept real end-of-day snapshots (T2, DoltHub). A chain worse than that, a
  SYNTHETIC chain (``attrs['synthetic']`` or fidelity column 3), or a snapshot
  older than the decision tolerance is refused (the decision is skipped and
  logged) - never silently replaced.
* ``allow_synthetic=True`` (explicit opt-in, never the default): accept whatever
  chain the portal returns (synthetic T3 chains, stale or EOD snapshots for
  intraday decisions) for MECHANICS / SENSITIVITY runs only. The options
  requirements become non-critical and every such decision is flagged
  ``Fidelity.T3_MODELED``. NOTE: the engine fills option orders only from
  ``portal.option_quotes`` snapshots; the production portal serves REAL snapshots
  only, so contracts that exist only in a synthetic chain will not fill there.
* "Action at HH:MM ET" = the decision is taken on the bar that ENDS at HH:MM; the
  order is eligible 1 s later and fills at the next snapshot. Early-close
  sessions: afternoon times (>= 12:00) shift with the close (15:30 -> 12:30).
* Delta hedges (S167, S170): share market orders submitted at the daily close
  from that close's option deltas; they fill at the next bar (next open).
* Daily-signal strategies (S165-S170) evaluate at the session close on the close
  snapshot; IV metrics are recomputed from mids on the parity forward
  (``atlasbt.options.metrics.chain_metrics``), never vendor IVs. Percentiles and
  z-scores use the previous ``LOOKBACK`` (252) VALID daily observations of the
  metric (sessions without a usable real snapshot are skipped, not filled); the
  search walks back at most 2 x LOOKBACK + 10 sessions (cap), else the value is NaN.
  Constant-maturity values flagged ``extrap_flat`` are treated as missing.
* When a held contract is missing from the real snapshot (DoltHub drops contracts
  below ~13 DTE), exit DECISIONS use ``options.marking.mark_contract`` on the same
  snapshot (T2 surface / T3 extrapolated; flagged); the closing order itself can
  only fill on a real quote, otherwise the position is settled by the engine's
  option lifecycle at expiry (flagged T3 with reason).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import pandas as pd

from ..core.clock import et, session_date_of, to_utc
from ..core.types import (AssetClass, Fidelity, Instrument, Leg, Order, OrderStatus, OrderType, Right,
                          TimeInForce)
from ..engine.strategy import Requirement, Strategy
from ..indicators import percentile_rank, rolling_std, session_vwap, sma
from ..options.chain import QualityRules, SelectionError, prepare_chain, quality_mask
from ..options.marking import mark_contract
from ..options.metrics import chain_metrics
from ..options.structures import OptLeg, Package
from ..pricing.bsm import bsm_greeks
from ..pricing.surface import constant_maturity_iv
from ..pricing.time import simple_to_cc
from ..risk import sizing as SZ

ACTIVE = (OrderStatus.PENDING, OrderStatus.WORKING, OrderStatus.PARTIAL)

ETF_CORE = ["SPY", "QQQ", "IWM", "DIA"]
SECTOR_SPDR = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]
#: ETF option universe for the daily vol-signal strategies: liquid ETFs whose real EOD chains exist
#: (DoltHub: SPY, DIA, the 11 sector SPDRs; QQQ/IWM are absent there).
OPT_ETF = ["SPY", "DIA"] + SECTOR_SPDR
BROAD_ETFS = {"SPY", "DIA", "QQQ", "IWM", "IVV", "VOO"}

COMMON_DEVIATIONS = (
    "Options family convention: option orders are COMBO orders filled by the engine at the first real quote "
    "snapshot after the decision (natural prices); no model fills.",
    "Chains worse than the strategy's fidelity gate, synthetic chains or stale snapshots are refused "
    "(decision skipped and logged); allow_synthetic=True (opt-in, never default) accepts them, flagged T3_MODELED.",
    "Afternoon decision times shift with early closes (15:30 -> 12:30 on 13:00 closes).",
)


# ============================================================================ time helpers
def label_ts(cal, session, label: str) -> pd.Timestamp:
    """UTC instant of ET wall-clock ``label`` on ``session``; afternoon labels shift with early closes."""
    t = et(session, label)
    if int(label[:2]) >= 12 and cal.is_early_close(session):
        _, c = cal.open_close(session)
        t = t - (et(session, "16:00") - c)
    return t


def at_label(ctx, label: str) -> bool:
    return ctx.now == label_ts(ctx.calendar(), ctx.session_date(), label)


def session_close(ctx, session=None) -> pd.Timestamp:
    return ctx.calendar().open_close(session if session is not None else ctx.session_date())[1]


def is_close_bar(ctx) -> bool:
    return ctx.now >= session_close(ctx)


# ============================================================================ package state
@dataclass
class PkgState:
    key: str
    und: str
    pkg: Package
    units: int
    kind: str                          # 'credit' | 'debit' | 'arb'
    open_order: Order
    opened_at: pd.Timestamp
    meta: dict = field(default_factory=dict)
    insts: list = field(default_factory=list)       # option Instruments of the package legs
    stock_legs: list = field(default_factory=list)  # (Instrument, shares per unit)
    filled_units: float = 0.0
    entry_cash: float = 0.0            # USD paid (+) / received (-) for the filled units, all legs
    close_order: Optional[Order] = None
    close_reason: str = ""
    last_mark_flag: str = ""

    @property
    def multiplier(self) -> float:
        return self.pkg.multiplier

    @property
    def entry_net(self) -> float:
        """Net price per unit in option-price units (+debit / -credit) from actual fills."""
        if self.filled_units <= 0:
            return float("nan")
        return self.entry_cash / (self.filled_units * self.multiplier)


def leg_instruments(pkg: Package) -> list[Instrument]:
    return [l.instrument() for l in pkg.legs]


# ============================================================================ series selection
def pick_series(x: pd.DataFrame, expiry, settlement: Optional[str] = None, root: Optional[str] = None) -> pd.DataFrame:
    """Rows of ONE expiry series (expiry + root + settlement passed explicitly to
    ``options.chain.slice_expiry``). When several series share the date and none is
    requested, the standard (monthly/quarterly) series is preferred, then the root name."""
    s = x[x["expiry"] == pd.Timestamp(expiry)]
    if settlement is not None and "settlement" in s.columns:
        s = s[s["settlement"].astype(str) == str(settlement)]
    if root is not None and "root" in s.columns:
        s = s[s["root"].astype(str) == str(root)]
    if s.empty:
        raise SelectionError(f"no series for expiry {pd.Timestamp(expiry).date()}")
    keys = [c for c in ("root", "settlement") if c in s.columns]
    if keys:
        cand = s[keys + (["kind"] if "kind" in s.columns else [])].astype(str).drop_duplicates()
        if "kind" in cand.columns:
            cand = cand.assign(_std=(~cand["kind"].isin(["MONTHLY", "QUARTERLY", "LEAPS"])).astype(int))
            cand = cand.sort_values(["_std"] + keys, kind="mergesort")
        else:
            cand = cand.sort_values(keys, kind="mergesort")
        first = cand.iloc[0]
        root = first["root"] if "root" in keys else root
        settlement = first["settlement"] if "settlement" in keys else settlement
        root = None if str(root) in ("nan", "None", "") else root
        settlement = None if str(settlement) in ("nan", "None", "") else settlement
    from ..options.chain import slice_expiry
    return slice_expiry(x, expiry, root=root, settlement=settlement)


def choose_series(x: pd.DataFrame, target_dte: float, min_dte: Optional[int], max_dte: Optional[int],
                  settlement: Optional[str] = None, root: Optional[str] = None) -> pd.DataFrame:
    from ..options.chain import choose_expiry
    exp = choose_expiry(x, target_dte, min_dte, max_dte, settlement=settlement, root=root)
    return pick_series(x, exp, settlement, root)


# ============================================================================ statistics on metric history
def pct_rank_prev(values, lookback: int) -> float:
    """Mid-rank percentile (0-100) of the LAST value vs the previous ``lookback`` values."""
    s = pd.Series(np.asarray(values, float))
    if len(s) < lookback + 1 or not np.isfinite(s.iloc[-1]):
        return float("nan")
    r = percentile_rank(s, lookback, include_current=False)
    v = float(np.asarray(r)[-1])
    return v * 100.0 if np.isfinite(v) else float("nan")


def z_prev(values, lookback: int) -> float:
    """z-score of the LAST value vs mean / sample stdev of the previous ``lookback`` values."""
    s = pd.Series(np.asarray(values, float))
    if len(s) < lookback + 1 or not np.isfinite(s.iloc[-1]):
        return float("nan")
    ref = s.shift(1)
    mu = float(np.asarray(sma(ref, lookback))[-1])
    sd = float(np.asarray(rolling_std(ref, lookback, ddof=1))[-1])
    if not (np.isfinite(mu) and np.isfinite(sd)) or sd <= 0:
        return float("nan")
    return (float(s.iloc[-1]) - mu) / sd


def resample_bars(b: pd.DataFrame, minutes: int, now: pd.Timestamp, session_open: pd.Timestamp) -> pd.DataFrame:
    """Causal resample of today's RTH 1m bars into ``minutes`` buckets anchored at the open;
    only buckets whose END <= now are returned (the in-progress bucket is dropped)."""
    if b is None or len(b) == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    td = pd.Timedelta(minutes=minutes)
    k = ((b.index - session_open) // td).astype(int)
    g = b.groupby(np.asarray(k))
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()})
    out.index = pd.DatetimeIndex([session_open + i * td for i in out.index], name="start")
    out = out[out.index + td <= now]
    out["session"] = "RTH"
    return out


# ============================================================================ base class
class OptionsEventBase(Strategy):
    """Common services: fidelity-gated chain access, package book, combo entry /
    exit, quote-based package valuation, IV-metric history, delta hedging."""

    asset = "OPT"
    #: worst acceptable chain fidelity for decisions (subclasses override)
    CHAIN_MAX_FID: Fidelity = Fidelity.T2_PROXY
    #: max age of the chain snapshot at a decision (None: same session for daily strategies)
    QUOTE_MAX_AGE: Optional[pd.Timedelta] = None
    #: worst chain fidelity accepted for the DAILY CLOSE metric history (IV percentile / z-score inputs);
    #: None = CHAIN_MAX_FID. An intraday-decision strategy may accept real EOD snapshots (T2) for the
    #: history of a daily statistic while its decisions stay on timestamped quotes (S160).
    HISTORY_MAX_FID: Optional[Fidelity] = None
    #: trailing window of the IV-metric statistics
    LOOKBACK: int = 252
    QUALITY = QualityRules(max_rel_spread=0.50)

    def __init__(self, **params):
        super().__init__(**params)
        self.allow_synthetic = bool(self.params.get("allow_synthetic", False))
        if self.allow_synthetic:
            self.requires = [replace(r, critical=False, min_fidelity=Fidelity.T3_MODELED) if r.kind == "options"
                             else r for r in type(self).requires]
        self.book: dict[str, PkgState] = {}
        self._oid: dict[int, tuple[str, str]] = {}
        self._flags: set = set()
        self._metrics: dict[str, dict] = {}
        self._done: set = set()

    # ------------------------------------------------------------------ bookkeeping
    def flag_once(self, ctx, tier: Fidelity, reason: str) -> None:
        if reason not in self._flags:
            self._flags.add(reason)
            ctx.flag_fidelity(tier, reason)

    #: runner activity-diagnostics categories
    ACTIVITY = ("selection_failure", "combo_cancelled", "order_rejected", "no_signal", "data_missing", "skip")

    def skip(self, ctx, what: str, activity: Optional[str] = None, **kv) -> None:
        """Log a skipped decision, tagged with an activity category (inferred when not given)."""
        if activity is None:
            w = what.lower()
            if "rejected" in w:
                activity = "order_rejected"
            elif any(k in w for k in ("chain", "snapshot", "fidelity", "synthetic", "no underlying", "unavailable",
                                      "no usable", "quotes", "no volume", "predates")):
                activity = "data_missing"
            else:
                activity = "skip"
        ctx.log(f"{self.id} skip: {what}", activity=activity, **kv)

    # ------------------------------------------------------------------ rates / spot
    def rate_cc(self, ctx, T: float = 0.25, as_of=None) -> float:
        try:
            r = ctx.portal().risk_free_rate(as_of, "3M") if as_of is not None else ctx.portal().risk_free_rate(tenor="3M")
        except Exception:
            r = float("nan")
        if not np.isfinite(r):
            self.flag_once(ctx, Fidelity.T2_PROXY, "risk-free rate unavailable: r = 0 used for option analytics")
            return 0.0
        return float(simple_to_cc(r, T))

    @staticmethod
    def chain_spot(ch: pd.DataFrame) -> float:
        if "underlying_price" in ch.columns:
            u = pd.to_numeric(ch["underlying_price"], errors="coerce")
            u = u[np.isfinite(u)]
            if len(u):
                return float(np.median(u))
        return float("nan")

    def spot(self, ctx, und: str, ch: Optional[pd.DataFrame] = None) -> float:
        s = self.chain_spot(ch) if ch is not None and len(ch) else float("nan")
        if np.isfinite(s):
            return s
        try:
            return float(ctx.mark(und))
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------ chains
    @staticmethod
    def chain_fidelity(ch: pd.DataFrame) -> int:
        if "fidelity" in ch.columns and len(ch):
            f = pd.to_numeric(ch["fidelity"], errors="coerce")
            if f.notna().any():
                return int(f.max())
        return int(ch.attrs.get("fidelity", Fidelity.T2_PROXY))

    def _enrich(self, ctx, ch: pd.DataFrame) -> pd.DataFrame:
        """Fill exercise/settlement/root/multiplier from the instrument master when a
        portal's chain frame lacks them (test portals); real chains carry them."""
        need = [c for c in ("exercise", "settlement", "root") if c not in ch.columns or ch[c].isna().any()]
        if not need or "symbol" not in ch.columns:
            return ch
        ch = ch.copy()
        ex, st, rt = [], [], []
        for s in ch["symbol"]:
            try:
                inst = ctx.instrument(s)
            except Exception:
                inst = None
            ok = inst is not None and inst.is_option
            ex.append(inst.exercise.value if ok and inst.exercise else "AMERICAN")
            st.append(inst.settlement.value if ok and inst.settlement else "PHYSICAL")
            rt.append(s[:6].strip() if ok else None)
        for c, vals in (("exercise", ex), ("settlement", st), ("root", rt)):
            if c not in ch.columns:
                ch[c] = vals
            else:
                ch[c] = ch[c].where(ch[c].notna(), pd.Series(vals, index=ch.index))
        return ch

    def get_chain(self, ctx, und: str, as_of=None, max_fid: Optional[Fidelity] = None,
                  max_age: Optional[pd.Timedelta] = None, session=None, quiet: bool = False
                  ) -> Optional[pd.DataFrame]:
        """Fidelity-gated chain snapshot (PIT). Returns None (logged) when refused."""
        max_fid = self.CHAIN_MAX_FID if max_fid is None else max_fid
        now = ctx.now if as_of is None else to_utc(as_of)
        try:
            ch = ctx.portal().option_chain(und, as_of=as_of) if as_of is not None else ctx.portal().option_chain(und)
        except Exception as e:
            if not quiet:
                self.skip(ctx, f"{und}: chain error {type(e).__name__}: {e}", activity="selection_failure")
            return None
        if ch is None or len(ch) == 0:
            if not quiet:
                self.skip(ctx, f"{und}: no chain")
            return None
        fid = self.chain_fidelity(ch)
        synthetic = bool(ch.attrs.get("synthetic", False)) or fid >= int(Fidelity.T3_MODELED)
        ts = pd.to_datetime(ch["ts"], utc=True).max() if "ts" in ch.columns else now
        stale = False
        if max_age is not None:
            stale = (now - ts) > max_age
        if session is not None and session_date_of(ts) != pd.Timestamp(session):
            stale = True
        bad = synthetic or fid > int(max_fid) or stale
        if bad and not self.allow_synthetic:
            if not quiet:
                why = "synthetic chain" if synthetic else (f"chain fidelity {Fidelity(fid).name} worse than "
                                                           f"{Fidelity(max_fid).name}" if fid > int(max_fid)
                                                           else f"stale snapshot {ts}")
                self.skip(ctx, f"{und}: {why} refused")
            return None
        if bad:
            self.flag_once(ctx, Fidelity.T3_MODELED, f"{self.id}: allow_synthetic opt-in: decisions on synthetic/"
                                                    f"stale/low-fidelity chains (mechanics only)")
        ch = self._enrich(ctx, ch)
        ch.attrs["snapshot_ts"] = ts
        return ch

    def prepared(self, ctx, ch: pd.DataFrame, und: str, now=None) -> Optional[pd.DataFrame]:
        now = ctx.now if now is None else now
        S = self.spot(ctx, und, ch)
        if not np.isfinite(S) or S <= 0:
            self.skip(ctx, f"{und}: no underlying price")
            return None
        r = self.rate_cc(ctx)
        try:
            x = prepare_chain(ch, now, r=r, spot=None if np.isfinite(self.chain_spot(ch)) else S)
        except Exception as e:
            self.skip(ctx, f"{und}: prepare_chain failed {e}", activity="selection_failure")
            return None
        x = x[(x["dte"] >= 0) & (x["T"] > 0)].copy()
        # greeks missing from the snapshot (gamma/vega/theta): BSM on the IV used for delta (documented)
        miss = [c for c in ("gamma", "vega", "theta") if c not in x.columns or x[c].isna().any()]
        if miss and len(x):
            iv = pd.to_numeric(x["iv_used"], errors="coerce").values
            g = bsm_greeks(x["spot"].values, x["strike"].values, x["T"].values, r, 0.0, iv, x["right"].values == "C")
            for c in miss:
                comp = getattr(g, c)
                x[c] = comp if c not in x.columns else np.where(x[c].isna(), comp, x[c])
        x.attrs.update(ch.attrs)
        x.attrs["r_cc"] = r
        return x

    def legs_pass(self, xe: pd.DataFrame, pkg: Package) -> bool:
        """True iff every leg of ``pkg`` passes the quote-quality (spread) filter."""
        u = self.usable(xe)
        for l in pkg.legs:
            m = np.isclose(xe["strike"], l.strike) & (xe["right"] == l.right) & (xe["expiry"] == pd.Timestamp(l.expiry))
            idx = xe.index[m]
            if len(idx) == 0 or not bool(u.loc[idx[0]]):
                return False
        return True

    def usable(self, x: pd.DataFrame) -> pd.Series:
        rules = QualityRules(max_rel_spread=float(self.params.get("max_rel_spread", self.QUALITY.max_rel_spread)))
        return quality_mask(x, rules)

    # ------------------------------------------------------------------ orders / book
    def open_package(self, ctx, key: str, und: str, pkg: Package, units: int, kind: str,
                     limit: Optional[float] = None, meta: Optional[dict] = None,
                     stock_legs: Optional[list] = None, tag: str = "",
                     tif: TimeInForce = TimeInForce.DAY) -> Optional[PkgState]:
        if units <= 0:
            self.skip(ctx, f"{key}: size 0")
            return None
        ok, why = self.verify_legs(ctx, pkg)
        if not ok:
            self.skip(ctx, f"{key}: contract verification failed ({why})")
            return None
        legs = list(pkg.core_legs())
        for inst, sh in (stock_legs or []):
            legs.append(Leg(inst, int(sh)))
        order = Order(None, float(units), OrderType.COMBO, limit_price=limit, tif=tif,
                      legs=tuple(legs), tag=tag or f"{self.id}:open")
        o = ctx.submit(order)
        if o.status == OrderStatus.REJECTED:
            self.skip(ctx, f"{key}: entry rejected ({o.reject_reason})")
            return None
        st = PkgState(key, und, pkg, int(units), kind, o, ctx.now, dict(meta or {}), leg_instruments(pkg),
                      list(stock_legs or []))
        self.book[key] = st
        self._oid[o.id] = (key, "open")
        ctx.log(f"{self.id} open {key}", pkg=pkg.describe(), units=units, kind=kind, **{k: v for k, v in (meta or {}).items()
                                                                                       if isinstance(v, (int, float, str))})
        return st

    @staticmethod
    def verify_legs(ctx, pkg: Package) -> tuple[bool, str]:
        """OPT_* sizing: 'integer contracts, multiplier/deliverable verified'. Every leg is checked
        against the instrument master (standard 100 multiplier / 100-share deliverable); a leg the
        master does not know as an option is checked as built from the chain row."""
        for inst in leg_instruments(pkg):
            try:
                m = ctx.instrument(inst.symbol)
            except Exception:
                m = None
            ref = m if (m is not None and m.is_option) else inst
            ok, why = SZ.verify_contract(ref, expected_multiplier=100.0, expected_deliverable=100.0)
            if not ok:
                return False, f"{inst.symbol}: {why}"
        return True, ""

    def on_fill(self, ctx, fill) -> None:
        ref = self._oid.get(fill.order_id)
        if ref is None:
            return
        key, what = ref
        st = self.book.get(key)
        if st is None or what != "open":
            return
        st.entry_cash += fill.quantity * fill.price * fill.instrument.multiplier
        if fill.instrument.is_option and st.insts and fill.instrument.symbol == st.insts[0].symbol:
            r0 = st.pkg.legs[0].ratio
            st.filled_units += abs(fill.quantity / r0)

    def positions_of(self, ctx, st: PkgState) -> dict[str, float]:
        return {i.symbol: ctx.position(i.symbol) for i in st.insts}

    def is_open(self, ctx, st: PkgState) -> bool:
        return any(abs(q) > 1e-9 for q in self.positions_of(ctx, st).values())

    def refresh(self, ctx) -> None:
        """Drop packages whose entry never filled or that are fully closed / expired."""
        for key in list(self.book):
            st = self.book[key]
            pos_open = self.is_open(ctx, st)
            entry_active = st.open_order.status in ACTIVE
            close_active = st.close_order is not None and st.close_order.status in ACTIVE
            if not pos_open and not entry_active and not close_active:
                if st.filled_units <= 0:
                    stt = st.open_order.status
                    ctx.log(f"{self.id} {key}: entry not filled ({stt.value})",
                            activity="order_rejected" if stt == OrderStatus.REJECTED else "combo_cancelled")
                elif not st.close_reason:
                    ctx.log(f"{self.id} {key} closed by option lifecycle (expiry/assignment)")
                self.book.pop(key)
                self.on_package_closed(ctx, st)
                continue
            if st.close_order is not None and not close_active and pos_open:
                st.close_order = None          # closing order expired/rejected unfilled: retry at next decision

    def on_package_closed(self, ctx, st: PkgState) -> None:
        pass

    def close_package(self, ctx, key: str, reason: str) -> Optional[Order]:
        st = self.book.get(key)
        if st is None or (st.close_order is not None and st.close_order.status in ACTIVE):
            return None
        if st.open_order.status in ACTIVE:
            ctx.cancel(st.open_order.id)
        pos = self.positions_of(ctx, st)
        legs = []
        for inst in st.insts:
            q = pos.get(inst.symbol, 0.0)
            if abs(q) > 1e-9:
                legs.append(Leg(inst, -int(round(q))))
        for inst, _sh in st.stock_legs:
            q = ctx.position(inst.symbol)
            if abs(q) > 1e-9:
                legs.append(Leg(inst, -int(round(q))))
        if not legs:
            return None
        o = ctx.submit(Order(None, 1.0, OrderType.COMBO, tif=TimeInForce.DAY, legs=tuple(legs),
                             tag=f"{self.id}:close:{reason}"))
        if o.status == OrderStatus.REJECTED:
            self.skip(ctx, f"{key}: close rejected ({o.reject_reason})")
            return None
        st.close_order = o
        st.close_reason = reason
        self._oid[o.id] = (key, "close")
        ctx.log(f"{self.id} close {key}", reason=reason)
        return o

    def open_option_debit(self, ctx) -> float:
        return float(sum(max(st.entry_cash, 0.0) if st.filled_units > 0 else
                         max(st.meta.get("planned_debit_usd", 0.0), 0.0) for st in self.book.values()
                         if st.kind == "debit"))

    # ------------------------------------------------------------------ valuation
    def leg_quotes(self, ctx, st: PkgState, max_age: Optional[pd.Timedelta]) -> Optional[dict]:
        out = {}
        for inst in st.insts:
            try:
                q = ctx.portal().option_quote(inst)
            except Exception:
                q = None
            if q is None or not (np.isfinite(q.bid) and np.isfinite(q.ask)) or q.ask <= 0 or q.bid < 0 \
                    or q.ask < q.bid:
                return None
            if max_age is not None and ctx.now - pd.Timestamp(q.ts) > max_age:
                return None
            out[inst.symbol] = (float(q.bid), float(q.ask), int(q.fidelity))
        return out

    def liquidation_value(self, ctx, st: PkgState, max_age: Optional[pd.Timedelta] = None,
                          x: Optional[pd.DataFrame] = None) -> tuple[float, int, str]:
        """USD received (+) / paid (-) to close the CURRENT option positions at natural
        prices (sell longs at bid, buy shorts at ask). Quotes first; with a prepared
        chain ``x`` a missing contract is marked from the same snapshot's surface."""
        pos = self.positions_of(ctx, st)
        qs = self.leg_quotes(ctx, st, max_age)
        total, worst, how = 0.0, 0, "quotes"
        for inst in st.insts:
            q = pos.get(inst.symbol, 0.0)
            if abs(q) < 1e-9:
                continue
            if qs is not None:
                b, a, f = qs[inst.symbol]
            else:
                if x is None:
                    return float("nan"), int(Fidelity.NA), "no quotes"
                try:
                    m = mark_contract(x, ctx.now, float(x.attrs.get("r_cc", 0.0)), inst.expiry, inst.strike,
                                      inst.right.value, exercise=(inst.exercise.value if inst.exercise else "AMERICAN"),
                                      settlement=(inst.settlement.value if inst.settlement else "PHYSICAL"),
                                      underlying=inst.underlying)
                except Exception as e:
                    return float("nan"), int(Fidelity.NA), f"mark failed {e}"
                if not np.isfinite(m.mid):
                    return float("nan"), int(Fidelity.NA), m.reason
                b, a, f = m.bid, m.ask, int(m.fidelity)
                how = "surface mark: " + m.reason
            px = b if q > 0 else a
            total += q * px * inst.multiplier
            worst = max(worst, f)
        return total, worst, how

    # ------------------------------------------------------------------ IV metric history
    def _metrics_at(self, ctx, und: str, session, as_of) -> Optional[dict]:
        ch = self.get_chain(ctx, und, as_of=as_of, session=session, quiet=True, max_fid=self.HISTORY_MAX_FID)
        if ch is None:
            return None
        if self.HISTORY_MAX_FID is not None and self.chain_fidelity(ch) > int(self.CHAIN_MAX_FID):
            self.flag_once(ctx, Fidelity.T2_PROXY, f"{self.id}: daily IV history from real EOD chain snapshots "
                                                   f"(capture minute approximate, T2)")
        S = self.chain_spot(ch)
        if not np.isfinite(S):
            try:
                b = ctx.portal().bars(und, "1d", as_of - pd.Timedelta(days=7), as_of + pd.Timedelta(1, "ns"))
                S = float(b["close"].iloc[-1]) if len(b) else float("nan")
            except Exception:
                S = float("nan")
        if not np.isfinite(S):
            return None
        r = self.rate_cc(ctx, as_of=as_of)
        try:
            m = chain_metrics(ch, as_of, r, spot=S, american=None)
        except Exception:
            return None
        if (m.get("synthetic") or int(m.get("fidelity", 0) or 0) >= int(Fidelity.T3_MODELED)) \
                and not self.allow_synthetic:
            return None
        atm = m.get("atm_by_expiry")
        out = {"session": pd.Timestamp(session), "spot": S}
        ok = lambda flag: flag in ("interp", "exact")
        out["atm_iv_30"] = m.get("atm_iv_30", np.nan) if ok(m.get("flag_30")) else np.nan
        out["atm_iv_60"] = m.get("atm_iv_60", np.nan) if ok(m.get("flag_60")) else np.nan
        out["skew25_30"] = m.get("skew25_30", np.nan) if ok(m.get("flag_skew")) else np.nan
        out["atm"] = atm
        return out

    def metrics_today(self, ctx, und: str) -> Optional[dict]:
        return self.metrics_for_session(ctx, und, ctx.session_date())

    def metrics_for_session(self, ctx, und: str, session) -> Optional[dict]:
        cache = self._metrics.setdefault(und, {})
        session = pd.Timestamp(session)
        if session not in cache:
            as_of = session_close(ctx, session)
            if as_of > ctx.now:
                return None
            cache[session] = self._metrics_at(ctx, und, session, as_of)
        return cache[session]

    def metric_series(self, ctx, und: str, name: str, extra: int = 0, fn=None) -> pd.Series:
        """Valid daily values of metric ``name`` (or ``fn(metrics)``): today (if computable) plus the
        previous LOOKBACK(+extra) VALID observations, found by walking back session by session
        (sessions without a usable real snapshot are skipped) up to a cap of
        2 x (LOOKBACK + extra) + 10 sessions; oldest first."""
        cal = ctx.calendar()
        today = ctx.session_date()
        need = self.LOOKBACK + extra + 1
        cap = 2 * (self.LOOKBACK + extra) + 10
        try:
            first = cal.prev_session(today, cap)
        except Exception:
            first = today
        sessions = list(cal.sessions(first, today))
        vals = {}
        for s in reversed(sessions):
            m = self.metrics_for_session(ctx, und, s)
            if m is not None:
                v = fn(m) if fn is not None else m.get(name, np.nan)
                if v is not None and np.isfinite(v):
                    vals[s] = float(v)
            if len(vals) >= need:
                break
        # keep the cache bounded
        cache = self._metrics.get(und, {})
        for s in [k for k in cache if k < first]:
            cache.pop(s, None)
        return pd.Series(vals, dtype=float).sort_index()

    @staticmethod
    def cm_iv(m: dict, tenor_days: float) -> float:
        atm = m.get("atm") if m else None
        if atm is None or len(atm) == 0:
            return float("nan")
        v, flag = constant_maturity_iv(atm, tenor_days)
        return float(v) if flag in ("interp", "exact") else float("nan")

    # ------------------------------------------------------------------ delta hedging
    def option_delta_shares(self, ctx, und: str, x: Optional[pd.DataFrame]) -> tuple[float, bool]:
        """Share-equivalent delta of all option positions on ``und`` (from the prepared
        chain's deltas; missing contracts marked from the same snapshot)."""
        tot, ok = 0.0, True
        for st in self.book.values():
            for inst in st.insts:
                if (inst.underlying or "").upper() != und.upper():
                    continue
                q = ctx.position(inst.symbol)
                if abs(q) < 1e-9:
                    continue
                d = float("nan")
                if x is not None:
                    hit = x[(x["expiry"] == pd.Timestamp(inst.expiry)) & np.isclose(x["strike"], inst.strike)
                            & (x["right"] == inst.right.value)]
                    if len(hit):
                        d = float(hit["delta_used"].iloc[0])
                    else:
                        try:
                            m = mark_contract(x, ctx.now, float(x.attrs.get("r_cc", 0.0)), inst.expiry, inst.strike,
                                              inst.right.value, underlying=inst.underlying)
                            d = m.delta
                            self.flag_once(ctx, Fidelity(int(max(m.fidelity, Fidelity.T2_PROXY))),
                                           f"{self.id}: hedge delta of contracts missing from the snapshot "
                                           f"from the same-day surface")
                        except Exception:
                            d = float("nan")
                if not np.isfinite(d):
                    ok = False
                    continue
                tot += q * d * inst.multiplier
        return tot, ok

    def hedge_to_neutral(self, ctx, und: str, x: Optional[pd.DataFrame], tag: str = "delta_hedge") -> float:
        from ..options.structures import hedge_shares
        opt_d, ok = self.option_delta_shares(ctx, und, x)
        if not ok:
            self.skip(ctx, f"{und}: hedge skipped (delta unavailable)")
            return 0.0
        cur = ctx.position(und)
        pend = sum(o.remaining for o in ctx.open_orders(und) if o.instrument is not None
                   and o.instrument.symbol == und)
        q = hedge_shares(opt_d, cur + pend, band_shares=0.0, round_lot=1.0)
        if q != 0:
            ctx.order(und, q, tag=tag)
        return q


def underlying_class(und: str) -> str:
    return "broad_etf" if und.upper() in BROAD_ETFS else ("broad_index" if und.upper() in ("SPX", "SPXW", "XSP")
                                                          else "equity")
