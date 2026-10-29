"""Options directional / income strategies S126-S157 (Trading Strategy Atlas).

Shared mechanics, data rules and the deviations common to the whole family are
in :mod:`atlasbt.strategies.options_income_base` (``COMMON_DEVIATIONS``);
covered / cash-secured / wheel strategies live in
:mod:`atlasbt.strategies.options_income_stock`. Every class carries the
verbatim sheet text in ``spec`` and the sheet grid in ``grid``.

Underlying: SPY by default (the only liquid index ETF with free real chains,
DoltHub 2020+). S129/S138 name SPX: they trade SPX options through the
synthetic T3 chain (no free SPX quotes; docs/OPTIONS.md §13).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from ..core.types import Fidelity, Leg, Order, OrderStatus, OrderType, TimeInForce
from ..engine.strategy import Requirement
from ..options import structures as st
from ..options.chain import SelectionError, get_row, select_by_delta, wing_by_points
from ..pricing.surface import interp_total_variance
from .options_income_base import (
    COMMON_DEVIATIONS, ChainView, OptionBase, OptPos, crossed_above, crossed_below, dte_window, finite,
    har_forecast_vol, trailing_rv, trend_state, vol_index_for,
)
from .options_income_specs import NAMES, SPECS
from .registry import register


# ============================================================================ template
class SinglePackage(OptionBase):
    """One package per underlying: entry signal at the close -> COMBO at the next snapshot; exits evaluated at
    every close (DTE, underlying condition, P&L on snapshot mids, max hold) -> closing COMBO.

    Subclass hooks: ``entry_signal``, ``build``, ``accept``, ``underlying_exit``, ``value_exit``."""

    RISK = "debit"               # debit | credit | naked
    DTE_OFFSETS = (-5, 5)        # sheet baseline window around params['dte']
    STANDARD_ONLY = False
    ROOT: Optional[str] = None
    NEEDS_VALUE = False
    EXIT_DTE = 7                 # default; params['exit_dte'] overrides
    MAX_HOLD: Optional[int] = None

    # ---------------------------------------------------------------- hooks
    def entry_signal(self, ctx, und: str, t: dict) -> bool:
        return True

    def build(self, ctx, und: str, cv: ChainView, xe: pd.DataFrame, us: pd.Series) -> Optional[st.Package]:
        raise NotImplementedError

    def accept(self, ctx, und: str, pkg: st.Package, cv: ChainView) -> bool:
        return True

    def underlying_exit(self, ctx, und: str, pos: OptPos, t: dict) -> Optional[str]:
        return None

    def value_exit(self, ctx, pos: OptPos, v: float) -> Optional[str]:
        return None

    def exit_dte(self) -> int:
        return int(self.params.get("exit_dte", self.EXIT_DTE))

    def max_hold(self) -> Optional[int]:
        return self.params.get("max_hold", self.MAX_HOLD)

    def target_dte(self) -> float:
        return float(self.params["dte"])

    def on_open_extra(self, ctx, und: str, pos: OptPos, cv: Optional[ChainView]) -> None:
        """Called every close while the package is OPEN (e.g. delta re-hedging)."""

    def on_close_extra(self, ctx, und: str, pos: OptPos) -> None:
        """Called when the closing combo is submitted (e.g. flatten a stock hedge)."""

    # ---------------------------------------------------------------- engine
    def needs_trend(self) -> bool:
        return True

    def step(self, ctx, und: str) -> None:
        t = trend_state(self.hist(ctx, und)) if self.needs_trend() else {}
        pos = self.first_live(und)
        if pos is not None:
            if pos.status == "OPEN":
                reason = self.exit_reason(ctx, und, pos, t)
                if reason:
                    if self.close_package(ctx, pos, reason) is not None:
                        self.on_close_extra(ctx, und, pos)
                else:
                    self.on_open_extra(ctx, und, pos, self.chain(ctx, und))
            return
        if not self.entry_signal(ctx, und, t):
            return
        cv = self.chain(ctx, und)
        if cv is None:
            ctx.log(f"{self.id} {und}: signal but no option chain")
            return
        pkg = self.select(ctx, und, cv)
        if pkg is None or not self.accept(ctx, und, pkg, cv):
            return
        n, coll = self.size(ctx, und, pkg, cv)
        self.submit_entry(ctx, und, pkg, n, cv, coll)

    def select(self, ctx, und: str, cv: ChainView) -> Optional[st.Package]:
        lo, hi = dte_window(self.target_dte(), self.DTE_OFFSETS)
        exp = self.pick_expiry(cv, self.target_dte(), lo, hi, self.STANDARD_ONLY, self.root(und))
        xe, us = self.slice(cv, exp, self.root(und))
        return self.build(ctx, und, cv, xe, us)

    def root(self, und: str) -> Optional[str]:
        return self.ROOT if und in ("SPX",) else None

    def submit_entry(self, ctx, und, pkg, n, cv, coll) -> Optional[OptPos]:
        return self.open_package(ctx, und, pkg, n, cv, collateral=coll, meta=self.entry_meta(pkg))

    def entry_meta(self, pkg: st.Package) -> dict:
        m = {"strikes": [l.strike for l in pkg.legs], "natural_open": pkg.natural_open()}
        if pkg.same_expiry:
            mpl = pkg.max_profit_loss(pkg.natural_open())
            m.update(max_profit=mpl["max_profit"], max_loss=mpl["max_loss"])
        return m

    def size(self, ctx, und: str, pkg: st.Package, cv: ChainView) -> tuple[int, float]:
        if self.RISK == "debit":
            return self.size_debit(ctx, pkg), 0.0
        if self.RISK == "credit":
            return self.size_credit(ctx, pkg)
        if self.RISK == "naked":
            return self.size_naked(ctx, pkg, cv.spot, und), 0.0
        raise ValueError(self.RISK)

    def exit_reason(self, ctx, und: str, pos: OptPos, t: dict) -> Optional[str]:
        if pos.meta.get("partial"):
            return "leg_settled"
        if self.dte(ctx, pos.expiry) <= self.exit_dte():
            return f"dte_le_{self.exit_dte()}"
        r = self.underlying_exit(ctx, und, pos, t)
        if r:
            return r
        if self.NEEDS_VALUE:
            v = self.package_value(ctx, self.chain(ctx, und), pos)
            if finite(v) and finite(pos.entry_net):
                r = self.value_exit(ctx, pos, v)
                if r:
                    return r
        mh = self.max_hold()
        if mh and self.sessions_since(ctx, pos.entry_session) >= mh:
            return f"time_{mh}d"
        return None

    # ---------------------------------------------------------------- P&L helpers (per unit)
    @staticmethod
    def pnl(pos: OptPos, v: float) -> float:
        return v - pos.entry_net

    @staticmethod
    def close_cost(v: float) -> float:
        """Debit to close a credit package (per unit)."""
        return -v


def _spec(sid: str) -> str:
    return SPECS[sid]


DEBIT_DEV = "OPT_DEBIT sizing (risk.sizing.size_opt_debit) on the natural debit (or the max theoretical loss if larger)."
CREDIT_DEV = ("OPT_CREDIT sizing (risk.sizing.size_opt_credit) on the max theoretical loss at natural prices "
              "(Package.max_profit_loss); collateral = n x max loss reserved in a CollateralBook and capped by unreserved cash.")
NAKED_DEV = ("OPT_NAKED sizing: risk.sizing predefined stress scenario (+-10/20% move, extrinsic x2) applied to ALL legs "
             "at once <= 1% NAV; Cboe strategy margin (options.margin.package_margin) must be > 0; the engine's Cboe "
             "margin model handles maintenance/margin calls. Stress is a lower bound for unbounded short calls.")
CREDIT_EXITS_DEV = ("'X% credit captured' = debit to close (snapshot mids) <= (1-X) x credit received; '2x credit' stop = "
                    "debit to close >= 2 x credit received.")


# ============================================================================ S126 / S127 long single options
class _LongSingle(SinglePackage):
    sizing = "OPT_DEBIT"
    asset = "OPT"
    RIGHT = "C"
    NEEDS_VALUE = True

    def build(self, ctx, und, cv, xe, us):
        d = abs(self.params["delta"]) * (1 if self.RIGHT == "C" else -1)
        return st.single(xe, d, ratio=1, usable=us)

    def value_exit(self, ctx, pos, v):
        stop, tp = self.params.get("opt_stop") or 0.0, self.params.get("opt_tp")
        if stop and v <= pos.entry_net * (1.0 - stop):
            return f"option_stop_{int(stop * 100)}"
        if tp and v >= pos.entry_net * (1.0 + tp):
            return f"option_tp_{int(tp * 100)}"
        return None


@register
class S126(_LongSingle):
    id = "S126"; name = NAMES["S126"]; bar_freq = "1d"
    params = {"underlying": "SPY", "delta": 0.50, "dte": 30, "exit_dte": 7, "opt_stop": 0.0, "opt_tp": None}
    grid = {"delta": [0.45, 0.50, 0.55], "dte": [21, 30, 45], "opt_stop": [0.0, 0.5]}
    spec = _spec("S126")
    RIGHT = "C"
    deviations = COMMON_DEVIATIONS + [
        DEBIT_DEV,
        "'EMA20 crosses above EMA50' = EMA20_{t-1} <= EMA50_{t-1} and EMA20_t > EMA50_t (SMA-seeded EMAs on 400 "
        "adjusted closes); 'closes above SMA200' = close_t > SMA200_t; both on the signal close.",
        "Option stop/target ('tested separately'): opt_stop=0.5 closes when the call's snapshot mid <= 50% of the "
        "entry fill price; the +100% target is available as param opt_tp=1.0 but is not in the sheet grid (off "
        "in the baseline).",
        "DTE window 25-35 around 30 (grid 21 -> 16-26, 45 -> 40-50).",
    ]

    def entry_signal(self, ctx, und, t):
        return finite(t.get("sma200")) and t["close"] > t["sma200"] and \
            crossed_above(t["ema20_prev"], t["ema20"], t["ema50_prev"], t["ema50"])

    def underlying_exit(self, ctx, und, pos, t):
        if finite(t.get("ema20")) and t["close"] < t["ema20"]:
            return "close_below_ema20"
        return None


@register
class S127(_LongSingle):
    id = "S127"; name = NAMES["S127"]; bar_freq = "1d"
    params = {"underlying": "SPY", "delta": 0.50, "dte": 30, "exit_dte": 7}
    grid = {"delta": [0.40, 0.50, 0.60], "dte": [21, 30, 45]}
    spec = _spec("S127")
    RIGHT = "P"
    deviations = COMMON_DEVIATIONS + [
        DEBIT_DEV,
        "'EMA20 crosses below EMA50' = EMA20_{t-1} >= EMA50_{t-1} and EMA20_t < EMA50_t; 'closes below SMA200' = "
        "close_t < SMA200_t; delta given as absolute value (put delta -0.50).",
        "DTE window 25-35 around 30 (grid 21 -> 16-26, 45 -> 40-50).",
    ]

    def entry_signal(self, ctx, und, t):
        return finite(t.get("sma200")) and t["close"] < t["sma200"] and \
            crossed_below(t["ema20_prev"], t["ema20"], t["ema50_prev"], t["ema50"])

    def underlying_exit(self, ctx, und, pos, t):
        if finite(t.get("ema20")) and t["close"] > t["ema20"]:
            return "close_above_ema20"
        return None


# ============================================================================ S134 / S135 debit verticals
class _DebitVertical(SinglePackage):
    sizing = "OPT_DEBIT"
    asset = "OPT"
    NEEDS_VALUE = True
    SIGN = 1

    def build(self, ctx, und, cv, xe, us):
        s = self.SIGN
        return st.vertical(xe, s * abs(self.params["long_delta"]), s * abs(self.params["short_delta"]), usable=us)

    def value_exit(self, ctx, pos, v):
        d = pos.entry_net
        k = pos.meta.get("strikes", [])
        width = abs(k[0] - k[1]) if len(k) == 2 else float("nan")
        pnl = self.pnl(pos, v)
        if finite(width) and pnl >= 0.5 * (width - d):
            return "tp_50pct_max_gain"
        if pnl <= -0.5 * d:
            return "sl_50pct_debit"
        return None


_VERT_DEV = ("'50% maximum theoretical payoff gain' = P&L (snapshot mids) >= 0.5 x (strike width - debit paid); "
             "'-50% paid debit' = P&L <= -0.5 x debit.")


@register
class S134(_DebitVertical):
    id = "S134"; name = NAMES["S134"]; bar_freq = "1d"
    params = {"underlying": "SPY", "long_delta": 0.60, "short_delta": 0.30, "dte": 30, "exit_dte": 7}
    grid = {"long_delta": [0.55, 0.60], "short_delta": [0.20, 0.30], "dte": [30, 45]}
    spec = _spec("S134")
    SIGN = 1
    DTE_OFFSETS = (-5, 15)
    deviations = COMMON_DEVIATIONS + [
        DEBIT_DEV, _VERT_DEV,
        "'20d bullish breakout' = FRESH close above the prior 20-session high (close_t > max(H_{t-20..t-1}) and "
        "close_{t-1} <= max(H_{t-21..t-2})).",
        "DTE window 25-45 with target 30 (grid 45 -> 40-60).",
    ]

    def entry_signal(self, ctx, und, t):
        return finite(t.get("dc_high"), t.get("dc_high_prev")) and t["close"] > t["dc_high"] and \
            t["close_prev"] <= t["dc_high_prev"]


@register
class S135(_DebitVertical):
    id = "S135"; name = NAMES["S135"]; bar_freq = "1d"
    params = {"underlying": "SPY", "long_delta": 0.60, "short_delta": 0.30, "dte": 30, "exit_dte": 7}
    grid = {"long_delta": [0.55, 0.60], "short_delta": [0.20, 0.30], "dte": [30, 45]}
    spec = _spec("S135")
    SIGN = -1
    DTE_OFFSETS = (-5, 5)
    deviations = COMMON_DEVIATIONS + [
        DEBIT_DEV, _VERT_DEV,
        "'20d bearish breakdown' = FRESH close below the prior 20-session low; deltas as absolute values.",
        "'same 30DTE expiry' = window 25-35 (grid 45 -> 40-50).",
    ]

    def entry_signal(self, ctx, und, t):
        return finite(t.get("dc_low"), t.get("dc_low_prev")) and t["close"] < t["dc_low"] and \
            t["close_prev"] >= t["dc_low_prev"]


# ============================================================================ S136 / S137 credit verticals
class _CreditVertical(SinglePackage):
    sizing = "OPT_CREDIT"
    asset = "OPT"
    RISK = "credit"
    NEEDS_VALUE = True
    DTE_OFFSETS = (0, 15)

    def value_exit(self, ctx, pos, v):
        c, cc = pos.credit, self.close_cost(v)
        if cc <= (1 - self.params.get("tp", 0.5)) * c:
            return "tp_50pct_credit"
        if cc >= 2.0 * c:
            return "stop_2x_credit"
        return None


@register
class S136(_CreditVertical):
    id = "S136"; name = NAMES["S136"]; bar_freq = "1d"
    params = {"underlying": "SPY", "short_delta": 0.30, "wing": "delta10", "dte": 30, "exit_dte": 7,
              "vix_pct": 60.0, "tp": 0.5}
    grid = {"short_delta": [0.20, 0.30], "wing": ["pts5", "pts10", "delta10"]}
    spec = _spec("S136")
    deviations = COMMON_DEVIATIONS + [
        CREDIT_DEV, CREDIT_EXITS_DEV,
        "'VIX 20d percentile > 60' = mid-rank percentile of today's VIX close within the 20 sessions ending today "
        "(indicators.percentile_rank, include_current=True) > 0.60; VIX from portal daily bars (Cboe).",
        "Wing grid '5/10 points or .1 delta' -> wing in {pts5, pts10, delta10}; point wings use the ACTUAL listed "
        "strike at or beyond short-5/10 (wing_by_points 'outward': defined risk never understated).",
        "DTE window 30-45 (target 30).",
    ]

    def extra_requires(self):
        return [Requirement("bars", "1d", symbols=("VIX",), note="VIX 20d percentile filter")]

    def entry_signal(self, ctx, und, t):
        if not (finite(t.get("sma200")) and t["close"] > t["sma200"]):
            return False
        h = ctx.history("VIX", 20, "1d", adjusted=False)
        if h is None or len(h) < 20:
            return False
        from ..indicators import percentile_rank
        p = percentile_rank(h["close"].to_numpy(float), 20, include_current=True)[-1]
        return finite(p) and p * 100.0 > self.params["vix_pct"]

    def build(self, ctx, und, cv, xe, us):
        sd = -abs(self.params["short_delta"])
        w = self.params["wing"]
        if w == "delta10":
            return st.vertical(xe, -0.10, sd, usable=us, name="bull_put_credit")
        pts = {"pts5": 5.0, "pts10": 10.0}[w]
        return st.vertical_by_points(xe, sd, pts, usable=us)


@register
class S137(_CreditVertical):
    id = "S137"; name = NAMES["S137"]; bar_freq = "1d"
    params = {"underlying": "SPY", "short_delta": 0.30, "long_delta": 0.10, "dte": 30, "exit_dte": 7, "tp": 0.5}
    grid = {"short_delta": [0.20, 0.30], "long_delta": [0.05, 0.10]}
    spec = _spec("S137")
    deviations = COMMON_DEVIATIONS + [
        CREDIT_DEV, CREDIT_EXITS_DEV,
        "'20d return<0' = close_t / close_{t-20} - 1 < 0 on adjusted closes; DTE window 30-45 (target 30).",
    ]

    def entry_signal(self, ctx, und, t):
        return finite(t.get("sma200"), t.get("ret20")) and t["close"] < t["sma200"] and t["ret20"] < 0

    def build(self, ctx, und, cv, xe, us):
        return st.vertical(xe, abs(self.params["long_delta"]), abs(self.params["short_delta"]), usable=us,
                           name="bear_call_credit")


# ============================================================================ S138 SPX iron condor
@register
class S138(SinglePackage):
    id = "S138"; name = NAMES["S138"]; bar_freq = "1mo"; sizing = "OPT_CREDIT"; asset = "OPT"
    params = {"underlying": "SPX", "short_delta": 0.20, "wing_delta": 0.05, "dte": 30, "exit_dte": 5, "tp": 0.5,
              "hold_to_settlement": False}
    grid = {"short_delta": [0.15, 0.20, 0.25], "wing_delta": [0.03, 0.05, 0.10], "dte": [30, 45]}
    spec = _spec("S138")
    RISK = "credit"
    NEEDS_VALUE = True
    STANDARD_ONLY = True
    ROOT = "SPX"
    DTE_OFFSETS = (-7, 14)
    deviations = COMMON_DEVIATIONS + [
        CREDIT_DEV,
        "SPX options exist only as the SYNTHETIC T3 chain (no free SPX quotes, docs/OPTIONS.md §13); every fill/mark "
        "is T3. underlying='SPY' is the declared T2 deviation (real DoltHub chains 2020+).",
        "'same monthly SPX expiration' = standard AM-settled SPX monthly (root SPX, kind MONTHLY/QUARTERLY) nearest to "
        "the target DTE inside [target-7, target+14] (monthlies are 4-5 weeks apart).",
        "'Roll at 5DTE' = close the condor at DTE <= 5; the next cycle opens at the first close after the closing "
        "combo filled (flat). '50% net credit captured' = debit to close <= 50% of the credit received.",
        "'stress hold-to-settlement separately' = param hold_to_settlement=True (no 5-DTE roll; settles per the actual "
        "AM contract via the engine lifecycle); not in the sheet grid.",
        "Entries whenever flat (monthly cycle); wings by delta (.03/.05/.10) on quality-filtered quotes; collapsed or "
        "unordered strikes -> no trade.",
    ]

    def needs_trend(self):
        return False

    def exit_dte(self):
        return -10_000 if self.params.get("hold_to_settlement") else int(self.params["exit_dte"])

    def build(self, ctx, und, cv, xe, us):
        return st.iron_condor(xe, abs(self.params["short_delta"]), abs(self.params["wing_delta"]), usable=us)

    def value_exit(self, ctx, pos, v):
        if self.close_cost(v) <= (1 - self.params["tp"]) * pos.credit:
            return "tp_50pct_credit"
        return None


# ============================================================================ S139 / S140 iron butterflies
@register
class S139(SinglePackage):
    id = "S139"; name = NAMES["S139"]; bar_freq = "1mo"; sizing = "OPT_CREDIT"; asset = "OPT"
    params = {"underlying": "SPY", "width": 10.0, "dte": 30, "exit_dte": 7, "tp": 0.50}
    grid = {"width": [5.0, 10.0, 20.0], "dte": [21, 30, 45]}
    spec = _spec("S139")
    RISK = "credit"
    NEEDS_VALUE = True
    DTE_OFFSETS = (-5, 15)
    deviations = COMMON_DEVIATIONS + [
        CREDIT_DEV, CREDIT_EXITS_DEV,
        "'Close 25/50% credit captured': baseline tp=0.50; tp=0.25 available as a param (not in the sheet grid).",
        "Wings at EXACTLY +/-width on actually listed strikes (structures.iron_butterfly policy 'exact'); not listed -> "
        "no trade. Width in underlying points as given (SPY: $5/$10/$20). DTE 25-45 around 30 (grid 21 -> 16-36).",
        "Entry whenever flat (monthly cycle, no market filter in the sheet).",
    ]

    def needs_trend(self):
        return False

    def build(self, ctx, und, cv, xe, us):
        return st.iron_butterfly(xe, float(self.params["width"]), short=True, ref_price=cv.spot, usable=us,
                                 policy="exact")

    def value_exit(self, ctx, pos, v):
        c, cc = pos.credit, self.close_cost(v)
        if cc <= (1 - self.params["tp"]) * c:
            return f"tp_{int(self.params['tp'] * 100)}pct_credit"
        if cc >= 2.0 * c:
            return "stop_2x_credit"
        return None


@register
class S140(SinglePackage):
    id = "S140"; name = NAMES["S140"]; bar_freq = "1mo"; sizing = "OPT_DEBIT"; asset = "OPT"
    params = {"underlying": "SPY", "width": 10.0, "dte": 30, "exit_dte": 7, "tp_frac": 0.30}
    grid = {"width": [5.0, 10.0, 20.0], "dte": [14, 30, 45]}
    spec = _spec("S140")
    NEEDS_VALUE = True
    deviations = COMMON_DEVIATIONS + [
        DEBIT_DEV,
        "'Close at 30% of theoretical maximum spread payoff' = P&L >= 30% of the max theoretical PROFIT (width - "
        "debit); the literal 'value >= 30% of max payoff' would trigger at entry (debit is usually > 30% of width).",
        "Symmetric listed wings required (policy 'exact'); requires net debit > 0 and finite max loss. DTE window "
        "target +/-5 (sheet gives no window; grid 14/30/45).",
    ]

    def needs_trend(self):
        return False

    def build(self, ctx, und, cv, xe, us):
        return st.iron_butterfly(xe, float(self.params["width"]), short=False, ref_price=cv.spot, usable=us,
                                 policy="exact")

    def accept(self, ctx, und, pkg, cv):
        mpl = pkg.max_profit_loss(pkg.natural_open())
        return pkg.natural_open() > 0 and finite(mpl["max_loss"])

    def value_exit(self, ctx, pos, v):
        mp = pos.meta.get("max_profit", float("nan"))
        if finite(mp) and mp > 0 and self.pnl(pos, v) >= self.params["tp_frac"] * mp:
            return "tp_30pct_max"
        return None


# ============================================================================ S141 / S142 / S155 butterflies
class _LongFly(SinglePackage):
    sizing = "OPT_DEBIT"
    asset = "OPT"
    bar_freq = "1mo"
    NEEDS_VALUE = True
    DTE_OFFSETS = (-5, 15)
    RIGHT = "C"
    EXIT_DTE = 5

    def needs_trend(self):
        return False

    def build(self, ctx, und, cv, xe, us):
        return st.butterfly(xe[us], self.RIGHT, float(self.params["width"]), ref_price=cv.spot)

    def value_exit(self, ctx, pos, v):
        if v >= 0.5 * float(self.params["width"]):
            return "value_ge_50pct_max_payoff"
        return None


_FLY_DEV = ("K2 = nearest listed ATM strike with usable quotes, wings at exactly +/-width (listed, usable) else no "
            "trade; 'market value >= 50% of max terminal payoff' = package mid value >= 0.5 x width. DTE 25-45 around "
            "30 (grid 45 -> 40-60). Entry whenever flat (monthly cycle).")


@register
class S141(_LongFly):
    id = "S141"; name = NAMES["S141"]
    params = {"underlying": "SPY", "width": 10.0, "dte": 30, "exit_dte": 5}
    grid = {"width": [5.0, 10.0, 20.0], "dte": [30, 45]}
    spec = _spec("S141")
    RIGHT = "C"
    deviations = COMMON_DEVIATIONS + [DEBIT_DEV, _FLY_DEV]


@register
class S142(_LongFly):
    id = "S142"; name = NAMES["S142"]
    params = {"underlying": "SPY", "width": 10.0, "dte": 30, "exit_dte": 5}
    grid = {"width": [5.0, 10.0, 20.0], "dte": [30, 45]}
    spec = _spec("S142")
    RIGHT = "P"
    deviations = COMMON_DEVIATIONS + [DEBIT_DEV, _FLY_DEV]


@register
class S155(SinglePackage):
    id = "S155"; name = NAMES["S155"]; bar_freq = "1mo"; sizing = "OPT_CREDIT"; asset = "OPT"
    params = {"underlying": "SPY", "width": 10.0, "far_multiple": 1.5, "dte": 30, "exit_dte": 7, "tp_frac": 0.5}
    grid = {"width": [5.0, 10.0], "far_multiple": [1.5, 2.0]}
    spec = _spec("S155")
    RISK = "credit"
    NEEDS_VALUE = True
    deviations = COMMON_DEVIATIONS + [
        CREDIT_DEV,
        "K2 = nearest listed ATM call strike, K1 = K2 - width, K3 = K2 + far_multiple x width (structures.butterfly); "
        "all three listed and usable else no trade. Baseline near width 10 (sheet grid 5/10): on $1-strike "
        "underlyings width 5 x 1.5 = 7.5 is never listed -> that grid point cannot trade (no strike invented).",
        "'credit/debit reaches 50% of terminal max payout' = P&L (snapshot mids) >= 50% of the package's max "
        "theoretical profit at expiry. DTE window 25-35 ('30DTE'). Entry whenever flat (monthly cycle).",
    ]

    def needs_trend(self):
        return False

    def build(self, ctx, und, cv, xe, us):
        return st.butterfly(xe[us], "C", float(self.params["width"]), far_multiple=float(self.params["far_multiple"]),
                            ref_price=cv.spot)

    def value_exit(self, ctx, pos, v):
        mp = pos.meta.get("max_profit", float("nan"))
        if finite(mp) and mp > 0 and self.pnl(pos, v) >= self.params["tp_frac"] * mp:
            return "tp_50pct_max"
        return None


# ============================================================================ S143 / S144 long vol
@register
class S143(SinglePackage):
    id = "S143"; name = NAMES["S143"]; bar_freq = "1d"; sizing = "OPT_DEBIT"; asset = "OPT"
    params = {"underlying": "SPY", "iv_pct": 20.0, "exit_pct": 60.0, "dte": 30, "exit_dte": 7, "max_hold": 10,
              "iv_source": "auto", "iv_lookback": 252}
    grid = {"iv_pct": [10.0, 20.0, 30.0], "dte": [21, 30, 45]}
    spec = _spec("S143")
    deviations = COMMON_DEVIATIONS + [
        DEBIT_DEV,
        "'forward implied ATM volatility' = 30-day constant-maturity ATM IV on the parity forward "
        "(options.metrics.chain_metrics) of REAL snapshots; percentile = mid-rank of today's value among the real "
        "snapshots of the PRIOR 252 sessions (>= 100 snapshots required; DoltHub ~3/week -> valid from ~2021).",
        "Synthetic-only dates (pre-2020) use the VIX close percentile vs the prior 252 VIX closes (iv_source='auto' "
        "fallback, flagged T2_PROXY); iv_source='chain' disables the fallback.",
        "Exits: 10 sessions after the fill session, IV percentile > 60 (same source as today's), or DTE <= 7; both legs "
        "close in ONE combo. Straddle strike = nearest ATM listed for both rights. DTE 25-35 (grid 21 -> 16-26).",
    ]

    def extra_requires(self):
        return [Requirement("bars", "1d", symbols=("VIX",), critical=False, note="IV fallback (synthetic dates)")]

    def needs_trend(self):
        return False

    def entry_signal(self, ctx, und, t):
        p, _, _ = self.iv_percentile(ctx, und, int(self.params["iv_lookback"]))
        return finite(p) and p * 100.0 < self.params["iv_pct"]

    def underlying_exit(self, ctx, und, pos, t):
        p, _, _ = self.iv_percentile(ctx, und, int(self.params["iv_lookback"]))
        if finite(p) and p * 100.0 > self.params["exit_pct"]:
            return "iv_pct_gt_60"
        return None

    def build(self, ctx, und, cv, xe, us):
        return st.straddle(xe, 1, ref_price=cv.spot, usable=us)


@register
class S144(SinglePackage):
    id = "S144"; name = NAMES["S144"]; bar_freq = "1d"; sizing = "OPT_DEBIT"; asset = "OPT"
    params = {"underlying": "SPY", "delta": 0.25, "edge": 3.0, "dte": 30, "exit_dte": 7, "max_hold": 10,
              "iv_source": "auto"}
    grid = {"delta": [0.15, 0.25, 0.35], "edge": [1.0, 3.0, 5.0]}
    spec = _spec("S144")
    DTE_OFFSETS = (0, 15)
    deviations = COMMON_DEVIATIONS + [
        DEBIT_DEV,
        "'20d realized vol forecast' = HAR-RV forecast of the next 20 sessions (predeclared model, see common "
        "deviations); 'ATM implied' = 30-day constant-maturity ATM IV of the real snapshot (VIX fallback on "
        "synthetic-only dates, flagged). Entry if forecast - ATM IV >= edge/100.",
        "Exit after 10 sessions or DTE <= 7, both legs in one combo; DTE window 30-45 (target 30).",
    ]

    def extra_requires(self):
        return [Requirement("bars", "1d", symbols=("VIX",), critical=False, note="IV fallback (synthetic dates)")]

    def needs_trend(self):
        return False

    def entry_signal(self, ctx, und, t):
        f = har_forecast_vol(self.hist(ctx, und, 800), 20)
        iv, _ = self.atm_iv30(ctx, und)
        return finite(f, iv) and (f - iv) * 100.0 >= self.params["edge"]

    def build(self, ctx, und, cv, xe, us):
        d = abs(self.params["delta"])
        return st.strangle(xe, d, -d, 1, usable=us)


# ============================================================================ S145 / S146 short vol
@register
class S145(SinglePackage):
    id = "S145"; name = NAMES["S145"]; bar_freq = "1mo"; sizing = "OPT_NAKED"; asset = "OPT"
    params = {"underlying": "SPY", "delta": 0.25, "ratio": 1.3, "dte": 30, "exit_dte": 7, "tp": 0.5,
              "iv_source": "auto"}
    grid = {"delta": [0.16, 0.25, 0.30], "ratio": [1.2, 1.3, 1.5]}
    spec = _spec("S145")
    RISK = "naked"
    NEEDS_VALUE = True
    DTE_OFFSETS = (0, 15)
    deviations = COMMON_DEVIATIONS + [
        NAKED_DEV,
        "'ATM IV30/realized20 forecast' = 30-day constant-maturity ATM IV (real snapshot; VIX fallback flagged) / "
        "HAR-RV 20-session forecast >= ratio.",
        "'2x initial credit loss' = loss (debit to close - credit) >= 2 x credit; '50% credit captured' = debit to "
        "close <= 50% of credit. DTE window 30-45 (target 30). Entry whenever flat and the filter holds.",
    ]

    def extra_requires(self):
        return [Requirement("bars", "1d", symbols=("VIX",), critical=False, note="IV fallback (synthetic dates)")]

    def needs_trend(self):
        return False

    def entry_signal(self, ctx, und, t):
        f = har_forecast_vol(self.hist(ctx, und, 800), 20)
        iv, _ = self.atm_iv30(ctx, und)
        return finite(f, iv) and f > 0 and iv / f >= self.params["ratio"]

    def build(self, ctx, und, cv, xe, us):
        d = abs(self.params["delta"])
        return st.strangle(xe, d, -d, -1, usable=us)

    def value_exit(self, ctx, pos, v):
        c, cc = pos.credit, self.close_cost(v)
        if cc <= (1 - self.params["tp"]) * c:
            return "tp_50pct_credit"
        if cc - c >= 2.0 * c:
            return "stop_loss_2x_credit"
        return None


class _HedgedStraddle(SinglePackage):
    """Straddle with a share delta hedge: the entry COMBO carries the stock leg (engine prices it from the first
    bar opening after submission, options from the first snapshot after submission), daily-close rehedges are
    market stock orders filling at the next open, and the hedge is flattened when the options are closed."""

    asset = "OPT+STK"
    HEDGE_BAND_PARAM = "band"

    def needs_trend(self):
        return False

    def quoted_atm_iv(self, ctx, und, pkg: st.Package, cv: ChainView) -> float:
        if not cv.synthetic:
            ivs = [l.iv for l in pkg.legs if finite(l.iv)]
            return float(np.mean(ivs)) if ivs else float("nan")
        v, src = self.atm_iv30(ctx, und)        # synthetic snapshot: never its smile -> vol index (flagged)
        return v if src == "index" else float("nan")

    def hedge_per_unit(self, pkg: st.Package) -> int:
        d = pkg.greeks()["delta"] * pkg.multiplier
        return int(-round(d)) if finite(d) else 0

    def submit_entry(self, ctx, und, pkg, n, cv, coll):
        h = self.hedge_per_unit(pkg)
        extra = (Leg(ctx.instrument(und), h),) if h != 0 else ()
        return self.open_package(ctx, und, pkg, n, cv, collateral=coll, extra_legs=extra,
                                 meta=self.entry_meta(pkg) | {"hedge_per_unit": h})

    def pending_stock(self, ctx, und) -> float:
        return sum(o.remaining for o in ctx.open_orders(und) if o.instrument is not None and not o.instrument.is_option)

    def band_shares(self, pos: OptPos) -> float:
        return float(self.params[self.HEDGE_BAND_PARAM]) * pos.multiplier * pos.units

    def on_open_extra(self, ctx, und, pos, cv):
        if cv is None:
            return
        dsh = self.package_delta_shares(ctx, cv, pos)
        if not finite(dsh):
            return
        cur = ctx.position(und) + self.pending_stock(ctx, und)
        q = st.hedge_shares(dsh, cur, self.band_shares(pos))
        if q:
            ctx.order(und, q, tag="rehedge")

    def on_close_extra(self, ctx, und, pos):
        for o in ctx.open_orders(und):
            if o.instrument is not None and not o.instrument.is_option:
                ctx.cancel(o.id)
        q = ctx.position(und)
        if q:
            ctx.order(und, -q, tag="unhedge")


@register
class S146(_HedgedStraddle):
    id = "S146"; name = NAMES["S146"]; bar_freq = "1d"; sizing = "OPT_NAKED"
    params = {"underlying": "SPY", "edge": 3.0, "hedge": "1d", "dte": 30, "exit_dte": 7, "band": 0.05,
              "iv_source": "auto"}
    grid = {"edge": [2.0, 3.0, 5.0], "hedge": ["1d", "1h"], "dte": [21, 30]}
    spec = _spec("S146")
    RISK = "naked"
    HEDGE_BAND_PARAM = "band"
    deviations = COMMON_DEVIATIONS + [
        NAKED_DEV,
        "'quoted ATM IV' = mean IV of the two straddle legs from the REAL snapshot quotes (quoted IV, else IV from the "
        "mid); on synthetic-only dates the Cboe vol index is used (flagged T2_PROXY). 'trailing 20d realized vol' = "
        "close-to-close 20d annualized (pricing.realized). Entry if IV - RV >= edge/100.",
        "Entry hedge: the opening COMBO includes round(-package delta x 100) shares per straddle (engine: stock leg at "
        "the first open after submission, options at the first snapshot). Rehedge at each daily close when |net "
        "delta| > 0.05 x 100 shares per straddle unit (options.structures.hedge_shares), filling at the next open; "
        "deltas from the snapshot (surface marks if not quoted).",
        "hedge='1h' needs intraday option quotes that do not exist in the free data -> critical Requirement "
        "'options_intraday' => NOT_TESTABLE by design.",
        "Close both options at DTE <= 7 in one combo; the share hedge is flattened by a market order submitted at the "
        "same close (fills next open). Dividends/borrow on the hedge are booked by the engine.",
    ]

    def extra_requires(self):
        req = [Requirement("bars", "1d", symbols=("VIX",), critical=False, note="IV fallback (synthetic dates)")]
        if self.params.get("hedge") == "1h":
            req.append(Requirement("options_intraday", "1h", symbols=tuple(self.universe),
                                   note="hourly re-hedging needs intraday option quotes (none in the free data)"))
        return req

    def entry_signal(self, ctx, und, t):
        return True                          # evaluated in accept() on the selected straddle

    def build(self, ctx, und, cv, xe, us):
        return st.straddle(xe, -1, ref_price=cv.spot, usable=us)

    def accept(self, ctx, und, pkg, cv):
        iv = self.quoted_atm_iv(ctx, und, pkg, cv)
        rv = trailing_rv(self.hist(ctx, und, 60), 20)
        return finite(iv, rv) and (iv - rv) * 100.0 >= self.params["edge"]


@register
class S147(_HedgedStraddle):
    id = "S147"; name = NAMES["S147"]; bar_freq = "1d"; sizing = "OPT_DEBIT"
    params = {"underlying": "SPY", "threshold": 0.10, "edge": 3.0, "dte": 30, "exit_dte": 7, "max_hold": 10,
              "iv_source": "auto"}
    grid = {"threshold": [0.05, 0.10, 0.20], "dte": [21, 30]}
    spec = _spec("S147")
    HEDGE_BAND_PARAM = "threshold"
    deviations = COMMON_DEVIATIONS + [
        DEBIT_DEV,
        "'realized vol forecast exceeds implied by 3 vol pts' = HAR-RV 20-session forecast - mean IV of the two "
        "straddle legs (real snapshot; vol-index fallback flagged) >= 3 vol points.",
        "Entry hedge in the opening COMBO (stock leg, see S146). 'Rehedge at delta drift >0.1 or daily close': no "
        "intraday option quotes exist, so the drift test is evaluated at each daily close only (|net delta| > threshold "
        "x 100 shares per straddle) and the rehedge fills at the next open.",
        "Exit at DTE <= 7 or 10 sessions after the fill session; options in one combo, hedge flattened by a market "
        "order at the same close.",
    ]

    def extra_requires(self):
        return [Requirement("bars", "1d", symbols=("VIX",), critical=False, note="IV fallback (synthetic dates)")]

    def build(self, ctx, und, cv, xe, us):
        return st.straddle(xe, 1, ref_price=cv.spot, usable=us)

    def accept(self, ctx, und, pkg, cv):
        iv = self.quoted_atm_iv(ctx, und, pkg, cv)
        f = har_forecast_vol(self.hist(ctx, und, 800), 20)
        return finite(iv, f) and (f - iv) * 100.0 >= self.params["edge"]


# ============================================================================ S148 / S149 calendars
class _Calendar(SinglePackage):
    sizing = "OPT_DEBIT"
    asset = "OPT"
    bar_freq = "1mo"
    NEEDS_VALUE = True
    RIGHT = "C"
    NEAR_OFFSETS = (-5, 5)
    FAR_OFFSETS = (-5, 15)
    LIMIT = False

    def needs_trend(self):
        return False

    def select(self, ctx, und, cv):
        n_t, f_t = float(self.params["near_dte"]), float(self.params["far_dte"])
        e_near = self.pick_expiry(cv, n_t, *dte_window(n_t, self.NEAR_OFFSETS))
        e_far = self.pick_expiry(cv, f_t, *dte_window(f_t, self.FAR_OFFSETS))
        if e_far <= e_near:
            raise SelectionError("far expiry not after near expiry")
        xn, un = self.slice(cv, e_near)
        xf, uf = self.slice(cv, e_far)
        pkg = st.calendar(xn[un], xf[uf], self.RIGHT, ref_price=cv.spot)
        return pkg if self.calendar_ok(ctx, und, pkg, cv) else None

    def calendar_ok(self, ctx, und, pkg, cv) -> bool:
        return pkg.natural_open() > 0

    def submit_entry(self, ctx, und, pkg, n, cv, coll):
        lim = round(pkg.natural_open(), 2) if self.LIMIT else None
        return self.open_package(ctx, und, pkg, n, cv, limit=lim, meta=self.entry_meta(pkg))

    def value_exit(self, ctx, pos, v):
        if self.pnl(pos, v) >= self.params["tp"] * pos.entry_net:
            return f"tp_{int(round(self.params['tp'] * 100))}pct_debit"
        return None


@register
class S148(_Calendar):
    id = "S148"; name = NAMES["S148"]
    params = {"underlying": "SPY", "near_dte": 30, "far_dte": 60, "tp": 0.30, "exit_dte": 7}
    grid = {"near_dte": [21, 30], "far_dte": [60, 90], "tp": [0.20, 0.30, 0.50]}
    spec = _spec("S148")
    RIGHT = "C"
    LIMIT = True
    deviations = COMMON_DEVIATIONS + [
        DEBIT_DEV,
        "Near 25-35 DTE around 30, far 55-75 around 60 (grid 90 -> 85-105: beyond the ~65-DTE reach of the real "
        "DoltHub snapshots, so 2020+ far=90 finds no expiry -> no trade; pre-2020 synthetic T3). Same ATM strike "
        "listed in both expiries (structures.calendar).",
        "'two-leg limit spread' = COMBO limit at the natural debit of the decision snapshot (GTC, cancelled after 5 "
        "sessions); the engine fills at the next snapshot only if its natural debit <= limit.",
        "Exit when the near leg has DTE <= 7 or P&L >= tp x initial debit (snapshot mids), both legs in one combo.",
    ]


@register
class S149(_Calendar):
    id = "S149"; name = NAMES["S149"]
    params = {"underlying": "SPY", "near_dte": 30, "far_dte": 60, "rel_iv": 5.0, "tp": 0.30, "exit_dte": 7,
              "iv_source": "auto"}
    grid = {"near_dte": [21, 30], "far_dte": [60, 90], "rel_iv": [-5.0, 0.0, 5.0]}
    spec = _spec("S149")
    RIGHT = "P"
    deviations = COMMON_DEVIATIONS + [
        DEBIT_DEV,
        "Near 25-35 DTE around 30, far 55-75 around 60 (grid 90 -> 85-105, see S148). 'far IV <= near IV + 5 vol "
        "points' uses the IVs of the two put legs from the REAL snapshot quotes; on synthetic-only dates the VIX "
        "term structure is used instead (near = VIX, far = VIX->VIX3M total-variance interpolation at the far DTE; "
        "SPX-like underlyings only; flagged T2_PROXY).",
        "Market combo (no limit stated); exit when the near leg has DTE <= 7 or P&L >= 30% of the debit.",
    ]

    def extra_requires(self):
        return [Requirement("bars", "1d", symbols=("VIX", "VIX3M"), critical=False, note="IV fallback (synthetic)")]

    def calendar_ok(self, ctx, und, pkg, cv):
        if pkg.natural_open() <= 0:
            return False
        near, far = pkg.legs[0], pkg.legs[1]
        mode = self.params.get("iv_source", "auto")
        if not cv.synthetic and mode in ("auto", "chain"):
            ivn, ivf = near.iv, far.iv
        elif mode in ("auto", "index") and vol_index_for(und) == "VIX":
            v30 = ctx.history("VIX", 1, "1d", adjusted=False)
            v93 = ctx.history("VIX3M", 1, "1d", adjusted=False)
            if not (len(v30) and len(v93)):
                return False
            ivn = float(v30["close"].iloc[-1]) / 100.0
            dte_far = self.dte(ctx, far.expiry)
            ivf, _ = interp_total_variance([30 / 365, 93 / 365], [ivn, float(v93["close"].iloc[-1]) / 100.0],
                                           dte_far / 365.0)
            ctx.flag_fidelity(Fidelity.T2_PROXY, f"{und} calendar IV filter from VIX/VIX3M (synthetic chain)")
        else:
            return False
        return finite(ivn, ivf) and ivf <= ivn + self.params["rel_iv"] / 100.0


# ============================================================================ S150 diagonal
@register
class S150(SinglePackage):
    id = "S150"; name = NAMES["S150"]; bar_freq = "1mo"; sizing = "OPT_DEBIT"; asset = "OPT"
    params = {"underlying": "SPY", "long_delta": 0.80, "far_dte": 180, "short_delta": 0.30, "near_dte": 30,
              "exit_dte": 7, "leap_exit_dte": 90}
    grid = {"long_delta": [0.75, 0.85], "far_dte": [180, 365], "short_delta": [0.20, 0.30]}
    spec = _spec("S150")
    NEAR_OFFSETS = (-5, 15)
    deviations = COMMON_DEVIATIONS + [
        DEBIT_DEV,
        "LEAP window [far_dte, far_dte+185] (baseline 180-365; grid 365 -> 365-550); short call 25-45 DTE. Real "
        "DoltHub snapshots reach ~65 DTE only, so 2020+ the LEAP leg has no quote -> no trade; pre-2020 synthetic T3.",
        "'underlying bullish trend filter' (unspecified) = close > SMA200 (adjusted); 'trend reversal' = close < SMA200.",
        "Near leg DTE <= 7 -> roll: ONE combo buying back the old short and selling a new 0.30-delta 25-45 DTE call "
        "(next snapshot); if no new short qualifies the old one is just bought back. LEAP DTE <= 90 or trend reversal "
        "-> close every remaining leg in one combo. The position is always the actual LEAP (no share equivalence).",
    ]

    def entry_signal(self, ctx, und, t):
        return finite(t.get("sma200")) and t["close"] > t["sma200"]

    def select(self, ctx, und, cv):
        f_t = float(self.params["far_dte"])
        e_far = self.pick_expiry(cv, f_t, int(f_t), int(f_t) + 185)
        n_t = float(self.params["near_dte"])
        e_near = self.pick_expiry(cv, n_t, *dte_window(n_t, self.NEAR_OFFSETS))
        xn, un = self.slice(cv, e_near)
        xf, uf = self.slice(cv, e_far)
        return st.diagonal(xn, xf, abs(self.params["long_delta"]), abs(self.params["short_delta"]), un, uf)

    def step(self, ctx, und):
        t = trend_state(self.hist(ctx, und))
        pos = self.first_live(und)
        if pos is None:
            return super().step(ctx, und)
        if pos.status != "OPEN" or pos.meta.get("roll_oid") is not None:
            return
        legs = {i.symbol: (i, r) for i, r in pos.option_legs() if abs(ctx.position(i.symbol)) > 1e-9}
        long_leg = next(((i, r) for i, r in legs.values() if r > 0), None)
        short_leg = next(((i, r) for i, r in legs.values() if r < 0), None)
        if long_leg is None:
            self.close_package(ctx, pos, "leap_gone")
            return
        if self.dte(ctx, long_leg[0].expiry) <= self.params["leap_exit_dte"]:
            self.close_package(ctx, pos, "leap_dte_le_90")
            return
        if finite(t.get("sma200")) and t["close"] < t["sma200"]:
            self.close_package(ctx, pos, "trend_reversal")
            return
        if short_leg is not None and self.dte(ctx, short_leg[0].expiry) <= self.exit_dte():
            self.roll_short(ctx, und, pos, short_leg[0])
        elif short_leg is None:
            self.roll_short(ctx, und, pos, None)

    def roll_short(self, ctx, und, pos, old) -> None:
        cv = self.chain(ctx, und)
        new = None
        if cv is not None:
            try:
                n_t = float(self.params["near_dte"])
                e = self.pick_expiry(cv, n_t, *dte_window(n_t, self.NEAR_OFFSETS))
                xn, un = self.slice(cv, e)
                row = select_by_delta(xn, abs(self.params["short_delta"]), "C", un)
                new = st.OptLeg.from_row(row, -1).instrument()
            except SelectionError as e:
                ctx.log(f"S150 {und}: no new short ({e})")
        if old is None and new is None:
            return
        legs = []
        if old is not None:
            legs.append(Leg(old, 1))
        if new is not None:
            legs.append(Leg(new, -1))
        if cv is not None:
            self.note_chain_use(ctx, cv, "S150 roll")
        o = ctx.submit(Order(None, float(pos.units), OrderType.COMBO, tif=TimeInForce.GTC, legs=tuple(legs),
                             tag="roll_short"))
        if o.status == OrderStatus.REJECTED:
            return
        pos.meta["roll_oid"] = o.id
        pos.meta["roll_new"] = new
        pos.meta["roll_old"] = old
        self.oid_map[o.id] = (pos, "roll")

    def on_fill(self, ctx, fill):
        hit = self.oid_map.get(fill.order_id)
        if hit is not None and hit[1] == "roll":
            pos = hit[0]
            new, old = pos.meta.get("roll_new"), pos.meta.get("roll_old")
            pos.meta["roll_filled"] = pos.meta.get("roll_filled", 0) + 1
            if pos.meta["roll_filled"] >= (int(new is not None) + int(old is not None)):
                keep = [(i, r) for i, r in pos.legs if old is None or i.symbol != old.symbol]
                if new is not None:
                    keep.append((new, -1))
                pos.legs = keep
                pos.meta.pop("partial", None)
                pos.meta["roll_oid"] = None
                pos.meta["roll_filled"] = 0
            return
        super().on_fill(ctx, fill)

    def reconcile(self, ctx, und):
        for pos in self.positions.get(und, []):
            rid = pos.meta.get("roll_oid")
            if rid is not None and not any(o.id == rid for o in ctx.open_orders()):
                pos.meta["roll_oid"] = None     # roll cancelled/expired without fill
        super().reconcile(ctx, und)


# ============================================================================ S151 / S152 backspreads
class _Backspread(SinglePackage):
    sizing = "OPT_DEBIT"
    asset = "OPT"
    bar_freq = "1mo"
    DTE_OFFSETS = (-15, 15)
    SIGN = 1

    def needs_trend(self):
        return False

    def build(self, ctx, und, cv, xe, us):
        s = self.SIGN
        return st.ratio_backspread(xe, s * abs(self.params["short_delta"]), s * abs(self.params["long_delta"]),
                                   int(self.params["ratio"]), usable=us)


@register
class S151(_Backspread):
    id = "S151"; name = NAMES["S151"]
    params = {"underlying": "SPY", "ratio": 2, "short_delta": 0.50, "long_delta": 0.25, "dte": 45, "exit_dte": 7,
              "dd_budget": 0.005}
    grid = {"ratio": [2], "short_delta": [0.45, 0.50], "long_delta": [0.20, 0.25]}
    spec = _spec("S151")
    SIGN = 1
    NEEDS_VALUE = True
    deviations = COMMON_DEVIATIONS + [
        DEBIT_DEV,
        "DTE window 30-60 (target 45). The package's natural net debit/credit and max loss are computed and recorded; "
        "packages without unbounded upside (max profit finite) are refused.",
        "'max drawdown reaches defined portfolio budget' = open loss of the package (snapshot mids) >= dd_budget x NAV "
        "with dd_budget = 0.50% (the OPT_DEBIT per-package budget). Entry whenever flat (monthly cycle).",
    ]

    def accept(self, ctx, und, pkg, cv):
        mpl = pkg.max_profit_loss(pkg.natural_open())
        return math.isinf(mpl["max_profit"]) and finite(mpl["max_loss"])

    def value_exit(self, ctx, pos, v):
        loss_usd = -self.pnl(pos, v) * pos.units * pos.multiplier
        if loss_usd >= self.params["dd_budget"] * ctx.nav:
            return "drawdown_budget"
        return None


@register
class S152(_Backspread):
    id = "S152"; name = NAMES["S152"]
    params = {"underlying": "SPY", "ratio": 2, "short_delta": 0.50, "long_delta": 0.25, "dte": 45, "exit_dte": 7}
    grid = {"ratio": [2], "long_delta": [0.20, 0.25], "short_delta": [0.45, 0.50]}
    spec = _spec("S152")
    SIGN = -1
    deviations = COMMON_DEVIATIONS + [
        DEBIT_DEV,
        "DTE window 30-60 (target 45); entry whenever flat (monthly cycle); natural net debit/credit recorded and "
        "carried in the P&L (entry fill net), exit at DTE <= 7.",
    ]


# ============================================================================ S153 / S154 risk reversals
@register
class S153(SinglePackage):
    id = "S153"; name = NAMES["S153"]; bar_freq = "1mo"; sizing = "OPT_NAKED"; asset = "OPT"
    params = {"underlying": "SPY", "delta": 0.25, "dte": 30, "exit_dte": 7}
    grid = {"delta": [0.20, 0.25, 0.30], "dte": [30, 45]}
    spec = _spec("S153")
    RISK = "naked"
    DTE_OFFSETS = (0, 15)
    deviations = COMMON_DEVIATIONS + [
        NAKED_DEV,
        "'predicted upside positive' (model unspecified) = predeclared 20-session time-series momentum: "
        "close_t / close_{t-20} - 1 > 0 (adjusted). 'signal crosses bearish' = close < SMA200 or 20d return < 0.",
        "Put side margined by the engine's Cboe strategy margin (option 'model margin'), not cash-secured. DTE 30-45.",
    ]

    def entry_signal(self, ctx, und, t):
        return finite(t.get("sma200"), t.get("ret20")) and t["close"] > t["sma200"] and t["ret20"] > 0

    def underlying_exit(self, ctx, und, pos, t):
        if finite(t.get("sma200"), t.get("ret20")) and (t["close"] < t["sma200"] or t["ret20"] < 0):
            return "signal_bearish"
        return None

    def build(self, ctx, und, cv, xe, us):
        return st.risk_reversal(xe, abs(self.params["delta"]), bullish=True, usable=us)


@register
class S154(SinglePackage):
    id = "S154"; name = NAMES["S154"]; bar_freq = "1mo"; sizing = "OPT_NAKED"; asset = "OPT"
    params = {"underlying": "SPY", "delta": 0.25, "dte": 30, "exit_dte": 7}
    grid = {"delta": [0.20, 0.25, 0.30]}
    spec = _spec("S154")
    RISK = "naked"
    DTE_OFFSETS = (0, 15)
    deviations = COMMON_DEVIATIONS + [
        NAKED_DEV,
        "'bearish trend ends' = close > SMA200 (adjusted). Naked short call margined by the engine's Cboe strategy "
        "margin. DTE window 30-45 (target 30).",
    ]

    def entry_signal(self, ctx, und, t):
        return finite(t.get("sma200")) and t["close"] < t["sma200"]

    def underlying_exit(self, ctx, und, pos, t):
        if finite(t.get("sma200")) and t["close"] > t["sma200"]:
            return "bearish_trend_ended"
        return None

    def build(self, ctx, und, cv, xe, us):
        return st.risk_reversal(xe, abs(self.params["delta"]), bullish=False, usable=us)


# ============================================================================ S156 jade lizard
@register
class S156(SinglePackage):
    id = "S156"; name = NAMES["S156"]; bar_freq = "1mo"; sizing = "OPT_NAKED"; asset = "OPT"
    params = {"underlying": "SPY", "put_delta": 0.30, "call_delta": 0.30, "wing_delta": 0.15, "dte": 30,
              "exit_dte": 7, "tp": 0.5, "require_no_upside": True}
    grid = {"put_delta": [0.20, 0.30], "call_delta": [0.20, 0.30], "wing_delta": [0.10, 0.15]}
    spec = _spec("S156")
    RISK = "naked"
    NEEDS_VALUE = True
    deviations = COMMON_DEVIATIONS + [
        NAKED_DEV,
        "'require total credit >= call vertical width if targeting zero upside loss': baseline targets zero upside "
        "loss (require_no_upside=True) using the NATURAL credit of the decision snapshot (structures.jade_lizard "
        "meta); otherwise no trade.",
        "'Close for 50% credit' = debit to close <= 50% of the credit received; DTE window 25-35 ('30DTE'). Entry "
        "whenever flat (monthly cycle). Put assignment handled by the engine lifecycle (reconciled: remaining legs "
        "closed).",
    ]

    def needs_trend(self):
        return False

    def build(self, ctx, und, cv, xe, us):
        return st.jade_lizard(xe, abs(self.params["put_delta"]), abs(self.params["call_delta"]),
                              abs(self.params["wing_delta"]), usable=us)

    def accept(self, ctx, und, pkg, cv):
        return (not self.params["require_no_upside"]) or bool(pkg.meta.get("no_upside_risk"))

    def value_exit(self, ctx, pos, v):
        if self.close_cost(v) <= (1 - self.params["tp"]) * pos.credit:
            return "tp_50pct_credit"
        return None


# stock-holding strategies (S128-S133, S157) register on import
from . import options_income_stock  # noqa: E402,F401
