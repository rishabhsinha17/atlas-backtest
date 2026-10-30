"""Daily volatility-signal option strategies S165-S170 (IV percentile, skew,
term structure, forecast vs implied, dispersion).  Family conventions in
``options_events_common`` (EOD close decisions on real snapshots, fills at the
next snapshot, percentiles / z-scores vs the previous 252 valid observations).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..core.types import Fidelity
from ..engine.strategy import Requirement
from ..options.chain import SelectionError
from ..options.margin import OptionPosition, package_margin, stress_loss
from ..options.structures import (Package, calendar, iron_condor, risk_reversal, straddle, vega_match_units,
                                  vertical)
from ..pricing.realized import daily_variance, har_rv_forecast, realized_vol
from ..risk import sizing as SZ
from .options_events_common import (ACTIVE, COMMON_DEVIATIONS, OPT_ETF, SECTOR_SPDR, OptionsEventBase,
                                    choose_series, pct_rank_prev, underlying_class, z_prev)
from .options_events_specs import NAMES, SPECS
from .registry import register

DAILY_DEVIATIONS = COMMON_DEVIATIONS + (
    "Signals at the session close on the real EOD snapshot (T2 by data: capture minute unknown); orders fill at "
    "the next snapshot (next close).",
    "Universe (sheet names no underlying): liquid ETF options with real EOD chains - SPY, DIA and the 11 sector "
    "SPDRs (QQQ/IWM absent from the free chain source).",
    "'NN DTE' = listed expiry closest to NN calendar days (ties earlier) within [0.6 NN, 1.6 NN].",
    "Exit 'at 7DTE' = closing order at the first close with DTE <= 7 of the nearest leg.",
    "Synthetic chains have deterministic smiles (docs/OPTIONS.md §7.4) and are refused for signals unless "
    "allow_synthetic=True (mechanics only, T3).",
)


def _series(x: pd.DataFrame, dte: float) -> pd.DataFrame:
    """One expiry series (expiry + root + settlement) closest to ``dte``."""
    return choose_series(x, dte, int(round(0.6 * dte)), int(round(1.6 * dte)))


def _pending_stock(ctx, und: str) -> bool:
    return any(o.instrument is not None and o.instrument.symbol == und for o in ctx.open_orders(und))


def _credit_legs(pkg: Package) -> list:
    return [SZ.OptLeg(l.instrument(), l.ratio, l.bid if l.ratio < 0 else l.ask) for l in pkg.legs]


class _DailyBase(OptionsEventBase):
    activity_note = ("Signals need 252 valid daily observations from REAL chains (DoltHub from 2020 -> first signals "
                     "2021) and are tail conditions (IV percentile / skew z / term spread / forecast edge); "
                     "synthetic chains are refused: low activity expected, zero before 2021.")
    bar_freq = "1d"
    universe = list(OPT_ETF)
    CHAIN_MAX_FID = Fidelity.T2_PROXY
    requires = [Requirement("options", "1d", min_fidelity=Fidelity.T2_PROXY, note="real EOD option chains"),
                Requirement("bars", "1d")]

    def today_chain(self, ctx, und: str) -> Optional[pd.DataFrame]:
        ch = self.get_chain(ctx, und, session=ctx.session_date())
        if ch is None:
            return None
        return self.prepared(ctx, ch, und)

    def held(self, und: str):
        for k, st in self.book.items():
            if st.und == und:
                return k, st
        return None, None

    def near_dte(self, ctx, st) -> int:
        return int(min((pd.Timestamp(i.expiry) - ctx.session_date()).days for i in st.insts))

    def credit_tp_hit(self, ctx, st, x, tp: float) -> bool:
        v, fid, how = self.liquidation_value(ctx, st, pd.Timedelta(hours=20), x)
        if not np.isfinite(v):
            return False
        if how != "quotes":
            self.flag_once(ctx, Fidelity(min(max(fid, 2), 3)), f"{self.id}: exit decision on same-day surface marks "
                                                                f"(contract missing from the snapshot)")
        credit = -st.entry_cash
        return credit > 0 and (v + credit) >= tp * credit

    def on_bar(self, ctx) -> None:
        self.refresh(ctx)
        for und in self.symbols(ctx):
            try:
                self.step(ctx, und)
            except SelectionError as e:
                self.skip(ctx, f"{und}: {e}", activity="selection_failure")

    def step(self, ctx, und: str) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


# ============================================================================ S165
@register
class S165(_DailyBase):
    id = "S165"
    name = NAMES["S165"]
    sizing = "OPT_CREDIT"
    params = {"iv_pct": 90, "exit_pct": 50, "short_delta": 0.20, "wing_delta": 0.05, "dte": 30, "tp": 0.50,
              "exit_dte": 7, "max_rel_spread": 0.50, "iv_source": "chain", "allow_synthetic": False}
    grid = {"iv_pct": [80, 90, 95], "exit_pct": [40, 50, 60]}
    deviations = DAILY_DEVIATIONS + (
        "'At 90th percentile' = percentile (mid-rank, previous 252 valid values) >= iv_pct.",
        "'Spread filters pass' = all four legs two-sided, uncrossed, nonzero bid, (ask-bid)/mid <= 50% "
        "(atlasbt.options.chain.QualityRules).",
        "iv_source='vix_proxy' (opt-in, SPY only, T2_PROXY): percentile of the Cboe VIX close (known 16:15 ET, so "
        "the prior session's close at a 16:00 decision) instead of the chain-derived 30D ATM IV.",
    )
    spec = SPECS["S165"]

    def __init__(self, **params):
        super().__init__(**params)
        if self.params["iv_source"] == "vix_proxy":
            self.universe = ["SPY"]
            self.requires = list(self.requires) + [Requirement("bars", "1d", symbols=("VIX",),
                                                               note="Cboe VIX close (proxy)")]

    def _pct(self, ctx, und: str) -> float:
        if self.params["iv_source"] == "vix_proxy":
            self.flag_once(ctx, Fidelity.T2_PROXY, "S165 vix_proxy: VIX close percentile instead of 30D ATM IV")
            h = ctx.history("VIX", self.LOOKBACK + 1, "1d", adjusted=False)
            if h is None or len(h) < self.LOOKBACK + 1:
                return float("nan")
            return pct_rank_prev(h["close"].values, self.LOOKBACK)
        s = self.metric_series(ctx, und, "atm_iv_30")
        if len(s) == 0 or s.index[-1] != ctx.session_date():
            return float("nan")
        return pct_rank_prev(s.values[-(self.LOOKBACK + 1):], self.LOOKBACK)

    def step(self, ctx, und):
        key, st = self.held(und)
        pct = self._pct(ctx, und)
        if st is not None:
            if st.close_order is not None or st.filled_units <= 0:
                return
            x = self.today_chain(ctx, und)
            reason = None
            if np.isfinite(pct) and pct <= self.params["exit_pct"]:
                reason = "iv_pct_exit"
            elif self.near_dte(ctx, st) <= self.params["exit_dte"]:
                reason = "dte"
            elif self.credit_tp_hit(ctx, st, x, self.params["tp"]):
                reason = "take_profit"
            if reason:
                self.close_package(ctx, key, reason)
            return
        if not (np.isfinite(pct) and pct >= self.params["iv_pct"]):
            return
        x = self.today_chain(ctx, und)
        if x is None:
            return
        xe = _series(x, self.params["dte"])
        pkg = iron_condor(xe, self.params["short_delta"], wing_delta=self.params["wing_delta"])
        if not self.legs_pass(xe, pkg):
            self.skip(ctx, f"{und}: spread filter failed")
            return
        credit = -pkg.natural_open()
        if not credit > 0:
            return
        legs = _credit_legs(pkg)
        ml = pkg.max_profit_loss()["max_loss_usd"]
        n, _ = SZ.size_opt_credit(ctx.nav, ml, legs=legs)
        self.open_package(ctx, f"{und}:{ctx.session_date().date()}", und, pkg, n, "credit",
                          meta={"iv_pct": pct, "credit": credit, "max_loss_usd": ml})


# ============================================================================ S166
@register
class S166(_DailyBase):
    id = "S166"
    name = NAMES["S166"]
    sizing = "OPT_DEBIT"
    params = {"skew_z": -2.0, "dte": 30, "long_delta": 0.25, "short_delta": 0.10, "exit_z": 0.0, "exit_dte": 7,
              "allow_synthetic": False}
    grid = {"skew_z": [-1.5, -2.0, -2.5], "dte": [30, 45]}
    deviations = DAILY_DEVIATIONS + (
        "skew = IV(-25 delta put) - IV(+25 delta call) at constant 30D maturity (delta-space interpolation, "
        "chain_metrics.skew25_30), as the sheet defines it; z vs mean / sample stdev of the previous 252 valid values.",
        "Grid 'DTE 30/45' = DTE of the traded put spread; the skew signal stays at 30D as the entry text defines it.",
        "'Directional delta not neutral; report separately': no hedge; package delta logged at entry.",
    )
    spec = SPECS["S166"]

    def step(self, ctx, und):
        key, st = self.held(und)
        s = self.metric_series(ctx, und, "skew25_30")
        z = z_prev(s.values[-(self.LOOKBACK + 1):], self.LOOKBACK) if len(s) and s.index[-1] == ctx.session_date() \
            else float("nan")
        if st is not None:
            if st.close_order is not None or st.filled_units <= 0:
                return
            if np.isfinite(z) and z >= self.params["exit_z"]:
                self.close_package(ctx, key, "skew_z_exit")
            elif self.near_dte(ctx, st) <= self.params["exit_dte"]:
                self.close_package(ctx, key, "dte")
            return
        if not (np.isfinite(z) and z < self.params["skew_z"]):
            return
        x = self.today_chain(ctx, und)
        if x is None:
            return
        xe = _series(x, self.params["dte"])
        pkg = vertical(xe, -abs(self.params["long_delta"]), -abs(self.params["short_delta"]), usable=self.usable(xe),
                       name="put_debit_spread")
        debit = pkg.natural_open()
        if not debit > 0:
            return
        n = SZ.size_opt_debit(ctx.nav, debit * pkg.multiplier, existing_option_debit=self.open_option_debit(ctx))
        self.open_package(ctx, f"{und}:{ctx.session_date().date()}", und, pkg, n, "debit",
                          meta={"skew_z": z, "debit": debit, "delta_per_unit": pkg.greeks()["delta"],
                                "planned_debit_usd": n * debit * pkg.multiplier})


# ============================================================================ S167
@register
class S167(_DailyBase):
    id = "S167"
    name = NAMES["S167"]
    sizing = "OPT_NAKED"
    params = {"skew_z": 2.0, "exit_z": 0.5, "delta": 0.25, "dte": 30, "exit_dte": 7, "hedge": "daily",
              "allow_synthetic": False}
    grid = {"skew_z": [1.5, 2.0, 2.5], "hedge": ["daily", "weekly"]}
    deviations = DAILY_DEVIATIONS + (
        "skew = put-minus-call 25-delta IV at 30D (as S166); risk reversal on the expiry closest to 30 DTE "
        "(sheet: 'same expiry', tenor unspecified -> the skew tenor).",
        "Delta hedge with shares to zero net delta at the close ('daily': every close; 'weekly': last session of "
        "each week + entry/exit days), executed at the next open; option deltas from the close snapshot.",
        "OPT_NAKED sizing: contracts so that the predefined-scenario stress loss of (risk reversal + its initial "
        "share hedge) <= 1% NAV (atlasbt.options.margin.stress_loss); Cboe strategy margin must be > 0.",
    )
    spec = SPECS["S167"]

    def _hedge_day(self, ctx) -> bool:
        if self.params["hedge"] == "daily":
            return True
        cal = ctx.calendar()
        nxt = cal.next_session(ctx.session_date())
        return nxt.isocalendar()[1] != ctx.session_date().isocalendar()[1]

    def step(self, ctx, und):
        key, st = self.held(und)
        s = self.metric_series(ctx, und, "skew25_30")
        z = z_prev(s.values[-(self.LOOKBACK + 1):], self.LOOKBACK) if len(s) and s.index[-1] == ctx.session_date() \
            else float("nan")
        x = None
        if st is not None:
            if st.close_order is None and st.filled_units > 0:
                if np.isfinite(z) and z < self.params["exit_z"]:
                    self.close_package(ctx, key, "skew_z_exit")
                elif self.near_dte(ctx, st) <= self.params["exit_dte"]:
                    self.close_package(ctx, key, "dte")
            x = self.today_chain(ctx, und)
            if st.close_order is None and st.filled_units > 0 and (self._hedge_day(ctx) or not st.meta.get("hedged")):
                self.hedge_to_neutral(ctx, und, x)
                st.meta["hedged"] = True
            return
        if abs(ctx.position(und)) > 0 and not any(s_.und == und for s_ in self.book.values()) \
                and not _pending_stock(ctx, und):
            ctx.order(und, -ctx.position(und), tag="hedge_unwind")
        if not (np.isfinite(z) and z > self.params["skew_z"]):
            return
        x = self.today_chain(ctx, und)
        if x is None:
            return
        xe = _series(x, self.params["dte"])
        pkg = risk_reversal(xe, self.params["delta"], bullish=True, usable=self.usable(xe))
        S = float(np.nanmedian(xe["spot"]))
        hedge_per_unit = -pkg.greeks()["delta"] * pkg.multiplier
        pos = [OptionPosition.from_leg(l, l.ratio, l.bid if l.ratio < 0 else l.ask) for l in pkg.legs]
        sl = stress_loss(pos, S, ctx.now, r=float(x.attrs.get("r_cc", 0.0)), stock_shares=hedge_per_unit)["worst_loss"]
        mg = package_margin(pkg, 1, S, underlying_class=underlying_class(und)).requirement
        try:
            n = SZ.size_opt_naked(ctx.nav, sl, mg)
        except ValueError as e:
            self.skip(ctx, f"{und}: {e}", activity="selection_failure")
            return
        self.open_package(ctx, f"{und}:{ctx.session_date().date()}", und, pkg, n, "credit" if pkg.natural_open() < 0
                          else "debit", meta={"skew_z": z, "stress_per_unit": sl, "margin_per_unit": mg,
                                              "hedged": False})

    def on_package_closed(self, ctx, st) -> None:
        q = ctx.position(st.und)
        if abs(q) > 1e-9 and not _pending_stock(ctx, st.und):
            ctx.order(st.und, -q, tag="hedge_unwind")


# ============================================================================ S168
@register
class S168(_DailyBase):
    id = "S168"
    name = NAMES["S168"]
    sizing = "OPT_DEBIT"
    params = {"term_spread": 5.0, "legs": (30, 60), "exit_spread": 0.0, "exit_dte": 7, "allow_synthetic": False}
    grid = {"term_spread": [2.0, 5.0, 8.0], "legs": [(30, 60), (21, 45)]}
    deviations = DAILY_DEVIATIONS + (
        "Term spread = constant-maturity ATM IV(near) - ATM IV(far) in vol points, near/far = the 'legs' tenors "
        "(30/60 or 21/45, total-variance interpolation); legs = ATM call strike listed in both expiries closest to "
        "the near/far tenors.",
    )
    spec = SPECS["S168"]

    def step(self, ctx, und):
        key, st = self.held(und)
        near, far = self.params["legs"]
        m = self.metrics_today(ctx, und)
        ts = (self.cm_iv(m, near) - self.cm_iv(m, far)) * 100.0 if m else float("nan")
        if st is not None:
            if st.close_order is None and st.filled_units > 0:
                if np.isfinite(ts) and ts <= self.params["exit_spread"]:
                    self.close_package(ctx, key, "term_spread_exit")
                elif self.near_dte(ctx, st) <= self.params["exit_dte"]:
                    self.close_package(ctx, key, "dte")
            return
        if not (np.isfinite(ts) and ts >= self.params["term_spread"]):
            return
        x = self.today_chain(ctx, und)
        if x is None:
            return
        xn, xf = _series(x, near), _series(x, far)
        if xf["expiry"].iloc[0] <= xn["expiry"].iloc[0]:
            self.skip(ctx, f"{und}: far expiry not after near")
            return
        pkg = calendar(xn, xf, "C")
        debit = pkg.natural_open()
        if not debit > 0:
            return
        n = SZ.size_opt_debit(ctx.nav, debit * pkg.multiplier, existing_option_debit=self.open_option_debit(ctx))
        self.open_package(ctx, f"{und}:{ctx.session_date().date()}", und, pkg, n, "debit",
                          meta={"term_spread": ts, "debit": debit, "planned_debit_usd": n * debit * pkg.multiplier})


# ============================================================================ S169
@register
class S169(_DailyBase):
    id = "S169"
    name = NAMES["S169"]
    sizing = "OPT_CREDIT"
    params = {"edge": 5.0, "forecast": "har", "short_delta": 0.25, "long_delta": 0.10, "dte": 30, "tp": 0.50,
              "exit_dte": 7, "exit_edge": 1.0, "allow_synthetic": False}
    grid = {"edge": [3.0, 5.0, 7.0], "forecast": ["har", "rv20"]}
    deviations = DAILY_DEVIATIONS + (
        "Forecast models fixed ex ante: 'har' = HAR-RV (Corsi 2009) of the average daily variance over the next 21 "
        "sessions (~30 calendar days), daily variance = Garman-Klass + squared overnight gap, walk-forward OLS refit "
        "every 21 sessions on targets fully realized at the fit date, expanding window, >= 252 training rows "
        "(atlasbt.pricing.realized.har_rv_forecast); 'rv20' = 20-day close-to-close realized vol. Adjusted daily bars.",
        "Edge = 30D constant-maturity ATM IV - forecast vol, in vol points.",
    )
    spec = SPECS["S169"]

    def _forecast(self, ctx, und: str) -> float:
        n = 100_000 if self.params["forecast"] == "har" else 30      # HAR: full history (expanding window)
        h = ctx.history(und, n, "1d", adjusted=True)
        if h is None or len(h) < 22:
            return float("nan")
        if self.params["forecast"] == "rv20":
            return float(realized_vol(h, 20, "cc").iloc[-1])
        f = har_rv_forecast(daily_variance(h, "gk"), horizon=21, min_train=252, refit_every=21)
        return float(f["vol_fcst"].iloc[-1])

    def step(self, ctx, und):
        key, st = self.held(und)
        m = self.metrics_today(ctx, und)
        iv = m.get("atm_iv_30", np.nan) if m else np.nan
        fc = self._forecast(ctx, und)
        edge = (iv - fc) * 100.0 if np.isfinite(iv) and np.isfinite(fc) else float("nan")
        if st is not None:
            if st.close_order is not None or st.filled_units <= 0:
                return
            x = self.today_chain(ctx, und)
            reason = None
            if np.isfinite(edge) and edge < self.params["exit_edge"]:
                reason = "edge_exit"
            elif self.near_dte(ctx, st) <= self.params["exit_dte"]:
                reason = "dte"
            elif self.credit_tp_hit(ctx, st, x, self.params["tp"]):
                reason = "take_profit"
            if reason:
                self.close_package(ctx, key, reason)
            return
        if not (np.isfinite(edge) and edge > self.params["edge"]):
            return
        x = self.today_chain(ctx, und)
        if x is None:
            return
        xe = _series(x, self.params["dte"])
        pkg = vertical(xe, -abs(self.params["long_delta"]), -abs(self.params["short_delta"]), usable=self.usable(xe),
                       name="put_credit_spread")
        credit = -pkg.natural_open()
        if not credit > 0:
            return
        legs = _credit_legs(pkg)
        ml = SZ.max_loss_vertical(legs)
        n, _ = SZ.size_opt_credit(ctx.nav, ml, legs=legs)
        self.open_package(ctx, f"{und}:{ctx.session_date().date()}", und, pkg, n, "credit",
                          meta={"edge": edge, "iv": iv, "forecast": fc, "credit": credit, "max_loss_usd": ml})


# ============================================================================ S170
#: fixed "5 highest-weight" sector SPDRs (S&P 500 GICS weights 2018-2026: IT, Health Care, Financials,
#: Consumer Discretionary, Communication Services); the 11-sector variant uses all SPDRs.
TOP5_SECTORS = ["XLK", "XLV", "XLF", "XLY", "XLC"]


@register
class S170(_DailyBase):
    activity_note = ("One dispersion package per month at most (month-end, when flat), only if all six real chains "
                     "are usable and every sector leg fills before the SPY leg is sent.")
    id = "S170"
    name = NAMES["S170"]
    sizing = "OPT_NAKED"
    universe = ["SPY"] + SECTOR_SPDR
    params = {"vega_ratio": 1.0, "n_sectors": 5, "dte": 30, "exit_dte": 7, "allow_synthetic": False}
    grid = {"vega_ratio": [0.75, 1.0, 1.25], "n_sectors": [5, 11]}
    deviations = DAILY_DEVIATIONS + (
        "'1mo' cadence: a new dispersion package is opened at the close of the LAST session of each month when flat "
        "(month-end rule); closed at the first close with DTE <= 7 on any leg.",
        "'Fixed 5 highest-weight sector ETFs' = XLK, XLV, XLF, XLY, XLC (largest GICS weights 2018-2026, fixed list, "
        "not re-selected); 11 = all sector SPDRs. A month is skipped if any selected sector has no usable chain.",
        "Dollar-vega matching: total long sector straddle $vega = vega_ratio x short SPY straddle $vega, split "
        "equally across the sectors; integer contracts (rounded, >= 1).",
        "One dispersion package (dispersion_id): the five (or 11) long sector straddles are submitted first (DAY "
        "COMBOs at the month-end close, filled at the next snapshot); the short SPY straddle only after ALL sector "
        "legs filled completely. If any member is unfilled / partially filled by the next bar, every working member "
        "is cancelled and every filled member closed (no naked index leg, no partial package).",
        "Net delta hedged per underlying with shares at each close once the package is complete (initial hedge the "
        "close after the SPY leg fills), executed at the next open; hedges are unwound once the options are closed.",
        "OPT_NAKED sizing on the short SPY straddle: stress loss (predefined scenarios) <= 1% NAV with a positive "
        "Cboe margin; sector straddles are long premium (bounded debit).",
        "'Model each ETF option quotes and cross-correlation': each ETF uses its own real chain; the implied "
        "correlation proxy (SPY IV / vega-weighted sector IV)^2 is logged at entry.",
    )
    spec = SPECS["S170"]

    def sectors(self) -> list[str]:
        return list(TOP5_SECTORS) if int(self.params["n_sectors"]) == 5 else list(SECTOR_SPDR)

    def __init__(self, **params):
        super().__init__(**params)
        self.disp: Optional[dict] = None          # the ONE dispersion package (all six straddles)

    def _members_filled(self, keys) -> bool:
        for k in keys:
            st = self.book.get(k)
            if st is None or st.open_order.status in ACTIVE or st.filled_units + 1e-9 < st.units:
                return False
        return True

    def _unwind(self, ctx, reason: str) -> None:
        for k in list(self.disp["members"]):
            if k in self.book:
                self.close_package(ctx, k, reason)       # cancels a working entry, closes filled legs
        self.disp["stage"] = "unwinding"
        ctx.log("S170 dispersion unwound", id=self.disp["id"], reason=reason, activity="combo_cancelled")

    def on_bar(self, ctx) -> None:
        self.refresh(ctx)
        chains: dict = {}

        def chain(u):
            if u not in chains:
                chains[u] = self.today_chain(ctx, u)
            return chains[u]

        d = self.disp
        if d is not None:
            if d["stage"] == "sectors" and ctx.now > d["submitted_at"]:
                if self._members_filled(d["sector_keys"]):
                    st = self.open_package(ctx, d["index_key"], "SPY", d["idx_pkg"], d["n_idx"], "credit",
                                           meta=d["idx_meta"] | {"dispersion_id": d["id"]})
                    if st is None:
                        self._unwind(ctx, "index_leg_rejected")
                    else:
                        d["members"].append(d["index_key"])
                        d["stage"], d["submitted_at"] = "index", ctx.now
                else:
                    self._unwind(ctx, "sector_leg_unfilled")
            elif d["stage"] == "index" and ctx.now > d["submitted_at"]:
                if self._members_filled(d["members"]):
                    d["stage"] = "complete"
                else:
                    self._unwind(ctx, "index_leg_unfilled")
            if d["stage"] == "complete":
                sts = [self.book[k] for k in d["members"] if k in self.book]
                if len(sts) < len(d["members"]):
                    self._unwind(ctx, "member_closed_by_lifecycle")
                elif any(self.near_dte(ctx, st) <= self.params["exit_dte"] for st in sts):
                    for k in d["members"]:
                        self.close_package(ctx, k, "dte")
                    d["stage"] = "closing"
            if d["stage"] in ("unwinding", "closing") and not any(k in self.book for k in d["members"]):
                self.disp = None
        # delta hedge per underlying only while the package is complete; unwind shares otherwise
        complete = self.disp is not None and self.disp["stage"] == "complete"
        hedged = {self.book[k].und for k in self.disp["members"] if k in self.book} if complete else set()
        unds = hedged | {s for s, q in ctx.positions().items() if abs(q) > 0 and not ctx.instrument(s).is_option}
        for u in sorted(unds):
            if u in hedged:
                self.hedge_to_neutral(ctx, u, chain(u))
            elif abs(ctx.position(u)) > 0 and not _pending_stock(ctx, u) and \
                    (self.disp is None or self.disp["stage"] != "closing"):
                ctx.order(u, -ctx.position(u), tag="hedge_unwind")
        if self.disp is not None or self.book or not ctx.calendar().is_last_session_of_month(ctx.session_date()):
            return
        self._open(ctx, chain)

    def _open(self, ctx, chain) -> None:
        x = chain("SPY")
        if x is None:
            return
        try:
            idx = straddle(_series(x, self.params["dte"]), -1, usable=self.usable(x))
        except SelectionError as e:
            self.skip(ctx, f"SPY: {e}", activity="selection_failure")
            return
        secs = {}
        for s in self.sectors():
            xs = chain(s)
            if xs is None:
                self.skip(ctx, f"{s}: no usable chain; month skipped")
                return
            try:
                secs[s] = straddle(_series(xs, self.params["dte"]), 1, usable=self.usable(xs))
            except SelectionError as e:
                self.skip(ctx, f"{s}: {e}; month skipped", activity="selection_failure")
                return
        S = float(np.nanmedian(x["spot"]))
        pos = [OptionPosition.from_leg(l, l.ratio, l.bid) for l in idx.legs]
        sl = stress_loss(pos, S, ctx.now, r=float(x.attrs.get("r_cc", 0.0)))["worst_loss"]
        mg = package_margin(idx, 1, S, underlying_class="broad_etf").requirement
        try:
            n_idx = SZ.size_opt_naked(ctx.nav, sl, mg)
        except ValueError as e:
            self.skip(ctx, str(e), activity="selection_failure")
            return
        if n_idx <= 0:
            self.skip(ctx, "SPY straddle size 0")
            return
        idx_vega = abs(idx.dollar_greeks(n_idx)["vega_usd"])
        target = self.params["vega_ratio"] * idx_vega / len(secs)
        units = {}
        for s, p in secs.items():
            try:
                u = int(round(vega_match_units(target, p)))
            except SelectionError:
                u = 0
            if u < 1:
                self.skip(ctx, f"{s}: vega-matched size < 1 contract; month skipped")
                return
            units[s] = u
        iv_idx = float(np.nanmean([l.iv for l in idx.legs]))
        w = np.array([units[s] * abs(secs[s].dollar_greeks(1)["vega_usd"]) for s in secs])
        ivs = np.array([np.nanmean([l.iv for l in secs[s].legs]) for s in secs])
        rho = (iv_idx / float(np.sum(w * ivs) / np.sum(w))) ** 2 if np.sum(w) > 0 else float("nan")
        tag = ctx.session_date().date()
        did = f"DISP:{tag}"
        keys = []
        for s, p in secs.items():
            st = self.open_package(ctx, f"{s}:{tag}", s, p, units[s], "debit", meta={"role": "sector",
                                                                                    "dispersion_id": did})
            if st is None:
                for k in keys:
                    self.close_package(ctx, k, "sector_leg_rejected")
                return
            keys.append(st.key)
        self.disp = {"id": did, "stage": "sectors", "submitted_at": ctx.now, "sector_keys": list(keys),
                     "members": list(keys), "index_key": f"SPY:{tag}", "idx_pkg": idx, "n_idx": n_idx,
                     "idx_meta": {"role": "index", "stress_per_unit": sl, "margin_per_unit": mg,
                                  "implied_corr_proxy": rho}}