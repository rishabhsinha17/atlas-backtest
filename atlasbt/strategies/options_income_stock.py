"""Stock-holding and cash-secured option strategies of the options income family:
S128 covered call, S129 Cboe-style SPX putwrite, S130 cash-secured put, S131
wheel, S132 protective put, S133 collar, S157 covered strangle.

Imported by :mod:`atlasbt.strategies.options_income` (registration happens on
import). Common mechanics: :mod:`atlasbt.strategies.options_income_base`.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Optional

import numpy as np
import pandas as pd

from ..core.clock import et
from ..core.types import Fidelity
from ..engine.strategy import Requirement
from ..options import structures as st
from ..options.chain import SelectionError, nearest_atm
from ..risk import sizing as sz
from .options_income_base import (
    COMMON_DEVIATIONS, ETF_SYMBOLS, INDEX_UNDERLYINGS, ChainView, OptionBase, OptPos, dte_window, finite,
    third_friday_session,
)
from .options_income_specs import NAMES, SPECS
from .registry import register

COVERED_DEV = ("OPT_COVERED sizing (risk.sizing.size_opt_covered): shares = whole lots of 100 with underlying gross "
               "<= 10% NAV, one call per 100 shares held ('own exactly 100 shares' read as exactly 100 shares PER "
               "contract). Shares are bought with a market order at the next open; options are written at the first "
               "close at which the shares are held (one session later).")
CASHSEC_DEV = ("OPT_CASHSEC sizing (risk.sizing.size_opt_cashsec + CollateralBook): strike x 100 cash reserved per "
               "short put from unreserved cash; assignment notional + shares already held <= 10% NAV per single name "
               "/ ETF (100% for a broad index putwrite).")
UNIVERSE_DEV = ("Underlying: SPY by default; use_megacap=True trades the declared static MEGACAP list "
                "(options_income_base.MEGACAP, survivorship caveat).")


def _replace_ratio(pkg: st.Package, ratios: list[int]) -> st.Package:
    legs = [dataclasses.replace(l, ratio=int(r)) for l, r in zip(pkg.legs, ratios) if int(r) != 0]
    return st.Package(pkg.name, legs, list(pkg.stock), dict(pkg.meta))


class _StockBase(OptionBase):
    """Helpers for strategies that hold shares of the underlying."""

    bar_freq = "1mo"
    asset = "STK+OPT"
    DTE_OFFSETS = (-5, 5)

    def pending_stock(self, ctx, und: str) -> float:
        return sum(o.remaining for o in ctx.open_orders(und) if o.instrument is not None and not o.instrument.is_option)

    def spot(self, ctx, und: str) -> float:
        return self.last_raw_close(ctx, und)

    def target_lots_covered(self, ctx, und: str, frac: float = 0.10) -> int:
        px = self.spot(ctx, und)
        if not finite(px):
            return 0
        _, lots = sz.size_opt_covered(ctx.nav, px, existing_shares=0.0, gross_cap=frac)
        return int(lots)

    def ensure_shares(self, ctx, und: str, target_shares: float) -> bool:
        """Submit a market order toward ``target_shares``; True if shares are (already) at target."""
        have = ctx.position(und)
        if abs(have - target_shares) < 1e-9:
            return True
        if abs(self.pending_stock(ctx, und)) > 1e-9:
            return False
        q = target_shares - have
        if q > 0:
            ctx.order(und, q, tag="buy_shares")
        return False

    def window(self) -> tuple[int, int]:
        return dte_window(float(self.params["dte"]), self.DTE_OFFSETS)

    def manage_open(self, ctx, und: str, pos: OptPos, exit_dte: int, reason: str) -> bool:
        """Common exits for an OPEN package; True if a close was submitted."""
        if pos.meta.get("partial"):
            return self.close_package(ctx, pos, "leg_settled") is not None
        if self.dte(ctx, pos.expiry) <= exit_dte:
            return self.close_package(ctx, pos, reason) is not None
        return False


# ============================================================================ S128 covered call
@register
class S128(_StockBase):
    id = "S128"; name = NAMES["S128"]; sizing = "OPT_COVERED"
    params = {"underlyings": ("SPY",), "use_megacap": False, "delta": 0.30, "dte": 30, "exit_dte": 5,
              "rebuy_after_assignment": True}
    grid = {"delta": [0.20, 0.30, 0.40], "dte": [21, 30, 45]}
    spec = SPECS["S128"]
    deviations = COMMON_DEVIATIONS + [
        COVERED_DEV, UNIVERSE_DEV,
        "Call = listed call with delta closest to +0.30 among strikes ABOVE the stock spot (decision snapshot's "
        "underlying price), 25-35 DTE (grid 21 -> 16-26, 45 -> 40-50); one cycle at a time (written again at the "
        "first close after the previous call is bought back / expires).",
        "'Buy back call at DTE<=5' = closing combo at the first close with DTE <= 5 (next snapshot). 'close on "
        "assignment': the engine lifecycle delivers the shares (expiry or modeled early assignment); the shares are "
        "re-bought at the next open so the covered-call cycle continues (rebuy_after_assignment=True; False keeps "
        "the account in cash). No buy-write rebalancing otherwise (shares kept).",
    ]

    def step(self, ctx, und):
        pos = self.first_live(und)
        if pos is not None:
            if pos.status == "OPEN":
                self.manage_open(ctx, und, pos, int(self.params["exit_dte"]), "buyback_dte")
            return
        shares = ctx.position(und)
        tgt = self.store.setdefault("target", {}).get(und)
        if tgt is None or (shares <= 0 and abs(self.pending_stock(ctx, und)) < 1e-9):
            if tgt is not None and not self.params["rebuy_after_assignment"]:
                return
            tgt = 100 * self.target_lots_covered(ctx, und)
            self.store["target"][und] = tgt
        if tgt <= 0:
            return
        if shares < tgt - 1e-9:
            self.ensure_shares(ctx, und, tgt)
            return
        lots = int(math.floor(shares / 100 + 1e-9))
        cv = self.chain(ctx, und)
        if cv is None or lots < 1:
            return
        exp = self.pick_expiry(cv, float(self.params["dte"]), *self.window())
        xe, us = self.slice(cv, exp)
        keep = (xe["right"] == "P") | (xe["strike"] > cv.spot)
        pkg = st.covered_call(xe[keep], und, abs(self.params["delta"]), usable=us[keep])
        self.open_package(ctx, und, pkg, lots, cv, meta={"strike": pkg.legs[0].strike})


# ============================================================================ S129 SPX putwrite
@register
class S129(_StockBase):
    id = "S129"; name = NAMES["S129"]; bar_freq = "1mo"; sizing = "OPT_CASHSEC"; asset = "OPT"
    params = {"underlying": "SPX", "strike_rule": "atm", "put_delta": 0.35, "close_before_settlement": False,
              "close_dte": 1, "roll_window": 4}
    grid = {"strike_rule": ["atm", "delta"]}
    spec = SPECS["S129"]
    deviations = COMMON_DEVIATIONS + [
        "SPX options exist only as the SYNTHETIC T3 chain (no free SPX quotes; replication of the Cboe PUT index: TE "
        "2.6%/yr, ~1.5%/yr rich, docs/OPTIONS.md §7.4). Every fill/mark/settlement is T3/T2.",
        "Roll date = the settlement session of the standard AM monthly (3rd Friday, holiday -> prior session). The new "
        "put (next standard AM monthly, root SPX) is sold at the first close inside [roll date, roll date + 4 "
        "sessions] while flat -> filled at the next snapshot (Cboe sells at a late-morning VWAP on the roll date). A "
        "backtest starting mid-cycle waits for the next roll date.",
        "Strike: nearest listed ATM put (baseline); grid 'alternate delta .35' -> strike_rule='delta' (put delta "
        "-0.35). 'roll monthly' is fixed (not a free parameter).",
        "T-bill collateral: the full strike x 100 per contract is kept as cash (CollateralBook), which earns the "
        "engine's cash rate (3M T-bill minus the broker spread), not the full 1m/3m T-bill yield: understates PUT-"
        "index collateral income by <= 0.5%/yr. Index putwrite collateral cap 100% NAV (not the 10% single-name cap).",
        "Settlement per the actual contract: AM-settled monthlies settle at the SOQ (proxied by the SPX opening "
        "print, engine lifecycle, T2). 'close near expiration before settlement as separate test' = param "
        "close_before_settlement=True (closing combo at DTE <= close_dte); not in the sheet grid.",
    ]

    def step(self, ctx, und):
        pos = self.first_live(und)
        if pos is not None:
            if pos.status == "OPEN" and self.params["close_before_settlement"]:
                self.manage_open(ctx, und, pos, int(self.params["close_dte"]), "close_before_settlement")
            return
        sess = ctx.session_date()
        cal = ctx.calendar()
        roll = third_friday_session(cal, sess.year, sess.month)
        if sess < roll:
            return
        n_after = len(cal.sessions(roll, sess)) - 1
        cycle = (sess.year, sess.month)
        done = self.store.setdefault("cycles", set())
        if n_after > int(self.params["roll_window"]) or cycle in done:
            return
        cv = self.chain(ctx, und)
        if cv is None:
            return
        root = "SPX" if und == "SPX" else None
        exp = self.pick_expiry(cv, 30.0, 7, 45, standard_only=True, root=root)
        xe, us = self.slice(cv, exp, root)
        if self.params["strike_rule"] == "atm":
            pkg = st.cash_secured_put(xe, atm=True, ref_price=cv.spot, usable=us)
        else:
            pkg = st.cash_secured_put(xe, -abs(self.params["put_delta"]), usable=us)
        leg = pkg.legs[0]
        n = self.size_cashsec(ctx, leg.strike, und, cv.spot, leg.multiplier)
        if self.open_package(ctx, und, pkg, n, cv, collateral=n * leg.strike * leg.multiplier,
                             meta={"cycle": cycle}) is not None:
            done.add(cycle)


# ============================================================================ S130 cash-secured put
@register
class S130(_StockBase):
    id = "S130"; name = NAMES["S130"]; sizing = "OPT_CASHSEC"; asset = "OPT"
    params = {"underlyings": ("SPY",), "use_megacap": False, "delta": 0.30, "dte": 30, "tp": 0.50, "exit_dte": 7,
              "assigned_shares": "hold", "earnings_proxy": False}
    grid = {"delta": [0.20, 0.30, 0.40], "dte": [30, 45], "tp": [0.25, 0.50, 0.75]}
    spec = SPECS["S130"]
    DTE_OFFSETS = (-5, 15)
    deviations = COMMON_DEVIATIONS + [
        CASHSEC_DEV, UNIVERSE_DEV,
        "DTE window 25-45 around target 30 (grid 45 -> 40-60). 'Buy to close at <=50% original credit' = debit to "
        "close (snapshot mids) <= (1 - tp) x credit; grid TP 25/50/75% = fraction of the credit captured.",
        "'no earnings before expiration': ETFs/indexes have none. Single stocks need a FORWARD earnings calendar known "
        "at entry (25-45 days ahead); the data layer only knows scheduled dates 7 days ahead (EARNINGS_SCHEDULE), so "
        "single-stock runs declare a critical Requirement 'earnings_calendar_forward' -> NOT_TESTABLE, unless the "
        "opt-in proxy earnings_proxy=True (next release expected 80-100 days after the last known release, plus any "
        "visible EARNINGS_SCHEDULE event; flagged T2_PROXY).",
        "'if assigned acquire shares and continue accounting': assigned shares are held and marked (assigned_shares="
        "'hold'); later puts are sized with those shares counted in the 10% name cap. 'sell' liquidates them at the "
        "next open (not in the sheet grid).",
    ]

    def extra_requires(self):
        stocks = tuple(u for u in self.universe if u not in ETF_SYMBOLS and u not in INDEX_UNDERLYINGS)
        if stocks and not self.params.get("earnings_proxy"):
            return [Requirement("earnings_calendar_forward", symbols=stocks,
                                note="earnings dates 25-45 days ahead known at entry (not in the free data)")]
        return []

    def earnings_block(self, ctx, und: str, expiry) -> bool:
        if und in ETF_SYMBOLS or und in INDEX_UNDERLYINGS:
            return False
        end = et(pd.Timestamp(expiry), "16:00")
        now = ctx.now
        p = ctx.portal()
        sched = p.events("EARNINGS_SCHEDULE", [und], now - pd.Timedelta(days=60), now)
        for e in sched:
            eff = e.effective_at if e.effective_at is not None else e.known_at
            if now < pd.Timestamp(eff) <= end:
                return True
        past = p.events("EARNINGS", [und], now - pd.Timedelta(days=200), now)
        if not past:
            ctx.flag_fidelity(Fidelity.T2_PROXY, f"{und}: no EARNINGS history visible -> earnings filter unverifiable")
            return False
        last = max(pd.Timestamp(e.known_at) for e in past)
        ctx.flag_fidelity(Fidelity.T2_PROXY, f"{und}: next earnings date proxied as last release + 80-100 days")
        lo, hi = last + pd.Timedelta(days=80), last + pd.Timedelta(days=100)
        return lo <= end and hi > now

    def step(self, ctx, und):
        shares = ctx.position(und)
        if self.params["assigned_shares"] == "sell" and shares > 0 and abs(self.pending_stock(ctx, und)) < 1e-9:
            ctx.order(und, -shares, tag="sell_assigned")
        pos = self.first_live(und)
        if pos is not None:
            if pos.status == "OPEN":
                if self.manage_open(ctx, und, pos, int(self.params["exit_dte"]), "dte_le_7"):
                    return
                v = self.package_value(ctx, self.chain(ctx, und), pos)
                if finite(v, pos.entry_net) and -v <= (1 - self.params["tp"]) * pos.credit:
                    self.close_package(ctx, pos, f"tp_{int(self.params['tp'] * 100)}pct")
            return
        cv = self.chain(ctx, und)
        if cv is None:
            return
        exp = self.pick_expiry(cv, float(self.params["dte"]), *self.window())
        if self.earnings_block(ctx, und, exp):
            ctx.log(f"S130 {und}: earnings before {exp.date()} -> no trade")
            return
        xe, us = self.slice(cv, exp)
        pkg = st.cash_secured_put(xe, -abs(self.params["delta"]), usable=us)
        leg = pkg.legs[0]
        n = self.size_cashsec(ctx, leg.strike, und, cv.spot, leg.multiplier)
        self.open_package(ctx, und, pkg, n, cv, collateral=n * leg.strike * leg.multiplier)


# ============================================================================ S131 wheel
@register
class S131(_StockBase):
    id = "S131"; name = NAMES["S131"]; sizing = "OPT_CASHSEC"
    params = {"underlyings": ("SPY",), "use_megacap": False, "delta": 0.30, "dte": 30, "roll": 7}
    grid = {"delta": [0.20, 0.30, 0.40], "dte": [30, 45], "roll": [7, 14]}
    spec = SPECS["S131"]
    deviations = COMMON_DEVIATIONS + [
        CASHSEC_DEV, UNIVERSE_DEV,
        "State is derived from the shares actually held after the engine's lifecycle (expiry assignment / early "
        "assignment, reconciled every close; ASSIGNMENT fills are also handled when delivered): >= 100 shares -> "
        "SHARE state (sell floor(shares/100) calls at +delta), else CASH state (sell cash-secured puts at -delta).",
        "'At 7 DTE roll profitable options': at DTE <= roll the short option is bought back if its snapshot mid is "
        "below the premium received, and the next option of the CURRENT state is sold at the following close (after "
        "the closing combo filled); unprofitable options are held to expiry (assignment drives the state change).",
        "DTE window 25-35 around 30 (grid 45 -> 40-50); dividends on shares are booked by the engine; P&L across all "
        "states is the account NAV.",
    ]

    def on_lifecycle_fill(self, ctx, fill):
        if fill.liquidity == "ASSIGNMENT":
            self.store.setdefault("assignments", []).append((fill.ts, fill.instrument.symbol, fill.quantity))

    def step(self, ctx, und):
        pos = self.first_live(und)
        if pos is not None:
            if pos.status == "OPEN" and self.dte(ctx, pos.expiry) <= int(self.params["roll"]):
                if not pos.meta.get("roll_checked_" + str(ctx.session_date().date())):
                    pos.meta["roll_checked_" + str(ctx.session_date().date())] = True
                    v = self.package_value(ctx, self.chain(ctx, und), pos)
                    if finite(v, pos.entry_net) and -v < pos.credit:
                        self.close_package(ctx, pos, "roll_profitable")
            return
        if abs(self.pending_stock(ctx, und)) > 1e-9:
            return
        shares = ctx.position(und)
        cv = self.chain(ctx, und)
        if cv is None:
            return
        exp = self.pick_expiry(cv, float(self.params["dte"]), *self.window())
        xe, us = self.slice(cv, exp)
        lots = int(math.floor(shares / 100 + 1e-9))
        if lots >= 1:
            pkg = st.covered_call(xe, und, abs(self.params["delta"]), usable=us)
            self.store.setdefault("state", {})[und] = "SHARE"
            self.open_package(ctx, und, pkg, lots, cv, meta={"state": "SHARE"})
        else:
            pkg = st.cash_secured_put(xe, -abs(self.params["delta"]), usable=us)
            leg = pkg.legs[0]
            n = self.size_cashsec(ctx, leg.strike, und, cv.spot, leg.multiplier)
            self.store.setdefault("state", {})[und] = "CASH"
            self.open_package(ctx, und, pkg, n, cv, collateral=n * leg.strike * leg.multiplier,
                              meta={"state": "CASH"})


# ============================================================================ S132 protective put
@register
class S132(_StockBase):
    id = "S132"; name = NAMES["S132"]; sizing = "OPT_COVERED"
    params = {"underlyings": ("SPY",), "use_megacap": False, "delta": 0.25, "dte": 60, "roll_dte": 30}
    grid = {"dte": [60, 90], "delta": [0.15, 0.25, 0.35]}
    spec = SPECS["S132"]
    DTE_OFFSETS = (0, 30)
    deviations = COMMON_DEVIATIONS + [
        COVERED_DEV.replace("one call per 100 shares", "one put per 100 shares"), UNIVERSE_DEV,
        "Put 60-90 DTE (target 60; grid 90 -> 90-120: beyond the ~65-DTE reach of real DoltHub snapshots -> no put "
        "2020+, synthetic T3 before), delta closest to -0.25.",
        "'on first month signal' = the first put is bought at the first session of a calendar month once the shares "
        "are held; afterwards 'roll when DTE<=30' sells the put at that close and the replacement is bought at the "
        "first close after the sale filled. Shares kept continuously (re-bought if ever delivered on exercise).",
    ]

    def step(self, ctx, und):
        tgt = self.store.setdefault("target", {}).get(und)
        if tgt is None or (ctx.position(und) <= 0 and abs(self.pending_stock(ctx, und)) < 1e-9):
            tgt = 100 * self.target_lots_covered(ctx, und)
            self.store["target"][und] = tgt
        if tgt <= 0:
            return
        pos = self.first_live(und)
        if pos is not None:
            if pos.status == "OPEN" and self.manage_open(ctx, und, pos, int(self.params["roll_dte"]), "roll_dte_le_30"):
                self.store.setdefault("rolling", set()).add(und)
            return
        if not self.ensure_shares(ctx, und, tgt):
            return
        rolling = self.store.setdefault("rolling", set())
        first_of_month = len(ctx.calendar().sessions(ctx.session_date().replace(day=1), ctx.session_date())) == 1
        if und not in rolling and not first_of_month:
            return
        cv = self.chain(ctx, und)
        if cv is None:
            return
        exp = self.pick_expiry(cv, float(self.params["dte"]), *self.window())
        xe, us = self.slice(cv, exp)
        pkg = st.protective_put(xe, -abs(self.params["delta"]), usable=us)
        lots = int(math.floor(ctx.position(und) / 100 + 1e-9))
        if self.open_package(ctx, und, pkg, lots, cv) is not None:
            rolling.discard(und)


# ============================================================================ S133 collar
@register
class S133(_StockBase):
    id = "S133"; name = NAMES["S133"]; sizing = "OPT_COVERED"
    params = {"underlyings": ("SPY",), "use_megacap": False, "delta": 0.25, "dte": 30, "roll": 5}
    grid = {"delta": [0.15, 0.25, 0.35], "roll": [5, 7]}
    spec = SPECS["S133"]
    deviations = COMMON_DEVIATIONS + [
        COVERED_DEV.replace("one call per 100 shares", "one put + one call per 100 shares"), UNIVERSE_DEV,
        "Grid 'deltas .15/.25/.35' applies the same absolute delta to the long put and the short call. 25-35 DTE.",
        "'Close/roll pair at 5DTE' = both legs closed in ONE combo at DTE <= roll; the new collar is opened at the "
        "first close after the closing combo filled. 'assignment handling throughout; keep stock': if the call is "
        "assigned (shares delivered) the remaining put is closed and the shares are re-bought at the next open.",
    ]

    def step(self, ctx, und):
        tgt = self.store.setdefault("target", {}).get(und)
        if tgt is None or (ctx.position(und) <= 0 and abs(self.pending_stock(ctx, und)) < 1e-9):
            tgt = 100 * self.target_lots_covered(ctx, und)
            self.store["target"][und] = tgt
        if tgt <= 0:
            return
        pos = self.first_live(und)
        if pos is not None:
            if pos.status == "OPEN":
                self.manage_open(ctx, und, pos, int(self.params["roll"]), "roll_dte")
            return
        if not self.ensure_shares(ctx, und, tgt):
            return
        cv = self.chain(ctx, und)
        if cv is None:
            return
        exp = self.pick_expiry(cv, float(self.params["dte"]), *self.window())
        xe, us = self.slice(cv, exp)
        d = abs(self.params["delta"])
        pkg = st.collar(xe, -d, d, usable=us)
        lots = int(math.floor(ctx.position(und) / 100 + 1e-9))
        self.open_package(ctx, und, pkg, lots, cv)


# ============================================================================ S157 covered strangle
@register
class S157(_StockBase):
    id = "S157"; name = NAMES["S157"]; sizing = "OPT_CASHSEC"
    params = {"underlyings": ("SPY",), "use_megacap": False, "delta": 0.20, "dte": 30, "exit_dte": 7,
              "share_frac": 0.05}
    grid = {"delta": [0.15, 0.20, 0.30], "dte": [30, 45]}
    spec = SPECS["S157"]
    deviations = COMMON_DEVIATIONS + [
        CASHSEC_DEV, UNIVERSE_DEV,
        "Shares: whole lots with gross <= share_frac (5%) of NAV so that the cash-secured puts fit under the 10% "
        "single-name cap (shares + put assignment notional <= 10% NAV). Calls = floor(shares/100) (100 shares per "
        "call), puts = OPT_CASHSEC count; one combo per cycle (units 1, leg ratios = contract counts).",
        "Grid 'deltas .15/.2/.3' = same absolute delta for the short call and the short put. 25-35 DTE (grid 45 -> "
        "40-50). 'Close/roll 7DTE' = both legs closed in one combo at DTE <= 7, next cycle at the first close after it "
        "filled. Put assignment adds shares, which are kept and covered by more calls in the next cycle.",
    ]

    def step(self, ctx, und):
        pos = self.first_live(und)
        if pos is not None:
            if pos.status == "OPEN":
                self.manage_open(ctx, und, pos, int(self.params["exit_dte"]), "roll_dte_le_7")
            return
        tgt = self.store.setdefault("target", {}).get(und)
        if tgt is None:
            tgt = 100 * self.target_lots_covered(ctx, und, frac=float(self.params["share_frac"]))
            self.store["target"][und] = tgt
        if tgt <= 0:
            return
        if ctx.position(und) < tgt - 1e-9:
            self.ensure_shares(ctx, und, tgt)
            return
        if abs(self.pending_stock(ctx, und)) > 1e-9:
            return
        cv = self.chain(ctx, und)
        if cv is None:
            return
        exp = self.pick_expiry(cv, float(self.params["dte"]), *self.window())
        xe, us = self.slice(cv, exp)
        d = abs(self.params["delta"])
        base = st.covered_strangle(xe, d, -d, usable=us)
        calls = int(math.floor(ctx.position(und) / 100 + 1e-9))
        put_leg = base.legs[1]
        puts = self.size_cashsec(ctx, put_leg.strike, und, cv.spot, put_leg.multiplier)
        pkg = _replace_ratio(base, [-calls, -puts])
        if not pkg.legs:
            return
        self.open_package(ctx, und, pkg, 1, cv, collateral=puts * put_leg.strike * put_leg.multiplier,
                          meta={"calls": calls, "puts": puts})
