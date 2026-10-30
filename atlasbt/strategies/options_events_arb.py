"""Arbitrage-style option strategies S171 (put-call parity screen) and S172
(SPX box-spread financing).  Family conventions in ``options_events_common``.
Both need timestamped executable quotes (S171: synchronized option NBBO + stock
NBBO; S172: simultaneous SPX quotes) and are NOT_TESTABLE on EOD snapshots.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..core.clock import session_date_of
from ..core.types import ActionType, Fidelity, Instrument, Leg, Order, OrderStatus, OrderType, TimeInForce
from ..engine.strategy import Requirement
from ..options.chain import SelectionError, get_row, listed_strikes
from ..options.parity import box_vs_tbill, conversion_edge, parity_screen, reversal_edge
from ..options.structures import Package, box, conversion
from ..pricing.time import year_fraction
from ..risk import sizing as SZ
from .options_events_common import (ACTIVE, COMMON_DEVIATIONS, ETF_CORE, OptionsEventBase, PkgState, choose_series,
                                    pick_series)
from .options_events_specs import NAMES, SPECS
from .registry import register


# ============================================================================ dividends
def dividend_schedule(ctx, sym: str, until: pd.Timestamp, today: pd.Timestamp) -> list[tuple[pd.Timestamp, float, bool]]:
    """Cash dividends with ex-date in (today, until]: declared ones known at ctx.now
    (portal corporate actions, PIT) plus a projection of the regular schedule
    (median gap of the last ex-dates, last regular amount) for later ex-dates."""
    try:
        acts = ctx.portal().corporate_actions([sym], today - pd.Timedelta(days=400), until + pd.Timedelta(days=1))
    except Exception:
        acts = []
    divs = sorted([(pd.Timestamp(a.ex_date).normalize(), float(a.amount)) for a in acts
                   if a.type == ActionType.CASH_DIVIDEND and a.amount > 0])
    past = [d for d in divs if d[0] <= today]
    fut = [(d, a, True) for d, a in divs if today < d <= until]
    out = list(fut)
    if len(past) >= 2:
        ex = [d for d, _ in past[-5:]]
        gap = pd.Timedelta(days=float(np.median(np.diff([e.value for e in ex])) / 86400e9))
        last_d, last_a = (fut[-1][0], fut[-1][1]) if fut else past[-1]
        nxt = last_d + gap
        while nxt <= until and gap > pd.Timedelta(days=20):
            if nxt > today and not any(abs(nxt - d) < gap / 2 for d, _, _ in out):
                out.append((nxt.normalize(), last_a, False))
            nxt = nxt + gap
    return sorted(out)


# ============================================================================ S171
@register
class S171(OptionsEventBase):
    activity_note = ("Parity arbitrage after all costs on synchronized NBBO is rare by construction; needs T0 option "
                     "and stock quotes (NOT_TESTABLE on free data); IOC entries often cancel when the edge vanishes.")
    id = "S171"
    name = NAMES["S171"]
    bar_freq = "1m"
    sizing = "OPT_ARBITRAGE"
    asset = "OPT+STK"
    universe = list(ETF_CORE)
    universe_name = None
    CHAIN_MAX_FID = Fidelity.T0_QUOTE
    QUOTE_MAX_AGE = pd.Timedelta(seconds=60)
    params = {"min_edge_bp": 10.0, "borrow_fee": None, "max_dte": None, "sync_s": 1.0, "fee_per_contract": 1.0,
              "stock_fee_per_share": 0.005, "max_contracts": 10, "notional_frac": 0.10, "scan_every": 1,
              "megacaps": True, "allow_synthetic": False}
    grid = {"min_edge_bp": [10.0, 25.0, 50.0], "borrow_fee": [0.0, 0.05, 0.20]}
    requires = [Requirement("options", "1m", min_fidelity=Fidelity.T0_QUOTE,
                            note="synchronized intraday option NBBO snapshots"),
                Requirement("quotes", min_fidelity=Fidelity.T0_QUOTE, note="stock NBBO quotes"),
                Requirement("borrow", critical=False, note="locates / borrow fee for reversals")]
    deviations = COMMON_DEVIATIONS + (
        "Universe: ETF_CORE + point-in-time SP500_TOP50_DV megacaps (STK/ETF intraday default); requirements "
        "checked statically on ETF_CORE, megacaps gated at runtime (option and stock quote fidelity T0).",
        "'Synchronized quotes' = option snapshot and stock NBBO timestamps within sync_s (1 s) of each other and "
        "<= 60 s old at the decision; locked/crossed or one-sided quotes skipped.",
        "Edges from atlasbt.options.parity (worst case over early assignment of the short American leg at each "
        "cum-dividend date / t=0 for reversals); dividends = declared (PIT) + projected regular schedule; rate = "
        "3M T-bill (continuous); borrow = portal fee (baseline borrow_fee=None) or the grid's stressed fee; "
        "fees per share = (2 x fee_per_contract + 100 x stock_fee_per_share) / 100.",
        "Baselines (sheet gives only the grid): min_edge_bp 10 (smallest tested), borrow from the portal.",
        "OPT_ARBITRAGE sizing: trade only if atlasbt.risk.sizing.arbitrage_edge is positive with locates; size = "
        "min(max_contracts, 10% NAV notional) contracts (displayed size not in chain frames).",
        "Execution: the option legs are one IOC COMBO (first snapshot after the decision) with a net limit that "
        "keeps the worst-case edge >= min_edge at the DECISION stock price: conversion cap on (P_ask - C_bid) = "
        "min_t[K e^-rt + PV(divs before t)] - S_ask - fee_ps - min_edge_ps; reversal cap on (C_ask - P_bid) = "
        "S_bid - fee_ps - max_t[(K + FV divs <= t) e^-(r-borrow)t] - min_edge_ps. The stock leg is NOT in the "
        "combo (combo units are option packages; a 100-share stock ratio would mix units in the combo limit; the "
        "engine would price a combo stock leg from the synchronized NBBO, with a bar-open fallback when no NBBO "
        "lies within 5 s): it is sent only after "
        "the combo fills, as a marketable LIMIT order priced so the locked edge stays >= 0 given the actual "
        "option fill; if it is not filled by the next scan it is cancelled and the package unwound (broken-package "
        "path). Early assignment is simulated by the engine's default option lifecycle (OccOptionLifecycle).",
        "'Hold until parity converges or expiration': closed when the natural unwind value >= the PV of the locked "
        "payoff (conversion) / unwind cost <= PV of the liability (reversal); otherwise held to expiry where the "
        "engine's lifecycle exercises/assigns.",
        "max_dte=None (default): every listed expiry is screened, as the sheet says ('same strike/expiry'); "
        "max_dte is only an optional speed cap, never set by default.",
    )
    spec = SPECS["S171"]

    def symbols(self, ctx) -> list[str]:
        syms = list(self.universe)
        if self.params["megacaps"]:
            try:
                syms += [s for s in ctx.portal().universe("SP500_TOP50_DV") if s not in syms]
            except Exception:
                pass
        return syms

    def _borrow(self, ctx, sym) -> tuple[bool, float]:
        try:
            ok, fee, fid = ctx.portal().borrow(sym)
        except Exception:
            ok, fee, fid = False, float("nan"), Fidelity.NA
        if self.params["borrow_fee"] is not None:
            return bool(ok), float(self.params["borrow_fee"])
        if fid >= Fidelity.T2_PROXY:
            self.flag_once(ctx, Fidelity.T2_PROXY, "S171: default borrow fee used (no borrow data)")
        return bool(ok), float(fee) if np.isfinite(fee) else 0.0

    def _fee_ps(self) -> float:
        return (2.0 * self.params["fee_per_contract"] + 100.0 * self.params["stock_fee_per_share"]) / 100.0

    def on_bar(self, ctx) -> None:
        self.refresh(ctx)
        self._send_stock_legs(ctx)
        if int(ctx.minutes_since_open()) % int(self.params["scan_every"]) != 0:
            return
        held = {st.und for st in self.book.values()}
        for sym in self.symbols(ctx):
            if sym in held:
                self._manage(ctx, sym)
            else:
                self._unwind_orphans(ctx, sym)
                self._scan(ctx, sym)

    # ------------------------------------------------------------------ data
    def _sync_inputs(self, ctx, sym):
        try:
            if ctx.portal().fidelity_of(sym, "quotes") > Fidelity.T0_QUOTE and not self.allow_synthetic:
                return None
        except Exception:
            return None
        q = ctx.last_quote(sym)
        if q is None or not (q.ask > q.bid > 0) or ctx.now - pd.Timestamp(q.ts) > self.QUOTE_MAX_AGE:
            return None
        ch = self.get_chain(ctx, sym, max_age=self.QUOTE_MAX_AGE, quiet=True)
        if ch is None:
            return None
        snap = pd.Timestamp(ch.attrs["snapshot_ts"])
        if abs((snap - pd.Timestamp(q.ts)).total_seconds()) > self.params["sync_s"] and not self.allow_synthetic:
            return None
        x = self.prepared(ctx, ch, sym)
        if x is None:
            return None
        return q, x

    def _divs(self, ctx, sym, expiry) -> list[tuple[float, float]]:
        today = ctx.session_date()
        sched = dividend_schedule(ctx, sym, pd.Timestamp(expiry), today)
        if any(not declared for _, _, declared in sched):
            self.flag_once(ctx, Fidelity.T2_PROXY, "S171: undeclared dividends projected from the regular schedule")
        return [((d - today).days / 365.0, a) for d, a, _ in sched]

    # ------------------------------------------------------------------ scan / enter
    def _scan(self, ctx, sym) -> None:
        inp = self._sync_inputs(ctx, sym)
        if inp is None:
            return
        q, x = inp
        r = float(x.attrs.get("r_cc", 0.0))
        shortable, bfee = self._borrow(ctx, sym)
        best = None
        mx = self.params["max_dte"]
        x = x[(x["dte"] >= 1) & ((x["dte"] <= mx) if mx is not None else True)]
        good = x[(x["bid"] > 0) & (x["ask"] > x["bid"])]
        skeys = [c for c in ("expiry", "root", "settlement") if c in good.columns]
        for sk, g in good.groupby(skeys):
            exp = pd.Timestamp(sk[0] if isinstance(sk, tuple) else sk)
            T = float(g["T"].iloc[0])
            divs = self._divs(ctx, sym, exp)
            american = str(g["exercise"].iloc[0]).upper() == "AMERICAN" if "exercise" in g.columns else True
            calls = g[g["right"] == "C"].set_index("strike")
            puts = g[g["right"] == "P"].set_index("strike")
            for K in sorted(set(calls.index) & set(puts.index)):
                c, p = calls.loc[K], puts.loc[K]
                if isinstance(c, pd.DataFrame):
                    c = c.iloc[0]
                if isinstance(p, pd.DataFrame):
                    p = p.iloc[0]
                res = parity_screen(q.bid, q.ask, float(c["bid"]), float(c["ask"]), float(p["bid"]), float(p["ask"]),
                                    float(K), T, r, divs, bfee, self._fee_ps(), self.params["min_edge_bp"], american)
                sig = res["signal"]
                if sig is None or (sig == "reversal" and not shortable):
                    continue
                e = res[sig]
                if best is None or e.edge_bp_of_spot > best[0].edge_bp_of_spot:
                    best = (e, sig, exp, float(K), T, divs, g, american)
        if best is None:
            return
        e, sig, exp, K, T, divs, g, american = best
        xe = pick_series(x, exp, settlement=str(g["settlement"].iloc[0]) if "settlement" in g.columns else None,
                         root=str(g["root"].iloc[0]) if "root" in g.columns else None)
        S_mid = 0.5 * (q.bid + q.ask)
        pkg = conversion(xe, K, sym, q.bid, q.ask, reverse=(sig == "reversal"))
        mult = pkg.multiplier
        n = int(min(self.params["max_contracts"], np.floor(self.params["notional_frac"] * ctx.nav / (S_mid * mult))))
        chk = SZ.arbitrage_edge(theoretical_value=(e.worst_case + e.outlay) * mult * n,
                                executable_cost=e.outlay * mult * n, shorts_located=(sig == "conversion" or shortable),
                                min_edge=self.params["min_edge_bp"] * 1e-4 * S_mid * mult * n)
        if n <= 0 or not chk.ok:
            return
        conv_pv, rev_req = self._locked_bounds(K, T, r, divs, bfee, american)
        min_edge_ps = self.params["min_edge_bp"] * 1e-4 * S_mid
        # combo legs as built by `conversion`: conversion = -C +P (net = P - C); reversal = +C -P (net = C - P)
        if sig == "conversion":
            limit = conv_pv - q.ask - self._fee_ps() - min_edge_ps
        else:
            limit = q.bid - self._fee_ps() - rev_req - min_edge_ps
        key = f"{sym}:{sig}:{exp.date()}:{K:g}:{ctx.now.isoformat()}"
        st = self.open_package(ctx, key, sym, pkg, n, "arb", limit=float(np.round(limit, 4)), tif=TimeInForce.IOC,
                               meta={"signal": sig, "edge_bp": e.edge_bp_of_spot, "worst_case_ps": e.worst_case,
                                     "K": K, "T": T, "expiry": exp, "divs": divs, "conv_pv": conv_pv,
                                     "rev_req": rev_req, "stock_done": 0.0, "stock_orders": [],
                                     "root": str(xe["root"].iloc[0]) if "root" in xe.columns else None,
                                     "settlement": str(xe["settlement"].iloc[0]) if "settlement" in xe.columns
                                     else None})
        if st is None:
            return
        st.stock_legs = [(ctx.instrument(sym), pkg.stock[0].shares)]

    @staticmethod
    def _locked_bounds(K, T, r, divs, borrow, american) -> tuple[float, float]:
        """(conversion: min over stopping times of PV received; reversal: max over stopping times of the
        liability discounted at the rebate rate r - borrow) per share."""
        stops = [T] + ([t for t, _ in divs if 0 < t <= T] if american else [])
        conv = min(K * np.exp(-r * t) + sum(a * np.exp(-r * tt) for tt, a in divs if 0 < tt < t) for t in stops)
        rr = r - borrow
        rstops = stops + ([0.0] if american else [])
        rev = max((K + sum(a * np.exp(rr * (t - tt)) for tt, a in divs if 0 < tt <= t)) * np.exp(-rr * t)
                  for t in rstops)
        return float(conv), float(rev)

    def _send_stock_legs(self, ctx) -> None:
        """Stock leg of each filled combo: marketable LIMIT keeping the locked edge >= 0."""
        for key, st in self.book.items():
            if not st.stock_legs or st.filled_units <= 0 or st.close_order is not None:
                continue
            inst, sh = st.stock_legs[0]
            todo = sh * st.filled_units - st.meta["stock_done"]
            if abs(todo) < 1e-9:
                continue
            net = st.entry_net                                  # option net per share actually paid (+) / received
            if st.meta["signal"] == "conversion":
                lim = st.meta["conv_pv"] - self._fee_ps() - net         # buy stock at <= lim
            else:
                lim = st.meta["rev_req"] + self._fee_ps() + net         # short stock at >= lim
            o = ctx.order(inst, todo, OrderType.LIMIT, limit_price=float(np.round(lim, 2 if sh > 0 else 2)),
                          tag=f"{self.id}:stock_leg")
            st.meta["stock_done"] += todo
            st.meta["stock_orders"].append(o)
            st.meta["stock_sent_at"] = ctx.now

    # ------------------------------------------------------------------ manage
    def _unwind_orphans(self, ctx, sym) -> None:
        q = ctx.position(sym)
        if abs(q) > 1e-9 and not any(o.instrument is not None and o.instrument.symbol == sym for o in ctx.open_orders(sym)):
            ctx.order(sym, -q, tag=f"{self.id}:orphan_unwind")

    def _close_all(self, ctx, key, st, reason) -> None:
        self.close_package(ctx, key, reason)
        qs = ctx.position(st.und)
        pend = any(o.instrument is not None and o.instrument.symbol == st.und for o in ctx.open_orders(st.und))
        if abs(qs) > 1e-9 and not pend:
            ctx.order(st.und, -qs, tag=f"{self.id}:close_stock:{reason}")

    def close_package(self, ctx, key, reason):
        st = self.book.get(key)
        if st is None:
            return None
        saved = st.stock_legs
        st.stock_legs = []            # stock handled by a separate quote-driven order
        try:
            return super().close_package(ctx, key, reason)
        finally:
            st.stock_legs = saved

    def _manage(self, ctx, sym) -> None:
        for key, st in [(k, s) for k, s in self.book.items() if s.und == sym]:
            if st.close_order is not None and st.close_order.status in ACTIVE:
                continue
            pos = self.positions_of(ctx, st)
            n_open = [abs(v) > 1e-9 for v in pos.values()]
            if st.open_order.status in ACTIVE:
                continue
            live = [o for o in st.meta.get("stock_orders", []) if o.status in ACTIVE]
            if live and st.meta.get("stock_sent_at") is not None and ctx.now > st.meta["stock_sent_at"]:
                for o in live:
                    ctx.cancel(o.id)                         # stock leg not filled by the next scan
                self._close_all(ctx, key, st, "broken_package")
                continue
            if live:
                continue
            exp_shares = st.stock_legs[0][1] * (st.filled_units or 0) if st.stock_legs else 0.0
            broken = (any(n_open) and not all(n_open)) or (abs(ctx.position(sym) - exp_shares) > 1e-6)
            if broken:
                self._close_all(ctx, key, st, "broken_package")
                continue
            inp = self._sync_inputs(ctx, sym)
            if inp is None:
                continue
            q, x = inp
            m = st.meta
            T = year_fraction(ctx.now, m["expiry"], "PHYSICAL")
            r = float(x.attrs.get("r_cc", 0.0))
            try:
                xe = pick_series(x, m["expiry"], settlement=m.get("settlement"), root=m.get("root"))
            except SelectionError:
                continue
            try:
                c, p = get_row(xe, m["K"], "C"), get_row(xe, m["K"], "P")
            except SelectionError:
                continue
            divs = self._divs(ctx, sym, m["expiry"])
            pv_div = sum(a * np.exp(-r * t) for t, a in divs if 0 < t <= T)
            liab = m["K"] * np.exp(-r * T) + pv_div
            if m["signal"] == "conversion":
                unwind = q.bid + float(p["bid"]) - float(c["ask"]) - self._fee_ps()
                converged = unwind >= liab
            else:
                cost = q.ask + float(p["ask"]) - float(c["bid"]) + self._fee_ps()
                converged = cost <= liab
            if converged:
                self._close_all(ctx, key, st, "converged")


# ============================================================================ S172
@register
class S172(OptionsEventBase):
    activity_note = ("At most one SPX box per month, only when the executable box rate beats T-bills after fees; "
                     "needs real SPX quotes (NOT_TESTABLE on free data).")
    id = "S172"
    name = NAMES["S172"]
    bar_freq = "1d"
    sizing = "OPT_ARBITRAGE"
    universe = ["SPX"]
    CHAIN_MAX_FID = Fidelity.T1_BAR
    params = {"dte": 30, "width": 50.0, "min_edge_bp": 0.0, "fee_per_contract": 1.40, "cash_frac": 0.50,
              "allow_synthetic": False}
    grid = {"dte": [30, 90, 180], "width": [25.0, 50.0, 100.0]}
    requires = [Requirement("options", "1d", symbols=("SPX",), min_fidelity=Fidelity.T1_BAR,
                            note="simultaneous executable SPX quotes with true timestamps")]
    deviations = COMMON_DEVIATIONS + (
        "'1mo': one long box at a time, opened at the close of the last session of the month when flat; held to "
        "cash settlement (engine option lifecycle; AM settlement priced at the SOQ proxy, flagged by the lifecycle).",
        "Only European, cash-settled SPX/SPXW series (exercise EUROPEAN, settlement CASH_AM/CASH_PM) are eligible; "
        "expiry closest to the DTE target within [0.5, 1.5] x target; K1 = listed strike closest to spot - width/2 "
        "with K2 = K1 + width listed, all four legs two-sided.",
        "Edge: box implied lending rate (natural debit + fees, continuous) minus the 3M T-bill rate converted from "
        "the portal's money-market yield to continuous over the box horizon; enter only if > min_edge_bp (0 = any "
        "positive edge after fees, OPT_ARBITRAGE). Margin: long box paid in full (no loan value, conservative).",
        "Size: cash_frac (50%) of NAV invested in boxes (sheet silent); fee_per_contract 1.40 (IBKR + Cboe SPX "
        "proprietary fee approx.).",
        "The long-box COMBO (DAY) carries the net limit w e^-(tbill_cc + min_edge) T_fill - 4 fee/multiplier, T_fill "
        "= horizon from the next close snapshot: it can only fill at a debit that still beats the bill.",
    )
    spec = SPECS["S172"]

    def on_bar(self, ctx) -> None:
        self.refresh(ctx)
        if self.book or not ctx.calendar().is_last_session_of_month(ctx.session_date()):
            return
        ch = self.get_chain(ctx, "SPX", session=ctx.session_date())
        if ch is None:
            return
        x = self.prepared(ctx, ch, "SPX")
        if x is None:
            return
        eu = x[(x["exercise"].astype(str).str.upper() == "EUROPEAN")
               & x["settlement"].astype(str).str.upper().isin(["CASH_AM", "CASH_PM"])]
        if eu.empty:
            self.skip(ctx, "no European cash-settled SPX series")
            return
        dte = float(self.params["dte"])
        try:
            xe = choose_series(eu, dte, int(0.5 * dte), int(1.5 * dte))
        except SelectionError as e:
            self.skip(ctx, str(e), activity="selection_failure")
            return
        exp = pd.Timestamp(xe["expiry"].iloc[0])
        xe = xe[(xe["bid"] > 0) & (xe["ask"] > xe["bid"])]
        S = float(np.nanmedian(xe["spot"])) if len(xe) else float("nan")
        w = float(self.params["width"])
        ks_c, ks_p = set(listed_strikes(xe, "C")), set(listed_strikes(xe, "P"))
        both = np.array(sorted(ks_c & ks_p))
        cand = [k for k in both if any(np.isclose(k + w, both))]
        if not cand or not np.isfinite(S):
            self.skip(ctx, f"no listed two-sided K1/K1+{w:g} pair")
            return
        k1 = float(min(cand, key=lambda k: (abs(k - (S - w / 2)), k)))
        k2 = k1 + w
        pkg = box(xe, k1, k2)
        legs = {(l.strike, l.right): l for l in pkg.legs}
        quotes = dict(c1_bid=legs[(k1, "C")].bid, c1_ask=legs[(k1, "C")].ask,
                      c2_bid=legs[(k2, "C")].bid, c2_ask=legs[(k2, "C")].ask,
                      p1_bid=legs[(k1, "P")].bid, p1_ask=legs[(k1, "P")].ask,
                      p2_bid=legs[(k2, "P")].bid, p2_ask=legs[(k2, "P")].ask)
        settle = str(xe["settlement"].iloc[0])
        T = year_fraction(ctx.now, exp, settle)
        try:
            rs = ctx.portal().risk_free_rate(tenor="3M")
        except Exception:
            rs = float("nan")
        if not np.isfinite(rs) or T <= 0:
            self.skip(ctx, "T-bill rate unavailable")
            return
        tb_cc = float(np.log1p(rs * T * 365.0 / 360.0) / T)
        res = box_vs_tbill(quotes, k1, k2, T, tb_cc, fee_per_contract=self.params["fee_per_contract"],
                           multiplier=pkg.multiplier)
        debit = res["prices"]["buy_debit"]
        ctx.log("S172 box", k1=k1, k2=k2, debit=debit, lend_cc=res["lend_rate_cc"], tbill_cc=tb_cc,
                edge_bp=res["lend_minus_tbill_bp"])
        if not (np.isfinite(res["lend_minus_tbill_bp"]) and res["lend_minus_tbill_bp"] > self.params["min_edge_bp"]
                and debit > 0):
            return
        n = int(np.floor(self.params["cash_frac"] * ctx.nav / (debit * pkg.multiplier)))
        fees = self.params["fee_per_contract"] * 4 * n
        chk = SZ.arbitrage_edge(theoretical_value=w * pkg.multiplier * n * np.exp(-tb_cc * T),
                                executable_cost=debit * pkg.multiplier * n, fees=fees)
        if n <= 0 or not chk.ok:
            return
        # net limit for the fill at the NEXT snapshot: width discounted at (T-bill + min edge) over the
        # remaining horizon at that snapshot, minus the 4 legs' fees per share
        cal = ctx.calendar()
        t_fill = cal.open_close(cal.next_session(ctx.session_date()))[1]
        T_fill = max(year_fraction(t_fill, exp, settle), 1e-6)
        limit = w * np.exp(-(tb_cc + self.params["min_edge_bp"] * 1e-4) * T_fill) \
            - 4.0 * self.params["fee_per_contract"] / pkg.multiplier
        self.open_package(ctx, f"BOX:{exp.date()}:{k1:g}-{k2:g}", "SPX", pkg, n, "arb", limit=float(np.round(limit, 4)),
                          meta={"limit": limit, "lend_cc": res["lend_rate_cc"], "tbill_cc": tb_cc,
                                "edge_bp": res["lend_minus_tbill_bp"], "debit": debit})
