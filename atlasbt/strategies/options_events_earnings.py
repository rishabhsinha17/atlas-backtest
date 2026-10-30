"""Earnings option strategies S158 (long straddle), S159 (short iron condor),
S160 (post-event short put vertical).  See ``options_events_common`` for the
family conventions.

Event timing (PIT):
* The pre-release entries (S158/S159) need the release to be known IN ADVANCE:
  they read ``EARNINGS_SCHEDULE`` events (expected release instant in
  ``effective_at``, ``payload['timing']`` BMO/AMC, known before the release) and
  confirm the ACTUAL release with ``EARNINGS`` events (``known_at`` = release
  timestamp).  A release already published before the entry instant cancels the
  entry (it would no longer be pre-release).
* Entry "15:30 ET day before an after-close release / 15:30 prior session for a
  before-open release" = the LAST session entry time strictly before the expected
  release (AMC on D -> D 15:30; BMO on D -> D-1 15:30).  DURING / UNKNOWN timings
  are not covered by the sheet text and are skipped.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..core.clock import session_date_of, to_utc
from ..core.types import Fidelity
from ..engine.strategy import Requirement
from ..indicators import opening_range
from ..options.chain import SelectionError, choose_expiry
from ..options.structures import iron_condor, straddle, vertical
from ..options.margin import package_margin  # noqa: F401  (documented dependency for margin checks)
from ..risk import sizing as SZ
from .options_events_common import (COMMON_DEVIATIONS, choose_series, pick_series, OptionsEventBase, at_label, is_close_bar, label_ts,
                                    pct_rank_prev, resample_bars, session_close)
from .options_events_specs import NAMES, SPECS
from .registry import register

EARN_EVENTS_REQ = (
    Requirement("events", "EARNINGS_SCHEDULE", symbols=("*",),
                note="expected release date/time + BMO/AMC known BEFORE the release (pre-release entries)"),
    Requirement("events", "EARNINGS", symbols=("*",), note="actual release timestamp (exit after the release)"),
)
INTRADAY_OPT_REQ = Requirement(
    "options", "1m", symbols=("*",), min_fidelity=Fidelity.T1_BAR,
    note="timestamped intraday option quotes for the earnings universe (e.g. Databento OPRA); EOD snapshots "
         "cannot provide 15:30 / 09:45 executable quotes")


def _payload(ev, key, default=None):
    p = getattr(ev, "payload", None) or {}
    return p.get(key, default)


class _EarningsBase(OptionsEventBase):
    """Pre-release scheduler shared by S158/S159."""

    activity_note = ("Pre-earnings entries need intraday timestamped option quotes (T0/T1) for each name; EOD (DoltHub "
                     "T2) and synthetic chains are refused -> NOT_TESTABLE / zero trades on free data by design.")
    bar_freq = "5m"
    universe = ()
    universe_name = "SP500"
    CHAIN_MAX_FID = Fidelity.T1_BAR
    QUOTE_MAX_AGE = pd.Timedelta(minutes=5)
    requires = list(EARN_EVENTS_REQ) + [INTRADAY_OPT_REQ]

    # ------------------------------------------------------------------ scheduling
    def _members(self, ctx) -> Optional[set]:
        try:
            u = ctx.portal().universe(self.universe_name)
        except Exception:
            return None
        return set(u) if u else None

    def _entry_instant(self, ctx, release: pd.Timestamp) -> Optional[pd.Timestamp]:
        cal = ctx.calendar()
        lab = self.params["entry"]
        d = session_date_of(release)
        s = d if cal.is_session(d) else cal.prev_session(d)
        for _ in range(5):
            t = label_ts(cal, s, lab)
            if t < release:
                return t
            s = cal.prev_session(s)
        return None

    def _actual_release(self, ctx, sym: str, expected: pd.Timestamp) -> Optional[pd.Timestamp]:
        try:
            evs = ctx.portal().events("EARNINGS", [sym], expected - pd.Timedelta(days=4), ctx.now + pd.Timedelta(days=1))
        except Exception:
            return None
        ts = [to_utc(e.known_at) for e in evs if e.symbol == sym and
              abs(to_utc(e.known_at) - expected) <= pd.Timedelta(days=3) and to_utc(e.known_at) <= ctx.now]
        return min(ts) if ts else None

    def _due_entries(self, ctx) -> list:
        try:
            evs = ctx.portal().events("EARNINGS_SCHEDULE", None, ctx.now - pd.Timedelta(days=20),
                                      ctx.now + pd.Timedelta(days=30))
        except Exception as e:
            self.skip(ctx, f"EARNINGS_SCHEDULE unavailable: {e}", activity="selection_failure")
            return []
        members = self._members(ctx)
        out = []
        for ev in evs:
            sym = ev.symbol
            if not sym or (members is not None and sym not in members):
                continue
            timing = str(_payload(ev, "timing", "UNKNOWN") or "UNKNOWN").upper()
            if timing not in ("BMO", "AMC"):
                continue
            rel = to_utc(ev.effective_at) if ev.effective_at is not None else to_utc(ev.known_at)
            if rel <= ctx.now:
                continue
            t_in = self._entry_instant(ctx, rel)
            if t_in is None or t_in != ctx.now:
                continue
            if int(getattr(ev, "fidelity", Fidelity.T1_BAR)) >= int(Fidelity.T2_PROXY):
                self.flag_once(ctx, Fidelity.T2_PROXY, f"{self.id}: expected release time approximated by the "
                                                       f"earnings calendar (BMO 08:00 / AMC 16:30)")
            out.append((sym, rel, timing))
        return sorted(set(out))

    def _post_release_expiry(self, ctx, x: pd.DataFrame, release: pd.Timestamp) -> pd.Timestamp:
        cal = ctx.calendar()
        d = session_date_of(release)
        react = d if (cal.is_session(d) and release < cal.open_close(d)[0]) else cal.next_session(d)
        lo, hi = self.dte_window()
        cand = x[(x["expiry"] >= react) & (x["dte"] >= lo) & (x["dte"] <= hi)]
        if cand.empty:
            raise SelectionError(f"no post-release expiry with DTE in [{lo},{hi}]")
        return choose_expiry(cand, self.dte_target(), lo, hi, prefer_standard=False)

    def dte_window(self) -> tuple[int, int]:
        return 3, 10

    def dte_target(self) -> float:
        return 3.0

    # ------------------------------------------------------------------ loop
    def on_bar(self, ctx) -> None:
        self.refresh(ctx)
        self._unwind_residual_stock(ctx)
        if at_label(ctx, self.params["exit"]):
            self._exits(ctx)
        for sym, rel, timing in self._due_entries(ctx):
            key = f"{sym}:{rel.isoformat()}"
            if key in self._done or key in self.book:
                continue
            self._done.add(key)
            if self._actual_release(ctx, sym, rel) is not None:
                self.skip(ctx, f"{sym}: release already published before the entry time")
                continue
            self._enter(ctx, key, sym, rel, timing)

    def _exits(self, ctx) -> None:
        cal = ctx.calendar()
        for key, st in list(self.book.items()):
            if st.close_order is not None or not self.is_open(ctx, st):
                continue
            rel = self._actual_release(ctx, st.und, st.meta["expected_release"])
            if rel is not None and rel < ctx.now:
                self.close_package(ctx, key, "post_release_exit")
                continue
            exp = min(pd.Timestamp(i.expiry) for i in st.insts)
            if ctx.session_date() >= exp:
                ctx.log(f"{self.id} {key}: release not confirmed by expiry session; closing")
                self.close_package(ctx, key, "expiry_session_exit")

    def _unwind_residual_stock(self, ctx) -> None:
        """Shares delivered by an exercise/assignment are not part of the strategy: sell/cover them
        at the next bar (only for underlyings without an open package)."""
        held = {st.und for st in self.book.values()}
        for sym, q in ctx.positions().items():
            if abs(q) < 1e-9 or ctx.instrument(sym).is_option or sym in held:
                continue
            if any(o.instrument is not None and o.instrument.symbol == sym for o in ctx.open_orders(sym)):
                continue
            ctx.order(sym, -q, tag="assignment_unwind")

    def _enter(self, ctx, key, sym, rel, timing) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


# ============================================================================ S158
@register
class S158(_EarningsBase):
    id = "S158"
    name = NAMES["S158"]
    sizing = "OPT_DEBIT"
    params = {"dte": 3, "entry": "15:30", "exit": "09:45", "allow_synthetic": False}
    grid = {"dte": [3, 7, 10], "entry": ["15:00", "15:30"], "exit": ["09:35", "09:45"]}
    deviations = COMMON_DEVIATIONS + (
        "'15:30 day before an after-close release' = the last session's entry time strictly before the expected "
        "release (AMC on D -> D 15:30; BMO on D -> D-1 15:30); DURING/UNKNOWN timings skipped (sheet silent).",
        "Grid 'DTE 3/7/10' = target DTE: among post-release expiries (expiring on/after the first session that "
        "trades after the release) with 3 <= DTE <= 10, the one closest to the target (ties: earlier); baseline 3 = "
        "the nearest post-release expiry at 3-10 DTE.",
        "Exit at the first 'exit' time (09:45) after the ACTUAL release (EARNINGS event); if the release is not "
        "confirmed by the expiry session the straddle is closed at that session's exit time.",
        "Universe: point-in-time S&P 500 members (portal universe 'SP500'); base bars 5m (15:00/15:30/09:35/09:45).",
        "Shares delivered by exercise are unwound at the next bar (not part of the strategy).",
    )
    spec = SPECS["S158"]

    def dte_target(self) -> float:
        return float(self.params["dte"])

    def _enter(self, ctx, key, sym, rel, timing) -> None:
        ch = self.get_chain(ctx, sym, max_age=self.QUOTE_MAX_AGE)
        if ch is None:
            return
        x = self.prepared(ctx, ch, sym)
        if x is None:
            return
        try:
            exp = self._post_release_expiry(ctx, x, rel)
            xe = pick_series(x, exp)
            pkg = straddle(xe, 1, usable=self.usable(xe))
        except SelectionError as e:
            self.skip(ctx, f"{sym}: {e}", activity="selection_failure")
            return
        debit = pkg.natural_open()
        if not (debit > 0):
            self.skip(ctx, f"{sym}: non-positive straddle debit")
            return
        n = SZ.size_opt_debit(ctx.nav, debit * pkg.multiplier, existing_option_debit=self.open_option_debit(ctx))
        self.open_package(ctx, key, sym, pkg, n, "debit",
                          meta={"expected_release": rel, "timing": timing, "planned_debit_usd": n * debit * pkg.multiplier,
                                "expiry": str(exp.date())})


# ============================================================================ S159
@register
class S159(_EarningsBase):
    id = "S159"
    name = NAMES["S159"]
    sizing = "OPT_CREDIT"
    params = {"short_delta": 0.20, "wing_delta": 0.05, "dte_range": (3, 10), "entry": "15:30", "exit": "09:45",
              "allow_synthetic": False}
    grid = {"short_delta": [0.15, 0.20, 0.25], "wing_delta": [0.05, 0.10], "dte_range": [(3, 10)]}
    deviations = COMMON_DEVIATIONS + (
        "'Before scheduled earnings' entry time unspecified: same rule as S158 (last 15:30 strictly before the "
        "expected release).",
        "'Same nearest post-release expiry' = nearest expiry on/after the first post-release session with DTE in "
        "dte_range (grid 'DTE 3-10' is one range value).",
        "'Only if credit>0 and protected wings': natural credit > 0 at the decision snapshot, wings strictly outside "
        "the shorts, defined risk (no naked leg); the COMBO carries a limit of -0.01 (credit >= $0.01 at the fill).",
        "'Exit next day 09:45' = first 09:45 after the actual release; unfilled / held positions are settled by the "
        "engine's option lifecycle at expiry. Assignment after an extreme gap (early exercise of deep ITM American "
        "legs) is simulated by the engine's default option lifecycle (OccOptionLifecycle with early assignment); "
        "delivered shares are unwound at the next bar.",
    )
    spec = SPECS["S159"]

    def dte_window(self) -> tuple[int, int]:
        lo, hi = self.params["dte_range"]
        return int(lo), int(hi)

    def dte_target(self) -> float:
        return float(self.dte_window()[0])

    def _enter(self, ctx, key, sym, rel, timing) -> None:
        ch = self.get_chain(ctx, sym, max_age=self.QUOTE_MAX_AGE)
        if ch is None:
            return
        x = self.prepared(ctx, ch, sym)
        if x is None:
            return
        try:
            exp = self._post_release_expiry(ctx, x, rel)
            xe = pick_series(x, exp)
            pkg = iron_condor(xe, self.params["short_delta"], wing_delta=self.params["wing_delta"],
                              usable=self.usable(xe))
        except SelectionError as e:
            self.skip(ctx, f"{sym}: {e}", activity="selection_failure")
            return
        credit = -pkg.natural_open()
        if not (credit > 0):
            self.skip(ctx, f"{sym}: no credit at natural prices")
            return
        legs = [SZ.OptLeg(l.instrument(), l.ratio, l.bid if l.ratio < 0 else l.ask) for l in pkg.legs]
        ok, why = SZ.is_defined_risk(legs)
        if not ok:
            self.skip(ctx, f"{sym}: {why}")
            return
        ml = pkg.max_profit_loss()["max_loss_usd"]
        n, _coll = SZ.size_opt_credit(ctx.nav, ml, legs=legs)
        self.open_package(ctx, key, sym, pkg, n, "credit", limit=-0.01,
                          meta={"expected_release": rel, "timing": timing, "credit": credit, "max_loss_usd": ml,
                                "expiry": str(exp.date())})


# ============================================================================ S160
@register
class S160(OptionsEventBase):
    activity_note = ("Needs intraday option quotes at 10:00 and 5m bars (or the T2 close timing proxy), a 252-day "
                     "real IV history per name (real chains from 2020 -> first signals 2021), IV pct > 70 AND a "
                     "break above the opening range: low activity expected.")
    id = "S160"
    name = NAMES["S160"]
    bar_freq = "5m"
    sizing = "OPT_CREDIT"
    universe = ()
    universe_name = "SP500"
    CHAIN_MAX_FID = Fidelity.T1_BAR
    QUOTE_MAX_AGE = pd.Timedelta(minutes=5)
    #: the 252-day IV history is a series of DAILY CLOSE values: real EOD snapshots (DoltHub, T2) are accepted
    #: for it; the 10:00 decision itself still needs a timestamped (T0/T1) snapshot
    HISTORY_MAX_FID = Fidelity.T2_PROXY
    params = {"iv_pct": 70, "short_delta": 0.20, "wing_delta": 0.10, "dte": 30, "tp": 0.50, "exit_dte": 7,
              "low_n": 5, "entry": "10:00", "or_minutes": 30, "timing_proxy": False, "allow_synthetic": False}
    grid = {"iv_pct": [60, 70, 80], "short_delta": [0.20, 0.30]}
    requires = [Requirement("events", "EARNINGS", symbols=("*",), note="actual release timestamp"),
                Requirement("bars", "5m", symbols=("*",), note="intraday bars for the 30-minute opening range"),
                INTRADAY_OPT_REQ]
    deviations = COMMON_DEVIATIONS + (
        "'First next trading-day 10:00 after earnings release' = the first session whose 10:00 ET is after the "
        "actual release timestamp (BMO on D -> D 10:00; AMC on D -> D+1 10:00).",
        "IV percentile = 30D constant-maturity ATM IV (real chain, total-variance interpolation) at the 10:00 snapshot "
        "vs the previous 252 valid daily close values of the same name (percentile_rank, previous values); the "
        "daily close history comes from real EOD snapshots (DoltHub, T2, flagged) -- intraday OPRA samples are only "
        "fetched around the events.",
        "'Only when stock closes above opening range' = opening range = first 30 minutes (09:30-10:00, the 10:00 "
        "decision bar completes it); the vertical is sold at the first 5m bar close ABOVE the OR high after 10:00 in "
        "the same session (one entry per release); no entry if that never happens.",
        "'0.20delta put vertical': long leg unspecified by the sheet -> 0.10-delta put (as S169's -.25/-.10 put spread "
        "of the same family); 30DTE = expiry closest to 30 DTE within 20-45 DTE.",
        "Exits evaluated at each session close on the close snapshot: 50% of the actual credit captured at natural "
        "close prices, near DTE <= 7, or underlying close below the lowest low of the previous 5 sessions.",
        "timing_proxy=True (opt-in, T2_PROXY, per docs/OPTIONS.md §13): decision at the post-release session CLOSE "
        "on the real EOD snapshot instead of 10:00 (OR condition = session close > OR high); the options requirement "
        "becomes EOD (T2) and non-critical (per-name runtime gate).",
        "Universe: point-in-time S&P 500 members (portal universe 'SP500').",
    )
    spec = SPECS["S160"]

    def __init__(self, **params):
        super().__init__(**params)
        self.proxy = bool(self.params["timing_proxy"])
        if self.proxy:
            self.CHAIN_MAX_FID = Fidelity.T2_PROXY
            reqs = [r for r in type(self).requires if r.kind != "options"]
            reqs.append(Requirement("options", "1d", symbols=("*",), critical=False, min_fidelity=Fidelity.T2_PROXY,
                                    note="real EOD snapshots (timing proxy)"))
            self.requires = reqs
        self._armed: dict[str, dict] = {}

    # ------------------------------------------------------------------ helpers
    def _releases_today(self, ctx) -> list:
        cal = ctx.calendar()
        try:
            evs = ctx.portal().events("EARNINGS", None, ctx.now - pd.Timedelta(days=6), ctx.now + pd.Timedelta(days=1))
        except Exception:
            return []
        try:
            members = set(ctx.portal().universe(self.universe_name)) or None
        except Exception:
            members = None
        today = ctx.session_date()
        out = []
        for ev in evs:
            sym = ev.symbol
            if not sym or (members is not None and sym not in members):
                continue
            rel = to_utc(ev.known_at)
            if rel > ctx.now:
                continue
            d = session_date_of(rel)
            s = d if cal.is_session(d) else cal.next_session(d)
            if label_ts(cal, s, self.params["entry"]) <= rel:
                s = cal.next_session(s)
            if s == today:
                out.append((sym, rel))
        return sorted(set(out))

    def _iv_pct(self, ctx, sym: str, ch: pd.DataFrame) -> float:
        """Percentile (0-100) of the current 30D ATM IV vs the previous LOOKBACK daily close values."""
        from ..options.metrics import chain_metrics
        hist = self.metric_series(ctx, sym, "atm_iv_30")
        hist = hist[hist.index < ctx.session_date()]
        try:
            m = chain_metrics(ch, ctx.now, self.rate_cc(ctx), spot=self.spot(ctx, sym, ch))
            cur = m.get("atm_iv_30", np.nan) if m.get("flag_30") in ("interp", "exact") else np.nan
        except Exception:
            cur = np.nan
        if not np.isfinite(cur) or len(hist) < self.LOOKBACK:
            return float("nan")
        vals = np.r_[hist.values[-self.LOOKBACK:], cur]
        return pct_rank_prev(vals, self.LOOKBACK)

    def _prior_lows(self, ctx, sym: str, n: int) -> pd.DataFrame:
        d = ctx.history(sym, n + 2, "1d", adjusted=True)
        if d is None or len(d) == 0:
            return pd.DataFrame()
        sess = pd.DatetimeIndex([session_date_of(t) for t in d.index])
        return d[sess < ctx.session_date()].iloc[-n:]

    def _or_high(self, ctx, sym: str) -> float:
        try:
            b = ctx.session_bars(sym, self.bar_freq)
        except Exception:
            return float("nan")
        if b is None or len(b) == 0:
            return float("nan")
        orr = opening_range(b, minutes=int(self.params["or_minutes"]))
        v = orr["or_high"].dropna()
        return float(v.iloc[-1]) if len(v) else float("nan")

    def _last_close(self, ctx, sym: str) -> float:
        try:
            b = ctx.session_bars(sym, self.bar_freq)
        except Exception:
            return float("nan")
        return float(b["close"].iloc[-1]) if b is not None and len(b) else float("nan")

    # ------------------------------------------------------------------ loop
    def on_bar(self, ctx) -> None:
        self.refresh(ctx)
        today = ctx.session_date()
        decision = is_close_bar(ctx) if self.proxy else at_label(ctx, self.params["entry"])
        if decision:
            for sym, rel in self._releases_today(ctx):
                key = f"{sym}:{rel.isoformat()}"
                if key in self._done or key in self.book or sym in {s.und for s in self.book.values()}:
                    continue
                self._done.add(key)
                self._decide(ctx, key, sym, rel)
        # intraday trigger: first 5m close above the OR high after the 10:00 decision
        if not self.proxy:
            for key, a in list(self._armed.items()):
                if a["session"] != today:
                    self._armed.pop(key)
                    continue
                if ctx.now <= a["decided_at"]:
                    continue
                c, orh = self._last_close(ctx, a["sym"]), self._or_high(ctx, a["sym"])
                if np.isfinite(c) and np.isfinite(orh) and c > orh:
                    self._armed.pop(key)
                    self._enter(ctx, key, a["sym"], a)
        if is_close_bar(ctx):
            self._exits(ctx)

    def _decide(self, ctx, key, sym, rel) -> None:
        ch = self.get_chain(ctx, sym, max_age=None if self.proxy else self.QUOTE_MAX_AGE,
                            session=ctx.session_date() if self.proxy else None)
        if ch is None:
            return
        if pd.Timestamp(ch.attrs["snapshot_ts"]) <= rel:
            self.skip(ctx, f"{sym}: latest snapshot predates the release")
            return
        if self.proxy:
            self.flag_once(ctx, Fidelity.T2_PROXY, "S160 timing_proxy: post-release session close snapshot used "
                                                   "for the 10:00 decision")
        x = self.prepared(ctx, ch, sym)
        if x is None:
            return
        pct = self._iv_pct(ctx, sym, ch)
        hit = bool(np.isfinite(pct) and pct > self.params["iv_pct"])
        ctx.log(f"{self.id} {sym} post-release IV pct", pct=pct, **({} if hit else {"activity": "no_signal"}))
        if not (np.isfinite(pct) and pct > self.params["iv_pct"]):
            return
        a = {"sym": sym, "session": ctx.session_date(), "decided_at": ctx.now, "rel": rel, "pct": pct}
        if self.proxy:
            c, orh = self._last_close(ctx, sym), self._or_high(ctx, sym)
            if np.isfinite(c) and np.isfinite(orh) and c > orh:
                self._enter(ctx, key, sym, a, x=x)
        else:
            self._armed[key] = a

    def _enter(self, ctx, key, sym, a, x: Optional[pd.DataFrame] = None) -> None:
        if x is None:
            ch = self.get_chain(ctx, sym, max_age=self.QUOTE_MAX_AGE)
            if ch is None:
                return
            x = self.prepared(ctx, ch, sym)
            if x is None:
                return
        try:
            xe = choose_series(x, self.params["dte"], 20, 45)
            exp = pd.Timestamp(xe["expiry"].iloc[0])
            pkg = vertical(xe, -abs(self.params["wing_delta"]), -abs(self.params["short_delta"]),
                           usable=self.usable(xe), name="put_credit_vertical")
        except SelectionError as e:
            self.skip(ctx, f"{sym}: {e}", activity="selection_failure")
            return
        credit = -pkg.natural_open()
        if not (credit > 0):
            self.skip(ctx, f"{sym}: no credit")
            return
        legs = [SZ.OptLeg(l.instrument(), l.ratio, l.bid if l.ratio < 0 else l.ask) for l in pkg.legs]
        ml = SZ.max_loss_vertical(legs)
        n, _ = SZ.size_opt_credit(ctx.nav, ml, legs=legs)
        self.open_package(ctx, key, sym, pkg, n, "credit",
                          meta={"credit": credit, "max_loss_usd": ml, "iv_pct": a.get("pct"), "expiry": str(exp.date())})

    def _exits(self, ctx) -> None:
        for key, st in list(self.book.items()):
            if st.close_order is not None or not self.is_open(ctx, st) or st.filled_units <= 0:
                continue
            reason = None
            exp = min(pd.Timestamp(i.expiry) for i in st.insts)
            if (exp - ctx.session_date()).days <= self.params["exit_dte"]:
                reason = "dte"
            if reason is None:
                n = int(self.params["low_n"])
                prev = self._prior_lows(ctx, st.und, n)
                c = self._last_close(ctx, st.und)
                if len(prev) == n and np.isfinite(c) and c < float(prev["low"].min()):
                    reason = "below_5d_low"
            if reason is None:
                ch = self.get_chain(ctx, st.und, max_age=self.QUOTE_MAX_AGE if not self.proxy else None,
                                    session=ctx.session_date() if self.proxy else None, quiet=True)
                x = self.prepared(ctx, ch, st.und) if ch is not None else None
                v, fid, how = self.liquidation_value(ctx, st, self.QUOTE_MAX_AGE if not self.proxy else
                                                     pd.Timedelta(days=1), x)
                if how != "quotes" and np.isfinite(v):
                    self.flag_once(ctx, Fidelity(min(max(fid, 2), 3)), f"{self.id}: exit decision on surface marks "
                                                                        f"(contract missing from snapshot)")
                credit = -st.entry_cash
                if np.isfinite(v) and credit > 0 and (v + credit) >= self.params["tp"] * credit:
                    reason = "take_profit"
            if reason:
                self.close_package(ctx, key, reason)
