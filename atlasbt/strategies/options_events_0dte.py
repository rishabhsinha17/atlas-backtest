"""SPX 0DTE strategies S161 (20-delta iron condor), S162 (put credit spread),
S163 (call debit breakout), S164 (FOMC long straddle).  Family conventions in
``options_events_common``.

0DTE conventions:
* "Confirmed PM-settled SPX 0DTE expiry" = the live chain snapshot lists an
  expiry equal to today's session with settlement CASH_PM (SPXW).  The listing
  calendar (``options.listing.has_0dte``) is only cross-checked (logged), the
  chain is authoritative.
* Decisions need a chain snapshot at most 2 minutes old with fidelity T0/T1
  (timestamped executable quotes); intraday monitoring (take-profit / stop) uses
  per-leg quote snapshots of the same age limit.
* "Close before 15:30" (S161) = closing order at the bar ending 15:29; "or 15:30"
  (S162/S163) = at the bar ending 15:30 (both shift with early closes).  A
  position still open at the close is PM cash-settled by the engine's option
  lifecycle (official close).
* SPX is an index without traded volume, so "SPX above VWAP" cannot be computed
  from SPX prints: by default it is evaluated on SPY (SPY close vs SPY session
  VWAP), declared and flagged T2_PROXY (``vwap_source='SPX'`` uses the SPX bars'
  own volume if a vendor provides it).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..core.clock import et, session_date_of, to_utc
from ..core.types import Fidelity
from ..engine.strategy import Requirement
from ..indicators import opening_range, session_vwap
from ..options.chain import SelectionError
from ..options.listing import has_0dte
from ..options.structures import iron_condor, straddle, vertical
from ..risk import sizing as SZ
from .options_events_common import (COMMON_DEVIATIONS, pick_series, OptionsEventBase, at_label, label_ts, resample_bars)
from .options_events_specs import NAMES, SPECS
from .registry import register

SPX = "SPX"
ZDTE_DEVIATIONS = COMMON_DEVIATIONS + (
    "0DTE expiry confirmed from the live chain (expiry == today, settlement CASH_PM); listing calendar only "
    "cross-checked.",
    "Decisions need a T0/T1 chain snapshot <= 2 min old; take-profit / stop monitored every minute on per-leg "
    "quote snapshots (natural close prices).",
)


class _ZeroDTEBase(OptionsEventBase):
    bar_freq = "1m"
    universe = [SPX]
    CHAIN_MAX_FID = Fidelity.T1_BAR
    QUOTE_MAX_AGE = pd.Timedelta(minutes=2)
    EXIT_LABEL = "15:30"
    requires = [Requirement("bars", "1m", symbols=(SPX,), note="SPX 1-minute index bars"),
                Requirement("options", "1m", symbols=(SPX,), min_fidelity=Fidelity.T1_BAR,
                            note="timestamped intraday SPXW quotes (e.g. Databento OPRA)")]

    def __init__(self, **params):
        super().__init__(**params)
        src = self.params.get("vwap_source")
        if src and src != SPX:
            self.requires = list(self.requires) + [Requirement("bars", "1m", symbols=(src,),
                                                               note=f"VWAP proxy source {src}")]

    # ------------------------------------------------------------------ chain
    def zero_dte(self, ctx) -> Optional[pd.DataFrame]:
        today = ctx.session_date()
        ch = self.get_chain(ctx, SPX, max_age=self.QUOTE_MAX_AGE)
        if ch is None:
            return None
        x = self.prepared(ctx, ch, SPX)
        if x is None:
            return None
        xe = x[(x["expiry"] == today) & (x["settlement"].astype(str) == "CASH_PM")]
        if xe.empty:
            self.skip(ctx, "no PM-settled 0DTE expiry in the chain")
            return None
        try:
            xe = pick_series(xe, today, settlement="CASH_PM")
        except SelectionError as e:
            self.skip(ctx, str(e))
            return None
        if not has_0dte(SPX, today):
            ctx.log(f"{self.id}: chain lists a 0DTE PM expiry the listing calendar does not model", day=str(today))
        return xe

    # ------------------------------------------------------------------ VWAP / bars
    def vwap_state(self, ctx, minutes: int = 1) -> tuple[float, float]:
        """(last close, session VWAP) of the VWAP source (bars of ``minutes``, causal)."""
        src = self.params.get("vwap_source", SPX)
        if src != SPX:
            self.flag_once(ctx, Fidelity.T2_PROXY, f"{self.id}: SPX VWAP evaluated on {src} (index has no volume)")
        b = ctx.session_bars(src, "1m")
        if b is None or len(b) == 0:
            return float("nan"), float("nan")
        if minutes > 1:
            b = resample_bars(b, minutes, ctx.now, ctx.calendar().open_close(ctx.session_date())[0])
            if len(b) == 0:
                return float("nan"), float("nan")
        if not (b["volume"].fillna(0) > 0).any():
            self.skip(ctx, f"VWAP source {src} has no volume")
            return float(b["close"].iloc[-1]), float("nan")
        vw = session_vwap(b)["vwap"]
        return float(b["close"].iloc[-1]), float(vw.iloc[-1])

    # ------------------------------------------------------------------ monitoring
    def monitor(self, ctx) -> None:
        exit_now = ctx.now >= label_ts(ctx.calendar(), ctx.session_date(), self.EXIT_LABEL)
        for key, st in list(self.book.items()):
            if st.close_order is not None or not self.is_open(ctx, st):
                continue
            if exit_now:
                self.close_package(ctx, key, "time_exit")
                continue
            if st.filled_units <= 0:
                continue
            v, _fid, how = self.liquidation_value(ctx, st, self.QUOTE_MAX_AGE)
            if not np.isfinite(v):
                continue
            reason = self.exit_rule(ctx, st, v)
            if reason:
                self.close_package(ctx, key, reason)

    def exit_rule(self, ctx, st, v: float) -> Optional[str]:
        return None

    def credit_exit(self, st, v: float) -> Optional[str]:
        credit = -st.entry_cash
        if credit <= 0:
            return None
        if v + credit >= self.params["tp"] * credit:
            return "take_profit"
        if -v >= self.params["stop_mult"] * credit:
            return "stop"
        return None

    def before_exit_label(self, ctx) -> bool:
        return ctx.now < label_ts(ctx.calendar(), ctx.session_date(), self.EXIT_LABEL)

    def size_credit(self, ctx, pkg) -> tuple[int, float]:
        legs = [SZ.OptLeg(l.instrument(), l.ratio, l.bid if l.ratio < 0 else l.ask) for l in pkg.legs]
        ok, why = SZ.is_defined_risk(legs)
        if not ok:
            return 0, float("nan")
        ml = pkg.max_profit_loss()["max_loss_usd"]
        n, _ = SZ.size_opt_credit(ctx.nav, ml, legs=legs)
        return n, ml


# ============================================================================ S161
@register
class S161(_ZeroDTEBase):
    id = "S161"
    name = NAMES["S161"]
    sizing = "OPT_CREDIT"
    EXIT_LABEL = "15:29"
    params = {"entry": "10:00", "short_delta": 0.20, "wing_delta": 0.05, "tp": 0.50, "stop_mult": 2.0,
              "allow_synthetic": False}
    grid = {"entry": ["09:45", "10:00", "11:00"], "short_delta": [0.10, 0.20, 0.25], "tp": [0.25, 0.50]}
    deviations = ZDTE_DEVIATIONS + (
        "'2x credit debit stop' = close when the natural cost to close >= 2 x the credit received; 'TP 50%' = close "
        "when (credit - cost to close) >= 50% of the credit.",
        "'Close both verticals before 15:30' = one COMBO closing all four legs at the bar ending 15:29.",
    )
    spec = SPECS["S161"]

    def on_bar(self, ctx) -> None:
        self.refresh(ctx)
        self.monitor(ctx)
        day = ctx.session_date()
        if at_label(ctx, self.params["entry"]) and day not in self._done and self.before_exit_label(ctx):
            self._done.add(day)
            xe = self.zero_dte(ctx)
            if xe is None:
                return
            try:
                pkg = iron_condor(xe, self.params["short_delta"], wing_delta=self.params["wing_delta"],
                                  usable=self.usable(xe))
            except SelectionError as e:
                self.skip(ctx, str(e))
                return
            credit = -pkg.natural_open()
            if not credit > 0:
                self.skip(ctx, "no credit")
                return
            n, ml = self.size_credit(ctx, pkg)
            self.open_package(ctx, f"IC:{day.date()}", SPX, pkg, n, "credit",
                              meta={"credit": credit, "max_loss_usd": ml})

    def exit_rule(self, ctx, st, v):
        return self.credit_exit(st, v)


# ============================================================================ S162
@register
class S162(_ZeroDTEBase):
    id = "S162"
    name = NAMES["S162"]
    sizing = "OPT_CREDIT"
    params = {"entry": "10:00", "short_delta": 0.20, "wing_delta": 0.05, "tp": 0.50, "stop_mult": 2.0,
              "vwap_source": "SPY", "allow_synthetic": False}
    grid = {"entry": ["10:00", "11:00"], "short_delta": [0.10, 0.20, 0.30], "wing_delta": [0.05, 0.10]}
    deviations = ZDTE_DEVIATIONS + (
        "'SPX above VWAP': SPX has no volume; evaluated as SPY last 1m close > SPY session VWAP (vwap_source='SPY', "
        "flagged T2_PROXY); 'open > prior close' on SPX 1m bars (first RTH bar open vs prior session's last 1m close).",
        "'Close at 50% original credit' = credit - cost to close >= 50% of credit; '2x credit to close' = cost to "
        "close >= 2 x credit; 15:30 = closing order at the bar ending 15:30.",
    )
    spec = SPECS["S162"]

    def _gap_up(self, ctx) -> Optional[bool]:
        b = ctx.session_bars(SPX, "1m")
        if b is None or len(b) == 0:
            return None
        h = ctx.history(SPX, len(b) + 1, "1m", adjusted=False)
        if h is None or len(h) < len(b) + 1:
            return None
        prev = h.iloc[0]
        if session_date_of(h.index[0]) != ctx.calendar().prev_session(ctx.session_date()):
            return None
        return bool(float(b["open"].iloc[0]) > float(prev["close"]))

    def on_bar(self, ctx) -> None:
        self.refresh(ctx)
        self.monitor(ctx)
        day = ctx.session_date()
        if at_label(ctx, self.params["entry"]) and day not in self._done and self.before_exit_label(ctx):
            self._done.add(day)
            c, vw = self.vwap_state(ctx)
            gap = self._gap_up(ctx)
            ctx.log(f"{self.id} filters", close=c, vwap=vw, gap_up=gap)
            if not (np.isfinite(vw) and c > vw and gap):
                return
            xe = self.zero_dte(ctx)
            if xe is None:
                return
            try:
                pkg = vertical(xe, -abs(self.params["wing_delta"]), -abs(self.params["short_delta"]),
                               usable=self.usable(xe), name="put_credit_spread")
            except SelectionError as e:
                self.skip(ctx, str(e))
                return
            credit = -pkg.natural_open()
            if not credit > 0:
                self.skip(ctx, "no credit")
                return
            n, ml = self.size_credit(ctx, pkg)
            self.open_package(ctx, f"PCS:{day.date()}", SPX, pkg, n, "credit",
                              meta={"credit": credit, "max_loss_usd": ml})

    def exit_rule(self, ctx, st, v):
        return self.credit_exit(st, v)


# ============================================================================ S163
@register
class S163(_ZeroDTEBase):
    id = "S163"
    name = NAMES["S163"]
    sizing = "OPT_DEBIT"
    params = {"or_minutes": 30, "long_delta": 0.55, "short_delta": 0.25, "max_loss": 0.50, "window_start": "10:00",
              "window_end": "12:00", "vwap_source": "SPY", "allow_synthetic": False}
    grid = {"or_minutes": [15, 30], "long_delta": [0.50, 0.55], "short_delta": [0.20, 0.25]}
    deviations = ZDTE_DEVIATIONS + (
        "'SPX 5m close above first 30m high': 5m bars resampled causally from SPX 1m bars (complete buckets only); "
        "opening range = first or_minutes of SPX 1m bars; one entry per session; signal 5m bars ending in "
        "[10:00, 12:00] ET (window not shifted on early closes).",
        "'Underlying below 5m VWAP': SPX has no volume -> SPY 5m close < SPY session VWAP of 5m bars "
        "(vwap_source='SPY', flagged T2_PROXY), checked at each 5m close.",
        "'Option debit loss 50%' = natural close value <= 50% of the debit paid (monitored each minute).",
    )
    spec = SPECS["S163"]

    def on_bar(self, ctx) -> None:
        self.refresh(ctx)
        self.monitor(ctx)
        day = ctx.session_date()
        o, _ = ctx.calendar().open_close(day)
        five = int(round((ctx.now - o).total_seconds())) % 300 == 0
        if not five:
            return
        # VWAP exit on 5m closes
        for key, st in list(self.book.items()):
            if st.close_order is None and self.is_open(ctx, st) and st.filled_units > 0:
                c5, vw5 = self.vwap_state(ctx, 5)
                if np.isfinite(vw5) and c5 < vw5:
                    self.close_package(ctx, key, "below_vwap")
        lo, hi = et(day, self.params["window_start"]), et(day, self.params["window_end"])
        if day in self._done or not (lo <= ctx.now <= hi) or not self.before_exit_label(ctx):
            return
        b1 = ctx.session_bars(SPX, "1m")
        if b1 is None or len(b1) == 0:
            return
        orr = opening_range(b1, minutes=int(self.params["or_minutes"]))["or_high"].dropna()
        b5 = resample_bars(b1, 5, ctx.now, o)
        if len(orr) == 0 or len(b5) == 0 or b5.index[-1] + pd.Timedelta(minutes=5) != ctx.now:
            return
        if not float(b5["close"].iloc[-1]) > float(orr.iloc[-1]):
            return
        self._done.add(day)
        xe = self.zero_dte(ctx)
        if xe is None:
            return
        try:
            pkg = vertical(xe, abs(self.params["long_delta"]), abs(self.params["short_delta"]),
                           usable=self.usable(xe), name="call_debit_spread")
        except SelectionError as e:
            self.skip(ctx, str(e))
            return
        debit = pkg.natural_open()
        if not debit > 0:
            self.skip(ctx, "non-positive debit")
            return
        n = SZ.size_opt_debit(ctx.nav, debit * pkg.multiplier, existing_option_debit=self.open_option_debit(ctx))
        self.open_package(ctx, f"CDS:{day.date()}", SPX, pkg, n, "debit",
                          meta={"debit": debit, "or_high": float(orr.iloc[-1]),
                                "planned_debit_usd": n * debit * pkg.multiplier})

    def exit_rule(self, ctx, st, v):
        debit = st.entry_cash
        if debit > 0 and v <= (1.0 - self.params["max_loss"]) * debit:
            return "debit_loss"
        return None


# ============================================================================ S164
@register
class S164(_ZeroDTEBase):
    id = "S164"
    name = NAMES["S164"]
    sizing = "OPT_DEBIT"
    CHAIN_MAX_FID = Fidelity.T0_QUOTE
    params = {"entry_offset": 5, "exit_offset": 30, "allow_synthetic": False}
    grid = {"entry_offset": [5, 10], "exit_offset": [15, 30, 60]}
    requires = [Requirement("events", "FOMC_SCHEDULE", symbols=("*",),
                            note="scheduled FOMC statement times known in advance (needed to act BEFORE the release)"),
                Requirement("events", "FOMC", symbols=("*",), note="actual statement release timestamp"),
                Requirement("options", "1m", symbols=(SPX,), min_fidelity=Fidelity.T0_QUOTE,
                            note="recorded SPXW NBBO snapshots around the statement")]
    deviations = COMMON_DEVIATIONS + (
        "Entering BEFORE a statement needs its scheduled time in advance: read from 'FOMC_SCHEDULE' events "
        "(effective_at = scheduled statement time, known before it); FOMC events (known_at = actual release) "
        "are visible only after the release (PIT). Only payload type STATEMENT (no press conferences).",
        "Grid 'entry -5/-10m' -> entry_offset minutes before the scheduled time (decision on the bar ending then, "
        "fill at the first NBBO snapshot after it); 'exit +15/+30/+60m' -> exit_offset after the ACTUAL release "
        "(closing order at the first bar ending at/after it, fill at the next snapshot). If the actual release is "
        "not published within 2 h of the schedule the straddle is closed (logged).",
        "Locked quotes: the engine's ComboFillModel accepts ask == bid, so 'first nonlocked NBBO' is not enforced "
        "by the fill model (reported engine issue).",
        "Needs a PM-settled SPX expiry on the statement day (SPXW Wednesdays from 2016-03); otherwise skipped.",
    )
    spec = SPECS["S164"]

    def _scheduled_today(self, ctx) -> list:
        try:
            evs = ctx.portal().events("FOMC_SCHEDULE", None, ctx.now - pd.Timedelta(days=400),
                                      ctx.now + pd.Timedelta(days=2))
        except Exception:
            return []
        out = []
        for ev in evs:
            if str((ev.payload or {}).get("type", "STATEMENT")).upper() != "STATEMENT":
                continue
            t = to_utc(ev.effective_at) if ev.effective_at is not None else None
            if t is not None and session_date_of(t) == ctx.session_date():
                out.append(t)
        return sorted(set(out))

    def _actual(self, ctx, sched: pd.Timestamp) -> Optional[pd.Timestamp]:
        try:
            evs = ctx.portal().events("FOMC", None, sched - pd.Timedelta(hours=6), ctx.now + pd.Timedelta(minutes=1))
        except Exception:
            return None
        ts = [to_utc(e.known_at) for e in evs if str((e.payload or {}).get("type", "STATEMENT")).upper() == "STATEMENT"
              and abs(to_utc(e.known_at) - sched) <= pd.Timedelta(hours=6) and to_utc(e.known_at) <= ctx.now]
        return min(ts, key=lambda t: abs(t - sched)) if ts else None

    def on_bar(self, ctx) -> None:
        self.refresh(ctx)
        for key, st in list(self.book.items()):
            if st.close_order is not None or not self.is_open(ctx, st):
                continue
            sched = st.meta["scheduled"]
            act = self._actual(ctx, sched)
            if act is not None and ctx.now >= act + pd.Timedelta(minutes=self.params["exit_offset"]):
                self.close_package(ctx, key, "post_statement_exit")
            elif act is None and ctx.now >= sched + pd.Timedelta(hours=2):
                ctx.log(f"{self.id}: statement not published within 2h of schedule; closing", key=key)
                self.close_package(ctx, key, "no_release_exit")
        for sched in self._scheduled_today(ctx):
            key = f"FOMC:{sched.isoformat()}"
            if key in self._done or ctx.now != sched - pd.Timedelta(minutes=self.params["entry_offset"]):
                continue
            self._done.add(key)
            if self._actual(ctx, sched) is not None:
                self.skip(ctx, "statement already released")
                continue
            xe = self.zero_dte(ctx)
            if xe is None:
                continue
            try:
                pkg = straddle(xe, 1, usable=self.usable(xe))
            except SelectionError as e:
                self.skip(ctx, str(e))
                continue
            debit = pkg.natural_open()
            if not debit > 0:
                continue
            n = SZ.size_opt_debit(ctx.nav, debit * pkg.multiplier, existing_option_debit=self.open_option_debit(ctx))
            self.open_package(ctx, key, SPX, pkg, n, "debit",
                              meta={"scheduled": sched, "debit": debit, "planned_debit_usd": n * debit * pkg.multiplier})
