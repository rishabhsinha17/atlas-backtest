"""Intraday breakout / momentum strategies S001-S020 (sheet family
"Intraday breakout / momentum"). Registered through ``atlasbt.strategies.intraday``.

Every class carries the sheet's baseline ``params``, the exact "Parameters to
test" ``grid``, its data ``requires``, the verbatim ``spec`` and every
interpretation in ``deviations`` (module-wide ones in
``intraday_base.MODULE_DEVIATIONS``).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..core.types import Fidelity
from ..engine.strategy import Requirement
from ..indicators import intraday as iind
from ..indicators.patterns import flag_pattern, nr_n, pivots
from .intraday_base import (
    ETF_CORE, MEGACAP_UNIVERSE, MODULE_DEVIATIONS, NS_MIN, IntradayStrategy, aggregate_frame, atr_stream,
    ema_stream, requires_intraday, rsi_stream,
)
from .registry import register

NAN = math.nan


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


# ===================================================================== shared gap / premarket helpers
class _PremarketMixin:
    """09:29 quote midpoint and premarket statistics (S005-S007)."""

    def pm_midpoint(self, ctx, sym, tp) -> float:
        t0, t1 = self.at("09:29"), self.open_ns
        mid = NAN
        try:   # last quote at or before 09:29:59 ET (premarket from 04:00)
            q = ctx.portal().quotes(sym, pd.Timestamp(self.at("04:00"), tz="UTC"), pd.Timestamp(t1, tz="UTC"))
        except Exception:  # pragma: no cover - portal without quotes
            q = None
        if q is not None and len(q):
            r = q.iloc[-1]
            b, a = _f(r["bid"]), _f(r["ask"])
            if a >= b > 0:
                mid = 0.5 * (a + b)
        if mid != mid and self.params.get("pm_mid_proxy"):
            w = tp.pre_window(t0, t1)
            if w.stop > w.start:
                mid = float(tp.pc[w.stop - 1])
                ctx.flag_fidelity(Fidelity.T2_PROXY, "09:29 midpoint proxied by last premarket 1m close (no quote)")
        if mid != mid:
            ctx.flag_fidelity(Fidelity.NA, f"09:29 premarket midpoint missing for {sym} on {self._session.date()}: "
                                           f"no quote at/before 09:29:59 ET (no signal that day)")
        return mid

    def pm_stats(self, tp, start: str, end: str) -> dict:
        w = tp.pre_window(self.at(start), self.at(end))
        if w.stop <= w.start:
            return {}
        idx = pd.DatetimeIndex(tp.pstart[w], tz="UTC", name="start")
        df = pd.DataFrame({"open": tp.po[w], "high": tp.ph[w], "low": tp.pl[w], "close": tp.pc[w],
                           "volume": tp.pv[w], "session": "PRE"}, index=idx)
        st = iind.window_stats(df, start, end, sessions=("PRE",))
        if st.empty:
            return {}
        r = st.iloc[-1]
        return {"high": float(r["high"]), "vwap": float(r["vwap"]), "volume": float(r["volume"])}


# ===================================================================== S001
@register
class S001(IntradayStrategy):
    id = "S001"; name = "5-minute opening range breakout long"; bar_freq = "1m"; asset = "STK/ETF"
    universe = ETF_CORE; universe_name = MEGACAP_UNIVERSE
    params = {"or_minutes": 5, "rvol": 1.5, "target_r": None, "stop": "or_low"}
    grid = {"or_minutes": [5, 15, 30], "rvol": [1.0, 1.5, 2.0], "target_r": [None, 1.5, 2.0],
            "stop": ["or_low", "atr5m"]}
    requires = requires_intraday("1m")
    deviations = MODULE_DEVIATIONS + (
        "Opening range = bars starting in [09:30, 09:30+or_minutes); RVOL = cumulative volume through the last OR "
        "minute / median of the same cumulative volume over the previous 20 sessions (the '09:35 RVOL(5m)' "
        "generalised to the OR length of the grid).",
        "Entry = first 1m bar starting at/after the OR end whose close > H and > session VWAP; RVOL gates the day.",
        "stop='atr5m' => entry reference (signal close) - 1 x Wilder ATR(14) of 5m buckets.",
        "Sheet baseline exit has the 2R target 'optional': baseline target_r=None (grid none/1.5R/2R).",
    )
    spec = 'At 09:35 ET calculate H=max(highs 09:30-09:34); enter at first 1m close >H after 09:35; require 09:35 RVOL(5m)>=1.5 versus time-of-day historical median and last price>session VWAP. || Stop L=min(lows 09:30-09:34); optional take 2R; otherwise exit 15:55; at most one entry/day.'

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        or_end = self.open_ns + int(p["or_minutes"]) * NS_MIN
        if now < or_end:
            return
        for sym in self._syms:
            d = self.st(sym)
            if d.get("dead") or self.entries_today(sym):
                continue
            tp = self.tape(ctx, sym)
            if "H" not in d:
                w = tp.window(self.open_ns, or_end)
                if w.stop <= w.start or not tp.has_open_bar():
                    d["dead"] = True
                    continue
                d["H"], d["L"] = float(tp.h[w].max()), float(tp.l[w].min())
                prof = self.profile(ctx, sym, cumulative=True)
                exp = _f(prof.get(self.minute_of(int(tp.start[w.stop - 1])), NAN))
                d["rvol"] = tp.cumv[w.stop - 1] / exp if exp > 0 else NAN
                if not d["rvol"] >= p["rvol"]:
                    d["dead"] = True
                    continue
                if p["stop"] == "atr5m":
                    d["atr"] = atr_stream(14, 5)
            i = tp.n - 1
            if i < 0 or tp.start[i] < or_end or tp.end(i) != now:
                continue
            c = float(tp.c[i])
            if c > d["H"] and c > tp.vwap[i]:
                if p["stop"] == "or_low":
                    stop = d["L"]
                else:
                    a = d["atr"].sync(ctx, tp, self.base_min)
                    stop = c - (a[-1] if a else NAN)
                self.enter(ctx, sym, 1, c, stop, target_r=p["target_r"], why=f"ORB rvol={d['rvol']:.2f}")


# ===================================================================== S002
@register
class S002(IntradayStrategy):
    id = "S002"; name = "15-minute opening range breakout short"; bar_freq = "1m"; asset = "STK/ETF"
    universe = ETF_CORE; universe_name = MEGACAP_UNIVERSE
    params = {"or_minutes": 15, "turnover": 0.0005, "target_r": 2.0}
    grid = {"or_minutes": [5, 15, 30], "turnover": [0.00025, 0.0005, 0.001], "target_r": [1.0, 2.0]}
    requires = requires_intraday("1m")
    deviations = MODULE_DEVIATIONS + (
        "Opening turnover = share volume of the OR bars vs 20d ADV in shares (prior 20 sessions, split-adjusted); "
        "'0.05%' => 0.0005.",
        "'before 12:00' = signal bar closes strictly before 12:00 ET; entry = first 1m close < L that is also below "
        "session VWAP.",
        "Borrow/locate and SSR are applied by the engine.",
    )
    spec = 'At 09:45 calculate L=min 09:30-09:44; short on first 1m close <L before 12:00, price<VWAP, opening 15m turnover >=0.05% of 20d ADV. || Stop H=max 09:30-09:44; exit 15:55 or 2R; one short/day.'

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        or_end = self.open_ns + int(p["or_minutes"]) * NS_MIN
        if now < or_end or now >= self.at("12:00"):
            return
        for sym in self._syms:
            d = self.st(sym)
            if d.get("dead") or self.entries_today(sym):
                continue
            tp = self.tape(ctx, sym)
            if "L" not in d:
                w = tp.window(self.open_ns, or_end)
                if w.stop <= w.start or not tp.has_open_bar():
                    d["dead"] = True
                    continue
                d["H"], d["L"] = float(tp.h[w].max()), float(tp.l[w].min())
                adv = self.daily(ctx, sym).adv(20)
                if not float(tp.v[w].sum()) >= p["turnover"] * adv:
                    d["dead"] = True
                    continue
            i = tp.n - 1
            if i < 0 or tp.start[i] < or_end or tp.end(i) != now:
                continue
            c = float(tp.c[i])
            if c < d["L"] and c < tp.vwap[i]:
                self.enter(ctx, sym, -1, c, d["H"], target_r=p["target_r"], why="ORB short")


# ===================================================================== S003
@register
class S003(IntradayStrategy):
    id = "S003"; name = "Opening-range breakout with retest"; bar_freq = "1m"; asset = "STK/ETF"
    universe = ETF_CORE; universe_name = MEGACAP_UNIVERSE
    params = {"retest_bars": 15, "buffer": 0.0005, "or_minutes": 15, "target_r": 2.0}
    grid = {"retest_bars": [5, 15, 30], "buffer": [0.0, 0.0005, 0.001]}
    requires = requires_intraday("1m")
    deviations = MODULE_DEVIATIONS + (
        "Sequence: (1) first 1m close > OR high (no entry); (2) one of the next retest_bars bars has low <= OR high "
        "(retest); (3) a LATER bar (after the retest bar, any time before the entry cutoff) closes > OR high x "
        "(1+buffer) -> enter next bar. No retest within the window -> no trade that day.",
        "Retest swing low = lowest low from the retest bar through the confirmation bar; stop = min(that, OR "
        "midpoint) as written.",
        "One entry per day (the sequence starts from the FIRST breakout close).",
    )
    spec = 'After 15m OR, first 1m close >OR high; do not chase. Enter when low of next 15 bars <=OR high and a later 1m close >OR high+0.05% (sequence ordered). || Stop min(retest swing low,OR midpoint); exit 15:55 or 2R.'

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        or_end = self.open_ns + int(p["or_minutes"]) * NS_MIN
        if now < or_end:
            return
        for sym in self._syms:
            d = self.st(sym)
            if d.get("dead") or self.entries_today(sym):
                continue
            tp = self.tape(ctx, sym)
            if "H" not in d:
                w = tp.window(self.open_ns, or_end)
                if w.stop <= w.start or not tp.has_open_bar():
                    d["dead"] = True
                    continue
                d["H"], d["L"] = float(tp.h[w].max()), float(tp.l[w].min())
                d["phase"], d["k"] = 0, w.stop
            H = d["H"]
            for i in range(d["k"], tp.n):
                d["k"] = i + 1
                if tp.start[i] < or_end:
                    continue
                ph = d["phase"]
                if ph == 0:
                    if tp.c[i] > H:
                        d["phase"], d["b"] = 1, i
                elif ph == 1:
                    if i - d["b"] > p["retest_bars"]:
                        d["dead"] = True
                        break
                    if tp.l[i] <= H:
                        d["phase"], d["r"] = 2, i
                elif ph == 2:
                    if tp.c[i] > H * (1.0 + p["buffer"]):
                        d["dead"] = True
                        if i == tp.n - 1 and tp.end(i) == now:
                            swing = float(tp.l[d["r"]:i + 1].min())
                            stop = min(swing, 0.5 * (H + d["L"]))
                            self.enter(ctx, sym, 1, float(tp.c[i]), stop, target_r=p["target_r"], why="ORB retest")
                        break


# ===================================================================== S004
@register
class S004(IntradayStrategy):
    id = "S004"; name = "Opening-drive momentum"; bar_freq = "1m"; asset = "STK/ETF"
    universe = ETF_CORE; universe_name = MEGACAP_UNIVERSE
    params = {"move": 0.01, "n_pos": 8, "cutoff": "11:30"}
    grid = {"move": [0.005, 0.01, 0.015], "n_pos": [7, 8, 9], "cutoff": ["10:30", "11:30"]}
    requires = requires_intraday("1m")
    deviations = MODULE_DEVIATIONS + (
        "'open' = 09:30 opening print (open of the 09:30 1m bar); 'last price' = close of the 09:39 bar; all 10 "
        "bars 09:30-09:39 must exist.",
        "'positive' 1m candle = close > open of the same bar.",
        "'VWAP slope > 0' = session VWAP after the 09:39 bar > VWAP after the 09:38 bar (one-bar slope).",
        "'two closes < VWAP' = two consecutive 1m closes below session VWAP on bars from the entry bar on; exit "
        "next bar open. 'at 11:30' = market exit filling at the 11:30 bar open (bracket time exit).",
    )
    spec = 'At 09:40 last price >= open*(1+0.01), first 10m closes >=8/10 positive, VWAP slope >0; enter next 1m open. || Stop 09:30-09:39 low; exit on two closes < VWAP or at 11:30.'

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        t940 = self.at("09:40")
        if now == t940:
            for sym in self._syms:
                tp = self.tape(ctx, sym)
                w = tp.window(self.open_ns, t940)
                if w.stop - w.start != 10 or not tp.has_open_bar():
                    continue
                o0, last = tp.first_open(), float(tp.c[w.stop - 1])
                npos = int(np.sum(tp.c[w] > tp.o[w]))
                slope = tp.vwap[w.stop - 1] - tp.vwap[w.stop - 2]
                if last >= o0 * (1 + p["move"]) and npos >= p["n_pos"] and slope > 0:
                    if self.enter(ctx, sym, 1, last, float(tp.l[w].min()), time_exit_ns=self.at(p["cutoff"]),
                                  why="opening drive") is not None:
                        self.st(sym)["entry_ns"] = now
        elif now > t940:
            for sym, q in ctx.positions().items():
                d = self.st(sym)
                if not q or "entry_ns" not in d:
                    continue
                tp = self.tape(ctx, sym)
                i = tp.n - 1
                if i >= 1 and tp.end(i) == now and tp.start[i - 1] >= d["entry_ns"] \
                        and tp.c[i] < tp.vwap[i] and tp.c[i - 1] < tp.vwap[i - 1]:
                    self.exit_position(ctx, sym, "vwap_exit")


# ===================================================================== S005 / S006
class _GapAndGo(_PremarketMixin, IntradayStrategy):
    side = 1

    def build_requires(self):
        return requires_intraday("1m", pre=True, quotes=not self.params.get("pm_mid_proxy"), quote_sessions=("PRE",))

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        for sym in self._syms:
            d = self.st(sym)
            if d.get("dead") or self.entries_today(sym):
                continue
            tp = self.tape(ctx, sym)
            if "gap" not in d:
                pc = self.daily(ctx, sym).prior_close
                mid = self.pm_midpoint(ctx, sym, tp)
                pm = self.pm_stats(tp, "04:00", "09:30")
                adv = self.daily(ctx, sym).adv(20)
                d["gap"] = mid / pc - 1.0 if pc > 0 else NAN
                ok = bool(pm) and tp.has_open_bar() and pm["volume"] >= p["pm_turnover"] * adv
                ok = ok and (d["gap"] >= p["gap"] if self.side > 0 else d["gap"] <= p["gap"])
                if not ok:
                    d["dead"] = True
                    continue
                d["pm_vwap"] = pm["vwap"]
            b = tp.agg(5)
            if b.n == 0 or b.end[-1] != now:
                continue
            o0, c = tp.first_open(), float(b.c[-1])
            if self.side > 0 and c > o0 and c > d["pm_vwap"]:
                self.enter(ctx, sym, 1, c, float(b.l[0]), target_r=p["target_r"], why=f"gap {d['gap']:.3f}")
            elif self.side < 0 and c < o0 and c < d["pm_vwap"]:
                self.enter(ctx, sym, -1, c, float(b.h[0]), target_r=p["target_r"], why=f"gap {d['gap']:.3f}")


_GAP_DEV = (
    "Gap = 09:29 quote midpoint (last NBBO quote at or before 09:29:59 ET, premarket from 04:00) / prior adjusted "
    "close - 1. PRE-session quotes are a CRITICAL requirement (Requirement('quotes', sessions=('PRE',)); "
    "NOT_TESTABLE without them) unless the opt-in pm_mid_proxy=True, which "
    "uses the close of the 09:29 premarket 1m bar (last premarket trade) and flags T2_PROXY. A day whose midpoint "
    "is still missing gets no signal and an NA fidelity flag naming the symbol and date.",
    "Premarket = PRE-session bars 04:00-09:29 ET (bars starting in [04:00, 09:30)); premarket VWAP = volume-"
    "weighted typical price (indicators.intraday.window_stats); premarket turnover/volume = PRE share volume vs "
    "20d ADV in shares. PRE bars are a critical requirement; a day without PRE bars gives no signal.",
    "'5m close' = close of a completed 09:30-aligned 5-minute bucket; the first bucket satisfying both conditions "
    "triggers; one entry per day.",
)


@register
class S005(_GapAndGo):
    id = "S005"; name = "Gap-and-go upward"; bar_freq = "1m"; asset = "STK"
    universe = (); universe_name = MEGACAP_UNIVERSE
    side = 1
    params = {"gap": 0.02, "pm_turnover": 0.2, "target_r": 2.0, "pm_mid_proxy": False}
    grid = {"gap": [0.01, 0.02, 0.04], "pm_turnover": [0.1, 0.2, 0.5]}
    requires = requires_intraday("1m", pre=True, quotes=True, quote_sessions=("PRE",))
    deviations = MODULE_DEVIATIONS + _GAP_DEV + ("Stop = low of the 09:30-09:34 bucket.",)
    spec = 'Premarket 09:29 midpoint / prior adjusted close -1 >=+2%; premarket turnover >=0.2*20d ADV; enter on first 5m close >09:30 open and above premarket VWAP. || Stop 09:30-09:34 low; exit 15:55 or 2R.'


@register
class S006(_GapAndGo):
    id = "S006"; name = "Gap-and-go downward"; bar_freq = "1m"; asset = "STK"
    universe = (); universe_name = MEGACAP_UNIVERSE
    side = -1
    params = {"gap": -0.02, "pm_turnover": 0.2, "target_r": 2.0, "pm_mid_proxy": False}
    grid = {"gap": [-0.01, -0.02, -0.04], "target_r": [1.5, 2.0]}
    requires = requires_intraday("1m", pre=True, quotes=True, quote_sessions=("PRE",))
    deviations = MODULE_DEVIATIONS + _GAP_DEV + (
        "Stop = high of the opening 5m bucket (09:30-09:34). 'profit 1.5/2R' => target_r grid; baseline 2R.",
        "Borrow/locate and SSR are applied by the engine.",
    )
    spec = 'Gap at 09:29 midpoint <=-2%; premarket volume >=0.2*20d ADV; enter short on first 5m close below 09:30 open and below premarket VWAP. || Stop opening 5m high; exit 15:55 or 2R.'


# ===================================================================== S007
@register
class S007(_PremarketMixin, IntradayStrategy):
    id = "S007"; name = "Premarket-high breakout"; bar_freq = "1m"; asset = "STK"
    universe = (); universe_name = MEGACAP_UNIVERSE
    params = {"gap": 0.01, "rvol": 2.0, "target_r": 2.0}
    grid = {"gap": [0.01, 0.02], "rvol": [1.5, 2.0, 3.0]}
    requires = requires_intraday("1m", pre=True)
    deviations = MODULE_DEVIATIONS + (
        "P = highest high of PRE bars starting 07:00-09:29 ET (critical PRE-bar requirement).",
        "'gap' = 09:30 opening print / prior adjusted close - 1 (the sheet does not say midpoint here, so no quote "
        "is needed).",
        "Signal bars: 1m bars closing at/after 09:35; RVOL = RTH cumulative volume through the bar / median "
        "cumulative volume at the same minute over the previous 20 sessions.",
        "'5m swing low before breakout' = lowest low of the (up to) five 1m RTH bars before the breakout bar.",
        "No 'first'/'one entry' clause: may re-enter after being flat.",
    )
    spec = 'Premarket 07:00-09:29 high P; gap >=1%; at 09:35 or later close 1m >P, price>VWAP and RTH volume to time>time-of-day median*2. || Stop 5m swing low before breakout; exit 15:55 or at 2R.'

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        if now < self.at("09:35"):
            return
        for sym in self._syms:
            d = self.st(sym)
            if d.get("dead"):
                continue
            tp = self.tape(ctx, sym)
            if "P" not in d:
                pm = self.pm_stats(tp, "07:00", "09:30")
                pc = self.daily(ctx, sym).prior_close
                if not pm or not tp.has_open_bar() or not (tp.first_open() / pc - 1.0 >= p["gap"]):
                    d["dead"] = True
                    continue
                d["P"] = pm["high"]
                d["prof"] = self.profile(ctx, sym, cumulative=True)
            i = tp.n - 1
            if i < 1 or tp.end(i) != now or self.has_exposure(ctx, sym):
                continue
            c = float(tp.c[i])
            exp = _f(d["prof"].get(self.minute_of(int(tp.start[i])), NAN))
            if c > d["P"] and c > tp.vwap[i] and exp > 0 and tp.cumv[i] > p["rvol"] * exp:
                stop = float(tp.l[max(0, i - 5):i].min())
                self.enter(ctx, sym, 1, c, stop, target_r=p["target_r"], why="PM high breakout")


# ===================================================================== S008
@register
class S008(IntradayStrategy):
    id = "S008"; name = "Previous-day-high intraday breakout"; bar_freq = "1m"; asset = "STK/ETF"
    universe = ETF_CORE; universe_name = MEGACAP_UNIVERSE
    params = {"lookback": 1, "confirm": "5m", "target_r": 3.0}
    grid = {"lookback": [1, 2, 5], "confirm": ["1m", "5m"]}
    requires = requires_intraday("1m")
    deviations = MODULE_DEVIATIONS + (
        "Prior RTH high = max daily high of the last `lookback` sessions (adjusted). 'After 09:45' = signal "
        "bar/bucket closing strictly after 09:45. 'confirmation 1/5m' = the breakout close is a 1m close or a "
        "5m-bucket close.",
        "'prior 20d return' = prior close / close 20 sessions earlier - 1 (adjusted).",
        "'most recent 5m swing low' = latest confirmed pivot low (2 bars each side, indicators.pivots) of today's "
        "5m buckets; if none is confirmed yet, the lowest 5m low of the session so far.",
        "One entry per day ('first').",
    )
    spec = 'After 09:45 enter long first 5m close >prior RTH high and day VWAP, if prior 20d return>0. || Stop most recent 5m swing low; 3R target or close 15:55.'

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        if now <= self.at("09:45"):
            return
        for sym in self._syms:
            d = self.st(sym)
            if d.get("dead") or self.entries_today(sym):
                continue
            if "ph" not in d:
                di = self.daily(ctx, sym)
                d["ph"] = di.max_high(int(p["lookback"]))
                if not (di.ret(20) > 0 and d["ph"] == d["ph"]):
                    d["dead"] = True
                    continue
            tp = self.tape(ctx, sym)
            b = tp.agg(5)
            if p["confirm"] == "5m":
                if b.n == 0 or b.end[-1] != now:
                    continue
                c, vw = float(b.c[-1]), float(tp.vwap[b.last[-1]])
            else:
                i = tp.n - 1
                if i < 0 or tp.end(i) != now:
                    continue
                c, vw = float(tp.c[i]), float(tp.vwap[i])
            if c > d["ph"] and c > vw:
                stop = NAN
                if b.n:
                    stop = _f(pivots(b.h, b.l, 2, 2)["last_pl"][-1])
                    if stop != stop:
                        stop = float(b.l.min())
                self.enter(ctx, sym, 1, c, stop, target_r=p["target_r"], why="prior-day-high breakout")


# ===================================================================== S009
@register
class S009(IntradayStrategy):
    id = "S009"; name = "Intraday NR7 compression break"; bar_freq = "5m"; asset = "STK/ETF"
    universe = ETF_CORE; universe_name = MEGACAP_UNIVERSE
    params = {"n": 7, "expiry": 6, "target_r": 2.0}
    grid = {"n": [5, 7, 10], "expiry": [4, 6, 10]}
    requires = requires_intraday("5m")
    deviations = MODULE_DEVIATIONS + (
        "Compression bar = the bar just completed before the signal bar, NR-n among today's 5m bars (indicators."
        "patterns.nr_n: range <= each of the previous n-1 ranges, ties count); the NEXT bar only must close above "
        "its high.",
        "'VWAP positive' = signal close > session VWAP.",
        "'expiration 4/6/10 bars' = the 'exit after 6 bars' holding period: market exit filling at the open of the "
        "bar `expiry` bars after the entry bar.",
        "No 'first' clause: may re-enter after being flat.",
    )
    spec = 'Identify prior 7 5m bar high-low ranges; if completed prior bar has minimum range of prior 7, enter when next 5m close crosses its high; VWAP positive. || Stop compression bar low; exit after 6 bars or 2R.'

    def step(self, ctx):
        p = self.params
        n = int(p["n"])
        now = int(ctx.now.value)
        for sym in self._syms:
            tp = self.tape(ctx, sym)
            i = tp.n - 1
            if i < n or tp.end(i) != now or self.has_exposure(ctx, sym):
                continue
            k = i - 1
            if not bool(nr_n(tp.h[:k + 1], tp.l[:k + 1], n)[-1]):
                continue
            c = float(tp.c[i])
            if c > tp.h[k] and c > tp.vwap[i]:
                self.enter(ctx, sym, 1, c, float(tp.l[k]), target_r=p["target_r"],
                           time_exit_ns=now + int(p["expiry"]) * self.bar_ns, why=f"NR{n} break")


# ===================================================================== S010
@register
class S010(IntradayStrategy):
    id = "S010"; name = "Intraday flag continuation"; bar_freq = "1m"; asset = "STK"
    universe = (); universe_name = MEGACAP_UNIVERSE
    params = {"impulse": 0.02, "retrace": (0.2, 0.5), "flag_max": 15, "flag_min": 5, "impulse_bars": 45,
              "target_r": 2.0}
    grid = {"impulse": [0.01, 0.02, 0.03], "retrace": [(0.2, 0.5), (0.3, 0.6)], "flag_max": [10, 20]}
    requires = requires_intraday("1m")
    deviations = MODULE_DEVIATIONS + (
        "Pattern = indicators.patterns.flag_pattern (one signal per session): impulse >= x% from the 09:30 open "
        "within the first 45 1m bars, flag = bars after the impulse peak, 5..flag_max bars, retracement of the "
        "impulse within the band, flag volume SUM < impulse volume SUM, signal on close > flag high.",
        "Baseline flag length 5-15 bars (entry text); the grid's 'max flag 10/20m' varies the maximum.",
    )
    spec = 'At least +2% from 09:30 open within 45m; next 5-15 bars retrace 20-50% of impulse and have lower aggregate volume than impulse; enter close above flag high. || Stop low of flag; exit 2R or 15:55.'

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        horizon = int(p["impulse_bars"]) + int(p["flag_max"]) + 2
        for sym in self._syms:
            d = self.st(sym)
            if d.get("dead") or self.entries_today(sym):
                continue
            tp = self.tape(ctx, sym)
            n = tp.n
            if n == 0 or tp.end(n - 1) != now:
                continue
            if not tp.has_open_bar() or n > horizon:
                d["dead"] = True
                continue
            fp = flag_pattern(tp.o[:n], tp.h[:n], tp.l[:n], tp.c[:n], tp.v[:n], np.zeros(n), p["impulse"],
                              int(p["impulse_bars"]), tuple(p["retrace"]), (int(p["flag_min"]), int(p["flag_max"])))
            if bool(fp["signal"][-1]):
                self.enter(ctx, sym, 1, float(tp.c[n - 1]), float(fp["flag_low"][-1]), target_r=p["target_r"],
                           why="flag")


# ===================================================================== S011
@register
class S011(IntradayStrategy):
    id = "S011"; name = "First-hour range break"; bar_freq = "5m"; asset = "STK/ETF"
    universe = ETF_CORE; universe_name = MEGACAP_UNIVERSE
    params = {"break_bp": 0.0, "rsi": 55}
    grid = {"break_bp": [0.0, 0.0005], "rsi": [50, 55, 60]}
    requires = requires_intraday("5m")
    deviations = MODULE_DEVIATIONS + (
        "First hour = 5m bars starting 09:30-10:25 (all 12 required); signal = first 5m bar after 10:30 whose "
        "close > high x (1+break) with Wilder RSI(14) of 5m closes (spanning sessions) > threshold; one entry/day.",
        "'Break threshold 0/5bp' => break_bp 0 / 0.0005.",
    )
    spec = 'At 10:30 store first-hour high and low; enter on first complete 5m close > high and RSI(14,5m)>55. || Stop first-hour midpoint; exit 15:55.'

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        t1030 = self.at("10:30")
        if now <= t1030:
            return
        for sym in self._syms:
            d = self.st(sym)
            if d.get("dead") or self.entries_today(sym):
                continue
            tp = self.tape(ctx, sym)
            if "H" not in d:
                w = tp.window(self.open_ns, t1030)
                if w.stop - w.start != 12:
                    d["dead"] = True
                    continue
                d["H"], d["L"] = float(tp.h[w].max()), float(tp.l[w].min())
                d["rsi"] = rsi_stream(14)
            i = tp.n - 1
            if i < 0 or tp.end(i) != now:
                continue
            r = d["rsi"].sync(ctx, tp, self.base_min)
            c = float(tp.c[i])
            if c > d["H"] * (1 + p["break_bp"]) and r[i] > p["rsi"]:
                self.enter(ctx, sym, 1, c, 0.5 * (d["H"] + d["L"]), why="first-hour break")


# ===================================================================== S012
@register
class S012(IntradayStrategy):
    id = "S012"; name = "Intraday VWAP continuation long"; bar_freq = "5m"; asset = "STK/ETF"
    universe = ETF_CORE; universe_name = MEGACAP_UNIVERSE
    params = {"touch": 0.0010, "confirm": 1}
    grid = {"touch": [0.0005, 0.0010, 0.0020], "confirm": [1, 2]}
    requires = requires_intraday("5m")
    deviations = MODULE_DEVIATIONS + (
        "'After 10:00' = signal bar starting at/after 10:00. 'VWAP rising 3 bars' = VWAP_t > VWAP_t-1 > VWAP_t-2 > "
        "VWAP_t-3. Pullback bar p = t - confirm with |low_p - VWAP_p| <= touch x VWAP_p; confirmation = each of "
        "the `confirm` bars after p closes above its previous bar's high.",
        "Exit: first 5m close < VWAP after entry -> market next bar.",
        "No 'first' clause: may re-enter after being flat.",
    )
    spec = 'After 10:00, require close>session VWAP, VWAP today rising 3 bars and a pullback low touches VWAP within 10bp then current 5m close>previous 5m high. || Stop pullback low; exit on 5m close <VWAP or 15:55.'

    def step(self, ctx):
        p = self.params
        k = int(p["confirm"])
        now = int(ctx.now.value)
        t10 = self.at("10:00")
        for sym in self._syms:
            tp = self.tape(ctx, sym)
            i = tp.n - 1
            if i < 0 or tp.end(i) != now:
                continue
            d = self.st(sym)
            if ctx.position(sym) > 0:
                if tp.start[i] >= d.get("entry_ns", 0) and tp.c[i] < tp.vwap[i]:
                    self.exit_position(ctx, sym, "vwap_exit")
                continue
            if tp.start[i] < t10 or i < max(3, k + 1) or self.has_exposure(ctx, sym):
                continue
            vw = tp.vwap
            pb = i - k
            ok = tp.c[i] > vw[i] and vw[i] > vw[i - 1] > vw[i - 2] > vw[i - 3]
            ok = ok and abs(tp.l[pb] - vw[pb]) <= p["touch"] * vw[pb]
            ok = ok and all(tp.c[j] > tp.h[j - 1] for j in range(pb + 1, i + 1))
            if ok and self.enter(ctx, sym, 1, float(tp.c[i]), float(tp.l[pb]), why="VWAP continuation") is not None:
                d["entry_ns"] = now


# ===================================================================== S013
@register
class S013(IntradayStrategy):
    id = "S013"; name = "Intraday VWAP reclaim long"; bar_freq = "1m"; asset = "STK/ETF"
    universe = ETF_CORE; universe_name = MEGACAP_UNIVERSE
    params = {"below_min": 15, "rel_vol": 1.5}
    grid = {"below_min": [10, 15, 30], "rel_vol": [1.0, 1.5, 2.0]}
    requires = requires_intraday("1m")
    deviations = MODULE_DEVIATIONS + (
        "Below-VWAP streak = consecutive 1m closes < session VWAP counted on bars starting at/after 10:00; once the "
        "streak reaches below_min the setup is armed and the FIRST subsequent 5m-bucket close above VWAP is the "
        "reclaim: it enters only if its volume > rel_vol x mean volume of the previous 20 completed 5m buckets "
        "(prior session included); otherwise the day is over (one entry/day).",
        "Exit: first 5m-bucket close below VWAP after entry -> market next bar.",
    )
    spec = 'After 10:00 require at least 15 consecutive minutes below VWAP; first 5m close above VWAP with volume >prior 20 completed 5m bars mean*1.5. || Stop low of reclaim 5m bar; exit on 5m close below VWAP or 15:55.'

    def _prior_bucket_volumes(self, ctx, sym, tp) -> np.ndarray:
        d = self.st(sym)
        if "pv5" not in d:
            h = ctx.history(sym, 2 * 390, adjusted=True, sessions=("RTH",))
            if h is not None and len(h):
                h = h[pd.DatetimeIndex(h.index).as_unit("ns").asi8 < self.open_ns]
                if "session" in h.columns:
                    h = h[h["session"] == "RTH"]
                d["pv5"] = aggregate_frame(h, 5)["volume"].to_numpy(float)[-20:] if len(h) else np.zeros(0)
            else:
                d["pv5"] = np.zeros(0)
        return d["pv5"]

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        t10 = self.at("10:00")
        if now <= t10:
            return
        for sym in self._syms:
            d = self.st(sym)
            pos = ctx.position(sym)
            if d.get("dead") and not pos:
                continue
            tp = self.tape(ctx, sym)
            if pos > 0:
                b = tp.agg(5)
                if b.n and b.end[-1] == now and b.start[-1] >= d.get("entry_ns", 0) \
                        and b.c[-1] < tp.vwap[b.last[-1]]:
                    self.exit_position(ctx, sym, "vwap_exit")
                continue
            if self.entries_today(sym):
                continue
            for i in range(d.get("k", 0), tp.n):
                d["k"] = i + 1
                if tp.start[i] < t10:
                    continue
                d["streak"] = d.get("streak", 0) + 1 if tp.c[i] < tp.vwap[i] else 0
                if d["streak"] >= p["below_min"]:
                    d["armed"] = True
            if not d.get("armed"):
                continue
            b = tp.agg(5)
            if b.n == 0 or b.end[-1] != now or not (b.c[-1] > tp.vwap[b.last[-1]]):
                continue
            prior = np.concatenate([self._prior_bucket_volumes(ctx, sym, tp), b.v[:-1]])[-20:]
            d["dead"] = True                     # the first reclaim is the only candidate
            if len(prior) == 20 and b.v[-1] > p["rel_vol"] * float(prior.mean()):
                if self.enter(ctx, sym, 1, float(b.c[-1]), float(b.l[-1]), why="VWAP reclaim") is not None:
                    d["entry_ns"] = now


# ===================================================================== S014
@register
class S014(IntradayStrategy):
    id = "S014"; name = "VWAP-band momentum acceleration"; bar_freq = "5m"; asset = "STK/ETF"
    universe = ETF_CORE; universe_name = MEGACAP_UNIVERSE
    params = {"z": 1.5, "hold": 6, "target_r": 2.0}
    grid = {"z": [1.0, 1.5, 2.0], "hold": [3, 6, 12]}
    requires = requires_intraday("5m")
    deviations = MODULE_DEVIATIONS + (
        "z = (close - VWAP) / volume-weighted intraday sd (indicators convention). 'last 3 5m bars rise' = three "
        "consecutive higher closes (c_t > c_t-1 > c_t-2 > c_t-3). Signal bar starts at/after 10:00.",
        "Stop = the nearer (higher) of session VWAP and close - 1 x ATR(14, 5m) at the signal bar.",
        "Time exit: market exit at the open of the bar `hold` bars after the entry bar.",
        "No 'first' clause: may re-enter after being flat.",
    )
    spec = 'After 10:00, z=(close-VWAP)/VWAP-weighted intraday stdev >=+1.5 and last 3 5m bars rise; buy next open. || Stop below session VWAP or 1ATR(5m), nearer; exit after 6 bars or at 2R.'

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        t10 = self.at("10:00")
        for sym in self._syms:
            tp = self.tape(ctx, sym)
            d = self.st(sym)
            if "atr" not in d:
                d["atr"] = atr_stream(14)
            i = tp.n - 1
            if i < 3 or tp.end(i) != now or tp.start[i] < t10 or self.has_exposure(ctx, sym):
                continue
            c, vw, sd = float(tp.c[i]), float(tp.vwap[i]), float(tp.sd[i])
            if not (sd > 0 and (c - vw) / sd >= p["z"] and tp.c[i] > tp.c[i - 1] > tp.c[i - 2] > tp.c[i - 3]):
                continue
            a = d["atr"].sync(ctx, tp, self.base_min)
            stop = max(vw, c - a[i])
            self.enter(ctx, sym, 1, c, stop, target_r=p["target_r"], time_exit_ns=now + int(p["hold"]) * self.bar_ns,
                       why="VWAP band acceleration")


# ===================================================================== S015
@register
class S015(IntradayStrategy):
    id = "S015"; name = "Closing half-hour momentum"; bar_freq = "1m"; asset = "ETF"; sizing = "DAY_NOTIONAL"
    universe = ETF_CORE; universe_name = None
    params = {"threshold": 0.005, "gate": True, "mid_proxy": False}
    grid = {"threshold": [0.0, 0.0025, 0.005], "gate": [True, False]}
    requires = requires_intraday("1m", quotes=True)
    deviations = MODULE_DEVIATIONS + (
        "Return = 15:25 quote midpoint (last quote at/before 15:25:00 within the prior minute) / previous adjusted "
        "close - 1. Quotes are CRITICAL (NOT_TESTABLE without) unless the opt-in mid_proxy=True uses the close of "
        "the 15:24 1m bar and flags T2_PROXY.",
        "First half-hour return = 10:00 price (close of the 09:59 bar) / previous close - 1 (Gao-Han-Li-Zhou "
        "definition); gate requires it to have the same sign as the trade (positive for longs, negative for the "
        "short reversal).",
        "'buy at 15:30 next bar' = market order submitted at 15:30 filling at the 15:30 1m bar open; DAY_NOTIONAL "
        "10% prior-day NAV; exit 15:55 (engine). Early-close sessions shift 15:25/15:30 to close-35/-30 min.",
        "Threshold 0: long if return >= 0 is checked first, short if return <= -0 otherwise.",
    )
    spec = 'At 15:25 compare current midpoint to previous day close: return >=+0.5%; if realized first half-hour return positive, buy at 15:30 next bar. || Exit at 15:55 ET; reverse and short on return <=-0.5%, subject to borrow.'

    def build_requires(self):
        return requires_intraday("1m", quotes=not self.params.get("mid_proxy"))

    def session_start(self, ctx):
        self.t_dec = min(self.at("15:25"), self.close_ns - 35 * NS_MIN)
        self.t_buy = min(self.at("15:30"), self.close_ns - 30 * NS_MIN)
        self.pending: dict[str, int] = {}

    def _mid(self, ctx, sym, tp) -> float:
        q = ctx.portal().quotes(sym, pd.Timestamp(self.t_dec - NS_MIN, tz="UTC"),
                                pd.Timestamp(self.t_dec + 1, tz="UTC"))
        if q is not None and len(q):
            r = q.iloc[-1]
            b, a = _f(r["bid"]), _f(r["ask"])
            if a >= b > 0:
                return 0.5 * (a + b)
        if self.params.get("mid_proxy"):
            i = tp.idx_end_le(self.t_dec)
            if i >= 0 and tp.end(i) == self.t_dec:
                ctx.flag_fidelity(Fidelity.T2_PROXY, "15:25 midpoint proxied by 15:24 1m close (no quote)")
                return float(tp.c[i])
        return NAN

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        if now == self.t_dec:
            for sym in self._syms:
                tp = self.tape(ctx, sym)
                pc = self.daily(ctx, sym).prior_close
                mid = self._mid(ctx, sym, tp)
                r = mid / pc - 1.0 if pc > 0 else NAN
                j = tp.idx_end_le(self.at("10:00"))
                fh = tp.c[j] / pc - 1.0 if (j >= 0 and tp.end(j) == self.at("10:00") and pc > 0) else NAN
                if r >= p["threshold"] and (not p["gate"] or fh > 0):
                    self.pending[sym] = 1
                elif r <= -p["threshold"] and (not p["gate"] or fh < 0):
                    self.pending[sym] = -1
        elif now == self.t_buy:
            for sym, side in self.pending.items():
                self.enter_notional(ctx, sym, side, ctx.mark(sym), why="closing half-hour momentum")
            self.pending = {}


# ===================================================================== S016
@register
class S016(IntradayStrategy):
    id = "S016"; name = "Afternoon high-of-day breakout"; bar_freq = "5m"; asset = "STK/ETF"
    universe = ETF_CORE; universe_name = MEGACAP_UNIVERSE
    params = {"buffer": 0.0, "rel_vol": 1.3}
    grid = {"buffer": [0.0, 0.0005, 0.001], "rel_vol": [1.0, 1.3, 2.0]}
    requires = requires_intraday("5m")
    deviations = MODULE_DEVIATIONS + (
        "Signal bars close in [13:30, 15:00] ET; high of day through the prior bar = indicators.intraday."
        "session_extremes hod_prior equivalent (max high of today's earlier 5m bars).",
        "Cumulative volume through the signal bar vs median cumulative volume at the same 5m slot over the "
        "previous 20 sessions.",
        "'preceding 3 completed 5m bars low' = lowest low of the 3 most recent completed bars at the decision "
        "(signal bar and the two before it).",
        "No 'first' clause: may re-enter after being flat.",
    )
    spec = 'Between 13:30-15:00 close >max high 09:30 through prior bar, while cumulative volume >=1.3*same-time historical median. || Stop preceding 3 completed 5m bars low; exit 15:55.'

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        if now < self.at("13:30") or now > self.at("15:00"):
            return
        for sym in self._syms:
            tp = self.tape(ctx, sym)
            i = tp.n - 1
            if i < 3 or tp.end(i) != now or self.has_exposure(ctx, sym) or not tp.has_open_bar():
                continue
            c = float(tp.c[i])
            if not c > float(tp.h[:i].max()) * (1 + p["buffer"]):
                continue
            exp = _f(self.profile(ctx, sym, cumulative=True).get(self.minute_of(int(tp.start[i])), NAN))
            if exp > 0 and tp.cumv[i] >= p["rel_vol"] * exp:
                self.enter(ctx, sym, 1, c, float(tp.l[i - 2:i + 1].min()), why="afternoon HOD breakout")


# ===================================================================== S017
@register
class S017(IntradayStrategy):
    id = "S017"; name = "First-pullback after high of day"; bar_freq = "1m"; asset = "STK/ETF"
    universe = ETF_CORE; universe_name = MEGACAP_UNIVERSE
    params = {"pb_bars": 3, "pb_max": 10, "retrace": 0.5, "target_r": 1.5}
    grid = {"pb_bars": [3, 5, 10], "retrace": [0.25, 0.5]}
    requires = requires_intraday("1m")
    deviations = MODULE_DEVIATIONS + (
        "Break = first 1m bar starting at/after 10:00 whose high exceeds the high of day through the prior bar. "
        "Impulse = peak - session low at the break; peak = running high (a new high restarts the pullback).",
        "Pullback = bars after the peak bar; valid when it has pb_bars..10 bars and (peak - pullback low) / impulse "
        "< retrace; entry signal = first close above the pullback high after at least pb_bars pullback bars (checked "
        "before the new-high test, so a confirmation bar whose high exceeds the peak still enters). A new high on a "
        "bar that does not confirm (pullback shorter than pb_bars, or no close above the pullback high) extends the "
        "impulse and restarts the pullback. 'Pullback bars 3/5/10' = the minimum pullback length (maximum stays 10).",
        "A pullback longer than 10 bars or deeper than the retrace limit ends the day (first pullback only).",
    )
    spec = 'First break of daily high after 10:00; wait first 3-10 1m bars with <0.5*impulse retracement; buy next 1m close >pullback high. || Stop pullback low; profit 1.5R or 15:55.'

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        t10 = self.at("10:00")
        if now <= t10:
            return
        for sym in self._syms:
            d = self.st(sym)
            if d.get("dead") or self.entries_today(sym):
                continue
            tp = self.tape(ctx, sym)
            for i in range(d.get("k", 0), tp.n):
                d["k"] = i + 1
                if tp.start[i] < t10 or i == 0:
                    continue
                if "peak" not in d:
                    if tp.h[i] > tp.h[:i].max():
                        d["peak"], d["pt"], d["base"] = float(tp.h[i]), i, float(tp.l[:i + 1].min())
                    continue
                pbs = slice(d["pt"] + 1, i)                  # pullback bars before bar i
                npb = i - d["pt"] - 1
                imp = d["peak"] - d["base"]
                # entry test FIRST: a confirmation close above the pullback high counts even if the same bar's
                # high prints above the prior peak
                if npb >= p["pb_bars"] and tp.c[i] > tp.h[pbs].max():
                    low = float(tp.l[pbs].min())
                    d["dead"] = True
                    if imp > 0 and (d["peak"] - low) / imp < p["retrace"] and i == tp.n - 1 and tp.end(i) == now:
                        self.enter(ctx, sym, 1, float(tp.c[i]), low, target_r=p["target_r"], why="first pullback")
                    break
                if tp.h[i] > d["peak"]:                      # new high without a valid confirmation: impulse
                    d["peak"], d["pt"] = float(tp.h[i]), i   # extends, the pullback restarts
                    continue
                lowi = float(tp.l[d["pt"] + 1:i + 1].min())
                if npb + 1 > p["pb_max"] or imp <= 0 or (d["peak"] - lowi) / imp >= p["retrace"]:
                    d["dead"] = True
                    break


# ===================================================================== S018
@register
class S018(IntradayStrategy):
    id = "S018"; name = "Trend day pullback 20EMA"; bar_freq = "5m"; asset = "STK/ETF"
    universe = ETF_CORE; universe_name = MEGACAP_UNIVERSE
    params = {"day_chg": 0.01, "ema": 20}
    grid = {"day_chg": [0.005, 0.01, 0.015], "ema": [10, 20, 30]}
    requires = requires_intraday("5m")
    deviations = MODULE_DEVIATIONS + (
        "Trend day = some 5m bar closing by 11:00 with close > open x (1+day_chg) (strictly) and close > VWAP; the "
        "entry may "
        "come any time after that bar.",
        "'daily VWAP breach since 10:00' = a 5m close below session VWAP on a bar starting at/after 10:00.",
        "EMA = SMA-seeded EMA of 5m closes spanning sessions. Signal: close_t > EMA_t and low_t-1 <= EMA_t-1; "
        "first such bar only (one entry/day).",
        "Stop = lowest low of the signal bar and the bar before it ('previous two bars'). Exit: two consecutive "
        "5m closes < EMA after entry -> market next bar.",
    )
    spec = 'By 11:00 price>open by 1%, >VWAP; buy first 5m close above EMA20 after prior bar low <=EMA20 and no daily VWAP breach since 10:00. || Stop previous two bars low; exit on 2 consecutive closes <EMA20 or 15:55.'

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        t10, t11 = self.at("10:00"), self.at("11:00")
        for sym in self._syms:
            d = self.st(sym)
            pos = ctx.position(sym)
            if d.get("dead") and not pos:
                continue
            tp = self.tape(ctx, sym)
            if "ema" not in d:
                d["ema"] = ema_stream(int(p["ema"]))
            e = d["ema"].sync(ctx, tp, self.base_min)
            i = tp.n - 1
            if i < 1 or tp.end(i) != now:
                continue
            if pos > 0:
                if tp.start[i - 1] >= d.get("entry_ns", 0) and tp.c[i] < e[i] and tp.c[i - 1] < e[i - 1]:
                    self.exit_position(ctx, sym, "ema_exit")
                continue
            if self.entries_today(sym):
                continue
            for j in range(d.get("k", 0), tp.n):
                d["k"] = j + 1
                if tp.end(j) <= t11 and tp.c[j] > tp.first_open() * (1 + p["day_chg"]) and tp.c[j] > tp.vwap[j]:
                    d["trend"] = True
                if tp.start[j] >= t10 and tp.c[j] < tp.vwap[j]:
                    d["dead"] = True                       # VWAP breach since 10:00
            if d.get("dead") or not d.get("trend") or not tp.has_open_bar():
                if now >= t11 and not d.get("trend"):
                    d["dead"] = True
                continue
            if tp.c[i] > e[i] and tp.l[i - 1] <= e[i - 1]:
                d["dead"] = True
                if self.enter(ctx, sym, 1, float(tp.c[i]), float(min(tp.l[i], tp.l[i - 1])),
                              why="trend-day EMA pullback") is not None:
                    d["entry_ns"] = now


# ===================================================================== S019
@register
class S019(IntradayStrategy):
    id = "S019"; name = "Intraday breakout with relative strength"; bar_freq = "5m"; asset = "STK"
    universe = (); universe_name = MEGACAP_UNIVERSE
    benchmark = "SPY"
    params = {"top": 0.10, "excess": 0.01}
    grid = {"top": [0.05, 0.10, 0.20], "excess": [0.005, 0.01, 0.02]}
    requires = requires_intraday("5m") + [Requirement("bars", "5m", symbols=("SPY",), note="benchmark RTH return")]
    deviations = MODULE_DEVIATIONS + (
        "RTH return at 11:00 = close of the 10:55-11:00 bar / 09:30 opening print - 1, for every MEGACAP_STK "
        "member with both bars; SPY (benchmark, not traded) the same.",
        "Top x% = the ceil(x * N) highest excess returns among the N ranked names (ties by symbol); candidates "
        "also need excess > threshold.",
        "'11:00 high' = session high through the 10:55-11:00 bar; signal = first 5m close above it on a bar "
        "closing by 14:00; one entry per name per day.",
        "'30m swing low' = lowest low of the last six 5m bars at the signal.",
    )
    spec = 'Rank RTH return minus SPY RTH return at 11:00; top 10% candidates with >1% excess; buy first close above 11:00 high until 14:00. || Stop 30m swing low; exit 15:55.'

    def session_start(self, ctx):
        self.cands: dict[str, float] = {}
        self.ranked = False

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        t11, t14 = self.at("11:00"), self.at("14:00")
        if now == t11 and not self.ranked:
            self.ranked = True

            def rth_ret(sym):
                tp = self.tape(ctx, sym)
                j = tp.idx_end_le(t11)
                if not tp.has_open_bar() or j < 0 or tp.end(j) != t11:
                    return NAN, NAN
                return float(tp.c[j]) / tp.first_open() - 1.0, float(tp.h[:j + 1].max())
            rb, _ = rth_ret(self.benchmark)
            if rb != rb:
                return
            ex = {}
            for sym in self._syms:
                if sym == self.benchmark:
                    continue
                r, hi = rth_ret(sym)
                if r == r:
                    ex[sym] = (r - rb, hi)
            if not ex:
                return
            order = sorted(ex, key=lambda s: (-ex[s][0], s))
            k = max(1, int(math.ceil(p["top"] * len(order) - 1e-9)))
            for s in order[:k]:
                if ex[s][0] > p["excess"]:
                    self.cands[s] = ex[s][1]
            return
        if now <= t11 or now > t14:
            return
        for sym, hi in list(self.cands.items()):
            if self.entries_today(sym):
                continue
            tp = self.tape(ctx, sym)
            i = tp.n - 1
            if i < 0 or tp.end(i) != now:
                continue
            if tp.c[i] > hi:
                self.enter(ctx, sym, 1, float(tp.c[i]), float(tp.l[max(0, i - 5):i + 1].min()), why="RS breakout")


# ===================================================================== S020
@register
class S020(IntradayStrategy):
    id = "S020"; name = "Intraday volume spike continuation"; bar_freq = "1m"; asset = "STK"
    universe = (); universe_name = MEGACAP_UNIVERSE
    params = {"spike": 5.0, "hi_n": 20, "hold_min": 15, "target_r": 2.0}
    grid = {"spike": [3.0, 5.0, 10.0], "hi_n": [10, 20]}
    requires = requires_intraday("1m")
    deviations = MODULE_DEVIATIONS + (
        "Volume spike = 1m volume / median volume of the same ET minute over the previous 20 sessions (non-"
        "cumulative). 'prior 20m high' = max high of the previous hi_n 1m bars of today (signal bar excluded). "
        "Signal bars start at/after 10:00.",
        "Stop = lower of the trigger bar low and close - 1 x Wilder ATR(14) of 1m bars (spanning sessions).",
        "Time exit 15 minutes after the entry bar opens (market at the open of the bar starting entry+15m).",
        "No 'first' clause: may re-enter after being flat.",
    )
    spec = 'After 10:00 1m volume >=5*historical same minute median and 1m close > prior 20m high; enter next bar. || Stop triggering bar low or 1ATR(1m), lower; exit after 15m or 2R.'

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        t10 = self.at("10:00")
        if now <= t10:
            return
        n_hi = int(p["hi_n"])
        for sym in self._syms:
            tp = self.tape(ctx, sym)
            i = tp.n - 1
            if i < n_hi or tp.end(i) != now or tp.start[i] < t10 or self.has_exposure(ctx, sym):
                continue
            c = float(tp.c[i])
            if not c > float(tp.h[i - n_hi:i].max()):
                continue
            exp = _f(self.profile(ctx, sym, cumulative=False).get(self.minute_of(int(tp.start[i])), NAN))
            if not (exp > 0 and tp.v[i] >= p["spike"] * exp):
                continue
            d = self.st(sym)
            if "atr" not in d:
                d["atr"] = atr_stream(14)
            a = d["atr"].sync(ctx, tp, self.base_min)
            stop = min(float(tp.l[i]), c - a[i])
            self.enter(ctx, sym, 1, c, stop, target_r=p["target_r"], time_exit_ns=now + int(p["hold_min"]) * NS_MIN,
                       why="volume spike")
