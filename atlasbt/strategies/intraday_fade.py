"""Intraday fades / reversals S021-S030 and the broad-market ETF opening gap
fade S116. Registered through ``atlasbt.strategies.intraday``."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..core.types import Fidelity
from ..indicators.patterns import double_bottom
from ..indicators.volatility import overnight_variance_share, realized_vol, vol_scaled_to_time
from .intraday_base import (
    ETF_CORE, MEGACAP_UNIVERSE, MODULE_DEVIATIONS, NS_MIN, IntradayStrategy, atr_stream, requires_intraday,
    rsi_stream,
)
from .registry import register

NAN = math.nan


def _sgn(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


# ===================================================================== S021
@register
class S021(IntradayStrategy):
    id = "S021"; name = "Gap fade to prior close"; bar_freq = "5m"; asset = "STK/ETF"
    universe = ETF_CORE; universe_name = MEGACAP_UNIVERSE
    params = {"gap": 0.015, "bar": 5}
    grid = {"gap": [0.01, 0.015, 0.03], "bar": [5, 15]}
    requires = requires_intraday("5m")
    deviations = MODULE_DEVIATIONS + (
        "Gap = 09:30 opening print / prior adjusted close - 1. Signal candle = the first `bar` minutes (09:30-09:34 "
        "for 5, 09:30-09:44 for 15); evaluated once at its close; RSI = Wilder RSI(14) of 5m closes spanning "
        "sessions.",
        "Mirror long: gap <= -x, candle close > 09:30 open and RSI(14,5m) > 50.",
        "Stop = extreme of the signal candle (= the 09:30-09:34 extreme at the baseline bar=5; for bar=15 the "
        "09:30-09:44 extreme, otherwise the stop could sit on the wrong side of the entry).",
        "Target = previous-day close (resting limit). 'avoid earnings test as separate stratum' is a reporting "
        "split (trades on earnings days can be separated using the EARNINGS events), not an entry filter.",
    )
    spec = 'At 09:35 gap >=+1.5%; 5m candle closes <09:30 open and RSI(14,5m)<50; short next open. Mirror long on -1.5% gap. || Target previous-day close; stop 09:30-09:34 extreme; exit 15:55 if not reached.'

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        t_sig = self.open_ns + int(p["bar"]) * NS_MIN
        if now != t_sig:
            return
        nb = int(p["bar"]) // self.base_min
        for sym in self._syms:
            tp = self.tape(ctx, sym)
            w = tp.window(self.open_ns, t_sig)
            if w.stop - w.start != nb or not tp.has_open_bar():
                continue
            pc = self.daily(ctx, sym).prior_close
            o0 = tp.first_open()
            gap = o0 / pc - 1.0 if pc > 0 else NAN
            c = float(tp.c[w.stop - 1])
            r = self.st(sym).setdefault("rsi", rsi_stream(14)).sync(ctx, tp, self.base_min)[w.stop - 1]
            if gap >= p["gap"] and c < o0 and r < 50:
                self.enter(ctx, sym, -1, c, float(tp.h[w].max()), target=pc, why=f"gap fade {gap:.3f}")
            elif gap <= -p["gap"] and c > o0 and r > 50:
                self.enter(ctx, sym, 1, c, float(tp.l[w].min()), target=pc, why=f"gap fade {gap:.3f}")


# ===================================================================== S022
@register
class S022(IntradayStrategy):
    id = "S022"; name = "Opening range false breakout reversal"; bar_freq = "1m"; asset = "STK/ETF"
    universe = ETF_CORE; universe_name = MEGACAP_UNIVERSE
    params = {"fail_bars": 10, "vwap_filter": True, "or_minutes": 15}
    grid = {"fail_bars": [5, 10, 20], "vwap_filter": [True, False]}
    requires = requires_intraday("1m")
    deviations = MODULE_DEVIATIONS + (
        "Up side: first 1m close > OR high = breakout bar b; the FIRST close back < OR high must come within bars "
        "b+1..b+fail_bars (failure bar); short next bar. Down side mirrors (first close < OR low, then first close "
        "> OR low -> long). Each side has one chance; at most one trade per day (the first side that triggers).",
        "'if VWAP below' read as the failure-bar close being on the trade side of session VWAP (close < VWAP for "
        "the short, > VWAP for the long mirror); vwap_filter=False drops it.",
        "Stop = extreme of the false break (highest high from the breakout bar through the failure bar; lowest "
        "low for the mirror); target = opposite side of the opening range.",
    )
    spec = 'After 15m range first 1m close >OR high but within next 10 bars first close <OR high; short next bar if VWAP below; mirror false break down. || Stop false break high; target OR low; exit 15:55.'

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        or_end = self.open_ns + int(p["or_minutes"]) * NS_MIN
        if now < or_end:
            return
        N = int(p["fail_bars"])
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
                d["H"], d["L"], d["k"] = float(tp.h[w].max()), float(tp.l[w].min()), w.stop
                d["up"], d["dn"] = [0, -1], [0, -1]           # [phase, breakout idx]; phase 2 = side done
            H, L = d["H"], d["L"]
            for i in range(d["k"], tp.n):
                d["k"] = i + 1
                if tp.start[i] < or_end:
                    continue
                c = float(tp.c[i])
                for side, st, broke, back in ((-1, d["up"], c > H, c < H), (1, d["dn"], c < L, c > L)):
                    if st[0] == 0 and broke:
                        st[0], st[1] = 1, i
                    elif st[0] == 1:
                        if i - st[1] > N:
                            st[0] = 2
                        elif back:
                            st[0] = 2
                            vw_ok = (not p["vwap_filter"]) or (c < tp.vwap[i] if side < 0 else c > tp.vwap[i])
                            if vw_ok and i == tp.n - 1 and tp.end(i) == now and not self.entries_today(sym):
                                if side < 0:
                                    stop, tgt = float(tp.h[st[1]:i + 1].max()), L
                                else:
                                    stop, tgt = float(tp.l[st[1]:i + 1].min()), H
                                self.enter(ctx, sym, side, c, stop, target=tgt, why="false OR break")
                if d["up"][0] == 2 and d["dn"][0] == 2:
                    d["dead"] = True


# ===================================================================== S023 / S024
class _VwapSigma(IntradayStrategy):
    side = 1
    daily_n = 210

    def degenerate(self) -> bool:
        """stop_z at/beyond the entry z threshold: every signal already sits at/beyond its stop."""
        p = self.params
        return p["stop_z"] >= p["z"] if self.side > 0 else p["stop_z"] <= p["z"]

    def init(self, ctx):
        self.skipped_beyond_stop = 0
        if self.degenerate():
            msg = (f"{self.id} degenerate grid cell z={self.params['z']}/stop_z={self.params['stop_z']}: stop_z is "
                   f"at/beyond the entry threshold, every signal has z at/beyond stop_z (no positive risk distance) "
                   f"-> never traded")
            ctx.flag_fidelity(Fidelity.NA, msg)
            ctx.log(msg)

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        t10 = self.at("10:00")
        for sym in self._syms:
            tp = self.tape(ctx, sym)
            d = self.st(sym)
            if "rsi" not in d:
                d["rsi"] = rsi_stream(2)
            i = tp.n - 1
            if i < 0 or tp.end(i) != now:
                continue
            c, vw, sd = float(tp.c[i]), float(tp.vwap[i]), float(tp.sd[i])
            z = (c - vw) / sd if sd > 0 else NAN
            pos = ctx.position(sym)
            if pos and (pos > 0) == (self.side > 0):
                self.manage(ctx, sym, tp, i, z, now)
                continue
            if tp.start[i] < t10 or self.has_exposure(ctx, sym):
                continue
            r = d["rsi"].sync(ctx, tp, self.base_min)[i]
            if self.signal(ctx, sym, z, r, c):
                if (z <= p["stop_z"]) if self.side > 0 else (z >= p["stop_z"]):
                    self.skipped_beyond_stop += 1
                    ctx.log("signal skipped: z already at/beyond stop_z (no positive risk distance)", symbol=sym,
                            z=z, stop_z=p["stop_z"])
                    continue
                stop_px = vw + p["stop_z"] * sd
                if self.enter(ctx, sym, self.side, c, stop_px, resting_stop=False,
                              why=f"VWAP sigma z={z:.2f}") is not None:
                    d["exit_at"] = now + int(p["hold"]) * self.bar_ns


@register
class S023(_VwapSigma):
    id = "S023"; name = "VWAP sigma reversion long"; bar_freq = "5m"; asset = "ETF"
    universe = ETF_CORE; universe_name = None
    side = 1
    params = {"z": -2.0, "stop_z": -3.0, "rsi": 10, "hold": 12}
    grid = {"z": [-1.5, -2.0, -2.5], "stop_z": [-2.5, -3.0, -3.5]}
    requires = requires_intraday("5m")
    deviations = MODULE_DEVIATIONS + (
        "z = (5m close - session VWAP) / volume-weighted intraday sd; signal bars start at/after 10:00; 'price > "
        "SMA200 daily' = signal close > SMA200 of the prior 200 daily closes (adjusted); RSI(2) of 5m closes "
        "(Wilder, spanning sessions) < 10.",
        "Exits evaluated at each 5m close and executed at the next bar open: close >= VWAP (exit), z <= stop_z "
        "('next-bar stops conservative'), or after 12 bars (time exit at the open of the 12th bar after entry). "
        "No resting stop order; the planned stop for sizing/R = VWAP + stop_z x sd at the signal bar.",
        "Signals whose z is already at/beyond stop_z are not traded (no positive risk distance for DAY_RISK sizing); "
        "each skip is logged. Grid cells with stop_z at/beyond the entry z (z=-2.5/stop_z=-2.5) can never trade and "
        "are surfaced with an NA fidelity flag ('degenerate grid cell') instead of reading as a plain 0-trade run.",
        "No 'first' clause: may re-enter after being flat.",
    )
    spec = 'After 10:00 z=(price-VWAP)/weighted intraday std <=-2; require price> SMA200 daily and 5m RSI(2)<10. || Exit on first 5m close >=VWAP or after 12 bars; stop at z=-3 (next-bar stops conservative).'

    def signal(self, ctx, sym, z, r, c):
        p = self.params
        return z <= p["z"] and r < p["rsi"] and c > self.daily(ctx, sym).sma(200)

    def manage(self, ctx, sym, tp, i, z, now):
        p = self.params
        d = self.st(sym)
        if tp.c[i] >= tp.vwap[i]:
            self.exit_position(ctx, sym, "vwap_exit")
        elif z <= p["stop_z"]:
            self.exit_position(ctx, sym, "stop_z")
        elif now >= d.get("exit_at", math.inf):
            self.exit_position(ctx, sym, "time_exit")


@register
class S024(_VwapSigma):
    id = "S024"; name = "VWAP sigma reversion short"; bar_freq = "5m"; asset = "ETF"
    universe = ETF_CORE; universe_name = None
    side = -1
    params = {"z": 2.0, "gate": True, "stop_z": 3.0, "rsi": 90, "hold": 12}
    grid = {"z": [1.5, 2.0, 2.5], "gate": [True, False]}
    requires = requires_intraday("5m")
    deviations = MODULE_DEVIATIONS + (
        "gate=True: prior daily close < SMA200 (adjusted daily closes); gate=False: 'no-trend filter' (no daily "
        "filter). RSI(2) of 5m closes > 90; signal bars start at/after 10:00.",
        "'Exit at VWAP' = a buy limit at the current session VWAP, replaced at every 5m close (fills only when "
        "traded through); 12-bar time exit; stop z >= +3 evaluated at the 5m close, executed next bar open "
        "(mirror of S023's next-bar stop). Planned stop for sizing/R = VWAP + 3 sd at the signal bar.",
        "Signals whose z is already at/beyond stop_z are not traded (no positive risk distance); each skip is "
        "logged; a cell with stop_z at/beyond the entry z would be flagged NA ('degenerate grid cell').",
        "No 'first' clause: may re-enter after being flat. Borrow/SSR applied by the engine.",
    )
    spec = 'After 10:00 z>=+2 and daily close<SMA200 or no-trend filter; 5m RSI(2)>90; short next bar. || Exit at VWAP/12 bars; stop z=+3.'

    def signal(self, ctx, sym, z, r, c):
        p = self.params
        if not (z >= p["z"] and r > p["rsi"]):
            return False
        if p["gate"]:
            di = self.daily(ctx, sym)
            return di.prior_close < di.sma(200)
        return True

    def manage(self, ctx, sym, tp, i, z, now):
        p = self.params
        d = self.st(sym)
        if z >= p["stop_z"]:
            self.exit_position(ctx, sym, "stop_z")
        elif now >= d.get("exit_at", math.inf):
            self.exit_position(ctx, sym, "time_exit")
        else:
            self.refresh_limit_exit(ctx, sym, float(tp.vwap[i]))


# ===================================================================== S025
@register
class S025(IntradayStrategy):
    id = "S025"; name = "First-hour oversold reversal"; bar_freq = "5m"; asset = "ETF"
    universe = ETF_CORE; universe_name = None
    params = {"loss": -0.01, "rsi": 5}
    grid = {"loss": [-0.005, -0.01, -0.015], "rsi": [2, 5, 10]}
    requires = requires_intraday("5m")
    deviations = MODULE_DEVIATIONS + (
        "At 10:30: return = close of the 10:25-10:30 bar / 09:30 opening print - 1 <= loss AND RSI(2) of 5m closes "
        "< threshold ('RSI 2/5/10' = thresholds; period 2) arm the setup; then the first later 5m bar closing above "
        "its previous bar's high triggers the buy (once per day).",
        "Stop = session low at the signal. 'exit VWAP' = sell limit at the current session VWAP replaced at every "
        "5m close.",
    )
    spec = 'At 10:30 return from 09:30 open <=-1%; buy if RSI(2,5m)<5 and first 5m bar closes above previous bar high. || Stop day low; exit VWAP or 15:55.'

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        t1030 = self.at("10:30")
        if now < t1030:
            return
        for sym in self._syms:
            d = self.st(sym)
            pos = ctx.position(sym)
            if pos > 0:
                tp = self.tape(ctx, sym)
                if tp.n and tp.end(tp.n - 1) == now:
                    self.refresh_limit_exit(ctx, sym, float(tp.vwap[tp.n - 1]))
                continue
            if d.get("dead") or self.entries_today(sym) or (now > t1030 and not d.get("armed")):
                continue
            tp = self.tape(ctx, sym)
            i = tp.n - 1
            if i < 1 or tp.end(i) != now:
                continue
            if now == t1030:
                if "rsi" not in d:
                    d["rsi"] = rsi_stream(2)
                r = d["rsi"].sync(ctx, tp, self.base_min)[i]
                ret = float(tp.c[i]) / tp.first_open() - 1.0
                if tp.has_open_bar() and ret <= p["loss"] and r < p["rsi"]:
                    d["armed"] = True
                else:
                    d["dead"] = True
                continue
            if d.get("armed") and tp.c[i] > tp.h[i - 1]:
                d["dead"] = True
                self.enter(ctx, sym, 1, float(tp.c[i]), float(tp.l[:i + 1].min()), why="first-hour reversal")


# ===================================================================== S026
@register
class S026(IntradayStrategy):
    id = "S026"; name = "Late-day mean reversion"; bar_freq = "5m"; asset = "ETF"
    universe = ETF_CORE; universe_name = None
    params = {"threshold": 0.0125, "vwap_dist": 0.01, "vol_mult": 1.5}
    grid = {"threshold": [0.0075, 0.0125, 0.02], "vwap_dist": [0.005, 0.01]}
    requires = requires_intraday("5m")
    daily_n = 275
    deviations = MODULE_DEVIATIONS + (
        "At 14:30: r = close of the 14:25-14:30 bar / prior adjusted close - 1 <= -threshold; |r| > 1.5 x sigma, "
        "sigma = 20d daily return sd (prior sessions) scaled to the time since the prior close with "
        "indicators.vol_scaled_to_time using the causal 252-session overnight variance share (no signal while that "
        "estimate is unavailable); close <= VWAP x (1 - vwap_dist).",
        "'buy at 14:35' = market order submitted at 14:35 (end of the 14:30-14:35 bar) filling at the 14:35 bar "
        "open; '14:30 low' = low of the 14:30-14:35 bar; stop = that low - 0.25 x ATR(14, 5m).",
        "'VWAP touch' = sell limit at the current session VWAP replaced at every 5m close.",
        "Early-close sessions: 14:30/14:35 shift to close-90/-85 min.",
    )
    spec = 'At 14:30 return from prev close <=-1.25%, >1.5*20d daily vol scaled to time, and price below VWAP by >=1%; buy at 14:35. || Exit at 15:55 or VWAP touch; stop below 14:30 low by .25 ATR5m.'

    def session_start(self, ctx):
        self.t_sig = min(self.at("14:30"), self.close_ns - 90 * NS_MIN)
        self.t_buy = self.t_sig + 5 * NS_MIN
        self.armed: set[str] = set()

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        if now == self.t_sig:
            for sym in self._syms:
                tp = self.tape(ctx, sym)
                i = tp.n - 1
                if i < 0 or tp.end(i) != now:
                    continue
                di = self.daily(ctx, sym)
                pc = di.prior_close
                c = float(tp.c[i])
                r = c / pc - 1.0 if pc > 0 else NAN
                if not (r <= -p["threshold"] and c <= tp.vwap[i] * (1 - p["vwap_dist"])):
                    continue
                if not di.ok or len(di.close) < 253:
                    continue
                sig_d = float(realized_vol(di.close, 20, annualize=None)[-1])
                share = float(overnight_variance_share(di.open, di.close, 252)[-1])
                mins = (now - self.open_ns) / NS_MIN
                sig_t = float(vol_scaled_to_time(sig_d, mins, (self.close_ns - self.open_ns) / NS_MIN, share))
                if sig_t == sig_t and abs(r) > p["vol_mult"] * sig_t:
                    self.armed.add(sym)
        elif now == self.t_buy:
            for sym in sorted(self.armed):
                tp = self.tape(ctx, sym)
                i = tp.n - 1
                if i < 0 or tp.end(i) != now:
                    continue
                a = self.st(sym).setdefault("atr", atr_stream(14)).sync(ctx, tp, self.base_min)[i]
                self.enter(ctx, sym, 1, float(tp.c[i]), float(tp.l[i]) - 0.25 * a, why="late-day reversion")
            self.armed = set()
        elif now > self.t_buy:
            for sym, q in ctx.positions().items():
                if q > 0:
                    tp = self.tape(ctx, sym)
                    if tp.n and tp.end(tp.n - 1) == now:
                        self.refresh_limit_exit(ctx, sym, float(tp.vwap[tp.n - 1]))


# ===================================================================== S027
@register
class S027(IntradayStrategy):
    id = "S027"; name = "Intraday double-bottom reversal"; bar_freq = "5m"; asset = "STK/ETF"
    universe = ETF_CORE; universe_name = MEGACAP_UNIVERSE
    params = {"tol": 0.0025, "sep": (3, 12), "target_r": 2.0}
    grid = {"tol": [0.001, 0.0025, 0.005], "sep": [(3, 12), (5, 20)]}
    requires = requires_intraday("5m")
    deviations = MODULE_DEVIATIONS + (
        "Pattern = indicators.patterns.double_bottom on today's 5m bars (pivot lows confirmed 2 bars later, no "
        "lower low between, neckline = max high between, signal on the first fresh close above the neckline after "
        "the second low is confirmed). Both low bars must start at/after 10:00 and the signal bar must close by "
        "14:00.",
        "Stop = min(two lows) - 0.1 x ATR(14, 5m) at the signal.",
        "No 'first' clause: may re-enter after being flat.",
    )
    spec = 'Between 10:00-14:00 identify two local 5m lows separated 3-12 bars with prices within 0.25% and neckline high between them; buy close above neckline. || Stop min(two lows)-0.1ATR5m; exit 2R or 15:55.'

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        t10, t14 = self.at("10:00"), self.at("14:00")
        if now <= t10 or now > t14:
            return
        lo, hi = p["sep"]
        for sym in self._syms:
            tp = self.tape(ctx, sym)
            n = tp.n
            if n < 2 * 2 + int(lo) or tp.end(n - 1) != now or self.has_exposure(ctx, sym):
                continue
            db = double_bottom(tp.h[:n], tp.l[:n], tp.c[:n], 2, p["tol"], int(lo), int(hi))
            if not bool(db["signal"][-1]):
                continue
            b1 = int(db["bar1"][-1])
            if tp.start[b1] < t10:
                continue
            a = self.st(sym).setdefault("atr", atr_stream(14)).sync(ctx, tp, self.base_min)[n - 1]
            stop = min(float(db["low1"][-1]), float(db["low2"][-1])) - 0.1 * a
            self.enter(ctx, sym, 1, float(tp.c[n - 1]), stop, target_r=p["target_r"], why="double bottom")


# ===================================================================== S028
@register
class S028(IntradayStrategy):
    id = "S028"; name = "Intraday RSI exhaustion fade"; bar_freq = "5m"; asset = "ETF"
    universe = ETF_CORE; universe_name = None
    params = {"rsi_ext": 80, "sigma": 2.0, "cross": 75}
    grid = {"rsi_ext": [75, 80, 85], "sigma": [1.5, 2.0, 2.5]}
    requires = requires_intraday("5m")
    deviations = MODULE_DEVIATIONS + (
        "Arm (bar starting at/after 10:00): Wilder RSI(14) of 5m closes (spanning sessions) >= rsi_ext and close >= "
        "VWAP + sigma x sd; short at the next bar after the first bar whose RSI crosses below 75 (RSI_t-1 >= 75 > "
        "RSI_t). Mirror: RSI <= 100-rsi_ext and close <= VWAP - sigma x sd, then RSI crossing above 25.",
        "Stop = day extreme at the signal (session high for the short, low for the long). 'cover at VWAP' = limit "
        "at the current session VWAP replaced at every 5m close.",
        "Arming resets after each trade; re-entry allowed when flat (no 'one per day' clause).",
    )
    spec = 'After 10:00 5m RSI14 >=80 and price>=VWAP+2 intraday σ; short on first RSI crossing below 75. Mirror RSI<=20 long. || Stop day extreme; cover at VWAP or 15:55.'

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        t10 = self.at("10:00")
        ext, cross = float(p["rsi_ext"]), float(p["cross"])
        for sym in self._syms:
            tp = self.tape(ctx, sym)
            d = self.st(sym)
            if "rsi" not in d:
                d["rsi"] = rsi_stream(14)
            i = tp.n - 1
            if i < 1 or tp.end(i) != now:
                continue
            if ctx.position(sym):
                self.refresh_limit_exit(ctx, sym, float(tp.vwap[i]))
                continue
            r = d["rsi"].sync(ctx, tp, self.base_min)
            if tp.start[i] < t10 or self.has_exposure(ctx, sym):
                continue
            c, vw, sd = float(tp.c[i]), float(tp.vwap[i]), float(tp.sd[i])
            if d.get("arm") == -1 and r[i - 1] >= cross > r[i]:
                d["arm"] = 0
                self.enter(ctx, sym, -1, c, float(tp.h[:i + 1].max()), why="RSI exhaustion short")
                continue
            if d.get("arm") == 1 and r[i - 1] <= 100 - cross < r[i]:
                d["arm"] = 0
                self.enter(ctx, sym, 1, c, float(tp.l[:i + 1].min()), why="RSI exhaustion long")
                continue
            if r[i] >= ext and c >= vw + p["sigma"] * sd:
                d["arm"] = -1
            elif r[i] <= 100 - ext and c <= vw - p["sigma"] * sd:
                d["arm"] = 1


# ===================================================================== S029
@register
class S029(IntradayStrategy):
    id = "S029"; name = "Previous-close magnet"; bar_freq = "5m"; asset = "ETF"
    universe = ETF_CORE; universe_name = None
    params = {"gap": 0.0075, "retrace": 0.25}
    grid = {"gap": [0.005, 0.0075, 0.01], "retrace": [0.25, 0.5]}
    requires = requires_intraday("5m")
    deviations = MODULE_DEVIATIONS + (
        "At 11:00: gap = 09:30 opening print / prior adjusted close - 1; retrace = (open - price)/(open - prior "
        "close) with price = close of the 10:55-11:00 bar; trade toward the prior close (short a gap up, buy a gap "
        "down) if VWAP slope (VWAP after the 11:00 bar minus VWAP one bar earlier) has the trade's sign.",
        "Target = prior close (limit); stop = session high (short) / low (long) through 11:00; skipped if price "
        "is already beyond the prior close. One decision per day.",
    )
    spec = 'At 11:00 gap from previous close absolute >=0.75%, price has retraced at least 25% of gap, trade toward previous close if VWAP slope supports trade. || Target previous close; stop at 11:00 RTH extreme; exit 15:55.'

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        if now != self.at("11:00"):
            return
        for sym in self._syms:
            tp = self.tape(ctx, sym)
            i = tp.n - 1
            if i < 1 or tp.end(i) != now or not tp.has_open_bar():
                continue
            pc = self.daily(ctx, sym).prior_close
            o0, c = tp.first_open(), float(tp.c[i])
            gap = o0 / pc - 1.0 if pc > 0 else NAN
            if not abs(gap) >= p["gap"]:
                continue
            side = -_sgn(gap)
            retr = (o0 - c) / (o0 - pc)
            slope = tp.vwap[i] - tp.vwap[i - 1]
            if retr >= p["retrace"] and _sgn(slope) == side:
                stop = float(tp.h[:i + 1].max()) if side < 0 else float(tp.l[:i + 1].min())
                self.enter(ctx, sym, side, c, stop, target=pc, why=f"prev-close magnet gap={gap:.4f}")


# ===================================================================== S030
@register
class S030(IntradayStrategy):
    id = "S030"; name = "Intraday exhaustion spike reversal"; bar_freq = "1m"; asset = "STK"
    universe = (); universe_name = MEGACAP_UNIVERSE
    params = {"range_mult": 4.0, "confirm": 1, "lookback_min": 60, "hold_min": 30}
    grid = {"range_mult": [3.0, 4.0, 5.0], "confirm": [1, 2]}
    requires = requires_intraday("1m")
    deviations = MODULE_DEVIATIONS + (
        "Spike = completed 1m bar (non-zero volume) with range >= range_mult x median range of the previous 60 1m "
        "RTH bars of the same session (so no spike before 10:30). Direction = up if close >= open else down.",
        "Confirmation = each of the next `confirm` bars closes beyond the spike bar's midpoint against the spike "
        "(below it for an up spike); fade at the next bar open. A new spike replaces a pending one.",
        "Stop = spike extreme; target = midpoint of the bar before the spike; time exit 30 min after the entry "
        "bar opens.",
        "Halts: the engine blocks fills during HALT events / zero-volume bars; additionally no entry is submitted "
        "while a HALT event known at the decision time is in force.",
        "No 'first' clause: may re-enter after being flat.",
    )
    spec = 'Completed 1m bar range>=4*median of previous 60m; next 1m closes inside prior bar midpoint, fade initial spike direction. || Stop spike extreme; target pre-spike midpoint; exit 30m later.'

    def _halted(self, ctx, sym) -> bool:
        try:
            evs = ctx.portal().events("HALT", [sym], ctx.now - pd.Timedelta(hours=12), ctx.now + pd.Timedelta(1, "ns"))
        except Exception:
            return False
        for e in evs or []:
            st = pd.Timestamp(e.effective_at if e.effective_at is not None else e.known_at)
            res = (e.payload or {}).get("resume_at")
            en = pd.Timestamp(res) if res is not None else None
            st = st.tz_localize("UTC") if st.tzinfo is None else st
            if en is not None and en.tzinfo is None:
                en = en.tz_localize("UTC")
            if st <= ctx.now and (en is None or ctx.now < en):
                return True
        return False

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        L = int(p["lookback_min"])
        k = int(p["confirm"])
        for sym in self._syms:
            d = self.st(sym)
            tp = self.tape(ctx, sym)
            for i in range(d.get("k", 0), tp.n):
                d["k"] = i + 1
                sp = d.get("spike")
                if sp is not None and not self.has_exposure(ctx, sym):
                    j, up, mid = sp
                    if (tp.c[i] < mid) if up else (tp.c[i] > mid):
                        if i - j == k:
                            d["spike"] = None
                            if i == tp.n - 1 and tp.end(i) == now and not self._halted(ctx, sym):
                                side = -1 if up else 1
                                stop = float(tp.h[j]) if up else float(tp.l[j])
                                tgt = 0.5 * float(tp.h[j - 1] + tp.l[j - 1])
                                self.enter(ctx, sym, side, float(tp.c[i]), stop, target=tgt,
                                           time_exit_ns=now + int(p["hold_min"]) * NS_MIN, why="spike fade")
                            continue
                    else:
                        d["spike"] = None
                if i >= L and tp.v[i] > 0:
                    rng = tp.h[i - L:i] - tp.l[i - L:i]
                    if tp.h[i] - tp.l[i] >= p["range_mult"] * float(np.median(rng)) > 0:
                        d["spike"] = (i, bool(tp.c[i] >= tp.o[i]), 0.5 * float(tp.h[i] + tp.l[i]))


# ===================================================================== S116
@register
class S116(IntradayStrategy):
    id = "S116"; name = "Broad-market ETF opening gap fade"; bar_freq = "1m"; asset = "ETF"
    universe = ("SPY", "QQQ", "IWM"); universe_name = None
    params = {"gap": 0.005, "reversal": 0.25}
    grid = {"gap": [0.003, 0.005, 0.0075], "reversal": [0.2, 0.25, 0.5]}
    requires = requires_intraday("1m")
    deviations = MODULE_DEVIATIONS + (
        "Gap = 09:30 opening print / prior adjusted close - 1; at 09:40 the close of the 09:39 bar must have moved "
        "back toward the prior close by >= reversal x gap ((open - price)/(open - prior close)); all ten 09:30-"
        "09:39 bars required.",
        "Trade toward the prior close at the 09:40 bar open; target = prior close (limit); stop = opposite "
        "extreme of the opening 10 minutes (high for a short, low for a long). One decision per day.",
    )
    spec = 'Gap SPY/QQQ/IWM abs>0.5%; at 09:40 trade toward prior-close only if first 10m closes back toward it at least 25%. || Take at prior close or exit 15:55, stop opposite opening 10m extreme.'

    def step(self, ctx):
        p = self.params
        now = int(ctx.now.value)
        t940 = self.at("09:40")
        if now != t940:
            return
        for sym in self._syms:
            tp = self.tape(ctx, sym)
            w = tp.window(self.open_ns, t940)
            if w.stop - w.start != 10 or not tp.has_open_bar():
                continue
            pc = self.daily(ctx, sym).prior_close
            o0, c = tp.first_open(), float(tp.c[w.stop - 1])
            gap = o0 / pc - 1.0 if pc > 0 else NAN
            if not abs(gap) > p["gap"]:
                continue
            side = -_sgn(gap)
            if (o0 - c) / (o0 - pc) >= p["reversal"]:
                stop = float(tp.h[w].max()) if side < 0 else float(tp.l[w].min())
                self.enter(ctx, sym, side, c, stop, target=pc, why=f"ETF gap fade {gap:.4f}")
