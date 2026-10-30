"""Swing trend / breakout strategies S046-S067 (daily bars).

Every class: signal at the daily close, entry at the next session (market at
the open unless the sheet asks for a resting stop), sizing per the sheet's
profile. Shared conventions: ``swing_common.COMMON_DEVIATIONS``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..engine.strategy import Requirement
from ..indicators import (atr, bandwidth_percentile, bollinger, bullish_engulfing, darvas_box, dmi, donchian, ema,
                          head_and_shoulders, high_ratio, ichimoku, inside_bar, macd, nr_n, parabolic_sar, percentile_rank,
                          pivots, prior_all_time_high, realized_vol, rsi, sma, squeeze_on, supertrend,
                          volatility_contraction)
from ..indicators import cross_section as xs
from .registry import register
from .swing_common import (COMMON_DEVIATIONS, SECTOR_CAVEAT, SP500_PIT, SWING_ETF, Bars, CrossSectionBase, Plan, PosState, SwingBase,
                           crossed_above, crossed_below, finite, tick_for)
from .swing_specs import NAMES, SPECS


PYRAMID_NOTE = (
    "Pyramiding under SWING_RISK: every add is sized as one risk unit (0.5% NAV at its own 2N stop) but the "
    "position's combined notional stays <= 10% NAV (and aggregate gross <= 100%). With typical ETF volatility "
    "(N ~ 1-2% of price) the FIRST unit already reaches the 10% notional cap, so adds are clipped: an add smaller "
    "than 25% of the first unit is not placed (logged, counted in _clipped_adds) and neither counts as a unit "
    "nor raises the stop; the pyramid grid dimension is then inert - faithful to the sheet's sizing profile. "
    "Add levels lie on the grid last fill + k*0.5N; a level the market closed through without triggering is "
    "re-anchored to the next grid level beyond the close. One add order rests at a time, so at most one add "
    "fills per session (a gap through several levels adds one unit).")


def _last(x, k: int = 1) -> float:
    a = np.asarray(x, dtype=float)
    return float(a[-k]) if a.shape[0] >= k else float("nan")


def _atr_last(b: Bars, n: int) -> float:
    return _last(atr(b.h, b.l, b.c, n))


def _sma_up(c: np.ndarray, n: int = 200, lag: int = 20) -> bool:
    """Guide rule 9: SMA_n today > SMA_n `lag` sessions ago."""
    s = np.asarray(sma(c, n))
    return s.shape[0] > lag and finite(s[-1], s[-1 - lag]) and s[-1] > s[-1 - lag]


# ===================================================================== S046
@register
class S046(SwingBase):
    id = "S046"; name = NAMES["S046"]; asset = "STK/ETF"; sizing = "SWING_RISK"; universe = SWING_ETF
    params = {"entry_n": 20, "exit_n": 10, "stop_atr": 2.0, "atr_n": 20, "short": True}
    grid = {"entry_n": [20, 40, 55], "exit_n": [10, 20], "stop_atr": [1.5, 2.0, 3.0]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S046"]
    allow_reversal = True
    deviations = COMMON_DEVIATIONS + [
        "Short mirror is part of the baseline (sheet entry text): C_t < min(L_{t-20:t-1}) -> short next open, stop "
        "2*ATR20 above fill, cover on close above prior 10d high. A long exit and an opposite breakout on the same "
        "close reverse the position at the next open.",
        "ATR20 = Wilder ATR (atr_n), measured at the signal close; stop = fill -/+ k*ATR20 (fixed, not trailing).",
    ]

    def window(self):
        return 400

    def entry(self, ctx, sym, b):
        p = self.params
        dc = donchian(b.h, b.l, p["entry_n"])
        c, up, lo = b.c[-1], _last(dc["upper"]), _last(dc["lower"])
        a = _atr_last(b, p["atr_n"])
        if not finite(up, lo, a):
            return None
        side = 1 if c > up else (-1 if p["short"] and c < lo else 0)
        if not side:
            return None
        return Plan(side, c, stop_mode="atr_fill", stop_k=p["stop_atr"], atr=a)

    def exit_signal(self, ctx, sym, b, ps):
        dc = donchian(b.h, b.l, self.params["exit_n"])
        c = b.c[-1]
        if ps.side > 0 and c < _last(dc["lower"]):
            return "channel_exit"
        if ps.side < 0 and c > _last(dc["upper"]):
            return "channel_exit"
        return None


# ===================================================================== S047
@register
class S047(SwingBase):
    id = "S047"; name = NAMES["S047"]; asset = "STK/ETF"; sizing = "SWING_RISK"; universe = SWING_ETF
    params = {"entry_n": 55, "exit_n": 20, "stop_atr": 2.0, "atr_n": 20, "adds": 0, "add_atr": 0.5, "short": True}
    grid = {"entry_n": [55, 60], "exit_n": [20, 25], "adds": [0, 1, 2]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S047"]
    allow_reversal = True
    deviations = COMMON_DEVIATIONS + [
        "No-skip variant (sheet: 'choose no-skip variant explicitly'): every 55d breakout is taken when flat.",
        "Short mirror is part of the baseline (sheet: '(short mirror)').",
        "Pyramiding (grid 0/1/2 adds; baseline 0, the first grid value since the sheet gives no baseline): Turtle "
        "rule - a resting buy (sell) stop at last fill +/- 0.5*N (N = Wilder ATR20 at the entry signal), each add "
        "sized as one SWING_RISK unit (0.5% NAV at its own 2N stop) within the 10% notional / 100% gross caps; after "
        "an add the protective stop for all units is raised to max(trailing stop, add fill - 2N).",
        PYRAMID_NOTE,
        "Trailing stop 2*ATR20: ratchet on closes, C_t - 2*ATR20_t (see common conventions).",
    ]

    def window(self):
        return 450

    def entry(self, ctx, sym, b):
        p = self.params
        dc = donchian(b.h, b.l, p["entry_n"])
        c, up, lo = b.c[-1], _last(dc["upper"]), _last(dc["lower"])
        n = _atr_last(b, p["atr_n"])
        if not finite(up, lo, n):
            return None
        side = 1 if c > up else (-1 if p["short"] and c < lo else 0)
        if not side:
            return None
        return Plan(side, c, stop_mode="atr_fill", stop_k=p["stop_atr"], atr=n, trail_k=p["stop_atr"],
                    trail_atr_n=p["atr_n"], max_adds=int(p["adds"]), add_atr=p["add_atr"])

    def exit_signal(self, ctx, sym, b, ps):
        dc = donchian(b.h, b.l, self.params["exit_n"])
        c = b.c[-1]
        if ps.side > 0 and c < _last(dc["lower"]):
            return "channel_exit"
        if ps.side < 0 and c > _last(dc["upper"]):
            return "channel_exit"
        return None


# ===================================================================== S048
class _S1Tracker:
    """Hypothetical Turtle S1 trades (taken or not) for the skip rule.

    Bar-by-bar, causal: a breakout at close i (C_i > prior entry_n high) with no
    hypothetical trade open starts one: entry = open_{i+1}, stop = entry - 2N_i;
    it exits at min(open, stop) when the low reaches the stop (gap-aware), or at
    the next open after a close below the prior exit_n low. Outcome = exit >
    entry. Positions are stored by bar start time and prices re-read from the
    current adjusted series (robust to splits/dividends)."""

    def __init__(self):
        self.last_start = None
        self.pending_signal = None          # signal bar start (entry next open)
        self.open = None                    # (entry_bar_start, signal_bar_start)
        self.exit_next = False
        self.last_outcome = None            # True = winner, False = loser

    def update(self, b: Bars, upper: np.ndarray, lower_exit: np.ndarray, n_atr: np.ndarray, k: float) -> None:
        st = b.start
        i0 = 0 if self.last_start is None else int(np.searchsorted(st, self.last_start, side="right"))
        pos = {int(s): j for j, s in enumerate(st)} if (self.open or self.pending_signal is not None) else None
        for i in range(i0, len(st)):
            if self.pending_signal is not None:
                self.open = (int(st[i]), self.pending_signal)
                self.pending_signal = None
                self.exit_next = False
                pos = pos or {int(s): j for j, s in enumerate(st)}
            if self.open is not None:
                pos = pos or {int(s): j for j, s in enumerate(st)}
                je, js = pos.get(self.open[0]), pos.get(self.open[1])
                if je is None or js is None:            # fell out of the window: drop
                    self.open = None
                else:
                    entry = b.o[je]
                    stop = entry - k * n_atr[js]
                    exit_px = None
                    if self.exit_next:
                        exit_px = b.o[i]
                    elif b.l[i] <= stop:
                        exit_px = min(b.o[i], stop)
                    elif np.isfinite(lower_exit[i]) and b.c[i] < lower_exit[i]:
                        self.exit_next = True
                    if exit_px is not None:
                        self.last_outcome = bool(exit_px > entry)
                        self.open = None
                        self.exit_next = False
            if self.open is None and self.pending_signal is None and np.isfinite(upper[i]) \
                    and np.isfinite(n_atr[i]) and b.c[i] > upper[i]:
                self.pending_signal = int(st[i])
        self.last_start = int(st[-1])


@register
class S048(SwingBase):
    id = "S048"; name = NAMES["S048"]; asset = "STK/ETF"; sizing = "SWING_RISK"; universe = SWING_ETF
    params = {"entry_n": 20, "exit_n": 10, "stop_atr": 2.0, "atr_n": 20, "skip": True, "add_atr": None,
              "max_units": 4}
    grid = {"skip": [True, False], "entry_n": [20, 55], "add_atr": [None, 0.5]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S048"]
    deviations = COMMON_DEVIATIONS + [
        "Skip rule (Turtle S1): every breakout (C_t > prior entry_n high) that occurs while no hypothetical S1 trade "
        "is open starts a hypothetical trade (entry next open, exit 2N stop gap-aware or close < prior 10d low -> next "
        "open), tracked whether or not it was taken. A real breakout entry is skipped when the last COMPLETED "
        "hypothetical trade was a winner (exit > entry, costs ignored). Tracking starts at the beginning of the "
        "fetched history window (>= 400 sessions before the first decision).",
        "Grid 'add threshold 0.5ATR' read as {None = baseline (sheet: 'no pyramids in baseline'), 0.5}: with 0.5, "
        "Turtle adds at +0.5N from the last fill up to max_units=4 units, stops for all units raised to last fill - 2N.",
        PYRAMID_NOTE,
        "2*Wilder ATR20 stop is fixed from the fill (not trailing); 'first' = whichever of the resting stop or the "
        "channel exit happens first.",
        "Long only (sheet entry text has no short mirror).",
    ]

    def window(self):
        return 450

    def init(self, ctx):
        super().init(ctx)
        self._trk: dict[str, _S1Tracker] = {}
        self._trk_ver: dict[str, int] = {}

    def _tracker(self, sym, b):
        p = self.params
        up = np.asarray(donchian(b.h, b.l, p["entry_n"])["upper"])
        lo = np.asarray(donchian(b.h, b.l, p["exit_n"])["lower"])
        n = np.asarray(atr(b.h, b.l, b.c, p["atr_n"]))
        t = self._trk.setdefault(sym, _S1Tracker())
        t.update(b, up, lo, n, p["stop_atr"])
        return t, up, n

    def on_day_start(self, ctx, syms):
        # keep the hypothetical tracker current for every symbol, also while in a position
        for sym in syms:
            b = self._bars(ctx, sym)
            if b is not None and len(b) and b.end[-1] == int(ctx.now.value):
                self._day = getattr(self, "_day", {})
                self._day[sym] = self._tracker(sym, b)

    def entry(self, ctx, sym, b):
        p = self.params
        got = getattr(self, "_day", {}).get(sym)
        if got is None:
            return None
        t, up, n = got
        c = b.c[-1]
        if not finite(up[-1], n[-1]) or not c > up[-1]:
            return None
        # the tracker just registered today's breakout as pending; the decision uses the prior COMPLETED one
        if p["skip"] and t.last_outcome is True:
            ctx.log("S048 skip: prior S1 breakout was a winner", symbol=sym)
            return None
        adds = (int(p["max_units"]) - 1) if p["add_atr"] is not None else 0
        return Plan(1, c, stop_mode="atr_fill", stop_k=p["stop_atr"], atr=float(n[-1]), max_adds=adds,
                    add_atr=p["add_atr"])

    def exit_signal(self, ctx, sym, b, ps):
        lo = _last(donchian(b.h, b.l, self.params["exit_n"])["lower"])
        return "channel_exit" if b.c[-1] < lo else None


# ===================================================================== S049
@register
class S049(SwingBase):
    id = "S049"; name = NAMES["S049"]; asset = "STK/ETF"; sizing = "SWING_NOTIONAL"; universe = SWING_ETF
    params = {"fast": 20, "slow": 50, "trend_n": 200, "trend_filter": True, "trail_atr": 2.5, "atr_n": 14}
    grid = {"fast": [10, 20, 30], "slow": [50, 100], "trend_filter": [True, False]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S049"]
    deviations = COMMON_DEVIATIONS + [
        "Grid 'filter 200 on/off' = trend_filter True/False (close > SMA200 required when on).",
        "2.5ATR trailing stop uses Wilder ATR14 (period not stated); resting stop, 'next fill' = gap-aware.",
    ]

    def entry(self, ctx, sym, b):
        p = self.params
        f, s = np.asarray(sma(b.c, p["fast"])), np.asarray(sma(b.c, p["slow"]))
        if len(b) < 2 or not crossed_above(f[-2], f[-1], s[-2], s[-1]):
            return None
        if p["trend_filter"]:
            t = _last(sma(b.c, p["trend_n"]))
            if not (finite(t) and b.c[-1] > t):
                return None
        a = _atr_last(b, p["atr_n"])
        if not finite(a):
            return None
        return Plan(1, b.c[-1], stop_mode="atr_fill", stop_k=p["trail_atr"], atr=a, trail_k=p["trail_atr"],
                    trail_atr_n=p["atr_n"])

    def exit_signal(self, ctx, sym, b, ps):
        p = self.params
        f, s = np.asarray(sma(b.c, p["fast"])), np.asarray(sma(b.c, p["slow"]))
        return "ma_cross_exit" if crossed_below(f[-2], f[-1], s[-2], s[-1]) else None


# ===================================================================== S050
@register
class S050(SwingBase):
    id = "S050"; name = NAMES["S050"]; asset = "STK/ETF"; sizing = "SWING_NOTIONAL"; universe = SWING_ETF
    params = {"fast": 50, "slow": 200}
    grid = {"fast": [40, 50, 60], "slow": [150, 200, 250]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S050"]
    deviations = COMMON_DEVIATIONS + [
        "No price stop (sheet default); cash interest on the uninvested balance is paid by the engine's financing model.",
    ]

    def window(self):
        return 400

    def entry(self, ctx, sym, b):
        p = self.params
        f, s = np.asarray(sma(b.c, p["fast"])), np.asarray(sma(b.c, p["slow"]))
        if len(b) < 2 or not crossed_above(f[-2], f[-1], s[-2], s[-1]):
            return None
        return Plan(1, b.c[-1])

    def exit_signal(self, ctx, sym, b, ps):
        p = self.params
        f, s = np.asarray(sma(b.c, p["fast"])), np.asarray(sma(b.c, p["slow"]))
        return "death_cross" if crossed_below(f[-2], f[-1], s[-2], s[-1]) else None


# ===================================================================== S051
@register
class S051(SwingBase):
    id = "S051"; name = NAMES["S051"]; asset = "STK/ETF"; sizing = "SWING_RISK"; universe = SWING_ETF
    params = {"fast": 9, "mid": 21, "slow": 50, "exit_ema": 21, "stop_lookback": 5}
    grid = {"fast": [8, 9, 10], "mid": [20, 21], "exit_ema": [20, 21]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S051"]
    deviations = COMMON_DEVIATIONS + [
        "'prior day low<=EMA9' compares L_{t-1} with EMA9_{t-1}; 'current close>prior high' = C_t > H_{t-1}.",
        "'Stop beneath last 5-day low' = resting stop 1 tick below min(L_{t-4..t}) (the 5 sessions ending at the "
        "signal), fixed for the life of the trade.",
    ]

    def entry(self, ctx, sym, b):
        p = self.params
        c = b.c
        ef, em, es = np.asarray(ema(c, p["fast"])), np.asarray(ema(c, p["mid"])), np.asarray(ema(c, p["slow"]))
        if len(b) < 2 or not finite(ef[-2], em[-1], es[-1]):
            return None
        if not (c[-1] > em[-1] > es[-1] and b.l[-2] <= ef[-2] and c[-1] > b.h[-2]):
            return None
        lo = float(np.min(b.l[-p["stop_lookback"]:]))
        stop = lo - tick_for(lo)
        return Plan(1, c[-1], stop_mode="level", stop_level=stop)

    def exit_signal(self, ctx, sym, b, ps):
        e = _last(ema(b.c, self.params["exit_ema"]))
        return "close_below_ema" if b.c[-1] < e else None


# ===================================================================== S052
@register
class S052(SwingBase):
    id = "S052"; name = NAMES["S052"]; asset = "STK/ETF"; sizing = "SWING_RISK"; universe = SWING_ETF
    params = {"fast": 12, "slow": 26, "signal": 9, "trend_n": 200, "trail_atr": 2.0, "atr_n": 14}
    grid = {"fast": [8, 12, 16], "slow": [21, 26, 35], "signal": [5, 9, 12]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S052"]
    deviations = COMMON_DEVIATIONS + [
        "Histogram crosses positive = hist_{t-1} <= 0 < hist_t (hist = MACD - signal EMA); 2ATR trailing uses ATR14.",
    ]

    def entry(self, ctx, sym, b):
        p = self.params
        m = macd(b.c, p["fast"], p["slow"], p["signal"])
        h, line = np.asarray(m["hist"]), np.asarray(m["macd"])
        t = _last(sma(b.c, p["trend_n"]))
        if len(b) < 2 or not finite(h[-2], h[-1], line[-1], t):
            return None
        if not (h[-2] <= 0 < h[-1] and line[-1] > 0 and b.c[-1] > t):
            return None
        a = _atr_last(b, p["atr_n"])
        return Plan(1, b.c[-1], stop_mode="atr_fill", stop_k=p["trail_atr"], atr=a, trail_k=p["trail_atr"],
                    trail_atr_n=p["atr_n"]) if finite(a) else None

    def exit_signal(self, ctx, sym, b, ps):
        p = self.params
        h = np.asarray(macd(b.c, p["fast"], p["slow"], p["signal"])["hist"])
        return "hist_negative" if finite(h[-2], h[-1]) and h[-2] >= 0 > h[-1] else None


# ===================================================================== S053
@register
class S053(SwingBase):
    id = "S053"; name = NAMES["S053"]; asset = "STK/ETF"; sizing = "SWING_RISK"; universe = SWING_ETF
    params = {"adx_min": 25, "channel": 20, "adx_n": 14, "exit_ema": 20, "trail_atr": 2.0, "atr_n": 14}
    grid = {"adx_min": [20, 25, 30], "channel": [20, 40, 55]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S053"]
    deviations = COMMON_DEVIATIONS + ["ADX/DI = Wilder DMI(14) at the signal close."]

    def entry(self, ctx, sym, b):
        p = self.params
        up = _last(donchian(b.h, b.l, p["channel"])["upper"])
        d = dmi(b.h, b.l, b.c, p["adx_n"])
        adx_, pdi, mdi = _last(d["adx"]), _last(d["plus_di"]), _last(d["minus_di"])
        if not finite(up, adx_, pdi, mdi):
            return None
        if not (b.c[-1] > up and adx_ > p["adx_min"] and pdi > mdi):
            return None
        a = _atr_last(b, p["atr_n"])
        return Plan(1, b.c[-1], stop_mode="atr_fill", stop_k=p["trail_atr"], atr=a, trail_k=p["trail_atr"],
                    trail_atr_n=p["atr_n"]) if finite(a) else None

    def exit_signal(self, ctx, sym, b, ps):
        e = _last(ema(b.c, self.params["exit_ema"]))
        return "close_below_ema" if b.c[-1] < e else None


# ===================================================================== S054
@register
class S054(CrossSectionBase):
    id = "S054"; name = NAMES["S054"]
    params = {"high_n": 252, "top": 0.10, "lag": 1, "exit_q": 5, "no_sector_caps": False}
    grid = {"high_n": [126, 252], "top": [0.10, 0.20], "lag": [1, 5]}
    spec = SPECS["S054"]
    deviations = COMMON_DEVIATIONS[1:2] + [
        "Universe: point-in-time portal universe 'SP500' (members at the month-end decision).",
        "Ratio = C_t / max(H over the high_n sessions ending at t) (indicators.high_ratio; full window required).",
        "Month-end = last NYSE session of the month; 'lag 1/5 days' = sessions between the month-end signal close "
        "and the rebalance open (1 = next open, 5 = open of the 5th session after month end), a formation/holding "
        "skip.",
        "Entries: top (bottom) 'top' fraction of names long (short); a name already held stays while it remains in "
        "its side's extreme quintile (exit buffer 'close if moves out of quintile at rebalance'); all held names are "
        "re-weighted to equal weights at each rebalance.",
        "CROSS_SECTION weights via risk.sizing.cross_section_weights (+50%/-50%, 2% name cap, 10% per-side sector "
        "cap; without sector data the run is NOT_TESTABLE unless no_sector_caps=True, then caps are dropped and "
        "the run flagged T2); names without a valid price are dropped.",
        SECTOR_CAVEAT,
    ]

    def is_signal_day(self, ctx):
        return ctx.calendar().is_last_session_of_month(ctx.session_date())

    def exec_lag(self):
        return int(self.params["lag"])

    def window(self):
        return int(self.params["high_n"]) + 5

    def targets(self, ctx, members):
        p = self.params
        n = int(p["high_n"])
        vals = {}
        for s in members:
            b = self._cache.get(ctx, s, n + 5)
            if b is None or len(b) < n or b.end[-1] != int(ctx.now.value):
                continue
            r = _last(high_ratio(b.c, b.h, n))
            if finite(r):
                vals[s] = r
        if len(vals) < 10:
            return None
        import pandas as pd
        x = pd.Series(vals)
        q_in = int(round(1.0 / p["top"]))
        b_in = xs.quantile_bucket(x, q_in)
        b_out = xs.quantile_bucket(x, int(p["exit_q"]))
        held = ctx.positions()
        longs = [s for s in x.index if b_in[s] == q_in or (held.get(s, 0) > 0 and b_out[s] == p["exit_q"])]
        shorts = [s for s in x.index if b_in[s] == 1 or (held.get(s, 0) < 0 and b_out[s] == 1)]
        sec = self.sectors(ctx, longs + shorts)
        from ..risk.sizing import cross_section_weights
        return cross_section_weights(sorted(longs), sorted(shorts), sec or None)


# ===================================================================== S055
@register
class S055(SwingBase):
    id = "S055"; name = NAMES["S055"]; asset = "STK"; sizing = "SWING_RISK"
    universe = (); universe_name = SP500_PIT
    params = {"min_history": 252, "exit_n": 20, "min_dollar_vol": 20e6, "dv_n": 20, "stop_atr": 3.0, "atr_n": 14}
    grid = {"min_history": [126, 252], "exit_n": [10, 20, 40]}
    requires = [Requirement("bars", "1d", symbols=("SPY",), critical=False,
                            note="universe members resolved PIT through portal.universe('SP500')")]
    spec = SPECS["S055"]
    deviations = COMMON_DEVIATIONS + [
        "All-time high = max of all prior split/dividend-adjusted highs served by the portal for the symbol; if the "
        "portal history starts after the IPO the ATH is relative to the first available bar (data-depth caveat).",
        "20d dollar volume = mean of raw close*volume over the 20 sessions ending at the signal (>= $20m).",
        "3ATR stop = Wilder ATR14, fixed from the fill; 'exit close<lowest prior Nd low' -> market at next open.",
    ]

    def window(self):
        return None            # full history (ATH since first bar)

    def init(self, ctx):
        super().init(ctx)
        self._ath: dict[str, tuple] = {}

    def _prior_ath(self, sym, b):
        """max(H_0..H_{t-1}): indicators.prior_all_time_high on a (re)fetch, then kept incrementally."""
        ver = self._cache.version(sym)
        st = self._ath.get(sym)
        if st is None or st[0] != ver:
            prior = _last(prior_all_time_high(b.h, 1))
        else:
            _, last_start, mx = st
            k = int(np.searchsorted(b.start, last_start, side="right"))
            new = b.h[k:-1]
            prior = float(np.fmax(mx, np.nanmax(new))) if new.size and np.isfinite(new).any() else mx
        hi = float(b.h[-1])
        mx_now = prior if not np.isfinite(hi) else (hi if not np.isfinite(prior) else max(prior, hi))
        self._ath[sym] = (ver, int(b.start[-1]), mx_now)
        return prior

    def prefilter(self, ctx, sym):
        # a breakout needs C_t > prior ATH >= max(H) through the last processed bar (exact, not a proxy)
        st = self._ath.get(sym)
        if st is None or self._cache.stale(sym) or st[0] != self._cache.version(sym):
            return True
        try:
            px = float(ctx.mark(sym))
        except Exception:
            return True
        return not (np.isfinite(px) and np.isfinite(st[2]) and px <= st[2])

    def entry(self, ctx, sym, b):
        p = self.params
        if len(b) < 2:
            return None
        ath = self._prior_ath(sym, b)
        if len(b) - 1 < p["min_history"] or not finite(ath) or not b.c[-1] > ath:
            return None
        raw = ctx.history(sym, p["dv_n"], "1d", adjusted=False)
        if raw is None or len(raw) < p["dv_n"]:
            return None
        dv = float((raw["close"] * raw["volume"]).mean())
        if not dv >= p["min_dollar_vol"]:
            return None
        bt = b.tail(400)
        a = _atr_last(bt, p["atr_n"])
        return Plan(1, b.c[-1], stop_mode="atr_fill", stop_k=p["stop_atr"], atr=a) if finite(a) else None

    def exit_signal(self, ctx, sym, b, ps):
        bt = b.tail(self.params["exit_n"] + 2)
        lo = _last(donchian(bt.h, bt.l, self.params["exit_n"])["lower"])
        return "channel_exit" if b.c[-1] < lo else None


# ===================================================================== S056
@register
class S056(SwingBase):
    id = "S056"; name = NAMES["S056"]; asset = "STK/ETF"; sizing = "SWING_RISK"; universe = SWING_ETF
    params = {"bw_pct": 0.10, "bb_n": 20, "bb_k": 2.0, "bw_lookback": 252, "trail_atr": 2.0, "atr_n": 14}
    grid = {"bw_pct": [0.05, 0.10, 0.20], "bb_k": [1.5, 2.0, 2.5]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S056"]
    deviations = COMMON_DEVIATIONS + [
        "Squeeze = bandwidth percentile (mid-rank among the previous 252 values, indicators.bandwidth_percentile) <= "
        "bw_pct on the day BEFORE the breakout; breakout = close crosses above the upper band "
        "(C_{t-1} <= U_{t-1}, C_t > U_t). Grid 'BB 20, 1.5/2/2.5' = bb_n 20 with sigma 1.5/2/2.5 (bandwidth uses the "
        "same sigma).",
        "2ATR trailing stop uses Wilder ATR14.",
    ]

    def window(self):
        return 450

    def entry(self, ctx, sym, b):
        p = self.params
        bb = bollinger(b.c, p["bb_n"], p["bb_k"])
        up = np.asarray(bb["upper"])
        pct = np.asarray(bandwidth_percentile(b.c, p["bb_n"], p["bb_k"], p["bw_lookback"]))
        if len(b) < 2 or not finite(pct[-2], up[-2], up[-1]):
            return None
        if not (pct[-2] <= p["bw_pct"] and b.c[-2] <= up[-2] and b.c[-1] > up[-1]):
            return None
        a = _atr_last(b, p["atr_n"])
        return Plan(1, b.c[-1], stop_mode="atr_fill", stop_k=p["trail_atr"], atr=a, trail_k=p["trail_atr"],
                    trail_atr_n=p["atr_n"]) if finite(a) else None

    def exit_signal(self, ctx, sym, b, ps):
        m = _last(sma(b.c, self.params["bb_n"]))
        return "close_below_sma20" if b.c[-1] < m else None


# ===================================================================== S057
@register
class S057(SwingBase):
    id = "S057"; name = NAMES["S057"]; asset = "STK/ETF"; sizing = "SWING_RISK"; universe = SWING_ETF
    params = {"kc_k": 1.5, "bb_k": 2.0, "n": 20, "stop_atr": 3.0, "atr_n": 14}
    grid = {"kc_k": [1.0, 1.5, 2.0], "bb_k": [1.5, 2.0]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S057"]
    deviations = COMMON_DEVIATIONS + [
        "Squeeze on t-1 = BB(20,bb_k) fully inside Keltner(EMA20 +/- kc_k*ATR20) (indicators.squeeze_on); 'on "
        "release' = squeeze off on t; plus C_t > BB upper_t.",
        "3ATR stop = Wilder ATR14 (period not stated), fixed from the fill.",
    ]

    def entry(self, ctx, sym, b):
        p = self.params
        sq = np.asarray(squeeze_on(b.h, b.l, b.c, p["n"], p["bb_k"], p["kc_k"], p["n"]))
        up = _last(bollinger(b.c, p["n"], p["bb_k"])["upper"])
        if len(b) < 2 or not finite(up):
            return None
        if not (bool(sq[-2]) and not bool(sq[-1]) and b.c[-1] > up):
            return None
        a = _atr_last(b, p["atr_n"])
        return Plan(1, b.c[-1], stop_mode="atr_fill", stop_k=p["stop_atr"], atr=a) if finite(a) else None

    def exit_signal(self, ctx, sym, b, ps):
        e = _last(ema(b.c, self.params["n"]))
        return "close_below_ema20" if b.c[-1] < e else None


# ===================================================================== S058 / S059 helpers
class _StopEntryBase(SwingBase):
    """Next-session resting stop entry at signal-bar extreme +/- 1 tick."""

    def _plan(self, b: Bars, side: int, target_r: float, hold: int) -> Plan:
        if side > 0:
            px = float(b.h[-1]) + tick_for(float(b.h[-1]))
            stop = float(b.l[-1])
        else:
            px = float(b.l[-1]) - tick_for(float(b.l[-1]))
            stop = float(b.h[-1])
        return Plan(side, px, entry_type="STP", entry_stop=px, stop_mode="level", stop_level=stop,
                    target_r=target_r, max_hold=hold)


# ===================================================================== S058
@register
class S058(_StopEntryBase):
    id = "S058"; name = NAMES["S058"]; asset = "STK/ETF"; sizing = "SWING_RISK"; universe = SWING_ETF
    params = {"nr": 7, "hold": 5, "target_r": 2.0}
    grid = {"nr": [4, 7, 10], "hold": [3, 5, 10]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S058"]
    deviations = COMMON_DEVIATIONS + [
        "NR-n (indicators.nr_n): today's range <= each of the previous n-1 ranges (ties count).",
        "Entry: DAY buy-stop for the next session only at signal high + 1 tick (Rule 612 tick); with daily bars the "
        "engine fills at max(stop, open) with the intrabar path assumption. Initial stop = signal low (resting); "
        "2R target from the actual fill (planned entry on the entry day); exit at the next open after 'hold' "
        "sessions (entry session = 1).",
    ]

    def window(self):
        return 60

    def entry(self, ctx, sym, b):
        p = self.params
        if not bool(np.asarray(nr_n(b.h, b.l, p["nr"]))[-1]):
            return None
        if not b.h[-1] > b.l[-1]:
            return None
        return self._plan(b, 1, p["target_r"], p["hold"])


# ===================================================================== S059
@register
class S059(_StopEntryBase):
    id = "S059"; name = NAMES["S059"]; asset = "STK/ETF"; sizing = "SWING_RISK"; universe = SWING_ETF
    params = {"nested": 1, "target_r": 2.0, "hold": 5, "side": "long"}
    grid = {"nested": [1, 2], "target_r": [1.5, 2.0]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S059"]
    deviations = COMMON_DEVIATIONS + [
        "Inside day = H_t < H_{t-1} and L_t > L_{t-1} (strict); nested=2 requires 2 consecutive inside days and uses "
        "the latest inside day's range.",
        "Baseline is long only; the sheet's 'short stop below low is separate' is the separate variant side='short' "
        "(sell stop 1 tick below the inside-day low, stop at the inside-day high).",
        "Next-day DAY stop entry, fill at max(stop, open); stop at the inside-day extreme; target 1.5/2R from the "
        "actual fill; time exit after 5 sessions.",
    ]

    def window(self):
        return 60

    def entry(self, ctx, sym, b):
        p = self.params
        if not bool(np.asarray(inside_bar(b.h, b.l, p["nested"]))[-1]):
            return None
        side = 1 if p["side"] == "long" else -1
        return self._plan(b, side, p["target_r"], p["hold"])


# ===================================================================== S060
@register
class S060(SwingBase):
    id = "S060"; name = NAMES["S060"]; asset = "STK/ETF"; sizing = "SWING_RISK"; universe = SWING_ETF
    params = {"lookback": 20, "depth_atr": 2.0, "recent": 5, "atr_n": 20, "exit_n": 20}
    grid = {"lookback": [15, 20, 30], "depth_atr": [1.0, 2.0, 3.0]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S060"]
    deviations = COMMON_DEVIATIONS + [
        "indicators.darvas_box: resistance = max(H_{t-L..t-1}); box valid if max(H_{t-5..t-1}) < resistance and "
        "min(L_{t-5..t-1}) > resistance - depth*ATR20_{t-1}; signal C_t > resistance; entry next open.",
        "'Stop below 5d floor' = resting stop 1 tick below min(L_{t-5..t-1}); 'exit after close under 20d low' = "
        "close < prior 20d low -> next open.",
    ]

    def entry(self, ctx, sym, b):
        p = self.params
        d = darvas_box(b.h, b.l, b.c, p["lookback"], p["recent"], p["depth_atr"], p["atr_n"])
        if not bool(np.asarray(d["signal"])[-1]):
            return None
        fl = _last(d["floor"])
        return Plan(1, b.c[-1], stop_mode="level", stop_level=fl - tick_for(fl))

    def exit_signal(self, ctx, sym, b, ps):
        lo = _last(donchian(b.h, b.l, self.params["exit_n"])["lower"])
        return "close_below_20d_low" if b.c[-1] < lo else None


# ===================================================================== S061
@register
class S061(SwingBase):
    id = "S061"; name = NAMES["S061"]; asset = "STK"; sizing = "SWING_RISK"
    universe = (); universe_name = SP500_PIT
    params = {"contractions": 3, "proximity": 0.05, "tr_n": 10, "high_n": 126, "breakout_n": 20, "stop_n": 10,
              "trail_atr": 3.0, "atr_n": 14}
    grid = {"contractions": [2, 3, 4], "proximity": [0.03, 0.05, 0.10]}
    requires = [Requirement("bars", "1d", symbols=("SPY",), critical=False,
                            note="universe members resolved PIT through portal.universe('SP500')")]
    spec = SPECS["S061"]
    deviations = COMMON_DEVIATIONS + [
        "'contractions' = number of successive declining non-overlapping 10-day mean true ranges "
        "(indicators.volatility_contraction n_means; 3 = sheet baseline).",
        "'C within 5% of 126d high' = C_t >= (1-0.05)*max(H over the 126 sessions ending at t); breakout C_t > prior "
        "20d high; entry next open.",
        "Initial stop = min(L over the prior 10 sessions) (resting); 3ATR (Wilder ATR14) trailing stop on closes; the "
        "effective resting stop is the higher of the two.",
    ]

    def entry(self, ctx, sym, b):
        p = self.params
        vcp = np.asarray(volatility_contraction(b.h, b.l, b.c, p["tr_n"], p["contractions"]))
        if not bool(vcp[-1]):
            return None
        hi = float(np.max(b.h[-p["high_n"]:])) if len(b) >= p["high_n"] else float("nan")
        up = _last(donchian(b.h, b.l, p["breakout_n"])["upper"])
        if not finite(hi, up) or not (b.c[-1] >= (1 - p["proximity"]) * hi and b.c[-1] > up):
            return None
        lo = _last(donchian(b.h, b.l, p["stop_n"])["lower"])
        a = _atr_last(b, p["atr_n"])
        if not finite(lo, a):
            return None
        return Plan(1, b.c[-1], stop_mode="level", stop_level=lo, trail_k=p["trail_atr"], trail_atr_n=p["atr_n"],
                    atr=a)


# ===================================================================== S062
@register
class S062(SwingBase):
    id = "S062"; name = NAMES["S062"]; asset = "STK/ETF"; sizing = "SWING_RISK"; universe = SWING_ETF
    params = {"tenkan": 9, "kijun": 26, "senkou_b": 52, "displacement": 26, "cloud_filter": True, "stop_atr": 2.0,
              "atr_n": 14}
    grid = {"tenkan": [9, 12], "kijun": [26, 30], "cloud_filter": [True, False]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S062"]
    deviations = COMMON_DEVIATIONS + [
        "Cloud is information-time aligned (indicators.ichimoku): the cloud plotted at t = Senkou spans computed at "
        "t-26; filter C_t > cloud_top_t. Senkou B 52 and displacement 26 are kept when kijun varies.",
        "2ATR stop = Wilder ATR14, fixed from the fill; reverse Tenkan/Kijun cross -> exit next open.",
    ]

    def _ich(self, b):
        p = self.params
        return ichimoku(b.h, b.l, b.c, p["tenkan"], p["kijun"], p["senkou_b"], p["displacement"])

    def entry(self, ctx, sym, b):
        p = self.params
        ic = self._ich(b)
        tk, kj = np.asarray(ic["tenkan"]), np.asarray(ic["kijun"])
        if len(b) < 2 or not crossed_above(tk[-2], tk[-1], kj[-2], kj[-1]):
            return None
        if p["cloud_filter"]:
            top = _last(ic["cloud_top"])
            if not (finite(top) and b.c[-1] > top):
                return None
        a = _atr_last(b, p["atr_n"])
        return Plan(1, b.c[-1], stop_mode="atr_fill", stop_k=p["stop_atr"], atr=a) if finite(a) else None

    def exit_signal(self, ctx, sym, b, ps):
        ic = self._ich(b)
        tk, kj = np.asarray(ic["tenkan"]), np.asarray(ic["kijun"])
        return "tk_cross_down" if crossed_below(tk[-2], tk[-1], kj[-2], kj[-1]) else None


# ===================================================================== S063
@register
class S063(SwingBase):
    id = "S063"; name = NAMES["S063"]; asset = "STK/ETF"; sizing = "SWING_RISK"; universe = SWING_ETF
    params = {"af0": 0.02, "af_max": 0.2, "trend_n": 200}
    grid = {"af0": [0.01, 0.02, 0.03], "af_max": [0.1, 0.2, 0.3]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S063"]
    deviations = COMMON_DEVIATIONS + [
        "Acceleration step = initial AF (Wilder); SAR on the trailing window (state converges after the first flip).",
        "SAR stop: resting stop at next_sar (the SAR for the next bar, known at the close), re-placed every close; "
        "a bearish flip at a close that the resting stop did not catch exits at the next open.",
    ]

    def _sar(self, b):
        p = self.params
        return parabolic_sar(b.h, b.l, p["af0"], p["af0"], p["af_max"])

    def entry(self, ctx, sym, b):
        s = self._sar(b)
        if _last(s["flip"]) != 1 or not _sma_up(b.c, self.params["trend_n"]):
            return None
        nxt = _last(s["next_sar"])
        if not finite(nxt) or nxt >= b.c[-1]:
            return None
        return Plan(1, b.c[-1], stop_mode="level", stop_level=nxt)

    def update_levels(self, ctx, sym, b, ps):
        nxt = _last(self._sar(b)["next_sar"])
        if finite(nxt):
            ps.stop = nxt

    def exit_signal(self, ctx, sym, b, ps):
        return "sar_flip" if _last(self._sar(b)["trend"]) == -1 else None


# ===================================================================== S064
@register
class S064(SwingBase):
    id = "S064"; name = NAMES["S064"]; asset = "STK/ETF"; sizing = "SWING_RISK"; universe = SWING_ETF
    params = {"atr_n": 10, "mult": 3.0}
    grid = {"atr_n": [7, 10, 14], "mult": [2.0, 3.0, 4.0]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S064"]
    deviations = COMMON_DEVIATIONS + [
        "indicators.supertrend (TradingView logic, Wilder ATR). The lower band at close t is the resting stop for "
        "session t+1 only (re-placed every close); a bearish flip at the close exits at the next open.",
    ]

    def _st(self, b):
        return supertrend(b.h, b.l, b.c, self.params["atr_n"], self.params["mult"])

    def entry(self, ctx, sym, b):
        st = self._st(b)
        if _last(st["flip"]) != 1:
            return None
        band = _last(st["supertrend"])
        if not finite(band) or band >= b.c[-1]:
            return None
        return Plan(1, b.c[-1], stop_mode="level", stop_level=band)

    def update_levels(self, ctx, sym, b, ps):
        st = self._st(b)
        if _last(st["direction"]) == 1:
            ps.stop = _last(st["supertrend"])

    def exit_signal(self, ctx, sym, b, ps):
        return "supertrend_flip" if _last(self._st(b)["direction"]) == -1 else None


# ===================================================================== S065
@register
class S065(SwingBase):
    id = "S065"; name = NAMES["S065"]; asset = "STK/ETF"; sizing = "SWING_RISK"; universe = SWING_ETF
    params = {"vol_pct": 0.80, "entry_n": 20, "rv_n": 20, "rv_window": 252, "exit_n": 10, "trail_atr": 2.0,
              "atr_n": 20}
    grid = {"vol_pct": [0.60, 0.80, 1.00], "entry_n": [20, 55]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S065"]
    deviations = COMMON_DEVIATIONS + [
        "Realized vol = sample stdev of the 20 daily returns ending at t; its percentile = mid-rank among the 252 "
        "values ending at t (indicators.percentile_rank, include_current); entry requires pct < vol_pct; "
        "vol_pct = 1.00 disables the filter.",
        "Exit: close < prior 10d low -> next open, or 2*ATR20 (Wilder) trailing resting stop, whichever first.",
    ]

    def window(self):
        return 450

    def entry(self, ctx, sym, b):
        p = self.params
        up = _last(donchian(b.h, b.l, p["entry_n"])["upper"])
        if not finite(up) or not b.c[-1] > up:
            return None
        if p["vol_pct"] < 1.0:
            rv = realized_vol(b.c, p["rv_n"], annualize=None)
            pct = _last(percentile_rank(rv, p["rv_window"], include_current=True))
            if not (finite(pct) and pct < p["vol_pct"]):
                return None
        a = _atr_last(b, p["atr_n"])
        return Plan(1, b.c[-1], stop_mode="atr_fill", stop_k=p["trail_atr"], atr=a, trail_k=p["trail_atr"],
                    trail_atr_n=p["atr_n"]) if finite(a) else None

    def exit_signal(self, ctx, sym, b, ps):
        lo = _last(donchian(b.h, b.l, self.params["exit_n"])["lower"])
        return "channel_exit" if b.c[-1] < lo else None


# ===================================================================== S066
@register
class S066(SwingBase):
    id = "S066"; name = NAMES["S066"]; asset = "STK/ETF"; sizing = "SWING_RISK"; universe = SWING_ETF
    params = {"rsi_max": 60, "hold": 3, "target_r": 2.0, "trend_n": 200, "rsi_n": 14}
    grid = {"rsi_max": [50, 60], "hold": [3, 5, 10]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S066"]
    deviations = COMMON_DEVIATIONS + [
        "Engulfing per indicators.bullish_engulfing; RSI14 = Wilder. Stop = min(signal low, prior low), resting; "
        "2R target from the actual fill; time exit after 'hold' sessions.",
    ]

    def window(self):
        return 400

    def entry(self, ctx, sym, b):
        p = self.params
        if len(b) < 2 or not bool(np.asarray(bullish_engulfing(b.o, b.h, b.l, b.c))[-1]):
            return None
        t, r = _last(sma(b.c, p["trend_n"])), _last(rsi(b.c, p["rsi_n"]))
        if not (finite(t, r) and b.c[-1] > t and r < p["rsi_max"]):
            return None
        stop = float(min(b.l[-1], b.l[-2]))
        return Plan(1, b.c[-1], stop_mode="level", stop_level=stop, target_r=p["target_r"], max_hold=p["hold"])


# ===================================================================== S067
def hs_neckline_bars(h, l, lag: int, ls: float, hd: float, rs: float, t: int) -> Optional[tuple[int, int]]:
    """Reaction-low bars (b1, b2) of the head-and-shoulders pattern reported by
    ``indicators.head_and_shoulders`` at bar ``t`` (whose shoulder/head highs are
    ls/hd/rs). Uses only pivots confirmed at or before t (causal)."""
    pv = pivots(h[: t + 1], l[: t + 1], lag, lag)
    bars = [int(x) for x in np.asarray(pv["ph_bar"])[np.asarray(pv["ph"], bool)]]
    for k in range(len(bars) - 1, 1, -1):
        s1, s2, s3 = bars[k - 2], bars[k - 1], bars[k]
        if h[s1] == ls and h[s2] == hd and h[s3] == rs and s2 > s1 + 1 and s3 > s2 + 1:
            b1 = s1 + 1 + int(np.nanargmin(l[s1 + 1:s2]))
            b2 = s2 + 1 + int(np.nanargmin(l[s2 + 1:s3]))
            return b1, b2
    return None


@register
class S067(SwingBase):
    id = "S067"; name = NAMES["S067"]; asset = "STK/ETF"; sizing = "SWING_RISK"; universe = SWING_ETF
    params = {"lag": 3, "sym_atr": 1.0, "head_atr": 1.0, "atr_n": 14, "target_r": 3.0}
    grid = {"lag": [3, 5], "sym_atr": [1.0, 2.0]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S067"]
    deviations = COMMON_DEVIATIONS + [
        "indicators.head_and_shoulders: three consecutive confirmed pivot highs (left=right=lag, known only lag bars "
        "after the pivot), head - max(shoulders) >= 1*ATR14, |LS-RS| <= sym_atr*ATR14 (ATR at RS confirmation), "
        "neckline through the two reaction lows extrapolated, short on a fresh close below it -> next open.",
        "Stop = right-shoulder high (resting buy stop); cover after a close above the extrapolated neckline -> next "
        "open, or 3R target (limit) from the actual fill.",
        "Short only (pattern is a top); locate/borrow and SSR handled by the engine.",
    ]

    def window(self):
        return 400

    def entry(self, ctx, sym, b):
        p = self.params
        hs = head_and_shoulders(b.h, b.l, b.c, p["lag"], p["atr_n"], p["head_atr"], p["sym_atr"])
        if not bool(np.asarray(hs["signal"])[-1]):
            return None
        rs, hd, ls = _last(hs["rs_high"]), _last(hs["head_high"]), _last(hs["ls_high"])
        t = len(b) - 1
        nb = hs_neckline_bars(b.h, b.l, p["lag"], ls, hd, rs, t)
        if nb is None:
            return None
        return Plan(-1, b.c[-1], stop_mode="level", stop_level=rs, target_r=p["target_r"],
                    data={"b1": int(b.start[nb[0]]), "b2": int(b.start[nb[1]])})

    def exit_signal(self, ctx, sym, b, ps):
        d = ps.plan.data
        i1 = int(np.searchsorted(b.start, d["b1"]))
        i2 = int(np.searchsorted(b.start, d["b2"]))
        if i1 >= len(b) or i2 >= len(b) or b.start[i1] != d["b1"] or b.start[i2] != d["b2"]:
            return None
        slope = (b.l[i2] - b.l[i1]) / (i2 - i1)
        neck = b.l[i1] + slope * (len(b) - 1 - i1)
        return "close_above_neckline" if b.c[-1] > neck else None
