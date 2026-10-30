"""Swing mean-reversion strategies S068-S084 (daily bars).

Shared conventions: ``swing_common.COMMON_DEVIATIONS``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..core.types import NotTestableError, OrderStatus, OrderType, TimeInForce
from ..engine.strategy import Requirement, Strategy
from ..indicators import (atr, bollinger, cci, donchian, down_streak, ema, ibs, percentile_rank, rolling_ols, rsi,
                          sma, stochastic, williams_r, zscore)
from ..indicators import cross_section as xs
from ..risk import sizing as _sz
from .registry import register
from .swing_common import (COMMON_DEVIATIONS, SECTOR_CAVEAT, SECTOR_ETF, SECTOR_SPDR, SP500_PIT, SWING_ETF, Bars, BarCache,
                           CrossSectionBase, Plan, SwingBase, crossed_above, crossed_below, finite, sector_etf_for,
                           tick_for)
from .swing_specs import NAMES, SPECS


def _last(x, k: int = 1) -> float:
    a = np.asarray(x, dtype=float)
    return float(a[-k]) if a.shape[0] >= k else float("nan")


def _sma_up(c, n: int = 200, lag: int = 20) -> bool:
    s = np.asarray(sma(c, n))
    return s.shape[0] > lag and finite(s[-1], s[-1 - lag]) and s[-1] > s[-1 - lag]


def _above_sma(c, n: int = 200) -> bool:
    s = _last(sma(c, n))
    return finite(s) and c[-1] > s


def _raw_dollar_volume(ctx, sym: str, n: int = 20) -> float:
    raw = ctx.history(sym, n, "1d", adjusted=False)
    if raw is None or len(raw) < n:
        return float("nan")
    return float((raw["close"] * raw["volume"]).mean())


STK_REQ = [Requirement("bars", "1d", symbols=("SPY",), critical=False,
                       note="universe members resolved PIT through portal.universe('SP500')")]


# ===================================================================== S068
@register
class S068(SwingBase):
    id = "S068"; name = NAMES["S068"]; asset = "STK/ETF"; sizing = "SWING_NOTIONAL"; universe = SWING_ETF
    params = {"rsi_n": 2, "rsi_max": 5, "trend_n": 200, "exit_sma": 5, "max_hold": None}
    grid = {"rsi_max": [2, 5, 10], "exit_sma": [3, 5, 10], "max_hold": [None, 10]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S068"]
    deviations = COMMON_DEVIATIONS + [
        "Grid 'RSI 2/5/10' = the RSI2 entry threshold (RSI2 < 2/5/10; the strategy is defined on Wilder RSI(2), "
        "rsi_n = 2 fixed).",
        "No protective stop (published baseline); notional capped by SWING_NOTIONAL (10% NAV); overnight gaps are "
        "modelled because every fill is at a real next-session open.",
    ]

    def entry(self, ctx, sym, b):
        p = self.params
        r = _last(rsi(b.c, p["rsi_n"]))
        if not (_above_sma(b.c, p["trend_n"]) and finite(r) and r < p["rsi_max"]):
            return None
        return Plan(1, b.c[-1], max_hold=p["max_hold"])

    def exit_signal(self, ctx, sym, b, ps):
        s = _last(sma(b.c, self.params["exit_sma"]))
        return "close_above_sma" if finite(s) and b.c[-1] > s else None


# ===================================================================== S069
@register
class S069(SwingBase):
    id = "S069"; name = NAMES["S069"]; asset = "STK/ETF"; sizing = "SWING_NOTIONAL"; universe = SWING_ETF
    params = {"oversold": 30, "exit_rsi": 55, "rsi_n": 14, "trend_n": 200, "max_hold": 15}
    grid = {"oversold": [20, 30, 40], "exit_rsi": [50, 55, 60]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S069"]
    deviations = COMMON_DEVIATIONS + ["RSI14 = Wilder; 'crosses above 30 from below' = RSI_{t-1} <= 30 < RSI_t."]

    def entry(self, ctx, sym, b):
        p = self.params
        r = np.asarray(rsi(b.c, p["rsi_n"]))
        L = p["oversold"]
        if not (_above_sma(b.c, p["trend_n"]) and crossed_above(r[-2], r[-1], L, L)):
            return None
        return Plan(1, b.c[-1], max_hold=p["max_hold"])

    def exit_signal(self, ctx, sym, b, ps):
        p = self.params
        r = _last(rsi(b.c, p["rsi_n"]))
        if finite(r) and r >= p["exit_rsi"]:
            return "rsi_exit"
        s = _last(sma(b.c, p["trend_n"]))
        return "close_below_sma200" if finite(s) and b.c[-1] < s else None


# ===================================================================== S070
@register
class S070(SwingBase):
    id = "S070"; name = NAMES["S070"]; asset = "STK/ETF"; sizing = "SWING_RISK"; universe = SWING_ETF
    params = {"bb_n": 20, "bb_k": 2.0, "trend_n": 200, "stop_atr": 3.0, "atr_n": 14, "max_hold": 15}
    grid = {"bb_n": [20, 30], "bb_k": [1.5, 2.0, 2.5]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S070"]
    deviations = COMMON_DEVIATIONS + [
        "'today close back inside band' = L_t <= C_t <= U_t after C_{t-1} < L_{t-1}; the exit 'close>=SMA20' uses the "
        "band middle SMA(bb_n) (SMA30 in the BB30 variant).",
        "Disaster stop 3*ATR14 (Wilder) from the fill, resting; also the SWING_RISK sizing stop.",
    ]

    def entry(self, ctx, sym, b):
        p = self.params
        bb = bollinger(b.c, p["bb_n"], p["bb_k"])
        lo, up = np.asarray(bb["lower"]), np.asarray(bb["upper"])
        if len(b) < 2 or not finite(lo[-2], lo[-1], up[-1]):
            return None
        if not (b.c[-2] < lo[-2] and lo[-1] <= b.c[-1] <= up[-1] and _sma_up(b.c, p["trend_n"])):
            return None
        a = _last(atr(b.h, b.l, b.c, p["atr_n"]))
        return Plan(1, b.c[-1], stop_mode="atr_fill", stop_k=p["stop_atr"], atr=a,
                    max_hold=p["max_hold"]) if finite(a) else None

    def exit_signal(self, ctx, sym, b, ps):
        m = _last(sma(b.c, self.params["bb_n"]))
        return "close_at_mid" if finite(m) and b.c[-1] >= m else None


# ===================================================================== S071
@register
class S071(SwingBase):
    id = "S071"; name = NAMES["S071"]; asset = "ETF"; sizing = "SWING_NOTIONAL"; universe = SWING_ETF
    params = {"bb_k": 2.0, "hold": 10, "bb_n": 20, "trend_n": 200, "stop": False, "stop_atr": 3.0, "atr_n": 14}
    grid = {"bb_k": [1.5, 2.0, 2.5], "hold": [5, 10, 15]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S071"]
    deviations = COMMON_DEVIATIONS + [
        "'optional stop 3ATR' is param stop (default False); when on, 3*ATR14 from the fill, resting.",
    ]

    def entry(self, ctx, sym, b):
        p = self.params
        lo = _last(bollinger(b.c, p["bb_n"], p["bb_k"])["lower"])
        if not (finite(lo) and b.c[-1] < lo and _above_sma(b.c, p["trend_n"])):
            return None
        if p["stop"]:
            a = _last(atr(b.h, b.l, b.c, p["atr_n"]))
            if not finite(a):
                return None
            return Plan(1, b.c[-1], stop_mode="atr_fill", stop_k=p["stop_atr"], atr=a, max_hold=p["hold"])
        return Plan(1, b.c[-1], max_hold=p["hold"])

    def exit_signal(self, ctx, sym, b, ps):
        m = _last(sma(b.c, self.params["bb_n"]))
        return "close_at_sma20" if finite(m) and b.c[-1] >= m else None


# ===================================================================== S072
@register
class S072(SwingBase):
    id = "S072"; name = NAMES["S072"]; asset = "ETF"; sizing = "SWING_NOTIONAL"; universe = SWING_ETF
    params = {"z_entry": -2.0, "max_hold": 10, "z_n": 20, "z_stop": -3.0, "trend_n": 200}
    grid = {"z_entry": [-1.5, -2.0, -2.5], "max_hold": [5, 10]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S072"]
    deviations = COMMON_DEVIATIONS + [
        "z = (C - SMA20)/population std20 (indicators.zscore); 'stop z<-3 by next-open fill' = a close with z < -3 "
        "after entry -> exit at the next open.",
    ]

    def entry(self, ctx, sym, b):
        p = self.params
        z = _last(zscore(b.c, p["z_n"]))
        if not (finite(z) and z < p["z_entry"] and _sma_up(b.c, p["trend_n"])):
            return None
        return Plan(1, b.c[-1], max_hold=p["max_hold"])

    def exit_signal(self, ctx, sym, b, ps):
        p = self.params
        z = _last(zscore(b.c, p["z_n"]))
        if not finite(z):
            return None
        if z >= 0:
            return "z_revert"
        if z < p["z_stop"]:
            return "z_stop"
        return None


# ===================================================================== S073
@register
class S073(SwingBase):
    id = "S073"; name = NAMES["S073"]; asset = "ETF"; sizing = "SWING_NOTIONAL"; universe = SWING_ETF
    params = {"n_down": 3, "exit_sma": 5, "max_hold": 5, "trend_n": 200}
    grid = {"n_down": [2, 3, 4, 5], "exit_sma": [3, 5]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S073"]
    deviations = COMMON_DEVIATIONS + [
        "C_t<C_{t-1}<C_{t-2}<C_{t-3} = at least n_down consecutive strictly lower closes ending at t "
        "(indicators.down_streak >= n_down).",
    ]

    def entry(self, ctx, sym, b):
        p = self.params
        if not (_last(down_streak(b.c)) >= p["n_down"] and _above_sma(b.c, p["trend_n"])):
            return None
        return Plan(1, b.c[-1], max_hold=p["max_hold"])

    def exit_signal(self, ctx, sym, b, ps):
        s = _last(sma(b.c, self.params["exit_sma"]))
        return "close_above_sma" if finite(s) and b.c[-1] > s else None


# ===================================================================== S074
@register
class S074(CrossSectionBase):
    id = "S074"; name = NAMES["S074"]
    params = {"formation": 5, "decile": 0.10, "sector_neutral": True, "no_sector_caps": False}
    grid = {"formation": [3, 5, 10], "decile": [0.10, 0.20], "sector_neutral": [True, False]}
    spec = SPECS["S074"]
    deviations = COMMON_DEVIATIONS[1:2] + [
        "Universe: point-in-time portal universe 'SP500'.",
        "Signal at the last NYSE session of each week ('Friday close'; Thursday when Friday is a holiday); formation "
        "return = adjusted (total-return) close over the last 3/5/10 sessions; orders fill at the next session open "
        "('Monday open'); positions are held to the next weekly rebalance and exit/re-weight at that open (the "
        "sheet's 'next Mon open' proxy, applied uniformly).",
        "Sector-neutral: deciles (quintiles for 20%) formed within each security-master sector "
        "(indicators.cross_section.group_quantile_bucket): long the lowest bucket, short the highest bucket of every "
        "sector. Without a sector column the run is NOT_TESTABLE (no proxy). Symmetric legs: in each sector with "
        "n >= q names the lowest and highest floor(n/q) names are taken (ties by symbol), so both legs hold the "
        "same number of names; sectors with fewer than q names are not traded (the non-neutral variant ranks the "
        "whole universe the same way).",
        "CROSS_SECTION weights via risk.sizing.cross_section_weights (+50%/-50%, 2% name cap, 10% per-side sector cap).",
        SECTOR_CAVEAT,
    ]

    def window(self):
        return 30

    def is_signal_day(self, ctx):
        cal = ctx.calendar()
        d = ctx.session_date()
        nxt = cal.next_session(d)
        return nxt.isocalendar()[:2] != d.isocalendar()[:2]

    def targets(self, ctx, members):
        p = self.params
        f = int(p["formation"])
        r = {}
        for s in members:
            b = self._cache.get(ctx, s, f + 1)
            if b is None or len(b) < f + 1 or b.end[-1] != int(ctx.now.value):
                continue
            v = b.c[-1] / b.c[-1 - f] - 1.0
            if np.isfinite(v):
                r[s] = v
        if len(r) < 10:
            return None
        x = pd.Series(r)
        q = int(round(1.0 / p["decile"]))
        if p["sector_neutral"]:
            self.need_sectors = True
            sec = self.sectors(ctx, list(x.index))
            x = x[[s for s in x.index if s in sec]]
            bk = xs.group_quantile_bucket(x, pd.Series(sec).reindex(x.index), q, min_group_size=q, symmetric=True)
        else:
            sec = self.sectors(ctx, list(x.index))
            bk = xs.group_quantile_bucket(x, pd.Series("ALL", index=x.index), q, symmetric=True)
        longs = sorted(s for s in x.index if bk[s] == 1)
        shorts = sorted(s for s in x.index if bk[s] == q)
        return _sz.cross_section_weights(longs, shorts, sec or None)


# ===================================================================== S075
@register
class S075(SwingBase):
    id = "S075"; name = NAMES["S075"]; asset = "ETF"; sizing = "SWING_NOTIONAL"; universe = SWING_ETF
    params = {"ibs_entry": 0.2, "ibs_exit": 0.8, "exit_sma": 5, "max_hold": 5, "trend_n": 200}
    grid = {"ibs_entry": [0.1, 0.2, 0.3], "ibs_exit": [0.7, 0.8, 0.9]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S075"]
    deviations = COMMON_DEVIATIONS + ["IBS undefined (H=L) days are skipped (no entry, no IBS exit)."]

    def entry(self, ctx, sym, b):
        p = self.params
        v = _last(ibs(b.h, b.l, b.c))
        if not (finite(v) and v < p["ibs_entry"] and _above_sma(b.c, p["trend_n"])):
            return None
        return Plan(1, b.c[-1], max_hold=p["max_hold"])

    def exit_signal(self, ctx, sym, b, ps):
        p = self.params
        v = _last(ibs(b.h, b.l, b.c))
        if finite(v) and v > p["ibs_exit"]:
            return "ibs_exit"
        s = _last(sma(b.c, p["exit_sma"]))
        return "close_above_sma" if finite(s) and b.c[-1] > s else None


# ===================================================================== S076
@register
class S076(SwingBase):
    id = "S076"; name = NAMES["S076"]; asset = "ETF"; sizing = "SWING_NOTIONAL"; universe = SWING_ETF
    params = {"wr_n": 14, "threshold": -90, "rearm": -80, "exit_wr": -20, "max_hold": 10, "trend_n": 200}
    grid = {"wr_n": [14, 21], "threshold": [-80, -90]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S076"]
    deviations = COMMON_DEVIATIONS + [
        "Setup/trigger sequence: %R crosses back above -80 at t (%R_{t-1} <= -80 < %R_t) and, during the run of bars "
        "with %R <= -80 that ended at t-1, at least one bar had %R <= threshold with C > SMA200; C_t > SMA200 at the "
        "trigger close; buy next open.",
    ]

    def entry(self, ctx, sym, b):
        p = self.params
        wr = np.asarray(williams_r(b.h, b.l, b.c, p["wr_n"]))
        s200 = np.asarray(sma(b.c, p["trend_n"]))
        if len(b) < 3 or not crossed_above(wr[-2], wr[-1], p["rearm"], p["rearm"]) or not b.c[-1] > s200[-1]:
            return None
        j = len(b) - 2
        while j >= 0 and np.isfinite(wr[j]) and wr[j] <= p["rearm"]:
            if wr[j] <= p["threshold"] and np.isfinite(s200[j]) and b.c[j] > s200[j]:
                return Plan(1, b.c[-1], max_hold=p["max_hold"])
            j -= 1
        return None

    def exit_signal(self, ctx, sym, b, ps):
        w = _last(williams_r(b.h, b.l, b.c, self.params["wr_n"]))
        return "wr_exit" if finite(w) and w > self.params["exit_wr"] else None


# ===================================================================== S077
@register
class S077(SwingBase):
    id = "S077"; name = NAMES["S077"]; asset = "STK/ETF"; sizing = "SWING_NOTIONAL"; universe = SWING_ETF
    params = {"k": 14, "oversold": 20, "smooth": 3, "d": 3, "overbought": 80, "max_hold": 15, "trend_n": 200}
    grid = {"k": [10, 14, 21], "oversold": [15, 20, 25]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S077"]
    deviations = COMMON_DEVIATIONS + [
        "Slow stochastic: %K = SMA3 of raw %K14, %D = SMA3 of %K (indicators.stochastic slow_k/slow_d); both < 20 "
        "at the cross close.",
    ]

    def _st(self, b):
        p = self.params
        st = stochastic(b.h, b.l, b.c, p["k"], p["smooth"], p["d"])
        return np.asarray(st["slow_k"]), np.asarray(st["slow_d"])

    def entry(self, ctx, sym, b):
        p = self.params
        k, d = self._st(b)
        if len(b) < 2 or not crossed_above(k[-2], k[-1], d[-2], d[-1]):
            return None
        if not (k[-1] < p["oversold"] and d[-1] < p["oversold"] and _above_sma(b.c, p["trend_n"])):
            return None
        return Plan(1, b.c[-1], max_hold=p["max_hold"])

    def exit_signal(self, ctx, sym, b, ps):
        k, d = self._st(b)
        if finite(k[-1]) and k[-1] > self.params["overbought"]:
            return "k_overbought"
        return "k_cross_down" if crossed_below(k[-2], k[-1], d[-2], d[-1]) else None


# ===================================================================== S078
@register
class S078(SwingBase):
    id = "S078"; name = NAMES["S078"]; asset = "ETF"; sizing = "SWING_NOTIONAL"; universe = SWING_ETF
    params = {"cci_n": 20, "threshold": -100, "exit_cci": 100, "max_hold": 10, "trend_n": 200}
    grid = {"cci_n": [14, 20, 30], "threshold": [-100, -150]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S078"]
    deviations = COMMON_DEVIATIONS + ["CCI = Lambert, constant 0.015, mean absolute deviation (indicators.cci)."]

    def entry(self, ctx, sym, b):
        p = self.params
        c = np.asarray(cci(b.h, b.l, b.c, p["cci_n"]))
        L = p["threshold"]
        if len(b) < 2 or not crossed_above(c[-2], c[-1], L, L) or not _above_sma(b.c, p["trend_n"]):
            return None
        return Plan(1, b.c[-1], max_hold=p["max_hold"])

    def exit_signal(self, ctx, sym, b, ps):
        c = _last(cci(b.h, b.l, b.c, self.params["cci_n"]))
        return "cci_exit" if finite(c) and c > self.params["exit_cci"] else None


# ===================================================================== S079
@register
class S079(SwingBase):
    id = "S079"; name = NAMES["S079"]; asset = "ETF"; sizing = "SWING_RISK"; universe = SWING_ETF
    params = {"drop_atr": 2.0, "vol_mult": 1.5, "atr_n": 14, "vol_n": 20, "exit_sma": 5, "max_hold": 7,
              "stop_atr": 3.0, "trend_n": 200}
    grid = {"drop_atr": [1.5, 2.0, 2.5], "vol_mult": [1.0, 1.5, 2.0]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S079"]
    deviations = COMMON_DEVIATIONS + [
        "Drop test uses ATR14 as of t-1 (the volatility before the selloff day): C_t <= C_{t-1} - k*ATR14_{t-1}.",
        "'20d average' volume = mean of the 20 sessions before t (t excluded).",
        "Stop 3*ATR14 (at the signal close) from the fill, resting.",
    ]

    def entry(self, ctx, sym, b):
        p = self.params
        a = np.asarray(atr(b.h, b.l, b.c, p["atr_n"]))
        if len(b) < p["vol_n"] + 2 or not finite(a[-2], a[-1]):
            return None
        if not b.c[-1] <= b.c[-2] - p["drop_atr"] * a[-2]:
            return None
        av = float(np.mean(b.v[-1 - p["vol_n"]:-1]))
        if not (np.isfinite(av) and b.v[-1] >= p["vol_mult"] * av and _sma_up(b.c, p["trend_n"])):
            return None
        return Plan(1, b.c[-1], stop_mode="atr_fill", stop_k=p["stop_atr"], atr=float(a[-1]), max_hold=p["max_hold"])

    def exit_signal(self, ctx, sym, b, ps):
        s = _last(sma(b.c, self.params["exit_sma"]))
        return "close_at_sma5" if finite(s) and b.c[-1] >= s else None


# ===================================================================== S080
@register
class S080(SwingBase):
    id = "S080"; name = NAMES["S080"]; asset = "STK/ETF"; sizing = "SWING_NOTIONAL"; universe = SWING_ETF
    params = {"near": 0.01, "rsi_max": 40, "rsi_n": 14, "trend_n": 200, "exit_sma": 20, "max_hold": 20,
              "stop_frac": 0.97}
    grid = {"near": [0.01, 0.02, 0.03], "rsi_max": [30, 40]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S080"]
    deviations = COMMON_DEVIATIONS + [
        "'within 1% of SMA200 from above' = SMA200 <= C_t <= (1+near)*SMA200; 'Exit at SMA20' = close >= SMA20 -> "
        "next open; stop 'C<0.97*SMA200' is close-based -> next open.",
    ]

    def entry(self, ctx, sym, b):
        p = self.params
        s = _last(sma(b.c, p["trend_n"]))
        r = _last(rsi(b.c, p["rsi_n"]))
        if not (finite(s, r) and _sma_up(b.c, p["trend_n"]) and s <= b.c[-1] <= (1 + p["near"]) * s
                and r < p["rsi_max"]):
            return None
        return Plan(1, b.c[-1], max_hold=p["max_hold"])

    def exit_signal(self, ctx, sym, b, ps):
        p = self.params
        m = _last(sma(b.c, p["exit_sma"]))
        if finite(m) and b.c[-1] >= m:
            return "close_at_sma20"
        s = _last(sma(b.c, p["trend_n"]))
        return "stop_below_sma200" if finite(s) and b.c[-1] < p["stop_frac"] * s else None


# ===================================================================== S081
@register
class S081(SwingBase):
    id = "S081"; name = NAMES["S081"]; asset = "ETF"; sizing = "SWING_NOTIONAL"; universe = SWING_ETF
    params = {"kc_k": 2.0, "hold": 12, "n": 20, "trend_n": 200, "stop_k": 3.0}
    grid = {"kc_k": [1.5, 2.0, 2.5], "hold": [8, 12]}
    requires = [Requirement("bars", "1d")]
    spec = SPECS["S081"]
    deviations = COMMON_DEVIATIONS + [
        "ATR20 = Wilder; 'EMA200 rising' = EMA200_t > EMA200_{t-20} (rule 9 applied to the EMA).",
        "Stop 'EMA20 - 3*ATR20' is a moving price level: a resting stop at EMA20_t - 3*ATR20_t re-placed every close; "
        "no entry when the signal close is already at/below that stop level (it would be stopped out at the open).",
    ]

    def _kc(self, b):
        p = self.params
        e = np.asarray(ema(b.c, p["n"]))
        a = np.asarray(atr(b.h, b.l, b.c, p["n"]))
        return e, a

    def entry(self, ctx, sym, b):
        p = self.params
        e, a = self._kc(b)
        e200 = np.asarray(ema(b.c, p["trend_n"]))
        if len(b) < 22 or not finite(e[-1], a[-1], e200[-1], e200[-21]):
            return None
        if not (b.c[-1] < e[-1] - p["kc_k"] * a[-1] and e200[-1] > e200[-21]):
            return None
        if not b.c[-1] > e[-1] - p["stop_k"] * a[-1]:
            return None                     # already through the stop level: the trade is invalid at entry
        return Plan(1, b.c[-1], stop_mode="level", stop_level=float(e[-1] - p["stop_k"] * a[-1]), max_hold=p["hold"])

    def update_levels(self, ctx, sym, b, ps):
        e, a = self._kc(b)
        if finite(e[-1], a[-1]):
            ps.stop = float(e[-1] - self.params["stop_k"] * a[-1])

    def exit_signal(self, ctx, sym, b, ps):
        e = _last(ema(b.c, self.params["n"]))
        return "close_at_ema20" if finite(e) and b.c[-1] >= e else None


# ===================================================================== S082
@register
class S082(CrossSectionBase):
    id = "S082"; name = NAMES["S082"]
    params = {"resid_n": 5, "beta_n": 252, "hold": 5, "q": 10, "market": "SPY", "no_sector_caps": False}
    grid = {"resid_n": [3, 5, 10], "beta_n": [126, 252]}
    requires = [Requirement("bars", "1d", symbols=("SPY",)),
                Requirement("bars", "1d", symbols=tuple(e for e in SECTOR_SPDR if e not in ("XLRE", "XLC"))),
                Requirement("bars", "1d", symbols=("XLRE", "XLC"), critical=False)]
    spec = SPECS["S082"]
    need_sectors = True
    deviations = COMMON_DEVIATIONS[1:2] + [
        "Universe: point-in-time portal universe 'SP500'; sector from the security master mapped to its Select "
        "Sector SPDR, only from the ETF's inception (XLRE 2015-10-08, XLC 2018-06-19; names in those sectors are not "
        "ranked before). Without sector data the run raises NotTestableError.",
        "Daily simple returns of the stock regressed on SPY and its sector SPDR (+intercept) over beta_n sessions "
        "ending at t-resid_n (indicators.rolling_ols; betas formed before the residual window). Residual = "
        "sum over the last resid_n sessions of (r - b_mkt*r_SPY - b_sec*r_sector) (intercept excluded).",
        "Rebalance every 5 sessions (calendar session index divisible by 5): long the lowest residual decile, short "
        "the highest (symmetric: floor(n/10) names each side); held to the next rebalance. CROSS_SECTION weights with 10% per-side sector caps; borrow and "
        "locates by the engine.",
        SECTOR_CAVEAT,
    ]

    def window(self):
        return int(self.params["beta_n"]) + int(self.params["resid_n"]) + 5

    def is_signal_day(self, ctx):
        return ctx.calendar().session_index(ctx.session_date()) % int(self.params["hold"]) == 0

    def _rets(self, ctx, sym, n):
        b = self._cache.get(ctx, sym, n)
        if b is None or len(b) < 2 or b.end[-1] != int(ctx.now.value):
            return None
        c = pd.Series(b.c, index=b.start)
        return c.pct_change().iloc[1:]

    def targets(self, ctx, members):
        p = self.params
        bn, rn = int(p["beta_n"]), int(p["resid_n"])
        need = bn + rn + 1
        mk = self._rets(ctx, p["market"], need + 1)
        if mk is None:
            return None
        today = ctx.session_date()
        sec = self.sectors(ctx, members)
        etf_rets = {}
        vals = {}
        for s in members:
            etf = sector_etf_for(sec.get(s), today)
            if etf is None:
                continue
            if etf not in etf_rets:
                etf_rets[etf] = self._rets(ctx, etf, need + 1)
            er = etf_rets[etf]
            if er is None:
                continue
            y = self._rets(ctx, s, need + 1)
            if y is None:
                continue
            df = pd.concat({"y": y, "m": mk, "s": er}, axis=1, join="inner").dropna()
            if len(df) < bn + rn or df.index[-1] != y.index[-1]:
                continue
            df = df.iloc[-(bn + rn):]
            ols = rolling_ols(df["y"], df[["m", "s"]], bn)
            row = ols.iloc[bn - 1]                 # fit on the first bn rows = through t - resid_n
            bm, bs = float(row["beta_m"]), float(row["beta_s"])
            if not finite(bm, bs):
                continue
            tail = df.iloc[-rn:]
            vals[s] = float((tail["y"] - bm * tail["m"] - bs * tail["s"]).sum())
        if len(vals) < 10:
            return None
        x = pd.Series(vals)
        q = int(p["q"])
        bk = xs.group_quantile_bucket(x, pd.Series("ALL", index=x.index), q, symmetric=True)
        longs = sorted(s for s in x.index if bk[s] == 1)
        shorts = sorted(s for s in x.index if bk[s] == q)
        return _sz.cross_section_weights(longs, shorts, sec)


# ===================================================================== S083
@register
class S083(Strategy):
    id = "S083"; name = NAMES["S083"]; bar_freq = "1d"; asset = "ETF"; sizing = "SWING_NOTIONAL"
    universe = ["SPY"]
    params = {"vix_pct": 0.90, "rsi_max": 5, "rsi_n": 2, "vix_window": 252, "exit_sma": 5, "vix_median_n": 20,
              "max_hold": 10, "vix_symbol": "VIX", "symbol": "SPY"}
    grid = {"vix_pct": [0.80, 0.90, 0.95], "rsi_max": [2, 5, 10]}
    requires = [Requirement("bars", "1d", symbols=("SPY",)),
                Requirement("bars", "1d", symbols=("VIX",), note="Cboe official VIX close (known 16:15 ET)")]
    spec = SPECS["S083"]
    deviations = [
        "Instruments named by the sheet: signal on the Cboe VIX index (portal symbol 'VIX', official close "
        "disseminated 16:15 ET, i.e. known_at = close + 15 min) and SPY; trades SPY only.",
        "Timing: the official VIX close is only known at 16:15 ET, after the 16:00 daily step, so the decision is "
        "taken at the next session start from the last completed SPY (16:00) and VIX (16:15) closes and sent as a "
        "market order executing at that session's open (= 'buy SPY next open'). Exits are evaluated the same way and "
        "also execute at the open.",
        "'VIX above trailing 252d 90th percentile' = mid-rank of VIX_t among the previous 252 closes "
        "(indicators.percentile_rank, include_current=False) > vix_pct.",
        "SPY RSI2 = Wilder RSI(2) on adjusted closes; grid 'RSI2 2/5/10' = entry thresholds.",
        "'VIX below its 20d median' = VIX_t < median of the 20 closes ending at t.",
        "Holding period: entry session = 1; after 10 completed sessions the exit executes at the next open.",
        "SWING_NOTIONAL: 10% NAV long.",
    ]

    def init(self, ctx):
        self._entry_session = None
        self._pending = None

    def on_session_start(self, ctx):
        p = self.params
        sym, vsym = p["symbol"], p["vix_symbol"]
        cal = ctx.calendar()
        prev = cal.prev_session(ctx.session_date())
        spy = ctx.history(sym, 400, "1d", adjusted=True)
        # (index: no corporate actions, adjusted == raw; the engine's raw fast path only covers the preload
        #  window, so the PIT adjusted path is used to get the full 252-session lookback)
        vix = ctx.history(vsym, 400, "1d", adjusted=True)
        if spy is None or vix is None or len(spy) < 210 or len(vix) < p["vix_window"] + 1:
            return
        sess = lambda df: pd.Timestamp(df.index[-1]).tz_convert("America/New_York").tz_localize(None).normalize()
        if sess(spy) != prev or sess(vix) != prev:
            return                                   # stale input: no decision
        c = spy["close"].to_numpy(float)
        v = vix["close"].to_numpy(float)
        q = ctx.position(sym)
        if self._pending is not None:
            o = self._pending
            if o.status not in (OrderStatus.PENDING, OrderStatus.WORKING, OrderStatus.PARTIAL):
                self._pending = None
        if q > 0:
            if self._entry_session is None:
                self._entry_session = prev
            held = cal.session_index(prev) - cal.session_index(self._entry_session) + 1
            s5 = _last(sma(c, p["exit_sma"]))
            med = float(np.median(v[-p["vix_median_n"]:]))
            reason = None
            if finite(s5) and c[-1] > s5:
                reason = "spy_above_sma5"
            elif v[-1] < med:
                reason = "vix_below_median"
            elif held >= p["max_hold"]:
                reason = "time_exit"
            if reason and not any(o.tag == reason for o in ctx.open_orders(sym)):
                ctx.cancel_all(sym)
                ctx.order(sym, -q, OrderType.MARKET, tif=TimeInForce.DAY, tag=reason)
            return
        self._entry_session = None
        if self._pending is not None:
            return
        pr = _last(percentile_rank(v, p["vix_window"], include_current=False))
        r = _last(rsi(c, p["rsi_n"]))
        if not (finite(pr, r) and pr > p["vix_pct"] and r < p["rsi_max"]):
            return
        qty = _sz.size_swing_notional_ctx(ctx, sym, float(c[-1]), 1)
        if qty > 0:
            o = ctx.order(sym, qty, OrderType.MARKET, tif=TimeInForce.DAY, tag="entry")
            if o.status != OrderStatus.REJECTED:
                self._pending = o

    def on_fill(self, ctx, fill):
        if self._pending is not None and fill.order_id == self._pending.id:
            self._entry_session = ctx.session_date()


# ===================================================================== S084
@register
class S084(SwingBase):
    id = "S084"; name = NAMES["S084"]; asset = "STK"; sizing = "SWING_RISK"
    universe = (); universe_name = SP500_PIT
    params = {"near": 0.03, "hold": 15, "low_n": 252, "rsi_n": 14, "rsi_level": 30, "min_dollar_vol": 20e6,
              "dv_n": 20, "exit_sma": 20}
    grid = {"near": [0.02, 0.03, 0.05], "hold": [10, 15, 20]}
    requires = STK_REQ
    spec = SPECS["S084"]
    deviations = COMMON_DEVIATIONS + [
        "'C within 3% of prior 252d lowest low' = |C_t / min(L_{t-252..t-1}) - 1| <= near; RSI14 (Wilder) crosses "
        "above 30; ADV = mean raw close*volume over the 20 sessions ending at t >= $20m.",
        "'stop below 252d low' = resting stop 1 tick below the prior 252d low at the signal (fixed).",
        "SURVIVORSHIP: delisting-sensitive (sheet failure mode). The PIT 'SP500' universe includes later-delisted "
        "names only if the portal holds their bars and DELIST actions (delisting value applied by the engine; "
        "missing value -> last mark, T2). Results are biased upward wherever delisted members or delisting returns "
        "are missing from the data (run meta 'survivorship_bias').",
    ]

    def entry(self, ctx, sym, b):
        p = self.params
        ll = _last(donchian(b.h, b.l, p["low_n"])["lower"])
        r = np.asarray(rsi(b.c, p["rsi_n"]))
        if not finite(ll) or ll <= 0 or abs(b.c[-1] / ll - 1.0) > p["near"]:
            return None
        if not crossed_above(r[-2], r[-1], p["rsi_level"], p["rsi_level"]):
            return None
        if not _raw_dollar_volume(ctx, sym, p["dv_n"]) >= p["min_dollar_vol"]:
            return None
        stop = ll - tick_for(ll)
        if stop >= b.c[-1]:
            return None
        return Plan(1, b.c[-1], stop_mode="level", stop_level=stop, max_hold=p["hold"])

    def exit_signal(self, ctx, sym, b, ps):
        s = _last(sma(b.c, self.params["exit_sma"]))
        return "close_at_sma20" if finite(s) and b.c[-1] >= s else None
