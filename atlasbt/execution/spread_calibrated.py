"""Calibrated, point-in-time, era-aware half-spread model (fidelity T2 "spread modelled (calibrated)").

Why: the daily-OHLC estimators in :mod:`atlasbt.execution.spread` (Corwin-Schultz 2012, Abdi-Ranaldo
2017) are dominated by intraday volatility for liquid names. For SPY they give 18-50 bp full
spreads against a real NBBO of ~0.3-0.6 bp, so every ETF / large-cap backtest paid a cost about
30x too high. This module predicts the quoted (default) or effective half-spread from PIT
liquidity characteristics and a fit to real consolidated quotes.

Model (all inputs from sessions strictly before the trading session):

    latent full $ spread   L = P * exp(x'b) * era(date, ADV$)
        x = [1, la, ls, lp, E, E*la, E*ls, E*lp, E*max(la - la($100M), 0), min(lp - lp($5), 0)]
        la/ls/lp = centred logs of ADV$, daily sigma, price; E = ETF indicator (terms: calibration json)
    quoted full $ spread   S = k * (1 + (L/k + c_d)^q)^(1/q)          (tick discreteness)
    half spread (frac)     h = S / (2 P)  [x effective/quoted ratio if measure='effective']

era(t, ADV$) = exp(a(t) + b(t) * la), (a, b) interpolated between anchors (2001 / 2003 literature
anchors, 2010+ = 2026 fit), and k the minimum quoting increment of the era (1/8 before 1997-06-24 on NYSE, 1/16 until
decimalisation 2001-01-29 (NYSE/AMEX) / 2001-04-09 (Nasdaq), $0.01 after, $0.0001 below $1 from
Reg NMS Rule 612 (2005-08-29)). ``S >= k`` always, so the tick floor is part of the model, and the
smooth max reproduces the observed ~1.0x-1.05x tick spreads of tick-constrained names.

Calibration (``calibration/spread_calibration.json``, built by ``calibration/build_calibration.py``
from data documented in docs/SPREADS.md): b, c_d, q fitted by robust NLS to time-weighted NBBO
half-spreads of ~all US common stocks and ETFs on 2026-07-01 (NYSE Daily TAQ sample);
era anchors from published TAQ statistics (Hendershott-Jones-Menkveld 2011 Table I, Furfine 2003);
time-of-day profile from 5-minute NBBO buckets; effective/quoted ratio from TAQ trades vs the
prevailing NBBO. IEX TOPS samples 2017-2026 are diagnostics only (single venue).

Corwin-Schultz/Abdi-Ranaldo are used only as a CAP where the model extrapolates below the
calibration range of dollar volume (the log-linear fit is not trusted there).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from .spread import SpreadSeries, abdi_ranaldo_rolling, corwin_schultz_rolling

CALIBRATION_PATH = Path(__file__).with_name("calibration") / "spread_calibration.json"

#: tick-regime dates
NYSE_SIXTEENTHS = pd.Timestamp("1997-06-24")
NASDAQ_SIXTEENTHS = pd.Timestamp("1997-06-02")
NYSE_DECIMAL = pd.Timestamp("2001-01-29")      # NYSE + AMEX full decimalisation
NASDAQ_DECIMAL = pd.Timestamp("2001-04-09")
RULE_612 = pd.Timestamp("2005-08-29")           # Reg NMS sub-penny rule effective date

_ETF_CLASSES = {"ETF", "ETP", "ETN", "FUND"}
REASON_CALIBRATED = "calibrated spread model (T2)"


def era_tick(date, price: float, listing: str = "NYSE") -> float:
    """Minimum quoting increment ($) for a stock/ETF quoted at ``price`` on ``date``.

    NYSE (default; also used for AMEX-listed ETFs): 1/8 for >= $1 (1/16 below) until
    1997-06-23; 1/16 for >= $1 (1/32 below) until 2001-01-26; $0.01 from 2001-01-29.
    Nasdaq: 1/8 for >= $10 (1/32 below) until 1997-05-30, then 1/16 for >= $10; decimal from
    2001-04-09. Below $1: $0.01 until Reg NMS Rule 612 (2005-08-29), then $0.0001.
    The SEC Tick Size Pilot ($0.05 for ~1,200 small caps, 2016-10-03..2018-09-28) is not
    applied (membership unknown); pass ``tick=`` explicitly for pilot names."""
    d = pd.Timestamp(date)
    if d.tzinfo is not None:
        d = d.tz_convert("America/New_York").tz_localize(None)
    d = d.normalize()
    nas = str(listing).upper().startswith("NAS")
    if nas:
        if d < NASDAQ_SIXTEENTHS:
            return 0.125 if price >= 10 else 1 / 32
        if d < NASDAQ_DECIMAL:
            return 0.0625 if price >= 10 else 1 / 32
    else:
        if d < NYSE_SIXTEENTHS:
            return 0.125 if price >= 1 else 0.0625
        if d < NYSE_DECIMAL:
            return 0.0625 if price >= 1 else 1 / 32
    if price >= 1.0 or d < RULE_612:
        return 0.01
    return 0.0001


def _era_tick_vec(dates: np.ndarray, price: np.ndarray, listing: str) -> np.ndarray:
    """Vectorised :func:`era_tick` (dates: datetime64[ns], naive ET dates)."""
    d = np.asarray(dates, dtype="datetime64[ns]")
    p = np.asarray(price, float)
    nas = str(listing).upper().startswith("NAS")
    out = np.where(p >= 1.0, 0.01, np.where(d < np.datetime64(RULE_612), 0.01, 0.0001))
    if nas:
        out = np.where(d < np.datetime64(NASDAQ_DECIMAL), np.where(p >= 10, 0.0625, 1 / 32), out)
        out = np.where(d < np.datetime64(NASDAQ_SIXTEENTHS), np.where(p >= 10, 0.125, 1 / 32), out)
    else:
        out = np.where(d < np.datetime64(NYSE_DECIMAL), np.where(p >= 1, 0.0625, 1 / 32), out)
        out = np.where(d < np.datetime64(NYSE_SIXTEENTHS), np.where(p >= 1, 0.125, 0.0625), out)
    return out


@lru_cache(maxsize=4)
def load_calibration(path: Optional[str] = None) -> dict:
    with open(path or CALIBRATION_PATH) as f:
        return json.load(f)


@dataclass
class CalibratedSpreadConfig:
    calibration_path: Optional[str] = None
    measure: str = "quoted"            # 'quoted' (buy at ask) | 'effective' (avg. trade vs mid)
    listing: str = "NYSE"              # tick regime for pre-2001 dates: 'NYSE' | 'NASDAQ'
    window: int = 20                   # ADV$ / sigma lookback (prior sessions)
    min_periods: int = 10
    cs_window: int = 21
    cs_cap: bool = True                # cap by CS/AR where ADV$ is below the calibration range
    era_adjust: bool = True
    tod: bool = True
    extended_hours_mult: float = 3.0
    default_half_spread: float = 0.0010   # no ADV/vol available (flagged 'default')
    max_half_spread: float = 0.05
    tick_floor: bool = True
    overrides: dict = field(default_factory=dict)   # symbol -> fixed half spread (fraction)


class CalibratedSpreadModel:
    """Drop-in for :class:`atlasbt.execution.spread.SpreadModel` (same ``series`` /
    ``full_spread`` contract: ``series`` returns PIT *full* relative spreads for session i
    using sessions <= i-1; ``full_spread`` applies time of day and the fill-price tick floor)
    plus a scalar :meth:`half_spread` API."""

    def __init__(self, cfg: Optional[CalibratedSpreadConfig] = None):
        self.cfg = cfg or CalibratedSpreadConfig()
        self.cal = load_calibration(self.cfg.calibration_path)
        self._terms = list(self.cal["terms"])
        self._b = np.array([self.cal["beta"][t] for t in self._terms], float)
        c = self.cal["center"]
        self._c = (c["log_adv"], c["log_sigma"], c["log_price"])
        kn = self.cal.get("knots", {})
        self._k_adv = math.log(kn.get("adv_hi", 1e8)) - self._c[0]
        self._k_px = math.log(kn.get("price_lo", 5.0)) - self._c[2]
        cl = self.cal["clip_log"]
        self._clip = ((cl["log_adv"][0], cl["log_sigma"][0], cl["log_price"][0]),
                      (cl["log_adv"][1], cl["log_sigma"][1], cl["log_price"][1]))
        self._cd = float(self.cal["c_d"])
        self._q = float(self.cal["q"])
        anchors = sorted(self.cal["era_anchors"])
        self._era_t = np.array([a[0] for a in anchors], float)
        self._era_a = np.array([a[1] for a in anchors], float)
        self._era_b = np.array([a[2] for a in anchors], float)
        self._adv_lo = float(self.cal["calib_range"]["adv_min"])
        tod = self.cal["tod_profile"]
        self._tod_liq = np.array(tod["liquid"], float)
        self._tod_ill = np.array(tod["illiquid"], float)
        self._tod_split = float(tod["adv_split"])
        self._tod_bucket = float(tod.get("bucket_minutes", 5.0))
        er = self.cal["effective_ratio"]
        self._eff = (float(er["a"]), float(er["b"]), float(er["center"]), float(er["min"]), float(er["max"]))
        self.method = "calibrated"

    # ------------------------------------------------------------------ core (vectorised)
    def era_multiplier(self, dates, adv_dollars=None) -> np.ndarray:
        """Multiplier on the latent (non-tick) spread relative to the 2026 NBBO fit:
        exp(a(t) + b(t) * (log ADV$ - centre)), (a, b) linearly interpolated between anchors."""
        d = pd.DatetimeIndex(np.atleast_1d(np.asarray(dates, dtype="datetime64[ns]")))
        if not self.cfg.era_adjust:
            return np.ones(len(d))
        yf = np.asarray(d.year + (d.dayofyear - 1) / 365.25, float)
        a = np.interp(yf, self._era_t, self._era_a)
        b = np.interp(yf, self._era_t, self._era_b)
        la = 0.0 if adv_dollars is None else np.log(np.maximum(np.asarray(adv_dollars, float), 1.0)) - self._c[0]
        return np.exp(a + b * la)

    def _latent_frac(self, adv, sigma, price, etf) -> np.ndarray:
        """exp(x'b): latent relative full spread at the 2026 calibration (before era/tick). Inputs are
        clipped to the calibration range (no extrapolation of the regressors themselves)."""
        lo, hi = self._clip
        la = np.clip(np.log(np.maximum(adv, 1.0)), lo[0], hi[0]) - self._c[0]
        ls = np.clip(np.log(np.maximum(sigma, 1e-6)), lo[1], hi[1]) - self._c[1]
        lp = np.clip(np.log(np.maximum(price, 1e-4)), lo[2], hi[2]) - self._c[2]
        cols = design_terms(self._terms, la, ls, lp, np.asarray(etf, float), self._k_adv, self._k_px)
        return np.exp(cols @ self._b)

    def _eff_ratio(self, adv) -> np.ndarray:
        a, b, c, lo, hi = self._eff
        return np.clip(np.exp(a + b * (np.log(np.maximum(adv, 1.0)) - c)), lo, hi)

    def half_spread_vec(self, dates, price, adv_dollars, daily_vol, etf, cs_half=None,
                        tick=None) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised model. Returns (half_spread_frac, reason_code) where reason_code is 0 ok,
        1 extrapolated (ADV$ below calibration range, CS-capped when available), 2 default (no
        inputs). ``dates`` naive ET session dates; all inputs must be PIT (prior sessions)."""
        price = np.asarray(price, float)
        n = len(price)
        adv = np.asarray(adv_dollars, float) * np.ones(n)
        sig = np.asarray(daily_vol, float) * np.ones(n)
        etf = np.asarray(etf, float) * np.ones(n)
        dates = np.asarray(dates, dtype="datetime64[ns]").reshape(-1)
        if len(dates) != n:
            dates = np.full(n, dates[0])
        k = _era_tick_vec(dates, price, self.cfg.listing) if tick is None else np.asarray(tick, float) * np.ones(n)
        ok = np.isfinite(adv) & (adv > 0) & np.isfinite(sig) & (sig > 0) & np.isfinite(price) & (price > 0)
        L = self._latent_frac(np.where(ok, adv, 1e7), np.where(ok, sig, 0.02), np.where(ok, price, 50.0), etf)
        lo, hi = self._clip
        L = L * self.era_multiplier(dates, np.clip(np.where(ok, adv, 1e7), math.exp(lo[0]), math.exp(hi[0])))
        p = np.where(ok, price, 50.0)
        r = L * p / k + self._cd
        S = k * (1.0 + r ** self._q) ** (1.0 / self._q)
        h = S / (2.0 * p)
        if self.cfg.measure == "effective":
            h = h * self._eff_ratio(np.where(ok, adv, 1e7))
        code = np.zeros(n, np.int8)
        ext = ok & (adv < self._adv_lo)
        code[ext] = 1
        if cs_half is not None and self.cfg.cs_cap:
            cs = np.asarray(cs_half, float) * np.ones(n)
            cap = ext & np.isfinite(cs) & (cs > 0)
            floor = k / (2.0 * p)
            h = np.where(cap, np.maximum(np.minimum(h, cs), floor), h)
        h = np.where(ok, h, self.cfg.default_half_spread)
        code[~ok] = 2
        if self.cfg.tick_floor:
            h = np.where(ok, np.maximum(h, k / (2.0 * p)), h)
        return np.minimum(h, self.cfg.max_half_spread), code

    # ------------------------------------------------------------------ scalar API
    def half_spread(self, symbol: Optional[str], ts, price: float, adv_dollars: float, daily_vol: float,
                    asset_class: Union[str, object] = "STK", cs_half: Optional[float] = None,
                    minutes_since_open: Optional[float] = None, minutes_to_close: Optional[float] = None,
                    tick: Optional[float] = None) -> tuple[float, str]:
        """Half spread as a fraction of price for one order.

        ``ts``: session date or timestamp (era); ``price``: last known RAW price (prior close);
        ``adv_dollars``/``daily_vol``: mean dollar volume and std of daily returns over prior
        sessions; ``asset_class``: 'STK' | 'ETF' (or AssetClass). Time of day is applied when
        ``minutes_since_open`` is given (daily average otherwise)."""
        if symbol is not None and symbol in self.cfg.overrides:
            return float(self.cfg.overrides[symbol]), "spread override (user)"
        d = pd.Timestamp(ts)
        if d.tzinfo is not None:
            d = d.tz_convert("America/New_York").tz_localize(None)
        d = d.normalize()
        h, code = self.half_spread_vec(np.array([d.to_datetime64()]), np.array([price], float),
                                       [adv_dollars], [daily_vol], [is_etf(asset_class)],
                                       None if cs_half is None else [cs_half],
                                       None if tick is None else [tick])
        hv = float(h[0]) * self.tod_multiplier(minutes_since_open, minutes_to_close, adv_dollars)
        return min(hv, self.cfg.max_half_spread), self._reason(int(code[0]), d)

    def _reason(self, code: int, d: pd.Timestamp) -> str:
        if code == 2:
            return "default"
        era = "" if d >= pd.Timestamp(self.cal["calib_years"][0]) else ", era-extrapolated"
        if code == 1:
            return f"{REASON_CALIBRATED}, ADV below calibration range{era}"
        return f"{REASON_CALIBRATED}{era}"

    # ------------------------------------------------------------------ time of day
    def tod_multiplier(self, minutes_since_open: Optional[float], minutes_to_close: Optional[float],
                       adv_dollars: Optional[float] = None) -> float:
        """Multiplier relative to the RTH time-weighted average spread (5-minute buckets from
        NBBO 2026-07-01; 'liquid' profile above ``adv_split`` dollar volume). None -> 1.0
        (daily-average spread); outside RTH -> ``extended_hours_mult``."""
        if not self.cfg.tod or minutes_since_open is None:
            return 1.0
        if minutes_since_open < 0 or (minutes_to_close is not None and minutes_to_close < 0):
            return self.cfg.extended_hours_mult
        prof = self._tod_liq if (adv_dollars is not None and np.isfinite(adv_dollars)
                                 and adv_dollars >= self._tod_split) else self._tod_ill
        nb = len(prof)
        if minutes_to_close is not None and minutes_to_close < 30:
            # last 30 minutes are anchored to the actual close (early-close days)
            i = nb - 1 - int(max(minutes_to_close - 1e-9, 0.0) // self._tod_bucket)
        else:
            i = min(int(minutes_since_open // self._tod_bucket), nb - 6)
        return float(prof[min(max(i, 0), nb - 1)])

    # ------------------------------------------------------------------ SpreadModel contract
    def series(self, high, low, close, *, raw_close=None, volume=None, dates=None,
               asset_class: Union[str, object] = "STK", adv_dollar=None, sigma=None) -> SpreadSeries:
        """PIT per-session FULL relative spread (2 x half), index i uses sessions <= i-1.

        ``high/low/close``: split-consistent (back-adjusted) arrays as passed to SpreadModel;
        ``raw_close``: as-traded closes (price level / tick regime; required for a calibrated
        estimate); ``volume``: as-traded shares; ``dates``: session dates (naive ET);
        ``adv_dollar``/``sigma``: optional precomputed PIT arrays (already shifted, as in
        engine.market.DailyStats) - computed here from raw_close*volume and close returns
        otherwise. Without raw_close/volume/dates the model cannot run and falls back to the
        Corwin-Schultz estimate capped at the model-free default (source 'Corwin-Schultz')."""
        cfg = self.cfg
        h = np.asarray(high, float); l = np.asarray(low, float); c = np.asarray(close, float)
        n = len(c)
        cs = corwin_schultz_rolling(h, l, c, cfg.cs_window, 5)
        ar = abdi_ranaldo_rolling(h, l, c, cfg.cs_window, 5)
        csx = np.where(np.isfinite(cs) & (cs > 0), cs, ar)
        cs_sh = np.full(n, np.nan)
        if n > 1:
            cs_sh[1:] = csx[:-1]
        if raw_close is None or (volume is None and adv_dollar is None) or dates is None:
            est = np.where(np.isfinite(cs_sh) & (cs_sh > 0), cs_sh, np.nan)
            src = np.where(np.isfinite(est), "Corwin-Schultz", "default").astype(object)
            return SpreadSeries(est, src)
        rc = np.asarray(raw_close, float)
        if adv_dollar is None:
            dv = pd.Series(rc * np.asarray(volume, float))
            adv_dollar = _shift(dv.rolling(cfg.window, min_periods=cfg.min_periods).mean().to_numpy())
        if sigma is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.full(n, np.nan)
                r[1:] = c[1:] / c[:-1] - 1.0
            sigma = _shift(pd.Series(r).rolling(cfg.window, min_periods=cfg.min_periods).std().to_numpy())
        prev = np.full(n, np.nan)
        if n > 1:
            prev[1:] = rc[:-1]
        d = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
        if d.tz is not None:
            d = d.tz_convert("America/New_York").tz_localize(None)
        d = d.normalize().to_numpy()
        hs, code = self.half_spread_vec(d, prev, adv_dollar, sigma, float(is_etf(asset_class)),
                                        cs_half=0.5 * cs_sh)
        est = np.where(code == 2, np.nan, 2.0 * hs)
        calib0 = np.datetime64(pd.Timestamp(self.cal["calib_years"][0]))
        src = np.empty(n, dtype=object)
        src[:] = "calibrated"
        src[(code == 0) & (d < calib0)] = "calibrated, era-extrapolated"
        src[code == 1] = "calibrated, ADV below calibration range"
        src[code == 2] = "default"
        return SpreadSeries(est, src)

    def full_spread(self, estimate: float, price: float, tick: Optional[float] = None,
                    minutes_since_open: Optional[float] = None,
                    minutes_to_close: Optional[float] = None, adv_dollars: Optional[float] = None) -> float:
        """Relative FULL spread at a fill: PIT estimate x time-of-day, >= 1 tick at the fill price."""
        cfg = self.cfg
        s = estimate if (estimate is not None and np.isfinite(estimate) and estimate > 0) else 2 * cfg.default_half_spread
        s *= self.tod_multiplier(minutes_since_open, minutes_to_close, adv_dollars)
        if price > 0 and cfg.tick_floor:
            tk = tick if tick is not None else (0.01 if price >= 1.0 else 0.0001)
            s = max(s, tk / price)
        return min(s, 2 * cfg.max_half_spread * cfg.extended_hours_mult)


TERMS_VOCAB = ("1", "la", "ls", "lp", "e", "e_la", "e_ls", "e_lp", "e_hinge_hi", "hinge_la_hi", "hinge_lp_lo",
               "e_s", "e_s_la", "e_s_ls", "e_s_lp")


def design_terms(terms, la, ls, lp, e, k_adv, k_px, s_width: float = 0.5) -> np.ndarray:
    """Regressors from centred logs of ADV$ (la), daily sigma (ls), price (lp) and the ETF flag (e):
    hinges at the ADV$ / price knots and a smooth 'liquid ETF' regime s = logistic((la - k_adv)/w)."""
    la = np.atleast_1d(np.asarray(la, float)); n = la.shape[0]
    ls = np.broadcast_to(np.asarray(ls, float), (n,)); lp = np.broadcast_to(np.asarray(lp, float), (n,))
    e = np.broadcast_to(np.asarray(e, float), (n,))
    sg = 1.0 / (1.0 + np.exp(-(la - k_adv) / s_width))
    T = {"1": np.ones(n), "la": la, "ls": ls, "lp": lp, "e": e, "e_la": e * la, "e_ls": e * ls, "e_lp": e * lp,
         "e_hinge_hi": e * np.maximum(la - k_adv, 0.0), "hinge_la_hi": np.maximum(la - k_adv, 0.0),
         "hinge_lp_lo": np.minimum(lp - k_px, 0.0), "e_s": e * sg, "e_s_la": e * sg * (la - k_adv),
         "e_s_ls": e * sg * ls, "e_s_lp": e * sg * lp}
    return np.column_stack([T[t] for t in terms])


def _shift(x: np.ndarray) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) > 1:
        out[1:] = x[:-1]
    return out


def is_etf(asset_class) -> bool:
    v = getattr(asset_class, "value", asset_class)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return bool(v)
    return str(v).upper() in _ETF_CLASSES
