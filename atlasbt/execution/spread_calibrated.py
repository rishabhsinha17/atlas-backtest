"""Calibrated, point-in-time, era-aware half-spread model (fidelity T2 "spread modelled (calibrated)").

Why: the daily-OHLC estimators in :mod:`atlasbt.execution.spread` (Corwin-Schultz 2012, Abdi-Ranaldo
2017) are dominated by intraday volatility for liquid names. For SPY they give 18-50 bp full
spreads against a real NBBO of ~0.3-0.6 bp, so every ETF / large-cap backtest paid a cost about
30x too high. This module predicts the quoted (default) or effective half-spread from PIT
liquidity characteristics and a fit to real consolidated quotes.

Model (all inputs from sessions strictly before the trading session):

    latent full $ spread   L = P * exp(x'b) * era(date, ADV$)
        x = [1, la, ls, lp, E, E*la, E*ls, E*lp, min(lp - lp($5), 0), E*s, E*s*ls, E*s*lp, E*s*(la - la($100M))]
        la/ls/lp = centred logs of ADV$, daily sigma, price; E = ETF; s = logistic((la - la($100M))/0.5)
        ETF branch: running minimum over ADV$ (more liquid is never wider)
    quoted full $ spread   S = k * (1 + (L/k + c_d)^q)^(1/q)          (tick discreteness)
    half spread (frac)     h = S / (2 P)  [x effective/quoted ratio if measure='effective']

era(t, ADV$): literature-anchored multiplier on the latent spread before 2010 (Furfine 2001,
Hendershott-Jones-Menkveld 2003 above HJM's smallest quintile; Hasbrouck 2009 annual effective-cost
medians/95th percentiles below it), forced non-increasing in t at fixed ADV$ (backward cumulative
maximum over later dates, so never below the 2026 fit). k = minimum quoting increment of the era
and listing venue (:func:`era_tick`): 1/8, 1/16, Amex 1/16 from 1997-05-07, SPDR/MidCap
SPDR/DIAMONDS/QQQ 1/64 on Amex, decimal from 2001-01-29 (NYSE/Amex) / 2001-04-09 (Nasdaq),
$0.0001 below $1 from Reg NMS Rule 612 compliance (2006-01-31). ``S >= k`` always.

Calibration (``calibration/spread_calibration.json``, built by ``calibration/build_calibration.py``
from data documented in docs/SPREADS.md): b, c_d, q fitted by robust NLS to time-weighted NBBO
half-spreads of ~all US common stocks and ETFs on 2026-07-01 (NYSE Daily TAQ sample), liquid-ETF
sigma slope bounded by the stock slope; time-of-day profiles (liquid stocks, liquid ETFs, illiquid)
from 5-minute NBBO buckets; effective/quoted ratio from TAQ trades vs the prevailing NBBO.

Corwin-Schultz/Abdi-Ranaldo: a CAP where ADV$ is below the calibration range, and the FALLBACK for
stale/halted names without volume. Missing / zero sigma (IPO, stale price) uses a sigma prior by
price bucket, shrunk with the observed returns.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .spread import SpreadSeries, abdi_ranaldo_rolling, corwin_schultz_rolling

CALIBRATION_PATH = Path(__file__).with_name("calibration") / "spread_calibration.json"

#: tick-regime dates (sources in docs/SPREADS.md section 3)
NYSE_SIXTEENTHS = pd.Timestamp("1997-06-24")
NASDAQ_SIXTEENTHS = pd.Timestamp("1997-06-02")
AMEX_SIXTEENTHS = pd.Timestamp("1997-05-07")
NYSE_DECIMAL = pd.Timestamp("2001-01-29")      # NYSE + Amex: all equities and ETFs in decimals
NASDAQ_DECIMAL = pd.Timestamp("2001-04-09")
RULE_612 = pd.Timestamp("2006-01-31")           # Reg NMS Rule 612 compliance date (70 FR 45529)
#: decimal pilot phases (membership lists are not shipped: use CalibratedSpreadConfig.decimal_dates)
NYSE_PILOT_PHASES = ("2000-08-28", "2000-09-25", "2000-12-04")
NASDAQ_PILOT_PHASES = ("2001-03-12", "2001-03-26")
#: Amex Portfolio Depositary Receipts quoted in 1/64 until decimalisation (Amex Rule 1000 Comm. .03;
#: SEC Rel. 34-38571 for SPY/MDY; FR 2002-05-08 SR-Amex-2002-21) -> (first listing date)
PDR_SIXTYFOURTHS = {"SPY": "1993-01-29", "MDY": "1995-05-04", "DIA": "1998-01-20", "QQQ": "1999-03-10"}

_ETF_CLASSES = {"ETF", "ETP", "ETN", "FUND"}
REASON_CALIBRATED = "calibrated spread model (T2)"
_EXCHANGE_ALIASES = {
    "NYSE": "NYSE", "N": "NYSE", "XNYS": "NYSE",
    "AMEX": "AMEX", "ASE": "AMEX", "A": "AMEX", "XASE": "AMEX", "NYSE MKT": "AMEX", "NYSEMKT": "AMEX",
    "NYSE AMERICAN": "AMEX", "NYSEAMERICAN": "AMEX",
    "ARCA": "ARCA", "NYSEARCA": "ARCA", "NYSE ARCA": "ARCA", "PCX": "ARCA", "P": "ARCA", "ARCX": "ARCA",
    "BATS": "BATS", "CBOE": "BATS", "BZX": "BATS", "Z": "BATS",
}


def normalize_exchange(exchange: Optional[str]) -> Optional[str]:
    """'NYSE' | 'AMEX' | 'NASDAQ' | 'ARCA' | 'BATS' | None (unknown)."""
    if exchange is None:
        return None
    e = str(exchange).strip().upper()
    if not e or e in ("NONE", "NAN", "UNKNOWN"):
        return None
    if e.startswith("NAS") or e in ("Q", "NMS", "NGM", "NCM", "XNAS", "NNM", "NSC"):
        return "NASDAQ"
    return _EXCHANGE_ALIASES.get(e, e)


def _naive_day(date) -> pd.Timestamp:
    d = pd.Timestamp(date)
    if d.tzinfo is not None:
        d = d.tz_convert("America/New_York").tz_localize(None)
    return d.normalize()


def era_tick(date, price: float, listing: str = "NYSE", symbol: Optional[str] = None,
             asset_class=None, decimal_date=None) -> float:
    """Minimum quoting increment ($) for a stock/ETF quoted at ``price`` on ``date``.

    ``listing``: listing venue ('NYSE' default, 'NASDAQ', 'AMEX'; 'ARCA'/'BATS' before 2001 are
    treated as Amex). NYSE: 1/8 (>= $1; 1/16 below) until 1997-06-23, 1/16 (>= $1; 1/32 below) until
    decimalisation. Nasdaq: 1/8 (>= $10; 1/32 below) until 1997-05-30, then 1/16 (>= $10; 1/32 below).
    Amex: 1/8 >= $10, 1/16 $0.25-10, 1/32 below until 1997-05-06, then 1/16 >= $0.25.
    Amex PDRs SPY / MDY / DIA / QQQ: 1/64. Decimal: 2001-01-29 NYSE/Amex, 2001-04-09 Nasdaq
    (``decimal_date`` overrides for pilot stocks). Below $1: $0.01 until Rule 612 compliance
    (2006-01-31), then $0.0001. The SEC Tick Size Pilot is not applied (pass ``tick=`` instead)."""
    d = _naive_day(date)
    return float(_era_tick_vec(np.array([d.to_datetime64()]), np.array([float(price)]), listing,
                               symbol, decimal_date)[0])


def _era_tick_vec(dates: np.ndarray, price: np.ndarray, listing: Optional[str], symbol: Optional[str] = None,
                  decimal_date=None) -> np.ndarray:
    d = np.asarray(dates, dtype="datetime64[ns]")
    p = np.asarray(price, float)
    ex = normalize_exchange(listing) or "NYSE"
    if ex in ("ARCA", "BATS"):
        ex = "AMEX"                      # pre-2001 exchange-traded products were Amex-listed
    sym = None if symbol is None else str(symbol).upper()
    dec = np.datetime64(pd.Timestamp(decimal_date)) if decimal_date is not None else \
        np.datetime64(NASDAQ_DECIMAL if ex == "NASDAQ" else NYSE_DECIMAL)
    post = np.where(p >= 1.0, 0.01, np.where(d < np.datetime64(RULE_612), 0.01, 0.0001))
    if sym in PDR_SIXTYFOURTHS:
        pre = np.full(len(p), 1 / 64)
        dec = np.datetime64(NYSE_DECIMAL) if decimal_date is None else dec
    elif ex == "NASDAQ":
        pre = np.where(d < np.datetime64(NASDAQ_SIXTEENTHS), np.where(p >= 10, 0.125, 1 / 32),
                       np.where(p >= 10, 0.0625, 1 / 32))
    elif ex == "AMEX":
        early = np.where(p >= 10, 0.125, np.where(p >= 0.25, 0.0625, 1 / 32))
        late = np.where(p >= 0.25, 0.0625, 1 / 32)
        pre = np.where(d < np.datetime64(AMEX_SIXTEENTHS), early, late)
    else:
        pre = np.where(d < np.datetime64(NYSE_SIXTEENTHS), np.where(p >= 1, 0.125, 0.0625),
                       np.where(p >= 1, 0.0625, 1 / 32))
    return np.where(d < dec, pre, post)


def load_calibration(path: Optional[str] = None) -> dict:
    return _load_calibration(str(path or CALIBRATION_PATH))


@lru_cache(maxsize=4)
def _load_calibration(path: str) -> dict:
    with open(path) as f:
        return normalize_calibration(json.load(f), source=path)


#: median daily sigma by price bucket used when a v1 calibration (no 'sigma_prior') is loaded
_DEFAULT_SIGMA_PRIOR = [[1.0, 0.08], [5.0, 0.045], [10.0, 0.03], [20.0, 0.024], [50.0, 0.02], [1e12, 0.017]]


def normalize_calibration(cal: dict, source: str = "<dict>") -> dict:
    """Accept calibration schema v1 ('era_anchors' [[t, a, b], ...]) and v2 ('era' {large, small,
    small_below_adv}, 'sigma_prior', optional 'liquid_etf' profile). Raises ValueError naming the
    missing keys otherwise."""
    cal = dict(cal)
    need = ("terms", "beta", "center", "clip_log", "c_d", "q", "tod_profile", "effective_ratio", "calib_range")
    missing = [k for k in need if k not in cal]
    if "era" not in cal and "era_anchors" not in cal:
        missing.append("era (v2) or era_anchors (v1)")
    if missing:
        raise ValueError(f"spread calibration {source}: unsupported schema, missing {missing}")
    if "era" not in cal:          # v1 -> v2: no small-cap slope, no monotone constraint data needed
        cal["era"] = {"large": [list(map(float, a)) for a in cal["era_anchors"]],
                      "small": [[1990.0, 0.0], [2100.0, 0.0]], "small_below_adv": 2.17e6}
    cal.setdefault("sigma_prior", _DEFAULT_SIGMA_PRIOR)
    cal.setdefault("version", 1)
    return cal


@dataclass
class CalibratedSpreadConfig:
    calibration_path: Optional[str] = None
    measure: str = "quoted"            # 'quoted' (buy at ask) | 'effective' (avg. trade vs mid)
    listing: str = "NYSE"              # fallback tick regime when the exchange is unknown (pre-2001; flagged)
    window: int = 20                   # ADV$ / sigma lookback (prior sessions)
    min_periods: int = 1               # ADV$ needs >= 1 prior session; sigma is shrunk to a prior
    sigma_prior_weight: float = 5.0    # pseudo-observations of the price-bucket sigma prior
    cs_window: int = 21
    cs_cap: bool = True                # cap by CS/AR where ADV$ is below the calibration range
    cs_fallback: bool = True           # CS/AR when there is no volume (stale / halted) instead of the default
    era_adjust: bool = True
    era_monotone: bool = True          # era(t, ADV$) non-increasing in t (backward cumulative max)
    etf_monotone: bool = True          # ETF branch: running minimum over ADV$
    tod: bool = True
    extended_hours_mult: float = 3.0
    default_half_spread: float = 0.0010   # no usable inputs and no CS/AR (flagged 'default')
    max_half_spread: float = 0.05
    tick_floor: bool = True
    overrides: dict = field(default_factory=dict)        # symbol -> fixed half spread (fraction)
    decimal_dates: dict = field(default_factory=dict)    # symbol -> decimal start (pilot stocks)


class CalibratedSpreadModel:
    """Drop-in for :class:`atlasbt.execution.spread.SpreadModel` (same ``series`` /
    ``full_spread`` contract: ``series`` returns PIT *full* relative spreads for session i
    using sessions <= i-1; ``full_spread`` applies time of day and the fill-price tick floor)
    plus a scalar :meth:`half_spread` API. ``calibration`` (dict) overrides the json file."""

    _GRID = 48

    def __init__(self, cfg: Optional[CalibratedSpreadConfig] = None, calibration: Optional[dict] = None):
        self.cfg = cfg or CalibratedSpreadConfig()
        self.cal = (normalize_calibration(calibration) if calibration is not None
                    else load_calibration(self.cfg.calibration_path))
        cal = self.cal
        self._terms = list(cal["terms"])
        self._b = np.array([cal["beta"][t] for t in self._terms], float)
        c = cal["center"]
        self._c = (c["log_adv"], c["log_sigma"], c["log_price"])
        kn = cal.get("knots", {})
        self._k_adv = math.log(kn.get("adv_hi", 1e8)) - self._c[0]
        self._k_px = math.log(kn.get("price_lo", 5.0)) - self._c[2]
        self._k_vhi = math.log(kn.get("adv_vhi", 5e9)) - self._c[0]
        cl = cal["clip_log"]
        self._clip = ((cl["log_adv"][0], cl["log_sigma"][0], cl["log_price"][0]),
                      (cl["log_adv"][1], cl["log_sigma"][1], cl["log_price"][1]))
        self._cd = float(cal["c_d"])
        self._q = float(cal["q"])
        era = cal["era"]
        la = sorted(era["large"])
        self._eL = (np.array([a[0] for a in la], float), np.array([a[1] for a in la], float),
                    np.array([a[2] for a in la], float))
        sm = sorted(era["small"])
        self._eS = (np.array([a[0] for a in sm], float), np.array([a[1] for a in sm], float))
        self._la_q5 = math.log(era["small_below_adv"]) - self._c[0]
        self._era_times = np.unique(np.concatenate([self._eL[0], self._eS[0]]))
        self._adv_lo = float(cal["calib_range"]["adv_min"])
        tod = cal["tod_profile"]
        self._tod = {k: np.array(tod[k], float) for k in ("liquid", "illiquid", "liquid_etf") if k in tod}
        self._tod_split = float(tod["adv_split"])
        self._tod_bucket = float(tod.get("bucket_minutes", 5.0))
        er = cal["effective_ratio"]
        self._eff = (float(er["a"]), float(er["b"]), float(er["center"]), float(er["min"]), float(er["max"]))
        sp = cal["sigma_prior"]
        self._sp_ub = np.array([x[0] for x in sp], float)
        self._sp_v = np.array([x[1] for x in sp], float)
        self._calib0 = np.datetime64(pd.Timestamp(cal["calib_years"][0]))
        self.method = "calibrated"

    # ------------------------------------------------------------------ era
    def _log_era_raw(self, yf: np.ndarray, la: np.ndarray) -> np.ndarray:
        tL, aL, bL = self._eL
        a = np.interp(yf, tL, aL)
        b = np.interp(yf, tL, bL)
        big = a + b * la
        bs = np.interp(yf, *self._eS)
        small = a + b * self._la_q5 + np.minimum(bs, 0.0) * (la - self._la_q5)
        return np.where(la >= self._la_q5, big, small)

    def era_multiplier(self, dates, adv_dollars=None) -> np.ndarray:
        """Multiplier on the latent (non-tick) spread relative to the 2026 NBBO fit. Piecewise-linear in
        time between literature anchors; above HJM's smallest quintile exp(a(t) + b(t)·la), below it
        a slope b_s(t) <= 0 matched to Hasbrouck (2009); with ``era_monotone`` the backward cumulative
        maximum over later dates (non-increasing in t at fixed ADV$, >= 1 before 2010)."""
        d = pd.DatetimeIndex(np.atleast_1d(np.asarray(dates, dtype="datetime64[ns]")))
        n = len(d)
        if not self.cfg.era_adjust:
            return np.ones(n)
        yf = np.asarray(d.year + (d.dayofyear - 1) / 365.25, float)
        if adv_dollars is None:
            la = np.zeros(n)
        else:
            lo, hi = self._clip
            la = np.clip(np.log(np.maximum(np.asarray(adv_dollars, float), 1.0)), lo[0], hi[0]) - self._c[0]
            la = la * np.ones(n)
        v = self._log_era_raw(yf, la)
        if self.cfg.era_monotone:
            for T in self._era_times:
                later = T > yf
                if later.any():
                    v = np.where(later, np.maximum(v, self._log_era_raw(np.full(n, T), la)), v)
        return np.exp(v)

    # ------------------------------------------------------------------ latent
    def _latent_raw(self, la, ls, lp, etf) -> np.ndarray:
        cols = design_terms(self._terms, la - self._c[0], ls - self._c[1], lp - self._c[2],
                            np.asarray(etf, float), self._k_adv, self._k_px, k_vhi=self._k_vhi)
        return np.exp(cols @ self._b)

    def _latent_frac(self, adv, sigma, price, etf, dates=None) -> np.ndarray:
        """exp(x'b) * era: latent relative full spread (before tick). Inputs are clipped to the
        calibration range; the ETF branch is made non-increasing in ADV$ by a running minimum."""
        lo, hi = self._clip
        la = np.clip(np.log(np.maximum(np.asarray(adv, float), 1.0)), lo[0], hi[0])
        ls = np.clip(np.log(np.maximum(np.asarray(sigma, float), 1e-6)), lo[1], hi[1])
        lp = np.clip(np.log(np.maximum(np.asarray(price, float), 1e-4)), lo[2], hi[2])
        e = np.asarray(etf, float) * np.ones(len(la))
        L = self._latent_raw(la, ls, lp, e)
        if dates is not None:
            L = L * self.era_multiplier(dates, np.exp(la))
        if self.cfg.etf_monotone and (e > 0).any():
            idx = np.nonzero(e > 0)[0]
            G = self._GRID
            w = np.linspace(0.0, 1.0, G)
            lag = lo[0] + (la[idx, None] - lo[0]) * w[None, :]          # grid from the range floor up to la
            m = len(idx)
            Lg = self._latent_raw(lag.ravel(), np.repeat(ls[idx], G), np.repeat(lp[idx], G),
                                  np.repeat(e[idx], G)).reshape(m, G)
            if dates is not None:
                dd = np.repeat(np.asarray(dates, dtype="datetime64[ns]").reshape(-1)[idx], G)
                Lg = Lg * self.era_multiplier(dd, np.exp(lag.ravel())).reshape(m, G)
            L[idx] = np.minimum(L[idx], Lg.min(axis=1))
        return L

    def _eff_ratio(self, adv) -> np.ndarray:
        a, b, c, lo, hi = self._eff
        return np.clip(np.exp(a + b * (np.log(np.maximum(adv, 1.0)) - c)), lo, hi)

    def sigma_prior(self, price) -> np.ndarray:
        """Median daily sigma of the fit-day universe by price bucket (IPO / stale-price prior)."""
        p = np.asarray(price, float)
        i = np.clip(np.searchsorted(self._sp_ub, p, side="right"), 0, len(self._sp_v) - 1)
        return self._sp_v[i]

    def half_spread_vec(self, dates, price, adv_dollars, daily_vol, etf, cs_half=None,
                        tick=None, symbol: Optional[str] = None, exchange: Optional[str] = None,
                        n_sigma=None) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised model. Returns (half_spread_frac, reason_code): 0 ok, 1 ADV$ below the
        calibration range (CS-capped when available), 2 default (no inputs), 3 CS/AR fallback (no
        volume), +8 when sigma came (partly) from the price-bucket prior, +16 when the tick regime was
        assumed (exchange unknown before 2001-04-09). ``dates`` naive ET session dates; all inputs PIT.
        ``n_sigma`` (optional) = number of returns behind ``daily_vol`` (shrinkage to the prior)."""
        price = np.asarray(price, float).reshape(-1)
        n = len(price)
        adv = np.asarray(adv_dollars, float) * np.ones(n)
        sig = np.asarray(daily_vol, float) * np.ones(n)
        etf = np.asarray(etf, float) * np.ones(n)
        dates = np.asarray(dates, dtype="datetime64[ns]").reshape(-1)
        if len(dates) != n:
            dates = np.full(n, dates[0])
        ex = normalize_exchange(exchange)
        code = np.zeros(n, np.int16)
        if tick is None:
            listing = ex or self.cfg.listing
            dec = self.cfg.decimal_dates.get(str(symbol).upper()) if symbol is not None else None
            k = _era_tick_vec(dates, price, listing, symbol, dec)
            if ex is None and not (symbol is not None and str(symbol).upper() in PDR_SIXTYFOURTHS):
                code[dates < np.datetime64(NASDAQ_DECIMAL)] |= 16
        else:
            k = np.asarray(tick, float) * np.ones(n)
        # sigma: shrink toward the price-bucket prior when few / no / zero returns
        sp = self.sigma_prior(np.where(np.isfinite(price) & (price > 0), price, 20.0))
        bad = ~(np.isfinite(sig) & (sig > 0))
        if n_sigma is not None:
            # shrink only incomplete windows (new listings): full windows use the sample sigma
            ns = np.where(bad, 0.0, np.asarray(n_sigma, float) * np.ones(n))
            short = ns < self.cfg.window - 1
            w = self.cfg.sigma_prior_weight
            shr = np.sqrt((ns * np.where(bad, 0.0, sig) ** 2 + w * sp ** 2) / (ns + w))
            sig = np.where(short, shr, sig)
            code[short] |= 8
        else:
            sig = np.where(bad, sp, sig)
            code[bad] |= 8
        ok = np.isfinite(adv) & (adv > 0) & np.isfinite(price) & (price > 0)
        L = self._latent_frac(np.where(ok, adv, 1e7), sig, np.where(ok, price, 50.0), etf, dates)
        p = np.where(ok, price, 50.0)
        r = L * p / k + self._cd
        S = k * (1.0 + r ** self._q) ** (1.0 / self._q)
        h = S / (2.0 * p)
        if self.cfg.measure == "effective":
            h = h * self._eff_ratio(np.where(ok, adv, 1e7))
        floor = k / (2.0 * p)
        ext = ok & (adv < self._adv_lo)
        code[ext] |= 1
        cs = None if cs_half is None else np.asarray(cs_half, float) * np.ones(n)
        if cs is not None and self.cfg.cs_cap:
            cap = ext & np.isfinite(cs) & (cs > 0)
            h = np.where(cap, np.maximum(np.minimum(h, cs), floor), h)
        # no volume / no price: CS/AR fallback (stale, halted) else default
        fb = ~ok
        if cs is not None and self.cfg.cs_fallback:
            use = fb & np.isfinite(cs) & (cs > 0)
            pp = np.where(np.isfinite(price) & (price > 0), price, np.nan)
            fl = np.where(np.isfinite(pp), _era_tick_vec(dates, np.nan_to_num(pp, nan=50.0),
                                                         ex or self.cfg.listing, symbol) / (2 * pp), 0.0)
            h = np.where(use, np.maximum(cs, np.nan_to_num(fl)), h)
            code[use] = (code[use] & 16) | 3
            fb = fb & ~use
        h = np.where(fb, self.cfg.default_half_spread, h)
        code[fb] = 2
        if self.cfg.tick_floor:
            h = np.where(ok, np.maximum(h, floor), h)
        return np.minimum(h, self.cfg.max_half_spread), code

    # ------------------------------------------------------------------ scalar API
    def half_spread(self, symbol: Optional[str], ts, price: float, adv_dollars: float, daily_vol: float,
                    asset_class: Union[str, object, None] = "STK", cs_half: Optional[float] = None,
                    minutes_since_open: Optional[float] = None, minutes_to_close: Optional[float] = None,
                    tick: Optional[float] = None, exchange: Optional[str] = None) -> tuple[float, str]:
        """Half spread as a fraction of price for one order.

        ``ts``: session date or timestamp (era); ``price``: last known price in today's share units
        (prior close, split-adjusted); ``adv_dollars``/``daily_vol``: mean dollar volume and std of daily
        returns over prior sessions; ``asset_class``: 'STK' | 'ETF' (or AssetClass; None = unknown,
        treated as stock and flagged); ``exchange``: listing venue (Instrument.exchange) for the
        pre-2001 tick regime. Time of day applies when ``minutes_since_open`` is given."""
        if symbol is not None and symbol in self.cfg.overrides:
            return float(self.cfg.overrides[symbol]), "spread override (user)"
        d = _naive_day(ts)
        etf = is_etf(asset_class) if asset_class is not None else False
        h, code = self.half_spread_vec(np.array([d.to_datetime64()]), np.array([price], float),
                                       [adv_dollars], [daily_vol], [float(etf)],
                                       None if cs_half is None else [cs_half],
                                       None if tick is None else [tick], symbol=symbol, exchange=exchange)
        hv = float(h[0]) * self.tod_multiplier(minutes_since_open, minutes_to_close, adv_dollars,
                                                 "ETF" if etf else "STK")
        return min(hv, self.cfg.max_half_spread), self._reason(int(code[0]), d, asset_class is None)

    def _reason(self, code: int, d: pd.Timestamp, ac_unknown: bool = False) -> str:
        base = code & 7
        if base == 2:
            return "default"
        parts = ["Corwin-Schultz/Abdi-Ranaldo fallback (no volume) (T2)" if base == 3 else REASON_CALIBRATED]
        if base != 3 and np.datetime64(d) < self._calib0:
            parts.append("era-extrapolated")
        if base == 1:
            parts.append("ADV below calibration range")
        if code & 8 and base != 3:
            parts.append("sigma prior")
        if code & 16:
            parts.append(f"listing assumed {self.cfg.listing}")
        if ac_unknown:
            parts.append("asset class inferred STK")
        return ", ".join(parts)

    # ------------------------------------------------------------------ time of day
    def tod_multiplier(self, minutes_since_open: Optional[float], minutes_to_close: Optional[float],
                       adv_dollars: Optional[float] = None, asset_class=None) -> float:
        """Multiplier relative to the RTH time-weighted average spread (5-minute buckets, NBBO
        2026-07-01). Profiles: 'liquid_etf' (ETF and ADV$ >= adv_split), 'liquid' (stocks >= adv_split),
        'illiquid' (else or ADV$ unknown). None -> 1.0 (daily average); outside RTH ->
        ``extended_hours_mult``."""
        if not self.cfg.tod or minutes_since_open is None:
            return 1.0
        if minutes_since_open < 0 or (minutes_to_close is not None and minutes_to_close < 0):
            return self.cfg.extended_hours_mult
        liquid = adv_dollars is not None and np.isfinite(adv_dollars) and adv_dollars >= self._tod_split
        if liquid and asset_class is not None and is_etf(asset_class) and "liquid_etf" in self._tod:
            prof = self._tod["liquid_etf"]
        else:
            prof = self._tod["liquid"] if liquid else self._tod["illiquid"]
        nb = len(prof)
        if minutes_to_close is not None and minutes_to_close < 30:
            i = nb - 1 - int(max(minutes_to_close - 1e-9, 0.0) // self._tod_bucket)
        else:
            i = min(int(minutes_since_open // self._tod_bucket), nb - 6)
        return float(prof[min(max(i, 0), nb - 1)])

    # ------------------------------------------------------------------ SpreadModel contract
    def series(self, high, low, close, *, raw_close=None, volume=None, dates=None,
               asset_class: Union[str, object, None] = "STK", adv_dollar=None, sigma=None,
               prior_close=None, exchange: Optional[str] = None, symbol: Optional[str] = None) -> SpreadSeries:
        """PIT per-session FULL relative spread (2 x half); index i uses sessions <= i-1.

        ``high/low/close``: split-consistent (back-adjusted) arrays as passed to SpreadModel;
        ``raw_close``: as-traded closes; ``volume``: as-traded shares; ``dates``: session dates (naive
        ET, datetime64 or int64 ns); ``prior_close``: price of session i-1 in session i's share units
        (engine.market.DailyStats.prior_close); if omitted it is raw_close[i-1] rescaled by the
        split/adjustment ratio implied by close/raw_close (so a split day is priced at the post-split
        level). ``adv_dollar``/``sigma``: optional precomputed PIT arrays (sigma is then shrunk to
        the price-bucket prior only where missing/zero). ``exchange``/``symbol`` select the pre-2001
        tick regime. Without raw_close/volume/dates the model cannot run and returns the Corwin-Schultz
        series (source 'Corwin-Schultz')."""
        cfg = self.cfg
        h = np.asarray(high, float); l = np.asarray(low, float); c = np.asarray(close, float)
        n = len(c)
        cs = corwin_schultz_rolling(h, l, c, cfg.cs_window, 5)
        ar = abdi_ranaldo_rolling(h, l, c, cfg.cs_window, 5)
        csx = np.where(np.isfinite(cs) & (cs > 0), cs, ar)
        cs_sh = _shift(csx)
        if raw_close is None or (volume is None and adv_dollar is None) or dates is None:
            est = np.where(np.isfinite(cs_sh) & (cs_sh > 0), cs_sh, np.nan)
            src = np.where(np.isfinite(est), "Corwin-Schultz", "default").astype(object)
            return SpreadSeries(est, src)
        rc = np.asarray(raw_close, float)
        if adv_dollar is None:
            dv = pd.Series(rc * np.asarray(volume, float))
            adv_dollar = _shift(dv.rolling(cfg.window, min_periods=max(cfg.min_periods, 1)).mean().to_numpy())
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.full(n, np.nan)
            r[1:] = c[1:] / c[:-1] - 1.0
        rr = pd.Series(np.where(np.isfinite(r) & (np.abs(r) < 0.5), r, np.nan))
        n_ret = _shift(rr.rolling(cfg.window, min_periods=1).count().to_numpy())
        if sigma is None:
            sigma = _shift(rr.rolling(cfg.window, min_periods=2).std().to_numpy())
            n_sig = np.nan_to_num(n_ret)
        else:
            sigma = np.asarray(sigma, float)
            n_sig = np.where(np.isfinite(sigma) & (sigma > 0), 1e9, 0.0)   # trust given sigma
        if prior_close is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                f = c / rc                                   # adjustment factor of each session
            prior_close = np.full(n, np.nan)
            if n > 1:
                prior_close[1:] = rc[:-1] * f[:-1] / f[1:]
        prev = np.asarray(prior_close, float)
        d = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
        if d.tz is not None:
            d = d.tz_convert("America/New_York").tz_localize(None)
        d = d.normalize().to_numpy()
        etf = is_etf(asset_class) if asset_class is not None else False
        hs, code = self.half_spread_vec(d, prev, adv_dollar, sigma, float(etf), cs_half=0.5 * cs_sh,
                                        symbol=symbol, exchange=exchange, n_sigma=n_sig)
        base = code & 7
        est = np.where(base == 2, np.nan, 2.0 * hs)
        src = np.empty(n, dtype=object)
        for i in range(n):
            b = base[i]
            if b == 2:
                src[i] = "default"
                continue
            parts = ["Corwin-Schultz fallback" if b == 3 else "calibrated"]
            if b != 3 and d[i] < self._calib0:
                parts.append("era-extrapolated")
            if b == 1:
                parts.append("ADV below calibration range")
            if code[i] & 8 and b != 3:
                parts.append("sigma prior")
            if code[i] & 16:
                parts.append("listing assumed")
            if asset_class is None:
                parts.append("asset class inferred")
            src[i] = ", ".join(parts)
        return SpreadSeries(est, src)

    def full_spread(self, estimate: float, price: float, tick: Optional[float] = None,
                    minutes_since_open: Optional[float] = None,
                    minutes_to_close: Optional[float] = None, adv_dollars: Optional[float] = None,
                    asset_class=None) -> float:
        """Relative FULL spread at a fill: PIT estimate x time-of-day, >= 1 tick at the fill price."""
        cfg = self.cfg
        s = estimate if (estimate is not None and np.isfinite(estimate) and estimate > 0) else 2 * cfg.default_half_spread
        s *= self.tod_multiplier(minutes_since_open, minutes_to_close, adv_dollars, asset_class)
        if price > 0 and cfg.tick_floor:
            tk = tick if tick is not None else (0.01 if price >= 1.0 else 0.0001)
            s = max(s, tk / price)
        return min(s, 2 * cfg.max_half_spread * cfg.extended_hours_mult)


TERMS_VOCAB = ("1", "la", "ls", "lp", "e", "e_la", "e_ls", "e_lp", "e_hinge_hi", "hinge_la_hi", "hinge_lp_lo",
               "e_s", "e_s_la", "e_s_ls", "e_s_lp", "e_hinge_vhi")


def design_terms(terms, la, ls, lp, e, k_adv, k_px, s_width: float = 0.5, k_vhi: float = 99.0) -> np.ndarray:
    """Regressors from centred logs of ADV$ (la), daily sigma (ls), price (lp) and the ETF flag (e):
    hinges at the ADV$ / price knots and a smooth 'liquid ETF' regime s = logistic((la - k_adv)/w)."""
    la = np.atleast_1d(np.asarray(la, float)); n = la.shape[0]
    ls = np.broadcast_to(np.asarray(ls, float), (n,)); lp = np.broadcast_to(np.asarray(lp, float), (n,))
    e = np.broadcast_to(np.asarray(e, float), (n,))
    sg = 1.0 / (1.0 + np.exp(-(la - k_adv) / s_width))
    T = {"1": np.ones(n), "la": la, "ls": ls, "lp": lp, "e": e, "e_la": e * la, "e_ls": e * ls, "e_lp": e * lp,
         "e_hinge_hi": e * np.maximum(la - k_adv, 0.0), "hinge_la_hi": np.maximum(la - k_adv, 0.0),
         "hinge_lp_lo": np.minimum(lp - k_px, 0.0), "e_s": e * sg, "e_s_la": e * sg * (la - k_adv),
         "e_s_ls": e * sg * ls, "e_s_lp": e * sg * lp, "e_hinge_vhi": e * np.maximum(la - k_vhi, 0.0)}
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
