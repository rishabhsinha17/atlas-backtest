"""SYNTHETIC option chains (fidelity T3_MODELED) for dates/underlyings without
real chain quotes.  Every synthetic frame carries ``fidelity = 3`` and
``attrs['source']`` describing the model.  NEVER mix silently with real quotes.

Model (docs/OPTIONS.md §5, calibration evidence in
``atlasbt/options/calibration/synthetic_params.json``):

1. Listed expiries / strikes / ticks: ``atlasbt.options.listing``.
2. ATM vol term structure from Cboe vol indices, point-in-time (an index
   close is usable only after 16:15 ET of its session):
     SPX/SPY/XSP: VIX1D (1d), VIX9D (9d), VIX (30d), VIX3M (93d), VIX6M (183d),
                  VIX1Y (365d) -> total-variance interpolation in T;
     QQQ/NDX -> VXN, IWM/RUT -> RVX, DIA -> VXD, AAPL -> VXAPL, AMZN -> VXAZN,
     GOOG(L) -> VXGOG, GS -> VXGS, IBM -> VXIBM, EEM -> VXEEM, EFA -> VXEFA,
     GDX -> VXGDX, SLV -> VXSLV, TLT -> VXTLT, GLD -> GVZ, USO -> OVX:
                  30d level from the mapped index, term SHAPE borrowed from the
                  VIX family (sigma(T) = idx30 * vixfam(T)/VIX);
     anything else: VIX family x ``rv_ratio`` (caller supplies the ratio of the
                  name's trailing realized vol to SPX's; see OPTIONS.md).
   Vol indices are variance-swap-like (include the skew premium), so the ATM
   implied vol is index x ``atm_ratio`` (calibrated on real DoltHub chains).
3. Smile: sigma(k) = sigma_atm * g(x), x = ln(K/F)/(sigma_atm sqrt(T)),
   g(x) = 1 + a x + b x^2 on |x| <= XMAX, linear continuation beyond,
   floored at G_FLOOR.  a depends on Cboe SKEW for index products:
   a = a0 + a1 (SKEW - 130)/10; single names use their own (a, b).
4. Prices: European BSM (index, cash-settled) or Bjerksund-Stensland 2002
   American (equity/ETF) with continuous carry q; greeks BSM.
5. Bid/ask: log-linear spread model calibrated on DoltHub quotes
   (``spread_model``), rounded outward to the historical tick grid; quotes
   with bid < 1 tick get bid 0 (no bid), like real far-OTM quotes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from atlasbt.core import clock
from atlasbt.core.types import Fidelity
from atlasbt.pricing.american import bs2002_price
from atlasbt.pricing.bsm import bsm_greeks, bsm_price
from atlasbt.pricing.time import year_fraction

from .listing import listed_expiries, product_spec, strike_grid, tick_size

ROOT = Path(__file__).resolve().parents[2]
CBOE_DIR = ROOT / "data_raw" / "cboe_indices"
PARAMS_PATH = Path(__file__).resolve().parent / "calibration" / "synthetic_params.json"

VIX_FAMILY = (("VIX1D", 1), ("VIX9D", 9), ("VIX", 30), ("VIX3M", 93), ("VIX6M", 183), ("VIX1Y", 365))
INDEX_MAP = {"QQQ": "VXN", "NDX": "VXN", "IWM": "RVX", "RUT": "RVX", "DIA": "VXD", "DJX": "VXD",
             "AAPL": "VXAPL", "AMZN": "VXAZN", "GOOG": "VXGOG", "GOOGL": "VXGOG", "GS": "VXGS", "IBM": "VXIBM",
             "EEM": "VXEEM", "EFA": "VXEFA", "GDX": "VXGDX", "SLV": "VXSLV", "TLT": "VXTLT", "GLD": "GVZ",
             "USO": "OVX"}
SPX_LIKE = {"SPX", "SPXW", "SPY", "XSP", "VOO", "IVV"}
VOL_INDEX_CLOSE_ET = "16:15"
XMAX = 1.5          # quadratic core |x| <= XMAX, linear wings beyond (calibrated slopes)
G_FLOOR = 0.35


# --------------------------------------------------------------------------
# Cboe index history (read-only loader of data_raw/cboe_indices)
# --------------------------------------------------------------------------
@lru_cache(maxsize=64)
def cboe_index_history(name: str, directory: str | None = None) -> pd.Series:
    """Close series of a Cboe index CSV (DATE,OPEN,HIGH,LOW,CLOSE or DATE,<NAME>),
    indexed by naive session date."""
    p = Path(directory) if directory else CBOE_DIR
    f = p / f"{name}_History.csv"
    if not f.exists():
        raise FileNotFoundError(f)
    df = pd.read_csv(f)
    col = "CLOSE" if "CLOSE" in df.columns else df.columns[-1]
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").values,
                  index=pd.to_datetime(df["DATE"], format="%m/%d/%Y"), name=name).dropna()
    s = s[s > 0]
    return s[~s.index.duplicated(keep="last")].sort_index()


def _last_known(series: pd.Series, as_of: pd.Timestamp, close_hhmm: str = VOL_INDEX_CLOSE_ET):
    """Latest value whose session close (hhmm ET) <= as_of; (value, session)."""
    as_of = clock.to_utc(as_of)
    sess = clock.session_date_of(as_of)
    cutoff = sess if as_of >= clock.et(sess, close_hhmm) else sess - pd.Timedelta(days=1)
    s = series.loc[:cutoff]
    if s.empty:
        return float("nan"), None
    return float(s.iloc[-1]), s.index[-1]


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------
@dataclass
class SyntheticParams:
    atm_ratio_index: float = 0.88        # SPX-family ATM IV / vol index (30d) at index level 20
    atm_ratio_logvix: float = 0.0        # d ratio / d ln(index level)  (regime dependence)
    atm_ratio_schedule: list = field(default_factory=list)   # point-in-time [{from, c0, c1, ...}]
    atm_ratio_obs: list = field(default_factory=list)        # per-roll PUT-implied {R, R2, vix, ratio}
    # window of the chain data behind skew/wing/spread parameters (NOT point-in-time)
    fit_window: list = field(default_factory=lambda: ["2019-08-15", "2026-09-16"])
    atm_ratio_single: float = 0.95       # single-name ATM IV / its vol index
    skew_a0: float = -0.30               # index products slope at SKEW=130
    skew_a1: float = -0.03               # per +10 SKEW points
    skew_b: float = 0.04
    skew_gamma: float = 0.0              # a(T) = a30 (DTE/30)^gamma
    wing_put: float = -0.25              # dg/dx for x < -XMAX (index products)
    wing_call: float = 0.0               # dg/dx for x > +XMAX
    single_wing_put: float = -0.15
    single_wing_call: float = 0.05
    skew_range: list = field(default_factory=lambda: [110.0, 160.0])   # SKEW values seen in calibration
    single_a: float = -0.12
    single_b: float = 0.03
    single_gamma: float = 0.0
    spread_coef: dict = field(default_factory=lambda: {"log_mid": 0.6, "log_mid2": 0.0, "absdelta": -1.0,
                                                       "log_dte": 0.0})
    spread_symbol_fe: dict = field(default_factory=dict)
    spread_default_fe: dict = field(default_factory=lambda: {"etf": -2.0, "stock": -2.0})
    spread_resid_sd: float = 0.6
    max_stale_days: int = 5              # vol index older than this -> NaN
    evidence: dict = field(default_factory=dict)

    @staticmethod
    def load(path: Path | str = PARAMS_PATH) -> "SyntheticParams":
        p = Path(path)
        if not p.exists():
            return SyntheticParams(evidence={"status": "UNCALIBRATED defaults"})
        d = json.loads(p.read_text())
        keep = {k: d[k] for k in SyntheticParams.__dataclass_fields__ if k in d}
        return SyntheticParams(**keep)


# --------------------------------------------------------------------------
# vol term structure
# --------------------------------------------------------------------------
@dataclass
class VolSource:
    """Point-in-time ATM vol term structure from Cboe vol indices."""

    params: SyntheticParams = field(default_factory=SyntheticParams.load)
    directory: Optional[str] = None

    def _idx(self, name, as_of):
        try:
            s = cboe_index_history(name, self.directory)
        except FileNotFoundError:
            return float("nan"), None
        v, d = _last_known(s, as_of)
        if d is not None and (clock.session_date_of(clock.to_utc(as_of)) - d).days > self.params.max_stale_days:
            return float("nan"), d
        return v, d

    def vix_family(self, as_of) -> pd.DataFrame:
        rows = []
        for name, days in VIX_FAMILY:
            v, d = self._idx(name, as_of)
            if np.isfinite(v):
                rows.append(dict(index=name, days=days, level=v / 100.0, session=d))
        return pd.DataFrame(rows)

    def atm_term(self, underlying: str, as_of, rv_ratio: float | None = None) -> pd.DataFrame:
        """ATM IV term points (days, atm_vol, source) for ``underlying``."""
        u = underlying.upper()
        fam = self.vix_family(as_of)
        if fam.empty or not (fam["index"] == "VIX").any():
            return pd.DataFrame(columns=["days", "atm_vol", "source"])
        vix30 = float(fam.loc[fam["index"] == "VIX", "level"].iloc[0])
        if u in SPX_LIKE:
            out = fam.assign(atm_vol=fam["level"] * atm_ratio(fam["level"].values, self.params, as_of=as_of),
                             source=fam["index"])
        elif u in INDEX_MAP:
            v, _ = self._idx(INDEX_MAP[u], as_of)
            if not np.isfinite(v):
                return pd.DataFrame(columns=["days", "atm_vol", "source"])
            shape = fam["level"] / vix30
            idx_like = u in ("QQQ", "NDX", "IWM", "RUT", "DIA", "DJX")
            ratio = atm_ratio(v / 100.0, self.params, single=not idx_like, as_of=as_of)
            out = fam.assign(atm_vol=v / 100.0 * shape * ratio, source=INDEX_MAP[u] + "*VIXshape")
        else:
            if rv_ratio is None or not np.isfinite(rv_ratio):
                return pd.DataFrame(columns=["days", "atm_vol", "source"])
            out = fam.assign(atm_vol=fam["level"] * atm_ratio(fam["level"].values, self.params, as_of=as_of) * rv_ratio,
                             source="VIXfam*rv_ratio")
        return out[["days", "atm_vol", "source"]].reset_index(drop=True)

    def skew_index(self, as_of) -> float:
        v, _ = self._idx("SKEW", as_of)
        return v


ATM_RATIO_PRIOR = (0.88, 0.0)   # judgmental prior (c0, c1) fixed BEFORE looking at post-2007 data
ATM_RATIO_PRIOR_N = 24          # rolls needed before the data fully replace the prior


def _huber_ratio_fit(obs: list) -> tuple[float, float]:
    import statsmodels.api as sm
    v = np.array([o["vix"] for o in obs], float)
    y = np.array([o["ratio"] for o in obs], float)
    if len(obs) < 8:
        return float(np.median(y)), 0.0
    X = sm.add_constant(np.log(v / 0.20))
    r = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    return float(r.params[0]), float(r.params[1])


def atm_ratio_params(p: SyntheticParams, as_of=None) -> tuple[float, float, str]:
    """(c0, c1, label) valid at ``as_of``.

    * From the first schedule entry on: the latest entry with from <= as_of
      (fitted only on PUT-index rolls settled before that year) -> 'PIT fit through ...'.
    * Before it: only rolls whose NEXT roll (settlement) is <= as_of are used; with
      n < ATM_RATIO_PRIOR_N such rolls the fit is blended with the prior
      ATM_RATIO_PRIOR (weight n/N) -> label starts with 'prior' and synthetic
      chains carry the fidelity reason 'atm_ratio_prior'."""
    sch = p.atm_ratio_schedule
    if not sch or as_of is None:
        return p.atm_ratio_index, p.atm_ratio_logvix, "static"
    d = clock.session_date_of(clock.to_utc(as_of)) if pd.Timestamp(as_of).tzinfo is not None else pd.Timestamp(as_of)
    ok = [e for e in sch if pd.Timestamp(e["from"]) <= d]
    if ok:
        e = ok[-1]
        return e["c0"], e["c1"], f"PIT fit through {e['fitted_through']}"
    obs = [o for o in getattr(p, "atm_ratio_obs", []) if pd.Timestamp(o["R2"]) <= d]
    c0p, c1p = ATM_RATIO_PRIOR
    if not obs:
        return c0p, c1p, "prior (no PUT-index roll settled before as_of)"
    c0f, c1f = _huber_ratio_fit(obs)
    w = min(len(obs) / ATM_RATIO_PRIOR_N, 1.0)
    last = max(pd.Timestamp(o["R2"]) for o in obs)
    return ((1 - w) * c0p + w * c0f, (1 - w) * c1p + w * c1f,
            f"prior-blend w={w:.2f} (n={len(obs)} rolls; fit through {last.date()})")


def atm_ratio(level, p: SyntheticParams, single: bool = False, as_of=None):
    """ATM-IV / vol-index ratio; for index products it depends on the index level:
    ratio = c0 + c1 ln(level/0.20), clipped to [0.6, 1.05]; (c0, c1) point-in-time."""
    if single:
        return p.atm_ratio_single
    c0, c1, _ = atm_ratio_params(p, as_of)
    lv = np.maximum(np.asarray(level, float), 0.03)
    return np.clip(c0 + c1 * np.log(lv / 0.20), 0.6, 1.05)


def atm_vol_at(term: pd.DataFrame, T: float) -> tuple[float, str]:
    """Total-variance interpolation of the ATM term structure at T (years);
    flat-vol extrapolation outside [shortest, longest] tenor (flagged)."""
    from atlasbt.pricing.surface import interp_total_variance
    if term.empty:
        return float("nan"), "none"
    return interp_total_variance(term["days"].values / 365.0, term["atm_vol"].values, max(T, 1e-6))


def smile_g(x, a: float, b: float, wing_put: float | None = None, wing_call: float | None = None):
    """Smile multiplier g(x): quadratic core 1 + a x + b x^2 on |x| <= XMAX,
    linear wings beyond with slopes ``wing_put`` (x < -XMAX) and ``wing_call``
    (x > XMAX) (default: tangent continuation), floored at G_FLOOR."""
    x = np.asarray(x, float)
    xc = np.clip(x, -XMAX, XMAX)
    g = 1 + a * xc + b * xc * xc
    sp = (a - 2 * b * XMAX) if wing_put is None else wing_put
    sc = (a + 2 * b * XMAX) if wing_call is None else wing_call
    g = g + np.where(x < -XMAX, sp * (x + XMAX), 0.0) + np.where(x > XMAX, sc * (x - XMAX), 0.0)
    return np.maximum(g, G_FLOOR)


def skew_params(underlying: str, skew_index: float, p: SyntheticParams, dte: float = 30.0) -> tuple:
    """(a, b, wing_put, wing_call) of the normalized smile at ``dte``:
    a(T) = a30 (DTE/30)^gamma (calibrated on per-expiry smiles); index products
    optionally shift a30 with Cboe SKEW (clipped to the calibration range;
    a1 = 0 unless significant)."""
    u = underlying.upper()
    d = max(float(dte), 1.0)
    if u in SPX_LIKE or u in ("QQQ", "NDX", "IWM", "RUT", "DIA", "DJX"):
        si = skew_index if np.isfinite(skew_index) else 130.0
        si = float(np.clip(si, p.skew_range[0], p.skew_range[1]))   # no extrapolation beyond calibration
        a30 = p.skew_a0 + p.skew_a1 * (si - 130.0) / 10.0
        return a30 * (d / 30.0) ** p.skew_gamma, p.skew_b, p.wing_put, p.wing_call
    return p.single_a * (d / 30.0) ** p.single_gamma, p.single_b, p.single_wing_put, p.single_wing_call


SPX_SPREAD_PROXY = ("SPY", 10.0)   # fallback only if SPX/SPXW were not in the calibration data


def params_fit_reasons(p: SyntheticParams, session) -> tuple[list, bool]:
    """(fidelity reasons, use_symbol_fe) for a session date.  Skew, wing and spread
    parameters were fitted on chains in ``p.fit_window``: sessions before the window
    END get 'params_fit_post_as_of'; before the window START the per-symbol spread
    fixed effects are not used (class defaults instead)."""
    d = pd.Timestamp(session).normalize()
    if d.tzinfo is not None:
        d = d.tz_convert("America/New_York").tz_localize(None).normalize()
    start, end = (pd.Timestamp(x) for x in p.fit_window)
    reasons = ["params_fit_post_as_of"] if d < end else []
    return reasons, d >= start


def model_spread(mid, dte, absdelta, underlying: str, p: SyntheticParams, is_etf: bool = False, vix: float = 0.20,
                 root: str | None = None, use_symbol_fe: bool = True):
    """Median quoted spread (ask - bid, $) from the calibrated log-linear model:
    log s = FE[key] + b1 log(mid) + b2 log(mid)^2 + b3 |delta| + b4 log(dte) + b5 log(VIX) (VIX decimal).
    key = option root if calibrated (SPX AM vs SPXW PM), else the underlying, else the
    class default (etf/stock).  If SPX family is uncalibrated, the SPY model is used on
    price/10 and scaled x10 (SPX_SPREAD_PROXY)."""
    u = underlying.upper()
    scale = 1.0
    if not use_symbol_fe:   # before the calibration window: no per-symbol fixed effects
        c = p.spread_coef
        fe = p.spread_default_fe["etf" if is_etf else "stock"]
        lm = np.log(np.maximum(np.asarray(mid, float), 0.005))
        ls = (fe + c.get("log_mid", 0.0) * lm + c.get("log_mid2", 0.0) * lm * lm
              + c.get("absdelta", 0.0) * np.asarray(absdelta, float)
              + c.get("log_dte", 0.0) * np.log(np.maximum(np.asarray(dte, float), 1.0))
              + c.get("log_vix", 0.0) * np.log(max(float(vix), 0.05)))
        return np.exp(ls)
    key = (root or u).upper()
    if key not in p.spread_symbol_fe and u in p.spread_symbol_fe:
        key = u
    if key not in p.spread_symbol_fe and u in ("SPX", "SPXW", "XSP"):
        key, scale = SPX_SPREAD_PROXY if u != "XSP" else ("SPY", 1.0)
    u = key
    c = p.spread_coef
    fe = p.spread_symbol_fe.get(u, p.spread_default_fe["etf" if is_etf else "stock"])
    lm = np.log(np.maximum(np.asarray(mid, float) / scale, 0.005))
    ls = (fe + c.get("log_mid", 0.0) * lm + c.get("log_mid2", 0.0) * lm * lm
          + c.get("absdelta", 0.0) * np.asarray(absdelta, float)
          + c.get("log_dte", 0.0) * np.log(np.maximum(np.asarray(dte, float), 1.0))
          + c.get("log_vix", 0.0) * np.log(max(float(vix), 0.05)))
    return np.exp(ls) * scale


# --------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------
@dataclass
class SyntheticChainGenerator:
    vol: VolSource = field(default_factory=VolSource)

    @property
    def params(self) -> SyntheticParams:
        return self.vol.params

    # ------------------------------------------------------------------ shared pricing
    def _price_strikes(self, u, spec, sess, spot, K, T, dte, r, q, F, atm, sk, right, is_etf, vix30, root, use_fe):
        """Model mid / iv / greeks / tick-rounded bid-ask for strikes K of one series
        (shared by ``chain`` and ``quote_contract`` so both are identical)."""
        K = np.asarray(K, float)
        Tm = max(T, 60.0 / (365 * 86400))           # smile coordinates need T > 0 (expiry-day close)
        a, b, wp, wc = skew_params(u, sk, self.params, Tm * 365.0)
        x = np.log(K / F) / (atm * np.sqrt(Tm))
        iv = atm * smile_g(x, a, b, wp, wc)
        cp = right == "C"
        if spec.exercise == "EUROPEAN":
            px = bsm_price(spot, K, T, r, q, iv, cp)
        else:
            px = bs2002_price(spot, K, T, r, q, iv, cp)
        px = np.atleast_1d(np.asarray(px, float))
        g = bsm_greeks(spot, K, Tm, r, q, iv, cp)
        spr = model_spread(px, max(dte, 1), np.abs(g.delta), u, self.params, is_etf, vix30, root=root,
                           use_symbol_fe=use_fe)
        tick = tick_size(u, px, sess)
        bid = np.floor((px - spr / 2) / tick + 1e-9) * tick
        ask = np.ceil((px + spr / 2) / tick - 1e-9) * tick
        ask = np.maximum(ask, bid + tick)
        bid = np.where(bid < tick - 1e-12, 0.0, bid)
        return {"mid": px, "iv": np.atleast_1d(iv), "bid": np.atleast_1d(bid), "ask": np.atleast_1d(ask),
                "greeks": g}

    def _context(self, u, as_of, rv_ratio):
        """PIT model inputs shared by chain/quote: term structure, SKEW, VIX, reasons."""
        sess = clock.session_date_of(as_of)
        term = self.vol.atm_term(u, as_of, rv_ratio)
        reasons = []
        ratio_label = atm_ratio_params(self.params, as_of)[2]
        uses_index_ratio = u in SPX_LIKE or u in ("QQQ", "NDX", "IWM", "RUT", "DIA", "DJX") or u not in INDEX_MAP
        if ratio_label.startswith("prior") and uses_index_ratio:
            reasons.append("atm_ratio_prior")
        if u in ("NDX", "RUT", "NDXP", "RUTW"):
            reasons.append("spread model uncalibrated for this index (no free quotes)")
        pf_reasons, use_fe = params_fit_reasons(self.params, sess)
        reasons += pf_reasons
        fam = self.vol.vix_family(as_of)
        vix30 = float(fam.loc[fam["index"] == "VIX", "level"].iloc[0]) if (fam["index"] == "VIX").any() else 0.20
        return dict(sess=sess, term=term, reasons=reasons, ratio_label=ratio_label, use_fe=use_fe,
                    sk=self.vol.skew_index(as_of), vix30=vix30)

    def quote_contract(self, underlying: str, as_of, spot: float, r: float, q: float, expiry, strike: float,
                       right: str, root: str | None = None, settlement: str | None = None,
                       rv_ratio: float | None = None, strict: bool = False) -> dict | None:
        """Synthetic quote of ONE contract at ``as_of`` with exactly the model used by
        ``chain`` (same series, smile, spread model, ticks).  Returns dict(bid, ask, mid,
        iv, delta, gamma, vega, theta, underlying_price, reasons) or None when the model
        has no inputs (or ``strict`` and the ATM ratio is a prior).  At/after the
        settlement instant returns None (the lifecycle settles the position)."""
        as_of = clock.to_utc(as_of)
        u = underlying.upper()
        spec = product_spec(u)
        ctx = self._context(u, as_of, rv_ratio)
        if ctx["term"].empty or (strict and "atm_ratio_prior" in ctx["reasons"]):
            return None
        st = settlement or ("CASH_PM" if spec.is_index else "PHYSICAL")
        T = year_fraction(as_of, pd.Timestamp(expiry), st)
        from atlasbt.pricing.time import settlement_instant
        if as_of > settlement_instant(pd.Timestamp(expiry), st):
            return None
        atm, _ = atm_vol_at(ctx["term"], max(T, 1e-6))
        if not np.isfinite(atm):
            return None
        F = spot * np.exp((r - q) * T)
        dte = (pd.Timestamp(expiry).normalize() - ctx["sess"]).days
        is_etf = (not spec.is_index) and spec.broad_based
        qd = self._price_strikes(u, spec, ctx["sess"], spot, [float(strike)], T, dte, r, q, F, atm, ctx["sk"],
                                 right.upper()[0], is_etf, ctx["vix30"], root, ctx["use_fe"])
        g = qd["greeks"]
        return {"bid": float(qd["bid"][0]), "ask": float(qd["ask"][0]), "mid": float(qd["mid"][0]),
                "iv": float(qd["iv"][0]), "delta": float(np.atleast_1d(g.delta)[0]),
                "gamma": float(np.atleast_1d(g.gamma)[0]), "vega": float(np.atleast_1d(g.vega)[0]),
                "theta": float(np.atleast_1d(g.theta)[0]), "underlying_price": float(spot),
                "reasons": list(ctx["reasons"])}

    def chain(self, underlying: str, as_of, spot: float, r: float, q: float = 0.0, max_dte: int = 400,
              min_dte: int = 0, rv_ratio: float | None = None, include_weeklies: bool | None = None,
              n_sd: float = 4.0, strict: bool = False) -> pd.DataFrame:
        """Canonical option_chain frame (fidelity T3_MODELED) at ``as_of`` (UTC).
        ``spot`` raw underlying price at as_of; ``r``/``q`` continuous.
        Returns an empty frame (attrs['reason']) if the vol inputs are missing.
        ``attrs['fidelity_reasons']`` lists caveats; 'atm_ratio_prior' when the ATM
        ratio still leans on the pre-sample prior (before 2010); with ``strict=True``
        such dates return an empty frame with Fidelity.NA instead."""
        as_of = clock.to_utc(as_of)
        sess = clock.session_date_of(as_of)
        u = underlying.upper()
        spec = product_spec(u)
        term = self.vol.atm_term(u, as_of, rv_ratio)
        cols = ["ts", "underlying", "symbol", "root", "expiry", "settlement_date", "kind", "strike", "right", "bid", "ask",
                "mid", "iv", "delta", "gamma", "vega", "theta", "underlying_price", "volume", "open_interest",
                "exercise", "settlement", "multiplier", "fidelity"]
        if term.empty:
            out = pd.DataFrame(columns=cols)
            out.attrs.update(fidelity=Fidelity.NA, source="synthetic", reason=f"no vol index inputs for {u} at {as_of}")
            return out
        ratio_label = atm_ratio_params(self.params, as_of)[2]
        reasons = []
        uses_index_ratio = u in SPX_LIKE or u in ("QQQ", "NDX", "IWM", "RUT", "DIA", "DJX") or u not in INDEX_MAP
        if ratio_label.startswith("prior") and uses_index_ratio:
            reasons.append("atm_ratio_prior")
            if strict:
                out = pd.DataFrame(columns=cols)
                out.attrs.update(fidelity=Fidelity.NA, source="synthetic", reason=f"atm_ratio_prior: {ratio_label}",
                                 fidelity_reasons=reasons)
                return out
        if u in ("NDX", "RUT", "NDXP", "RUTW"):
            reasons.append("spread model uncalibrated for this index (no free quotes)")
        pf_reasons, use_fe = params_fit_reasons(self.params, sess)
        reasons += pf_reasons
        sk = self.vol.skew_index(as_of)
        fam = self.vol.vix_family(as_of)
        vix30 = float(fam.loc[fam["index"] == "VIX", "level"].iloc[0]) if (fam["index"] == "VIX").any() else 0.20
        exps = listed_expiries(u, sess, horizon_days=max_dte, include_weeklies=include_weeklies)
        if exps.attrs.get("fidelity") == Fidelity.NA:
            out = pd.DataFrame(columns=cols)
            out.attrs.update(fidelity=Fidelity.NA, source="synthetic", reason=exps.attrs.get("reason", "no listing model"))
            return out
        rows = []
        is_etf = (not spec.is_index) and spec.broad_based
        from atlasbt.core.types import Instrument, Right, ExerciseStyle, SettlementType
        for e in exps.itertuples(index=False):
            T = year_fraction(as_of, e.expiry, e.settlement)
            dte = (e.expiry - sess).days
            if dte < min_dte or T <= 0:
                continue
            atm, _flag = atm_vol_at(term, T)
            if not np.isfinite(atm):
                continue
            F = spot * np.exp((r - q) * T)
            K = strike_grid(u, spot, sess, dte, atm, n_sd)
            for right in ("C", "P"):
                qd = self._price_strikes(u, spec, sess, spot, K, T, dte, r, q, F, atm, sk, right, is_etf, vix30,
                                         e.root, use_fe)
                px, iv, bid, ask = qd["mid"], qd["iv"], qd["bid"], qd["ask"]
                g = qd["greeks"]
                for j, k in enumerate(K):
                    ins = Instrument.option(spec.underlying if spec.is_index else u, e.expiry, float(k), Right(right),
                                            ExerciseStyle(e.exercise), SettlementType(e.settlement), root=e.root)
                    rows.append((as_of, u, ins.symbol, e.root, e.expiry, e.settlement_date, e.kind, float(k), right,
                                 float(bid[j]), float(ask[j]), float(px[j]), float(iv[j]), float(g.delta[j]),
                                 float(g.gamma[j]), float(g.vega[j]), float(g.theta[j]), float(spot), np.nan, np.nan,
                                 e.exercise, e.settlement, spec.multiplier, int(Fidelity.T3_MODELED)))
        out = pd.DataFrame(rows, columns=cols)
        out.attrs.update(fidelity=Fidelity.T3_MODELED, fidelity_reasons=reasons,
                         source=f"synthetic: listing calendar + {term['source'].iloc[0]} term structure x atm_ratio "
                                f"({atm_ratio_params(self.params, as_of)[2]}), "
                                f"smile a30={skew_params(u, sk, self.params, 30.0)[0]:.3f} (SKEW={sk}), calibrated spread model",
                         atm_term=term.to_dict("list"))
        return out
