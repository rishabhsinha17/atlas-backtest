"""Volatility surface utilities.

* ``smile_frame``: per-option IV/delta/log-moneyness for one snapshot (IV from
  mid via the European solver on the parity forward, i.e. Black-76 on F —
  for American chains use quoted IV or de-Americanized mids; see OPTIONS.md).
* ``atm_iv_per_expiry``: IV at k = ln(K/F) = 0, linear interpolation of total
  variance in k between the two bracketing OTM quotes.
* ``constant_maturity_iv``: interpolation **linear in total variance**
  w = sigma^2 T across expiries (calendar-arbitrage-consistent); outside the
  expiry range the nearest vol is held flat and the result is flagged.
* ``iv_at_delta`` / ``constant_maturity_delta_iv`` / ``skew_25d``: IV at a
  fixed BSM delta (e.g. 25-delta put / call), interpolated in delta per expiry
  and in total variance across expiries.
* SVI raw parametrisation per expiry (Gatheral 2004) with no-arbitrage sanity
  checks (Lee moment bound, Gatheral-Jacquier butterfly density g(k) >= 0,
  calendar monotonicity of total variance); fallback: smoothing spline of IV
  in call-delta space.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares

from .bsm import as_is_call, bsm_delta
from .iv import implied_vol


# --------------------------------------------------------------------------
# smile extraction
# --------------------------------------------------------------------------
def smile_frame(chain: pd.DataFrame, forwards: pd.DataFrame, use_quoted_iv: bool = False,
                price_col: str = "mid") -> pd.DataFrame:
    """Attach F, T, r, k=ln(K/F), iv_fit and bsm delta (on F) to each row.

    ``forwards`` from ``atlasbt.pricing.forward.infer_forwards`` (expiry,T,r,F).
    OTM side only is kept for fitting (puts K<F, calls K>=F) in ``otm`` column.
    """
    df = chain.merge(forwards[["expiry", "T", "r", "F"]].rename(columns={"T": "T_fwd"}), on="expiry", how="inner")
    if "T" not in df.columns:
        df["T"] = df["T_fwd"]
    df = df.drop(columns=["T_fwd"])
    df = df[np.isfinite(df["F"]) & (df["T"] > 0)].copy()
    cp = as_is_call(df["right"].astype(str).str.upper().str[0].values)
    if use_quoted_iv and "iv" in df.columns:
        iv = pd.to_numeric(df["iv"], errors="coerce").values
    else:
        px = df[price_col].values if price_col in df.columns else 0.5 * (df["bid"].values + df["ask"].values)
        iv = implied_vol(px, df["F"].values, df["strike"].values, df["T"].values, df["r"].values, 0.0, cp,
                         model="black76").iv
    df["iv_fit"] = iv
    df["k"] = np.log(df["strike"].values / df["F"].values)
    df["delta_f"] = bsm_delta(df["F"].values, df["strike"].values, df["T"].values, 0.0, 0.0,
                              np.where(np.isfinite(iv), iv, np.nan), cp)
    df["otm"] = np.where(cp, df["k"].values >= 0, df["k"].values < 0)
    return df


def _interp_w_at_k(k, w, k0=0.0):
    o = np.argsort(k)
    k, w = k[o], w[o]
    if len(k) == 0:
        return np.nan, "none"
    if k0 < k[0] or k0 > k[-1]:
        j = 0 if k0 < k[0] else -1
        return w[j], "extrap_flat"
    return float(np.interp(k0, k, w)), "interp"


def atm_iv_per_expiry(smile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exp, g in smile.groupby("expiry"):
        g = g[g["otm"] & np.isfinite(g["iv_fit"])]
        T = float(g["T"].iloc[0]) if len(g) else np.nan
        if len(g) == 0:
            rows.append(dict(expiry=exp, T=T, atm_iv=np.nan, flag="none"))
            continue
        w = g["iv_fit"].values ** 2 * T
        wk, flag = _interp_w_at_k(g["k"].values, w, 0.0)
        rows.append(dict(expiry=exp, T=T, atm_iv=float(np.sqrt(wk / T)), flag=flag))
    return pd.DataFrame(rows)


def interp_total_variance(Ts, vols, T_target: float) -> tuple[float, str]:
    """Constant-maturity vol at ``T_target`` from (T_i, vol_i): linear in total
    variance between bracketing points; flat vol extrapolation (flagged)."""
    Ts, vols = np.asarray(Ts, float), np.asarray(vols, float)
    m = np.isfinite(Ts) & np.isfinite(vols) & (Ts > 0)
    Ts, vols = Ts[m], vols[m]
    if len(Ts) == 0:
        return float("nan"), "none"
    o = np.argsort(Ts)
    Ts, vols = Ts[o], vols[o]
    if T_target <= Ts[0]:
        return float(vols[0]), ("exact" if np.isclose(T_target, Ts[0]) else "extrap_flat")
    if T_target >= Ts[-1]:
        return float(vols[-1]), ("exact" if np.isclose(T_target, Ts[-1]) else "extrap_flat")
    w = vols ** 2 * Ts
    wt = float(np.interp(T_target, Ts, w))
    return float(np.sqrt(wt / T_target)), "interp"


def constant_maturity_iv(atm: pd.DataFrame, days: float) -> tuple[float, str]:
    """e.g. 30D / 60D ATM IV from ``atm_iv_per_expiry`` output."""
    return interp_total_variance(atm["T"].values, atm["atm_iv"].values, days / 365.0)


def iv_at_delta(smile_expiry: pd.DataFrame, target_delta: float) -> float:
    """IV at a BSM delta (on F) within one expiry.  target_delta>0 -> calls,
    <0 -> puts.  Linear interpolation of IV vs delta on the OTM wing of that
    right; NaN if the target is outside the quoted delta range."""
    is_c = target_delta > 0
    g = smile_expiry[(smile_expiry["right"].astype(str).str.upper().str[0] == ("C" if is_c else "P"))
                     & np.isfinite(smile_expiry["iv_fit"]) & np.isfinite(smile_expiry["delta_f"])]
    if len(g) < 2:
        return float("nan")
    d, v = g["delta_f"].values, g["iv_fit"].values
    o = np.argsort(d)
    d, v = d[o], v[o]
    if target_delta < d[0] or target_delta > d[-1]:
        return float("nan")
    return float(np.interp(target_delta, d, v))


def constant_maturity_delta_iv(smile: pd.DataFrame, target_delta: float, days: float) -> tuple[float, str]:
    Ts, vs = [], []
    for exp, g in smile.groupby("expiry"):
        v = iv_at_delta(g, target_delta)
        if np.isfinite(v):
            Ts.append(float(g["T"].iloc[0]))
            vs.append(v)
    return interp_total_variance(Ts, vs, days / 365.0)


def skew_25d(smile: pd.DataFrame, days: float = 30.0) -> tuple[float, str]:
    """IV(25-delta put) - IV(25-delta call) at constant maturity (S166/S167)."""
    p, f1 = constant_maturity_delta_iv(smile, -0.25, days)
    c, f2 = constant_maturity_delta_iv(smile, 0.25, days)
    flag = f1 if f1 != "interp" else f2
    return p - c, flag


# --------------------------------------------------------------------------
# SVI
# --------------------------------------------------------------------------
@dataclass
class SVIFit:
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    T: float
    rmse_vol: float
    ok: bool
    issues: list = field(default_factory=list)

    def w(self, k):
        k = np.asarray(k, float)
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma ** 2))

    def iv(self, k):
        return np.sqrt(np.maximum(self.w(k), 0.0) / self.T)


def svi_w(params, k):
    a, b, rho, m, s = params
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s * s))


def svi_g(fit: SVIFit, k):
    """Gatheral-Jacquier butterfly density function g(k); arbitrage-free iff g>=0."""
    k = np.asarray(k, float)
    a, b, rho, m, s = fit.a, fit.b, fit.rho, fit.m, fit.sigma
    x = k - m
    sq = np.sqrt(x * x + s * s)
    w = a + b * (rho * x + sq)
    w1 = b * (rho + x / sq)
    w2 = b * s * s / sq ** 3
    return (1 - k * w1 / (2 * w)) ** 2 - (w1 ** 2 / 4) * (1 / w + 0.25) + w2 / 2


def fit_svi(k, iv, T: float, weights=None) -> SVIFit:
    """Fit raw SVI to (k, iv) of one expiry by least squares in total variance."""
    k = np.asarray(k, float)
    iv = np.asarray(iv, float)
    m_ = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    k, iv = k[m_], iv[m_]
    wts = np.ones_like(k) if weights is None else np.asarray(weights, float)[m_]
    issues = []
    if len(k) < 5:
        return SVIFit(np.nan, np.nan, np.nan, np.nan, np.nan, T, np.nan, False, ["too_few_points"])
    w = iv ** 2 * T
    wmin = float(np.min(w))
    x0 = [max(wmin * 0.5, 1e-6), 0.1, -0.3, 0.0, 0.1]
    lb = [-np.max(w), 1e-6, -0.999, min(k.min(), -1.0), 1e-4]
    ub = [np.max(w) * 2, 4.0, 0.999, max(k.max(), 1.0), 5.0]

    def resid(p):
        return (svi_w(p, k) - w) * np.sqrt(wts)

    best = None
    for rho0 in (-0.6, -0.3, 0.0):
        for s0 in (0.05, 0.2):
            x0[2], x0[4] = rho0, s0
            try:
                r = least_squares(resid, x0, bounds=(lb, ub), method="trf", max_nfev=2000)
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
    if best is None:
        return SVIFit(np.nan, np.nan, np.nan, np.nan, np.nan, T, np.nan, False, ["fit_failed"])
    a, b, rho, m, s = best.x
    fit = SVIFit(a, b, rho, m, s, T, 0.0, True, issues)
    fit.rmse_vol = float(np.sqrt(np.mean((fit.iv(k) - iv) ** 2)))
    # sanity / no-arbitrage checks
    if a + b * s * np.sqrt(1 - rho * rho) < 0:
        issues.append("negative_min_variance")
    if b * (1 + abs(rho)) > 4.0:
        issues.append("lee_moment_bound")
    grid = np.linspace(min(k.min(), -0.5), max(k.max(), 0.5), 201)
    if np.min(svi_g(fit, grid)) < -1e-8:
        issues.append("butterfly_arbitrage")
    if fit.rmse_vol > 0.02:
        issues.append("poor_fit")
    fit.ok = len(issues) == 0
    return fit


def calendar_arbitrage_free(fits: list[SVIFit], k_grid=None, tol: float = 1e-8) -> bool:
    """Total variance must be non-decreasing in T at every k."""
    k_grid = np.linspace(-0.5, 0.5, 101) if k_grid is None else np.asarray(k_grid)
    fits = sorted([f for f in fits if np.isfinite(f.a)], key=lambda f: f.T)
    for f1, f2 in zip(fits[:-1], fits[1:]):
        if np.any(f2.w(k_grid) < f1.w(k_grid) - tol):
            return False
    return True


@dataclass
class SmileModel:
    """Per-expiry smile: SVI if it passes the checks, else smoothing spline of
    IV in call-delta space (delta on F)."""

    kind: str               # 'svi' | 'spline_delta' | 'none'
    T: float
    svi: SVIFit | None = None
    spline: object | None = None

    def iv_k(self, k):
        if self.kind == "svi":
            return self.svi.iv(k)
        return np.full(np.shape(k), np.nan)

    def iv_call_delta(self, cd):
        if self.kind == "spline_delta":
            return self.spline(np.asarray(cd, float))
        return np.full(np.shape(cd), np.nan)


def fit_smile(smile_expiry: pd.DataFrame) -> SmileModel:
    g = smile_expiry[smile_expiry["otm"] & np.isfinite(smile_expiry["iv_fit"])]
    T = float(smile_expiry["T"].iloc[0]) if len(smile_expiry) else np.nan
    if len(g) >= 5:
        f = fit_svi(g["k"].values, g["iv_fit"].values, T)
        if f.ok:
            return SmileModel("svi", T, svi=f)
    if len(g) >= 4:
        cd = np.where(g["right"].astype(str).str.upper().str[0] == "C", g["delta_f"], 1.0 + g["delta_f"])
        o = np.argsort(cd)
        cd, v = np.asarray(cd)[o], g["iv_fit"].values[o]
        keep = np.r_[True, np.diff(cd) > 1e-9]
        if keep.sum() >= 4:
            sp = UnivariateSpline(cd[keep], v[keep], k=3, s=len(v[keep]) * (0.005 ** 2))
            return SmileModel("spline_delta", T, spline=sp)
    return SmileModel("none", T)
