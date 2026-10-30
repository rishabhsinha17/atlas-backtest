"""European option pricing: Black-Scholes-Merton (continuous yield), escrowed
discrete-dividend BSM, and Black-76.  All functions are numpy-vectorized and
broadcast their inputs.

Conventions (see docs/OPTIONS.md "Conventions"):
  * ``T``      year fraction to expiry/settlement, ACT/365F (calendar days/365;
               intraday: minutes/(365*1440)).  Use ``atlasbt.pricing.time``.
  * ``r``      continuously-compounded risk-free rate (decimal).
  * ``q``      continuously-compounded dividend yield (decimal).  Use the
               escrowed model (``pv_dividends`` + ``q=0``) for discrete cash
               dividends.
  * ``sigma``  annualised volatility (decimal, 0.20 == 20 vol points).
  * ``is_call`` boolean (array) or 'C'/'P' strings (array).
  * Prices are per unit of underlying (per share / per index point); multiply
    by the contract multiplier for per-contract values.

Greek units returned by ``bsm_greeks``/``black76_greeks`` (market convention,
identical to the DoltHub chain columns):
  delta per $1, gamma per $1 per $1, vega per 1 vol point (0.01),
  theta per calendar day, rho per 1 rate point (0.01).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

_SQRT2PI = np.sqrt(2.0 * np.pi)


def _npdf(x):
    return np.exp(-0.5 * x * x) / _SQRT2PI


def as_is_call(right) -> np.ndarray:
    """Normalise 'C'/'P'/True/False (scalar or array) to a boolean ndarray."""
    a = np.asarray(right)
    if a.dtype == bool:
        return a
    if a.dtype.kind in "iuf":
        return a.astype(bool)
    s = np.char.upper(a.astype(str))
    out = np.char.startswith(s, "C")
    bad = ~(out | np.char.startswith(s, "P"))
    if np.any(bad):
        raise ValueError(f"right must be 'C'/'P' (got {np.unique(a[bad]) if a.ndim else a})")
    return out


def _bcast(*args):
    return np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in args])


# --------------------------------------------------------------------------
# BSM with continuous dividend yield (generalised: cost of carry b = r - q)
# --------------------------------------------------------------------------
def bsm_d1d2(S, K, T, r, q, sigma):
    S, K, T, r, q, sigma = _bcast(S, K, T, r, q, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        vsqt = sigma * np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vsqt
        d2 = d1 - vsqt
    return d1, d2


def bsm_price(S, K, T, r, q, sigma, is_call):
    """European BSM price.  T<=0 or sigma<=0 return discounted intrinsic of the
    forward (the deterministic limit)."""
    cp = as_is_call(is_call)
    S, K, T, r, q, sigma, cpb = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in (S, K, T, r, q, sigma)], cp)
    cpb = cpb.astype(bool)
    Tpos = np.where(T > 0, T, 0.0)
    dfq = np.exp(-q * Tpos)
    dfr = np.exp(-r * Tpos)
    degenerate = (T <= 0) | (sigma <= 0)
    d1, d2 = bsm_d1d2(S, K, np.where(degenerate, 1.0, T), r, q, np.where(degenerate, 1.0, sigma))
    call = S * dfq * ndtr(d1) - K * dfr * ndtr(d2)
    put = K * dfr * ndtr(-d2) - S * dfq * ndtr(-d1)
    px = np.where(cpb, call, put)
    fwd_intr = np.where(cpb, np.maximum(S * dfq - K * dfr, 0.0), np.maximum(K * dfr - S * dfq, 0.0))
    px = np.where(degenerate, fwd_intr, px)
    return px[()] if px.ndim == 0 else px


@dataclass(frozen=True)
class Greeks:
    """Per-unit greeks (see module docstring for units)."""

    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray     # per 1 vol point
    theta: np.ndarray    # per calendar day
    rho: np.ndarray      # per 1 rate point

    def per_contract(self, multiplier: float = 100.0) -> "Greeks":
        m = float(multiplier)
        return Greeks(self.price * m, self.delta * m, self.gamma * m, self.vega * m,
                      self.theta * m, self.rho * m)

    def as_dict(self) -> dict:
        return dict(price=self.price, delta=self.delta, gamma=self.gamma, vega=self.vega,
                    theta=self.theta, rho=self.rho)


def bsm_greeks(S, K, T, r, q, sigma, is_call) -> Greeks:
    cp = as_is_call(is_call)
    S, K, T, r, q, sigma, cpb = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in (S, K, T, r, q, sigma)], cp)
    cpb = cpb.astype(bool)
    Tc = np.maximum(T, 1e-12)
    sc = np.maximum(sigma, 1e-12)
    d1, d2 = bsm_d1d2(S, K, Tc, r, q, sc)
    dfq = np.exp(-q * Tc)
    dfr = np.exp(-r * Tc)
    pdf1 = _npdf(d1)
    sqt = np.sqrt(Tc)
    price = bsm_price(S, K, T, r, q, sigma, cpb)
    delta = np.where(cpb, dfq * ndtr(d1), dfq * (ndtr(d1) - 1.0))
    gamma = dfq * pdf1 / (S * sc * sqt)
    vega = S * dfq * pdf1 * sqt
    common = -S * dfq * pdf1 * sc / (2.0 * sqt)
    theta_c = common - r * K * dfr * ndtr(d2) + q * S * dfq * ndtr(d1)
    theta_p = common + r * K * dfr * ndtr(-d2) - q * S * dfq * ndtr(-d1)
    theta = np.where(cpb, theta_c, theta_p)
    rho = np.where(cpb, K * Tc * dfr * ndtr(d2), -K * Tc * dfr * ndtr(-d2))
    return Greeks(np.asarray(price), delta, gamma, vega / 100.0, theta / 365.0, rho / 100.0)


def bsm_delta(S, K, T, r, q, sigma, is_call):
    cp = as_is_call(is_call)
    d1, _ = bsm_d1d2(S, K, np.maximum(np.asarray(T, float), 1e-12), r, q, np.maximum(np.asarray(sigma, float), 1e-12))
    dfq = np.exp(-np.asarray(q, float) * np.maximum(np.asarray(T, float), 0))
    return np.where(cp, dfq * ndtr(d1), dfq * (ndtr(d1) - 1.0))


def strike_from_delta(delta, S, T, r, q, sigma, is_call):
    """Strike whose BSM (spot, premium-unadjusted) delta equals ``delta``
    (calls: 0<delta<e^{-qT}; puts: -e^{-qT}<delta<0)."""
    from scipy.special import ndtri
    cp = as_is_call(is_call)
    delta, S, T, r, q, sigma = _bcast(delta, S, T, r, q, sigma)
    dfq = np.exp(-q * T)
    x = np.where(cp, delta / dfq, delta / dfq + 1.0)
    d1 = ndtri(np.clip(x, 1e-12, 1 - 1e-12))
    return S * np.exp(-(d1 * sigma * np.sqrt(T)) + (r - q + 0.5 * sigma ** 2) * T)


# --------------------------------------------------------------------------
# Discrete cash dividends: escrowed-dividend model
# --------------------------------------------------------------------------
def pv_dividends(div_times, div_amounts, T, r) -> float:
    """Present value of cash dividends with ex-dates in (0, T]
    (times in years from valuation, ACT/365F).  Scalar helper."""
    t = np.asarray(div_times, dtype=float).ravel()
    d = np.asarray(div_amounts, dtype=float).ravel()
    m = (t > 0) & (t <= T)
    return float(np.sum(d[m] * np.exp(-r * t[m])))


def bsm_price_discrete_div(S, K, T, r, sigma, is_call, div_times=(), div_amounts=()):
    """European BSM with escrowed discrete dividends (Hull ch. 15/17):
    S* = S - PV(dividends with ex-date before expiry), then BSM with q=0."""
    T = float(T)
    S_star = np.asarray(S, dtype=float) - pv_dividends(div_times, div_amounts, T, float(r))
    return bsm_price(S_star, K, T, r, 0.0, sigma, is_call)


def bsm_greeks_discrete_div(S, K, T, r, sigma, is_call, div_times=(), div_amounts=()) -> Greeks:
    T = float(T)
    S_star = np.asarray(S, dtype=float) - pv_dividends(div_times, div_amounts, T, float(r))
    return bsm_greeks(S_star, K, T, r, 0.0, sigma, is_call)


# --------------------------------------------------------------------------
# Black-76 (options on forwards/futures; also European index options given F)
# --------------------------------------------------------------------------
def black76_price(F, K, T, r, sigma, is_call):
    """Black (1976): c = e^{-rT}[F N(d1) - K N(d2)].  Equals BSM with S=F, q=r."""
    return bsm_price(F, K, T, r, r, sigma, is_call)


def black76_greeks(F, K, T, r, sigma, is_call) -> Greeks:
    """Greeks w.r.t. the forward F (delta = e^{-rT}N(d1) for calls).  theta is
    the derivative w.r.t. calendar time holding F fixed; rho holds F fixed."""
    cp = as_is_call(is_call)
    F, K, T, r, sigma, cpb = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in (F, K, T, r, sigma)], cp)
    cpb = cpb.astype(bool)
    Tc = np.maximum(T, 1e-12)
    sc = np.maximum(sigma, 1e-12)
    d1, d2 = bsm_d1d2(F, K, Tc, r, r, sc)
    df = np.exp(-r * Tc)
    pdf1 = _npdf(d1)
    sqt = np.sqrt(Tc)
    price = black76_price(F, K, T, r, sigma, cpb)
    delta = np.where(cpb, df * ndtr(d1), df * (ndtr(d1) - 1.0))
    gamma = df * pdf1 / (F * sc * sqt)
    vega = F * df * pdf1 * sqt
    theta = -F * df * pdf1 * sc / (2 * sqt) + r * price
    rho = -Tc * price
    return Greeks(np.asarray(price), delta, gamma, vega / 100.0, theta / 365.0, rho / 100.0)


# --------------------------------------------------------------------------
# Bounds (used by the IV solver and parity screens)
# --------------------------------------------------------------------------
def european_bounds(S, K, T, r, q, is_call):
    """(lower, upper) no-arbitrage bounds for a European option."""
    cp = as_is_call(is_call)
    S, K, T, r, q = _bcast(S, K, T, r, q)
    dfq, dfr = np.exp(-q * T), np.exp(-r * T)
    lo = np.where(cp, np.maximum(S * dfq - K * dfr, 0), np.maximum(K * dfr - S * dfq, 0))
    hi = np.where(cp, S * dfq, K * dfr)
    return lo, hi
