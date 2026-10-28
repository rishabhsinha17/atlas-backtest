"""Implied volatility.

``implied_vol`` (European BSM / Black-76) is vectorized: a safeguarded
Newton-bisection iteration on a maintained bracket [lo, hi] (a Newton step that
leaves the bracket is replaced by bisection, so it cannot diverge), followed by
``scipy.optimize.brentq`` on [VOL_MIN, VOL_MAX] for any element that did not
converge.  Quotes outside no-arbitrage bounds return NaN with a reason code:

    OK, BELOW_INTRINSIC, ABOVE_MAX, NONPOSITIVE_PRICE, EXPIRED,
    VOL_BELOW_MIN, VOL_ABOVE_MAX, NO_CONVERGENCE, BAD_INPUT

``american_implied_vol`` inverts an American pricer directly (Brent per
element; 'bs2002' fast default, 'crr' for discrete dividends).
``deamericanize`` converts an American quote into the European-equivalent
price at the same implied vol (used for parity/forward inference).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .american import american_price
from .bsm import as_is_call, bsm_d1d2, bsm_price, european_bounds, _npdf

VOL_MIN = 1e-4
VOL_MAX = 5.0

OK = "OK"
BELOW_INTRINSIC = "BELOW_INTRINSIC"
ABOVE_MAX = "ABOVE_MAX"
NONPOSITIVE_PRICE = "NONPOSITIVE_PRICE"
EXPIRED = "EXPIRED"
VOL_BELOW_MIN = "VOL_BELOW_MIN"
VOL_ABOVE_MAX = "VOL_ABOVE_MAX"
NO_CONVERGENCE = "NO_CONVERGENCE"
BAD_INPUT = "BAD_INPUT"


@dataclass(frozen=True)
class IVResult:
    iv: np.ndarray        # NaN where not solvable
    reason: np.ndarray    # str codes (see module docstring)

    @property
    def ok(self) -> np.ndarray:
        return self.reason == OK


def _vega_raw(S, K, T, r, q, sigma):
    d1, _ = bsm_d1d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * _npdf(d1) * np.sqrt(T)


def implied_vol(price, S, K, T, r, q, is_call, model: str = "bsm", tol: float = 1e-10,
                max_iter: int = 100, intrinsic_tol: float = 0.0) -> IVResult:
    """European implied volatility.

    model='bsm': S is spot, q continuous yield.  model='black76': S is the
    forward F (q is ignored and set to r).  ``intrinsic_tol`` lets quotes that are
    below the lower bound by at most this amount be solved at VOL_MIN instead of
    rejected (useful for $0.01-rounded deep ITM quotes); default strict.
    """
    cp = as_is_call(is_call)
    price, S, K, T, r, q, cpb = np.broadcast_arrays(*[np.asarray(a, float) for a in (price, S, K, T, r, q)], cp)
    cpb = cpb.astype(bool)
    if model == "black76":
        q = r.copy()
    elif model != "bsm":
        raise ValueError(model)
    shape = price.shape
    price, S, K, T, r, q, cpb = (a.ravel() for a in (price, S, K, T, r, q, cpb))
    n = price.size
    iv = np.full(n, np.nan)
    reason = np.full(n, OK, dtype=object)

    bad = ~np.isfinite(price) | ~np.isfinite(S) | ~np.isfinite(K) | ~np.isfinite(T) | (S <= 0) | (K <= 0)
    reason[bad] = BAD_INPUT
    reason[~bad & (T <= 0)] = EXPIRED
    reason[(reason == OK) & (price <= 0)] = NONPOSITIVE_PRICE
    lo_b, hi_b = european_bounds(S, K, np.maximum(T, 0), r, q, cpb)
    m = reason == OK
    below = m & (price < lo_b - intrinsic_tol)
    reason[below] = BELOW_INTRINSIC
    above = m & (price >= hi_b)
    reason[above] = ABOVE_MAX
    work = np.where(reason == OK)[0]
    if work.size == 0:
        return IVResult(iv.reshape(shape), reason.reshape(shape))

    P, s_, k_, t_, r_, q_, c_ = (a[work] for a in (price, S, K, T, r, q, cpb))
    pmin = bsm_price(s_, k_, t_, r_, q_, VOL_MIN, c_)
    pmax = bsm_price(s_, k_, t_, r_, q_, VOL_MAX, c_)
    res = np.full(work.size, np.nan)
    rsn = np.full(work.size, OK, dtype=object)
    lowv = P <= pmin
    # within intrinsic tolerance or genuinely below min vol
    rsn[lowv] = VOL_BELOW_MIN
    res[lowv & (P >= lo_b[work] - intrinsic_tol) & (intrinsic_tol > 0)] = VOL_MIN
    rsn[lowv & np.isfinite(res)] = OK
    highv = P >= pmax
    rsn[highv] = VOL_ABOVE_MAX
    act = ~(lowv | highv)

    lo = np.full(work.size, VOL_MIN)
    hi = np.full(work.size, VOL_MAX)
    # Brenner-Subrahmanyam / Corrado-Miller style start
    x = np.sqrt(2 * np.pi / np.maximum(t_, 1e-12)) * P / np.maximum(s_, 1e-12)
    x = np.clip(np.where(np.isfinite(x), x, 0.3), 0.05, 2.0)
    conv = np.zeros(work.size, dtype=bool)
    for _ in range(max_iter):
        a = act & ~conv
        if not np.any(a):
            break
        pv = bsm_price(s_[a], k_[a], t_[a], r_[a], q_[a], x[a], c_[a])
        diff = pv - P[a]
        done = np.abs(diff) < tol * np.maximum(1.0, P[a])
        idx = np.where(a)[0]
        conv[idx[done]] = True
        res[idx[done]] = x[idx[done]]
        # update bracket
        hi[idx] = np.where(diff > 0, x[idx], hi[idx])
        lo[idx] = np.where(diff <= 0, x[idx], lo[idx])
        vg = _vega_raw(s_[a], k_[a], t_[a], r_[a], q_[a], x[a])
        with np.errstate(all="ignore"):
            newton = x[idx] - diff / vg
        inside = np.isfinite(newton) & (newton > lo[idx]) & (newton < hi[idx])
        x[idx] = np.where(inside, newton, 0.5 * (lo[idx] + hi[idx]))
        # bracket collapsed
        tiny = (hi[idx] - lo[idx]) < 1e-14
        conv[idx[tiny]] = True
        res[idx[tiny]] = x[idx[tiny]]
    # Brent fallback
    left = act & ~conv
    for j in np.where(left)[0]:
        f = lambda v, j=j: float(bsm_price(s_[j], k_[j], t_[j], r_[j], q_[j], v, c_[j])) - P[j]  # noqa: E731
        try:
            res[j] = brentq(f, VOL_MIN, VOL_MAX, xtol=1e-12, maxiter=200)
        except Exception:
            rsn[j] = NO_CONVERGENCE
    res[rsn != OK] = np.nan
    iv[work] = res
    reason[work] = rsn
    return IVResult(iv.reshape(shape), reason.reshape(shape))


def american_implied_vol(price, S, K, T, r, q, is_call, method: str = "bs2002", steps: int = 200,
                         div_times=(), div_amounts=()) -> IVResult:
    """American IV by direct inversion of the American pricer (Brent per element).
    Lower bound is intrinsic (immediate exercise), upper bound S (call) / K (put)."""
    cp = as_is_call(is_call)
    price, S, K, T, r, q, cpb = np.broadcast_arrays(*[np.asarray(a, float) for a in (price, S, K, T, r, q)], cp)
    shape = price.shape
    price, S, K, T, r, q, cpb = (a.ravel() for a in (price, S, K, T, r, q, cpb.astype(bool)))
    iv = np.full(price.size, np.nan)
    reason = np.full(price.size, OK, dtype=object)
    for j in range(price.size):
        p, s, k, t, rr, qq, c = price[j], S[j], K[j], T[j], r[j], q[j], bool(cpb[j])
        if not all(np.isfinite([p, s, k, t])) or s <= 0 or k <= 0:
            reason[j] = BAD_INPUT
            continue
        if t <= 0:
            reason[j] = EXPIRED
            continue
        if p <= 0:
            reason[j] = NONPOSITIVE_PRICE
            continue
        intr = max(s - k, 0.0) if c else max(k - s, 0.0)
        if p < intr:
            reason[j] = BELOW_INTRINSIC
            continue
        if p >= (s if c else k):
            reason[j] = ABOVE_MAX
            continue
        f = lambda v: float(american_price(s, k, t, rr, qq, v, c, method, steps, div_times, div_amounts)) - p  # noqa: E731
        flo, fhi = f(VOL_MIN), f(VOL_MAX)
        if flo >= 0:
            reason[j] = VOL_BELOW_MIN
            continue
        if fhi <= 0:
            reason[j] = VOL_ABOVE_MAX
            continue
        try:
            iv[j] = brentq(f, VOL_MIN, VOL_MAX, xtol=1e-8, maxiter=200)
        except Exception:
            reason[j] = NO_CONVERGENCE
    return IVResult(iv.reshape(shape), reason.reshape(shape))


def deamericanize(price, S, K, T, r, q, is_call, method: str = "bs2002"):
    """European-equivalent price of an American quote: invert American IV then
    price European at that IV.  Returns (euro_price, IVResult)."""
    res = american_implied_vol(price, S, K, T, r, q, is_call, method=method)
    eu = bsm_price(S, K, T, r, q, np.where(res.ok, res.iv, np.nan), is_call)
    return eu, res
