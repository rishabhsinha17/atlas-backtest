"""American option pricing.

* ``crr_price``: Cox-Ross-Rubinstein binomial tree, continuous yield ``q`` and/or
  discrete cash dividends (escrowed-dividend tree: the lattice is built on
  S* = S - PV(dividends before expiry) and the PV of not-yet-paid dividends is
  added back at every node for the early-exercise test, Hull ch. 21).
  Odd/even oscillation is damped by averaging N and N+1 steps (``smooth``).
  Uses numba if importable, else a numpy kernel (identical results).
* ``bs2002_price``: Bjerksund & Stensland (2002) closed-form approximation
  (continuous carry only), port of Haug (2007) "The Complete Guide to Option
  Pricing Formulas", 2nd ed., ch. 3.  Puts via the put-call transformation
  P(S,K,T,r,b,v) = C(K,S,T,r-b,-b,v).
* ``american_greeks``: central finite differences of either pricer.

Inputs follow atlasbt.pricing.bsm conventions (r, q continuous; T ACT/365F).
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .bsm import Greeks, as_is_call, bsm_price

try:  # optional acceleration
    import numba as _nb  # type: ignore

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False


# --------------------------------------------------------------------------
# Bivariate normal CDF (Genz 2004 / Drezner-Wesolowsky, as in Haug 2007 "CBND")
# --------------------------------------------------------------------------
_GL_X, _GL_W = np.polynomial.legendre.leggauss(20)


def cbnd(a, b, rho):
    """P(X <= a, Y <= b) for standard bivariate normal with correlation rho.
    Vectorized; accuracy ~1e-14 (20-point Gauss-Legendre)."""
    a, b, rho = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float), np.asarray(rho, float))
    out = np.empty(a.shape, dtype=float)
    h = -a
    k = -b
    lo = np.abs(rho) < 0.925
    if np.any(lo):
        hl, kl, rl = h[lo], k[lo], rho[lo]
        hk = hl * kl
        hs = (hl * hl + kl * kl) / 2.0
        asr = np.arcsin(rl)
        sn = np.sin(asr[:, None] * (_GL_X[None, :] + 1.0) / 2.0)
        s = np.sum(_GL_W[None, :] * np.exp((sn * hk[:, None] - hs[:, None]) / (1.0 - sn * sn)), axis=1)
        out[lo] = s * asr / (4.0 * np.pi) + ndtr(-hl) * ndtr(-kl)
    hi = ~lo
    if np.any(hi):
        hh, kh, rh = h[hi], k[hi].copy(), rho[hi]
        kh = np.where(rh < 0, -kh, kh)
        hk = hh * kh
        bvn = np.zeros_like(hh)
        lt1 = np.abs(rh) < 1.0
        ass = (1.0 - rh) * (1.0 + rh)
        aa = np.sqrt(np.maximum(ass, 0.0))
        bs = (hh - kh) ** 2
        c = (4.0 - hk) / 8.0
        d = (12.0 - hk) / 16.0
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            asr = -(bs / ass + hk) / 2.0
            t1 = np.where(asr > -100, aa * np.exp(asr) * (1 - c * (bs - ass) * (1 - d * bs / 5) / 3 + c * d * ass * ass / 5), 0.0)
            bb = np.sqrt(bs)
            t2 = np.where(-hk < 100, np.exp(-hk / 2) * np.sqrt(2 * np.pi) * ndtr(-bb / aa) * bb * (1 - c * bs * (1 - d * bs / 5) / 3), 0.0)
            bvn = t1 - t2
            a2 = aa / 2.0
            xs = (a2[:, None] * (_GL_X[None, :] + 1.0)) ** 2
            rs = np.sqrt(1.0 - xs)
            asr2 = -(bs[:, None] / xs + hk[:, None]) / 2.0
            term = np.where(asr2 > -100,
                            a2[:, None] * _GL_W[None, :] * np.exp(asr2) *
                            (np.exp(-hk[:, None] * (1 - rs) / (2 * (1 + rs))) / rs - (1 + c[:, None] * xs * (1 + d[:, None] * xs))),
                            0.0)
            bvn = bvn + np.sum(term, axis=1)  # full node set == Genz's (+/-) half-node double loop
            bvn = -bvn / (2 * np.pi)
        bvn = np.where(lt1, bvn, 0.0)
        pos = rh > 0
        res_pos = bvn + ndtr(-np.maximum(hh, kh))
        res_neg = -bvn + np.where(kh > hh, ndtr(kh) - ndtr(hh), 0.0)
        out[hi] = np.where(pos, res_pos, res_neg)
    return out[()] if out.ndim == 0 else out


# --------------------------------------------------------------------------
# Bjerksund-Stensland 2002
# --------------------------------------------------------------------------
def _gbs_call(S, X, T, r, b, v):
    return bsm_price(S, X, T, r, r - b, v, True)


def _phi(S, T, gamma, h, i, r, b, v):
    lam = (-r + gamma * b + 0.5 * gamma * (gamma - 1) * v * v) * T
    d = -(np.log(S / h) + (b + (gamma - 0.5) * v * v) * T) / (v * np.sqrt(T))
    kappa = 2 * b / (v * v) + 2 * gamma - 1
    return np.exp(lam) * S ** gamma * (ndtr(d) - (i / S) ** kappa * ndtr(d - 2 * np.log(i / S) / (v * np.sqrt(T))))


def _ksi(S, T2, gamma, h, I2, I1, t1, r, b, v):
    m = (b + (gamma - 0.5) * v * v)
    st1 = v * np.sqrt(t1)
    sT2 = v * np.sqrt(T2)
    e1 = (np.log(S / I1) + m * t1) / st1
    e2 = (np.log(I2 * I2 / (S * I1)) + m * t1) / st1
    e3 = (np.log(S / I1) - m * t1) / st1
    e4 = (np.log(I2 * I2 / (S * I1)) - m * t1) / st1
    f1 = (np.log(S / h) + m * T2) / sT2
    f2 = (np.log(I2 * I2 / (S * h)) + m * T2) / sT2
    f3 = (np.log(I1 * I1 / (S * h)) + m * T2) / sT2
    f4 = (np.log(S * I1 * I1 / (h * I2 * I2)) + m * T2) / sT2
    rho = np.sqrt(t1 / T2)
    lam = -r + gamma * b + 0.5 * gamma * (gamma - 1) * v * v
    kappa = 2 * b / (v * v) + (2 * gamma - 1)
    return np.exp(lam * T2) * S ** gamma * (
        cbnd(-e1, -f1, rho) - (I2 / S) ** kappa * cbnd(-e2, -f2, rho)
        - (I1 / S) ** kappa * cbnd(-e3, -f3, -rho) + (I1 / I2) ** kappa * cbnd(-e4, -f4, -rho))


def _bs2002_call(S, X, T, r, b, v):
    with np.errstate(all="ignore"):
        return _bs2002_call_impl(S, X, T, r, b, v)


def _bs2002_call_impl(S, X, T, r, b, v):
    S, X, T, r, b, v = np.broadcast_arrays(*[np.asarray(a, float) for a in (S, X, T, r, b, v)])
    out = _gbs_call(S, X, T, r, b, v) * np.ones_like(S)
    am = (b < r) & (T > 0) & (v > 0)
    if not np.any(am):
        return out
    S_, X_, T_, r_, b_, v_ = (a[am] for a in (S, X, T, r, b, v))
    v2 = v_ * v_
    t1 = 0.5 * (np.sqrt(5.0) - 1.0) * T_
    beta = (0.5 - b_ / v2) + np.sqrt((b_ / v2 - 0.5) ** 2 + 2 * r_ / v2)
    binf = beta / (beta - 1) * X_
    with np.errstate(divide="ignore", invalid="ignore"):
        b0 = np.maximum(X_, np.where(r_ - b_ != 0, r_ / (r_ - b_) * X_, np.inf))
        ht1 = -(b_ * t1 + 2 * v_ * np.sqrt(t1)) * X_ * X_ / ((binf - b0) * b0)
        ht2 = -(b_ * T_ + 2 * v_ * np.sqrt(T_)) * X_ * X_ / ((binf - b0) * b0)
    I1 = b0 + (binf - b0) * (1 - np.exp(ht1))
    I2 = b0 + (binf - b0) * (1 - np.exp(ht2))
    alfa1 = (I1 - X_) * I1 ** (-beta)
    alfa2 = (I2 - X_) * I2 ** (-beta)
    ex = S_ >= I2
    Sx = np.where(ex, 0.5 * I2, S_)  # dummy for exercised nodes (avoid log issues)
    one = np.ones_like(Sx)
    zero = np.zeros_like(Sx)
    val = (alfa2 * Sx ** beta - alfa2 * _phi(Sx, t1, beta, I2, I2, r_, b_, v_)
           + _phi(Sx, t1, one, I2, I2, r_, b_, v_) - _phi(Sx, t1, one, I1, I2, r_, b_, v_)
           - X_ * _phi(Sx, t1, zero, I2, I2, r_, b_, v_) + X_ * _phi(Sx, t1, zero, I1, I2, r_, b_, v_)
           + alfa1 * _phi(Sx, t1, beta, I1, I2, r_, b_, v_) - alfa1 * _ksi(Sx, T_, beta, I1, I2, I1, t1, r_, b_, v_)
           + _ksi(Sx, T_, one, I1, I2, I1, t1, r_, b_, v_) - _ksi(Sx, T_, one, X_, I2, I1, t1, r_, b_, v_)
           - X_ * _ksi(Sx, T_, zero, I1, I2, I1, t1, r_, b_, v_) + X_ * _ksi(Sx, T_, zero, X_, I2, I1, t1, r_, b_, v_))
    val = np.where(ex, S_ - X_, val)
    bad = ~np.isfinite(val) | (val > S_ * (1 + 1e-9)) | (val < -1e-9)
    if np.any(bad):  # numerical overflow at very low vol (kappa = 2b/v^2 huge): fall back to CRR
        bi = np.where(bad)[0]
        val = val.copy()
        for j in bi:
            val[j] = crr_price(S_[j], X_[j], T_[j], r_[j], v_[j], True, steps=300, q=r_[j] - b_[j])
    # never below the European value or intrinsic
    val = np.maximum(val, np.maximum(out[am], S_ - X_))
    out = out.copy()
    out[am] = val
    return out


def bs1993_call(S, X, T, r, b, v):
    """Bjerksund-Stensland (1993) American call (flat boundary).  Kept as a
    cross-check: reproduces Haug (2007) 5.2704 for S=42,X=40,T=.75,r=.04,b=-.04,v=.35."""
    S, X, T, r, b, v = np.broadcast_arrays(*[np.asarray(a, float) for a in (S, X, T, r, b, v)])
    eu = _gbs_call(S, X, T, r, b, v) * np.ones_like(S)
    v2 = v * v
    beta = (0.5 - b / v2) + np.sqrt((b / v2 - 0.5) ** 2 + 2 * r / v2)
    binf = beta / (beta - 1) * X
    with np.errstate(divide="ignore", invalid="ignore"):
        b0 = np.maximum(X, np.where(r - b != 0, r / (r - b) * X, np.inf))
        ht = -(b * T + 2 * v * np.sqrt(T)) * b0 / (binf - b0)
    I = b0 + (binf - b0) * (1 - np.exp(ht))
    al = (I - X) * I ** (-beta)
    one, zero = np.ones_like(S), np.zeros_like(S)
    Sx = np.where(S >= I, 0.5 * I, S)
    val = (al * Sx ** beta - al * _phi(Sx, T, beta, I, I, r, b, v) + _phi(Sx, T, one, I, I, r, b, v)
           - _phi(Sx, T, one, X, I, r, b, v) - X * _phi(Sx, T, zero, I, I, r, b, v) + X * _phi(Sx, T, zero, X, I, r, b, v))
    val = np.where(S >= I, S - X, val)
    out = np.where(b >= r, eu, np.maximum(val, eu))
    return out[()] if out.ndim == 0 else out


def bs2002_price(S, K, T, r, q, sigma, is_call):
    """Bjerksund-Stensland (2002) American approximation; continuous yield q
    (cost of carry b = r - q)."""
    cp = as_is_call(is_call)
    S, K, T, r, q, sigma, cpb = np.broadcast_arrays(*[np.asarray(a, float) for a in (S, K, T, r, q, sigma)], cp)
    cpb = cpb.astype(bool)
    b = r - q
    out = np.empty(S.shape, dtype=float)
    expired = (T <= 0) | (sigma <= 0)
    intr = np.where(cpb, np.maximum(S - K, 0), np.maximum(K - S, 0))
    c = cpb & ~expired
    p = ~cpb & ~expired
    if np.any(c):
        out[c] = _bs2002_call(S[c], K[c], T[c], r[c], b[c], sigma[c])
    if np.any(p):
        out[p] = _bs2002_call(K[p], S[p], T[p], r[p] - b[p], -b[p], sigma[p])
    out[expired] = intr[expired]
    if np.any(sigma <= 0) and np.any(T > 0):  # deterministic limit: max(intrinsic, discounted forward intrinsic)
        det = (sigma <= 0) & (T > 0)
        eu = bsm_price(S[det], K[det], T[det], r[det], q[det], 0.0, cpb[det])
        out[det] = np.maximum(eu, intr[det])
    return out[()] if out.ndim == 0 else out


# --------------------------------------------------------------------------
# CRR binomial
# --------------------------------------------------------------------------
def _crr_kernel_py(S_star, K, T, r, q, sigma, is_call, american, N, div_t, div_a):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    a = np.exp((r - q) * dt)
    p = (a - d) / (u - d)
    if p <= 0.0 or p >= 1.0:  # very low vol: drift-centred tree keeps 0<p<1
        u = np.exp((r - q) * dt + sigma * np.sqrt(dt))
        d = np.exp((r - q) * dt - sigma * np.sqrt(dt))
        p = (a - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(N + 1)
    ST = S_star * u ** j * d ** (N - j)
    vals = np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0)
    for i in range(N - 1, -1, -1):
        vals = disc * (p * vals[1:] + (1.0 - p) * vals[:-1])
        if american:
            t = i * dt
            pvd = 0.0
            for k in range(div_t.shape[0]):
                if div_t[k] > t and div_t[k] <= T:
                    pvd += div_a[k] * np.exp(-r * (div_t[k] - t))
            jj = np.arange(i + 1)
            Si = S_star * u ** jj * d ** (i - jj) + pvd
            ex = (Si - K) if is_call else (K - Si)
            vals = np.maximum(vals, ex)
    return vals[0]


if _HAVE_NUMBA:
    @_nb.njit(cache=False)
    def _crr_kernel_nb(S_star, K, T, r, q, sigma, is_call, american, N, div_t, div_a):  # pragma: no cover - same math
        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        a = np.exp((r - q) * dt)
        p = (a - d) / (u - d)
        if p <= 0.0 or p >= 1.0:
            u = np.exp((r - q) * dt + sigma * np.sqrt(dt))
            d = np.exp((r - q) * dt - sigma * np.sqrt(dt))
            p = (a - d) / (u - d)
        disc = np.exp(-r * dt)
        vals = np.empty(N + 1)
        for j in range(N + 1):
            st = S_star * u ** j * d ** (N - j)
            vals[j] = max(st - K, 0.0) if is_call else max(K - st, 0.0)
        for i in range(N - 1, -1, -1):
            t = i * dt
            pvd = 0.0
            if american:
                for k in range(div_t.shape[0]):
                    if div_t[k] > t and div_t[k] <= T:
                        pvd += div_a[k] * np.exp(-r * (div_t[k] - t))
            for j in range(i + 1):
                v = disc * (p * vals[j + 1] + (1.0 - p) * vals[j])
                if american:
                    si = S_star * u ** j * d ** (i - j) + pvd
                    ex = (si - K) if is_call else (K - si)
                    if ex > v:
                        v = ex
                vals[j] = v
        return vals[0]


def crr_price(S, K, T, r, sigma, is_call, american: bool = True, steps: int = 400, q=0.0,
              div_times=(), div_amounts=(), smooth: bool = True, use_numba: bool | None = None):
    """CRR binomial price (vectorized over S, K, T, r, sigma, is_call, q by looping).

    Discrete dividends (``div_times`` years from now, ``div_amounts`` cash) are
    shared by all elements.  ``smooth`` averages N and N+1 steps.
    """
    cp = as_is_call(is_call)
    S, K, T, r, sigma, q, cpb = np.broadcast_arrays(*[np.asarray(a, float) for a in (S, K, T, r, sigma, q)], cp)
    cpb = cpb.astype(bool)
    dt_ = np.asarray(div_times, float).ravel()
    da_ = np.asarray(div_amounts, float).ravel()
    kern = _crr_kernel_nb if (_HAVE_NUMBA and (use_numba is None or use_numba)) else _crr_kernel_py
    out = np.empty(S.shape, dtype=float)
    it = np.nditer(S, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        s, k, t, rr, sg, qq, c = S[idx], K[idx], T[idx], r[idx], sigma[idx], q[idx], bool(cpb[idx])
        if t <= 0 or sg <= 0:
            pvd = float(np.sum(da_[(dt_ > 0) & (dt_ <= t)] * np.exp(-rr * dt_[(dt_ > 0) & (dt_ <= t)]))) if t > 0 else 0.0
            intr = max(s - k, 0.0) if c else max(k - s, 0.0)
            eu = float(bsm_price(s - pvd, k, max(t, 0), rr, qq, 0.0, c)) if t > 0 else intr
            out[idx] = max(intr, eu) if american else eu
            continue
        pvd = float(np.sum(da_[(dt_ > 0) & (dt_ <= t)] * np.exp(-rr * dt_[(dt_ > 0) & (dt_ <= t)])))
        s_star = s - pvd
        v1 = kern(s_star, k, t, rr, qq, sg, c, american, int(steps), dt_, da_)
        if smooth:
            v2 = kern(s_star, k, t, rr, qq, sg, c, american, int(steps) + 1, dt_, da_)
            v1 = 0.5 * (v1 + v2)
        out[idx] = v1
    return out[()] if out.ndim == 0 else out


def american_price(S, K, T, r, q, sigma, is_call, method: str = "bs2002", steps: int = 400,
                   div_times=(), div_amounts=()):
    """Dispatch: 'bs2002' (continuous q only) or 'crr' (supports discrete dividends)."""
    if len(np.atleast_1d(div_times)) and method == "bs2002":
        method = "crr"
    if method == "bs2002":
        return bs2002_price(S, K, T, r, q, sigma, is_call)
    if method == "crr":
        return crr_price(S, K, T, r, sigma, is_call, american=True, steps=steps, q=q,
                         div_times=div_times, div_amounts=div_amounts)
    raise ValueError(method)


def american_greeks(S, K, T, r, q, sigma, is_call, method: str = "bs2002", steps: int = 400,
                    div_times=(), div_amounts=()) -> Greeks:
    """Finite-difference greeks of the American price (same units as bsm_greeks)."""
    S = np.asarray(S, float)
    f = lambda s=S, t=T, rr=r, v=sigma: american_price(s, K, t, rr, q, v, is_call, method, steps,  # noqa: E731
                                                       np.asarray(div_times, float) - (np.asarray(T, float) - t) if len(np.atleast_1d(div_times)) else (),
                                                       div_amounts)
    h = np.maximum(S * 1e-3, 1e-4)
    p0 = f()
    pu, pd_ = f(s=S + h), f(s=S - h)
    delta = (pu - pd_) / (2 * h)
    gamma = (pu - 2 * p0 + pd_) / (h * h)
    dv = 1e-3
    vega = (f(v=np.asarray(sigma, float) + dv) - f(v=np.maximum(np.asarray(sigma, float) - dv, 1e-6))) / (2 * dv) / 100.0
    dT = np.minimum(1.0 / 365.0, np.asarray(T, float) * 0.5)
    theta = (f(t=np.asarray(T, float) - dT) - p0) / np.where(dT > 0, dT, 1.0) / 365.0
    dr = 1e-4
    rho = (f(rr=np.asarray(r, float) + dr) - f(rr=np.asarray(r, float) - dr)) / (2 * dr) / 100.0
    return Greeks(np.asarray(p0), np.asarray(delta), np.asarray(gamma), np.asarray(vega), np.asarray(theta), np.asarray(rho))
