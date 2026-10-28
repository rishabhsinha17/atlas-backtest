"""Data-snooping-robust tests across many strategies.

* White (2000) "A Reality Check for Data Snooping", Econometrica 68(5).
* Hansen (2005) "A Test for Superior Predictive Ability", JBES 23(4):
  studentised statistic T = max(max_k sqrt(n) dbar_k / omega_k, 0); bootstrap
  null re-centred with g_l(x)=max(x,0) (lower / liberal), g_c(x)=x*1{x >=
  -omega sqrt(2 loglog n / n)} (consistent) and g_u(x)=x (upper = RC when not
  studentised). omega_k^2 from the stationary-bootstrap kernel estimator
  (Hansen 2005 eq. (A.?) / Politis-Romano):
      omega^2 = g0 + 2 sum_i kappa(n,i) g_i,
      kappa(n,i) = (1-i/n)(1-q)^i + (i/n)(1-q)^(n-i), q = 1/block.
  Bootstraps: stationary (Politis-Romano 1994), shared indices across models.
* Benjamini & Hochberg (1995) FDR step-up (and Benjamini-Yekutieli 2001).

Inputs are "performance differentials" d[t, k]: the per-period excess of
model k over the benchmark (e.g. strategy return - benchmark return, or
strategy excess return over cash with benchmark = 0). Positive = better.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bootstrap import bootstrap_indices, optimal_block_length


def _as2d(d) -> np.ndarray:
    a = d.to_numpy(float) if isinstance(d, (pd.DataFrame, pd.Series)) else np.asarray(d, float)
    if a.ndim == 1:
        a = a[:, None]
    return a[~np.isnan(a).any(axis=1)]


def _autocov_fft(e: np.ndarray) -> np.ndarray:
    """Biased autocovariances g_i = sum_t e_t e_{t+i} / n for i = 0..n-1, per column."""
    n = e.shape[0]
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(e, nfft, axis=0)
    ac = np.fft.irfft(f * np.conj(f), nfft, axis=0)[:n]
    return ac / n


def kernel_variance(d: np.ndarray, block: float) -> np.ndarray:
    """Stationary-bootstrap-consistent long-run variance of sqrt(n)*dbar per column."""
    n = d.shape[0]
    e = d - d.mean(0)
    q = 1.0 / block
    g = _autocov_fft(e)
    i = np.arange(1, n)
    kappa = (1 - i / n) * (1 - q) ** i + (i / n) * (1 - q) ** (n - i)
    return g[0] + 2.0 * (kappa[:, None] * g[1:]).sum(0)


def _boot_means(a: np.ndarray, idx: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Bootstrap means via index-count matrices (BLAS friendly)."""
    reps, n = idx.shape
    out = np.empty((reps, a.shape[1]))
    for s in range(0, reps, chunk):
        sub = idx[s:s + chunk]
        cnt = np.zeros((sub.shape[0], n))
        rows = np.repeat(np.arange(sub.shape[0]), n)
        np.add.at(cnt, (rows, sub.ravel()), 1.0)
        out[s:s + chunk] = cnt @ a / n
    return out


def _default_block(d: np.ndarray) -> float:
    return float(np.median([optimal_block_length(d[:, j])["stationary"] for j in range(d.shape[1])]))


def hansen_spa(d, reps: int = 2000, block: float | None = None, seed: int | None = 0, studentize: bool = True,
               method: str = "stationary") -> dict:
    """Hansen's SPA p-values (lower, consistent, upper) for H0: no model beats the benchmark."""
    a = _as2d(d)
    n, k = a.shape
    block = block or _default_block(a)
    dbar = a.mean(0)
    omega2 = kernel_variance(a, block)
    omega = np.sqrt(np.maximum(omega2, 1e-300))
    scale = omega if studentize else np.ones(k)
    stat = max(float(np.max(np.sqrt(n) * dbar / scale)), 0.0) if studentize else float(np.max(np.sqrt(n) * dbar))
    thr = -np.sqrt(omega2 / n * 2 * np.log(np.log(n)))
    centers = {"lower": np.maximum(dbar, 0.0), "consistent": np.where(dbar >= thr, dbar, 0.0), "upper": dbar}
    idx = bootstrap_indices(n, reps, method, block, seed)
    boot_means = _boot_means(a, idx)  # (reps, k)
    out = {"statistic": stat, "block": block, "reps": reps, "studentize": studentize}
    for name, mu in centers.items():
        z = np.sqrt(n) * (boot_means - mu) / scale
        tstar = np.max(z, axis=1)
        if studentize:
            tstar = np.maximum(tstar, 0.0)
        out[name] = float(np.mean(tstar > stat))
    return out


def white_reality_check(d, reps: int = 2000, block: float | None = None, seed: int | None = 0,
                        method: str = "stationary") -> dict:
    """White's Reality Check p-value (non-studentised, fully re-centred)."""
    r = hansen_spa(d, reps, block, seed, studentize=False, method=method)
    return {"pvalue": r["upper"], "statistic": r["statistic"], "block": r["block"], "reps": reps}


def benjamini_hochberg(pvalues, q: float = 0.05, method: str = "bh"):
    """FDR control. Returns (reject: ndarray[bool], qvalues: ndarray) in input order.

    method='bh' (independent/PRDS) or 'by' (arbitrary dependence)."""
    p = np.asarray(pvalues, float)
    m = p.size
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    c = np.sum(1.0 / np.arange(1, m + 1)) if method == "by" else 1.0
    adj = ranked * m * c / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    qv = np.empty(m)
    qv[order] = adj
    return qv <= q, qv
