"""Dependent-data bootstraps for confidence intervals on performance metrics.

* Stationary bootstrap - Politis & Romano (1994), JASA 89(428): blocks start
  at uniform random positions, lengths ~ Geometric(p = 1/b), wrap-around.
* Circular block bootstrap - Politis & Romano (1992): fixed-length blocks with
  wrap-around.
* Automatic block length - Politis & White (2004), Econometric Reviews 23(1),
  with the correction of Patton, Politis & White (2009), Econometric Reviews
  28(4): flat-top lag window, m chosen as twice the first lag after which K_n
  consecutive autocorrelations are insignificant at c*sqrt(log10(n)/n), c=2.

CIs are percentile intervals by default (``ci='basic'`` gives the reverse /
"basic" interval). Metrics are recomputed on each resampled return path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- block length
def optimal_block_length(x) -> dict:
    """Politis-White (2004) / Patton-Politis-White (2009) optimal block lengths.

    Returns {'stationary': b_SB, 'circular': b_CB, 'm': m_hat}.
    """
    a = np.asarray(pd.Series(np.asarray(x, float)).dropna(), float)
    n = a.shape[0]
    if n < 10:
        raise ValueError("need at least 10 observations")
    e = a - a.mean()
    kn = max(5, int(np.ceil(np.log10(n))))
    m_max = int(np.ceil(np.sqrt(n))) + kn
    b_max = np.ceil(min(3.0 * np.sqrt(n), n / 3.0))
    c = 2.0
    gamma = np.array([e[k:] @ e[:n - k] / n for k in range(m_max + kn + 1)])
    rho = gamma / gamma[0]
    band = c * np.sqrt(np.log10(n) / n)
    m_hat = None
    for m in range(0, m_max + 1):
        if np.all(np.abs(rho[m + 1:m + kn + 1]) < band):
            m_hat = m
            break
    M = m_max if m_hat is None else min(2 * max(m_hat, 1), m_max)
    k = np.arange(1, M + 1)
    lam = np.where(k / M <= 0.5, 1.0, 2.0 * (1.0 - k / M))
    G = 2.0 * np.sum(lam * k * gamma[1:M + 1])
    g0 = gamma[0] + 2.0 * np.sum(lam * gamma[1:M + 1])
    if g0 <= 0:
        return {"stationary": 1.0, "circular": 1.0, "m": M}
    d_sb = 2.0 * g0 ** 2
    d_cb = (4.0 / 3.0) * g0 ** 2
    b_sb = (2.0 * G ** 2 / d_sb) ** (1 / 3) * n ** (1 / 3)
    b_cb = (2.0 * G ** 2 / d_cb) ** (1 / 3) * n ** (1 / 3)
    return {"stationary": float(min(max(b_sb, 1.0), b_max)), "circular": float(min(max(b_cb, 1.0), b_max)),
            "m": M}


# ---------------------------------------------------------------- index generators
def stationary_indices(n: int, mean_block: float, reps: int, rng: np.random.Generator) -> np.ndarray:
    """(reps, n) index matrix for the stationary bootstrap."""
    p = 1.0 / max(float(mean_block), 1.0)
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, reps)
    new_block = rng.random((reps, n)) < p
    starts = rng.integers(0, n, (reps, n))
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def circular_indices(n: int, block: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """(reps, n) index matrix for the circular block bootstrap."""
    b = max(int(round(block)), 1)
    nb = int(np.ceil(n / b))
    starts = rng.integers(0, n, (reps, nb))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % n
    return idx.reshape(reps, nb * b)[:, :n]


def bootstrap_indices(n: int, reps: int, method: str = "stationary", block: float | None = None,
                      seed: int | None = 0, data=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if block is None:
        if data is None:
            raise ValueError("block=None requires data for automatic selection")
        bl = optimal_block_length(data)
        block = bl["stationary"] if method == "stationary" else bl["circular"]
    if method == "stationary":
        return stationary_indices(n, block, reps, rng)
    if method == "circular":
        return circular_indices(n, int(round(block)), reps, rng)
    if method == "iid":
        return rng.integers(0, n, (reps, n))
    raise ValueError("method must be stationary|circular|iid")


# ---------------------------------------------------------------- metrics on return arrays
def sharpe_annual(r: np.ndarray, periods: int = 252) -> np.ndarray:
    """Annualised Sharpe along the last axis (r already in excess of rf)."""
    mu = r.mean(axis=-1)
    sd = r.std(axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, mu / sd * np.sqrt(periods), np.nan)


def cagr(r: np.ndarray, periods: int = 252) -> np.ndarray:
    n = r.shape[-1]
    g = np.prod(1.0 + r, axis=-1)
    with np.errstate(invalid="ignore"):
        return np.where(g > 0, g ** (periods / n) - 1.0, -1.0)


def max_drawdown(r: np.ndarray) -> np.ndarray:
    """Maximum drawdown (negative number) along the last axis."""
    eq = np.cumprod(1.0 + r, axis=-1)
    peak = np.maximum.accumulate(np.concatenate([np.ones(eq.shape[:-1] + (1,)), eq], axis=-1), axis=-1)[..., 1:]
    return np.min(eq / peak - 1.0, axis=-1)


METRICS: dict[str, Callable] = {"sharpe": sharpe_annual, "cagr": cagr, "max_drawdown": max_drawdown}


@dataclass
class BootstrapCI:
    metric: str
    point: float
    lo: float
    hi: float
    se: float
    alpha: float
    method: str
    block: float
    reps: int
    samples: np.ndarray = field(repr=False, default=None)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("metric", "point", "lo", "hi", "se", "alpha", "method", "block", "reps")}


def bootstrap_ci(returns, metric: str | Callable = "sharpe", reps: int = 2000, method: str = "stationary",
                 block: float | None = None, alpha: float = 0.05, seed: int | None = 0, ci: str = "percentile",
                 periods: int = 252) -> BootstrapCI:
    """Bootstrap confidence interval for a performance metric of a return series."""
    r = np.asarray(pd.Series(np.asarray(returns, float)).dropna(), float)
    n = r.shape[0]
    if block is None:
        bl = optimal_block_length(r)
        block = bl["stationary"] if method == "stationary" else bl["circular"]
    idx = bootstrap_indices(n, reps, method, block, seed)
    fn = METRICS[metric] if isinstance(metric, str) else metric
    name = metric if isinstance(metric, str) else getattr(metric, "__name__", "metric")
    kw = {"periods": periods} if fn in (sharpe_annual, cagr) else {}
    point = float(fn(r, **kw))
    samples = np.asarray(fn(r[idx], **kw), float)
    s = samples[~np.isnan(samples)]
    qlo, qhi = np.quantile(s, [alpha / 2, 1 - alpha / 2])
    if ci == "basic":
        lo, hi = 2 * point - qhi, 2 * point - qlo
    else:
        lo, hi = qlo, qhi
    return BootstrapCI(name, point, float(lo), float(hi), float(s.std(ddof=1)), alpha, method, float(block), reps,
                       samples)


def bootstrap_metrics(returns, metrics=("sharpe", "cagr", "max_drawdown"), **kw) -> dict:
    """CIs for several metrics using the SAME resamples (same seed/block)."""
    return {m: bootstrap_ci(returns, m, **kw).as_dict() for m in metrics}
