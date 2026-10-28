"""Probability of Backtest Overfitting via Combinatorially Symmetric
Cross-Validation (CSCV) - Bailey, Borwein, Lopez de Prado & Zhu (2017),
"The Probability of Backtest Overfitting", J. of Computational Finance 20(4).

Procedure (paper section 2.3):
 1. M = T x N matrix of per-period returns of N trial configurations.
 2. Split rows into S (even) contiguous blocks of equal size (the first
    T mod S rows are dropped so the most recent data are kept).
 3. For each of the C(S, S/2) combinations J of S/2 blocks: train = rows in J,
    test = the complement. n* = argmax_n metric_train(n) (ties -> lowest index).
 4. w = rank of n* among the N test metrics / (N + 1) (average ranks,
    ascending); logit lambda = ln(w / (1 - w)).
 5. PBO = share of combinations with lambda <= 0 (n* at or below the
    median out-of-sample).
Also reported: performance degradation (OLS slope of test on train metric of
n*), probability of OOS loss (test metric of n* < 0), and the logits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PBOResult:
    pbo: float
    n_combinations: int
    logits: np.ndarray = field(repr=False)
    train_best: np.ndarray = field(repr=False)
    test_of_best: np.ndarray = field(repr=False)
    degradation_slope: float = float("nan")
    prob_oos_loss: float = float("nan")
    S: int = 16

    def as_dict(self) -> dict:
        return {"pbo": self.pbo, "n_combinations": self.n_combinations, "degradation_slope": self.degradation_slope,
                "prob_oos_loss": self.prob_oos_loss, "S": self.S,
                "median_logit": float(np.median(self.logits)) if self.logits.size else float("nan")}


def _sharpe_from_sums(s1, s2, n):
    mu = s1 / n
    var = (s2 - n * mu * mu) / (n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(var > 0, mu / np.sqrt(var), np.nan)


def pbo_cscv(M, S: int = 16, metric: str = "sharpe", max_combinations: int | None = None,
             seed: int | None = 0) -> PBOResult:
    """CSCV PBO for a T x N performance matrix (DataFrame or ndarray).

    metric: 'sharpe' (per-period mean/std) or 'mean'. max_combinations: if
    set and smaller than C(S, S/2), a random subset of combinations is used.
    """
    X = M.to_numpy(float) if isinstance(M, pd.DataFrame) else np.asarray(M, float)
    if S % 2 or S < 2:
        raise ValueError("S must be an even integer >= 2")
    T, N = X.shape
    if N < 2:
        raise ValueError("need at least 2 trials")
    X = np.nan_to_num(X, nan=0.0)
    bs = T // S
    if bs < 2:
        raise ValueError("too few rows for S blocks")
    X = X[T - bs * S:]
    blocks = X.reshape(S, bs, N)
    s1 = blocks.sum(axis=1)            # (S, N)
    s2 = (blocks ** 2).sum(axis=1)
    combos = list(combinations(range(S), S // 2))
    if max_combinations and len(combos) > max_combinations:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(combos), max_combinations, replace=False)
        combos = [combos[i] for i in sorted(pick)]
    tot1, tot2 = s1.sum(0), s2.sum(0)
    nh = bs * (S // 2)
    logits = np.empty(len(combos))
    tb = np.empty(len(combos))
    tt = np.empty(len(combos))
    for c, J in enumerate(combos):
        J = list(J)
        a1, a2 = s1[J].sum(0), s2[J].sum(0)
        b1, b2 = tot1 - a1, tot2 - a2
        if metric == "sharpe":
            mtr, mte = _sharpe_from_sums(a1, a2, nh), _sharpe_from_sums(b1, b2, nh)
        elif metric == "mean":
            mtr, mte = a1 / nh, b1 / nh
        else:
            raise ValueError("metric must be sharpe|mean")
        mtr = np.where(np.isnan(mtr), -np.inf, mtr)
        mte_r = np.where(np.isnan(mte), -np.inf, mte)
        best = int(np.argmax(mtr))
        rk = stats.rankdata(mte_r, method="average")[best]
        w = rk / (N + 1.0)
        logits[c] = np.log(w / (1.0 - w))
        tb[c] = mtr[best]
        tt[c] = mte[best]
    pbo = float(np.mean(logits <= 0))
    ok = np.isfinite(tb) & np.isfinite(tt)
    slope = float(np.polyfit(tb[ok], tt[ok], 1)[0]) if ok.sum() > 2 and np.ptp(tb[ok]) > 0 else float("nan")
    return PBOResult(pbo=pbo, n_combinations=len(combos), logits=logits, train_best=tb, test_of_best=tt,
                     degradation_slope=slope, prob_oos_loss=float(np.mean(tt[ok] < 0)) if ok.any() else float("nan"),
                     S=S)
