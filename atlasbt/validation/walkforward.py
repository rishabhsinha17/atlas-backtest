"""Walk-forward parameter selection with an untouched final holdout.

Two modes:

1. Matrix mode (``walk_forward_matrix``): you already have, for every
   parameter combination, the strategy's daily returns over the full sample
   (each run is causal, so return t only depends on data <= t). For each fold
   the combination with the best objective on the TRAIN window is applied to
   the following TEST window; OOS test segments are stitched together.
   Caveat (documented): a strategy run from the sample start carries state
   (positions, warm-up) into each test window that a fresh start would not;
   for most bar strategies this is immaterial, for slow position strategies
   use the callable mode.
2. Callable mode (``walk_forward_run``): ``run(params, start, end) -> returns``
   is invoked for every combination on every train window and for the chosen
   one on the test window (the runner handles its own warm-up from history
   before ``start``).

Holdout: the last ``holdout`` share (or a start date) of the sample is cut
off BEFORE any fold is built; selection never sees it. ``evaluate_holdout``
re-selects on the train window immediately preceding the holdout (same
train-length rule; anchored = all pre-holdout data) and reports the chosen
combination's holdout performance exactly once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd

from .bootstrap import sharpe_annual


def expand_grid(grid: dict) -> list[dict]:
    """Cartesian product of a {param: [values]} dict in deterministic order."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in product(*[list(grid[k]) for k in keys])]


def param_key(p: dict) -> str:
    return "|".join(f"{k}={p[k]}" for k in sorted(p))


@dataclass
class Fold:
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex


@dataclass
class Splits:
    folds: list[Fold]
    pre_holdout: pd.DatetimeIndex
    holdout: pd.DatetimeIndex
    anchored: bool
    train_len: int


def _resolve_len(index: pd.DatetimeIndex, x) -> int:
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float) and x < 1:
        return int(round(len(index) * x))
    # pandas offset string like '3Y', '252D' -> number of bars from the first date
    off = pd.tseries.frequencies.to_offset(x) if isinstance(x, str) else x
    end = index[0] + off
    return int(np.searchsorted(index, end, side="left"))


def walk_forward_splits(index, train, test, step=None, anchored: bool = False, holdout=0.2) -> Splits:
    """Build rolling (or anchored/expanding) train/test folds on pre-holdout data.

    train/test/step: number of bars (int), a fraction (<1) of the sample, or an
    offset string ('3Y', '6M'). If step < test, every fold's test segment is
    truncated to `step` bars (OOS segments tile without overlap; used by both
    ``walk_forward_matrix`` and ``walk_forward_run``). holdout: fraction (<1), number of bars, or a
    timestamp (holdout = index >= timestamp)."""
    idx = pd.DatetimeIndex(index)
    if isinstance(holdout, (pd.Timestamp, str)):
        ts = pd.Timestamp(holdout)
        if idx.tz is not None and ts.tzinfo is None:
            ts = ts.tz_localize(idx.tz)
        cut = int(np.searchsorted(idx, ts, side="left"))
    elif isinstance(holdout, float) and holdout < 1:
        cut = len(idx) - int(round(len(idx) * holdout))
    else:
        cut = len(idx) - int(holdout or 0)
    pre, hold = idx[:cut], idx[cut:]
    ntr, nte = _resolve_len(idx, train), _resolve_len(idx, test)
    nst = _resolve_len(idx, step) if step is not None else nte
    if ntr <= 0 or nte <= 0:
        raise ValueError("train/test lengths must be positive")
    folds = []
    s = ntr
    while s + 1 <= len(pre):
        tr = pre[:s] if anchored else pre[s - ntr:s]
        # with step < test, consecutive test windows would overlap; truncate each
        # test segment to `step` bars so the stitched OOS dates tile exactly
        te = pre[s:min(s + min(nte, nst), len(pre))]
        if len(te) == 0:
            break
        folds.append(Fold(tr, te))
        s += nst
    return Splits(folds, pre, hold, anchored, ntr)


def _objective(name_or_fn):
    if callable(name_or_fn):
        return name_or_fn
    if name_or_fn == "sharpe":
        return lambda r: float(sharpe_annual(np.asarray(r, float)))
    if name_or_fn == "total_return":
        return lambda r: float(np.prod(1 + np.asarray(r, float)) - 1)
    if name_or_fn == "mean":
        return lambda r: float(np.mean(r))
    raise ValueError("objective must be sharpe|total_return|mean or a callable")


def _ratio(a, b) -> float:
    return float(a / b) if (b and np.isfinite(b) and np.isfinite(a)) else float("nan")


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series
    folds: pd.DataFrame
    holdout_returns: pd.Series | None = None
    holdout_params: str | None = None
    holdout_metric: float | None = None
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        r = self.oos_returns.dropna()
        d = {"n_folds": len(self.folds), "oos_sharpe": float(sharpe_annual(r.to_numpy())) if len(r) > 2 else np.nan,
             "oos_total_return": float(np.prod(1 + r) - 1),
             "wf_efficiency": _ratio(self.folds["test_metric"].mean(), self.folds["train_metric"].mean())
             if len(self.folds) else np.nan,
             "param_stability": float(self.folds["chosen"].value_counts(normalize=True).iloc[0]) if len(self.folds) else np.nan}
        if self.holdout_returns is not None:
            h = self.holdout_returns.dropna()
            d.update({"holdout_params": self.holdout_params,
                      "holdout_sharpe": float(sharpe_annual(h.to_numpy())) if len(h) > 2 else np.nan,
                      "holdout_total_return": float(np.prod(1 + h) - 1)})
        return d


def walk_forward_matrix(returns_by_param: pd.DataFrame, splits: Splits, objective="sharpe",
                        evaluate_holdout: bool = True) -> WalkForwardResult:
    """Matrix-mode walk-forward. ``returns_by_param``: index = dates, one column per
    parameter combination (use ``param_key`` names)."""
    f = _objective(objective)
    R = returns_by_param
    rows, oos = [], []
    for i, fold in enumerate(splits.folds):
        tr = R.loc[R.index.isin(fold.train)]
        scores = {c: f(tr[c].dropna().to_numpy()) for c in R.columns}
        best = max(scores, key=lambda c: (np.nan_to_num(scores[c], nan=-np.inf), -list(R.columns).index(c)))
        te = R.loc[R.index.isin(fold.test), best]
        oos.append(te)
        rows.append({"fold": i, "train_start": fold.train[0], "train_end": fold.train[-1], "test_start": fold.test[0],
                     "test_end": fold.test[-1], "chosen": best, "train_metric": scores[best],
                     "test_metric": f(te.dropna().to_numpy())})
    res = WalkForwardResult(pd.concat(oos) if oos else pd.Series(dtype=float), pd.DataFrame(rows),
                            meta={"anchored": splits.anchored, "train_len": splits.train_len})
    if evaluate_holdout and len(splits.holdout):
        pre = splits.pre_holdout
        trw = pre if splits.anchored else pre[-splits.train_len:]
        tr = R.loc[R.index.isin(trw)]
        scores = {c: f(tr[c].dropna().to_numpy()) for c in R.columns}
        best = max(scores, key=lambda c: (np.nan_to_num(scores[c], nan=-np.inf), -list(R.columns).index(c)))
        h = R.loc[R.index.isin(splits.holdout), best]
        res.holdout_returns, res.holdout_params, res.holdout_metric = h, best, f(h.dropna().to_numpy())
    return res


def walk_forward_run(run: Callable[[dict, pd.Timestamp, pd.Timestamp], pd.Series], grid: dict, splits: Splits,
                     objective="sharpe", evaluate_holdout: bool = True) -> WalkForwardResult:
    """Callable-mode walk-forward (fresh runs per window)."""
    f = _objective(objective)
    combos = expand_grid(grid)
    rows, oos = [], []

    def select(train_idx):
        best, bs = None, -np.inf
        for p in combos:
            s = f(run(p, train_idx[0], train_idx[-1]).dropna().to_numpy())
            if np.nan_to_num(s, nan=-np.inf) > bs:
                best, bs = p, s
        return best, bs

    for i, fold in enumerate(splits.folds):
        p, s = select(fold.train)
        te = run(p, fold.test[0], fold.test[-1])
        oos.append(te)
        rows.append({"fold": i, "train_start": fold.train[0], "train_end": fold.train[-1], "test_start": fold.test[0],
                     "test_end": fold.test[-1], "chosen": param_key(p), "train_metric": s,
                     "test_metric": f(te.dropna().to_numpy())})
    res = WalkForwardResult(pd.concat(oos) if oos else pd.Series(dtype=float), pd.DataFrame(rows),
                            meta={"anchored": splits.anchored, "train_len": splits.train_len})
    if evaluate_holdout and len(splits.holdout):
        pre = splits.pre_holdout
        trw = pre if splits.anchored else pre[-splits.train_len:]
        p, _ = select(trw)
        h = run(p, splits.holdout[0], splits.holdout[-1])
        res.holdout_returns, res.holdout_params, res.holdout_metric = h, param_key(p), f(h.dropna().to_numpy())
    return res
