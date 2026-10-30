"""Purged & embargoed k-fold cross-validation (Lopez de Prado 2018,
"Advances in Financial Machine Learning", ch. 7, snippets 7.1-7.3).

Each observation i has an information interval [t0_i, t1_i] (t0 = when the
features are observed / the trade is entered, t1 = when its label is
resolved / the trade exits). For each test fold (a contiguous block of
observations):
  * purge: drop from training every observation whose interval overlaps the
    test fold's span [min t0_test, max t1_test];
  * embargo: additionally drop training observations whose t0 falls within
    the `embargo` share of observations immediately after the test fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class PurgedKFold:
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        if n_splits < 2:
            raise ValueError("n_splits >= 2")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, t0, t1):
        """t0, t1: sequences of timestamps (or numbers) per observation, sorted by t0.
        Yields (train_positions, test_positions)."""
        t0 = pd.Index(t0)
        t1 = pd.Index(t1)
        n = len(t0)
        if len(t1) != n:
            raise ValueError("t0 and t1 lengths differ")
        if not t0.is_monotonic_increasing:
            raise ValueError("observations must be sorted by t0")
        t0v, t1v = t0.to_numpy(), t1.to_numpy()
        emb = int(np.ceil(n * self.embargo_pct))
        bounds = np.array_split(np.arange(n), self.n_splits)
        for test in bounds:
            lo, hi = test[0], test[-1]
            span0, span1 = t0v[lo], t1v[test].max()
            overlap = (t0v <= span1) & (t1v >= span0)
            train = ~overlap
            train[test] = False
            if emb:
                train[hi + 1:min(n, hi + 1 + emb)] = False
            yield np.flatnonzero(train), test
