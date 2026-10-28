"""Cross-sectional helpers for factor / ranking strategies.

All functions take ONE cross-section (a Series indexed by symbol, observed
at a single as-of time) or a wide panel (index = dates, columns = symbols;
applied row by row, so each row only uses that date's information).

Tie and bucket rules (documented in docs/ANALYTICS.md):
* ``quantile_bucket`` (no breakpoint universe): average ranks, bucket =
  ceil(rank * q / n) -> labels 1..q (1 = lowest). Tied values always share a
  bucket; with n divisible by q and no ties every bucket has n/q names.
* With a breakpoint universe (e.g. NYSE-only, Fama & French 1993): breakpoints
  are the k/q quantiles (numpy 'linear') of the universe's values; a value
  equal to a breakpoint goes to the LOWER bucket (searchsorted side='left').
* NaNs are never ranked (bucket NaN).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _cs(x) -> pd.Series:
    return x if isinstance(x, pd.Series) else pd.Series(np.asarray(x, float))


def quantile_bucket(x, q: int = 10, universe_mask=None) -> pd.Series:
    """Bucket labels 1..q (float, NaN for missing values)."""
    s = _cs(x).astype(float)
    out = pd.Series(np.nan, index=s.index)
    ok = s.notna()
    if universe_mask is None:
        n = int(ok.sum())
        if n == 0:
            return out
        r = s[ok].rank(method="average")
        out[ok] = np.ceil(r.to_numpy() * q / n - 1e-9).clip(1, q)
        return out
    um = _cs(universe_mask).reindex(s.index).fillna(False).astype(bool) & ok
    if um.sum() == 0:
        return out
    bps = breakpoints(s[um], q)
    out[ok] = np.searchsorted(bps, s[ok].to_numpy(), side="left") + 1.0
    return out


def breakpoints(x, q: int) -> np.ndarray:
    """Interior quantile breakpoints k/q, k = 1..q-1 (numpy linear interpolation)."""
    v = _cs(x).dropna().to_numpy(float)
    return np.quantile(v, np.arange(1, q) / q)


def quantile_bucket_panel(df: pd.DataFrame, q: int = 10, universe_mask: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for d, row in df.iterrows():
        um = None if universe_mask is None else universe_mask.loc[d]
        rows.append(quantile_bucket(row, q, um))
    return pd.DataFrame(rows, index=df.index, columns=df.columns)


def long_short(buckets: pd.Series, q: int, long_top: bool = True) -> pd.Series:
    """+1 for the long bucket, -1 for the short bucket, 0 otherwise."""
    hi, lo = (q, 1) if long_top else (1, q)
    return pd.Series(np.where(buckets == hi, 1.0, np.where(buckets == lo, -1.0, 0.0)), index=buckets.index)


def winsorize(x, lower: float = 0.01, upper: float = 0.99, groups=None) -> pd.Series:
    """Clip at the cross-sectional lower/upper quantiles (optionally per group)."""
    s = _cs(x).astype(float)
    if groups is None:
        lo, hi = s.quantile(lower), s.quantile(upper)
        return s.clip(lo, hi)
    g = _cs(groups).reindex(s.index)
    return s.groupby(g, group_keys=False).apply(lambda v: v.clip(v.quantile(lower), v.quantile(upper))).reindex(s.index)


def zscore(x, ddof: int = 1) -> pd.Series:
    s = _cs(x).astype(float)
    sd = s.std(ddof=ddof)
    return (s - s.mean()) / sd if sd > 0 else s * np.nan


def group_zscore(x, groups, ddof: int = 1, min_count: int = 2) -> pd.Series:
    """Industry/sector-neutral z-score: (x - group mean)/group stdev.
    Groups with fewer than min_count valid names -> NaN."""
    s = _cs(x).astype(float)
    g = _cs(groups).reindex(s.index)
    grp = s.groupby(g)
    mu = grp.transform("mean")
    sd = grp.transform(lambda v: v.std(ddof=ddof))
    n = grp.transform("count")
    out = (s - mu) / sd.where(sd > 0)
    return out.where(n >= min_count)


def group_demean(x, groups) -> pd.Series:
    s = _cs(x).astype(float)
    g = _cs(groups).reindex(s.index)
    return s - s.groupby(g).transform("mean")


def group_rank(x, groups, pct: bool = True) -> pd.Series:
    """Rank within group (average ties); pct=True -> rank/n_group in (0, 1]."""
    s = _cs(x).astype(float)
    g = _cs(groups).reindex(s.index)
    return s.groupby(g).rank(method="average", pct=pct)


def group_quantile_bucket(x, groups, q: int = 10) -> pd.Series:
    """Quantile buckets computed separately inside each group (sector-neutral sorts)."""
    s = _cs(x).astype(float)
    g = _cs(groups).reindex(s.index)
    out = pd.Series(np.nan, index=s.index)
    for _, idx in s.groupby(g).groups.items():
        out.loc[idx] = quantile_bucket(s.loc[idx], q)
    return out


def size_buckets(mcap, is_nyse, q: int = 2) -> pd.Series:
    """Size buckets using NYSE breakpoints (Fama-French): q=2 -> median split;
    q=5/10 -> NYSE quintiles/deciles. Labels 1 (small) .. q (big)."""
    return quantile_bucket(mcap, q, universe_mask=is_nyse)


def value_weights(mcap, members=None) -> pd.Series:
    m = _cs(mcap).astype(float)
    if members is not None:
        m = m.where(_cs(members).reindex(m.index).fillna(False).astype(bool))
    return m / m.sum()
