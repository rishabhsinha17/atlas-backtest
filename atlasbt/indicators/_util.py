"""Internal helpers shared by the indicator modules.

Every public indicator accepts numpy arrays or pandas Series. When given a
Series the result is a Series (or DataFrame) on the same index; when given an
array the result is an ndarray. All helpers here are causal.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

try:  # numba is optional; everything works (slower) without it.
    from numba import njit as _njit

    def njit(*args, **kwargs):  # pragma: no cover - thin wrapper
        kwargs.setdefault("cache", False)
        return _njit(*args, **kwargs)

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(f):
            return f

        return deco


def as_float_array(x: Any) -> np.ndarray:
    """Float64 copy-free view of a Series/array/list."""
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=np.float64, na_value=np.nan)
    if isinstance(x, pd.DataFrame):
        raise TypeError("expected a 1-D series, got a DataFrame")
    return np.asarray(x, dtype=np.float64)


def index_of(*xs) -> pd.Index | None:
    for x in xs:
        if isinstance(x, (pd.Series, pd.DataFrame)):
            return x.index
    return None


def wrap(values: np.ndarray, like_index: pd.Index | None, name: str | None = None):
    if like_index is None:
        return values
    return pd.Series(values, index=like_index, name=name)


def wrap_frame(cols: dict[str, np.ndarray], like_index: pd.Index | None):
    if like_index is None:
        return {k: v for k, v in cols.items()}
    return pd.DataFrame(cols, index=like_index)


def check_period(n: int, name: str = "n") -> int:
    n = int(n)
    if n < 1:
        raise ValueError(f"{name} must be >= 1, got {n}")
    return n


def first_valid(a: np.ndarray) -> int:
    """Index of first non-NaN value, or len(a) if none."""
    ok = np.flatnonzero(~np.isnan(a))
    return int(ok[0]) if ok.size else len(a)


def rolling_apply_1d(a: np.ndarray, n: int, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply a vectorised reducer `fn(windows) -> (m,)` over trailing windows of
    length n (window ending at t, inclusive). Positions t < n-1 get NaN. Any
    NaN inside a window yields NaN (the reducer sees the NaN)."""
    out = np.full(a.shape[0], np.nan)
    if a.shape[0] < n:
        return out
    w = np.lib.stride_tricks.sliding_window_view(a, n)
    out[n - 1:] = fn(w)
    return out


def rolling_max(a: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(a).rolling(n, min_periods=n).max().to_numpy()


def rolling_min(a: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(a).rolling(n, min_periods=n).min().to_numpy()


def shift(a: np.ndarray, k: int = 1) -> np.ndarray:
    """Causal lag by k>=0 positions (NaN fill)."""
    out = np.full_like(a, np.nan, dtype=np.float64)
    if k == 0:
        return a.astype(np.float64).copy()
    if k < 0:
        raise ValueError("negative shifts look into the future; not allowed in indicators")
    if k < a.shape[0]:
        out[k:] = a[:-k]
    return out
