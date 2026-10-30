"""Synthetic data builders for indicator tests (mechanics only - synthetic_*)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIX = Path(__file__).parent / "fixtures" / "stockcharts"


def load_fixture(name: str) -> pd.DataFrame:
    return pd.read_csv(FIX / name, comment="#")


def synthetic_ohlcv(n: int = 400, seed: int = 0, start: str = "2015-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.012, n)
    close = 100 * np.exp(np.cumsum(r))
    open_ = close * np.exp(rng.normal(0, 0.004, n))
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.006, n)))
    vol = rng.lognormal(13, 0.4, n)
    idx = pd.bdate_range(start, periods=n, tz="UTC") + pd.Timedelta(hours=21)
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": vol}, index=idx)


def synthetic_intraday(n_sessions: int = 25, seed: int = 0, start: str = "2024-03-04", pre: bool = False,
                       freq_min: int = 1) -> pd.DataFrame:
    """Minute bars 09:30-16:00 ET (optionally 07:00-09:30 PRE) on business days."""
    rng = np.random.default_rng(seed)
    frames = []
    px = 100.0
    for d in pd.bdate_range(start, periods=n_sessions):
        o = pd.Timestamp(d.date()).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
        rth = pd.date_range(o, o + pd.Timedelta(hours=6, minutes=30), freq=f"{freq_min}min", inclusive="left")
        idxs = [(rth, "RTH")]
        if pre:
            p0 = o - pd.Timedelta(hours=2, minutes=30)
            idxs.insert(0, (pd.date_range(p0, o, freq=f"{freq_min}min", inclusive="left"), "PRE"))
        for idx, sess in idxs:
            m = len(idx)
            c = px * np.exp(np.cumsum(rng.normal(0, 0.0008, m)))
            op = np.concatenate([[px], c[:-1]])
            h = np.maximum(op, c) * (1 + np.abs(rng.normal(0, 0.0004, m)))
            l = np.minimum(op, c) * (1 - np.abs(rng.normal(0, 0.0004, m)))
            mod = np.arange(m)
            shape = 1.0 + 2.0 * np.exp(-mod / 30) + 1.5 * np.exp(-(m - mod) / 30)
            v = rng.lognormal(8, 0.5, m) * (shape if sess == "RTH" else 0.2)
            frames.append(pd.DataFrame({"open": op, "high": h, "low": l, "close": c, "volume": v, "session": sess},
                                       index=idx.tz_convert("UTC")))
            px = c[-1]
    df = pd.concat(frames)
    df.index.name = "start"
    return df
