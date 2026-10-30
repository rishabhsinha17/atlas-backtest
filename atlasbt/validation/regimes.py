"""Regime labels and per-regime performance breakdowns.

Labels are EX-ANTE by default: the regime assigned to day t is computed from
information available at the close of t-1 (``lag=1``), so a breakdown never
conditions on the same day's outcome.

* trend (bull/bear): SPY close vs its SMA200 (Faber 2007 style).
* VIX terciles: cut points are expanding-window terciles of VIX through t-1
  (``causal=True``) or full-sample terciles (``causal=False``, descriptive only).
* rate regimes: 3M T-bill level buckets (default <1%, 1-3%, >=3%) and/or
  direction (12-month change sign).
* decades.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from atlasbt.indicators.moving import sma


def trend_regime(spy_close: pd.Series, n: int = 200, lag: int = 1) -> pd.Series:
    s = spy_close.astype(float)
    m = sma(s, n)
    lab = pd.Series(np.where(s > m, "BULL", "BEAR"), index=s.index).where(m.notna())
    return lab.shift(lag) if lag else lab


def vix_terciles(vix: pd.Series, causal: bool = True, min_history: int = 252, lag: int = 1) -> pd.Series:
    v = vix.astype(float)
    if causal:
        lo = v.expanding(min_history).quantile(1 / 3)
        hi = v.expanding(min_history).quantile(2 / 3)
    else:
        lo = pd.Series(v.quantile(1 / 3), index=v.index)
        hi = pd.Series(v.quantile(2 / 3), index=v.index)
    lab = pd.Series(np.where(v <= lo, "VIX_LOW", np.where(v <= hi, "VIX_MID", "VIX_HIGH")), index=v.index)
    lab = lab.where(lo.notna() & v.notna())
    return lab.shift(lag) if lag else lab


def rate_regime(rate: pd.Series, bins=(0.01, 0.03), lag: int = 1, direction_window: int | None = None) -> pd.Series:
    """Level buckets of an annualised decimal rate; with direction_window
    (e.g. 252) append '_UP'/'_DOWN' by the sign of the change over that window."""
    r = rate.astype(float)
    edges = [-np.inf, *bins, np.inf]
    names = [f"RATE_LT_{bins[0]:.0%}"] + [f"RATE_{a:.0%}_{b:.0%}" for a, b in zip(bins[:-1], bins[1:])] + [
        f"RATE_GE_{bins[-1]:.0%}"]
    lab = pd.cut(r, edges, labels=names, right=False).astype(object)
    if direction_window:
        ch = r.diff(direction_window)
        lab = lab + np.where(ch > 0, "_UP", "_DOWN")
        lab = lab.where(ch.notna())
    return lab.shift(lag) if lag else lab


def decade(index: pd.DatetimeIndex) -> pd.Series:
    idx = pd.DatetimeIndex(index)
    return pd.Series([f"{(y // 10) * 10}s" for y in idx.year], index=idx)


def regime_table(returns: pd.Series, labels: pd.Series, periods: int = 252) -> pd.DataFrame:
    """Per-regime stats of daily returns: days, share, ann. mean, ann. vol,
    Sharpe (rf=0 unless returns are excess), compounded return, hit rate,
    max drawdown of the regime-concatenated path."""
    df = pd.concat([returns.rename("r"), labels.rename("regime")], axis=1).dropna()
    rows = []
    for g, sub in df.groupby("regime", sort=True):
        r = sub["r"].to_numpy(float)
        sd = r.std(ddof=1) if r.size > 1 else np.nan
        eq = np.cumprod(1 + r)
        dd = float(np.min(eq / np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:] - 1)) if r.size else np.nan
        rows.append({"regime": g, "days": r.size, "share": r.size / len(df), "ann_mean": r.mean() * periods,
                     "ann_vol": sd * np.sqrt(periods) if sd == sd else np.nan,
                     "sharpe": r.mean() / sd * np.sqrt(periods) if sd and sd > 0 else np.nan,
                     "total_return": float(eq[-1] - 1) if r.size else np.nan, "hit_rate": float(np.mean(r > 0)),
                     "max_drawdown": dd})
    return pd.DataFrame(rows).set_index("regime")


def all_regimes(returns: pd.Series, spy_close: pd.Series | None = None, vix: pd.Series | None = None,
                rate: pd.Series | None = None) -> dict[str, pd.DataFrame]:
    """Convenience: every available breakdown, aligned on the return index."""
    out = {"decade": regime_table(returns, decade(returns.index))}
    if spy_close is not None:
        out["trend"] = regime_table(returns, trend_regime(spy_close).reindex(returns.index, method="ffill"))
    if vix is not None:
        out["vix"] = regime_table(returns, vix_terciles(vix).reindex(returns.index, method="ffill"))
    if rate is not None:
        out["rates"] = regime_table(returns, rate_regime(rate).reindex(returns.index, method="ffill"))
    return out
