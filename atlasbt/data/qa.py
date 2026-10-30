"""Data QA: bad ticks, OHLC consistency, calendar gaps, zero volume, split cross-check,
stale prices, adjusted-close cross-check. `qa_report(portal, symbols)` returns one row per
symbol and (optionally) writes cache/qa/qa_<freq>_<date>.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, frames

COMMON_RATIOS = np.array([2, 3, 4, 5, 7, 8, 10, 15, 20, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10, 3 / 2, 2 / 3, 5 / 4])


def ohlc_violations(b: pd.DataFrame) -> pd.Series:
    hi_bad = b["high"] + 1e-9 < b[["open", "close"]].max(axis=1)
    lo_bad = b["low"] - 1e-9 > b[["open", "close"]].min(axis=1)
    nonpos = (b[["open", "high", "low", "close"]] <= 0).any(axis=1)
    return hi_bad | lo_bad | nonpos | (b["low"] > b["high"])


def bad_ticks(close: pd.Series, k: float = 6.0, min_move: float = 0.08, revert: float = 0.25,
              market: pd.Series | None = None, market_move: float = 0.03) -> pd.Series:
    """Spike-and-revert glitches: |r_t| > max(k x rolling 60-bar median |r|, min_move) AND the next
    bar reverses almost all of it (|r_t + r_{t+1}| < revert x |r_t|). Default 6 MADs / 8 % (red-team:
    DoltHub CB 2014-09-08 -13.6 % whole-bar glitch must be caught). `market` (e.g. SPY closes on the
    same index): days when the market itself moved > market_move are never flagged (2020-03 crash /
    rebound days are real)."""
    r = np.log(close).diff()
    mad = r.abs().rolling(60, min_periods=20).median().shift(1)
    big = r.abs() > np.maximum(k * mad.fillna(r.abs().median()), min_move)
    rev = (r + r.shift(-1)).abs() < revert * r.abs()
    out = big & rev
    if market is not None:
        mr = np.log(market).diff().reindex(r.index)
        mr1 = mr.shift(-1)
        out &= ~((mr.abs() > market_move) | (mr1.abs() > market_move)).fillna(False)
    return out.fillna(False)


def calendar_gaps(b: pd.DataFrame) -> pd.DatetimeIndex:
    d = frames.to_naive_date(b.index)
    if not len(d):
        return pd.DatetimeIndex([])
    sch = frames.schedule().index
    need = sch[(sch >= d[0]) & (sch <= d[-1])]
    return need[~need.isin(d)]


def stale_runs(close: pd.Series, n: int = 5) -> int:
    same = close.diff().fillna(1) == 0
    runs = same.groupby((~same).cumsum()).sum()
    return int((runs >= n - 1).sum())


def split_crosscheck(b: pd.DataFrame, splits: pd.DataFrame, tol: float = 0.15) -> dict:
    """Overnight raw price jumps (open_t / close_{t-1}) that look like a split ratio vs
    recorded splits (ratio r -> expected price factor 1/r)."""
    d = frames.to_naive_date(b.index)
    gap = pd.Series(b["open"].to_numpy() / b["close"].shift(1).to_numpy(), index=d)
    # only overnight gaps between CONSECUTIVE sessions are evidence
    sch = frames.schedule().index
    pos = sch.get_indexer(d)
    consecutive = np.r_[False, np.diff(pos) == 1]
    gap = gap[consecutive & (pos >= 0)]
    rec = {pd.Timestamp(x).normalize(): float(r) for x, r in zip(splits.get("ex_date", []), splits.get("ratio", []))}
    suspects = []
    for day, g in gap.dropna().items():
        if 0.6 < g < 1.6:
            continue
        near = np.abs(np.log(g) - np.log(1 / COMMON_RATIOS)).min() < tol
        if near and day not in rec:
            suspects.append(str(day.date()))
    missing_jump = []
    for day, r in rec.items():
        if day in gap.index:
            g = gap.loc[day]
            if np.isfinite(g) and abs(np.log(g * r)) > 0.35:   # after a r:1 split, open ~ close/r
                missing_jump.append(f"{day.date()} ratio {r:g} gap {g:.3f}")
    return {"unrecorded_split_suspects": suspects, "recorded_split_without_jump": missing_jump}


def adjclose_crosscheck(portal, symbol: str, as_of) -> float:
    """Max abs daily total-return difference between our PIT-adjusted closes and Yahoo's Adj
    Close (both as of the latest data). Small (< 1e-3) means actions & un-adjustment agree."""
    raw = portal._read("bars_1d", "yahoo", symbol)
    if "yahoo_adjclose" not in raw or raw.empty:
        return float("nan")
    adj = portal.bars(symbol, "1d", raw.index[0], raw.index[-1] + pd.Timedelta(days=1), as_of, adjusted=True)
    a = adj["close"].pct_change()
    y = raw["yahoo_adjclose"].reindex(adj.index).pct_change()
    return float((a - y).abs().max())


def qa_daily(portal, symbol: str, as_of=None) -> dict:
    src = portal._bar_source(symbol, "1d")
    if src is None:
        return {"symbol": symbol, "source": None, "rows": 0}
    b = portal._read("bars_1d", src, symbol)
    acts = portal._read("actions", "yahoo", symbol)
    sp = acts[acts["type"] == "SPLIT"] if len(acts) else pd.DataFrame(columns=["ex_date", "ratio"])
    gaps = calendar_gaps(b)
    sc = split_crosscheck(b, sp)
    out = {"symbol": symbol, "source": src, "rows": len(b), "first": str(b.index[0].date()),
           "last": str(b.index[-1].date()), "ohlc_violations": int(ohlc_violations(b).sum()),
           "bad_ticks": int(bad_ticks(b["close"]).sum()),
           "zero_volume_days": int((b["volume"].fillna(0) <= 0).sum()) if src != "cboe" else None,
           "calendar_gaps": len(gaps), "calendar_gap_sample": [str(x.date()) for x in gaps[:5]],
           "stale_runs_5d": stale_runs(b["close"]), "splits_recorded": len(sp),
           "unrecorded_split_suspects": sc["unrecorded_split_suspects"][:10],
           "recorded_split_without_jump": sc["recorded_split_without_jump"][:10]}
    if src == "yahoo" and as_of is not None:
        out["adjclose_max_abs_ret_diff"] = adjclose_crosscheck(portal, symbol, as_of)
    return out


def qa_intraday(portal, symbol: str, freq: str = "1m") -> dict:
    src = portal._bar_source(symbol, freq)
    if src is None:
        return {"symbol": symbol, "freq": freq, "rows": 0}
    b = portal._read(f"bars_{freq}", src, symbol)
    rth = b[b["session"] == "RTH"]
    d = frames.to_naive_date(rth.index)
    per = pd.Series(1, index=d).groupby(level=0).sum()
    sch = frames.schedule().reindex(per.index)
    expected = ((sch["close"] - sch["open"]) / frames.freq_to_timedelta(freq)).astype(int)
    return {"symbol": symbol, "freq": freq, "source": src, "rows": len(b), "sessions": len(per),
            "first": str(b.index[0]), "last": str(b.index[-1]),
            "ohlc_violations": int(ohlc_violations(b).sum()),
            "rth_zero_volume_bars": int((rth["volume"].fillna(0) <= 0).sum()),
            "rth_missing_bars": int((expected - per).clip(lower=0).sum()),
            "rth_completeness": round(float(per.sum() / max(expected.sum(), 1)), 4),
            "pre_bars": int((b["session"] == "PRE").sum()), "post_bars": int((b["session"] == "POST").sum()),
            "bad_ticks": int(bad_ticks(rth["close"], k=15, min_move=0.03).sum())}


def qa_report(portal, symbols, freq: str = "1d", as_of=None, write: bool = True) -> pd.DataFrame:
    rows = []
    for s in symbols:
        try:
            rows.append(qa_daily(portal, s, as_of) if freq == "1d" else qa_intraday(portal, s, freq))
        except Exception as e:  # report, don't crash
            rows.append({"symbol": s, "error": str(e)})
    df = pd.DataFrame(rows)
    if write:
        d = config.CACHE_ROOT / "qa"
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / f"qa_{freq}_{pd.Timestamp.now():%Y%m%d}.csv", index=False)
    return df
