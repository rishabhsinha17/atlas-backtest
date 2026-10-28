"""Bulk loaders for full-table CSV exports of the DoltHub post-no-preference databases.

Source: Hugging Face mirrors by siddharthmb (CSV dumps of the DoltHub tables, last
updated 2026-01-29; no license declared on the mirror -- the underlying DoltHub repos are
community data without an explicit license):
  https://huggingface.co/datasets/siddharthmb/stocks-options/resolve/main/option_chain.csv (9.1 GB)
  https://huggingface.co/datasets/siddharthmb/stocks-ohlcv/resolve/main/ohlcv.csv          (1.2 GB)
These are ~100x faster than the DoltHub SQL API (1000-row cap, ~5-40 s per query); the SQL
API (providers/dolthub.py) is then used only for dates after the mirror's last date.
The mirror was cross-checked against the SQL API on sample (date, symbol) pairs (see
tests/data/test_dolthub.py and docs/DATA.md).

Downloaded files are treated as untrusted data: parsed as CSV only, never executed.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from atlasbt.core.types import Fidelity

from .. import frames
from ..store import ParquetStore, default_store
from .dolthub import snapshot_ts, SOURCE

log = logging.getLogger("atlasbt.data.dolthub_bulk")


def occ_symbols(underlying: pd.Series, expiry: pd.Series, right: pd.Series, strike: pd.Series) -> pd.Series:
    """Vectorized core.types.Instrument.option(...).symbol."""
    root = underlying.str.upper().str.pad(6, side="right")
    k = (strike.astype(float) * 1000).round().astype("int64").astype(str).str.zfill(8)
    return root + pd.to_datetime(expiry).dt.strftime("%y%m%d") + right + k


def canonical_chain(df: pd.DataFrame) -> pd.DataFrame:
    """Raw option_chain rows (strings or numbers) -> canonical option_chain frame (vectorized)."""
    out = pd.DataFrame(index=df.index)
    dates = pd.to_datetime(df["date"])
    uniq = {d: snapshot_ts(d) for d in dates.unique()}
    out["ts"] = dates.map(uniq)
    out["underlying"] = df["act_symbol"].astype(str).str.upper()
    out["expiry"] = pd.to_datetime(df["expiration"])
    out["strike"] = pd.to_numeric(df["strike"], errors="coerce").astype(float)
    out["right"] = np.where(df["call_put"].astype(str).str.upper().str.startswith("C"), "C", "P")
    out["symbol"] = occ_symbols(out["underlying"], out["expiry"], pd.Series(out["right"], index=df.index), out["strike"])
    bid = pd.to_numeric(df["bid"], errors="coerce").astype(float)
    ask = pd.to_numeric(df["ask"], errors="coerce").astype(float)
    out["bid"], out["ask"] = bid, ask
    out["mid"] = np.where((bid > 0) | (ask > 0), 0.5 * (bid + ask), np.nan)
    out["iv"] = pd.to_numeric(df["vol"], errors="coerce")
    for c in ("delta", "gamma", "vega", "theta", "rho"):
        out[c] = pd.to_numeric(df[c], errors="coerce")
    out["underlying_price"] = np.nan
    out["volume"] = np.nan
    out["open_interest"] = np.nan
    out["exercise"] = "AMERICAN"
    out["settlement"] = "PHYSICAL"
    out["multiplier"] = 100.0
    out["fidelity"] = int(Fidelity.T2_PROXY)
    out["source_date"] = dates
    return out


N_BUCKETS = 64


def _bucket(sym: pd.Series) -> np.ndarray:
    # deterministic, process-independent bucket of a symbol
    return np.array([sum(map(ord, x)) % N_BUCKETS for x in sym.astype(str)])


def load_option_chain_csv(path: str | Path, symbols: Optional[Iterable[str]] = None,
                          store: Optional[ParquetStore] = None, chunksize: int = 4_000_000,
                          parts_dir: Optional[str | Path] = None) -> dict:
    """Filter the full option_chain CSV to `symbols` (None = all) and write
    option_chain/dolthub/<SYM>/<YYYY>.parquet. Two phases: (1) stream the CSV in chunks,
    keep wanted rows, write them into 64 symbol-hash buckets; (2) per bucket, canonicalize
    each symbol (adds underlying_price from yahoo raw closes) and write it to the store.
    `parts_dir`: resume phase 2 from an earlier phase-1 directory of part_*.parquet files."""
    store = store or default_store()
    want = {s.upper() for s in symbols} if symbols else None
    tmp = Path(tempfile.mkdtemp(prefix="optbuckets_", dir=str(store.root.parent)))
    n_in = n_keep = 0
    try:
        if parts_dir is not None:
            src_iter = (pq.read_table(p).to_pandas() for p in sorted(Path(parts_dir).glob("part_*.parquet")))
        else:
            src_iter = pd.read_csv(path, chunksize=chunksize, dtype=str, keep_default_na=False, na_values=[""])
        for i, chunk in enumerate(src_iter):
            n_in += len(chunk)
            if want is not None:
                chunk = chunk[chunk["act_symbol"].str.upper().isin(want)]
            if not len(chunk):
                continue
            n_keep += len(chunk)
            bk = _bucket(chunk["act_symbol"])
            for bucket in np.unique(bk):
                d = tmp / f"b{bucket:02d}"
                d.mkdir(exist_ok=True)
                pq.write_table(pa.Table.from_pandas(chunk[bk == bucket], preserve_index=False), d / f"{i:05d}.parquet")
            log.info("option_chain chunk %d: read %d rows, kept %d", i, n_in, n_keep)
        syms_done, last_date = [], None
        for d in sorted(tmp.iterdir()):
            df = pd.concat([pq.read_table(p).to_pandas() for p in sorted(d.glob("*.parquet"))], ignore_index=True)
            for s, raw in df.groupby("act_symbol"):
                chain = canonical_chain(raw.reset_index(drop=True))
                closes = store.read("bars_1d", "yahoo", s.replace(".", "-"), columns=["close", "known_at"])
                if len(closes):
                    m = dict(zip(closes["known_at"], closes["close"]))
                    chain["underlying_price"] = chain["ts"].map(m).astype(float)
                chain = chain.drop_duplicates(["ts", "symbol"], keep="last")
                store.write("option_chain", SOURCE, s, chain.reset_index(drop=True), partition="year", time_col="ts",
                            key=["ts", "symbol"], fidelity=int(Fidelity.T2_PROXY), replace=True,
                            extra={"snapshot_convention": "close of last session <= dolthub date",
                                   "origin": "hf:siddharthmb/stocks-options option_chain.csv",
                                   "first_ts": str(chain["ts"].min()), "last_ts": str(chain["ts"].max())})
                syms_done.append(s)
                dmax = raw["date"].max()
                last_date = max(last_date, dmax) if last_date else dmax
            log.info("option_chain bucket %s done (%d symbols so far)", d.name, len(syms_done))
        store.write("reference", SOURCE, "option_chain_bulk",
                    pd.DataFrame({"symbol": syms_done, "bulk_last_date": last_date}), key=["symbol"])
        return {"rows_read": n_in, "rows_kept": n_keep, "symbols": len(syms_done), "last_date": last_date}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def load_ohlcv_csv(path: str | Path, symbols: Optional[Iterable[str]] = None,
                   store: Optional[ParquetStore] = None, chunksize: int = 5_000_000) -> dict:
    """DoltHub stocks.ohlcv (2011-01-03 ->, includes since-delisted symbols) ->
    bars_1d/dolthub/<SYM>. Prices appear to be AS-TRADED (raw; verified on AAPL around the
    2014-06-09 and 2020-08-31 splits in tests) -- volume as reported."""
    store = store or default_store()
    want = {s.upper() for s in symbols} if symbols else None
    parts = []
    n_in = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, dtype={"act_symbol": str}, keep_default_na=False,
                             na_values=[""]):
        n_in += len(chunk)
        chunk["act_symbol"] = chunk["act_symbol"].str.upper()
        if want is not None:
            chunk = chunk[chunk["act_symbol"].isin(want)]
        if len(chunk):
            parts.append(chunk)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out = {"rows_read": n_in, "rows_kept": len(df), "symbols": 0}
    if df.empty:
        return out
    df["date"] = pd.to_datetime(df["date"])
    for s, g in df.groupby("act_symbol"):
        g = g.set_index("date").sort_index()
        bars, dropped = frames.daily_bars(g[["open", "high", "low", "close", "volume"]])
        if bars.empty:
            continue
        store.write("bars_1d", SOURCE, s.replace(".", "-"), bars, fidelity=int(Fidelity.T1_BAR), replace=True,
                    extra={"origin": "hf:siddharthmb/stocks-ohlcv ohlcv.csv", "raw": True,
                           "first_ts": str(bars.index[0]), "last_ts": str(bars.index[-1])})
        out["symbols"] += 1
    return out


def load_eps_estimates(path: str | Path, store: Optional[ParquetStore] = None) -> dict:
    """DoltHub earnings.eps_estimate (weekly consensus snapshots, 2017-10-26 ->) mirror ->
    reference/dolthub/eps_estimate ('Current Quarter' rows only). A snapshot dated d is
    treated as known at 23:59 ET on d."""
    store = store or default_store()
    keep = []
    for chunk in pd.read_csv(path, chunksize=2_000_000, dtype=str, keep_default_na=False, na_values=[""]):
        keep.append(chunk[chunk["period"] == "Current Quarter"])
    df = pd.concat(keep, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["period_end_date"] = pd.to_datetime(df["period_end_date"], errors="coerce")
    for c in ("consensus", "recent", "count", "high", "low", "year_ago"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["act_symbol"] = df["act_symbol"].str.upper()
    store.write("reference", SOURCE, "eps_estimate", df, key=["date", "act_symbol", "period"], replace=True,
                extra={"origin": "hf:siddharthmb/stocks-earnings-eps_estimate", "known_at_rule": "date 23:59 ET"})
    return {"rows": len(df), "symbols": df["act_symbol"].nunique(), "first": str(df["date"].min().date()),
            "last": str(df["date"].max().date())}


def load_small_tables(store: Optional[ParquetStore] = None) -> dict:
    """Scout's full CSV exports in data_raw/dolthub_post_no_preference (earnings_calendar,
    eps_history, dividend, split) -> reference/dolthub/*."""
    from .. import config
    store = store or default_store()
    root = config.DATA_RAW / "dolthub_post_no_preference"
    out = {}
    for name, key in (("earnings__earnings_calendar", ["act_symbol", "date"]),
                      ("earnings__eps_history", ["act_symbol", "period_end_date"]),
                      ("stocks__dividend", ["act_symbol", "ex_date"]), ("stocks__split", ["act_symbol", "ex_date"])):
        p = root / f"{name}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, dtype={"act_symbol": str}, keep_default_na=False, na_values=[""])
        df["act_symbol"] = df["act_symbol"].str.upper()
        store.write("reference", SOURCE, name.split("__")[1], df, key=key, replace=True,
                    extra={"origin": f"data_raw/dolthub_post_no_preference/{name}.csv"})
        out[name] = len(df)
    return out
