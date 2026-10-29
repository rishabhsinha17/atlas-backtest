"""Partitioned parquet store with provenance metadata.

Layout:   <root>/<kind>/<source>/<SYMBOL>[/<YYYY>].parquet
          kind   e.g. bars_1d, bars_1m, actions, events, option_chain, fundamentals,
                      rates, factors, reference
          source e.g. yahoo, cboe, fred, dolthub, sec, ...
Each file carries parquet key-value metadata b'atlasbt' = JSON
  {source, kind, symbol, fetched_at, fidelity, adjusted, index, key, rows, ...extra}.

Writes are idempotent upserts: new rows are merged with existing rows on the
declared key (default: the index), the newest version of a duplicated key
wins, output is sorted and atomically replaced (with an fcntl lock so parallel
fetch jobs cannot lose updates).
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config

_META_KEY = b"atlasbt"


def safe_symbol(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]", "_", symbol.upper())


@contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    with open(lock, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


class ParquetStore:
    def __init__(self, root: Optional[Path | str] = None):
        self.root = Path(root or config.STORE_ROOT)

    # ------------------------------------------------------------------ paths
    def _dir(self, kind: str, source: str) -> Path:
        return self.root / kind / source

    def path(self, kind: str, source: str, symbol: str, year: Optional[int] = None) -> Path:
        base = self._dir(kind, source)
        if year is None:
            return base / f"{safe_symbol(symbol)}.parquet"
        return base / safe_symbol(symbol) / f"{year}.parquet"

    def _files(self, kind: str, source: str, symbol: str) -> list[Path]:
        flat = self.path(kind, source, symbol)
        if flat.exists():
            return [flat]
        d = self._dir(kind, source) / safe_symbol(symbol)
        return sorted(d.glob("*.parquet")) if d.is_dir() else []

    # ------------------------------------------------------------------ write
    def write(self, kind: str, source: str, symbol: str, df: pd.DataFrame, *,
              fidelity: int | str | None = None, adjusted: bool = False,
              partition: Optional[str] = None, key: Optional[Sequence[str]] = None,
              time_col: Optional[str] = None, replace: bool = False, extra: Optional[dict] = None) -> int:
        """Upsert `df`. partition='year' splits by the year of `time_col` (default:
        the index, which must then be a DatetimeIndex). `key` = columns that
        identify a row (default: index). Returns total rows stored for symbol."""
        if df is None or len(df) == 0:
            return 0
        index_name = df.index.name if not isinstance(df.index, pd.RangeIndex) else None
        flat = df.reset_index() if index_name else df.reset_index(drop=True)
        keycols = list(key) if key else ([index_name] if index_name else list(flat.columns))
        meta = {"source": source, "kind": kind, "symbol": symbol.upper(), "fidelity": fidelity,
                "adjusted": bool(adjusted), "index": index_name, "key": keycols,
                "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(), **(extra or {})}
        if partition == "year":
            tcol = time_col or index_name
            years = pd.DatetimeIndex(flat[tcol]).year
            total = 0
            for y in sorted(set(years)):
                part = flat[years == y]
                total += self._upsert(self.path(kind, source, symbol, int(y)), part, keycols, meta, replace)
            return total
        return self._upsert(self.path(kind, source, symbol), flat, keycols, meta, replace)

    def _upsert(self, path: Path, flat: pd.DataFrame, keycols, meta, replace: bool) -> int:
        with _locked(path):
            if path.exists() and not replace:
                old = pq.read_table(path).to_pandas()
                old_meta = self._read_meta(path)
                merged = pd.concat([old, flat], ignore_index=True)
                merged = merged.drop_duplicates(subset=keycols, keep="last")
                meta = {**old_meta, **meta, "first_fetched_at": old_meta.get("first_fetched_at",
                                                                             old_meta.get("fetched_at"))}
            else:
                merged = flat.drop_duplicates(subset=keycols, keep="last")
                meta = {**meta, "first_fetched_at": meta["fetched_at"]}
            merged = merged.sort_values(keycols, kind="mergesort").reset_index(drop=True)
            meta["rows"] = int(len(merged))
            table = pa.Table.from_pandas(merged, preserve_index=False)
            md = dict(table.schema.metadata or {})
            md[_META_KEY] = json.dumps(meta, default=str).encode()
            table = table.replace_schema_metadata(md)
            tmp = path.with_suffix(".tmp%d" % os.getpid())
            pq.write_table(table, tmp, compression="zstd")
            tmp.replace(path)
            return len(merged)

    # ------------------------------------------------------------------ read
    @staticmethod
    def _read_meta(path: Path) -> dict:
        md = pq.read_schema(path).metadata or {}
        raw = md.get(_META_KEY)
        return json.loads(raw) if raw else {}

    def meta(self, kind: str, source: str, symbol: str) -> dict:
        files = self._files(kind, source, symbol)
        if not files:
            return {}
        m = self._read_meta(files[-1])
        m["rows"] = sum(self._read_meta(f).get("rows", 0) for f in files)
        m["files"] = len(files)
        return m

    def exists(self, kind: str, source: str, symbol: str) -> bool:
        return bool(self._files(kind, source, symbol))

    def read(self, kind: str, source: str, symbol: str, start=None, end=None,
             columns: Optional[Sequence[str]] = None, time_col: Optional[str] = None) -> pd.DataFrame:
        """Rows with time in [start, end) (on index or `time_col`). Empty frame if absent."""
        files = self._files(kind, source, symbol)
        if not files:
            return pd.DataFrame()
        if len(files) > 1 and (start is not None or end is not None):
            y0 = pd.Timestamp(start).year - 1 if start is not None else -10**9
            y1 = pd.Timestamp(end).year + 1 if end is not None else 10**9
            kept = [f for f in files if not f.stem.isdigit() or y0 <= int(f.stem) <= y1]
            if not kept:
                # no partition overlaps [start, end): empty frame, but keep the symbol's meta
                meta = self._read_meta(files[0])
                out = pd.DataFrame()
                out.attrs.update({"source": meta.get("source"), "fidelity": meta.get("fidelity"),
                                  "adjusted": meta.get("adjusted", False), "kind": kind})
                return out
            files = kept
        meta = self._read_meta(files[0])
        idx = meta.get("index")
        frames = [pq.read_table(f, columns=(list(columns) + ([idx] if idx else [])) if columns else None).to_pandas()
                  for f in files]
        frames = [f for f in frames if len(f)]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        tcol = time_col or idx
        if tcol and (start is not None or end is not None):
            t = df[tcol]
            m = pd.Series(True, index=df.index)
            if start is not None:
                m &= t >= _like(start, t)
            if end is not None:
                m &= t < _like(end, t)
            df = df[m]
        if idx:
            df = df.set_index(idx)
        df.attrs.update({"source": meta.get("source"), "fidelity": meta.get("fidelity"),
                         "adjusted": meta.get("adjusted", False), "kind": kind})
        return df

    # ------------------------------------------------------------------ catalog
    def symbols(self, kind: str, source: str) -> list[str]:
        d = self._dir(kind, source)
        if not d.is_dir():
            return []
        out = set()
        for p in d.iterdir():
            if p.suffix == ".parquet":
                out.add(p.stem)
            elif p.is_dir() and any(p.glob("*.parquet")):
                out.add(p.name)
        return sorted(out)

    def sources(self, kind: str) -> list[str]:
        d = self.root / kind
        return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else []

    def kinds(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir()) if self.root.is_dir() else []

    def size_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.root.rglob("*.parquet")) if self.root.exists() else 0

    def inventory(self) -> pd.DataFrame:
        rows = []
        for kind in self.kinds():
            for src in self.sources(kind):
                for sym in self.symbols(kind, src):
                    files = self._files(kind, src, sym)
                    m = self._read_meta(files[-1]) if files else {}
                    rows.append({"kind": kind, "source": src, "symbol": sym,
                                 "rows": sum(self._read_meta(f).get("rows", 0) for f in files),
                                 "bytes": sum(f.stat().st_size for f in files),
                                 "fidelity": m.get("fidelity"), "fetched_at": m.get("fetched_at"),
                                 "first": m.get("first_ts"), "last": m.get("last_ts")})
        return pd.DataFrame(rows)


def _like(x, series: pd.Series):
    """Coerce a bound to be comparable with the series' dtype (tz-aware vs naive)."""
    x = pd.Timestamp(x)
    tz = getattr(series.dtype, "tz", None)
    if tz is not None and x.tzinfo is None:
        return x.tz_localize("UTC")
    if tz is None and x.tzinfo is not None:
        return x.tz_convert("America/New_York").tz_localize(None)
    return x


_DEFAULT: Optional[ParquetStore] = None


def default_store() -> ParquetStore:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ParquetStore()
    return _DEFAULT
