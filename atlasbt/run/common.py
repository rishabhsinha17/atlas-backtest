"""Shared helpers for the atlas runner: paths, provenance, seeds, (de)serialisation,
benchmark / factor / regime inputs read from the cache."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import pickle
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO / "results"

#: per-frequency classes used throughout the runner
INTRADAY_FREQS = {"1m", "5m", "15m", "30m", "60m", "1h"}


def is_intraday(freq: str) -> bool:
    return str(freq).lower() in INTRADAY_FREQS


# ----------------------------------------------------------------- json
def jsonable(x: Any):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if isinstance(x, pd.DataFrame):
        return jsonable(x.reset_index().to_dict(orient="records"))
    if isinstance(x, pd.Series):
        return jsonable({str(k): v for k, v in x.items()})
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return None if math.isnan(v) else (("inf" if v > 0 else "-inf") if math.isinf(v) else v)
    if isinstance(x, (pd.Timestamp,)):
        return str(x)
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if hasattr(x, "value") and not isinstance(x, (str, bytes)) and type(x).__module__ != "builtins":
        return jsonable(x.value)
    if hasattr(x, "__dataclass_fields__"):
        return jsonable({k: getattr(x, k) for k in x.__dataclass_fields__})
    if x is None or isinstance(x, (str, int, bool)):
        return x
    return str(x)


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=2))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def write_pickle(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=4) as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def read_pickle(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------- seeds
def stable_seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def seed_everything(*parts) -> int:
    s = stable_seed(*parts)
    random.seed(s)
    np.random.seed(s)
    return s


def param_key(p: dict) -> str:
    return "|".join(f"{k}={p[k]}" for k in sorted(p))


def config_id(p: dict) -> str:
    return hashlib.sha1(param_key(p).encode()).hexdigest()[:12]


# ----------------------------------------------------------------- provenance
def git_commit(repo: Path = REPO) -> str:
    """Commit hash: ATLASBT_RUN_COMMIT (pinned worktree launch) or read from .git files."""
    if os.environ.get("ATLASBT_RUN_COMMIT"):
        return os.environ["ATLASBT_RUN_COMMIT"]
    g = repo / ".git"
    if g.is_file():                                   # worktree: ".git" is "gitdir: <path>"
        g = Path(g.read_text().split(":", 1)[1].strip())
    try:
        head = (g / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1].strip()
            p = g / ref
            if p.exists():
                return p.read_text().strip()
            packed = g / "packed-refs"
            if packed.exists():
                for line in packed.read_text().splitlines():
                    if line.endswith(" " + ref):
                        return line.split()[0]
            return f"unborn:{ref}"
        return head
    except Exception as e:  # pragma: no cover
        return f"unknown ({e})"


_CODE_HASH = None


def code_fingerprint(repo: Path = REPO) -> str:
    """sha256 over every atlasbt/**/*.py (captures uncommitted changes)."""
    global _CODE_HASH
    if _CODE_HASH is None:
        h = hashlib.sha256()
        for f in sorted((repo / "atlasbt").rglob("*.py")):
            h.update(str(f.relative_to(repo)).encode())
            h.update(f.read_bytes())
        _CODE_HASH = h.hexdigest()[:16]
    return _CODE_HASH


def environment_info() -> dict:
    import importlib.metadata as md

    vers = {}
    for p in ("numpy", "pandas", "scipy", "statsmodels", "exchange_calendars", "pyarrow", "numba"):
        try:
            vers[p] = md.version(p)
        except Exception:
            vers[p] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": vers,
            "git_commit": git_commit(), "code_sha256_16": code_fingerprint(),
            "pinned_worktree": os.environ.get("ATLASBT_RUN_WORKTREE"),
            "runner_overlay_sha": os.environ.get("ATLASBT_RUNNER_OVERLAY_SHA"),
            "cache_root": os.environ.get("ATLASBT_CACHE", str(REPO / "cache")),
            "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat()}


def data_provenance(portal, symbols, freqs=("1d",)) -> dict:
    """Store metadata (source, fetched_at, rows, files) for every symbol/freq used."""
    out = {}
    store = getattr(portal, "store", None)
    if store is None:
        return out
    for sym in sorted(set(symbols or [])):
        for f in freqs:
            kind = f"bars_{f}"
            for src in store.sources(kind):
                try:
                    if store.exists(kind, src, sym):
                        m = store.meta(kind, src, sym)
                        out[f"{kind}/{src}/{sym}"] = {k: m.get(k) for k in
                                                       ("source", "fetched_at", "first_fetched_at", "rows", "files",
                                                        "fidelity", "adjusted")}
                except Exception as e:  # pragma: no cover
                    out[f"{kind}/{src}/{sym}"] = {"error": str(e)}
    try:
        out["_store_root"] = str(store.root)
    except Exception:
        pass
    return out


# ----------------------------------------------------------------- reference series
def session_close_index(dates) -> pd.DatetimeIndex:
    """Naive ET session dates -> UTC session close timestamps (as the engine's equity index)."""
    from atlasbt.core.clock import nyse

    cal = nyse()
    return pd.DatetimeIndex([cal.open_close(pd.Timestamp(d))[1] for d in dates])


def to_session_dates(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    return idx.normalize()


def adjusted_close(portal, symbol: str, start="1990-01-01", end=None) -> pd.Series:
    """Total-return adjusted daily close (as of the end), indexed by naive session date."""
    end = pd.Timestamp(end or pd.Timestamp.now(tz="UTC"))
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    df = portal.bars(symbol, "1d", pd.Timestamp(start, tz="UTC"), end, as_of=end, adjusted=True)
    s = df["close"].astype(float)
    s.index = to_session_dates(df.index)
    return s[~s.index.duplicated(keep="last")]


def benchmark_returns(portal, index, symbol: str = "SPY") -> pd.Series:
    """SPY total-return daily returns aligned to a result's equity index (UTC closes)."""
    px = adjusted_close(portal, symbol)
    r = px.pct_change()
    d = to_session_dates(index)
    out = pd.Series(r.reindex(d).to_numpy(), index=index)
    return out


def ff_factors_daily(portal, index) -> pd.DataFrame | None:
    try:
        f = portal.store.read("factors", "french", "F-F_RESEARCH_DATA_FACTORS_DAILY")
    except Exception:
        return None
    if f is None or not len(f):
        return None
    f = f.drop(columns=[c for c in ("known_at",) if c in f.columns]).astype(float)
    f.index = pd.DatetimeIndex(f.index).normalize()
    d = to_session_dates(index)
    out = f.reindex(d)
    out.index = index
    return out


def regime_inputs(portal) -> dict:
    out = {}
    try:
        out["spy_close"] = adjusted_close(portal, "SPY")
    except Exception:
        pass
    try:
        out["vix"] = adjusted_close(portal, "VIX")
    except Exception:
        pass
    try:
        rs = portal.rate_series("DTB3")
        col = "value" if "value" in rs.columns else rs.columns[0]
        s = rs[col].astype(float)
        s.index = to_session_dates(pd.DatetimeIndex(s.index if not isinstance(s.index, pd.RangeIndex) else rs["date"]))
        out["rate"] = s / (100.0 if s.abs().max() > 1 else 1.0)
    except Exception:
        pass
    return out


def regimes_for(returns: pd.Series, inputs: dict) -> dict:
    from atlasbt.validation.regimes import all_regimes

    r = returns.copy()
    r.index = to_session_dates(r.index)
    try:
        return all_regimes(r, inputs.get("spy_close"), inputs.get("vix"), inputs.get("rate"))
    except Exception as e:  # pragma: no cover
        return {"error": pd.DataFrame({"error": [str(e)]})}


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    @property
    def sec(self) -> float:
        return time.perf_counter() - self.t0
