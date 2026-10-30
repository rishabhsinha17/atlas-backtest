"""Golden-result harness for performance work (docs/PERF.md).

Every optimisation of the engine / strategy hot paths must leave results BIT-FOR-BIT identical. This module
defines ~12 representative real-cache runs (``CASES``), runs one case with whatever ``atlasbt`` is first on
``sys.path``, and reduces the result to a canonical, order-preserving digest of every output: equity, returns,
fills, trades, positions, ledger, fidelity scorecard, status/reason, diagnostics (orders, rejects, order
events, warnings, logs, ...) and meta (minus wall-clock fields).

CLI (always offline)::

    # record (e.g. against the pre-optimisation snapshot of the code)
    ATLASBT_OFFLINE=1 ATLASBT_CACHE=/home/user/atlas_bt/cache \
        python tests/perf/golden.py record --code-root results/_worktrees/perf_base --out tests/perf/golden S103
    # check the live tree against the stored golden files
    ATLASBT_OFFLINE=1 python tests/perf/golden.py check S103 S046 ...

The per-case files under ``tests/perf/golden/<case>/`` are ``digest.json`` (hashes + wall-clock) and small
parquet copies of equity / fills / trades (for diffing when a hash differs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GOLDEN_DIR = HERE / "golden"

MEGA20 = ("SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "JPM",
          "V", "MA", "UNH", "XOM", "LLY", "COST", "NFLX")

# name -> spec. ``universe`` overrides the class universe (harness subclass; strategy logic unchanged).
CASES: dict[str, dict] = {
    "S046_etf_2015_2018": dict(sid="S046", start="2015-01-02", end="2018-12-31"),
    "S068_etf_2015_2018": dict(sid="S068", start="2015-01-02", end="2018-12-31"),
    # SP500 cross-sectional: Q1 only for S085/S074 -- at f9702e1 one SIMULATED YEAR of either takes ~3 CPU-hours
    # (the PIT S&P 500 membership was rebuilt inside every Portal.borrow call), too slow to re-run as a test
    "S085_sp500_2019q1": dict(sid="S085", start="2019-01-02", end="2019-03-29"),
    "S074_sp500_2019q1": dict(sid="S074", start="2019-01-02", end="2019-03-29"),
    "S106_sp500_2019": dict(sid="S106", start="2019-01-02", end="2019-12-31"),
    # 1m: the Yahoo 1m window (the last ~30 days in the cache), ETF_CORE + 16 mega caps (static universe)
    "S001_1m_yahoo_2026_10": dict(sid="S001", start="2026-10-01", end="2026-10-27", universe=MEGA20),
    "S013_1m_yahoo_2026_10": dict(sid="S013", start="2026-10-01", end="2026-10-27", universe=MEGA20),
    "S103_spy_2010_2018": dict(sid="S103", start="2010-01-04", end="2018-12-31"),
    "S105_2015_2018": dict(sid="S105", start="2015-01-02", end="2018-12-31"),
    "S126_spy_opt_2020h1": dict(sid="S126", start="2020-01-02", end="2020-06-30"),
    "S134_opt_2020": dict(sid="S134", start="2020-01-02", end="2020-12-31"),
    "S129_spx_2020h1": dict(sid="S129", start="2020-01-02", end="2020-06-30"),
    "S138_spx_2020h1": dict(sid="S138", start="2020-01-02", end="2020-06-30"),
    # short-holding control (runner's replicate CoinFlip: long/short SPY, every session ends with borrow())
    "COINFLIP_spy_2015h1": dict(cls="atlasbt.run.replicate:CoinFlip", start="2015-01-02", end="2015-06-30"),
    "S046_etf_short_2015h1": dict(sid="S046", start="2015-01-02", end="2015-06-30", params={"short": True}),
    "COINFLIP_spy_2015": dict(cls="atlasbt.run.replicate:CoinFlip", start="2015-01-02", end="2015-12-31"),
}

#: meta / diagnostics keys that legitimately differ run to run (wall clock, object ids)
VOLATILE_KEYS = {"runtime", "runtime_sec", "runtime_s", "elapsed", "wall_clock", "t_run", "code_hash"}


# ----------------------------------------------------------------------------- canonical digest
def _canon(x, depth: int = 0):
    """Deterministic, type-tagged text form of nested results (floats via repr = exact bits)."""
    import numpy as np
    import pandas as pd

    if depth > 40:
        return "<deep>"
    if x is None or isinstance(x, (bool, str, int)):
        return repr(x)
    if isinstance(x, float):
        return "f" + (repr(x) if not math.isnan(x) else "nan")
    if isinstance(x, (np.floating,)):
        return _canon(float(x), depth)
    if isinstance(x, (np.integer, np.bool_)):
        return repr(x.item())
    if isinstance(x, pd.DataFrame):
        return "DF" + frame_digest(x)
    if isinstance(x, pd.Series):
        return "S" + frame_digest(x.to_frame("__v__"))
    if isinstance(x, np.ndarray):
        if x.dtype.kind in "fiub":
            return "A" + str(x.dtype) + hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        return "A[" + ",".join(_canon(v, depth + 1) for v in x.tolist()) + "]"
    if isinstance(x, dict):
        items = sorted(((str(k), v) for k, v in x.items() if str(k) not in VOLATILE_KEYS), key=lambda kv: kv[0])
        return "{" + ",".join(k + ":" + _canon(v, depth + 1) for k, v in items) + "}"
    if isinstance(x, (list, tuple)):
        return "[" + ",".join(_canon(v, depth + 1) for v in x) + "]"
    if isinstance(x, (set, frozenset)):
        return "set(" + ",".join(sorted(_canon(v, depth + 1) for v in x)) + ")"
    if isinstance(x, pd.Timestamp):
        return "T" + repr(x)
    if hasattr(x, "value") and hasattr(x, "name") and type(x).__module__.startswith("atlasbt"):   # enums
        return f"E{type(x).__name__}.{x.name}"
    if hasattr(x, "__dataclass_fields__"):
        return type(x).__name__ + _canon({k: getattr(x, k) for k in x.__dataclass_fields__}, depth + 1)
    r = repr(x)
    if " at 0x" in r:      # object identity in repr: not a result
        return "<" + type(x).__name__ + ">"
    return r


def frame_digest(df) -> str:
    """sha256 over index + every column (numeric columns bit-exact, others via _canon of each value)."""
    import numpy as np
    import pandas as pd

    h = hashlib.sha256()
    if df is None:
        return "none"
    h.update(repr(list(map(str, df.columns))).encode())
    h.update(repr([str(t) for t in df.dtypes]).encode())
    cols = [("__index__", pd.Series(df.index))] + [(str(c), df[c]) for c in df.columns]
    for name, s in cols:
        h.update(name.encode())
        if isinstance(s, pd.DataFrame):          # duplicated column label
            s = s.iloc[:, 0]
        kind = s.dtype.kind if hasattr(s.dtype, "kind") else "O"
        if kind in "fiub":
            h.update(np.ascontiguousarray(s.to_numpy()).tobytes())
        elif kind == "M":
            h.update(np.asarray(s.astype("int64") if s.dt.tz is None else s.dt.tz_convert("UTC").astype("int64")).tobytes())
            h.update(str(s.dtype).encode())
        else:
            for v in s.tolist():
                h.update(_canon(v).encode())
                h.update(b"\x1f")
    return h.hexdigest()


def digest_result(res) -> dict:
    """{component: sha256} for every output of a BacktestResult."""
    out = {
        "status": str(getattr(res.status, "value", res.status)),
        "not_testable_reason": hashlib.sha256(str(res.not_testable_reason).encode()).hexdigest(),
        "equity": frame_digest(res.equity.to_frame("nav") if res.equity is not None else None),
        "returns": frame_digest(res.returns.to_frame("r") if res.returns is not None else None),
        "fills": frame_digest(res.fills),
        "trades": frame_digest(res.trades),
        "positions": frame_digest(res.positions),
        "ledger": frame_digest(res.ledger),
        "fidelity": hashlib.sha256(_canon(res.fidelity).encode()).hexdigest(),
        "params": hashlib.sha256(_canon(res.params).encode()).hexdigest(),
    }
    diag = dict(res.diagnostics or {})
    for k in sorted(diag):
        if k in VOLATILE_KEYS:
            continue
        out["diag." + k] = hashlib.sha256(_canon(diag[k]).encode()).hexdigest()
    meta = dict(res.meta or {})
    for k in sorted(meta):
        if k in VOLATILE_KEYS:
            continue
        out["meta." + k] = hashlib.sha256(_canon(meta[k]).encode()).hexdigest()
    return out


def summary(res) -> dict:
    import numpy as np

    eq = res.equity
    return {"status": str(getattr(res.status, "value", res.status)), "reason": str(res.not_testable_reason)[:300],
            "n_fills": int(len(res.fills)) if res.fills is not None else 0,
            "n_trades": int(len(res.trades)) if res.trades is not None else 0,
            "n_sessions": int(len(eq)) if eq is not None else 0,
            "final_nav": float(eq.iloc[-1]) if eq is not None and len(eq) else float("nan"),
            "grade": (res.fidelity or {}).get("grade"),
            "guard_violations": (res.diagnostics or {}).get("lookahead_guard_violations"),
            "nav_sum": float(np.nansum(eq.to_numpy())) if eq is not None and len(eq) else 0.0}


# ----------------------------------------------------------------------------- running
def make_strategy(spec: dict):
    if spec.get("cls"):
        import importlib
        mod, _, name = spec["cls"].partition(":")
        cls = getattr(importlib.import_module(mod), name)
    else:
        from atlasbt.strategies import registry as R
        cls = R.get(spec["sid"])
    if spec.get("universe") is not None:
        cls = type(cls.__name__, (cls,), {"universe": tuple(spec["universe"]), "universe_name": None,
                                          "__module__": cls.__module__})
    params = dict(cls.params)
    params.update(spec.get("params") or {})
    return cls(**params)


def run_case(name: str):
    """Run one golden case with the runner's production setup (FastPortal + make_config defaults)."""
    os.environ.setdefault("ATLASBT_OFFLINE", "1")
    from atlasbt.engine import Backtester, BacktestConfig
    try:
        from atlasbt.run.fastportal import FastPortal as P
    except Exception:  # pragma: no cover
        from atlasbt.data.portal import Portal as P
    try:
        from atlasbt.run.offline import enable
        enable(strict=False)
    except Exception:  # pragma: no cover
        pass
    spec = CASES[name]
    strat = make_strategy(spec)
    cfg = dict(initial_cash=1_000_000.0, fees="IBKR_PRO_TIERED", use_quotes="auto", min_universe_coverage=0.95)
    cfg.update(spec.get("config") or {})
    portal = P(**(spec.get("portal_kw") or {}))
    t0, c0 = time.perf_counter(), time.process_time()
    res = Backtester(strat, portal, BacktestConfig(**cfg)).run(spec["start"], spec["end"])
    LAST_CPU[0] = time.process_time() - c0      # CPU seconds (the machine is shared: wall clock is noisy)
    return res, time.perf_counter() - t0


LAST_CPU = [float("nan")]


def save_case(name: str, res, secs: float, out_dir: Path) -> dict:
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    dig = digest_result(res)
    rec = {"case": name, "spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CASES[name].items()},
           "seconds": round(secs, 2), "cpu_seconds": round(LAST_CPU[0], 2), "summary": summary(res), "digest": dig}
    (d / "digest.json").write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
    _save_frames(res, d)
    return rec


def _save_frames(res, d: Path) -> None:
    """Small parquet copies (equity/fills/trades) for diffing; object columns stringified."""
    import pandas as pd

    frames = {"equity": res.equity.to_frame("nav") if res.equity is not None else None,
              "fills": res.fills, "trades": res.trades}
    for k, df in frames.items():
        if df is None:
            continue
        df = df.copy()
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].map(lambda v: None if v is None else str(v))
        try:
            df.to_parquet(d / f"{k}.parquet", compression="zstd")
        except Exception:
            df.to_pickle(d / f"{k}.pkl")


def load_golden(name: str, golden_dir: Path = GOLDEN_DIR) -> dict:
    return json.loads((golden_dir / name / "digest.json").read_text())


def compare(golden: dict, dig: dict) -> list[str]:
    g = golden["digest"]
    return [k for k in sorted(set(g) | set(dig)) if g.get(k) != dig.get(k)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "check", "list"])
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--code-root", default=None, help="put this directory first on sys.path (atlasbt to test)")
    ap.add_argument("--out", default=str(GOLDEN_DIR))
    ap.add_argument("--golden", default=str(GOLDEN_DIR))
    ap.add_argument("--timing-out", default=None, help="append {case, seconds, ok} json lines here")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path(a.code_root).resolve()) if a.code_root else str(REPO))
    if a.mode == "list":
        print("\n".join(CASES))
        return 0
    names = a.cases or list(CASES)
    rc = 0
    import atlasbt
    print("atlasbt from", Path(atlasbt.__file__).parent, flush=True)
    for nm in names:
        res, secs = run_case(nm)
        if a.mode == "record":
            rec = save_case(nm, res, secs, Path(a.out))
            print(f"{nm}: recorded {secs:.1f}s {rec['summary']}", flush=True)
            ok = True
        else:
            dig = digest_result(res)
            bad = compare(load_golden(nm, Path(a.golden)), dig)
            ok = not bad
            print(f"{nm}: {'IDENTICAL' if ok else 'DIFF ' + str(bad)} {secs:.1f}s cpu {LAST_CPU[0]:.1f}s "
                  f"{summary(res)}", flush=True)
            rc |= 0 if ok else 1
        if a.timing_out:
            with open(a.timing_out, "a") as f:
                f.write(json.dumps({"case": nm, "seconds": round(secs, 2), "cpu_seconds": round(LAST_CPU[0], 2),
                                    "ok": ok, "mode": a.mode}) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
