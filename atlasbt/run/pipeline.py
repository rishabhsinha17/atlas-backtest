"""Per-strategy pipeline: baseline run -> grid runs -> finalize (walk-forward, holdout, PBO,
cost stress, bootstrap, report).

Every task writes its outputs atomically under ``results/runs/<SID>/`` so an interrupted atlas run
resumes where it stopped (see ``scheduler.py``):

    runs/<SID>/config.json          run options, plan row, provenance (git commit, code hash, data files)
    runs/<SID>/baseline/status.json status, runtime, reason/traceback
    runs/<SID>/baseline/result.pkl.gz  the BacktestResult (diagnostics pruned)
    runs/<SID>/baseline/equity.csv  NAV + daily returns
    runs/<SID>/grid_plan.json       configs selected for the grid (subset reason if capped)
    runs/<SID>/grid/<cfg>.json      per-config status/runtime/metrics
    runs/<SID>/grid/<cfg>.parquet   per-config equity + returns
    runs/<SID>/final/summary.json   everything the atlas step needs
    reports/<SID>/report.{md,json} + PNGs   (reporting.write_strategy_report, plus a WF/PBO section)
"""
from __future__ import annotations

import math
import shutil
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .common import (DEFAULT_RESULTS, Timer, benchmark_returns, config_id, data_provenance, environment_info,
                     ff_factors_daily, is_intraday, jsonable, param_key, read_json, read_pickle, regime_inputs,
                     regimes_for, seed_everything, write_json, write_pickle)

TERMINAL = ("OK", "NOT_TESTABLE", "FAILED", "TIMEOUT")


def _activity(sid: str, res, plan_row: dict, log_path: str = "") -> dict:
    from .activity import classify_activity, declared_note_for, diagnose

    a = classify_activity(res, plan_row.get("bar_freq") or getattr(res, "meta", {}).get("bar_freq", "1d"),
                          plan_row.get("sizing") or (getattr(res, "meta", {}) or {}).get("sizing") or "",
                          declared_note_for(sid))
    if a.get("activity") != "n/a":
        a["diagnostics"] = diagnose(res, log_path)     # always kept: options strategies need the counts
    return a


@dataclass
class RunOptions:
    results_dir: str = str(DEFAULT_RESULTS)
    max_configs: Optional[int] = None          # cap on grid configs per strategy (None = all)
    budget_intraday_min: Optional[float] = 30.0  # wall-clock budget for grid runs, intraday strategies
    budget_daily_min: Optional[float] = None     # same for daily+ strategies (None = unlimited)
    force: bool = False
    workers: int = 4
    task_timeout_min: float = 360.0
    mem_reserve_gb: float = 3.0
    bootstrap_reps: int = 1000
    holdout: float = 0.20
    wf_daily: tuple = ("3Y", "1Y")
    wf_intraday: tuple = ("6M", "2M")
    fees: str = "IBKR_PRO_TIERED"
    spread_model: Optional[str] = None      # None = engine default ('calibrated' since 2026-10-29)
    #: 'never' until the ENGINE's 'auto' mode stops switching SPY/QQQ/AAPL to quote-only fills because
    #: single sample days of quotes exist (IEX 2019-12-24, Dukascopy 2026-10-27) -- see docs/RUNNING.md
    use_quotes: str = "never"
    #: 'filter' = stock universes restricted to members with daily data (CoverageFilteredPortal, coverage
    #: recorded per result); 'strict' = engine behaviour (NOT_TESTABLE if any member lacks data)
    universe_coverage: str = "filter"
    min_universe_coverage: float = 0.95
    initial_cash: float = 1_000_000.0
    make_charts: bool = True
    start_override: Optional[str] = None
    end_override: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------- paths
def run_dir(opts: RunOptions, sid: str) -> Path:
    return Path(opts.results_dir) / "runs" / sid


def report_dir(opts: RunOptions) -> Path:
    return Path(opts.results_dir) / "reports"


def reset_strategy(opts: RunOptions, sid: str) -> None:
    for d in (run_dir(opts, sid), report_dir(opts) / sid):
        if d.exists():
            shutil.rmtree(d)


def baseline_status(opts, sid) -> dict:
    return read_json(run_dir(opts, sid) / "baseline" / "status.json", {}) or {}


def final_status(opts, sid) -> dict:
    return read_json(run_dir(opts, sid) / "final" / "summary.json", {}) or {}


# ----------------------------------------------------------------- engine run
def make_config(opts: RunOptions, **over):
    from atlasbt.engine import BacktestConfig

    kw = {"initial_cash": opts.initial_cash, "fees": opts.fees, "use_quotes": opts.use_quotes}
    if opts.spread_model:
        kw["spread_model"] = opts.spread_model
    kw.update(over)
    return BacktestConfig(**kw)


def _spread_model(opts) -> str:
    try:
        cfg = make_config(opts)
        return cfg.resolved_spread_model() if hasattr(cfg, "resolved_spread_model") else "corwin_schultz"
    except Exception as e:  # pragma: no cover
        return f"unknown ({e})"


def strategy_params(sid: str, combo: Optional[dict] = None) -> dict:
    from atlasbt.strategies import registry as R

    cls = R.get(sid)
    p = dict(cls.params)
    if combo:
        p.update(combo)
    return p


def window(plan_row: dict, opts: RunOptions) -> tuple[str, str]:
    return (opts.start_override or plan_row["start"], opts.end_override or plan_row["end"])


def _prune_diagnostics(diag: dict) -> dict:
    out = {}
    for k, v in (diag or {}).items():
        if k in ("orders", "logs", "intraday_equity"):
            try:
                out[f"n_{k}"] = len(v)
            except Exception:
                pass
            continue
        if isinstance(v, list) and len(v) > 300:
            out[k] = v[:300]
            out[f"n_{k}"] = len(v)
        else:
            out[k] = v
    return out


def run_engine(sid: str, params: dict, start: str, end: str, opts: RunOptions, portal=None, **cfg_over):
    from atlasbt.engine import Backtester
    from atlasbt.strategies import registry as R

    from .fastportal import CoverageFilteredPortal, FastPortal

    portal = portal or (CoverageFilteredPortal() if opts.universe_coverage == "filter" else FastPortal())
    cls = R.get(sid)
    strat = cls(**params)
    bt = Backtester(strat, portal, make_config(opts, **cfg_over))
    return bt.run(start, end), strat, portal


def _metrics_quick(returns: pd.Series) -> dict:
    r = returns.astype(float).dropna().to_numpy()
    if len(r) < 2:
        return {"n": int(len(r))}
    sd = r.std(ddof=1)
    return {"n": int(len(r)), "mean_daily": float(r.mean()), "sharpe_raw_ann": float(r.mean() / sd * math.sqrt(252))
            if sd > 0 else float("nan"), "sr_period_raw": float(r.mean() / sd) if sd > 0 else float("nan"),
            "total_return": float(np.prod(1 + r) - 1)}


def _excess(returns: pd.Series, result) -> pd.Series:
    """Daily returns minus the engine's own rf accrual (meta['rf_daily'])."""
    rf = None
    try:
        from atlasbt.reporting.metrics import risk_free_returns

        rf, _ = risk_free_returns(result)
    except Exception:
        rf = None
    if rf is None or not len(rf):
        return returns
    return returns - rf.reindex(returns.index).fillna(0.0)


# ----------------------------------------------------------------- tasks
def task_baseline(sid: str, plan_row: dict, opts: RunOptions) -> dict:
    """Baseline (sheet params, full window). Writes baseline/* and config.json."""
    d = run_dir(opts, sid)
    (d / "baseline").mkdir(parents=True, exist_ok=True)
    seed = seed_everything(sid, "baseline")
    start, end = window(plan_row, opts)
    params = strategy_params(sid)
    t = Timer()
    res, strat, portal = run_engine(sid, params, start, end, opts)
    status = getattr(res.status, "value", str(res.status))
    act = _activity(sid, res, plan_row, str(Path(opts.results_dir) / "logs" / f"{sid}.log"))   # before pruning
    res.diagnostics = _prune_diagnostics(res.diagnostics)
    syms = set(getattr(strat, "universe", []) or [])
    for r in getattr(strat, "requires", []) or []:
        if r.symbols:
            syms |= {s for s in r.symbols if s != "*"}
    if getattr(strat, "universe_name", None):
        try:
            syms |= set(portal.universe(strat.universe_name, pd.Timestamp(start, tz="UTC") + pd.Timedelta(hours=14)))
        except Exception:
            pass
    freqs = tuple(dict.fromkeys(["1d"] + ([strat.bar_freq] if is_intraday(strat.bar_freq) else [])))
    ucov = universe_coverage_record(portal, strat, start, end)
    if hasattr(portal, "coverage_series"):
        cs = portal.coverage_series()
        if len(cs):
            cs.to_csv(d / "baseline" / "universe_coverage.csv", index=False)
    write_json(d / "config.json", {
        "strategy_id": sid, "params": params, "grid": strat.grid, "window": [start, end], "seed": seed,
        "plan": plan_row, "options": opts.to_dict(), "engine_config": res.meta.get("config"),
        "spread_model": _spread_model(opts),
        "deviations": list(getattr(strat, "deviations", []) or []), "provenance": environment_info(),
        "data_provenance": data_provenance(portal, syms, freqs), "survivorship": res.meta.get("survivorship_bias"),
        "universe_coverage": ucov})
    write_pickle(d / "baseline" / "result.pkl.gz", res)
    if len(res.equity):
        pd.DataFrame({"nav": res.equity, "ret": res.returns}).to_csv(d / "baseline" / "equity.csv")
    st = {"status": status, "runtime_sec": t.sec, "reason": res.not_testable_reason,
          "activity": act.get("activity"), "activity_detail": act, "universe_coverage": ucov,
          "spread_model": _spread_model(opts), "code_sha_at_launch": opts.extra.get("code_sha_at_launch"),
          "traceback": res.diagnostics.get("traceback", ""), "n_days": int(len(res.equity)),
          "n_fills": int(len(res.fills)), "params": params, "window": [start, end],
          "lookahead_guard_violations": res.diagnostics.get("lookahead_guard_violations"),
          **_metrics_quick(res.returns)}
    write_json(d / "baseline" / "status.json", st)
    return st


def universe_coverage_record(portal, strat, start: str, end: str) -> dict:
    """Survivorship coverage of the strategy's PIT universe: DATA's universe_survivorship_bias at the
    window start / middle / end, plus the runner filter log (members dropped for lack of data)."""
    un = getattr(strat, "universe_name", None)
    if not un:
        return {"universe": "static list", "note": "static universe: survivorship bias possible (engine meta)"}
    out: dict = {"universe": un}
    base = "SP500" if un.upper().startswith("SP500") else un
    pts = {}
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    for lab, d in (("start", a), ("mid", a + (b - a) / 2), ("end", b)):
        try:
            r = portal.universe_survivorship_bias(base, pd.Timestamp(d.date(), tz="UTC") + pd.Timedelta(hours=21))
            pts[lab] = {"date": str(d.date()), "members": r.get("members"), "with_data": r.get("with_data"),
                        "coverage": r.get("coverage"), "missing": list(r.get("missing", []))[:12]}
        except Exception as e:
            pts[lab] = {"error": f"{type(e).__name__}: {e}"}
    out["survivorship_bias"] = pts
    if hasattr(portal, "coverage_summary"):
        out["runner_filter"] = portal.coverage_summary()
    return out


def grid_plan(sid: str, plan_row: dict, opts: RunOptions, baseline_runtime: float) -> dict:
    """Configs to run for the grid (deterministic subset when capped)."""
    from atlasbt.strategies import registry as R

    cls = R.get(sid)
    combos = R.expand_grid(cls.grid)
    base = dict(cls.params)
    is_base = [all(base.get(k) == v for k, v in c.items()) for c in combos]
    n_full = len(combos)
    cap = n_full
    reasons = []
    if opts.max_configs is not None and opts.max_configs < cap:
        cap = int(opts.max_configs)
        reasons.append(f"--max-configs {opts.max_configs}")
    budget = opts.budget_intraday_min if is_intraday(plan_row.get("bar_freq", "1d")) else opts.budget_daily_min
    if budget is not None:
        per = max(float(baseline_runtime), 1.0)
        afford = int(budget * 60 // per)
        if afford < cap:
            cap = max(afford, 1)
            reasons.append(f"budget {budget:g} min at {per:.0f} s/config")
    order = list(range(n_full))
    if cap < n_full:
        rng = np.random.default_rng(seed_everything(sid, "grid_subset"))
        base_idx = [i for i in order if is_base[i]]
        rest = [i for i in order if not is_base[i]]
        pick = list(rng.permutation(rest)[:max(cap - len(base_idx), 0)])
        order = sorted(base_idx[:cap] + [int(i) for i in pick])
    selected = [{"cfg": config_id(combos[i]), "combo": combos[i], "params": {**base, **combos[i]},
                 "is_baseline": bool(is_base[i])} for i in order]
    gp = {"n_full": n_full, "n_selected": len(selected), "subset": cap < n_full, "reasons": reasons,
          "selection": "seeded permutation (sha256(sid|'grid_subset')); baseline config always kept",
          "configs": selected}
    write_json(run_dir(opts, sid) / "grid_plan.json", gp)
    return gp


def grid_config_done(opts, sid, cfg) -> bool:
    st = read_json(run_dir(opts, sid) / "grid" / f"{cfg}.json", {})
    return bool(st) and st.get("status") in TERMINAL


def task_config(sid: str, cfg: dict, plan_row: dict, opts: RunOptions) -> dict:
    d = run_dir(opts, sid) / "grid"
    d.mkdir(parents=True, exist_ok=True)
    seed_everything(sid, cfg["cfg"])
    start, end = window(plan_row, opts)
    t = Timer()
    res, _, _ = run_engine(sid, cfg["params"], start, end, opts)
    status = getattr(res.status, "value", str(res.status))
    if len(res.equity):
        ex = _excess(res.returns, res)
        pd.DataFrame({"nav": res.equity, "ret": res.returns, "excess": ex}).to_parquet(d / f"{cfg['cfg']}.parquet")
    st = {"status": status, "cfg": cfg["cfg"], "combo": cfg["combo"], "key": param_key(cfg["combo"]),
          "runtime_sec": t.sec, "reason": res.not_testable_reason,
          "activity": _activity(sid, res, plan_row).get("activity"),
          "traceback": (res.diagnostics or {}).get("traceback", ""), "n_fills": int(len(res.fills)),
          "fidelity_grade": (res.fidelity or {}).get("grade"), **_metrics_quick(res.returns)}
    write_json(d / f"{cfg['cfg']}.json", st)
    return st


def write_baseline_as_config(sid: str, cfg: dict, opts: RunOptions) -> None:
    """The grid config equal to the sheet params reuses the baseline run."""
    d = run_dir(opts, sid)
    res = read_pickle(d / "baseline" / "result.pkl.gz")
    bst = baseline_status(opts, sid)
    (d / "grid").mkdir(parents=True, exist_ok=True)
    status = getattr(res.status, "value", str(res.status))
    if len(res.equity):
        pd.DataFrame({"nav": res.equity, "ret": res.returns, "excess": _excess(res.returns, res)}).to_parquet(
            d / "grid" / f"{cfg['cfg']}.parquet")
    write_json(d / "grid" / f"{cfg['cfg']}.json", {
        "status": status, "cfg": cfg["cfg"], "combo": cfg["combo"], "key": param_key(cfg["combo"]),
        "runtime_sec": bst.get("runtime_sec"), "reason": res.not_testable_reason, "n_fills": int(len(res.fills)),
        "activity": bst.get("activity"),
        "fidelity_grade": (res.fidelity or {}).get("grade"), "reused_baseline": True, **_metrics_quick(res.returns)})


# ----------------------------------------------------------------- finalize
def atlas_trial_total() -> int:
    from atlasbt.strategies import registry as R
    from atlasbt.validation.sharpe_tests import total_atlas_trials

    grids = {s: len(R.expand_grid(R.get(s).grid)) for s in R.all_ids()}
    return int(max(total_atlas_trials(grids), total_atlas_trials()))


def to_offset(x):
    """'3Y' / '1Y' / '6M' / '2M' -> pd.DateOffset (pandas 3 rejects the 'Y' alias in to_offset)."""
    if not isinstance(x, str):
        return x
    n, unit = int(x[:-1]), x[-1].upper()
    return {"Y": pd.DateOffset(years=n), "M": pd.DateOffset(months=n), "W": pd.DateOffset(weeks=n),
            "D": pd.DateOffset(days=n)}[unit]


def _pbo_blocks(T: int) -> Optional[int]:
    for S in (16, 14, 12, 10, 8, 6, 4):
        if T // S >= 5:
            return S
    return None


def load_grid_matrix(opts: RunOptions, sid: str, index: pd.DatetimeIndex, col: str = "ret"):
    gp = read_json(run_dir(opts, sid) / "grid_plan.json", {}) or {}
    cols, info = {}, []
    for c in gp.get("configs", []):
        st = read_json(run_dir(opts, sid) / "grid" / f"{c['cfg']}.json", {}) or {}
        info.append({"cfg": c["cfg"], "key": param_key(c["combo"]), "status": st.get("status", "MISSING"),
                     "runtime_sec": st.get("runtime_sec"), "sharpe_raw_ann": st.get("sharpe_raw_ann"),
                     "total_return": st.get("total_return"), "n_fills": st.get("n_fills"),
                     "reason": (st.get("reason") or "")[:200], "is_baseline": c.get("is_baseline", False),
                     "activity": st.get("activity")})
        p = run_dir(opts, sid) / "grid" / f"{c['cfg']}.parquet"
        # ZERO_ACTIVITY cells are degenerate (never trade): excluded from WF / PBO, listed in the report
        if st.get("status") == "OK" and st.get("activity") != "ZERO_ACTIVITY" and p.exists():
            s = pd.read_parquet(p)[col]
            cols[param_key(c["combo"]) or "baseline"] = s
    M = pd.DataFrame(cols)
    if len(M):
        M = M.reindex(index.union(M.index)).fillna(0.0) if len(index) else M.fillna(0.0)
    return M, pd.DataFrame(info), gp


def task_finalize(sid: str, plan_row: dict, opts: RunOptions) -> dict:
    """Walk-forward, untouched holdout, PBO, cost stress, bootstrap CIs and the report."""
    from atlasbt.reporting import summarize_result, write_strategy_report
    from atlasbt.validation import bootstrap_ci, pbo_cscv, walk_forward_matrix, walk_forward_splits

    from .fastportal import FastPortal

    d = run_dir(opts, sid)
    (d / "final").mkdir(parents=True, exist_ok=True)
    seed_everything(sid, "finalize")
    t = Timer()
    bst = baseline_status(opts, sid)
    summary: dict = {"strategy_id": sid, "status": bst.get("status", "MISSING"), "plan": plan_row,
                     "universe_coverage": bst.get("universe_coverage"), "window": bst.get("window"),
                     "spread_model": bst.get("spread_model"),
                     "run_label": opts.extra.get("label") or ("window-override" if opts.start_override else "full"),
                     "activity": bst.get("activity"), "activity_detail": bst.get("activity_detail"),
                     "baseline": {k: v for k, v in bst.items() if k != "traceback"},
                     "baseline_traceback": bst.get("traceback", "")}
    if bst.get("status") != "OK":
        summary["reason"] = bst.get("reason", "")
        res = read_pickle(d / "baseline" / "result.pkl.gz") if (d / "baseline" / "result.pkl.gz").exists() else None
        if res is not None:
            try:
                write_strategy_report(res, report_dir(opts), make_charts=False)
                summary["leaderboard_row"] = summarize_result(res)
            except Exception as e:
                summary["report_error"] = f"{type(e).__name__}: {e}"
        write_json(d / "final" / "summary.json", summary)
        return {"status": summary["status"]}

    res = read_pickle(d / "baseline" / "result.pkl.gz")
    idx = pd.DatetimeIndex(res.returns.index)
    freq = plan_row.get("bar_freq", "1d")
    N_atlas = atlas_trial_total()
    M, info, gp = load_grid_matrix(opts, sid, idx, "ret")
    Mx, _, _ = load_grid_matrix(opts, sid, idx, "excess")
    summary["grid"] = {"n_full": gp.get("n_full"), "n_selected": gp.get("n_selected"), "subset": gp.get("subset"),
                       "reasons": gp.get("reasons"), "n_ok": int(M.shape[1]) if len(M) else 0,
                       "configs": info.to_dict("records") if len(info) else []}
    trial_sr = []
    if len(Mx):
        for c in Mx.columns:
            x = Mx[c].to_numpy(float)
            sd = x.std(ddof=1)
            trial_sr.append(float(x.mean() / sd) if sd > 0 else float("nan"))
    summary["grid_sr_period"] = trial_sr
    var_sr = float(np.nanvar(trial_sr, ddof=1)) if np.isfinite(trial_sr).sum() >= 2 else None

    # ---- walk-forward with untouched holdout
    tr, te = opts.wf_intraday if is_intraday(freq) else opts.wf_daily
    wf_out: dict = {"train": tr, "test": te, "holdout_frac": opts.holdout, "mode": "matrix (full-window runs)"}
    wf = None
    if len(M) and M.shape[1] >= 1:
        try:
            splits = walk_forward_splits(M.index, to_offset(tr), to_offset(te), step=to_offset(te), anchored=False,
                                         holdout=opts.holdout)
            wf = walk_forward_matrix(M, splits, objective="sharpe", evaluate_holdout=True)
            s = wf.summary()
            wf_out.update({k: v for k, v in s.items()})
            wf_out["n_folds"] = len(splits.folds)
            wf_out["holdout_start"] = str(splits.holdout[0]) if len(splits.holdout) else None
            wf_out["holdout_end"] = str(splits.holdout[-1]) if len(splits.holdout) else None
            wf_out["holdout_n"] = int(len(splits.holdout))
            wf_out["folds"] = wf.folds.to_dict("records") if len(wf.folds) else []
            if not len(splits.folds):
                wf_out["note"] = (f"window ({len(M)} sessions) shorter than train {tr} + test {te}: no WF folds; "
                                  f"holdout params chosen on all pre-holdout data")
            hb = res.returns.reindex(splits.holdout).dropna()
            if len(hb) > 2 and hb.std() > 0:
                wf_out["holdout_baseline_sharpe"] = float(hb.mean() / hb.std(ddof=1) * math.sqrt(252))
                wf_out["holdout_baseline_total_return"] = float(np.prod(1 + hb) - 1)
            oos = wf.oos_returns.dropna()
            if len(oos) >= 30 and oos.std() > 0:
                wf_out["oos_bootstrap_sharpe"] = bootstrap_ci(oos.to_numpy(), "sharpe", reps=opts.bootstrap_reps,
                                                              seed=0).as_dict()
            pd.DataFrame({"oos": wf.oos_returns}).to_csv(d / "final" / "wf_oos_returns.csv")
            if wf.holdout_returns is not None:
                pd.DataFrame({"holdout": wf.holdout_returns}).to_csv(d / "final" / "holdout_returns.csv")
            pre = M.loc[M.index.isin(splits.pre_holdout)]
        except Exception as e:
            wf_out["error"] = f"{type(e).__name__}: {e}"
            wf_out["traceback"] = traceback.format_exc()[-2000:]
            pre = M.iloc[: int(len(M) * (1 - opts.holdout))]
    else:
        pre = M
    summary["walk_forward"] = wf_out

    # ---- PBO (CSCV) on the pre-holdout grid matrix
    pbo_out: dict = {}
    if len(pre) and pre.shape[1] >= 2:
        S = _pbo_blocks(len(pre))
        if S:
            try:
                pr = pbo_cscv(pre, S=S, metric="sharpe", seed=0)
                pbo_out = pr.as_dict()
                pbo_out["T"] = int(len(pre))
                pbo_out["N"] = int(pre.shape[1])
            except Exception as e:
                pbo_out = {"error": f"{type(e).__name__}: {e}"}
        else:
            pbo_out = {"note": f"too few observations ({len(pre)}) for CSCV"}
    else:
        pbo_out = {"note": "fewer than 2 grid configs ran OK; PBO undefined"}
    summary["pbo"] = pbo_out

    # ---- report (cost stress A-E, break-even, bootstrap CIs, fidelity scorecard)
    portal = FastPortal()
    bench = factors = regimes = None
    try:
        bench = benchmark_returns(portal, idx, "SPY")
    except Exception as e:
        summary["benchmark_error"] = str(e)
    try:
        factors = ff_factors_daily(portal, idx)
    except Exception:
        factors = None
    try:
        regimes = regimes_for(res.returns, regime_inputs(portal))
    except Exception:
        regimes = None
    extra = {"walk_forward": wf_out, "pbo": pbo_out, "grid": {k: v for k, v in summary["grid"].items()
                                                              if k != "configs"},
             "n_trials_atlas": N_atlas, "var_sr_grid_period": var_sr}
    payload = write_strategy_report(res, report_dir(opts), benchmark=bench, factors=factors, n_trials=N_atlas,
                                    var_sr=var_sr, regimes=regimes, extra=extra, make_charts=opts.make_charts,
                                    bootstrap_reps=opts.bootstrap_reps)
    val = payload.get("validation", {})
    be = (val.get("break_even") or {}).get("cagr", {})
    summary["report"] = {"dsr": val.get("dsr"), "psr_0": val.get("psr_0"), "n_trials": val.get("n_trials"),
                         "break_even": val.get("break_even"), "bootstrap": val.get("bootstrap"),
                         "cost_stress": val.get("cost_stress"), "fidelity_grade": payload.get("fidelity_grade"),
                         "performance": (payload.get("metrics") or {}).get("performance")}
    cov_png = _coverage_chart(d / "baseline" / "universe_coverage.csv", report_dir(opts) / sid / "universe_coverage.png",
                              summary.get("run_label"))
    summary["coverage_chart"] = cov_png
    _append_report_section(report_dir(opts) / sid / "report.md", summary,
                           {**plan_row, "start": window(plan_row, opts)[0], "end": window(plan_row, opts)[1]})
    be_bp = be.get("bp") if isinstance(be, dict) else None
    row = summarize_result(res, pbo=pbo_out.get("pbo"), oos_sharpe=wf_out.get("oos_sharpe"),
                           break_even_bp=be_bp, n_trials=gp.get("n_full"))
    row["activity"] = bst.get("activity")
    row["run_label"] = summary.get("run_label")
    uc = bst.get("universe_coverage") or {}
    sb = uc.get("survivorship_bias") or {}
    row["universe_coverage_start"] = (sb.get("start") or {}).get("coverage")
    row["universe_coverage_min_filter"] = (uc.get("runner_filter") or {}).get("coverage_min")
    ad = bst.get("activity_detail") or {}
    row["n_entries"] = ad.get("n_entries")
    if bst.get("activity") == "ZERO_ACTIVITY":
        # never reported as "0 return, OK": separate status, own table in the atlas report
        dg = ad.get("diagnostics") or {}
        row["status"] = "ZERO_ACTIVITY"
        row["reason"] = ("no entries over the window" + (f"; declared: {ad.get('declared_note')}" if ad.get("declared_note") else "")
                         + (f"; rejects: {dg.get('top_reject_reasons')}" if dg.get("top_reject_reasons") else "")
                         + (f"; warnings: {dg.get('warnings')[:2]}" if dg.get("warnings") else "")
                         + (f"; fidelity: {dg.get('fidelity_reasons')[:2]}" if dg.get("fidelity_reasons") else ""))[:600]
        summary["status"] = "ZERO_ACTIVITY"
    row["holdout_sharpe"] = wf_out.get("holdout_sharpe")
    row["holdout_params"] = wf_out.get("holdout_params")
    summary["leaderboard_row"] = row
    summary["finalize_runtime_sec"] = t.sec
    write_json(d / "final" / "summary.json", summary)
    return {"status": "OK"}


def _coverage_chart(csv: Path, out: Path, label=None):
    """Plot the PIT universe price-coverage series recorded by CoverageFilteredPortal."""
    if not Path(csv).exists():
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cs = pd.read_csv(csv, parse_dates=["date"])
        fig, ax = plt.subplots(figsize=(8, 3))
        for u, g in cs.groupby("universe"):
            ax.plot(g["date"], g["coverage"] * 100, label=f"{u}: members with price data")
        ax.axhline(95, color="grey", lw=0.8, ls="--")
        ax.set_ylabel("% of PIT members")
        ax.set_ylim(0, 101)
        ttl = "Universe survivorship coverage"
        if label and str(label).upper().startswith("EXTENDED"):
            ttl += " - EXTENDED_SURVIVORSHIP_BIASED (results likely optimistic)"
        ax.set_title(ttl, fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=110)
        plt.close(fig)
        return Path(out).name
    except Exception:
        return None


def _f(v, k="f2"):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    if k == "pct":
        return f"{v:.2%}"
    return f"{v:.2f}" if isinstance(v, (int, float)) else str(v)


def _append_report_section(path: Path, s: dict, plan_row: dict) -> None:
    wf, pbo, g = s.get("walk_forward", {}), s.get("pbo", {}), s.get("grid", {})
    from .notices import banner_md

    lines = ["", "## Walk-forward, untouched holdout, PBO (atlas runner)", "", banner_md(),
             f"Window {plan_row.get('start')} -> {plan_row.get('end')} ({plan_row.get('binding_constraint', '')}). "
             f"Grid: {g.get('n_ok')} of {g.get('n_selected')} selected configs OK (full grid {g.get('n_full')})"
             + (f"; SUBSET: {', '.join(g.get('reasons') or [])}" if g.get("subset") else "") + ".", "",
             f"Walk-forward (rolling train {wf.get('train')} / test {wf.get('test')}, objective Sharpe, matrix mode "
             f"on full-window runs): folds {wf.get('n_folds', 0)}, OOS Sharpe {_f(wf.get('oos_sharpe'))}, OOS total "
             f"return {_f(wf.get('oos_total_return'), 'pct')}, WF efficiency {_f(wf.get('wf_efficiency'))}, "
             f"parameter stability {_f(wf.get('param_stability'))}.", ""]
    if str(s.get("run_label") or "").upper().startswith("EXTENDED"):
        lines += ["> **EXTENDED_SURVIVORSHIP_BIASED**: extended-history run from the first month with >= 60% PIT "
                  "price coverage. Delisted / failed names are under-represented, so results are likely OPTIMISTIC. "
                  "Never compare with or rank against primary (>= 95% coverage) results.", ""]
    if s.get("coverage_chart"):
        lines += [f"![universe coverage]({s['coverage_chart']})", ""]
    uc = s.get("universe_coverage") or {}
    if uc.get("survivorship_bias"):
        sb = uc["survivorship_bias"]
        lines += ["Universe survivorship coverage (" + str(uc.get("universe")) + "): " + "; ".join(
            f"{k} {v.get('date')}: {v.get('with_data')}/{v.get('members')} ({_f(v.get('coverage'), 'pct')})"
            for k, v in sb.items() if isinstance(v, dict) and "date" in v)
            + (f"; runner filter min coverage {_f((uc.get('runner_filter') or {}).get('coverage_min'), 'pct')}, "
               f"most dropped {((uc.get('runner_filter') or {}).get('most_dropped') or [])[:6]}" if uc.get("runner_filter") else ""), ""]
    ad = s.get("activity_detail") or {}
    if ad:
        lines += [f"Activity safeguard: **{ad.get('activity')}** ({ad.get('n_entries')} {ad.get('entry_unit')} over "
                  f"{ad.get('years')} years; floor {ad.get('floor')} = {ad.get('rule')})"
                  + (f"; declared note: {ad.get('declared_note')}" if ad.get("declared_note") else "") + ".", ""]
        dg = ad.get("diagnostics")
        if dg:
            lines += [f"Diagnostics: rejected orders {dg.get('n_rejected')} (top: {dg.get('top_reject_reasons')}); "
                      f"warnings {dg.get('warnings')[:3]}; fidelity reasons {dg.get('fidelity_reasons')[:3]}; "
                      f"log {dg.get('log')}", ""]
    if wf.get("note"):
        lines += [f"Note: {wf['note']}", ""]
    if wf.get("oos_bootstrap_sharpe"):
        b = wf["oos_bootstrap_sharpe"]
        lines += [f"OOS Sharpe 95% CI (stationary bootstrap): [{_f(b.get('lo'))}, {_f(b.get('hi'))}]", ""]
    lines += [f"Untouched holdout (last {int(100 * (wf.get('holdout_frac') or 0.2))}% of the window, "
              f"{wf.get('holdout_start')} -> {wf.get('holdout_end')}, evaluated once): params "
              f"`{wf.get('holdout_params')}`, Sharpe {_f(wf.get('holdout_sharpe'))}, total return "
              f"{_f(wf.get('holdout_total_return'), 'pct')}; baseline params on the same holdout: Sharpe "
              f"{_f(wf.get('holdout_baseline_sharpe'))}.", "",
              "PBO (CSCV on the pre-holdout grid matrix): " + (
                  f"{_f(pbo.get('pbo'))} (S={pbo.get('S')}, {pbo.get('n_combinations')} splits, N={pbo.get('N')}, "
                  f"T={pbo.get('T')}), degradation slope {_f(pbo.get('degradation_slope'))}, P(OOS loss) "
                  f"{_f(pbo.get('prob_oos_loss'))}" if "pbo" in pbo else (pbo.get("note") or pbo.get("error", "n/a"))), ""]
    if g.get("configs"):
        lines += ["| Config | Status | Activity | Sharpe (raw) | Total return | Fills | Runtime s |",
                  "|---|---|---|---:|---:|---:|---:|"]
        for c in g["configs"]:
            lines.append(f"| `{c.get('key') or 'baseline'}`{' (sheet)' if c.get('is_baseline') else ''} | "
                         f"{c.get('status')} | {c.get('activity')} | {_f(c.get('sharpe_raw_ann'))} | {_f(c.get('total_return'), 'pct')} | "
                         f"{c.get('n_fills')} | {_f(c.get('runtime_sec'))} |")
    with open(path, "a") as fh:
        fh.write("\n".join(lines) + "\n")
