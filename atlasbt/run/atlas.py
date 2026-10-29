"""`atlas`: atlas-wide statistics over every strategy's finalized run.

* Leaderboard (``reporting.build_leaderboard``): DSR deflated with the TOTAL atlas trial count
  (max of the summed registry grids and the parsed sheet count), V[SR] from the cross-section of
  every grid configuration's per-period Sharpe across the atlas (López de Prado's trial variance),
  floored by each strategy's own estimation noise; p = 1 - PSR(0); Benjamini-Hochberg q-values.
* Hansen SPA / White Reality Check (``validation.multiple_testing``) on daily differentials of the
  baseline (sheet-parameter) runs (a) vs cash (excess over the engine's rf) and (b) vs SPY total
  return, on the common window of every daily+ strategy with at least 5 years of overlap; repeated
  on the walk-forward OOS + holdout series (parameters chosen out of sample).
* NOT_TESTABLE / FAILED tables and the fidelity grade distribution.
Outputs: results/ATLAS_REPORT.md, results/leaderboard.csv (+ leaderboard.md, not_testable.csv).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import pipeline as P
from .common import DEFAULT_RESULTS, is_intraday, read_json, to_session_dates, write_json
from .plan import all_sheet_ids, load_plan

MIN_COMMON_YEARS = 5.0


def _baseline_series(opts, sid, col="excess") -> pd.Series | None:
    gp = read_json(P.run_dir(opts, sid) / "grid_plan.json", {}) or {}
    for c in gp.get("configs", []):
        if c.get("is_baseline"):
            f = P.run_dir(opts, sid) / "grid" / f"{c['cfg']}.parquet"
            if f.exists():
                return pd.read_parquet(f)[col]
    f = P.run_dir(opts, sid) / "baseline" / "result.pkl.gz"
    if f.exists():
        from .common import read_pickle

        res = read_pickle(f)
        return P._excess(res.returns, res) if col == "excess" else res.returns
    return None


def _oos_series(opts, sid) -> pd.Series | None:
    d = P.run_dir(opts, sid) / "final"
    parts = []
    for name, col in (("wf_oos_returns.csv", "oos"), ("holdout_returns.csv", "holdout")):
        f = d / name
        if f.exists():
            s = pd.read_csv(f, index_col=0, parse_dates=True)[col]
            parts.append(s)
    if not parts:
        return None
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated(keep="first")]


def _common_matrix(series: dict, min_years: float = MIN_COMMON_YEARS):
    """Daily matrix on the common window; drops strategies that would shrink it below min_years."""
    if not series:
        return pd.DataFrame(), []
    spans = {k: (v.index.min(), v.index.max()) for k, v in series.items() if v is not None and len(v)}
    if not spans:
        return pd.DataFrame(), []
    end = min(e for _, e in spans.values())
    keep = [k for k, (s, e) in spans.items() if (end - s).days / 365.25 >= min_years]
    dropped = sorted(set(spans) - set(keep))
    if not keep:
        return pd.DataFrame(), dropped
    start = max(spans[k][0] for k in keep)
    M = pd.DataFrame({k: series[k] for k in keep})
    M.index = to_session_dates(M.index)
    M = M[~M.index.duplicated()]
    M = M.loc[(M.index >= to_session_dates([start])[0]) & (M.index <= to_session_dates([end])[0])].fillna(0.0)
    return M, dropped


def _spa_block(M: pd.DataFrame, bench: pd.Series | None, label: str, reps: int = 2000) -> dict:
    from atlasbt.validation import hansen_spa, white_reality_check

    out = {"label": label, "n_strategies": int(M.shape[1]), "n_days": int(M.shape[0]),
           "start": str(M.index.min().date()) if len(M) else None, "end": str(M.index.max().date()) if len(M) else None}
    if M.shape[1] < 1 or M.shape[0] < 252:
        out["note"] = "insufficient common data"
        return out
    d = M if bench is None else M.sub(bench.reindex(M.index).fillna(0.0), axis=0)
    spa = hansen_spa(d.to_numpy(), reps=reps, seed=0)
    rc = white_reality_check(d.to_numpy(), reps=reps, seed=0)
    best = d.mean().idxmax()
    out.update({"spa_consistent_p": spa["consistent"], "spa_lower_p": spa["lower"], "spa_upper_p": spa["upper"],
                "rc_p": rc["pvalue"], "statistic": spa["statistic"], "block": spa["block"],
                "best_strategy": best, "best_mean_ann": float(d[best].mean() * 252)})
    return out


def run_atlas(opts: P.RunOptions, log=print, allow_stock: bool = False) -> dict:
    from atlasbt.reporting import build_leaderboard, render_leaderboard_md

    from .fastportal import FastPortal
    from .holds import held, is_stale
    from .plan import annotate_plan

    res_dir = Path(opts.results_dir)
    annotate_plan(res_dir, allow_stock)
    plan = load_plan(res_dir)
    prow = {r["id"]: r for r in plan.to_dict("records")} if len(plan) else {}
    ids = all_sheet_ids()
    rows, sr_all, extra = [], [], {}
    for sid in ids:
        fs = P.final_status(opts, sid)
        pr = prow.get(sid, {})
        if fs and fs.get("leaderboard_row"):
            row = dict(fs["leaderboard_row"])
            if fs.get("status") != "OK":
                row["status"] = fs.get("status")
                row["reason"] = row.get("reason") or fs.get("reason", "")
            sr_all += [x for x in fs.get("grid_sr_period", []) if x is not None and np.isfinite(x)]
        elif fs:
            row = {"strategy_id": sid, "status": fs.get("status", "FAILED"), "reason": fs.get("reason", ""),
                   "fidelity_grade": "NA"}
        elif pr and pr.get("testable") == "N":
            row = {"strategy_id": sid, "status": "NOT_TESTABLE", "reason": f"plan: {pr.get('reason')}",
                   "fidelity_grade": "NA"}
        else:
            row = {"strategy_id": sid, "status": "NOT_RUN", "reason": "no finalized run in results/runs",
                   "fidelity_grade": "NA"}
        stale, h = is_stale(res_dir, sid), held(sid, allow_stock)
        if stale or (h and fs):
            row = {"strategy_id": sid, "status": "STALE", "fidelity_grade": row.get("fidelity_grade", "NA"),
                   "reason": (stale or h) + f" | last status {row.get('status')}"}
        elif h:
            row = {"strategy_id": sid, "status": "HOLD", "reason": h, "fidelity_grade": "NA"}
        if row.get("status") == "FAILED":
            tb = (fs or {}).get("baseline_traceback") or ""
            last = [l for l in tb.strip().splitlines() if l.strip()][-1:] if tb else []
            row["reason"] = (str(row.get("reason") or "") + (f" | {last[0]}" if last and last[0] not in str(row.get('reason')) else "")
                             + f" | log: results/logs/{sid}.log")[:400]
        rows.append(row)
        win = ((fs or {}).get("baseline") or {}).get("window") or [pr.get("start", ""), pr.get("end", "")]
        extra[sid] = {"family": pr.get("family", ""), "bar_freq": pr.get("bar_freq", ""), "start": win[0],
                      "end": win[1], "run_label": (fs or {}).get("run_label", ""), "holdout_sharpe": row.get("holdout_sharpe"),
                      "holdout_params": row.get("holdout_params"),
                      "grid_run": (fs or {}).get("grid", {}).get("n_ok"), "grid_full": (fs or {}).get("grid", {}).get("n_full"),
                      "grid_subset": (fs or {}).get("grid", {}).get("subset")}
    N = P.atlas_trial_total()
    tested, other = build_leaderboard(rows, n_trials_total=N, trial_sharpes=sr_all if len(sr_all) >= 2 else None)
    if len(tested):
        tested["activity"] = [next((r.get("activity") for r in rows if r["strategy_id"] == s), None)
                              for s in tested["strategy_id"]]
        for c in ("family", "bar_freq", "start", "end", "run_label", "holdout_sharpe", "holdout_params", "grid_run", "grid_full",
                  "grid_subset"):
            tested[c] = [extra[s][c] for s in tested["strategy_id"]]
    tested.to_csv(res_dir / "leaderboard.csv", index=False)
    other.to_csv(res_dir / "not_testable.csv", index=False)
    from .notices import banner_md

    (res_dir / "leaderboard.md").write_text(banner_md() + "\n" + render_leaderboard_md(tested, other))

    # ---- SPA / RC
    portal = FastPortal()
    try:
        from .common import adjusted_close

        spy = adjusted_close(portal, "SPY").pct_change()
    except Exception:
        spy = None
    ok_ids = [r["strategy_id"] for r in rows if r.get("status") == "OK"]
    daily_ok = [s for s in ok_ids if not is_intraday(prow.get(s, {}).get("bar_freq", "1d"))]
    base_ex = {s: _baseline_series(opts, s, "excess") for s in daily_ok}
    base_raw = {s: _baseline_series(opts, s, "ret") for s in daily_ok}
    Mx, dropped = _common_matrix({k: v for k, v in base_ex.items() if v is not None})
    Mr, _ = _common_matrix({k: v for k, v in base_raw.items() if v is not None and k in Mx.columns})
    spa = []
    if len(Mx):
        spa.append(_spa_block(Mx, None, "baseline vs cash (excess over rf)"))
    if len(Mr) and spy is not None:
        spa.append(_spa_block(Mr, spy, "baseline vs SPY total return"))
    oos = {s: _oos_series(opts, s) for s in daily_ok}
    Mo, dropped_o = _common_matrix({k: v for k, v in oos.items() if v is not None})
    if len(Mo) and spy is not None:
        spa.append(_spa_block(Mo, spy, "walk-forward OOS + holdout (raw returns) vs SPY"))
    if len(Mo):
        spa.append(_spa_block(Mo, None, "walk-forward OOS + holdout (raw returns) vs zero"))

    # ---- fidelity distribution
    fdist = pd.crosstab(pd.Series([r.get("status") for r in rows], name="status"),
                        pd.Series([r.get("fidelity_grade") or "NA" for r in rows], name="grade"))
    out = {"n_strategies": len(ids), "n_trials_total": N, "n_trial_sharpes": len(sr_all),
           "status_counts": pd.Series([r.get("status") for r in rows]).value_counts().to_dict(),
           "spa": spa, "spa_dropped_short_overlap": dropped, "fdr_pass": int(tested["fdr_pass"].sum()) if len(tested) else 0}
    ext = extended_table(res_dir / "extended")
    if len(ext):
        ext.to_csv(res_dir / "extended_survivorship_biased.csv", index=False)
    out["n_extended"] = int(len(ext))
    write_json(res_dir / "atlas_stats.json", out)
    _write_report(res_dir, out, tested, other, fdist, plan, rows, ext)
    log(f"atlas: {out['status_counts']} N={N}; report -> {res_dir / 'ATLAS_REPORT.md'}")
    return out


def _fmt(v, k="f2"):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    if k == "pct":
        return f"{v:.1%}"
    if k == "p":
        return f"{v:.3f}"
    return f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)


def extended_table(ext_dir: Path) -> pd.DataFrame:
    """EXTENDED_SURVIVORSHIP_BIASED runs (results/extended): own table, never ranked with the primary
    leaderboard."""
    rows = []
    for f in sorted(Path(ext_dir).glob("runs/S*/final/summary.json")):
        s = read_json(f, {}) or {}
        r = s.get("leaderboard_row") or {}
        uc = s.get("universe_coverage") or {}
        sb = (uc.get("survivorship_bias") or {})
        rows.append({"strategy_id": s.get("strategy_id"), "status": r.get("status", s.get("status")),
                     "label": s.get("run_label"), "window": "..".join(map(str, s.get("window") or [])),
                     "coverage_start": (sb.get("start") or {}).get("coverage"),
                     "coverage_min_filter": (uc.get("runner_filter") or {}).get("coverage_min"),
                     "sharpe_raw": r.get("sharpe"), "cagr": r.get("cagr"), "max_drawdown": r.get("max_drawdown"),
                     "psr_0": (s.get("report") or {}).get("psr_0"), "dsr": (s.get("report") or {}).get("dsr"),
                     "activity": r.get("activity"), "fidelity_grade": r.get("fidelity_grade")})
    return pd.DataFrame(rows)


def _write_report(res_dir: Path, out, tested, other, fdist, plan, rows, ext=None):
    from .common import environment_info
    from .notices import banner_md

    env = environment_info()
    L = ["# Atlas report", "",
         f"Generated {env['timestamp_utc']} | git {env['git_commit'][:12]} | code sha256 {env['code_sha256_16']}", "",
         banner_md(), "## Summary", "",
         "| Status | Strategies |", "|---|---:|"]
    for k, v in sorted(out["status_counts"].items()):
        L.append(f"| {k} | {v} |")
    L += ["", f"Total configurations deflated against (DSR): **N = {out['n_trials_total']}** "
              f"(max of summed registry grids and the parsed sheet count). V[SR] across trials from "
              f"{out['n_trial_sharpes']} grid-configuration Sharpe ratios (floored by each strategy's own "
              f"estimation noise). Benjamini-Hochberg (q = 0.10) on p = 1 - PSR(0): "
              f"**{out['fdr_pass']}** strategies pass.", "",
          "## Data-snooping tests (Hansen SPA / White Reality Check)", "",
          "Differentials are daily returns of every strategy minus the benchmark on the common window of all "
          "daily+ strategies with >= 5 years of overlap (stationary bootstrap, Politis-White block, 2,000 reps). "
          "H0: no strategy beats the benchmark.", "",
          "| Test | Strategies | Days | Window | SPA consistent p | SPA lower p | RC p | Best (ann. mean diff) |",
          "|---|---:|---:|---|---:|---:|---:|---|"]
    for s in out["spa"]:
        if "note" in s:
            L.append(f"| {s['label']} | {s['n_strategies']} | {s['n_days']} | | {s['note']} | | | |")
        else:
            L.append(f"| {s['label']} | {s['n_strategies']} | {s['n_days']} | {s['start']} .. {s['end']} | "
                     f"{_fmt(s['spa_consistent_p'], 'p')} | {_fmt(s['spa_lower_p'], 'p')} | {_fmt(s['rc_p'], 'p')} | "
                     f"{s['best_strategy']} ({_fmt(s['best_mean_ann'], 'pct')}) |")
    if out.get("spa_dropped_short_overlap"):
        L += ["", f"Excluded from SPA (less than 5 years of common history): "
                  f"{', '.join(out['spa_dropped_short_overlap'])}"]
    L += ["", "## Leaderboard (sorted by DSR, then PBO, then q-value; raw Sharpe for reference only)", ""]
    if len(tested):
        L += ["| # | Strategy | Family | DSR | PSR(0) | q | FDR | PBO | WF OOS SR | Holdout SR | raw SR | CAGR | MaxDD | "
              "BE bp | Grade | Activity | Window |",
              "|---:|---|---|---:|---:|---:|:-:|---:|---:|---:|---:|---:|---:|---:|:-:|---|---|"]
        for r in tested.itertuples():
            L.append(f"| {r.rank} | {r.strategy_id} | {str(getattr(r, 'family', ''))[:28]} | {_fmt(r.dsr, 'p')} | "
                     f"{_fmt(r.psr_0, 'p')} | {_fmt(r.q_value, 'p')} | {'yes' if r.fdr_pass else 'no'} | "
                     f"{_fmt(getattr(r, 'pbo', None))} | {_fmt(getattr(r, 'oos_sharpe', None))} | "
                     f"{_fmt(getattr(r, 'holdout_sharpe', None))} | {_fmt(r.sharpe)} | {_fmt(getattr(r, 'cagr', None), 'pct')} | "
                     f"{_fmt(getattr(r, 'max_drawdown', None), 'pct')} | {_fmt(getattr(r, 'break_even_bp', None))} | "
                     f"{r.fidelity_grade} | {getattr(r, 'activity', '')} | {getattr(r, 'start', '')}..{getattr(r, 'end', '')}"
                     f"{' (' + str(getattr(r, 'run_label', '')) + ')' if getattr(r, 'run_label', '') not in ('', 'full', None) else ''} |")
    else:
        L.append("No strategy finished with status OK.")
    L += ["", "## Fidelity grade distribution", "", fdist.to_markdown() if _tab() else fdist.to_string(), "",
          "Grades (reporting.fidelity): A quote-level, B real bars + modelled spread, C proxies, D mostly modelled, "
          "E/NA not testable.", ""]
    for status, title in (("ZERO_ACTIVITY", "ZERO_ACTIVITY (status OK but no entries; excluded from the leaderboard)"),
                          ("FAILED", "FAILED (owner action needed)"), ("TIMEOUT", "TIMEOUT"),
                          ("STALE", "STALE (results invalidated by a data hold; re-run after the fix)"),
                          ("HOLD", "HOLD (not run: integrator data hold)"),
                          ("NOT_TESTABLE", "NOT_TESTABLE"), ("NOT_RUN", "Not run")):
        sub = [r for r in rows if r.get("status") == status]
        if not sub:
            continue
        L += [f"## {title} ({len(sub)})", "", "| Strategy | Reason |", "|---|---|"]
        for r in sub:
            L.append(f"| {r['strategy_id']} | {str(r.get('reason', '')).replace('|', '/').replace(chr(10), ' ')[:300]} |")
        L.append("")
    if ext is not None and len(ext):
        L += ["## EXTENDED_SURVIVORSHIP_BIASED runs (NOT part of the leaderboard)", "",
              "Secondary extended-history runs of stock strategies from the first month with >= 60% PIT S&P 500 "
              "price coverage (results/extended). Delisted and failed names are under-represented, so these results "
              "are likely OPTIMISTIC. Coverage series: reports/<SID>/universe_coverage.png in results/extended.", "",
              "| Strategy | Status | Window | Coverage at start | Min coverage | raw SR | CAGR | MaxDD | PSR(0) | DSR | Activity |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in ext.itertuples():
            L.append(f"| {r.strategy_id} | {r.status} | {r.window} | {_fmt(r.coverage_start, 'pct')} | "
                     f"{_fmt(r.coverage_min_filter, 'pct')} | {_fmt(r.sharpe_raw)} | {_fmt(r.cagr, 'pct')} | "
                     f"{_fmt(r.max_drawdown, 'pct')} | {_fmt(r.psr_0, 'p')} | {_fmt(r.dsr, 'p')} | {r.activity} |")
        L.append("")
    L += ["## Method notes", "",
          "See docs/RUNNING.md. Baseline = sheet parameters over the full planned window; grid = registry.expand_grid "
          "(subsets are deterministic and recorded); walk-forward = rolling train 3y / test 1y (daily+) or 6m / 2m "
          "(intraday) on the grid return matrix with an untouched final 20% holdout evaluated once; PBO via CSCV on "
          "the pre-holdout grid matrix; cost stress A-E and break-even via validation.cost_stress; bootstrap CIs "
          "via validation.bootstrap (stationary).", ""]
    (res_dir / "ATLAS_REPORT.md").write_text("\n".join(L) + "\n")


def _tab() -> bool:
    try:
        import tabulate  # noqa: F401

        return True
    except Exception:
        return False
