"""Atlas runner CLI.

    python -m atlasbt.run plan      [--ids S050,S103] [--results-dir results]
    python -m atlasbt.run run       [--ids ...] [--workers 4] [--max-configs N] [--budget MIN]
                                    [--daily-budget MIN] [--force] [--timeout MIN] [--start D --end D]
    python -m atlasbt.run atlas     [--results-dir results]
    python -m atlasbt.run audit     [--ids ...] [--live]
    python -m atlasbt.run replicate [--only spy,s105,put,umd,random,lookahead] [--quick]

See docs/RUNNING.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .common import DEFAULT_RESULTS


def _ids(arg) -> list[str] | None:
    if not arg:
        return None
    out = []
    for tok in arg.replace(" ", ",").split(","):
        tok = tok.strip().upper()
        if not tok:
            continue
        if "-" in tok and tok.count("S") == 2:          # range S046-S084
            a, b = tok.split("-")
            out += [f"S{i:03d}" for i in range(int(a[1:]), int(b[1:]) + 1)]
        else:
            out.append(tok)
    return out


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", buffering=1)

    def __call__(self, msg: str):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")


def cmd_plan(a) -> int:
    from .plan import make_plan, write_plan

    log = Logger(Path(a.results_dir) / "logs" / "plan.log")
    df = make_plan(_ids(a.ids), Path(a.results_dir), log=log,
                   min_coverage=1.0 if a.universe_coverage == "strict" else a.min_universe_coverage)
    out = write_plan(df, Path(a.results_dir))
    from .plan import annotate_plan

    annotate_plan(Path(a.results_dir), getattr(a, "allow_stock_universes", False))
    n_y = int((df["testable"] == "Y").sum())
    log(f"plan: {n_y}/{len(df)} testable -> {out}")
    return 0


def _opts(a):
    from .pipeline import RunOptions

    return RunOptions(results_dir=str(a.results_dir), max_configs=a.max_configs,
                      budget_intraday_min=a.budget, budget_daily_min=a.daily_budget, force=a.force,
                      workers=a.workers, task_timeout_min=a.timeout, mem_reserve_gb=a.mem_reserve,
                      bootstrap_reps=a.bootstrap_reps, make_charts=not a.no_charts,
                      start_override=a.start, end_override=a.end, fees=a.fees, use_quotes=a.use_quotes,
                      spread_model=a.spread_model,
                      extra={"extend_grid": getattr(a, "extend_grid", False), "label": getattr(a, "label", None)},
                      universe_coverage=a.universe_coverage, min_universe_coverage=a.min_universe_coverage)


def cmd_run(a) -> int:
    from .plan import load_plan, make_plan, write_plan
    from .scheduler import Scheduler
    from .worktree import running_pinned

    log = Logger(Path(a.results_dir) / "logs" / "run.log")
    if not running_pinned() and not a.allow_live_tree:
        log("refusing to run from the live working tree (other agents edit it concurrently): use "
            "`python -m atlasbt.run launch run ...` (pinned clean worktree) or pass --allow-live-tree for smoke tests")
        return 2
    ids = _ids(a.ids)
    plan = load_plan(Path(a.results_dir))
    have = set(plan["id"]) if len(plan) else set()
    missing = [i for i in (ids or []) if i not in have]
    if (ids is None and not len(plan)) or missing:
        log(f"planning {missing or 'all'} first")
        write_plan(make_plan(missing or None, Path(a.results_dir), log=log,
                             min_coverage=1.0 if a.universe_coverage == "strict" else a.min_universe_coverage),
                   Path(a.results_dir))
        plan = load_plan(Path(a.results_dir))
    rows = {r["id"]: r for r in plan.to_dict("records")}
    if ids is None:
        ids = [i for i, r in rows.items() if r.get("testable") == "Y"]
    from .holds import held, mark_stale
    from .plan import annotate_plan

    run_ids = []
    for i in ids:
        r = rows.get(i)
        h = held(i, a.allow_stock_universes)
        if h:
            if mark_stale(Path(a.results_dir), i, h):
                log(f"{i}: existing results marked STALE")
            log(f"{i}: {h}; skipped (--allow-stock-universes once the integrator lifts the hold)")
            continue
        if r is None:
            log(f"{i}: not in plan, skipped")
        elif r.get("testable") != "Y" and not a.include_untestable:
            log(f"{i}: NOT testable per plan ({r.get('reason')}); skipped (use --include-untestable to run anyway)")
        else:
            if r.get("testable") != "Y":
                if not (a.start and a.end):
                    log(f"{i}: untestable and no --start/--end given; skipped")
                    continue
            run_ids.append(i)
    opts = _opts(a)
    log(f"run: {len(run_ids)} strategies, workers={opts.workers}, max_configs={opts.max_configs}, "
        f"budget intraday={opts.budget_intraday_min} daily={opts.budget_daily_min} min, force={opts.force}, "
        f"window override={opts.start_override}..{opts.end_override}")
    Scheduler(rows, opts, log=log).run(run_ids)
    annotate_plan(Path(a.results_dir), a.allow_stock_universes)
    log("run: finished")
    return 0


def cmd_atlas(a) -> int:
    from .atlas import run_atlas

    log = Logger(Path(a.results_dir) / "logs" / "atlas.log")
    run_atlas(_opts(a), log=log, allow_stock=a.allow_stock_universes)
    return 0


def cmd_audit(a) -> int:
    from .audit import run_audit

    log = Logger(Path(a.results_dir) / "logs" / "audit.log")
    bad = run_audit(_opts(a), ids=_ids(a.ids), live=a.live, allow_stock=a.allow_stock_universes, log=log)
    return 1 if bad else 0


def cmd_launch(a) -> int:
    from .worktree import launch

    rest = list(a.rest)
    if rest and rest[0] == "--":
        rest = rest[1:]
    info = launch(rest, commit=a.commit)
    print(f"launched pid {info['pid']} at commit {info['commit'][:12]} in {info['worktree']} "
          f"(overlay {info['overlay']}); log {info['log']}", flush=True)
    return 0


def cmd_replicate(a) -> int:
    from .replicate import run_replications

    log = Logger(Path(a.results_dir) / "logs" / "replicate.log")
    run_replications(Path(a.results_dir), only=[x.strip().lower() for x in a.only.split(",")] if a.only else None,
                     quick=a.quick, log=log)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m atlasbt.run", description="atlasbt atlas runner")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="resolve requirements/windows -> results/plan.csv")
    p.add_argument("--ids", help="comma list or ranges (S046-S084); default all 172")
    p.add_argument("--allow-stock-universes", action="store_true")
    p.add_argument("--universe-coverage", default="filter", choices=["filter", "strict"])
    p.add_argument("--min-universe-coverage", type=float, default=0.95)
    for name, h in (("run", "baseline + grid + WF/holdout + PBO + cost stress + bootstrap + reports"),
                    ("atlas", "atlas-wide DSR/SPA/RC/FDR leaderboard -> ATLAS_REPORT.md"),
                    ("audit", "activity audit of baseline runs (ZERO/LOW_ACTIVITY) -> AUDIT.md")):
        q = sub.add_parser(name, help=h)
        q.add_argument("--ids")
        q.add_argument("--workers", type=int, default=4)
        q.add_argument("--max-configs", type=int, default=None, help="cap on grid configs per strategy")
        q.add_argument("--budget", type=float, default=30.0,
                       help="grid wall-clock budget (minutes) per INTRADAY strategy; deterministic subset if exceeded")
        q.add_argument("--daily-budget", type=float, default=None,
                       help="same budget for daily+ strategies (default unlimited)")
        q.add_argument("--force", action="store_true", help="rerun strategies that already have results")
        q.add_argument("--timeout", type=float, default=360.0, help="per-task wall-clock limit (minutes)")
        q.add_argument("--mem-reserve", type=float, default=3.0, help="GB of MemAvailable to keep free")
        q.add_argument("--bootstrap-reps", type=int, default=1000)
        q.add_argument("--no-charts", action="store_true")
        q.add_argument("--start", help="override the planned window start (smoke tests)")
        q.add_argument("--end", help="override the planned window end (smoke tests)")
        q.add_argument("--fees", default="IBKR_PRO_TIERED")
        q.add_argument("--use-quotes", default="never", choices=["never", "auto", "always"],
                       help="engine quote mode (default never: 'auto' breaks SPY/QQQ/AAPL, see RUNNING.md)")
        q.add_argument("--include-untestable", action="store_true")
        q.add_argument("--allow-live-tree", action="store_true",
                       help="permit `run` from the live working tree (smoke tests only; long runs use `launch`)")
        q.add_argument("--spread-model", default=None, choices=["calibrated", "corwin_schultz"],
                       help="engine spread model (default: engine default, 'calibrated')")
        q.add_argument("--label", help="run label recorded in summaries/leaderboard (e.g. shakeout)")
        q.add_argument("--universe-coverage", default="filter", choices=["filter", "strict"],
                       help="filter: stock universes restricted to members with data (coverage recorded); "
                            "strict: engine rule (NOT_TESTABLE if any member lacks data)")
        q.add_argument("--min-universe-coverage", type=float, default=0.95)
        q.add_argument("--extend-grid", action="store_true",
                       help="grow existing strategies' grid subsets to the current caps (adds configs, re-finalizes)")
        q.add_argument("--allow-stock-universes", action="store_true",
                       help="lift the integrator's hold on single-stock strategies (data identity fix pending)")
        if name == "audit":
            q.add_argument("--live", action="store_true",
                           help="run missing baselines now (scheduler, baseline only) instead of only reading results")
    lp = sub.add_parser("launch", help="run any subcommand detached from a clean pinned git worktree")
    lp.add_argument("--commit", default="HEAD")
    lp.add_argument("rest", nargs=argparse.REMAINDER, help="subcommand and its args, e.g. run --ids S050")
    r = sub.add_parser("replicate", help="end-to-end known-answer validations -> REPLICATION.md")
    r.add_argument("--only", help="comma list of: spy,s105,put,umd,hml,random,lookahead")
    r.add_argument("--quick", action="store_true", help="shorter windows (used by the pytest)")
    a = ap.parse_args(argv)
    a.results_dir = Path(a.results_dir)
    return {"plan": cmd_plan, "run": cmd_run, "atlas": cmd_atlas, "audit": cmd_audit,
            "replicate": cmd_replicate, "launch": cmd_launch}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
