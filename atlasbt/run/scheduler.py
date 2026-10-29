"""Resumable multi-process scheduler for the atlas run.

Each unit of work (baseline run, one grid config, finalize) runs in its own forked process, so
memory is returned to the OS after every task, a crash or OOM kill of one task cannot take down the
atlas run, and a per-task wall-clock timeout can be enforced. At most ``workers`` tasks run at once
and no new task starts while MemAvailable is below ``mem_reserve_gb`` (unless nothing is running).

Priorities: finalize > grid configs > baselines (strategies complete one after another while all
cores stay busy); baselines are started longest-estimate first.

Resumability: every task writes its outputs atomically; on restart, completed baselines, configs and
finalizations are detected from the files and skipped (``--force`` wipes a strategy first).
"""
from __future__ import annotations

import heapq
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

from . import pipeline as P
from .common import read_json, write_json


def mem_available_gb() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return 99.0


def _child(kind: str, sid: str, payload: dict, plan_row: dict, opts: P.RunOptions, log_path: str,
           status_path: str) -> None:
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)
    print(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} {kind} {sid} {payload.get('cfg', '')}", flush=True)
    try:
        if kind == "baseline":
            out = P.task_baseline(sid, plan_row, opts)
        elif kind == "config":
            out = P.task_config(sid, payload, plan_row, opts)
        elif kind == "finalize":
            out = P.task_finalize(sid, plan_row, opts)
        else:
            raise ValueError(kind)
        write_json(Path(status_path), {"ok": True, "out": out})
    except BaseException as e:  # noqa: BLE001 - every failure must be recorded, never propagate
        tb = traceback.format_exc()
        print(tb, flush=True)
        write_json(Path(status_path), {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": tb})
    print(f"=== done {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


class Scheduler:
    PRIORITY = {"finalize": 0, "config": 1, "baseline": 2}

    def __init__(self, plan_rows: dict, opts: P.RunOptions, log=print):
        self.rows = plan_rows
        self.opts = opts
        self.log = log
        self.heap: list = []
        self.running: dict = {}
        self.counter = 0
        self.ctx = mp.get_context("fork")
        self.logdir = Path(opts.results_dir) / "logs"
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.tmpdir = Path(opts.results_dir) / "runs" / "_task_status"
        self.tmpdir.mkdir(parents=True, exist_ok=True)
        self.pending_configs: dict[str, set] = {}
        self.t0 = time.time()
        # Pin the code for the whole run: import everything the tasks use in the parent BEFORE forking, so
        # every baseline/config/finalize of this run executes the same engine/data/strategy code even while
        # other owners patch files on disk. The code hash at launch is recorded in every config.json.
        import atlasbt.engine  # noqa: F401
        import atlasbt.data.portal  # noqa: F401
        import atlasbt.reporting  # noqa: F401
        import atlasbt.validation  # noqa: F401
        from atlasbt.strategies import registry as _R
        from .common import code_fingerprint

        _R.load_all()
        opts.extra["code_sha_at_launch"] = code_fingerprint()
        opts.extra["launched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.done: list = []

    # -------------------------------------------------------------- queue
    def push(self, kind, sid, payload=None, est=0.0):
        self.counter += 1
        heapq.heappush(self.heap, (self.PRIORITY[kind], -float(est or 0), self.counter, kind, sid, payload or {}))

    def _strategy_state(self, sid):
        o = self.opts
        if P.final_status(o, sid):
            return "done"
        b = P.baseline_status(o, sid)
        if not b or b.get("status") not in P.TERMINAL:
            return "baseline"
        return "grid"

    def seed_queue(self, ids):
        for sid in ids:
            if self.opts.force:
                P.reset_strategy(self.opts, sid)
            st = self._strategy_state(sid)
            row = self.rows[sid]
            if self.opts.extra.get("extend_grid") and st in ("done", "grid"):
                if self._maybe_extend(sid):
                    st = "grid"
            if st == "done" or (st == "grid" and self.opts.extra.get("baseline_only")):
                self.log(f"skip {sid}: already finalized (use --force to rerun)")
                continue
            if st == "baseline":
                est = float(row.get("est_runtime_min_baseline") or 0) if str(row.get("est_runtime_min_baseline")) not in ("", "nan") else 0.0
                self.push("baseline", sid, est=est)
            else:
                self.after_baseline(sid)

    def _maybe_extend(self, sid) -> bool:
        """--extend-grid: recompute the grid subset with the current caps; seeded permutations are
        prefix-stable, so a larger cap only ADDS configs. Re-finalize when anything was added."""
        b = P.baseline_status(self.opts, sid)
        old = read_json(P.run_dir(self.opts, sid) / "grid_plan.json")
        if b.get("status") != "OK" or not old:
            return False
        new = P.grid_plan(sid, self.rows[sid], self.opts, b.get("runtime_sec") or 1.0)
        if new["n_selected"] <= old["n_selected"]:
            from .common import write_json as _wj

            _wj(P.run_dir(self.opts, sid) / "grid_plan.json", old)      # keep the original record
            return False
        f = P.run_dir(self.opts, sid) / "final" / "summary.json"
        if f.exists():
            f.unlink()
        self.log(f"{sid}: grid extended {old['n_selected']} -> {new['n_selected']}/{new['n_full']}")
        return True

    def after_baseline(self, sid):
        if self.opts.extra.get("baseline_only"):
            return
        b = P.baseline_status(self.opts, sid)
        if b.get("status") != "OK":
            self.push("finalize", sid)
            return
        gp = read_json(P.run_dir(self.opts, sid) / "grid_plan.json")
        if not gp:
            gp = P.grid_plan(sid, self.rows[sid], self.opts, b.get("runtime_sec") or 1.0)
            if gp["subset"]:
                self.log(f"{sid}: grid subset {gp['n_selected']}/{gp['n_full']} ({'; '.join(gp['reasons'])})")
        todo = set()
        for c in gp["configs"]:
            if P.grid_config_done(self.opts, sid, c["cfg"]):
                continue
            if c["is_baseline"]:
                P.write_baseline_as_config(sid, c, self.opts)
                continue
            todo.add(c["cfg"])
            self.push("config", sid, c, est=b.get("runtime_sec") or 0)
        self.pending_configs[sid] = todo
        if not todo:
            self.push("finalize", sid)

    # -------------------------------------------------------------- run
    def start(self, item):
        _, _, n, kind, sid, payload = item
        status_path = self.tmpdir / f"{sid}_{kind}_{payload.get('cfg', '')}_{n}.json"
        if status_path.exists():
            status_path.unlink()
        log_path = self.logdir / f"{sid}.log"
        p = self.ctx.Process(target=_child, args=(kind, sid, payload, self.rows[sid], self.opts, str(log_path),
                                                   str(status_path)), daemon=False)
        p.start()
        self.running[p.pid] = (p, kind, sid, payload, time.time(), status_path)
        self.log(f"start {kind:8s} {sid} {payload.get('cfg', '')} pid={p.pid} running={len(self.running)} "
                 f"queued={len(self.heap)} memavail={mem_available_gb():.1f}GB")

    def _record_failure(self, kind, sid, payload, why, tb=""):
        o = self.opts
        if kind == "baseline":
            d = P.run_dir(o, sid) / "baseline"
            d.mkdir(parents=True, exist_ok=True)
            write_json(d / "status.json", {"status": "FAILED" if "timeout" not in why.lower() else "TIMEOUT",
                                           "reason": why, "traceback": tb})
        elif kind == "config":
            d = P.run_dir(o, sid) / "grid"
            d.mkdir(parents=True, exist_ok=True)
            write_json(d / f"{payload['cfg']}.json", {"status": "FAILED" if "timeout" not in why.lower() else "TIMEOUT",
                                                       "cfg": payload["cfg"], "combo": payload["combo"],
                                                       "reason": why, "traceback": tb})
        else:
            d = P.run_dir(o, sid) / "final"
            d.mkdir(parents=True, exist_ok=True)
            write_json(d / "summary.json", {"strategy_id": sid, "status": "FAILED", "reason": f"finalize: {why}",
                                            "baseline_traceback": tb, "plan": self.rows[sid],
                                            "baseline": P.baseline_status(o, sid)})

    def reap(self):
        for pid in list(self.running):
            p, kind, sid, payload, t_start, status_path = self.running[pid]
            elapsed = time.time() - t_start
            if p.is_alive():
                if elapsed > self.opts.task_timeout_min * 60:
                    p.kill()
                    p.join(5)
                    self._record_failure(kind, sid, payload, f"timeout after {elapsed / 60:.0f} min")
                    self.log(f"TIMEOUT {kind} {sid} {payload.get('cfg', '')} after {elapsed / 60:.0f} min")
                    del self.running[pid]
                    self.on_done(kind, sid, payload)
                continue
            p.join()
            del self.running[pid]
            st = read_json(status_path, None)
            if st is None:
                why = f"worker exited with code {p.exitcode} without a status (killed / out of memory?)"
                self._record_failure(kind, sid, payload, why)
                self.log(f"CRASH {kind} {sid}: {why}")
            elif not st.get("ok"):
                self._record_failure(kind, sid, payload, st.get("error", "error"), st.get("traceback", ""))
                self.log(f"ERROR {kind} {sid} {payload.get('cfg', '')}: {st.get('error')}")
            else:
                out = st.get("out") or {}
                self.log(f"done  {kind:8s} {sid} {payload.get('cfg', '')} {out.get('status', '')} "
                         f"{elapsed:.0f}s {str(out.get('reason', ''))[:100]}")
            try:
                status_path.unlink()
            except FileNotFoundError:
                pass
            self.on_done(kind, sid, payload)

    def on_done(self, kind, sid, payload):
        if kind == "baseline":
            self.after_baseline(sid)
            if self.opts.extra.get("baseline_only"):
                self.done.append(sid)
        elif kind == "config" and not self.opts.extra.get("baseline_only"):
            s = self.pending_configs.get(sid, set())
            s.discard(payload["cfg"])
            if not s:
                self.push("finalize", sid)
        else:
            self.done.append(sid)
            fs = P.final_status(self.opts, sid)
            self.log(f"FINISHED {sid}: {fs.get('status')} {str(fs.get('reason', ''))[:120]} "
                     f"[{len(self.done)} finished, {(time.time() - self.t0) / 60:.1f} min elapsed]")

    def run(self, ids):
        self.seed_queue(ids)
        while self.heap or self.running:
            while self.heap and len(self.running) < self.opts.workers:
                if self.running and mem_available_gb() < self.opts.mem_reserve_gb:
                    break
                self.start(heapq.heappop(self.heap))
            time.sleep(1.0)
            self.reap()
        return self.done
