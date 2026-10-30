"""Launch long runs from a clean, pinned `git worktree` (integrator rule, 2026-10-29).

Many agents edit the live working tree concurrently; a multi-hour run that imports from it can mix code
versions (e.g. a calibration JSON rewritten mid-run). ``launch`` therefore:

1. resolves the commit (default HEAD) and creates ``results/_worktrees/<hash12>`` with
   ``git worktree add --detach`` (no commits, branches or pushes);
2. overlays the runner package ``atlasbt/run`` from the main tree if the commit lacks it or the main
   tree's runner differs (runner-owned code ahead of the last commit); its sha256 is recorded as
   ``runner_overlay_sha`` (empty when the committed runner is used as is);
3. REFUSES to launch if ``git status --porcelain`` in the worktree shows anything other than the overlay;
4. starts ``python -m atlasbt.run <args>`` detached inside the worktree with ``ATLASBT_CACHE`` = the main
   repo's cache (shared, read-only for runs), ``--results-dir`` = the main repo's results, and
   ``ATLASBT_RUN_COMMIT`` / ``ATLASBT_RUN_WORKTREE`` / ``ATLASBT_RUNNER_OVERLAY_SHA`` exported so every
   result records them (common.environment_info).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .common import REPO


def _git(*args, cwd: Path = REPO, strip: bool = True) -> str:
    out = subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True).stdout
    return out.strip() if strip else out


def tree_sha(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(Path(path).rglob("*.py")):
        h.update(str(f.relative_to(path)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


OVERLAY = ("atlasbt/run",)


def prepare(commit: str = "HEAD", main: Path = REPO, overlay_runner: bool = True) -> dict:
    full = _git("rev-parse", commit, cwd=main)
    wt = main / "results" / "_worktrees" / full[:12]
    if not wt.exists():
        wt.parent.mkdir(parents=True, exist_ok=True)
        _git("worktree", "add", "--detach", str(wt), full, cwd=main)
    overlay = {}
    for rel in (OVERLAY if overlay_runner else ()):
        src, dst = main / rel, wt / rel
        tracked = bool(_git("ls-tree", "-r", "--name-only", full, rel, cwd=main))
        # overlay the runner when the commit lacks it OR the main tree has newer runner code
        if not tracked or tree_sha(src) != tree_sha(dst):
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        # record the overlay whenever the worktree's runner differs from the commit (also when an earlier
        # launch already synced it)
        if not tracked or _git("status", "--porcelain", rel, cwd=wt, strip=False).strip():
            overlay[rel] = tree_sha(dst)
    # porcelain v1: "XY <path>" (two status columns + space); never strip the leading column
    allowed = OVERLAY if overlay_runner else ()
    dirty = [l for l in _git("status", "--porcelain", cwd=wt, strip=False).splitlines()
             if l.strip() and not any(l[3:].startswith(o) for o in allowed)]
    return {"commit": full, "worktree": str(wt), "overlay": overlay, "dirty": dirty}


def launch(args: list[str], commit: str = "HEAD", log_path: str | None = None, main: Path = REPO,
           overlay_runner: bool = True) -> dict:
    info = prepare(commit, main, overlay_runner)
    if info["dirty"]:
        raise SystemExit(f"refusing to launch: worktree {info['worktree']} is dirty: {info['dirty'][:10]}")
    env = dict(os.environ)
    env.update({"ATLASBT_CACHE": str(main / "cache"), "ATLASBT_RUN_COMMIT": info["commit"],
                "ATLASBT_RUN_WORKTREE": info["worktree"], "PYTHONPATH": info["worktree"],
                "ATLASBT_RUNNER_OVERLAY_SHA": ",".join(f"{k}:{v}" for k, v in info["overlay"].items())})
    if "--results-dir" not in args:
        args = ["--results-dir", str(main / "results")] + list(args)
    log = Path(log_path or (main / "results" / "logs" / f"launch_{info['commit'][:12]}.log"))
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "a")
    p = subprocess.Popen([sys.executable, "-m", "atlasbt.run", *args], cwd=info["worktree"], env=env,
                         stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    info.update({"pid": p.pid, "log": str(log), "args": args})
    return info


def running_pinned() -> bool:
    return bool(os.environ.get("ATLASBT_RUN_COMMIT"))
