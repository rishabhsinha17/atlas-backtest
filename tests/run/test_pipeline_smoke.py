"""End-to-end smoke of `python -m atlasbt.run run/atlas` on a short window (real cache)."""
import json
import shutil

from atlasbt.run.__main__ import main
from atlasbt.run.common import DEFAULT_RESULTS

from .conftest import needs_cache


@needs_cache
def test_run_pipeline_end_to_end_and_resume(tmp_path):
    if (DEFAULT_RESULTS / "plan_cache.json").exists():
        shutil.copy(DEFAULT_RESULTS / "plan_cache.json", tmp_path / "plan_cache.json")
    args = ["--results-dir", str(tmp_path), "run", "--ids", "S105", "--start", "2015-01-02", "--end",
            "2019-12-31", "--max-configs", "3", "--bootstrap-reps", "100", "--no-charts", "--workers", "2",
            "--allow-live-tree"]
    assert main(args) == 0
    s = json.loads((tmp_path / "runs/S105/final/summary.json").read_text())
    assert s["status"] == "OK" and s["activity"] == "OK"
    assert s["walk_forward"]["n_folds"] >= 1 and s["walk_forward"]["holdout_params"]
    assert s["grid"]["n_selected"] == 3 and s["grid"]["subset"]
    assert "pbo" in s["pbo"]
    assert s["report"]["cost_stress"] and s["report"]["break_even"]
    cfg = json.loads((tmp_path / "runs/S105/config.json").read_text())
    assert cfg["provenance"]["git_commit"] and cfg["data_provenance"]
    md = (tmp_path / "reports/S105/report.md").read_text()
    assert "Walk-forward, untouched holdout" in md and "Cost stress" in md
    mtime = (tmp_path / "runs/S105/final/summary.json").stat().st_mtime
    assert main(args) == 0                                          # resumable: nothing re-run
    assert (tmp_path / "runs/S105/final/summary.json").stat().st_mtime == mtime
    assert main(["--results-dir", str(tmp_path), "atlas"]) == 0
    assert (tmp_path / "ATLAS_REPORT.md").exists() and (tmp_path / "leaderboard.csv").exists()
