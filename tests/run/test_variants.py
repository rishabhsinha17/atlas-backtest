"""Proxy variants (atlasbt/run/variants.py): plan, grid, engine labelling, atlas separation.

Fast and offline: an InMemoryPortal with SYNTHETIC daily bars (mechanics only) and hand-made run summaries."""
import json

import numpy as np
import pandas as pd
import pytest

from atlasbt.engine.testing import InMemoryPortal, synthetic_daily_bars
from atlasbt.run import variants as V
from atlasbt.run.plan import PlanContext, plan_strategy, write_plan
from atlasbt.strategies.calendar_events import SWING_ETF

START, END = "2014-01-02", "2016-12-30"


@pytest.fixture(scope="module")
def daily_portal():
    """SYNTHETIC daily bars for the SWING_ETF universe (no 1-minute bars at all)."""
    bars = {(s, "1d"): synthetic_daily_bars(START, END, start_price=50 + 5 * i, seed=i)
            for i, s in enumerate(SWING_ETF)}
    return InMemoryPortal(bars=bars)


@pytest.fixture()
def pc(daily_portal, tmp_path):
    return PlanContext(daily_portal, tmp_path, min_coverage=0.95)


# ----------------------------------------------------------------- registry
def test_registry_rules():
    assert set(V.variant_ids()) >= {"S031@proxy", "S034@proxy", "S035@proxy", "S036@proxy", "S037@proxy"}
    for vid, v in V.VARIANTS.items():
        assert v.base == V.base_of(vid) and v.reason
        assert "allow_synthetic" not in v.params          # OPTIONS.md: synthetic smiles never drive signals
        cls = V.strategy_class(vid)
        assert all(k in cls.params for k in v.params)     # only the strategy's own opt-in params
    assert not any(k.startswith("S16") for k in V.VARIANTS)
    assert V.normalize_id("s031@PROXY") == "S031@proxy" and V.normalize_id("s050") == "S050"


def test_cli_ids_accept_variants():
    from atlasbt.run.__main__ import _ids

    assert _ids("S031@proxy,s034@Proxy,S050-S051") == ["S031@proxy", "S034@proxy", "S050", "S051"]
    assert _ids("variants") == V.variant_ids()


def test_grid_is_base_grid_restricted_to_variant_params(monkeypatch):
    from atlasbt.strategies import registry as R

    g = V.expand_grid("S031@proxy")
    assert len(g) == len(R.expand_grid(R.get("S031").grid)) == 9      # proxy is not a grid key: full base grid
    assert V.default_params("S031@proxy")["proxy"] is True
    monkeypatch.setitem(V.VARIANTS, "S031@n3", V._v("S031@n3", {"proxy": True, "nights": 3}, "test"))
    g3 = V.expand_grid("S031@n3")
    assert len(g3) == 3 and all(c["nights"] == 3 for c in g3)


def test_grid_plan_runs_every_config_with_proxy(tmp_path):
    from atlasbt.run.pipeline import RunOptions, grid_plan

    gp = grid_plan("S031@proxy", {"bar_freq": "1d"}, RunOptions(results_dir=str(tmp_path)), 1.0)
    assert gp["n_full"] == 9 and gp["variant"]["variant_of"] == "S031"
    assert all(c["params"]["proxy"] is True for c in gp["configs"])
    assert sum(c["is_baseline"] for c in gp["configs"]) == 1
    assert (tmp_path / "runs" / "S031@proxy" / "grid_plan.json").exists()


# ----------------------------------------------------------------- plan
def test_plan_proxy_variant_testable_on_daily_data(pc, tmp_path):
    base = plan_strategy("S031", pc)
    assert base["testable"] == "N" and "1m" in base["reason"] and base["variant_of"] == ""
    v = plan_strategy("S031@proxy", pc)
    assert v["testable"] == "Y", v["reason"]
    assert v["variant_of"] == "S031" and v["bar_freq"] == "1d"
    assert v["expected_fidelity"] == "T2_PROXY" and "PROXY VARIANT" in v["fidelity_notes"]
    assert v["n_grid"] == 9 and v["n_sessions"] >= 252
    for vid in ("S034@proxy", "S035@proxy", "S036@proxy", "S037@proxy"):
        assert plan_strategy(vid, pc)["testable"] == "Y", vid
    out = write_plan(pd.DataFrame([base, v]), tmp_path)
    df = pd.read_csv(out, dtype=str).fillna("")
    assert "variant_of" in df.columns
    assert df.set_index("id").loc["S031@proxy", "variant_of"] == "S031"


@pytest.mark.parametrize("vid", ["S032@proxy", "S033@proxy"])
def test_plan_refused_proxy_stays_not_testable(pc, vid):
    r = plan_strategy(vid, pc)
    assert r["testable"] == "N" and "daily_proxy_unsupported" in r["reason"]
    assert r["variant_of"] == vid.split("@")[0]


def test_plan_unknown_variant(pc):
    r = plan_strategy("S031@bogus", pc)
    assert r["testable"] == "N" and "unknown proxy variant" in r["reason"]


# ----------------------------------------------------------------- engine
def test_engine_run_labels_variant_and_fills_at_auctions(daily_portal, tmp_path):
    from atlasbt.run.pipeline import RunOptions, run_engine, strategy_params

    p = strategy_params("S031@proxy", {"mom_days": 0})
    assert p["proxy"] is True
    res, strat, _ = run_engine("S031@proxy", p, "2016-03-01", "2016-04-29", RunOptions(results_dir=str(tmp_path)),
                               portal=daily_portal)
    assert getattr(res.status, "value", str(res.status)) == "OK", res.not_testable_reason
    assert res.strategy_id == "S031@proxy" and res.meta["variant"]["variant_of"] == "S031"
    assert strat.bar_freq == "1d" and len(res.fills) > 0
    reasons = json.dumps(res.fidelity, default=str)
    assert "daily proxy" in reasons and "T2" in reasons


# ----------------------------------------------------------------- atlas separation
def _row(sid, sr, n=1500, seed=0):
    return {"strategy_id": sid, "status": "OK", "reason": "", "fidelity_grade": "C", "n_obs": n, "sr_period": sr,
            "skew": 0.0, "kurt": 3.0, "sharpe": sr * np.sqrt(252), "max_drawdown": -0.2, "cagr": 0.05, "pbo": 0.3,
            "oos_sharpe": 0.4, "break_even_bp": 5.0, "grid_trials": 6, "activity": "OK"}


def _final(res, sid, row, variant=None):
    from atlasbt.run.common import write_json

    write_json(res / "runs" / sid / "final" / "summary.json",
               {"strategy_id": sid, "status": "OK", "leaderboard_row": row, "variant": variant or {},
                "grid_sr_period": [row["sr_period"], row["sr_period"] * 0.5],
                "baseline": {"window": ["2005-01-03", "2026-10-27"]}, "grid": {"n_ok": 2, "n_full": 6}})


def test_atlas_keeps_variants_out_of_primary_leaderboard(tmp_path, monkeypatch):
    import atlasbt.run.atlas as A
    from atlasbt.run import pipeline as P

    monkeypatch.setattr(A, "all_sheet_ids", lambda: ["S050", "S031"])
    plan = pd.DataFrame([
        {"id": "S050", "variant_of": "", "testable": "Y", "bar_freq": "1d", "family": "f", "start": "2005-01-03",
         "end": "2026-10-27"},
        {"id": "S031", "variant_of": "", "testable": "N", "reason": "required bars 1m for EFA unavailable"},
        {"id": "S031@proxy", "variant_of": "S031", "testable": "Y", "bar_freq": "1d"},
        {"id": "S032@proxy", "variant_of": "S032", "testable": "N",
         "reason": "required daily_proxy_unsupported unavailable"}])
    write_plan(plan, tmp_path)
    _final(tmp_path, "S050", _row("S050", 0.03))
    _final(tmp_path, "S031@proxy", _row("S031@proxy", 0.08), V.info("S031@proxy"))   # a "better" variant
    opts = P.RunOptions(results_dir=str(tmp_path), make_charts=False)
    out = A.run_atlas(opts, log=lambda *a: None, allow_stock=True)

    lb = pd.read_csv(tmp_path / "leaderboard.csv")
    assert list(lb["strategy_id"]) == ["S050"]                       # variant never ranked with faithful runs
    assert int(lb["n_trials_total"].iloc[0]) == P.atlas_trial_total()  # primary N excludes variant trials
    assert "S031@proxy" not in out["status_counts"] and out["status_counts"].get("NOT_TESTABLE") == 1
    pv = pd.read_csv(tmp_path / "proxy_variants.csv").set_index("strategy_id")
    assert pv.loc["S031@proxy", "status"] == "OK" and pv.loc["S031@proxy", "variant_of"] == "S031"
    assert pv.loc["S032@proxy", "status"] == "NOT_TESTABLE"
    nv = int(pv.loc["S031@proxy", "n_trials_total"])
    assert nv == P.atlas_trial_total() + len(V.expand_grid("S031@proxy")) == out["proxy_variants"]["n_trials_total"]
    md = (tmp_path / "ATLAS_REPORT.md").read_text()
    lead = md.split("## Leaderboard", 1)[1].split("\n## ", 1)[0]
    assert "S031@proxy" not in lead and "S050" in lead
    sec = md.split("## PROXY VARIANTS — T2, not faithful to the sheet's timing", 1)[1].split("\n## ", 1)[0]
    assert "S031@proxy" in sec and "S032@proxy" in sec and f"N = {nv}" in sec


def test_audit_covers_variants(tmp_path):
    from atlasbt.run.audit import audit_rows
    from atlasbt.run.pipeline import RunOptions

    write_plan(pd.DataFrame([{"id": "S032@proxy", "variant_of": "S032", "testable": "N", "reason": "refused"}]),
               tmp_path)
    recs = {r["id"]: r for r in audit_rows(RunOptions(results_dir=str(tmp_path)), None)}
    assert set(V.variant_ids()) <= set(recs)
    assert recs["S032@proxy"]["verdict"] == "PASS" and recs["S032@proxy"]["variant_of"] == "S032"
    assert recs["S031@proxy"]["verdict"] == "NOT_RUN"
    assert "S031@proxy" not in {r["id"] for r in audit_rows(RunOptions(results_dir=str(tmp_path)), None,
                                                             variants=False)}
