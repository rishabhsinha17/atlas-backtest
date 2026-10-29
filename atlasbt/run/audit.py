"""`audit`: activity audit of every registered strategy's baseline run.

A strategy fails the audit when its baseline run (sheet params, full plan window, real cache) has
status OK but ZERO_ACTIVITY and the strategy declares no ``activity_note``. NOT_TESTABLE runs (and
plan-level NOT_TESTABLE) pass: the engine said why. LOW_ACTIVITY, declared notes, FAILED runs and
integrator holds are listed for review. Output: results/AUDIT.md and results/audit.json; the pytest
``tests/run/test_audit.py`` fails on the offenders.
"""
from __future__ import annotations

from pathlib import Path

from . import pipeline as P
from .common import read_json, read_pickle, write_json


#: Diagnosed infrastructure causes, listed in the audit until fixed. (The synthetic-option fill gap was
#: fixed on 2026-10-29: Portal.option_quotes now falls back to synthetic snapshots.)
KNOWN_CAUSES: list = []


def known_cause(sid: str, plan_row: dict, diag: dict) -> str:
    """Infrastructure causes of zero activity that are already diagnosed (listed until fixed)."""
    text = " ".join([str(plan_row.get("fidelity_notes", "")),
                     " ".join(map(str, (diag or {}).get("fidelity_reasons") or []))]).lower()
    for pattern, cause in KNOWN_CAUSES:
        if pattern in text:
            return cause
    o = (diag or {}).get("orders") or {}
    if o.get("combo_submitted") and not o.get("combo_filled"):
        return (f"{o['combo_submitted']} combo orders submitted, none filled "
                f"(cancelled {o.get('combo_cancelled')}, rejected {o.get('combo_rejected')}: "
                f"{o.get('cancel_reject_reasons')})")
    sel = ((diag or {}).get("strategy_logs") or {}).get("by_category", {}).get("selection_failure")
    if sel:
        return f"{sel} option selection failures logged by the strategy"
    return ""


def baseline_activity(opts, sid: str, plan_row: dict) -> dict:
    st = P.baseline_status(opts, sid)
    if not st:
        return {}
    if st.get("status") == "OK" and not st.get("activity"):
        f = P.run_dir(opts, sid) / "baseline" / "result.pkl.gz"
        if f.exists():
            a = P._activity(sid, read_pickle(f), plan_row)
            st["activity"], st["activity_detail"] = a.get("activity"), a
    return st


def audit_rows(opts, ids=None, allow_stock: bool = False) -> list[dict]:
    from atlasbt.strategies import registry as R

    from .activity import declared_note_for
    from .holds import held, is_stale
    from .plan import load_plan

    plan = load_plan(Path(opts.results_dir))
    rows = {r["id"]: r for r in plan.to_dict("records")} if len(plan) else {}
    out = []
    for sid in ids or R.all_ids():
        pr = rows.get(sid, {})
        note = declared_note_for(sid)
        rec = {"id": sid, "family": pr.get("family", ""), "bar_freq": pr.get("bar_freq", ""), "declared_note": note}
        h = held(sid, allow_stock)
        if pr.get("testable") == "N":
            rec.update(verdict="PASS", status="NOT_TESTABLE (plan)", detail=pr.get("reason", ""))
        elif h:
            rec.update(verdict="HOLD", status="HOLD", detail=h + (" (existing results STALE)" if is_stale(opts.results_dir, sid) else ""))
        else:
            st = baseline_activity(opts, sid, pr)
            if not st:
                rec.update(verdict="NOT_RUN", status="NOT_RUN", detail="no baseline run yet (python -m atlasbt.run audit --live)")
            elif st.get("status") == "NOT_TESTABLE":
                rec.update(verdict="PASS", status="NOT_TESTABLE", detail=st.get("reason", ""))
            elif st.get("status") in ("FAILED", "TIMEOUT"):
                tb = (st.get("traceback") or "").strip().splitlines()
                rec.update(verdict="FAILED", status=st["status"],
                           detail=(st.get("reason") or "") + (f" | {tb[-1]}" if tb else ""))
            else:
                a = st.get("activity_detail") or {}
                act = st.get("activity")
                dg = a.get("diagnostics") or {}
                det = (f"{a.get('n_entries')} {a.get('entry_unit')} in {a.get('years')}y (floor {a.get('floor')}); "
                       f"fills {a.get('n_fills')}; rejects {dg.get('top_reject_reasons')}; warnings "
                       f"{(dg.get('warnings') or [])[:2]}; fidelity {(dg.get('fidelity_reasons') or [])[:2]}; "
                       f"log results/logs/{sid}.log")
                if act == "ZERO_ACTIVITY":
                    cause = known_cause(sid, pr, dg)
                    rec.update(verdict="DECLARED" if note else "OFFENDER", status=act,
                               detail=(f"KNOWN CAUSE: {cause} || " if cause else "") + det)
                elif act == "LOW_ACTIVITY":
                    rec.update(verdict="REVIEW", status=act, detail=det)
                else:
                    rec.update(verdict="PASS", status=act or "OK",
                               detail=f"{a.get('n_entries')} {a.get('entry_unit')} in {a.get('years')}y")
        out.append(rec)
    return out


def run_audit(opts, ids=None, live: bool = False, allow_stock: bool = False, log=print) -> list[str]:
    from .holds import held
    from .plan import annotate_plan, load_plan, make_plan, write_plan
    from .scheduler import Scheduler

    res_dir = Path(opts.results_dir)
    plan = load_plan(res_dir)
    if not len(plan):
        write_plan(make_plan(None, res_dir, log=log), res_dir)
        plan = load_plan(res_dir)
    if live:
        rows = {r["id"]: r for r in plan.to_dict("records")}
        todo = [s for s in (ids or list(rows)) if rows.get(s, {}).get("testable") == "Y" and not held(s, allow_stock)
                and not P.baseline_status(opts, s)]
        if todo:
            opts.extra["baseline_only"] = True
            log(f"audit: running {len(todo)} missing baselines")
            Scheduler(rows, opts, log=log).run(todo)
    recs = audit_rows(opts, ids, allow_stock)
    annotate_plan(res_dir, allow_stock)
    write_json(res_dir / "audit.json", recs)
    _write_md(res_dir / "AUDIT.md", recs)
    bad = [r["id"] for r in recs if r["verdict"] == "OFFENDER"]
    log(f"audit: {len(bad)} offenders {bad}; -> {res_dir / 'AUDIT.md'}")
    return bad


def _write_md(path: Path, recs: list[dict]) -> None:
    from collections import Counter

    c = Counter(r["verdict"] for r in recs)
    L = ["# Activity audit", "",
         "Baseline runs (sheet params, planned window, real cache) checked by the runner's activity safeguard "
         "(atlasbt/run/activity.py). OFFENDER = status OK but ZERO_ACTIVITY without a declared `activity_note`; "
         "REVIEW = LOW_ACTIVITY (below the family plausibility floor); DECLARED = zero activity with a declared "
         "reason; HOLD = integrator data hold; NOT_RUN = no baseline yet.", "",
         "| Verdict | Count |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in sorted(c.items())] + [""]
    for verdict in ("OFFENDER", "REVIEW", "DECLARED", "FAILED", "HOLD", "NOT_RUN", "PASS"):
        sub = [r for r in recs if r["verdict"] == verdict]
        if not sub:
            continue
        L += [f"## {verdict} ({len(sub)})", "", "| Strategy | Family | Freq | Status | Detail |", "|---|---|---|---|---|"]
        for r in sub:
            L.append(f"| {r['id']} | {str(r.get('family', ''))[:26]} | {r.get('bar_freq', '')} | {r.get('status', '')} | "
                     f"{str(r.get('detail', '')).replace('|', '/')[:400]}"
                     + (f" (declared: {r['declared_note']})" if r.get("declared_note") else "") + " |")
        L.append("")
    path.write_text("\n".join(L) + "\n")
