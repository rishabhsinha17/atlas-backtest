"""Re-tier stored fills of finished runs with the ENGINE's deterministic reason->tier mapping.

dd5e315 made calibrated bar fills T1 (the contract) and exposed ``atlasbt.execution.retier_fill`` so
results produced before it (batch A at 8985af1, options at 307fa95) need not be re-run. For every
``runs/<SID>/baseline/result.pkl.gz`` not yet re-tiered:

* each fill's ``fidelity`` is replaced by ``retier_fill(fidelity, liquidity, fidelity_reason, asset_class)``
  (option/combo, quote, auction and lifecycle fills keep their recorded tier);
* ``result.fidelity['tiers']`` is rebuilt from the re-tiered fills (fills, notional, notional_share);
  the engine's original grade and tiers are kept as ``grade_before_retier`` / ``tiers_before_retier``;
* ``result.meta['retiered_with'] = <label>`` plus the (old -> new) change counts;
* finalized strategies are re-finalized so report.md/json and the leaderboard row use the new tiers.
Idempotent: results whose meta already carries ``retiered_with`` are skipped.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from atlasbt.core.types import Fidelity

from . import pipeline as P
from .common import read_pickle, write_json, write_pickle


def retier_result(res, label: str) -> dict:
    from atlasbt.execution import retier_fill

    f = res.fills
    if f is None or not len(f) or "fidelity" not in f.columns:
        res.meta["retiered_with"] = label
        res.meta["retier_changes"] = {}
        return {}
    reasons = f["fidelity_reason"] if "fidelity_reason" in f.columns else [""] * len(f)
    liq = f["liquidity"] if "liquidity" in f.columns else ["TAKER"] * len(f)
    ac = f["asset_class"] if "asset_class" in f.columns else ["STK"] * len(f)
    old = [int(x) for x in f["fidelity"]]
    new = [int(retier_fill(o, l, r, a)) for o, l, r, a in zip(old, liq, reasons, ac)]
    changes = Counter((Fidelity(o).name, Fidelity(n).name) for o, n in zip(old, new) if o != n)
    f = f.copy()
    f["fidelity"] = new
    if "fidelity_name" in f.columns:
        f["fidelity_name"] = [Fidelity(n).name for n in new]
    res.fills = f
    fd = dict(res.fidelity or {})
    notional = f["notional"].abs() if "notional" in f.columns else (f["quantity"] * f["price"]).abs()
    tot = float(notional.sum()) or 1.0
    tiers = {}
    for t, idx in f.groupby("fidelity").groups.items():
        n = float(notional.loc[idx].sum())
        tiers[Fidelity(int(t)).name] = {"fills": int(len(idx)), "notional": n, "notional_share": n / tot}
    fd["tiers_before_retier"] = fd.get("tiers")
    fd["grade_before_retier"] = fd.get("grade")
    fd["tiers"] = tiers
    fd["grade"] = None                       # reporting.fidelity.grade recomputes from the tiers
    fd.setdefault("reasons", [])
    fd["reasons"] = list(fd["reasons"]) + [{"tier": "INFO", "reason": f"fills re-tiered with {label} "
                                            f"(engine reason->tier mapping): {dict(changes)}", "count": sum(changes.values())}]
    res.fidelity = fd
    res.meta = dict(res.meta or {})
    res.meta["retiered_with"] = label
    res.meta["retier_changes"] = {f"{a}->{b}": n for (a, b), n in changes.items()}
    return res.meta["retier_changes"]


def retier_dir(results_dir: Path, label: str, refinalize: bool = True, log=print) -> list[dict]:
    from .plan import load_plan

    results_dir = Path(results_dir)
    plan = load_plan(results_dir)
    rows = {r["id"]: r for r in plan.to_dict("records")} if len(plan) else {}
    out = []
    for d in sorted((results_dir / "runs").glob("S*")):
        sid = d.name
        pk = d / "baseline" / "result.pkl.gz"
        if not pk.exists():
            continue
        res = read_pickle(pk)
        if (res.meta or {}).get("retiered_with"):
            continue
        ch = retier_result(res, label)
        write_pickle(pk, res)
        st = P.baseline_status(P.RunOptions(results_dir=str(results_dir)), sid)
        if st:
            st["retiered_with"] = label
            st["retier_changes"] = ch
            write_json(d / "baseline" / "status.json", st)
        rec = {"id": sid, "changes": ch, "refinalized": False}
        if refinalize and (d / "final" / "summary.json").exists() and sid in rows:
            from .common import read_json

            old = read_json(d / "final" / "summary.json", {}) or {}
            opts = P.RunOptions(results_dir=str(results_dir), extra={"label": old.get("run_label")})
            P.task_finalize(sid, rows[sid], opts)
            rec["refinalized"] = True
        log(f"retier {sid}: {ch or 'no change'}{' (re-finalized)' if rec['refinalized'] else ''}")
        out.append(rec)
    write_json(results_dir / f"retier_{label}.json", out)
    return out
