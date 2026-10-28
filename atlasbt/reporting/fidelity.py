"""Fidelity scorecard: normalise, grade and render.

The engine's ``BacktestResult.fidelity`` dict is read through ``normalize_scorecard``
which accepts these shapes (and falls back to recomputing from the fills'
``fidelity`` column when no tier table is present):

  {'tiers': {'T1_BAR': {'fills': 120, 'notional': 1.2e6}, ...}, 'reasons': [...]}
  {'T1_BAR': {'fills': ..., 'notional': ...}, ..., 'reasons': {...}}
  reasons: list[str] | {reason: count} | list[{'tier':..., 'reason':..., 'count':...}]

Grade (notional-weighted mean tier w, Fidelity enum values 0..3):
  A: w <= 0.25 (essentially quote-level)   B: w <= 1.0 (real bars, modelled spread)
  C: w <= 1.75   D: w <= 2.5   E: w > 2.5 (mostly modelled data)
  NA: any notional at tier NA (9) or run NOT_TESTABLE;  '-': no fills.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from atlasbt.core.types import Fidelity

from .schema import _tier, normalize_fills

TIER_NAMES = {int(f): f.name for f in Fidelity}


def normalize_scorecard(fidelity: dict | None, fills=None) -> dict:
    fd = dict(fidelity or {})
    tiers_raw = fd.get("tiers")
    if tiers_raw is None:
        tiers_raw = {k: v for k, v in fd.items() if not np.isnan(_tier(k)) and isinstance(v, dict)}
    tiers: dict[str, dict] = {}
    for k, v in (tiers_raw or {}).items():
        t = _tier(k)
        if np.isnan(t):
            continue
        name = TIER_NAMES.get(int(t), str(k))
        tiers[name] = {"fills": int(v.get("fills", v.get("count", 0))), "notional": float(v.get("notional", 0.0))}
    if not tiers and fills is not None:
        f = normalize_fills(fills)
        if not f.empty and f["fidelity"].notna().any():
            for t, g in f.groupby("fidelity"):
                tiers[TIER_NAMES.get(int(t), str(t))] = {"fills": int(len(g)), "notional": float(g["notional"].sum())}
    reasons_raw = fd.get("reasons", [])
    reasons: list[dict] = []
    if isinstance(reasons_raw, dict):
        reasons = [{"reason": str(k), "count": (v if isinstance(v, (int, float)) else len(v) if hasattr(v, "__len__") else 1)}
                   for k, v in reasons_raw.items()]
    else:
        for r in reasons_raw or []:
            if isinstance(r, dict):
                reasons.append({"reason": str(r.get("reason", r)), "tier": r.get("tier"), "count": r.get("count", 1)})
            else:
                reasons.append({"reason": str(r), "count": 1})
    return {"tiers": tiers, "reasons": reasons}


def grade(scorecard: dict, status: str = "OK") -> str:
    if str(status).upper().endswith("NOT_TESTABLE"):
        return "NA"
    tiers = scorecard.get("tiers", {})
    tot = sum(v["notional"] for v in tiers.values())
    if not tiers or tot <= 0:
        cnt = sum(v["fills"] for v in tiers.values())
        if cnt == 0:
            return "-"
        w = sum(int(Fidelity[k]) * v["fills"] for k, v in tiers.items() if k in Fidelity.__members__) / cnt
    else:
        if any(k == "NA" and v["notional"] > 0 for k, v in tiers.items()):
            return "NA"
        w = sum(int(Fidelity[k]) * v["notional"] for k, v in tiers.items() if k in Fidelity.__members__) / tot
    for g, lim in (("A", 0.25), ("B", 1.0), ("C", 1.75), ("D", 2.5)):
        if w <= lim + 1e-12:
            return g
    return "E"


def scorecard_table(scorecard: dict) -> pd.DataFrame:
    tiers = scorecard.get("tiers", {})
    rows = []
    tot_n = sum(v["fills"] for v in tiers.values()) or 1
    tot_x = sum(v["notional"] for v in tiers.values()) or 1.0
    for f in Fidelity:
        if f.name in tiers:
            v = tiers[f.name]
            rows.append({"tier": f.name, "fills": v["fills"], "fill_share": v["fills"] / tot_n,
                         "notional": v["notional"], "notional_share": v["notional"] / tot_x})
    return pd.DataFrame(rows, columns=["tier", "fills", "fill_share", "notional", "notional_share"])


def render_scorecard_md(scorecard: dict, status: str = "OK") -> str:
    t = scorecard_table(scorecard)
    lines = [f"**Fidelity grade: {grade(scorecard, status)}**", ""]
    if t.empty:
        lines.append("_No fills recorded._")
    else:
        lines += ["| Tier | Fills | % fills | Notional (USD) | % notional |", "|---|---:|---:|---:|---:|"]
        for _, r in t.iterrows():
            lines.append(f"| {r.tier} | {r.fills:,} | {r.fill_share:.1%} | {r.notional:,.0f} | {r.notional_share:.1%} |")
    rs = scorecard.get("reasons", [])
    if rs:
        lines += ["", "Reasons:"]
        for r in sorted(rs, key=lambda x: -float(x.get("count", 1) or 1)):
            tier = f" [{r['tier']}]" if r.get("tier") is not None else ""
            lines.append(f"- {r['reason']}{tier} (x{r.get('count', 1)})")
    return "\n".join(lines)
