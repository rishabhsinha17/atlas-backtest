"""Atlas-wide leaderboard sorted by statistically adjusted metrics.

Rows come from ``summarize_result`` (one per strategy; for a strategy with a
parameter grid, summarise the walk-forward OOS or the selected configuration
and pass its grid size so the trial count is honest).

Columns: strategy_id, status, sharpe (raw, for reference only), psr_0,
dsr (deflated with the TOTAL number of trials across the atlas), p_value
(= 1 - PSR(0)), q_value (Benjamini-Hochberg over all tested strategies),
fdr_pass, pbo, oos_sharpe, cagr, max_drawdown, break_even_bp, fidelity_grade.

Sort order: DSR desc -> PBO asc (missing last) -> q-value asc -> PSR desc.
Raw Sharpe never determines rank. NOT_TESTABLE / FAILED strategies are listed
separately with their reasons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fidelity import grade, normalize_scorecard
from .metrics import risk_free_returns


def summarize_result(result, pbo: float | None = None, oos_sharpe: float | None = None,
                     break_even_bp: float | None = None, n_trials: int | None = None) -> dict:
    from atlasbt.validation.sharpe_tests import moments

    status = getattr(result.status, "value", str(result.status))
    reason = getattr(result, "not_testable_reason", "") or ""
    if not reason and isinstance(getattr(result, "diagnostics", None), dict):
        reason = str(result.diagnostics.get("error", ""))
    row = {"strategy_id": result.strategy_id, "status": status, "reason": reason}
    sc = normalize_scorecard(result.fidelity, result.fills if status == "OK" else None)
    row["fidelity_grade"] = grade(sc, status)
    if status == "OK" and result.returns is not None and len(result.returns) > 2:
        r = result.returns.astype(float).dropna()
        rf, _ = risk_free_returns(result)
        ex = (r - rf.reindex(r.index).fillna(0.0)).to_numpy()
        m = moments(ex)
        row.update({"n_obs": m["n"], "sr_period": m["sr"], "skew": m["skew"], "kurt": m["kurt"],
                    "sharpe": m["sr"] * np.sqrt(252)})
        eq = result.equity.astype(float)
        row["max_drawdown"] = float((eq / eq.cummax() - 1).min())
        years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9) if isinstance(eq.index, pd.DatetimeIndex) else len(eq) / 252
        g = float(np.prod(1 + r.to_numpy()))
        row["cagr"] = g ** (1 / years) - 1 if g > 0 else -1.0
    row.update({"pbo": pbo, "oos_sharpe": oos_sharpe, "break_even_bp": break_even_bp, "grid_trials": n_trials})
    return row


def build_leaderboard(rows: list[dict], n_trials_total: int | None = None, var_sr: float | None = None,
                      fdr_q: float = 0.10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (tested_leaderboard, not_testable_and_failed).

    n_trials_total: total configurations tried across the atlas (default:
    sum of ``grid_trials`` in rows, or the atlas-wide parsed count, whichever
    is larger). var_sr: cross-trial variance of per-period Sharpe; default =
    sample variance of the tested strategies' per-period Sharpes (floored at
    1e-6)."""
    from atlasbt.validation.multiple_testing import benjamini_hochberg
    from atlasbt.validation.sharpe_tests import dsr, psr, total_atlas_trials

    df = pd.DataFrame(rows)
    if df.empty:
        return df, df
    ok = df[(df["status"] == "OK") & df.get("sr_period", pd.Series(index=df.index, dtype=float)).notna()].copy()
    other = df[~df.index.isin(ok.index)][["strategy_id", "status", "reason", "fidelity_grade"]].copy()
    if ok.empty:
        return ok, other
    grid_sum = int(pd.to_numeric(ok.get("grid_trials"), errors="coerce").fillna(1).sum()) if "grid_trials" in ok else len(ok)
    N = n_trials_total or max(grid_sum, total_atlas_trials())
    v = var_sr if var_sr is not None else max(float(np.var(ok["sr_period"], ddof=1)) if len(ok) > 1 else 0.0, 1e-6)
    ok["psr_0"] = [psr(r.sr_period, 0.0, int(r.n_obs), r.skew, r.kurt) for r in ok.itertuples()]
    ok["dsr"] = [dsr(r.sr_period, int(r.n_obs), r.skew, r.kurt, N, v) for r in ok.itertuples()]
    ok["p_value"] = 1.0 - ok["psr_0"]
    rej, qv = benjamini_hochberg(ok["p_value"].fillna(1.0).to_numpy(), fdr_q)
    ok["q_value"] = qv
    ok["fdr_pass"] = rej
    ok["n_trials_total"] = N
    ok["var_sr_period"] = v
    ok["_pbo_sort"] = pd.to_numeric(ok["pbo"], errors="coerce").fillna(2.0)
    ok = ok.sort_values(["dsr", "_pbo_sort", "q_value", "psr_0"], ascending=[False, True, True, False],
                        na_position="last", kind="mergesort").drop(columns="_pbo_sort")
    ok.insert(0, "rank", np.arange(1, len(ok) + 1))
    cols = ["rank", "strategy_id", "dsr", "psr_0", "q_value", "fdr_pass", "pbo", "oos_sharpe", "sharpe", "cagr",
            "max_drawdown", "break_even_bp", "fidelity_grade", "n_obs", "n_trials_total"]
    return ok[[c for c in cols if c in ok.columns]].reset_index(drop=True), other.reset_index(drop=True)


def render_leaderboard_md(tested: pd.DataFrame, other: pd.DataFrame, title: str = "Atlas leaderboard") -> str:
    def f(v, k="f"):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        if k == "pct":
            return f"{v:.1%}"
        if k == "b":
            return "yes" if v else "no"
        return f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)

    lines = [f"# {title}", "",
             "Ranked by Deflated Sharpe Ratio (all-atlas trial count), then PBO, then BH q-value. "
             "Raw Sharpe is shown for reference only.", ""]
    if not tested.empty:
        n = int(tested["n_trials_total"].iloc[0]) if "n_trials_total" in tested else None
        lines += [f"Trials deflated against: N = {n}", "",
                  "| # | Strategy | DSR | PSR(0) | q (BH) | FDR pass | PBO | OOS SR | raw SR | CAGR | MaxDD | Break-even bp | Fidelity |",
                  "|---:|---|---:|---:|---:|:-:|---:|---:|---:|---:|---:|---:|:-:|"]
        for r in tested.itertuples():
            lines.append(f"| {r.rank} | {r.strategy_id} | {f(r.dsr)} | {f(r.psr_0)} | {f(r.q_value)} | {f(r.fdr_pass, 'b')} | "
                         f"{f(getattr(r, 'pbo', None))} | {f(getattr(r, 'oos_sharpe', None))} | {f(r.sharpe)} | "
                         f"{f(getattr(r, 'cagr', None), 'pct')} | {f(getattr(r, 'max_drawdown', None), 'pct')} | "
                         f"{f(getattr(r, 'break_even_bp', None))} | {r.fidelity_grade} |")
    if not other.empty:
        lines += ["", "## Not testable / failed", "", "| Strategy | Status | Reason |", "|---|---|---|"]
        for r in other.itertuples():
            lines.append(f"| {r.strategy_id} | {r.status} | {str(r.reason).replace('|', '/') or 'n/a'} |")
    return "\n".join(lines) + "\n"


def write_leaderboard(rows: list[dict], out_dir, **kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    from pathlib import Path

    tested, other = build_leaderboard(rows, **kw)
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "leaderboard.md").write_text(render_leaderboard_md(tested, other))
    tested.to_csv(d / "leaderboard.csv", index=False)
    other.to_csv(d / "not_testable.csv", index=False)
    return tested, other
