"""Per-strategy report: markdown + JSON + PNG charts.

``write_strategy_report(result, out_dir, ...)`` writes
  <out_dir>/<strategy_id>/report.md, report.json, equity.png, drawdown.png,
  rolling_sharpe.png, monthly_returns.png, r_distribution.png
and returns the JSON payload. Statistical adjustments (DSR with the
atlas-wide trial count, bootstrap CIs, cost-stress table, break-even cost)
are included so a single report cannot be read without them.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import charts
from .fidelity import grade, normalize_scorecard, render_scorecard_md
from .metrics import compute_metrics, risk_free_returns
from .schema import normalize_trades


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, pd.DataFrame):
        return to_jsonable(x.reset_index().to_dict(orient="records"))
    if isinstance(x, pd.Series):
        return to_jsonable(x.to_dict())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if math.isnan(v) else (str(v) if math.isinf(v) else v)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp, _dt.datetime, _dt.date)):
        return str(x)
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if hasattr(x, "value") and not isinstance(x, (str, bytes)):
        return to_jsonable(x.value)
    return x


def _fmt(v, kind="f2"):
    if v is None or (isinstance(v, float) and (math.isnan(v))):
        return "n/a"
    if kind == "pct":
        return f"{v:.2%}"
    if kind == "int":
        return f"{int(v):,}"
    if kind == "usd":
        return f"${v:,.0f}"
    return f"{v:.2f}"


def strategy_validation(result, n_trials: int | None = None, var_sr: float | None = None, bootstrap_reps: int = 1000,
                        seed: int = 0, cost_stress: bool = True) -> dict:
    """Statistical adjustments for one result (DSR, PSR, bootstrap CIs, cost stress, break-even)."""
    from atlasbt.validation.bootstrap import bootstrap_metrics
    from atlasbt.validation.cost_stress import break_even_cost, cost_stress_table
    from atlasbt.validation.sharpe_tests import dsr, moments, psr, total_atlas_trials

    out: dict = {}
    r = result.returns.astype(float).dropna()
    rf, _ = risk_free_returns(result)
    ex = (r - rf.reindex(r.index).fillna(0.0)).to_numpy()
    if len(ex) < 30:
        return {"note": "fewer than 30 return observations; statistics skipped"}
    m = moments(ex)
    N = n_trials if n_trials is not None else total_atlas_trials()
    v = var_sr if var_sr is not None else (0.5 / 252)  # documented fallback: annualised SR variance 0.5
    out["moments"] = m
    out["psr_0"] = psr(m["sr"], 0.0, m["n"], m["skew"], m["kurt"])
    out["dsr"] = dsr(m["sr"], m["n"], m["skew"], m["kurt"], N, v)
    out["n_trials"] = N
    out["var_sr_per_period"] = v
    out["var_sr_source"] = "argument" if var_sr is not None else "fallback(0.5 annualised)"
    try:
        out["bootstrap"] = bootstrap_metrics(ex, reps=bootstrap_reps, seed=seed)
    except Exception as e:  # pragma: no cover
        out["bootstrap"] = {"error": str(e)}
    if cost_stress and len(getattr(result, "fills", [])):
        out["cost_stress"] = cost_stress_table(result)
        out["break_even"] = {k: break_even_cost(result, k) for k in ("cagr", "sharpe")}
    return out


def write_strategy_report(result, out_dir, benchmark: pd.Series | None = None, factors: pd.DataFrame | None = None,
                          n_trials: int | None = None, var_sr: float | None = None, regimes: dict | None = None,
                          extra: dict | None = None, make_charts: bool = True, bootstrap_reps: int = 1000) -> dict:
    sid = result.strategy_id
    d = Path(out_dir) / sid
    d.mkdir(parents=True, exist_ok=True)
    status = getattr(result.status, "value", str(result.status))
    sc = normalize_scorecard(result.fidelity, result.fills if status == "OK" else None)
    payload: dict = {"strategy_id": sid, "status": status, "params": result.params,
                     "fidelity": sc, "fidelity_grade": grade(sc, status)}
    if status != "OK":
        payload["not_testable_reason"] = getattr(result, "not_testable_reason", "")
        md = [f"# {sid} - {status}", "", f"Reason: {payload['not_testable_reason'] or 'n/a'}", "",
              render_scorecard_md(sc, status)]
        (d / "report.md").write_text("\n".join(md) + "\n")
        (d / "report.json").write_text(json.dumps(to_jsonable(payload), indent=2))
        return payload
    met = compute_metrics(result, benchmark, factors)
    val = strategy_validation(result, n_trials, var_sr, bootstrap_reps)
    payload["metrics"] = met
    payload["validation"] = val
    if regimes:
        payload["regimes"] = regimes
    if extra:
        payload["extra"] = extra
    files = {}
    if make_charts:
        eq = result.equity.astype(float)
        files["equity"] = charts.equity_chart(eq, d / "equity.png")
        files["drawdown"] = charts.drawdown_chart(eq, d / "drawdown.png")
        files["rolling_sharpe"] = charts.rolling_sharpe_chart(result.returns, d / "rolling_sharpe.png")
        files["monthly"] = charts.monthly_heatmap(result.returns, d / "monthly_returns.png")
        files["r_distribution"] = charts.r_distribution_chart(normalize_trades(result.trades)["r_multiple"],
                                                              d / "r_distribution.png")
    payload["charts"] = {k: Path(v).name for k, v in files.items() if v}
    p = met["performance"]
    t = met.get("trades", {})
    md = [f"# {sid} - backtest report", "",
          f"Params: `{json.dumps(to_jsonable(result.params))}`  ", f"Period: {p['start']} -> {p['end']} ({p['n_days']} days)",
          "", "## Headline (statistically adjusted first)", "",
          "| Metric | Value |", "|---|---:|",
          f"| Deflated Sharpe (N={val.get('n_trials', 'n/a')}) | {_fmt(val.get('dsr'))} |",
          f"| Probabilistic Sharpe vs 0 | {_fmt(val.get('psr_0'))} |",
          f"| Sharpe (rf: {p['rf_source']}) | {_fmt(p['sharpe'])} |",
          f"| Sortino | {_fmt(p['sortino'])} |",
          f"| CAGR | {_fmt(p['cagr'], 'pct')} |", f"| Ann. vol | {_fmt(p['ann_vol'], 'pct')} |",
          f"| Max drawdown | {_fmt(p['max_drawdown'], 'pct')} |", f"| Calmar | {_fmt(p['calmar'])} |",
          f"| Longest drawdown (days) | {_fmt(p.get('longest_dd_days'), 'int')} |",
          f"| Ulcer index | {_fmt(p['ulcer_index'])} |",
          f"| Trades | {_fmt(t.get('n_trades'), 'int')} |", f"| Hit rate | {_fmt(t.get('hit_rate'), 'pct')} |",
          f"| Payoff ratio | {_fmt(t.get('payoff_ratio'))} |", f"| Expectancy (R) | {_fmt(t.get('expectancy_R'))} |",
          f"| Avg holding (days) | {_fmt(t.get('avg_holding_days'))} |",
          f"| Turnover (x NAV / yr, two-sided) | {_fmt(met['turnover'].get('turnover_two_sided'))} |",
          f"| Time in market | {_fmt(met.get('exposure', {}).get('time_in_market'), 'pct')} |",
          f"| Fidelity grade | {payload['fidelity_grade']} |", ""]
    bs = val.get("bootstrap", {})
    if bs and "error" not in bs:
        md += ["## Bootstrap 95% CIs (stationary bootstrap, Politis-White block)", "", "| Metric | Point | Low | High |",
               "|---|---:|---:|---:|"]
        for k, v in bs.items():
            md.append(f"| {k} | {v['point']:.3f} | {v['lo']:.3f} | {v['hi']:.3f} |")
        md.append("")
    if "cost_stress" in val:
        cs = val["cost_stress"]
        md += ["## Cost stress (sheet scenarios A-E, extra bp per side on every fill)", "",
               "| Scen. | Extra bp/side | All-in RT bp | CAGR | Sharpe | MaxDD | Extra cost |", "|---|---:|---:|---:|---:|---:|---:|"]
        for s, row in cs.iterrows():
            md.append(f"| {s} | {row.extra_bp_per_side:g} | {row.all_in_round_trip_bp:.2f} | {row.cagr:.2%} | {row.sharpe:.2f} | "
                      f"{row.max_drawdown:.2%} | ${row.extra_cost_usd:,.0f} |")
        be = val.get("break_even", {})
        md += ["", "Break-even extra cost per side: " + ", ".join(
            f"{k}: {v['bp']:.1f} bp ({v['status']})" if np.isfinite(v["bp"]) else f"{k}: >{10_000} bp ({v['status']})"
            for k, v in be.items()), ""]
    c = met["costs"]
    md += ["## Cost drag", "", "| Component | USD | bp / yr of NAV |", "|---|---:|---:|"]
    for k, v in c["usd"].items():
        md.append(f"| {k} | {v:,.0f} | {c['bp_per_year'][k]:.1f} |")
    cap = met.get("capacity", {})
    md += ["", f"Capacity: {_fmt(cap.get('capacity_nav'), 'usd') if cap.get('capacity_nav') else 'n/a'} ({cap.get('note', '')})", ""]
    if met.get("vs_benchmark"):
        b = met["vs_benchmark"]
        md += [f"Benchmark regression: beta {b.get('beta_mkt', float('nan')):.2f} (t {b.get('t_mkt', float('nan')):.1f}), "
               f"alpha {b.get('alpha_ann', float('nan')):.2%}/yr (t {b.get('alpha_t', float('nan')):.1f}), IR {b.get('information_ratio', float('nan')):.2f}", ""]
    if met.get("factor_regression"):
        fr = met["factor_regression"]
        md += ["Factor regression: " + ", ".join(f"{k}={v:.3f}" for k, v in fr.items() if k.startswith(("beta_", "alpha"))), ""]
    tl = met.get("tails", {})
    if tl:
        md += [f"Tails: skew {tl['skew']:.2f}, excess kurtosis {tl['excess_kurtosis']:.2f}, "
               f"VaR95 {tl['var_95']:.2%}, CVaR95 {tl['cvar_95']:.2%}, CVaR99 {tl['cvar_99']:.2%}", ""]
    if regimes:
        md += ["## Regimes", ""]
        for name, tab in regimes.items():
            if isinstance(tab, pd.DataFrame) and not tab.empty:
                md += [f"### {name}", "", tab.to_markdown(floatfmt=".3f") if _has_tabulate() else tab.to_string(), ""]
    md += ["## Fidelity", "", render_scorecard_md(sc, status), ""]
    for k, f in payload["charts"].items():
        md.append(f"![{k}]({f})")
    (d / "report.md").write_text("\n".join(md) + "\n")
    (d / "report.json").write_text(json.dumps(to_jsonable(payload), indent=2))
    return payload


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401

        return True
    except Exception:
        return False
