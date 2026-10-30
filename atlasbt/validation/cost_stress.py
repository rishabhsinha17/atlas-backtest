"""Cost Stress Calculator (sheet "Cost Stress Calculator" of
docs/Trading_Strategy_Atlas_172_Systems.xlsx) applied to a backtest's fills.

Sheet definition (per scenario):
    All-in round-trip bp = 2 * ExtraBpPerSide + MeasuredQuotedSpreadBp + 2 * CommissionBpPerSide
    scenarios A..E: ExtraBpPerSide = 0, 2, 5, 10, 25 (sheet's illustrative inputs:
    quoted spread 1.5 bp, commissions 0.1 bp per side; all-in $ = notional * bp/1e4).

How atlasbt applies it (re-pricing of the trade list; exact):
  * The backtest's fills ALREADY contain the measured/modelled costs: the fill
    price includes half-spread and impact (``spread_cost``, ``impact_cost``
    are informational $ amounts vs mid) and explicit fees are separate
    (``commission``, ``exchange_fees``, ``regulatory_fees``). We MEASURE them
    from the fills (notional-weighted): measured quoted spread bp (round trip)
    = 2 * sum(spread_cost)/sum(notional)*1e4; commission bp per side =
    sum(explicit fees)/sum(notional)*1e4; impact bp per side likewise.
  * Stress = an ADDITIONAL cost of ExtraBpPerSide on every fill:
    c_i = extra_bp * 1e-4 * |qty_i| * price_i * multiplier_i. For options the
    notional is the PREMIUM notional (documented limitation: bp-of-premium
    understates option execution risk; use the options spread model instead).
  * Re-pricing of the NAV path: fill i is assigned to the first equity
    timestamp >= its ts (day t). mode='proportional' (default; sizing scales
    with NAV): r'_t = r_t - sum_{i in t} c_i / NAV_{t-1}, NAV' = NAV_0 * prod(1+r').
    mode='additive' (fixed-dollar sizing): NAV'_t = NAV_t - sum_{i: day<=t} c_i.
    Both are first-order: fills, sizes and signals are NOT re-simulated (a
    strategy whose decisions depend on its own NAV may diverge slightly).
  * Trade list: pnl' = pnl - extra*(entry notional + exit notional);
    R' = pnl' / (pnl / R).
Break-even cost: the extra bp per side at which a metric (CAGR, Sharpe,
total return) reaches a target (default 0), solved with Brent's method.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from atlasbt.reporting.schema import normalize_fills, normalize_trades
from .bootstrap import max_drawdown as _mdd
from .bootstrap import sharpe_annual

SCENARIOS = {"A": 0.0, "B": 2.0, "C": 5.0, "D": 10.0, "E": 25.0}
SHEET_ILLUSTRATIVE = {"measured_spread_bp": 1.5, "commission_bp_per_side": 0.1, "example_notional": 100_000.0}


def all_in_round_trip_bp(extra_bp_per_side: float, spread_bp: float, commission_bp_per_side: float) -> float:
    """Sheet formula E = 2*B + C + 2*D."""
    return 2.0 * extra_bp_per_side + spread_bp + 2.0 * commission_bp_per_side


def measured_costs_bp(fills) -> dict:
    f = normalize_fills(fills)
    notional = float(f["notional"].sum())
    if notional <= 0:
        return {"notional": 0.0, "spread_bp_round_trip": np.nan, "commission_bp_per_side": np.nan,
                "impact_bp_per_side": np.nan, "n_fills": 0}
    explicit = float((f["commission"] + f["exchange_fees"] + f["regulatory_fees"]).sum())
    return {"notional": notional, "n_fills": int(len(f)),
            "spread_bp_round_trip": 2.0 * float(f["spread_cost"].sum()) / notional * 1e4,
            "commission_bp_per_side": explicit / notional * 1e4,
            "impact_bp_per_side": float(f["impact_cost"].sum()) / notional * 1e4}


def _base_series(result):
    eq = result.equity.astype(float).dropna()
    r = result.returns.astype(float).reindex(eq.index) if result.returns is not None and len(result.returns) else None
    if r is None or r.isna().all():
        r = eq.pct_change()
    return eq, r.fillna(0.0)


def _daily_costs(eq_index: pd.DatetimeIndex, fills: pd.DataFrame, bp: float) -> np.ndarray:
    c = np.zeros(len(eq_index))
    if bp == 0 or fills.empty:
        return c
    cost = bp * 1e-4 * fills["notional"].to_numpy(float)
    ts = pd.DatetimeIndex(fills["ts"])
    if eq_index.tz is None and ts.tz is not None:
        ts = ts.tz_convert(None)
    pos = np.searchsorted(pd.DatetimeIndex(eq_index).as_unit("ns").asi8, ts.as_unit("ns").asi8, side="left")
    pos = np.clip(pos, 0, len(eq_index) - 1)
    np.add.at(c, pos, cost)
    return c


@dataclass
class StressedResult:
    extra_bp_per_side: float
    equity: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    extra_cost_total: float
    mode: str


def apply_cost_stress(result, extra_bp_per_side: float, mode: str = "proportional") -> StressedResult:
    """Re-price a BacktestResult with an extra cost per side on every fill."""
    eq, r = _base_series(result)
    f = normalize_fills(result.fills)
    c = _daily_costs(eq.index, f, extra_bp_per_side)
    meta = getattr(result, "meta", None) or {}
    nav0 = float(meta.get("initial_capital", eq.iloc[0] / (1.0 + r.iloc[0])))
    if mode == "proportional":
        prev = np.concatenate([[nav0], eq.to_numpy()[:-1]])
        r2 = r.to_numpy() - c / prev
        eq2 = pd.Series(nav0 * np.cumprod(1 + r2), index=eq.index)
        ret2 = pd.Series(r2, index=eq.index)
    elif mode == "additive":
        eq2 = eq - np.cumsum(c)
        prev2 = np.concatenate([[nav0], eq2.to_numpy()[:-1]])
        ret2 = pd.Series(eq2.to_numpy() / prev2 - 1.0, index=eq.index)
    else:
        raise ValueError("mode must be proportional|additive")
    t0 = normalize_trades(result.trades)
    t = t0.copy()
    if not t.empty and extra_bp_per_side:
        extra = extra_bp_per_side * 1e-4 * t["quantity"].abs() * (t["entry_price"].abs() + t["exit_price"].abs()) * t["multiplier"]
        t["pnl"] = t["pnl"] - extra.fillna(0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            risk = t0["pnl"] / t0["r_multiple"]
        t["r_multiple"] = np.where(np.isfinite(risk) & (risk != 0), t["pnl"] / risk, t["r_multiple"])
    return StressedResult(extra_bp_per_side, eq2, ret2, t, float(c.sum()), mode)


def _metrics(eq: pd.Series, r: pd.Series, periods: int = 252) -> dict:
    a = r.to_numpy(float)
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9) if isinstance(eq.index, pd.DatetimeIndex) else len(a) / periods
    nav0 = eq.iloc[0] / (1 + a[0])
    tot = float(eq.iloc[-1] / nav0 - 1)
    return {"total_return": tot, "cagr": float((1 + tot) ** (1 / years) - 1) if tot > -1 else -1.0,
            "sharpe": float(sharpe_annual(a, periods)), "max_drawdown": float(_mdd(a))}


def cost_stress_table(result, scenarios: dict | None = None, mode: str = "proportional") -> pd.DataFrame:
    """One row per scenario with the sheet's columns plus re-priced metrics."""
    sc = scenarios or SCENARIOS
    mc = measured_costs_bp(result.fills)
    rows = []
    for name, bp in sc.items():
        s = apply_cost_stress(result, bp, mode)
        m = _metrics(s.equity, s.returns)
        rows.append({"scenario": name, "extra_bp_per_side": bp,
                     "measured_spread_bp": mc["spread_bp_round_trip"],
                     "commission_bp_per_side": mc["commission_bp_per_side"],
                     "impact_bp_per_side": mc["impact_bp_per_side"],
                     "all_in_round_trip_bp": all_in_round_trip_bp(bp, mc["spread_bp_round_trip"], mc["commission_bp_per_side"]),
                     "all_in_round_trip_bp_incl_impact": all_in_round_trip_bp(bp, mc["spread_bp_round_trip"], mc["commission_bp_per_side"]) + 2 * mc["impact_bp_per_side"],
                     "extra_cost_usd": s.extra_cost_total, **m})
    return pd.DataFrame(rows).set_index("scenario")


def break_even_cost(result, metric: str = "cagr", target: float = 0.0, hi: float = 10_000.0,
                    mode: str = "proportional") -> dict:
    """Extra bp per side at which ``metric`` falls to ``target``.

    Returns {'bp': value, 'status': 'ok'|'already_below'|'never_within_hi'|'no_trades'}."""
    f = normalize_fills(result.fills)
    if f.empty or f["notional"].sum() <= 0:
        return {"bp": float("inf"), "status": "no_trades", "metric": metric}

    def g(bp):
        s = apply_cost_stress(result, bp, mode)
        return _metrics(s.equity, s.returns)[metric] - target

    g0 = g(0.0)
    if not np.isfinite(g0) or g0 <= 0:
        return {"bp": 0.0, "status": "already_below", "metric": metric, "base": g0 + target}
    ghi = g(hi)
    if ghi > 0:
        return {"bp": float("inf"), "status": "never_within_hi", "metric": metric, "base": g0 + target}
    bp = brentq(g, 0.0, hi, xtol=1e-6, rtol=1e-10)
    return {"bp": float(bp), "status": "ok", "metric": metric, "base": g0 + target}
