"""Performance metrics computed from a ``core.types.BacktestResult``.

Conventions (docs/ANALYTICS.md "Metrics"):
* Returns are the result's daily simple NAV returns. Annualisation uses
  252 periods/year for volatility and Sharpe; CAGR uses calendar time
  (365.25-day years between the first NAV's prior close and the last NAV).
* Risk-free: per-period simple rf returns taken, in order of preference, from
  the ``rf`` argument, ``result.meta['rf_daily']`` (per-period simple return
  aligned to ``returns``), or ``result.meta['risk_free_rate']`` (annualised
  simple rate, converted with actual/360 day counts between NAV stamps - the
  same convention the portfolio uses for cash interest). If none is present,
  rf = 0 and ``rf_source='none'`` is reported.
* Sharpe = mean(r - rf)/std(r - rf, ddof=1)*sqrt(252). Sortino = mean(r - rf) /
  sqrt(mean(min(r - rf, 0)^2))*sqrt(252) (target = rf, Sortino & Price 1994).
* Max drawdown on the NAV path including the initial capital; duration = the
  longest peak-to-recovery span (calendar days and bars; an unrecovered
  drawdown runs to the last observation and is flagged).
* Ulcer index = sqrt(mean(DD_pct^2)) with DD in percent (Martin 1987).
* Calmar = CAGR / |MaxDD|.
* Trades: hit rate = share of trades with pnl > 0; payoff = mean win / |mean
  loss|; expectancy_R = mean R multiple; holding period from entry/exit stamps.
* Turnover = sum |fill notional| / mean NAV / years (two-sided, "traded
  notional per year as a multiple of capital"); one-sided = half of it.
* Exposure: share of days with any open position; mean gross / net exposure
  as fraction of NAV from the positions snapshot.
* Cost drag: each component (commission, exchange & regulatory fees,
  spread, impact from fills; borrow, financing debit/credit, dividends from
  the ledger) in $ and in annualised bp of mean NAV.
* Capacity: participation_i = |qty_i| / ADV_i (shares; from the fills' ``adv``
  column). With p95 participation pi and a limit L (default 5% of ADV), the
  capacity estimate is mean NAV * L / pi (linear scaling, ignores the
  growth of sqrt impact - an optimistic upper bound, flagged as such).
* Tails: skew (bias-corrected), excess kurtosis, historical VaR/CVaR 95/99.
* Beta/alpha vs SPY: OLS of (r - rf) on (r_spy - rf) with Newey-West (HAC)
  t-stats; Fama-French: OLS of (r - rf) on the supplied factor columns.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .schema import classify_reason, normalize_fills, normalize_ledger, normalize_positions, normalize_trades

PERIODS = 252


# ---------------------------------------------------------------- basic series metrics
def drawdown_series(equity: pd.Series) -> pd.Series:
    eq = equity.astype(float)
    return eq / eq.cummax() - 1.0


def max_drawdown_info(equity: pd.Series) -> dict:
    """Max drawdown, its peak/trough/recovery, and the longest underwater spell
    (peak bar -> first bar back at/above that peak, or the last bar if unrecovered)."""
    eq = equity.astype(float).dropna()
    if eq.empty:
        return {"max_drawdown": np.nan}
    peak_val = eq.cummax()
    dd = eq / peak_val - 1.0
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    rec = eq.loc[trough:]
    rec = rec[rec >= eq.loc[peak]]
    recovery = rec.index[0] if len(rec) else None
    a = eq.to_numpy()
    at_peak = np.flatnonzero(a >= peak_val.to_numpy())
    best = (0, 0, 0, True)
    for p0, p1 in zip(at_peak[:-1], at_peak[1:]):
        if p1 - p0 > best[0]:
            best = (p1 - p0, p0, p1, True)
    if at_peak[-1] != len(a) - 1 and (len(a) - 1 - at_peak[-1]) > best[0]:
        best = (len(a) - 1 - at_peak[-1], at_peak[-1], len(a) - 1, False)
    out = {"max_drawdown": float(dd.min()), "peak": peak, "trough": trough, "recovery": recovery,
           "recovered": recovery is not None or float(dd.min()) == 0.0, "longest_dd_bars": int(best[0]),
           "longest_dd_recovered": best[3]}
    if isinstance(eq.index, pd.DatetimeIndex):
        out["longest_dd_days"] = int((eq.index[best[2]] - eq.index[best[1]]).days)
    return out


def ulcer_index(equity: pd.Series) -> float:
    dd = drawdown_series(equity) * 100.0
    return float(np.sqrt(np.mean(dd.to_numpy() ** 2)))


def cagr_from_equity(equity: pd.Series, nav0: float | None = None) -> float:
    eq = equity.astype(float).dropna()
    start = nav0 if nav0 is not None else eq.iloc[0]
    if isinstance(eq.index, pd.DatetimeIndex):
        days = (eq.index[-1] - eq.index[0]).days + (1 if nav0 is not None else 0)
        years = max(days, 1) / 365.25
    else:
        years = len(eq) / PERIODS
    g = eq.iloc[-1] / start
    return float(g ** (1 / years) - 1) if g > 0 else -1.0


def sharpe(excess: np.ndarray, periods: int = PERIODS) -> float:
    x = np.asarray(excess, float)
    x = x[~np.isnan(x)]
    sd = x.std(ddof=1) if x.size > 1 else np.nan
    return float(x.mean() / sd * math.sqrt(periods)) if sd and sd > 0 else float("nan")


def sortino(excess: np.ndarray, periods: int = PERIODS) -> float:
    x = np.asarray(excess, float)
    x = x[~np.isnan(x)]
    dd = math.sqrt(np.mean(np.minimum(x, 0.0) ** 2)) if x.size else np.nan
    return float(x.mean() / dd * math.sqrt(periods)) if dd and dd > 0 else float("nan")


def tail_stats(r: np.ndarray) -> dict:
    x = np.asarray(r, float)
    x = x[~np.isnan(x)]
    if x.size < 4:
        return {}
    out = {"skew": float(stats.skew(x, bias=False)), "excess_kurtosis": float(stats.kurtosis(x, bias=False))}
    for a in (0.95, 0.99):
        q = np.quantile(x, 1 - a)
        out[f"var_{int(a * 100)}"] = float(-q)
        out[f"cvar_{int(a * 100)}"] = float(-x[x <= q].mean())
    out["worst_day"] = float(x.min())
    out["best_day"] = float(x.max())
    return out


# ---------------------------------------------------------------- rf
def risk_free_returns(result, rf=None, with_coverage: bool = False):
    """Per-period simple rf returns aligned to ``result.returns``.

    Returns (series, source) or, with ``with_coverage=True``, (series, source,
    coverage) where coverage is the share of periods with an rf value. Sources:
    'argument', 'meta.rf_daily', 'meta.risk_free_rate' (Series or float,
    annualised simple, accrued act/360 using the rate known at the PREVIOUS
    stamp; the first stamp uses its own rate - bfill(limit=1) for that stamp
    only), 'none'. Periods before the first available rate get 0 and are
    reported through coverage < 1 (never back-filled from later data).
    """
    idx = result.returns.index
    cov = 1.0
    if rf is not None:
        s = rf if isinstance(rf, pd.Series) else pd.Series(rf, index=idx)
        s = s.reindex(idx)
        cov, out, src = float(s.notna().mean()) if len(s) else 1.0, s.fillna(0.0), "argument"
    else:
        meta = getattr(result, "meta", None) or {}
        rate_obj = meta.get("risk_free_rate")
        if isinstance(meta.get("rf_daily"), pd.Series):
            s = meta["rf_daily"].reindex(idx)
            cov, out, src = float(s.notna().mean()) if len(s) else 1.0, s.fillna(0.0), "meta.rf_daily"
        elif isinstance(rate_obj, (pd.Series, int, float)) and not isinstance(rate_obj, bool):
            if isinstance(rate_obj, pd.Series):
                rate = rate_obj.reindex(idx, method="ffill") if isinstance(idx, pd.DatetimeIndex) else rate_obj.reindex(idx)
            else:
                rate = pd.Series(float(rate_obj), index=idx)
            if isinstance(idx, pd.DatetimeIndex):
                days = pd.Series(idx, index=idx).diff().dt.days.fillna(1).to_numpy()
            else:
                days = np.ones(len(idx))
            known = rate.shift(1)
            if len(known):
                known.iloc[0] = rate.iloc[0]  # first stamp only: its own rate (no earlier stamp exists)
            cov = float(known.notna().mean()) if len(known) else 1.0
            out = pd.Series(known.to_numpy() * days / 360.0, index=idx).fillna(0.0)
            src = "meta.risk_free_rate"
        else:
            out, src = pd.Series(0.0, index=idx), "none"
    return (out, src, cov) if with_coverage else (out, src)


def nav_path(result) -> dict:
    """Shared NAV conventions: returns {'eq', 'r', 'nav0', 'full_eq'} where
    full_eq prepends NAV0 (meta['initial_capital'], else equity[0]/(1+r[0]))
    one day before the first stamp. Used by compute_metrics and the leaderboard."""
    eq = result.equity.astype(float).dropna()
    r = (result.returns.astype(float).reindex(eq.index).fillna(0.0) if result.returns is not None
         and len(result.returns) else eq.pct_change().fillna(0.0))
    nav0 = float((getattr(result, "meta", None) or {}).get("initial_capital", eq.iloc[0] / (1 + r.iloc[0])))
    first = [eq.index[0] - pd.Timedelta(days=1)] if isinstance(eq.index, pd.DatetimeIndex) else [-1]
    full_eq = pd.concat([pd.Series([nav0], index=first), eq])
    return {"eq": eq, "r": r, "nav0": nav0, "full_eq": full_eq}


# ---------------------------------------------------------------- regression vs benchmark / factors
def _ols_hac(y: np.ndarray, X: np.ndarray, names: list[str], periods: int = PERIODS) -> dict:
    import statsmodels.api as sm

    Xc = sm.add_constant(X, has_constant="add")
    lags = int(np.floor(4 * (len(y) / 100) ** (2 / 9)))
    fit = sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": max(lags, 1)})
    out = {"alpha_ann": float(fit.params[0] * periods), "alpha_t": float(fit.tvalues[0]), "r2": float(fit.rsquared),
           "n": int(len(y))}
    for i, n in enumerate(names, start=1):
        out[f"beta_{n}"] = float(fit.params[i])
        out[f"t_{n}"] = float(fit.tvalues[i])
    return out


def benchmark_regression(r: pd.Series, rf: pd.Series, bench: pd.Series) -> dict:
    df = pd.concat([r.rename("r"), rf.rename("rf"), bench.rename("b")], axis=1).dropna()
    if len(df) < 30:
        return {}
    y = (df["r"] - df["rf"]).to_numpy()
    x = (df["b"] - df["rf"]).to_numpy()[:, None]
    out = _ols_hac(y, x, ["mkt"])
    te = (df["r"] - df["b"]).std(ddof=1) * math.sqrt(PERIODS)
    out["tracking_error"] = float(te)
    out["information_ratio"] = float((df["r"] - df["b"]).mean() * PERIODS / te) if te > 0 else np.nan
    out["correlation"] = float(np.corrcoef(df["r"], df["b"])[0, 1])
    return out


def factor_regression(r: pd.Series, rf: pd.Series, factors: pd.DataFrame) -> dict:
    """factors: daily decimal returns, e.g. columns Mkt-RF, SMB, HML, RMW, CMA, MOM
    (if an 'RF' column exists it is used as rf)."""
    f = factors.copy()
    if "RF" in f.columns:
        rf = f.pop("RF")
    df = pd.concat([r.rename("__r"), rf.rename("__rf"), f], axis=1).dropna()
    if len(df) < 30:
        return {}
    y = (df["__r"] - df["__rf"]).to_numpy()
    cols = [c for c in f.columns]
    return _ols_hac(y, df[cols].to_numpy(), [str(c) for c in cols])


# ---------------------------------------------------------------- trades / turnover / exposure / costs
def trade_stats(trades) -> dict:
    t = normalize_trades(trades)
    if t.empty:
        return {"n_trades": 0}
    pnl = t["pnl"].dropna()
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    out = {"n_trades": int(len(t)), "hit_rate": float((pnl > 0).mean()) if len(pnl) else np.nan,
           "avg_win": float(wins.mean()) if len(wins) else np.nan,
           "avg_loss": float(losses.mean()) if len(losses) else np.nan,
           "payoff_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else np.nan,
           "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else np.nan,
           "expectancy_usd": float(pnl.mean()) if len(pnl) else np.nan,
           "total_pnl": float(pnl.sum())}
    r = t["r_multiple"].dropna()
    out["expectancy_R"] = float(r.mean()) if len(r) else np.nan
    out["median_R"] = float(r.median()) if len(r) else np.nan
    if t["entry_ts"].notna().any() and t["exit_ts"].notna().any():
        hold = (t["exit_ts"] - t["entry_ts"]).dt.total_seconds() / 86400.0
        out["avg_holding_days"] = float(hold.mean())
        out["median_holding_days"] = float(hold.median())
    if t["bars_held"].notna().any():
        out["avg_holding_bars"] = float(t["bars_held"].mean())
    if t["quantity"].notna().any():
        out["long_share"] = float((t["quantity"] > 0).mean())
    return out


def turnover_stats(fills, equity: pd.Series) -> dict:
    f = normalize_fills(fills)
    if f.empty or equity.empty:
        return {"turnover_two_sided": 0.0, "turnover_one_sided": 0.0, "traded_notional": 0.0}
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 252) if isinstance(equity.index, pd.DatetimeIndex) else len(equity) / PERIODS
    tn = float(f["notional"].sum())
    two = tn / float(equity.mean()) / years
    return {"traded_notional": tn, "turnover_two_sided": two, "turnover_one_sided": two / 2.0, "n_fills": int(len(f))}


def _day_key(idx) -> pd.DatetimeIndex:
    d = pd.DatetimeIndex(idx)
    if d.tz is not None:
        d = d.tz_convert("America/New_York").tz_localize(None)
    return d.normalize()


def exposure_stats(positions, equity: pd.Series) -> dict:
    """Time in market and mean gross/net exposure (fraction of NAV), averaged
    over ALL NAV days (flat days count as zero)."""
    p = normalize_positions(positions)
    if p.empty:
        return {}
    p = p[p["quantity"].fillna(0) != 0]
    nav = pd.Series(equity.to_numpy(float), index=_day_key(equity.index))
    if p.empty:
        return {"time_in_market": 0.0, "mean_gross_exposure": 0.0, "mean_net_exposure": 0.0}
    p = p.assign(day=_day_key(p["ts"]))
    g = p.groupby("day")["market_value"]
    gross = g.apply(lambda v: np.abs(v).sum()).reindex(nav.index, fill_value=0.0)
    net = g.sum().reindex(nav.index, fill_value=0.0)
    held = p.groupby("day").size().reindex(nav.index, fill_value=0) > 0
    return {"time_in_market": float(held.mean()), "mean_gross_exposure": float((gross / nav).mean()),
            "mean_net_exposure": float((net / nav).mean()), "max_gross_exposure": float((gross / nav).max())}


def cost_drag(fills, ledger, equity: pd.Series) -> dict:
    f = normalize_fills(fills)
    led = normalize_ledger(ledger)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 252) if isinstance(equity.index, pd.DatetimeIndex) and len(equity) > 1 else max(len(equity), 1) / PERIODS
    nav = float(equity.mean()) if len(equity) else np.nan
    comps = {}
    for c in ("commission", "exchange_fees", "regulatory_fees", "spread_cost", "impact_cost"):
        comps[c] = float(f[c].sum()) if not f.empty else 0.0
    # fees_other is kept for a stable output schema; no LEDGER_REASONS code maps to it
    # today (per-fill fees come from the fills, never from the ledger)
    for b in ("borrow", "financing_debit", "financing_credit", "dividends", "fees_other"):
        comps[b] = 0.0
    if not led.empty:
        led = led.assign(bucket=led["reason"].map(classify_reason))
        for b, v in led.groupby("bucket")["amount"].sum().items():
            if b in comps:
                # costs are positive numbers; credits (interest income, dividends received) stay signed as income
                comps[b] = float(-v) if b in ("borrow", "financing_debit", "fees_other") else float(v)
    out = {"usd": comps}
    out["bp_per_year"] = {k: (v / nav / years * 1e4 if nav and nav > 0 else np.nan) for k, v in comps.items()}
    explicit = comps["commission"] + comps["exchange_fees"] + comps["regulatory_fees"]
    implicit = comps["spread_cost"] + comps["impact_cost"]
    out["total_trading_cost_usd"] = explicit + implicit
    out["total_carry_cost_usd"] = comps["borrow"] + comps["financing_debit"] - comps["financing_credit"]
    return out


def capacity_estimate(fills, equity: pd.Series, limit: float = 0.05, quantile: float = 0.95) -> dict:
    f = normalize_fills(fills)
    if f.empty or f["adv"].isna().all():
        return {"capacity_nav": None, "note": "no ADV on fills; capacity not estimated"}
    part = (f["quantity"].abs() / f["adv"].astype(float)).replace([np.inf], np.nan).dropna()
    if part.empty:
        return {"capacity_nav": None, "note": "ADV missing/zero"}
    pq = float(part.quantile(quantile))
    nav = float(equity.mean())
    cap = nav * limit / pq if pq > 0 else float("inf")
    return {"participation_p50": float(part.median()), f"participation_p{int(quantile * 100)}": pq,
            "participation_max": float(part.max()), "limit": limit, "capacity_nav": cap,
            "note": "linear scaling to the participation limit; ignores sqrt-impact growth (optimistic)"}


# ---------------------------------------------------------------- top level
def compute_metrics(result, benchmark: pd.Series | None = None, factors: pd.DataFrame | None = None, rf=None,
                    capacity_limit: float = 0.05) -> dict:
    """All metrics for one BacktestResult as a JSON-friendly dict."""
    out: dict = {"strategy_id": result.strategy_id, "params": result.params,
                 "status": getattr(result.status, "value", str(result.status))}
    if out["status"] != "OK" or result.equity is None or len(result.equity) < 2:
        out["not_testable_reason"] = getattr(result, "not_testable_reason", "")
        return out
    nv = nav_path(result)
    eq, r, nav0, full_eq = nv["eq"], nv["r"], nv["nav0"], nv["full_eq"]
    rfs, src, cov = risk_free_returns(result, rf, with_coverage=True)
    if src != "none" and cov < 1.0:
        src = f"{src} (partial coverage {cov:.0%}; rf=0 where missing)"
    rfs = rfs.reindex(r.index).fillna(0.0)
    ex = (r - rfs).to_numpy()
    dd = max_drawdown_info(full_eq)
    cg = cagr_from_equity(eq, nav0)
    perf = {"start": str(eq.index[0]), "end": str(eq.index[-1]), "n_days": int(len(r)), "nav0": nav0,
            "final_nav": float(eq.iloc[-1]), "total_return": float(eq.iloc[-1] / nav0 - 1), "cagr": cg,
            "ann_vol": float(np.std(r.to_numpy(), ddof=1) * math.sqrt(PERIODS)),
            "sharpe": sharpe(ex), "sortino": sortino(ex), "rf_source": src,
            "rf_ann_mean": float(rfs.mean() * PERIODS),
            "max_drawdown": dd.get("max_drawdown"), "calmar": cg / abs(dd["max_drawdown"]) if dd.get("max_drawdown") else np.nan,
            "max_dd_peak": str(dd.get("peak")), "max_dd_trough": str(dd.get("trough")),
            "max_dd_recovered": dd.get("recovered"), "longest_dd_days": dd.get("longest_dd_days"),
            "longest_dd_bars": dd.get("longest_dd_bars"), "ulcer_index": ulcer_index(full_eq)}
    from atlasbt.validation.sharpe_tests import moments, psr

    m = moments(ex)
    perf["psr_0"] = psr(m["sr"], 0.0, m["n"], m["skew"], m["kurt"])
    out["performance"] = perf
    out["tails"] = tail_stats(r.to_numpy())
    out["trades"] = trade_stats(result.trades)
    out["turnover"] = turnover_stats(result.fills, eq)
    out["exposure"] = exposure_stats(result.positions, eq)
    out["costs"] = cost_drag(result.fills, result.ledger, eq)
    out["capacity"] = capacity_estimate(result.fills, eq, capacity_limit)
    if benchmark is not None:
        b = benchmark.reindex(r.index)
        out["vs_benchmark"] = benchmark_regression(r, rfs, b)
    if factors is not None:
        out["factor_regression"] = factor_regression(r, rfs, factors.reindex(r.index))
    return out
