"""Metric known answers, schema normalisation, fidelity scorecard, charts,
per-strategy report and the statistically-adjusted leaderboard."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.types import (AssetClass, BacktestResult, CostBreakdown, Fidelity, Fill, Instrument, RunStatus)
from atlasbt.reporting import charts, fidelity, leaderboard, metrics, report, schema


def make_result(returns, sid="SYN", nav0=100_000.0, start="2020-01-02", **kw):
    r = np.asarray(returns, float)
    idx = pd.bdate_range(start, periods=len(r), tz="UTC") + pd.Timedelta(hours=21)
    eq = pd.Series(nav0 * np.cumprod(1 + r), index=idx)
    base = dict(strategy_id=sid, params={"p": 1}, status=RunStatus.OK, equity=eq, returns=pd.Series(r, index=idx),
                fills=pd.DataFrame(), trades=pd.DataFrame(), positions=pd.DataFrame(), ledger=pd.DataFrame(),
                fidelity={}, meta={"initial_capital": nav0})
    base.update(kw)
    return BacktestResult(**base)


# ---------------------------------------------------------------- performance metrics
def test_drawdown_duration_ulcer_cagr():
    eq = pd.Series([100, 100, 110, 99, 120.0], index=pd.bdate_range("2021-01-04", periods=5))
    dd = metrics.max_drawdown_info(eq)
    assert dd["max_drawdown"] == pytest.approx(-0.1)
    assert dd["longest_dd_bars"] == 2 and dd["recovered"]
    assert metrics.ulcer_index(eq) == pytest.approx(math.sqrt(100 / 5))
    eq2 = pd.Series([100, 90, 95.0], index=pd.bdate_range("2021-01-04", periods=3))
    d2 = metrics.max_drawdown_info(eq2)
    assert d2["longest_dd_bars"] == 2 and not d2["longest_dd_recovered"]
    one_year = pd.Series([100.0, 121.0], index=pd.DatetimeIndex(["2020-01-01", "2021-12-31"]))
    assert metrics.cagr_from_equity(one_year) == pytest.approx(1.21 ** (365.25 / 730) - 1)


def test_sharpe_sortino_with_rf_sources():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 500)
    res = make_result(r)
    rf = pd.Series(0.0001, index=res.returns.index)
    res.meta["rf_daily"] = rf
    m = metrics.compute_metrics(res)
    ex = r - 0.0001
    assert m["performance"]["sharpe"] == pytest.approx(ex.mean() / ex.std(ddof=1) * math.sqrt(252))
    assert m["performance"]["sortino"] == pytest.approx(ex.mean() / math.sqrt(np.mean(np.minimum(ex, 0) ** 2)) * math.sqrt(252))
    assert m["performance"]["rf_source"] == "meta.rf_daily"
    res2 = make_result(r, meta={"initial_capital": 1e5, "risk_free_rate": pd.Series(0.036, index=res.returns.index)})
    rfs, src = metrics.risk_free_returns(res2)
    assert src == "meta.risk_free_rate"
    # act/360: 1 day -> 0.0001; Monday after a weekend (3 days) -> 0.0003
    mon = [i for i, t in enumerate(rfs.index) if t.dayofweek == 0 and i > 0][0]
    assert rfs.iloc[mon] == pytest.approx(0.0003) and rfs.iloc[mon + 1] == pytest.approx(0.0001)
    assert metrics.risk_free_returns(make_result(r))[1] == "none"


def test_trade_turnover_cost_exposure_capacity():
    res = make_result(np.zeros(4))
    days = res.equity.index
    res.trades = pd.DataFrame({"symbol": ["A"] * 4, "entry_ts": days[[0, 0, 1, 2]], "exit_ts": days[[1, 2, 3, 3]],
                               "quantity": [1, 1, -1, 1], "entry_price": 1.0, "exit_price": 1.0,
                               "pnl": [100, -50, 200, -50.0], "R": [1, -0.5, 2, -0.5]})
    t = metrics.trade_stats(res.trades)
    assert t["hit_rate"] == 0.5 and t["payoff_ratio"] == 3.0 and t["expectancy_R"] == 0.5
    assert t["profit_factor"] == 3.0 and t["avg_holding_days"] == pytest.approx((1 + 4 + 4 + 1) / 4)  # calendar days
    res.fills = pd.DataFrame({"ts": days[:2], "symbol": "A", "quantity": [500, -500], "price": [100.0, 100.0],
                              "commission": [1.0, 1.0], "spread_cost": [2.5, 2.5], "adv": [10_000, 50_000]})
    tv = metrics.turnover_stats(res.fills, res.equity)
    years = (days[-1] - days[0]).days / 365.25
    assert tv["turnover_two_sided"] == pytest.approx(100_000 / 100_000 / years)
    res.ledger = pd.DataFrame({"ts": days[[1, 2, 3, 3]], "amount": [-10, -5, 3, 7.0],
                               "reason": ["BORROW_FEE", "MARGIN_INTEREST", "CASH_INTEREST", "DIVIDEND"]})
    c = metrics.cost_drag(res.fills, res.ledger, res.equity)
    u = c["usd"]
    assert (u["commission"], u["spread_cost"], u["borrow"], u["financing_debit"], u["financing_credit"],
            u["dividends"]) == (2.0, 5.0, 10.0, 5.0, 3.0, 7.0)
    assert c["total_carry_cost_usd"] == 12.0
    res.positions = pd.DataFrame({"ts": days[[0, 1]], "symbol": "A", "quantity": [500, 500],
                                  "market_value": [50_000.0, 50_000.0]})
    e = metrics.exposure_stats(res.positions, res.equity)
    assert e["time_in_market"] == 0.5 and e["mean_gross_exposure"] == pytest.approx(0.25)
    cap = metrics.capacity_estimate(res.fills, res.equity, limit=0.05, quantile=0.95)
    p95 = np.quantile([0.05, 0.01], 0.95)
    assert cap["capacity_nav"] == pytest.approx(100_000 * 0.05 / p95)


def test_benchmark_and_factor_regression_and_tails():
    rng = np.random.default_rng(1)
    b = rng.normal(0.0004, 0.01, 1000)
    f2 = rng.normal(0, 0.005, 1000)
    r = 0.0002 + 0.5 * b + 0.3 * f2 + rng.normal(0, 0.002, 1000)
    res = make_result(r)
    bench = pd.Series(b, index=res.returns.index)
    fac = pd.DataFrame({"Mkt-RF": b, "SMB": f2, "RF": 0.0}, index=res.returns.index)
    m = metrics.compute_metrics(res, benchmark=bench, factors=fac)
    assert m["vs_benchmark"]["beta_mkt"] == pytest.approx(0.5, abs=0.03)
    assert m["factor_regression"]["beta_SMB"] == pytest.approx(0.3, abs=0.03)
    assert m["factor_regression"]["alpha_ann"] == pytest.approx(0.0002 * 252, abs=0.02)
    ts = metrics.tail_stats(np.array([-0.05, 0.01, 0.02, 0.0, 0.01] * 20))
    assert ts["var_95"] == pytest.approx(0.05) and ts["cvar_95"] == pytest.approx(0.05)


# ---------------------------------------------------------------- schema
def test_normalize_fills_from_core_fill_objects():
    inst = Instrument.option("SPY", pd.Timestamp("2024-01-19"), 470, __import__("atlasbt.core.types", fromlist=["Right"]).Right.CALL)
    f = Fill(order_id=1, instrument=inst, ts=pd.Timestamp("2024-01-02 15:00", tz="UTC"), quantity=-2, price=3.5,
             costs=CostBreakdown(commission=1.3, exchange_fees=0.1, spread_cost=5.0), fidelity=Fidelity.T3_MODELED)
    n = schema.normalize_fills([f])
    row = n.iloc[0]
    assert row.symbol == inst.symbol and row.multiplier == 100 and row.notional == pytest.approx(700)
    assert row.commission == 1.3 and row.spread_cost == 5.0 and row.fidelity == 3 and row.asset_class == "OPT"
    df = pd.DataFrame({"instrument": [Instrument("SPY", AssetClass.ETF)], "ts": [pd.Timestamp("2024-01-02", tz="UTC")],
                       "qty": [10], "price": [400.0], "costs": [CostBreakdown(regulatory_fees=0.02)], "fidelity": ["T1_BAR"]})
    n2 = schema.normalize_fills(df)
    assert n2.iloc[0].regulatory_fees == 0.02 and n2.iloc[0].fidelity == 1 and n2.iloc[0].notional == 4000


# ---------------------------------------------------------------- fidelity
def test_fidelity_scorecard_shapes_and_grade():
    sc = fidelity.normalize_scorecard({"tiers": {"T1_BAR": {"fills": 8, "notional": 8e5},
                                                 "T2_PROXY": {"fills": 2, "notional": 2e5}},
                                       "reasons": {"spread estimated (Corwin-Schultz)": 10}})
    assert fidelity.grade(sc) == "C"  # mean tier 1.2 -> C ... (B <= 1.0)
    sc_b = fidelity.normalize_scorecard({1: {"fills": 5, "notional": 1e5}, "reasons": ["borrow fee default 0.30%"]})
    assert fidelity.grade(sc_b) == "B" and sc_b["reasons"][0]["reason"].startswith("borrow")
    assert fidelity.grade(fidelity.normalize_scorecard({"T0_QUOTE": {"fills": 3, "notional": 1e4}})) == "A"
    assert fidelity.grade(fidelity.normalize_scorecard({"NA": {"fills": 1, "notional": 1.0}})) == "NA"
    assert fidelity.grade({"tiers": {}}, "NOT_TESTABLE") == "NA"
    fills = pd.DataFrame({"ts": [pd.Timestamp("2024-01-02", tz="UTC")] * 2, "symbol": "X", "quantity": [1, 1],
                          "price": [10.0, 10.0], "fidelity": [Fidelity.T3_MODELED, 3]})
    sc3 = fidelity.normalize_scorecard({}, fills)
    assert sc3["tiers"]["T3_MODELED"]["fills"] == 2 and fidelity.grade(sc3) == "E"
    md = fidelity.render_scorecard_md(sc)
    assert "| T1_BAR | 8 | 80.0% |" in md and "Corwin-Schultz" in md


# ---------------------------------------------------------------- charts & reports
def test_charts_and_strategy_report(tmp_path):
    rng = np.random.default_rng(2)
    r = rng.normal(0.0004, 0.01, 400)
    res = make_result(r, sid="S999")
    days = res.equity.index
    res.fills = pd.DataFrame({"ts": days[::20], "symbol": "SPY", "quantity": np.tile([100, -100], 10),
                              "price": 100.0, "commission": 0.5, "spread_cost": 1.0, "fidelity": 1})
    res.trades = pd.DataFrame({"symbol": "SPY", "entry_ts": days[0:400:40], "exit_ts": days[20:400:40],
                               "quantity": 100, "entry_price": 100.0, "exit_price": 101.0,
                               "pnl": rng.normal(50, 100, 10), "r_multiple": rng.normal(0.3, 1, 10)})
    payload = report.write_strategy_report(res, tmp_path, n_trials=1450, var_sr=0.5 / 252, bootstrap_reps=200)
    d = tmp_path / "S999"
    for f in ("report.md", "report.json", "equity.png", "drawdown.png", "rolling_sharpe.png", "monthly_returns.png",
              "r_distribution.png"):
        assert (d / f).exists(), f
    assert (d / "equity.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    js = json.loads((d / "report.json").read_text())
    assert js["validation"]["n_trials"] == 1450 and 0 <= js["validation"]["dsr"] <= 1
    assert set(js["validation"]["cost_stress"][0]) >= {"scenario", "extra_bp_per_side", "all_in_round_trip_bp"}
    md = (d / "report.md").read_text()
    assert "Deflated Sharpe" in md and "Cost stress" in md and md.index("Deflated Sharpe") < md.index("| Sharpe")
    assert payload["fidelity_grade"] == "B"
    tab = charts.monthly_table(res.returns)
    assert tab.shape[1] <= 12 and np.isfinite(tab.iloc[0, 0])


def test_not_testable_report(tmp_path):
    res = BacktestResult(strategy_id="S110", params={}, status=RunStatus.NOT_TESTABLE, equity=pd.Series(dtype=float),
                         returns=pd.Series(dtype=float), fills=pd.DataFrame(), trades=pd.DataFrame(),
                         positions=pd.DataFrame(), ledger=pd.DataFrame(), fidelity={},
                         not_testable_reason="no historical timestamped iNAV")
    p = report.write_strategy_report(res, tmp_path)
    assert p["fidelity_grade"] == "NA"
    assert "iNAV" in (tmp_path / "S110" / "report.md").read_text()


# ---------------------------------------------------------------- leaderboard
def test_leaderboard_sorts_by_adjusted_metric_not_raw_sharpe(tmp_path):
    rng = np.random.default_rng(3)
    # A: highest raw Sharpe (~4.9) but a 60-day sample with one crash (skew ~ -7.6, kurt ~ 59)
    a = np.full(60, 0.006)
    a[40] = -0.10
    # B: lower raw Sharpe (~3.2) on a long, well-behaved sample
    b = rng.normal(0.002, 0.01, 2500)
    c = rng.normal(0.0, 0.01, 2500)
    ra, rb, rc = make_result(a, "SA"), make_result(b, "SB"), make_result(c, "SC")
    nt = BacktestResult(strategy_id="SX", params={}, status=RunStatus.NOT_TESTABLE, equity=pd.Series(dtype=float),
                        returns=pd.Series(dtype=float), fills=pd.DataFrame(), trades=pd.DataFrame(),
                        positions=pd.DataFrame(), ledger=pd.DataFrame(), fidelity={},
                        not_testable_reason="needs PIT consensus EPS")
    rows = [leaderboard.summarize_result(x) for x in (ra, rb, rc, nt)]
    assert rows[0]["sharpe"] > rows[1]["sharpe"]
    tested, other = leaderboard.write_leaderboard(rows, tmp_path, n_trials_total=1450, var_sr=0.5 / 252)
    assert list(tested.strategy_id)[0] == "SB"
    assert tested.dsr.is_monotonic_decreasing
    assert list(other.strategy_id) == ["SX"] and "PIT consensus" in other.reason.iloc[0]
    assert (tested.n_trials_total == 1450).all() and tested.q_value.between(0, 1).all()
    md = (tmp_path / "leaderboard.md").read_text()
    assert "Not testable" in md and "N = 1450" in md
