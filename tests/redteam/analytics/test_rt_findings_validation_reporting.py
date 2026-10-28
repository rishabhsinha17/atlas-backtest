"""Red-team findings in atlasbt.validation and atlasbt.reporting. Each test
asserts the correct behaviour and FAILS on the current code."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.types import BacktestResult, RunStatus
from atlasbt.reporting import leaderboard, metrics, report
from atlasbt.validation import (benjamini_hochberg, expected_max_sharpe, total_atlas_trials, walk_forward_matrix,
                                walk_forward_splits)


def make_result(returns, sid="SYN", nav0=100_000.0, start="2020-01-02", **kw):
    r = np.asarray(returns, float)
    idx = pd.bdate_range(start, periods=len(r), tz="UTC") + pd.Timedelta(hours=21)
    eq = pd.Series(nav0 * np.cumprod(1 + r), index=idx)
    base = dict(strategy_id=sid, params={}, status=RunStatus.OK, equity=eq, returns=pd.Series(r, index=idx),
                fills=pd.DataFrame(), trades=pd.DataFrame(), positions=pd.DataFrame(), ledger=pd.DataFrame(),
                fidelity={}, meta={"initial_capital": nav0})
    base.update(kw)
    return BacktestResult(**base)


# ---------------------------------------------------------------- CRITICAL: leaderboard DSR is not deflated
def test_leaderboard_dsr_is_actually_deflated_for_atlas_trial_count():
    """build_leaderboard's default V[SR] = cross-strategy variance of the TESTED
    strategies' Sharpes floored at 1e-6 (per period). With one strategy (or a set
    of similar ones) V = 1e-6 => SR0 = 0.0034/period = 0.053 annualised even with
    N = 1450 trials, so DSR ~= PSR (no deflation). The per-strategy report uses
    V = 0.5/252 (SR0 = 2.2 annualised) for the SAME strategy -> DSR 2.5e-12.
    Even the pure estimation-noise variance 1/T gives SR0 = 1.06 annualised.
    A strategy with annualised SR ~0.3 over 10y (PSR 0.83) must not get DSR 0.78."""
    rng = np.random.default_rng(1)
    res = make_result(rng.normal(0.0003, 0.01, 2520), sid="S1")
    tested, _ = leaderboard.build_leaderboard([leaderboard.summarize_result(res)])
    lb_dsr = float(tested["dsr"].iloc[0])
    rep_dsr = report.strategy_validation(res, bootstrap_reps=20, cost_stress=False)["dsr"]
    assert tested["n_trials_total"].iloc[0] >= 1450
    assert lb_dsr < 0.05, f"leaderboard DSR {lb_dsr:.3f} (PSR {float(tested['psr_0'].iloc[0]):.3f})"  # actual 0.780
    assert lb_dsr == pytest.approx(rep_dsr, abs=0.05)  # actual 0.780 vs ~1e-11


def test_leaderboard_sr0_floor_is_below_estimation_noise():
    """(Rewritten per integrator item 1.) A one-row board has no trial-Sharpe
    dispersion, so V[SR] must fall back to the Sharpe's own sampling variance,
    which is at least (1 + 0.5 SR^2)/(T-1) for symmetric (skew 0), fat-tailed
    (kurtosis >= 3) returns. The old 1e-6 floor was ~400x below that."""
    rng = np.random.default_rng(7)
    x = rng.standard_t(5, 1260) * 0.006
    r = np.concatenate([x, -x]) + 0.0004  # exactly symmetric about the mean -> sample skew 0, kurt > 3
    T = r.size
    tested, _ = leaderboard.build_leaderboard([leaderboard.summarize_result(make_result(r))])
    sr = r.mean() / r.std(ddof=1)
    assert tested["var_sr_source"].iloc[0] == "estimation_noise"
    assert tested["var_sr_period"].iloc[0] >= (1 + 0.5 * sr ** 2) / (T - 1)
    sr0 = expected_max_sharpe(total_atlas_trials(), float(tested["var_sr_period"].iloc[0])) * math.sqrt(252)
    assert sr0 >= 0.5 * expected_max_sharpe(total_atlas_trials(), 1.0 / T) * math.sqrt(252)


# ---------------------------------------------------------------- MAJOR: walk-forward duplicates OOS days when step < test
def test_walk_forward_oos_has_no_duplicate_dates_when_step_lt_test():
    """walk_forward_splits accepts step < test; consecutive test windows then
    overlap and walk_forward_matrix concatenates them: 1581 OOS rows over 548
    unique dates. oos_sharpe and oos_total_return (which COMPOUNDS each day ~3x)
    are computed on the duplicated series."""
    idx = pd.bdate_range("2010-01-01", periods=1000)
    rng = np.random.default_rng(0)
    R = pd.DataFrame(rng.normal(0.0003, 0.01, (1000, 4)), index=idx, columns=list("abcd"))
    sp = walk_forward_splits(idx, 252, 63, step=21, holdout=0.2)
    res = walk_forward_matrix(R, sp)
    assert res.oos_returns.index.is_unique, (len(res.oos_returns), res.oos_returns.index.nunique())


# ---------------------------------------------------------------- MINOR: BH with one NaN p-value
def test_benjamini_hochberg_single_nan_does_not_erase_all_qvalues():
    rej, q = benjamini_hochberg([0.001, 0.002, np.nan, 0.5], q=0.05)
    assert rej[0] and rej[1]  # actual: all False, all q = NaN (np.minimum.accumulate propagates NaN)


# ---------------------------------------------------------------- MINOR: cost-drag ledger bucketing with the engine's reason codes
def test_cost_drag_does_not_double_count_engine_fee_reasons():
    """The engine ledger posts EXCHANGE_FEE / REGULATORY_FEE rows for fees that
    cost_drag already takes from the fills; 'FEE' matches the fees_other bucket,
    so the report's cost-drag table lists them twice. SPLIT_CASH_IN_LIEU (split
    fractional-share proceeds) matches 'IN_LIEU' and is booked as dividends."""
    res = make_result(np.zeros(5))
    d = res.equity.index
    fills = pd.DataFrame({"ts": d[:2], "symbol": "A", "quantity": [100, -100], "price": [50.0, 50.0],
                          "commission": [1.0, 1.0], "exchange_fees": [0.3, 0.3], "regulatory_fees": [0.0, 0.2]})
    ledger = pd.DataFrame({"ts": d[[0, 0, 1, 1, 1, 3]], "amount": [-1.0, -0.3, -1.0, -0.3, -0.2, 12.5],
                           "reason": ["COMMISSION", "EXCHANGE_FEE", "COMMISSION", "EXCHANGE_FEE", "REGULATORY_FEE",
                                      "SPLIT_CASH_IN_LIEU"]})
    c = metrics.cost_drag(fills, ledger, res.equity)["usd"]
    assert c["exchange_fees"] == pytest.approx(0.6) and c["regulatory_fees"] == pytest.approx(0.2)
    assert c["fees_other"] == pytest.approx(0.0)  # actual 0.8 (= exchange + regulatory again)


def test_split_cash_in_lieu_is_not_classified_as_dividend():
    from atlasbt.reporting.schema import classify_reason
    assert classify_reason("SPLIT_CASH_IN_LIEU") != "dividends"  # actual 'dividends' (matches 'IN_LIEU')


# ---------------------------------------------------------------- MINOR: rf back-filled from the future
def test_risk_free_rate_series_is_not_backfilled_from_future():
    """risk_free_returns(meta['risk_free_rate']) ffills onto the return index and
    then .shift(1).bfill(): returns before the first rate observation get a rate
    published months later."""
    r = np.zeros(120)
    res = make_result(r)
    late = pd.Series(0.05, index=res.returns.index[60:])
    res.meta["risk_free_rate"] = late
    rfs, src = metrics.risk_free_returns(res)
    assert src == "meta.risk_free_rate"
    assert (rfs.iloc[:60] == 0).all()  # actual 0.05*days/360 on the first 60 days


# ---------------------------------------------------------------- MINOR: leaderboard MaxDD ignores NAV0
def test_leaderboard_max_drawdown_includes_initial_capital():
    r = np.r_[-0.10, np.full(299, 0.001)]
    res = make_result(r)
    row = leaderboard.summarize_result(res)
    full = metrics.compute_metrics(res)["performance"]["max_drawdown"]
    assert full == pytest.approx(-0.10)
    assert row["max_drawdown"] == pytest.approx(full)  # actual 0.0
