"""Walk-forward/holdout, purged k-fold, cost stress (sheet A-E), regimes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlasbt.validation import cost_stress as cs
from atlasbt.validation import cv, regimes
from atlasbt.validation import walkforward as wf
from val_data import fills_frame, synthetic_result


# ---------------------------------------------------------------- walk-forward
def test_expand_grid_order():
    g = wf.expand_grid({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(g) == 6 and g[0] == {"a": 1, "b": "x"} and g[-1] == {"a": 2, "b": "z"}
    assert wf.param_key({"b": 1, "a": 2}) == "a=2|b=1"


def test_splits_rolling_anchored_and_holdout():
    idx = pd.bdate_range("2010-01-01", periods=1000, tz="UTC")
    s = wf.walk_forward_splits(idx, train=300, test=100, holdout=0.2)
    assert len(s.holdout) == 200 and s.holdout[0] == idx[800]
    for f in s.folds:
        assert len(f.train) == 300 and f.train[-1] < f.test[0]
        assert f.test[-1] < s.holdout[0]
    assert [f.test[0] for f in s.folds] == [idx[i] for i in (300, 400, 500, 600, 700)]
    a = wf.walk_forward_splits(idx, train=300, test=100, anchored=True, holdout="2013-01-01")
    assert all(f.train[0] == idx[0] for f in a.folds) and a.holdout[0] >= pd.Timestamp("2013-01-01", tz="UTC")


def test_walk_forward_matrix_selects_and_never_peeks_at_holdout():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-01", periods=1200, tz="UTC")
    R = pd.DataFrame(rng.normal(0, 0.01, (1200, 5)), index=idx, columns=[f"p{i}" for i in range(5)])
    R["p3"] += 0.002
    s = wf.walk_forward_splits(idx, 250, 125, holdout=0.25)
    res = wf.walk_forward_matrix(R, s)
    assert (res.folds.chosen == "p3").all() and res.holdout_params == "p3"
    assert res.oos_returns.index.max() < s.holdout[0]
    assert len(res.oos_returns) == sum(len(f.test) for f in s.folds)
    R2 = R.copy()
    R2.loc[s.holdout, "p1"] += 1.0  # absurd holdout performance must not influence fold selection
    res2 = wf.walk_forward_matrix(R2, s)
    pd.testing.assert_frame_equal(res.folds, res2.folds)
    assert res2.holdout_params == "p3"
    sm = res.summary()
    assert sm["param_stability"] == 1.0 and sm["n_folds"] == len(s.folds)


def test_walk_forward_callable_mode():
    idx = pd.bdate_range("2012-01-01", periods=600, tz="UTC")
    rng = np.random.default_rng(1)
    base = pd.Series(rng.normal(0, 0.01, 600), index=idx)

    def run(p, start, end):
        return (base + 0.001 * p["edge"]).loc[start:end]

    s = wf.walk_forward_splits(idx, 200, 100, holdout=100)
    res = wf.walk_forward_run(run, {"edge": [0, 1, 2]}, s)
    assert (res.folds.chosen == "edge=2").all() and res.holdout_params == "edge=2"


# ---------------------------------------------------------------- purged k-fold
def test_purged_kfold_purges_and_embargoes():
    t0 = np.arange(10)
    t1 = t0 + 2
    folds = list(cv.PurgedKFold(5, embargo_pct=0.1).split(t0, t1))
    train, test = folds[0]
    assert list(test) == [0, 1]
    # test span [0, 3]: obs 0..3 overlap -> purged; obs 2 also embargoed
    assert list(train) == [4, 5, 6, 7, 8, 9]
    train, test = folds[2]
    assert list(test) == [4, 5]  # span [4, 7]; overlapping obs 2..7 purged; embargo obs 6
    assert list(train) == [0, 1, 8, 9]
    for tr, te in folds:
        assert not set(tr) & set(te)
        for i in tr:  # no training label interval intersects the test span
            assert t1[i] < t0[te].min() or t0[i] > t1[te].max()


# ---------------------------------------------------------------- cost stress
def _flat_result_with_fills():
    r = np.zeros(10)
    res = synthetic_result(r)
    ts = [res.equity.index[3] - pd.Timedelta(hours=5), res.equity.index[6] - pd.Timedelta(hours=5)]
    # 100k notional per fill; spread_cost 7.5$ (=0.75 bp half-spread), commission 1$ (=0.1 bp)
    res.fills = fills_frame(ts, [1000, -1000], [100.0, 100.0], spread_cost=7.5, commission=1.0)
    return res


def test_sheet_formula_and_scenarios():
    assert cs.SCENARIOS == {"A": 0, "B": 2, "C": 5, "D": 10, "E": 25}
    sheet = {"A": 1.7, "B": 5.7, "C": 11.7, "D": 21.7, "E": 51.7}  # E = 2B + 1.5 + 2*0.1
    for k, bp in cs.SCENARIOS.items():
        assert cs.all_in_round_trip_bp(bp, 1.5, 0.1) == pytest.approx(sheet[k])
    tab = cs.cost_stress_table(_flat_result_with_fills())
    np.testing.assert_allclose(tab.measured_spread_bp, 1.5)
    np.testing.assert_allclose(tab.commission_bp_per_side, 0.1)
    np.testing.assert_allclose(tab.all_in_round_trip_bp.to_numpy(), [sheet[k] for k in "ABCDE"])
    assert tab.loc["D", "extra_cost_usd"] == pytest.approx(200.0)  # 2 fills x 100k x 10bp


def test_apply_cost_stress_repricing_exact():
    res = _flat_result_with_fills()
    p = cs.apply_cost_stress(res, 10.0, mode="proportional")
    assert p.returns.iloc[3] == pytest.approx(-0.001) and p.returns.iloc[6] == pytest.approx(-0.001)
    assert p.equity.iloc[-1] == pytest.approx(100_000 * 0.999 ** 2)
    assert p.equity.iloc[2] == pytest.approx(100_000)
    a = cs.apply_cost_stress(res, 10.0, mode="additive")
    assert a.equity.iloc[-1] == pytest.approx(99_800.0) and a.equity.iloc[5] == pytest.approx(99_900.0)
    z = cs.apply_cost_stress(res, 0.0)
    pd.testing.assert_series_equal(z.equity, res.equity, check_names=False)


def test_trade_list_repricing():
    res = _flat_result_with_fills()
    res.trades = pd.DataFrame({"symbol": ["SPY"], "entry_ts": [res.fills.ts[0]], "exit_ts": [res.fills.ts[1]],
                               "quantity": [1000], "entry_price": [100.0], "exit_price": [101.0],
                               "pnl": [1000.0], "r_multiple": [2.0]})
    s = cs.apply_cost_stress(res, 5.0)
    extra = 5e-4 * 1000 * (100 + 101)
    assert s.trades.pnl.iloc[0] == pytest.approx(1000 - extra)
    assert s.trades.r_multiple.iloc[0] == pytest.approx(2.0 * (1000 - extra) / 1000)


def test_break_even_cost_solver():
    n = 252
    res = synthetic_result(np.full(n, 0.001))
    ts = list(res.equity.index - pd.Timedelta(hours=3))
    res.fills = fills_frame(ts + ts, [1000] * n + [-1000] * n, [100.0] * (2 * n))
    be = cs.break_even_cost(res, "cagr")
    assert be["status"] == "ok" and 3.0 < be["bp"] < 7.0
    s = cs.apply_cost_stress(res, be["bp"])
    assert s.equity.iloc[-1] == pytest.approx(100_000, rel=1e-6)
    neg = synthetic_result(np.full(50, -0.001), fills=fills_frame([pd.Timestamp("2020-01-03", tz="UTC")], [1], [1.0]))
    assert cs.break_even_cost(neg)["status"] == "already_below"
    assert cs.break_even_cost(synthetic_result(np.full(50, 0.001)))["status"] == "no_trades"


# ---------------------------------------------------------------- regimes
def test_regime_labels_are_ex_ante():
    idx = pd.bdate_range("2000-01-03", periods=400, tz="UTC")
    spy = pd.Series(np.r_[np.linspace(100, 200, 250), np.linspace(200, 120, 150)], index=idx)
    lab = regimes.trend_regime(spy, 200, lag=1)
    assert lab.iloc[:200].isna().all() and lab.iloc[249] == "BULL" and lab.iloc[-1] == "BEAR"
    # lag: label at t uses close of t-1
    raw = regimes.trend_regime(spy, 200, lag=0)
    assert lab.iloc[300] == raw.iloc[299]
    vix = pd.Series(np.r_[np.full(300, 15.0), np.full(100, 40.0)], index=idx)
    vt = regimes.vix_terciles(vix, causal=True, min_history=252)
    assert vt.iloc[:252].isna().all() and vt.iloc[301] == "VIX_HIGH"
    rr = regimes.rate_regime(pd.Series(np.r_[np.full(200, 0.005), np.full(200, 0.05)], index=idx))
    assert rr.iloc[100] == "RATE_LT_1%" and rr.iloc[-1] == "RATE_GE_3%"
    d = regimes.decade(pd.DatetimeIndex(["1999-12-31", "2000-01-03"]))
    assert list(d) == ["1990s", "2000s"]


def test_regime_table_numbers():
    idx = pd.bdate_range("2001-01-01", periods=4, tz="UTC")
    r = pd.Series([0.01, -0.02, 0.03, 0.0], index=idx)
    lab = pd.Series(["A", "B", "A", "B"], index=idx)
    t = regimes.regime_table(r, lab)
    assert t.loc["A", "days"] == 2 and t.loc["A", "total_return"] == pytest.approx(1.01 * 1.03 - 1)
    assert t.loc["B", "max_drawdown"] == pytest.approx(-0.02) and t.loc["A", "hit_rate"] == 1.0
    allr = regimes.all_regimes(r)
    assert "decade" in allr and allr["decade"].index[0] == "2000s"
