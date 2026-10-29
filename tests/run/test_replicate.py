"""Known-answer validations of the whole stack on the real cache (quick windows)."""
import pytest

from atlasbt.run import replicate as R

from .conftest import needs_cache


@needs_cache
def test_lookahead_guard_rejects_perfect_foresight(portal):
    r = R.rep_lookahead(portal, **R.QUICK["lookahead"])
    assert r["modes"]["future_as_of"]["raised"] and r["modes"]["future_actions"]["raised"]
    assert r["modes"]["clipped_end"]["leaks"] == 0 and r["modes"]["history_peek"]["leaks"] == 0
    assert r["pass"]


@needs_cache
def test_spy_buy_and_hold_vs_adjclose(portal):
    r = R.rep_spy(portal, **R.QUICK["spy"])
    d = r["default_fees"]
    assert d["accounting_residual_max_abs_frac_nav"] < 1e-4         # exact accounting reconciliation
    assert abs(r["reinvest_default_fees"]["td_vs_adjclose_ann"]) < 0.002
    assert d["engine_cagr"] < r["reinvest_default_fees"]["engine_cagr"]   # cash dividends drag in a bull market
    assert r["fee_effect_ann"] >= 0


@needs_cache
def test_s105_vs_pandas_reference(portal):
    r = R.rep_s105(portal, **R.QUICK["s105"])
    assert r["daily_corr"] > 0.99 and abs(r["td_ann"]) < 0.005


@needs_cache
def test_random_entries_have_no_positive_fill_bias(portal):
    r = R.rep_random(portal, **R.QUICK["random"])
    assert r["fills_better_than_mid"] == 0
    assert abs(r["gross_mid_t"]) < 3
    assert r["cost_mean_bp"] > 0 and r["net_mean_bp"] < r["gross_mid_mean_bp"]


@needs_cache
@pytest.mark.slow
def test_put_index_through_engine():
    r = R.rep_put(**R.QUICK["put"])
    assert r["n_fills"] > 0 and r["mean_notional_coverage"] > 0.95
    assert r["monthly_corr"] > 0.7 and r["tracking_error_ann"] < 0.05


@needs_cache
@pytest.mark.slow
def test_momentum_spread_positively_correlated_with_umd(portal):
    r = R.rep_umd(portal, **R.QUICK["umd"])
    assert r["corr_full"] > 0.3
