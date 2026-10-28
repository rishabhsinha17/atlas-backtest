"""Synthetic chain generator (T3_MODELED).  Uses the real Cboe index files in
data_raw/cboe_indices (read-only) and the committed calibration params."""
import numpy as np
import pandas as pd
import pytest

from atlasbt.core import clock
from atlasbt.options.synthetic import (CBOE_DIR, SyntheticChainGenerator, SyntheticParams, VolSource, atm_vol_at,
                                       cboe_index_history, model_spread, smile_g)
from atlasbt.pricing import bsm_price

pytestmark = pytest.mark.skipif(not (CBOE_DIR / "VIX_History.csv").exists(), reason="Cboe index files missing")


@pytest.fixture(scope="module")
def gen():
    return SyntheticChainGenerator()


def test_vol_index_point_in_time():
    vs = VolSource()
    vix = cboe_index_history("VIX")
    d = pd.Timestamp("2023-06-13")
    prev = vix.loc[:d - pd.Timedelta(days=1)].iloc[-1]
    before = vs.vix_family(clock.et(d, "10:00"))
    after = vs.vix_family(clock.et(d, "16:20"))
    assert before.loc[before["index"] == "VIX", "level"].iloc[0] == pytest.approx(prev / 100)
    assert after.loc[after["index"] == "VIX", "level"].iloc[0] == pytest.approx(vix.loc[d] / 100)


def test_spx_chain_0dte_and_canonical_columns(gen):
    as_of = clock.et("2023-06-13", "10:00")
    ch = gen.chain("SPX", as_of, spot=4350.0, r=0.05, max_dte=60)
    assert int(ch.attrs["fidelity"]) == 3 and (ch["fidelity"] == 3).all()
    for c in ["ts", "underlying", "symbol", "expiry", "strike", "right", "bid", "ask", "mid", "iv", "delta", "gamma",
              "vega", "theta", "underlying_price", "volume", "open_interest", "exercise", "settlement", "multiplier"]:
        assert c in ch.columns
    zero = ch[(ch["expiry"] == pd.Timestamp("2023-06-13")) & (ch["settlement"] == "CASH_PM")]
    assert len(zero) > 0 and (zero["root"] == "SPXW").all()
    assert (ch["bid"] <= ch["mid"] + 1e-9).all() and (ch["mid"] <= ch["ask"] + 1e-9).all()
    ticks = np.where(ch["mid"] < 3, 0.05, 0.10)
    assert np.allclose(np.round(ch["ask"] / 0.05) * 0.05, ch["ask"])
    # AM monthly present with Thursday last trading day
    am = ch[ch["settlement"] == "CASH_AM"]
    assert (am["expiry"] == pd.Timestamp("2023-06-15")).any()


def test_spx_put_call_parity_and_skew(gen):
    as_of = clock.et("2021-03-17", "16:30")
    S, r = 3974.0, 0.001
    ch = gen.chain("SPX", as_of, spot=S, r=r, max_dte=45, min_dte=25)
    e = ch[ch["settlement"] == "CASH_PM"]["expiry"].iloc[0]
    x = ch[(ch["expiry"] == e) & (ch["settlement"] == "CASH_PM")]
    c = x[x.right == "C"].set_index("strike")["mid"]
    p = x[x.right == "P"].set_index("strike")["mid"]
    from atlasbt.pricing.time import year_fraction
    T = year_fraction(as_of, e, "CASH_PM")
    np.testing.assert_allclose((c - p).values, S - c.index.values * np.exp(-r * T), atol=1e-6)
    # put skew: 25d put IV > ATM > 25d call IV
    put25 = x[(x.right == "P")].iloc[(x[x.right == "P"]["delta"] + 0.25).abs().argsort()[:1]]["iv"].iloc[0]
    call25 = x[(x.right == "C")].iloc[(x[x.right == "C"]["delta"] - 0.25).abs().argsort()[:1]]["iv"].iloc[0]
    assert put25 > call25
    # ATM ~ VIX x atm_ratio
    atm = x.iloc[(x["strike"] - S).abs().argsort()[:1]]["iv"].iloc[0]
    vix = cboe_index_history("VIX").loc["2021-03-17"] / 100
    from atlasbt.options.synthetic import atm_ratio
    assert atm == pytest.approx(vix * float(atm_ratio(vix, gen.params)), rel=0.15)


def test_equity_chain_american_and_mapped_index(gen):
    ch = gen.chain("AAPL", clock.et("2022-06-15", "16:30"), spot=135.0, r=0.015, q=0.006, max_dte=60)
    assert (ch["exercise"] == "AMERICAN").all() and (ch["settlement"] == "PHYSICAL").all()
    assert "VXAPL" in ch.attrs["source"]
    deep_put = ch[(ch.right == "P") & (ch.strike == ch.strike.max())].iloc[-1]
    assert deep_put["mid"] >= deep_put["strike"] - 135.0 - 1e-9   # American >= intrinsic


def test_unmapped_name_needs_rv_ratio(gen):
    ch = gen.chain("KO", clock.et("2022-06-15", "16:30"), spot=62.0, r=0.015)
    assert ch.empty and "no vol index" in ch.attrs["reason"]
    ch2 = gen.chain("KO", clock.et("2022-06-15", "16:30"), spot=62.0, r=0.015, rv_ratio=0.8)
    assert not ch2.empty and "rv_ratio" in ch2.attrs["source"]


def test_no_data_before_index_history(gen):
    ch = gen.chain("SPX", clock.et("1985-06-14", "16:30"), spot=190.0, r=0.07)
    assert ch.empty


def test_term_structure_interpolation_and_smile_shape():
    term = pd.DataFrame({"days": [9, 30, 93], "atm_vol": [0.15, 0.18, 0.20], "source": "x"})
    v, f = atm_vol_at(term, 30 / 365)
    assert v == pytest.approx(0.18) and f in ("interp", "exact")
    v1, f1 = atm_vol_at(term, 1 / 365)
    assert f1 == "extrap_flat" and v1 == pytest.approx(0.15)
    g = smile_g(np.array([-6, -3, 0, 3, 6]), -0.3, 0.04)
    assert g[2] == 1.0 and g[0] > g[1] > g[2] and np.all(g >= 0.35)


def test_spread_model_monotone_in_price():
    p = SyntheticParams.load()
    s = model_spread(np.array([0.1, 1.0, 10.0]), 30, np.array([0.1, 0.3, 0.6]), "SPY", p, True)
    assert np.all(s > 0) and s[2] > s[0]


def test_known_answer_put_index_replication():
    """Synthetic SPX chain reproduces the Cboe PUT index (ATM put-write) monthly returns."""
    from atlasbt.options.calibrate import ROOT, replicate_put_index
    if not (ROOT / "data_raw" / "yfinance" / "daily" / "IDX_GSPC.csv").exists():
        pytest.skip("SPX daily OHLC missing")
    # Documented accuracy (docs/OPTIONS.md 7.4): full 2008-2026 TE 2.6%/yr, corr 0.98, synthetic
    # +1.5%/yr rich; the calm 2014-2017 window is the worst sub-period (+2.9%/yr).
    res = replicate_put_index(start="2014-01-01", end="2017-12-31")
    assert res["n_months"] >= 45
    assert res["correlation"] > 0.9
    assert res["tracking_error_ann"] < 0.04
    assert abs(res["ann_return_synthetic"] - res["ann_return_put_index"]) < 0.035
