import numpy as np
import pandas as pd
import pytest

from atlasbt.core.types import OrderType
from atlasbt.options import structures as st
from atlasbt.options.chain import (QualityRules, SelectionError, choose_expiry, flag_stale_vs_previous, nearest_atm,
                                   prepare_chain, quality_mask, select_by_delta, slice_expiry, wing_by_points)

NOW = pd.Timestamp("2024-03-04 21:00", tz="UTC")


def prep(ch):
    return prepare_chain(ch, NOW, r=0.04)


def test_dte_and_expiry_choice(chain):
    x = prep(chain)
    assert set(x["dte"]) == {11, 46, 74}
    assert choose_expiry(x, 30, 25, 50) == pd.Timestamp("2024-04-19")
    assert choose_expiry(x, 60) == pd.Timestamp("2024-04-19")  # |46-60|=14 < |74-60|=14? tie -> earlier
    with pytest.raises(SelectionError):
        choose_expiry(x, 30, 200, 300)


def test_select_by_delta_uses_quoted_then_computed(chain):
    x = prep(chain)
    e = slice_expiry(x, "2024-04-19")
    r = select_by_delta(e, 0.30)
    assert r["right"] == "C" and abs(r["delta_used"] - 0.30) < 0.03
    p = select_by_delta(e, -0.25)
    assert p["right"] == "P" and abs(p["delta_used"] + 0.25) < 0.03
    # remove quoted deltas -> computed from quoted iv, same selection
    x2 = prep(chain.drop(columns=["delta"]))
    e2 = slice_expiry(x2, "2024-04-19")
    assert (e2["delta_source"] == "from_quoted_iv").all()
    assert select_by_delta(e2, 0.30)["strike"] == r["strike"]
    # remove iv too -> from mid IV
    x3 = prep(chain.drop(columns=["delta", "iv"]))
    assert (slice_expiry(x3, "2024-04-19")["delta_source"].isin(["from_mid_iv", "none"])).all()


def test_nearest_atm_tie_lower_strike(chain):
    e = slice_expiry(prep(chain), "2024-04-19")
    assert nearest_atm(e, 100.5) == 100.0
    assert nearest_atm(e, 100.2) == 100.0


def test_wing_by_points_listed_strikes():
    from tests.options.conftest import synthetic_chain
    ch = synthetic_chain(strikes=[90, 95, 100, 105, 110, 120])
    e = slice_expiry(prep(ch), "2024-04-19")
    assert wing_by_points(e, 105, 10, "C") == 120  # 115 not listed -> outward
    assert wing_by_points(e, 105, 10, "C", "inward") == 110
    with pytest.raises(SelectionError):
        wing_by_points(e, 105, 10, "C", "exact")
    assert wing_by_points(e, 100, 5, "P") == 95


def test_quality_filters(chain):
    x = prep(chain)
    x.loc[x.index[0], ["bid", "ask"]] = [1.0, 1.0]       # locked
    x.loc[x.index[1], ["bid", "ask"]] = [0.0, 0.05]      # zero bid
    x.loc[x.index[2], ["bid", "ask"]] = [1.0, 3.0]       # wide
    ok = quality_mask(x, QualityRules(max_rel_spread=0.5))
    rs = x.attrs["quality_reasons"]
    assert not ok.iloc[0] and rs.iloc[0] == "locked_or_crossed"
    assert not ok.iloc[1] and rs.iloc[1] == "zero_bid"
    assert not ok.iloc[2] and rs.iloc[2] == "wide_spread"


def test_stale_detection(chain):
    prev = prep(chain)
    cur = prev.copy()
    cur["underlying_price"] = 101.0
    cur.loc[cur.index[5:], "bid"] += 0.5
    stale = flag_stale_vs_previous(cur, prev)
    assert stale.iloc[:5].any() and not stale.iloc[5:].any()


def test_vertical_iron_condor_payoffs(chain):
    e = slice_expiry(prep(chain), "2024-04-19")
    bull = st.vertical(e, 0.60, 0.30)
    k1, k2 = bull.legs[0].strike, bull.legs[1].strike
    debit = bull.natural_open()
    mp = bull.max_profit_loss()
    assert mp["max_loss"] == pytest.approx(debit)
    assert mp["max_profit"] == pytest.approx(k2 - k1 - debit)
    assert bull.breakevens() == [pytest.approx(k1 + debit)]
    ic = st.iron_condor(e, 0.20, 0.05)
    credit = -ic.natural_open()
    assert credit > 0
    widths = (ic.legs[1].strike - ic.legs[0].strike, ic.legs[3].strike - ic.legs[2].strike)
    assert ic.max_profit_loss()["max_loss"] == pytest.approx(max(widths) - credit)
    assert len(ic.breakevens()) == 2
    ic_pts = st.iron_condor(e, 0.20, wing_points=5)
    assert ic_pts.legs[3].strike - ic_pts.legs[2].strike == 5


def test_unbounded_structures(chain):
    e = slice_expiry(prep(chain), "2024-04-19")
    ss = st.strangle(e, 0.25, -0.25, ratio=-1)
    assert ss.max_profit_loss()["max_loss"] == np.inf
    back = st.ratio_backspread(e, 0.50, 0.25)
    assert back.max_profit_loss()["max_profit"] == np.inf
    rr = st.risk_reversal(e, 0.25, bullish=False)
    assert rr.max_profit_loss()["max_loss"] == np.inf


def test_butterflies_and_broken_wing(chain):
    e = slice_expiry(prep(chain), "2024-04-19")
    bf = st.butterfly(e, "C", 5)
    k = sorted(l.strike for l in bf.legs)
    assert k[1] - k[0] == k[2] - k[1] == 5
    assert bf.max_profit_loss(entry_price=bf.mid_value())["max_profit"] == pytest.approx(5 - bf.mid_value())
    bw = st.butterfly(e, "C", 4, far_multiple=1.5)
    ks = [l.strike for l in bw.legs]
    assert ks[2] - ks[1] == pytest.approx(1.5 * (ks[1] - ks[0]))
    pf = st.butterfly(e, "P", 5)
    assert sorted(l.ratio for l in pf.legs) == [-2, 1, 1]
    fly = st.iron_butterfly(e, 5, short=True)
    assert fly.max_profit_loss()["max_loss"] == pytest.approx(5 + fly.natural_open())


def test_calendar_diagonal_box_conversion(chain):
    x = prep(chain)
    near, far = slice_expiry(x, "2024-04-19"), slice_expiry(x, "2024-05-17")
    cal = st.calendar(near, far, "C")
    assert cal.legs[0].strike == cal.legs[1].strike and cal.natural_open() > 0
    with pytest.raises(ValueError):
        cal.max_profit_loss()
    dg = st.diagonal(near, far, 0.80, 0.30)
    assert dg.legs[1].expiry > dg.legs[0].expiry
    bx = st.box(near, 95, 105)
    assert bx.payoff(np.array([50, 100, 150])) == pytest.approx([10, 10, 10])
    conv = st.conversion(near, 100, "XYZ", 99.99, 100.01)
    assert np.allclose(conv.payoff(np.array([80, 100, 120])), 100.0 / 1)  # locks K per share


def test_jade_lizard_and_collar_and_csp(chain):
    e = slice_expiry(prep(chain), "2024-04-19")
    jl = st.jade_lizard(e, 0.30, 0.30, 0.15)
    assert "no_upside_risk" in jl.meta
    c = st.collar(e)
    assert c.legs[0].right == "P" and c.legs[1].ratio == -1
    csp = st.cash_secured_put(e, -0.30)
    assert csp.meta["collateral"] == csp.legs[0].strike * 100
    atm = st.cash_secured_put(e, atm=True)
    assert atm.legs[0].strike == 100


def test_order_conversion(chain):
    e = slice_expiry(prep(chain), "2024-04-19")
    ic = st.iron_condor(e, 0.2, 0.05)
    o = ic.to_order(2, limit_price=ic.natural_open())
    assert o.type == OrderType.COMBO and len(o.legs) == 4 and o.quantity == 2
    assert o.legs[0].instrument.symbol.startswith("XYZ   240419P")


def test_chain_metrics_on_synthetic(chain):
    from atlasbt.options.metrics import chain_metrics
    m = chain_metrics(chain, NOW, r=0.04, american=False)
    assert m["atm_iv_30"] == pytest.approx(0.2, abs=0.01) and m["flag_30"] == "interp"
    assert m["atm_iv_60"] == pytest.approx(0.2, abs=0.01)
    assert m["skew25_30"] > 0.03
    assert abs(m["q_near"]) < 0.01


def test_hedge_and_vega_matching(chain):
    e = slice_expiry(prep(chain), "2024-04-19")
    strd = st.straddle(e, ratio=-1)
    g = strd.dollar_greeks(10)
    assert g["vega_usd"] < 0 and abs(g["delta_shares"]) < 200
    assert st.hedge_shares(g["delta_shares"], 0, band_shares=5000) == 0.0
    sh = st.hedge_shares(523.0, -100.0)
    assert sh == -423.0
    long_strd = st.straddle(e)
    units = st.vega_match_units(-g["vega_usd"], long_strd, ratio=1.0)
    assert units == pytest.approx(10.0, rel=1e-9)


def test_mark_contract_quoted_interpolated_extrapolated(chain):
    from atlasbt.core.types import Fidelity
    from atlasbt.options.marking import mark_contract
    from atlasbt.pricing import bsm_price
    ch = chain.copy()
    ch["fidelity"] = 2
    m = mark_contract(ch, NOW, 0.04, "2024-04-19", 100, "C", exercise="EUROPEAN")
    assert m.reason == "quoted" and m.fidelity == Fidelity.T2_PROXY
    # drop the strike -> same-expiry surface interpolation
    m2 = mark_contract(ch[ch.strike != 100], NOW, 0.04, "2024-04-19", 100, "C", exercise="EUROPEAN")
    T = (pd.Timestamp("2024-04-19") - pd.Timestamp("2024-03-04")).days / 365
    assert m2.fidelity == Fidelity.T2_PROXY and m2.mid == pytest.approx(float(bsm_price(100, 100, T, 0.04, 0, 0.2, "C")), rel=0.02)
    # expiry between listed ones -> total variance interpolation (T2)
    m3 = mark_contract(ch, NOW, 0.04, "2024-05-03", 100, "P", exercise="EUROPEAN")
    assert m3.fidelity == Fidelity.T2_PROXY and m3.iv == pytest.approx(0.2, abs=0.01)
    # before the nearest listed expiry -> extrapolated (T3)
    m4 = mark_contract(ch, NOW, 0.04, "2024-03-08", 100, "P", exercise="EUROPEAN")
    assert m4.fidelity == Fidelity.T3_MODELED and m4.bid <= m4.mid <= m4.ask
