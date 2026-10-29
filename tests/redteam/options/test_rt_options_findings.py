"""RED-TEAM findings for atlasbt.pricing / atlasbt.options.

Every test here encodes the CORRECT behaviour (reference: OCC adjustment memos,
Cboe Margin Manual Nov-2021, arbitrage bounds) and FAILS on the code as reviewed.
Run:  python -m pytest -q tests/redteam/options -o addopts=""
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlasbt.core import clock
from atlasbt.core.types import ExerciseStyle, Instrument, Right, SettlementType
from atlasbt.pricing.bsm import bsm_greeks


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _opt(K, right, expiry="2024-03-15", und="XYZ", st=SettlementType.PHYSICAL):
    return Instrument.option(und, pd.Timestamp(expiry), K, Right(right), ExerciseStyle.AMERICAN, st)


def _ref(S):
    return lambda u, s, kind: S


def _bsm_chain(spot, session, expiries, strikes, vol=0.30, r=0.01, half_spread=0.03, underlying="SPY"):
    sess = pd.Timestamp(session)
    rows = []
    for e in expiries:
        e = pd.Timestamp(e)
        T = (e - sess).days / 365.0
        for K in strikes:
            for cp in ("C", "P"):
                g = bsm_greeks(spot, K, T, r, 0.0, vol, cp)
                m = float(g.price)
                rows.append(dict(ts=clock.et(sess, "16:00"), underlying=underlying, expiry=e, strike=float(K),
                                 right=cp, bid=round(max(m - half_spread, 0.0), 2), ask=round(m + half_spread, 2),
                                 iv=vol, delta=float(g.delta), underlying_price=spot, exercise="AMERICAN",
                                 settlement="PHYSICAL", multiplier=100.0, fidelity=2))
    return pd.DataFrame(rows)


# ============================================================================
# F1  OCC lifecycle ignores non-standard deliverables (3:2 split, reverse split)
# ============================================================================
def test_rt_occ_expiry_nonstandard_deliverable_cash_is_strike_times_multiplier():
    """OCC: after a 3-for-2 split the contract delivers 150 shares, strike unchanged,
    multiplier 100 -> aggregate exercise price = 50 x 100 = $5,000 (NOT 150 x 50)."""
    from atlasbt.options.lifecycle import adjust_for_split, expiration_events
    adj, q = adjust_for_split(_opt(50.0, "C"), 1, 1.5)
    assert adj.deliverable_shares == 150 and adj.strike == 50 and q == 1
    ev = expiration_events({adj: 1}, "2024-03-15", _ref(60.0))[0]
    assert ev.kind == "EXERCISE" and ev.stock_qty == 150
    assert ev.cash == pytest.approx(-5000.0)          # actual: -7500 (150 x 50)
    assert ev.stock_price == pytest.approx(5000.0 / 150)  # actual: 50


def test_rt_occ_expiry_nonstandard_deliverable_itm_test_uses_deliverable_value():
    """3:2-adjusted call K=50 (150 sh) with post-split S=40: value = 150*40 - 100*50 = $1,000 ITM,
    yet the code compares S=40 with K=50 and lets it expire worthless."""
    from atlasbt.options.lifecycle import adjust_for_split, expiration_events
    adj, _ = adjust_for_split(_opt(50.0, "C"), 1, 1.5)
    ev = expiration_events({adj: 1}, "2024-03-15", _ref(40.0))[0]
    assert ev.kind == "EXERCISE"                       # actual: EXPIRE_WORTHLESS (lost $1,000)


def test_rt_occ_expiry_reverse_split_put():
    """1-for-10 reverse split: deliverable 10 sh, strike 50 unchanged.  Post-split S=450
    (pre-split 45): put value = 100*50 - 10*450 = $500 ITM -> must be exercised."""
    from atlasbt.options.lifecycle import adjust_for_split, expiration_events
    adj, _ = adjust_for_split(_opt(50.0, "P"), 1, 0.1)
    ev = expiration_events({adj: 1}, "2024-03-15", _ref(450.0))[0]
    assert ev.kind == "EXERCISE" and ev.stock_qty == -10   # actual: EXPIRE_WORTHLESS


def test_rt_occ_plugin_is_engine_default():
    """Severity context: the defective OccOptionLifecycle is the engine default."""
    """(context, PASSES) BacktestConfig.option_lifecycle=None resolves to OccOptionLifecycle."""
    from atlasbt.engine.config import BacktestConfig
    from atlasbt.options.plugins import OccOptionLifecycle
    lc = BacktestConfig.lifecycle(type("C", (), {"option_lifecycle": None})())
    assert isinstance(lc, OccOptionLifecycle)


# ============================================================================
# F2  Margin: long legs > 9 months inside spreads charged 75% (Cboe: paid in full)
# ============================================================================
def test_rt_margin_spread_long_leg_over_9_months_paid_in_full():
    """Cboe Margin Manual p.10 (Put Spread, Call Spread): 'Long option(s) must be paid for in
    full' - the 75% loan value applies only to stand-alone long options > 9 months.
    S150 diagonal: short 30-DTE 110C @1.00, long 400-DTE 80C @25.00, S=100."""
    from atlasbt.options.margin import OptionPosition as P, strategy_margin
    now = pd.Timestamp("2024-01-02")
    pos = [P(110, "C", now + pd.Timedelta(days=30), -1, 1.00), P(80, "C", now + pd.Timedelta(days=400), 1, 25.00)]
    r = strategy_margin(pos, 100.0, now=now)
    # loss 0 (long strike below short), long paid in full 2,500 -> requirement 2,500
    assert r.requirement == pytest.approx(2500.0)     # actual: 1,875 (75%)


def test_rt_margin_vertical_both_legs_over_9_months_paid_in_full():
    from atlasbt.options.margin import OptionPosition as P, strategy_margin
    now = pd.Timestamp("2024-01-02")
    e = now + pd.Timedelta(days=400)
    r = strategy_margin([P(100, "C", e, 1, 20.0), P(110, "C", e, -1, 15.0)], 100.0, now=now)
    assert r.net_requirement == pytest.approx(500.0)  # net debit; actual: 0 (1,500 - 1,500)


def test_rt_margin_spread_is_lesser_of_max_loss_and_naked():
    """Cboe p.10: spread requirement = LESSER of the short option's naked requirement and the
    spread's max potential loss (code always charges max loss -> over-charges capital)."""
    from atlasbt.options.margin import OptionPosition as P, naked_requirement, strategy_margin
    now = pd.Timestamp("2024-01-02")
    pos = [P(100, "C", now + pd.Timedelta(days=30), -1, 3.0), P(200, "C", now + pd.Timedelta(days=60), 1, 0.05)]
    r = strategy_margin(pos, 100.0, now=now)
    naked = naked_requirement("C", 100, 3.0, 100.0)
    assert r.requirement <= naked + 5.0 + 1e-6        # actual: 10,005


# ============================================================================
# F3  BS-2002 American call below intrinsic when r < 0
# ============================================================================
def test_rt_bs2002_call_not_below_intrinsic_negative_rate():
    from atlasbt.pricing.american import bs2002_price, crr_price
    px = float(bs2002_price(120.0, 100.0, 1.0, -0.01, 0.0, 0.10, True))
    assert px >= 20.0 - 1e-9                           # actual: 19.185 (= European)
    assert float(crr_price(120.0, 100.0, 1.0, -0.01, 0.10, True)) >= 20.0 - 1e-9  # CRR is fine


# ============================================================================
# F4  mark_contract: spread model evaluated at VIX = 20 regardless of the date
# ============================================================================
def test_rt_mark_contract_spread_uses_point_in_time_vix():
    from atlasbt.options.marking import mark_contract
    from atlasbt.options.synthetic import SyntheticParams, cboe_index_history, model_spread
    now = clock.et(pd.Timestamp("2020-03-16"), "16:30")
    ch = _bsm_chain(240.0, "2020-03-16", ["2020-03-27", "2020-04-17"], np.arange(200, 281, 2.0), vol=0.70)
    m = mark_contract(ch, now, 0.01, "2020-03-20", 230.0, "P", underlying="SPY")   # not quoted -> modeled
    assert np.isfinite(m.mid) and m.reason.endswith("bid/ask from synthetic spread model")
    vix = float(cboe_index_history("VIX").loc[:"2020-03-16"].iloc[-1]) / 100.0   # 82.69 close
    p = SyntheticParams.load()
    dte = (pd.Timestamp("2020-03-20") - pd.Timestamp("2020-03-16")).days
    expected = float(model_spread(m.mid, dte, abs(m.delta), "SPY", p, is_etf=True, vix=vix))
    # bid/ask are rounded OUTWARD to the $0.01 tick (item 4) -> allow up to one tick per side
    assert abs((m.ask - m.bid) - expected) <= 0.10 * expected + 0.02 + 1e-9   # was ~0.37x expected (VIX=20 used)


# ============================================================================
# F5  SPX: monthly AM (expiry=Thursday LTD) and SPXW Thursday PM share one expiry date;
#     builders silently mix AM- and PM-settled legs in one package
# ============================================================================
def test_rt_spx_structure_does_not_mix_am_and_pm_settlement():
    from atlasbt.options import structures as st
    from atlasbt.options.chain import choose_expiry, prepare_chain, slice_expiry
    from atlasbt.options.synthetic import SyntheticChainGenerator
    now = clock.et(pd.Timestamp("2023-06-20"), "16:30")
    ch = SyntheticChainGenerator().chain("SPX", now, 4400.0, 0.05, 0.015, max_dte=45)
    x = prepare_chain(ch, now, r=0.05)
    e = choose_expiry(x, 30, 25, 35)
    # integrator item 1: a date carrying both SPX AM and SPXW PM series must be disambiguated
    from atlasbt.options.chain import SelectionError
    if x[x["expiry"] == e][["root", "settlement"]].drop_duplicates().shape[0] > 1:
        with pytest.raises(SelectionError):
            slice_expiry(x, e)
    s = slice_expiry(x, e, settlement="CASH_PM")
    pkg = st.iron_condor(s, 0.20, 0.05)
    assert pkg.same_expiry
    assert len({l.settlement for l in pkg.legs}) == 1, [(l.strike, l.right, l.root, l.settlement) for l in pkg.legs]
    # an SPXW (PM) leg and an SPX AM leg on the same date are NOT the same expiry
    am = slice_expiry(x, e, settlement="CASH_AM")
    if len(am):
        mixed = st.Package("mixed", [pkg.legs[0], st.OptLeg.from_row(am.iloc[0], 1)])
        assert not mixed.same_expiry


# ============================================================================
# F6  Synthetic index products other than SPX get SPX roots / SPX listing calendar
# ============================================================================
@pytest.mark.parametrize("und,spot", [("XSP", 450.0), ("NDX", 15000.0), ("RUT", 2000.0)])
def test_rt_synthetic_index_chain_root_matches_underlying(und, spot):
    from atlasbt.options.synthetic import SyntheticChainGenerator
    now = clock.et(pd.Timestamp("2023-06-13"), "16:30")
    ch = SyntheticChainGenerator().chain(und, now, spot, 0.05, 0.015, max_dte=40)
    assert len(ch)
    assert not ch["root"].isin(["SPX", "SPXW"]).any(), ch["root"].unique()
    assert not ch["symbol"].str.startswith("SPX").any()


# ============================================================================
# F7  Lookahead: ATM-ratio "point-in-time" schedule used for dates before its fit end
# ============================================================================
def test_rt_atm_ratio_params_not_fitted_after_as_of():
    from atlasbt.options.synthetic import SyntheticParams, atm_ratio_params
    p = SyntheticParams.load()
    as_of = pd.Timestamp("2008-10-15")           # GFC; VIX ~ 70
    c0, c1, label = atm_ratio_params(p, as_of)
    # integrator: accept a labelled prior (blend) or an NA flag; any fitted part must end <= as_of
    assert label.startswith("prior") or label.startswith("PIT"), label
    if "fit through " in label:
        fitted_through = pd.Timestamp(label.split("fit through ")[1].rstrip(")"))
        assert fitted_through <= as_of, label    # was: 'pre-sample (fit through 2009-11-20)'
    from atlasbt.options.synthetic import SyntheticChainGenerator
    gen = SyntheticChainGenerator()
    ch = gen.chain("SPX", clock.et(as_of, "16:30"), 900.0, 0.01, 0.02, max_dte=40)
    assert "atm_ratio_prior" in ch.attrs.get("fidelity_reasons", [])
    strict = gen.chain("SPX", clock.et(as_of, "16:30"), 900.0, 0.01, 0.02, max_dte=40, strict=True)
    assert strict.empty and int(strict.attrs["fidelity"]) == 9


# ============================================================================
# F8  wing_by_points(policy='nearest') can return a wing on the WRONG side
# ============================================================================
def test_rt_wing_nearest_never_inside_base():
    from atlasbt.options.chain import wing_by_points
    df = pd.DataFrame({"strike": [95.0, 99.0, 100.0, 115.0], "right": ["C"] * 4})
    k = wing_by_points(df, 100.0, 5.0, "C", policy="nearest")
    assert k > 100.0                              # actual: 99 (call wing BELOW the short strike)


# ============================================================================
# F9  chain_metrics accepts synthetic (T3) chains silently (no fidelity in output)
# ============================================================================
def test_rt_chain_metrics_marks_or_refuses_synthetic_input():
    from atlasbt.options.metrics import chain_metrics
    from atlasbt.options.synthetic import SyntheticChainGenerator
    out = []
    for d in ("2018-02-05", "2019-08-15"):      # VIX 37 vs 22
        now = clock.et(pd.Timestamp(d), "16:30")
        ch = SyntheticChainGenerator().chain("SPX", now, 2700.0, 0.02, 0.018, max_dte=70)
        m = chain_metrics(ch, now, 0.02, spot=2700.0, american=False)
        out.append(m)
    # synthetic 25d skew is a fixed multiple of ATM vol -> skew z-scores are VIX z-scores
    ratios = [m["skew25_30"] / m["atm_iv_30"] for m in out]
    assert abs(ratios[0] - ratios[1]) < 0.01, ratios   # documents the degeneracy (passes)
    assert any(k in out[0] for k in ("fidelity", "synthetic")), "metrics carry no fidelity/synthetic marker"
    assert out[0]["synthetic"] is True and out[0]["fidelity"] == 3
    with pytest.raises(ValueError):          # integrator: refusing synthetic input is also acceptable
        chain_metrics(ch, now, 0.02, spot=2700.0, american=False, allow_synthetic=False)
