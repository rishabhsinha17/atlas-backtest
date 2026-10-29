"""Calibrated spread model: known answers on calibration points, tick eras, monotonicity, PIT,
vectorised == scalar, SpreadModel contract."""
import math

import numpy as np
import pandas as pd
import pytest

from atlasbt.execution.spread import SpreadModel
from atlasbt.execution.spread_calibrated import (
    CALIBRATION_PATH, CalibratedSpreadConfig, CalibratedSpreadModel, era_tick, load_calibration,
)

CAL = load_calibration()
M = CalibratedSpreadModel()
NBBO = CALIBRATION_PATH.parent / f"nbbo_{CAL['fit_day']}.csv.gz"


def _hand(price, adv, sigma, etf, date, tick, mult=None):
    """Formula of the module docstring evaluated term by term from the json."""
    b, c, kn, cl = CAL["beta"], CAL["center"], CAL["knots"], CAL["clip_log"]
    clip = lambda v, k: min(max(math.log(v), cl[k][0]), cl[k][1])
    dfl = float(M.log_deflator([np.datetime64(pd.Timestamp(date))])[0])     # relative liquidity (market $ volume)
    la = clip(adv * math.exp(dfl), "log_adv") - c["log_adv"]
    ls = clip(sigma, "log_sigma") - c["log_sigma"]
    lp = clip(price, "log_price") - c["log_price"]
    ka = math.log(kn["adv_hi"]) - c["log_adv"]
    s = 1.0 / (1.0 + math.exp(-(la - ka) / 0.5))
    terms = {"1": 1.0, "la": la, "ls": ls, "lp": lp, "e": etf, "e_la": etf * la, "e_ls": etf * ls,
             "e_lp": etf * lp, "e_hinge_hi": etf * max(la - ka, 0.0), "hinge_la_hi": max(la - ka, 0.0),
             "hinge_lp_lo": min(lp - (math.log(kn["price_lo"]) - c["log_price"]), 0.0),
             "e_s": etf * s, "e_s_la": etf * s * (la - ka), "e_s_ls": etf * s * ls, "e_s_lp": etf * s * lp,
             "e_hinge_vhi": etf * max(la - (math.log(kn["adv_vhi"]) - c["log_adv"]), 0.0)}
    x = sum(b[t] * terms[t] for t in CAL["terms"])
    if mult is None:
        mult = float(M.era_multiplier([np.datetime64(pd.Timestamp(date))], relative_adv=[math.exp(la + c["log_adv"])])[0])
    L = math.exp(x) * mult * price
    S = tick * (1 + (L / tick + CAL["c_d"]) ** CAL["q"]) ** (1 / CAL["q"])
    return S / (2 * price)


# ------------------------------------------------------------------ known answers
def test_formula_matches_hand_calculation():
    h, why = M.half_spread("XYZ", "2026-07-01", 50.0, 2e7, 0.02, "STK")
    assert h == pytest.approx(_hand(50.0, 2e7, 0.02, 0, "2026-07-01", 0.01), rel=1e-9)
    assert why.startswith("calibrated spread model (T2)")
    nomono = CalibratedSpreadModel(CalibratedSpreadConfig(etf_monotone=False))
    h2, _ = nomono.half_spread("ETFX", "2026-07-01", 300.0, 5e9, 0.01, "ETF")
    assert h2 == pytest.approx(_hand(300.0, 5e9, 0.01, 1, "2026-07-01", 0.01), rel=1e-9)
    assert M.half_spread("ETFX", "2026-07-01", 300.0, 5e9, 0.01, "ETF")[0] <= h2 + 1e-15   # running min


def test_era_multiplier_is_one_on_fit_day():
    assert M.era_multiplier([np.datetime64(pd.Timestamp(CAL["fit_day"]))])[0] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.skipif(not NBBO.exists(), reason="calibration dataset not shipped")
def test_calibration_points_spy_qqq_iwm():
    """Observed 2026-07-01 NBBO RTH time-weighted half spreads (TAQ) vs model on PIT inputs."""
    df = pd.read_csv(NBBO, dtype={"symbol": str})
    for sym in ("SPY", "QQQ", "IWM"):
        r = df[df.symbol == sym].iloc[0]
        h, _ = M.half_spread(sym, CAL["fit_day"], r.prev_close, r.adv_dollar, r.sigma, "ETF")
        assert abs(math.log(h * 1e4 / r.tw_hs_bp)) < math.log(2.0), sym   # within the fit's dispersion
        assert h * 1e4 < 1.5                                            # SPY-class ETFs: ~0.1-1 bp, never 10+ bp
        cs_half = 0.5e4 * r.cs_full
        assert cs_half > 5 * r.tw_hs_bp                                  # the bias this model fixes


@pytest.mark.skipif(not NBBO.exists(), reason="calibration dataset not shipped")
def test_calibration_dataset_median_error_small():
    df = pd.read_csv(NBBO, dtype={"symbol": str})
    df = df[df.sec_type.isin(["A", "ETF"]) & (df.rth_valid_s > 0.8 * 23400) & (df.prev_close >= 1)
            & np.isfinite(df.tw_hs_bp) & (np.abs(np.log(df.tw_mid / df.prev_close)) < 0.3)]
    h, _ = M.half_spread_vec(np.full(len(df), np.datetime64(pd.Timestamp(CAL["fit_day"]))),
                             df.prev_close.to_numpy(), df.adv_dollar.to_numpy(), df.sigma.to_numpy(),
                             (df.sec_type == "ETF").to_numpy(float))
    lr = np.log(h * 1e4 / df.tw_hs_bp.to_numpy())
    assert abs(np.median(lr)) < 0.15
    assert np.median(np.abs(lr)) < math.log(1.6)


# ------------------------------------------------------------------ tick eras
def test_era_ticks():
    assert era_tick("1996-05-01", 50.0) == 0.125
    assert era_tick("1996-05-01", 0.8) == 0.0625
    assert era_tick("1999-03-01", 50.0) == 0.0625
    assert era_tick("1999-03-01", 0.5) == pytest.approx(1 / 32)
    assert era_tick("2001-02-15", 50.0) == 0.01
    assert era_tick("2001-02-15", 50.0, listing="NASDAQ") == 0.0625    # Nasdaq decimal 2001-04-09
    assert era_tick("2001-04-09", 50.0, listing="NASDAQ") == 0.01
    assert era_tick("1996-05-01", 8.0, listing="NASDAQ") == pytest.approx(1 / 32)
    assert era_tick("2005-03-01", 0.5) == 0.01
    assert era_tick("2006-03-01", 0.5) == 0.0001
    assert era_tick(pd.Timestamp("1997-06-24 15:00", tz="America/New_York"), 20.0) == 0.0625
    assert era_tick(pd.Timestamp("1997-06-23 15:00", tz="America/New_York"), 20.0) == 0.125


@pytest.mark.parametrize("date,tick", [("1996-06-03", 0.125), ("1999-06-01", 0.0625), ("2005-06-01", 0.01)])
def test_tick_floor_by_era(date, tick):
    """Even a hyper-liquid name pays at least half a tick of its era; 1/8 era => >= 6.25c/2."""
    p = 40.0
    h, _ = M.half_spread("BIG", date, p, 5e10, 0.005, "STK")
    assert h >= tick / (2 * p) - 1e-12
    assert h == pytest.approx(_hand(p, 5e10, 0.005, 0, date, tick), rel=1e-9)
    # tick-constrained: within ~1.5 ticks for a very liquid name
    assert h * 2 * p / tick < 1.5


def test_eighths_wider_than_sixteenths_wider_than_decimal():
    args = (40.0, 2e8, 0.015, "STK")
    h96 = M.half_spread("X", "1996-06-03", *args)[0]
    h99 = M.half_spread("X", "1999-06-01", *args)[0]
    h05 = M.half_spread("X", "2005-06-01", *args)[0]
    assert h96 > h99 > h05


# ------------------------------------------------------------------ monotonicity
def test_more_liquid_is_tighter():
    advs = np.logspace(5, 10.5, 30)
    h, _ = M.half_spread_vec(np.full(30, np.datetime64("2024-03-01")), np.full(30, 30.0), advs,
                             np.full(30, 0.02), np.zeros(30))
    assert np.all(np.diff(h) <= 1e-15)
    assert h[0] > 5 * h[-1]


def test_more_volatile_is_wider():
    for etf in ("STK", "ETF"):
        hs = [M.half_spread("X", "2024-03-01", 30.0, 5e7, s, etf)[0] for s in (0.005, 0.01, 0.02, 0.04, 0.08)]
        assert np.all(np.diff(hs) > 0)


def test_etf_more_liquid_is_tighter_and_spy_class_is_one_tick():
    advs = np.logspace(6, 10.7, 25)
    h, _ = M.half_spread_vec(np.full(25, np.datetime64("2024-03-01")), np.full(25, 450.0), advs,
                             np.full(25, 0.01), np.ones(25))
    assert np.all(np.diff(h) <= 1e-15)
    assert h[-1] * 2 * 450.0 == pytest.approx(0.01, rel=0.05)      # SPY-like: one-cent spread


# ------------------------------------------------------------------ PIT and vectorised == scalar
def _bars(n=80, seed=0):
    rng = np.random.default_rng(seed)
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    h = c * (1 + np.abs(rng.normal(0, 0.006, n)))
    l = c * (1 - np.abs(rng.normal(0, 0.006, n)))
    v = rng.uniform(5e5, 2e6, n)
    d = pd.bdate_range("2024-01-02", periods=n)
    return h, l, c, v, d


def test_series_is_point_in_time():
    h, l, c, v, d = _bars()
    s1 = M.series(h, l, c, raw_close=c, volume=v, dates=d, asset_class="STK")
    k = 50
    h2, l2, c2, v2 = h.copy(), l.copy(), c.copy(), v.copy()
    h2[k:] *= 3; l2[k:] *= 0.5; c2[k:] *= 2; v2[k:] *= 100        # perturb sessions >= k
    s2 = M.series(h2, l2, c2, raw_close=c2, volume=v2, dates=d, asset_class="STK")
    np.testing.assert_array_equal(s1.est[:k + 1], s2.est[:k + 1])   # estimate for session k uses <= k-1
    assert not np.allclose(s1.est[k + 2:], s2.est[k + 2:])


def test_vectorised_equals_scalar():
    h, l, c, v, d = _bars(seed=3)
    s = M.series(h, l, c, raw_close=c, volume=v, dates=d, asset_class="ETF")
    adv = pd.Series(c * v).rolling(20, min_periods=10).mean().shift(1).to_numpy()
    r = pd.Series(c).pct_change()
    sig = r.rolling(20, min_periods=10).std().shift(1).to_numpy()
    for i in (25, 50, 79):        # full 20-return windows (shorter windows shrink sigma to the prior)
        hs, why = M.half_spread("E", d[i], c[i - 1], adv[i], sig[i], "ETF")
        assert s.est[i] == pytest.approx(2 * hs, rel=1e-12)
        assert s.at(i)[1] == "calibrated"
    assert "sigma prior" in s.at(15)[1]
    assert np.isnan(s.est[0]) and s.src[0] == "default"


def test_series_without_volume_falls_back_to_cs():
    h, l, c, v, d = _bars()
    s = M.series(h, l, c)
    ref = SpreadModel().series(h, l, c)
    i = 40
    assert s.src[i] == "Corwin-Schultz"
    assert s.est[i] == pytest.approx(ref.est[i])


def test_full_spread_contract_and_tod():
    est = 2 * M.half_spread("X", "2024-03-01", 100.0, 1e9, 0.015, "STK")[0]
    assert M.full_spread(est, 100.0) == pytest.approx(max(est, 0.01 / 100))
    open_ = M.full_spread(est, 100.0, None, 1.0, 389.0, adv_dollars=1e9)
    mid = M.full_spread(est, 100.0, None, 200.0, 190.0, adv_dollars=1e9)
    assert open_ > 1.5 * mid                       # spreads are widest in the first minutes
    assert M.full_spread(est, 100.0, None, -10.0, 400.0) == pytest.approx(est * 3.0)
    # 1-tick floor at the fill price
    assert M.full_spread(1e-7, 5.0) == pytest.approx(0.01 / 5.0)
    # profile averages to ~1 over the session
    prof = [M.tod_multiplier(m + 2.5, 390 - m - 2.5, 1e9) for m in range(0, 390, 5)]
    assert np.mean(prof) == pytest.approx(1.0, rel=0.02)


def test_default_and_cap_and_effective():
    h, why = M.half_spread("X", "2024-03-01", 30.0, float("nan"), 0.02, "STK")
    assert why == "default" and h == CalibratedSpreadConfig().default_half_spread
    lo = CAL["calib_range"]["adv_min"] / 10
    raw, why = M.half_spread("X", "2024-03-01", 3.0, lo, 0.06, "STK")
    assert "ADV below calibration range" in why
    capped, _ = M.half_spread("X", "2024-03-01", 3.0, lo, 0.06, "STK", cs_half=0.002)
    assert capped <= max(0.002, 0.005 / 3.0) + 1e-12 and capped <= raw
    eff = CalibratedSpreadModel(CalibratedSpreadConfig(measure="effective"))
    q = M.half_spread("X", "2024-03-01", 30.0, 5e7, 0.02, "STK")[0]
    e = eff.half_spread("X", "2024-03-01", 30.0, 5e7, 0.02, "STK")[0]
    assert 0.25 * q <= e <= q
    o = CalibratedSpreadModel(CalibratedSpreadConfig(overrides={"SPY": 0.00002}))
    assert o.half_spread("SPY", "2024-03-01", 500.0, 3e10, 0.01, "ETF") == (0.00002, "spread override (user)")


# ------------------------------------------------------------------ red-team fixes (v2)
def test_exchange_specific_tick_schedules():
    assert era_tick("1999-06-01", 130.0, "AMEX", symbol="SPY") == pytest.approx(1 / 64)   # SPDRs in 1/64
    assert era_tick("1996-06-03", 66.0, None, symbol="SPY") == pytest.approx(1 / 64)
    assert era_tick("1996-06-03", 40.0, "AMEX", symbol="MDY") == pytest.approx(1 / 64)
    assert era_tick("2001-01-29", 130.0, "AMEX", symbol="SPY") == 0.01
    assert era_tick("1996-06-03", 5.0, "AMEX") == 0.0625 and era_tick("1996-06-03", 20.0, "AMEX") == 0.125
    assert era_tick("1997-05-07", 20.0, "AMEX") == 0.0625
    assert era_tick("2001-01-26", 50.0, "NYSE") == 0.0625 and era_tick("2001-01-29", 50.0, "NYSE") == 0.01
    assert era_tick("2001-03-30", 50.0, "NASDAQ") == 0.0625 and era_tick("2001-04-09", 50.0, "NASDAQ") == 0.01
    assert era_tick("2005-10-03", 0.5) == 0.01 and era_tick("2006-01-31", 0.5) == 0.0001     # Rule 612
    assert era_tick("2000-09-01", 50.0, "NYSE", decimal_date="2000-08-28") == 0.01            # pilot stock
    pilot = CalibratedSpreadModel(CalibratedSpreadConfig(decimal_dates={"FDX": "2000-08-28"}))
    assert pilot.half_spread("FDX", "2000-09-01", 40.0, 5e7, 0.02, "STK", exchange="NYSE")[0] < \
        M.half_spread("FDX", "2000-09-01", 40.0, 5e7, 0.02, "STK", exchange="NYSE")[0]


def test_unknown_exchange_is_flagged_pre_decimal_only():
    assert "listing assumed NYSE" in M.half_spread("X", "1999-06-01", 30.0, 5e7, 0.02, "STK")[1]
    assert "listing assumed" not in M.half_spread("X", "1999-06-01", 30.0, 5e7, 0.02, "STK", exchange="NASDAQ")[1]
    assert "listing assumed" not in M.half_spread("X", "2012-06-01", 30.0, 5e7, 0.02, "STK")[1]
    nas = M.half_spread("X", "1999-06-01", 8.0, 5e6, 0.03, "STK", exchange="NASDAQ")[0]
    nyse = M.half_spread("X", "1999-06-01", 8.0, 5e6, 0.03, "STK", exchange="NYSE")[0]
    assert nas < nyse                       # Nasdaq < $10 quoted in 1/32 vs NYSE 1/16


def test_liquid_etf_intraday_profile_and_full_spread_asset_class():
    o_etf = M.tod_multiplier(0, 390, 1e10, "ETF"); o_stk = M.tod_multiplier(0, 390, 1e10, "STK")
    assert 1.4 < o_etf < 2.5 < o_stk
    assert M.tod_multiplier(388, 2, 1e10, "ETF") > M.tod_multiplier(388, 2, 1e10, "STK")
    est = 2e-5
    assert M.full_spread(est, 500.0, 0.01, 1.0, 389.0, adv_dollars=1e10, asset_class="ETF") == \
        pytest.approx(max(est * o_etf, 0.01 / 500))


def test_split_day_and_prior_close():
    n = 40
    rc = np.where(np.arange(n) < 30, 500.0, 50.0)
    adj = np.full(n, 50.0)
    d = pd.bdate_range("2024-01-02", periods=n)
    a = M.series(adj * 1.01, adj * 0.99, adj, raw_close=rc, volume=np.where(np.arange(n) < 30, 2e5, 2e6), dates=d)
    pc = np.r_[np.nan, rc[:-1]]; pc[30] = 50.0
    b = M.series(adj * 1.01, adj * 0.99, adj, raw_close=rc, volume=np.where(np.arange(n) < 30, 2e5, 2e6), dates=d,
                 prior_close=pc)
    assert a.est[30] == pytest.approx(b.est[30]) == pytest.approx(a.est[31], rel=0.05)


def test_asset_class_unknown_is_flagged():
    h, why = M.half_spread("X", "2024-03-01", 30.0, 5e7, 0.02, None)
    assert "asset class inferred STK" in why and h == M.half_spread("X", "2024-03-01", 30.0, 5e7, 0.02, "STK")[0]


def test_relative_liquidity_and_stress_flag():
    assert abs(M.log_deflator([np.datetime64(pd.Timestamp(CAL["fit_day"]))])[0]) < 1e-5
    assert M.log_deflator([np.datetime64("2003-06-01")])[0] > 2.5          # market $ volume ~19x lower in 2003
    # era non-increasing in t at fixed RELATIVE ADV$, >= 1 before 2010
    for rel in (1e5, 3e6, 1e8, 1e10):
        seq = [float(M.era_multiplier([np.datetime64(f"{y}-06-01")], relative_adv=[rel])[0]) for y in range(1993, 2027)]
        assert all(x >= y - 1e-12 for x, y in zip(seq, seq[1:])) and min(seq) >= 1 - 1e-12
    h, why = M.half_spread("SPY", "2020-03-18", 252.8, 6.8e10, 0.0517, "ETF")
    assert why.endswith("stress regime: spread uncertainty high")
    assert "stress" not in M.half_spread("SPY", "2020-01-15", 330.0, 2.5e10, 0.006, "ETF")[1]
    assert "stress" not in M.half_spread("X", "2024-03-01", 30.0, 5e7, 0.02, "STK")[1]
