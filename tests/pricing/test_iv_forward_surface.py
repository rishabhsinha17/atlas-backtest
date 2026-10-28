import numpy as np
import pandas as pd
import pytest

from atlasbt.pricing import (american_implied_vol, bs2002_price, bsm_price, deamericanize, implied_vol,
                             infer_forwards, parity_forward)
from atlasbt.pricing.surface import (atm_iv_per_expiry, calendar_arbitrage_free, constant_maturity_iv, fit_smile,
                                     fit_svi, interp_total_variance, skew_25d, smile_frame, svi_g, SVIFit)


def test_iv_roundtrip_vectorized():
    rng = np.random.default_rng(3)
    n = 400
    S = rng.uniform(20, 500, n)
    K = S * np.exp(rng.normal(0, 0.25, n))
    T = rng.uniform(1 / 365, 2.0, n)
    r, q = rng.uniform(0, 0.06, n), rng.uniform(0, 0.03, n)
    v = rng.uniform(0.05, 1.5, n)
    cp = rng.random(n) < 0.5
    px = bsm_price(S, K, T, r, q, v, cp)
    res = implied_vol(px, S, K, T, r, q, cp)
    ok = res.ok
    # vega-negligible deep OTM/ITM cases may legitimately be unsolvable at machine precision
    assert ok.mean() > 0.97
    # price round-trip everywhere; vol round-trip where vega is not negligible (IV identifiable)
    rep = bsm_price(S[ok], K[ok], T[ok], r[ok], q[ok], res.iv[ok], cp[ok])
    np.testing.assert_allclose(rep, px[ok], rtol=1e-8, atol=1e-8)
    from atlasbt.pricing import bsm_greeks
    vega = bsm_greeks(S, K, T, r, q, v, cp).vega * 100
    sens = ok & (vega > 1e-3 * S)
    np.testing.assert_allclose(res.iv[sens], v[sens], rtol=1e-5, atol=1e-6)


def test_iv_reason_codes():
    r = implied_vol([1.0, 13.0, -1, 5.0, 101.0], 100, [90, 90, 100, 100, 100], [0.5, 0.5, 0.5, 0.0, 0.5], 0.02, 0.0,
                    ["C", "C", "C", "C", "C"])
    assert list(r.reason) == ["BELOW_INTRINSIC", "OK", "NONPOSITIVE_PRICE", "EXPIRED", "ABOVE_MAX"]
    assert np.isnan(r.iv[0]) and np.isfinite(r.iv[1])


def test_black76_iv():
    from atlasbt.pricing import black76_price
    p = black76_price(19, 19, 0.75, 0.10, 0.28, "C")
    assert implied_vol(p, 19, 19, 0.75, 0.10, 0.0, "C", model="black76").iv == pytest.approx(0.28, abs=1e-8)


def test_american_iv_roundtrip_and_deamericanize():
    p = bs2002_price(100, 110, 0.5, 0.05, 0.0, 0.3, "P")
    res = american_implied_vol(p, 100, 110, 0.5, 0.05, 0.0, "P")
    assert res.iv[()] == pytest.approx(0.3, abs=1e-6)
    eu, _ = deamericanize(p, 100, 110, 0.5, 0.05, 0.0, "P")
    assert eu == pytest.approx(bsm_price(100, 110, 0.5, 0.05, 0, 0.3, "P"), abs=1e-6)
    assert eu < p
    r2 = american_implied_vol(9.0, 100, 110, 0.5, 0.05, 0.0, "P")  # below intrinsic 10
    assert r2.reason[()] == "BELOW_INTRINSIC"


def _synthetic_chain(S=100.0, r=0.03, q=0.01, vols=None, T=(30 / 365, 60 / 365), half_spread=0.02):
    """synthetic_* fixture: European chain generated from a known smile (mechanics test only)."""
    rows = []
    for t in T:
        F = S * np.exp((r - q) * t)
        for K in np.arange(70, 131, 2.5):
            k = np.log(K / F)
            v = 0.2 - 0.3 * k + 0.5 * k * k if vols is None else vols(k, t)
            for cp in ("C", "P"):
                m = float(bsm_price(S, K, t, r, q, v, cp))
                rows.append(dict(expiry=pd.Timestamp("2024-01-01") + pd.Timedelta(days=round(t * 365)), T=t,
                                 strike=K, right=cp, bid=max(m - half_spread, 0.0), ask=m + half_spread, mid=m,
                                 underlying_price=S, exercise="EUROPEAN"))
    return pd.DataFrame(rows)


def test_parity_forward_recovers_carry():
    ch = _synthetic_chain()
    fw = infer_forwards(ch, r=0.03, american=False)
    for _, row in fw.iterrows():
        assert row["F"] == pytest.approx(100 * np.exp(0.02 * row["T"]), abs=1e-6)
        assert row["q_implied"] == pytest.approx(0.01, abs=1e-6)


def test_parity_forward_american_deamericanized():
    # American chain priced with BS2002 at known q; raw parity is biased, de-Americanized should be closer
    S, r, q, t = 100.0, 0.05, 0.0, 0.25
    rows = []
    for K in np.arange(80, 121, 2.5):
        for cp in ("C", "P"):
            m = float(bs2002_price(S, K, t, r, q, 0.25, cp))
            rows.append(dict(expiry=pd.Timestamp("2024-04-01"), T=t, strike=K, right=cp, bid=m - 0.01, ask=m + 0.01,
                             underlying_price=S, exercise="AMERICAN"))
    ch = pd.DataFrame(rows)
    raw = infer_forwards(ch, r=r, american=False).iloc[0]
    adj = infer_forwards(ch, r=r, american=True).iloc[0]
    true_F = S * np.exp(r * t)
    assert abs(adj["F"] - true_F) < abs(raw["F"] - true_F)
    assert abs(adj["q_implied"]) < 0.004


def test_constant_maturity_total_variance():
    v, flag = interp_total_variance([20 / 365, 40 / 365], [0.20, 0.30], 30 / 365)
    w = (0.2 ** 2 * 20 + 0.3 ** 2 * 40) / 2 / 365
    assert flag == "interp" and v == pytest.approx(np.sqrt(w / (30 / 365)))
    v2, f2 = interp_total_variance([20 / 365, 40 / 365], [0.20, 0.30], 90 / 365)
    assert f2 == "extrap_flat" and v2 == pytest.approx(0.30)


def test_smile_atm_and_skew_from_synthetic_chain():
    ch = _synthetic_chain()
    fw = infer_forwards(ch, r=0.03, american=False)
    sm = smile_frame(ch, fw)
    atm = atm_iv_per_expiry(sm)
    np.testing.assert_allclose(atm["atm_iv"], 0.20, atol=2e-3)
    v30, flag = constant_maturity_iv(atm, 30)
    assert v30 == pytest.approx(0.20, abs=2e-3)
    sk, _ = skew_25d(sm, 30)
    assert sk > 0.02  # put skew positive for downward-sloping smile


def test_svi_fit_and_arbitrage_checks():
    k = np.linspace(-0.4, 0.3, 25)
    true = SVIFit(0.01, 0.1, -0.5, 0.02, 0.15, 0.25, 0, True)
    iv = true.iv(k)
    fit = fit_svi(k, iv, 0.25)
    assert fit.ok and fit.rmse_vol < 1e-3
    assert np.min(svi_g(fit, np.linspace(-1, 1, 101))) >= 0
    # calendar: scaling total variance up with T is arbitrage-free; reverse is not
    f2 = SVIFit(0.02, 0.12, -0.5, 0.02, 0.15, 0.5, 0, True)
    assert calendar_arbitrage_free([fit, f2])
    assert not calendar_arbitrage_free([SVIFit(0.05, 0.1, -0.5, 0, 0.1, 0.1, 0, True), SVIFit(0.0, 0.1, -0.5, 0, 0.1, 0.5, 0, True)])
    # butterfly-arbitrage smile is flagged (strongly negative curvature in w)
    bad = fit_svi(k, np.sqrt(np.maximum(0.06 - 0.5 * k ** 2, 1e-4) / 0.25), 0.25)
    assert not bad.ok


def test_fit_smile_fallback_spline():
    ch = _synthetic_chain(T=(30 / 365,))
    fw = infer_forwards(ch, r=0.03, american=False)
    sm = smile_frame(ch, fw).query("strike >= 92.5 and strike <= 105")
    m = fit_smile(sm)
    assert m.kind in ("svi", "spline_delta")
