"""Validation of the calibrated spread model against Alpaca SIP NBBO (atlasbt/execution/calibration/alpaca_validate.py).
Offline: synthetic_ rows exercise the mechanics; the shipped sample (derived statistics only) is checked for shape."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlasbt.execution.calibration import alpaca_validate as V
from atlasbt.execution.spread_calibrated import CalibratedSpreadModel


def _synthetic_sample(n_days=4):
    rows = []
    for i, d in enumerate(pd.bdate_range("2019-03-13", periods=n_days, freq="7D")):
        for sym, etf, adv, px, sig, obs in (("SYNA", 1, 3e10, 300.0, 0.01, 0.2), ("SYNB", 0, 2e8, 40.0, 0.02, 3.0)):
            for mso, mtc in ((2, 388), (182, 208), (382, 8)):
                rows.append({"date": d.strftime("%Y-%m-%d"), "mso": mso, "mtc": mtc, "symbol": sym, "etf": etf,
                             "prev_close": px, "adv_dollar": adv, "sigma": sig, "n_sigma": 19,
                             "tw_hs_bp": obs * (1.5 if mso == 2 else 1.0), "n_quotes": 50})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_synthetic_predict_matches_scalar_api():
    df = V.predict(_synthetic_sample(1))
    m = CalibratedSpreadModel()
    r = df.iloc[1]
    h, _ = m.half_spread(r["symbol"], r["date"], r["prev_close"], r["adv_dollar"], r["sigma"],
                         asset_class="ETF" if r["etf"] else "STK", minutes_since_open=r["mso"],
                         minutes_to_close=r["mtc"], exchange="NYSE")
    # the engine's full_spread floors the time-of-day adjusted spread at one tick
    assert df["pred_bp"].iloc[1] == pytest.approx(max(h, 0.005 / r["prev_close"]) * 1e4, rel=1e-9)
    assert (df["pred_bp"] > 0).all() and (df["tod"] > 0).all()


def test_synthetic_report_known_answer():
    d = V.predict(_synthetic_sample())
    d["pred_bp"] = d["tw_hs_bp"] * 2.0          # model exactly 2x the observed spread everywhere
    rep = V.report(d)
    assert rep["all"]["mean_log_ratio"] == pytest.approx(np.log(2), abs=1e-3)
    assert rep["all"]["sd_log_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert rep["all"]["within_x2"] == 1.0 and rep["all"]["share_pred_below_obs"] == 0.0
    assert set(rep["by_kind"]) == {"ETF", "stock"} and set(rep["by_year"]) == {"2019"}
    # observed time-of-day shape: the first window is 1.5x the others -> 1.5 / mean(1.5, 1, 1)
    assert rep["tod_shape"]["ETF open+2m"]["obs_median"] == pytest.approx(1.5 / (3.5 / 3), abs=1e-3)


def test_shipped_sample_shape_if_present():
    if not V.SAMPLE.exists():
        pytest.skip("alpaca_nbbo_2016_2026.csv.gz not built")
    df = V.load_sample()
    assert {"date", "mso", "symbol", "etf", "prev_close", "adv_dollar", "sigma", "tw_hs_bp"} <= set(df.columns)
    assert df["date"].min() >= pd.Timestamp("2016-01-04") and (df["tw_hs_bp"] > 0).all()


# ------------------------------------------------------------------------------------- v4 calibration file
from atlasbt.execution.spread_calibrated import CalibratedSpreadConfig, load_calibration  # noqa: E402

V3 = V.HERE / "spread_calibration_v3.json"


def _models():
    v4 = CalibratedSpreadModel()
    v3 = CalibratedSpreadModel(CalibratedSpreadConfig(calibration_path=str(V3)))
    rel = CalibratedSpreadModel(CalibratedSpreadConfig(pessimistic_floor_2010_2025=False))
    return v4, v3, rel


def test_loader_reads_v4_and_keeps_v3_compatible():
    cal = load_calibration()
    assert cal["version"] == 4 and cal["nbbo_evidence"]["start"] == "2016-01-04"
    old = load_calibration(str(V3))
    assert old["version"] == 3 and old["nbbo_evidence"] is None
    for k in ("beta", "era", "tod_profile", "market_volume"):
        assert cal[k] == old[k]                      # v4 only adds the evidence block
    for y, m in cal["nbbo_evidence"]["multipliers"].items():
        assert m["stock"] >= 1.0 and m["etf"] >= 1.0  # evidence can only raise the relative model


def test_v4_evidence_regime_replaces_the_floor_only_where_evidence_exists():
    v4, v3, rel = _models()
    ev = load_calibration()["nbbo_evidence"]
    hi_adv = 4 * ev["adv_band"][1]
    lo_adv = 0.5 * ev["adv_band"][0]
    # liquid S&P-like stock in 2020: relative model x multiplier, below the v3 nominal floor
    h4, why4 = v4.half_spread(None, "2020-06-01", 60.0, hi_adv, 0.02, "STK")
    h3, why3 = v3.half_spread(None, "2020-06-01", 60.0, hi_adv, 0.02, "STK")
    hr, _ = rel.half_spread(None, "2020-06-01", 60.0, hi_adv, 0.02, "STK")
    m = ev["multipliers"]["2020"]["stock"]
    assert "era checked against SIP NBBO" in why4 and "pessimistic floor" not in why4
    assert h4 == pytest.approx(max(hr * m, 0.005 / 60.0), rel=1e-6) and h4 <= h3
    # below the evidence ADV$ band, before 2016 and in 2026: exactly v3
    for d, adv in (("2020-06-01", lo_adv), ("2014-06-02", hi_adv), ("2026-07-01", hi_adv), ("2005-06-01", hi_adv)):
        assert v4.half_spread(None, d, 60.0, adv, 0.02, "STK") == v3.half_spread(None, d, 60.0, adv, 0.02, "STK")
    # nbbo_evidence=False reproduces v3 from the v4 file
    off = CalibratedSpreadModel(CalibratedSpreadConfig(nbbo_evidence=False))
    assert off.half_spread(None, "2020-06-01", 60.0, hi_adv, 0.02, "STK") == (h3, why3)


def test_v4_blend_is_monotone_across_the_band():
    v4, _, _ = _models()
    a0, a1 = load_calibration()["nbbo_evidence"]["adv_band"]
    advs = np.exp(np.linspace(np.log(a0) - 1, np.log(a1) + 1, 40))
    h, _ = v4.half_spread_vec(np.array([np.datetime64("2019-03-13")] * 40), np.full(40, 40.0), advs,
                              np.full(40, 0.02), np.zeros(40))
    assert (np.diff(h) <= 1e-15).all()             # more liquid is never more expensive
