"""validate_real.surface_bias: synthetic surface vs stored real OPRA quotes, by moneyness / DTE bucket.

Fixture: European SPXW quotes generated from Black-Scholes at a KNOWN vol, priced by a stub surface at a
different known vol, so the expected bias (vol points and dollars) is analytic. Offline, no Cboe files."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlasbt.options import validate_real as V
from atlasbt.pricing.bsm import bsm_price
from atlasbt.pricing.time import year_fraction

TS = pd.Timestamp("2024-01-10 20:59", tz="UTC")          # 15:59 ET
S = 4750.0
REAL_VOL, SYN_VOL = 0.15, 0.13


def _osi(root, exp, right, K):
    return f"{root:<6}{pd.Timestamp(exp):%y%m%d}{right}{int(round(K * 1000)):08d}"


def _real_rows(ts=TS, vol=REAL_VOL, spot=S, half_spread=0.05):
    """Stored-format rows (databento.QCOLS) of SPXW PM series: 36 and 64 DTE, strikes 0.94..1.06 S, C and P."""
    rows = []
    for exp in ("2024-02-15", "2024-03-14"):
        T = year_fraction(ts, pd.Timestamp(exp), "CASH_PM")
        for K in (4465.0, 4560.0, 4750.0, 4940.0, 5035.0):
            for right in "CP":
                px = float(bsm_price(spot, K, T, 0.0, 0.0, vol, right == "C"))
                rows.append({"ts": ts, "symbol": _osi("SPXW", exp, right, K), "bid": round(px - half_spread, 2),
                             "ask": round(px + half_spread, 2), "bid_size": 10, "ask_size": 10, "age_s": 1.0,
                             "chain": True})
    return pd.DataFrame(rows)


class FlatSurface:
    """Stub synthetic generator with SyntheticChainGenerator.quote_contract's signature (flat vol)."""

    def __init__(self, vol):
        self.vol, self.calls = vol, []

    def quote_contract(self, underlying, as_of, spot, r, q, expiry, strike, right, root=None, settlement=None,
                       rv_ratio=None, strict=False):
        self.calls.append((underlying, as_of, spot, root, settlement))
        T = year_fraction(as_of, pd.Timestamp(expiry), settlement)
        mid = float(bsm_price(spot, strike, T, r, q, self.vol, right == "C"))
        return {"mid": mid, "bid": mid - 0.05, "ask": mid + 0.05, "iv": self.vol, "underlying_price": spot,
                "reasons": []}


def test_surface_bias_vol_points_and_dollars_by_bucket():
    real = _real_rows()
    gen = FlatSurface(SYN_VOL)
    res = V.surface_bias(real, synthetic=gen, rates=lambda und, ts: (0.0, 0.0))
    det, summ = res.detail, res.summary
    # OTM only by default: puts K < F, calls K > F, ATM (K == F) both sides -> 2 expiries x 6 contracts
    assert len(det) == 12 and set(det["underlying"]) == {"SPX"} and set(det["root"]) == {"SPXW"}
    # spot for an index underlying = put-call parity of the same snapshot (r = q = 0 -> exactly S)
    assert det["spot"].to_numpy() == pytest.approx(S, abs=0.02)
    assert all(c[0] == "SPX" and c[1] == TS and c[3] == "SPXW" and c[4] == "CASH_PM" for c in gen.calls)
    # vol error = (synthetic iv - real iv) in vol points: -2.0 everywhere (quotes rounded to cents)
    assert det["err_volpts"].to_numpy() == pytest.approx(-2.0, abs=0.05)
    # dollar error = synthetic mid - real mid, analytic
    T = year_fraction(TS, pd.Timestamp(det["expiry"].iloc[0]), "CASH_PM")
    row = det.iloc[0]
    exp_err = float(bsm_price(row["spot"], row["strike"], T, 0, 0, SYN_VOL, row["right"] == "C")) - row["real_mid"]
    assert row["err_usd"] == pytest.approx(exp_err, abs=1e-6)
    assert (det["err_usd"] < 0).all()                       # a lower vol prices every option cheaper
    # buckets
    assert set(det["dte_bucket"]) == {"31-60", "61-120"}
    assert set(det["mny_bucket"]) == {"0.90-0.95", "0.98-1.02", "1.02-1.05", "1.05-1.10", "0.95-0.98"}
    s = summ.set_index(["underlying", "dte_bucket", "mny_bucket"])
    atm = s.loc[("SPX", "31-60", "0.98-1.02")]
    assert atm["n"] == 2 and atm["err_volpts_median"] == pytest.approx(-2.0, abs=0.05)
    assert atm["err_usd_median"] < 0 and atm["err_rel_median"] < 0
    assert int(summ["n"].sum()) == len(det)
    assert {"err_volpts_mean", "err_usd_mean", "real_spread_median", "syn_spread_median"} <= set(summ.columns)


def test_surface_bias_is_zero_on_the_true_vol_and_skips_unusable_quotes():
    real = _real_rows()
    # unusable quotes: crossed, no offer, expired at ts
    bad = pd.DataFrame([
        {"ts": TS, "symbol": _osi("SPXW", "2024-02-15", "P", 4600.0), "bid": 5.0, "ask": 4.0},
        {"ts": TS, "symbol": _osi("SPXW", "2024-02-15", "P", 4610.0), "bid": 0.0, "ask": 0.0},
        {"ts": TS, "symbol": _osi("SPXW", "2024-01-09", "P", 4600.0), "bid": 1.0, "ask": 1.1},
    ]).assign(bid_size=1, ask_size=1, age_s=1.0, chain=True)
    res = V.surface_bias(pd.concat([real, bad], ignore_index=True), synthetic=FlatSurface(REAL_VOL),
                         rates=lambda und, ts: (0.0, 0.0))
    assert len(res.detail) == 12
    assert res.detail["err_volpts"].abs().max() < 0.05
    assert res.detail["err_usd"].abs().max() < 0.006       # cent rounding of the fixture quotes only
    assert res.n_skipped == 3


def test_surface_bias_equity_needs_a_spot_and_uses_the_given_one():
    rows = []
    T = year_fraction(TS, pd.Timestamp("2024-02-16"), "PHYSICAL")
    for K in (440.0, 470.0, 500.0):
        for right in "CP":
            px = float(bsm_price(470.0, K, T, 0.0, 0.0, REAL_VOL, right == "C"))
            rows.append({"ts": TS, "symbol": _osi("SPY", "2024-02-16", right, K), "bid": round(px - 0.01, 2),
                         "ask": round(px + 0.01, 2), "bid_size": 1, "ask_size": 1, "age_s": 1.0, "chain": True})
    real = pd.DataFrame(rows)
    gen = FlatSurface(SYN_VOL)
    res = V.surface_bias(real, synthetic=gen, spot=lambda und, ts: 470.0, rates=lambda und, ts: (0.0, 0.0))
    assert set(res.detail["underlying"]) == {"SPY"} and len(res.detail) == 4
    assert (res.detail["spot"] == 470.0).all()
    # American OTM options at r = q = 0: early exercise is worthless -> same -2 vol pt bias
    assert res.detail["err_volpts"].to_numpy() == pytest.approx(-2.0, abs=0.1)
    # without a spot source an equity snapshot is skipped, not guessed
    res2 = V.surface_bias(real, synthetic=gen, rates=lambda und, ts: (0.0, 0.0))
    assert res2.detail.empty and res2.n_skipped == len(real)


def test_surface_bias_synthetic_time_offset():
    """syn_as_of lets the surface be built later (e.g. close + 20 min, once the same-day VIX close is known)."""
    gen = FlatSurface(SYN_VOL)
    V.surface_bias(_real_rows(), synthetic=gen, rates=lambda und, ts: (0.0, 0.0),
                   syn_as_of=lambda ts: ts + pd.Timedelta(minutes=21))
    assert {c[1] for c in gen.calls} == {TS + pd.Timedelta(minutes=21)}


def test_load_stored_quotes_closing_sample_and_root(tmp_path):
    from atlasbt.data.providers import databento as D
    rows = _real_rows()
    early = rows.assign(ts=TS - pd.Timedelta(minutes=30))
    spx_am = rows.iloc[:2].assign(symbol=[_osi("SPX", "2024-02-15", "C", 4750.0), _osi("SPX", "2024-02-15", "P", 4750.0)])
    D.write_rows(pd.concat([rows, early, spx_am], ignore_index=True).assign(underlying="SPX"), tmp_path)
    allr = V.load_stored_quotes("SPX", root=tmp_path)
    assert set(allr["ts"]) == {TS} and len(allr) == len(rows) + 2          # 15:59 ET only, both roots
    w = V.load_stored_quotes("SPXW", root=tmp_path)
    assert len(w) == len(rows) and set(w["symbol"].str[:4]) == {"SPXW"}
    res = V.surface_bias(w, synthetic=FlatSurface(SYN_VOL), rates=lambda und, ts: (0.0, 0.0))
    assert len(res.detail) == 12
    assert V.load_stored_quotes("QQQ", root=tmp_path).empty
