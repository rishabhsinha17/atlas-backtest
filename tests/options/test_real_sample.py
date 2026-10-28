"""Real-data checks on the stored DoltHub calibration sample
(atlasbt/options/calibration/dolthub_sample.csv.gz, fetched 2026-10-28)."""
import json

import numpy as np
import pandas as pd
import pytest

from atlasbt.options.calibrate import SAMPLE_PATH, from_dolthub, snapshot_stats
from atlasbt.options.synthetic import PARAMS_PATH, SyntheticParams
from atlasbt.pricing.forward import infer_forwards
from atlasbt.pricing.surface import smile_frame

pytestmark = pytest.mark.skipif(not SAMPLE_PATH.exists(), reason="calibration sample missing")


@pytest.fixture(scope="module")
def sample():
    return from_dolthub(pd.read_csv(SAMPLE_PATH))


def test_sample_shape(sample):
    assert sample["date"].min() >= pd.Timestamp("2020-01-01")
    assert sample["dte"].min() >= 7 and sample["dte"].max() <= 80          # sparse expiry coverage
    assert (sample.groupby(["underlying", "date"])["expiry"].nunique() <= 5).all()


def test_our_iv_matches_vendor_iv_near_the_money(sample):
    spy = sample[sample["underlying"] == "SPY"]
    d = sorted(spy["date"].unique())[len(spy["date"].unique()) // 2]
    snap = spy[spy["date"] == d]
    st = snapshot_stats(snap, r=0.02)
    assert np.isfinite(st["spot"]) and 0.05 < st["atm30"] < 1.0
    snap = snap.assign(underlying_price=st["spot"])
    fw = infer_forwards(snap, r=0.02, american=True, iterations=1)
    sm = smile_frame(snap, fw)
    near = sm[sm["otm"] & (np.abs(sm["k"]) < 0.05) & np.isfinite(sm["iv_fit"])]
    assert len(near) >= 4
    assert np.median(np.abs(near["iv_fit"] - near["iv"])) < 0.02     # within 2 vol points of vendor IV


def test_parity_forwards_consistent_across_expiries(sample):
    spy = sample[sample["underlying"] == "SPY"]
    for d, snap in list(spy.groupby("date"))[:4]:
        st = snapshot_stats(snap, r=0.01)
        fw = infer_forwards(snap.assign(underlying_price=st["spot"]), r=0.01, american=True, iterations=1)
        fw = fw[np.isfinite(fw["F"])]
        # implied carry (dividend yield + borrow) of SPY within a sane band
        assert fw["q_implied"].between(-0.03, 0.06).all()


def test_params_file_has_evidence():
    assert PARAMS_PATH.exists()
    d = json.loads(PARAMS_PATH.read_text())
    ev = d["evidence"]
    assert ev["spread_model"]["n_quotes"] > 1000 and ev["spread_model"]["r2"] > 0.5
    assert 0.6 < d["atm_ratio_index"] < 1.05
    p = SyntheticParams.load()
    assert p.skew_a0 < 0 and p.spread_symbol_fe
