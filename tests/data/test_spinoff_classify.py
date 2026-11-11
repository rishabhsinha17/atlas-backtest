"""Spin-offs encoded as split factors (Yahoo and Tiingo): one SPINOFF row per event, never a SPLIT
next to it, and classification is idempotent (Yahoo actions are stored classified and classified
again on read)."""
import numpy as np
import pandas as pd

from atlasbt.data.actions_classify import classify
from atlasbt.data.providers import tiingo


def _bars(days, opens, closes):
    idx = pd.DatetimeIndex(pd.to_datetime(days))
    return pd.DataFrame({"open": opens, "high": closes, "low": opens, "close": closes, "volume": 1e6}, index=idx)


def _split(day, ratio):
    return {"type": "SPLIT", "ex_date": pd.Timestamp(day), "known_at": pd.Timestamp(day, tz="UTC"),
            "ratio": ratio, "amount": 0.0, "fidelity": 1}


def _child(px):
    return lambda sym, ex: px


def test_known_spinoff_with_near_common_factor_is_not_also_a_split():
    # GE 2024-04-02: factor 1.253 is within rounding of 5:4, so it used to stay a SPLIT next to the SPINOFF
    bars = _bars(["2024-04-01", "2024-04-02"], [175.0, 140.0], [174.6, 136.0])
    a = classify("GE", pd.DataFrame([_split("2024-04-02", 1.253)]), bars, _child(140.0))
    assert list(a["type"]) == ["SPINOFF"]
    assert np.isclose(a["amount"].iloc[0], 35.0)


def test_classify_is_idempotent_on_known_spinoffs():
    bars = _bars(["2023-01-03", "2023-01-04"], [100.0, 80.0], [100.0, 81.0])
    once = classify("GE", pd.DataFrame([_split("2023-01-04", 1.281)]), bars)          # no child price (fetch time)
    twice = classify("GE", once, bars, _child(58.1))                                   # on read, with child price
    sp = twice[twice["type"] == "SPINOFF"]
    assert len(sp) == 1 and np.isclose(sp["amount"].iloc[0], 58.1 / 3)
    assert not (twice["type"] == "SPLIT").any()


def test_tiingo_unknown_spinoff_factor_becomes_spinoff_and_passes_the_drop_check():
    # FTI 2021-02-16: Tiingo splitFactor 1.344 (Technip Energies), not a common split ratio
    bars = _bars(["2021-02-12", "2021-02-16"], [10.0, 8.2], [10.0, 8.3])
    acts = tiingo.classified_actions(object(), "FTI", bars, pd.DataFrame([_split("2021-02-16", 1.344)]))
    assert list(acts["type"]) == ["SPINOFF"]
    assert np.isclose(acts["amount"].iloc[0], 10.0 * (1 - 1 / 1.344))
    chk = tiingo._spinoff_check(bars, acts)
    assert len(chk) == 1 and chk[0]["ok"]


def test_tiingo_spinoff_check_nets_out_a_same_day_reverse_split():
    # DD 2019-06-03: 1:3 reverse split + Corteva -> open/prev close 2.15, x 1/3 = 0.72 (a drop)
    bars = _bars(["2019-05-31", "2019-06-03"], [30.0, 66.0], [30.52, 76.10])
    acts = pd.DataFrame([{**_split("2019-06-03", 1 / 3)},
                         {"type": "SPINOFF", "ex_date": pd.Timestamp("2019-06-03"), "known_at": None,
                          "ratio": np.nan, "amount": 8.27, "fidelity": 2}])
    chk = tiingo._spinoff_check(bars, acts)
    assert len(chk) == 1 and chk[0]["ok"] and np.isclose(chk[0]["split_same_day"], 1 / 3)
    # without the split the same gap would be a price RISE -> rejected
    assert not tiingo._spinoff_check(bars, acts[acts["type"] == "SPINOFF"])[0]["ok"]
