"""Regression: internal OCC-style symbols for pre-2000 expiries decode to 19yy (runner finding)."""
import pandas as pd

from atlasbt.core.types import Instrument, Right, SettlementType, ExerciseStyle


def test_pre2000_and_post2000_symbols_round_trip():
    from atlasbt.data.portal import Portal
    p = Portal()
    for exp in ("1991-02-15", "1999-12-17", "2016-02-18", "2069-01-18"):
        inst = Instrument.option("SPX", pd.Timestamp(exp), 330.0, Right.PUT,
                                 exercise=ExerciseStyle.EUROPEAN, settlement=SettlementType.CASH_AM)
        got = p.instrument(inst.symbol)
        assert pd.Timestamp(got.expiry).normalize() == pd.Timestamp(exp), (inst.symbol, got.expiry)
