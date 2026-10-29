"""synthetic_option_quotes: real -> same-day-surface anchor -> synthetic, PIT, expiry-consistent."""
import numpy as np
import pandas as pd
import pytest

from atlasbt.core import clock
from atlasbt.core.types import ExerciseStyle, Fidelity, Instrument, Right, SettlementType
from atlasbt.options.marking import QUOTE_COLS, synthetic_option_quotes
from atlasbt.options.synthetic import CBOE_DIR, SyntheticChainGenerator

pytestmark = pytest.mark.skipif(not (CBOE_DIR / "VIX_History.csv").exists(), reason="Cboe index files missing")


class FakePortal:
    """synthetic_* mechanics fixture: constant spot, flat rate, optional real data."""

    def __init__(self, spot, real_quotes=None, real_chain=None):
        self.spot, self.real_quotes, self.real_chain = spot, real_quotes, real_chain
        self.seen_as_of = []

    def _spot(self, u, ts):
        self.seen_as_of.append(ts)
        return self.spot

    def _div_yield(self, u, ts, spot):
        return 0.015

    def risk_free_rate(self, ts, tenor="3M"):
        return 0.05

    def option_quotes(self, instrument, start, end, as_of):
        if self.real_quotes is None:
            return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="ts"))
        q = self.real_quotes
        return q[(q.index > start) & (q.index <= min(end, as_of))]

    def option_chain(self, underlying, as_of, synthetic=False, **kw):
        if self.real_chain is None:
            return pd.DataFrame()
        ch = self.real_chain
        snap = pd.Timestamp(ch["ts"].iloc[0])
        if snap > as_of:
            return pd.DataFrame()
        ch = ch.copy()
        ch.attrs.update(snapshot_ts=snap, synthetic=False)
        return ch


def spxw(expiry="2023-06-30", K=4400.0, right="P"):
    return Instrument.option("SPX", pd.Timestamp(expiry), K, Right(right), ExerciseStyle.EUROPEAN,
                             SettlementType.CASH_PM, root="SPXW")


def test_pure_synthetic_matches_chain_and_is_pit():
    gen = SyntheticChainGenerator()
    ins = spxw()
    p = FakePortal(4400.0)
    start, end = clock.et("2023-06-12", "17:00"), clock.et("2023-06-16", "17:00")
    q = synthetic_option_quotes(ins, start, end, as_of=clock.et("2023-06-15", "16:30"), portal=p, generator=gen)
    assert list(q.columns) == QUOTE_COLS
    assert len(q) == 3                                   # closes 06-13, 06-14, 06-15 (as_of cuts 06-16)
    assert (q["fidelity"] == int(Fidelity.T3_MODELED)).all() and q.attrs["synthetic"]
    assert all(ts <= clock.et("2023-06-15", "16:30") for ts in p.seen_as_of)   # inputs only known at snapshot
    t = q.index[0]
    ch = gen.chain("SPX", t, 4400.0, float(np.log(1 + 0.05 * 91 / 360) * 365 / 91), 0.015, max_dte=30)
    row = ch[(ch["root"] == "SPXW") & (ch["expiry"] == ins.expiry) & (ch["strike"] == 4400.0) & (ch["right"] == "P")]
    assert len(row) == 1
    assert q["bid"].iloc[0] == pytest.approx(row["bid"].iloc[0]) and q["ask"].iloc[0] == pytest.approx(row["ask"].iloc[0])
    assert "params_fit_post_as_of" in q["reason"].iloc[0]


def test_stops_at_last_trading_close_pm_and_am():
    p = FakePortal(4400.0)
    q = synthetic_option_quotes(spxw("2023-06-14"), clock.et("2023-06-12", "17:00"), clock.et("2023-06-20", "17:00"),
                                clock.et("2023-06-20", "17:00"), p)
    assert q.index.max() == clock.nyse().open_close(pd.Timestamp("2023-06-14"))[1]   # expiry-day close (PM)
    am = Instrument.option("SPX", pd.Timestamp("2023-06-15"), 4400.0, Right.CALL, ExerciseStyle.EUROPEAN,
                           SettlementType.CASH_AM, root="SPX")
    q2 = synthetic_option_quotes(am, clock.et("2023-06-12", "17:00"), clock.et("2023-06-20", "17:00"),
                                 clock.et("2023-06-20", "17:00"), p)
    assert q2.index.max() == clock.nyse().open_close(pd.Timestamp("2023-06-15"))[1]  # Thursday LTD; SOQ via lifecycle


def test_real_first_then_anchored_then_synthetic():
    ins = Instrument.option("SPY", pd.Timestamp("2023-06-23"), 430.0, Right.PUT)
    t1 = clock.nyse().open_close(pd.Timestamp("2023-06-12"))[1]
    t2 = clock.nyse().open_close(pd.Timestamp("2023-06-13"))[1]
    real = pd.DataFrame({"bid": [3.10], "ask": [3.15], "mid": [3.125], "iv": [0.15], "delta": [-0.4],
                         "underlying_price": [432.0], "fidelity": [2], "source": ["dolthub"]},
                        index=pd.DatetimeIndex([t1], name="ts"))
    from tests.options.conftest import synthetic_chain
    ch = synthetic_chain(spot=432.0, session="2023-06-13", expiries=(("2023-07-14", "PHYSICAL"), ("2023-08-18", "PHYSICAL")),
                         strikes=np.arange(400, 461, 5.0), underlying="SPY")
    ch["ts"] = t2
    ch["fidelity"] = 2
    p = FakePortal(432.0, real_quotes=real, real_chain=ch)
    q = synthetic_option_quotes(ins, clock.et("2023-06-11", "17:00"), clock.et("2023-06-14", "17:00"),
                                clock.et("2023-06-14", "17:00"), p)
    assert q.loc[t1, "reason"] == "real quote" and q.loc[t1, "bid"] == 3.10
    assert q.loc[t2, "source"] == "anchored_real_surface"                    # contract absent (< 13 DTE)
    assert q.loc[t2, "fidelity"] in (int(Fidelity.T2_PROXY), int(Fidelity.T3_MODELED))
    t3 = clock.nyse().open_close(pd.Timestamp("2023-06-14"))[1]
    assert q.loc[t3, "source"] in ("synthetic", "anchored_real_surface")    # chain snapshot from 06-13 is not same-day
    assert q.loc[t3, "source"] == "synthetic"


def test_include_open_and_empty_window():
    p = FakePortal(4400.0)
    q = synthetic_option_quotes(spxw(), clock.et("2023-06-12", "17:00"), clock.et("2023-06-13", "17:00"),
                                clock.et("2023-06-13", "17:00"), p, include_open=True)
    assert len(q) == 2
    e = synthetic_option_quotes(spxw(), clock.et("2023-07-03", "17:00"), clock.et("2023-07-10", "17:00"),
                                clock.et("2023-07-10", "17:00"), p)
    assert e.empty and e.attrs["fidelity"] == int(Fidelity.NA)
