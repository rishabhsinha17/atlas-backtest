"""Yahoo parsing: un-adjusting split-adjusted prices (known answers), actions, calendar
alignment of daily bars (known_at = actual close incl. early closes), intraday sessions."""
import numpy as np
import pandas as pd
import pytest

from atlasbt.data import frames
from atlasbt.data.providers import yahoo

from .conftest import FIX, yahoo_frame


def _close(bars, day):
    return float(bars.loc[bars.index.normalize() == pd.Timestamp(day, tz="UTC"), "close"].iloc[0])


def test_aapl_raw_prices_known_answers():
    bars, acts, _ = yahoo.parse_daily("AAPL", yahoo_frame("AAPL"))
    # as-traded closes (Nasdaq official): 4:1 split ex 2020-08-31, 7:1 split ex 2014-06-09
    assert _close(bars, "2020-08-28") == pytest.approx(499.23, abs=0.01)
    assert _close(bars, "2020-08-31") == pytest.approx(129.04, abs=0.01)
    assert _close(bars, "2014-06-06") == pytest.approx(645.57, abs=0.01)
    assert _close(bars, "2014-06-09") == pytest.approx(93.70, abs=0.01)
    sp = acts[acts["type"] == "SPLIT"].set_index("ex_date")["ratio"]
    assert sp[pd.Timestamp("2020-08-31")] == 4.0 and sp[pd.Timestamp("2014-06-09")] == 7.0
    # raw dividend: $2.65 per (pre-split) share, ex 2012-08-09
    dv = acts[acts["type"] == "CASH_DIVIDEND"].set_index("ex_date")["amount"]
    assert dv[pd.Timestamp("2012-08-09")] == pytest.approx(2.65, abs=1e-3)


def test_aapl_raw_volume_unadjusted():
    raw = yahoo_frame("AAPL")
    bars, _, _ = yahoo.parse_daily("AAPL", raw)
    y = raw.loc[raw.index.tz_localize(None).normalize() == pd.Timestamp("2020-08-28"), "Volume"].iloc[0]
    v = float(bars.loc[bars.index.normalize() == pd.Timestamp("2020-08-28", tz="UTC"), "volume"].iloc[0])
    assert v == pytest.approx(y / 4.0)
    assert v == pytest.approx(46_907_479, rel=1e-3)   # Nasdaq-reported AAPL volume 2020-08-28


def test_tsla_splits_known_answers():
    bars, acts, _ = yahoo.parse_daily("TSLA", yahoo_frame("TSLA"))
    assert _close(bars, "2020-08-28") == pytest.approx(2213.40, abs=0.02)   # 5:1 ex 2020-08-31
    assert _close(bars, "2020-08-31") == pytest.approx(498.32, abs=0.02)
    assert _close(bars, "2022-08-24") == pytest.approx(891.29, abs=0.02)    # 3:1 ex 2022-08-25
    assert _close(bars, "2022-08-25") == pytest.approx(296.07, abs=0.02)
    sp = acts[acts["type"] == "SPLIT"].set_index("ex_date")["ratio"]
    assert sp[pd.Timestamp("2020-08-31")] == 5.0 and sp[pd.Timestamp("2022-08-25")] == 3.0


def test_action_known_at_is_prev_session_close():
    _, acts, _ = yahoo.parse_daily("AAPL", yahoo_frame("AAPL"))
    r = acts[(acts["type"] == "SPLIT") & (acts["ex_date"] == pd.Timestamp("2020-08-31"))].iloc[0]
    assert r["known_at"] == pd.Timestamp("2020-08-28 20:00", tz="UTC")
    assert r["fidelity"] == 2


def test_daily_bar_calendar_alignment_incl_early_close():
    bars, _, _ = yahoo.parse_daily("SPY", yahoo_frame("SPY"))
    assert not frames.validate_bars(bars)
    k = bars["known_at"]
    # day after Thanksgiving 2020: early close 13:00 EST = 18:00 UTC
    assert k[pd.Timestamp("2020-11-27 14:30", tz="UTC")] == pd.Timestamp("2020-11-27 18:00", tz="UTC")
    # July 3 2019: early close 13:00 EDT = 17:00 UTC; regular day 16:00 EDT = 20:00 UTC
    assert k[pd.Timestamp("2019-07-03 13:30", tz="UTC")] == pd.Timestamp("2019-07-03 17:00", tz="UTC")
    assert k[pd.Timestamp("2019-07-05 13:30", tz="UTC")] == pd.Timestamp("2019-07-05 20:00", tz="UTC")
    # DST: winter open 14:30 UTC
    assert pd.Timestamp("2020-11-30 14:30", tz="UTC") in bars.index
    # no bar on a holiday (2019-07-04)
    assert not (bars.index.normalize() == pd.Timestamp("2019-07-04", tz="UTC")).any()
    assert (bars["session"] == "DAY").all()


def test_spy_dividend_raw_amount():
    _, acts, _ = yahoo.parse_daily("SPY", yahoo_frame("SPY"))
    dv = acts[acts["type"] == "CASH_DIVIDEND"].set_index("ex_date")["amount"]
    assert dv[pd.Timestamp("2024-03-15")] == pytest.approx(1.594937, abs=1e-4)   # SPDR Q1-2024 distribution


def test_intraday_sessions_and_known_at():
    df = pd.read_csv(FIX / "yahoo_1m_SPY_20261023.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    bars = yahoo.parse_intraday("SPY", df, "1m", store=None)
    assert set(bars["session"]) <= {"PRE", "RTH", "POST"}
    rth = bars[bars["session"] == "RTH"]
    assert rth.index[0] == pd.Timestamp("2026-10-23 13:30", tz="UTC")       # 09:30 EDT
    assert rth.index[-1] < pd.Timestamp("2026-10-23 20:00", tz="UTC")
    assert len(rth) == 390
    assert ((bars["known_at"] - bars.index) == pd.Timedelta(minutes=1)).all()
    assert (bars[bars["session"] == "PRE"].index < pd.Timestamp("2026-10-23 13:30", tz="UTC")).all()
    assert (bars[bars["session"] == "POST"].index >= pd.Timestamp("2026-10-23 20:00", tz="UTC")).all()


def test_raw_reconstruction_matches_independent_dolthub_ohlcv():
    """Cross-source known answer: Yahoo split-adjusted rows un-adjusted by us equal DoltHub's
    as-traded stocks.ohlcv closes around the AAPL 7:1 / 4:1 and TSLA 5:1 / 3:1 splits."""
    d = pd.read_csv(FIX / "dolthub_ohlcv_AAPL_TSLA_splits.csv", parse_dates=["date"])
    n = 0
    for sym in ("AAPL", "TSLA"):
        bars, _, _ = yahoo.parse_daily(sym, yahoo_frame(sym))
        y = pd.Series(bars["close"].to_numpy(), index=frames.to_naive_date(bars.index))
        dd = d[d["act_symbol"] == sym].set_index("date")["close"]
        common = dd.index.intersection(y.index)
        assert len(common) >= 4
        assert np.allclose(y[common].to_numpy(), dd[common].to_numpy(), atol=0.011)
        n += len(common)
    assert n >= 15
