"""QA checks and survivorship diagnostics (synthetic_ frames + recorded Yahoo fixtures)."""
import numpy as np
import pandas as pd

from atlasbt.data import frames, qa, universes as U
from atlasbt.data.portal import Portal

from .conftest import load_yahoo, utc


def synthetic_daily(n=80, start="2023-01-03"):
    d = frames.schedule().loc[start:].index[:n]
    px = 100 * np.exp(np.cumsum(np.full(n, 0.001)))
    df = pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99, "close": px, "volume": 1e6}, index=d)
    return df


def test_qa_detects_bad_tick_ohlc_and_unrecorded_split():
    df = synthetic_daily()
    df.iloc[40, df.columns.get_loc("close")] *= 1.6                 # one-day spike (bad tick)
    df.iloc[41, df.columns.get_loc("high")] = df.iloc[41]["low"] * 0.9   # high < low
    df.iloc[60:, :4] /= 2.0                                             # unrecorded 2:1 split
    bars, _ = frames.daily_bars(df)
    assert qa.bad_ticks(bars["close"]).sum() >= 1
    assert qa.ohlc_violations(bars).sum() >= 1
    sc = qa.split_crosscheck(bars, pd.DataFrame(columns=["ex_date", "ratio"]))
    assert str(df.index[60].date()) in sc["unrecorded_split_suspects"]


def test_qa_gaps_and_stale():
    df = synthetic_daily().drop(index=synthetic_daily().index[10])
    df.iloc[20:27, df.columns.get_loc("close")] = 50.0
    bars, _ = frames.daily_bars(df)
    assert len(qa.calendar_gaps(bars)) == 1
    assert qa.stale_runs(bars["close"]) >= 1


def test_qa_daily_on_real_aapl_fixture(store):
    load_yahoo(store, "AAPL")
    r = qa.qa_daily(Portal(store), "AAPL")
    assert r["ohlc_violations"] == 0
    assert r["recorded_split_without_jump"] == []             # both 2014 and 2020 splits show a price jump


def test_membership_universe_and_bias_flag(store):
    m = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"], "base": ["AAA", "BBB", "CCC"],
                      "start": pd.to_datetime(["2000-01-03", "2000-01-03", "2010-01-04"]),
                      "end": pd.to_datetime(["2005-01-03", None, None]),
                      "historical_identity": False, "left_censored": False, "yahoo": ["AAA", "BBB", "CCC"]})
    store.write("reference", "sp500", "membership", m, key=["ticker", "start"])
    p = Portal(store)
    assert p.universe("SP500", utc("2004-06-01")) == ["AAA", "BBB"]
    assert p.universe("SP500", utc("2005-01-03 15:00")) == ["BBB"]          # removed effective that session
    assert p.universe("SP500", utc("2012-01-03")) == ["BBB", "CCC"]
    bias = p.universe_survivorship_bias("SP500", utc("2004-06-01"))
    assert bias["survivorship_bias"] and bias["coverage"] == 0.0          # no price data in this store


def test_sector_mapping():
    assert U.sic_to_sector(3571) == "Information Technology"       # AAPL
    assert U.sic_to_sector(6021) == "Financials"                   # JPM
    assert U.sic_to_sector(2834) == "Health Care"                  # pharma
    assert U.sic_to_sector(6798) == "Real Estate"                  # REIT
    assert U.sector_etf_for("Real Estate", "2014-01-02") is None   # XLRE launched 2015-10-08
    assert U.sector_etf_for("Real Estate", "2016-01-04") == "XLRE"
    assert U.sector_etf_for("Communication Services", "2018-06-18") is None


def test_daily_splice_and_cik_alias(store):
    """synthetic_: yahoo stub for OLD, older raw history in dolthub, and a renamed ticker NEW
    (same CIK in the ZipLime events) carrying the earliest history."""
    def bars(start, n, px):
        d = frames.schedule().loc[start:].index[:n]
        df = pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1.0}, index=d)
        return frames.daily_bars(df)[0]
    store.write("bars_1d", "yahoo", "OLD", bars("2024-01-02", 3, 30.0))
    store.write("bars_1d", "dolthub", "OLD", bars("2023-06-01", 100, 20.0))
    store.write("bars_1d", "yahoo", "NEW", bars("2022-01-03", 600, 10.0))
    ev = pd.DataFrame({"kind": "EARNINGS", "symbol": ["OLD", "NEW"], "cik": ["0000000001", "0000000001"],
                       "known_at": [utc("2020-01-01"), utc("2023-01-01")], "accession": ["a", "b"]})
    store.write("events", "ziplime", "EARNINGS", ev, partition="year", time_col="known_at", key=["accession", "symbol"])
    p = Portal(store)
    assert p.alias_for("OLD") == "NEW"
    b = p.bars("OLD", "1d", "2022-01-01", "2024-02-01", as_of=utc("2024-02-01"))
    assert b.attrs["source"] == "yahoo+dolthub+alias:NEW" and b.attrs["fidelity"] == 2
    assert b.loc[b.index < utc("2023-06-01"), "close"].eq(10.0).all()        # from the alias
    dd = frames.schedule().loc["2023-06-01":].index[:100]
    assert b.loc[utc("2023-06-01"):utc(str(dd[-1].date()) + " 23:00"), "close"].eq(20.0).all()  # dolthub fills
    assert b.loc[utc("2024-01-02"):utc("2024-01-05"), "close"].eq(30.0).all()      # yahoo wins where present
