"""Parsing tests for every provider, on recorded real responses (fixtures/) or, for the
key-gated providers we cannot reach, synthetic_ payloads shaped like the official docs."""
import json

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.types import Fidelity
from atlasbt.data.providers import (alpaca, alphavantage, cboe, databento, dolthub, dolthub_bulk, dukascopy, fomc,
                                    french, nasdaqtrader, polygon, sec, sp500, tiingo, ziplime)

from .conftest import FIX, utc


def test_cboe_parse_ohlc_and_single_value():
    v = cboe.parse_csv("VIX", (FIX / "cboe_VIX_History.csv").read_bytes())
    assert not v.attrs["single_value"]
    # 2022-07-04 (US holiday, Cboe published a VIX value) is dropped: no NYSE session
    assert not (v.index.normalize() == pd.Timestamp("2022-07-04", tz="UTC")).any()
    # known at close + 15 min; 2020-11-27 early close 13:00 EST -> 18:15 UTC
    r = v.loc[v.index.normalize() == pd.Timestamp("2020-11-27", tz="UTC")]
    assert r["known_at"].iloc[0] == utc("2020-11-27 18:15") and r["close"].iloc[0] == 20.84
    s = cboe.parse_csv("SKEW", (FIX / "cboe_SKEW_History.csv").read_bytes())
    assert s.attrs["single_value"] and (s["open"] == s["close"]).all()


def test_french_parse_decimal():
    df = french.parse_zip((FIX / "french_FF3_daily_truncated.zip").read_bytes())
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert df.loc["1926-07-01", "Mkt-RF"] == pytest.approx(0.0009)
    assert df.loc["2024-01-03", "Mkt-RF"] == pytest.approx(-0.0101)


def test_dolthub_option_rows_canonical():
    rows = json.load(open(FIX / "dolthub_option_chain_SPY_20240103.json"))["rows"]
    c = dolthub.parse_option_rows(rows)
    assert len(c) == 104
    assert (c["ts"] == utc("2024-01-03 21:00")).all()          # snapshot convention: session close
    assert c["fidelity"].eq(int(Fidelity.T2_PROXY)).all()
    r = c[(c["strike"] == 469.0) & (c["right"] == "C") & (c["expiry"] == pd.Timestamp("2024-01-17"))].iloc[0]
    assert r["symbol"] == "SPY   240117C00469000" and r["bid"] == 4.46 and r["ask"] == 4.48
    assert r["mid"] == pytest.approx(4.47) and r["iv"] == pytest.approx(0.1134)
    # vectorized bulk path produces identical canonical rows
    b = dolthub_bulk.canonical_chain(pd.DataFrame(rows))
    pd.testing.assert_series_equal(b["symbol"].reset_index(drop=True), c["symbol"].reset_index(drop=True))
    assert np.allclose(b["mid"].fillna(-1), c["mid"].fillna(-1))


def test_dolthub_weekend_snapshot_maps_to_friday_close():
    assert dolthub.snapshot_ts("2019-02-09") == utc("2019-02-08 21:00")   # Saturday -> Friday 16:00 EST


def test_dolthub_earnings_parse():
    d = json.load(open(FIX / "dolthub_earnings_AAPL.json"))
    ev = dolthub.parse_earnings("AAPL", d["earnings_calendar"], d["eps_history"])
    r = ev[ev["release_date"] == pd.Timestamp("2023-02-02")].iloc[0]
    assert r["timing"] == "AMC" and r["known_at"] == utc("2023-02-02 21:30")    # 16:30 EST approx
    assert r["fiscal_period"] == pd.Timestamp("2022-12-31") and r["eps_actual"] == 1.88
    assert r["estimate_known_at"] < r["known_at"]


def test_sec_parsers():
    main = json.load(open(FIX / "sec_submissions_AAPL_subset.json"))
    f = sec.parse_filings(pd.DataFrame(main["filings"]["recent"]))
    ev = sec.earnings_8k_events("AAPL", f)
    assert len(ev) and (ev["timing"] == "AMC").mean() > 0.9
    hit = ev[ev["accn"] == "0000320193-24-000080"]
    if len(hit):
        assert hit["known_at"].iloc[0] == utc("2024-08-01 20:30:26")
    facts = json.load(open(FIX / "sec_companyfacts_AAPL_subset.json"))
    accept = dict(zip(f["accessionNumber"], f["acceptance"]))
    fund = sec.parse_companyfacts("AAPL", facts, accept)
    assert {"eps_diluted_q", "total_assets", "revenue_q"} <= set(fund["field"])
    q = fund[(fund["field"] == "eps_diluted_q") & (fund["period_end"] == pd.Timestamp("2023-12-30"))]
    assert q["value"].iloc[0] == 2.18            # AAPL fiscal Q1-2024 diluted EPS
    # known_at is the filing's acceptance time; EDGAR dates filings accepted after 17:30 ET as the
    # next business day, so acceptance may precede the `filed` date (AAPL 10-K accepted
    # 2022-10-27 18:01 ET, filed 2022-10-28) but never by more than a few days
    acc = fund[fund["known_at_method"] == "acceptance"]
    assert len(acc) and (acc["accn"].map(accept) == acc["known_at"]).all()
    lag = acc["filed"] - acc["known_at"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    assert lag.between(pd.Timedelta(0), pd.Timedelta(days=4)).all()
    fb = fund[fund["known_at_method"] != "acceptance"]
    assert (fb["known_at"] == [pd.Timestamp(d).tz_localize("America/New_York").replace(hour=17, minute=30)
                               .tz_convert("UTC") for d in fb["filed"]]).all()


def test_fomc_parsers():
    page = (FIX / "fomc_statement_20240131.htm").read_text(errors="ignore")
    assert fomc.parse_release_time(page) == "14:00"
    links = fomc.statement_links((FIX / "fomc_historical2008_links.htm").read_text())
    d = dict(links)
    assert "2008-01-22" in d and d["2008-01-22"].endswith("/newsevents/pressreleases/monetary20080122b.htm")


def test_sp500_parsers():
    mem = sp500.parse_fja((FIX / "fja_sp500_subset.csv").read_bytes())
    assert len(mem) and {"ticker", "start", "end", "base"} <= set(mem.columns)
    wiki = sp500.parse_wiki_changes((FIX / "wiki_sp500_changes_snippet.wikitext").read_text())
    r = wiki[wiki["added"] == "TWLO"].iloc[0]
    assert r["effective_date"] == pd.Timestamp("2026-10-06") and r["removed"] == "WBD"
    assert r["announce_date"] == pd.Timestamp("2026-10-01")       # from the S&P press-release URL
    be = wiki[wiki["added"] == "BE"]
    if len(be):
        assert be["announce_date"].iloc[0] == pd.Timestamp("2026-09-04")   # |date= of the citation


def test_nasdaqtrader_parse():
    df, footer = nasdaqtrader.parse("other", (FIX / "nasdaqtrader_other.txt").read_bytes())
    spy = df[df["symbol"] == "SPY"].iloc[0]
    assert spy["etf"] and spy["exchange"] == "NYSE Arca"
    assert footer.startswith("File Creation Time")
    df2, _ = nasdaqtrader.parse("nasdaq", (FIX / "nasdaqtrader_nasdaq.txt").read_bytes())
    assert not df2[df2["symbol"] == "AAPL"]["etf"].iloc[0]


def test_dukascopy_decoders():
    t = dukascopy.decode_ticks((FIX / "dukascopy_QQQUSUSD_20240116_14h_ticks.bi5").read_bytes(),
                               utc("2024-01-16 14:00"))
    assert len(t) == 15829
    assert t.index[0] == pd.Timestamp("2024-01-16 14:30:00.156", tz="UTC")     # RTH open 09:30 EST
    assert t["ask"].iloc[0] == pytest.approx(408.324) and t["bid"].iloc[0] == pytest.approx(408.296)
    assert (t["ask"] >= t["bid"]).all()
    b = dukascopy.decode_candles((FIX / "dukascopy_SPYUSUSD_20240116_BID_candles_min_1.bi5").read_bytes(),
                                 utc("2024-01-16"))
    a = dukascopy.decode_candles((FIX / "dukascopy_SPYUSUSD_20240116_ASK_candles_min_1.bi5").read_bytes(),
                                 utc("2024-01-16"))
    rth = b.loc["2024-01-16 14:30":"2024-01-16 20:59"]
    assert len(rth) == 390 and 470 < rth["close"].median() < 480      # SPY traded ~472-476 that day
    assert ((a["close"] - b["close"]).loc[rth.index] >= 0).all()


def test_ziplime_events():
    a = pd.read_parquet(FIX / "ziplime_announcements_subset.parquet")
    e = pd.read_parquet(FIX / "ziplime_earnings_subset.parquet")
    ev = ziplime.build_events(a, e)
    r = ev[(ev["symbol"] == "AAPL") & (ev["accession"] == "0000320193-23-000005")].iloc[0]
    assert r["known_at"] == utc("2023-02-02 21:30:33") and r["timing"] == "AMC"
    assert r["eps_actual"] == 1.88 and r["first_tradeable_session"] == pd.Timestamp("2023-02-03")
    assert set(ev["timing"]) <= {"BMO", "AMC", "DURING", "UNKNOWN"}


# ---- key-gated providers: synthetic_ payloads copied from the shape in the official docs
def test_synthetic_alpaca_parse():
    bars = alpaca.parse_bars([{"t": "2024-01-03T14:30:00Z", "o": 470.43, "h": 470.6, "l": 470.1, "c": 470.2,
                               "v": 1234, "n": 50, "vw": 470.3}])
    assert bars.index[0] == utc("2024-01-03 14:30") and bars["vwap"].iloc[0] == 470.3
    q = alpaca.parse_quotes([{"t": "2024-01-03T14:30:00.5Z", "bp": 470.1, "bs": 3, "ap": 470.11, "as": 2,
                              "bx": "P", "ax": "Q"}])
    assert q["bid_size"].iloc[0] == 300 and q["ask"].iloc[0] == 470.11


def test_synthetic_polygon_parse():
    df = polygon.parse_aggs([{"t": 1704292200000, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10, "vw": 1.2, "n": 3}])
    assert df.index[0] == utc("2024-01-03 14:30")


def test_synthetic_alphavantage_and_databento_parse():
    av = alphavantage.parse_csv(b"timestamp,open,high,low,close,volume\n2024-01-03 09:30:00,1,2,0.5,1.5,100\n")
    assert av.index[0] == utc("2024-01-03 14:30")
    db = databento.parse_ohlcv_csv(b"ts_event,rtype,publisher_id,instrument_id,open,high,low,close,volume,symbol\n"
                                   b"2024-01-03T14:30:00.000000000Z,33,2,15144,470.4,470.6,470.1,470.2,1000,SPY\n")
    assert db.index[0] == utc("2024-01-03 14:30") and db["close"].iloc[0] == 470.2


def test_keyed_providers_degrade_without_keys(monkeypatch):
    for k in ("ALPACA_API_KEY", "ALPACA_API_SECRET", "POLYGON_API_KEY", "TIINGO_API_KEY", "DATABENTO_API_KEY",
              "ALPHAVANTAGE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert not any(m.available() for m in (alpaca, polygon, tiingo, databento, alphavantage))
    assert "skipped" in alpaca.fetch_bars("SPY", "2024-01-03", "2024-01-04")
    assert "skipped" in tiingo.fetch_daily("SPY")


def test_iex_tops_parse():
    from atlasbt.data.providers import iex
    q, t = iex.parse_tops(FIX / "iex_tops16_20191224_SPY_packets.pcapng.gz", ["SPY"])
    spy = q["SPY"]
    assert len(spy) > 0 and spy.index.tz is not None
    # scout-verified message: 2019-12-24 13:34:55.892 UTC QuoteUpdate bid 2000 @ 321.55
    hit = spy[(spy["bid"] == 321.55) & (spy["bid_size"] == 2000)]
    assert len(hit) and hit.index[0].floor("s") == utc("2019-12-24 13:34:55")
    assert spy.loc[spy["ask"].notna(), "ask"].iloc[0] == 321.61
    assert spy["ask"].isna().any()                    # zero price = no quote -> NaN


def test_offline_flag_serves_cache_and_never_fetches(tmp_path, monkeypatch):
    """ATLASBT_OFFLINE=1: cached responses are served (even past ttl); anything else raises
    FetchError without touching the network. cache_only=True does the same per call."""
    import pytest
    from atlasbt.data.http import FetchError, HttpClient, cache_key

    c = HttpClient(raw_root=tmp_path)

    def boom(*a, **k):
        raise AssertionError("network used")
    monkeypatch.setattr(c, "_get_follow", boom)
    url = "https://data.sec.gov/submissions/CIK0000000001.json"
    body_p, meta_p = c._paths(url, cache_key(url, None, ""))
    c._store(body_p, meta_p, url, None, 200, b'{"name": "X"}')
    monkeypatch.setenv("ATLASBT_OFFLINE", "1")
    assert c.get(url, ttl=0.0) == b'{"name": "X"}'
    with pytest.raises(FetchError):
        c.get("https://data.sec.gov/submissions/CIK0000000002.json")
    monkeypatch.setenv("ATLASBT_OFFLINE", "0")
    with pytest.raises(FetchError):
        c.get("https://data.sec.gov/submissions/CIK0000000003.json", cache_only=True)
