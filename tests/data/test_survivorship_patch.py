"""Survivorship patch (2026-10-30): Alpaca SIP daily bars for delisted members (providers/alpaca_eod.py),
their use by the identity layer (as-traded by-ticker rows, tape-absence relabelling of DoltHub rows),
the share-class sibling identity rule and the PIT rebase of fja's renamed membership intervals.
Synthetic frames only (named synthetic_*): no network."""
import numpy as np
import pandas as pd
import pytest

from atlasbt.data import frames
from atlasbt.data.identity import IdentityIndex, build_series
from atlasbt.data.portal import Portal
from atlasbt.data.providers import alpaca_eod as AE
from atlasbt.data.providers import sp500
from atlasbt.data.store import ParquetStore


def _sessions(a, b):
    s = frames.schedule().index
    return s[(s >= pd.Timestamp(a)) & (s <= pd.Timestamp(b))]


def synthetic_bars(a, b, start_px=50.0, vol=1e6, seed=0):
    d = _sessions(a, b)
    rng = np.random.default_rng(seed)
    c = start_px * np.exp(np.cumsum(rng.normal(0, 0.01, len(d))))
    raw = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": vol}, index=d)
    return frames.daily_bars(raw)[0]


def seed(store, evidence, ticker_map=None):
    """evidence: list of (ticker, cik, first, last, conf) -> monthly evidence rows."""
    parts = []
    for t, c, a, b, conf in evidence:
        ds = pd.date_range(a, b, freq="MS")
        parts.append(pd.DataFrame({"ticker": t, "cik": c, "conf": conf, "date": ds}))
    ev = pd.concat(parts, ignore_index=True).astype({"conf": "int8"})
    store.write("reference", "identity", "evidence", ev, key=["ticker", "cik", "conf", "date"], replace=True)
    tm = ticker_map or {}
    store.write("reference", "sec", "ticker_map",
                pd.DataFrame({"cik": list(tm.values()) or [1], "name": list(tm) or ["ZZZZ"], "ticker": list(tm) or ["ZZZZ"],
                              "exchange": "X"}), key=["ticker", "cik"], replace=True)


# ------------------------------------------------------------------------------------------- source
def test_synthetic_alpaca_filler_bars_dropped():
    # Alpaca emits zero-trade bars at the last print after a symbol stops trading (DD 2017-09 ..)
    idx = pd.DatetimeIndex(pd.to_datetime(["2017-08-30 04:00", "2017-08-31 04:00", "2017-09-01 04:00",
                                           "2017-09-05 04:00"], utc=True), name="start")
    df = pd.DataFrame({"open": [81.7, 83.5, 83.93, 83.93], "high": [83.3, 85.2, 83.93, 83.93],
                       "low": [81.6, 83.3, 83.93, 83.93], "close": [83.19, 83.93, 83.93, 83.93],
                       "volume": [1.9e6, 5.0e7, 0.0, 0.0], "vwap": np.nan, "trades": [1, 1, 0, 0]}, index=idx)
    out = AE.to_daily(df)
    assert list(frames.to_naive_date(out.index)) == [pd.Timestamp("2017-08-30"), pd.Timestamp("2017-08-31")]
    assert out["close"].iloc[-1] == 83.93 and (out["session"] == "DAY").all()


def test_synthetic_compare_and_split_check():
    a = synthetic_bars("2020-01-02", "2020-06-30")
    b = a.copy()
    assert AE.compare(a, b)["ok"]
    b2 = b.copy()
    b2["close"] *= 1.03                                  # 3 % off on every day -> rejected
    assert not AE.compare(a, b2)["ok"]
    # a 2:1 split on 2020-03-02 shows up as a 0.5 open / previous close gap
    a2 = a.copy()
    m = frames.to_naive_date(a2.index) >= pd.Timestamp("2020-03-02")
    a2.loc[m, ["open", "high", "low", "close"]] /= 2
    sp = pd.DataFrame({"ex_date": [pd.Timestamp("2020-03-02")], "ratio": [2.0]})
    assert AE.split_check(a2, sp)[0]["ok"]
    assert not AE.split_check(a, sp)[0]["ok"]


def test_fetch_refused_without_source_validation(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(AE, "download_many", lambda *a, **k: called.append(1) or {})
    st = ParquetStore(tmp_path / "s")
    out = AE.fetch_symbols(Portal(st), ["BCR"], {"accepted": False, "pass_share": 0.5})
    assert "skipped" in out and not called and not st.exists("bars_1d", AE.SOURCE, "BCR")


def test_fetch_rejects_symbol_disagreeing_with_dolthub(tmp_path, monkeypatch):
    st = ParquetStore(tmp_path / "s")
    dolt = synthetic_bars("2018-01-02", "2018-12-31", seed=1)
    st.write("bars_1d", "dolthub", "XYZ", dolt)
    other = dolt.copy()
    other["close"] *= 1.5                                # another company's prices under the same ticker
    monkeypatch.setattr(AE, "download_many", lambda syms, *a, **k: {"XYZ": other, "OK": dolt})
    out = AE.fetch_symbols(Portal(st), ["XYZ", "OK"], {"accepted": True})
    assert out["rejected"] == ["XYZ"] and out["stored"] == ["OK"]
    assert not st.exists("bars_1d", AE.SOURCE, "XYZ")


# ------------------------------------------------------------------------------------------- identity
def test_synthetic_alpaca_rows_build_delisted_series(tmp_path):
    # issuer 900010 traded as 'DLS' 2006 .. 2017-06 and was acquired; no Yahoo, no DoltHub (DoltHub
    # has nothing that stopped before 2018) -> its 2016-2017 history comes from Alpaca rows
    st = ParquetStore(tmp_path / "s")
    seed(st, [("DLS", 900010, "2006-01-01", "2017-06-01", 0)])
    st.write("bars_1d", AE.SOURCE, "DLS", synthetic_bars("2016-01-04", "2017-06-15"))
    p = Portal(st)
    m, acts, lg = build_series(p, IdentityIndex(st), 900010, "DLS")
    assert len(m) and set(m["src"]) == {"alpaca_eod"}
    d = frames.to_naive_date(m.index)
    assert d[0] == pd.Timestamp("2016-01-04") and d[-1] == pd.Timestamp("2017-06-15")


def test_synthetic_dolthub_rows_off_the_tape_lose_identity(tmp_path):
    # DoltHub files a successor's history under the old ticker after the old issuer stopped trading;
    # the consolidated tape (Alpaca) shows no trade under the ticker -> those rows are not the issuer's
    st = ParquetStore(tmp_path / "s")
    seed(st, [("OLDX", 900011, "2006-01-01", "2019-12-01", 0)])         # old issuer keeps filing (debt)
    alp = synthetic_bars("2016-01-04", "2017-08-31", start_px=80, seed=2)
    st.write("bars_1d", AE.SOURCE, "OLDX", alp)
    st.write("bars_1d", "dolthub", "OLDX", synthetic_bars("2017-09-05", "2018-12-31", start_px=60, seed=3))
    p = Portal(st)
    m, acts, lg = build_series(p, IdentityIndex(st), 900011, "OLDX")
    d = frames.to_naive_date(m.index)
    assert d[-1] == pd.Timestamp("2017-08-31"), lg
    assert any("not on the SIP tape" in x for x in lg)


def test_class_sibling_rule(tmp_path):
    st = ParquetStore(tmp_path / "s")
    seed(st, [("VIAX", 900020, "2006-01-01", "2019-12-01", 0),          # class A evidenced as 'VIAX'
              ("VIAX", 900099, "2021-10-01", "2024-06-01", 0),          # root reused by another issuer
              ("CCX", 900030, "2010-01-01", "2016-05-01", 0)],
         ticker_map={"OWNB": 900040})
    idx = IdentityIndex(st)
    assert idx.class_sibling("VIAXB") == "VIAX"
    lab = idx.label_dates("VIAXB", pd.DatetimeIndex(["2012-06-01", "2022-06-01", "2005-01-03"]))
    assert list(lab) == [900020, 900099, 0]        # the sibling's own date labels, never extended
    assert idx.class_sibling("CCXP") is None        # 'P' is not a share-class letter (CCEP is not CCE)
    assert idx.class_sibling("OWNB") is None        # has a current SEC owner -> existing rule


def test_pit_rebase_of_renamed_closed_interval(tmp_path):
    st = ParquetStore(tmp_path / "s")
    seed(st, [("OLDT", 900050, "2006-01-01", "2015-07-01", 0), ("NEWT", 900050, "2015-08-01", "2026-10-01", 0),
              ("LIVE", 900060, "2006-01-01", "2026-10-01", 0)])
    idx = IdentityIndex(st)
    mem = pd.DataFrame({"ticker": ["NEWT", "NEWT", "LIVE"], "base": ["NEWT", "NEWT", "LIVE"],
                        "start": pd.to_datetime(["2008-01-02", "2018-01-02", "2008-01-02"]),
                        "end": pd.to_datetime(["2013-12-23", None, "2012-01-03"]),
                        "historical_identity": False, "left_censored": False})
    out = sp500.pit_rebase(mem, idx)
    assert list(out["base"]) == ["OLDT", "NEWT", "LIVE"]   # closed pre-rename interval only
    assert list(out["fja_base"]) == ["NEWT", "NEWT", "LIVE"]
    assert "CIK 900050" in out["rebase_reason"].iloc[0]


# ------------------------------------------------------------------------------------------- Tiingo
def _tiingo_line(store, t, bars, n_lines=1):
    store.write("bars_1d", "tiingo", t, bars, extra={"n_lines": n_lines})


def test_tiingo_parse_and_actions():
    from atlasbt.data.providers import tiingo as TI
    rows = [{"date": "2014-06-06T00:00:00.000Z", "open": 649.9, "high": 651.3, "low": 644.5, "close": 645.57,
             "volume": 87484600, "divCash": 0.0, "splitFactor": 1.0},
            {"date": "2014-06-09T00:00:00.000Z", "open": 92.7, "high": 93.88, "low": 91.75, "close": 93.7,
             "volume": 75415000, "divCash": 0.0, "splitFactor": 7.0},
            {"date": "2014-08-07T00:00:00.000Z", "open": 94.9, "high": 95.0, "low": 94.1, "close": 94.48,
             "volume": 46711100, "divCash": 0.47, "splitFactor": 1.0}]
    bars, acts = TI.parse_eod(rows)
    assert len(bars) == 3 and bars["close"].iloc[0] == 645.57
    sp = acts[acts["type"] == "SPLIT"].iloc[0]
    assert sp["ratio"] == 7.0 and sp["ex_date"] == pd.Timestamp("2014-06-09")
    dv = acts[acts["type"] == "CASH_DIVIDEND"].iloc[0]
    assert dv["amount"] == 0.47 and dv["known_at"] < pd.Timestamp("2014-08-07", tz="UTC")


def test_tiingo_budget_hourly_symbol_cap(tmp_path):
    from atlasbt.data.providers import tiingo as TI
    b = TI.Budget(tmp_path / "ledger.jsonl")
    now = 1_800_000_000.0
    import json
    with b.path.open("w") as f:
        for i in range(TI.MAX_SYMBOLS_HOUR):
            f.write(json.dumps({"t": now - 1800 + i, "s": f"S{i}"}) + "\n")
    assert b.wait_s("S0", now) == 0.0                  # a symbol already counted this hour is free
    w = b.wait_s("NEW", now)
    assert 1700 < w <= 1801                           # wait until the oldest symbol leaves the hour


def test_tiingo_line_owner_rule(tmp_path):
    # 'DEAD' denoted 900070 2006 .. 2013-06 (acquired); its Tiingo line ends 2013-06-28 -> accepted.
    # 'REUS' denoted 900080 2006 .. 2010 and 900081 from 2012: the Tiingo line (keyed by the CURRENT
    # holder) carries 900081's history -> never served as 900080.
    st = ParquetStore(tmp_path / "s")
    seed(st, [("DEAD", 900070, "2006-01-01", "2013-06-01", 0),
              ("REUS", 900080, "2006-01-01", "2010-12-01", 0), ("REUS", 900081, "2012-01-01", "2026-10-01", 0)])
    _tiingo_line(st, "DEAD", synthetic_bars("2006-01-03", "2013-06-28", seed=4))
    _tiingo_line(st, "REUS", synthetic_bars("2006-01-03", "2026-10-28", seed=5))
    p = Portal(st)
    idx = IdentityIndex(st)
    m, _, lg = build_series(p, idx, 900070, "DEAD")
    d = frames.to_naive_date(m.index)
    assert set(m["src"]) == {"tiingo"} and d[0] == pd.Timestamp("2006-01-03") and d[-1] == pd.Timestamp("2013-06-28")
    m, _, lg = build_series(p, idx, 900080, "REUS")
    assert not len(m) and any("line owner CIK 900081" in x for x in lg)
    m, _, lg = build_series(p, idx, 900081, "REUS")
    d = frames.to_naive_date(m.index)
    assert len(m) and d[0] >= pd.Timestamp("2011-09-01")      # not 900080's years


def test_identity_break_inside_alpaca_chunk_fails_closed(tmp_path):
    # the old issuer's evidence edge still labels the first sessions of the next security that took the
    # ticker (AmSurg -> 'EVHC' 2016-12-02, x3.1): the rest of that Alpaca chunk is dropped
    st = ParquetStore(tmp_path / "s")
    seed(st, [("BRK", 900090, "2006-01-01", "2017-01-01", 0)])
    a = synthetic_bars("2016-01-04", "2016-12-30", start_px=20, seed=6)
    m = frames.to_naive_date(a.index) >= pd.Timestamp("2016-12-02")
    a.loc[m, ["open", "high", "low", "close"]] *= 3.1
    st.write("bars_1d", AE.SOURCE, "BRK", a)
    ser, _, lg = build_series(Portal(st), IdentityIndex(st), 900090, "BRK")
    assert frames.to_naive_date(ser.index)[-1] == pd.Timestamp("2016-12-01")
    assert any("identity break" in x for x in lg)


def test_split_without_price_gap_or_double_encoded_spinoff_dropped():
    # Yahoo encodes the GE Vernova spin-off (2024-04-02) as a 1.253 split AND it is carried as SPINOFF;
    # a 1.33 split with no price gap (SSP 2025-12-08) is not a split either
    from atlasbt.data.identity import _validate
    b = synthetic_bars("2024-03-01", "2024-04-30", start_px=175, seed=7)
    b["src"], b["src_ticker"], b["rank"] = "yahoo", "GEX", 0
    d = frames.to_naive_date(b.index)
    j = list(d).index(pd.Timestamp("2024-04-02"))
    b.iloc[j:, [b.columns.get_loc(c) for c in ("open", "high", "low", "close")]] *= 0.80
    acts = pd.DataFrame([{"type": "SPINOFF", "ex_date": pd.Timestamp("2024-04-02"), "ratio": 0.25,
                          "amount": float(b["close"].iloc[j - 1]) * 0.2},
                         {"type": "SPLIT", "ex_date": pd.Timestamp("2024-04-02"), "ratio": 1.253, "amount": 0.0},
                         {"type": "SPLIT", "ex_date": pd.Timestamp("2024-04-15"), "ratio": 1.33, "amount": 0.0}])
    lg = []
    _, out = _validate(b, acts, lg)
    assert list(out["type"]) == ["SPINOFF"], lg


def test_tiingo_zero_volume_tail_dropped():
    from atlasbt.data.providers import tiingo as TI
    rows = [{"date": "2013-05-06T00:00:00.000Z", "open": 50.0, "high": 50.3, "low": 49.9, "close": 49.99,
             "volume": 4062677, "divCash": 0.0, "splitFactor": 1.0},
            {"date": "2013-05-07T00:00:00.000Z", "open": 49.99, "high": 49.99, "low": 49.99, "close": 49.99,
             "volume": 0, "divCash": 0.0, "splitFactor": 1.0}]
    bars, _ = TI.parse_eod(rows)
    assert list(frames.to_naive_date(bars.index)) == [pd.Timestamp("2013-05-06")]
