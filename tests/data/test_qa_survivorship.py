"""QA checks and survivorship diagnostics (synthetic_ frames + recorded Yahoo fixtures)."""
import numpy as np
import pytest
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


def _bars(start, n, px, vol=1e6):
    d = frames.schedule().loc[start:].index[:n]
    p = np.full(n, float(px)) if np.isscalar(px) else np.asarray(px, dtype=float)
    df = pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "volume": vol}, index=d)
    return frames.daily_bars(df)[0]


def _evidence(store, rows):
    """synthetic_ identity evidence rows: (ticker, cik, first_month, last_month) as_of monthly."""
    from .conftest import seed_identity
    parts = []
    for t, c, a, b in rows:
        m = pd.date_range(a, b, freq="MS")
        parts.append(pd.DataFrame({"ticker": t, "cik": c, "conf": 0, "date": m}))
    ev = pd.concat(parts).astype({"conf": "int8"})
    store.write("reference", "identity", "evidence", ev, key=["ticker", "cik", "conf", "date"], replace=True)


def test_identity_renamed_company_uses_current_ticker_history(store):
    """synthetic_: OLD renamed to NEW (same CIK 777). Yahoo keeps the history under NEW; a request
    for the PIT ticker OLD in 2021 is served from it (identity verified by CIK + DoltHub agreement)."""
    _evidence(store, [("OLD", 777, "2000-01-01", "2022-12-01"), ("NEW", 777, "2023-01-01", "2026-10-01")])
    store.write("reference", "sec", "ticker_map", pd.DataFrame({"cik": [777], "name": ["New"], "ticker": ["NEW"],
                                                                "exchange": ["X"]}), key=["ticker", "cik"], replace=True)
    store.write("bars_1d", "yahoo", "NEW", _bars("2020-01-02", 1500, 40.0))
    store.write("bars_1d", "dolthub", "OLD", _bars("2021-01-04", 300, 40.0))
    p = Portal(store)
    b = p.bars("OLD", "1d", "2021-03-01", "2021-03-10", as_of=utc("2021-03-10"))
    assert len(b) and b.attrs["security_id"] == "CIK0000000777" and (b["close"] == 40.0).all()
    assert p.security_id("OLD", utc("2021-03-10")) == "CIK0000000777"


def test_identity_reused_ticker_fails_closed(store):
    """synthetic_: REU was company 111 until 2010, unused, then company 222 from 2015 (Yahoo REU =
    222). History requested for 2008 must NOT be served from 222's Yahoo series."""
    from atlasbt.core.types import DataUnavailableError
    _evidence(store, [("REU", 111, "2000-01-01", "2010-06-01"), ("REU", 222, "2015-01-01", "2026-10-01")])
    store.write("reference", "sec", "ticker_map", pd.DataFrame({"cik": [222], "name": ["Two"], "ticker": ["REU"],
                                                                "exchange": ["X"]}), key=["ticker", "cik"], replace=True)
    store.write("bars_1d", "yahoo", "REU", _bars("2005-01-03", 5000, 5.0))       # Yahoo glues history
    p = Portal(store)
    with pytest.raises(DataUnavailableError):
        p.bars("REU", "1d", "2008-03-01", "2008-03-10", as_of=utc("2026-01-02"))
    b = p.bars("REU", "1d", "2020-03-02", "2020-03-10", as_of=utc("2026-01-02"))
    assert len(b) and (b["close"] == 5.0).all()
    rep = p.universe_survivorship_bias("SP500", utc("2020-03-10")) if False else None


def _sp500_refs(store, membership, current=(), wiki=()):
    m = pd.DataFrame(membership, columns=["ticker", "start", "end"])
    m["start"], m["end"] = pd.to_datetime(m["start"]), pd.to_datetime(m["end"])
    m["base"] = m["yahoo"] = m["ticker"]
    m["historical_identity"] = m["left_censored"] = False
    store.write("reference", "sp500", "membership", m, key=["ticker", "start"], replace=True)
    cur = pd.DataFrame(list(current), columns=["symbol", "name", "cik"])
    if len(cur):
        store.write("reference", "sp500", "current_constituents", cur, key=["symbol"], replace=True)
    w = pd.DataFrame(list(wiki), columns=["effective_date", "added", "added_name", "removed", "removed_name"])
    if len(w):
        w["effective_date"] = pd.to_datetime(w["effective_date"])
        store.write("reference", "sp500", "wiki_changes", w, key=["effective_date", "added", "removed"], replace=True)


def test_identity_pre_evidence_confirmed_by_member_cik(store, monkeypatch):
    """synthetic_: as-of ticker evidence starts 2006 (as in ZipLime). A 1996 request for a
    continuous S&P member is served when Wikipedia's CIK of the current constituent equals the
    earliest evidenced issuer -- flagged T2 with a log note."""
    from atlasbt.data import identity
    monkeypatch.setitem(identity._DATED_NAMES, 4242, [(pd.Timestamp("1900-01-01"), pd.Timestamp("2262-01-01"), "OLDCO INC")])
    _evidence(store, [("OLDCO", 4242, "2006-01-01", "2026-10-01")])
    store.write("reference", "sec", "ticker_map", pd.DataFrame({"cik": [4242], "name": ["Oldco"], "ticker": ["OLDCO"],
                                                                "exchange": ["X"]}), key=["ticker", "cik"], replace=True)
    store.write("bars_1d", "yahoo", "OLDCO", _bars("1995-01-03", 8000, 30.0))
    _sp500_refs(store, [("OLDCO", "1990-01-02", None)], current=[("OLDCO", "Oldco Inc", 4242)])
    b = Portal(store).bars("OLDCO", "1d", "1996-06-03", "1996-06-10", as_of=utc("2026-01-02"))
    assert len(b) and b.attrs["security_id"] == "CIK0000004242" and b.attrs["fidelity"] == 2
    assert "'pre'" in b.attrs["identity_log"][-1]


def test_identity_pre_evidence_reused_ticker_fails_closed(store, monkeypatch):
    """synthetic_: RX was company A (removed 1999, no evidence left) and is today company 5151
    (first evidence 2006). The 1997 member's recorded name is A's -> must fail closed."""
    from atlasbt.core.types import DataUnavailableError
    from atlasbt.data import identity
    monkeypatch.setitem(identity._NAMES, 5151, ["NEWCO HOLDINGS INC"])
    monkeypatch.setitem(identity._DATED_NAMES, 5151, [(pd.Timestamp("1900-01-01"), pd.Timestamp("2262-01-01"),
                                                       "NEWCO HOLDINGS INC")])
    _evidence(store, [("RX", 5151, "2006-01-01", "2026-10-01")])
    store.write("reference", "sec", "ticker_map", pd.DataFrame({"cik": [5151], "name": ["Newco"], "ticker": ["RX"],
                                                                "exchange": ["X"]}), key=["ticker", "cik"], replace=True)
    store.write("bars_1d", "yahoo", "RX", _bars("1995-01-03", 8000, 12.0))
    _sp500_refs(store, [("RX", "1990-01-02", "1999-03-01")],
                wiki=[("1999-03-01", "ZZZ", "Zed", "RX", "Alpha Telephone Corp")])
    with pytest.raises(DataUnavailableError):
        Portal(store).bars("RX", "1d", "1997-06-02", "1997-06-10", as_of=utc("2026-01-02"))


def test_identity_pre_evidence_renamed_issuer_fails_closed(store, monkeypatch):
    """synthetic_ (SBC -> AT&T pattern): the issuer of 'TT' in 2006+ was named differently in 2004
    (it then traded under its old ticker) -> a 2004 request for TT must fail closed."""
    from atlasbt.core.types import DataUnavailableError
    from atlasbt.data import identity
    monkeypatch.setitem(identity._DATED_NAMES, 6161, [(pd.Timestamp("1994-01-01"), pd.Timestamp("2005-11-18"), "SBX COMMUNICATIONS INC"),
                                                      (pd.Timestamp("2005-11-18"), pd.Timestamp("2262-01-01"), "TELCO INC")])
    _evidence(store, [("TT", 6161, "2006-01-01", "2026-10-01")])
    store.write("reference", "sec", "ticker_map", pd.DataFrame({"cik": [6161], "name": ["Telco"], "ticker": ["TT"],
                                                                "exchange": ["X"]}), key=["ticker", "cik"], replace=True)
    store.write("bars_1d", "yahoo", "TT", _bars("1995-01-03", 8000, 24.0))
    _sp500_refs(store, [("TT", "1990-01-02", None)], current=[("TT", "Telco", 6161)])
    with pytest.raises(DataUnavailableError):
        Portal(store).bars("TT", "1d", "2004-06-01", "2004-06-10", as_of=utc("2026-01-02"))


def _yahoo_only(store, sym, cik, px, start="2010-01-04"):
    _evidence(store, [(sym, cik, "2006-01-01", "2026-10-01")])
    store.write("reference", "sec", "ticker_map", pd.DataFrame({"cik": [cik], "name": [sym], "ticker": [sym],
                                                                "exchange": ["X"]}), key=["ticker", "cik"], replace=True)
    store.write("bars_1d", "yahoo", sym, _bars(start, len(px), px))
    return Portal(store).bars(sym, "1d", "2009-01-01", "2026-01-01", as_of=utc("2026-01-02"))


def test_yahoo_volatile_crash_not_cut(store):
    """synthetic_ (CVNA 2021-12 .. 2022 pattern): alternating -10 % / +9 % days inside a sustained
    98 % decline are real -> the history before them must NOT be cut."""
    n = 1500
    px = np.full(n, 300.0)
    px[1000:1250] = 300.0 * np.exp(np.linspace(0, np.log(0.02), 250))     # sustained crash
    for i in (1010, 1012, 1014, 1016):                                      # -10 % then +9 % next day
        px[i] = px[i - 1] * 0.90
        px[i + 1] = px[i] * 1.09 * (px[i + 1] / px[i - 1])
    px[1250:] = px[1249]
    b = _yahoo_only(store, "CVX1", 910001, px)
    assert len(b) == n and not any("trim" in x for x in b.attrs["identity_log"])


def test_yahoo_quantized_glitch_cluster_cut(store):
    """synthetic_ thin early data: exact one-tick flips (price returns to the same print) clustered
    in a flat stretch -> the history up to the cluster is cut; later history served."""
    n = 1500
    px = np.full(n, 0.625 * 16)
    for i in (100, 105, 110, 115):
        px[i] = 0.5 * 16
    b = _yahoo_only(store, "QNT", 910002, px)
    assert len(b) < n - 100 and any("trim" in x for x in b.attrs["identity_log"])


def test_has_window_agrees_with_bars(store):
    """synthetic_: reused ticker REU (111 until 2010, 222 from 2015). Windowed has() must answer
    exactly what bars() serves: False where bars() fails closed, True where it serves."""
    _evidence(store, [("REU", 111, "2000-01-01", "2010-06-01"), ("REU", 222, "2015-01-01", "2026-10-01")])
    store.write("reference", "sec", "ticker_map", pd.DataFrame({"cik": [222], "name": ["Two"], "ticker": ["REU"],
                                                                "exchange": ["X"]}), key=["ticker", "cik"], replace=True)
    store.write("bars_1d", "yahoo", "REU", _bars("2005-01-03", 5000, 5.0))
    p = Portal(store)
    A = utc("2026-01-02")
    assert not p.has("REU", "bars", "1d", start=utc("2008-03-01"), end=utc("2008-03-10"), as_of=A)
    assert p.has("REU", "bars", "1d", start=utc("2020-03-01"), end=utc("2020-03-10"), as_of=A)
    cov = p.bars_coverage("REU", utc("2020-03-02"), utc("2020-03-07"), A)
    assert cov["available"] and cov["sessions"] == cov["expected_sessions"] == 5 and cov["coverage"] == 1.0
    assert "fail closed" in p.bars_coverage("REU", utc("2008-03-01"), utc("2008-03-10"), A)["reason"]


def test_membership_rename_back_and_forth(store):
    """synthetic_ (FISV -> FI -> FISV pattern): no 8-K evidence for the middle ticker, but the S&P
    intervals of the other ticker bracket it and denote the same CIK -> served from that CIK's
    Yahoo history (T2, note); a ticker whose neighbours denote different CIKs fails closed."""
    _evidence(store, [("FSV", 7171, "2006-01-01", "2023-06-01"), ("FSV", 7171, "2025-12-01", "2026-10-01"),
                      ("FRK", 8181, "2014-01-01", "2021-05-01")])
    store.write("reference", "sec", "ticker_map", pd.DataFrame({"cik": [7171], "name": ["Fsv"], "ticker": ["FSV"],
                                                                "exchange": ["X"]}), key=["ticker", "cik"], replace=True)
    store.write("bars_1d", "yahoo", "FSV", _bars("2010-01-04", 4200, 100.0))
    _sp500_refs(store, [("FSV", "2010-01-04", "2023-06-07"), ("FRK", "2023-06-07", "2025-11-11"),
                        ("FSV", "2025-11-11", None)])
    b = Portal(store).bars("FRK", "1d", "2024-01-02", "2024-01-10", as_of=utc("2026-01-02"))
    assert len(b) and b.attrs["security_id"] == "CIK0000007171" and b.attrs["fidelity"] == 2
    assert "'membership'" in b.attrs["identity_log"][-1]


def test_identity_is_property_of_bar_date_not_as_of(store):
    """synthetic_ BBT pattern: 'RU' = issuer 111 (2006-2015, as-traded DoltHub rows) and later
    issuer 222 (2020+, Yahoo 'RU' = 222). (1) a bar on 2012-03-01 is the same whatever the run's
    end / as_of; (2) a plain-ticker window spanning both issuers is never spliced (error naming
    the security keys); (3) universe() returns security keys for the reused ticker and a key
    serves only that security."""
    from atlasbt.core.types import DataUnavailableError
    _evidence(store, [("RU", 111, "2006-01-01", "2015-06-01"), ("RU", 222, "2020-01-01", "2026-10-01")])
    store.write("reference", "sec", "ticker_map", pd.DataFrame({"cik": [222], "name": ["Two"], "ticker": ["RU"],
                                                                "exchange": ["X"]}), key=["ticker", "cik"], replace=True)
    store.write("bars_1d", "dolthub", "RU", _bars("2011-01-03", 1100, 50.0))
    store.write("bars_1d", "yahoo", "RU", _bars("2019-06-03", 1500, 7.0))
    _sp500_refs(store, [("RU", "2011-01-03", "2015-06-01"), ("RU", "2021-01-04", None)])
    p = Portal(store)
    day = (utc("2012-03-01"), utc("2012-03-02"))
    early = p.bars("RU", "1d", *day, as_of=utc("2012-03-02"))
    with pytest.raises(DataUnavailableError, match="never spliced"):
        p.bars("RU", "1d", utc("2012-03-01"), utc("2026-01-01"), as_of=utc("2026-01-02"))
    late = p.bars("RU", "1d", *day, as_of=utc("2026-01-02"))
    assert len(early) == len(late) == 1 and float(early["close"].iloc[0]) == float(late["close"].iloc[0]) == 50.0
    assert early.attrs["security_id"] == late.attrs["security_id"] == "CIK0000000111"
    k_old = p.universe("SP500", utc("2012-03-01"))
    k_new = p.universe("SP500", utc("2023-03-01"))
    assert k_old == ["RU~CIK0000000111"] and k_new == ["RU~CIK0000000222"]
    b = p.bars(k_old[0], "1d", utc("2011-01-01"), utc("2026-01-01"), as_of=utc("2026-01-02"))
    assert len(b) and (b["close"] == 50.0).all() and b.attrs["security_id"] == "CIK0000000111"
    assert p.instrument(k_old[0]).symbol == k_old[0]
    assert p.security_id("RU", utc("2012-03-01")) == "CIK0000000111"
    assert p.security_id("RU", utc("2023-03-01")) == "CIK0000000222"


def test_dolthub_collision_segment_rejected(store):
    """synthetic_: DoltHub rows alternating between two price levels (two securities mixed under
    one ticker) fail the collision test -> no bars (fail closed)."""
    from atlasbt.core.types import DataUnavailableError
    _evidence(store, [("MIX", 555, "2010-01-01", "2026-10-01")])
    px = np.full(300, 100.0)
    px[[50, 52, 54, 120, 200]] = 87.0                   # level alternation (two securities interleaved)
    store.write("bars_1d", "dolthub", "MIX", _bars("2015-01-02", 300, px))
    p = Portal(store)
    with pytest.raises(DataUnavailableError):
        p.bars("MIX", "1d", "2015-01-01", "2016-01-01", as_of=utc("2016-01-01"))
    log = p._identity_frame("MIX", pd.Timestamp("2015-06-01")).attrs["identity_log"]
    assert any("collision test failed" in x for x in log)


def test_isolated_dolthub_glitch_dropped(store):
    _evidence(store, [("GL", 556, "2010-01-01", "2026-10-01")])
    px = np.full(300, 100.0)
    px[150] = 86.4                                       # one whole-bar glitch, reverts next day
    store.write("bars_1d", "dolthub", "GL", _bars("2015-01-02", 300, px))
    p = Portal(store)
    b = p.bars("GL", "1d", "2015-01-01", "2016-06-01", as_of=utc("2016-06-01"))
    assert len(b) == 299 and (b["close"] == 100.0).all()


def test_spinoff_reclassified_and_adjusted(store):
    """synthetic_ HPQ-like: Yahoo encodes a spin-off as a 2.202 'split'. It must be served as
    SPINOFF (not SPLIT) with amount = P_prev (1 - 1/2.202), and adjusted history must use the same
    factor Yahoo used (continuous total return)."""
    from atlasbt.core.types import ActionType
    from atlasbt.data.actions_classify import classify
    d = frames.schedule().loc["2015-10-01":].index[:40]
    ex = d[22]
    px = np.where(d < ex, 29.36, 13.32)
    raw = frames.daily_bars(pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1e6}, index=d))[0]
    acts = pd.DataFrame([{"symbol": "HPQ", "type": "SPLIT", "ex_date": ex, "known_at": raw["known_at"].iloc[21],
                          "ratio": 2.202, "amount": 0.0, "pay_date": pd.NaT, "new_symbol": None, "fidelity": 2}])
    c = classify("HPQ", acts, raw)
    assert c["type"].iloc[0] == "SPINOFF" and c["new_symbol"].iloc[0] == "HPE"
    # no child price available -> value implied by the ex-date open (P_prev - open_ex)
    assert c["amount"].iloc[0] == pytest.approx(29.36 - 13.32)
    # value-based: HPE's first ex-date close x 1 HPE per HPQ, known at that close
    cv = classify("HPQ", acts, raw, child_close=lambda s_, e: 15.5 if s_ == "HPE" else np.nan)
    assert cv["amount"].iloc[0] == pytest.approx(15.5) and cv["known_at"].iloc[0] == raw["known_at"].iloc[22]
    acts["type"], acts["ex_date"] = "SPLIT", ex
    _evidence(store, [("HPQ", 47217, "2000-01-01", "2026-10-01")])
    store.write("reference", "sec", "ticker_map", pd.DataFrame({"cik": [47217], "name": ["HP"], "ticker": ["HPQ"],
                                                                "exchange": ["X"]}), key=["ticker", "cik"], replace=True)
    store.write("bars_1d", "yahoo", "HPQ", raw)
    store.write("actions", "yahoo", "HPQ", acts, key=["ex_date", "type"])
    p = Portal(store)
    ca = p.corporate_actions(["HPQ"], d[0], d[-1], utc("2016-01-01"))
    assert [a.type for a in ca] == [ActionType.SPINOFF]
    adj = p.bars("HPQ", "1d", d[0], d[-1], utc("2016-01-01"), adjusted=True)
    r = adj["close"].pct_change().dropna()
    assert r.abs().max() < 2e-3            # the spin-off does not show up as a -55 % return


def test_spinoff_duplicate_encoding_and_value_factor():
    """synthetic_ DHR-2016-like: Yahoo encodes ONE distribution both as a 'dividend' and as a
    fractional split. One SPINOFF must remain; with a value-based amount the adjusted ex-date
    return equals the true total return (S + D) / P_prev - 1, not S / (P_prev - D) - 1."""
    from atlasbt.data import adjust
    from atlasbt.data.actions_classify import classify
    d = frames.schedule().loc["2016-06-01":].index[:30]
    ex = pd.Timestamp("2016-07-05")                     # DHR / Fortive ex-date (known spin-off table)
    i = d.get_loc(ex)
    px = np.where(d < ex, 100.0, 80.0)
    px[i] = 84.0                                       # ex-date close after an intraday rally
    o = px.copy()
    o[i] = 76.0
    raw = frames.daily_bars(pd.DataFrame({"open": o, "high": px, "low": o, "close": px, "volume": 1e6}, index=d))[0]
    k0 = raw["known_at"].iloc[i - 1]
    acts = pd.DataFrame([{"symbol": "DHR", "type": "CASH_DIVIDEND", "ex_date": ex, "known_at": k0, "ratio": 1.0,
                          "amount": 24.0, "pay_date": pd.NaT, "new_symbol": None, "fidelity": 2},
                         {"symbol": "DHR", "type": "SPLIT", "ex_date": ex, "known_at": k0, "ratio": 1.319,
                          "amount": 0.0, "pay_date": pd.NaT, "new_symbol": None, "fidelity": 2}])
    c = classify("DHR", acts, raw, child_close=lambda s_, e: 48.0 if s_ == "FTV" else np.nan)
    assert list(c["type"]) == ["SPINOFF"] and c["amount"].iloc[0] == pytest.approx(24.0)   # 0.5 x 48
    from atlasbt.core.types import ActionType, CorporateAction
    ca = [CorporateAction(symbol="DHR", type=ActionType.SPINOFF, ex_date=ex, known_at=c["known_at"].iloc[0],
                          amount=24.0, ratio=0.5)]
    dd = frames.to_naive_date(raw.index)
    pf, _ = adjust.adjustment_factors(dd, pd.Series(px, index=dd), ca)
    adj = px * pf
    assert adj[i] / adj[i - 1] - 1 == pytest.approx((84.0 + 24.0) / 100.0 - 1)
    # an UNKNOWN event with the same double encoding collapses to one SPINOFF too
    u = classify("ZZZ", acts.assign(symbol="ZZZ"), raw)
    assert list(u["type"]) == ["SPINOFF"]


def test_round_stock_dividend_stays_split():
    from atlasbt.data.actions_classify import classify
    a = pd.DataFrame([{"symbol": "ADM", "type": "SPLIT", "ex_date": pd.Timestamp("1990-08-14"), "known_at": utc("1990-08-13"),
                       "ratio": 1.05, "amount": 0.0}, {"symbol": "ADM", "type": "CASH_DIVIDEND",
                       "ex_date": pd.Timestamp("1991-01-02"), "known_at": utc("1990-12-31"), "ratio": 1.0, "amount": 3.0}])
    raw = _bars("1990-08-01", 120, 20.0)
    c = classify("ADM", a, raw)
    assert list(c["type"]) == ["SPLIT", "SPECIAL_DIVIDEND"]      # 5 % stock dividend; 15 % cash -> special
