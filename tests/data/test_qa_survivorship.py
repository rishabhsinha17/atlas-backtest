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
    assert "pre-evidence" in b.attrs["identity_log"][-1]


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
    assert c["amount"].iloc[0] == pytest.approx(29.36 * (1 - 1 / 2.202))
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


def test_round_stock_dividend_stays_split():
    from atlasbt.data.actions_classify import classify
    a = pd.DataFrame([{"symbol": "ADM", "type": "SPLIT", "ex_date": pd.Timestamp("1990-08-14"), "known_at": utc("1990-08-13"),
                       "ratio": 1.05, "amount": 0.0}, {"symbol": "ADM", "type": "CASH_DIVIDEND",
                       "ex_date": pd.Timestamp("1991-01-02"), "known_at": utc("1990-12-31"), "ratio": 1.0, "amount": 3.0}])
    raw = _bars("1990-08-01", 120, 20.0)
    c = classify("ADM", a, raw)
    assert list(c["type"]) == ["SPLIT", "SPECIAL_DIVIDEND"]      # 5 % stock dividend; 15 % cash -> special
