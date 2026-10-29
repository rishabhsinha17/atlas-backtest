"""Adjusted-as-of correctness and point-in-time filtering of the Portal."""
import numpy as np
import pandas as pd
import pytest

from atlasbt.core.types import ActionType, CorporateAction, Fidelity, LookaheadError
from atlasbt.data import adjust, frames
from atlasbt.data.portal import Portal

from .conftest import load_yahoo, utc


def test_split_not_applied_before_ex_date(store, portal):
    load_yahoo(store, "AAPL")
    a0 = portal.bars("AAPL", "1d", "2020-08-25", "2020-08-29", as_of=utc("2020-08-28 21:00"), adjusted=True)
    raw = portal.bars("AAPL", "1d", "2020-08-25", "2020-08-29", as_of=utc("2020-08-28 21:00"), adjusted=False)
    # the 2020-08-31 split (known_at 2020-08-28 20:00) is known but its ex-date is after as_of:
    # history must be untouched
    pd.testing.assert_frame_equal(a0[["open", "close", "volume"]], raw[["open", "close", "volume"]])
    assert a0["close"].iloc[-1] == pytest.approx(499.23, abs=0.01)


def test_split_applied_on_ex_date(store, portal):
    load_yahoo(store, "AAPL")
    a1 = portal.bars("AAPL", "1d", "2020-08-25", "2020-09-01", as_of=utc("2020-08-31 15:00"), adjusted=True)
    assert a1["close"].iloc[-1] == pytest.approx(499.23 / 4, abs=0.01)
    # the ex-date bar itself is not yet complete at 15:00 UTC -> excluded
    assert a1.index[-1] == pd.Timestamp("2020-08-28 13:30", tz="UTC")
    vol_raw = portal.bars("AAPL", "1d", "2020-08-28", "2020-08-29", as_of=utc("2020-09-01"))["volume"].iloc[0]
    assert a1["volume"].iloc[-1] == pytest.approx(vol_raw * 4)


def test_future_actions_do_not_change_history(store):
    """Lookahead perturbation: removing every action with ex_date > as_of (or changing
    them arbitrarily) leaves the adjusted history served at as_of unchanged."""
    load_yahoo(store, "AAPL")
    as_of = utc("2016-01-04 21:00")
    p1 = Portal(store)
    h1 = p1.bars("AAPL", "1d", "2012-01-01", "2016-01-01", as_of=as_of, adjusted=True)
    acts = store.read("actions", "yahoo", "AAPL")
    store.write("actions", "yahoo", "AAPL", acts[acts["ex_date"] <= pd.Timestamp("2016-01-04")],
                key=["ex_date", "type"], replace=True)
    h2 = Portal(store).bars("AAPL", "1d", "2012-01-01", "2016-01-01", as_of=as_of, adjusted=True)
    pd.testing.assert_frame_equal(h1, h2)


def test_dividend_total_return_factor_synthetic():
    """synthetic: close 100 before ex, dividend 1.00 -> earlier prices x 0.99 (CRSP method)."""
    dates = pd.DatetimeIndex(["2024-03-12", "2024-03-13", "2024-03-14", "2024-03-15"])
    close = pd.Series([98.0, 99.0, 100.0, 99.5], index=dates)
    a = CorporateAction("X", ActionType.CASH_DIVIDEND, pd.Timestamp("2024-03-15"), utc("2024-03-14 20:00"), amount=1.0)
    pf, vf = adjust.adjustment_factors(dates, close, [a])
    assert pf.tolist() == pytest.approx([0.99, 0.99, 0.99, 1.0])
    # adjusted return across the ex-date = P_ex / (P_prev - D) - 1 (dividend reinvested at the
    # ex-adjusted price; the CRSP / Yahoo Adj Close convention)
    adj = close * pf
    assert adj.iloc[3] / adj.iloc[2] - 1 == pytest.approx(99.5 / (100.0 - 1.0) - 1)
    assert vf.tolist() == [1, 1, 1, 1]


def test_dividend_adjustment_matches_yahoo_adjclose(store, portal):
    """Real SPY 2024-03-15 ex-dividend: our PIT total-return adjustment reproduces Yahoo's
    Adj Close return across the ex-date."""
    load_yahoo(store, "SPY")
    b = portal.bars("SPY", "1d", "2024-03-12", "2024-03-20", as_of=utc("2024-03-20"), adjusted=True)
    raw = store.read("bars_1d", "yahoo", "SPY")
    y = raw["yahoo_adjclose"].reindex(b.index)
    ours = b["close"].pct_change().dropna()
    theirs = y.pct_change().dropna()
    assert np.abs(ours - theirs).max() < 2e-5


def test_bars_known_at_filter(store, portal):
    load_yahoo(store, "SPY")
    # 2019-07-03 is an early close (17:00 UTC): known at 17:00, not before
    b = portal.bars("SPY", "1d", "2019-07-01", "2019-07-10", as_of=utc("2019-07-03 16:59"))
    assert b.index[-1] == pd.Timestamp("2019-07-02 13:30", tz="UTC")
    b = portal.bars("SPY", "1d", "2019-07-01", "2019-07-10", as_of=utc("2019-07-03 17:00"))
    assert b.index[-1] == pd.Timestamp("2019-07-03 13:30", tz="UTC")
    assert (b["known_at"] <= utc("2019-07-03 17:00")).all()
    assert b.attrs["fidelity"] == int(Fidelity.T1_BAR) and b.attrs["source"] == "yahoo"


def test_corporate_actions_pit(store, portal):
    load_yahoo(store, "AAPL")
    acts = portal.corporate_actions(["AAPL"], "2020-01-01", "2020-12-31", as_of=utc("2020-08-28 19:59"))
    assert not any(a.type == ActionType.SPLIT for a in acts)
    acts = portal.corporate_actions(["AAPL"], "2020-01-01", "2020-12-31", as_of=utc("2020-08-28 20:00"))
    sp = [a for a in acts if a.type == ActionType.SPLIT]
    assert len(sp) == 1 and sp[0].ratio == 4.0 and sp[0].ex_date == pd.Timestamp("2020-08-31")


def _synthetic_intraday(store):
    """synthetic_ 1m bars for one session to exercise resampling & session filters."""
    sess = pd.Timestamp("2024-01-03")
    o, c = frames.schedule().loc[sess, ["open", "close"]]
    idx = pd.date_range(o - pd.Timedelta(minutes=30), c + pd.Timedelta(minutes=30), freq="1min", inclusive="left")
    px = 100 + np.arange(len(idx)) * 0.01
    df = pd.DataFrame({"open": px, "high": px + 0.05, "low": px - 0.05, "close": px + 0.01,
                       "volume": 100.0}, index=idx)
    bars = frames.intraday_bars(df, "1m")
    store.write("bars_1m", "yahoo", "SYN", bars, partition="year")
    return bars


def test_intraday_resample_is_causal(store, portal):
    _synthetic_intraday(store)
    as_of = utc("2024-01-03 14:42")       # 09:42 ET
    b5 = portal.bars("SYN", "5m", "2024-01-03", "2024-01-04", as_of=as_of)
    assert b5.index[0] == utc("2024-01-03 14:30")           # anchored at the open
    assert b5.index[-1] == utc("2024-01-03 14:35")          # [14:40,14:45) not complete yet
    assert (b5["known_at"] <= as_of).all()
    assert b5["volume"].iloc[0] == 500
    pre = portal.bars("SYN", "1m", "2024-01-03", "2024-01-04", as_of=as_of, sessions=("PRE",))
    assert len(pre) == 30 and (pre["session"] == "PRE").all()


def test_events_known_at_filter(store, portal):
    ev = pd.DataFrame({"kind": ["FOMC", "FOMC"], "symbol": [None, None],
                       "known_at": [utc("2024-01-31 19:00"), utc("2024-03-20 18:00")],
                       "effective_at": [utc("2024-01-31 19:00"), utc("2024-03-20 18:00")],
                       "type": "STATEMENT", "date": pd.to_datetime(["2024-01-31", "2024-03-20"]),
                       "fidelity": 1, "source": "fomc"})
    store.write("events", "fomc", "FOMC", ev, key=["date"])
    assert len(portal.events("FOMC", None, "2024-01-01", "2024-12-31", as_of=utc("2024-01-31 18:59"))) == 0
    out = portal.events("FOMC", None, "2024-01-01", "2024-12-31", as_of=utc("2024-01-31 19:00"))
    assert len(out) == 1 and out[0].payload["type"] == "STATEMENT"


def test_fundamentals_pit_with_restatement(store, portal):
    """synthetic_: Q value filed 2023-05-01, restated in a later filing 2023-08-01."""
    f = pd.DataFrame({"symbol": "X", "field": "eps_diluted_q", "value": [1.00, 1.10, 1.25],
                      "period_end": pd.to_datetime(["2023-03-31", "2023-03-31", "2023-06-30"]),
                      "known_at": [utc("2023-05-01 20:00"), utc("2023-08-01 20:00"), utc("2023-08-01 20:00")],
                      "source": "sec", "period_start": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-04-01"]),
                      "accn": ["a1", "a2", "a2"]})
    store.write("fundamentals", "sec", "X", f, key=["field", "period_start", "period_end", "accn"])
    r = portal.fundamentals(["X"], ["eps_diluted_q"], as_of=utc("2023-07-01"))
    assert r["value"].iloc[0] == 1.00
    r = portal.fundamentals(["X"], ["eps_diluted_q"], as_of=utc("2023-08-02"))
    assert r["value"].iloc[0] == 1.25 and r["period_end"].iloc[0] == pd.Timestamp("2023-06-30")
    assert portal.fundamentals(["X"], ["eps_diluted_q"], as_of=utc("2023-04-01")).empty


def test_option_chain_snapshot_pit(store, portal):
    import json
    from atlasbt.data.providers import dolthub
    from .conftest import FIX
    rows = json.load(open(FIX / "dolthub_option_chain_SPY_20240103.json"))["rows"]   # raw SQL API response
    chain = dolthub.parse_option_rows(rows)
    store.write("option_chain", "dolthub", "SPY", chain, partition="year", time_col="ts", key=["ts", "symbol"])
    assert portal.option_chain("SPY", as_of=utc("2024-01-03 20:59")).empty     # before the close snapshot
    c = portal.option_chain("SPY", as_of=utc("2024-01-03 21:00"), min_dte=10, max_dte=20)
    assert len(c) and (c["ts"] == utc("2024-01-03 21:00")).all()
    assert set((pd.to_datetime(c["expiry"]) - pd.Timestamp("2024-01-03")).dt.days) <= set(range(10, 21))
    inst = portal.instrument(c["symbol"].iloc[0])
    q = portal.option_quote(inst, as_of=utc("2024-01-04 15:00"))
    assert q is not None and q.bid == c["bid"].iloc[0] and q.fidelity == Fidelity.T2_PROXY
    assert portal.option_quote(inst, as_of=utc("2024-01-20")) is None          # stale snapshot


def test_universe_static_respects_inception(store, portal):
    load_yahoo(store, "SPY")
    assert portal.universe("BROAD_INDEX_ETF", utc("2019-07-02")) == ["SPY"]
    assert portal.universe("BROAD_INDEX_ETF", utc("2019-06-01")) == []          # no bar known yet in store


def test_option_quotes_window(store, portal):
    import json
    from atlasbt.data.providers import dolthub
    from .conftest import FIX
    rows = json.load(open(FIX / "dolthub_option_chain_SPY_20240103.json"))["rows"]
    c1 = dolthub.parse_option_rows(rows)
    rows2 = [dict(r, date="2024-01-04") for r in rows]          # synthetic_ second snapshot (same quotes)
    c2 = dolthub.parse_option_rows(rows2)
    store.write("option_chain", "dolthub", "SPY", pd.concat([c1, c2]), partition="year", time_col="ts",
                key=["ts", "symbol"])
    inst = portal.instrument(c1["symbol"].iloc[0])
    q = portal.option_quotes(inst, utc("2024-01-03 12:00"), utc("2024-01-05"), as_of=utc("2024-01-05"))
    assert list(q.index) == [utc("2024-01-03 21:00"), utc("2024-01-04 21:00")]
    # (start, end] is open on the left and capped by as_of
    q = portal.option_quotes(inst, utc("2024-01-03 21:00"), utc("2024-01-05"), as_of=utc("2024-01-04 20:59"),
                             synthetic=False)
    assert q.empty
    q = portal.option_quotes(inst, utc("2024-01-03 21:00"), utc("2024-01-05"), as_of=utc("2024-01-04 21:00"))
    assert list(q.index) == [utc("2024-01-04 21:00")] and q["bid"].iloc[0] == c1["bid"].iloc[0]


def test_option_quotes_synthetic_fallback_never_mixed(store, portal, monkeypatch):
    import json
    from atlasbt.data.providers import dolthub
    from atlasbt.options import marking
    from .conftest import FIX
    rows = json.load(open(FIX / "dolthub_option_chain_SPY_20240103.json"))["rows"]
    c1 = dolthub.parse_option_rows(rows)
    store.write("option_chain", "dolthub", "SPY", c1, partition="year", time_col="ts", key=["ts", "symbol"])
    inst = portal.instrument(c1["symbol"].iloc[0])
    calls = []

    def fake(instrument, start, end, as_of, portal_, real_first=True, **kw):
        calls.append(real_first)
        f = pd.DataFrame({"bid": [1.0], "ask": [1.2], "fidelity": [int(Fidelity.T3_MODELED)]},
                         index=pd.DatetimeIndex([utc("2024-01-08 21:00")], name="ts"))
        f.attrs.update(fidelity=Fidelity.T3_MODELED, synthetic=True)
        return f
    monkeypatch.setattr(marking, "synthetic_option_quotes", fake)
    real = portal.option_quotes(inst, utc("2024-01-03 12:00"), utc("2024-01-05"), as_of=utc("2024-01-05"))
    assert len(real) and not calls and not real.attrs.get("synthetic")       # real rows -> no synthesis
    syn = portal.option_quotes(inst, utc("2024-01-05"), utc("2024-01-09"), as_of=utc("2024-01-09"))
    assert calls == [False] and syn.attrs["synthetic"] and (syn["fidelity"] == int(Fidelity.T3_MODELED)).all()
    assert portal.option_quotes(inst, utc("2024-01-05"), utc("2024-01-09"), as_of=utc("2024-01-09"),
                                synthetic=False).empty


def test_options_has_honours_freq(store, portal):
    import json
    from atlasbt.data.providers import dolthub
    from .conftest import FIX
    rows = json.load(open(FIX / "dolthub_option_chain_SPY_20240103.json"))["rows"]
    store.write("option_chain", "dolthub", "SPY", dolthub.parse_option_rows(rows), partition="year",
                time_col="ts", key=["ts", "symbol"])
    assert portal.has("SPY", "options", "1d") and portal.has("*", "options", "1d")
    assert not portal.has("SPY", "options", "1m") and not portal.has("*", "options", "1m")
    assert portal.fidelity_of("SPY", "options", "1m") == Fidelity.NA
    assert portal.fidelity_of("SPY", "options", "1d") == Fidelity.T2_PROXY
    # a timestamped intraday OPRA store (e.g. Databento when keyed) is T0 and honours freq
    q = pd.DataFrame({"symbol": ["SPY   240119C00470000"], "bid": [1.0], "ask": [1.1],
                      "ts": [utc("2024-01-03 15:00")], "known_at": [utc("2024-01-03 15:00")]})
    store.write("option_bars_1m", "databento", "SPY", q, time_col="ts", key=["ts", "symbol"])
    assert portal.has("SPY", "options", "1m") and portal.fidelity_of("SPY", "options", "1m") == Fidelity.T0_QUOTE
    assert portal.fidelity_of("SPY", "options", "1d") == Fidelity.T2_PROXY
    assert not portal.has("QQQ", "options", "1m") and portal.fidelity_of("QQQ", "options", "1m") == Fidelity.NA


def test_lookahead_perturbation_bars(store):
    """Perturb every bar known after t: what the portal serves at t must not change."""
    load_yahoo(store, "SPY")
    t = utc("2020-11-27 18:00")                     # early-close bar just became known
    h1 = Portal(store).bars("SPY", "1d", "2019-01-01", "2021-01-01", as_of=t, adjusted=True)
    raw = store.read("bars_1d", "yahoo", "SPY")
    pert = raw.copy()
    late = pert["known_at"] > t
    pert.loc[late, ["open", "high", "low", "close"]] *= 3.0
    store.write("bars_1d", "yahoo", "SPY", pert, replace=True)
    acts = store.read("actions", "yahoo", "SPY")
    extra = pd.DataFrame([{"symbol": "SPY", "type": "SPLIT", "ex_date": pd.Timestamp("2020-11-30"),
                           "known_at": utc("2020-11-27 18:01"), "ratio": 10.0, "amount": 0.0, "fidelity": 2}])
    store.write("actions", "yahoo", "SPY", pd.concat([acts, extra]), key=["ex_date", "type"], replace=True)
    h2 = Portal(store).bars("SPY", "1d", "2019-01-01", "2021-01-01", as_of=t, adjusted=True)
    pd.testing.assert_frame_equal(h1, h2)
    assert h1.index[-1] == utc("2020-11-27 14:30")


def test_earnings_events_pit(store, portal):
    from atlasbt.data.providers import ziplime
    from .conftest import FIX
    ev = ziplime.build_events(pd.read_parquet(FIX / "ziplime_announcements_subset.parquet"),
                              pd.read_parquet(FIX / "ziplime_earnings_subset.parquet"))
    store.write("events", "ziplime", "EARNINGS", ev, partition="year", time_col="known_at", key=["accession", "symbol"])
    est = pd.DataFrame({"date": pd.to_datetime(["2023-01-22", "2023-01-29"]), "act_symbol": "AAPL",
                        "period": "Current Quarter", "period_end_date": pd.Timestamp("2022-12-31"),
                        "consensus": [1.95, 1.94]})
    store.write("reference", "dolthub", "eps_estimate", est, key=["date", "act_symbol", "period"])
    w = ("2023-01-01", "2023-03-01")
    assert portal.events("EARNINGS", ["AAPL"], *w, as_of=utc("2023-02-02 21:30:32")) == []
    out = portal.events("EARNINGS", ["AAPL"], *w, as_of=utc("2023-02-02 21:30:33"))
    assert len(out) == 1
    p = out[0].payload
    assert p["timing"] == "AMC" and p["eps_actual"] == 1.88
    assert p["eps_estimate"] == 1.94 and pd.Timestamp(p["estimate_known_at"]) < out[0].known_at


def test_intraday_adjusted_as_of(store, portal):
    """synthetic_: 2:1 split ex 2024-01-04 (known at the 2024-01-03 close). Intraday history of
    2024-01-03 is halved only once the ex-date session has started."""
    _synthetic_intraday(store)
    acts = pd.DataFrame([{"symbol": "SYN", "type": "SPLIT", "ex_date": pd.Timestamp("2024-01-04"),
                          "known_at": utc("2024-01-03 21:00"), "ratio": 2.0, "amount": 0.0, "fidelity": 2}])
    store.write("actions", "yahoo", "SYN", acts, key=["ex_date", "type"])
    raw = portal.bars("SYN", "1m", "2024-01-03", "2024-01-04", as_of=utc("2024-01-04 01:00"))
    before = portal.bars("SYN", "1m", "2024-01-03", "2024-01-04", as_of=utc("2024-01-04 01:00"), adjusted=True)
    pd.testing.assert_frame_equal(raw, before)                        # ex-date session not reached yet
    after = portal.bars("SYN", "1m", "2024-01-03", "2024-01-04", as_of=utc("2024-01-04 14:00"), adjusted=True)
    assert np.allclose(after["close"], raw["close"] / 2) and np.allclose(after["volume"], raw["volume"] * 2)


def test_chain_annotation_last_trading_day():
    from atlasbt.data.option_meta import annotate_chain
    df = pd.DataFrame({"symbol": ["SPY   250418C00500000", "SPX   190816C02900000", "SPXW  190816C02900000",
                                  "SPY   240117C00469000"],
                       "expiry": pd.to_datetime(["2025-04-18", "2019-08-16", "2019-08-16", "2024-01-17"]),
                       "underlying_price": [np.nan, 2847.6, 2847.6, 468.79]})
    a = annotate_chain(df)
    # Good Friday 2025-04-18 holiday -> last trading day Thursday; SPX AM monthly -> Thursday,
    # settles Friday; SPXW PM -> Friday; ordinary weekly unchanged
    assert list(a["expiry"]) == list(pd.to_datetime(["2025-04-17", "2019-08-15", "2019-08-16", "2024-01-17"]))
    assert list(a["settlement_date"]) == list(pd.to_datetime(["2025-04-17", "2019-08-16", "2019-08-16", "2024-01-17"]))
    assert list(a["kind"]) == ["MONTHLY", "MONTHLY", "MONTHLY", "WEEKLY"]
    assert list(a["root"]) == ["SPY", "SPX", "SPXW", "SPY"]


def test_instrument_parse_spx_am(portal):
    i = portal.instrument("SPX   190816C02900000")
    assert i.expiry == pd.Timestamp("2019-08-15") and i.settlement.value == "CASH_AM" and i.exercise.value == "EUROPEAN"
    assert portal.instrument("SPXW  190819P02800000").settlement.value == "CASH_PM"
    assert portal.instrument("SPY   240117C00469000").expiry == pd.Timestamp("2024-01-17")


def test_synthetic_fallback_is_flagged_and_never_mixed(store, portal):
    """No real chain in the store -> synthetic chain (atlasbt.options.synthetic, reads the real
    Cboe index files under data_raw/) with fidelity 3; with synthetic=False -> empty."""
    load_yahoo(store, "SPY")
    as_of = utc("2024-03-19 20:30")
    assert portal.option_chain("SPY", as_of, synthetic=False).empty
    c = portal.option_chain("SPY", as_of, min_dte=20, max_dte=40)
    assert len(c) and c.attrs["synthetic"] is True
    assert (c["fidelity"] == int(Fidelity.T3_MODELED)).all()
    assert c["underlying_price"].iloc[0] == pytest.approx(
        store.read("bars_1d", "yahoo", "SPY")["close"].iloc[-1])            # PIT last raw close
    # real chain present -> real rows only
    import json
    from atlasbt.data.providers import dolthub
    from .conftest import FIX
    rows = json.load(open(FIX / "dolthub_option_chain_SPY_20240103.json"))["rows"]
    store.write("option_chain", "dolthub", "SPY", dolthub.parse_option_rows(rows), partition="year", time_col="ts",
                key=["ts", "symbol"])
    r = Portal(store).option_chain("SPY", utc("2024-01-04 15:00"))
    assert r.attrs["synthetic"] is False and (r["fidelity"] == int(Fidelity.T2_PROXY)).all()


def test_resample_early_close_and_alignment(store, portal):
    """synthetic_ 1m bars on the 2023-11-24 half day (close 13:00 ET = 18:00 UTC)."""
    sess = pd.Timestamp("2023-11-24")
    o, c = frames.schedule().loc[sess, ["open", "close"]]
    assert c == utc("2023-11-24 18:00")
    idx = pd.date_range(o, c, freq="1min", inclusive="left")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    store.write("bars_1m", "yahoo", "HALF", frames.intraday_bars(df, "1m"), partition="year")
    b5 = portal.bars("HALF", "5m", "2023-11-24", "2023-11-25", as_of=utc("2023-11-25"))
    assert b5.index[0] == utc("2023-11-24 14:30") and len(b5) == 42                 # 09:30 .. 12:55 ET
    assert ((b5.index - utc("2023-11-24 14:30")) % pd.Timedelta(minutes=5) == pd.Timedelta(0)).all()
    b60 = portal.bars("HALF", "60m", "2023-11-24", "2023-11-25", as_of=utc("2023-11-25"))
    assert b60.index[-1] == utc("2023-11-24 17:30") and b60["known_at"].iloc[-1] == c   # capped at the close
    assert b60["volume"].iloc[-1] == 30


def test_top_dollar_volume_uses_prior_sessions_only(store):
    """synthetic_: B overtakes A only on the as_of session itself -> must not count."""
    m = pd.DataFrame({"ticker": ["A", "B"], "base": ["A", "B"], "start": pd.to_datetime(["2000-01-03"] * 2),
                      "end": pd.NaT, "historical_identity": False, "left_censored": False, "yahoo": ["A", "B"]})
    store.write("reference", "sp500", "membership", m, key=["ticker", "start"])
    d = frames.schedule().loc["2024-01-02":"2024-02-15"].index
    for sym, vol in (("A", 2e6), ("B", 1e6)):
        v = np.full(len(d), vol)
        if sym == "B":
            v[-1] = 1e9                                   # huge volume on the last session only
        df = pd.DataFrame({"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": v}, index=d)
        store.write("bars_1d", "yahoo", sym, frames.daily_bars(df)[0])
    p = Portal(store)
    assert p.universe("SP500_TOP1_DV", utc(f"{d[-1].date()} 20:30")) == ["A"]
    assert p.universe("SP500_TOP1_DV", utc(f"{d[-1].date() + pd.Timedelta(days=4)} 15:00")) == ["B"]


def test_fundamentals_history_pit(store, portal):
    """synthetic_: quarterly facts incl. a restatement and a prior-year comparative."""
    f = pd.DataFrame({"symbol": "X", "field": "revenue_q", "value": [10.0, 11.0, 12.0, 10.5],
                      "period_end": pd.to_datetime(["2023-03-31", "2023-06-30", "2024-03-31", "2023-03-31"]),
                      "period_start": pd.to_datetime(["2023-01-01", "2023-04-01", "2024-01-01", "2023-01-01"]),
                      "known_at": [utc("2023-05-01 20:00"), utc("2023-08-01 20:00"), utc("2024-05-01 20:00"),
                                   utc("2024-05-01 20:00")],          # comparative restated in the 2024 10-Q
                      "fp": ["Q1", "Q2", "Q1", "Q1"], "fy": [2023, 2023, 2024, 2024], "form": "10-Q",
                      "accn": ["a1", "a2", "a3", "a3"], "source": "sec"})
    store.write("fundamentals", "sec", "X", f, key=["field", "period_start", "period_end", "accn"])
    h = portal.fundamentals_history(["X"], ["revenue_q"], as_of=utc("2023-12-31"))
    assert list(h["value"]) == [10.0, 11.0] and "fiscal_period" in h.columns
    h = portal.fundamentals_history(["X"], ["revenue_q"], as_of=utc("2024-06-01"))
    assert len(h) == 4 and (h["known_at"] <= utc("2024-06-01")).all()
    q1 = h[h["period_end"] == pd.Timestamp("2023-03-31")]
    assert list(q1["value"]) == [10.0, 10.5]                     # both vintages, in known_at order
    h = portal.fundamentals_history(["X"], ["revenue_q"], as_of=utc("2024-06-01"), start="2024-01-01")
    assert list(h["value"]) == [12.0]
    # fundamentals() unchanged: latest period only
    assert portal.fundamentals(["X"], ["revenue_q"], utc("2024-06-01"))["value"].iloc[0] == 12.0


def test_sec_book_equity_fields_from_fixture():
    import json
    from atlasbt.data.providers import sec
    from .conftest import FIX
    assert {"preferred_stock", "deferred_taxes", "short_term_debt", "stock_issuance_proceeds",
            "repurchase"} <= set(sec.FIELD_MAP)
    facts = json.load(open(FIX / "sec_companyfacts_AAPL_subset.json"))
    facts["facts"]["us-gaap"]["DeferredTaxLiabilitiesNoncurrent"] = {"units": {"USD": [
        {"end": "2023-09-30", "val": 123, "accn": "x1", "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2023-11-03"}]}}
    facts["facts"]["us-gaap"]["DeferredIncomeTaxLiabilitiesNet"] = {"units": {"USD": [
        {"end": "2023-09-30", "val": 999, "accn": "x1", "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2023-11-03"}]}}
    fund = sec.parse_companyfacts("AAPL", facts, {})
    dt = fund[fund["field"] == "deferred_taxes"]
    assert list(dt["value"]) == [123.0] and dt["concept"].iloc[0] == "DeferredTaxLiabilitiesNoncurrent"   # fallback order


def test_earnings_gaap_vs_street_eps(store, portal):
    """ZipLime gives GAAP diluted EPS; DoltHub eps_history gives street EPS + consensus. Both are
    exposed; eps_actual (for surprise vs consensus) is the street figure when available."""
    from atlasbt.data.providers import ziplime
    from .conftest import FIX
    ev = ziplime.build_events(pd.read_parquet(FIX / "ziplime_announcements_subset.parquet"),
                              pd.read_parquet(FIX / "ziplime_earnings_subset.parquet"))
    store.write("events", "ziplime", "EARNINGS", ev, partition="year", time_col="known_at", key=["accession", "symbol"])
    import json
    h = pd.DataFrame(json.load(open(FIX / "dolthub_earnings_AAPL.json"))["eps_history"])
    store.write("reference", "dolthub", "eps_history", h, key=["act_symbol", "period_end_date"])
    out = portal.events("EARNINGS", ["AAPL"], "2023-01-01", "2023-03-01", as_of=utc("2023-03-01"))
    p = out[0].payload
    assert p["eps_actual_gaap"] == 1.88 and p["eps_actual_street"] == 1.88   # AAPL reports no adjustments
    assert p["eps_actual_basis"] == "street" and p["eps_actual"] == p["eps_actual_street"]
    assert p["eps_estimate"] == 1.93 and p["estimate_fidelity"] == 2         # final consensus, post hoc
    assert p["estimate_reason"] == "consensus recorded post hoc; revisions not PIT"


def test_fundamentals_ignore_future_period_end(store, portal):
    """A fact tagged with a period_end far after its filing (mis-tagged XBRL context) must not
    become 'the latest period' and block newer filings."""
    f = pd.DataFrame({"symbol": "X", "field": "shares_outstanding", "value": [1.0, 2.0],
                      "period_end": pd.to_datetime(["2027-07-17", "2026-09-30"]), "period_start": pd.NaT,
                      "known_at": [utc("2026-07-23 20:00"), utc("2026-10-25 20:00")], "source": "sec",
                      "accn": ["a1", "a2"]})
    store.write("fundamentals", "sec", "X", f, key=["field", "period_start", "period_end", "accn"])
    r = portal.fundamentals(["X"], ["shares_outstanding"], utc("2026-10-28"))
    assert r["value"].iloc[0] == 2.0 and r["period_end"].iloc[0] == pd.Timestamp("2026-09-30")
