"""Databento OPRA provider: budget ledger, CSV parsing, OSI -> internal symbols, per-day quote store,
Portal serving (chain freshness / chain flag / T0 quotes / no synthetic substitute), SPX parity levels.
Offline: no network (requests are monkeypatched)."""
from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.types import Fidelity
from atlasbt.data import config
from atlasbt.data.providers import databento as D
from atlasbt.data.store import ParquetStore


def utc(s):
    return pd.Timestamp(s, tz="UTC")


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(config, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(config, "STORE_ROOT", tmp_path / "store")
    monkeypatch.setenv("DATABENTO_API_KEY", "test-key-not-real")
    monkeypatch.delenv("ATLASBT_OFFLINE", raising=False)
    from atlasbt.data import store as store_mod
    monkeypatch.setattr(store_mod, "_DEFAULT", None)      # never touch the real cache store
    D._LISTING_MEMO.clear()
    return tmp_path


def cbbo_csv(rows):
    """rows: (ts_recv, ts_event, raw_symbol, bid, ask)."""
    head = "ts_recv,ts_event,rtype,publisher_id,instrument_id,side,price,size,flags,bid_px_00,ask_px_00,bid_sz_00," \
           "ask_sz_00,bid_pb_00,ask_pb_00,symbol\n"
    body = "".join(f"{a},{b},193,30,1,N,1.0,1,200,{bid},{ask},10,12,0,0,{sym}\n" for a, b, sym, bid, ask in rows)
    return (head + body).encode()


# ---------------------------------------------------------------- symbology
def test_osi_to_internal_spx_am_and_spxw_pm():
    m = D.internal_map(["SPX   240119C04750000", "SPXW  240102P04700000", "AAPL  240119C00190000"]).set_index("raw_symbol")
    # AM monthly: OPRA lists the settlement Friday; internal expiry = Thursday last trading day
    am = m.loc["SPX   240119C04750000"]
    assert am["expiry"] == pd.Timestamp("2024-01-18") and am["settlement"] == "CASH_AM" and am["exercise"] == "EUROPEAN"
    assert am["symbol"] == "SPX   240118C04750000" and am["underlying"] == "SPX"
    pm = m.loc["SPXW  240102P04700000"]
    assert pm["expiry"] == pd.Timestamp("2024-01-02") and pm["settlement"] == "CASH_PM" and pm["underlying"] == "SPX"
    assert pm["strike"] == 4700.0 and pm["right"] == "P"
    eq = m.loc["AAPL  240119C00190000"]
    assert eq["settlement"] == "PHYSICAL" and eq["exercise"] == "AMERICAN" and eq["strike"] == 190.0
    # the internal symbol round-trips through the Portal's OCC parser
    from atlasbt.data.portal import Portal
    inst = Portal.instrument(Portal.__new__(Portal), "SPX   240118C04750000")
    assert inst.expiry == pd.Timestamp("2024-01-18") and inst.settlement.value == "CASH_AM"


def test_opra_root_mapping():
    assert D.opra_root("BRK-B") == "BRKB" and D.opra_root("ACN~CIK0001467373") == "ACN"


# ---------------------------------------------------------------- budget / ledger
def test_budget_guard_refuses_and_ledger_records(cache, monkeypatch):
    calls = []

    class R:
        def __init__(self, text=b"", content=b""):
            self.text, self.content, self.status_code, self.headers = text, content, 200, {}

    def fake_post(endpoint, data, timeout=600):
        calls.append(endpoint)
        if endpoint == "metadata.get_cost":
            return R(text="0.6")
        return R(content=cbbo_csv([("2024-01-02T14:31:00Z", "2024-01-02T14:30:59Z", "SPXW  240102C04700000", 1, 2)]))

    monkeypatch.setattr(D, "_post", fake_post)
    monkeypatch.setenv("DATABENTO_BUDGET_USD", "1.0")
    body = D.get_range_csv(D.OPRA, "cbbo-1m", ["SPXW  240102C04700000"], utc("2024-01-02 14:30"), utc("2024-01-02 14:32"))
    assert b"SPXW" in body
    lg = D.read_ledger()
    assert list(lg["status"]) == ["requested"] and abs(D.spent_usd() - 0.6) < 1e-9
    # same request again: served from the raw cache, no new charge
    D.get_range_csv(D.OPRA, "cbbo-1m", ["SPXW  240102C04700000"], utc("2024-01-02 14:30"), utc("2024-01-02 14:32"))
    assert calls.count("timeseries.get_range") == 1 and len(D.read_ledger()) == 1
    # a second distinct request would exceed the $1 budget -> refused, logged, nothing downloaded
    with pytest.raises(D.BudgetExceeded):
        D.get_range_csv(D.OPRA, "cbbo-1m", ["SPXW  240102C04705000"], utc("2024-01-02 14:30"), utc("2024-01-02 14:32"))
    lg = D.read_ledger()
    assert list(lg["status"]) == ["requested", "refused"] and calls.count("timeseries.get_range") == 1
    assert abs(D.spent_usd() - 0.6) < 1e-9
    assert "test-key-not-real" not in (cache / "raw" / "databento_ledger.csv").read_text()


def test_offline_mode_blocks_requests(cache, monkeypatch):
    monkeypatch.setenv("ATLASBT_OFFLINE", "1")
    with pytest.raises(RuntimeError):
        D.get_cost(D.OPRA, "cbbo-1m", "SPY.OPT", "2024-01-02", "2024-01-03", "parent")


# ---------------------------------------------------------------- store + portal
def _sample_day(root):
    """SPXW 0DTE contracts on 2024-01-02 (+ one 30-DTE mark), samples at 15:00 and 15:01 UTC (10:00/10:01 ET)."""
    rows = []
    for ts, F in (("2024-01-02T15:00:00Z", 4750.0), ("2024-01-02T15:01:00Z", 4752.0)):
        for K in (4740, 4745, 4750, 4755, 4760):
            c = max(F - K, 0) + 2.0
            p = max(K - F, 0) + 2.0
            rows.append((ts, "2024-01-02T14:59:59Z", f"SPXW  240102C0{K}000", round(c - 0.05, 2), round(c + 0.05, 2)))
            rows.append((ts, "2024-01-02T14:59:59Z", f"SPXW  240102P0{K}000", round(p - 0.05, 2), round(p + 0.05, 2)))
    raw = D.parse_cbbo_csv(cbbo_csv(rows))
    D.write_rows(D.to_store_rows(raw, chain=True), root)
    mark = D.parse_cbbo_csv(cbbo_csv([("2024-01-02T21:00:00Z", "2024-01-02T20:59:00Z", "SPX   240216C04750000", 80, 81)]))
    D.write_rows(D.to_store_rows(mark, chain=False), root)


def test_parse_and_day_store(cache):
    root = cache / "store"
    _sample_day(root)
    f = pd.read_parquet(D.day_path("SPX", "2024-01-02", root))
    assert len(f) == 21 and set(f.columns) == set(D.QCOLS)
    assert f["ts"].min() == utc("2024-01-02 15:00") and f["age_s"].iloc[0] == pytest.approx(1.0)
    # idempotent upsert
    _sample_day(root)
    assert len(pd.read_parquet(D.day_path("SPX", "2024-01-02", root))) == 21


def test_parity_spot_recovers_forward(cache):
    root = cache / "store"
    _sample_day(root)
    f = D.OpraReader(root).frame("SPX", "2024-01-02")
    lvl, exp, n = D.parity_spot(f[f["ts"] == utc("2024-01-02 15:00")])
    assert lvl == pytest.approx(4750.0) and exp == pd.Timestamp("2024-01-02") and n == 5


def test_portal_serves_opra_chain_quotes_and_bars(cache):
    from atlasbt.data.portal import Portal
    root = cache / "store"
    _sample_day(root)
    D.rebuild_spx_parity_bars([pd.Timestamp("2024-01-02")], root)
    p = Portal(ParquetStore(root))
    # intraday availability / tier for the planner
    assert p.has("SPX", "options", "1m") and p.fidelity_of("SPX", "options", "1m") == Fidelity.T0_QUOTE
    assert p.has("SPX", "options") and p.fidelity_of("SPX", "options", "1d") == Fidelity.T0_QUOTE
    ch = p.option_chain("SPX", utc("2024-01-02 15:00:30"))
    assert ch.attrs["source"] == "databento" and not ch.attrs["synthetic"]
    assert ch.attrs["snapshot_ts"] == utc("2024-01-02 15:00") and (ch["ts"] <= utc("2024-01-02 15:00:30")).all()
    assert (ch["fidelity"] == int(Fidelity.T0_QUOTE)).all() and (ch["settlement"] == "CASH_PM").all()
    assert ch["underlying_price"].iloc[0] == pytest.approx(4750.0)
    # point in time: nothing before the first sample; the 15:01 sample is not visible at 15:00:59
    assert p.option_chain("SPX", utc("2024-01-02 15:00:59")).attrs["snapshot_ts"] == utc("2024-01-02 15:00")
    # stale (> 15 min) intraday chain is not served as a chain; close-time marks (chain=False) never are
    late = p.option_chain("SPX", utc("2024-01-02 21:00"), synthetic=False)
    assert late.empty or late.attrs.get("source") != "databento"
    inst = p.instrument("SPXW  240102C04750000")
    q = p.option_quote(inst, utc("2024-01-02 15:01:30"))
    assert q.fidelity == Fidelity.T0_QUOTE and q.ts == utc("2024-01-02 15:01") and q.ask > q.bid
    qs = p.option_quotes(inst, utc("2024-01-02 15:00"), utc("2024-01-02 15:05"), utc("2024-01-02 15:05"))
    assert list(qs.index) == [utc("2024-01-02 15:01")] and (qs["fidelity"] == 0).all()
    # covered contract, window without samples: no synthetic substitute
    q2 = p.option_quotes(inst, utc("2024-01-02 17:00"), utc("2024-01-02 21:00"), utc("2024-01-02 21:00"))
    assert len(q2) == 0
    # mark rows serve quotes (valuation / fills) even though they are not a chain
    mk = p.instrument("SPX   240215C04750000")
    assert p.option_quote(mk, utc("2024-01-02 21:00")).bid == 80
    # SPX minute levels from parity (T2 proxy bars)
    b = p.bars("SPX", "1m", utc("2024-01-02 14:00"), utc("2024-01-02 21:00"), as_of=utc("2024-01-02 21:00"))
    assert b.attrs["source"] == "opra_parity" and p.fidelity_of("SPX", "bars", "1m") == Fidelity.T2_PROXY
    assert b["close"].iloc[-1] == pytest.approx(4752.0) and b["known_at"].iloc[-1] == utc("2024-01-02 15:01")


def test_label_and_sample_windows():
    # early close 2023-11-24 (13:00 ET): 15:30 label shifts to 12:30
    assert D.label_utc("2023-11-24", "15:30") == pd.Timestamp("2023-11-24 12:30", tz="America/New_York").tz_convert("UTC")
    a, b = D.samples_window(utc("2024-01-02 15:00"), utc("2024-01-02 15:01"))
    assert a == utc("2024-01-02 14:59") and b == utc("2024-01-02 15:02")


def test_closing_sample_is_one_minute_before_close_and_coverage(cache):
    _o, c = D.open_close("2024-01-31")
    a, b = D.samples_window(*D._closing(c))
    assert a == utc("2024-01-31 20:58") and b == utc("2024-01-31 21:00")      # one sample: 20:59 (15:59 ET)
    root = cache / "store"
    _sample_day(root)
    cov = D.coverage("SPX", root)
    assert cov["first"] == cov["last"] == "2024-01-02" and cov["n_days"] == 1


def test_run_jobs_chain_cutoff_and_progress(cache, monkeypatch):
    rows = [("2024-01-02T15:00:00Z", "2024-01-02T14:59:59Z", "SPXW  240102C04750000", 1.0, 1.1),
            ("2024-01-02T20:50:00Z", "2024-01-02T20:49:59Z", "SPXW  240102C04750000", 1.0, 1.1)]
    monkeypatch.setattr(D, "listing", lambda parents, day: D.parse_osi(["SPXW  240102C04750000"]))
    monkeypatch.setattr(D, "get_range_csv", lambda *a, **k: cbbo_csv(rows))
    job = D.Job("t|2024-01-02", ["SPXW"], "2024-01-02", lambda l: l["raw_symbol"].tolist(),
                [(utc("2024-01-02 14:30"), utc("2024-01-02 21:01"))], chain_before=utc("2024-01-02 20:40"))
    out = D.run_jobs([job])
    assert out["rows"] == 2 and "t|2024-01-02" in D.done_keys()
    f = pd.read_parquet(D.day_path("SPX", "2024-01-02"))
    assert f.set_index("ts")["chain"].to_dict() == {utc("2024-01-02 15:00"): True, utc("2024-01-02 20:50"): False}
    assert D.run_jobs([job])["skipped_done"] == 1


# ---------------------------------------------------------------- point in time (cbbo-1m stamps)
def _write(root, rows, chain=True):
    D.write_rows(D.to_store_rows(D.parse_cbbo_csv(cbbo_csv(rows)), chain=chain), root)


def test_cbbo_minute_record_visible_only_at_interval_end(cache):
    """Databento stamps the cbbo-1m record of the minute [T, T+1m) with ts_recv = T+1m (the interval END,
    docs: 'the end timestamp of the interval, clamped to the minute boundary'). That record carries the
    book at T+1m, so it must not be visible at any as_of < T+1m (the same rule as bar known_at = bar end)."""
    from atlasbt.data.portal import Portal
    root = cache / "store"
    sym = "SPXW  240102C04750000"
    # minute [15:00, 15:01) -> record stamped 15:01 with the book after a change at 15:00:40
    _write(root, [("2024-01-02T15:00:00Z", "2024-01-02T14:59:10Z", sym, 1.00, 1.10),
                  ("2024-01-02T15:01:00Z", "2024-01-02T15:00:40Z", sym, 2.00, 2.10)])
    p = Portal(ParquetStore(root))
    inst = p.instrument(sym)
    for as_of, bid in (("2024-01-02 15:00:00", 1.00), ("2024-01-02 15:00:40", 1.00),
                       ("2024-01-02 15:00:59.999999", 1.00), ("2024-01-02 15:01:00", 2.00)):
        q = p.option_quote(inst, utc(as_of))
        assert q.bid == bid and q.ts <= utc(as_of), as_of
    qs = p.option_quotes(inst, utc("2024-01-02 14:00"), utc("2024-01-02 16:00"), utc("2024-01-02 15:00:59"))
    assert list(qs.index) == [utc("2024-01-02 15:00")] and (qs.index <= utc("2024-01-02 15:00:59")).all()
    ch = p.option_chain("SPX", utc("2024-01-02 15:00:59"))
    assert ch.attrs["snapshot_ts"] == utc("2024-01-02 15:00") and float(ch["bid"].iloc[0]) == 1.00
    # nothing at all before the first record
    assert p.option_quote(inst, utc("2024-01-02 14:59:59")) is None


def test_parse_cbbo_requires_ts_recv_and_ceils_off_grid_stamps():
    # ts_event is the LAST TRADE time: using it as the knowledge time would date a later book too early
    head = b"ts_event,bid_px_00,ask_px_00,bid_sz_00,ask_sz_00,symbol\n2024-01-02T15:00:00Z,1,2,1,1,SPXW  240102C04750000\n"
    with pytest.raises(ValueError):
        D.parse_cbbo_csv(head)
    # a stamp off the minute grid is moved to the NEXT boundary (never earlier)
    raw = D.parse_cbbo_csv(cbbo_csv([("2024-01-02T15:00:00.5Z", "2024-01-02T14:59:00Z", "SPXW  240102C04750000", 1, 2),
                                     ("2024-01-02T15:01:00Z", "2024-01-02T14:59:00Z", "SPXW  240102C04750000", 1, 2)]),
                           interval="1min")
    assert list(raw["ts"]) == [utc("2024-01-02 15:01"), utc("2024-01-02 15:01")]
    # cbbo-1s keeps its own grid
    raw = D.parse_cbbo_csv(cbbo_csv([("2024-01-02T15:00:00.5Z", "2024-01-02T14:59:00Z", "SPXW  240102C04750000", 1, 2)]),
                           interval="1s")
    assert list(raw["ts"]) == [utc("2024-01-02 15:00:01")]


def test_synthetic_fallback_decision_uses_only_samples_known_at_as_of(cache, monkeypatch):
    """A contract whose first sample of the session comes AFTER as_of must not suppress the fallback at
    as_of (knowing that samples will exist later that day is future information)."""
    from atlasbt.data.portal import Portal
    import atlasbt.options.marking as marking
    root = cache / "store"
    sym = "SPXW  240102C04750000"
    _write(root, [("2024-01-02T18:00:00Z", "2024-01-02T17:59:00Z", sym, 1.0, 1.1)])
    calls = []

    def fake_syn(instrument, start, end, as_of, portal, real_first=True, **kw):
        calls.append(as_of)
        out = pd.DataFrame({"bid": [0.9], "ask": [1.2]}, index=pd.DatetimeIndex([utc("2024-01-02 16:00")], name="ts"))
        return out

    monkeypatch.setattr(marking, "synthetic_option_quotes", fake_syn)
    p = Portal(ParquetStore(root))
    inst = p.instrument(sym)
    q = p.option_quotes(inst, utc("2024-01-02 14:30"), utc("2024-01-02 16:00"), utc("2024-01-02 16:00"))
    assert len(calls) == 1 and q.attrs.get("synthetic") is True
    # once a sample is known that session, a window without samples gets NO synthetic substitute
    q2 = p.option_quotes(inst, utc("2024-01-02 18:30"), utc("2024-01-02 19:00"), utc("2024-01-02 19:00"))
    assert len(q2) == 0 and len(calls) == 1


# ---------------------------------------------------------------- SPX (AM) vs SPXW (PM) identity
def test_spx_am_and_spxw_pm_keep_root_and_settlement_through_serving(cache):
    from atlasbt.core.types import SettlementType
    from atlasbt.data.portal import Portal
    from atlasbt.pricing.time import settlement_instant, settlement_session
    root = cache / "store"
    ts = "2024-01-10T20:59:00Z"                          # 15:59 ET closing sample
    rows = []
    # SPX AM monthly (OPRA lists the settlement Friday 01-19), SPXW PM Thursday 01-18, SPXW PM Friday 01-19;
    # same strike everywhere so only the root separates them
    for vend in ("SPX   240119", "SPXW  240118", "SPXW  240119", "SPXW  240110"):
        for K in (4700, 4750, 4800):
            for r_ in "CP":
                rows.append((ts, "2024-01-10T20:58:00Z", f"{vend}{r_}0{K}000", 10.0, 10.5))
    _write(root, rows)
    p = Portal(ParquetStore(root))
    ch = p.option_chain("SPX", utc("2024-01-10 21:00"))
    assert ch.attrs["source"] == "databento" and len(ch) == 24
    by = ch.set_index("symbol")
    am = by.loc["SPX   240118C04750000"]                  # internal: last trading day Thursday
    assert am["root"] == "SPX" and am["settlement"] == "CASH_AM" and am["exercise"] == "EUROPEAN"
    assert pd.Timestamp(am["expiry"]) == pd.Timestamp("2024-01-18")
    assert pd.Timestamp(am["settlement_date"]) == pd.Timestamp("2024-01-19") and am["kind"] == "MONTHLY"
    thu = by.loc["SPXW  240118C04750000"]
    assert thu["root"] == "SPXW" and thu["settlement"] == "CASH_PM"
    assert pd.Timestamp(thu["settlement_date"]) == pd.Timestamp("2024-01-18")
    fri = by.loc["SPXW  240119C04750000"]
    assert fri["settlement"] == "CASH_PM" and pd.Timestamp(fri["settlement_date"]) == pd.Timestamp("2024-01-19")
    # the instrument the engine builds from a chain symbol settles like the chain row says
    for s, st, sess, inst_hhmm in (("SPX   240118C04750000", SettlementType.CASH_AM, "2024-01-19", "14:30"),
                                   ("SPXW  240118C04750000", SettlementType.CASH_PM, "2024-01-18", "21:00")):
        inst = p.instrument(s)
        assert inst.settlement == st and inst.underlying == "SPX"
        assert settlement_session(inst.expiry, inst.settlement) == pd.Timestamp(sess)
        assert settlement_instant(inst.expiry, inst.settlement) == utc(f"{sess} {inst_hhmm}")
        q = p.option_quote(inst, utc("2024-01-10 21:00"))
        assert q is not None and q.fidelity == Fidelity.T0_QUOTE and q.instrument.settlement == st
    # SPXW requested as its own underlying: only the PM root, still T0 from the SPX files
    w = p.option_chain("SPXW", utc("2024-01-10 21:00"))
    assert w.attrs["source"] == "databento" and set(w["root"]) == {"SPXW"} and (w["settlement"] == "CASH_PM").all()
    assert p.has("SPXW", "options", "1m") and p.fidelity_of("SPXW", "options", "1m") == Fidelity.T0_QUOTE
    # option_quotes rows carry the settlement identity too
    qs = p.option_quotes(p.instrument("SPX   240118P04700000"), utc("2024-01-10 14:00"), utc("2024-01-10 21:00"),
                         utc("2024-01-10 21:00"), synthetic=False)
    assert len(qs) == 1 and qs["settlement"].iloc[0] == "CASH_AM" and qs["root"].iloc[0] == "SPX"
    assert pd.Timestamp(qs["settlement_date"].iloc[0]) == pd.Timestamp("2024-01-19")


def test_opra_quotes_underlying_price_is_pit_intraday(cache):
    """OPRA quote rows get the underlying level known at each sample (1m bars ending <= ts), not the
    previous session's daily close."""
    from atlasbt.data.portal import Portal
    root = cache / "store"
    _sample_day(root)
    D.rebuild_spx_parity_bars([pd.Timestamp("2024-01-02")], root)
    p = Portal(ParquetStore(root))
    inst = p.instrument("SPXW  240102C04750000")
    qs = p.option_quotes(inst, utc("2024-01-02 14:00"), utc("2024-01-02 15:05"), utc("2024-01-02 15:05"),
                         synthetic=False)
    assert list(qs.index) == [utc("2024-01-02 15:00"), utc("2024-01-02 15:01")]
    # parity bar [14:59, 15:00) closes at the 15:00 level (known 15:00); [15:00, 15:01) at 15:01
    assert qs["underlying_price"].iloc[0] == pytest.approx(4750.0)
    assert qs["underlying_price"].iloc[1] == pytest.approx(4752.0)
    assert (qs["underlying_price_source"] == "pit_1m_bar_close_at_ts").all()


# ---------------------------------------------------------------- coverage (planner hook)
def test_coverage_by_underlying_and_minutes(cache):
    from atlasbt.data.portal import Portal
    root = cache / "store"
    _sample_day(root)
    _write(root, [("2024-01-03T20:59:00Z", "2024-01-03T20:58:00Z", "SPY   240119C00470000", 1.0, 1.1)])
    cov = D.coverage(root=root)
    assert cov["underlyings"] == 2 and cov["n_days"] == 2 and cov["first"] == "2024-01-02"
    assert set(cov["by_underlying"]) == {"SPX", "SPY"}
    assert cov["by_underlying"]["SPX"]["days"] == ["2024-01-02"]
    det = D.coverage("SPX", root=root, detail=True)["minutes"]
    r = det.set_index("day").loc["2024-01-02"]
    # 15:00, 15:01 chain samples + one 21:00 mark
    assert r["n_ts"] == 3 and r["n_chain_ts"] == 2 and r["first_ts"] == utc("2024-01-02 15:00")
    assert r["last_ts"] == utc("2024-01-02 21:00") and set(r["roots"].split(",")) == {"SPX", "SPXW"}
    mins = D.coverage_minutes("SPX", "2024-01-02", root=root)
    assert list(mins) == [utc("2024-01-02 15:00"), utc("2024-01-02 15:01"), utc("2024-01-02 21:00")]
    assert list(D.coverage_minutes("SPX", "2024-01-02", root=root, chain_only=True)) == \
        [utc("2024-01-02 15:00"), utc("2024-01-02 15:01")]
    assert list(D.coverage_minutes("SPX", "2024-01-02", root=root, symbol="SPX   240215C04750000")) == \
        [utc("2024-01-02 21:00")]
    assert len(D.coverage_minutes("QQQ", "2024-01-02", root=root)) == 0
    # the Portal exposes the same view for the planner, rooted at its own store
    pc = Portal(ParquetStore(root)).opra_coverage("SPXW")
    assert pc["n_days"] == 1 and pc["underlyings"] == 1


# ---------------------------------------------------------------- fetch task databento_options (offline)
class _Resp:
    def __init__(self, text="", content=b"", js=None):
        self.text, self.content, self.status_code, self.headers, self._js = text, content, 200, {}, js

    def json(self):
        return self._js


def _fake_client(calls, cost=0.01, listed=("SPXW  240102C04750000", "SPXW  240102P04750000",
                                            "SPXW  240103C04750000")):
    def fake_post(endpoint, data, timeout=600):
        calls.append((endpoint, dict(data)))
        if endpoint == "symbology.resolve":
            return _Resp(js={"result": {s: [] for s in listed}})
        if endpoint == "metadata.get_cost":
            return _Resp(text=str(cost))
        if endpoint == "timeseries.get_range":
            day = data["start"][:10]
            syms = data["symbols"].split(",")
            rows = [(f"{day}T15:00:00Z", f"{day}T14:59:30Z", s, 1.0, 1.2) for s in syms]
            return _Resp(content=cbbo_csv(rows))
        raise AssertionError(endpoint)
    return fake_post


def _ref(days):
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    return pd.DataFrame({"ratio": 10.0, "hi": 4760.0, "lo": 4740.0, "close": 4750.0, "prev_close": 4750.0,
                         "vix_prev": 13.0}, index=idx)


def test_fetch_task_databento_options_offline(cache, monkeypatch):
    from atlasbt.data import fetch
    calls = []
    monkeypatch.setattr(D, "_post", _fake_client(calls))
    monkeypatch.setattr(D, "spx_reference", lambda portal, start, end: _ref(["2024-01-02", "2024-01-03"]))
    monkeypatch.setattr(D, "store_spx_parity_bars", lambda raw, day: 0)
    monkeypatch.setenv("ATLASBT_DATABENTO_PLANS", "spx0dte")
    monkeypatch.setenv("ATLASBT_DATABENTO_START", "2024-01-02")
    monkeypatch.setenv("ATLASBT_DATABENTO_END", "2024-01-03")
    monkeypatch.setenv("DATABENTO_BUDGET_USD", "71.58")
    monkeypatch.delenv("ATLASBT_DATABENTO_DRY_RUN", raising=False)
    out = fetch.task_databento_options()
    r = out["spx0dte"]
    assert r["stopped"] is None and r["jobs"] == 2 and r["requests"] == 2
    # every download was priced first, with the identical request
    eps = [e for e, _ in calls if e != "symbology.resolve"]
    assert eps == ["metadata.get_cost", "timeseries.get_range"] * 2
    for i, (e, d) in enumerate(calls):
        if e == "timeseries.get_range":
            pe, pd_ = calls[i - 1]
            assert pe == "metadata.get_cost"
            assert {k: pd_[k] for k in ("dataset", "schema", "symbols", "start", "end", "stype_in")} == \
                {k: d[k] for k in ("dataset", "schema", "symbols", "start", "end", "stype_in")}
    lg = D.read_ledger()
    assert list(lg["status"]) == ["requested", "requested"] and D.spent_usd() == pytest.approx(0.02)
    assert set(lg["tag"]) == {"spx0dte"} and (lg["dataset"] == D.OPRA).all() and (lg["schema"] == "cbbo-1m").all()
    assert "test-key-not-real" not in D.ledger_path().read_text()
    assert D.day_path("SPX", "2024-01-02").exists() and D.day_path("SPX", "2024-01-03").exists()
    assert {"spx0dte|2024-01-02", "spx0dte|2024-01-03"} <= D.done_keys()
    # resumable: a re-run pays nothing and downloads nothing
    n = len(calls)
    out2 = fetch.task_databento_options()["spx0dte"]
    assert out2["skipped_done"] == 2 and out2["requests"] == 0
    assert [e for e, _ in calls[n:] if e in ("metadata.get_cost", "timeseries.get_range")] == []
    assert len(D.read_ledger()) == 2


def test_fetch_task_dry_run_prices_only(cache, monkeypatch):
    from atlasbt.data import fetch
    calls = []
    monkeypatch.setattr(D, "_post", _fake_client(calls))
    monkeypatch.setattr(D, "spx_reference", lambda portal, start, end: _ref(["2024-01-02"]))
    monkeypatch.setenv("ATLASBT_DATABENTO_PLANS", "spx0dte")
    monkeypatch.setenv("ATLASBT_DATABENTO_START", "2024-01-02")
    monkeypatch.setenv("ATLASBT_DATABENTO_END", "2024-01-02")
    monkeypatch.setenv("ATLASBT_DATABENTO_DRY_RUN", "1")
    r = fetch.task_databento_options()["spx0dte"]
    assert r["cost_usd"] == pytest.approx(0.01) and r["requests"] == 1
    assert "timeseries.get_range" not in [e for e, _ in calls]
    assert D.read_ledger().empty and not D.done_keys()


def test_fetch_task_budget_refusal_stops_and_is_logged(cache, monkeypatch):
    """DATABENTO_BUDGET_USD = 71.58 (the user's remaining cap after the $18.42 whose ledger was lost):
    the request that would cross it is refused (BudgetExceeded), logged 'refused', never downloaded,
    and the task stops cleanly; the job is NOT marked done, so a later run with more budget resumes it."""
    from atlasbt.data import fetch
    calls = []
    monkeypatch.setattr(D, "_post", _fake_client(calls, cost=0.05))
    monkeypatch.setattr(D, "spx_reference", lambda portal, start, end: _ref(["2024-01-02", "2024-01-03"]))
    monkeypatch.setattr(D, "store_spx_parity_bars", lambda raw, day: 0)
    monkeypatch.setenv("ATLASBT_DATABENTO_PLANS", "spx0dte")
    monkeypatch.setenv("ATLASBT_DATABENTO_START", "2024-01-02")
    monkeypatch.setenv("ATLASBT_DATABENTO_END", "2024-01-03")
    monkeypatch.setenv("DATABENTO_BUDGET_USD", "71.58")
    monkeypatch.delenv("ATLASBT_DATABENTO_DRY_RUN", raising=False)
    D._ledger_append({"timestamp": "2026-11-01T00:00:00Z", "dataset": D.OPRA, "schema": "cbbo-1m",
                      "cost_usd": 71.50, "status": "requested", "tag": "seed"})
    # through the task: first job fits (71.50 + 0.05 <= 71.58), second is refused
    r = fetch.task_databento_options()["spx0dte"]
    assert str(r["stopped"]).startswith("budget") and r["jobs"] == 2
    assert [e for e, _ in calls].count("timeseries.get_range") == 1
    # direct call: the guard raises
    with pytest.raises(D.BudgetExceeded):
        D.get_range_csv(D.OPRA, "cbbo-1m", ["SPXW  240102C04755000"], utc("2024-01-02 14:59"), utc("2024-01-02 15:01"))
    assert [e for e, _ in calls].count("timeseries.get_range") == 1
    lg = D.read_ledger()
    assert list(lg["status"]) == ["requested", "requested", "refused", "refused"]
    assert D.spent_usd() == pytest.approx(71.55)
    assert "spx0dte|2024-01-02" in D.done_keys() and "spx0dte|2024-01-03" not in D.done_keys()
    # more budget later -> resumes with the unfinished job only
    monkeypatch.setenv("DATABENTO_BUDGET_USD", "71.70")
    r2 = fetch.task_databento_options()["spx0dte"]
    assert r2["skipped_done"] == 1 and r2["requests"] == 1 and r2["stopped"] is None
    assert "spx0dte|2024-01-03" in D.done_keys()


def test_budget_default_is_the_remaining_cap(monkeypatch):
    monkeypatch.delenv("DATABENTO_BUDGET_USD", raising=False)
    assert D.budget_usd() == pytest.approx(71.58)
    monkeypatch.setenv("DATABENTO_BUDGET_USD", "12.5")
    assert D.budget_usd() == pytest.approx(12.5)


# ---------------------------------------------------------------- spx0dte strike band: no lookahead
class _DailyPortal:
    """bars(sym, '1d', ...) for SPY / SPX / VIX from {sym: DataFrame[close, high, low]} by ET session date."""

    def __init__(self, tabs):
        self.tabs = tabs

    def bars(self, sym, freq, start, end, as_of=None, adjusted=False):
        t = self.tabs[sym]
        idx = (pd.DatetimeIndex(t.index) + pd.Timedelta(hours=16)).tz_localize("America/New_York").tz_convert("UTC")
        return pd.DataFrame(t.to_numpy(), columns=t.columns, index=idx)


def test_spx0dte_strikes_use_only_pre_open_information():
    """Perturb everything about session d that is unknown at its open (SPY/SPX high, low, close of d, VIX
    close of d): the strike set of d's 0DTE fetch must not change. Perturbing the previous close must."""
    days = pd.DatetimeIndex(["2024-01-08", "2024-01-09", "2024-01-10"])
    d = days[-1]

    def tabs(spy_d=(475.0, 476.0, 474.0), spx_d=4750.0, vix_d=13.0, spx_prev=4750.0):
        spy = pd.DataFrame({"close": [474.0, 475.0, spy_d[0]], "high": [475.0, 476.0, spy_d[1]],
                            "low": [473.0, 474.0, spy_d[2]]}, index=days)
        spx = pd.DataFrame({"close": [4740.0, spx_prev, spx_d], "high": np.nan, "low": np.nan}, index=days)
        vix = pd.DataFrame({"close": [12.0, 13.0, vix_d], "high": np.nan, "low": np.nan}, index=days)
        return {"SPY": spy, "SPX": spx, "VIX": vix}

    lst = D.parse_osi([f"SPXW  240110{r_}0{K}000" for K in range(4000, 5505, 5) for r_ in "CP"])

    def strikes(t):
        jobs = D.plan_spx_0dte(d, d, D.spx_reference(_DailyPortal(t), days[0], d))
        assert len(jobs) == 1
        return set(D.parse_osi(jobs[0].select(lst))["strike"])

    base = strikes(tabs())
    assert strikes(tabs(spy_d=(440.0, 480.0, 430.0), spx_d=4400.0, vix_d=40.0)) == base
    lo, hi = D.spx0dte_band(4750.0, 13.0)          # VIX_prev 13 -> sigma 0.82 %: the 5 % floor binds
    assert (lo, hi) == pytest.approx((4750.0 * 0.95, 4750.0 * 1.05))
    assert min(base) == 4515.0 and max(base) == 4985.0
    assert strikes(tabs(spx_prev=4600.0)) != base   # the previous close does move the band


def test_run_jobs_stops_on_a_failed_billable_transport(cache, monkeypatch):
    """A transport failure of timeseries.get_range is not retried and stops run_jobs: the next job is not
    requested, the failed job is not marked done, and its 'requested' row stays counted."""
    calls = []
    inner = _fake_client(calls)

    def flaky(endpoint, data, timeout=600):
        if endpoint == "timeseries.get_range":
            calls.append((endpoint, dict(data)))
            raise D.TransportFailed("read timed out")
        return inner(endpoint, data, timeout)

    monkeypatch.setattr(D, "_post", flaky)
    monkeypatch.setenv("DATABENTO_BUDGET_USD", "71.58")
    jobs = D.plan_spx_0dte("2024-01-02", "2024-01-03", _ref(["2024-01-02", "2024-01-03"]))
    r = D.run_jobs(jobs, log=lambda *a: None)
    assert str(r["stopped"]).startswith("transport")
    assert [e for e, _ in calls].count("timeseries.get_range") == 1
    lg = D.read_ledger()
    assert list(lg["status"]) == ["requested", "failed:TransportFailed"]
    assert D.spent_usd() == pytest.approx(0.01) and not D.done_keys()


def test_post_never_retries_billable_endpoints(cache, monkeypatch):
    import requests as rq
    sent = []

    def boom(url, **kw):
        sent.append(url.rsplit("/", 1)[-1])
        raise rq.exceptions.ConnectionError("reset")

    monkeypatch.setattr(D.requests, "post", boom)
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    with pytest.raises(D.TransportFailed):
        D._post("timeseries.get_range", {})
    with pytest.raises(RuntimeError, match="retries exhausted"):
        D._post("metadata.get_cost", {})          # free endpoint: backs off and retries
    assert sent == ["timeseries.get_range"] + ["metadata.get_cost"] * 5
