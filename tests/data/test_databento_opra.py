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
