"""Red team (gap closing): Databento OPRA as T0 (339d0f0) -- tiering, fetch selection, budget ledger.

Offline: no network (requests are monkeypatched). Each test asserts the correct behaviour; a failure is a
finding.
"""
from __future__ import annotations

import threading

import pandas as pd
import pytest
import requests

from atlasbt.core.types import Fidelity
from atlasbt.data import config
from atlasbt.data.providers import databento as D
from atlasbt.data.store import ParquetStore
from atlasbt.engine.strategy import Requirement
from atlasbt.run.plan import PlanContext, _plan_instance

from tests.run.test_plan_intraday import ALL, _IInst, _IPortal
from tests.run.test_plan_star import _row


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
    monkeypatch.setattr(store_mod, "_DEFAULT", None)
    D._LISTING_MEMO.clear()
    return tmp_path


def cbbo_csv(rows):
    """rows: (ts_recv, ts_event, raw_symbol, bid, ask)."""
    head = "ts_recv,ts_event,rtype,publisher_id,instrument_id,side,price,size,flags,bid_px_00,ask_px_00,bid_sz_00," \
           "ask_sz_00,bid_pb_00,ask_pb_00,symbol\n"
    body = "".join(f"{a},{b},193,30,1,N,1.0,1,200,{bid},{ask},10,12,0,0,{sym}\n" for a, b, sym, bid, ask in rows)
    return (head + body).encode()


def _write(root, rows, chain=True):
    D.write_rows(D.to_store_rows(D.parse_cbbo_csv(cbbo_csv(rows)), chain=chain), root)


JAN = pd.bdate_range("2024-01-02", "2024-01-31").difference(pd.DatetimeIndex(["2024-01-15"]))   # NYSE sessions


def _spxw_0dte_month(root):
    """SPXW 0DTE samples at 10:00 ET on every session of January 2024 (what the spx0dte plan stores)."""
    rows = []
    for d in JAN:
        ymd = f"{d:%y%m%d}"
        for K in (4740, 4750, 4760):
            for r_ in "CP":
                rows.append((f"{d:%Y-%m-%d}T15:00:00Z", f"{d:%Y-%m-%d}T14:59:30Z", f"SPXW  {ymd}{r_}0{K}000", 2.0, 2.2))
    _write(root, rows)


class _OPortal(_IPortal):
    """Planner portal: synthetic bars (_IPortal) + the REAL Portal's options answers on the same store."""

    def __init__(self, root, **kw):
        super().__init__(root, **kw)
        from atlasbt.data.portal import Portal
        self.real = Portal(ParquetStore(root / "store"))

    def has(self, sym, kind, freq=None, sessions=None, **kw):
        if kind == "options":
            return self.real.has(sym, kind, freq)
        return super().has(sym, kind, freq, sessions, **kw)

    def fidelity_of(self, sym, kind, freq=None, sessions=None):
        if kind == "options":
            return self.real.fidelity_of(sym, kind, freq)
        return super().fidelity_of(sym, kind, freq, sessions)

    def opra_coverage(self, underlying="*", detail=False):
        return self.real.opra_coverage(underlying, detail)


# ------------------------------------------------------------------------------------------------ (d)
def test_rt_spxw_t0_claimed_when_spx_files_hold_no_spxw_rows(cache):
    """_opra_target('SPXW') only checks that SPX files exist, so has()/fidelity_of('SPXW') say T0 even when
    those files hold only SPX AM series; option_chain('SPXW') then serves the synthetic fallback (T3) and
    coverage('SPXW') reports zero days."""
    from atlasbt.data.portal import Portal
    root = cache / "store"
    rows = [("2024-01-10T15:00:00Z", "2024-01-10T14:59:00Z", f"SPX   240216{r_}0{K}000", 50.0, 51.0)
            for K in (4700, 4750, 4800) for r_ in "CP"]
    _write(root, rows)
    p = Portal(ParquetStore(root))
    assert D.coverage("SPXW", root=root)["n_days"] == 0
    ch = p.option_chain("SPXW", utc("2024-01-10 15:00:30"))
    assert ch.attrs.get("source") != "databento"                 # not served from OPRA
    assert not p.has("SPXW", "options", "1m"), (
        f"has('SPXW','options','1m') is True and fidelity_of is {Fidelity(int(p.fidelity_of('SPXW', 'options', '1m'))).name}, "
        f"but no SPXW row is stored and option_chain('SPXW') is {ch.attrs.get('source') or 'synthetic'}")


def test_rt_planner_t0_options_window_not_bounded_by_opra_coverage(cache):
    """A critical Requirement('options', '1m', symbols=('SPX',), min_fidelity=T1_BAR) is satisfied by OPRA
    files covering January 2024 only, yet the planner's window is 2016-01-04..today at tier T1 (bars):
    window-free has()/fidelity_of() decide, and the coverage API the commit built for planners
    (Portal.opra_coverage) is never consulted. Outside January 2024 the run has no T0 chain."""
    root = cache / "store"
    _spxw_0dte_month(root)
    port = _OPortal(cache, spans={"SPY": ALL})
    cov = port.opra_coverage("SPX")
    assert (cov["first"], cov["last"]) == ("2024-01-02", "2024-01-31")
    reqs = [Requirement("bars", "1m", symbols=("SPY",)),
            Requirement("options", "1m", symbols=("SPX",), min_fidelity=Fidelity.T1_BAR)]
    r = _plan_instance("T", _IInst(reqs, universe=["SPY"]), "1m", PlanContext(port, cache / "res"), _row())
    assert r["testable"] == "Y", r["reason"]
    assert cov["first"] <= r["start"] and r["end"] <= cov["last"], (
        f"planner window {r['start']}..{r['end']} at {r['expected_fidelity']}; OPRA (the only T0/T1 option "
        f"source) covers {cov['first']}..{cov['last']}")


def test_rt_planner_labels_opra_backed_spx_options_t3_synthetic(cache):
    """Option strategy on SPX (S161-S164 shape: asset OPT, universe [SPX]): the planner's asset block forces
    T3_MODELED 'synthetic chain only' for SPX/SPXW whatever the portal serves, although every session of the
    window has OPRA chains the Portal serves as T0_QUOTE."""
    root = cache / "store"
    _spxw_0dte_month(root)
    port = _OPortal(cache, spans={"SPX": [("2024-01-02", "2024-01-31")]})
    ch = port.real.option_chain("SPX", utc("2024-01-10 15:00:30"))
    assert ch.attrs["source"] == "databento" and ch.attrs["fidelity"] == int(Fidelity.T0_QUOTE)
    inst = _IInst([Requirement("bars", "1m", symbols=("SPX",)),
                   Requirement("options", "1m", symbols=("SPX",), min_fidelity=Fidelity.T1_BAR)], universe=["SPX"])
    inst.asset = "OPT"
    r = _plan_instance("T", inst, "1m", PlanContext(port, cache / "res"), _row())
    assert r["testable"] == "Y", r["reason"]
    assert (r["start"], r["end"]) == ("2024-01-02", "2024-01-31")
    assert r["expected_fidelity"] != "T3_MODELED", (r["expected_fidelity"], r["fidelity_notes"])


# ------------------------------------------------------------------------------------------------ (a)
def test_rt_spx0dte_strike_set_depends_on_the_days_realized_high_low(cache):
    """plan_spx_0dte keeps only strikes inside [day LOW x (1-m), day HIGH x (1+m)] of THAT session
    (spx_reference: SPY high/low of day d). The stored -- and served as T0 from the first minute -- 0DTE
    chain therefore encodes the day's future range: two days identical at the open (same prior close, same
    VIX_prev) get different strike sets because their afternoons differ."""
    d = pd.Timestamp("2024-01-10")
    lst = D.parse_osi([f"SPXW  240110{r_}0{K}000" for K in range(4400, 5105, 5) for r_ in "CP"])

    def ref(hi):
        return pd.DataFrame({"ratio": 10.0, "hi": hi, "lo": 4700.0, "close": 4750.0, "vix_prev": 13.0},
                            index=pd.DatetimeIndex([d]))

    calm, rally = (D.plan_spx_0dte(d, d, ref(h)) for h in (4760.0, 4860.0))
    assert len(calm) == len(rally) == 1
    k_calm = set(D.parse_osi(calm[0].select(lst))["strike"])
    k_rally = set(D.parse_osi(rally[0].select(lst))["strike"])
    assert k_calm == k_rally, (
        f"strikes stored for 09:30 depend on the session's later high: calm max {max(k_calm)}, rally max "
        f"{max(k_rally)} ({len(k_rally - k_calm)} extra strikes on the rally day)")


# ------------------------------------------------------------------------------------------------ (e)
class _R:
    def __init__(self, text="", content=b""):
        self.text, self.content, self.status_code, self.headers = text, content, 200, {}


def test_rt_get_range_retry_after_transport_error_is_not_ledgered(cache, monkeypatch):
    """_post retries timeseries.get_range up to 5 times on a transport error (e.g. a read timeout after the
    server started streaming). get_range_csv writes ONE 'requested' ledger row, so every retried download
    is invisible to spent_usd() and the cap."""
    body = cbbo_csv([("2024-01-02T14:31:00Z", "2024-01-02T14:30:59Z", "SPXW  240102C04700000", 1, 2)])
    sent = []

    def fake_post(url, data=None, auth=None, timeout=None, headers=None):
        ep = url.rsplit("/", 1)[-1]
        if ep == "metadata.get_cost":
            return _R(text="0.5")
        sent.append(ep)
        if len(sent) == 1:
            raise requests.exceptions.ReadTimeout("read timed out")
        return _R(content=body)

    monkeypatch.setattr(D.requests, "post", fake_post)
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    monkeypatch.setenv("DATABENTO_BUDGET_USD", "1.0")
    D.get_range_csv(D.OPRA, "cbbo-1m", ["SPXW  240102C04700000"], utc("2024-01-02 14:30"), utc("2024-01-02 14:32"))
    lg = D.read_ledger()
    n_req = int((lg["status"] == "requested").sum())
    assert len(sent) == n_req, f"{len(sent)} get_range downloads sent, {n_req} priced in the ledger"


def test_rt_budget_check_and_ledger_append_are_not_atomic(cache, monkeypatch):
    """run_jobs expects concurrent fetch processes ('another process may have done it'), but get_range_csv
    reads spent_usd(), compares with the cap and appends its row without a lock. Two requests of $0.60
    against a $1.00 cap that read the ledger before either appends both pass: $1.20 spent."""
    body = cbbo_csv([("2024-01-02T14:31:00Z", "2024-01-02T14:30:59Z", "SPXW  240102C04700000", 1, 2)])

    def fake_post(endpoint, data, timeout=600):
        return _R(text="0.6") if endpoint == "metadata.get_cost" else _R(content=body)

    monkeypatch.setattr(D, "_post", fake_post)
    monkeypatch.setenv("DATABENTO_BUDGET_USD", "1.0")
    orig = D.spent_usd
    gate = threading.Barrier(2, timeout=5)

    def interleaved(ledger=None):              # both processes read the ledger before either appends
        v = orig(ledger)
        try:
            gate.wait()
        except threading.BrokenBarrierError:
            pass
        return v

    monkeypatch.setattr(D, "spent_usd", interleaved)
    errs = []

    def one(k):
        try:
            D.get_range_csv(D.OPRA, "cbbo-1m", [f"SPXW  240102C0{k}000"], utc("2024-01-02 14:30"),
                            utc("2024-01-02 14:32"))
        except D.BudgetExceeded as e:
            errs.append(e)

    th = [threading.Thread(target=one, args=(k,)) for k in (4700, 4705)]
    [t.start() for t in th]
    [t.join() for t in th]
    spent = orig()
    assert spent <= D.budget_usd() + 1e-9, f"spent ${spent:.2f} > cap ${D.budget_usd():.2f} ({len(errs)} refused)"


def test_planner_eod_options_accepting_t3_are_not_cut_to_opra(cache):
    """(verifier) counterpart of the OPRA bound: an EOD options requirement that accepts T3 is served by the
    synthetic fallback outside the OPRA sessions, so the window is not cut, and the run is tiered T3."""
    _spxw_0dte_month(cache / "store")
    port = _OPortal(cache, spans={"SPY": ALL})
    reqs = [Requirement("bars", "1m", symbols=("SPY",)), Requirement("options", None, symbols=("SPX",))]
    r = _plan_instance("T", _IInst(reqs, universe=["SPY"]), "1m", PlanContext(port, cache / "res"), _row())
    assert r["testable"] == "Y", r["reason"]
    assert r["start"] < "2024-01-02" and r["expected_fidelity"] == "T3_MODELED", r
