"""IBorrowDesk keyed v2 provider (providers/iborrowdesk.py) and Portal.borrow on real borrow rows.

No network: the payload below is synthetic_ but follows the documented v2 response shape
(https://www.iborrowdesk.com/api/v2/agent-skill/REFERENCE.md, read 2026-10-30)."""
import json

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.types import Fidelity
from atlasbt.data.http import FetchError
from atlasbt.data.portal import Portal
from atlasbt.data.providers import borrow as B
from atlasbt.data.providers import iborrowdesk as I


def synthetic_payload(sym="AAPL", name="APPLE INC", dates=("2024-03-04", "2024-03-05", "2024-03-06"),
                      fees=(0.25, 0.30, 5.0), avail=(5_000_000, 0, 100_000)):
    obs = [{"date": d, "fee": {"open": f, "high": f, "low": f, "close": f},
            "rebate": {"open": 5.0, "high": 5.0, "low": 5.0, "close": 5.0 - f},
            "available": {"open": a, "high": a, "low": a, "close": a}} for d, f, a in zip(dates, fees, avail)]
    return {"data": {sym: {"figi": "BBG000B9XRY4", "name": name, "country": "usa", "currency": "USD",
                           "previous_identities": [], "units": 1, "first_date": dates[0], "last_date": dates[-1],
                           "observations": obs}},
            "meta": {"start": None, "end": dates[-1], "unresolved": ["NOTATICKER"], "units_billed": 1}}


def test_parse_daily_is_pit():
    df = I.parse_daily(synthetic_payload())["AAPL"]
    assert list(df["fee"]) == [0.25, 0.30, 5.0]
    # the daily close is IB's last reading inside the session -> known at 16:00 ET of that date
    assert df["time"].iloc[0] == pd.Timestamp("2024-03-04 16:00", tz="America/New_York")
    assert df["figi"].iloc[0] == "BBG000B9XRY4" and df["source"].iloc[0] == "iborrowdesk"


def test_non_us_lines_dropped():
    p = synthetic_payload()
    p["data"]["AAPL"]["country"] = "germany"
    assert I.parse_daily(p) == {}


def test_units_and_batches():
    # all-time daily = 12 units / symbol (2015-07-13 .. 2026) -> at most 12 symbols per call (<=150 units)
    assert I.units_worst_case(1, "2015-07-13", "2026-10-29") == 12
    b = I.batches([f"S{i}" for i in range(30)], "2015-07-13", "2026-10-29")
    assert [len(x) for x in b] == [12, 12, 6]
    assert max(len(x) for x in I.batches([f"S{i}" for i in range(250)], "2026-01-01", "2026-10-29")) == 100


def test_no_key_is_skipped_without_network(monkeypatch, store):
    monkeypatch.delenv(I.ENV_KEY, raising=False)
    monkeypatch.setattr(I, "client", lambda: (_ for _ in ()).throw(AssertionError("network")))
    out = I.fetch_daily(["AAPL", "SPY"], store=store)
    assert "NEEDS USER" in out["skipped"] and out["symbols"] == 2
    assert not store.symbols("borrow", "iborrowdesk")


class _Client:
    def __init__(self, bodies):
        self.bodies, self.calls = list(bodies), []

    def get(self, url, params=None, headers=None, **kw):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        b = self.bodies.pop(0)
        if isinstance(b, Exception):
            raise b
        return json.dumps(b).encode()


def test_fetch_stores_and_checks_identity(monkeypatch, store):
    monkeypatch.setenv(I.ENV_KEY, "ibd_test")
    monkeypatch.setattr(I, "MIN_INTERVAL", 0.0)
    pay = synthetic_payload()
    pay["data"].update(synthetic_payload("XYZ", name="SOMETHING ELSE CORP")["data"])
    fc = _Client([pay])
    monkeypatch.setattr(I, "client", lambda: fc)
    out = I.fetch_daily(["AAPL", "XYZ"], store=store, names={"AAPL": "Apple Inc.", "XYZ": "Xyz Widgets Inc"})
    assert out["stored"] == 1 and out["name_mismatch"] == ["XYZ"] and out["unresolved"] == ["NOTATICKER"]
    url, params, headers = fc.calls[0]
    assert url.endswith("/api/v2/daily/borrow") and params["symbols"] == "AAPL,XYZ"
    assert headers["Authorization"] == "Bearer ibd_test" and "ibd_test" not in json.dumps(params)
    assert store.symbols("borrow", "iborrowdesk") == ["AAPL"]


def test_fetch_stops_on_non_retryable(monkeypatch, store):
    monkeypatch.setenv(I.ENV_KEY, "ibd_test")
    monkeypatch.setattr(I, "MIN_INTERVAL", 0.0)
    fc = _Client([FetchError("u", 402, "units_exhausted"), synthetic_payload()])
    monkeypatch.setattr(I, "client", lambda: fc)
    out = I.fetch_daily([f"S{i}" for i in range(20)], start="2015-07-13", end="2026-10-29", store=store)
    assert "402" in out["error"] and len(fc.calls) == 1


def test_names_agree():
    assert I.names_agree("META PLATFORMS INC-CLASS A", "Meta Platforms, Inc.")
    assert not I.names_agree("GAMESTOP CORP-CLASS A", "Apple Inc.")
    assert I.names_agree(None, "Apple Inc.")


def _with_rows(store):
    df = I.parse_daily(synthetic_payload())["AAPL"]
    store.write("borrow", "iborrowdesk", "AAPL", df, key=["date"], fidelity=1)
    return Portal(store)


def test_portal_borrow_serves_real_rows_pit(store):
    p = _with_rows(store)
    t = lambda s: pd.Timestamp(s, tz="America/New_York").tz_convert("UTC")
    assert p.has("AAPL", "borrow") and p.fidelity_of("AAPL", "borrow") == Fidelity.T1_BAR
    assert p.borrow("AAPL", t("2024-03-04 16:00")) == (True, 0.0025, Fidelity.T1_BAR)
    # before 16:00 on 03-05 only the 03-04 close is known
    assert p.borrow("AAPL", t("2024-03-05 15:59")) == (True, 0.0025, Fidelity.T1_BAR)
    ok, fee, fid = p.borrow("AAPL", t("2024-03-05 16:00"))
    assert (ok, fid) == (False, Fidelity.T1_BAR) and fee == pytest.approx(0.003)     # 0 shares available
    assert p.borrow("AAPL", t("2024-03-08 12:00"))[1] == pytest.approx(0.05)


def test_portal_borrow_default_outside_real_rows(store):
    p = _with_rows(store)
    ref = Portal(store)
    before = pd.Timestamp("2024-03-04 15:59", tz="America/New_York")
    stale = pd.Timestamp("2024-03-06 16:00", tz="America/New_York") + pd.Timedelta(days=Portal.MAX_BORROW_STALE_DAYS, seconds=1)
    for a in (before, stale):
        r = p.borrow("AAPL", a)
        assert r[2] == Fidelity.T2_PROXY
        assert r == B.default_fee("AAPL", is_etf=False, is_index_member=False) or r[1] in (B.ETB, B.GC)
    # symbols without rows: unchanged default path (bit-for-bit what the default model returns)
    assert p.borrow("TSLA", before) == ref.borrow("TSLA", before)
    assert p.fidelity_of("TSLA", "borrow") == Fidelity.T2_PROXY and not p.has("TSLA", "borrow")


def test_portal_borrow_memo_matches_frame_scan(store):
    p = _with_rows(store)
    df = store.read("borrow", "iborrowdesk", "AAPL")
    for a in pd.date_range("2024-03-04 20:00", "2024-03-07 23:00", freq="7h", tz="UTC"):
        known = df[pd.to_datetime(df["time"], utc=True) <= a]
        if len(known):
            exp = float(known["fee"].iloc[-1]) / 100.0
            assert p.borrow("AAPL", a)[1] == exp and p.borrow("AAPL", a)[0] == bool(known["available"].iloc[-1] > 0)
    assert np.isfinite(p.borrow("AAPL", pd.Timestamp("2024-03-07", tz="UTC"))[1])
