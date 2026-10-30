"""DIVIDEND_DECL events (providers/dividends_decl.py) and EDGAR full-text-search parsing (providers/sec.py).

Fixtures: efts_aapl_dividend_2019_2020.json = the real EFTS response for AAPL (CIK 320193) to
dividends_decl.DECL_QUERY, trimmed to 2019-2020 hits and to the fields we read (recorded 2026-10-30);
aapl_acceptance_2019_2020.csv = their acceptanceDateTime from data.sec.gov submissions. AAPL ex-dates /
amounts below are Apple's published 2019-2020 dividends; the declaration dates they must map to are
Apple's earnings-release dates (dividend declared in the same press release)."""
import json

import pandas as pd
import pytest

from atlasbt.core.types import Fidelity
from atlasbt.data.portal import Portal
from atlasbt.data.providers import dividends_decl as D
from atlasbt.data.providers import sec

from .conftest import FIX


class _FakeClient:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def get(self, url, params=None, **kw):
        self.calls.append(dict(params or {}))
        frm = int((params or {}).get("from", 0))
        return json.dumps(self.pages[frm]).encode()


def _aapl_hits(monkeypatch):
    page = json.loads((FIX / "efts_aapl_dividend_2019_2020.json").read_text())
    monkeypatch.setattr(sec, "client", lambda: _FakeClient({0: page}))
    acc = pd.read_csv(FIX / "aapl_acceptance_2019_2020.csv")
    monkeypatch.setattr(sec, "acceptance_map", lambda cik: dict(zip(acc["adsh"], pd.to_datetime(acc["acceptance"], utc=True))))
    return D.issuer_hits(320193, end="2020-12-31")


def test_fts_search_parses_documents_and_pages(monkeypatch):
    def hit(i):
        return {"_id": f"0000000001-24-00000{i}:ex99.htm", "_source": {
            "ciks": ["0000000001"], "file_date": "2024-01-0%d" % (i + 1), "form": "8-K", "adsh": f"0000000001-24-00000{i}",
            "root_forms": ["8-K"], "file_type": "EX-99.1", "file_description": "PRESS RELEASE", "items": ["8.01", "9.01"],
            "display_names": ["SYN CORP (SYN) (CIK 0000000001)"]}}
    pages = {0: {"hits": {"total": {"value": 3}, "hits": [hit(0), hit(1)]}},
             2: {"hits": {"total": {"value": 3}, "hits": [hit(2)]}}}
    fc = _FakeClient(pages)
    monkeypatch.setattr(sec, "client", lambda: fc)
    df = sec.fts_search('"declared a quarterly"', ciks=[1], end="2024-02-01")
    assert len(df) == 3 and [c.get("from", 0) for c in fc.calls] == [0, 2]
    assert fc.calls[0]["ciks"] == "0000000001" and fc.calls[0]["forms"] == "8-K"
    r = df.iloc[0]
    assert r["adsh"] == "0000000001-24-000000" and r["filename"] == "ex99.htm" and r["cik"] == 1
    assert r["items"] == "8.01,9.01" and r["file_type"] == "EX-99.1"


def test_filing_known_at_fallback_is_never_early():
    k, m = sec.filing_known_at("x", pd.Timestamp("2024-03-01"), {})
    assert m == "filed_1730ET" and k == pd.Timestamp("2024-03-01 17:30", tz="America/New_York")
    a = pd.Timestamp("2024-03-01 21:05:00", tz="UTC")
    assert sec.filing_known_at("x", pd.Timestamp("2024-03-01"), {"x": a}) == (a, "acceptance")


def test_fetch_document_needs_contact_string(monkeypatch):
    from atlasbt.data import config
    monkeypatch.setattr(config, "SEC_USER_AGENT", None)
    with pytest.raises(sec.SecContactRequired):
        sec.fetch_document(320193, "0000320193-19-000007", "a.htm")


def test_aapl_declarations_match_earnings_release_dates(monkeypatch):
    hits = _aapl_hits(monkeypatch)
    assert len(hits) == 8 and (hits["known_at_method"] == "acceptance").all()
    divs = pd.DataFrame({"ex_date": pd.to_datetime(["2019-02-08", "2019-05-10", "2019-08-09", "2019-11-07",
                                                    "2020-02-07", "2020-05-08", "2020-08-07", "2020-11-06"]),
                         "amount": [0.73, 0.77, 0.77, 0.77, 0.77, 0.82, 0.82, 0.205]})
    ev = D.match_declarations(divs, hits, "AAPL", 320193)
    assert len(ev) == 8
    decl = ev["known_at"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d").tolist()
    assert decl == ["2019-01-29", "2019-04-30", "2019-07-30", "2019-10-30",
                    "2020-01-28", "2020-04-30", "2020-07-30", "2020-10-29"]
    # 8-K accepted 16:30 ET, after the close: known only from then on
    assert (ev["known_at"].dt.tz_convert("America/New_York").dt.hour >= 16).all()
    assert (ev["fidelity"] == int(Fidelity.T2_PROXY)).all() and (ev["cik"] == 320193).all()
    assert ev["lead_days"].between(5, 20).all()


def _h(*ts):
    return pd.DataFrame({"adsh": [f"a{i}" for i in range(len(ts))], "known_at": pd.to_datetime(list(ts), utc=True),
                         "known_at_method": "acceptance", "items": "8.01", "file_type": "EX-99.1", "filename": "x.htm"})


def test_match_latest_in_window_and_not_before_previous_ex():
    divs = pd.DataFrame({"ex_date": pd.to_datetime(["2024-03-15", "2024-06-14", "2024-09-13"]), "amount": [1.0, 1.0, 1.1]})
    hits = _h("2024-02-20 21:00", "2024-03-01 21:00",      # two hits before ex #1: the LATER one is used
              "2024-03-14 21:00",                          # after ex #1's decision but before ex #1 open -> ex #1
              "2024-05-01 12:00")                          # ex #2
    ev = D.match_declarations(divs, hits, "SYN", 1).set_index("ex_date")
    assert ev.loc[pd.Timestamp("2024-03-15"), "accn"] == "a2"
    assert ev.loc[pd.Timestamp("2024-06-14"), "accn"] == "a3"
    # ex #3: the only hit inside (ex - 75d) is a3 (2024-05-01), but it predates ex #2 -> unmatched
    assert pd.Timestamp("2024-09-13") not in ev.index
    # a hit at/after the ex-date open never counts for that ex-date
    ev2 = D.match_declarations(divs.iloc[:1], _h("2024-03-15 13:30"), "SYN", 1)
    assert ev2.empty


def _store_events(store, rows):
    df = pd.DataFrame(rows)
    store.write("events", "sec_fts", "DIVIDEND_DECL", df, key=["symbol", "ex_date"], fidelity=2)


def test_portal_serves_stored_declarations_pit(store):
    p0 = Portal(store)
    assert not p0.has("*", "events", "DIVIDEND_DECL")
    _store_events(store, [
        {"kind": "DIVIDEND_DECL", "symbol": "AAPL", "cik": 320193, "known_at": pd.Timestamp("2024-02-01 21:30", tz="UTC"),
         "effective_at": pd.Timestamp("2024-02-09 14:30", tz="UTC"), "ex_date": pd.Timestamp("2024-02-09"),
         "amount": 0.24, "accn": "a", "fidelity": 2, "source": "sec_fts"},
        {"kind": "DIVIDEND_DECL", "symbol": "SYN", "cik": 900002, "known_at": pd.Timestamp("2024-02-05 21:30", tz="UTC"),
         "effective_at": pd.Timestamp("2024-02-20 14:30", tz="UTC"), "ex_date": pd.Timestamp("2024-02-20"),
         "amount": 0.5, "accn": "b", "fidelity": 2, "source": "sec_fts"}])
    p = Portal(store)
    assert p.has("*", "events", "DIVIDEND_DECL") and p.has("AAPL", "events", "DIVIDEND_DECL")
    assert not p.has("TSLA", "events", "DIVIDEND_DECL")
    assert p.fidelity_of("*", "events", "DIVIDEND_DECL") == Fidelity.T2_PROXY
    lo, hi = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC")
    ev = p.events("DIVIDEND_DECL", ["AAPL", "SYN"], lo, hi, pd.Timestamp("2024-02-02", tz="UTC"))
    assert [e.symbol for e in ev] == ["AAPL"]                      # SYN declared later: not yet known
    e = ev[0]
    assert pd.Timestamp(e.payload["ex_date"]) == pd.Timestamp("2024-02-09") and e.payload["amount"] == 0.24
    assert e.fidelity == Fidelity.T2_PROXY
    # security key: only the issuer named by the key's CIK
    assert len(p.events("DIVIDEND_DECL", ["AAPL~CIK0000320193"], lo, hi, hi)) == 1
    assert p.events("DIVIDEND_DECL", ["AAPL~CIK0000999999"], lo, hi, hi) == []
