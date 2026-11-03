"""BUYBACK pipeline (providers/buybacks.py): authorization parsing, event assembly, NEEDS-USER path.
Sentences are synthetic_ paraphrases of typical press-release wording (no network)."""
import pandas as pd
import pytest

from atlasbt.core.types import Fidelity
from atlasbt.data.portal import Portal
from atlasbt.data.providers import buybacks as BB
from atlasbt.data.providers import sec


@pytest.mark.parametrize("text, amount, shares", [
    ("The Board of Directors has authorized the repurchase of up to $10 billion of the Company's common stock.", 10e9, None),
    ("The company announced a new $100 billion share repurchase authorization and raised its dividend.", 100e9, None),
    ("The board increased its share repurchase authorization by $2.5 billion.", 2.5e9, None),
    ("Directors approved an additional $750 million stock buyback program.", 750e6, None),
    ("The Board approved a program to repurchase up to 20 million shares of common stock.", None, 20e6),
])
def test_parse_authorization(text, amount, shares):
    a = BB.parse_authorization(text)
    assert a["amount_usd"] == (pytest.approx(amount) if amount else None)
    assert a["shares"] == (pytest.approx(shares) if shares else None)


@pytest.mark.parametrize("text", [
    "During the quarter we repurchased $2.1 billion of shares under our existing program.",
    "The Board declared a quarterly dividend of $0.25 per share.",
    "Revenue was $5 billion, up 10%.",
])
def test_parse_authorization_ignores_non_authorizations(text):
    assert BB.parse_authorization(text) == {}


def test_html_to_text():
    t = BB.html_to_text(b"<html><style>x{}</style><p>Authorized&nbsp;the repurchase of up to <b>$1.5</b> billion.</p></html>")
    assert BB.parse_authorization(t)["amount_usd"] == pytest.approx(1.5e9)


class _P:
    def fundamentals(self, syms, fields, as_of):
        return pd.DataFrame({"symbol": syms, "field": fields, "value": [1e9]})

    def bars(self, sym, freq, start, end, as_of):
        return pd.DataFrame({"close": [50.0]})


def _cands(items="8.01,9.01"):
    return pd.DataFrame([{"symbol": "SYN", "cik": 900002, "adsh": "0000900002-24-000001", "file_date": pd.Timestamp("2024-03-04"),
                          "form": "8-K", "items": items, "file_type": "EX-99.1", "filename": "ex991.htm",
                          "known_at": pd.Timestamp("2024-03-04 13:05", tz="UTC"), "known_at_method": "acceptance",
                          "with_earnings": "2.02" in items}])


def test_build_events_pct_mcap_and_earnings_flag():
    doc = b"<p>The Board authorized the repurchase of up to $3 billion of common stock.</p>"
    ev = BB.build_events(_cands(), _P(), fetch=lambda c, a, f: doc)
    assert len(ev) == 1
    r = ev.iloc[0]
    assert r["pct_mcap"] == pytest.approx(3e9 / 50e9) and not r["with_earnings"]
    assert r["fidelity"] == int(Fidelity.T2_PROXY) and r["known_at"] == pd.Timestamp("2024-03-04 13:05", tz="UTC")
    ev2 = BB.build_events(_cands("2.02,9.01"), _P(), fetch=lambda c, a, f: doc, skip_earnings=False)
    assert bool(ev2.iloc[0]["with_earnings"])
    # default: earnings-bundled 8-Ks are not read (S045 excludes them) -> no request, no event
    def no_fetch(*a):
        raise AssertionError("fetched an earnings-bundled document")
    assert BB.build_events(_cands("2.02,9.01"), _P(), fetch=no_fetch).empty


def test_without_contact_string_no_events(monkeypatch, store):
    from atlasbt.data import config
    monkeypatch.setattr(config, "SEC_USER_AGENT", None)
    sm = pd.DataFrame({"symbol": ["SYN"], "cik": [900002]})
    store.write("reference", "sec", "security_master", sm, key=["symbol"])
    monkeypatch.setattr(BB, "issuer_candidates", lambda s, c, start: _cands())
    out = BB.build(store=store, portal=_P())
    assert out["candidates"] == 1 and out["events"] == 0 and "SEC_USER_AGENT" in out["needs_user"]
    assert store.exists("reference", "sec_fts", "BUYBACK_CANDIDATES")
    p = Portal(store)
    assert not p.has("*", "events", "BUYBACK")            # candidates are never served as events


def test_fetch_document_requires_contact(monkeypatch):
    from atlasbt.data import config
    monkeypatch.setattr(config, "SEC_USER_AGENT", None)
    with pytest.raises(sec.SecContactRequired):
        BB.build_events(_cands(), _P())


@pytest.mark.parametrize("text", [
    "In June 2007, the Board of Directors authorized the repurchase of up to $750 million of its common stock. x",
    "In January, 2014, the company announced its Board had approved an increase to up to 28 million shares. x",
    "The existing program had previously authorized the repurchase of up to 10,944,125 shares of common stock. x",
    "In addition to the $300 million remaining, the Board authorized the repurchase of up to $1 billion of stock. x",
    "Credit Quality " * 30 + "authorized the repurchase of up to $3 billion of stock. x",
])
def test_parse_rejects_stale_past_and_slide_text(text):
    assert BB.parse_authorization(text, filed=pd.Timestamp("2017-11-03", tz="UTC")) == {}


def test_parse_keeps_fresh_dated_authorization():
    t = "On November 2, 2017, the Board authorized the repurchase of up to $500 million of common stock through 2019. x"
    assert BB.parse_authorization(t, filed=pd.Timestamp("2017-11-03", tz="UTC"))["amount_usd"] == pytest.approx(5e8)


def test_identity_filter_drops_reused_ticker_issuer(store):
    from atlasbt.data.providers.dividends_decl import identity_filter
    p = Portal(store)                          # synthetic identity: AAPL -> 320193 for all dates
    ev = pd.DataFrame({"symbol": ["AAPL", "AAPL"], "cik": [320193, 999], "known_at":
                       [pd.Timestamp("2024-03-04 13:00", tz="UTC")] * 2})
    assert list(identity_filter(ev, p)["cik"]) == [320193]


def test_earnings_bundling_from_nearby_2_02(store):
    er = pd.DataFrame({"kind": "EARNINGS_RELEASE", "symbol": "SYN", "known_at": [pd.Timestamp("2024-03-04 21:00", tz="UTC")],
                       "accn": ["x"]})
    store.write("events", "sec", "EARNINGS_RELEASE_SYN", er, key=["accn"])
    ev = pd.DataFrame({"symbol": ["SYN", "SYN"], "cik": [900002, 900002], "with_earnings": [False, False],
                       "known_at": [pd.Timestamp("2024-03-04 21:05", tz="UTC"), pd.Timestamp("2024-03-08 13:00", tz="UTC")]})
    out = BB.flag_earnings_bundling(ev, store)
    assert list(out["with_earnings"]) == [True, False]
