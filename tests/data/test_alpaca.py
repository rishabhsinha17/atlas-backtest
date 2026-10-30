"""Alpaca SIP provider (atlasbt/data/providers/alpaca.py) and its Portal integration. No network, no secrets:
recorded real API pages under tests/data/fixtures/alpaca_*.json (recorded 2026-10-30, provenance in
docs/DATA.md §4.15) and synthetic_* frames/sessions that only exercise mechanics."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from atlasbt.data import frames
from atlasbt.data.portal import Portal
from atlasbt.data.providers import alpaca as A

from .conftest import FIX

ET = "America/New_York"


def utc(s):
    return pd.Timestamp(s, tz="UTC")


def et(s):
    return pd.Timestamp(s, tz=ET).tz_convert("UTC")


def rec(name):
    return json.loads((FIX / f"alpaca_{name}.json").read_text())


# ------------------------------------------------------------------------------------- parsing (real pages)
def test_bars_are_labelled_by_start_and_match_the_daily_bar():
    r = rec("bars_spy_20160104")
    m = A.parse_bars(r["bars_1m"]["SPY"])
    d = A.parse_bars(r["bars_1d"]["SPY"])
    c = A.canonical_bars(m, "1m")
    assert c.index.name == "start" and str(c.index.tz) == "UTC"
    assert not frames.validate_bars(c)
    first_rth = c[c["session"] == "RTH"].iloc[0]
    # the 14:30Z bar opens at the official open -> 't' is the bar START (not its end)
    assert c[c["session"] == "RTH"].index[0] == et("2016-01-04 09:30")
    assert first_rth["open"] == d["open"].iloc[0] == 200.49
    assert first_rth["known_at"] == et("2016-01-04 09:31")
    # the closing auction prints in the 16:00 (POST) minute, not in the last RTH bar
    post = c[c["session"] == "POST"]
    assert post.index[0] == et("2016-01-04 16:00") and post["volume"].iloc[0] > 5 * post["volume"].iloc[1]
    assert post["volume"].iloc[0] > c[c["session"] == "RTH"]["volume"].iloc[-1]
    assert {"PRE", "RTH", "POST"} <= set(c["session"])
    assert set(A.canonical_bars(m, "1m", ("PRE", "RTH"))["session"]) == {"PRE", "RTH"}


def test_quote_sizes_switch_from_round_lots_to_shares_on_2025_11_03():
    r = rec("quotes_spy_size_switch")
    old = A.parse_quotes(r["2025-10-31"])
    new = A.parse_quotes(r["2025-11-03"])
    raw_old = np.array([x["bs"] for x in r["2025-10-31"]], float)
    raw_new = np.array([x["bs"] for x in r["2025-11-03"]], float)
    assert (old["bid_size"].to_numpy() == raw_old * 100).all()       # round lots -> shares
    assert (new["bid_size"].to_numpy() == raw_new).all()             # already shares
    assert (raw_new % 40 == 0).all() and raw_old.min() < 40           # SPY's new 40-share round lot
    assert old.index.name == "ts" and str(old.index.tz) == "UTC"


def test_recorded_quotes_are_nbbo_not_exchange_bbo():
    q = A.parse_quotes(rec("quotes_spy_size_switch")["2025-10-31"])
    # a per-exchange BBO stream would carry the same exchange on both sides of every update
    assert (q["bid_exchange"] != q["ask_exchange"]).any()


def test_synthetic_clean_nbbo_rules():
    t = pd.date_range("2024-01-03 15:00", periods=8, freq="100ms", tz="UTC")
    q = pd.DataFrame({"bid": [10.0, 10.0, 10.02, 10.01, 10.0, 10.0, 0.0, 10.0],
                      "ask": [10.01, 10.01, 10.01, 10.01, 10.02, 10.02, 10.02, 10.02],
                      "bid_size": [100, 100, 100, 100, 100, 100, 100, 200],
                      "ask_size": [100, 100, 100, 100, 100, 100, 100, 200],
                      "conditions": ["R", "R", "R", "R", "N", "?", "R", "O,R"]}, index=t.rename("ts"))
    c, st = A.clean_nbbo(q)
    assert st == {"raw": 8, "one_sided": 1, "condition": 1, "crossed": 1, "locked": 1, "kept": 3}
    # row 1 duplicates row 0 (dropped), crossed row 2, locked row 3, non-firm 'N' row 4, one-sided row 6
    assert list(c.index) == [t[0], t[5], t[7]]
    assert list(c.columns) == ["bid", "ask", "bid_size", "ask_size"]


def test_synthetic_parse_quotes_legacy_payload():
    q = A.parse_quotes([{"t": "2024-01-03T14:30:00.5Z", "bp": 470.1, "bs": 3, "ap": 470.11, "as": 2,
                         "bx": "P", "ax": "Q", "c": ["R"], "z": "B"}])
    assert q["bid_size"].iloc[0] == 300 and q["conditions"].iloc[0] == "R"


# ------------------------------------------------------------------------------------- client mechanics
class _Resp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body or {}, headers or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


class _FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params), dict(headers)))
        return self.responses.pop(0)


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "synthetic-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "synthetic-secret")
    monkeypatch.setenv("ATLASBT_OFFLINE", "0")


def test_client_paginates_and_backs_off_on_429(keys, monkeypatch):
    sleeps = []
    monkeypatch.setattr(A.time, "sleep", lambda s: sleeps.append(s))
    now = [1_000_000.0]
    monkeypatch.setattr(A.time, "time", lambda: now[0])
    sess = _FakeSession([
        _Resp(429, {"message": "too many"}, {"X-RateLimit-Reset": str(int(now[0]) + 5)}),
        _Resp(200, {"bars": {"SPY": [{"t": "2024-01-03T14:30:00Z"}]}, "next_page_token": "abc"}),
        _Resp(200, {"bars": {"SPY": [{"t": "2024-01-03T14:31:00Z"}], "QQQ": [{"t": "2024-01-03T14:31:00Z"}]},
                    "next_page_token": None})])
    cli = A.AlpacaClient(rpm=180, session=sess)

    def fake_sleep(s):
        sleeps.append(s)
        now[0] += s
    monkeypatch.setattr(A.time, "sleep", fake_sleep)
    pages = list(cli.pages("bars", ["SPY", "QQQ"], {"timeframe": "1Min"}))
    assert [sorted(p) for p in pages] == [["SPY"], ["QQQ", "SPY"]]
    assert sum(sleeps) >= 5                                # waited for X-RateLimit-Reset
    assert sess.calls[2][1]["page_token"] == "abc" and sess.calls[1][1]["symbols"] == "SPY,QQQ"
    h = sess.calls[0][2]
    assert h["User-Agent"] == "atlasbt-research/0.1" and set(h) >= {"APCA-API-KEY-ID", "APCA-API-SECRET-KEY"}


def test_client_never_retries_auth_errors(keys):
    sess = _FakeSession([_Resp(403, {"message": "forbidden"})])
    with pytest.raises(A.AlpacaError) as e:
        A.AlpacaClient(session=sess).get("bars", {})
    assert e.value.status == 403 and len(sess.calls) == 1
    assert "synthetic-secret" not in str(e.value)


def test_client_refuses_network_offline(keys, monkeypatch):
    monkeypatch.setenv("ATLASBT_OFFLINE", "1")
    with pytest.raises(A.AlpacaError, match="offline"):
        A.AlpacaClient(session=_FakeSession([])).get("bars", {})


def test_rate_limiter_caps_requests_per_rolling_minute(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(A.time, "time", lambda: now[0])

    def adv(s):
        now[0] += s
    monkeypatch.setattr(A.time, "sleep", adv)
    lim = A.RateLimiter(500)            # capped at MAX_RPM
    assert lim.rpm == A.MAX_RPM == 180
    stamps = []
    for _ in range(400):
        lim.wait()
        stamps.append(now[0])
    s = np.array(stamps)
    worst = max(int(np.sum((s >= x) & (s < x + 60.0))) for x in s)
    assert worst <= 180


# ------------------------------------------------------------------------------------- identity routing
class _SynthPortal:
    """synthetic_ portal: verified daily closes of two securities that both traded as 'OLD' (reused ticker)."""

    def __init__(self, ref):
        self.ref = ref

    def _split_key(self, s):
        t, _, k = s.partition("~")
        return t, (int(k[3:]) if k else None)

    def _is_plain(self, t):
        return False

    def _window_identity(self, symbol, start, end):
        return self.ref[symbol], None, {}

    def identity_index(self):
        class I:
            def lineage(self, cik, t):
                return ([t, "NEW"] if cik == 7 else [t]), []

            def sec_tickers_of(self, cik):
                return []
        return I()


def test_synthetic_route_tapes_never_splices_reused_tickers(monkeypatch):
    days = pd.DatetimeIndex(pd.bdate_range("2024-01-02", periods=6))
    idx = pd.DatetimeIndex([et(f"{d.date()} 09:30") for d in days], name="start")
    mk = lambda c, v, src: pd.DataFrame({"close": c, "volume": v, "src_ticker": src}, index=idx)
    ref = {
        # security 7 traded as OLD for 3 days, then renamed NEW; security 9 took the OLD ticker afterwards
        "OLD~CIK0000000007": mk([10, 11, 12, 13, 14, 15.0], [1e6] * 6, "OLD"),
        "OLD~CIK0000000009": mk([50, 51, 52, 53, 54, 55.0], [2e5] * 6, "OLD"),
    }
    tape = {"OLD": [10, 11, 12, 53, 54, 55.0], "NEW": [np.nan, np.nan, np.nan, 13, 14, 15.0]}
    vol = {"OLD": [1e6, 1e6, 1e6, 2e5, 2e5, 9e5], "NEW": [np.nan, np.nan, np.nan, 1e6, 1e6, 1e6]}

    def fake_raw(tickers, start, end, freq, cli=None):
        assert freq == "1d"
        return {t: pd.DataFrame({"close": tape[t], "volume": vol[t]}, index=idx).dropna() for t in tickers if t in tape}
    monkeypatch.setattr(A, "fetch_bars_raw", fake_raw)
    wanted = {k: days for k in ref}
    routes, summ = A.route_tapes(_SynthPortal(ref), wanted, cli=object())
    r7 = routes[routes["symbol"] == "OLD~CIK0000000007"].set_index("date")["tape"]
    r9 = routes[routes["symbol"] == "OLD~CIK0000000009"].set_index("date")["tape"]
    assert list(r7) == ["OLD", "OLD", "OLD", "NEW", "NEW", "NEW"]
    # security 9: only the sessions where the OLD tape matches its own verified close AND volume
    assert list(r9.index) == list(days[3:5])
    s9 = summ.set_index("symbol").loc["OLD~CIK0000000009"]
    assert s9["routed"] == 2 and s9["reference"] == 6


# ------------------------------------------------------------------------------------- portal integration
@pytest.fixture
def tmp_portal(store):
    return Portal(store), store


def _synthetic_minutes(day: str, n: int = 390, start="09:30") -> pd.DataFrame:
    t0 = et(f"{day} {start}")
    idx = pd.DatetimeIndex([t0 + pd.Timedelta(minutes=i) for i in range(n)], name="start")
    px = 100 + np.arange(n) * 0.01
    return pd.DataFrame({"open": px, "high": px + 0.02, "low": px - 0.02, "close": px + 0.01, "volume": 1000.0},
                        index=idx)


def test_synthetic_portal_prefers_alpaca_1m_and_resamples_it_over_short_native_5m(tmp_portal):
    p, st = tmp_portal
    A_ = A.canonical_bars(_synthetic_minutes("2024-01-03"), "1m")
    st.write("bars_1m", "alpaca", "SPY", A_, partition="year")
    y5 = frames.intraday_bars(_synthetic_minutes("2024-01-03", 2).iloc[:1], "5m")
    st.write("bars_5m", "yahoo", "SPY", y5, partition="year")
    a, b = et("2024-01-03 09:30"), et("2024-01-03 16:00")
    b1 = p.bars("SPY", "1m", a, b, as_of=b)
    assert b1.attrs["source"] == "alpaca" and len(b1) == 390
    b5 = p.bars("SPY", "5m", a, b, as_of=b)
    assert len(b5) == 78 and b5.attrs["source"] == "alpaca"
    assert b5["volume"].iloc[0] == 5000 and b5["open"].iloc[0] == b1["open"].iloc[0]


def test_synthetic_portal_reads_identity_keyed_intraday_and_quotes(tmp_portal):
    p, st = tmp_portal
    key = "OLD~CIK0000000007"
    st.write("bars_1m", "alpaca", key, A.canonical_bars(_synthetic_minutes("2024-01-03"), "1m"), partition="year")
    a, b = et("2024-01-03 09:30"), et("2024-01-03 16:00")
    assert len(p.bars(key, "1m", a, b, as_of=b)) == 390
    assert p.has(key, "bars", "1m")
    q = pd.DataFrame({"bid": [10.0], "ask": [10.01], "bid_size": [100.0], "ask_size": [100.0]},
                     index=pd.DatetimeIndex([et("2024-01-03 15:50:00.5")], name="ts"))
    st.write("quotes", "alpaca", key, q, partition="year")
    st.write("reference", "alpaca", "quote_windows",
             pd.DataFrame({"symbol": [key], "w0": [et("2024-01-03 15:50")], "w1": [et("2024-01-03 15:51")],
                           "window": ["close_m10"], "n": [1]}), key=["symbol", "w0"])
    assert p.has(key, "quotes") and int(p.fidelity_of(key, "quotes")) == 0
    got = p.quotes(key, et("2024-01-03 15:50") + pd.Timedelta(1, "ns"), et("2024-01-03 15:51") + pd.Timedelta(1, "ns"),
                   as_of=et("2024-01-03 15:51"))
    assert len(got) == 1 and got.attrs["source"] == "alpaca"


def test_synthetic_alpaca_quotes_served_only_inside_recorded_windows(tmp_portal):
    p, st = tmp_portal
    ts = [et("2024-01-03 09:21"), et("2024-01-03 09:29:30"), et("2024-01-03 15:50:10"), et("2024-01-03 15:50:40"),
          et("2024-01-03 15:51")]
    q = pd.DataFrame({"bid": [9.9, 9.95, 10.0, 10.01, 10.02], "ask": [10.1, 10.0, 10.01, 10.02, 10.03],
                      "bid_size": 100.0, "ask_size": 100.0}, index=pd.DatetimeIndex(ts, name="ts"))
    st.write("quotes", "alpaca", "SPY", q, partition="year")
    w = pd.DataFrame({"symbol": "SPY", "w0": [et("2024-01-03 09:20"), et("2024-01-03 15:50")],
                      "w1": [et("2024-01-03 09:30"), et("2024-01-03 15:51")], "window": ["pre_0920", "close_m10"],
                      "n": [2, 3]})
    st.write("reference", "alpaca", "quote_windows", w, key=["symbol", "w0"])
    ns = pd.Timedelta(1, "ns")

    def n(a, b):
        return len(p.quotes("SPY", a, b, as_of=b))
    # engine 1m fill window (15:50, 15:51] -> fully covered (the 15:51:00 update is inside [w0, w1])
    assert n(et("2024-01-03 15:50") + ns, et("2024-01-03 15:51") + ns) == 3
    # a 5m fill window (15:50, 15:55] is only partly covered -> NO quotes (bar fills), never a partial path
    assert n(et("2024-01-03 15:50") + ns, et("2024-01-03 15:55") + ns) == 0
    # daily-strategy window (whole session): uncovered -> no quotes
    assert n(et("2024-01-02 16:00") + ns, et("2024-01-03 16:00") + ns) == 0
    # premarket last-quote lookup [04:00, 09:30): its last minute is covered -> served (09:29 midpoint)
    pm = p.quotes("SPY", et("2024-01-03 04:00"), et("2024-01-03 09:30"), as_of=et("2024-01-03 09:31"))
    assert len(pm) == 2 and pm["bid"].iloc[-1] == 9.95
    # availability scans longer than a week (engine probes) see everything stored
    assert n(et("2023-12-20"), et("2024-01-10")) == 5
    # outside every window: empty frame, not an exception (the symbol has quotes)
    e = p.quotes("SPY", et("2024-01-03 12:00"), et("2024-01-03 12:01"), as_of=et("2024-01-03 12:01"))
    assert e.empty and "outside" in e.attrs["note"]


def test_shared_rate_limiter_caps_all_processes_together(tmp_path, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(A.time, "time", lambda: now[0])

    def adv(s):
        now[0] += s
    monkeypatch.setattr(A.time, "sleep", adv)
    a = A.SharedRateLimiter(tmp_path / "rl.json", rpm=180)
    b = A.SharedRateLimiter(tmp_path / "rl.json", rpm=180)       # a second process, same state file
    stamps = []
    for i in range(300):
        (a if i % 2 else b).wait()
        stamps.append(now[0])
    s = np.array(stamps)
    assert max(int(np.sum((s >= x) & (s < x + 60.0))) for x in s) <= 180
    a.pause(now[0] + 30)
    t0 = now[0]
    b.wait()
    assert now[0] >= t0 + 30
