"""Alpaca Market Data v2 -- SIP minute bars and NBBO quotes (LIVE-VERIFIED 2026-10-30 with a free paper key).

Env: ALPACA_API_KEY, ALPACA_API_SECRET (read from the environment only; never logged, cached or stored).
Headers APCA-API-KEY-ID / APCA-API-SECRET-KEY, User-Agent atlasbt-research/0.1.

Endpoints used (multi-symbol forms; the single-symbol `/v2/stocks/{s}/bars` behaves the same):
  GET https://data.alpaca.markets/v2/stocks/bars?symbols=A,B&timeframe=1Min|5Min|1Day&start&end
      &limit=10000&adjustment=raw&feed=sip&asof=-&page_token=
      -> {"bars": {"A": [{"t","o","h","l","c","v","n","vw"}, ...]}, "next_page_token"}
  GET https://data.alpaca.markets/v2/stocks/quotes?symbols=...&start&end&limit=10000&feed=sip&asof=-
      -> {"quotes": {"A": [{"t","bx","bp","bs","ax","ap","as","c":[..],"z"}]}, "next_page_token"}
`limit` caps the rows of a page across ALL symbols; pages are ordered by symbol, then time.

Facts verified against the live API (2026-10-30; see docs/DATA.md §4.15 for the probes):
* Coverage: SIP history starts 2016-01-04 (nothing earlier); the free plan serves it up to 15 min ago.
* `t` is the bar START (UTC): the 14:30Z 1-minute bar opens at the official open; the 1Day bar is stamped
  at ET midnight. Minute bars include extended hours (04:00-20:00 ET) -- there is no session filter, the
  canonical builder tags PRE/RTH/POST from the NYSE schedule.
* `asof`: by DEFAULT Alpaca maps renamed tickers to the current entity (META in 2016 returns FB's tape).
  We always send asof=- (no mapping): every request is for the ticker AS TRADED on the requested dates;
  which tape ticker belongs to which security on which date is decided by atlasbt's identity layer
  (`route_tapes` below), never by the vendor.
* Quotes are the consolidated NBBO stream (bx != ax on ~75-80 % of updates, so they are not per-exchange
  BBOs): each update carries the national best bid/offer and the exchange of each side. No NBBO
  rebuild from exchange quotes is needed; we clean it (see `clean_nbbo`).
* Quote SIZES: round lots (x100 shares) for sessions before 2025-11-03; SHARES from 2025-11-03 on (the
  SIP round-lot reform: SPY 2025-10-31 sizes 1, 2, 6..; 2025-11-03 sizes 40, 80, 320 = multiples of the
  new 40-share lot; KO/INTC/JPM switch the same day on every tape).
* Quote conditions: '?' (not reported) through ~2021, then 'R' (regular); 'O' (opening) is rare.
  Crossed (ask < bid) and locked (ask == bid) NBBO updates occur (0.1-4 % of updates, mostly premarket
  and in fast markets).
* Rate limit: header X-RateLimit-Limit = 200/min PER ACCOUNT on the free plan; we stay <= 180/min with a
  per-process bucket (ATLASBT_ALPACA_RPM) plus a cross-process bucket in cache/logs/alpaca_ratelimit.json
  (ATLASBT_ALPACA_SHARED_RPM), pause everyone until X-RateLimit-Reset on 429 (61 s without the header) and when
  X-RateLimit-Remaining < 10 (other consumers of the same key are invisible to our buckets).

Fidelity: bars T1_BAR (SIP consolidated trades); cleaned NBBO quotes T0_QUOTE.
Raw responses are NOT written to the HTTP disk cache (multi-GB of JSON); the parquet store is the cache.
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from typing import Iterable, Iterator, Optional

import numpy as np
import pandas as pd

from atlasbt.core.types import Fidelity

from .. import config, frames
from ._keyed import env, skipped, store_intraday, store_quotes

log = logging.getLogger("atlasbt.data.alpaca")
SOURCE = "alpaca"
BASE = "https://data.alpaca.markets/v2/stocks"
KEYS = ("ALPACA_API_KEY", "ALPACA_API_SECRET")
FIRST_SESSION = pd.Timestamp("2016-01-04")
#: first session whose SIP quote sizes are reported in shares (before: round lots of 100 shares)
SIZE_SHARES_FROM = pd.Timestamp("2025-11-03", tz="America/New_York").tz_convert("UTC")
#: NBBO update conditions kept (anything else -- non-firm, closed, halted, slow/manual -- is dropped)
QUOTE_CONDITIONS_OK = frozenset({"R", "?", "O"})
MAX_RPM = 180
TIMEFRAMES = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "60m": "1Hour", "1d": "1Day"}


class AlpacaError(RuntimeError):
    def __init__(self, status: Optional[int], msg: str):
        super().__init__(f"alpaca {status}: {msg}")
        self.status = status


def available() -> bool:
    return env(*KEYS) is not None


def _headers() -> dict:
    k = env(*KEYS)
    if k is None:
        raise AlpacaError(None, "no API key in environment")
    return {"APCA-API-KEY-ID": k["ALPACA_API_KEY"], "APCA-API-SECRET-KEY": k["ALPACA_API_SECRET"],
            "User-Agent": config.USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def alpaca_symbol(sym: str) -> str:
    """Store/identity ticker (BRK-B) -> Alpaca ticker (BRK.B)."""
    return str(sym).strip().upper().replace("-", ".").replace("/", ".")


# ------------------------------------------------------------------------------------- client
class RateLimiter:
    """Token bucket: at most `rpm` requests per rolling minute (thread-safe)."""

    def __init__(self, rpm: int = MAX_RPM):
        self.rpm = max(1, min(int(rpm), MAX_RPM))
        self._lock = threading.Lock()
        self._stamps: list[float] = []
        self._pause_until = 0.0

    def pause(self, until_epoch: float) -> None:
        with self._lock:
            self._pause_until = max(self._pause_until, until_epoch)

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.time()
                if now < self._pause_until:
                    delay = self._pause_until - now
                else:
                    self._stamps = [t for t in self._stamps if now - t < 60.0]
                    if len(self._stamps) < self.rpm:
                        self._stamps.append(now)
                        return
                    delay = 60.0 - (now - self._stamps[0]) + 0.01
            time.sleep(max(delay, 0.01))


class SharedRateLimiter:
    """Account-wide cap shared by every process on this machine: request stamps of the last minute live in
    a small file under an fcntl lock (the free plan's limit is per ACCOUNT, and several fetch tasks --
    intraday bars, quotes, the survivorship EOD patch -- may run at once)."""

    def __init__(self, path, rpm: int = MAX_RPM):
        from pathlib import Path
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rpm = max(1, min(int(rpm), MAX_RPM))

    def pause(self, until_epoch: float) -> None:
        self._update(lambda st: st.__setitem__("pause", max(st.get("pause", 0.0), until_epoch)))

    def _update(self, fn):
        import fcntl
        import json as _json
        with open(self.path, "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.seek(0)
                try:
                    st = _json.loads(fh.read() or "{}")
                except ValueError:
                    st = {}
                out = fn(st)
                fh.seek(0)
                fh.truncate()
                fh.write(_json.dumps(st))
                return out
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def wait(self) -> None:
        def take(st):
            now = time.time()
            if now < st.get("pause", 0.0):
                return st["pause"] - now
            stamps = [t for t in st.get("stamps", []) if now - t < 60.0]
            if len(stamps) < self.rpm:
                stamps.append(now)
                st["stamps"] = stamps
                return 0.0
            st["stamps"] = stamps
            return 60.0 - (now - stamps[0]) + 0.01
        while True:
            d = self._update(take)
            if d <= 0:
                return
            time.sleep(max(d, 0.01))


class AlpacaClient:
    def __init__(self, rpm: Optional[int] = None, session=None, max_retries: int = 8, timeout: float = 90.0):
        self.limiter = RateLimiter(rpm or int(os.environ.get("ATLASBT_ALPACA_RPM", MAX_RPM)))
        # real network clients also share the account-wide 180/min bucket across processes
        self.shared = SharedRateLimiter(config.CACHE_ROOT / "logs" / "alpaca_ratelimit.json",
                                        int(os.environ.get("ATLASBT_ALPACA_SHARED_RPM", MAX_RPM))) \
            if session is None else None
        self._fixed_session = session
        self._local = threading.local()             # one requests.Session per thread
        self.max_retries, self.timeout = max_retries, timeout
        self.requests = 0

    @property
    def session(self):
        import requests
        if self._fixed_session is not None:
            return self._fixed_session
        s = getattr(self._local, "s", None)
        if s is None:
            s = self._local.s = requests.Session()
        return s

    def get(self, path: str, params: dict) -> dict:
        import requests
        if config.offline():
            raise AlpacaError(None, "offline (ATLASBT_OFFLINE=1)")
        url = f"{BASE}/{path}"
        last = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            if self.shared is not None:
                self.shared.wait()
            try:
                r = self.session.get(url, params=params, headers=_headers(), timeout=self.timeout)
            except requests.RequestException as e:
                last = AlpacaError(None, f"{type(e).__name__}")
                time.sleep(min(60.0, 2 ** attempt + random.random()))
                continue
            self.requests += 1
            if r.status_code == 200:
                # the quota is per ACCOUNT: other consumers (another machine, another tool) are invisible to
                # our buckets -- when the server says the minute is nearly used up, everyone waits for reset
                rem, reset = r.headers.get("X-RateLimit-Remaining"), r.headers.get("X-RateLimit-Reset")
                if rem and rem.isdigit() and int(rem) < 10 and reset and reset.isdigit():
                    self._pause(float(reset) + 1.0)
                return r.json()
            if r.status_code == 429:
                reset = r.headers.get("X-RateLimit-Reset")
                # no reset header: wait a full minute (retrying every few seconds keeps the quota exhausted)
                until = float(reset) + 1.0 if reset and reset.isdigit() else time.time() + 61.0
                log.warning("alpaca 429: pausing %.0fs", max(until - time.time(), 0))
                self._pause(until)
                last = AlpacaError(429, "rate limited")
                continue
            if r.status_code >= 500:
                last = AlpacaError(r.status_code, r.text[:200])
                time.sleep(min(60.0, 2 ** attempt + random.random()))
                continue
            raise AlpacaError(r.status_code, r.text[:300])          # 400/401/403/422: never retried
        raise last or AlpacaError(None, "gave up")

    def _pause(self, until: float) -> None:
        self.limiter.pause(until)
        if self.shared is not None:
            self.shared.pause(until)

    def pages(self, kind: str, symbols: Iterable[str], params: dict) -> Iterator[dict]:
        """Yield {alpaca_symbol: rows} per page of /v2/stocks/{kind}?symbols=..."""
        p = dict(params, symbols=",".join(symbols))
        token = None
        while True:
            q = dict(p, **({"page_token": token} if token else {}))
            d = self.get(kind, q)
            yield d.get(kind) or {}
            token = d.get("next_page_token")
            if not token:
                return


_CLIENT: Optional[AlpacaClient] = None


def client() -> AlpacaClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = AlpacaClient()
    return _CLIENT


def _iso(ts) -> str:
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ------------------------------------------------------------------------------------- parsing
def parse_bars(rows: list[dict]) -> pd.DataFrame:
    """Alpaca bar rows -> frame indexed by bar START (UTC): open, high, low, close, volume, vwap, trades."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = pd.DataFrame({"open": df["o"].astype(float).to_numpy(), "high": df["h"].astype(float).to_numpy(),
                        "low": df["l"].astype(float).to_numpy(), "close": df["c"].astype(float).to_numpy(),
                        "volume": df["v"].astype(float).to_numpy(),
                        "vwap": df["vw"].astype(float).to_numpy() if "vw" in df else np.nan,
                        "trades": df["n"].astype(float).to_numpy() if "n" in df else np.nan},
                       index=pd.DatetimeIndex(pd.to_datetime(df["t"], utc=True, format="ISO8601"), name="start"))
    return out


def parse_quotes(rows: list[dict]) -> pd.DataFrame:
    """Alpaca NBBO rows -> frame indexed by ts (UTC): bid, ask, bid_size, ask_size (SHARES), bid_exchange,
    ask_exchange, conditions (','-joined), tape. Sizes before 2025-11-03 are round lots -> x100."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    ts = pd.DatetimeIndex(pd.to_datetime(df["t"], utc=True, format="ISO8601"), name="ts")
    mult = np.where(ts < SIZE_SHARES_FROM, 100.0, 1.0)
    cond = df["c"].map(lambda c: ",".join(c) if isinstance(c, (list, tuple)) else ("" if c is None else str(c))) \
        if "c" in df else pd.Series("", index=df.index)
    return pd.DataFrame({"bid": df["bp"].astype(float).to_numpy(), "ask": df["ap"].astype(float).to_numpy(),
                         "bid_size": df["bs"].astype(float).to_numpy() * mult,
                         "ask_size": df["as"].astype(float).to_numpy() * mult,
                         "bid_exchange": df["bx"].astype(str).to_numpy() if "bx" in df else "",
                         "ask_exchange": df["ax"].astype(str).to_numpy() if "ax" in df else "",
                         "conditions": cond.to_numpy(),
                         "tape": df["z"].astype(str).to_numpy() if "z" in df else ""}, index=ts)


def clean_nbbo(q: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Documented NBBO cleaning (docs/DATA.md §4.15):
      1. drop one-sided / non-positive sides (bid <= 0 or ask <= 0) and non-positive sizes;
      2. drop updates whose condition set is not a subset of {'R', '?', 'O'} (non-firm, closed, halted,
         slow/manual quotes);
      3. drop CROSSED (ask < bid) and LOCKED (ask == bid) updates -- not executable at a positive spread;
      4. several updates with the same timestamp -> keep the last (the state after that instant);
      5. drop consecutive duplicates of (bid, ask, bid_size, ask_size) (lossless for the NBBO state).
    Returns (clean frame [bid, ask, bid_size, ask_size], counts)."""
    n0 = len(q)
    if not n0:
        return q, {"raw": 0}
    bad_side = (q["bid"] <= 0) | (q["ask"] <= 0) | (q["bid_size"] <= 0) | (q["ask_size"] <= 0)
    cond = q["conditions"] if "conditions" in q else pd.Series("", index=q.index)
    bad_cond = ~cond.map(lambda c: set(filter(None, str(c).split(","))) <= QUOTE_CONDITIONS_OK).astype(bool)
    crossed = q["ask"] < q["bid"]
    locked = q["ask"] == q["bid"]
    keep = ~(bad_side | bad_cond | crossed | locked)
    c = q.loc[keep, ["bid", "ask", "bid_size", "ask_size"]]
    c = c[~c.index.duplicated(keep="last")]
    if len(c):
        v = c.to_numpy()
        same = np.r_[False, (v[1:] == v[:-1]).all(axis=1)]
        c = c[~same]
    stats = {"raw": int(n0), "one_sided": int(bad_side.sum()), "condition": int((bad_cond & ~bad_side).sum()),
             "crossed": int((crossed & ~bad_side).sum()), "locked": int((locked & ~bad_side).sum()),
             "kept": int(len(c))}
    return c, stats


# ------------------------------------------------------------------------------------- fetching
def fetch_bars_raw(tickers: list[str], start, end, freq: str = "1m", feed: str = "sip",
                   cli: Optional[AlpacaClient] = None) -> dict[str, pd.DataFrame]:
    """Raw (unadjusted, as-traded tape, asof=-) bars of `tickers` (Alpaca spelling) in [start, end]:
    {ticker: frame indexed by bar start}. No identity routing, no storage."""
    cli = cli or client()
    params = {"timeframe": TIMEFRAMES[freq], "start": _iso(start), "end": _iso(end), "limit": 10000,
              "adjustment": "raw", "feed": feed, "asof": "-", "sort": "asc"}
    acc: dict[str, list] = {}
    for page in cli.pages("bars", tickers, params):
        for s, rows in page.items():
            acc.setdefault(s, []).extend(rows)
    return {s: parse_bars(rows) for s, rows in acc.items() if rows}


def fetch_quotes_raw(tickers: list[str], start, end, feed: str = "sip",
                     cli: Optional[AlpacaClient] = None) -> dict[str, pd.DataFrame]:
    """Raw NBBO updates of `tickers` in [start, end] (both inclusive at the API): {ticker: parse_quotes}."""
    cli = cli or client()
    params = {"start": _iso(start), "end": _iso(end), "limit": 10000, "feed": feed, "asof": "-", "sort": "asc"}
    acc: dict[str, list] = {}
    for page in cli.pages("quotes", tickers, params):
        for s, rows in page.items():
            acc.setdefault(s, []).extend(rows)
    return {s: parse_quotes(rows) for s, rows in acc.items() if rows}


def canonical_bars(df: pd.DataFrame, freq: str, sessions: Optional[tuple] = None) -> pd.DataFrame:
    """Alpaca raw bars -> canonical intraday frame (ARCHITECTURE §3): session tags from the NYSE
    schedule, known_at = bar end capped at the close; rows on non-sessions dropped."""
    if df is None or not len(df):
        return frames.empty_bars()
    out = frames.intraday_bars(df, freq)
    if sessions:
        out = out[out["session"].isin(tuple(sessions))]
    return out


def fetch_bars(symbol: str, start, end, freq: str = "1m", feed: str = "sip", sessions=None,
               store=None) -> dict:
    """Single-symbol convenience: fetch the RAW tape of `symbol` (no identity routing -- use
    route_tapes + fetch_routed_bars for universe members) and store bars_<freq>/alpaca/<symbol>."""
    if not available():
        return skipped("alpaca")
    got = fetch_bars_raw([alpaca_symbol(symbol)], start, end, freq, feed)
    df = canonical_bars(next(iter(got.values()), pd.DataFrame()), freq, sessions)
    n = store_intraday(SOURCE, symbol, freq, df, Fidelity.T1_BAR, store=store,
                       extra={"feed": feed, "asof": "-", "adjustment": "raw", "verified": True}) if len(df) else 0
    return {"symbol": symbol, "rows": n}


def fetch_quotes(symbol: str, start, end, feed: str = "sip", store=None) -> dict:
    """Single-symbol convenience: cleaned NBBO of `symbol` in [start, end] -> quotes/alpaca/<symbol>.
    (Coverage windows are recorded only by fetch_quote_windows.)"""
    if not available():
        return skipped("alpaca")
    got = fetch_quotes_raw([alpaca_symbol(symbol)], start, end, feed)
    q, stats = clean_nbbo(next(iter(got.values()), pd.DataFrame()))
    n = store_quotes(SOURCE, symbol, q, Fidelity.T0_QUOTE, store=store, extra={"feed": feed, "nbbo": "sip"}) \
        if len(q) else 0
    return {"symbol": symbol, "rows": n, **stats}


# =====================================================================================================
# Universe fetches with identity routing (python -m atlasbt.data.fetch --set intraday)
# =====================================================================================================
# Alpaca serves tape tickers; the engine asks for universe symbols (a ticker, or a security key
# 'TICKER~CIK##########' for reused tickers). Which tape ticker carried the security on which session is
# decided ONLY from our identity-verified daily series (the same one bars(..., '1d') serves): for every
# wanted session the RAW daily close/volume of the security is compared with Alpaca's 1Day bar of each
# candidate tape ticker (the symbol's own ticker first, then the CIK's evidenced/SEC tickers). A tape is
# accepted for a session only if |log close ratio| <= 0.5 % and |log volume ratio| <= log(2.5); sessions
# without an accepted tape get NO intraday bars (fail closed; counted in the route log). Intraday rows are
# stored under the universe symbol (bars_<freq>/alpaca/<symbol>), so reused tickers never splice and a
# renamed security (FB -> META) carries its own tape history.
CLOSE_TOL = 0.005
VOLUME_TOL = float(np.log(2.5))


def _ledger_path(name: str):
    p = config.CACHE_ROOT / "logs" / f"alpaca_progress_{name}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class Ledger:
    """Append-only JSONL of completed work units (resumable fetches)."""

    def __init__(self, name: str):
        import json
        self.path = _ledger_path(name)
        self._lock = threading.Lock()
        self.done: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    r = json.loads(line)
                    self.done[r["k"]] = r
                except Exception:
                    continue

    def has(self, key: str) -> bool:
        return key in self.done

    def add(self, key: str, **info) -> None:
        import json
        rec = {"k": key, "at": pd.Timestamp.now(tz="UTC").isoformat(), **info}
        with self._lock:
            self.done[key] = rec
            with self.path.open("a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")


def _reference_daily(portal, symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Identity-verified RAW daily close/volume of `symbol` on `dates` (naive session dates) + src tickers."""
    from atlasbt.core.types import DataUnavailableError
    t, kc = portal._split_key(symbol)
    if kc is None and portal._is_plain(t):
        f, allowed = portal._plain_frame(t), None
    else:
        try:
            f, allowed, _ = portal._window_identity(symbol, pd.Timestamp(dates[0], tz="UTC"),
                                                    pd.Timestamp(dates[-1], tz="UTC") + pd.Timedelta(days=1))
        except DataUnavailableError as e:
            log.warning("route %s: %s", symbol, e)
            return pd.DataFrame()
    if not len(f):
        return pd.DataFrame()
    dd = frames.to_naive_date(f.index)
    m = dd.isin(dates)
    if allowed is not None:
        m &= dd.isin(allowed)
    out = pd.DataFrame({"close": f["close"].to_numpy()[m], "volume": f["volume"].to_numpy()[m]}, index=dd[m])
    out["src_ticker"] = f["src_ticker"].to_numpy()[m] if "src_ticker" in f.columns else t
    return out[~out.index.duplicated(keep="last")]


def _candidates(portal, symbol: str, ref: pd.DataFrame) -> list[str]:
    from atlasbt.data.identity import norm
    t, kc = portal._split_key(symbol)
    out = [t]
    out += [norm(x) for x in pd.unique(ref["src_ticker"].dropna().astype(str))] if len(ref) else []
    if not (kc is None and portal._is_plain(t)):
        idx = portal.identity_index()
        cik = kc
        if cik is None:
            c, _, _ = portal._identity_by_date(t, [ref.index[-1]]) if len(ref) else ([0], None, None)
            cik = int(c[0]) or None
        if cik:
            main, extra = idx.lineage(int(cik), t)
            out += main + extra + idx.sec_tickers_of(int(cik))
    return list(dict.fromkeys(alpaca_symbol(x) for x in out if x and x != "NAN"))


def route_tapes(portal, wanted: dict[str, pd.DatetimeIndex], cli: Optional[AlpacaClient] = None,
                chunk: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (routes [symbol, date, tape, ref_close, tape_close], summary per symbol). See block comment."""
    cli = cli or client()
    refs, cands = {}, {}
    for s, dates in wanted.items():
        dates = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(dates).normalize())))
        ref = _reference_daily(portal, s, dates) if len(dates) else pd.DataFrame()
        refs[s] = ref
        cands[s] = _candidates(portal, s, ref) if len(ref) else []
    alltk = sorted({c for v in cands.values() for c in v})
    lo = min((r.index[0] for r in refs.values() if len(r)), default=None)
    hi = max((r.index[-1] for r in refs.values() if len(r)), default=None)
    daily: dict[str, pd.DataFrame] = {}
    if lo is not None:
        for i in range(0, len(alltk), chunk):
            got = fetch_bars_raw(alltk[i:i + chunk], pd.Timestamp(lo, tz="America/New_York"),
                                 pd.Timestamp(hi, tz="America/New_York") + pd.Timedelta(days=1), "1d", cli=cli)
            for tk, df in got.items():
                daily[tk] = pd.DataFrame({"close": df["close"].to_numpy(), "volume": df["volume"].to_numpy()},
                                         index=frames.to_naive_date(df.index))
    rows, summ = [], []
    for s, ref in refs.items():
        n_want = len(wanted[s])
        if not len(ref):
            summ.append({"symbol": s, "wanted": n_want, "reference": 0, "routed": 0, "tapes": ""})
            continue
        best = pd.Series(index=ref.index, dtype=object)
        tclose = pd.Series(np.nan, index=ref.index)
        for tk in cands[s]:                      # the symbol's own ticker first: it wins ties
            d = daily.get(tk)
            if d is None or not len(d):
                continue
            d = d[~d.index.duplicated(keep="last")].reindex(ref.index)
            with np.errstate(divide="ignore", invalid="ignore"):
                cr = np.abs(np.log(d["close"] / ref["close"]))
                vr = np.abs(np.log(d["volume"] / ref["volume"]))
            ok = (cr <= CLOSE_TOL) & ((vr <= VOLUME_TOL) | ~(ref["volume"] > 0)) & best.isna()
            best[ok] = tk
            tclose[ok] = d["close"][ok]
        m = best.notna()
        for dte, tk, rc, tc in zip(ref.index[m], best[m], ref["close"][m], tclose[m]):
            rows.append((s, dte, tk, float(rc), float(tc)))
        summ.append({"symbol": s, "wanted": n_want, "reference": int(len(ref)), "routed": int(m.sum()),
                     "tapes": ",".join(pd.unique(best[m].astype(str)))})
    routes = pd.DataFrame(rows, columns=["symbol", "date", "tape", "ref_close", "tape_close"])
    return routes, pd.DataFrame(summ)


def _session_bounds(dates: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC [04:00 ET of the first session, 20:00 ET of the last session]."""
    a = pd.Timestamp(dates.min()).tz_localize("America/New_York") + pd.Timedelta(hours=4)
    b = pd.Timestamp(dates.max()).tz_localize("America/New_York") + pd.Timedelta(hours=20)
    return a.tz_convert("UTC"), b.tz_convert("UTC")


def fetch_routed_bars(routes: pd.DataFrame, freq: str = "1m", sessions: Optional[tuple] = ("PRE", "RTH"),
                      ledger: Optional[Ledger] = None, chunk: int = 10, workers: int = 2, store=None,
                      budget_bytes: Optional[int] = None, cli: Optional[AlpacaClient] = None,
                      slim: bool = False, newest_first: bool = False) -> dict:
    """Fetch `freq` bars for every (symbol, session) in `routes` (one tape per session), month by month,
    `chunk` symbols per multi-symbol request; store bars_<freq>/alpaca/<symbol>. A (symbol, month) is
    recorded in the ledger once stored; a month still in progress (< 1 day old) is never recorded."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from ..store import default_store
    cli = cli or client()
    store = store or default_store()
    ledger = ledger or Ledger(f"bars_{freq}")
    r = routes.copy()
    r["date"] = pd.to_datetime(r["date"])
    r["month"] = r["date"].dt.strftime("%Y-%m")
    now_d = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    units = []
    groups = list(r.groupby("month", sort=True))
    for month, g in (reversed(groups) if newest_first else groups):
        todo = sorted(s for s in g["symbol"].unique() if not ledger.has(f"{freq}|{s}|{month}"))
        for i in range(0, len(todo), chunk):
            units.append((month, todo[i:i + chunk], g[g["symbol"].isin(todo[i:i + chunk])]))
    stats = {"units": len(units), "rows": 0, "symbol_months": 0, "errors": 0, "stopped": None}
    month_end = {m: (pd.Timestamp(m + "-01") + pd.offsets.MonthEnd(0)) for m in r["month"].unique()}

    def work(u):
        month, syms, g = u
        tapes = sorted(g["tape"].unique())
        a, b = _session_bounds(pd.DatetimeIndex(g["date"]))
        raw = fetch_bars_raw(tapes, a, b, freq, cli=cli)
        out = {}
        for s in syms:
            gs = g[g["symbol"] == s]
            parts = []
            for tk, gt in gs.groupby("tape"):
                df = raw.get(tk)
                if df is None or not len(df):
                    continue
                keep = frames.to_naive_date(df.index).isin(pd.DatetimeIndex(gt["date"]))
                parts.append(df[keep])
            df = pd.concat(parts).sort_index() if parts else pd.DataFrame()
            c = canonical_bars(df, freq, sessions)
            if slim:                       # optional columns dropped (disk budget; S039/S040 read OHLCV only)
                c = c.drop(columns=[x for x in ("vwap", "trades") if x in c.columns])
            n = store_intraday(SOURCE, s, freq, c, Fidelity.T1_BAR, store=store,
                               extra={"feed": "sip", "asof": "-", "adjustment": "raw", "verified": True,
                                      "identity_routed": True, "sessions": list(sessions or [])}) if len(c) else 0
            out[s] = (len(c), n)
        return month, out

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {}
        it = iter(units)
        for u in it:
            futs[ex.submit(work, u)] = u
            if len(futs) >= workers * 2:
                break
        while futs:
            for f in as_completed(list(futs)):
                u = futs.pop(f)
                try:
                    month, out = f.result()
                    complete = month_end[month] < now_d
                    for s, (k, _) in out.items():
                        stats["rows"] += k
                        stats["symbol_months"] += 1
                        if complete:
                            ledger.add(f"{freq}|{s}|{month}", rows=k)
                    log.info("bars %s %s: %d symbols, %d rows (requests so far %d)", freq, month, len(out),
                             sum(k for k, _ in out.values()), cli.requests)
                except Exception as e:  # keep going; the unit stays out of the ledger -> retried next run
                    stats["errors"] += 1
                    log.error("bars %s %s %s failed: %s", freq, u[0], u[1][:3], e)
                if budget_bytes and stats["symbol_months"] % 50 == 0 and \
                        _kind_bytes(store, f"bars_{freq}") > budget_bytes:
                    stats["stopped"] = "disk budget reached"
                    log.error("bars_%s/alpaca exceeds the disk budget (%.2f GB): stopping", freq, budget_bytes / 1e9)
                    for f2 in futs:
                        f2.cancel()
                    futs = {}
                    break
                nxt = next(it, None)
                if nxt is not None:
                    futs[ex.submit(work, nxt)] = nxt
                break
    return stats


def _kind_bytes(store, kind: str) -> int:
    d = store.root / kind / SOURCE
    return sum(f.stat().st_size for f in d.rglob("*.parquet")) if d.is_dir() else 0


# ------------------------------------------------------------------------------- quote windows
#: Decision windows (minutes relative to the session open / close; early closes shift with the close).
#: Each covers a whole engine 1-minute fill window (bar [T, T+1m) is filled from quotes in (T, T+1m]);
#: coverage is recorded as the closed interval [w0, w1] actually fetched.
QUOTE_WINDOWS = {
    "pre_0920": ("open", -10, 0),        # S005/S006: last NBBO at/before 09:29:59 (premarket midpoint)
    "close_m36": ("close", -36, -35),    # S015: 15:25 midpoint (last quote in [15:24, 15:25])
    "close_m30": ("close", -30, -29),    # S015: market entry at 15:30 -> fills in (15:30, 15:31]
    "close_m10": ("close", -10, -9),     # S031-S037: entries at 15:50 -> fills in (15:50, 15:51]
    "close_m5": ("close", -5, -4),       # intraday flat_by 15:55 exits -> fills in (15:55, 15:56]
    "open_p5": ("open", 5, 6),           # S031-S037: exits at 09:35 -> fills in (09:35, 09:36]
}


def window_bounds(session, name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    anchor, a, b = QUOTE_WINDOWS[name]
    o, c = frames.schedule().loc[pd.Timestamp(session).normalize(), ["open", "close"]]
    base = pd.Timestamp(o if anchor == "open" else c)
    base = base.tz_localize("UTC") if base.tzinfo is None else base.tz_convert("UTC")
    return base + pd.Timedelta(minutes=a), base + pd.Timedelta(minutes=b)


def fetch_quote_windows(plan: pd.DataFrame, ledger: Optional[Ledger] = None, store=None, workers: int = 2,
                        budget_bytes: Optional[int] = None, cli: Optional[AlpacaClient] = None) -> dict:
    """plan: [symbol, date, tape, window]. Per (month, window): for each session one multi-symbol NBBO request
    over [w0, w1] (paginated), cleaned (clean_nbbo), routed to the universe symbols, written once per
    (symbol, month), and the coverage [w0, w1] of every (symbol, session) recorded in
    reference/alpaca/quote_windows (the Portal serves Alpaca quotes only inside recorded windows)."""
    from concurrent.futures import ThreadPoolExecutor
    from ..store import default_store
    cli = cli or client()
    store = store or default_store()
    ledger = ledger or Ledger("quotes")
    p = plan.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["month"] = p["date"].dt.strftime("%Y-%m")
    now_d = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    stats = {"units": 0, "raw": 0, "kept": 0, "crossed": 0, "locked": 0, "condition": 0, "one_sided": 0,
             "errors": 0, "stopped": None}
    units = [(m, w, g) for (m, w), g in p.groupby(["month", "window"], sort=True)
             if not ledger.has(f"q|{w}|{m}")]
    stats["units"] = len(units)
    lock = threading.Lock()

    def day(args):
        d, w, g = args
        w0, w1 = window_bounds(d, w)
        got = fetch_quotes_raw(sorted(g["tape"].unique()), w0, w1, cli=cli)
        return d, w0, w1, got

    for month, w, g in units:
        acc: dict[str, list] = {}
        cov = []
        try:
            jobs = [(d, w, gd) for d, gd in g.groupby("date")]
            with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                for d, w0, w1, got in ex.map(day, jobs):
                    gd = g[g["date"] == d]
                    cleaned = {}
                    for tk, q in got.items():
                        c, st_ = clean_nbbo(q)
                        cleaned[tk] = c
                        with lock:
                            for k in ("raw", "kept", "crossed", "locked", "condition", "one_sided"):
                                stats[k] += st_.get(k, 0)
                    for s, tk in zip(gd["symbol"], gd["tape"]):
                        c = cleaned.get(tk)
                        if c is not None and len(c):
                            acc.setdefault(s, []).append(c)
                        cov.append((s, w0, w1, w, 0 if c is None else len(c)))
            for s, parts in acc.items():
                store_quotes(SOURCE, s, pd.concat(parts), Fidelity.T0_QUOTE, store=store,
                             extra={"feed": "sip", "nbbo": "sip consolidated, cleaned (clean_nbbo)",
                                    "asof": "-", "windows": "reference/alpaca/quote_windows", "verified": True})
            cv = pd.DataFrame(cov, columns=["symbol", "w0", "w1", "window", "n"])
            store.write("reference", SOURCE, "quote_windows", cv, key=["symbol", "w0"],
                        extra={"note": "closed UTC intervals [w0, w1] fully fetched from Alpaca SIP NBBO"})
            if (pd.Timestamp(month + "-01") + pd.offsets.MonthEnd(0)) < now_d:
                ledger.add(f"q|{w}|{month}", symbols=len(acc), quotes=int(sum(len(x) for v in acc.values()
                                                                             for x in v)))
            log.info("quotes %s %s: %d symbols, %d sessions (requests so far %d)", w, month, len(acc),
                     g["date"].nunique(), cli.requests)
        except Exception as e:
            stats["errors"] += 1
            log.error("quotes %s %s failed: %s", w, month, e)
        if budget_bytes and _kind_bytes(store, "quotes") > budget_bytes:
            stats["stopped"] = "disk budget reached"
            log.error("quotes/alpaca exceeds the disk budget (%.2f GB): stopping", budget_bytes / 1e9)
            break
    return stats
