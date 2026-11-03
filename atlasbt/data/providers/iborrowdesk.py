"""IBorrowDesk -- Interactive Brokers stock-loan history (borrow fee, rebate, shares available).

What is reachable (checked 2026-10-30, UA `atlasbt-research/0.1`):
  * `iborrowdesk.com` (bare host) only redirects to `www.iborrowdesk.com`; the old internal
    endpoint `/api/ticker/{T}` drops the connection for scripted clients. The site's llms.txt
    says those `/api/` endpoints are the web app's internal interface that "block scripted
    user agents" -- we do NOT use them.
  * The supported programmatic access is the KEYED v2 API (paid plan: $29/month with 5,000
    units, unit packs on top): `GET /api/v2/daily/borrow?symbols=A,B&start=YYYY-MM-DD` ->
    daily open/high/low/close of fee, rebate and availability back to 2015-07-13. One unit per
    symbol per 365 days of history returned; a request's worst case must stay <= 150 units;
    <= 100 symbols per call; 60 requests/min. `GET /api/v2/coverage` is free (no key).
  * IBKR's own public short-stock files (ftp3.interactivebrokers.com, user `shortstock`) are a
    CURRENT snapshot only (no history) and FTP is not reachable from this environment.

=> NEEDS USER: an IBorrowDesk API key (env IBORROWDESK_API_KEY). Without it `available()` is False
and `fetch_daily` returns {'skipped': ...}; the portal keeps the default borrow model (T2).

Stored frame (borrow/iborrowdesk/<TICKER>, one row per US trading date), PIT:
  date (naive ET date), time (= known_at, UTC): 16:00 ET of `date` -- the daily `close` is IB's
  last reading inside the US session, so it is known by the close at the latest (early closes:
  still 16:00, i.e. conservative); fee (close, % p.a.), fee_open/high/low, rebate (close, % p.a.),
  available (close, shares; the feed caps at 10,000,000), available_open, figi, name, source.
`Portal.borrow(symbol, as_of)` serves the last row with time <= as_of (fee / 100, shortable =
available > 0) as T1_BAR, falls back to the default model (T2_PROXY) before the first row or when
the last row is older than Portal.MAX_BORROW_STALE_DAYS.

Identity: the API resolves a ticker to the security that CURRENTLY trades under it and returns
that security's history across renames (its example: META back to 2015 = the FB years). We
therefore request current tickers only, store them ticker-keyed like the other CURRENT stores
(Portal.resolve_symbol serves them to a security key only when the key's CIK is the ticker's
current registrant) and drop a response whose name shares no word with the SEC registrant name.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from typing import Iterable, Optional

import pandas as pd

from atlasbt.core.clock import et
from atlasbt.core.types import Fidelity

from ..http import FetchError, client
from ..store import ParquetStore, default_store

log = logging.getLogger("atlasbt.data.iborrowdesk")
SOURCE = "iborrowdesk"
BASE = "https://www.iborrowdesk.com/api/v2"
ENV_KEY = "IBORROWDESK_API_KEY"
DAILY_FLOOR = "2015-07-13"         # /coverage daily_earliest_available (2026-10-30)
MAX_SYMBOLS = 100
MAX_UNITS_PER_REQUEST = 150
UNIT_DAYS = 365
MIN_INTERVAL = 1.1                 # seconds between requests (API limit 60/min; be polite)
NON_RETRYABLE = (400, 401, 402, 403)

_last_call = [0.0]


def available() -> bool:
    return bool(os.environ.get(ENV_KEY, "").strip())


def _wait():
    dt = time.monotonic() - _last_call[0]
    if dt < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - dt)
    _last_call[0] = time.monotonic()


def coverage() -> dict:
    """Free, unauthenticated limits document (cached one day)."""
    _wait()
    return json.loads(client().get(f"{BASE}/coverage", ttl=86400))


def units_worst_case(n_symbols: int, start: Optional[str], end: Optional[str] = None) -> int:
    s = pd.Timestamp(start or DAILY_FLOOR)
    e = pd.Timestamp(end) if end else pd.Timestamp.now().normalize()
    days = max(1, (e - s).days + 1)
    return n_symbols * int(math.ceil(days / UNIT_DAYS))


def batches(symbols: list[str], start: Optional[str], end: Optional[str] = None) -> list[list[str]]:
    per = units_worst_case(1, start, end)
    size = max(1, min(MAX_SYMBOLS, MAX_UNITS_PER_REQUEST // per))
    return [symbols[i:i + size] for i in range(0, len(symbols), size)]


def parse_daily(payload: dict) -> dict[str, pd.DataFrame]:
    """v2 daily response -> {TICKER: frame} (columns in the module doc). US lines only."""
    out: dict[str, pd.DataFrame] = {}
    for sym, node in (payload.get("data") or {}).items():
        if not isinstance(node, dict):
            continue
        if str(node.get("country") or "usa").lower() not in ("usa", "us"):
            continue
        rows = []
        for o in node.get("observations") or []:
            d = pd.Timestamp(o["date"]).normalize()
            fee, reb, av = o.get("fee") or {}, o.get("rebate") or {}, o.get("available") or {}
            if fee.get("close") is None:
                continue
            rows.append({"date": d, "time": et(d, "16:00").tz_convert("UTC"),
                         "fee": float(fee["close"]), "fee_open": _f(fee.get("open")), "fee_high": _f(fee.get("high")),
                         "fee_low": _f(fee.get("low")), "rebate": _f(reb.get("close")),
                         "available": _f(av.get("close")), "available_open": _f(av.get("open"))})
        if not rows:
            continue
        df = pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        df["figi"] = node.get("figi")
        df["name"] = node.get("name")
        df["source"] = SOURCE
        out[str(sym).upper()] = df
    return out


def _f(x):
    return float(x) if x is not None else float("nan")


_STOP = {"INC", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "PLC", "THE", "HOLDINGS", "GROUP", "CLASS",
         "CL", "A", "B", "C", "NEW", "DE", "LP", "LLC", "N.V.", "NV", "SA", "AG", "TRUST", "ETF", "&", "OF"}


def names_agree(a: Optional[str], b: Optional[str]) -> bool:
    """True when two issuer names share a significant word (or either is unknown)."""
    if not a or not b:
        return True
    ta = {w for w in re.split(r"[^A-Z0-9]+", a.upper()) if w and w not in _STOP and len(w) > 1}
    tb = {w for w in re.split(r"[^A-Z0-9]+", b.upper()) if w and w not in _STOP and len(w) > 1}
    return not ta or not tb or bool(ta & tb)


def short_universe(store: Optional[ParquetStore] = None, since: str = DAILY_FLOOR) -> list[str]:
    """Every stock / ETF a short-selling strategy can touch from `since`: PIT S&P 500 members whose
    membership overlaps [since, today] (Yahoo-style base tickers) + the fixed ETF lists."""
    from ..universes import CORE_ETFS, LEVERAGED_ETFS
    store = store or default_store()
    syms: set[str] = set(CORE_ETFS) | set(LEVERAGED_ETFS)
    m = store.read("reference", "sp500", "membership")
    if len(m):
        keep = m["end"].isna() | (pd.to_datetime(m["end"]) >= pd.Timestamp(since))
        syms |= {str(b).upper().replace(".", "-") for b in m.loc[keep, "base"]}
    return sorted(syms)


def fetch_daily(symbols: Optional[Iterable[str]] = None, start: Optional[str] = None, end: Optional[str] = None,
                store: Optional[ParquetStore] = None, names: Optional[dict] = None) -> dict:
    """Fetch daily borrow history (keyed v2 API) and store it PIT. `names`: ticker -> SEC registrant
    name for the identity check (default: the SEC ticker map)."""
    store = store or default_store()
    syms = sorted({s.upper() for s in (symbols or short_universe(store))})
    plan = {"symbols": len(syms), "worst_case_units": units_worst_case(len(syms), start, end),
            "requests": len(batches(syms, start, end))}
    if not available():
        return {"skipped": f"{ENV_KEY} not set (IBorrowDesk v2 API is keyed/paid) -- NEEDS USER", **plan}
    key = os.environ[ENV_KEY].strip()
    if names is None:
        tm = store.read("reference", "sec", "ticker_map")
        names = dict(zip(tm["ticker"].str.upper(), tm["name"])) if len(tm) else {}
    out = {"stored": 0, "rows": 0, "unresolved": [], "name_mismatch": [], "units_billed": 0, **plan}
    for b in batches(syms, start, end):
        params = {"symbols": ",".join(b)}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        _wait()
        try:
            body = client().get(f"{BASE}/daily/borrow", params=params,
                                headers={"Authorization": f"Bearer {key}"}, ttl=86400)
        except FetchError as e:
            if e.status in NON_RETRYABLE:      # bad key / no plan / units exhausted: stop, never hammer
                out["error"] = f"HTTP {e.status}: {str(e)[:200]}"
                break
            out.setdefault("failed_batches", []).append(b[0])
            continue
        payload = json.loads(body)
        meta = payload.get("meta") or {}
        out["units_billed"] += int(meta.get("units_billed") or 0)
        out["unresolved"] += list(meta.get("unresolved") or [])
        for sym, df in parse_daily(payload).items():
            if not names_agree(df["name"].iloc[0], names.get(sym)):
                out["name_mismatch"].append(sym)
                log.warning("iborrowdesk %s: name %r != SEC %r -> dropped", sym, df["name"].iloc[0], names.get(sym))
                continue
            store.write("borrow", SOURCE, sym, df, key=["date"], replace=start is None,
                        fidelity=int(Fidelity.T1_BAR), extra={"api": "v2/daily/borrow"})
            out["stored"] += 1
            out["rows"] += len(df)
    return out
