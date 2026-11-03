"""Tiingo end-of-day prices -- survivorship patch for S&P 500 members delisted before 2016.

LIVE-VERIFIED 2026-10-30 with the user's free key (env TIINGO_API_KEY; sent ONLY as the
`Authorization: Token ...` header, so it never reaches the raw-response cache, whose metadata keeps
url + params):
  GET https://api.tiingo.com/tiingo/daily/{t}/prices?startDate&endDate&format=json
  -> [{"date","open","high","low","close","volume" (AS TRADED), "adj*", "divCash","splitFactor"}]
  BCR 2011-01-03: close 92.25, volume 855,600 (C.R. Bard, delisted 2017-12-29).
  Supported-ticker list (free, no key): https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip
  (ticker, exchange, assetType, priceCurrency, startDate, endDate; one row per price LINE: reused
  tickers have several rows, e.g. DD 1962-2026 and 2017-08-03..2019-05-31).

Semantics (why the identity layer treats Tiingo differently from DoltHub / Alpaca): a Tiingo line is
keyed by the ticker the security LAST traded under, and carries that security's whole history
(renamed history included, like Yahoo). /tiingo/daily/BBT for 2008 serves the line that holds 'BBT'
now, not BB&T. So identity.build_series accepts Tiingo rows for CIK c only when
  * the line's owner -- the issuer the ticker denoted at the line's end date (as-of 8-K evidence;
    the SEC current map for live lines) -- is c, and the ticker has ONE line in the supported list
    (several lines = ambiguous -> rows need price agreement with a verified segment), and
  * the date lies inside c's evidence span (first evidence of c under any ticker - 120 days),
  * rows on dates the ticker denoted ANOTHER issuer are dropped; rows on dates the ticker denoted
    no one yet (renamed history) need price agreement with a verified segment.

Free-tier limits (tiingo.com/about/pricing, 2026-10): 50 unique symbols / hour, 1,000 requests /
day, 500 unique symbols / month. `Budget` keeps a persistent request ledger (cache/logs/
tiingo_requests.jsonl) and stays at <= 40 symbols / rolling hour, <= 400 requests / day and
<= 400 symbols / month; responses are cached on disk, so `fetch_eod_many` is resumable across days.

Validation before ingest (`validate`): Tiingo raw close / volume vs overlapping trusted as-traded
sources (Yahoo raw, Alpaca SIP, DoltHub) and Tiingo splits vs Yahoo splits; `ingest` refuses to
store anything when the source-level check fails and drops a symbol whose overlap disagrees.
Tiingo is used as a BACKFILL: where a trusted source covers the same issuer later, a symbol is
gated on the first 250 overlapping sessions (junction) and only rows up to that window are stored.
Stored: bars_1d/tiingo/<T> (raw, T1_BAR; served through the identity layer, T2 in bars() attrs)
and actions/tiingo/<T> (divCash -> CASH_DIVIDEND, splitFactor -> SPLIT; known_at = previous close).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from atlasbt.core.types import Fidelity

from .. import config, frames
from ..http import client
from ..store import ParquetStore, default_store
from ._keyed import env, skipped, store_intraday

log = logging.getLogger("atlasbt.data.tiingo")
SOURCE = "tiingo"
EOD_URL = "https://api.tiingo.com/tiingo/daily/{t}/prices"
SUPPORTED_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
START = "2006-01-01"                 # identity evidence (ZipLime as-of tickers) starts 2006
MAX_SYMBOLS_HOUR, MAX_REQ_DAY, MAX_SYMBOLS_MONTH = 40, 400, 400
VALIDATION_SET = ("AAPL", "KO", "XOM", "JPM", "GE", "MSFT", "PG", "INTC")   # survivors: Yahoo overlap 2006 ->
JUNCTION = 250                       # sessions of trusted overlap that gate a symbol (see validate)


def available() -> bool:
    return env("TIINGO_API_KEY") is not None


def _headers() -> dict:
    return {"Authorization": f"Token {env('TIINGO_API_KEY')['TIINGO_API_KEY']}", "Content-Type": "application/json"}


def tiingo_symbol(t: str) -> str:
    return str(t).strip().lower().replace(".", "-")


# ------------------------------------------------------------------------------------- parsing
def parse_eod(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (raw daily bars, actions). splitFactor = new shares per old share on the ex-date."""
    df = pd.DataFrame(rows)
    if df.empty:
        return frames.empty_bars(), pd.DataFrame()
    df.index = pd.DatetimeIndex(pd.to_datetime(df["date"].astype(str).str[:10]))
    # after a delisting Tiingo repeats the last close with volume 0 for months (CVH 2013-05-07 ..
    # 2013-10-17, AYE, MHS): not prices -> dropped (the line's real end is its last traded session)
    df = df[(pd.to_numeric(df["close"], errors="coerce") > 0) & (pd.to_numeric(df["volume"], errors="coerce") > 0)]
    bars, _ = frames.daily_bars(df[["open", "high", "low", "close", "volume"]])
    sch = frames.schedule()
    acts = []
    for d, r in df.iterrows():
        if d not in sch.index:
            continue
        i = sch.index.get_loc(d)
        known = sch["close"].iloc[i - 1] if i > 0 else sch["open"].iloc[i]
        sf = float(r.get("splitFactor") or 1.0)
        if sf > 0 and abs(sf - 1.0) > 1e-9:
            acts.append({"type": "SPLIT", "ex_date": d, "known_at": known, "ratio": sf, "amount": 0.0,
                         "fidelity": int(Fidelity.T2_PROXY)})
        dv = float(r.get("divCash") or 0.0)
        if dv > 0:
            acts.append({"type": "CASH_DIVIDEND", "ex_date": d, "known_at": known, "ratio": 1.0, "amount": dv,
                         "fidelity": int(Fidelity.T2_PROXY)})
    return bars, pd.DataFrame(acts)


# ------------------------------------------------------------------------------------- budget
class Budget:
    """Persistent free-tier budget (survives restarts / days): one JSON line per network request."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or (config.CACHE_ROOT / "logs" / "tiingo_requests.jsonl"))
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for ln in self.path.read_text().splitlines():
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
        return out

    def wait_s(self, symbol: str, now: Optional[float] = None) -> float:
        """Seconds to wait before a request for `symbol` fits all limits (inf: monthly cap reached)."""
        now = time.time() if now is None else now
        rows = self._rows()
        hour = [r for r in rows if now - r["t"] < 3600]
        day = [r for r in rows if now - r["t"] < 86400]
        month = {r["s"] for r in rows if now - r["t"] < 30 * 86400}
        if symbol not in month and len(month) >= MAX_SYMBOLS_MONTH:
            return float("inf")
        w = 0.0
        if len(day) >= MAX_REQ_DAY:
            w = max(w, day[0]["t"] + 86400 - now)
        hs = list(dict.fromkeys(r["s"] for r in hour))
        if symbol not in hs and len(hs) >= MAX_SYMBOLS_HOUR:
            first = {}
            for r in hour:
                first.setdefault(r["s"], r["t"])
            w = max(w, min(first.values()) + 3600 - now)
        return max(w, 0.0)

    def record(self, symbol: str) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps({"t": time.time(), "s": symbol}) + "\n")


def _valid_body(b: bytes) -> bool:
    try:
        return isinstance(json.loads(b), list)
    except ValueError:
        return False


def eod_params(start=START, end=None) -> dict:
    end = end or "2026-10-29"           # fixed end -> stable cache key (the patch targets the past)
    return {"startDate": str(pd.Timestamp(start).date()), "endDate": str(pd.Timestamp(end).date()), "format": "json"}


def cached_eod(symbol: str, start=START, end=None) -> Optional[list]:
    """Cached response rows ([] when Tiingo answered 404), None when not fetched yet."""
    url, params = EOD_URL.format(t=tiingo_symbol(symbol)), eod_params(start, end)
    try:
        b = client().get(url, params=params, cache_only=True, allow_404=True, validate=_valid_body)
    except Exception:
        return None
    return json.loads(b) if b else []


def fetch_eod_many(symbols: Iterable[str], start=START, end=None, budget: Optional[Budget] = None,
                   max_wait_s: float = 4 * 3600, log_fn=log.info) -> dict:
    """Download (into the raw HTTP cache only -- nothing is stored before `ingest`) the EOD history of
    every symbol not yet cached, within the free-tier budget; sleeps while the hourly window is full."""
    if not available():
        return skipped("tiingo")
    budget = budget or Budget()
    out = {"cached": [], "fetched": [], "missing": [], "deferred": []}
    for s in dict.fromkeys(str(x).upper() for x in symbols):
        if cached_eod(s, start, end) is not None:
            out["cached"].append(s)
            continue
        w = budget.wait_s(s)
        if w == float("inf") or w > max_wait_s:
            out["deferred"].append(s)
            continue
        if w > 0:
            log_fn(f"tiingo budget: waiting {w / 60:.0f} min before {s}")
            time.sleep(w + 1)
        budget.record(s)
        try:
            b = client().get(EOD_URL.format(t=tiingo_symbol(s)), params=eod_params(start, end), headers=_headers(),
                             allow_404=True, validate=_valid_body, timeout=120)
        except Exception as e:                      # never log the request headers
            out["deferred"].append(s)
            log_fn(f"tiingo {s}: {type(e).__name__}")
            continue
        (out["fetched"] if b and json.loads(b) else out["missing"]).append(s)
    return out


# ------------------------------------------------------------------------------------- lines
def supported_lines(store: Optional[ParquetStore] = None, refresh: bool = False) -> pd.DataFrame:
    """Tiingo's supported-ticker list (USD stocks): one row per price line -> reference/tiingo/lines."""
    store = store or default_store()
    if store.exists("reference", SOURCE, "lines") and not refresh:
        return store.read("reference", SOURCE, "lines")
    import io
    import zipfile
    b = client().get(SUPPORTED_URL, ttl=7 * 86400)
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), keep_default_na=False)
    df = df[(df["priceCurrency"] == "USD") & (df["assetType"].isin(["Stock", "ETF"]))].copy()
    df["ticker"] = df["ticker"].str.upper().str.replace(".", "-", regex=False)
    for c in ("startDate", "endDate"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df.dropna(subset=["startDate", "endDate"]).reset_index(drop=True)
    store.write("reference", SOURCE, "lines", df, key=["ticker", "exchange", "startDate"], replace=True,
                extra={"origin": SUPPORTED_URL})
    return df


# ------------------------------------------------------------------------------------- validation
def validate(portal, symbols: Iterable[str], start=START, end=None) -> dict:
    """Source-level check on cached responses: Tiingo raw vs every overlapping trusted as-traded
    source of the same ticker (Yahoo raw for survivors; Alpaca SIP / DoltHub for names that died
    2016+, restricted to dates the ticker denotes the same issuer in both), and Tiingo splits vs
    Yahoo splits. Accepted when >= 95 % of comparisons pass and every split matches."""
    from .alpaca_eod import compare, split_check
    idx = portal.identity_index()
    rows, splits = [], []
    for s in symbols:
        r = cached_eod(s, start, end)
        if not r:
            continue
        t_bars, t_acts = parse_eod(r)
        if not len(t_bars):
            continue
        td = frames.to_naive_date(t_bars.index)
        own = int(idx.label_dates(s, pd.DatetimeIndex([td[-1]]))[0]) or (idx.cik_now(s) or 0)
        for ref in ("yahoo", "alpaca_eod", "dolthub"):
            if not portal.store.exists("bars_1d", ref, s):
                continue
            b = portal._read("bars_1d", ref, s)
            if not len(b) or not own:
                continue
            # compare only dates on which the ticker denoted the Tiingo line's owner (a reused or
            # reorganized ticker -- CB = old Chubb before 2016, DD, CHK 2021 -- is not a price error)
            bd = frames.to_naive_date(b.index)
            b = b[idx.label_dates(s, bd) == own]
            c = compare(t_bars, b)
            if c["n"] < 20:
                continue
            # junction window: the first JUNCTION sessions of trusted coverage, right after the gap
            # Tiingo fills -> per-symbol gate, and the last date Tiingo rows are stored for
            common = frames.to_naive_date(b.index)
            common = common[common.isin(td)][:JUNCTION]
            j = compare(t_bars, b[frames.to_naive_date(b.index).isin(common)])
            rows.append({"symbol": s, "ref": ref, **c, "junction_ok": j["ok"], "junction_n": j["n"],
                         "trusted_from": common[0], "checked_to": common[-1]})
        if portal.store.exists("actions", "yahoo", s) and len(t_acts):
            ya = portal._read("actions", "yahoo", s)
            ysp = ya[ya["type"] == "SPLIT"] if len(ya) else ya
            tsp = t_acts[t_acts["type"] == "SPLIT"]
            yr = {pd.Timestamp(a).normalize(): float(b) for a, b in zip(ysp["ex_date"], ysp["ratio"])} if len(ysp) else {}
            for x in split_check(t_bars, tsp):
                ex = pd.Timestamp(x["ex_date"])
                if not own or int(idx.label_dates(s, pd.DatetimeIndex([ex]))[0]) != own:
                    continue                      # another issuer's split under a reused ticker (CB 2006)
                # consistent = the same split (date, ratio within 1 %) in Yahoo's list; a price-gap
                # mismatch alone is tolerated when Yahoo agrees (composite events: DD 2019-06-03
                # 1:3 reverse split + Corteva spin-off)
                yv = yr.get(ex)
                x["in_yahoo"] = yv is not None and abs(np.log(yv / x["ratio"])) < 0.01
                # composite event: Yahoo folds a same-day spin-off into its split ratio (DD 2019-06-03:
                # Yahoo 0.4725, Tiingo 1:3 reverse split; the open/close gap x ratio = 0.72 = the value
                # that left with Corteva) -> consistent when the extra drop is a distribution (0.5 .. 1)
                if not x["in_yahoo"] and yv is not None and 0.5 < x["gap"] * x["ratio"] < 1.0:
                    x["in_yahoo"], x["composite"] = True, True
                x["ok"] = bool(x["in_yahoo"] or x["ok"])
                splits.append({"symbol": s, **x})
    df = pd.DataFrame(rows)
    sp = pd.DataFrame(splits)
    share = float(df["ok"].mean()) if len(df) else 0.0
    rep = {"comparisons": int(len(df)), "symbols": int(df["symbol"].nunique()) if len(df) else 0,
           "pass_share": round(share, 4),
           "close_med_median": float(df["close_med"].median()) if len(df) else np.nan,
           "vol_med_median": float(df["vol_med"].median()) if len(df) else np.nan,
           "splits_checked": int(len(sp)), "splits_ok": int((sp["ok"] & sp["in_yahoo"]).sum()) if len(sp) else 0,
           "splits_in_yahoo": int(sp["in_yahoo"].sum()) if len(sp) else 0,
           "junction_pass_share": round(float(df["junction_ok"].mean()), 4) if len(df) else 0.0,
           "failed": df.loc[~df["ok"], ["symbol", "ref", "close_med", "close_p99", "vol_med", "junction_ok"]]
           .to_dict("records") if len(df) else []}
    rep["accepted"] = bool(len(df) >= 20 and share >= 0.95 and (not len(sp) or bool((sp["ok"] & sp["in_yahoo"]).all())))
    rep["detail"], rep["split_detail"] = df, sp
    return rep


def ingest(portal, symbols: Iterable[str], validation: dict, start=START, end=None, log_fn=log.info) -> dict:
    """Store cached Tiingo histories (bars + actions) after source-level validation. A symbol whose
    overlap with a trusted source disagrees (validation detail) is not stored."""
    if not validation.get("accepted"):
        return {"skipped": "tiingo source validation failed"}
    bad, upto = set(), {}
    det = validation.get("detail")
    if det is not None and len(det):
        bad = set(det.loc[~det["junction_ok"].astype(bool), "symbol"])
        # Tiingo is a BACKFILL: rows are kept up to the end of the validated junction window, the
        # trusted sources serve everything after it
        upto = det.groupby("symbol")["checked_to"].max().to_dict()
    lines = supported_lines(portal.store)
    out = {"stored": [], "empty": [], "rejected": sorted(bad), "not_cached": []}
    for s in dict.fromkeys(str(x).upper() for x in symbols):
        if s in bad:
            continue
        r = cached_eod(s, start, end)
        if r is None:
            out["not_cached"].append(s)
            continue
        bars, acts = parse_eod(r)
        if s in upto:
            keep = frames.to_naive_date(bars.index) <= pd.Timestamp(upto[s])
            bars = bars[keep]
            if len(acts):
                acts = acts[pd.to_datetime(acts["ex_date"]) <= pd.Timestamp(upto[s])]
        if not len(bars):
            out["empty"].append(s)
            continue
        ln = lines[lines["ticker"] == s]
        end_d = frames.to_naive_date(bars.index)[-1]
        if s in upto:
            # truncated backfill: the line's owner is decided at the line's TRUE end (identity layer)
            end_d = frames.to_naive_date(parse_eod(r)[0].index)[-1]
        portal.store.write("bars_1d", SOURCE, s, bars, fidelity=int(Fidelity.T1_BAR), replace=True,
                           extra={"origin": "tiingo /tiingo/daily/<t>/prices (raw)", "raw": True,
                                  "line_end": str(end_d.date()), "n_lines": int(len(ln)),
                                  "validation_pass_share": validation.get("pass_share")})
        if len(acts):
            portal.store.write("actions", SOURCE, s, acts.assign(symbol=s), key=["type", "ex_date"], replace=True,
                               fidelity=int(Fidelity.T2_PROXY), extra={"origin": "tiingo divCash/splitFactor"})
        out["stored"].append(s)
    return out


def actions(portal, symbol: str) -> pd.DataFrame:
    if not portal.store.exists("actions", SOURCE, symbol):
        return pd.DataFrame()
    a = portal._read("actions", SOURCE, symbol)
    return a if a is not None else pd.DataFrame()


# ------------------------------------------------------------------------------------- legacy
def fetch_daily(symbol: str, start=START, end=None) -> dict:
    """Single-symbol download + store WITHOUT validation (kept for API compatibility; prefer
    fetch_eod_many -> validate -> ingest)."""
    if not available():
        return skipped("tiingo")
    fetch_eod_many([symbol], start, end)
    r = cached_eod(symbol, start, end) or []
    bars, acts = parse_eod(r)
    if len(bars):
        default_store().write("bars_1d", SOURCE, symbol, bars, fidelity=int(Fidelity.T1_BAR), replace=True,
                              extra={"verified": False})
    return {"symbol": symbol, "rows": len(bars), "actions": len(acts)}


def fetch_iex_intraday(symbol: str, start, end, freq: str = "1m") -> dict:
    """IEX-venue intraday (UNVERIFIED; single venue -> T2_PROXY)."""
    if not available():
        return skipped("tiingo")
    b = client().get(f"https://api.tiingo.com/iex/{tiingo_symbol(symbol)}/prices",
                     params={"startDate": str(pd.Timestamp(start).date()), "endDate": str(pd.Timestamp(end).date()),
                             "resampleFreq": freq.replace("m", "min"), "columns": "open,high,low,close,volume"},
                     headers=_headers())
    df = pd.DataFrame(json.loads(b))
    if df.empty:
        return {"symbol": symbol, "rows": 0}
    df.index = pd.DatetimeIndex(pd.to_datetime(df["date"], utc=True), name="start")
    n = store_intraday(SOURCE, symbol, freq, df, Fidelity.T2_PROXY, extra={"venue": "IEX only"})
    return {"symbol": symbol, "rows": n}


def needed_symbols(portal, start="2011-01-03", end="2015-12-31") -> list[str]:
    """Tickers worth a Tiingo request: for every S&P 500 member without verified bars on some month
    start in [start, end], the member ticker and its issuer's last lineage ticker -- kept only if the
    Tiingo line the API serves for it (the latest line of the ticker) starts before `end` (Tiingo
    has many delisted names only from 2016-01-04, and a reused ticker serves its current holder)."""
    idx = portal.identity_index()
    lines = supported_lines(portal.store)
    want: list[str] = []
    for d in pd.date_range(start, end, freq="MS"):
        a = pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=21)
        kept = set(portal.universe("SP500", a, require_bars=True))
        for k in portal.universe("SP500", a):
            if k in kept:
                continue
            t, c = portal._split_key(k)
            if not c:
                cc, _, _ = portal._identity_by_date(t, [pd.Timestamp(d)])
                c = int(cc[0]) or None
            zl = idx.zl_tickers_of(int(c)) if c else []
            for x in dict.fromkeys([t] + zl[-1:]):
                if x not in want:
                    want.append(x)
    out = []
    for x in want:
        ln = lines[lines["ticker"] == x].sort_values("endDate")
        if len(ln) and ln["startDate"].iloc[-1] < pd.Timestamp(end):
            out.append(x)
    return out
