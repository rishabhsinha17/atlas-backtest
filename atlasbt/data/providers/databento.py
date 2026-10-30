"""Databento historical API: OPRA option quotes (LIVE-VERIFIED 2026-10-30) + legacy equity helpers.

Env: DATABENTO_API_KEY (HTTP basic auth: key as username, empty password). Read from the
environment only; never logged, cached, written to metadata or committed.

Budget discipline (new accounts run on free credits):
* EVERY data request is priced first with ``metadata.get_cost`` and appended to the ledger
  ``<cache>/raw/databento_ledger.csv`` (timestamp, dataset, schema, symbols, window, cost, status)
  BEFORE the download; a request that would push the cumulative spend above
  ``DATABENTO_BUDGET_USD`` (default $90) is refused (``BudgetExceeded``).
* Raw responses are cached (gzip CSV under ``<cache>/raw/hist.databento.com/``) keyed by the
  request, so a re-run never pays twice.

Endpoints (https://hist.databento.com/v0/, POST form):
  metadata.get_cost       dataset, symbols, schema, start, end, stype_in -> USD (float)
  timeseries.get_range    ... encoding=csv, pretty_px, pretty_ts, map_symbols -> CSV

OPRA (dataset OPRA.PILLAR, history from 2023-03-28):
  * symbols: ``stype_in=parent`` with ``<ROOT>.OPT`` (whole chain of a root, e.g. SPXW.OPT, SPY.OPT)
    or ``stype_in=raw_symbol`` with OSI symbols ``ROOT(6, space padded)YYMMDD{C|P}STRIKE*1000(8)``,
    e.g. ``SPXW  240102C04700000``. The date is the OPRA expiration date (for AM-settled SPX
    monthlies the SETTLEMENT Friday; converted to the internal last-trading-day convention by
    ``option_meta.last_trading_day`` exactly like DoltHub / historicaloptiondata rows).
  * schema ``cbbo-1m``: consolidated (across all OPRA exchanges) best bid/offer SAMPLED on a 1-minute
    grid. Verified semantics are recorded in docs/DATA.md §4.15c (ts_recv = sample instant on the
    minute grid; ts_event = time of the last book change at or before it).
Stored quotes: kind ``option_quotes``, source ``databento``, one parquet file per (underlying,
session) ``option_quotes/databento/<UND>/<YYYY-MM-DD>.parquet`` (rows: ts, symbol, bid, ask, sizes,
ts_event); served by ``Portal.option_chain`` / ``option_quote`` / ``option_quotes`` as T0 quote
snapshots (``portal.OPTION_SOURCE_FIDELITY['databento']``), ``known_at = ts``.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import requests

from atlasbt.core.types import Fidelity

from .. import config
from ._keyed import env, skipped, store_intraday, store_quotes

SOURCE = "databento"
BASE = "https://hist.databento.com/v0/"
OPRA = "OPRA.PILLAR"
OPRA_START = pd.Timestamp("2023-03-28", tz="UTC")
KIND = "option_quotes"
RAW_CACHE_MAX_BYTES = 3_000_000
LEDGER_COLS = ["timestamp", "dataset", "schema", "stype_in", "n_symbols", "symbols", "start", "end", "cost_usd",
               "status", "tag", "request_key"]


_LAST_COST = [0.0]          # cost of the last request this process paid for (0 on a raw-cache hit)


class BudgetExceeded(RuntimeError):
    pass


class NeedsUser(RuntimeError):
    """The account requires something from the user (payment details, credits exhausted, auth)."""


def available() -> bool:
    return env("DATABENTO_API_KEY") is not None


def budget_usd() -> float:
    return float(os.environ.get("DATABENTO_BUDGET_USD", "90"))


def ledger_path() -> Path:
    return config.RAW_ROOT / "databento_ledger.csv"


def raw_dir() -> Path:
    return config.RAW_ROOT / "hist.databento.com"


# ============================================================================ HTTP
def _post(endpoint: str, data: dict, timeout: int = 600) -> requests.Response:
    if os.environ.get("ATLASBT_OFFLINE") == "1":
        raise RuntimeError("ATLASBT_OFFLINE=1: Databento requests disabled")
    key = env("DATABENTO_API_KEY")
    if key is None:
        raise RuntimeError("DATABENTO_API_KEY not set")
    last = None
    for attempt in range(5):
        try:
            r = requests.post(BASE + endpoint, data=data, auth=(key["DATABENTO_API_KEY"], ""), timeout=timeout,
                              headers={"User-Agent": config.USER_AGENT})
        except requests.RequestException as e:          # network: back off and retry
            last = e
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (429,) or r.status_code >= 500:
            last = RuntimeError(f"HTTP {r.status_code}")
            time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
            continue
        if r.status_code in (401, 402, 403):
            # never echo the key; the body is Databento's error message
            raise NeedsUser(f"Databento {endpoint}: HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code >= 400:
            raise RuntimeError(f"Databento {endpoint}: HTTP {r.status_code}: {r.text[:500]}")
        return r
    raise RuntimeError(f"Databento {endpoint}: retries exhausted ({last})")


def _iso(t) -> str:
    t = pd.Timestamp(t)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _params(dataset, schema, symbols, start, end, stype_in) -> dict:
    syms = symbols if isinstance(symbols, str) else ",".join(symbols)
    return {"dataset": dataset, "schema": schema, "symbols": syms, "start": _iso(start), "end": _iso(end),
            "stype_in": stype_in}


def get_cost(dataset: str, schema: str, symbols, start, end, stype_in: str = "raw_symbol") -> float:
    r = _post("metadata.get_cost", _params(dataset, schema, symbols, start, end, stype_in), timeout=120)
    return float(json.loads(r.text))


# ============================================================================ ledger
def read_ledger() -> pd.DataFrame:
    p = ledger_path()
    if not p.exists():
        return pd.DataFrame(columns=LEDGER_COLS)
    return pd.read_csv(p)


def spent_usd(ledger: Optional[pd.DataFrame] = None) -> float:
    """Cumulative spend: every row that reached the download stage ('requested' rows; a request that
    failed afterwards is still counted -- conservative)."""
    lg = read_ledger() if ledger is None else ledger
    if lg.empty:
        return 0.0
    return float(pd.to_numeric(lg.loc[lg["status"] == "requested", "cost_usd"], errors="coerce").fillna(0).sum())


def _ledger_append(row: dict) -> None:
    p = ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with open(p, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_COLS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LEDGER_COLS})


def _request_key(p: dict) -> str:
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:24]


def _sym_summary(symbols) -> tuple[int, str]:
    if isinstance(symbols, str):
        symbols = symbols.split(",")
    symbols = list(symbols)
    s = ";".join(symbols[:3]) + (f";...(+{len(symbols) - 3})" if len(symbols) > 3 else "")
    return len(symbols), s


def get_range_csv(dataset: str, schema: str, symbols, start, end, stype_in: str = "raw_symbol",
                  tag: str = "", max_cost_usd: Optional[float] = None) -> bytes:
    """Budget-guarded, cached ``timeseries.get_range`` (CSV, pretty px/ts, symbols mapped).
    Order: raw cache hit -> no charge; else get_cost -> refuse if spent + cost > budget (or > max_cost_usd)
    -> ledger row 'requested' -> download -> raw cache."""
    p = _params(dataset, schema, symbols, start, end, stype_in)
    p.update({"encoding": "csv", "pretty_px": "true", "pretty_ts": "true", "map_symbols": "true",
              "compression": "none"})
    key = _request_key(p)
    cache = raw_dir() / dataset / schema / f"{key}.csv.gz"
    if cache.exists():
        _LAST_COST[0] = 0.0
        with gzip.open(cache, "rb") as fh:
            return fh.read()
    n, ssum = _sym_summary(symbols)
    cost = get_cost(dataset, schema, symbols, start, end, stype_in)
    spent = spent_usd()
    base = {"timestamp": pd.Timestamp.now(tz="UTC").isoformat(), "dataset": dataset, "schema": schema,
            "stype_in": stype_in, "n_symbols": n, "symbols": ssum, "start": p["start"], "end": p["end"],
            "cost_usd": round(cost, 6), "tag": tag, "request_key": key}
    if spent + cost > budget_usd() or (max_cost_usd is not None and cost > max_cost_usd):
        _ledger_append({**base, "status": "refused"})
        raise BudgetExceeded(f"cost ${cost:.4f} (spent ${spent:.2f}, budget ${budget_usd():.2f}, "
                             f"per-request cap {max_cost_usd})")
    _ledger_append({**base, "status": "requested"})
    _LAST_COST[0] = cost
    try:
        r = _post("timeseries.get_range", p)
    except Exception as e:
        _ledger_append({**base, "cost_usd": 0.0, "status": f"failed:{type(e).__name__}"})
        raise
    body = r.content
    if len(body) <= RAW_CACHE_MAX_BYTES:
        # small responses are kept (re-runs never pay twice); large ones are not (disk budget): the
        # parquet store + the progress ledger (databento_progress.jsonl) prevent refetching them
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        with gzip.open(tmp, "wb") as fh:
            fh.write(body)
        tmp.replace(cache)
    return body


# ============================================================================ legacy equity helpers
def cost(dataset: str, symbol: str, schema: str, start, end) -> float:
    return get_cost(dataset, schema, symbol, start, end, "raw_symbol")


def parse_ohlcv_csv(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content))
    df.index = pd.DatetimeIndex(pd.to_datetime(df["ts_event"], utc=True), name="start")
    return df[["open", "high", "low", "close", "volume"]]


def parse_bbo_csv(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content))
    ts = df["ts_recv"] if "ts_recv" in df else df["ts_event"]
    return pd.DataFrame({"bid": df["bid_px_00"], "ask": df["ask_px_00"], "bid_size": df["bid_sz_00"],
                         "ask_size": df["ask_sz_00"]}, index=pd.DatetimeIndex(pd.to_datetime(ts, utc=True), name="ts"))


def fetch(symbol: str, start, end, schema: str = "ohlcv-1m", dataset: str = "XNAS.ITCH",
          max_cost_usd: float = float(os.environ.get("DATABENTO_MAX_COST", "1.0"))) -> dict:
    """Equity bars/quotes (single venue XNAS.ITCH by default -> bars T1, quotes T2). Budget-guarded."""
    if not available():
        return skipped("databento")
    try:
        content = get_range_csv(dataset, schema, symbol, start, end, "raw_symbol", tag=f"equity:{symbol}",
                                max_cost_usd=max_cost_usd)
    except BudgetExceeded as e:
        return {"symbol": symbol, "refused": str(e)}
    if schema.startswith("ohlcv"):
        freq = {"ohlcv-1m": "1m", "ohlcv-1h": "60m"}[schema]
        n = store_intraday(SOURCE, symbol, freq, parse_ohlcv_csv(content), Fidelity.T1_BAR, extra={"dataset": dataset})
    else:
        n = store_quotes(SOURCE, symbol, parse_bbo_csv(content), Fidelity.T2_PROXY, extra={"dataset": dataset})
    return {"symbol": symbol, "rows": n}


# ============================================================================ OPRA symbology
#: index option roots: exercise EUROPEAN, cash-settled (AM: SOQ at the next session open; PM: close)
INDEX_ROOTS = {"SPX", "SPXW", "XSP", "NDX", "NDXP", "RUT", "RUTW", "DJX", "OEX", "XEO", "VIX"}
PM_INDEX_ROOTS = {"SPXW", "XSP", "NDXP", "RUTW"}
UNDERLYING_OF_ROOT = {"SPXW": "SPX", "NDXP": "NDX", "RUTW": "RUT"}
MAX_SYMBOLS_PER_REQUEST = 2000


def opra_root(ticker: str) -> str:
    """Universe symbol -> OPRA/OSI root of its standard options (BRK-B -> BRKB, 'XYZ~CIK..' -> XYZ)."""
    return str(ticker).split("~")[0].upper().replace("-", "").replace(".", "").replace("/", "")


def underlying_of_root(root: str) -> str:
    root = root.strip().upper()
    return UNDERLYING_OF_ROOT.get(root, root)


def parse_osi(raw) -> pd.DataFrame:
    """OSI symbols 'ROOT(6)YYMMDD{C|P}KKKKKKKK' -> root, vendor_expiry (OPRA expiration date), right, strike."""
    s = pd.Series(raw, dtype=str).reset_index(drop=True)
    root = s.str[:6].str.strip()
    ymd = s.str[6:12]
    return pd.DataFrame({"raw_symbol": s, "root": root,
                         "vendor_expiry": pd.to_datetime("20" + ymd, format="%Y%m%d", errors="coerce"),
                         "right": s.str[12], "strike": pd.to_numeric(s.str[13:21], errors="coerce") / 1000.0})


_RAW_MEMO: dict = {}
_IMAP_COLS = ["raw_symbol", "root", "vendor_expiry", "right", "strike", "expiry", "settlement_date", "symbol",
              "underlying", "exercise", "settlement", "kind"]


def internal_map(raw_symbols) -> pd.DataFrame:
    """OSI (vendor) symbols -> internal convention (option_meta: expiry = LAST TRADING DAY, SPX AM
    monthlies listed under the settlement Friday move to the Thursday). Columns: raw_symbol, symbol,
    underlying, root, expiry, settlement_date, right, strike, exercise, settlement, kind. Memoized per symbol."""
    from ..option_meta import is_monthly, internal_symbols, last_trading_day
    want = pd.unique(pd.Series(list(raw_symbols), dtype=str))
    miss = [r for r in want if r not in _RAW_MEMO]
    if miss:
        u = parse_osi(miss)
        u = u[u["vendor_expiry"].notna() & u["strike"].notna() & u["right"].isin(["C", "P"])].reset_index(drop=True)
        if len(u):
            exp, settle = last_trading_day(u["vendor_expiry"], u["root"])
            u["expiry"], u["settlement_date"] = exp, settle
            u["symbol"] = internal_symbols(u["root"], exp, u["right"], u["strike"]).to_numpy()
            u["underlying"] = [underlying_of_root(r) for r in u["root"]]
            idx = u["root"].isin(list(INDEX_ROOTS))
            u["exercise"] = np.where(idx, "EUROPEAN", "AMERICAN")
            u["settlement"] = np.where(u["root"].isin(list(PM_INDEX_ROOTS)), "CASH_PM",
                                       np.where(idx, "CASH_AM", "PHYSICAL"))
            u["kind"] = np.where(is_monthly(u["vendor_expiry"]), "MONTHLY", "WEEKLY")
            if len(_RAW_MEMO) > 3_000_000:
                _RAW_MEMO.clear()
            for t in u[_IMAP_COLS].itertuples(index=False, name=None):
                _RAW_MEMO[t[0]] = t
        for r in miss:
            _RAW_MEMO.setdefault(r, None)
    vals = [_RAW_MEMO[r] for r in want if _RAW_MEMO.get(r) is not None]
    out = pd.DataFrame(vals, columns=_IMAP_COLS)
    for c in ("vendor_expiry", "expiry", "settlement_date"):
        out[c] = pd.to_datetime(out[c])
    return out


_LISTING_MEMO: dict = {}


def listing(parents: Sequence[str], day) -> pd.DataFrame:
    """Every OPRA contract listed on ``day`` for the given roots (free ``symbology.resolve``
    parent -> instrument_id; the result keys are the raw OSI symbols). Columns as parse_osi.
    Small single-root listings are cached on disk (``raw/hist.databento.com/symbology``)."""
    day = pd.Timestamp(day).normalize()
    parents = sorted({p.upper() for p in parents})
    key = (tuple(parents), day)
    if key in _LISTING_MEMO:
        return _LISTING_MEMO[key]
    cache = raw_dir() / "symbology" / f"{'_'.join(parents)[:80]}_{day:%Y%m%d}.parquet" if len(parents) <= 2 else None
    if cache is not None and cache.exists():
        out = pd.read_parquet(cache)
    else:
        syms = []
        for i in range(0, len(parents), 500):
            r = _post("symbology.resolve", {"dataset": OPRA, "symbols": ",".join(f"{p}.OPT" for p in parents[i:i + 500]),
                                            "stype_in": "parent", "stype_out": "instrument_id",
                                            "start_date": f"{day:%Y-%m-%d}",
                                            "end_date": f"{day + pd.Timedelta(days=1):%Y-%m-%d}"}, timeout=300)
            syms += list((r.json().get("result") or {}).keys())
        out = parse_osi(syms)
        out = out[out["vendor_expiry"].notna() & out["strike"].notna()].reset_index(drop=True)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(cache, index=False)
    if len(_LISTING_MEMO) > 64:
        _LISTING_MEMO.clear()
    _LISTING_MEMO[key] = out
    return out


# ============================================================================ quote store (per underlying-day)
QCOLS = ["ts", "symbol", "bid", "ask", "bid_size", "ask_size", "age_s", "chain"]


def _store_root():
    from ..store import default_store
    return default_store().root


def validation_dir(root: Optional[Path] = None) -> Path:
    """OPRA samples fetched only to validate the synthetic surface (not part of the served quote store)."""
    return Path(root or _store_root()) / "reference" / "databento_validation"


def day_dir(und: str, root: Optional[Path] = None) -> Path:
    from ..store import safe_symbol
    return Path(root or _store_root()) / KIND / SOURCE / safe_symbol(und)


def day_path(und: str, day, root: Optional[Path] = None) -> Path:
    return day_dir(und, root) / f"{pd.Timestamp(day):%Y-%m-%d}.parquet"


def parse_cbbo_csv(content: bytes) -> pd.DataFrame:
    """cbbo-1m / cbbo-1s / bbo CSV (pretty_px, pretty_ts, map_symbols) -> ts (= ts_recv, the sample
    instant on the interval grid), raw_symbol, bid, ask, bid_size, ask_size, ts_event (last book change)."""
    if not content or not content.strip():
        return pd.DataFrame(columns=["ts", "raw_symbol", "bid", "ask", "bid_size", "ask_size", "ts_event"])
    df = pd.read_csv(io.BytesIO(content), dtype={"symbol": str})
    tcol = "ts_recv" if "ts_recv" in df.columns else "ts_event"
    return pd.DataFrame({"ts": pd.to_datetime(df[tcol], utc=True), "raw_symbol": df["symbol"].astype(str),
                         "bid": pd.to_numeric(df["bid_px_00"], errors="coerce"),
                         "ask": pd.to_numeric(df["ask_px_00"], errors="coerce"),
                         "bid_size": pd.to_numeric(df["bid_sz_00"], errors="coerce").fillna(0).astype("int64"),
                         "ask_size": pd.to_numeric(df["ask_sz_00"], errors="coerce").fillna(0).astype("int64"),
                         "ts_event": pd.to_datetime(df["ts_event"], utc=True)})


def to_store_rows(raw: pd.DataFrame, chain: bool = True) -> pd.DataFrame:
    """Parsed vendor rows -> stored rows (internal symbols) with an `underlying` column."""
    if raw.empty:
        return pd.DataFrame(columns=QCOLS + ["underlying"])
    m = internal_map(raw["raw_symbol"].unique()).set_index("raw_symbol")
    raw = raw[raw["raw_symbol"].isin(m.index)]
    # zero/NaN ask = no offer; keep zero bids (real 'no bid' quotes), drop rows without any side
    raw = raw[(raw["ask"].fillna(0) > 0) | (raw["bid"].fillna(0) > 0)]
    out = pd.DataFrame({"ts": raw["ts"].to_numpy(), "symbol": m.loc[raw["raw_symbol"], "symbol"].to_numpy(),
                        "underlying": m.loc[raw["raw_symbol"], "underlying"].to_numpy(),
                        "bid": raw["bid"].to_numpy(float), "ask": raw["ask"].to_numpy(float),
                        "bid_size": raw["bid_size"].to_numpy("int64").clip(0, 2**31 - 1).astype("int32"),
                        "ask_size": raw["ask_size"].to_numpy("int64").clip(0, 2**31 - 1).astype("int32"),
                        "age_s": ((raw["ts"] - raw["ts_event"]).dt.total_seconds()).to_numpy("float32"),
                        "chain": bool(chain)})
    return out


def write_rows(rows: pd.DataFrame, root: Optional[Path] = None) -> int:
    """Upsert rows (with `underlying`) into one file per (underlying, ET session date of ts).
    Key (ts, symbol); a row fetched for a chain snapshot keeps chain=True if fetched twice."""
    from ..store import _locked
    if rows is None or rows.empty:
        return 0
    from ..frames import schedule
    n = 0
    sess_date = pd.DatetimeIndex(rows["ts"]).tz_convert("America/New_York").tz_localize(None).normalize()
    for (und, day), g in rows.groupby([rows["underlying"].to_numpy(), sess_date]):
        p = day_path(und, day, root)
        g = g[QCOLS]
        with _locked(p):
            if p.exists():
                old = pd.read_parquet(p)
                g = pd.concat([old, g], ignore_index=True)
                ch = g.groupby(["ts", "symbol"], sort=False)["chain"].transform("max")
                g = g.assign(chain=ch).drop_duplicates(["ts", "symbol"], keep="last")
            g = g.sort_values(["ts", "symbol"], kind="mergesort").reset_index(drop=True)
            tmp = p.with_suffix(".tmp%d" % os.getpid())
            g.to_parquet(tmp, index=False, compression="zstd")
            tmp.replace(p)
        n += len(g)
    return n


# ============================================================================ reader (Portal hooks)
#: a stored chain snapshot is served by Portal.option_chain only if it is at most this old at as_of
CHAIN_FRESH = pd.Timedelta(minutes=15)
#: quotes older than this at their sample instant (no book change since) are flagged stale in `age_s`
STALE_QUOTE_S = 600.0


class OpraReader:
    """Read side of the per-(underlying, session) OPRA quote files; memoizes day frames."""

    def __init__(self, root: Optional[Path] = None, max_days: int = 48):
        self.root = Path(root or _store_root())
        self.max_days = max_days
        self._days: dict = {}
        self._frames: dict = {}

    def has(self, und: str) -> bool:
        return len(self.days(und)) > 0

    def days(self, und: str) -> list:
        key = str(und).upper()
        if key not in self._days:
            d = day_dir(key, self.root)
            self._days[key] = sorted(pd.Timestamp(p.stem) for p in d.glob("*.parquet")) if d.is_dir() else []
        return self._days[key]

    def frame(self, und: str, day) -> pd.DataFrame:
        key = (str(und).upper(), pd.Timestamp(day).normalize())
        f = self._frames.get(key)
        if f is None:
            p = day_path(key[0], key[1], self.root)
            f = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=QCOLS)
            if len(f):
                f["ts"] = pd.to_datetime(f["ts"], utc=True)
                f = f.sort_values(["ts", "symbol"], kind="mergesort").reset_index(drop=True)
            f.attrs["ts_ns"] = f["ts"].astype("int64").to_numpy() if len(f) else np.array([], "int64")
            if len(self._frames) >= self.max_days:
                self._frames.pop(next(iter(self._frames)))
            self._frames[key] = f
        return f

    def _days_upto(self, und: str, as_of: pd.Timestamp, back_days: float) -> list:
        sess = as_of.tz_convert("America/New_York").tz_localize(None).normalize()
        lo = sess - pd.Timedelta(days=back_days + 1)
        return [d for d in reversed(self.days(und)) if lo <= d <= sess]

    def snapshot(self, und: str, as_of, chain_only: bool = True, max_age: pd.Timedelta = CHAIN_FRESH) -> pd.DataFrame:
        """Rows of the latest sample instant ts <= as_of (chain rows only by default), if as_of - ts <= max_age."""
        as_of = pd.Timestamp(as_of)
        for d in self._days_upto(und, as_of, max_age / pd.Timedelta(days=1)):
            f = self.frame(und, d)
            if not len(f):
                continue
            g = f[f["chain"]] if chain_only else f
            g = g[g["ts"] <= as_of]
            if len(g):
                ts = g["ts"].iloc[-1]
                if as_of - ts > max_age:
                    return g.iloc[0:0]
                return g[g["ts"] == ts]
        return pd.DataFrame(columns=QCOLS)

    def contract(self, und: str, symbol: str, start, end) -> pd.DataFrame:
        """All samples of ``symbol`` with start < ts <= end (ascending)."""
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        parts = []
        s0 = start.tz_convert("America/New_York").tz_localize(None).normalize()
        s1 = end.tz_convert("America/New_York").tz_localize(None).normalize()
        for d in self.days(und):
            if s0 <= d <= s1:
                f = self.frame(und, d)
                if len(f):
                    g = f[f["symbol"] == symbol]
                    g = g[(g["ts"] > start) & (g["ts"] <= end)]
                    if len(g):
                        parts.append(g)
        if not parts:
            return pd.DataFrame(columns=QCOLS)
        return pd.concat(parts, ignore_index=True)

    def latest(self, und: str, symbol: str, as_of, max_stale_days: float = 4.0) -> Optional[pd.Series]:
        as_of = pd.Timestamp(as_of)
        for d in self._days_upto(und, as_of, max_stale_days):
            f = self.frame(und, d)
            if not len(f):
                continue
            g = f[(f["symbol"] == symbol) & (f["ts"] <= as_of)]
            if len(g):
                r = g.iloc[-1]
                if (as_of - r["ts"]) / pd.Timedelta(days=1) > max_stale_days:
                    return None
                return r
        return None

    def covers(self, und: str, symbol: str, start, end) -> bool:
        """True when the contract has samples on any session of [start, end] (real T0 data exists that
        day, so a synthetic fallback must not be substituted for a window without samples)."""
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        s0 = start.tz_convert("America/New_York").tz_localize(None).normalize()
        s1 = end.tz_convert("America/New_York").tz_localize(None).normalize()
        for d in self.days(und):
            if s0 <= d <= s1:
                f = self.frame(und, d)
                if len(f) and (f["symbol"] == symbol).any():
                    return True
        return False


def parity_spot(rows: pd.DataFrame, n_strikes: int = 5) -> tuple[float, Optional[pd.Timestamp], int]:
    """Index level implied by put-call parity on the NEAREST expiry of a snapshot (European cash-settled
    index options; PM series preferred): F_K = K + (C_mid - P_mid) on the n strikes with the smallest
    |C_mid - P_mid| among quoted pairs (ask > 0; a zero bid counts, mid = ask/2); median. For a same-day (0DTE) expiry the carry over the
    remaining hours, (r - q) T < 3e-5 relative, is ignored, so spot = F. Returns (level, expiry, n)."""
    if rows is None or rows.empty:
        return float("nan"), None, 0
    x = rows
    if "expiry" not in x.columns or "settlement" not in x.columns:
        x = x.merge(internal_map_from_internal(x["symbol"].unique()), on="symbol", how="left")
    # a zero bid is a real quote for a (nearly) worthless option (0DTE wings): mid = ask / 2
    good = x[(x["bid"] >= 0) & (x["ask"] > 0) & (x["ask"] >= x["bid"])]
    if good.empty:
        return float("nan"), None, 0
    order = sorted(good[["expiry", "settlement"]].drop_duplicates().itertuples(index=False),
                   key=lambda t: (pd.Timestamp(t[0]), 0 if t[1] == "CASH_PM" else 1))
    for exp, st in order:
        g = good[(good["expiry"] == exp) & (good["settlement"] == st)]
        c = g[g["right"] == "C"].groupby("strike")[["bid", "ask"]].first()
        p = g[g["right"] == "P"].groupby("strike")[["bid", "ask"]].first()
        k = c.index.intersection(p.index)
        if len(k) < 3:
            continue
        cm = (c.loc[k, "bid"] + c.loc[k, "ask"]) / 2
        pm = (p.loc[k, "bid"] + p.loc[k, "ask"]) / 2
        d = (cm - pm)
        sel = d.abs().sort_values().index[:n_strikes]
        f = np.asarray(sel, float) + d.loc[sel].to_numpy(float)
        return float(np.median(f)), pd.Timestamp(exp), int(len(sel))
    return float("nan"), None, 0


def chain_frame(rows: pd.DataFrame, underlying: str, spot_fn=None) -> pd.DataFrame:
    """Stored rows of ONE sample instant -> canonical chain frame (frames.CHAIN_COLS + root,
    settlement_date, kind, underlying_price_source, bid_size, ask_size, age_s), fidelity T0_QUOTE.
    Vendor IV / greeks are not part of cbbo (NaN; strategies derive them from mids).
    underlying_price: index underlyings -> parity_spot of the same snapshot; others -> spot_fn(ts)
    (the Portal passes the last 1m bar close known at ts)."""
    from .. import frames
    if rows is None or rows.empty:
        return frames.empty_chain()
    m = internal_map_from_internal(rows["symbol"].unique())
    x = rows.merge(m, on="symbol", how="left")
    x["underlying"] = str(underlying).upper()
    x["mid"] = np.where((x["bid"] > 0) & (x["ask"] >= x["bid"]), (x["bid"] + x["ask"]) / 2,
                        np.where(x["ask"] > 0, x["ask"] / 2, np.nan))
    for c in ("iv", "delta", "gamma", "vega", "theta", "volume", "open_interest"):
        x[c] = np.nan
    x["multiplier"] = 100.0
    x["fidelity"] = int(Fidelity.T0_QUOTE)
    ts = pd.Timestamp(x["ts"].iloc[0])
    if str(underlying).upper() in INDEX_ROOTS:
        s, _e, n = parity_spot(x)
        x["underlying_price"] = s
        x["underlying_price_source"] = f"opra_parity_nearest_expiry(n={n})"
    else:
        s = float(spot_fn(ts)) if spot_fn is not None else float("nan")
        x["underlying_price"] = s
        x["underlying_price_source"] = "pit_1m_bar_close_at_ts"
    cols = frames.CHAIN_COLS + ["root", "settlement_date", "kind", "underlying_price_source", "bid_size",
                                "ask_size", "age_s"]
    out = x[cols].reset_index(drop=True)
    out.attrs.update({"source": SOURCE, "fidelity": int(Fidelity.T0_QUOTE), "synthetic": False, "snapshot_ts": ts,
                      "note": "OPRA consolidated BBO sampled on the 1-minute grid (Databento cbbo-1m); ts = sample "
                              "instant; quotes are executable NBBO at ts"})
    return out


_INTERNAL_MEMO: dict = {}


def internal_map_from_internal(symbols) -> pd.DataFrame:
    """Internal OCC symbols (date = last trading day) -> symbol, root, expiry, strike, right, exercise,
    settlement, settlement_date, kind."""
    from ..option_meta import is_monthly, settlement_for_internal
    syms = [s for s in pd.unique(pd.Series(symbols, dtype=str))]
    miss = [s for s in syms if s not in _INTERNAL_MEMO]
    if miss:
        u = parse_osi(miss)
        idx = u["root"].isin(list(INDEX_ROOTS))
        u["expiry"] = u["vendor_expiry"]
        sd = [settlement_for_internal(e, r) for e, r in zip(u["expiry"], u["root"])]
        u["settlement_date"] = pd.to_datetime(sd)
        u["exercise"] = np.where(idx, "EUROPEAN", "AMERICAN")
        u["settlement"] = np.where(u["root"].isin(list(PM_INDEX_ROOTS)), "CASH_PM", np.where(idx, "CASH_AM", "PHYSICAL"))
        am = u["root"].isin(list(INDEX_ROOTS - PM_INDEX_ROOTS))
        u["kind"] = np.where(is_monthly(np.where(am, u["settlement_date"], u["expiry"])), "MONTHLY", "WEEKLY")
        if len(_INTERNAL_MEMO) > 500_000:
            _INTERNAL_MEMO.clear()
        for r in u.itertuples(index=False):
            _INTERNAL_MEMO[r.raw_symbol] = (r.root, r.expiry, r.strike, r.right, r.exercise, r.settlement,
                                            r.settlement_date, r.kind)
    vals = [(s,) + _INTERNAL_MEMO[s] for s in syms]
    return pd.DataFrame(vals, columns=["symbol", "root", "expiry", "strike", "right", "exercise", "settlement",
                                       "settlement_date", "kind"])


def quote_rows_frame(rows: pd.DataFrame, underlying: str) -> pd.DataFrame:
    """Contract samples -> Portal._real_option_quotes frame (option_chain columns, index ts)."""
    if rows.empty:
        return rows
    m = internal_map_from_internal(rows["symbol"].unique())
    x = rows.merge(m, on="symbol", how="left")
    x["underlying"] = str(underlying).upper()
    x["mid"] = np.where((x["bid"] > 0) & (x["ask"] >= x["bid"]), (x["bid"] + x["ask"]) / 2, np.nan)
    for c in ("iv", "delta", "gamma", "vega", "theta", "volume", "open_interest", "underlying_price"):
        x[c] = np.nan
    x["multiplier"] = 100.0
    x["fidelity"] = int(Fidelity.T0_QUOTE)
    x["source"] = SOURCE
    x["underlying_price_source"] = None
    return x


# ============================================================================ fetch jobs (resumable, budget-guarded)
def progress_path() -> Path:
    return config.CACHE_ROOT / "logs" / "databento_progress.jsonl"


def done_keys() -> set:
    p = progress_path()
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text().splitlines():
        try:
            out.add(json.loads(line)["key"])
        except Exception:
            continue
    return out


def _mark_done(key: str, **info) -> None:
    p = progress_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps({"key": key, "at": pd.Timestamp.now(tz="UTC").isoformat(), **info}, default=str) + "\n")


class Job:
    """One resumable unit: list the contracts of ``parents`` on ``day`` (free), ``select`` the raw
    symbols, download every window (cbbo-1m), store rows (chain flag), run ``post`` (e.g. parity bars)."""

    def __init__(self, key: str, parents, day, select, windows, chain: bool = True, tag: str = "",
                 schema: str = "cbbo-1m", post=None, store: bool = True, chain_before=None,
                 needs_listing: bool = True):
        self.key, self.parents, self.day = key, sorted(set(parents)), pd.Timestamp(day).normalize()
        #: rows sampled after this instant are stored as valuation marks (chain=False)
        self.chain_before = chain_before
        self.needs_listing = needs_listing
        self.select, self.windows, self.chain, self.tag = select, list(windows), chain, tag or key.split("|")[0]
        self.schema, self.post, self.store = schema, post, store


def _chunks(seq, n):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def run_jobs(jobs: Iterable[Job], dry_run: bool = False, log=print, max_cost_per_request: float = 5.0,
             stop_after_usd: Optional[float] = None) -> dict:
    """Execute jobs in order. dry_run: price every request with get_cost only (nothing downloaded,
    nothing written to the ledger). Stops cleanly on BudgetExceeded / NeedsUser."""
    done = done_keys()
    tot = {"jobs": 0, "skipped_done": 0, "requests": 0, "cost_usd": 0.0, "rows": 0, "symbols": 0, "empty": 0,
           "stopped": None}
    start_spent = spent_usd()
    for job in jobs:
        if job.key in done or (not dry_run and job.key in done_keys()):     # another process may have done it
            tot["skipped_done"] += 1
            continue
        try:
            # jobs whose contracts are already known (marks) skip the day's listing: select(None)
            lst = listing(job.parents, job.day) if job.needs_listing else None
            syms = sorted(set(job.select(lst)))
        except NeedsUser as e:
            tot["stopped"] = f"NEEDS USER: {e}"
            break
        except Exception as e:
            log(f"[databento] {job.key}: listing/select failed: {type(e).__name__}: {e}")
            continue
        tot["jobs"] += 1
        tot["symbols"] += len(syms)
        if not syms:
            tot["empty"] += 1
            if not dry_run:
                _mark_done(job.key, symbols=0, rows=0, cost=0.0)
            continue
        rows_n, cost_n = 0, 0.0
        try:
            for (a, b) in job.windows:
                for ch in _chunks(syms, MAX_SYMBOLS_PER_REQUEST):
                    tot["requests"] += 1
                    if dry_run:
                        c = get_cost(OPRA, job.schema, ch, a, b, "raw_symbol")
                        cost_n += c
                        continue
                    body = get_range_csv(OPRA, job.schema, ch, a, b, "raw_symbol", tag=job.tag,
                                         max_cost_usd=max_cost_per_request)
                    cost_n += _LAST_COST[0]
                    raw = parse_cbbo_csv(body)
                    raw = raw[(raw["ts"] > pd.Timestamp(a)) & (raw["ts"] < pd.Timestamp(b))]
                    if not job.store:              # validation samples: kept apart, never served
                        vr = to_store_rows(raw, chain=False)
                        if len(vr):
                            vp = validation_dir() / f"{job.day:%Y-%m-%d}.parquet"
                            vp.parent.mkdir(parents=True, exist_ok=True)
                            if vp.exists():
                                vr = pd.concat([pd.read_parquet(vp), vr]).drop_duplicates(["ts", "symbol"], keep="last")
                            vr.to_parquet(vp, index=False)
                            rows_n += len(vr)
                    if job.store:
                        rows = to_store_rows(raw, chain=job.chain)
                        if job.chain_before is not None and len(rows):
                            rows.loc[rows["ts"] > pd.Timestamp(job.chain_before), "chain"] = False
                        rows_n += write_rows(rows)
                    if job.post is not None:
                        job.post(raw)
        except BudgetExceeded as e:
            tot["stopped"] = f"budget: {e}"
            log(f"[databento] STOP {job.key}: {e}")
            break
        except NeedsUser as e:
            tot["stopped"] = f"NEEDS USER: {e}"
            log(f"[databento] STOP {job.key}: {e}")
            break
        except Exception as e:
            log(f"[databento] {job.key} failed: {type(e).__name__}: {e}")
            continue
        tot["cost_usd"] += cost_n
        tot["rows"] += rows_n
        if not dry_run:
            _mark_done(job.key, symbols=len(syms), rows=rows_n, cost=round(cost_n, 6))
        if tot["jobs"] % 25 == 0:
            log(f"[databento] {job.key}: jobs {tot['jobs']} cost ${tot['cost_usd']:.3f} rows {tot['rows']} "
                f"spent ${spent_usd():.2f}")
        if stop_after_usd is not None and (tot["cost_usd"] if dry_run else spent_usd() - start_spent) > stop_after_usd:
            tot["stopped"] = f"stop_after_usd {stop_after_usd}"
            break
    tot["spent_total_usd"] = round(spent_usd(), 4)
    return tot


# ============================================================================ plan helpers
def _schedule():
    from ..frames import schedule
    return schedule()


def sessions(start, end) -> list:
    sch = _schedule()
    s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    return [d for d in sch.index if s <= d <= e]


def open_close(day) -> tuple[pd.Timestamp, pd.Timestamp]:
    r = _schedule().loc[pd.Timestamp(day).normalize()]
    o, c = pd.Timestamp(r["open"]), pd.Timestamp(r["close"])
    o = o.tz_localize("UTC") if o.tzinfo is None else o.tz_convert("UTC")
    c = c.tz_localize("UTC") if c.tzinfo is None else c.tz_convert("UTC")
    return o, c


def label_utc(day, label: str) -> pd.Timestamp:
    """ET wall-clock label on a session; afternoon labels (>= 12:00) shift with early closes
    (same rule as strategies.options_events_common.label_ts)."""
    d = pd.Timestamp(day).normalize()
    t = pd.Timestamp(f"{d:%Y-%m-%d} {label}").tz_localize("America/New_York").tz_convert("UTC")
    o, c = open_close(d)
    if int(label[:2]) >= 12:
        reg = pd.Timestamp(f"{d:%Y-%m-%d} 16:00").tz_localize("America/New_York").tz_convert("UTC")
        t = t - (reg - c)
    return t


def samples_window(first: pd.Timestamp, last: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Request window whose 1-minute samples (ts_recv on the minute grid, interval [ts-1m, ts)) are
    exactly first, first+1m, ..., last."""
    return pd.Timestamp(first) - pd.Timedelta(minutes=1), pd.Timestamp(last) + pd.Timedelta(minutes=1)


def _closing(close: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """'Closing' sample = the 1-minute sample ONE MINUTE BEFORE the official close (15:59:00 ET on regular days).
    The sample stamped exactly at the close bell is unreliable for non-expiring SPX/SPXW series (quotes are
    being re-set: a 30-DTE ATM SPXW call quoted 69.9/75.9 at 16:00:00 vs 75.2/75.8 at 15:59 and 71.2/71.6 at
    16:01 on 2024-01-31)."""
    t = pd.Timestamp(close) - pd.Timedelta(minutes=1)
    return t, t


def _dte(lst: pd.DataFrame, day) -> pd.Series:
    return (lst["vendor_expiry"] - pd.Timestamp(day).normalize()).dt.days


def _daily_close_table(portal, symbol: str, start, end) -> pd.Series:
    """Raw daily closes by session date (selection only -- never served to strategies)."""
    b = portal.bars(symbol, "1d", pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=10),
                    pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=2), as_of=pd.Timestamp.now(tz="UTC"),
                    adjusted=False)
    idx = pd.DatetimeIndex(b.index).tz_convert("America/New_York").tz_localize(None).normalize()
    return pd.DataFrame({"close": b["close"].to_numpy(), "high": b["high"].to_numpy(), "low": b["low"].to_numpy()},
                        index=idx)


def spx_reference(portal, start, end) -> pd.DataFrame:
    """Per session: SPX-scaled day high/low and previous VIX close, for 0DTE strike bands (fetch
    selection only). SPX level = SPY raw high/low x (SPX close / SPY close) of the previous session."""
    spy = _daily_close_table(portal, "SPY", start, end)
    spx = _daily_close_table(portal, "SPX", start, end)
    vix = _daily_close_table(portal, "VIX", start, end)
    ratio = (spx["close"] / spy["close"]).dropna()
    out = pd.DataFrame(index=spy.index)
    out["ratio"] = ratio.reindex(out.index).ffill().shift(1).bfill()
    out["hi"] = spy["high"] * out["ratio"]
    out["lo"] = spy["low"] * out["ratio"]
    out["close"] = spy["close"] * out["ratio"]
    out["vix_prev"] = vix["close"].reindex(out.index).ffill().shift(1).fillna(20.0)
    return out


# ============================================================================ plans
def plan_spx_0dte(start, end, ref: pd.DataFrame) -> list:
    """S161-S164: every SPXW contract expiring THAT session (0DTE, PM-settled) with a strike inside
    [low x (1-m), high x (1+m)], m = max(3%, 3.5 x VIX_prev/sqrt(252)) (covers 5-delta wings at the
    09:45-11:00 decisions), sampled every minute from the open to the close (cbbo-1m)."""
    jobs = []
    for d in sessions(max(pd.Timestamp(start), OPRA_START.tz_localize(None)), end):
        if d not in ref.index or not np.isfinite(ref.loc[d, "hi"]):
            continue
        r = ref.loc[d]
        m = max(0.03, 3.5 * float(r["vix_prev"]) / 100.0 / np.sqrt(252))
        lo, hi = float(r["lo"]) * (1 - m), float(r["hi"]) * (1 + m)
        o, c = open_close(d)

        def sel(lst, d=d, lo=lo, hi=hi):
            x = lst[(lst["root"] == "SPXW") & (lst["vendor_expiry"] == d) & lst["strike"].between(lo, hi)]
            return x["raw_symbol"].tolist()

        # samples in the last 20 minutes are marks only: a fresh 0DTE-only "chain" at the close would
        # otherwise be served to SPX end-of-day consumers (Portal serves chains <= CHAIN_FRESH old)
        jobs.append(Job(f"spx0dte|{d:%Y-%m-%d}", ["SPXW"], d, sel, [samples_window(o, c)], chain=True,
                        post=lambda raw, d=d: store_spx_parity_bars(raw, d),
                        chain_before=c - pd.Timedelta(minutes=20)))
    return jobs


def _s172_series(lst: pd.DataFrame, day, target: float) -> Optional[tuple]:
    """Replicates S172's expiry choice on a listing: European cash-settled SPX/SPXW series, expiry
    closest to the DTE target within [0.5, 1.5] x target (choose_expiry rules: distance, earlier, standard
    monthly first, root), then the standard series of that date (pick_series). -> (expiry, root)."""
    from atlasbt.options.chain import SelectionError, choose_expiry
    m = internal_map(lst.loc[lst["root"].isin(["SPX", "SPXW"]), "raw_symbol"])
    if m.empty:
        return None
    m["dte"] = (m["expiry"] - pd.Timestamp(day).normalize()).dt.days
    try:
        e = choose_expiry(m, target, int(0.5 * target), int(1.5 * target))
    except SelectionError:
        return None
    s = m[m["expiry"] == e][["root", "settlement", "kind"]].drop_duplicates()
    s = s.assign(_std=(~s["kind"].isin(["MONTHLY", "QUARTERLY", "LEAPS"])).astype(int))
    r = s.sort_values(["_std", "root", "settlement"], kind="mergesort").iloc[0]
    return pd.Timestamp(e), str(r["root"])


def plan_spx_eod(start, end, ref: pd.DataFrame, targets=(30, 90, 180), half_width: float = 75.0) -> list:
    """S172 (SPX box, monthly at the close of the last session of the month): closing-minute samples.
    * Month-end sessions and the session after (decision + fill at the next close): CHAIN snapshot of SPX
      (AM) + SPXW (PM) contracts with |K/S-1| <= 15% for DTE 1-60 and <= 6% for DTE 61-300 (wide enough
      that other SPX end-of-month consumers -- S129 put-write, S138 iron condor -- are not handed a
      truncated chain).
    * Every other session: valuation marks only (chain=False) of the contracts a box opened at one of the
      previous month-ends could hold: the series S172 selects for each DTE target (30/90/180) and strikes
      within S_month_end +- 75 points (width <= 100), until expiry."""
    days = sessions(max(pd.Timestamp(start), OPRA_START.tz_localize(None)), end)
    allsch = _schedule().index
    month_end = [d for d in days if allsch[allsch > d][0].month != d.month]
    after = {allsch[allsch > d][0] for d in month_end}
    jobs = []
    for d in days:
        if d not in ref.index or not np.isfinite(ref.loc[d, "close"]):
            continue
        S = float(ref.loc[d, "close"])
        o, c = open_close(d)
        if d in month_end or d in after:
            def sel(lst, d=d, S=S):
                x = lst[lst["root"].isin(["SPX", "SPXW"])]
                dte = _dte(x, d)
                mny = (x["strike"] / S - 1).abs()
                keep = ((dte >= 1) & (dte <= 60) & (mny <= 0.15)) | ((dte > 60) & (dte <= 300) & (mny <= 0.06))
                return x.loc[keep, "raw_symbol"].tolist()

            jobs.append(Job(f"spxeod|{d:%Y-%m-%d}", ["SPX", "SPXW"], d, sel, [samples_window(*_closing(c))], chain=True))
            continue
        active = [m for m in month_end if m < d <= m + pd.Timedelta(days=int(1.5 * max(targets)) + 5)]
        if not active:
            continue

        def sel_marks(_lst, d=d, active=active):
            out = []
            for me in active:
                Sme = float(ref.loc[me, "close"]) if me in ref.index else float("nan")
                if not np.isfinite(Sme):
                    continue
                lme = listing(["SPX", "SPXW"], me)          # disk-cached: no request on later days
                m_int = internal_map(lme.loc[lme["root"].isin(["SPX", "SPXW"]), "raw_symbol"])
                for t in targets:
                    ser = _s172_series(lme, me, t)
                    if ser is None or ser[0] < d:
                        continue
                    k = m_int[(m_int["expiry"] == ser[0]) & (m_int["root"] == ser[1])
                              & m_int["strike"].between(Sme - half_width, Sme + half_width)]
                    out += k["raw_symbol"].tolist()
            return out

        jobs.append(Job(f"spxeod|{d:%Y-%m-%d}", ["SPX", "SPXW"], d, sel_marks, [samples_window(*_closing(c))],
                        chain=False, needs_listing=False))
    return jobs


# ============================================================================ SPX minute levels from parity
PARITY_SOURCE = "opra_parity"


def parity_series(raw: pd.DataFrame) -> pd.DataFrame:
    """Vendor rows (parse_cbbo_csv) of an SPX/SPXW snapshot stream -> per sample instant: level
    (parity_spot), expiry used, n strikes, median quote age (s)."""
    if raw.empty:
        return pd.DataFrame(columns=["level", "n", "age_s"])
    rows = to_store_rows(raw)
    rows = rows.merge(internal_map_from_internal(rows["symbol"].unique()), on="symbol", how="left")
    out = []
    for ts, g in rows.groupby("ts", sort=True):
        lvl, _e, n = parity_spot(g)
        out.append((ts, lvl, n, float(np.nanmedian(g["age_s"]))))
    return pd.DataFrame(out, columns=["ts", "level", "n", "age_s"]).set_index("ts")


def parity_bars(levels: pd.DataFrame, day) -> pd.DataFrame:
    """1-minute 'bars' of the parity-implied index level: bar [t-1m, t) has close = level(t), open =
    level(t-1m) (first bar: the open sample if its quotes are fresh (median age < 120 s), else
    level(t)), high/low = max/min of the two samples (no intra-minute path; T2_PROXY), volume 0."""
    o, c = open_close(day)
    lv = levels[(levels.index >= o) & (levels.index <= c)]
    lv = lv[np.isfinite(lv["level"])]
    if len(lv) < 2:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    ends = lv.index[lv.index > o]
    prev = lv["level"].shift(1)
    if lv.index[0] == o and lv["age_s"].iloc[0] >= 120:
        prev.iloc[1] = lv["level"].iloc[1]
    rows = []
    for t in ends:
        cl = float(lv.at[t, "level"])
        op = float(prev.at[t]) if np.isfinite(prev.at[t]) else cl
        rows.append((t - pd.Timedelta(minutes=1), op, max(op, cl), min(op, cl), cl, 0.0))
    df = pd.DataFrame(rows, columns=["start", "open", "high", "low", "close", "volume"]).set_index("start")
    return df


def store_spx_parity_bars(raw: pd.DataFrame, day) -> int:
    """Post-hook of the spx0dte jobs: rebuild the day's parity bars from the stored samples."""
    return rebuild_spx_parity_bars([pd.Timestamp(day).normalize()])


def _bands_ok(lst: pd.DataFrame, root: str) -> pd.DataFrame:
    return lst[(lst["root"] == root)]


class _Closes:
    """Raw daily closes per symbol (selection only), memoized."""

    def __init__(self, portal, start, end):
        self.p, self.start, self.end, self.memo = portal, start, end, {}

    def get(self, sym: str) -> pd.Series:
        if sym not in self.memo:
            try:
                self.memo[sym] = _daily_close_table(self.p, sym, self.start, self.end)["close"]
            except Exception:
                self.memo[sym] = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        return self.memo[sym]

    def at(self, sym: str, day) -> float:
        s = self.get(sym)
        if not len(s) or not isinstance(s.index, pd.DatetimeIndex):
            return float("nan")
        s = s[s.index <= pd.Timestamp(day)]
        return float(s.iloc[-1]) if len(s) else float("nan")


def _sp500_members(portal, day, memo: dict) -> set:
    if day not in memo:
        o, _ = open_close(day)
        try:
            memo[day] = set(portal.universe("SP500", o))
        except Exception:
            memo[day] = set()
    return memo[day]


def earnings_schedule(portal, start, end) -> pd.DataFrame:
    """EARNINGS_SCHEDULE events (expected release, BMO/AMC) as the strategies read them."""
    now = pd.Timestamp.now(tz="UTC")
    ev = portal.events("EARNINGS_SCHEDULE", None, pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=30),
                       pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=5), as_of=now)
    rows = []
    for e in ev:
        t = str((e.payload or {}).get("timing", "UNKNOWN") or "UNKNOWN").upper()
        if t not in ("BMO", "AMC") or not e.symbol:
            continue
        rel = pd.Timestamp(e.effective_at if e.effective_at is not None else e.known_at)
        rel = rel.tz_localize("UTC") if rel.tzinfo is None else rel.tz_convert("UTC")
        rows.append((e.symbol, rel, t))
    df = pd.DataFrame(rows, columns=["symbol", "rel", "timing"]).drop_duplicates()
    return df.sort_values(["rel", "symbol"]).reset_index(drop=True)


def plan_earnings(portal, start, end) -> list:
    """S158/S159: for every scheduled BMO/AMC release of a PIT S&P 500 member, the entry session E (last
    session whose 15:30 is before the expected release) and the next session X: contracts of the
    standard root with expiry in [max(X, E+2d), E+11d] and strike within +-25% of E's close; samples
    E 15:00-15:31 (decisions 15:00/15:30, fills +1m) and X 09:30-09:46 (exits 09:35/09:45, fills +1m).
    One job per entry session (all names in the same requests)."""
    sch = _schedule().index
    s0 = max(pd.Timestamp(start), OPRA_START.tz_localize(None))
    ev = earnings_schedule(portal, s0, end)
    closes, memb = _Closes(portal, s0 - pd.Timedelta(days=10), end), {}
    by_e: dict = {}
    for r in ev.itertuples(index=False):
        d = r.rel.tz_convert("America/New_York").tz_localize(None).normalize()
        s = d if d in sch else sch[sch < d][-1]
        E = None
        for _ in range(5):
            if label_utc(s, "15:30") < r.rel:
                E = s
                break
            s = sch[sch < s][-1]
        if E is None or E < s0 or E > pd.Timestamp(end):
            continue
        if r.symbol not in _sp500_members(portal, E, memb):
            continue
        X = sch[sch > E][0]
        S = closes.at(r.symbol, E)
        if not np.isfinite(S):
            continue
        by_e.setdefault(E, {})[opra_root(r.symbol)] = (S, X)
    jobs = []
    for E, names in sorted(by_e.items()):
        X = sch[sch > E][0]
        oX, _ = open_close(X)
        wins = [samples_window(label_utc(E, "15:00"), label_utc(E, "15:30") + pd.Timedelta(minutes=1)),
                samples_window(oX, oX + pd.Timedelta(minutes=16))]

        def sel(lst, E=E, names=names):
            out = []
            for root, (S, X) in names.items():
                x = lst[lst["root"] == root]
                lo_e = max(X, E + pd.Timedelta(days=2))
                x = x[(x["vendor_expiry"] >= lo_e) & (x["vendor_expiry"] <= E + pd.Timedelta(days=11))
                      & x["strike"].between(0.75 * S, 1.25 * S)]
                out += x["raw_symbol"].tolist()
            return out

        jobs.append(Job(f"earn|{E:%Y-%m-%d}", list(names), E, sel, wins, chain=True))
    return jobs


def earnings_actual(portal, start, end) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    ev = portal.events("EARNINGS", None, pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=7),
                       pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1), as_of=now)
    rows = []
    for e in ev:
        if not e.symbol:
            continue
        t = pd.Timestamp(e.known_at)
        rows.append((e.symbol, t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")))
    return pd.DataFrame(rows, columns=["symbol", "rel"]).drop_duplicates().sort_values(["rel", "symbol"])


def _choose_30d(lst_root: pd.DataFrame, day) -> Optional[pd.Timestamp]:
    """Replicates S160's choose_series(x, 30, 20, 45) on the listing (internal expiries, kind)."""
    from atlasbt.options.chain import SelectionError, choose_expiry
    m = internal_map(lst_root["raw_symbol"])
    if m.empty:
        return None
    m["dte"] = (m["expiry"] - pd.Timestamp(day).normalize()).dt.days
    try:
        return choose_expiry(m, 30, 20, 45)
    except SelectionError:
        return None


def _s160_chosen_path() -> Path:
    return config.CACHE_ROOT / "logs" / "databento_s160_expiries.jsonl"


def _s160_chosen_save(R, chosen: dict) -> None:
    p = _s160_chosen_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps({"R": f"{pd.Timestamp(R):%Y-%m-%d}", "chosen": chosen}) + "\n")


_CHOSEN_MEMO: dict = {"mtime": None, "data": {}}


def _s160_chosen_load(R) -> Optional[dict]:
    p = _s160_chosen_path()
    if not p.exists():
        return None
    mt = p.stat().st_mtime
    if _CHOSEN_MEMO["mtime"] != mt:
        data = {}
        for line in p.read_text().splitlines():
            try:
                j = json.loads(line)
            except Exception:
                continue
            data[j.get("R")] = j["chosen"]
        _CHOSEN_MEMO.update(mtime=mt, data=data)
    return _CHOSEN_MEMO["data"].get(f"{pd.Timestamp(R):%Y-%m-%d}")


def plan_s160(portal, start, end) -> list:
    """S160: decision session R = first session whose 10:00 is after the ACTUAL release (PIT S&P 500
    member). Samples on R: 10:00-10:01 calls+puts DTE 14-60 (puts 0.8-1.05 S, calls 0.95-1.2 S: 30D ATM IV
    and parity forward); 10:00-close puts of the 30-DTE expiry the strategy selects (strike 0.7-1.02 S, i.e.
    the 0.10-0.30 delta puts of the grid;
    the entry fires at the first 5m close above the opening range). Then closing samples (marks,
    chain=False) of those puts every session until the expiry is <= 6 days away (exit at DTE <= 7)."""
    sch = _schedule().index
    s0 = max(pd.Timestamp(start), OPRA_START.tz_localize(None))
    ev = earnings_actual(portal, s0, end)
    closes, memb = _Closes(portal, s0 - pd.Timedelta(days=10), end), {}
    by_r: dict = {}
    for r in ev.itertuples(index=False):
        d = r.rel.tz_convert("America/New_York").tz_localize(None).normalize()
        s = d if d in sch else sch[sch > d][0]
        if label_utc(s, "10:00") <= r.rel:
            s = sch[sch > s][0]
        R = s
        if R < s0 or R > pd.Timestamp(end):
            continue
        if r.symbol not in _sp500_members(portal, R, memb):
            continue
        S = closes.at(r.symbol, sch[sch < R][-1])
        if np.isfinite(S):
            by_r.setdefault(R, {})[opra_root(r.symbol)] = S
    jobs, marks = [], {}
    for R, names in sorted(by_r.items()):
        o, c = open_close(R)
        t10 = label_utc(R, "10:00")

        def sel_snap(lst, names=names, R=R):
            out = []
            for root, S in names.items():
                x = lst[lst["root"] == root]
                dte = _dte(x, R)
                x = x[(dte >= 14) & (dte <= 60)]
                k = ((x["right"] == "P") & x["strike"].between(0.8 * S, 1.05 * S)) | \
                    ((x["right"] == "C") & x["strike"].between(0.95 * S, 1.2 * S))
                out += x.loc[k, "raw_symbol"].tolist()
            return out

        def sel_puts(lst, names=names, R=R):
            out, chosen = [], {}
            for root, S in names.items():
                x = lst[(lst["root"] == root) & (lst["right"] == "P")]
                e = _choose_30d(x, R)
                if e is None:
                    continue
                m = internal_map(x["raw_symbol"])
                raws = m.loc[(m["expiry"] == e) & m["strike"].between(0.7 * S, 1.02 * S), "raw_symbol"].tolist()
                chosen[root] = [str(e.date()), raws]
                out += raws
            _s160_chosen_save(R, chosen)
            return out

        jobs.append(Job(f"s160snap|{R:%Y-%m-%d}", list(names), R, sel_snap, [samples_window(t10, t10 + pd.Timedelta(minutes=1))]))
        # last 20 minutes stored as marks: no fresh single-expiry "chain" is served to end-of-day consumers
        jobs.append(Job(f"s160puts|{R:%Y-%m-%d}", list(names), R, sel_puts, [samples_window(t10, c)],
                        chain_before=c - pd.Timedelta(minutes=20)))
        marks[R] = names
    # closing marks: one job per session over all active events (selection re-derived from R's listing)
    last_data = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize() - pd.Timedelta(days=1)
    days = sessions(s0, min(pd.Timestamp(end) + pd.Timedelta(days=46), last_data))   # marks run past `end`
    for d in days:
        act = [(R, names) for R, names in marks.items() if R < d <= R + pd.Timedelta(days=45)]
        if not act:
            continue
        _o, c = open_close(d)

        def sel_marks(_lst, d=d, act=act):
            out = []
            for R, names in act:
                ch = _s160_chosen_load(R)
                if ch is None:                 # puts job of R not run yet: nothing is held
                    continue
                for root, S in names.items():
                    v = ch.get(root)
                    if v is None:
                        continue
                    if isinstance(v, str):     # early progress format (expiry only): derive from R's listing
                        e = pd.Timestamp(v)
                        x = listing([root], R)
                        m = internal_map(x.loc[(x["root"] == root) & (x["right"] == "P"), "raw_symbol"])
                        raws = m.loc[(m["expiry"] == e) & m["strike"].between(0.7 * S, 1.02 * S), "raw_symbol"].tolist()
                    else:
                        e, raws = pd.Timestamp(v[0]), v[1]
                    if (e - d).days >= 6:
                        out += list(raws)
            return out

        roots = sorted({r for _R, names in act for r in names})
        jobs.append(Job(f"s160close|{d:%Y-%m-%d}", roots, d, sel_marks, [samples_window(*_closing(c))], chain=False,
                        needs_listing=False))
    return jobs


def plan_s171(start, end, ref_closes: dict) -> list:
    """S171: ETF_CORE contracts with DTE 1-45 and |K/S-1| <= 5% (budget: a restricted contract universe, recorded
    in docs), samples close-36m .. close-33m (15:24-15:27:
    the Alpaca NBBO window close_m36 [15:24, 15:25] gives the synchronized stock quote; fill at +1m)."""
    jobs = []
    for d in sessions(max(pd.Timestamp(start), OPRA_START.tz_localize(None)), end):
        _o, c = open_close(d)
        S = {k: v.get(d, np.nan) for k, v in ref_closes.items()}
        if not all(np.isfinite(list(S.values()))):
            continue

        def sel(lst, d=d, S=S):
            out = []
            for root, s in S.items():
                x = lst[lst["root"] == root]
                dte = _dte(x, d)
                x = x[(dte >= 1) & (dte <= 45) & ((x["strike"] / s - 1).abs() <= 0.05)]
                out += x["raw_symbol"].tolist()
            return out

        jobs.append(Job(f"s171|{d:%Y-%m-%d}", list(S), d, sel,
                        [samples_window(c - pd.Timedelta(minutes=36), c - pd.Timedelta(minutes=33))]))
    return jobs


def plan_validation(days, ref_closes: dict) -> list:
    """Synthetic-surface check: SPY/QQQ/IWM closing-minute samples, DTE 5-70, |K/S-1| <= 12%; NOT stored in
    the served quote store (raw responses are cached; see options.validate_real)."""
    jobs = []
    for d in days:
        _o, c = open_close(d)
        S = {k: v.get(d, np.nan) for k, v in ref_closes.items()}

        def sel(lst, d=d, S=S):
            out = []
            for root, s in S.items():
                if not np.isfinite(s):
                    continue
                x = lst[lst["root"] == root]
                dte = _dte(x, d)
                x = x[(dte >= 5) & (dte <= 70) & ((x["strike"] / s - 1).abs() <= 0.12)]
                out += x["raw_symbol"].tolist()
            return out

        jobs.append(Job(f"valid|{d:%Y-%m-%d}", list(S), d, sel, [samples_window(*_closing(c))], store=False))
    return jobs


PLANS = ("spx0dte", "spxeod", "earnings", "s160", "s171", "validation")


def validation_days(start="2023-04-01", end="2026-10-29") -> list:
    """Third Wednesday-ish session of every second month (deterministic)."""
    out = []
    for m in pd.date_range(start, end, freq="2MS"):
        ss = sessions(m + pd.Timedelta(days=14), m + pd.Timedelta(days=24))
        if ss:
            out.append(ss[0])
    return out


def build_jobs(plans: Sequence[str], start, end, portal=None) -> list:
    from ..portal import Portal
    portal = portal or Portal()
    jobs = []
    ref = None
    for p in plans:
        if p in ("spx0dte", "spxeod") and ref is None:
            ref = spx_reference(portal, start, end)
        if p == "spx0dte":
            jobs += plan_spx_0dte(start, end, ref)
        elif p == "spxeod":
            jobs += plan_spx_eod(start, end, ref)
        elif p == "earnings":
            jobs += plan_earnings(portal, start, end)
        elif p == "s160":
            jobs += plan_s160(portal, start, end)
        elif p in ("s171", "validation"):
            syms = ["SPY", "QQQ", "IWM", "DIA"] if p == "s171" else ["SPY", "QQQ", "IWM"]
            closes = {s: _daily_close_table(portal, s, start, end)["close"].to_dict() for s in syms}
            jobs += plan_s171(start, end, closes) if p == "s171" else plan_validation(
                [d for d in validation_days(start, end)], closes)
    return jobs


def task(plans: Sequence[str] = ("spx0dte", "spxeod", "earnings", "s160"), start="2023-03-28", end=None,
         dry_run: bool = False, log=print, stop_after_usd: Optional[float] = None) -> dict:
    """Entry point of ``python -m atlasbt.data.fetch --only databento_options`` (env
    ATLASBT_DATABENTO_PLANS / _START / _END / _DRY_RUN select the plans and window)."""
    if not available():
        return skipped("databento")
    end = end or (pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize() - pd.Timedelta(days=1))
    out = {}
    for p in plans:
        jobs = build_jobs([p], start, end)
        log(f"[databento] plan {p}: {len(jobs)} jobs ({start} .. {pd.Timestamp(end).date()})")
        out[p] = run_jobs(jobs, dry_run=dry_run, log=log, stop_after_usd=stop_after_usd)
        log(f"[databento] plan {p}: {out[p]}")
        if out[p].get("stopped") and not str(out[p]["stopped"]).startswith("stop_after"):
            break
    return out


def rebuild_spx_parity_bars(days=None, root: Optional[Path] = None) -> int:
    """Recompute the opra_parity SPX 1m bars from the stored 0DTE samples (idempotent upsert)."""
    rd = OpraReader(root)
    n = 0
    for d in (days or rd.days("SPX")):
        f = rd.frame("SPX", d)
        if not len(f):
            continue
        f = f.merge(internal_map_from_internal(f["symbol"].unique()), on="symbol", how="left")
        f = f[(f["expiry"] == pd.Timestamp(d)) & (f["root"] == "SPXW")]
        out = []
        for ts, g in f.groupby("ts", sort=True):
            lvl, _e, k = parity_spot(g)
            out.append((ts, lvl, k, float(np.nanmedian(g["age_s"]))))
        lv = pd.DataFrame(out, columns=["ts", "level", "n", "age_s"]).set_index("ts")
        bars = parity_bars(lv, d)
        if len(bars):
            from ..store import ParquetStore
            n += store_intraday(PARITY_SOURCE, "SPX", "1m", bars, Fidelity.T2_PROXY,
                                store=ParquetStore(rd.root),
                                extra={"method": "SPXW 0DTE put-call parity (5 strikes nearest C=P, median); OHLC "
                                                 "from consecutive 1-minute samples (Databento OPRA cbbo-1m)",
                                       "verified": True})
    return n


def coverage(underlying: str = "*", root: Optional[Path] = None) -> dict:
    """OPRA quote coverage for planners: {'first', 'last', 'n_days', 'underlyings'} of the stored
    per-session files (underlying '*' = union over all underlyings)."""
    base = Path(root or _store_root()) / KIND / SOURCE
    if not base.is_dir():
        return {"first": None, "last": None, "n_days": 0, "underlyings": 0}
    from ..store import safe_symbol
    dirs = [d for d in base.iterdir() if d.is_dir()] if underlying in ("*", None, "") else \
        [base / safe_symbol(underlying)]
    days = set()
    n_und = 0
    for d in dirs:
        if d.is_dir():
            ds = {p.stem for p in d.glob("*.parquet")}
            n_und += bool(ds)
            days |= ds
    days = sorted(days)
    return {"first": days[0] if days else None, "last": days[-1] if days else None, "n_days": len(days),
            "underlyings": n_und}
