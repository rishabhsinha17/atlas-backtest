"""Alpaca SIP daily bars for DELISTED / renamed S&P 500 members (survivorship patch, 2016-01 ->).

Why: DoltHub `stocks.ohlcv` has no symbol that stopped trading before 2018 (measured on the full
HF mirror 2026-10-30: 18,893 symbols, none with a last date < 2018-01-01) and starts only in 2018
for most names delisted 2018-2020; Yahoo serves no delisted tickers. Alpaca's market-data API
(free plan: consolidated SIP history since 2016-01) still serves the as-traded history of
since-delisted symbols (BCR 2016-01-04 .. 2017-12-28, BRCM .. 2016-01-29, LLTC .. 2017-03-10).

Access: read-only reuse of `providers/alpaca.py` (`fetch_bars_raw` with an own 60 req/min token
bucket; keys from the environment only, never stored) with
  GET /v2/stocks/bars?symbols=...&timeframe=1Day&adjustment=raw&feed=sip&asof=-
`asof=-` disables Alpaca's symbol mapping, so a request for T returns the bars that traded
under the symbol T on each date (AS-TRADED BY TICKER, the same semantics as DoltHub; the
default asof=today would map META -> FB history). The identity layer then attributes each
date to an issuer (CIK) exactly as for DoltHub rows (identity.build_series): rows on dates the
ticker is evidenced to denote the CIK, other rows only by price agreement / renamed-history
continuity. Daily bar `t` = 00:00 ET of the session date.

Validation (validate_source, run before any store write; results persisted in
reference/alpaca_eod/validation): on survivors with an overlapping trusted as-traded source
(Yahoo raw rebuilt with Yahoo's split list, DoltHub) the close must agree (median |log ratio|
<= 0.2 %, 99th pct <= 2 %), the volume must agree (median |log ratio| <= 25 %: SIP vs vendor
consolidated volume) and every Yahoo split must appear as the same raw price ratio. Per symbol,
`fetch_symbols` refuses to store a delisted ticker whose overlap with DoltHub rows of the same
ticker disagrees (same thresholds as identity._agree). Stored as bars_1d/alpaca_eod/<T>,
fidelity T1_BAR (SIP consolidated), raw (as traded).
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from atlasbt.core.types import Fidelity

from .. import frames
from ..store import ParquetStore

log = logging.getLogger("atlasbt.data.alpaca_eod")
SOURCE = "alpaca_eod"
START = "2016-01-01"            # free-plan SIP history starts 2016-01

# source-level acceptance thresholds (close: as-traded prints of the same consolidated tape)
CLOSE_MED, CLOSE_P99 = 0.002, 0.02
VOL_MED = 0.25
MIN_PASS_SHARE = 0.95
#: survivors used for the source-level validation (Yahoo raw + DoltHub overlap 2016 ->, many splits)
VALIDATION_SET = ("AAPL MSFT JPM XOM GE KO PG JNJ WMT T F BAC C INTC CSCO PFE MRK CVX HD DIS MCD IBM ORCL NKE "
                  "LOW AMGN GILD MO PM UNH CAT BA MMM HON UPS SBUX NFLX AMZN GOOGL TSLA NVDA CMG AVGO NEE DUK "
                  "SO BRK-B").split()


def available() -> bool:
    from . import alpaca
    return alpaca.available()


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """alpaca.parse_bars frame (1Day bars, start = 00:00 ET of the session, in UTC) -> canonical
    raw daily bars (zero-volume filler bars removed)."""
    if df is None or not len(df):
        return frames.empty_bars()
    # Alpaca emits zero-trade filler bars (o=h=l=c = last print, volume 0) on sessions without any
    # trade, e.g. DD 2017-09-01 .. 2019-05-31 at old DuPont's last close after the DowDuPont merger:
    # they are not prices -> dropped
    df = df[df["volume"].astype(float) > 0]
    if not len(df):
        return frames.empty_bars()
    d = pd.DatetimeIndex(df.index).tz_convert("America/New_York").tz_localize(None).normalize()
    raw = pd.DataFrame({c: df[c].astype(float).to_numpy() for c in ("open", "high", "low", "close", "volume")},
                       index=d)
    bars, _ = frames.daily_bars(raw)
    return bars


_CLI = None


def _client():
    """Own token bucket at a third of Alpaca's 180 req/min: the intraday fetcher may run concurrently."""
    global _CLI
    if _CLI is None:
        from . import alpaca
        _CLI = alpaca.AlpacaClient(rpm=60)
    return _CLI


def download_many(symbols: Iterable[str], start=START, end=None, batch: int = 50) -> dict[str, pd.DataFrame]:
    """Raw daily bars as traded under each symbol (asof=-), without storing: {store ticker: bars}."""
    from . import alpaca
    end = pd.Timestamp(end or pd.Timestamp.now(tz="UTC").normalize())
    syms = list(dict.fromkeys(str(s).upper() for s in symbols))
    back = {alpaca.alpaca_symbol(s): s for s in syms}
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(syms), batch):
        chunk = [alpaca.alpaca_symbol(s) for s in syms[i:i + batch]]
        got = alpaca.fetch_bars_raw(chunk, pd.Timestamp(start), end, freq="1d", cli=_client())
        for a, df in got.items():
            out[back.get(a, a.replace(".", "-"))] = to_daily(df)
    return out


def download(symbol: str, start=START, end=None) -> pd.DataFrame:
    return download_many([symbol], start, end).get(str(symbol).upper(), frames.empty_bars())


# ---------------------------------------------------------------------------- validation
def _logratio(a: pd.Series, b: pd.Series) -> np.ndarray:
    common = a.index.intersection(b.index)
    r = np.log(a.loc[common].to_numpy(dtype=float) / b.loc[common].to_numpy(dtype=float))
    return r[np.isfinite(r)]


def compare(cand: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """Close / volume agreement of two raw daily bar frames on common sessions."""
    c = pd.Series(cand["close"].to_numpy(), index=frames.to_naive_date(cand.index))
    r = pd.Series(ref["close"].to_numpy(), index=frames.to_naive_date(ref.index))
    lc = np.abs(_logratio(c, r))
    cv = pd.Series(cand["volume"].to_numpy(), index=c.index)
    rv = pd.Series(ref["volume"].to_numpy(), index=r.index)
    cv, rv = cv[cv > 0], rv[rv > 0]
    lv = np.abs(_logratio(cv, rv))
    out = {"n": int(len(lc)), "close_med": float(np.median(lc)) if len(lc) else np.nan,
           "close_p99": float(np.quantile(lc, 0.99)) if len(lc) else np.nan,
           "vol_med": float(np.median(lv)) if len(lv) else np.nan}
    out["ok"] = bool(len(lc) >= 20 and out["close_med"] <= CLOSE_MED and out["close_p99"] <= CLOSE_P99
                     and (not len(lv) or out["vol_med"] <= VOL_MED))
    return out


def split_check(cand: pd.DataFrame, splits: pd.DataFrame) -> list[dict]:
    """Every reference split (ex_date, ratio = new shares per old) must show up as the same raw
    open / previous-close ratio in the candidate (within 15 %)."""
    out = []
    c = pd.Series(cand["close"].to_numpy(), index=frames.to_naive_date(cand.index))
    o = pd.Series(cand["open"].to_numpy(), index=c.index)
    for r in splits.itertuples():
        ex = pd.Timestamp(r.ex_date).normalize()
        if ex not in c.index or c.index.get_loc(ex) == 0:
            continue
        j = c.index.get_loc(ex)
        g = float(o.iloc[j] / c.iloc[j - 1])
        out.append({"ex_date": str(ex.date()), "ratio": float(r.ratio), "gap": g,
                    "ok": bool(abs(np.log(g * float(r.ratio))) < 0.15)})
    return out


def validate_source(portal, symbols: Iterable[str], start=START, end=None, fetched: Optional[dict] = None) -> dict:
    """Source-level validation on survivors: Alpaca raw vs Yahoo raw (as-traded, rebuilt) and vs
    DoltHub; Yahoo splits vs Alpaca raw price ratios. Returns a report; `accepted` only if >=
    MIN_PASS_SHARE of the compared symbols pass close+volume and every split check passes."""
    rows, splits = [], []
    for s in symbols:
        a = fetched.get(s) if fetched is not None else download(s, start, end)
        if a is None or not len(a):
            rows.append({"symbol": s, "ref": "-", "error": "no alpaca rows"})
            continue
        for ref in ("yahoo", "dolthub"):
            if not portal.store.exists("bars_1d", ref, s):
                continue
            b = portal._read("bars_1d", ref, s)
            if not len(b):
                continue
            rows.append({"symbol": s, "ref": ref, **compare(a, b)})
        acts = portal._read("actions", "yahoo", s)
        if len(acts):
            sp = acts[acts["type"] == "SPLIT"]
            for x in split_check(a, sp):
                splits.append({"symbol": s, **x})
    df = pd.DataFrame(rows)
    cmp_ = df[df.get("ok").notna()] if "ok" in df else df.iloc[0:0]
    share = float(cmp_["ok"].mean()) if len(cmp_) else 0.0
    sp = pd.DataFrame(splits)
    rep = {"symbols": len(set(df["symbol"])) if len(df) else 0, "comparisons": int(len(cmp_)),
           "pass_share": round(share, 4),
           "close_med_median": float(cmp_["close_med"].median()) if len(cmp_) else np.nan,
           "vol_med_median": float(cmp_["vol_med"].median()) if len(cmp_) else np.nan,
           "splits_checked": int(len(sp)), "splits_ok": int(sp["ok"].sum()) if len(sp) else 0,
           "failed": cmp_.loc[~cmp_["ok"].astype(bool), ["symbol", "ref", "close_med", "close_p99", "vol_med"]]
           .to_dict("records") if len(cmp_) else []}
    rep["accepted"] = bool(len(cmp_) >= 20 and share >= MIN_PASS_SHARE and (not len(sp) or bool(sp["ok"].all())))
    rep["detail"] = df
    rep["split_detail"] = sp
    return rep


def fetch_symbols(portal, symbols: Iterable[str], validation: dict, start=START, end=None,
                  store: Optional[ParquetStore] = None, log_fn=print) -> dict:
    """Download and store bars_1d/alpaca_eod/<T> for `symbols`, only if `validation['accepted']`.
    A symbol whose rows overlap DoltHub rows of the same ticker (>= 20 sessions) must agree with
    them (identity._agree thresholds), otherwise it is rejected (not stored)."""
    from ..identity import _agree, _close_series
    store = store or portal.store
    if not validation.get("accepted"):
        return {"skipped": "source validation failed", "validation": {k: v for k, v in validation.items()
                                                                      if k not in ("detail", "split_detail")}}
    out = {"stored": [], "empty": [], "rejected": [], "errors": []}
    symbols = list(dict.fromkeys(str(s).upper() for s in symbols))
    got = download_many(symbols, start, end)
    for s in symbols:
        a = got.get(s)
        if a is None or not len(a):
            out["empty"].append(s)
            continue
        if store.exists("bars_1d", "dolthub", s):
            v = _agree(_close_series(a), _close_series(portal._read("bars_1d", "dolthub", s)))
            if v is False:
                out["rejected"].append(s)
                log_fn(f"alpaca_eod {s}: disagrees with DoltHub {s} on overlapping sessions -> not stored")
                continue
        store.write("bars_1d", SOURCE, s, a, fidelity=int(Fidelity.T1_BAR), replace=True,
                    extra={"origin": "alpaca v2 stocks bars 1Day feed=sip adjustment=raw asof=-", "raw": True,
                           "first_ts": str(a.index[0]), "last_ts": str(a.index[-1]),
                           "validation_pass_share": validation.get("pass_share")})
        out["stored"].append(s)
    return out


def needed_symbols(portal, start="2016-01-04", end="2018-12-31", step: str = "MS") -> list[str]:
    """Tickers of S&P 500 members (and their CIK lineage tickers) without identity-verified bars on
    some month start in [start, end] -- what this source can patch."""
    idx = portal.identity_index()
    out = set()
    for d in pd.date_range(start, end, freq=step):
        a = pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=21)
        allm = portal.universe("SP500", a)
        kept = set(portal.universe("SP500", a, require_bars=True))
        for k in allm:
            if k in kept:
                continue
            t, c = portal._split_key(k)
            out.add(t)
            cc = c or idx.ticker_cik_at(t, d)
            if cc:
                out.update(idx.zl_tickers_of(int(cc)))
    return sorted(out)
