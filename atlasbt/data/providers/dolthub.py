"""DoltHub SQL API (https://www.dolthub.com/api/v1alpha1/<owner>/<db>/<branch>?q=SQL).

Databases (owner post-no-preference):
  options   option_chain(date, act_symbol, expiration, strike, call_put, bid, ask, vol,
            delta, gamma, theta, vega, rho)  PK(date, act_symbol, expiration, strike, call_put)
            volatility_history(date, act_symbol, hv_*, iv_*)
  earnings  earnings_calendar(act_symbol, date, when) PK(act_symbol, date)  -- since 2020
            eps_history(act_symbol, period_end_date, reported, estimate)  -- since ~2016
            eps_estimate(date, act_symbol, period, period_end_date, consensus, ...) weekly snapshots
  stocks    dividend(act_symbol, ex_date, amount), split(act_symbol, ex_date, to_factor, for_factor)
  rates     us_treasury(date, 1_month ... 30_year)

API behaviour measured 2026-10-28: max 1000 rows per response (status 'RowLimit'),
~55 s server deadline (aggregates/scans over many dates fail with 'context deadline
exceeded'); key-prefix lookups are fast (0.3-40 s depending on server cache).
So we (1) enumerate snapshot dates by index seeks (`WHERE date > d ORDER BY date LIMIT 1`),
(2) fetch each (date, symbol-group) with exact-key predicates, splitting groups that
hit the row cap. Responses are cached on disk only when status == Success.

Option snapshot timing (measured via put-call parity vs SPY daily bars): row date d
holds quotes from ~the close of session d (weekend dates = previous Friday's close;
e.g. 2024-01-03 parity-implied spot 468.8 vs close 468.79; 2026-10-27 769.9 vs 769.66).
=> ts := official close of the last NYSE session <= d (UTC).  Exact capture minute is
unknown and on volatile days quotes deviate from the closing print by up to ~0.5%
(2022-06-15) -> fidelity T2_PROXY. Coverage is SPARSE: ~3-4 expirations (nearest
~2 weeks to ~2 months) and ~20-50 strikes per expiry; no QQQ/IWM/SPX/GLD/TLT.
"""
from __future__ import annotations

import json
import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from atlasbt.core.types import Fidelity, Instrument, Right

from .. import frames
from ..http import FetchError, client
from ..store import ParquetStore, default_store

log = logging.getLogger("atlasbt.data.dolthub")
SOURCE = "dolthub"
API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/{db}/master"
ROW_LIMIT = 1000

# Underlyings verified present in option_chain (probe 2022-06-15 & 2026-10-27).
# Absent: QQQ, IWM, SPX, NDX, RUT, GLD, TLT, EEM, EFA, HYG, LQD, SMH, ...
# in fetch priority order
DEFAULT_OPTION_UNDERLYINGS = ["SPY", "AAPL", "MSFT", "DIA", "XLF", "XLE", "XLK", "NVDA", "AMZN", "TSLA",
                              "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "GOOGL", "META",
                              "JPM", "XOM", "KRE", "XBI", "XOP"]


class DoltQueryError(RuntimeError):
    pass


def _ok(content: bytes) -> bool:
    try:
        return json.loads(content).get("query_execution_status") == "Success"
    except Exception:
        return False


def query(db: str, sql: str, *, cache: bool = True, retries: int = 3) -> list[dict]:
    """Run SQL; returns rows. Raises DoltQueryError on RowLimit / timeout after retries."""
    last = None
    for i in range(retries):
        body = client().get(API.format(db=db), params={"q": sql}, cache=cache, validate=_ok,
                            timeout=120, refresh=i > 0)
        d = json.loads(body)
        st = d.get("query_execution_status")
        if st == "Success":
            return d.get("rows", [])
        if st == "RowLimit":
            raise DoltQueryError("RowLimit")
        last = d.get("query_execution_message")
    raise DoltQueryError(f"{db}: {last} :: {sql[:200]}")


def _in(symbols: Iterable[str]) -> str:
    return ",".join("'" + s.replace("'", "") + "'" for s in symbols)


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------
def enumerate_dates(db: str, table: str, start: str = "2010-01-01", store: Optional[ParquetStore] = None) -> list[str]:
    """Distinct `date` values via index seeks; persisted in reference/dolthub/<table>_dates."""
    store = store or default_store()
    key = f"{table}_dates"
    have = store.read("reference", SOURCE, key)
    dates = sorted(have["date"].astype(str)) if len(have) else []
    cur = dates[-1] if dates else start
    new = []
    while True:
        try:
            rows = query(db, f"SELECT date FROM {table} WHERE date > '{cur}' ORDER BY date LIMIT 1",
                         cache=False)
        except DoltQueryError as e:
            log.warning("seek after %s failed: %s", cur, e)
            break
        if not rows:
            break
        cur = rows[0]["date"]
        new.append(cur)
        if len(new) % 50 == 0:
            log.info("%s: enumerated %d new dates (at %s)", table, len(new), cur)
            store.write("reference", SOURCE, key, pd.DataFrame({"date": new}), key=["date"])
    if new:
        store.write("reference", SOURCE, key, pd.DataFrame({"date": new}), key=["date"])
    return sorted(set(dates) | set(new))


def snapshot_ts(date) -> pd.Timestamp:
    """Close (UTC) of the last NYSE session on or before `date`."""
    sch = frames.schedule()
    d = pd.Timestamp(date).normalize()
    i = sch.index.searchsorted(d, side="right") - 1
    return sch["close"].iloc[i]


def parse_option_rows(rows: list[dict], underlying_close: Optional[dict] = None) -> pd.DataFrame:
    """DoltHub option_chain rows -> canonical option_chain frame."""
    if not rows:
        return frames.empty_chain()
    df = pd.DataFrame(rows)
    num = ["strike", "bid", "ask", "vol", "delta", "gamma", "theta", "vega", "rho"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out = pd.DataFrame(index=df.index)
    uniq = {d: snapshot_ts(d) for d in df["date"].unique()}
    out["ts"] = df["date"].map(uniq)
    out["underlying"] = df["act_symbol"].str.upper()
    out["expiry"] = pd.to_datetime(df["expiration"])
    out["strike"] = df["strike"].astype(float)
    out["right"] = np.where(df["call_put"].str.upper().str.startswith("C"), "C", "P")
    out["symbol"] = [Instrument.option(u, e, k, Right(r)).symbol
                     for u, e, k, r in zip(out["underlying"], out["expiry"], out["strike"], out["right"])]
    bid, ask = df["bid"].astype(float), df["ask"].astype(float)
    out["bid"], out["ask"] = bid, ask
    out["mid"] = np.where((bid > 0) | (ask > 0), 0.5 * (bid + ask), np.nan)
    out["iv"] = df["vol"]
    out["delta"], out["gamma"], out["vega"], out["theta"] = df["delta"], df["gamma"], df["vega"], df["theta"]
    out["rho"] = df["rho"]
    if underlying_close:
        out["underlying_price"] = [underlying_close.get((u, pd.Timestamp(t))) for u, t in zip(out["underlying"], out["ts"])]
    else:
        out["underlying_price"] = np.nan
    out["underlying_price"] = pd.to_numeric(out["underlying_price"], errors="coerce")
    out["volume"] = np.nan
    out["open_interest"] = np.nan
    out["exercise"] = "AMERICAN"
    out["settlement"] = "PHYSICAL"
    out["multiplier"] = 100.0
    out["fidelity"] = int(Fidelity.T2_PROXY)
    out["source_date"] = pd.to_datetime(df["date"])
    from ..option_meta import to_internal
    return to_internal(out)        # internal symbol convention at ingestion


def _fetch_date_symbols(date: str, symbols: list[str]) -> list[dict]:
    try:
        return query("options", f"SELECT * FROM option_chain WHERE date='{date}' AND act_symbol IN ({_in(symbols)})")
    except DoltQueryError as e:
        if "RowLimit" in str(e) and len(symbols) > 1:
            h = len(symbols) // 2
            return _fetch_date_symbols(date, symbols[:h]) + _fetch_date_symbols(date, symbols[h:])
        if "RowLimit" in str(e):  # one symbol > 1000 rows: page by expiration
            exps = query("options", f"SELECT DISTINCT expiration FROM option_chain WHERE date='{date}' "
                                    f"AND act_symbol='{symbols[0]}'")
            out = []
            for e in exps:
                out += query("options", f"SELECT * FROM option_chain WHERE date='{date}' AND act_symbol="
                                        f"'{symbols[0]}' AND expiration='{e['expiration']}'")
            return out
        raise


def _underlying_closes(symbols, store: ParquetStore) -> dict:
    out = {}
    for s in symbols:
        b = store.read("bars_1d", "yahoo", s, columns=["close", "known_at"])
        if len(b):
            out.update({(s, pd.Timestamp(k)): c for k, c in zip(b["known_at"], b["close"])})
    return out


def candidate_dates(store: ParquetStore, daily_from: str = "2020-01-06") -> list[str]:
    """Snapshot dates to try: the enumerated weekly 2019 snapshots (Saturdays) plus every
    NYSE session from `daily_from` (the table is daily from 2020-01-06; enumerating dates by
    index seeks costs ~10 s each, the same as probing a session directly)."""
    have = store.read("reference", SOURCE, "option_chain_dates")
    early = sorted(d for d in have["date"].astype(str) if d < daily_from) if len(have) else []
    if not early:
        early = [d for d in enumerate_dates("options", "option_chain", start="2010-01-01", store=store)
                 if d < daily_from]
    sch = frames.schedule()
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    sess = sch.index[(sch.index >= pd.Timestamp(daily_from)) & (sch.index < today)]
    return early + [d.strftime("%Y-%m-%d") for d in sess]


def fetch_option_chains(symbols: Optional[list[str]] = None, start: str = "2010-01-01",
                        end: Optional[str] = None, store: Optional[ParquetStore] = None,
                        workers: int = 3, newest_first: bool = False) -> dict:
    """Pull chain snapshots into option_chain/dolthub/<UNDERLYING>/<YYYY>.parquet.

    One pass per underlying, in priority order, one exact-key query per (date, symbol)
    (measured: IN-lists force a scan of the whole date, ~40 s; single-symbol lookups take
    3-8 s, ~28 s when the date is cold). Resumable: (date, symbol) pairs already done are
    recorded in reference/dolthub/option_chain_done (rows=0: no data for that pair)."""
    store = store or default_store()
    symbols = [s.upper() for s in (symbols or DEFAULT_OPTION_UNDERLYINGS)]
    dates = [d for d in candidate_dates(store) if d >= start and (end is None or d <= end)]
    if newest_first:
        dates = dates[::-1]
    import concurrent.futures as cf
    stats = {"dates_total": len(dates), "pairs_fetched": 0, "pairs_empty": 0, "rows": 0, "errors": 0}
    for sym in symbols:
        done_df = store.read("reference", SOURCE, "option_chain_done")
        done = set()
        if len(done_df) and "symbol" in done_df:
            dd = done_df[done_df["symbol"] == sym]
            done = set(dd["date"].astype(str))
        todo = [d for d in dates if d not in done]
        closes = _underlying_closes([sym], store)
        log.info("option chains %s: %d candidate dates, %d to do", sym, len(dates), len(todo))

        def one(d, sym=sym):
            return d, query("options", f"SELECT * FROM option_chain WHERE date='{d}' AND act_symbol='{sym}'")

        buf, bdone = [], []
        with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(one, d): d for d in todo}
            for f in cf.as_completed(futs):
                d = futs[f]
                try:
                    _, rows = f.result()
                except (DoltQueryError, FetchError) as e:
                    stats["errors"] += 1
                    log.warning("options %s %s failed: %s", sym, d, e)
                    continue
                if rows:
                    buf.append(parse_option_rows(rows, closes))
                    stats["pairs_fetched"] += 1
                else:
                    stats["pairs_empty"] += 1
                bdone.append({"date": d, "symbol": sym, "rows": len(rows)})
                stats["rows"] += len(rows)
                if len(bdone) >= 25:
                    _flush_chains(store, sym, buf, bdone)
                    buf, bdone = [], []
                    log.info("options progress %s %s (last %s)", sym, stats, d)
        _flush_chains(store, sym, buf, bdone)
    return stats


def _flush_chains(store: ParquetStore, sym: str, buf: list, bdone: list) -> None:
    if buf:
        chain = pd.concat(buf, ignore_index=True)
        store.write("option_chain", SOURCE, sym, chain, partition="year", time_col="ts", key=["ts", "symbol"],
                    fidelity=int(Fidelity.T2_PROXY),
                    extra={"snapshot_convention": "close of last session <= dolthub date"})
    if bdone:
        store.write("reference", SOURCE, "option_chain_done", pd.DataFrame(bdone), key=["date", "symbol"])


def probe_option_coverage(dates: list[str], candidates: list[str]) -> pd.DataFrame:
    """Which candidate underlyings exist on given dates (rows, expiries, strike range)."""
    out = []
    for d in dates:
        rows = query("options", "SELECT act_symbol, COUNT(*) n, COUNT(DISTINCT expiration) ne, MIN(expiration) e0, "
                                "MAX(expiration) e1, MIN(strike) k0, MAX(strike) k1 FROM option_chain WHERE "
                                f"date='{d}' AND act_symbol IN ({_in(candidates)}) GROUP BY act_symbol")
        for r in rows:
            out.append({"date": d, **r})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# earnings
# ---------------------------------------------------------------------------
TIMING = {"before market open": "BMO", "after market close": "AMC", "during market hours": "DURING"}
# Approximate release clock times used for known_at when only the calendar is known (T2):
#   BMO -> 08:00 ET (typical 06:00-08:30), AMC -> 16:30 ET (typical 16:01-16:30),
#   DURING -> 12:00 ET, UNKNOWN -> next session open after the date (conservative).
# Prefer ZipLime / SEC 8-K acceptance times (exact) whenever available (events.py does).
TIMING_CLOCK = {"BMO": "08:00", "AMC": "16:30", "DURING": "12:00"}


def earnings_known_at(date, timing: str) -> pd.Timestamp:
    from atlasbt.core.clock import et
    d = pd.Timestamp(date).normalize()
    if timing in TIMING_CLOCK:
        return et(d, TIMING_CLOCK[timing])
    # UNKNOWN: conservatively assume results are usable only at the next session open after d
    sch = frames.schedule()
    i = sch.index.searchsorted(d, side="right")
    return sch["open"].iloc[i] if i < len(sch) else et(d, "16:30")


def parse_earnings(symbol: str, cal_rows: list[dict], eps_rows: list[dict],
                   est_rows: Optional[list[dict]] = None) -> pd.DataFrame:
    """-> events frame (one row per release) with payload columns."""
    if not cal_rows:
        return pd.DataFrame()
    cal = pd.DataFrame(cal_rows)
    cal["date"] = pd.to_datetime(cal["date"])
    eps = pd.DataFrame(eps_rows)
    if len(eps):
        eps["period_end_date"] = pd.to_datetime(eps["period_end_date"])
        eps["reported"] = pd.to_numeric(eps["reported"], errors="coerce")
        eps["estimate"] = pd.to_numeric(eps["estimate"], errors="coerce")
        eps = eps.sort_values("period_end_date")
    est = pd.DataFrame(est_rows or [])
    if len(est):
        est["date"] = pd.to_datetime(est["date"])
        est["period_end_date"] = pd.to_datetime(est["period_end_date"])
        est["consensus"] = pd.to_numeric(est["consensus"], errors="coerce")
    out = []
    for _, r in cal.sort_values("date").iterrows():
        timing = TIMING.get(str(r.get("when") or "").strip().lower(), "UNKNOWN")
        known = earnings_known_at(r["date"], timing)
        fp = None
        eps_a = eps_e = np.nan
        est_known = pd.NaT
        if len(eps):
            # fiscal period = latest period end at least ~10 days before the release
            cand = eps[eps["period_end_date"] <= r["date"] - pd.Timedelta(days=10)]
            if len(cand):
                row = cand.iloc[-1]
                if (r["date"] - row["period_end_date"]).days <= 120:
                    fp = row["period_end_date"]
                    eps_a, eps_e = row["reported"], row["estimate"]
                    # the final consensus is the pre-release estimate: known before release
                    est_known = known - pd.Timedelta(minutes=1)
        if fp is not None and len(est):
            snaps = est[(est["period_end_date"] == fp) & (est["date"] < r["date"])]
            if len(snaps):
                s = snaps.sort_values("date").iloc[-1]
                eps_e = s["consensus"]
                # snapshot date's values known by end of that day
                from atlasbt.core.clock import et
                est_known = et(s["date"], "23:59")
        out.append({"kind": "EARNINGS", "symbol": symbol.upper(), "known_at": known,
                    "effective_at": known, "timing": timing, "eps_actual": eps_a, "eps_estimate": eps_e,
                    "estimate_known_at": est_known, "fiscal_period": fp, "release_date": r["date"],
                    "fidelity": int(Fidelity.T2_PROXY), "source": SOURCE})
    return pd.DataFrame(out)


def fetch_earnings(symbol: str, store: Optional[ParquetStore] = None) -> dict:
    store = store or default_store()
    s = symbol.upper().replace("-", ".")   # DoltHub uses BRK.B style
    cal = query("earnings", f"SELECT * FROM earnings_calendar WHERE act_symbol='{s}' ORDER BY date")
    eps = query("earnings", f"SELECT * FROM eps_history WHERE act_symbol='{s}' ORDER BY period_end_date")
    ev = parse_earnings(symbol, cal, eps, _cached_estimates(s, store))
    if len(ev):
        store.write("events", SOURCE, f"EARNINGS_{symbol.upper()}", ev, key=["release_date"], replace=True,
                    fidelity=int(Fidelity.T2_PROXY))
    return {"symbol": symbol, "events": len(ev),
            "first": str(ev["release_date"].min().date()) if len(ev) else None}


def _cached_estimates(symbol: str, store: ParquetStore) -> list[dict]:
    e = store.read("reference", SOURCE, "eps_estimate")
    if not len(e):
        return []
    e = e[e["act_symbol"] == symbol]
    return e.to_dict("records")


def fetch_eps_estimate_snapshots(symbols: list[str], store: Optional[ParquetStore] = None,
                                 start: str = "2016-01-01") -> dict:
    """Weekly consensus snapshots ('Current Quarter') for `symbols` -> reference/dolthub/eps_estimate."""
    store = store or default_store()
    syms = [s.upper().replace("-", ".") for s in symbols]
    dates = enumerate_dates("earnings", "eps_estimate", start=start, store=store)
    have = store.read("reference", SOURCE, "eps_estimate_done")
    done = set(have["date"].astype(str)) if len(have) else set()
    n = 0
    for d in dates:
        if d in done:
            continue
        rows = []
        for i in range(0, len(syms), 300):
            try:
                rows += query("earnings", f"SELECT * FROM eps_estimate WHERE date='{d}' AND period='Current Quarter' "
                                          f"AND act_symbol IN ({_in(syms[i:i + 300])})")
            except DoltQueryError as e:
                log.warning("eps_estimate %s: %s", d, e)
        if rows:
            store.write("reference", SOURCE, "eps_estimate", pd.DataFrame(rows), key=["date", "act_symbol", "period"])
        store.write("reference", SOURCE, "eps_estimate_done", pd.DataFrame({"date": [d], "rows": [len(rows)]}),
                    key=["date"])
        n += len(rows)
    return {"snapshot_dates": len(dates), "rows": n}


# ---------------------------------------------------------------------------
# dividends / splits cross-check (DoltHub 'stocks')
# ---------------------------------------------------------------------------
def fetch_stock_actions(symbol: str) -> pd.DataFrame:
    s = symbol.upper().replace("-", ".")
    d = query("stocks", f"SELECT * FROM dividend WHERE act_symbol='{s}' ORDER BY ex_date")
    sp = query("stocks", f"SELECT * FROM split WHERE act_symbol='{s}' ORDER BY ex_date")
    rows = [{"type": "CASH_DIVIDEND", "ex_date": pd.Timestamp(r["ex_date"]), "amount": float(r["amount"]),
             "ratio": 1.0} for r in d]
    rows += [{"type": "SPLIT", "ex_date": pd.Timestamp(r["ex_date"]), "amount": 0.0,
              "ratio": float(r["to_factor"]) / float(r["for_factor"])} for r in sp]
    return pd.DataFrame(rows)
