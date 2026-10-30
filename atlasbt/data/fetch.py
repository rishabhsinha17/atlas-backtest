"""Downloader CLI.

    python -m atlasbt.data.fetch --set core            # everything in the CORE dataset
    python -m atlasbt.data.fetch --only intraday       # one task (see TASKS)
    python -m atlasbt.data.fetch --only daily_etf --symbols SPY,QQQ

Every task is idempotent (store upserts + raw HTTP cache), so re-running only
fetches what is new. Progress is logged to cache/logs/fetch_<task>.log and a
JSON summary is written to cache/logs/summary_<task>.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from typing import Callable

import numpy as np
import pandas as pd

from . import config
from . import universes as U
from .store import default_store

log = logging.getLogger("atlasbt.data.fetch")


def _budget_ok() -> bool:
    used = default_store().size_bytes()
    raw = sum(f.stat().st_size for f in config.RAW_ROOT.rglob("*") if f.is_file()) if config.RAW_ROOT.exists() else 0
    if used + raw > config.CACHE_BUDGET_BYTES:
        log.error("cache budget exceeded: %.1f GB", (used + raw) / 1e9)
        return False
    return True


def _each(symbols, fn: Callable, label: str) -> list:
    out = []
    for i, s in enumerate(symbols):
        try:
            r = fn(s)
            out.append(r)
            log.info("[%s %d/%d] %s", label, i + 1, len(symbols), r)
        except Exception as e:  # keep going; log the failure
            log.error("[%s] %s failed: %s\n%s", label, s, e, traceback.format_exc(limit=3))
            out.append({"symbol": s, "error": str(e)})
        if i % 25 == 0 and not _budget_ok():
            break
    return out


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------
def task_daily_etf(symbols=None):
    from .providers import yahoo
    return _each(symbols or U.CORE_ETFS + U.MEGACAPS, yahoo.fetch_daily, "daily")


def task_intraday(symbols=None):
    from .providers import yahoo
    syms = symbols or U.INTRADAY_CORE
    out = []
    for iv in ("1m", "5m", "60m"):
        out += _each(syms, lambda s: yahoo.fetch_intraday(s, iv), f"intraday-{iv}")
    return out


def task_cboe(symbols=None):
    from .providers import cboe
    return _each(symbols or cboe.DEFAULT_INDICES, cboe.fetch_index, "cboe")


def task_fred(symbols=None):
    from .providers import fred
    return _each(symbols or list(fred.SERIES), fred.fetch_series, "fred")


def task_french(symbols=None):
    from .providers import french
    return _each(symbols or list(french.DATASETS), french.fetch_dataset, "french")


def task_sp500(symbols=None):
    from .providers import sp500
    return [sp500.build_membership()]


def task_daily_sp500(symbols=None):
    from .providers import sp500, yahoo
    syms = symbols or sp500.union_tickers()
    return _each(syms, yahoo.fetch_daily, "daily-sp500")


def task_sec(symbols=None):
    from .providers import sec
    out = [sec.fetch_ticker_map()]
    from .providers import sp500
    syms = symbols or sorted(set(sp500.union_tickers()) | set(U.MEGACAPS))
    out += _each(syms, sec.fetch_company, "sec")
    return out


def task_fomc(symbols=None):
    from .providers import fomc
    return [fomc.fetch_fomc()]


def task_earnings(symbols=None):
    from .providers import dolthub
    from .providers import sp500
    syms = symbols or sorted(set(sp500.union_tickers()) | set(U.MEGACAPS))
    return _each(syms, dolthub.fetch_earnings, "earnings")


def task_options(symbols=None):
    """DoltHub SQL API, only for snapshot dates after the bulk CSV mirror (2026-01-27);
    priority underlyings only (one exact-key query per (date, symbol), ~3-8 s each)."""
    from .providers import dolthub
    b = default_store().read("reference", "dolthub", "option_chain_bulk")
    start = (pd.Timestamp(b["bulk_last_date"].max()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") \
        if len(b) else "2010-01-01"
    syms = symbols or ["SPY", "AAPL", "MSFT", "DIA", "XLF", "XLE", "XLK", "NVDA", "AMZN", "TSLA"]
    return [dolthub.fetch_option_chains(syms, start=start, workers=2)]


def task_ziplime(symbols=None):
    from .providers import ziplime
    return [ziplime.load()]


def task_dolthub_tables(symbols=None):
    """Scout's full DoltHub table exports (earnings calendar, eps history, dividends, splits)
    plus the eps_estimate mirror if ATLASBT_EPS_ESTIMATE_CSV points to it."""
    import os
    from .providers import dolthub_bulk
    out = [dolthub_bulk.load_small_tables()]
    if os.environ.get("ATLASBT_EPS_ESTIMATE_CSV"):
        out.append(dolthub_bulk.load_eps_estimates(os.environ["ATLASBT_EPS_ESTIMATE_CSV"]))
    return out


def task_iex_sample(symbols=None):
    """Parse the scout's IEX TOPS pcap (2019-12-24) for a few symbols."""
    from .providers import iex
    p = config.DATA_RAW / "iex_hist" / "20191224_IEXTP1_TOPS1.6.pcap.gz"
    return [iex.load_tops_file(p, symbols or ["SPY", "QQQ", "AAPL"])] if p.exists() else []


def task_hod(symbols=None):
    from .providers import hod
    syms = symbols or (["SPX", "SPXW", "NDX", "RUT", "VIX", "QQQ", "IWM", "GLD", "TLT", "EEM", "EFA"]
                       + U.CORE_ETFS + U.MEGACAPS)
    return [hod.load(sorted(set(syms)))]


def task_nasdaqtrader(symbols=None):
    from .providers import nasdaqtrader
    return [nasdaqtrader.fetch_directory()]


def task_borrow(symbols=None):
    # IBorrowDesk keyed v2 API (IBORROWDESK_API_KEY); without a key: {'skipped': ..., cost plan}
    from .providers import iborrowdesk
    return [iborrowdesk.fetch_daily(symbols)]


def task_dividend_decl(symbols=None):
    # events DIVIDEND_DECL from EDGAR full-text search (efts.sec.gov; plain UA accepted)
    from .providers import dividends_decl
    return [dividends_decl.build(symbols)]


def task_buybacks(symbols=None):
    # BUYBACK candidates (EFTS); events only with SEC_USER_AGENT (filing text on www.sec.gov)
    from .providers import buybacks
    return [buybacks.build(symbols)]


def task_dukascopy(symbols=None):
    from .providers import dukascopy
    return _each(symbols or ["USA500IDXUSD"], lambda s: dukascopy.fetch_recent(s, days=5), "dukascopy")


def task_dukascopy_backfill(symbols=None):
    """Multi-year 1m mid-quote bars (CFD quotes, T2) for liquid ETFs."""
    from .providers import dukascopy
    start = "2017-01-01"
    end = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return _each(symbols or ["SPY", "QQQ", "IWM", "DIA"],
                 lambda s: dukascopy.fetch_range(s, start, end, "bars"), "dukascopy-backfill")


def _bulk_symbols():
    from .providers import sp500
    syms = set(sp500.union_tickers()) | set(U.CORE_ETFS) | set(U.MEGACAPS) | {
        "KRE", "XBI", "XOP", "SMH", "GDX", "XRT", "XHB", "ITB", "IYR", "OIH", "KWEB", "FXI", "EWZ", "USO", "SLV",
        "IEMG", "VXX", "UVXY", "ARKK", "VOO", "IVV", "VTI", "TSM"}
    m = default_store().read("reference", "sp500", "membership")
    if len(m):
        syms |= set(m["base"].dropna())
    return sorted(syms | {s.replace("-", ".") for s in syms})


def task_options_bulk(symbols=None):
    """Full DoltHub option_chain CSV mirror (path in env ATLASBT_OPTION_CSV)."""
    import os
    from .providers import dolthub_bulk
    return [dolthub_bulk.load_option_chain_csv(os.environ.get("ATLASBT_OPTION_CSV"), symbols or _bulk_symbols(),
                                               parts_dir=os.environ.get("ATLASBT_OPTION_PARTS"))]


def task_ohlcv_bulk(symbols=None):
    """Full DoltHub stocks.ohlcv CSV mirror (path in env ATLASBT_OHLCV_CSV): S&P 500 PIT union
    incl. delisted names, 2011+, raw."""
    import os
    from .providers import dolthub_bulk
    return [dolthub_bulk.load_ohlcv_csv(os.environ["ATLASBT_OHLCV_CSV"], symbols or _bulk_symbols())]


def task_alpaca_eod(symbols=None):
    """Survivorship patch: Alpaca SIP daily bars (2016 ->) for S&P 500 members without verified bars
    (delisted / renamed; DoltHub has nothing that stopped before 2018). Validates the source on
    survivors against Yahoo / DoltHub first (providers/alpaca_eod.py); run --only identity after."""
    from .portal import Portal
    from .providers import alpaca_eod as AE
    if not AE.available():
        log.error("alpaca_eod: ALPACA_API_KEY / ALPACA_API_SECRET not set in the environment -> skipped")
        return {"skipped": "no keys"}
    p = Portal()
    val = AE.validate_source(p, AE.VALIDATION_SET)
    detail = val.pop("detail")
    sd = val.pop("split_detail")
    if len(detail):
        p.store.write("reference", AE.SOURCE, "validation", detail.assign(validated_at=pd.Timestamp.now(tz="UTC")),
                      key=["symbol", "ref"], replace=True,
                      extra={"summary": {k: v for k, v in val.items() if k != "failed"}, "splits": sd.to_dict("records")})
    log.info("alpaca_eod validation: %s", {k: v for k, v in val.items() if k != "failed"})
    wanted = list(symbols) if symbols else AE.needed_symbols(p)
    return AE.fetch_symbols(p, wanted, val, log_fn=log.info)


def task_tiingo_eod(symbols=None):
    """Survivorship patch before 2016 (key-gated, TIINGO_API_KEY): download within the free-tier
    budget (resumable across days), validate against Yahoo / Alpaca / DoltHub, ingest
    (providers/tiingo.py); run --only identity after."""
    from .portal import Portal
    from .providers import tiingo as TI
    if not TI.available():
        log.error("tiingo: TIINGO_API_KEY not set in the environment -> skipped")
        return {"skipped": "no key"}
    p = Portal()
    TI.supported_lines(p.store, refresh=True)
    wanted = list(symbols) if symbols else TI.needed_symbols(p)
    got = TI.fetch_eod_many(list(TI.VALIDATION_SET) + wanted, log_fn=log.info)
    val = TI.validate(p, list(TI.VALIDATION_SET) + wanted)
    log.info("tiingo validation: %s", {k: v for k, v in val.items() if k not in ("detail", "split_detail", "failed")})
    return {"fetch": {k: len(v) for k, v in got.items()} if isinstance(got, dict) else got,
            "ingest": TI.ingest(p, wanted, val, log_fn=log.info)}


def task_identity(symbols=None):
    """Persist, for every S&P 500 membership ticker and the megacaps, the bar-date identity
    timeline and the verified daily series of every security the ticker denotes (plus the
    member security keys it maps to)."""
    from .portal import Portal
    p = Portal()
    m = default_store().read("reference", "sp500", "membership")
    tickers = sorted({str(b).replace(".", "-").upper() for b in m["base"]} | set(U.MEGACAPS))
    if symbols:
        tickers = [t for t in tickers if t in symbols]
    out = []
    for i, t in enumerate(tickers):
        try:
            out.append(p.persist_identity(t))
        except Exception as e:
            log.error("identity %s: %s", t, e)
        if i % 50 == 0:
            log.info("identity %d/%d", i, len(tickers))
    return out


def task_sec_submissions(symbols=None):
    """Precache data.sec.gov submissions JSON (registrant names + formerNames with dates) for
    every CIK the identity layer may consult: all issuers evidenced under any S&P 500 membership
    ticker, the SEC current owners of those tickers, the security master and Wikipedia's current
    constituents. Portal use reads them cache-only (never fetches during a backtest)."""
    from .http import FetchError, client
    from .identity import IdentityIndex, norm
    st = default_store()
    idx = IdentityIndex(st)
    m = st.read("reference", "sp500", "membership")
    ciks: set = set()
    tickers = {norm(str(b)) for b in m["base"]} | {norm(s) for s in U.MEGACAPS}
    for src in ("yahoo", "dolthub"):
        tickers |= {norm(s) for s in st.symbols("bars_1d", src)}
    for t in tickers:
        e = idx.by_ticker.get(t)
        sib = idx.class_sibling(t) if (e is None or not len(e)) else None   # VIAB -> VIA (identity.py)
        if sib:
            e = idx.by_ticker.get(sib)
        if e is not None and len(e):
            ciks |= {int(c) for c in e["cik"].unique()}
        c = idx.sec_now.get(t)
        if c:
            ciks.add(int(c))
    sm = st.read("reference", "sec", "security_master")
    if len(sm):
        ciks |= {int(c) for c in pd.to_numeric(sm["cik"], errors="coerce").dropna()}
    cur = st.read("reference", "sp500", "current_constituents")
    if len(cur):
        ciks |= {int(c) for c in pd.to_numeric(cur["cik"], errors="coerce").dropna()}
    got = missing = 0
    for i, c in enumerate(sorted(ciks)):
        try:
            b = client().get(f"https://data.sec.gov/submissions/CIK{c:010d}.json", allow_404=True, ttl=None)
            got += b is not None
            missing += b is None
        except FetchError as e:
            missing += 1
            log.warning("sec submissions CIK %s: %s", c, e)
        if i % 500 == 0:
            log.info("sec submissions %d/%d", i, len(ciks))
    return {"ciks": len(ciks), "cached": got, "missing": missing}


def task_option_rekey(symbols=None):
    """Re-key stored option chains to the INTERNAL symbol convention (date = last trading day,
    ARCHITECTURE §3). Only files containing a symbol that differs between the vendor and the
    internal convention (AM-settled index roots, holiday-listed dates) are rewritten; the vendor
    symbol is kept in `vendor_symbol`. Idempotent."""
    import pyarrow.parquet as pq
    from .option_meta import internal_symbols, last_trading_day, to_internal
    st = default_store()
    done = {"files_checked": 0, "files_rekeyed": 0, "rows_rekeyed": 0}
    for src in st.sources("option_chain"):
        for sym in (symbols or st.symbols("option_chain", src)):
            for path in st._files("option_chain", src, sym):
                done["files_checked"] += 1
                cols = pq.read_schema(path).names
                if "vendor_symbol" in cols:
                    continue
                t = pq.read_table(path, columns=[c for c in ("symbol", "expiry", "root", "right", "strike") if c in cols])
                f = t.to_pandas()
                roots = f["root"] if "root" in f.columns else f["symbol"].astype(str).str[:6].str.strip()
                exp, _ = last_trading_day(pd.to_datetime(f["expiry"]), roots)
                new = internal_symbols(roots, exp, f["right"], f["strike"]).to_numpy()
                n = int((new != f["symbol"].astype(str).to_numpy()).sum())
                if not n:
                    continue
                full = pq.read_table(path).to_pandas()
                meta = st._read_meta(path)
                conv = to_internal(full)
                st._upsert(path, conv, meta.get("key") or ["ts", "symbol"],
                           {**meta, "symbol_convention": "internal", "rekeyed_at": pd.Timestamp.now(tz="UTC").isoformat()},
                           replace=True)
                done["files_rekeyed"] += 1
                done["rows_rekeyed"] += n
                log.info("option re-key %s %s %s: %d symbols changed", src, sym, path.name, n)
    return done


# ---------------------------------------------------------------------------
# Alpaca SIP intraday (key-gated: ALPACA_API_KEY / ALPACA_API_SECRET in the environment)
#   python -m atlasbt.data.fetch --set intraday         (resumable; see docs/DATA.md §4.15)
# ---------------------------------------------------------------------------
#: ETFs traded intraday by the sheet: ETF_CORE (S001-S030, S116) + the calendar/event ETF set (S031-S037)
ALPACA_ETFS = ["SPY", "QQQ", "IWM", "DIA"] + list(U.SECTOR_SPDRS) + ["TLT", "GLD", "EFA", "EEM"]
ALPACA_ETF_CORE = ["SPY", "QQQ", "IWM", "DIA"]
ALPACA_START = "2016-01-04"            # first SIP session Alpaca serves
ALPACA_SP500_5M_START = "2017-11-01"   # S039/S040 cannot start before the S&P 500 PIT coverage date (2018-02-01)
ALPACA_WARMUP_SESSIONS = 30            # intraday history before a first top-N session (20-session RVOL/ToD profiles + ATR)
ALPACA_TAIL_SESSIONS = 5
BUDGET_1M = float(os.environ.get("ATLASBT_ALPACA_1M_GB", "5.5")) * 1e9
BUDGET_5M = float(os.environ.get("ATLASBT_ALPACA_5M_GB", "2.5")) * 1e9
BUDGET_QUOTES = float(os.environ.get("ATLASBT_ALPACA_QUOTES_GB", "3.0")) * 1e9


def _alpaca_ready() -> bool:
    from .providers import alpaca
    if not alpaca.available():
        log.error("alpaca: ALPACA_API_KEY / ALPACA_API_SECRET not set in the environment -> skipped")
        return False
    return True


def _sessions(start, end=None) -> pd.DatetimeIndex:
    from . import frames
    sch = frames.schedule()
    now = pd.Timestamp.now(tz="UTC")
    s = sch.index[(sch.index >= pd.Timestamp(start)) & (sch["close"] <= now)]
    return s[s <= pd.Timestamp(end)] if end is not None else s


def task_alpaca_universe(symbols=None):
    """PIT S&P 500 members (rank 0) and a SUPERSET of SP500_TOP50_DV: the top-70 by mean raw close x volume
    over the 20 sessions strictly before each session (>= 10 present), from each member's identity-verified
    daily series (ranks 1-70) -> reference/alpaca/universe_pit, incremental. Vectorised re-implementation of
    Portal._top_dollar_volume (which costs ~2-4 s per session): checked against the exact Portal ranking on
    1,277 sessions 2016-01-04..2021-01-05 -- every exact top-50 member is inside this top-60 on every session,
    and 2 exact top-60 symbol-sessions fall outside this top-70."""
    from . import frames
    from .portal import Portal
    st = default_store()
    have = st.read("reference", "alpaca", "universe_pit")
    done = set(pd.to_datetime(have.loc[have["rank"] > 0, "date"]).dt.normalize()) if len(have) else set()
    sess = _sessions(ALPACA_START)
    todo = [d for d in sess if d not in done]
    if not todo:
        return {"sessions_added": 0}
    p = Portal()
    sch = frames.schedule()
    members = {d: p.universe("SP500", sch.loc[d, "open"] + pd.Timedelta(minutes=30)) for d in todo}
    now = pd.Timestamp.now(tz="UTC")
    dv = {}
    for i, s_ in enumerate(sorted({x for v in members.values() for x in v})):
        try:
            f, allowed, _ = p._window_identity(s_, pd.Timestamp(min(todo)).tz_localize("UTC") - pd.Timedelta(days=60), now)
        except Exception:
            continue
        if not len(f):
            continue
        ser = pd.Series((f["close"] * f["volume"]).to_numpy(), index=frames.to_naive_date(f.index))
        if allowed is not None:
            ser = ser[ser.index.isin(allowed)]
        dv[s_] = ser[~ser.index.duplicated(keep="last")]
        if i % 100 == 99:
            p.clear_cache()
    pos = {d: i for i, d in enumerate(sch.index)}
    rows = []
    for d in todo:
        i = pos[d]
        lo, hi = sch.index[i - 20], sch.index[i - 1]
        sc = {}
        for s_ in members[d]:
            ser = dv.get(s_)
            if ser is None:
                continue
            w = ser[(ser.index >= lo) & (ser.index <= hi)]
            if len(w) >= 10:
                sc[s_] = float(w.mean())
        rows += [(d, s_, k + 1) for k, s_ in enumerate(sorted(sc, key=lambda k: -sc[k])[:70])]
        rows += [(d, s_, 0) for s_ in members[d]]
    st.write("reference", "alpaca", "universe_pit", pd.DataFrame(rows, columns=["date", "symbol", "rank"]),
             key=["date", "symbol", "rank"], extra={"note": "rank 1-70 = top-70 S&P 500 by 20d dollar volume "
                                                            "(superset of SP500_TOP50_DV); rank 0 = S&P 500 member"})
    return {"sessions_added": len(todo)}


def _member_windows(dates: pd.DatetimeIndex, before: int, after: int, floor=None) -> pd.DatetimeIndex:
    """Sessions within [d - before, d + after] sessions of any member date d."""
    allsess = _sessions(floor or ALPACA_START)
    pos = allsess.get_indexer(pd.DatetimeIndex(dates))
    pos = pos[pos >= 0]
    keep = np.zeros(len(allsess), bool)
    for k in pos:
        keep[max(0, k - before):k + after + 1] = True
    return allsess[keep]


def _alpaca_wanted(kind: str) -> dict:
    st = default_store()
    if kind == "etf":
        s = _sessions(ALPACA_START)
        return {e: s for e in ALPACA_ETFS}
    u = st.read("reference", "alpaca", "universe_pit")
    if not len(u):
        raise RuntimeError("run --only alpaca_universe first")
    u["date"] = pd.to_datetime(u["date"])
    if kind == "top":
        g = u[(u["rank"] >= 1) & (u["rank"] <= 60)].groupby("symbol")["date"]   # contains the exact top-50
        return {s: _member_windows(d.unique(), ALPACA_WARMUP_SESSIONS, ALPACA_TAIL_SESSIONS) for s, d in g}
    if kind == "sp500":
        g = u[(u["rank"] == 0) & (u["date"] >= pd.Timestamp(ALPACA_SP500_5M_START))].groupby("symbol")["date"]
        return {s: _member_windows(d.unique(), ALPACA_WARMUP_SESSIONS, ALPACA_TAIL_SESSIONS,
                                   floor=ALPACA_SP500_5M_START) for s, d in g}
    raise KeyError(kind)


def _alpaca_routes(kind: str, symbols=None) -> pd.DataFrame:
    """Tape routes for the wanted (symbol, session)s; cached in reference/alpaca/tape_routes_<kind>."""
    from .portal import Portal
    from .providers import alpaca
    st = default_store()
    wanted = _alpaca_wanted(kind)
    if symbols:
        wanted = {k: v for k, v in wanted.items() if k in symbols}
    have = st.read("reference", "alpaca", f"tape_routes_{kind}")
    if len(have):
        have["date"] = pd.to_datetime(have["date"])
        hv = have.groupby("symbol")["date"].agg(["min", "max"])
        # only symbols whose wanted window extends beyond what was routed before are re-routed
        # (daily references lag the calendar by a few sessions: a re-route needs > 5 new days)
        wanted = {k: v for k, v in wanted.items() if k not in hv.index
                  or v.max() > hv.loc[k, "max"] + pd.Timedelta(days=5) or v.min() < hv.loc[k, "min"] - pd.Timedelta(days=5)}
    if wanted:
        routes, summ = alpaca.route_tapes(Portal(), wanted)
        if len(routes):
            st.write("reference", "alpaca", f"tape_routes_{kind}", routes, key=["symbol", "date"],
                     extra={"note": "tape ticker per (universe symbol, session): identity-verified daily close "
                                    "within 0.5% and volume within 2.5x of Alpaca's 1Day bar"})
        st.write("reference", "alpaca", f"route_summary_{kind}", summ.assign(at=pd.Timestamp.now(tz="UTC")),
                 key=["symbol"])
        bad = summ[summ["routed"] < 0.95 * summ["reference"]]
        for r in bad.itertuples():
            log.warning("route %s: only %d of %d verified sessions matched an Alpaca tape (%s)", r.symbol,
                        r.routed, r.reference, r.tapes)
    out = st.read("reference", "alpaca", f"tape_routes_{kind}")
    if len(out):
        out["date"] = pd.to_datetime(out["date"])
        if symbols:
            out = out[out["symbol"].isin(symbols)]
    return out


def task_alpaca_bars_etf(symbols=None):
    """SIP 1m bars (PRE+RTH) 2016-> for the intraday ETF set (ETF_CORE + sector SPDRs + TLT/GLD/EFA/EEM)."""
    if not _alpaca_ready():
        return {"skipped": "no key"}
    from .providers import alpaca
    return alpaca.fetch_routed_bars(_alpaca_routes("etf", symbols), "1m", ("PRE", "RTH"),
                                    budget_bytes=BUDGET_1M, chunk=4, workers=3)


def task_alpaca_bars_top(symbols=None):
    """SIP 1m bars (PRE+RTH) of every stock ever in the PIT top-70 by dollar volume (SP500_TOP50_DV + margin),
    on its member sessions +- warm-up, identity-routed."""
    if not _alpaca_ready():
        return {"skipped": "no key"}
    from .providers import alpaca
    return alpaca.fetch_routed_bars(_alpaca_routes("top", symbols), "1m", ("PRE", "RTH"),
                                    budget_bytes=BUDGET_1M, chunk=6, workers=3)


def task_alpaca_bars_sp500_5m(symbols=None):
    """SIP 5m RTH bars of every PIT S&P 500 member from 2017-11 (S039/S040), identity-routed."""
    if not _alpaca_ready():
        return {"skipped": "no key"}
    from .providers import alpaca
    return alpaca.fetch_routed_bars(_alpaca_routes("sp500", symbols), "5m", ("RTH",),
                                    budget_bytes=BUDGET_5M, chunk=25, workers=3, slim=True,
                                    newest_first=True)     # 2023+ first: S158-S160 (options) need it soonest


def alpaca_quote_plan() -> pd.DataFrame:
    """[symbol, date, tape, window] in priority order (testability windows first)."""
    st = default_store()
    etf = st.read("reference", "alpaca", "tape_routes_etf")
    top = st.read("reference", "alpaca", "tape_routes_top")
    u = st.read("reference", "alpaca", "universe_pit")
    parts = []
    if len(top) and len(u):
        top["date"] = pd.to_datetime(top["date"])
        u["date"] = pd.to_datetime(u["date"])
        m = u[(u["rank"] >= 1) & (u["rank"] <= 60)][["symbol", "date"]]     # superset of the exact top-50
        parts.append(top.merge(m, on=["symbol", "date"]).assign(window="pre_0920"))
    if len(etf):
        etf["date"] = pd.to_datetime(etf["date"])
        core = etf[etf["symbol"].isin(ALPACA_ETF_CORE)]
        for w in ("close_m36", "close_m30", "close_m5"):
            parts.append(core.assign(window=w))
        for w in ("close_m10", "open_p5"):
            parts.append(etf.assign(window=w))
    if not parts:
        return pd.DataFrame(columns=["symbol", "date", "tape", "window"])
    return pd.concat(parts, ignore_index=True)[["symbol", "date", "tape", "window"]]


def task_alpaca_quotes(symbols=None):
    """Cleaned SIP NBBO in the strategies' decision windows (alpaca.QUOTE_WINDOWS), priority order."""
    if not _alpaca_ready():
        return {"skipped": "no key"}
    from .providers import alpaca
    plan = alpaca_quote_plan()
    if symbols:
        plan = plan[plan["symbol"].isin(symbols)]
    out = {}
    for w in ("pre_0920", "close_m36", "close_m30", "close_m5", "close_m10", "open_p5"):
        sub = plan[plan["window"] == w]
        if len(sub):
            out[w] = alpaca.fetch_quote_windows(sub, budget_bytes=BUDGET_QUOTES, workers=3)
            if out[w].get("stopped"):
                break
    return out


def task_databento_options(symbols=None):
    """Databento OPRA option quotes (key-gated, budget-guarded, resumable; docs/DATA.md §4.15c).
    Env: ATLASBT_DATABENTO_PLANS (default spx0dte,spxeod,earnings,s160), ATLASBT_DATABENTO_START /
    _END (default 2023-03-28 .. yesterday), ATLASBT_DATABENTO_DRY_RUN=1 (price only),
    DATABENTO_BUDGET_USD (cumulative cap, default 90; ledger cache/raw/databento_ledger.csv)."""
    from .providers import databento
    plans = [p.strip() for p in os.environ.get("ATLASBT_DATABENTO_PLANS", "spx0dte,spxeod,earnings,s160").split(",")
             if p.strip()]
    return databento.task(plans, os.environ.get("ATLASBT_DATABENTO_START", "2023-03-28"),
                          os.environ.get("ATLASBT_DATABENTO_END") or None,
                          dry_run=os.environ.get("ATLASBT_DATABENTO_DRY_RUN") == "1", log=log.info)


TASKS: dict[str, Callable] = {
    "daily_etf": task_daily_etf, "intraday": task_intraday, "cboe": task_cboe, "fred": task_fred,
    "french": task_french, "sp500": task_sp500, "daily_sp500": task_daily_sp500, "sec": task_sec,
    "fomc": task_fomc, "earnings": task_earnings, "options": task_options,
    "nasdaqtrader": task_nasdaqtrader, "borrow": task_borrow, "dukascopy": task_dukascopy,
    "dukascopy_backfill": task_dukascopy_backfill, "options_bulk": task_options_bulk,
    "ohlcv_bulk": task_ohlcv_bulk, "ziplime": task_ziplime, "dolthub_tables": task_dolthub_tables,
    "iex_sample": task_iex_sample, "hod": task_hod, "identity": task_identity, "option_rekey": task_option_rekey,
    "sec_submissions": task_sec_submissions,
    "dividend_decl": task_dividend_decl, "buybacks": task_buybacks,
    "alpaca_eod": task_alpaca_eod,          # survivorship patch (key-gated); then --only identity
    "tiingo_eod": task_tiingo_eod,          # survivorship patch < 2016 (key-gated); then --only identity
    # Alpaca SIP intraday (key-gated; --set intraday)
    "alpaca_universe": task_alpaca_universe, "alpaca_bars_etf": task_alpaca_bars_etf,
    "alpaca_bars_top": task_alpaca_bars_top, "alpaca_quotes": task_alpaca_quotes,
    "alpaca_bars_sp500_5m": task_alpaca_bars_sp500_5m,
    "databento_options": task_databento_options,     # OPRA quotes (key-gated, budget-guarded)
}
SETS = {
    # time-sensitive first (Yahoo 1m keeps only ~30 days)
    "core": ["intraday", "daily_etf", "cboe", "fred", "french", "fomc", "nasdaqtrader", "sp500",
             "daily_sp500", "ziplime", "dolthub_tables", "hod", "iex_sample", "sec", "sec_submissions",
             "borrow", "dukascopy", "options", "identity"],
    # bulk mirrors: set ATLASBT_OPTION_CSV / ATLASBT_OHLCV_CSV / ATLASBT_EPS_ESTIMATE_CSV to the
    # downloaded HF CSVs (see docs/DATA.md), then run --set bulk
    "bulk": ["ohlcv_bulk", "options_bulk", "dolthub_tables"],
    "daily": ["intraday", "daily_etf", "cboe", "fred", "fomc", "nasdaqtrader", "sp500", "options"],
    "quick": ["daily_etf", "cboe", "fred", "french"],
    # Alpaca SIP intraday (needs ALPACA_API_KEY / ALPACA_API_SECRET; resumable, ledgers in cache/logs/)
    "intraday": ["alpaca_universe", "alpaca_bars_etf", "alpaca_bars_top", "alpaca_quotes", "alpaca_bars_sp500_5m"],
}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m atlasbt.data.fetch")
    p.add_argument("--set", default=None, choices=sorted(SETS))
    p.add_argument("--only", default=None, help="comma-separated task names: " + ",".join(TASKS))
    p.add_argument("--symbols", default=None, help="comma-separated symbols override")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    tasks = a.only.split(",") if a.only else SETS[a.set or "core"]
    syms = [s.strip().upper() for s in a.symbols.split(",")] if a.symbols else None
    logdir = config.CACHE_ROOT / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    tag = "_".join(tasks) if len(tasks) < 3 else (a.set or "multi")
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(logdir / f"fetch_{tag}.log"), logging.StreamHandler(sys.stdout)])
    for noisy in ("yfinance", "urllib3", "peewee"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)
    summary = {}
    for t in tasks:
        t0 = time.time()
        log.info("=== task %s ===", t)
        try:
            res = TASKS[t](syms)
        except Exception as e:
            log.error("task %s failed: %s\n%s", t, e, traceback.format_exc())
            res = [{"error": str(e)}]
        summary[t] = {"seconds": round(time.time() - t0, 1), "results": res}
        (logdir / f"summary_{t}.json").write_text(json.dumps(summary[t], default=str, indent=1))
    log.info("done: %s", {k: v["seconds"] for k, v in summary.items()})


if __name__ == "__main__":
    main()
