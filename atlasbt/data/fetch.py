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
import sys
import time
import traceback
from typing import Callable

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
    from .providers import borrow
    return [borrow.probe_iborrowdesk(symbols or ["SPY", "AAPL"])]


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


def task_identity(symbols=None):
    """Build + persist identity-verified daily series for every S&P 500 member interval (resolved
    at the interval's start, middle and end) and the megacaps."""
    from .portal import Portal
    p = Portal()
    m = default_store().read("reference", "sp500", "membership")
    jobs = []
    for r in m.to_dict("records"):
        t = str(r["base"]).replace(".", "-")
        s0 = max(pd.Timestamp(r["start"]), pd.Timestamp("1996-01-02"))
        e0 = pd.Timestamp(r["end"]) if pd.notna(r["end"]) else pd.Timestamp.now().normalize()
        for d in {s0, s0 + (e0 - s0) / 2, e0 - pd.Timedelta(days=1)}:
            jobs.append((t, d))
    jobs += [(s, None) for s in U.MEGACAPS]
    if symbols:
        jobs = [j for j in jobs if j[0] in symbols]
    out = []
    for i, (t, d) in enumerate(jobs):
        try:
            out.append(p.persist_identity(t, d))
        except Exception as e:
            log.error("identity %s %s: %s", t, d, e)
        if i % 200 == 0:
            log.info("identity %d/%d", i, len(jobs))
    return out


TASKS: dict[str, Callable] = {
    "daily_etf": task_daily_etf, "intraday": task_intraday, "cboe": task_cboe, "fred": task_fred,
    "french": task_french, "sp500": task_sp500, "daily_sp500": task_daily_sp500, "sec": task_sec,
    "fomc": task_fomc, "earnings": task_earnings, "options": task_options,
    "nasdaqtrader": task_nasdaqtrader, "borrow": task_borrow, "dukascopy": task_dukascopy,
    "dukascopy_backfill": task_dukascopy_backfill, "options_bulk": task_options_bulk,
    "ohlcv_bulk": task_ohlcv_bulk, "ziplime": task_ziplime, "dolthub_tables": task_dolthub_tables,
    "iex_sample": task_iex_sample, "hod": task_hod, "identity": task_identity,
}
SETS = {
    # time-sensitive first (Yahoo 1m keeps only ~30 days)
    "core": ["intraday", "daily_etf", "cboe", "fred", "french", "fomc", "nasdaqtrader", "sp500",
             "daily_sp500", "ziplime", "dolthub_tables", "hod", "iex_sample", "sec", "borrow", "dukascopy",
             "options"],
    # bulk mirrors: set ATLASBT_OPTION_CSV / ATLASBT_OHLCV_CSV / ATLASBT_EPS_ESTIMATE_CSV to the
    # downloaded HF CSVs (see docs/DATA.md), then run --set bulk
    "bulk": ["ohlcv_bulk", "options_bulk", "dolthub_tables"],
    "daily": ["intraday", "daily_etf", "cboe", "fred", "fomc", "nasdaqtrader", "sp500", "options"],
    "quick": ["daily_etf", "cboe", "fred", "french"],
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
