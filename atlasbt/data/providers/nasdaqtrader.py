"""Nasdaq Trader symbol directories (current snapshot only; NOT point-in-time).

  https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt
      Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
  https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt
      ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
Last line: 'File Creation Time: mmddyyyyhh:mm|...'. Exchange codes (otherlisted):
A=NYSE American, N=NYSE, P=NYSE Arca, Z=Cboe BZX, V=IEX.
Each fetch is stored as a dated snapshot (reference/nasdaqtrader/directory keyed by
snapshot_date+symbol) so listing status accumulates over time going forward.
"""
from __future__ import annotations

import io

import pandas as pd

from ..http import client
from ..store import ParquetStore, default_store

SOURCE = "nasdaqtrader"
URLS = {"nasdaq": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "other": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"}
EXCH = {"A": "NYSE American", "N": "NYSE", "P": "NYSE Arca", "Z": "Cboe BZX", "V": "IEX", "Q": "Nasdaq"}


def parse(kind: str, content: bytes) -> tuple[pd.DataFrame, str]:
    text = content.decode("utf-8", "ignore")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    footer = lines[-1] if lines[-1].startswith("File Creation Time") else ""
    df = pd.read_csv(io.StringIO("\n".join(lines[:-1] if footer else lines)), sep="|", dtype=str)
    if kind == "nasdaq":
        out = pd.DataFrame({"symbol": df["Symbol"], "name": df["Security Name"], "exchange": "Nasdaq",
                            "etf": df["ETF"].eq("Y"), "test_issue": df["Test Issue"].eq("Y"),
                            "round_lot": pd.to_numeric(df["Round Lot Size"], errors="coerce"),
                            "market_category": df["Market Category"], "financial_status": df["Financial Status"]})
    else:
        out = pd.DataFrame({"symbol": df["ACT Symbol"], "name": df["Security Name"],
                            "exchange": df["Exchange"].map(EXCH).fillna(df["Exchange"]), "etf": df["ETF"].eq("Y"),
                            "test_issue": df["Test Issue"].eq("Y"),
                            "round_lot": pd.to_numeric(df["Round Lot Size"], errors="coerce"),
                            "market_category": None, "financial_status": None})
    out = out[out["symbol"].notna()]
    return out, footer


def fetch_directory(store: ParquetStore | None = None, ttl_hours: float = 20) -> dict:
    store = store or default_store()
    parts = []
    stamp = None
    for kind, url in URLS.items():
        df, footer = parse(kind, client().get(url, ttl=ttl_hours * 3600))
        parts.append(df)
        stamp = stamp or footer
    df = pd.concat(parts, ignore_index=True)
    df = df[~df["test_issue"]]
    df["snapshot_date"] = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    df["file_creation"] = stamp
    store.write("reference", SOURCE, "directory", df, key=["snapshot_date", "symbol"])
    return {"symbols": len(df), "etfs": int(df["etf"].sum()), "snapshot": str(df["snapshot_date"].iloc[0].date())}
