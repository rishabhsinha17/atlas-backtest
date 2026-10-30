"""Databento historical API (UNVERIFIED; from databento.com/docs/api-reference-historical).

Env: DATABENTO_API_KEY (HTTP basic auth: key as username, empty password). New accounts get
free credits; ALWAYS check cost first (metadata.get_cost) -- we refuse requests above
`max_cost_usd`.
  GET https://hist.databento.com/v0/metadata.get_cost?dataset&symbols&schema&start&end&stype_in
  GET https://hist.databento.com/v0/timeseries.get_range
      ?dataset=XNAS.ITCH&symbols=SPY&schema=ohlcv-1m|tbbo|bbo-1m|mbp-1&start=ISO&end=ISO
       &encoding=csv&pretty_px=true&pretty_ts=true&map_symbols=true&stype_in=raw_symbol
  ohlcv-1m csv: ts_event (bar open), rtype, publisher_id, instrument_id, open, high, low, close,
  volume[, symbol].  bbo-1m: ts_recv, ..., bid_px_00, ask_px_00, bid_sz_00, ask_sz_00, ...
Fidelity: XNAS.ITCH is a single venue (Nasdaq) book -> bars T1_BAR (venue-only volume!),
quotes T2_PROXY (venue BBO, not NBBO). A consolidated dataset (e.g. EQUS.*) would be T0.
"""
from __future__ import annotations

import io
import json
import os

import pandas as pd
import requests

from atlasbt.core.types import Fidelity

from .. import config
from ._keyed import env, skipped, store_intraday, store_quotes

SOURCE = "databento"
BASE = "https://hist.databento.com/v0/"


def available() -> bool:
    return env("DATABENTO_API_KEY") is not None


def _get(endpoint: str, params: dict) -> bytes:
    key = env("DATABENTO_API_KEY")["DATABENTO_API_KEY"]
    r = requests.get(BASE + endpoint, params=params, auth=(key, ""), timeout=300,
                     headers={"User-Agent": config.USER_AGENT})
    r.raise_for_status()
    return r.content


def cost(dataset: str, symbol: str, schema: str, start, end) -> float:
    return float(json.loads(_get("metadata.get_cost", {"dataset": dataset, "symbols": symbol, "schema": schema,
                                                        "start": pd.Timestamp(start).isoformat(),
                                                        "end": pd.Timestamp(end).isoformat(),
                                                        "stype_in": "raw_symbol"})))


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
    if not available():
        return skipped("databento")
    c = cost(dataset, symbol, schema, start, end)
    if c > max_cost_usd:
        return {"symbol": symbol, "refused": f"cost ${c:.2f} > max ${max_cost_usd:.2f}"}
    content = _get("timeseries.get_range", {"dataset": dataset, "symbols": symbol, "schema": schema,
                                            "start": pd.Timestamp(start).isoformat(), "end": pd.Timestamp(end).isoformat(),
                                            "encoding": "csv", "pretty_px": "true", "pretty_ts": "true",
                                            "map_symbols": "true", "stype_in": "raw_symbol"})
    if schema.startswith("ohlcv"):
        freq = {"ohlcv-1m": "1m", "ohlcv-1h": "60m"}[schema]
        n = store_intraday(SOURCE, symbol, freq, parse_ohlcv_csv(content), Fidelity.T1_BAR, extra={"dataset": dataset})
    else:
        n = store_quotes(SOURCE, symbol, parse_bbo_csv(content), Fidelity.T2_PROXY, extra={"dataset": dataset})
    return {"symbol": symbol, "rows": n, "cost_usd": c}
