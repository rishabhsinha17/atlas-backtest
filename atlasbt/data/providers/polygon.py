"""Polygon.io aggregates (UNVERIFIED; written from polygon.io/docs/rest/stocks/aggregates/custom-bars).
NOTE: api.polygon.io CONNECT was rejected (403) by this environment's egress proxy on
2026-10-28, so even with a key it cannot run here.

Env: POLYGON_API_KEY.
  GET https://api.polygon.io/v2/aggs/ticker/{T}/range/{mult}/{timespan}/{from}/{to}
      ?adjusted=false&sort=asc&limit=50000&apiKey=KEY
      -> {"results":[{"t" (ms epoch, bar start), "o","h","l","c","v","vw","n"}], "next_url"}
Free tier: 5 requests/min, ~2 years of minute history.
"""
from __future__ import annotations

import json
import time
from typing import Optional

import pandas as pd

from atlasbt.core.types import Fidelity

from ..http import client
from ._keyed import env, skipped, store_intraday

SOURCE = "polygon"
URL = "https://api.polygon.io/v2/aggs/ticker/{t}/range/{m}/{span}/{a}/{b}"


def available() -> bool:
    return env("POLYGON_API_KEY") is not None


def parse_aggs(results: list[dict]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df.index = pd.DatetimeIndex(pd.to_datetime(df["t"], unit="ms", utc=True), name="start")
    return df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap",
                              "n": "trades"})[["open", "high", "low", "close", "volume", "vwap", "trades"]]


def fetch_bars(symbol: str, start, end, freq: str = "1m") -> dict:
    if not available():
        return skipped("polygon")
    key = env("POLYGON_API_KEY")["POLYGON_API_KEY"]
    m, span = {"1m": (1, "minute"), "5m": (5, "minute"), "60m": (1, "hour"), "1d": (1, "day")}[freq]
    url = URL.format(t=symbol.upper(), m=m, span=span, a=pd.Timestamp(start).strftime("%Y-%m-%d"),
                     b=pd.Timestamp(end).strftime("%Y-%m-%d"))
    params: Optional[dict] = {"adjusted": "false", "sort": "asc", "limit": 50000, "apiKey": key}
    parts = []
    while url:
        d = json.loads(client().get(url, params=params))
        parts.append(parse_aggs(d.get("results") or []))
        url = d.get("next_url")
        params = {"apiKey": key} if url else None
        time.sleep(12.5)          # free tier: 5 req/min
    df = pd.concat([p for p in parts if len(p)]) if any(len(p) for p in parts) else pd.DataFrame()
    n = store_intraday(SOURCE, symbol, freq, df, Fidelity.T1_BAR) if len(df) and freq != "1d" else 0
    return {"symbol": symbol, "rows": n}
