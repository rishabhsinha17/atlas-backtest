"""Alpaca Market Data v2 (UNVERIFIED -- written from docs.alpaca.markets/reference/stockbars,
/stockquotes; no key available to live-test).

Env: ALPACA_API_KEY, ALPACA_API_SECRET.  Headers APCA-API-KEY-ID / APCA-API-SECRET-KEY.
  GET https://data.alpaca.markets/v2/stocks/{symbol}/bars
      ?timeframe=1Min&start=RFC3339&end=RFC3339&limit=10000&adjustment=raw&feed=sip&page_token=
      -> {"bars":[{"t","o","h","l","c","v","n","vw"}], "next_page_token"}   (t = bar START, UTC)
  GET https://data.alpaca.markets/v2/stocks/{symbol}/quotes?start&end&limit=10000&feed=sip&page_token=
      -> {"quotes":[{"t","bx","bp","bs","ax","ap","as","c","z"}], "next_page_token"}
Free plan: historical SIP (consolidated) since 2016 except the most recent 15 minutes.
Fidelity: bars T1_BAR (SIP consolidated incl. extended hours); quotes T0_QUOTE (NBBO-ish:
SIP quotes are per-exchange BBO updates; NBBO must be rebuilt -- we store the raw stream).
Sizes: Alpaca reports quote sizes in round lots (100 shares) for SIP -> multiplied by 100.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import pandas as pd

from atlasbt.core.types import Fidelity

from ..http import client
from ._keyed import env, skipped, store_intraday, store_quotes

log = logging.getLogger("atlasbt.data.alpaca")
SOURCE = "alpaca"
BASE = "https://data.alpaca.markets/v2/stocks/{symbol}/{kind}"
KEYS = ("ALPACA_API_KEY", "ALPACA_API_SECRET")


def available() -> bool:
    return env(*KEYS) is not None


def _headers():
    k = env(*KEYS)
    return {"APCA-API-KEY-ID": k["ALPACA_API_KEY"], "APCA-API-SECRET-KEY": k["ALPACA_API_SECRET"]}


def _pages(symbol: str, kind: str, params: dict):
    token = None
    while True:
        p = dict(params, **({"page_token": token} if token else {}))
        body = client().get(BASE.format(symbol=symbol.upper(), kind=kind), params=p, headers=_headers(),
                            cache_extra="alpaca")
        d = json.loads(body)
        yield d.get(kind) or []
        token = d.get("next_page_token")
        if not token:
            break


def parse_bars(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex(pd.to_datetime(df["t"], utc=True), name="start")
    return df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
                              "vw": "vwap", "n": "trades"})[["open", "high", "low", "close", "volume", "vwap", "trades"]]


def parse_quotes(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = pd.DataFrame({"bid": df["bp"].astype(float).to_numpy(), "ask": df["ap"].astype(float).to_numpy(),
                        "bid_size": df["bs"].astype(float).to_numpy() * 100,
                        "ask_size": df["as"].astype(float).to_numpy() * 100,
                        "bid_exchange": df["bx"].to_numpy() if "bx" in df else None,
                        "ask_exchange": df["ax"].to_numpy() if "ax" in df else None},
                       index=pd.DatetimeIndex(pd.to_datetime(df["t"], utc=True), name="ts"))
    return out


def fetch_bars(symbol: str, start, end, freq: str = "1m", feed: str = "sip") -> dict:
    if not available():
        return skipped("alpaca")
    tf = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "60m": "1Hour"}[freq]
    params = {"timeframe": tf, "start": pd.Timestamp(start).isoformat(), "end": pd.Timestamp(end).isoformat(),
              "limit": 10000, "adjustment": "raw", "feed": feed}
    frames_ = [parse_bars(rows) for rows in _pages(symbol, "bars", params)]
    df = pd.concat([f for f in frames_ if len(f)]) if any(len(f) for f in frames_) else pd.DataFrame()
    n = store_intraday(SOURCE, symbol, freq, df, Fidelity.T1_BAR, extra={"feed": feed}) if len(df) else 0
    return {"symbol": symbol, "rows": n}


def fetch_quotes(symbol: str, start, end, feed: str = "sip") -> dict:
    if not available():
        return skipped("alpaca")
    params = {"start": pd.Timestamp(start).isoformat(), "end": pd.Timestamp(end).isoformat(), "limit": 10000,
              "feed": feed}
    total = 0
    for rows in _pages(symbol, "quotes", params):
        q = parse_quotes(rows)
        if len(q):
            total = store_quotes(SOURCE, symbol, q, Fidelity.T0_QUOTE, extra={"feed": feed})
    return {"symbol": symbol, "rows": total}
