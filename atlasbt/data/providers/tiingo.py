"""Tiingo (UNVERIFIED; from tiingo.com/documentation/end-of-day and /iex).

Env: TIINGO_API_KEY.
  EOD:  GET https://api.tiingo.com/tiingo/daily/{t}/prices?startDate&endDate&format=json&token=
        -> [{"date","open","high","low","close","volume","adjClose",...,"divCash","splitFactor"}]
        (raw OHLC as traded + divCash/splitFactor on ex-dates; covers many DELISTED tickers,
        which makes it the best free survivorship patch for daily bars)
  IEX:  GET https://api.tiingo.com/iex/{t}/prices?startDate&endDate&resampleFreq=1min
             &columns=open,high,low,close,volume&token=
        -> [{"date","open","high","low","close","volume"}]  IEX-only prints (not consolidated): T2_PROXY.
"""
from __future__ import annotations

import json

import pandas as pd

from atlasbt.core.types import Fidelity

from .. import frames
from ..http import client
from ..store import default_store
from ._keyed import env, skipped, store_intraday

SOURCE = "tiingo"


def available() -> bool:
    return env("TIINGO_API_KEY") is not None


def parse_eod(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(rows)
    if df.empty:
        return frames.empty_bars(), pd.DataFrame()
    df.index = frames.to_naive_date(pd.to_datetime(df["date"], utc=True))
    bars, _ = frames.daily_bars(df[["open", "high", "low", "close", "volume"]])
    acts = []
    for d, r in df.iterrows():
        if r.get("splitFactor", 1.0) not in (1.0, 0.0, None):
            acts.append({"type": "SPLIT", "ex_date": d, "ratio": float(r["splitFactor"]), "amount": 0.0})
        if r.get("divCash", 0.0):
            acts.append({"type": "CASH_DIVIDEND", "ex_date": d, "ratio": 1.0, "amount": float(r["divCash"])})
    return bars, pd.DataFrame(acts)


def fetch_daily(symbol: str, start="1990-01-01", end=None) -> dict:
    if not available():
        return skipped("tiingo")
    tok = env("TIINGO_API_KEY")["TIINGO_API_KEY"]
    rows = json.loads(client().get(f"https://api.tiingo.com/tiingo/daily/{symbol.lower()}/prices",
                                   params={"startDate": start, "endDate": end or pd.Timestamp.today().strftime("%Y-%m-%d"),
                                           "format": "json", "token": tok}))
    bars, acts = parse_eod(rows)
    default_store().write("bars_1d", SOURCE, symbol, bars, fidelity=int(Fidelity.T1_BAR), replace=True,
                          extra={"verified": False})
    return {"symbol": symbol, "rows": len(bars), "actions": len(acts)}


def fetch_iex_intraday(symbol: str, start, end, freq: str = "1m") -> dict:
    if not available():
        return skipped("tiingo")
    tok = env("TIINGO_API_KEY")["TIINGO_API_KEY"]
    rows = json.loads(client().get(f"https://api.tiingo.com/iex/{symbol.lower()}/prices",
                                   params={"startDate": str(pd.Timestamp(start).date()), "endDate": str(pd.Timestamp(end).date()),
                                           "resampleFreq": freq.replace("m", "min"),
                                           "columns": "open,high,low,close,volume", "token": tok}))
    df = pd.DataFrame(rows)
    if df.empty:
        return {"symbol": symbol, "rows": 0}
    df.index = pd.DatetimeIndex(pd.to_datetime(df["date"], utc=True), name="start")
    n = store_intraday(SOURCE, symbol, freq, df, Fidelity.T2_PROXY, extra={"venue": "IEX only"})
    return {"symbol": symbol, "rows": n}
