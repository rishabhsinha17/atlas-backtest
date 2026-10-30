"""Alpha Vantage (UNVERIFIED; from alphavantage.co/documentation).

Env: ALPHAVANTAGE_API_KEY.  Free tier: 25 requests/day.
  GET https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=SPY&interval=1min
      &month=YYYY-MM&outputsize=full&extended_hours=true&adjusted=false&datatype=csv&apikey=KEY
      -> csv: timestamp (US/Eastern, bar START), open, high, low, close, volume
  One month of 1-min bars per request, history back to 2000-01. adjusted=false -> raw.
Fidelity T1_BAR (consolidated per vendor; not independently verified).
"""
from __future__ import annotations

import io

import pandas as pd

from atlasbt.core.types import Fidelity

from ..http import client
from ._keyed import env, skipped, store_intraday

SOURCE = "alphavantage"
URL = "https://www.alphavantage.co/query"


def available() -> bool:
    return env("ALPHAVANTAGE_API_KEY") is not None


def parse_csv(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content))
    if "timestamp" not in df:
        raise ValueError(f"unexpected Alpha Vantage response: {content[:200]!r}")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"]).dt.tz_localize("America/New_York"), name="start")
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def fetch_month(symbol: str, month: str, freq: str = "1m") -> dict:
    if not available():
        return skipped("alphavantage")
    key = env("ALPHAVANTAGE_API_KEY")["ALPHAVANTAGE_API_KEY"]
    content = client().get(URL, params={"function": "TIME_SERIES_INTRADAY", "symbol": symbol.upper(),
                                        "interval": freq.replace("m", "min"), "month": month, "outputsize": "full",
                                        "extended_hours": "true", "adjusted": "false", "datatype": "csv",
                                        "apikey": key})
    n = store_intraday(SOURCE, symbol, freq, parse_csv(content), Fidelity.T1_BAR)
    return {"symbol": symbol, "month": month, "rows": n}
