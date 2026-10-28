"""Cboe index histories: https://cdn.cboe.com/api/global/us_indices/daily_prices/<SYM>_History.csv

Two layouts: 'DATE,OPEN,HIGH,LOW,CLOSE' (VIX family) and 'DATE,<SYM>' (single value:
SKEW, PUT, BXM, CNDR, BFLY, ...). Single-value series are stored with
open=high=low=close=value (only the close is real) and volume NaN.

PIT: Cboe volatility indices are disseminated until 16:15 ET (SPX options trade to
16:15); the strategy benchmark indices (BXM, PUT, ...) are computed from closing
values. known_at := session close + 15 min (16:15 ET; 13:15 on early closes).
Fidelity T1_BAR (official values; daily granularity only).
"""
from __future__ import annotations

import io
import logging
from typing import Optional

import pandas as pd

from atlasbt.core.types import Fidelity

from .. import frames
from ..http import client
from ..store import ParquetStore, default_store

log = logging.getLogger("atlasbt.data.cboe")
SOURCE = "cboe"
URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
KNOWN_LAG = pd.Timedelta(minutes=15)

DEFAULT_INDICES = ["VIX", "VIX9D", "VIX3M", "VIX6M", "VIX1Y", "VIX1D", "VVIX", "SKEW", "VXN", "RVX", "VXD",
                   "GVZ", "OVX", "VXEEM", "VXEFA", "VXTLT", "PUT", "BXM", "BXMD", "BXMW", "CNDR", "BFLY",
                   "CLL", "CMBO", "PPUT", "PUTY", "RXM", "BXY", "COR1M", "COR3M", "DSPX", "SPX", "OEX",
                   "XEO", "RUT", "DJX", "VXAPL", "VXAZN", "VXGOG", "VXGS", "VXIBM", "VXGDX", "VXSLV"]


def parse_csv(sym: str, content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content))
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
    df = df.set_index("DATE").sort_index()
    if {"OPEN", "HIGH", "LOW", "CLOSE"} <= set(df.columns):
        o = df[["OPEN", "HIGH", "LOW", "CLOSE"]].rename(columns=str.lower)
        single = False
    else:
        col = [c for c in df.columns if c != "DATE"][0]
        v = pd.to_numeric(df[col], errors="coerce")
        o = pd.DataFrame({"open": v, "high": v, "low": v, "close": v}, index=df.index)
        single = True
    o = o.apply(pd.to_numeric, errors="coerce")
    o = o[o["close"].notna()]
    # Cboe fills OHLC with 0 for some early VIX days -> treat as missing (close only)
    for c in ("open", "high", "low"):
        bad = (o[c] <= 0) | o[c].isna()
        o.loc[bad, c] = o.loc[bad, "close"]
    o["volume"] = float("nan")
    bars, dropped = frames.daily_bars(o)
    bars["known_at"] = bars["known_at"] + KNOWN_LAG
    bars.attrs["single_value"] = single
    bars.attrs["dropped"] = list(dropped)
    return bars


def fetch_index(sym: str, store: Optional[ParquetStore] = None, ttl_hours: float = 12) -> dict:
    store = store or default_store()
    content = client().get(URL.format(sym=sym.upper()), ttl=ttl_hours * 3600)
    bars = parse_csv(sym, content)
    store.write("bars_1d", SOURCE, sym, bars, fidelity=int(Fidelity.T1_BAR), replace=True,
                extra={"single_value": bars.attrs["single_value"], "first_ts": str(bars.index[0]),
                       "last_ts": str(bars.index[-1]), "known_at_rule": "close+15min"})
    return {"symbol": sym, "rows": len(bars), "first": str(bars.index[0].date()),
            "last": str(bars.index[-1].date()), "single_value": bars.attrs["single_value"],
            "dropped_non_sessions": len(bars.attrs["dropped"])}
