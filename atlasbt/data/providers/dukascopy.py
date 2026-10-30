"""Dukascopy datafeed (Swiss FX bank; CFD quotes on US stocks/ETFs/indices).

  https://datafeed.dukascopy.com/datafeed/{INSTR}/{YYYY}/{MM0}/{DD}/{HH}h_ticks.bi5
      MM0 = month index 0-based; HH = UTC hour. LZMA-compressed records of 20 bytes,
      big-endian: uint32 ms-offset-in-hour, uint32 ask, uint32 bid, float32 ask_vol, float32 bid_vol.
  https://datafeed.dukascopy.com/datafeed/{INSTR}/{YYYY}/{MM0}/{DD}/{BID|ASK}_candles_min_1.bi5
      per UTC day, 24-byte records: uint32 sec-offset, uint32 open, close, low, high, float32 vol.
Prices are integers / 1000 for US stock CFDs and index CFDs (verified: QQQUSUSD 2024-01-16
14:30Z ask 408.324 vs QQQ trading ~408).
Instruments: <TICKER>USUSD (e.g. SPYUSUSD, QQQUSUSD, IWMUSUSD, XLKUSUSD, AAPLUSUSD) quoted
during US RTH only; index CFDs USA500IDXUSD (S&P 500), USATECHIDXUSD (Nasdaq 100),
USA30IDXUSD (Dow) quoted ~23h.

FIDELITY: these are Dukascopy's OWN CFD bid/ask quotes, not the consolidated NBBO and not
exchange prints: spreads/levels can differ from the US market (especially at the open/close
and for single stocks); volumes are Dukascopy's (meaningless for US ADV). -> T2_PROXY.
Availability: the server frequently throttles/timeouts (HTTP 503 / no response); we back off.
"""
from __future__ import annotations

import logging
import lzma
from typing import Optional

import numpy as np
import pandas as pd

from atlasbt.core.types import Fidelity

from .. import frames
from ..http import FetchError, client
from ..store import ParquetStore, default_store

log = logging.getLogger("atlasbt.data.dukascopy")
SOURCE = "dukascopy"
BASE = "https://datafeed.dukascopy.com/datafeed/{instr}/{y:04d}/{m0:02d}/{d:02d}/"
SCALE = 1000.0
TICK_DT = np.dtype([("ms", ">u4"), ("ask", ">u4"), ("bid", ">u4"), ("ask_vol", ">f4"), ("bid_vol", ">f4")])
CANDLE_DT = np.dtype([("sec", ">u4"), ("open", ">u4"), ("close", ">u4"), ("low", ">u4"), ("high", ">u4"),
                      ("vol", ">f4")])
SYMBOL_MAP = {"SPX": "USA500IDXUSD", "NDX": "USATECHIDXUSD", "DJI": "USA30IDXUSD"}


def instrument_for(symbol: str) -> str:
    s = symbol.upper()
    return SYMBOL_MAP.get(s, s if s.endswith("USD") else f"{s}USUSD")


def decode_ticks(raw_lzma: bytes, hour_start: pd.Timestamp) -> pd.DataFrame:
    if not raw_lzma:
        return pd.DataFrame(columns=["bid", "ask", "bid_size", "ask_size"])
    raw = lzma.decompress(raw_lzma)
    a = np.frombuffer(raw, dtype=TICK_DT, count=len(raw) // TICK_DT.itemsize)
    ts = hour_start + pd.to_timedelta(a["ms"].astype(np.int64), unit="ms")
    return pd.DataFrame({"bid": a["bid"] / SCALE, "ask": a["ask"] / SCALE, "bid_size": a["bid_vol"].astype(float),
                         "ask_size": a["ask_vol"].astype(float)}, index=pd.DatetimeIndex(ts, name="ts"))


def decode_candles(raw_lzma: bytes, day_start: pd.Timestamp) -> pd.DataFrame:
    if not raw_lzma:
        return pd.DataFrame(columns=["open", "high", "low", "close", "vol"])
    raw = lzma.decompress(raw_lzma)
    a = np.frombuffer(raw, dtype=CANDLE_DT, count=len(raw) // CANDLE_DT.itemsize)
    ts = day_start + pd.to_timedelta(a["sec"].astype(np.int64), unit="s")
    return pd.DataFrame({c: a[c] / SCALE for c in ("open", "high", "low", "close")} | {"vol": a["vol"].astype(float)},
                        index=pd.DatetimeIndex(ts, name="start"))


def _url(instr: str, t: pd.Timestamp) -> str:
    return BASE.format(instr=instr, y=t.year, m0=t.month - 1, d=t.day)


def fetch_ticks_hour(instr: str, hour: pd.Timestamp) -> pd.DataFrame:
    body = client().get(_url(instr, hour) + f"{hour.hour:02d}h_ticks.bi5", allow_404=True, timeout=30)
    return decode_ticks(body or b"", hour)


def fetch_minute_bars_day(instr: str, day: pd.Timestamp) -> pd.DataFrame:
    """Bid & ask 1m candles for a UTC day -> mid OHLC bars with bid/ask closes."""
    bid = decode_candles(client().get(_url(instr, day) + "BID_candles_min_1.bi5", allow_404=True, timeout=30) or b"", day)
    ask = decode_candles(client().get(_url(instr, day) + "ASK_candles_min_1.bi5", allow_404=True, timeout=30) or b"", day)
    if bid.empty or ask.empty:
        return pd.DataFrame()
    j = bid.join(ask, lsuffix="_bid", rsuffix="_ask", how="inner")
    # Dukascopy emits flat candles (vol 0) for minutes with no quote update -> keep (quotes persist)
    out = pd.DataFrame({c: 0.5 * (j[f"{c}_bid"] + j[f"{c}_ask"]) for c in ("open", "high", "low", "close")},
                       index=j.index)
    out["volume"] = np.nan
    out["bid_close"], out["ask_close"] = j["close_bid"], j["close_ask"]
    return out


def fetch_range(symbol: str, start, end, kind: str = "bars", store: Optional[ParquetStore] = None) -> dict:
    """kind='bars' -> bars_1m/dukascopy (mid OHLC + bid/ask closes, T2);
    kind='ticks' -> quotes/dukascopy (every top-of-book update, T2). Only NYSE RTH hours."""
    store = store or default_store()
    instr = instrument_for(symbol)
    sch = frames.schedule()
    days = sch.loc[pd.Timestamp(start).normalize():pd.Timestamp(end).normalize()]
    n = 0
    for d, row in days.iterrows():
        try:
            if kind == "bars":
                b = fetch_minute_bars_day(instr, pd.Timestamp(d).tz_localize("UTC"))
                if b.empty:
                    continue
                if instr.endswith("USUSD"):   # US stock/ETF CFDs are quoted during RTH only
                    b = b[(b.index >= row["open"]) & (b.index < row["close"])]
                else:
                    b = b[(b.index >= row["open"] - pd.Timedelta(hours=5, minutes=30)) & (b.index < row["close"] + pd.Timedelta(hours=4))]
                bars = frames.intraday_bars(b, "1m", extra_cols=("bid_close", "ask_close"))
                bars["volume"] = np.nan
                n = store.write("bars_1m", SOURCE, symbol, bars, fidelity=int(Fidelity.T2_PROXY), partition="year",
                                extra={"instrument": instr, "note": "CFD mid quotes, not trades"})
            else:
                hours = pd.date_range(row["open"].floor("h"), row["close"], freq="h", inclusive="left")
                q = pd.concat([fetch_ticks_hour(instr, h) for h in hours])
                q = q[(q.index >= row["open"]) & (q.index < row["close"])]
                if len(q):
                    n = store.write("quotes", SOURCE, symbol, q, fidelity=int(Fidelity.T2_PROXY), partition="year",
                                    extra={"instrument": instr, "note": "Dukascopy CFD quotes, not NBBO"})
        except FetchError as e:
            log.warning("dukascopy %s %s: %s", instr, d, e)
    return {"symbol": symbol, "instrument": instr, "kind": kind, "rows": n}


def fetch_recent(symbol: str, days: int = 5) -> dict:
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=days * 7 // 5 + 2)
    r1 = fetch_range(symbol, start, end, "bars")
    r2 = fetch_range(symbol, end - pd.Timedelta(days=3), end, "ticks")
    return {"bars": r1, "ticks": r2}
