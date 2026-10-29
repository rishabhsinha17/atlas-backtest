"""FRED series via https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>

Stored raw (percent units as published) in rates/fred/<ID> with columns
value (percent), known_at.  Conversion to the engine's convention happens in
atlasbt.data.rates (see DTB3 discount -> money-market yield).

PIT: H.15 daily rates for date d are published the next Federal Reserve business day
(~16:15 ET; Fed holidays incl. Columbus & Veterans Day, Sunday holidays observed Monday,
Saturday holidays not observed); EFFR (DFF) is published by the NY Fed ~09:00 ET next
business day. One conservative rule for all daily series: known_at = 16:30 ET of the next
Fed business day after d. Monthly macro series (CPI) are not used for trading signals here;
their real release lag (~2 weeks) is NOT modelled -> known_at = period start + 45d
(conservative), flagged T2. FRED serves the latest vintage (revisions not PIT;
irrelevant for T-bill/treasury yields, relevant for macro series).
"""
from __future__ import annotations

import io
import logging
from typing import Optional

import pandas as pd

from atlasbt.core.clock import et
from atlasbt.core.types import Fidelity

from .. import frames
from .. import config
from ..http import FetchError, HttpClient, client
from ..store import ParquetStore, default_store

log = logging.getLogger("atlasbt.data.fred")
SOURCE = "fred"
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

SERIES = {
    "DTB3": "3-Month T-bill secondary market rate, DISCOUNT basis, % (daily, 1954-)",
    "DTB4WK": "4-Week T-bill secondary market rate, discount basis (2001-)",
    "DTB6": "6-Month T-bill secondary market, discount basis (1958-)",
    "DTB1YR": "1-Year T-bill secondary market, discount basis (1959-2001)",
    "DGS1MO": "1-Month Treasury constant maturity, bond-equivalent, % (2001-)",
    "DGS3MO": "3-Month Treasury constant maturity, bond-equivalent (1981-)",
    "DGS6MO": "6-Month Treasury constant maturity (1981-)",
    "DGS1": "1-Year CMT", "DGS2": "2-Year CMT", "DGS5": "5-Year CMT", "DGS10": "10-Year CMT", "DGS30": "30-Year CMT",
    "DFF": "Effective federal funds rate, % (ACT/360 money-market)",
    "EFFR": "Effective federal funds rate (NY Fed, 2000-)", "DPRIME": "Bank prime loan rate",
    "SOFR": "Secured overnight financing rate (2018-)",
    "DFEDTARU": "Fed funds target range upper (2008-12-16-)",
    "DFEDTAR": "Fed funds target (until 2008-12-15)",
    "T10Y2Y": "10y minus 2y CMT spread", "T10Y3M": "10y minus 3m CMT spread",
    "BAMLH0A0HYM2": "ICE BofA US HY OAS (last ~3y only on FRED due to licensing)",
    "BAMLC0A0CM": "ICE BofA US IG OAS (last ~3y only)",
    "DFII10": "10-Year TIPS real yield",
    "CPIAUCSL": "CPI-U SA, monthly (latest vintage; NOT PIT)",
}
MONTHLY = {"CPIAUCSL"}


def parse_csv(series_id: str, content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content))
    date_col = df.columns[0]
    v = pd.to_numeric(df[df.columns[1]].replace(".", None), errors="coerce")
    out = pd.DataFrame({"value": v.to_numpy()}, index=pd.DatetimeIndex(pd.to_datetime(df[date_col]), name="date"))
    out = out[out["value"].notna()]
    out["known_at"] = known_at_for(out.index, monthly=series_id in MONTHLY)
    return out


def _nth_weekday(y, m, wd, n):
    d = pd.Timestamp(y, m, 1)
    d += pd.Timedelta(days=(wd - d.weekday()) % 7)
    return d + pd.Timedelta(weeks=n - 1)


def _last_weekday(y, m, wd):
    d = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
    return d - pd.Timedelta(days=(d.weekday() - wd) % 7)


def fed_holidays(y0: int = 1950, y1: int = 2035) -> set:
    """Federal Reserve Bank holidays. Fixed-date holidays falling on Sunday are observed Monday;
    on Saturday they are NOT observed (the Fed stays open the Friday before)."""
    out = set()
    for y in range(y0, y1 + 1):
        fixed = [(1, 1), (7, 4), (11, 11), (12, 25)] + ([(6, 19)] if y >= 2022 else [])
        for m, d in fixed:
            t = pd.Timestamp(y, m, d)
            if t.weekday() == 6:
                t += pd.Timedelta(days=1)
            if t.weekday() < 5:
                out.add(t)
        if y >= 1986:
            out.add(_nth_weekday(y, 1, 0, 3))          # MLK
        out.add(_nth_weekday(y, 2, 0, 3))              # Washington's Birthday
        out.add(_last_weekday(y, 5, 0))                # Memorial Day
        out.add(_nth_weekday(y, 9, 0, 1))              # Labor Day
        out.add(_nth_weekday(y, 10, 0, 2))             # Columbus Day
        out.add(_nth_weekday(y, 11, 3, 4))             # Thanksgiving
    return out


_FED_HOL: set = set()


def next_fed_business_day(d: pd.Timestamp) -> pd.Timestamp:
    if not _FED_HOL:
        _FED_HOL.update(fed_holidays())
    t = pd.Timestamp(d).normalize() + pd.Timedelta(days=1)
    while t.weekday() >= 5 or t in _FED_HOL:
        t += pd.Timedelta(days=1)
    return t


def known_at_for(dates: pd.DatetimeIndex, monthly: bool = False) -> pd.DatetimeIndex:
    """H.15 publishes day-d rates on the next FEDERAL RESERVE business day ~16:15 ET (Columbus and
    Veterans Day are Fed holidays while NYSE trades) -> known_at = 16:30 ET of that day."""
    if monthly:
        return pd.DatetimeIndex([et(d + pd.Timedelta(days=45), "08:30") for d in dates])
    cache: dict = {}
    out = []
    for d in pd.DatetimeIndex(dates):
        if d not in cache:
            cache[d] = et(next_fed_business_day(d), "16:30")
        out.append(cache[d])
    return pd.DatetimeIndex(out)


def fetch_series(series_id: str, store: Optional[ParquetStore] = None, ttl_hours: float = 12) -> dict:
    """Network first (cached); FRED stopped answering the plain 'atlasbt-research/0.1' UA on
    2026-10-28 (read timeouts), so on failure we fall back to the stale raw cache and then to
    the scout's copy under data_raw/fred/<ID>.csv (same fredgraph.csv format)."""
    store = store or default_store()
    origin = "network"
    try:
        content = HttpClient(max_retries=1, timeout=25).get(URL, params={"id": series_id}, ttl=ttl_hours * 3600)
    except FetchError:
        content = client().cached(URL, params={"id": series_id})
        origin = "stale_raw_cache"
        if content is None:
            local = config.DATA_RAW / "fred" / f"{series_id}.csv"
            if not local.exists():
                raise
            content, origin = local.read_bytes(), f"data_raw/fred/{series_id}.csv"
    df = parse_csv(series_id, content)
    fid = Fidelity.T2_PROXY if series_id in MONTHLY else Fidelity.T1_BAR
    store.write("rates", SOURCE, series_id, df, fidelity=int(fid), replace=True,
                extra={"description": SERIES.get(series_id, ""), "units": "percent", "origin": origin,
                       "first_ts": str(df.index[0].date()), "last_ts": str(df.index[-1].date())})
    return {"symbol": series_id, "rows": len(df), "first": str(df.index[0].date()), "last": str(df.index[-1].date()),
            "origin": origin}
