"""Survivorship diagnostics & delisting detection.

What is survivorship-free and what is not (see docs/DATA.md for numbers):
  * S&P 500 MEMBERSHIP (fja05680 + Wikipedia) is point-in-time and includes removed names.
  * PRICES are not: Yahoo only serves symbols that still trade (plus a few leftovers), so most
    removed members (acquired / bankrupt / renamed tickers) have NO Yahoo history. DoltHub's
    stocks.ohlcv (2011+) includes delisted symbols and fills part of the gap; there is no free
    source for pre-2011 delisted prices.
  * A reused ticker can attach a DIFFERENT company's Yahoo history to a historical member
    (e.g. a ticker freed by an acquisition and later re-issued). We flag members whose bars
    do not cover their membership window.

Delisting detection: a symbol whose last daily bar is > 10 sessions before the latest market
session and which is absent from the latest Nasdaq Trader symbol directory is treated as
delisted: DELIST action with ex_date = next session after the last bar, amount = last close
(proxy for the delisting value; true cash-merger / liquidation value unknown), known_at =
close of the last trading day -> fidelity T2_PROXY.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from atlasbt.core.types import Fidelity

from . import frames


def _latest_market_session(portal) -> Optional[pd.Timestamp]:
    key = ("latest_session",)
    if key not in portal._cache:
        b = portal._read("bars_1d", "yahoo", "SPY", columns=["known_at"])
        portal._cache[key] = frames.to_naive_date(b.index)[-1] if len(b) else None
    return portal._cache[key]


def _listed_now(portal) -> set:
    key = ("listed_now",)
    if key not in portal._cache:
        d = portal._directory()
        if len(d):
            last = d["snapshot_date"].max()
            syms = d.loc[d["snapshot_date"] == last, "symbol"].astype(str).str.upper()
            portal._cache[key] = set(syms) | set(syms.str.replace(".", "-", regex=False))
        else:
            portal._cache[key] = set()
    return portal._cache[key]


def delisting_for(portal, symbol: str):
    """-> (ex_date naive, known_at UTC, last_close) or None."""
    key = ("delist", symbol.upper())
    if key in portal._cache:
        return portal._cache[key]
    res = None
    latest = _latest_market_session(portal)
    listed = _listed_now(portal)
    b = portal._daily_frame(symbol)
    if len(b) and not str(b.attrs.get("source", "")).startswith("cboe") and latest is not None and listed:
        if True:
            last = frames.to_naive_date(b.index)[-1]
            sch = frames.schedule()
            gap = int(((sch.index > last) & (sch.index <= latest)).sum())
            if gap > 10 and symbol.upper() not in listed:
                nxt = sch.index[sch.index.searchsorted(last, side="right")]
                res = (nxt, b["known_at"].iloc[-1], float(b["close"].iloc[-1]))
    portal._cache[key] = res
    return res


def membership_coverage(portal) -> pd.DataFrame:
    """Per S&P 500 membership interval: is there daily price data covering it?"""
    m = portal._read("reference", "sp500", "membership")
    rows = []
    sch = frames.schedule()
    latest = _latest_market_session(portal)
    for r in m.to_dict("records"):
        sym = r.get("yahoo") or str(r["base"]).replace(".", "-")
        start = pd.Timestamp(r["start"])
        end = pd.Timestamp(r["end"]) if pd.notna(r["end"]) else latest
        b = portal._daily_frame(sym)
        src = b.attrs.get("source")
        first = last = None
        cov = 0.0
        if len(b):
            d = frames.to_naive_date(b.index)
            if len(d):
                first, last = d[0], d[-1]
                need = sch.index[(sch.index >= max(start, pd.Timestamp("1996-01-02"))) & (sch.index < end)]
                cov = float(need.isin(d).mean()) if len(need) else 1.0
        rows.append({"ticker": r["ticker"], "symbol": sym, "start": start, "end": r["end"], "source": src,
                     "first_bar": first, "last_bar": last, "coverage": cov,
                     "identity_suspect": bool(first is not None and pd.notna(r["end"]) and first > pd.Timestamp(r["end"]))})
    return pd.DataFrame(rows)


def universe_bias_report(portal, name: str, as_of) -> dict:
    as_of = pd.Timestamp(as_of)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of
    n = name.upper()
    if n not in ("SP500", "S&P500", "SPX_MEMBERS"):
        members = portal.universe(n, as_of)
        return {"universe": n, "as_of": str(as_of), "members": len(members), "with_data": len(members),
                "coverage": 1.0, "survivorship_bias": False,
                "note": "fixed ETF list; members without listing history at as_of are excluded"}
    members = portal.universe("SP500", as_of)
    d = as_of.tz_convert("America/New_York").tz_localize(None).normalize()
    have, missing = [], []
    for s in members:
        b = portal._daily_frame(s)
        ok = False
        if len(b):
            dd = frames.to_naive_date(b.index)
            ok = len(dd) > 0 and dd[0] <= d and dd[-1] >= d - pd.Timedelta(days=10)
        (have if ok else missing).append(s)
    cov = len(have) / max(len(members), 1)
    return {"universe": "SP500", "as_of": str(as_of), "members": len(members), "with_data": len(have),
            "coverage": round(cov, 4), "missing": missing, "survivorship_bias": cov < 0.999,
            "fidelity": int(Fidelity.T1_BAR if cov >= 0.999 else Fidelity.T2_PROXY),
            "note": ("members without price data are typically delisted/acquired names absent from Yahoo; "
                     "cross-sectional results on this universe are biased toward survivors")}
