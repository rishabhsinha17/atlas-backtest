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


def delisting_for(portal, symbol: str, ident=None):
    """-> (ex_date naive, known_at UTC, last_close) or None. The identity's verified series ends
    (merger / bankruptcy / take-private) more than 10 sessions before the latest session and the
    ticker is not listed today (or now belongs to another company)."""
    b = ident if ident is not None else portal._identity_frame(symbol)
    sid = b.attrs.get("security_id") if hasattr(b, "attrs") else None
    key = ("delist", symbol.upper(), sid)
    if key in portal._cache:
        return portal._cache[key]
    res = None
    latest = _latest_market_session(portal)
    listed = _listed_now(portal)
    if len(b) and not str(b.attrs.get("source", "")).startswith("cboe") and latest is not None and listed:
        last = frames.to_naive_date(b.index)[-1]
        sch = frames.schedule()
        gap = int(((sch.index > last) & (sch.index <= latest)).sum())
        still_same = symbol.upper() in listed and sid and sid.startswith("TKR:")
        if gap > 10 and not still_same:
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
        start = pd.Timestamp(r["start"])
        end = pd.Timestamp(r["end"]) if pd.notna(r["end"]) else latest
        # the member's security key (bar-date identity / membership mapping), as universe() returns
        sym = portal._member_key(str(r["base"]).upper().replace(".", "-"), max(start, pd.Timestamp("1996-01-02")))
        b = portal._daily_frame(sym, min(end, latest) if end is not None and latest is not None else start,
                                start=max(start, pd.Timestamp("1996-01-02")))
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


def universe_bias_report(portal, name: str, as_of, window: int = 20) -> dict:
    """Survivorship / coverage report (red-team #5, #13).

    SP500 (and every universe derived from it, e.g. SP500_TOP50_DV): members at as_of whose
    identity-verified series has a bar on the session (or within the 3 previous sessions) count as
    covered; `stale` lists members whose last bar is 1-3 sessions old; `member_session_coverage` is
    the share of (member, session) pairs over the trailing `window` sessions that have a bar.
    Derived universes inherit the base coverage: they rank only members WITH data, so the bias is
    the base universe's. Fixed ETF lists: no survivorship issue (members without listing history at
    as_of are excluded)."""
    as_of = pd.Timestamp(as_of)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of
    n = name.upper()
    base = "SP500" if n in ("SP500", "S&P500", "SPX_MEMBERS") or n.startswith("SP500") else None
    if base is None:
        members = portal.universe(n, as_of)
        return {"universe": n, "as_of": str(as_of), "members": len(members), "with_data": len(members),
                "coverage": 1.0, "survivorship_bias": False, "fidelity": int(Fidelity.T1_BAR),
                "note": "fixed ETF list; members without listing history at as_of are excluded"}
    members = portal.universe("SP500", as_of)
    d = as_of.tz_convert("America/New_York").tz_localize(None).normalize()
    sch = frames.schedule().index
    past = sch[sch <= d]
    win = past[-window:]
    recent = past[-4:]
    have, missing, stale = [], [], []
    pairs = covered = 0
    for s in members:
        b = portal._daily_frame(s, d)
        dd = frames.to_naive_date(b.index) if len(b) else pd.DatetimeIndex([])
        inwin = dd[(dd >= win[0]) & (dd <= d)] if len(dd) else dd
        pairs += len(win)
        covered += len(set(inwin) & set(win))
        if len(dd) and dd[dd <= d].size and dd[dd <= d][-1] >= recent[0]:
            have.append(s)
            if dd[dd <= d][-1] < past[-1]:
                stale.append(s)
        else:
            missing.append(s)
    cov = len(have) / max(len(members), 1)
    rep = {"universe": "SP500", "as_of": str(as_of), "members": len(members), "with_data": len(have),
           "coverage": round(cov, 4), "missing": missing, "stale": stale,
           "member_session_coverage": round(covered / max(pairs, 1), 4),
           "survivorship_bias": bool(cov < 0.999),
           "fidelity": int(Fidelity.T1_BAR if cov >= 0.999 else Fidelity.T2_PROXY),
           "note": ("members whose security identity / prices cannot be established are missing (fail "
                    "closed); results on this universe are biased toward survivors")}
    if n != "SP500" and n not in ("S&P500", "SPX_MEMBERS"):
        derived = portal.universe(n, as_of)
        rep.update({"universe": n, "base_universe": "SP500", "derived_members": len(derived),
                    "base_coverage": rep["coverage"],
                    "note": "derived from SP500 members WITH data: inherits the base universe's survivorship bias"})
    return rep
