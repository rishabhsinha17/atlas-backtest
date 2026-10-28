"""Yahoo Finance via the `yfinance` library.

Daily:  history(period='max', auto_adjust=False, actions=True, repair=False).
        Yahoo's "raw" OHLC are split-adjusted to the fetch date (verified: AAPL
        2020-08-28 Close = 124.81 = 499.23/4). We UN-adjust using Yahoo's own
        split list so the store holds as-traded prices (adjust.unadjust_yahoo);
        volume likewise. Dividends from Yahoo are split-adjusted per current
        share; un-adjusted to the per-share amount paid at the time.
        Yahoo's 'Adj Close' is kept as column `yahoo_adjclose` for QA only.
Actions: Yahoo gives ex-dates & amounts but no declaration / record / pay dates.
        known_at := close of the session BEFORE ex_date (latest moment the
        market must have known; real declarations are typically 2-6 weeks
        earlier), fidelity T2_PROXY for the timing. pay_date unknown -> None
        (engine should assume a lag; see docs/DATA.md).
Intraday: 1m (last ~30 days, 8-day windows), 5m/15m (60 days), 60m (730 days),
        prepost=True. Bars labelled by start (ET) -> UTC; sessions tagged
        PRE/RTH/POST with the NYSE calendar (early closes honoured).
        Accumulated over time into the store (idempotent upsert).

Note: yfinance manages its own HTTP session (it is the user-approved access
path for Yahoo); we only throttle calls (>=1 s between symbols) and back off
on rate-limit errors.
"""
from __future__ import annotations

import logging
import time
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from atlasbt.core.types import ActionType, Fidelity

from .. import frames
from ..adjust import split_factor_after, unadjust_yahoo
from ..store import ParquetStore, default_store

log = logging.getLogger("atlasbt.data.yahoo")
SOURCE = "yahoo"
_MIN_GAP = 1.0
_last_call = [0.0]

INTRADAY_LIMITS = {  # interval -> (max lookback days, max days per request)
    "1m": (29, 7), "2m": (59, 59), "5m": (59, 59), "15m": (59, 59), "30m": (59, 59),
    "60m": (729, 729), "1h": (729, 729),
}


def yahoo_symbol(symbol: str) -> str:
    s = symbol.upper().replace(".", "-")
    return {"VIX": "^VIX", "SPX": "^GSPC"}.get(s, s)


def _throttle():
    dt = time.monotonic() - _last_call[0]
    if dt < _MIN_GAP:
        time.sleep(_MIN_GAP - dt)
    _last_call[0] = time.monotonic()


def _history(symbol: str, retries: int = 4, **kw) -> pd.DataFrame:
    import yfinance as yf
    last = None
    for i in range(retries):
        _throttle()
        try:
            h = yf.Ticker(yahoo_symbol(symbol)).history(raise_errors=True, **kw)
            return h
        except Exception as e:  # yfinance raises various types
            msg = str(e)
            last = e
            if "Rate" in msg or "429" in msg or "Too Many" in msg:
                time.sleep(min(120, 15 * (2 ** i)))
                continue
            if "delisted" in msg.lower() or "no data" in msg.lower() or "not found" in msg.lower() \
                    or "No timezone" in msg or "possibly delisted" in msg:
                return pd.DataFrame()
            time.sleep(2 * (i + 1))
    log.warning("yahoo %s failed: %s", symbol, last)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# parsing (pure functions; unit-tested on recorded responses)
# ---------------------------------------------------------------------------
def parse_daily(symbol: str, h: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """yfinance history frame -> (raw canonical daily bars, actions frame, info)."""
    if h is None or h.empty:
        return frames.empty_bars(), pd.DataFrame(), {"rows": 0}
    h = h.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
                          "Adj Close": "yahoo_adjclose", "Volume": "volume", "Dividends": "div",
                          "Stock Splits": "split", "Capital Gains": "capgain"})
    dates = frames.to_naive_date(h.index)
    h.index = dates
    h = h[~h.index.duplicated(keep="last")]
    dates = pd.DatetimeIndex(h.index)
    splits = h.get("split", pd.Series(0.0, index=dates)).fillna(0.0)
    splits = splits[splits > 0]
    raw = unadjust_yahoo(h[["open", "high", "low", "close", "volume"]], dates, splits)
    if "yahoo_adjclose" in h:
        raw["yahoo_adjclose"] = h["yahoo_adjclose"]
    # bars with all-NaN prices (Yahoo sometimes emits them for halted days) are dropped
    raw = raw.dropna(subset=["open", "high", "low", "close"], how="all")
    bars, dropped = frames.daily_bars(raw, extra_cols=("yahoo_adjclose",))
    # actions
    sch = frames.schedule()
    rows = []
    for d, r in splits.items():
        rows.append(_action_row(symbol, ActionType.SPLIT, d, ratio=float(r), sch=sch))
    divs = h.get("div", pd.Series(dtype=float)).fillna(0.0)
    divs = divs[divs > 0]
    if len(divs):
        f = split_factor_after(pd.DatetimeIndex(divs.index), pd.DatetimeIndex(splits.index), splits.to_numpy())
        for (d, amt), fac in zip(divs.items(), f):
            rows.append(_action_row(symbol, ActionType.CASH_DIVIDEND, d, amount=float(amt) * fac,
                                    yahoo_amount=float(amt), sch=sch))
    cg = h.get("capgain", pd.Series(dtype=float)).fillna(0.0)
    cg = cg[cg > 0]
    if len(cg):
        f = split_factor_after(pd.DatetimeIndex(cg.index), pd.DatetimeIndex(splits.index), splits.to_numpy())
        for (d, amt), fac in zip(cg.items(), f):
            rows.append(_action_row(symbol, ActionType.SPECIAL_DIVIDEND, d, amount=float(amt) * fac,
                                    yahoo_amount=float(amt), sch=sch, note="capital gains distribution"))
    acts = pd.DataFrame(rows)
    info = {"rows": len(bars), "dropped_non_sessions": [str(x.date()) for x in dropped],
            "first": str(bars.index[0]) if len(bars) else None,
            "last": str(bars.index[-1]) if len(bars) else None}
    return bars, acts, info


def _action_row(symbol, typ: ActionType, ex_date, ratio=1.0, amount=0.0, yahoo_amount=np.nan,
                sch=None, note=""):
    ex = pd.Timestamp(ex_date).normalize()
    known = conservative_known_at(ex, sch)
    return {"symbol": symbol.upper(), "type": typ.value, "ex_date": ex, "known_at": known,
            "ratio": float(ratio), "amount": float(amount), "pay_date": pd.NaT, "new_symbol": None,
            "fidelity": int(Fidelity.T2_PROXY), "source": SOURCE, "yahoo_amount": yahoo_amount,
            "known_at_method": "prev_session_close", "note": note}


def conservative_known_at(ex_date: pd.Timestamp, sch: Optional[pd.DataFrame] = None) -> pd.Timestamp:
    """Close of the last session strictly before ex_date (UTC)."""
    sch = frames.schedule() if sch is None else sch
    i = sch.index.searchsorted(pd.Timestamp(ex_date).normalize(), side="left")
    if i == 0:
        return pd.Timestamp(ex_date).tz_localize("UTC")
    return sch["close"].iloc[i - 1]


# ---------------------------------------------------------------------------
# fetch + store
# ---------------------------------------------------------------------------
def fetch_daily(symbol: str, store: Optional[ParquetStore] = None, refresh_days: float = 0.5) -> dict:
    """Download full daily history + actions for `symbol` into the store.

    Re-downloads the full history each time it is called after `refresh_days`
    (Yahoo re-bases split adjustments, so incremental appends of split-adjusted
    prices would be inconsistent; we always rebuild raw from a full pull)."""
    store = store or default_store()
    m = store.meta("bars_1d", SOURCE, symbol)
    if m and refresh_days and (pd.Timestamp.now(tz="UTC") - pd.Timestamp(m["fetched_at"])) \
            < pd.Timedelta(days=refresh_days):
        return {"symbol": symbol, "skipped": True, "rows": m.get("rows")}
    h = _history(symbol, period="max", interval="1d", auto_adjust=False, actions=True, repair=False)
    bars, acts, info = parse_daily(symbol, h)
    if bars.empty:
        store.write("reference", SOURCE, "_missing", pd.DataFrame(
            {"symbol": [symbol.upper()], "checked_at": [pd.Timestamp.now(tz="UTC")]}).set_index("symbol"))
        return {"symbol": symbol, "rows": 0, "missing": True}
    extra = {"first_ts": info["first"], "last_ts": info["last"], "raw": True,
             "price_basis": "as-traded (un-split-adjusted from Yahoo)",
             "dropped_non_sessions": info["dropped_non_sessions"][:50]}
    store.write("bars_1d", SOURCE, symbol, bars, fidelity=int(Fidelity.T1_BAR), adjusted=False,
                replace=True, extra=extra)
    if len(acts):
        store.write("actions", SOURCE, symbol, acts, key=["ex_date", "type"], replace=True,
                    fidelity=int(Fidelity.T2_PROXY))
    return {"symbol": symbol, **info, "actions": len(acts)}


def fetch_intraday(symbol: str, interval: str = "1m", store: Optional[ParquetStore] = None,
                   lookback_days: Optional[int] = None) -> dict:
    """Pull all intraday bars Yahoo still serves for `interval` (with pre/post)."""
    store = store or default_store()
    maxlb, chunk = INTRADAY_LIMITS[interval]
    lb = min(lookback_days or maxlb, maxlb)
    now = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
    start = now - pd.Timedelta(days=lb)
    # skip what we already have (keep 1 day overlap for revisions)
    m = store.meta(f"bars_{_norm(interval)}", SOURCE, symbol)
    if m.get("last_ts"):
        start = max(start, pd.Timestamp(m["last_ts"]).normalize() - pd.Timedelta(days=1))
    parts = []
    t = start
    while t < now:
        e = min(t + pd.Timedelta(days=chunk), now)
        h = _history(symbol, start=t.strftime("%Y-%m-%d"), end=e.strftime("%Y-%m-%d"), interval=interval,
                     prepost=True, auto_adjust=False, actions=False, repair=False)
        if h is not None and not h.empty:
            parts.append(h)
        t = e
    if not parts:
        return {"symbol": symbol, "interval": interval, "rows": 0}
    h = pd.concat(parts)
    bars = parse_intraday(symbol, h, interval, store)
    if bars.empty:
        return {"symbol": symbol, "interval": interval, "rows": 0}
    prev_first = m.get("first_ts")
    first = min(str(bars.index[0]), prev_first) if prev_first else str(bars.index[0])
    n = store.write(f"bars_{_norm(interval)}", SOURCE, symbol, bars, fidelity=int(Fidelity.T1_BAR),
                    partition="year", extra={"first_ts": first, "last_ts": str(bars.index[-1]), "raw": True})
    return {"symbol": symbol, "interval": interval, "rows": n, "new": len(bars),
            "first": str(bars.index[0]), "last": str(bars.index[-1])}


def parse_intraday(symbol: str, h: pd.DataFrame, interval: str,
                   store: Optional[ParquetStore] = None) -> pd.DataFrame:
    h = h.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    h = h[["open", "high", "low", "close", "volume"]]
    h = h[~h.index.duplicated(keep="last")].sort_index()
    bars = frames.intraday_bars(h, _norm(interval))
    # undo split adjustment for intraday bars before a split in the window
    if store is not None:
        acts = store.read("actions", SOURCE, symbol)
        if len(acts):
            sp = acts[acts["type"] == ActionType.SPLIT.value]
            if len(sp):
                d = frames.to_naive_date(bars.index)
                f = split_factor_after(d, pd.DatetimeIndex(sp["ex_date"]), sp["ratio"].to_numpy())
                if (f != 1).any():
                    for c in ("open", "high", "low", "close"):
                        bars[c] = bars[c].to_numpy() * f
                    bars["volume"] = bars["volume"].to_numpy() / f
    # Yahoo emits a few zero-length/duplicate bars at session edges; drop all-NaN rows
    return bars.dropna(subset=["open", "close"], how="all")


def _norm(interval: str) -> str:
    return {"1h": "60m"}.get(interval, interval)
