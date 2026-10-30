"""Option chain annotation shared by all real-chain sources (docs/DATA.md §4.6).

SYMBOL CONVENTION (ARCHITECTURE §3, core Instrument.option): the internal OCC-style symbol
encodes Instrument.expiry = the LAST TRADING DAY; the root (SPX vs SPXW) tells the series.
Vendor symbols (DoltHub, historicaloptiondata) encode the vendor date (settlement Friday for
SPX AM monthlies). `to_internal` converts vendor rows ONCE, at ingestion (the vendor symbol is
kept in `vendor_symbol`); legacy stored frames without `vendor_symbol` are converted on read.

* `expiry` := LAST TRADING DAY (ARCHITECTURE / OPTIONS contract):
    - equity/ETF options and PM-settled index series (SPXW): the vendor date, moved to the
      previous NYSE session when it is not a session (holiday Fridays, e.g. Good Friday);
    - AM-settled index monthlies (root SPX, NDX, RUT, ...): vendors list the settlement Friday;
      trading ends the session before -> expiry = previous session, settlement_date = Friday.
* `settlement_date`: the vendor date for AM-settled index options, else = expiry.
* `kind`: MONTHLY (third-Friday cycle, incl. holiday-shifted Thursdays), WEEKLY otherwise.
* `root`: OCC root parsed from the symbol (SPX vs SPXW...).
* `underlying_price_source`: how underlying_price was obtained ('vendor' or 'pit_bar_close_at_ts').
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import frames

AM_INDEX_ROOTS = {"SPX", "NDX", "RUT", "DJX", "OEX", "XEO", "VIX"}


def _prev_session(d: pd.DatetimeIndex) -> pd.DatetimeIndex:
    sch = frames.schedule().index
    pos = sch.searchsorted(d, side="left") - 1
    return pd.DatetimeIndex(sch[np.clip(pos, 0, len(sch) - 1)])


def last_trading_day(dates, roots) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """VENDOR date -> (expiry as last trading day, settlement_date).
    PM / physical series: vendor date, or the previous session if it is not a session.
    AM-settled index series: settlement = vendor date (previous session if the vendor lists a
    holiday Friday), last trading day = the session before settlement."""
    d = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    roots = pd.Series(roots).astype(str).str.strip().str.upper().to_numpy()
    sch = frames.schedule().index
    is_session = d.isin(sch)
    am = np.isin(roots, list(AM_INDEX_ROOTS))
    prev = _prev_session(d)
    settle_am = pd.DatetimeIndex(np.where(is_session, d.values, prev.values))
    prev_am = _prev_session(settle_am)
    expiry = pd.DatetimeIndex(np.where(am, prev_am.values, np.where(is_session, d.values, prev.values)))
    settle = pd.DatetimeIndex(np.where(am, settle_am.values, expiry.values))
    return expiry, settle


def settlement_for_internal(expiry, root: str) -> pd.Timestamp:
    """Settlement date of an INTERNAL symbol (date = last trading day): AM-settled roots settle
    at the next session's open (SOQ); everything else on the last trading day."""
    e = pd.Timestamp(expiry).normalize()
    if str(root).strip().upper() in AM_INDEX_ROOTS:
        sch = frames.schedule().index
        nxt = sch[sch > e]
        return pd.Timestamp(nxt[0]) if len(nxt) else e + pd.Timedelta(days=1)
    return e


def internal_symbols(root, expiry, right, strike) -> pd.Series:
    root = pd.Series(root).astype(str).str.strip().str.upper().str.pad(6, side="right").to_numpy()
    k = (pd.Series(strike).astype(float).to_numpy() * 1000).round().astype("int64")
    e = pd.DatetimeIndex(pd.to_datetime(expiry)).strftime("%y%m%d")
    r = pd.Series(right).astype(str).to_numpy()
    return pd.Series([f"{a}{b}{c}{d:08d}" for a, b, c, d in zip(root, e, r, k)])


def to_internal(df: pd.DataFrame) -> pd.DataFrame:
    """Vendor chain rows (expiry = vendor date, vendor OCC symbol) -> internal convention:
    expiry = last trading day, settlement_date, kind, root, symbol rebuilt from root + last
    trading day, original symbol in `vendor_symbol`. Idempotent."""
    if df is None or df.empty or "vendor_symbol" in df.columns:
        return df
    out = df.copy()
    roots = (out["root"] if "root" in out.columns else out["symbol"].astype(str).str[:6]).astype(str).str.strip()
    vendor = pd.to_datetime(out["expiry"])
    exp, settle = last_trading_day(vendor, roots)
    out["root"] = roots.to_numpy()
    out["kind"] = np.where(is_monthly(vendor), "MONTHLY", "WEEKLY")
    out["expiry"] = exp
    out["settlement_date"] = settle
    out["vendor_symbol"] = out["symbol"].astype(str)
    sym = out["symbol"].astype(str)
    right = out["right"] if "right" in out.columns else sym.str[-9]
    strike = out["strike"] if "strike" in out.columns else sym.str[-8:].astype(int) / 1000.0
    out["symbol"] = internal_symbols(out["root"], exp, right, strike).to_numpy()
    out.attrs.update(df.attrs)
    return out


def is_monthly(vendor_dates) -> np.ndarray:
    d = pd.DatetimeIndex(pd.to_datetime(vendor_dates))
    third_fri = (d.weekday == 4) & (d.day >= 15) & (d.day <= 21)
    thu_before = (d.weekday == 3) & (d.day >= 14) & (d.day <= 20)    # holiday-shifted monthly
    return np.asarray(third_fri | thu_before)


def annotate_chain(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "_annotated" in df.attrs:
        return df
    out = to_internal(df)       # no-op for frames ingested in the internal convention
    out = out.copy() if out is df else out
    if "underlying_price_source" not in out.columns:
        out["underlying_price_source"] = np.where(out["underlying_price"].notna(), "pit_bar_close_at_ts", None)
    out.attrs.update(df.attrs)
    out.attrs["_annotated"] = True
    return out
