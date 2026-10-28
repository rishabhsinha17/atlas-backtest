"""Option chain annotation shared by all real-chain sources (docs/DATA.md §4.6).

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
    """-> (expiry as last trading day, settlement_date)."""
    d = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    roots = pd.Series(roots).astype(str).str.strip().str.upper().to_numpy()
    sch = frames.schedule().index
    is_session = d.isin(sch)
    am = np.isin(roots, list(AM_INDEX_ROOTS))
    prev = _prev_session(d)
    expiry = pd.DatetimeIndex(np.where(am | ~is_session, prev.values, d.values))
    settle = pd.DatetimeIndex(np.where(am, d.values, expiry.values))
    return expiry, settle


def is_monthly(vendor_dates) -> np.ndarray:
    d = pd.DatetimeIndex(pd.to_datetime(vendor_dates))
    third_fri = (d.weekday == 4) & (d.day >= 15) & (d.day <= 21)
    thu_before = (d.weekday == 3) & (d.day >= 14) & (d.day <= 20)    # holiday-shifted monthly
    return np.asarray(third_fri | thu_before)


def annotate_chain(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "_annotated" in df.attrs:
        return df
    out = df.copy()
    roots = out["root"] if "root" in out.columns else out["symbol"].astype(str).str[:6].str.strip()
    vendor = pd.to_datetime(out["expiry"])
    exp, settle = last_trading_day(vendor, roots)
    out["root"] = roots.astype(str).str.strip().to_numpy()
    out["kind"] = np.where(is_monthly(vendor), "MONTHLY", "WEEKLY")
    out["expiry"] = exp
    out["settlement_date"] = settle
    if "underlying_price_source" not in out.columns:
        out["underlying_price_source"] = np.where(out["underlying_price"].notna(), "pit_bar_close_at_ts", None)
    out.attrs.update(df.attrs)
    out.attrs["_annotated"] = True
    return out
