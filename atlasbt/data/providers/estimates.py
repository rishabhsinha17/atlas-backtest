"""Point-in-time analyst consensus, next-12-months EPS (S041) from DoltHub `earnings.eps_estimate`.

Source: DoltHub post-no-preference/earnings `eps_estimate` -- WEEKLY snapshots (mostly Sundays) of
the consensus per (act_symbol, period) with period in {Current Quarter, Next Quarter, Current Year,
Next Year}: period_end_date, consensus, recent, count (number of estimates), high, low, year_ago.
Snapshots start 2017-10-26. Loaded from the Hugging Face CSV mirror of the table
(`hf:siddharthmb/stocks-earnings-eps_estimate`, ends 2026-01-25; `load_mirror`) and extended
from the DoltHub SQL API for later snapshot dates (`fetch_recent`; dolthub.query, read-only use).
Only 'Current Year' / 'Next Year' rows are kept here ('Current Quarter' already lives in
reference/dolthub/eps_estimate for EARNINGS).

Point in time: a snapshot dated d is a capture of the consensus around d (its exact capture time
and when DoltHub published the commit are not documented), so it is stamped known at the CLOSE of
the first NYSE session strictly after d (a Sunday snapshot -> Monday 16:00 ET; >= one full session
of lag; conservative). The CSV mirror is a later copy of the same snapshot rows (no revisions of
past snapshots are visible in it -- not verifiable, hence T2).

Derived fields (fundamentals-style long frame, kind 'fundamentals', source 'estimates', one file
per DoltHub ticker in internal spelling, e.g. BRK-B):
  eps_consensus_fy1 / eps_consensus_fy2   'Current Year' / 'Next Year' consensus
  eps_consensus_ntm  = w * FY1 + (1 - w) * FY2, w = clip(days from the snapshot date to the FY1
                       fiscal-year end / 365.25, 0, 1) -- the standard time-weighted 12-month-forward
                       blend (I/B/E/S-style NTM is not published free) -> T2_PROXY
  eps_num_analysts_ntm = min(count FY1, count FY2) (the smaller panel: conservative for '>= 5')
  period_end = snapshot date, known_at as above, cik = issuer the ticker denoted on the snapshot
  date (identity layer, Portal._identity_by_date); rows whose identity is unknown (cik 0) are
  dropped (fail closed).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from atlasbt.core.types import Fidelity

from .. import frames
from ..store import ParquetStore, default_store

log = logging.getLogger("atlasbt.data.estimates")
SOURCE = "estimates"
KIND = "fundamentals"
PERIODS = ("Current Year", "Next Year")
FIELDS = ("eps_consensus_ntm", "eps_num_analysts_ntm", "eps_consensus_fy1", "eps_consensus_fy2")
FIRST_SNAPSHOT = "2017-10-26"
MAX_STALE_DAYS = 21          # a ticker whose latest snapshot is older than this is not served
RAW = ("reference", SOURCE, "fy_consensus_raw")


def _internal(sym: str) -> str:
    return str(sym).upper().replace(".", "-").strip()


def target_tickers(store: ParquetStore) -> set[str]:
    """S&P 500 PIT-union tickers (membership tickers + bases, SEC security master)."""
    out: set[str] = set()
    m = store.read("reference", "sp500", "membership")
    if len(m):
        for c in ("ticker", "base", "yahoo"):
            if c in m.columns:
                out |= {_internal(x) for x in m[c].dropna()}
    sm = store.read("reference", "sec", "security_master")
    if len(sm):
        out |= {_internal(x) for x in sm["symbol"].dropna()}
    return out


def load_mirror(path: str | Path, store: Optional[ParquetStore] = None, symbols: Optional[Iterable[str]] = None,
                chunksize: int = 1_000_000) -> dict:
    """CSV mirror -> raw FY1/FY2 rows (reference/estimates/fy_consensus_raw)."""
    store = store or default_store()
    want = {_internal(s) for s in symbols} if symbols else target_tickers(store)
    keep = []
    for ch in pd.read_csv(path, chunksize=chunksize, dtype={"act_symbol": str, "period": str}):
        ch = ch[ch["period"].isin(PERIODS)]
        ch = ch.assign(symbol=ch["act_symbol"].map(_internal))
        keep.append(ch[ch["symbol"].isin(want)])
    df = pd.concat(keep, ignore_index=True)
    store.write(*RAW, df, key=["date", "act_symbol", "period"], replace=True,
                extra={"origin": "hf:siddharthmb/stocks-earnings-eps_estimate (DoltHub earnings.eps_estimate)"})
    return {"rows": len(df), "symbols": int(df["symbol"].nunique()), "first": str(df["date"].min()),
            "last": str(df["date"].max())}


def fetch_recent(store: Optional[ParquetStore] = None, max_dates: int = 200) -> dict:
    """Snapshot dates after the stored ones from the DoltHub SQL API (index seeks, exact-key queries,
    300 symbols per query; cached responses only when Success)."""
    from . import dolthub
    store = store or default_store()
    raw = store.read(*RAW)
    cur = str(pd.to_datetime(raw["date"]).max().date()) if len(raw) else FIRST_SNAPSHOT
    syms = sorted({s.replace("-", ".") for s in target_tickers(store)})
    dates, rows = [], []
    while len(dates) < max_dates:
        try:
            r = dolthub.query("earnings", f"SELECT date FROM eps_estimate WHERE date > '{cur}' ORDER BY date LIMIT 1",
                              cache=False)
        except dolthub.DoltQueryError as e:
            log.warning("date seek after %s failed: %s", cur, e)
            break
        if not r:
            break
        cur = r[0]["date"]
        dates.append(cur)
        for i in range(0, len(syms), 300):
            try:
                rows += dolthub.query("earnings", "SELECT * FROM eps_estimate WHERE date='" + cur + "' AND period IN "
                                      "('Current Year','Next Year') AND act_symbol IN (" + dolthub._in(syms[i:i + 300]) + ")")
            except dolthub.DoltQueryError as e:
                log.warning("eps_estimate %s: %s", cur, e)
    if rows:
        df = pd.DataFrame(rows)
        df["symbol"] = df["act_symbol"].map(_internal)
        for c in ("consensus", "recent", "count", "high", "low", "year_ago"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        store.write(*RAW, df, key=["date", "act_symbol", "period"])
    return {"new_dates": len(dates), "rows": len(rows), "last": cur}


def known_at_for(dates: pd.Series) -> pd.Series:
    """Close (UTC) of the first NYSE session strictly after each snapshot date."""
    sch = frames.schedule()
    idx = sch.index.values
    d = pd.to_datetime(dates).dt.normalize().to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(idx, d, side="right")
    closes = pd.DatetimeIndex(sch["close"]).tz_convert("UTC")
    return pd.Series(closes[pos], index=dates.index)


def ntm_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw FY1/FY2 rows of ONE ticker -> wide frame per snapshot: date, fy1, fy2, fy1_end, n1, n2, ntm, n."""
    r = raw.copy()
    r["date"] = pd.to_datetime(r["date"]).dt.normalize()
    r["period_end_date"] = pd.to_datetime(r["period_end_date"], errors="coerce")
    r["consensus"] = pd.to_numeric(r["consensus"], errors="coerce")
    r["count"] = pd.to_numeric(r["count"], errors="coerce")
    a = r[r["period"] == "Current Year"].drop_duplicates("date", keep="last").set_index("date")
    b = r[r["period"] == "Next Year"].drop_duplicates("date", keep="last").set_index("date")
    w = pd.DataFrame({"fy1": a["consensus"], "fy1_end": a["period_end_date"], "n1": a["count"]}).join(
        pd.DataFrame({"fy2": b["consensus"], "n2": b["count"]}), how="inner")
    w = w.dropna(subset=["fy1", "fy2", "fy1_end"])
    wt = ((w["fy1_end"] - w.index.to_series()).dt.days / 365.25).clip(0.0, 1.0)
    w["w_fy1"] = wt
    w["ntm"] = wt * w["fy1"] + (1.0 - wt) * w["fy2"]
    w["n"] = np.fmin(w["n1"], w["n2"])
    return w.reset_index().rename(columns={"index": "date"})


def build(store: Optional[ParquetStore] = None, portal=None, symbols: Optional[Iterable[str]] = None) -> dict:
    """Raw rows -> fundamentals/estimates/<TICKER> long frames (identity-checked)."""
    store = store or default_store()
    if portal is None:
        from ..portal import Portal
        portal = Portal(store)
    raw = store.read(*RAW)
    if not len(raw):
        return {"skipped": "no raw estimate rows (run load_mirror / fetch_recent)"}
    if symbols:
        raw = raw[raw["symbol"].isin({_internal(s) for s in symbols})]
    stats = {"tickers": 0, "rows": 0, "snapshots_dropped_identity": 0}
    for sym, g in raw.groupby("symbol"):
        w = ntm_frame(g)
        if not len(w):
            continue
        try:
            cik = portal._identity_by_date(sym, list(w["date"]))[0].astype("int64")
        except Exception:
            cik = np.zeros(len(w), dtype="int64")
        ok = cik != 0
        stats["snapshots_dropped_identity"] += int((~ok).sum())
        w, cik = w[ok], cik[ok]
        if not len(w):
            continue
        ka = known_at_for(w["date"])
        parts = []
        for field, col in (("eps_consensus_ntm", "ntm"), ("eps_num_analysts_ntm", "n"),
                           ("eps_consensus_fy1", "fy1"), ("eps_consensus_fy2", "fy2")):
            parts.append(pd.DataFrame({"symbol": sym, "field": field, "value": w[col].astype(float).to_numpy(),
                                       "period_end": w["date"].to_numpy(), "known_at": ka.to_numpy(),
                                       "fy1_end": w["fy1_end"].to_numpy(), "w_fy1": w["w_fy1"].to_numpy(),
                                       "cik": cik, "source": "dolthub_eps_estimate",
                                       "fidelity": int(Fidelity.T2_PROXY)}))
        df = pd.concat(parts, ignore_index=True)
        store.write(KIND, SOURCE, sym, df, key=["field", "period_end"], replace=True, fidelity=int(Fidelity.T2_PROXY),
                    extra={"known_at_rule": "close of the first NYSE session after the snapshot date"})
        stats["tickers"] += 1
        stats["rows"] += len(df)
    return stats
