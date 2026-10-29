"""Event assembly for Portal.events(). Returns flat frames: kind, symbol, known_at,
effective_at, fidelity, source + payload columns (documented in docs/DATA.md).

EARNINGS            ZipLime 8-K 2.02 releases (exact acceptance time, 2003-04 .. 2026-10) as the
                    spine; DoltHub earnings_calendar rows (2020+) not matched within +-3 days are
                    added with approximate clock times (BMO 08:00 / AMC 16:30 ET; T2).
                    eps_estimate: latest DoltHub weekly consensus snapshot ('Current Quarter',
                    same fiscal period end +-7 d) dated before the release (estimate_known_at =
                    snapshot date 23:59 ET); else DoltHub eps_history final consensus with
                    estimate_known_at = release - 1 min (T2).
EARNINGS_SCHEDULE   DoltHub calendar entries as *expected* dates: known_at = 7 calendar days
                    before the expected date at 00:00 ET (ASSUMPTION; the date a calendar entry
                    first appeared is unknown) -> T2_PROXY.
EARNINGS_RELEASE    SEC submissions 8-K Item 2.02 acceptance times (per-symbol fetch).
INDEX_ADD/INDEX_DEL S&P 500 changes (providers/sp500.py).
FOMC                statement releases (providers/fomc.py).
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from atlasbt.core.clock import et
from atlasbt.core.types import Fidelity

TIMING = {"before market open": "BMO", "after market close": "AMC", "during market hours": "DURING"}
CLOCK = {"BMO": "08:00", "AMC": "16:30", "DURING": "12:00", "UNKNOWN": "16:30"}


def _sym_filter(df: pd.DataFrame, symbols) -> pd.DataFrame:
    if symbols is None or df.empty:
        return df
    want = {s.upper() for s in symbols} | {s.upper().replace("-", ".") for s in symbols}
    return df[df["symbol"].str.upper().isin(want)]


def _window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if df.empty:
        return df
    k = pd.to_datetime(df["known_at"], utc=True)
    return df[(k >= start) & (k <= end)]


def load_events(portal, kind: str, symbols: Optional[Iterable[str]], start, end) -> pd.DataFrame:
    k = kind.upper()
    st = portal.store
    if k in ("INDEX_ADD", "INDEX_DEL"):
        df = portal._read("events", "sp500", "INDEX")
        df = df[df["kind"] == k] if len(df) else df
        return _window(_sym_filter(df, symbols), start, end)
    if k == "FOMC":
        return _window(portal._read("events", "fomc", "FOMC"), start, end)
    if k == "EARNINGS_RELEASE":
        syms = symbols or [s.replace("EARNINGS_RELEASE_", "") for s in st.symbols("events", "sec")
                           if s.startswith("EARNINGS_RELEASE_")]
        parts = [portal._read("events", "sec", f"EARNINGS_RELEASE_{s.upper()}") for s in syms]
        parts = [p for p in parts if len(p)]
        return _window(pd.concat(parts, ignore_index=True), start, end) if parts else pd.DataFrame()
    if k == "EARNINGS_SCHEDULE":
        cal = _dolt_calendar(portal)
        if cal.empty:
            return cal
        cal = cal.assign(kind="EARNINGS_SCHEDULE",
                         known_at=[et(d - pd.Timedelta(days=7), "00:00") for d in cal["release_date"]],
                         effective_at=cal["known_at"], fidelity=int(Fidelity.T2_PROXY),
                         known_at_rule="expected_date - 7 calendar days (assumption)")
        return _window(_sym_filter(cal, symbols), start, end)
    if k == "EARNINGS":
        return _earnings(portal, symbols, start, end)
    return pd.DataFrame()


def _dolt_calendar(portal) -> pd.DataFrame:
    key = ("dolt_calendar",)
    if key in portal._cache:
        return portal._cache[key]
    c = portal._read("reference", "dolthub", "earnings_calendar")
    if not len(c):
        portal._cache[key] = pd.DataFrame()
        return portal._cache[key]
    c = c.reset_index(drop=True)
    timing = c["when"].fillna("").str.strip().str.lower().map(TIMING).fillna("UNKNOWN")
    rd = pd.to_datetime(c["date"])
    from .providers.dolthub import earnings_known_at
    known = [earnings_known_at(d, t) for d, t in zip(rd, timing)]
    out = pd.DataFrame({"kind": "EARNINGS", "symbol": c["act_symbol"].str.upper().str.replace(".", "-", regex=False),
                        "known_at": known, "effective_at": known, "timing": timing.to_numpy(), "release_date": rd,
                        "fidelity": int(Fidelity.T2_PROXY), "source": "dolthub"})
    portal._cache[key] = out
    return out


def _earnings(portal, symbols, start, end) -> pd.DataFrame:
    z = portal.store.read("events", "ziplime", "EARNINGS", start=start - pd.Timedelta(days=1),
                          end=end + pd.Timedelta(days=1), time_col="known_at")
    z = _window(_sym_filter(z.reset_index(drop=True) if len(z) else z, symbols), start, end)
    cal = _window(_sym_filter(_dolt_calendar(portal), symbols), start, end)
    if len(cal):
        if len(z):
            zz = z[["symbol", "known_at"]].copy()
            zz["d"] = pd.to_datetime(zz["known_at"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
            cal = cal.copy()
            cal["_m"] = False
            for sym, g in cal.groupby("symbol"):
                zd = zz.loc[zz["symbol"] == sym, "d"].to_numpy()
                if len(zd):
                    diff = np.abs((g["release_date"].to_numpy()[:, None] - zd[None, :]).astype("timedelta64[D]").astype(int))
                    cal.loc[g.index, "_m"] = (diff <= 3).any(axis=1)
            cal = cal[~cal["_m"]].drop(columns="_m")
        cal = cal.assign(period_end=pd.NaT, eps_actual=np.nan)
    df = pd.concat([x for x in (z, cal) if len(x)], ignore_index=True) if (len(z) or len(cal)) else pd.DataFrame()
    if df.empty:
        return df
    return _attach_estimates(portal, df)


def _attach_estimates(portal, df: pd.DataFrame) -> pd.DataFrame:
    est = portal._read("reference", "dolthub", "eps_estimate")
    hist = portal._read("reference", "dolthub", "eps_history")
    df = df.copy()
    df["eps_estimate"] = np.nan
    df["estimate_known_at"] = pd.NaT
    df["estimate_source"] = None
    if "period_end" not in df:
        df["period_end"] = pd.NaT
    rel = pd.to_datetime(df["known_at"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    pe = pd.to_datetime(df["period_end"])
    syms = df["symbol"].str.upper().str.replace("-", ".", regex=False)
    if len(est):
        e = est[est["act_symbol"].isin(set(syms))]
        eg = {s: g.sort_values("date") for s, g in e.groupby("act_symbol")}
    else:
        eg = {}
    if len(hist):
        h = hist[hist["act_symbol"].isin(set(syms))].copy()
        h["period_end_date"] = pd.to_datetime(h["period_end_date"])
        hg = {s: g for s, g in h.groupby("act_symbol")}
    else:
        hg = {}
    vals, kas, srcs, fids, reasons, street = [], [], [], [], [], []
    for s, p, r, ka in zip(syms, pe, rel, df["known_at"]):
        v, k, src = np.nan, pd.NaT, None
        # street (adjusted, consensus-basis) actual EPS for the same fiscal period: DoltHub eps_history
        st_v = np.nan
        h = hg.get(s)
        if h is not None:
            if pd.notna(p):
                mh = h[(h["period_end_date"] - p).abs() <= pd.Timedelta(days=7)]
            else:
                lag = (r.normalize() - h["period_end_date"]).dt.days
                mh = h[(lag >= 10) & (lag <= 120)].sort_values("period_end_date").tail(1)
            if len(mh) and pd.notna(mh["reported"].iloc[-1]):
                st_v = float(mh["reported"].iloc[-1])
        street.append(st_v)
        g = eg.get(s)
        if g is not None and pd.notna(p):
            m = g[((g["period_end_date"] - p).abs() <= pd.Timedelta(days=7)) & (g["date"] < r.normalize())]
            if len(m) and pd.notna(m["consensus"].iloc[-1]):
                v, k, src = float(m["consensus"].iloc[-1]), et(m["date"].iloc[-1], "23:59"), "dolthub_eps_estimate"
        if src is None:
            h = hg.get(s)
            if h is not None:
                if pd.notna(p):
                    m = h[(h["period_end_date"] - p).abs() <= pd.Timedelta(days=7)]
                else:   # no period: the latest period ending 10-120 days before release
                    lag = (r.normalize() - h["period_end_date"]).dt.days
                    m = h[(lag >= 10) & (lag <= 120)].sort_values("period_end_date").tail(1)
                if len(m) and pd.notna(m["estimate"].iloc[-1]):
                    v, k, src = float(m["estimate"].iloc[-1]), pd.Timestamp(ka) - pd.Timedelta(minutes=1), "dolthub_eps_history"
        vals.append(v)
        kas.append(k)
        srcs.append(src)
        if src == "dolthub_eps_estimate":
            fids.append(int(Fidelity.T1_BAR))
            reasons.append("weekly consensus snapshot dated before the release (PIT)")
        elif src == "dolthub_eps_history":
            fids.append(int(Fidelity.T2_PROXY))
            reasons.append("consensus recorded post hoc; revisions not PIT")
        else:
            fids.append(None)
            reasons.append(None)
    # EPS bases: ZipLime figures are GAAP diluted EPS (XBRL); DoltHub `reported` and all consensus
    # values are street (adjusted) EPS. A surprise must compare street vs street:
    #   surprise = eps_actual_street - eps_estimate.  eps_actual = street when available, else GAAP
    #   (eps_actual_basis says which).
    gaap = pd.to_numeric(df["eps_actual"], errors="coerce") if "eps_actual" in df else pd.Series(np.nan, index=df.index)
    if "source" in df:
        gaap = gaap.where(df["source"] == "ziplime")
    if "figure_quarters" in df:          # Q4 releases carry full-year GAAP figures -> not a quarter EPS
        gaap = gaap.where(pd.to_numeric(df["figure_quarters"], errors="coerce") == 1)
    df["eps_actual_gaap"] = gaap.to_numpy()
    df["eps_actual_street"] = street
    has_street = pd.notna(df["eps_actual_street"])
    df["eps_actual"] = np.where(has_street, df["eps_actual_street"], df["eps_actual_gaap"])
    df["eps_actual_basis"] = np.where(has_street, "street", np.where(pd.notna(df["eps_actual_gaap"]), "gaap", None))
    df["eps_estimate"] = vals
    df["estimate_known_at"] = pd.to_datetime(pd.Series(kas), utc=True).to_numpy()
    df["estimate_source"] = srcs
    df["estimate_fidelity"] = fids
    df["estimate_reason"] = reasons
    return df
