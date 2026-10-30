"""S&P 500 point-in-time membership.

Sources
  * fja05680/sp500 'S&P 500 Historical Components & Changes (Updated).csv' (GitHub raw):
    one row per change date with the full member list (1996-01-02 ->). Tickers with a
    '-YYYYMM' suffix are historical identities (ticker since changed/delisted).
  * Wikipedia 'List of S&P 500 companies' (current members, GICS sector/sub-industry, CIK)
  * Wikipedia 'Historical components of the S&P 500' wikitext: effective date, added/removed
    tickers, reason and REFERENCES -- the S&P press-release citations usually carry the
    announcement date (|date= or a date in the URL), which we parse.

PIT conventions
  * Membership interval [start, end): `start` is the fja row date = effective date (member
    from that session's open). Universe('SP500', as_of) = members whose interval contains
    the ET session date of as_of. NOTE the membership list itself is not "announced" at
    as_of: membership is a fact known from the effective date (index providers publish
    changes in advance, so this never leaks).
  * INDEX_ADD / INDEX_DEL events: effective_at = effective session open;
    known_at = announcement date 17:15 ET when parsed from the S&P press-release citation
    (S&P typically publishes after the close ~17:15 ET; time-of-day approximate -> T2_PROXY);
    otherwise conservative known_at = 17:15 ET of the session BEFORE the effective date
    (payload announce_source='fallback_prev_session'), also T2_PROXY.
"""
from __future__ import annotations

import io
import logging
import re
from typing import Optional
from urllib.parse import quote

import numpy as np
import pandas as pd

from atlasbt.core.clock import et
from atlasbt.core.types import Fidelity

from .. import frames
from ..http import client
from ..store import ParquetStore, default_store

log = logging.getLogger("atlasbt.data.sp500")
SOURCE = "sp500"
FJA_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
           + quote("S&P 500 Historical Components & Changes (Updated).csv"))
WIKI_CURRENT = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_HIST_RAW = "https://en.wikipedia.org/w/index.php?title=Historical_components_of_the_S%26P_500&action=raw"

# Known ticker renames where the SAME company continues under a new symbol that Yahoo
# serves (historical fja identity -> current Yahoo ticker). Extend as verified.
RENAMES = {"FB": "META", "ANTM": "ELV", "FISV": "FI", "PKI": "RVTY", "RE": "EG", "FLT": "CPAY",
           "ABC": "COR", "WLTW": "WTW", "DISCA": "WBD", "CTL": "LUMN", "HFC": "DINO", "TMK": "GL",
           "BLL": "BALL", "PEAK": "DOC", "COG": "CTRA", "ADS": "BFH", "INFO": "INFO"}


# ---------------------------------------------------------------------------
def parse_fja(content: bytes) -> pd.DataFrame:
    """-> membership intervals: ticker (as listed, incl. suffix), base, start, end (NaT=current)."""
    df = pd.read_csv(io.BytesIO(content))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    sets = [set(t.strip() for t in str(s).split(",") if t.strip()) for s in df["tickers"]]
    dates = list(df["date"])
    open_since: dict[str, pd.Timestamp] = {}
    rows = []
    prev: set = set()
    for d, cur in zip(dates, sets):
        for t in cur - prev:
            open_since[t] = d
        for t in prev - cur:
            rows.append((t, open_since.pop(t), d))
        prev = cur
    for t, s in open_since.items():
        rows.append((t, s, pd.NaT))
    out = pd.DataFrame(rows, columns=["ticker", "start", "end"])
    out["base"] = out["ticker"].str.replace(r"-\d{6}$", "", regex=True)
    out["historical_identity"] = out["ticker"] != out["base"]
    out["left_censored"] = out["start"] == dates[0]       # member before data begins (1996-01-02)
    return out.sort_values(["start", "ticker"]).reset_index(drop=True)


_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


def _ann_date_from_ref(ref: str) -> Optional[pd.Timestamp]:
    m = re.search(r"\|\s*date\s*=\s*([^|}]+)", ref)
    if m:
        d = pd.to_datetime(m.group(1).strip(), errors="coerce")
        if pd.notna(d):
            return d
    m = re.search(r"press\.spglobal\.com/(\d{4}-\d{2}-\d{2})-", ref)
    if m:
        return pd.Timestamp(m.group(1))
    m = re.search(r"/(20\d{2})(\d{2})(\d{2})-\d+", ref)
    if m:
        return pd.to_datetime(f"{m.group(1)}-{m.group(2)}-{m.group(3)}", errors="coerce")
    return None


def parse_wiki_changes(wikitext: str) -> pd.DataFrame:
    """Parse the changes table of 'Historical components of the S&P 500' wikitext."""
    start = wikitext.find("Effective Date")
    end = wikitext.find("\n|}", start)
    body = wikitext[start:end]
    # resolve named refs: name -> full definition
    named = {}
    for m in re.finditer(r'<ref name\s*=\s*"?([^">/]+?)"?\s*>(.*?)</ref>', wikitext, flags=re.S):
        named[m.group(1).strip()] = m.group(2)
    rows = []
    for chunk in body.split("\n|-")[1:]:
        lines = [ln for ln in chunk.strip().split("\n") if ln.startswith("|")]
        cells = []
        for ln in lines:
            ln = ln[2:] if ln.startswith("||") else ln[1:]
            cells.extend([c.strip() for c in re.split(r"\s*\|\|\s*", ln)] if "||" in ln and "<ref" not in ln else [ln.strip()])
        if len(cells) < 6 or cells[0].startswith("!") or "Ticker" in cells[0]:
            continue
        eff = pd.to_datetime(re.sub(r"\[\[|\]\]", "", cells[0]), errors="coerce")
        if pd.isna(eff):
            continue
        clean = lambda s: re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", s).strip()
        refs = " ".join(cells[6:]) if len(cells) > 6 else ""
        ann = None
        for m in re.finditer(r'<ref name\s*=\s*"?([^">/]+?)"?\s*/>', refs):
            d = _ann_date_from_ref(named.get(m.group(1).strip(), ""))
            ann = ann or d
        ann = _ann_date_from_ref(refs) or ann
        if ann is not None and not (eff - pd.Timedelta(days=75) <= ann <= eff):
            ann = None  # citation date is not an announcement date (access dates, news later)
        rows.append({"effective_date": eff.normalize(), "added": clean(cells[1]).upper() or None,
                     "added_name": clean(cells[2]) or None, "removed": clean(cells[3]).upper() or None,
                     "removed_name": clean(cells[4]) or None, "reason": clean(cells[5])[:300],
                     "announce_date": ann.normalize() if ann is not None else pd.NaT})
    return pd.DataFrame(rows)


def build_events(members: pd.DataFrame, wiki: pd.DataFrame) -> pd.DataFrame:
    sch = frames.schedule()
    ev = []
    for _, r in members.iterrows():
        for kind, d in (("INDEX_ADD", r["start"]), ("INDEX_DEL", r["end"])):
            if pd.isna(d) or (kind == "INDEX_ADD" and r["left_censored"]):
                continue
            col = "added" if kind == "INDEX_ADD" else "removed"
            ann = pd.NaT
            if len(wiki):
                w = wiki[(wiki[col] == r["base"]) & ((wiki["effective_date"] - d).abs() <= pd.Timedelta(days=4))]
                if len(w):
                    ann = w["announce_date"].iloc[0]
            i = sch.index.searchsorted(d, side="left")
            eff_open = sch["open"].iloc[i] if i < len(sch) else pd.NaT
            prev_sess = sch.index[max(i - 1, 0)]
            if pd.notna(ann):
                known, src = et(ann, "17:15"), "sp_press_release_citation"
                if known >= eff_open:
                    known, src = et(prev_sess, "17:15"), "fallback_prev_session"
            else:
                known, src = et(prev_sess, "17:15"), "fallback_prev_session"
            ev.append({"kind": kind, "symbol": r["base"], "fja_ticker": r["ticker"], "known_at": known,
                       "effective_at": eff_open, "index": "SP500", "effective_date": d,
                       "announce_date": ann, "announce_source": src, "fidelity": int(Fidelity.T2_PROXY),
                       "source": SOURCE})
    return pd.DataFrame(ev).sort_values(["effective_date", "kind", "symbol"]).reset_index(drop=True)


def build_membership(store: Optional[ParquetStore] = None, ttl_hours: float = 24) -> dict:
    store = store or default_store()
    mem = parse_fja(client().get(FJA_URL, ttl=ttl_hours * 3600))
    wiki_txt = client().get(WIKI_HIST_RAW, ttl=ttl_hours * 3600).decode("utf-8")
    wiki = parse_wiki_changes(wiki_txt)
    # extend fja with Wikipedia changes after the last fja change date
    last = mem["start"].max()
    last = max(last, mem["end"].max())
    newer = wiki[wiki["effective_date"] > last].sort_values("effective_date")
    for _, w in newer.iterrows():
        if isinstance(w["removed"], str) and w["removed"]:
            m = (mem["base"] == w["removed"]) & mem["end"].isna()
            mem.loc[m, "end"] = w["effective_date"]
        if isinstance(w["added"], str) and w["added"] and not ((mem["base"] == w["added"]) & mem["end"].isna()).any():
            mem = pd.concat([mem, pd.DataFrame([{"ticker": w["added"], "base": w["added"], "start": w["effective_date"],
                                                 "end": pd.NaT, "historical_identity": False,
                                                 "left_censored": False}])], ignore_index=True)
    mem = mem[mem["base"].notna()].reset_index(drop=True)
    mem["yahoo"] = [RENAMES.get(b, b).replace(".", "-") if (not h or b in RENAMES) else None
                    for b, h in zip(mem["base"].astype(str), mem["historical_identity"])]
    events = build_events(mem, wiki)
    store.write("reference", SOURCE, "membership", mem, key=["ticker", "start"], replace=True,
                fidelity=int(Fidelity.T1_BAR), extra={"fja_last_change": str(last.date()),
                                                       "wiki_rows": len(wiki)})
    store.write("reference", SOURCE, "wiki_changes", wiki, key=["effective_date", "added", "removed"],
                replace=True)
    store.write("events", SOURCE, "INDEX", events, key=["kind", "fja_ticker", "effective_date"], replace=True,
                fidelity=int(Fidelity.T2_PROXY))
    cur = fetch_current_constituents(store, ttl_hours)
    n_ann = int((events["announce_source"] == "sp_press_release_citation").sum())
    return {"intervals": len(mem), "tickers": mem["ticker"].nunique(), "events": len(events),
            "events_with_parsed_announcement": n_ann, "wiki_rows": len(wiki), "current": cur,
            "historical_identity_intervals": int(mem["historical_identity"].sum())}


def fetch_current_constituents(store: ParquetStore, ttl_hours: float = 24) -> int:
    html = client().get(WIKI_CURRENT, ttl=ttl_hours * 3600).decode("utf-8")
    t = pd.read_html(io.StringIO(html))[0]
    t = t.rename(columns={"Symbol": "symbol", "Security": "name", "GICS Sector": "gics_sector",
                          "GICS Sub-Industry": "gics_sub_industry", "Date added": "date_added", "CIK": "cik"})
    t["symbol"] = t["symbol"].str.upper()
    t["as_of"] = pd.Timestamp.now(tz="UTC").normalize()
    keep = ["symbol", "name", "gics_sector", "gics_sub_industry", "date_added", "cik", "as_of"]
    store.write("reference", SOURCE, "current_constituents", t[keep], key=["symbol"], replace=True,
                fidelity=int(Fidelity.T2_PROXY))
    return len(t)


def union_tickers(store: Optional[ParquetStore] = None) -> list[str]:
    """Yahoo symbols for every S&P 500 member since 1996 that Yahoo may still serve."""
    store = store or default_store()
    m = store.read("reference", SOURCE, "membership")
    if not len(m):
        build_membership(store)
        m = store.read("reference", SOURCE, "membership")
    return sorted(set(m["yahoo"].dropna()))
