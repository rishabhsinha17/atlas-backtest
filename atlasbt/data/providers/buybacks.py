"""Share-repurchase AUTHORIZATION announcements from SEC EDGAR (events kind 'BUYBACK', for S045).

S045 needs, per announcement: known_at, the authorized amount as a fraction of market cap
(`pct_mcap`) and whether the release was bundled with earnings (`with_earnings`).

Pipeline
  1. Candidates (no contact string needed): EDGAR full-text search (efts.sec.gov, 2001+) per issuer
     CIK for AUTH_QUERY on 8-K filings incl. exhibits (press releases). One row per matching
     document: accession, filing date, items, exhibit type; known_at = SEC acceptance time
     (submissions JSON; fallback filed date 17:30 ET -- never early). `with_earnings` = the 8-K
     carries Item 2.02 (Results of Operations). Stored as reference/sec_fts/BUYBACK_CANDIDATES --
     NOT as events: a phrase hit is not yet an authorization (earnings releases routinely mention
     existing programs) and carries no amount.
  2. Text (needs SEC_USER_AGENT -- www.sec.gov/Archives answers the plain UA with 403): each
     candidate document is read and `parse_authorization` extracts a NEW/INCREASED authorization
     amount in dollars ("authorized the repurchase of up to $10 billion", "a new $5 billion share
     repurchase program", "increased its authorization by $2 billion"). Share-count-only
     authorizations ("up to 10 million shares") get amount_usd = shares x the prior close.
  3. pct_mcap = amount_usd / market cap at the last close before known_at (shares outstanding =
     latest XBRL dei/us-gaap value filed before known_at, price = raw close). Rows with an amount
     become events/sec_fts/BUYBACK (fidelity T2_PROXY: amount/intent parsed from free text).

Without SEC_USER_AGENT only step 1 runs; no BUYBACK events exist and S045 stays NOT_TESTABLE
(NEEDS USER: "SEC_USER_AGENT contact string").
"""
from __future__ import annotations

import html
import logging
import re
from typing import Iterable, Optional

import pandas as pd

from atlasbt.core.types import Fidelity

from ..http import FetchError
from ..store import ParquetStore, default_store
from . import sec
from .dividends_decl import default_targets, identity_filter

log = logging.getLogger("atlasbt.data.buybacks")
KIND = "BUYBACK"
SOURCE = "sec_fts"
AUTH_QUERY = ('"authorized the repurchase" OR "authorized a new" OR "approved a new share repurchase" OR '
              '"new share repurchase" OR "new stock repurchase" OR "new repurchase program" OR '
              '"increase in its share repurchase" OR "increased its share repurchase" OR '
              '"additional share repurchase" OR "share repurchase authorization" OR '
              '"stock repurchase authorization" OR "repurchase of up to"')
CAND_COLS = ["symbol", "cik", "adsh", "file_date", "form", "items", "file_type", "filename", "known_at",
             "known_at_method", "with_earnings"]

_UNITS = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_MONEY = r"\$\s?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(thousand|million|billion|trillion)?"
_VERB = r"(?:authoriz|approv|announc|adopt|increas|expand|add)"
_PAT = [
    # "authorized (the repurchase of) up to $10 billion (of its common stock)"
    re.compile(_VERB + r"\w*[^.;]{0,120}?(?:repurchase|buy\s?back)[^.;]{0,60}?(?:up to|of)\s+(?:an additional\s+)?" + _MONEY, re.I),
    # "a new $5 billion share repurchase program" / "an additional $2.5 billion stock buyback"
    re.compile(r"(?:new|additional|incremental|increased?)\s+" + _MONEY + r"[^.;]{0,40}?(?:share|stock|common)?\s*(?:repurchase|buy\s?back)", re.I),
    # "increased its share repurchase authorization by $2 billion"
    re.compile(r"increas\w*[^.;]{0,40}?(?:repurchase|buy\s?back)\s+(?:authorization|program)\s+by\s+" + _MONEY, re.I),
]
_SHARES = re.compile(_VERB + r"\w*[^.;]{0,120}?(?:repurchase|buy\s?back)[^.;]{0,40}?(?:up to\s+)?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(million|thousand)?\s+(?:shares|of its (?:outstanding )?shares)", re.I)


def html_to_text(raw: bytes) -> str:
    t = raw.decode("utf-8", errors="replace")
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = html.unescape(t).replace("\xa0", " ")
    return re.sub(r"\s+", " ", t)


def _money(num: str, unit: Optional[str]) -> float:
    return float(num.replace(",", "")) * _UNITS.get((unit or "").lower(), 1.0)


_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_DATE = re.compile(r"\b(" + _MONTHS + r")\.?,?\s+(?:(\d{1,2}),?\s+)?((?:19|20)\d\d)\b")
#: wording of an EARLIER program (not an announcement): such sentences are skipped
_PAST = re.compile(r"\b(previously|remaining|years ago|had (?:been )?(?:approved|authorized))\b", re.I)
MAX_SENTENCE = 350           # longer "sentences" are slide / table text (investor presentations)
STALE_DAYS = 60


def stale_reference(sentence: str, filed: Optional[pd.Timestamp]) -> bool:
    """True when the sentence dates the authorization more than STALE_DAYS before the filing
    ("In June 2007, the Board authorized ..." in a 2009 filing): the earliest 'Month [day,] year'
    mentioned is used (later dates in the same sentence are usually expiries)."""
    if filed is None:
        return False
    ds = []
    for m in _DATE.finditer(sentence):
        try:
            ds.append(pd.Timestamp(f"{m.group(1)} {m.group(2) or 1} {m.group(3)}"))
        except ValueError:
            continue
    if not ds:
        return False
    f = pd.Timestamp(filed)
    f = f.tz_convert("America/New_York").tz_localize(None) if f.tzinfo else f
    return min(ds) < f.normalize() - pd.Timedelta(days=STALE_DAYS)


def parse_authorization(text: str, filed: Optional[pd.Timestamp] = None) -> dict:
    """-> {amount_usd, shares, sentence} for the LARGEST new/increased authorization stated in the
    text (amount_usd None when only a share count is given; {} when nothing matches). Amounts below
    $1 million are ignored (per-share figures, fees). With `filed`, sentences that date the
    authorization more than STALE_DAYS before the filing are ignored (stale_reference)."""
    best = None
    for p in _PAT:
        for m in p.finditer(text):
            amt = _money(m.group(1), m.group(2))
            if amt < 1e6:
                continue
            if best is None or amt > best["amount_usd"]:
                sent = _sentence(text, m.start(), m.end())
                if not _usable(sent, filed):
                    continue
                best = {"amount_usd": amt, "shares": None, "sentence": sent[:400]}
    if best is not None:
        return best
    for m in _SHARES.finditer(text):
        n = float(m.group(1).replace(",", "")) * _UNITS.get((m.group(2) or "").lower(), 1.0)
        sent = _sentence(text, m.start(), m.end())
        if n >= 1e4 and _usable(sent, filed):
            return {"amount_usd": None, "shares": n, "sentence": sent[:400]}
    return {}


def _usable(sent: str, filed) -> bool:
    return len(sent) <= MAX_SENTENCE and not _PAST.search(sent) and not stale_reference(sent, filed)


def _sentence(text: str, i0: int, i1: int) -> str:
    """The sentence around [i0, i1): split on '. ' (not on decimals such as $2.5)."""
    s0 = max(text.rfind(". ", 0, i0), text.rfind("; ", 0, i0))
    s1 = text.find(". ", i1)
    return text[s0 + 2 if s0 >= 0 else 0:s1 if s1 > 0 else len(text)].strip()


def issuer_candidates(symbol: str, cik: int, start: str = "2001-01-01", end: Optional[str] = None) -> pd.DataFrame:
    hits = sec.fts_search(AUTH_QUERY, ciks=[cik], forms="8-K", start=start, end=end)
    if not len(hits):
        return pd.DataFrame(columns=CAND_COLS)
    hits = hits[hits["cik"] == int(cik)].copy()
    acc = sec.acceptance_map(cik)
    ka = [sec.filing_known_at(a, fd, acc) for a, fd in zip(hits["adsh"], hits["file_date"])]
    hits["known_at"] = [k for k, _ in ka]
    hits["known_at_method"] = [m for _, m in ka]
    hits["with_earnings"] = hits["items"].fillna("").str.contains(r"\b2\.02\b")
    hits["symbol"] = symbol.upper()
    return hits[CAND_COLS]


def market_cap(portal, symbol: str, known_at: pd.Timestamp) -> Optional[float]:
    """shares outstanding (latest filed before known_at) x last raw close before known_at."""
    try:
        f = portal.fundamentals([symbol], ["shares_outstanding"], known_at)
        if f is None or not len(f):
            return None
        sh = float(f["value"].iloc[0])
        b = portal.bars(symbol, "1d", known_at - pd.Timedelta(days=10), known_at, known_at)
        if not len(b):
            return None
        return sh * float(b["close"].iloc[-1])
    except Exception:
        return None


#: seconds between www.sec.gov document requests (<= 4/s; SEC fair access allows 10/s)
SEC_DOC_MIN_INTERVAL = 0.25
MAX_PCT_MCAP = 1.0


def build_events(cands: pd.DataFrame, portal, fetch=sec.fetch_document, skip_earnings: bool = True) -> pd.DataFrame:
    """Candidates -> BUYBACK events (needs document text; see module doc). skip_earnings: do not read
    documents of 8-Ks carrying Item 2.02 -- S045 excludes earnings-bundled releases, so their amounts
    are never used (saves ~2/3 of the requests and cache); such filings get no event."""
    from .. import config
    config.HOST_MIN_INTERVAL["www.sec.gov"] = max(config.HOST_MIN_INTERVAL.get("www.sec.gov", 0.0),
                                                   SEC_DOC_MIN_INTERVAL)
    if skip_earnings and len(cands):
        cands = cands[~cands["with_earnings"].astype(bool)]
    rows = []
    for r in cands.itertuples(index=False):
        try:
            txt = html_to_text(fetch(r.cik, r.adsh, r.filename))
        except sec.SecContactRequired:
            raise
        except FetchError as e:
            log.warning("document %s %s unavailable: %s", r.adsh, r.filename, e)
            continue
        a = parse_authorization(txt, filed=pd.Timestamp(r.known_at))
        if not a:
            continue
        ka = pd.Timestamp(r.known_at)
        mcap = market_cap(portal, r.symbol, ka)
        amt = a["amount_usd"]
        if amt is None and a.get("shares") and mcap:
            b = portal.bars(r.symbol, "1d", ka - pd.Timedelta(days=10), ka, ka)
            amt = a["shares"] * float(b["close"].iloc[-1]) if len(b) else None
        if amt is None or not mcap or not (0 < amt / mcap <= MAX_PCT_MCAP):
            continue                     # > 100% of market cap: share-count / unit error, not a real program
        rows.append({"kind": KIND, "symbol": r.symbol, "cik": r.cik, "known_at": ka, "effective_at": ka,
                     "amount_usd": float(amt), "pct_mcap": float(amt) / mcap, "market_cap": mcap,
                     "with_earnings": bool(r.with_earnings), "accn": r.adsh, "filename": r.filename,
                     "items": r.items, "sentence": a["sentence"], "known_at_method": r.known_at_method,
                     "fidelity": int(Fidelity.T2_PROXY), "source": SOURCE})
    ev = pd.DataFrame(rows)
    if len(ev):   # one event per filing: the largest authorization stated in any of its documents
        ev = ev.sort_values("amount_usd").drop_duplicates(["symbol", "accn"], keep="last").sort_values("known_at")
    return ev


def build(symbols: Optional[Iterable[str]] = None, store: Optional[ParquetStore] = None, portal=None,
          start: str = "2001-01-01", text: bool = True, reuse_candidates: bool = False) -> dict:
    store = store or default_store()
    if reuse_candidates and store.exists("reference", SOURCE, "BUYBACK_CANDIDATES"):
        return _events_from(store.read("reference", SOURCE, "BUYBACK_CANDIDATES"), store, portal, symbols)
    targets = default_targets(store)
    if symbols:
        want = {s.upper() for s in symbols}
        targets = [t for t in targets if t[0] in want]
    parts, stats = [], {"issuers": len(targets), "candidates": 0, "with_earnings": 0, "errors": 0}
    for sym, cik in targets:
        try:
            c = issuer_candidates(sym, cik, start)
        except FetchError as e:
            stats["errors"] += 1
            log.warning("EFTS failed for %s: %s", sym, e)
            continue
        if len(c):
            parts.append(c)
    if not parts:
        return stats
    cands = pd.concat(parts, ignore_index=True)
    cands["known_at"] = pd.to_datetime(cands["known_at"], utc=True)
    stats["candidates"] = len(cands)
    stats["candidate_filings"] = int(cands["adsh"].nunique())
    stats["with_earnings"] = int(cands.drop_duplicates("adsh")["with_earnings"].sum())
    store.write("reference", SOURCE, "BUYBACK_CANDIDATES", cands, key=["symbol", "adsh", "filename"],
                replace=not symbols, extra={"query": AUTH_QUERY})
    if not text:
        return stats
    if portal is None:
        from ..portal import Portal
        portal = Portal(store)
    try:
        ev = build_events(cands, portal)
    except sec.SecContactRequired as e:
        stats["events"] = 0
        stats["needs_user"] = str(e)
        return stats
    ev = flag_earnings_bundling(identity_filter(ev, portal), store)
    stats["events"] = len(ev)
    if len(ev):
        stats["with_earnings_nearby"] = int(ev["earnings_within_1d"].sum())
        store.write("events", SOURCE, KIND, ev, key=["symbol", "accn"], replace=not symbols,
                    fidelity=int(Fidelity.T2_PROXY))
    return stats


def _events_from(cands: pd.DataFrame, store: ParquetStore, portal, symbols=None) -> dict:
    if symbols:
        cands = cands[cands["symbol"].isin({x.upper() for x in symbols})]
    if portal is None:
        from ..portal import Portal
        portal = Portal(store)
    stats = {"candidates": len(cands), "filings_read": int(cands.loc[~cands["with_earnings"].astype(bool), "adsh"].nunique())}
    try:
        ev = build_events(cands, portal)
    except sec.SecContactRequired as e:
        return {**stats, "events": 0, "needs_user": str(e)}
    ev = flag_earnings_bundling(identity_filter(ev, portal), store)
    stats["events"] = len(ev)
    if len(ev):
        stats["with_earnings_nearby"] = int(ev["earnings_within_1d"].sum())
        store.write("events", SOURCE, KIND, ev, key=["symbol", "accn"], replace=not symbols,
                    fidelity=int(Fidelity.T2_PROXY))
    return stats


def flag_earnings_bundling(ev: pd.DataFrame, store: ParquetStore, window_hours: float = 24.0) -> pd.DataFrame:
    """Also mark `with_earnings` when the issuer filed an 8-K Item 2.02 (events/sec EARNINGS_RELEASE)
    within +-window_hours of the announcement: a buyback released alongside earnings in a separate 8-K
    is still "bundled with earnings" (S045 excludes it). `earnings_within_1d` records the test."""
    if ev is None or not len(ev):
        return ev
    ev = ev.copy()
    near = []
    for r in ev.itertuples(index=False):
        e = store.read("events", "sec", f"EARNINGS_RELEASE_{str(r.symbol).upper()}")
        hit = False
        if len(e):
            dt = (pd.to_datetime(e["known_at"], utc=True) - pd.Timestamp(r.known_at)).abs()
            if "cik" in e.columns and pd.notna(r.cik):
                dt = dt[(e["cik"].isna() | (e["cik"] == r.cik)).to_numpy()]
            hit = bool((dt <= pd.Timedelta(hours=window_hours)).any())
        near.append(hit)
    ev["earnings_within_1d"] = near
    ev["with_earnings"] = ev["with_earnings"].astype(bool) | ev["earnings_within_1d"]
    return ev

