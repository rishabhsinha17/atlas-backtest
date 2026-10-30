"""Dividend DECLARATION timestamps from SEC EDGAR (events kind 'DIVIDEND_DECL').

Why: Yahoo / DoltHub dividends carry ex-date and amount but no declaration date (their known_at is
the close of the session before the ex-date, T2_PROXY). S042 needs "the next announced cash
dividend recorded point-in-time".

Sources tried (2026-10-30):
  * Nasdaq dividend-history API (api.nasdaq.com/api/quote/{s}/dividends): answers the honest UA,
    but only for Nasdaq-listed symbols (SPY/DIA/IWM/JNJ: "not available"; QQQ: declaration 'N/A')
    and its robots.txt is `Disallow: /` -> NOT used.
  * DoltHub `dividend` table (cached): act_symbol, ex_date, amount only -> no declaration date.
  * SEC EDGAR full-text search (efts.sec.gov, 2001+, plain UA accepted) -> USED here.
  * ETF sponsors' distribution pages (SPY / QQQ / IWM / DIA): no 8-Ks and no free PIT archive.

Method (per issuer CIK):
  1. EFTS query DECL_QUERY ("declared a quarterly" OR "declared a cash dividend" OR ...) on forms
     8-K (incl. exhibits such as EX-99.1 press releases), 2001+ -> documents that contain a
     declaration phrase. known_at of a hit = the filing's SEC acceptance time (submissions JSON;
     fallback filed date 17:30 ET, never early). The press release itself may precede the 8-K,
     so this is an UPPER bound on the announcement time (conservative).
  2. Each regular CASH_DIVIDEND (ex-date E, amount from the project's corporate actions) is
     matched to the LATEST hit accepted in (E - LOOKBACK_DAYS, open of E) such that no other
     cash-dividend ex-date of the issuer lies between the hit and E (the hit is about THIS
     dividend, not an earlier one). Latest = conservative (later known_at).
  3. Unmatched dividends get no DIVIDEND_DECL event (consumers must treat them as not
     point-in-time visible).

Fidelity T2_PROXY: the timestamp is an exact SEC acceptance time, but the hit -> dividend
association is inferred from timing (the search index returns no text; reading the filing text
needs SEC_USER_AGENT, see sec.fetch_document). Amount / ex-date in the payload come from the
corporate-action record (the declaration announces them; not re-read from the filing).

Event columns: kind, symbol, cik, known_at (UTC), effective_at (ex-date open, UTC), ex_date (naive),
amount, accn, items, file_type, filename, lead_days (calendar days hit -> ex-date), known_at_method,
match_rule, fidelity, source.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from atlasbt.core.clock import et
from atlasbt.core.types import Fidelity

from ..http import FetchError
from ..store import ParquetStore, default_store
from . import sec

log = logging.getLogger("atlasbt.data.dividends_decl")
KIND = "DIVIDEND_DECL"
SOURCE = "sec_fts"
LOOKBACK_DAYS = 75
DECL_QUERY = ('"declared a quarterly" OR "declared a cash dividend" OR "declared a regular" OR '
              '"declared its regular" OR "declared its quarterly" OR "declared a monthly" OR '
              '"declared a semi-annual" OR "declared a semiannual" OR "declared a dividend" OR '
              '"approved a quarterly" OR "declared quarterly" OR "declared an increased"')
MATCH_RULE = (f"latest 8-K/exhibit containing a declaration phrase accepted in (ex - {LOOKBACK_DAYS}d, "
              f"ex-date open) with no earlier ex-date in between")
#: documents that can announce a declaration: the 8-K body or a press-release exhibit (EX-99.x), filed under
#: an announcement item (2.02 results, 7.01 Reg FD, 8.01 other events). Contracts, indentures, charters
#: and bylaws (EX-1/2/3/4/10, "... CHARTER") also contain "declared a dividend" and are excluded.
DOC_TYPE_RE = r"^(?!.*CERT)(?:8-K(?:/A)?|EX-99(?:[.\w()]*)?(?: PR RELEASE| PDF)?)$"
ANNOUNCE_ITEMS = ("2.02", "7.01", "8.01")
EVENT_COLS = ["kind", "symbol", "cik", "known_at", "effective_at", "ex_date", "amount", "accn", "items",
              "file_type", "filename", "lead_days", "known_at_method", "match_rule", "fidelity", "source"]


def match_declarations(divs: pd.DataFrame, hits: pd.DataFrame, symbol: str, cik: Optional[int]) -> pd.DataFrame:
    """divs: ex_date (naive), amount. hits: adsh, known_at (UTC), known_at_method, items, file_type,
    filename. -> DIVIDEND_DECL event rows (see module doc). Pure function (tested offline)."""
    if divs is None or not len(divs) or hits is None or not len(hits):
        return pd.DataFrame(columns=EVENT_COLS)
    d = divs.copy()
    d["ex_date"] = pd.to_datetime(d["ex_date"]).dt.tz_localize(None).dt.normalize()
    d = d.sort_values("ex_date").drop_duplicates("ex_date", keep="first").reset_index(drop=True)
    h = announcement_docs(hits)
    if not len(h):
        return pd.DataFrame(columns=EVENT_COLS)
    h["known_at"] = pd.to_datetime(h["known_at"], utc=True)
    h = h.sort_values("known_at").drop_duplicates("adsh", keep="first").reset_index(drop=True)
    ka = h["known_at"].to_numpy(dtype="datetime64[ns]")
    rows = []
    prev_ex_open = None
    for r in d.itertuples(index=False):
        ex_open = et(r.ex_date, "09:30")
        lo = ex_open - pd.Timedelta(days=LOOKBACK_DAYS)
        if prev_ex_open is not None:
            lo = max(lo, prev_ex_open)          # the hit must postdate the previous ex-date
        prev_ex_open = ex_open
        m = (ka > lo.tz_convert("UTC").tz_localize(None).to_datetime64()) & \
            (ka < ex_open.tz_convert("UTC").tz_localize(None).to_datetime64())
        idx = np.flatnonzero(m)
        if not len(idx):
            continue
        x = h.iloc[idx[-1]]
        k = pd.Timestamp(x["known_at"])
        rows.append({"kind": KIND, "symbol": symbol.upper(), "cik": cik, "known_at": k,
                     "effective_at": ex_open.tz_convert("UTC"), "ex_date": r.ex_date, "amount": float(r.amount),
                     "accn": x["adsh"], "items": x.get("items"), "file_type": x.get("file_type"),
                     "filename": x.get("filename"),
                     "lead_days": float((ex_open.tz_convert("UTC") - k).total_seconds() / 86400.0),
                     "known_at_method": x.get("known_at_method", "acceptance"), "match_rule": MATCH_RULE,
                     "fidelity": int(Fidelity.T2_PROXY), "source": SOURCE})
    return pd.DataFrame(rows, columns=EVENT_COLS)


def announcement_docs(hits: pd.DataFrame) -> pd.DataFrame:
    """Hits that are announcement documents (DOC_TYPE_RE) of 8-Ks filed under ANNOUNCE_ITEMS."""
    ft = hits["file_type"].fillna("").astype(str).str.upper().str.strip()
    it = hits["items"].fillna("").astype(str)
    ok = ft.str.match(DOC_TYPE_RE) & it.apply(lambda x: any(i in x.split(",") for i in ANNOUNCE_ITEMS))
    return hits[ok.to_numpy()].copy()


def issuer_hits(cik: int, start: str = "2001-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """EFTS declaration-phrase hits for one CIK with known_at (acceptance) attached."""
    hits = sec.fts_search(DECL_QUERY, ciks=[cik], forms="8-K", start=start, end=end)
    if not len(hits):
        return hits
    hits = hits[hits["cik"] == int(cik)].copy()     # first filer must be the issuer itself
    acc = sec.acceptance_map(cik)
    ka = [sec.filing_known_at(a, fd, acc) for a, fd in zip(hits["adsh"], hits["file_date"])]
    hits["known_at"] = [k for k, _ in ka]
    hits["known_at_method"] = [m for _, m in ka]
    return hits


def cash_dividends(portal, symbol: str, start="2001-01-01", end=None) -> pd.DataFrame:
    """Regular CASH_DIVIDEND actions (ex_date, amount) the portal serves for `symbol` (identity-
    verified segments; SPECIAL_DIVIDEND excluded)."""
    from atlasbt.core.types import ActionType
    end = end or pd.Timestamp.now().normalize()
    acts = portal.corporate_actions([symbol], pd.Timestamp(start), pd.Timestamp(end),
                                    pd.Timestamp("2262-01-01", tz="UTC"))
    rows = [(pd.Timestamp(a.ex_date).tz_localize(None) if pd.Timestamp(a.ex_date).tzinfo else pd.Timestamp(a.ex_date),
             float(a.amount)) for a in acts if a.type == ActionType.CASH_DIVIDEND and a.amount > 0]
    return pd.DataFrame(rows, columns=["ex_date", "amount"])


def default_targets(store: ParquetStore) -> list[tuple[str, int]]:
    """(symbol, CIK) of every S&P 500 PIT-union ticker the SEC security master resolved."""
    sm = store.read("reference", "sec", "security_master")
    if not len(sm):
        return []
    sm = sm[sm["cik"].notna()]
    return sorted({(str(s).upper(), int(c)) for s, c in zip(sm["symbol"], sm["cik"])})


def build(symbols: Optional[Iterable[str]] = None, store: Optional[ParquetStore] = None, portal=None,
          start: str = "2001-01-01") -> dict:
    """Fetch EFTS hits per issuer, match them to the portal's cash dividends and store
    events/sec_fts/DIVIDEND_DECL (+ reference/sec_fts/DIVIDEND_DECL_HITS for audit)."""
    store = store or default_store()
    if portal is None:
        from ..portal import Portal
        portal = Portal(store)
    targets = default_targets(store)
    if symbols:
        want = {s.upper() for s in symbols}
        targets = [t for t in targets if t[0] in want]
    ev_parts, hit_parts, div_parts, stats = [], [], [], {"issuers": 0, "with_dividends": 0, "dividends": 0, "matched": 0,
                                          "hits": 0, "errors": 0}
    for sym, cik in targets:
        stats["issuers"] += 1
        try:
            divs = cash_dividends(portal, sym, start)
        except Exception as e:          # identity unknown etc.: no dividends served
            log.debug("%s: no dividends (%s)", sym, e)
            continue
        if not len(divs):
            continue
        stats["with_dividends"] += 1
        stats["dividends"] += len(divs)
        div_parts.append(divs.assign(symbol=sym, cik=cik))
        try:
            hits = issuer_hits(cik, start)
        except FetchError as e:
            stats["errors"] += 1
            log.warning("EFTS failed for %s (CIK %s): %s", sym, cik, e)
            continue
        stats["hits"] += len(hits)
        if len(hits):
            hit_parts.append(hits.assign(symbol=sym))
        ev = match_declarations(divs, hits, sym, cik)
        stats["matched"] += len(ev)
        if len(ev):
            ev_parts.append(ev)
    if hit_parts:
        hh = pd.concat(hit_parts, ignore_index=True)
        hh["known_at"] = pd.to_datetime(hh["known_at"], utc=True)
        store.write("reference", SOURCE, "DIVIDEND_DECL_HITS", hh, key=["symbol", "adsh", "filename"], replace=not symbols)
    if div_parts:
        store.write("reference", SOURCE, "DIVIDEND_DECL_DIVS", pd.concat(div_parts, ignore_index=True),
                    key=["symbol", "ex_date"], replace=not symbols)
    if ev_parts:
        ev = identity_filter(pd.concat(ev_parts, ignore_index=True), portal)
        store.write("events", SOURCE, KIND, ev, key=["symbol", "ex_date"], replace=not symbols,
                    fidelity=int(Fidelity.T2_PROXY), extra={"match_rule": MATCH_RULE, "query": DECL_QUERY})
    log.info("DIVIDEND_DECL: %s", stats)
    return stats


def rematch(store: Optional[ParquetStore] = None, portal=None) -> dict:
    """Re-derive events/sec_fts/DIVIDEND_DECL from the stored hits and dividends (no network) --
    after a change of the matching rule."""
    store = store or default_store()
    hits = store.read("reference", SOURCE, "DIVIDEND_DECL_HITS")
    divs = store.read("reference", SOURCE, "DIVIDEND_DECL_DIVS")
    parts = []
    for (sym, cik), d in divs.groupby(["symbol", "cik"]):
        h = hits[hits["symbol"] == sym]
        ev = match_declarations(d[["ex_date", "amount"]], h, sym, int(cik))
        if len(ev):
            parts.append(ev)
    ev = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=EVENT_COLS)
    if len(ev):
        if portal is None:
            from ..portal import Portal
            portal = Portal(store)
        ev = identity_filter(ev, portal)
    store.write("events", SOURCE, KIND, ev, key=["symbol", "ex_date"], replace=True,
                fidelity=int(Fidelity.T2_PROXY), extra={"match_rule": MATCH_RULE, "query": DECL_QUERY})
    return {"dividends": len(divs), "matched": len(ev), "issuers": int(ev["symbol"].nunique()) if len(ev) else 0}


def identity_filter(ev: pd.DataFrame, portal) -> pd.DataFrame:
    """Keep events whose issuer CIK is the issuer the ticker denoted on the announcement date
    (fail closed: unknown identity -> dropped). Guards reused tickers (e.g. BBT = BB&T until 2019,
    Berkshire Hills/Beacon later)."""
    if ev is None or not len(ev):
        return ev
    keep = []
    for r in ev.itertuples(index=False):
        d = pd.Timestamp(r.known_at).tz_convert("America/New_York").tz_localize(None).normalize()
        try:
            c = int(portal._identity_by_date(str(r.symbol), [d])[0][0])
        except Exception:
            c = 0
        keep.append(c != 0 and pd.notna(r.cik) and c == int(r.cik))
    return ev[keep].copy()
