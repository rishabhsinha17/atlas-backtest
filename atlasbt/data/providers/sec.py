"""SEC EDGAR (data.sec.gov / www.sec.gov). Requires an honest User-Agent with a
contact e-mail and <= 10 requests/s (we use ~6.6/s with backoff).

  * company_tickers_exchange.json -> CIK <-> ticker map (CURRENT tickers only).
  * submissions/CIK##########.json (+ older 'files' pages) -> SIC code, names, former
    names, exchanges, and every filing with accessionNumber + acceptanceDateTime (UTC;
    verified: AAPL 8-K 2.02 2024-08-01T20:30:26Z = 16:30:26 EDT, the release time).
  * api/xbrl/companyfacts/CIK##########.json -> XBRL facts (2009+ for large filers).

Fundamentals long frame (canonical): symbol, field, value, period_end, known_at, source
  + period_start, duration_days, fy, fp, form, accn, unit, concept, filed.
known_at = acceptanceDateTime of the filing (accession number) that reported the fact.
If the accession is not in the submissions index: filed date 17:30 ET (EDGAR dates
filings accepted after 17:30 as the next business day, so this is never early).
Restatements: every (period, filing) pair is kept; the Portal picks, for each
(symbol, field), the latest period_end among facts with known_at <= as_of, and for that
period the most recently filed value -- i.e. exactly what an investor could see.

Duration fields are suffixed: _q (80-100 days), _fy (350-380 days), _ytd (other);
instant fields (balance sheet, shares) have no suffix. Q4 quarterly values are not
filed separately (only FY) -- not derived here.

8-K Item 2.02 (Results of Operations) filings -> events kind 'EARNINGS_RELEASE'
with known_at = acceptanceDateTime (an UPPER bound on the press-release time: the
8-K is often filed minutes after the wire release, occasionally hours).
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np
import pandas as pd

from atlasbt.core.clock import et
from atlasbt.core.types import Fidelity

from ..http import FetchError, client
from ..store import ParquetStore, default_store
from ..universes import sic_to_sector

log = logging.getLogger("atlasbt.data.sec")
SOURCE = "sec"
TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

FIELD_MAP: dict[str, list[str]] = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"],
    "net_income": ["NetIncomeLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "eps_basic": ["EarningsPerShareBasic"],
    "operating_income": ["OperatingIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "pretax_income": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"],
    "income_tax": ["IncomeTaxExpenseBenefit"],
    "interest_expense": ["InterestExpense"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
    "depreciation": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "buyback": ["PaymentsForRepurchaseOfCommonStock"],
    "dividends_paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "dividends_per_share": ["CommonStockDividendsPerShareDeclared"],
    "weighted_shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "stockholders_equity": ["StockholdersEquity",
                            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "inventory": ["InventoryNet"],
    "receivables": ["AccountsReceivableNetCurrent"],
    "ppe_net": ["PropertyPlantAndEquipmentNet"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "shares_outstanding": ["dei:EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"],
    # Fama-French book equity = SE + deferred taxes (- investment tax credit) - preferred.
    # Lists are in fallback order: the first concept a filing reports for a period wins.
    "preferred_stock": ["PreferredStockValue", "PreferredStockCarryingAmount",
                        "PreferredStockValueOutstanding", "PreferredStockIncludingAdditionalPaidInCapital"],
    "deferred_taxes": ["DeferredTaxLiabilitiesNoncurrent", "DeferredIncomeTaxLiabilitiesNet",
                       "DeferredTaxLiabilities", "DeferredIncomeTaxLiabilities"],
    "short_term_debt": ["DebtCurrent", "ShortTermBorrowings", "LongTermDebtCurrent",
                        "LongTermDebtAndCapitalLeaseObligationsCurrent"],
    "stock_issuance_proceeds": ["ProceedsFromIssuanceOfCommonStock", "ProceedsFromStockOptionsExercised"],
    "repurchase": ["PaymentsForRepurchaseOfCommonStock", "PaymentsForRepurchaseOfEquity"],
}
FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "40-F", "10-KT", "20-F/A", "8-K", "6-K"}


# ---------------------------------------------------------------------------
def fetch_ticker_map(store: Optional[ParquetStore] = None, ttl_hours: float = 24) -> dict:
    """www.sec.gov refuses requests whose UA lacks a contact (HTTP 403). Without SEC_USER_AGENT
    we degrade: stale cached copy if any, else CIKs from the Wikipedia S&P 500 table (current
    members only) -- data.sec.gov (submissions, companyfacts) accepts the plain UA."""
    store = store or default_store()
    try:
        d = json.loads(client().get(TICKERS_URL, ttl=ttl_hours * 3600))
        origin = "sec"
    except FetchError as e:
        raw = client().cached(TICKERS_URL)
        if raw is not None:
            d, origin = json.loads(raw), "stale_cache"
        else:
            cur = store.read("reference", "sp500", "current_constituents")
            if not len(cur):
                return {"symbol": "ticker_map", "rows": 0, "error": str(e)}
            df = pd.DataFrame({"cik": cur["cik"].astype(int), "name": cur["name"], "ticker": cur["symbol"],
                               "exchange": None})
            store.write("reference", SOURCE, "ticker_map", df, key=["ticker", "cik"], replace=True,
                        extra={"origin": "wikipedia_sp500_current"})
            return {"symbol": "ticker_map", "rows": len(df), "origin": "wikipedia_sp500_current"}
    df = pd.DataFrame(d["data"], columns=d["fields"])
    df["ticker"] = df["ticker"].str.upper()
    store.write("reference", SOURCE, "ticker_map", df, key=["ticker", "cik"], replace=True,
                extra={"origin": origin})
    return {"symbol": "ticker_map", "rows": len(df), "origin": origin}


def cik_for(symbol: str, store: Optional[ParquetStore] = None) -> Optional[int]:
    store = store or default_store()
    m = store.read("reference", SOURCE, "ticker_map")
    if not len(m):
        fetch_ticker_map(store)
        m = store.read("reference", SOURCE, "ticker_map")
    s = symbol.upper()
    for cand in (s, s.replace("-", "."), s.replace(".", "-"), s.replace("-", "")):
        hit = m[m["ticker"] == cand]
        if len(hit):
            return int(hit["cik"].iloc[0])
    # historical tickers (delisted / renamed): ZipLime's as-of ticker -> CIK resolution
    z = _ziplime_ciks(store)
    return z.get(s) or z.get(s.replace("-", "."))


_ZL: dict = {}


def _ziplime_ciks(store: ParquetStore) -> dict:
    if "map" not in _ZL:
        ev = store.read("events", "ziplime", "EARNINGS", columns=["symbol", "cik", "known_at"])
        if len(ev):
            ev = ev.sort_values("known_at")
            ev = ev[ev["cik"].notna()]
            _ZL["map"] = {s: int(c) for s, c in zip(ev["symbol"], ev["cik"])}   # latest mapping wins
        else:
            _ZL["map"] = {}
    return _ZL["map"]


def _submissions(cik: int, ttl_days: float = 7) -> tuple[dict, pd.DataFrame]:
    main = json.loads(client().get(SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json"), ttl=ttl_days * 86400))
    frames_ = [pd.DataFrame(main["filings"]["recent"])]
    for f in main["filings"].get("files", []):
        # older pages never change -> cache forever
        older = json.loads(client().get(SUBMISSIONS_URL.format(name=f["name"])))
        frames_.append(pd.DataFrame(older))
    filings = pd.concat(frames_, ignore_index=True)
    return main, filings


def parse_filings(filings: pd.DataFrame) -> pd.DataFrame:
    f = filings.copy()
    f["acceptance"] = pd.to_datetime(f["acceptanceDateTime"], utc=True, errors="coerce")
    f["filingDate"] = pd.to_datetime(f["filingDate"], errors="coerce")
    f["items"] = f.get("items", pd.Series("", index=f.index)).fillna("").astype(str)
    return f


def earnings_8k_events(symbol: str, filings: pd.DataFrame) -> pd.DataFrame:
    f = filings[filings["form"].isin(["8-K", "8-K/A"]) & filings["items"].str.contains(r"2\.02")]
    if not len(f):
        return pd.DataFrame()
    known = f["acceptance"]
    t_et = known.dt.tz_convert("America/New_York")
    hhmm = t_et.dt.hour * 100 + t_et.dt.minute
    timing = np.where(hhmm < 930, "BMO", np.where(hhmm >= 1600, "AMC", "DURING"))
    return pd.DataFrame({"kind": "EARNINGS_RELEASE", "symbol": symbol.upper(), "known_at": known.to_numpy(),
                         "effective_at": known.to_numpy(), "timing": timing, "form": f["form"].to_numpy(),
                         "accn": f["accessionNumber"].to_numpy(), "items": f["items"].to_numpy(),
                         "filing_date": f["filingDate"].to_numpy(), "report_date": f["reportDate"].to_numpy(),
                         "fidelity": int(Fidelity.T1_BAR), "source": SOURCE}).sort_values("known_at")


def parse_companyfacts(symbol: str, facts: dict, accept: dict) -> pd.DataFrame:
    """facts JSON -> fundamentals long frame. accept: accn -> acceptance UTC Timestamp."""
    rows = []
    tax = facts.get("facts", {})
    for field, concepts in FIELD_MAP.items():
        for concept in concepts:
            ns, name = concept.split(":") if ":" in concept else ("us-gaap", concept)
            node = tax.get(ns, {}).get(name)
            if not node:
                continue
            for unit, arr in node.get("units", {}).items():
                for x in arr:
                    if x.get("form") not in FORMS:
                        continue
                    rows.append((field, concept, unit, x.get("start"), x["end"], x["val"], x.get("accn"),
                                 x.get("fy"), x.get("fp"), x.get("form"), x.get("filed")))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["base_field", "concept", "unit", "period_start", "period_end", "value",
                                     "accn", "fy", "fp", "form", "filed"])
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["period_start"] = pd.to_datetime(df["period_start"])
    df["filed"] = pd.to_datetime(df["filed"])
    dur = (df["period_end"] - df["period_start"]).dt.days
    df["duration_days"] = dur
    suffix = np.where(dur.isna(), "", np.where(dur.between(80, 100), "_q", np.where(dur.between(350, 380), "_fy", "_ytd")))
    df["field"] = df["base_field"] + suffix
    acc = df["accn"].map(accept)
    fallback = pd.DatetimeIndex([et(d, "17:30") if pd.notna(d) else pd.NaT for d in df["filed"]])
    df["known_at"] = acc.where(acc.notna(), pd.Series(fallback, index=df.index))
    df["known_at_method"] = np.where(acc.notna(), "acceptance", "filed_1730ET")
    df["symbol"] = symbol.upper()
    df["source"] = SOURCE
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype(float)
    # keep first concept per (field, period, accn) by FIELD_MAP priority
    pri = {c: i for cs in FIELD_MAP.values() for i, c in enumerate(cs)}
    df["_p"] = df["concept"].map(pri)
    df = df.sort_values("_p").drop_duplicates(["field", "period_start", "period_end", "accn"], keep="first")
    cols = ["symbol", "field", "value", "period_end", "known_at", "source", "period_start", "duration_days",
            "fy", "fp", "form", "accn", "unit", "concept", "filed", "known_at_method"]
    return df[cols].sort_values(["field", "period_end", "known_at"]).reset_index(drop=True)


def fetch_company(symbol: str, store: Optional[ParquetStore] = None, facts: bool = True) -> dict:
    store = store or default_store()
    cik = cik_for(symbol, store)
    if cik is None:
        return {"symbol": symbol, "cik": None}
    try:
        main, filings = _submissions(cik)
    except FetchError as e:
        return {"symbol": symbol, "cik": cik, "error": f"SEC refused/unavailable: {e.status}"}
    filings = parse_filings(filings)
    sic = main.get("sic")
    master = pd.DataFrame([{
        "symbol": symbol.upper(), "cik": cik, "name": main.get("name"), "sic": sic,
        "sic_description": main.get("sicDescription"), "sector_sic": sic_to_sector(sic),
        "entity_type": main.get("entityType"), "category": main.get("category"),
        "exchanges": ",".join(x or "" for x in main.get("exchanges", [])),
        "tickers": ",".join(main.get("tickers", [])), "fiscal_year_end": main.get("fiscalYearEnd"),
        "former_names": json.dumps(main.get("formerNames", [])),
        "first_filing": filings["filingDate"].min(), "last_filing": filings["filingDate"].max(),
        "fetched_at": pd.Timestamp.now(tz="UTC")}])
    store.write("reference", SOURCE, "security_master", master, key=["symbol"])
    ev = earnings_8k_events(symbol, filings)
    if len(ev):
        store.write("events", SOURCE, f"EARNINGS_RELEASE_{symbol.upper()}", ev, key=["accn"], replace=True,
                    fidelity=int(Fidelity.T1_BAR))
    out = {"symbol": symbol, "cik": cik, "sic": sic, "filings": len(filings), "earnings_8k": len(ev)}
    if facts:
        body = client().get(FACTS_URL.format(cik=cik), allow_404=True, ttl=7 * 86400)
        if body:
            accept = dict(zip(filings["accessionNumber"], filings["acceptance"]))
            fund = parse_companyfacts(symbol, json.loads(body), accept)
            if len(fund):
                store.write("fundamentals", SOURCE, symbol, fund, replace=True, fidelity=int(Fidelity.T1_BAR),
                            key=["field", "period_start", "period_end", "accn"])
                out["fundamental_rows"] = len(fund)
                out["fields"] = fund["field"].nunique()
    return out


# ---------------------------------------------------------------------------
# EDGAR full-text search (efts.sec.gov, 2001+) and filing documents (www.sec.gov/Archives)
#
# efts.sec.gov answers the plain UA `atlasbt-research/0.1` (verified 2026-10-30: HTTP 200).
# www.sec.gov/Archives refuses it (HTTP 403 "undeclared automated tool") -> reading filing TEXT
# needs SEC_USER_AGENT (a contact string supplied by the user); without it `fetch_document`
# raises SecContactRequired and callers keep only what the search index itself says
# (accession, form, items, exhibit type, filing date). Search hits carry no text snippets.
FTS_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{adsh_nodash}/{filename}"
FTS_PAGE = 100            # hits per page (EFTS default)
FTS_MAX = 10_000          # EFTS refuses offsets beyond 10,000


class SecContactRequired(RuntimeError):
    """www.sec.gov needs a User-Agent with a contact (env SEC_USER_AGENT) -- NEEDS USER."""


def fts_search(q: str, *, ciks=None, forms: str = "8-K", start: str = "2001-01-01",
               end: Optional[str] = None, max_hits: int = FTS_MAX, ttl: Optional[float] = None) -> pd.DataFrame:
    """All EDGAR full-text-search hits for query `q` (EFTS syntax: "phrase", OR, AND), one row per
    matching DOCUMENT: adsh, cik (first filer), ciks, file_date, form, root_form, file_type,
    file_description, items, filename, display_name. Paged 100 at a time (polite: the shared
    HTTP client throttles per host and backs off on 429/5xx; responses are disk-cached)."""
    end = end or pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    params = {"q": q, "forms": forms, "dateRange": "custom", "startdt": start, "enddt": end}
    if ciks:
        params["ciks"] = ",".join(f"{int(c):010d}" for c in ([ciks] if isinstance(ciks, (int, str)) else ciks))
    if ttl is None:   # a window that ended > 30 days ago is final; recent windows are re-read daily
        ttl = None if pd.Timestamp(end) < pd.Timestamp.now() - pd.Timedelta(days=30) else 86400.0
    rows, frm, total = [], 0, None
    while True:
        p = dict(params, **({"from": frm} if frm else {}))
        d = json.loads(client().get(FTS_URL, params=p, ttl=ttl))
        hits = d.get("hits", {})
        total = int(hits.get("total", {}).get("value", 0)) if total is None else total
        page = hits.get("hits", [])
        for h in page:
            s = h.get("_source", {})
            _id = str(h.get("_id", ""))
            rows.append({"adsh": s.get("adsh") or _id.split(":")[0],
                         "cik": int(s["ciks"][0]) if s.get("ciks") else None,
                         "ciks": ",".join(str(int(c)) for c in s.get("ciks") or []),
                         "file_date": s.get("file_date"), "form": s.get("form"),
                         "root_form": ",".join(s.get("root_forms") or []),
                         "file_type": s.get("file_type"), "file_description": s.get("file_description"),
                         "items": ",".join(s.get("items") or []),
                         "filename": _id.split(":", 1)[1] if ":" in _id else None,
                         "display_name": (s.get("display_names") or [None])[0]})
        frm += len(page)
        if not page or frm >= min(total, max_hits, FTS_MAX):
            break
    df = pd.DataFrame(rows, columns=["adsh", "cik", "ciks", "file_date", "form", "root_form", "file_type",
                                     "file_description", "items", "filename", "display_name"])
    if len(df):
        df["file_date"] = pd.to_datetime(df["file_date"], errors="coerce")
    df.attrs["total"] = total or 0
    return df


def acceptance_map(cik: int) -> dict:
    """accession number -> acceptance datetime (UTC) from the cached / fetched submissions JSON
    (data.sec.gov accepts the plain UA). Empty dict when unavailable."""
    try:
        _, filings = _submissions(int(cik))
    except (FetchError, ValueError, KeyError) as e:
        log.warning("submissions for CIK %s unavailable: %s", cik, e)
        return {}
    f = parse_filings(filings)
    return dict(zip(f["accessionNumber"], f["acceptance"]))


def filing_known_at(adsh: str, file_date, accept: dict) -> tuple[pd.Timestamp, str]:
    """(known_at UTC, method): the acceptance time when known, else file date 17:30 ET -- never
    early (EDGAR dates filings accepted after 17:30 ET as the next business day)."""
    a = accept.get(adsh)
    if a is not None and pd.notna(a):
        return pd.Timestamp(a), "acceptance"
    return et(pd.Timestamp(file_date).normalize(), "17:30"), "filed_1730ET"


def fetch_document(cik: int, adsh: str, filename: str) -> bytes:
    """Raw bytes of one filing document from www.sec.gov/Archives (cached forever). Requires
    SEC_USER_AGENT (www.sec.gov refuses the plain UA with 403): raises SecContactRequired
    without it -- NEEDS USER: "SEC_USER_AGENT contact string"."""
    from .. import config
    if not config.SEC_USER_AGENT:
        raise SecContactRequired("www.sec.gov/Archives refuses requests without a contact in the User-Agent; "
                                 "set SEC_USER_AGENT (NEEDS USER)")
    url = ARCHIVES_URL.format(cik=int(cik), adsh_nodash=adsh.replace("-", ""), filename=filename)
    return client().get(url)
