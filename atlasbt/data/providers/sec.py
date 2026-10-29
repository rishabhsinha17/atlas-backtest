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
