"""ZipLime US earnings calendar (Hugging Face dataset ZipLime/earnings-calendar, rev f8d9be6,
Apache-2.0), on disk under data_raw/hf_ziplime_earnings_calendar (scout download; local loader).

455k 8-K Item 2.02 (Results of Operations) releases 2003-04-25 .. 2026-09-04, with
`announced_at` read from each filing's SGML header ACCEPTANCE-DATETIME (unambiguous US
Eastern, converted to UTC), `session` (pre_market / market_hours / after_close /
non_session) and `first_tradeable_session` (first NYSE session whose close is after the
acceptance; honours half days and closures). Tickers are resolved as of the day
(`ticker_confidence` in {as_of, carried_back, ...}); includes since-delisted companies.
Figures (eps_basic/diluted, revenue, net income, SUE) come from XBRL company facts, matched
to the release; they are attached with known_at = announcement (the press release carries
them) and the statement acceptance time is kept in `statement_accepted_at`.

Verification: EDGAR submissions-JSON `acceptanceDateTime` ("...Z") is Eastern local time for
a large share of pre-2021 filings (ZipLime README); we checked our own SEC-API rows against
ZipLime's header times (tests/data/test_earnings.py, docs/DATA.md).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from atlasbt.core.types import Fidelity

from .. import config
from ..store import ParquetStore, default_store

log = logging.getLogger("atlasbt.data.ziplime")
SOURCE = "ziplime"
ROOT = config.DATA_RAW / "hf_ziplime_earnings_calendar" / "data"
SESSION_TO_TIMING = {"pre_market": "BMO", "after_close": "AMC", "market_hours": "DURING",
                     "non_session": "UNKNOWN"}


def build_events(announcements: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    a = announcements[announcements["form"].str.startswith("8-K")].copy()
    fig = earnings[["accession_number", "period_end", "fiscal_year", "fiscal_period", "eps_basic", "eps_diluted",
                    "revenue", "net_income", "sue", "eps_quality", "statement_accepted_at"]]
    a = a.merge(fig, on="accession_number", how="left")
    known = a["announced_at"]
    out = pd.DataFrame({
        "kind": "EARNINGS", "symbol": a["ticker"].str.upper().str.replace(".", "-", regex=False),
        "known_at": known, "effective_at": known,
        "timing": a["session"].map(SESSION_TO_TIMING).fillna("UNKNOWN"),
        "eps_actual": a["eps_diluted"].where(a["eps_diluted"].notna(), a["eps_basic"]),
        "eps_estimate": np.nan, "estimate_known_at": pd.NaT,
        "fiscal_period": a["fiscal_period"], "period_end": pd.to_datetime(a["period_end"], errors="coerce"),
        "fiscal_year": a["fiscal_year"], "revenue": a["revenue"], "net_income": a["net_income"], "sue": a["sue"],
        "eps_quality": a["eps_quality"], "statement_accepted_at": a["statement_accepted_at"],
        "first_tradeable_session": pd.to_datetime(a["first_tradeable_session"], errors="coerce"),
        "accession": a["accession_number"], "cik": a["cik"], "ticker_confidence": a["ticker_confidence"],
        "acceptance_source": a["acceptance_source"], "items": a["items"],
        "fidelity": int(Fidelity.T1_BAR), "source": SOURCE})
    out = out[out["symbol"].notna() & (out["symbol"] != "")]
    return out.sort_values("known_at").reset_index(drop=True)


def load(store: Optional[ParquetStore] = None, root: Path = ROOT) -> dict:
    store = store or default_store()
    ann = pd.concat([pd.read_parquet(p) for p in sorted((root / "announcements").glob("*.parquet"))], ignore_index=True)
    ern = pd.concat([pd.read_parquet(p) for p in sorted((root / "earnings").glob("*.parquet"))], ignore_index=True)
    ev = build_events(ann, ern)
    store.write("events", SOURCE, "EARNINGS", ev, partition="year", time_col="known_at",
                key=["accession", "symbol"], replace=True, fidelity=int(Fidelity.T1_BAR),
                extra={"origin": "data_raw/hf_ziplime_earnings_calendar (Apache-2.0)",
                       "first_ts": str(ev["known_at"].min()), "last_ts": str(ev["known_at"].max())})
    return {"events": len(ev), "symbols": ev["symbol"].nunique(), "first": str(ev["known_at"].min()),
            "last": str(ev["known_at"].max()), "with_eps": int(ev["eps_actual"].notna().sum()),
            "timing": ev["timing"].value_counts().to_dict()}
