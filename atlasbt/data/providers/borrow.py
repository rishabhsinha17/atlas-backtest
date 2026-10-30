"""Stock borrow availability & fees.

iborrowdesk.com (Interactive Brokers' public short-stock availability, scraped by a third
party) was probed once politely on 2026-10-28: the connection timed out (host unreachable
from this environment) -> no historical borrow data. We therefore expose a configurable
DEFAULT model, always flagged T2_PROXY:

  * ETFs and S&P 500 members (easy-to-borrow, 'ETB'): fee = 0.30% p.a. (ARCHITECTURE: 0.25-0.5%)
  * other listed stocks with market cap unknown: 'GC+' tier 1.0% p.a.
  * leveraged/inverse ETFs: 2.0% p.a. (these are frequently hard-to-borrow)
  * shortable is always True (no locate data); strategies needing real HTB data must
    declare the deviation.
Override via env ATLASBT_BORROW_ETB, ATLASBT_BORROW_GC, ATLASBT_BORROW_HTB (decimals).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import pandas as pd

from atlasbt.core.types import Fidelity

from ..http import FetchError, HttpClient
from ..store import ParquetStore, default_store
from ..universes import CORE_ETFS, LEVERAGED_ETFS

log = logging.getLogger("atlasbt.data.borrow")
SOURCE = "borrow"
IBD_URL = "https://iborrowdesk.com/api/ticker/{t}"

ETB = float(os.environ.get("ATLASBT_BORROW_ETB", "0.003"))
GC = float(os.environ.get("ATLASBT_BORROW_GC", "0.01"))
HTB = float(os.environ.get("ATLASBT_BORROW_HTB", "0.02"))


def default_fee(symbol: str, is_etf: bool = False, is_index_member: bool = False) -> tuple[bool, float, Fidelity]:
    s = symbol.upper()
    if s in LEVERAGED_ETFS:
        return True, HTB, Fidelity.T2_PROXY
    if is_etf or is_index_member or s in CORE_ETFS:
        return True, ETB, Fidelity.T2_PROXY
    return True, GC, Fidelity.T2_PROXY


def probe_iborrowdesk(symbols: list[str], store: Optional[ParquetStore] = None) -> dict:
    """One polite attempt per symbol (no retries); stores whatever comes back."""
    store = store or default_store()
    http = HttpClient(max_retries=1, timeout=20)
    out = {}
    for s in symbols:
        try:
            body = http.get(IBD_URL.format(t=s.upper()), allow_404=True)
        except Exception as e:  # unreachable / blocked: record and stop
            out[s] = f"unavailable: {type(e).__name__}"
            log.warning("iborrowdesk unavailable (%s); using default borrow model", e)
            break
        if not body:
            out[s] = "404"
            continue
        try:
            d = json.loads(body)
            daily = pd.DataFrame(d.get("daily") or d.get("real_time") or [])
            if len(daily):
                daily["symbol"] = s.upper()
                store.write("borrow", "iborrowdesk", s, daily.reset_index(drop=True), replace=True,
                            fidelity=int(Fidelity.T1_BAR))
            out[s] = f"ok rows={len(daily)}"
        except ValueError:
            out[s] = "unparseable"
    store.write("reference", SOURCE, "iborrowdesk_probe",
                pd.DataFrame([{"checked_at": pd.Timestamp.now(tz="UTC"), "result": json.dumps(out)}]),
                key=["checked_at"])
    return out
