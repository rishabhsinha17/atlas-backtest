"""Stock borrow availability & fees.

Real history: `providers/iborrowdesk.py` (IBorrowDesk keyed v2 API = Interactive Brokers stock-loan
data, daily since 2015-07-13; needs IBORROWDESK_API_KEY -- NEEDS USER, paid). When a symbol has
stored rows, `Portal.borrow` serves them (T1_BAR). Otherwise (no key / before the first row /
stale) the configurable DEFAULT model below is used, always flagged T2_PROXY:

  * ETFs and S&P 500 members (easy-to-borrow, 'ETB'): fee = 0.30% p.a. (ARCHITECTURE: 0.25-0.5%)
  * other listed stocks with market cap unknown: 'GC+' tier 1.0% p.a.
  * leveraged/inverse ETFs: 2.0% p.a. (these are frequently hard-to-borrow)
  * shortable is always True (no locate data); strategies needing real HTB data must
    declare the deviation.
Override via env ATLASBT_BORROW_ETB, ATLASBT_BORROW_GC, ATLASBT_BORROW_HTB (decimals).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from atlasbt.core.types import Fidelity

from ..store import ParquetStore
from ..universes import CORE_ETFS, LEVERAGED_ETFS

log = logging.getLogger("atlasbt.data.borrow")
SOURCE = "borrow"

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
    """Kept for the fetch CLI: delegates to the keyed v2 API (iborrowdesk.fetch_daily). The site's
    internal /api/ticker endpoint (used here until 2026-10-29) blocks scripted clients and is no
    longer called."""
    from . import iborrowdesk
    return iborrowdesk.fetch_daily(symbols, store=store)
