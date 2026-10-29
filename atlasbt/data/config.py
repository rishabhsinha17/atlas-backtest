"""Paths, identity and global settings for the data layer."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = Path(os.environ.get("ATLASBT_CACHE", REPO_ROOT / "cache"))
STORE_ROOT = CACHE_ROOT / "store"          # canonical parquet store
RAW_ROOT = CACHE_ROOT / "raw"              # raw HTTP responses (never fetched twice)
DATA_RAW = REPO_ROOT / "data_raw"          # scout samples (read-only for us)

# Honest, non-browser identification. No personal contact data is embedded in code.
# Override with ATLASBT_USER_AGENT.
USER_AGENT = os.environ.get("ATLASBT_USER_AGENT", "atlasbt-research/0.1")
# SEC asks for a contact in the UA (https://www.sec.gov/os/accessing-edgar-data). Supply it
# via SEC_USER_AGENT (e.g. "atlasbt-research/0.1 you@example.org"); otherwise the plain UA is
# used and, if SEC refuses it (403), the SEC provider degrades gracefully.
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "").strip() or None
SEC_HOSTS = {"data.sec.gov", "www.sec.gov", "efts.sec.gov"}

# Per-host minimum interval between requests (seconds). SEC allows <=10 req/s.
HOST_MIN_INTERVAL = {
    "data.sec.gov": 0.15,
    "www.sec.gov": 0.15,
    "www.dolthub.com": 0.5,
    "fred.stlouisfed.org": 0.5,
    "cdn.cboe.com": 0.3,
    "datafeed.dukascopy.com": 1.0,
    "www.federalreserve.gov": 1.0,
    "en.wikipedia.org": 1.0,
    "raw.githubusercontent.com": 0.3,
    "www.nasdaqtrader.com": 1.0,
    "mba.tuck.dartmouth.edu": 1.0,
    "iborrowdesk.com": 2.0,
    "default": 0.5,
}

# Keep the cache comfortably below this (bytes). The fetch CLI checks it.
CACHE_BUDGET_BYTES = int(float(os.environ.get("ATLASBT_CACHE_BUDGET_GB", "40")) * 1e9)


def offline() -> bool:
    """ATLASBT_OFFLINE=1: no network at all; atlasbt.data.http serves its disk cache (ignoring
    ttl) and raises FetchError for anything not cached. Backtests / tests set it."""
    return os.environ.get("ATLASBT_OFFLINE", "0").strip().lower() in ("1", "true", "yes", "on")
