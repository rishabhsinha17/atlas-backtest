"""tests/run: offline always (ARCHITECTURE §10). Real-cache tests are marked `slow` and skipped unless
ATLASBT_SLOW=1 (or `-m slow`); the default suite runs in well under a minute."""
import os

import pytest

from atlasbt.run.offline import enable

enable(strict=True)          # any network attempt raises (HTTP client: FetchError; requests/sockets: OfflineError)


def _cache_ok() -> bool:
    try:
        from atlasbt.data.store import default_store

        s = default_store()
        return s.exists("bars_1d", "yahoo", "SPY") and s.exists("bars_1d", "yahoo", "IEF")
    except Exception:
        return False


CACHE = _cache_ok()
needs_cache = pytest.mark.skipif(not CACHE, reason="real data cache (cache/store) not available")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: real-cache end-to-end validations (run with ATLASBT_SLOW=1 or -m slow)")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("ATLASBT_SLOW") == "1" or "slow" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="slow real-cache test: set ATLASBT_SLOW=1 or run with -m slow")
    for it in items:
        if "slow" in it.keywords and str(it.fspath).find(os.sep + "run" + os.sep) >= 0:
            it.add_marker(skip)


@pytest.fixture(scope="session")
def portal():
    from atlasbt.run.fastportal import FastPortal

    return FastPortal()
