import pytest

from atlasbt.data.store import default_store


def _cache_ok() -> bool:
    try:
        s = default_store()
        return s.exists("bars_1d", "yahoo", "SPY") and s.exists("bars_1d", "yahoo", "IEF")
    except Exception:
        return False


CACHE = _cache_ok()
needs_cache = pytest.mark.skipif(not CACHE, reason="real data cache (cache/store) not available")


@pytest.fixture(scope="session")
def portal():
    from atlasbt.run.fastportal import FastPortal

    return FastPortal()


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long real-data validations (PUT index, UMD)")
