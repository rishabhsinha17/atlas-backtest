"""tests/perf: offline always. The golden-result runs (real cache, minutes each) are marked `slow` and
skipped unless ATLASBT_SLOW=1 (or `-m slow`); the equivalence / lookahead tests run in the default suite."""
import os

import pytest

from atlasbt.run.offline import enable

enable(strict=True)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: real-cache end-to-end validations (run with ATLASBT_SLOW=1 or -m slow)")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("ATLASBT_SLOW") == "1" or "slow" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="slow golden-result run: set ATLASBT_SLOW=1 or run with -m slow")
    for it in items:
        if "slow" in it.keywords and (os.sep + "perf" + os.sep) in str(it.fspath):
            it.add_marker(skip)
