import numpy as np
import pandas as pd

from atlasbt.data.portal import Portal

from .conftest import needs_cache


@needs_cache
def test_fastportal_corporate_actions_identical_to_portal(portal):
    p = Portal()
    syms = ["SPY", "QQQ", "IEF", "XLE", "AAPL", "GE"]
    rng = np.random.default_rng(7)
    for _ in range(40):
        a = pd.Timestamp("1996-01-01") + pd.Timedelta(days=int(rng.integers(0, 10500)))
        b = a + pd.Timedelta(days=int(rng.integers(0, 500)))
        asof = pd.Timestamp(b, tz="America/New_York") + pd.Timedelta(hours=int(rng.integers(-72, 72)))
        assert p.corporate_actions(syms, a, b, asof) == portal.corporate_actions(syms, a, b, asof)


@needs_cache
def test_fastportal_never_returns_actions_known_after_as_of(portal):
    asof = pd.Timestamp("2015-06-01 20:00", tz="UTC")
    acts = portal.corporate_actions(["SPY", "AAPL"], "2015-01-01", "2016-12-31", asof)
    assert acts and all(pd.Timestamp(x.known_at) <= asof for x in acts)
