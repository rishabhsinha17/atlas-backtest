import pytest
import numpy as np
import pandas as pd

from atlasbt.data.portal import Portal

from .conftest import needs_cache


@needs_cache
@pytest.mark.slow
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
@pytest.mark.slow
def test_fastportal_never_returns_actions_known_after_as_of(portal):
    asof = pd.Timestamp("2015-06-01 20:00", tz="UTC")
    acts = portal.corporate_actions(["SPY", "AAPL"], "2015-01-01", "2016-12-31", asof)
    assert acts and all(pd.Timestamp(x.known_at) <= asof for x in acts)


@needs_cache
@pytest.mark.slow
def test_fastportal_universe_memo_identical_to_portal(portal):
    p = Portal()
    for d in ("2015-03-02 15:00", "2015-03-02 21:30", "2019-01-02 14:31", "2024-06-03 20:00"):
        a = pd.Timestamp(d, tz="UTC")
        assert portal.universe("SP500", a) == p.universe("SP500", a)
        assert portal.universe("SECTOR_SPDR", a) == p.universe("SECTOR_SPDR", a)
    a = pd.Timestamp("2020-01-02 21:00", tz="UTC")
    assert portal.borrow("SPY", a) == p.borrow("SPY", a)


@needs_cache
@pytest.mark.slow
def test_fastportal_borrow_identical_to_portal(portal):
    """borrow() fast path (base-ticker membership) == Portal.borrow on symbols, ETFs, keys, delisted names."""
    p = Portal()
    dates = [pd.Timestamp(d, tz="UTC") for d in ("2009-06-01 20:00", "2015-03-02 21:00", "2019-01-02 15:00",
                                                  "2020-03-16 20:00", "2024-06-03 20:00")]
    for a in dates:
        members = p.universe("SP500", a)
        keys = [m for m in members if "~CIK" in m][:5]
        syms = ["SPY", "QQQ", "AAPL", "XOM", "GE", "TSLA", "LEH", "ENRN", "BBT", "XLK", "ZZZZ"] + members[:15] + keys
        for s in syms:
            assert portal.borrow(s, a) == p.borrow(s, a), (s, a)
