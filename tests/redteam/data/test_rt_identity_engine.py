"""ENGINE-path identity regressions (coordinator audit #1): runs load whole windows with
as_of = range_end, so identity must be a property of the bar date. Real store; skipped if absent."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine_exec"))
from rt_ee_helpers import make, zero_cost  # noqa: E402

from atlasbt.data import config, frames
from atlasbt.data.portal import Portal
from atlasbt.data.store import ParquetStore
from atlasbt.engine import Backtester


@pytest.fixture(scope="module")
def portal():
    if not (config.STORE_ROOT / "bars_1d" / "yahoo").is_dir():
        pytest.skip("real cache absent")
    return Portal(ParquetStore(config.STORE_ROOT))


def _dolt_close(p, sym, d):
    x = p.store.read("bars_1d", "dolthub", sym)
    x.index = frames.to_naive_date(x.index)
    return float(x.loc[pd.Timestamp(d), "close"])


def _run(portal, sym, buy_day, start, end):
    def script(ctx, st):
        if ctx.session_date() == pd.Timestamp(buy_day) and not st.seen:
            st.seen.append(ctx.order(sym, 100))
    S = make(universe=[sym])
    st = S(script)
    Backtester(st, portal, zero_cost()).run(pd.Timestamp(start), pd.Timestamp(end))
    return st


@pytest.mark.parametrize("ticker,buy_day,sid,dolt_ref", [
    ("BBT", "2019-12-02", "CIK0000092230", "BBT"),     # BB&T; 'BBT' later reused (2025+)
    ("IR", "2019-06-03", "CIK0001466258", "IR"),       # Ingersoll-Rand plc; 'IR' = Gardner Denver from 2020
])
def test_engine_long_run_fills_pit_security(portal, ticker, buy_day, sid, dolt_ref):
    """A run from 2019 to 2026 must trade the security the ticker denoted on the buy day: the
    universe hands out the security key, and the fill is at that security's as-traded price."""
    day = pd.Timestamp(buy_day + " 21:00", tz="UTC")
    keys = [k for k in portal.universe("SP500", day) if k.split("~")[0] == ticker]
    assert keys == [f"{ticker}~{sid}"], keys
    st = _run(portal, keys[0], buy_day, pd.Timestamp(buy_day) - pd.Timedelta(days=7), "2026-10-23")
    assert st.fills, "no fill"
    nxt = frames.schedule().index[frames.schedule().index > pd.Timestamp(buy_day)][0]
    ref = _dolt_close(portal, dolt_ref, buy_day)
    assert st.fills[0].price == pytest.approx(ref, rel=0.06), (st.fills[0].price, ref, nxt)


def test_engine_plain_reused_ticker_never_spliced(portal):
    """A hand-coded plain 'BBT' over 2019-2026 spans two issuers: no bars are served (no fill at
    the other company's price)."""
    st = _run(portal, "BBT", "2019-12-02", "2019-11-25", "2026-10-23")
    assert not st.fills or st.fills[0].price == pytest.approx(_dolt_close(portal, "BBT", "2019-12-02"), rel=0.06)
