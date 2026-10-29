from types import SimpleNamespace

import pandas as pd

from atlasbt.core.types import RunStatus
from atlasbt.run.activity import classify_activity


def _synthetic_result(n_trades, years=10, status=RunStatus.OK, n_fill_days=None):
    idx = pd.date_range("2010-01-04", periods=int(252 * years), freq="B", tz="UTC")
    eq = pd.Series(1e6, index=idx)
    ts = list(idx[:: max(len(idx) // max(n_trades, 1), 1)][:n_trades])
    fills = pd.DataFrame({"ts": ts, "quantity": [1.0] * len(ts), "liquidity": ["TAKER"] * len(ts)})
    trades = pd.DataFrame({"entry_ts": ts})
    return SimpleNamespace(status=status, equity=eq, returns=eq.pct_change(), fills=fills, trades=trades,
                           diagnostics={}, fidelity={}, not_testable_reason="")


def test_synthetic_zero_activity_is_flagged():
    assert classify_activity(_synthetic_result(0), "1d", "SWING_RISK")["activity"] == "ZERO_ACTIVITY"


def test_synthetic_low_activity_floor_daily():
    a = classify_activity(_synthetic_result(3, years=10), "1d", "SWING_RISK")
    assert a["activity"] == "LOW_ACTIVITY" and a["floor"] == 5
    assert classify_activity(_synthetic_result(6, years=10), "1d", "SWING_RISK")["activity"] == "OK"


def test_synthetic_rebalance_never_rebalances():
    a = classify_activity(_synthetic_result(1, years=5), "1mo", "PORTFOLIO")
    assert a["activity"] == "LOW_ACTIVITY"


def test_synthetic_intraday_floor_one_per_month():
    a = classify_activity(_synthetic_result(1, years=1), "5m", "DAY_RISK")
    assert a["activity"] == "LOW_ACTIVITY" and a["floor"] == 12


def test_not_ok_runs_are_not_labelled():
    assert classify_activity(_synthetic_result(0, status=RunStatus.NOT_TESTABLE), "1d", "")["activity"] == "n/a"
