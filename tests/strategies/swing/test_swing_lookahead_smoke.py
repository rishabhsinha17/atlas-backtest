"""Causality (truncated vs full data -> identical orders and fills up to t) and
end-to-end smoke runs through ``run_backtest`` for every swing strategy S046-S084.

SYNTHETIC random-walk data (mechanics only).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from swing_fixtures import INDEX_INSTRUMENTS, SectorPortal, vix_df, with_universe  # noqa: E402

from atlasbt.core.types import RunStatus  # noqa: E402
from atlasbt.engine import BacktestConfig  # noqa: E402
from atlasbt.engine.backtester import Backtester, run_backtest  # noqa: E402
from atlasbt.engine.strategy import Requirement  # noqa: E402
from atlasbt.engine.testing import synthetic_daily_bars  # noqa: E402
from atlasbt.strategies import registry  # noqa: E402
from atlasbt.strategies import swing as sw  # noqa: E402

DATA_START, DATA_END = "2016-01-04", "2018-06-29"
RUN_START, RUN_END = "2017-08-01", "2018-03-29"
CUT = pd.Timestamp("2018-01-12")
ETFS = ["AAA", "BBB", "CCC", "DDD"]
STOCKS = [f"T{i:02d}" for i in range(14)]
SECT = {s: ["Information Technology", "Financials", "Health Care"][i % 3] for i, s in enumerate(STOCKS)}
XS = {"S054", "S074", "S082"}
STK_SINGLE = {"S055", "S061", "S084"}


def _bars():
    out = {}
    specs = [(0.0009, 0.016), (0.0004, 0.022), (0.0, 0.012), (-0.0004, 0.02)]
    for i, s in enumerate(ETFS):
        d, v = specs[i]
        out[s] = synthetic_daily_bars(DATA_START, DATA_END, start_price=60 + 10 * i, daily_vol=v, seed=11 + i,
                                      drift=d, volume=4_000_000.0)
    for i, s in enumerate(STOCKS):
        out[s] = synthetic_daily_bars(DATA_START, DATA_END, start_price=30 + 3 * i, daily_vol=0.015 + 0.002 * (i % 5),
                                      seed=50 + i, drift=0.0006 * ((i % 4) - 1), volume=2_000_000.0)
    for i, s in enumerate(["SPY", "XLK", "XLF", "XLV"]):
        out[s] = synthetic_daily_bars(DATA_START, DATA_END, start_price=200 - 40 * i, daily_vol=0.01, seed=90 + i,
                                      drift=0.0003, volume=20_000_000.0)
    out["VIX"] = vix_df(DATA_START, DATA_END)
    return out


BARS = _bars()


def _portal(cut=None):
    bars = {}
    for s, df in BARS.items():
        d = df
        if cut is not None:
            d = df[df.index <= pd.Timestamp(cut).tz_localize("UTC") + pd.Timedelta(hours=23)]
        bars[(s, "1d")] = d
    return SectorPortal(bars=bars, sectors=SECT, universes={"SP500": list(STOCKS)}, instruments=INDEX_INSTRUMENTS)


def _cls(sid):
    cls = registry.get(sid)
    if sid in XS:
        req = [Requirement("bars", "1d", symbols=("SPY",))]
        return type(f"{sid}_LA", (cls,), {"requires": req})
    if sid in STK_SINGLE:
        return type(f"{sid}_LA", (cls,), {"requires": [Requirement("bars", "1d", symbols=("SPY",), critical=False)]})
    if sid == "S083":
        return cls
    return with_universe(cls, ETFS)


def _cfg():
    return BacktestConfig()


_SUBMITS: list = []
_orig_submit = Backtester.submit


def _recording_submit(self, order, *a, **kw):
    o = _orig_submit(self, order, *a, **kw)
    _SUBMITS.append({"order_id": o.id, "symbol": getattr(o.instrument, "symbol", None), "type": o.type.value,
                     "quantity": float(o.quantity), "limit_price": o.limit_price, "stop_price": o.stop_price,
                     "tag": o.tag, "submitted_at": o.submitted_at, "status": o.status.value})
    return o


def _run(sid, portal, end):
    """run_backtest recording every order AS SUBMITTED (OCO reduce-on-fill later mutates quantities)."""
    _SUBMITS.clear()
    Backtester.submit = _recording_submit
    try:
        res = run_backtest(_cls(sid)(), portal, RUN_START, end, _cfg(), raise_errors=True)
    finally:
        Backtester.submit = _orig_submit
    res.diagnostics["submitted"] = pd.DataFrame(list(_SUBMITS))
    return res


def _orders_upto(res, t):
    o = res.diagnostics["submitted"]
    if not len(o):
        return o
    return o[o["submitted_at"] <= t].reset_index(drop=True)


def _fills_upto(res, t):
    f = res.fills
    if not len(f):
        return f
    return f[f["ts"] <= t][["ts", "order_id", "symbol", "quantity", "price"]].reset_index(drop=True)


RESULTS = {}


def _full(sid):
    if sid not in RESULTS:
        RESULTS[sid] = _run(sid, _portal(), RUN_END)
    return RESULTS[sid]


@pytest.mark.parametrize("sid", sw.IDS)
def test_smoke_run_backtest(sid):
    res = _full(sid)
    assert res.status == RunStatus.OK, res.not_testable_reason
    assert len(res.equity) > 100 and np.isfinite(res.equity.to_numpy()).all()
    assert res.diagnostics["lookahead_guard_violations"] == 0
    assert res.meta["n_trials"] == int(np.prod([len(v) for v in registry.get(sid).grid.values()]))


@pytest.mark.parametrize("sid", sw.IDS)
def test_lookahead_truncated_vs_full_identical_orders(sid):
    full = _full(sid)
    trunc = _run(sid, _portal(cut=CUT), CUT)
    from atlasbt.core.clock import nyse
    t = nyse().open_close(CUT)[1]
    a, b = _orders_upto(full, t), _orders_upto(trunc, t)
    assert len(a) == len(b)
    if len(a):
        pd.testing.assert_frame_equal(a, b, check_dtype=False)
    pd.testing.assert_frame_equal(_fills_upto(full, t), _fills_upto(trunc, t), check_dtype=False)


def test_smoke_most_strategies_trade_on_random_data():
    traded = [sid for sid in sw.IDS if len(_full(sid).fills) > 0]
    assert len(traded) >= 30, sorted(set(sw.IDS) - set(traded))


def test_registry_metadata_complete():
    import json
    rows = {r["ID"]: r for r in json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs",
                                                            "strategies.json")))}
    for sid in sw.IDS:
        cls = registry.get(sid)
        r = rows[sid]
        assert cls.spec == f"{r['Entry rule']} || {r['Exit / risk rule']}"
        assert cls.sizing == r["Sizing"] and cls.bar_freq == "1d"
        assert cls.name == r["Strategy"]
        assert set(cls.grid) <= set(cls.params), sid
        for k, v in cls.grid.items():
            assert cls.params[k] in v, (sid, k)          # baseline is one of the tested values
        assert cls.deviations and cls.requires
