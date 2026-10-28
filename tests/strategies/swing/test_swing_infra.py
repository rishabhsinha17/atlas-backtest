"""Shared swing machinery: point-in-time bar cache (vs ctx.history adjusted),
split handling of resting protective orders, gross-cap ordering, S055 running ATH.

SYNTHETIC data, mechanics only.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from swing_fixtures import (CAL, cent_rows, close_ts, flat_rows, orders, portal_for, rows_from_closes, run,  # noqa: E402
                            sessions, to_df, trend_closes, with_universe, zero_cost)

from atlasbt.core.types import ActionType, CorporateAction  # noqa: E402
from atlasbt.engine.backtester import run_backtest  # noqa: E402
from atlasbt.engine.strategy import Requirement, Strategy  # noqa: E402
from atlasbt.engine.testing import InMemoryPortal, synthetic_daily_bars  # noqa: E402
from atlasbt.indicators import prior_all_time_high  # noqa: E402
from atlasbt.strategies import swing as sw  # noqa: E402
from atlasbt.strategies.swing_common import BarCache  # noqa: E402


def _split_portal():
    bars = {(s, "1d"): synthetic_daily_bars("2014-01-01", "2016-12-30", seed=i) for i, s in enumerate(["A", "B"])}
    df = bars[("A", "1d")]
    m = df.index >= pd.Timestamp("2016-03-01 14:30", tz="UTC")
    df.loc[m, ["open", "high", "low", "close"]] /= 2.0
    acts = [CorporateAction("A", ActionType.SPLIT, pd.Timestamp("2016-03-01"), pd.Timestamp("2016-02-01", tz="UTC"),
                            ratio=2.0),
            CorporateAction("B", ActionType.CASH_DIVIDEND, pd.Timestamp("2016-05-02"),
                            pd.Timestamp("2016-04-01", tz="UTC"), amount=1.0, pay_date=pd.Timestamp("2016-05-20"))]
    return InMemoryPortal(bars=bars, actions=acts)


class _CacheProbe(Strategy):
    id = "CACHEPROBE"; bar_freq = "1d"; universe = ["A", "B"]; requires = [Requirement("bars", "1d")]

    def init(self, ctx):
        self.cache = BarCache(keep=1000, fetch=300)
        self.checked = 0
        self.bad = []

    def on_bar(self, ctx):
        self.cache.refresh_actions(ctx)
        for s in self.universe:
            b = self.cache.get(ctx, s, 300)
            ref = ctx.history(s, 300, "1d", adjusted=True)
            self.checked += 1
            ok = len(b) == len(ref) and all(np.allclose(getattr(b, k), ref[c].to_numpy(), rtol=1e-12)
                                            for k, c in (("o", "open"), ("h", "high"), ("l", "low"),
                                                         ("c", "close"), ("v", "volume")))
            ok = ok and (b.end[-1] == int(ctx.now.value))
            if not ok:
                self.bad.append((ctx.session_date(), s))


def test_bar_cache_equals_pit_adjusted_history_across_split_and_dividend():
    st = _CacheProbe()
    run_backtest(st, _split_portal(), "2015-11-02", "2016-07-29", zero_cost(debug=False), raise_errors=True)
    assert st.checked > 300 and st.bad == []


def test_resting_stop_is_split_adjusted_and_reissued():
    """A long S046 position held through a 2:1 split keeps a protective stop at stop/2 for 2x shares."""
    c = trend_closes(260, 40.0, 0.002)
    rows = rows_from_closes(c)
    p = float(c[-1])
    rows += flat_rows(25, p)
    s = len(rows)
    rows += [(p, p * 1.03, p * 0.995, p * 1.025, 1e6)] + flat_rows(6, p * 1.025, 0.002)
    rows = cent_rows(rows)
    ss = sessions(len(rows))
    split_day = ss[s + 4]
    df = to_df(rows)
    m = df.index >= CAL.open_close(split_day)[0]
    df.loc[m, ["open", "high", "low", "close"]] /= 2.0
    df.loc[m, "volume"] *= 2.0
    act = CorporateAction("XYZ", ActionType.SPLIT, split_day, close_ts(ss[s]), ratio=2.0)
    portal = portal_for({"XYZ": df}, actions=[act])
    T = with_universe(sw.S046, ["XYZ"])
    res = run(T, portal, ss[s - 1], ss[s + 6], {"stop_atr": 3.0})
    o = orders(res)
    from atlasbt.indicators import atr
    a = np.asarray(atr(*(np.array([r[k] for r in rows[: s + 1]]) for k in (1, 2, 3)), 20))[-1]
    fill = res.fills[res.fills["tag"] == "entry"]
    q0, f0 = float(fill["quantity"].sum()), float(fill["price"].iloc[0])
    stops = o[o["tag"] == "stop"].sort_values("order_id")
    after = stops[stops["submitted_at"] >= CAL.open_close(split_day)[0]].iloc[0]
    assert after["submitted_at"] == CAL.open_close(split_day)[0]           # re-issued before the open
    assert after["quantity"] == pytest.approx(-2 * q0)
    assert after["stop_price"] == pytest.approx((f0 - 3.0 * a) / 2.0, abs=0.011)
    assert len(res.fills[res.fills["tag"] == "stop"]) == 0                 # no spurious stop-out on the split


def test_gross_cap_counts_pending_entries():
    """Many simultaneous SWING_NOTIONAL entries never exceed 100% gross (10% each, pending orders counted)."""
    syms = [f"E{i:02d}" for i in range(14)]
    dfs = {}
    for k, sname in enumerate(syms):
        cc = list(trend_closes(260, 40.0 + k, 0.002, amp=0.002))
        for r in [-0.015] * 3 + [0.0] * 4:
            cc.append(cc[-1] * (1 + r))
        dfs[sname] = to_df(cent_rows(rows_from_closes(cc)))
    ss = sessions(267)
    T = with_universe(sw.S068, syms)
    res = run(T, portal_for(dfs), ss[258], ss[264])
    e = orders(res)
    e = e[e["tag"] == "entry"]
    first_day = e["submitted_at"].min()
    day = e[e["submitted_at"] == first_day]
    assert 10 <= len(day) <= 11                                             # 10 x ~10% + remainder of the cap
    assert list(day["symbol"]) == sorted(day["symbol"])                     # alphabetical priority
    d = first_day.tz_convert("America/New_York").tz_localize(None).normalize()
    i = ss.index(d)
    notional = [abs(r["quantity"]) * dfs[r["symbol"]]["close"].iloc[i] for _, r in day.iterrows()]
    assert sum(notional) <= 1_000_000.0 + 1e-6
    assert all(x <= 100_000.0 + 1e-6 for x in notional)
    assert len(day) == 10 or notional[-1] < 0.5 * notional[0]


def test_s055_running_ath_equals_indicator():
    rng = np.random.default_rng(0)
    h = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, 600)))
    s = sw.S055()
    s._ath = {}

    class _C:
        def version(self, sym):
            return 1
    s._cache = _C()

    class B:
        pass
    ref = np.asarray(prior_all_time_high(h, 1))
    for t in range(5, 600, 7):
        b = B()
        b.h = h[: t + 1]
        b.start = np.arange(t + 1, dtype=np.int64)
        got = s._prior_ath("X", b)
        assert got == pytest.approx(ref[t])
