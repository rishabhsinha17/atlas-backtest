"""Mechanics of the intraday family's shared machinery + registry metadata +
smoke runs (SYNTHETIC data, mechanics only)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.clock import et
from atlasbt.core.types import RunStatus
from atlasbt.engine.backtester import run_backtest
from atlasbt.engine.config import BacktestConfig
from atlasbt.engine.testing import InMemoryPortal, daily_from_intraday, synthetic_intraday_bars
from atlasbt.indicators import atr, ema, rsi
from atlasbt.indicators import intraday as iind
from atlasbt.strategies import registry
import atlasbt.strategies.intraday as fam
from atlasbt.strategies.intraday_base import (
    MODULE_DEVIATIONS, Tape, aggregate_frame, atr_stream, ema_stream, rsi_stream, tod_profile,
)

from ._synth import CAL, SYM, Scenario, assert_status_ok, day_frame, history, orders, prior_sessions, run

ROOT = Path(__file__).resolve().parents[3]
SHEET = {r["ID"]: r for r in json.loads((ROOT / "docs" / "strategies.json").read_text())}


# ===================================================================== registry metadata
@pytest.mark.parametrize("sid", fam.IDS)
def test_registered_metadata_matches_sheet(sid):
    cls = registry.get(sid)
    row = SHEET[sid]
    assert cls.id == sid and cls.name == row["Strategy"]
    assert cls.spec == f'{row["Entry rule"]} || {row["Exit / risk rule"]}'
    assert cls.bar_freq == row["Bars"]
    assert cls.sizing == row["Sizing"]
    assert cls.asset == row["Assets"]
    assert set(cls.grid) <= set(cls.params) and cls.grid
    for k, vals in cls.grid.items():
        assert len(vals) >= 2
    for d in MODULE_DEVIATIONS:
        assert d in cls.deviations
    assert len(cls.deviations) > len(MODULE_DEVIATIONS)
    for combo in registry.expand_grid(cls.grid):        # every grid point constructs
        cls(**combo)
    assert cls.universe or cls.universe_name
    st = cls()
    assert any(r.kind == "bars" and r.freq == cls.bar_freq and r.critical for r in st.requires)


def test_universe_choices():
    for sid in fam.IDS:
        cls = registry.get(sid)
        a = SHEET[sid]["Assets"]
        if sid == "S116":
            assert tuple(cls.universe) == ("SPY", "QQQ", "IWM") and cls.universe_name is None
        elif a == "STK/ETF":
            assert tuple(cls.universe) == fam.ETF_CORE and cls.universe_name == fam.MEGACAP_UNIVERSE
        elif a == "STK":
            assert not cls.universe and cls.universe_name == fam.MEGACAP_UNIVERSE
        elif a == "ETF":
            assert tuple(cls.universe) == fam.ETF_CORE and cls.universe_name is None


def test_premarket_strategies_require_pre_bars():
    for sid in ("S005", "S006", "S007"):
        st = registry.get(sid)()
        assert any(r.kind == "bars" and "PRE" in r.sessions and r.critical for r in st.requires)
        assert st.data_sessions() == ("PRE", "RTH")


# ===================================================================== helpers vs indicators
def _frame_1m(days, seed=0):
    df = synthetic_intraday_bars(str(days[0].date()), str(days[-1].date()), "1m", volume_per_bar=1000.0, seed=seed)
    rng = np.random.default_rng(seed)
    df["volume"] = rng.integers(500, 5000, len(df)).astype(float)
    return df


def test_tod_profile_matches_indicator_relative_volume():
    days = prior_sessions(pd.Timestamp("2024-04-17"), 26)
    df = _frame_1m(days, 1)
    today = days[-1]
    key = iind.session_key(df.index)
    hist = df[key < np.datetime64(today)]
    for cumulative in (True, False):
        ref = iind.relative_volume(df, lookback=20, cumulative=cumulative, min_sessions=15)
        prof = tod_profile(hist, 20, cumulative, 15)
        t = df[key == np.datetime64(today)]
        slots = iind.minute_of_day(t.index)
        got = np.array([prof.get(s, np.nan) for s in slots])
        np.testing.assert_allclose(got, ref.loc[t.index, "expected"].to_numpy(), rtol=1e-12)


def test_aggregate_frame_and_tape_agg_complete_buckets_only():
    s = prior_sessions(pd.Timestamp("2024-04-17"), 1)[0]
    df = synthetic_intraday_bars(str(s.date()), str(s.date()), "1m", seed=3)
    agg = aggregate_frame(df, 5)
    assert len(agg) == 78 and agg.index[0] == et(s, "09:30") and agg.index[1] == et(s, "09:35")
    np.testing.assert_allclose(agg["high"].iloc[0], df["high"].iloc[:5].max())
    np.testing.assert_allclose(agg["volume"].iloc[1], df["volume"].iloc[5:10].sum())
    o, c = CAL.open_close(s)
    tp = Tape(SYM, int(o.value), int(c.value), int(o.value), 60_000_000_000)

    class _Ctx:                                     # minimal context: completed bars only
        def __init__(self, now):
            self.now = now

        def history(self, sym, n, adjusted=False, sessions=("RTH",)):
            return df[df["known_at"] <= self.now].iloc[-n:]
    for k in range(1, 13):
        tp.update(_Ctx(o + pd.Timedelta(minutes=k)), ("RTH",))
        b = tp.agg(5)
        assert b.n == k // 5                         # a bucket appears only once its 5th minute completed
    vw = iind.session_vwap(df.iloc[:12])
    np.testing.assert_allclose(tp.vwap[:12], vw["vwap"].to_numpy(), rtol=1e-12)


def test_barstream_matches_batch_indicators():
    days = prior_sessions(pd.Timestamp("2024-04-17"), 4)
    df = synthetic_intraday_bars(str(days[0].date()), str(days[-1].date()), "5m", seed=4)
    today = days[-1]
    o, c = CAL.open_close(today)
    tp = Tape(SYM, int(o.value), int(c.value), int(o.value), 300_000_000_000)

    class _Ctx:
        def __init__(self, now):
            self.now = now

        def history(self, sym, n, adjusted=False, sessions=("RTH",), freq=None):
            return df[df["known_at"] <= self.now].iloc[-n:]
    now = o + pd.Timedelta(minutes=60)
    ctx = _Ctx(now)
    tp.update(ctx, ("RTH",))
    streams = {"rsi": rsi_stream(14), "ema": ema_stream(20), "atr": atr_stream(14)}
    vals = {k: v.sync(ctx, tp, 5) for k, v in streams.items()}
    n_today = tp.n
    for k, s in streams.items():
        hist = df[df["known_at"] <= now].iloc[-(s.warm_bars + n_today):]     # warm_bars prior bars + today
        if k == "rsi":
            ref = rsi(hist["close"].to_numpy(), 14)
        elif k == "ema":
            ref = ema(hist["close"].to_numpy(), 20)
        else:
            ref = atr(hist["high"].to_numpy(), hist["low"].to_numpy(), hist["close"].to_numpy(), 14)
        np.testing.assert_allclose(vals[k], ref[-n_today:], rtol=1e-9)


# ===================================================================== early close
def _early_close_scenario(break_time="10:00"):
    day = pd.Timestamp("2024-11-29")                     # NYSE 13:00 close
    assert CAL.is_early_close(day)
    hist = history(prior_sessions(day, 22), 1)
    o, _ = CAL.open_close(day)
    k = int((et(day, break_time) - o) / pd.Timedelta(minutes=1))
    v = 2e5
    rows = [(100.0, 100.5, 99.8, 100.1, v)] + [(100.1, 100.3, 99.9, 100.2, v)] * 4
    rows += [(100.2, 100.3, 100.1, 100.2, 1e5)] * (k - 5)
    rows += [(100.2, 100.8, 100.15, 100.7, 1e5)]
    return Scenario("S001", {(SYM, "1m"): pd.concat([hist, day_frame(day, rows, 1)])}, day=day)


def test_early_close_session_flattens_5min_before_close():
    sc = _early_close_scenario("10:00")
    r = run(registry.get("S001"), sc)
    assert_status_ok(r)
    e = orders(r, "entry")
    assert len(e) == 1 and e.iloc[0].submitted_at == et(sc.day, "10:01")
    fx = r.fills[r.fills.tag.isin(["flat_early_close", "flat_by"])]      # engine or module, same time
    assert len(fx) == 1 and fx.iloc[0].ts == et(sc.day, "12:55")
    assert r.positions.empty or (r.positions.quantity == 0).all()


def test_early_close_session_no_entries_after_close_minus_45min():
    sc = _early_close_scenario("12:20")                  # breakout after 12:15 (= 13:00 - 45 min)
    r = run(registry.get("S001"), sc)
    assert_status_ok(r)
    assert len(orders(r, "entry")) == 0


# ===================================================================== smoke
def _smoke_portal():
    days = prior_sessions(pd.Timestamp("2024-03-15"), 30) + [pd.Timestamp("2024-03-15")]
    m1 = synthetic_intraday_bars(str(days[0].date()), str(days[-1].date()), "1m", daily_vol=0.03,
                                 volume_per_bar=50_000, seed=11, include_extended=True)
    rth = m1[m1["session"] == "RTH"]
    m5 = aggregate_frame(rth, 5)
    m5["session"] = "RTH"
    m5["known_at"] = m5.index + pd.Timedelta(minutes=5)
    d = daily_from_intraday(m1)
    bars = {}
    for s in ("XYZ", "SPY"):
        bars[(s, "1m")], bars[(s, "5m")], bars[(s, "1d")] = m1, m5, d
    q = pd.DataFrame({"bid": m1["close"] - 0.01, "ask": m1["close"] + 0.01, "bid_size": 100.0, "ask_size": 100.0},
                     index=m1.index + pd.Timedelta(seconds=30))
    return InMemoryPortal(bars=bars, quotes={"XYZ": q}, universes={"SP500_TOP50_DV": ["XYZ"]}, rf=0.02)


@pytest.fixture(scope="module")
def smoke_portal():
    return _smoke_portal()


@pytest.mark.parametrize("sid", fam.IDS)
def test_smoke_run_backtest(sid, smoke_portal):
    cls = registry.get(sid)
    T = type(sid + "Smoke", (cls,), {"universe": ("XYZ",) if cls.universe else (),
                                      "universe_name": None if cls.universe else "SP500_TOP50_DV"})
    r = run_backtest(T(), smoke_portal, "2024-03-11", "2024-03-15", BacktestConfig(use_quotes="never"))
    assert r.status == RunStatus.OK, (r.not_testable_reason, r.diagnostics.get("traceback"))
    assert r.diagnostics["lookahead_guard_violations"] == 0
    assert len(r.equity) == 5
    # intraday: never hold overnight
    assert r.positions.empty or (r.positions.quantity == 0).all()
    if len(r.fills):
        ent = r.fills[r.fills.tag == "entry"]
        tod = ent.ts.dt.tz_convert("America/New_York")
        assert ((tod.dt.hour * 60 + tod.dt.minute) <= 15 * 60 + 15).all() or cls.sizing == "DAY_NOTIONAL"
