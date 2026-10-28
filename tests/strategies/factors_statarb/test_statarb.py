"""S106-S115, S123 mechanics on SYNTHETIC data: exact signal/fill bars, legs and sizes, exits,
near misses, lookahead (truncated vs full data) and smoke runs through run_backtest."""
import math

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.types import OrderType, RunStatus
from atlasbt.indicators import engle_granger
from atlasbt.strategies import statarb as SA

from fsa_fixtures import (CAL, NAV0, build_portal, close_ts, closes_from_returns, fills_at, next_session, open_ts,
                          order_key, orders, run, sessions)
import scen_statarb as SC

BUILDERS = {"S106": SC.s106, "S107": SC.s107, "S108": SC.s108, "S109": SC.s109, "S111": SC.s111, "S112": SC.s112,
            "S113": SC.s113, "S114": SC.s114, "S115": SC.s115, "S123": SC.s123}
IDS = list(BUILDERS)
_cache: dict = {}


def scen(sid):
    if sid not in _cache:
        _cache[sid] = BUILDERS[sid]()
    return _cache[sid]


def _run(sc, start=None, end=None, cut=None, **params):
    p = dict(sc["params"])
    p.update(params)
    strat = sc["cls"](**p)
    res = run(strat, sc["build"](cut=cut), start or sc["start"], end or sc["end"])
    return res, strat


def _px(sc, sym, session):
    return float(sc["closes"][sym].loc[pd.Timestamp(session)])


def _entries(res, sig, ignore=()):
    o = orders(res)
    o = o[(o["submitted_at"] == close_ts(sig)) & (o["type"] == OrderType.MARKET.value) & ~o["symbol"].isin(list(ignore))]
    return o


@pytest.mark.parametrize("sid", IDS)
def test_synthetic_trigger_signal_fill_and_size(sid):
    sc = scen(sid)
    res, strat = _run(sc)
    assert res.status == RunStatus.OK, res.not_testable_reason
    ign = sc.get("ignore", set())
    sig = sc["signal"]
    o = orders(res)
    before = o[(o["submitted_at"] < close_ts(sig)) & ~o["symbol"].isin(list(ign))]
    assert len(before) == 0, "no entries before the signal bar"
    e = _entries(res, sig, ign)
    assert {r.symbol: int(np.sign(r.quantity)) for r in e.itertuples()} == sc["legs"]
    w = sc.get("weights") or {s: 0.05 * sg for s, sg in sc["legs"].items()}
    for r in e.itertuples():
        exp = math.trunc(w[r.symbol] * NAV0 / _px(sc, r.symbol, sig) + (1e-9 if w[r.symbol] > 0 else -1e-9))
        assert r.quantity == exp
    nxt = next_session(sig)
    f = fills_at(res, open_ts(nxt))
    for r in e.itertuples():
        fr = f[(f["symbol"] == r.symbol) & (np.sign(f["quantity"]) == np.sign(r.quantity))]
        assert len(fr) == 1 and fr["quantity"].iloc[0] == r.quantity
        assert fr["price"].iloc[0] == pytest.approx(_px(sc, r.symbol, sig), abs=0.0101)   # next open = prior close here
    assert res.diagnostics["lookahead_guard_violations"] == 0


NEAR = {
    "S106": lambda: SC.s106(z_path=(1.9, 1.9, 1.9, 0.2)),
    "S107": lambda: SC.s107(z_target=(1.9, 1.0, -0.3)),
    "S108": lambda: SC.s108(z_target=1.9),
    "S109": lambda: SC.s109(jump_day="2022-02-01"),     # outside both the residual and the regression window
    "S111": lambda: SC.s111(dev_sd=1.5),
    "S112": lambda: scen("S112"),
    "S113": lambda: scen("S113"),
    "S114": lambda: SC.s114(shock_sd=1.5),
    "S115": lambda: SC.s115(kind="near"),
    "S123": lambda: SC.s123(excess=0.015),
}


@pytest.mark.parametrize("sid", IDS)
def test_synthetic_near_miss(sid):
    sc = NEAR[sid]()
    res, _ = _run(sc)
    o = orders(res)
    if sid in ("S112", "S113"):
        assert not set(sc["near_miss"]) & set(o["symbol"])
        return
    if sid == "S109":
        legs = {r.symbol: int(np.sign(r.quantity)) for r in o.itertuples() if r.symbol != "SPY"}
        assert legs != scen("S109")["legs"]                                     # old jumps are not 5d residuals
        assert not {"S00", "S01"} & set(legs)
        res2, _ = _run(scen("S109"), end=pd.Timestamp("2023-03-16"))          # Thursday: not a rebalance day
        assert len(orders(res2)) == 0
        return
    if sid == "S111":
        assert abs(sc["z"]) < 2.0
    if sid == "S115":
        assert sc["slope_k"] < 0 and sc["signal"] is None
    if sid == "S123":
        assert 0 < sc["excess"] < 0.02
    if sid == "S107":      # later AR(1) noise may legitimately trigger; the designed window must stay flat
        o = o[o["submitted_at"] <= close_ts(sc["exit_day"])]
    assert len(o) == 0


@pytest.mark.parametrize("sid", IDS)
def test_lookahead_truncated_vs_full(sid):
    sc = scen(sid)
    cut = close_ts(sc["signal"])
    full, _ = _run(sc)
    trunc, _ = _run(sc, end=sc["signal"], cut=cut)
    a, b = order_key(full, cut), order_key(trunc, cut)
    assert a and a == b


@pytest.mark.parametrize("sid", IDS)
def test_smoke_run_backtest(sid):
    sc = scen(sid)
    end = pd.Timestamp("2023-07-14") if sid == "S113" else pd.Timestamp("2023-06-30")
    res, _ = _run(sc, start=pd.Timestamp("2023-03-01"), end=end)
    assert res.status == RunStatus.OK
    assert res.diagnostics["lookahead_guard_violations"] == 0
    assert len(res.fills) > 0


# ----------------------------------------------------------------------------- S106 exits
def _exit_orders(res, day, syms):
    o = orders(res)
    return o[(o["submitted_at"] == close_ts(day)) & o["symbol"].isin(syms) & o["tag"].str.startswith("exit")]


def test_s106_exits_converge_stop_time():
    days = sessions("2023-03-15", "2023-03-31")
    sc = scen("S106")
    res, strat = _run(sc)
    ex = _exit_orders(res, days[3], ["A0", "B0"])
    assert set(ex["tag"]) == {"exit:converged"} and len(ex) == 2
    assert dict(zip(ex["symbol"], np.sign(ex["quantity"]))) == {"A0": 1, "B0": -1}
    sc2 = SC.s106(z_path=(3.0, 4.5, 4.5, 0.2))
    res2, _ = _run(sc2)
    assert set(_exit_orders(res2, days[1], ["A0", "B0"])["tag"]) == {"exit:stop_z"}
    sc3 = SC.s106(z_path=(3.0, 3.0, 3.0, 3.0, 3.0), max_hold=3)
    res3, _ = _run(sc3)
    assert set(_exit_orders(res3, days[3], ["A0", "B0"])["tag"]) == {"exit:time"}


def test_s106_formation_selects_same_sector_pairs():
    sc = scen("S106")
    _, strat = _run(sc, end=pd.Timestamp("2023-03-02"))
    assert sorted(strat.pairs) == [("A0", "B0"), ("C0", "D0")]


# ----------------------------------------------------------------------------- S107
def test_s107_cointegration_screen_and_exit():
    sc = scen("S107")
    res, strat = _run(sc)
    px = pd.DataFrame(sc["closes"]).loc[:SC.RUN_START].iloc[-252:]
    assert engle_granger(np.log(px["A1"]), np.log(px["B1"])).pvalue < 0.05
    assert engle_granger(np.log(px["C1"]), np.log(px["D1"])).pvalue >= 0.05
    assert set(strat.active) == {("A1", "B1")}
    assert strat.active[("A1", "B1")]["beta"] == pytest.approx(sc["beta"], rel=1e-9)
    ex = _exit_orders(res, sc["exit_day"], ["A1", "B1"])
    assert set(ex["tag"]) == {"exit:z_cross_0"} and len(ex) == 2


# ----------------------------------------------------------------------------- S108
def test_s108_no_signal_before_spike():
    sc = scen("S108")
    assert max(abs(z) for z in sc["pre_z"]) < 2.0


# ----------------------------------------------------------------------------- S112
def test_s112_time_exit_after_5_sessions():
    sc = scen("S112")
    res, _ = _run(sc)
    days = sessions(SC.K_DAY, "2023-03-31")
    ex = orders(res)
    ex = ex[ex["tag"].str.startswith("exit") & ex["symbol"].isin(list(sc["legs"]))]
    assert set(ex["submitted_at"]) == {close_ts(days[5])}
    assert set(ex["tag"]) == {"exit:time"}


# ----------------------------------------------------------------------------- S114
def test_s114_moc_exit_next_close():
    sc = scen("S114")
    res, _ = _run(sc)
    o = orders(res)
    moc = o[(o["type"] == OrderType.MOC.value) & (o["submitted_at"] == close_ts(SC.K_DAY))]
    e = _entries(res, SC.K_DAY)
    assert dict(zip(moc["symbol"], moc["quantity"])) == {r.symbol: -r.quantity for r in e.itertuples()}
    nxt = next_session(SC.K_DAY)
    f = fills_at(res, close_ts(nxt))
    for r in moc.itertuples():
        fr = f[f["symbol"] == r.symbol]
        assert len(fr) == 1 and fr["price"].iloc[0] == pytest.approx(_px(sc, r.symbol, nxt), abs=0.0101)
    assert all(res.positions[res.positions["ts"] == close_ts(nxt)]["quantity"] == 0)


def test_s114_hold3_exit_and_htb_screen():
    sc = SC.s114(hold_days=3)
    res, _ = _run(sc)
    days = sessions(SC.K_DAY, "2023-03-31")
    o = orders(res)
    moc = o[o["type"] == OrderType.MOC.value]
    assert set(moc["submitted_at"]) == {close_ts(days[2])}
    assert set(fills_at(res, close_ts(days[3]))["symbol"]) == set(sc["legs"])
    sc2 = SC.s114(htb="T09")
    res2, _ = _run(sc2)
    e = _entries(res2, SC.K_DAY)
    assert "T09" not in set(e["symbol"])
    assert dict(zip(e["symbol"], np.sign(e["quantity"]))) == {"T00": 1, "T08": -1}


# ----------------------------------------------------------------------------- S115 / S123
def test_s115_exit_on_ma_cross():
    sc = scen("S115")
    res, _ = _run(sc)
    ex = orders(res)
    ex = ex[ex["tag"].str.startswith("exit")]
    assert set(ex["tag"]) == {"exit:ma_cross"}
    day = pd.Timestamp(ex["submitted_at"].iloc[0]).tz_convert("America/New_York").normalize().tz_localize(None)
    s = sc["s"]
    i = list(s.index).index(day)
    assert s.iloc[i] < s.iloc[i - 19:i + 1].mean() and s.iloc[i - 1] >= s.iloc[i - 20:i].mean()


def test_s123_exit_when_excess_fades():
    sc = SC.s123(july_exit=True)
    assert sc["excess"] > 0.02 and 0 < sc["excess_july"] < 0.005
    res, strat = _run(sc, end=pd.Timestamp("2023-08-03"))
    o = orders(res)
    jul = o[o["submitted_at"] == close_ts(pd.Timestamp("2023-07-31"))]
    assert dict(zip(jul["symbol"], np.sign(jul["quantity"]))) == {"RSP": -1, "SPY": 1}
    assert strat.state_ == 0


# ----------------------------------------------------------------------------- S110
def test_s110_not_testable_without_inav():
    dates = sessions("2023-01-03", "2023-03-31")
    closes = {s: closes_from_returns(dates, np.zeros(len(dates)), 100.0) for s in ("SPY", "QQQ", "IWM", "DIA")}
    res = run(SA.S110(), build_portal(closes), "2023-02-01", "2023-03-31")
    assert res.status == RunStatus.NOT_TESTABLE
    assert "inav" in res.not_testable_reason
    assert any(r.kind == "inav" and r.critical for r in SA.S110.requires)
