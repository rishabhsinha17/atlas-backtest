"""Trigger / near-miss / lookahead / smoke tests for S126-S157 on SYNTHETIC
fixtures (mechanics only). Expected sessions, strikes and exit levels are
computed independently in ``oi_scenarios``; here we check that the strategy
submits its COMBO at the signal close, that the engine fills it at the next
snapshot, that the legs are the expected ones, and that the exit fires at the
first close where the sheet's level is reached."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.types import RunStatus
from atlasbt.strategies import registry

import oi_fixtures as fx
import oi_scenarios as sc_

SIDS = [f"S{i}" for i in range(126, 158)]
EPS = 1e-9          # float ties at a level count as reached (strategy compares the same mids)


@lru_cache(maxsize=None)
def trigger_run(sid):
    sc = sc_.get(sid)
    strat = registry.get(sid)(**sc.params)
    return fx.run(strat, sc.portal(), sc.ss, start_i=sc.start_i)


def opens(res):
    co = fx.combo_orders(res)
    return co[co["tag"].str.startswith("open")].reset_index(drop=True)


def exits(res):
    co = fx.combo_orders(res)
    return co[co["tag"].str.startswith("exit")].reset_index(drop=True)


def fill_ts(res, oid):
    f = res.fills[res.fills["order_id"] == oid]
    return sorted(set(pd.to_datetime(f["ts"], utc=True)))


def package_value(portal, ss, j, legs_exp):
    """Sum ratio x mid at session j's snapshot; legs_exp = [(symbol, ratio)]."""
    snap = portal.option_chain(fx.UND, fx.close_ts(ss[j])).set_index("symbol")
    return float(sum(ratio * snap.loc[sym, "mid"] for sym, ratio in legs_exp))


def entry_info(res, oid):
    f = fx.option_fills(res)
    f = f[f["order_id"] == oid]
    qs = f["quantity"].abs().astype(int).tolist()
    units = int(np.gcd.reduce(qs))
    legs = [(s, int(q // units)) for s, q in zip(f["symbol"], f["quantity"])]
    net = float((f["quantity"] * f["price"]).sum()) / units
    return legs, units, net


# ------------------------------------------------------------------ exit-level rules (sheet levels)
def expected_exit(sid, sc, res, oid):
    """(session index, tag) of the first close where the sheet's exit rule fires, recomputed from the fixture
    mids and the ACTUAL entry fills; None if the scenario's exit is given explicitly."""
    kind = sc.extra.get("check")
    if kind is None:
        return None
    portal, ss = sc.portal(), sc.ss
    legs, units, D = entry_info(res, oid)
    E = sc.expiry
    strikes = sorted({fx.parse_occ(s)[3] for s, _ in legs})
    p = registry.get(sid)(**sc.params).params
    for j in range(sc.signal_i + 1, sc.n):
        dte = (E - ss[j]).days
        v = package_value(portal, ss, j, legs)
        pnl, C, cc = v - D + EPS, -D, -v - EPS
        if kind == "debit_vertical":
            w = abs(strikes[1] - strikes[0])
            if dte <= 7:
                return j, "exit_dte_le_7"
            if pnl >= 0.5 * (w - D):
                return j, "exit_tp_50pct_max_gain"
            if pnl <= -0.5 * D:
                return j, "exit_sl_50pct_debit"
        elif kind in ("credit_tp_stop", "credit_tp50", "credit_tp_lossstop"):
            if dte <= 7:
                return j, "exit_dte_le_7"
            if sid == "S130":
                if cc <= 0.5 * C:
                    return j, "exit_tp_50pct"
                continue
            if cc <= 0.5 * C:
                return j, "exit_tp_50pct_credit"
            if kind == "credit_tp_stop" and cc >= 2 * C:
                return j, "exit_stop_2x_credit"
            if kind == "credit_tp_lossstop" and cc - C >= 2 * C:
                return j, "exit_stop_loss_2x_credit"
        elif kind == "credit_tp_dte5":
            if dte <= 5:
                return j, "exit_dte_le_5"
            if cc <= 0.5 * C:
                return j, "exit_tp_50pct_credit"
        elif kind == "debit_tp30":
            w = strikes[-1] - strikes[1] if len(strikes) == 3 else strikes[1] - strikes[0]
            if dte <= 7:
                return j, "exit_dte_le_7"
            if pnl >= 0.30 * (w - D):
                return j, "exit_tp_30pct_max"
        elif kind == "fly_value":
            if dte <= 5:
                return j, "exit_dte_le_5"
            if v >= 0.5 * p["width"]:
                return j, "exit_value_ge_50pct_max_payoff"
        elif kind == "calendar":
            if dte <= 7:
                return j, "exit_dte_le_7"
            if pnl >= 0.30 * D:
                return j, "exit_tp_30pct_debit"
        elif kind == "wheel":
            return None
    return None


# ------------------------------------------------------------------ trigger
@pytest.mark.parametrize("sid", SIDS)
def test_synthetic_trigger(sid):
    sc = sc_.get(sid)
    res = trigger_run(sid)
    assert res.status == RunStatus.OK
    ss = sc.ss
    op = opens(res)
    assert len(op), f"{sid}: no opening combo"
    o0 = op.iloc[0]
    assert pd.Timestamp(o0["submitted_at"]) == fx.close_ts(ss[sc.signal_i]), \
        (sid, o0["submitted_at"], ss[sc.signal_i])
    # filled at the FIRST snapshot after submission = next session close
    first_fill = fill_ts(res, o0["order_id"])
    if sid == "S148" and not first_fill:
        pytest.fail("S148 limit calendar never filled in the scenario")
    assert first_fill and first_fill[0] == fx.close_ts(ss[sc.signal_i + 1])
    # structure: rights / strikes / ratios / expiry
    legs, units, _ = entry_info(res, o0["order_id"])
    got = sorted((fx.parse_occ(s)[2], fx.parse_occ(s)[3], r) for s, r in legs)
    exp_legs = sorted((r, k, q) for r, k, q in sc.legs)
    if sid == "S157":   # units 1, ratios = contract counts
        assert [(r, k, int(np.sign(q))) for r, k, q in got] == [(r, k, int(np.sign(q))) for r, k, q in exp_legs]
    else:
        assert got == exp_legs, (sid, got, exp_legs)
    exps = {fx.parse_occ(s)[1] for s, _ in legs}
    if sid in ("S148", "S149"):
        assert exps == {sc.expiry, sc.extra["far"]}
    elif sid == "S150":
        assert exps == {sc.expiry, sc.extra["far"]}
    else:
        assert exps == {sc.expiry}
    if sc.units is not None:
        assert units == sc.units, (sid, units, sc.units)
    assert units >= 1
    # exits
    ex = exits(res)
    rule = expected_exit(sid, sc, res, o0["order_id"])
    want = rule if rule is not None else ((sc.exit_i, sc.exit_tag) if sc.exit_i is not None else None)
    if want is not None:
        j, tag = want
        assert len(ex), f"{sid}: no exit (expected {tag} at {ss[j].date()})"
        e0 = ex.iloc[0]
        assert pd.Timestamp(e0["submitted_at"]) == fx.close_ts(ss[j]), (sid, e0["submitted_at"], ss[j], e0["tag"])
        assert e0["tag"].startswith(tag), (sid, e0["tag"], tag)
        if j + 1 < sc.n:
            assert fill_ts(res, e0["order_id"]) == [fx.close_ts(ss[j + 1])]
            # every leg reversed in the same combo
            f = fx.option_fills(res)
            fo = f[f["order_id"] == o0["order_id"]].groupby("symbol")["quantity"].sum()
            fc = f[f["order_id"] == e0["order_id"]].groupby("symbol")["quantity"].sum()
            assert (fo + fc.reindex(fo.index).fillna(0)).abs().max() < 1e-9


# ------------------------------------------------------------------ strategy-specific mechanics
def test_s146_entry_hedge_in_combo_and_rehedges():
    sc = sc_.get("S146")
    res = trigger_run("S146")
    o0 = opens(res).iloc[0]
    f = res.fills[res.fills["order_id"] == o0["order_id"]]
    stock = f[f["asset_class"] != "OPT"]
    assert len(stock) == 1 and pd.Timestamp(stock["ts"].iloc[0]) == fx.close_ts(sc.ss[sc.signal_i + 1])
    # stock leg priced at the first open after submission
    b = sc.portal().bars(fx.UND, "1d", fx.open_ts(sc.ss[sc.signal_i + 1]), fx.close_ts(sc.ss[sc.signal_i + 1]),
                         fx.close_ts(sc.ss[sc.signal_i + 1]))
    assert stock["price"].iloc[0] == pytest.approx(b["open"].iloc[0], abs=0.011)
    orders = res.diagnostics["orders"]
    assert (orders["tag"] == "unhedge").sum() >= 1
    assert (orders["tag"] == "rehedge").sum() >= 1
    ex = exits(res).iloc[0]
    un = orders[orders["tag"] == "unhedge"].iloc[0]
    assert un["submitted_at"] == ex["submitted_at"]


def test_s148_limit_is_decision_natural():
    sc = sc_.get("S148")
    res = trigger_run("S148")
    o0 = opens(res).iloc[0]
    assert o0["limit_price"] == pytest.approx(sc.extra["limit"])


def test_s131_wheel_switches_to_calls_after_assignment():
    sc = sc_.get("S131")
    res = trigger_run("S131")
    ev = res.diagnostics["option_events"]
    assert any(e["kind"] == "ASSIGNMENT" for e in ev)
    op = opens(res)
    assert op.iloc[0]["tag"] == "open_cash_secured_put"
    nxt = fx.CAL.next_session(sc.expiry)
    i = sc.ss.index(nxt)
    assert op.iloc[1]["tag"] == "open_covered_call"
    assert pd.Timestamp(op.iloc[1]["submitted_at"]) == fx.close_ts(sc.ss[i])
    ex = exits(res)                                 # the losing put is NOT rolled at 7 DTE (held to assignment)
    assert not (pd.to_datetime(ex["submitted_at"], utc=True) <= fx.close_ts(sc.expiry)).any()


def test_s129_rolls_on_the_settlement_of_the_previous_put():
    sc = sc_.get("S129")
    res = trigger_run("S129")
    op = opens(res)
    assert len(op) >= 2
    assert pd.Timestamp(op.iloc[1]["submitted_at"]) == fx.close_ts(sc.ss[sc.extra["second_entry"]])


def test_s132_rebuys_put_after_roll():
    sc = sc_.get("S132")
    res = trigger_run("S132")
    ex = exits(res).iloc[0]
    j = sc.ss.index(pd.Timestamp(ex["submitted_at"]).tz_convert("America/New_York").tz_localize(None).normalize())
    op = opens(res)
    assert pd.Timestamp(op.iloc[1]["submitted_at"]) == fx.close_ts(sc.ss[j + 1])


@pytest.mark.parametrize("sid", ["S128", "S133", "S157", "S132"])
def test_shares_bought_before_options(sid):
    sc = sc_.get(sid)
    res = trigger_run(sid)
    o = res.diagnostics["orders"]
    buy = o[o["tag"] == "buy_shares"].iloc[0]
    assert pd.Timestamp(buy["submitted_at"]) == fx.close_ts(sc.ss[0])
    f = res.fills[res.fills["order_id"] == buy["order_id"]]
    assert pd.Timestamp(f["ts"].iloc[0]) >= fx.open_ts(sc.ss[1])
    assert float(f["quantity"].sum()) % 100 == 0


def test_s150_rolls_short_call():
    sc = sc_.get("S150")
    res = trigger_run("S150")
    co = fx.combo_orders(res)
    roll = co[co["tag"] == "roll_short"]
    j = sc_.first_dte_le(sc.ss, sc.signal_i + 2, sc.expiry, 7)
    assert len(roll) and pd.Timestamp(roll.iloc[0]["submitted_at"]) == fx.close_ts(sc.ss[j])
    f = fx.option_fills(res)
    rf = f[f["order_id"] == roll.iloc[0]["order_id"]]
    old = [s for s, q in zip(rf["symbol"], rf["quantity"]) if q > 0]
    new = [s for s, q in zip(rf["symbol"], rf["quantity"]) if q < 0]
    assert len(old) == 1 and fx.parse_occ(old[0])[1] == sc.expiry and fx.parse_occ(old[0])[3] == sc.legs[0][1]
    assert len(new) == 1 and fx.parse_occ(new[0])[1] > sc.expiry
    assert pd.Timestamp(rf["ts"].iloc[0]) == fx.close_ts(sc.ss[j + 1])


def test_s143_chain_iv_percentile_source():
    """Chain-mode IV percentile: real snapshots backfilled through the PIT portal; vol drops -> low percentile."""
    n = 60
    ss = fx.sessions(n)
    closes = sc_.noisy_path(n, 0.006, seed=21)
    vol = np.r_[np.full(45, 0.30), np.full(n - 45, 0.15)]
    p = fx.make_portal(closes, ss, chain_from=0, vol=vol, band=0.12, max_dte=45)
    s = registry.get("S143")(iv_source="chain", iv_lookback=30)
    res = fx.run(s, p, ss, start_i=40)
    op = opens(res)
    assert len(op) and pd.Timestamp(op.iloc[0]["submitted_at"]) == fx.close_ts(ss[45])


# ------------------------------------------------------------------ near miss
@pytest.mark.parametrize("sid", SIDS)
def test_synthetic_near_miss(sid):
    sc = sc_.get(sid)
    params = dict(sc.params)
    params.update(sc.near_params or {})
    portal = sc.near_portal() if (sc.near_closes is not None or sc.near_portal_kw) else sc.portal()
    start_i, end_i = sc.start_i, None
    if sid == "S132":
        end_i = sc.extra["near_end"]
    if sid == "S129":
        start_i = sc.extra["near_start"]
        end_i = sc.ss.index(sc.expiry) - 1
    res = fx.run(registry.get(sid)(**params), portal, sc.ss, start_i=start_i, end_i=end_i)
    assert res.status == RunStatus.OK
    assert len(opens(res)) == 0, (sid, opens(res)[["tag", "submitted_at"]])


# ------------------------------------------------------------------ lookahead
@pytest.mark.parametrize("sid", SIDS)
def test_lookahead_truncation(sid):
    sc = sc_.get(sid)
    cut = min(sc.signal_i + 4, sc.n)
    full = trigger_run(sid)
    trunc = fx.run(registry.get(sid)(**sc.params), sc.portal(cut), sc.ss[:cut], start_i=sc.start_i)
    t = fx.close_ts(sc.ss[cut - 1])
    a, b = fx.orders_until(full, t), fx.orders_until(trunc, t)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    pd.testing.assert_frame_equal(fx.fills_until(full, t), fx.fills_until(trunc, t), check_dtype=False)
    assert len(a) >= 1


# ------------------------------------------------------------------ smoke (run_backtest end to end)
@lru_cache(maxsize=1)
def smoke_portal():
    n = 300
    ss = fx.sessions(n)
    rng = np.random.default_rng(3)
    c = np.round(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n))), 2)
    vix = 20 + 5 * np.sin(np.arange(n) / 15)
    return fx.make_portal(c, ss, chain_from=272, extra_bars={"VIX": vix, "VIX3M": vix + 1}), ss


@pytest.mark.parametrize("sid", SIDS)
def test_smoke_run_backtest(sid):
    p, ss = smoke_portal()
    cls = registry.get(sid)
    kw = {"underlying": "SPY"} if cls.params.get("underlying") == "SPX" else {}
    res = fx.run(cls(**kw), p, ss, start_i=0)
    assert res.status == RunStatus.OK, res.diagnostics.get("traceback")
    assert res.diagnostics.get("lookahead_guard_violations", 0) == 0
    assert np.isfinite(res.equity.iloc[-1])
