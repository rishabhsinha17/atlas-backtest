"""S085-S100 mechanics on SYNTHETIC data: exact signal bar, fill bar, baskets and sizes,
near misses, lookahead (truncated vs full data) and a smoke run through run_backtest."""
import math

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.types import Fidelity, RunStatus
from atlasbt.strategies import factors as F

from fsa_fixtures import NAV0, close_ts, fills_at, next_session, open_ts, order_key, orders, run
import scen_factors as SC

IDS = list(SC.SCENARIOS)
_cache: dict = {}


def scen(sid):
    if sid not in _cache:
        _cache[sid] = SC.SCENARIOS[sid]()
    return _cache[sid]


def _run(sc, start=SC.START, end=SC.END, cut=None, **params):
    p = dict(sc["params"])
    p.update(params)
    return run(sc["cls"](**p), sc["build"](cut=cut), start, end)


def _closes_at(portal, syms, session):
    o, c = open_ts(session), close_ts(session)
    out = {}
    for s in syms:
        b = portal.bars(s, "1d", o - pd.Timedelta(hours=1), c + pd.Timedelta(1, "ns"), as_of=c)
        out[s] = float(b["close"].iloc[-1])
    return out


@pytest.mark.parametrize("sid", IDS)
def test_synthetic_trigger_signal_fill_and_size(sid):
    sc = scen(sid)
    res = _run(sc)
    assert res.status == RunStatus.OK, res.not_testable_reason
    o = orders(res)
    sig = close_ts(SC.SIGNAL)
    assert len(o) and (o["submitted_at"] == sig).all(), "orders only at the month-end signal close"
    ign = sc.get("ignore", set())
    got_l = set(o.loc[o["quantity"] > 0, "symbol"]) - ign
    got_s = set(o.loc[o["quantity"] < 0, "symbol"]) - ign
    assert got_l == set(sc["longs"])
    assert got_s == set(sc["shorts"])
    # sizes: CROSS_SECTION 2% single-name cap binds (2-4 names per side) unless the scenario says otherwise
    portal = sc["build"]()
    px = _closes_at(portal, list(got_l | got_s), SC.SIGNAL)
    w = sc.get("weights") or {**{s: 0.02 for s in got_l}, **{s: -0.02 for s in got_s}}
    for r in o.itertuples():
        if r.symbol in ign:
            continue
        assert r.quantity == math.trunc(w[r.symbol] * NAV0 / px[r.symbol] + (1e-9 if w[r.symbol] > 0 else -1e-9))
    # fills at the NEXT session open (the bar open = previous close in these fixtures)
    nxt = next_session(SC.SIGNAL)
    f = fills_at(res, open_ts(nxt))
    assert set(f["symbol"]) >= (got_l | got_s)
    for r in f.itertuples():
        if r.symbol in ign:
            continue
        assert r.price == pytest.approx(px[r.symbol], abs=0.0101)
    assert res.diagnostics["lookahead_guard_violations"] == 0


@pytest.mark.parametrize("sid", IDS)
def test_synthetic_near_miss(sid):
    sc = scen(sid)
    res = _run(sc)
    o = orders(res)
    for s in sc["near_miss"]:
        assert s not in set(o["symbol"]), f"{s} must not trade"
    # a window without a month-end produces no orders
    res2 = _run(sc, start="2023-06-05", end="2023-06-29")
    assert len(orders(res2)) == 0


@pytest.mark.parametrize("sid", IDS)
def test_lookahead_truncated_vs_full(sid):
    sc = scen(sid)
    cut = close_ts(SC.SIGNAL)
    full = _run(sc, end="2023-08-10")
    trunc = _run(sc, end=SC.SIGNAL, cut=cut)
    a, b = order_key(full, cut), order_key(trunc, cut)
    assert a and a == b


@pytest.mark.parametrize("sid", IDS)
def test_smoke_run_backtest(sid):
    sc = scen(sid)
    res = _run(sc, start="2023-03-01", end="2023-08-31")
    assert res.status == RunStatus.OK
    assert res.diagnostics["lookahead_guard_violations"] == 0
    assert len(res.equity) > 100
    assert res.meta["universe"] == "SP500"
    assert any("survivorship" in d for d in res.meta["deviations"])


# ----------------------------------------------------------------------------- specific behaviours
def test_s092_negative_earners_included_when_allowed():
    sc = SC.s092(exclude_negative=False)
    o = orders(_run(sc))
    assert set(o.loc[o["quantity"] < 0, "symbol"]) == set(sc["shorts"])
    assert "S07" in sc["shorts"]


def test_s093_publication_lag_30d_ignores_recent_filing():
    sc = SC.s093(lag="30d")
    o = orders(_run(sc))
    assert set(o.loc[o["quantity"] > 0, "symbol"]) == set(sc["longs"])
    assert set(o.loc[o["quantity"] < 0, "symbol"]) == set(sc["shorts"])
    assert set(sc["longs"]) != set(SC.s093("exact")["longs"]) or set(sc["shorts"]) != set(SC.s093("exact")["shorts"])


def test_s086_overlapping_cohorts_hold3():
    sc = scen("S086")
    strat = F.S086(hold_m=3)
    run(strat, sc["build"](), "2023-04-24", "2023-07-05")
    assert len(strat.cohorts) == 3
    exp: dict = {}
    for c in strat.cohorts:
        for s, v in c.items():
            exp[s] = exp.get(s, 0.0) + v / 3
    assert strat.last_weights == pytest.approx({s: v for s, v in exp.items() if abs(v) > 1e-12})


def test_s094_monthly_maintenance_drops_index_leavers():
    sc = scen("S094")

    class LeaverPortal(type(sc["build"]())):
        def universe(self, name, as_of):
            u = super().universe(name, as_of)
            return [s for s in u if not (s == sc["longs"][0] and pd.Timestamp(as_of) > pd.Timestamp("2023-07-15", tz="UTC"))]

    base = sc["build"]()
    p = LeaverPortal(bars={k: v for k, v in base._bars.items()}, fundamentals=base._fund, universes=base._universes,
                     master=base._master)
    strat = F.S094()
    res = run(strat, p, SC.START, "2023-08-03")
    o = orders(res)
    july = o[o["submitted_at"] == close_ts(pd.Timestamp("2023-07-31"))]
    assert sc["longs"][0] in set(july["symbol"])
    assert july.loc[july["symbol"] == sc["longs"][0], "quantity"].iloc[0] < 0      # position closed
    new = set(july["symbol"]) - set(sc["longs"]) - set(sc["shorts"])
    assert not new                                                                   # no new names in maintenance


def test_nyse_breakpoints_without_exchange_data_flags_t2():
    sc = scen("S096")
    portal = sc["build"]()
    portal._master = portal._master.drop(columns=["exchange"])
    res = run(F.S096(), portal, SC.START, SC.END)
    reasons = [r["reason"] for r in res.fidelity["reasons"]]
    assert any("NYSE size breakpoints approximated" in r for r in reasons)


# ----------------------------------------------------------------------------- helper known-answer tests
class _Ctx:
    def __init__(self, now):
        self.now = pd.Timestamp(now, tz="UTC")


def _fh(rows):
    fh = F.FundHistory(["x"], 0)
    fh._add(pd.DataFrame(rows))
    return fh


def test_fundhistory_ttm_known_answer_and_pit():
    from fsa_fixtures import fund_row
    rows = [fund_row("A", "ni_fy", 100.0, "2021-12-31", "2022-02-20"),
            fund_row("A", "ni_q", 20.0, "2022-03-31", "2022-05-05"),         # Q1-22 (YTD)
            fund_row("A", "ni_ytd", 45.0, "2022-06-30", "2022-08-05"),       # H1-22
            fund_row("A", "ni_q", 25.0, "2022-06-30", "2022-08-05"),         # Q2-22 discrete (not YTD)
            fund_row("A", "ni_fy", 120.0, "2022-12-31", "2023-02-20"),
            fund_row("A", "ni_q", 35.0, "2023-03-31", "2023-05-05"),
            fund_row("A", "ni_ytd", 70.0, "2023-06-30", "2023-08-04"),
            fund_row("A", "ni_fy", 999.0, "2022-12-31", "2023-09-01")]        # restatement known later
    fh = _fh(rows)
    t = lambda d: pd.Timestamp(d, tz="UTC")
    assert fh.ttm("A", "ni", t("2023-06-30 20:00")) == pytest.approx(120 + 35 - 20)
    assert fh.ttm("A", "ni", t("2023-08-10")) == pytest.approx(120 + 70 - 45)
    assert fh.ttm("A", "ni", t("2023-04-01")) == pytest.approx(120)               # only FY known
    assert math.isnan(fh.ttm("A", "ni", t("2022-01-01")))                         # nothing known yet
    assert fh.ttm("A", "ni", t("2023-09-02")) == pytest.approx(999 + 70 - 45)      # restated FY once known
    v, pe, ka = fh.latest("A", "ni_q", t("2023-06-30"))
    assert (v, pe) == (35.0, pd.Timestamp("2023-03-31"))


def test_month_end_sessions_and_week_end():
    me = F.month_end_sessions(SC.CAL, pd.Timestamp("2023-06-30"), 3)
    assert me == [pd.Timestamp(x) for x in ("2023-06-30", "2023-05-31", "2023-04-28", "2023-03-31")]
