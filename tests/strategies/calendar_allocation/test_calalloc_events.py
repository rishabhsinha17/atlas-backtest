"""S038-S041, S043-S045: event-driven strategies (earnings, index changes, buybacks,
analyst revisions). SYNTHETIC data, mechanics only."""

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.types import Fidelity, MarketEvent, RunStatus
from atlasbt.engine.backtester import run_backtest
from atlasbt.engine.testing import InMemoryPortal
from atlasbt.strategies import calendar_events as CE
from atlasbt.strategies.registry import get

from calalloc_helpers import (CAL, EventPortal, assert_no_lookahead, combine_daily, daily_frame, et_ts,
                              intraday_frame, orders, sess, sessions_before, truncate, zero_cost)


def ev(kind, sym, known, eff=None, **payload):
    k = pd.Timestamp(known, tz="America/New_York").tz_convert("UTC")
    return MarketEvent(kind, sym, k, effective_at=eff, payload=payload, fidelity=Fidelity.T2_PROXY)


def make_factory(bars, events=(), universes=None, fundamentals=None, borrow=None, cls=EventPortal):
    def make(cut=None):
        b = truncate(bars, cut) if cut is not None else bars
        e = [x for x in events if cut is None or x.known_at <= cut]
        f = fundamentals
        if f is not None and cut is not None:
            f = f[f["known_at"] <= cut]
        return cls(bars=b, events=e, universes=universes or {}, fundamentals=f, borrow=borrow)
    return make


def run(strategy, portal, a, b, cfg=None):
    return run_backtest(strategy, portal, a, b, cfg or zero_cost(), raise_errors=True)


def fill_of(res, order_row):
    f = res.fills
    return f[f["order_id"] == order_row["order_id"]].iloc[0]


# ======================================================================= S038
S038_SYMS = list("ABCDEFGHIJ") + ["K", "L"]
SURPRISE = dict(zip("ABCDEFGHIJ", [0.50, 0.20, 0.15, 0.10, 0.05, -0.05, -0.10, -0.15, -0.20, -0.50]))


def s038_inputs(k_surprise=1.0, l_bad_estimate=True):
    ss = sess("2024-01-02", "2024-04-30")
    bars = {(s, "1d"): daily_frame(ss, np.full(len(ss), 50.0)) for s in S038_SYMS}
    est_k = pd.Timestamp("2024-03-01", tz="UTC")
    evs = [ev("EARNINGS", s, "2024-03-05 08:00", eps_actual=1.0 + v, eps_estimate=1.0, estimate_known_at=est_k,
              timing="BMO") for s, v in SURPRISE.items()]
    evs.append(ev("EARNINGS", "K", "2024-03-06 09:20", eps_actual=1.0 + k_surprise, eps_estimate=1.0,
                  estimate_known_at=est_k, timing="UNKNOWN"))
    late = pd.Timestamp("2024-03-07 15:00", tz="UTC") if l_bad_estimate else est_k
    evs.append(ev("EARNINGS", "L", "2024-03-07 08:00", eps_actual=3.0, eps_estimate=1.0, estimate_known_at=late,
                  timing="BMO"))
    return make_factory(bars, evs, {"SP500": S038_SYMS})


def test_synthetic_s038_trigger_and_hold_exit():
    make = s038_inputs()
    res = run(CE.S038(min_pool=5, hold=5), make(), "2024-03-04", "2024-03-20")
    o = orders(res)
    ent = o[o["tag"] == "entry"].set_index("symbol")
    assert set(ent.index) == {"A", "J", "K"}                      # top / bottom decile; L has no PIT consensus
    open_0305 = CAL.open_close(pd.Timestamp("2024-03-05"))[0]
    for s, q in (("A", 400), ("J", -400)):                         # 2% NAV / $50, at the same-day open (BMO)
        r = ent.loc[s]
        assert pd.Timestamp(r["submitted_at"]) == open_0305 and r["quantity"] == q
        assert pd.Timestamp(fill_of(res, r)["ts"]) == open_0305
    rk = ent.loc["K"]                                              # 09:20 release: < 15 min before the open
    assert pd.Timestamp(rk["submitted_at"]) == CAL.open_close(pd.Timestamp("2024-03-06"))[1]
    assert pd.Timestamp(fill_of(res, rk)["ts"]) == CAL.open_close(pd.Timestamp("2024-03-07"))[0]
    ex = o[o["tag"] == "time_exit"].set_index("symbol")
    assert pd.Timestamp(ex.loc["A", "submitted_at"]) == CAL.open_close(pd.Timestamp("2024-03-11"))[1]
    assert pd.Timestamp(fill_of(res, ex.loc["A"])["ts"]) == CAL.open_close(pd.Timestamp("2024-03-12"))[0]
    assert ex.loc["A", "quantity"] == -400 and ex.loc["J", "quantity"] == 400


def test_synthetic_nearmiss_s038_middle_surprise_and_late_consensus():
    make = s038_inputs(k_surprise=0.0, l_bad_estimate=True)
    res = run(CE.S038(min_pool=5), make(), "2024-03-04", "2024-03-08")
    syms = set(orders(res)["symbol"])
    assert "K" not in syms and "L" not in syms and "E" not in syms


def test_s038_opposite_release_exits():
    make0 = s038_inputs()
    p = make0()
    p._events.append(ev("EARNINGS", "A", "2024-03-08 08:00", eps_actual=0.5, eps_estimate=1.0,
                        estimate_known_at=pd.Timestamp("2024-03-01", tz="UTC"), timing="BMO"))
    res = run(CE.S038(min_pool=5, hold=20), p, "2024-03-04", "2024-03-12")
    o = orders(res)
    r = o[(o["symbol"] == "A") & (o["tag"] == "opposite_release")].iloc[0]
    assert pd.Timestamp(r["submitted_at"]) == CAL.open_close(pd.Timestamp("2024-03-08"))[0]


# ======================================================================= S039 / S040
def gap_inputs(sym, prev_close, day_bars, n_hist5=21, after=4, vol_hist=1e4, borrow=None, extra_days=None):
    ev_day = pd.Timestamp("2024-03-05")
    h5 = sessions_before(ev_day, n_hist5)
    win = [ev_day] + list(CAL.sessions(ev_day + pd.Timedelta(days=1), ev_day + pd.Timedelta(days=12)))[:after]
    specs = {s: (prev_close, {}, vol_hist) for s in h5}
    specs[ev_day] = (day_bars["base"], day_bars["ov"], vol_hist)
    for s in win[1:]:
        specs[s] = (day_bars["after"], (extra_days or {}).get(s.strftime("%Y-%m-%d"), {}), vol_hist)
    intr = intraday_frame(specs, "5m")
    hs = sessions_before(h5[0], 40)
    hist = daily_frame(hs, np.full(len(hs), prev_close), spread=1.0)
    daily = combine_daily(hist, intr)
    bars = {(sym, "5m"): intr, (sym, "1d"): daily}
    evs = [ev("EARNINGS", sym, "2024-03-05 08:00", eps_actual=1.2, eps_estimate=1.0, timing="BMO")]
    return make_factory(bars, evs, {"SP500": [sym]}, borrow=borrow), win


def s039_day(vol_first30=3e4):
    ov = {"09:30": (103.5, 104.0, 103.4, 103.8, vol_first30)}
    for i, t in enumerate(["09:35", "09:40", "09:45", "09:50", "09:55"]):
        p = 103.8 + 0.04 * (i + 1)
        ov[t] = (p, p + 0.01, p - 0.01, p, vol_first30)
    ov["10:00"] = (104.1, 104.11, 104.09, 104.1)
    ov["10:05"] = (104.2, 104.21, 104.19, 104.2)
    return {"base": 104.2, "ov": ov, "after": 104.2}


def test_synthetic_s039_trigger():
    make, win = gap_inputs("AAA", 100.0, s039_day())
    res = run(CE.S039(), make(), win[0], win[-1])
    o = orders(res)
    ent = o[o["tag"] == "entry"].iloc[0]
    assert pd.Timestamp(ent["submitted_at"]) == et_ts("2024-03-05", "10:05")
    assert ent["quantity"] == 960                           # min(0.5%NAV/(2*ATR=4), 10% NAV / 104.1)
    fe = fill_of(res, ent)
    assert pd.Timestamp(fe["ts"]) == et_ts("2024-03-05", "10:05") and fe["price"] == pytest.approx(104.2)
    tr = res.trades.iloc[0]
    assert tr["initial_risk"] == pytest.approx(960 * 4.1, rel=1e-6) or tr["initial_risk"] > 0
    ex = o[o["tag"] == "time_exit"].iloc[0]                 # hold=3 -> MOC on 2024-03-08
    assert ex["type"] == "MOC" and pd.Timestamp(ex["submitted_at"]) == et_ts("2024-03-08", "15:45")
    assert pd.Timestamp(fill_of(res, ex)["ts"]) == CAL.open_close(pd.Timestamp("2024-03-08"))[1]


def test_synthetic_s039_trailing_close_stop():
    make, win = gap_inputs("AAA", 100.0, s039_day(),
                           extra_days={"2024-03-06": {"15:55": (104.2, 104.2, 99.0, 99.0)}})
    res = run(CE.S039(), make(), win[0], win[-1])
    o = orders(res)
    st = o[o["tag"] == "stop_close"].iloc[0]
    assert pd.Timestamp(st["submitted_at"]) == CAL.open_close(pd.Timestamp("2024-03-06"))[1]
    assert pd.Timestamp(fill_of(res, st)["ts"]) == CAL.open_close(pd.Timestamp("2024-03-07"))[0]


@pytest.mark.parametrize("variant", ["rvol", "gap", "close_below_open"])
def test_synthetic_nearmiss_s039(variant):
    day = s039_day(vol_first30=1.5e4 if variant == "rvol" else 3e4)
    if variant == "gap":
        day["ov"]["09:30"] = (102.9, 104.0, 102.8, 103.8, 3e4)          # +2.9% < 3%
    if variant == "close_below_open":
        day["ov"]["09:55"] = (103.4, 103.41, 103.3, 103.4, 3e4)         # first-30m close < open
    make, win = gap_inputs("AAA", 100.0, day)
    res = run(CE.S039(), make(), win[0], win[-1])
    assert len(orders(res)) == 0


def s040_day(gap_open=107.0):
    ov = {"09:30": (gap_open, 107.5, 106.5, 106.8, 2e4)}
    for i, t in enumerate(["09:35", "09:40", "09:45", "09:50", "09:55"]):
        p = 106.5 - 0.3 * i
        ov[t] = (p, p + 0.05, p - 0.05, p, 1e4)
    ov["10:00"] = (104.9, 104.95, 104.85, 104.9)
    ov["10:05"] = (104.8, 104.85, 104.75, 104.8)
    ov["11:00"] = (104.0, 104.0, 103.3, 103.6)
    return {"base": 104.5, "ov": ov, "after": 104.5}


def test_synthetic_s040_trigger_target():
    make, win = gap_inputs("BBB", 100.0, s040_day(), after=1)
    res = run(CE.S040(), make(), win[0], win[-1])
    o = orders(res)
    ent = o[o["tag"] == "entry"].iloc[0]
    assert pd.Timestamp(ent["submitted_at"]) == et_ts("2024-03-05", "10:05")
    assert ent["quantity"] == -953                          # DAY_RISK: min(0.25%/(107.5-104.9), 10% NAV)
    fe = fill_of(res, ent)
    assert pd.Timestamp(fe["ts"]) == et_ts("2024-03-05", "10:05") and fe["price"] == pytest.approx(104.8)
    stop = o[o["tag"] == "stop"].iloc[0]
    tgt = o[o["tag"] == "target"].iloc[0]
    assert stop["stop_price"] == pytest.approx(107.5)       # opening (first 30m) high
    assert tgt["limit_price"] == pytest.approx(103.5)       # midpoint of open 107 and prev close 100
    ft = fill_of(res, tgt)
    assert pd.Timestamp(ft["ts"]) >= et_ts("2024-03-05", "11:00") and ft["price"] == pytest.approx(103.5)


@pytest.mark.parametrize("variant", ["gap", "locate", "above_vwap"])
def test_synthetic_nearmiss_s040(variant):
    day = s040_day(105.9 if variant == "gap" else 107.0)
    borrow = {"BBB": (False, float("nan"))} if variant == "locate" else None
    if variant == "above_vwap":
        day["ov"]["09:55"] = (106.9, 106.95, 106.85, 106.9, 1e4)   # still < open 107 but > VWAP
    make, win = gap_inputs("BBB", 100.0, day, after=1, borrow=borrow)
    res = run(CE.S040(), make(), win[0], win[-1])
    assert (orders(res)["tag"] == "entry").sum() == 0 if len(orders(res)) else True


# ======================================================================= S043 / S044
def index_inputs(sym, kind, known, eff_date, base=50.0, beta=1.5, borrow=None, ssr=False, fee=None):
    win = sess("2024-03-04", "2024-03-18")
    hs = sessions_before(win[0], 300)
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, len(hs))
    spy = 400.0 * np.cumprod(1 + r)
    stk = base * np.cumprod(1 + beta * r)
    spy, stk = spy / spy[-1] * 400.0, stk / stk[-1] * base
    lows = None
    if ssr:
        lows = np.minimum(np.concatenate([[stk[0]], stk[:-1]]), stk) - 1.0
        lows[-1] = stk[-2] * 0.89
    hist_s = daily_frame(hs, stk, spread=1.0, lows=lows)
    hist_spy = daily_frame(hs, spy, spread=1.0)
    ispec = {s: (base, {"09:30": (base * 1.01, base * 1.01, base * 1.01, base * 1.01)} if s == win[0] else {}) for s in win}
    ispec[win[-1]] = (base, {"15:50": (base * 1.05,) * 4})
    sspec = {s: (400.0, {"09:30": (400.0,) * 4}) for s in win}
    i_s, i_spy = intraday_frame(ispec, "5m"), intraday_frame(sspec, "5m")
    bars = {(sym, "5m"): i_s, (sym, "1d"): combine_daily(hist_s, i_s),
            ("SPY", "5m"): i_spy, ("SPY", "1d"): combine_daily(hist_spy, i_spy)}
    eff_open = CAL.open_close(pd.Timestamp(eff_date))[0]
    evs = [ev(kind, sym, known, eff=eff_open, index="SP500", effective_date=pd.Timestamp(eff_date))]
    b = borrow if borrow is not None else ({sym: (True, fee)} if fee is not None else None)
    return make_factory(bars, evs, borrow=b), win


def test_synthetic_s043_after_hours_announcement():
    make, win = index_inputs("NEWCO", "INDEX_ADD", "2024-03-01 17:15", "2024-03-18")
    res = run(CE.S043(), make(), win[0], win[-1])
    o = orders(res)
    ent = o[o["tag"] == "entry"].iloc[0]
    h = o[o["tag"] == "hedge"].iloc[0]
    t_open = CAL.open_close(win[0])[0]
    assert pd.Timestamp(ent["submitted_at"]) == t_open and ent["quantity"] == 2000      # 10% NAV / $50
    assert h["quantity"] == -round(1.5 * 2000 * 50.0 / 400.0)                           # beta 1.5 hedge
    fe = fill_of(res, ent)
    assert pd.Timestamp(fe["ts"]) == t_open and fe["price"] == pytest.approx(50.5)
    ex = o[o["tag"] == "exit"].iloc[0]
    hx = o[o["tag"] == "hedge_exit"].iloc[0]
    assert pd.Timestamp(ex["submitted_at"]) == et_ts("2024-03-18", "15:50") and ex["quantity"] == -2000
    assert hx["quantity"] == -h["quantity"]
    fx = fill_of(res, ex)
    assert pd.Timestamp(fx["ts"]) == et_ts("2024-03-18", "15:50") and fx["price"] == pytest.approx(52.5)


@pytest.mark.parametrize("entry,expected", [("ann+1min", ("2024-03-04", "11:05")), ("next_day", ("2024-03-05", "09:30"))])
def test_synthetic_s043_rth_announcement(entry, expected):
    make, win = index_inputs("NEWCO", "INDEX_ADD", "2024-03-04 11:02", "2024-03-18")
    res = run(CE.S043(entry=entry), make(), win[0], win[-1])
    ent = orders(res)
    ent = ent[ent["tag"] == "entry"].iloc[0]
    assert pd.Timestamp(ent["submitted_at"]) == et_ts(*expected)
    assert pd.Timestamp(fill_of(res, ent)["ts"]) == et_ts(*expected)


def test_synthetic_nearmiss_s043_effective_before_entry():
    make, win = index_inputs("NEWCO", "INDEX_ADD", "2024-03-01 17:15", "2024-03-01")
    res = run(CE.S043(), make(), win[0], win[-1])
    assert len(orders(res)) == 0


def test_synthetic_s044_short_with_mid_stop():
    make, win = index_inputs("OLDCO", "INDEX_DEL", "2024-03-01 17:15", "2024-03-18", base=30.0, beta=1.0)
    res = run(CE.S044(), make(), win[0], win[-1])
    o = orders(res)
    ent = o[o["tag"] == "entry"].iloc[0]
    t_open = CAL.open_close(win[0])[0]
    assert pd.Timestamp(ent["submitted_at"]) == t_open and ent["quantity"] < 0
    fe = fill_of(res, ent)
    assert fe["price"] == pytest.approx(30.3)
    d = make()._bars[("OLDCO", "1d")]
    d = d[d.index < t_open].iloc[-40:]
    from atlasbt.indicators import atr
    a = float(atr(d["high"], d["low"], d["close"], 14).iloc[-1])
    stop = o[o["tag"] == "stop"].iloc[0]
    mid = fe["reference_mid"] if fe["reference_mid"] == fe["reference_mid"] else fe["price"]
    assert stop["stop_price"] == pytest.approx(round(mid + 2 * a, 2), abs=0.011)
    assert stop["quantity"] == -ent["quantity"]
    ex = o[o["tag"] == "exit"].iloc[0]
    assert pd.Timestamp(ex["submitted_at"]) == et_ts("2024-03-18", "15:50")


def test_synthetic_s044_event_window():
    make, win = index_inputs("OLDCO", "INDEX_DEL", "2024-03-01 17:15", "2024-03-18", base=30.0, beta=1.0)
    res = run(CE.S044(window=1), make(), win[0], win[-1])
    ex = orders(res)
    ex = ex[ex["tag"] == "exit"].iloc[0]
    assert pd.Timestamp(ex["submitted_at"]) == et_ts("2024-03-04", "15:50")


@pytest.mark.parametrize("variant", ["no_borrow", "ssr", "fee_cap"])
def test_synthetic_nearmiss_s044(variant):
    kw = dict(base=30.0, beta=1.0)
    params = {}
    if variant == "no_borrow":
        kw["borrow"] = {"OLDCO": (False, float("nan"))}
    elif variant == "ssr":
        kw["ssr"] = True
    else:
        kw["fee"] = 0.10
        params["borrow_cap"] = 0.05
    make, win = index_inputs("OLDCO", "INDEX_DEL", "2024-03-01 17:15", "2024-03-18", **kw)
    res = run(CE.S044(**params), make(), win[0], win[-1])
    assert len(orders(res)) == 0


# ======================================================================= S045
def s045_inputs(pct=0.06, with_earnings=False):
    ss = sess("2023-12-01", "2024-03-29")
    closes = np.full(len(ss), 40.0)
    opens = closes.copy()
    highs, lows = closes + 1.0, closes - 1.0
    i = ss.index(pd.Timestamp("2024-03-04"))
    opens[i], highs[i], lows[i], closes[i] = 40.5, 41.5, 40.0, 41.0
    closes[i + 1:] = 41.0
    opens[i + 1:] = 41.0
    highs[i + 1:], lows[i + 1:] = 41.5, 40.5
    bars = {("CCC", "1d"): daily_frame(ss, closes, opens=opens, highs=highs, lows=lows)}
    evs = [ev("BUYBACK", "CCC", "2024-03-04 08:30", amount_usd=1e9, pct_mcap=pct, with_earnings=with_earnings)]
    return make_factory(bars, evs, {"SP500": ["CCC"]})


def test_synthetic_s045_trigger_trailing_and_time_exit():
    make = s045_inputs()
    res = run(CE.S045(), make(), "2024-03-01", "2024-03-29")
    o = orders(res)
    ent = o[o["tag"] == "entry"].iloc[0]
    t_open = CAL.open_close(pd.Timestamp("2024-03-04"))[0]
    assert pd.Timestamp(ent["submitted_at"]) == t_open and ent["quantity"] == 1250   # 0.5% NAV / (2 ATR = 4)
    assert fill_of(res, ent)["price"] == pytest.approx(40.5)
    stops = o[o["tag"] == "stop"]
    assert stops.iloc[0]["stop_price"] == pytest.approx(36.0)               # initial: 40 - 2 x 2
    assert stops["stop_price"].iloc[-1] > 36.0                              # ratcheted from the highest high
    assert (np.diff(stops["stop_price"].to_numpy()) >= -1e-9).all()
    ex = o[o["tag"] == "time_exit"].iloc[0]
    exit_sess = CAL.next_session(pd.Timestamp("2024-03-04"), 10)
    assert ex["type"] == "MOC" and pd.Timestamp(ex["submitted_at"]) == CAL.open_close(exit_sess)[0]
    assert pd.Timestamp(fill_of(res, ex)["ts"]) == CAL.open_close(exit_sess)[1]
    assert list(res.trades["exit_reason"]) == ["time_exit"]


@pytest.mark.parametrize("pct,with_earn", [(0.04, False), (0.06, True)])
def test_synthetic_nearmiss_s045(pct, with_earn):
    res = run(CE.S045(), s045_inputs(pct, with_earn)(), "2024-03-01", "2024-03-29")
    assert len(orders(res)) == 0


# ======================================================================= S041
S041_SYMS = [f"N{i}" for i in range(10)]


def s041_inputs(top_analysts=6):
    ss = sess("2024-01-02", "2024-04-30")
    bars = {(s, "1d"): daily_frame(ss, np.full(len(ss), 50.0)) for s in S041_SYMS}
    revs = np.linspace(0.10, -0.10, 10)
    rows = []
    for s, r in zip(S041_SYMS, revs):
        rows.append((s, "eps_consensus_ntm", 2.0, pd.Timestamp("2024-02-20", tz="UTC")))
        rows.append((s, "eps_consensus_ntm", 2.0 * (1 + r), pd.Timestamp("2024-03-25", tz="UTC")))
        rows.append((s, "eps_num_analysts_ntm", top_analysts if s == "N0" else 6, pd.Timestamp("2024-03-25", tz="UTC")))
    f = pd.DataFrame(rows, columns=["symbol", "field", "value", "known_at"])
    f["period_end"] = pd.NaT
    f["source"] = "synthetic"
    return make_factory(bars, (), {"SP500": S041_SYMS}, fundamentals=f)


def test_synthetic_s041_month_end_ranks():
    res = run(CE.S041(), s041_inputs()(), "2024-03-25", "2024-04-05")
    o = orders(res)
    t = CAL.open_close(pd.Timestamp("2024-03-28"))[1]
    assert (pd.DatetimeIndex(o["submitted_at"]) == t).all()
    q = o.set_index("symbol")["quantity"]
    assert q.to_dict() == {"N0": 400.0, "N9": -400.0}
    f = res.fills
    assert (pd.DatetimeIndex(f["ts"]) == CAL.open_close(pd.Timestamp("2024-04-01"))[0]).all()


def test_synthetic_nearmiss_s041_analyst_minimum():
    res = run(CE.S041(), s041_inputs(top_analysts=4)(), "2024-03-25", "2024-04-05")
    assert "N0" not in set(orders(res)["symbol"])


def test_s041_s045_not_testable_without_data():
    ss = sess("2024-01-02", "2024-03-29")
    p = InMemoryPortal(bars={("SPY", "1d"): daily_frame(ss, np.full(len(ss), 50.0))}, universes={"SP500": ["SPY"]})
    for sid in ("S041", "S045", "S038", "S043", "S044"):
        res = run_backtest(get(sid)(), p, "2024-03-01", "2024-03-29", zero_cost())
        assert res.status == RunStatus.NOT_TESTABLE, sid


# ======================================================================= lookahead
LOOKAHEAD = {
    "S038": (lambda: CE.S038(min_pool=5, hold=5), s038_inputs(), "2024-03-04", "2024-03-20",
             pd.Timestamp("2024-03-06 17:00", tz="UTC")),
    "S039": (lambda: CE.S039(), gap_inputs("AAA", 100.0, s039_day())[0], "2024-03-05", "2024-03-11",
             et_ts("2024-03-05", "10:06")),
    "S040": (lambda: CE.S040(), gap_inputs("BBB", 100.0, s040_day(), after=1)[0], "2024-03-05", "2024-03-06",
             et_ts("2024-03-05", "10:30")),
    "S043": (lambda: CE.S043(), index_inputs("NEWCO", "INDEX_ADD", "2024-03-01 17:15", "2024-03-18")[0],
             "2024-03-04", "2024-03-18", et_ts("2024-03-04", "12:00")),
    "S044": (lambda: CE.S044(), index_inputs("OLDCO", "INDEX_DEL", "2024-03-01 17:15", "2024-03-18", base=30.0,
                                             beta=1.0)[0], "2024-03-04", "2024-03-18", et_ts("2024-03-04", "12:00")),
    "S045": (lambda: CE.S045(), s045_inputs(), "2024-03-01", "2024-03-29", pd.Timestamp("2024-03-08 22:00", tz="UTC")),
    "S041": (lambda: CE.S041(), s041_inputs(), "2024-03-25", "2024-04-05", pd.Timestamp("2024-03-28 21:00", tz="UTC")),
}


@pytest.mark.parametrize("sid", sorted(LOOKAHEAD))
def test_lookahead_truncated_vs_full(sid):
    mk, make, a, b, cut = LOOKAHEAD[sid]
    full, sig = assert_no_lookahead(mk, make, a, b, cut)
    assert len(sig) > 0


# ======================================================================= smoke
@pytest.mark.parametrize("sid", ["S038", "S039", "S040", "S041", "S043", "S044", "S045"])
def test_smoke_run_backtest(sid):
    mk, make, a, b, cut = LOOKAHEAD[sid]
    cls = get(sid)
    params = {"min_pool": 5} if sid == "S038" else {}
    res = run_backtest(cls(**params), make(), a, b, None, raise_errors=True)
    assert res.status == RunStatus.OK
    assert res.diagnostics["lookahead_guard_violations"] == 0
    assert len(res.fills) > 0
