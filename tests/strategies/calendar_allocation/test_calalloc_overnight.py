"""S031-S037, S042: overnight / calendar strategies on 1m bars (and the daily proxy).

SYNTHETIC data, mechanics only: exact signal bar, order submission time, fill bar
and price, near misses, lookahead (truncated vs full data) and smoke runs.
"""
import math

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.types import ActionType, CorporateAction, RunStatus
from atlasbt.engine.backtester import run_backtest
from atlasbt.engine.testing import InMemoryPortal, synthetic_daily_bars, synthetic_intraday_bars, daily_from_intraday
from atlasbt.strategies import calendar_events as CE
from atlasbt.strategies.registry import get

from calalloc_helpers import (CAL, assert_no_lookahead, combine_daily, daily_frame, et_ts, intraday_frame, orders,
                              sess, sessions_before, truncate, zero_cost)

SYM = "SPY"


def small(cls, **attrs):
    return type(cls.__name__ + "T", (cls,), {"universe": [SYM], **attrs})


def hist_closes(kind: str, n: int = 260, last: float = 100.0) -> np.ndarray:
    t = np.arange(n)
    if kind == "flat":
        return np.full(n, last)
    if kind == "up":            # trend up with +/-1% alternating noise -> close > SMA200
        base = np.exp(0.002 * (t - n + 1))
    elif kind == "down":
        base = np.exp(-0.002 * (t - n + 1))
    elif kind == "alt":         # no trend, alternating +/-1%
        base = np.ones(n)
    else:
        raise ValueError(kind)
    noise = np.where(t % 2 == 0, 1.0, 1.01)
    c = base * noise
    return c / c[-1] * last


def build(window, day_specs, hist="up", actions=None, last=100.0, proxy=False):
    """-> portal factory(cut) over SPY with 260 sessions of daily history before ``window``."""
    hs = sessions_before(window[0], 260)
    h = daily_frame(hs, hist_closes(hist, 260, last))
    intr = intraday_frame(day_specs)
    daily = combine_daily(h, intr)
    bars = {(SYM, "1d"): daily} if proxy else {(SYM, "1m"): intr, (SYM, "1d"): daily}

    def make(cut=None):
        b = truncate(bars, cut) if cut is not None else bars
        acts = [a for a in (actions or []) if cut is None or a.known_at <= cut]
        return InMemoryPortal(bars=b, actions=acts)
    return make


def flat_days(window, base=100.0, **ov):
    """{session: (base, overrides)} with overrides keyed by session date string."""
    return {s: (base, ov.get(s.strftime("%Y-%m-%d"), {})) for s in window}


def bar(p):
    return (p, p, p, p)


def run(strategy, portal, window, cfg=None):
    return run_backtest(strategy, portal, window[0], window[-1], cfg or zero_cost(), raise_errors=True)


def entries_exits(res):
    o = orders(res)
    f = res.fills
    return o, f


# ====================================================================== scenarios
def sc_s031():
    w = sess("2024-03-04", "2024-03-06")
    days = flat_days(w, 100.0, **{"2024-03-04": {"15:50": bar(101.0)}, "2024-03-05": {"09:35": bar(102.0)}})
    return small(CE.S031), {}, build(w, days, "up"), w, dict(
        entry=(et_ts("2024-03-04", "15:50"), 101.0, 1000), exit=(et_ts("2024-03-05", "09:35"), 102.0))


def sc_s032(signal_time="15:45"):
    w = sess("2024-03-04", "2024-03-05")
    sig_bar = "15:44" if signal_time == "15:45" else "15:49"
    days = flat_days(w, 100.0, **{"2024-03-04": {sig_bar: (100.0, 100.0, 97.0, 97.0), "15:50": bar(97.2)},
                                   "2024-03-05": {"09:35": bar(99.0)}})
    mark = 97.0 if signal_time == "15:50" else 100.0
    return small(CE.S032), {"signal_time": signal_time}, build(w, days, "alt"), w, dict(
        entry=(et_ts("2024-03-04", "15:50"), 97.2, math.floor(100000 / mark)),
        exit=(et_ts("2024-03-05", "09:35"), 99.0))


def sc_s033():
    w = sess("2024-03-04", "2024-03-05")
    days = flat_days(w, 100.0, **{"2024-03-04": {"15:44": (100.0, 103.0, 100.0, 103.0), "15:50": bar(103.1)},
                                   "2024-03-05": {"09:35": bar(104.0)}})
    return small(CE.S033), {}, build(w, days, "up"), w, dict(
        entry=(et_ts("2024-03-04", "15:50"), 103.1, 1000), exit=(et_ts("2024-03-05", "09:35"), 104.0))


def sc_s034():
    w = sess("2024-03-26", "2024-04-05")
    days = flat_days(w, 100.0, **{"2024-03-27": {"15:50": bar(101.0)}, "2024-04-04": {"09:35": bar(103.0)}})
    return small(CE.S034), {}, build(w, days, "flat"), w, dict(
        entry=(et_ts("2024-03-27", "15:50"), 101.0, 1000), exit=(et_ts("2024-04-04", "09:35"), 103.0))


def sc_s035():
    w = sess("2024-03-27", "2024-04-02")
    days = flat_days(w, 100.0, **{"2024-03-28": {"15:50": bar(101.0)}, "2024-04-01": {"15:50": bar(99.0)}})
    return small(CE.S035), {}, build(w, days, "flat"), w, dict(
        entry=(et_ts("2024-03-28", "15:50"), 101.0, 1000), exit=(et_ts("2024-04-01", "15:50"), 99.0))


def sc_s036():
    w = sess("2024-03-04", "2024-03-08")
    days = flat_days(w, 100.0, **{"2024-03-04": {"09:35": bar(100.5)}, "2024-03-08": {"15:50": bar(102.0)}})
    return small(CE.S036), {}, build(w, days, "up"), w, dict(
        entry=(et_ts("2024-03-04", "09:35"), 100.5, 1000), exit=(et_ts("2024-03-08", "15:50"), 102.0))


def sc_s037():
    w = sess("2024-07-01", "2024-07-05")
    # 2024-07-03 is an early close (13:00): "15:50" becomes 12:50
    days = flat_days(w, 100.0, **{"2024-07-02": {"15:50": bar(101.0)}, "2024-07-03": {"12:50": bar(102.0)}})
    return small(CE.S037), {}, build(w, days, "flat"), w, dict(
        entry=(et_ts("2024-07-02", "15:50"), 101.0, 1000), exit=(et_ts("2024-07-03", "12:50"), 102.0))


def div(known="2024-03-01 12:00"):
    return CorporateAction(SYM, ActionType.CASH_DIVIDEND, pd.Timestamp("2024-03-15"),
                           pd.Timestamp(known, tz="America/New_York").tz_convert("UTC"), amount=1.5,
                           pay_date=pd.Timestamp("2024-04-30"))


def sc_s042(known="2024-03-01 12:00"):
    w = sess("2024-03-13", "2024-03-15")
    days = flat_days(w, 100.0, **{"2024-03-14": {"15:50": bar(100.2)}, "2024-03-15": {"09:35": bar(98.7)}})
    return small(CE.S042), {}, build(w, days, "flat", actions=[div(known)]), w, dict(
        entry=(et_ts("2024-03-14", "15:50"), 100.2, 1000), exit=(et_ts("2024-03-15", "09:35"), 98.7))


SCENARIOS = {"S031": sc_s031, "S032": sc_s032, "S033": sc_s033, "S034": sc_s034, "S035": sc_s035,
             "S036": sc_s036, "S037": sc_s037, "S042": sc_s042}


def _check_trade(res, exp):
    o, f = entries_exits(res)
    ent = o[o["tag"] == "entry"].iloc[0]
    ex = o[o["tag"] == "exit"].iloc[0]
    t_sub, px, q = exp["entry"]
    assert pd.Timestamp(ent["submitted_at"]) == t_sub
    assert ent["quantity"] == q
    fe = f[f["order_id"] == ent["order_id"]].iloc[0]
    assert pd.Timestamp(fe["ts"]) == t_sub                     # bar starting at HH:MM
    assert fe["price"] == pytest.approx(px)
    t_ex, px_ex = exp["exit"]
    assert pd.Timestamp(ex["submitted_at"]) == t_ex
    assert ex["quantity"] == -q
    fx = f[f["order_id"] == ex["order_id"]].iloc[0]
    assert pd.Timestamp(fx["ts"]) == t_ex
    assert fx["price"] == pytest.approx(px_ex)


# ====================================================================== trigger tests
@pytest.mark.parametrize("sid", sorted(SCENARIOS))
def test_synthetic_trigger_exact_timing(sid):
    cls, params, make, w, exp = SCENARIOS[sid]()
    res = run(cls(**params), make(), w)
    assert res.status == RunStatus.OK
    _check_trade(res, exp)
    # the decision never used a bar that ended after the decision time
    o = orders(res)
    first = o[o["tag"] == "entry"].iloc[0]
    assert pd.Timestamp(first["submitted_at"]) <= exp["entry"][0]


def test_synthetic_s032_signal_time_1550():
    cls, params, make, w, exp = sc_s032("15:50")
    res = run(cls(**params), make(), w)
    _check_trade(res, exp)


def test_synthetic_s034_close_to_close_variant():
    w = sess("2024-03-26", "2024-04-05")
    days = flat_days(w, 100.0, **{"2024-03-28": {"15:50": bar(101.0)}, "2024-04-03": {"15:50": bar(103.0)}})
    res = run(small(CE.S034)(start=0, end=3, exit_time="15:50"), build(w, days, "flat")(), w)
    _check_trade(res, dict(entry=(et_ts("2024-03-28", "15:50"), 101.0, 1000),
                           exit=(et_ts("2024-04-03", "15:50"), 103.0)))


def test_synthetic_s036_monday_holiday_enters_tuesday():
    w = sess("2024-02-20", "2024-02-23")                 # 2024-02-19 Presidents Day
    days = flat_days(w, 100.0, **{"2024-02-20": {"09:35": bar(100.5)}, "2024-02-23": {"15:50": bar(101.0)}})
    res = run(small(CE.S036)(), build(w, days, "up")(), w)
    _check_trade(res, dict(entry=(et_ts("2024-02-20", "09:35"), 100.5, 1000),
                           exit=(et_ts("2024-02-23", "15:50"), 101.0)))


def test_synthetic_s031_nights_3_holds_continuously():
    w = sess("2024-03-04", "2024-03-08")
    days = flat_days(w, 100.0, **{"2024-03-04": {"15:50": bar(101.0)}, "2024-03-07": {"09:35": bar(102.0)}})
    res = run(small(CE.S031)(nights=3), build(w, days, "up")(), w)
    o = orders(res)
    ent = o[o["tag"] == "entry"]
    assert list(pd.DatetimeIndex(ent["submitted_at"])[:2]) == [et_ts("2024-03-04", "15:50"), et_ts("2024-03-07", "15:50")]
    ex = o[o["tag"] == "exit"].iloc[0]
    assert pd.Timestamp(ex["submitted_at"]) == et_ts("2024-03-07", "09:35")


# ====================================================================== near misses
def test_synthetic_nearmiss_s031_zero_momentum():
    w = sess("2024-03-04", "2024-03-06")
    res = run(small(CE.S031)(), build(w, flat_days(w), "flat")(), w)
    assert len(orders(res)) == 0
    res0 = run(small(CE.S031)(mom_days=0), build(w, flat_days(w), "flat")(), w)   # unconditional comparison
    assert (orders(res0)["tag"] == "entry").sum() == 3


def test_synthetic_nearmiss_s032_shock_too_small():
    w = sess("2024-03-04", "2024-03-05")
    days = flat_days(w, 100.0, **{"2024-03-04": {"15:44": (100.0, 100.0, 98.5, 98.5)}})
    res = run(small(CE.S032)(), build(w, days, "alt")(), w)
    assert len(orders(res)) == 0


def test_synthetic_nearmiss_s033_spy_below_sma200():
    w = sess("2024-03-04", "2024-03-05")
    days = flat_days(w, 100.0, **{"2024-03-04": {"15:44": (100.0, 103.0, 100.0, 103.0)}})
    res = run(small(CE.S033)(), build(w, days, "down")(), w)
    assert len(orders(res)) == 0


def test_synthetic_nearmiss_s034_s035_no_month_end():
    w = sess("2024-03-11", "2024-03-15")
    for cls in (CE.S034, CE.S035):
        res = run(small(cls)(), build(w, flat_days(w), "flat")(), w)
        assert len(orders(res)) == 0


def test_synthetic_nearmiss_s036_trend_gate():
    w = sess("2024-03-04", "2024-03-08")
    assert len(orders(run(small(CE.S036)(), build(w, flat_days(w), "down")(), w))) == 0
    assert len(orders(run(small(CE.S036)(trend_gate=False), build(w, flat_days(w), "down")(), w))) == 2


def test_synthetic_nearmiss_s037_degenerate_and_no_holiday():
    w = sess("2024-07-01", "2024-07-05")
    assert len(orders(run(small(CE.S037)(days_before=1, exit="before"), build(w, flat_days(w), "flat")(), w))) == 0
    w2 = sess("2024-03-04", "2024-03-08")
    assert len(orders(run(small(CE.S037)(), build(w2, flat_days(w2), "flat")(), w2))) == 0


def test_synthetic_nearmiss_s042_dividend_not_yet_known():
    cls, params, make, w, exp = sc_s042(known="2024-03-14 16:30")   # declared after the 15:50 decision
    res = run(cls(**params), make(), w)
    assert len(orders(res)) == 0


def test_s042_dividend_entitlement_booked():
    cls, params, make, w, exp = sc_s042()
    res = run(cls(**params), make(), w)
    ca = res.diagnostics["corporate_actions"]
    assert any(c.get("symbol") == SYM for c in ca)


def test_s037_unscheduled_closures_are_not_holidays():
    pre = CE.pre_holiday_sessions(CAL, "2012-10-01", "2012-10-31")
    assert pd.Timestamp("2012-10-26") not in pre                    # Sandy closure 10-29/30
    pre18 = CE.pre_holiday_sessions(CAL, "2018-12-01", "2018-12-31")
    assert pd.Timestamp("2018-12-04") not in pre18                  # Bush day of mourning
    assert pd.Timestamp("2018-12-24") in pre18                      # Christmas


# ====================================================================== daily proxy
def test_synthetic_s031_proxy_moc_moo():
    w = sess("2024-03-04", "2024-03-06")
    hs = sessions_before(w[0], 260)
    closes = np.concatenate([hist_closes("up", 260), [100.0, 101.0, 102.0]])
    opens = np.concatenate([hist_closes("up", 260), [99.5, 100.7, 101.4]])
    daily = daily_frame(hs + w, closes, opens=opens)
    portal = InMemoryPortal(bars={(SYM, "1d"): daily})
    res = run(small(CE.S031)(proxy=True), portal, w)
    assert res.status == RunStatus.OK
    o, f = orders(res), res.fills
    ent = o[o["tag"] == "entry"].iloc[0]
    assert ent["type"] == "MOC" and pd.Timestamp(ent["submitted_at"]) == CAL.open_close(w[0])[0]
    fe = f[f["order_id"] == ent["order_id"]].iloc[0]
    assert pd.Timestamp(fe["ts"]) == CAL.open_close(w[0])[1] and fe["price"] == pytest.approx(100.0)
    ex = o[o["tag"] == "exit"].iloc[0]
    assert ex["type"] == "MOO" and pd.Timestamp(ex["submitted_at"]) == CAL.open_close(w[0])[1]
    fx = f[f["order_id"] == ex["order_id"]].iloc[0]
    assert pd.Timestamp(fx["ts"]) == CAL.open_close(w[1])[0] and fx["price"] == pytest.approx(100.7)
    assert any("proxy" in r["reason"] for r in res.fidelity["reasons"]) or res.fidelity["grade"] in ("C", "D", "E")


@pytest.mark.parametrize("sid", ["S032", "S033"])
def test_proxy_unsupported_is_not_testable(sid):
    w = sess("2024-03-04", "2024-03-06")
    daily = daily_frame(sessions_before(w[0], 260) + w, np.full(263, 100.0))
    res = run_backtest(small(get(sid))(proxy=True), InMemoryPortal(bars={(SYM, "1d"): daily}), w[0], w[-1],
                       zero_cost())
    assert res.status == RunStatus.NOT_TESTABLE


@pytest.mark.parametrize("sid", sorted(SCENARIOS))
def test_default_requires_intraday_bars(sid):
    """Daily data only -> the faithful default is NOT_TESTABLE (never silently proxied)."""
    w = sess("2024-03-04", "2024-03-06")
    daily = daily_frame(sessions_before(w[0], 260) + w, np.full(263, 100.0))
    res = run_backtest(small(get(sid))(), InMemoryPortal(bars={(SYM, "1d"): daily}), w[0], w[-1], zero_cost())
    assert res.status == RunStatus.NOT_TESTABLE


# ====================================================================== lookahead
@pytest.mark.parametrize("sid", sorted(SCENARIOS))
def test_lookahead_truncated_vs_full(sid):
    cls, params, make, w, exp = SCENARIOS[sid]()
    cut = exp["entry"][0] + pd.Timedelta(minutes=1)            # just after the entry decision
    full, sig = assert_no_lookahead(lambda: cls(**params), make, w[0], w[-1], cut)
    assert any(s[3] == "entry" for s in sig)
    cut2 = exp["exit"][0] + pd.Timedelta(minutes=1)
    assert_no_lookahead(lambda: cls(**params), make, w[0], w[-1], cut2)


# ====================================================================== smoke
@pytest.mark.parametrize("sid", sorted(SCENARIOS))
def test_smoke_run_backtest(sid):
    syms = ["SPY", "QQQ"]
    start, end = "2024-06-24", "2024-07-09"
    bars = {}
    for i, s in enumerate(syms):
        intr = synthetic_intraday_bars(start, end, "1m", 100.0 + 50 * i, seed=i, volume_per_bar=1e6)
        hist = synthetic_daily_bars("2023-01-03", "2024-06-21", 100.0 + 50 * i, seed=10 + i, volume=1e8)
        d = daily_from_intraday(intr)
        d["volume"] = 1e8
        scale = d["open"].iloc[0] / hist["close"].iloc[-1]
        for c in ("open", "high", "low", "close"):
            hist[c] = hist[c] * scale
        bars[(s, "1m")] = intr
        bars[(s, "1d")] = pd.concat([hist, d])
    acts = [CorporateAction("SPY", ActionType.CASH_DIVIDEND, pd.Timestamp("2024-06-28"),
                            pd.Timestamp("2024-06-01", tz="UTC"), amount=0.5, pay_date=pd.Timestamp("2024-07-31"))]
    portal = InMemoryPortal(bars=bars, actions=acts)
    cls = type(sid + "Smoke", (get(sid),), {"universe": syms})
    res = run_backtest(cls(), portal, start, end, None, raise_errors=True)
    assert res.status == RunStatus.OK
    assert res.diagnostics["lookahead_guard_violations"] == 0
    assert len(res.equity) == len(sess(start, end))


@pytest.mark.parametrize("sid", ["S031", "S034", "S035", "S036", "S037", "S042"])
def test_proxy_smoke_auction_orders_only(sid):
    """proxy=True (opt-in, T2_PROXY): daily bars only, 15:50 -> MOC, 09:35 -> MOO."""
    start, end = "2024-06-03", "2024-07-31"
    syms = ["SPY", "QQQ"]
    bars = {(s, "1d"): synthetic_daily_bars("2023-01-03", end, 100.0 + 50 * i, seed=20 + i, volume=1e8,
                                            drift=0.001) for i, s in enumerate(syms)}
    acts = [CorporateAction("SPY", ActionType.CASH_DIVIDEND, pd.Timestamp("2024-06-21"),
                            pd.Timestamp("2024-06-01", tz="UTC"), amount=0.5, pay_date=pd.Timestamp("2024-07-31"))]
    cls = type(sid + "Proxy", (get(sid),), {"universe": syms})
    res = run_backtest(cls(proxy=True), InMemoryPortal(bars=bars, actions=acts), start, end, None, raise_errors=True)
    assert res.status == RunStatus.OK and res.meta["bar_freq"] == "1d"
    o = orders(res)
    assert len(o) > 0 and set(o["type"]) <= {"MOC", "MOO", "MKT"}
    assert not res.diagnostics["rejected_orders"]
    assert any(f["tier"] == "T2_PROXY" for f in res.fidelity.get("flags", [])) or res.fidelity["grade"] >= "C"


# ====================================================================== S042 declaration timestamps
def test_s042_dividend_without_declaration_time_is_not_testable():
    from atlasbt.core.types import Fidelity
    w = sess("2024-03-13", "2024-03-15")
    known = CAL.open_close(pd.Timestamp("2024-03-14"))[1]          # project convention: prior-session close
    a = CorporateAction(SYM, ActionType.CASH_DIVIDEND, pd.Timestamp("2024-03-15"), known, amount=1.5,
                        pay_date=pd.Timestamp("2024-04-30"), fidelity=Fidelity.T2_PROXY)
    days = flat_days(w, 100.0)
    res = run_backtest(small(CE.S042)(), build(w, days, "flat", actions=[a])(), w[0], w[-1], zero_cost(debug=False))
    assert res.status == RunStatus.NOT_TESTABLE and "declaration" in res.not_testable_reason.lower()


def test_s042_decl_proxy_requires_wrapper_and_trades_with_it():
    from atlasbt.core.types import Fidelity
    w = sess("2024-03-13", "2024-03-15")
    known = CAL.open_close(pd.Timestamp("2024-03-14"))[1]
    a = CorporateAction(SYM, ActionType.CASH_DIVIDEND, pd.Timestamp("2024-03-15"), known, amount=1.5,
                        pay_date=pd.Timestamp("2024-04-30"), fidelity=Fidelity.T2_PROXY)
    days = flat_days(w, 100.0, **{"2024-03-14": {"15:50": bar(100.2)}, "2024-03-15": {"09:35": bar(98.7)}})
    base = build(w, days, "flat", actions=[a])
    res0 = run_backtest(small(CE.S042)(decl_proxy=True), base(), w[0], w[-1], zero_cost())
    assert res0.status == RunStatus.NOT_TESTABLE                      # no wrapper -> never silently proxied
    portal = CE.DividendDeclarationLeadPortal(base(), lead_sessions=10)
    res = run(small(CE.S042)(decl_proxy=True, decl_lead=10), portal, w)
    _check_trade(res, dict(entry=(et_ts("2024-03-14", "15:50"), 100.2, 1000),
                           exit=(et_ts("2024-03-15", "09:35"), 98.7)))
    assert any("declaration" in r["reason"] for r in res.fidelity["reasons"])
    # the wrapper re-stamps known_at 10 sessions before the ex-date, never later than the original
    acts = portal.corporate_actions([SYM], pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-31"),
                                    pd.Timestamp("2024-03-02", tz="UTC"))
    assert len(acts) == 1 and acts[0].known_at == CAL.open_close(CAL.prev_session(pd.Timestamp("2024-03-15"), 10))[1]
    assert portal.corporate_actions([SYM], pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-31"),
                                    pd.Timestamp("2024-02-27", tz="UTC")) == []
