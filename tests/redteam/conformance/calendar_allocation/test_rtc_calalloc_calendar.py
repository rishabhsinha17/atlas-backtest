"""Red-team conformance probes: S031-S037, S042 (overnight / calendar arithmetic).

SYNTHETIC data, mechanics only. Each test constructs the exact sheet scenario and asserts
signal bar, fill bar, fill price and exit timing.
"""
import math

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.types import ActionType, CorporateAction, Fidelity, RunStatus
from atlasbt.engine.testing import InMemoryPortal
from atlasbt.strategies import calendar_events as CE

from rtc_calalloc_helpers import (CAL, combine_daily, daily_frame, et_ts, fills, intraday_frame, orders, run,
                                  sess, sessions_before)

SYM = "SPY"


def small(cls, **attrs):
    return type(cls.__name__ + "RT", (cls,), {"universe": [SYM], **attrs})


def up_hist(n=260, last=100.0):
    t = np.arange(n)
    c = np.exp(0.002 * (t - n + 1)) * np.where(t % 2 == 0, 1.0, 1.01)
    return c / c[-1] * last


def portal(window, day_specs, hist_closes, actions=(), daily_only=False):
    hs = sessions_before(window[0], len(hist_closes))
    h = daily_frame(hs, hist_closes)
    intr = intraday_frame(day_specs)
    daily = combine_daily(h, intr)
    bars = {(SYM, "1d"): daily} if daily_only else {(SYM, "1m"): intr, (SYM, "1d"): daily}
    return InMemoryPortal(bars=bars, actions=list(actions))


def flat(window, base=100.0, ov=None):
    ov = ov or {}
    return {s: (base, ov.get(s.strftime("%Y-%m-%d"), {})) for s in window}


def bar(p):
    return (p, p, p, p)


def tagged(res, tag):
    o = orders(res)
    return o[o["tag"] == tag] if len(o) else o


def fill_ts_px(res, order_row):
    f = fills(res)
    r = f[f["order_id"] == order_row["order_id"]].iloc[0]
    return pd.Timestamp(r["ts"]), float(r["price"])


# ============================================================ S031 trailing window
def _s031_hist(c_m11, c_m10_to_m2, c_m1, c_m12=105.0):
    h = list(np.full(250, 100.0)) + [c_m12, c_m11] + [c_m10_to_m2] * 9 + [c_m1]
    return np.array(h, float)


def test_s031_uses_exactly_10_completed_sessions_trigger():
    w = sess("2024-03-04", "2024-03-05")
    p = portal(w, flat(w, 100.0, {"2024-03-04": {"15:50": bar(100.5)}, "2024-03-05": {"09:35": bar(101.0)}}),
               _s031_hist(100.0, 99.0, 100.01))          # 10-session ret > 0, 11-session ret < 0
    res = run(small(CE.S031)(), p, w[0], w[-1])
    e = tagged(res, "entry")
    e = e[pd.DatetimeIndex(e["submitted_at"]) < et_ts("2024-03-05", "09:30")]
    assert len(e) == 1
    assert pd.Timestamp(e.iloc[0]["submitted_at"]) == et_ts("2024-03-04", "15:50")
    assert fill_ts_px(res, e.iloc[0]) == (et_ts("2024-03-04", "15:50"), 100.5)


def test_s031_nearmiss_9_session_window_positive_10_session_negative():
    w = sess("2024-03-04", "2024-03-05")
    p = portal(w, flat(w), _s031_hist(100.02, 99.0, 100.01, c_m12=90.0))
    res = run(small(CE.S031)(), p, w[0], w[0])
    assert len(tagged(res, "entry")) == 0


def test_s031_exit_0935_next_session_after_good_friday():
    # entry Thu 2024-03-28 15:50 -> Good Friday closed -> exit Mon 2024-04-01 09:35
    w = sess("2024-03-28", "2024-04-01")
    p = portal(w, flat(w, 100.0, {"2024-03-28": {"15:50": bar(100.5)}, "2024-04-01": {"09:35": bar(102.0)}}),
               up_hist())
    res = run(small(CE.S031)(), p, w[0], w[-1])
    x = tagged(res, "exit").iloc[0]
    assert fill_ts_px(res, x) == (et_ts("2024-04-01", "09:35"), 102.0)


def test_s031_early_close_decision_1245_entry_1250():
    # 2024-11-29 closes 13:00: 15:45/15:50 shift to 12:45/12:50; a drop after 12:50 is irrelevant
    w = sess("2024-11-29", "2024-12-02")
    p = portal(w, flat(w, 100.0, {"2024-11-29": {"12:50": bar(100.4)}, "2024-12-02": {"09:35": bar(101.0)}}),
               up_hist())
    res = run(small(CE.S031)(), p, w[0], w[-1])
    e = tagged(res, "entry").iloc[0]
    assert fill_ts_px(res, e) == (et_ts("2024-11-29", "12:50"), 100.4)


# ============================================================ S032 decision bar
def _alt_hist():
    t = np.arange(260)
    return 100.0 * np.where(t % 2 == 0, 1.0, 1.01)


def test_s032_decision_uses_bar_ending_1545_only():
    """Drop in the bar 15:44-15:45 that fully recovers by 15:49: the 15:45 decision triggers."""
    w = sess("2024-03-04", "2024-03-05")
    ov = {"15:44": (100.0, 100.0, 97.0, 97.0), "15:45": bar(100.0), "15:50": bar(100.2)}
    p = portal(w, flat(w, 100.0, {"2024-03-04": ov, "2024-03-05": {"09:35": bar(101.0)}}), _alt_hist())
    res = run(small(CE.S032)(), p, w[0], w[-1])
    e = tagged(res, "entry")
    assert len(e) == 1 and fill_ts_px(res, e.iloc[0]) == (et_ts("2024-03-04", "15:50"), 100.2)


def test_s032_nearmiss_drop_after_1545_ignored():
    """Drop only in 15:46-15:49 (after the 15:45 decision) must NOT trigger the baseline."""
    w = sess("2024-03-04", "2024-03-05")
    ov = {"15:46": (100.0, 100.0, 97.0, 97.0), "15:47": bar(97.0), "15:48": bar(97.0), "15:49": bar(97.0)}
    p = portal(w, flat(w, 100.0, {"2024-03-04": ov}), _alt_hist())
    res = run(small(CE.S032)(), p, w[0], w[-1])
    assert len(tagged(res, "entry")) == 0


# ============================================================ S034 turn of month
@pytest.mark.parametrize("t_minus_1,exit_day,w0,w1", [
    ("2024-08-29", "2024-09-06", "2024-08-29", "2024-09-06"),     # Labor Day inside the window
    ("2024-11-27", "2024-12-05", "2024-11-27", "2024-12-05"),     # T0 = 11-29 early close, 11-28 holiday
])
def test_s034_t_minus_1_entry_and_fourth_trading_day_exit(t_minus_1, exit_day, w0, w1):
    w = sess(w0, w1)
    p = portal(w, flat(w, 100.0, {t_minus_1: {"15:50": bar(100.5)}, exit_day: {"09:35": bar(103.0)}}),
               np.full(260, 100.0))
    res = run(small(CE.S034)(), p, w[0], w[-1])
    e, x = tagged(res, "entry"), tagged(res, "exit")
    assert len(e) == 1 and len(x) == 1
    assert fill_ts_px(res, e.iloc[0]) == (et_ts(t_minus_1, "15:50"), 100.5)
    assert fill_ts_px(res, x.iloc[0]) == (et_ts(exit_day, "09:35"), 103.0)
    # fourth trading day of the new month
    nm = pd.Timestamp(exit_day)
    assert list(CAL.sessions_in_month(nm)).index(nm) == 3


# ============================================================ S035 early-close month end
def test_s035_month_end_on_early_close_day():
    w = sess("2024-11-29", "2024-12-02")
    p = portal(w, flat(w, 100.0, {"2024-11-29": {"12:50": bar(100.5)}, "2024-12-02": {"15:50": bar(99.0)}}),
               np.full(260, 100.0))
    res = run(small(CE.S035)(), p, w[0], w[-1])
    e, x = tagged(res, "entry").iloc[0], tagged(res, "exit").iloc[0]
    assert fill_ts_px(res, e) == (et_ts("2024-11-29", "12:50"), 100.5)
    assert fill_ts_px(res, x) == (et_ts("2024-12-02", "15:50"), 99.0)


# ============================================================ S036 week with Good Friday
def test_s036_exit_on_final_trading_day_of_week_good_friday():
    w = sess("2024-03-25", "2024-03-28")
    p = portal(w, flat(w, 100.0, {"2024-03-25": {"09:35": bar(100.5)}, "2024-03-28": {"15:50": bar(102.0)}}),
               up_hist())
    res = run(small(CE.S036)(), p, w[0], w[-1])
    e, x = tagged(res, "entry").iloc[0], tagged(res, "exit").iloc[0]
    assert fill_ts_px(res, e) == (et_ts("2024-03-25", "09:35"), 100.5)
    assert fill_ts_px(res, x) == (et_ts("2024-03-28", "15:50"), 102.0)


# ============================================================ S037 pre-holiday
@pytest.mark.parametrize("p2,p1", [
    ("2024-01-11", "2024-01-12"),   # MLK Monday 2024-01-15 -> P1 = Friday
    ("2024-03-27", "2024-03-28"),   # Good Friday 2024-03-29
    ("2024-06-17", "2024-06-18"),   # Juneteenth Wednesday 2024-06-19
    ("2021-12-22", "2021-12-23"),   # Christmas observed Friday 2021-12-24 (Saturday holiday)
])
def test_s037_entry_two_sessions_before_full_holiday(p2, p1):
    w = sess(p2, p1)
    p = portal(w, flat(w, 100.0, {p2: {"15:50": bar(100.5)}, p1: {"15:50": bar(101.5)}}), np.full(260, 100.0))
    res = run(small(CE.S037)(), p, w[0], w[-1])
    e, x = tagged(res, "entry"), tagged(res, "exit")
    assert len(e) == 1 and len(x) == 1
    assert fill_ts_px(res, e.iloc[0]) == (et_ts(p2, "15:50"), 100.5)
    assert fill_ts_px(res, x.iloc[0]) == (et_ts(p1, "15:50"), 101.5)


def test_s037_nearmiss_early_close_day_is_not_a_holiday():
    # 2024-12-24 is a 13:00 early close, not a closure: 12-20/12-23 are not "two sessions before a holiday"
    # (the real holiday is 12-25 -> P1 = 12-24, P2 = 12-23)
    w = sess("2024-12-19", "2024-12-20")
    p = portal(w, flat(w), np.full(260, 100.0))
    res = run(small(CE.S037)(), p, w[0], w[-1])
    assert len(tagged(res, "entry")) == 0


# ============================================================ S042 dividend timing
def _div(known_utc, ex="2024-03-15", amount=1.5, fidelity=Fidelity.T1_BAR):
    return CorporateAction(SYM, ActionType.CASH_DIVIDEND, pd.Timestamp(ex), known_utc, amount=amount,
                           pay_date=pd.Timestamp("2024-04-30"), fidelity=fidelity)


def test_s042_entitlement_and_ex_price_drop_in_trade_pnl():
    w = sess("2024-03-14", "2024-03-15")
    known = pd.Timestamp("2024-03-01 17:00", tz="UTC")
    p = portal(w, {pd.Timestamp("2024-03-14"): (100.0, {}), pd.Timestamp("2024-03-15"): (98.5, {})},
               np.full(260, 100.0), actions=[_div(known)])
    res = run(small(CE.S042)(), p, w[0], w[-1])
    e, x = tagged(res, "entry").iloc[0], tagged(res, "exit").iloc[0]
    assert fill_ts_px(res, e)[0] == et_ts("2024-03-14", "15:50")
    assert fill_ts_px(res, x)[0] == et_ts("2024-03-15", "09:35")
    t = res.trades.iloc[0]
    q = float(t["quantity"])
    # price leg (100 -> 98.5) is offset exactly by the 1.5 dividend entitlement
    assert float(t["pnl"]) == pytest.approx(0.0, abs=1e-6 * q)
    led = res.ledger
    assert (led["reason"] == "DIVIDEND").any() or float(t["pnl"]) == pytest.approx(0.0, abs=1e-6 * q)


def test_s042_trades_with_project_dividend_known_at_convention():
    """FINDING: docs/DATA.md §'Corporate actions': Yahoo/DoltHub dividends carry NO declaration date;
    known_at = close (16:00) of the session BEFORE the ex-date (T2). The sheet's 15:50 buy on that very
    session can never see such a dividend, so on project data S042 never trades and still returns OK
    (silent no-trade). Expected: either trade, or NOT_TESTABLE because PIT declaration timestamps are
    unavailable."""
    w = sess("2024-03-14", "2024-03-15")
    known = CAL.open_close(pd.Timestamp("2024-03-14"))[1]          # 16:00 ET of the session before ex-date
    p = portal(w, flat(w), np.full(260, 100.0), actions=[_div(known, fidelity=Fidelity.T2_PROXY)])
    res = run(small(CE.S042)(), p, w[0], w[-1])
    traded = len(tagged(res, "entry")) > 0
    assert traded or res.status == RunStatus.NOT_TESTABLE, \
        "S042 silently never trades when dividend known_at = prior-session close (project data convention)"
