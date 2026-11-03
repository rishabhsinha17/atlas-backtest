"""S042 with point-in-time dividend DECLARATION events (events kind DIVIDEND_DECL, providers/dividends_decl.py).

SYNTHETIC data, mechanics only: a declared dividend trades (15:50 the session before the ex-date, out at
09:35 on the ex-date) and is flagged T2; a dividend with only a proxied (T2) action and no declaration
record is skipped -- not NOT_TESTABLE -- when the portal has a DIVIDEND_DECL source; a declaration made
after the 15:50 decision is never used (PIT).
"""
import pandas as pd

from atlasbt.core.types import ActionType, CorporateAction, Fidelity, MarketEvent, RunStatus
from atlasbt.engine.backtester import run_backtest
from atlasbt.engine.testing import InMemoryPortal
from atlasbt.strategies import calendar_events as CE

from calalloc_helpers import CAL, combine_daily, daily_frame, et_ts, intraday_frame, orders, sess, sessions_before, \
    zero_cost
import numpy as np

SYM = "SPY"
EX = pd.Timestamp("2024-03-15")


def small():
    return type("S042T", (CE.S042,), {"universe": [SYM]})


def _portal(events, actions):
    w = sess("2024-03-13", "2024-03-15")
    hs = sessions_before(w[0], 260)
    days = {s: (100.0, {}) for s in w}
    days[pd.Timestamp("2024-03-14")] = (100.0, {"15:50": (100.2,) * 4})
    days[pd.Timestamp("2024-03-15")] = (100.0, {"09:35": (98.7,) * 4})
    intr = intraday_frame(days)
    daily = combine_daily(daily_frame(hs, np.full(260, 100.0)), intr)
    return InMemoryPortal(bars={(SYM, "1m"): intr, (SYM, "1d"): daily}, actions=actions, events=events), w


def _t2_action():
    return CorporateAction(SYM, ActionType.CASH_DIVIDEND, EX, CAL.open_close(pd.Timestamp("2024-03-14"))[1],
                           amount=1.5, pay_date=pd.Timestamp("2024-04-30"), fidelity=Fidelity.T2_PROXY)


def _decl(sym=SYM, known="2024-03-01 16:31"):
    return MarketEvent(kind="DIVIDEND_DECL", symbol=sym,
                       known_at=pd.Timestamp(known, tz="America/New_York").tz_convert("UTC"),
                       effective_at=pd.Timestamp("2024-03-15 09:30", tz="America/New_York").tz_convert("UTC"),
                       payload={"ex_date": EX, "amount": 1.5, "accn": "0000000000-24-000001"},
                       fidelity=Fidelity.T2_PROXY)


def _run(events, actions):
    p, w = _portal(events, actions)
    return run_backtest(small()(), p, w[0], w[-1], zero_cost(), raise_errors=True)


def _reasons(res):
    return " | ".join(r["reason"] for r in res.fidelity.get("reasons", []))


def test_declared_dividend_trades_and_is_t2():
    res = _run([_decl()], [_t2_action()])
    assert res.status == RunStatus.OK
    o = orders(res)
    ent, ex = o[o["tag"] == "entry"], o[o["tag"] == "exit"]
    assert len(ent) == 1 and pd.Timestamp(ent["submitted_at"].iloc[0]) == et_ts("2024-03-14", "15:50")
    assert pd.Timestamp(ex["submitted_at"].iloc[0]) == et_ts("2024-03-15", "09:35")
    assert "declaration time = SEC acceptance" in _reasons(res)


def test_undeclared_dividend_skipped_with_decl_source():
    # the portal has a DIVIDEND_DECL source (another issuer), SPY's dividend has only a proxied action
    res = _run([_decl(sym="OTHER")], [_t2_action()])
    assert res.status == RunStatus.OK
    assert (orders(res)["tag"] == "entry").sum() == 0 if len(orders(res)) else True
    assert "without a point-in-time declaration record" in _reasons(res)


def test_undeclared_dividend_without_decl_source_still_not_testable():
    p, w = _portal([], [_t2_action()])
    res = run_backtest(small()(), p, w[0], w[-1], zero_cost(debug=False))
    assert res.status == RunStatus.NOT_TESTABLE and "declaration" in res.not_testable_reason.lower()


def test_declaration_after_the_decision_is_not_used():
    # declared 17:00 on the session before the ex-date: after the 15:50 entry decision
    res = _run([_decl(known="2024-03-14 17:00")], [_t2_action()])
    assert res.status == RunStatus.OK
    o = orders(res)
    assert len(o) == 0 or (o["tag"] == "entry").sum() == 0
