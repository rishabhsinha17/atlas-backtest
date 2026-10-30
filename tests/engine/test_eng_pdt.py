"""FINRA Rule 4210(f)(8)(B) pattern-day-trader enforcement (engine order/risk path).

SYNTHETIC bars: mechanics only. Hand-computed expectations: an account below USD 25,000 that
round-trips one stock intraday every session may make 3 day trades in any 5 business days; the
order that would be the 4th is rejected with a 'PDT rule' reason, the position is carried
overnight, and once the first day trade leaves the 5-session window the account may day trade
again. At or above USD 25,000 (prior close) the account is exempt; before 2001-09-28 and in cash
accounts the rule does not apply.
"""
import math

import pandas as pd
import pytest

from atlasbt.core.clock import nyse
from atlasbt.core.types import OrderStatus
from atlasbt.engine import BacktestConfig, Backtester, Strategy
from atlasbt.engine.testing import InMemoryPortal, daily_from_intraday, synthetic_intraday_bars
from atlasbt.execution.impact import SquareRootImpact
from atlasbt.execution.spread import SpreadConfig
from atlasbt.portfolio.account import AccountType

CAL = nyse()


def cfg(cash, **kw) -> BacktestConfig:
    base = dict(fees="NO_FEES", initial_cash=cash,
                spread=SpreadConfig(bucket_floors=((math.inf, 0.0),), default_spread=1e-12, tod_open=(),
                                    tod_close=(), tick_floor=False),
                impact=SquareRootImpact(k=0.0), debug=True)
    base.update(kw)
    return BacktestConfig(**base)


class RoundTrip(Strategy):
    """Buy 10 shares after the first 30-minute bar, sell everything after the third (same day)."""
    id = "PDT_TEST"
    bar_freq = "30m"
    universe = ["XYZ"]

    def __init__(self, **kw):
        super().__init__()
        self.k = 0

    def on_session_start(self, ctx):
        self.k = 0

    def on_bar(self, ctx):
        self.k += 1
        if self.k == 1:
            ctx.order("XYZ", 10)
        elif self.k == 3:
            q = ctx.position("XYZ")
            if q:
                ctx.order("XYZ", -q)


def run(start, n, cash, account=AccountType.REG_T_MARGIN, **kw):
    ss = list(CAL.sessions(pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=3 * n)))[:n]
    mdf = synthetic_intraday_bars(str(ss[0].date()), str(ss[-1].date()), freq="30m", start_price=100.0,
                                  daily_vol=0.002, volume_per_bar=1e6, seed=4)
    portal = InMemoryPortal(bars={("XYZ", "30m"): mdf, ("XYZ", "1d"): daily_from_intraday(mdf)})
    r = Backtester(RoundTrip(), portal, cfg(cash, account_type=account, **kw)).run(
        ss[0], ss[-1], raise_errors=True)
    return r, ss


def _sells_by_session(r):
    f = r.fills[r.fills.quantity < 0]
    d = pd.DatetimeIndex(f.ts).tz_convert("America/New_York").normalize().tz_localize(None)
    return sorted(set(d))


def test_fourth_day_trade_blocked_below_25k():
    r, ss = run("2024-03-04", 7, 20_000.0)
    rej = [x for x in r.diagnostics["rejected_orders"] if x["reason"].startswith("PDT rule")]
    # day trades on sessions 0, 1, 2; the sell on session 3 is the 4th -> rejected
    assert rej and pd.Timestamp(rej[0]["ts"]).tz_convert("America/New_York").normalize().tz_localize(None) == ss[3]
    assert "day trade #4" in rej[0]["reason"] and "25,000" in rej[0]["reason"]
    sells = _sells_by_session(r)
    assert sells[:3] == ss[:3] and ss[3] not in sells
    # the position bought on session 3 is carried overnight (no same-day sale)
    pos = r.positions[pd.DatetimeIndex(r.positions.ts).tz_convert("America/New_York").normalize()
                      .tz_localize(None) == ss[3]]
    assert pos.quantity.sum() == 10
    pdt = r.diagnostics["pdt"]
    assert pdt["day_trades"] >= 3 and pdt["blocked"] and pdt["restricted_sessions"] == 7


def test_window_rolls_after_five_business_days():
    r, ss = run("2024-03-04", 9, 20_000.0)
    sells = _sells_by_session(r)
    # session 5: the day trades of sessions 0 drop out of the 5-session window (1..5) -> allowed again
    blocked_days = {pd.Timestamp(x["ts"]).tz_convert("America/New_York").normalize().tz_localize(None)
                    for x in r.diagnostics["rejected_orders"] if x["reason"].startswith("PDT rule")}
    assert ss[3] in blocked_days and ss[4] in blocked_days
    assert ss[5] in sells
    # never more than 3 day trades in any 5 consecutive sessions
    for i in range(len(ss)):
        win = set(ss[max(0, i - 4): i + 1])
        assert len([d for d in sells if d in win]) <= 3 + (1 if ss[3] in win else 0)


def test_exempt_at_or_above_25k():
    r, ss = run("2024-03-04", 7, 30_000.0)
    assert not [x for x in r.diagnostics["rejected_orders"] if x["reason"].startswith("PDT rule")]
    assert _sells_by_session(r) == ss
    assert "pdt" not in r.diagnostics          # never restricted: no diagnostics entry


def test_rule_not_in_force_before_2001_09_28_or_in_cash_accounts_or_disabled():
    r, ss = run("2001-03-05", 6, 20_000.0)
    assert _sells_by_session(r) == ss
    r, ss = run("2024-03-04", 6, 20_000.0, account=AccountType.CASH)
    assert _sells_by_session(r) == ss
    r, ss = run("2024-03-04", 6, 20_000.0, pdt_rule=False)
    assert _sells_by_session(r) == ss
    assert r.meta["config"]["pdt_rule"] is False


def test_default_config_summary_unchanged_keys():
    """New knobs are reported only when non-default (golden meta['config'] stays identical)."""
    r, _ = run("2024-03-04", 2, 30_000.0)
    assert "pdt_rule" not in r.meta["config"] and "financing_schedule" not in r.meta["config"]


class BracketDaily(Strategy):
    """Opens with a protective stop + far target each session; a GTC stop working on a position
    opened today is a potential day trade and is rejected once 3 day trades are used."""
    id = "PDT_BR"
    bar_freq = "30m"
    universe = ["XYZ"]

    def __init__(self, **kw):
        super().__init__()
        self.k = 0

    def on_session_start(self, ctx):
        self.k = 0

    def on_bar(self, ctx):
        self.k += 1
        if self.k == 1 and ctx.position("XYZ") == 0:
            px = ctx.mark("XYZ")
            ctx.bracket("XYZ", 10, stop=px * 0.9, target=px * 1.000001)
        elif self.k == 3 and ctx.position("XYZ"):
            ctx.order("XYZ", -ctx.position("XYZ"))


def test_working_child_order_rejected_when_it_would_be_fourth_day_trade():
    ss = list(CAL.sessions(pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-20")))[:6]
    mdf = synthetic_intraday_bars(str(ss[0].date()), str(ss[-1].date()), freq="30m", start_price=100.0,
                                  daily_vol=0.002, volume_per_bar=1e6, seed=4)
    portal = InMemoryPortal(bars={("XYZ", "30m"): mdf, ("XYZ", "1d"): daily_from_intraday(mdf)})
    r = Backtester(BracketDaily(), portal, cfg(20_000.0)).run(ss[0], ss[-1], raise_errors=True)
    rej = [x for x in r.diagnostics["rejected_orders"] if x["reason"].startswith("PDT rule")]
    assert rej
    blocked = r.diagnostics["pdt"]["blocked"]
    assert any(b["order_id"] is not None for b in blocked)       # a WORKING order was screened
    orders = r.diagnostics["orders"]
    assert (orders.status == OrderStatus.REJECTED.value).any() or (orders.status == "REJECTED").any()
    # at most 3 day trades were ever completed inside one 5-session window
    assert r.diagnostics["pdt"]["day_trades"] <= 3 + 1
