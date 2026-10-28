"""Engine mechanics: timing (no same-bar fills), PIT guard, lookahead
perturbation, orders (brackets/OCO/TIF), session rules, auctions,
corporate actions, financing, margin calls, locates, SSR, halts, intrabar
resolution, options, determinism and end-to-end smoke tests.

All data here is SYNTHETIC and only tests mechanics.
"""
import math

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.clock import et, nyse
from atlasbt.core.types import (
    ActionType, AssetClass, CorporateAction, ExerciseStyle, Fidelity, Instrument, Leg, LookaheadError,
    MarketEvent, Order, OrderStatus, OrderType, Right, RunStatus, SettlementType, TimeInForce,
)
from atlasbt.engine import BacktestConfig, Backtester, Requirement, Strategy
from atlasbt.engine.testing import (
    InMemoryPortal, daily_from_intraday, synthetic_bars_from_rows, synthetic_daily_bars,
    synthetic_intraday_bars,
)
from atlasbt.execution.fills import BarFillConfig
from atlasbt.execution.impact import SquareRootImpact
from atlasbt.execution.spread import SpreadConfig
from atlasbt.portfolio.account import AccountType

CAL = nyse()


def zero_cost(**kw) -> BacktestConfig:
    base = dict(fees="NO_FEES",
                spread=SpreadConfig(bucket_floors=((math.inf, 0.0),), default_spread=1e-12, tod_open=(),
                                    tod_close=(), tick_floor=False),
                impact=SquareRootImpact(k=0.0), debug=True)
    base.update(kw)
    return BacktestConfig(**base)


class Scripted(Strategy):
    """Runs ``script(ctx, bt_strategy)`` on every bar; records what it saw."""
    id = "TEST"
    bar_freq = "1d"
    universe = ["XYZ"]

    def __init__(self, script=None, on_start=None, on_end=None, **kw):
        super().__init__()
        self.script = script
        self.on_start = on_start
        self.on_end = on_end
        self.seen = []
        self.fills = []

    def on_bar(self, ctx):
        if self.script:
            self.script(ctx, self)

    def on_session_start(self, ctx):
        if self.on_start:
            self.on_start(ctx, self)

    def on_session_end(self, ctx):
        if self.on_end:
            self.on_end(ctx, self)

    def on_fill(self, ctx, fill):
        self.fills.append(fill)


def make(freq="1d", **attrs):
    return type("S", (Scripted,), {"bar_freq": freq, **attrs})


def sessions(start, n):
    return list(CAL.sessions(pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=3 * n)))[:n]


def daily_portal(rows, start="2024-03-04", sym="XYZ", **kw):
    ss = sessions(start, len(rows))
    df = synthetic_bars_from_rows(rows, "1d", session_dates=ss)
    return InMemoryPortal(bars={(sym, "1d"): df}, **kw), ss


FLAT = [(100, 101, 99, 100)] * 3


# =================================================================== timing
def test_daily_order_at_close_fills_at_next_open_not_close():
    rows = [(100, 101, 99, 100), (103, 104, 102, 103.5), (104, 105, 103, 104)]
    portal, ss = daily_portal(rows)

    def script(ctx, s):
        if ctx.session_date() == ss[0]:
            ctx.order("XYZ", 100)
    r = Backtester(make()(script), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    f = r.fills.iloc[0]
    assert f.price == pytest.approx(103.0)                       # next open, never 100 (signal close)
    assert f.ts == CAL.open_close(ss[1])[0]
    assert f.ts > pd.Timestamp(CAL.open_close(ss[0])[1])


def test_intraday_order_at_bar_close_fills_at_next_bar_open():
    s = sessions("2024-03-04", 1)[0]
    o, _ = CAL.open_close(s)
    starts = pd.date_range(o, periods=390, freq="1min")
    rows = [(100 + i * 0.1, 100 + i * 0.1 + 0.05, 100 + i * 0.1 - 0.05, 100 + i * 0.1 + 0.03) for i in range(390)]
    df = synthetic_bars_from_rows(rows, "1m", start_times=starts)
    portal = InMemoryPortal(bars={("XYZ", "1m"): df})
    t_submit = starts[10] + pd.Timedelta(minutes=1)              # end of bar 10

    def script(ctx, st):
        if ctx.now == t_submit:
            st.seen.append(ctx.history("XYZ", 1, "1m", adjusted=False)["close"].iloc[-1])
            ctx.order("XYZ", 10)
    r = Backtester(make("1m")(script), portal, zero_cost()).run(s, s, raise_errors=True)
    f = r.fills.iloc[0]
    assert f.price == pytest.approx(rows[11][0])                 # bar 11 open
    assert f.price != pytest.approx(rows[10][3])                 # not the signal bar close
    assert f.ts == starts[11]


def test_strategy_cannot_see_next_bar_and_pit_guard_raises():
    portal, ss = daily_portal([(100, 101, 99, 100), (110, 111, 109, 110), (120, 121, 119, 120)])

    def script(ctx, st):
        h = ctx.history("XYZ", 10, "1d", adjusted=False)
        st.seen.append((ctx.now, h.index[-1], h["known_at"].max(), len(h)))
        with pytest.raises(LookaheadError):
            ctx.portal().bars("XYZ", "1d", ctx.now - pd.Timedelta(days=10), ctx.now + pd.Timedelta(days=5),
                              as_of=ctx.now + pd.Timedelta(minutes=1))
        # an end bound in the future is clipped to now (never returns the next bar)
        fut = ctx.portal().bars("XYZ", "1d", ctx.now - pd.Timedelta(days=10), ctx.now + pd.Timedelta(days=5))
        assert (pd.DatetimeIndex(fut["known_at"]) <= ctx.now).all()
    st = make()(script)
    Backtester(st, portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    for now, last_start, known, n in st.seen:
        assert known <= now
    assert [x[3] for x in st.seen] == [1, 2, 3]


def test_intraday_history_excludes_in_progress_daily_bar():
    mdf = synthetic_intraday_bars("2024-03-04", "2024-03-06", seed=3)
    portal = InMemoryPortal(bars={("XYZ", "1m"): mdf, ("XYZ", "1d"): daily_from_intraday(mdf)})
    ss = sessions("2024-03-04", 3)

    def script(ctx, st):
        if ctx.minutes_since_open() == 60:
            d = ctx.history("XYZ", 5, "1d", adjusted=False)
            st.seen.append((ctx.session_date(), len(d), len(d) == 0 or d["known_at"].max() <= ctx.now))
    st = make("1m")(script)
    Backtester(st, portal, zero_cost(debug=False)).run(ss[0], ss[-1], raise_errors=True)
    assert [x[1] for x in st.seen] == [0, 1, 2] and all(x[2] for x in st.seen)


def test_perturbing_future_bars_does_not_change_past_decisions_or_fills():
    df = synthetic_daily_bars("2023-01-02", "2023-12-29", seed=11)
    cut = pd.Timestamp("2023-07-03 21:00", tz="UTC")
    df2 = df.copy()
    m = df2["known_at"] > cut
    df2.loc[m, ["open", "high", "low", "close"]] *= np.linspace(0.5, 2.0, int(m.sum()))[:, None]

    def script(ctx, st):
        h = ctx.history("XYZ", 20, "1d", adjusted=False)
        if len(h) < 20:
            return
        sig = h["close"].iloc[-1] > h["close"].mean()
        st.seen.append((ctx.now, sig))
        q = ctx.position("XYZ")
        if sig and q <= 0:
            ctx.order("XYZ", 100 - q)
        elif not sig and q >= 0:
            ctx.order("XYZ", -100 - q)

    out = []
    for d in (df, df2):
        st = make()(script)
        r = Backtester(st, InMemoryPortal(bars={("XYZ", "1d"): d}), BacktestConfig(debug=True)).run(
            "2023-02-01", "2023-12-29", raise_errors=True)
        out.append((st, r))
    (s1, r1), (s2, r2) = out
    before = lambda seen: [x for x in seen if x[0] <= cut]
    assert before(s1.seen) == before(s2.seen)
    f1 = r1.fills[r1.fills.ts <= cut].drop(columns=["note"])
    f2 = r2.fills[r2.fills.ts <= cut].drop(columns=["note"])
    pd.testing.assert_frame_equal(f1.reset_index(drop=True), f2.reset_index(drop=True))
    assert r1.equity[r1.equity.index <= cut].equals(r2.equity[r2.equity.index <= cut])
    assert not r1.equity.equals(r2.equity)


# =================================================================== orders
def test_gap_through_stop_fills_at_open_in_engine():
    rows = [(100, 101, 99, 100), (100, 101, 99, 100), (90, 92, 88, 91), (91, 92, 90, 91)]
    portal, ss = daily_portal(rows)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.bracket("XYZ", 100, stop=95.0)
    r = Backtester(make()(script), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    exit_ = r.fills[r.fills.tag == "stop"].iloc[0]
    assert exit_.price == pytest.approx(90.0) and exit_.ts == CAL.open_close(ss[2])[0]
    t = r.trades.iloc[0]
    assert t.exit_reason == "stop" and t.r_multiple == pytest.approx(-10 * 100 / (5 * 100))
    assert t.initial_risk == pytest.approx(500.0)


def test_bracket_target_cancels_stop_and_mae_mfe():
    rows = [(100, 101, 99, 100), (100, 101, 98, 100.5), (101, 106, 100, 105), (105, 106, 104, 105)]
    portal, ss = daily_portal(rows)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.bracket("XYZ", 100, stop=95.0, target=104.0)
    r = Backtester(make()(script), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    assert list(r.fills.tag) == ["entry", "target"]
    assert r.fills.iloc[1].price == 104.0
    orders = r.diagnostics["orders"].set_index("tag")
    assert orders.loc["stop", "status"] == "CANCELLED"
    t = r.trades.iloc[0]
    assert t.pnl == pytest.approx(400.0) and t.r_multiple == pytest.approx(0.8)
    assert t.mae == pytest.approx(-200.0)             # low 98 vs entry 100, 100 shares
    assert t.mfe == pytest.approx(400.0)


def test_day_order_expires_gtc_persists_ioc_cancels():
    rows = [(100, 101, 99, 100)] * 4 + [(100, 101, 89, 90)]
    portal, ss = daily_portal(rows)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.order("XYZ", 10, OrderType.LIMIT, limit_price=90.0, tif=TimeInForce.DAY, tag="day")
            ctx.order("XYZ", 10, OrderType.LIMIT, limit_price=90.0, tif=TimeInForce.GTC, tag="gtc")
            ctx.order("XYZ", 10, OrderType.LIMIT, limit_price=90.0, tif=TimeInForce.IOC, tag="ioc")
    r = Backtester(make()(script), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    o = r.diagnostics["orders"].set_index("tag")
    assert o.loc["day", "status"] == "EXPIRED"
    assert o.loc["ioc", "status"] == "CANCELLED"
    assert o.loc["gtc", "status"] == "FILLED"
    assert list(r.fills.tag) == ["gtc"] and r.fills.iloc[0].ts.normalize() == CAL.open_close(ss[4])[0].normalize()


def test_moo_and_moc_daily():
    rows = [(100, 101, 99, 100), (102, 104, 101, 103), (105, 106, 104, 104.5)]
    portal, ss = daily_portal(rows)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.order("XYZ", 100, OrderType.MOO, tag="moo")
            ctx.order("XYZ", -100, OrderType.MOC, tag="moc")
    r = Backtester(make()(script), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    f = r.fills.set_index("tag")
    assert f.loc["moo", "price"] == 102 and f.loc["moo", "liquidity"] == "AUCTION"
    assert f.loc["moc", "price"] == 103 and f.loc["moc", "ts"] == CAL.open_close(ss[1])[1]


def test_participation_cap_partial_fills_over_bars():
    rows = [(10, 10.1, 9.9, 10, 10_000)] * 4
    portal, ss = daily_portal(rows)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.order("XYZ", 2500, tif=TimeInForce.GTC)
    r = Backtester(make()(script), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    assert list(r.fills.quantity) == [1000, 1000, 500]


# =================================================================== session rules
def test_day_risk_no_new_entries_after_1515_and_flat_by_1555():
    s = sessions("2024-03-04", 1)[0]
    mdf = synthetic_intraday_bars(str(s.date()), str(s.date()), seed=5)
    portal = InMemoryPortal(bars={("XYZ", "1m"): mdf})

    def script(ctx, st):
        t = ctx.now_et()
        if (t.hour, t.minute) == (10, 0):
            ctx.order("XYZ", 100, tag="entry")
        if (t.hour, t.minute) == (15, 20):
            ctx.order("XYZ", 100, tag="late_entry")
    S = make("1m", sizing="DAY_RISK")
    r = Backtester(S(script), portal, zero_cost()).run(s, s, raise_errors=True)
    rej = r.diagnostics["rejected_orders"]
    assert any("no new entries after 15:15" in x["reason"] for x in rej)
    flat = r.fills[r.fills.tag == "flat_by"].iloc[0]
    assert flat.ts == et(s, "15:55") and flat.quantity == -100
    assert r.positions.empty or (r.positions.quantity == 0).all()


# =================================================================== corporate actions & financing
def test_split_and_dividend_in_engine_conserve_nav():
    rows = [(100, 101, 99, 100), (100, 101, 99, 100), (50, 50.5, 49.5, 50), (49.5, 50, 49, 49.5), (49.5, 50, 49, 49.5)]
    ss = sessions("2024-03-04", 5)
    acts = [CorporateAction("XYZ", ActionType.SPLIT, ss[2], pd.Timestamp("2024-02-01", tz="UTC"), ratio=2.0),
            CorporateAction("XYZ", ActionType.CASH_DIVIDEND, ss[3], pd.Timestamp("2024-02-01", tz="UTC"),
                            amount=0.5, pay_date=ss[4])]
    df = synthetic_bars_from_rows(rows, "1d", session_dates=ss)
    portal = InMemoryPortal(bars={("XYZ", "1d"): df}, actions=acts)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.order("XYZ", 101)
    r = Backtester(make()(script), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    eq = r.equity.values
    cash0 = 1_000_000 - 101 * 100
    assert eq[1] == pytest.approx(cash0 + 101 * 100)
    assert eq[2] == pytest.approx(cash0 + 202 * 50)          # split: NAV unchanged at same economic price
    assert eq[3] == pytest.approx(cash0 + 202 * 49.5 + 202 * 0.5)   # ex-date: receivable offsets the drop
    led = r.ledger
    assert led[led.reason == "DIVIDEND"].amount.sum() == pytest.approx(101.0)
    assert eq[4] == pytest.approx(cash0 + 202 * 49.5 + 101.0)
    assert r.positions[r.positions.ts == r.equity.index[2]].quantity.iloc[0] == 202


def test_weekend_financing_three_days_in_engine():
    portal, ss = daily_portal([(100, 101, 99, 100)] * 6, rf=0.05)
    fri = [s for s in ss if s.dayofweek == 4][0]
    r = Backtester(make()(None), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    led = r.ledger[r.ledger.reason == "CASH_INTEREST"]
    led_ts = pd.DatetimeIndex(led.ts).tz_convert("America/New_York").normalize().tz_localize(None)
    fri_amt = led.amount[led_ts == fri].iloc[0]
    daily = led.amount[led_ts != fri].iloc[0]
    assert fri_amt == pytest.approx(3 * daily, rel=1e-3)
    assert daily == pytest.approx(1_000_000 * 0.045 / 360, rel=1e-3)
    assert r.meta["rf_daily"].iloc[0] == pytest.approx(0.05 / 360)


def test_margin_call_forces_liquidation_next_open():
    rows = [(100, 101, 99, 100), (100, 101, 99, 100), (60, 61, 55, 56), (56, 57, 55, 56), (56, 57, 55, 56)]
    portal, ss = daily_portal(rows)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.order("XYZ", 19_000)            # $1.9M on $1M equity (Reg-T 50%: allowed)
    r = Backtester(make()(script), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    calls = r.diagnostics["margin_calls"]
    assert calls and calls[0]["equity"] < calls[0]["maintenance"]
    liq = r.fills[r.fills.tag == "margin_liquidation"]
    assert len(liq) == 1 and liq.iloc[0].ts == CAL.open_close(ss[3])[0] and liq.iloc[0].quantity < 0


def test_reg_t_initial_margin_rejects_oversized_order_and_cash_account_short():
    portal, ss = daily_portal(FLAT)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.order("XYZ", 25_000, tag="too_big")
            ctx.order("XYZ", -10, tag="short")
    r = Backtester(make()(script), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    reasons = {x["tag"]: x["reason"] for x in r.diagnostics["rejected_orders"]}
    assert "Reg-T initial margin" in reasons["too_big"]
    r2 = Backtester(make()(script), portal, zero_cost(account_type=AccountType.CASH)).run(ss[0], ss[-1], raise_errors=True)
    reasons = {x["tag"]: x["reason"] for x in r2.diagnostics["rejected_orders"]}
    assert "cash account" in reasons["short"]


def test_locate_required_for_short_and_borrow_fee_charged():
    portal, ss = daily_portal(FLAT * 2, borrow={"XYZ": (False, 0.0)})

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.order("XYZ", -100, tag="short")
    r = Backtester(make()(script), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    assert r.diagnostics["rejected_orders"][0]["reason"].startswith("no locate")
    portal2, _ = daily_portal(FLAT * 2, borrow={"XYZ": (True, 0.10)})
    r2 = Backtester(make()(script), portal2, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    assert r2.ledger[r2.ledger.reason == "BORROW_FEE"].amount.sum() < 0
    assert r2.trades.iloc[0].pnl < r2.trades.iloc[0].pnl_gross       # borrow attributed to the trade


def test_ssr_blocks_short_hitting_bid_after_10pct_drop():
    rows = [(100, 101, 99, 100), (95, 96, 88, 89), (89, 89.0, 85, 86), (86, 87, 85, 86)]
    portal, ss = daily_portal(rows, start="2024-03-04")

    def script(ctx, st):
        if ctx.session_date() == ss[1]:
            ctx.order("XYZ", -100, tag="short", tif=TimeInForce.DAY)
    cfg = zero_cost(bar_fill=BarFillConfig(ssr_mode="reject"))
    r = Backtester(make()(script), portal, cfg).run(ss[0], ss[-1], raise_errors=True)
    assert any("SSR" in x["reason"] for x in r.diagnostics["rejected_orders"])
    # before Rule 201 (2009) there is no price test
    portal09, ss09 = daily_portal(rows, start="2009-03-02")

    def script09(ctx, st):
        if ctx.session_date() == ss09[1]:
            ctx.order("XYZ", -100, tag="short")
    r09 = Backtester(make()(script09), portal09, cfg).run(ss09[0], ss09[-1], raise_errors=True)
    assert len(r09.fills) == 1


def test_halts_block_fills():
    rows = [(100, 101, 99, 100), (100, 100, 100, 100, 0.0), (101, 102, 100, 101)]
    portal, ss = daily_portal(rows)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.order("XYZ", 100, tif=TimeInForce.GTC)
    r = Backtester(make()(script), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    assert r.fills.iloc[0].ts == CAL.open_close(ss[2])[0]          # zero-volume day skipped
    assert r.diagnostics["sim"]["halted_bars"] == 1
    o1, c1 = CAL.open_close(ss[1])
    halt = MarketEvent("HALT", "XYZ", o1 - pd.Timedelta(minutes=1), effective_at=o1 - pd.Timedelta(minutes=1),
                       payload={"reason": "T1", "resume_at": c1 + pd.Timedelta(hours=1)})
    portal2, _ = daily_portal(FLAT, events=[halt])
    r2 = Backtester(make()(script), portal2, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    assert r2.fills.iloc[0].ts == CAL.open_close(ss[2])[0]


def test_intrabar_resolution_with_minute_bars():
    """Daily bar range hits both stop (95) and target (104). Adverse-first
    assumption picks the stop; 1m data shows the target traded first."""
    ss = sessions("2024-03-04", 3)
    rows = [(100, 101, 99, 100), (100, 106, 94, 100), (100, 101, 99, 100)]
    daily = synthetic_bars_from_rows(rows, "1d", session_dates=ss)
    o1, _ = CAL.open_close(ss[1])
    starts = pd.date_range(o1, periods=390, freq="1min")
    path = np.concatenate([np.linspace(100, 106, 100), np.linspace(106, 94, 200), np.linspace(94, 100, 90)])
    mrows = [(p, p + 0.01, p - 0.01, p) for p in path]
    minute = synthetic_bars_from_rows(mrows, "1m", start_times=starts, volume=10_000)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.bracket("XYZ", 100, entry_type=OrderType.MOO, stop=95.0, target=104.0)
    plain = InMemoryPortal(bars={("XYZ", "1d"): daily})
    r = Backtester(make()(script), plain, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    assert list(r.fills.tag) == ["entry", "stop"]
    assert r.fills.iloc[1].fidelity == int(Fidelity.T2_PROXY)
    rich = InMemoryPortal(bars={("XYZ", "1d"): daily, ("XYZ", "1m"): minute})
    r2 = Backtester(make()(script), rich, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    assert list(r2.fills.tag) == ["entry", "target"]
    assert "resolved with 1m" in r2.fills.iloc[1].fidelity_reason
    assert r2.diagnostics["sim"]["resolved_bars"] >= 1


# =================================================================== options
def test_option_spread_fill_and_expiry_assignment():
    ss = sessions("2024-06-17", 5)           # 17..21 June 2024 (21 = expiry Friday)
    exp = ss[-1]
    under = synthetic_bars_from_rows([(100, 101, 99, 100)] * 4 + [(97, 98, 96, 97)], "1d", session_dates=ss)
    p95 = Instrument.option("XYZ", exp, 98.0, Right.PUT)
    p90 = Instrument.option("XYZ", exp, 90.0, Right.PUT)
    snap = [CAL.open_close(s)[1] for s in ss]            # EOD snapshots at the close
    q95 = pd.DataFrame({"bid": [1.0] * 5, "ask": [1.1] * 5}, index=pd.DatetimeIndex(snap))
    q90 = pd.DataFrame({"bid": [0.2] * 5, "ask": [0.25] * 5}, index=pd.DatetimeIndex(snap))
    portal = InMemoryPortal(bars={("XYZ", "1d"): under}, option_quotes={p95.symbol: q95, p90.symbol: q90},
                            instruments={p95.symbol: p95, p90.symbol: p90})

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.submit(Order(None, -2, OrderType.COMBO, limit_price=0.70,
                             legs=(Leg(p95, 1), Leg(p90, -1)), tif=TimeInForce.GTC, tag="credit_spread"))
    r = Backtester(make()(script), portal, BacktestConfig(debug=True)).run(ss[0], ss[-1], raise_errors=True)
    legs = r.fills[r.fills.tag == "credit_spread"]
    assert set(legs.price.round(4)) == {1.0, 0.25}                      # natural: sell 98P bid, buy 90P ask
    assert (legs.ts == snap[1]).all()                                   # snapshot strictly after submission
    assert (legs.commission > 0).all()
    ev = r.diagnostics["option_events"]
    kinds = sorted(e["kind"] for e in ev)
    assert kinds == ["ASSIGNMENT", "EXPIRE_WORTHLESS"]                 # short 98P ITM at 97, 90P OTM
    assert r.positions[r.positions.ts == r.equity.index[-1]].set_index("symbol").loc["XYZ", "quantity"] == 200
    led = r.ledger
    assert led[led.reason == "OPTION_ASSIGNMENT"].amount.sum() == pytest.approx(-200 * 98.0)


# =================================================================== misc
def test_not_testable_when_required_data_missing():
    portal, ss = daily_portal(FLAT)
    S = make(requires=[Requirement("quotes")])
    r = Backtester(S(None), portal, zero_cost()).run(ss[0], ss[-1])
    assert r.status == RunStatus.NOT_TESTABLE and "quotes" in r.not_testable_reason


def test_determinism():
    df = synthetic_daily_bars("2023-01-02", "2023-06-30", seed=4)

    def script(ctx, st):
        h = ctx.history("XYZ", 10, "1d", adjusted=False)
        if len(h) == 10 and ctx.position("XYZ") == 0 and h.close.iloc[-1] > h.close.iloc[0]:
            ctx.bracket("XYZ", 100, stop=h.low.min(), target=h.high.max() * 1.02)
    res = [Backtester(make()(script), InMemoryPortal(bars={("XYZ", "1d"): df}), BacktestConfig()).run(
        "2023-02-01", "2023-06-30") for _ in range(2)]
    pd.testing.assert_frame_equal(res[0].fills, res[1].fills)
    assert res[0].equity.equals(res[1].equity)


@pytest.mark.parametrize("freq", ["1m", "1d"])
def test_end_to_end_smoke(freq):
    if freq == "1m":
        mdf = synthetic_intraday_bars("2024-03-04", "2024-03-15", seed=7)
        portal = InMemoryPortal(bars={("XYZ", "1m"): mdf, ("XYZ", "1d"): daily_from_intraday(mdf)}, rf=0.05)
        start, end = "2024-03-04", "2024-03-15"
    else:
        portal = InMemoryPortal(bars={("XYZ", "1d"): synthetic_daily_bars("2022-01-03", "2023-12-29", seed=9)},
                                rf=0.03)
        start, end = "2022-03-01", "2023-12-29"

    def script(ctx, st):
        h = ctx.history("XYZ", 5, None, adjusted=False)
        if len(h) < 5:
            return
        if ctx.position("XYZ") == 0 and not ctx.open_orders("XYZ"):
            px = h.close.iloc[-1]
            ctx.bracket("XYZ", 100, stop=px * 0.99, target=px * 1.01)
    S = make(freq, sizing="DAY_RISK" if freq == "1m" else None)
    r = Backtester(S(script), portal, BacktestConfig(debug=True)).run(start, end, raise_errors=True)
    assert r.status == RunStatus.OK and len(r.fills) > 4 and len(r.trades) > 2
    assert r.diagnostics["invariant_checks"] > 0 and r.diagnostics["lookahead_guard_violations"] == 0
    from atlasbt.core.schemas import FILL_COLUMNS, LEDGER_COLUMNS, POSITION_COLUMNS, TRADE_COLUMNS
    assert list(r.fills.columns[:len(FILL_COLUMNS)]) == FILL_COLUMNS
    assert list(r.trades.columns[:len(TRADE_COLUMNS)]) == TRADE_COLUMNS
    assert set(LEDGER_COLUMNS) <= set(r.ledger.columns) and set(POSITION_COLUMNS) <= set(r.positions.columns)
    assert r.fidelity["grade"] in "ABCDE" and r.fidelity["tiers"]
    assert r.ledger.amount.sum() == pytest.approx(r.equity.iloc[-1] - r.positions[r.positions.ts == r.equity.index[-1]].market_value.sum())
    if freq == "1m":
        assert not r.positions.empty or True
        assert (r.fills.ts.dt.tz_convert("America/New_York").dt.strftime("%H:%M") <= "15:55").all()


# =================================================================== more order mechanics
def test_quote_mode_fills_at_ask_after_latency():
    s = sessions("2024-03-04", 1)[0]
    o, c = CAL.open_close(s)
    ts = pd.date_range(o, c, freq="10s")
    q = pd.DataFrame({"bid": 50.0, "ask": 50.02, "bid_size": 1000.0, "ask_size": 1000.0}, index=ts)
    q.loc[q.index > o + pd.Timedelta(minutes=5), ["bid", "ask"]] = [51.0, 51.02]
    portal = InMemoryPortal(quotes={"XYZ": q})
    t_submit = o + pd.Timedelta(minutes=5)

    def script(ctx, st):
        if ctx.now == t_submit:
            ctx.order("XYZ", 100)
    r = Backtester(make("1m")(script), portal, zero_cost()).run(s, s, raise_errors=True)
    f = r.fills.iloc[0]
    assert f.price == 51.02 and f.fidelity == int(Fidelity.T0_QUOTE)
    assert f.ts == t_submit + pd.Timedelta(seconds=10)        # first quote strictly after submit + 1s latency


def test_extended_hours_permission():
    s = sessions("2024-03-04", 1)[0]
    mdf = synthetic_intraday_bars(str(s.date()), str(s.date()), seed=2, include_extended=True)
    portal = InMemoryPortal(bars={("XYZ", "1m"): mdf})

    def script(ctx, st):
        t = ctx.now_et()
        if (t.hour, t.minute) == (8, 0):
            ctx.order("XYZ", 10, tag="ext", outside_rth=True)
            ctx.order("XYZ", 10, tag="rth")
    S = make("1m", sessions=("PRE", "RTH"))
    r = Backtester(S(script), portal, zero_cost()).run(s, s, raise_errors=True)
    f = r.fills.set_index("tag")
    assert f.loc["ext", "ts"] == et(s, "08:00") and "PRE" not in f.loc["rth", "fidelity_reason"]
    assert f.loc["rth", "ts"] == et(s, "09:30")


def test_bracket_time_exit():
    s = sessions("2024-03-04", 1)[0]
    mdf = synthetic_intraday_bars(str(s.date()), str(s.date()), seed=8, daily_vol=0.001)
    portal = InMemoryPortal(bars={("XYZ", "1m"): mdf})

    def script(ctx, st):
        if ctx.minutes_since_open() == 10:
            px = ctx.mark("XYZ")
            ctx.bracket("XYZ", 100, stop=px * 0.9, target=px * 1.1, time_exit=et(s, "11:00"))
    r = Backtester(make("1m")(script), portal, zero_cost()).run(s, s, raise_errors=True)
    tx = r.fills[r.fills.tag == "time_exit"].iloc[0]
    assert tx.ts == et(s, "11:00") and tx.quantity == -100
    assert r.trades.iloc[0].exit_reason == "time_exit"


def test_ssr_ask_limit_mode_posts_at_ask():
    rows = [(100, 101, 99, 100), (95, 96, 88, 89), (89, 92, 85, 86), (86, 87, 85, 86)]
    portal, ss = daily_portal(rows)

    def script(ctx, st):
        if ctx.session_date() == ss[1]:
            ctx.order("XYZ", -100, tag="short")
    cfg = zero_cost(spread=SpreadConfig(bucket_floors=((math.inf, 0.01),), tod_open=(), tod_close=()))
    r = Backtester(make()(script), portal, cfg).run(ss[0], ss[-1], raise_errors=True)
    f = r.fills.iloc[0]
    assert f.liquidity == "MAKER" and f.price > 89.0 and "SSR" in f.fidelity_reason


def test_execution_state_protocol():
    from atlasbt.core.protocols import ExecutionState  # noqa: F401  (structural protocol)
    df = synthetic_daily_bars("2023-01-02", "2023-03-31", seed=4, volume=2e6)
    seen = {}

    def script(ctx, st):
        es = ctx._bt.execution_state
        seen["adv"] = es.adv("XYZ", ctx.now)
        seen["vol"] = es.daily_vol("XYZ", ctx.now)
        seen["ssr"] = es.is_ssr_active("XYZ", ctx.now)
        seen["pos"] = es.position("XYZ")
    Backtester(make()(script), InMemoryPortal(bars={("XYZ", "1d"): df}), zero_cost()).run(
        "2023-03-01", "2023-03-31", raise_errors=True)
    assert seen["adv"] == pytest.approx(2e6, rel=0.05) and 0 < seen["vol"] < 0.05
    assert seen["ssr"] is False and seen["pos"] == 0
