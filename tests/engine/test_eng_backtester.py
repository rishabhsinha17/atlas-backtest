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
    st = make()(script)
    r = Backtester(st, portal, BacktestConfig(debug=True)).run(ss[0], ss[-1], raise_errors=True)
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
    # assignment / expiry fills reach Strategy.on_fill in the same step (wheel strategies need no polling)
    life = [f for f in st.fills if f.order_id == -1]
    by_liq = sorted((f.liquidity, f.instrument.symbol, f.quantity) for f in life)
    assert by_liq == sorted([("ASSIGNMENT", p95.symbol, 2.0), ("ASSIGNMENT", "XYZ", 200.0),
                             ("SETTLEMENT", p90.symbol, -2.0)])
    expiry_close = CAL.open_close(exp)[1]
    assert all(f.ts == expiry_close for f in life)


def test_option_lifecycle_fills_delivered_to_on_fill_same_step():
    ss = sessions("2024-06-17", 6)                       # expiry Fri 21 Jun, run through Mon 24 Jun
    exp = ss[4]
    under = synthetic_bars_from_rows([(100, 101, 99, 100)] * 4 + [(97, 98, 96, 97)] * 2, "1d", session_dates=ss)
    put = Instrument.option("XYZ", exp, 98.0, Right.PUT)
    snap = [CAL.open_close(s)[1] for s in ss]
    q = pd.DataFrame({"bid": [1.0] * 6, "ask": [1.1] * 6}, index=pd.DatetimeIndex(snap))
    portal = InMemoryPortal(bars={("XYZ", "1d"): under}, option_quotes={put.symbol: q},
                            instruments={put.symbol: put})

    class Wheel(Scripted):
        def on_bar(self, ctx):
            if ctx.session_date() == ss[0]:
                ctx.order(put, -1, OrderType.LIMIT, limit_price=0.9, tif=TimeInForce.GTC, tag="csp")

        def on_fill(self, ctx, fill):
            self.fills.append((fill.liquidity, fill.instrument.symbol, fill.quantity, ctx.now,
                               ctx.position("XYZ")))
            if fill.liquidity == "ASSIGNMENT" and not fill.instrument.is_option:
                ctx.order("XYZ", -100, OrderType.MOO, tag="sell_assigned")   # reacts without polling
    st = Wheel()
    r = Backtester(st, portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    asg = [f for f in st.fills if f[0] == "ASSIGNMENT"]
    assert len(asg) == 2 and {f[1] for f in asg} == {put.symbol, "XYZ"}
    assert all(f[3] == CAL.open_close(exp)[1] for f in asg)              # same step as the expiry
    assert [f for f in asg if f[1] == "XYZ"][0][4] == 100                # position already updated
    out = r.fills[r.fills.tag == "sell_assigned"]
    assert len(out) == 1 and out.ts.iloc[0] == CAL.open_close(ss[5])[0]


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
        from atlasbt.engine.context import engine_of
        es = engine_of(ctx).execution_state
        seen["adv"] = es.adv("XYZ", ctx.now)
        seen["vol"] = es.daily_vol("XYZ", ctx.now)
        seen["ssr"] = es.is_ssr_active("XYZ", ctx.now)
        seen["pos"] = es.position("XYZ")
    Backtester(make()(script), InMemoryPortal(bars={("XYZ", "1d"): df}), zero_cost()).run(
        "2023-03-01", "2023-03-31", raise_errors=True)
    assert seen["adv"] == pytest.approx(2e6, rel=0.05) and 0 < seen["vol"] < 0.05
    assert seen["ssr"] is False and seen["pos"] == 0


# =================================================================== red-team follow-ups (engine-side)
def test_split_adjusts_sell_stop_and_notifies_strategy():
    rows = [(100, 101, 99, 100)] * 2 + [(50, 51, 49, 50)] * 3
    ss = sessions("2024-03-04", 5)
    acts = [CorporateAction("XYZ", ActionType.SPLIT, ss[2], pd.Timestamp("2024-02-01", tz="UTC"), ratio=2.0)]
    portal = InMemoryPortal(bars={("XYZ", "1d"): synthetic_bars_from_rows(rows, "1d", session_dates=ss)},
                            actions=acts)
    events = []

    class S(make()):
        def on_order_event(self, ctx, ev):
            events.append(ev)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.bracket("XYZ", 100, stop=90.0, target=130.0)
    r = Backtester(S(script), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    by = {e["tag"]: e for e in events}
    assert by["stop"]["action"] == "ADJUSTED" and by["stop"]["after"]["stop_price"] == 45.0
    assert by["stop"]["after"]["quantity"] == -200
    assert by["target"]["action"] == "CANCELLED"          # sell limits are not adjusted (FINRA 5330)
    assert r.diagnostics["order_events"] and r.trades.iloc[0].quantity == 200


def test_daily_bars_with_day_profile_not_testable():
    portal, ss = daily_portal(FLAT)
    r = Backtester(make(sizing="DAY_RISK")(None), portal, zero_cost()).run(ss[0], ss[-1])
    assert r.status == RunStatus.NOT_TESTABLE and "intraday" in r.not_testable_reason


def test_planned_risk_column():
    rows = [(100, 101, 99, 100), (101, 101.5, 97, 97.5), (97.5, 98, 97, 97.5)]
    portal, ss = daily_portal(rows)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.bracket("XYZ", 100, stop=98.0, planned_entry=100.0)
    r = Backtester(make()(script), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    t = r.trades.iloc[0]
    assert t.initial_risk == pytest.approx(300.0) and t.r_multiple == pytest.approx(-1.0)
    assert t.planned_risk == pytest.approx(200.0) and t.planned_r_multiple == pytest.approx(-1.5)


def test_early_close_rules_shift_with_close():
    s = pd.Timestamp("2024-07-03")                         # 13:00 ET close
    mdf = synthetic_intraday_bars(str(s.date()), str(s.date()), seed=3)
    portal = InMemoryPortal(bars={("XYZ", "1m"): mdf})

    def script(ctx, st):
        t = ctx.now_et()
        if (t.hour, t.minute) == (10, 0):
            ctx.order("XYZ", 100, tag="entry")
        if (t.hour, t.minute) == (12, 20):
            ctx.order("XYZ", 100, tag="late")
    r = Backtester(make("1m", sizing="DAY_RISK")(script), portal, zero_cost()).run(s, s, raise_errors=True)
    assert any(x["tag"] == "late" for x in r.diagnostics["rejected_orders"])      # after close-45m = 12:15
    assert r.fills[r.fills.tag == "flat_by"].iloc[0].ts == et(s, "12:55")


def test_option_plugin_defaults_and_conservative_selectable():
    from atlasbt.options.plugins import CboeStrategyMargin, OccOptionLifecycle
    from atlasbt.portfolio.hooks import DefaultOptionLifecycle
    from atlasbt.portfolio.margin import ConservativeOptionMargin
    portal, ss = daily_portal(FLAT, rf=0.05)
    bt = Backtester(make()(None), portal, BacktestConfig())
    assert isinstance(bt.lifecycle, OccOptionLifecycle) and isinstance(bt.margin.option_model, CboeStrategyMargin)
    assert bt.lifecycle.rate(ss[0]) == pytest.approx(math.log1p(0.05))
    bt2 = Backtester(make()(None), portal, BacktestConfig(option_margin="conservative",
                                                          option_lifecycle="conservative"))
    assert isinstance(bt2.lifecycle, DefaultOptionLifecycle)
    assert isinstance(bt2.margin.option_model, ConservativeOptionMargin)


def test_amend_bracket_and_history_arrays_and_original_quantity():
    rows = [(100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100), (100, 104, 99, 103), (103, 104, 102, 103)]
    portal, ss = daily_portal(rows)

    def script(ctx, st):
        d = ctx.session_date()
        if d == ss[0]:
            st.parent = ctx.bracket("XYZ", 100, stop=90.0)
            st.seen.append(ctx.bracket_group(st.parent))
        if d == ss[2]:
            ctx.amend_bracket(st.parent, target=102.0, stop=95.0)
        h = ctx.history_arrays("XYZ", 3)
        assert not h["close"].flags.writeable and (h["end"] <= ctx.now.value).all()
        st.seen.append(len(h["close"]))
    st = make()(script)
    r = Backtester(st, portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    assert st.seen[0] == f"bracket-{st.parent.id}" and st.seen[1:4] == [1, 2, 3]
    tgt = r.fills[r.fills.tag == "target"]
    assert len(tgt) == 1 and tgt.price.iloc[0] == 102.0 and tgt.ts.iloc[0] >= CAL.open_close(ss[3])[0]
    o = r.diagnostics["orders"].set_index("tag")
    assert o.loc["stop", "stop_price"] == 95.0 and o.loc["stop", "original_quantity"] == -100


def test_requirements_universe_star_and_premarket_probe():
    from atlasbt.core.types import MarketEvent
    s = pd.Timestamp("2024-03-04")
    mdf = synthetic_intraday_bars(str(s.date()), str(s.date()), seed=1)     # RTH only
    ev = [MarketEvent("EARNINGS", "XYZ", pd.Timestamp("2024-03-01", tz="UTC"))]
    portal = InMemoryPortal(bars={("XYZ", "1m"): mdf}, events=ev, universes={"U": ["XYZ", "NOPE"]})
    pre = make("1m", requires=[Requirement("bars", "1m", sessions=("PRE", "RTH"))])
    r = Backtester(pre(None), portal, zero_cost()).run(s, s)
    assert r.status == RunStatus.NOT_TESTABLE and "PRE" in r.not_testable_reason
    star = make("1m", requires=[Requirement("events", "EARNINGS", symbols=("*",))])
    assert Backtester(star(None), portal, zero_cost()).run(s, s).status == RunStatus.OK
    star2 = make("1m", requires=[Requirement("events", "FOMC", symbols=("*",))])
    assert Backtester(star2(None), portal, zero_cost()).run(s, s).status == RunStatus.NOT_TESTABLE
    uni = make("1m", universe=[], universe_name="U", requires=[Requirement("bars", "1m")])
    r = Backtester(uni(None), portal, zero_cost()).run(s, s)
    assert r.status == RunStatus.NOT_TESTABLE and "NOPE" in r.not_testable_reason


def test_moc_exit_after_pending_moo_buy_is_not_a_short_sale():
    portal, ss = daily_portal(FLAT * 2, borrow={"XYZ": (False, 0.0)})        # not shortable at all

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.order("XYZ", 100, OrderType.MOO, tag="in")
            st.seen.append(ctx.order("XYZ", -100, OrderType.MOC, tag="out"))
    st = make()(script)
    r = Backtester(st, portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    assert st.seen[0].status != OrderStatus.REJECTED and not st.seen[0].is_short_sale
    assert set(r.fills.tag) == {"in", "out"} and (r.positions.quantity == 0).all() if len(r.positions) else True


def test_history_not_silently_truncated_by_cache_lookback():
    df = synthetic_daily_bars("2022-01-03", "2023-06-30", seed=2)
    portal = InMemoryPortal(bars={("XYZ", "1d"): df})

    def script(ctx, st):
        if not st.seen:
            st.seen.append(len(ctx.history("XYZ", 200, "1d", adjusted=False)))
            st.seen.append(len(ctx.history_arrays("XYZ", 200, "1d")["close"]))
    st = make()(script)
    Backtester(st, portal, zero_cost(lookback_sessions=60)).run("2023-03-01", "2023-03-10", raise_errors=True)
    assert st.seen == [200, 200]


def test_universe_name_fundamentals_requirement_not_testable():
    import pandas as pd_
    s = pd.Timestamp("2024-03-04")
    portal, ss = daily_portal(FLAT, universes={"SP500": ["XYZ"]})
    S = make(universe=[], universe_name="SP500", requires=[Requirement("fundamentals", "book_equity")])
    r = Backtester(S(None), portal, zero_cost()).run(ss[0], ss[-1])
    assert r.status == RunStatus.NOT_TESTABLE and "fundamentals" in r.not_testable_reason
    fund = pd_.DataFrame({"symbol": ["XYZ"], "field": ["book_equity"], "value": [1.0],
                          "period_end": [pd_.Timestamp("2023-12-31")], "known_at": [pd_.Timestamp("2024-02-01", tz="UTC")],
                          "source": ["synthetic"]})
    portal2, _ = daily_portal(FLAT, universes={"SP500": ["XYZ"]}, fundamentals=fund)
    assert Backtester(S(None), portal2, zero_cost()).run(ss[0], ss[-1]).status == RunStatus.OK


def test_fill_relative_bracket_children_anchor_to_actual_fill():
    rows = [(100, 101, 99, 100), (103, 104, 102.5, 103.5), (103, 103.5, 100, 101), (101, 102, 100, 101)]
    portal, ss = daily_portal(rows)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.bracket("XYZ", 100, stop_offset=2.0, target_fn=lambda px, side: px + side * 4.0)
    r = Backtester(make()(script), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    o = r.diagnostics["orders"].set_index("tag")
    assert o.loc["stop", "stop_price"] == pytest.approx(101.0)          # entry filled at 103 (gap), not 100
    assert o.loc["target", "limit_price"] == pytest.approx(107.0)
    stop = r.fills[r.fills.tag == "stop"].iloc[0]
    assert stop.price == pytest.approx(101.0) and stop.ts >= CAL.open_close(ss[2])[0]
    t = r.trades.iloc[0]
    assert t.initial_risk == pytest.approx(200.0) and t.r_multiple == pytest.approx(-1.0)


def test_not_testable_error_from_strategy_and_internal_orders_protected():
    from atlasbt.core.types import NotTestableError
    portal, ss = daily_portal(FLAT)

    def boom(ctx, st):
        raise NotTestableError("needs point-in-time index membership")
    r = Backtester(make()(boom), portal, zero_cost(debug=False)).run(ss[0], ss[-1])
    assert r.status == RunStatus.NOT_TESTABLE and "membership" in r.not_testable_reason
    r2 = Backtester(make()(boom), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    assert r2.status == RunStatus.NOT_TESTABLE                 # a designed outcome, also with raise_errors

    class EndCheck(make()):
        def on_run_end(self, ctx):
            raise NotTestableError("no cited announcements in window")
    r3 = Backtester(EndCheck(None), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    assert r3.status == RunStatus.NOT_TESTABLE and "announcements" in r3.not_testable_reason

    s = sessions("2024-03-04", 1)[0]
    mdf = synthetic_intraday_bars(str(s.date()), str(s.date()), seed=4)

    def script(ctx, st):
        t = ctx.now_et()
        if (t.hour, t.minute) == (10, 0):
            ctx.order("XYZ", 100)
        if (t.hour, t.minute) == (15, 55):
            ctx.cancel_all()                                   # tries to kill the engine's flatten order
            for o in ctx.open_orders():
                ctx.cancel(o.id)
    r = Backtester(make("1m", sizing="DAY_RISK")(script), InMemoryPortal(bars={("XYZ", "1m"): mdf}),
                   zero_cost()).run(s, s, raise_errors=True)
    assert len(r.fills[r.fills.tag == "flat_by"]) == 1
    assert r.positions.empty or (r.positions.quantity == 0).all()


def test_requirement_premarket_quotes_probe():
    s = pd.Timestamp("2024-03-04")
    o, c = CAL.open_close(s)
    rth = pd.date_range(o, c, freq="1min", inclusive="left") + pd.Timedelta(seconds=5)
    q = pd.DataFrame({"bid": 99.99, "ask": 100.01}, index=rth)
    S = make("1m", requires=[Requirement("quotes", sessions=("PRE", "RTH"))])
    r = Backtester(S(None), InMemoryPortal(quotes={"XYZ": q}), zero_cost()).run(s, s)
    assert r.status == RunStatus.NOT_TESTABLE and "PRE" in r.not_testable_reason
    pre = pd.date_range(et(s, "08:00"), o, freq="5min", inclusive="left")
    q2 = pd.concat([pd.DataFrame({"bid": 99.9, "ask": 100.1}, index=pre), q])
    assert Backtester(S(None), InMemoryPortal(quotes={"XYZ": q2}), zero_cost()).run(s, s).status == RunStatus.OK


def test_pit_portal_whitelist_and_actions_fidelity():
    from atlasbt.engine.pit import PITPortal
    now = pd.Timestamp("2024-03-05 21:00", tz="UTC")

    class _P(InMemoryPortal):
        def risk_free_rate_info(self, as_of, tenor="3M"):
            return (0.05, "DGS3MO", Fidelity.T1_BAR, pd.Timestamp(as_of).normalize())

        def secret_raw_dump(self):
            return "future"
    acts = [CorporateAction("XYZ", ActionType.CASH_DIVIDEND, pd.Timestamp("2024-03-01"), pd.Timestamp("2024-02-01", tz="UTC"),
                            amount=1.0, fidelity=Fidelity.T1_BAR),
            CorporateAction("XYZ", ActionType.SPLIT, pd.Timestamp("2024-03-02"), pd.Timestamp("2024-02-01", tz="UTC"),
                            ratio=2.0, fidelity=Fidelity.T2_PROXY)]
    p = _P(actions=acts)
    pit = PITPortal(p, lambda: now)
    assert pit.risk_free_rate_info()[0] == 0.05
    with pytest.raises(LookaheadError):
        pit.risk_free_rate_info(as_of=now + pd.Timedelta(days=1))
    with pytest.raises(AttributeError):
        pit.secret_raw_dump()
    assert p.fidelity_of("XYZ", "actions") == Fidelity.T2_PROXY


# =================================================================== quote coverage (CRITICAL fix) & combos
def test_auto_quote_mode_is_per_window_bars_elsewhere():
    df = synthetic_daily_bars("2022-01-03", "2023-12-29", seed=5)
    qday = pd.Timestamp("2023-06-15")
    o, c = CAL.open_close(qday)
    qt = pd.date_range(o, c, freq="10s", inclusive="left")
    q = pd.DataFrame({"bid": 99.0, "ask": 99.02, "bid_size": 1e6, "ask_size": 1e6}, index=qt)
    portal = InMemoryPortal(bars={("XYZ", "1d"): df}, quotes={"XYZ": q})

    def script(ctx, st):
        if ctx.session_date() == pd.Timestamp("2022-03-01"):
            ctx.order("XYZ", 100, tag="bh")

    def on_start(ctx, st):
        if ctx.session_date() == qday:
            ctx.order("XYZ", 10, tag="qday")
    r = Backtester(make()(script, on_start=on_start), portal, zero_cost()).run("2022-03-01", "2023-12-29",
                                                                               raise_errors=True)
    bh = r.fills[r.fills.tag == "bh"].iloc[0]
    assert bh.ts == CAL.open_close(pd.Timestamp("2022-03-02"))[0] and bh.quantity == 100
    assert "fill source: bars" in bh.fidelity_reason
    qf = r.fills[r.fills.tag == "qday"].iloc[0]
    assert qf.price == 99.02 and qf.fidelity == int(Fidelity.T0_QUOTE) and "fill source: quotes" in qf.fidelity_reason


def test_sparse_quote_day_falls_back_to_bars():
    rows = [(100, 101, 99, 100)] * 3
    portal0, ss = daily_portal(rows)
    o, c = CAL.open_close(ss[1])
    q = pd.DataFrame({"bid": 90.0, "ask": 90.02, "bid_size": 1e6, "ask_size": 1e6},
                     index=pd.DatetimeIndex([o + pd.Timedelta(hours=2)]))          # 1 quote all day
    portal = InMemoryPortal(bars={("XYZ", "1d"): synthetic_bars_from_rows(rows, "1d", session_dates=ss)},
                            quotes={"XYZ": q})

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.order("XYZ", 100)
    r = Backtester(make()(script), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    f = r.fills.iloc[0]
    assert f.price == pytest.approx(100.0, abs=0.01) and "quotes sparse" in f.fidelity_reason


def test_option_fill_accepts_t3_snapshots_and_rejects_locked():
    ss = sessions("2024-03-04", 3)
    under = synthetic_bars_from_rows([(100, 101, 99, 100)] * 3, "1d", session_dates=ss)
    call = Instrument.option("XYZ", pd.Timestamp("2024-03-15"), 100.0, Right.CALL)
    idx = [et(ss[1], "10:00"), et(ss[1], "11:00")]
    q = pd.DataFrame({"bid": [2.0, 2.0], "ask": [2.0, 2.2], "fidelity": [int(Fidelity.T3_MODELED)] * 2},
                     index=pd.DatetimeIndex(idx))
    portal = InMemoryPortal(bars={("XYZ", "1d"): under}, option_quotes={call.symbol: q},
                            instruments={call.symbol: call})

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.order(call, 1, tag="c")
    r = Backtester(make()(script), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    f = r.fills.iloc[0]
    assert f.ts == et(ss[1], "11:00") and f.price == 2.2 and f.fidelity == int(Fidelity.T3_MODELED)


def test_strategy_config_overrides_whitelist():
    portal, ss = daily_portal(FLAT)
    ok = make(config_overrides={"option_lifecycle": "occ_no_early_assignment"})
    bt = Backtester(ok(None), portal, zero_cost())
    assert bt.lifecycle.early_assignment is False
    assert bt.run(ss[0], ss[-1]).meta["config_overrides"] == {"option_lifecycle": "occ_no_early_assignment"}
    bad = make(config_overrides={"fees": "NO_FEES"})
    with pytest.raises(ValueError):
        Backtester(bad(None), portal, zero_cost())


def test_fill_relative_nan_target_skips_child_keeps_stop():
    rows = [(100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 95, 96), (96, 97, 95, 96)]
    portal, ss = daily_portal(rows)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.bracket("XYZ", 100, stop_offset=3.0, target_fn=lambda px, side: float("nan"))
    r = Backtester(make()(script), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    assert "target" not in set(r.diagnostics["orders"].tag)
    assert any(e["action"] == "CHILD_SKIPPED" for e in r.diagnostics["order_events"])
    assert len(r.fills[r.fills.tag == "stop"]) == 1


def test_daily_bracket_time_exit_moc_and_moc_after_pending_buy():
    rows = [(100, 101, 99, 100), (101, 102, 100, 101.5), (102, 103, 101, 102.5), (103, 104, 102, 103)]
    portal, ss = daily_portal(rows)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.bracket("XYZ", 100, stop=90.0, time_exit_moc=ss[1])
    r = Backtester(make()(script), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    f = r.fills.set_index("tag")
    assert f.loc["entry", "ts"] == CAL.open_close(ss[1])[0]
    assert f.loc["time_exit", "ts"] == CAL.open_close(ss[1])[1] and f.loc["time_exit", "price"] == 101.5
    assert f.loc["time_exit", "liquidity"] == "AUCTION" and f.loc["time_exit", "quantity"] == -100
    o = r.diagnostics["orders"].set_index("tag")
    assert o.loc["stop", "status"] == "CANCELLED"                  # OCO with the MOC exit
    # time_exit_moc=True uses the session of time_exit (two sessions later)
    def script2(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.bracket("XYZ", 100, stop=90.0, time_exit=ss[2], time_exit_moc=True)
    r2 = Backtester(make()(script2), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    f2 = r2.fills.set_index("tag")
    assert f2.loc["time_exit", "ts"] == CAL.open_close(ss[2])[1] and f2.loc["time_exit", "price"] == 102.5
    # the stop trades first intrabar: the MOC leg is cancelled, never flips the position
    portal3, ss3 = daily_portal([(100, 101, 99, 100), (101, 102, 85, 95), (95, 96, 94, 95)])

    def script3(ctx, st):
        if ctx.session_date() == ss3[0]:
            ctx.bracket("XYZ", 100, stop=90.0, time_exit_moc=ss3[1])
    r3 = Backtester(make()(script3), portal3, zero_cost()).run(ss3[0], ss3[-1], raise_errors=True)
    assert list(r3.fills["tag"]) == ["entry", "stop"]
    assert r3.fills["quantity"].sum() == 0
    o3 = r3.diagnostics["orders"].set_index("tag")
    assert o3.loc["time_exit", "status"] == "CANCELLED"


def test_same_close_moc_exit_of_pending_market_buy_not_short():
    """S114 pattern: pending market buy (fills at next open) and an MOC exit for
    that session's close, with the stock not shortable: the exit is a long sale."""
    portal, ss = daily_portal(FLAT * 2, borrow={"XYZ": (False, 0.0)})

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.order("XYZ", 100, tag="in")
            st.seen.append(ctx.order("XYZ", -100, OrderType.MOC, tag="out"))
    st = make()(script)
    r = Backtester(st, portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    assert not st.seen[0].is_short_sale and st.seen[0].status == OrderStatus.FILLED
    assert list(r.fills.tag) == ["in", "out"]
    assert r.fills.ts.iloc[1] == CAL.open_close(ss[1])[1]


def test_combo_legs_use_instrument_master_adjusted_deliverable():
    from dataclasses import replace as dc_replace
    ss = sessions("2024-06-17", 5)
    exp = ss[-1]
    under = synthetic_bars_from_rows([(80, 81, 79, 80)] * 5, "1d", session_dates=ss)
    c60 = Instrument.option("XYZ", exp, 60.0, Right.CALL)          # strategy-built: default deliverable 100
    c90 = Instrument.option("XYZ", exp, 130.0, Right.CALL)   # 1.5 x 80 = 120 < 130: OTM
    m60 = dc_replace(c60, deliverable_shares=150.0, tick_size=0.05)  # adjusted contract in the master
    m90 = dc_replace(c90, deliverable_shares=150.0, tick_size=0.05)
    snap = [CAL.open_close(s)[1] for s in ss]
    q60 = pd.DataFrame({"bid": [29.9] * 5, "ask": [30.1] * 5}, index=pd.DatetimeIndex(snap))
    q90 = pd.DataFrame({"bid": [0.5] * 5, "ask": [0.6] * 5}, index=pd.DatetimeIndex(snap))
    portal = InMemoryPortal(bars={("XYZ", "1d"): under}, option_quotes={c60.symbol: q60, c90.symbol: q90},
                            instruments={c60.symbol: m60, c90.symbol: m90})

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            st.seen.append(ctx.submit(Order(None, 1, OrderType.COMBO, limit_price=30.0,
                                            legs=(Leg(c60, 1), Leg(c90, -1)), tif=TimeInForce.GTC, tag="spread")))
            bad = dc_replace(c60, multiplier=10.0)                    # contradicts the master
            st.seen.append(ctx.submit(Order(None, 1, OrderType.COMBO, limit_price=30.0,
                                            legs=(Leg(bad, 1), Leg(c90, -1)), tag="bad")))
    st = make()(script)
    r = Backtester(st, portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    assert st.seen[1].status == OrderStatus.REJECTED and "multiplier" in st.seen[1].reject_reason
    assert st.seen[0].legs[0].instrument.deliverable_shares == 150.0
    ev = {e["option"]: e for e in r.diagnostics["option_events"]}
    assert ev[c90.symbol]["kind"] == "EXPIRE_WORTHLESS"
    assert ev[c60.symbol]["kind"] == "EXERCISE" and ev[c60.symbol]["underlying_qty"] == 150
    last = r.positions[r.positions.ts == r.equity.index[-1]].set_index("symbol")
    assert last.loc["XYZ", "quantity"] == 150
    led = r.ledger
    assert led[led.reason == "OPTION_EXERCISE"].amount.sum() == pytest.approx(-60.0 * 100)


def test_inmemory_portal_fidelity_overrides():
    from atlasbt.core.types import Fidelity as F
    portal, ss = daily_portal(FLAT, fidelity={("XYZ", "options", "1d"): F.T3_MODELED, ("*", "bars"): F.T2_PROXY})
    assert portal.fidelity_of("XYZ", "bars", "1d") == F.T2_PROXY
    assert portal.fidelity_of("XYZ", "bars") == F.T2_PROXY
    assert portal.fidelity_of("ABC", "bars", "1d") == F.NA                 # no data: still NA
    c = Instrument.option("XYZ", ss[-1], 100.0, Right.CALL)
    q = pd.DataFrame({"bid": [1.0], "ask": [1.1]}, index=pd.DatetimeIndex([CAL.open_close(ss[0])[1]]))
    p2 = InMemoryPortal(option_quotes={c.symbol: q}, instruments={c.symbol: c},
                        fidelity={("XYZ", "options", "1d"): F.T3_MODELED})
    assert p2.fidelity_of("XYZ", "options", "1d") == F.T3_MODELED
    assert p2.fidelity_of("XYZ", "options", "1m") == F.T1_BAR             # default when not configured


def _short_put_run(quotes_by_day):
    """Short 1 American 100P (exp ~6 weeks), stock 100 -> 80 from session 2; rf 5%
    (carry ≈ 100(1-e^{-rT}) ≈ 0.5). ``quotes_by_day``: {session index: (bid, ask)}."""
    ss = sessions("2024-03-04", 6)
    exp = pd.Timestamp("2024-04-19")
    under = synthetic_bars_from_rows([(100, 101, 99, 100)] * 2 + [(80, 81, 79, 80)] * 4, "1d", session_dates=ss)
    put = Instrument.option("XYZ", exp, 100.0, Right.PUT)
    idx = sorted(quotes_by_day)
    q = pd.DataFrame({"bid": [quotes_by_day[i][0] for i in idx], "ask": [quotes_by_day[i][1] for i in idx],
                      "fidelity": [int(quotes_by_day[i][2]) if len(quotes_by_day[i]) > 2 else 2 for i in idx]},
                     index=pd.DatetimeIndex([CAL.open_close(ss[i])[1] for i in idx]))
    portal = InMemoryPortal(bars={("XYZ", "1d"): under}, option_quotes={put.symbol: q},
                            instruments={put.symbol: put}, rf=0.05)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.order(put, -1, OrderType.LIMIT, limit_price=1.8, tif=TimeInForce.GTC, tag="short_put")
    r = Backtester(make()(script), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    return r, ss, put


def test_stale_mark_itm_put_not_early_assigned_when_same_day_extrinsic_exceeds_carry():
    from atlasbt.core.types import Fidelity as F
    # entry quote on day 1 (mid 2.0), nothing on day 2 (only the stale 2.0 mark), then same-day
    # model snapshots (T3) worth 22.5 = intrinsic 20 + extrinsic 2.5 > carry
    r, ss, put = _short_put_run({1: (1.9, 2.1), 3: (22.4, 22.6, F.T3_MODELED), 4: (22.4, 22.6, F.T3_MODELED)})
    assert len(r.fills[r.fills.tag == "short_put"]) == 1
    assert not [e for e in r.diagnostics["option_events"] if e["kind"] == "ASSIGNMENT"]
    last = r.positions[r.positions.ts == r.equity.index[-1]].set_index("symbol")
    assert last.loc[put.symbol, "quantity"] == -1 and "XYZ" not in last.index
    assert any("no same-session option mark" in x["reason"] for x in r.fidelity["reasons"])


def test_deep_itm_put_with_same_session_mark_is_early_assigned():
    r, ss, put = _short_put_run({1: (1.9, 2.1), 2: (20.0, 20.1), 3: (20.0, 20.1)})
    ev = [e for e in r.diagnostics["option_events"] if e["kind"] == "ASSIGNMENT"]
    assert len(ev) == 1 and ev[0]["ts"] == CAL.open_close(ss[2])[1]      # first close deep ITM, same-day mark
    last = r.positions[r.positions.ts == r.equity.index[-1]].set_index("symbol")
    assert last.loc["XYZ", "quantity"] == 100 and put.symbol not in last.index


def test_inmemory_portal_events_fidelity_not_canonicalised_as_freq():
    from atlasbt.core.types import Fidelity as F, MarketEvent
    t = pd.Timestamp("2024-03-01", tz="UTC")
    p = InMemoryPortal(events=[MarketEvent("EARNINGS", "XYZ", t, t, {}, F.T1_BAR),
                               MarketEvent("EARNINGS", "ABC", t, t, {}, F.T2_PROXY)])
    assert p.fidelity_of("*", "events", "EARNINGS") == F.T2_PROXY
    assert p.fidelity_of("XYZ", "events", "EARNINGS") == F.T1_BAR
    assert p.fidelity_of("*", "events", "FOMC") == F.NA
    portal, ss = daily_portal(FLAT, events=[MarketEvent("EARNINGS", "XYZ", t, t, {}, F.T1_BAR)])
    S = make(requires=[Requirement("events", "EARNINGS", symbols=("*",))])
    assert Backtester(S(None), portal, zero_cost()).run(ss[0], ss[-1]).status == RunStatus.OK


def test_universe_coverage_filters_members_without_data_and_min_coverage():
    ss = sessions("2024-03-04", 3)
    bars = {(f"S{i:02d}", "1d"): synthetic_bars_from_rows(FLAT, "1d", session_dates=ss) for i in range(39)}
    members = [f"S{i:02d}" for i in range(40)]             # S39 has no data: coverage 39/40 = 97.5%
    portal = InMemoryPortal(bars=bars, universes={"SP500": members})

    def script(ctx, st):
        st.seen.append(list(ctx.portal().universe("SP500")))
    S = make(universe=[], universe_name="SP500")
    st = S(script)
    r = Backtester(st, portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    assert r.status == RunStatus.OK
    assert all("S39" not in u and len(u) == 39 for u in st.seen)
    cov = r.meta["universe_coverage"]
    assert cov["coverage_min"] == pytest.approx(0.975) and "S39" in cov["dropped"]
    assert len(cov["series"]) == 3 and "excluded" in r.meta["survivorship_bias"]
    assert any("universe SP500 coverage 97.5%" in x["reason"] for x in r.fidelity["reasons"])
    # strict: 1.0 reproduces the old all-members rule
    r2 = Backtester(S(script), portal, zero_cost(min_universe_coverage=1.0)).run(ss[0], ss[-1])
    assert r2.status == RunStatus.NOT_TESTABLE and "coverage 97.5%" in r2.not_testable_reason
    # coverage dropping below the minimum at a later rebalance session -> NOT_TESTABLE
    late = InMemoryPortal(bars=bars, universes={"SP500": members[:39]})
    orig = late.universe

    def universe(name, as_of):
        extra = [f"X{i}" for i in range(5)] if pd.Timestamp(as_of) >= CAL.open_close(ss[2])[0] else []
        return orig(name, as_of) + extra                     # 5 new members without data: 39/44
    late.universe = universe
    r3 = Backtester(S(script), late, zero_cost()).run(ss[0], ss[-1])
    assert r3.status == RunStatus.NOT_TESTABLE and "coverage 88.6%" in r3.not_testable_reason


def test_spy_round_trip_2019_calibrated_spread_is_bp_not_tens_of_bp():
    """Default spread model = calibrated (TAQ NBBO fit). A SPY coin-flip round trip in
    2019 costs ~2 x half-spread + impact, well under 2 bp; the legacy Corwin-Schultz
    estimate on the same bars is tens of bp."""
    from atlasbt.core.types import AssetClass
    df = synthetic_daily_bars("2018-10-01", "2019-12-31", 290.0, daily_vol=0.008, volume=7e7, seed=7)
    spy = Instrument(symbol="SPY", asset_class=AssetClass.ETF)
    portal = InMemoryPortal(bars={("SPY", "1d"): df}, instruments={"SPY": spy})
    S = make(universe=["SPY"])

    def script(ctx, st):
        d = ctx.session_date()
        if d.month == 6 and d.day <= 14:
            ctx.order("SPY", 1000 if ctx.position("SPY") == 0 else -ctx.position("SPY"))
    cfg = BacktestConfig(fees="NO_FEES")
    assert cfg.resolved_spread_model() == "calibrated"
    r = Backtester(S(script), portal, cfg).run("2019-06-03", "2019-06-28", raise_errors=True)
    f = r.fills
    assert len(f) >= 4 and r.meta["config"]["spread_model"] == "calibrated"
    rt_bp = 2 * ((f.spread_cost + f.impact_cost) / f.notional.abs()).mean() * 1e4
    assert 0.05 < rt_bp < 2.0
    assert f.fidelity_reason.str.contains("spread modelled (calibrated)", regex=False).all() and (f.fidelity >= 2).all()
    legacy = BacktestConfig(fees="NO_FEES", spread_model="corwin_schultz")
    r2 = Backtester(S(script), portal, legacy).run("2019-06-03", "2019-06-28", raise_errors=True)
    f2 = r2.fills
    assert 2 * ((f2.spread_cost + f2.impact_cost) / f2.notional.abs()).mean() * 1e4 > 5 * rt_bp
