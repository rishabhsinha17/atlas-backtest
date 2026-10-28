"""Red-team findings: money in atlasbt/portfolio, atlasbt/risk and the result
assembly (atlasbt/engine/results.py). Each test asserts the CORRECT behaviour
and FAILS on the current code. SYNTHETIC data, mechanics only."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from atlasbt.core.types import ActionType, CorporateAction, OrderType
from rt_pf_helpers import CAL, KNOWN, daily_portal, run, sessions, zero_cost


def _flat(n, px=100.0):
    return [(px, px + 1, px - 1, px)] * n


# ------------------------------------------------------------------ splits vs trades table
def test_split_round_trip_trade_pnl_matches_nav_change():
    """Buy 100 @100, 2:1 split, sell all 200 @50: economically flat, NAV
    unchanged. The trades table must show pnl ~ 0, not -10,000."""
    ss = sessions("2024-03-04", 5)
    rows = _flat(2, 100.0) + _flat(3, 50.0)
    acts = [CorporateAction("XYZ", ActionType.SPLIT, ss[2], KNOWN, ratio=2.0)]
    portal = daily_portal({"XYZ": rows}, ss, actions=acts)

    def script(ctx, st):
        d = ctx.session_date()
        if d == ss[0]:
            ctx.order("XYZ", 100)
        if d == ss[3] and ctx.position("XYZ"):
            ctx.order("XYZ", -ctx.position("XYZ"), tag="exit")
    r, _ = run(script, portal, ss)
    assert r.equity.iloc[-1] == pytest.approx(1_000_000.0)
    closed = r.trades[r.trades.closed]
    assert len(closed) == 1 and len(r.trades) == 1
    assert closed.pnl.sum() == pytest.approx(r.equity.iloc[-1] - 1_000_000.0, abs=1e-6)


def test_reverse_split_round_trip_shows_no_phantom_profit():
    """Buy 100 @10, 1:10 reverse split (10 sh @100), sell 10 @100: NAV flat.
    The trades table must contain one CLOSED trade with pnl ~ 0 (today the
    tracker still thinks 90 shares are open and books +900 phantom gross)."""
    ss = sessions("2024-03-04", 5)
    rows = _flat(2, 10.0) + _flat(3, 100.0)
    acts = [CorporateAction("XYZ", ActionType.SPLIT, ss[2], KNOWN, ratio=0.1)]
    portal = daily_portal({"XYZ": rows}, ss, actions=acts)

    def script(ctx, st):
        d = ctx.session_date()
        if d == ss[0]:
            ctx.order("XYZ", 100)
        if d == ss[3] and ctx.position("XYZ"):
            ctx.order("XYZ", -ctx.position("XYZ"), tag="exit")
    r, _ = run(script, portal, ss)
    assert r.positions[r.positions.ts == r.equity.index[-1]].empty
    assert r.equity.iloc[-1] == pytest.approx(1_000_000.0)
    assert bool(r.trades.closed.all()), r.trades[["symbol", "quantity", "pnl", "closed"]]
    assert r.trades.pnl.sum() == pytest.approx(0.0, abs=1e-6)


def test_split_keeps_protective_stop_working_adjusted():
    """Bracket long 100 @100 with stop 90; 2:1 split; post-split price falls
    to 40 (= 80 pre-split, through the stop). In real life the GTC stop is
    adjusted to 45 x 200 (or the trader re-enters it); the engine silently
    cancels it (only a diagnostics warning, no callback) and the position
    rides the loss unprotected."""
    ss = sessions("2024-03-04", 6)
    rows = _flat(2, 100.0) + [(50, 51, 49, 50), (40, 41, 39, 40), (40, 41, 39, 40), (40, 41, 39, 40)]
    acts = [CorporateAction("XYZ", ActionType.SPLIT, ss[2], KNOWN, ratio=2.0)]
    portal = daily_portal({"XYZ": rows}, ss, actions=acts)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.bracket("XYZ", 100, stop=90.0)
    r, _ = run(script, portal, ss)
    stops = r.fills[r.fills.tag == "stop"]
    assert len(stops) == 1 and stops.quantity.iloc[0] == -200


def test_two_dividends_same_ex_date_not_double_attributed_to_trade():
    """Regular 0.50 + special 2.00 with the same ex-date. Price drops by the
    2.50 on ex-date; position sold. NAV change = 0, so trade pnl must be 0.
    _apply_action re-attributes every accrual with ex_date == session on each
    action -> the first dividend is attributed twice (+50 phantom)."""
    ss = sessions("2024-03-04", 6)
    rows = _flat(2, 100.0) + _flat(4, 97.5)
    acts = [CorporateAction("XYZ", ActionType.CASH_DIVIDEND, ss[2], KNOWN, amount=0.5, pay_date=ss[3]),
            CorporateAction("XYZ", ActionType.SPECIAL_DIVIDEND, ss[2], KNOWN, amount=2.0, pay_date=ss[3])]
    portal = daily_portal({"XYZ": rows}, ss, actions=acts)

    def script(ctx, st):
        d = ctx.session_date()
        if d == ss[0]:
            ctx.order("XYZ", 100)
        if d == ss[3] and ctx.position("XYZ"):
            ctx.order("XYZ", -100, tag="exit")
    r, _ = run(script, portal, ss)
    nav_change = r.equity.iloc[-1] - 1_000_000.0
    assert nav_change == pytest.approx(0.0, abs=1e-6)
    led = r.ledger
    assert led[led.reason == "DIVIDEND"].amount.sum() == pytest.approx(250.0)
    assert r.trades.pnl.sum() == pytest.approx(nav_change, abs=1e-6)


def test_symbol_change_keeps_round_trip_in_trades_table():
    """Buy 100 XYZ @100, XYZ -> ABC, sell 100 ABC @110: +1,000 NAV. Trades
    table must hold one closed trade with pnl 1,000 (today: an orphan 'open'
    XYZ trade with pnl 0 and the ABC sale is not tracked at all)."""
    ss = sessions("2024-03-04", 5)
    acts = [CorporateAction("XYZ", ActionType.SYMBOL_CHANGE, ss[2], KNOWN, new_symbol="ABC")]
    portal = daily_portal({"XYZ": _flat(2, 100.0), "ABC": _flat(2, 100.0) + _flat(3, 110.0)}, ss, actions=acts)

    def script(ctx, st):
        d = ctx.session_date()
        if d == ss[0]:
            ctx.order("XYZ", 100)
        if d == ss[3] and ctx.position("ABC"):
            ctx.order("ABC", -100, tag="exit")
    r, _ = run(script, portal, ss)
    assert r.equity.iloc[-1] == pytest.approx(1_001_000.0)
    assert len(r.trades) == 1 and bool(r.trades.closed.iloc[0])
    assert r.trades.pnl.sum() == pytest.approx(1_000.0)


def test_spinoff_round_trip_trades_sum_to_nav_change():
    """Hold 100 PAR @100; 1:1 spin-off of KID (20% of basis); PAR trades at
    80, KID at 20 -> economically flat. Sell both. Sum of trade pnl must equal
    the NAV change (0). Today the PAR trade keeps its pre-spin entry (pnl
    -2,000) and the KID round trip is missing."""
    ss = sessions("2024-03-04", 6)
    acts = [CorporateAction("PAR", ActionType.SPINOFF, ss[2], KNOWN, ratio=1.0, amount=0.2, new_symbol="KID")]
    portal = daily_portal({"PAR": _flat(2, 100.0) + _flat(4, 80.0),
                           "KID": _flat(2, 20.0) + _flat(4, 20.0)}, ss, actions=acts)

    def script(ctx, st):
        d = ctx.session_date()
        if d == ss[0]:
            ctx.order("PAR", 100)
        if d == ss[3]:
            for s in ("PAR", "KID"):
                if ctx.position(s):
                    ctx.order(s, -ctx.position(s), tag="exit")
    r, _ = run(script, portal, ss, universe=["PAR", "KID"])
    nav_change = r.equity.iloc[-1] - 1_000_000.0
    assert nav_change == pytest.approx(0.0, abs=1e-6)
    assert r.trades.pnl.sum() == pytest.approx(nav_change, abs=1e-6)


# ------------------------------------------------------------------ sizing-profile session rules
def test_day_risk_flat_by_1555_holds_with_30min_bars():
    """DAY_RISK 'flat by 15:55 ET'. With 30-minute bars the engine only checks
    flat_by at step ends (15:30, 16:00); at 16:00 it submits a DAY market order
    that is valid for the NEXT session, so the 'intraday' position is carried
    overnight and exits at the next open (gap risk the profile forbids)."""
    from atlasbt.engine.testing import InMemoryPortal, synthetic_intraday_bars
    ss = sessions("2024-03-04", 2)
    df = synthetic_intraday_bars(str(ss[0].date()), str(ss[1].date()), freq="30m", seed=1)
    portal = InMemoryPortal(bars={("XYZ", "30m"): df})

    def script(ctx, st):
        t = ctx.now_et()
        if ctx.session_date() == ss[0] and (t.hour, t.minute) == (10, 30):
            ctx.order("XYZ", 100, tag="entry")
    r, _ = run(script, portal, ss, freq="30m", sizing="DAY_RISK")
    # no position may survive the first session close
    eod0 = r.positions[r.positions.ts == r.equity.index[0]]
    assert eod0.empty or (eod0.quantity == 0).all(), eod0[["symbol", "quantity"]]


# ------------------------------------------------------------------ results meta
def test_meta_risk_free_rate_is_annual_rate_not_inflated_by_weekends():
    """rf constant 5%. meta['risk_free_rate'] (documented 'float fallback,
    annualised simple') = mean(rf*days/360)*360 = rf * mean calendar days per
    session (~1.4 x over a 2-week window)."""
    ss = sessions("2024-03-04", 10)
    portal = daily_portal({"XYZ": _flat(10)}, ss, rf=0.05)
    r, _ = run(None, portal, ss)
    assert r.meta["risk_free_rate"] == pytest.approx(0.05, rel=1e-6)


# ------------------------------------------------------------------ pricing seen by the strategy on a split ex-date
def test_ctx_mark_on_split_ex_date_session_start_is_post_split():
    """On the ex-date, on_session_start sees qty already doubled but
    ctx.mark() still returns the PRE-split prior close -> position value (and
    any sizing / gross-exposure computed from it) is 2x the truth."""
    ss = sessions("2024-03-04", 4)
    rows = _flat(2, 100.0) + _flat(2, 50.0)
    acts = [CorporateAction("XYZ", ActionType.SPLIT, ss[2], KNOWN, ratio=2.0)]
    portal = daily_portal({"XYZ": rows}, ss, actions=acts)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.order("XYZ", 100)

    def on_start(ctx, st):
        if ctx.session_date() == ss[2]:
            st.seen.append((ctx.position("XYZ"), ctx.mark("XYZ"), ctx.nav - ctx.cash))
    r, st = run(script, portal, ss, on_start=on_start)
    qty, mark, mv = st.seen[0]
    assert qty == 200
    assert qty * mark == pytest.approx(mv)          # 200 x 50 = 10,000 (today 200 x 100)


# ------------------------------------------------------------------ sizing functions
class _FakeCtx:
    def __init__(self, nav, pos, px):
        self.nav = nav
        self.prior_nav = nav
        self._pos = pos
        self._px = px

    def positions(self):
        return dict(self._pos)

    def position(self, s):
        return self._pos.get(s, 0.0)

    def mark(self, s):
        return self._px[s]

    def instrument(self, s):
        from atlasbt.core.types import AssetClass, Instrument
        return Instrument(symbol=s, asset_class=AssetClass.STOCK)


def test_swing_notional_respects_10pct_per_instrument_with_existing_position():
    """SWING_NOTIONAL: 'Long notional 10% of NAV per instrument'. Already
    holding 10% of NAV in XYZ, the sizer must return 0; it returns another
    1,000 shares (20% single-name) because neither size_swing_notional nor
    the ctx helper looks at the existing position."""
    from atlasbt.risk import size_swing_notional_ctx
    ctx = _FakeCtx(1_000_000.0, {"XYZ": 1_000.0}, {"XYZ": 100.0})
    assert size_swing_notional_ctx(ctx, "XYZ", 100.0, side=1) == 0


def test_cashsec_reserves_strike_times_deliverable_for_mini_options():
    """OPT_CASHSEC: '100*put strike USD reserved per standard contract or
    equivalent deliverable cash'. A mini option (10 shares, multiplier 10)
    must reserve strike*10 per contract. per = strike*deliverable*(mult/100)
    = strike*1 -> 10x too many contracts for the cash."""
    from atlasbt.risk import size_opt_cashsec
    n = size_opt_cashsec(10_000.0, 100.0, multiplier=10.0, deliverable=10.0)
    assert n == 10                                   # 10 x (100 x 10) = 10,000 cash secured


# ------------------------------------------------------------------ margin / buying power
def _opt(strike=100.0, right=None, exp="2024-06-21"):
    from atlasbt.core.types import Instrument, Right
    return Instrument.option("XYZ", pd.Timestamp(exp), strike, right or Right.CALL)


def _fill(inst, q, px, ts="2024-03-04 15:00"):
    from atlasbt.core.types import CostBreakdown, Fidelity, Fill
    return Fill(1, inst, pd.Timestamp(ts, tz="UTC"), q, px, CostBreakdown(), Fidelity.T1_BAR)


def test_reg_t_long_options_have_no_loan_value():
    """Reg T: long options (<= 9 months) must be paid in full and have no loan
    value. Account: $1M cash spent on $1M of long calls, then buy $2M of
    stock. Equity (NAV) = $1M, Reg-T initial on the stock = $1M -> the model
    approves, i.e. the option premium is treated as margin collateral and
    the account is levered 3x."""
    from atlasbt.core.types import AssetClass, Instrument
    from atlasbt.portfolio import Account, RegTMarginModel
    acct = Account(1_000_000.0)
    call = _opt()
    acct.apply_fill(_fill(call, 100, 100.0))                   # 100 x 100 x 100 = $1M premium
    acct.mark(call.symbol, 100.0)
    stk = Instrument(symbol="XYZ", asset_class=AssetClass.STOCK)
    ok, why = RegTMarginModel().check_order(acct, stk, 20_000, 100.0, lambda s: 100.0)
    assert not ok


def test_cash_account_stock_purchase_cannot_use_cash_securing_short_puts():
    """OPT_CASHSEC / cash account: cash reserved for short puts ('margin cannot
    be double counted') must not also pay for stock. $1M cash, short 100 puts
    K=100 ($1M reserved). A $990k stock purchase is approved because the cash
    check for a BUY ignores existing put reserves."""
    from atlasbt.core.types import AssetClass, Instrument, Right
    from atlasbt.portfolio import Account, AccountType, RegTMarginModel
    acct = Account(1_000_000.0, AccountType.CASH)
    put = _opt(100.0, Right.PUT)
    acct.apply_fill(_fill(put, -100, 1.0))                      # +$10k premium
    acct.mark(put.symbol, 1.0)
    stk = Instrument(symbol="XYZ", asset_class=AssetClass.STOCK)
    ok, why = RegTMarginModel().check_order(acct, stk, 9_900, 100.0, lambda s: 100.0)
    assert not ok


def test_marketable_short_limit_cannot_bypass_reg_t_check():
    """Buying-power check prices an order at its LIMIT. A marketable sell-short
    limit at $1 on a $100 stock (fills at ~$100) is checked as $1 x 50,000 =
    $50k and approved; it fills as a $5M short on $1M equity (5x, Reg-T
    allows 2x)."""
    ss = sessions("2024-03-04", 3)
    portal = daily_portal({"XYZ": _flat(3)}, ss)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.order("XYZ", -50_000, OrderType.LIMIT, limit_price=1.0, tag="short")
    r, _ = run(script, portal, ss)
    shorted = r.fills[r.fills.tag == "short"].quantity.sum()
    assert abs(shorted) * 100.0 <= 2.0 * 1_000_000.0 + 1e-6, f"filled {shorted} shares"


# ------------------------------------------------------------------ options held through an underlying split
def test_short_call_is_adjusted_on_underlying_split():
    """Short 1 XYZ 100C; XYZ splits 2:1 before expiry; at expiry XYZ = 60
    (= 120 pre-split). OCC adjusts the contract (2 x 50C or 1 x 100C on 200
    shares) -> ITM -> assignment, loss $2,000 on intrinsic. The engine adjusts
    only the stock: the call keeps strike 100 / 100 shares and expires
    WORTHLESS (liability vanishes -> optimistic)."""
    from atlasbt.core.types import Instrument, Right
    from atlasbt.engine.testing import InMemoryPortal, synthetic_bars_from_rows
    ss = sessions("2024-06-14", 6)                     # 14..21 June 2024, 21 = expiry Friday
    exp = ss[-1]
    rows = _flat(2, 100.0) + _flat(3, 50.0) + [(60, 61, 59, 60)]
    under = synthetic_bars_from_rows(rows, "1d", session_dates=ss)
    call = Instrument.option("XYZ", exp, 100.0, Right.CALL)
    snap = [CAL.open_close(s)[1] for s in ss]
    q = pd.DataFrame({"bid": [2.0] * 6, "ask": [2.1] * 6}, index=pd.DatetimeIndex(snap))
    acts = [CorporateAction("XYZ", ActionType.SPLIT, ss[2], KNOWN, ratio=2.0)]
    portal = InMemoryPortal(bars={("XYZ", "1d"): under}, option_quotes={call.symbol: q},
                            instruments={call.symbol: call}, actions=acts)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.order(call, -1, OrderType.MARKET, tag="short_call")
    r, _ = run(script, portal, ss)
    assert len(r.fills[r.fills.tag == "short_call"]) == 1
    kinds = [e["kind"] for e in r.diagnostics["option_events"]]
    assert "ASSIGNMENT" in kinds, kinds


# ------------------------------------------------------------------ marks / corporate-action dates
def test_position_in_symbol_that_stops_trading_is_flagged():
    """XYZ has bars for 3 sessions only (no DELIST action in the data, the
    usual survivorship hole). The position is carried at its last close for
    27 sessions with no warning or fidelity flag."""
    ss = sessions("2024-03-04", 30)
    portal = daily_portal({"XYZ": _flat(3), "SPY": _flat(30)}, ss)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.order("XYZ", 1000)
    r, _ = run(script, portal, ss, universe=["XYZ", "SPY"])
    texts = [x["reason"] for x in r.fidelity["reasons"]] + list(r.diagnostics["warnings"])
    assert any("stale" in t.lower() or "no price" in t.lower() for t in texts), texts


def test_split_with_ex_date_on_exchange_holiday_is_still_applied():
    """Reverse split 1:10 recorded with ex-date Good Friday 2024-03-29 (NYSE
    holiday). The engine only applies actions whose ex_date == session, so the
    split is never applied: qty stays 1,000 while the price goes 10 -> 100 and
    NAV shows a +$90,000 phantom gain."""
    ss = sessions("2024-03-26", 6)                      # 26,27,28 Mar, 1,2,3 Apr
    assert pd.Timestamp("2024-03-29") not in ss
    rows = _flat(3, 10.0) + _flat(3, 100.0)
    acts = [CorporateAction("XYZ", ActionType.SPLIT, pd.Timestamp("2024-03-29"), KNOWN, ratio=0.1)]
    portal = daily_portal({"XYZ": rows}, ss, actions=acts)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.order("XYZ", 1000)
    r, _ = run(script, portal, ss)
    assert r.equity.iloc[-1] == pytest.approx(1_000_000.0, abs=1.0)


# ------------------------------------------------------------------ DAY_RISK entry cutoff with resting orders
def test_day_risk_resting_entry_order_cannot_fill_after_1515():
    """'No new entries after 15:15 ET'. A buy-stop entry submitted at 10:00
    that triggers at 15:40 opens a new position at 15:40: the cutoff is only
    checked at submission, resting entry orders are not cancelled."""
    from atlasbt.engine.testing import InMemoryPortal, synthetic_bars_from_rows
    s = sessions("2024-03-04", 1)[0]
    o, _ = CAL.open_close(s)
    starts = pd.date_range(o, periods=390, freq="1min")
    trig = 370                                           # 15:40 ET
    rows = [(100, 100.05, 99.95, 100)] * trig + [(105, 105.05, 104.95, 105)] * (390 - trig)
    df = synthetic_bars_from_rows(rows, "1m", start_times=starts)
    portal = InMemoryPortal(bars={("XYZ", "1m"): df})

    def script(ctx, st):
        t = ctx.now_et()
        if (t.hour, t.minute) == (10, 0):
            ctx.order("XYZ", 100, OrderType.STOP, stop_price=103.0, tag="entry")
    r, _ = run(script, portal, [s], freq="1m", sizing="DAY_RISK")
    ent = r.fills[r.fills.tag == "entry"]
    late = ent[pd.DatetimeIndex(ent.ts).tz_convert("America/New_York").time > pd.Timestamp("15:15").time()]
    assert late.empty, late[["ts", "quantity", "price"]]


# ------------------------------------------------------------------ margin call side effects
def test_margin_call_does_not_cancel_unrelated_protective_stops():
    """A margin call (from AAA) cancels EVERY working order in the account,
    including BBB's protective GTC stop; BBB then falls through its stop with
    no exit. Brokers liquidate to cure the call; they do not delete the
    customer's other stop orders."""
    ss = sessions("2024-03-04", 7)
    aaa = _flat(2, 100.0) + [(60, 61, 55, 56)] + _flat(4, 56.0)
    bbb = _flat(4, 50.0) + [(40, 41, 39, 40)] * 3
    portal = daily_portal({"AAA": aaa, "BBB": bbb}, ss)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.order("AAA", 18_000)                       # $1.8M on $1M (Reg-T OK)
            ctx.bracket("BBB", 100, stop=45.0)
    r, _ = run(script, portal, ss, universe=["AAA", "BBB"])
    assert r.diagnostics["margin_calls"]
    stops = r.fills[(r.fills.symbol == "BBB") & (r.fills.tag == "stop")]
    # (integrator M11) the exit is kept and shrunk to the post-liquidation position: -74, no short flip
    assert len(stops) == 1 and stops.quantity.iloc[0] == pytest.approx(-74.0)
    bbb_pos = r.positions[r.positions.symbol == "BBB"].quantity
    assert (bbb_pos >= 0).all()


def test_resting_far_sell_limit_does_not_create_buying_power():
    """Pre-trade check simulates ALL pending orders filling at their limit
    (not only exposure-increasing ones). A resting sell limit 1,000 @ $10,000
    on a $100 stock (never fills) adds $10M of phantom cash/equity to the
    shadow account, so a $10M buy on $1M equity is approved and filled (10x)."""
    ss = sessions("2024-03-04", 4)
    from atlasbt.core.types import TimeInForce
    portal = daily_portal({"XYZ": _flat(4)}, ss)

    def script2(ctx, st):
        d = ctx.session_date()
        if d == ss[0]:
            ctx.order("XYZ", 1_000, tag="core")
        if d == ss[1]:
            ctx.order("XYZ", -1_000, OrderType.LIMIT, limit_price=10_000.0, tif=TimeInForce.GTC, tag="far_sell")
            ctx.order("XYZ", 100_000, tag="big")
    r, _ = run(script2, portal, ss)
    big = r.fills[r.fills.tag == "big"].quantity.sum()
    assert big * 100.0 <= 2.0 * 1_000_000.0, f"bought {big} shares (${big * 100:,.0f}) on $1M equity"


# ------------------------------------------------------------------ R multiple
def test_r_multiple_uses_planned_risk_not_slipped_fill():
    """Bracket decided at close 100 with stop 98 (planned risk $2/sh x 100 =
    $200, which is what DAY/SWING_RISK sizing budgets). Entry gaps to 101 and
    the stop fills at 98: loss $300 = -1.5R. The tracker re-bases risk on the
    slipped fill (|101-98| x 100 = $300) and reports -1.0R, hiding gap/slippage
    in every R statistic (expectancy_R is optimistic for losers)."""
    ss = sessions("2024-03-04", 3)
    rows = [(100, 101, 99, 100), (101, 101.5, 97, 97.5), (97.5, 98, 97, 97.5)]
    portal = daily_portal({"XYZ": rows}, ss)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.bracket("XYZ", 100, stop=98.0)
    r, _ = run(script, portal, ss)
    t = r.trades.iloc[0]
    assert t.pnl == pytest.approx(-300.0)
    # (integrator M14: finding INVALID) R is measured from the actual entry fill (STRATEGY_GUIDE 2.4)
    assert t.initial_risk == pytest.approx(300.0)
    assert t.r_multiple == pytest.approx(-1.0)


# ------------------------------------------------------------------ option expiry without an underlying price
def test_short_itm_option_not_expired_worthless_when_settlement_price_missing():
    """DefaultOptionLifecycle: if the underlying close is missing on expiry,
    every option 'expires worthless at 0' (T2 flag). For a short deep-ITM put
    (last mark $30) the liability simply disappears -> optimistic. The last
    known underlying price / option mark should be used instead."""
    from atlasbt.core.types import Right
    from atlasbt.portfolio import Account
    from atlasbt.portfolio.hooks import DefaultOptionLifecycle
    acct = Account(1_000_000.0)
    put = _opt(100.0, Right.PUT, exp="2024-06-21")
    acct.apply_fill(_fill(put, -10, 30.0))
    acct.mark(put.symbol, 30.0)

    class _NoPx:
        def close(self, s, d):
            return float("nan")

        def open(self, s, d):
            return float("nan")
    evs = DefaultOptionLifecycle().at_session_end(pd.Timestamp("2024-06-21"), acct.option_positions(), _NoPx())
    assert evs and evs[0].kind != "EXPIRE_WORTHLESS"


def test_cash_account_buying_power_includes_commissions():
    """Cash account: 'purchases need cash'. Buying exactly cash/price shares is
    approved (the check ignores commissions/fees) and the fill drives the
    cash account's balance below zero (-$67 here; with rf > 0 the account is
    then charged MARGIN_INTEREST at rf + 1.5%)."""
    from atlasbt.portfolio import AccountType
    ss = sessions("2024-03-04", 3)
    portal = daily_portal({"XYZ": _flat(3)}, ss, rf=0.0)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.order("XYZ", int(ctx.cash // 100.0), tag="all_in")
    r, _ = run(script, portal, ss, cfg=zero_cost(fees="IBKR_PRO_TIERED", account_type=AccountType.CASH))
    assert r.diagnostics["cash_account_negative_cash_days"] == 0
