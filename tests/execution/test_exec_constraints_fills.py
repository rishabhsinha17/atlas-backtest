"""Tick rounding, Reg SHO state machine, bar/quote/auction/combo fill models."""
import math

import pandas as pd
import pytest

from atlasbt.core.types import AssetClass, Fidelity, Instrument, Leg, Order, OrderType, Right
from atlasbt.execution.constraints import (
    SSRStateMachine, ShortSaleRules, bar_is_halted, round_limit_for_side, round_stop_for_side,
    round_to_tick, session_allows, valid_tick,
)
from atlasbt.execution.fills import (
    AuctionFillModel, BarFillConfig, BarFillModel, BarView, ComboConfig, ComboFillModel, LegQuote, Path,
    QuoteFillModel, QuoteView, SymbolMarket,
)
from atlasbt.execution.impact import SquareRootImpact
from atlasbt.execution.spread import SpreadConfig, SpreadModel

STK = Instrument("XYZ", AssetClass.STOCK)
ZERO_SPREAD = SpreadConfig(bucket_floors=((math.inf, 0.0),), default_spread=1e-12,
                           tod_open=(), tod_close=(), tick_floor=False)


class Ex:
    def __init__(self, eligible_ns=0):
        self.eligible_ns = eligible_ns
        self.outside_rth = False
        self.activate_at_ns = None
        self.in_bar_from = None
        self.evaluated = False


def run(model, orders, bars, pos0=0.0, ssr=False):
    """Mini driver: returns list of FillDecision; updates position."""
    state = {"q": pos0}
    fills = []
    exs = {o.id: Ex() for o in orders}
    mkt = SymbolMarket(STK, 1e-12, "test", float("nan"), float("nan"), lambda b: ssr, lambda: state["q"])

    def get():
        return [(o, exs[o.id]) for o in orders if abs(o.remaining) > 1e-9 and o.reject_reason == ""]

    def on_fill(d):
        d.order.filled_qty += d.qty
        state["q"] += d.qty
        fills.append(d)
        if d.order.oco_group:
            for o in orders:
                if o is not d.order and o.oco_group == d.order.oco_group:
                    o.quantity = o.filled_qty     # cancel
    rejects = []

    def on_rej(o, r):
        o.reject_reason = r
        rejects.append(r)
    model.simulate(mkt, bars, get, on_fill, on_rej)
    return fills, rejects


def mk(q, t=OrderType.MARKET, lp=None, sp=None, oid=1, oco=None):
    o = Order(STK, q, t, lp, sp, oco_group=oco)
    o.id = oid
    return o


def bar(o, h, l, c, v=1e6, mso=60.0):
    return BarView(0, 60 * 10**9, o, h, l, c, v, "RTH", mso, 300.0)


def model(**kw):
    return BarFillModel(BarFillConfig(**kw), SpreadModel(ZERO_SPREAD), SquareRootImpact(k=0.0))


# --------------------------------------------------------------------------- ticks
def test_tick_rounding_rules():
    assert valid_tick(5.0) == 0.01 and valid_tick(0.5) == 0.0001
    assert round_to_tick(10.004, 0.01) == 10.0 and round_to_tick(10.005, 0.01, "up") == 10.01
    assert round_limit_for_side(10.019, 0.01, True) == 10.01      # buy limit rounds down
    assert round_limit_for_side(10.011, 0.01, False) == 10.02     # sell limit rounds up
    assert round_stop_for_side(10.011, 0.01, True) == 10.02
    assert round_to_tick(0.12345, 0.0001, "down") == 0.1234
    opt = Instrument.option("XYZ", pd.Timestamp("2024-07-19"), 50, Right.CALL, tick_size=0.05)
    assert valid_tick(3.0, opt) == 0.05


# --------------------------------------------------------------------------- SSR
def test_ssr_state_machine_trigger_and_expiry():
    m = SSRStateMachine()
    d1, d2, d3 = pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-05"), pd.Timestamp("2024-03-06")
    t = pd.Timestamp("2024-03-04 16:00", tz="UTC")
    assert not m.on_price("XYZ", d1, t, prior_close=100.0, price_low=90.01)    # -9.99%: no trigger
    assert not m.is_active("XYZ", d1, t)
    assert m.on_price("XYZ", d1, t, prior_close=100.0, price_low=90.0)         # -10%: trigger
    assert not m.is_active("XYZ", d1, t - pd.Timedelta(minutes=1))              # not before trigger time
    assert m.is_active("XYZ", d1, t) and m.state("XYZ", d1) == "active_today"
    assert m.is_active("XYZ", d2, prev_session=d1) and m.state("XYZ", d2, d1) == "active_next_day"
    assert not m.is_active("XYZ", d3, prev_session=d2)                          # expired


def test_short_sale_eras():
    r = ShortSaleRules()
    assert r.era(pd.Timestamp("2005-01-03")) == "TICK_TEST"
    assert r.era(pd.Timestamp("2009-01-05")) == "NONE"
    assert r.era(pd.Timestamp("2011-02-28")) == "RULE201"
    m = SSRStateMachine()
    assert m.is_active("X", pd.Timestamp("2005-01-03"))      # pre-2007 tick test: always restricted
    assert not m.on_price("X", pd.Timestamp("2009-01-05"), pd.Timestamp("2009-01-05 15:00", tz="UTC"), 100, 50)


def test_halt_and_hours_helpers():
    assert bar_is_halted(0.0, "RTH", 10.0) and bar_is_halted(float("nan"), "DAY", 10.0)
    assert not bar_is_halted(0.0, "PRE", 10.0)
    assert bar_is_halted(100.0, "RTH", float("nan"))
    assert session_allows("RTH", False) and not session_allows("PRE", False) and session_allows("POST", True)


# --------------------------------------------------------------------------- path
def test_path_first_reach():
    p = Path([100, 95, 110, 105])
    pos, px, gap = p.first_reach(0, 97, up=False)
    assert not gap and px == 97 and 0 < pos < 1
    pos, px, gap = p.first_reach(0, 101, up=True)
    assert 1 < pos < 2 and px == 101
    assert p.first_reach(0, 111, up=True) is None
    assert p.first_reach(0, 100, up=False)[2] is True     # already at level


# --------------------------------------------------------------------------- bar fills
def test_market_fills_at_open_with_half_spread():
    m = BarFillModel(BarFillConfig(), SpreadModel(SpreadConfig()), SquareRootImpact(k=0.0))
    o = mk(100)
    f, _ = run(m, [o], [bar(50.0, 51, 49, 50.5, mso=120)])
    hs = 0.5 * m.spread.full_spread(1e-12, 50.0, 0.01, 120, 300)
    assert f[0].price == pytest.approx(50.0 * (1 + hs)) and f[0].pos == 0.0 and f[0].liquidity == "TAKER"


def test_gap_through_stop_fills_at_open():
    o = mk(-100, OrderType.STOP, sp=95.0)
    f, _ = run(model(), [o], [bar(90.0, 92, 88, 91)], pos0=100)
    assert f[0].price == pytest.approx(90.0, abs=1e-6)          # gap open, not the stop price
    assert "gapped" in " ".join(f[0].reasons)


def test_stop_inside_range_fills_at_stop():
    o = mk(-100, OrderType.STOP, sp=97.0)
    f, _ = run(model(), [o], [bar(100.0, 101, 96, 99)], pos0=100)
    assert f[0].price == pytest.approx(97.0)


def test_limit_touch_is_not_a_fill_but_trade_through_is():
    o = mk(100, OrderType.LIMIT, lp=95.0)
    f, _ = run(model(), [o], [bar(100.0, 101, 95.0, 99)])            # low == limit: touch only
    assert f == []
    o2 = mk(100, OrderType.LIMIT, lp=95.0, oid=2)
    f, _ = run(model(), [o2], [bar(100.0, 101, 94.99, 99)])          # 1 tick through
    assert len(f) == 1 and f[0].price == 95.0 and f[0].liquidity == "MAKER"


def test_marketable_limit_fills_at_open_capped():
    o = mk(100, OrderType.LIMIT, lp=101.0)
    f, _ = run(model(), [o], [bar(100.0, 102, 99, 101)])
    assert f[0].price == pytest.approx(100.0) and f[0].liquidity == "TAKER"


def test_stop_limit():
    # triggers at 101, limit 101.5 -> marketable at trigger
    o = mk(100, OrderType.STOP_LIMIT, lp=101.5, sp=101.0)
    f, _ = run(model(), [o], [bar(100.0, 103, 99.5, 102)])
    assert f[0].price == pytest.approx(101.0)
    # gaps above the limit -> rests as limit; never trades back through -> no fill
    o2 = mk(100, OrderType.STOP_LIMIT, lp=101.5, sp=101.0, oid=2)
    f, _ = run(model(), [o2], [bar(103.0, 104, 102, 103.5)])
    assert f == []


def test_participation_cap_partial_fill():
    o = mk(5000)
    f, _ = run(model(participation_cap=0.10), [o], [bar(10.0, 10.1, 9.9, 10, v=20_000),
                                                    bar(10.0, 10.1, 9.9, 10, v=20_000),
                                                    bar(10.0, 10.1, 9.9, 10, v=20_000)])
    assert [d.qty for d in f] == [2000, 2000, 1000]
    assert f[0].partial and not f[2].partial


def test_zero_volume_bar_is_halted_no_fill():
    o = mk(100)
    f, _ = run(model(), [o], [bar(10.0, 10, 10, 10, v=0.0)])
    assert f == []


def test_adverse_first_stop_before_target_same_bar():
    stop = mk(-100, OrderType.STOP, sp=98.0, oid=1, oco="b")
    tgt = mk(-100, OrderType.LIMIT, lp=103.0, oid=2, oco="b")
    f, _ = run(model(), [stop, tgt], [bar(100.0, 105, 97, 104)], pos0=100)
    assert len(f) == 1 and f[0].order is stop
    assert f[0].fidelity == Fidelity.T2_PROXY        # ambiguity flagged
    # short position: high first -> buy stop above hit before buy limit below
    stop2 = mk(100, OrderType.STOP, sp=102.0, oid=3, oco="c")
    tgt2 = mk(100, OrderType.LIMIT, lp=97.0, oid=4, oco="c")
    f, _ = run(model(), [stop2, tgt2], [bar(100.0, 105, 95, 99)], pos0=-100)
    assert f[0].order is stop2


def test_ssr_short_sale_cannot_hit_bid():
    m = BarFillModel(BarFillConfig(ssr_mode="ask_limit"), SpreadModel(SpreadConfig()), SquareRootImpact(k=0))
    o = mk(-100)
    o.is_short_sale = True
    f, _ = run(m, [o], [bar(50.0, 50.5, 49, 49.5, mso=120)], ssr=True)
    hs = 0.5 * m.spread.full_spread(1e-12, 50.0, 0.01, 120, 300)
    assert f[0].price == pytest.approx(50.0 * (1 + hs)) and f[0].liquidity == "MAKER"
    o2 = mk(-100, oid=2)
    o2.is_short_sale = True
    f, _ = run(m, [o2], [bar(50.0, 50.0, 49, 49.5, mso=120)], ssr=True)    # never trades above the ask
    assert f == []
    m3 = BarFillModel(BarFillConfig(ssr_mode="reject"), SpreadModel(SpreadConfig()), SquareRootImpact(k=0))
    o3 = mk(-100, oid=3)
    o3.is_short_sale = True
    f, rej = run(m3, [o3], [bar(50.0, 51, 49, 49.5)], ssr=True)
    assert f == [] and rej and "SSR" in rej[0]


def test_impact_increases_buy_price():
    m = BarFillModel(BarFillConfig(), SpreadModel(ZERO_SPREAD), SquareRootImpact(k=0.1))
    o = mk(10_000)
    state = {"q": 0.0}
    mkt = SymbolMarket(STK, 1e-12, "t", 1e7, 0.02, lambda b: False, lambda: state["q"])
    out = []
    m.simulate(mkt, [bar(10.0, 10.2, 9.8, 10, v=1e6)], lambda: [(o, Ex())] if not out else [],
               lambda d: out.append(d), lambda o, r: None)
    exp = 10.0 * (1 + 0.1 * 0.02 * math.sqrt(1e5 / 1e7))
    assert out[0].price == pytest.approx(exp)
    assert out[0].impact_cost == pytest.approx((exp - 10.0) * 10_000, rel=1e-6)


# --------------------------------------------------------------------------- quotes
def test_quote_fill_model():
    qm = QuoteFillModel(SquareRootImpact(k=0.0))
    mkt = SymbolMarket(STK)
    buy = mk(100)
    ex = Ex(eligible_ns=5)
    out = []
    qs = [QuoteView(5, 9.99, 10.01, 500, 500),     # at eligible_at: not usable
          QuoteView(6, 10.00, 10.02, 500, 50),     # ask size 50 -> partial
          QuoteView(7, 10.01, 10.03, 500, 500)]
    qm.simulate(mkt, qs, lambda: [(buy, ex)] if abs(buy.remaining) > 0 else [],
                lambda d: (out.append(d), setattr(buy, "filled_qty", buy.filled_qty + d.qty)), lambda o, r: None)
    assert [(d.qty, d.price, d.ts_ns) for d in out] == [(50, 10.02, 6), (50, 10.03, 7)]
    assert out[0].fidelity == Fidelity.T0_QUOTE


# --------------------------------------------------------------------------- auction
def test_auction_fill():
    am = AuctionFillModel(impact=SquareRootImpact(k=0.0))
    o = mk(100, OrderType.MOO)
    d = am.fill(o, 50.0, 1e6, 0, SymbolMarket(STK), is_open=True, daily_bar=True)
    assert d.price == 50.0 and d.liquidity == "AUCTION" and d.qty == 100
    big = mk(10_000, OrderType.MOC, oid=2)
    d = am.fill(big, 50.0, 1e6, 0, SymbolMarket(STK), is_open=False, daily_bar=True)
    assert d.qty == pytest.approx(0.10 * 0.08 * 1e6) and d.partial


# --------------------------------------------------------------------------- combos
def _opt(k, r=Right.CALL, exp="2024-07-19"):
    return Instrument.option("XYZ", pd.Timestamp(exp), k, r)


def test_combo_natural_and_improvement():
    c1, c2 = _opt(100), _opt(105)
    o = Order(None, 2, OrderType.COMBO, limit_price=2.25, legs=(Leg(c1, 1), Leg(c2, -1)))
    o.id = 9
    legs = [(Leg(c1, 1), LegQuote(c1, 10, 3.0, 3.2, Fidelity.T0_QUOTE)),
            (Leg(c2, -1), LegQuote(c2, 10, 1.0, 1.1, Fidelity.T0_QUOTE))]
    st, fills, net, _ = ComboFillModel().evaluate(o, legs, eligible_ns=5)
    assert st == "FILL" and net == pytest.approx(3.2 - 1.0)
    assert [(f[1], f[2]) for f in fills] == [(2, 3.2), (-2, 1.0)]
    st, _, net, _ = ComboFillModel(ComboConfig(price_improvement=0.5)).evaluate(o, legs, 5)
    assert net == pytest.approx(3.15 - 1.025)
    o.limit_price = 2.0
    assert ComboFillModel().evaluate(o, legs, 5)[0] == "WAIT"


def test_combo_requires_same_snapshot_and_fresh_quotes():
    c1, c2 = _opt(100), _opt(105)
    o = Order(None, 1, OrderType.COMBO, legs=(Leg(c1, 1), Leg(c2, -1)))
    o.id = 1
    mixed = [(Leg(c1, 1), LegQuote(c1, 10, 3.0, 3.2, Fidelity.T0_QUOTE)),
             (Leg(c2, -1), LegQuote(c2, 11, 1.0, 1.1, Fidelity.T0_QUOTE))]
    assert ComboFillModel().evaluate(o, mixed, 5)[0] == "REJECT"
    stale = [(Leg(c1, 1), LegQuote(c1, 5, 3.0, 3.2, Fidelity.T0_QUOTE)),
             (Leg(c2, -1), LegQuote(c2, 5, 1.0, 1.1, Fidelity.T0_QUOTE))]
    assert ComboFillModel().evaluate(o, stale, 5)[0] == "WAIT"    # snapshot not after eligible_at


def test_combo_sell_package_credit():
    p1, p2 = _opt(95, Right.PUT), _opt(90, Right.PUT)
    # sell put spread: package defined as +1 95P / -1 90P; selling it receives the credit
    o = Order(None, -1, OrderType.COMBO, limit_price=1.0, legs=(Leg(p1, 1), Leg(p2, -1)))
    o.id = 1
    legs = [(Leg(p1, 1), LegQuote(p1, 10, 2.0, 2.1, Fidelity.T3_MODELED)),
            (Leg(p2, -1), LegQuote(p2, 10, 0.8, 0.9, Fidelity.T3_MODELED))]
    st, fills, net, _ = ComboFillModel().evaluate(o, legs, 5)
    assert st == "FILL" and net == pytest.approx(2.0 - 0.9)
    assert fills[0][1] == -1 and fills[1][1] == 1 and fills[0][4] == Fidelity.T3_MODELED
