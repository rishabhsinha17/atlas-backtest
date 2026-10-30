"""RED-TEAM findings for atlasbt.engine / atlasbt.execution (time, information, execution realism).

Every test asserts the REALISTIC behaviour and therefore FAILS on the code as reviewed.
All data is SYNTHETIC and only exercises mechanics.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from rt_ee_helpers import CAL, daily_portal, intraday_frame, make, sessions, zero_cost  # noqa: E402

from atlasbt.core.clock import et
from atlasbt.core.types import Instrument, Leg, Order, OrderType, Right
from atlasbt.engine import BacktestConfig, Backtester
from atlasbt.engine.testing import InMemoryPortal, synthetic_bars_from_rows
from atlasbt.execution.fills import BarFillConfig
from atlasbt.execution.spread import SpreadConfig


# ============================================================ F1 (CRITICAL)
def test_limit_entry_bracket_target_not_assumed_after_entry_same_bar():
    """Flat + buy-limit bracket: the engine assumes open->HIGH->low->close to fill the entry
    late (at the low), then RESTARTS the path from the entry price (entry->low->HIGH->close),
    so the bar's high is used a second time, after the entry, and the target fills.
    The high cannot be both before and after the entry; with daily bars only nothing
    shows that price rose to 105 after the buy at 95."""
    rows = [(100, 101, 99, 100), (100, 106, 94, 96), (96, 97, 95, 96)]
    portal, ss = daily_portal(rows)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.bracket("XYZ", 100, entry_type=OrderType.LIMIT, entry_limit=95, stop=90, target=105)
    r = Backtester(make()(script), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    day2 = r.fills[(r.fills.ts >= CAL.open_close(ss[1])[0]) & (r.fills.ts <= CAL.open_close(ss[1])[1])]
    assert list(day2.tag) == ["entry"], (
        f"target filled on the entry bar using the bar high again after a limit entry: {list(day2.tag)} "
        f"prices={list(day2.price)}")


# ============================================================ F2 (CRITICAL)
def test_combo_stock_leg_not_priced_from_signal_bar_close():
    """Buy-write submitted at 10:00 (end of the 09:59 bar, close 100). The stock trades at 105
    from 10:00 on. The option snapshot is 10:00:30, but the stock leg is priced from the close
    of the bar that ENDED at 10:00 (known when the order was sent) and stamped 10:00:30."""
    s = pd.Timestamp("2024-03-04")
    o, c = CAL.open_close(s)
    starts = pd.date_range(o, c, freq="1min", inclusive="left")
    t_sub = et(s, "10:00")
    rows = [((100.0 if st < t_sub else 105.0),) for st in starts]
    rows = [(a, a + 0.01, a - 0.01, a, 10_000.0) for (a,) in rows]
    df = synthetic_bars_from_rows(rows, "1m", start_times=starts)
    call = Instrument.option("XYZ", pd.Timestamp("2024-03-15"), 100.0, Right.CALL)
    q = pd.DataFrame({"bid": 2.0, "ask": 2.1}, index=starts + pd.Timedelta(seconds=30))
    portal = InMemoryPortal(bars={("XYZ", "1m"): df}, option_quotes={call.symbol: q},
                            instruments={call.symbol: call})
    stk = portal.instrument("XYZ")

    def script(ctx, st):
        if ctx.now == t_sub:
            ctx.submit(Order(None, 1, OrderType.COMBO, legs=(Leg(stk, 100), Leg(call, -1)), tag="buywrite"))
    r = Backtester(make("1m")(script), portal, zero_cost()).run(s, s, raise_errors=True)
    stock_fill = r.fills[r.fills.symbol == "XYZ"].iloc[0]
    assert stock_fill.price >= 105.0 - 1e-9, (
        f"stock leg bought at {stock_fill.price} (close of the bar that ended at submission), "
        f"market after submission was 105")


# ============================================================ F3 (MAJOR)
def test_option_order_fills_at_next_snapshot_not_latest():
    """Daily strategy buys a call at the close; snapshots exist at 09:31 and 16:00 each day.
    The next eligible quote is the 09:31 one (ask 2.1); the engine uses the LATEST snapshot
    <= now (16:00, ask 3.1) - a full session later."""
    ss = sessions("2024-03-04", 3)
    under = synthetic_bars_from_rows([(100, 101, 99, 100)] * 3, "1d", session_dates=ss)
    call = Instrument.option("XYZ", pd.Timestamp("2024-03-15"), 100.0, Right.CALL)
    idx, bids = [], []
    for s in ss:
        idx += [et(s, "09:31"), et(s, "16:00")]
        bids += [2.0, 3.0]
    q = pd.DataFrame({"bid": bids, "ask": [b + 0.1 for b in bids]}, index=pd.DatetimeIndex(idx))
    portal = InMemoryPortal(bars={("XYZ", "1d"): under}, option_quotes={call.symbol: q},
                            instruments={call.symbol: call})

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.order(call, 1, tag="buy_call")
    r = Backtester(make()(script), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    f = r.fills.iloc[0]
    assert f.ts == et(ss[1], "09:31") and f.price == pytest.approx(2.1), (f.ts, f.price)


# ============================================================ F4 (MAJOR)
def test_quote_mode_rth_day_order_does_not_fill_after_hours():
    """RTH-only DAY market order sent at 16:00 is valid for the NEXT session. Quote mode fills
    it at a 16:00:05 after-hours quote of the same day (no session/outside_rth/validity check)."""
    s1, s2 = pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-05")
    df = pd.concat([intraday_frame(s, "1m") for s in (s1, s2)])
    ts = list(pd.date_range(*CAL.open_close(s1), freq="1min", inclusive="left") + pd.Timedelta(seconds=10))
    ts += [et(s1, "16:00") + pd.Timedelta(seconds=5), et(s2, "07:00")]
    ts += list(pd.date_range(*CAL.open_close(s2), freq="1min", inclusive="left") + pd.Timedelta(seconds=10))
    qt = pd.DataFrame({"bid": 99.99, "ask": 100.01, "bid_size": 1000.0, "ask_size": 1000.0},
                      index=pd.DatetimeIndex(ts))
    qt.loc[et(s1, "16:00") + pd.Timedelta(seconds=5), ["bid", "ask"]] = [95.0, 95.5]
    qt.loc[et(s2, "07:00"), ["bid", "ask"]] = [96.0, 96.4]
    portal = InMemoryPortal(bars={("XYZ", "1m"): df}, quotes={"XYZ": qt})

    def script(ctx, st):
        if ctx.now == et(s1, "16:00"):
            ctx.order("XYZ", 100, tag="day_mkt_after_close")
    r = Backtester(make("1m")(script), portal, zero_cost()).run(s1, s2, raise_errors=True)
    f = r.fills.iloc[0]
    o2, _ = CAL.open_close(s2)
    assert f.ts > o2, f"RTH-only order filled at {f.ts} @ {f.price} (extended-hours quote, before its session)"


# ============================================================ F5 (MAJOR)
def test_quote_mode_displayed_size_not_reused_by_several_orders():
    """Ask 100.01 x 100 shares. Three 100-share market orders all fill 100 at 100.01 from the
    same quote: 300 shares taken from 100 displayed."""
    s = pd.Timestamp("2024-03-04")
    df = intraday_frame(s, "1m")
    o, c = CAL.open_close(s)
    t_sub = et(s, "10:00")
    ts = pd.date_range(o, c, freq="1min", inclusive="left") + pd.Timedelta(seconds=10)
    qt = pd.DataFrame({"bid": 99.99, "ask": 100.01, "bid_size": 100.0, "ask_size": 100.0}, index=ts)
    qt.loc[qt.index > t_sub + pd.Timedelta(seconds=10), "ask"] = 101.0
    portal = InMemoryPortal(bars={("XYZ", "1m"): df}, quotes={"XYZ": qt})

    def script(ctx, st):
        if ctx.now == t_sub:
            for tag in "abc":
                ctx.order("XYZ", 100, tag=tag)
    r = Backtester(make("1m")(script), portal, zero_cost()).run(s, s, raise_errors=True)
    first_quote = r.fills[r.fills.ts == t_sub + pd.Timedelta(seconds=10)]
    assert first_quote.quantity.sum() <= 100, (
        f"{first_quote.quantity.sum()} shares filled against 100 displayed at {first_quote.price.unique()}")


# ============================================================ F6 (MAJOR)
def test_day_risk_flat_by_on_early_close_day():
    """2024-11-29 closes at 13:00 ET. DAY_RISK must be flat before that close; the engine's
    fixed 15:55 ET flat_by never fires, the position is carried over the weekend and only
    flattened on Monday 15:55."""
    s1, s2 = pd.Timestamp("2024-11-29"), pd.Timestamp("2024-12-02")
    assert CAL.is_early_close(s1)
    df = pd.concat([intraday_frame(s, "1m") for s in (s1, s2)])
    portal = InMemoryPortal(bars={("XYZ", "1m"): df})

    def script(ctx, st):
        if ctx.now == et(s1, "10:00"):
            ctx.order("XYZ", 100)
    r = Backtester(make("1m", sizing="DAY_RISK")(script), portal, zero_cost()).run(s1, s2, raise_errors=True)
    eod = r.positions[r.positions.ts == CAL.open_close(s1)[1]]
    assert eod.quantity.abs().sum() == 0, f"DAY_RISK position held through the early close: {eod.quantity.tolist()}"


def test_day_risk_no_new_entries_on_early_close_day():
    """On a 13:00 close the 15:15 cutoff never arrives, so entries are accepted up to the bell."""
    s1 = pd.Timestamp("2024-11-29")
    portal = InMemoryPortal(bars={("XYZ", "1m"): intraday_frame(s1, "1m")})

    def script(ctx, st):
        if ctx.now == et(s1, "12:58"):
            st.seen.append(ctx.order("XYZ", 100).status.value)
    st = make("1m", sizing="DAY_RISK")(script)
    Backtester(st, portal, zero_cost()).run(s1, s1, raise_errors=True)
    assert st.seen == ["REJECTED"], f"new entry 2 minutes before an early close accepted: {st.seen}"


def test_day_risk_flat_by_with_30m_bars_fills_same_day():
    """With 30m bars the first step >= 15:55 is 16:00; flatten orders are then valid for the
    NEXT session and fill at its open: the position is held overnight."""
    s1, s2 = pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-05")
    df = pd.concat([intraday_frame(s, "30m") for s in (s1, s2)])
    portal = InMemoryPortal(bars={("XYZ", "30m"): df})

    def script(ctx, st):
        if ctx.now == et(s1, "10:30"):
            ctx.order("XYZ", 100)
    r = Backtester(make("30m", sizing="DAY_RISK")(script), portal, zero_cost()).run(s1, s2, raise_errors=True)
    eod = r.positions[r.positions.ts == CAL.open_close(s1)[1]]
    assert eod.quantity.abs().sum() == 0, f"held overnight: {eod.quantity.tolist()}; fills={r.fills[['ts','tag']].values.tolist()}"


# ============================================================ F7 (MAJOR)
def test_day_risk_flat_after_strategy_exit_races_flatten():
    """At 15:55 the engine submits the flatten sell; the strategy (still seeing +100 because the
    flatten has not filled) also sends a non-increasing -100. Both fill -> short 100 carried
    overnight; flat_by is one-shot and never re-checks."""
    s1, s2 = pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-05")
    df = pd.concat([intraday_frame(s, "1m") for s in (s1, s2)])
    portal = InMemoryPortal(bars={("XYZ", "1m"): df})

    def script(ctx, st):
        if ctx.session_date() != s1:
            return
        if ctx.now == et(s1, "10:00"):
            ctx.order("XYZ", 100)
        if ctx.now == et(s1, "15:55"):
            ctx.order("XYZ", -100, tag="own_exit")
    r = Backtester(make("1m", sizing="DAY_RISK")(script), portal, zero_cost()).run(s1, s2, raise_errors=True)
    eod = r.positions[r.positions.ts == CAL.open_close(s1)[1]]
    assert eod.quantity.abs().sum() == 0, f"position after flat_by: {eod.quantity.tolist()}"


def test_day_risk_no_new_short_after_cutoff_via_pending_sells():
    """Long 100 at 15:30: two -100 sells each look 'non-increasing' against the CURRENT position
    (pending orders ignored), so a new short entry is opened after 15:15."""
    s1 = pd.Timestamp("2024-03-04")
    portal = InMemoryPortal(bars={("XYZ", "1m"): intraday_frame(s1, "1m")})

    def script(ctx, st):
        if ctx.now == et(s1, "10:00"):
            ctx.order("XYZ", 100)
        if ctx.now == et(s1, "15:30"):
            ctx.order("XYZ", -100, tag="x1")
            ctx.order("XYZ", -100, tag="x2")
    r = Backtester(make("1m", sizing="DAY_RISK")(script), portal, zero_cost()).run(s1, s1, raise_errors=True)
    assert "x2" not in set(r.fills.tag), "second sell opened a new short after 15:15"


def test_day_risk_resting_entry_not_filled_after_cutoff():
    """Buy-stop entry placed at 10:00 triggers at 15:30: a new entry after 15:15 ET."""
    s1 = pd.Timestamp("2024-03-04")
    o, c = CAL.open_close(s1)
    starts = pd.date_range(o, c, freq="1min", inclusive="left")
    rows = [((100, 100.05, 99.95, 100) if st < et(s1, "15:30") else (101, 101.05, 100.95, 101)) + (10_000.0,)
            for st in starts]
    portal = InMemoryPortal(bars={("XYZ", "1m"): synthetic_bars_from_rows(rows, "1m", start_times=starts)})

    def script(ctx, st):
        if ctx.now == et(s1, "10:00"):
            ctx.order("XYZ", 100, OrderType.STOP, stop_price=100.5, tag="stop_entry")
    r = Backtester(make("1m", sizing="DAY_RISK")(script), portal, zero_cost()).run(s1, s1, raise_errors=True)
    late = r.fills[(r.fills.tag == "stop_entry") & (r.fills.ts > et(s1, "15:15"))]
    assert late.empty, f"entry filled after 15:15: {late.ts.tolist()}"


# ============================================================ F8 (MAJOR)
def test_nan_volume_extended_hours_bar_does_not_fill_unlimited():
    """POST bars with NaN volume: participation cap becomes infinite and 1,000,000 shares fill
    in a single after-hours minute (RTH/DAY NaN bars are treated as halted, PRE/POST are not)."""
    s = pd.Timestamp("2024-03-04")
    rth = intraday_frame(s, "1m")
    _, c = CAL.open_close(s)
    pst = pd.date_range(c, et(s, "20:00"), freq="1min", inclusive="left")
    post = synthetic_bars_from_rows([(100, 100.05, 99.95, 100, np.nan)] * len(pst), "1m", start_times=pst)
    post["session"] = "POST"
    portal = InMemoryPortal(bars={("XYZ", "1m"): pd.concat([rth, post])})

    def script(ctx, st):
        if ctx.now == et(s, "16:00"):
            ctx.submit(Order(ctx.instrument("XYZ"), 1_000_000, tag="huge_post"), outside_rth=True)
    cfg = zero_cost(check_margin_on_submit=False, initial_cash=1e9)
    r = Backtester(make("1m", sessions=("RTH", "POST"))(script), portal, cfg).run(s, s, raise_errors=True)
    assert r.fills.quantity.sum() == 0, f"filled {r.fills.quantity.sum():,.0f} shares on NaN-volume POST bars"


# ============================================================ F9 (MINOR)
def test_bar_fill_prices_on_tick_grid_rounded_adversely():
    """Taker fills are base*(1+hs+imp) with no tick rounding: sub-penny prints for a $33 stock."""
    rows = [(33.33, 33.33, 33.33, 33.33)] * 30          # zero range -> default 20bp spread
    portal, ss = daily_portal(rows, start="2024-01-02")

    def script(ctx, st):
        if ctx.session_date() == ss[25]:
            ctx.order("XYZ", 100)
    r = Backtester(make()(script), portal, BacktestConfig()).run(ss[0], ss[-1], raise_errors=True)
    px = r.fills.price.iloc[0]
    assert abs(px * 100 - round(px * 100)) < 1e-6 and px >= 33.33, f"buy filled off the penny grid at {px!r}"


# ============================================================ F10 (MINOR)
def test_same_bar_exit_timestamp_not_before_entry():
    """After a late limit entry the path is restarted at the bar START, so the stop exit is
    time-stamped hours BEFORE the entry it closes."""
    rows = [(100, 101, 99, 100), (100, 106, 94, 96), (96, 97, 95, 96)]
    portal, ss = daily_portal(rows)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            ctx.bracket("XYZ", 100, entry_type=OrderType.LIMIT, entry_limit=95, stop=94.5)
    r = Backtester(make()(script), portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    e = r.fills[r.fills.tag == "entry"].iloc[0]
    x = r.fills[r.fills.tag == "stop"].iloc[0]
    assert x.ts >= e.ts, f"exit {x.ts} before entry {e.ts}"


# ============================================================ F11 (MINOR)
def test_short_sale_flag_recomputed_ignores_same_bar_sells():
    """Long 100; two -100 market sells. Submit-time correctly flags the second as a short sale,
    but _process_fills recomputes is_short_sale from the CURRENT position for each order
    (100-100 = 0 -> False), so the second sale fills as a long sale on an SSR day (ssr_mode
    'reject')."""
    rows = [(100, 101, 99, 100)] * 25 + [(100, 100, 85, 88), (88, 90, 86, 89), (89, 91, 87, 90)]
    portal, ss = daily_portal(rows, start="2024-01-02")

    def script(ctx, st):
        if ctx.session_date() == ss[20]:
            ctx.order("XYZ", 100)
        if ctx.session_date() == ss[25]:           # SSR triggered on ss[25] -> active on ss[26]
            ctx.order("XYZ", -100, tag="s1")
            ctx.order("XYZ", -100, tag="s2")
    r = Backtester(make()(script), portal, zero_cost(bar_fill=BarFillConfig(ssr_mode="reject"))).run(
        ss[0], ss[-1], raise_errors=True)
    assert "s2" not in set(r.fills.tag), "short sale hit the bid while Rule 201 was active"


# ============================================================ F12 (MINOR)
def test_pit_portal_has_no_escape_hatch():
    """ctx.portal()._p and ctx._bt.md hand the strategy the raw portal / full preloaded arrays."""
    rows = [(100, 101, 99, 100), (110, 111, 109, 110), (120, 121, 119, 120)]
    portal, ss = daily_portal(rows)

    def script(ctx, st):
        if ctx.session_date() == ss[0]:
            raw = getattr(ctx.portal(), "_p", None)
            if raw is not None:
                fut = raw.bars("XYZ", "1d", ctx.now, ctx.now + pd.Timedelta(days=10),
                               as_of=ctx.now + pd.Timedelta(days=10))
                st.seen.append(list(fut.close))
    st = make()(script)
    Backtester(st, portal, zero_cost()).run(ss[0], ss[-1], raise_errors=True)
    assert not st.seen or len(st.seen[0]) == 0, f"strategy read future closes {st.seen[0]} via ctx.portal()._p"


# ============================================================ F13 (MINOR)
def test_marketable_limit_does_not_waive_modelled_impact():
    """Buy LIMIT 100.15, mid 100, est. half-spread 0.10%, modelled impact 0.20% for 1,000 sh ->
    modelled cost 100.30 > limit. (Integrator F13 correction: the original assertion demanded a
    fill above the limit.) A marketable limit may only take the size whose modelled price is
    within the limit, at that modelled price; the rest rests passively (or nothing fills)."""
    from atlasbt.core.types import AssetClass
    from atlasbt.execution.fills import BarFillModel, BarView, SymbolMarket
    from atlasbt.execution.spread import SpreadModel
    m = BarFillModel(BarFillConfig(), SpreadModel(SpreadConfig(tod_open=(), tod_close=())))
    inst = Instrument("XYZ", AssetClass.STOCK)
    pos = {"q": 0.0}
    mkt = SymbolMarket(inst, spread_estimate=0.002, spread_source="Corwin-Schultz", adv_dollar=100_000.0, sigma=0.02,
                       position=lambda: pos["q"])
    o = Order(inst, 1000, OrderType.LIMIT, limit_price=100.15)
    o.id = 1

    class _Ex:
        eligible_ns = 0
        outside_rth = False
        activate_at_ns = None
        in_bar_from = None
        evaluated = False

    ex = _Ex()
    fills = []

    def on_fill(d):
        o.filled_qty += d.qty
        pos["q"] += d.qty
        fills.append(d)
    bar = BarView(0, 60_000_000_000, 100.0, 100.0, 100.0, 100.0, 1e6, "DAY")   # never trades through 100.14
    m.simulate(mkt, [bar], lambda: [(o, ex)] if abs(o.remaining) > 1e-9 else [], on_fill, lambda o_, r_: None)
    taker = [d for d in fills if d.liquidity == "TAKER"]
    if not taker:
        return                                     # no fill is acceptable
    q = sum(d.qty for d in taker)
    px = taker[0].price
    modelled = 100.0 * (1 + 0.001 + m.impact.impact_fraction(q * 100.0, 100_000.0, 0.02))
    assert q < 1000 and px <= 100.15 + 1e-9
    assert px == pytest.approx(modelled, abs=0.01)     # modelled price (rounded up to the tick)
