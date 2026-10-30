"""S158-S160 (earnings option strategies) mechanics on SYNTHETIC 5-minute data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.clock import et
from atlasbt.core.types import Fidelity, RunStatus
from atlasbt.strategies.options_events import S158, S159, S160

from .oe_fix import (daily_from, ev, fills, intraday_bars, minute_snaps, option_book, order_key, orders, run, sess)

SESS = sess("2024-03-04", "2024-03-19")
D = pd.Timestamp("2024-03-13")            # Wednesday: AMC release
D1 = pd.Timestamp("2024-03-14")
# European exercise only to keep the per-snapshot surface fits fast (mechanics tests)
EXPIRIES = [(e, "PHYSICAL", "EUROPEAN", None) for e in ("2024-03-15", "2024-03-22", "2024-04-19", "2024-05-17")]


def T(day, hhmm):
    return et(pd.Timestamp(day), hhmm)


def ymd(t):
    return t.tz_convert("America/New_York").tz_localize(None).normalize()


def spec_for(price=lambda t: 100.0, vol=lambda t, K, e, Ty: 0.40, events=(), overrides=None, sessions=SESS,
             snaps_sessions=None):
    intra = intraday_bars(sessions, price, "5m", overrides=overrides)
    snaps = minute_snaps(snaps_sessions or sessions, step_min=5)
    optq, insts = option_book("XYZ", EXPIRIES, np.arange(80.0, 121.0, 1.0), snaps, price, vol, hs=lambda p: 0.02)
    return {"bars": {("XYZ", "5m"): intra, ("XYZ", "1d"): daily_from(intra)}, "option_quotes": optq,
            "instruments": insts, "events": list(events), "universes": {"SP500": ["XYZ"]}}


def amc_events(actual="16:05", sched_timing="AMC", sched_time=None, day=D):
    st = sched_time or ("16:30" if sched_timing == "AMC" else "08:00")
    sday = day if sched_timing == "AMC" else D1
    return [ev("EARNINGS_SCHEDULE", "XYZ", "2024-03-01 00:00", T(sday, st), timing=sched_timing),
            ev("EARNINGS", "XYZ", T(sday, actual), T(sday, actual), timing=sched_timing)]


def combo(res, prefix):
    o = orders(res)
    return o[o["tag"].astype(str).str.startswith(prefix)]


def legs(res, prefix):
    f = fills(res, prefix)
    return sorted(zip(f["symbol"].str[12:13], (f["symbol"].str[13:].astype(int) / 1000.0),
                      f["symbol"].str[6:12], np.sign(f["quantity"])))


# ---------------------------------------------------------------------------- S158
@pytest.mark.parametrize("timing,actual", [("AMC", "16:05"), ("BMO", "07:30")])
def test_synthetic_S158_pre_release_straddle(timing, actual):
    spec = spec_for(events=amc_events(actual=actual, sched_timing=timing))
    r = run(S158(), spec, D, D1)
    op = combo(r, "S158:open")
    assert len(op) == 1 and pd.Timestamp(op["submitted_at"].iloc[0]) == T(D, "15:30")   # last 15:30 before release
    assert set(fills(r, "S158:open")["ts"]) == {T(D, "15:35")}
    assert legs(r, "S158:open") == [("C", 100.0, "240322", 1), ("P", 100.0, "240322", 1)]  # nearest post-release 3-10DTE
    cl = combo(r, "S158:close")
    assert pd.Timestamp(cl["submitted_at"].iloc[0]) == T(D1, "09:45")
    assert set(fills(r, "S158:close")["ts"]) == {T(D1, "09:50")}
    f = fills(r, "S158:open")
    n = int(f["quantity"].abs().iloc[0])
    debit = (f["quantity"] * f["price"]).sum() / n * 100
    assert n * debit <= 0.005 * 1e6 < (n + 1) * debit


@pytest.mark.parametrize("variant", ["during", "already_released", "entry_1500_grid"])
def test_synthetic_S158_near_misses_and_grid(variant):
    if variant == "during":
        evs = [ev("EARNINGS_SCHEDULE", "XYZ", "2024-03-01", T(D, "12:00"), timing="DURING")]
        r = run(S158(), spec_for(events=evs), D, D1)
        assert len(combo(r, "S158")) == 0
    elif variant == "already_released":
        evs = amc_events(actual="15:00")             # published before the 15:30 entry
        r = run(S158(), spec_for(events=evs), D, D1)
        assert len(combo(r, "S158")) == 0
    else:
        r = run(S158(entry="15:00", exit="09:35", dte=10), spec_for(events=amc_events()), D, D1)
        op = combo(r, "S158:open")
        assert pd.Timestamp(op["submitted_at"].iloc[0]) == T(D, "15:00")
        assert pd.Timestamp(combo(r, "S158:close")["submitted_at"].iloc[0]) == T(D1, "09:35")


def test_S158_not_testable_without_intraday_option_quotes():
    spec = spec_for(events=amc_events())
    r = run(S158(), spec, D, D1, option_fidelity=Fidelity.T2_PROXY)
    assert r.status == RunStatus.NOT_TESTABLE
    spec["events"] = spec["events"][1:]                     # no EARNINGS_SCHEDULE
    r = run(S158(), spec, D, D1)
    assert r.status == RunStatus.NOT_TESTABLE and "EARNINGS_SCHEDULE" in r.not_testable_reason


# ---------------------------------------------------------------------------- S159
def test_synthetic_S159_iron_condor_credit_and_exit():
    r = run(S159(), spec_for(events=amc_events()), D, D1)
    op = combo(r, "S159:open")
    assert len(op) == 1 and pd.Timestamp(op["submitted_at"].iloc[0]) == T(D, "15:30")
    assert float(op["limit_price"].iloc[0]) == -0.01                      # credit > 0 enforced at the fill
    lg = legs(r, "S159:open")
    assert {x[2] for x in lg} == {"240322"}
    puts = sorted([(k, s) for rt, k, _, s in lg if rt == "P"])
    calls = sorted([(k, s) for rt, k, _, s in lg if rt == "C"])
    assert puts[0][1] == 1 and puts[1][1] == -1 and calls[0][1] == -1 and calls[1][1] == 1   # wings outside
    assert puts[0][0] < puts[1][0] < 100 < calls[0][0] < calls[1][0]
    f = fills(r, "S159:open")
    assert (f["quantity"] * f["price"]).sum() < 0                          # net credit
    assert pd.Timestamp(combo(r, "S159:close")["submitted_at"].iloc[0]) == T(D1, "09:45")


def test_synthetic_S159_near_miss_wings_collapse():
    # strikes too coarse around the money: wings collapse onto shorts -> no trade
    spec = spec_for(events=amc_events())
    keep = {s: q for s, q in spec["option_quotes"].items() if int(s[13:]) / 1000.0 in (90.0, 100.0, 110.0)}
    spec["option_quotes"] = keep
    r = run(S159(), spec, D, D1)
    assert len(combo(r, "S159")) == 0


# ---------------------------------------------------------------------------- S160
class S160T(S160):
    LOOKBACK = 5


def s160_spec(break_to=101.5, post_vol=0.50, crash_day=None):
    t935, t945, t1015, t930 = T(D1, "09:35"), T(D1, "09:45"), T(D1, "10:15"), T(D1, "09:30")
    d1_end = et(D1 + pd.Timedelta(days=1), "00:00")
    crash = et(pd.Timestamp(crash_day), "00:00") if crash_day is not None else None

    def price(t):
        if crash is not None and t >= crash:
            return 90.0
        if t930 <= t < d1_end:
            if t935 <= t < t945:
                return 101.0                            # opening-range high 101
            return break_to if t >= t1015 else 100.0
        return break_to if t >= d1_end else 100.0

    vol = lambda t, K, e, Ty: post_vol if t >= t930 else 0.30
    return spec_for(price=price, vol=vol, events=amc_events())


def test_synthetic_S160_post_event_put_spread():
    r = run(S160T(), s160_spec(crash_day="2024-03-18"), "2024-03-04", "2024-03-19")
    op = combo(r, "S160:open")
    assert len(op) == 1 and pd.Timestamp(op["submitted_at"].iloc[0]) == T(D1, "10:15")   # first 5m close > OR high
    assert set(fills(r, "S160:open")["ts"]) == {T(D1, "10:20")}
    lg = legs(r, "S160:open")
    assert [x[0] for x in lg] == ["P", "P"] and {x[2] for x in lg} == {"240419"}      # ~30 DTE
    (k_long, _, _, s_long), (k_short, _, _, s_short) = [(k, rt, e, s) for rt, k, e, s in lg]
    assert k_long < k_short and s_long == 1 and s_short == -1
    cl = combo(r, "S160:close")
    assert cl["tag"].iloc[0] == "S160:close:below_5d_low"
    assert pd.Timestamp(cl["submitted_at"].iloc[0]) == et(pd.Timestamp("2024-03-18"), "16:00")
    assert set(fills(r, "S160:close")["ts"]) == {T("2024-03-19", "09:35")}


@pytest.mark.parametrize("kw", [{"break_to": 100.5}, {"post_vol": 0.25}])
def test_synthetic_S160_near_misses(kw):
    r = run(S160T(), s160_spec(**kw), "2024-03-04", "2024-03-15")
    assert len(combo(r, "S160")) == 0


def test_S160_timing_proxy_requirements():
    s = S160(timing_proxy=True)
    opt = [q for q in s.requires if q.kind == "options"]
    assert len(opt) == 1 and not opt[0].critical and opt[0].min_fidelity == Fidelity.T2_PROXY
    r = run(S160T(timing_proxy=True), s160_spec(), "2024-03-04", "2024-03-15")
    op = combo(r, "S160:open")
    assert len(op) == 1 and pd.Timestamp(op["submitted_at"].iloc[0]) == et(D1, "16:00")   # close proxy
    assert any("timing_proxy" in str(x["reason"]) for x in r.fidelity["reasons"])


# ---------------------------------------------------------------------------- lookahead + smoke
@pytest.mark.parametrize("cls,spec_fn,start,end,cut", [
    (S158, lambda: spec_for(events=amc_events()), D, D1, T(D, "15:45")),
    (S159, lambda: spec_for(events=amc_events()), D, D1, T(D, "15:45")),
    (S160T, lambda: s160_spec(crash_day="2024-03-18"), "2024-03-04", "2024-03-19", T(D1, "11:00")),
])
def test_lookahead_earnings(cls, spec_fn, start, end, cut):
    spec = spec_fn()
    full = run(cls(), spec, start, end)
    trunc = run(cls(), spec, start, end, cut=cut)
    a, b = order_key(orders(full), cut), order_key(orders(trunc), cut)
    assert a == b and len(a) >= 1


@pytest.mark.parametrize("cls", [S158, S159, S160T])
def test_smoke_earnings(cls):
    r = run(cls(), spec_for(events=amc_events()), "2024-03-11", "2024-03-15")
    assert r.status == RunStatus.OK
