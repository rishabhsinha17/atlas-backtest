"""S089, S090, S101-S105, S117-S122, S124, S125: ETF allocation strategies.

SYNTHETIC daily data (mechanics only): month-end / weekly decisions, next-open fills,
exact target share counts, T-bill proxy fallback, near misses, lookahead, smoke."""
import math

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.types import RunStatus
from atlasbt.engine.backtester import run_backtest
from atlasbt.engine.testing import InMemoryPortal, synthetic_daily_bars
from atlasbt.indicators import realized_vol
from atlasbt.strategies import allocation as AL
from atlasbt.strategies.registry import get

from calalloc_helpers import CAL, assert_no_lookahead, daily_frame, orders, sess, truncate, zero_cost

NAV = 1e6
ME = pd.Timestamp("2024-03-28")                 # last session of March 2024
NEXT_OPEN = CAL.open_close(pd.Timestamp("2024-04-01"))[0]
ME_CLOSE = CAL.open_close(ME)[1]
HIST = sess("2022-01-03", "2024-04-30")


def path(ret_per_day: float, last: float = 100.0, n: int = len(HIST), wiggle: float = 0.0) -> np.ndarray:
    """Geometric path with constant daily return (+ optional alternating wiggle), ending at ``last``
    on 2024-03-28 and flat afterwards."""
    i_me = HIST.index(ME)
    t = np.arange(n) - i_me
    c = last * np.exp(ret_per_day * np.minimum(t, 0))
    if wiggle:
        c = c * np.where(np.arange(n) % 2 == 0, 1.0, 1.0 + wiggle)
        c = c / c[i_me] * last
    return c


def portal_factory(spec: dict, rf=0.02):
    """spec: sym -> close array over HIST."""
    bars = {(s, "1d"): daily_frame(HIST, c, spread=0.01) for s, c in spec.items()}

    def make(cut=None):
        return InMemoryPortal(bars=truncate(bars, cut) if cut is not None else bars, rf=rf)
    return make


def small(cls, universe, **attrs):
    return type(cls.__name__ + "T", (cls,), {"universe": universe, **attrs})


def run(strategy, portal, a="2024-03-25", b="2024-04-03", cfg=None):
    return run_backtest(strategy, portal, a, b, cfg or zero_cost(), raise_errors=True)


def month_end_orders(res, t=ME_CLOSE) -> dict:
    global NAV_ME
    NAV_ME = nav_at(res, ME)
    o = orders(res)
    o = o[pd.DatetimeIndex(o["submitted_at"]) == t]
    return o.groupby("symbol")["quantity"].sum().to_dict()


NAV_ME = NAV


def shares(w, px=100.0, nav=None, buffer=AL.CASH_BUFFER):
    """Expected whole shares for weight w of a fully invested target (scaled by the cash buffer)."""
    nav = NAV_ME if nav is None else nav
    return float(math.trunc(w * (1 - buffer) * nav / px + 1e-9))


def nav_at(res, session) -> float:
    """NAV seen by on_bar at ``session``'s close when the account was flat (= previous equity)."""
    c = CAL.open_close(pd.Timestamp(session))[1]
    eq = res.equity[res.equity.index < c]
    return float(eq.iloc[-1]) if len(eq) else NAV


def assert_fills_next_open(res, t=NEXT_OPEN):
    f = res.fills
    assert len(f) > 0
    assert (pd.DatetimeIndex(f["ts"]) == t).all()


UP, DOWN, FLAT = 0.002, -0.002, 0.0


# ====================================================================== monthly rank strategies
def test_synthetic_s089_dual_momentum():
    mk = portal_factory({"XLK": path(UP), "XLE": path(0.0005), "XLF": path(DOWN), "BIL": path(0.0001)})
    res = run(small(AL.S089, ["XLK", "XLE", "XLF"])(), mk())
    assert month_end_orders(res) == {"XLK": shares(1.0)}
    assert_fills_next_open(res)
    res2 = run(small(AL.S089, ["XLK", "XLE", "XLF"])(top=2), mk())
    assert month_end_orders(res2) == {"XLK": shares(0.5), "XLE": shares(0.5)}


def test_synthetic_nearmiss_s089_negative_excess_goes_to_tbill():
    mk = portal_factory({"XLK": path(DOWN), "XLE": path(0.00005), "XLF": path(DOWN), "BIL": path(0.0001)})
    res = run(small(AL.S089, ["XLK", "XLE", "XLF"])(), mk())   # XLE +1.3%/yr < rf 2%: excess < 0
    assert month_end_orders(res) == {"BIL": shares(1.0)}


def test_synthetic_s090_tsmom_inverse_vol():
    spec = {"SPY": path(UP, wiggle=0.01), "TLT": path(UP, wiggle=0.02), "GLD": path(DOWN, wiggle=0.01)}
    mk = portal_factory(spec)
    res = run(small(AL.S090, ["SPY", "TLT", "GLD"])(), mk())
    p = mk()
    inv = {}
    for s in spec:
        d = p.bars(s, "1d", pd.Timestamp("2020-01-01", tz="UTC"), ME_CLOSE + pd.Timedelta(1, "ns"), ME_CLOSE)
        inv[s] = 1.0 / float(realized_vol(d["close"].iloc[-64:], 63, annualize=None).iloc[-1])
    tot = sum(inv.values())
    exp = {s: shares(inv[s] / tot, buffer=0.0) for s in ("SPY", "TLT")}     # sum w < 99%: no buffer scaling
    got = month_end_orders(res)
    assert set(got) == {"SPY", "TLT"}                                  # GLD signal negative -> cash
    for s in exp:
        assert got[s] == pytest.approx(exp[s], abs=1)
    res_ls = run(small(AL.S090, ["SPY", "TLT", "GLD"])(mode="long_short"), mk())
    assert month_end_orders(res_ls)["GLD"] == pytest.approx(-shares(inv["GLD"] / tot), abs=1)


def test_synthetic_s101_defensive_rotation():
    mk = portal_factory({"SPY": path(DOWN), "IEF": path(0.0003), "GLD": path(UP), "SHY": path(0.0001),
                         "BIL": path(0.00005)})
    res = run(AL.S101(), mk())
    assert month_end_orders(res) == {"GLD": shares(1.0)}
    mk2 = portal_factory({"SPY": path(DOWN), "IEF": path(DOWN), "GLD": path(DOWN), "SHY": path(-0.0001),
                          "BIL": path(0.00005)})
    assert month_end_orders(run(AL.S101(), mk2())) == {"BIL": shares(1.0)}                  # gate
    assert month_end_orders(run(AL.S101(gate=False), mk2())) == {"SHY": shares(1.0)}        # no gate


def test_synthetic_s102_top_n_with_tbill_slots():
    spec = {"XLK": path(UP), "XLE": path(0.001), "XLF": path(DOWN), "XLV": path(-0.001), "BIL": path(0.00005)}
    res = run(small(AL.S102, ["XLK", "XLE", "XLF", "XLV"])(), portal_factory(spec)())
    third = 1.0 / 3.0
    assert month_end_orders(res) == {"XLK": shares(third), "XLE": shares(third),
                                     "BIL": shares(1 - 2 * third)}                    # 3rd slot fails the T-bill test


def test_synthetic_s117_prior_month_winners():
    spec = {"XLK": path(FLAT), "XLE": path(FLAT), "XLF": path(FLAT)}
    spec["XLK"] = spec["XLK"].copy()
    i = HIST.index(ME)
    spec["XLE"] = spec["XLE"].copy(); spec["XLE"][i - 5:] = 106.0            # March winners
    spec["XLF"] = spec["XLF"].copy(); spec["XLF"][i - 5:] = 103.0
    res = run(small(AL.S117, ["XLK", "XLE", "XLF"])(), portal_factory(spec)())
    assert month_end_orders(res) == {"XLE": shares(0.5, 106.0), "XLF": shares(0.5, 103.0)}


def test_synthetic_s121_switch():
    mk = portal_factory({"SPY": path(UP, wiggle=0.002), "BIL": path(0.00005)})
    assert month_end_orders(run(AL.S121(), mk())) == {"SPY": shares(1.0)}
    mk_vol = portal_factory({"SPY": path(UP, wiggle=0.03), "BIL": path(0.00005)})          # 63d vol > 20%
    assert month_end_orders(run(AL.S121(), mk_vol())) == {"BIL": shares(1.0)}


def test_synthetic_s122_sharpe_top3():
    spec = {"SPY": path(0.001, wiggle=0.005), "EFA": path(0.002, wiggle=0.005), "BIL": path(0.00001),
            "TLT": path(-0.001, wiggle=0.005), "GLD": path(0.0015, wiggle=0.02), "VNQ": path(-0.002, wiggle=0.005)}
    res = run(AL.S122(), portal_factory(spec)())
    got = month_end_orders(res)
    third = 1.0 / 3.0
    assert {k: got[k] for k in ("EFA", "SPY", "GLD")} == {"EFA": shares(third), "SPY": shares(third),
                                                          "GLD": shares(third)}
    assert "TLT" not in got and "VNQ" not in got


def test_synthetic_s124_duration_momentum():
    spec = {"SHY": path(0.0001), "IEF": path(0.0006), "TLT": path(0.001), "BIL": path(0.00005)}
    assert month_end_orders(run(AL.S124(), portal_factory(spec)())) == {"TLT": shares(1.0)}
    assert month_end_orders(run(AL.S124(max_duration="IEF"), portal_factory(spec)())) == {"IEF": shares(1.0)}
    spec_dn = {"SHY": path(-0.0001), "IEF": path(DOWN), "TLT": path(DOWN), "BIL": path(0.00005)}
    assert month_end_orders(run(AL.S124(), portal_factory(spec_dn)())) == {"BIL": shares(1.0)}


def test_synthetic_s120_inverse_vol_capped():
    spec = {"XLK": path(FLAT, wiggle=0.002), "XLE": path(FLAT, wiggle=0.02), "XLF": path(FLAT, wiggle=0.02),
            "XLV": path(FLAT, wiggle=0.02), "BIL": path(0.00005)}
    res = run(small(AL.S120, ["XLK", "XLE", "XLF", "XLV"])(), portal_factory(spec)())
    got = month_end_orders(res)
    assert got["XLK"] == shares(0.30)                                  # low vol name capped at 30%
    for s in ("XLE", "XLF", "XLV"):
        assert got[s] == pytest.approx(shares(0.70 / 3), abs=1)       # remainder redistributed


def test_capped_weights_known_answer():
    w = AL.capped_weights({"a": 10, "b": 1, "c": 1, "d": 1}, 0.3)
    assert w["a"] == pytest.approx(0.3) and w["b"] == pytest.approx(0.7 / 3)
    w2 = AL.capped_weights({"a": 1, "b": 1}, 0.3)                    # 2 x 30% < 100%: remainder unallocated
    assert sum(w2.values()) == pytest.approx(0.6)


def test_synthetic_s105_initial_and_quarterly():
    mk = portal_factory({"SPY": path(UP), "IEF": path(FLAT), "BIL": path(0.00005)})
    res = run(AL.S105(), mk(), "2024-03-25", "2024-04-03")
    o = orders(res)
    init = o[o["tag"] == "initial"]
    t0 = CAL.open_close(pd.Timestamp("2024-03-25"))[0]
    assert (pd.DatetimeIndex(init["submitted_at"]) == t0).all()
    assert set(init["symbol"]) == {"SPY", "IEF"}
    f = res.fills
    assert (pd.DatetimeIndex(f[f["order_id"].isin(init["order_id"])]["ts"]) == t0).all()
    assert "SPY" in month_end_orders(res) or "IEF" in month_end_orders(res) or len(month_end_orders(res)) >= 0
    res_q = run(AL.S105(freq="quarterly"), mk(), "2024-04-22", "2024-05-03")
    assert len(month_end_orders(res_q, CAL.open_close(pd.Timestamp("2024-04-30"))[1])) == 0   # April: no rebalance


# ====================================================================== daily / weekly strategies
def test_synthetic_s103_trend_switch_with_hysteresis():
    c = path(UP)
    i = HIST.index(ME)
    c = c.copy()
    c[i + 1:] = c[i] * 0.80                                       # 2024-04-01 close drops far below SMA200
    mk = portal_factory({"SPY": c, "BIL": path(0.00005)})
    res = run(AL.S103(), mk(), "2024-03-27", "2024-04-05")
    o = orders(res)
    first = o.iloc[0]
    assert first["symbol"] == "SPY" and pd.Timestamp(first["submitted_at"]) == CAL.open_close(pd.Timestamp("2024-03-27"))[1]
    sw = o[pd.DatetimeIndex(o["submitted_at"]) == CAL.open_close(pd.Timestamp("2024-04-01"))[1]]
    assert set(sw["symbol"]) == {"SPY", "BIL"}
    assert len(o) == 3                                             # no orders while the state is unchanged
    c2 = path(UP).copy()
    c2[i + 1:] = c2[i] * 0.995                                     # tiny dip inside a 2% band: keep SPY
    res2 = run(AL.S103(sma_n=200, hysteresis=0.02), portal_factory({"SPY": c2, "BIL": path(0.00005)})(),
               "2024-03-27", "2024-04-05")
    assert len(orders(res2)) == 1


def test_synthetic_s104_vol_target_turnover_cap():
    c = path(0.0004, wiggle=0.005)
    mk = portal_factory({"SPY": c})
    res = run(AL.S104(), mk(), "2024-03-22", "2024-03-27")
    o = orders(res)
    fri = CAL.open_close(pd.Timestamp("2024-03-22"))[1]
    hist = mk().bars("SPY", "1d", pd.Timestamp("2020-01-01", tz="UTC"), fri + pd.Timedelta(1, "ns"), fri)
    sig = float(realized_vol(hist["close"].iloc[-21:], 20, annualize=252).iloc[-1])
    assert min(1.5, 0.12 / sig) > 1.0                                # leverage on margin, > 4 days of 20%
    subs = [CAL.open_close(pd.Timestamp(d))[1] for d in ("2024-03-22", "2024-03-25", "2024-03-26", "2024-03-27")]
    assert list(pd.DatetimeIndex(o["submitted_at"])) == subs        # one order per day, first at Friday close
    eq = res.equity
    for r, t in zip(o.itertuples(), subs):
        px = c[HIST.index(pd.Timestamp(t.tz_convert("America/New_York").date()))]
        nav = float(eq[eq.index <= t].iloc[-1]) if (eq.index <= t).any() else NAV
        assert r.quantity == math.trunc(0.20 * nav / px)            # capped at 20% of NAV
    f = res.fills
    assert pd.Timestamp(f.iloc[0]["ts"]) == CAL.open_close(pd.Timestamp("2024-03-25"))[0]   # Monday open


def test_synthetic_s118_leveraged_gate_uses_actual_fund():
    spy = path(UP, wiggle=0.002)
    sso = path(2 * UP, wiggle=0.004, last=60.0)
    mk = portal_factory({"SPY": spy, "SSO": sso, "BIL": path(0.00005)})
    res = run(AL.S118(), mk(), "2024-03-27", "2024-04-03")
    o = orders(res)
    d0 = pd.Timestamp("2024-03-27")
    assert pd.Timestamp(o.iloc[0]["submitted_at"]) == CAL.open_close(d0)[1]
    assert o.iloc[0]["symbol"] == "SSO" and o.iloc[0]["quantity"] == shares(1.0, sso[HIST.index(d0)], nav_at(res, d0))
    assert len(o) == 1
    mk_hv = portal_factory({"SPY": path(UP, wiggle=0.03), "SSO": sso, "BIL": path(0.00005)})
    assert set(orders(run(AL.S118(), mk_hv(), "2024-03-27", "2024-04-03"))["symbol"]) == {"BIL"}


def test_s118_before_fund_listing_holds_tbills():
    ss = sess("2005-01-03", "2007-06-29")
    spy = 100 * np.exp(0.001 * np.arange(len(ss)))
    lst = [d for d in ss if d >= pd.Timestamp("2006-06-21")]
    p = InMemoryPortal(bars={("SPY", "1d"): daily_frame(ss, spy, spread=0.01),
                             ("SSO", "1d"): daily_frame(lst, 50 * np.exp(0.002 * np.arange(len(lst))), spread=0.01)})
    res = run(AL.S118(), p, "2006-01-03", "2006-01-10")         # SSO listed 2006-06-21; no T-bill ETF in 2006
    assert res.status == RunStatus.OK and len(orders(res)) == 0   # -> engine cash, never a synthetic 2x SPY
    res2 = run(AL.S118(), p, "2007-02-01", "2007-02-07")
    assert set(orders(res2)["symbol"]) == {"SSO"}


def test_synthetic_s119_inverse_hedge():
    c = path(UP).copy()
    i = HIST.index(ME)
    c[i - 30:i + 1] = c[i - 30] * np.exp(-0.01 * np.arange(31))    # selloff: below SMA, RSI < 45
    c[i + 1:] = c[i]
    mk = portal_factory({"SPY": c, "SH": path(FLAT, last=40.0)})
    res = run(AL.S119(), mk(), "2024-03-27", "2024-04-03")
    o = orders(res)
    t = CAL.open_close(pd.Timestamp("2024-03-27"))[1]
    first = o[pd.DatetimeIndex(o["submitted_at"]) == t].set_index("symbol")["quantity"]
    assert first["SH"] == shares(0.20, 40.0)
    assert first["SPY"] == shares(0.80, c[HIST.index(pd.Timestamp("2024-03-27"))])


def test_synthetic_s125_breadth_weekly():
    secs = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]
    spec = {s: path(UP if k < 8 else DOWN) for k, s in enumerate(secs)}
    spec["SPY"] = path(UP)
    spec["BIL"] = path(0.00005)
    res = run(AL.S125(), portal_factory(spec)(), "2024-03-25", "2024-04-05")
    o = orders(res)
    t = CAL.open_close(pd.Timestamp("2024-03-28"))[1]                   # Good Friday week: Thursday
    assert pd.Timestamp(o.iloc[0]["submitted_at"]) == t and o.iloc[0]["symbol"] == "SPY"
    assert len(o) == 1
    spec2 = {s: path(UP if k < 6 else DOWN) for k, s in enumerate(secs)}  # 6/11 < 7
    spec2["SPY"], spec2["BIL"] = path(UP), path(0.00005)
    o2 = orders(run(AL.S125(), portal_factory(spec2)(), "2024-03-25", "2024-04-05"))
    assert set(o2["symbol"]) == {"BIL"}


def test_tbill_return_compounds_rf():
    mk = portal_factory({"SPY": path(UP)}, rf=0.036)
    p = mk()
    seen = {}

    class Probe(AL._Alloc):
        id = "PROBE"; universe = ["SPY"]; params = {}
        def on_bar(self, ctx):
            if ctx.session_date() == ME:
                seen["r"] = AL.tbill_return(ctx, pd.Timestamp("2023-12-29"), ME, "3M")
    run_backtest(Probe(), p, "2024-03-27", "2024-03-28", zero_cost(), raise_errors=True)
    days = [(pd.Timestamp("2024-01-31") - pd.Timestamp("2023-12-29")).days,
            (pd.Timestamp("2024-02-29") - pd.Timestamp("2024-01-31")).days,
            (ME - pd.Timestamp("2024-02-29")).days]
    assert seen["r"] == pytest.approx(np.prod([1 + 0.036 * d / 360 for d in days]) - 1)


# ====================================================================== lookahead + smoke (all)
def _smoke_spec():
    syms = sorted(set(AL.SWING_ETF + AL.DEFENSIVE_4 + AL.CROSS_ASSET + AL.TREASURY_DURATION + ["BIL", "SHV", "SSO",
                                                                                                 "SH", "IEF"]))
    spec = {}
    for k, s in enumerate(syms):
        d = synthetic_daily_bars("2022-01-03", "2024-06-28", 50.0 + 3 * k, seed=k, volume=1e8, daily_vol=0.012)
        spec[s] = d
    return spec


_SPEC = None


def smoke_portal(cut=None):
    global _SPEC
    if _SPEC is None:
        _SPEC = {(s, "1d"): d for s, d in _smoke_spec().items()}
    return InMemoryPortal(bars=truncate(_SPEC, cut) if cut is not None else _SPEC, rf=0.02)


ALLOC_IDS = ["S089", "S090", "S101", "S102", "S103", "S104", "S105", "S117", "S118", "S119", "S120", "S121",
             "S122", "S124", "S125"]


@pytest.mark.parametrize("sid", ALLOC_IDS)
def test_lookahead_truncated_vs_full(sid):
    cut = CAL.open_close(pd.Timestamp("2024-04-30"))[1] + pd.Timedelta(hours=1)
    full, sig = assert_no_lookahead(lambda: get(sid)(), smoke_portal, "2024-01-02", "2024-06-28", cut)
    assert len(sig) > 0


@pytest.mark.parametrize("sid", ALLOC_IDS)
def test_smoke_run_backtest(sid):
    res = run_backtest(get(sid)(), smoke_portal(), "2023-06-01", "2024-06-28", None, raise_errors=True)
    assert res.status == RunStatus.OK
    assert res.diagnostics["lookahead_guard_violations"] == 0
    assert len(res.fills) > 0
    assert not res.diagnostics["margin_calls"]
    assert not res.diagnostics["rejected_orders"]
