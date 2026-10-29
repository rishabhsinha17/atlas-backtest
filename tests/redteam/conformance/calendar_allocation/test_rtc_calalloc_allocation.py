"""Red-team conformance probes: allocation strategies (total return, T-bill excess, frozen universes with
listing dates, leverage/turnover caps, requirements).

SYNTHETIC daily data, mechanics only.
"""
import math

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.types import ActionType, CorporateAction, RunStatus
from atlasbt.engine.testing import InMemoryPortal
from atlasbt.strategies import allocation as AL

from rtc_calalloc_helpers import CAL, daily_frame, fills, orders, run, sess

ME = pd.Timestamp("2024-03-28")
ME_CLOSE = CAL.open_close(ME)[1]
HIST = sess("2022-01-03", "2024-04-30")


def small(cls, universe, **attrs):
    return type(cls.__name__ + "RT", (cls,), {"universe": universe, **attrs})


def month_end_symbols(res, t=ME_CLOSE) -> set:
    o = orders(res)
    o = o[(pd.DatetimeIndex(o["submitted_at"]) == t) & (o["quantity"] > 0)]
    return set(o["symbol"])


def wiggle_path(n, drift=0.0, w=0.004, last=100.0, seed=0):
    rng = np.random.default_rng(seed)
    r = drift + w * rng.standard_normal(n)
    c = np.exp(np.cumsum(r))
    return c / c[-1] * last


# ============================================================ total return ranking
def test_s089_ranks_by_total_return_not_price_return():
    """XLE: price return over 12m = -1% but a $6 dividend went ex inside the window -> total return ~+5.3%;
    XLK: price return +4%, no dividend. 'use historical total return' -> XLE must be chosen."""
    i_me = HIST.index(ME)
    ref = AL.month_end_ref(CAL, ME, 12)
    i_ref = HIST.index(ref)
    xlk = np.full(len(HIST), 100.0)
    xlk[i_ref:i_me + 1] = np.linspace(100.0, 104.0, i_me - i_ref + 1)
    xlk[i_me + 1:] = 104.0
    ex = pd.Timestamp("2023-09-15")
    i_ex = HIST.index(ex)
    xle = np.full(len(HIST), 100.0)
    xle[i_ex:i_me + 1] = np.linspace(94.0, 99.0, i_me - i_ex + 1)
    xle[i_me + 1:] = 99.0
    bars = {("XLK", "1d"): daily_frame(HIST, xlk, spread=0.01), ("XLE", "1d"): daily_frame(HIST, xle, spread=0.01),
            ("BIL", "1d"): daily_frame(HIST, np.full(len(HIST), 91.0), spread=0.01)}
    div = CorporateAction("XLE", ActionType.CASH_DIVIDEND, ex, pd.Timestamp("2023-09-01 20:00", tz="UTC"),
                          amount=6.0, pay_date=pd.Timestamp("2023-09-20"))
    p = InMemoryPortal(bars=bars, actions=[div], rf=0.0)
    res = run(small(AL.S089, ["XLK", "XLE"])(), p, "2024-03-25", "2024-04-03")
    assert month_end_symbols(res) == {"XLE"}


def test_s089_excess_over_tbill_gate():
    """Positive 12m total return (+1%) but below the compounded T-bill return (rf 5%) -> T-bill ETF."""
    i_me = HIST.index(ME)
    ref = AL.month_end_ref(CAL, ME, 12)
    i_ref = HIST.index(ref)
    xlk = np.full(len(HIST), 100.0)
    xlk[i_ref:i_me + 1] = np.linspace(100.0, 101.0, i_me - i_ref + 1)
    xlk[i_me + 1:] = 101.0
    bars = {("XLK", "1d"): daily_frame(HIST, xlk, spread=0.01),
            ("BIL", "1d"): daily_frame(HIST, np.full(len(HIST), 91.0), spread=0.01)}
    res = run(small(AL.S089, ["XLK"])(), InMemoryPortal(bars=bars, rf=0.05), "2024-03-25", "2024-04-03")
    assert month_end_symbols(res) == {"BIL"}


# ============================================================ frozen universe / listing dates
H15 = sess("2014-06-02", "2015-11-30")
ME15 = pd.Timestamp("2015-10-30")


LEGACY = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
S120_NO_XLC = small(AL.S120, LEGACY + ["XLRE"])      # XLC (2018) has no bars in a 2015 test window


def _sector_bars(backfill_xlre: bool, xlre_drift=0.0):
    legacy = LEGACY
    bars = {(s, "1d"): daily_frame(H15, wiggle_path(len(H15), 0.0, 0.01, seed=k), spread=0.01)
            for k, s in enumerate(legacy)}
    xs = H15 if backfill_xlre else [d for d in H15 if d >= pd.Timestamp("2015-10-08")]
    bars[("XLRE", "1d")] = daily_frame(xs, wiggle_path(len(xs), xlre_drift, 0.001, seed=99), spread=0.01)
    bars[("SPY", "1d")] = daily_frame(H15, wiggle_path(len(H15), 0.0, 0.01, seed=7), spread=0.01)
    bars[("BIL", "1d")] = daily_frame(H15, np.full(len(H15), 91.0), spread=0.01)
    return bars


def test_s120_excludes_etf_whose_window_predates_listing():
    """FINDING (MINOR): allocation.py docstring: 'eligible only after its listing date AND ... (no
    back-filled or proxy histories)'. S120 calls daily_closes(sym, vol_n+1, d), which only checks that the
    DECISION date is after INCEPTION. A portal that serves pre-inception (spliced/back-filled) XLRE bars makes
    XLRE eligible at 2015-10-30 with a 63-day window starting ~2015-07-30 (XLRE first traded 2015-10-08).
    period_return-based strategies check listed(sym, ref); the vol/SMA-window ones (S090, S120, S125) do not."""
    p = InMemoryPortal(bars=_sector_bars(backfill_xlre=True), rf=0.0)
    res = run(S120_NO_XLC(), p, "2015-10-28", "2015-11-03")
    assert res.status == RunStatus.OK and len(orders(res)) > 0
    assert "XLRE" not in month_end_symbols(res, CAL.open_close(ME15)[1])


def test_s120_true_listing_history_is_excluded():
    p = InMemoryPortal(bars=_sector_bars(backfill_xlre=False), rf=0.0)
    res = run(S120_NO_XLC(), p, "2015-10-28", "2015-11-03")
    syms = month_end_symbols(res, CAL.open_close(ME15)[1])
    assert "XLRE" not in syms and len(syms) == 9


# ============================================================ S104 caps
def test_s104_leverage_cap_and_daily_turnover_cap_hold_over_two_weeks():
    c = wiggle_path(len(HIST), 0.0002, 0.002, seed=1)          # ~3% vol -> target = cap 1.5
    p = InMemoryPortal(bars={("SPY", "1d"): daily_frame(HIST, c, spread=0.01)}, rf=0.0)
    res = run(AL.S104(), p, "2024-03-01", "2024-03-22")
    f = fills(res)
    eq = res.equity
    for r in f.itertuples():
        t = pd.Timestamp(r.ts)
        nav = float(eq[eq.index < t].iloc[-1]) if (eq.index < t).any() else 1e6
        assert abs(r.quantity) * r.price <= 0.20 * nav * 1.02          # one day's trade <= 20% NAV (+gap)
    pos = res.positions
    pos = pos[pos["symbol"] == "SPY"]
    mv = pos.set_index(pd.DatetimeIndex(pos["ts"]))["market_value"]
    lev = (mv / eq.reindex(mv.index)).dropna()
    assert len(lev) >= 10 and lev.max() > 1.3                      # reached the cap region ...
    assert lev.max() <= 1.5 * 1.02                                  # ... never above min(1.5, .12/sigma)
    # all fills at session opens (weekly Monday start, then one per day)
    opens = {CAL.open_close(d)[0] for d in sess("2024-03-01", "2024-03-22")}
    assert set(pd.DatetimeIndex(f["ts"])) <= opens


# ============================================================ S105 requirements / first session
def test_s105_initial_buy_fills_at_first_session_open():
    bars = {(s, "1d"): daily_frame(HIST, np.full(len(HIST), px), spread=0.01) for s, px in (("SPY", 400.0),
                                                                                          ("IEF", 95.0))}
    res = run(AL.S105(), InMemoryPortal(bars=bars), "2024-03-04", "2024-03-06")
    f = fills(res)
    t0 = CAL.open_close(pd.Timestamp("2024-03-04"))[0]
    assert set(f[pd.DatetimeIndex(f["ts"]) == t0]["symbol"]) == {"SPY", "IEF"}


def test_s105_distributions_are_a_declared_requirement():
    """FINDING: sheet S105 'total-return prices ... include expense ratios and distributions'. No
    Requirement('actions') is declared (nor for any allocation strategy), so a portal without
    dividend data silently runs a PRICE-return 60/40 (status OK) instead of NOT_TESTABLE."""
    bars = {(s, "1d"): daily_frame(HIST, np.full(len(HIST), px), spread=0.01) for s, px in (("SPY", 400.0),
                                                                                          ("IEF", 95.0))}
    res = run(AL.S105(), InMemoryPortal(bars=bars), "2024-03-04", "2024-03-06")
    assert res.status == RunStatus.NOT_TESTABLE or any(r.kind == "actions" for r in AL.S105.requires)
