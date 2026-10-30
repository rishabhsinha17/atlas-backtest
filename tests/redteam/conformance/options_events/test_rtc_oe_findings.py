"""RED-TEAM spec-conformance probes for the options event / skew / short-dated family (S158-S172).

SYNTHETIC patterns (mechanics only; fixtures from tests/strategies/options_events/oe_fix.py).

* ``test_finding_*`` encode the SHEET behaviour (docs/strategies.json + Sizing Profiles) and are
  expected to FAIL against the current implementation; each documents one finding of the report.
* ``test_conf_*`` are conformance probes that PASS (clause checked and found conformant).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "strategies", "options_events"))
from oe_fix import (close_of, close_snaps, daily_bars, ev, fills, index_instrument, intraday_bars,  # noqa: E402
                    minute_snaps, option_book, orders, run, sess)

from atlasbt.core.clock import et  # noqa: E402
from atlasbt.core.types import Fidelity, RunStatus  # noqa: E402
from atlasbt.strategies.options_events import (S158, S162, S164, S165, S166, S169, S170, S171,  # noqa: E402
                                               S172)
from atlasbt.strategies.options_events_vol import TOP5_SECTORS  # noqa: E402


def combo(res, prefix):
    o = orders(res)
    return o[o["tag"].astype(str).str.startswith(prefix)]


def ny_day(t):
    return pd.Timestamp(t).tz_convert("America/New_York").tz_localize(None).normalize()


# =============================================================================== S171
DAY = pd.Timestamp("2024-03-13")


def T(hhmm, day=DAY):
    return et(day, hhmm)


def parity_spec(gap=0.25, expiry="2024-04-19"):
    """XYZ at 100 (NBBO 99.99/100.01). ONLY the 10:00 snapshot shows a conversion edge (100-put
    offered `gap` below the 100-call bid); every later snapshot is a fair BSM market."""
    snaps = minute_snaps([DAY])
    optq, insts = option_book("XYZ", [(expiry, "PHYSICAL", "AMERICAN", None)], [95.0, 100.0, 105.0], snaps,
                              lambda t: 100.0, lambda t, K, e, Ty: 0.20, hs=lambda p: 0.05, max_dte=200)
    c100 = [s for s, i in insts.items() if i.strike == 100.0 and i.right.value == "C"][0]
    p100 = [s for s, i in insts.items() if i.strike == 100.0 and i.right.value == "P"][0]
    c, p = optq[c100].copy(), optq[p100].copy()
    t10 = T("10:00")
    p.loc[t10, "ask"] = c.loc[t10, "bid"] - gap
    p.loc[t10, "bid"] = p.loc[t10, "ask"] - 0.10
    optq[c100], optq[p100] = c, p
    qts = pd.DataFrame({"bid": 99.99, "ask": 100.01, "bid_size": 1e5, "ask_size": 1e5}, index=snaps)
    bars = {("XYZ", "1m"): intraday_bars([DAY], lambda t: 100.0)}
    return {"bars": bars, "option_quotes": optq, "instruments": insts, "quotes": {"XYZ": qts}}


S171T = type("S171T", (S171,), {"universe": ["XYZ"]})


def test_finding_S171_entry_fills_without_arbitrage_after_signal_vanishes():
    """Sheet: 'ENTER only if arbitrage after all costs'; Sizing OPT_ARBITRAGE: 'Position only if
    executable quoted market ... shows positive modeled edge'. The edge exists only in the 10:00
    snapshot; the option COMBO has NO limit price, so it fills at the 10:01 fair market and the
    executed conversion locks in a LOSS (outlay > K)."""
    r = run(S171T(), parity_spec(), DAY, DAY)
    op = combo(r, "S171:open")
    assert len(op) == 1
    fo = fills(r, "S171:open")
    fs = fills(r, "S171:stock_leg")
    if len(fo) == 0:
        return                                                        # not entered: conformant
    n = int(abs(fo["quantity"].iloc[0]))
    outlay_ps = (float((fo["quantity"] * fo["price"]).sum()) * 100.0
                 + float((fs["quantity"] * fs["price"]).sum())) / (100.0 * n)
    locked = 100.0 - outlay_ps                                         # r = 0, no dividends: K - outlay
    assert locked >= 0.0, f"executed conversion locks {locked:+.3f}/share (no limit on the entry combo)"


def test_finding_S171_undeclared_max_dte_filter():
    """Sheet: 'For same strike/expiry ...' (any listed expiry). Code scans only DTE <= 60
    (param max_dte, not in the sheet, not in deviations): a 98-DTE conversion edge is ignored."""
    r = run(S171T(), parity_spec(expiry="2024-06-21"), DAY, DAY)
    assert len(combo(r, "S171:open")) == 1


# =============================================================================== S172
BSESS = sess("2024-01-29", "2024-02-06")


def box_spec_signal_only_at_month_end():
    """Long box cheap (implied lending 8% vs 5% bills) ONLY in the 2024-01-31 close snapshot; from the
    next snapshot on, quotes are BSM at r = 5% (natural box debit above the bill-discounted width)."""
    exps = [("2024-03-01", "CASH_PM", "EUROPEAN", "SPXW")]
    snaps = close_snaps(BSESS)
    strikes = np.arange(4700.0, 5101.0, 25.0)
    cheap, insts = option_book("SPX", exps, strikes, snaps, lambda t: 4900.0, lambda t, K, e, Ty: 0.15, r=0.08,
                               hs=lambda p: 0.01)
    fair, _ = option_book("SPX", exps, strikes, snaps, lambda t: 4900.0, lambda t, K, e, Ty: 0.15, r=0.05,
                          hs=lambda p: 0.25)
    me = close_of("2024-01-31")
    optq = {}
    for s in cheap:
        q = fair[s].copy()
        q.loc[me] = cheap[s].loc[me]
        optq[s] = q
    insts["SPX"] = index_instrument("SPX")
    return {"bars": {("SPX", "1d"): daily_bars(BSESS, np.full(len(BSESS), 4900.0))}, "option_quotes": optq,
            "instruments": insts, "rf": 0.05}


def test_finding_S172_box_entry_without_limit_fills_at_negative_edge():
    """Sheet: 'use simultaneous executable quote' and 'compare discounted guaranteed terminal width*100
    against net debit, funding rate, fees'. The decision uses the 01-31 close snapshot; the COMBO
    carries no limit, so it fills at the next snapshot whatever the box costs then (here above the
    bill-discounted width: negative carry locked in)."""
    r = run(S172(), box_spec_signal_only_at_month_end(), BSESS[0], BSESS[-1])
    op = combo(r, "S172:open")
    assert len(op) == 1
    f = fills(r, "S172:open")
    if len(f) == 0:
        return
    n = int(abs(f["quantity"].iloc[0]))
    debit = float((f["quantity"] * f["price"]).sum()) / n                # per share of the box (width 50)
    T_ = (pd.Timestamp("2024-03-01") - pd.Timestamp("2024-02-01")).days / 365.0
    assert debit <= 50.0 * np.exp(-0.05 * T_), f"box bought at {debit:.3f} > PV(width) {50*np.exp(-0.05*T_):.3f}"


# =============================================================================== S170
SESS70 = sess("2024-01-22", "2024-02-09")
EXPS70 = [(e, "PHYSICAL", "EUROPEAN", None) for e in ("2024-02-09", "2024-03-01", "2024-03-15", "2024-04-19")]


def s170_spec(missing_after=None, missing_sym="XLC"):
    optq, insts, bars = {}, {}, {}
    for u in ["SPY"] + TOP5_SECTORS:
        q, i = option_book(u, EXPS70, np.arange(80.0, 121.0, 2.0), close_snaps(SESS70), lambda t: 100.0,
                           lambda t, K, e, Ty: 0.20, hs=lambda p: 0.02, fidelity=2, max_dte=100)
        if u == missing_sym and missing_after is not None:
            q = {s: d[d.index <= missing_after] for s, d in q.items()}
        optq.update(q)
        insts.update(i)
        bars[(u, "1d")] = daily_bars(SESS70, np.full(len(SESS70), 100.0))
    return {"bars": bars, "option_quotes": optq, "instruments": insts}


S170T = type("S170T", (S170,), {"universe": ["SPY"] + TOP5_SECTORS, "LOOKBACK": 10})


def test_finding_S170_unbalanced_dispersion_after_a_leg_fails():
    """Sheet: 'Sell 30DTE ATM SPY straddle AND buy dollar-vega matched ATM straddles across fixed 5
    sector ETFs'. The six straddles are independent DAY COMBOs; when one sector straddle cannot fill,
    the strategy keeps the naked short SPY straddle (and the other 4 legs) until DTE<=7 instead of
    completing or unwinding the package."""
    me = close_of("2024-01-31")
    r = run(S170T(), s170_spec(missing_after=me), SESS70[0], SESS70[-1])
    op = combo(r, "S170:open")
    # fix: the five sector straddles are submitted first; the SPY leg only after all five filled
    assert len(op) == 5 and not op["symbol"].str.contains("SPY").any()
    f = fills(r, "S170:open")
    assert not any(f["symbol"].str.startswith("XLC")), "fixture: XLC straddle must not fill"
    pos = r.positions
    last = pos[pos["ts"] == pos["ts"].max()] if "ts" in pos.columns else pos
    spy_opt = last[last["symbol"].astype(str).str.startswith("SPY   ")]
    held = float(spy_opt["quantity"].abs().sum()) if len(spy_opt) else 0.0
    assert held == 0.0, f"short SPY straddle still held ({held:g} contracts) without the XLC leg"
    af = r.fills                                     # positions frame lists only non-zero holdings: use fills
    sec = af[af["symbol"].astype(str).str[:6].str.strip().isin(TOP5_SECTORS) & (af["symbol"].str.len() > 10)]
    net = sec.groupby("symbol")["quantity"].sum()
    assert len(sec) > 0 and (net.abs() < 1e-9).all(), "filled sector straddles must be closed"


# =============================================================================== S164
DAY64 = pd.Timestamp("2024-03-13")
PREV64 = pd.Timestamp("2024-03-12")
SIG_T0 = 0.15 * np.sqrt(6.0 / (365 * 24))


def fomc_spec(with_spx_bars=True, locked_at=None):
    spot = lambda t: 4000.0 + (20.0 if t >= et(DAY64, "14:00") else 0.0)
    snaps = minute_snaps([DAY64])
    vol = lambda t, K, e, Ty: SIG_T0 / np.sqrt(Ty)

    def lock(t, inst, b, a):
        if locked_at is not None and t == et(DAY64, locked_at):
            return a, a                                                # locked NBBO (bid == ask)
        return b, a
    optq, insts = option_book("SPX", [(DAY64, "CASH_PM", "EUROPEAN", "SPXW")], np.arange(3900, 4101, 5.0), snaps,
                              spot, vol, hs=lambda p: 0.05, price_override=lock)
    insts["SPX"] = index_instrument("SPX")
    bars = {("SPX", "1m"): intraday_bars([PREV64, DAY64], spot)} if with_spx_bars else {}
    events = [ev("FOMC_SCHEDULE", None, "2024-01-01 12:00", et(DAY64, "14:00"), type="STATEMENT"),
              ev("FOMC", None, et(DAY64, "14:00"), et(DAY64, "14:00"), type="STATEMENT")]
    return {"bars": bars, "option_quotes": optq, "instruments": insts, "events": events}


def test_finding_S164_missing_underlying_data_is_silent_no_trade():
    """S164 overrides `requires` without the SPX price input that `prepared()` needs when the chain has
    no underlying_price: without SPX bars the run is OK with zero trades (silent no-trade) instead of
    NOT_TESTABLE."""
    r = run(S164(), fomc_spec(with_spx_bars=False), DAY64, DAY64)
    assert not (r.status == RunStatus.OK and len(combo(r, "S164")) == 0), "silent no-trade"


def test_conf_S164_exit_skips_locked_nbbo():
    """'Exit ... using first available nonlocked NBBO': the 14:31 snapshot is locked -> fill at 14:32."""
    r = run(S164(), fomc_spec(locked_at="14:31"), PREV64, DAY64)
    assert pd.Timestamp(combo(r, "S164:close")["submitted_at"].iloc[0]) == et(DAY64, "14:30")
    assert set(fills(r, "S164:close")["ts"]) == {et(DAY64, "14:32")}


def test_finding_S164_deviation_misstates_locked_quote_handling():
    """The declared deviation says the engine accepts locked quotes ('first nonlocked NBBO is not
    enforced'); the engine skips them (test_conf_S164_exit_skips_locked_nbbo). Declaration is wrong."""
    assert not any("not enforced" in d for d in S164.deviations)


# =============================================================================== S162 grid
def test_finding_S162_grid_has_degenerate_short_equals_wing_combo():
    """Sheet grid 'short delta .1/.2/.3; wing .05/.1' contains short=.1 / wing=.1, for which both legs
    collapse onto one strike (vertical() raises -> every day skipped). The interpretation is not
    declared in deviations; the grid cell silently produces zero trades."""
    g = S162.grid
    degenerate = [(s, w) for s in g["short_delta"] for w in g["wing_delta"] if abs(s - w) < 1e-12]
    declared = any("wing" in d.lower() and ("collapse" in d.lower() or "degenerate" in d.lower())
                   for d in S162.deviations)
    assert not degenerate or declared


# =============================================================================== S169
class _Ctx:
    def __init__(self, h):
        self._h = h

    def history(self, sym, n, freq, adjusted=True):
        return self._h.iloc[-n:]


def test_finding_S169_har_window_is_rolling_600_not_expanding():
    """Deviation declares 'walk-forward OLS refit every 21 sessions ... expanding window'. The code feeds
    only the last 600 daily bars to har_rv_forecast, so the training window is a rolling ~580 rows and
    the refit dates shift every day; the forecast differs from the declared expanding-window model."""
    from atlasbt.pricing.realized import daily_variance, har_rv_forecast
    rng = np.random.default_rng(7)
    n = 900
    vol = np.r_[np.full(300, 0.04), np.full(n - 300, 0.01)]          # regime change early in the history
    ret = rng.normal(0, vol)
    close = 100 * np.exp(np.cumsum(ret))
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    h = pd.DataFrame({"open": np.r_[100, close[:-1]], "close": close}, index=idx)
    h["high"] = np.maximum(h["open"], h["close"]) * 1.003
    h["low"] = np.minimum(h["open"], h["close"]) * 0.997
    s = S169()
    got = s._forecast(_Ctx(h), "XYZ")
    expanding = float(har_rv_forecast(daily_variance(h, "gk"), horizon=21, min_train=252,
                                      refit_every=21)["vol_fcst"].iloc[-1])
    assert abs(got - expanding) < 1e-9, f"rolling-600 {got:.5f} vs declared expanding {expanding:.5f}"


# =============================================================================== metric windows
SESS65 = sess("2024-01-02", "2024-03-28")
EXPS65 = [(e, "PHYSICAL", "EUROPEAN", None) for e in pd.date_range("2024-01-05", "2024-07-31", freq="14D")]


def s165_gap_spec(drop_idx=(), spike_idx=40):
    def vol(t, K, e, Ty):
        d = ny_day(t)
        i = SESS65.index(d) if d in SESS65 else 0
        if i >= spike_idx:
            return 0.32
        return 0.26 - 0.0015 * i                                        # declining: pct ~0 before the spike
    snaps = close_snaps(SESS65)
    optq, insts = option_book("XYZ", EXPS65, np.arange(70.0, 131.0, 2.0), snaps, lambda t: 100.0, vol,
                              hs=lambda p: 0.02, fidelity=2, max_dte=100)
    drop = {close_of(SESS65[i]) for i in drop_idx}
    optq = {s: d[~d.index.isin(list(drop))] for s, d in optq.items()}
    return {"bars": {("XYZ", "1d"): daily_bars(SESS65, np.full(len(SESS65), 100.0))}, "option_quotes": optq,
            "instruments": insts}


S165T = type("S165T", (S165,), {"universe": ["XYZ"], "LOOKBACK": 10})


def test_conf_S165_percentile_prior_window_entry():
    r = run(S165T(), s165_gap_spec(), SESS65[0], SESS65[42])
    op = combo(r, "S165:open")
    assert len(op) == 1 and pd.Timestamp(op["submitted_at"].iloc[0]) == close_of(SESS65[40])


def test_finding_S165_percentile_window_lost_after_gaps():
    """Family convention (module docstring): 'percentiles and z-scores use the previous LOOKBACK VALID
    daily observations (sessions without a usable real snapshot are skipped)'. metric_series only
    searches LOOKBACK+10 sessions back, so > 10 missing snapshots inside that span make the
    percentile NaN and the entry is silently skipped although LOOKBACK valid prior values exist."""
    r = run(S165T(), s165_gap_spec(drop_idx=range(24, 35)), SESS65[0], SESS65[42])
    op = combo(r, "S165:open")
    assert len(op) == 1 and pd.Timestamp(op["submitted_at"].iloc[0]) == close_of(SESS65[40])


# =============================================================================== OPT_DEBIT deliverable
def test_finding_S166_opt_debit_does_not_verify_deliverable():
    """Sizing OPT_DEBIT: 'integer contracts, multiplier/deliverable verified'. No strategy of the family
    calls atlasbt.risk.sizing.verify_contract; a chain of OCC-adjusted contracts (150-share
    deliverable) is traded as if standard."""
    from atlasbt.core.types import Instrument
    import dataclasses
    hist = {i: (-0.40 if i % 2 == 0 else -0.48) for i in range(0, 15)}
    hist[15] = -0.10

    def slope(t):
        d = ny_day(t)
        i = SESS65.index(d) if d in SESS65 else 0
        return hist.get(i, -0.10)
    vol = lambda t, K, e, Ty: max(0.20 + slope(t) * np.log(K / 100.0), 0.05)
    optq, insts = option_book("XYZ", EXPS65, np.arange(70.0, 131.0, 2.0), close_snaps(SESS65), lambda t: 100.0, vol,
                              hs=lambda p: 0.02, fidelity=2, max_dte=100)
    insts = {s: dataclasses.replace(i, deliverable_shares=150.0) for s, i in insts.items()}
    spec = {"bars": {("XYZ", "1d"): daily_bars(SESS65, np.full(len(SESS65), 100.0))}, "option_quotes": optq,
            "instruments": insts}
    S166T = type("S166T", (S166,), {"universe": ["XYZ"], "LOOKBACK": 10})
    r = run(S166T(), spec, SESS65[0], SESS65[18])
    assert len(combo(r, "S166:open")) == 0, "non-standard deliverable traded under OPT_DEBIT"


# =============================================================================== S158 DTE window
D58 = pd.Timestamp("2024-03-12")          # Tuesday, AMC release -> entry 03-12 15:30
D58_1 = pd.Timestamp("2024-03-13")
SESS58 = sess("2024-03-11", "2024-03-15")


def s158_spec(expiries):
    from oe_fix import daily_from
    price = lambda t: 100.0
    intra = intraday_bars(SESS58, price, "5m")
    snaps = minute_snaps(SESS58, step_min=5)
    exps = [(e, "PHYSICAL", "EUROPEAN", None) for e in expiries]
    optq, insts = option_book("XYZ", exps, np.arange(90.0, 111.0, 1.0), snaps, price,
                              lambda t, K, e, Ty: 0.40, hs=lambda p: 0.02)
    evs = [ev("EARNINGS_SCHEDULE", "XYZ", "2024-03-01 00:00", et(D58, "16:30"), timing="AMC"),
           ev("EARNINGS", "XYZ", et(D58, "16:05"), et(D58, "16:05"), timing="AMC")]
    return {"bars": {("XYZ", "5m"): intra, ("XYZ", "1d"): daily_from(intra)}, "option_quotes": optq,
            "instruments": insts, "events": evs, "universes": {"SP500": ["XYZ"]}}


@pytest.mark.parametrize("kw,expiries,want", [
    ({}, ("2024-03-13", "2024-03-15", "2024-03-22"), "240315"),           # DTE 3 = lower bound inclusive
    ({"dte": 10}, ("2024-03-14", "2024-03-22", "2024-03-25"), "240322"),  # DTE 10 = upper bound inclusive
    ({}, ("2024-03-14", "2024-03-25"), None),                             # DTE 2 / 13 only -> no trade
    ({}, ("2024-03-13", "2024-03-22"), "240322"),                         # 03-13 (react day) has DTE 1
])
def test_conf_S158_dte_window_inclusive(kw, expiries, want):
    r = run(S158(**kw), s158_spec(expiries), D58, D58_1)
    f = fills(r, "S158:open")
    if want is None:
        assert len(combo(r, "S158:open")) == 0
    else:
        assert set(f["symbol"].str[6:12]) == {want}
        assert pd.Timestamp(combo(r, "S158:close")["submitted_at"].iloc[0]) == et(D58_1, "09:45")


# =============================================================================== stale declarations
def test_finding_S159_S171_deviations_misstate_engine_defaults():
    """S159/S171 deviations say early assignment 'needs BacktestConfig(option_lifecycle=...OccOptionLifecycle)'
    / 'needs the OccOptionLifecycle plug-in in the config'; S171 also says 'the engine prices COMBO stock
    legs from bar opens'. The default BacktestConfig already uses OccOptionLifecycle with early
    assignment, and COMBO stock legs are priced from a synchronized NBBO when quotes exist."""
    from atlasbt.engine import BacktestConfig
    from atlasbt.options.plugins import OccOptionLifecycle
    from atlasbt.strategies.options_events import S159
    lc = BacktestConfig().lifecycle()
    assert isinstance(lc, OccOptionLifecycle)
    stale = [d for d in S159.deviations + S171.deviations
             if "needs BacktestConfig(option_lifecycle" in d or "needs the OccOptionLifecycle plug-in" in d
             or "prices COMBO stock legs from bar opens" in d]
    assert not stale, stale
