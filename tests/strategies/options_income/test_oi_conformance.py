"""Conformance of S126-S157 with the sheet (docs/strategies.json) and unit
tests of the family helpers. No network; synthetic data only for mechanics."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.types import Instrument, Right, RunStatus
from atlasbt.options import structures as st
from atlasbt.risk import sizing as sz
from atlasbt.strategies import registry
from atlasbt.strategies.options_income_base import OptionBase, package_stress_loss

import oi_fixtures as fx

ROOT = Path(__file__).resolve().parents[3]
SHEET = {r["ID"]: r for r in json.loads((ROOT / "docs" / "strategies.json").read_text())}
IDS = [f"S{i}" for i in range(126, 158)]
BARS = {"1d": "1d", "1mo": "1mo"}


def test_all_registered():
    registry.load_all()
    assert registry.load_all().get("atlasbt.strategies.options_income") is None
    for sid in IDS:
        assert registry.get(sid).id == sid


@pytest.mark.parametrize("sid", IDS)
def test_class_matches_sheet(sid):
    cls = registry.get(sid)
    row = SHEET[sid]
    assert cls.spec == row["Entry rule"] + " || " + row["Exit / risk rule"]
    assert cls.name == row["Strategy"]
    assert cls.sizing == row["Sizing"]
    assert cls.bar_freq == BARS[row["Bars"]]
    assert set(cls.grid) <= set(cls.params), "grid keys must be params"
    assert all(len(v) >= 1 for v in cls.grid.values())
    assert len(cls.deviations) > 0
    s = cls()
    assert s.universe, "universe must be declared"
    for combo in registry.expand_grid(cls.grid):
        cls(**combo)                      # every grid point instantiates


def test_grid_values_verbatim_examples():
    g = {sid: registry.get(sid).grid for sid in IDS}
    assert g["S126"] == {"delta": [0.45, 0.50, 0.55], "dte": [21, 30, 45], "opt_stop": [0.0, 0.5]}
    assert g["S136"]["wing"] == ["pts5", "pts10", "delta10"]
    assert g["S138"] == {"short_delta": [0.15, 0.20, 0.25], "wing_delta": [0.03, 0.05, 0.10], "dte": [30, 45]}
    assert g["S146"]["hedge"] == ["1d", "1h"]
    assert g["S130"]["tp"] == [0.25, 0.50, 0.75]


def test_default_underlyings():
    for sid in IDS:
        s = registry.get(sid)()
        exp = ["SPX"] if sid in ("S129", "S138") else ["SPY"]
        assert s.universe == exp, sid
    s = registry.get("S131")(use_megacap=True)
    assert "AAPL" in s.universe and len(s.universe) >= 5


# ------------------------------------------------------------------ NOT_TESTABLE by design
def _tiny_portal(und="SPY"):
    ss = fx.sessions(30)
    c = np.linspace(100, 103, 30)
    return fx.make_portal(c, ss, und=und, chain_from=0, band=0.1), ss


def test_s146_hourly_hedge_not_testable():
    p, ss = _tiny_portal()
    r = fx.run(registry.get("S146")(hedge="1h"), p, ss)
    assert r.status == RunStatus.NOT_TESTABLE and "options_intraday" in r.not_testable_reason


def test_s130_single_stock_needs_forward_earnings_calendar():
    p, ss = _tiny_portal("AAPL")
    r = fx.run(registry.get("S130")(underlyings=("AAPL",)), p, ss)
    assert r.status == RunStatus.NOT_TESTABLE and "earnings_calendar_forward" in r.not_testable_reason
    r2 = fx.run(registry.get("S130")(underlyings=("AAPL",), earnings_proxy=True), p, ss)
    assert r2.status == RunStatus.OK


@pytest.mark.parametrize("sid", ["S129", "S138"])
def test_spx_strategies_need_spx_data(sid):
    p, ss = _tiny_portal()
    r = fx.run(registry.get(sid)(), p, ss)
    assert r.status == RunStatus.NOT_TESTABLE and "SPX" in r.not_testable_reason


# ------------------------------------------------------------------ helpers
def test_package_stress_equals_sizing_single_short():
    """package_stress_loss applies risk.sizing's scenario to all legs; for one short option it must equal
    sizing.stress_loss_short_option (independent reference)."""
    exp = pd.Timestamp("2024-03-15")
    leg = st.OptLeg("SPY", exp, 95.0, "P", -1, bid=1.20, ask=1.30, mid=1.25)
    pkg = st.Package("short_put", [leg])
    ref = sz.stress_loss_short_option(Instrument.option("SPY", exp, 95.0, Right.PUT), 100.0, 1.20)
    assert package_stress_loss(pkg, 100.0) == pytest.approx(ref)
    # hand computation: -20% move: intrinsic 15 + extrinsic 1.2*2 - premium 1.2 = 16.2 -> $1,620
    assert ref == pytest.approx(1620.0)


def test_package_stress_strangle_not_sum_of_legs():
    exp = pd.Timestamp("2024-03-15")
    put = st.OptLeg("SPY", exp, 90.0, "P", -1, bid=1.0, ask=1.1, mid=1.05)
    call = st.OptLeg("SPY", exp, 110.0, "C", -1, bid=1.0, ask=1.1, mid=1.05)
    pkg = st.Package("short_strangle", [put, call])
    # -20%: put intrinsic 10 + 2*1.0 - 1.0 = 11 ; call 0 + 2*1.0 - 1.0 = 1 -> 12 per share
    assert package_stress_loss(pkg, 100.0) == pytest.approx(1200.0)


def test_metrics_rows_brackets_30d():
    ss = fx.sessions(10)
    c = np.full(10, 100.0)
    p = fx.make_portal(c, ss, chain_from=0, band=0.3)
    now = fx.close_ts(ss[5])
    ch = p.option_chain("SPY", now)
    rows = OptionBase.metrics_rows(ch, now, 100.0)
    dte = sorted({(pd.Timestamp(e) - ss[5]).days for e in rows["expiry"]})
    all_dte = sorted({(pd.Timestamp(e) - ss[5]).days for e in ch["expiry"]})
    below = max(d for d in all_dte if 1 <= d <= 30)
    above = min(d for d in all_dte if d >= 30)
    assert dte == sorted({below, above})
    assert rows["strike"].between(80, 120).all()


# ------------------------------------------------------------------ delivered shares (option-only strategies)
def test_delivered_shares_closed_next_open_and_booked_to_package():
    """S141 call fly whose closing combo cannot fill (snapshots stop 8 days before expiry): at expiry the long
    K1 call is exercised (+100/fly) and the short K2 calls are assigned (-200/fly) -> -100 shares/fly delivered.
    The strategy (holds no stock by design) buys them back at the NEXT session open and books the stock P&L to
    the package."""
    n = 40
    ss = fx.sessions(n)
    closes = np.r_[np.full(10, 100.0), np.full(n - 10, 103.0)]
    exp = [pd.Timestamp("2021-02-05")]
    stop = next(i for i, d in enumerate(ss) if (exp[0] - d).days <= 8)
    p = fx.make_portal(closes, ss, chain_from=0, chain_to=stop, expiries=exp, band=0.2)
    s = registry.get("S141")(width=5.0)
    r = fx.run(s, p, ss)
    f = r.fills
    deliv = f[(f["symbol"] == fx.UND) & (f["liquidity"].isin(["ASSIGNMENT", "EXERCISE"]))]
    assert len(deliv) >= 2
    exp_i = ss.index(exp[0])
    close = f[(f["symbol"] == fx.UND) & (f["tag"] == "close_delivered")]
    assert len(close) == 1
    assert pd.Timestamp(close["ts"].iloc[0]) == fx.open_ts(ss[exp_i + 1])
    assert float(close["quantity"].iloc[0]) == pytest.approx(-float(deliv["quantity"].sum()))
    last = r.positions[r.positions["ts"] == r.equity.index[-1]]          # snapshots list nonzero positions only
    assert float(last[last["symbol"] == fx.UND]["quantity"].sum()) == 0.0
    pos = s.positions[fx.UND][0]
    assert pos.meta["delivered"] == pytest.approx(float(deliv["quantity"].sum()))
    assert np.isfinite(pos.meta["delivery_pnl"])


def test_activity_diagnostics_surface_skips():
    """Selection failures / skips are surfaced as 'activity <SID>: ...' fidelity notes (runner diagnose)."""
    ss = fx.sessions(20)
    p = fx.make_portal(np.full(20, 100.0), ss, chain_from=0, max_dte=20)       # no expiry in 25-45
    r = fx.run(registry.get("S139")(), p, ss)
    reasons = [x["reason"] for x in r.fidelity["reasons"]]
    assert any(x.startswith("activity S139:") and "selection" in x for x in reasons)
    assert any(l.get("activity") == "selection_failure" for l in r.diagnostics["logs"])


def test_s150_declares_activity_note():
    assert registry.get("S150").activity_note
