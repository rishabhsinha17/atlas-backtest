"""RED-TEAM spec-conformance probes for the options directional / income family (S126-S157).

SYNTHETIC patterns (mechanics only) on the options-income fixtures
(tests/strategies/options_income/oi_fixtures.py: EOD BSM chain snapshots, $1 strikes, weekly + monthly
expiries).  Each test builds an exact sheet scenario (docs/strategies.json) and asserts engine-level
behaviour.

* ``test_finding_*`` encode the SHEET / sizing-profile behaviour and are expected to FAIL against the current
  implementation (each documents a finding of the red-team report).
* ``test_conf_*`` are conformance probes that pass (clauses checked and found conformant).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "strategies", "options_income"))
import oi_fixtures as fx  # noqa: E402

from atlasbt.core.types import ActionType, CorporateAction, RunStatus  # noqa: E402
from atlasbt.strategies import registry  # noqa: E402
import atlasbt.strategies.options_income  # noqa: E402,F401
from atlasbt.strategies.options_income import S147  # noqa: E402


def _final_position(res, sym: str) -> float:
    p = res.positions
    if p is None or not len(p):
        return 0.0
    last_ts = res.equity.index[-1]          # snapshots list nonzero positions only: use the run's last close
    q = p[(p["ts"] == last_ts) & (p["symbol"] == sym)]["quantity"]
    return float(q.sum()) if len(q) else 0.0


def _stock_fills(res, sym: str) -> pd.DataFrame:
    f = res.fills
    return f[(f["symbol"] == sym)].reset_index(drop=True)


# ============================================================================ F1: assigned stock never closed
def _assignment_portal(n: int = 45):
    """Flat at 100 for 6 sessions, rally to 125 by session 14, then flat. Ex-dividend $3 on 2021-01-27
    (known well before) -> the engine's OCC lifecycle early-assigns deep-ITM short calls on 2021-01-26."""
    ss = fx.sessions(n)
    closes = np.r_[np.full(6, 100.0), np.linspace(100.0, 125.0, 9)[1:], np.full(n - 14, 125.0)]
    div = CorporateAction(symbol=fx.UND, type=ActionType.CASH_DIVIDEND, ex_date=pd.Timestamp("2021-01-27"),
                          known_at=pd.Timestamp("2020-12-15 21:00", tz="UTC"), amount=3.0,
                          pay_date=pd.Timestamp("2021-02-03"))
    p = fx.make_portal(closes, ss, chain_from=0, vol=0.20, band=0.40, actions=[div])
    return p, ss


def test_finding_s141_assigned_short_call_shares_left_open():
    """S141 long call butterfly: 'risk=debit'. When the short K2 calls are (early) assigned the engine
    delivers -200 shares per fly; the strategy closes the remaining long calls ('leg_settled') but never
    closes the assigned stock, leaving a naked short-stock position far beyond the OPT_DEBIT budget.
    (Observed on real DoltHub data too: S141 2023-06-15, -4000 SPY held to the end of the run.)"""
    p, ss = _assignment_portal()
    strat = registry.get("S141")(width=5.0)     # width 10 debit >= 50% width -> entry skipped (fix 7)
    res = fx.run(strat, p, ss)
    f = res.fills
    assert (f["liquidity"] == "ASSIGNMENT").any(), "fixture must produce an early assignment of the short calls"
    assert any(q.close_reason == "leg_settled" for q in strat.positions[fx.UND])
    # sheet: risk = debit -> after the package is gone the account must be flat in the underlying
    assert _final_position(res, fx.UND) == 0.0, (
        f"assigned stock left open: {_final_position(res, fx.UND)} shares of {fx.UND} after the butterfly closed")


def test_finding_s141_loss_exceeds_debit_after_assignment():
    """OPT_DEBIT: max initial debit <= 0.50% NAV is the risk budget; with the assigned short stock left open the
    run's loss is unbounded by the debit (here the fixture is flat after the rally, so we only check the gross
    exposure the strategy carries vs the 5% NAV option-debit cap)."""
    p, ss = _assignment_portal()
    strat = registry.get("S141")(width=5.0)     # width 10 debit >= 50% width -> entry skipped (fix 7)
    res = fx.run(strat, p, ss)
    pos = res.positions
    last = pos[pos["ts"] == res.equity.index[-1]]
    gross = float(last["market_value"].abs().sum())
    nav = float(res.equity.iloc[-1])
    assert gross <= 0.05 * nav, f"gross exposure {gross:,.0f} = {gross / nav:.1%} NAV after an OPT_DEBIT package"


# ============================================================================ F2: S150 silent no-trade
def test_finding_s150_no_leap_listed_is_silent_ok():
    """S150 needs a 180-365 DTE long call. On a chain that never lists > 70 DTE (DoltHub-like: max 65 DTE,
    2020+) the strategy can never trade; the run must say so (NOT_TESTABLE, a fidelity/data reason, or a
    declared activity_note) instead of status OK with zero fills and the reason only in ctx.log."""
    n = 300
    ss = fx.sessions(n)
    closes = np.round(80 * np.exp(0.0015 * np.arange(n)), 2)
    p = fx.make_portal(closes, ss, chain_from=240, max_dte=70)
    strat = registry.get("S150")()
    res = fx.run(strat, p, ss)
    n_fills = 0 if res.fills is None else len(res.fills)
    reasons = " ".join(str(r.get("reason", "")) for r in (res.fidelity or {}).get("reasons", []))
    surfaced = (res.status != RunStatus.OK) or ("expiry" in reasons.lower()) or bool(getattr(strat, "activity_note", ""))
    assert n_fills == 0
    assert surfaced, "S150 cannot trade (no LEAP listed) but the run is OK with no surfaced reason"


# ============================================================================ F3: S156 zero-upside check not enforced at fill
def test_finding_s156_credit_below_call_width_at_fill():
    """S156: 'require total credit >= call vertical width if targeting zero upside loss'. The check is made on
    the decision snapshot, but the opening combo is an unlimited market combo filled at the NEXT snapshot; if
    premiums fall in between, the package is opened with credit < call width (upside risk), violating the
    clause the baseline (require_no_upside=True) promises."""
    n = 60
    ss = fx.sessions(n)
    rng = np.random.default_rng(8)
    closes = np.round(100 * np.exp(np.cumsum(rng.normal(0, 0.003, n))), 2)
    vol = np.full(n, 0.40)
    vol[6:] = 0.12                       # implied vol collapses between decision (k=5) and fill (k=6)
    p = fx.make_portal(closes, ss, chain_from=5, vol=vol, skew=3.0, band=0.3)
    strat = registry.get("S156")()
    res = fx.run(strat, p, ss)
    co = fx.combo_orders(res)
    op = co[co["tag"].str.startswith("open")]
    assert len(op) >= 1
    oid = int(op["order_id"].iloc[0])
    lim = op["limit_price"].iloc[0]
    assert lim == lim and lim < 0, "baseline opening combo must carry a net-credit limit"
    f = fx.option_fills(res)
    f = f[f["order_id"] == oid]
    if len(f) == 0:      # resolution: the credit limit (>= call width) is not met after the IV collapse -> no fill
        assert op["status"].iloc[0] in ("CANCELLED", "EXPIRED", "WORKING")
        return
    assert len(f) == 3, "jade lizard opened"
    credit = -float((f["quantity"] * f["price"]).sum()) / float(f["quantity"].abs().min())
    calls = sorted(fx.parse_occ(s)[3] for s, q in zip(f["symbol"], f["quantity"]) if fx.parse_occ(s)[2] == "C")
    width = calls[1] - calls[0]
    assert credit >= width - 1e-9, f"opened with credit {credit:.2f} < call width {width:.2f} (upside loss)"


# ============================================================================ F4: S147 daily-close rehedge
class _S147Probe(S147):
    """Records, at every close while the straddle is open, the share-equivalent net delta AFTER the strategy's
    own rehedge orders (option delta + shares held + pending stock orders)."""

    def on_bar(self, ctx):
        super().on_bar(ctx)
        pos = self.first_live(fx.UND)
        if pos is None or pos.status != "OPEN":
            return
        cv = self.chain(ctx, fx.UND)
        dsh = self.package_delta_shares(ctx, cv, pos)
        net = dsh + ctx.position(fx.UND) + self.pending_stock(ctx, fx.UND)
        self.store.setdefault("net", []).append((ctx.session_date(), net / pos.units))


def test_finding_s147_daily_close_rehedge_only_beyond_threshold():
    """S147: 'Rehedge stock at delta drift >0.1 OR daily close'. Without intraday option quotes the drift
    trigger cannot be observed, but the daily-close rehedge can: at every close the stock hedge should bring
    net delta back to ~0. The implementation rehedges at the close only when |net delta| > threshold x 100
    shares, so drifts up to 10 shares per straddle (20 with threshold .20) stay unhedged overnight."""
    n = 360
    k = 322
    rng = np.random.default_rng(11)
    closes = np.round(100 * np.exp(np.cumsum(rng.normal(0, 0.016, n))), 2)
    p = fx.make_portal(closes, fx.sessions(n), chain_from=k, vol=0.10)
    strat = _S147Probe()
    fx.run(strat, p, fx.sessions(n))
    nets = strat.store.get("net", [])
    assert nets, "straddle must have been opened"
    worst = max(abs(x) for _, x in nets)
    assert worst <= 1.0, f"net delta left after the daily-close rehedge up to {worst:.1f} shares per straddle"


# ============================================================================ conformance probes
def test_conf_s126_entry_signal_fill_next_snapshot_and_delta():
    """S126: signal at the close (SMA200 + EMA20/50 cross), COMBO filled at the next snapshot, call delta
    closest to +0.50 inside 25-35 DTE (re-uses the independent scenario of the strategy tests)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "strategies", "options_income"))
    import oi_scenarios as sc_
    sc = sc_.get("S126")
    strat = registry.get("S126")(**sc.params)
    res = fx.run(strat, sc.portal(), sc.ss, start_i=sc.start_i)
    co = fx.combo_orders(res)
    o0 = co[co["tag"].str.startswith("open")].iloc[0]
    assert pd.Timestamp(o0["submitted_at"]) == fx.close_ts(sc.ss[sc.signal_i])
    legs = fx.legs_of_order(res, int(o0["order_id"]))
    assert legs == [("C", sc.legs[0][1], float(sc.units))]
    f = fx.option_fills(res)
    assert pd.Timestamp(f[f["order_id"] == o0["order_id"]]["ts"].iloc[0]) == fx.close_ts(sc.ss[sc.signal_i + 1])


@pytest.mark.parametrize("sid,ratios", [("S141", [1, -2, 1]), ("S142", [1, -2, 1]), ("S155", [1, -2, 1]),
                                        ("S151", [-1, 2]), ("S152", [-1, 2])])
def test_conf_structure_ratios(sid, ratios):
    """Butterflies 1:-2:1 (S155 far wing 1.5 x near width), backspreads 1:2 from the package builder."""
    n = 40
    ss = fx.sessions(n)
    closes = np.full(n, 100.0)
    p = fx.make_portal(closes, ss, chain_from=0, band=0.4, max_dte=70, vol=0.40)
    strat = registry.get(sid)()
    res = fx.run(strat, p, ss, end_i=3)
    co = fx.combo_orders(res)
    o0 = co[co["tag"].str.startswith("open")].iloc[0]
    legs = fx.legs_of_order(res, int(o0["order_id"]))
    units = int(np.gcd.reduce([int(abs(q)) for *_, q in legs]))
    by_strike = sorted(legs, key=lambda x: x[1])
    got = [int(q // units) for *_, q in by_strike]
    if sid == "S152":
        got = got[::-1]            # puts: short ATM-ish (higher K) first
    assert got == ratios
    if sid == "S155":
        ks = [k for _, k, _ in by_strike]
        assert ks[2] - ks[1] == pytest.approx(1.5 * (ks[1] - ks[0]))
    expiries = {fx.parse_occ(s)[1] for s in fx.option_fills(res)["symbol"]}
    assert len(expiries) == 1, "same-expiry legs"


# ============================================================================ F5: S130 dormant after first assignment
def test_finding_s130_stops_selling_puts_after_assignment():
    """S130 'Cash-secured 0.30 delta put': '... if assigned acquire shares and continue accounting'. With the
    assigned shares held ('hold') and counted in the 10% name cap, one assignment of the baseline size (2 SPY-
    priced contracts on $1M) leaves < 1 contract of cap, so the strategy cannot sell another put while it holds
    the shares. Resolution (adjudicated): either a new put is sold, or the run SURFACES the 'name cap reached'
    diagnostic (result.fidelity reasons, read by atlasbt.run.activity.diagnose) instead of going silent."""
    n = 70
    ss = fx.sessions(n)
    closes = np.r_[np.full(3, 400.0), np.linspace(400.0, 340.0, 8)[1:], np.full(n - 10, 340.0)]
    p = fx.make_portal(closes, ss, chain_from=0, vol=0.20, band=0.30)
    p._rf = 0.05                         # American put early-exercise test needs r > 0
    strat = registry.get("S130")()
    res = fx.run(strat, p, ss)
    f = res.fills
    asg = f[f["liquidity"] == "ASSIGNMENT"]
    assert len(asg), "fixture must produce a put assignment"
    t_asg = pd.Timestamp(asg["ts"].min())
    co = fx.combo_orders(res)
    later = co[co["tag"].str.startswith("open") & (pd.to_datetime(co["submitted_at"], utc=True) > t_asg)]
    held = _final_position(res, fx.UND)
    assert held > 0
    reasons = " ".join(str(r.get("reason", "")) for r in (res.fidelity or {}).get("reasons", []))
    assert len(later) >= 1 or "name cap reached" in reasons, (
        f"no put sold in the {n} sessions after the assignment ({held:.0f} shares held) and no diagnostic; "
        f"logs: {[l['msg'] for l in res.diagnostics.get('logs', [])][-2:]}")


# ============================================================================ conformance: covered call ex-div assignment
def test_conf_s128_early_assignment_on_exdiv_rebuys_and_rewrites():
    """S128: 'Buy back call at DTE<=5 or close on assignment; keep shares'. A deep-ITM covered call is early
    assigned the session before a $3 ex-date (engine OCC lifecycle); the strategy re-buys the 100-share lots at the
    next open and writes a new 25-35 DTE call once the shares are held (one call per 100 shares)."""
    p, ss = _assignment_portal(60)
    strat = registry.get("S128")()
    res = fx.run(strat, p, ss)
    f = res.fills
    asg = f[f["liquidity"] == "ASSIGNMENT"]
    assert len(asg) >= 1
    t_asg = pd.Timestamp(asg["ts"].min())
    buys = f[(f["symbol"] == fx.UND) & (f["liquidity"] != "ASSIGNMENT") & (pd.to_datetime(f["ts"], utc=True) > t_asg)]
    assert len(buys) and float(buys["quantity"].iloc[0]) > 0
    co = fx.combo_orders(res)
    new = co[co["tag"].str.startswith("open") & (pd.to_datetime(co["submitted_at"], utc=True) > t_asg)]
    assert len(new) >= 1
    o = fx.option_fills(res)
    o = o[o["order_id"] == int(new["order_id"].iloc[0])]
    shares_then = float(buys["quantity"].iloc[0])
    assert -float(o["quantity"].sum()) * 100 == pytest.approx(shares_then)


# ============================================================================ F6: S129 SPY variant sized at 10% NAV
def test_finding_s129_spy_variant_not_fully_collateralized_putwrite():
    """S129 'hold T-bill collateral sufficient for maximum ... put settlement; sell near ATM monthly put' is the
    Cboe PUT index (fully collateralized: notional ~ 100% NAV). The declared T2 variant underlying='SPY' is sized
    with the 10% single-name cap (OptionBase.size_cashsec: cap 1.0 only for SPX/NDX/RUT...), although the
    docstring says 'broad indexes / index ETF putwrites: 100%' -> the SPY putwrite is a 10%-NAV strategy."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "strategies", "options_income"))
    import oi_scenarios as sc_
    sc = sc_.get("S129")
    strat = registry.get("S129")(**sc.params)
    res = fx.run(strat, sc.portal(), sc.ss, start_i=sc.start_i)
    co = fx.combo_orders(res)
    o0 = co[co["tag"].str.startswith("open")].iloc[0]
    (_, k, q), = fx.legs_of_order(res, int(o0["order_id"]))
    notional = abs(q) * k * 100
    assert notional >= 0.8 * 1e6, f"SPY putwrite notional {notional:,.0f} = {notional / 1e6:.0%} of NAV"


# ============================================================================ F7: S141/S142 take-profit already met at entry
@pytest.mark.parametrize("sid", ["S141", "S142"])
def test_finding_fly_tp_already_met_at_entry_churns(sid):
    """S141/S142 'Close when market value >= 50% of max terminal payoff': with the grid width 20 on a $400
    underlying in a calm market (vol 10%) the ATM fly costs more than 50% of the width, so the exit is already
    true at the fill and the package is closed at the next close and re-opened the session after (churning the
    spread). S140 guards against exactly this (P&L-based TP), S141/S142 do not (undeclared)."""
    n = 30
    ss = fx.sessions(n)
    p = fx.make_portal(np.full(n, 400.0), ss, chain_from=0, vol=0.10, band=0.12)
    strat = registry.get(sid)(width=20.0)
    res = fx.run(strat, p, ss)
    co = fx.combo_orders(res)
    n_open = int(co["tag"].str.startswith("open").sum())
    tp = [q for q in strat.positions[fx.UND] if q.close_reason == "value_ge_50pct_max_payoff" and q.status == "CLOSED"]
    held = [len(fx.CAL.sessions(q.entry_session, q.meta["closed_session"])) - 1 for q in tp]
    assert n_open <= 2 and not any(h <= 1 for h in held), (
        f"{n_open} flies opened in {n} flat sessions; TP exits after {held} sessions (entry debit >= TP level)")
