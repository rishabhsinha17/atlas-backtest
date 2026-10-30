"""End-to-end known-answer validations of the whole stack on real cached data.

Each ``rep_*`` function runs the real engine (Backtester + FastPortal over cache/store) and compares it
with an independent reference; ``run_replications`` writes results/REPLICATION.md + replication.json.
The pytest ``tests/run/test_replicate.py`` runs the ``quick=True`` windows (skips without the cache).

1. ``rep_spy``      SPY buy-and-hold through the engine (default IBKR fees vs zero commission, dividends as
                    cash, and a dividend-reinvesting variant) vs Yahoo adjusted-close total return, plus an
                    exact apples-to-apples accounting reconciliation (raw closes + cash dividends + ledger
                    interest + fees must reproduce the engine NAV).
2. ``rep_s105``     S105 (60/40 SPY/IEF, month-end -> next-open rebalance) vs a pandas reference built from
                    total-return adjusted opens/closes.
3. ``rep_put``      S129 (SPX ATM put-write, synthetic T3 chain) through the full engine path vs the Cboe PUT
                    index. Since 2026-10-29 the production ``Portal.option_quotes`` falls back to synthetic
                    session-close snapshots; before that the runner's ``SyntheticQuotePortal`` bridge did the
                    same (kept for comparison, ``bridge=True``).
4. ``rep_umd``      12-2 momentum decile spread (S085 logic) on the PIT S&P 500 universe vs Ken French UMD;
   ``rep_hml``      book-to-market tercile spread (S091 logic, XBRL book equity) vs HML. Both use single-stock
                    histories -> PROVISIONAL while the integrator's stock-data hold is active.
5. ``rep_random``   coin-flip entries (random side, fixed holding period) on SPY: gross return at the mid must be
                    ~0, every fill must be at or worse than the reference mid, net = gross - costs.
6. ``rep_lookahead`` a perfect-foresight strategy must be rejected by the PIT guard (LookaheadError); clipped
                    queries must never return a future bar.
"""
from __future__ import annotations

import math
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from atlasbt.core.types import Fidelity, LookaheadError, OptionQuote
from atlasbt.engine.strategy import Requirement, Strategy

from .common import DEFAULT_RESULTS, adjusted_close, jsonable, to_session_dates, write_json
from .fastportal import FastPortal

LAST = "2026-10-27"
#: see RunOptions.use_quotes (engine 'auto' = quotes only where they cover the bar window)
USE_QUOTES = "auto"


# ===================================================================== validation strategies
class BuyHold(Strategy):
    """Buy whole shares with (1 - buffer) of cash at the first open; optionally reinvest cash dividends
    (whole shares, next open after the cash is received)."""

    id = "R_BUYHOLD"
    name = "validation: buy and hold"
    bar_freq = "1d"
    asset = "ETF"
    universe = ["SPY"]
    params = {"symbol": "SPY", "reinvest": False, "buffer": 0.002}
    requires = [Requirement("bars", "1d")]

    def __init__(self, **p):
        super().__init__(**p)
        self.universe = [self.params["symbol"]]

    def on_bar(self, ctx):
        s = self.params["symbol"]
        if ctx.open_orders(s):
            return
        px = ctx.mark(s)
        if not (px and px > 0):
            return
        pos = ctx.position(s)
        if pos == 0:
            q = math.floor(ctx.cash * (1 - self.params["buffer"]) / px)
            if q > 0:
                ctx.order(s, q, tag="buy")
        elif self.params["reinvest"]:
            spare = ctx.cash - ctx.nav * self.params["buffer"]
            q = math.floor(spare / (px * 1.01))
            if q >= 1:
                ctx.order(s, q, tag="reinvest")


class CoinFlip(Strategy):
    """Random-entry control: when flat, with probability p enter long or short (fair coin) at the next
    open, ``notional`` x NAV; exit at the open after ``hold`` sessions."""

    id = "R_COINFLIP"
    name = "validation: random entries"
    bar_freq = "1d"
    asset = "ETF"
    universe = ["SPY"]
    params = {"symbol": "SPY", "seed": 0, "hold": 5, "p_enter": 0.5, "notional": 0.5}
    requires = [Requirement("bars", "1d")]

    def __init__(self, **p):
        super().__init__(**p)
        self.universe = [self.params["symbol"]]
        self.rng = np.random.default_rng(int(self.params["seed"]))
        self.held = 0

    def on_bar(self, ctx):
        s = self.params["symbol"]
        if ctx.open_orders(s):
            return
        pos = ctx.position(s)
        if pos != 0:
            self.held += 1
            if self.held >= self.params["hold"]:
                ctx.order(s, -pos, tag="exit")
            return
        self.held = 0
        u, side = self.rng.random(), self.rng.random()
        if u < self.params["p_enter"]:
            px = ctx.mark(s)
            q = math.floor(self.params["notional"] * ctx.nav / px)
            if q > 0:
                ctx.order(s, q if side < 0.5 else -q, tag="entry")


class PerfectForesight(Strategy):
    """Tries to read the future in several ways. 'future_as_of' / 'future_actions' must raise
    LookaheadError; 'clipped_end' asks for bars ending in the future without an as_of and records whether
    any returned bar is known after now (must never happen); then trades on tomorrow's close if it got it."""

    id = "R_FORESIGHT"
    name = "validation: perfect foresight"
    bar_freq = "1d"
    asset = "ETF"
    universe = ["SPY"]
    params = {"mode": "future_as_of"}
    requires = [Requirement("bars", "1d")]

    def __init__(self, **p):
        super().__init__(**p)
        self.leaks = 0
        self.queries = 0

    def on_bar(self, ctx):
        now = ctx.now
        pit = ctx.portal()
        m = self.params["mode"]
        self.queries += 1
        if m == "future_as_of":
            fut = pit.bars("SPY", "1d", now, now + pd.Timedelta(days=10), as_of=now + pd.Timedelta(days=10))
        elif m == "future_actions":
            fut = pit.corporate_actions(["SPY"], now, now + pd.Timedelta(days=120), as_of=now + pd.Timedelta(days=120))
        elif m == "history_peek":
            fut = ctx.history("SPY", 5, "1d")
            fut = fut[pd.to_datetime(fut["known_at"], utc=True) > now] if len(fut) and "known_at" in fut else fut.iloc[0:0]
        else:  # clipped_end
            fut = pit.bars("SPY", "1d", now - pd.Timedelta(days=5), now + pd.Timedelta(days=10))
            if fut is not None and len(fut) and "known_at" in fut.columns:
                fut = fut[pd.to_datetime(fut["known_at"], utc=True) > now]
            else:
                fut = fut.iloc[0:0] if fut is not None else []
        if fut is not None and len(fut):
            self.leaks += 1
            if ctx.position("SPY") == 0 and not ctx.open_orders("SPY"):
                ctx.order("SPY", 100)


# ===================================================================== synthetic quote bridge
class SyntheticQuotePortal(FastPortal):
    """FastPortal that, when no REAL option snapshot exists for a contract, serves the Portal's own
    synthetic chain as session-close snapshots (ts = official close, T3_MODELED). Used only by the PUT
    replication; production runs use FastPortal (real quotes only)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._syn: dict = {}
        from atlasbt.core.clock import nyse

        self._cal = nyse()

    def _syn_chain(self, und: str, ts: pd.Timestamp) -> pd.DataFrame:
        key = (und, ts.value)
        if key not in self._syn:
            ch = self.synthetic_chain(und, ts, 0, 400)
            self._syn[key] = ch
            if len(self._syn) > 64:
                self._syn.pop(next(iter(self._syn)))
        return self._syn[key]

    def _closes(self, start, end) -> list:
        s0 = to_session_dates([pd.Timestamp(start)])[0] - pd.Timedelta(days=1)
        s1 = to_session_dates([pd.Timestamp(end)])[0] + pd.Timedelta(days=1)
        out = []
        for d in self._cal.sessions(s0, s1):
            c = self._cal.open_close(d)[1]
            if pd.Timestamp(start) < c <= pd.Timestamp(end):
                out.append(c)
        return out

    def option_quotes(self, instrument, start, end, as_of):
        real = super().option_quotes(instrument, start, end, as_of)
        if len(real):
            return real
        hi = min(pd.Timestamp(end), pd.Timestamp(as_of))
        rows = []
        for c in self._closes(start, hi):
            ch = self._syn_chain(instrument.underlying, c)
            if not len(ch):
                continue
            hit = self._match_instrument(ch, instrument)
            if len(hit):
                rows.append(hit.iloc[0])
        if not rows:
            return real
        out = pd.DataFrame(rows)
        out = out.set_index(pd.DatetimeIndex(out["ts"], name="ts"))
        out["fidelity"] = int(Fidelity.T3_MODELED)
        out.attrs.update({"fidelity": int(Fidelity.T3_MODELED), "synthetic": True})
        return out

    def option_quote(self, instrument, as_of, max_stale_days: float = 4.0):
        q = super().option_quote(instrument, as_of, max_stale_days)
        if q is not None:
            return q
        a = pd.Timestamp(as_of)
        closes = self._closes(a - pd.Timedelta(days=6), a)
        if not closes:
            return None
        c = closes[-1]
        ch = self._syn_chain(instrument.underlying, c)
        hit = self._match_instrument(ch, instrument) if len(ch) else ch
        if not len(hit):
            return None
        r = hit.iloc[0]
        return OptionQuote(instrument=instrument, ts=c, bid=float(r["bid"]), ask=float(r["ask"]),
                           underlying_price=float(r["underlying_price"]), iv=float(r["iv"]),
                           delta=float(r["delta"]), gamma=float(r["gamma"]), vega=float(r["vega"]),
                           theta=float(r["theta"]), fidelity=Fidelity.T3_MODELED)


# ===================================================================== helpers
def _bt(strategy, portal, start, end, **cfg):
    from atlasbt.engine import Backtester, BacktestConfig

    cfg.setdefault("use_quotes", USE_QUOTES)
    return Backtester(strategy, portal, BacktestConfig(**cfg)).run(start, end)


def _ann(total: float, days: float) -> float:
    return (1 + total) ** (365.25 / max(days, 1)) - 1 if total > -1 else -1.0


def _ledger_sum(res, reasons) -> pd.Series:
    lg = res.ledger
    if lg is None or not len(lg):
        return pd.Series(dtype=float)
    m = lg["reason"].isin(reasons)
    s = lg.loc[m].groupby(pd.to_datetime(lg.loc[m, "ts"], utc=True))["amount"].sum()
    return s


# ===================================================================== 1. SPY
def rep_spy(portal=None, start="1993-02-01", end=LAST) -> dict:
    from atlasbt.execution.fees import ZeroCommissionRetail

    portal = portal or FastPortal()
    out = {"window": [start, end]}
    runs = {"default_fees": (BuyHold(), {}), "zero_commission": (BuyHold(), {"fees": ZeroCommissionRetail(historical=False)}),
            "reinvest_default_fees": (BuyHold(reinvest=True), {})}
    adj = adjusted_close(portal, "SPY", "1990-01-01", pd.Timestamp(end) + pd.Timedelta(days=3))
    rawdf = portal.bars("SPY", "1d", pd.Timestamp("1990-01-01", tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1),
                        as_of=pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1), adjusted=False)
    raw = rawdf["close"].astype(float)
    raw.index = to_session_dates(rawdf.index)
    acts = portal.corporate_actions(["SPY"], pd.Timestamp("1990-01-01"), pd.Timestamp(end), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=2))
    divs = pd.Series({pd.Timestamp(a.ex_date): a.amount for a in acts if a.type.value in ("CASH_DIVIDEND", "SPECIAL_DIVIDEND")},
                     dtype=float).sort_index()
    for name, (strat, cfg) in runs.items():
        t = time.time()
        res = _bt(strat, portal, start, end, **cfg)
        r: dict = {"status": res.status.value, "runtime_sec": round(time.time() - t, 1)}
        if res.status.value != "OK":
            r["reason"] = res.not_testable_reason
            out[name] = r
            continue
        f = res.fills.sort_values("ts")
        buy = f[f["tag"].astype(str) == "buy"] if "tag" in f.columns else f.iloc[:1]
        f0 = buy.iloc[0].copy()
        f0["quantity"] = float(buy["quantity"].sum())
        f0["price"] = float((buy["quantity"] * buy["price"]).sum() / buy["quantity"].sum())
        for c_ in ("commission", "exchange_fees", "regulatory_fees"):
            f0[c_] = float(buy[c_].sum())
        f = pd.concat([buy.iloc[:1], f[~f.index.isin(buy.index)]])
        d_fill = to_session_dates([f0["ts"]])[0]
        nav = res.equity.copy()
        nav.index = to_session_dates(nav.index)
        nav = nav[nav.index >= d_fill]
        days = (nav.index[-1] - nav.index[0]).days
        eng_tot = nav.iloc[-1] / nav.iloc[0] - 1
        a = adj.reindex(nav.index)
        ref_tot = a.iloc[-1] / a.iloc[0] - 1
        r.update({"fill_date": str(d_fill.date()), "shares": float(f0["quantity"]), "fill_price": float(f0["price"]),
                  "fill_reference_mid": float(f0.get("reference_mid", np.nan)),
                  "entry_cost_bp": float((f0["price"] / f0.get("reference_mid", np.nan) - 1) * 1e4),
                  "entry_fees_usd": float(f0["commission"] + f0["exchange_fees"] + f0["regulatory_fees"]),
                  "engine_cagr": _ann(eng_tot, days), "adjclose_cagr": _ann(ref_tot, days),
                  "td_vs_adjclose_ann": _ann(eng_tot, days) - _ann(ref_tot, days), "years": days / 365.25,
                  "n_fills": int(len(f))})
        if name != "reinvest_default_fees":
            # apples to apples: raw close x shares + cash dividends (ex-date entitlement) + ledger interest
            sh = float(f0["quantity"])
            cash0 = float(res.meta["initial_capital"]) - sh * float(f0["price"]) - r["entry_fees_usd"]
            dv = divs[(divs.index > d_fill) & (divs.index <= nav.index[-1])] * sh
            cum_div = dv.reindex(nav.index, fill_value=0.0).cumsum()
            inter = _ledger_sum(res, ["CASH_INTEREST", "MARGIN_INTEREST", "BORROW_FEE", "SHORT_REBATE"])
            inter.index = to_session_dates(inter.index)
            inter = inter.groupby(level=0).sum().reindex(nav.index, fill_value=0.0).cumsum()
            fees_later = f.iloc[1:]
            ref_b = cash0 + sh * raw.reindex(nav.index) + cum_div + inter
            resid = (nav - ref_b) / nav
            ref_b_noint = cash0 + sh * raw.reindex(nav.index) + cum_div
            r.update({"accounting_residual_max_abs_frac_nav": float(resid.abs().max()),
                      "accounting_residual_end_usd": float((nav - ref_b).iloc[-1]),
                      "cash_interest_total_usd": float(inter.iloc[-1]), "later_fills": int(len(fees_later)),
                      "dividends_total_usd": float(cum_div.iloc[-1]),
                      "dividends_as_cash_cagr_no_interest": _ann(ref_b_noint.iloc[-1] / ref_b_noint.iloc[0] - 1, days),
                      "div_reinvestment_drag_ann": _ann(ref_b_noint.iloc[-1] / ref_b_noint.iloc[0] - 1, days)
                      - _ann(ref_tot, days),
                      "cash_interest_contrib_ann": _ann(eng_tot, days) - _ann(ref_b_noint.iloc[-1] / ref_b_noint.iloc[0] - 1, days)})
        out[name] = r
    d, z = out.get("default_fees", {}), out.get("zero_commission", {})
    if "engine_cagr" in d and "engine_cagr" in z:
        out["fee_effect_ann"] = d["engine_cagr"] - z["engine_cagr"]
    rv = out.get("reinvest_default_fees", {})
    out["pass"] = bool(d.get("accounting_residual_max_abs_frac_nav", 1) < 1e-4
                       and abs(rv.get("td_vs_adjclose_ann", 1)) < 0.0020)
    out["criteria"] = ("accounting residual < 1 bp of NAV (dividends-as-cash runs); reinvesting run within "
                       "20 bp/yr of the Adj-Close total return")
    return out


# ===================================================================== 2. S105
def _s105_reference(portal, start, end, w_eq=0.60, first_session=None) -> pd.Series:
    from atlasbt.core.clock import nyse

    cal = nyse()
    endts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    px = {}
    for s in ("SPY", "IEF"):
        b = portal.bars(s, "1d", pd.Timestamp("2002-01-01", tz="UTC"), endts, as_of=endts, adjusted=True)
        b.index = to_session_dates(b.index)
        px[s] = b[["open", "close"]].astype(float)
    sess = cal.sessions(pd.Timestamp(first_session or start), pd.Timestamp(end))
    w = {"SPY": w_eq, "IEF": 1 - w_eq}
    units = None
    nav_prev_close = 1.0
    out = {}
    reb_next = True
    for d in sess:
        o = {s: px[s].loc[d, "open"] for s in w}
        c = {s: px[s].loc[d, "close"] for s in w}
        if units is None or reb_next:
            v_open = nav_prev_close if units is None else sum(units[s] * o[s] for s in w)
            units = {s: w[s] * v_open / o[s] for s in w}
        nav = sum(units[s] * c[s] for s in w)
        out[d] = nav
        nav_prev_close = nav
        reb_next = cal.is_last_session_of_month(d)
    return pd.Series(out)


def rep_s105(portal=None, start="2003-07-30", end=LAST) -> dict:
    from atlasbt.strategies import registry as R

    portal = portal or FastPortal()
    t = time.time()
    res = _bt(R.get("S105")(), portal, start, end)
    out = {"window": [start, end], "status": res.status.value, "runtime_sec": round(time.time() - t, 1)}
    if res.status.value != "OK" or not len(res.fills):
        out["reason"] = res.not_testable_reason or "no fills"
        out["pass"] = False
        return out
    first = to_session_dates([res.fills["ts"].min()])[0]
    nav = res.equity.copy()
    nav.index = to_session_dates(nav.index)
    nav = nav[nav.index >= first]
    ref = _s105_reference(portal, start, end, float(R.get("S105").params["equity"]), first)
    ref = ref.reindex(nav.index)
    re_, rr = nav.pct_change().dropna(), ref.pct_change().dropna()
    j = re_.index.intersection(rr.index)
    days = (nav.index[-1] - nav.index[0]).days
    e_c = _ann(nav.iloc[-1] / nav.iloc[0] - 1, days)
    r_c = _ann(ref.iloc[-1] / ref.iloc[0] - 1, days)
    diff = re_[j] - rr[j]
    out.update({"first_fill": str(first.date()), "engine_cagr": e_c, "reference_cagr": r_c, "td_ann": e_c - r_c,
                "daily_corr": float(re_[j].corr(rr[j])), "tracking_error_ann": float(diff.std() * math.sqrt(252)),
                "n_days": int(len(j)), "n_fills": int(len(res.fills)),
                "costs_usd": float((res.fills[["commission", "exchange_fees", "regulatory_fees"]].sum().sum())),
                "spread_impact_usd": float(res.fills[["spread_cost", "impact_cost"]].sum().sum()),
                "cash_interest_usd": float(_ledger_sum(res, ["CASH_INTEREST"]).sum())})
    out["pass"] = bool(out["daily_corr"] > 0.99 and abs(out["td_ann"]) < 0.005)
    out["criteria"] = "daily return correlation > 0.99 and |annualised tracking difference| < 50 bp"
    return out


# ===================================================================== 3. PUT index
def rep_put(start="2008-01-02", end=LAST, initial_cash=1e8, bridge: bool = False) -> dict:
    """bridge=False (default since 2026-10-29): the production path, Portal.option_quotes' own synthetic
    fallback. bridge=True: the runner's SyntheticQuotePortal (pre-fallback validation)."""
    from atlasbt.strategies import registry as R

    portal = SyntheticQuotePortal() if bridge else FastPortal()
    t = time.time()
    strat = R.get("S129")()
    res = _bt(strat, portal, start, end, initial_cash=initial_cash, option_margin="cboe", option_lifecycle="occ")
    out = {"window": [start, end], "status": res.status.value, "runtime_sec": round(time.time() - t, 1),
           "params": strat.params,
           "quote_path": "runner SyntheticQuotePortal bridge" if bridge else "production Portal.option_quotes (synthetic fallback)"}
    if res.status.value != "OK":
        out.update(reason=res.not_testable_reason, traceback=(res.diagnostics or {}).get("traceback", "")[-1500:], **{"pass": False})
        return out
    out["n_fills"] = int(len(res.fills))
    if not len(res.fills):
        out.update(reason="no option fills", **{"pass": False})
        return out
    nav = res.equity.copy()
    nav.index = to_session_dates(nav.index)
    first = to_session_dates([res.fills["ts"].min()])[0]
    nav = nav[nav.index >= first - pd.Timedelta(days=1)]
    put = adjusted_close(portal, "PUT", "2006-01-01", pd.Timestamp(end) + pd.Timedelta(days=3))
    me = nav.resample("ME").last().pct_change().dropna()
    mp = put.resample("ME").last().pct_change().reindex(me.index).dropna()
    j = me.index.intersection(mp.index)[1:]          # skip the partial first month
    me, mp = me[j], mp[j]
    yrs = len(j) / 12
    e_a = float(np.prod(1 + me) ** (1 / yrs) - 1)
    p_a = float(np.prod(1 + mp) ** (1 / yrs) - 1)
    opt = res.fills[res.fills["asset_class"].astype(str).str.contains("OPT")] if "asset_class" in res.fills else res.fills
    cov = []
    for _, fr in opt.iterrows():
        if fr["quantity"] < 0:
            try:
                k = float(str(fr["symbol"])[-8:]) / 1000
                d = to_session_dates([fr["ts"]])[0]
                nv = nav[nav.index <= d]
                if len(nv):
                    cov.append(abs(fr["quantity"]) * 100 * k / nv.iloc[-1])
            except Exception:
                pass
    out.update({"months": int(len(j)), "engine_ann_return": e_a, "put_index_ann_return": p_a, "diff_ann": e_a - p_a,
                "tracking_error_ann": float((me - mp).std() * math.sqrt(12)), "monthly_corr": float(me.corr(mp)),
                "mean_notional_coverage": float(np.mean(cov)) if cov else None,
                "fidelity": (res.fidelity or {}).get("grade"),
                "options_agent_reference": "calibrate.replicate_put_index (no engine): TE 2.63%/yr, corr 0.979, "
                                           "+1.5%/yr rich (2008-01..2026-09, at mid)"})
    out["pass"] = bool(out["monthly_corr"] > 0.9 and out["tracking_error_ann"] < 0.05)
    out["criteria"] = "monthly correlation > 0.9 and tracking error < 5%/yr (options agent: 2.6%/yr without the engine)"
    return out


# ===================================================================== 4. UMD / HML
def _sp500_panel(portal, start, end) -> tuple[pd.DataFrame, dict]:
    """Monthly total-return closes for every symbol that was ever a PIT S&P 500 member in the window, and
    the membership at each month end."""
    from atlasbt.core.clock import nyse

    cal = nyse()
    months = pd.date_range(pd.Timestamp(start) - pd.DateOffset(months=14), end, freq="ME")
    mem = {}
    allsyms = set()
    for m in months:
        d = cal.sessions(m - pd.Timedelta(days=10), m)[-1]
        u = portal.universe("SP500", cal.open_close(d)[1])
        mem[m] = set(u)
        allsyms |= set(u)
    endts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=2)
    cols = {}
    for s in sorted(allsyms):
        try:
            b = portal.bars(s, "1d", pd.Timestamp(months[0] - pd.Timedelta(days=40), tz="UTC"), endts, as_of=endts,
                            adjusted=True)
        except Exception:
            continue
        if b is None or not len(b):
            continue
        c = b["close"].astype(float)
        c.index = to_session_dates(b.index)
        cols[s] = c.resample("ME").last()
    P = pd.DataFrame(cols).reindex(months)
    return P, mem


def rep_umd(portal=None, start="2000-01-31", end="2026-09-30", decile=0.1) -> dict:
    portal = portal or FastPortal()
    t = time.time()
    P, mem = _sp500_panel(portal, start, end)
    R1 = P.pct_change(fill_method=None)
    mom = P.shift(2) / P.shift(12) - 1                     # 12-2: return from t-12 to t-2 (month ends)
    rows = {}
    for i, m in enumerate(P.index[:-1]):
        if m < pd.Timestamp(start):
            continue
        nxt = P.index[i + 1]
        u = [s for s in mem[m] if s in P.columns]
        sig = mom.loc[m, u].dropna()
        ret = R1.loc[nxt, sig.index]
        ok = ret.notna()
        sig, ret = sig[ok], ret[ok]
        if len(sig) < 50:
            continue
        q_hi, q_lo = sig.quantile(1 - decile), sig.quantile(decile)
        rows[nxt] = {"long": ret[sig >= q_hi].mean(), "short": ret[sig <= q_lo].mean(), "n": int(len(sig))}
    S = pd.DataFrame(rows).T
    S["spread"] = S["long"] - S["short"]
    ff = portal.store.read("factors", "french", "F-F_MOMENTUM_FACTOR")
    umd = ff["Mom"].astype(float)
    umd.index = pd.DatetimeIndex(umd.index).to_period("M").to_timestamp("M")
    S.index = pd.DatetimeIndex(S.index).to_period("M").to_timestamp("M")
    j = S.index.intersection(umd.index)
    out = {"window": [str(j.min().date()), str(j.max().date())], "months": int(len(j)),
           "corr_full": float(S.loc[j, "spread"].corr(umd[j])), "runtime_sec": round(time.time() - t, 1),
           "spread_ann_mean": float(S.loc[j, "spread"].mean() * 12), "umd_ann_mean": float(umd[j].mean() * 12),
           "median_names": float(S["n"].median())}
    for a, b in (("2000-01-01", "2011-12-31"), ("2012-01-01", "2026-12-31")):
        k = j[(j >= a) & (j <= b)]
        if len(k) > 24:
            out[f"corr_{a[:4]}_{b[:4]}"] = float(S.loc[k, "spread"].corr(umd[k]))
    out["pass"] = bool(out["corr_full"] > 0.3)
    out["criteria"] = "positive correlation (> 0.3) with UMD; < 1 expected (S&P 500 only, equal weight, deciles)"
    return out


def rep_hml(portal=None, start="2010-06-30", end="2026-09-30") -> dict:
    """B/M tercile spread: book equity = latest stockholders_equity filed (known_at <= month end), market
    equity = shares_outstanding (latest filed) x raw close; top 30% B/M minus bottom 30%, equal weight."""
    portal = portal or FastPortal()
    t = time.time()
    P, mem = _sp500_panel(portal, start, end)
    R1 = P.pct_change(fill_method=None)
    endts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=2)
    syms = list(P.columns)
    be, sh = {}, {}
    for s in syms:
        try:
            fs = portal._fund_symbol(s) if hasattr(portal, "_fund_symbol") else s   # key 'T~CIK' -> XBRL store symbol
            raw = portal.store.read("fundamentals", "sec", fs) if fs else None
        except Exception:
            raw = None
        if raw is None or not len(raw):
            continue
        for fld, dst in (("stockholders_equity", be), ("shares_outstanding", sh)):
            x = raw[raw["field"] == fld][["known_at", "value", "period_end"]].dropna()
            if len(x):
                x = x.sort_values(["known_at", "period_end"])
                k = pd.to_datetime(x["known_at"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
                dst[s] = pd.Series(x["value"].astype(float).to_numpy(), index=k).groupby(level=0).last()
    rawpx = {}
    for s in syms:
        try:
            b = portal.bars(s, "1d", pd.Timestamp(P.index[0] - pd.Timedelta(days=40), tz="UTC"), endts, as_of=endts,
                            adjusted=False)
            c = b["close"].astype(float)
            c.index = to_session_dates(b.index)
            rawpx[s] = c.resample("ME").last()
        except Exception:
            pass
    RP = pd.DataFrame(rawpx).reindex(P.index)
    rows = {}
    for i, m in enumerate(P.index[:-1]):
        if m < pd.Timestamp(start):
            continue
        nxt = P.index[i + 1]
        bm = {}
        for s in mem[m]:
            if s not in be or s not in sh or s not in RP.columns:
                continue
            b_ = be[s][be[s].index <= m]
            h_ = sh[s][sh[s].index <= m]
            px = RP.loc[m, s]
            if len(b_) and len(h_) and np.isfinite(px) and h_.iloc[-1] > 0 and b_.iloc[-1] > 0:
                bm[s] = b_.iloc[-1] / (h_.iloc[-1] * px)
        sig = pd.Series(bm)
        ret = R1.loc[nxt, sig.index] if len(sig) else pd.Series(dtype=float)
        ok = ret.notna()
        sig, ret = sig[ok], ret[ok]
        if len(sig) < 50:
            continue
        rows[nxt] = {"long": ret[sig >= sig.quantile(0.7)].mean(), "short": ret[sig <= sig.quantile(0.3)].mean(),
                     "n": int(len(sig))}
    S = pd.DataFrame(rows).T
    if not len(S):
        return {"pass": False, "reason": "insufficient fundamentals coverage"}
    S["spread"] = S["long"] - S["short"]
    ff = portal.store.read("factors", "french", "F-F_RESEARCH_DATA_FACTORS")
    hml = ff["HML"].astype(float)
    hml.index = pd.DatetimeIndex(hml.index).to_period("M").to_timestamp("M")
    S.index = pd.DatetimeIndex(S.index).to_period("M").to_timestamp("M")
    j = S.index.intersection(hml.index)
    out = {"window": [str(j.min().date()), str(j.max().date())], "months": int(len(j)),
           "corr_full": float(S.loc[j, "spread"].corr(hml[j])), "median_names": float(S["n"].median()),
           "spread_ann_mean": float(S.loc[j, "spread"].mean() * 12), "hml_ann_mean": float(hml[j].mean() * 12),
           "runtime_sec": round(time.time() - t, 1)}
    out["pass"] = bool(out["corr_full"] > 0.3)
    out["criteria"] = "positive correlation (> 0.3) with HML (S&P 500 only, XBRL book equity 2009+)"
    return out


# ===================================================================== 5. random entries
def _round_trips(fills: pd.DataFrame) -> pd.DataFrame:
    f = fills.sort_values("ts").reset_index(drop=True)
    pos, rows, cur = 0.0, [], []
    for _, r in f.iterrows():
        cur.append(r)
        pos += float(r["quantity"])
        if abs(pos) < 1e-9 and cur:
            ent, ex = cur[0], cur[-1]
            q = float(ent["quantity"])
            notional = abs(q) * float(ent["reference_mid"])
            cash = -sum(float(x["quantity"]) * float(x["price"]) for x in cur)
            fees = sum(float(x["commission"] + x["exchange_fees"] + x["regulatory_fees"]) for x in cur)
            gross_mid = q * (float(ex["reference_mid"]) - float(ent["reference_mid"]))
            rows.append({"entry_ts": ent["ts"], "side": np.sign(q), "notional": notional,
                         "gross_mid_ret": gross_mid / notional, "net_ret": (cash - fees) / notional,
                         "cost_ret": (gross_mid - (cash - fees)) / notional})
            cur = []
    return pd.DataFrame(rows, columns=["entry_ts", "side", "notional", "gross_mid_ret", "net_ret", "cost_ret"])


def rep_random(portal=None, start="2010-01-04", end=LAST, seeds=range(8), symbol="SPY",
               spread_model: Optional[str] = None) -> dict:
    portal = portal or FastPortal()
    t = time.time()
    trips, fills_all, per_seed = [], [], []
    cfg = {"spread_model": spread_model} if spread_model else {}
    for sd in seeds:
        res = _bt(CoinFlip(seed=int(sd), symbol=symbol), portal, start, end, **cfg)
        if res.status.value != "OK" or not len(res.fills):
            per_seed.append({"seed": int(sd), "status": res.status.value, "reason": res.not_testable_reason})
            continue
        rt = _round_trips(res.fills)
        trips.append(rt.assign(seed=int(sd)))
        fills_all.append(res.fills)
        per_seed.append({"seed": int(sd), "status": "OK", "n_trips": int(len(rt)),
                         "nav_end": float(res.equity.iloc[-1]), "mean_net_bp": float(rt["net_ret"].mean() * 1e4)})
    T = pd.concat(trips, ignore_index=True)
    F = pd.concat(fills_all, ignore_index=True)
    side = np.sign(F["quantity"].astype(float))
    slip = side * (F["price"].astype(float) - F["reference_mid"].astype(float))
    g = T["gross_mid_ret"]
    out = {"window": [start, end], "symbol": symbol, "spread_model": spread_model or _default_spread_model(),
           "seeds": len(per_seed), "n_round_trips": int(len(T)),
           "gross_mid_mean_bp": float(g.mean() * 1e4), "gross_mid_t": float(g.mean() / (g.std(ddof=1) / math.sqrt(len(g)))),
           "net_mean_bp": float(T["net_ret"].mean() * 1e4), "cost_mean_bp": float(T["cost_ret"].mean() * 1e4),
           "long_share": float((T["side"] > 0).mean()),
           "gross_mid_mean_long_bp": float(g[T["side"] > 0].mean() * 1e4),
           "gross_mid_mean_short_bp": float(g[T["side"] < 0].mean() * 1e4),
           "fills": int(len(F)), "fills_better_than_mid": int((slip < -1e-9).sum()),
           "mean_slippage_bp": float((slip / F["reference_mid"].astype(float)).mean() * 1e4),
           "per_seed": per_seed, "runtime_sec": round(time.time() - t, 1)}
    yr = pd.to_datetime(F["ts"], utc=True).dt.year
    out["slippage_bp_by_year"] = {int(k): round(float(v), 2) for k, v in
                                  (slip / F["reference_mid"].astype(float) * 1e4).groupby(yr.to_numpy()).mean().items()}
    out["spread_check"] = observed_vs_modelled_spread(portal, symbol)
    out["pass"] = bool(abs(out["gross_mid_t"]) < 3 and out["fills_better_than_mid"] == 0 and out["cost_mean_bp"] > 0
                       and out["net_mean_bp"] < out["gross_mid_mean_bp"])
    out["criteria"] = ("|t| of mean gross (mid-to-mid) trip return < 3; no fill better than its reference mid; "
                       "net = gross - costs with costs > 0")
    return out


def _default_spread_model() -> str:
    from atlasbt.engine import BacktestConfig

    c = BacktestConfig()
    return c.resolved_spread_model() if hasattr(c, "resolved_spread_model") else "corwin_schultz"


def rep_spreadcost(portal=None, start="2018-01-02", end=LAST, seeds=range(4), symbol="SPY") -> dict:
    """SPY coin-flip round-trip cost under the calibrated spread model vs legacy Corwin-Schultz (same seeds,
    same window), against the quoted SPY half-spread on the IEX day."""
    portal = portal or FastPortal()
    out = {"window": [start, end], "symbol": symbol}
    for m in ("calibrated", "corwin_schultz"):
        r = rep_random(portal, start, end, seeds, symbol, spread_model=m)
        out[m] = {k: r.get(k) for k in ("n_round_trips", "cost_mean_bp", "mean_slippage_bp", "gross_mid_mean_bp",
                                        "gross_mid_t", "net_mean_bp", "fills_better_than_mid", "slippage_bp_by_year")}
    out["quoted"] = observed_vs_modelled_spread(portal, symbol)
    c, l = out["calibrated"]["cost_mean_bp"], out["corwin_schultz"]["cost_mean_bp"]
    out["cs_over_calibrated_x"] = (l / c) if c else None
    q = (out["quoted"] or {}).get("iex_median_half_spread_bp")
    out["calibrated_half_spread_slippage_vs_quoted_x"] = (out["calibrated"]["mean_slippage_bp"] / q) if q else None
    out["pass"] = bool(c is not None and l is not None and c < l and out["calibrated"]["fills_better_than_mid"] == 0)
    out["criteria"] = ("calibrated round-trip cost below Corwin-Schultz on the same trades; no fill better than mid. "
                       "Per-fill slippage includes impact (k*sigma*sqrt(Q/ADV)) on top of the half spread.")
    return out


def observed_vs_modelled_spread(portal, symbol="SPY") -> dict:
    """Quoted half-spread on the only real quote day(s) in the cache (IEX TOPS) vs the engine's bar-mode
    spread estimate (execution.spread.SpreadModel, Corwin-Schultz) for the same session."""
    from atlasbt.execution.spread import SpreadModel

    out = {}
    try:
        q = portal.store.read("quotes", "iex", symbol)
        q = q[(q["bid"] > 0) & (q["ask"] > q["bid"])]
        idx = pd.DatetimeIndex(q.index)
        et = idx.tz_convert("America/New_York")
        rth = (et.hour * 60 + et.minute >= 570) & (et.hour * 60 + et.minute < 960)
        q = q[rth]
        hs = (q["ask"] - q["bid"]) / (q["ask"] + q["bid"]) * 1e4
        day = to_session_dates(idx[rth][:1])[0]
        out.update({"quote_day": str(day.date()), "iex_median_half_spread_bp": float(hs.median()),
                    "iex_mean_half_spread_bp": float(hs.mean()), "n_quotes": int(len(q))})
        e = pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1)
        b = portal.bars(symbol, "1d", pd.Timestamp(day, tz="UTC") - pd.Timedelta(days=60), e, as_of=e, adjusted=True)
        ss = SpreadModel().series(b["high"].to_numpy(float), b["low"].to_numpy(float), b["close"].to_numpy(float))
        out["model_half_spread_bp_same_day"] = float(ss.est[-1] * 1e4 / 2)
        out["overstatement_x"] = out["model_half_spread_bp_same_day"] / out["iex_median_half_spread_bp"]
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


# ===================================================================== 6. lookahead
def rep_lookahead(portal=None, start="2020-01-02", end="2020-03-31") -> dict:
    portal = portal or FastPortal()
    out = {"window": [start, end], "modes": {}}
    for mode, expect in (("future_as_of", "raise"), ("future_actions", "raise"), ("clipped_end", "no_leak"),
                         ("history_peek", "no_leak")):
        st = PerfectForesight(mode=mode)
        rec = {"expect": expect}
        try:
            res = _bt(st, portal, start, end)
            rec.update({"raised": False, "status": res.status.value, "leaks": st.leaks, "queries": st.queries,
                        "n_fills": int(len(res.fills)),
                        "guard_violations": (res.diagnostics or {}).get("lookahead_guard_violations")})
        except LookaheadError as e:
            rec.update({"raised": True, "error": f"LookaheadError: {str(e)[:200]}", "queries": st.queries,
                        "leaks": st.leaks})
        except Exception as e:
            rec.update({"raised": False, "error": f"{type(e).__name__}: {e}"})
        rec["pass"] = bool(rec.get("raised")) if expect == "raise" else (not rec.get("raised") and rec.get("leaks") == 0
                                                                          and rec.get("queries", 0) > 0)
        out["modes"][mode] = rec
    out["pass"] = all(m["pass"] for m in out["modes"].values())
    out["criteria"] = "explicit future as_of -> LookaheadError; clipped queries never return a bar known after now"
    return out


# ===================================================================== driver
QUICK = {"spy": dict(start="2019-01-02", end="2021-12-31"), "s105": dict(start="2019-01-02", end="2021-12-31"),
         "put": dict(start="2016-01-04", end="2017-12-29", initial_cash=1e8),
         "umd": dict(start="2014-01-31", end="2020-12-31"), "hml": dict(start="2014-01-31", end="2020-12-31"),
         "random": dict(start="2018-01-02", end="2020-12-31", seeds=range(3)),
         "lookahead": dict(start="2020-01-02", end="2020-02-28"),
         "spreadcost": dict(start="2019-01-02", end="2020-12-31", seeds=range(2))}
FUNCS = {"spy": rep_spy, "s105": rep_s105, "put": rep_put, "umd": rep_umd, "hml": rep_hml, "random": rep_random,
         "lookahead": rep_lookahead, "spreadcost": rep_spreadcost}


def run_replications(results_dir: Path = DEFAULT_RESULTS, only=None, quick: bool = False, log=print) -> dict:
    from .holds import stock_hold_active

    results_dir = Path(results_dir)
    path = results_dir / ("replication_quick.json" if quick else "replication.json")
    from .common import read_json

    allres = read_json(path, {}) or {}
    for k in (only or list(FUNCS)):
        kw = dict(QUICK[k]) if quick else {}
        log(f"replicate {k} {'(quick)' if quick else ''} ...")
        t = time.time()
        try:
            r = FUNCS[k](**kw)
        except Exception as e:
            r = {"pass": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
        if k in ("umd", "hml") and stock_hold_active():
            r["provisional"] = "single-stock data identity fix pending (integrator hold): result PROVISIONAL"
        r["elapsed_sec"] = round(time.time() - t, 1)
        allres[k] = r
        log(f"replicate {k}: pass={r.get('pass')} ({r['elapsed_sec']}s) {r.get('error', '')}")
        write_json(path, allres)
    if not quick:
        write_replication_md(results_dir / "REPLICATION.md", allres)
    return allres


def _p(v, k="pct"):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    if k == "pct":
        return f"{v:.2%}"
    if k == "bp":
        return f"{v:.1f} bp"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def write_replication_md(path: Path, R_: dict) -> None:
    from .common import environment_info

    env = environment_info()
    L = ["# End-to-end replication (known-answer) results", "",
         f"Generated {env['timestamp_utc']} | git {env['git_commit'][:12]} | code {env['code_sha256_16']}. "
         "Every number below comes from the real engine over the real cache (`python -m atlasbt.run replicate`); "
         "functions in `atlasbt/run/replicate.py`, quick-window pytest in `tests/run/test_replicate.py`.", "",
         __import__("atlasbt.run.notices", fromlist=["banner_md"]).banner_md(),
         "| Check | Result | Key numbers |", "|---|---|---|"]

    def row(name, r, txt):
        st = "PASS" if r.get("pass") else ("ERROR" if r.get("error") else "FAIL")
        if r.get("provisional"):
            st += " (STALE)" if "STALE" in str(r.get("provisional")) else " (provisional)"
        L.append(f"| {name} | {st} | {txt} |")

    s = R_.get("spy", {})
    if s:
        d, z, rv = s.get("default_fees", {}), s.get("zero_commission", {}), s.get("reinvest_default_fees", {})
        row("SPY buy & hold vs Adj Close", s,
            f"dividends as cash: TD {_p(d.get('td_vs_adjclose_ann'))}/yr (zero comm. {_p(z.get('td_vs_adjclose_ann'))}); "
            f"reinvested: TD {_p(rv.get('td_vs_adjclose_ann'))}/yr; accounting residual {_p(d.get('accounting_residual_max_abs_frac_nav'))} of NAV")
    s = R_.get("s105", {})
    if s:
        row("S105 60/40 vs pandas", s, f"corr {_p(s.get('daily_corr'), 'f')}, TD {_p(s.get('td_ann'))}/yr, TE {_p(s.get('tracking_error_ann'))}")
    s = R_.get("put", {})
    if s:
        row("S129 (engine) vs Cboe PUT", s, f"ann {_p(s.get('engine_ann_return'))} vs {_p(s.get('put_index_ann_return'))}, "
            f"TE {_p(s.get('tracking_error_ann'))}/yr, corr {_p(s.get('monthly_corr'), 'f')}"
            + (f"; BLOCKED: {s['blocked_by']}" if s.get("blocked_by") else ""))
        pv = s.get("previous_engine_run")
        if pv:
            row("S129 (engine, bridge, earlier run) vs Cboe PUT", pv,
                f"ann {_p(pv.get('engine_ann_return'))} vs {_p(pv.get('put_index_ann_return'))} over {pv.get('months')} months, "
                f"TE {_p(pv.get('tracking_error_ann'))}/yr, corr {_p(pv.get('monthly_corr'), 'f')} ({pv.get('note', '')[:120]})")
    s = R_.get("umd", {})
    if s:
        row("12-2 momentum decile spread vs UMD", s, f"corr {_p(s.get('corr_full'), 'f')} over {s.get('months')} months")
    s = R_.get("hml", {})
    if s:
        row("B/M tercile spread vs HML", s, f"corr {_p(s.get('corr_full'), 'f')} over {s.get('months')} months")
    s = R_.get("random", {})
    if s:
        row("Random entries (coin flip)", s, f"gross(mid) {_p(s.get('gross_mid_mean_bp'), 'bp')} (t {_p(s.get('gross_mid_t'), 'f')}), "
            f"net {_p(s.get('net_mean_bp'), 'bp')}, cost {_p(s.get('cost_mean_bp'), 'bp')} per trip; fills better than mid: {s.get('fills_better_than_mid')}")
    s = R_.get("random", {})
    sc = (s or {}).get("spread_check") or {}
    if sc and "error" not in sc:
        L.append(f"| Spread model vs real quotes (SPY, {sc.get('quote_day')}) | EVIDENCE | IEX quoted half-spread median "
                 f"{_p(sc.get('iex_median_half_spread_bp'), 'bp')} vs engine bar-mode (Corwin-Schultz) "
                 f"{_p(sc.get('model_half_spread_bp_same_day'), 'bp')} = {sc.get('overstatement_x', 0):.0f}x; "
                 f"engine slippage per fill by year under the default model ({s.get('spread_model', 'n/a')}): "
                 f"{s.get('slippage_bp_by_year')} |")
    s = R_.get("spreadcost", {})
    if s and "calibrated" in s:
        row("SPY coin-flip cost: calibrated vs Corwin-Schultz", s,
            f"round trip {_p(s['calibrated'].get('cost_mean_bp'), 'bp')} (calibrated) vs "
            f"{_p(s['corwin_schultz'].get('cost_mean_bp'), 'bp')} (CS) = {s.get('cs_over_calibrated_x') or 0:.1f}x; "
            f"per-fill slippage {_p(s['calibrated'].get('mean_slippage_bp'), 'bp')} vs quoted half-spread "
            f"{_p((s.get('quoted') or {}).get('iex_median_half_spread_bp'), 'bp')}")
    s = R_.get("lookahead", {})
    if s:
        row("Perfect foresight vs PIT guard", s, "; ".join(f"{k}: {'raised' if v.get('raised') else 'leaks=' + str(v.get('leaks'))}"
                                                     for k, v in (s.get("modes") or {}).items()))
    L.append("")
    for k, title in (("spy", "1. SPY buy-and-hold"), ("s105", "2. S105 60/40"), ("put", "3. Cboe PUT index via S129"),
                     ("umd", "4a. Momentum vs UMD"), ("hml", "4b. Value vs HML"), ("random", "5. Random entries"),
                     ("lookahead", "6. Perfect foresight"), ("spreadcost", "7. Spread model cost check")):
        if k not in R_:
            continue
        r = dict(R_[k])
        tb = r.pop("traceback", None)
        L += [f"## {title}", "", f"Criteria: {r.get('criteria', 'n/a')}", "", "```json",
              __import__("json").dumps(jsonable(r), indent=1)[:6000], "```", ""]
        if tb:
            L += ["```", tb[-1500:], "```", ""]
    L += ["## Interpretation notes", "",
          "* SPY: the engine pays dividends as cash (ex-date entitlement, assumed pay date ex+7d, cash earns the "
          "broker credit rate); Yahoo Adj Close assumes reinvestment at the ex-date close. The dividends-as-cash runs "
          "are therefore compared with an exact apples-to-apples reconstruction (raw close x shares + cash "
          "dividends + ledger interest); the reinvesting run (whole shares, next open after the cash arrives) is "
          "compared with Adj Close directly.",
          "* S105 reference rebalances at the next open after each month-end session with adjusted prices "
          "(dividends reinvested), fractional units, no costs; the engine uses whole shares, cash dividends, "
          "modelled spread/impact and IBKR fees.",
          "* PUT: S129 sells the listed near-ATM SPX monthly (AM) put at the synthetic bid through the engine (combo "
          "order, Cboe margin, OCC lifecycle, SOQ proxied by the SPX open); the PUT index rolls at a late-morning VWAP "
          "and invests in 1m/3m bills. Quote path recorded in `quote_path` (production synthetic fallback, or the "
          "runner bridge used before it existed).",
          "* UMD/HML: S&P 500 PIT universe only, equal-weighted deciles/terciles, adjusted closes (pre-2018 "
          "survivorship-biased, DATA.md §8); French factors use all NYSE/AMEX/Nasdaq stocks, value weights, "
          "size-neutral 2x3 sorts, so correlation must be positive but well below 1."]
    path.write_text("\n".join(L) + "\n")
