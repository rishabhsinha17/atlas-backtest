"""Statistical arbitrage / relative value: S106-S115, S123.

Module-wide conventions:

* Daily strategies signal at the session close; market orders fill at the next
  session open (engine). "after N days" / "N trading days later" = the exit
  order is submitted at the close of the N-th session after the entry signal,
  so it fills at the open of session N+1 (the position is held N sessions).
* PAIRS sizing via ``risk.sizing.size_pairs`` (5% long + 5% short of NAV,
  dollar neutral). The hedge ratio from the formation window is used only for
  the signal (sizing module rule) unless the sheet text says otherwise (S113).
* Basket strategies with PAIRS sizing (S109, S112, S113, S114) are traded as
  matched long/short pairs; the sheet gives no pair count, so the number of
  simultaneously open pairs is capped at ``max_pairs`` = 10 (100% NAV gross).
* Pairs never share a symbol, so closing a pair = closing both symbols'
  actual positions (partial fills included).
* Total-return prices = split + dividend back-adjusted closes as of the signal.
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from ..core.types import Fidelity, OrderType
from ..engine.strategy import Requirement, Strategy
from ..indicators import (engle_granger, kalman_hedge, linreg_slope, pair_spread_z, pca_residuals, rolling_ols,
                          select_pairs, sma, ssd_pairs, zscore)
from ..indicators import cross_section as xs
from ..risk.sizing import size_pairs
from .factors import (SECTOR_ETFS, SECTOR_SPDR_9, SURVIVORSHIP_NOTE, SecMaster, close_panel, closes, daily_returns,
                      is_month_end, is_week_end, ols_beta, rebalance_to, sector_etf)
from .registry import register

COMMON_PAIR_DEVIATIONS = [
    "Predeclared pair lists live in the module; param `pairs` (default None = whole list) only restricts it",
    "Signals at the session close; market orders fill at the next session open",
    "'after N days' = exit order at the close of the N-th session after the entry signal (fills next open)",
    "PAIRS sizing: size_pairs 5% long + 5% short of current NAV (dollar neutral); hedge ratio used for signals only",
]


class ReturnCache:
    """Incremental cache of daily total returns (session-date index). A return
    for day d is computed at d's close from closes adjusted as of that close;
    later back-adjustments rescale prices but not returns, so cached values stay
    valid (and PIT)."""

    def __init__(self, keep: int = 600):
        self.r: dict[str, pd.Series] = {}
        self.keep = keep

    def update(self, ctx, syms, need: int) -> None:
        cal = ctx.calendar()
        today = ctx.session_date()
        prev = cal.prev_session(today)
        for s in syms:
            cur = self.r.get(s)
            if cur is not None and len(cur) and cur.index[-1] == today:
                continue
            if cur is not None and len(cur) >= need and cur.index[-1] == prev:
                c = closes(ctx, s, 2)
                if len(c) == 2 and c.index[-1] == today and c.index[0] == prev:
                    self.r[s] = pd.concat([cur, pd.Series([c.iloc[1] / c.iloc[0] - 1.0], index=[today])]).iloc[-self.keep:]
                continue
            c = closes(ctx, s, need + 1)
            if len(c) >= 2:
                self.r[s] = (c / c.shift(1) - 1.0).iloc[1:].iloc[-self.keep:]

    def panel(self, syms, n: int) -> pd.DataFrame:
        d = {s: self.r[s] for s in syms if s in self.r}
        if not d:
            return pd.DataFrame()
        return pd.DataFrame(d).sort_index().iloc[-n:]


# =====================================================================================
class _PairBase(Strategy):
    bar_freq = "1d"
    sizing = "PAIRS"
    #: symbols that are hedges/overlays, never treated as orphans
    overlay_symbols: tuple = ()

    def init(self, ctx) -> None:
        self.bar_i = 0
        self.open: dict = {}          # pid -> {"long", "short", "sign", "entry_i", ...}
        self.events: list = []        # (session, pid, action, reason) for tests/diagnostics

    def on_bar(self, ctx) -> None:
        self.bar_i += 1
        self.step(ctx)

    def step(self, ctx) -> None:  # pragma: no cover
        raise NotImplementedError

    def held(self, pid) -> int:
        return self.bar_i - self.open[pid]["entry_i"]

    def open_pair(self, ctx, pid, long_sym: str, short_sym: str, *, gross: float = 0.10,
                  long_notional: Optional[float] = None, short_notional: Optional[float] = None, **meta) -> bool:
        pl, ps = float(ctx.mark(long_sym)), float(ctx.mark(short_sym))
        if not (pl == pl and ps == ps and pl > 0 and ps > 0):
            return False
        if long_notional is None:
            ql, qs = size_pairs(ctx.nav, pl, ps, gross=gross)
        else:
            ql, qs = math.trunc(long_notional / pl), -math.trunc(short_notional / ps)
        if ql == 0 or qs == 0:
            return False
        ctx.order(long_sym, ql, tag=f"entry_long:{pid}")
        ctx.order(short_sym, qs, tag=f"entry_short:{pid}")
        self.open[pid] = {"long": long_sym, "short": short_sym, "qty_long": ql, "qty_short": qs,
                          "entry_i": self.bar_i, "entry_session": ctx.session_date(), **meta}
        self.events.append((ctx.session_date(), pid, "entry", f"long {long_sym} short {short_sym}"))
        return True

    def close_pair(self, ctx, pid, reason: str, order_type: OrderType = OrderType.MARKET) -> None:
        p = self.open.pop(pid)
        for sym in (p["long"], p["short"]):
            ctx.cancel_all(sym)
            q = ctx.position(sym)
            if q:
                ctx.order(sym, -q, order_type, tag=f"exit:{reason}")
        self.events.append((ctx.session_date(), pid, "exit", reason))

    def legs(self) -> set:
        out = set()
        for p in self.open.values():
            out.update((p["long"], p["short"]))
        return out

    def flatten_orphans(self, ctx) -> None:
        keep = self.legs() | set(self.overlay_symbols)
        for s, q in ctx.positions().items():
            if q and s not in keep and not ctx.open_orders(s):
                ctx.order(s, -q, tag="exit:orphan")


def _pair_requires(symbols, critical=True, note="") -> list[Requirement]:
    return [Requirement("bars", "1d", symbols=tuple(symbols), critical=critical, note=note)]


# =====================================================================================
@register
class S106(_PairBase):
    id = "S106"
    name = "Distance pairs trading"
    asset = "STK"
    universe_name = "SP500"
    params = {"formation": 252, "z_entry": 2.0, "z_exit": 0.5, "z_stop": 4.0, "max_hold": 60, "n_pairs": 20,
              "trading_period": 126}
    grid = {"formation": [126, 252], "z_entry": [1.5, 2.0, 2.5], "z_exit": [0.0, 0.5]}
    requires = [Requirement("bars", "1d", note="daily bars for the PIT SP500 universe")]
    deviations = COMMON_PAIR_DEVIATIONS + [
        SURVIVORSHIP_NOTE,
        "Gatev-Goetzmann-Rouwenhorst schedule: pairs formed on the trailing `formation` sessions, traded for the "
        "next trading_period=126 sessions (GGR 6 months), then re-formed; pairs still open at re-formation are "
        "closed (exit reason 'period_end'); no trading on the formation bar itself",
        "Eligible = PIT SP500 members with a complete formation window; same sector = same SIC->GICS-like sector (current snapshot, T2) "
        "(security_master); best 20 = smallest SSD, greedy non-overlapping (indicators.select_pairs)",
        "z = (normalised spread - formation mean)/formation stdev (indicators.pair_spread_z, moments frozen at "
        "formation end); a pair is rich when z > 0 -> short a / long b",
        "Exit rule 'close at |z|<=x' generalised as sign(z_entry)*z <= x so z_exit=0 means 'z crosses 0'; stop when "
        "sign(z_entry)*z >= 4; entries require z_entry <= |z| < 4; re-entry allowed within the period",
        "Up to 20 pairs x 10% gross = 200% NAV; Reg-T initial margin may reject the last entries (engine)",
    ]
    spec = ("Form eligible same-sector stock pairs using 252d sum squared distance of normalized total-return "
            "prices; choose best 20 nonoverlapping pairs; spread=z of standardized normalized difference; enter "
            "short-rich long-cheap if |z|>=2. || Close pair at |z|<=0.5, z beyond 4, or after 60d; hedge equal "
            "dollars.")

    def init(self, ctx):
        super().init(ctx)
        self.sm = SecMaster()
        self.pairs: Optional[list[tuple[str, str]]] = None
        self.formed_i = -10**9
        self.f_start = self.f_end = None

    def form(self, ctx):
        for pid in list(self.open):
            self.close_pair(ctx, pid, "period_end")
        W = int(self.params["formation"])
        syms = self.symbols(ctx)
        self.sm.refresh(ctx)
        px = close_panel(ctx, syms, W)
        self.formed_i = self.bar_i
        self.pairs = []
        if px.empty or len(px) < W or px.index[-1] != ctx.session_date():
            return
        px = px.dropna(axis=1, how="any")
        groups = {s: self.sm.sector(s) or f"UNKNOWN:{s}" for s in px.columns}
        sel = select_pairs(ssd_pairs(px, groups), int(self.params["n_pairs"]), nonoverlapping=True)
        self.pairs = [(r["a"], r["b"]) for _, r in sel.iterrows()]
        self.f_start, self.f_end = px.index[0], px.index[-1]

    def step(self, ctx):
        p = self.params
        if self.pairs is None or self.bar_i - self.formed_i >= int(p["trading_period"]):
            self.form(ctx)
            return
        n = len(ctx.calendar().sessions(self.f_start, ctx.session_date()))
        for a, b in self.pairs:
            pid = (a, b)
            ca, cb = closes(ctx, a, n), closes(ctx, b, n)
            if len(ca) == 0 or len(cb) == 0 or ca.index[-1] != ctx.session_date() or cb.index[-1] != ctx.session_date():
                continue
            zs = pair_spread_z(ca, cb, self.f_start, self.f_end)
            z = float(zs["z"].iloc[-1])
            if not z == z:
                continue
            if pid in self.open:
                sg = self.open[pid]["sign"]
                if sg * z <= p["z_exit"]:
                    self.close_pair(ctx, pid, "converged")
                elif sg * z >= p["z_stop"]:
                    self.close_pair(ctx, pid, "stop_z")
                elif self.held(pid) >= p["max_hold"]:
                    self.close_pair(ctx, pid, "time")
            elif p["z_entry"] <= abs(z) < p["z_stop"]:
                if z > 0:
                    self.open_pair(ctx, pid, b, a, sign=1.0, z_at_entry=z)
                else:
                    self.open_pair(ctx, pid, a, b, sign=-1.0, z_at_entry=z)


# =====================================================================================
@register
class S107(_PairBase):
    id = "S107"
    name = "Cointegrated stock pairs"
    asset = "STK"
    universe_name = "SP500"
    params = {"p_value": 0.05, "z_entry": 2.0, "formation": 252, "sigma_window": 60, "max_hold": 30,
              "max_pairs": 10, "industry_map": "SIC"}
    grid = {"p_value": [0.01, 0.05], "z_entry": [1.5, 2.5], "formation": [126, 252]}
    requires = [Requirement("bars", "1d", note="daily bars for the PIT SP500 universe")]
    deviations = COMMON_PAIR_DEVIATIONS + [
        SURVIVORSHIP_NOTE,
        "Industry pairs = same 3-digit SIC industry group (baseline; latest SEC SIC, T2) or, with "
        "industry_map='GICS', same GICS sub-industry (gics_sub_industry_current: current members only, ~59% PIT "
        "coverage in 2012, T2); flagged once with coverage; NotTestableError if the column is missing or coverage "
        "< 50% of the PIT universe; "
        "Engle-Granger (indicators.engle_granger, MacKinnon p) on log total-return prices over the trailing "
        "`formation` sessions, y = alphabetically first symbol",
        "Monthly review at month-end close: open pairs re-tested (p >= threshold -> closed, 'coint_fail'); new "
        "pairs = passing pairs by ascending p, non-overlapping, up to max_pairs=10 active pairs (sheet gives no "
        "count); new pairs trade from the next session",
        "Spread s = log y - beta*log x with beta fixed at formation; level = mean of s over the formation window "
        "(recomputed on as-of-now adjusted prices, consistent with total returns); "
        "z = (s_t - level)/stdev(s over the 60 prior sessions t-60..t-1)",
        "Entry |z| > z_entry: z > 0 -> short y / long x; exit when z crosses 0 (sign change vs entry), coint "
        "failure at review, or 30 sessions",
        "Sizing dollar neutral (PAIRS profile); hedge beta used for the signal only",
    ]
    spec = ("For industry pairs use 252d Engle-Granger residual ADF p<0.05, OLS hedge beta fixed in formation, z "
            "residual using 60d prior rolling sigma; enter |z|>2. || Close z crosses 0, ADF cointegration fails at "
            "monthly review, or 30d.")

    def init(self, ctx):
        super().init(ctx)
        self.sm = SecMaster()
        self.active: Optional[dict] = None     # pid -> {"y","x","beta","f_start","f_end","since_i"}

    def _eg(self, px: pd.DataFrame, y: str, x: str):
        d = np.log(px[[y, x]].dropna())
        if len(d) < 0.95 * len(px):
            return None
        try:
            return engle_granger(d[y], d[x])
        except Exception:
            return None

    def review(self, ctx):
        p = self.params
        W = int(p["formation"])
        syms = self.symbols(ctx)
        self.sm.refresh(ctx, uses_sector=False)
        self.sm.require_industry(ctx, syms, p["industry_map"], "S107")
        px = close_panel(ctx, sorted(set(syms) | self.legs()), W)
        if px.empty or px.index[-1] != ctx.session_date():
            self.active = self.active or {}
            return
        px = px.iloc[-W:]
        keep = {}
        for pid in list(self.open):
            a = (self.active or {}).get(pid)
            r = self._eg(px, pid[0], pid[1]) if a and pid[0] in px and pid[1] in px else None
            if r is None or not r.pvalue < p["p_value"]:
                self.close_pair(ctx, pid, "coint_fail")
            else:
                keep[pid] = a
        used = {s for pid in keep for s in pid}
        ind: dict[str, list[str]] = {}
        for s in px.columns:
            if s in set(syms) and px[s].notna().all():
                k = self.sm.industry(s, p["industry_map"])
                if k:
                    ind.setdefault(k, []).append(s)
        cands = []
        for names in ind.values():
            for y, x in combinations(sorted(names), 2):
                r = self._eg(px, y, x)
                if r is not None and r.pvalue < p["p_value"]:
                    cands.append((r.pvalue, y, x, r.beta))
        cands.sort()
        for pv, y, x, beta in cands:
            if len(keep) >= int(p["max_pairs"]):
                break
            if y in used or x in used:
                continue
            keep[(y, x)] = {"y": y, "x": x, "beta": float(beta), "f_start": px.index[0], "f_end": px.index[-1],
                            "since_i": self.bar_i + 1, "pvalue": float(pv)}
            used.update((y, x))
        self.active = keep

    def step(self, ctx):
        p = self.params
        if self.active is None or is_month_end(ctx):
            self.review(ctx)
        cal = ctx.calendar()
        for pid, a in list((self.active or {}).items()):
            if self.bar_i < a["since_i"]:
                continue
            n = len(cal.sessions(a["f_start"], ctx.session_date()))
            cy, cx = closes(ctx, a["y"], n), closes(ctx, a["x"], n)
            d = pd.concat([cy.rename("y"), cx.rename("x")], axis=1).dropna()
            if len(d) == 0 or d.index[-1] != ctx.session_date():
                continue
            s = np.log(d["y"]) - a["beta"] * np.log(d["x"])
            level = float(s[(s.index >= a["f_start"]) & (s.index <= a["f_end"])].mean())
            prior = s.iloc[-1 - int(p["sigma_window"]):-1]
            if len(prior) < int(p["sigma_window"]):
                continue
            sd = float(prior.std(ddof=1))
            if not sd > 0:
                continue
            z = (float(s.iloc[-1]) - level) / sd
            if pid in self.open:
                sg = self.open[pid]["sign"]
                if sg * z <= 0:
                    self.close_pair(ctx, pid, "z_cross_0")
                elif self.held(pid) >= p["max_hold"]:
                    self.close_pair(ctx, pid, "time")
            elif abs(z) > p["z_entry"]:
                if z > 0:
                    self.open_pair(ctx, pid, a["x"], a["y"], sign=1.0, z_at_entry=z)
                else:
                    self.open_pair(ctx, pid, a["y"], a["x"], sign=-1.0, z_at_entry=z)


# =====================================================================================
#: Predeclared same-sector fund pairs (Select Sector SPDR vs Vanguard sector ETF, different
#: index families) with first trading dates (issuer records; Vanguard sector ETFs 2004).
S108_PAIRS = {
    ("XLK", "VGT"): "2004-01-26", ("XLF", "VFH"): "2004-01-26", ("XLV", "VHT"): "2004-01-26",
    ("XLU", "VPU"): "2004-01-26", ("XLP", "VDC"): "2004-01-26", ("XLY", "VCR"): "2004-01-26",
    ("XLB", "VAW"): "2004-01-26", ("XLE", "VDE"): "2004-09-23", ("XLI", "VIS"): "2004-09-23",
}


@register
class S108(_PairBase):
    id = "S108"
    name = "Kalman-filter adaptive pairs"
    asset = "ETF"
    universe = sorted({s for p in S108_PAIRS for s in p})
    params = {"q_ratio": 0.001, "z_entry": 2.0, "z_exit": 0.5, "max_hold": 20, "beta_lo": 0.5, "beta_hi": 2.0,
              "kf_window": 504, "warmup": 60, "pairs": None}
    grid = {"q_ratio": [0.001, 0.01], "z_entry": [1.5, 2.0, 2.5]}
    requires = _pair_requires(sorted({s for p in S108_PAIRS for s in p}), note="predeclared same-sector ETF pairs")
    deviations = COMMON_PAIR_DEVIATIONS + [
        "Predeclared pairs (module S108_PAIRS): Select Sector SPDR vs Vanguard sector ETF of the same sector; "
        "traded from the later fund's first trading date (issuer records) once kf_window sessions exist",
        "Filter (indicators.kalman_hedge, intercept, Q = q_ratio*R*I, R and prior from OLS on the first 60 points) "
        "re-run each day on the trailing kf_window=504 sessions of total-return prices (bounded compute; PIT); "
        "y = SPDR, x = Vanguard",
        "Beta constraint: the prior (one-step-ahead) beta is clipped to [0.5, 2]; innovation recomputed with the "
        "clipped beta, z = e/sqrt(S_t) (S_t = filter innovation variance)",
        "z < -z_entry: long spread (buy y, sell x); z > z_entry: short spread; exit |z| < 0.5 or 20 sessions",
    ]
    spec = ("Within predeclared paired ETFs e.g. same-sector funds, estimate dynamic hedge coefficient via "
            "one-step-ahead Kalman filter; long spread when filtered innovation z<-2, short z>2. || Exit |z|<0.5 "
            "or 20d; constrain beta coefficient [0.5,2].")

    def zscore_today(self, ctx, y: str, x: str) -> float:
        p = self.params
        n = int(p["kf_window"])
        d = pd.concat([closes(ctx, y, n).rename("y"), closes(ctx, x, n).rename("x")], axis=1).dropna()
        if len(d) < n or d.index[-1] != ctx.session_date():
            return float("nan")
        k = kalman_hedge(d["y"], d["x"], q_ratio=float(p["q_ratio"]), warmup=int(p["warmup"]))
        r = k.iloc[-1]
        if not (r["innovation_var"] == r["innovation_var"] and r["innovation_var"] > 0):
            return float("nan")
        b = min(max(float(r["beta_prior"]), p["beta_lo"]), p["beta_hi"])
        e = float(d["y"].iloc[-1]) - float(r["alpha_prior"]) - b * float(d["x"].iloc[-1])
        return e / math.sqrt(float(r["innovation_var"]))

    def step(self, ctx):
        p = self.params
        sel = p["pairs"]
        for (y, x), inc in S108_PAIRS.items():
            if (sel is not None and (y, x) not in sel) or ctx.session_date() < pd.Timestamp(inc):
                continue
            pid = (y, x)
            z = self.zscore_today(ctx, y, x)
            if not z == z:
                continue
            if pid in self.open:
                if abs(z) < p["z_exit"]:
                    self.close_pair(ctx, pid, "converged")
                elif self.held(pid) >= p["max_hold"]:
                    self.close_pair(ctx, pid, "time")
            elif z < -p["z_entry"]:
                self.open_pair(ctx, pid, y, x, sign=-1.0, z_at_entry=z)
            elif z > p["z_entry"]:
                self.open_pair(ctx, pid, x, y, sign=1.0, z_at_entry=z)


# =====================================================================================
@register
class S109(_PairBase):
    id = "S109"
    name = "Sector-neutral residual equity spread"
    asset = "STK"
    universe_name = "SP500"
    params = {"resid_days": 5, "select": 0.10, "reg_window": 252, "max_pairs": 10, "beta_hedge": True}
    grid = {"resid_days": [3, 5, 10], "select": [0.05, 0.10, 0.20]}
    overlay_symbols = ("SPY",)
    requires = [Requirement("bars", "1d", note="daily bars for the PIT SP500 universe"),
                Requirement("bars", "1d", symbols=("SPY",) + SECTOR_SPDR_9, note="SPY and sector SPDR regressors"),
                Requirement("bars", "1d", symbols=("XLRE", "XLC"), critical=False, note="later sector SPDRs")]
    deviations = COMMON_PAIR_DEVIATIONS + [
        SURVIVORSHIP_NOTE,
        "Weekly: formed at the close of the last session of each week ('Friday', earlier on holiday weeks), "
        "held until the next weekly rebalance (5 sessions) - the rebalance replaces the book",
        "Residual of day d = daily total return minus fit on SPY + the stock's sector SPDR with betas fit on the "
        "252 sessions through d-1 (indicators.rolling_ols resid_oos); score = sum over the last resid_days",
        "The sector SPDR regressor is used only if it has >= reg_window returns before the residual window; "
        "otherwise (XLRE before ~2016-10, XLC before ~2019-06, no SPDR for the sector) the stock gets a "
        "market-only regression; stocks without a sector (needed for sector-neutral pairing) are excluded",
        "(most negative with most positive) -> sector-neutral pairs; pairs ranked by residual spread, at most "
        "max_pairs=10 (100% NAV gross), each 5%/5% dollar neutral",
        "Beta neutral: net market beta of the book (univariate 252d beta vs SPY) hedged with SPY (beta_hedge)",
    ]
    spec = ("Rolling 252d regression stock daily return vs SPY and corresponding fixed sector ETF; long bottom 10% "
            "of cumulative 5-day regression residuals and short top 10%. || Exit after 5d, rebalance Friday; beta "
            "and sector neutral.")

    def init(self, ctx):
        super().init(ctx)
        self.sm = SecMaster()
        self.last_weights: dict = {}

    def step(self, ctx):
        if not is_week_end(ctx):
            return
        p = self.params
        W, L = int(p["reg_window"]), int(p["resid_days"])
        syms = self.symbols(ctx)
        self.sm.refresh(ctx)
        today = ctx.session_date()
        etf_of = {s: sector_etf(self.sm.sector(s), today) for s in syms}
        etfs = sorted({e for e in etf_of.values() if e})
        px = close_panel(ctx, list(syms) + ["SPY"] + etfs, W + L + 2)
        if px.empty or "SPY" not in px or px.index[-1] != today:
            return
        r = daily_returns(px)
        usable = {e for e in etfs if e in r and int(r[e].iloc[:-L].notna().sum()) >= W}
        score, beta, sector = {}, {}, {}
        for s in syms:
            sec = self.sm.sector(s)
            if not sec or s not in r:
                continue
            e = etf_of.get(s)
            o = rolling_ols(r[s], r[["SPY"] + ([e] if e in usable else [])], W)
            res = o["resid_oos"].iloc[-L:]
            if res.notna().sum() < L:
                continue
            score[s] = float(res.sum())
            beta[s] = ols_beta(r[s].iloc[-W:], r["SPY"].iloc[-W:], int(0.9 * W))
            sector[s] = sec
        if len(score) < 2:
            return
        sc = pd.Series(score)
        q = int(round(1.0 / p["select"]))
        b = xs.quantile_bucket(sc, q)
        pairs = []
        for sec in sorted(set(sector.values())):
            m = [s for s in sc.index if sector[s] == sec]
            lo = sorted([s for s in m if b[s] == 1], key=lambda s: sc[s])
            hi = sorted([s for s in m if b[s] == q], key=lambda s: -sc[s])
            for l, h in zip(lo, hi):
                pairs.append((sc[h] - sc[l], l, h))
        pairs.sort(key=lambda t: (-t[0], t[1]))
        pairs = pairs[: int(p["max_pairs"])]
        w: dict[str, float] = {}
        for _, l, h in pairs:
            w[l] = 0.05
            w[h] = -0.05
        if p["beta_hedge"] and w:
            nb = sum(v * beta[s] for s, v in w.items() if beta.get(s) == beta.get(s))
            if abs(nb) > 1e-6:
                w["SPY"] = -nb
        self.open = {(l, h): {"long": l, "short": h, "entry_i": self.bar_i} for _, l, h in pairs}
        self.last_weights = w
        self.events.append((today, "book", "rebalance", f"{len(pairs)} pairs"))
        rebalance_to(ctx, w)


# =====================================================================================
@register
class S110(_PairBase):
    id = "S110"
    name = "ETF tracking discount convergence proxy"
    asset = "ETF"
    universe = ["SPY", "QQQ", "IWM", "DIA"]
    params = {"deviation_bp": 50, "stale_sec": 5, "exit_bp": 10}
    grid = {"deviation_bp": [20, 50, 100], "stale_sec": [1, 5, 10]}
    requires = [Requirement("inav", "1s", symbols=("SPY", "QQQ", "IWM", "DIA"), critical=True, min_fidelity=Fidelity.T0_QUOTE,
                            note="timestamped official intraday iNAV (IIV) synchronous with ETF NBBO quotes; no free "
                                 "historical source exists -> NOT_TESTABLE"),
                Requirement("quotes", None, symbols=("SPY", "QQQ", "IWM", "DIA"), critical=True,
                            min_fidelity=Fidelity.T0_QUOTE, note="ETF NBBO at the iNAV timestamps")]
    deviations = [
        "NOT_TESTABLE by design: the entry needs published intraday iNAV with timestamps simultaneous with ETF "
        "quotes; daily NAV or modelled basket values would be a fabricated proxy, so none is offered",
    ]
    spec = ("If timestamped published official intraday iNAV and ETF quote are available simultaneously, enter when "
            "mid-iNAV deviation < -0.5% and estimated basket spread lower than deviation. || Exit deviation within "
            "0.1% or day close; skip stale/illiquid NAV; no guaranteed creation/redemption access.")

    def step(self, ctx):
        # Unreachable in practice: the critical 'inav' requirement makes the run NOT_TESTABLE.
        return


# =====================================================================================
#: Same-index ETF pairs (identical published index), first trading dates (issuer records).
S111_PAIRS = {
    ("SPY", "IVV"): ("S&P 500", "2000-05-19"),
    ("QQQ", "QQQM"): ("Nasdaq-100", "2020-10-13"),
    ("IWM", "VTWO"): ("Russell 2000", "2010-09-24"),
    ("MDY", "IJH"): ("S&P MidCap 400", "2000-05-26"),
    ("IJR", "VIOO"): ("S&P SmallCap 600", "2010-09-09"),
}


@register
class S111(_PairBase):
    id = "S111"
    name = "Same-index ETF spread"
    asset = "ETF"
    universe = sorted({s for p in S111_PAIRS for s in p})
    params = {"z_lookback": 126, "z_entry": 2.0, "z_exit": 0.25, "pairs": None}
    grid = {"z_lookback": [60, 126, 252], "z_entry": [1.5, 2.0]}
    requires = (_pair_requires(("SPY", "IVV"), note="oldest same-index pair")
                + _pair_requires(("QQQ", "QQQM", "IWM", "VTWO", "MDY", "IJH", "IJR", "VIOO"), critical=False,
                                 note="other same-index pairs (used from inception)"))
    deviations = COMMON_PAIR_DEVIATIONS + [
        "Pairs (module S111_PAIRS): SPY/IVV, QQQ/QQQM, IWM/VTWO, MDY/IJH, IJR/VIOO - same published index, "
        "non-overlapping; a pair trades only from the later fund's first trading date and once z_lookback "
        "sessions of both exist (VOO omitted to keep pairs non-overlapping)",
        "Ratio = log(total-return price a / total-return price b); z = (ratio - mean)/population stdev over the "
        "trailing z_lookback sessions including today (indicators.zscore)",
        "z > z_entry: a expensive -> short a / long b; z < -z_entry: long a / short b; exit |z| <= 0.25",
        "Borrow, dividends and expense drag are carried by the engine (borrow fees, dividends in lieu) and the "
        "total-return series",
    ]
    spec = ("Pair two ETFs tracking same published index; compute 126d total-return ratio z; buy cheap short "
            "expensive at |z|>2 with dollar-neutral exposure. || Exit z within 0.25; borrow, dividends and expense "
            "drag.")

    def step(self, ctx):
        p = self.params
        n = int(p["z_lookback"])
        sel = p["pairs"]
        for (a, b), (_, inc) in S111_PAIRS.items():
            if (sel is not None and (a, b) not in sel) or ctx.session_date() < pd.Timestamp(inc):
                continue
            d = pd.concat([closes(ctx, a, n).rename("a"), closes(ctx, b, n).rename("b")], axis=1).dropna()
            if len(d) < n or d.index[-1] != ctx.session_date():
                continue
            ratio = np.log(d["a"] / d["b"])
            z = float(zscore(ratio, n).iloc[-1])
            if not z == z:
                continue
            pid = (a, b)
            if pid in self.open:
                if abs(z) <= p["z_exit"]:
                    self.close_pair(ctx, pid, "converged")
            elif z > p["z_entry"]:
                self.open_pair(ctx, pid, b, a, sign=1.0, z_at_entry=z)
            elif z < -p["z_entry"]:
                self.open_pair(ctx, pid, a, b, sign=-1.0, z_at_entry=z)


# =====================================================================================
@register
class S112(_PairBase):
    id = "S112"
    name = "PCA residual statistical arbitrage"
    asset = "STK"
    universe_name = "SP500"
    params = {"n_pcs": 5, "formation": 252, "resid_days": 5, "n_liquid": 100, "hold": 5, "max_pairs": 10,
              "beta_hedge": True, "hedge_band": 0.005}
    grid = {"n_pcs": [3, 5, 10], "formation": [126, 252], "resid_days": [3, 5, 10]}
    overlay_symbols = ("SPY",)
    requires = [Requirement("bars", "1d", note="daily bars for the PIT SP500 universe"),
                Requirement("bars", "1d", symbols=("SPY",), note="beta hedge")]
    deviations = COMMON_PAIR_DEVIATIONS + [
        SURVIVORSHIP_NOTE,
        "Top-100 liquid = PIT SP500 members ranked by 20-session mean raw dollar volume, re-ranked at each month "
        "end (and at the first bar)",
        "Residuals: indicators.pca_residuals on daily total returns, covariance PCA (standardize=False) of the "
        "`formation` sessions through t-1, first n_pcs components, refit daily, names need a complete window",
        "Score = sum of the last resid_days residuals; bottom decile long, top decile short among names not "
        "already held; matched as pairs (most negative with most positive), at most max_pairs=10 open pairs "
        "(100% NAV gross), each 5%/5% dollar neutral",
        "Exit 5 sessions after entry or when the pair's residual spread (score short - score long) crosses 0",
        "Net beta neutral: SPY overlay = -sum((position + working-order quantity) x price x 252d beta) over all "
        "stock holdings, rebalanced when it moves by more than hedge_band=0.5% NAV; leftover stock positions not "
        "in an open pair (partial fills, rejected legs) are flattened at the next close",
    ]
    spec = ("For PIT universe top 100 liquid stocks run PCA on 252d covariance through t-1; reconstruct daily "
            "returns using first 5 PCs; cumulate residual over 5d; buy bottom decile sell top. || Exit 5 trading "
            "days later or if residual z crosses 0; net beta neutral.")

    def init(self, ctx):
        super().init(ctx)
        self.cache = ReturnCache()
        self.liquid: list[str] = []
        self.betas: dict[str, float] = {}

    def rank_liquidity(self, ctx):
        dv = {}
        for s in self.symbols(ctx):
            try:
                h = ctx.history(s, 20, "1d", adjusted=False)
            except Exception:
                continue
            if h is not None and len(h) >= 20:
                dv[s] = float((h["close"] * h["volume"]).mean())
        self.liquid = list(pd.Series(dv).sort_values(ascending=False, kind="mergesort").index[: int(self.params["n_liquid"])]) if dv else []

    def step(self, ctx):
        p = self.params
        if not self.liquid or is_month_end(ctx):
            self.rank_liquidity(ctx)
        W, L = int(p["formation"]), int(p["resid_days"])
        names = sorted(set(self.liquid) | self.legs())
        self.cache.update(ctx, names + ["SPY"], W + L + 5)
        R = self.cache.panel(names, W + L)
        if R.empty or R.index[-1] != ctx.session_date() or len(R) < W + L:
            return
        res = pca_residuals(R, window=W, n_components=int(p["n_pcs"]), standardize=False)
        tail = res.iloc[-L:]
        score = tail.sum().where(tail.notna().all())
        # exits
        for pid in list(self.open):
            o = self.open[pid]
            spread = score.get(o["short"], np.nan) - score.get(o["long"], np.nan)
            if self.held(pid) >= p["hold"]:
                self.close_pair(ctx, pid, "time")
            elif spread == spread and spread <= 0:
                self.close_pair(ctx, pid, "resid_cross_0")
        # entries
        held = self.legs()
        cand = score[[s for s in self.liquid if s in score.index]].dropna()
        if len(cand) >= 10:
            b = xs.quantile_bucket(cand, 10)
            lo = sorted([s for s in cand.index if b[s] == 1 and s not in held], key=lambda s: (cand[s], s))
            hi = sorted([s for s in cand.index if b[s] == 10 and s not in held], key=lambda s: (-cand[s], s))
            room = int(p["max_pairs"]) - len(self.open)
            spy = self.cache.panel(["SPY"], W)
            for l, h in list(zip(lo, hi))[: max(room, 0)]:
                for s in (l, h):
                    rr = self.cache.panel([s], W)
                    if not rr.empty and not spy.empty:
                        self.betas[s] = ols_beta(rr[s], spy["SPY"], int(0.9 * W))
                self.open_pair(ctx, (l, h, ctx.session_date()), l, h)
        self.flatten_orphans(ctx)
        if p["beta_hedge"]:
            self.hedge(ctx)

    @staticmethod
    def exposure(ctx, sym: str) -> float:
        """Position plus the remaining quantity of working orders (what the book will be)."""
        return ctx.position(sym) + sum(o.quantity - o.filled_qty for o in ctx.open_orders(sym))

    def hedge(self, ctx):
        nb = 0.0
        syms = (self.legs() | {x for x, q in ctx.positions().items() if q}) - set(self.overlay_symbols)
        for s in sorted(syms):
            q = self.exposure(ctx, s)
            if not q:
                continue
            b = self.betas.get(s, float("nan"))
            px = ctx.mark(s)
            if b == b and px == px:
                nb += q * px * b
        spx = ctx.mark("SPY")
        if not (spx == spx and spx > 0):
            return
        tgt = -math.trunc(nb / spx)
        cur = self.exposure(ctx, "SPY")
        if abs(tgt - cur) * spx > self.params["hedge_band"] * ctx.nav:
            ctx.cancel_all("SPY")
            d = tgt - ctx.position("SPY")
            if d:
                ctx.order("SPY", d, tag="beta_hedge")


# =====================================================================================
@register
class S113(_PairBase):
    id = "S113"
    name = "Beta-neutral relative strength pairs"
    asset = "STK"
    bar_freq = "1mo"
    universe_name = "SP500"
    params = {"mom_diff": 0.20, "beta_window": 126, "mom_months": 3, "max_pairs": 10, "pair_gross": 0.10,
              "industry_map": "SIC"}
    grid = {"mom_diff": [0.10, 0.20, 0.30], "beta_window": [126, 252]}
    requires = [Requirement("bars", "1d", note="daily bars for the PIT SP500 universe"),
                Requirement("bars", "1d", symbols=("SPY",), note="beta regressor")]
    deviations = COMMON_PAIR_DEVIATIONS[:1] + [
        SURVIVORSHIP_NOTE,
        "Matching industry pairs = same 3-digit SIC industry group among PIT SP500 members (baseline; latest SEC "
        "SIC, T2) or same GICS sub-industry with industry_map='GICS' (gics_sub_industry_current, current members "
        "only, T2); flagged once with coverage; NotTestableError if the column is missing or coverage < 50%",
        "3m momentum = total return over the last 3 calendar month-ends P(M0)/P(M3)-1; pairs with |difference| "
        "> mom_diff qualify, ranked by difference, non-overlapping, at most max_pairs=10 (sheet gives no count)",
        "'beta-adjusted' (strategy text overrides the dollar-neutral PAIRS default): betas = OLS of daily returns "
        "on SPY over beta_window sessions (floored at 0.1); leg notionals N_long = 10%*b_short/(b_long+b_short), "
        "N_short = 10%*b_long/(b_long+b_short) -> pair beta neutral, pair gross 10% NAV",
        "Close after one month = the next month-end rebalance replaces the whole book (fills next open)",
    ]
    spec = ("For matching industry pairs regress 126d beta; if 3m momentum difference >20pp, long winner/short "
            "loser beta-adjusted (distinct from reversion). || Close after one month; monthly re-sort.")

    def init(self, ctx):
        super().init(ctx)
        self.sm = SecMaster()
        self.last_weights: dict = {}

    def step(self, ctx):
        if not is_month_end(ctx):
            return
        p = self.params
        from .factors import month_end_sessions, n_sessions_between
        cal = ctx.calendar()
        M = int(p["mom_months"])
        me = month_end_sessions(cal, ctx.session_date(), M)
        Bw = int(p["beta_window"])
        n = max(n_sessions_between(cal, me[-1], ctx.session_date()) + 1, Bw + 1)
        syms = self.symbols(ctx)
        self.sm.refresh(ctx, uses_sector=False)
        self.sm.require_industry(ctx, syms, p["industry_map"], "S113")
        px = close_panel(ctx, list(syms) + ["SPY"], n)
        if px.empty or "SPY" not in px or px.index[-1] != ctx.session_date() or me[-1] not in px.index:
            return
        mom = px.loc[me[0]] / px.loc[me[-1]] - 1.0
        r = daily_returns(px).iloc[-Bw:]
        ind: dict[str, list[str]] = {}
        for s in syms:
            if s in mom.index and mom[s] == mom[s]:
                k = self.sm.industry(s, p["industry_map"])
                if k:
                    ind.setdefault(k, []).append(s)
        cands = []
        for names in ind.values():
            for a, b in combinations(sorted(names), 2):
                d = mom[a] - mom[b]
                if abs(d) > p["mom_diff"]:
                    w_, l_ = (a, b) if d > 0 else (b, a)
                    cands.append((-abs(d), w_, l_))
        cands.sort()
        used, pairs = set(), []
        for nd, w_, l_ in cands:
            if len(pairs) >= int(p["max_pairs"]):
                break
            if w_ in used or l_ in used:
                continue
            bw = ols_beta(r[w_], r["SPY"], int(0.9 * Bw))
            bl = ols_beta(r[l_], r["SPY"], int(0.9 * Bw))
            if not (bw == bw and bl == bl):
                continue
            pairs.append((w_, l_, max(bw, 0.1), max(bl, 0.1)))
            used.update((w_, l_))
        g = float(p["pair_gross"])
        w: dict[str, float] = {}
        for w_, l_, bw, bl in pairs:
            w[w_] = g * bl / (bw + bl)
            w[l_] = -g * bw / (bw + bl)
        self.open = {(a, b): {"long": a, "short": b, "entry_i": self.bar_i} for a, b, _, _ in pairs}
        self.last_weights = w
        self.events.append((ctx.session_date(), "book", "rebalance", f"{len(pairs)} pairs"))
        rebalance_to(ctx, w)


# =====================================================================================
@register
class S114(_PairBase):
    id = "S114"
    name = "ETF/sector basket lagged underreaction"
    asset = "STK/ETF"
    universe_name = "SP500"
    params = {"shock_sigma": 2.0, "hold_days": 1, "sigma_window": 60, "beta_window": 252, "select": 0.20,
              "two_sided": False, "htb_max_fee": 0.01, "max_pairs": 10}
    grid = {"shock_sigma": [1.5, 2.0, 2.5], "hold_days": [1, 3]}
    requires = [Requirement("bars", "1d", note="daily bars for the PIT SP500 universe"),
                Requirement("bars", "1d", symbols=SECTOR_SPDR_9, note="sector SPDR shocks"),
                Requirement("bars", "1d", symbols=("XLRE", "XLC"), critical=False, note="later sector SPDRs"),
                Requirement("borrow", None, critical=False, note="hard-to-borrow screen (default borrow if missing)")]
    deviations = COMMON_PAIR_DEVIATIONS[:1] + [
        SURVIVORSHIP_NOTE,
        "Shock: sector SPDR daily total return > shock_sigma x stdev of its returns over the 60 prior sessions "
        "(t-60..t-1); literal reading 'return > 2 sigma' (up-shocks only); two_sided=True also uses |r|",
        "Members = PIT SP500 stocks whose sector (SIC->GICS-like, current snapshot, T2) maps to that SPDR; a SPDR "
        "is only traded once it has beta_window=252 sessions of returns (XLRE from ~2016-10, XLC from ~2019-06; "
        "before that rolling_ols has no full window); beta-residual return of the shock day = "
        "r_i - (alpha + beta r_ETF) with alpha/beta fit on the 252 sessions through t-1 (rolling_ols resid_oos)",
        "Long the bottom `select`=20% residuals (underperformers), short the top 20% (overperformers), matched as "
        "pairs (most under with most over), at most max_pairs=10 open pairs, each 5%/5% dollar neutral",
        "Hard-to-borrow screen: short leg only if portal.borrow says shortable and fee <= 1% p.a.",
        "'Exit next close or 3d' = MOC exit at the close of the hold_days-th session after the signal; for "
        "hold_days=1 the MOC exit must be submitted together with the entry at the signal close (the 15:50 MOC "
        "cutoff of the next session precedes any daily-bar callback; a bracket time_exit child fills at the open "
        "AFTER the next close in daily mode, which would not be 'next close'). To keep that MOC equal to the "
        "position actually acquired, the pair is skipped unless each leg is <= participation_cap (10%) x the "
        "minimum daily volume of the last 20 sessions (the entry then cannot be volume-capped)",
        "Positions left by unfilled/partial orders are flattened at the next close (orphan cleanup)",
    ]
    spec = ("At day close a sector ETF daily return>2σ, rank member stock beta-residual return; long member "
            "underperformers in that sector, short overperformers to neutral. || Exit next close or 3d; costs, "
            "hard-to-borrow screened.")

    def init(self, ctx):
        super().init(ctx)
        self.sm = SecMaster()
        self.cache = ReturnCache()

    def step(self, ctx):
        p = self.params
        today = ctx.session_date()
        H = int(p["hold_days"])
        # exits: MOC at the close of session entry+H -> submitted at the close of entry+H-1
        for pid in list(self.open):
            if H > 1 and self.held(pid) >= H - 1:
                self.close_pair(ctx, pid, "time_moc", OrderType.MOC)
            elif H == 1 and self.held(pid) >= 1:
                self.open.pop(pid)        # MOC exit already submitted with the entry
        self.flatten_orphans(ctx)
        etfs = [e for sec, (e, inc) in SECTOR_ETFS.items() if today >= pd.Timestamp(inc)]
        Sw = int(p["sigma_window"])
        self.cache.update(ctx, etfs, Sw + 5)
        shocked = []
        for e in etfs:
            r = self.cache.r.get(e)
            if r is None or len(r) < Sw + 1 or r.index[-1] != today:
                continue
            sd = float(r.iloc[-Sw - 1:-1].std(ddof=1))
            x = float(r.iloc[-1])
            if sd > 0 and (x > p["shock_sigma"] * sd or (p["two_sided"] and abs(x) > p["shock_sigma"] * sd)):
                shocked.append(e)
        if not shocked:
            return
        self.sm.refresh(ctx)
        syms = self.symbols(ctx)
        etf_of = {s: sector_etf(self.sm.sector(s), today) for s in syms}
        Bw = int(p["beta_window"])
        for e in shocked:
            members = [s for s in syms if etf_of.get(s) == e and s not in self.legs()]
            if len(members) < 2:
                continue
            px = close_panel(ctx, members + [e], Bw + 2)
            if px.empty or e not in px or px.index[-1] != today:
                continue
            r = daily_returns(px)
            res = {}
            for s in members:
                if s not in r:
                    continue
                v = float(rolling_ols(r[s], r[e], Bw)["resid_oos"].iloc[-1])
                if v == v:
                    res[s] = v
            if len(res) < 2:
                continue
            sc = pd.Series(res)
            q = int(round(1.0 / p["select"]))
            b = xs.quantile_bucket(sc, q)
            lo = sorted([s for s in sc.index if b[s] == 1], key=lambda s: (sc[s], s))
            hi = [s for s in sorted(sc.index, key=lambda s: (-sc[s], s)) if b[s] == q and self._borrowable(ctx, s)]
            for l, h in zip(lo, hi):
                if len(self.open) >= int(p["max_pairs"]):
                    return
                pid = (l, h, today)
                if H == 1 and not self._fully_fillable(ctx, l, h):
                    ctx.log("S114 pair skipped: entry could be volume-capped", long=l, short=h)
                    continue
                if self.open_pair(ctx, pid, l, h, etf=e) and H == 1:
                    o = self.open[pid]
                    ctx.order(l, -o["qty_long"], OrderType.MOC, tag="exit:time_moc")
                    ctx.order(h, -o["qty_short"], OrderType.MOC, tag="exit:time_moc")

    def _fully_fillable(self, ctx, l: str, h: str, cap: float = 0.10) -> bool:
        pl, ps = float(ctx.mark(l)), float(ctx.mark(h))
        if not (pl == pl and ps == ps and pl > 0 and ps > 0):
            return False
        ql, qs = size_pairs(ctx.nav, pl, ps)
        for sym, q in ((l, ql), (h, qs)):
            v = ctx.history(sym, 20, "1d", adjusted=False)
            if v is None or len(v) < 20 or not abs(q) <= cap * float(v["volume"].min()):
                return False
        return True

    def _borrowable(self, ctx, s) -> bool:
        try:
            ok, fee, _ = ctx.portal().borrow(s)
        except Exception:
            return False
        return bool(ok) and (not fee == fee or fee <= self.params["htb_max_fee"])


# =====================================================================================
#: Predeclared ETF spread pairs (style / sector rotation spreads), non-overlapping; all
#: listed before 2004 (first trading dates: XLK/XLU/XLY/XLP/XLF/XLV/XLE/XLB 1998-12-22,
#: IWM 2000-05-26, QQQ 1999-03-10, DIA 1998-01-20, EFA 2001-08-17, EEM 2003-04-14,
#: TLT/IEF 2002-07-30).
S115_PAIRS = [("XLK", "XLU"), ("XLY", "XLP"), ("XLF", "XLV"), ("XLE", "XLB"), ("IWM", "SPY"), ("QQQ", "DIA"),
              ("EEM", "EFA"), ("TLT", "IEF")]


@register
class S115(_PairBase):
    id = "S115"
    name = "Spread momentum after pairs breakout"
    asset = "ETF"
    universe = sorted({s for p in S115_PAIRS for s in p})
    params = {"breakout": 60, "exit_ma": 20, "slope_n": 20, "max_hold": 30, "pairs": None}
    grid = {"breakout": [40, 60, 90], "exit_ma": [10, 20]}
    requires = _pair_requires(sorted({s for p in S115_PAIRS for s in p}), note="predeclared ETF pairs")
    deviations = COMMON_PAIR_DEVIATIONS + [
        "Pairs (module S115_PAIRS) predeclared, non-overlapping style/sector spreads",
        "Spread s = log(TR price a) - log(TR price b) (dollar-neutral spread). The standardized spread "
        "(s - mu)/sigma with parameters fixed at t is an increasing affine map of s, so 'exceeds prior 60d "
        "extreme' is evaluated on s: s_t > max(s_{t-60..t-1}) (up) or < min (down)",
        "20d spread slope = OLS slope of s over the last 20 sessions (indicators.linreg_slope), same sign as the "
        "breakout; up-break -> long a / short b; down-break -> short a / long b",
        "Exit when s crosses its exit_ma-session SMA against the position, or after 30 sessions",
    ]
    spec = ("If standardized spread of two ETFs exceeds prior 60d extreme and 20d spread slope same direction, trade "
            "with spread direction; not a reversion trade. || Exit spread crosses 20d mean or 30d.")

    def step(self, ctx):
        p = self.params
        N, M = int(p["breakout"]), int(p["exit_ma"])
        n = max(N, M, int(p["slope_n"])) + 1
        for a, b in S115_PAIRS:
            if p["pairs"] is not None and (a, b) not in p["pairs"]:
                continue
            d = pd.concat([closes(ctx, a, n).rename("a"), closes(ctx, b, n).rename("b")], axis=1).dropna()
            if len(d) < n or d.index[-1] != ctx.session_date():
                continue
            s = np.log(d["a"]) - np.log(d["b"])
            st = float(s.iloc[-1])
            pid = (a, b)
            if pid in self.open:
                ma = float(sma(s, M).iloc[-1])
                sg = self.open[pid]["sign"]
                if (sg > 0 and st < ma) or (sg < 0 and st > ma):
                    self.close_pair(ctx, pid, "ma_cross")
                elif self.held(pid) >= p["max_hold"]:
                    self.close_pair(ctx, pid, "time")
                continue
            prior = s.iloc[-N - 1:-1]
            slope = float(linreg_slope(s, int(p["slope_n"])).iloc[-1])
            if st > prior.max() and slope > 0:
                self.open_pair(ctx, pid, a, b, sign=1.0)
            elif st < prior.min() and slope < 0:
                self.open_pair(ctx, pid, b, a, sign=-1.0)


# =====================================================================================
@register
class S123(_PairBase):
    id = "S123"
    name = "Equal weight vs cap weight spread"
    asset = "ETF"
    bar_freq = "1mo"
    universe = ["RSP", "SPY"]
    params = {"threshold": 0.02, "exit": 0.005, "lookback": 63}
    grid = {"threshold": [0.01, 0.02, 0.03], "exit": [0.0, 0.005, 0.01]}
    requires = _pair_requires(("RSP", "SPY"), note="RSP from 2003-04-24")
    deviations = COMMON_PAIR_DEVIATIONS[:1] + [
        "Evaluated only at month-end closes ('rebalance monthly'); 63d excess momentum = total return RSP - total "
        "return SPY over the last 63 sessions",
        "Hysteresis: flat -> enter long RSP/short SPY if excess > threshold, short RSP/long SPY if < -threshold; "
        "long -> exit when excess < +exit, reverse if also < -threshold; short symmetric",
        "Held positions are re-sized to 5%/5% of NAV at each month end (monthly rebalance)",
    ]
    spec = ("Compute 63d excess momentum RSP vs SPY; hold RSP and short equal-notional SPY if excess >2%, reverse if "
            "<-2%; no position in band. || Exit/reverse when signal crosses +/-0.5%; rebalance monthly.")

    def init(self, ctx):
        super().init(ctx)
        self.state_ = 0
        self.last_excess = float("nan")

    def step(self, ctx):
        if not is_month_end(ctx):
            return
        p = self.params
        n = int(p["lookback"]) + 1
        d = pd.concat([closes(ctx, "RSP", n).rename("RSP"), closes(ctx, "SPY", n).rename("SPY")], axis=1).dropna()
        if len(d) < n or d.index[-1] != ctx.session_date():
            return
        ex = float((d["RSP"].iloc[-1] / d["RSP"].iloc[0] - 1) - (d["SPY"].iloc[-1] / d["SPY"].iloc[0] - 1))
        self.last_excess = ex
        st = self.state_
        new = st
        if st == 0:
            new = 1 if ex > p["threshold"] else (-1 if ex < -p["threshold"] else 0)
        elif st == 1 and ex < p["exit"]:
            new = -1 if ex < -p["threshold"] else 0
        elif st == -1 and ex > -p["exit"]:
            new = 1 if ex > p["threshold"] else 0
        self.state_ = new
        w = {} if new == 0 else {"RSP": 0.05 * new, "SPY": -0.05 * new}
        self.events.append((ctx.session_date(), "RSP/SPY", "state", f"{st}->{new} excess={ex:.4f}"))
        rebalance_to(ctx, w, tag="pair_rebalance")
