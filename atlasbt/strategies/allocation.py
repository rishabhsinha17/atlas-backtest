"""ETF portfolio / allocation strategies: S089, S090, S101-S105, S117-S122, S124, S125.

Module-wide conventions
-----------------------
* PORTFOLIO sizing (atlasbt.risk.sizing.portfolio_weights / weights_to_shares /
  rebalance_deltas): sum(|w|) <= 100% unless the sheet states leverage (S104: 1.5x on
  Reg-T margin, no leveraged product). Decisions are taken at a session close; the
  rebalance orders are market orders that fill at the NEXT session's open. Share
  counts use the decision close and NAV (whole shares), so the realised weights
  drift slightly with the overnight gap.
* T-bill proxy (STRATEGY_GUIDE TBILL_PROXY): BIL from its listing (2007-05-30), else SHV
  (2007-01-11), else engine cash (which earns the financing rate). Explicit "T-bills"
  allocations and the residual (1 - sum w) go to the proxy, except where the sheet
  says "cash" for the residual (S090, S104, S119) - those stay in engine cash.
* Total return everywhere: split+dividend adjusted daily closes as of the decision
  time (portal adjusted=True; only actions with ex_date <= now applied).
* "k-month return" at a month end = close today / close at the last NYSE session of
  the month k months earlier - 1 (calendar months). "Excess" = minus the T-bill return
  over the same dates, compounded month by month from portal.risk_free_rate known at
  the start of each month (actual/360); tenor '1M' where the sheet says 1m T-bill
  (falls back to '3M' if the portal has no 1M series), else '3M'.
* Frozen universes are defined below with first-trading dates; an ETF is eligible
  only after its listing date AND when the portal serves enough history for the
  lookback ending today (no back-filled or proxy histories).
* Month end = last NYSE session of the month (decision at its close).
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from ..core.types import Fidelity
from ..engine.strategy import Requirement, Strategy
from ..indicators import realized_vol, rsi, sma
from ..risk import sizing as SZ
from .registry import register

# first trading dates (approximate to the day; listing guard only, history must also exist)
INCEPTION = {
    "SPY": "1993-01-29", "QQQ": "1999-03-10", "IWM": "2000-05-26", "DIA": "1998-01-20",
    "XLB": "1998-12-22", "XLE": "1998-12-22", "XLF": "1998-12-22", "XLI": "1998-12-22", "XLK": "1998-12-22",
    "XLP": "1998-12-22", "XLU": "1998-12-22", "XLV": "1998-12-22", "XLY": "1998-12-22",
    "XLRE": "2015-10-08", "XLC": "2018-06-19",
    "SHY": "2002-07-26", "IEF": "2002-07-26", "TLT": "2002-07-26", "AGG": "2003-09-29",
    "EFA": "2001-08-17", "EEM": "2003-04-14", "GLD": "2004-11-18", "VNQ": "2004-09-29",
    "BIL": "2007-05-30", "SHV": "2007-01-11",
    "SSO": "2006-06-21", "QLD": "2006-06-21", "UPRO": "2009-06-25", "TQQQ": "2010-02-11",
    "SH": "2006-06-21", "SDS": "2006-07-13", "PSQ": "2006-06-21",
}
SECTOR_SPDR = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]
ETF_CORE = ["SPY", "QQQ", "IWM", "DIA"]
SWING_ETF = ETF_CORE + SECTOR_SPDR + ["TLT", "GLD", "EFA", "EEM"]
TBILL_CHAIN = ("BIL", "SHV")
#: fraction of NAV left in cash at every rebalance (1%) so that next-open gaps and fees fit inside the gross limit
#: (the engine's Reg-T pre-trade check gives pending sells no credit, so a 100% -> 100% rotation needs it)
CASH_BUFFER = 0.01
DEFENSIVE_4 = ["SPY", "IEF", "GLD", "SHY"]                  # S101: equity, int. T-note, gold, short Treasury
CROSS_ASSET = ["SPY", "EFA", "BIL", "TLT", "GLD", "VNQ"]    # S122
TREASURY_DURATION = ["SHY", "IEF", "TLT"]                   # S124 (short -> long duration)
LEVERAGED = {"SSO": ("SPY", 2), "UPRO": ("SPY", 3), "QLD": ("QQQ", 2), "TQQQ": ("QQQ", 3)}

COMMON = (
    "Decision at the session close, market orders fill at the next session open (PORTFOLIO next-session "
    "rebalancing); whole shares from the decision close.",
    "T-bill proxy BIL (2007-05-30+) else SHV (2007-01-11+) else engine cash.",
    "Total return = adjusted closes as of now; k-month windows aligned to calendar month ends.",
    "Distributions are part of the test (total return / engine dividend bookings): ETF strategies that rank "
    "on total-return series (S089, S101, S102, S105, S117, S121, S122, S124) declare "
    "Requirement('actions') on their universe; symbols without any corporate-action data are flagged "
    "T2_PROXY at run start ('distributions unavailable: price return used'). The requirement is "
    "non-critical: making it critical conflicts with red-team tests test_s089_* / "
    "test_s105_initial_buy (portals without dividend data must still run).",
    "An ETF is eligible only when its whole lookback window starts on/after its listing date.",
    "Targets are scaled to (1 - 1%) of the gross limit at each rebalance (cash buffer for next-open gaps and "
    "fees; the engine's pre-trade margin check gives pending sells no credit).",
)


# ======================================================================= helpers
def listed(sym: str, session) -> bool:
    d = INCEPTION.get(sym)
    return d is None or pd.Timestamp(session) >= pd.Timestamp(d)


def _sessions_of(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(idx.tz_convert("America/New_York").tz_localize(None).normalize())


def daily_closes(ctx, sym: str, n: int, session) -> Optional[pd.Series]:
    """Adjusted daily closes (last ``n``) indexed by session date; None unless the
    series ends today, has ``n`` rows and its first row is on/after the listing date."""
    if not listed(sym, session):
        return None
    try:
        df = ctx.history(sym, n, "1d", adjusted=True)
    except Exception:
        return None
    if df is None or len(df) < n:
        return None
    s = pd.Series(df["close"].to_numpy(float), index=_sessions_of(df.index))
    if s.index[-1] != pd.Timestamp(session) or s.isna().any():
        return None
    if not listed(sym, s.index[0]):          # window must not start before the listing (no back-filled history)
        return None
    return s


def month_end_ref(cal, session, k: int) -> pd.Timestamp:
    """Last NYSE session of the month k calendar months before ``session``'s month."""
    first = pd.Timestamp(session).replace(day=1)
    target_month_start = first - pd.DateOffset(months=k)
    nxt = target_month_start + pd.DateOffset(months=1)
    return cal.prev_session(nxt)


def period_return(ctx, sym: str, session, months: int) -> float:
    cal = ctx.calendar()
    ref = month_end_ref(cal, session, months)
    n = cal.session_index(session) - cal.session_index(ref) + 1
    s = daily_closes(ctx, sym, n, session)
    if s is None or s.index[0] != ref or not listed(sym, ref):
        return float("nan")
    return float(s.iloc[-1] / s.iloc[0] - 1.0)


def _flag_rf(ctx, tenor: str, used: str) -> None:
    flags = getattr(ctx, "state", {}).setdefault("_rf_flags", set())
    if tenor not in flags:
        flags.add(tenor)
        ctx.flag_fidelity(Fidelity.T2_PROXY, f"{tenor} T-bill rate unavailable at some dates; {used} used")


def _rf(ctx, as_of, tenor: str) -> float:
    """Annual simple T-bill rate known at ``as_of``. Uses portal.risk_free_rate_info (rate, series_used,
    fidelity, ...) when the portal exposes it, so substituted tenors are flagged T2; otherwise
    risk_free_rate(tenor) with an explicit 3M fallback (flagged), and a 1M value identical to the 3M value
    is treated as a silent substitution (flagged)."""
    p = ctx.portal()
    info = getattr(p, "risk_free_rate_info", None)
    if info is not None:
        try:
            r, used, fid = info(as_of, tenor)[:3]
            r = float(r)
            if r == r:
                if int(fid) >= Fidelity.T2_PROXY:
                    _flag_rf(ctx, tenor, str(used))
                return r
        except Exception:
            pass
    tries = (tenor, "3M") if tenor != "3M" else ("3M",)
    for t in tries:
        try:
            r = float(p.risk_free_rate(as_of, t))
        except Exception:
            r = float("nan")
        if r == r:
            if t != tenor:
                _flag_rf(ctx, tenor, t)
            elif tenor != "3M":
                try:
                    r3 = float(p.risk_free_rate(as_of, "3M"))
                except Exception:
                    r3 = float("nan")
                if r3 == r:
                    _flag_rf(ctx, tenor, "3M (identical value: probable silent fallback)")
            return r
    return float("nan")


def tbill_return(ctx, start_session, end_session, tenor: str = "3M") -> float:
    """Compounded T-bill return from the close of start_session to the close of
    end_session; each month segment earns the rate known at its start (actual/360)."""
    cal = ctx.calendar()
    a, b = pd.Timestamp(start_session), pd.Timestamp(end_session)
    if b <= a:
        return 0.0
    pts = [a]
    cur = a
    while True:
        m = month_end_ref(cal, cur, 0)
        if m <= cur:
            m = month_end_ref(cal, cur.replace(day=1) + pd.DateOffset(months=1), 0)
        if m >= b:
            break
        pts.append(m)
        cur = m
    pts.append(b)
    g = 1.0
    for x, y in zip(pts[:-1], pts[1:]):
        r = _rf(ctx, cal.open_close(x)[1], tenor)
        if r != r:
            return float("nan")
        g *= 1.0 + r * (y - x).days / 360.0
    return g - 1.0


def excess_return(ctx, sym: str, session, months: int, tenor: str = "3M") -> float:
    r = period_return(ctx, sym, session, months)
    if r != r:
        return r
    rf = tbill_return(ctx, month_end_ref(ctx.calendar(), session, months), session, tenor)
    return r - rf


def has_today(ctx, sym: str, session) -> bool:
    return daily_closes(ctx, sym, 1, session) is not None


def tbill_symbol(ctx, session) -> Optional[str]:
    for s in TBILL_CHAIN:
        if listed(s, session) and has_today(ctx, s, session):
            return s
    return None


def week_last_session(cal, session) -> bool:
    """True if ``session`` is the last NYSE session of its (Mon-Sun) week."""
    nxt = cal.next_session(session)
    s = pd.Timestamp(session)
    return (nxt - pd.Timedelta(days=nxt.weekday())) != (s - pd.Timedelta(days=s.weekday()))


def capped_weights(raw: dict, cap: float, max_iter: int = 100) -> dict:
    """Normalise ``raw`` (>0) to sum 1, cap each at ``cap`` and redistribute the excess
    iteratively pro rata to the uncapped names; leftover (if n*cap < 1) stays unallocated."""
    tot = sum(v for v in raw.values() if v > 0)
    if tot <= 0:
        return {}
    w = {k: v / tot for k, v in raw.items() if v > 0}
    fixed: dict = {}
    for _ in range(max_iter):
        over = {k for k, v in w.items() if v > cap + 1e-12}
        if not over:
            break
        for k in over:
            fixed[k] = cap
            w.pop(k)
        rem = 1.0 - sum(fixed.values())
        t = sum(w.values())
        if t <= 0 or rem <= 0:
            w = {k: 0.0 for k in w}
            break
        w = {k: v / t * rem for k, v in w.items()}
    out = dict(fixed)
    out.update(w)
    return out


# ======================================================================= base
class _Alloc(Strategy):
    bar_freq = "1mo"
    sizing = "PORTFOLIO"
    asset = "ETF"
    residual_to_tbill = True
    allow_short = False

    def max_gross(self) -> float:
        return 1.0

    def init(self, ctx):
        self.st = {"last_target": None}
        if any(r.kind == "actions" for r in self.requires):
            missing = []
            for sym in self.universe:
                try:
                    ok = bool(ctx.portal().has(sym, "actions"))
                except Exception:
                    ok = False
                if not ok:
                    missing.append(sym)
            if missing:
                ctx.flag_fidelity(Fidelity.T2_PROXY, f"{self.id}: distributions unavailable for "
                                  f"{', '.join(missing)}: price return used instead of total return")

    def is_rebalance_day(self, ctx, session) -> bool:
        return ctx.calendar().is_last_session_of_month(session)

    def target_weights(self, ctx, session) -> Optional[dict]:  # pragma: no cover - abstract
        raise NotImplementedError

    def on_bar(self, ctx):
        d = ctx.session_date()
        if not self.is_rebalance_day(ctx, d):
            return
        w = self.target_weights(ctx, d)
        if w is not None:
            self.rebalance(ctx, w, d)

    def rebalance(self, ctx, weights: dict, session) -> dict:
        w, resid = SZ.portfolio_weights(weights, max_gross=self.max_gross(), allow_short=self.allow_short)
        w = {s: v for s, v in w.items() if abs(v) > 1e-12}
        if self.residual_to_tbill and resid > 1e-9:
            tb = tbill_symbol(ctx, session)
            if tb is not None:
                w[tb] = w.get(tb, 0.0) + resid
        g = sum(abs(v) for v in w.values())
        lim = self.max_gross() * (1.0 - CASH_BUFFER)
        if g > lim:
            w = {s: v * lim / g for s, v in w.items()}
        prices = {}
        for s in set(w) | set(ctx.positions()):
            prices[s] = ctx.mark(s)
        for s in list(w):
            if not (prices.get(s, float("nan")) > 0):
                ctx.log(f"{self.id}: no price for {s}; its weight stays in cash")
                w.pop(s)
        tgt = SZ.weights_to_shares(w, ctx.nav, prices)
        deltas = SZ.rebalance_deltas(tgt, ctx.positions())
        ctx.cancel_all()
        for s in sorted(deltas, key=lambda x: deltas[x]):        # sells first, then buys
            ctx.order(s, deltas[s], tag="rebalance")
        self.st["last_target"] = dict(w)
        return w


def _tbill_weights(ctx, session) -> dict:
    tb = tbill_symbol(ctx, session)
    return {tb: 1.0} if tb else {}


# ================================================================== S089
@register
class S089(_Alloc):
    id = "S089"
    name = "Dual momentum sector ETF rotation"
    universe = SECTOR_SPDR
    params = {"lookback": 12, "top": 1, "gate": 0.0}
    grid = {"lookback": [3, 6, 12], "top": [1, 2, 3], "gate": [0.0]}
    requires = [Requirement("bars", "1d"), Requirement("bars", "1d", symbols=TBILL_CHAIN, critical=False),
                Requirement("actions", critical=False, note="distributions for total return")]
    deviations = COMMON + (
        "Predefined sector list = the 11 Select Sector SPDRs (frozen; XLRE/XLC only after listing).",
        "Excess = k-month total return minus the compounded 1M T-bill return over the same dates; each of the "
        "top-N slots gets 1/N if its excess > gate, else that slot goes to the T-bill ETF.",
    )
    spec = ("At month end rank 12m total return minus 1m T-bill return across predefined sector ETF list; buy "
            "strongest ETF only if excess>0, else invest in short T-bill ETF (use historical total return). || "
            "Rebalance next trading day opening; hold 1m.")

    def target_weights(self, ctx, d):
        k = int(self.params["lookback"])
        ex = {s: excess_return(ctx, s, d, k, "1M") for s in self.universe}
        ex = {s: v for s, v in ex.items() if v == v}
        if not ex:
            return None
        n = int(self.params["top"])
        top = sorted(ex, key=lambda s: (-ex[s], s))[:n]
        return {s: 1.0 / n for s in top if ex[s] > float(self.params["gate"])}


# ================================================================== S090
@register
class S090(_Alloc):
    id = "S090"
    name = "Time-series momentum ETF"
    universe = SWING_ETF
    params = {"lookback": 12, "vol_n": 63, "mode": "long_cash"}
    grid = {"lookback": [3, 6, 12], "vol_n": [20, 63], "mode": ["long_cash", "long_short"]}
    requires = [Requirement("bars", "1d")]
    residual_to_tbill = False
    deviations = COMMON + (
        "ETF list = SWING_ETF (STRATEGY_GUIDE 'ETF' default; the sheet names none), frozen, listing-gated.",
        "Signal: k-month total return > compounded 3M T-bill return over the same dates -> long; otherwise "
        "cash (long_cash) or short (long_short, the sheet's separate mirror test).",
        "Volatility normalisation: w_i = signal_i x (1/sigma_i) / sum_j(1/sigma_j) over all eligible ETFs "
        "(sigma = sample stdev of the trailing vol_n daily returns), so sum|w| <= 100% and a full-long "
        "signal is the inverse-vol portfolio; non-long weight stays in engine cash ('hold cash').",
    )
    spec = ("Monthly for each ETF compute trailing 12m total return; long next month if >risk-free return for "
            "period; otherwise hold cash (or mirror short as separate test). || Rebalance next month open; "
            "volatility normalize by trailing 63d sigma.")

    def __init__(self, **p):
        super().__init__(**p)
        self.allow_short = self.params["mode"] == "long_short"

    def target_weights(self, ctx, d):
        k, vn = int(self.params["lookback"]), int(self.params["vol_n"])
        sig, inv = {}, {}
        for s in self.universe:
            ex = excess_return(ctx, s, d, k, "3M")
            c = daily_closes(ctx, s, vn + 1, d)
            if ex != ex or c is None:
                continue
            v = float(realized_vol(c, vn, annualize=None).iloc[-1])
            if not v > 0:
                continue
            inv[s] = 1.0 / v
            sig[s] = 1.0 if ex > 0 else (-1.0 if self.allow_short else 0.0)
        tot = sum(inv.values())
        if tot <= 0:
            return None
        return {s: sig[s] * inv[s] / tot for s in inv if sig[s] != 0}


# ================================================================== S101
@register
class S101(_Alloc):
    id = "S101"
    name = "ETF defensive asset rotation"
    universe = DEFENSIVE_4
    params = {"lookback": 3, "top": 1, "gate": True}
    grid = {"lookback": [1, 3, 6, 12], "top": [1, 2], "gate": [True, False]}
    requires = [Requirement("bars", "1d"), Requirement("actions", critical=False, note="distributions for total return")]
    deviations = COMMON + (
        "Frozen universe SPY (broad equity), IEF (intermediate T-note), GLD (gold), SHY (short Treasury); "
        "ranked among those listed with full lookback history (>= 2 required).",
        "Excess = k-month total return minus compounded 3M T-bill return; top-N slots 1/N each if excess > 0 "
        "(gate=True) else that slot -> T-bill proxy ('short-T bills'); gate=False always buys the top N.",
    )
    spec = ("Universe historical 1 broad equity, 1 intermediate T-note, 1 gold ETF, 1 short Treasury ETF; rank 3m "
            "excess total returns; buy top ETF if return>0 else short-T bills. || Monthly rebalance; hold 1m; no "
            "in-sample universe selection.")

    def target_weights(self, ctx, d):
        k = int(self.params["lookback"])
        ex = {s: excess_return(ctx, s, d, k, "3M") for s in self.universe}
        ex = {s: v for s, v in ex.items() if v == v}
        if len(ex) < 2:
            return None
        n = int(self.params["top"])
        top = sorted(ex, key=lambda s: (-ex[s], s))[:n]
        return {s: 1.0 / n for s in top if (ex[s] > 0 or not self.params["gate"])}


# ================================================================== S102
@register
class S102(_Alloc):
    id = "S102"
    name = "Top-N sector relative strength"
    universe = SECTOR_SPDR
    params = {"lookback": 6, "top": 3}
    grid = {"lookback": [3, 6, 12], "top": [1, 3, 5]}
    requires = [Requirement("bars", "1d"), Requirement("actions", critical=False, note="distributions for total return")]
    deviations = COMMON + (
        "Fixed universe = 11 Select Sector SPDRs (listing-gated). 'return > 3m Tbill' = k-month total return > "
        "compounded 3M T-bill return over the same k months; each slot 1/N; unfilled slots -> T-bill ETF.",
    )
    spec = ("At month end rank eligible fixed sector ETFs by 6m return, choose top 3 if each return>3m Tbill; "
            "equal capital weight. || Monthly rebalance next session; cash in T-bill ETF for unfilled slots.")

    def target_weights(self, ctx, d):
        k = int(self.params["lookback"])
        r = {s: period_return(ctx, s, d, k) for s in self.universe}
        r = {s: v for s, v in r.items() if v == v}
        if not r:
            return None
        rf = tbill_return(ctx, month_end_ref(ctx.calendar(), d, k), d, "3M")
        if rf != rf:
            return None
        n = int(self.params["top"])
        top = sorted(r, key=lambda s: (-r[s], s))[:n]
        return {s: 1.0 / n for s in top if r[s] > rf}


# ================================================================== S103
@register
class S103(_Alloc):
    id = "S103"
    name = "200d trend-filtered buy-and-hold"
    bar_freq = "1d"
    universe = ["SPY"]
    params = {"sma_n": 200, "hysteresis": 0.0}
    grid = {"sma_n": [150, 200, 250], "hysteresis": [0.0, 0.01, 0.02]}
    requires = [Requirement("bars", "1d")]
    deviations = COMMON + (
        "Treasury proxy = BIL (1-3 month T-bills) else SHV else cash. Hysteresis h: switch to SPY when "
        "close > SMA x (1+h), to T-bills when close < SMA x (1-h), otherwise keep; first allocation SPY iff "
        "close > SMA. Orders only when the state changes.",
    )
    spec = ("Hold SPY if last close>SMA200; otherwise hold 1-3m Treasury proxy; compute signal close then buy next "
            "open. || Update only after daily signal changes; execute following open.")

    def is_rebalance_day(self, ctx, session):
        return True

    def target_weights(self, ctx, d):
        n, h = int(self.params["sma_n"]), float(self.params["hysteresis"])
        c = daily_closes(ctx, "SPY", n, d)
        if c is None:
            return None
        m = float(sma(c, n).iloc[-1])
        last = c.iloc[-1]
        state = self.st.get("state")
        if state is None:
            new = "SPY" if last > m else "TB"
        elif last > m * (1 + h):
            new = "SPY"
        elif last < m * (1 - h):
            new = "TB"
        else:
            new = state
        if new == state:
            return None
        self.st["state"] = new
        return {"SPY": 1.0} if new == "SPY" else _tbill_weights(ctx, d)


# ================================================================== S104
@register
class S104(_Alloc):
    id = "S104"
    name = "Volatility targeting equity ETF"
    bar_freq = "1d"
    universe = ["SPY"]
    params = {"target_vol": 0.12, "lookback": 20, "cap": 1.5, "turnover_cap": 0.20}
    grid = {"target_vol": [0.08, 0.12, 0.15], "lookback": [20, 63], "cap": [1.0, 1.5]}
    requires = [Requirement("bars", "1d")]
    residual_to_tbill = False
    deviations = COMMON + (
        "sigma = sample stdev of the last `lookback` daily close-to-close returns x sqrt(252), computed at the "
        "close of the last session of the week; the new target fills at the next week's first open (Monday, "
        "or Tuesday after a Monday holiday).",
        "Exposure > 100% is held with Reg-T margin (debit pays rf + broker spread); no leveraged ETF.",
        "Daily turnover cap: each day's SPY trade <= 20% of NAV; an unfinished move continues at the "
        "following opens (one order per day) until the weekly target is reached or replaced.",
        "Residual is engine cash ('cash earning Tbill'): the engine credits max(rf - 0.50%, 0) on cash.",
    )
    spec = ("Compute 20d annualized realized sigma using prior closing returns; at next open hold SPY exposure "
            "min(1.5,0.12/sigma); cash earning Tbill on residual. || Rebalance weekly Monday open; limit daily "
            "turnover to 20% equity; avoid leveraged product proxy unless instrument matches.")

    def max_gross(self):
        return float(self.params["cap"])

    def on_bar(self, ctx):
        d = ctx.session_date()
        cal = ctx.calendar()
        if week_last_session(cal, d):
            n = int(self.params["lookback"])
            c = daily_closes(ctx, "SPY", n + 1, d)
            if c is not None:
                v = float(realized_vol(c, n, annualize=252).iloc[-1])
                if v > 0:
                    self.st["target"] = min(float(self.params["cap"]), float(self.params["target_vol"]) / v)
                    self.st["pending"] = True
        if not self.st.get("pending"):
            return
        px, nav = ctx.mark("SPY"), ctx.nav
        if not (px > 0 and nav > 0):
            return
        cur = ctx.position("SPY") * px / nav
        gap = self.st["target"] - cur
        step = max(-float(self.params["turnover_cap"]), min(float(self.params["turnover_cap"]), gap))
        if abs(gap) <= float(self.params["turnover_cap"]) + 1e-12:
            self.st["pending"] = False
        q = math.trunc(step * nav / px)
        ctx.cancel_all("SPY")
        if q:
            ctx.order("SPY", q, tag="rebalance")


# ================================================================== S105
@register
class S105(_Alloc):
    id = "S105"
    name = "Monthly 60/40 balanced allocation"
    universe = ["SPY", "IEF"]
    params = {"equity": 0.60, "freq": "monthly"}
    grid = {"equity": [0.40, 0.60, 0.80], "freq": ["quarterly", "monthly"]}
    requires = [Requirement("bars", "1d"), Requirement("actions", critical=False, note="distributions for total return")]
    deviations = COMMON + (
        "Equity ETF = SPY, T-note ETF = IEF (intermediate Treasuries; listed 2002-07-26): nothing is bought "
        "before both are listed. Initial purchase: market orders at the open of the first possible session "
        "(submitted at the session start, sized from the prior close). Quarterly = rebalance at the last "
        "session of Mar/Jun/Sep/Dec.",
        "Expense ratios and distributions are in the traded prices / engine dividend bookings (no extra "
        "adjustment); the whole-share remainder stays in engine cash.",
    )
    spec = ("Buy equity ETF at 60% NAV and T-note ETF at 40% at first possible session; total-return prices. || "
            "At month end rebalance to 60/40 next open; include expense ratios and distributions.")

    def _w(self):
        e = float(self.params["equity"])
        return {"SPY": e, "IEF": 1.0 - e}

    def on_session_start(self, ctx):
        if self.st.get("started"):
            return
        d = ctx.session_date()
        if not all(listed(s, d) for s in self.universe):
            return
        if not all(ctx.mark(s) > 0 for s in self.universe):
            return
        self.st["started"] = True
        w = self._w()
        prices = {s: ctx.mark(s) for s in w}
        for s, q in SZ.weights_to_shares(w, ctx.nav, prices).items():
            if q:
                ctx.order(s, q, tag="initial")

    def is_rebalance_day(self, ctx, session):
        if not self.st.get("started") or not ctx.calendar().is_last_session_of_month(session):
            return False
        return self.params["freq"] == "monthly" or pd.Timestamp(session).month in (3, 6, 9, 12)

    def target_weights(self, ctx, d):
        if not all(has_today(ctx, s, d) for s in self.universe):
            return None
        return self._w()


# ================================================================== S117
@register
class S117(_Alloc):
    id = "S117"
    name = "Sector ETF monthly winner rotation"
    universe = SECTOR_SPDR
    params = {"lookback": 1, "top": 2}
    grid = {"lookback": [1, 3, 6], "top": [1, 2, 3]}
    requires = [Requirement("bars", "1d"), Requirement("actions", critical=False, note="distributions for total return")]
    deviations = COMMON + (
        "Frozen universe = 11 Select Sector SPDRs (listing-gated). 'Prior calendar-month total return' = "
        "return from the previous month-end close to this month-end close (lookback 1); equal weight 1/N.",
    )
    spec = ("Rank frozen universe of sector ETFs by prior calendar-month total return, buy best 2 next first RTH "
            "open. || Hold next month, equal weight, monthly rebalance.")

    def target_weights(self, ctx, d):
        k = int(self.params["lookback"])
        r = {s: period_return(ctx, s, d, k) for s in self.universe}
        r = {s: v for s, v in r.items() if v == v}
        if not r:
            return None
        n = int(self.params["top"])
        top = sorted(r, key=lambda s: (-r[s], s))[:n]
        return {s: 1.0 / n for s in top}


# ================================================================== S118
@register
class S118(_Alloc):
    id = "S118"
    name = "Leveraged ETF 200SMA gate"
    bar_freq = "1d"
    universe = ["SSO", "SPY"]
    params = {"etf": "SSO", "sma_n": 200, "vol_max": 0.25, "vol_n": 20}
    grid = {"sma_n": [150, 200], "vol_max": [0.20, 0.25, 0.30]}
    requires = [Requirement("bars", "1d")]
    deviations = COMMON + (
        "Specified fund: SSO (ProShares Ultra S&P500, 2x daily reset, listed 2006-06-21) on reference SPY; "
        "UPRO/QLD/TQQQ selectable via `etf`. Only the actual fund's traded prices are used; before its "
        "listing (or without its bars) the strategy holds the T-bill proxy.",
        "Reference vol = sample stdev of the last 20 daily returns x sqrt(252). Orders only when the state "
        "changes.",
    )
    spec = ("Hold a specified historically listed 2x or 3x daily reset ETF only when its unlevered reference ETF "
            "close>SMA200 and reference 20d vol<25% annualized; otherwise T-bills. || Switch next open; no "
            "synthetic leveraged return scaling across multiple days.")

    def __init__(self, **p):
        super().__init__(**p)
        etf = self.params["etf"]
        if etf not in LEVERAGED:
            raise ValueError(f"S118: unknown leveraged ETF {etf}")
        self.universe = [etf, LEVERAGED[etf][0]]

    def is_rebalance_day(self, ctx, session):
        return True

    def target_weights(self, ctx, d):
        etf = self.params["etf"]
        ref = LEVERAGED[etf][0]
        n, vn = int(self.params["sma_n"]), int(self.params["vol_n"])
        c = daily_closes(ctx, ref, max(n, vn + 1), d)
        if c is None:
            return None
        on = bool(c.iloc[-1] > sma(c, n).iloc[-1]) and \
            float(realized_vol(c, vn, annualize=252).iloc[-1]) < float(self.params["vol_max"])
        on = on and listed(etf, d) and has_today(ctx, etf, d)
        new = "ETF" if on else "TB"
        if new == self.st.get("state"):
            return None
        self.st["state"] = new
        return {etf: 1.0} if on else _tbill_weights(ctx, d)


# ================================================================== S119
@register
class S119(_Alloc):
    id = "S119"
    name = "Inverse ETF tactical hedge"
    bar_freq = "1d"
    universe = ["SPY", "SH"]
    params = {"hedge": 0.20, "sma_n": 200, "rsi_n": 14, "rsi_max": 45.0}
    grid = {"hedge": [0.10, 0.20, 0.30], "sma_n": [150, 200]}
    requires = [Requirement("bars", "1d")]
    residual_to_tbill = False
    deviations = COMMON + (
        "Baseline portfolio = 100% SPY; hedge on (SPY close < SMA and RSI14(Wilder) < 45 at the close) -> "
        "SPY 1-h, SH (ProShares Short S&P500, -1x, listed 2006-06-21) h; hedge off when SPY close > SMA. "
        "Signals evaluated every close (trade next open on a change); weights reset to target at the last "
        "session of each week (fills the next week's first open). Before SH is listed the hedge is "
        "unavailable (100% SPY). Whole-share remainder stays in cash ('cash remainder').",
    )
    spec = ("If SPY close<SMA200 and RSI14<45, move 20% NAV to inverse -1x ETF, funded from SPY allocation; cash "
            "remainder as fixed baseline. || Close inverse position if SPY close>SMA200; rebalance weekly.")

    def on_bar(self, ctx):
        d = ctx.session_date()
        n = int(self.params["sma_n"])
        c = daily_closes(ctx, "SPY", n + 1, d)
        if c is None:
            return
        last, m = c.iloc[-1], float(sma(c, n).iloc[-1])
        r = float(rsi(c, int(self.params["rsi_n"])).iloc[-1])
        hedged = self.st.get("hedged", False)
        new = hedged
        if not hedged and last < m and r < float(self.params["rsi_max"]):
            new = True
        elif hedged and last > m:
            new = False
        if new and not (listed("SH", d) and has_today(ctx, "SH", d)):
            new = False
        changed = new != hedged or self.st.get("state") is None
        self.st["hedged"] = new
        self.st["state"] = True
        if changed or week_last_session(ctx.calendar(), d):
            h = float(self.params["hedge"])
            self.rebalance(ctx, {"SPY": 1.0 - h, "SH": h} if new else {"SPY": 1.0}, d)


# ================================================================== S120
@register
class S120(_Alloc):
    id = "S120"
    name = "Volatility-managed sector basket"
    universe = SECTOR_SPDR
    params = {"vol_n": 63, "cap": 0.30}
    grid = {"vol_n": [20, 63, 126], "cap": [0.20, 0.30, 0.40]}
    requires = [Requirement("bars", "1d")]
    deviations = COMMON + (
        "Eligible = listed sector SPDRs with vol_n + 1 adjusted closes ending today; w_i ~ 1/sigma_i (sample "
        "stdev of daily returns); weights > cap are set to cap and the excess is redistributed pro rata to "
        "the uncapped names until none exceeds the cap; if n x cap < 100% the remainder -> T-bill proxy.",
    )
    spec = ("Equal risk contributions approximated weights proportional to 1/63d sigma for fixed eligible sector "
            "ETF universe (exclude missing listing history). || Monthly rebalance; cap each weight 30%, "
            "redistribute remainder iteratively.")

    def target_weights(self, ctx, d):
        vn = int(self.params["vol_n"])
        inv = {}
        for s in self.universe:
            c = daily_closes(ctx, s, vn + 1, d)
            if c is None:
                continue
            v = float(realized_vol(c, vn, annualize=None).iloc[-1])
            if v > 0:
                inv[s] = 1.0 / v
        if not inv:
            return None
        return capped_weights(inv, float(self.params["cap"]))


# ================================================================== S121
@register
class S121(_Alloc):
    id = "S121"
    name = "Bond-equity defensive switch"
    universe = ["SPY"]
    params = {"trend": 12, "vol_max": 0.20, "vol_n": 63}
    grid = {"trend": [6, 12], "vol_max": [0.15, 0.20, 0.25]}
    requires = [Requirement("bars", "1d"), Requirement("actions", critical=False, note="distributions for total return")]
    deviations = COMMON + (
        "Excess = k-month SPY total return minus compounded 3M T-bill return; 63d vol = sample stdev of the "
        "last 63 daily returns x sqrt(252); '1-3m Treasury ETF' = BIL (else SHV, else cash).",
    )
    spec = ("At month end choose SPY if SPY 12m excess return >0 and its 63d volatility<=20%, else 1-3m Treasury "
            "ETF. || Rebalance next first-month open.")

    def target_weights(self, ctx, d):
        ex = excess_return(ctx, "SPY", d, int(self.params["trend"]), "3M")
        vn = int(self.params["vol_n"])
        c = daily_closes(ctx, "SPY", vn + 1, d)
        if ex != ex or c is None:
            return None
        v = float(realized_vol(c, vn, annualize=252).iloc[-1])
        if ex > 0 and v <= float(self.params["vol_max"]):
            return {"SPY": 1.0}
        return _tbill_weights(ctx, d)


# ================================================================== S122
@register
class S122(_Alloc):
    id = "S122"
    name = "Cross-asset relative momentum ETFs"
    universe = CROSS_ASSET
    params = {"window": 6, "top": 3}
    grid = {"window": [3, 6, 12], "top": [2, 3, 4]}
    requires = [Requirement("bars", "1d"), Requirement("actions", critical=False, note="distributions for total return")]
    deviations = COMMON + (
        "Frozen basket: SPY (US equity), EFA (intl equity), BIL (T-bills), TLT (long Treasuries), GLD (gold), "
        "VNQ (REITs); listing-gated.",
        "Sharpe score = (k-month total return - compounded 3M T-bill return) / (stdev of the window's daily "
        "returns x sqrt(#returns)); top N with score > 0 get 1/N each; unfilled slots -> T-bill proxy.",
    )
    spec = ("Frozen diversified ETF basket (US equity, intl equity, T-bills, long Treasuries, gold, REITs); choose "
            "top 3 by 6m Sharpe (excess/vol), positive score required. || Equal weight top 3 next-month open, fill "
            "cash otherwise.")

    def target_weights(self, ctx, d):
        k = int(self.params["window"])
        cal = ctx.calendar()
        ref = month_end_ref(cal, d, k)
        n = cal.session_index(d) - cal.session_index(ref) + 1
        rf = tbill_return(ctx, ref, d, "3M")
        if rf != rf:
            return None
        score = {}
        for s in self.universe:
            c = daily_closes(ctx, s, n, d)
            if c is None or c.index[0] != ref or not listed(s, ref):
                continue
            rets = c.pct_change().dropna()
            sd = float(rets.std(ddof=1))
            if not sd > 0:
                continue
            score[s] = (float(c.iloc[-1] / c.iloc[0] - 1.0) - rf) / (sd * math.sqrt(len(rets)))
        if not score:
            return None
        nt = int(self.params["top"])
        top = sorted(score, key=lambda s: (-score[s], s))[:nt]
        return {s: 1.0 / nt for s in top if score[s] > 0}


# ================================================================== S124
@register
class S124(_Alloc):
    id = "S124"
    name = "Treasury duration momentum"
    universe = TREASURY_DURATION
    params = {"lookback": 3, "max_duration": "TLT"}
    grid = {"lookback": [1, 3, 6], "max_duration": ["IEF", "TLT"]}
    requires = [Requirement("bars", "1d"), Requirement("actions", critical=False, note="distributions for total return")]
    deviations = COMMON + (
        "Frozen basket SHY (1-3y), IEF (7-10y), TLT (20+y), all listed 2002-07-26. 'max duration class' is "
        "read as the longest duration ETF admitted to the basket (IEF -> SHY/IEF only; TLT -> all three).",
        "Choose the max k-month total return if it exceeds the compounded 3M T-bill return, else T-bill proxy.",
    )
    spec = ("Frozen basket of 3 fixed-duration Treasury ETFs; choose duration ETF with max previous 3m total return "
            "when >Tbill, else short T-bills. || Monthly switch next first session.")

    def target_weights(self, ctx, d):
        k = int(self.params["lookback"])
        basket = TREASURY_DURATION[:TREASURY_DURATION.index(self.params["max_duration"]) + 1]
        r = {s: period_return(ctx, s, d, k) for s in basket}
        r = {s: v for s, v in r.items() if v == v}
        if not r:
            return None
        rf = tbill_return(ctx, month_end_ref(ctx.calendar(), d, k), d, "3M")
        if rf != rf:
            return None
        best = sorted(r, key=lambda s: (-r[s], s))[0]
        return {best: 1.0} if r[best] > rf else _tbill_weights(ctx, d)


# ================================================================== S125
@register
class S125(_Alloc):
    id = "S125"
    name = "Sector breadth risk-on filter"
    bar_freq = "1d"
    universe = ["SPY"] + SECTOR_SPDR
    params = {"threshold": 7, "sma_n": 200}
    grid = {"threshold": [5, 7, 9], "sma_n": [150, 200]}
    requires = [Requirement("bars", "1d")]
    deviations = COMMON + (
        "Evaluated at the close of the last session of each week (Friday, or Thursday before a Friday "
        "holiday); switch at the next open; orders only when the state changes.",
        "Before XLRE/XLC existed fewer than 11 sectors are eligible (listed with sma_n closes): the "
        "threshold is scaled to ceil(threshold x n_eligible / 11).",
    )
    spec = ("At day close count fixed sector ETFs above their SMA200; if count>=7 out of 11 invest in SPY, "
            "otherwise in Tbill proxy. || Next open switch; evaluate weekly Fridays to reduce churn.")

    def is_rebalance_day(self, ctx, session):
        return week_last_session(ctx.calendar(), session)

    def target_weights(self, ctx, d):
        n = int(self.params["sma_n"])
        above = elig = 0
        for s in SECTOR_SPDR:
            c = daily_closes(ctx, s, n, d)
            if c is None:
                continue
            elig += 1
            above += int(c.iloc[-1] > sma(c, n).iloc[-1])
        if elig == 0 or not has_today(ctx, "SPY", d):
            return None
        need = math.ceil(int(self.params["threshold"]) * elig / 11.0 - 1e-9)
        new = "SPY" if above >= need else "TB"
        if new == self.st.get("state"):
            return None
        self.st["state"] = new
        return {"SPY": 1.0} if new == "SPY" else _tbill_weights(ctx, d)
