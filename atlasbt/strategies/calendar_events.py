"""Overnight / calendar / event strategies S031-S045 (Trading Strategy Atlas).

Module-wide conventions (declared once, apply to every class here)
------------------------------------------------------------------
* "Action at HH:MM ET" (buy 15:50, sell 09:35, exit 15:50, buy 10:05) = a market
  order submitted at the close of the bar that ENDS at HH:MM; it fills at the open
  of the bar that starts at HH:MM (first executable quote after HH:MM when quotes
  exist). Decisions "at 15:45" use only bars that ended at or before 15:45.
* Early-close sessions (13:00 ET): every afternoon time (>= 12:00) is shifted by
  the same amount as the close (15:50 -> 12:50, 15:45 -> 12:45); morning times are
  unchanged.
* S031-S037 and S042 need intraday prices at 15:45/15:50/09:35 and therefore run on
  1-minute RTH bars by default (critical Requirement). An opt-in daily proxy
  (``proxy=True``, never the default) maps "15:50" to a market-on-close order at the
  official close and "09:35" to a market-on-open order at the official open, flagged
  ``Fidelity.T2_PROXY``. S032/S033 cannot have a daily proxy (their signal is the
  15:45 price, so a daily version would signal and fill on the same closing print,
  the failure mode the sheet warns about): ``proxy=True`` makes them NOT_TESTABLE.
* NYSE calendar knowledge (holidays, early closes, month ends) is treated as known in
  advance. Unscheduled closures (2001-09-11..14, Hurricane Sandy 2012-10-29/30, days
  of mourning) are NOT "exchange holidays" for S037 (they were not known in advance).
* Universes (STRATEGY_GUIDE §3): "ETF" swing -> SWING_ETF; "STK" -> SP500 point-in-time
  (portal universe "SP500"); "ETF/STK" (S042) -> ETF_CORE + portal "SP500_TOP50_DV".
* Sizing via ``atlasbt.risk.sizing`` only. When several entries are sized at the
  same instant, the aggregate-gross cap is applied in universe order, counting the
  notional of entries already sized in that batch.
* Event requirements use ``Requirement("events", <KIND>, symbols=("*",))``: the
  engine asks ``portal.has("*", "events", KIND)``; a portal that cannot serve the
  event kind makes the run NOT_TESTABLE (contract request: see report).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from ..core.clock import et, session_date_of, to_utc
from ..core.types import ActionType, Fidelity, OrderType, TimeInForce
from ..engine.strategy import Requirement, Strategy
from ..indicators import atr, realized_vol, relative_volume, rolling_beta, session_vwap, sma
from ..indicators.cross_section import quantile_bucket
from ..portfolio.corporate_actions import as_session_date
from ..risk import sizing as SZ
from .registry import register

# ----------------------------------------------------------------- universes
ETF_CORE = ["SPY", "QQQ", "IWM", "DIA"]
SECTOR_SPDR = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]
SWING_ETF = ETF_CORE + SECTOR_SPDR + ["TLT", "GLD", "EFA", "EEM"]

#: weekday NYSE closures that were not scheduled exchange holidays (not known in advance)
UNSCHEDULED_CLOSURES = frozenset(pd.Timestamp(d) for d in (
    "1994-04-27",                                           # Nixon day of mourning
    "2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14",  # 9/11
    "2004-06-11",                                           # Reagan
    "2007-01-02",                                           # Ford
    "2012-10-29", "2012-10-30",                             # Hurricane Sandy
    "2018-12-05",                                           # G.H.W. Bush
    "2025-01-09",                                           # Carter
))

COMMON_DEVIATIONS = (
    "Action at HH:MM = market order submitted at the bar ending HH:MM, filled at the next bar's open "
    "(the first executable price after HH:MM); module convention.",
    "Early closes: afternoon times shift with the close (15:50 -> 12:50 on 13:00 closes).",
)


# ======================================================================= helpers
def label_ns(cal, session, label: str) -> int:
    """UTC ns of ET wall-clock ``label`` on ``session``; afternoon labels shift with early closes."""
    t = et(session, label)
    if int(label[:2]) >= 12 and cal.is_early_close(session):
        _, c = cal.open_close(session)
        t = t - (et(session, "16:00") - c)
    return int(t.value)


def _is_morning(label: str) -> bool:
    return int(label[:2]) < 12


def _daily(ctx, sym: str, n: int, adjusted: bool = True) -> pd.DataFrame:
    """Last ``n`` completed daily bars (adjusted as of now)."""
    try:
        df = ctx.history(sym, n, "1d", adjusted=adjusted)
    except Exception:
        return pd.DataFrame()
    return df if df is not None else pd.DataFrame()


def _bar_sessions(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(df.index.tz_convert("America/New_York").tz_localize(None).normalize())


def _fresh(df: pd.DataFrame, cal, session) -> bool:
    """Daily history ends with the session immediately before ``session``."""
    if df is None or len(df) == 0:
        return False
    return _bar_sessions(df)[-1] == cal.prev_session(session)


def _flag_once(st, ctx, tier: Fidelity, reason: str) -> None:
    seen = st.setdefault("_flags", set())
    if reason not in seen:
        seen.add(reason)
        ctx.flag_fidelity(tier, reason)


def _events_known(ctx, kind: str, since: pd.Timestamp, until: pd.Timestamp, symbols=None) -> list:
    """Events of ``kind`` with since < known_at <= until (until <= ctx.now)."""
    try:
        evs = ctx.portal().events(kind, symbols, since, ctx.now + pd.Timedelta(days=3650))
    except Exception:
        return []
    out = [e for e in evs if since < to_utc(e.known_at) <= until]
    return sorted(out, key=lambda e: (to_utc(e.known_at), e.symbol or ""))


def _prev_session_time(ctx, which: str = "open") -> pd.Timestamp:
    """Open/close (UTC) of the session before today: start of the event window on the first callback
    of a run, so announcements made just before the run start are still acted on."""
    cal = ctx.calendar()
    o, c = cal.open_close(cal.prev_session(ctx.session_date()))
    return o if which == "open" else c


def _universe(ctx, name: str) -> set:
    try:
        return set(ctx.portal().universe(name))
    except Exception:
        return set()


def _sector_map(st: dict, ctx) -> dict:
    """symbol -> sector from the portal security master when it has a 'sector' column (cached per run)."""
    if st.get("sectors") is None:
        try:
            sm = ctx.portal().security_master()
            st["sectors"] = dict(zip(sm["symbol"], sm["sector"])) if "sector" in sm.columns else {}
        except Exception:
            st["sectors"] = {}
    return st["sectors"]


def _atr14(df: pd.DataFrame, n: int = 14) -> float:
    if len(df) < n + 1:
        return float("nan")
    a = atr(df["high"], df["low"], df["close"], n)
    return float(a.iloc[-1])


def _last_intraday_bar(ctx, sym: str, session) -> Optional[pd.Series]:
    """Latest completed base-frequency RTH bar of today (raw)."""
    try:
        df = ctx.history(sym, 1, adjusted=False)
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    if session_date_of(df.index[-1]) != pd.Timestamp(session):
        return None
    return df.iloc[-1]


def _ssr_active(ctx, sym: str, session) -> bool:
    """Reg SHO Rule 201 (from 2011-02-28): triggered when a session's low <= 90% of its prior close;
    active for the rest of that day and the whole next day. Checked at the open of ``session``."""
    if pd.Timestamp(session) < pd.Timestamp("2011-02-28"):
        return False
    d = _daily(ctx, sym, 3, adjusted=True)
    if len(d) < 2 or not _fresh(d, ctx.calendar(), session):
        return False
    return bool(d["low"].iloc[-1] <= 0.9 * d["close"].iloc[-2])


# ========================================================== timed ETF base class
class _TimedBase(Strategy):
    """Scheduler for 'buy at HH:MM / sell at HH:MM on a calendar-defined session'.

    Subclasses implement ``entry_label`` (when to buy today, or None),
    ``decision_label`` (when the signal is evaluated, default = entry time),
    ``signal`` and ``exit_plan`` (session, label). Intraday mode runs on 1m bars;
    ``proxy=True`` runs on daily bars with MOC/MOO auction orders (T2_PROXY)."""

    bar_freq = "1m"
    sizing = "SWING_NOTIONAL"
    asset = "ETF"
    universe = SWING_ETF
    proxy_ok = True
    needs_daily = True
    INTRADAY_FREQ = "1m"

    def __init__(self, **params):
        super().__init__(**params)
        self.proxy = bool(self.params.get("proxy", False))
        base_reqs = [r for r in type(self).requires if not (r.kind == "bars" and r.freq == self.INTRADAY_FREQ)]
        if self.proxy:
            self.bar_freq = "1d"
            reqs = [r for r in base_reqs if not (r.kind == "bars" and r.freq == "1d")] + [Requirement("bars", "1d")]
            if not self.proxy_ok:
                reqs.append(Requirement(
                    "daily_proxy_unsupported", symbols=("SPY",),
                    note="signal needs the 15:45 intraday price; a daily proxy would signal and fill on the same close"))
            self.requires = reqs

    # ------------------------------------------------------------ hooks
    def entry_label(self, ctx, session) -> Optional[str]:
        return None

    def decision_label(self, session, entry_label: str) -> str:
        return entry_label

    def signal(self, ctx, sym: str, session) -> bool:
        return True

    def exit_plan(self, ctx, sym: str, session) -> Optional[tuple]:
        return None

    def symbols_today(self, ctx) -> list:
        return list(self.universe)

    # ------------------------------------------------------------ engine callbacks
    def init(self, ctx):
        self.st = {"plans": {}, "cands": None, "entry_ns": None, "dec_ns": None, "entry_label": None}
        if self.proxy:
            _flag_once(self.st, ctx, Fidelity.T2_PROXY,
                       f"{self.id}: daily proxy - 15:50 fills = official close (MOC), 09:35 fills = official open (MOO)")

    def _plan(self, cal, sess, label) -> tuple:
        return (pd.Timestamp(sess), label, label_ns(cal, sess, label))

    def _flat(self, ctx, sym) -> bool:
        return ctx.position(sym) == 0 and not ctx.open_orders(sym)

    def _evaluate(self, ctx, session, entry_ns: int) -> list:
        plans = self.st["plans"]
        out = []
        for s in self.symbols_today(ctx):
            p = plans.get(s)
            if p is not None and p[2] > entry_ns:
                continue                       # still holding: no overlapping entries
            if p is None and not self._flat(ctx, s):
                continue
            try:
                ok = self.signal(ctx, s, session)
            except Exception as e:             # data problems never become trades
                ctx.log(f"{self.id} signal error {s}: {e}")
                ok = False
            if ok:
                out.append(s)
        return out

    def _buy(self, ctx, syms, session, order_type: OrderType, gross: float) -> float:
        cal = ctx.calendar()
        for s in syms:
            plan = self.exit_plan(ctx, s, session)
            if plan is None:
                continue
            px = ctx.mark(s)
            if not (px == px and px > 0):
                ctx.log(f"{self.id}: no price for {s}")
                continue
            if order_type == OrderType.MARKET and _last_intraday_bar(ctx, s, ctx.session_date()) is None:
                ctx.log(f"{self.id}: no intraday bar today for {s}; entry skipped")
                continue
            q = SZ.size_swing_notional(ctx.nav, px, 1, current_gross=gross)
            if q <= 0:
                continue
            ctx.order(s, q, type=order_type, tag="entry")
            gross += q * px
            self.st["plans"][s] = self._plan(cal, plan[0], plan[1])
        return gross

    def _exit(self, ctx, sym: str, order_type: OrderType) -> None:
        q = ctx.position(sym)
        ctx.cancel_all(sym)
        if q:
            ctx.order(sym, -q, type=order_type, tag="exit")

    # ---- intraday mode
    def on_session_start(self, ctx):
        d = ctx.session_date()
        cal = ctx.calendar()
        if self.proxy:
            self._proxy_session_start(ctx, d)
            return
        el = self.entry_label(ctx, d)
        self.st["entry_label"] = el
        self.st["cands"] = None
        if el is None:
            self.st["entry_ns"] = self.st["dec_ns"] = None
        else:
            self.st["entry_ns"] = label_ns(cal, d, el)
            self.st["dec_ns"] = label_ns(cal, d, self.decision_label(d, el))

    def on_bar(self, ctx):
        if self.proxy:
            self._proxy_close(ctx)
            return
        t = ctx.now.value
        st = self.st
        d = ctx.session_date()
        if st["dec_ns"] is not None and t == st["dec_ns"]:
            st["cands"] = self._evaluate(ctx, d, st["entry_ns"])
        entries = []
        if st["entry_ns"] is not None and t == st["entry_ns"]:
            entries = st["cands"] or []
            st["cands"] = None
        due = [s for s, p in st["plans"].items() if p[2] <= t]
        for s in due:
            if s in entries:              # exit and re-entry at the same instant: keep the position
                entries.remove(s)
                nxt = self.exit_plan(ctx, s, d)
                if nxt is not None:
                    st["plans"][s] = self._plan(ctx.calendar(), *nxt)
                    continue
            st["plans"].pop(s, None)
            self._exit(ctx, s, OrderType.MARKET)
        if entries:
            self._buy(ctx, entries, d, OrderType.MARKET, SZ.equity_gross(ctx))

    # ---- daily proxy mode
    def _proxy_session_start(self, ctx, d):
        """15:50 actions of session d -> MOC orders submitted at d's open."""
        cal = ctx.calendar()
        st = self.st
        due = [s for s, p in st["plans"].items() if p[0] == d and not _is_morning(p[1])]
        el = self.entry_label(ctx, d)
        entries = []
        if el is not None and not _is_morning(el):
            entries = self._evaluate(ctx, d, label_ns(cal, d, el))
        for s in due:
            if s in entries:
                entries.remove(s)
                nxt = self.exit_plan(ctx, s, d)
                if nxt is not None:
                    st["plans"][s] = self._plan(cal, *nxt)
                    continue
            st["plans"].pop(s, None)
            self._exit(ctx, s, OrderType.MOC)
        if entries:
            self._buy(ctx, entries, d, OrderType.MOC, SZ.equity_gross(ctx))

    def _proxy_close(self, ctx):
        """09:35 actions of the next session -> MOO orders submitted at today's close."""
        cal = ctx.calendar()
        st = self.st
        d = ctx.session_date()
        n = cal.next_session(d)
        t = ctx.now.value
        for s, p in list(st["plans"].items()):
            if p[2] < t:                         # overdue (auction missed): plain market order
                st["plans"].pop(s)
                self._exit(ctx, s, OrderType.MARKET)
        due = [s for s, p in st["plans"].items() if p[0] == n and _is_morning(p[1])]
        el = self.entry_label(ctx, n)
        entries = []
        if el is not None and _is_morning(el):
            entries = self._evaluate(ctx, n, label_ns(cal, n, el))
        for s in due:
            if s in entries:
                entries.remove(s)
                nxt = self.exit_plan(ctx, s, n)
                if nxt is not None:
                    st["plans"][s] = self._plan(cal, *nxt)
                    continue
            st["plans"].pop(s, None)
            self._exit(ctx, s, OrderType.MOO)
        if entries:
            self._buy(ctx, entries, n, OrderType.MOO, SZ.equity_gross(ctx))


_INTRADAY_REQS = [Requirement("bars", "1m", sessions=("RTH",)), Requirement("bars", "1d")]


# ================================================================== S031
@register
class S031(_TimedBase):
    id = "S031"
    name = "Overnight close-to-next-open drift"
    params = {"mom_days": 10, "nights": 1, "proxy": False}
    grid = {"mom_days": [0, 5, 10], "nights": [1, 3, 5]}
    requires = _INTRADAY_REQS
    deviations = COMMON_DEVIATIONS + (
        "Universe SWING_ETF (STRATEGY_GUIDE: 'ETF' swing); sheet does not name instruments.",
        "Momentum filter 0 = the sheet's 'all-days long unconditional' comparison.",
        "'N nights' = position held continuously and sold at 09:35 of the N-th session after entry; "
        "no new entry while held.",
        "Total return from split/dividend-adjusted daily closes as of the decision time.",
    )
    spec = ("At 15:45 ET if trailing 10 COMPLETED sessions close-to-close total return >0, buy at first "
            "executable quote after 15:50 ET; do not use the final close. || Sell at next RTH 09:35 ET; no "
            "overnight stop assumed; compare all-days long unconditional.")

    def entry_label(self, ctx, session):
        return "15:50"

    def decision_label(self, session, entry_label):
        return "15:45"

    def signal(self, ctx, sym, session):
        m = int(self.params["mom_days"])
        if m <= 0:
            return True
        d = _daily(ctx, sym, m + 1)
        if len(d) < m + 1 or not _fresh(d, ctx.calendar(), session):
            return False
        c = d["close"].to_numpy(float)
        return bool(c[-1] / c[0] - 1.0 > 0.0)

    def exit_plan(self, ctx, sym, session):
        return ctx.calendar().next_session(session, int(self.params["nights"])), "09:35"


# ================================================================== S032 / S033
class _ShockBase(_TimedBase):
    proxy_ok = False

    def _shock_return(self, ctx, sym, session) -> tuple[float, float]:
        """(same-day return at the decision time vs prior completed close, 20d prior sigma)."""
        bar = _last_intraday_bar(ctx, sym, session)
        d = _daily(ctx, sym, 21)
        if bar is None or len(d) < 21 or not _fresh(d, ctx.calendar(), session):
            return float("nan"), float("nan")
        closes = d["close"]
        sig = float(realized_vol(closes, 20, annualize=None).iloc[-1])
        return float(bar["close"]) / float(closes.iloc[-1]) - 1.0, sig

    def exit_plan(self, ctx, sym, session):
        return ctx.calendar().next_session(session, int(self.params.get("nights", 1))), "09:35"


@register
class S032(_ShockBase):
    id = "S032"
    name = "Overnight large-down-day reversal"
    params = {"shock": 2.0, "signal_time": "15:45", "proxy": False}
    grid = {"shock": [1.5, 2.0, 2.5], "signal_time": ["15:45", "15:50"]}
    requires = _INTRADAY_REQS
    deviations = COMMON_DEVIATIONS + (
        "Universe SWING_ETF (STRATEGY_GUIDE: 'ETF' swing).",
        "Same-day return = close of the last 1m bar ending at the signal time / prior completed (adjusted) "
        "daily close - 1; sigma = sample stdev of the 20 prior completed daily close-to-close returns.",
        "signal_time 15:50: signal on the bar ending 15:50 and buy at the next bar open (15:50 print).",
        "'Unless next day is holiday then next RTH': exit at 09:35 of the next NYSE session.",
        "No daily proxy: proxy=True is NOT_TESTABLE (signal needs the intraday 15:45 price).",
    )
    spec = ("If 15:45 same-day return relative to prior completed close <=-2*20d prior-completed daily realized "
            "sigma, buy using 15:50 quote. || Exit next day 09:35, unless next day is holiday then next RTH; "
            "hard risk cap via size only.")

    def entry_label(self, ctx, session):
        return "15:50"

    def decision_label(self, session, entry_label):
        return self.params["signal_time"]

    def signal(self, ctx, sym, session):
        r, sig = self._shock_return(ctx, sym, session)
        return bool(r == r and sig == sig and sig > 0 and r <= -float(self.params["shock"]) * sig)


@register
class S033(_ShockBase):
    id = "S033"
    name = "Overnight large-up-day continuation"
    params = {"shock": 1.5, "nights": 1, "sma_n": 200, "proxy": False}
    grid = {"shock": [1.0, 1.5, 2.0], "nights": [1, 3]}
    requires = _INTRADAY_REQS + [Requirement("bars", "1d", symbols=("SPY",))]
    deviations = COMMON_DEVIATIONS + (
        "Universe SWING_ETF (STRATEGY_GUIDE: 'ETF' swing); trend gate always on SPY.",
        "Baseline horizon = next 09:35 (1 night); 3 nights = held continuously, sold 09:35 of the 3rd session.",
        "SPY gate: last completed SPY daily close > SMA200 of completed adjusted closes.",
        "No daily proxy: proxy=True is NOT_TESTABLE (signal needs the intraday 15:45 price).",
    )
    spec = ("If 15:45 same-day return relative to prior close >=+1.5*20d realized daily volatility computed "
            "from prior completed days and previous SPY close>SMA200, buy 15:50. || Exit next 09:35 or after 3 "
            "overnight intervals.")

    def entry_label(self, ctx, session):
        return "15:50"

    def decision_label(self, session, entry_label):
        return "15:45"

    def _spy_gate(self, ctx, session) -> bool:
        n = int(self.params["sma_n"])
        d = _daily(ctx, "SPY", n)
        if len(d) < n or not _fresh(d, ctx.calendar(), session):
            return False
        return bool(d["close"].iloc[-1] > sma(d["close"], n).iloc[-1])

    def _evaluate(self, ctx, session, entry_ns):
        if not self._spy_gate(ctx, session):
            return []
        return super()._evaluate(ctx, session, entry_ns)

    def signal(self, ctx, sym, session):
        r, sig = self._shock_return(ctx, sym, session)
        return bool(r == r and sig == sig and sig > 0 and r >= float(self.params["shock"]) * sig)


# ================================================================== S034
@register
class S034(_TimedBase):
    id = "S034"
    name = "Turn-of-month market exposure"
    params = {"start": -1, "end": 4, "exit_time": "09:35", "proxy": False}
    grid = {"start": [-1, 0], "end": [2, 3, 4]}
    requires = _INTRADAY_REQS
    deviations = COMMON_DEVIATIONS + (
        "Universe SWING_ETF (STRATEGY_GUIDE: 'ETF' swing).",
        "Offsets relative to T0 = last NYSE session of the month: entry 15:50 on T0+start (T-1 baseline), "
        "exit 09:35 on T0+end where T0+k is the k-th session of the new month (end=4 = fourth trading day).",
        "The sheet's 'T0-close to T+3-close' comparison = start=0, end=3, exit_time='15:50' (not in the grid).",
    )
    spec = ("Buy at 15:50 on trading day immediately before the final trading day of the month (T-1). || Sell "
            "at 09:35 on fourth trading day of new month; compare a T0-close to T+3-close variant.")

    def _t0(self, cal, session):
        s = int(self.params["start"])
        t0 = cal.next_session(session, -s) if s < 0 else pd.Timestamp(session)
        return t0 if cal.is_last_session_of_month(t0) else None

    def entry_label(self, ctx, session):
        return "15:50" if self._t0(ctx.calendar(), session) is not None else None

    def exit_plan(self, ctx, sym, session):
        cal = ctx.calendar()
        t0 = self._t0(cal, session)
        if t0 is None:
            return None
        return cal.next_session(t0, int(self.params["end"])), self.params["exit_time"]


# ================================================================== S035
@register
class S035(_TimedBase):
    id = "S035"
    name = "First trading day monthly seasonality"
    params = {"horizon": 1, "proxy": False}
    grid = {"horizon": [1, 2, 3]}
    requires = _INTRADAY_REQS
    deviations = COMMON_DEVIATIONS + (
        "Universe SWING_ETF (STRATEGY_GUIDE: 'ETF' swing).",
        "'Near close' = 15:50 for both legs; horizon h = sell 15:50 on the h-th session of the new month.",
    )
    spec = ("Buy near close of final trading day of month (15:50). || Sell near close of first trading day "
            "next month at 15:50.")

    def entry_label(self, ctx, session):
        return "15:50" if ctx.calendar().is_last_session_of_month(session) else None

    def exit_plan(self, ctx, sym, session):
        return ctx.calendar().next_session(session, int(self.params["horizon"])), "15:50"


# ================================================================== S036
@register
class S036(_TimedBase):
    id = "S036"
    name = "Monday open to Friday close"
    params = {"trend_gate": True, "start": "Mon", "exit": "Fri", "sma_n": 200, "proxy": False}
    grid = {"trend_gate": [True, False], "start": ["Mon", "Tue"], "exit": ["Thu", "Fri"]}
    requires = _INTRADAY_REQS + [Requirement("bars", "1d", symbols=("SPY",))]
    deviations = COMMON_DEVIATIONS + (
        "Universe SWING_ETF (STRATEGY_GUIDE: 'ETF' swing); trend gate on SPY.",
        "'First Monday RTH' = first session of the week on/after the start weekday (Tuesday if Monday is a "
        "holiday); 'previous Friday' = last session of the previous week (also for start=Tue, literally).",
        "Exit 15:50 on the last session of the week on/before the exit weekday; weeks where that session is "
        "before the entry session are skipped.",
    )
    spec = ("At first Monday RTH 09:35 buy only if previous Friday SPY close>SMA200. || Sell Friday 15:50 or "
            "final trading day that week; hold through week.")
    _WD = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4}

    @staticmethod
    def _monday(session) -> pd.Timestamp:
        s = pd.Timestamp(session)
        return s - pd.Timedelta(days=s.weekday())

    def _exit_session(self, cal, session):
        mon = self._monday(session)
        ss = cal.sessions(mon, mon + pd.Timedelta(days=self._WD[self.params["exit"]]))
        return ss[-1] if len(ss) else None

    def entry_label(self, ctx, session):
        cal = ctx.calendar()
        s = pd.Timestamp(session)
        wd = self._WD[self.params["start"]]
        if s.weekday() < wd:
            return None
        prev = cal.prev_session(s)
        if self._monday(prev) == self._monday(s) and prev.weekday() >= wd:
            return None                      # not the first eligible session of the week
        ex = self._exit_session(cal, s)
        return "09:35" if ex is not None and ex >= s else None

    def _evaluate(self, ctx, session, entry_ns):
        if self.params["trend_gate"]:
            n = int(self.params["sma_n"])
            mon = self._monday(session)
            d = _daily(ctx, "SPY", n + 5)
            if len(d):
                d = d[_bar_sessions(d) < mon]
            if len(d) < n or _bar_sessions(d)[-1] != ctx.calendar().prev_session(mon):
                return []                    # previous week's last close unavailable
            if not d["close"].iloc[-1] > sma(d["close"], n).iloc[-1]:
                return []
        return super()._evaluate(ctx, session, entry_ns)

    def exit_plan(self, ctx, sym, session):
        ex = self._exit_session(ctx.calendar(), session)
        return (ex, "15:50") if ex is not None else None


# ================================================================== S037
def pre_holiday_sessions(cal, start, end) -> set:
    """Sessions immediately preceding a scheduled full-day exchange holiday (weekday closure)."""
    hol = [h for h in cal.holidays(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=10))
           if h not in UNSCHEDULED_CLOSURES]
    out = set()
    for h in hol:
        try:
            out.add(cal.prev_session(h))
        except Exception:
            pass
    return out


@register
class S037(_TimedBase):
    id = "S037"
    name = "Preholiday drift"
    params = {"days_before": 2, "exit": "before", "proxy": False}
    grid = {"days_before": [1, 2, 3], "exit": ["before", "after"]}
    requires = _INTRADAY_REQS
    deviations = COMMON_DEVIATIONS + (
        "Universe SWING_ETF (STRATEGY_GUIDE: 'ETF' swing).",
        "Holiday = weekday NYSE full-day closure known in advance (unscheduled closures excluded); P1 = the "
        "session before it; entry 15:50 on the session (days_before-1) sessions before P1.",
        "exit='before' sells 15:50 on P1; exit='after' sells 15:50 on the first session after the holiday. "
        "days_before=1 with exit='before' is degenerate (entry = exit) and never trades.",
    )
    spec = ("Buy 15:50 on trading day two business sessions before a full exchange holiday. || Sell 15:50 on the "
            "day immediately preceding holiday.")

    def _p1(self, cal, session):
        k = int(self.params["days_before"])
        p1 = cal.next_session(session, k - 1) if k > 1 else pd.Timestamp(session)
        cache = self.st.setdefault("pre_hol", {})
        yr = p1.year
        if yr not in cache:
            cache[yr] = pre_holiday_sessions(cal, f"{yr}-01-01", f"{yr}-12-31")
        return p1 if p1 in cache[yr] else None

    def entry_label(self, ctx, session):
        if int(self.params["days_before"]) == 1 and self.params["exit"] == "before":
            return None
        return "15:50" if self._p1(ctx.calendar(), session) is not None else None

    def exit_plan(self, ctx, sym, session):
        cal = ctx.calendar()
        p1 = self._p1(cal, session)
        if p1 is None:
            return None
        return (p1 if self.params["exit"] == "before" else cal.next_session(p1)), "15:50"


# ================================================================== S042
@register
class S042(_TimedBase):
    id = "S042"
    name = "Scheduled dividend ex-date gap test"
    asset = "ETF/STK"
    universe = ETF_CORE
    params = {"window": 1, "yield_split": "all", "proxy": False}
    grid = {"yield_split": ["high", "low"], "window": [1, 2]}
    requires = _INTRADAY_REQS + [Requirement("actions", symbols=("SPY",),
                                             note="point-in-time dividend declarations (known_at); SPY probes "
                                                  "that the portal serves corporate actions")]
    deviations = COMMON_DEVIATIONS + (
        "Universe ETF_CORE + portal 'SP500_TOP50_DV' (STRATEGY_GUIDE 'STK/ETF'); requirements are checked on "
        "ETF_CORE (static part); megacaps without intraday bars are skipped and logged.",
        "Only regular CASH_DIVIDEND actions whose declaration known_at <= decision time and whose ex-date is "
        "exactly `window` sessions after the entry session; special dividends excluded.",
        "Yield split: event yield = dividend / price at decision; high/low = above/at-or-below the median "
        "event yield of dividends with ex-date in the prior 365 days seen by the strategy (needs >= 8); "
        "baseline 'all' (no split) is not a grid value.",
        "Dividend entitlement, ex-date price drop and withholding are booked by the engine "
        "(CorporateActionConfig.withholding_rate, default 0); no other taxes modelled.",
    )
    spec = ("Buy at 15:50 one trading day before known ex-dividend date, with next announced cash dividend "
            "recorded point-in-time. || Sell next 09:35 ex-date; include dividend entitlement and ex-price "
            "adjustment, tax and withholding if applicable.")

    def init(self, ctx):
        super().init(ctx)
        self.st["divs"] = {}          # (sym, ex_date) -> amount
        self.st["yields"] = {}        # (sym, ex_date) -> yield at first sight
        self.st["today"] = None

    def symbols_today(self, ctx):
        syms = list(self.universe)
        for s in sorted(_universe(ctx, "SP500_TOP50_DV")):
            if s not in syms:
                syms.append(s)
        return syms

    def entry_label(self, ctx, session):
        return "15:50"

    def _divs_for(self, ctx, session) -> dict:
        if self.st["today"] != (session, ctx.now.value):
            cal = ctx.calendar()
            target = cal.next_session(session, int(self.params["window"]))
            out = {}
            try:
                acts = ctx.portal().corporate_actions(self.symbols_today(ctx), pd.Timestamp(session),
                                                      pd.Timestamp(session) + pd.Timedelta(days=21))
            except Exception:
                acts = []
            for a in acts:
                if a.type != ActionType.CASH_DIVIDEND or not (a.amount > 0):
                    continue
                exd = as_session_date(a.ex_date)
                key = (a.symbol, exd)
                if key not in self.st["yields"]:
                    px = ctx.mark(a.symbol)
                    if px == px and px > 0:
                        self.st["yields"][key] = a.amount / px
                if exd == target:
                    out[a.symbol] = (exd, float(a.amount))
            self.st["today"] = (session, ctx.now.value)
            self.st["divs_today"] = out
        return self.st["divs_today"]

    def signal(self, ctx, sym, session):
        dv = self._divs_for(ctx, session).get(sym)
        if dv is None:
            return False
        split = self.params["yield_split"]
        if split == "all":
            return True
        y = self.st["yields"].get((sym, dv[0]))
        lo = pd.Timestamp(session) - pd.Timedelta(days=365)
        hist = [v for (s, e), v in self.st["yields"].items() if lo <= e < pd.Timestamp(session)]
        if y is None or len(hist) < 8:
            return False
        med = float(np.median(hist))
        return bool(y > med) if split == "high" else bool(y <= med)

    def exit_plan(self, ctx, sym, session):
        dv = self._divs_for(ctx, session).get(sym)
        return (dv[0], "09:35") if dv is not None else None


# ================================================================== S038
@register
class S038(Strategy):
    id = "S038"
    name = "Post-earnings announcement drift SUE"
    bar_freq = "1d"
    sizing = "CROSS_SECTION"
    asset = "STK"
    universe_name = "SP500"
    params = {"pct": 0.10, "hold": 20, "sigma_window": 63, "min_pool": 20}
    grid = {"pct": [0.05, 0.10, 0.20], "hold": [5, 20, 60]}
    requires = [Requirement("bars", "1d"),
                Requirement("events", "EARNINGS", symbols=("*",), min_fidelity=Fidelity.T2_PROXY,
                            note="release timestamps + consensus EPS frozen before release (estimate_known_at)")]
    deviations = (
        "Universe SP500 point-in-time (STRATEGY_GUIDE 'STK').",
        "Releases arrive staggered, so the cross-section is the pool of releases known in the trailing "
        "`sigma_window` (63) sessions incl. the new one: sigma = stdev of raw surprises in the pool, decile "
        "breakpoints = pool quantiles of SUE (top/bottom `pct`); needs >= min_pool releases.",
        "Surprise = eps_actual - eps_estimate (per share, not price-scaled, as in the sheet); events whose "
        "estimate_known_at is missing or not strictly before known_at are ignored (no PIT consensus).",
        "Entry: market order at the first RTH open >= 15 min after known_at (BMO releases known by 09:15 trade "
        "the same open; others the next open). Release clock times are bucketed (BMO/AMC) in the data -> T2.",
        "Exit after `hold` sessions (sold at the open of entry+hold) or at the first open after the next "
        "release of the same stock if its surprise has the opposite sign; no overlapping positions.",
        "CROSS_SECTION sizing per entry: weight from cross_section_weights over the active basket incl. the "
        "new name (+50%/-50%, <=2% per name, sector caps when the security master has 'sector'); existing "
        "names are not re-weighted when the basket changes (avoids daily churn).",
    )
    spec = ("After public earnings timestamp, compute surprise=(reported quarterly EPS - consensus EPS frozen "
            "before release)/cross-sectional surprise σ; go long top decile surprise, short bottom decile; enter "
            "first RTH open >=15m after disseminated release. || Exit after 20 trading sessions or opposite next "
            "earnings release; no overlapping same stock.")

    def init(self, ctx):
        self.st = {"pool": [], "pos": {}, "last": None, "sectors": None, "_flags": set()}

    def _sectors(self, ctx):
        return _sector_map(self.st, ctx)

    @staticmethod
    def _surprise(e) -> float:
        p = e.payload or {}
        a, est = p.get("eps_actual"), p.get("eps_estimate")
        ek = p.get("estimate_known_at")
        try:
            a, est = float(a), float(est)
        except (TypeError, ValueError):
            return float("nan")
        if not (a == a and est == est) or ek is None or pd.isna(ek):
            return float("nan")
        if to_utc(ek) >= to_utc(e.known_at):
            return float("nan")
        return a - est

    def _process(self, ctx, until: pd.Timestamp, at_open: bool) -> None:
        since = self.st["last"] if self.st["last"] is not None else \
            _prev_session_time(ctx, "open") - pd.Timedelta(minutes=15)
        if until <= since:
            return
        evs = _events_known(ctx, "EARNINGS", since, until)
        self.st["last"] = until
        if not evs:
            return
        members = _universe(ctx, "SP500")
        pool = self.st["pool"]
        new = []
        for e in evs:
            s = self._surprise(e)
            if s != s:
                continue
            if int(getattr(e, "fidelity", Fidelity.T1_BAR)) >= Fidelity.T2_PROXY:
                _flag_once(self.st, ctx, Fidelity.T2_PROXY, "S038: earnings release time approximated (BMO/AMC)")
            pool.append((to_utc(e.known_at), s))
            new.append((e, s))
        cal = ctx.calendar()
        lo = cal.open_close(cal.prev_session(ctx.session_date(), int(self.params["sigma_window"])))[0]
        pool[:] = [(k, s) for k, s in pool if k > lo]
        if not new:
            return
        arr = np.array([s for _, s in pool], float)
        if len(arr) < int(self.params["min_pool"]):
            return
        sig = float(np.std(arr, ddof=1))
        if not sig > 0:
            return
        sue_pool = arr / sig
        q = float(self.params["pct"])
        hi_bp, lo_bp = np.quantile(sue_pool, 1 - q), np.quantile(sue_pool, q)
        for e, s in new:
            sym = e.symbol
            sue = s / sig
            held = self.st["pos"].get(sym)
            if held is not None:
                if np.sign(s) != 0 and np.sign(s) != held["side"]:
                    held["exit_now"] = True             # opposite next release
                continue
            if sym not in members:
                continue
            side = 1 if sue >= hi_bp else (-1 if sue <= lo_bp else 0)
            if side:
                self._enter(ctx, sym, side, at_open)

    def _enter(self, ctx, sym, side, at_open: bool):
        px = ctx.mark(sym)
        if not (px == px and px > 0):
            return
        longs = [s for s, p in self.st["pos"].items() if p["side"] > 0]
        shorts = [s for s, p in self.st["pos"].items() if p["side"] < 0]
        (longs if side > 0 else shorts).append(sym)
        w = SZ.cross_section_weights(longs, shorts, self._sectors(ctx) or None).get(sym, 0.0)
        side_gross = sum(abs(ctx.position(s)) * ctx.mark(s) for s in (longs if side > 0 else shorts) if s != sym)
        room = max(0.5 * ctx.nav - side_gross, 0.0)
        notional = min(abs(w) * ctx.nav, room)
        q = side * math.floor(notional / px)
        if q == 0:
            return
        ctx.order(sym, q, tag="entry")
        d = ctx.session_date()
        entry = d if at_open else ctx.calendar().next_session(d)     # session whose open fills the order
        self.st["pos"][sym] = {"side": side, "entry": entry, "exit_now": False}

    def _exits(self, ctx, at_close: bool):
        cal = ctx.calendar()
        d = ctx.session_date()
        for sym, p in list(self.st["pos"].items()):
            nxt = cal.next_session(d)
            time_up = at_close and nxt >= cal.next_session(p["entry"], int(self.params["hold"]))
            if p["exit_now"] or time_up:
                ctx.cancel_all(sym)
                q = ctx.position(sym)
                if q:
                    ctx.order(sym, -q, tag="time_exit" if time_up and not p["exit_now"] else "opposite_release")
                self.st["pos"].pop(sym)

    def on_session_start(self, ctx):
        o = ctx.now
        self._process(ctx, o - pd.Timedelta(minutes=15), True)
        self._exits(ctx, at_close=False)

    def on_bar(self, ctx):
        self._process(ctx, ctx.now, False)
        self._exits(ctx, at_close=True)


# ================================================================== S039 / S040
class _EarningsGapBase(Strategy):
    bar_freq = "5m"
    asset = "STK"
    universe_name = "SP500"
    requires = [Requirement("bars", "5m", sessions=("RTH",)), Requirement("bars", "1d"),
                Requirement("events", "EARNINGS", symbols=("*",), min_fidelity=Fidelity.T2_PROXY,
                            note="earnings release timestamps")]

    def init(self, ctx):
        self.st = {"cands": {}, "pending": {}, "pos": {}, "_flags": set()}

    def on_session_start(self, ctx):
        cal = ctx.calendar()
        d = ctx.session_date()
        o, _ = cal.open_close(d)
        po, _ = cal.open_close(cal.prev_session(d))
        self.st["cands"] = {}
        self.st["pending"] = {}
        evs = _events_known(ctx, "EARNINGS", po, o)       # first RTH day after the report = today
        if not evs:
            return
        members = _universe(ctx, "SP500")
        for e in evs:
            if e.symbol not in members or e.symbol in self.st["pos"] or ctx.position(e.symbol) != 0:
                continue
            dd = _daily(ctx, e.symbol, 30)
            if len(dd) < 15 or not _fresh(dd, cal, d):
                continue
            self.st["cands"][e.symbol] = {"pc": float(dd["close"].iloc[-1]), "atr": _atr14(dd)}
            _flag_once(self.st, ctx, Fidelity.T2_PROXY, f"{self.id}: earnings release time approximated (BMO/AMC)")

    def _first30(self, ctx, sym, d):
        o, _ = ctx.calendar().open_close(d)
        b = ctx.session_bars(sym)
        if b is None or len(b) == 0 or b.index[0] != o:
            return None
        b = b[b.index < o + pd.Timedelta(minutes=30)]
        if len(b) == 0 or b["known_at"].iloc[-1] != o + pd.Timedelta(minutes=30):
            return None
        return b

    def on_bar(self, ctx):
        cal = ctx.calendar()
        d = ctx.session_date()
        t = ctx.now.value
        if self.st["cands"] and t == label_ns(cal, d, "10:00"):
            for sym, c in self.st["cands"].items():
                b = self._first30(ctx, sym, d)
                if b is None:
                    continue
                info = self.check(ctx, sym, d, b, c)
                if info is not None:
                    self.st["pending"][sym] = info
            self.st["cands"] = {}
        if self.st["pending"] and t == label_ns(cal, d, "10:05"):
            for sym, info in self.st["pending"].items():
                self.enter(ctx, sym, d, info)
            self.st["pending"] = {}
        self.manage(ctx, d, t)

    def check(self, ctx, sym, d, b, c):  # pragma: no cover - abstract
        raise NotImplementedError

    def enter(self, ctx, sym, d, info):  # pragma: no cover - abstract
        raise NotImplementedError

    def manage(self, ctx, d, t):
        pass


@register
class S039(_EarningsGapBase):
    id = "S039"
    name = "Post-earnings gap continuation"
    sizing = "SWING_RISK"
    params = {"gap": 0.03, "hold": 3, "rvol": 2.0, "atr_mult": 2.0}
    grid = {"gap": [0.02, 0.03, 0.05], "hold": [1, 3, 5]}
    deviations = COMMON_DEVIATIONS + (
        "Universe SP500 point-in-time (STRATEGY_GUIDE 'STK'); 5m RTH bars.",
        "'First RTH day after report' = first session whose open is after the release known_at.",
        "Gap = first 5m bar open / prior completed (adjusted) daily close - 1; first 30m close = close of the "
        "bar ending 10:00; RVOL = cumulative volume 09:30-10:00 / median of the same over the prior 20 "
        "sessions (indicators.relative_volume, cumulative); all evaluated at 10:00, buy submitted at the bar "
        "ending 10:05 -> fills at the 10:05 bar open ('buy 10:05').",
        "Stop: daily close < (highest daily close since entry) - 2 x ATR14 (Wilder, daily, updated each close) "
        "-> market sell at the next open. Initial stop for SWING_RISK sizing / R = entry price - 2 x ATR14 "
        "(daily, through the prior session).",
        "Time exit: MOC at the close of the `hold`-th session after the entry session.",
    )
    spec = ("First RTH day after report: open gap>=+3% and first 30m close above open; buy 10:05 next bar with "
            "RVOL >=2. || Exit after 3 sessions or 2 ATR14 trailing close stop.")

    def check(self, ctx, sym, d, b, c):
        op = float(b["open"].iloc[0])
        if not (op / c["pc"] - 1.0 >= float(self.params["gap"])):
            return None
        if not float(b["close"].iloc[-1]) > op:
            return None
        h = ctx.history(sym, 78 * 23, "5m", adjusted=True)
        if h is None or len(h) == 0:
            return None
        rv = relative_volume(h, lookback=20, cumulative=True)
        k = b.index[-1]
        if k not in rv.index:
            return None
        r = float(rv.loc[k, "rvol"])
        if not r >= float(self.params["rvol"]):
            return None
        return {"atr": c["atr"]}

    def enter(self, ctx, sym, d, info):
        px = ctx.mark(sym)
        a = info["atr"]
        if not (px == px and px > 0 and a == a and a > 0):
            return
        stop = px - float(self.params["atr_mult"]) * a
        q = SZ.size_swing_risk_ctx(ctx, sym, px, stop)
        if q <= 0:
            return
        ctx.order(sym, q, tag="entry", initial_stop=stop)
        self.st["pos"][sym] = {"entry": d, "max_close": float("-inf"),
                               "exit_session": ctx.calendar().next_session(d, int(self.params["hold"]))}

    def manage(self, ctx, d, t):
        if not self.st["pos"]:
            return
        cal = ctx.calendar()
        _, c = cal.open_close(d)
        for sym, p in list(self.st["pos"].items()):
            q = ctx.position(sym)
            if q == 0 and not ctx.open_orders(sym) and d > p["entry"]:
                self.st["pos"].pop(sym)
                continue
            if d == p["exit_session"] and t == label_ns(cal, d, "15:45") and q > 0:
                ctx.cancel_all(sym)
                ctx.order(sym, -q, type=OrderType.MOC, tag="time_exit")
                continue
            if t == c.value and q > 0 and d < p["exit_session"]:
                dd = _daily(ctx, sym, 40)
                if len(dd) == 0 or _bar_sessions(dd)[-1] != d:
                    continue
                cl = float(dd["close"].iloc[-1])
                p["max_close"] = max(p["max_close"], cl)
                a = _atr14(dd)
                if a == a and cl < p["max_close"] - float(self.params["atr_mult"]) * a:
                    ctx.cancel_all(sym)
                    ctx.order(sym, -q, tag="stop_close")
                    self.st["pos"].pop(sym)


@register
class S040(_EarningsGapBase):
    id = "S040"
    name = "Post-earnings gap fade"
    sizing = "DAY_RISK"
    params = {"gap": 0.06, "target_frac": 0.50}
    grid = {"gap": [0.04, 0.06, 0.10], "target_frac": [0.25, 0.50, 1.00]}
    deviations = COMMON_DEVIATIONS + (
        "Universe SP500 point-in-time (STRATEGY_GUIDE 'STK'); 5m RTH bars.",
        "'First RTH day after report' = first session whose open is after the release known_at.",
        "First 30m close = close of the bar ending 10:00; VWAP = session VWAP (typical price) through that "
        "bar; short submitted at the bar ending 10:05 (fills at the 10:05 bar open).",
        "'Opening high' = high of the first 30 minutes (the signal window). Target = open - target_frac x "
        "(open - previous close) (0.5 = the sheet's midpoint). Locate: portal.borrow shortable, else skipped.",
        "Exit 15:55 enforced by the engine (DAY_RISK flat_by).",
    )
    spec = ("First RTH day after report: open gap >=+6%; first 30m close below open and below VWAP; short 10:05, "
            "if locate. || Target midpoint between open and previous close; stop opening high; exit 15:55.")

    def check(self, ctx, sym, d, b, c):
        op = float(b["open"].iloc[0])
        if not (op / c["pc"] - 1.0 >= float(self.params["gap"])):
            return None
        c30 = float(b["close"].iloc[-1])
        vw = float(session_vwap(b)["vwap"].iloc[-1])
        if not (c30 < op and c30 < vw):
            return None
        return {"open": op, "pc": c["pc"], "oh": float(b["high"].max())}

    def enter(self, ctx, sym, d, info):
        try:
            ok, fee, _ = ctx.portal().borrow(sym)
        except Exception:
            ok = False
        if not ok:
            ctx.log(f"S040: no locate for {sym}")
            return
        px = ctx.mark(sym)
        stop = info["oh"]
        if not (px == px and px > 0 and stop > px):
            return
        target = info["open"] - float(self.params["target_frac"]) * (info["open"] - info["pc"])
        q = SZ.size_day_risk_ctx(ctx, sym, px, stop)
        if q >= 0:
            return
        ctx.bracket(sym, q, stop=stop, target=target if target < px else None, tag="entry")


# ================================================================== S043 / S044
class _IndexEventBase(Strategy):
    bar_freq = "5m"
    asset = "STK"
    universe = ()

    def init(self, ctx):
        self.st = {"last": None, "queue": [], "pos": {}, "_flags": set()}

    def _exit_ns(self, ctx, e, entry_session) -> Optional[int]:
        cal = ctx.calendar()
        p = e.payload or {}
        eff = p.get("effective_date")
        if eff is None or pd.isna(eff):
            eff = e.effective_at
        if eff is None or pd.isna(eff):
            return None
        es = as_session_date(eff)
        if not cal.is_session(es):
            es = cal.next_session(es)
        off = int(self.params.get("effective_offset", 0))
        if off:
            es = cal.next_session(es, off)
        w = self.params.get("window")
        if w:
            ws = cal.next_session(entry_session, int(w) - 1) if int(w) > 1 else pd.Timestamp(entry_session)
            es = min(es, ws)
        if es < pd.Timestamp(entry_session):
            return None
        return label_ns(cal, es, "15:50")

    def _manage_exits(self, ctx):
        t = ctx.now.value
        for sym, p in list(self.st["pos"].items()):
            if t >= p["exit_ns"]:
                self.close(ctx, sym, p)
                self.st["pos"].pop(sym)

    def close(self, ctx, sym, p):
        ctx.cancel_all(sym)
        q = ctx.position(sym)
        if q:
            ctx.order(sym, -q, tag="exit")


@register
class S043(_IndexEventBase):
    id = "S043"
    name = "Index addition announcement drift"
    sizing = "SWING_NOTIONAL"
    params = {"entry": "ann+1min", "effective_offset": 0, "hedge": True, "beta_n": 252}
    grid = {"entry": ["ann+1min", "next_day"], "effective_offset": [0, 1]}
    requires = [Requirement("bars", "5m", symbols=("SPY",)), Requirement("bars", "1d", symbols=("SPY",)),
                Requirement("events", "INDEX_ADD", symbols=("*",), min_fidelity=Fidelity.T2_PROXY,
                            note="timestamped S&P 500 addition announcements (historical archive)")]
    deviations = COMMON_DEVIATIONS + (
        "Traded names come from INDEX_ADD events (any index in the archive; S&P 500 in practice); 5m bars.",
        "entry='ann+1min': first 5m bar opening >= known_at + 1 min during RTH; after-hours announcements -> "
        "next RTH open (no extended-hours fills). entry='next_day': open of the first session after the "
        "announcement's session. Announcement clock times are approximate in the data (T2_PROXY).",
        "Effective date = payload effective_date (first session as an index member); exit 15:50 on it "
        "(+effective_offset sessions). Events whose exit precedes the entry are skipped.",
        "Beta hedge: short SPY beta x stock notional at entry, covered at exit; beta = 252d OLS beta of daily "
        "returns vs SPY through the prior session (min 60 obs, else 1.0). Gross (stock + hedge) <= 100% NAV; "
        "stock leg 10% NAV (SWING_NOTIONAL); the hedge is not limited by the 5% short-notional rule.",
    )
    spec = ("Following timestamped public index addition announcement, buy first next RTH open if announcement "
            "is after-hours, or next 5m open if during RTH. || Exit at index effective date 15:50; net return vs "
            "beta hedge SPY.")

    def _beta(self, ctx, sym) -> float:
        n = int(self.params["beta_n"])
        a = _daily(ctx, sym, n + 1)
        b = _daily(ctx, "SPY", n + 1)
        if len(a) < 61 or len(b) < 61:
            return 1.0
        df = pd.concat({"y": a["close"].pct_change(), "x": b["close"].pct_change()}, axis=1).dropna()
        if len(df) < 60:
            return 1.0
        beta = rolling_beta(df["y"], df["x"], window=min(n, len(df)), min_periods=60).iloc[-1]
        return float(beta) if beta == beta else 1.0

    def _enter(self, ctx, e):
        sym = e.symbol
        d = ctx.session_date()
        if sym in self.st["pos"] or ctx.position(sym) != 0 or sym == "SPY":
            return
        exit_ns = self._exit_ns(ctx, e, d)
        if exit_ns is None or exit_ns <= ctx.now.value:
            return
        px = ctx.mark(sym)
        spy = ctx.mark("SPY")
        if not (px == px and px > 0):
            ctx.log(f"S043: no price for {sym}")
            return
        _flag_once(self.st, ctx, Fidelity.T2_PROXY, "S043: index announcement time approximate")
        beta = self._beta(ctx, sym) if self.params["hedge"] else 0.0
        gross = SZ.equity_gross(ctx)
        notional = min(0.10 * ctx.nav, max(ctx.nav - gross, 0.0) / (1.0 + abs(beta)))
        q = math.floor(notional / px)
        if q <= 0:
            return
        ctx.order(sym, q, tag="entry")
        h = 0.0
        if self.params["hedge"] and spy == spy and spy > 0:
            h = -float(round(beta * q * px / spy))
            if h:
                ctx.order("SPY", h, tag="hedge")
        self.st["pos"][sym] = {"exit_ns": exit_ns, "hedge": h}

    def close(self, ctx, sym, p):
        super().close(ctx, sym, p)
        h = p["hedge"]
        spy_pos = ctx.position("SPY")
        if h < 0 and spy_pos < 0:
            ctx.order("SPY", min(-h, -spy_pos), tag="hedge_exit")

    def on_session_start(self, ctx):
        now = ctx.now
        last = self.st["last"] if self.st["last"] is not None else _prev_session_time(ctx, "close")
        for e in _events_known(ctx, "INDEX_ADD", last, now):
            self._enter(ctx, e)                         # after-hours announcement -> this open
        q = self.st["queue"]
        due = [x for x in q if x[0] <= now.value]
        self.st["queue"] = [x for x in q if x[0] > now.value]
        for _, e in due:
            self._enter(ctx, e)
        self.st["last"] = now

    def on_bar(self, ctx):
        now = ctx.now
        cal = ctx.calendar()
        d = ctx.session_date()
        self._manage_exits(ctx)
        last = self.st["last"] if self.st["last"] is not None else _prev_session_time(ctx, "close")
        _, c = cal.open_close(d)
        for e in _events_known(ctx, "INDEX_ADD", last, now):
            k = to_utc(e.known_at)
            if self.params["entry"] == "next_day":
                nxt = cal.next_session(session_date_of(k))
                self.st["queue"].append((int(cal.open_close(nxt)[0].value), e))
            else:
                self.st["queue"].append((int((k + pd.Timedelta(minutes=1)).value), e))
        self.st["last"] = now
        if now < c:
            due = [x for x in self.st["queue"] if x[0] <= now.value]
            self.st["queue"] = [x for x in self.st["queue"] if x[0] > now.value]
            for _, e in due:
                self._enter(ctx, e)


@register
class S044(_IndexEventBase):
    id = "S044"
    name = "Index deletion event short"
    sizing = "SWING_RISK"
    params = {"window": None, "borrow_cap": None, "atr_mult": 2.0}
    grid = {"window": [1, 5, 10], "borrow_cap": [0.05, 0.20, 0.50]}
    requires = [Requirement("bars", "5m", symbols=("SPY",)), Requirement("bars", "1d", symbols=("SPY",)),
                Requirement("events", "INDEX_DEL", symbols=("*",), min_fidelity=Fidelity.T2_PROXY,
                            note="timestamped S&P 500 deletion announcements (historical archive)"),
                Requirement("borrow", symbols=("SPY",), critical=False,
                            note="historical borrow availability/fees; default easy-to-borrow is T2")]
    deviations = COMMON_DEVIATIONS + (
        "Traded names come from INDEX_DEL events; 5m bars. Short at the open of the first session after the "
        "announcement (RTH announcements also wait for the next open, as the sheet says).",
        "Borrow: portal.borrow shortable required; borrow_cap (grid) skips names whose annual fee exceeds it; "
        "baseline has no fee cap. SSR: Reg SHO Rule 201 from 2011-02-28 (prior session low <= 90% of its "
        "prior close -> restricted today) -> skipped; earlier tick-test regimes are left to the engine.",
        "Initial mid = the entry fill's reference mid (bar mode: modelled mid); resting buy-stop at mid + "
        "2 x ATR14 (daily Wilder, through the prior session), placed when the engine reports the entry fill "
        "(on_fill arrives at the next 5m step, so the stop rests from ~5 minutes after the open). SWING_RISK size "
        "uses the prior close as the expected mid.",
        "Cover at 15:50 on the effective date (payload effective_date). 'Event window' N (grid) = also cover "
        "at 15:50 of the N-th session counting the entry session as 1, whichever comes first; baseline none.",
    )
    spec = ("Following timestamped index deletion announcement, short at next RTH open only if borrow available "
            "and no SSR restriction. || Cover effective date 15:50, or if stop 2ATR above initial mid.")

    def init(self, ctx):
        super().init(ctx)
        self.st["entries"] = {}

    def on_session_start(self, ctx):
        now = ctx.now
        d = ctx.session_date()
        last = self.st["last"] if self.st["last"] is not None else _prev_session_time(ctx, "open")
        self.st["last"] = now
        for e in _events_known(ctx, "INDEX_DEL", last, now):
            sym = e.symbol
            if sym in self.st["pos"] or ctx.position(sym) != 0:
                continue
            exit_ns = self._exit_ns(ctx, e, d)
            if exit_ns is None:
                continue
            try:
                ok, fee, _ = ctx.portal().borrow(sym)
            except Exception:
                ok, fee = False, float("nan")
            cap = self.params["borrow_cap"]
            if not ok or (cap is not None and not (fee <= float(cap))):
                ctx.log(f"S044: borrow unavailable/too expensive for {sym}")
                continue
            if _ssr_active(ctx, sym, d):
                ctx.log(f"S044: SSR active for {sym}")
                continue
            dd = _daily(ctx, sym, 30)
            if len(dd) < 15 or not _fresh(dd, ctx.calendar(), d):
                continue
            a = _atr14(dd)
            px = float(dd["close"].iloc[-1])
            if not (a == a and a > 0):
                continue
            stop_est = px + float(self.params["atr_mult"]) * a
            q = SZ.size_swing_risk_ctx(ctx, sym, px, stop_est)
            if q >= 0:
                continue
            _flag_once(self.st, ctx, Fidelity.T2_PROXY, "S044: index announcement time approximate")
            o = ctx.order(sym, q, tag="entry", initial_stop=stop_est)
            self.st["entries"][o.id] = {"sym": sym, "atr": a, "stop": None}
            self.st["pos"][sym] = {"exit_ns": exit_ns}

    def on_fill(self, ctx, fill):
        info = self.st["entries"].get(fill.order_id)
        if info is None:
            return
        if info["stop"] is None:
            mid = fill.reference_mid if fill.reference_mid == fill.reference_mid else fill.price
            info["stop"] = mid + float(self.params["atr_mult"]) * info["atr"]
        ctx.order(info["sym"], -fill.quantity, type=OrderType.STOP, stop_price=info["stop"],
                  tif=TimeInForce.GTC, tag="stop")

    def on_bar(self, ctx):
        self._manage_exits(ctx)


# ================================================================== S041
@register
class S041(Strategy):
    id = "S041"
    name = "Analyst earnings revision drift"
    bar_freq = "1mo"
    sizing = "CROSS_SECTION"
    asset = "STK"
    universe_name = "SP500"
    params = {"window_days": 30, "bucket": 0.10, "min_analysts": 5}
    grid = {"window_days": [30, 60], "bucket": [0.10, 0.20], "min_analysts": [3, 5]}
    FIELD_EPS = "eps_consensus_ntm"
    FIELD_N = "eps_num_analysts_ntm"
    requires = [Requirement("bars", "1d"),
                Requirement("fundamentals", "eps_consensus_ntm", symbols=("*",),
                            note="PIT history of next-12m consensus EPS and analyst counts (I/B/E/S-like); "
                                 "DoltHub eps_estimate snapshots hold only quarter/year consensus since ~2019 "
                                 "and are not ingested as NTM -> NOT_TESTABLE by design")]
    deviations = (
        "Universe SP500 point-in-time. Consensus read through portal.fundamentals fields "
        "'eps_consensus_ntm' / 'eps_num_analysts_ntm' (known_at = snapshot time); value 30 days ago = "
        "fundamentals(as_of = now - window).",
        "Revision = (C_now - C_then) / |C_then|; analyst count measured now.",
        "Month end = last NYSE session (decision at its close, orders fill next open). Buckets via "
        "indicators.cross_section.quantile_bucket; held names are kept while in the top/bottom quintile, new "
        "names enter from the top/bottom `bucket`; all names rebalanced to CROSS_SECTION weights monthly "
        "(sector caps when the security master has a 'sector' column).",
    )
    spec = ("At month-end rank change in consensus next-12m EPS last 30 calendar days, require >=5 analysts; long "
            "top decile, short bottom decile next RTH open. || Rebalance monthly; close if rank leaves top/bottom "
            "quintile.")

    def init(self, ctx):
        self.st = {"longs": set(), "shorts": set(), "sectors": None}

    def on_bar(self, ctx):
        cal = ctx.calendar()
        if not cal.is_last_session_of_month(ctx.session_date()):
            return
        members = sorted(_universe(ctx, "SP500"))
        if not members:
            return
        p = ctx.portal()
        now = p.fundamentals(members, [self.FIELD_EPS, self.FIELD_N])
        then = p.fundamentals(members, [self.FIELD_EPS], as_of=ctx.now - pd.Timedelta(days=int(self.params["window_days"])))
        if now is None or len(now) == 0 or then is None or len(then) == 0:
            return
        cn = now[now["field"] == self.FIELD_EPS].set_index("symbol")["value"].astype(float)
        na = now[now["field"] == self.FIELD_N].set_index("symbol")["value"].astype(float)
        c0 = then[then["field"] == self.FIELD_EPS].set_index("symbol")["value"].astype(float)
        df = pd.concat({"cn": cn, "na": na, "c0": c0}, axis=1).dropna()
        df = df[(df["na"] >= float(self.params["min_analysts"])) & (df["c0"].abs() > 0)]
        if len(df) < 10:
            return
        rev = (df["cn"] - df["c0"]) / df["c0"].abs()
        qn = int(round(1.0 / float(self.params["bucket"])))
        b = quantile_bucket(rev, qn)
        b5 = quantile_bucket(rev, 5)
        keep_l = {s for s in self.st["longs"] if b5.get(s) == 5}
        keep_s = {s for s in self.st["shorts"] if b5.get(s) == 1}
        longs = keep_l | set(b[b == qn].index)
        shorts = (keep_s | set(b[b == 1].index)) - longs
        self.st["longs"], self.st["shorts"] = longs, shorts
        w = SZ.cross_section_weights(sorted(longs), sorted(shorts), _sector_map(self.st, ctx) or None)
        prices = {s: ctx.mark(s) for s in set(w) | set(ctx.positions())}
        tgt = SZ.weights_to_shares(w, ctx.nav, prices)
        for s, dq in SZ.rebalance_deltas(tgt, ctx.positions()).items():
            ctx.cancel_all(s)
            ctx.order(s, dq, tag="rebalance")


# ================================================================== S045
@register
class S045(Strategy):
    id = "S045"
    name = "Buyback announcement continuation"
    bar_freq = "1d"
    sizing = "SWING_RISK"
    asset = "STK"
    universe_name = "SP500"
    params = {"size": 0.05, "hold": 10, "atr_mult": 2.0}
    grid = {"size": [0.03, 0.05, 0.10], "hold": [5, 10, 20]}
    requires = [Requirement("bars", "1d"),
                Requirement("events", "BUYBACK", symbols=("*",), min_fidelity=Fidelity.T2_PROXY,
                            note="timestamped repurchase authorizations with % of market cap and an "
                                 "earnings-bundling flag; no free archive known -> NOT_TESTABLE by design")]
    deviations = (
        "Universe SP500 point-in-time; events BUYBACK{pct_mcap, with_earnings}. pct_mcap >= size (decimal).",
        "Entry: market order at the first RTH open after known_at (announcements known before the open trade "
        "that open; RTH announcements the next open).",
        "Trailing stop: resting stop at (highest daily high since entry) - 2 x ATR14 (Wilder, daily), "
        "ratcheted up at each close; initial stop/R = expected entry (last close) - 2 x ATR14.",
        "Time exit: MOC at the close of the `hold`-th session after the entry session.",
    )
    spec = ("Publicly announced repurchase authorization >=5% of market cap, exclude releases bundled with "
            "earnings, buy first next RTH open. || Exit after 10 trading sessions or 2ATR trailing stop.")

    def init(self, ctx):
        self.st = {"last": None, "pos": {}, "orders": {}, "_flags": set()}

    def _process(self, ctx, until):
        last = self.st["last"] if self.st["last"] is not None else _prev_session_time(ctx, "open")
        self.st["last"] = until
        evs = _events_known(ctx, "BUYBACK", last, until)
        if not evs:
            return
        members = _universe(ctx, "SP500")
        for e in evs:
            p = e.payload or {}
            sym = e.symbol
            try:
                pct = float(p.get("pct_mcap"))
            except (TypeError, ValueError):
                continue
            if bool(p.get("with_earnings", False)) or not (pct >= float(self.params["size"])):
                continue
            if sym not in members or sym in self.st["pos"] or ctx.position(sym) != 0:
                continue
            dd = _daily(ctx, sym, 30)
            if len(dd) < 15:
                continue
            a = _atr14(dd)
            px = float(dd["close"].iloc[-1])
            if not (a == a and a > 0):
                continue
            stop = px - float(self.params["atr_mult"]) * a
            q = SZ.size_swing_risk_ctx(ctx, sym, px, stop)
            if q <= 0:
                continue
            o = ctx.bracket(sym, q, stop=stop, tag="entry")      # protective stop live from the entry fill
            self.st["orders"][o.id] = sym
            self.st["pos"][sym] = {"entry": None, "hh": float("-inf"), "atr": a, "stop": stop, "stop_id": None}

    def on_fill(self, ctx, fill):
        sym = self.st["orders"].get(fill.order_id)
        if sym is None or sym not in self.st["pos"]:
            return
        p = self.st["pos"][sym]
        if p["entry"] is None:
            p["entry"] = ctx.session_date()

    def _set_stop(self, ctx, sym, p, level):
        q = ctx.position(sym)
        if q <= 0:
            return
        ctx.cancel_all(sym)                              # bracket child or previous trailing stop
        o = ctx.order(sym, -q, type=OrderType.STOP, stop_price=level, tif=TimeInForce.GTC, tag="stop",
                      oco_group=f"S045-{sym}")
        p["stop_id"], p["stop"] = o.id, level

    def on_session_start(self, ctx):
        cal = ctx.calendar()
        d = ctx.session_date()
        for sym, p in list(self.st["pos"].items()):
            if p["entry"] is not None and d == cal.next_session(p["entry"], int(self.params["hold"])):
                q = ctx.position(sym)
                if q > 0:
                    self._set_stop(ctx, sym, p, p["stop"])     # stop and MOC share one OCO group
                    ctx.order(sym, -q, type=OrderType.MOC, tag="time_exit", oco_group=f"S045-{sym}")
        self._process(ctx, ctx.now)

    def on_bar(self, ctx):
        d = ctx.session_date()
        for sym, p in list(self.st["pos"].items()):
            q = ctx.position(sym)
            if p["entry"] is None:
                if not ctx.open_orders(sym):
                    self.st["pos"].pop(sym)          # entry never filled
                continue
            if q <= 0:
                ctx.cancel_all(sym)
                self.st["pos"].pop(sym)
                continue
            dd = _daily(ctx, sym, 40)
            if len(dd) == 0 or _bar_sessions(dd)[-1] != d:
                continue
            sess = _bar_sessions(dd)
            p["hh"] = max(p["hh"], float(dd["high"][sess >= p["entry"]].max()))
            a = _atr14(dd)
            lvl = max(p["hh"] - float(self.params["atr_mult"]) * (a if a == a else p["atr"]), p["stop"])
            if p["stop_id"] is None or lvl > p["stop"]:
                self._set_stop(ctx, sym, p, lvl)        # ratchet up only
        self._process(ctx, ctx.now)
