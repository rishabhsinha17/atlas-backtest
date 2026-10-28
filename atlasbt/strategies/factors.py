"""Position / cross-sectional factor strategies: S085-S088, S091-S100.

Common mechanics (declared once, module-wide; every class repeats the relevant
ones in ``deviations``):

* Universe: portal universe ``"SP500"`` (point-in-time membership, STRATEGY_GUIDE
  SP500_PIT). Survivorship caveat: constituent history before 2011 is less
  reliable in free sources -> recorded in ``deviations`` (copied to result meta).
* Timing: signals at the close of the last NYSE session of the month (rule 10);
  market orders submitted at that close fill at the NEXT session open (engine).
* Prices: total-return (split + dividend back-adjusted as of the signal time)
  daily closes from ``ctx.history(adjusted=True)``; month-end values are the
  closes on the calendar month-end sessions (a name without a bar on that exact
  session has no value -> not ranked).
* Sizing: CROSS_SECTION via ``atlasbt.risk.sizing.cross_section_weights``
  (+50%/-50% NAV equal weight, single name <= 2% NAV, sector caps +/-10% per
  side); weight removed by caps stays in cash (sizing module rule). Orders =
  ``rebalance_deltas`` between target and current shares, risk-reducing orders
  first.
* Fundamentals: ``portal.fundamentals`` returns only the latest value per
  (symbol, field) known at ``as_of``. Factors that need history (prior fiscal
  year, TTM, 5y growth) use :class:`FundHistory`, which re-queries the portal
  at past month-end closes (``as_of`` <= now, PIT-safe) and keeps every value as
  it was known then; ``known_at`` = SEC acceptance time, so only filings public
  at the signal close are used.
* Sectors/industries/exchange come from ``portal.security_master(as_of)``
  (columns tried in order: sector ``gics_sector``/``sector``/``sector_sic``;
  industry ``gics_sub_industry``/``gics_industry``/``industry``; ``sic``;
  exchange ``exchange``/``primary_exchange``/``listing_exchange``/``exchanges``).
  Current GICS from Wikipedia is not point-in-time (declared). If exchange data
  is missing, NYSE breakpoints are approximated with the universe (T2 flag).
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.types import ActionType, Fidelity
from ..engine.strategy import Requirement, Strategy
from ..indicators import cross_section as xs
from ..indicators import realized_vol, rolling_ols
from ..risk.sizing import cross_section_weights, rebalance_deltas, weights_to_shares
from .registry import register

ET = "America/New_York"

# Select Sector SPDRs (sheet SECTOR_SPDR) with first trading dates (STRATEGY_GUIDE §3).
SECTOR_ETFS = {
    "Materials": ("XLB", "1998-12-22"), "Energy": ("XLE", "1998-12-22"), "Financials": ("XLF", "1998-12-22"),
    "Industrials": ("XLI", "1998-12-22"), "Information Technology": ("XLK", "1998-12-22"),
    "Consumer Staples": ("XLP", "1998-12-22"), "Utilities": ("XLU", "1998-12-22"),
    "Health Care": ("XLV", "1998-12-22"), "Consumer Discretionary": ("XLY", "1998-12-22"),
    "Real Estate": ("XLRE", "2015-10-08"), "Communication Services": ("XLC", "2018-06-19"),
}
SECTOR_SPDR_9 = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")

SURVIVORSHIP_NOTE = ("Universe = portal 'SP500' point-in-time membership; constituent history before 2011 "
                     "is reconstructed from free sources and may carry survivorship bias (caveat recorded)")
COMMON_XS_DEVIATIONS = [
    SURVIVORSHIP_NOTE,
    "Month-end = last NYSE session of the month; signal at its close, market orders fill at the next session "
    "open (rule 1/10)",
    "Total returns from split+dividend back-adjusted closes as of the signal time",
    "CROSS_SECTION caps via risk.sizing.cross_section_weights: weight removed by the 2% name cap / 10% sector "
    "cap stays in cash (not redistributed); sector caps per side",
    "GICS sector/industry from security_master is the current classification (not point-in-time)",
    "Names whose sector is unknown are treated as their own group (cap not enforceable) and flagged T2",
]


# =====================================================================================
# generic helpers (also used by atlasbt.strategies.statarb)
# =====================================================================================
def session_dates(idx) -> pd.DatetimeIndex:
    """UTC bar-start index -> naive ET session dates."""
    di = pd.DatetimeIndex(idx)
    if di.tz is None:
        di = di.tz_localize("UTC")
    return di.tz_convert(ET).tz_localize(None).normalize()


def closes(ctx, symbol: str, n: int, adjusted: bool = True) -> pd.Series:
    """Last ``n`` completed daily closes (session-date index)."""
    try:
        df = ctx.history(symbol, n, "1d", adjusted=adjusted)
    except Exception:
        return pd.Series(dtype=float)
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(df["close"].to_numpy(float), index=session_dates(df.index))
    return s[~s.index.duplicated(keep="last")]


def close_panel(ctx, symbols: Iterable[str], n: int, adjusted: bool = True) -> pd.DataFrame:
    cols = {}
    for s in symbols:
        c = closes(ctx, s, n, adjusted)
        if len(c):
            cols[s] = c
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index()


def raw_close_on(ctx, symbol: str, session: pd.Timestamp) -> float:
    """Unadjusted close of ``session`` (direct PIT portal query: the engine's
    history fast path only holds ~60 sessions before the run start)."""
    cal = ctx.calendar()
    o, c = cal.open_close(session)
    try:
        df = ctx.portal().bars(symbol, "1d", o - pd.Timedelta(hours=1), c + pd.Timedelta(1, "ns"), adjusted=False)
    except Exception:
        return float("nan")
    if df is None or len(df) == 0:
        return float("nan")
    return float(df["close"].iloc[-1])


def month_end_sessions(cal, session: pd.Timestamp, k: int) -> list[pd.Timestamp]:
    """[M_0, M_1, ..., M_k]: last sessions of the current month and the k prior months."""
    session = pd.Timestamp(session).normalize()
    start = (session - pd.DateOffset(months=k + 1)).replace(day=1)
    ss = cal.sessions(start, session)
    s = pd.Series(ss, index=ss)
    last = s.groupby([ss.year, ss.month]).last().tolist()
    last = [pd.Timestamp(x) for x in last if pd.Timestamp(x) <= session]
    out = list(reversed(last))[: k + 1]
    return out


def is_month_end(ctx) -> bool:
    return bool(ctx.calendar().is_last_session_of_month(ctx.session_date()))


def is_week_end(ctx) -> bool:
    cal = ctx.calendar()
    s = ctx.session_date()
    nxt = cal.next_session(s)
    return pd.Timestamp(nxt).isocalendar()[1] != pd.Timestamp(s).isocalendar()[1] or (nxt - s).days > 6


def n_sessions_between(cal, a: pd.Timestamp, b: pd.Timestamp) -> int:
    return len(cal.sessions(pd.Timestamp(a), pd.Timestamp(b)))


def daily_returns(px: pd.DataFrame) -> pd.DataFrame:
    return px / px.shift(1) - 1.0


def q_of(frac: float) -> int:
    return int(round(1.0 / float(frac)))


def sector_etf(sector: Optional[str], session) -> Optional[str]:
    if not sector or sector not in SECTOR_ETFS:
        return None
    etf, inc = SECTOR_ETFS[sector]
    return etf if pd.Timestamp(session) >= pd.Timestamp(inc) else None


def ols_beta(y: pd.Series, x: pd.Series, min_obs: int) -> float:
    d = pd.concat([y, x], axis=1).dropna()
    if len(d) < min_obs:
        return float("nan")
    xv = d.iloc[:, 1].to_numpy()
    yv = d.iloc[:, 0].to_numpy()
    vx = np.var(xv, ddof=1)
    if not vx > 0:
        return float("nan")
    return float(np.cov(yv, xv, ddof=1)[0, 1] / vx)


class SecMaster:
    """Security-master lookups (sector / industry / SIC / exchange)."""

    SECTOR_COLS = ("gics_sector", "sector", "sector_sic")
    INDUSTRY_COLS = ("gics_sub_industry", "gics_industry", "industry")
    EXCH_COLS = ("exchange", "primary_exchange", "listing_exchange", "exchanges")

    def __init__(self):
        self._df = pd.DataFrame()
        self._session = None

    def refresh(self, ctx) -> None:
        s = ctx.session_date()
        if self._session is not None and s == self._session:
            return
        try:
            df = ctx.portal().security_master()
        except Exception:
            df = None
        if df is None or len(df) == 0 or "symbol" not in df.columns:
            self._df = pd.DataFrame()
        else:
            self._df = df.drop_duplicates("symbol", keep="last").set_index("symbol")
        self._session = s

    def _get(self, sym: str, cols: Sequence[str]):
        if self._df.empty or sym not in self._df.index:
            return None
        row = self._df.loc[sym]
        for c in cols:
            if c in self._df.columns:
                v = row[c]
                if v is not None and not (isinstance(v, float) and math.isnan(v)) and str(v) != "":
                    return v
        return None

    def sector(self, sym: str) -> Optional[str]:
        v = self._get(sym, self.SECTOR_COLS)
        return None if v is None else str(v)

    def industry(self, sym: str, mode: str = "GICS") -> Optional[str]:
        if mode.upper() == "SIC":
            v = self._get(sym, ("sic",))
            try:
                return f"SIC{int(float(v)) // 100:02d}" if v is not None else None
            except (TypeError, ValueError):
                return None
        v = self._get(sym, self.INDUSTRY_COLS)
        return None if v is None else str(v)

    def is_nyse(self, sym: str) -> Optional[bool]:
        v = self._get(sym, self.EXCH_COLS)
        if v is None:
            return None
        toks = [t.strip().upper() for t in str(v).replace(";", ",").split(",")]
        return any(t in ("NYSE", "N", "XNYS", "NEW YORK STOCK EXCHANGE") for t in toks)

    def sectors(self, syms: Iterable[str]) -> dict[str, str]:
        return {s: (self.sector(s) or f"UNKNOWN:{s}") for s in syms}


def nyse_mask(ctx, sm: SecMaster, syms: Sequence[str]) -> pd.Series:
    """Boolean NYSE flags; if exchange data is missing for most names the whole
    universe is used for breakpoints (declared approximation, T2 flag)."""
    flags = {s: sm.is_nyse(s) for s in syms}
    known = [v for v in flags.values() if v is not None]
    if len(known) < 0.5 * max(len(syms), 1) or not any(known):
        ctx.flag_fidelity(Fidelity.T2_PROXY, "NYSE size breakpoints approximated by the SP500 universe "
                                             "(no exchange listing data)")
        return pd.Series(True, index=list(syms))
    return pd.Series({s: bool(v) for s, v in flags.items()})


def flag_unknown_sectors(ctx, sectors: dict[str, str]) -> None:
    n = sum(1 for v in sectors.values() if v.startswith("UNKNOWN:"))
    if n:
        ctx.flag_fidelity(Fidelity.T2_PROXY, "sector unknown for some names: sector cap not enforceable for them")


def rebalance_to(ctx, weights: dict[str, float], tag: str = "rebalance") -> list:
    """Submit market orders moving current positions to target weights (fraction
    of current NAV). Risk-reducing orders first. Names without a price are left alone."""
    pos = {s: q for s, q in ctx.positions().items() if q and not ctx.instrument(s).is_option}
    syms = set(weights) | set(pos)
    prices = {}
    for s in syms:
        try:
            prices[s] = float(ctx.mark(s))
        except Exception:
            prices[s] = float("nan")
    good = {s for s, p in prices.items() if p == p and p > 0}
    target = weights_to_shares({s: w for s, w in weights.items() if s in good}, ctx.nav, prices)
    deltas = rebalance_deltas(target, {s: q for s, q in pos.items() if s in good})
    ctx.cancel_all()
    orders = []
    reduce_ = [(s, d) for s, d in deltas.items() if abs(pos.get(s, 0.0) + d) < abs(pos.get(s, 0.0))]
    grow = [(s, d) for s, d in deltas.items() if (s, d) not in reduce_]
    for s, d in sorted(reduce_) + sorted(grow):
        orders.append(ctx.order(s, d, tag=tag))
    return orders


# =====================================================================================
# fundamentals history (PIT)
# =====================================================================================
class FundHistory:
    """Point-in-time history of fundamentals built from repeated
    ``portal.fundamentals(..., as_of)`` queries at month-end closes.

    The DataPortal contract only returns the latest value per (symbol, field)
    known at ``as_of``. Re-querying at past month-ends (as_of <= now) recovers
    each fiscal period's value *as it was known then*; new month-ends are
    appended as the backtest advances. Values are never back-filled."""

    COLS = ["symbol", "field", "value", "period_end", "known_at"]

    def __init__(self, fields: Sequence[str], bootstrap_months: int):
        self.fields = tuple(dict.fromkeys(fields))
        self.boot = int(bootstrap_months)
        self._parts: list[pd.DataFrame] = []
        self._seen: set[str] = set()
        self._by: Optional[dict[str, pd.DataFrame]] = None
        self.queried: set[str] = set()

    # ---------------------------------------------------------------- loading
    def _add(self, df: Optional[pd.DataFrame]) -> None:
        if df is None or len(df) == 0 or not set(self.COLS) <= set(df.columns):
            return
        d = df[self.COLS].copy()
        pe = pd.to_datetime(d["period_end"])
        if getattr(pe.dt, "tz", None) is not None:
            pe = pe.dt.tz_convert(ET).dt.tz_localize(None)
        d["period_end"] = pe.dt.normalize()
        ka = pd.to_datetime(d["known_at"])
        d["known_at"] = ka.dt.tz_localize("UTC") if getattr(ka.dt, "tz", None) is None else ka.dt.tz_convert("UTC")
        d["value"] = pd.to_numeric(d["value"], errors="coerce").astype(float)
        self._parts.append(d.dropna(subset=["value"]))
        self._by = None

    def update(self, ctx, symbols: Sequence[str]) -> None:
        syms = [s for s in symbols]
        new = [s for s in syms if s not in self._seen]
        portal = ctx.portal()
        if new and self.boot > 0:
            for m in month_end_sessions(ctx.calendar(), ctx.session_date(), self.boot)[1:]:
                asof = ctx.calendar().open_close(m)[1]
                if asof >= ctx.now:
                    continue
                try:
                    self._add(portal.fundamentals(new, list(self.fields), as_of=asof))
                except Exception:
                    pass
        self._seen.update(new)
        try:
            self._add(portal.fundamentals(syms, list(self.fields)))
        except Exception:
            pass
        self.queried.update(syms)

    def _index(self) -> dict[str, pd.DataFrame]:
        if self._by is None:
            if not self._parts:
                self._by = {}
            else:
                d = pd.concat(self._parts, ignore_index=True)
                d = d.drop_duplicates(["symbol", "field", "period_end", "known_at"], keep="last")
                d = d.sort_values(["known_at", "period_end"], kind="mergesort")
                self._parts = [d]
                self._by = {s: g for s, g in d.groupby("symbol", sort=False)}
        return self._by

    def has_any(self) -> bool:
        return bool(self._index())

    # ---------------------------------------------------------------- queries
    def series(self, sym: str, field: str, cutoff) -> pd.Series:
        """value by period_end, using for each period the latest filing known at cutoff."""
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        g = self._index().get(sym)
        if g is None:
            return empty
        c = pd.Timestamp(cutoff)
        f = g[(g["field"] == field) & (g["known_at"] <= c)]
        if f.empty:
            return empty
        f = f.groupby("period_end", sort=True).tail(1).sort_values("period_end")
        return pd.Series(f["value"].to_numpy(), index=pd.DatetimeIndex(f["period_end"]))

    def latest(self, sym: str, field: str, cutoff) -> tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """(value, period_end, known_at) of the latest period known at cutoff (portal semantics)."""
        g = self._index().get(sym)
        if g is None:
            return float("nan"), None, None
        c = pd.Timestamp(cutoff)
        f = g[(g["field"] == field) & (g["known_at"] <= c)]
        if f.empty:
            return float("nan"), None, None
        pe = f["period_end"].max()
        r = f[f["period_end"] == pe].iloc[-1]
        return float(r["value"]), pd.Timestamp(pe), pd.Timestamp(r["known_at"])

    def value_at(self, sym: str, field: str, period_end, cutoff, tol_days: int = 20) -> float:
        s = self.series(sym, field, cutoff)
        if s.empty or period_end is None:
            return float("nan")
        d = np.abs((s.index - pd.Timestamp(period_end)).days)
        i = int(np.argmin(d))
        return float(s.iloc[i]) if d[i] <= tol_days else float("nan")

    def fy_end(self, sym: str, cutoff, fy_fields: Sequence[str], not_after=None) -> Optional[pd.Timestamp]:
        ends = []
        for f in fy_fields:
            s = self.series(sym, f, cutoff)
            if not_after is not None:
                s = s[s.index <= pd.Timestamp(not_after)]
            if len(s):
                ends.append(s.index.max())
        return max(ends) if ends else None

    def fy_values(self, sym: str, base: str, cutoff) -> pd.Series:
        return self.series(sym, base + "_fy", cutoff)

    def ttm(self, sym: str, base: str, cutoff, max_age_days: int = 550) -> float:
        """Trailing-12-month flow: FY + YTD(current) - YTD(same period last year).
        Q1 YTD = the 3-month ``_q`` value ending 75-105 days after a fiscal
        year end; later YTDs from ``_ytd``. If no comparable prior-year YTD is
        known, the latest FY value is used (declared)."""
        fy = self.series(sym, base + "_fy", cutoff)
        if fy.empty:
            return float("nan")
        E = fy.index.max()
        if (pd.Timestamp(cutoff).tz_convert(ET).tz_localize(None) - E).days > max_age_days:
            return float("nan")
        ytd = dict(self.series(sym, base + "_ytd", cutoff).items())
        q = self.series(sym, base + "_q", cutoff)
        for pe, v in q.items():
            d = (pe - fy.index).days
            if np.any((d >= 75) & (d <= 105)):
                ytd.setdefault(pe, v)
        later = [p for p in ytd if p > E]
        if not later:
            return float(fy[E])
        P = max(later)
        prev = [p for p in ytd if 350 <= (P - p).days <= 380]
        if not prev:
            return float(fy[E])
        Pp = min(prev, key=lambda p: abs((P - p).days - 365))
        return float(fy[E] + ytd[P] - ytd[Pp])


def split_factor(ctx, sym: str, after_date, through_date) -> float:
    """Product of split ratios with ex_date in (after_date, through_date]."""
    a = pd.Timestamp(after_date)
    if a.tzinfo is not None:
        a = a.tz_convert(ET).tz_localize(None)
    a = a.normalize()
    b = pd.Timestamp(through_date).normalize()
    if b <= a:
        return 1.0
    try:
        acts = ctx.portal().corporate_actions([sym], a, b)
    except Exception:
        return 1.0
    f = 1.0
    for x in acts:
        ex = pd.Timestamp(x.ex_date)
        if ex.tzinfo is not None:
            ex = ex.tz_convert(ET).tz_localize(None)
        ex = ex.normalize()
        if x.type == ActionType.SPLIT and a < ex <= b and x.ratio and x.ratio > 0:
            f *= float(x.ratio)
    return f


def market_cap(ctx, fh: FundHistory, sym: str, session=None, cutoff=None) -> float:
    """raw close(session) x shares_outstanding latest known at cutoff (default: that
    session's close), shares adjusted for splits between their filing and session."""
    cal = ctx.calendar()
    session = pd.Timestamp(session or ctx.session_date()).normalize()
    close_ts = cal.open_close(session)[1]
    cut = min(pd.Timestamp(cutoff) if cutoff is not None else close_ts, ctx.now)
    sh, _, kat = fh.latest(sym, "shares_outstanding", cut)
    if not (sh == sh and sh > 0):
        return float("nan")
    px = float(ctx.mark(sym)) if session == ctx.session_date() else raw_close_on(ctx, sym, session)
    if not (px == px and px > 0):
        return float("nan")
    return px * sh * split_factor(ctx, sym, kat, session)


# =====================================================================================
# base class
# =====================================================================================
class _XSBase(Strategy):
    bar_freq = "1mo"
    sizing = "CROSS_SECTION"
    asset = "STK"
    universe_name = "SP500"
    #: fundamentals fields kept in the PIT history (empty: no fundamentals)
    fund_fields: tuple = ()
    fund_bootstrap_months: int = 0
    long_gross: float = 0.50
    short_gross: float = 0.50

    def init(self, ctx) -> None:
        self.sm = SecMaster()
        self.fh = FundHistory(self.fund_fields, self.fund_bootstrap_months) if self.fund_fields else None
        self.book: dict = {}
        self.last_weights: dict[str, float] = {}
        self.n_rebalances = 0

    # schedule ------------------------------------------------------------------
    def is_rebalance(self, ctx) -> bool:
        return True

    def on_bar(self, ctx) -> None:
        if not is_month_end(ctx):
            return
        syms = [s for s in self.symbols(ctx)]
        if not syms:
            return
        self.sm.refresh(ctx)
        if self.fh is not None:
            self.fh.update(ctx, syms)
        if self.is_rebalance(ctx):
            w = self.target_weights(ctx, syms)
        else:
            w = self.maintenance_weights(ctx, syms)
        if w is None:
            return
        self.last_weights = dict(w)
        self.n_rebalances += 1
        rebalance_to(ctx, w)

    def target_weights(self, ctx, syms: list[str]) -> Optional[dict[str, float]]:  # pragma: no cover
        raise NotImplementedError

    def maintenance_weights(self, ctx, syms: list[str]) -> Optional[dict[str, float]]:
        return None

    # helpers -------------------------------------------------------------------
    def weights_from(self, ctx, longs: Sequence[str], shorts: Sequence[str], long_gross=None,
                     short_gross=None) -> dict[str, float]:
        longs, shorts = sorted(longs), sorted(shorts)
        self.book = {"long": list(longs), "short": list(shorts)}
        sec = self.sm.sectors(list(longs) + list(shorts))
        flag_unknown_sectors(ctx, sec)
        return cross_section_weights(longs, shorts, sec,
                                     long_gross=self.long_gross if long_gross is None else long_gross,
                                     short_gross=self.short_gross if short_gross is None else short_gross)

    def month_end_prices(self, ctx, syms: Sequence[str], k: int, extra_sessions: int = 0) -> tuple[pd.DataFrame, list]:
        cal = ctx.calendar()
        me = month_end_sessions(cal, ctx.session_date(), k)
        if len(me) < k + 1:
            return pd.DataFrame(), me
        n = n_sessions_between(cal, me[-1], ctx.session_date()) + 1 + extra_sessions
        px = close_panel(ctx, syms, n)
        return px, me

    @staticmethod
    def at(px: pd.DataFrame, date) -> pd.Series:
        if px.empty or date not in px.index:
            return pd.Series(np.nan, index=px.columns)
        return px.loc[date]

    def select(self, score: pd.Series, q: int, high_long: bool = True, groups: Optional[pd.Series] = None,
               mask: Optional[pd.Series] = None) -> tuple[list[str], list[str]]:
        s = score.dropna()
        if len(s) < 2:
            return [], []
        if groups is not None:
            b = xs.group_quantile_bucket(s, groups.reindex(s.index), q)
        else:
            b = xs.quantile_bucket(s, q, None if mask is None else mask.reindex(s.index).fillna(False))
        top = list(b.index[b == q])
        bot = list(b.index[b == 1])
        return (top, bot) if high_long else (bot, top)

    def select_in_size_buckets(self, ctx, score: pd.Series, size: pd.Series, q: int, high_long: bool = True,
                               nyse: Optional[pd.Series] = None, n_size: int = 2) -> tuple[list[str], list[str]]:
        """Fama-French style: NYSE size breakpoints (median split by default), then
        score breakpoints from NYSE names inside each size group."""
        idx = score.dropna().index.intersection(size.dropna().index)
        if len(idx) < 2:
            return [], []
        nyse = (nyse if nyse is not None else pd.Series(True, index=idx)).reindex(idx).fillna(False)
        if not nyse.any():
            nyse = pd.Series(True, index=idx)
        sb = xs.size_buckets(size.reindex(idx), nyse, n_size)
        longs, shorts = [], []
        for g in sorted(sb.dropna().unique()):
            m = sb.index[sb == g]
            um = nyse.reindex(m)
            if not um.any():
                um = pd.Series(True, index=m)
            l, s = self.select(score.reindex(m), q, high_long, mask=um)
            longs += l
            shorts += s
        return longs, shorts


def _common_requires(fund: bool = False, spy: bool = False, sector_etfs: bool = False) -> list[Requirement]:
    r = [Requirement("bars", "1d", note="daily bars for the PIT SP500 universe (portal universe 'SP500')")]
    if spy:
        r.append(Requirement("bars", "1d", symbols=("SPY",), note="market regressor / hedge"))
    if sector_etfs:
        r.append(Requirement("bars", "1d", symbols=SECTOR_SPDR_9, note="sector ETF regressors"))
        r.append(Requirement("bars", "1d", symbols=("XLRE", "XLC"), critical=False,
                             note="later sector SPDRs (used from inception)"))
    if fund:
        r.append(Requirement("fundamentals", None, note="SEC XBRL fundamentals with known_at = filing acceptance"))
    return r


# =====================================================================================
# S085 - S088 momentum family
# =====================================================================================
@register
class S085(_XSBase):
    id = "S085"
    name = "Cross-sectional 12-2 momentum"
    params = {"lookback_m": 12, "skip_m": 1, "decile": 0.10, "industry_neutral": False, "vol_scale": False,
              "target_vol": 0.12}
    grid = {"lookback_m": [6, 9, 12], "skip_m": [1], "decile": [0.10, 0.20]}
    requires = _common_requires()
    deviations = COMMON_XS_DEVIATIONS + [
        "Formation 't-12 to t-2' read with t = holding month: at signal month-end M0 the score is "
        "P(M_skip)/P(M_lookback) - 1 (11 monthly returns for 12/1), i.e. the signal month is skipped",
        "'decile 10/20%' = top/bottom 10% (deciles) or 20% (quintiles) of ranked names",
        "Optional industry neutrality = deciles formed within GICS sector (param industry_neutral, default off)",
        "Optional volatility scaling (param vol_scale, default off) = Barroso & Santa-Clara (2015): book scaled "
        "by min(1, target_vol / realized vol of the formation long-short portfolio over the prior 126 sessions)",
    ]
    spec = ("At last trading day month t rank cumulative stock total return t-12 to t-2, skipping most recent 1m; "
            "long top decile, short bottom, enter next first-month RTH open. || Rebalance monthly; top/bottom "
            "deciles; volatility scale and industry neutral optionally.")

    def target_weights(self, ctx, syms):
        p = self.params
        L, k = int(p["lookback_m"]), int(p["skip_m"])
        px, me = self.month_end_prices(ctx, syms, L, extra_sessions=130 if p["vol_scale"] else 0)
        if px.empty:
            return None
        score = self.at(px, me[k]) / self.at(px, me[L]) - 1.0
        groups = pd.Series(self.sm.sectors(score.index)) if p["industry_neutral"] else None
        longs, shorts = self.select(score, q_of(p["decile"]), True, groups)
        w = self.weights_from(ctx, longs, shorts)
        if p["vol_scale"] and longs and shorts:
            r = daily_returns(px).iloc[-126:]
            wml = r[longs].mean(axis=1) - r[shorts].mean(axis=1)
            vol = float(wml.std(ddof=1) * math.sqrt(252))
            if vol > 0:
                f = min(1.0, float(p["target_vol"]) / vol)
                w = {s: v * f for s, v in w.items()}
        return w


@register
class S086(_XSBase):
    id = "S086"
    name = "Cross-sectional 6-1 momentum"
    params = {"formation_m": 6, "hold_m": 1}
    grid = {"formation_m": [3, 6, 9], "hold_m": [1, 3]}
    requires = _common_requires()
    deviations = COMMON_XS_DEVIATIONS + [
        "'months t-6 to t-1' with t = holding month = the 6 most recent completed months: P(M0)/P(M6)-1",
        "hold_m=3: Jegadeesh-Titman overlapping cohorts; target = average of the last 3 monthly cohorts' "
        "CROSS_SECTION weights (each cohort 1/3 of the book; book under-invested during the first 2 months)",
    ]
    spec = ("At month end rank total return over months t-6 to t-1; long top decile, short bottom, enter "
            "next-month opening. || Monthly rebalance; replace positions exiting decile.")

    def init(self, ctx):
        super().init(ctx)
        self.cohorts: list[dict] = []

    def target_weights(self, ctx, syms):
        F, H = int(self.params["formation_m"]), int(self.params["hold_m"])
        px, me = self.month_end_prices(ctx, syms, F)
        if px.empty:
            return None
        score = self.at(px, me[0]) / self.at(px, me[F]) - 1.0
        longs, shorts = self.select(score, 10, True)
        self.cohorts.append(self.weights_from(ctx, longs, shorts))
        self.cohorts = self.cohorts[-H:]
        out: dict[str, float] = {}
        for c in self.cohorts:
            for s, v in c.items():
                out[s] = out.get(s, 0.0) + v / H
        return {s: v for s, v in out.items() if abs(v) > 1e-12}


@register
class S087(_XSBase):
    id = "S087"
    name = "Industry momentum"
    params = {"industry_map": "GICS", "rank_m": 6, "q_industries": 5}
    grid = {"industry_map": ["GICS", "SIC"], "rank_m": [3, 6, 12]}
    fund_fields = ("shares_outstanding",)
    fund_bootstrap_months = 0
    requires = _common_requires(fund=True)
    deviations = COMMON_XS_DEVIATIONS + [
        "Industry = GICS sub-industry (security_master gics_sub_industry; fallback gics_industry) or SIC 2-digit "
        "major group; GICS sectors would leave ~2 sectors per side, which the 10% sector cap cannot hold",
        "Formation 'previous 6m excluding latest 1m' = P(M1)/P(M_{1+rank_m}) - 1 per stock",
        "Value weights = market cap at formation start, backed out as current market cap (raw close x latest "
        "filed shares_outstanding, split-adjusted) / (1 + total return since formation start)",
        "Names without shares_outstanding are excluded from the industry return",
        "'sector-balanced long/short' implemented by the CROSS_SECTION sector caps (+/-10% per side)",
        "Industries ranked into quintiles (q_industries=5); all stocks of top-quintile industries long, "
        "bottom-quintile industries short, equal weight",
    ]
    spec = ("At month end compute value-weight industry returns over previous 6m excluding latest 1m; long stocks "
            "in top quintile industries vs bottom quintile industries. || Rebalance next-month open, "
            "sector-balanced long/short.")

    def target_weights(self, ctx, syms):
        F = int(self.params["rank_m"])
        px, me = self.month_end_prices(ctx, syms, F + 1)
        if px.empty:
            return None
        ret = self.at(px, me[1]) / self.at(px, me[1 + F]) - 1.0
        since = self.at(px, me[0]) / self.at(px, me[1 + F]) - 1.0
        rows = []
        for s in ret.dropna().index:
            ind = self.sm.industry(s, self.params["industry_map"])
            if ind is None:
                continue
            mc = market_cap(ctx, self.fh, s)
            if not (mc == mc and mc > 0) or not (since[s] == since[s]):
                continue
            rows.append((s, ind, ret[s], mc / (1.0 + since[s])))
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["sym", "ind", "ret", "w"])
        ind_ret = df.groupby("ind").apply(lambda g: float(np.sum(g["ret"] * g["w"]) / np.sum(g["w"])),
                                          include_groups=False)
        if len(ind_ret) < 2:
            return None
        q = int(self.params["q_industries"])
        b = xs.quantile_bucket(ind_ret, q)
        top = set(b.index[b == q])
        bot = set(b.index[b == 1])
        longs = df.loc[df["ind"].isin(top), "sym"].tolist()
        shorts = df.loc[df["ind"].isin(bot), "sym"].tolist()
        return self.weights_from(ctx, longs, shorts)


@register
class S088(_XSBase):
    id = "S088"
    name = "Residual momentum"
    params = {"reg_window": 252, "mom_m": 12, "decile": 0.10, "beta_hedge": True}
    grid = {"reg_window": [126, 252], "mom_m": [6, 12]}
    requires = _common_requires(spy=True, sector_etfs=True)
    deviations = COMMON_XS_DEVIATIONS + [
        "Daily total returns regressed on SPY and the stock's sector SPDR (rolling OLS, window reg_window); the "
        "residual of day d uses betas fit through d-1 only (indicators.rolling_ols resid_oos)",
        "Sector SPDR before its inception (XLRE 2015-10, XLC 2018-06) or unknown sector: market-only regression",
        "Score = sum of daily OOS residuals over months t-12..t-2 = sessions in (M_mom, M1]; >=90% of days required",
        "'preserve dollar and estimated beta neutrality': equal-weight stock book is dollar neutral; residual "
        "net market beta (univariate reg_window beta vs SPY, sum w*beta) hedged with an SPY position (beta_hedge)",
    ]
    spec = ("OLS each stock daily return vs market and sector past 252 sessions, rank residual cumulative return "
            "months t-12:t-2; long top decile, short bottom. || Rebalance monthly; preserve dollar and estimated "
            "beta neutrality.")

    def target_weights(self, ctx, syms):
        p = self.params
        W, M = int(p["reg_window"]), int(p["mom_m"])
        cal = ctx.calendar()
        me = month_end_sessions(cal, ctx.session_date(), M)
        if len(me) < M + 1:
            return None
        form_days = n_sessions_between(cal, me[M], me[1]) - 1
        n = n_sessions_between(cal, me[M], ctx.session_date()) + W + 2
        etfs = sorted({e for e in (sector_etf(self.sm.sector(s), ctx.session_date()) for s in syms) if e})
        px = close_panel(ctx, list(syms) + ["SPY"] + etfs, n)
        if px.empty or "SPY" not in px:
            return None
        r = daily_returns(px)
        fwin = (r.index > me[M]) & (r.index <= me[1])
        score, beta = {}, {}
        for s in syms:
            if s not in r:
                continue
            e = sector_etf(self.sm.sector(s), ctx.session_date())
            X = r[["SPY"] + ([e] if e and e in r else [])]
            o = rolling_ols(r[s], X, W)
            res = o["resid_oos"][fwin]
            if res.notna().sum() < 0.9 * max(form_days, 1):
                continue
            score[s] = float(res.sum())
            beta[s] = ols_beta(r[s].iloc[-W:], r["SPY"].iloc[-W:], int(0.9 * W))
        if not score:
            return None
        longs, shorts = self.select(pd.Series(score), q_of(p["decile"]), True)
        w = self.weights_from(ctx, longs, shorts)
        if p["beta_hedge"]:
            nb = sum(v * beta.get(s, float("nan")) for s, v in w.items() if beta.get(s) == beta.get(s))
            if abs(nb) > 1e-6:
                w["SPY"] = w.get("SPY", 0.0) - nb
        return w


# =====================================================================================
# S091 - S100 fundamentals / risk factors
# =====================================================================================
FY_ANCHORS = ("revenue_fy", "net_income_fy")


@register
class S091(_XSBase):
    id = "S091"
    name = "Value factor (book-to-market)"
    params = {"decile": 0.10, "frequency": "annual"}
    grid = {"decile": [0.10, 0.20], "frequency": ["annual", "semiannual"]}
    fund_fields = ("stockholders_equity", "shares_outstanding") + FY_ANCHORS
    fund_bootstrap_months = 18
    requires = _common_requires(fund=True)
    deviations = COMMON_XS_DEVIATIONS + [
        "Book equity: Fama-French BE = SE + deferred taxes/ITC - preferred stock, but the data layer's XBRL map "
        "only has StockholdersEquity -> BE = stockholders_equity at the fiscal-year end (no DT/ITC, no preferred "
        "adjustment); BE <= 0 excluded",
        "Fiscal year end = period_end of the latest revenue_fy/net_income_fy known at the signal with period_end "
        "<= the end of the market-equity month (December 31 of t-1 for June formation)",
        "ME(December t-1) = raw close on the last December session x shares_outstanding known at that close "
        "(split-adjusted to that date)",
        "'within NYSE size breakpoints': NYSE median size split on June ME, B/M decile breakpoints from NYSE names "
        "inside each size group (Fama-French 1993 style); long top / short bottom bucket of both groups",
        "semiannual: additional formation at December end with ME lagged 6 months (June) and FY end <= that June",
        "Index exits between annual rebalances are held (delistings handled by engine corporate actions)",
    ]
    spec = ("At June end use latest filing-available prior fiscal-year book equity / December previous year market "
            "equity; within NYSE size breakpoints long top value decile, short bottom. || Rebalance annually in "
            "July; track delists; release-lagged filings.")

    def is_rebalance(self, ctx):
        m = ctx.session_date().month
        return m == 6 or (self.params["frequency"] == "semiannual" and m == 12)

    def target_weights(self, ctx, syms):
        cal = ctx.calendar()
        me = month_end_sessions(cal, ctx.session_date(), 6)
        if len(me) < 7:
            return None
        me_date = me[6]
        me_close = cal.open_close(me_date)[1]
        bm, size = {}, {}
        for s in syms:
            fy = self.fh.fy_end(s, ctx.now, FY_ANCHORS, not_after=me_date + pd.offsets.MonthEnd(0))
            if fy is None or (me_date - fy).days > 550:
                continue
            be = self.fh.value_at(s, "stockholders_equity", fy, ctx.now, tol_days=10)
            if not (be == be and be > 0):
                continue
            me_ = market_cap(ctx, self.fh, s, me_date, cutoff=me_close)
            sz = market_cap(ctx, self.fh, s)
            if me_ == me_ and me_ > 0 and sz == sz and sz > 0:
                bm[s] = be / me_
                size[s] = sz
        if not bm:
            return None
        nyse = nyse_mask(ctx, self.sm, list(bm))
        longs, shorts = self.select_in_size_buckets(ctx, pd.Series(bm), pd.Series(size), q_of(self.params["decile"]),
                                                    True, nyse)
        return self.weights_from(ctx, longs, shorts)


QUARTER_END_MONTHS = (3, 6, 9, 12)


@register
class S092(_XSBase):
    id = "S092"
    name = "Earnings yield value"
    params = {"earnings": "TTM", "top": 0.20, "exclude_negative": True}
    grid = {"earnings": ["TTM", "FY"], "top": [0.10, 0.20], "exclude_negative": [True, False]}
    fund_fields = ("net_income_fy", "net_income_q", "net_income_ytd", "shares_outstanding")
    fund_bootstrap_months = 27
    requires = _common_requires(fund=True)
    deviations = COMMON_XS_DEVIATIONS + [
        "Quarter end = last NYSE session of Mar/Jun/Sep/Dec",
        "Reported earnings = net_income (NetIncomeLoss). TTM = latest FY + current YTD - prior-year same YTD "
        "(10-Qs do not file Q4); falls back to the latest FY if no comparable YTD is known; FY older than 550 "
        "days -> excluded",
        "Market cap = raw close x latest filed shares_outstanding (split-adjusted)",
        "exclude_negative=True: names with earnings <= 0 removed before ranking ('among positive earners')",
    ]
    spec = ("At each quarter end rank trailing-12m reported earnings / current market cap, only filings public "
            "before signal; long top quintile, short bottom among positive earners. || Rebalance quarterly.")

    def is_rebalance(self, ctx):
        return ctx.session_date().month in QUARTER_END_MONTHS

    def target_weights(self, ctx, syms):
        p = self.params
        ey = {}
        for s in syms:
            if p["earnings"] == "TTM":
                e = self.fh.ttm(s, "net_income", ctx.now)
            else:
                e, pe, _ = self.fh.latest(s, "net_income_fy", ctx.now)
            if not e == e or (p["exclude_negative"] and e <= 0):
                continue
            mc = market_cap(ctx, self.fh, s)
            if mc == mc and mc > 0:
                ey[s] = e / mc
        if not ey:
            return None
        longs, shorts = self.select(pd.Series(ey), q_of(p["top"]), True)
        return self.weights_from(ctx, longs, shorts)


@register
class S093(_XSBase):
    id = "S093"
    name = "Free-cash-flow yield value"
    params = {"rank": 0.20, "lag": "exact"}
    grid = {"rank": [0.10, 0.20], "lag": ["exact", "30d"]}
    fund_fields = ("operating_cash_flow_fy", "operating_cash_flow_q", "operating_cash_flow_ytd", "capex_fy",
                   "capex_q", "capex_ytd", "long_term_debt", "cash", "shares_outstanding")
    fund_bootstrap_months = 27
    requires = _common_requires(fund=True)
    deviations = COMMON_XS_DEVIATIONS + [
        "Quarter end = last NYSE session of Mar/Jun/Sep/Dec",
        "FCF TTM = TTM(operating_cash_flow) - TTM(capex) (YTD-based TTM; capex = PaymentsToAcquirePPE)",
        "EV = market cap + long_term_debt - cash (short-term debt, preferred and minority interest are not in the "
        "data layer's XBRL map); missing debt counted as 0, missing cash as 0",
        "EV <= 0 excluded; negative FCF kept and ranked (falls into the short side) - explicit handling",
        "lag='30d': only filings with known_at <= signal - 30 days (all fundamentals incl. shares)",
    ]
    spec = ("Rank (cash flow from operations - capital expenditures TTM)/enterprise value (EV positive), with "
            "publication lag; long top quintile, short bottom. || Quarterly rebalance; handle negative FCF and EV "
            "explicitly.")

    def is_rebalance(self, ctx):
        return ctx.session_date().month in QUARTER_END_MONTHS

    def target_weights(self, ctx, syms):
        cut = ctx.now - (pd.Timedelta(days=30) if self.params["lag"] == "30d" else pd.Timedelta(0))
        y = {}
        for s in syms:
            cfo = self.fh.ttm(s, "operating_cash_flow", cut)
            capex = self.fh.ttm(s, "capex", cut)
            if not (cfo == cfo and capex == capex):
                continue
            mc = market_cap(ctx, self.fh, s, cutoff=cut)
            if not (mc == mc and mc > 0):
                continue
            debt, _, _ = self.fh.latest(s, "long_term_debt", cut)
            cash, _, _ = self.fh.latest(s, "cash", cut)
            ev = mc + (debt if debt == debt else 0.0) - (cash if cash == cash else 0.0)
            if ev <= 0:
                continue
            y[s] = (cfo - abs(capex)) / ev
        if not y:
            return None
        longs, shorts = self.select(pd.Series(y), q_of(self.params["rank"]), True)
        return self.weights_from(ctx, longs, shorts)


@register
class S094(_XSBase):
    id = "S094"
    name = "Gross profitability factor"
    params = {"rank": 0.20, "update": "annual"}
    grid = {"rank": [0.10, 0.20], "update": ["annual", "quarterly"]}
    fund_fields = ("gross_profit_fy", "gross_profit_q", "gross_profit_ytd", "revenue_fy", "revenue_q",
                   "revenue_ytd", "cost_of_revenue_fy", "cost_of_revenue_q", "cost_of_revenue_ytd",
                   "total_assets", "shares_outstanding", "net_income_fy")
    fund_bootstrap_months = 27
    requires = _common_requires(fund=True)
    deviations = COMMON_XS_DEVIATIONS + [
        "annual: at June end GP/A = gross_profit_fy (else revenue_fy - cost_of_revenue_fy) / total_assets at "
        "the same fiscal-year end, latest annual report known at the signal",
        "quarterly: rebalanced at quarter ends with TTM gross profit / latest total_assets",
        "Size buckets = NYSE median split on current market cap; quintile/decile breakpoints from NYSE names "
        "within each size group",
        "'monthly maintenance' = other month ends: names that left the PIT universe are dropped and the remaining "
        "basket is re-equal-weighted (no new names)",
    ]
    spec = ("Gross profits divided by assets from latest public annual report; within size buckets long top "
            "quintile, short bottom, enter next open. || Annual June rebalance and monthly maintenance.")

    def is_rebalance(self, ctx):
        m = ctx.session_date().month
        return m == 6 if self.params["update"] == "annual" else m in QUARTER_END_MONTHS

    def _gp(self, s, cut):
        if self.params["update"] == "annual":
            fy = self.fh.fy_end(s, cut, ("gross_profit_fy", "revenue_fy", "net_income_fy"))
            if fy is None:
                return float("nan")
            gp = self.fh.value_at(s, "gross_profit_fy", fy, cut, 10)
            if not gp == gp:
                gp = self.fh.value_at(s, "revenue_fy", fy, cut, 10) - self.fh.value_at(s, "cost_of_revenue_fy", fy,
                                                                                       cut, 10)
            ta = self.fh.value_at(s, "total_assets", fy, cut, 10)
        else:
            gp = self.fh.ttm(s, "gross_profit", cut)
            if not gp == gp:
                gp = self.fh.ttm(s, "revenue", cut) - self.fh.ttm(s, "cost_of_revenue", cut)
            ta, _, _ = self.fh.latest(s, "total_assets", cut)
        return gp / ta if ta == ta and ta > 0 and gp == gp else float("nan")

    def target_weights(self, ctx, syms):
        gpa, size = {}, {}
        for s in syms:
            v = self._gp(s, ctx.now)
            mc = market_cap(ctx, self.fh, s)
            if v == v and mc == mc and mc > 0:
                gpa[s], size[s] = v, mc
        if not gpa:
            return None
        nyse = nyse_mask(ctx, self.sm, list(gpa))
        longs, shorts = self.select_in_size_buckets(ctx, pd.Series(gpa), pd.Series(size), q_of(self.params["rank"]),
                                                    True, nyse)
        return self.weights_from(ctx, longs, shorts)

    def maintenance_weights(self, ctx, syms):
        if not self.book:
            return None
        u = set(syms)
        return self.weights_from(ctx, [s for s in self.book["long"] if s in u],
                                 [s for s in self.book["short"] if s in u])


@register
class S095(_XSBase):
    id = "S095"
    name = "Quality minus junk composite"
    params = {"metric_weights": "equal", "top": 0.10, "winsorize": 0.01}
    grid = {"metric_weights": ["equal", "weighted"], "top": [0.10, 0.20], "winsorize": [0.01, 0.05]}
    fund_fields = ("net_income_fy", "net_income_q", "net_income_ytd", "gross_profit_fy", "gross_profit_q",
                   "gross_profit_ytd", "revenue_fy", "revenue_q", "revenue_ytd", "cost_of_revenue_fy",
                   "cost_of_revenue_q", "cost_of_revenue_ytd", "stockholders_equity", "total_assets",
                   "long_term_debt")
    fund_bootstrap_months = 75
    long_gross = 0.60
    short_gross = 0.60
    requires = _common_requires(fund=True)
    WEIGHTS = {"equal": {"roe": 1, "gpa": 1, "sg5": 1, "lev": 1, "evol": 1},
               "weighted": {"roe": 1 / 6, "gpa": 1 / 6, "sg5": 1 / 3, "lev": 1 / 6, "evol": 1 / 6}}
    deviations = COMMON_XS_DEVIATIONS + [
        "Book 60%/60% long/short as stated in the strategy text (overrides CROSS_SECTION 50/50); name cap 2% and "
        "sector caps 10% per side still applied",
        "Industry cross-section = GICS sector (sub-industries too thin for z-scores in the S&P 500); group_zscore "
        "requires >= 2 names",
        "ROE = TTM net income / latest stockholders_equity; GP/A = TTM gross profit (or revenue - cost of revenue) "
        "/ latest total_assets; 5y sales growth = (revenue_fy / revenue_fy five fiscal years earlier)^(1/5)-1; "
        "debt/assets = long_term_debt / total_assets (short-term debt not in data map); earnings volatility = "
        "stdev of fiscal-year ROE over up to the last 5 fiscal years (>= 3 required)",
        "'weighted' = AQR QMJ pillars: profitability (ROE, GP/A) 1/3, growth 1/3, safety (-D/A, -EVOL) 1/3",
        "Winsorize = cross-sectional clip at (p, 1-p) per metric before z-scoring",
        "Sector neutrality = deciles formed within each GICS sector (group_quantile_bucket)",
    ]
    spec = ("For each industry cross-section z-score ROE, gross profit/assets, 5y sales growth, -debt/assets, "
            "-earnings volatility; mean available z; long top decile and short bottom. || Monthly rebalance; 60/60 "
            "long/short; sector neutrality.")

    def _metrics(self, s, cut) -> dict:
        fh = self.fh
        out = {}
        se, _, _ = fh.latest(s, "stockholders_equity", cut)
        ta, _, _ = fh.latest(s, "total_assets", cut)
        ni = fh.ttm(s, "net_income", cut)
        if se == se and se > 0 and ni == ni:
            out["roe"] = ni / se
        gp = fh.ttm(s, "gross_profit", cut)
        if not gp == gp:
            gp = fh.ttm(s, "revenue", cut) - fh.ttm(s, "cost_of_revenue", cut)
        if ta == ta and ta > 0 and gp == gp:
            out["gpa"] = gp / ta
        rev = fh.fy_values(s, "revenue", cut)
        if len(rev):
            E = rev.index.max()
            old = rev[np.abs((rev.index - (E - pd.DateOffset(years=5))).days) <= 20]
            if len(old) and old.iloc[-1] > 0 and rev[E] > 0:
                out["sg5"] = (rev[E] / old.iloc[-1]) ** 0.2 - 1.0
        debt, _, _ = fh.latest(s, "long_term_debt", cut)
        if ta == ta and ta > 0 and debt == debt:
            out["lev"] = -debt / ta
        nfy = fh.fy_values(s, "net_income", cut)
        roes = []
        for E in list(nfy.index[-5:]):
            b = fh.value_at(s, "stockholders_equity", E, cut, 10)
            if b == b and b > 0:
                roes.append(nfy[E] / b)
        if len(roes) >= 3:
            out["evol"] = -float(np.std(roes, ddof=1))
        return out

    def target_weights(self, ctx, syms):
        p = self.params
        m = pd.DataFrame({s: self._metrics(s, ctx.now) for s in syms}).T
        if m.empty:
            return None
        sec = pd.Series(self.sm.sectors(m.index))
        z = pd.DataFrame(index=m.index)
        for c in m.columns:
            col = m[c].dropna()
            if len(col) < 2:
                continue
            col = xs.winsorize(col, p["winsorize"], 1 - p["winsorize"])
            z[c] = xs.group_zscore(col, sec.reindex(col.index)).reindex(m.index)
        if z.empty:
            return None
        wts = pd.Series(self.WEIGHTS[p["metric_weights"]]).reindex(z.columns).fillna(0.0)
        avail = z.notna().astype(float)
        num = (z.fillna(0.0) * wts).sum(axis=1)
        den = (avail * wts).sum(axis=1)
        comp = (num / den.where(den > 0)).dropna()
        longs, shorts = self.select(comp, q_of(p["top"]), True, groups=sec.reindex(comp.index))
        return self.weights_from(ctx, longs, shorts)


@register
class S096(_XSBase):
    id = "S096"
    name = "Low-volatility equity factor"
    params = {"vol_lookback": 252, "decile": 0.10, "sector_neutral": False}
    grid = {"vol_lookback": [63, 126, 252], "decile": [0.10, 0.20]}
    fund_fields = ("shares_outstanding",)
    fund_bootstrap_months = 0
    requires = _common_requires(fund=True)
    deviations = COMMON_XS_DEVIATIONS + [
        "Realized vol = sample stdev of daily total returns over the last vol_lookback sessions (all required)",
        "NYSE size-bucket neutral = NYSE median split on market cap (raw close x filed shares, split-adjusted); "
        "vol deciles (NYSE breakpoints) inside each size group; names without shares excluded",
        "Optional sector neutrality (param sector_neutral, default off) = deciles within sector inside each size group",
    ]
    spec = ("At each month end estimate 252d daily realized volatility; long bottom decile stocks, short top decile "
            "stocks, NYSE size-bucket neutral. || Monthly rebalance; sector neutral optional.")

    def target_weights(self, ctx, syms):
        p = self.params
        n = int(p["vol_lookback"])
        px = close_panel(ctx, syms, n + 1)
        if px.empty:
            return None
        r = daily_returns(px).iloc[-n:]
        vol = r.std(ddof=1).where(r.notna().sum() >= n)
        size = pd.Series({s: market_cap(ctx, self.fh, s) for s in vol.dropna().index}, dtype=float)
        size = size[size > 0]
        if size.empty:
            return None
        q = q_of(p["decile"])
        nyse = nyse_mask(ctx, self.sm, list(size.index))
        if not p["sector_neutral"]:
            return self.weights_from(ctx, *self.select_in_size_buckets(ctx, vol, size, q, False, nyse))
        sec = pd.Series(self.sm.sectors(size.index))
        sb = xs.size_buckets(size, nyse.reindex(size.index), 2)
        longs, shorts = [], []
        for g in sorted(sb.dropna().unique()):
            m = sb.index[sb == g]
            l, s = self.select(vol.reindex(m), q, False, groups=sec.reindex(m))
            longs += l
            shorts += s
        return self.weights_from(ctx, longs, shorts)


@register
class S097(_XSBase):
    id = "S097"
    name = "Low-beta defensive factor"
    params = {"beta_lookback": 252, "shrink": 0.0}
    grid = {"beta_lookback": [126, 252], "shrink": [0.0, 0.5]}
    requires = _common_requires(spy=True)
    deviations = COMMON_XS_DEVIATIONS + [
        "Beta = OLS slope of daily total returns on SPY over the last beta_lookback sessions (through the signal "
        "close, >= 90% of days); shrink s: beta = (1-s)*beta + s*1",
        "'equal beta gross exposure': equal dollar weights within each side; side gross set so that "
        "sum(w*beta) long = sum(|w|*beta) short, by scaling DOWN the side with the larger beta exposure "
        "(leverage constrained to the CROSS_SECTION 50% per side, name/sector caps applied first)",
    ]
    spec = ("Compute trailing 252d OLS beta to SPY; long lowest beta quintile, short highest beta quintile; target "
            "equal beta gross exposure rather than equal dollars. || Monthly rebalance; beta neutral, leverage and "
            "borrowing constrained.")

    def target_weights(self, ctx, syms):
        L = int(self.params["beta_lookback"])
        s_ = float(self.params["shrink"])
        px = close_panel(ctx, list(syms) + ["SPY"], L + 1)
        if px.empty or "SPY" not in px:
            return None
        r = daily_returns(px).iloc[-L:]
        betas = {}
        for s in syms:
            if s in r:
                b = ols_beta(r[s], r["SPY"], int(0.9 * L))
                if b == b:
                    betas[s] = (1 - s_) * b + s_ * 1.0
        if not betas:
            return None
        bs = pd.Series(betas)
        longs, shorts = self.select(bs, 5, False)
        w = self.weights_from(ctx, longs, shorts)
        bl = sum(v * bs[s] for s, v in w.items() if v > 0)
        bsh = sum(-v * bs[s] for s, v in w.items() if v < 0)
        if bl > 0 and bsh > 0:
            if bl > bsh:
                w = {s: (v * bsh / bl if v > 0 else v) for s, v in w.items()}
            else:
                w = {s: (v * bl / bsh if v < 0 else v) for s, v in w.items()}
        return w


@register
class S098(_XSBase):
    id = "S098"
    name = "Shareholder yield"
    params = {"issuance_window": "TTM", "rank": 0.20}
    grid = {"issuance_window": ["TTM", "3y"], "rank": [0.10, 0.20]}
    fund_fields = ("dividends_paid_fy", "dividends_paid_q", "dividends_paid_ytd", "shares_outstanding")
    fund_bootstrap_months = 40
    requires = _common_requires(fund=True)
    deviations = COMMON_XS_DEVIATIONS + [
        "Quarter end = last NYSE session of Mar/Jun/Sep/Dec",
        "Dividend yield = TTM dividends_paid (cash flow statement, YTD-based TTM) / market cap",
        "Net repurchase yield = -(net issuance) measured from split-adjusted shares_outstanding: TTM: "
        "-(sh_now/sh_1y - 1); 3y: -((sh_now/sh_3y)^(1/3) - 1); share counts as filed and known 1y/3y before the "
        "signal; issuance proceeds are not in the data layer's XBRL map, so the share-count measure is used",
        "Missing dividends counted as 0 when shares data exist; names without both measures excluded",
    ]
    spec = ("Quarterly rank cash dividends yield + net share repurchase yield (net issuance negative); public "
            "filing only; long top quintile/short bottom. || Rebalance quarterly; dividend adjusted total return.")

    def is_rebalance(self, ctx):
        return ctx.session_date().month in QUARTER_END_MONTHS

    def target_weights(self, ctx, syms):
        years = 1 if self.params["issuance_window"] == "TTM" else 3
        y = {}
        for s in syms:
            sh, _, kat = self.fh.latest(s, "shares_outstanding", ctx.now)
            old_cut = ctx.now - pd.DateOffset(years=years)
            sh0, _, kat0 = self.fh.latest(s, "shares_outstanding", old_cut)
            mc = market_cap(ctx, self.fh, s)
            if not (sh == sh and sh0 == sh0 and sh0 > 0 and mc == mc and mc > 0):
                continue
            sh0_adj = sh0 * split_factor(ctx, s, kat0, pd.Timestamp(kat).tz_convert(ET).tz_localize(None))
            nr = -((sh / sh0_adj) ** (1.0 / years) - 1.0)
            dv = self.fh.ttm(s, "dividends_paid", ctx.now)
            dy = abs(dv) / mc if dv == dv else 0.0
            y[s] = dy + nr
        if not y:
            return None
        longs, shorts = self.select(pd.Series(y), q_of(self.params["rank"]), True)
        return self.weights_from(ctx, longs, shorts)


@register
class S099(_XSBase):
    id = "S099"
    name = "Investment (conservative minus aggressive)"
    params = {"growth_years": 1, "rank": 0.20}
    grid = {"growth_years": [1, 2], "rank": [0.10, 0.20]}
    fund_fields = ("total_assets",) + FY_ANCHORS
    fund_bootstrap_months = 40
    requires = _common_requires(fund=True)
    deviations = COMMON_XS_DEVIATIONS + [
        "A_t = total_assets at the latest fiscal-year end known at June end (FY end = period_end of latest "
        "revenue_fy/net_income_fy); A_{t-k} = total_assets at the fiscal-year end k years earlier (+/-20 days); "
        "2y window = A_t/A_{t-2} - 1",
        "Index exits between annual rebalances are held",
    ]
    spec = ("At June end rank fiscal year asset growth=(A_t-A_{t-1})/A_{t-1} using public filings; long bottom "
            "quintile, short top quintile. || Rebalance following first July open annually.")

    def is_rebalance(self, ctx):
        return ctx.session_date().month == 6

    def target_weights(self, ctx, syms):
        k = int(self.params["growth_years"])
        g = {}
        for s in syms:
            E = self.fh.fy_end(s, ctx.now, FY_ANCHORS)
            if E is None:
                continue
            a1 = self.fh.value_at(s, "total_assets", E, ctx.now, 10)
            a0 = self.fh.value_at(s, "total_assets", E - pd.DateOffset(years=k), ctx.now, 20)
            if a1 == a1 and a0 == a0 and a0 > 0:
                g[s] = a1 / a0 - 1.0
        if not g:
            return None
        longs, shorts = self.select(pd.Series(g), q_of(self.params["rank"]), False)
        return self.weights_from(ctx, longs, shorts)


@register
class S100(_XSBase):
    id = "S100"
    name = "Multifactor value-quality-momentum"
    params = {"weights": "equal", "rank": 0.10, "turnover_cap": None}
    grid = {"weights": ["equal", "0.25/0.25/0.5"], "rank": [0.10, 0.20]}
    fund_fields = ("stockholders_equity", "total_assets", "shares_outstanding", "net_income_fy", "net_income_q",
                   "net_income_ytd")
    fund_bootstrap_months = 27
    requires = _common_requires(fund=True)
    W = {"equal": (1 / 3, 1 / 3, 1 / 3), "0.25/0.25/0.5": (0.25, 0.25, 0.5)}
    deviations = COMMON_XS_DEVIATIONS + [
        "B/M = latest stockholders_equity / current market cap (monthly, no FF June lag); ROA = TTM net income / "
        "latest total_assets; momentum 12-2 = P(M1)/P(M12)-1",
        "Industry z-scores = group_zscore within GICS sector (>= 2 names); names need all three z",
        "Weights '0.25/0.25/0.5' read in the listed order (V, Q, M): momentum gets 0.5",
        "Optional turnover cap (param turnover_cap, e.g. 0.15; default None): if one-sided turnover "
        "sum|w_new - w_old|/2 exceeds the cap, the move is scaled to w_old + (cap/turnover)(w_new - w_old)",
    ]
    spec = ("Monthly industry z-score book-to-market, ROA, and 12-2 momentum; combined score=(zV+zQ+zM)/3; long top "
            "decile, short bottom. || Rebalance monthly with 15% monthly turnover optional cap.")

    def target_weights(self, ctx, syms):
        px, me = self.month_end_prices(ctx, syms, 12)
        if px.empty:
            return None
        mom = self.at(px, me[1]) / self.at(px, me[12]) - 1.0
        rows = {}
        for s in syms:
            se, _, _ = self.fh.latest(s, "stockholders_equity", ctx.now)
            ta, _, _ = self.fh.latest(s, "total_assets", ctx.now)
            ni = self.fh.ttm(s, "net_income", ctx.now)
            mc = market_cap(ctx, self.fh, s)
            rows[s] = {"bm": se / mc if se == se and mc == mc and mc > 0 else np.nan,
                       "roa": ni / ta if ni == ni and ta == ta and ta > 0 else np.nan,
                       "mom": mom.get(s, np.nan)}
        f = pd.DataFrame(rows).T.astype(float)
        sec = pd.Series(self.sm.sectors(f.index))
        z = pd.DataFrame({c: xs.group_zscore(f[c], sec) for c in ("bm", "roa", "mom")})
        wv, wq, wm = self.W[self.params["weights"]]
        comp = (wv * z["bm"] + wq * z["roa"] + wm * z["mom"]).dropna()
        longs, shorts = self.select(comp, q_of(self.params["rank"]), True)
        w = self.weights_from(ctx, longs, shorts)
        cap = self.params["turnover_cap"]
        if cap is not None and self.last_weights:
            old = self.last_weights
            keys = set(w) | set(old)
            to = sum(abs(w.get(k, 0.0) - old.get(k, 0.0)) for k in keys) / 2.0
            if to > cap > 0:
                lam = cap / to
                w = {k: old.get(k, 0.0) + lam * (w.get(k, 0.0) - old.get(k, 0.0)) for k in keys}
                w = {k: v for k, v in w.items() if abs(v) > 1e-12}
        return w
