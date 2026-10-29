"""Shared machinery for the intraday family (S001-S030, S116).

Everything here is causal: it only consumes bars returned by the strategy
Context (completed bars, PIT) and keeps per-session state that resets at
every session start.

* :class:`Tape` - today's completed base bars for one symbol (RTH, and PRE
  when the strategy loads extended hours) kept in numpy arrays, updated
  incrementally each step with ``ctx.history`` (engine fast path), with a
  streaming session VWAP (``indicators.streaming.StreamingSessionVWAP``,
  typical price, resets each session) and k-minute buckets aligned to the
  09:30 open (``Tape.agg``) for 1m strategies whose rules speak of "5m"
  closes.
* :class:`BarStream` - a streaming indicator (RSI / EMA / ATR from
  ``indicators.streaming``) on base or k-minute bars that spans sessions:
  warmed once per session from prior completed bars (``adjusted=True`` as of
  now, i.e. in today's share units) and then fed today's raw bars.
* :func:`tod_profile` - expected volume per time-of-day slot = median of the
  same ET slot over the PREVIOUS ``lookback`` sessions (same convention as
  ``indicators.intraday.time_of_day_profile``; see its docstring). The
  indicator only returns the profile for rows that already exist, so a
  "profile for the session that is starting" is computed here from the same
  definition (consistency test in tests/strategies/intraday).
* :class:`IntradayStrategy` - base class: universe (static + PIT portal
  universe), session timing incl. early closes, DAY_RISK / DAY_NOTIONAL
  sizing via ``atlasbt.risk.sizing``, brackets, R-multiple targets placed from
  the ACTUAL entry fill, refreshed limit exits, condition exits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.clock import et
from ..core.types import Order, OrderStatus, OrderType, TimeInForce
from ..engine.strategy import Requirement, Strategy
from ..engine.timeutil import canonical_freq, freq_to_timedelta
from ..indicators import intraday as iind
from ..indicators import streaming as stm
from ..indicators.intraday import adv as ind_adv
from ..indicators.moving import sma as ind_sma
from ..risk.sizing import size_day_notional, size_day_risk

NS_MIN = 60_000_000_000
OPEN_MIN = 9 * 60 + 30          # 09:30 ET in minutes

# ----------------------------------------------------------------- universes (STRATEGY_GUIDE §3)
ETF_CORE = ("SPY", "QQQ", "IWM", "DIA")
MEGACAP_UNIVERSE = "SP500_TOP50_DV"     # MEGACAP_STK = PIT top-N S&P 500 by 20d dollar volume

#: Interpretations shared by every strategy of the family (declared once, module-wide).
MODULE_DEVIATIONS = (
    "Early closes: '15:55 exit' = 5 min before the close and 'no new entries after 15:15' = 45 min before the "
    "close (STRATEGY_GUIDE §1). The engine shifts its flat_by / entry cutoff with the actual close, and the module "
    "also flattens at close-5min on early-close sessions (only positions without a working engine flat_by order) "
    "and applies the shifted cutoff in its own sizing.",
    "Universe: 'STK/ETF' intraday = ETF_CORE + MEGACAP_STK (portal 'SP500_TOP50_DV', PIT); 'STK' = MEGACAP_STK; "
    "'ETF' intraday = ETF_CORE (STRATEGY_GUIDE §3; the guide names no intraday ETF set, the most liquid ETFs are "
    "used).",
    "Session VWAP = cumulative volume-weighted typical price (H+L+C)/3 of today's RTH bars (indicators "
    "convention); intraday sd = volume-weighted population sd.",
    "'5m' bars inside a 1m strategy = 5-minute buckets aligned to 09:30 built from completed 1m bars; a bucket "
    "is used only after its last minute has completed.",
    "Indicators on intraday bars that need warm-up (RSI, EMA, ATR) span sessions (prior sessions' bars, "
    "split/dividend-adjusted as of today) then continue on today's raw bars; ATR = Wilder ATR(14), the sheet "
    "gives no period.",
    "Time-of-day RVOL medians: same ET slot over the previous 20 sessions (STRATEGY_GUIDE §2.8), sessions "
    "lacking the slot (early closes) skipped, >= 15 valid sessions required; volumes split-adjusted.",
    "When the sheet does not say 'first' or 'one entry/day' a strategy may re-enter after it is flat (one "
    "position per symbol at a time); 'first ...' and 'one entry/day' => at most one entry order per symbol per "
    "session, including orders that did not fill (STRATEGY_GUIDE §2.5).",
    "Several conditions in one sentence are read as a conjunction evaluated on the same completed bar ('first "
    "bar where all hold') unless the sheet orders them ('then', 'after', 'sequence ordered').",
    "R-multiple targets: limit at fill +/- R x |fill - initial stop| using the actual (average) entry fill; it "
    "is submitted when the fill is known, so it can first fill on the bar after the entry bar.",
    "Prices compared with prior-session values (prior close/high, gaps) use daily bars adjusted as of today "
    "(today's share units); today's intraday bars are raw.",
)


#: engine session codes (history_arrays 'session': 0 RTH, 1 PRE, 2 POST, 3 DAY) -> labels
_SESSION_NAMES = np.array(["RTH", "PRE", "POST", "DAY"], dtype=object)


def _frame_to_arrays(df) -> dict:
    """ctx.history frame -> the history_arrays dict layout (start ns, OHLCV floats, session codes)."""
    if df is None or len(df) == 0:
        z = np.zeros(0)
        return {"start": np.zeros(0, np.int64), "open": z, "high": z, "low": z, "close": z, "volume": z,
                "session": np.zeros(0, np.int8)}
    idx = df.index
    st = idx.asi8 if getattr(idx, "unit", "ns") == "ns" else pd.DatetimeIndex(idx).as_unit("ns").asi8
    out = {"start": st}
    for k in ("open", "high", "low", "close", "volume"):
        out[k] = df[k].to_numpy(dtype=float)
    codes = {"RTH": 0, "PRE": 1, "POST": 2, "DAY": 3}
    out["session"] = (np.array([codes.get(str(x), 0) for x in df["session"]], np.int8) if "session" in df.columns
                      else np.zeros(len(df), np.int8))
    return out


def _hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def et_minute(t_ns: int) -> int:
    """Minute of day (ET) of a UTC ns timestamp."""
    t = pd.Timestamp(int(t_ns), tz="UTC").tz_convert("America/New_York")
    return t.hour * 60 + t.minute


# =================================================================== Tape
@dataclass
class Buckets:
    start: np.ndarray
    end: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    last: np.ndarray          # index (in the base arrays) of the last base bar of each bucket

    @property
    def n(self) -> int:
        return len(self.c)


class Tape:
    """Today's completed bars for one symbol (RTH arrays + PRE arrays)."""

    def __init__(self, symbol: str, open_ns: int, close_ns: int, day_start_ns: int, bar_ns: int, cap: int = 512):
        self.symbol = symbol
        self.open_ns, self.close_ns, self.day_start_ns, self.bar_ns = open_ns, close_ns, day_start_ns, bar_ns
        self.n = 0
        self.start = np.zeros(cap, np.int64)
        self.o, self.h, self.l, self.c, self.v = (np.zeros(cap) for _ in range(5))
        self.vwap, self.sd, self.cumv = (np.zeros(cap) for _ in range(3))
        self.npre = 0
        self.pstart = np.zeros(cap, np.int64)
        self.po, self.ph, self.pl, self.pc, self.pv = (np.zeros(cap) for _ in range(5))
        self.last_start = -1
        self.updated_at = -1
        self._vw = stm.StreamingSessionVWAP("typical")
        self._agg_cache: dict[int, tuple[int, int, Buckets]] = {}

    # ------------------------------------------------------------- maintenance
    def _grow(self, pre: bool) -> None:
        names = ("pstart", "po", "ph", "pl", "pc", "pv") if pre else \
            ("start", "o", "h", "l", "c", "v", "vwap", "sd", "cumv")
        for nm in names:
            a = getattr(self, nm)
            b = np.zeros(len(a) * 2, a.dtype)
            b[:len(a)] = a
            setattr(self, nm, b)

    def update(self, ctx, data_sessions: Sequence[str]) -> None:
        now = int(ctx.now.value)
        if now == self.updated_at:
            return
        last_end = max(self.last_start + self.bar_ns, self.day_start_ns)
        n = int(max(now - last_end, 0) // self.bar_ns) + 2
        # perf (docs/PERF.md): history_arrays = the same completed raw bars as ctx.history (engine cache
        # views, same PIT fallback) without building a DataFrame per symbol and step
        if hasattr(ctx, "history_arrays"):
            a = ctx.history_arrays(self.symbol, n, adjusted=False, sessions=tuple(data_sessions))
        else:                                   # minimal contexts (tests): same arrays from the frame
            a = _frame_to_arrays(ctx.history(self.symbol, n, adjusted=False, sessions=tuple(data_sessions)))
        self.updated_at = now
        if len(a["close"]) == 0:
            return
        st = a["start"]
        sel = np.nonzero((st >= self.day_start_ns) & (st > self.last_start))[0]
        if sel.size == 0:
            return
        o, h, l, c, v = (a[k] for k in ("open", "high", "low", "close", "volume"))
        rth_only = tuple(data_sessions) == ("RTH",)
        ses = None if rth_only else _SESSION_NAMES[a["session"]]
        for i in sel:
            s = "RTH" if rth_only else ses[i]
            if s == "RTH":
                if self.n >= len(self.c):
                    self._grow(False)
                k = self.n
                self.start[k], self.o[k], self.h[k], self.l[k], self.c[k], self.v[k] = st[i], o[i], h[i], l[i], c[i], v[i]
                vv = v[i] if v[i] == v[i] else 0.0
                self._vw.update(0, h[i], l[i], c[i], vv)
                self.vwap[k], self.sd[k] = self._vw.vwap, self._vw.sd
                self.cumv[k] = (self.cumv[k - 1] if k else 0.0) + vv
                self.n = k + 1
            elif s == "PRE":
                if self.npre >= len(self.pc):
                    self._grow(True)
                k = self.npre
                self.pstart[k], self.po[k], self.ph[k], self.pl[k], self.pc[k], self.pv[k] = st[i], o[i], h[i], l[i], c[i], v[i]
                self.npre = k + 1
        self.last_start = int(st[sel[-1]])

    # ------------------------------------------------------------- queries
    def end(self, i: int) -> int:
        return int(min(self.start[i] + self.bar_ns, self.close_ns))

    def first_open(self) -> float:
        """Opening print: open of the first RTH bar (the 09:30 bar)."""
        return float(self.o[0]) if self.n else math.nan

    def has_open_bar(self) -> bool:
        return self.n > 0 and int(self.start[0]) == self.open_ns

    def idx_end_le(self, t_ns: int) -> int:
        """Index of the last bar whose end <= t (or -1)."""
        ends = np.minimum(self.start[:self.n] + self.bar_ns, self.close_ns)
        return int(np.searchsorted(ends, t_ns, side="right")) - 1

    def window(self, t0_ns: int, t1_ns: int) -> slice:
        """Bars whose START is in [t0, t1)."""
        s = self.start[:self.n]
        return slice(int(np.searchsorted(s, t0_ns, side="left")), int(np.searchsorted(s, t1_ns, side="left")))

    def agg(self, k_min: int) -> Buckets:
        """Complete k-minute buckets aligned to the 09:30 open."""
        key = (self.n, self.updated_at)
        hit = self._agg_cache.get(k_min)
        if hit is not None and hit[:2] == key:
            return hit[2]
        n = self.n
        if n == 0:
            e = np.zeros(0)
            b = Buckets(np.zeros(0, np.int64), np.zeros(0, np.int64), e, e, e, e, e, np.zeros(0, np.int64))
        else:
            kn = k_min * NS_MIN
            bi = (self.start[:n] - self.open_ns) // kn
            ub, first = np.unique(bi, return_index=True)
            last = np.append(first[1:] - 1, n - 1)
            ends = np.minimum(self.open_ns + (ub + 1) * kn, self.close_ns)
            done = ends <= self.updated_at
            b = Buckets(self.open_ns + ub * kn, ends, self.o[first], np.maximum.reduceat(self.h[:n], first),
                        np.minimum.reduceat(self.l[:n], first), self.c[last], np.add.reduceat(self.v[:n], first),
                        last)
            if not done.all():
                m = done
                b = Buckets(b.start[m], b.end[m], b.o[m], b.h[m], b.l[m], b.c[m], b.v[m], b.last[m])
        self._agg_cache[k_min] = (key[0], key[1], b)
        return b

    def pre_window(self, t0_ns: int, t1_ns: int) -> slice:
        s = self.pstart[:self.npre]
        return slice(int(np.searchsorted(s, t0_ns, side="left")), int(np.searchsorted(s, t1_ns, side="left")))


def aggregate_frame(df: pd.DataFrame, k_min: int) -> pd.DataFrame:
    """k-minute buckets (aligned to 09:30 ET, per ET date) of a base-bar frame
    (all rows complete). Used for warm-up history only."""
    if df is None or len(df) == 0 or k_min <= 1:
        return df
    idx = pd.DatetimeIndex(df.index)
    e = idx.tz_convert("America/New_York")
    mod = (e.hour * 60 + e.minute).to_numpy()
    day = e.tz_localize(None).normalize().to_numpy()
    b = (mod - OPEN_MIN) // k_min
    aligned = idx - pd.to_timedelta((mod - OPEN_MIN) % k_min, unit="min")      # bucket start
    g = pd.DataFrame({"d": day, "b": b, "open": df["open"].to_numpy(float), "high": df["high"].to_numpy(float),
                      "low": df["low"].to_numpy(float), "close": df["close"].to_numpy(float),
                      "volume": df["volume"].to_numpy(float), "t": aligned}).groupby(["d", "b"], sort=True)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum(), "t": g["t"].first()})
    out.index = pd.DatetimeIndex(out["t"])
    return out.drop(columns="t")


# =================================================================== streaming indicators across sessions
class BarStream:
    """Streaming indicator on base (k_min=None) or k-minute buckets of today's
    tape, warmed with prior completed bars. ``values[i]`` = indicator after
    today's bar/bucket i (aligned to ``tape`` arrays or ``tape.agg(k)``)."""

    def __init__(self, make: Callable[[], object], feed: Callable, k_min: Optional[int] = None,
                 warm_bars: int = 200):
        self.obj = make()
        self.feed = feed
        self.k_min = k_min
        self.warm_bars = warm_bars
        self.values: list[float] = []
        self._warm = False

    def sync(self, ctx, tape: Tape, base_min: int) -> list[float]:
        if not self._warm:
            self._warm = True
            # exactly `warm_bars` prior-session bars, whenever the stream is first synced (no path dependence)
            nb = self.warm_bars * (self.k_min // base_min if self.k_min else 1)
            hist = ctx.history(tape.symbol, nb + tape.n, adjusted=True, sessions=("RTH",))
            if hist is not None and len(hist):
                hist = hist[pd.DatetimeIndex(hist.index).as_unit("ns").asi8 < tape.open_ns]
                if "session" in hist.columns:
                    hist = hist[hist["session"] == "RTH"]
                hist = hist.iloc[-nb:]
                if self.k_min and self.k_min != base_min:
                    hist = aggregate_frame(hist, self.k_min)
                for r in zip(hist["open"].to_numpy(float), hist["high"].to_numpy(float), hist["low"].to_numpy(float),
                             hist["close"].to_numpy(float), hist["volume"].to_numpy(float)):
                    self.feed(self.obj, *r)
        if self.k_min and self.k_min != base_min:
            b = tape.agg(self.k_min)
            arrs, n = (b.o, b.h, b.l, b.c, b.v), b.n
        else:
            arrs, n = (tape.o, tape.h, tape.l, tape.c, tape.v), tape.n
        for i in range(len(self.values), n):
            self.values.append(float(self.feed(self.obj, *(float(a[i]) for a in arrs))))
        return self.values


def rsi_stream(n: int, k_min: Optional[int] = None) -> BarStream:
    return BarStream(lambda: stm.StreamingRSI(n), lambda ob, o, h, l, c, v: ob.update(c), k_min, warm_bars=max(10 * n, 60))


def ema_stream(n: int, k_min: Optional[int] = None) -> BarStream:
    return BarStream(lambda: stm.StreamingEMA(n), lambda ob, o, h, l, c, v: ob.update(c), k_min, warm_bars=max(10 * n, 60))


def atr_stream(n: int = 14, k_min: Optional[int] = None) -> BarStream:
    return BarStream(lambda: stm.StreamingATR(n), lambda ob, o, h, l, c, v: ob.update(h, l, c), k_min,
                     warm_bars=max(10 * n, 60))


# =================================================================== time-of-day profile
def tod_profile(hist: pd.DataFrame, lookback: int = 20, cumulative: bool = False, min_valid: int = 15,
                k_min: int = 1) -> pd.Series:
    """Expected volume per ET minute-slot (slot = minute of day of the bar
    start) for the session that follows ``hist``: median over the last
    ``lookback`` sessions in ``hist`` (all strictly before today) of the
    same-slot volume (``cumulative``: session-to-date volume through the slot).
    Sessions lacking a slot (early closes / missing bars at the end of the
    session) are skipped for that slot; NaN if fewer than ``min_valid``."""
    if hist is None or len(hist) == 0:
        return pd.Series(dtype=float)
    if "session" in hist.columns:
        hist = hist[hist["session"] == "RTH"]
    if k_min > 1:
        hist = aggregate_frame(hist, k_min)
    key = iind.session_key(pd.DatetimeIndex(hist.index))
    slot = iind.minute_of_day(pd.DatetimeIndex(hist.index))
    piv = pd.DataFrame({"k": key, "s": slot, "v": hist["volume"].to_numpy(float)}).pivot_table(
        index="k", columns="s", values="v", aggfunc="sum").sort_index()
    piv = piv.iloc[-lookback:]
    if cumulative:
        raw_valid = piv.notna()
        later_valid = piv.bfill(axis=1).notna()
        piv = piv.fillna(0.0).cumsum(axis=1).where(raw_valid | later_valid)
    cnt = piv.notna().sum(axis=0)
    med = piv.median(axis=0, skipna=True)
    return med.where(cnt >= min_valid)


# =================================================================== daily info
@dataclass
class DailyInfo:
    ok: bool
    open: np.ndarray = field(default_factory=lambda: np.zeros(0))
    high: np.ndarray = field(default_factory=lambda: np.zeros(0))
    low: np.ndarray = field(default_factory=lambda: np.zeros(0))
    close: np.ndarray = field(default_factory=lambda: np.zeros(0))
    volume: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def prior_close(self) -> float:
        return float(self.close[-1]) if self.ok and len(self.close) else math.nan

    def adv(self, n: int = 20) -> float:
        if not self.ok or len(self.volume) < n:
            return math.nan
        return float(ind_adv(pd.Series(self.volume), n).iloc[-1])

    def ret(self, n: int = 20) -> float:
        if not self.ok or len(self.close) < n + 1:
            return math.nan
        return float(self.close[-1] / self.close[-1 - n] - 1.0)

    def sma(self, n: int = 200) -> float:
        if not self.ok or len(self.close) < n:
            return math.nan
        return float(ind_sma(self.close, n)[-1])

    def max_high(self, n: int) -> float:
        if not self.ok or len(self.high) < n:
            return math.nan
        return float(np.max(self.high[-n:]))


# =================================================================== entry bookkeeping
@dataclass
class EntryInfo:
    symbol: str
    side: int
    stop: float
    target_r: Optional[float]
    signal_ts: pd.Timestamp
    filled: float = 0.0
    cost: float = 0.0
    target_id: Optional[int] = None


# =================================================================== base strategy
class IntradayStrategy(Strategy):
    """Base class: per-session state, tapes, sizing, order helpers."""

    sizing = "DAY_RISK"
    #: extra non-traded symbols (benchmarks) the strategy reads
    benchmarks: tuple = ()
    #: daily bars needed per symbol (prior sessions)
    daily_n: int = 25

    def __init__(self, **params):
        super().__init__(**params)
        self.requires = list(self.build_requires())
        self._session = None
        self._entries: dict[int, EntryInfo] = {}

    # ------------------------------------------------------------- declarations
    def build_requires(self) -> list[Requirement]:
        return list(type(self).requires)

    @property
    def base_min(self) -> int:
        return int(freq_to_timedelta(self.bar_freq) / pd.Timedelta(minutes=1))

    def data_sessions(self) -> tuple[str, ...]:
        """Same derivation as the engine (timeline sessions + base-frequency bar requirements)."""
        base = canonical_freq(self.bar_freq)
        ds = set(self.sessions)
        for r in self.requires:
            if r.kind == "bars" and (r.freq is None or canonical_freq(r.freq) == base):
                ds |= set(r.sessions)
        order = {"PRE": 0, "RTH": 1, "POST": 2}
        return tuple(sorted((s for s in ds if s in order), key=order.get)) or ("RTH",)

    def symbols(self, ctx) -> list[str]:
        out = list(self.universe)
        if self.universe_name:
            try:
                out += [s for s in ctx.portal().universe(self.universe_name) if s not in out]
            except Exception:  # pragma: no cover - portal without universes
                pass
        return out

    # ------------------------------------------------------------- session lifecycle
    def on_session_start(self, ctx) -> None:
        s = ctx.session_date()
        self._session = s
        o, c = ctx.calendar().open_close(s)
        self.open_ns, self.close_ns = int(o.value), int(c.value)
        self.exit_ns = min(int(et(s, "15:55").value), self.close_ns - 5 * NS_MIN)
        self.cutoff_ns = min(int(et(s, "15:15").value), self.close_ns - 45 * NS_MIN)
        self.engine_flat_ns = int(et(s, "15:55").value)
        self.early_close = self.exit_ns < self.engine_flat_ns
        ds = self.data_sessions()
        self._ds = ds
        self.day_start_ns = int(et(s, "04:00").value) if "PRE" in ds else self.open_ns
        self.bar_ns = int(freq_to_timedelta(self.bar_freq).value)
        self._syms = self.symbols(ctx)
        self._tapes: dict[str, Tape] = {}
        self._daily: dict[str, DailyInfo] = {}
        self._st: dict[str, dict] = {}
        self._n_entries: dict[str, int] = {}
        self._limit_exit: dict[str, int] = {}
        self._self_flattened = False
        self.session_start(ctx)

    def session_start(self, ctx) -> None:  # hook
        pass

    def on_bar(self, ctx) -> None:
        if self._session is None:
            self.on_session_start(ctx)
        now = int(ctx.now.value)
        if self.early_close and now >= self.exit_ns:
            if not self._self_flattened:
                self._self_flattened = True
                for sym, q in sorted(ctx.positions().items()):
                    if q and not ctx.instrument(sym).is_option \
                            and not any(o.tag == "flat_by" for o in ctx.open_orders(sym)):
                        ctx.flatten(sym, tag="flat_early_close")
            return
        if now >= self.engine_flat_ns:
            return
        self.step(ctx)

    def step(self, ctx) -> None:  # override
        raise NotImplementedError

    # ------------------------------------------------------------- data helpers
    def tape(self, ctx, sym: str) -> Tape:
        tp = self._tapes.get(sym)
        if tp is None:
            tp = Tape(sym, self.open_ns, self.close_ns, self.day_start_ns, self.bar_ns)
            self._tapes[sym] = tp
        tp.update(ctx, self._ds)
        return tp

    def st(self, sym: str) -> dict:
        d = self._st.get(sym)
        if d is None:
            d = {}
            self._st[sym] = d
        return d

    def daily(self, ctx, sym: str) -> DailyInfo:
        di = self._daily.get(sym)
        if di is not None:
            return di
        di = DailyInfo(False)
        try:
            df = ctx.history(sym, self.daily_n, "1d", adjusted=True)
        except Exception:  # pragma: no cover - portal without daily bars
            df = None
        if df is not None and len(df):
            last_day = pd.Timestamp(df.index[-1]).tz_convert("America/New_York").tz_localize(None).normalize()
            prev = ctx.calendar().prev_session(self._session)
            if last_day == prev:
                di = DailyInfo(True, df["open"].to_numpy(float), df["high"].to_numpy(float),
                               df["low"].to_numpy(float), df["close"].to_numpy(float),
                               df["volume"].to_numpy(float))
            else:
                ctx.log("prior session daily bar missing", symbol=sym, last=str(last_day.date()))
        self._daily[sym] = di
        return di

    def profile(self, ctx, sym: str, cumulative: bool, k_min: Optional[int] = None, lookback: int = 20,
                min_valid: int = 15) -> pd.Series:
        key = ("_prof", cumulative, k_min)
        d = self.st(sym)
        if key in d:
            return d[key]
        base = self.base_min
        nbars = (lookback + 3) * (390 // base)
        hist = ctx.history(sym, nbars, adjusted=True, sessions=("RTH",))
        if hist is not None and len(hist):
            hist = hist[pd.DatetimeIndex(hist.index).as_unit("ns").asi8 < self.open_ns]
        prof = tod_profile(hist, lookback, cumulative, min_valid, k_min if (k_min and k_min != base) else 1)
        d[key] = prof
        return prof

    def minute_of(self, t_ns: int) -> int:
        return et_minute(t_ns)

    def at(self, hhmm: str) -> int:
        """UTC ns of an ET wall-clock time today."""
        return int(et(self._session, hhmm).value)

    # ------------------------------------------------------------- trading helpers
    def entries_today(self, sym: str) -> int:
        return self._n_entries.get(sym, 0)

    def has_exposure(self, ctx, sym: str) -> bool:
        """Position or a working entry order on the symbol."""
        if ctx.position(sym):
            return True
        return any(o.tag == "entry" for o in ctx.open_orders(sym))

    def _cutoff_hhmm(self) -> str:
        t = pd.Timestamp(self.cutoff_ns, tz="UTC").tz_convert("America/New_York")
        return f"{t.hour:02d}:{t.minute:02d}"

    def enter(self, ctx, sym: str, side: int, entry_ref: float, stop: float, *, target: Optional[float] = None,
              target_r: Optional[float] = None, time_exit_ns: Optional[int] = None, resting_stop: bool = True,
              why: str = "") -> Optional[Order]:
        """DAY_RISK entry: market order at the next bar with a protective STOP
        (``resting_stop``), optional fixed target / R-target / time exit.
        Returns None if sizing gives 0 (degenerate stop, after cutoff, cap)."""
        if not (entry_ref == entry_ref and stop == stop) or entry_ref <= 0:
            return None
        if (side > 0 and not stop < entry_ref) or (side < 0 and not stop > entry_ref):
            ctx.log("entry skipped: stop on wrong side", symbol=sym, entry=entry_ref, stop=stop)
            return None
        if target is not None and ((side > 0 and not target > entry_ref) or (side < 0 and not target < entry_ref)):
            ctx.log("entry skipped: target on wrong side", symbol=sym, entry=entry_ref, target=target)
            return None
        if int(ctx.now.value) > self.cutoff_ns:
            return None
        qty = size_day_risk(ctx.prior_nav, entry_ref, stop, existing_qty=ctx.position(sym), now_et=ctx.now_et(),
                            cutoff=self._cutoff_hhmm())
        if qty == 0 or (qty > 0) != (side > 0):
            return None
        te = pd.Timestamp(time_exit_ns, tz="UTC") if time_exit_ns is not None else None
        if resting_stop:
            o = ctx.bracket(sym, qty, stop=stop, target=target, time_exit=te, tag="entry")
        elif target is not None or te is not None:
            o = ctx.bracket(sym, qty, stop=None, target=target, time_exit=te, tag="entry")
        else:
            o = ctx.order(sym, qty, tag="entry", initial_stop=stop)
        self._n_entries[sym] = self._n_entries.get(sym, 0) + 1
        if o.status != OrderStatus.REJECTED:
            self._entries[o.id] = EntryInfo(sym, side, float(stop), target_r, ctx.now)
            self.st(sym)["entry_info"] = self._entries[o.id]
            self.st(sym)["stop"] = float(stop)
        ctx.log("entry", symbol=sym, side=side, ref=entry_ref, stop=stop, target=target, target_r=target_r, why=why)
        return o

    def enter_notional(self, ctx, sym: str, side: int, price: float, why: str = "") -> Optional[Order]:
        """DAY_NOTIONAL entry (10% prior-day NAV, one position per instrument)."""
        qty = size_day_notional(ctx.prior_nav, price, side, existing_qty=ctx.position(sym))
        if qty == 0:
            return None
        o = ctx.order(sym, qty, tag="entry")
        self._n_entries[sym] = self._n_entries.get(sym, 0) + 1
        ctx.log("entry", symbol=sym, side=side, ref=price, why=why)
        return o

    def on_fill(self, ctx, fill) -> None:
        info = self._entries.get(fill.order_id)
        if info is None:
            return
        info.filled += fill.quantity
        info.cost += fill.quantity * fill.price
        if info.target_r is None:
            return
        pos = ctx.position(info.symbol)
        if pos == 0 or (pos > 0) != (info.side > 0):
            return
        avg = info.cost / info.filled
        tgt = avg + info.target_r * (avg - info.stop)
        oco = None
        for o in ctx.open_orders(info.symbol):
            if o.tag == "stop" and o.oco_group:
                oco = o.oco_group
        if oco is None:
            oco = f"bracket-{fill.order_id}"
        if info.target_id is not None:
            ctx.cancel(info.target_id)
        t = ctx.order(info.symbol, -pos, OrderType.LIMIT, limit_price=tgt, tif=TimeInForce.GTC, tag="target",
                      oco_group=oco)
        info.target_id = t.id

    def exit_position(self, ctx, sym: str, tag: str) -> None:
        """Close-evaluated exit: cancel the symbol's orders, market order next bar."""
        q = ctx.position(sym)
        if not q:
            return
        if any(o.tag == tag for o in ctx.open_orders(sym)):
            return
        ctx.flatten(sym, tag=tag)

    def refresh_limit_exit(self, ctx, sym: str, price: float, tag: str = "vwap_exit") -> None:
        """Resting exit limit at ``price`` (e.g. today's VWAP), replaced every
        bar; joins the bracket's OCO group so a stop fill cancels it."""
        q = ctx.position(sym)
        old = self._limit_exit.get(sym)
        if old is not None:
            ctx.cancel(old)
            self._limit_exit.pop(sym, None)
        if not q or not (price == price):
            return
        oco = None
        for o in ctx.open_orders(sym):
            if o.oco_group and o.tag in ("stop", "target", "time_exit"):
                oco = o.oco_group
        t = ctx.order(sym, -q, OrderType.LIMIT, limit_price=float(price), tif=TimeInForce.DAY, tag=tag, oco_group=oco)
        if t.status != OrderStatus.REJECTED:
            self._limit_exit[sym] = t.id


def requires_intraday(freq: str, *, pre: bool = False, quotes: bool = False, daily: bool = True,
                      quote_sessions: tuple = ("RTH",)) -> list[Requirement]:
    out = [Requirement("bars", freq, sessions=("PRE", "RTH") if pre else ("RTH",),
                       note="premarket + RTH bars" if pre else "")]
    if daily:
        out.append(Requirement("bars", "1d", sessions=("RTH",), note="prior close / ADV / daily filters"))
    if quotes:
        out.append(Requirement("quotes", None, sessions=tuple(quote_sessions),
                               note="quote midpoint at the decision time"))
    return out
