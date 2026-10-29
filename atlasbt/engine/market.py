"""Engine-internal market data cache (numpy arrays) with a hard time guard.

The engine pre-loads raw bars per symbol once (one portal call per symbol &
frequency over the whole backtest range, ``as_of`` = range end) because
per-step portal calls would be far too slow for 20 years of minute bars.
Access is gated: :meth:`MarketData.window` and every price lookup refuse to
return a bar whose ``known_at`` (end) is later than the engine's current time
(``now_ns``) and raise :class:`LookaheadError` otherwise. Strategies never see
this object; they go through the PIT-wrapped portal in the Context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..core.types import ActionType, LookaheadError
from ..execution.spread import SpreadModel, SpreadSeries
from ..portfolio.corporate_actions import as_session_date
from .timeutil import as_ns_array, canonical_freq, freq_to_timedelta, is_daily, ts_utc, utc_index

SESSION_CODE = {"RTH": 0, "PRE": 1, "POST": 2, "DAY": 3}
CODE_SESSION = {v: k for k, v in SESSION_CODE.items()}


class BarArrays:
    __slots__ = ("symbol", "freq", "start", "end", "o", "h", "l", "c", "v", "sess", "n")

    def __init__(self, symbol: str, freq: str, df: Optional[pd.DataFrame]):
        self.symbol = symbol
        self.freq = freq
        if df is None or len(df) == 0:
            z = np.zeros(0)
            self.start = np.zeros(0, np.int64)
            self.end = np.zeros(0, np.int64)
            self.o = self.h = self.l = self.c = self.v = z
            self.sess = np.zeros(0, np.int8)
            self.n = 0
            return
        df = df.sort_index()
        self.start = as_ns_array(df.index)
        if "known_at" in df.columns:
            self.end = as_ns_array(df["known_at"])
        else:
            self.end = self.start + freq_to_timedelta(freq).value
        self.o = df["open"].to_numpy(float)
        self.h = df["high"].to_numpy(float)
        self.l = df["low"].to_numpy(float)
        self.c = df["close"].to_numpy(float)
        self.v = df["volume"].to_numpy(float) if "volume" in df.columns else np.full(len(df), np.nan)
        if "session" in df.columns:
            self.sess = np.array([SESSION_CODE.get(str(s), 0) for s in df["session"].to_numpy()], np.int8)
        else:
            self.sess = np.full(len(df), 3 if is_daily(freq) else 0, np.int8)
        self.n = len(df)

    def range_by_end(self, lo_excl: int, hi_incl: int) -> tuple[int, int]:
        i0 = int(np.searchsorted(self.end, lo_excl, side="right"))
        i1 = int(np.searchsorted(self.end, hi_incl, side="right"))
        return i0, i1

    def last_index_at(self, t: int) -> int:
        return int(np.searchsorted(self.end, t, side="right")) - 1

    def frame(self, i0: int, i1: int) -> pd.DataFrame:
        idx = pd.DatetimeIndex(self.start[i0:i1], tz="UTC", name="start")
        return pd.DataFrame({"open": self.o[i0:i1], "high": self.h[i0:i1], "low": self.l[i0:i1],
                             "close": self.c[i0:i1], "volume": self.v[i0:i1],
                             "session": [CODE_SESSION[int(x)] for x in self.sess[i0:i1]],
                             "known_at": pd.DatetimeIndex(self.end[i0:i1], tz="UTC")}, index=idx)


class DailyStats:
    """Per-symbol daily arrays with PIT derived statistics.

    For session index i, ``adv_dollar[i]``, ``sigma[i]`` and ``spread`` use
    sessions <= i-1 only. ``prior_close[i]`` is the raw close of session i-1
    expressed in session i's share units (divided by a split ratio effective
    on session i)."""

    def __init__(self, symbol: str, bars: BarArrays, sessions_naive: np.ndarray,
                 split_ratio_on: dict[int, float], spread_model: SpreadModel, window: int = 20,
                 asset_class: str = "STK"):
        self.symbol = symbol
        self.bars = bars
        self.session_ns = sessions_naive        # naive ET date ns per row
        self.index = {int(s): i for i, s in enumerate(sessions_naive)}
        n = bars.n
        ratio = np.ones(n)
        for i, s in enumerate(sessions_naive):
            r = split_ratio_on.get(int(s))
            if r:
                ratio[i] = r
        # back-adjustment factor (scale-invariant uses only)
        cum = np.ones(n)
        acc = 1.0
        for i in range(n - 1, -1, -1):
            cum[i] = acc
            acc /= ratio[i]
        self.adj = cum
        c = bars.c
        ac = c * cum
        self.prior_close = np.full(n, np.nan)
        if n > 1:
            self.prior_close[1:] = c[:-1] / ratio[1:]
        ret = np.full(n, np.nan)
        if n > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                ret[1:] = ac[1:] / ac[:-1] - 1.0
        self.ret = ret
        dv = c * bars.v
        self.adv_dollar = _shifted_rolling_mean(dv, window)
        self.sigma = _shifted_rolling_std(ret, window)
        if getattr(spread_model, "method", None) == "calibrated":
            self.spread: SpreadSeries = spread_model.series(
                bars.h * cum, bars.l * cum, ac, raw_close=c, volume=bars.v, dates=sessions_naive,
                asset_class=asset_class, adv_dollar=self.adv_dollar, sigma=self.sigma)
        else:
            self.spread = spread_model.series(bars.h * cum, bars.l * cum, ac)

    def idx(self, session_ns: int) -> int:
        return self.index.get(int(session_ns), -1)


def _shifted_rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    s = pd.Series(x).rolling(w, min_periods=max(5, w // 4)).mean().to_numpy()
    out = np.full(len(x), np.nan)
    if len(x) > 1:
        out[1:] = s[:-1]
    return out


def _shifted_rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    s = pd.Series(x).rolling(w, min_periods=max(5, w // 4)).std().to_numpy()
    out = np.full(len(x), np.nan)
    if len(x) > 1:
        out[1:] = s[:-1]
    return out


class MarketData:
    def __init__(self, portal, calendar, first_session: pd.Timestamp, last_session: pd.Timestamp,
                 base_freq: str, data_sessions: Sequence[str], spread_model: SpreadModel,
                 lookback_sessions: int = 60, stats_window: int = 20):
        self.portal = portal
        self.cal = calendar
        self.base_freq = canonical_freq(base_freq)
        self.data_sessions = tuple(data_sessions)
        self.spread_model = spread_model
        self.stats_window = stats_window
        self.first_session = pd.Timestamp(first_session)
        self.last_session = pd.Timestamp(last_session)
        all_s = calendar.all_sessions
        i0 = max(int(all_s.searchsorted(self.first_session)) - lookback_sessions, 0)
        self.load_first_session = all_s[i0]
        self.range_start = calendar.open_close(self.load_first_session)[0] - pd.Timedelta(hours=12)
        self.range_end = calendar.open_close(self.last_session)[1] + pd.Timedelta(hours=8)
        self.now_ns: int = -(2 ** 62)
        self._bars: dict[tuple[str, str], BarArrays] = {}
        self._daily: dict[str, Optional[DailyStats]] = {}
        self._splits: dict[str, dict[int, float]] = {}
        self.guard_violations = 0

    # ------------------------------------------------------------- loading
    def bars(self, symbol: str, freq: Optional[str] = None) -> BarArrays:
        f = self.base_freq if freq is None else canonical_freq(freq)
        key = (symbol, f)
        b = self._bars.get(key)
        if b is None:
            sessions = ("DAY",) if f == "1d" else self.data_sessions
            try:
                df = self.portal.bars(symbol, f, self.range_start, self.range_end, as_of=self.range_end,
                                      adjusted=False, sessions=sessions)
            except Exception:
                df = None
            b = BarArrays(symbol, f, df)
            self._bars[key] = b
        return b

    def splits(self, symbol: str) -> dict[int, float]:
        sp = self._splits.get(symbol)
        if sp is None:
            sp = {}
            try:
                acts = self.portal.corporate_actions([symbol], self.range_start, self.range_end,
                                                     as_of=self.range_end)
            except Exception:
                acts = []
            for a in acts:
                if a.type == ActionType.SPLIT and a.ratio and a.ratio > 0:
                    sp[int(as_session_date(a.ex_date).value)] = float(a.ratio)
            self._splits[symbol] = sp
        return sp

    def _asset_class(self, symbol: str) -> str:
        """'ETF' or 'STK' from the instrument master (ETFs have their own spread curve)."""
        try:
            inst = self.portal.instrument(symbol)
        except Exception:
            inst = None
        ac = getattr(inst, "asset_class", None)
        return "ETF" if ac is not None and getattr(ac, "value", ac) == "ETF" else "STK"

    def daily(self, symbol: str) -> Optional[DailyStats]:
        if symbol in self._daily:
            return self._daily[symbol]
        b = self.bars(symbol, "1d")
        if b.n == 0 and self.base_freq != "1d":
            # aggregate from base intraday bars (RTH only)
            ib = self.bars(symbol, self.base_freq)
            b = _aggregate_daily(symbol, ib, self.cal)
            self._bars[(symbol, "1d")] = b
        if b.n == 0:
            self._daily[symbol] = None
            return None
        sess = (pd.DatetimeIndex(b.start, tz="UTC").tz_convert("America/New_York").tz_localize(None)
                .normalize().as_unit("ns").asi8.astype(np.int64))
        st = DailyStats(symbol, b, sess, self.splits(symbol), self.spread_model, self.stats_window,
                        asset_class=self._asset_class(symbol))
        self._daily[symbol] = st
        return st

    def may_be_truncated(self, b: BarArrays) -> bool:
        """True if the cache window may cut off older history (data starts at
        the cache's load boundary or nothing was loaded)."""
        return b.n == 0 or int(b.start[0]) <= int((self.range_start + pd.Timedelta(days=5)).value)

    # ------------------------------------------------------------- guarded access
    def check(self, hi_ns: int) -> None:
        if hi_ns > self.now_ns:
            self.guard_violations += 1
            raise LookaheadError(f"engine requested data up to {ts_utc(hi_ns)} at now={ts_utc(self.now_ns)}")

    def window(self, symbol: str, lo_excl: int, hi_incl: int, freq: Optional[str] = None) -> tuple[BarArrays, int, int]:
        self.check(hi_incl)
        b = self.bars(symbol, freq)
        i0, i1 = b.range_by_end(lo_excl, hi_incl)
        return b, i0, i1

    def last_close(self, symbol: str, t_ns: Optional[int] = None, freq: Optional[str] = None,
                   rth_only: bool = False) -> float:
        t = self.now_ns if t_ns is None else t_ns
        self.check(t)
        b = self.bars(symbol, freq)
        i = b.last_index_at(t)
        while i >= 0 and rth_only and b.sess[i] not in (0, 3):
            i -= 1
        return float(b.c[i]) if i >= 0 else float("nan")

    def last_bar_index(self, symbol: str, t_ns: Optional[int] = None, freq: Optional[str] = None) -> int:
        t = self.now_ns if t_ns is None else t_ns
        self.check(t)
        return self.bars(symbol, freq).last_index_at(t)

    def session_close_price(self, symbol: str, session: pd.Timestamp) -> float:
        """Official close of `session` (daily bar close, else last RTH bar)."""
        _, c = self.cal.open_close(session)
        self.check(int(c.value))
        st = self.daily(symbol)
        if st is not None:
            i = st.idx(pd.Timestamp(session).value)
            if i >= 0 and st.bars.end[i] <= self.now_ns:
                return float(st.bars.c[i])
        return self.last_close(symbol, int(c.value), rth_only=True)

    def session_open_price(self, symbol: str, session: pd.Timestamp) -> float:
        o, _ = self.cal.open_close(session)
        b = self.bars(symbol)
        i0 = int(np.searchsorted(b.start, int(o.value), side="left"))
        if i0 < b.n and b.start[i0] == int(o.value):
            self.check(int(o.value))     # the opening print is known at the open
            return float(b.o[i0])
        st = self.daily(symbol)
        if st is not None:
            i = st.idx(pd.Timestamp(session).value)
            if i >= 0:
                return float(st.bars.o[i])
        return float("nan")


def _aggregate_daily(symbol: str, ib: BarArrays, cal) -> BarArrays:
    if ib.n == 0:
        return BarArrays(symbol, "1d", None)
    df = ib.frame(0, ib.n)
    rth = df[df["session"] == "RTH"]
    if rth.empty:
        return BarArrays(symbol, "1d", None)
    sess = pd.DatetimeIndex(rth.index.tz_convert("America/New_York").tz_localize(None).normalize())
    g = rth.groupby(sess)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()})
    oc = [cal.open_close(s) for s in out.index]
    out.index = pd.DatetimeIndex([o for o, _ in oc], name="start")
    out["session"] = "DAY"
    out["known_at"] = pd.DatetimeIndex([c for _, c in oc])
    return BarArrays(symbol, "1d", out)
