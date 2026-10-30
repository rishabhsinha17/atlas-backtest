"""Incremental (streaming) indicator state machines for bar-by-bar strategies.

Each object consumes one completed bar at a time via ``update(...)`` and
returns the current value (NaN during warm-up). Their outputs are identical
to the batch functions (tested to 1e-12), so a strategy may use either form.
Intraday strategies running inside ``on_bar`` should prefer these: O(1) per
bar and impossible to leak future data (they never see it).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np


class StreamingSMA:
    def __init__(self, n: int):
        self.n = int(n)
        self._buf: deque[float] = deque()
        self._sum = 0.0
        self.value = math.nan

    def update(self, x: float) -> float:
        if x is None or math.isnan(x):
            self._buf.clear()
            self._sum = 0.0
            self.value = math.nan
            return self.value
        self._buf.append(x)
        self._sum += x
        if len(self._buf) > self.n:
            self._sum -= self._buf.popleft()
        # recompute periodically to avoid drift
        if len(self._buf) == self.n:
            self.value = math.fsum(self._buf) / self.n
        else:
            self.value = math.nan
        return self.value


class StreamingEMA:
    """EMA with SMA seed (seed='sma') or first-value seed (seed='first');
    alpha = 2/(n+1) unless alpha is given. NaN inputs are skipped."""

    def __init__(self, n: int | None = None, alpha: float | None = None, seed: str = "sma"):
        if alpha is None:
            alpha = 2.0 / (n + 1.0)
        if n is None:
            n = max(1, int(round(2.0 / alpha - 1.0)))
        self.n, self.alpha, self.seed = int(n), float(alpha), seed
        self._seed_buf: list[float] = []
        self.value = math.nan
        self._ready = False

    def update(self, x: float) -> float:
        if x is None or math.isnan(x):
            if not self._ready:
                self._seed_buf.clear()
            return math.nan
        if not self._ready:
            if self.seed == "first":
                self.value = x
                self._ready = True
                return self.value
            self._seed_buf.append(x)
            if len(self._seed_buf) == self.n:
                self.value = sum(self._seed_buf) / self.n
                self._ready = True
                return self.value
            return math.nan
        self.value = self.value + self.alpha * (x - self.value)
        return self.value


class StreamingWilder(StreamingEMA):
    def __init__(self, n: int):
        super().__init__(n=n, alpha=1.0 / n, seed="sma")


class StreamingRSI:
    """Wilder RSI (same seeding as ``oscillators.rsi``). A NaN close makes the
    change at that bar and the next one NaN: output NaN, Wilder state carried
    (and the seed window restarted if not yet seeded), exactly as the batch form."""

    def __init__(self, n: int = 14):
        self._g = StreamingWilder(n)
        self._l = StreamingWilder(n)
        self._prev = math.nan
        self._started = False
        self.value = math.nan

    def update(self, close: float) -> float:
        if not self._started:
            self._started = True
            self._prev = close
            self.value = math.nan
            return self.value
        d = close - self._prev
        self._prev = close
        if math.isnan(d):
            self._g.update(math.nan)
            self._l.update(math.nan)
            self.value = math.nan
            return self.value
        ag = self._g.update(max(d, 0.0))
        al = self._l.update(max(-d, 0.0))
        if math.isnan(ag):
            self.value = math.nan
        elif al == 0:
            self.value = 100.0 if ag > 0 else 50.0
        else:
            self.value = 100.0 - 100.0 / (1.0 + ag / al)
        return self.value


class StreamingATR:
    """Wilder ATR with TR_0 = H_0 - L_0 (same as ``volatility.atr``)."""

    def __init__(self, n: int = 14, method: str = "wilder"):
        self._avg = StreamingWilder(n) if method == "wilder" else StreamingSMA(n)
        self._pc = math.nan
        self.value = math.nan

    def update(self, high: float, low: float, close: float) -> float:
        if math.isnan(self._pc):
            tr = high - low
        else:
            tr = max(high, self._pc) - min(low, self._pc)
        self._pc = close
        self.value = self._avg.update(tr)
        return self.value


class RollingExtreme:
    """O(1) amortised rolling max/min over the last n values (monotonic deque).
    NaN while a NaN is inside the window (same as pandas rolling with min_periods=n)."""

    def __init__(self, n: int, kind: str = "max"):
        self.n = int(n)
        self.kind = kind
        self._dq: deque[tuple[int, float]] = deque()
        self._i = -1
        self._last_nan = -(10 ** 18)
        self.value = math.nan

    def update(self, x: float) -> float:
        self._i += 1
        if x is None or math.isnan(x):
            self._last_nan = self._i
        else:
            better = (lambda a, b: a >= b) if self.kind == "max" else (lambda a, b: a <= b)
            while self._dq and better(x, self._dq[-1][1]):
                self._dq.pop()
            self._dq.append((self._i, x))
        while self._dq and self._dq[0][0] <= self._i - self.n:
            self._dq.popleft()
        ok = self._i >= self.n - 1 and self._last_nan <= self._i - self.n and self._dq
        self.value = self._dq[0][1] if ok else math.nan
        return self.value


class StreamingSessionVWAP:
    """Session VWAP + volume-weighted sd; call ``update(session_key, h, l, c, v)``.
    Resets automatically when session_key changes."""

    def __init__(self, price: str = "typical"):
        self.price = price
        self._key = None
        self._pv = self._pv2 = self._v = 0.0
        self._p0 = None
        self.vwap = self.sd = math.nan

    def update(self, key, high: float, low: float, close: float, volume: float, bar_vwap: float | None = None):
        if key != self._key:
            self._key = key
            self._pv = self._pv2 = self._v = 0.0
            self._p0 = None
        if self.price == "typical":
            p = (high + low + close) / 3.0
        elif self.price == "close":
            p = close
        elif self.price == "bar_vwap":
            p = bar_vwap if bar_vwap is not None and not math.isnan(bar_vwap) else (high + low + close) / 3.0
        else:
            raise ValueError("price must be typical|close|bar_vwap")
        v = 0.0 if volume is None or math.isnan(volume) else float(volume)
        if self._p0 is None:
            self._p0 = p  # centring reference (numerical stability)
        q = p - self._p0
        self._pv += v * q
        self._pv2 += v * q * q
        self._v += v
        if self._v > 0:
            m = self._pv / self._v
            self.vwap = m + self._p0
            self.sd = math.sqrt(max(self._pv2 / self._v - m * m, 0.0))
        else:
            self.vwap = self.sd = math.nan
        return self.vwap


class StreamingOpeningRange:
    """Opening range over [open, open+minutes), identical to ``intraday.opening_range``.

    update(key, end_minute, high, low) where end_minute = minutes from the
    session open to the bar's END. A bar is included only if its start >= 0 and
    its end <= minutes (premarket bars and the 09:29 bar are excluded). Returns
    NaNs until the range is complete (bar end >= minutes)."""

    def __init__(self, minutes: int = 5, bar_minutes: float = 1.0):
        self.minutes = minutes
        self.bar_minutes = float(bar_minutes)
        self._key = None
        self._h = -math.inf
        self._l = math.inf

    def update(self, key, end_minute: float, high: float, low: float):
        if key != self._key:
            self._key = key
            self._h, self._l = -math.inf, math.inf
        start_minute = end_minute - self.bar_minutes
        if start_minute >= 0 and end_minute <= self.minutes:
            self._h = max(self._h, high)
            self._l = min(self._l, low)
        if end_minute >= self.minutes and self._h > -math.inf:
            return self._h, self._l
        return math.nan, math.nan


def run_streaming(obj, *columns) -> np.ndarray:
    """Feed arrays through a streaming object; returns the outputs (testing aid)."""
    cols = [np.asarray(c, float) for c in columns]
    return np.array([obj.update(*[c[i] for c in cols]) for i in range(len(cols[0]))], dtype=float)
