"""NYSE trading calendar and timezone helpers (implements protocols.TradingCalendar).

Session dates are naive midnight Timestamps representing the ET date.
Open/close times are tz-aware UTC Timestamps and honour early closes and
historical closures, via exchange_calendars' XNYS calendar.
"""
from __future__ import annotations

from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd

ET = "America/New_York"
UTC = "UTC"


def to_utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)


def to_et(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError("to_et expects a tz-aware timestamp")
    return ts.tz_convert(ET)


def et(date, hhmm: str) -> pd.Timestamp:
    """UTC timestamp for wall-clock `hhmm` ET on `date` (DST-correct)."""
    d = pd.Timestamp(date)
    if d.tzinfo is not None:
        d = d.tz_convert(ET).tz_localize(None)
    h, m = (int(x) for x in hhmm.split(":"))
    return pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=h, minute=m).tz_localize(ET).tz_convert(UTC)


def session_date_of(ts) -> pd.Timestamp:
    """ET calendar date (naive midnight) of a UTC timestamp."""
    return to_et(to_utc(ts)).tz_localize(None).normalize()


class NYSECalendar:
    """protocols.TradingCalendar backed by exchange_calendars XNYS."""

    def __init__(self, start: str = "1990-01-01", end: str | None = None):
        end = end or (pd.Timestamp.today() + pd.DateOffset(years=2)).strftime("%Y-%m-%d")
        self._cal = xcals.get_calendar("XNYS", start=start, end=end)
        sched = self._cal.schedule
        idx = pd.DatetimeIndex(sched.index).tz_localize(None) if sched.index.tz is not None else pd.DatetimeIndex(sched.index)
        self._sessions = idx.normalize()
        self._open = pd.Series(pd.DatetimeIndex(sched["open"]).tz_convert(UTC), index=self._sessions)
        self._close = pd.Series(pd.DatetimeIndex(sched["close"]).tz_convert(UTC), index=self._sessions)
        regular_close_et = pd.DatetimeIndex(sched["close"]).tz_convert(ET)
        self._early = pd.Series((regular_close_et.hour < 16), index=self._sessions)
        self._pos = {d: i for i, d in enumerate(self._sessions)}

    # ---- protocol
    def sessions(self, start, end) -> pd.DatetimeIndex:
        s, e = _naive(start), _naive(end)
        return self._sessions[(self._sessions >= s) & (self._sessions <= e)]

    def open_close(self, session) -> tuple[pd.Timestamp, pd.Timestamp]:
        d = _naive(session)
        return self._open[d], self._close[d]

    def is_session(self, date) -> bool:
        return _naive(date) in self._pos

    def next_session(self, date, n: int = 1) -> pd.Timestamp:
        d = _naive(date)
        i = self._sessions.searchsorted(d, side="right")
        return self._sessions[i + n - 1]

    def prev_session(self, date, n: int = 1) -> pd.Timestamp:
        d = _naive(date)
        i = self._sessions.searchsorted(d, side="left")
        return self._sessions[i - n]

    def is_early_close(self, session) -> bool:
        return bool(self._early[_naive(session)])

    def is_last_session_of_month(self, session) -> bool:
        d = _naive(session)
        nxt = self.next_session(d)
        return nxt.month != d.month

    def sessions_in_month(self, session) -> pd.DatetimeIndex:
        d = _naive(session)
        m = self._sessions[(self._sessions.year == d.year) & (self._sessions.month == d.month)]
        return m

    def holidays(self, start, end) -> pd.DatetimeIndex:
        """Weekdays in [start, end] that are not sessions (full-day closures)."""
        s, e = _naive(start), _naive(end)
        wd = pd.bdate_range(s, e)
        return wd[~wd.isin(self._sessions)]

    # ---- extras
    def session_index(self, session) -> int:
        return self._pos[_naive(session)]

    def session_minutes(self, session, freq: str = "1min") -> pd.DatetimeIndex:
        """Bar START times (UTC) of regular-session bars for `session`."""
        o, c = self.open_close(session)
        return pd.date_range(o, c, freq=freq, inclusive="left")

    @property
    def all_sessions(self) -> pd.DatetimeIndex:
        return self._sessions


def _naive(d) -> pd.Timestamp:
    d = pd.Timestamp(d)
    if d.tzinfo is not None:
        d = d.tz_convert(ET).tz_localize(None)
    return d.normalize()


@lru_cache(maxsize=1)
def nyse() -> NYSECalendar:
    return NYSECalendar()
