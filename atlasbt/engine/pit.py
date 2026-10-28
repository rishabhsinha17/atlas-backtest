"""Point-in-time guard around a DataPortal.

Every query is forced to ``as_of = now`` (the end of the last completed bar).
Passing an explicit ``as_of`` later than now raises :class:`LookaheadError`.
Results are re-checked: any row whose knowledge time exceeds ``now`` makes the
wrapper raise (that would be a DATA-layer contract breach, never silently
dropped), and bars whose ``known_at`` > now — the in-progress bar — are never
returned. There is no escape hatch.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import pandas as pd

from ..core.types import Fidelity, Instrument, LookaheadError
from .timeutil import freq_to_timedelta, utc_index


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


class PITPortal:
    """Implements the DataPortal protocol with ``as_of`` pinned to ``now``.

    Slotted with name-mangled fields: ``ctx.portal()._p`` does not exist, and
    tests/engine/test_eng_strategy_lint.py forbids private-attribute access
    from strategy code."""

    __slots__ = ("__p", "__now", "__on_access")

    def __init__(self, portal, now_fn: Callable[[], pd.Timestamp], on_access: Optional[Callable] = None):
        self.__p = portal
        self.__now = now_fn
        self.__on_access = on_access

    # ------------------------------------------------------------- helpers
    def _as_of(self, as_of) -> pd.Timestamp:
        now = self.__now()
        if as_of is None:
            return now
        a = _utc(as_of)
        if a > now:
            raise LookaheadError(f"as_of {a} is after ctx.now {now}")
        return a

    def _note(self, kind: str, symbol, fidelity=None):
        if self.__on_access is not None:
            self.__on_access(kind, symbol, fidelity)

    # ------------------------------------------------------------- reference
    def instrument(self, symbol: str) -> Instrument:
        return self.__p.instrument(symbol)

    def security_master(self, as_of=None) -> pd.DataFrame:
        return self.__p.security_master(self._as_of(as_of))

    def universe(self, name: str, as_of=None) -> list[str]:
        return self.__p.universe(name, self._as_of(as_of))

    def has(self, symbol: str, kind: str, freq: Optional[str] = None) -> bool:
        return self.__p.has(symbol, kind, freq)

    def fidelity_of(self, symbol: str, kind: str, freq: Optional[str] = None) -> Fidelity:
        return self.__p.fidelity_of(symbol, kind, freq)

    # ------------------------------------------------------------- prices
    def bars(self, symbol: str, freq: str, start, end=None, as_of=None, adjusted: bool = False,
             sessions: Sequence[str] = ("RTH",)) -> pd.DataFrame:
        a = self._as_of(as_of)
        end = a + pd.Timedelta(1, "ns") if end is None else min(_utc(end), a + pd.Timedelta(1, "ns"))
        df = self.__p.bars(symbol, freq, _utc(start), end, as_of=a, adjusted=adjusted, sessions=sessions)
        if df is None or len(df) == 0:
            return df
        if "known_at" in df.columns:
            known = utc_index(df["known_at"])
        else:
            known = utc_index(df.index) + freq_to_timedelta(freq)
        if (known > a).any():
            raise LookaheadError(f"portal returned {symbol} {freq} bars known after {a}")
        self._note("bars", symbol, df.attrs.get("fidelity"))
        return df

    def quotes(self, symbol: str, start, end=None, as_of=None) -> pd.DataFrame:
        a = self._as_of(as_of)
        end = a + pd.Timedelta(1, "ns") if end is None else min(_utc(end), a + pd.Timedelta(1, "ns"))
        df = self.__p.quotes(symbol, _utc(start), end, as_of=a)
        if df is not None and len(df) and (utc_index(df.index) > a).any():
            raise LookaheadError(f"portal returned {symbol} quotes after {a}")
        return df

    # ------------------------------------------------------------- events
    def corporate_actions(self, symbols: Optional[Iterable[str]], start, end, as_of=None):
        a = self._as_of(as_of)
        out = self.__p.corporate_actions(symbols, start, end, as_of=a)
        for x in out:
            if _utc(x.known_at) > a:
                raise LookaheadError(f"corporate action {x} known after {a}")
        return out

    def events(self, kind: str, symbols: Optional[Iterable[str]], start, end, as_of=None):
        a = self._as_of(as_of)
        out = self.__p.events(kind, symbols, start, end, as_of=a)
        for x in out:
            if _utc(x.known_at) > a:
                raise LookaheadError(f"event {x} known after {a}")
        return out

    # ------------------------------------------------------------- options
    def option_chain(self, underlying: str, as_of=None, min_dte: int = 0, max_dte: int = 400) -> pd.DataFrame:
        a = self._as_of(as_of)
        df = self.__p.option_chain(underlying, a, min_dte, max_dte)
        if df is not None and len(df) and "ts" in df.columns and (utc_index(df["ts"]) > a).any():
            raise LookaheadError(f"option chain for {underlying} has snapshots after {a}")
        return df

    def option_quotes(self, instrument: Instrument, start, end=None, as_of=None) -> pd.DataFrame:
        a = self._as_of(as_of)
        end = a if end is None else min(_utc(end), a)
        df = self.__p.option_quotes(instrument, _utc(start), end, as_of=a)
        if df is not None and len(df) and (utc_index(df.index) > a).any():
            raise LookaheadError(f"option quotes {instrument.symbol} after {a}")
        return df

    def option_quote(self, instrument: Instrument, as_of=None):
        a = self._as_of(as_of)
        q = self.__p.option_quote(instrument, a)
        if q is not None and _utc(q.ts) > a:
            raise LookaheadError(f"option quote {instrument.symbol} at {q.ts} after {a}")
        return q

    # ------------------------------------------------------------- rates etc
    def risk_free_rate(self, as_of=None, tenor: str = "3M") -> float:
        return self.__p.risk_free_rate(self._as_of(as_of), tenor)

    def borrow(self, symbol: str, as_of=None):
        return self.__p.borrow(symbol, self._as_of(as_of))

    def fundamentals(self, symbols, fields, as_of=None) -> pd.DataFrame:
        a = self._as_of(as_of)
        df = self.__p.fundamentals(symbols, fields, a)
        if df is not None and len(df) and "known_at" in df.columns and (utc_index(df["known_at"]) > a).any():
            raise LookaheadError(f"fundamentals known after {a}")
        return df
