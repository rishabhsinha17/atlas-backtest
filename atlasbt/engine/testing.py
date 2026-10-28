"""In-memory DataPortal test double + synthetic fixtures.

``InMemoryPortal`` implements :class:`atlasbt.core.protocols.DataPortal` over
DataFrames that follow the canonical frames of docs/ARCHITECTURE.md §3. It is
used by the engine's own tests and examples. The ``synthetic_*`` builders
create clearly fake data **for testing mechanics only** — never use them to
evaluate a strategy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.clock import NYSECalendar, et, nyse
from ..core.types import (
    ActionType, AssetClass, CorporateAction, Fidelity, Instrument, MarketEvent,
    OptionQuote,
)
from .timeutil import canonical_freq, freq_to_timedelta, is_daily, utc_index

BAR_COLS = ["open", "high", "low", "close", "volume"]


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _session_date(x) -> pd.Timestamp:
    from ..portfolio.corporate_actions import as_session_date
    return as_session_date(x)


class InMemoryPortal:
    """DataPortal over in-memory frames.

    Parameters
    ----------
    bars : mapping (symbol, freq) -> canonical bar frame (index ``start`` UTC,
        columns open/high/low/close/volume/session/known_at).
    quotes : mapping symbol -> frame indexed by ``ts`` with bid/ask/bid_size/ask_size.
    option_quotes : mapping option symbol -> frame indexed by ``ts`` with
        bid/ask and optional iv/delta/underlying_price/fidelity columns.
    instruments : mapping symbol -> Instrument (default: STOCK, tick 0.01).
    actions / events : lists of CorporateAction / MarketEvent.
    rf : constant annual rate or a Series indexed by (UTC) known time.
    borrow : mapping symbol -> (shortable, fee). Missing -> ``default_borrow``.
    """

    def __init__(self, bars: Optional[Mapping[tuple[str, str], pd.DataFrame]] = None,
                 quotes: Optional[Mapping[str, pd.DataFrame]] = None,
                 option_quotes: Optional[Mapping[str, pd.DataFrame]] = None,
                 instruments: Optional[Mapping[str, Instrument]] = None,
                 actions: Optional[Sequence[CorporateAction]] = None,
                 events: Optional[Sequence[MarketEvent]] = None,
                 rf: float | pd.Series = 0.0,
                 borrow: Optional[Mapping[str, tuple[bool, float]]] = None,
                 default_borrow: Optional[tuple[bool, float]] = (True, 0.003),
                 universes: Optional[Mapping[str, list[str]]] = None,
                 fundamentals: Optional[pd.DataFrame] = None,
                 calendar: Optional[NYSECalendar] = None):
        self._bars: dict[tuple[str, str], pd.DataFrame] = {}
        for (sym, fq), df in (bars or {}).items():
            self._bars[(sym, canonical_freq(fq))] = self._normalize_bars(df, fq)
        self._quotes = {s: self._normalize_ts_frame(df) for s, df in (quotes or {}).items()}
        self._optq = {s: self._normalize_ts_frame(df) for s, df in (option_quotes or {}).items()}
        self._instruments = dict(instruments or {})
        self._actions = list(actions or [])
        self._events = list(events or [])
        self._rf = rf
        self._borrow = dict(borrow or {})
        self._default_borrow = default_borrow
        self._universes = dict(universes or {})
        self._fund = fundamentals
        self.calendar = calendar or nyse()
        self.calls: list[tuple] = []           # (method, as_of) audit trail for tests

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _normalize_bars(df: pd.DataFrame, freq: str) -> pd.DataFrame:
        df = df.copy()
        df.index = utc_index(df.index)
        df.index.name = "start"
        if "session" not in df.columns:
            df["session"] = "DAY" if is_daily(freq) else "RTH"
        if "known_at" not in df.columns:
            df["known_at"] = df.index + freq_to_timedelta(freq)
        df["known_at"] = utc_index(df["known_at"])
        for c in BAR_COLS:
            df[c] = df[c].astype(float)
        df = df.sort_index()
        df.attrs["fidelity"] = Fidelity.T1_BAR
        df.attrs["source"] = "synthetic/in-memory"
        return df

    @staticmethod
    def _normalize_ts_frame(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.index = utc_index(df.index)
        df.index.name = "ts"
        return df.sort_index()

    def add_bars(self, symbol: str, freq: str, df: pd.DataFrame) -> None:
        self._bars[(symbol, canonical_freq(freq))] = self._normalize_bars(df, freq)

    # ------------------------------------------------------------ reference
    def instrument(self, symbol: str) -> Instrument:
        inst = self._instruments.get(symbol)
        if inst is not None:
            return inst
        return Instrument(symbol=symbol, asset_class=AssetClass.STOCK)

    def security_master(self, as_of: pd.Timestamp) -> pd.DataFrame:
        syms = sorted({s for s, _ in self._bars} | set(self._instruments))
        return pd.DataFrame({"symbol": syms,
                             "asset_class": [self.instrument(s).asset_class.value for s in syms]})

    def universe(self, name: str, as_of: pd.Timestamp) -> list[str]:
        return list(self._universes.get(name, []))

    # ------------------------------------------------------------ prices
    def bars(self, symbol: str, freq: str, start: pd.Timestamp, end: pd.Timestamp,
             as_of: pd.Timestamp, adjusted: bool = False,
             sessions: Sequence[str] = ("RTH",)) -> pd.DataFrame:
        self.calls.append(("bars", symbol, freq, _utc(as_of)))
        f = canonical_freq(freq)
        df = self._bars.get((symbol, f))
        if df is None:
            return pd.DataFrame(columns=BAR_COLS + ["session", "known_at"],
                                index=pd.DatetimeIndex([], tz="UTC", name="start"))
        s, e, a = _utc(start), _utc(end), _utc(as_of)
        sel = df[(df.index >= s) & (df.index < e) & (df["known_at"] <= a)]
        if not is_daily(f):
            sel = sel[sel["session"].isin(tuple(sessions))]
        if adjusted:
            sel = self._adjust(symbol, sel, a)
        return sel

    def _adjust(self, symbol: str, df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        """Split + dividend back-adjustment using actions with ex_date <= as_of."""
        if df.empty:
            return df
        acts = [x for x in self._actions if x.symbol == symbol and _session_date(x.ex_date) <= _session_date(as_of)
                and x.type in (ActionType.SPLIT, ActionType.CASH_DIVIDEND, ActionType.SPECIAL_DIVIDEND)]
        if not acts:
            return df
        out = df.copy()
        sess = pd.DatetimeIndex([_session_date(t) for t in out.index])
        full = self._bars.get((symbol, "1d"))
        for x in sorted(acts, key=lambda z: _session_date(z.ex_date)):
            exd = _session_date(x.ex_date)
            mask = sess < exd
            if not mask.any():
                continue
            if x.type == ActionType.SPLIT:
                f = 1.0 / x.ratio
                out.loc[mask, ["open", "high", "low", "close"]] *= f
                out.loc[mask, "volume"] /= f
            else:
                prev_close = None
                if full is not None:
                    fs = pd.DatetimeIndex([_session_date(t) for t in full.index])
                    pc = full["close"][fs < exd]
                    if len(pc):
                        prev_close = float(pc.iloc[-1])
                if prev_close is None:
                    prev_close = float(out.loc[mask, "close"].iloc[-1])
                f = 1.0 - x.amount / prev_close
                out.loc[mask, ["open", "high", "low", "close"]] *= f
        return out

    def quotes(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp,
               as_of: pd.Timestamp) -> pd.DataFrame:
        df = self._quotes.get(symbol)
        if df is None:
            return pd.DataFrame(columns=["bid", "ask", "bid_size", "ask_size"],
                                index=pd.DatetimeIndex([], tz="UTC", name="ts"))
        s, e, a = _utc(start), _utc(end), _utc(as_of)
        return df[(df.index >= s) & (df.index < e) & (df.index <= a)]

    def has(self, symbol: str, kind: str, freq: Optional[str] = None) -> bool:
        """``symbol="*"`` asks portal-wide; for kind 'events' ``freq`` is the event
        KIND and for 'fundamentals' it is the FIELD (contract used by
        Requirement(symbols=("*",)))."""
        if kind == "events":
            kinds = {e.kind for e in self._events if symbol == "*" or e.symbol == symbol}
            return (freq in kinds) if freq else bool(kinds)
        if kind == "fundamentals":
            f = self._fund
            if f is None or len(f) == 0:
                return False
            m = f if symbol == "*" else f[f["symbol"] == symbol]
            return bool(len(m)) if freq is None else bool((m["field"] == freq).any())
        if symbol == "*":
            if kind == "bars":
                return any(freq is None or fq == canonical_freq(freq) for _, fq in self._bars)
            if kind == "quotes":
                return bool(self._quotes)
            if kind == "options":
                return bool(self._optq)
            if kind == "actions":
                return bool(self._actions)
            if kind == "borrow":
                return bool(self._borrow)
            return False
        if kind == "bars":
            if freq is None:
                return any(s == symbol for s, _ in self._bars)
            return (symbol, canonical_freq(freq)) in self._bars
        if kind == "quotes":
            return symbol in self._quotes
        if kind == "options":
            return any(self.instrument(s).underlying == symbol or s.startswith(symbol) for s in self._optq)
        if kind == "actions":
            return any(a.symbol == symbol for a in self._actions)
        if kind == "borrow":
            return symbol in self._borrow
        return False

    def fidelity_of(self, symbol: str, kind: str, freq: Optional[str] = None) -> Fidelity:
        if not self.has(symbol, kind, freq):
            return Fidelity.NA
        return Fidelity.T0_QUOTE if kind == "quotes" else Fidelity.T1_BAR

    # ------------------------------------------------------------ actions/events
    def corporate_actions(self, symbols: Optional[Iterable[str]], start: pd.Timestamp,
                          end: pd.Timestamp, as_of: pd.Timestamp) -> list[CorporateAction]:
        syms = None if symbols is None else set(symbols)
        s, e, a = _session_date(start), _session_date(end), _utc(as_of)
        return [x for x in self._actions
                if (syms is None or x.symbol in syms) and s <= _session_date(x.ex_date) <= e
                and _utc(x.known_at) <= a]

    def events(self, kind: str, symbols: Optional[Iterable[str]], start: pd.Timestamp,
               end: pd.Timestamp, as_of: pd.Timestamp) -> list[MarketEvent]:
        syms = None if symbols is None else set(symbols)
        s, e, a = _utc(start), _utc(end), _utc(as_of)
        out = []
        for x in self._events:
            if x.kind != kind or (syms is not None and x.symbol not in syms):
                continue
            t = _utc(x.known_at)
            if t > a:
                continue
            eff = _utc(x.effective_at) if x.effective_at is not None else t
            if eff < e and (t >= s or eff >= s):
                out.append(x)
        return out

    # ------------------------------------------------------------ options
    def option_chain(self, underlying: str, as_of: pd.Timestamp, min_dte: int = 0,
                     max_dte: int = 400) -> pd.DataFrame:
        a = _utc(as_of)
        rows = []
        for sym, df in self._optq.items():
            inst = self.instrument(sym)
            if inst.underlying != underlying:
                continue
            d = df[df.index <= a]
            if d.empty:
                continue
            r = d.iloc[-1]
            rows.append({"ts": d.index[-1], "underlying": underlying, "symbol": sym,
                         "expiry": inst.expiry, "strike": inst.strike,
                         "right": inst.right.value if inst.right else None,
                         "bid": r["bid"], "ask": r["ask"], "mid": 0.5 * (r["bid"] + r["ask"]),
                         "multiplier": inst.multiplier,
                         "fidelity": r.get("fidelity", Fidelity.T0_QUOTE)})
        df = pd.DataFrame(rows)
        if not df.empty:
            latest = df["ts"].max()
            df = df[df["ts"] == latest]
        return df

    def option_quotes(self, instrument: Instrument, start: pd.Timestamp, end: pd.Timestamp,
                      as_of: pd.Timestamp) -> pd.DataFrame:
        """Snapshots with ts in (start, end] and ts <= as_of, ascending."""
        df = self._optq.get(instrument.symbol)
        if df is None:
            return pd.DataFrame(columns=["bid", "ask"], index=pd.DatetimeIndex([], tz="UTC", name="ts"))
        s, e, a = _utc(start), _utc(end), _utc(as_of)
        out = df[(df.index > s) & (df.index <= e) & (df.index <= a)].copy()
        if "fidelity" not in out.columns:
            out["fidelity"] = int(Fidelity.T0_QUOTE)
        return out

    def option_quote(self, instrument: Instrument, as_of: pd.Timestamp) -> Optional[OptionQuote]:
        df = self._optq.get(instrument.symbol)
        if df is None:
            return None
        d = df[df.index <= _utc(as_of)]
        if d.empty:
            return None
        r = d.iloc[-1]
        fid = r.get("fidelity", Fidelity.T0_QUOTE)
        return OptionQuote(instrument=instrument, ts=d.index[-1], bid=float(r["bid"]), ask=float(r["ask"]),
                           underlying_price=float(r.get("underlying_price", np.nan)),
                           iv=float(r.get("iv", np.nan)), delta=float(r.get("delta", np.nan)),
                           fidelity=Fidelity(int(fid)))

    # ------------------------------------------------------------ rates etc
    def risk_free_rate(self, as_of: pd.Timestamp, tenor: str = "3M") -> float:
        if isinstance(self._rf, pd.Series):
            s = self._rf
            idx = utc_index(s.index)
            m = idx <= _utc(as_of)
            return float(s.values[m][-1]) if m.any() else float("nan")
        return float(self._rf)

    def borrow(self, symbol: str, as_of: pd.Timestamp) -> tuple[bool, float, Fidelity]:
        if symbol in self._borrow:
            ok, fee = self._borrow[symbol]
            return bool(ok), float(fee), Fidelity.T1_BAR
        if self._default_borrow is None:
            return False, float("nan"), Fidelity.NA
        ok, fee = self._default_borrow
        return bool(ok), float(fee), Fidelity.T2_PROXY

    def fundamentals(self, symbols: Optional[Iterable[str]], fields: Sequence[str],
                     as_of: pd.Timestamp) -> pd.DataFrame:
        if self._fund is None:
            return pd.DataFrame(columns=["symbol", "field", "value", "period_end", "known_at", "source"])
        f = self._fund
        m = (f["known_at"] <= _utc(as_of)) & f["field"].isin(list(fields))
        if symbols is not None:
            m &= f["symbol"].isin(list(symbols))
        return f[m].sort_values("known_at").groupby(["symbol", "field"]).tail(1)


# ==========================================================================
# Synthetic fixtures (TESTING MECHANICS ONLY)
# ==========================================================================
def _sessions(start: str, end: str, calendar=None) -> pd.DatetimeIndex:
    cal = calendar or nyse()
    return cal.sessions(pd.Timestamp(start), pd.Timestamp(end))


def synthetic_daily_bars(start: str, end: str, start_price: float = 100.0, daily_vol: float = 0.015,
                         volume: float = 1_000_000.0, seed: int = 0, drift: float = 0.0,
                         calendar=None) -> pd.DataFrame:
    """Random-walk daily OHLCV on NYSE sessions. Index = session open (UTC),
    known_at = session close (UTC). SYNTHETIC — mechanics tests only."""
    cal = calendar or nyse()
    sess = _sessions(start, end, cal)
    rng = np.random.default_rng(seed)
    n = len(sess)
    rets = rng.normal(drift, daily_vol, n)
    close = start_price * np.exp(np.cumsum(rets))
    prev = np.concatenate([[start_price], close[:-1]])
    gap = rng.normal(0, daily_vol * 0.3, n)
    open_ = prev * np.exp(gap)
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, daily_vol * 0.5, n)))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, daily_vol * 0.5, n)))
    oc = [cal.open_close(s) for s in sess]
    idx = pd.DatetimeIndex([o for o, _ in oc], name="start")
    df = pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close,
                       "volume": np.full(n, float(volume)), "session": "DAY",
                       "known_at": pd.DatetimeIndex([c for _, c in oc])}, index=idx)
    return df


def synthetic_intraday_bars(start: str, end: str, freq: str = "1m", start_price: float = 100.0,
                            daily_vol: float = 0.015, volume_per_bar: float = 10_000.0, seed: int = 0,
                            calendar=None, include_extended: bool = False) -> pd.DataFrame:
    """Random-walk RTH (optionally PRE/POST) bars. SYNTHETIC — mechanics tests only."""
    cal = calendar or nyse()
    sess = _sessions(start, end, cal)
    td = freq_to_timedelta(freq)
    rng = np.random.default_rng(seed)
    frames = []
    price = start_price
    for s in sess:
        o, c = cal.open_close(s)
        parts = []
        if include_extended:
            parts.append(("PRE", et(s, "04:00"), o))
        parts.append(("RTH", o, c))
        if include_extended:
            parts.append(("POST", c, et(s, "20:00")))
        for code, a, b in parts:
            starts = pd.date_range(a, b, freq=td, inclusive="left")
            n = len(starts)
            if n == 0:
                continue
            nbar = max(int(round(6.5 * 3600 / td.total_seconds())), 1)
            sig = daily_vol / np.sqrt(nbar)
            r = rng.normal(0, sig, n)
            closes = price * np.exp(np.cumsum(r))
            opens = np.concatenate([[price], closes[:-1]])
            hi = np.maximum(opens, closes) * np.exp(np.abs(rng.normal(0, sig * 0.5, n)))
            lo = np.minimum(opens, closes) * np.exp(-np.abs(rng.normal(0, sig * 0.5, n)))
            vol = np.full(n, volume_per_bar * (0.1 if code != "RTH" else 1.0))
            known = np.minimum((starts + td).asi8, pd.DatetimeIndex([b] * n).as_unit(starts.unit).asi8)
            frames.append(pd.DataFrame({"open": opens, "high": hi, "low": lo, "close": closes,
                                        "volume": vol, "session": code,
                                        "known_at": pd.DatetimeIndex(known, tz="UTC").as_unit(starts.unit)},
                                       index=starts))
            price = closes[-1]
    df = pd.concat(frames)
    df.index.name = "start"
    return df


def daily_from_intraday(df: pd.DataFrame, calendar=None) -> pd.DataFrame:
    """Aggregate RTH intraday bars into canonical daily bars."""
    cal = calendar or nyse()
    rth = df[df["session"] == "RTH"]
    sess = pd.DatetimeIndex(rth.index.tz_convert("America/New_York").tz_localize(None).normalize())
    g = rth.groupby(sess)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()})
    oc = [cal.open_close(s) for s in out.index]
    out.index = pd.DatetimeIndex([o for o, _ in oc], name="start")
    out["session"] = "DAY"
    out["known_at"] = pd.DatetimeIndex([c for _, c in oc])
    return out


def synthetic_bars_from_rows(rows: Sequence[tuple], freq: str = "1d", session_dates: Optional[Sequence] = None,
                             start_times: Optional[Sequence] = None, calendar=None,
                             volume: float = 1_000_000.0) -> pd.DataFrame:
    """Build bars from explicit (open, high, low, close[, volume]) rows.

    Daily: give ``session_dates`` (one per row). Intraday: give ``start_times``
    (UTC). SYNTHETIC — mechanics tests only."""
    cal = calendar or nyse()
    o = np.array([r[0] for r in rows], float)
    h = np.array([r[1] for r in rows], float)
    l = np.array([r[2] for r in rows], float)
    c = np.array([r[3] for r in rows], float)
    v = np.array([r[4] if len(r) > 4 else volume for r in rows], float)
    if is_daily(freq):
        assert session_dates is not None
        oc = [cal.open_close(pd.Timestamp(s)) for s in session_dates]
        idx = pd.DatetimeIndex([x for x, _ in oc], name="start")
        known = pd.DatetimeIndex([y for _, y in oc])
        sess = "DAY"
    else:
        assert start_times is not None
        idx = utc_index(start_times)
        idx.name = "start"
        known = idx + freq_to_timedelta(freq)
        sess = "RTH"
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v,
                         "session": sess, "known_at": known}, index=idx)
