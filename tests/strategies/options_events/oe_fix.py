"""SYNTHETIC fixtures for the options-events strategy tests (S158-S172).

Everything here is fake data for testing MECHANICS only (timing, contract
selection, fills, exits) - never for evaluating a strategy.  Option quotes are
BSM prices from a caller-defined spot path and vol function, rounded to cents
with a fixed half-spread.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import pandas as pd

from atlasbt.core.clock import et, nyse
from atlasbt.core.types import (AssetClass, ExerciseStyle, Fidelity, Instrument, MarketEvent, Right,
                                SettlementType)
from atlasbt.engine import BacktestConfig
from atlasbt.engine.backtester import run_backtest
from atlasbt.engine.testing import InMemoryPortal
from atlasbt.execution.impact import SquareRootImpact
from atlasbt.execution.spread import SpreadConfig
from atlasbt.pricing.bsm import bsm_price
from atlasbt.pricing.time import year_fraction

CAL = nyse()


def zero_cost(**kw) -> BacktestConfig:
    base = dict(fees="NO_FEES",
                spread=SpreadConfig(bucket_floors=((math.inf, 0.0),), default_spread=1e-12, tod_open=(),
                                    tod_close=(), tick_floor=False, max_spread=1e-12),
                impact=SquareRootImpact(k=0.0), debug=False)
    base.update(kw)
    return BacktestConfig(**base)


def sess(a, b) -> list:
    return list(CAL.sessions(pd.Timestamp(a), pd.Timestamp(b)))


def close_of(s) -> pd.Timestamp:
    return CAL.open_close(pd.Timestamp(s))[1]


# ------------------------------------------------------------------ bars
def daily_bars(sessions, closes, highs=None, lows=None, opens=None, volume=1e8) -> pd.DataFrame:
    c = np.asarray(closes, float)
    o = np.asarray(opens, float) if opens is not None else np.r_[c[0], c[:-1]]
    h = np.asarray(highs, float) if highs is not None else np.maximum(o, c) + 0.5
    l = np.asarray(lows, float) if lows is not None else np.minimum(o, c) - 0.5
    oc = [CAL.open_close(s) for s in sessions]
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": float(volume), "session": "DAY",
                         "known_at": pd.DatetimeIndex([y for _, y in oc])},
                        index=pd.DatetimeIndex([x for x, _ in oc], name="start"))


def intraday_bars(sessions, price: Callable[[pd.Timestamp], float], freq: str = "1m", volume=1e6,
                  overrides: Optional[dict] = None) -> pd.DataFrame:
    """Bars with o=h=l=c=price(bar END) except ``overrides`` {(session, 'HH:MM' start ET): (o,h,l,c)}."""
    td = pd.Timedelta(freq.replace("m", "min"))
    frames = []
    for s in sessions:
        o_, c_ = CAL.open_close(s)
        starts = pd.date_range(o_, c_, freq=td, inclusive="left")
        p = np.array([price(t + td) for t in starts], float)
        prev = np.r_[price(starts[0]), p[:-1]]
        df = pd.DataFrame({"open": prev, "high": np.maximum(prev, p), "low": np.minimum(prev, p), "close": p,
                           "volume": float(volume), "session": "RTH", "known_at": starts + td},
                          index=pd.DatetimeIndex(starts, name="start"))
        for (ss, hhmm), row in (overrides or {}).items():
            if pd.Timestamp(ss) == pd.Timestamp(s):
                t = et(s, hhmm)
                df.loc[t, ["open", "high", "low", "close"]] = row
        frames.append(df)
    return pd.concat(frames)


def daily_from(intra: pd.DataFrame) -> pd.DataFrame:
    sess_ = pd.DatetimeIndex(intra.index.tz_convert("America/New_York").tz_localize(None).normalize())
    g = intra.groupby(sess_)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()})
    oc = [CAL.open_close(s) for s in out.index]
    out.index = pd.DatetimeIndex([x for x, _ in oc], name="start")
    out["session"] = "DAY"
    out["known_at"] = pd.DatetimeIndex([y for _, y in oc])
    return out


# ------------------------------------------------------------------ options
def option_book(und: str, expiries, strikes, snaps, spot: Callable[[pd.Timestamp], float],
                vol: Callable[[pd.Timestamp, float, pd.Timestamp, float], float], r: float = 0.0,
                hs: Callable[[float], float] = lambda px: 0.05, fidelity: int = 0,
                price_override: Optional[Callable] = None, max_dte: int = 120):
    """Quote frames {symbol: DataFrame(ts index; bid, ask, fidelity)} and instruments.

    expiries: list of (expiry, settlement, exercise, root).  vol(ts, K, expiry, T) -> sigma.
    price_override(ts, inst, bid, ask) -> (bid, ask) for hand-made quotes."""
    snaps = pd.DatetimeIndex(snaps)
    S = np.array([spot(t) for t in snaps], float)
    optq, insts = {}, {}
    for exp, settle, exercise, root in expiries:
        exp = pd.Timestamp(exp)
        T = np.array([year_fraction(t, exp, settle) for t in snaps])
        dte = np.array([(exp - t.tz_convert("America/New_York").tz_localize(None).normalize()).days for t in snaps])
        live = (T > 0) & (dte <= max_dte)
        if not live.any():
            continue
        for K in strikes:
            for right in ("C", "P"):
                inst = Instrument.option(und, exp, float(K), Right(right), ExerciseStyle(exercise),
                                         SettlementType(settle), root=root)
                ts = snaps[live]
                sig = np.array([vol(t, float(K), exp, tt) for t, tt in zip(ts, T[live])], float)
                px = bsm_price(S[live], float(K), T[live], r, 0.0, sig, right == "C")
                h = np.array([hs(p) for p in px])
                bid = np.maximum(np.round(px - h, 2), 0.0)
                ask = np.maximum(np.round(px + h, 2), 0.01)
                ask = np.maximum(ask, bid + 0.01)
                if price_override is not None:
                    pairs = [price_override(t, inst, b, a) for t, b, a in zip(ts, bid, ask)]
                    bid = np.array([p[0] for p in pairs])
                    ask = np.array([p[1] for p in pairs])
                optq[inst.symbol] = pd.DataFrame({"bid": bid, "ask": ask, "fidelity": int(fidelity)},
                                                 index=pd.DatetimeIndex(ts, name="ts"))
                insts[inst.symbol] = inst
    return optq, insts


def minute_snaps(sessions, step_min: int = 1, start="09:30", end=None) -> pd.DatetimeIndex:
    out = []
    for s in sessions:
        o_, c_ = CAL.open_close(s)
        a = et(s, start) if start != "09:30" else o_
        out.append(pd.date_range(a + pd.Timedelta(minutes=step_min), c_, freq=f"{step_min}min"))
    return out[0].append(out[1:]) if len(out) > 1 else out[0]


def close_snaps(sessions) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([close_of(s) for s in sessions])


# ------------------------------------------------------------------ portal
class OEPortal(InMemoryPortal):
    """InMemoryPortal with configurable fidelity for options (to exercise the gates)."""

    def __init__(self, *a, option_fidelity: Fidelity = Fidelity.T0_QUOTE, event_kinds=(), **kw):
        super().__init__(*a, **kw)
        self.option_fidelity = option_fidelity
        self.event_kinds = set(event_kinds)

    def _chain_index(self, underlying):
        cache = self.__dict__.setdefault("_oe_chain_cache", {})
        if underlying not in cache:
            parts = []
            for sym, df in self._optq.items():
                inst = self.instrument(sym)
                if inst.underlying != underlying or not len(df):
                    continue
                parts.append(pd.DataFrame({"ts": df.index, "symbol": sym, "bid": df["bid"].to_numpy(float),
                                           "ask": df["ask"].to_numpy(float),
                                           "fidelity": df["fidelity"].to_numpy() if "fidelity" in df.columns
                                           else int(Fidelity.T0_QUOTE)}))
            big = pd.concat(parts, ignore_index=True).sort_values(["ts", "symbol"]) if parts else pd.DataFrame()
            cache[underlying] = big
        return cache[underlying]

    def option_chain(self, underlying, as_of, min_dte=0, max_dte=400):
        """Same semantics as InMemoryPortal.option_chain (latest snapshot per contract, then only
        the rows at the latest timestamp), vectorized for speed."""
        big = self._chain_index(underlying)
        if not len(big):
            return pd.DataFrame()
        a = pd.Timestamp(as_of)
        a = a.tz_localize("UTC") if a.tzinfo is None else a.tz_convert("UTC")
        d = big[big["ts"] <= a]
        if not len(d):
            return pd.DataFrame()
        d = d.groupby("symbol", sort=False).tail(1)
        d = d[d["ts"] == d["ts"].max()].copy()
        insts = [self.instrument(s) for s in d["symbol"]]
        d["underlying"] = underlying
        d["expiry"] = [i.expiry for i in insts]
        d["strike"] = [i.strike for i in insts]
        d["right"] = [i.right.value for i in insts]
        d["mid"] = 0.5 * (d["bid"] + d["ask"])
        d["multiplier"] = [i.multiplier for i in insts]
        return d.reset_index(drop=True)

    def has(self, symbol, kind, freq=None):
        if kind == "events" and symbol == "*" and freq in self.event_kinds:
            return True            # the event SOURCE exists even if no event is known yet (truncated data)
        return super().has(symbol, kind, freq)

    def fidelity_of(self, symbol, kind, freq=None):
        f = super().fidelity_of(symbol, kind, freq)
        if kind == "options" and f != Fidelity.NA:
            return self.option_fidelity
        return f


def make_portal(spec: dict, cut: Optional[pd.Timestamp] = None, **kw) -> OEPortal:
    """Build a portal from a spec dict; ``cut`` removes everything known after it."""
    bars = dict(spec.get("bars", {}))
    optq = dict(spec.get("option_quotes", {}))
    quotes = dict(spec.get("quotes", {}))
    events = list(spec.get("events", []))
    kw.setdefault("event_kinds", {e.kind for e in events})
    if cut is not None:
        cut = pd.Timestamp(cut)
        bars = {k: v[v["known_at"] <= cut] for k, v in bars.items()}
        optq = {k: v[v.index <= cut] for k, v in optq.items()}
        optq = {k: v for k, v in optq.items() if len(v)}
        quotes = {k: v[v.index <= cut] for k, v in quotes.items()}
        events = [e for e in events if pd.Timestamp(e.known_at) <= cut]
    return OEPortal(bars=bars, option_quotes=optq, quotes=quotes, instruments=spec.get("instruments", {}),
                    events=events, rf=spec.get("rf", 0.0), universes=spec.get("universes", {}),
                    actions=spec.get("actions", []), borrow=spec.get("borrow"), **kw)


def run(strategy, spec, start, end, cut=None, config=None, **kw):
    return run_backtest(strategy, make_portal(spec, cut, **kw), start, end, config or zero_cost(), raise_errors=True)


def orders(res) -> pd.DataFrame:
    o = res.diagnostics["orders"]
    return o if len(o) else pd.DataFrame(columns=["submitted_at", "tag", "quantity", "symbol", "status"])


def order_key(df: pd.DataFrame, upto) -> list:
    d = df[pd.to_datetime(df["submitted_at"], utc=True) <= pd.Timestamp(upto)]
    return [(str(r.submitted_at), r.tag, round(float(r.quantity), 6), r.symbol) for r in d.itertuples()]


def fills(res, tag_prefix: str = "") -> pd.DataFrame:
    f = res.fills
    if not len(f):
        return f
    return f[f["tag"].astype(str).str.startswith(tag_prefix)] if tag_prefix else f


def _aware(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("America/New_York").tz_convert("UTC") if t.tzinfo is None else t


def ev(kind, symbol, known_at, effective_at=None, **payload) -> MarketEvent:
    """Event (naive times are ET)."""
    return MarketEvent(kind=kind, symbol=symbol, known_at=_aware(known_at),
                       effective_at=None if effective_at is None else _aware(effective_at),
                       payload=payload, fidelity=Fidelity.T1_BAR)


def index_instrument(sym="SPX") -> Instrument:
    return Instrument(symbol=sym, asset_class=AssetClass.INDEX)
