"""SYNTHETIC fixtures for factors/statarb strategy mechanics tests (not for evaluation)."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from atlasbt.core.clock import nyse
from atlasbt.engine import BacktestConfig
from atlasbt.engine.backtester import run_backtest
from atlasbt.engine.testing import InMemoryPortal
from atlasbt.execution.impact import SquareRootImpact
from atlasbt.execution.spread import SpreadConfig

CAL = nyse()
NAV0 = 1_000_000.0
SECTORS5 = ["Information Technology", "Financials", "Health Care", "Energy", "Industrials"]


def zero_cost(**kw) -> BacktestConfig:
    base = dict(fees="NO_FEES",
                spread=SpreadConfig(bucket_floors=((math.inf, 0.0),), default_spread=1e-12, tod_open=(),
                                    tod_close=(), tick_floor=False, max_spread=0.0),
                impact=SquareRootImpact(k=0.0), debug=False)
    base.update(kw)
    return BacktestConfig(**base)


def sessions(start, end) -> list[pd.Timestamp]:
    return list(CAL.sessions(pd.Timestamp(start), pd.Timestamp(end)))


def close_ts(session) -> pd.Timestamp:
    return CAL.open_close(pd.Timestamp(session))[1]


def open_ts(session) -> pd.Timestamp:
    return CAL.open_close(pd.Timestamp(session))[0]


def next_session(s) -> pd.Timestamp:
    return CAL.next_session(pd.Timestamp(s))


def bars_from_closes(closes: pd.Series, volume: float = 5_000_000.0, opens: Optional[pd.Series] = None) -> pd.DataFrame:
    """Daily bars: open = previous close (unless given), high/low +/-0.2% around open/close."""
    c = closes.astype(float)
    o = opens.astype(float) if opens is not None else c.shift(1).fillna(c.iloc[0])
    hi = np.maximum(o, c) * 1.002
    lo = np.minimum(o, c) * 0.998
    oc = [CAL.open_close(s) for s in c.index]
    idx = pd.DatetimeIndex([a for a, _ in oc], name="start")
    return pd.DataFrame({"open": o.to_numpy(), "high": hi.to_numpy(), "low": lo.to_numpy(), "close": c.to_numpy(),
                         "volume": float(volume), "session": "DAY",
                         "known_at": pd.DatetimeIndex([b for _, b in oc])}, index=idx)


def closes_from_returns(dates, rets, p0: float = 100.0) -> pd.Series:
    r = np.asarray(rets, float)
    return pd.Series(p0 * np.cumprod(1.0 + r), index=pd.DatetimeIndex(dates))


class SMPortal(InMemoryPortal):
    """InMemoryPortal + a security master frame (sector / sub-industry / exchange)."""

    def __init__(self, *a, master: Optional[pd.DataFrame] = None, **kw):
        super().__init__(*a, **kw)
        self._master = master

    def security_master(self, as_of):
        if self._master is None:
            return super().security_master(as_of)
        return self._master.copy()

    def fundamentals_history(self, symbols, fields, as_of, start=None):
        """Every filed value with known_at <= as_of (and period_end >= start), like the real Portal."""
        cols = ["symbol", "field", "value", "period_end", "known_at", "source"]
        f = self._fund
        if f is None or not len(f):
            return pd.DataFrame(columns=cols)
        a = pd.Timestamp(as_of)
        a = a.tz_localize("UTC") if a.tzinfo is None else a.tz_convert("UTC")
        m = f["field"].isin(list(fields)) & (pd.to_datetime(f["known_at"], utc=True) <= a)
        if symbols is not None:
            m &= f["symbol"].isin(list(symbols))
        if start is not None:
            m &= pd.to_datetime(f["period_end"]) >= pd.Timestamp(start)
        return f[m].sort_values(["symbol", "field", "period_end", "known_at"]).reset_index(drop=True)


def master(symbols, sectors=None, sub=None, exchange="NYSE", sic=None) -> pd.DataFrame:
    """Security master with the REAL Portal column names (sector = SIC->GICS-like, gics_*_current,
    exchange, sic). Default SIC: one 3-digit group per sub-industry label."""
    n = len(symbols)
    sec = sectors if sectors is not None else [SECTORS5[i % 5] for i in range(n)]
    sb = sub if sub is not None else [f"SUB{i % 10}" for i in range(n)]
    if sic is None:
        uniq = list(dict.fromkeys(sb))
        sic = [2000 + 10 * uniq.index(x) for x in sb]
    return pd.DataFrame({
        "symbol": list(symbols), "sector": sec, "gics_sector_current": sec, "gics_sub_industry_current": sb,
        "exchange": exchange if isinstance(exchange, list) else [exchange] * n, "sic": sic,
    })


def fund_row(sym, field, value, period_end, known_at) -> dict:
    return {"symbol": sym, "field": field, "value": float(value), "period_end": pd.Timestamp(period_end),
            "known_at": pd.Timestamp(known_at, tz="UTC") if pd.Timestamp(known_at).tzinfo is None
            else pd.Timestamp(known_at).tz_convert("UTC"), "source": "synthetic"}


def build_portal(closes: dict, *, universe=None, master_df=None, fundamentals=None, actions=None, borrow=None,
                 cut: Optional[pd.Timestamp] = None, volumes: Optional[dict] = None, opens: Optional[dict] = None,
                 universe_name: str = "SP500") -> SMPortal:
    """Portal over close paths. ``cut`` truncates everything known after it (lookahead tests)."""
    bars = {}
    for s, c in closes.items():
        b = bars_from_closes(c, (volumes or {}).get(s, 5_000_000.0), (opens or {}).get(s))
        if cut is not None:
            b = b[b["known_at"] <= cut]
        bars[(s, "1d")] = b
    f = None
    if fundamentals is not None and len(fundamentals):
        f = pd.DataFrame(fundamentals) if not isinstance(fundamentals, pd.DataFrame) else fundamentals.copy()
        f["known_at"] = pd.to_datetime(f["known_at"], utc=True)
        f["period_end"] = pd.to_datetime(f["period_end"])
        if cut is not None:
            f = f[f["known_at"] <= cut]
    acts = list(actions or [])
    if cut is not None:
        acts = [a for a in acts if pd.Timestamp(a.known_at) <= cut]
    uni = {universe_name: list(universe if universe is not None else closes)}
    return SMPortal(bars=bars, fundamentals=f, actions=acts, borrow=borrow, universes=uni, master=master_df)


def run(strategy, portal, start, end, **cfg):
    return run_backtest(strategy, portal, start, end, zero_cost(**cfg), raise_errors=True)


def orders(res) -> pd.DataFrame:
    o = res.diagnostics["orders"]
    if len(o) == 0:
        return pd.DataFrame(columns=["symbol", "quantity", "type", "tag", "status", "submitted_at"])
    return o[o["status"] != "REJECTED"]


def order_key(res, upto: pd.Timestamp) -> list[tuple]:
    o = res.diagnostics["orders"]
    if len(o) == 0:
        return []
    o = o[o["submitted_at"] <= upto]
    return sorted((str(r.symbol), float(r.quantity), str(r.type), str(r.tag), pd.Timestamp(r.submitted_at))
                  for r in o.itertuples())


def fills_at(res, ts: pd.Timestamp) -> pd.DataFrame:
    f = res.fills
    return f[f["ts"] == ts] if len(f) else f


def gbm(n, vol, seed, drift=0.0) -> np.ndarray:
    return np.random.default_rng(seed).normal(drift, vol, n)
