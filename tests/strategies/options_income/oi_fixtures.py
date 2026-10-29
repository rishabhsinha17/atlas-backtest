"""SYNTHETIC fixtures for the options-income strategy tests (S126-S157).

Mechanics only (signal timing, combo fills at the next snapshot, structure
selection, exits, causality) -- never use them to evaluate a strategy.

* Underlying daily bars from a hand-built close path (open = previous close).
* An EOD option chain snapshot at every session close from ``chain_from`` on:
  monthly (3rd-Friday) expiries up to ``max_dte`` calendar days, $1 strikes,
  European BSM prices (r = q = 0) at a per-session vol, quoted iv/delta like
  the DoltHub feed, symmetric spread max($0.01, 1% of mid) on the cent grid.
* ``FastOptionPortal`` serves those snapshots per timestamp (the InMemory
  portal's generic option_chain scans every contract on every call).
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from atlasbt.core.clock import nyse
from atlasbt.core.types import AssetClass, Fidelity, Instrument, Right
from atlasbt.engine.backtester import run_backtest
from atlasbt.engine.config import BacktestConfig
from atlasbt.engine.testing import InMemoryPortal, synthetic_bars_from_rows
from atlasbt.execution.impact import SquareRootImpact
from atlasbt.execution.spread import SpreadConfig
from atlasbt.options.listing import monthly_expiration_day
from atlasbt.pricing.bsm import bsm_delta, bsm_price

CAL = nyse()
UND = "SPY"


def zero_cost(**kw) -> BacktestConfig:
    base = dict(fees="NO_FEES",
                spread=SpreadConfig(bucket_floors=((math.inf, 0.0),), default_spread=1e-12, max_spread=1e-12,
                                    tod_open=(), tod_close=(), tick_floor=False),
                impact=SquareRootImpact(k=0.0), debug=True)
    base.update(kw)
    return BacktestConfig(**base)


def sessions(n: int, start: str = "2021-01-04") -> list[pd.Timestamp]:
    s = pd.Timestamp(start)
    return list(CAL.sessions(s, s + pd.Timedelta(days=int(n * 1.6) + 30)))[:n]


def close_ts(d) -> pd.Timestamp:
    return CAL.open_close(pd.Timestamp(d))[1]


def open_ts(d) -> pd.Timestamp:
    return CAL.open_close(pd.Timestamp(d))[0]


def monthly_expiries(first: pd.Timestamp, last: pd.Timestamp) -> list[pd.Timestamp]:
    out = []
    m = pd.Timestamp(first.year, first.month, 1)
    while m <= last + pd.Timedelta(days=200):
        d = pd.Timestamp(monthly_expiration_day(m.year, m.month)).normalize()
        if not CAL.is_session(d):
            d = CAL.prev_session(d)
        out.append(d)
        m = m + pd.DateOffset(months=1)
    return out


def bars_frame(closes: Sequence[float], ss: Sequence[pd.Timestamp], rng: float = 0.003,
               opens: Optional[Sequence[float]] = None) -> pd.DataFrame:
    rows, prev = [], float(closes[0])
    for i, c in enumerate(closes):
        c = round(float(c), 2)
        o = round(float(opens[i]), 2) if opens is not None else round(prev, 2)
        rows.append((o, round(max(o, c) * (1 + rng), 2), round(min(o, c) * (1 - rng), 2), c, 5_000_000.0))
        prev = c
    return synthetic_bars_from_rows(rows, "1d", session_dates=list(ss))


class FastOptionPortal(InMemoryPortal):
    """InMemoryPortal + vectorized option data: per-timestamp chain snapshots and per-contract quote arrays
    (the InMemory portal's generic option_chain scans every contract on every call)."""

    def __init__(self, *a, chain: Optional[pd.DataFrame] = None, **kw):
        super().__init__(*a, **kw)
        chain = chain if chain is not None else pd.DataFrame()
        self.snap = {ts: g.reset_index(drop=True) for ts, g in chain.groupby("ts")} if len(chain) else {}
        self.snap_ts = pd.DatetimeIndex(sorted(self.snap.keys()))
        self.q = {}
        if len(chain):
            ts_ns = pd.DatetimeIndex(chain["ts"]).as_unit("ns").asi8
            for sym, idx in chain.groupby("symbol").indices.items():
                self.q[sym] = (ts_ns[idx], chain["bid"].to_numpy(float)[idx], chain["ask"].to_numpy(float)[idx],
                               chain["iv"].to_numpy(float)[idx], chain["delta"].to_numpy(float)[idx],
                               chain["underlying_price"].to_numpy(float)[idx])
        self.unds = set(chain["underlying"]) if len(chain) else set()

    @staticmethod
    def _ns(t) -> int:
        a = pd.Timestamp(t)
        a = a.tz_localize("UTC") if a.tzinfo is None else a.tz_convert("UTC")
        return int(a.value)

    def has(self, symbol, kind, freq=None):
        if kind == "options":
            return symbol in self.unds
        return super().has(symbol, kind, freq)

    def option_chain(self, underlying, as_of, min_dte: int = 0, max_dte: int = 400):
        a = pd.Timestamp(self._ns(as_of), tz="UTC")
        i = int(self.snap_ts.searchsorted(a, side="right")) - 1
        if i < 0:
            return pd.DataFrame()
        df = self.snap[self.snap_ts[i]]
        out = df[df["underlying"] == underlying].reset_index(drop=True)
        out.attrs.update(source="synthetic-test-fixture", synthetic=False, fidelity=int(Fidelity.T2_PROXY))
        return out

    def option_quotes(self, instrument, start, end, as_of):
        cols = ["bid", "ask", "iv", "delta", "underlying_price", "fidelity"]
        z = self.q.get(instrument.symbol)
        if z is None:
            return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz="UTC", name="ts"))
        s, e, a = self._ns(start), self._ns(end), self._ns(as_of)
        m = (z[0] > s) & (z[0] <= e) & (z[0] <= a)
        idx = pd.DatetimeIndex(pd.to_datetime(z[0][m], utc=True), name="ts")
        return pd.DataFrame({"bid": z[1][m], "ask": z[2][m], "iv": z[3][m], "delta": z[4][m],
                             "underlying_price": z[5][m], "fidelity": int(Fidelity.T2_PROXY)}, index=idx)

    def option_quote(self, instrument, as_of):
        from atlasbt.core.types import OptionQuote
        z = self.q.get(instrument.symbol)
        if z is None:
            return None
        a = self._ns(as_of)
        j = int(np.searchsorted(z[0], a, side="right")) - 1
        if j < 0:
            return None
        return OptionQuote(instrument=instrument, ts=pd.Timestamp(int(z[0][j]), tz="UTC"), bid=float(z[1][j]),
                           ask=float(z[2][j]), underlying_price=float(z[5][j]), iv=float(z[3][j]),
                           delta=float(z[4][j]), fidelity=Fidelity.T2_PROXY)

    def instrument(self, symbol):
        inst = self._instruments.get(symbol)
        if inst is not None:
            return inst
        if len(symbol) > 15 and symbol[12] in "CP":
            root, exp, right, k = parse_occ(symbol)
            return Instrument.option(root, exp, k, Right(right))
        return Instrument(symbol=symbol, asset_class=AssetClass.STOCK)


def weekly_expiries(first: pd.Timestamp, last: pd.Timestamp) -> list[pd.Timestamp]:
    """Every Friday (holiday -> prior session) plus the monthlies: SPY-like weeklies."""
    out = set(monthly_expiries(first, last))
    for f in pd.date_range(first, last + pd.Timedelta(days=200), freq="W-FRI"):
        d = f.normalize()
        if not CAL.is_session(d):
            d = CAL.prev_session(d)
        out.add(d)
    return sorted(out)


def option_chain_frame(und: str, ss: Sequence[pd.Timestamp], closes: Sequence[float], chain_from: int,
                       vol, strike_lo: float, strike_hi: float, step: float = 1.0, max_dte: int = 70,
                       spread_frac: float = 0.01, chain_to: Optional[int] = None, expiries=None,
                       term_slope: float = 0.0, skew: float = 0.0) -> pd.DataFrame:
    """Long frame of EOD snapshots (one per session close in [chain_from, chain_to)).
    Per-contract vol = vol_session x (1 + term_slope (DTE-30)/30) x (1 - skew ln(K/S)), floored at 1%."""
    vols = np.broadcast_to(np.asarray(vol, float), (len(ss),))
    exps = expiries if expiries is not None else weekly_expiries(ss[0], ss[-1])
    strikes = np.round(np.arange(strike_lo, strike_hi + 1e-9, step), 2)
    last = len(ss) if chain_to is None else chain_to
    cols = {k: [] for k in ("ts", "expiry", "strike", "right", "bid", "ask", "iv", "delta", "underlying_price")}
    exps_arr = pd.DatetimeIndex(exps)
    for i in range(chain_from, last):
        d, S, sig = ss[i], round(float(closes[i]), 2), float(vols[i])
        ts = close_ts(d)
        dtes = (exps_arr - d).days.to_numpy()
        live = exps_arr[(dtes >= 0) & (dtes <= max_dte)]
        if not len(live):
            continue
        ld = (live - d).days.to_numpy()
        E = np.repeat(np.asarray(live), len(strikes) * 2)
        T = np.repeat(np.maximum(ld, 0.5) / 365.0, len(strikes) * 2)
        K = np.tile(np.concatenate([strikes, strikes]), len(live))
        cp = np.tile(np.r_[np.ones(len(strikes), bool), np.zeros(len(strikes), bool)], len(live))
        D = np.repeat(ld, len(strikes) * 2)
        sv = np.maximum(sig * (1 + term_slope * (D - 30) / 30.0) * (1 - skew * np.log(K / S)), 0.01)
        px = np.asarray(bsm_price(S, K, T, 0.0, 0.0, sv, cp), float)
        dl = np.asarray(bsm_delta(S, K, T, 0.0, 0.0, sv, cp), float)
        hs = np.maximum(0.01, spread_frac * px)
        bid = np.round(np.maximum(px - hs, 0.0), 2)
        ask = np.round(px + hs, 2)
        bid = np.where(bid < 0.01, 0.0, bid)
        ask = np.maximum(ask, bid + 0.01)
        nrow = len(K)
        cols["ts"].append(np.full(nrow, ts.value, dtype=np.int64))
        cols["expiry"].append(E)
        cols["strike"].append(K)
        cols["right"].append(np.where(cp, "C", "P"))
        cols["bid"].append(bid)
        cols["ask"].append(ask)
        cols["iv"].append(sv)
        cols["delta"].append(dl)
        cols["underlying_price"].append(np.full(nrow, S))
    if not cols["ts"]:
        return pd.DataFrame()
    df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["expiry"] = pd.to_datetime(df["expiry"])
    df["underlying"] = und
    df["mid"] = np.round(0.5 * (df["bid"] + df["ask"]), 4)
    ek = pd.Series(df["expiry"].dt.strftime("%y%m%d").to_numpy())
    kk = pd.Series((df["strike"].to_numpy() * 1000).round().astype(np.int64)).map("{:08d}".format)
    df["symbol"] = (f"{und:<6}" + ek + pd.Series(df["right"].to_numpy()) + kk).to_numpy()
    df["exercise"] = "AMERICAN"
    df["settlement"] = "PHYSICAL"
    df["multiplier"] = 100.0
    df["fidelity"] = int(Fidelity.T2_PROXY)
    return df


def make_portal(closes, ss, *, und: str = UND, chain_from: int = 0, vol=0.20, step: float = 1.0,
                band: float = 0.20, max_dte: int = 70, extra_bars: Optional[dict] = None,
                cut: Optional[int] = None, chain_to: Optional[int] = None, expiries=None,
                events=None, actions=None, term_slope: float = 0.0, skew: float = 0.0) -> FastOptionPortal:
    """Portal over the first ``cut`` sessions (all if None): truncated views for lookahead tests."""
    n = len(ss) if cut is None else cut
    closes = list(closes)[:n]
    ss = list(ss)[:n]
    ref = float(np.median(closes[chain_from:])) if chain_from < len(closes) else float(closes[-1])
    lo = math.floor(ref * (1 - band) / step) * step
    hi = math.ceil(ref * (1 + band) / step) * step
    chain = option_chain_frame(und, ss, closes, chain_from, vol, lo, hi, step, max_dte,
                               chain_to=min(chain_to, n) if chain_to is not None else None, expiries=expiries,
                               term_slope=term_slope, skew=skew)
    bars = {(und, "1d"): bars_frame(closes, ss)}
    insts = {und: Instrument(und, AssetClass.ETF)}
    for sym, cl in (extra_bars or {}).items():
        bars[(sym, "1d")] = bars_frame(list(cl)[:n], ss, rng=0.0)
        insts[sym] = Instrument(sym, AssetClass.INDEX)
    return FastOptionPortal(bars=bars, instruments=insts, chain=chain, events=events or [], actions=actions or [])


def run(strategy, portal, ss, start_i: int = 0, end_i: Optional[int] = None, **cfg):
    end = ss[-1] if end_i is None else ss[end_i]
    return run_backtest(strategy, portal, ss[start_i], end, zero_cost(**cfg), raise_errors=True)


def combo_orders(res) -> pd.DataFrame:
    o = res.diagnostics["orders"]
    if o is None or not len(o):
        return pd.DataFrame(columns=["order_id", "type", "tag", "submitted_at", "quantity", "status"])
    return o[o["type"] == "COMBO"].reset_index(drop=True)


def option_fills(res) -> pd.DataFrame:
    f = res.fills
    if f is None or not len(f):
        return pd.DataFrame(columns=["ts", "symbol", "quantity", "price", "order_id", "asset_class"])
    return f[f["asset_class"] == "OPT"].reset_index(drop=True)


def parse_occ(sym: str):
    """(root, expiry, right, strike) from an OCC-style symbol."""
    root = sym[:6].strip()
    exp = pd.Timestamp("20" + sym[6:12])
    right = sym[12]
    strike = int(sym[13:]) / 1000.0
    return root, exp, right, strike


def legs_of_order(res, order_id: int) -> list[tuple[str, float, float]]:
    """[(right, strike, signed qty)] of the option fills of an order, sorted by (right, strike)."""
    f = option_fills(res)
    f = f[f["order_id"] == order_id]
    out = []
    for s, q in zip(f["symbol"], f["quantity"]):
        _, _, r, k = parse_occ(s)
        out.append((r, k, float(q)))
    return sorted(out)


def orders_until(res, ts) -> pd.DataFrame:
    """Orders submitted at or before ``ts`` (for lookahead comparison)."""
    o = res.diagnostics["orders"]
    if o is None or not len(o):
        return pd.DataFrame()
    o = o[pd.to_datetime(o["submitted_at"], utc=True) <= ts]
    cols = ["symbol", "type", "quantity", "limit_price", "stop_price", "tif", "tag", "submitted_at"]
    return o[cols].reset_index(drop=True)


def fills_until(res, ts) -> pd.DataFrame:
    f = res.fills
    if f is None or not len(f):
        return pd.DataFrame()
    f = f[pd.to_datetime(f["ts"], utc=True) <= ts]
    return f[["ts", "symbol", "quantity", "price", "tag"]].reset_index(drop=True)
