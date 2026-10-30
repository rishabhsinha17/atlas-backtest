"""Tolerant normalisers for the frames inside ``core.types.BacktestResult``.

``BacktestResult`` fixes the *names* of its frames but not their columns; the
engine is built concurrently. Analytics therefore reads them through these
normalisers, which accept the obvious variants and emit canonical columns.
The canonical schema (proposed to the integrator as a contract) is:

fills     : ts, order_id, symbol, asset_class, multiplier, quantity, price,
            commission, exchange_fees, regulatory_fees, spread_cost,
            impact_cost, fidelity (int tier), liquidity, note, reference_mid,
            adv (shares, optional), notional (= |qty|*price*multiplier)
            - also accepted: a list of core.types.Fill objects, an
              'instrument' column of Instrument objects, a 'costs' column of
              CostBreakdown objects, ts as the index.
trades    : symbol, entry_ts, exit_ts, quantity, entry_price, exit_price,
            pnl (net of all costs), r_multiple, mae, mfe, bars_held, multiplier
ledger    : ts, amount (signed cash, + = inflow), reason (str)
positions : ts, symbol, quantity, market_value (signed)
"""
from __future__ import annotations

from dataclasses import is_dataclass

import numpy as np
import pandas as pd

from atlasbt.core.types import Fidelity

COST_COLS = ["commission", "exchange_fees", "regulatory_fees", "spread_cost", "impact_cost"]
FILL_COLS = ["ts", "order_id", "symbol", "asset_class", "multiplier", "quantity", "price"] + COST_COLS + [
    "fidelity", "liquidity", "note", "reference_mid", "adv", "notional"]

_ALIASES = {
    "ts": ["ts", "time", "timestamp", "fill_ts", "datetime"],
    "quantity": ["quantity", "qty", "filled_qty", "shares"],
    "price": ["price", "fill_price", "px"],
    "symbol": ["symbol", "ticker"],
    "entry_ts": ["entry_ts", "entry_time", "open_ts", "entry_date", "entry"],
    "exit_ts": ["exit_ts", "exit_time", "close_ts", "exit_date", "exit"],
    "pnl": ["pnl", "net_pnl", "pnl_net", "profit"],
    "r_multiple": ["r_multiple", "R", "r", "r_mult", "rmult"],
    "amount": ["amount", "cash", "value", "delta_cash"],
    "reason": ["reason", "code", "type", "kind"],
    "market_value": ["market_value", "mv", "value", "notional"],
}


def _pick(df: pd.DataFrame, key: str):
    for c in _ALIASES.get(key, [key]):
        if c in df.columns:
            return df[c]
    return None


def _tier(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, Fidelity):
        return float(int(x))
    if isinstance(x, (int, np.integer, float, np.floating)):
        return float(x)
    s = str(x).split(".")[-1].upper()
    for f in Fidelity:
        if s == f.name or s == f.name.split("_")[0]:
            return float(int(f))
    return np.nan


def _to_frame(obj) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, (list, tuple)):
        if not obj:
            return pd.DataFrame()
        if is_dataclass(obj[0]):
            rows = []
            for f in obj:
                d = {k: getattr(f, k) for k in f.__dataclass_fields__}
                rows.append(d)
            return pd.DataFrame(rows)
        return pd.DataFrame(obj)
    raise TypeError(f"cannot convert {type(obj)} to DataFrame")


def normalize_fills(fills) -> pd.DataFrame:
    df = _to_frame(fills)
    if df.empty:
        return pd.DataFrame(columns=FILL_COLS)
    if "ts" not in df.columns and _pick(df, "ts") is None and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={df.index.name or "index": "ts"})
    out = pd.DataFrame(index=df.index)
    out["ts"] = _pick(df, "ts")
    out["order_id"] = df["order_id"] if "order_id" in df.columns else -1
    inst = df["instrument"] if "instrument" in df.columns else None
    sym = _pick(df, "symbol")
    if sym is None and inst is not None:
        sym = inst.map(lambda i: getattr(i, "symbol", str(i)))
    out["symbol"] = sym if sym is not None else ""
    if "asset_class" in df.columns:
        out["asset_class"] = df["asset_class"].map(lambda a: getattr(a, "value", a))
    elif inst is not None:
        out["asset_class"] = inst.map(lambda i: getattr(getattr(i, "asset_class", None), "value", None))
    else:
        out["asset_class"] = None
    if "multiplier" in df.columns:
        out["multiplier"] = df["multiplier"].astype(float)
    elif inst is not None:
        out["multiplier"] = inst.map(lambda i: float(getattr(i, "multiplier", 1.0)))
    else:
        out["multiplier"] = 1.0
    out["quantity"] = _pick(df, "quantity").astype(float)
    out["price"] = _pick(df, "price").astype(float)
    costs = df["costs"] if "costs" in df.columns else None
    for c in COST_COLS:
        if c in df.columns:
            out[c] = df[c].astype(float).fillna(0.0)
        elif costs is not None:
            out[c] = costs.map(lambda x, c=c: float(getattr(x, c, (x or {}).get(c, 0.0) if isinstance(x, dict) else 0.0)))
        else:
            out[c] = 0.0
    out["fidelity"] = df["fidelity"].map(_tier) if "fidelity" in df.columns else np.nan
    for c, default in (("liquidity", "TAKER"), ("note", ""), ("reference_mid", np.nan), ("adv", np.nan)):
        out[c] = df[c] if c in df.columns else default
    out["notional"] = (out["quantity"].abs() * out["price"].abs() * out["multiplier"]).astype(float)
    if out["ts"].notna().any():
        out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out.reset_index(drop=True)


def normalize_trades(trades) -> pd.DataFrame:
    df = _to_frame(trades)
    cols = ["symbol", "entry_ts", "exit_ts", "quantity", "entry_price", "exit_price", "pnl", "r_multiple", "mae",
            "mfe", "bars_held", "multiplier"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(index=df.index)
    for c in cols:
        v = _pick(df, c)
        out[c] = v if v is not None else np.nan
    for c in ("entry_ts", "exit_ts"):
        if out[c].notna().any():
            out[c] = pd.to_datetime(out[c], utc=True)
    for c in ("quantity", "entry_price", "exit_price", "pnl", "r_multiple", "mae", "mfe", "bars_held", "multiplier"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["multiplier"] = out["multiplier"].fillna(1.0)
    return out.reset_index(drop=True)


def normalize_ledger(ledger) -> pd.DataFrame:
    df = _to_frame(ledger)
    if df.empty:
        return pd.DataFrame(columns=["ts", "amount", "reason"])
    if _pick(df, "ts") is None and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={df.index.name or "index": "ts"})
    out = pd.DataFrame({"ts": _pick(df, "ts"), "amount": pd.to_numeric(_pick(df, "amount"), errors="coerce"),
                        "reason": _pick(df, "reason").astype(str).str.upper()})
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out.reset_index(drop=True)


def normalize_positions(positions) -> pd.DataFrame:
    df = _to_frame(positions)
    if df.empty:
        return pd.DataFrame(columns=["ts", "symbol", "quantity", "market_value"])
    if _pick(df, "ts") is None:
        if "date" in df.columns:
            df = df.rename(columns={"date": "ts"})
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": "ts"})
    out = pd.DataFrame({"ts": _pick(df, "ts"), "symbol": _pick(df, "symbol"),
                        "quantity": pd.to_numeric(_pick(df, "quantity"), errors="coerce")})
    mv = _pick(df, "market_value")
    if mv is None and {"mark", "quantity"} <= set(df.columns):
        mult = df["multiplier"] if "multiplier" in df.columns else 1.0
        mv = df["mark"] * df["quantity"] * mult
    out["market_value"] = pd.to_numeric(mv, errors="coerce") if mv is not None else np.nan
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out.reset_index(drop=True)


# ledger reason classification for cost-drag decomposition, by EXACT codes
# from atlasbt.core.schemas.LEDGER_REASONS. Fill-level cash flows (TRADE and the
# per-fill fees COMMISSION / EXCHANGE_FEE / REGULATORY_FEE, which the engine also
# posts to the ledger) are taken from the fills and excluded here ('fill_level')
# so they are never double counted.
LEDGER_BUCKETS = {
    "borrow": ("BORROW_FEE",),
    "financing_debit": ("MARGIN_INTEREST",),
    "financing_credit": ("CASH_INTEREST", "SHORT_REBATE"),
    "dividends": ("DIVIDEND", "DIVIDEND_IN_LIEU", "WITHHOLDING_TAX"),
    "fill_level": ("TRADE", "COMMISSION", "EXCHANGE_FEE", "REGULATORY_FEE"),
}
_REASON_TO_BUCKET = {code: b for b, codes in LEDGER_BUCKETS.items() for code in codes}


def classify_reason(reason: str) -> str:
    """Exact-code mapping; anything else (SPLIT_CASH_IN_LIEU, DELIST_CASHOUT,
    OPTION_*, DEPOSIT, WITHDRAWAL, unknown codes) -> 'other'."""
    return _REASON_TO_BUCKET.get(str(reason).strip().upper(), "other")
