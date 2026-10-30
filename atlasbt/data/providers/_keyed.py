"""Shared helpers for key-gated providers (activate only when env keys are set).

All key-gated providers are UNVERIFIED: written from the official API docs
(fetched 2026-10-28) but never exercised against the live API (no keys here).
They degrade gracefully: available() is False and fetch functions return
{'skipped': 'no API key'} without touching the network.
"""
from __future__ import annotations

import os

import pandas as pd

from atlasbt.core.types import Fidelity

from .. import frames
from ..store import ParquetStore, default_store


def env(*names: str) -> dict[str, str] | None:
    vals = {n: os.environ.get(n, "").strip() for n in names}
    return vals if all(vals.values()) else None


def skipped(provider: str) -> dict:
    return {"skipped": f"{provider}: no API key in environment"}


def store_intraday(source: str, symbol: str, freq: str, df: pd.DataFrame, fidelity: Fidelity,
                   store: ParquetStore | None = None, extra: dict | None = None) -> int:
    """df: index = bar start (tz-aware), columns open/high/low/close/volume[/vwap/trades]."""
    store = store or default_store()
    bars = frames.intraday_bars(df, freq)
    if bars.empty:
        return 0
    return store.write(f"bars_{freq}", source, symbol, bars, fidelity=int(fidelity), partition="year",
                       extra={"first_ts": str(bars.index[0]), "last_ts": str(bars.index[-1]), "raw": True,
                              "verified": False, **(extra or {})})


def store_quotes(source: str, symbol: str, q: pd.DataFrame, fidelity: Fidelity,
                 store: ParquetStore | None = None, extra: dict | None = None) -> int:
    """q: index ts (UTC), columns bid, ask, bid_size, ask_size."""
    store = store or default_store()
    q = q.sort_index()
    q.index.name = "ts"
    return store.write("quotes", source, symbol, q, fidelity=int(fidelity), partition="year",
                       extra={"verified": False, **(extra or {})})
