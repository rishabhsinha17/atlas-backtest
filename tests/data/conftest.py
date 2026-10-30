"""Shared fixtures for atlasbt.data tests. No network: everything comes from real
responses recorded under tests/data/fixtures (provenance in docs/DATA.md) or from
synthetic frames (named synthetic_*) that only exercise mechanics."""
from pathlib import Path

import pandas as pd
import pytest

from atlasbt.data.portal import Portal
from atlasbt.data.store import ParquetStore

FIX = Path(__file__).parent / "fixtures"


# synthetic_ identity evidence for the tmp stores: ticker -> fake CIK, "as_of" monthly 1990-2026
# (the real store builds it from the ZipLime 8-K corpus; see atlasbt/data/identity.py)
SYNTHETIC_CIKS = {"AAPL": 320193, "TSLA": 1318605, "X": 900001, "SYN": 900002, "HALF": 900003,
                  "A": 900004, "B": 900005, "OLD": 900006, "NEW": 900007}


def seed_identity(store, mapping=None):
    mapping = mapping or SYNTHETIC_CIKS
    months = pd.date_range("1960-01-01", "2026-12-01", freq="MS")
    ev = pd.concat([pd.DataFrame({"ticker": t, "cik": c, "conf": 0, "date": months}) for t, c in mapping.items()])
    store.write("reference", "identity", "evidence", ev.astype({"conf": "int8"}), key=["ticker", "cik", "conf", "date"],
                replace=True)
    tm = pd.DataFrame({"cik": list(mapping.values()), "name": list(mapping), "ticker": list(mapping), "exchange": "X"})
    store.write("reference", "sec", "ticker_map", tm, key=["ticker", "cik"], replace=True)


@pytest.fixture
def store(tmp_path):
    st = ParquetStore(tmp_path / "store")
    seed_identity(st)
    return st


@pytest.fixture
def portal(store):
    return Portal(store)


def yahoo_frame(sym: str) -> pd.DataFrame:
    df = pd.read_csv(FIX / f"yahoo_daily_{sym}.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    return df


def load_yahoo(store, sym: str):
    from atlasbt.data.providers import yahoo
    bars, acts, _ = yahoo.parse_daily(sym, yahoo_frame(sym))
    store.write("bars_1d", "yahoo", sym, bars, fidelity=1)
    if len(acts):
        store.write("actions", "yahoo", sym, acts, key=["ex_date", "type"], fidelity=2)
    return bars, acts


def utc(s):
    return pd.Timestamp(s, tz="UTC")
