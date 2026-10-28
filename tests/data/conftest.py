"""Shared fixtures for atlasbt.data tests. No network: everything comes from real
responses recorded under tests/data/fixtures (provenance in docs/DATA.md) or from
synthetic frames (named synthetic_*) that only exercise mechanics."""
from pathlib import Path

import pandas as pd
import pytest

from atlasbt.data.portal import Portal
from atlasbt.data.store import ParquetStore

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def store(tmp_path):
    return ParquetStore(tmp_path / "store")


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
