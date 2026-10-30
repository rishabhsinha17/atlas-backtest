"""Fixtures for indicator tests."""
import pytest

from ind_data import synthetic_intraday, synthetic_ohlcv


@pytest.fixture
def ohlcv():
    return synthetic_ohlcv()


@pytest.fixture
def intraday():
    return synthetic_intraday()
