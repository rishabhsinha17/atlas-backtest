"""FRED via the official API (FRED_API_KEY): parsed like fredgraph.csv, cached under the key-free
fredgraph URL, and the key never reaches the raw cache, the store metadata or error messages."""
import json

import pandas as pd
import pytest

from atlasbt.data import config
from atlasbt.data.http import FetchError, HttpClient
from atlasbt.data.providers import fred
from atlasbt.data.store import ParquetStore

KEY = "k3y-should-never-be-written-0123456789"


def _payload():
    return json.dumps({"observations": [
        {"date": "2024-11-08", "value": "4.42"},
        {"date": "2024-11-11", "value": "."},          # Veterans Day: no H.15 value
        {"date": "2024-11-12", "value": "4.44"},
    ]}).encode()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(config, "offline", lambda: False)
    monkeypatch.setenv("FRED_API_KEY", KEY)
    import atlasbt.data.http as http
    monkeypatch.setattr(http, "_CLIENT", None, raising=False)
    monkeypatch.setattr(fred, "client", lambda: HttpClient(raw_root=tmp_path / "raw"))
    return tmp_path


def test_api_path_parses_caches_without_key(env, monkeypatch):
    calls = []

    def fake_get(self, url, params=None, **kw):
        calls.append((url, dict(params or {}), kw.get("cache", True)))
        assert url == fred.API_URL
        return _payload()

    monkeypatch.setattr(HttpClient, "get", fake_get)
    store = ParquetStore(env / "store")
    res = fred.fetch_series("DTB3", store=store)
    assert res["origin"] == "fred_api" and res["rows"] == 2
    assert calls == [(fred.API_URL, {"series_id": "DTB3", "api_key": KEY, "file_type": "json"}, False)]
    df = store.read("rates", "fred", "DTB3")
    assert list(df["value"]) == [4.42, 4.44]
    # day-d rate known 16:30 ET next Fed business day; 2024-11-11 is a Fed holiday
    assert pd.Timestamp(df["known_at"].iloc[0]).tz_convert("America/New_York") == \
        pd.Timestamp("2024-11-12 16:30", tz="America/New_York")
    # the CSV is reusable offline under the fredgraph URL, and the key appears nowhere on disk
    assert HttpClient(raw_root=env / "raw").cached(fred.URL, params={"id": "DTB3"}) is not None
    for p in env.rglob("*"):
        if p.is_file() and p.suffix in (".json", ".bin", ".gz", ".parquet"):
            assert KEY.encode() not in p.read_bytes(), p


def test_api_error_is_sanitized_and_falls_back(env, monkeypatch):
    def boom(self, url, params=None, **kw):
        if url == fred.API_URL:
            raise RuntimeError(f"connection reset for {url}?api_key={params['api_key']}")
        raise FetchError(url, None, "fredgraph unreachable")

    monkeypatch.setattr(HttpClient, "get", boom)
    with pytest.raises(FetchError) as ei:
        fred._fetch_api("DTB3", KEY)
    assert KEY not in str(ei.value)
    # no fredgraph, no raw cache, no data_raw copy -> the fallback chain raises (no silent empty data)
    monkeypatch.setattr(config, "DATA_RAW", env / "nope")
    with pytest.raises(FetchError):
        fred.fetch_series("DTB3", store=ParquetStore(env / "store"))
