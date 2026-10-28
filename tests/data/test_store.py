"""ParquetStore mechanics (synthetic frames)."""
import pandas as pd

from atlasbt.data.store import ParquetStore


def synthetic_bars(n=5, start="2023-12-28"):
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC", name="start")
    return pd.DataFrame({"close": range(n), "known_at": idx + pd.Timedelta(hours=6)}, index=idx)


def test_upsert_idempotent_and_partitioned(tmp_path):
    st = ParquetStore(tmp_path)
    df = synthetic_bars()
    st.write("bars_1m", "src", "AAA", df, partition="year", fidelity=1)
    st.write("bars_1m", "src", "AAA", df, partition="year", fidelity=1)           # same data again
    assert len(st.read("bars_1m", "src", "AAA")) == 5
    assert sorted(p.name for p in (tmp_path / "bars_1m" / "src" / "AAA").iterdir() if p.suffix == ".parquet") \
        == ["2023.parquet", "2024.parquet"]
    upd = synthetic_bars(3, "2024-01-01")
    upd["close"] = 99
    st.write("bars_1m", "src", "AAA", upd, partition="year")
    out = st.read("bars_1m", "src", "AAA")
    assert len(out) == 7 and (out.loc["2024-01-01":, "close"] == 99).all()   # 12-28..01-03
    m = st.meta("bars_1m", "src", "AAA")
    assert m["source"] == "src" and m["rows"] == 7 and "fetched_at" in m and "first_fetched_at" in m


def test_read_range_and_attrs(tmp_path):
    st = ParquetStore(tmp_path)
    st.write("bars_1d", "src", "BBB", synthetic_bars(), fidelity=1, adjusted=False)
    r = st.read("bars_1d", "src", "BBB", start="2023-12-29", end="2023-12-31")
    assert list(r["close"]) == [1, 2]
    assert r.attrs["source"] == "src" and r.attrs["fidelity"] == 1 and r.attrs["adjusted"] is False
    assert st.read("bars_1d", "src", "NOPE").empty
    assert st.symbols("bars_1d", "src") == ["BBB"]
