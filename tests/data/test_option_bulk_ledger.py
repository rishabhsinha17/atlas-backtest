"""The DoltHub option bulk load rewrites year partitions; SQL dates after the mirror's last date
must leave the option_chain_done ledger so `--only options` refetches them."""
import pandas as pd

from atlasbt.data.providers import dolthub_bulk
from atlasbt.data.store import ParquetStore


def _ledger(store):
    return store.read("reference", "dolthub", "option_chain_done")


def test_sql_dates_after_bulk_are_forgotten(tmp_path):
    st = ParquetStore(tmp_path)
    st.write("reference", "dolthub", "option_chain_done", pd.DataFrame({
        "date": ["2026-01-27", "2026-01-28", "2026-11-06", "2026-11-06"],
        "symbol": ["SPY", "SPY", "SPY", "QQQ"], "rows": [10, 11, 12, 13]}), key=["date", "symbol"])
    n = dolthub_bulk._forget_sql_dates_after(st, ["SPY"], "2026-01-27")
    assert n == 2
    d = _ledger(st).sort_values(["date", "symbol"])
    assert list(zip(d["date"], d["symbol"])) == [("2026-01-27", "SPY"), ("2026-11-06", "QQQ")]


def test_all_rows_stale_removes_the_ledger(tmp_path):
    st = ParquetStore(tmp_path)
    st.write("reference", "dolthub", "option_chain_done",
             pd.DataFrame({"date": ["2026-02-01"], "symbol": ["SPY"], "rows": [1]}), key=["date", "symbol"])
    assert dolthub_bulk._forget_sql_dates_after(st, ["spy"], "2026-01-27") == 1
    assert not len(_ledger(st))


def test_no_last_date_or_ledger_is_a_noop(tmp_path):
    st = ParquetStore(tmp_path)
    assert dolthub_bulk._forget_sql_dates_after(st, ["SPY"], None) == 0
    assert dolthub_bulk._forget_sql_dates_after(st, ["SPY"], "2026-01-27") == 0
