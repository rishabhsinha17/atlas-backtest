"""Corporate-action cross-check (atlasbt.data.ca_crosscheck) on a SYNTHETIC tmp store.

synthetic_ data only exercises the matching rules: split agreement / ratio mismatch / missing in
either source, the independent DoltHub price-gap test, dividend amount and split-basis
matching, the identity guard (ticker reused by another security) and the double-encoded
spin-off detector. No network, no real cache."""
import numpy as np
import pandas as pd
import pytest

from atlasbt.core.clock import nyse
from atlasbt.data.ca_crosscheck import CrossChecker, flags, summarize, write_report
from atlasbt.data.store import ParquetStore

CAL = nyse()


def synthetic_bars(start, end, closes_fn):
    ss = CAL.sessions(pd.Timestamp(start), pd.Timestamp(end))
    rows = []
    for d in ss:
        o, c = CAL.open_close(d)
        px_open, px_close = closes_fn(d)
        rows.append({"start": o, "open": px_open, "high": max(px_open, px_close) * 1.001,
                     "low": min(px_open, px_close) * 0.999, "close": px_close, "volume": 1e6,
                     "session": "DAY", "known_at": c})
    return pd.DataFrame(rows).set_index("start")


def synthetic_actions(sym, rows):
    out = []
    for typ, ex, ratio, amount in rows:
        out.append({"symbol": sym, "type": typ, "ex_date": pd.Timestamp(ex), "known_at": pd.Timestamp(ex, tz="UTC"),
                    "ratio": ratio, "amount": amount, "pay_date": pd.NaT, "new_symbol": None, "fidelity": 2,
                    "source": "yahoo", "yahoo_amount": amount, "known_at_method": "synthetic", "note": ""})
    return pd.DataFrame(out)


@pytest.fixture
def store(tmp_path):
    st = ParquetStore(tmp_path)
    split_day = pd.Timestamp("2020-08-31")

    def px(d):          # 4:1 split on 2020-08-31: raw price drops from 400 to 100
        return (100.0, 100.0) if d >= split_day else (400.0, 400.0)
    for src in ("yahoo", "dolthub"):
        st.write("bars_1d", src, "SYN", synthetic_bars("2020-06-01", "2020-12-31", px))
    # Yahoo: the real 4:1 split, a phantom 2:1 split (no gap in DoltHub prices), dividends
    st.write("actions", "yahoo", "SYN", synthetic_actions("SYN", [
        ("SPLIT", "2020-08-31", 4.0, 0.0),
        ("SPLIT", "2020-10-01", 2.0, 0.0),
        ("CASH_DIVIDEND", "2020-07-01", 1.0, 0.80),     # pre-split as-traded amount
        ("CASH_DIVIDEND", "2020-09-15", 1.0, 0.20),
        ("CASH_DIVIDEND", "2020-12-01", 1.0, 0.25),     # DoltHub says 0.30
    ]))
    # ticker reused: Yahoo 'REU' is a $10 company, DoltHub 'REU' a $50 one
    st.write("bars_1d", "yahoo", "REU", synthetic_bars("2020-06-01", "2020-12-31", lambda d: (10.0, 10.0)))
    st.write("bars_1d", "dolthub", "REU", synthetic_bars("2020-06-01", "2020-12-31", lambda d: (50.0, 50.0)))
    st.write("actions", "yahoo", "REU", synthetic_actions("REU", [("CASH_DIVIDEND", "2020-09-01", 1.0, 0.10)]))
    st.write("reference", "dolthub", "SPLIT", pd.DataFrame({
        "act_symbol": ["SYN", "SYN", "SYN", "REU"], "ex_date": ["2020-08-20", "2020-09-01", "2020-11-02", "2020-09-01"],
        "to_factor": [4.0, 4.0, 3.0, 2.0], "for_factor": [1.0, 1.0, 1.0, 1.0]}), key=["act_symbol", "ex_date"])
    st.write("reference", "dolthub", "DIVIDEND", pd.DataFrame({
        "act_symbol": ["SYN", "SYN", "SYN", "REU"], "ex_date": ["2020-07-01", "2020-09-15", "2020-12-01", "2020-09-01"],
        "amount": [0.20, 0.20, 0.30, 0.10]}), key=["act_symbol", "ex_date"])
    return st


def by_key(rep):
    return {(r.symbol, r.kind, r.ex_date, r.raw_status): r for r in rep.itertuples()}


def test_split_and_dividend_rules(store):
    rep = CrossChecker(store).run(["SYN", "REU"])
    k = by_key(rep)
    # real 4:1 split: DoltHub has it 1 day later (within tolerance), price gap 0.25 confirms
    r = k[("SYN", "SPLIT", "2020-08-31", "agree")]
    assert r.status == "agree" and r.price_confirms is True and r.dolthub_price_gap == pytest.approx(0.25)
    # DoltHub repeats the 4:1 split on its announcement date: not counted against Yahoo
    assert [x.status for x in rep.itertuples() if x.ex_date == "2020-08-20"] == ["dolthub_extra_row"]
    # phantom 2:1 split: missing in DoltHub AND contradicted by DoltHub's prices -> error
    r = [x for x in rep.itertuples() if x.ex_date == "2020-10-01"][0]
    assert r.status == "price_contradicts" and r.severity == "error" and r.price_confirms is False
    # DoltHub-only 3:1 split without a gap: missing_in_yahoo, warn (Yahoo is the served source)
    r = [x for x in rep.itertuples() if x.ex_date == "2020-11-02"][0]
    assert r.status == "missing_in_yahoo" and r.price_confirms is False
    # dividends: 0.80 pre-split vs DoltHub 0.20 = split-adjusted basis -> flagged, not a mismatch
    assert k[("SYN", "DIVIDEND", "2020-07-01", "amount_split_basis")].severity == "warn"
    assert k[("SYN", "DIVIDEND", "2020-09-15", "agree")].status == "agree"
    assert k[("SYN", "DIVIDEND", "2020-12-01", "amount_mismatch")].severity == "warn"
    # reused ticker: identity guard excludes the events from the rate
    assert set(rep[rep.symbol == "REU"].status) == {"identity_unconfirmed"}


def test_summary_rates_and_flags(store, tmp_path):
    rep = CrossChecker(store).run(["SYN", "REU"])
    s = summarize(rep)
    sp, dv = s["SPLIT"], s["DIVIDEND"]
    assert sp["events_compared"] == 3 and sp["agree"] == 1 and sp["mismatches"] == 2
    assert sp["mismatch_rate"] == pytest.approx(2 / 3)
    assert s["not_counted"] == {"SPLIT:dolthub_extra_row": 1}
    assert dv["events_compared"] == 3 and dv["mismatches"] == 1 and dv["amount_split_basis"] == 1
    assert dv["identity_unconfirmed"] == 1
    f = flags(rep, ("error",))
    assert list(f.ex_date) == ["2020-10-01"]
    p1, p2 = write_report(rep, s, tmp_path / "qa")
    assert pd.read_csv(p1).shape[0] == len(rep)


def test_known_spinoff_served_as_split_is_an_error(tmp_path):
    st = ParquetStore(tmp_path)
    st.write("bars_1d", "yahoo", "GE", synthetic_bars("2024-03-01", "2024-04-30", lambda d: (100.0, 100.0)))
    st.write("actions", "yahoo", "GE", synthetic_actions("GE", [("SPLIT", "2024-04-02", 1.253, 0.0)]))
    st.write("reference", "dolthub", "SPLIT", pd.DataFrame({"act_symbol": ["X"], "ex_date": ["2020-01-02"],
                                                            "to_factor": [2.0], "for_factor": [1.0]}),
             key=["act_symbol", "ex_date"])
    st.write("reference", "dolthub", "DIVIDEND", pd.DataFrame({"act_symbol": ["X"], "ex_date": ["2020-01-02"],
                                                               "amount": [0.1]}), key=["act_symbol", "ex_date"])
    rep = CrossChecker(st).run(["GE"])
    assert list(rep.status) == ["spinoff_served_as_split"] and list(rep.severity) == ["error"]
    assert summarize(rep)["SPLIT"]["spinoff_served_as_split"] == ["GE 2024-04-02"]
