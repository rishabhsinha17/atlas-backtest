"""Weekly consensus snapshots -> NTM consensus (providers/estimates.py) and their PIT serving through
Portal.fundamentals. synthetic_ rows in DoltHub's eps_estimate layout; no network."""
import pandas as pd
import pytest

from atlasbt.core.types import Fidelity
from atlasbt.data.portal import Portal
from atlasbt.data.providers import estimates as E


def synthetic_raw(sym="AAPL", dates=("2024-06-02", "2024-06-09", "2024-07-07"), fy1=(6.57, 6.60, 6.70),
                  fy2=(7.21, 7.30, 7.50), n1=(11, 11, 12), n2=(11, 9, 12), fy1_end="2024-09-30", fy2_end="2025-09-30"):
    rows = []
    for d, a, b, c1, c2 in zip(dates, fy1, fy2, n1, n2):
        rows.append({"date": d, "act_symbol": sym, "period": "Current Year", "period_end_date": fy1_end,
                     "consensus": a, "count": c1, "symbol": sym})
        rows.append({"date": d, "act_symbol": sym, "period": "Next Year", "period_end_date": fy2_end,
                     "consensus": b, "count": c2, "symbol": sym})
    return pd.DataFrame(rows)


def test_ntm_blend_and_analyst_count():
    w = E.ntm_frame(synthetic_raw()).set_index("date")
    r = w.loc[pd.Timestamp("2024-06-02")]
    wt = (pd.Timestamp("2024-09-30") - pd.Timestamp("2024-06-02")).days / 365.25
    assert r["w_fy1"] == pytest.approx(wt)
    assert r["ntm"] == pytest.approx(wt * 6.57 + (1 - wt) * 7.21)
    assert w.loc[pd.Timestamp("2024-06-09"), "n"] == 9            # min(FY1, FY2) counts
    # FY1 already ended (not yet reported): NTM = FY2
    w2 = E.ntm_frame(synthetic_raw(fy1_end="2024-05-31")).iloc[0]
    assert w2["w_fy1"] == 0.0 and w2["ntm"] == pytest.approx(7.21)


def test_known_at_is_close_of_next_session():
    k = E.known_at_for(pd.Series(pd.to_datetime(["2024-06-02", "2024-06-07", "2023-11-23"]))).dt.tz_convert("America/New_York")
    assert list(k) == [pd.Timestamp("2024-06-03 16:00", tz="America/New_York"),     # Sunday -> Monday close
                       pd.Timestamp("2024-06-10 16:00", tz="America/New_York"),     # Friday -> Monday close
                       pd.Timestamp("2023-11-24 13:00", tz="America/New_York")]     # Thanksgiving -> half day


def _built(store, raw):
    store.write(*E.RAW, raw, key=["date", "act_symbol", "period"], replace=True)
    out = E.build(store, Portal(store))
    return out, Portal(store)


def test_build_and_serve_pit(store):
    raw = pd.concat([synthetic_raw(), synthetic_raw("ZZZZ")])            # ZZZZ: no identity evidence
    out, p = _built(store, raw)
    assert out["tickers"] == 1 and out["snapshots_dropped_identity"] == 3   # fail closed on unknown identity
    assert p.has("*", "fundamentals", "eps_consensus_ntm") and p.has("AAPL", "fundamentals", "eps_num_analysts_ntm")
    assert not p.has("TSLA", "fundamentals", "eps_consensus_ntm")
    assert p.fidelity_of("AAPL", "fundamentals", "eps_consensus_ntm") == Fidelity.T2_PROXY
    t = lambda s: pd.Timestamp(s, tz="America/New_York")
    f = ["eps_consensus_ntm", "eps_num_analysts_ntm"]
    assert p.fundamentals(["AAPL"], f, t("2024-06-03 15:59")).empty             # Sunday snapshot not known yet
    a = p.fundamentals(["AAPL"], f, t("2024-06-03 16:00")).set_index("field")
    wt = (pd.Timestamp("2024-09-30") - pd.Timestamp("2024-06-02")).days / 365.25
    assert a.loc["eps_consensus_ntm", "value"] == pytest.approx(wt * 6.57 + (1 - wt) * 7.21)
    assert a.loc["eps_num_analysts_ntm", "value"] == 11
    b = p.fundamentals(["AAPL"], f, t("2024-06-20 12:00")).set_index("field")
    assert b.loc["eps_num_analysts_ntm", "value"] == 9                          # 2024-06-09 snapshot
    # stale: latest snapshot (known 2024-06-10) older than 21 days on 2024-07-05 -> nothing
    assert p.fundamentals(["AAPL"], f, t("2024-07-05 12:00")).empty
    # security keys: only the key's issuer
    assert len(p.fundamentals(["AAPL~CIK0000320193"], f, t("2024-06-20"))) == 2
    assert p.fundamentals(["AAPL~CIK0000999999"], f, t("2024-06-20")).empty


def test_sec_fields_unchanged_when_mixed(store):
    _, p = _built(store, synthetic_raw())
    a = pd.Timestamp("2024-06-20", tz="UTC")
    assert p.fundamentals(["AAPL"], ["revenue_q"], a).empty                      # no SEC facts in this store
    mixed = p.fundamentals(["AAPL"], ["revenue_q", "eps_consensus_ntm"], a)
    assert list(mixed["field"]) == ["eps_consensus_ntm"]


def test_security_key_spans_renamed_tickers(store):
    from tests.data.conftest import seed_identity, SYNTHETIC_CIKS
    seed_identity(store, {**SYNTHETIC_CIKS, "OLDT": 900010, "NEWT": 900010})
    raw = pd.concat([synthetic_raw("OLDT", dates=("2024-06-02",)), synthetic_raw("NEWT", dates=("2024-06-09",))])
    _, p = _built(store, raw)
    f = ["eps_consensus_ntm"]
    t = pd.Timestamp("2024-06-05", tz="UTC")
    assert len(p.fundamentals(["NEWT~CIK0000900010"], f, t)) == 1       # OLDT-era snapshot served to the key
    assert p.fundamentals(["NEWT"], f, t).empty                         # the bare ticker has no rows yet
