"""Tests for the conformance-review fixes (security-master T2 flags / NotTestable, symmetric group
buckets, S093 EV, S098 cash net repurchases, S100 B/M with SE<=0, S112 hedge+orphans, S114 MOC guard).
SYNTHETIC data, mechanics only."""
import math

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.types import RunStatus
from atlasbt.indicators import cross_section as xs
from atlasbt.strategies import factors as F
from atlasbt.strategies import statarb as SA

from fsa_fixtures import NAV0, build_portal, close_ts, fund_row, orders, run, sessions
import scen_factors as SCF
import scen_statarb as SCS


def _reasons(res):
    return [r["reason"] for r in res.fidelity["reasons"]]


# ---------------------------------------------------------------------- #8 symmetric group buckets
def test_group_quantile_bucket_symmetric_known_answer():
    x = pd.Series(np.arange(25, dtype=float), index=[f"N{i:02d}" for i in range(25)])
    g = pd.Series(["A"] * 11 + ["B"] * 9 + ["C"] * 4 + [np.nan], index=x.index)
    b = xs.group_quantile_bucket(x, g, 10, min_group_size=10, symmetric=True)
    assert list(b.index[b == 10]) == ["N10"] and list(b.index[b == 1]) == ["N00"]       # group of 11 -> 1 + 1
    assert b[g == "B"].isna().all() and b[g == "C"].isna().all() and math.isnan(b["N24"])
    # default behaviour unchanged
    d = xs.group_quantile_bucket(x, g, 10)
    ref = pd.concat([xs.quantile_bucket(x[g == k], 10) for k in ("A", "B", "C")])
    pd.testing.assert_series_equal(d.dropna().sort_index(), ref.sort_index(), check_names=False)


# ---------------------------------------------------------------------- #1 / #2 security master
def test_industry_column_missing_is_not_testable():
    sc = SCF.s087()
    p = sc["build"]()
    p._master = p._master.drop(columns=["gics_sub_industry_current"])
    from atlasbt.engine.backtester import run_backtest
    from fsa_fixtures import zero_cost
    res = run_backtest(F.S087(), p, SCF.START, SCF.END, zero_cost())
    assert res.status == RunStatus.NOT_TESTABLE and "industry column" in res.not_testable_reason
    sc7 = SCS.s107()
    p7 = sc7["build"]()
    p7._master = p7._master.drop(columns=["sic"])
    res7 = run_backtest(SA.S107(), p7, sc7["start"], sc7["end"], zero_cost())
    assert res7.status == RunStatus.NOT_TESTABLE


def test_industry_coverage_flag_and_floor():
    from atlasbt.engine.backtester import run_backtest
    from fsa_fixtures import zero_cost
    sc = SCF.s087()
    p = sc["build"]()
    m = p._master.copy()
    m.loc[m.index[:6], "gics_sub_industry_current"] = None          # 14/20 = 70% coverage
    p._master = m
    st = F.S087()
    res = run_backtest(st, p, SCF.START, SCF.END, zero_cost())
    assert res.status == RunStatus.OK
    flags = [r for r in _reasons(res) if r.startswith("S087: industry")]
    assert len(flags) == 1 and "coverage 70%" in flags[0]
    assert st.sm.coverage[-1][2] == pytest.approx(0.70)
    m.loc[m.index[:12], "gics_sub_industry_current"] = None         # 40% < 50% floor
    p._master = m
    res2 = run_backtest(F.S087(), p, SCF.START, SCF.END, zero_cost())
    assert res2.status == RunStatus.NOT_TESTABLE and "coverage 40%" in res2.not_testable_reason


def test_sector_flagged_once_as_t2():
    sc = SCF.s085()
    res = run(F.S085(), sc["build"](), "2023-04-25", "2023-07-05")       # three month-ends
    flags = [r for r in res.fidelity["reasons"] if "SIC->GICS-like sector" in r["reason"]]
    assert len(flags) == 1 and flags[0]["tier"] in ("T2_PROXY", 2, "T2")


def test_security_master_failure_propagates():
    sc = SCF.s085()
    p = sc["build"]()

    class Broken(type(p)):
        def security_master(self, as_of):
            raise RuntimeError("security master store missing")

    b = Broken(bars=dict(p._bars), universes=p._universes)
    with pytest.raises(RuntimeError):
        run(F.S085(), b, SCF.START, SCF.END)


# ---------------------------------------------------------------------- #6 S093 EV, S098 cash measure
def test_s093_ev_includes_short_term_debt_and_preferred():
    sc = SCF.s093()
    p = sc["build"]()
    f = p._fund.copy()
    extra = pd.DataFrame([fund_row("S03", "short_term_debt", 9e10, "2022-12-31", "2023-02-21 21:00"),
                          fund_row("S03", "preferred_stock", 2e10, "2022-12-31", "2023-02-21 21:00")])
    extra["known_at"] = pd.to_datetime(extra["known_at"], utc=True)
    p._fund = pd.concat([f, extra], ignore_index=True)
    st = F.S093()
    res = run(st, p, SCF.START, SCF.END)
    # S03 had EV <= 0 (huge cash); with 110bn of extra claims EV > 0 and its high FCF yield makes it long
    o = orders(res)
    assert "S03" in set(o.loc[o["quantity"] > 0, "symbol"])


def test_s098_cash_net_repurchase_known_answer():
    dates = sessions("2021-01-04", "2023-08-31")
    syms = [f"S{i:02d}" for i in range(10)]
    closes = {s: pd.Series(50.0, index=pd.DatetimeIndex(dates)) for s in syms}
    rows = []
    for i, s in enumerate(syms):
        rows.append(fund_row(s, "shares_outstanding", 1e8, "2023-04-20", "2023-05-05 21:00"))
        rows.append(fund_row(s, "shares_outstanding", 1e8, "2022-04-20", "2022-05-05 21:00"))
        if s == "S09":
            continue                                    # no cash-flow facts -> share-count fallback (0)
        rp, iss = 1e8 * i, 2e8 if s == "S00" else 0.0
        rows += [fund_row(s, "repurchase_fy", rp, "2022-12-31", "2023-02-21 21:00"),
                 fund_row(s, "stock_issuance_proceeds_fy", iss, "2022-12-31", "2023-02-21 21:00")]
    p = build_portal(closes, universe=syms, master_df=SCF.master(syms), fundamentals=pd.DataFrame(rows))
    st = F.S098()
    res = run(st, p, SCF.START, SCF.END)
    o = orders(res)
    # yields: S00 = -2e8/5e9 (lowest), S01 = 1e8/5e9, ..., S08 = 8e8/5e9 (highest); S09 = 0 via fallback
    assert set(o.loc[o["quantity"] > 0, "symbol"]) == {"S07", "S08"}
    assert set(o.loc[o["quantity"] < 0, "symbol"]) == {"S00", "S09"}
    assert any("share-count net-issuance fallback" in r for r in _reasons(res))


# ---------------------------------------------------------------------- #9 S100 B/M with SE <= 0
def test_s100_negative_equity_not_ranked():
    sc = SCF.s100()
    p = sc["build"]()
    f = p._fund.copy()
    tgt = sc["longs"][0]
    f.loc[(f["symbol"] == tgt) & (f["field"] == "stockholders_equity"), "value"] = -5e9
    p._fund = f
    st = F.S100()
    res = run(st, p, SCF.START, SCF.END)
    assert tgt not in set(orders(res)["symbol"])


# ---------------------------------------------------------------------- #10 S112 hedge + orphans
def test_s112_hedge_sized_from_pending_orders():
    sc = SCS.s112()
    st = SA.S112()
    res = run(st, sc["build"](), sc["start"], sc["end"])
    o = orders(res)
    sig = close_ts(SCS.K_DAY)
    ent = o[(o["submitted_at"] == sig) & (o["symbol"] != "SPY")]
    h = o[(o["submitted_at"] == sig) & (o["symbol"] == "SPY")]
    nb = sum(r.quantity * float(sc["closes"][r.symbol].loc[SCS.K_DAY]) * st.betas[r.symbol] for r in ent.itertuples())
    spx = float(sc["closes"]["SPY"].loc[SCS.K_DAY])
    exp = -math.trunc(nb / spx)
    if abs(exp) * spx > 0.005 * NAV0:
        assert len(h) == 1 and abs(h["quantity"].iloc[0] - exp) <= 1 and np.sign(h["quantity"].iloc[0]) == np.sign(exp)
    else:
        assert len(h) == 0


def test_s112_orphan_position_is_flattened():
    sc = SCS.s112()

    class WithOrphan(SA.S112):
        def step(self, ctx):
            if self.bar_i == 1:
                ctx.order("P19", 100, tag="manual")
            super().step(ctx)

    res = run(WithOrphan(), sc["build"](), sc["start"], sc["end"])
    o = orders(res)
    orph = o[(o["symbol"] == "P19") & (o["tag"] == "exit:orphan")]
    days = sessions(SCS.K_DAY, "2023-03-31")
    assert len(orph) == 1 and orph["quantity"].iloc[0] == -100
    assert orph["submitted_at"].iloc[0] == close_ts(days[1])


# ---------------------------------------------------------------------- #11 S114 MOC guard
def test_s114_skips_pair_when_entry_could_be_volume_capped():
    sc = SCS.s114()
    from fsa_fixtures import build_portal as bp
    low_vol = {s: 5_000_000.0 for s in sc["closes"]}
    low_vol["T09"] = 1_000.0                                   # 5% NAV / ~33 = ~1500 sh > 10% x 1000
    p = bp(sc["closes"], universe=SCS.S114_STOCKS, master_df=sc["build"]()._master, volumes=low_vol)
    res = run(SA.S114(), p, sc["start"], sc["end"])
    o = orders(res)
    e = o[(o["submitted_at"] == close_ts(SCS.K_DAY)) & o["tag"].str.startswith("entry")]
    assert "T09" not in set(e["symbol"]) and "T00" not in set(e["symbol"])     # its pair (T00, T09) skipped
    assert {"T01", "T08"} <= set(e["symbol"])
    moc = o[(o["type"] == "MOC")]
    assert dict(zip(moc["symbol"], moc["quantity"])) == {r.symbol: -r.quantity for r in e.itertuples()}


# ---------------------------------------------------------------------- FundHistory via fundamentals_history
def test_fundhistory_uses_fundamentals_history_when_available():
    from atlasbt.core.types import LookaheadError
    rows = pd.DataFrame([fund_row("A", "ni_fy", 100.0, "2021-12-31", "2022-02-20"),
                         fund_row("A", "ni_q", 20.0, "2022-03-31", "2022-05-05"),
                         fund_row("A", "ni_fy", 120.0, "2022-12-31", "2023-02-20"),
                         fund_row("A", "ni_q", 35.0, "2023-03-31", "2023-05-05")])
    now = pd.Timestamp("2023-06-30 20:00", tz="UTC")

    class P:
        def __init__(self, df):
            self.df = df

        def fundamentals_history(self, symbols, fields, as_of, start=None):
            return self.df[self.df["known_at"] <= as_of]

    class C:
        def __init__(self, df, now_):
            self.now, self._p = now_, P(df)

        def portal(self):
            return self._p

        def session_date(self):
            return pd.Timestamp("2023-06-30")

    fh = F.FundHistory(["ni_fy", "ni_q"], 27)
    fh.update(C(rows, now), ["A"])
    assert fh.mode == "fundamentals_history"
    assert fh.ttm("A", "ni", now) == pytest.approx(120 + 35 - 20)
    leak = pd.concat([rows, pd.DataFrame([fund_row("A", "ni_q", 1.0, "2023-06-30", "2023-08-01")])])

    class Leaky(P):
        def fundamentals_history(self, symbols, fields, as_of, start=None):
            return self.df

    c = C(rows, now)
    c._p = Leaky(leak)
    with pytest.raises(LookaheadError):
        F.FundHistory(["ni_q"], 0).update(c, ["A"])
