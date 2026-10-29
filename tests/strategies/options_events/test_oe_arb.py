"""S171 (parity screen) and S172 (SPX box) mechanics on SYNTHETIC quotes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.clock import et
from atlasbt.core.types import Fidelity, RunStatus
from atlasbt.strategies.options_events import S171, S172

from .oe_fix import (close_of, close_snaps, daily_bars, fills, index_instrument, intraday_bars, minute_snaps,
                     option_book, order_key, orders, run, sess)

DAY = pd.Timestamp("2024-03-13")


def T(hhmm):
    return et(DAY, hhmm)


def combo(res, prefix):
    o = orders(res)
    return o[o["tag"].astype(str).str.startswith(prefix)]


# ---------------------------------------------------------------------------- S171
def parity_spec(gap=0.25, converge_at="11:00"):
    """XYZ at 100 (quotes 99.99/100.01); at 10:00 the 100-strike put is offered `gap` below the
    call bid (conversion edge); at `converge_at` the call ask drops (unwind >= locked value)."""
    snaps = minute_snaps([DAY])
    t10, tc = T("10:00"), (T(converge_at) if converge_at else None)

    optq, insts = option_book("XYZ", [("2024-04-19", "PHYSICAL", "AMERICAN", None)], [95.0, 100.0, 105.0], snaps,
                              lambda t: 100.0, lambda t, K, e, Ty: 0.20, hs=lambda p: 0.05)
    c100 = [s for s, i in insts.items() if i.strike == 100.0 and i.right.value == "C"][0]
    p100 = [s for s, i in insts.items() if i.strike == 100.0 and i.right.value == "P"][0]
    c, p = optq[c100].copy(), optq[p100].copy()
    p.loc[t10, "ask"] = c.loc[t10, "bid"] - gap
    p.loc[t10, "bid"] = p.loc[t10, "ask"] - 0.10
    if tc is not None:
        c.loc[tc, "ask"] = c.loc[tc, "bid"] - 0.40
        c.loc[tc, "bid"] = c.loc[tc, "ask"] - 0.10
    optq[c100], optq[p100] = c, p
    qts = pd.DataFrame({"bid": 99.99, "ask": 100.01, "bid_size": 1e5, "ask_size": 1e5}, index=snaps)
    bars = {("XYZ", "1m"): intraday_bars([DAY], lambda t: 100.0)}
    return {"bars": bars, "option_quotes": optq, "instruments": insts, "quotes": {"XYZ": qts}}


S171T = type("S171T", (S171,), {"universe": ["XYZ"]})


def test_synthetic_S171_conversion_entry_and_convergence_exit():
    r = run(S171T(), parity_spec(), DAY, DAY)
    op = combo(r, "S171:open")
    assert len(op) == 1 and pd.Timestamp(op["submitted_at"].iloc[0]) == T("10:00")
    st = combo(r, "S171:stock_leg")
    assert len(st) == 1 and pd.Timestamp(st["submitted_at"].iloc[0]) == T("10:00")
    f = fills(r, "S171:open")
    assert set(f["ts"]) == {T("10:01")}
    side = dict(zip(f["symbol"].str[12:13], np.sign(f["quantity"])))
    assert side == {"C": -1, "P": 1}                                     # conversion: short call, long put
    n = int(abs(f["quantity"].iloc[0]))
    sf = fills(r, "S171:stock_leg")
    assert float(sf["quantity"].sum()) == 100 * n and (sf["ts"] > T("10:00")).all()
    cl = combo(r, "S171:close")
    assert pd.Timestamp(cl["submitted_at"].iloc[0]) == T("11:00") and cl["tag"].iloc[0] == "S171:close:converged"
    cs = combo(r, "S171:close_stock")
    assert float(cs["quantity"].iloc[0]) == -100 * n


@pytest.mark.parametrize("kw", [{"gap": 0.10}, {"min_edge_bp": 50.0}])
def test_synthetic_S171_near_miss(kw):
    gap = kw.pop("gap", 0.25)
    r = run(S171T(**kw), parity_spec(gap=gap, converge_at=None), DAY, DAY)
    assert len(combo(r, "S171")) == 0


def test_S171_not_testable_on_eod_options():
    r = run(S171T(), parity_spec(), DAY, DAY, option_fidelity=Fidelity.T2_PROXY)
    assert r.status == RunStatus.NOT_TESTABLE


# ---------------------------------------------------------------------------- S172
SESS = sess("2024-01-29", "2024-02-02")


def box_spec(r_model=0.08, hs=0.01):
    exps = [("2024-02-15", "CASH_AM", "EUROPEAN", "SPX"), ("2024-03-01", "CASH_PM", "EUROPEAN", "SPXW"),
            ("2024-03-14", "CASH_AM", "EUROPEAN", "SPX"), ("2024-03-14", "CASH_PM", "EUROPEAN", "SPXW")]
    optq, insts = option_book("SPX", exps, np.arange(4700.0, 5101.0, 25.0), close_snaps(SESS), lambda t: 4900.0,
                              lambda t, K, e, Ty: 0.15, r=r_model, hs=lambda p: hs)
    insts["SPX"] = index_instrument("SPX")
    return {"bars": {("SPX", "1d"): daily_bars(SESS, np.full(len(SESS), 4900.0))}, "option_quotes": optq,
            "instruments": insts, "rf": 0.05}


def test_synthetic_S172_box_lending_vs_tbill():
    r = run(S172(), box_spec(), SESS[0], SESS[-1])
    op = combo(r, "S172:open")
    assert len(op) == 1 and pd.Timestamp(op["submitted_at"].iloc[0]) == close_of("2024-01-31")   # month end
    f = fills(r, "S172:open")
    assert set(f["ts"]) == {close_of("2024-02-01")}
    legs = sorted(zip(f["symbol"].str[:6].str.strip(), f["symbol"].str[6:12], f["symbol"].str[12:13],
                      f["symbol"].str[13:].astype(int) / 1000.0, np.sign(f["quantity"])))
    assert {x[0] for x in legs} == {"SPXW"} and {x[1] for x in legs} == {"240301"}   # 30DTE PM European series
    assert [(x[2], x[3], x[4]) for x in legs] == [("C", 4875.0, 1), ("C", 4925.0, -1), ("P", 4875.0, -1),
                                                  ("P", 4925.0, 1)]
    n = int(abs(f["quantity"].iloc[0]))
    debit = float((f["quantity"] * f["price"]).sum()) / n
    assert n == int(np.floor(0.5 * 1e6 / debit / 100 * 100)) or abs(n * debit * 100 - 0.5e6) < debit * 100 * 2


def test_synthetic_S172_near_miss_no_edge_and_not_testable():
    r = run(S172(), box_spec(r_model=0.05), SESS[0], SESS[-1])                # box priced at the bill rate
    assert len(combo(r, "S172")) == 0
    r = run(S172(), box_spec(), SESS[0], SESS[-1], option_fidelity=Fidelity.T2_PROXY)
    assert r.status == RunStatus.NOT_TESTABLE


@pytest.mark.parametrize("cls,spec_fn,start,end,cut", [
    (S171T, parity_spec, DAY, DAY, T("10:30")),
    (S172, box_spec, SESS[0], SESS[-1], close_of("2024-01-31") + pd.Timedelta(hours=1)),
])
def test_lookahead_arb(cls, spec_fn, start, end, cut):
    spec = spec_fn()
    full = run(cls(), spec, start, end)
    trunc = run(cls(), spec, start, end, cut=cut)
    a, b = order_key(orders(full), cut), order_key(orders(trunc), cut)
    assert a == b and len(a) >= 1


def test_smoke_arb():
    assert run(S171T(), parity_spec(converge_at=None), DAY, DAY).status == RunStatus.OK
    assert run(S172(width=100.0, dte=90), box_spec(), SESS[0], SESS[-1]).status == RunStatus.OK
