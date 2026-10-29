"""Conformance of the OPTIONS plug-ins to the ENGINE agent's protocols."""
import pandas as pd
import pytest

from atlasbt.core.types import AssetClass, ExerciseStyle, Instrument, Right, SettlementType
from atlasbt.options.plugins import CboeStrategyMargin, OccOptionLifecycle, underlying_class

hooks = pytest.importorskip("atlasbt.portfolio.hooks")
acct_mod = pytest.importorskip("atlasbt.portfolio.account")
margin_mod = pytest.importorskip("atlasbt.portfolio.margin")


class Prices:
    def __init__(self, o, c):
        self.o, self.c = o, c

    def open(self, sym, session):
        return self.o

    def close(self, sym, session):
        return self.c


def opt(und, exp, K, right, st=SettlementType.PHYSICAL, ex=ExerciseStyle.AMERICAN, root=None):
    return Instrument.option(und, pd.Timestamp(exp), K, Right(right), ex, st, root=root)


def test_protocol_conformance():
    assert isinstance(OccOptionLifecycle(), hooks.OptionLifecycle)
    assert isinstance(CboeStrategyMargin(), margin_mod.OptionMarginModel)


def test_lifecycle_am_and_pm():
    P = acct_mod.Position
    am = opt("SPX", "2024-03-14", 5100, "C", SettlementType.CASH_AM, ExerciseStyle.EUROPEAN)
    pm = opt("SPX", "2024-03-15", 5100, "P", SettlementType.CASH_PM, ExerciseStyle.EUROPEAN, root="SPXW")
    phys = opt("XYZ", "2024-03-15", 50, "C")
    positions = [P(am, -1), P(pm, 2), P(phys, 3)]
    lc = OccOptionLifecycle()
    ev = lc.at_session_start(pd.Timestamp("2024-03-15"), pd.Timestamp("2024-03-14"), positions, Prices(5130.0, 5000.0))
    assert len(ev) == 1 and ev[0].kind == "CASH_SETTLE" and ev[0].option_price == pytest.approx(30.0)
    assert ev[0].option_qty == 1
    assert lc.at_session_start(pd.Timestamp("2024-03-14"), None, positions, Prices(5130.0, 5000.0)) == []
    ev2 = lc.at_session_end(pd.Timestamp("2024-03-15"), positions, Prices(5130.0, 5000.0))
    kinds = {e.option.symbol: e for e in ev2}
    assert kinds[pm.symbol].kind == "CASH_SETTLE" and kinds[pm.symbol].option_price == pytest.approx(100.0)
    assert kinds[phys.symbol].kind == "EXERCISE" and kinds[phys.symbol].underlying_qty == 300  # close 5000 vs K 50


def test_lifecycle_early_assignment_hook():
    P = acct_mod.Position
    c = opt("XYZ", "2024-04-19", 90, "C")
    same = pd.Timestamp("2024-03-06 21:00", tz="UTC")
    pos = [P(c, -2, avg_price=5.0, mark=10.1, mark_ts=same)]
    lc = OccOptionLifecycle(rate=lambda s: 0.05, dividends=lambda u, s: [(pd.Timestamp("2024-03-07"), 1.0)])
    ev = lc.at_session_end(pd.Timestamp("2024-03-06"), pos, Prices(100.0, 100.0))
    assert len(ev) == 1 and ev[0].kind == "ASSIGNMENT" and ev[0].underlying_qty == -200
    # stale (previous-session) mark or missing mark -> no assignment; model fallback used if supplied
    stale = [P(c, -2, avg_price=5.0, mark=10.1, mark_ts=pd.Timestamp("2024-03-05 21:00", tz="UTC"))]
    assert lc.at_session_end(pd.Timestamp("2024-03-06"), stale, Prices(100.0, 100.0)) == []
    nomark = [P(c, -2, avg_price=5.0)]
    assert lc.at_session_end(pd.Timestamp("2024-03-06"), nomark, Prices(100.0, 100.0)) == []
    lc2 = OccOptionLifecycle(rate=lambda s: 0.05, dividends=lambda u, s: [(pd.Timestamp("2024-03-07"), 1.0)],
                             model_value=lambda ins, s: 10.05)
    ev2 = lc2.at_session_end(pd.Timestamp("2024-03-06"), stale, Prices(100.0, 100.0))
    assert len(ev2) == 1 and "PIT model value" in ev2[0].note


def test_margin_plugin_values():
    A = acct_mod.Account(100000.0)
    P = acct_mod.Position
    # credit put spread on SPY-like broad ETF: 425/430 with marks 6.40/7.80 -> net 360
    lp = opt("SPY", "2025-12-19", 425, "P")
    sp = opt("SPY", "2025-12-19", 430, "P")
    A.positions[lp.symbol] = P(lp, 1, 6.40, 6.40)
    A.positions[sp.symbol] = P(sp, -1, 7.80, 7.80)
    m = CboeStrategyMargin()
    assert m.requirement(A, lambda u: 433.35) == pytest.approx(360.0)
    # naked short call on equity -> 2,570 net of the premium
    B = acct_mod.Account(100000.0)
    c = opt("XYZ", "2025-11-21", 120, "C")
    B.positions[c.symbol] = P(c, -1, 8.40, 8.40)
    assert m.requirement(B, lambda u: 128.50) == pytest.approx(2570.0)
    # covered by stock -> 0 (credit not used)
    stk = Instrument("XYZ", AssetClass.STOCK)
    B.positions["XYZ"] = P(stk, 100, 128.5, 128.5)
    assert m.requirement(B, lambda u: 128.50) == 0.0


def test_underlying_class():
    assert underlying_class("SPX") == ("broad_index", 1.0)
    assert underlying_class("SPY") == ("broad_etf", 1.0)
    assert underlying_class("XLE") == ("equity", 1.0)
    assert underlying_class("TQQQ") == ("broad_etf", 3.0)


def test_plugin_nonstandard_deliverable_and_margin_attribute():
    from atlasbt.options.lifecycle import adjust_for_split
    P = acct_mod.Position
    adj, q = adjust_for_split(opt("XYZ", "2024-03-15", 50.0, "C"), 2, 1.5)     # 3:2 split -> 150 sh
    ev = OccOptionLifecycle().at_session_end(pd.Timestamp("2024-03-15"), [P(adj, q)], Prices(40.0, 40.0))
    assert ev[0].kind == "EXERCISE" and ev[0].underlying_qty == 300
    assert ev[0].underlying_price == pytest.approx(5000.0 / 150)              # K x multiplier / deliverable
    assert CboeStrategyMargin().includes_long_premium is True


def test_slice_expiry_requires_series_when_mixed():
    from atlasbt.options.chain import SelectionError, slice_expiry
    df = pd.DataFrame({"expiry": [pd.Timestamp("2024-03-14")] * 2, "root": ["SPX", "SPXW"],
                       "settlement": ["CASH_AM", "CASH_PM"], "strike": [5000.0] * 2, "right": ["C"] * 2})
    with pytest.raises(SelectionError):
        slice_expiry(df, "2024-03-14")
    assert len(slice_expiry(df, "2024-03-14", root="SPXW")) == 1
    assert len(slice_expiry(df, "2024-03-14", settlement="CASH_AM")) == 1
