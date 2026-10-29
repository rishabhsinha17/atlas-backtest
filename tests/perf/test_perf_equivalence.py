"""Performance changes are pure re-implementations / memoization (docs/PERF.md). Each test pins one of
them against the reference computation it replaced, and checks that the point-in-time guards still fire.

Synthetic data (mechanics only) for the engine/strategy helpers; the real cache for the data-layer memos
(skipped when the cache is absent)."""
import copy

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.clock import nyse
from atlasbt.core.types import LookaheadError
from atlasbt.engine import BacktestConfig, Backtester, Strategy
from atlasbt.engine.market import CODE_SESSION, BarArrays, MarketData
from atlasbt.engine.testing import InMemoryPortal, synthetic_daily_bars, synthetic_intraday_bars

CAL = nyse()


# ----------------------------------------------------------------------------- engine: BarArrays.frame
def _frame_reference(b: BarArrays, i0: int, i1: int) -> pd.DataFrame:
    """The pre-optimisation BarArrays.frame (verbatim)."""
    idx = pd.DatetimeIndex(b.start[i0:i1], tz="UTC", name="start")
    return pd.DataFrame({"open": b.o[i0:i1], "high": b.h[i0:i1], "low": b.l[i0:i1],
                         "close": b.c[i0:i1], "volume": b.v[i0:i1],
                         "session": [CODE_SESSION[int(x)] for x in b.sess[i0:i1]],
                         "known_at": pd.DatetimeIndex(b.end[i0:i1], tz="UTC")}, index=idx)


@pytest.mark.parametrize("freq", ["1d", "1m"])
def test_bararrays_frame_identical_to_reference(freq):
    if freq == "1d":
        df = synthetic_daily_bars("2024-01-02", "2024-06-28")
    else:
        df = synthetic_intraday_bars("2024-03-04", "2024-03-06", "1m", include_extended=True)
    b = BarArrays("XYZ", freq, df)
    n = b.n
    for i0, i1 in [(0, 0), (0, 1), (5, 8), (n - 3, n), (-5, n), (-10, -2), (3, 1), (n - 1, n + 50), (-n - 7, 3)]:
        got, ref = b.frame(i0, i1), _frame_reference(b, i0, i1)
        pd.testing.assert_frame_equal(got, ref, check_exact=True, check_freq=True, check_flags=True)
        assert list(got.dtypes) == list(ref.dtypes) and got.index.name == ref.index.name
        assert got.index.freq == ref.index.freq
    e = BarArrays("E", freq, None)
    pd.testing.assert_frame_equal(e.frame(0, 0), _frame_reference(e, 0, 0), check_exact=True)


# ----------------------------------------------------------------------------- strategy helpers
class _Probe(Strategy):
    id = "PERF_PROBE"
    universe = ["XYZ"]

    def __init__(self, fn, **kw):
        super().__init__(**kw)
        self.fn = fn
        self.n_checks = 0

    def on_bar(self, ctx):
        self.fn(ctx, self)


def _zero_cfg():
    return BacktestConfig(fees="NO_FEES", debug=True)


def test_swing_recent_raw_equals_history_frame_arrays():
    """swing_common._recent_raw (history_arrays) == _frame_arrays(ctx.history(raw)) at every step."""
    from atlasbt.strategies.swing_common import _frame_arrays, _recent_raw
    df = synthetic_daily_bars("2023-01-03", "2023-12-29", seed=3)
    portal = InMemoryPortal(bars={("XYZ", "1d"): df})

    def fn(ctx, st):
        for n in (1, 3, 25):
            got = _recent_raw(ctx, "XYZ", n)
            h = ctx.history("XYZ", n, "1d", adjusted=False)
            ref = _frame_arrays(h) if h is not None and len(h) else None
            assert (got is None) == (ref is None)
            if got is not None:
                for a, b in zip([got[0], got[1], *got[2], got[3]], [ref[0], ref[1], *ref[2], ref[3]]):
                    assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
                assert int(got[1][-1]) <= int(ctx.now.value)          # never the in-progress bar
                st.n_checks += 1
    s = _Probe(fn)
    Backtester(s, portal, _zero_cfg()).run("2023-03-01", "2023-06-30", raise_errors=True)
    assert s.n_checks > 100


def test_intraday_tape_history_arrays_matches_history_frame():
    """intraday_base.Tape.update reads ctx.history_arrays; the arrays (and session labels) equal the
    DataFrame path it replaced, with PRE bars included."""
    from atlasbt.strategies.intraday_base import _SESSION_NAMES
    df = synthetic_intraday_bars("2024-03-04", "2024-03-07", "1m", include_extended=True)
    portal = InMemoryPortal(bars={("XYZ", "1m"): df})

    class P(_Probe):
        bar_freq = "1m"
        sessions = ("PRE", "RTH")

    def fn(ctx, st):
        if st.n_checks > 400:
            return
        a = ctx.history_arrays("XYZ", 7, adjusted=False, sessions=("PRE", "RTH"))
        h = ctx.history("XYZ", 7, adjusted=False, sessions=("PRE", "RTH"))
        assert np.array_equal(a["start"], pd.DatetimeIndex(h.index).as_unit("ns").asi8)
        for k in ("open", "high", "low", "close", "volume"):
            assert a[k].tobytes() == h[k].to_numpy(dtype=float).tobytes()
        assert list(_SESSION_NAMES[a["session"]]) == list(h["session"].to_numpy(object))
        assert (a["end"] <= int(ctx.now.value)).all()
        st.n_checks += 1
    s = P(fn)
    Backtester(s, portal, _zero_cfg()).run("2024-03-05", "2024-03-06", raise_errors=True)
    assert s.n_checks > 300


# ----------------------------------------------------------------------------- lookahead guards still fire
def test_lookahead_guards_still_fire():
    df = synthetic_daily_bars("2023-01-03", "2023-06-30", seed=1)
    portal = InMemoryPortal(bars={("XYZ", "1d"): df})

    def fn(ctx, st):
        with pytest.raises(LookaheadError):
            ctx.portal().bars("XYZ", "1d", ctx.now - pd.Timedelta(days=30), as_of=ctx.now + pd.Timedelta(seconds=1))
        with pytest.raises(LookaheadError):
            ctx.portal().corporate_actions(["XYZ"], ctx.now - pd.Timedelta(days=30), ctx.now,
                                           as_of=ctx.now + pd.Timedelta(days=1))
        a = ctx.history_arrays("XYZ", 5, "1d")
        h = ctx.history("XYZ", 5, "1d", adjusted=False)
        assert (a["end"] <= int(ctx.now.value)).all()
        assert (pd.DatetimeIndex(h["known_at"]) <= ctx.now).all()
        st.n_checks += 1
    s = _Probe(fn)
    Backtester(s, portal, _zero_cfg()).run("2023-03-01", "2023-03-31", raise_errors=True)
    assert s.n_checks > 10
    # the engine's own cache guard
    md = MarketData(portal, CAL, pd.Timestamp("2023-03-01"), pd.Timestamp("2023-03-31"), "1d", ("RTH",),
                    spread_model=None)
    md.now_ns = int(pd.Timestamp("2023-03-10 21:00", tz="UTC").value)
    with pytest.raises(LookaheadError):
        md.window("XYZ", 0, md.now_ns + 1)
    with pytest.raises(LookaheadError):
        md.last_close("XYZ", md.now_ns + 10**9)
    assert md.guard_violations == 2


def test_lookahead_error_propagates_from_strategy_with_bar_cache():
    """A strategy using the swing BarCache that asks for the future still fails the run loudly."""
    from atlasbt.strategies.swing_common import BarCache
    df = synthetic_daily_bars("2023-01-03", "2023-06-30", seed=2)
    portal = InMemoryPortal(bars={("XYZ", "1d"): df})
    cache = BarCache(keep=100, fetch=50)

    def fn(ctx, st):
        b = cache.get(ctx, "XYZ", 20)
        assert b.end[-1] <= int(ctx.now.value)
        if ctx.session_date() == pd.Timestamp("2023-03-15"):
            ctx.portal().bars("XYZ", "1d", ctx.now, as_of=ctx.now + pd.Timedelta(days=1))
    with pytest.raises(LookaheadError):
        Backtester(_Probe(fn), portal, _zero_cfg()).run("2023-03-01", "2023-03-31")


# ----------------------------------------------------------------------------- data layer memos (real cache)
def _real_portal():
    try:
        from atlasbt.data.portal import Portal
        from atlasbt.data.store import default_store
        if not default_store().exists("bars_1d", "yahoo", "SPY"):
            return None
        return Portal
    except Exception:
        return None


real = pytest.mark.skipif(_real_portal() is None, reason="real data cache not available")


@real
def test_portal_quote_read_memo_equals_store_read():
    P = _real_portal()
    p = P()
    lo, hi = pd.Timestamp("2019-06-03 13:30", tz="UTC"), pd.Timestamp("2019-06-05 20:00", tz="UTC")
    for sym in ("SPY", "QQQ"):
        for src in ("iex", "dukascopy"):
            if not p.store.exists("quotes", src, sym):
                continue
            for a, b in [(lo, hi), (lo - pd.Timedelta(days=900), lo), (pd.Timestamp("2026-10-01", tz="UTC"),
                                                                       pd.Timestamp("2026-10-03", tz="UTC"))]:
                for _ in range(2):                                # cold, then memoized
                    got = p._store_read_memo("quotes", src, sym, start=a, end=b)
                    ref = p.store.read("quotes", src, sym, start=a, end=b)
                    pd.testing.assert_frame_equal(got, ref, check_exact=True)
                    assert got.attrs == ref.attrs


@real
def test_portal_universe_borrow_identity_memos():
    from atlasbt.data.providers import borrow as B
    P = _real_portal()
    p, cold = P(), P()
    t = pd.Timestamp("2019-03-29 21:00", tz="UTC")
    u1 = p.universe("SP500", t)
    u1.append("MUTATED")                                       # the memo hands out copies
    u2 = p.universe("SP500", t)
    assert "MUTATED" not in u2 and u2 == cold.universe("SP500", t)
    members = set(u2) | {k.split("~")[0] for k in u2}
    for s in ["SPY", "XLF", "TLT", "GLD", "AAPL", u2[3], u2[200], "ZZZZ"]:
        su = p._split_key(s)[0]
        ref = B.default_fee(su, is_etf=cold._is_etf(su), is_index_member=s.upper() in members)
        assert p.borrow(s, t) == ref == cold.borrow(s, t)
    # single-date fast path of _identity_by_date == DatetimeIndex path
    for s in u2[:40]:
        t_ = p._split_key(s)[0]
        d = pd.Timestamp("2019-03-29")
        one = p._identity_by_date(t_, [d])
        two = p._identity_by_date(t_, pd.DatetimeIndex([d, d]))
        assert int(one[0][0]) == int(two[0][1]) and one[1][0] == two[1][1] and one[2][0] == two[2][1]


@real
def test_portal_adjusted_bars_and_actions_warm_equal_cold():
    """Memoized paths (attrs-free selection, naive-date memo, actions memo) == a cold portal, incl. attrs."""
    P = _real_portal()
    warm = P()
    t = pd.Timestamp("2019-03-29 21:00", tz="UTC")
    syms = warm.universe("SP500", t)[::50][:8] + ["SPY", "XLE"]
    for k, asof in enumerate(pd.date_range("2019-01-31 21:00", periods=3, freq="ME", tz="UTC")):
        cold = P()
        for s in syms:
            for adj in (True, False):
                a = warm.bars(s, "1d", asof - pd.Timedelta(days=500), asof, as_of=asof, adjusted=adj)
                b = cold.bars(s, "1d", asof - pd.Timedelta(days=500), asof, as_of=asof, adjusted=adj)
                pd.testing.assert_frame_equal(a, b, check_exact=True)
                assert a.attrs.keys() == b.attrs.keys()
                for key in a.attrs:
                    x, y = a.attrs[key], b.attrs[key]
                    if isinstance(x, pd.DataFrame):
                        pd.testing.assert_frame_equal(x, y, check_exact=True)
                    else:
                        assert x == y
            ca = warm.corporate_actions([s], asof - pd.Timedelta(days=800), asof, asof)
            assert ca == cold.corporate_actions([s], asof - pd.Timedelta(days=800), asof, asof)
            assert copy.deepcopy(ca) == ca
