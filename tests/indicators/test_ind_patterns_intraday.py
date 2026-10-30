"""Known-answer tests for pattern detectors (with confirmation lags) and
intraday/session indicators, on hand-constructed synthetic bars."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import atlasbt.indicators as ind


def _hlc(c, half=0.5):
    c = np.asarray(c, float)
    return c + half, c - half, c


# ---------------------------------------------------------------- simple patterns
def test_nr_inside_ibs_engulfing_downstreak():
    h = np.array([15, 14, 13, 12, 11.0]); l = np.array([10, 10, 10, 10, 10.0])
    assert list(ind.nr_n(h, l, 3)) == [False, False, True, True, True]
    assert list(ind.nr_n(np.array([2, 2.0]), np.zeros(2), 2)) == [False, True]  # tie counts
    H = np.array([10, 9, 8.5, 9.5]); L = np.array([5, 6, 6.5, 6.0])
    assert list(ind.inside_bar(H, L)) == [False, True, True, False]
    assert list(ind.inside_bar(H, L, nested=2)) == [False, False, True, False]
    ib = ind.ibs(np.array([10, 5.0]), np.array([8, 5.0]), np.array([9.5, 5.0]))
    assert ib[0] == pytest.approx(0.75) and np.isnan(ib[1])
    o = np.array([10, 9.0]); c = np.array([9.5, 10.5])
    assert list(ind.bullish_engulfing(o, o + 1, o - 1, c)) == [False, True]
    assert list(ind.bullish_engulfing(o, o + 1, o - 1, np.array([9.5, 9.9]))) == [False, False]
    assert list(ind.down_streak(np.array([5, 4, 3, 3, 2.0]))) == [0, 1, 2, 0, 1]


def test_pivots_reported_at_confirmation_bar():
    h = np.array([1, 2, 3, 10, 3, 2, 1, 1.0])
    p = ind.pivots(h, h - 1, 2, 2)
    assert not p["ph"][3] and p["ph"][5]
    assert p["ph_bar"][5] == 3 and p["ph_price"][5] == 10
    assert np.isnan(p["last_ph"][4]) and p["last_ph"][7] == 10
    # plateau: only the first of two equal highs is a pivot
    q = ind.pivots(np.array([1, 5, 5, 1, 1.0]), np.zeros(5), 1, 1)
    assert list(q["ph"]) == [False, False, True, False, False] and q["ph_bar"][2] == 1


def test_double_bottom_constructed():
    c = np.concatenate([np.linspace(100, 90, 11), np.linspace(92, 98, 4), np.linspace(96.025, 90.1, 4),
                        np.linspace(91.5875, 102, 8)])
    h, l, cl = _hlc(c)
    db = ind.double_bottom(h, l, cl, lag=2, tol=0.0025, min_sep=3, max_sep=12)
    sig = np.flatnonzero(db["signal"])
    assert list(sig) == [24]
    assert db["neckline"][24] == pytest.approx(98.5)
    assert db["bar1"][24] == 10 and db["bar2"][24] == 18
    # a tighter tolerance rejects the pattern
    assert not ind.double_bottom(h, l, cl, lag=2, tol=0.0005)["signal"].any()


def test_head_and_shoulders_constructed():
    c = np.concatenate([np.linspace(95, 105, 11), np.linspace(103.75, 100, 4), np.linspace(101.6667, 110, 6),
                        np.linspace(108.4167, 100.5, 6), np.linspace(101.46, 105.3, 5), np.linspace(104.3, 96.3, 9)])
    h, l, cl = _hlc(c)
    hs = ind.head_and_shoulders(h, l, cl, lag=3, atr_n=14, head_atr=1.0, sym_atr=1.0)
    sig = np.flatnonzero(hs["signal"])
    assert len(sig) == 1
    t = sig[0]
    # neckline through lows (bar 14, 99.5) and (bar 26, 100.0)
    assert hs["neckline"][t] == pytest.approx(99.5 + (t - 14) * 0.5 / 12)
    assert hs["rs_high"][t] == pytest.approx(105.8) and hs["head_high"][t] == pytest.approx(110.5)
    assert t >= 31 + 3  # never before the right shoulder is confirmed
    assert cl[t] < hs["neckline"][t] and cl[t - 1] >= 99.5 + (t - 1 - 14) * 0.5 / 12


def test_flag_pattern_constructed():
    c = np.concatenate([100 + 0.25 * np.arange(1, 11), 102.5 - 0.1 * np.arange(1, 9), [102.7]])
    o = np.concatenate([[100], c[:-1]])
    h = np.maximum(o, c) + 0.05
    l = np.minimum(o, c) - 0.05
    v = np.concatenate([np.full(10, 1000.0), np.full(8, 200.0), [500.0]])
    sess = np.zeros(len(c))
    f = ind.flag_pattern(o, h, l, c, v, sess, impulse_pct=0.02, impulse_bars=45, retrace=(0.2, 0.5), flag_bars=(5, 15))
    assert list(np.flatnonzero(f["signal"])) == [18]
    assert f["flag_high"][18] == pytest.approx(102.55) and f["flag_low"][18] == pytest.approx(101.65)
    v2 = v.copy(); v2[10:18] = 2000
    assert not ind.flag_pattern(o, h, l, c, v2, sess, 0.02, 45, (0.2, 0.5), (5, 15))["signal"].any()


def test_darvas_and_vcp_constructed():
    n = 30
    H = np.full(n, 11.0); L = np.full(n, 10.0); C = np.full(n, 10.5)
    H[15], C[15] = 12.0, 11.5
    H[25], L[25], C[25] = 12.6, 10.5, 12.5
    d = ind.darvas_box(H, L, C, lookback=20, recent=5, depth_atr=2.0, atr_n=20)
    assert list(np.flatnonzero(d["signal"])) == [25] and d["resistance"][25] == 12.0
    r = np.concatenate([np.full(10, 3.0), np.full(10, 2.0), np.full(10, 1.0)])
    v = ind.volatility_contraction(10 + r / 2, 10 - r / 2, np.full(30, 10.0), n=10, n_means=3)
    assert v[29] and not v[28]


# ---------------------------------------------------------------- intraday
def _bars(rows, day="2024-03-05"):
    """rows: list of (HH:MM, o, h, l, c, v)."""
    idx = [pd.Timestamp(f"{day} {t}").tz_localize("America/New_York").tz_convert("UTC") for t, *_ in rows]
    df = pd.DataFrame([r[1:] for r in rows], columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex(idx))
    df["session"] = "RTH"
    return df


def test_session_vwap_hand_and_reset():
    a = _bars([("09:30", 10, 11, 9, 10, 1), ("09:31", 11, 12, 10, 11, 2), ("09:32", 12, 13, 11, 12, 1)])
    b = _bars([("09:30", 50, 51, 49, 50, 5), ("09:31", 50, 51, 49, 50, 5)], day="2024-03-06")
    out = ind.session_vwap(pd.concat([a, b]))
    assert out.vwap.iloc[2] == pytest.approx(11.0)
    assert out.sd.iloc[2] == pytest.approx(np.sqrt(0.5))
    assert out.vwap.iloc[3] == pytest.approx(50.0) and out.sd.iloc[3] == pytest.approx(0.0)
    assert out.z.iloc[1] == pytest.approx((11 - 32 / 3) / np.sqrt((1 * (10 - 32 / 3) ** 2 + 2 * (11 - 32 / 3) ** 2) / 3))
    assert ind.session_vwap(a, price="close").vwap.iloc[1] == pytest.approx(32 / 3)


def test_opening_range_available_only_when_complete():
    rows = [(f"09:{30 + i:02d}", 10, 10 + i, 9 - i * 0.1, 10, 1) for i in range(8)]
    orr = ind.opening_range(_bars(rows), minutes=5)
    assert orr.or_high.iloc[:4].isna().all()
    assert orr.or_high.iloc[4] == 14 and orr.or_low.iloc[4] == pytest.approx(8.6)
    assert orr.or_high.iloc[7] == 14  # bars after the window do not extend it


def test_relative_volume_uses_prior_sessions_only():
    days = ["2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07"]
    vols = [100, 200, 300, 400]
    df = pd.concat([_bars([("09:30", 10, 11, 9, 10, v), ("09:31", 10, 11, 9, 10, v)], d) for d, v in zip(days, vols)])
    rv = ind.relative_volume(df, lookback=3)
    assert rv.expected.iloc[6] == pytest.approx(200) and rv.rvol.iloc[6] == pytest.approx(2.0)
    assert rv.expected.iloc[:6].isna().all()  # fewer than 3 prior sessions
    cum = ind.relative_volume(df, lookback=3, cumulative=True)
    assert cum.actual.iloc[7] == 800 and cum.expected.iloc[7] == pytest.approx(400)
    lo = ind.relative_volume(df, lookback=3, min_sessions=1)
    assert lo.expected.iloc[2] == pytest.approx(100)  # only session 1 is prior to session 2


def test_session_extremes_prior_bar():
    df = _bars([("09:30", 10, 11, 9, 10, 1), ("09:31", 10, 12, 9.5, 11, 1), ("09:32", 11, 11.5, 8, 9, 1)])
    e = ind.session_extremes(df)
    assert np.isnan(e.hod_prior.iloc[0]) and e.hod_prior.iloc[2] == 12 and e.hod.iloc[2] == 12
    assert e.lod.iloc[2] == 8 and e.lod_prior.iloc[2] == 9


def test_adv_and_dollar_volume():
    v = pd.Series([1, 2, 3, 4.0]); c = pd.Series([10, 10, 20, 20.0])
    assert ind.adv(v, 2).iloc[3] == 3.5 and ind.adv(v, 2, lag=1).iloc[3] == 2.5
    assert ind.dollar_volume(c, v, 2).iloc[3] == pytest.approx((60 + 80) / 2)
