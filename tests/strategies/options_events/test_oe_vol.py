"""S165-S170 (daily vol-signal option strategies) mechanics on SYNTHETIC EOD chains."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlasbt.core.types import Fidelity, RunStatus
from atlasbt.strategies.options_events import S165, S166, S167, S168, S169, S170
from atlasbt.strategies.options_events_vol import TOP5_SECTORS

from .oe_fix import (close_of, close_snaps, daily_bars, fills, option_book, order_key, orders, run, sess)

SESS = sess("2024-01-02", "2024-03-15")
EXPS = [(e, "PHYSICAL", "EUROPEAN", None) for e in pd.date_range("2024-01-05", "2024-07-31", freq="14D")]
LB = 10                                                     # short lookback for mechanics tests


def day(i):
    return SESS[i]


def by_day(values: dict, default):
    """Piecewise-constant daily parameter: {session index: value} applies from that session on."""
    keys = sorted(values)

    def f(t):
        d = t.tz_convert("America/New_York").tz_localize(None).normalize()
        v = default
        for k in keys:
            if d >= SESS[k]:
                v = values[k]
        return v
    return f


def chain_spec(und="XYZ", vol=lambda t, K, e, Ty: 0.20, spot=100.0, strikes=np.arange(70.0, 131.0, 2.0),
               closes=None, extra_bars=None):
    snaps = close_snaps(SESS)
    optq, insts = option_book(und, EXPS, strikes, snaps, lambda t: spot, vol, hs=lambda p: 0.02, fidelity=2,
                              max_dte=100)
    c = closes if closes is not None else np.full(len(SESS), spot)
    bars = {(und, "1d"): daily_bars(SESS, c)}
    bars.update(extra_bars or {})
    return {"bars": bars, "option_quotes": optq, "instruments": insts}


def merge(*specs):
    out = {"bars": {}, "option_quotes": {}, "instruments": {}}
    for s in specs:
        for k in out:
            out[k].update(s[k])
    return out


def sub(cls, **attrs):
    attrs.setdefault("LOOKBACK", LB)
    attrs.setdefault("universe", ["XYZ"])
    return type(cls.__name__ + "T", (cls,), attrs)


def combo(res, prefix):
    o = orders(res)
    return o[o["tag"].astype(str).str.startswith(prefix)]


def sub_at(o, i=0):
    return pd.Timestamp(o["submitted_at"].iloc[i])


def legs(res, prefix):
    f = fills(res, prefix)
    return sorted(zip(f["symbol"].str[12:13], f["symbol"].str[13:].astype(int) / 1000.0, np.sign(f["quantity"])))


# ---------------------------------------------------------------------------- S165
def s165_spec(near_miss=False):
    path = {15: 0.30, 20: 0.15}
    if near_miss:
        path = {5: 0.32, 6: 0.20, 7: 0.32, 8: 0.20, 15: 0.30, 16: 0.20}   # spikes before the warm-up ends
    v = by_day(path, 0.20)
    return chain_spec(vol=lambda t, K, e, Ty: v(t))


def test_synthetic_S165_iv_percentile_condor():
    r = run(sub(S165)(), s165_spec(), SESS[0], day(24))
    op = combo(r, "S165:open")
    assert len(op) == 1 and sub_at(op) == close_of(day(15))           # 30D ATM IV jumps to the top decile
    assert set(fills(r, "S165:open")["ts"]) == {close_of(day(16))}   # next EOD snapshot
    lg = legs(r, "S165:open")
    assert [(rt, s) for rt, _, s in lg] == [("C", -1), ("C", 1), ("P", 1), ("P", -1)]
    ks = [k for _, k, _ in lg]
    assert ks[2] < ks[3] < 100 < ks[0] < ks[1]
    cl = combo(r, "S165:close")
    assert cl["tag"].iloc[0] == "S165:close:iv_pct_exit" and sub_at(cl) == close_of(day(20))
    assert set(fills(r, "S165:close")["ts"]) == {close_of(day(21))}


def test_synthetic_S165_near_miss_and_spread_filter():
    r = run(sub(S165)(), s165_spec(near_miss=True), SESS[0], day(18))
    assert len(combo(r, "S165")) == 0                                # 80th percentile < 90
    r = run(sub(S165)(max_rel_spread=0.01), s165_spec(), SESS[0], day(18))
    assert len(combo(r, "S165")) == 0                                # spread filter fails on the wings


# ---------------------------------------------------------------------------- S166 / S167 (skew)
def skew_spec(shock_day, shock_slope, exit_day=None, exit_slope=-0.44):
    hist = {i: (-0.40 if i % 2 == 0 else -0.48) for i in range(0, shock_day)}
    hist[shock_day] = shock_slope
    if exit_day is not None:
        for i in range(shock_day + 1, exit_day):
            hist[i] = shock_slope
        hist[exit_day] = exit_slope
    sl = by_day(hist, -0.44)
    return chain_spec(vol=lambda t, K, e, Ty: max(0.20 + sl(t) * np.log(K / 100.0), 0.05))


def test_synthetic_S166_skew_flattening_put_spread():
    r = run(sub(S166)(), skew_spec(15, -0.10, exit_day=19, exit_slope=-0.55), SESS[0], day(22))
    op = combo(r, "S166:open")
    assert len(op) == 1 and sub_at(op) == close_of(day(15))
    assert set(fills(r, "S166:open")["ts"]) == {close_of(day(16))}
    (r1, k1, s1), (r2, k2, s2) = legs(r, "S166:open")
    assert r1 == r2 == "P" and k1 < k2 < 100 and s1 == -1 and s2 == 1     # long ~25d put, short lower ~10d put
    cl = combo(r, "S166:close")
    assert cl["tag"].iloc[0] == "S166:close:skew_z_exit" and sub_at(cl) == close_of(day(19))


def test_synthetic_S166_near_miss():
    r = run(sub(S166)(), skew_spec(15, -0.37), SESS[0], day(18))
    assert len(combo(r, "S166")) == 0


def test_synthetic_S167_risk_reversal_with_share_hedge():
    r = run(sub(S167)(), skew_spec(15, -0.90, exit_day=20, exit_slope=-0.44), SESS[0], day(24))
    op = combo(r, "S167:open")
    assert len(op) == 1 and sub_at(op) == close_of(day(15))
    assert set(fills(r, "S167:open")["ts"]) == {close_of(day(16))}
    lg = legs(r, "S167:open")
    assert [(rt, s) for rt, _, s in lg] == [("C", 1), ("P", -1)]          # buy 25d call, sell 25d put
    n = int(abs(fills(r, "S167:open")["quantity"].iloc[0]))
    o = orders(r)
    h = o[o["tag"] == "delta_hedge"]
    assert sub_at(h) == close_of(day(16))                                # hedge at the close after the fill
    q0 = float(h["quantity"].iloc[0])
    assert -0.75 * 100 * n < q0 < -0.3 * 100 * n                          # bullish RR -> short shares
    hf = fills(r)
    hf = hf[hf["tag"] == "delta_hedge"]
    assert pd.Timestamp(hf["ts"].iloc[0]) < close_of(day(17))           # executed at the next bar (open)
    cl = combo(r, "S167:close")
    assert cl["tag"].iloc[0] == "S167:close:skew_z_exit" and sub_at(cl) == close_of(day(20))
    unw = o[o["tag"] == "hedge_unwind"]
    assert len(unw) == 1 and sub_at(unw) == close_of(day(21))           # shares flattened once options are gone


def test_synthetic_S167_near_miss_and_weekly_hedge():
    r = run(sub(S167)(), skew_spec(15, -0.50), SESS[0], day(18))
    assert len(combo(r, "S167")) == 0
    r = run(sub(S167)(hedge="weekly"), skew_spec(15, -0.90), SESS[0], day(24))
    h = orders(r)
    h = h[h["tag"] == "delta_hedge"]
    days = [pd.Timestamp(t).tz_convert("America/New_York").tz_localize(None).normalize() for t in h["submitted_at"]]
    assert days[0] == day(16)                                            # initial hedge right after the fill
    assert all(d.dayofweek == 4 for d in days[1:])                        # then only on the last session of the week


# ---------------------------------------------------------------------------- S168
def term_spec(shock_day, short_vol, exit_day=None):
    sv = by_day({shock_day: short_vol} | ({exit_day: 0.20} if exit_day else {}), 0.20)
    return chain_spec(vol=lambda t, K, e, Ty: sv(t) if Ty * 365 <= 45 else 0.20)


def test_synthetic_S168_term_structure_calendar():
    r = run(sub(S168)(), term_spec(15, 0.27, exit_day=19), SESS[0], day(22))
    op = combo(r, "S168:open")
    assert len(op) == 1 and sub_at(op) == close_of(day(15))
    f = fills(r, "S168:open")
    assert set(f["ts"]) == {close_of(day(16))}
    exps = sorted(set(f["symbol"].str[6:12]))
    sides = dict(zip(f["symbol"].str[6:12], np.sign(f["quantity"])))
    assert len(exps) == 2 and sides[exps[0]] == -1 and sides[exps[1]] == 1      # sell near, buy far
    assert len(set(f["symbol"].str[13:])) == 1 and set(f["symbol"].str[12:13]) == {"C"}   # same strike calls
    cl = combo(r, "S168:close")
    assert cl["tag"].iloc[0] == "S168:close:term_spread_exit" and sub_at(cl) == close_of(day(19))
    r = run(sub(S168)(), term_spec(15, 0.23), SESS[0], day(18))           # 3 vol pts < 5: near miss
    assert len(combo(r, "S168")) == 0


# ---------------------------------------------------------------------------- S169
def s169_spec(iv_path, n_hist=40):
    pre = sess("2022-06-01", "2023-12-29")[-n_hist:]
    allsess = pre + SESS
    closes = 100.0 * np.cumprod(np.r_[1.0, np.where(np.arange(len(allsess) - 1) % 2 == 0, 1.005, 1 / 1.005)])
    spec = chain_spec(vol=lambda t, K, e, Ty: iv_path(t))
    spec["bars"][("XYZ", "1d")] = daily_bars(allsess, closes)
    return spec


def test_synthetic_S169_forecast_vs_implied():
    iv = by_day({10: 0.085}, 0.20)
    r = run(sub(S169)(forecast="rv20"), s169_spec(iv), SESS[0], day(14))
    op = combo(r, "S169:open")
    assert len(op) == 1 and sub_at(op) == close_of(day(0))               # edge ~12 vol pts > 5 from day one
    (r1, k1, s1), (r2, k2, s2) = legs(r, "S169:open")
    assert r1 == r2 == "P" and k1 < k2 and s1 == 1 and s2 == -1           # sell -.25, buy -.10
    cl = combo(r, "S169:close")
    assert cl["tag"].iloc[0] == "S169:close:edge_exit" and sub_at(cl) == close_of(day(10))
    r = run(sub(S169)(forecast="rv20"), s169_spec(lambda t: 0.12), SESS[0], day(5))
    assert len(combo(r, "S169")) == 0                                    # edge ~4 < 5


def test_S169_har_forecast_walk_forward():
    r = run(sub(S169)(), s169_spec(lambda t: 0.30, n_hist=320), SESS[0], day(3))
    assert r.status == RunStatus.OK
    assert len(combo(r, "S169:open")) == 1                               # HAR forecast ~16% vs IV 30%
    r = run(sub(S169)(), s169_spec(lambda t: 0.20, n_hist=320), SESS[0], day(3))
    assert len(combo(r, "S169")) == 0                                    # edge ~4 vol pts < 5


# ---------------------------------------------------------------------------- S170
def s170_spec():
    specs = [chain_spec(u, strikes=np.arange(80.0, 121.0, 2.0)) for u in ["SPY"] + TOP5_SECTORS]
    return merge(*specs)


S170T = sub(S170, universe=["SPY"] + TOP5_SECTORS)


def test_synthetic_S170_dispersion_month_end():
    r = run(S170T(), s170_spec(), SESS[0], day(45))
    op = combo(r, "S170:open")
    me = close_of(pd.Timestamp("2024-01-31"))
    first = op[pd.to_datetime(op["submitted_at"], utc=True) == me]
    assert len(first) == 6
    later = op[pd.to_datetime(op["submitted_at"], utc=True) > me]
    assert set(pd.to_datetime(later["submitted_at"], utc=True)) <= {close_of(pd.Timestamp("2024-02-29"))}
    f = fills(r, "S170:open")
    f = f[f["ts"] == close_of(pd.Timestamp("2024-02-01"))]
    spy = f[f["symbol"].str.startswith("SPY")]
    sec = f[~f["symbol"].str.startswith("SPY")]
    assert (spy["quantity"] < 0).all() and (sec["quantity"] > 0).all()   # short index, long sector straddles
    assert set(sec["symbol"].str[:6].str.strip()) == set(TOP5_SECTORS)
    hed = orders(r)
    hed = hed[hed["tag"] == "delta_hedge"]
    assert len(hed) >= 1 and min(pd.to_datetime(hed["submitted_at"], utc=True)) == close_of(pd.Timestamp("2024-02-01"))
    cl = combo(r, "S170:close")
    assert len(cl) == 6 and set(cl["tag"]) == {"S170:close:dte"}
    t_close = max(pd.to_datetime(cl["submitted_at"], utc=True))
    assert t_close == close_of(pd.Timestamp("2024-02-23"))                # first close with DTE <= 7 (03-01 expiry)
    assert len(later) == 0 or min(pd.to_datetime(later["submitted_at"], utc=True)) > t_close   # no overlap


# ---------------------------------------------------------------------------- gates, lookahead, smoke
def test_vol_signals_refuse_synthetic_chains():
    spec = s165_spec()
    for q in spec["option_quotes"].values():
        q["fidelity"] = 3
    r = run(sub(S165)(), spec, SESS[0], day(18))
    assert len(combo(r, "S165")) == 0
    r = run(sub(S165)(allow_synthetic=True), spec, SESS[0], day(18))
    assert len(combo(r, "S165:open")) == 1


@pytest.mark.parametrize("cls,spec_fn,cut", [
    (sub(S165), s165_spec, 17),
    (sub(S166), lambda: skew_spec(15, -0.10, exit_day=19, exit_slope=-0.55), 17),
    (sub(S167), lambda: skew_spec(15, -0.90, exit_day=20), 18),
    (sub(S168), lambda: term_spec(15, 0.27, exit_day=19), 17),
    (sub(S169, params=dict(S169.params, forecast="rv20")), lambda: s169_spec(by_day({10: 0.085}, 0.20)), 12),
    (S170T, s170_spec, 25),
])
def test_lookahead_vol(cls, spec_fn, cut):
    spec = spec_fn()
    end = day(min(cut + 6, len(SESS) - 1))
    t = close_of(day(cut))
    full = run(cls(), spec, SESS[0], end)
    trunc = run(cls(), spec, SESS[0], end, cut=t)
    a, b = order_key(orders(full), t), order_key(orders(trunc), t)
    assert a == b and len(a) >= 1


@pytest.mark.parametrize("cls", [S165, S166, S167, S168, S169])
def test_smoke_vol(cls):
    r = run(sub(cls)(), s165_spec(), SESS[0], day(12))
    assert r.status == RunStatus.OK
    r = run(sub(cls)(), s165_spec(), SESS[0], day(3), option_fidelity=Fidelity.NA)
    assert r.status == RunStatus.NOT_TESTABLE
