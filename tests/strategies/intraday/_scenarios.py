"""Hand-built SYNTHETIC patterns, one trigger + one near miss per strategy
(mechanics only). Expected levels are written out from the bar values (or
recomputed with atlasbt.indicators where the rule involves an indicator)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from atlasbt.core.clock import et
from atlasbt.core.types import MarketEvent
from atlasbt.engine.testing import synthetic_daily_bars
from atlasbt.indicators import atr as ind_atr
from atlasbt.indicators import intraday as iind

from ._synth import (
    DAY, SYM, V1, V5, Scenario, day_frame, daily_ramp, flat, frame, history, path, pre_frame,
    prior_sessions, starts,
)

HIST = prior_sessions(DAY, 22)


def _h1(**kw):
    return history(HIST, 1, **kw)


def _h5(**kw):
    return history(HIST, 5, **kw)


def _one(freq, day_rows, hist=None, sym=SYM, **dkw):
    f = 1 if freq == "1m" else 5
    h = hist if hist is not None else (_h1() if f == 1 else _h5())
    return {(sym, freq): pd.concat([h, day_frame(DAY, day_rows, f, **dkw)])}


def _atr_expected(intra: pd.DataFrame, upto: pd.Timestamp, k_min=None, n=14) -> float:
    """Wilder ATR(n) of completed bars (optionally k-minute buckets) up to `upto` (bar start inclusive)."""
    d = intra[intra.index <= upto]
    d = d[d["session"] == "RTH"]
    if k_min:
        from atlasbt.strategies.intraday_base import aggregate_frame
        d = aggregate_frame(d, k_min)
    d = d.iloc[-200:]
    return float(ind_atr(d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy(), n)[-1])


def _quote(ts_et: str, bid: float, ask: float) -> pd.DataFrame:
    t = et(DAY, ts_et[:5]) + pd.Timedelta(seconds=int(ts_et[6:8]) if len(ts_et) > 5 else 0)
    return pd.DataFrame({"bid": [bid], "ask": [ask], "bid_size": [500.0], "ask_size": [500.0]},
                        index=pd.DatetimeIndex([t], name="ts"))


def _or_rows(n=15, H=100.5, L=99.5, p=100.0, v=V1):
    rows = flat(p, n, rng=0.1, v=v)
    rows[3] = (p, H, p - 0.1, p, v)
    if n > 8:
        rows[8] = (p, p + 0.1, L, p, v)
    return rows


# ======================================================================== S001-S010
def s001():
    v = 2e5
    rows = [(100.0, 100.2, 99.8, 100.1, v), (100.1, 100.3, 100.0, 100.2, v), (100.2, 100.5, 100.1, 100.3, v),
            (100.3, 100.4, 100.1, 100.2, v), (100.2, 100.4, 100.0, 100.3, v), (100.3, 100.45, 100.2, 100.4, V1),
            (100.4, 100.7, 100.35, 100.6, V1), (100.62, 100.7, 100.6, 100.65, V1)]
    trig = Scenario("S001", _one("1m", rows), signal="09:37", side=1, stop=99.80)
    near_rows = [r[:4] + (1.2e5,) if i < 5 else r for i, r in enumerate(rows)]
    return trig, replace(trig, intraday=_one("1m", near_rows))


def s002():
    rows = _or_rows() + [(100.0, 100.05, 99.55, 99.6, V1), (99.6, 99.65, 99.35, 99.4, V1), (99.38, 99.42, 99.33, 99.4, V1)]
    trig = Scenario("S002", _one("1m", rows), signal="09:47", side=-1, stop=100.5, target_r=2.0)
    near = _or_rows() + [(100.0, 100.05, 99.55, 99.6, V1), (99.6, 99.65, 99.51, 99.52, V1)]
    return trig, replace(trig, intraday=_one("1m", near))


def s003():
    rows = _or_rows() + [(100.0, 100.7, 99.95, 100.6, V1), (100.6, 100.65, 100.45, 100.5, V1),
                         (100.5, 100.62, 100.4, 100.58, V1), (100.6, 100.65, 100.55, 100.6, V1)]
    trig = Scenario("S003", _one("1m", rows), signal="09:48", side=1, stop=100.0, target_r=2.0)
    near = _or_rows() + [(100.0, 100.7, 99.95, 100.6, V1), (100.6, 100.7, 100.55, 100.65, V1)]
    return trig, replace(trig, intraday=_one("1m", near, fill=100.8, fill_rng=0.05))


def s004():
    closes = [100.15, 100.30, 100.45, 100.60, 100.55, 100.70, 100.85, 101.00, 101.15, 101.30]
    rows = path(100.0, closes, rng=0.02) + [(101.3, 101.32, 101.28, 101.3, V1)]

    def chk(r, sc):
        tx = r.fills[r.fills.tag == "time_exit"]
        assert len(tx) == 1 and tx.iloc[0].ts == et(DAY, "11:30")
    trig = Scenario("S004", _one("1m", rows, fill_rng=0.02), signal="09:40", side=1, stop=99.98, check=chk)
    near_closes = [100.15, 100.30, 100.25, 100.40, 100.35, 100.50, 100.65, 100.60, 100.85, 101.10]
    return trig, replace(trig, intraday=_one("1m", path(100.0, near_closes, rng=0.02)))


def s004_vwap_exit():
    closes = [100.15, 100.30, 100.45, 100.60, 100.55, 100.70, 100.85, 101.00, 101.15, 101.30]
    rows = path(100.0, closes, rng=0.02) + [(101.3, 101.32, 101.28, 101.3, V1), (101.3, 101.3, 100.48, 100.5, V1),
                                            (100.5, 100.52, 100.38, 100.4, V1)]

    def chk(r, sc):
        ex = r.fills[r.fills.tag == "vwap_exit"]
        assert len(ex) == 1 and ex.iloc[0].ts == et(DAY, "09:43") and ex.iloc[0].price == 100.4
    return Scenario("S004", _one("1m", rows), signal="09:40", side=1, stop=99.98, check=chk)


def _gap_pre(price, v=30_000.0):
    return pre_frame(DAY, price, v=v)


def s005():
    rows = [(103.2, 103.3, 103.1, 103.25, V1), (103.25, 103.4, 103.2, 103.35, V1), (103.35, 103.45, 103.3, 103.4, V1),
            (103.4, 103.5, 103.35, 103.45, V1), (103.45, 103.55, 103.4, 103.5, V1), (103.5, 103.55, 103.45, 103.5, V1)]
    intra = {(SYM, "1m"): pd.concat([_h1(), _gap_pre(103.0), day_frame(DAY, rows, 1)]).sort_index()}
    trig = Scenario("S005", intra, quotes={SYM: _quote("09:29:30", 102.9, 103.1)}, signal="09:35", side=1,
                    stop=103.1, target_r=2.0)
    return trig, replace(trig, quotes={SYM: _quote("09:29:30", 101.4, 101.6)})


def s006():
    rows = [(96.8, 96.9, 96.7, 96.75, V1), (96.75, 96.8, 96.6, 96.65, V1), (96.65, 96.7, 96.55, 96.6, V1),
            (96.6, 96.65, 96.5, 96.55, V1), (96.55, 96.6, 96.45, 96.5, V1), (96.5, 96.55, 96.45, 96.5, V1)]
    intra = {(SYM, "1m"): pd.concat([_h1(), _gap_pre(97.0), day_frame(DAY, rows, 1)]).sort_index()}
    trig = Scenario("S006", intra, quotes={SYM: _quote("09:29:30", 96.9, 97.1)}, signal="09:35", side=-1,
                    stop=96.9, target_r=2.0)
    near_intra = {(SYM, "1m"): pd.concat([_h1(), _gap_pre(97.0, v=10_000.0), day_frame(DAY, rows, 1)]).sort_index()}
    return trig, replace(trig, intraday=near_intra)


def s007(v=3e5):
    pre = pre_frame(DAY, 101.0)
    pre.loc[et(DAY, "05:00"), "high"] = 102.0          # outside 07:00-09:29: must be ignored
    pre.loc[et(DAY, "08:00"), "high"] = 101.5          # P
    rows = [(101.2, 101.3, 101.1, 101.25, v), (101.25, 101.35, 101.2, 101.3, v), (101.3, 101.4, 101.25, 101.35, v),
            (101.35, 101.45, 101.3, 101.4, v), (101.4, 101.48, 101.35, 101.45, v), (101.45, 101.65, 101.4, 101.6, v),
            (101.6, 101.65, 101.55, 101.6, V1)]
    intra = {(SYM, "1m"): pd.concat([_h1(), pre, day_frame(DAY, rows, 1)]).sort_index()}
    return Scenario("S007", intra, signal="09:36", side=1, stop=101.1, target_r=2.0)


def s007_pair():
    trig = s007()
    near = s007(v=V1)
    return trig, near


def s008():
    up = [95 + 0.25 * k for k in range(22)]
    rows = flat(100.25, 15, rng=0.05)
    rows[7] = (100.25, 100.3, 100.0, 100.25, V1)
    rows += path(100.25, [100.32, 100.40, 100.48, 100.55, 100.6], rng=0.03) + [(100.6, 100.63, 100.57, 100.6, V1)]
    trig = Scenario("S008", _one("1m", rows, hist=_h1(prices=up)), signal="09:50", side=1, stop=100.0,
                    target_r=3.0)
    down = list(reversed(up))
    return trig, replace(trig, intraday=_one("1m", rows, hist=_h1(prices=down)))


def s009():
    rows = [(100.0, 100.2, 99.8, 100.0, V5)] * 7 + [(100.0, 100.05, 99.95, 100.0, V5), (100.0, 100.25, 99.98, 100.2, V5),
                                                  (100.2, 100.2, 100.2, 100.2, V5)]

    def chk(r, sc):
        tx = r.fills[r.fills.tag == "time_exit"]
        assert len(tx) == 1 and tx.iloc[0].ts == et(DAY, "10:45")
    trig = Scenario("S009", _one("5m", rows), signal="10:15", side=1, stop=99.95, target_r=2.0, check=chk)
    near = rows[:8] + [(100.0, 100.25, 99.98, 100.04, V5)]
    return trig, replace(trig, intraday=_one("5m", near), check=None)


def s010(flag_v=1e5):
    imp = path(100.0, [100 + 0.25 * (k + 1) for k in range(10)], rng=0.02, v=3e5)
    flag_c = [102.4, 102.3, 102.2, 102.1, 102.0, 102.1]
    fl, o = [], 102.5
    for c in flag_c:
        fl.append((o, max(o, c) + 0.01, min(o, c) - 0.01, c, flag_v))
        o = c
    rows = imp + fl + [(102.1, 102.62, 102.09, 102.6, 1e5), (102.6, 102.62, 102.58, 102.6, V1)]
    return Scenario("S010", _one("1m", rows), signal="09:47", side=1, stop=101.99, target_r=2.0)


def s010_pair():
    return s010(), s010(flag_v=6e5)


def s011():
    rows = _or_rows(12, p=100.0, v=V5) + [(100.0, 100.45, 99.95, 100.4, V5), (100.4, 100.75, 100.35, 100.7, V5)]
    trig = Scenario("S011", _one("5m", rows), signal="10:40", side=1, stop=100.0)
    near = _or_rows(12, p=100.0, v=V5) + [(100.0, 100.45, 99.95, 100.4, V5), (100.4, 100.5, 100.35, 100.5, V5)]
    return trig, replace(trig, intraday=_one("5m", near))


def s012(low6=100.36):
    rows = path(100.0, [100.1, 100.2, 100.3, 100.4, 100.5, 100.6], rng=0.05, v=V5)
    rows += [(100.6, 100.65, low6, 100.55, V5), (100.55, 100.75, 100.5, 100.7, V5), (100.7, 100.72, 100.68, 100.7, V5)]
    return Scenario("S012", _one("5m", rows), signal="10:10", side=1, stop=low6)


def s012_pair():
    return s012(), s012(low6=100.5)


def s013(rally_v=4e5):
    rows = flat(100.0, 30, rng=0.05)
    rows += [(100.0, 100.0, 99.45, 99.5, V1)] + [(99.5, 99.55, 99.45, 99.5, V1)] * 19
    rows += [(99.5, 100.0, 99.45, 99.95, rally_v)] + [(99.95, 100.05, 99.9, 100.0, rally_v)] * 4
    rows += [(100.0, 100.02, 99.98, 100.0, V1)]
    return Scenario("S013", _one("1m", rows), signal="10:25", side=1, stop=99.45)


def s013_pair():
    return s013(), s013(rally_v=V1)


def s014(c8=100.6):
    rows = flat(100.0, 6, rng=0.1, v=V5) + [(100.0, 100.25, 99.95, 100.2, V5), (100.2, 100.45, 100.15, 100.4, V5),
                                           (100.4, max(100.65, c8 + 0.05), 100.35, c8, V5)]
    rows += [(c8, c8, c8, c8, V5)]
    intra = _one("5m", rows)
    df = intra[(SYM, "5m")]
    sig = et(DAY, "10:10")
    a = _atr_expected(df, sig)
    vw = float(iind.session_vwap(df[df.index.normalize() >= df.index[-1].normalize()][:9])["vwap"].iloc[-1])
    sc = Scenario("S014", intra, signal="10:15", side=1, stop=max(vw, c8 - a), target_r=2.0)

    def chk(r, s):
        tx = r.fills[r.fills.tag == "time_exit"]
        assert len(tx) == 1 and tx.iloc[0].ts == et(DAY, "10:45")
    return replace(sc, check=chk)


def s014_pair():
    return s014(), replace(s014(c8=100.35), check=None)


# ======================================================================== S015-S020
def s015(side=1, gate_ok=True, quotes=True):
    if side > 0:
        fh = [100 + (k + 1) / 60 for k in range(30)] if gate_ok else [100 - (k + 1) / 150 for k in range(30)]
        px, bid, ask = 100.8, 100.75, 100.85
    else:
        fh = [100 - (k + 1) / 60 for k in range(30)]
        px, bid, ask = 99.2, 99.15, 99.25
    rows = path(100.0, fh, rng=0.02)
    q = {SYM: _quote("15:24:30", bid, ask)} if quotes else {}
    qty = int(0.10 * 1_000_000 / px) * side

    def chk(r, sc):
        e = r.fills[r.fills.tag == "entry"]
        assert len(e) == 1 and e.iloc[0].quantity == qty
        fb = r.fills[r.fills.tag == "flat_by"]
        assert len(fb) == 1 and fb.iloc[0].ts == et(DAY, "15:55")
    return Scenario("S015", _one("1m", rows, fill=px), quotes=q, signal="15:30", side=side, check=chk)


def s015_pair():
    return s015(), replace(s015(gate_ok=False), check=None)


def s016(v=8e5):
    rows = flat(100.0, 47, rng=0.1, v=v)
    rows[3] = (100.0, 100.5, 99.9, 100.0, v)
    rows += [(100.0, 100.35, 99.95, 100.3, v), (100.3, 100.65, 100.25, 100.6, v), (100.6, 100.6, 100.6, 100.6, v)]
    return Scenario("S016", _one("5m", rows), signal="13:35", side=1, stop=99.9)


def s016_pair():
    return s016(), s016(v=V5)


def s017(low33=100.62):
    rows = flat(100.0, 30, rng=0.1)
    rows += [(100.0, 100.6, 99.95, 100.55, V1), (100.55, 100.8, 100.5, 100.75, V1), (100.75, 100.78, 100.65, 100.7, V1),
             (100.7, 100.74, low33, 100.66, V1), (100.66, 100.72, 100.6, 100.68, V1), (100.68, 100.75, 100.63, 100.7, V1),
             (100.7, 100.8, 100.69, 100.79, V1), (100.79, 100.8, 100.78, 100.79, V1)]
    return Scenario("S017", _one("1m", rows), signal="10:07", side=1, stop=min(100.6, low33), target_r=1.5)


def s017_pair():
    return s017(), s017(low33=100.3)


def s018(low7=100.45):
    rows = path(100.0, [100.2, 100.4, 100.6, 100.8, 101.0, 101.2], rng=0.05, v=V5)
    rows += [(101.2, 101.3, 101.15, 101.25, V5), (101.25, 101.3, low7, 100.9, V5), (100.9, 101.1, 100.85, 101.05, V5),
             (101.05, 101.05, 101.05, 101.05, V5)]
    return Scenario("S018", _one("5m", rows), signal="10:15", side=1, stop=min(low7, 100.85))


def s018_pair():
    return s018(), s018(low7=100.6)


def s019(top_close=103.0):
    names = [f"N{k}" for k in range(10)]
    intra = {}
    for s in names + ["SPY"]:
        if s == "N0":
            closes = [100 + (k + 1) * (top_close - 100) / 18 for k in range(18)]
            rows = path(100.0, closes, rng=0.05, v=V5)
            hi = max(r[1] for r in rows)
            rows += [(top_close, hi + 0.25, top_close - 0.02, hi + 0.15, V5)]
        elif s == "SPY":
            rows = path(100.0, [100 + (k + 1) * 0.5 / 18 for k in range(18)], rng=0.05, v=V5)
        else:
            rows = path(100.0, [100.2] * 18, rng=0.05, v=V5)
        intra[(s, "5m")] = pd.concat([_h5(), day_frame(DAY, rows, 5)])
    n0 = intra[("N0", "5m")]
    today = n0[n0.index >= et(DAY, "09:30")]
    stop = float(today["low"].iloc[13:19].min())
    return Scenario("S019", intra, universe=(), universe_name="SP500_TOP50_DV",
                    universes={"SP500_TOP50_DV": names}, signal="11:05", side=1, stop=stop, symbol="N0")


def s019_pair():
    return s019(), s019(top_close=101.3)


def s020(v=6e5):
    rows = flat(100.0, 45, rng=0.1) + [(100.0, 100.35, 99.98, 100.3, v), (100.3, 100.3, 100.3, 100.3, V1)]
    intra = _one("1m", rows, hist=_h1(rng=0.1))
    a = _atr_expected(intra[(SYM, "1m")], et(DAY, "10:15"))
    sc = Scenario("S020", intra, signal="10:16", side=1, stop=min(99.98, 100.3 - a), target_r=2.0)

    def chk(r, s):
        tx = r.fills[r.fills.tag == "time_exit"]
        assert len(tx) == 1 and tx.iloc[0].ts == et(DAY, "10:31")
    return replace(sc, check=chk)


def s020_pair():
    return s020(), replace(s020(v=4e5), check=None)


# ======================================================================== S021-S030, S116
def s021(open_=102.0):
    days = HIST
    h = [history(days[:-1], 5, price=106.0)]
    idx = starts(days[-1], 5)
    last = flat(106.0, 58, v=V5) + path(106.0, [106.0 - 0.3 * (k + 1) for k in range(20)], rng=0.05, v=V5)
    h.append(frame(last, idx, 5))
    hist = pd.concat(h)
    c0 = open_ - 0.3
    rows = [(open_, open_ + 0.3, open_ - 0.4, c0, V5), (c0, c0 + 0.05, c0 - 0.2, c0 - 0.1, V5)]
    rows += path(c0 - 0.1, list(np.linspace(c0 - 0.2, 100.2, 8)), rng=0.03, v=V5)
    rows += [(100.2, 100.25, 99.9, 99.95, V5)]

    def chk(r, sc):
        t = r.fills[r.fills.tag == "target"]
        assert len(t) == 1 and t.iloc[0].price == 100.0 and et(DAY, "10:20") <= t.iloc[0].ts < et(DAY, "10:25")
    return Scenario("S021", _one("5m", rows, hist=hist), signal="09:35", side=-1, stop=open_ + 0.3, target=100.0,
                    check=chk if open_ == 102.0 else None)


def s021_pair():
    return s021(), s021(open_=101.0)


def s022(fail_late=False):
    rows = [(100.4, 100.5, 100.35, 100.45, V1)] * 15
    rows[8] = (100.4, 100.45, 99.5, 100.4, V1)
    rows += [(100.45, 100.7, 100.4, 100.6, V1)]
    if fail_late:
        rows += [(100.55, 100.6, 100.52, 100.55, V1)] * 10
    rows += [(100.6, 100.62, 100.3, 100.35, V1), (100.35, 100.36, 100.3, 100.33, V1)]
    sig = "09:47" if not fail_late else None
    return Scenario("S022", _one("1m", rows), signal=sig, side=-1, stop=100.7, target=99.5)


def s022_pair():
    return s022(), s022(fail_late=True)


def _vsig_rows(sign):
    if sign > 0:
        rows = flat(100, 4, rng=0.05, v=V5, alt=0.2)
        rows += path(rows[-1][3], [99.9, 99.7, 99.3], 0.05, V5)
    else:
        rows = flat(100, 4, rng=0.05, v=V5, alt=0.2)
        rows += path(rows[-1][3], [100.1, 100.3, 100.7], 0.05, V5)
    return rows


def _daily_extra(trend: float, n=200):
    days = prior_sessions(HIST[0], n)
    closes = list(np.linspace(100 - trend, 100 - trend / n, n))
    return {SYM: daily_ramp(days, closes)}


def s023(trend=10.0):
    rows = _vsig_rows(1) + [(99.3, 99.35, 99.25, 99.3, V5), (99.3, 100.1, 99.28, 100.05, V5), (100.05,) * 4 + (V5,)]

    def chk(r, sc):
        ex = r.fills[r.fills.tag == "vwap_exit"]
        assert len(ex) == 1 and ex.iloc[0].ts == et(DAY, "10:15") and ex.iloc[0].price == 100.05
        assert len(r.fills[r.fills.tag == "stop"]) == 0          # no resting stop (next-bar stop)
    return Scenario("S023", _one("5m", rows), daily_extra=_daily_extra(trend), signal="10:05", side=1, check=chk)


def s023_pair():
    return s023(), replace(s023(trend=-10.0), check=None)


def s024(trend=-10.0):
    rows = _vsig_rows(-1) + [(100.7, 100.75, 100.65, 100.7, V5), (100.7, 100.72, 100.0, 100.1, V5), (100.1,) * 4 + (V5,)]
    intra = _one("5m", rows)
    df = intra[(SYM, "5m")]
    v7 = float(iind.session_vwap(df[df.index >= et(DAY, "09:30")].iloc[:8])["vwap"].iloc[-1])
    lim = np.floor(v7 * 100 + 1e-9) / 100

    def chk(r, sc):
        ex = r.fills[r.fills.tag == "vwap_exit"]
        assert len(ex) == 1 and ex.iloc[0].ts > et(DAY, "10:10") and ex.iloc[0].price == lim
    return Scenario("S024", intra, daily_extra=_daily_extra(trend), signal="10:05", side=-1, check=chk)


def s024_pair():
    return s024(), replace(s024(trend=10.0), check=None)


def s025(step=0.1):
    rows = path(100.0, [100 - step * (k + 1) for k in range(12)], rng=0.05, v=V5)
    c11 = rows[-1][3]
    rows += [(c11, c11 + 0.05, c11 - 0.15, c11 - 0.1, V5), (c11 - 0.1, c11 + 0.25, c11 - 0.12, c11 + 0.2, V5)]
    c13 = c11 + 0.2
    rows += [(c13, c13, c13, c13, V5)] * 6 + [(c13, 99.8, c13 - 0.02, 99.7, V5), (99.7, 99.7, 99.7, 99.7, V5)]
    intra = _one("5m", rows)
    df = intra[(SYM, "5m")]
    v19 = float(iind.session_vwap(df[df.index >= et(DAY, "09:30")].iloc[:20])["vwap"].iloc[-1])
    lim = np.ceil(v19 * 100 - 1e-9) / 100

    def chk(r, sc):
        ex = r.fills[r.fills.tag == "vwap_exit"]
        assert len(ex) == 1 and et(DAY, "11:10") <= ex.iloc[0].ts < et(DAY, "11:15") and ex.iloc[0].price == lim
    return Scenario("S025", intra, signal="10:40", side=1, stop=round(c11 - 0.15, 2), check=chk)


def s025_pair():
    return s025(), replace(s025(step=0.05), check=None)


def s026(low_c=98.5):
    rows = flat(100.0, 48, rng=0.05, v=V5)
    rows += path(100.0, list(np.linspace(100.0 - (100 - low_c) / 12, low_c, 12)), rng=0.05, v=V5)
    rows += [(low_c, low_c + 0.1, low_c - 0.2, low_c - 0.05, V5), (low_c - 0.05,) * 4 + (V5,)]
    intra = _one("5m", rows)
    ex_days = prior_sessions(HIST[0], 270)
    dx = synthetic_daily_bars(str(ex_days[0].date()), str(ex_days[-1].date()), start_price=100.0, daily_vol=0.005,
                              volume=39_000_000.0, seed=3)
    a = _atr_expected(intra[(SYM, "5m")], et(DAY, "14:30"))
    return Scenario("S026", intra, daily_extra={SYM: dx}, signal="14:35", side=1, stop=low_c - 0.2 - 0.25 * a)


def s026_pair():
    return s026(), s026(low_c=99.2)


def s027(near=False):
    rows = flat(100.0, 6, rng=0.05, v=V5)
    pat = [(100, 100.05, 99.6, 99.7), (99.7, 99.75, 99.3, 99.4), (99.4, 99.45, 99.0, 99.2), (99.2, 99.6, 99.15, 99.5),
           (99.5, 99.8, 99.45, 99.7), (99.7, 99.75, 99.4, 99.45), (99.45, 99.5, 99.2, 99.3), (99.3, 99.35, 99.1, 99.2),
           (99.2, 99.4, 99.15, 99.35), (99.35, 99.6, 99.3, 99.55), (99.55, 99.9, 99.5, 99.85)]
    if near:
        pat[5:10] = [(99.7, 99.75, 99.5, 99.55), (99.55, 99.6, 99.45, 99.5), (99.5, 99.55, 99.4, 99.45),
                     (99.45, 99.5, 99.42, 99.48), (99.48, 99.6, 99.45, 99.55)]
    rows += [p + (V5,) for p in pat] + [(99.85, 99.85, 99.85, 99.85, V5)]
    intra = _one("5m", rows)
    a = _atr_expected(intra[(SYM, "5m")], et(DAY, "10:50"))
    return Scenario("S027", intra, signal="10:55", side=1, stop=99.0 - 0.1 * a, target_r=2.0)


def s027_pair():
    return s027(), s027(near=True)


def s028(drop=0.5):
    rows = flat(100, 6, rng=0.05, v=V5) + path(100, [100.3, 100.6, 100.9, 101.2], 0.05, V5)
    rows += path(101.2, [101.2 - drop], 0.05, V5) + [(101.2 - drop,) * 4 + (V5,)]
    hod = max(r[1] for r in rows[:11])
    return Scenario("S028", _one("5m", rows), signal="10:25", side=-1, stop=hod)


def s028_pair():
    return s028(), s028(drop=0.0)


def s029(c17=100.6):
    closes = [101.1, 101.2, 101.3] + list(np.linspace(101.25, c17, 15))
    rows = path(101.0, closes, rng=0.05, v=V5) + [(c17, c17, c17, c17, V5)]
    hod = max(r[1] for r in rows[:18])
    return Scenario("S029", _one("5m", rows), signal="11:00", side=-1, stop=hod, target=100.0)


def s029_pair():
    return s029(), s029(c17=100.9)


def s030(c71=100.4, halt=False):
    rows = flat(100.0, 70, rng=0.05) + [(100.0, 101.0, 99.98, 100.9, V1)]
    rows += [(100.9, 100.92, 100.35, c71, V1) if c71 < 100.49 else (100.9, 100.95, 100.55, c71, V1)]
    rows += [(c71, c71, c71, c71, V1)]
    evs = []
    if halt:
        evs = [MarketEvent("HALT", SYM, known_at=et(DAY, "10:41"), effective_at=et(DAY, "10:41"),
                           payload={"reason": "LUDP", "resume_at": et(DAY, "10:50")})]

    def chk(r, sc):
        tx = r.fills[r.fills.tag == "time_exit"]
        assert len(tx) == 1 and tx.iloc[0].ts == et(DAY, "11:12")
    return Scenario("S030", _one("1m", rows), events=evs, signal="10:42", side=-1, stop=101.0, target=100.0,
                    check=None if (halt or c71 >= 100.49) else chk)


def s030_pair():
    return s030(), s030(c71=100.6)


def s116(open_=101.0):
    closes = [open_ - 0.1, open_ - 0.15, open_ - 0.2, open_ - 0.25, open_ - 0.3, open_ - 0.35, open_ - 0.4,
              open_ - 0.35, open_ - 0.4, open_ - 0.4]
    rows = path(open_, closes, rng=0.05) + [(open_ - 0.4,) * 4 + (V1,)]
    return Scenario("S116", _one("1m", rows, sym="SPY"), universe=("SPY",), symbol="SPY", signal="09:40", side=-1,
                    stop=open_ + 0.05, target=100.0)


def s116_pair():
    return s116(), s116(open_=100.4)


PAIRS = {
    "S001": s001, "S002": s002, "S003": s003, "S004": s004, "S005": s005, "S006": s006, "S007": s007_pair,
    "S008": s008, "S009": s009, "S010": s010_pair, "S011": s011, "S012": s012_pair, "S013": s013_pair,
    "S014": s014_pair, "S015": s015_pair, "S016": s016_pair, "S017": s017_pair, "S018": s018_pair,
    "S019": s019_pair, "S020": s020_pair, "S021": s021_pair, "S022": s022_pair, "S023": s023_pair,
    "S024": s024_pair, "S025": s025_pair, "S026": s026_pair, "S027": s027_pair, "S028": s028_pair,
    "S029": s029_pair, "S030": s030_pair, "S116": s116_pair,
}
