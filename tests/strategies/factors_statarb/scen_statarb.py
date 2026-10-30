"""SYNTHETIC scenarios (mechanics only) for S106-S115, S123.

Each builder returns dict(cls, params, build(cut=None), start, end, signal (session of the entry signal),
legs={sym: +1/-1}, notional (per leg fraction of NAV), extra info). Expected signal days are derived here
with independent reference arithmetic on the synthetic inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from atlasbt.indicators import kalman_hedge
from atlasbt.strategies import statarb as SA
from atlasbt.strategies.factors import SECTOR_SPDR_9

from fsa_fixtures import CAL, build_portal, closes_from_returns, master, sessions

RUN_START = pd.Timestamp("2023-03-01")
K_DAY = pd.Timestamp("2023-03-15")


def _idx(dates, d):
    return list(dates).index(pd.Timestamp(d))


def _filler(dates, syms, seed=0, p0=50.0):
    return {s: closes_from_returns(dates, np.random.default_rng(seed + i).normal(0, 0.006, len(dates)), p0 + i)
            for i, s in enumerate(syms)}


# --------------------------------------------------------------------------- S106 distance pairs
def s106(z_path=(3.0, 3.0, 3.0, 0.2), max_hold=60):
    dates = sessions("2022-01-03", "2023-06-30")
    W = 252
    f_end = _idx(dates, RUN_START)
    f_start = f_end - W + 1
    t = np.arange(len(dates))
    out = {}
    specs = {"A0": ("B0", 11), "C0": ("D0", 22)}
    for a, (b, seed) in specs.items():
        nb = np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, len(dates))))
        nb = nb / nb[f_start]
        d = 0.01 * np.sin(2 * np.pi * t / 17.0)
        d = d - d[f_start]                              # normalised spread = 0 at formation start
        mu, sd = d[f_start:f_end + 1].mean(), d[f_start:f_end + 1].std(ddof=1)
        if a == "A0":
            k = _idx(dates, K_DAY)
            for j, z in enumerate(z_path):
                d[k + j] = mu + z * sd
        na = nb + d
        out[a] = pd.Series(40.0 * na, index=pd.DatetimeIndex(dates))
        out[b] = pd.Series(40.0 * nb, index=pd.DatetimeIndex(dates))
        if a == "A0":
            info = dict(mu=mu, sd=sd)
    m = master(list(out), sectors=["Information Technology", "Information Technology", "Financials", "Financials"],
                sub=["X", "X", "Y", "Y"])
    uni = ["A0", "B0", "C0", "D0"]
    return dict(cls=SA.S106, params={"max_hold": max_hold}, closes=out,
                build=lambda cut=None: build_portal(out, universe=uni, master_df=m, cut=cut),
                start=RUN_START, end=pd.Timestamp("2023-03-31"), signal=K_DAY, legs={"A0": -1, "B0": +1}, **info)


# --------------------------------------------------------------------------- S107 cointegration
def s107(z_target=(3.5, 1.0, -0.3)):
    dates = sessions("2022-01-03", "2023-06-30")
    n = len(dates)
    rng = np.random.default_rng(31)
    lb = np.cumsum(rng.normal(0, 0.01, n)) + np.log(50)
    u = np.zeros(n)
    e = rng.normal(0, 0.01, n)
    for i in range(1, n):
        u[i] = 0.5 * u[i - 1] + e[i]
    la = 0.5 + 1.2 * lb + u
    W = 252
    f_end = _idx(dates, RUN_START)
    f_start = f_end - W + 1
    beta, alpha = np.polyfit(lb[f_start:f_end + 1], la[f_start:f_end + 1], 1)
    s = la - beta * lb
    level = s[f_start:f_end + 1].mean()
    k = _idx(dates, K_DAY)
    for j, z in enumerate(z_target):
        sdp = s[k + j - 60:k + j].std(ddof=1)
        s[k + j] = level + z * sdp
        la[k + j] = s[k + j] + beta * lb[k + j]
    out = {"A1": pd.Series(np.exp(la), index=pd.DatetimeIndex(dates)),
           "B1": pd.Series(np.exp(lb), index=pd.DatetimeIndex(dates))}
    out.update({"C1": closes_from_returns(dates, rng.normal(0.001, 0.015, n), 30.0),
                "D1": closes_from_returns(dates, rng.normal(-0.001, 0.015, n), 30.0)})
    m = master(list(out), sectors=["Information Technology"] * 2 + ["Financials"] * 2, sub=["X", "X", "Y", "Y"])
    uni = list(out)
    return dict(cls=SA.S107, params={}, closes=out,
                build=lambda cut=None: build_portal(out, universe=uni, master_df=m, cut=cut),
                start=RUN_START, end=pd.Timestamp("2023-03-30"), signal=K_DAY, legs={"A1": -1, "B1": +1},
                beta=beta, exit_day=dates[k + 2])


# --------------------------------------------------------------------------- S108 Kalman
S108_SYMS = sorted({s for p in SA.S108_PAIRS for s in p})


def _kalman_z(y, x, k, window=504, q=0.001):
    yy, xx = y.iloc[k - window + 1:k + 1], x.iloc[k - window + 1:k + 1]
    r = kalman_hedge(yy, xx, q_ratio=q, warmup=60).iloc[-1]
    b = min(max(r["beta_prior"], 0.5), 2.0)
    return r, b


def s108(z_target=3.0):
    dates = sessions("2020-06-01", "2023-06-30")
    n = len(dates)
    t = np.arange(n)
    x = 100 * np.exp(np.cumsum(np.random.default_rng(41).normal(0, 0.01, n)))
    y = 10 + 0.9 * x + 0.3 * np.sin(t / 3.0)
    di = pd.DatetimeIndex(dates)
    ys, xs = pd.Series(y, index=di), pd.Series(x, index=di)
    k = _idx(dates, K_DAY)
    r, b = _kalman_z(ys, xs, k)
    e0 = ys.iloc[k] - r["alpha_prior"] - b * xs.iloc[k]
    ys.iloc[k] = ys.iloc[k] + (z_target * np.sqrt(r["innovation_var"]) - e0)
    pre = []
    for j in range(_idx(dates, RUN_START), k):
        rr, bb = _kalman_z(ys, xs, j)
        pre.append((ys.iloc[j] - rr["alpha_prior"] - bb * xs.iloc[j]) / np.sqrt(rr["innovation_var"]))
    out = _filler(dates, [s for s in S108_SYMS if s not in ("XLK", "VGT")], seed=500)
    out["XLK"], out["VGT"] = ys, xs
    return dict(cls=SA.S108, params={"pairs": [("XLK", "VGT")]}, closes=out,
                build=lambda cut=None: build_portal(out, universe=S108_SYMS, cut=cut),
                start=RUN_START, end=pd.Timestamp("2023-03-20"), signal=K_DAY,
                legs={"XLK": -1, "VGT": +1} if z_target > 0 else {"XLK": +1, "VGT": -1}, pre_z=pre)


# --------------------------------------------------------------------------- S109 residual spread
S109_STOCKS = [f"S{i:02d}" for i in range(20)]
FRIDAY = pd.Timestamp("2023-03-17")


def s109(jump_day="2023-03-14"):
    dates = sessions("2021-06-01", "2023-06-30")
    n = len(dates)
    di = pd.DatetimeIndex(dates)
    rng = np.random.default_rng(51)
    spy = rng.normal(0.0002, 0.008, n)
    etf = {"XLK": rng.normal(0, 0.006, n), "XLV": rng.normal(0, 0.006, n)}
    out = {"SPY": closes_from_returns(dates, spy, 380.0)}
    out.update(_filler(dates, [e for e in SECTOR_SPDR_9 if e not in etf], seed=600, p0=60))
    for e, r in etf.items():
        out[e] = closes_from_returns(dates, r, 70.0)
    jumps = {"S00": -0.06, "S01": 0.06, "S10": -0.05, "S11": 0.05}
    jd = di == pd.Timestamp(jump_day)       # factors flat on the jump day: the jump is purely idiosyncratic
    spy[jd] = 0.0
    for v in etf.values():
        v[jd] = 0.0
    out["SPY"] = closes_from_returns(dates, spy, 380.0)
    for e, v in etf.items():
        out[e] = closes_from_returns(dates, v, 70.0)
    for i, s in enumerate(S109_STOCKS):
        e = "XLK" if i < 10 else "XLV"
        r = spy + 0.5 * etf[e] + np.random.default_rng(700 + i).normal(0, 0.002, n)
        r = r + np.where(di == pd.Timestamp(jump_day), jumps.get(s, 0.0), 0.0)
        out[s] = closes_from_returns(dates, r, 40.0 + i)
    m = master(S109_STOCKS, sectors=["Information Technology"] * 10 + ["Health Care"] * 10)
    return dict(cls=SA.S109, params={}, closes=out,
                build=lambda cut=None: build_portal(out, universe=S109_STOCKS, master_df=m, cut=cut),
                start=pd.Timestamp("2023-03-13"), end=pd.Timestamp("2023-03-22"), signal=FRIDAY,
                legs={"S00": +1, "S01": -1, "S10": +1, "S11": -1}, ignore={"SPY"})


# --------------------------------------------------------------------------- S111 same-index ETFs
S111_SYMS = sorted({s for p in SA.S111_PAIRS for s in p})


def s111(dev_sd=10.0):
    dates = sessions("2022-03-01", "2023-06-30")
    n = len(dates)
    t = np.arange(n)
    di = pd.DatetimeIndex(dates)
    spy = closes_from_returns(dates, np.random.default_rng(61).normal(0.0003, 0.01, n), 400.0)
    lr = 0.0005 * np.sin(t / 4.0)
    k = _idx(dates, K_DAY)
    prev = lr[k - 126 + 1:k]
    lr[k] = prev.mean() + dev_sd * prev.std(ddof=0)
    ivv = spy * np.exp(-lr)
    out = {"SPY": spy, "IVV": pd.Series(ivv.to_numpy(), index=di)}
    out.update(_filler(dates, [s for s in S111_SYMS if s not in out], seed=800, p0=100))
    win = lr[k - 125:k + 1]
    zk = (win[-1] - win.mean()) / win.std(ddof=0)
    return dict(cls=SA.S111, params={"pairs": [("SPY", "IVV")]}, closes=out,
                build=lambda cut=None: build_portal(out, universe=S111_SYMS, cut=cut),
                start=RUN_START, end=pd.Timestamp("2023-03-20"), signal=K_DAY, legs={"SPY": -1, "IVV": +1}, z=zk)


# --------------------------------------------------------------------------- S112 PCA residuals
S112_STOCKS = [f"P{i:02d}" for i in range(20)]


def s112():
    dates = sessions("2021-06-01", "2023-06-30")
    n = len(dates)
    di = pd.DatetimeIndex(dates)
    rng = np.random.default_rng(71)
    f = rng.normal(0, 0.008, (n, 3))
    load = rng.uniform(0.3, 1.2, (20, 3))
    jumps = {"P00": -0.06, "P01": 0.06, "P02": -0.04, "P03": 0.04}
    out = {"SPY": closes_from_returns(dates, f[:, 0], 400.0)}
    for i, s in enumerate(S112_STOCKS):
        r = f @ load[i] + np.random.default_rng(900 + i).normal(0, 0.002, n)
        r = r + np.where(di == K_DAY, jumps.get(s, 0.0), 0.0)
        if s == "P05":
            r = r + np.where(di == pd.Timestamp("2023-03-07"), 0.08, 0.0)
        out[s] = closes_from_returns(dates, r, 30.0 + i)
    m = master(S112_STOCKS)
    return dict(cls=SA.S112, params={}, closes=out,
                build=lambda cut=None: build_portal(out, universe=S112_STOCKS, master_df=m, cut=cut),
                start=K_DAY, end=pd.Timestamp("2023-03-24"), signal=K_DAY,
                legs={"P00": +1, "P01": -1, "P02": +1, "P03": -1}, ignore={"SPY"}, near_miss=["P05"])


# --------------------------------------------------------------------------- S113 RS pairs
JUNE_END = pd.Timestamp("2023-06-30")


def s113():
    dates = sessions("2022-01-03", "2023-08-31")
    n = len(dates)
    di = pd.DatetimeIndex(dates)
    rng = np.random.default_rng(81)
    spy = rng.normal(0.0, 0.008, n)
    m3 = pd.Timestamp("2023-03-31")
    win = (di > m3) & (di <= JUNE_END)
    nwin = int(win.sum())
    spec = {"A2": (1.5, 0.30), "B2": (0.7, 0.0), "C2": (1.0, 0.10), "D2": (1.0, 0.0), "E2": (1.0, 0.18),
            "F2": (1.0, 0.0)}
    out = {"SPY": closes_from_returns(dates, spy, 400.0)}
    for j, (s, (b, mom)) in enumerate(spec.items()):
        r = b * spy + np.random.default_rng(1000 + j).normal(0, 0.003, n)
        c = closes_from_returns(dates, r, 50.0)
        # impose the 3-month total return exactly: scale the in-window path
        base = c[win]
        target = c[di <= m3].iloc[-1] * (1 + mom)
        adj = np.ones(n)
        g = (target / base.iloc[-1]) ** (1.0 / nwin)
        adj[win] = g ** np.arange(1, nwin + 1)
        adj[di > JUNE_END] = g ** nwin
        out[s] = c * adj
    syms = list(spec)
    m = master(syms, sectors=["Information Technology"] * 6, sub=["X", "X", "Y", "Y", "Z", "Z"])
    px = pd.DataFrame(out)
    r = (px / px.shift(1) - 1).loc[:JUNE_END].iloc[-126:]
    ba = max(np.polyfit(r["SPY"], r["A2"], 1)[0], 0.1)
    bb = max(np.polyfit(r["SPY"], r["B2"], 1)[0], 0.1)
    w = {"A2": 0.10 * bb / (ba + bb), "B2": -0.10 * ba / (ba + bb)}
    return dict(cls=SA.S113, params={}, closes=out,
                build=lambda cut=None: build_portal(out, universe=syms, master_df=m, cut=cut),
                start=pd.Timestamp("2023-06-26"), end=pd.Timestamp("2023-07-07"), signal=JUNE_END,
                legs={"A2": +1, "B2": -1}, weights=w, near_miss=["C2", "D2", "E2", "F2"])


# --------------------------------------------------------------------------- S114 sector shock
S114_STOCKS = [f"T{i:02d}" for i in range(10)]


def s114(shock_sd=8.0, hold_days=1, htb=None):
    dates = sessions("2021-06-01", "2023-06-30")
    n = len(dates)
    di = pd.DatetimeIndex(dates)
    rng = np.random.default_rng(91)
    xlk = rng.normal(0, 0.005, n)
    k = _idx(dates, K_DAY)
    run_days = (di >= pd.Timestamp("2023-03-01")) & (di != K_DAY)
    xlk[run_days] = 0.0005
    sd = xlk[k - 60:k].std(ddof=1)
    xlk[k] = shock_sd * sd
    out = {"XLK": closes_from_returns(dates, xlk, 150.0)}
    out.update(_filler(dates, [e for e in SECTOR_SPDR_9 if e != "XLK"], seed=1100, p0=60))
    res = {"T00": -0.03, "T01": -0.02, "T08": 0.02, "T09": 0.03}
    for i, s in enumerate(S114_STOCKS):
        r = xlk + np.random.default_rng(1200 + i).normal(0, 0.002, n)
        r = r + np.where(di == K_DAY, res.get(s, 0.0), 0.0)
        out[s] = closes_from_returns(dates, r, 30.0 + i)
    m = master(S114_STOCKS, sectors=["Information Technology"] * 10)
    borrow = {s: (True, 0.003) for s in S114_STOCKS}
    if htb:
        borrow[htb] = (True, 0.05)
    return dict(cls=SA.S114, params={"hold_days": hold_days}, closes=out,
                build=lambda cut=None: build_portal(out, universe=S114_STOCKS, master_df=m, borrow=borrow, cut=cut),
                start=pd.Timestamp("2023-03-13"), end=pd.Timestamp("2023-03-24"), signal=K_DAY,
                legs={"T00": +1, "T09": -1, "T01": +1, "T08": -1}, near_miss=["T02", "T07"])


# --------------------------------------------------------------------------- S115 spread momentum
S115_SYMS = sorted({s for p in SA.S115_PAIRS for s in p})


def _slope(v):
    x = np.arange(len(v))
    return np.polyfit(x, v, 1)[0]


def s115(kind="up"):
    dates = sessions("2022-03-01", "2023-06-30")
    n = len(dates)
    t = np.arange(n)
    di = pd.DatetimeIndex(dates)
    s = 0.02 * np.sin(2 * np.pi * (t % 20) / 20.0)      # exactly periodic: no spurious new extremes
    k = _idx(dates, K_DAY)
    if kind == "up":
        for j in range(k - 4, k + 6):
            s[j] = s[k - 5] + 0.012 * (j - (k - 5))
    else:  # near miss: new 60d high on day k but 20d slope negative
        hi = s[k - 60:k].max()
        s[k - 19:k] = np.linspace(hi - 0.005, hi - 0.035, 19)
        s[k] = hi + 0.001
        s[k + 1:k + 6] = hi - 0.03
    xlu = closes_from_returns(dates, np.random.default_rng(1300).normal(0, 0.008, n), 60.0)
    xlk = xlu * np.exp(s)
    out = {"XLK": xlk, "XLU": xlu}
    out.update(_filler(dates, [x for x in S115_SYMS if x not in out], seed=1400, p0=80))
    first = None
    for j in range(_idx(dates, RUN_START), k + 6):
        if s[j] > s[j - 60:j].max() and _slope(s[j - 19:j + 1]) > 0:
            first = dates[j]
            break
    return dict(cls=SA.S115, params={"pairs": [("XLK", "XLU")]}, closes=out,
                build=lambda cut=None: build_portal(out, universe=S115_SYMS, cut=cut),
                start=RUN_START, end=pd.Timestamp("2023-03-24"), signal=first, legs={"XLK": +1, "XLU": -1},
                slope_k=_slope(s[k - 19:k + 1]), s=pd.Series(s, index=di))


# --------------------------------------------------------------------------- S123 RSP vs SPY
def s123(excess=0.03, july_exit=False):
    dates = sessions("2022-06-01", "2023-08-31")
    n = len(dates)
    spy = closes_from_returns(dates, np.random.default_rng(1500).normal(0.0003, 0.008, n), 400.0)
    k = _idx(dates, JUNE_END)
    kj = _idx(dates, pd.Timestamp("2023-07-31"))
    L = np.log(1 + excess)
    rel = np.zeros(n)
    rel[k - 62:k + 1] = np.linspace(0, L, 64)[1:]
    rel[k + 1:] = L
    if july_exit:     # relative strength fades in July so the 63d excess at July 31 is ~ +0.2%
        target = rel[kj - 63] + np.log(1.002)
        rel[k + 1:kj + 1] = np.linspace(L, target, kj - k + 1)[1:]
        rel[kj + 1:] = target
    rsp = 100.0 * (spy / spy.iloc[0]) * np.exp(rel)
    out = {"RSP": rsp, "SPY": spy}

    def exc(i):
        return rsp.iloc[i] / rsp.iloc[i - 63] - 1 - (spy.iloc[i] / spy.iloc[i - 63] - 1)

    return dict(cls=SA.S123, params={}, closes=out, build=lambda cut=None: build_portal(out, universe=["RSP", "SPY"],
                                                                                         cut=cut),
                start=pd.Timestamp("2023-06-26"), end=pd.Timestamp("2023-07-07"), signal=JUNE_END,
                legs={"RSP": +1, "SPY": -1}, excess=exc(k), excess_july=exc(kj))
