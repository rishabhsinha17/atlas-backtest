"""Hand-built SYNTHETIC scenarios for S126-S157 (mechanics only).

Each builder returns a :class:`Scen`: the price path, the option-chain
fixture settings, the strategy params, and the EXPECTED signal session,
opening legs and (where deterministic) exit session/tag, computed here
independently from the sheet rule (indicators recomputed on the fixture's
bars; strikes picked by brute force on the fixture's snapshot), plus a
near-miss variant that must not open any option position.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Optional

import numpy as np
import pandas as pd

from atlasbt.indicators import donchian, ema, percentile_rank, sma
from atlasbt.risk import sizing as sz

import oi_fixtures as fx


# ------------------------------------------------------------------ independent selection helpers
def usable(df: pd.DataFrame) -> pd.DataFrame:
    mid = 0.5 * (df["bid"] + df["ask"])
    return df[(df["bid"] > 0) & (df["ask"] > df["bid"]) & ((df["ask"] - df["bid"]) / mid <= 0.5)]


def snapshot(portal, ss, i) -> pd.DataFrame:
    return portal.option_chain(fx.UND, fx.close_ts(ss[i]))


def pick_expiry(snap, session, target, lo, hi) -> pd.Timestamp:
    u = usable(snap)
    ex = sorted({pd.Timestamp(e) for e in u["expiry"]})
    cand = [(abs((e - session).days - target), (e - session).days, e) for e in ex if lo <= (e - session).days <= hi]
    assert cand, f"fixture has no expiry in [{lo},{hi}]"
    return sorted(cand)[0][2]


def by_delta(snap, expiry, right, target) -> float:
    u = usable(snap)
    u = u[(u["expiry"] == expiry) & (u["right"] == right)]
    S = float(u["underlying_price"].iloc[0])
    o = u.assign(e=(u["delta"] - target).abs().round(10), a=(u["strike"] - S).abs()).sort_values(["e", "a", "strike"])
    return float(o["strike"].iloc[0])


def atm(snap, expiry, both=True) -> float:
    u = usable(snap)
    u = u[u["expiry"] == expiry]
    S = float(u["underlying_price"].iloc[0])
    ks = set(u[u["right"] == "C"]["strike"]) & set(u[u["right"] == "P"]["strike"]) if both else set(u["strike"])
    ks = np.array(sorted(ks))
    d = np.round(np.abs(ks - S), 10)
    return float(ks[np.lexsort((ks, d))][0])


def quote(snap, expiry, right, strike) -> pd.Series:
    r = snap[(snap["expiry"] == expiry) & (snap["right"] == right) & np.isclose(snap["strike"], strike)]
    assert len(r) == 1
    return r.iloc[0]


def natural_open(snap, expiry, legs) -> float:
    """legs [(right, strike, ratio)] -> net debit per unit at natural prices."""
    tot = 0.0
    for r, k, q in legs:
        row = quote(snap, expiry if not isinstance(expiry, dict) else expiry[(r, k, q)], r, k)
        tot += q * (row["ask"] if q > 0 else row["bid"])
    return tot


def first_dte_le(ss, start_i, expiry, x) -> int:
    for j in range(start_i, len(ss)):
        if (expiry - ss[j]).days <= x:
            return j
    raise AssertionError("no session reaches the DTE")


# ------------------------------------------------------------------ paths
def bull_path(n=300, dip=0.004, rec=0.006):
    i = np.arange(n)
    c = 80 * np.exp(0.0015 * i)
    c[215:240] = c[215] * np.exp(-dip * np.arange(25))
    c[240:] = c[239] * np.exp(rec * np.arange(n - 240))
    return np.round(c, 2)


def bear_path(n=300, dip=0.004):
    return np.round(80 * 80 * 1.5 / bull_path(n, dip), 2)


def trend_path(n, g, p0=100.0):
    return np.round(p0 * np.exp(g * np.arange(n)), 2)


def noisy_path(n, daily_vol, seed=1, p0=100.0, drift=0.0):
    rng = np.random.default_rng(seed)
    return np.round(p0 * np.exp(np.cumsum(rng.normal(drift, daily_vol, n))), 2)


# ------------------------------------------------------------------ scenario
@dataclass
class Scen:
    sid: str
    closes: np.ndarray
    n: int
    portal_kw: dict
    params: dict = field(default_factory=dict)
    start_i: int = 0
    signal_i: int = -1
    legs: list = field(default_factory=list)                 # [(right, strike, sign of opening qty)]
    expiry: Optional[pd.Timestamp] = None
    units: Optional[int] = None
    exit_i: Optional[int] = None
    exit_tag: Optional[str] = None
    near_closes: Optional[np.ndarray] = None
    near_portal_kw: Optional[dict] = None
    near_params: Optional[dict] = None
    stock: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def ss(self):
        return fx.sessions(self.n)

    def portal(self, cut=None):
        return _portal(self.sid, cut, False)

    def near_portal(self):
        return _portal(self.sid, None, True)


_SCEN: dict[str, Scen] = {}
_BUILDERS: dict[str, Callable[[], Scen]] = {}


def builder(*sids):
    def deco(f):
        for s in sids:
            _BUILDERS[s] = (lambda f=f, s=s: f(s))
        return f
    return deco


def get(sid) -> Scen:
    if sid not in _SCEN:
        _SCEN[sid] = _BUILDERS[sid]()
    return _SCEN[sid]


@lru_cache(maxsize=None)
def _portal(sid, cut, near):
    sc = get(sid) if sid in _SCEN else _BUILDERS[sid]()
    ss = fx.sessions(sc.n)
    if near:
        kw = dict(sc.portal_kw)
        kw.update(sc.near_portal_kw or {})
        closes = sc.near_closes if sc.near_closes is not None else sc.closes
        return fx.make_portal(closes, ss, **kw)
    return fx.make_portal(sc.closes, ss, cut=cut, **sc.portal_kw)


def base_portal(sc: Scen):
    return fx.make_portal(sc.closes, fx.sessions(sc.n), **sc.portal_kw)


# ================================================================== builders
@builder("S126", "S127")
def _cross(sid):
    bull = sid == "S126"
    closes = bull_path() if bull else bear_path()
    n = len(closes)
    sc = Scen(sid, closes, n, dict(chain_from=240), start_i=0)
    p = base_portal(sc)
    ss = sc.ss
    s = pd.Series(closes)
    e20, e50, s200 = ema(s, 20), ema(s, 50), sma(s, 200)
    if bull:
        k = next(j for j in range(241, n) if e20[j - 1] <= e50[j - 1] and e20[j] > e50[j] and closes[j] > s200[j])
    else:
        k = next(j for j in range(241, n) if e20[j - 1] >= e50[j - 1] and e20[j] < e50[j] and closes[j] < s200[j])
    snap = snapshot(p, ss, k)
    E = pick_expiry(snap, ss[k], 30, 25, 35)
    right = "C" if bull else "P"
    K = by_delta(snap, E, right, 0.5 if bull else -0.5)
    ask = float(quote(snap, E, right, K)["ask"])
    sc.signal_i, sc.expiry, sc.legs = k, E, [(right, K, 1)]
    sc.units = sz.size_opt_debit(1e6, ask * 100)
    # exit: close vs EMA20 or DTE <= 7 (whichever first, from the session after the fill)
    for j in range(k + 2, n):
        if (bull and closes[j] < e20[j]) or ((not bull) and closes[j] > e20[j]):
            sc.exit_i, sc.exit_tag = j, "exit_close_" + ("below" if bull else "above") + "_ema20"
            break
        if (E - ss[j]).days <= 7:
            sc.exit_i, sc.exit_tag = j, "exit_dte_le_7"
            break
    sc.near_closes = bull_path(dip=0.016) if bull else bear_path(dip=0.016)
    return sc


@builder("S134", "S135")
def _breakout(sid):
    bull = sid == "S134"
    closes = bull_path() if bull else bear_path()
    n = len(closes)
    sc = Scen(sid, closes, n, dict(chain_from=243), start_i=0)
    p = base_portal(sc)
    ss = sc.ss
    b = p.bars(fx.UND, "1d", fx.open_ts(ss[0]), fx.close_ts(ss[-1]), fx.close_ts(ss[-1]))
    dc = donchian(b["high"].to_numpy(), b["low"].to_numpy(), 20)
    up, lo = dc["upper"], dc["lower"]
    if bull:
        k = next(j for j in range(243, n) if closes[j] > up[j] and closes[j - 1] <= up[j - 1])
    else:
        k = next(j for j in range(243, n) if closes[j] < lo[j] and closes[j - 1] >= lo[j - 1])
    snap = snapshot(p, ss, k)
    E = pick_expiry(snap, ss[k], 30, 25, 45 if bull else 35)
    r = "C" if bull else "P"
    s = 1 if bull else -1
    sc.signal_i, sc.expiry = k, E
    sc.legs = [(r, by_delta(snap, E, r, s * 0.6), 1), (r, by_delta(snap, E, r, s * 0.3), -1)]
    sc.extra["check"] = "debit_vertical"
    # near miss: price stalls just below the breakout level (flat after the dip)
    nc = closes.copy()
    j0 = 241
    nc[j0:] = nc[j0 - 1]
    sc.near_closes = nc
    return sc


@builder("S136")
def _s136(sid):
    closes = trend_path(280, 0.001)
    n = len(closes)
    k = 262
    vix = np.full(n, 20.0)
    vix[k:] = 26.0
    vix_near = np.full(n, 20.0)
    vix_near[k:] = 16.0
    sc = Scen(sid, closes, n, dict(chain_from=255, extra_bars={"VIX": vix}), start_i=0)
    p = base_portal(sc)
    ss = sc.ss
    assert percentile_rank(vix, 20, include_current=True)[k] > 0.6
    assert percentile_rank(vix, 20, include_current=True)[k - 1] <= 0.6
    snap = snapshot(p, ss, k)
    E = pick_expiry(snap, ss[k], 30, 30, 45)
    legs = [("P", by_delta(snap, E, "P", -0.10), 1), ("P", by_delta(snap, E, "P", -0.30), -1)]
    sc.signal_i, sc.expiry, sc.legs = k, E, legs
    nat = natural_open(snap, E, legs)
    width = legs[1][1] - legs[0][1]
    sc.units = sz.size_opt_credit(1e6, (width + nat) * 100)[0]
    sc.extra["check"] = "credit_tp_stop"
    sc.near_portal_kw = dict(extra_bars={"VIX": vix_near})
    return sc


@builder("S137", "S154")
def _down(sid):
    closes = trend_path(295, -0.002)
    n = len(closes)
    sc = Scen(sid, closes, n, dict(chain_from=255), start_i=0)
    p = base_portal(sc)
    ss = sc.ss
    k = 255
    s = pd.Series(closes)
    assert closes[k] < sma(s, 200)[k] and closes[k] / closes[k - 20] - 1 < 0
    snap = snapshot(p, ss, k)
    if sid == "S137":
        E = pick_expiry(snap, ss[k], 30, 30, 45)
        sc.legs = [("C", by_delta(snap, E, "C", 0.10), 1), ("C", by_delta(snap, E, "C", 0.30), -1)]
        sc.extra["check"] = "credit_tp_stop"
    else:
        E = pick_expiry(snap, ss[k], 30, 30, 45)
        sc.legs = [("C", by_delta(snap, E, "C", 0.25), -1), ("P", by_delta(snap, E, "P", -0.25), 1)]
        sc.exit_i, sc.exit_tag = first_dte_le(ss, k + 2, E, 7), "exit_dte_le_7"
    sc.signal_i, sc.expiry = k, E
    sc.near_closes = trend_path(295, 0.002)
    return sc


@builder("S153")
def _s153(sid):
    closes = trend_path(295, 0.002)
    n = len(closes)
    sc = Scen(sid, closes, n, dict(chain_from=255), start_i=0)
    p = base_portal(sc)
    ss = sc.ss
    k = 255
    snap = snapshot(p, ss, k)
    E = pick_expiry(snap, ss[k], 30, 30, 45)
    sc.signal_i, sc.expiry = k, E
    sc.legs = [("C", by_delta(snap, E, "C", 0.25), 1), ("P", by_delta(snap, E, "P", -0.25), -1)]
    sc.exit_i, sc.exit_tag = first_dte_le(ss, k + 2, E, 7), "exit_dte_le_7"
    sc.near_closes = trend_path(295, -0.002)
    return sc


@builder("S138")
def _s138(sid):
    closes = noisy_path(60, 0.004, seed=2)
    sc = Scen(sid, closes, 60, dict(chain_from=5, band=0.3), params={"underlying": "SPY"}, start_i=0)
    p = base_portal(sc)
    ss = sc.ss
    k = 5
    snap = snapshot(p, ss, k)
    E = pick_expiry(snap, ss[k], 30, 23, 44)
    sc.signal_i, sc.expiry = k, E
    sc.legs = [("P", by_delta(snap, E, "P", -0.05), 1), ("P", by_delta(snap, E, "P", -0.20), -1),
               ("C", by_delta(snap, E, "C", 0.20), -1), ("C", by_delta(snap, E, "C", 0.05), 1)]
    sc.extra["check"] = "credit_tp_dte5"
    sc.near_portal_kw = dict(max_dte=20)
    return sc


@builder("S139", "S140")
def _ifly(sid):
    closes = noisy_path(60, 0.004, seed=4)
    sc = Scen(sid, closes, 60, dict(chain_from=5), start_i=0)
    p = base_portal(sc)
    ss = sc.ss
    k = 5
    snap = snapshot(p, ss, k)
    E = pick_expiry(snap, ss[k], 30, 25, 45) if sid == "S139" else pick_expiry(snap, ss[k], 30, 25, 35)
    K = atm(snap, E)
    s = 1 if sid == "S139" else -1          # S139 short body
    sc.signal_i, sc.expiry = k, E
    sc.legs = [("P", K - 10, s), ("P", K, -s), ("C", K, -s), ("C", K + 10, s)]
    sc.extra["check"] = "credit_tp_stop" if sid == "S139" else "debit_tp30"
    sc.near_portal_kw = dict(step=3.0)      # +-10 wings not listed on a $3 grid -> no trade
    return sc


@builder("S141", "S142")
def _fly(sid):
    closes = noisy_path(60, 0.004, seed=5)
    sc = Scen(sid, closes, 60, dict(chain_from=5), start_i=0)
    p = base_portal(sc)
    ss = sc.ss
    k = 5
    snap = snapshot(p, ss, k)
    E = pick_expiry(snap, ss[k], 30, 25, 45)
    r = "C" if sid == "S141" else "P"
    K = atm(snap, E, both=False)
    sc.signal_i, sc.expiry = k, E
    sc.legs = [(r, K - 10, 1), (r, K, -2), (r, K + 10, 1)]
    sc.extra["check"] = "fly_value"
    sc.near_portal_kw = dict(step=3.0)
    return sc


@builder("S155")
def _s155(sid):
    closes = noisy_path(60, 0.004, seed=6)
    sc = Scen(sid, closes, 60, dict(chain_from=5, vol=0.30, band=0.3), start_i=0)
    p = base_portal(sc)
    ss = sc.ss
    k = 5
    snap = snapshot(p, ss, k)
    E = pick_expiry(snap, ss[k], 30, 25, 35)
    K = atm(snap, E, both=False)
    sc.signal_i, sc.expiry = k, E
    sc.legs = [("C", K - 10, 1), ("C", K, -2), ("C", K + 15, 1)]
    sc.near_params = {"width": 5.0}         # K3 = K2 + 7.5 is never listed on a $1 grid -> no trade
    return sc


@builder("S151", "S152")
def _backspread(sid):
    closes = noisy_path(60, 0.004, seed=7)
    sc = Scen(sid, closes, 60, dict(chain_from=5, max_dte=80), start_i=0)
    p = base_portal(sc)
    ss = sc.ss
    k = 5
    snap = snapshot(p, ss, k)
    E = pick_expiry(snap, ss[k], 45, 30, 60)
    r, s = ("C", 1) if sid == "S151" else ("P", -1)
    sc.signal_i, sc.expiry = k, E
    sc.legs = [(r, by_delta(snap, E, r, s * 0.5), -1), (r, by_delta(snap, E, r, s * 0.25), 2)]
    sc.exit_i, sc.exit_tag = first_dte_le(ss, k + 2, E, 7), "exit_dte_le_7"
    sc.near_portal_kw = dict(max_dte=25)
    return sc


@builder("S156")
def _s156(sid):
    closes = noisy_path(60, 0.003, seed=8)
    sc = Scen(sid, closes, 60, dict(chain_from=5, skew=3.0, vol=0.40, band=0.3), start_i=0)
    p = base_portal(sc)
    ss = sc.ss
    k = 5
    snap = snapshot(p, ss, k)
    E = pick_expiry(snap, ss[k], 30, 25, 35)
    legs = [("P", by_delta(snap, E, "P", -0.30), -1), ("C", by_delta(snap, E, "C", 0.30), -1),
            ("C", by_delta(snap, E, "C", 0.15), 1)]
    credit = -natural_open(snap, E, legs)
    assert credit >= legs[2][1] - legs[1][1], "fixture must allow zero upside risk"
    sc.signal_i, sc.expiry, sc.legs = k, E, legs
    sc.extra["check"] = "credit_tp50"
    sc.near_portal_kw = dict(skew=0.0)       # flat smile: credit < call-vertical width -> refused
    return sc


@builder("S143")
def _s143(sid):
    n = 300
    closes = noisy_path(n, 0.008, seed=9)
    k = 285
    vix = np.full(n, 30.0)
    vix[:200] = np.linspace(25, 35, 200)
    vix[k:] = 15.0
    sc = Scen(sid, closes, n, dict(chain_from=280, extra_bars={"VIX": vix}), params={"iv_source": "index"})
    p = base_portal(sc)
    ss = sc.ss
    pr = percentile_rank(vix, 252, include_current=False)
    assert pr[k] < 0.2 and pr[k - 1] >= 0.2
    snap = snapshot(p, ss, k)
    E = pick_expiry(snap, ss[k], 30, 25, 35)
    K = atm(snap, E)
    sc.signal_i, sc.expiry, sc.legs = k, E, [("C", K, 1), ("P", K, 1)]
    sc.exit_i, sc.exit_tag = k + 1 + 10, "exit_time_10d"
    vn = vix.copy()
    vn[k:] = 30.0
    sc.near_portal_kw = dict(extra_bars={"VIX": vn})
    return sc


@builder("S144", "S145", "S146", "S147")
def _volsig(sid):
    n = 360
    k = 322
    rv_daily = 0.016 if sid in ("S144", "S147") else 0.005
    iv = {"S144": 0.10, "S147": 0.10, "S145": 0.30, "S146": 0.30}[sid]
    iv_near = {"S144": 0.40, "S147": 0.40, "S145": 0.08, "S146": 0.085}[sid]
    closes = noisy_path(n, rv_daily, seed=11)
    sc = Scen(sid, closes, n, dict(chain_from=k, vol=iv), start_i=0)
    p = base_portal(sc)
    ss = sc.ss
    snap = snapshot(p, ss, k)
    if sid in ("S144", "S145"):
        E = pick_expiry(snap, ss[k], 30, 30, 45)
        d = 0.25
        sgn = 1 if sid == "S144" else -1
        sc.legs = [("C", by_delta(snap, E, "C", d), sgn), ("P", by_delta(snap, E, "P", -d), sgn)]
    else:
        E = pick_expiry(snap, ss[k], 30, 25, 35)
        K = atm(snap, E)
        sgn = -1 if sid == "S146" else 1
        sc.legs = [("C", K, sgn), ("P", K, sgn)]
        sc.stock = True
    sc.signal_i, sc.expiry = k, E
    if sid in ("S144", "S147"):
        sc.exit_i, sc.exit_tag = k + 1 + 10, "exit_time_10d"
    elif sid == "S146":
        sc.exit_i, sc.exit_tag = first_dte_le(ss, k + 2, E, 7), "exit_dte_le_7"
    else:
        sc.extra["check"] = "credit_tp_lossstop"
    sc.near_portal_kw = dict(vol=iv_near)
    return sc


@builder("S148", "S149")
def _calendar(sid):
    n = 60
    closes = np.round(100.0 + np.r_[np.zeros(6), np.full(n - 6, 4.0)], 2)    # spot leaves the strike
    sc = Scen(sid, closes, n, dict(chain_from=5, max_dte=100), start_i=0)
    p = base_portal(sc)
    ss = sc.ss
    k = 5
    snap = snapshot(p, ss, k)
    En = pick_expiry(snap, ss[k], 30, 25, 35)
    Ef = pick_expiry(snap, ss[k], 60, 55, 75)
    r = "C" if sid == "S148" else "P"
    u = usable(snap)
    common = sorted(set(u[(u.expiry == En) & (u.right == r)].strike) & set(u[(u.expiry == Ef) & (u.right == r)].strike))
    S = float(snap["underlying_price"].iloc[0])
    K = min(common, key=lambda x: (round(abs(x - S), 10), x))
    sc.signal_i, sc.expiry = k, En
    sc.legs = [(r, K, -1), (r, K, 1)]
    sc.extra.update(far=Ef, check="calendar")
    if sid == "S148":
        sc.extra["limit"] = round(natural_open(snap, {(r, K, -1): En, (r, K, 1): Ef}, [(r, K, -1), (r, K, 1)]), 2)
        sc.near_portal_kw = dict(max_dte=50)            # no far expiry in 55-75
    else:
        sc.near_portal_kw = dict(term_slope=0.6)        # far IV >> near IV + 5 pts -> refused
    return sc


@builder("S150")
def _s150(sid):
    n = 290
    closes = trend_path(n, 0.0015)
    ss = fx.sessions(n)
    exps = sorted(set(fx.weekly_expiries(ss[0], ss[-1])) |
                  set(fx.monthly_expiries(ss[0], ss[-1] + pd.Timedelta(days=420))))
    sc = Scen(sid, closes, n, dict(chain_from=250, max_dte=400, expiries=exps, band=0.3), start_i=0)
    p = base_portal(sc)
    k = 250
    snap = snapshot(p, ss, k)
    Ef = pick_expiry(snap, ss[k], 180, 180, 365)
    En = pick_expiry(snap, ss[k], 30, 25, 45)
    sc.signal_i, sc.expiry = k, En
    sc.legs = [("C", by_delta(snap, En, "C", 0.30), -1), ("C", by_delta(snap, Ef, "C", 0.80), 1)]
    sc.extra.update(far=Ef)
    sc.near_closes = trend_path(n, -0.0015)
    return sc


# ------------------------------------------------------------------ stock-holding strategies
@builder("S128", "S130", "S133", "S157", "S131")
def _stock(sid):
    n = 60
    if sid == "S131":
        closes = np.round(np.r_[np.full(3, 100.0), np.linspace(100, 85, n - 3)], 2)   # put assigned
    else:
        closes = noisy_path(n, 0.004, seed=13)
    sc = Scen(sid, closes, n, dict(chain_from=0), start_i=0, stock=sid in ("S128", "S133", "S157"))
    p = base_portal(sc)
    ss = sc.ss
    k = 1 if sid in ("S128", "S133", "S157") else 0       # shares bought first (fill next open), then options
    snap = snapshot(p, ss, k)
    S = float(snap["underlying_price"].iloc[0])
    if sid == "S128":
        E = pick_expiry(snap, ss[k], 30, 25, 35)
        u = usable(snap)
        u = u[(u.expiry == E) & (u.right == "C") & (u.strike > S)]
        K = float(u.assign(e=(u.delta - 0.30).abs().round(10), a=(u.strike - S).abs()).sort_values(["e", "a", "strike"]).strike.iloc[0])
        sc.legs = [("C", K, -1)]
        sc.exit_i, sc.exit_tag = first_dte_le(ss, k + 2, E, 5), "exit_buyback_dte"
        sc.units = sz.size_opt_covered(1e6, closes[0])[1]
    elif sid == "S130":
        E = pick_expiry(snap, ss[k], 30, 25, 45)
        K = by_delta(snap, E, "P", -0.30)
        sc.legs = [("P", K, -1)]
        sc.units = sz.size_opt_cashsec(1e6, K, nav=1e6)
        sc.extra["check"] = "credit_tp50"
    elif sid == "S131":
        E = pick_expiry(snap, ss[k], 30, 25, 35)
        K = by_delta(snap, E, "P", -0.30)
        sc.legs = [("P", K, -1)]
        sc.extra["check"] = "wheel"
    elif sid == "S133":
        E = pick_expiry(snap, ss[k], 30, 25, 35)
        sc.legs = [("P", by_delta(snap, E, "P", -0.25), 1), ("C", by_delta(snap, E, "C", 0.25), -1)]
        sc.exit_i, sc.exit_tag = first_dte_le(ss, k + 2, E, 5), "exit_roll_dte"
    else:  # S157
        E = pick_expiry(snap, ss[k], 30, 25, 35)
        sc.legs = [("C", by_delta(snap, E, "C", 0.20), -1), ("P", by_delta(snap, E, "P", -0.20), -1)]
        sc.exit_i, sc.exit_tag = first_dte_le(ss, k + 2, E, 7), "exit_roll_dte_le_7"
    sc.signal_i, sc.expiry = k, E
    sc.near_portal_kw = dict(max_dte=20)
    return sc


@builder("S132")
def _s132(sid):
    n = 70
    closes = noisy_path(n, 0.004, seed=14)
    sc = Scen(sid, closes, n, dict(chain_from=0, max_dte=100), start_i=0, stock=True)
    p = base_portal(sc)
    ss = sc.ss
    k = next(i for i, d in enumerate(ss) if d.month == 2)       # first session of February
    snap = snapshot(p, ss, k)
    E = pick_expiry(snap, ss[k], 60, 60, 90)
    sc.signal_i, sc.expiry = k, E
    sc.legs = [("P", by_delta(snap, E, "P", -0.25), 1)]
    sc.exit_i, sc.exit_tag = first_dte_le(ss, k + 2, E, 30), "exit_roll_dte_le_30"
    sc.extra["near_end"] = k - 1                                  # stop before the first month signal
    return sc


@builder("S129")
def _s129(sid):
    n = 50
    closes = noisy_path(n, 0.004, seed=15)
    ss = fx.sessions(n)
    exps = fx.monthly_expiries(ss[0], ss[-1])
    sc = Scen(sid, closes, n, dict(chain_from=0, expiries=exps), params={"underlying": "SPY"}, start_i=0)
    p = base_portal(sc)
    roll = pd.Timestamp("2021-01-15")                              # 3rd Friday Jan 2021
    k = ss.index(roll)
    snap = snapshot(p, ss, k)
    E = pick_expiry(snap, ss[k], 30, 7, 45)
    sc.signal_i, sc.expiry = k, E
    sc.legs = [("P", atm(snap, E, both=False), -1)]
    sc.units = sz.size_opt_cashsec(1e6, sc.legs[0][1], nav=1e6)
    sc.extra["second_entry"] = ss.index(CAL_next(pd.Timestamp(E)))  # PM/physical: settled at E's close
    sc.extra["near_start"] = k + 6                                     # after the roll window, before next roll
    return sc


def CAL_next(d):
    return fx.CAL.next_session(d)
