"""Bar-pattern detectors with explicit confirmation lags.

Every detector returns values *at the bar where the information becomes
known*. Pivots, for example, are reported at the confirmation bar
t = s + right (s = the pivot bar), never back-dated to s.

Definitions
-----------
* NR-n (Crabel 1990): range_t = H_t - L_t is <= every range of the previous
  n-1 bars (n bars including today). Ties count as narrowest (documented).
* Inside bar: H_t < H_{t-1} and L_t > L_{t-1} (strict). ``nested=k`` requires
  k consecutive inside bars.
* IBS = (C - L)/(H - L); NaN when H == L (S075 "skip H=L").
* Bullish engulfing (S066): C_{t-1} < O_{t-1}, C_t > O_t, O_t < C_{t-1}, C_t > O_{t-1}.
* Pivot high at bar s: H_s > max(H_{s-left..s-1}) and H_s >= max(H_{s+1..s+right});
  confirmed (known) at t = s + right. Pivot lows mirror with L.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._util import (as_float_array, check_period, index_of, rolling_max, rolling_min, shift, wrap,
                    wrap_frame)
from .volatility import atr as _atr


# ---------------------------------------------------------------- simple bar patterns
def nr_n(high, low, n: int = 7):
    """True if today's range is the narrowest of the last n bars (incl. today)."""
    n = check_period(n)
    r = as_float_array(high) - as_float_array(low)
    mn = rolling_min(r, n)
    out = (r <= mn) & ~np.isnan(mn)
    return wrap(out, index_of(high, low), f"nr{n}")


def inside_bar(high, low, nested: int = 1):
    h, l = as_float_array(high), as_float_array(low)
    ins = (h < shift(h, 1)) & (l > shift(l, 1))
    if nested > 1:
        run = np.zeros(h.shape[0])
        c = 0
        for i, v in enumerate(ins):
            c = c + 1 if v else 0
            run[i] = c
        ins = run >= nested
    return wrap(ins, index_of(high, low), "inside")


def ibs(high, low, close):
    h, l, c = as_float_array(high), as_float_array(low), as_float_array(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(h > l, (c - l) / (h - l), np.nan)
    return wrap(out, index_of(high, low, close), "ibs")


def bullish_engulfing(open_, high, low, close):
    o, c = as_float_array(open_), as_float_array(close)
    po, pc = shift(o, 1), shift(c, 1)
    out = (pc < po) & (c > o) & (o < pc) & (c > po)
    return wrap(out, index_of(open_, close), "bull_engulf")


def down_streak(close):
    """Number of consecutive strictly lower closes ending at t (0 if C_t >= C_{t-1})."""
    c = as_float_array(close)
    out = np.zeros(c.shape[0])
    k = 0
    for i in range(1, c.shape[0]):
        k = k + 1 if c[i] < c[i - 1] else 0
        out[i] = k
    return wrap(out, index_of(close), "down_streak")


# ---------------------------------------------------------------- pivots
def pivots(high, low, left: int = 3, right: int = 3):
    """Confirmed pivot highs/lows reported at their confirmation bar.

    Columns: ph (bool, a pivot high was confirmed at t), ph_price, ph_bar
    (positional index of the pivot bar s = t - right), pl, pl_price, pl_bar,
    last_ph / last_pl (most recent confirmed pivot price known at t).
    """
    h, l = as_float_array(high), as_float_array(low)
    m = h.shape[0]
    lmax = shift(rolling_max(h, left), 1) if left > 0 else np.full(m, -np.inf)
    lmin = shift(rolling_min(l, left), 1) if left > 0 else np.full(m, np.inf)
    rmax = rolling_max(h, right) if right > 0 else np.full(m, -np.inf)
    rmin = rolling_min(l, right) if right > 0 else np.full(m, np.inf)
    ph = np.zeros(m, bool)
    pl = np.zeros(m, bool)
    if m > right:
        s = np.arange(m - right)
        t = s + right
        ph[t] = (h[s] > lmax[s]) & (h[s] >= rmax[t])
        pl[t] = (l[s] < lmin[s]) & (l[s] <= rmin[t])
    pos = np.arange(m)
    ph_price = np.where(ph, shift(h, right), np.nan)
    pl_price = np.where(pl, shift(l, right), np.nan)
    ph_bar = np.where(ph, pos - right, np.nan)
    pl_bar = np.where(pl, pos - right, np.nan)
    last_ph = pd.Series(ph_price).ffill().to_numpy()
    last_pl = pd.Series(pl_price).ffill().to_numpy()
    return wrap_frame({"ph": ph, "ph_price": ph_price, "ph_bar": ph_bar, "pl": pl, "pl_price": pl_price,
                       "pl_bar": pl_bar, "last_ph": last_ph, "last_pl": last_pl}, index_of(high, low))


def _groups(group, m):
    if group is None:
        return np.zeros(m, dtype=np.int64)
    g = pd.Series(np.asarray(group)).factorize()[0] if not isinstance(group, pd.Series) else group.factorize()[0]
    return np.asarray(g)


def double_bottom(high, low, close, lag: int = 2, tol: float = 0.0025, min_sep: int = 3, max_sep: int = 12,
                  expiry: int | None = None, group=None):
    """Double-bottom detector honouring pivot confirmation lag.

    Two confirmed pivot lows (left = right = ``lag``) separated by
    min_sep..max_sep bars with |L1-L2|/min(L1,L2) <= tol and no lower low in
    between; neckline = max high strictly between them. The pattern becomes
    known when the second low is confirmed (bar s2 + lag). The signal fires at
    the first bar t >= confirmation with C_t > neckline and C_{t-1} <= neckline
    (a fresh break; if price is already above the neckline at confirmation, no
    signal - we never "chase" a break that happened before the pattern was
    knowable). Pattern cancels on a low below min(L1, L2) or after ``expiry``
    bars. ``group`` (e.g. session date) keeps patterns inside one group.
    Columns: signal, neckline, low1, low2, bar1, bar2.
    """
    h, l, c = as_float_array(high), as_float_array(low), as_float_array(close)
    m = h.shape[0]
    g = _groups(group, m)
    pv = pivots(h, l, lag, lag)
    pl, pl_bar = pv["pl"], pv["pl_bar"]
    out = {k: np.full(m, np.nan) for k in ("neckline", "low1", "low2", "bar1", "bar2")}
    sig = np.zeros(m, bool)
    lows: list[tuple[int, float]] = []
    active = None  # (neck, l1, l2, b1, b2, t_conf)
    for t in range(m):
        if t > 0 and g[t] != g[t - 1]:
            lows, active = [], None
        if pl[t]:
            s2 = int(pl_bar[t])
            if s2 >= 0 and g[s2] == g[t]:
                L2 = l[s2]
                for s1, L1 in reversed(lows):
                    sep = s2 - s1
                    if sep < min_sep:
                        continue
                    if sep > max_sep:
                        break
                    lo = min(L1, L2)
                    if abs(L1 - L2) / lo > tol:
                        continue
                    between_l = l[s1 + 1:s2]
                    if between_l.size == 0 or np.nanmin(between_l) < lo:
                        continue
                    neck = float(np.nanmax(h[s1 + 1:s2]))
                    if neck <= max(L1, L2):
                        continue
                    active = (neck, L1, L2, s1, s2, t)
                    break
                lows.append((s2, L2))
        if active is not None:
            neck, L1, L2, b1, b2, tc = active
            if expiry is not None and t - tc > expiry:
                active = None
            elif l[t] < min(L1, L2) and t > b2:
                active = None
            elif t >= tc and t > 0 and c[t] > neck and c[t - 1] <= neck:
                sig[t] = True
                for k, v in zip(("neckline", "low1", "low2", "bar1", "bar2"), (neck, L1, L2, b1, b2)):
                    out[k][t] = v
                active = None
    out["signal"] = sig
    return wrap_frame(out, index_of(high, low, close))


def head_and_shoulders(high, low, close, lag: int = 3, atr_n: int = 14, head_atr: float = 1.0,
                       sym_atr: float = 1.0, expiry: int | None = None):
    """Head-and-shoulders top with fixed pivot confirmation lag (S067).

    Three consecutive confirmed pivot highs LS, H, RS (left = right = lag);
    H - max(LS, RS) >= head_atr*ATR and |LS - RS| <= sym_atr*ATR, ATR taken at
    the RS confirmation bar. Neckline = straight line through the two reaction
    lows (lowest low between LS..H and H..RS), extrapolated to t. Signal (short)
    at the first t >= RS confirmation with C_t < neck_t and C_{t-1} >= neck_{t-1}.
    Cancels if C_t > RS high or after ``expiry`` bars (default 2*(RS-LS)).
    Columns: signal, neckline, rs_high, head_high, ls_high.
    """
    h, l, c = as_float_array(high), as_float_array(low), as_float_array(close)
    m = h.shape[0]
    a = as_float_array(_atr(h, l, c, atr_n))
    pv = pivots(h, l, lag, lag)
    ph, ph_bar = pv["ph"], pv["ph_bar"]
    sig = np.zeros(m, bool)
    neck_o = np.full(m, np.nan)
    rs_o = np.full(m, np.nan)
    hd_o = np.full(m, np.nan)
    ls_o = np.full(m, np.nan)
    highs: list[int] = []
    active = None
    for t in range(m):
        if ph[t]:
            highs.append(int(ph_bar[t]))
            if len(highs) >= 3 and not np.isnan(a[t]):
                s1, s2, s3 = highs[-3:]
                H1, H2, H3 = h[s1], h[s2], h[s3]
                if H2 - max(H1, H3) >= head_atr * a[t] and abs(H1 - H3) <= sym_atr * a[t]:
                    b1 = s1 + 1 + int(np.nanargmin(l[s1 + 1:s2])) if s2 > s1 + 1 else None
                    b2 = s2 + 1 + int(np.nanargmin(l[s2 + 1:s3])) if s3 > s2 + 1 else None
                    if b1 is not None and b2 is not None and b2 > b1:
                        slope = (l[b2] - l[b1]) / (b2 - b1)
                        exp = expiry if expiry is not None else 2 * (s3 - s1)
                        active = (l[b1], b1, slope, H3, H2, H1, t, exp)
        if active is not None:
            y1, b1, slope, H3, H2, H1, tc, exp = active
            nk = y1 + slope * (t - b1)
            nk_prev = y1 + slope * (t - 1 - b1)
            if t - tc > exp or c[t] > H3:
                active = None
            elif t >= tc and c[t] < nk and c[t - 1] >= nk_prev:
                sig[t] = True
                neck_o[t], rs_o[t], hd_o[t], ls_o[t] = nk, H3, H2, H1
                active = None
    return wrap_frame({"signal": sig, "neckline": neck_o, "rs_high": rs_o, "head_high": hd_o, "ls_high": ls_o},
                      index_of(high, low, close))


# ---------------------------------------------------------------- flags / boxes / contraction
def flag_pattern(open_, high, low, close, volume, session, impulse_pct: float = 0.02,
                 impulse_bars: int = 45, retrace=(0.2, 0.5), flag_bars=(5, 15), volume_compare: str = "sum"):
    """Intraday bull-flag continuation (S010), one signal per session.

    Impulse: from the session's first open O, the running high reaches
    O*(1+impulse_pct) within the first `impulse_bars` bars; the impulse peak P
    is the highest high so far (a new high resets the flag). Flag = bars after
    the peak bar. At bar t (flag = bars peak+1 .. t-1) a signal fires if
      * flag length in [flag_bars[0], flag_bars[1]],
      * retracement (P - min flag low)/(P - O) in [retrace[0], retrace[1]],
      * flag volume < impulse volume (``volume_compare='sum'`` literal spec,
        'mean' compares per-bar averages),
      * C_t > max high of the flag bars.
    Columns: signal, flag_high, flag_low, peak, impulse_open.
    """
    o, h, l, c, v = (as_float_array(x) for x in (open_, high, low, close, volume))
    m = c.shape[0]
    g = _groups(session, m)
    sig = np.zeros(m, bool)
    fh_o, fl_o, pk_o, io_o = (np.full(m, np.nan) for _ in range(4))
    t0 = 0
    for t in range(m):
        if t == 0 or g[t] != g[t - 1]:
            t0, O, peak, peak_t, done, ok_imp = t, o[t], -np.inf, -1, False, False
        if done:
            continue
        k = t - t0
        # evaluate the flag BEFORE updating the peak with bar t
        if ok_imp and peak_t >= 0 and t > peak_t:
            fb = slice(peak_t + 1, t)
            nfb = t - peak_t - 1
            if flag_bars[0] <= nfb <= flag_bars[1]:
                fhi = np.max(h[fb])
                flo = np.min(l[fb])
                rt = (peak - flo) / (peak - O) if peak > O else np.nan
                vi = v[t0:peak_t + 1]
                vf = v[fb]
                if volume_compare == "sum":
                    vol_ok = vf.sum() < vi.sum()
                else:
                    vol_ok = vf.mean() < vi.mean()
                if retrace[0] <= rt <= retrace[1] and vol_ok and c[t] > fhi:
                    sig[t] = True
                    fh_o[t], fl_o[t], pk_o[t], io_o[t] = fhi, flo, peak, O
                    done = True
                    continue
        if not ok_imp:
            if k >= impulse_bars:
                done = True  # no qualifying impulse inside the window
                continue
            if h[t] > peak:
                peak, peak_t = h[t], t
                if peak >= O * (1 + impulse_pct):
                    ok_imp = True
        else:
            if h[t] > peak:
                if k < impulse_bars:
                    peak, peak_t = h[t], t  # impulse still extending; flag restarts
                else:
                    done = True  # took out the peak without a valid flag
                    continue
            if t > peak_t and (t - peak_t) > flag_bars[1]:
                done = True  # flag too long
    return wrap_frame({"signal": sig, "flag_high": fh_o, "flag_low": fl_o, "peak": pk_o, "impulse_open": io_o},
                      index_of(open_, high, low, close, volume))


def darvas_box(high, low, close, lookback: int = 20, recent: int = 5, depth_atr: float = 2.0, atr_n: int = 20):
    """Darvas-style box (S060), causal form.

    resistance_t = max(H_{t-lookback..t-1}); box valid if the last `recent`
    bars stayed below it (max(H_{t-recent..t-1}) < resistance) and the floor
    min(L_{t-recent..t-1}) > resistance - depth_atr*ATR_{t-1}. Signal: valid
    box and C_t > resistance.
    """
    h, l, c = as_float_array(high), as_float_array(low), as_float_array(close)
    res = shift(rolling_max(h, lookback), 1)
    rec_hi = shift(rolling_max(h, recent), 1)
    floor = shift(rolling_min(l, recent), 1)
    a = shift(as_float_array(_atr(h, l, c, atr_n)), 1)
    valid = (rec_hi < res) & (floor > res - depth_atr * a)
    valid = np.where(np.isnan(res) | np.isnan(a), False, valid)
    sig = valid & (c > res)
    return wrap_frame({"resistance": res, "floor": floor, "valid": valid, "signal": sig}, index_of(high, low, close))


def volatility_contraction(high, low, close, n: int = 10, n_means: int = 3):
    """True when n_means successive non-overlapping n-bar mean true ranges
    decline: ATRn_t < ATRn_{t-n} < ... < ATRn_{t-(n_means-1)n} (S061)."""
    a = as_float_array(_atr(high, low, close, n, method="sma"))
    ok = np.ones(a.shape[0], bool)
    for j in range(1, n_means):
        ok &= shift(a, (j - 1) * n) < shift(a, j * n)
    return wrap(ok, index_of(high, low, close), "vcp")
