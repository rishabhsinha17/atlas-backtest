"""Trend indicators: ADX/DMI, Parabolic SAR, Supertrend, Ichimoku, linear-regression
slope, 52-week-high ratio, all-time high.

* ADX/DMI - Wilder (1978) ch. IV. TR, +DM, -DM start at bar 1 (they need a
  prior bar). TR14/+DM14/-DM14 use Wilder's running-sum smoothing
  (first = sum of the first 14 values, then S - S/14 + x); +DI = 100*+DM14/TR14;
  DX = 100*|+DI - -DI|/(+DI + -DI); ADX first = mean of the first 14 DX
  values (index 2n-1), then Wilder smoothing. Matches StockCharts cs-adx.xls
  (QQQQ, 504 bars) to 1e-9.
* Parabolic SAR - Wilder (1978) ch. II. Implementation follows the
  well-known TA-Lib reference algorithm, which encodes Wilder's rules:
  AF starts at `af0`, increases by `step` at each new extreme point up to
  `af_max`; the SAR for the next bar may never be inside (above, in an
  uptrend) the lows of the current and previous bar (and symmetrically for
  downtrends); a reversal occurs when price touches/penetrates the SAR
  (low <= SAR in an uptrend), the new SAR is the prior trend's extreme point
  (clamped outside the current and prior bar ranges), AF resets.
  Initial direction: short if the first bar pair shows a dominant -DM,
  else long (TA-Lib convention).
* Supertrend - Olivier Seban; band logic identical to TradingView
  ``ta.supertrend`` (hl2 +/- mult*ATR, Wilder ATR, ratcheting final bands,
  initial state = downtrend).
* Ichimoku - Goichi Hosoda. See ``ichimoku`` for the information-time
  alignment of the forward-displaced cloud.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._util import (as_float_array, check_period, index_of, njit, rolling_max, rolling_min, shift, wrap,
                    wrap_frame)
from .moving import _ewm_seeded, _wilder_sum
from .volatility import atr as _atr


# ---------------------------------------------------------------- ADX / DMI
def dmi(high, low, close, n: int = 14):
    """Wilder's Directional Movement system. Returns a frame with
    tr, plus_dm, minus_dm, tr_n, plus_dm_n, minus_dm_n, plus_di, minus_di, dx, adx."""
    n = check_period(n)
    h, l, c = as_float_array(high), as_float_array(low), as_float_array(close)
    m = h.shape[0]
    tr = np.full(m, np.nan)
    pdm = np.full(m, np.nan)
    mdm = np.full(m, np.nan)
    if m > 1:
        pc = c[:-1]
        tr[1:] = np.maximum(h[1:], pc) - np.minimum(l[1:], pc)
        up = h[1:] - h[:-1]
        dn = l[:-1] - l[1:]
        pdm[1:] = np.where((up > dn) & (up > 0), up, 0.0)
        mdm[1:] = np.where((dn > up) & (dn > 0), dn, 0.0)
    trn = _wilder_sum(tr, n, 1)
    pdmn = _wilder_sum(pdm, n, 1)
    mdmn = _wilder_sum(mdm, n, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = np.where(trn > 0, 100.0 * pdmn / trn, np.nan)
        mdi = np.where(trn > 0, 100.0 * mdmn / trn, np.nan)
        s = pdi + mdi
        dx = np.where(s > 0, 100.0 * np.abs(pdi - mdi) / s, np.where(np.isnan(s), np.nan, 0.0))
    adx = _ewm_seeded(dx, 1.0 / n, n, True)
    return wrap_frame({"tr": tr, "plus_dm": pdm, "minus_dm": mdm, "tr_n": trn, "plus_dm_n": pdmn,
                       "minus_dm_n": mdmn, "plus_di": pdi, "minus_di": mdi, "dx": dx, "adx": adx},
                      index_of(high, low, close))


def adx(high, low, close, n: int = 14):
    return dmi(high, low, close, n)["adx"]


# ---------------------------------------------------------------- Parabolic SAR
@njit
def _psar(h, l, af0, step, af_max):
    m = h.shape[0]
    sar_out = np.full(m, np.nan)
    trend = np.zeros(m)
    nxt = np.full(m, np.nan)
    af_out = np.full(m, np.nan)
    ep_out = np.full(m, np.nan)
    if m < 2:
        return sar_out, trend, nxt, af_out, ep_out
    diff_p = h[1] - h[0]
    diff_m = l[0] - l[1]
    minus_dm = diff_m if (diff_m > 0 and diff_m > diff_p) else 0.0
    is_long = not (minus_dm > 0)
    af = af0
    if is_long:
        ep = h[1]
        sar = l[0]
    else:
        ep = l[1]
        sar = h[0]
    new_low = l[0]
    new_high = h[0]
    for t in range(1, m):
        prev_low = new_low
        prev_high = new_high
        new_low = l[t]
        new_high = h[t]
        if is_long:
            if new_low <= sar:
                # reversal to short
                is_long = False
                sar = ep
                if sar < prev_high:
                    sar = prev_high
                if sar < new_high:
                    sar = new_high
                sar_out[t] = sar
                trend[t] = -1.0
                af = af0
                ep = new_low
                sar = sar + af * (ep - sar)
                if sar < prev_high:
                    sar = prev_high
                if sar < new_high:
                    sar = new_high
            else:
                sar_out[t] = sar
                trend[t] = 1.0
                if new_high > ep:
                    ep = new_high
                    af += step
                    if af > af_max:
                        af = af_max
                sar = sar + af * (ep - sar)
                if sar > prev_low:
                    sar = prev_low
                if sar > new_low:
                    sar = new_low
        else:
            if new_high >= sar:
                # reversal to long
                is_long = True
                sar = ep
                if sar > prev_low:
                    sar = prev_low
                if sar > new_low:
                    sar = new_low
                sar_out[t] = sar
                trend[t] = 1.0
                af = af0
                ep = new_high
                sar = sar + af * (ep - sar)
                if sar > prev_low:
                    sar = prev_low
                if sar > new_low:
                    sar = new_low
            else:
                sar_out[t] = sar
                trend[t] = -1.0
                if new_low < ep:
                    ep = new_low
                    af += step
                    if af > af_max:
                        af = af_max
                sar = sar + af * (ep - sar)
                if sar < prev_high:
                    sar = prev_high
                if sar < new_high:
                    sar = new_high
        nxt[t] = sar
        af_out[t] = af
        ep_out[t] = ep
    return sar_out, trend, nxt, af_out, ep_out


def parabolic_sar(high, low, af0: float = 0.02, step: float | None = None, af_max: float = 0.2):
    """Wilder Parabolic SAR.

    Returns columns:
      sar      - stop-and-reverse level in force during bar t (known at t-1's
                 close, except on a reversal bar where it is the prior EP);
      trend    - +1 long / -1 short after processing bar t;
      flip     - +1 bullish flip at bar t, -1 bearish flip, 0 otherwise;
      next_sar - the stop level for bar t+1, fully determined at t's close
                 (use this to place a gap-aware stop order for the next bar);
      af, ep   - acceleration factor and extreme point after bar t.
    """
    step = af0 if step is None else step
    h, l = as_float_array(high), as_float_array(low)
    s, tr, nx, af, ep = _psar(h, l, float(af0), float(step), float(af_max))
    prev = np.concatenate([[0.0], tr[:-1]])
    flip = np.where((tr != 0) & (prev != 0) & (tr != prev), tr, 0.0)
    return wrap_frame({"sar": s, "trend": tr, "flip": flip, "next_sar": nx, "af": af, "ep": ep},
                      index_of(high, low))


# ---------------------------------------------------------------- Supertrend
@njit
def _supertrend(hl2, c, a, mult):
    m = c.shape[0]
    st = np.full(m, np.nan)
    d = np.zeros(m)
    fu_o = np.full(m, np.nan)
    fl_o = np.full(m, np.nan)
    started = False
    fu = 0.0
    fl = 0.0
    dirn = -1.0
    for t in range(m):
        if np.isnan(a[t]) or np.isnan(c[t]):
            continue
        bu = hl2[t] + mult * a[t]
        bl = hl2[t] - mult * a[t]
        if not started:
            fu, fl = bu, bl
            dirn = -1.0
            started = True
        else:
            cp = c[t - 1]
            nfl = bl if (bl > fl or cp < fl) else fl
            nfu = bu if (bu < fu or cp > fu) else fu
            if dirn == -1.0:  # previous supertrend was the upper band
                dirn = 1.0 if c[t] > nfu else -1.0
            else:
                dirn = -1.0 if c[t] < nfl else 1.0
            fu, fl = nfu, nfl
        st[t] = fl if dirn == 1.0 else fu
        d[t] = dirn
        fu_o[t] = fu
        fl_o[t] = fl
    return st, d, fu_o, fl_o


def supertrend(high, low, close, atr_n: int = 10, mult: float = 3.0):
    """Supertrend. Columns: supertrend, direction (+1 up / -1 down), upper, lower, flip."""
    h, l, c = as_float_array(high), as_float_array(low), as_float_array(close)
    a = as_float_array(_atr(h, l, c, atr_n))
    st, d, fu, fl = _supertrend((h + l) / 2.0, c, a, float(mult))
    prev = np.concatenate([[0.0], d[:-1]])
    flip = np.where((d != 0) & (prev != 0) & (d != prev), d, 0.0)
    return wrap_frame({"supertrend": st, "direction": d, "upper": fu, "lower": fl, "flip": flip},
                      index_of(high, low, close))


# ---------------------------------------------------------------- Ichimoku
def ichimoku(high, low, close, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52,
             displacement: int = 26):
    """Ichimoku Kinko Hyo aligned to INFORMATION TIME.

    Classic charts plot Senkou spans `displacement` bars into the future and
    the Chikou span `displacement` bars into the past. A naive dataframe that
    stores "what is drawn at date d" therefore contains, for the Chikou span,
    the close of d+displacement (future data). This function never does that.

    Columns (all known at the close of bar t):
      tenkan, kijun        - midpoints of the (high, low) range over 9/26 bars ending at t
      senkou_a_lead        - (tenkan+kijun)/2 computed at t; charts draw it at t+displacement
      senkou_b_lead        - 52-bar midpoint computed at t; drawn at t+displacement
      cloud_a, cloud_b     - the cloud **plotted at t** = lead values computed at
                             t-displacement (``senkou_*_lead.shift(displacement)``)
      cloud_top/bottom     - max/min of cloud_a, cloud_b ("the previously plotted
                             known cloud", S062)
      chikou_ref           - close_{t-displacement}: compare close_t with it instead of
                             plotting close_t back in time (no future leakage)

    Charting software that plots with offset displacement-1 (e.g. TradingView's
    default) corresponds to ``displacement=25`` here.
    """
    h, l, c = as_float_array(high), as_float_array(low), as_float_array(close)
    tk = 0.5 * (rolling_max(h, tenkan) + rolling_min(l, tenkan))
    kj = 0.5 * (rolling_max(h, kijun) + rolling_min(l, kijun))
    sa = 0.5 * (tk + kj)
    sb = 0.5 * (rolling_max(h, senkou_b) + rolling_min(l, senkou_b))
    ca, cb = shift(sa, displacement), shift(sb, displacement)
    return wrap_frame({"tenkan": tk, "kijun": kj, "senkou_a_lead": sa, "senkou_b_lead": sb,
                       "cloud_a": ca, "cloud_b": cb, "cloud_top": np.maximum(ca, cb),
                       "cloud_bottom": np.minimum(ca, cb), "chikou_ref": shift(c, displacement)},
                      index_of(high, low, close))


# ---------------------------------------------------------------- misc trend
def linreg_slope(x, n: int, normalize: bool = False):
    """OLS slope of the last n values on time index 0..n-1 (per bar).
    normalize=True divides by the window mean (fractional slope per bar)."""
    n = check_period(n)
    if n < 2:
        raise ValueError("n must be >= 2")
    a = as_float_array(x)
    out = np.full(a.shape[0], np.nan)
    if a.shape[0] >= n:
        i = np.arange(n, dtype=float)
        w = (i - i.mean()) / np.sum((i - i.mean()) ** 2)
        win = np.lib.stride_tricks.sliding_window_view(a, n)
        sl = win @ w
        if normalize:
            sl = sl / win.mean(axis=1)
        out[n - 1:] = sl
    return wrap(out, index_of(x), f"slope{n}")


def high_ratio(close, high, n: int = 252, min_periods: int | None = None):
    """C_t / max(H over the n bars ending at t) (George & Hwang 2004 52-week-high ratio)."""
    c, h = as_float_array(close), as_float_array(high)
    mx = pd.Series(h).rolling(n, min_periods=min_periods or n).max().to_numpy()
    return wrap(c / mx, index_of(close, high), f"high_ratio{n}")


def prior_all_time_high(high, min_history: int = 252):
    """max(H_0..H_{t-1}) using the split-adjusted (as-of-now) series; NaN until
    at least `min_history` prior bars exist. Breakout: close_t > this value."""
    h = as_float_array(high)
    mx = shift(np.fmax.accumulate(np.where(np.isnan(h), -np.inf, h)), 1)
    mx = np.where(np.isinf(mx), np.nan, mx)
    cnt = shift(np.cumsum(~np.isnan(h)).astype(float), 1)
    mx = np.where(cnt >= min_history, mx, np.nan)
    return wrap(mx, index_of(high), "prior_ath")
