"""Forward / dividend-yield inference from put-call parity on real chains.

For each expiry: for every strike with a two-sided, uncrossed call AND put
quote, F_K = K + e^{rT} (C_mid - P_mid).  The forward is the inverse-spread
weighted median of F_K over the ``n_strikes`` strikes where |C - P| is smallest
(nearest the forward, where parity is least distorted by wide quotes and by the
American early-exercise premium).  For American chains ``american=True`` iterates:
de-Americanize mids (Bjerksund-Stensland 2002 at the current carry estimate),
recompute F, repeat.  The implied continuous carry is q = r - ln(F/S)/T; it
lumps dividends, borrow and any rate mismatch together (documented limitation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _wmedian(x, w):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


def parity_forward(strikes, call_bid, call_ask, put_bid, put_ask, T: float, r: float,
                   n_strikes: int = 5) -> tuple[float, float, int]:
    """Forward from one expiry's quotes.  Returns (F, dispersion, n_used)
    where dispersion is the weighted MAD of the per-strike forwards."""
    K = np.asarray(strikes, float)
    cb, ca, pb, pa = (np.asarray(a, float) for a in (call_bid, call_ask, put_bid, put_ask))
    ok = (cb > 0) & (pb > 0) & (ca > cb) & (pa > pb) & np.isfinite(K)
    if ok.sum() == 0:
        return float("nan"), float("nan"), 0
    K, cm, pm = K[ok], 0.5 * (cb[ok] + ca[ok]), 0.5 * (pb[ok] + pa[ok])
    spr = (ca[ok] - cb[ok]) + (pa[ok] - pb[ok])
    order = np.argsort(np.abs(cm - pm))[:n_strikes]
    Fk = K[order] + np.exp(r * T) * (cm[order] - pm[order])
    w = 1.0 / np.maximum(spr[order], 1e-4)
    F = _wmedian(Fk, w)
    disp = _wmedian(np.abs(Fk - F), w) if len(Fk) > 1 else float("nan")
    return F, disp, int(len(Fk))


def infer_forwards(chain: pd.DataFrame, r, spot: float | None = None, now: pd.Timestamp | None = None,
                   n_strikes: int = 5, american: bool | None = None, iterations: int = 2) -> pd.DataFrame:
    """Per-expiry forwards for a canonical option_chain frame.

    ``chain`` needs columns expiry, strike, right, bid, ask and either a ``T``
    column (years) or ``now`` to compute it (ACT/365F to settlement).  ``r`` is a
    continuous rate (scalar or callable T->r).  ``spot`` defaults to the chain's
    underlying_price.  Returns columns: expiry, T, r, F, dispersion, n, q_implied.
    """
    from .time import year_fractions
    df = chain.copy()
    if "T" not in df.columns:
        if now is None:
            raise ValueError("need T column or now")
        df["T"] = year_fractions(now, df["expiry"], df.get("settlement"))
    if spot is None and "underlying_price" in df.columns:
        spot = float(pd.to_numeric(df["underlying_price"], errors="coerce").dropna().iloc[0]) \
            if df["underlying_price"].notna().any() else None
    if american is None:
        american = bool((df.get("exercise", pd.Series(["AMERICAN"])) .astype(str).str.upper() == "AMERICAN").any())
    rows = []
    for exp, g in df.groupby("expiry", sort=True):
        T = float(g["T"].iloc[0])
        rr = float(r(T)) if callable(r) else float(r)
        c = g[g["right"].astype(str).str.upper().str[0] == "C"].set_index("strike")
        p = g[g["right"].astype(str).str.upper().str[0] == "P"].set_index("strike")
        ks = c.index.intersection(p.index)
        if len(ks) == 0 or T <= 0:
            rows.append(dict(expiry=exp, T=T, r=rr, F=np.nan, dispersion=np.nan, n=0, q_implied=np.nan))
            continue
        c, p = c.loc[ks], p.loc[ks]
        c = c[~c.index.duplicated()]
        p = p[~p.index.duplicated()]
        ks = c.index
        F, disp, n = parity_forward(ks.values, c["bid"].values, c["ask"].values, p["bid"].values, p["ask"].values, T, rr, n_strikes)
        if american and spot and np.isfinite(F) and n > 0:
            from .iv import deamericanize
            cm_all = 0.5 * (c["bid"].values + c["ask"].values)
            pm_all = 0.5 * (p["bid"].values + p["ask"].values)
            two = (c["bid"].values > 0) & (p["bid"].values > 0)
            # only the strikes nearest the forward are used -> de-Americanize just those
            cand = np.where(two)[0][np.argsort(np.abs(cm_all - pm_all)[two])[:2 * n_strikes]]
            kc = ks.values[cand]
            for _ in range(iterations):
                q = rr - np.log(F / spot) / T
                ce, _r1 = deamericanize(cm_all[cand], spot, kc, T, rr, q, True)
                pe, _r2 = deamericanize(pm_all[cand], spot, kc, T, rr, q, False)
                good = np.isfinite(ce) & np.isfinite(pe)
                if good.sum() == 0:
                    break
                hs_c = 0.5 * (c["ask"].values - c["bid"].values)[cand]
                hs_p = 0.5 * (p["ask"].values - p["bid"].values)[cand]
                F2, disp2, n2 = parity_forward(kc[good], (ce - hs_c)[good], (ce + hs_c)[good],
                                               (pe - hs_p)[good], (pe + hs_p)[good], T, rr, n_strikes)
                if not np.isfinite(F2):
                    break
                F, disp, n = F2, disp2, n2
        q_impl = rr - np.log(F / spot) / T if (spot and np.isfinite(F) and F > 0) else np.nan
        rows.append(dict(expiry=exp, T=T, r=rr, F=F, dispersion=disp, n=n, q_implied=q_impl))
    return pd.DataFrame(rows)
