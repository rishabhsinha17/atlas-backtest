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


def series_keys(df: pd.DataFrame) -> list:
    """Expiry-series identity: expiry plus root/settlement columns when present."""
    return ["expiry"] + [c for c in ("root", "settlement") if c in df.columns]


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
    keys = series_keys(df)
    recs = []   # per series: dict(sk, T, rr, F, disp, n, cand arrays) -- de-Americanized in ONE batch below
    for kv, g in df.groupby(keys, sort=True):
        kv = kv if isinstance(kv, tuple) else (kv,)
        sk = dict(zip(keys, kv))
        T = float(g["T"].iloc[0])
        rr = float(r(T)) if callable(r) else float(r)
        rt = g["right"].astype(str).str.upper().str[0]
        c = g[rt == "C"].set_index("strike")
        p = g[rt == "P"].set_index("strike")
        ks = c.index.intersection(p.index)
        if len(ks) == 0 or T <= 0:
            recs.append(dict(sk=sk, T=T, rr=rr, F=np.nan, disp=np.nan, n=0))
            continue
        c = c.loc[ks]
        p = p.loc[ks]
        c = c[~c.index.duplicated()]
        p = p[~p.index.duplicated()]
        ks = c.index
        cb, ca, pb, pa = (np.asarray(x, float) for x in (c["bid"].values, c["ask"].values, p["bid"].values, p["ask"].values))
        F, disp, n = parity_forward(ks.values, cb, ca, pb, pa, T, rr, n_strikes)
        rec = dict(sk=sk, T=T, rr=rr, F=F, disp=disp, n=n)
        if american and spot and np.isfinite(F) and n > 0:
            cm_all, pm_all = 0.5 * (cb + ca), 0.5 * (pb + pa)
            two = (cb > 0) & (pb > 0)
            # only the strikes nearest the forward are used -> de-Americanize just those
            cand = np.where(two)[0][np.argsort(np.abs(cm_all - pm_all)[two])[:2 * n_strikes]]
            rec.update(kc=ks.values[cand].astype(float), cm=cm_all[cand], pm=pm_all[cand],
                       hc=0.5 * (ca - cb)[cand], hp=0.5 * (pa - pb)[cand])
        recs.append(rec)
    if american and spot:
        from .iv import deamericanize
        act = [i for i, rc in enumerate(recs) if "kc" in rc and len(rc["kc"])]
        for _ in range(iterations):
            if not act:
                break
            sizes = [len(recs[i]["kc"]) for i in act]
            K = np.concatenate([recs[i]["kc"] for i in act])
            Tv = np.concatenate([np.full(m, recs[i]["T"]) for i, m in zip(act, sizes)])
            Rv = np.concatenate([np.full(m, recs[i]["rr"]) for i, m in zip(act, sizes)])
            Qv = np.concatenate([np.full(m, recs[i]["rr"] - np.log(recs[i]["F"] / spot) / recs[i]["T"])
                                 for i, m in zip(act, sizes)])
            ce, _ = deamericanize(np.concatenate([recs[i]["cm"] for i in act]), spot, K, Tv, Rv, Qv, True)
            pe, _ = deamericanize(np.concatenate([recs[i]["pm"] for i in act]), spot, K, Tv, Rv, Qv, False)
            off = np.cumsum([0] + sizes)
            nxt = []
            for j, i in enumerate(act):
                rc = recs[i]
                sl = slice(off[j], off[j + 1])
                c_e, p_e = ce[sl], pe[sl]
                good = np.isfinite(c_e) & np.isfinite(p_e)
                if good.sum() == 0:
                    continue
                F2, d2, n2 = parity_forward(rc["kc"][good], (c_e - rc["hc"])[good], (c_e + rc["hc"])[good],
                                            (p_e - rc["hp"])[good], (p_e + rc["hp"])[good], rc["T"], rc["rr"], n_strikes)
                if np.isfinite(F2):
                    rc.update(F=F2, disp=d2, n=n2)
                    nxt.append(i)
            act = nxt
    for rc in recs:
        F, T, rr = rc["F"], rc["T"], rc["rr"]
        q_impl = rr - np.log(F / spot) / T if (spot and np.isfinite(F) and F > 0 and T > 0) else np.nan
        rows.append(dict(rc["sk"], T=T, r=rr, F=F, dispersion=rc["disp"], n=rc["n"], q_implied=q_impl))
    return pd.DataFrame(rows)
