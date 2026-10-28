"""Mark / price a specific contract from a chain snapshot.

Real DoltHub snapshots list only ~3-4 expiries (nearest ~13 DTE) and a sparse
strike set, so a held contract often disappears from the snapshot (e.g. the
"exit at DTE <= 7" rules).  ``mark_contract`` returns, in order of preference:

  1. the contract's own quote (bid/ask/mid)                 -> fidelity of the chain row
  2. price from the SAME snapshot's surface, interpolated:
     IV linear in k = ln(K/F) within the expiry (or between
     bracketing expiries in total variance at the same k)   -> T2_PROXY
  3. extrapolated in maturity (T outside listed expiries,
     flat vol at the nearest expiry's same-k IV) or in strike
     (flat beyond the last listed strike)                   -> T3_MODELED
Bid/ask for cases 2-3 come from the synthetic spread model (flagged in reason).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from atlasbt.core.types import Fidelity
from atlasbt.pricing.american import bs2002_price
from atlasbt.pricing.bsm import bsm_greeks, bsm_price
from atlasbt.pricing.forward import infer_forwards
from atlasbt.pricing.surface import smile_frame
from atlasbt.pricing.time import year_fraction

from .chain import prepare_chain


@dataclass(frozen=True)
class Mark:
    bid: float
    ask: float
    mid: float
    iv: float
    delta: float
    fidelity: Fidelity
    reason: str


def _iv_at_k(sm_e: pd.DataFrame, k: float) -> tuple[float, bool]:
    g = sm_e[sm_e["otm"] & np.isfinite(sm_e["iv_fit"])].sort_values("k")
    if len(g) == 0:
        return float("nan"), False
    ks, vs = g["k"].values, g["iv_fit"].values
    inside = ks[0] <= k <= ks[-1]
    return float(np.interp(k, ks, vs)), inside


def mark_contract(chain: pd.DataFrame, now, r: float, expiry, strike: float, right: str, q: float | None = None,
                  exercise: str = "AMERICAN", settlement: str = "PHYSICAL", underlying: str | None = None,
                  spread_params=None) -> Mark:
    right = right.upper()[0]
    expiry = pd.Timestamp(expiry).normalize()
    x = prepare_chain(chain, now, r=r)
    row = x[(x["expiry"] == expiry) & np.isclose(x["strike"], strike) & (x["right"] == right)]
    if len(row) and row["bid"].iloc[0] >= 0 and row["ask"].iloc[0] > 0:
        rw = row.iloc[0]
        fid = Fidelity(int(rw["fidelity"])) if "fidelity" in rw and pd.notna(rw["fidelity"]) else Fidelity.T2_PROXY
        return Mark(float(rw["bid"]), float(rw["ask"]), float(rw["mid"]), float(rw["iv_used"]), float(rw["delta_used"]),
                    fid, "quoted")
    S = float(np.nanmedian(x["spot"]))
    good = x[(x["bid"] > 0) & (x["ask"] > x["bid"])]
    fw = infer_forwards(good, r=r, spot=S, american=exercise.upper() == "AMERICAN")
    fw = fw[np.isfinite(fw["F"])].sort_values("T")
    if fw.empty:
        return Mark(np.nan, np.nan, np.nan, np.nan, np.nan, Fidelity.NA, "no usable quotes in snapshot")
    T = year_fraction(now, expiry, settlement) if pd.Timestamp(now).tzinfo is not None else \
        max((expiry - pd.Timestamp(now).normalize()).days, 0) / 365.0
    qq = q if q is not None else float(fw["q_implied"].iloc[0]) if np.isfinite(fw["q_implied"].iloc[0]) else 0.0
    F = S * np.exp((r - qq) * T)
    k = np.log(strike / F) if F > 0 else np.nan
    sm = smile_frame(good, fw)
    exps = fw["expiry"].tolist()
    Ts = fw["T"].values
    reason = []
    if expiry in exps:
        iv, inside = _iv_at_k(sm[sm["expiry"] == expiry], k)
        fid = Fidelity.T2_PROXY if inside else Fidelity.T3_MODELED
        reason.append("same-expiry surface" + ("" if inside else ", strike extrapolated flat"))
    elif Ts[0] <= T <= Ts[-1]:
        j = int(np.searchsorted(Ts, T))
        e1, e2 = exps[j - 1], exps[j]
        v1, in1 = _iv_at_k(sm[sm["expiry"] == e1], k)
        v2, in2 = _iv_at_k(sm[sm["expiry"] == e2], k)
        w = np.interp(T, [Ts[j - 1], Ts[j]], [v1 ** 2 * Ts[j - 1], v2 ** 2 * Ts[j]])
        iv = float(np.sqrt(max(w, 0.0) / T))
        fid = Fidelity.T2_PROXY if (in1 and in2) else Fidelity.T3_MODELED
        reason.append(f"total-variance interpolation between {e1.date()} and {e2.date()}")
    else:
        e = exps[0] if T < Ts[0] else exps[-1]
        iv, _ = _iv_at_k(sm[sm["expiry"] == e], k)
        fid = Fidelity.T3_MODELED
        reason.append(f"maturity extrapolated flat from {pd.Timestamp(e).date()}")
    if not np.isfinite(iv):
        return Mark(np.nan, np.nan, np.nan, np.nan, np.nan, Fidelity.NA, "; ".join(reason + ["no smile"]))
    cp = right == "C"
    if exercise.upper() == "AMERICAN":
        mid = float(bs2002_price(S, strike, T, r, qq, iv, cp))
    else:
        mid = float(bsm_price(S, strike, T, r, qq, iv, cp))
    delta = float(bsm_greeks(S, strike, T, r, qq, iv, cp).delta)
    from .synthetic import SyntheticParams, model_spread
    p = spread_params or SyntheticParams.load()
    und = (underlying or (str(x["underlying"].iloc[0]) if "underlying" in x.columns else "")).upper()
    spr = float(model_spread(mid, max(T * 365, 1), abs(delta), und, p))
    reason.append("bid/ask from synthetic spread model")
    return Mark(max(mid - spr / 2, 0.0), mid + spr / 2, mid, iv, delta, fid, "; ".join(reason))
