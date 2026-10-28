"""Per-snapshot volatility metrics for signals (S143, S144, S145, S146, S149,
S160, S165-S169): constant-maturity ATM IV (30D/60D, total-variance
interpolation), 25-delta put/call IVs and skew, term spread, parity forwards.

``chain_metrics`` works on any canonical chain (real or synthetic).  The
result's ``*_flag`` values are 'interp' (bracketed), 'exact', 'extrap_flat'
(target tenor outside the listed expiries -> treat as lower fidelity) or
'none'.  Time series of these metrics (for percentiles / z-scores) must be
built causally by the caller from snapshots with ts <= now.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from atlasbt.pricing.forward import infer_forwards
from atlasbt.pricing.surface import (atm_iv_per_expiry, constant_maturity_delta_iv, constant_maturity_iv, smile_frame)

from .chain import prepare_chain


def chain_metrics(chain: pd.DataFrame, now, r: float, spot: float | None = None, american: bool | None = None,
                  min_dte: int = 1) -> dict:
    """Metrics dict: atm_iv_30, atm_iv_60, term_spread_30_60 (=30D-60D),
    put25_iv_30, call25_iv_30, skew25_30 (put-call), forwards (DataFrame),
    q_near, flags, n_expiries.  IVs are recomputed from mids on the parity
    forward (Black-76, OTM side) so real and synthetic chains are treated
    identically; quoted ``iv`` is not used."""
    x = prepare_chain(chain, now, r=r, spot=spot)
    x = x[(x["dte"] >= min_dte) & (x["bid"] > 0) & (x["ask"] > x["bid"])]
    out = {"n_expiries": int(x["expiry"].nunique())}
    if x.empty:
        return out | {"atm_iv_30": np.nan, "atm_iv_60": np.nan}
    S = float(np.nanmedian(x["spot"]))
    fw = infer_forwards(x, r=r, spot=S, american=american)
    fw = fw[np.isfinite(fw["F"])]
    if fw.empty:
        return out | {"atm_iv_30": np.nan, "atm_iv_60": np.nan}
    sm = smile_frame(x, fw)
    atm = atm_iv_per_expiry(sm)
    a30, f30 = constant_maturity_iv(atm, 30)
    a60, f60 = constant_maturity_iv(atm, 60)
    p25, fp = constant_maturity_delta_iv(sm, -0.25, 30)
    c25, fc = constant_maturity_delta_iv(sm, 0.25, 30)
    out.update(atm_iv_30=a30, atm_iv_60=a60, flag_30=f30, flag_60=f60, term_spread_30_60=a30 - a60,
               put25_iv_30=p25, call25_iv_30=c25, skew25_30=p25 - c25, flag_skew=fp if fp != "interp" else fc,
               forwards=fw, atm_by_expiry=atm, spot=S,
               q_near=float(fw.sort_values("T")["q_implied"].iloc[0]))
    return out
