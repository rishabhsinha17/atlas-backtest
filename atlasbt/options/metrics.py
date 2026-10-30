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

from atlasbt.core.types import Fidelity

from atlasbt.pricing.forward import infer_forwards
from atlasbt.pricing.surface import (worse_flag, atm_iv_per_expiry, constant_maturity_delta_iv, constant_maturity_iv, smile_frame)

from .chain import prepare_chain


def _input_fidelity(chain: pd.DataFrame) -> int:
    vals = []
    if "fidelity" in chain.columns:
        vals += [int(v) for v in pd.to_numeric(chain["fidelity"], errors="coerce").dropna().unique()]
    if chain.attrs.get("fidelity") is not None:
        vals.append(int(chain.attrs["fidelity"]))
    return max(vals) if vals else int(Fidelity.T2_PROXY)


_CACHE: "OrderedDict" = None
_CACHE_MAX = 512


def _cache_key(chain: pd.DataFrame, now, r, spot, american, min_dte):
    """Content key: underlying(s), snapshot ts, ET session of ``now``, inputs and a
    checksum of the quotes (a modified chain never hits a stale entry)."""
    from atlasbt.core import clock
    n = pd.Timestamp(now)
    sess = clock.session_date_of(n) if n.tzinfo is not None else n.normalize()
    und = tuple(sorted(map(str, chain["underlying"].unique()))) if "underlying" in chain.columns else ()
    ts = str(pd.to_datetime(chain["ts"]).max()) if "ts" in chain.columns and len(chain) else ""
    num = pd.to_numeric(chain["bid"], errors="coerce").fillna(0).values * 1.0001 + \
        pd.to_numeric(chain["ask"], errors="coerce").fillna(0).values * 0.9999 + \
        pd.to_numeric(chain["strike"], errors="coerce").fillna(0).values * 1e-3
    up = pd.to_numeric(chain["underlying_price"], errors="coerce").fillna(0).sum() if "underlying_price" in chain.columns else 0.0
    return (und, ts, str(sess), round(float(r), 10), None if spot is None else round(float(spot), 8), american,
            int(min_dte), len(chain), round(float(np.sum(num)), 6), round(float(up), 6),
            int(_input_fidelity(chain)))


def chain_metrics(chain: pd.DataFrame, now, r: float, spot: float | None = None, american: bool | None = None,
                  min_dte: int = 1, allow_synthetic: bool = True, use_cache: bool = True) -> dict:
    """Cached wrapper (per snapshot content + session); see ``_chain_metrics``."""
    global _CACHE
    from collections import OrderedDict
    if _CACHE is None:
        _CACHE = OrderedDict()
    fid = _input_fidelity(chain)
    if fid >= int(Fidelity.T3_MODELED) and not allow_synthetic:
        raise ValueError("chain_metrics: synthetic (T3) chain refused (allow_synthetic=False)")
    if not use_cache:
        return _chain_metrics(chain, now, r, spot, american, min_dte)
    key = _cache_key(chain, now, r, spot, american, min_dte)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        out = _CACHE[key]
    else:
        out = _chain_metrics(chain, now, r, spot, american, min_dte)
        _CACHE[key] = out
        if len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in out.items()}


def _chain_metrics(chain: pd.DataFrame, now, r: float, spot: float | None = None, american: bool | None = None,
                   min_dte: int = 1, allow_synthetic: bool = True) -> dict:
    """Metrics dict: atm_iv_30, atm_iv_60, term_spread_30_60 (=30D-60D),
    put25_iv_30, call25_iv_30, skew25_30 (put-call), forwards (DataFrame),
    q_near, flags, n_expiries.  IVs are recomputed from mids on the parity
    forward (Black-76, OTM side) so real and synthetic chains are treated
    identically; quoted ``iv`` is not used.

    Output always carries ``fidelity`` (worst input tier; chains without a
    fidelity column/attr count as T2) and ``synthetic`` (True if any row is
    T3_MODELED).  Synthetic smiles are deterministic in the vol indices, so their
    skew/term metrics are NOT valid signals; ``allow_synthetic=False`` raises."""
    fid = _input_fidelity(chain)
    synth = fid >= int(Fidelity.T3_MODELED)
    if synth and not allow_synthetic:
        raise ValueError("chain_metrics: synthetic (T3) chain refused (allow_synthetic=False)")
    x = prepare_chain(chain, now, r=r, spot=spot)
    x = x[(x["dte"] >= min_dte) & (x["bid"] > 0) & (x["ask"] > x["bid"])]
    out = {"n_expiries": int(x["expiry"].nunique()), "fidelity": fid, "synthetic": synth}
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
               put25_iv_30=p25, call25_iv_30=c25, skew25_30=p25 - c25, flag_skew=worse_flag(fp, fc),
               forwards=fw, atm_by_expiry=atm, spot=S,
               q_near=float(fw.sort_values("T")["q_implied"].iloc[0]))
    return out
