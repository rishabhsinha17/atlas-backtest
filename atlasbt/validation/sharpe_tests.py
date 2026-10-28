"""Probabilistic & Deflated Sharpe ratios and atlas-wide trial accounting.

* PSR - Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient
  Frontier", J. of Risk 15(2):
      PSR(SR*) = Phi( (SR - SR*) sqrt(T-1) / sqrt(1 - g3 SR + (g4-1)/4 SR^2) )
  with SR the NON-annualised per-period Sharpe, g3 skewness and g4 the
  (non-excess) kurtosis of the returns.
* DSR - Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality",
  J. of Portfolio Management 40(5): PSR evaluated at
      SR0 = sqrt(V[SR_n]) * ((1-gamma) Z^{-1}(1-1/N) + gamma Z^{-1}(1-1/(N e)))
  (expected maximum Sharpe among N independent trials with cross-trial
  Sharpe variance V[SR_n]; gamma = Euler-Mascheroni).
  Paper's numerical example (SR_ann=2.5, T=1250, skew=-3, kurt=10, N=100,
  V_ann=0.5, 250 obs/yr) gives SR0=0.1132 (per period) and DSR=0.9004 - tested.
* Minimum track record length - same 2012 paper.

Trials across the WHOLE atlas: ``atlas_trials`` counts every parameter
combination of every strategy (from the strategy registry's ``grid`` when
available, else parsed from the sheet's "Parameters to test" column). The
leaderboard deflates every strategy with this total N (conservative) and,
optionally, an effective N from the average correlation of trial returns.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def moments(returns) -> dict:
    r = pd.Series(np.asarray(returns, float)).dropna().to_numpy()
    sd = r.std(ddof=1)
    return {"n": r.size, "mean": r.mean(), "sd": sd, "sr": (r.mean() / sd) if sd > 0 else np.nan,
            "skew": float(stats.skew(r, bias=False)) if r.size > 2 else 0.0,
            "kurt": float(stats.kurtosis(r, fisher=False, bias=False)) if r.size > 3 else 3.0}


def psr(sr: float, sr_benchmark: float, n: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Probabilistic Sharpe Ratio (per-period SRs; kurt is NON-excess)."""
    den = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if n < 2 or den <= 0 or not np.isfinite(sr):
        return float("nan")
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(den)
    return float(stats.norm.cdf(z))


def psr_from_returns(returns, sr_benchmark: float = 0.0) -> float:
    m = moments(returns)
    return psr(m["sr"], sr_benchmark, m["n"], m["skew"], m["kurt"])


def expected_max_sharpe(n_trials: float, var_sr: float, mean_sr: float = 0.0) -> float:
    """E[max SR] over N independent trials (per-period units of var_sr)."""
    if n_trials <= 1:
        return float(mean_sr)
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(mean_sr + math.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def dsr(sr: float, n: int, skew: float, kurt: float, n_trials: float, var_sr: float) -> float:
    """Deflated Sharpe Ratio = PSR(SR0 = expected max SR of n_trials)."""
    return psr(sr, expected_max_sharpe(n_trials, var_sr), n, skew, kurt)


def dsr_from_returns(returns, n_trials: float, var_sr: float) -> float:
    m = moments(returns)
    return dsr(m["sr"], m["n"], m["skew"], m["kurt"], n_trials, var_sr)


def min_track_record_length(sr: float, sr_benchmark: float, skew: float = 0.0, kurt: float = 3.0,
                            prob: float = 0.95) -> float:
    """MinTRL (observations) needed for PSR(SR*) >= prob."""
    if sr <= sr_benchmark:
        return float("inf")
    z = stats.norm.ppf(prob)
    return float(1 + (1 - skew * sr + (kurt - 1) / 4 * sr * sr) * (z / (sr - sr_benchmark)) ** 2)


def annual_to_period_sr(sr_annual: float, periods: int = 252) -> float:
    return sr_annual / math.sqrt(periods)


def effective_trials(n_trials: int, avg_corr: float) -> float:
    """Effective number of independent trials from the average pairwise
    correlation of trial returns: N_eff = rho + (1-rho) N (a simple bound
    used in Lopez de Prado's work on clustered trials). rho clipped to [0,1]."""
    rho = min(max(float(avg_corr), 0.0), 1.0)
    return rho + (1 - rho) * n_trials


# ---------------------------------------------------------------- trial accounting
_DEFAULT_ATLAS = Path(__file__).resolve().parents[2] / "docs" / "strategies.json"


def parse_grid_size(text: str | None) -> int:
    """Approximate number of combinations described by the sheet's
    'Parameters to test' text: product over ';'-separated parameters of the
    number of '/'-separated (or ' vs '-separated) options."""
    if not text:
        return 1
    total = 1
    for chunk in str(text).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        n_slash = chunk.count("/") + 1
        n_vs = len(re.split(r"\bvs\b", chunk)) if re.search(r"\bvs\b", chunk) else 1
        total *= max(n_slash, n_vs, 1)
    return int(total)


def atlas_trials(registry_grids: dict[str, int] | None = None, atlas_json: str | Path | None = None) -> pd.DataFrame:
    """Per-strategy trial counts for all 172 strategies.

    registry_grids: {strategy_id: number_of_grid_combinations} from the
    strategy registry (authoritative when present). Missing ids fall back to
    parsing docs/strategies.json. Returns a frame [strategy_id, n_trials, source]."""
    path = Path(atlas_json) if atlas_json else _DEFAULT_ATLAS
    rows = []
    data = json.loads(path.read_text()) if path.exists() else []
    for s in data:
        sid = s["ID"]
        if registry_grids and sid in registry_grids:
            rows.append((sid, int(registry_grids[sid]), "registry"))
        else:
            rows.append((sid, parse_grid_size(s.get("Parameters to test")), "parsed"))
    if registry_grids:
        known = {r[0] for r in rows}
        rows += [(k, int(v), "registry") for k, v in registry_grids.items() if k not in known]
    return pd.DataFrame(rows, columns=["strategy_id", "n_trials", "source"])


def total_atlas_trials(registry_grids: dict[str, int] | None = None, atlas_json=None) -> int:
    return int(atlas_trials(registry_grids, atlas_json)["n_trials"].sum())
