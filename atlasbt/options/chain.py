"""Chain utilities over the canonical ``option_chain`` frame
(docs/ARCHITECTURE.md §3: ts, underlying, symbol, expiry, strike, right, bid,
ask, mid, iv, delta, gamma, vega, theta, underlying_price, volume,
open_interest, exercise, settlement, multiplier, fidelity).

Deterministic selection rules (documented in docs/OPTIONS.md):
  * DTE = calendar days from the ET session date of ``now`` to ``expiry``.
  * ``choose_expiry``: minimise |dte - target|; ties -> earlier expiry; then
    standard monthly before weekly (if ``prefer_standard``); then root name.
  * ``select_by_delta``: minimise |delta - target|; ties -> strike closer to
    the reference price; then lower strike.  Delta = quoted ``delta`` if finite,
    else BSM delta from quoted ``iv``, else from the IV implied by mid
    (column ``delta_source`` records which).
  * ``nearest_atm``: minimise |K - ref| (ref = forward if given else spot);
    ties -> lower strike.
  * Wings by points use ACTUAL listed strikes: exact strike if listed, else
    policy 'outward' (next listed strike further OTM, i.e. wider -> defined
    risk never understated), 'inward', 'nearest' or 'exact' (None if missing).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from atlasbt.core import clock
from atlasbt.core.types import AtlasError
from atlasbt.pricing.bsm import bsm_delta
from atlasbt.pricing.iv import implied_vol
from atlasbt.pricing.time import year_fractions


class SelectionError(AtlasError):
    """No listed contract satisfies the selection rule (reason in message)."""


def _right_str(s) -> pd.Series:
    return pd.Series(s).astype(str).str.upper().str[0]


def session_of(now) -> pd.Timestamp:
    ts = pd.Timestamp(now)
    if ts.tzinfo is None:
        return ts.normalize()
    return clock.session_date_of(ts)


# --------------------------------------------------------------------------
# preparation
# --------------------------------------------------------------------------
def prepare_chain(chain: pd.DataFrame, now, r: float = 0.0, q: float | None = None,
                  spot: float | None = None) -> pd.DataFrame:
    """Return a copy with: right (C/P), expiry (naive), dte, T, mid, spread,
    rel_spread, delta_used, iv_used, delta_source.

    ``r``, ``q`` continuous (q defaults to 0 or the column ``q`` if present).
    Missing deltas are computed with BSM on spot (European approximation; for
    American puts the quoted delta from the data source is preferred).
    """
    df = chain.copy()
    df["right"] = _right_str(df["right"]).values
    exp = pd.to_datetime(df["expiry"])
    if exp.dt.tz is not None:
        exp = exp.dt.tz_localize(None)
    df["expiry"] = exp.dt.normalize()
    sess = session_of(now)
    df["dte"] = (df["expiry"] - sess).dt.days.astype(int)
    if "settlement" not in df.columns:
        df["settlement"] = "PHYSICAL"
    nowu = clock.to_utc(now) if pd.Timestamp(now).tzinfo is not None else clock.et(sess, "16:00")
    df["T"] = year_fractions(nowu, df["expiry"].values, df["settlement"].values)
    for c in ("bid", "ask"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "mid" not in df.columns or df["mid"].isna().any():
        df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["spread"] = df["ask"] - df["bid"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["rel_spread"] = np.where(df["mid"] > 0, df["spread"] / df["mid"], np.inf)
    S = spot
    if S is None and "underlying_price" in df.columns:
        S = pd.to_numeric(df["underlying_price"], errors="coerce").values
    if S is None:
        raise ValueError("spot or underlying_price required")
    S = np.broadcast_to(np.asarray(S, float), (len(df),))
    qv = df["q"].values if (q is None and "q" in df.columns) else (0.0 if q is None else q)
    is_c = df["right"].values == "C"
    qd = pd.to_numeric(df["delta"], errors="coerce").values if "delta" in df.columns else np.full(len(df), np.nan)
    qiv = pd.to_numeric(df["iv"], errors="coerce").values if "iv" in df.columns else np.full(len(df), np.nan)
    src = np.where(np.isfinite(qd), "quoted", "")
    iv_used = qiv.copy()
    need_iv = ~np.isfinite(iv_used)
    if need_iv.any():
        iv_mid = implied_vol(df["mid"].values[need_iv], S[need_iv], df["strike"].values[need_iv],
                             df["T"].values[need_iv], r, np.broadcast_to(qv, (len(df),))[need_iv] if np.ndim(qv) else qv,
                             is_c[need_iv]).iv
        iv_used[need_iv] = iv_mid
    comp = bsm_delta(S, df["strike"].values, df["T"].values, r, qv, iv_used, is_c)
    delta_used = np.where(np.isfinite(qd), qd, comp)
    src = np.where(np.isfinite(qd), "quoted", np.where(np.isfinite(qiv), "from_quoted_iv",
                                                     np.where(np.isfinite(comp), "from_mid_iv", "none")))
    df["iv_used"] = iv_used
    df["delta_used"] = delta_used
    df["delta_source"] = src
    df["spot"] = S
    return df


# --------------------------------------------------------------------------
# quality
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class QualityRules:
    max_rel_spread: float = 0.50       # (ask-bid)/mid
    max_abs_spread: float | None = None
    require_bid: bool = True           # nonzero bid
    reject_locked_crossed: bool = True
    max_quote_age: pd.Timedelta | None = None   # vs now, uses 'ts' column
    min_open_interest: float | None = None


def quality_mask(df: pd.DataFrame, rules: QualityRules = QualityRules(), now=None) -> pd.Series:
    """Boolean Series (True = usable).  Sets ``df.attrs['quality_reasons']`` to
    a Series of the first failing reason per row ('' if OK)."""
    reason = pd.Series("", index=df.index, dtype=object)

    def mark(cond, why):
        nonlocal reason
        reason = reason.where(~(cond & (reason == "")), why)

    bid, ask = df["bid"], df["ask"]
    mark(~np.isfinite(bid) | ~np.isfinite(ask), "missing_quote")
    if rules.reject_locked_crossed:
        mark(ask <= bid, "locked_or_crossed")
    if rules.require_bid:
        mark(bid <= 0, "zero_bid")
    mid = 0.5 * (bid + ask)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(mid > 0, (ask - bid) / mid, np.inf)
    mark(pd.Series(rel, index=df.index) > rules.max_rel_spread, "wide_spread")
    if rules.max_abs_spread is not None:
        mark((ask - bid) > rules.max_abs_spread, "wide_abs_spread")
    if rules.max_quote_age is not None and now is not None and "ts" in df.columns:
        age = clock.to_utc(now) - pd.to_datetime(df["ts"], utc=True)
        mark(age > rules.max_quote_age, "stale_age")
    if rules.min_open_interest is not None and "open_interest" in df.columns:
        mark(pd.to_numeric(df["open_interest"], errors="coerce").fillna(0) < rules.min_open_interest, "low_oi")
    df.attrs["quality_reasons"] = reason
    return reason == ""


def flag_stale_vs_previous(cur: pd.DataFrame, prev: pd.DataFrame, min_underlying_move: float = 0.005,
                           key=("expiry", "strike", "right")) -> pd.Series:
    """Stale-quote detector between two snapshots: a contract whose bid AND ask
    are unchanged while the underlying moved by >= ``min_underlying_move``
    (relative) and whose |delta| >= 0.05 is flagged stale (True)."""
    k = list(key)
    m = cur.merge(prev[k + ["bid", "ask", "underlying_price"]], on=k, how="left", suffixes=("", "_prev"))
    du = (m["underlying_price"] / m["underlying_price_prev"] - 1).abs()
    dcol = "delta_used" if "delta_used" in m.columns else ("delta" if "delta" in m.columns else None)
    sens = (m[dcol].abs() >= 0.05) if dcol else True
    stale = (m["bid"] == m["bid_prev"]) & (m["ask"] == m["ask_prev"]) & (du >= min_underlying_move) & sens
    return pd.Series(stale.fillna(False).values, index=cur.index)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def filter_dte(df: pd.DataFrame, min_dte: int | None = None, max_dte: int | None = None) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    if min_dte is not None:
        m &= df["dte"] >= min_dte
    if max_dte is not None:
        m &= df["dte"] <= max_dte
    return df[m]


def expiries(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["expiry", "dte"] + [c for c in ("root", "settlement", "kind") if c in df.columns]
    return df[cols].drop_duplicates().sort_values(["expiry"] + cols[2:]).reset_index(drop=True)


def choose_expiry(df: pd.DataFrame, target_dte: float, min_dte: int | None = None, max_dte: int | None = None,
                  settlement: str | None = None, root: str | None = None, prefer_standard: bool = True,
                  tie: str = "earlier") -> pd.Timestamp:
    """Expiry closest to ``target_dte`` within [min_dte, max_dte] (deterministic
    tie-breaks, see module doc).  Raises SelectionError if none."""
    d = filter_dte(df, min_dte, max_dte)
    if settlement is not None:
        d = d[d["settlement"].astype(str) == str(settlement)]
    if root is not None and "root" in d.columns:
        d = d[d["root"] == root]
    if d.empty:
        raise SelectionError(f"no expiry with dte in [{min_dte},{max_dte}]")
    e = expiries(d)
    e = e.assign(_dist=(e["dte"] - target_dte).abs(),
                 _ord=e["dte"] if tie == "earlier" else -e["dte"],
                 _std=0 if "kind" not in e.columns else (~e["kind"].isin(["MONTHLY", "QUARTERLY", "LEAPS"])).astype(int))
    keys = ["_dist", "_ord"] + (["_std"] if prefer_standard else [])
    e = e.sort_values(keys + [c for c in ("root",) if c in e.columns], kind="mergesort")
    return pd.Timestamp(e["expiry"].iloc[0])


def slice_expiry(df: pd.DataFrame, expiry, root: str | None = None, settlement: str | None = None) -> pd.DataFrame:
    s = df[df["expiry"] == pd.Timestamp(expiry)]
    if root is not None and "root" in s.columns:
        s = s[s["root"] == root]
    if settlement is not None:
        s = s[s["settlement"].astype(str) == str(settlement)]
    return s


def select_by_delta(df_exp: pd.DataFrame, target_delta: float, right: str | None = None,
                    usable: pd.Series | None = None, ref_price: float | None = None,
                    max_delta_error: float | None = None) -> pd.Series:
    """Row (Series) whose delta is closest to ``target_delta`` (calls +, puts -).
    ``right`` inferred from sign if omitted."""
    right = (right or ("C" if target_delta > 0 else "P")).upper()[0]
    d = df_exp[df_exp["right"] == right]
    if usable is not None:
        d = d[usable.reindex(d.index).fillna(False).astype(bool)]
    d = d[np.isfinite(d["delta_used"])]
    if d.empty:
        raise SelectionError(f"no {right} with finite delta")
    ref = ref_price if ref_price is not None else float(np.nanmedian(d["spot"])) if "spot" in d.columns else 0.0
    d = d.assign(_e=(d["delta_used"] - target_delta).abs().round(10), _a=(d["strike"] - ref).abs())
    d = d.sort_values(["_e", "_a", "strike"], kind="mergesort")
    row = d.iloc[0]
    if max_delta_error is not None and row["_e"] > max_delta_error:
        raise SelectionError(f"closest delta {row['delta_used']:.3f} too far from {target_delta}")
    return row.drop(labels=["_e", "_a"])


def nearest_atm(df_exp: pd.DataFrame, ref_price: float, right: str | None = None,
                usable: pd.Series | None = None, both_rights: bool = False) -> float:
    """ATM strike (float).  ``both_rights``: only strikes listed (and usable)
    for both calls and puts (straddles / iron flies)."""
    d = df_exp if right is None else df_exp[df_exp["right"] == right.upper()[0]]
    if usable is not None:
        d = d[usable.reindex(d.index).fillna(False).astype(bool)]
    if both_rights:
        ks = set(d[d["right"] == "C"]["strike"]) & set(d[d["right"] == "P"]["strike"])
    else:
        ks = set(d["strike"])
    if not ks:
        raise SelectionError("no strikes")
    ks = np.array(sorted(ks))
    dist = np.round(np.abs(ks - ref_price), 10)
    return float(ks[np.lexsort((ks, dist))][0])


def get_row(df_exp: pd.DataFrame, strike: float, right: str) -> pd.Series:
    d = df_exp[(np.isclose(df_exp["strike"], strike)) & (df_exp["right"] == right.upper()[0])]
    if d.empty:
        raise SelectionError(f"strike {strike}{right} not listed")
    return d.iloc[0]


def listed_strikes(df_exp: pd.DataFrame, right: str | None = None, usable: pd.Series | None = None) -> np.ndarray:
    d = df_exp if right is None else df_exp[df_exp["right"] == right.upper()[0]]
    if usable is not None:
        d = d[usable.reindex(d.index).fillna(False).astype(bool)]
    return np.array(sorted(set(d["strike"].astype(float))))


def wing_by_points(df_exp: pd.DataFrame, base_strike: float, points: float, right: str,
                   policy: str = "outward", usable: pd.Series | None = None) -> float:
    """Strike at base +/- points (calls: above, puts: below), snapped to an
    actually listed strike per ``policy``."""
    right = right.upper()[0]
    target = base_strike + points if right == "C" else base_strike - points
    ks = listed_strikes(df_exp, right, usable)
    if ks.size == 0:
        raise SelectionError("no strikes")
    hit = ks[np.isclose(ks, target)]
    if hit.size:
        return float(hit[0])
    if policy == "exact":
        raise SelectionError(f"wing strike {target} not listed")
    out = ks[ks > target] if right == "C" else ks[ks < target]
    inn = ks[(ks < target) & (ks > base_strike)] if right == "C" else ks[(ks > target) & (ks < base_strike)]
    if policy == "outward":
        if out.size == 0:
            raise SelectionError(f"no listed strike beyond {target}")
        return float(out.min() if right == "C" else out.max())
    if policy == "inward":
        if inn.size == 0:
            raise SelectionError(f"no listed strike inside {target}")
        return float(inn.max() if right == "C" else inn.min())
    if policy == "nearest":
        cand = ks[ks != base_strike]
        dist = np.round(np.abs(cand - target), 10)
        return float(cand[np.lexsort((-cand if right == "C" else cand, dist))][0])
    raise ValueError(policy)
