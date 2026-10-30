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

from atlasbt.core import clock
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
                  spread_params=None, root: str | None = None, is_etf: bool | None = None) -> Mark:
    """``root``/``settlement`` select the contract's SERIES (SPX AM vs SPXW PM share
    expiry dates); the surface fallback uses only that series."""
    right = right.upper()[0]
    expiry = pd.Timestamp(expiry).normalize()
    x = prepare_chain(chain, now, r=r)
    if root is not None and "root" in x.columns:
        x = x[x["root"].astype(str) == str(root)]
    if "settlement" in x.columns and (x["settlement"].astype(str) == str(settlement)).any() \
            and x["settlement"].astype(str).nunique() > 1:
        x = x[x["settlement"].astype(str) == str(settlement)]
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
    from .listing import WEEKLY_ETFS, product_spec, tick_size
    from .synthetic import SyntheticParams, VolSource, model_spread
    p = spread_params or SyntheticParams.load()
    und = (underlying or (str(x["underlying"].iloc[0]) if "underlying" in x.columns else "")).upper()
    if is_etf is None:
        spec = product_spec(und)
        is_etf = (not spec.is_index) and (spec.broad_based or und in WEEKLY_ETFS)
    # point-in-time VIX (close usable only after 16:15 ET) drives the spread level
    fam = VolSource(params=p).vix_family(now if pd.Timestamp(now).tzinfo is not None else clock.et(now, "16:30"))
    vix = float(fam.loc[fam["index"] == "VIX", "level"].iloc[0]) if (fam["index"] == "VIX").any() else 0.20
    if not (fam["index"] == "VIX").any():
        reason.append("VIX unavailable -> spread at VIX=20")
    dte = max((expiry - (pd.Timestamp(now).tz_convert("America/New_York").tz_localize(None).normalize()
                         if pd.Timestamp(now).tzinfo is not None else pd.Timestamp(now).normalize())).days, 1)
    from .synthetic import params_fit_reasons
    pf, use_fe = params_fit_reasons(p, now if pd.Timestamp(now).tzinfo is None else
                                    pd.Timestamp(now).tz_convert("America/New_York").tz_localize(None))
    reason += pf
    spr = float(model_spread(mid, dte, abs(delta), und, p, is_etf=is_etf, vix=vix, root=root, use_symbol_fe=use_fe))
    sess = pd.Timestamp(now).tz_convert("America/New_York").tz_localize(None).normalize() \
        if pd.Timestamp(now).tzinfo is not None else pd.Timestamp(now)
    tick = float(tick_size(und, mid, sess))
    bid = np.floor((mid - spr / 2) / tick + 1e-9) * tick
    ask = np.ceil((mid + spr / 2) / tick - 1e-9) * tick
    ask = max(ask, bid + tick)
    bid = 0.0 if bid < tick - 1e-12 else bid
    reason.append(f"PIT VIX={vix:.4f}, tick {tick:g}")
    reason.append("bid/ask from synthetic spread model")
    return Mark(float(bid), float(ask), mid, iv, delta, fid, "; ".join(reason))


# ==========================================================================
# Fallback quote stream for the engine's Portal.option_quotes
# ==========================================================================
QUOTE_COLS = ["underlying", "symbol", "expiry", "strike", "right", "bid", "ask", "mid", "iv", "delta", "gamma",
              "vega", "theta", "underlying_price", "volume", "open_interest", "exercise", "settlement", "multiplier",
              "fidelity", "root", "settlement_date", "kind", "underlying_price_source", "source", "reason"]


def _pit_inputs(portal, und: str, ts: pd.Timestamp) -> dict:
    """Spot, r (continuous), q and rv_ratio known at ``ts`` — the same inputs the DATA
    agent's ``Portal.synthetic_chain`` uses (its helpers if present, else public bars)."""
    from .synthetic import INDEX_MAP, SPX_LIKE
    spot = float("nan")
    if hasattr(portal, "_spot"):
        spot = float(portal._spot(und, ts))
    if not np.isfinite(spot):
        try:
            b = portal.bars({"SPXW": "SPX"}.get(und, und), "1d", ts - pd.Timedelta(days=10), ts, ts)
            spot = float(b["close"].iloc[-1]) if len(b) else float("nan")
        except Exception:
            pass
    try:
        from atlasbt.data.rates import money_market_to_continuous
        r = float(money_market_to_continuous(portal.risk_free_rate(ts, "3M"), 91))
    except Exception:
        r = 0.0
    q = float(portal._div_yield(und, ts, spot)) if hasattr(portal, "_div_yield") and np.isfinite(spot) else 0.0
    rv = None
    if not (und in SPX_LIKE or und in INDEX_MAP) and hasattr(portal, "_rv_ratio"):
        rv = float(portal._rv_ratio(und, ts))
    return {"spot": spot, "r": r, "q": q, "rv_ratio": rv}


def synthetic_option_quotes(instrument, start, end, as_of, portal, real_first: bool = True,
                            include_open: bool = False, generator=None, strict: bool = False) -> pd.DataFrame:
    """Quote snapshots of ONE contract with ts in (start, end] and ts <= as_of, for
    use as the engine's fallback when ``Portal.option_quotes`` has no real rows.

    Snapshot times: each NYSE session close (and, if ``include_open``, each session
    open) up to the contract's last tradable close (expiry session close for PM /
    physical; the last-trading-day close for AM-settled SPX — settlement itself is
    the lifecycle's job).  Per snapshot, in order of preference:

      1. the REAL quote of the contract at that ts (portal.option_quotes)  -> its tier (T2)
      2. a mark anchored to the SAME-DAY real chain (contract missing, e.g. DoltHub
         drops < ~13 DTE): ``mark_contract`` on that snapshot -> T2_PROXY inside the
         quoted range, T3_MODELED when extrapolated
      3. purely synthetic: ``SyntheticChainGenerator.quote_contract`` with PIT spot,
         r, q, rv_ratio, VIX family and SKEW known at ts -> T3_MODELED; reasons such as
         'params_fit_post_as_of' / 'atm_ratio_prior' are in the ``reason`` column.

    Returns a frame indexed by ts (UTC) with ``QUOTE_COLS``; attrs: fidelity (worst
    row), synthetic (any T3 row), symbol."""
    from atlasbt.core.types import SettlementType
    from atlasbt.pricing.time import settlement_instant, settlement_session
    from .synthetic import SyntheticChainGenerator
    start, end, as_of = clock.to_utc(start), clock.to_utc(end), clock.to_utc(as_of)
    hi = min(end, as_of)
    und = (instrument.underlying or "").upper()
    root = instrument.symbol[:6].strip()
    st = (instrument.settlement or SettlementType.PHYSICAL).value
    ex = instrument.exercise.value if instrument.exercise is not None else "AMERICAN"
    expiry = pd.Timestamp(instrument.expiry).normalize()
    cal = clock.nyse()
    ltd_close = cal.open_close(expiry)[1] if cal.is_session(expiry) else clock.et(expiry, "16:00")
    hi = min(hi, ltd_close, settlement_instant(expiry, st))
    out_empty = pd.DataFrame(columns=QUOTE_COLS, index=pd.DatetimeIndex([], tz="UTC", name="ts"))
    out_empty.attrs.update(fidelity=int(Fidelity.NA), synthetic=False, symbol=instrument.symbol)
    if hi <= start:
        return out_empty
    stamps = []
    for sess in cal.sessions(clock.session_date_of(start), clock.session_date_of(hi)):
        o, c = cal.open_close(sess)
        for t in ((o, c) if include_open else (c,)):
            if start < t <= hi:
                stamps.append(t)
    real = None
    if real_first:
        try:
            real = portal.option_quotes(instrument, start, end, as_of)
        except Exception:
            real = None
    gen = generator or SyntheticChainGenerator()
    sdate = settlement_session(expiry, st)
    rows, idx = [], []
    for t in stamps:
        base = dict(underlying=und, symbol=instrument.symbol, expiry=expiry, strike=float(instrument.strike),
                    right=instrument.right.value, volume=np.nan, open_interest=np.nan, exercise=ex, settlement=st,
                    multiplier=float(instrument.multiplier), root=root, settlement_date=sdate, kind=None)
        if real is not None and len(real) and t in real.index:
            rr = real.loc[[t]].iloc[-1]
            row = {**base, **{c: rr[c] for c in QUOTE_COLS if c in rr.index and c not in ("reason",)},
                   "reason": "real quote", "source": rr.get("source", "real")}
            rows.append(row)
            idx.append(t)
            continue
        inp = _pit_inputs(portal, und, t)
        anchored = None
        try:
            ch = portal.option_chain(und, t, synthetic=False)
        except TypeError:
            ch = pd.DataFrame()
        except Exception:
            ch = pd.DataFrame()
        if ch is not None and len(ch) and not ch.attrs.get("synthetic", False):
            snap_ts = ch.attrs.get("snapshot_ts", pd.to_datetime(ch["ts"]).max() if "ts" in ch.columns else None)
            if snap_ts is not None and clock.session_date_of(pd.Timestamp(snap_ts)) == clock.session_date_of(t):
                m = mark_contract(ch, t, inp["r"], expiry, float(instrument.strike), instrument.right.value,
                                  q=None, exercise=ex, settlement=st, underlying=und, root=root)
                if np.isfinite(m.mid):
                    anchored = m
        if anchored is not None:
            S = float(pd.to_numeric(ch["underlying_price"], errors="coerce").dropna().median()) \
                if "underlying_price" in ch.columns else inp["spot"]
            rows.append({**base, "bid": anchored.bid, "ask": anchored.ask, "mid": anchored.mid, "iv": anchored.iv,
                         "delta": anchored.delta, "gamma": np.nan, "vega": np.nan, "theta": np.nan,
                         "underlying_price": S, "fidelity": int(anchored.fidelity),
                         "underlying_price_source": "real_chain_snapshot", "source": "anchored_real_surface",
                         "reason": "anchored to same-day real surface: " + anchored.reason})
            idx.append(t)
            continue
        if not np.isfinite(inp["spot"]):
            continue
        qd = gen.quote_contract(und, t, inp["spot"], inp["r"], inp["q"], expiry, float(instrument.strike),
                                instrument.right.value, root=root, settlement=st, rv_ratio=inp["rv_ratio"],
                                strict=strict)
        if qd is None:
            continue
        rows.append({**base, **{k: qd[k] for k in ("bid", "ask", "mid", "iv", "delta", "gamma", "vega", "theta",
                                                     "underlying_price")},
                     "fidelity": int(Fidelity.T3_MODELED), "underlying_price_source": "pit_last_price",
                     "source": "synthetic", "reason": "synthetic" + ("; " + "; ".join(qd["reasons"]) if qd["reasons"] else "")})
        idx.append(t)
    if not rows:
        return out_empty
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="ts"))[QUOTE_COLS]
    fids = pd.to_numeric(out["fidelity"], errors="coerce")
    out.attrs.update(fidelity=int(fids.max()), synthetic=bool((fids >= int(Fidelity.T3_MODELED)).any()),
                     symbol=instrument.symbol)
    return out
