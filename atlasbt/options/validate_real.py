"""Synthetic (T3) chain vs real OPRA closing quotes (Databento cbbo-1m, docs/OPTIONS.md §7.4b).

For each validation session (``databento.validation_days``: one mid-month session every second month,
2023-04 .. 2026-10) and each of SPY / QQQ / IWM, the real chain is the OPRA consolidated BBO sampled at
15:59 ET (DTE 5-70, |K/S-1| <= 12%) and the synthetic chain is ``Portal.synthetic_chain`` built at 16:20 ET
from the same session's PIT inputs (VIX family / VXN / RVX closes known after 16:15). Both go through
``options.metrics.chain_metrics`` (IV from OTM mids on the parity forward), so differences are
model differences, not methodology. Also compared: the 30-DTE ATM put premium (mid / spot) and the
quoted spread of near-the-money (|delta| 0.3-0.7) options at 20-40 DTE.

    python -m atlasbt.options.validate_real            # prints the table, writes reference/databento_validation/summary.csv
    python -m atlasbt.options.validate_real --bias     # per-contract bias by moneyness / DTE bucket (bias.csv)

``surface_bias(real_quotes, ...)`` is the contract-level measurement: every usable OTM real quote is
priced on the synthetic surface at the same instant (``SyntheticChainGenerator.quote_contract``, same spot,
r, q) and the mid error is reported in dollars and in implied-vol points, aggregated by moneyness
(K / S) and DTE bucket. ``bias_from_store`` runs it on the stored samples.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from atlasbt.core.clock import et
from atlasbt.data.providers import databento as D
from atlasbt.options.chain import prepare_chain
from atlasbt.options.metrics import chain_metrics


def _atm_put(x: pd.DataFrame, S: float) -> tuple[float, float]:
    """(mid/S, dte) of the put nearest ATM in the expiry closest to 30 DTE (20-45)."""
    e = x[(x["dte"] >= 20) & (x["dte"] <= 45)]
    if e.empty:
        return float("nan"), float("nan")
    tgt = e.loc[(e["dte"] - 30).abs().idxmin(), "expiry"]
    p = e[(e["expiry"] == tgt) & (e["right"] == "P")]
    if p.empty:
        return float("nan"), float("nan")
    r = p.loc[(p["strike"] - S).abs().idxmin()]
    return float(r["mid"]) / S, float(r["dte"])


def _ntm_spread(x: pd.DataFrame) -> float:
    e = x[(x["dte"] >= 20) & (x["dte"] <= 40) & (x["delta_used"].abs().between(0.3, 0.7))]
    s = (e["ask"] - e["bid"])[(e["ask"] > e["bid"]) & (e["bid"] > 0)]
    return float(s.median()) if len(s) else float("nan")


def compare(portal=None, days=None, symbols=("SPY", "QQQ", "IWM")) -> pd.DataFrame:
    from atlasbt.data.portal import Portal
    p = portal or Portal()
    rows = []
    for d in (days or sorted(pd.Timestamp(f.stem) for f in D.validation_dir().glob("*.parquet"))):
        f = D.validation_dir() / f"{pd.Timestamp(d):%Y-%m-%d}.parquet"
        if not f.exists():
            continue
        q = pd.read_parquet(f)
        q["und"] = q["symbol"].str[:6].str.strip()
        _o, c = D.open_close(d)
        t_real = c - pd.Timedelta(minutes=1)
        t_syn = c + pd.Timedelta(minutes=20)
        r = p.risk_free_rate(t_syn, "3M")
        r_cc = float(np.log1p(r * 91 / 360) * 365 / 91) if np.isfinite(r) else 0.0
        for u in symbols:
            g = q[(q["und"] == u) & (q["ts"] == t_real)]
            if g.empty:
                continue
            S = p._opra_spot(u, t_real)
            if not np.isfinite(S):
                continue
            real = D.chain_frame(g.drop(columns=["und"]), u, spot_fn=lambda t, S=S: S)
            syn = p.synthetic_chain(u, t_syn, min_dte=5, max_dte=70)
            if syn.empty:
                continue
            try:
                mr = chain_metrics(real, t_real, r_cc, spot=S, american=True, use_cache=False)
                ms = chain_metrics(syn, t_syn, r_cc, spot=float(syn.attrs.get("inputs", {}).get("spot", S)),
                                   american=True, use_cache=False)
                xr = prepare_chain(real, t_real, r=r_cc)
                xs = prepare_chain(syn, t_syn, r=r_cc)
            except Exception:
                continue
            pr, dr = _atm_put(xr, S)
            ps, ds = _atm_put(xs, float(syn["underlying_price"].iloc[0]))
            rows.append({"day": pd.Timestamp(d).date(), "und": u, "spot": S,
                         "atm30_real": mr.get("atm_iv_30"), "atm30_syn": ms.get("atm_iv_30"),
                         "skew25_real": mr.get("skew25_30"), "skew25_syn": ms.get("skew25_30"),
                         "put30_real": pr, "put30_syn": ps, "spread_real": _ntm_spread(xr),
                         "spread_syn": _ntm_spread(xs)})
    return pd.DataFrame(rows)


def compare_spx(portal=None) -> pd.DataFrame:
    """SPX: month-end closing CHAIN samples of the served store (plan spxeod, 15:59 ET, SPX AM + SPXW PM) vs
    the synthetic SPX chain at 16:20 ET. Spot = parity of the nearest expiry of the same snapshot."""
    from atlasbt.data.portal import Portal
    p = portal or Portal()
    rd = D.OpraReader(p.store.root)
    rows = []
    for d in rd.days("SPX"):
        _o, c = D.open_close(d)
        t_real = c - pd.Timedelta(minutes=1)
        f = rd.frame("SPX", d)
        g = f[(f["ts"] == t_real) & f["chain"]]
        if len(g) < 500:              # month-end (wide) snapshots only
            continue
        real = D.chain_frame(g, "SPX")
        S = float(real["underlying_price"].iloc[0])
        t_syn = c + pd.Timedelta(minutes=20)
        r = p.risk_free_rate(t_syn, "3M")
        r_cc = float(np.log1p(r * 91 / 360) * 365 / 91) if np.isfinite(r) else 0.0
        syn = p.synthetic_chain("SPX", t_syn, min_dte=5, max_dte=70)
        if syn.empty or not np.isfinite(S):
            continue
        real = real[(pd.to_datetime(real["expiry"]) - pd.Timestamp(d)).dt.days.between(5, 70)]
        try:
            mr = chain_metrics(real, t_real, r_cc, spot=S, american=False, use_cache=False)
            ms = chain_metrics(syn, t_syn, r_cc, spot=float(syn["underlying_price"].iloc[0]), american=False,
                               use_cache=False)
            xr, xs = prepare_chain(real, t_real, r=r_cc), prepare_chain(syn, t_syn, r=r_cc)
        except Exception:
            continue
        pr, _ = _atm_put(xr, S)
        ps, _ = _atm_put(xs, float(syn["underlying_price"].iloc[0]))
        rows.append({"day": pd.Timestamp(d).date(), "und": "SPX", "spot": S,
                     "atm30_real": mr.get("atm_iv_30"), "atm30_syn": ms.get("atm_iv_30"),
                     "skew25_real": mr.get("skew25_30"), "skew25_syn": ms.get("skew25_30"),
                     "put30_real": pr, "put30_syn": ps, "spread_real": _ntm_spread(xr), "spread_syn": _ntm_spread(xs)})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for u, g in df.groupby("und"):
        out.append({"und": u, "n_days": len(g),
                    "atm30_diff_volpts_median": float(np.nanmedian((g["atm30_syn"] - g["atm30_real"]) * 100)),
                    "atm30_diff_volpts_mean": float(np.nanmean((g["atm30_syn"] - g["atm30_real"]) * 100)),
                    "skew_ratio_median": float(np.nanmedian(g["skew25_syn"] / g["skew25_real"])),
                    "put30_premium_ratio_median": float(np.nanmedian(g["put30_syn"] / g["put30_real"])),
                    "ntm_spread_ratio_median": float(np.nanmedian(g["spread_syn"] / g["spread_real"]))})
    return pd.DataFrame(out)


# ============================================================================ contract-level bias
#: K / S bucket edges (left-closed) and DTE bucket edges (calendar days from the session of ts, left-closed)
MNY_EDGES = (0.0, 0.90, 0.95, 0.98, 1.02, 1.05, 1.10, np.inf)
DTE_EDGES = (0, 1, 8, 31, 61, 121, 100_000)
#: |K/S - 1| below this counts as at the money: both the call and the put are kept by otm_only
ATM_TOL = 0.005


def _labels(edges, fmt) -> list:
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if a == edges[0] and fmt == "mny":
            out.append(f"<{b:.2f}")
        elif not np.isfinite(b) or b >= 100_000:
            out.append(f">{a:.2f}" if fmt == "mny" else f"{a}+")
        elif fmt == "mny":
            out.append(f"{a:.2f}-{b:.2f}")
        else:
            out.append(str(a) if b - a == 1 else f"{a}-{b - 1}")
    return out


DETAIL_COLS = ["ts", "underlying", "symbol", "root", "expiry", "settlement", "exercise", "strike", "right", "dte",
               "spot", "moneyness", "mny_bucket", "dte_bucket", "real_bid", "real_ask", "real_mid", "real_iv",
               "syn_mid", "syn_iv", "err_usd", "err_rel", "err_volpts", "real_spread", "syn_spread"]
SUMMARY_COLS = ["underlying", "dte_bucket", "mny_bucket", "n", "err_volpts_mean", "err_volpts_median",
                "err_usd_mean", "err_usd_median", "err_rel_median", "real_spread_median", "syn_spread_median"]


@dataclass
class BiasResult:
    """detail: one row per priced contract (DETAIL_COLS); summary: per (underlying, DTE bucket, moneyness
    bucket) statistics (SUMMARY_COLS). Errors are synthetic - real: err_usd per share (x100 per contract),
    err_rel = err_usd / real mid, err_volpts = (synthetic IV - real IV) x 100. n_skipped = quotes that could
    not be measured (no two-sided quote, crossed, expired at ts, no spot, model returned nothing); quotes
    left out by ``otm_only`` are not counted."""
    detail: pd.DataFrame
    summary: pd.DataFrame
    n_skipped: int = 0
    notes: list = field(default_factory=list)


def _portal_rates(portal, und: str, ts: pd.Timestamp, spot: float) -> tuple[float, float]:
    from atlasbt.data.rates import money_market_to_continuous
    try:
        r = float(money_market_to_continuous(portal.risk_free_rate(ts, "3M"), 91))
    except Exception:
        r = 0.0
    try:
        q = float(portal._div_yield(und, ts, spot)) if np.isfinite(spot) else 0.0
    except Exception:
        q = 0.0
    return r, q


def _iv(prices, S, K, T, r, q, is_call, american: bool) -> np.ndarray:
    from atlasbt.pricing.iv import american_implied_vol_fast, implied_vol
    if not len(prices):
        return np.array([])
    f = american_implied_vol_fast if american else implied_vol
    return np.asarray(f(np.asarray(prices, float), S, np.asarray(K, float), np.asarray(T, float), r, q,
                        np.asarray(is_call, bool)).iv, float)


def summarize_bias(detail: pd.DataFrame) -> pd.DataFrame:
    if detail is None or detail.empty:
        return pd.DataFrame(columns=SUMMARY_COLS)
    g = detail.groupby(["underlying", "dte_bucket", "mny_bucket"], observed=True, sort=True)
    out = g.agg(n=("err_usd", "size"), err_volpts_mean=("err_volpts", "mean"),
                err_volpts_median=("err_volpts", "median"), err_usd_mean=("err_usd", "mean"),
                err_usd_median=("err_usd", "median"), err_rel_median=("err_rel", "median"),
                real_spread_median=("real_spread", "median"), syn_spread_median=("syn_spread", "median"))
    out = out.reset_index()
    for c in ("dte_bucket", "mny_bucket"):
        out[c] = out[c].astype(str)
    return out[SUMMARY_COLS]


def surface_bias(real_quotes: pd.DataFrame, synthetic=None, spot: Optional[Callable] = None,
                 rates: Optional[Callable] = None, portal=None, syn_as_of: Optional[Callable] = None,
                 otm_only: bool = True, mny_edges=MNY_EDGES, dte_edges=DTE_EDGES) -> BiasResult:
    """Synthetic-surface bias against stored REAL quotes.

    ``real_quotes``: rows in the stored OPRA format (``databento.QCOLS``: ts, symbol (internal OCC), bid, ask,
    ...), e.g. an ``option_quotes/databento/<UND>/<day>.parquet`` file or ``reference/databento_validation``;
    several instants / underlyings are fine (underlying = the root's, SPXW -> SPX).
    Per (underlying, ts) snapshot: spot = ``spot(und, ts)`` if given, else put-call parity of the snapshot for
    index roots (``databento.parity_spot``), else ``portal._opra_spot`` (1m bar known at ts) when a portal is
    available; snapshots without a spot are skipped. ``rates(und, ts) -> (r_cont, q)``; default the Portal's
    3M T-bill (continuous) and trailing dividend yield (a default Portal is built only in that case).
    Each usable quote (bid > 0, ask >= bid, ts before settlement; OTM or within ATM_TOL when ``otm_only``) is
    priced with ``synthetic.quote_contract(und, syn_as_of(ts), spot, r, q, expiry, strike, right, root=,
    settlement=)`` (default ``SyntheticChainGenerator()``; ``syn_as_of`` default = ts, i.e. what a strategy
    deciding at ts would be shown). Both mids are inverted with the same solver (European BSM for index
    options, Bjerksund-Stensland 2002 for American) at their own instants."""
    from atlasbt.data.providers import databento as D
    from atlasbt.pricing.time import year_fraction
    notes = []
    if real_quotes is None or real_quotes.empty:
        return BiasResult(pd.DataFrame(columns=DETAIL_COLS), pd.DataFrame(columns=SUMMARY_COLS), 0, notes)
    if synthetic is None:
        from atlasbt.options.synthetic import SyntheticChainGenerator
        synthetic = SyntheticChainGenerator()
    if rates is None and portal is None:
        from atlasbt.data.portal import Portal
        portal = Portal()
    q = real_quotes.copy()
    q["ts"] = pd.to_datetime(q["ts"], utc=True)
    m = D.internal_map_from_internal(q["symbol"].astype(str).unique())
    x = q.merge(m, on="symbol", how="inner")
    x["underlying"] = [D.underlying_of_root(r) for r in x["root"]]
    skipped = len(q) - len(x)
    mny_lab, dte_lab = _labels(mny_edges, "mny"), _labels(dte_edges, "dte")
    rows = []
    for (und, ts), g in x.groupby(["underlying", "ts"], sort=True):
        if spot is not None:
            S = float(spot(und, ts))
        elif und in D.INDEX_ROOTS:
            S = float(D.parity_spot(g)[0])
        elif portal is not None:
            S = float(portal._opra_spot(und, ts))
        else:
            S = float("nan")
        if not np.isfinite(S) or S <= 0:
            skipped += len(g)
            continue
        r, qy = rates(und, ts) if rates is not None else _portal_rates(portal, und, ts, S)
        t_syn = syn_as_of(ts) if syn_as_of is not None else ts
        sess = ts.tz_convert("America/New_York").tz_localize(None).normalize()
        for row in g.itertuples(index=False):
            bid, ask, K, right = float(row.bid), float(row.ask), float(row.strike), str(row.right)
            T = year_fraction(ts, pd.Timestamp(row.expiry), row.settlement)
            if not (np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask >= bid) or T <= 0:
                skipped += 1
                continue
            mny = K / S
            if otm_only and abs(mny - 1) > ATM_TOL and ((right == "C" and mny < 1) or (right == "P" and mny > 1)):
                continue
            try:
                sq = synthetic.quote_contract(und, t_syn, S, r, qy, pd.Timestamp(row.expiry), K, right,
                                              root=row.root, settlement=row.settlement)
            except Exception as e:                       # model without inputs at t_syn
                sq = None
                if len(notes) < 20:
                    notes.append(f"{row.symbol} {ts}: {type(e).__name__}: {e}")
            if not sq or not np.isfinite(sq.get("mid", np.nan)):
                skipped += 1
                continue
            rows.append({"ts": ts, "underlying": und, "symbol": row.symbol, "root": row.root,
                         "expiry": pd.Timestamp(row.expiry), "settlement": row.settlement, "exercise": row.exercise,
                         "strike": K, "right": right, "dte": int((pd.Timestamp(row.expiry) - sess).days),
                         "spot": S, "moneyness": mny, "real_bid": bid, "real_ask": ask, "real_mid": (bid + ask) / 2,
                         "syn_mid": float(sq["mid"]), "syn_bid": float(sq.get("bid", np.nan)),
                         "syn_ask": float(sq.get("ask", np.nan)), "syn_iv_model": float(sq.get("iv", np.nan)),
                         "T": T, "T_syn": year_fraction(t_syn, pd.Timestamp(row.expiry), row.settlement),
                         "r": r, "q": qy})
    if not rows:
        return BiasResult(pd.DataFrame(columns=DETAIL_COLS), pd.DataFrame(columns=SUMMARY_COLS), skipped, notes)
    d = pd.DataFrame(rows)
    d["real_iv"] = np.nan
    d["syn_iv"] = np.nan
    for am in (False, True):
        k = (d["exercise"] == "AMERICAN") == am
        if not k.any():
            continue
        z = d[k]
        cp = (z["right"] == "C").to_numpy()
        d.loc[k, "real_iv"] = _iv(z["real_mid"], z["spot"], z["strike"], z["T"], z["r"], z["q"], cp, am)
        siv = _iv(z["syn_mid"], z["spot"], z["strike"], z["T_syn"].clip(lower=1e-9), z["r"], z["q"], cp, am)
        d.loc[k, "syn_iv"] = np.where(np.isfinite(siv), siv, z["syn_iv_model"])
    d["err_usd"] = d["syn_mid"] - d["real_mid"]
    d["err_rel"] = d["err_usd"] / d["real_mid"]
    d["err_volpts"] = (d["syn_iv"] - d["real_iv"]) * 100.0
    d["real_spread"] = d["real_ask"] - d["real_bid"]
    d["syn_spread"] = d["syn_ask"] - d["syn_bid"]
    d["mny_bucket"] = pd.cut(d["moneyness"], list(mny_edges), labels=mny_lab, right=False).astype(str)
    d["dte_bucket"] = pd.cut(d["dte"], list(dte_edges), labels=dte_lab, right=False).astype(str)
    d = d[DETAIL_COLS].sort_values(["underlying", "ts", "expiry", "strike", "right"]).reset_index(drop=True)
    return BiasResult(d, summarize_bias(d), skipped, notes)


def load_stored_quotes(underlying: str, days=None, validation: bool = False, at_close_minus_1m: bool = True,
                       root=None) -> pd.DataFrame:
    """Stored OPRA rows of ``underlying`` (served store, or ``reference/databento_validation`` when
    ``validation``) for ``days`` (default: every stored session), optionally only the 15:59 ET closing
    sample (close - 1 minute, DATA.md §4.15c). No network."""
    und, rf = D.store_underlying(underlying)
    if validation:
        files = sorted(D.validation_dir(root).glob("*.parquet"))
    else:
        files = sorted(D.day_dir(und, root).glob("*.parquet")) if D.day_dir(und, root).is_dir() else []
    want = None if days is None else {pd.Timestamp(d).normalize() for d in days}
    parts = []
    for f in files:
        d = pd.Timestamp(f.stem)
        if want is not None and d not in want:
            continue
        x = pd.read_parquet(f)
        x["ts"] = pd.to_datetime(x["ts"], utc=True)
        roots = x["symbol"].str[:6].str.strip()
        x = x[roots == rf] if rf is not None else x[[D.underlying_of_root(r) == und for r in roots]]
        if at_close_minus_1m and len(x):
            _o, c = D.open_close(d)
            x = x[x["ts"] == c - pd.Timedelta(minutes=1)]
        if len(x):
            parts.append(x)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=D.QCOLS)


def bias_from_store(symbols=("SPY", "QQQ", "IWM", "SPX"), portal=None, syn_lag_minutes: float = 21.0,
                    days=None) -> BiasResult:
    """surface_bias over the stored closing samples: SPY/QQQ/IWM from the validation set, SPX from the served
    store. syn_lag_minutes = 21: the synthetic surface is built at close + 20 min (same-day vol-index closes,
    the convention of ``compare``); 0 = what a strategy deciding at 15:59 sees."""
    from atlasbt.data.portal import Portal
    p = portal or Portal()
    parts = [load_stored_quotes(u, days=days, validation=u not in D.INDEX_ROOTS, root=p.store.root)
             for u in symbols]
    real = pd.concat([x for x in parts if len(x)], ignore_index=True) if any(len(x) for x in parts) else \
        pd.DataFrame(columns=D.QCOLS)
    lag = pd.Timedelta(minutes=syn_lag_minutes)
    return surface_bias(real, portal=p, syn_as_of=(lambda t: t + lag) if syn_lag_minutes else None)


if __name__ == "__main__":  # pragma: no cover
    import sys
    pd.set_option("display.width", 200)
    if "--bias" in sys.argv:
        res = bias_from_store()
        print(res.summary.round(4).to_string())
        print(f"priced {len(res.detail)}, skipped {res.n_skipped}")
        res.detail.to_csv(D.validation_dir() / "bias_detail.csv", index=False)
        res.summary.to_csv(D.validation_dir() / "bias.csv", index=False)
        sys.exit(0)
    df = pd.concat([compare(), compare_spx()], ignore_index=True)
    print(df.round(4).to_string())
    s = summarize(df)
    print(s.round(3).to_string())
    df.to_csv(D.validation_dir() / "detail.csv", index=False)
    s.to_csv(D.validation_dir() / "summary.csv", index=False)
