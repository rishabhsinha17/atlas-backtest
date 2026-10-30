"""Synthetic (T3) chain vs real OPRA closing quotes (Databento cbbo-1m, docs/OPTIONS.md §7.4b).

For each validation session (``databento.validation_days``: one mid-month session every second month,
2023-04 .. 2026-10) and each of SPY / QQQ / IWM, the real chain is the OPRA consolidated BBO sampled at
15:59 ET (DTE 5-70, |K/S-1| <= 12%) and the synthetic chain is ``Portal.synthetic_chain`` built at 16:20 ET
from the same session's PIT inputs (VIX family / VXN / RVX closes known after 16:15). Both go through
``options.metrics.chain_metrics`` (IV from OTM mids on the parity forward), so differences are
model differences, not methodology. Also compared: the 30-DTE ATM put premium (mid / spot) and the
quoted spread of near-the-money (|delta| 0.3-0.7) options at 20-40 DTE.

    python -m atlasbt.options.validate_real            # prints the table, writes reference/databento_validation/summary.csv
"""
from __future__ import annotations

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


if __name__ == "__main__":  # pragma: no cover
    df = pd.concat([compare(), compare_spx()], ignore_index=True)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string())
    s = summarize(df)
    print(s.round(3).to_string())
    df.to_csv(D.validation_dir() / "detail.csv", index=False)
    s.to_csv(D.validation_dir() / "summary.csv", index=False)
