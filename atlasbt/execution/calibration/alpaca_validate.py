"""Validate the calibrated spread model against real SIP NBBO 2016-2026 (Alpaca sample; docs/SPREADS.md §10).

    python -m atlasbt.execution.calibration.alpaca_validate [--calibration PATH] [--json OUT]

Input: ``alpaca_nbbo_2016_2026.csv.gz`` (built by tools/alpaca_nbbo_sample.py): per (session, 1-minute window,
symbol) the time-weighted mean relative NBBO half spread and the PIT features of the calibration (prior close,
20-day ADV$, 20-day sigma). Prediction = what the engine charges: max(model half spread (quoted) x time-of-day
multiplier of the window, half a tick at the prior close) (``full_spread``).
Errors are reported in bp of half spread and as log ratios ln(pred / obs), by year, ADV$ bucket, asset type and
window. ``level`` rows compare the symbol-day mean over the windows with the model's mean over the same windows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..spread_calibrated import CalibratedSpreadConfig, CalibratedSpreadModel

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "alpaca_nbbo_2016_2026.csv.gz"
ADV_BUCKETS = [(0, 1e8, "<$100M"), (1e8, 5e8, "$100M-500M"), (5e8, 2e9, "$500M-2B"), (2e9, 1e10, "$2B-10B"),
               (1e10, np.inf, ">=$10B")]


def load_sample(path: Optional[Path] = None, min_quotes: int = 3) -> pd.DataFrame:
    df = pd.read_csv(path or SAMPLE)
    df = df[np.isfinite(df["tw_hs_bp"]) & (df["n_quotes"] >= min_quotes) & (df["tw_hs_bp"] > 0)
            & (df["adv_dollar"] > 0) & (df["prev_close"] > 0)].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def predict(df: pd.DataFrame, model: Optional[CalibratedSpreadModel] = None) -> pd.DataFrame:
    """Adds pred_bp (model half spread x ToD multiplier, bp), base_bp (daily level), code."""
    m = model or CalibratedSpreadModel()
    out = df.copy()
    h, code = m.half_spread_vec(out["date"].to_numpy(), out["prev_close"].to_numpy(), out["adv_dollar"].to_numpy(),
                                out["sigma"].to_numpy(), out["etf"].to_numpy(float),
                                n_sigma=out["n_sigma"].to_numpy(float) if "n_sigma" in out else None,
                                exchange="NYSE")
    tod = np.array([m.tod_multiplier(a, b, adv, "ETF" if e else "STK")
                    for a, b, adv, e in zip(out["mso"], out["mtc"], out["adv_dollar"], out["etf"])])
    px = out["prev_close"].to_numpy(float)
    half_tick = np.where(px >= 1.0, 0.01, 0.0001) / (2.0 * px)     # engine full_spread: >= 1 tick at the fill
    out["base_bp"] = h * 1e4
    out["tod"] = tod
    out["pred_bp"] = np.maximum(h * tod, half_tick) * 1e4
    out["code"] = code
    return out


def _stats(obs: np.ndarray, pred: np.ndarray) -> dict:
    lr = np.log(pred / obs)
    return {"n": int(len(obs)), "mae_bp": round(float(np.mean(np.abs(pred - obs))), 3),
            "median_ae_bp": round(float(np.median(np.abs(pred - obs))), 3),
            "obs_median_bp": round(float(np.median(obs)), 3), "pred_median_bp": round(float(np.median(pred)), 3),
            "mean_log_ratio": round(float(np.mean(lr)), 3), "sd_log_ratio": round(float(np.std(lr)), 3),
            "median_log_ratio": round(float(np.median(lr)), 3),
            "within_x2": round(float(np.mean(np.abs(lr) <= np.log(2))), 3),
            "share_pred_below_obs": round(float(np.mean(pred < obs)), 3)}


def bucket(adv) -> np.ndarray:
    a = np.asarray(adv, float)
    out = np.empty(len(a), object)
    for lo, hi, name in ADV_BUCKETS:
        out[(a >= lo) & (a < hi)] = name
    return out


def report(df: pd.DataFrame) -> dict:
    """Error tables: all, by year, by ADV bucket x type, by window; plus the day-level comparison."""
    d = df.copy()
    d["year"] = d["date"].dt.year
    d["kind"] = np.where(d["etf"] > 0, "ETF", "stock")
    d["adv_bucket"] = bucket(d["adv_dollar"])
    d["window"] = d["mso"].round().astype(int).map(lambda x: f"open+{x}m")
    obs, pred = d["tw_hs_bp"].to_numpy(), d["pred_bp"].to_numpy()
    rep: dict = {"all": _stats(obs, pred)}
    for key in ("year", "kind", "adv_bucket", "window"):
        rep[f"by_{key}"] = {str(k): _stats(g["tw_hs_bp"].to_numpy(), g["pred_bp"].to_numpy())
                            for k, g in d.groupby(key, sort=True)}
    rep["by_kind_bucket"] = {f"{k[0]} {k[1]}": _stats(g["tw_hs_bp"].to_numpy(), g["pred_bp"].to_numpy())
                             for k, g in d.groupby(["kind", "adv_bucket"], sort=True) if len(g) >= 20}
    rep["by_year_kind"] = {f"{k[0]} {k[1]}": _stats(g["tw_hs_bp"].to_numpy(), g["pred_bp"].to_numpy())
                           for k, g in d.groupby(["year", "kind"], sort=True)}
    lvl = d.groupby(["date", "symbol"]).agg(obs=("tw_hs_bp", "mean"), pred=("pred_bp", "mean"), year=("year", "first"),
                                            kind=("kind", "first"))
    rep["level_all"] = _stats(lvl["obs"].to_numpy(), lvl["pred"].to_numpy())
    rep["level_by_year"] = {str(k): _stats(g["obs"].to_numpy(), g["pred"].to_numpy()) for k, g in lvl.groupby("year")}
    # observed vs modelled time-of-day shape (ratio of each window to the symbol-day mean)
    d["obs_rel"] = d["tw_hs_bp"] / d.groupby(["date", "symbol"])["tw_hs_bp"].transform("mean")
    d["mod_rel"] = d["tod"] / d.groupby(["date", "symbol"])["tod"].transform("mean")
    rep["tod_shape"] = {f"{k[0]} {k[1]}": {"obs_median": round(float(g["obs_rel"].median()), 3),
                                           "model": round(float(g["mod_rel"].median()), 3), "n": int(len(g))}
                        for k, g in d.groupby(["kind", "window"], sort=True)}
    return rep


EVIDENCE_START, EVIDENCE_END = "2016-01-04", "2025-12-31"
EVIDENCE_QUANTILE = 0.40     # pessimistic: per year/kind, raise the relative model until >= 60% of symbol-days
                             # are at or above the observed NBBO half spread (multiplier never below 1)


def fit_evidence(df: pd.DataFrame, quantile: float = EVIDENCE_QUANTILE) -> dict:
    """v4 'nbbo_evidence' block from a sample: per (year, ETF/stock) multiplier on the RELATIVE model (v3 without
    the 2010-2025 floor) = exp(max(0, -q)) with q the `quantile` of ln(pred / obs) over symbol-days (mean over the
    windows); ADV$ band = [2nd, 10th] percentile of the sample stocks' nominal ADV$ (below: the floor is kept)."""
    rel = CalibratedSpreadModel(CalibratedSpreadConfig(pessimistic_floor_2010_2025=False, nbbo_evidence=False))
    d = predict(df[(df["date"] >= EVIDENCE_START) & (df["date"] <= EVIDENCE_END)], rel)
    d["year"] = d["date"].dt.year
    lvl = d.groupby(["date", "symbol"]).agg(obs=("tw_hs_bp", "mean"), pred=("pred_bp", "mean"),
                                            year=("year", "first"), etf=("etf", "first"))
    lvl["lr"] = np.log(lvl["pred"] / lvl["obs"])
    mult: dict = {}
    for y in range(pd.Timestamp(EVIDENCE_START).year, pd.Timestamp(EVIDENCE_END).year + 1):
        mult[str(y)] = {}
        for kind, e in (("stock", 0), ("etf", 1)):
            g = lvl[(lvl["year"] == y) & (lvl["etf"] == e)]["lr"]
            mult[str(y)][kind] = float(np.exp(max(0.0, -float(np.quantile(g, quantile))))) if len(g) >= 20 else None
    for kind in ("stock", "etf"):            # a year without evidence: the largest multiplier of any year
        mx = max([v[kind] for v in mult.values() if v[kind] is not None] or [1.0])
        for v in mult.values():
            if v[kind] is None:
                v[kind] = mx
    stk = d.loc[d["etf"] == 0, "adv_dollar"]
    return {"start": EVIDENCE_START, "end": EVIDENCE_END,
            "adv_band": [float(np.quantile(stk, 0.02)), float(np.quantile(stk, 0.10))],
            "multipliers": mult, "quantile": quantile,
            "source": "Alpaca SIP NBBO sample (calibration/alpaca_nbbo_2016_2026.csv.gz): 2nd Wednesday of each "
                      "month, 7 one-minute windows, 19 ETFs + 50 PIT S&P 500 members spread over ADV$ ranks",
            "n_symbol_days": int(len(lvl)), "n_rows": int(len(d))}


def write_v4(evidence: dict, base_path: Optional[Path] = None, out_path: Optional[Path] = None) -> Path:
    """spread_calibration.json v4 = the v3 fit + 'nbbo_evidence' (the v3 file is kept as spread_calibration_v3.json)."""
    base_path = base_path or HERE / "spread_calibration_v3.json"
    out_path = out_path or HERE / "spread_calibration.json"
    cal = json.loads(Path(base_path).read_text())
    cal["version"] = 4
    cal["nbbo_evidence"] = evidence
    cal["description"] = str(cal.get("description", "")) + (
        " | v4 (2026-10-30): 2016-2025 inside the evidence ADV$ band uses the relative model x per-year SIP NBBO "
        "evidence multipliers (Alpaca sample) instead of the nominal-2026 pessimistic floor; floor kept for "
        "2010-2015 and below the band.")
    Path(out_path).write_text(json.dumps(cal, indent=1))
    return Path(out_path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--sample", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    m = CalibratedSpreadModel(CalibratedSpreadConfig(calibration_path=a.calibration))
    rep = report(predict(load_sample(Path(a.sample) if a.sample else None), m))
    rep["calibration"] = a.calibration or "default"
    s = json.dumps(rep, indent=1)
    if a.json:
        Path(a.json).write_text(s)
    print(s)


if __name__ == "__main__":
    main()


def window_sample_from_store(symbols=None, store=None, portal=None) -> pd.DataFrame:
    """Same statistics as the sampling tool, computed from the decision-window quotes in the store
    (quotes/alpaca + reference/alpaca/quote_windows): every recorded RTH window of every symbol, with the
    PIT features of that session. Used for the dense ETF validation (every session 2016->)."""
    from atlasbt.data import frames
    from atlasbt.data.portal import Portal
    from atlasbt.data.store import default_store
    store = store or default_store()
    portal = portal or Portal(store)
    w = store.read("reference", "alpaca", "quote_windows")
    if not len(w):
        return pd.DataFrame()
    w = w[w["window"] != "pre_0920"]
    if symbols:
        w = w[w["symbol"].isin(symbols)]
    sch = frames.schedule()
    rows = []
    for sym, g in w.groupby("symbol"):
        q = store.read("quotes", "alpaca", sym)
        if not len(q):
            continue
        t = q.index.asi8
        hs = ((q["ask"] - q["bid"]) / (q["ask"] + q["bid"])).to_numpy() * 1e4
        d = portal._daily_frame(sym, pd.Timestamp.now().normalize(), start=pd.Timestamp("2015-10-01"))
        if not len(d):
            continue
        dd = frames.to_naive_date(d.index)
        c, v = d["close"].to_numpy(float), d["volume"].to_numpy(float)
        lr = np.r_[np.nan, np.diff(np.log(c))]
        lr[np.abs(lr) > 0.4] = np.nan
        f = pd.DataFrame({"prev_close": c, "dv": c * v, "lr": lr}, index=dd)
        f["adv_dollar"] = f["dv"].rolling(20, min_periods=1).mean().shift(1)
        f["sigma"] = f["lr"].rolling(20, min_periods=2).std().shift(1)
        f["n_sigma"] = f["lr"].rolling(20, min_periods=1).count().shift(1)
        f["prev_close"] = f["prev_close"].shift(1)
        w0 = pd.DatetimeIndex(g["w0"]).tz_convert("UTC")
        w1 = pd.DatetimeIndex(g["w1"]).tz_convert("UTC")
        for a, b, name in zip(w0, w1, g["window"]):
            i, j = np.searchsorted(t, a.value, "left"), np.searchsorted(t, b.value, "left")
            if j - i < 1:
                continue
            dt = np.diff(np.r_[t[i:j], b.value]).astype(float)
            h = hs[i:j]
            ok = np.isfinite(h) & (h < 5000)
            if dt[ok].sum() <= 0:
                continue
            day = a.tz_convert("America/New_York").tz_localize(None).normalize()
            if day not in f.index:
                continue
            o, cl = sch.loc[day, ["open", "close"]]
            r = f.loc[day]
            rows.append({"date": day, "mso": (a - pd.Timestamp(o)).total_seconds() / 60, "mtc": (pd.Timestamp(cl) - a).total_seconds() / 60,
                         "symbol": sym, "etf": int(portal._is_etf(sym) or portal._is_plain(sym)), "window": name,
                         "prev_close": r["prev_close"], "adv_dollar": r["adv_dollar"], "sigma": r["sigma"],
                         "n_sigma": r["n_sigma"], "tw_hs_bp": float(np.sum(h[ok] * dt[ok]) / dt[ok].sum()),
                         "n_quotes": int(j - i)})
    out = pd.DataFrame(rows)
    return out[np.isfinite(out["prev_close"]) & (out["adv_dollar"] > 0)] if len(out) else out
