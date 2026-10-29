"""Rebuild ``spread_calibration.json`` from the per-symbol calibration datasets.

Inputs (all produced from real quotes, see docs/SPREADS.md section 2):
  nbbo_<YYYYMMDD>.csv.gz   per-symbol RTH time-weighted NBBO half spreads (NYSE Daily TAQ sample),
                           effective spreads vs the prevailing NBBO, 5-minute bucket profile,
                           joined with PIT features (prior 20 sessions) and security type.
  iex_<YYYYMMDD>.csv.gz    the same statistics computed on IEX TOPS (IEX top of book only)
                           for 12 sample days 2017-2026, joined with PIT features.

Usage: python -m atlasbt.execution.calibration.build_calibration [--fit-day 20260701] [--write]
Prints the fit and validation tables used in docs/SPREADS.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

from atlasbt.execution.spread_calibrated import design_terms

HERE = Path(__file__).parent
STOCK_TYPES = {"A", "H", "M", "P"}          # common, ADR, NY registry, closed-end fund
ETF_TYPES = {"ETF", "ETS", "ETV", "ETN"}
NB = 78

# Hendershott, Jones & Menkveld (2011, JF 66(1)) Table I: NYSE, Dec 2002 - Jul 2003, market-cap
# quintiles (Q1 largest): share-volume-weighted quoted half-spread (bp), effective half-spread (bp),
# average daily dollar volume ($M), Parkinson daily volatility (%), closing price ($).
HJM_2003 = {
    "qhalf_bp": [5.19, 6.82, 9.17, 11.68, 19.89],
    "ehalf_bp": [3.63, 4.79, 6.56, 8.46, 14.50],
    "dvol_musd": [94.71, 24.09, 10.12, 5.32, 2.17],
    "vol_pct": [1.47, 1.56, 1.63, 1.74, 2.06],
    "price": [40.01, 32.05, 25.86, 23.93, 16.41],
    "date": "2003-04-01",
}


def holdout(sym: str, frac: float = 0.3) -> bool:
    """Deterministic symbol hold-out (md5 of the ticker)."""
    return int(hashlib.md5(sym.encode()).hexdigest()[:8], 16) / 2 ** 32 < frac


def load(kind: str, day: str) -> pd.DataFrame:
    return pd.read_csv(HERE / f"{kind}_{day}.csv.gz", dtype={"symbol": str, "sec_type": str})


TERMS = ["1", "la", "ls", "lp", "e", "e_la", "e_ls", "e_lp", "hinge_lp_lo", "e_s", "e_s_ls", "e_s_lp", "e_s_la"]
KNOTS = {"adv_hi": 1e8, "price_lo": 5.0}
NT = len(TERMS)


CLIP: dict = {}


def design(df: pd.DataFrame, center: dict, terms=None) -> np.ndarray:
    """Same regressors as the module (inputs clipped to the fit-day range once CLIP is set)."""
    def c(v, k):
        v = np.log(np.maximum(v, 1e-12))
        return np.clip(v, *CLIP[k]) if k in CLIP else v
    la = c(df.adv_dollar.to_numpy(float), "log_adv") - center["log_adv"]
    ls = c(df.sigma.to_numpy(float), "log_sigma") - center["log_sigma"]
    lp = c(df.prev_close.to_numpy(float), "log_price") - center["log_price"]
    e = df.etf.to_numpy(float)
    return design_terms(terms or TERMS, la, ls, lp, e, math.log(KNOTS["adv_hi"]) - center["log_adv"],
                        math.log(KNOTS["price_lo"]) - center["log_price"])


def model_half(theta: np.ndarray, X: np.ndarray, price: np.ndarray, tick: np.ndarray,
               mult: float | np.ndarray = 1.0) -> np.ndarray:
    k = X.shape[1]
    cd = theta[k]
    q = math.exp(theta[k + 1])
    L = np.exp(X @ theta[:k]) * mult
    r = L * price / tick + cd
    S = tick * (1.0 + r ** q) ** (1.0 / q)
    return S / (2 * price)


def fit(df: pd.DataFrame, center: dict, y_col: str = "tw_hs_bp"):
    X = design(df, center)
    p = df.prev_close.to_numpy()
    tk = np.where(p >= 1, 0.01, 0.0001)
    y = np.log(df[y_col].to_numpy() / 1e4)
    th0 = np.zeros(NT + 2); th0[0] = math.log(5e-4); th0[NT] = 0.1; th0[NT + 1] = math.log(3.0)
    lb = np.full(NT + 2, -20.0); ub = np.full(NT + 2, 20.0)
    lb[NT], ub[NT] = 0.0, 2.0                        # c_d
    lb[NT + 1], ub[NT + 1] = 0.0, math.log(20.0)     # log q

    w = np.sqrt(liquidity_weight(df))

    def res(th):
        return w * (np.log(model_half(th, X, p, tk)) - y)
    r = least_squares(res, th0, loss="soft_l1", f_scale=0.5, bounds=(lb, ub))
    return r.x, r


def liquidity_weight(df: pd.DataFrame) -> np.ndarray:
    """Fit weight log10(ADV$) - 5 clipped to [0.5, 5]: tradable names count more than micro caps."""
    return np.clip(np.log10(df.adv_dollar.to_numpy(float)) - 5.0, 0.5, 5.0)


def stats(obs_bp: np.ndarray, pred_bp: np.ndarray) -> dict:
    lr = np.log(pred_bp / obs_bp)
    return {"n": int(len(obs_bp)), "mae_bp": float(np.mean(np.abs(pred_bp - obs_bp))),
            "median_ae_bp": float(np.median(np.abs(pred_bp - obs_bp))),
            "mean_log_ratio": float(np.mean(lr)), "sd_log_ratio": float(np.std(lr)),
            "median_abs_log_ratio": float(np.median(np.abs(lr))),
            "within_x2": float(np.mean(np.abs(lr) < math.log(2)))}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["etf"] = df.sec_type.isin(ETF_TYPES).astype(float)
    keep = df.sec_type.isin(STOCK_TYPES | ETF_TYPES)
    keep &= np.isfinite(df.tw_hs_bp) & (df.tw_hs_bp > 0)
    keep &= df.rth_valid_s > 0.8 * 23400
    keep &= (df.prev_close >= 0.1) & (df.adv_dollar > 0) & (df.sigma > 0)
    keep &= np.abs(np.log(df.tw_mid / df.prev_close)) < 0.3
    return df[keep].reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-day", default="20260701")
    ap.add_argument("--oos-days", default="")
    ap.add_argument("--iex-days", default="20170322,20180314,20190313,20200318,20200923,20210915,"
                                         "20220914,20230803,20240911,20250910,20260115,20260701")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    report: dict = {}

    nb = prepare(load("nbbo", a.fit_day))
    center = {"log_adv": float(np.median(np.log(nb.adv_dollar))),
              "log_sigma": float(np.median(np.log(nb.sigma))),
              "log_price": float(np.median(np.log(nb.prev_close)))}
    for k, col in (("log_adv", "adv_dollar"), ("log_sigma", "sigma"), ("log_price", "prev_close")):
        v = np.log(nb[col].to_numpy(float))
        CLIP[k] = (float(np.quantile(v, 0.001)) if k != "log_adv" else float(v.min()),
                   float(np.quantile(v, 0.999)) if k != "log_adv" else float(v.max()))
    nb["hold"] = nb.symbol.map(holdout)
    tr, te = nb[~nb.hold], nb[nb.hold]
    th, _ = fit(tr, center)
    print("theta", dict(zip(TERMS + ["c_d", "log_q"], np.round(th, 4))))

    def pred(df, theta=th, mult=1.0):
        p = df.prev_close.to_numpy()
        return 1e4 * model_half(theta, design(df, center), p, np.where(p >= 1, 0.01, 0.0001), mult)

    te = te.copy(); te["pred"] = pred(te); te["cs_half_bp"] = 0.5e4 * te.cs_full
    tr = tr.copy(); tr["pred"] = pred(tr)
    report["in_sample"] = stats(tr.tw_hs_bp.to_numpy(), tr.pred.to_numpy())
    report["holdout_symbols"] = stats(te.tw_hs_bp.to_numpy(), te.pred.to_numpy())
    grp = {
        "ETF": te.etf == 1, "stock": te.etf == 0,
        "ADV>=$1B": te.adv_dollar >= 1e9, "ADV $100M-1B": (te.adv_dollar >= 1e8) & (te.adv_dollar < 1e9),
        "ADV $10M-100M": (te.adv_dollar >= 1e7) & (te.adv_dollar < 1e8),
        "ADV $1M-10M": (te.adv_dollar >= 1e6) & (te.adv_dollar < 1e7), "ADV <$1M": te.adv_dollar < 1e6,
        "price<$5": te.prev_close < 5, "price $5-20": (te.prev_close >= 5) & (te.prev_close < 20),
        "price>=$20": te.prev_close >= 20,
    }
    report["holdout_groups"] = {k: stats(te.tw_hs_bp[m].to_numpy(), te.pred[m].to_numpy()) for k, m in grp.items() if m.sum() > 5}
    csok = np.isfinite(te.cs_half_bp) & (te.cs_half_bp > 0)
    report["holdout_cs"] = stats(te.tw_hs_bp[csok].to_numpy(), te.cs_half_bp[csok].to_numpy())
    report["holdout_model_same_rows"] = stats(te.tw_hs_bp[csok].to_numpy(), te.pred[csok].to_numpy())
    report["holdout_cs_groups"] = {k: stats(te.tw_hs_bp[m & csok].to_numpy(), te.cs_half_bp[m & csok].to_numpy())
                                   for k, m in grp.items() if (m & csok).sum() > 5}

    # refit on all symbols of the fit day (final coefficients)
    th_all, _ = fit(nb, center)
    nb["pred"] = pred(nb, th_all)
    nb["cs_half_bp"] = 0.5e4 * nb.cs_full
    named = ["SPY", "QQQ", "IWM", "DIA", "EEM", "TLT", "GLD", "XLF", "AAPL", "MSFT", "NVDA", "AMZN", "JPM",
             "BAC", "F", "T", "INTC", "AA"]
    report["named"] = nb[nb.symbol.isin(named)][["symbol", "prev_close", "adv_dollar", "sigma", "tw_hs_bp",
                                                  "tw_med_hs_bp", "pred", "cs_half_bp"]].round(4).to_dict("records")

    # leave-named-out: refit without these symbols and predict them (SPY/QQQ/IWM are training symbols in the hash split)
    loo_syms = ["SPY", "QQQ", "IWM", "DIA", "IVV", "VOO", "TLT", "GLD", "XLF", "AAPL", "MSFT", "NVDA", "AMZN", "JPM"]
    th_loo, _ = fit(nb[~nb.symbol.isin(loo_syms)], center)
    lo = nb[nb.symbol.isin(loo_syms)].copy(); lo["pred_loo"] = pred(lo, th_loo)
    report["leave_named_out"] = lo[["symbol", "tw_hs_bp", "pred_loo", "cs_half_bp"]].round(4).to_dict("records")

    # other NBBO days: fully out of sample (day and coefficients)
    report["oos_days"] = {}
    for d in [x for x in a.oos_days.split(",") if x]:
        o = prepare(load("nbbo", d)); o["pred"] = pred(o, th_all); o["cs_half_bp"] = 0.5e4 * o.cs_full
        ok = np.isfinite(o.cs_half_bp) & (o.cs_half_bp > 0)
        report["oos_days"][d] = {"model": stats(o.tw_hs_bp.to_numpy(), o.pred.to_numpy()),
                                 "cs": stats(o.tw_hs_bp[ok].to_numpy(), o.cs_half_bp[ok].to_numpy()),
                                 "holdout_symbols_model": stats(o.tw_hs_bp[o.symbol.map(holdout)].to_numpy(),
                                                                o.pred[o.symbol.map(holdout)].to_numpy())}

    # effective / quoted (lit venues, dollar-volume weighted, vs prevailing NBBO mid)
    e = nb[np.isfinite(nb.lit_eff_hs_bp_dvw) & (nb.lit_eff_hs_bp_dvw > 0) & (nb.tr_n >= 50)]
    lr = np.log(e.lit_eff_hs_bp_dvw / e.tw_hs_bp)
    la = np.log(e.adv_dollar) - center["log_adv"]
    B = np.polyfit(la, lr, 1)
    eff = {"a": float(B[1]), "b": float(B[0]), "center": center["log_adv"], "min": 0.3, "max": 1.0,
           "median_ratio": float(np.exp(np.median(lr))), "n": int(len(e))}
    print("effective/quoted", eff)

    # time-of-day profile (5-minute buckets; ratio to the symbol's RTH average)
    bk = nb[[f"b{k:02d}" for k in range(NB)]].to_numpy(float)
    full = np.isfinite(bk).all(axis=1) & (nb.tr_n >= 200).to_numpy()
    rel = bk[full] / nb.tw_hs_bp.to_numpy()[full, None]
    split = 1e8
    liq = nb.adv_dollar.to_numpy()[full] >= split
    prof_l = np.median(rel[liq], axis=0); prof_i = np.median(rel[~liq], axis=0)
    prof_l /= prof_l.mean(); prof_i /= prof_i.mean()
    tod = {"bucket_minutes": 5.0, "adv_split": split, "liquid": [round(float(x), 4) for x in prof_l],
           "illiquid": [round(float(x), 4) for x in prof_i], "n_liquid": int(liq.sum()), "n_illiquid": int((~liq).sum())}

    # IEX vs NBBO on the overlap day (IEX statistic: time-weighted MEDIAN of the IEX BBO half spread)
    iex_ov = prepare_iex(load("iex", a.fit_day))
    m = iex_ov.merge(nb[["symbol", "tw_hs_bp", "pred"]].rename(columns={"tw_hs_bp": "nbbo_bp"}), on="symbol")
    m = m[m.valid_frac > 0.9]
    m["lr"] = np.log(m.iex_med_bp / m.nbbo_bp)
    lam = np.log(m.adv_dollar) - center["log_adv"]
    Bi = np.polyfit(lam, m.lr, 1)
    report["iex_vs_nbbo"] = {"n": int(len(m)), "slope": float(Bi[0]), "intercept": float(Bi[1]),
                             "median_ratio_adv_ge_1B": float(np.exp(np.median(m.lr[m.adv_dollar >= 1e9]))),
                             "median_ratio_adv_100M_1B": float(np.exp(np.median(m.lr[(m.adv_dollar >= 1e8) & (m.adv_dollar < 1e9)]))),
                             "median_ratio_adv_10M_100M": float(np.exp(np.median(m.lr[(m.adv_dollar >= 1e7) & (m.adv_dollar < 1e8)]))),
                             "median_ratio_adv_lt_10M": float(np.exp(np.median(m.lr[m.adv_dollar < 1e7])))}

    # HJM 2003 anchor: latent multiplier exp(a + b * (log ADV$ - centre)) fitted to the 5 quintiles
    h = HJM_2003
    hx = pd.DataFrame({"adv_dollar": np.array(h["dvol_musd"]) * 1e6, "sigma": np.array(h["vol_pct"]) / 100,
                       "prev_close": h["price"], "etf": 0.0})
    hp = hx.prev_close.to_numpy(); HX = design(hx, center); htk = np.full(5, 0.01)
    hy = np.log(np.array(h["qhalf_bp"]) / 1e4)
    hla = HX[:, 1]
    r03 = least_squares(lambda ab: np.log(model_half(th_all, HX, hp, htk, np.exp(ab[0] + ab[1] * hla))) - hy,
                        np.array([0.0, 0.0]))
    a03, b03 = (float(v) for v in r03.x)
    report["hjm_2003"] = {"a": a03, "b": b03,
                          "pred_bp_2026_coefficients": [round(float(v), 3) for v in 1e4 * model_half(th_all, HX, hp, htk)],
                          "pred_bp_with_era": [round(float(v), 3) for v in 1e4 * model_half(th_all, HX, hp, htk, np.exp(a03 + b03 * hla))],
                          "obs_bp": h["qhalf_bp"]}
    print("HJM 2003", report["hjm_2003"])

    # era anchors [decimal year, a, b]: 2003 literature anchor held back to 1990 (pre-2001 differences come
    # from the tick regime), linear transition to the 2026 NBBO fit by 2010.0 (assumption), 2026 fit after.
    # 2001 level: Furfine (2003) post-decimal mean quoted spread of the most active NYSE stocks (7.0 cents),
    # inputs for 8 such names from Yahoo daily data (historical_checks.csv); the 2003 slope is kept.
    hc = pd.read_csv(HERE / "historical_checks.csv"); hc["etf"] = hc.etf.astype(float)
    fpost = hc[hc.check == "furfine_2001_post_decimal"]
    FX = design(fpost, center); fp = fpost.prev_close.to_numpy(); ftk = fpost.tick.to_numpy()

    def gf(d01):
        cents = 1e4 * model_half(th_all, FX, fp, ftk, np.exp(a03 + d01 + b03 * FX[:, 1])) / 1e4 * 2 * fp * 100
        return float(np.mean(cents)) - float(fpost.obs_value.iloc[0])
    d01 = brentq(gf, -3, 3)
    a01 = a03 + d01
    report["furfine_2001_level"] = {"a": a01, "delta_vs_2003": d01}
    anchors = [[1990.0, round(a01, 5), round(b03, 5)], [2001.08, round(a01, 5), round(b03, 5)],
               [2003.25, round(a03, 5), round(b03, 5)], [2010.0, 0.0, 0.0], [2100.0, 0.0, 0.0]]
    report["era_anchors"] = anchors

    def era_ab(dates):
        yf = np.array([t.year + (t.dayofyear - 1) / 365.25 for t in pd.to_datetime(dates)])
        ts = [x[0] for x in anchors]
        return np.interp(yf, ts, [x[1] for x in anchors]), np.interp(yf, ts, [x[2] for x in anchors])

    # historical checks: Lean SPY NBBO 2013/2023 and Furfine pre-decimal (1/16 tick) are out of sample;
    # the Furfine post-decimal group mean is matched by construction (2001 level above)
    ea, eb = era_ab(hc.date)
    HXc = design(hc, center)
    hc["pred_half_bp"] = 1e4 * model_half(th_all, HXc, hc.prev_close.to_numpy(), hc.tick.to_numpy(),
                                          np.exp(ea + eb * HXc[:, 1]))
    hc["pred_full_cents"] = hc.pred_half_bp / 1e4 * 2 * hc.prev_close * 100
    hc["cs_half_bp_na"] = np.nan
    chk = {}
    for name, g in hc.groupby("check"):
        chk[name] = {"pred_half_bp_mean": float(g.pred_half_bp.mean()), "pred_full_cents_mean": float(g.pred_full_cents.mean()),
                     "obs": float(g.obs_value.iloc[0]), "obs_unit": g.obs_unit.iloc[0], "source": g.source.iloc[0]}
    report["historical_checks"] = chk
    print("historical", json.dumps(chk, indent=1))

    # IEX diagnostics: IEX BBO median vs the NBBO model on liquid names (IEX is one venue => ratio >= 1;
    # compare with the ratio measured on the NBBO overlap day). Not used in the fit.
    diag = {}
    for d in [x for x in a.iex_days.split(",") if x]:
        try:
            x = prepare_iex(load("iex", d))
        except FileNotFoundError:
            continue
        x = x[(x.valid_frac > 0.9) & (x.adv_dollar >= 1e9)]
        if len(x) < 10:
            continue
        ea, eb = era_ab([d] * len(x)); X = design(x, center); p = x.prev_close.to_numpy()
        pr = 1e4 * model_half(th_all, X, p, np.where(p >= 1, 0.01, 0.0001), np.exp(ea + eb * X[:, 1]))
        spy = x.symbol.to_numpy() == "SPY"
        diag[d] = {"n_adv_ge_1B": int(len(x)), "median_iex_over_model": float(np.median(x.iex_med_bp.to_numpy() / pr)),
                   "spy_iex_med_bp": float(x.iex_med_bp[spy].iloc[0]) if spy.any() else None,
                   "spy_model_bp": float(pr[spy][0]) if spy.any() else None}
    report["iex_diagnostics"] = diag
    print("iex diag", json.dumps(diag, indent=1))

    lr_all = np.log(nb.pred / nb.tw_hs_bp)
    cal = {
        "version": 1,
        "description": "Calibrated half-spread model, see docs/SPREADS.md",
        "fit_day": a.fit_day, "calib_years": ["2010-01-01", str(pd.Timestamp(a.fit_day).date())],
        "target": "RTH time-weighted mean NBBO quoted half-spread (fraction of mid)",
        "terms": TERMS, "knots": KNOTS,
        "beta": dict(zip(TERMS, [round(float(v), 6) for v in th_all[:NT]])),
        "c_d": round(float(th_all[NT]), 6), "q": round(float(math.exp(th_all[NT + 1])), 6),
        "center": {k: round(v, 6) for k, v in center.items()},
        "clip_log": {k: [round(v[0], 6), round(v[1], 6)] for k, v in CLIP.items()},
        "fit_weight": "log10(ADV$) - 5 clipped to [0.5, 5]",
        "calib_range": {"adv_min": float(np.quantile(nb.adv_dollar, 0.02)),
                        "adv_max": float(nb.adv_dollar.max()),
                        "price_min": float(nb.prev_close.min()), "sigma_q01": float(np.quantile(nb.sigma, 0.01)),
                        "sigma_q99": float(np.quantile(nb.sigma, 0.99))},
        "residual_sd_log": float(np.std(lr_all)),
        "era_anchors": anchors,
        "tod_profile": tod,
        "effective_ratio": eff,
        "n_symbols": int(len(nb)),
    }
    print(json.dumps({k: v for k, v in report.items() if k not in ("named",)}, indent=1, default=float)[:20000])
    print(pd.DataFrame(report["named"]).to_string())
    if a.write:
        (HERE / "spread_calibration.json").write_text(json.dumps(cal, indent=1))
        (HERE / "validation_report.json").write_text(json.dumps(report, indent=1, default=float))
    return cal, report


def prepare_iex(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["etf"] = df.sec_type.isin(ETF_TYPES).astype(float) if "sec_type" in df else 0.0
    df["valid_frac"] = df.rth_valid_s / 23400.0
    df["iex_med_bp"] = df.tw_med_hs_bp
    keep = np.isfinite(df.iex_med_bp) & (df.iex_med_bp > 0) & (df.prev_close >= 1) & (df.adv_dollar > 0) & (df.sigma > 0)
    keep &= np.abs(np.log(df.tw_mid / df.prev_close)) < 0.3
    if "sec_type" in df:
        keep &= df.sec_type.isin(STOCK_TYPES | ETF_TYPES)
    return df[keep].reset_index(drop=True)


if __name__ == "__main__":
    main()
