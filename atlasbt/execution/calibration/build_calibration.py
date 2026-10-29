"""Rebuild ``spread_calibration.json`` from the per-symbol calibration datasets.

Inputs (all produced from real quotes or published tables, see docs/SPREADS.md section 2):
  nbbo_<YYYYMMDD>.csv.gz   per-symbol RTH time-weighted NBBO half spreads (NYSE Daily TAQ sample),
                           effective spreads vs the prevailing NBBO, 5-minute bucket profile,
                           joined with PIT features (prior 20 sessions) and security type.
  iex_<YYYYMMDD>.csv.gz    the same statistics computed on IEX TOPS (IEX top of book only)
                           for 12 sample days 2017-2026, joined with PIT features (diagnostics only).
  historical_checks.csv    Lean SPY NBBO 2013/2023 and Furfine (2003) 2001 NYSE names (as-traded inputs).
  hasbrouck_xs.csv.gz      PIT features of the Yahoo/qlib US common-stock cross-section 2000-2005
                           (survivor-biased), used to match Hasbrouck (2009) annual effective costs.

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

from atlasbt.execution.spread_calibrated import CalibratedSpreadConfig, CalibratedSpreadModel, design_terms

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


TERMS = ["1", "la", "ls", "lp", "e", "e_la", "e_ls", "e_lp", "hinge_lp_lo", "e_s", "e_s_ls", "e_s_lp", "e_s_la",
         "e_hinge_vhi"]
KNOTS = {"adv_hi": 1e8, "price_lo": 5.0, "adv_vhi": 5e9}
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
                        math.log(KNOTS["price_lo"]) - center["log_price"],
                        k_vhi=math.log(KNOTS["adv_vhi"]) - center["log_adv"])


def model_half(theta: np.ndarray, X: np.ndarray, price: np.ndarray, tick: np.ndarray,
               mult: float | np.ndarray = 1.0) -> np.ndarray:
    k = X.shape[1]
    cd = theta[k]
    q = math.exp(theta[k + 1])
    L = np.exp(X @ theta[:k]) * mult
    r = L * price / tick + cd
    S = tick * (1.0 + r ** q) ** (1.0 / q)
    return S / (2 * price)




I_ELS, I_ESLS = None, None   # set below


def fit(df: pd.DataFrame, center: dict, y_col: str = "tw_hs_bp"):
    """Weighted robust NLS. The liquid-ETF sigma slope is bounded by the stock slope:
    e_s_ls = u - e_ls with u <= 0, so ls + e_ls + s*e_s_ls <= ls for s in [0, 1] (e_ls <= 0)."""
    X = design(df, center)
    p = df.prev_close.to_numpy()
    tk = np.where(p >= 1, 0.01, 0.0001)
    y = np.log(df[y_col].to_numpy() / 1e4)
    i_els, i_esls = TERMS.index("e_ls"), TERMS.index("e_s_ls")
    th0 = np.zeros(NT + 2); th0[0] = math.log(5e-4); th0[NT] = 0.1; th0[NT + 1] = math.log(3.0)
    th0[i_esls] = -0.1
    lb = np.full(NT + 2, -20.0); ub = np.full(NT + 2, 20.0)
    lb[NT], ub[NT] = 0.0, 2.0                        # c_d
    lb[NT + 1], ub[NT + 1] = 0.0, math.log(20.0)     # log q
    ub[i_els] = 0.0                                   # ETF sigma slope <= stock slope
    ub[i_esls] = 0.0                                  # u <= 0
    ub[TERMS.index("e_hinge_vhi")] = 0.0              # more liquid -> never wider
    w = np.sqrt(liquidity_weight(df))

    def unpack(th):
        t = th.copy(); t[i_esls] = th[i_esls] - th[i_els]
        return t

    def res(th):
        return w * (np.log(model_half(unpack(th), X, p, tk)) - y)
    r = least_squares(res, th0, loss="soft_l1", f_scale=0.5, bounds=(lb, ub))
    return unpack(r.x), r


TOD_SPLIT = 1e8
NOCAP = CalibratedSpreadConfig(max_half_spread=0.5)
HASBROUCK = {"median": (1993.0, 0.010, 2005.0, 0.004), "p95": (1993.0, 0.05, 2005.0, 0.02),
             "eff_over_quoted": 0.72}   # Hasbrouck (2009) p.1457 text on Fig. 2; HJM 2003 eff/quoted 0.70-0.73


def hasbrouck_target(yf: float, which: str) -> float:
    t0, v0, t1, v1 = HASBROUCK[which]
    return v0 + (v1 - v0) * (min(max(yf, t0), t1) - t0) / (t1 - t0)


def yearfrac(d) -> float:
    d = pd.Timestamp(d)
    return d.year + (d.dayofyear - 1) / 365.25


def make_cal(theta, center, clip, era, tod, eff, sigma_prior, calib_range, fit_day) -> dict:
    return {"terms": TERMS, "knots": KNOTS, "beta": dict(zip(TERMS, [float(v) for v in theta[:NT]])),
            "c_d": float(theta[NT]), "q": float(math.exp(theta[NT + 1])), "center": center,
            "clip_log": {k: [float(v[0]), float(v[1])] for k, v in clip.items()}, "era": era,
            "tod_profile": tod, "effective_ratio": eff, "sigma_prior": sigma_prior,
            "calib_range": calib_range, "calib_years": ["2010-01-01", str(pd.Timestamp(fit_day).date())]}


def model_pred_bp(cal: dict, df: pd.DataFrame, date, cfg=None, tick=None) -> np.ndarray:
    m = CalibratedSpreadModel(cfg or CalibratedSpreadConfig(), calibration=cal)
    d = np.asarray(pd.to_datetime(df["date"]) if (date is None) else np.full(len(df), np.datetime64(pd.Timestamp(date))),
                   dtype="datetime64[ns]")
    h, _ = m.half_spread_vec(d, df.prev_close.to_numpy(float), df.adv_dollar.to_numpy(float),
                             df.sigma.to_numpy(float), df.etf.to_numpy(float),
                             tick=None if tick is None else np.asarray(tick, float))
    return 1e4 * h


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-day", default="20260701")
    ap.add_argument("--oos-days", default="")
    ap.add_argument("--iex-days", default="20170322,20180314,20190313,20200318,20200923,20210915,"
                                         "20220914,20230803,20240911,20250910,20260115,20260701")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    report: dict = {}
    fit_day = a.fit_day

    nb = prepare(load("nbbo", fit_day))
    center = {"log_adv": float(np.median(np.log(nb.adv_dollar))),
              "log_sigma": float(np.median(np.log(nb.sigma))),
              "log_price": float(np.median(np.log(nb.prev_close)))}
    CLIP.clear()
    for k, col in (("log_adv", "adv_dollar"), ("log_sigma", "sigma"), ("log_price", "prev_close")):
        v = np.log(nb[col].to_numpy(float))
        CLIP[k] = (float(np.quantile(v, 0.001)) if k != "log_adv" else float(v.min()),
                   float(np.quantile(v, 0.999)) if k != "log_adv" else float(v.max()))
    calib_range = {"adv_min": float(np.quantile(nb.adv_dollar, 0.02)), "adv_max": float(nb.adv_dollar.max()),
                   "price_min": float(nb.prev_close.min()), "sigma_q01": float(np.quantile(nb.sigma, 0.01)),
                   "sigma_q99": float(np.quantile(nb.sigma, 0.99))}

    # ---- sigma prior by price bucket (stocks), effective/quoted, time of day
    stk = nb[nb.etf == 0]
    ubs = [1.0, 5.0, 10.0, 20.0, 50.0, 1e12]
    sp, lo_ = [], 0.0
    for ub in ubs:
        mm = (stk.prev_close >= lo_) & (stk.prev_close < ub)
        sp.append([ub if ub < 1e12 else 1e12, float(np.median(stk.sigma[mm]))]); lo_ = ub
    sigma_prior = [[float(u), round(v, 6)] for u, v in sp]
    e = nb[np.isfinite(nb.lit_eff_hs_bp_dvw) & (nb.lit_eff_hs_bp_dvw > 0) & (nb.tr_n >= 50)]
    lr = np.log(e.lit_eff_hs_bp_dvw / e.tw_hs_bp)
    B = np.polyfit(np.log(e.adv_dollar) - center["log_adv"], lr, 1)
    eff = {"a": float(B[1]), "b": float(B[0]), "center": center["log_adv"], "min": 0.3, "max": 1.0,
           "median_ratio": float(np.exp(np.median(lr))), "n": int(len(e))}
    bk = nb[[f"b{k:02d}" for k in range(NB)]].to_numpy(float)
    full = np.isfinite(bk).all(axis=1) & (nb.tr_n >= 200).to_numpy()
    rel = bk[full] / nb.tw_hs_bp.to_numpy()[full, None]
    advf = nb.adv_dollar.to_numpy()[full]; etff = nb.etf.to_numpy()[full] == 1
    groups = {"liquid": (advf >= TOD_SPLIT) & ~etff, "liquid_etf": (advf >= TOD_SPLIT) & etff,
              "illiquid": advf < TOD_SPLIT}
    tod = {"bucket_minutes": 5.0, "adv_split": TOD_SPLIT}
    for g, mk in groups.items():
        pr = np.median(rel[mk], axis=0); pr /= pr.mean()
        tod[g] = [round(float(x), 4) for x in pr]; tod[f"n_{g}"] = int(mk.sum())

    # ---- fits
    nb["hold"] = nb.symbol.map(holdout)
    tr, te = nb[~nb.hold].copy(), nb[nb.hold].copy()
    th, _ = fit(tr, center)
    th_all, _ = fit(nb, center)
    print("theta", dict(zip(TERMS + ["c_d", "log_q"], np.round(th_all, 4))))
    era0 = {"large": [[1990.0, 0.0, 0.0], [2100.0, 0.0, 0.0]], "small": [[1990.0, 0.0], [2100.0, 0.0]],
            "small_below_adv": HJM_2003["dvol_musd"][-1] * 1e6}
    base = dict(center=center, clip=CLIP, tod=tod, eff=eff, sigma_prior=sigma_prior, calib_range=calib_range,
                fit_day=fit_day)
    cal_tr = make_cal(th, era=era0, **base)
    cal0 = make_cal(th_all, era=era0, **base)

    # ---- era: HJM 2003 (above Q5), Furfine 2001 (as-traded), Hasbrouck small-cap slopes; monotone in t
    h = HJM_2003
    hx = pd.DataFrame({"adv_dollar": np.array(h["dvol_musd"]) * 1e6, "sigma": np.array(h["vol_pct"]) / 100,
                       "prev_close": h["price"], "etf": 0.0, "date": h["date"]})
    hy = np.log(np.array(h["qhalf_bp"]))
    hc = pd.read_csv(HERE / "historical_checks.csv"); hc["etf"] = hc.etf.astype(float)
    fpost = hc[hc.check == "furfine_2001_post_decimal"]

    def with_era(large, small):
        era = dict(era0); era["large"] = large; era["small"] = small
        return make_cal(th_all, era=era, **base)

    def hjm_err(ab, a01=None):
        large = [[2003.25, ab[0], ab[1]], [2010.0, 0.0, 0.0], [2100.0, 0.0, 0.0]]
        if a01 is not None:
            large = [[2001.08, a01, ab[1]]] + large
        return np.log(model_pred_bp(with_era(large, era0["small"]), hx, h["date"], tick=np.full(5, 0.01))) - hy
    r03 = least_squares(hjm_err, np.array([0.0, 0.0]))
    a03, b03 = (float(v) for v in r03.x)

    def furf(d01):
        large = [[2001.08, a03 + d01, b03], [2003.25, a03, b03], [2010.0, 0.0, 0.0], [2100.0, 0.0, 0.0]]
        bp = model_pred_bp(with_era(large, era0["small"]), fpost.assign(date=fpost.date), None,
                           tick=fpost.tick.to_numpy())
        return float(np.mean(bp / 1e4 * 2 * fpost.prev_close.to_numpy() * 100)) - float(fpost.obs_value.iloc[0])
    try:
        d01 = brentq(furf, -3, 3)
    except ValueError:            # monotone floor already above 7.0c: keep the 2003 level
        d01 = 0.0
    a01 = a03 + d01
    large = [[1990.0, a01, b03], [2001.08, a01, b03], [2003.25, a03, b03], [2010.0, 0.0, 0.0], [2100.0, 0.0, 0.0]]

    xs = pd.read_csv(HERE / "hasbrouck_xs.csv.gz", parse_dates=["date"]); xs["etf"] = 0.0
    dates = sorted(xs.date.unique(), reverse=True)
    small = [[2010.0, 0.0], [2100.0, 0.0]]
    hb = {}
    r_eq = HASBROUCK["eff_over_quoted"]
    for D in dates:
        g = xs[xs.date == D]; yf = yearfrac(D)
        tm, tp = hasbrouck_target(yf, "median"), hasbrouck_target(yf, "p95")

        def loss(bs):
            # the small-cap slope only moves names below HJM Q5: match the 95th percentile (the median firm
            # sits at ~Q5 and is governed by the HJM/monotone large-cap branch; reported, not fitted)
            cal = with_era(large, sorted(small + [[round(yf, 4), float(bs)]]))
            e_ = r_eq * model_pred_bp(cal, g, D, cfg=NOCAP) / 1e4
            return (math.log(np.quantile(e_, 0.95) / tp)) ** 2
        grid = np.linspace(-2.0, 0.0, 41)
        ls_ = np.array([loss(b) for b in grid])
        bs = float(grid[np.nonzero(ls_ <= ls_.min() + 1e-9)[0].max()])   # ties -> the slope closest to 0
        small = sorted(small + [[round(yf, 4), bs]])
        cal = with_era(large, small)
        e_ = r_eq * model_pred_bp(cal, g, D, cfg=NOCAP) / 1e4
        e_old = r_eq * model_pred_bp(with_era(large, era0["small"]), g, D, cfg=NOCAP) / 1e4
        hb[str(pd.Timestamp(D).date())] = {"n": int(len(g)), "b_s": bs, "target_median": tm, "target_p95": tp,
                                          "model_median": float(np.median(e_)), "model_p95": float(np.quantile(e_, 0.95)),
                                          "no_small_slope_median": float(np.median(e_old)),
                                          "no_small_slope_p95": float(np.quantile(e_old, 0.95))}
        print("hasbrouck", hb[str(pd.Timestamp(D).date())])
    era = {"large": large, "small": small, "small_below_adv": era0["small_below_adv"],
           "hasbrouck_eff_over_quoted": r_eq}
    cal = make_cal(th_all, era=era, **base)
    cal_tr = make_cal(th, era=era, **base)
    report["era"] = era
    report["hasbrouck_match"] = hb
    hp = model_pred_bp(cal, hx, h["date"], tick=np.full(5, 0.01))
    report["hjm_2003"] = {"a": a03, "b": b03, "obs_bp": h["qhalf_bp"],
                          "pred_bp_2026_no_era": [round(float(v), 3) for v in model_pred_bp(cal0, hx, h["date"], tick=np.full(5, 0.01))],
                          "pred_bp_final": [round(float(v), 3) for v in hp]}
    report["furfine_2001_level"] = {"a": a01, "delta_vs_2003": d01}
    print("HJM", report["hjm_2003"])

    # ---- validation on the fit day (module predictions)
    def pred(df, c_):
        return model_pred_bp(c_, df, pd.Timestamp(fit_day))
    te["pred"] = pred(te, cal_tr); tr["pred"] = pred(tr, cal_tr); te["cs_half_bp"] = 0.5e4 * te.cs_full
    report["in_sample"] = stats(tr.tw_hs_bp.to_numpy(), tr.pred.to_numpy())
    report["holdout_symbols"] = stats(te.tw_hs_bp.to_numpy(), te.pred.to_numpy())
    grp = {
        "ETF": te.etf == 1, "stock": te.etf == 0,
        "ADV>=$1B": te.adv_dollar >= 1e9, "ADV $100M-1B": (te.adv_dollar >= 1e8) & (te.adv_dollar < 1e9),
        "ADV $10M-100M": (te.adv_dollar >= 1e7) & (te.adv_dollar < 1e8),
        "ADV $1M-10M": (te.adv_dollar >= 1e6) & (te.adv_dollar < 1e7), "ADV <$1M": te.adv_dollar < 1e6,
        "price<$5": te.prev_close < 5, "price $5-20": (te.prev_close >= 5) & (te.prev_close < 20),
        "price>=$20": te.prev_close >= 20, "ETF ADV>=$100M": (te.etf == 1) & (te.adv_dollar >= 1e8),
    }
    report["holdout_groups"] = {k: stats(te.tw_hs_bp[m].to_numpy(), te.pred[m].to_numpy()) for k, m in grp.items() if m.sum() > 5}
    csok = np.isfinite(te.cs_half_bp) & (te.cs_half_bp > 0)
    report["holdout_cs"] = stats(te.tw_hs_bp[csok].to_numpy(), te.cs_half_bp[csok].to_numpy())
    report["holdout_model_same_rows"] = stats(te.tw_hs_bp[csok].to_numpy(), te.pred[csok].to_numpy())
    report["holdout_cs_groups"] = {k: stats(te.tw_hs_bp[m & csok].to_numpy(), te.cs_half_bp[m & csok].to_numpy())
                                   for k, m in grp.items() if (m & csok).sum() > 5}
    nb["pred"] = pred(nb, cal); nb["cs_half_bp"] = 0.5e4 * nb.cs_full
    named = ["SPY", "QQQ", "IWM", "DIA", "EEM", "TLT", "GLD", "XLF", "AAPL", "MSFT", "NVDA", "AMZN", "JPM",
             "BAC", "F", "T", "INTC", "AA", "SOXL", "SOXX", "SMH", "MUU"]
    report["named"] = nb[nb.symbol.isin(named)][["symbol", "prev_close", "adv_dollar", "sigma", "tw_hs_bp",
                                                  "tw_med_hs_bp", "pred", "cs_half_bp"]].round(4).to_dict("records")
    loo_syms = ["SPY", "QQQ", "IWM", "DIA", "IVV", "VOO", "TLT", "GLD", "XLF", "AAPL", "MSFT", "NVDA", "AMZN", "JPM"]
    th_loo, _ = fit(nb[~nb.symbol.isin(loo_syms)], center)
    lo = nb[nb.symbol.isin(loo_syms)].copy(); lo["pred_loo"] = pred(lo, make_cal(th_loo, era=era, **base))
    report["leave_named_out"] = lo[["symbol", "tw_hs_bp", "pred_loo", "cs_half_bp"]].round(4).to_dict("records")

    # ---- other NBBO days: time stability (all symbols) and true OOS (hold-out symbols, train-only fit)
    report["oos_days"] = {}
    for d in [x for x in a.oos_days.split(",") if x]:
        o = prepare(load("nbbo", d)); o["cs_half_bp"] = 0.5e4 * o.cs_full
        o["pred"] = model_pred_bp(cal, o, pd.Timestamp(d)); o["pred_tr"] = model_pred_bp(cal_tr, o, pd.Timestamp(d))
        ok = np.isfinite(o.cs_half_bp) & (o.cs_half_bp > 0)
        hm = o.symbol.map(holdout)
        report["oos_days"][d] = {"time_stability_all_symbols": stats(o.tw_hs_bp.to_numpy(), o.pred.to_numpy()),
                                 "cs": stats(o.tw_hs_bp[ok].to_numpy(), o.cs_half_bp[ok].to_numpy()),
                                 "oos_symbol_and_day": stats(o.tw_hs_bp[hm].to_numpy(), o.pred_tr[hm].to_numpy())}

    # ---- historical checks
    ck = {}
    for name, g in hc.groupby("check"):
        bp = model_pred_bp(cal, g, None, tick=g.tick.to_numpy())
        if "spy" in name:     # SPY: per-symbol tick (decimal era -> 1 cent)
            bp = model_pred_bp(cal, g, None)
        ck[name] = {"pred_half_bp_mean": float(np.mean(bp)),
                    "pred_full_cents_mean": float(np.mean(bp / 1e4 * 2 * g.prev_close.to_numpy() * 100)),
                    "obs": float(g.obs_value.iloc[0]), "obs_unit": g.obs_unit.iloc[0]}
    # CRS 1992 (NYSE, 1/8): proxy cross-section = 2000-07 features with price >= $5, dated 1992-06-01
    g = xs[(xs.date == xs.date.min()) & (xs.prev_close >= 5)]
    bp = model_pred_bp(cal, g, "1992-06-01")
    dol = bp / 1e4 * 2 * g.prev_close.to_numpy()
    ck["crs_1992_proxy"] = {"pred_median_quoted_usd": float(np.median(dol)), "pred_median_quoted_pct": float(np.median(2 * bp / 100)),
                            "obs_median_quoted_usd": 0.27, "obs_median_quoted_pct": 1.15, "n": int(len(g)),
                            "note": "inputs: 2000-07 Yahoo cross-section (price >= $5) re-dated to 1992 (volume then was lower -> model biased tight)"}
    # Hasbrouck NYSE size deciles 1927-1998 (Gibbs c, long-run averages): ADV deciles of the proxy at 1998-06
    g = xs[xs.date == xs.date.min()].copy()
    g["dec"] = pd.qcut(np.log(g.adv_dollar), 10, labels=False)
    e98 = r_eq * model_pred_bp(cal, g, "1998-06-01", cfg=NOCAP) / 1e4
    ck["hasbrouck_size_deciles_1998_proxy"] = {"model_eff_cost_by_adv_decile_1_5_10": [float(np.mean(e98[g.dec.to_numpy() == k])) for k in (0, 4, 9)],
                                              "hasbrouck_nyse_1927_1998_decile_1_5_10": [0.023, 0.007, 0.003]}
    report["historical_checks"] = ck
    print("historical", json.dumps(ck, indent=1))

    # ---- IEX diagnostics (single venue; not used in the fit)
    m_ = nb.merge(prepare_iex(load("iex", fit_day))[["symbol", "iex_med_bp", "valid_frac"]], on="symbol")
    m_ = m_[m_.valid_frac > 0.9]
    lrr = np.log(m_.iex_med_bp / m_.tw_hs_bp)
    report["iex_vs_nbbo"] = {"n": int(len(m_)), **{k: float(np.exp(np.median(lrr[mk]))) for k, mk in (
        ("median_ratio_adv_ge_1B", m_.adv_dollar >= 1e9), ("median_ratio_adv_100M_1B", (m_.adv_dollar >= 1e8) & (m_.adv_dollar < 1e9)),
        ("median_ratio_adv_10M_100M", (m_.adv_dollar >= 1e7) & (m_.adv_dollar < 1e8)), ("median_ratio_adv_lt_10M", m_.adv_dollar < 1e7))}}
    diag = {}
    for d in [x for x in a.iex_days.split(",") if x]:
        try:
            x = prepare_iex(load("iex", d))
        except FileNotFoundError:
            continue
        x = x[(x.valid_frac > 0.9) & (x.adv_dollar >= 1e9)]
        if len(x) < 10:
            continue
        pr = model_pred_bp(cal, x, pd.Timestamp(d))
        spy = x.symbol.to_numpy() == "SPY"
        diag[d] = {"n_adv_ge_1B": int(len(x)), "median_iex_over_model": float(np.median(x.iex_med_bp.to_numpy() / pr)),
                   "spy_iex_med_bp": float(x.iex_med_bp[spy].iloc[0]) if spy.any() else None,
                   "spy_model_bp": float(pr[spy][0]) if spy.any() else None}
    report["iex_diagnostics"] = diag

    lr_all = np.log(nb.pred / nb.tw_hs_bp)
    out = dict(cal)
    out.update({"version": 2, "description": "Calibrated half-spread model, see docs/SPREADS.md",
                "fit_day": fit_day, "target": "RTH time-weighted mean NBBO quoted half-spread (fraction of mid)",
                "fit_weight": "log10(ADV$) - 5 clipped to [0.5, 5]", "residual_sd_log": float(np.std(lr_all)),
                "n_symbols": int(len(nb))})
    out["beta"] = {k: round(v, 6) for k, v in out["beta"].items()}
    out["c_d"] = round(out["c_d"], 6); out["q"] = round(out["q"], 6)
    print(json.dumps({k: v for k, v in report.items() if k not in ("named", "leave_named_out")}, indent=1, default=float)[:6000])
    print(pd.DataFrame(report["named"]).to_string())
    if a.write:
        (HERE / "spread_calibration.json").write_text(json.dumps(out, indent=1))
        (HERE / "validation_report.json").write_text(json.dumps(report, indent=1, default=float))
    return out, report


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
