"""Calibration of the synthetic-chain model to REAL chain snapshots (DoltHub
``post-no-preference/options`` EOD chains) — reproducible from the sample
stored in ``atlasbt/options/calibration/dolthub_sample.csv.gz``.

Run:  python -m atlasbt.options.calibrate   (rewrites synthetic_params.json)

Steps per (symbol, date) snapshot:
  1. r = FRED DTB3 (bank discount, ACT/360) converted to continuous ACT/365.
  2. Forward per expiry by put-call parity on de-Americanized mids (the
     DoltHub frame has NO underlying price; spot := forward of the nearest
     expiry discounted at r - q_hat, q_hat iterated from parity).
  3. OTM IVs (Black-76 on F), ATM IV per expiry, 30-day constant-maturity ATM
     IV (total variance), smile regression iv/atm - 1 = a x + b x^2,
     x = ln(K/F)/(atm sqrt T), |x| <= 3.
  4. ATM ratio = ATM30 / vol-index close (VIX for SPY; VXAPL/VXAZN/... where
     mapped).  SKEW dependence of the SPY slope a by OLS.
  5. Spread model: OLS of log(ask-bid) on log(mid), log(spot), log(dte),
     |delta|, |delta|^2, etf flag; quotes with bid>0, ask>bid.  Evidence:
     in-sample R^2, residual sd, and out-of-sample (time split) median
     absolute log error.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from atlasbt.pricing.forward import infer_forwards
from atlasbt.pricing.surface import atm_iv_per_expiry, constant_maturity_iv, smile_frame
from atlasbt.pricing.time import tbill_discount_to_cc

from .synthetic import INDEX_MAP, PARAMS_PATH, ROOT, SPX_LIKE, cboe_index_history

SAMPLE_PATH = Path(__file__).resolve().parent / "calibration" / "dolthub_sample.csv.gz"
ETFS = {"SPY", "DIA", "QQQ", "IWM", "XLE", "XLF", "XLK", "XLU", "XLB", "XLC", "XLI", "XLP", "XLRE", "XLV", "XLY"}


def from_dolthub(df: pd.DataFrame) -> pd.DataFrame:
    """Map DoltHub option_chain rows to canonical column names (no spot)."""
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"]),
        "underlying": df["act_symbol"].astype(str),
        "expiry": pd.to_datetime(df["expiration"]),
        "strike": pd.to_numeric(df["strike"]),
        "right": df["call_put"].astype(str).str[0].str.upper(),
        "bid": pd.to_numeric(df["bid"]), "ask": pd.to_numeric(df["ask"]),
        "iv": pd.to_numeric(df["vol"]), "delta": pd.to_numeric(df["delta"]),
    })
    out["mid"] = 0.5 * (out["bid"] + out["ask"])
    out["dte"] = (out["expiry"] - out["date"]).dt.days
    out["T"] = out["dte"] / 365.0
    out["exercise"] = "AMERICAN"
    out["source"] = "dolthub"
    return out


L3_PATH = ROOT / "data_raw" / "historicaloptiondata_free_samples" / "Sample_L3_20190815.zip"
L3_SYMBOLS = ("SPX", "SPXW", "SPY", "QQQ", "IWM", "DIA", "AAPL", "AMZN", "MSFT", "TSLA", "NVDA", "KO", "F", "XLF",
              "XLE", "XLU", "GLD", "TLT", "EEM", "JPM", "XOM", "AMD")


def load_l3(symbols=L3_SYMBOLS, path: Path = L3_PATH) -> pd.DataFrame:
    """historicaloptiondata.com free full-market EOD sample (2019-08-15), rows of
    ``symbols`` (the option ROOT: SPX = AM monthlies, SPXW = PM weeklies/dailies),
    mapped to the calibration frame WITH the vendor's real underlying price.
    Vendor convention: SPX AM expiration is the settlement Friday (we keep it;
    T is measured to that date's open)."""
    import zipfile
    z = zipfile.ZipFile(path)
    it = pd.read_csv(z.open("L3_options_20190815.csv"), chunksize=500_000)
    raw = pd.concat([c[c["UnderlyingSymbol"].isin(set(symbols))] for c in it])
    raw.columns = [c.strip() for c in raw.columns]
    out = pd.DataFrame({
        "date": pd.to_datetime(raw["DataDate"]), "underlying": raw["UnderlyingSymbol"].astype(str),
        "expiry": pd.to_datetime(raw["Expiration"]), "strike": raw["Strike"].astype(float),
        "right": raw["Type"].str[0].str.upper(), "bid": raw["Bid"].astype(float), "ask": raw["Ask"].astype(float),
        "iv": raw["IVMean"].astype(float), "delta": raw["Delta"].astype(float),
        "spot_true": raw["UnderlyingPrice"].astype(float), "source": "L3_20190815"})
    out["mid"] = 0.5 * (out["bid"] + out["ask"])
    out["dte"] = (out["expiry"] - out["date"]).dt.days
    am = out["underlying"] == "SPX"
    out["T"] = np.where(am, (out["dte"] - 6.5 / 24) / 365.0, out["dte"] / 365.0)
    out["exercise"] = np.where(out["underlying"].isin(["SPX", "SPXW"]), "EUROPEAN", "AMERICAN")
    return out[out["dte"] >= 1].reset_index(drop=True)


def dtb3() -> pd.Series:
    f = ROOT / "data_raw" / "fred" / "DTB3.csv"
    d = pd.read_csv(f)
    s = pd.Series(pd.to_numeric(d.iloc[:, 1], errors="coerce").values, index=pd.to_datetime(d.iloc[:, 0])).dropna()
    return s / 100.0


def snapshot_stats(snap: pd.DataFrame, r: float, spot: float | None = None) -> dict:
    """Forward, ATM-30 and smile coefficients for one (symbol, date) snapshot.
    ``spot``: real underlying price if known (L3); else inferred from parity."""
    s0 = snap.copy()
    if spot is None:
        # European parity on the nearest expiry gives a spot proxy (q ~ 0 over 2-3 weeks)
        fw0 = infer_forwards(s0, r=r, spot=1.0, american=False)
        fw0 = fw0[np.isfinite(fw0["F"])]
        if fw0.empty:
            return {}
        near = fw0.sort_values("T").iloc[0]
        spot = float(near["F"] * np.exp(-r * near["T"]))
    s0["underlying_price"] = spot
    american = bool((s0["exercise"] == "AMERICAN").any()) if "exercise" in s0.columns else True
    fw = infer_forwards(s0, r=r, spot=spot, american=american, iterations=1)
    fw = fw[np.isfinite(fw["F"])]
    sm = smile_frame(s0, fw)
    atm = atm_iv_per_expiry(sm)
    atm = atm[np.isfinite(atm["atm_iv"])]
    if atm.empty:
        return {}
    a30, flag = constant_maturity_iv(atm, 30)
    sm = sm.merge(atm[["expiry", "atm_iv"]], on="expiry")
    sm = sm[sm["otm"] & np.isfinite(sm["iv_fit"]) & (sm["bid"] > 0)]
    sm["x"] = sm["k"] / (sm["atm_iv"] * np.sqrt(sm["T"]))
    sm = sm[np.abs(sm["x"]) <= 4]
    a = b = np.nan
    if len(sm) >= 6:
        A = np.c_[sm["x"], sm["x"] ** 2]  # snapshot-pooled diagnostic only
        y = sm["iv_fit"] / sm["atm_iv"] - 1
        a, b = np.linalg.lstsq(A, y.values, rcond=None)[0]
    per_exp = []
    from .synthetic import XMAX
    for e, g in sm.groupby("expiry"):
        core = g[np.abs(g["x"]) <= XMAX]
        if len(core) >= 5 and core["x"].min() < -0.5 and core["x"].max() > 0.3:
            ae, be = np.linalg.lstsq(np.c_[core["x"], core["x"] ** 2], (core["iv_fit"] / core["atm_iv"] - 1).values,
                                     rcond=None)[0]
            rec = {"dte": float(g["T"].iloc[0] * 365), "a": float(ae), "b": float(be), "n": int(len(core)),
                   "wing_put": np.nan, "wing_call": np.nan}
            gx = g["iv_fit"] / g["atm_iv"]
            for side, m, edge in (("wing_put", g["x"] < -XMAX, -XMAX), ("wing_call", g["x"] > XMAX, XMAX)):
                if m.sum() >= 2:
                    base = 1 + ae * edge + be * edge * edge
                    dx = (g.loc[m, "x"] - edge).values
                    rec[side] = float(np.sum(dx * (gx[m].values - base)) / np.sum(dx * dx))
            per_exp.append(rec)
    return {"spot": spot, "atm30": a30, "atm30_flag": flag, "a": a, "b": b, "n_smile": len(sm), "per_expiry": per_exp,
            "q_near": float(fw.sort_values("T")["q_implied"].iloc[0]) if len(fw) else np.nan}


SPREAD_FEATURES = ("log_mid", "log_mid2", "absdelta", "log_dte", "log_vix")


def spread_design(q: pd.DataFrame) -> pd.DataFrame:
    lm = np.log(np.maximum(q["mid"].values, 0.005))
    return pd.DataFrame({"log_mid": lm, "log_mid2": lm * lm, "absdelta": np.abs(q["delta"].values),
                         "log_dte": np.log(np.maximum(q["dte"].values, 1)),
                         "log_vix": np.log(q["vix"].values)}, index=q.index)


def _fit_fe(q: pd.DataFrame):
    X = spread_design(q)
    syms = sorted(q["underlying"].unique())
    D = pd.get_dummies(q["underlying"]).reindex(columns=syms, fill_value=0).astype(float)
    A = np.c_[X.values, D.values]
    beta = np.linalg.lstsq(A, q["y"].values, rcond=None)[0]
    coef = dict(zip(X.columns, map(float, beta[:X.shape[1]])))
    fe = dict(zip(syms, map(float, beta[X.shape[1]:])))
    res = q["y"].values - A @ beta
    return coef, fe, res


def fit_spread_model(q: pd.DataFrame, split_date=None) -> dict:
    """log(ask-bid) = FE[underlying] + b1 log(mid) + b2 log(mid)^2 + b3 |delta| + b4 log(dte) + b5 log(VIX)."""
    q = q[(q["bid"] > 0) & (q["ask"] > q["bid"]) & np.isfinite(q["delta"]) & np.isfinite(q["vix"])].copy()
    q["y"] = np.log(q["ask"] - q["bid"])
    coef, fe, res = _fit_fe(q)
    etf_fe = [v for k, v in fe.items() if k in ETFS and k != "SPY"]
    stk_fe = [v for k, v in fe.items() if k not in ETFS]
    default = {"etf": float(np.median(etf_fe)) if etf_fe else float(np.median(list(fe.values()))),
               "stock": float(np.median(stk_fe)) if stk_fe else float(np.median(list(fe.values())))}
    ev = {"n_quotes": int(len(q)), "r2": float(1 - res.var() / q["y"].var()), "resid_sd_log": float(res.std()),
          "median_abs_log_err": float(np.median(np.abs(res))),
          "median_abs_log_err_by_symbol": {k: float(np.median(np.abs(res[(q["underlying"] == k).values])))
                                           for k in sorted(fe)}}
    if "source" in q.columns:
        ev["median_abs_log_err_by_source"] = {k: float(np.median(np.abs(res[(q["source"] == k).values])))
                                              for k in sorted(q["source"].dropna().unique())}
    if split_date is not None:
        tr = (q["date"] < split_date).values
        if tr.sum() > 50 and (~tr).sum() > 50 and set(q.loc[~tr, "underlying"]) <= set(q.loc[tr, "underlying"]):
            c2, fe2, _ = _fit_fe(q[tr])
            Xo = spread_design(q[~tr])
            pred = Xo.values @ np.array([c2[c] for c in Xo.columns]) + q.loc[~tr, "underlying"].map(fe2).values
            r_oos = q["y"].values[~tr] - pred
            ev.update({"oos_split": str(pd.Timestamp(split_date).date()), "oos_n": int((~tr).sum()),
                       "oos_median_abs_log_err": float(np.median(np.abs(r_oos))),
                       "oos_median_model_over_actual": float(np.exp(np.median(-r_oos)))})
    q["absres"] = np.abs(res)
    q["bucket"] = pd.cut(q["delta"].abs(), [0, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 1.0])
    ev["median_abs_log_err_by_absdelta"] = {str(k): float(v) for k, v in q.groupby("bucket", observed=True)["absres"].median().items()}
    return {"coef": coef, "symbol_fe": fe, "default_fe": default, "evidence": ev}


INDEX_LIKE = SPX_LIKE | {"QQQ", "IWM", "DIA"}


def calibrate(sample: pd.DataFrame, write: bool = True, path: Path = PARAMS_PATH, extra: list | None = None) -> dict:
    df = from_dolthub(sample) if "act_symbol" in sample.columns else sample
    if extra:
        df = pd.concat([df] + list(extra), ignore_index=True)
    rates = dtb3()
    vix, skew = cboe_index_history("VIX"), cboe_index_history("SKEW")
    stats = []
    spot_by = {}
    for (sym, d), snap in df.groupby(["underlying", "date"]):
        rr = rates.loc[:d]
        r = float(tbill_discount_to_cc(rr.iloc[-1], 91)) if len(rr) else 0.0
        sp_true = float(snap["spot_true"].iloc[0]) if "spot_true" in snap.columns and snap["spot_true"].notna().any() else None
        st = snapshot_stats(snap, r, sp_true)
        if not st:
            continue
        st.update(symbol=sym, date=d, r=r)
        idx_name = "VIX" if sym in SPX_LIKE else INDEX_MAP.get(sym)
        st["source"] = str(snap["source"].iloc[0]) if "source" in snap.columns else ""
        st["index_name"] = idx_name
        st["index"] = np.nan
        if idx_name:
            try:
                s = cboe_index_history(idx_name)
                st["index"] = float(s.loc[:d].iloc[-1]) / 100.0 if len(s.loc[:d]) else np.nan
            except FileNotFoundError:
                pass
        st["vix"] = float(vix.loc[:d].iloc[-1]) / 100.0
        st["skew"] = float(skew.loc[:d].iloc[-1])
        stats.append(st)
        spot_by[(sym, d)] = st["spot"]
    S = pd.DataFrame(stats)
    out: dict = {}
    spy = S[S["symbol"].isin(INDEX_LIKE) & np.isfinite(S["atm30"]) & np.isfinite(S["index"])]
    idx_ratio = (spy["atm30"] / spy["index"])
    out["atm_ratio_index"] = float(idx_ratio.median()) if len(spy) else 0.88
    out["atm_ratio_logvix"] = 0.0
    ratio_note = "constant (median)"
    if len(spy) >= 10:
        x = np.log(spy["index"].values / 0.20)
        A = np.c_[np.ones(len(spy)), x]
        coef, *_ = np.linalg.lstsq(A, idx_ratio.values, rcond=None)
        res_ = idx_ratio.values - A @ coef
        se = np.sqrt(np.sum(res_ ** 2) / (len(spy) - 2) * np.linalg.inv(A.T @ A)[1, 1])
        if abs(coef[1] / se) > 2:
            out["atm_ratio_index"], out["atm_ratio_logvix"] = float(coef[0]), float(coef[1])
            ratio_note = f"ratio = c0 + c1 ln(index/0.20), t-stat {coef[1] / se:.2f}, resid sd {res_.std():.3f}"
    sing = S[~S["symbol"].isin(INDEX_LIKE) & S["index_name"].notna() & np.isfinite(S["atm30"]) & np.isfinite(S["index"])]
    out["atm_ratio_single"] = float((sing["atm30"] / sing["index"]).median()) if len(sing) else 0.95
    sp = S[S["symbol"].isin(INDEX_LIKE) & np.isfinite(S["a"])]
    E = pd.DataFrame([dict(symbol=row.symbol, date=row.date, **pe) for row in S.itertuples() for pe in row.per_expiry])
    skew_note = "a(T) = a30 (DTE/30)^gamma, robust fit of log(-a) on log(DTE/30) over per-expiry smiles; b = median"

    def _term_fit(Ex):
        import statsmodels.api as sm_
        Ex = Ex[(Ex["a"] < -0.01) & (Ex["dte"] >= 1)]
        if len(Ex) < 8:
            return None
        X = sm_.add_constant(np.log(Ex["dte"].values / 30.0))
        r_ = sm_.RLM(np.log(-Ex["a"].values), X, M=sm_.robust.norms.HuberT()).fit()
        return {"a30": float(-np.exp(r_.params[0])), "gamma": float(r_.params[1]), "b": float(Ex["b"].median()),
                "wing_put": float(np.nanmedian(Ex["wing_put"])), "wing_call": float(np.nanmedian(Ex["wing_call"])),
                "n_wing_put": int(Ex["wing_put"].notna().sum()), "n_wing_call": int(Ex["wing_call"].notna().sum()),
                "n_expiries": int(len(Ex)), "median_abs_resid_log": float(np.median(np.abs(r_.resid)))}
    tf_idx = _term_fit(E[E["symbol"].isin(INDEX_LIKE)]) if len(E) else None
    if tf_idx:
        out.update(skew_a0=tf_idx["a30"], skew_a1=0.0, skew_gamma=tf_idx["gamma"], skew_b=tf_idx["b"],
                   wing_put=tf_idx["wing_put"], wing_call=tf_idx["wing_call"],
                   skew_range=[float(sp["skew"].min()), float(sp["skew"].max())] if len(sp) else [110.0, 160.0])
    tf_single = _term_fit(E[~E["symbol"].isin(INDEX_LIKE | ETFS)]) if len(E) else None
    if tf_single:
        out.update(single_a=tf_single["a30"], single_gamma=tf_single["gamma"], single_b=tf_single["b"],
                   single_wing_put=tf_single["wing_put"], single_wing_call=tf_single["wing_call"])
    singles = S[~S["symbol"].isin(INDEX_LIKE | ETFS) & np.isfinite(S["a"])]
    # spreads
    df = df.copy()
    df["spot"] = [spot_by.get((s, d), np.nan) for s, d in zip(df["underlying"], df["date"])]
    df["vix"] = [float(vix.loc[:d].iloc[-1]) / 100.0 for d in df["date"]]
    dates = sorted(df["date"].unique())
    split = dates[int(len(dates) * 0.7)] if len(dates) > 3 else None
    sm = fit_spread_model(df, split)
    out["spread_coef"] = sm["coef"]
    out["spread_symbol_fe"] = sm["symbol_fe"]
    out["spread_default_fe"] = sm["default_fe"]
    q = df[(df["bid"] > 0) & (df["ask"] > df["bid"])]
    out["spread_resid_sd"] = sm["evidence"]["resid_sd_log"]
    out["evidence"] = {
        "data": {"source": "DoltHub post-no-preference/options option_chain (EOD, 2020-2026 sample) + "
                           "historicaloptiondata.com L3 full-market EOD 2019-08-15 (data_raw)", "n_rows": int(len(df)),
                 "symbols": sorted(df["underlying"].unique().tolist()),
                 "dates": [str(pd.Timestamp(x).date()) for x in dates], "n_snapshots": int(len(S))},
        "atm_ratio_index": {"n": int(len(spy)), "rule": ratio_note, "median": float(idx_ratio.median()) if len(spy) else None,
                            "iqr": [float(idx_ratio.quantile(.25)), float(idx_ratio.quantile(.75))] if len(spy) else None,
                            "per_snapshot": [f"{sy} {pd.Timestamp(d).date()}: {v:.3f}" for sy, d, v in zip(spy["symbol"], spy["date"], idx_ratio)]},
        "atm_ratio_single": {"n": int(len(sing)), "pairs": sorted(set(zip(sing["symbol"], sing["index_name"])))},
        "skew_term_fit_index": {"rule": skew_note, **(tf_idx or {})},
        "skew_term_fit_singles": tf_single or {},
        "spread_model": sm["evidence"],
        "zero_bid_fraction": float((df["bid"] <= 0).mean()),
        "n_spread_quotes": int(len(q)),
    }
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1, default=str))
    out["_stats"] = S
    return out


def validate_against_put_index(params=None, q_spx: float = 0.018, start_year: int = 2008) -> dict:
    """Independent check of the synthetic premium LEVEL on SPX (no SPX chains
    needed): the Cboe PUT index sells a one-month ATM SPX put at each monthly
    roll (3rd Friday) fully collateralised in T-bills.  From consecutive roll
    values, premium_actual ~= ret*K - interest + max(K - S_settle, 0)
    (S_settle proxied by the SPX close of the last trading day).  Compared
    with the synthetic BSM premium at the roll-day close (VIX x atm_ratio,
    q = 1.8%).  Noise from the settlement/roll-time proxies is zero-mean."""
    from atlasbt.core import clock
    from atlasbt.pricing import bsm_price
    from .listing import _prev_business, third_friday
    from .synthetic import SyntheticParams
    p = params or SyntheticParams.load()
    put, spx, vix, rates = cboe_index_history("PUT"), cboe_index_history("SPX"), cboe_index_history("VIX"), dtb3()
    cal = clock.nyse()
    rows = []
    for y in range(start_year, pd.Timestamp.today().year + 1):
        for m in range(1, 13):
            R = _prev_business(third_friday(y, m))
            R2 = _prev_business(third_friday(y + (m == 12), m % 12 + 1))
            if R2 > put.index[-1] or R not in put.index or R2 not in put.index or R not in spx.index:
                continue
            S = spx.loc[R]
            K = np.round(S / 5) * 5
            T = (R2 - R).days / 365.0 - 6.5 / 24 / 365
            r = float(tbill_discount_to_cc(rates.loc[:R].iloc[-1], 30))
            from .synthetic import atm_ratio
            lvl = vix.loc[:R].iloc[-1] / 100
            sig = lvl * float(atm_ratio(lvl, p))
            psyn = float(bsm_price(S, K, T, r, q_spx, sig, "P"))
            settle = spx.loc[:cal.prev_session(R2)].iloc[-1]
            ret = put.loc[R2] / put.loc[R] - 1
            pact = ret * K - K * (np.exp(r * (R2 - R).days / 365) - 1) + max(K - settle, 0)
            rows.append(dict(R=R, psyn=psyn, pact=pact))
    d = pd.DataFrame(rows)
    if d.empty:
        return {}
    yr = d.groupby(d["R"].dt.year).apply(lambda g: g["psyn"].sum() / g["pact"].sum())
    return {"n_months": int(len(d)), "period": [str(d["R"].min().date()), str(d["R"].max().date())],
            "median_ratio_synthetic_to_implied": float((d["psyn"] / d["pact"]).median()),
            "aggregate_ratio": float(d["psyn"].sum() / d["pact"].sum()),
            "yearly_aggregate_ratio": {int(k): round(float(v), 3) for k, v in yr.items()},
            "mean_premium_synthetic": float(d["psyn"].mean()), "mean_premium_implied": float(d["pact"].mean())}


def gspc_daily() -> pd.DataFrame:
    f = ROOT / "data_raw" / "yfinance" / "daily" / "IDX_GSPC.csv"
    d = pd.read_csv(f)
    d.index = pd.to_datetime(d["Date"].str[:10])
    return d[["Open", "Close"]].astype(float)


def put_implied_atm(start="2008-01-01", end="2016-12-31", q_spx: float = 0.018) -> pd.DataFrame:
    """ATM SPX implied vol backed out of the Cboe PUT index at each monthly roll
    (premium = ((ret - rf) K + payoff)/(1 + rf), payoff from the SPX opening print
    of the next roll as SOQ proxy).  Real SPX market information (the index is
    computed from traded SPX put prices) available back to 2007."""
    from atlasbt.pricing import implied_vol
    from .listing import _prev_business, third_friday
    put, vix, rates, g = cboe_index_history("PUT"), cboe_index_history("VIX"), dtb3(), gspc_daily()
    rows = []
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        for m in range(1, 13):
            R = _prev_business(third_friday(y, m))
            R2 = _prev_business(third_friday(y + (m == 12), m % 12 + 1))
            if not (pd.Timestamp(start) <= R <= pd.Timestamp(end)) or R2 > put.index[-1]:
                continue
            if R not in put.index or R2 not in put.index or R not in g.index or R2 not in g.index:
                continue
            S = float(g.loc[R, "Close"])
            K = np.round(S / 5) * 5
            days = (R2 - R).days
            T = days / 365 - 6.5 / 24 / 365
            r = float(tbill_discount_to_cc(rates.loc[:R].iloc[-1], 30))
            rf = np.exp(r * days / 365) - 1
            payoff = max(K - float(g.loc[R2, "Open"]), 0.0)
            prem = ((put.loc[R2] / put.loc[R] - 1 - rf) * K + payoff) / (1 + rf)
            iv = float(implied_vol(prem, S, K, T, r, q_spx, "P").iv)
            rows.append(dict(R=R, vix=float(vix.loc[:R].iloc[-1]) / 100, iv=iv))
    d = pd.DataFrame(rows).dropna()
    d["ratio"] = d["iv"] / d["vix"]
    return d


def fit_atm_ratio(d: pd.DataFrame) -> dict:
    """ratio = c0 + c1 ln(VIX/0.20), robust Huber regression (the settlement /
    roll-time proxies create outlier months)."""
    import statsmodels.api as sm
    x = sm.add_constant(np.log(d["vix"].values / 0.20))
    r = sm.RLM(d["ratio"].values, x, M=sm.robust.norms.HuberT()).fit()
    return {"c0": float(r.params[0]), "c1": float(r.params[1]), "se_c1": float(r.bse[1]),
            "median_abs_resid": float(np.median(np.abs(r.resid))), "n": int(len(d))}


def atm_ratio_schedule(min_months: int = 24, last_year: int | None = None) -> list:
    """Point-in-time schedule: for each year Y, (c0, c1) fitted on PUT-implied
    ATM IVs of rolls whose settlement is <= Y-1 year end (expanding window,
    >= ``min_months``).  Entry {'from': 'Y-01-01', c0, c1, n, fitted_through}."""
    d = put_implied_atm("2007-01-01", f"{(last_year or pd.Timestamp.today().year)}-12-31")
    out = []
    for y in range(d["R"].min().year + 1, (last_year or pd.Timestamp.today().year) + 1):
        h = d[d["R"] < pd.Timestamp(f"{y - 1}-12-01")]   # roll settled before year end
        if len(h) < min_months:
            continue
        f = fit_atm_ratio(h)
        out.append({"from": f"{y}-01-01", "c0": f["c0"], "c1": f["c1"], "n": f["n"],
                    "fitted_through": str(h["R"].max().date())})
    return out


def replicate_put_index(start: str = "2008-01-01", end: str | None = None, price: str = "mid",
                        q_spx: float = 0.018, generator=None) -> dict:
    """Known-answer fidelity test of the SYNTHETIC surface: reproduce the Cboe
    PUT index (one-month ATM SPX put-write, T-bill collateral) roll-to-roll
    monthly returns using ``SyntheticChainGenerator`` end to end.

    At each roll session R (3rd Friday, holiday-adjusted) at 16:30 ET: SPX
    synthetic chain (spot = SPX close, VIX family + SKEW known at that time),
    sell the AM-settled SPX monthly expiring next cycle at the listed strike
    nearest spot, at the synthetic ``price`` ('mid' or 'bid'); collateral K earns
    the 1-month T-bill (DTB3 proxy); settlement = SOQ proxied by the SPX
    opening print of the next roll session.  Monthly return
    = (premium*(1+rf) - payoff)/K + rf.  Deviations from Cboe's methodology
    (roll at the 11:30-12:00 VWAP, exact SOQ, 1/3-month bill mix) are noise
    sources included in the reported tracking error."""
    from atlasbt.core import clock
    from .chain import nearest_atm
    from .listing import _prev_business, third_friday
    from .synthetic import SyntheticChainGenerator
    gen = generator or SyntheticChainGenerator()
    put, rates, g = cboe_index_history("PUT"), dtb3(), gspc_daily()
    end = pd.Timestamp(end) if end else put.index[-1]
    rolls = []
    for y in range(pd.Timestamp(start).year, end.year + 1):
        for m in range(1, 13):
            R = _prev_business(third_friday(y, m))
            if pd.Timestamp(start) <= R <= end:
                rolls.append(R)
    rows = []
    for R, R2 in zip(rolls[:-1], rolls[1:]):
        if R not in put.index or R2 not in put.index or R not in g.index or R2 not in g.index:
            continue
        S = float(g.loc[R, "Close"])
        days = (R2 - R).days
        r = float(tbill_discount_to_cc(rates.loc[:R].iloc[-1], 30))
        ch = gen.chain("SPX", clock.et(R, "16:30"), S, r, q_spx, max_dte=days + 3, min_dte=days - 3)
        ch = ch[(ch["root"] == "SPX") & (ch["settlement"] == "CASH_AM") & (ch["settlement_date"] == R2)]
        if ch.empty:
            continue
        K = nearest_atm(ch, S, "P")
        row = ch[(ch["strike"] == K) & (ch["right"] == "P")].iloc[0]
        prem = float(row[price])
        rf = np.exp(r * days / 365.0) - 1
        payoff = max(K - float(g.loc[R2, "Open"]), 0.0)
        syn = (prem * (1 + rf) - payoff) / K + rf
        act = put.loc[R2] / put.loc[R] - 1
        rows.append(dict(R=R, K=K, prem=prem, iv=float(row["iv"]), syn=syn, act=act))
    d = pd.DataFrame(rows)
    if d.empty:
        return {}
    diff = d["syn"] - d["act"]
    return {"n_months": int(len(d)), "period": [str(d["R"].min().date()), str(d["R"].max().date())], "price": price,
            "ann_return_synthetic": float((1 + d["syn"]).prod() ** (12 / len(d)) - 1),
            "ann_return_put_index": float((1 + d["act"]).prod() ** (12 / len(d)) - 1),
            "mean_monthly_diff_bp": float(diff.mean() * 1e4),
            "tracking_error_ann": float(diff.std() * np.sqrt(12)),
            "correlation": float(np.corrcoef(d["syn"], d["act"])[0, 1]),
            "median_abs_monthly_diff_bp": float(diff.abs().median() * 1e4),
            "worst_month_diff_bp": float(diff.abs().max() * 1e4)}


def l3_spx_surface_check(dtes=(7, 14, 30, 60, 90)) -> dict:
    """Synthetic SPX chain vs the REAL full SPXW chain on 2019-08-15 (L3 file):
    for the PM expiries nearest each target DTE compare ATM IV, 25-delta put/call
    IV (from mids, Black-76 on parity forwards) and quoted spreads."""
    from atlasbt.core import clock
    from atlasbt.options.metrics import chain_metrics
    from .synthetic import SyntheticChainGenerator
    real = load_l3(("SPXW",))
    day = pd.Timestamp("2019-08-15")
    S = float(real["spot_true"].iloc[0])
    r = float(tbill_discount_to_cc(dtb3().loc[:day].iloc[-1], 30))
    now = clock.et(day, "16:30")
    rc = real.assign(underlying="SPX", settlement="CASH_PM", underlying_price=S)
    syn = SyntheticChainGenerator().chain("SPX", now, S, r, 0.018, max_dte=100)
    syn = syn[syn["settlement"] == "CASH_PM"]
    rows = []
    for t in dtes:
        exp = sorted(rc["expiry"].unique(), key=lambda e: abs((pd.Timestamp(e) - day).days - t))[0]
        mr = chain_metrics(rc[rc["expiry"] == exp], now, r, spot=S, american=False)
        ms = chain_metrics(syn[syn["expiry"] == exp], now, r, spot=S, american=False)
        atm_r = mr["atm_by_expiry"]["atm_iv"].iloc[0] if len(mr.get("atm_by_expiry", [])) else np.nan
        atm_s = ms["atm_by_expiry"]["atm_iv"].iloc[0] if len(ms.get("atm_by_expiry", [])) else np.nan
        # spreads near the money (|K/S-1| < 3%)
        nr = rc[(rc["expiry"] == exp) & ((rc["strike"] / S - 1).abs() < 0.03) & (rc["bid"] > 0)]
        ns = syn[(syn["expiry"] == exp) & ((syn["strike"] / S - 1).abs() < 0.03)]
        rows.append(dict(expiry=str(pd.Timestamp(exp).date()), dte=int((pd.Timestamp(exp) - day).days),
                         atm_real=round(float(atm_r), 4), atm_syn=round(float(atm_s), 4),
                         skew25_real=round(float(mr.get("skew25_30", np.nan)), 4),
                         skew25_syn=round(float(ms.get("skew25_30", np.nan)), 4),
                         med_spread_real=float((nr["ask"] - nr["bid"]).median()),
                         med_spread_syn=float((ns["ask"] - ns["bid"]).median())))
    return {"date": "2019-08-15", "spot": S, "rows": rows,
            "note": "skew25 per single expiry (constant-maturity function evaluated on one expiry -> flat in T)"}


LEAN_SPX = ROOT / "data_raw" / "gh_quantconnect_lean_data" / "Data" / "indexoption" / "usa" / "minute" / "spx"


def lean_spx_spread_check(params=None, window=("15:00", "15:45")) -> dict:
    """Out-of-sample check of the SPX spread model against the only free
    intraday SPX option quotes (QuantConnect Lean fixtures, AlgoSeek minute
    quotes, 2021-01-04 and 2021-01-14, a handful of Jan-15-2021 AM contracts).
    Compares the median quoted spread in ``window`` ET with the model."""
    import zipfile
    from atlasbt.pricing import bsm_greeks, implied_vol
    from .synthetic import SyntheticParams, model_spread
    p = params or SyntheticParams.load()
    g, vix, rates = gspc_daily(), cboe_index_history("VIX"), dtb3()
    lo = int(window[0][:2]) * 3600000 + int(window[0][3:]) * 60000
    hi = int(window[1][:2]) * 3600000 + int(window[1][3:]) * 60000
    rows = []
    for zf in sorted(LEAN_SPX.glob("*_quote_european.zip")):
        day = pd.Timestamp(zf.name[:8])
        S = float(g.loc[day, "Close"])
        r = float(tbill_discount_to_cc(rates.loc[:day].iloc[-1], 30))
        z = zipfile.ZipFile(zf)
        for name in z.namelist():
            parts = name[:-4].split("_")
            right, strike, exp = parts[5][0].upper(), int(parts[6]) / 10000.0, pd.Timestamp(parts[7])
            q = pd.read_csv(z.open(name), header=None)
            q = q[(q[0] >= lo) & (q[0] <= hi)]
            bid, ask = q[4] / 10000.0, q[9] / 10000.0
            ok = (bid > 0) & (ask > bid)
            if ok.sum() < 5:
                continue
            spr, mid = float((ask - bid)[ok].median()), float(((ask + bid) / 2)[ok].median())
            dte = (exp - day).days
            T = (dte - 6.5 / 24) / 365.0
            iv = implied_vol(mid, S, strike, T, r, 0.015, right).iv
            d = float(bsm_greeks(S, strike, T, r, 0.015, iv if np.isfinite(iv) else 0.3, right).delta)
            m = float(model_spread(mid, dte, abs(d), "SPX", p, vix=float(vix.loc[day]) / 100, root="SPX"))
            rows.append(dict(day=str(day.date()), contract=f"{strike:g}{right} {exp.date()}", mid=mid, actual=spr,
                             model=m, ratio=m / spr))
    if not rows:
        return {}
    d = pd.DataFrame(rows)
    return {"n_contract_days": int(len(d)), "median_model_over_actual": float(d["ratio"].median()),
            "median_abs_log_err": float(np.median(np.abs(np.log(d["ratio"])))), "rows": d.round(3).to_dict("records")}


def main():  # pragma: no cover
    sample = pd.read_csv(SAMPLE_PATH)
    extra = [load_l3()] if L3_PATH.exists() else None
    res = calibrate(sample, extra=extra)
    S = res.pop("_stats")
    # ATM ratio for index products: calibrate on SPX itself (PUT-implied IVs, 2008-2016 in-sample);
    # the DoltHub/L3 chain-based fit is kept as a cross-check in the evidence.
    d = json.loads(PARAMS_PATH.read_text())
    sched = atm_ratio_schedule()
    d["evidence"]["atm_ratio_chains_fit_2019_2026"] = {"c0": d["atm_ratio_index"], "c1": d.get("atm_ratio_logvix", 0.0),
                                                      "note": "cross-check only (SPY/DIA/QQQ/IWM/SPX chains)"}
    d["evidence"]["atm_ratio_put_implied_full_fit"] = fit_atm_ratio(put_implied_atm("2007-01-01", "2026-12-31"))
    d["atm_ratio_schedule"] = sched
    # static fallback (used when as_of is unknown) = latest PIT fit
    d["atm_ratio_index"], d["atm_ratio_logvix"] = sched[-1]["c0"], sched[-1]["c1"]
    PARAMS_PATH.write_text(json.dumps(d, indent=1, default=str))
    val = validate_against_put_index()
    d = json.loads(PARAMS_PATH.read_text())
    d["evidence"]["validation_put_index"] = val
    d["evidence"]["replication_put_index_mid"] = replicate_put_index(price="mid")
    d["evidence"]["replication_put_index_bid"] = replicate_put_index(price="bid")
    d["evidence"]["replication_put_index_mid_2014_2017"] = replicate_put_index("2014-01-01", "2017-12-31")
    d["evidence"]["replication_put_index_mid_OOS_2017_2026"] = replicate_put_index("2017-01-01")
    d["evidence"]["replication_put_index_mid_IS_2008_2016"] = replicate_put_index("2008-01-01", "2016-12-31")
    d["evidence"]["lean_spx_intraday_spread_check"] = lean_spx_spread_check()
    d["evidence"]["l3_spx_surface_check"] = l3_spx_surface_check()
    print(json.dumps(d["evidence"]["replication_put_index_mid"], indent=1))
    PARAMS_PATH.write_text(json.dumps(d, indent=1, default=str))
    print(json.dumps(val, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "evidence"}, indent=1))
    print(S.describe().T)


if __name__ == "__main__":  # pragma: no cover
    main()
