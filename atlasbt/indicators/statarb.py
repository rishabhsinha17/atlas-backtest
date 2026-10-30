"""Statistical-arbitrage building blocks (all causal).

* ``rolling_ols``      - trailing-window OLS of y on X (+const); betas fit on
                         [t-w+1, t]; ``resid`` uses beta_t (in-window, known at t),
                         ``resid_oos`` uses beta_{t-1} (fit through t-1, strictly
                         out-of-sample for observation t).
* ``kalman_hedge``     - time-varying hedge ratio (random-walk state, Chan 2013
                         "Algorithmic Trading" ch. 3; Harvey 1989). One-step-ahead:
                         the innovation at t uses the PRIOR state (data <= t-1);
                         the state is UPDATED only after observing t.
* ``engle_granger``    - Engle & Granger (1987) two-step test; ADF on the OLS
                         residuals with MacKinnon (1994/2010) p-values via
                         statsmodels ``coint``.
* ``pca_residuals``    - Avellaneda & Lee (2010) style: eigen-decomposition of the
                         correlation (or covariance) matrix of returns over
                         [t-w, t-1]; residual of r_t after removing the top-k
                         factor reconstruction.
* ``ssd_pairs``        - Gatev, Goetzmann & Rouwenhorst (2006) distance method:
                         sum of squared differences of normalised cumulative
                         total-return prices over the formation window.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from ._util import as_float_array, index_of


# ---------------------------------------------------------------- rolling OLS
def rolling_ols(y, X, window: int, add_const: bool = True, min_periods: int | None = None) -> pd.DataFrame:
    """Trailing-window OLS with NaN-row dropping.

    y: Series; X: Series or DataFrame (same index). Returns DataFrame with
    columns beta_<name> (and 'alpha' when add_const), resid, resid_oos, sigma
    (residual stdev, ddof = k), n (observations used).
    """
    ys = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y, float))
    Xd = X.to_frame() if isinstance(X, pd.Series) else (X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, float), index=ys.index))
    Xd = Xd.astype(float)
    names = [f"beta_{c}" for c in Xd.columns]
    Xa = Xd.to_numpy()
    if add_const:
        Xa = np.column_stack([np.ones(len(Xa)), Xa])
        names = ["alpha"] + names
    ya = ys.to_numpy(float)
    T, k = Xa.shape
    valid = ~np.isnan(ya) & ~np.isnan(Xa).any(axis=1)
    Xz = np.where(valid[:, None], Xa, 0.0)
    yz = np.where(valid, ya, 0.0)
    mp = min_periods or window
    # rolling sums of cross products (pandas rolling sum uses compensated summation)
    prods = {}
    for i in range(k):
        for j in range(i, k):
            prods[(i, j)] = pd.Series(Xz[:, i] * Xz[:, j]).rolling(window, min_periods=1).sum().to_numpy()
    xy = [pd.Series(Xz[:, i] * yz).rolling(window, min_periods=1).sum().to_numpy() for i in range(k)]
    yy = pd.Series(yz * yz).rolling(window, min_periods=1).sum().to_numpy()
    cnt = pd.Series(valid.astype(float)).rolling(window, min_periods=1).sum().to_numpy()
    Sxx = np.empty((T, k, k))
    for (i, j), v in prods.items():
        Sxx[:, i, j] = v
        Sxx[:, j, i] = v
    Sxy = np.column_stack(xy)
    beta = np.full((T, k), np.nan)
    ok = cnt >= max(mp, k + 1)
    if ok.any():
        A = Sxx[ok]
        b = Sxy[ok]
        try:
            sol = np.linalg.solve(A, b[..., None])[..., 0]
        except np.linalg.LinAlgError:
            sol = np.einsum("tij,tj->ti", np.linalg.pinv(A), b)
        beta[ok] = sol
    fitted = np.einsum("ti,ti->t", Xa, beta)
    resid = ya - fitted
    beta_prev = np.vstack([np.full((1, k), np.nan), beta[:-1]])
    resid_oos = ya - np.einsum("ti,ti->t", Xa, beta_prev)
    sse = yy - 2 * np.einsum("ti,ti->t", beta, Sxy) + np.einsum("ti,tij,tj->t", beta, Sxx, beta)
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = np.sqrt(np.clip(sse, 0, None) / (cnt - k))
    out = pd.DataFrame(beta, index=ys.index, columns=names)
    out["resid"] = resid
    out["resid_oos"] = resid_oos
    out["sigma"] = np.where(ok, sigma, np.nan)
    out["n"] = cnt
    return out


def rolling_beta(y, x, window: int = 252, min_periods: int | None = None) -> pd.Series:
    """Rolling OLS beta of y on x (with intercept)."""
    r = rolling_ols(y, x, window, True, min_periods)
    return r.iloc[:, 1].rename("beta")


# ---------------------------------------------------------------- Kalman hedge
def kalman_hedge(y, x, q_ratio: float = 1e-3, obs_var: float | None = None, intercept: bool = True,
                 warmup: int = 60, p0: float = 1.0) -> pd.DataFrame:
    """Kalman-filter dynamic hedge ratio for y_t = alpha_t + beta_t x_t + v_t.

    State (beta, alpha) follows a random walk with covariance Q = q_ratio * R * I,
    R = obs_var (observation-noise variance). If obs_var is None, R and the
    initial state are estimated by OLS on the first ``warmup`` observations
    and outputs start at index ``warmup`` (so no output uses its own data).

    For each t the filter (1) predicts with the prior state from t-1,
    (2) records the one-step-ahead innovation e_t = y_t - H_t state_{t|t-1},
    its variance S_t and z_t = e_t/sqrt(S_t), then (3) updates the state with
    y_t. Columns: beta_prior, alpha_prior, innovation, innovation_var, z,
    beta, alpha (posterior, known at t's close; trade at the next open).
    """
    ya, xa = as_float_array(y), as_float_array(x)
    T = ya.shape[0]
    k = 2 if intercept else 1
    cols = {c: np.full(T, np.nan) for c in ("beta_prior", "alpha_prior", "innovation", "innovation_var", "z",
                                           "beta", "alpha")}
    start = 0
    if obs_var is None:
        w = slice(0, warmup)
        ok = ~np.isnan(ya[w]) & ~np.isnan(xa[w])
        if ok.sum() < k + 2:
            return pd.DataFrame(cols, index=index_of(y, x))
        Xw = np.column_stack([xa[w][ok]] + ([np.ones(ok.sum())] if intercept else []))
        coef, *_ = np.linalg.lstsq(Xw, ya[w][ok], rcond=None)
        res = ya[w][ok] - Xw @ coef
        R = float(res.var(ddof=k))
        state = coef.astype(float)
        P = R * np.linalg.pinv(Xw.T @ Xw)  # OLS coefficient covariance as the prior
        start = warmup
    else:
        R = float(obs_var)
        state = np.zeros(k)
        P = np.eye(k) * p0
    Q = np.eye(k) * q_ratio * R
    for t in range(start, T):
        P = P + Q  # predict (random walk: state unchanged)
        if np.isnan(ya[t]) or np.isnan(xa[t]):
            continue
        H = np.array([xa[t], 1.0]) if intercept else np.array([xa[t]])
        yhat = H @ state
        e = ya[t] - yhat
        S = float(H @ P @ H + R)
        cols["beta_prior"][t] = state[0]
        if intercept:
            cols["alpha_prior"][t] = state[1]
        cols["innovation"][t] = e
        cols["innovation_var"][t] = S
        cols["z"][t] = e / np.sqrt(S)
        K = P @ H / S
        state = state + K * e
        I_KH = np.eye(k) - np.outer(K, H)
        P = I_KH @ P @ I_KH.T + np.outer(K, K) * R  # Joseph form (numerically stable)
        cols["beta"][t] = state[0]
        if intercept:
            cols["alpha"][t] = state[1]
    return pd.DataFrame(cols, index=index_of(y, x))


# ---------------------------------------------------------------- cointegration
@dataclass
class EGResult:
    alpha: float
    beta: float
    adf_stat: float
    pvalue: float
    crit_values: dict
    resid: pd.Series
    nobs: int


def adf(x, regression: str = "c", autolag: str | None = "AIC", maxlag: int | None = None) -> dict:
    """Augmented Dickey-Fuller test (statsmodels). Returns stat, pvalue, usedlag, nobs, crit."""
    from statsmodels.tsa.stattools import adfuller

    a = pd.Series(as_float_array(x)).dropna().to_numpy()
    try:
        r = adfuller(a, maxlag=maxlag, regression=regression, autolag=autolag, result_object=False)
    except TypeError:  # older statsmodels without result_object
        r = adfuller(a, maxlag=maxlag, regression=regression, autolag=autolag)
    return {"stat": float(r[0]), "pvalue": float(r[1]), "usedlag": int(r[2]), "nobs": int(r[3]), "crit": r[4]}


def engle_granger(y, x, trend: str = "c", autolag: str | None = "aic", maxlag: int | None = None) -> EGResult:
    """Engle-Granger cointegration test on aligned (NaN-dropped) series.
    The hedge beta is the formation-window OLS slope of y on x (+const)."""
    from statsmodels.tsa.stattools import coint

    ys = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y, float))
    xs = x if isinstance(x, pd.Series) else pd.Series(np.asarray(x, float), index=ys.index)
    df = pd.concat([ys.rename("y"), xs.rename("x")], axis=1).dropna()
    X = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, df["y"].to_numpy(), rcond=None)
    resid = df["y"] - X @ coef
    stat, p, crit = coint(df["y"].to_numpy(), df["x"].to_numpy(), trend=trend, autolag=autolag, maxlag=maxlag)
    return EGResult(alpha=float(coef[0]), beta=float(coef[1]), adf_stat=float(stat), pvalue=float(p),
                    crit_values={"1%": crit[0], "5%": crit[1], "10%": crit[2]}, resid=resid, nobs=len(df))


# ---------------------------------------------------------------- PCA residuals
def pca_residuals(rets: pd.DataFrame, window: int = 252, n_components: int = 5, standardize: bool = True,
                  refit_every: int = 1, min_coverage: float = 1.0) -> pd.DataFrame:
    """Residual returns after removing the top-k principal components.

    At each t >= window the loadings are estimated from rows [t-window, t)
    (i.e. through t-1) using only columns with >= min_coverage non-NaN share in
    that window (missing values in kept columns are filled with the column's
    window mean, i.e. zero after standardising). r_t is standardised with the
    window mean/stdev, projected on the k eigenvectors of the correlation
    matrix (covariance if standardize=False), and the residual is mapped back
    to return units. ``refit_every`` > 1 reuses the last fit for that many rows
    (still only past data). Returns a frame shaped like ``rets``.
    """
    R = rets.to_numpy(float)
    T, N = R.shape
    out = np.full((T, N), np.nan)
    fit = None
    for t in range(window, T):
        if fit is None or (t - fit["t"]) >= refit_every:
            W = R[t - window:t]
            cov_ok = np.mean(~np.isnan(W), axis=0) >= min_coverage
            cols = np.flatnonzero(cov_ok)
            if cols.size <= n_components:
                fit = None
                continue
            Wc = W[:, cols]
            mu = np.nanmean(Wc, axis=0)
            sd = np.nanstd(Wc, axis=0, ddof=1) if standardize else np.ones(cols.size)
            sd = np.where(sd > 0, sd, np.nan)
            Z = (Wc - mu) / sd
            Z = np.where(np.isnan(Z), 0.0, Z)
            C = (Z.T @ Z) / (window - 1)
            vals, vecs = np.linalg.eigh(C)
            E = vecs[:, ::-1][:, :n_components]
            fit = {"t": t, "cols": cols, "mu": mu, "sd": sd, "E": E}
        if fit is None:
            continue
        cols, mu, sd, E = fit["cols"], fit["mu"], fit["sd"], fit["E"]
        r = R[t, cols]
        z = (r - mu) / sd
        ok = ~np.isnan(z)
        if ok.sum() <= n_components:
            continue
        # factor returns from available names (least squares on the loadings)
        f, *_ = np.linalg.lstsq(E[ok], z[ok], rcond=None)
        zr = z - E @ f
        out[t, cols] = np.where(ok, zr * sd, np.nan)  # residual net of window mean
    return pd.DataFrame(out, index=rets.index, columns=rets.columns)


# ---------------------------------------------------------------- distance pairs
def normalized_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Divide each column by its first non-NaN value in the window."""
    first = prices.apply(lambda s: s.loc[s.first_valid_index()] if s.first_valid_index() is not None else np.nan)
    return prices / first


def ssd_pairs(prices: pd.DataFrame, groups: dict | pd.Series | None = None) -> pd.DataFrame:
    """Sum of squared deviations between normalised price paths for all pairs
    with complete data in the formation window (pass the formation slice of
    total-return prices). ``groups`` restricts to same-group pairs.
    Returns DataFrame [a, b, ssd, spread_std] sorted by ssd ascending."""
    P = prices.dropna(axis=1, how="any")
    Nn = normalized_prices(P)
    A = Nn.to_numpy()
    names = list(Nn.columns)
    gmap = dict(groups) if groups is not None else None
    rows = []
    for i, j in combinations(range(len(names)), 2):
        if gmap is not None and gmap.get(names[i]) != gmap.get(names[j]):
            continue
        d = A[:, i] - A[:, j]
        rows.append((names[i], names[j], float(np.sum(d * d)), float(np.std(d, ddof=1))))
    df = pd.DataFrame(rows, columns=["a", "b", "ssd", "spread_std"])
    return df.sort_values(["ssd", "a", "b"], kind="mergesort").reset_index(drop=True)


def select_pairs(ssd: pd.DataFrame, n: int = 20, nonoverlapping: bool = True) -> pd.DataFrame:
    """Greedy pick of the n smallest-SSD pairs; with nonoverlapping=True each
    symbol appears in at most one pair."""
    used: set = set()
    keep = []
    for idx, r in ssd.iterrows():
        if nonoverlapping and (r["a"] in used or r["b"] in used):
            continue
        keep.append(idx)
        used.update([r["a"], r["b"]])
        if len(keep) >= n:
            break
    return ssd.loc[keep].reset_index(drop=True)


def pair_spread_z(pa: pd.Series, pb: pd.Series, formation_start, formation_end) -> pd.DataFrame:
    """Normalised-price spread standardised with FORMATION-window moments.

    Both legs are normalised by their price at formation_start; spread =
    na - nb; z = (spread - mean_f)/std_f with mean/std frozen at formation_end
    (Gatev et al. 2006). Rows <= formation_end are in-sample (flagged)."""
    df = pd.concat([pa.rename("a"), pb.rename("b")], axis=1)
    base = df.loc[formation_start:formation_end].dropna()
    na = df["a"] / base["a"].iloc[0]
    nb = df["b"] / base["b"].iloc[0]
    sp = na - nb
    f = sp.loc[formation_start:formation_end]
    mu, sd = float(f.mean()), float(f.std(ddof=1))
    out = pd.DataFrame({"spread": sp, "z": (sp - mu) / sd})
    out["in_formation"] = out.index <= formation_end
    return out
