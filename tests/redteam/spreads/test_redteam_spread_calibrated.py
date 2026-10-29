"""Red-team tests for atlasbt/execution/spread_calibrated.py (calibrated half-spread model).

Convention: tests marked ``xfail(strict=True)`` assert the behaviour the model SHOULD have and
currently fail (each one documents a defect; a fix turns it into an XPASS -> failure, so remove
the marker then). Unmarked tests pass and pin down evidence (numbers quoted in the red-team report)
or confirm what is sound.

External evidence used (not in the repo):
  * SEC Release 34-38571 (May 5, 1997, SR-Amex-97-14), fn 6: "SPDRs and S&P MidCap 400 SPDRs will
    continue to trade in 1/64s"; Amex stocks $0.25-$10 in 1/16 since 1992/1995.
  * Jones (2002) "A Century of Stock Market Liquidity and Trading Costs" p.22: DJIA time-weighted
    NBBO proportional quoted spread averaged 0.181% (full) during 2000 (1/16 tick).
  * SEC Reg NMS (70 FR 45529, 2005-08-08): Rule 612 compliance date extended to 2006-01-31.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atlasbt.execution.spread_calibrated import (CalibratedSpreadConfig, CalibratedSpreadModel,
                                                 era_tick)

ROOT = Path(__file__).resolve().parents[3]
CALDIR = ROOT / "atlasbt" / "execution" / "calibration"
QLIB_ZIP = ROOT / "data_raw" / "qlib_us_1d" / "qlib_data_us_1d_latest.zip"


@pytest.fixture(scope="module")
def model():
    return CalibratedSpreadModel()


@pytest.fixture(scope="module")
def build():
    from atlasbt.execution.calibration import build_calibration as bc
    with contextlib.redirect_stdout(io.StringIO()):
        cal, rep = bc.main(["--oos-days", "20260702"])           # no --write: nothing on disk changes
    return bc, cal, rep


def _qlib(symbols):
    """Raw-ish daily series from the qlib (Yahoo) archive: close/factor, volume*factor."""
    if not QLIB_ZIP.exists():
        pytest.skip("qlib archive not present")
    out = {}
    with zipfile.ZipFile(QLIB_ZIP) as z:
        cal = pd.to_datetime(z.read("calendars/day.txt").decode().split())

        def load(s, f):
            a = np.frombuffer(z.read(f"features/{s}/{f}.day.bin"), dtype="<f4")
            i = int(a[0])
            return pd.Series(a[1:], index=cal[i:i + len(a) - 1]).astype(float)
        for s in symbols:
            c, fa, v = load(s, "close"), load(s, "factor"), load(s, "volume")
            out[s] = (c, c / fa, v * fa)
    return out


# ----------------------------------------------------------------------------- (2) reproducibility
def test_validation_report_reproducible_from_build_script(build):
    """SOUND: spread_calibration.json and validation_report.json are reproduced to 1e-3 relative
    from build_calibration.py + the shipped per-symbol datasets (the TAQ/IEX extraction itself
    cannot be re-run: raw files were deleted)."""
    _, cal, rep = build
    shipped_c = json.loads((CALDIR / "spread_calibration.json").read_text())
    shipped_r = json.loads((CALDIR / "validation_report.json").read_text())
    for t, b in shipped_c["beta"].items():
        assert cal["beta"][t] == pytest.approx(b, abs=1e-5)
    assert cal["q"] == pytest.approx(shipped_c["q"], rel=1e-5)
    assert cal["era_anchors"] == shipped_c["era_anchors"]
    for k in ("in_sample", "holdout_symbols"):
        assert rep[k]["mae_bp"] == pytest.approx(shipped_r[k]["mae_bp"], rel=1e-6)
    assert rep["oos_days"]["20260702"]["model"]["mae_bp"] == pytest.approx(
        shipped_r["oos_days"]["20260702"]["model"]["mae_bp"], rel=1e-6)


def test_next_day_oos_is_mostly_symbol_persistence(model):
    """The 'out-of-sample day' (07-02) shares ~86% of its log residual with 07-01 per symbol, so the
    all-symbol next-day MAE is largely an in-sample statistic; only the hash hold-out is informative."""
    from atlasbt.execution.calibration.build_calibration import load, prepare
    a, b = prepare(load("nbbo", "20260701")), prepare(load("nbbo", "20260702"))

    def resid(df, d):
        h, _ = model.half_spread_vec(np.full(len(df), np.datetime64(d)), df.prev_close.values,
                                     df.adv_dollar.values, df.sigma.values, df.etf.values)
        return pd.Series(np.log(1e4 * h / df.tw_hs_bp.values), index=df.symbol.values)
    ra, rb = resid(a, "2026-07-01"), resid(b, "2026-07-02")
    common = ra.index.intersection(rb.index)
    corr = np.corrcoef(ra.loc[common].groupby(level=0).first(), rb.loc[common].groupby(level=0).first())[0, 1]
    assert corr > 0.8


# ----------------------------------------------------------------------------- (1) historical evidence
def test_lean_spy_out_of_time_checks_are_tick_bound(model):
    """The Lean SPY 2013 'out-of-time check' cannot falsify the 2010+ == 2026 assumption: scaling the
    latent spread by anything from 0.1x to 3x leaves the prediction at 0.296 bp (one tick)."""
    preds = []
    for mult in (0.1, 0.5, 1.0, 2.0, 3.0):
        L = model._latent_frac(np.array([19336350113.0]), np.array([0.00644]), np.array([168.89]),
                               np.array([1.0]))[0] * mult
        k, p = 0.01, 168.89
        S = k * (1 + (L * p / k) ** model._q) ** (1 / model._q)
        preds.append(1e4 * S / (2 * p))
    assert max(preds) - min(preds) < 0.005


def test_jones_2000_djia_level_matches():
    """SOUND (independent check): DJIA members over 2000 (1/16 tick, NYSE; INTC/MSFT Nasdaq),
    PIT inputs from qlib with raw prices repaired for post-2000 splits. Jones (2002): 0.181% full.
    Model: ~0.181% (0.166% if the Furfine anchor is refit with correct XOM/KO prices)."""
    fix = {"cat": 2, "c": 0.1, "ko": 2, "xom": 2, "jnj": 2, "msft": 2, "mmm": 2, "pg": 2}
    names = ["axp", "t", "ba", "cat", "c", "ko", "dis", "xom", "ge", "hd", "hpq", "hon", "intc", "ibm",
             "ip", "jpm", "jnj", "mcd", "mrk", "msft", "mmm", "mo", "pg", "wmt"]
    data = _qlib(names)
    mN = CalibratedSpreadModel(CalibratedSpreadConfig(listing="NYSE"))
    mQ = CalibratedSpreadModel(CalibratedSpreadConfig(listing="NASDAQ"))
    per = []
    for s in names:
        c, raw, vol = data[s]
        raw, vol = raw * fix.get(s, 1), vol / fix.get(s, 1)
        adv = (raw * vol).rolling(20, min_periods=10).mean().shift(1)
        sig = np.log(c).diff().rolling(20, min_periods=10).std().shift(1)
        m = (c.index >= "2000-02-01") & (c.index <= "2000-12-29")
        mod = mQ if s in ("intc", "msft") else mN
        h, _ = mod.half_spread_vec(c.index[m].values, raw.shift(1)[m].values, adv[m].values, sig[m].values,
                                   np.zeros(m.sum()))
        per.append(2 * h.mean())
    assert 100 * np.mean(per) == pytest.approx(0.181, rel=0.2)


def test_furfine_anchor_inputs_use_split_adjusted_prices():
    """historical_checks.csv feeds XOM/KO at ~$41/$28 on 2001-01-16; their raw closes were ~$87/$61
    (XOM 2:1 split 2001-07-19, KO 2:1 2012-08-13 are baked into qlib's 'raw' = close/factor). The
    2001 era level a01 is solved from these rows (brentq on the 7.0c group mean)."""
    hc = pd.read_csv(CALDIR / "historical_checks.csv")
    post = hc[hc.check == "furfine_2001_post_decimal"].set_index("symbol")
    assert post.loc["XOM", "prev_close"] < 45 and post.loc["KO", "prev_close"] < 30   # i.e. half the raw price


def test_furfine_anchor_shift_when_prices_repaired(build):
    bc, cal, _ = build
    center = cal["center"]
    th = np.array([cal["beta"][t] for t in bc.TERMS] + [cal["c_d"], math.log(cal["q"])])
    a03, b03 = cal["era_anchors"][2][1], cal["era_anchors"][2][2]
    hc = pd.read_csv(CALDIR / "historical_checks.csv")
    hc["etf"] = hc.etf.astype(float)
    post = hc[hc.check == "furfine_2001_post_decimal"].copy()
    from scipy.optimize import brentq

    def anchor(df):
        FX = bc.design(df, center)
        fp, ftk = df.prev_close.to_numpy(), df.tick.to_numpy()
        return a03 + brentq(lambda d: float(np.mean(bc.model_half(th, FX, fp, ftk, np.exp(a03 + d + b03 * FX[:, 1]))
                                                    * 2 * fp * 100)) - 7.0, -3, 3)
    a_ship = anchor(post)
    post.loc[post.symbol.isin(["XOM", "KO"]), "prev_close"] *= 2
    a_fix = anchor(post)
    assert a_ship == pytest.approx(cal["era_anchors"][0][1], abs=1e-3)
    assert a_fix - a_ship == pytest.approx(-0.153, abs=0.02)     # 2001-and-earlier latent level -14%


@pytest.mark.xfail(strict=True, reason="era b>0 + linear 2003->2010 ramp: at fixed (ADV$, sigma, P) small-cap "
                                       "spreads RISE 2-4x from 2003 to 2010 (contradicts Hasbrouck 2009 Fig.2, "
                                       "Knight R2000 median spreads flat/down 2003-07)")
def test_small_cap_cost_should_not_rise_2003_to_2010_at_fixed_characteristics(model):
    for adv in (5e4, 2e5, 1e6):
        h03, _ = model.half_spread(None, "2003-04-01", 8.0, adv, 0.035, "STK")
        h10, _ = model.half_spread(None, "2010-01-04", 8.0, adv, 0.035, "STK")
        assert h03 >= 0.9 * h10


def test_era_multiplier_values_pinned(model):
    """Evidence for the report: era multipliers (latent, vs 2026) at fixed ADV$."""
    m = lambda d, a: float(model.era_multiplier([d], [a])[0])
    assert m("2003-04-01", 1e5) == pytest.approx(0.27, abs=0.02)
    assert m("1995-06-01", 1e6) == pytest.approx(0.56, abs=0.02)
    assert m("2003-04-01", 1e10) == pytest.approx(2.96, abs=0.05)


# ----------------------------------------------------------------------------- (4) tick regimes
@pytest.mark.xfail(strict=True, reason="SPDRs traded in 1/64 on Amex (SEC Rel. 34-38571); model floors SPY at "
                                       "1/8 (pre-1997-06-24) / 1/16 (to 2001-01-29)")
def test_spy_pre_decimal_floor_should_be_one_sixty_fourth(model):
    data = _qlib(["spy"])
    c, raw, vol = data["spy"]
    adv = (raw * vol).rolling(20).mean().shift(1)
    sig = np.log(c).diff().rolling(20).std().shift(1)
    m = (c.index >= "2000-02-01") & (c.index < "2001-01-26")
    args = (c.index[m].values, raw.shift(1)[m].values, adv[m].values, sig[m].values, np.ones(m.sum()))
    h_model, _ = model.half_spread_vec(*args)
    h_164, _ = model.half_spread_vec(*args, tick=1 / 64)
    # currently 2.20 bp vs 0.65 bp (3.4x)
    assert h_model.mean() <= 1.5 * h_164.mean()


@pytest.mark.xfail(strict=True, reason="Rule 612 compliance date was 2006-01-31, not 2005-08-29")
def test_rule_612_compliance_date():
    assert era_tick("2005-10-03", 0.50) == 0.01


# ----------------------------------------------------------------------------- (5) edge cases
@pytest.mark.xfail(strict=True, reason="series() uses the UNADJUSTED prior raw close on a split day "
                                       "(DailyStats.prior_close is split-adjusted but not passed)")
def test_split_day_spread_uses_split_adjusted_price(model):
    n = 40
    rng = np.random.default_rng(0)
    noise = 1 + 0.01 * rng.standard_normal(n)
    rc = np.where(np.arange(n) < 30, 500.0, 50.0) * noise          # 10:1 split effective session 30
    vol = np.where(np.arange(n) < 30, 2e5, 2e6)
    adj = np.full(n, 50.0) * noise
    dates = pd.bdate_range("2024-01-02", periods=n)
    ss = model.series(adj * 1.01, adj * 0.99, adj, raw_close=rc, volume=vol, dates=dates, asset_class="STK")
    # currently 33.4 bp on the split day vs 7.3 bp the next day (4.6x)
    assert ss.est[30] == pytest.approx(ss.est[31], rel=0.25)


def test_stale_or_new_listing_falls_back_to_10bp_default(model):
    """Evidence: zero ADV, NaN history (IPO first sessions) and zero sigma (stale/halted price) all
    get the size-blind 10 bp half-spread default, whatever the stock."""
    assert model.half_spread(None, "2024-01-02", 3.0, 0.0, 0.02, "STK") == (0.001, "default")
    assert model.half_spread(None, "2024-01-02", 3.0, np.nan, np.nan, "STK") == (0.001, "default")
    c = np.full(40, 4.0)
    ss = model.series(c, c, c, raw_close=c, volume=np.full(40, 1e4), dates=pd.bdate_range("2024-01-02", periods=40))
    assert ss.src[-1] == "default"


def test_etf_flag_is_worth_4_to_5x(model):
    """Evidence: a dead/misflagged ETF priced as a stock costs 3.9x (SPY) to 5.3x (small ETF)."""
    r_spy = (model.half_spread(None, "2026-07-01", 746.77, 4.714e10, 0.01143, "STK")[0]
             / model.half_spread(None, "2026-07-01", 746.77, 4.714e10, 0.01143, "ETF")[0])
    r_small = (model.half_spread(None, "2026-07-01", 30, 1e6, 0.012, "STK")[0]
               / model.half_spread(None, "2026-07-01", 30, 1e6, 0.012, "ETF")[0])
    assert 3.5 < r_spy < 4.5 and 4.5 < r_small < 6


def test_liquid_high_vol_etfs_overstated(model):
    """Evidence: in the liquid-ETF regime the sigma elasticity is ~2.0 (0.96-0.49+1.53) and price ~+1.0;
    in-sample high-priced leveraged/sector ETFs are overstated 2-8x (SOXL 2.7x, SOXX 2.8x, MUU 8.1x)."""
    from atlasbt.execution.calibration.build_calibration import load, prepare
    nb = prepare(load("nbbo", "20260701")).set_index("symbol")
    ratios = {}
    for s in ("SOXL", "SOXX", "SMH", "MUU"):
        r = nb.loc[s]
        h, _ = model.half_spread(s, "2026-07-01", r.prev_close, r.adv_dollar, r.sigma, "ETF")
        ratios[s] = 1e4 * h / r.tw_hs_bp
    assert min(ratios.values()) > 2.0


# ----------------------------------------------------------------------------- (6) monotonicity
@pytest.mark.xfail(strict=True, reason="liquid-ETF logistic regime (e_s_ls, e_s_lp) makes the ETF spread RISE "
                                       "with ADV$ across the $100M knot for high-price/high-vol ETFs")
def test_etf_spread_monotone_decreasing_in_adv():
    m = CalibratedSpreadModel(CalibratedSpreadConfig(tick_floor=False))
    advs = np.exp(np.linspace(np.log(1e6), np.log(1e9), 300))
    h, _ = m.half_spread_vec(np.full(300, np.datetime64("2026-07-01")), np.full(300, 300.0), advs,
                             np.full(300, 0.08), np.ones(300))
    assert np.all(np.diff(h) <= 1e-15)


# ----------------------------------------------------------------------------- (7) time of day
def _bucket_profile(day, etf_only):
    from atlasbt.execution.calibration.build_calibration import load, prepare
    nb = prepare(load("nbbo", day))
    bk = nb[[f"b{k:02d}" for k in range(78)]].to_numpy(float)
    ok = np.isfinite(bk).all(1) & (nb.adv_dollar.to_numpy() >= 1e8)
    ok &= (nb.etf.to_numpy() == 1) if etf_only else (nb.etf.to_numpy() == 0)
    p = np.median(bk[ok] / nb.tw_hs_bp.to_numpy()[ok, None], 0)
    return p / p.mean()


def test_tod_close_narrower_replicates_next_day_for_stocks():
    """SOUND: on 2026-07-02 (not used for the profile) liquid stocks are 4.45x at 09:30 and 0.44x
    at 15:55, matching the shipped 4.30 / 0.51."""
    p = _bucket_profile("20260702", etf_only=False)
    assert p[0] > 3.5 and p[-1] < 0.6


@pytest.mark.xfail(strict=True, reason="'liquid' profile is stock-dominated (1104 of 1411); liquid ETFs are "
                                       "~1.9x at the open and ~0.97x at the close, not 4.3x / 0.51x")
def test_tod_profile_fits_liquid_etfs(model):
    p = _bucket_profile("20260702", etf_only=True)
    shipped_open = model.tod_multiplier(0, 390, 1e10)
    shipped_close = model.tod_multiplier(388, 2, 1e10)
    assert shipped_open == pytest.approx(p[0], rel=0.3)
    assert shipped_close == pytest.approx(p[-1], rel=0.3)
