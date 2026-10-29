# Calibrated bid-ask spreads (`atlasbt/execution/spread_calibrated.py`)

Owner: SPREAD CALIBRATION agent. Status: built and validated on 2026-10-29. It is not wired in yet;
the ENGINE owner wires it (section 8). Fidelity of every fill priced with it: **T2, "spread modelled
(calibrated)"**.

## 1. Problem

In bar mode the engine charged `half-spread = Corwin-Schultz (2012) 21-day estimate / 2`, falling back
to Abdi-Ranaldo (2017). Both estimators back the spread out of the daily high-low range. For liquid
names the range is almost all volatility, so the estimate measures volatility rather than the spread.
On 2026-07-01 the prior-21-day CS estimate is compared below with the consolidated NBBO from the NYSE
Daily TAQ sample. In both tables, "half bp" is the RTH time-weighted mean half spread in basis points
of the mid.

| symbol (2026-07-01 inputs) | NBBO observed, half bp | CS (old engine), half bp | calibrated model, half bp |
|---|---|---|---|
| SPY (P $747, ADV $47.14B, σ 1.1%) | 0.130 | 17.4 | 0.121 |
| QQQ (P $736, ADV $38.41B, σ 2.1%) | 0.266 | 23.7 | 0.457 |
| IWM (P $300, ADV $8.84B, σ 1.4%) | 0.305 | 27.1 | 0.180 |
| DIA (P $522, ADV $2.87B, σ 0.9%) | 0.353 | 20.3 | 0.204 |
| EEM (P $68, ADV $2.16B, σ 2.6%) | 0.747 | 19.6 | 0.731 |
| TLT (P $86, ADV $2.17B, σ 0.6%) | 0.584 | 5.0 | 0.579 |
| GLD (P $368, ADV $3.22B, σ 1.9%) | 0.596 | 9.1 | 0.565 |
| XLF (P $54, ADV $1.99B, σ 0.9%) | 0.915 | 19.9 | 0.933 |
| AAPL (P $289, ADV $19.82B, σ 2.2%) | 0.597 | 52.6 | 0.824 |
| NVDA (P $200, ADV $33.09B, σ 2.7%) | 0.511 | 45.7 | 0.550 |
| MSFT (P $373, ADV $18.60B, σ 2.4%) | 1.000 | 27.7 | 1.090 |
| AMZN (P $238, ADV $15.07B, σ 2.4%) | 0.941 | 41.5 | 0.910 |
| JPM (P $327, ADV $3.19B, σ 1.6%) | 1.496 | 38.7 | 1.949 |
| INTC (P $140, ADV $16.01B, σ 6.0%) | 1.407 | 68.8 | 1.490 |
| BAC (P $57, ADV $2.15B, σ 1.2%) | 0.888 | 33.1 | 0.896 |
| T (P $21, ADV $1.41B, σ 2.4%) | 2.540 | 29.5 | 2.416 |
| F (P $14, ADV $0.81B, σ 1.9%) | 3.696 | 32.6 | 3.597 |
| AA (P $52, ADV $0.38B, σ 4.3%) | 4.009 | 37.0 | 5.779 |

Across the out-of-sample symbols (section 5), CS overstates the spread by a factor of
e^2.98 ≈ 20× for names with ADV ≥ $1B and ≈ 8× for $100M–1B. Over a round trip,
the old engine charged SPY about 35 bp, against about 0.24 bp from the model and 0.26 bp from the NBBO. For a strategy
trading ETFs daily that drag is up to several %/yr.

## 2. Data (all real quotes; nothing synthetic)

| Source | What | Used for |
|---|---|---|
| **NYSE Daily TAQ sample** `ftp.nyse.com/Historical Data Samples/DAILY TAQ/EQY_US_ALL_NBBO_20260701.gz` (9.6 GB) + `EQY_US_ALL_TRADE_20260701.gz` (4.4 GB) + `EQY_US_ALL_REF_MASTER_20260701.gz` | Consolidated **NBBO** for every US-listed security, full day, ns timestamps. Every trade. Security type (common / ADR / CEF / ETF / ETN ...) | **Fit** (2026-07-01) |
| Same, `EQY_US_ALL_NBBO_20260702.gz` (12.7 GB) | Next day's NBBO | Out-of-sample day (section 5) |
| **IEX HIST TOPS** pcaps (`iextrading.com/api/1.0/hist`), 12 days: 2017-03-22, 2018-03-14, 2019-03-13, 2020-03-18, 2020-09-23, 2021-09-15, 2022-09-14, 2023-08-03, 2024-09-11, 2025-09-10, 2026-01-15, 2026-07-01 (0.8–11.7 GB each) | IEX top of book (one venue) | IEX-vs-NBBO comparison and per-year diagnostics. **Not used in the fit** (section 6) |
| QuantConnect Lean fixtures `data/equity/usa/minute/spy/*_quote.zip` | SPY minute NBBO bars, 6 days in Oct 2013 and 2023-08-03 | Out-of-time checks |
| Literature (TAQ based) | Hendershott, Jones & Menkveld (2011, *JF* 66(1)) Table I: NYSE Dec 2002–Jul 2003 by market-cap quintile. Quoted and effective half spreads, $ volume, volatility, price. Furfine (2003, Chicago Fed *Economic Perspectives* Q4): NYSE most-active stocks, mean quoted spread 11.3¢ in the 2 weeks before decimalisation (1/16 tick), 7.0¢ after. Chordia, Roll & Subrahmanyam (2001, AIMR monograph *Common Determinants of Liquidity and Trading*) Table 1: NYSE 1992 mean / median quoted spread $0.32 / $0.27, proportional 1.60% / 1.15% | Era anchors and checks |
| PIT daily features | DoltHub `stocks.ohlcv` (HF CSV mirror to 2026-01-28; SQL API for 2026-04-15..07-02), Yahoo daily (SPY/QQQ/IWM; store), qlib/Yahoo (2001 NYSE names) | ADV$, σ, prior close (sessions strictly before the quote day) |

The tools are small C stream parsers (TAQ pipe text, IEX-TP pcapng) plus a feature builder. They live
in the agent's scratchpad, and their outputs are the per-symbol tables in `calibration/`. Pcaps and
TAQ files were streamed or deleted after extraction, peaking at about 14 GB transient (TAQ files on disk while parsing). The user agent
was always `atlasbt-research/0.1`. Downloads were plain HTTP or range requests with pauses, with no
bot-check circumvention.

Per-symbol statistics (`calibration/nbbo_<day>.csv.gz`, `iex_<day>.csv.gz`) are computed over RTH
09:30–16:00 ET:

* `tw_hs_bp` is the time-weighted mean of the relative half spread, `(ask − bid) / (2·mid)`, over
  valid two-sided quotes. Crossed or locked quotes and spreads above 50% are excluded. This is the
  **target**: the expected cost of a marketable order at a random time.
* `tw_med_hs_bp` is the time-weighted median, and the 78 five-minute bucket means `b00..b77` give
  the intraday profile.
* `eff_hs_bp_dvw` is the dollar-volume-weighted effective half spread, `|p − mid| / mid` against
  the prevailing NBBO. Regular-way RTH trades only (sale conditions `TUZLCNRWPBGHMQO4567893V`
  excluded). The `lit_` variant drops TRF (off-exchange) prints.
* Features are joined on the prior-session symbol: `prev_close` (raw), `adv_dollar` (mean
  close × volume over the prior 20 sessions), `sigma` (std of daily log returns over the prior 20,
  dropping |r| > 40% to remove split artefacts), and `cs_full` / `ar_full` (the old engine's 21-day
  CS/AR).

Fit universe: common stocks, ADRs, NY-registry shares and closed-end funds (stock), plus
ETF/ETS/ETV/ETN (ETF). The day must be ≥ 80% two-sided, the prior close ≥ $0.10, and the
mid / prev_close ratio within e^±0.3 (symbol-mapping guard). Warrants, rights, units, preferreds
and notes are excluded. 10,956 symbols qualify on the fit day.

## 3. Model

```
x  = b·[1, la, ls, lp, E, E·la, E·ls, E·lp, min(lp − lp($5), 0), E·s, E·s·ls, E·s·lp, E·s·(la − la($100M))]
     s  = logistic((la − la($100M)) / 0.5)   (smooth 'liquid ETF' regime)
     la = ln ADV$ − c_a,  ls = ln σ − c_s,  lp = ln P − c_p   (centred at the fit-day medians), E = ETF
L  = P · exp(x) · era(t, ADV$)                 latent (continuous) full spread, $
S  = k · (1 + (L/k + c_d)^q)^(1/q)             quoted full spread, $  (k = tick of the era)
h  = S / (2P)                                   half spread, fraction of price
```

* **Tick discreteness**: `S ≥ k` always. The smooth maximum (fitted q = 5.80, c_d = 0.00)
  reproduces the 1.00–1.05-tick average spreads of tick-constrained names (SPY, BAC, F ...) and
  the transition region. The same curve gives the pre-decimal spreads once k is set to 1/8 or 1/16.
* **Tick regimes** (`era_tick`):
  * NYSE / AMEX (the default): 1/8 for prices ≥ $1 until 1997-06-23, then 1/16 until 2001-01-26.
  * Nasdaq (`listing='NASDAQ'`): 1/8 for prices ≥ $10 (1/32 below) until 1997-05-30, then 1/16
    until 2001-04-06.
  * After decimalisation: $0.01, and $0.0001 below $1 from Reg NMS Rule 612 (2005-08-29).
  * The SEC Tick Size Pilot ($0.05 for about 1,200 small caps, 2016-10-03 to 2018-09-28) is not
    applied because membership is unknown. Pass `tick=` for those names.
* **Fit**: robust NLS (`scipy.optimize.least_squares`, soft-L1, f_scale 0.5) of `ln h` on
  `ln tw_hs`. Coefficients and centring are in `calibration/spread_calibration.json`. Rebuild
  with `python -m atlasbt.execution.calibration.build_calibration --write`.

| term | meaning | coef |
|---|---|---|
| `1` | constant | -4.8236 |
| `la` | ln ADV$ | -0.6247 |
| `ls` | ln σ (daily) | +0.9609 |
| `lp` | ln price | +0.6659 |
| `e` | ETF | -1.4279 |
| `e_la` | ETF × ln ADV$ | +0.4397 |
| `e_ls` | ETF × ln σ | -0.4944 |
| `e_lp` | ETF × ln price | -0.7432 |
| `hinge_lp_lo` | min(ln P − ln 5, 0) | -0.6374 |
| `e_s` | ETF × s, s = logistic((la − la($100M))/0.5) | -3.2168 |
| `e_s_ls` | ETF × s × ln σ | +1.5295 |
| `e_s_lp` | ETF × s × ln price | +1.1203 |
| `e_s_la` | ETF × s × (la − la($100M)) | -0.2843 |

Centres: ln ADV$ 14.734 (≈ $2.5M), ln σ -3.948 (≈ 1.93%), ln P 3.309 (≈ $27.36). Regressors are clipped to the fit-day range. The fit weight is log10(ADV$) - 5 clipped to [0.5, 5], so tradable names count more than micro caps.

Reading the coefficients:
* Spread falls with dollar volume (stocks, elasticity -0.62).
* Spread rises with volatility (+0.96).
* At fixed dollar volume, a higher price means fewer shares traded, so the relative spread rises
  (+0.67). Below $5, low prices add width beyond the tick (`hinge_lp_lo`).
* ETFs are much tighter than stocks with the same characteristics. ETFs with ADV above about $100M switch smoothly into their own regime (`e_s*`)
  with steeper volatility and price slopes. The many small ETFs would otherwise set the ETF slope.

* **Era multiplier** `era(t, ADV$) = exp(a(t) + b(t)·la)`, linear in time between anchors:

| anchor | a | b | evidence |
|---|---|---|---|
| ≤ 2001-01-29 (held back to 1990) | -0.383 | +0.207 | Furfine (2003) post-decimal level: 8 most-active NYSE names, model 6.99¢ vs 7.0¢ observed (matched by construction) |
| 2003-04 | -0.634 | +0.207 | HJM (2011) Table I quintiles Q1..Q5: observed 5.19, 6.82, 9.17, 11.68, 19.89 bp; model with 2026 coefficients 4.12, 8.85, 13.76, 20.80, 33.31; with the era term 4.65, 7.51, 9.76, 12.90, 17.16 |
| ≥ 2010-01-01 | 0 | 0 | 2026-07-01 NBBO fit |

  * 2010 onwards uses the 2026 fit unchanged. This is an assumption: spreads at a given
    nominal ADV were broadly stable in the HFT era. The out-of-time Lean SPY checks (2013, 2023)
    are consistent with it.
  * 2003: a and b are fitted to the five HJM quintiles. Spreads at a given ADV were wider for
    the largest names and narrower for small ones than in 2026.
  * 2001: the level is matched to Furfine's post-decimal 7.0¢ for 8 most-active NYSE names. The
    inputs come from Yahoo daily via qlib: GE, PFE, XOM, MRK, IBM, T, KO, WMT.
  * Before 2001 the 2001 latent level is held. The tick regime does the rest.

* **Time of day** (`tod_multiplier`) uses 78 five-minute multipliers relative to the RTH
  time-weighted mean, as medians over symbols with complete buckets and ≥ 200 trades. There are
  two profiles: ADV ≥ $100M ("liquid") and below. The first five minutes are
  4.3× (liquid) / 2.9× (illiquid) the daily average, the minutes after 10:00 are about
  1.0×, and the last five minutes are 0.51 (liquid) / 0.79 (illiquid)×. When no minute is given (daily bars) the
  multiplier is 1.0, so the expected cost is the daily average. Outside RTH it is ×3, an
  unmeasured engineering default.
* **Effective vs quoted**: `measure='effective'` multiplies by exp(a + b·la), clipped to
  [0.3, 1]. This is fitted to lit dollar-weighted effective / quoted. The median ratio is
  0.50, because many lit executions are hidden or midpoint orders. The engine default is
  **quoted**, which is conservative for a strategy that does not rest orders or get price
  improvement.
* **Corwin-Schultz is only a cap.** It applies when ADV$ is below the 2nd percentile of the fit
  universe ($9,033/day). There the log-linear fit extrapolates, and the PIT CS/AR estimate caps
  the half spread, but never below half a tick. It is not a regressor; adding it as a regressor
  improved the hold-out log-ratio SD by only about 0.02.
* **PIT**: `series()` uses, for session i, only data from sessions ≤ i−1: the prior raw close, and
  ADV$ and σ over the prior 20 sessions (min 10). This is the same convention as
  `engine.market.DailyStats`. There is a perturbation test.

## 4. API

```python
from atlasbt.execution.spread_calibrated import CalibratedSpreadModel, CalibratedSpreadConfig
m = CalibratedSpreadModel()                       # CalibratedSpreadConfig(measure='quoted', listing='NYSE', ...)
h, reason = m.half_spread("SPY", "2026-07-01", price=746.77, adv_dollars=4.7e10, daily_vol=0.0114,
                          asset_class="ETF")      # -> (fraction, "calibrated spread model (T2)")
h, reason = m.half_spread("XYZ", ts, price, adv, vol, "STK", minutes_since_open=3, minutes_to_close=387)
hs, code = m.half_spread_vec(dates, prices, advs, vols, etf_flags, cs_half=None, tick=None)   # vectorised
ss = m.series(h_adj, l_adj, c_adj, raw_close=c_raw, volume=v_raw, dates=sessions, asset_class="ETF",
              adv_dollar=None, sigma=None)        # SpreadSeries (FULL spreads, PIT), like SpreadModel.series
full = m.full_spread(ss.est[i], fill_price, tick, mso, mtc, adv_dollars=adv)   # like SpreadModel.full_spread
```

Reasons are "calibrated spread model (T2)", plus ", era-extrapolated" before 2010, or ", ADV below
calibration range". "default" means the inputs were missing. `SpreadSeries.src` values are
`calibrated`, `calibrated, era-extrapolated`, `calibrated, ADV below calibration range` and `default`.
If `series` is called without raw_close, volume or dates, it falls back to the CS series (source
`Corwin-Schultz`).

## 5. Validation (2026-07-01 unless noted)

Hold-out = a deterministic 30% of symbols (md5 of the ticker). A separate fit on the other 70% predicts them.
The shipped coefficients are refitted on all symbols. MAE is in bp of half spread. "log ratio" is ln(pred / obs).

**Overall**

| set | n | MAE bp | median AE bp | mean log ratio | SD log ratio | median abs log ratio | within ×2 |
|---|---|---|---|---|---|---|---|
| in-sample (the 70% training symbols) | 7704 | 18.89 | 4.09 | -0.03 | 0.64 | 0.34 | 77% |
| **hold-out symbols, model** | 3252 | 18.19 | 4.24 | -0.04 | 0.64 | 0.34 | 77% |
| hold-out rows with a CS value, model | 3194 | 18.27 | 4.15 | -0.05 | 0.63 | 0.33 | 77% |
| same rows, **Corwin-Schultz (old engine)** | 3194 | 38.40 | 20.30 | +0.45 | 1.72 | 1.05 | 34% |
| **20260702 (next day, out of sample), model, all symbols** | 10940 | 19.25 | 4.10 | -0.05 | 0.62 | 0.35 | 78% |
| 20260702, model, hold-out symbols only (out of sample in both day and symbol) | 3245 | 19.76 | 4.21 | -0.06 | 0.62 | 0.35 | 78% |
| 20260702, Corwin-Schultz | 10762 | 38.06 | 19.54 | +0.46 | 1.59 | 1.00 | 37% |

MAE is dominated by illiquid names, whose spreads are tens to hundreds of bp. The log-ratio columns are the scale-free measures.

**Hold-out groups (model vs CS on the same symbols)**

| group | n | model MAE bp | model mean / SD log ratio | model within ×2 | CS MAE bp | CS mean / SD log ratio | CS within ×2 |
|---|---|---|---|---|---|---|---|
| ETF | 1584 | 8.09 | -0.05 / 0.73 | 69% | 20.5 | +0.05 / 2.07 | 35% |
| stock | 1668 | 27.78 | -0.04 / 0.53 | 84% | 54.8 | +0.83 / 1.21 | 33% |
| ADV>=$1B | 56 | 1.46 | +0.13 / 0.40 | 91% | 54.3 | +2.98 / 0.61 | 0% |
| ADV $100M-1B | 337 | 2.70 | -0.05 / 0.42 | 92% | 43.3 | +2.05 / 0.84 | 4% |
| ADV $10M-100M | 712 | 6.27 | -0.01 / 0.52 | 86% | 42.7 | +1.43 / 0.81 | 18% |
| ADV $1M-10M | 890 | 14.88 | -0.12 / 0.64 | 76% | 34.1 | +0.63 / 0.84 | 44% |
| ADV <$1M | 1257 | 32.17 | -0.01 / 0.74 | 68% | 36.9 | -0.82 / 1.89 | 46% |
| price<$5 | 465 | 59.60 | -0.09 / 0.67 | 74% | 80.3 | +0.45 / 0.92 | 49% |
| price $5-20 | 710 | 18.75 | -0.07 / 0.60 | 80% | 51.0 | +0.54 / 1.51 | 28% |
| price>=$20 | 2077 | 8.72 | -0.03 / 0.64 | 76% | 24.4 | +0.42 / 1.92 | 32% |

**Named liquid symbols, leave-out refit**: the model is refitted without these 14 symbols and then predicts them (2026-07-01, half bp):

| symbol | NBBO | model (left out) | CS |
|---|---|---|---|
| AAPL | 0.597 | 0.828 | 52.6 |
| AMZN | 0.941 | 0.914 | 41.5 |
| DIA | 0.353 | 0.142 | 20.3 |
| GLD | 0.596 | 0.418 | 9.1 |
| IVV | 0.306 | 0.166 | 20.5 |
| IWM | 0.305 | 0.168 | 27.1 |
| JPM | 1.496 | 1.956 | 38.7 |
| MSFT | 1.000 | 1.095 | 27.7 |
| NVDA | 0.511 | 0.553 | 45.7 |
| QQQ | 0.266 | 0.287 | 23.7 |
| SPY | 0.130 | 0.076 | 17.4 |
| TLT | 0.584 | 0.579 | 5.0 |
| VOO | 0.261 | 0.164 | 18.4 |
| XLF | 0.915 | 0.933 | 19.9 |

**Next day 2026-07-02 (out of sample in time; PIT inputs as of 07-01 close), half bp, NBBO / model / CS**:
SPY 0.160 / 0.120 / 17.9; QQQ 0.355 / 0.457 / 24.2; IWM 0.383 / 0.180 / 26.8; DIA 0.419 / 0.203 / 20.4; AAPL 0.679 / 0.808 / 52.6;
NVDA 0.581 / 0.554 / 44.2; MSFT 1.244 / 1.124 / 27.7; TSLA 1.296 / 1.671 / 35.7; F 3.759 / 3.666 / 28.2; BAC 0.880 / 0.884 / 31.8.
Small and mid caps on 07-02:
* stocks with ADV $10M–100M (n = 1,640): MAE 9.2 bp, mean log ratio −0.11, SD 0.51
* stocks with ADV $1M–10M (n = 1,495): MAE 23.5 bp, mean −0.14, SD 0.57
* sub-$5 names (n = 1,572): MAE 58.7 bp, mean −0.05, SD 0.68

The model slightly understates the very largest high-priced index ETFs (IWM, DIA, IVV, VOO: about 0.5–0.6× when left
out of the fit). Their quotes average 1.5–3 ticks, but the liquid-ETF regime has only about 40 members. In
cost terms this is ≤ 0.2 bp per side.

**Out-of-time checks (not fitted)**

| check | observed | model |
|---|---|---|
| SPY 2013-10-04..11 (Lean minute NBBO, 6 days) | 0.298 bp half (1.00 tick) | 0.296 bp (1.00¢) |
| SPY 2023-08-03 (Lean minute NBBO) | 0.114 bp half (1.03 ticks) | 0.111 bp (1.00¢) |
| Most-active NYSE, Jan 2001, 1/16 tick (Furfine 2003) | 11.3¢ full | 9.5¢ |
| Same, after decimalisation (used for the 2001 level) | 7.0¢ | 7.0¢ |
| HJM 2003 quintiles (used for the 2003 anchor, 2 params / 5 points) | 5.19, 6.82, 9.17, 11.68, 19.89 bp | 4.65, 7.51, 9.76, 12.90, 17.16 bp |
| CRS 1992 NYSE cross-section (1/8 tick) | median $0.27 / 1.15% quoted | qualitative only: the model's floor is 1 tick = $0.125, and median-stock inputs are unavailable |

**SPY over time (model, PIT inputs from Yahoo daily)**: 1996-06 12.5¢ (1/8 tick bound), 1999-06 6.25¢ (1/16), 2001-02 1.04¢, 2003-04 1.00¢, 2008-10-15 1.01¢, 2013 1.00¢, 2020-03-18 3.5¢ (σ 5.2%), 2023 1.00¢, 2026-07-01 1.8¢ (observed 1.95¢). SPY's real pre-1997 increment on AMEX may have been finer than 1/8 (not verified), so 1990s ETF costs are conservative.

**IEX per-year diagnostics** (names with ADV ≥ $1B; the IEX median is an upper bound on NBBO):

| day | n | median IEX / model | SPY IEX median bp | SPY model bp |
|---|---|---|---|---|
| 20170322 | 15 | 2.0 | 0.8511 | 0.214 |
| 20180314 | 37 | 12.1 | 0.8913 | 0.181 |
| 20190313 | 31 | 2.9 | 0.3548 | 0.179 |
| 20200318 | 25 | 79.9 | 4.6774 | 0.700 |
| 20200923 | 34 | 61.3 | 0.9333 | 0.154 |
| 20210915 | 51 | 8.0 | 0.3388 | 0.113 |
| 20220914 | 46 | 66.3 | 1.7783 | 0.156 |
| 20230803 | 62 | 7.2 | 0.3388 | 0.111 |
| 20240911 | 48 | 17.3 | 1.9498 | 0.100 |
| 20250910 | 79 | 1.1 | 0.1549 | 0.077 |
| 20260115 | 124 | 2.8 | 0.0708 | 0.072 |
| 20260701 | 177 | 8.6 | 0.2042 | 0.121 |

The model never materially exceeds IEX for SPY. On 2026-01-15, IEX quoted SPY at one tick most of the day, and the IEX median (0.071 bp) equals the model (0.072 bp).

**Round-trip cost, SPY 2026-07-01**: old engine (CS) 2 × 17.4 = 34.9 bp. Calibrated 2 × 0.121 = 0.24 bp; the observed NBBO is 0.26 bp.

**Time-of-day profile** (multiplier vs RTH average, 5-minute bucket starting at):

| bucket | 09:30 | 09:35 | 09:40 | 09:55 | 10:25 | 11:25 | 13:25 | 15:25 | 15:40 | 15:50 | 15:55 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| liquid (ADV ≥ $100M, n=1164) | 4.30 | 2.57 | 2.17 | 1.51 | 1.14 | 0.97 | 0.81 | 0.71 | 0.66 | 0.68 | 0.51 |
| illiquid (n=5097) | 2.88 | 1.98 | 1.73 | 1.48 | 1.10 | 0.96 | 0.87 | 0.84 | 0.81 | 0.83 | 0.79 |

Spreads are widest at the open and narrowest into the close. This replaces the old engine's assumed ×1.3 for the last 5 minutes.

## 6. IEX vs NBBO (why IEX is a diagnostic, not a calibration source)

IEX TOPS is one venue, about 2–3% of volume. Its BBO is never tighter than the NBBO, and it is
often one-sided or far away. On the overlap day 2026-07-01, the median ratio of IEX time-weighted
median half spread to NBBO time-weighted mean was:

| ADV$ bucket | median IEX / NBBO |
|---|---|
| ≥ $1B | 10.5× |
| $100M–1B | 12.4× |
| $10M–100M | 1.9× |
| < $10M | 2.8× |

The ≥ $100M buckets are dominated by large single stocks, whose IEX book is often thin, for example AAPL: IEX median 1.35 bp vs NBBO 0.60 bp, and A: 309 bp vs 8.5 bp. Tick-constrained ETFs are close: EEM 0.74 vs 0.75, BAC 0.85 vs 0.89.

Even for SPY the IEX median was 0.85 bp in 2017 and 0.89 bp in 2018 (about 4 ticks), against a
1-tick NBBO of about 0.2 bp. In 2026 it was 0.20 bp. IEX's quoting competitiveness changed
substantially over 2017–2026, so a fixed IEX→NBBO mapping cannot recover era multipliers. Per-day
diagnostics are in `validation_report.json` (`iex_diagnostics`). The 2020-03-18 sample shows the
COVID spike: SPY IEX median 4.7 bp, while the model with that day's PIT σ (5.2%) gives
0.70 bp.

## 7. Limitations

* There is one consolidated-NBBO fit day, plus one out-of-sample day (2026-07-02). The
  cross-section is large, but day-to-day level shifts (stress days) enter only through σ.
* Before 2010 the model is anchored to published TAQ aggregates, not to quotes we hold.
  2003–2010 is interpolated. Pre-decimal clustering (odd-eighth avoidance, Christie & Schultz
  1994) is not modelled; the Furfine pre-decimal check is
  9.5¢ predicted vs 11.3¢ observed (-16%) (out of sample).
* The target is the time-weighted quoted spread. Fills of real orders are often better (effective
  ≈ 0.50× quoted on lit venues) or worse for size, which is covered by the square-root
  impact model, not here.
* Several characteristics are not features: market cap, index membership, ETF underlying
  liquidity, short-sale restrictions, news and earnings days. Residual SD of ln is
  0.64 (in-sample).
* ETF flags for dead ETFs (before 2026) fall back to the current Nasdaq Trader directory, so a dead
  ETF counts as a stock.
* The Tick Size Pilot is not modelled. The extended-hours multiplier is not measured.
* License: NYSE TAQ sample files are NYSE's public historical-data samples. Only derived
  per-symbol daily statistics are shipped (`calibration/*.csv.gz`), never quotes. IEX HIST is
  free under IEX terms with attribution ("Data provided by IEX").

## 8. Integration (ENGINE owner)

The model is a drop-in for `SpreadModel`. It has the same `series(...) -> SpreadSeries` (FULL relative
spreads, PIT) and `full_spread(...)` contract, and an attribute `method == "calibrated"`. Five edits:

1. `engine/config.py` (`BacktestConfig`): add
   ```python
   from ..execution.spread_calibrated import CalibratedSpreadConfig
   spread_model: str = "calibrated"            # 'calibrated' (default) | 'corwin_schultz' (legacy SpreadModel)
   spread_calibrated: CalibratedSpreadConfig = field(default_factory=CalibratedSpreadConfig)
   ```
2. `engine/backtester.py::_reset_state` (line `self.spread_model = SpreadModel(cfg.spread)`): replace with
   ```python
   if cfg.spread_model == "calibrated":
       self.spread_model = CalibratedSpreadModel(cfg.spread_calibrated)
   elif cfg.spread_model == "corwin_schultz":
       self.spread_model = SpreadModel(cfg.spread)
   else:
       raise ValueError(cfg.spread_model)
   ```
   and record `"spread_model": cfg.spread_model` next to `"spread_method"` in the run meta.
3. `engine/market.py::DailyStats.__init__`: replace the single line
   `self.spread: SpreadSeries = spread_model.series(bars.h * cum, bars.l * cum, ac)` with
   ```python
   if getattr(spread_model, "method", None) == "calibrated":
       self.spread = spread_model.series(bars.h * cum, bars.l * cum, ac, raw_close=c, volume=bars.v,
                                         dates=sessions_naive, asset_class=asset_class,
                                         adv_dollar=self.adv_dollar, sigma=self.sigma)
   else:
       self.spread = spread_model.series(bars.h * cum, bars.l * cum, ac)
   ```
   Add an `asset_class` argument to `DailyStats`, passed by `MarketData.daily(symbol)` from the
   instrument: `"ETF"` for `AssetClass.ETF`, else `"STK"`. `self.adv_dollar` and `self.sigma` are
   already PIT (shifted). `sessions_naive` can be int64 ns or datetime64.
4. `execution/fills.py::BarFillModel.half_spread`: pass ADV$ so the time-of-day profile matches the
   name's liquidity:
   ```python
   kw = {"adv_dollars": mkt.adv_dollar} if getattr(self.spread, "method", None) == "calibrated" else {}
   full = self.spread.full_spread(mkt.spread_estimate, price, tick, bar.mso, bar.mtc, **kw)
   ```
   Without it the call still works but uses the illiquid profile.
   Fidelity: when `mkt.spread_source.startswith("calibrated")` and the fill is TAKER, append the reason
   `"spread modelled (calibrated)"` and set `fid = max(fid, Fidelity.T2_PROXY)`. This is the tier this
   agent reports. Sources `default` and `Corwin-Schultz` keep the current behaviour.
5. `execution/__init__.py`: export `CalibratedSpreadModel, CalibratedSpreadConfig`. `docs/ENGINE.md`
   §3: the spread-estimator, floor and time-of-day rows should point to `docs/SPREADS.md`.
   `daily_open_spread_mult` keeps working (it multiplies the daily-average estimate).

Not needed: `SpreadConfig`, the price-bucket floors and the CS time-of-day table stay for the legacy
path. The calibrated path has its own tick floor (era ticks inside `series`, Rule 612 at the fill price
in `full_spread`) and needs no bucket floors. `run/replicate.py` uses `SpreadModel()` directly for a
CS comparison and can stay.

Tests: `tests/execution/test_spread_calibrated.py` (17 tests). They cover known answers from the json, calibration points (SPY/QQQ/IWM within 2× of the TAQ observation, CS ≥ 5× too wide), tick eras (1996 1/8, 1999 1/16, 2001 Nasdaq vs NYSE decimalisation, 2005/2006 Rule 612), monotonicity, PIT perturbation, vectorised == scalar, CS fallback, the full_spread contract and time of day, default / cap / effective / overrides.
