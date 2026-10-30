# Calibrated bid-ask spreads (`atlasbt/execution/spread_calibrated.py`)

Owner: SPREAD CALIBRATION agent. Calibration schema **v4** (2026-10-30: v3 + SIP NBBO evidence 2016-2025 from Alpaca, section 10; v3 = 2026-10-29: red-team fixes, then relative liquidity and stress flag; section 9). The engine
default is `BacktestConfig.spread_model='calibrated'`. Fidelity of every fill priced with it: **T2, "spread modelled
(calibrated)"**.

## 1. Problem

In bar mode the engine used to charge `half-spread = Corwin-Schultz (2012) 21-day estimate / 2` (Abdi-Ranaldo 2017 fallback).
Both estimators back the spread out of the daily high-low range, which for liquid names is mostly volatility. On 2026-07-01,
against the consolidated NBBO from the NYSE Daily TAQ sample (half bp = RTH time-weighted mean half spread):

| symbol (2026-07-01 PIT inputs) | NBBO observed, half bp | CS (old engine), half bp | calibrated model, half bp |
|---|---|---|---|
| SPY (P $747, ADV $47.14B, σ 1.1%) | 0.130 | 17.4 | 0.120 |
| QQQ (P $736, ADV $38.41B, σ 2.1%) | 0.266 | 23.7 | 0.263 |
| IWM (P $300, ADV $8.84B, σ 1.4%) | 0.305 | 27.1 | 0.332 |
| DIA (P $522, ADV $2.87B, σ 0.9%) | 0.353 | 20.3 | 0.557 |
| EEM (P $68, ADV $2.16B, σ 2.6%) | 0.747 | 19.6 | 0.740 |
| TLT (P $86, ADV $2.17B, σ 0.6%) | 0.584 | 5.0 | 0.579 |
| GLD (P $368, ADV $3.22B, σ 1.9%) | 0.596 | 9.1 | 0.903 |
| XLF (P $54, ADV $1.99B, σ 0.9%) | 0.915 | 19.9 | 0.933 |
| SOXL (P $267, ADV $14.08B, σ 15.7%) | 5.110 | 167.2 | 2.354 |
| SMH (P $656, ADV $7.65B, σ 4.4%) | 1.699 | 46.8 | 1.894 |
| AAPL (P $289, ADV $19.82B, σ 2.2%) | 0.597 | 52.6 | 0.815 |
| NVDA (P $200, ADV $33.09B, σ 2.7%) | 0.511 | 45.7 | 0.545 |
| MSFT (P $373, ADV $18.60B, σ 2.4%) | 1.000 | 27.7 | 1.083 |
| AMZN (P $238, ADV $15.07B, σ 2.4%) | 0.941 | 41.5 | 0.901 |
| JPM (P $327, ADV $3.19B, σ 1.6%) | 1.496 | 38.7 | 1.914 |
| INTC (P $140, ADV $16.01B, σ 6.0%) | 1.407 | 68.8 | 1.520 |
| BAC (P $57, ADV $2.15B, σ 1.2%) | 0.888 | 33.1 | 0.889 |
| T (P $21, ADV $1.41B, σ 2.4%) | 2.540 | 29.5 | 2.416 |
| F (P $14, ADV $0.81B, σ 1.9%) | 3.696 | 32.6 | 3.597 |
| AA (P $52, ADV $0.38B, σ 4.3%) | 4.009 | 37.0 | 5.849 |

On the hold-out symbols CS overstates by ≈ 20× for ADV ≥ $1B and ≈ 8× for $100M–1B.
SPY round trip: CS 34.9 bp; model 0.24 bp; NBBO 0.26 bp.

## 2. Data (all real quotes; nothing synthetic)

| Source | What | Used for |
|---|---|---|
| **NYSE Daily TAQ sample** `ftp.nyse.com/Historical Data Samples/DAILY TAQ/EQY_US_ALL_NBBO_20260701.gz` (9.6 GB) + `EQY_US_ALL_TRADE_20260701.gz` (4.4 GB) + `EQY_US_ALL_REF_MASTER_20260701.gz` | Consolidated **NBBO** for every US-listed security, full day, ns timestamps. Every trade. Security type (common / ADR / CEF / ETF / ETN ...) | **Fit** (2026-07-01) |
| Same, `EQY_US_ALL_NBBO_20260702.gz` (12.7 GB) | Next day's NBBO | Time-stability day and true out-of-sample (hold-out symbols × next day) (section 5) |
| **IEX HIST TOPS** pcaps (`iextrading.com/api/1.0/hist`), 12 days: 2017-03-22, 2018-03-14, 2019-03-13, 2020-03-18, 2020-09-23, 2021-09-15, 2022-09-14, 2023-08-03, 2024-09-11, 2025-09-10, 2026-01-15, 2026-07-01 (0.8–11.7 GB each) | IEX top of book (one venue) | IEX-vs-NBBO comparison and per-year diagnostics. **Not used in the fit** (section 6) |
| QuantConnect Lean fixtures `data/equity/usa/minute/spy/*_quote.zip` | SPY minute NBBO bars, 6 days in Oct 2013 and 2023-08-03 | Out-of-time checks |
| Literature (TAQ based) | Hendershott, Jones & Menkveld (2011, *JF* 66(1)) Table I: NYSE Dec 2002–Jul 2003 by market-cap quintile. Quoted and effective half spreads, $ volume, volatility, price. Furfine (2003, Chicago Fed *Economic Perspectives* Q4): NYSE most-active stocks, mean quoted spread 11.3¢ in the 2 weeks before decimalisation (1/16 tick), 7.0¢ after. Chordia, Roll & Subrahmanyam (2001, AIMR monograph *Common Determinants of Liquidity and Trading*) Table 1: NYSE 1992 mean / median quoted spread $0.32 / $0.27, proportional 1.60% / 1.15% | Era anchors and checks |
| Hasbrouck (2009, *JF* 64(3)) p.1457 / Fig. 2 | Annual TAQ effective cost of a cap-stratified random sample, 1993–2005: median ≈ 1.0% (1993) → 0.4% (2005), 95th percentile 5% → 2% (text values, linearly interpolated); Table V NYSE size deciles 1927–1998 (Gibbs) | Small-cap era slopes (95th pct) and checks |
| PIT daily features | DoltHub `stocks.ohlcv` (HF CSV mirror to 2026-01-28; SQL API for 2026-04-15..07-02), Yahoo daily (atlasbt store: SPY/QQQ/IWM, 2001 NYSE names as traded), qlib/Yahoo 2000–2005 common-stock cross-section (`calibration/hasbrouck_xs.csv.gz`, survivor-biased) | ADV$, σ, prior close (sessions strictly before the quote day) |

The tools are small C stream parsers (TAQ pipe text, IEX-TP pcapng) plus a feature builder. They are in
`calibration/tools/`, and their outputs are the per-symbol tables in `calibration/`. Pcaps and
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

No other consolidated-NBBO day is available for free: the NYSE `Historical Data Samples/DAILY TAQ` folder holds only 2026-07-01/02 NBBO (older files there are ADMIN messages; the venue BBO folders are single-exchange), so a non-2026 NBBO validation day cannot be added (checked 2026-10-29).

## 3. Model

```
x  = b·[1, la, ls, lp, E, E·la, E·ls, E·lp, min(lp − lp($5), 0), E·s, E·s·ls, E·s·lp, E·s·(la − la($100M)), E·max(la − la($5B), 0)]
     la, ls, lp = centred ln ADV$, ln σ, ln P (clipped to the fit-day range); E = ETF; s = logistic((la − la($100M))/0.5)
L  = P · exp(x) · era(t, ADV$)          latent full spread ($); ETF branch: running minimum over ADV$
S  = k · (1 + (L/k + c_d)^q)^(1/q)      quoted full spread ($), k = tick of the era/venue/symbol, S ≥ k
h  = S / (2P)                            half spread (× effective/quoted ratio if measure='effective')
```

| term | meaning | coef |
|---|---|---|
| `1` | constant | -4.8299 |
| `la` | ln ADV$ | -0.6274 |
| `ls` | ln σ (daily) | +0.9960 |
| `lp` | ln price | +0.6719 |
| `e` | ETF | -1.4209 |
| `e_la` | ETF × ln ADV$ | +0.4380 |
| `e_ls` | ETF × ln σ (bounded ≤ 0) | -0.5114 |
| `e_lp` | ETF × ln price | -0.7406 |
| `hinge_lp_lo` | min(ln P − ln 5, 0) | -0.6309 |
| `e_s` | ETF × s, s = logistic((la − la($100M))/0.5) | -2.4775 |
| `e_s_ls` | ETF × s × ln σ (e_ls + e_s_ls ≤ 0) | +0.5114 |
| `e_s_lp` | ETF × s × ln price | +0.6741 |
| `e_s_la` | ETF × s × (la − la($100M)) | -0.0287 |
| `e_hinge_vhi` | ETF × max(la − la($5B), 0) (bounded ≤ 0) | -0.6016 |
| c_d | discreteness offset | 0.000 |
| q | smooth-max exponent | 6.371 |

Centres: ln ADV$ 14.734 (≈ $2.5M), ln σ -3.948 (≈ 1.93%), ln P 3.309 (≈ $27.36).
Weighted robust NLS (soft-L1; weight log10 ADV$ − 5 clipped to [0.5, 5]). The liquid-ETF σ elasticity bound binds: it equals
the stock elasticity (0.996). The very-high-ADV ETF hinge rests on ~10 ETFs above $5B.

**Tick regimes** (`era_tick(date, price, listing, symbol)`; listing = `Instrument.exchange`):

| venue / symbol | switch date | regime |
|---|---|---|
| NYSE | 1997-06-24 | 1/8 (≥ $1), 1/16 below; then 1/16 (≥ $1), 1/32 below |
| Nasdaq | 1997-06-02 | 1/8 (≥ $10), 1/32 below; then 1/16 (≥ $10), 1/32 below |
| Amex stocks and other Amex ETFs | 1997-05-07 | 1/8 (≥ $10), 1/16 ($0.25–10), 1/32 below; then 1/16 (≥ $0.25) |
| SPY (from 1993-01-29), MDY (1995-05-04), DIA (1998-01-20), QQQ (1999-03-10) | until 2001-01-29 | 1/64 (Amex PDRs: SEC Rel. 34-38571 fn 6; Amex Rule 1000 Comm. .03, deleted in 67 FR 30984) |
| decimal | NYSE/Amex 2001-01-29 (Amex: all equities and ETFs, 67 FR 30984), Nasdaq 2001-04-09 | $0.01; < $1: $0.01 until Rule 612 compliance **2006-01-31** (70 FR 45529), then $0.0001 |

Decimal pilots (NYSE 2000-08-28 / 09-25 / 12-04, Nasdaq 2001-03-12 / 03-26) need membership lists, which are not shipped:
pass `CalibratedSpreadConfig(decimal_dates={symbol: date})`. An unknown exchange before 2001-04-09 uses `cfg.listing`
('NYSE') and is flagged (`listing assumed NYSE`). ARCA/BATS before 2001 are treated as Amex. The Tick Size Pilot is not
modelled (pass `tick=`).

**Relative liquidity.** ADV$ enters the model (and the era term) relative to the market: la_rel = ln(ADV$ · MV(fit date) / MV(t)), where MV is
the median daily US listed $ volume (2000 $67B, 2003 $57B, 2007 $180B, 2010 $209B, 2015 $232B, 2020 $455B, 2026 $1073B; `calibration/market_dollar_volume.csv`, 2000–2010 qlib/Yahoo all symbols,
survivor-biased (low), 2011–2026 DoltHub; log-linear in time, held flat before 2000 and after 2026). Nominal ADV$ is not
comparable across eras (MV grew ~19× 2003→2026). For 2010–2025 the deflated model alone would make a given nominal ADV$ cheaper than in 2026
($10M ADV, $20, σ 2.5%: 2010 6.2 bp, 2015 6.7 bp, 2020 10.1 bp, 2025 16.0 bp, 2026 17.6 bp half spread). There is no
NBBO evidence for those years, so by default (`pessimistic_floor_2010_2025=True`) the output for dates 2010-01-01..2025-12-31
is max(relative, nominal-ADV 2026-calibrated) estimate: the same stock is 17.6 bp in every year 2010–2025, and the reason
gets ", no NBBO evidence for era: pessimistic floor" (code +64) when the floor binds. Set the flag False for the pure
relative model.

**Era multiplier** `era(t, la_rel)` (latent spread relative to the 2026 fit) = upper envelope (pessimistic) of two literature
surfaces, each linear in time between anchors, then the backward cumulative maximum over later dates (`era_monotone`; non-increasing
in t at fixed RELATIVE ADV$, ≥ 1 before 2010):
* **A, HJM 2003 + Furfine 2001**: exp(a + b·la_rel), 2003 fitted to the five HJM quintiles (a = 0.600, b = 0.212), 2001
  level refitted to Furfine's 7.0¢ (as-traded inputs), held back to 1990, 0 from 2010.
* **B, Hasbrouck 2009**: per year 2000.5–2005.5, exp(a_H + b_H·min(la_rel, k) − 2·max(la_rel − k, 0)) with b_H ≤ 0, fitted (least
  squares on the log errors) to Hasbrouck's annual median and 95th-percentile effective cost on the qlib cross-section (model
  effective = 0.72 × quoted); k = the cross-section's median name, above which Hasbrouck carries no information and the surface
  fades out. 0 from 2010 (same kink), held back before 2000.

**Time of day**: three 5-minute profiles (medians over symbols with complete buckets and ≥ 200 trades, normalised to mean 1):

| profile | n | first 5 min | last 5 min |
|---|---|---|---|
| liquid stocks (ADV ≥ $100M) | 892 | 4.82× | 0.45× |
| **liquid ETFs** | 272 | 1.92× | 0.97× |
| illiquid | 5097 | 2.88× | 0.79× |

Daily bars (no minute) use 1.0. Extended hours use ×3, an unmeasured default. `full_spread(..., adv_dollars=, asset_class=)`
selects the profile.

**Inputs and fallbacks** (all PIT, sessions ≤ i−1):
* Price is the prior close in today's share units (`prior_close`, or derived from `close/raw_close`), so split days are
  priced post-split.
* ADV$ is the mean of raw close × volume over at most 20 prior sessions; one session is enough.
* σ is the std of prior returns. When the window is incomplete it is shrunk toward a price-bucket prior (< $1: 6.4%, < $5: 4.8%, < $10: 3.4%, < $20: 2.3%, < $50: 2.3%, else 2.3%) with 5
  pseudo-observations. When it is missing or zero (stale price) the prior replaces it.
* No volume (halted or stale): Corwin-Schultz/Abdi-Ranaldo, floored at half a tick, when available; otherwise the flagged
  10 bp default.
* ADV$ below the 2nd percentile of the fit universe ($9,033/day): capped by CS/AR.
* **Stress regime**: when σ ≥ 3 × the fit-day reference σ (stocks: price-bucket median above;
  ETFs: ETF median 1.10%), the reason gets ", stress regime: spread uncertainty high" (code +32). The
  bounded σ slope keeps stress-day spreads modest (SPY 2020-03-18: model 0.24 bp half; IEX upper bound 4.7 bp); use the atlas
  cost-stress scenarios (+2/+5/+10/+25 bp per side) and the break-even cost solver for that uncertainty.
* `measure='effective'`: effective/quoted ratio, median 0.50, clipped to [0.3, 1].

## 4. API (stable; new arguments are optional keywords)

```python
m = CalibratedSpreadModel(cfg=None, calibration=None)       # calibration dict overrides the json (v1 or v2 schema)
h, reason = m.half_spread(symbol, ts, price, adv_dollars, daily_vol, asset_class="STK", cs_half=None,
                          minutes_since_open=None, minutes_to_close=None, tick=None, exchange=None)
hs, code = m.half_spread_vec(dates, prices, advs, vols, etf, cs_half=None, tick=None, symbol=None, exchange=None, n_sigma=None)
ss = m.series(h_adj, l_adj, c_adj, raw_close=, volume=, dates=, asset_class=, adv_dollar=None, sigma=None,
              prior_close=None, exchange=None, symbol=None)            # SpreadSeries of FULL spreads (PIT)
full = m.full_spread(est, fill_price, tick, mso, mtc, adv_dollars=None, asset_class=None)
m.tod_multiplier(mso, mtc, adv_dollars=None, asset_class=None); m.era_multiplier(dates, adv); era_tick(...)
```

`code` bits: 0 ok, 1 ADV below range, 2 default, 3 CS/AR fallback; add 8 for the sigma prior, 16 when the listing was
assumed, 32 in the stress regime and 64 when the 2010–2025 pessimistic floor binds. `m.log_deflator(dates)` gives ln MV(fit)/MV(t); `m.era_multiplier(dates, adv_dollars=None,
relative_adv=None)`.

Reasons start with "calibrated spread model (T2)" and may add ", era-extrapolated" (before 2010), ", ADV below calibration
range", ", sigma prior", ", listing assumed NYSE", ", stress regime: spread uncertainty high" or ", asset class inferred STK"
(asset_class None). The other two reasons are
"Corwin-Schultz/Abdi-Ranaldo fallback (no volume) (T2)" and "default".

The loader accepts schema v1 (`era_anchors`), v2 (`era.large/small`) and v3 (`era.surfaces`, `market_volume`, `sigma_ref_etf`);
v1/v2 load with nominal ADV$. Anything else raises ValueError.

## 5. Validation

Hold-out = a deterministic 30% of symbols (md5). A fit on the other 70% predicts them. MAE is in bp of half spread; log ratio = ln(pred/obs).

| set | n | MAE bp | median AE bp | mean log ratio | SD log ratio | median abs log ratio | within ×2 |
|---|---|---|---|---|---|---|---|
| in-sample (70% training symbols), 07-01 | 7704 | 17.49 | 4.10 | -0.03 | 0.64 | 0.35 | 76% |
| **hold-out symbols, 07-01** | 3252 | 16.93 | 4.32 | -0.05 | 0.64 | 0.34 | 77% |
| **true out-of-sample: hold-out symbols on 2026-07-02, train-only fit** | 3245 | 18.68 | 4.32 | -0.06 | 0.62 | 0.36 | 77% |
| time stability, not out-of-sample: all symbols 2026-07-02, final fit (per-symbol residual correlation with 07-01 > 0.8) | 10940 | 17.95 | 4.15 | -0.06 | 0.62 | 0.35 | 77% |
| Corwin-Schultz, hold-out rows 07-01 | 3194 | 38.40 | 20.30 | +0.45 | 1.72 | 1.05 | 34% |
| Corwin-Schultz, 07-02 | 10762 | 38.06 | 19.54 | +0.46 | 1.59 | 1.00 | 37% |

| hold-out group (07-01) | n | model MAE bp | model mean / SD log ratio | within ×2 | CS MAE bp | CS mean / SD | CS within ×2 |
|---|---|---|---|---|---|---|---|
| ETF | 1584 | 8.16 | -0.05 / 0.74 | 68% | 20.5 | +0.05 / 2.07 | 35% |
| stock | 1668 | 25.26 | -0.04 / 0.53 | 85% | 54.8 | +0.83 / 1.21 | 33% |
| ADV>=$1B | 56 | 1.40 | +0.09 / 0.42 | 91% | 54.3 | +2.98 / 0.61 | 0% |
| ADV $100M-1B | 337 | 2.74 | -0.05 / 0.43 | 93% | 43.3 | +2.05 / 0.84 | 4% |
| ADV $10M-100M | 712 | 6.36 | -0.01 / 0.53 | 85% | 42.7 | +1.43 / 0.81 | 18% |
| ADV $1M-10M | 890 | 14.86 | -0.12 / 0.64 | 75% | 34.1 | +0.63 / 0.84 | 44% |
| ADV <$1M | 1257 | 28.87 | -0.02 / 0.74 | 68% | 36.9 | -0.82 / 1.89 | 46% |
| price<$5 | 465 | 54.27 | -0.09 / 0.66 | 75% | 80.3 | +0.45 / 0.92 | 49% |
| price $5-20 | 710 | 18.30 | -0.07 / 0.60 | 80% | 51.0 | +0.54 / 1.51 | 28% |
| price>=$20 | 2077 | 8.10 | -0.03 / 0.65 | 76% | 24.4 | +0.42 / 1.92 | 32% |
| ETF ADV>=$100M | 88 | 1.00 | -0.14 / 0.47 | 88% | 37.7 | +2.67 / 0.99 | 6% |

**Leave-named-out refit** (2026-07-01 half bp; NBBO / model with the symbol left out of the fit / CS):

| symbol | NBBO | model (left out) | CS |
|---|---|---|---|
| AAPL | 0.597 | 0.820 | 52.6 |
| AMZN | 0.941 | 0.906 | 41.5 |
| DIA | 0.353 | 0.641 | 20.3 |
| GLD | 0.596 | 1.028 | 9.1 |
| IVV | 0.306 | 0.638 | 20.5 |
| IWM | 0.305 | 0.456 | 27.1 |
| JPM | 1.496 | 1.923 | 38.7 |
| MSFT | 1.000 | 1.088 | 27.7 |
| NVDA | 0.511 | 0.548 | 45.7 |
| QQQ | 0.266 | 0.583 | 23.7 |
| SPY | 0.130 | 0.284 | 17.4 |
| TLT | 0.584 | 0.579 | 5.0 |
| VOO | 0.261 | 0.655 | 18.4 |
| XLF | 0.915 | 0.933 | 19.9 |

When left out, the largest high-priced index ETFs are overstated by up to about 2.5×. In-sample they fit (section 1).

**Historical and era checks**

| check | observed | model |
|---|---|---|
| SPY 2013-10 (Lean minute NBBO) | 0.298 bp half | 0.296 bp |
| SPY 2023-08-03 (Lean minute NBBO) | 0.114 bp half | 0.111 bp |
| Furfine 2001 most-active NYSE, post-decimal (as-traded inputs, 7 names) | 7.0¢ | 7.0¢ (monotone floor, not fitted) |
| Furfine 2001, pre-decimal (1/16 tick) | 11.3¢ | 9.0¢ |
| HJM 2003 NYSE quintiles Q1..Q5, quoted half bp | 5.19, 6.82, 9.17, 11.68, 19.89 | 4.65, 7.51, 9.74, 12.89, 65.16 |
| CRS 1992 NYSE median quoted (proxy: 2000-07 cross-section, P ≥ $5, re-dated to 1992) | $0.27 / 1.15% | $0.20 / 1.58% (proxy median price ≈ $12, so dollars are lower) |
| Hasbrouck NYSE size deciles 1 / 5 / 10 (Gibbs, 1927–1998 average) vs model effective at 1998 on proxy ADV deciles | 0.023 / 0.007 / 0.003 | 0.0644 / 0.0151 / 0.0016 |

**Hasbrouck (2009) annual effective cost, cross-section 2000–2005** (bp; the target is the paper's text values, the model is 0.72 × quoted on the qlib cross-section):

| date | n | a_H | b_H | target median | model median | surface A only | target p95 | model p95 |
|---|---|---|---|---|---|---|---|---|
| 2000-07-03 | 1615 | +0.50 | -0.20 | 62 | 58 | 26 | 312 | 694 |
| 2001-01-16 | 1703 | +0.50 | -0.20 | 60 | 65 | 28 | 299 | 897 |
| 2001-07-02 | 1704 | +0.50 | -0.20 | 58 | 38 | 13 | 288 | 642 |
| 2002-01-15 | 1793 | +0.50 | -0.20 | 55 | 38 | 12 | 274 | 565 |
| 2002-07-01 | 1821 | +0.50 | -0.20 | 53 | 31 | 12 | 263 | 556 |
| 2003-01-15 | 1871 | +0.50 | -0.20 | 50 | 36 | 11 | 249 | 619 |
| 2003-07-01 | 1987 | +0.50 | -0.20 | 48 | 29 | 10 | 238 | 467 |
| 2004-01-15 | 2088 | +2.53 | -0.00 | 45 | 24 | 8 | 224 | 423 |
| 2004-07-01 | 2146 | +2.37 | -0.00 | 43 | 23 | 8 | 213 | 399 |
| 2005-01-18 | 2258 | +2.37 | -0.00 | 40 | 22 | 7 | 200 | 363 |
| 2005-07-01 | 2286 | +2.37 | -0.00 | 40 | 20 | 7 | 200 | 396 |

**Residual misfit (documented, not hidden).** With both surfaces and the pessimistic envelope, HJM Q1–Q4 are reproduced (4.65, 7.51, 9.74, 12.89 vs 5.19, 6.82, 9.17, 11.68 bp) but HJM Q5 is 65.2 vs 19.89 bp (3.3×): at Q5's relative liquidity the Hasbrouck surface is higher and wins (pessimistic). Against Hasbrouck the model median is 0.51–1.08× and the 95th percentile 1.82–3.00× the target: with b_H ≤ 0 the survivor-biased proxy cross-section is more dispersed than Hasbrouck's sample, so median and p95 cannot both be met; the fit balances the two log errors. Surface A alone would put the median at only 0.15–0.45× Hasbrouck (the NYSE-only HJM sample vs Hasbrouck's all-market, cap-stratified sample is the conflict).

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
0.24 bp (IEX is an upper bound; the NBBO that day is unknown).

## 7. Limitations

* There is one consolidated-NBBO fit day. 2026-07-02 checks time stability and, with the hold-out symbols, out-of-sample
  error. NYSE's sample folders hold no free consolidated-NBBO day from another year. Since v4, 2016-2025 is checked
  against SIP NBBO samples from Alpaca (section 10) -- S&P 500 members and the 19 intraday ETFs only.
* Before 2010 the model rests on published aggregates (HJM 2003 NYSE quintiles, Furfine 2001, Hasbrouck 2009 text values) and
  on the monotone-in-time assumption, not on quotes we hold. 2003–2010 is interpolated under the monotone floor. The
  residual misfit against HJM Q5 and Hasbrouck is quantified in section 5. Market $ volume 2000–2010 is survivor-biased (low).
* Pre-decimal odd-eighth avoidance (Christie & Schultz 1994) is not modelled; the Furfine pre-decimal check is −12%.
* Not shipped:
  * decimal-pilot membership;
  * the 2016–18 Tick Size Pilot;
  * per-product Amex Index Fund Share increments (1/16, 1/32, 1/64). Only SPY/MDY/DIA/QQQ get 1/64.

  An unknown listing venue before 2001-04-09 falls back to NYSE and is flagged.
* Market cap, index membership, ETF underlying liquidity and news days are not features (residual SD of ln ≈ 0.64).
* ETF flags are point-in-time only as far as the caller's `asset_class` is. A missing class is treated as a stock and flagged.
* The extended-hours ×3 multiplier is not measured.
* Licensing: only derived per-symbol statistics are shipped (NYSE TAQ samples; IEX HIST, "Data provided by IEX").
  `hasbrouck_xs.csv.gz` holds Yahoo-derived features for research use.

## 8. Integration (ENGINE owner): already wired; v2/v3 additions

The engine constructs `CalibratedSpreadModel(cfg.spread_calibrated)` when `spread_model='calibrated'` (the default). Using the
v2 features needs only new keyword arguments; no signature breaks:

1. `engine/market.py::DailyStats.__init__`: call
   ```python
   spread_model.series(bars.h*cum, bars.l*cum, ac, raw_close=c, volume=bars.v, dates=sessions_naive,
                       asset_class=asset_class, prior_close=self.prior_close,
                       exchange=<Instrument.exchange or None>, symbol=symbol)
   ```
   and **drop** `adv_dollar=`/`sigma=`. The engine's arrays need 5 sessions; the model's own PIT arrays need 1 session and
   shrink σ to the price-bucket prior, which is the IPO fix. If they are kept, IPO sessions 1–4 fall back to CS/AR or the
   default.
2. `execution/fills.py::BarFillModel.half_spread`: pass
   `full_spread(est, price, tick, bar.mso, bar.mtc, adv_dollars=mkt.adv_dollar, asset_class=mkt.instrument.asset_class)`.
   This selects the liquid-ETF intraday profile.
3. Fidelity: sources that start with `calibrated` or `Corwin-Schultz fallback` map to T2 with the reason
   `spread modelled (calibrated)`. When the source contains `listing assumed`, add that text to the reason.
4. Tests:
   * `tests/execution/test_spread_calibrated.py`;
   * `tests/execution/test_spread_calibrated_loading.py`: loads the shipped JSON, checks v1 compatibility, and builds and
     runs a `Backtester` with the default `BacktestConfig`;
   * `tests/redteam/spreads/`.

## 9. Red-team review and integrator decisions (v1 → v3)

| item | change | before → after |
|---|---|---|
| 1 small caps pre-2010 | b_s ≤ 0 below HJM Q5, matched to Hasbrouck's 95th percentile; era non-increasing in t (backward cumulative max) | era(2003, $100k ADV) 0.27 → ≥ 1. For an $8 stock with $50k ADV and σ 3.5%, the 2003 half spread was 2–4× below 2010's; now 478 vs 369 bp |
| 2 Furfine inputs | as-traded prices (XOM $82.8, KO $56.6); old AT&T unavailable, so T dropped | post-decimal 7.0¢ (fitted on halved prices) → 7.0¢ (not fitted; monotone floor) |
| 4 liquid-ETF σ slope | bounded by the stock slope in the fit (the bound binds) | σ elasticity 2.0 → 1.00. SOXL/SOXX/SMH/MUU model/obs 2.7–8.1× → 0.46/1.25/1.11/0.98×. Hold-out ETFs ≥ $100M: median abs log ratio 0.10 |
| 5 tick schedules | per venue and per symbol (table in section 3), from `Instrument.exchange`; unknown → NYSE plus a flag | SPY 2000-02..2001-01 mean half spread 2.20 bp (1/16) → within 1.5× of the 1/64-bound value |
| 6 Rule 612 | 2005-08-29 → 2006-01-31 | `era_tick('2005-10-03', 0.50)`: 0.0001 → 0.01 |
| 7 intraday | third profile for liquid ETFs | ETF open / close 4.3× / 0.51× → 1.92× / 0.97× (07-02 replicates ≈ 1.9 / 0.97) |
| 8 split / IPO / stale / ETF flag | prior_close in today's units; ADV needs 1 session; σ prior; CS/AR fallback; asset class None flagged | split day 33.4 bp vs 7.3 bp the next day → equal; IPO session 2 priced (was the 10 bp default) |
| 9 ETF monotone in ADV | running minimum over ADV$ in the ETF branch | rising across the $100M knot → non-increasing (test) |
| 11 docs | 07-02 all-symbol row relabelled "time stability"; true OOS = hold-out symbols × 07-02 | true OOS MAE 18.68 bp, SD log ratio 0.62 |
| schema | loader accepts v1, v2 and v3 (`normalize_calibration`); clear error otherwise | none |
| v3: relative liquidity | ADV$ deflated by market $ volume in the model and the era; monotone constraint at fixed relative ADV$; HJM + Hasbrouck surfaces, upper envelope | HJM Q1–Q5 (v2) 5.2, 8.7, 13.5, 20.5, 33.1 → 4.7, 7.5, 9.7, 12.9, 65.2 bp (obs 5.19, 6.82, 9.17, 11.68, 19.89); Hasbrouck median 0.35–0.70× → 0.51–1.08×; Furfine post-decimal 8.0¢ → 7.0¢ (fitted) |
| v3: 2010–2025 floor | default `pessimistic_floor_2010_2025=True`: max(relative, nominal 2026) for 2010–2025, qualifier "no NBBO evidence for era: pessimistic floor" | $20 / $10M / σ 2.5%: 2010 6.2 → 17.6 bp, 2015 6.7 → 17.6, 2020 10.1 → 17.6, 2025 16.0 → 17.6 (2026 17.6) |
| v3: stress | reason qualifier "stress regime: spread uncertainty high" when σ ≥ 3 × reference σ | SPY 2020-03-18 and 2008-10-15 flagged; normal days not |

## 10. Validation against real SIP NBBO 2016-2025 (Alpaca) and the v4 refit

**Data.** Alpaca's historical SIP quotes are the consolidated NBBO stream (docs/DATA.md §4.15), cleaned the same
way as the TAQ statistics (crossed / locked / non-firm updates dropped). Sample
(`calibration/tools/alpaca_nbbo_sample.py` -> `calibration/alpaca_nbbo_2016_2026.csv.gz`, derived statistics
only): the 2nd Wednesday of a month (quarterly months first; the fit used the 40 quarterly days 2016-01-13 ..
2025-10-08), seven 1-minute windows at the middle of the model's 5-minute time-of-day buckets (09:32, 09:47,
10:32, 12:32, 14:32, 15:32, 15:52 ET), the 19 intraday ETFs plus 50 PIT S&P 500 members spread evenly over that
day's ADV$ ranks (632 symbols in all), each identity-routed to its tape ticker. Statistic: time-weighted mean
relative half spread in the minute. Features exactly as in the calibration (prior raw close, 20-day ADV$,
20-day sigma, sessions strictly before the day). Prediction = what the engine charges: max(model half spread x
time-of-day multiplier, half a tick) (`calibration/alpaca_validate.py`). 19,066 symbol-windows, 2,746
symbol-days.

**Finding.** ETFs: the model is unbiased in every year (median ln(pred/obs) -0.00 .. +0.04; within 2x 91-100 %).
S&P 500 stocks: v3 overstated the half spread by **~1.9x (median ln ratio +0.63)** in 2016-2024 -- the 2010-2025
pessimistic floor (nominal-ADV 2026 estimate, used because there was no NBBO evidence) binds for most large
caps, while the relative-liquidity model without the floor is close to unbiased (2016-2018 median ln ratio
-0.00 .. +0.01, 2019-2023 +0.11 .. +0.31). 2025 is the exception: the floor was right (-0.03), the relative model
alone is too low there.

**v4 refit** (`spread_calibration.json` version 4; v3 kept as `spread_calibration_v3.json`; the loader accepts
v1-v4, `CalibratedSpreadConfig(nbbo_evidence=False)` reproduces v3). New block `nbbo_evidence`: for sessions
2016-01-04 .. 2025-12-31 the nominal floor is replaced by the relative model x a per-year multiplier m(y, kind)
= exp(max(0, -q40)), q40 = 40th percentile of ln(relative model / observed) over that year's symbol-days, so the
model is at or above the observed spread on >= 60 % of symbol-days (pessimistic) and is never below the
relative model. Multipliers (stock / ETF): 2016 1.14 / 1.00, 2017 1.30 / 1.00, 2018 1.08 / 1.00, 2019-2023
1.00 / 1.00, 2024 1.05 / 1.00, 2025 1.49 / 1.03. The evidence covers S&P 500-sized liquidity only, so it applies
fully above ADV$ $76.7M (10th percentile of the sample stocks), not at all below $36.2M (2nd percentile; the
v3 floor stays), log-linear in between (monotone in ADV$, tested). 2010-2015 and smaller names keep the floor
("no NBBO evidence for era: pessimistic floor"); evidence rows carry "era checked against SIP NBBO 2016-2025"
(code +128).

**Errors before (v3) / after (v4)**, bp of half spread; v4 numbers are **out of fold** (multipliers fitted on the
Jan/Jul days, evaluated on Apr/Oct, and vice versa):

| set | n | obs median bp | MAE bp v3 / v4 | median ln(pred/obs) | SD ln | within x2 | pred < obs |
|---|---|---|---|---|---|---|---|
| all (symbol x window) | 19066 | 1.65 | 3.27 / 1.76 | +0.41 / +0.07 | 0.57 / 0.47 | 65% / 84% | 18% / 33% |
| symbol-day level | 2746 | 2.60 | 3.09 / 1.44 | +0.46 / +0.15 | 0.44 / 0.35 | 69% / 93% | 11% / 27% |
| S&P 500 stocks | 13820 | 2.47 | 4.44 / 2.37 | +0.63 / +0.18 | 0.58 / 0.52 | 53% / 78% | 13% / 32% |
| ETFs | 5246 | 0.63 | 0.18 / 0.17 | +0.01 / +0.00 | 0.28 / 0.27 | 97% / 98% | 31% / 37% |
| stocks 2016 | 1387 | 1.90 | 4.04 / 1.93 | +0.79 / +0.12 | 0.62 / 0.60 | 44% / 72% | 8% / 33% |
| stocks 2018 | 1387 | 2.17 | 3.33 / 1.49 | +0.61 / +0.01 | 0.57 / 0.54 | 54% / 80% | 11% / 45% |
| stocks 2020 | 1394 | 3.11 | 7.94 / 3.46 | +0.87 / +0.34 | 0.52 / 0.49 | 38% / 74% | 5% / 21% |
| stocks 2022 | 1397 | 2.65 | 5.37 / 2.63 | +0.71 / +0.29 | 0.45 / 0.45 | 47% / 78% | 4% / 21% |
| stocks 2024 | 1377 | 2.67 | 2.98 / 2.08 | +0.41 / +0.11 | 0.48 / 0.48 | 70% / 82% | 20% / 39% |
| stocks 2025 | 1342 | 4.61 | 3.43 / 3.81 | -0.03 / +0.07 | 0.49 / 0.53 | 83% / 79% | 55% / 43% |
| stocks ADV < $100M | 2616 | 2.99 | 5.07 / 3.46 | +0.74 / +0.31 | 0.63 / 0.62 | 44% / 66% | 11% / 28% |
| stocks $100M-500M | 8710 | 2.53 | 4.66 / 2.24 | +0.64 / +0.17 | 0.55 / 0.47 | 53% / 81% | 12% / 31% |
| stocks $500M-2B | 2053 | 1.87 | 3.28 / 1.81 | +0.47 / +0.10 | 0.59 / 0.50 | 61% / 81% | 20% / 37% |
| stocks $2B-10B | 266 | 1.04 | 1.95 / 1.12 | +0.51 / +0.12 | 0.60 / 0.56 | 62% / 74% | 19% / 37% |
| stocks >= $10B | 175 | 1.26 | 1.34 / 0.81 | +0.15 / -0.17 | 0.64 / 0.57 | 75% / 79% | 37% / 60% |
| window 09:32 | 2713 | 6.08 | 9.53 / 5.57 | +0.65 / +0.40 | 0.59 / 0.59 | 58% / 75% | 13% / 28% |
| window 10:32 | 2726 | 1.98 | 2.40 / 1.18 | +0.35 / +0.03 | 0.51 / 0.40 | 69% / 90% | 18% / 37% |
| window 12:32 | 2705 | 1.56 | 1.70 / 0.83 | +0.29 / +0.01 | 0.52 / 0.40 | 71% / 89% | 22% / 39% |
| window 15:32 | 2740 | 1.23 | 1.61 / 0.82 | +0.33 / +0.01 | 0.58 / 0.43 | 65% / 85% | 20% / 33% |
| window 15:52 | 2744 | 1.10 | 1.60 / 0.82 | +0.35 / +0.02 | 0.63 / 0.47 | 62% / 81% | 19% / 30% |

The 2025 multiplier makes v4 slightly worse there out of fold (MAE 3.43 -> 3.81 bp) -- the price of the
pessimistic quantile. Residual: the largest stocks (ADV >= $10B, n = 175) are now understated by ~16 % in the
median (no liquidity term in the evidence multiplier).

**Time of day** (window / symbol-day mean, median over symbol-days): stocks observed 2.70 (09:32), 1.20, 0.84,
0.62, 0.59, 0.45, 0.39 (15:52) vs the model's liquid-stock profile 3.06, 1.37, 0.78, 0.53, 0.47, 0.41, 0.38 --
close; the open is ~13 % too wide in the model. Large ETFs are flat in reality (0.99-1.03 at every window: they
quote at one tick all day) while the liquid-ETF profile gives 1.68x at 09:32 and 0.85x midday; for tick-bound
ETFs the engine's one-tick floor dominates (SPY ~0.2 bp), so the error is < 0.1 bp. The profiles are unchanged
in v4.

**2026 cross-check** (4 quarterly days 2026-01..10, model unchanged by v4): S&P stocks median ln(pred/obs) +0.06
(MAE 2.98 bp on a 3.73 bp median, within 2x 86 %), ETFs +0.02 (within 2x 93 %) -- the TAQ 2026-07-01 fit and
Alpaca's NBBO agree, which also validates the Alpaca data path. The shipped sample has 44 days / 20,957 rows.

**Dense ETF check** (every session): the decision-window quotes stored for the strategies (15:24 ET, ETF_CORE,
2016-2021 first pass) give model vs observed median ln ratio -0.04 (pred 0.302 vs obs 0.308 bp, MAE 0.065 bp,
within 2x 97.6 %; 2020: +0.09, 2021: +0.16); `alpaca_validate.window_sample_from_store()` recomputes it.

Rebuild: `python atlasbt/execution/calibration/tools/alpaca_nbbo_sample.py` (key-gated, resumable), then
`--assemble`, then `alpaca_validate.fit_evidence` + `write_v4` (see the module); validation report:
`python -m atlasbt.execution.calibration.alpaca_validate --json ...`.
