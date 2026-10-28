# atlasbt options & pricing (`atlasbt.pricing`, `atlasbt.options`)

Owner: OPTIONS agent. Contracts: `docs/ARCHITECTURE.md` §3/§7, `atlasbt/core/types.py`.
Everything here is a pure function or a small dataclass; the engine plugs it in through
`atlasbt.options.plugins` (§11). Tests: `tests/pricing/`, `tests/options/` (no network).

## 1. Module map

| module | contents |
|---|---|
| `pricing/bsm.py` | BSM (continuous q), escrowed discrete-dividend BSM, Black-76, greeks (`Greeks`), bounds, strike-from-delta |
| `pricing/american.py` | CRR binomial (continuous q + discrete cash dividends, numba optional), Bjerksund-Stensland 2002 (+1993 cross-check), bivariate normal, FD greeks |
| `pricing/iv.py` | vectorized European IV (safeguarded Newton-bisection + Brent fallback, reason codes), American IV (direct inversion), de-Americanization |
| `pricing/forward.py` | parity forward / implied carry per expiry (American-adjusted) |
| `pricing/surface.py` | smile extraction, ATM per expiry, constant-maturity IV (total variance), delta-space IV, 25Δ skew, SVI raw fit + no-arbitrage checks, spline-in-delta fallback |
| `pricing/realized.py` | close-close, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang; EWMA; walk-forward HAR-RV |
| `pricing/time.py` | settlement instants, ACT/365F year fractions, rate conversions (simple / T-bill discount → continuous) |
| `options/chain.py` | chain preparation (dte, T, mid, delta source), quality filters, stale detection, expiry/delta/ATM/wing selection |
| `options/structures.py` | `OptLeg`/`StockLeg`/`Package`, builders for every S126–S172 structure, mid/natural valuation, payoff, max P/L, breakevens, COMBO orders, greeks, hedge & vega-match helpers |
| `options/metrics.py` | `chain_metrics`: ATM 30D/60D, 25Δ put/call IV & skew, term spread, forwards |
| `options/marking.py` | mark any contract from a snapshot: quoted → same-day surface (T2) → extrapolated (T3) |
| `options/listing.py` | historical listed-expiry calendar (SPX AM/PM, SPXW weekdays, SPY/QQQ/IWM weeklies), strike grids, tick sizes |
| `options/synthetic.py` | synthetic chain generator (T3_MODELED) from Cboe vol indices + SKEW + calibrated spread model |
| `options/calibrate.py` | calibration on DoltHub chains + validation vs Cboe PUT index |
| `options/lifecycle.py` | expiration (OCC ex-by-ex), AM/PM cash settlement, physical delivery, early assignment, pin risk, split/special-dividend adjustments |
| `options/margin.py` | Cboe strategy-based margin (Margin Manual 2021), stress-scenario loss |
| `options/parity.py` | conversion/reversal executable edges (dividends, borrow, early exercise), American parity bounds, box-spread rates |
| `options/plugins.py` | `OccOptionLifecycle`, `CboeStrategyMargin` implementing `atlasbt.portfolio` protocols |

## 2. Conventions

* **Expiry** (`Instrument.expiry`, chain `expiry`): the LAST TRADING DAY. Vendors (L3, OCC) list SPX
  AM monthlies under the settlement Friday — loaders must convert to the Thursday last trading day. For AM-settled SPX
  monthlies that is the Thursday before the 3rd Friday; the SOQ settlement session is
  `pricing.time.settlement_session()` (next session). PM/physical: settlement = expiry day.
* **Time**: `T` = ACT/365F year fraction from `now` to the settlement instant
  (`settlement_instant`): official close of the expiry session for PHYSICAL/CASH_PM
  (13:00 ET on early-close days, from `core.clock`), 09:30 ET open of the settlement session for
  CASH_AM. Intraday (0DTE) T is exact to the second. DTE = calendar days from the ET session date.
* **Rates**: all pricers take continuously-compounded `r`. `DataPortal.risk_free_rate` (simple)
  → `simple_to_cc(r, T)`; FRED DTB3 (bank discount, ACT/360) → `tbill_discount_to_cc(d, days)`.
* **Dividends**: continuous yield `q` (index products; or the parity-implied carry, which also
  absorbs borrow) or discrete cash dividends via the escrowed model (European: S* = S − PV(D);
  CRR tree on S* with PV of remaining dividends added back for early-exercise tests).
* **Units**: prices per share/index point; `× multiplier` (100) per contract. Greeks: delta per $1,
  gamma per $1², **vega per 1 vol point**, **theta per calendar day**, rho per 1 rate point
  (same as DoltHub columns); `Greeks.per_contract(100)`.
* **Signs**: package ratios +long/−short; package price = net debit (+) / credit (−);
  `natural_open` buys at ask / sells at bid, `natural_close` the reverse.

## 3. Pricing models and validation (tests/pricing)

* BSM: Hull Ex. 15.6 (4.76/0.81), Haug 2.1334 (generalized BSM) and 2.4648 (Merton put),
  greeks vs Hull ch.19 table (Δ .522, Γ .066, vega 12.1, Θ −4.31/yr, ρ 8.91) and vs finite differences.
* Black-76: Haug 1.7011 (F=K=19, T=.75, r=10%, σ=28%), Hull futures put 1.12.
* Discrete dividends: Hull Ex. 15.10 (c = 3.67, S* = 39.0259); CRR European with dividends → escrowed BSM.
* CRR: Hull Ex. 21.1 5-step American put 4.49; convergence to BSM (1000 steps, < 2e-3).
  At very low vol (p ∉ (0,1)) the tree switches to a drift-centred lattice.
* Bjerksund-Stensland: the shared φ function reproduces Haug's BS-1993 value 5.2704 exactly;
  BS-2002 (5.2869 for the same inputs) is checked within 0.6% of a 2000-step CRR for calls and puts
  and above BS-1993. Numerical overflow at tiny vol falls back to CRR.
* Bivariate normal: Genz (2004) with 20-point Gauss-Legendre, |err| < 1e-7 vs scipy.
* IV: vectorized safeguarded Newton-bisection (bracket kept, cannot diverge) + `brentq` fallback on
  [1e-4, 5]. NaN with reason: `BELOW_INTRINSIC` (below discounted-forward intrinsic), `ABOVE_MAX`,
  `NONPOSITIVE_PRICE`, `EXPIRED`, `VOL_BELOW_MIN`, `VOL_ABOVE_MAX`, `NO_CONVERGENCE`, `BAD_INPUT`.
  American IV inverts BS2002 (or CRR with discrete dividends) directly; `deamericanize` returns the
  European price at the American IV (used for parity forwards).

## 4. Forwards, surfaces, realized vol

* Forward per expiry: inverse-spread-weighted median of K + e^{rT}(C−P) over the 5 strikes with the
  smallest |C−P|; American chains iterate de-Americanization (2 passes). Implied carry
  q = r − ln(F/S)/T lumps dividends + borrow + rate mismatch.
* IV for surfaces: Black-76 on the parity forward from OTM mids (puts K<F, calls K≥F) so real and
  synthetic chains are treated identically (quoted vendor IVs are not trusted for signals).
* Constant maturity (S143/S165/S168): linear interpolation in total variance w=σ²T between
  bracketing expiries; flat-vol extrapolation flagged `extrap_flat`.
* Delta-space (S166/S167): IV vs forward delta interpolated per expiry, total variance across expiries;
  `skew_25d` = IV(−25Δ put) − IV(+25Δ call) at 30D.
* SVI raw per expiry (least squares in total variance, multi-start); `ok` only if: min variance ≥ 0,
  Lee bound b(1+|ρ|) ≤ 4, Gatheral-Jacquier g(k) ≥ 0 on a grid (no butterfly arbitrage), RMSE ≤ 2 vol
  pts. `calendar_arbitrage_free` checks w non-decreasing in T. Fallback: smoothing spline in call-delta.
* Realized: CC, Parkinson, GK, RS, YZ (annualized ×252, rolling, causal). EWMA λ=0.94.
  HAR-RV (Corsi 2009) target = mean daily variance over the next h sessions; walk-forward OLS refit
  every 21 sessions using only rows i with i+h ≤ t (no lookahead; tested by perturbing the future).
  Daily variance proxy: GK + squared overnight gap (or squared CC return).

## 5. Chain selection rules (deterministic)

* `prepare_chain(chain, now, r)`: dte, T, mid (if missing), spread, rel_spread, `delta_used`
  (quoted delta → BSM from quoted iv → BSM from mid IV; `delta_source` column).
* `choose_expiry(target_dte, min, max)`: min |dte−target| → earlier expiry → standard monthly before
  weekly → root.
* `select_by_delta`: min |Δ−target| → strike closer to spot → lower strike. Optional `max_delta_error`.
* `nearest_atm`: min |K−ref| → lower strike; `both_rights=True` for straddles/flies.
* Wings by points use ACTUAL listed strikes: exact, else `outward` (default; wider → defined risk
  never understated), `inward`, `nearest`, or `exact` (raise).
* Quality (`QualityRules`): two-sided, uncrossed/unlocked, nonzero bid, rel. spread ≤ 50% (default),
  optional abs spread / quote age / OI. `flag_stale_vs_previous`: unchanged bid & ask while the
  underlying moved ≥ 0.5% and |Δ| ≥ 0.05.
* `SelectionError` (subclass of `AtlasError`) when no listed contract satisfies a rule — strategies
  should skip the trade, never invent a strike.

## 6. Structures (`options.structures`)

Builders: `single`, `atm_single`, `vertical` (deltas), `vertical_by_points`, `straddle`, `strangle`,
`iron_condor` (wing by delta or points), `iron_butterfly` (symmetric, listed), `butterfly`
(call/put, `far_multiple` for broken wing S155), `ratio_backspread`, `risk_reversal`, `jade_lizard`
(`meta['no_upside_risk']`), `covered_call`, `cash_secured_put` (`meta['collateral']`),
`protective_put`, `collar`, `covered_strangle`, `calendar`, `diagonal`, `box`, `conversion`/reversal.
`Package`: `mid_value`, `natural_open/close`, `price_with_improvement(frac)`, `payoff`,
`max_profit_loss` (∞ when unbounded; same-expiry only), `breakevens`, `greeks`/`dollar_greeks`,
`to_order(units, limit)` → core `Order(COMBO)` (stock legs submitted separately).
Helpers: `hedge_shares` (delta-hedge with band, S146/S147/S167/S170), `vega_match_units` (S170).

## 7. Synthetic chains (fidelity T3_MODELED)

Used only when no real chain exists (pre-2020, SPX/QQQ/IWM, LEAPS, short-dated expiries, intraday).
Every frame has `fidelity = 3` and `attrs['source']`.

### 7.1 Listed expiries (`options.listing`, facts in `LISTING_FACTS` with confidence)

| series | first expiry modeled | source / confidence |
|---|---|---|
| SPX standard monthly, AM (SOQ) | ≥1992-06 | Cboe SPX specs (high) |
| SPX Friday Weeklys, AM | 2005-11-04 … 2010-12-03 | Cboe: first Weeklys listed 2005-10-28; last AM weekly 2010-12-03 (high) |
| SPXW Friday, PM | 2010-12-10 (listed 2010-12-02) | Cboe End-of-Week pilot, SEC approval 2010-09-14 (high) |
| SPXW 3rd-Friday PM | 2017-05-19 (class move 2017-05-01) | SPXPM→SPXW symbol change (medium; earlier C2 SPXPM not modeled) |
| SPXW Wednesday | 2016-03-02 | Cboe 2016-02-01 release, RG16-024 (high) |
| SPXW Monday | 2016-08-22 | Cboe 2016-07-11 release (high) |
| SPXW Tuesday | 2022-04-26 (listing 2022-04-18) | Cboe 2022-04-13 release; first expiry inferred from 1-week pattern (medium) |
| SPXW Thursday | 2022-05-19 (listing 2022-05-11) | same (medium) → every weekday from 2022-05-19 |
| SPY/QQQ/DIA/IWM Friday Weeklys | 2010-06-11 (listed 2010-06-04) | Cboe ETF Weeklys launch (high) |
| SPY Wednesday | 2016-08-31 | SEC approval of BOX Wed SPY 2016-08-24; first expiry assumed +1wk (low) |
| SPY Monday | 2018-02-26 | Cboe listing 2018-02-16 (medium) |
| SPY/QQQ Tue/Thu | 2022-11-15 / 2022-11-17 | MIAX alert 2022-11-14 (high) |
| QQQ Mon/Wed | 2021-01-04 | UNVERIFIED start (conservative; existed per 2021 filings) (low) |
| IWM Mon/Wed | 2021-10-04 | SEC Rel. 34-93157 (2021-09-28) (low) |

Rules: monthlies = 3rd Friday, holiday → prior session (Good Friday); weekly holiday shifts:
Monday → next session, other days → prior session; SPX Weeklys not listed in standard-expiration
week before the 2017 SPXW 3rd-Friday move. Monthly set: 4 serial months + quarterly to 1y + Jan LEAPS
(~2.5y); SPX: 12 serial + quarterly to 3y + Dec to 5y. Weekly horizon 35 days (pre-2016) / 50 days.
`has_0dte('SPX', '2023-06-13') → True`, `('SPX', '2015-06-09') → False` (tested).
Single-stock weeklies are not modeled by default (per-name history unavailable).

Strikes (`strike_grid`): SPX 5 pts within ±15%, 25 beyond / long-dated; SPY/QQQ/IWM/DIA $1 (≤400 DTE);
single names pre-2012 2.5/5/10, 2012+ 1/2.5/5/10 by price band; span ±max(15%, 4σ√T) ≤ 80%.
Ticks: pre-2007-01-26 $0.05/<$3 else $0.10; Penny Pilot after: SPY/QQQ/IWM $0.01, others $0.01/$0.05;
SPX $0.05/$0.10. (Approximations; documented, tested.)

### 7.2 Volatility

* ATM ratio (index products) is point-in-time (§7.4): `atm_ratio_params(p, as_of)`.
* ATM term structure, point-in-time (an index close is used only after 16:15 ET of its session;
  older than 5 days → missing): VIX1D(1d), VIX9D(9), VIX(30), VIX3M(93), VIX6M(183), VIX1Y(365),
  total-variance interpolation, flat extrapolation. VIX1D history before its 2023-04-24 launch is
  Cboe's back-calculation (computed from then-existing quotes, but not published then).
* Vol indices are variance-swap style (they include the skew premium), so ATM IV =
  index × `atm_ratio` (calibrated, §7.4).
* Other underlyings: QQQ/NDX→VXN, IWM/RUT→RVX, DIA→VXD, AAPL→VXAPL, AMZN→VXAZN, GOOG(L)→VXGOG,
  GS→VXGS, IBM→VXIBM, EEM→VXEEM, EFA→VXEFA, GDX→VXGDX, SLV→VXSLV, TLT→VXTLT, GLD→GVZ, USO→OVX:
  30-day level from the mapped index, term shape from the VIX family. Unmapped names need the caller's
  `rv_ratio` (trailing realized vol of the name / of SPX, e.g. 60d Yang-Zhang from adjusted bars);
  without it the generator returns an empty frame with a reason (never a silent default).
* Smile: σ(K) = σ_ATM·g(x), x = ln(K/F)/(σ_ATM√T), g = 1 + a x + b x² for |x| ≤ 1.5 (the strategy
  region, ≈ 10–90Δ) with separately calibrated linear put/call wings beyond, floor 0.35.
  a(T) = a30 (DTE/30)^γ (the normalized slope steepens with maturity: real SPXW 2019-08-15 a ≈ −0.08
  at 1 DTE, −0.20 at 30, −0.25 at 150). Index products may shift a30 with Cboe SKEW (PIT, clipped to
  the calibration range; disabled unless significant); single names use their own parameters.
* Prices: European BSM for SPX (cash-settled), BS-2002 American for equity/ETF with carry q.

### 7.3 Bid/ask model

log(ask−bid) = FE[underlying] + b1 log(mid) + b2 log(mid)² + b3|Δ| + b4 log(DTE) + b5 log(VIX),
OLS with underlying fixed effects; unknown names use the median ETF/stock fixed effect; SPX uses the
SPX AM (root SPX) and SPXW have their own fixed effects calibrated on the 2019-08-15 full-market day
(fallback to the SPY model ×10 only if a future recalibration lacks SPX quotes).
Quotes are rounded outward to the tick grid; a bid below one tick becomes 0 (no bid), as in real
far-OTM quotes.

### 7.4 Calibration evidence

All numbers are reproducible with `python -m atlasbt.options.calibrate` (≈ 6 min) and are stored in
`atlasbt/options/calibration/synthetic_params.json` (`evidence`). Calibration data:

* **DoltHub** `post-no-preference/options` EOD chains: 19 dates (2020-01-15 … 2026-09-16, one per
  ~quarter) × SPY, DIA, XLF, XLE, XLK, XLU, AAPL, AMZN, MSFT, NVDA, TSLA, AMD, JPM, XOM, KO, F
  (sample stored as `calibration/dolthub_sample.csv.gz`, 21,842 rows; fetched 2026-10-28 with UA
  `atlasbt-research/0.1`). No spot in the data: spot from parity on the nearest expiry.
* **historicaloptiondata.com L3** full-market EOD day 2019-08-15 (`data_raw/historicaloptiondata_free_samples`,
  read in place): SPX, SPXW, SPY, QQQ, IWM, DIA, GLD, TLT, EEM + 13 stocks, with the vendor's
  underlying price (56,276 rows). Only free source with SPX/SPXW/QQQ/IWM quotes.
* **Cboe PUT index** (daily 2007–2026) + ^GSPC open/close + FRED DTB3: SPX ATM put premia backed out
  of the index (validation and point-in-time ATM-ratio schedule).
* **QuantConnect Lean fixtures** (AlgoSeek minute quotes, SPX Jan-15-2021 AM contracts on
  2021-01-04 and 2021-01-14): out-of-sample intraday SPX spread check.

Results (205 snapshots, 71,030 two-sided quotes):

| item | value |
|---|---|
| ATM ratio, index products | **point-in-time schedule** (`atm_ratio_schedule`): for year Y, Huber fit of PUT-implied ATM IV / VIX = c0 + c1 ln(VIX/0.20) on rolls settled before Y (2010: c0 .934, c1 −.030; 2026: c0 .840, c1 .162). Dates before 2010 use the 2007–09 fit ("pre-sample", labelled in `attrs['source']`). Regime dependence is strong: PUT-implied ratio 0.73 at VIX≈11 vs 0.92 at VIX>24. Cross-check on SPY/DIA/QQQ/IWM/SPX chains 2019–2026: c0 .870, c1 .073 (n = 43). |
| ATM ratio, single names vs own vol index | 0.900 (n = 29; AAPL/VXAPL, AMZN/VXAZN, EEM/VXEEM, GLD/GVZ, TLT/VXTLT) |
| smile (index products) | a(T) = −0.222 (DTE/30)^0.104, b = 0.003, put wing slope −0.174, call wing +0.329 (224 expiries; median |log resid| of a 0.13); SKEW dependence not significant → a1 = 0 |
| smile (single names) | a(T) = −0.096 (DTE/30)^0.117, b = 0.034, wings −0.224 / +0.190 (439 expiries; noisier, 0.42) |
| spread model | log s = FE + 0.347 log mid + 0.008 log² mid + 1.111 |Δ| + 0.197 log DTE + 0.708 log VIX; R² 0.83; residual sd 0.72 (log); median |log error| 0.46 (≈ ×1.6): L3 day 0.42, DoltHub 0.61; SPX 0.27, SPXW 0.34, SPY 0.48. Time-split OOS (fit < 2024-06-12, test after, n = 5,976): median |log err| 0.79, model/actual 0.64 → **recent DoltHub spreads are ~1.5× wider than predicted**; use cost-stress ×1.5–2 on synthetic fills. |
| SPX intraday spreads (Lean, OOS) | 9 contract-days, model/actual median 1.16, median |log err| 0.28 |
| zero-bid share in real quotes | 9.0% (synthetic quotes reproduce "no bid" below one tick) |

**Known-answer validation — Cboe PUT index replicated with the synthetic chain**
(`replicate_put_index`: listing calendar → AM monthly SPX → listed ATM strike → synthetic price →
SOQ proxied by the ^GSPC open → T-bill collateral; point-in-time parameters):

| window | months | synthetic ann. return | PUT index | tracking error | corr |
|---|---|---|---|---|---|
| 2008-01 … 2026-09, at mid | 225 | 9.09% | 7.59% | 2.63%/yr | 0.979 |
| same, sold at synthetic bid | 225 | 8.67% | 7.59% | 2.64%/yr | 0.979 |
| 2008–2016 | 107 | 7.59% | 6.72% | 3.18%/yr | 0.970 |
| 2017–2026 | 117 | 10.39% | 8.31% | 2.01%/yr | 0.989 |
| 2014–2017 (calm; worst) | 47 | 10.72% | 7.90% | 2.06%/yr | 0.948 |

Premium level: synthetic / PUT-implied premium median 1.014, aggregate 1.054 (yearly 0.85–1.45).
**The synthetic put-write is rich by ≈ +1.5%/yr (≈ +11 bp/month)**; part of it is methodology
(Cboe rolls at a late-morning VWAP, not the close; strike rule and bill mix not reproduced; q = 1.8%
flat), part is model error. Treat synthetic short-premium P&L as optimistic by that order.

**Synthetic vs the real full SPXW chain on 2019-08-15** (`l3_spx_surface_check`): ATM IV 1.2–1.6 vol
pts low at 6–29 DTE (0.6 at 92 DTE); 25Δ put-call skew 5.0–6.1 vs real 6.1–8.1 vol pts (synthetic skew
≈ 75–80% of real); near-the-money spreads 0.70–1.70 vs real 0.60–0.90 (too wide for longer DTE);
the real chain lists more SPXW Fridays (e.g. 57 DTE) than the modeled 50-day weekly horizon.

**Parameter lookahead statement**: the spread model, smile shape and single-name ATM ratio are fitted
on 2019–2026 data and applied to all dates (structural parameters; no free pre-2019 option quotes
exist). The SPX ATM ratio — the parameter that drives short-premium returns — is point-in-time.
Synthetic smiles are deterministic in (VIX family, SKEW-clipped), so **synthetic chains must not be
used as a source of skew/term-structure SIGNALS** (S166/S167/S168 z-scores would be degenerate).


## 8. Lifecycle (`options.lifecycle`, engine adapter `OccOptionLifecycle`)

* Expiration on the settlement session: PHYSICAL/CASH_PM at the end of the expiry session (official
  close); CASH_AM at the START of the settlement session at the SOQ, proxied by the index opening
  print (flagged T2). Overdue positions (skipped sessions) are settled at the next available print.
* OCC exercise-by-exception: ITM ≥ $0.01 → long exercised, short assigned in full (aggregate holder
  behaviour assumed; conservative). Thresholds before 2008 (customer $0.75/$0.25 eras) are NOT
  reproduced (configurable `ExpiryPolicy.threshold`).
* Physical delivery: shares = `deliverable_shares × |qty|` at the strike; cash-settled: intrinsic ×
  multiplier. Events convert to core `Fill`s (`EXERCISE` / `ASSIGNMENT` / `SETTLEMENT`) or to the
  portfolio's `OptionEvent`s. Exercise/assignment fees are the cost model's job.
* Pin risk: `ExpiryPolicy(pin_band, pin_policy='adverse')` assigns short physical options within the
  band even when marginally OTM (after-hours move / contrarian exercise); default 'occ'.
* Early exercise / assignment (American, evaluated at each session end, whole position):
  call — tomorrow is an ex-date of dividend D, ITM and extrinsic (mid − intrinsic) < D;
  put — ITM and extrinsic < K(1−e^{−rT}) − PV(dividends before expiry).
  Requires the dividend schedule known at the session (`dividends` callable); without it only the
  put rule runs (call/ex-div assignment then NOT modeled — flag it).
* Corporate actions (OCC): integer forward split → contracts ×n, strike ÷n; other ratios (3:2,
  reverse splits) → same contracts/strike, deliverable 100×ratio shares, root `XYZ1`;
  special cash dividend ≥ $12.50/contract → strike − amount; ordinary dividends unadjusted.

## 9. Margin (`options.margin`, engine adapter `CboeStrategyMargin`)

Source: Cboe Margin Manual (Nov 30 2021, cdn.cboe.com/resources/membership/Margin_Manual.pdf), Cboe
Rule 10.3 / FINRA 4210(f)(2). Implemented: long ≤9m paid in full / >9m 75%; naked call/put
(premium + 20% (15% broad index / broad-based ETF) − OTM, min premium + 10% of underlying (calls) /
strike (puts)), leverage factor multiplies both; spreads by the maximum-potential-loss computation
over every strike for whole same-expiry groups (verticals, iron condors/flies = riskier side,
butterflies incl. broken wing, condors, boxes, ratio spreads with surplus longs), calendars/diagonals
only if the long expires with/after the short, short straddle/strangle = greater side + other premium,
covered call/put, cash account (cash-secured puts at aggregate strike; proceeds not applied by
default; naked calls and non-European spreads not permitted).
Every worked example on pp. 23–58 of the manual that falls in scope is a test
(`tests/options/test_margin.py`), including the SMA debit figures.
**Portfolio rule**: `net_requirement = requirement − short option market value`; compliant iff
NAV ≥ stock Reg-T requirement + Σ max(0, net_requirement). Box-spread loan value (50% of width,
European only) is not applied (conservative). House margin is not modeled.
`stress_loss`: predefined shocks (−25%/+20 vol, −15%/+10, ±15%, ±25%, +15/+25% with −5 vol) after one
day of decay, BSM or BS-2002 repricing → worst loss for OPT_NAKED sizing (a lower bound for naked calls).

## 10. Parity & boxes (`options.parity`)

* American bounds (Hull): S − PV(D) − K ≤ C − P ≤ S − Ke^{−rT}.
* `conversion_edge` (buy S at ask, buy P at ask, sell C at bid): PV(K) + PV(D) − outlay − fees;
  worst case over early assignment of the short call on each cum-dividend day.
  `reversal_edge`: proceeds earn r − borrow fee, dividends owed; worst case over put assignment at t=0,
  each ex-date and expiry. `parity_screen` signals only if the worst-case edge ≥ min_edge_bp of spot.
* Boxes: natural buy/sell prices, implied lending/borrowing rate (cc and simple) incl. fees, vs T-bill
  (`tbill_discount_to_cc`). Only European cash-settled boxes are true arbitrage (SPX); American boxes
  can be broken by early exercise (documented, not modeled as riskless).

## 11. Engine plug-ins

`atlasbt.options.plugins.OccOptionLifecycle(policy, rate, dividends)` implements
`portfolio.hooks.OptionLifecycle` (`at_session_start`: AM settlement; `at_session_end`: PM/physical
expiry + early assignment). `CboeStrategyMargin()` implements `portfolio.margin.OptionMarginModel`
(per-underlying strategy margin on current marks; Cboe classes from `underlying_class`). Both are
covered by `tests/options/test_plugins.py`.

## 12. Real option data available

Besides DoltHub (below): `data_raw/historicaloptiondata_free_samples` has ONE full-market EOD day
(2019-08-15, all US options incl. SPX/SPXW/QQQ/IWM, with underlying price) — used for calibration,
not enough for any backtest; `data_raw/gh_quantconnect_lean_data` has minute quotes for a handful of
SPX contracts (2021-01-04, 2021-01-14) and SPY options (2023-08-03) — test fixtures only;
`hf_samples/fabhaus_equities_5m_options` has one month (2024-01) of 5-minute option TRADE bars
(no quotes). None of these makes an intraday strategy testable over a meaningful period.

### DoltHub `post-no-preference/options`

Measured by this agent and the DATA agent (2026-10-28): the table has rows from 2019-02, but probes for
SPY and other liquid names on 2019-02-27, 2019-03-15, 2019-09-13/16 and 2019-12-16 returned none, so
usable coverage for this project starts **2020-01**; row date d ≈ close of the last session ≤ d (exact capture time unknown → T2_PROXY);
**3–4 expiries only** (nearest ≈ 13–16 DTE, ≈ 30 DTE, ≈ 45–65 DTE; minimum DTE in our sample 13,
maximum 65), **≈ 10–30 strikes per expiry** (irregular increments); no underlying price column;
vendor `vol`/greeks present. Underlyings: SPY, DIA, sector SPDRs (XLB…XLY), ~S&P 500 large caps;
**absent: SPX/SPXW/XSP, QQQ, IWM, GLD, TLT, EEM**. No intraday option quotes anywhere in the free data.

Consequences: contracts drop out of the snapshot before DTE ≈ 13 (all "exit at DTE ≤ 7" rules), and
long tenors (90–365 DTE) are never quoted. `options.marking.mark_contract` prices such contracts from
the same day's real surface: T2_PROXY inside the quoted expiry/strike range, T3_MODELED when
extrapolated in maturity or strike; bid/ask from the spread model.

## 13. Fidelity by strategy (best achievable with free data)

T2 = real EOD chain quotes at a proxied time (DoltHub, 2020+); T3 = synthetic or model-extrapolated;
NT = recommend `RunStatus.NOT_TESTABLE`. Pre-2020 everything option-priced is T3 at best.

| id | structure / need | 2020+ real (DoltHub) | notes |
|---|---|---|---|
| S126, S127 | 25–35 DTE ATM call/put, exit DTE ≤ 7 | entries T2; exits/marks at DTE<13 T3 | marking via same-day surface |
| S128 | covered call 25–35 DTE δ.30, buy back DTE ≤ 5 | T2 / T3 at DTE ≤ 13 | assignment via lifecycle |
| S129 | SPX ATM putwrite, AM/PM settlement | T3 (no SPX chains) | synthetic replication of the Cboe PUT index: TE 2.6%/yr, corr 0.98, ≈ +1.5%/yr rich (§7.4); SPY proxy = declared deviation |
| S130, S131 | CSP δ.30 25–45 DTE, wheel | T2 / T3 near expiry | earnings filter needs EARNINGS events |
| S132 | protective put 60–90 DTE | 60–65 DTE T2; 90 DTE T3 | roll at DTE ≤ 30 T2 |
| S133 | collar 25–35 DTE | T2 / T3 at 5–7 DTE | |
| S134–S137 | verticals 25–45 DTE | T2 / T3 at 7 DTE | 5/10-point wings often not listed in sparse strikes → SelectionError |
| S138 | SPX 20Δ/5Δ iron condor | T3 (no SPX) | SPY version T2 = declared deviation |
| S139, S140 | iron flies with point wings | T2 if symmetric wings listed | sparse strikes |
| S141, S142, S155 | butterflies (5/10/20 pt) | T2 if strikes listed | |
| S143, S144 | long straddle/strangle, IV percentile / RV forecast | T2 / T3 at 7 DTE | IV history from 2020 only → 252d percentile valid from 2021 |
| S145, S146 | short strangle / delta-hedged straddle | T2 (EOD hedges) | 1h hedge variant NT (no intraday quotes) |
| S147 | gamma scalp | T2 EOD hedges | intraday rehedge NT |
| S148, S149, S168 | calendars 30/60 (90) DTE | 60 DTE leg T2 (≤ 65), 90 DTE T3 | near-leg exit ≤ 7 DTE T3 |
| S150 | diagonal with 180–365 DTE LEAP | T3 (no LEAP quotes) | |
| S151–S154 | backspreads, risk reversals 30–60 DTE | T2 / T3 at 7 DTE | |
| S156, S157 | jade lizard, covered strangle | T2 | |
| S158, S159 | earnings 3–10 DTE, 15:30 → 09:45 | **NT** | no intraday quotes, no 3–10 DTE expiries; synthetic cannot model event IV crush |
| S160 | post-earnings 10:00 entry, 30 DTE | T2 with close-for-10:00 timing proxy | declare deviation |
| S161–S163 | SPX 0DTE intraday | **NT** (T3 only illustrative) | no SPX, no intraday option quotes; VIX1D daily close cannot capture intraday vol |
| S164 | FOMC 0DTE straddle at recorded NBBO | **NT** | needs timestamped NBBO |
| S165–S167, S169 | IV percentile / skew z / forecast vs implied | T2 (signals from real 30D surfaces, 2020+) | z-scores need 252d history → from 2021 |
| S170 | SPY vs sector-ETF straddles (dispersion) | T2 (all in DoltHub) | vega matching + EOD delta hedges |
| S171 | 1-minute synchronized parity screen | **NT** | EOD snapshots are neither synchronized nor executable intraday |
| S172 | SPX European box financing | **NT** | no SPX quotes; a synthetic box prices at the model rate by construction (circular) |

## 14. Limitations / open items

* Spread model residual sd is large and recent spreads are under-predicted (§7.4): treat synthetic
  transaction costs as uncertain by ×1.6 and run cost stress ×1.5–2.
* Synthetic SPX put-write is ≈ 1.5%/yr rich vs the Cboe PUT index; synthetic 25Δ skew ≈ 75–80% of
  real SPX on the 2019-08-15 check day.
* Synthetic chains have no event (earnings) variance, no per-name skew dynamics beyond (a, b), no
  intraday vol dynamics; vol-index mapping for unmapped names relies on the caller's `rv_ratio`.
* Early-assignment model is a deterministic rule of thumb (no probabilistic assignment).
* Listing facts marked low/medium confidence (§7.1) affect only the earliest weeks of each series.
* House margin, portfolio margin (TIMS) and box loan value not modeled.
