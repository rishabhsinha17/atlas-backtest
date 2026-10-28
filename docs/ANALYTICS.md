# atlasbt analytics: indicators, validation, reporting

Owner: ANALYTICS agent. Packages: `atlasbt/indicators`, `atlasbt/validation`, `atlasbt/reporting`.
Tests: `tests/indicators`, `tests/validation`, `tests/reporting` (`python -m pytest -q tests/indicators tests/validation tests/reporting`).

## 1. General conventions (all indicators)

* **Causality.** The value at position *t* depends only on inputs at positions ≤ *t*. For bar data the
  row is labelled by bar START and becomes usable at the bar END (`known_at`). An indicator value on row
  *t* is therefore usable from row *t*'s `known_at` on. Every public indicator has a perturbation test
  (`tests/indicators/test_ind_causality.py`): future bars are randomly altered and the series is also
  recomputed on the truncated prefix; outputs at ≤ *t* must be identical.
* **No negative shifts.** `_util.shift` refuses negative lags.
* **Warm-up.** NaN until enough data exists. Nothing is back-filled or seeded from future data.
* **Internal NaNs.** Rolling-window functions return NaN for any window containing a NaN. Recursive
  smoothers (EMA, Wilder) skip a NaN observation: they emit NaN on that bar and carry the state forward.
  Nothing is imputed.
* **Types.** Every function accepts a numpy array or a pandas Series. A Series input returns a Series or
  DataFrame on the same index; an array input returns arrays or a dict of arrays.
* **Batch and streaming.** `indicators.streaming` provides O(1)-per-bar state machines (`StreamingSMA`,
  `StreamingEMA`, `StreamingWilder`, `StreamingRSI`, `StreamingATR`, `RollingExtreme`,
  `StreamingSessionVWAP`, `StreamingOpeningRange`). Tests check that they match the batch versions to
  1e-10. Intraday strategies running in `on_bar` should use them.
* **Numba.** Recursive kernels use `numba.njit` when it is installed and plain Python otherwise.

## 2. Indicator definitions

Known-answer references: the StockCharts ChartSchool worked spreadsheets (cs-rsi, cs-movavg, cs-atr,
cs-adx, cs-bbands, cs-cci, cs-soo, cs-percentr), downloaded on 2026-10-28 with UA
`atlasbt-research/0.1` and stored in `tests/indicators/fixtures/stockcharts/*.csv` at full Excel
precision. Our values match them to ≤1e-8 (most to 1e-12). These spreadsheets implement Wilder's
procedures (seeding and smoothing) exactly. Wilder's 1978 book worksheets could not be obtained in
machine-readable form (see Limitations).

| Function | Definition / convention | Reference |
|---|---|---|
| `sma(x,n)` | mean of the n values ending at t | — |
| `ema(x,n, alpha=None, seed='sma')` | α = 2/(n+1) ("span"); **SMA seed**: first value = SMA of the first n values at index n−1, then EMA_t = EMA_{t−1}+α(x_t−EMA_{t−1}). `seed='first'` = pandas `adjust=False` | StockCharts cs-movavg |
| `wilder(x,n)` | α = 1/n, SMA seed (RMA/SMMA) | Wilder 1978 |
| `rsi(c,n,method)` | Wilder: avg gain/loss seeded with the mean of the first n changes (first RSI at index n), then Wilder smoothing. `method='cutler'`: SMA of gains/losses. Edge cases: loss=0 → 100, gain=0 → 0, both 0 → 50 | Wilder 1978 ch. VI; cs-rsi |
| `true_range` | max(H,C_{t−1})−min(L,C_{t−1}); TR_0 = H_0−L_0 | Wilder 1978 |
| `atr(h,l,c,n,method)` | Wilder: seed = mean(TR_0..TR_{n−1}) at n−1 (matches cs-atr); `'sma'` = n-bar mean of TR (S061); `'ema'` | cs-atr |
| `dmi/adx` | +DM/−DM/TR from bar 1; Wilder running-sum smoothing (S−S/n+x); DI = 100·DM_n/TR_n; DX = 100·\|DI+−DI−\|/(DI++DI−); ADX = mean of the first n DX values (index 2n−1), then Wilder | Wilder 1978 ch. IV; cs-adx (504 bars) |
| `macd(12,26,9)` | EMA_fast − EMA_slow (SMA-seeded); signal = SMA-seeded EMA of MACD; hist = MACD − signal | Appel |
| `bollinger(c,20,2,ddof=0)` | **population** σ (ddof=0) as specified by Bollinger and used by StockCharts (the test fails with ddof=1). bandwidth = (U−L)/mid as a fraction (StockCharts shows ×100); %b | Bollinger 2001; cs-bbands |
| `bandwidth_percentile(c,20,2,252)` | mid-rank percentile of today's bandwidth among the **previous** 252 values (S056) | — |
| `keltner(h,l,c,n,k,atr_n)` | EMA_n(close) ± k·ATR_{atr_n} (Wilder ATR; atr_n defaults to n; the spec uses ATR20) | Keltner/Raschke |
| `squeeze_on` | BB fully inside KC | Carter |
| `donchian(h,l,n,include_current=False)` | default = the **prior** n bars t−n..t−1 (the spec's "prior N-day high") | Donchian |
| `stochastic(h,l,c,14,3,3)` | fast %K = 100(C−LL)/(HH−LL); slow %K = SMA3(fast %K); slow %D = SMA3(slow %K); HH = LL → NaN | Lane; cs-soo |
| `williams_r` | −100(HH−C)/(HH−LL) | Williams; cs-percentr |
| `cci(h,l,c,20,0.015)` | TP=(H+L+C)/3; (TP−SMA)/(0.015·MD); MD = mean absolute deviation around the **current** SMA | Lambert 1980; cs-cci (two sheets) |
| `parabolic_sar(af0=.02, step, af_max=.2)` | Wilder rules in the TA-Lib reference form. Reversal when price touches or penetrates the SAR. The next SAR is never inside the current or previous bar's range. On reversal SAR = prior EP (clamped outside the current and prior bar), AF resets. Initial direction: short if the first bar pair shows a dominant −DM, else long. Outputs `sar` (in force during t), `next_sar` (stop for t+1, known at t's close; used for gap-aware stop orders), `trend`, `flip`, `af`, `ep` | Wilder 1978 ch. II; hand-traced test |
| `supertrend(atr_n=10, mult=3)` | hl2 ± mult·ATR (Wilder); ratcheting final bands; initial state downtrend; identical logic to TradingView `ta.supertrend` | Seban; hand-traced test |
| `ichimoku(9,26,52,26)` | See §2.1 | Hosoda |
| `nr_n(h,l,7)` | today's range ≤ all of the previous n−1 ranges (ties count) | Crabel 1990 |
| `inside_bar(h,l,nested)` | H<H₋₁ and L>L₋₁ (strict); `nested=k` → k consecutive | — |
| `ibs` | (C−L)/(H−L); NaN when H=L | — |
| `zscore(x,n,ddof=0)` | (x−SMA)/σ_pop, so z≤−k is the same condition as close < lower BB(n,k) | — |
| `linreg_slope(x,n)` | OLS slope on 0..n−1 (optionally ÷ window mean) | — |
| `percentile_rank(x,w,include_current)` | (#below + ½#ties)/#valid in the reference window; `include_current=False` → previous w values only | — |
| `realized_vol(close,n,annualize,log,ddof=1,lag)` | rolling sample σ of returns; `lag=1` = prior completed sessions (S032/S033) | — |
| `vol_scaled_to_time(σ_d, minutes, 390, overnight_share)` | σ_d·√(s_on + (1−s_on)·min(m/390,1)). Use `overnight_variance_share` (causal rolling var(close→open)/var(close→close)) when measuring from the prior close (S026) | square-root-of-time |
| `high_ratio(c,h,252)` | C_t / max(H over the 252 bars ending at t) | George & Hwang 2004 |
| `prior_all_time_high(h,min_history)` | max of all prior split-adjusted highs (excluding t) once ≥ min_history bars exist. Input must be the as-of-now adjusted series served by the DataPortal | — |
| `pivots(h,l,left,right)` | pivot high at s: H_s > max(left window), H_s ≥ max(right window). **Reported at the confirmation bar t = s+right** (`ph`, `ph_bar`, `ph_price`, `last_ph`) | — |
| `double_bottom` | two confirmed pivot lows separated by min..max bars, within tol, no lower low between them; neckline = max high between. Signal on the first *fresh* close above the neckline **after** the second low is confirmed. A break that happened before confirmation is not chased | S027 |
| `head_and_shoulders` | three consecutive confirmed pivot highs (lag 3/5); head−max(shoulders) ≥ k·ATR; \|LS−RS\| ≤ k·ATR (ATR at RS confirmation); neckline through the two reaction lows, extrapolated; short on a fresh close below it after RS confirmation | S067 |
| `flag_pattern` | per session: impulse ≥ x% from the open within N bars; flag = bars after the peak; flag length, retracement band and flag volume < impulse volume (sum by default per the spec, or mean) are all required; signal on close > flag high; one per session | S010 |
| `darvas_box` | R = max(H_{t−20..t−1}); max(H_{t−5..t−1}) < R; min(L_{t−5..t−1}) > R − depth·ATR_{t−1}; signal on C_t > R | S060 |
| `volatility_contraction(n=10, n_means=3)` | ATRn_t < ATRn_{t−n} < ATRn_{t−2n} (n-bar mean TR) | S061 |
| `bullish_engulfing`, `down_streak` | as in S066 / S073 | — |

### 2.1 Ichimoku: information-time alignment (S062)

Charts draw the Senkou spans *displacement* bars **ahead** and the Chikou span *displacement* bars
**behind**. A frame holding "what is drawn on date d" leaks the future through Chikou (the close of
d+26). Our output therefore holds only values known at t:
`senkou_a_lead`/`senkou_b_lead` are computed at t and drawn at t+26. `cloud_a`/`cloud_b`/`cloud_top`/
`cloud_bottom` are **the cloud drawn at t**, which is the lead values computed at t−26 (`shift(+26)`).
`chikou_ref = close_{t−26}`: compare close_t with it instead of plotting close_t back in time. S062's
"price exceeds previously plotted known cloud" is `close_t > cloud_top_t`. TradingView-style offset
(displacement−1) = `displacement=25`. A test confirms that `cloud_a` equals ((tenkan+kijun)/2).shift(26).

### 2.2 Intraday (session-aware) functions: `indicators.intraday`

The input is a canonical bars frame (UTC index of bar starts, a `session` column if available). The session key is the ET date.
* `session_vwap(price='typical'|'close'|'ohlc4'|'bar_vwap')` returns `vwap`, the volume-weighted **population**
  `sd` = √(Σv p²/Σv − vwap²), `z`, and `cum_volume`. It resets every session. Prices are centred on the
  session's first price before accumulating, to avoid catastrophic cancellation. `bar_vwap` uses the
  feed's per-bar VWAP and is the most faithful choice. `vwap_bands(k)` = vwap ± k·sd (S014/S023/S028).
* `window_stats(start,end)` / `premarket()` give per-session high, low, VWAP, volume, first_open and last_close,
  plus `known_at` = the end of the last bar in the window (S005–S007).
* `opening_range(minutes)` is defined on a row only once bar end ≥ open+minutes. Earlier rows are NaN.
* `session_extremes` returns hod/lod and `hod_prior`/`lod_prior` through the previous bar (S016).
* `time_of_day_profile` / `relative_volume(lookback=20, cumulative, stat='median')`: the expected value for a
  slot is the median of the same ET minute-slot over the **previous** `lookback` sessions. The current
  session is excluded (`shift(1)` over sessions). `cumulative=True` compares session-to-date volume
  (S001 "09:35 RVOL(5m)", S016 "cumulative volume vs same-time median"). Early-close sessions simply have
  no late slots.
* `adv(volume,20,lag)` and `dollar_volume(close,volume,20,lag)`: use `lag=1` intraday (today is incomplete).

### 2.3 Stat-arb: `indicators.statarb`

* `rolling_ols(y, X, w)`: betas fit on [t−w+1, t] with NaN rows dropped (`min_periods` defaults to w valid
  rows). `resid` uses β_t. `resid_oos` uses β_{t−1}, which is strictly out of sample. Use it with market and
  sector regressors for S082/S088/S109.
* `kalman_hedge(y, x, q_ratio, obs_var, warmup)`: the state (β, α) is a random walk with Q = q_ratio·R·I (the spec's
  "Q/R ratio"). **One step ahead:** the innovation e_t = y_t − H_t·state_{t|t−1} and z_t = e_t/√S_t use
  the prior state. The state is updated only after observing y_t (Joseph form). With `obs_var=None`, R and the
  prior come from OLS on the first `warmup` points and outputs start after them. Trade at the next open
  on `beta`/`z` of t. Clipping to [0.5, 2] (S108) is the strategy's job.
* `engle_granger` (OLS hedge + statsmodels `coint` MacKinnon p-values), `adf`.
* `pca_residuals(rets, 252, k)`: loadings come from the correlation matrix of [t−w, t−1] (through t−1), r_t is
  standardised with the window moments and projected on the top-k eigenvectors. The residual is mapped back to
  return units (net of the window mean). `refit_every` reuses past fits only.
* `ssd_pairs` / `select_pairs` (greedy, non-overlapping) / `pair_spread_z` (formation-window mean and σ
  frozen at formation end): Gatev, Goetzmann & Rouwenhorst (2006).

### 2.4 Cross-section: `indicators.cross_section`

`quantile_bucket(x, q)`: average ranks, bucket = ⌈rank·q/n⌉ (1 = lowest). **Ties share a bucket.**
NaNs are never ranked. With `universe_mask` (e.g. NYSE-only breakpoints, Fama & French 1993) the
breakpoints are the k/q quantiles (numpy linear) of the universe. A value equal to a breakpoint goes to the
**lower** bucket. Also available: `group_zscore` (industry/sector-neutral; NaN for groups with fewer than 2 names),
`group_demean`, `group_rank`, `group_quantile_bucket` (within-sector sorts), `winsorize` (quantile clip,
optionally per group), `size_buckets` (NYSE breakpoints), `value_weights`, `long_short`.
Panel functions work row by row, so each date uses only its own cross-section.

## 3. Validation (`atlasbt.validation`)

* **Walk-forward** (`walkforward`). Splits are rolling or anchored (`walk_forward_splits(index, train, test, step,
  anchored, holdout)`). The **holdout** (a fraction, a bar count or a date) is removed before any fold is built.
  `walk_forward_matrix` (one return column per grid combination; each full-sample run is causal) or
  `walk_forward_run(run(params,start,end))` (fresh runs per window) picks the best train-window combination
  (objective: Sharpe, total return, mean or a callable; ties go to the earliest column) and stitches the OOS test
  segments together. The holdout is evaluated **once**, with the combination selected on the train window just before it.
  A test checks that absurd holdout returns do not change any fold's choice. Reported: OOS Sharpe, WF efficiency
  (mean test / mean train metric) and parameter stability.
  Caveat for matrix mode: positions and warm-up state carried into a test window come from the full-sample run.
* **Purged k-fold** (`cv.PurgedKFold`), López de Prado (2018) ch. 7. Each observation has an interval [t0,t1].
  Training observations that overlap the test span are purged, and an embargo of ⌈n·pct⌉ observations after the
  test block is dropped.
* **Bootstraps** (`bootstrap`). Stationary (Politis & Romano 1994; geometric blocks with mean b, wrap-around) and
  circular block (Politis & Romano 1992). **Automatic b** uses Politis & White (2004) with the Patton, Politis &
  White (2009) correction (flat-top window, m = 2·m̂, K_n = max(5, ⌈log10 n⌉), band 2√(log10 n/n),
  b_SB = (2G²/D_SB)^{1/3}n^{1/3}, D_SB = 2g², D_CB = (4/3)g², capped at min(3√n, n/3)). It is tested against AR(1)
  theory (b = (2φ/(1−φ²))^{2/3} n^{1/3}) and against `arch`. `bootstrap_ci/metrics` give percentile (or basic) CIs for
  Sharpe, CAGR, MaxDD or any callable. A coverage test shows the stationary bootstrap reaches ≈ nominal coverage on AR(1)
  data, while the iid bootstrap undercovers.
* **PSR / DSR** (`sharpe_tests`). PSR(SR*) = Φ((SR−SR*)√(T−1)/√(1−γ₃SR+(γ₄−1)/4·SR²)) with
  **per-period** SR and **non-excess** kurtosis (Bailey & López de Prado 2012). DSR = PSR(SR₀) with
  SR₀ = √V[SR_n]·((1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(Ne))) (Bailey & López de Prado 2014). Test: the paper's example
  (SR_ann 2.5, T=1250, skew −3, kurt 10, N=100, V_ann=0.5) → SR₀ = 0.1132 and **DSR = 0.9004**. MinTRL is also provided.
  **Trial count for the WHOLE atlas:** `atlas_trials()` gives every strategy's grid size, from the strategy
  registry's `grid` when supplied and otherwise parsed from `docs/strategies.json` "Parameters to test" (product of the
  '/'-separated options per ';'-parameter). The parsed total is **1450** configurations over 172 strategies. Every DSR in the
  leaderboard is deflated with this total N. V[SR_n] defaults to the cross-strategy variance of per-period Sharpes.
  `effective_trials(N, ρ̄)` = ρ̄ + (1−ρ̄)N is available for correlated trials.
* **PBO via CSCV** (`pbo.pbo_cscv`), Bailey, Borwein, López de Prado & Zhu (2017). There are S (even, default 16) equal
  contiguous blocks. The first T mod S rows are dropped so the most recent data are kept. All C(S,S/2) train/test
  splits are used (or a random subset). n* = argmax train Sharpe; ω̄ = rank_test(n*)/(N+1) with average ranks;
  λ = logit ω̄; PBO = share of λ ≤ 0. Also reported: degradation slope and the probability of OOS loss. Tests: a
  constructed always-overfit case → PBO = 1; one genuinely skilled strategy → PBO ≈ 0; pure noise → mean PBO ≈ 0.5.
* **White's Reality Check / Hansen SPA** (`multiple_testing`). The input is performance differentials d[t,k] (strategy minus
  benchmark, e.g. excess over cash), with shared stationary-bootstrap indices. SPA is studentised with the
  stationary-bootstrap kernel variance ω² = γ₀ + 2Σκ(n,i)γ_i, κ = (1−i/n)(1−q)^i + (i/n)(1−q)^{n−i}, and gives
  lower / consistent / upper p-values (re-centring max(d̄,0); d̄·1{d̄ ≥ −ω√(2 log log n/n)}; d̄).
  RC = non-studentised upper. Tested against `arch.bootstrap.SPA` (within Monte Carlo error).
* **FDR** (`benjamini_hochberg`, BH 1995; `method='by'` for Benjamini–Yekutieli 2001). q-values match
  statsmodels. Leaderboard p-values are p = 1 − PSR(0).
* **Regimes** (`regimes`). All labels are **ex ante** (lag 1): BULL/BEAR = SPY close vs SMA200; VIX terciles from
  expanding-window cut points (`causal=True`, or full-sample for description only); rate level buckets (<1%, 1–3%,
  ≥3%, optional rising/falling); decades. `regime_table` gives days, share, annualised mean/vol, Sharpe, compounded return,
  hit rate and MaxDD per regime.

### 3.1 Cost Stress Calculator (sheet "Cost Stress Calculator")

The sheet gives **All-in round-trip bp = 2·Extra + MeasuredQuotedSpread + 2·CommissionPerSide**, with scenarios
**A–E = 0, 2, 5, 10, 25 extra bp per side**. Its illustrative inputs are 1.5 bp spread and 0.1 bp commission, i.e.
1.7 / 5.7 / 11.7 / 21.7 / 51.7 bp. The tests reproduce these exactly from fills built with those costs.

How it is applied (`cost_stress.apply_cost_stress`). This is exact for the cash flows and first-order for behaviour:
1. *Measured* costs come from the fills, notional-weighted: quoted spread (round trip) = 2·Σspread_cost/Σnotional;
   commission per side = Σ(commission+exchange+regulatory)/Σnotional; impact per side is reported separately.
   These costs are **already** in the backtest (half-spread and impact inside the fill price, fees debited).
2. The stress adds c_i = extra_bp·1e−4·|qty_i|·price_i·multiplier_i to **every fill**. For options the base is the
   **premium** notional.
3. NAV re-pricing: each fill is assigned to the first NAV stamp ≥ its timestamp. `mode='proportional'`
   (default; sizing scales with NAV): r'_t = r_t − Σc_i/NAV_{t−1}, NAV' = NAV₀·Π(1+r'). `mode='additive'` (fixed
   dollar sizing): NAV'_t = NAV_t − Σ_{≤t}c_i.
4. The trade list is re-priced as well: pnl' = pnl − extra·(entry+exit notional), R' = pnl'/(pnl/R).
5. Signals, sizes and fills are **not** re-simulated. A strategy whose decisions depend on its own NAV can diverge
   slightly; rerun the engine with a cost add-on for that.

`cost_stress_table` returns the sheet's columns plus CAGR, Sharpe, MaxDD and the $ extra cost per scenario.
`break_even_cost(metric='cagr'|'sharpe'|'total_return', target=0)` uses Brent's method to find the extra bp per side
at which the metric reaches the target. Its status is `ok`, `already_below`, `never_within_hi` or `no_trades`.

## 4. Reporting (`atlasbt.reporting`)

### 4.1 Result schema read by analytics (`reporting.schema`)
`BacktestResult` names its frames but not their columns, so analytics reads them through tolerant normalisers.
The canonical columns are:
* fills: `ts, order_id, symbol, asset_class, multiplier, quantity, price, commission, exchange_fees,
  regulatory_fees, spread_cost, impact_cost, fidelity, liquidity, note, reference_mid, adv, notional`. Also accepted:
  a list of `core.types.Fill`, an `instrument` column, a `costs` column of `CostBreakdown`.
* trades: `symbol, entry_ts, exit_ts, quantity, entry_price, exit_price, pnl (net), r_multiple, mae, mfe, bars_held,
  multiplier`.
* ledger: `ts, amount (+ inflow), reason`. Reasons are bucketed by keyword: BORROW/LOCATE → borrow;
  MARGIN/DEBIT_INTEREST → financing_debit; CASH/CREDIT_INTEREST/REBATE → financing_credit; DIVIDEND/IN_LIEU → dividends.
* positions: `ts, symbol, quantity, market_value`.
* risk-free: `meta['rf_daily']` (per-period simple return series) or `meta['risk_free_rate']` (annualised simple rate,
  accrued act/360 between NAV stamps using the rate known at the previous stamp). If neither is present, rf = 0 and this
  is reported as `rf_source='none'`. `meta['initial_capital']` is used as NAV₀ when present.

### 4.2 Metrics (`metrics.compute_metrics`)
* CAGR uses calendar time from NAV₀. Annualised vol uses 252 periods and ddof 1. Sharpe and Sortino use returns in excess
  of rf; Sortino's downside deviation is √mean(min(r−rf,0)²).
* Calmar = CAGR/|MaxDD|. MaxDD is measured on the NAV path including NAV₀. **Duration** = the longest span from a peak to
  the first bar back at that peak (or to the end, flagged unrecovered), in bars and calendar days.
* Ulcer index = √mean(DD%²).
* Trades: hit rate, payoff (mean win / |mean loss|), profit factor, expectancy in $ and **in R**, average and median holding
  (calendar days; bars if given).
* Turnover = Σ|fill notional| / mean NAV / years (two-sided; the one-sided figure is half).
* Exposure: time in market, and mean/max gross and net exposure as a fraction of NAV averaged over **all** days.
* **Cost drag**: commission, exchange, regulatory, spread and impact (from fills); borrow, financing debit/credit and dividends
  (from the ledger). Each is given in $ and in bp per year of mean NAV.
* **Capacity**: participation = |qty|/ADV per fill. Capacity NAV = mean NAV·limit/p95(participation), with limit 5% by default.
  This is linear and ignores √-impact growth, so it is an optimistic upper bound, and is flagged as such.
* **Tails**: skew, excess kurtosis, historical VaR and CVaR at 95/99, best and worst day.
* **β/α vs SPY**: OLS of excess returns on SPY excess returns with Newey–West HAC t-stats (lags = ⌊4(n/100)^{2/9}⌋),
  tracking error, IR and correlation. **Fama–French**: regression on the supplied factor columns (an `RF` column is used as rf).
* Each report also carries PSR(0).

### 4.3 Fidelity scorecard (`reporting.fidelity`)
It accepts `{'tiers': {...}, 'reasons': ...}`, tier-keyed dicts, or recomputes from the fills' `fidelity` column.
**Grade** = notional-weighted mean tier w: A ≤ 0.25 (quote level), B ≤ 1.0 (real bars, modelled spread), C ≤ 1.75,
D ≤ 2.5, E > 2.5 (mostly modelled). NA is assigned when any notional is at tier NA or the run is NOT_TESTABLE. Rendered as a
markdown table with reasons sorted by count.

### 4.4 Per-strategy report (`report.write_strategy_report`)
Writes `<out>/<SID>/report.md`, `report.json` and the PNGs `equity` (log), `drawdown`, `rolling_sharpe` (126d), `monthly_returns`
(diverging heat-map) and `r_distribution`. The headline table shows **DSR (with the atlas trial count) and PSR first**, then the
raw metrics. Sections follow for bootstrap CIs, cost stress A–E with break-even, cost drag, capacity, benchmark and factor
regressions, tails, regimes and fidelity. NOT_TESTABLE runs produce a short report with the reason.

### 4.5 Leaderboard (`leaderboard.build_leaderboard / write_leaderboard`)
Rows come from `summarize_result` (optionally with PBO, walk-forward OOS Sharpe and break-even bp). DSR is computed with the
**total atlas trial count** (max of the summed registry grids and the parsed 1450). q-values are BH over all tested strategies.
**Sort: DSR ↓, then PBO ↑, then q ↑, then PSR ↓.** Raw Sharpe is shown for reference only and never sets the rank. NOT_TESTABLE
and FAILED strategies are listed separately with their reasons (`not_testable.csv`, plus a markdown section). The test builds a
case where the strategy with the highest raw Sharpe (short sample, crash skew) ranks below a long, well-behaved one.

## 5. Limitations
* Wilder's original 1978 worksheet values were not available in machine-readable form. The StockCharts spreadsheets (which
  follow Wilder's seeding exactly) serve as the published reference. Parabolic SAR and Supertrend have no downloadable reference
  table (ChartSchool shows them as images), so they are verified by hand-traced tests.
* The cost stress is first-order: no re-simulation of signals or sizing, and options are stressed on premium notional.
* Capacity is a linear participation bound, not an impact-optimal one.
* Walk-forward matrix mode carries full-sample state into test windows. Use callable mode for slow strategies.
* The parsed trial count approximates grids written in prose. Pass registry grids (`atlas_trials(registry_grids=...)`) once
  strategies exist.
* Pattern detectors (double bottom, H&S, flag) encode one explicit reading of the sheet text and are documented in their docstrings.

## 6. References
Wilder (1978) *New Concepts in Technical Trading Systems*; StockCharts ChartSchool worked spreadsheets; Appel (MACD); Lane
(stochastics); Lambert (1980) CCI; Bollinger (2001); Crabel (1990); George & Hwang (2004); Gatev, Goetzmann & Rouwenhorst
(2006); Engle & Granger (1987); MacKinnon (2010); Avellaneda & Lee (2010); Chan (2013); Fama & French (1993); Politis & Romano
(1992, 1994); Politis & White (2004); Patton, Politis & White (2009); Bailey & López de Prado (2012, 2014); Bailey, Borwein,
López de Prado & Zhu (2017); López de Prado (2018) *AFML*; White (2000); Hansen (2005); Benjamini & Hochberg (1995); Benjamini &
Yekutieli (2001); Sortino & Price (1994); Martin (1987) Ulcer index; Newey & West (1987).
