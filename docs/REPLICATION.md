# End-to-end replication (known-answer) results

Generated 2026-10-29T12:05:59.966614+00:00 | git 8985af142b09 | code fa45593678e33f83. Every number below comes from the real engine over the real cache (`python -m atlasbt.run replicate`); functions in `atlasbt/run/replicate.py`, quick-window pytest in `tests/run/test_replicate.py`.

> **Active notices**
> * **CALIBRATED-SPREADS**: Taker costs use the engine's calibrated spread model (v3 + 2010-2025 pessimistic floor, T2 'spread modelled (calibrated)'): SPY coin-flip round trip ~1.0 bp vs 31 bp under the legacy Corwin-Schultz model; per-fill slippage 0.26 bp vs 0.31 bp quoted (IEX 2019-12-24).
> * **STOCK-COVERAGE**: Stock-universe (S&P 500 PIT) primary results start at the first month with >= 95% PIT price coverage; members without verified data are excluded by the engine (coverage recorded per result, flagged T2). EXTENDED_SURVIVORSHIP_BIASED secondary runs (>= 60% coverage, results/extended) are never ranked: delisted and failed names are under-represented, so they are likely optimistic.
> * **SYNTHETIC-FILLS-T3**: Options filled from synthetic session-close snapshots (no real chain: SPX/QQQ/IWM, SPY before 2019/2020) are T3_MODELED (OPTIONS.md §7.4: synthetic put-writes ~+1.5%/yr rich vs the Cboe PUT index); the fidelity scorecard shows the T2/T3 split.
> * **QUOTES-AUTO**: use_quotes='auto': quote fills only where cached quotes cover the bar window (IEX 2019-12-24, Dukascopy 2026-10-27 sample days); bars elsewhere.

| Check | Result | Key numbers |
|---|---|---|
| SPY buy & hold vs Adj Close | PASS | dividends as cash: TD -1.75%/yr (zero comm. -1.75%); reinvested: TD -0.02%/yr; accounting residual 0.01% of NAV |
| S105 60/40 vs pandas | PASS | corr 1.000, TD -0.08%/yr, TE 0.14% |
| S129 (engine) vs Cboe PUT | PASS | ann 6.00% vs 7.45%, TE 4.06%/yr, corr 0.929 |
| 12-2 momentum decile spread vs UMD | PASS | corr 0.888 over 320 months |
| B/M tercile spread vs HML | PASS | corr 0.820 over 195 months |
| Random entries (coin flip) | PASS | gross(mid) -1.0 bp (t -0.317), net -2.6 bp, cost 1.6 bp per trip; fills better than mid: 0 |
| Spread model vs real quotes (SPY, 2019-12-24) | EVIDENCE | IEX quoted half-spread median 0.3 bp vs engine bar-mode (Corwin-Schultz) 4.5 bp = 15x; engine slippage per fill by year under the default model (calibrated): {2010: 0.88, 2011: 0.79, 2012: 0.73, 2013: 0.61, 2014: 0.52, 2015: 0.48, 2016: 0.48, 2017: 0.41, 2018: 0.37, 2019: 0.36, 2020: 0.33, 2021: 0.24, 2022: 0.25, 2023: 0.24, 2024: 0.21, 2025: 0.2, 2026: 0.17} |
| SPY coin-flip cost: calibrated vs Corwin-Schultz | PASS | round trip 1.0 bp (calibrated) vs 31.2 bp (CS) = 30.6x; per-fill slippage 0.3 bp vs quoted half-spread 0.3 bp |
| Perfect foresight vs PIT guard | PASS | future_as_of: raised; future_actions: raised; clipped_end: leaks=0; history_peek: leaks=0 |

## 1. SPY buy-and-hold

Criteria: accounting residual < 1 bp of NAV (dividends-as-cash runs); reinvesting run within 20 bp/yr of the Adj-Close total return

```json
{
 "window": [
  "1993-02-01",
  "2026-10-27"
 ],
 "default_fees": {
  "status": "OK",
  "runtime_sec": 17.9,
  "fill_date": "1993-02-02",
  "shares": 20130.0,
  "fill_price": 44.24,
  "fill_reference_mid": 44.21875,
  "entry_cost_bp": 4.805653710246993,
  "entry_fees_usd": 134.92278442499997,
  "engine_cagr": 0.09041720218973559,
  "adjclose_cagr": 0.10790791345971984,
  "td_vs_adjclose_ann": -0.01749071126998425,
  "years": 33.73032169746749,
  "n_fills": 1,
  "accounting_residual_max_abs_frac_nav": 6.840559769395064e-05,
  "accounting_residual_end_usd": 67.26154442876577,
  "cash_interest_total_usd": 688618.6995022035,
  "later_fills": 0,
  "dividends_total_usd": 2282238.7499999995,
  "dividends_as_cash_cagr_no_interest": 0.08919883245317362,
  "div_reinvestment_drag_ann": -0.018709081006546224,
  "cash_interest_contrib_ann": 0.001218369736561975
 },
 "zero_commission": {
  "status": "OK",
  "runtime_sec": 18.6,
  "fill_date": "1993-02-02",
  "shares": 20130.0,
  "fill_price": 44.24,
  "fill_reference_mid": 44.21875,
  "entry_cost_bp": 4.805653710246993,
  "entry_fees_usd": 0.0,
  "engine_cagr": 0.09041333764237902,
  "adjclose_cagr": 0.10790791345971984,
  "td_vs_adjclose_ann": -0.01749457581734082,
  "years": 33.73032169746749,
  "n_fills": 1,
  "accounting_residual_max_abs_frac_nav": 6.83962016719493e-05,
  "accounting_residual_end_usd": 67.2615444213152,
  "cash_interest_total_usd": 688764.3471372694,
  "later_fills": 0,
  "dividends_total_usd": 2282238.7499999995,
  "dividends_as_cash_cagr_no_interest": 0.08919472800966854,
  "div_reinvestment_drag_ann": -0.018713185450051295,
  "cash_interest_contrib_ann": 0.0012186096327104767
 },
 "reinvest_default_fees": {
  "status": "OK",
  "runtime_sec": 25.6,
  "fill_date": "1993-02-02",
  "shares": 20130.0,
  "fill_price": 44.24,
  "fill_reference_mid": 44.21875,
  "entry_cost_bp": 4.805653710246993,
  "entry_fees_usd": 134.92278442499997,
  "engine_cagr": 0.1077302084208891,
  "adjclose_cagr": 0.10790791345971984,
  "td_vs_adjclose_ann": -0.0001777050388307444,
  "years": 33.73032169746749,
  "n_fills": 548
 },
 "fee_effect_ann": 3.864547356569048e-06,
 "pass": true,
 "criteria": "accounting residual < 1 bp of NAV (dividends-as-cash runs); reinvesting run within 20 bp/yr of the Adj-Close total return",
 "elapsed_sec": 64.6
}
```

## 2. S105 60/40

Criteria: daily return correlation > 0.99 and |annualised tracking difference| < 50 bp

```json
{
 "window": [
  "2003-07-30",
  "2026-10-27"
 ],
 "status": "OK",
 "runtime_sec": 37.2,
 "first_fill": "2003-07-30",
 "engine_cagr": 0.08298065912939712,
 "reference_cagr": 0.0838158952936503,
 "td_ann": -0.0008352361642531836,
 "daily_corr": 0.9999803174687617,
 "tracking_error_ann": 0.0014375216160219797,
 "n_days": 5848,
 "n_fills": 556,
 "costs_usd": 813.7965381150001,
 "spread_impact_usd": 1075.5234567260386,
 "cash_interest_usd": 12607.041853982959,
 "pass": true,
 "criteria": "daily return correlation > 0.99 and |annualised tracking difference| < 50 bp",
 "elapsed_sec": 41.7
}
```

## 3. Cboe PUT index via S129

Criteria: monthly correlation > 0.9 and tracking error < 5%/yr (options agent: 2.6%/yr without the engine)

```json
{
 "window": [
  "2008-01-02",
  "2026-10-27"
 ],
 "status": "OK",
 "runtime_sec": 482.0,
 "params": {
  "underlying": "SPX",
  "strike_rule": "atm",
  "put_delta": 0.35,
  "close_before_settlement": false,
  "close_dte": 1,
  "roll_window": 4
 },
 "quote_path": "production Portal.option_quotes (synthetic fallback)",
 "n_fills": 451,
 "months": 224,
 "engine_ann_return": 0.05998541534844248,
 "put_index_ann_return": 0.07448181260958053,
 "diff_ann": -0.01449639726113805,
 "tracking_error_ann": 0.04058286511412226,
 "monthly_corr": 0.9289066046721728,
 "mean_notional_coverage": 0.9992950707351103,
 "fidelity": "D",
 "options_agent_reference": "calibrate.replicate_put_index (no engine): TE 2.63%/yr, corr 0.979, +1.5%/yr rich (2008-01..2026-09, at mid)",
 "pass": true,
 "criteria": "monthly correlation > 0.9 and tracking error < 5%/yr (options agent: 2.6%/yr without the engine)",
 "elapsed_sec": 482.3
}
```

## 4a. Momentum vs UMD

Criteria: positive correlation (> 0.3) with UMD; < 1 expected (S&P 500 only, equal weight, deciles)

```json
{
 "window": [
  "2000-02-29",
  "2026-09-30"
 ],
 "months": 320,
 "corr_full": 0.8880979640536154,
 "runtime_sec": 740.1,
 "spread_ann_mean": -0.01291533077807481,
 "umd_ann_mean": 0.0243525,
 "median_names": 393.0,
 "corr_2000_2011": 0.890361457865197,
 "corr_2012_2026": 0.8895166706169767,
 "pass": true,
 "criteria": "positive correlation (> 0.3) with UMD; < 1 expected (S&P 500 only, equal weight, deciles)",
 "elapsed_sec": 740.4
}
```

## 4b. Value vs HML

Criteria: positive correlation (> 0.3) with HML (S&P 500 only, XBRL book equity 2009+)

```json
{
 "window": [
  "2010-07-31",
  "2026-09-30"
 ],
 "months": 195,
 "corr_full": 0.8196789154062302,
 "median_names": 431.0,
 "spread_ann_mean": -0.01017393314454488,
 "hml_ann_mean": -0.00779076923076924,
 "runtime_sec": 663.7,
 "pass": true,
 "criteria": "positive correlation (> 0.3) with HML (S&P 500 only, XBRL book equity 2009+)",
 "elapsed_sec": 664.2
}
```

## 5. Random entries

Criteria: |t| of mean gross (mid-to-mid) trip return < 3; no fill better than its reference mid; net = gross - costs with costs > 0

```json
{
 "window": [
  "2010-01-04",
  "2026-10-27"
 ],
 "symbol": "SPY",
 "spread_model": "calibrated",
 "seeds": 8,
 "n_round_trips": 4828,
 "gross_mid_mean_bp": -1.0148521552041343,
 "gross_mid_t": -0.31726845017328464,
 "net_mean_bp": -2.6232280530285674,
 "cost_mean_bp": 1.6083758978244338,
 "long_share": 0.4987572493786247,
 "gross_mid_mean_long_bp": 25.47370895700172,
 "gross_mid_mean_short_bp": -27.37206503048996,
 "fills": 9670,
 "fills_better_than_mid": 0,
 "mean_slippage_bp": 0.4274371708066115,
 "per_seed": [
  {
   "seed": 0,
   "status": "OK",
   "n_trips": 605,
   "nav_end": 644692.7087815662,
   "mean_net_bp": -18.326690013567347
  },
  {
   "seed": 1,
   "status": "OK",
   "n_trips": 607,
   "nav_end": 1137871.9706342812,
   "mean_net_bp": 0.33360430794396345
  },
  {
   "seed": 2,
   "status": "OK",
   "n_trips": 603,
   "nav_end": 970957.5754296796,
   "mean_net_bp": -5.556429168294035
  },
  {
   "seed": 3,
   "status": "OK",
   "n_trips": 606,
   "nav_end": 859550.1575282181,
   "mean_net_bp": -8.559048724218703
  },
  {
   "seed": 4,
   "status": "OK",
   "n_trips": 600,
   "nav_end": 1400243.966654336,
   "mean_net_bp": 6.176880042211872
  },
  {
   "seed": 5,
   "status": "OK",
   "n_trips": 602,
   "nav_end": 900872.2226296712,
   "mean_net_bp": -8.231779058892227
  },
  {
   "seed": 6,
   "status": "OK",
   "n_trips": 599,
   "nav_end": 1290018.8194062538,
   "mean_net_bp": 4.899239283788537
  },
  {
   "seed": 7,
   "status": "OK",
   "n_trips": 606,
   "nav_end": 1474478.6138632458,
   "mean_net_bp": 8.37008837583966
  }
 ],
 "runtime_sec": 298.5,
 "slippage_bp_by_year": {
  "2010": 0.88,
  "2011": 0.79,
  "2012": 0.73,
  "2013": 0.61,
  "2014": 0.52,
  "2015": 0.48,
  "2016": 0.48,
  "2017": 0.41,
  "2018": 0.37,
  "2019": 0.36,
  "2020": 0.33,
  "2021": 0.24,
  "2022": 0.25,
  "2023": 0.24,
  "2024": 0.21,
  "2025": 0.2,
  "2026": 0.17
 },
 "spread_check": {
  "quote_day": "2019-12-24",
  "iex_median_half_spread_bp": 0.31140036745215033,
  "iex_mean_half_spread_bp": 0.39365476212617706,
  "n_quotes": 111965,
  "model_half_spread_bp_same_day": 4.540436811718526,
  "overstatement_x": 14.580704733484966
 },
 "pass": true,
 "criteria": "|t| of mean gross (mid-to-mid) trip return < 3; no fill better than its reference mid; net = gross - costs with costs > 0",
 "elapsed_sec": 298.6
}
```

## 6. Perfect foresight

Criteria: explicit future as_of -> LookaheadError; clipped queries never return a bar known after now

```json
{
 "window": [
  "2020-01-02",
  "2020-03-31"
 ],
 "modes": {
  "future_as_of": {
   "expect": "raise",
   "raised": true,
   "error": "LookaheadError: as_of 2020-01-12 21:00:00+00:00 is after ctx.now 2020-01-02 21:00:00+00:00",
   "queries": 1,
   "leaks": 0,
   "pass": true
  },
  "future_actions": {
   "expect": "raise",
   "raised": true,
   "error": "LookaheadError: as_of 2020-05-01 21:00:00+00:00 is after ctx.now 2020-01-02 21:00:00+00:00",
   "queries": 1,
   "leaks": 0,
   "pass": true
  },
  "clipped_end": {
   "expect": "no_leak",
   "raised": false,
   "status": "OK",
   "leaks": 0,
   "queries": 62,
   "n_fills": 0,
   "guard_violations": 0,
   "pass": true
  },
  "history_peek": {
   "expect": "no_leak",
   "raised": false,
   "status": "OK",
   "leaks": 0,
   "queries": 62,
   "n_fills": 0,
   "guard_violations": 0,
   "pass": true
  }
 },
 "pass": true,
 "criteria": "explicit future as_of -> LookaheadError; clipped queries never return a bar known after now",
 "elapsed_sec": 3.8
}
```

## 7. Spread model cost check

Criteria: calibrated round-trip cost below Corwin-Schultz on the same trades; no fill better than mid. Per-fill slippage includes impact (k*sigma*sqrt(Q/ADV)) on top of the half spread.

```json
{
 "window": [
  "2018-01-02",
  "2026-10-27"
 ],
 "symbol": "SPY",
 "calibrated": {
  "n_round_trips": 1251,
  "cost_mean_bp": 1.0199311346393984,
  "mean_slippage_bp": 0.2677136322982945,
  "gross_mid_mean_bp": -1.3105173409528454,
  "gross_mid_t": -0.1854789616021646,
  "net_mean_bp": -2.330448475592243,
  "fills_better_than_mid": 0,
  "slippage_bp_by_year": {
   "2018": 0.36,
   "2019": 0.38,
   "2020": 0.34,
   "2021": 0.24,
   "2022": 0.25,
   "2023": 0.23,
   "2024": 0.21,
   "2025": 0.2,
   "2026": 0.18
  }
 },
 "corwin_schultz": {
  "n_round_trips": 1251,
  "cost_mean_bp": 31.1689945592343,
  "mean_slippage_bp": 15.27075055421302,
  "gross_mid_mean_bp": -1.270593532525794,
  "gross_mid_t": -0.17976350385616058,
  "net_mean_bp": -32.43958809176009,
  "fills_better_than_mid": 0,
  "slippage_bp_by_year": {
   "2018": 13.02,
   "2019": 11.33,
   "2020": 25.96,
   "2021": 12.94,
   "2022": 22.48,
   "2023": 12.48,
   "2024": 11.73,
   "2025": 15.85,
   "2026": 10.51
  }
 },
 "quoted": {
  "quote_day": "2019-12-24",
  "iex_median_half_spread_bp": 0.31140036745215033,
  "iex_mean_half_spread_bp": 0.39365476212617706,
  "n_quotes": 111965,
  "model_half_spread_bp_same_day": 4.540436811718526,
  "overstatement_x": 14.580704733484966
 },
 "cs_over_calibrated_x": 30.55990105670639,
 "calibrated_half_spread_slippage_vs_quoted_x": 0.8597087874002951,
 "pass": true,
 "criteria": "calibrated round-trip cost below Corwin-Schultz on the same trades; no fill better than mid. Per-fill slippage includes impact (k*sigma*sqrt(Q/ADV)) on top of the half spread.",
 "elapsed_sec": 178.4
}
```

## Interpretation notes

* SPY: the engine pays dividends as cash (ex-date entitlement, assumed pay date ex+7d, cash earns the broker credit rate); Yahoo Adj Close assumes reinvestment at the ex-date close. The dividends-as-cash runs are therefore compared with an exact apples-to-apples reconstruction (raw close x shares + cash dividends + ledger interest); the reinvesting run (whole shares, next open after the cash arrives) is compared with Adj Close directly.
* S105 reference rebalances at the next open after each month-end session with adjusted prices (dividends reinvested), fractional units, no costs; the engine uses whole shares, cash dividends, modelled spread/impact and IBKR fees.
* PUT: S129 sells the listed near-ATM SPX monthly (AM) put at the synthetic bid through the engine (combo order, Cboe margin, OCC lifecycle, SOQ proxied by the SPX open); the PUT index rolls at a late-morning VWAP and invests in 1m/3m bills. Quote path recorded in `quote_path` (production synthetic fallback, or the runner bridge used before it existed).
* UMD/HML: S&P 500 PIT universe only, equal-weighted deciles/terciles, adjusted closes (pre-2018 survivorship-biased, DATA.md §8); French factors use all NYSE/AMEX/Nasdaq stocks, value weights, size-neutral 2x3 sorts, so correlation must be positive but well below 1.
