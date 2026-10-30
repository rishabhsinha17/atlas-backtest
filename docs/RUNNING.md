# Running the atlas (`python -m atlasbt.run`)

Owner: RUNNER agent. Package `atlasbt/run/`, tests `tests/run/`, outputs `results/` (git-ignored except
`results/README.md`). The runner orchestrates the other packages and adds no trading logic. It plans
windows, runs the engine, applies the validation and reporting functions, and writes the evidence.

```
python -m atlasbt.run plan                       # results/plan.csv for all 172 strategies
python -m atlasbt.run run [--ids S046-S084,S103]  # default: every plan-testable, non-held strategy
python -m atlasbt.run atlas                      # results/ATLAS_REPORT.md, leaderboard.csv
python -m atlasbt.run audit [--live]             # results/AUDIT.md (activity safeguard)
python -m atlasbt.run replicate [--quick]        # results/REPLICATION.md (known-answer checks)
```

Common options: `--results-dir DIR`, `--workers 4`, `--max-configs N`, `--budget MIN` (intraday grid
budget per strategy, default 30), `--daily-budget MIN` (same for daily+ strategies, default unlimited),
`--force` (wipe and rerun a strategy), `--timeout MIN` (per task, default 360), `--mem-reserve GB`
(default 3), `--start/--end` (window override, smoke tests only; recorded in config.json),
`--use-quotes never|auto|always` (default `auto`, the engine default since its per-window quote-coverage fix; see §8), `--universe-coverage filter|strict` and
`--min-universe-coverage 0.95` (§2), `--label TEXT` (a run label shown in summaries and the leaderboard,
e.g. `shakeout-2y`), `--extend-grid` (grows existing grid subsets to the current caps; the seeded subsets
are prefix-stable, so it only adds configs and re-finalizes), `--allow-stock-universes`/env
`ATLASBT_STOCK_HOLD=1` (data holds, §8), `--no-charts`, `--bootstrap-reps`.

**Long runs never execute from the live working tree.** Many agents edit it concurrently: on 2026-10-29 a
calibration JSON rewritten mid-run crashed every backtest. Use:

```
python -m atlasbt.run launch [--commit HEAD] run --ids ... --daily-budget 30 --label calibrated-v1
python -m atlasbt.run launch replicate
```

`launch --no-overlay` uses the committed runner exactly. This is the rule for final runs: pin an
explicit hash (`--commit 4ea9ad1`) so that no uncommitted edits by other agents can leak in. `python -m
atlasbt.run launch --commit <hash> --no-overlay final ...` runs the whole atlas in one pinned process:
plans (primary >= 95%, extended >= 60%); primary runs with per-group grid budgets (`--etf-budget` -1 =
full, `--options-budget 45`, `--stock-budget 0`); all replications; audit; atlas; the
EXTENDED_SURVIVORSHIP_BIASED stock runs; then atlas again.

`launch` (`worktree.py`) creates `results/_worktrees/<commit12>` with `git worktree add --detach`, with no
commits or branches. It overlays the runner package `atlasbt/run` when the commit lacks it, and records
the overlay's sha. It REFUSES to start if anything else in the worktree is dirty. It then starts the
subcommand detached, with `ATLASBT_CACHE` = the main repo's `cache/` (shared, read-only for runs),
`--results-dir` = the main repo's `results/`, and the commit / worktree / overlay hashes exported, so
every `config.json` and report records them. `run` from the live tree is refused unless
`--allow-live-tree` is given (smoke tests). The scheduler imports every package before forking, so one
run uses one code version, and the code hash at launch is recorded.
Several runs can share one results dir when their id sets are disjoint. The 2026-10-29 atlas used three:
* A: ETF, allocation and intraday strategies, full windows, `--daily-budget 30`.
* S: stock universes, `--start 2024-10-01 --end 2026-10-27 --daily-budget 0 --label shakeout-2y`.
* O: options, `--start 2020-01-02 --label shakeout-realchains --daily-budget 0`.
Progress goes to `results/logs/run.log` (one line per task start/end). Each strategy's engine output
and tracebacks go to `results/logs/<SID>.log`.

## 1. Active notices (`atlasbt/run/notices.py`)

Every report carries the integrator notices that are currently in force: provisional costs (the spread
model is being recalibrated), `use_quotes='never'`, the single-stock data hold, and the T3 status of
synthetic-option fills. Remove a notice only after its fix lands and the affected strategies have been rerun with `--force`.

## 2. `plan`: requirements, windows, testability

For each sheet id the runner instantiates the registered class with its sheet params. Some families
derive `universe` and `requires` from params; options strategies, for example, take them from
`underlying`. Each critical `Requirement` is then resolved the way the engine resolves it at run start:
`portal.has` must be true, and `portal.fidelity_of` must be neither NA nor worse than `min_fidelity`.
The plan also checks facts that `has` cannot see. Intraday option quotes do not exist. Quote
requirements are covered on 1 or 2 sample sessions only. `inav` has no source. A DAY_* sizing profile
on a daily bar frequency is untestable, which is the engine's own rule.

Window start is the first session at which every critical input exists:

| input | rule |
|---|---|
| bars of explicitly named symbols | latest first bar |
| bars of a static universe | all members if <= 6 names; else the date by which 80% are listed (strategies trade late listers once they have bars) |
| PIT universe `SP500` | first month start at which PIT price coverage (`universe(require_bars=True)` / `universe()`, the engine's own windowed-`has()` definition) >= `--min-universe-coverage` (default 0.95 → 2018-04). The engine NOT_TESTABLEs a run when ANY member lacks data at the first session, and under the fail-closed identity layer some member always does (2008: 94 of 497 lack data, 2018: 8, 2024: 1). Since eea82fa the engine filters natively (`BacktestConfig.min_universe_coverage`; coverage in `meta['universe_coverage']`, flagged T2), and `--universe-coverage engine` (default) passes the threshold through. The runner's earlier `CoverageFilteredPortal` is kept as `--universe-coverage filter`. It drops members without daily data (has() true AND bars returned), logs every drop, and records the coverage per result. `strict` reproduces the engine rule (coverage 1.0, never met). Members without INTRADAY data are not dropped: those stay NOT_TESTABLE. |
| events / fundamentals | first event in the store / first XBRL filing (2009-07-22) |
| real option chains (`min_fidelity <= T2`) | first DoltHub snapshot of the underlying |

When the binding constraint is a listing date, daily+ strategies get a 252-session warm-up, because no
pre-listing history exists for indicators. Intraday strategies get none: they read prior daily and
intraday history through the PIT portal. Window end is the last cached session common to the critical
bars. Minimum windows are 252 sessions (daily+) and 10 sessions (intraday). `plan.csv` columns: id,
family, module, bar_freq, sizing, testable, reason, start, end, n_sessions, expected_fidelity (worst
critical tier), fidelity_notes, n_grid, est_runtime_min_baseline/grid (cost model in `plan.py`,
calibrated on S050/S105 runs), universe, n_symbols, binding_constraint, hold, activity, stale. Facts
(bar ranges, SP500 coverage) are memoized in `results/plan_cache.json`. Delete that file after a data refresh.

## 3. `run`: per-strategy pipeline (`pipeline.py`, `scheduler.py`)

1. **Baseline**: sheet params over the full planned window, `BacktestConfig(initial_cash=1e6,
   fees=IBKR_PRO_TIERED, use_quotes='never')`, otherwise engine defaults. The result is pickled with
   pruned diagnostics, and NAV/returns go to CSV.
2. **Grid**: `registry.expand_grid(cls.grid)`. Each config is a full-window run, saved as a daily
   return column. The config equal to the sheet params reuses the baseline. Caps: `--max-configs`, and
   for wall-clock budgets `floor(budget / baseline_runtime)` configs. When capped, the subset is a seeded
   permutation (`sha256(SID|grid_subset)`) that always keeps the sheet config, and the cap is recorded
   in `grid_plan.json` and in the report. Default budgets: intraday 30 min. Daily is unlimited unless
   `--daily-budget` is given; the long atlas run uses `--daily-budget 45`.
3. **Walk-forward** (`validation.walk_forward_matrix` on the grid return matrix): rolling train 3y /
   test 1y (step 1y) for daily+ strategies, and 6m / 2m for intraday ones. The objective is Sharpe, ties
   go to the earliest column, and OOS segments are stitched. The **untouched holdout** is the last 20% of
   the window's sessions. It is removed before any fold is built and evaluated once, with the params
   selected on the train window just before it. Baseline params on the same holdout are shown for
   reference. Offsets are passed as `pd.DateOffset`, because pandas 3 rejects the `'3Y'` alias. When the
   window is shorter than train + test (all Yahoo-1m intraday strategies), there are no folds, the
   holdout params are chosen on all pre-holdout data, and the report says so. Matrix-mode caveat
   (ANALYTICS.md §3): state from the full-sample run carries into test windows.
4. **Cost stress A-E and break-even** (`validation.cost_stress` via `reporting.write_strategy_report`)
   on the baseline result.
5. **Stationary-bootstrap CIs** (Politis-White block) for baseline Sharpe/CAGR/MaxDD, plus an OOS
   Sharpe CI for the walk-forward series.
6. **PBO** (`validation.pbo_cscv`) on the pre-holdout grid matrix, S = 16 blocks (fewer if T is short),
   N >= 2 configs.
7. **Report** `results/reports/<SID>/report.{md,json}` + PNGs (`reporting.write_strategy_report`). It
   includes DSR deflated with the atlas trial total, V[SR] from the strategy's grid Sharpes, SPY
   benchmark regression, FF3 factor regression, regimes, and the fidelity scorecard. The runner appends
   a section with walk-forward, holdout, PBO, the activity label and per-config results.

**Scheduling**: every task (baseline, grid config, finalize) runs in its own forked process. At most
`--workers` run at once, and no new task starts while MemAvailable is below `--mem-reserve` GB.
Priorities are finalize, then configs, then baselines (longest estimate first). An exception, OOM kill
or timeout of one task is recorded as FAILED/TIMEOUT with the traceback and never stops the atlas.
**Resumable**: outputs are written atomically, finished baselines, configs and finalizations are
skipped on restart, and `--force` wipes a strategy first. **Provenance** (`runs/<SID>/config.json`):
run options, plan row, engine config, deviations, git commit (read from `.git` files), a sha256 of
every `atlasbt/**/*.py`, package versions, and store metadata (source, fetched_at, rows) for every
symbol/frequency used. **Seeds**: `sha256(SID|task)` seeds `random` and `numpy` per task, and
bootstrap and PBO use seed 0.

## 4. Activity safeguard (`activity.py`)

A run with status OK that never trades must not show up as "0% return, OK". Baseline and grid runs are labelled:

* `ZERO_ACTIVITY`: no fills, or no entry.
* `LOW_ACTIVITY`: below the family floor. Intraday: >= 1 entry per month. Rebalance-based (PORTFOLIO,
  CROSS_SECTION): >= 2 trading sessions and >= 1 per 2 years. Every other daily+ strategy: >= 5
  entries per 10 years.

Entries are trade entries, or distinct fill sessions for rebalance-based strategies. ZERO_ACTIVITY
baselines get status `ZERO_ACTIVITY` in the summary and leaderboard rows. They are excluded from the
ranked leaderboard and listed in their own ATLAS_REPORT table, together with rejected-order reasons,
engine warnings, fidelity reasons and the log path. ZERO_ACTIVITY grid cells are excluded from WF and
PBO. A strategy can declare why low activity is expected with a class attribute `activity_note`.
`audit` writes `results/AUDIT.md` and `audit.json`. `tests/run/test_audit.py` fails for every
registered strategy whose baseline is OK + ZERO_ACTIVITY without a declared note; strategies with no
baseline yet are skipped, and the skip names the command. Diagnosed causes are printed as `KNOWN CAUSE`
in the offender's row: combos submitted but never filled, and option selection failures.

**Diagnostics interface (strategies → runner).** `activity.diagnose` reads the UNPRUNED result:
* order outcomes from `diagnostics['orders']`: counts by type/status, combos submitted, filled,
  cancelled and rejected, and the top cancel/reject reasons;
* strategy logs `ctx.log(msg, activity="<category>")` counted by category. The categories are
  `selection_failure`, `combo_cancelled`, `order_rejected`, `no_signal`, `data_missing` and `skip`; logs
  without the keyword are categorised by text;
* aggregated fidelity notes `ctx.flag_fidelity(T1, "activity <SID>: <generic reason>")`, which the
  options_income family already emits.

## 5. `atlas`

* Leaderboard via `reporting.build_leaderboard`. DSR uses N = max(registry n_trials() = 1443, parsed sheet
  count). V[SR] is the variance of every grid configuration's per-period Sharpe across the atlas, floored
  by each strategy's own estimation noise. p = 1 - PSR(0); BH q at 0.10. Sort order: DSR, then PBO, then
  q, then PSR. Extra columns: holdout Sharpe and params, grid run/full, window, activity.
* **Hansen SPA / White RC** on daily differentials of baseline runs: vs cash (excess over the engine's
  rf) and vs SPY total return, plus WF OOS + holdout series vs SPY and vs zero. The sample is the common
  window of the daily+ OK strategies with >= 5 years of overlap; the others are listed as excluded.
  Intraday strategies (20 sessions) are never in SPA.
* Tables: ZERO_ACTIVITY, FAILED (last traceback line + log path), TIMEOUT, STALE, HOLD, NOT_TESTABLE
  (plan or engine reason), NOT_RUN. Also the fidelity grade distribution by status.

## 5b. EXTENDED_SURVIVORSHIP_BIASED runs (stock strategies)

Primary results for S&P 500 stock strategies use the >= 95% coverage window, which starts 2018-04. Each
stock strategy also gets a secondary extended-history run in a SEPARATE results namespace:

```
python -m atlasbt.run --results-dir results/extended plan --ids <stock ids> --min-universe-coverage 0.60
python -m atlasbt.run --results-dir results/extended run --ids <stock ids> --min-universe-coverage 0.60 \
       --label EXTENDED_SURVIVORSHIP_BIASED --daily-budget 0
```

Windows start at the first month with >= 60% PIT price coverage (2009-04 for most). The coverage
series is recorded and plotted in `results/extended/reports/<SID>/universe_coverage.png`, and every
report carries an EXTENDED_SURVIVORSHIP_BIASED banner. The primary `atlas` lists these runs in their
own table ("NOT part of the leaderboard"; also `extended_survivorship_biased.csv`) with the note that
delisted and failed names are under-represented, so results are likely optimistic. They are never
ranked with primary results.

## 6. `replicate`: end-to-end known answers (`replicate.py`, `tests/run/test_replicate.py`)

| check | reference | pass criterion |
|---|---|---|
| SPY buy & hold (IBKR fees, zero commission, dividend reinvesting) | Yahoo Adj Close TR; exact reconstruction raw close x shares + cash dividends + ledger interest | residual < 1 bp NAV; reinvesting TD < 20 bp/yr |
| S105 60/40 | pandas: adjusted opens/closes, next-open rebalance after month end | corr > 0.99, \|TD\| < 50 bp/yr |
| S129 vs Cboe PUT | monthly PUT index returns | corr > 0.9, TE < 5%/yr |
| 12-2 momentum decile spread | Ken French UMD | corr > 0.3 (S&P 500 only, equal weight) |
| B/M tercile spread | Ken French HML | corr > 0.3 |
| coin-flip entries on SPY | zero gross alpha | \|t\| < 3; no fill better than mid; net = gross - costs |
| perfect foresight | PIT guard | LookaheadError on future as_of; clipped queries never leak |

The engine pays dividends as cash: an ex-date entitlement, with the pay date assumed at ex+7d. Yahoo Adj
Close reinvests at the ex-date close. So the dividends-as-cash runs are compared with an exact
apples-to-apples reconstruction, and the reinvesting run is compared with Adj Close directly. For S129
the runner's `SyntheticQuotePortal` serves `Portal.synthetic_chain` as session-close T3 snapshots; it was
needed before the engine could fill synthetic contracts (§8). Since 2026-10-29 the production `Portal.option_quotes` has its own synthetic fallback, so
`rep_put` uses the plain portal by default; `bridge=True` reproduces the earlier path. UMD/HML use single-stock data and are PROVISIONAL while the stock-data hold is active.
`--quick` uses short windows (pytest). PUT and UMD tests are marked `slow`.

## 7. Performance

`FastPortal` (`fastportal.py`) memoizes `Portal.corporate_actions` per security identity. The parent
rebuilt objects on every call (~100 ms per call for 10 symbols; the engine calls it every session),
which made a 21-year 19-ETF backtest take ~40 min. With the memo it takes ~11 min, and the results are
identical (tested). The memo belongs in the data layer (request filed). Remaining cost is mostly
strategy-side pandas work per bar, e.g. swing_common `BarCache.get` + SMA recomputation (about 6 ms per
symbol-day). Measured costs: S105 (2 ETFs, monthly) ~12 s/yr; S050 (19 ETFs) ~32 s/yr; S129 with
synthetic chains ~2 min/yr. Memory: ~0.4-1.3 GB per task.

`FastPortal` also memoizes `universe('SP500', as_of)` per session date, a pure function of the session
date. At f9702e1, `Portal.borrow()` rebuilt the ~500 security keys on every call (~4 s), and the engine
calls `borrow()` for every short position at every session end. A 1-symbol coin flip took 503 s per
simulated year, against 5.5 s with the memo. Bit-for-bit verified: equity and fills are identical
(`test_fastportal_universe_memo_identical_to_portal`, commit 4ea9ad1).

## 8. Known cross-layer issues the runner works around or reports

* ENGINE `use_quotes='auto'` (fixed): it used to make SPY/QQQ/AAPL quote-only because single sample
  days of quotes exist. Now quotes are used only where they cover the bar window. Spot check (SPY coin
  flip and buy & hold, Dec 2019 - Jan 2020): NAV is identical to `'never'` until the only quote-covered
  day (2019-12-24). The default is back to `auto`.
* ENGINE spread estimator (fixed in e3d3d4d): Corwin-Schultz on daily bars put SPY's full spread at
  18-50 bp, against ~0.6 bp quoted. The engine default is now the calibrated model: an SPY coin-flip round
  trip costs 1.0 bp (31 bp under CS), and calibrated per-fill slippage is 0.27 bp vs 0.31 bp quoted
  (REPLICATION.md §7). `--spread-model corwin_schultz` reproduces the legacy model.
* OPTIONS/DATA/ENGINE (fixed 2026-10-29): options fill only from `portal.option_quotes`, which used to
  serve real quotes only, so synthetic-chain strategies never filled. It now falls back to synthetic
  session-close snapshots (T3). Options runs made before the fix are invalid; rerun them with `--force`.
* DATA: single-stock identity. The first fix (568edb7, fail-closed PIT identity) briefly lifted the hold.
  A second hold (07:10) followed, because the engine path resolved identity at as_of = range_end, so BBT
  2019 filled at Berkshire Hills prices. It was LIFTED for the final atlas after the date-based identity
  redesign (f39aa6b: `TICKER~CIK` security keys; 50c0d9a: a single key resolver) and engine coverage by
  windowed `has()` (eea82fa). Re-arm with `ATLASBT_STOCK_HOLD=1`; held outputs are marked STALE.
  Reports show keys as `BBT (CIK0000092230)`. Every stock result records its universe coverage:
  DATA's `universe_survivorship_bias` at window start, middle and end, plus the runner filter log.
* DATA: `has(sym,'bars','1d')` can be True while `bars()` returns nothing (ABC, 2026-10-29). The coverage
  filter checks for rows.
* DATA/OPTIONS: synthetic SPX AM OCC symbols carry the last trading day, but `Portal.instrument` decoded
  them as settlement dates, so every S129 combo was rejected ("disagrees with the instrument master").
  The fix is dispatched to DATA.
* ANALYTICS: `walk_forward_splits(train='3Y')` fails under pandas 3 (`'Y'` alias removed); pass
  `pd.DateOffset`.
