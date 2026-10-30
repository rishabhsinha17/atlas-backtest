# atlasbt performance work (PERF agent)

Goal: make the atlas affordable (it was estimated at ~320 CPU-hours) WITHOUT changing a single result.
Every change below is either a memo or an equivalent re-implementation. The golden harness enforces
bit-for-bit equality of all outputs.

## 1. Golden-result harness (`tests/perf/`)

* `tests/perf/golden.py` holds 16 representative real-cache runs (`CASES`): ETF swing with shorts (S046, S068),
  S&P 500 cross-section and pairs (S085, S074, S106), 1-minute intraday on the Yahoo 1m window (S001, S013),
  allocation (S103, S105), options with real chains and synthetic SPX (S126, S134, S129, S138), and the
  runner's short-holding CoinFlip control. The setup is the runner's production setup: `FastPortal`,
  IBKR tiered fees, `use_quotes='auto'`, coverage 0.95, offline.
* Each run is reduced to sha256 digests of every output: equity, returns, fills, trades, positions,
  ledger, the fidelity scorecard (tiers and reasons), status and reason, every `diagnostics` entry (orders,
  rejects, order events, warnings, logs, data-access counters, ...), and `meta` minus wall-clock fields.
  Floats are hashed as raw bytes. Equity, fills and trades are also stored as small parquet files for
  diffing (`tests/perf/golden/<case>/`, about 0.6 MB in total).
* The goldens were recorded with the pre-optimisation code: a snapshot of the tree at f9702e1 in
  `results/_worktrees/perf_base` (engine, strategies and data files identical to that commit's index).
  `ATLASBT_CACHE` pointed to the shared cache.
* `ATLASBT_SLOW=1 python -m pytest tests/perf -m slow` re-runs every case and requires identical digests.
  CLI: `python tests/perf/golden.py check [CASE ...]` (or `record --code-root DIR`).
* `tests/perf/test_perf_equivalence.py` runs in the default suite. It checks each re-implementation
  against the code it replaced: `BarArrays.frame`, `history_arrays` vs `history` for the swing `BarCache`
  and the intraday `Tape`, and the portal memos (quote reads, SP500 membership, borrow, identity
  fast path, adjusted bars + attrs, actions). It also checks that `LookaheadError` still fires: PIT
  `as_of > now`, the `MarketData` guard, and a BarCache strategy that asks for the future.

## 2. What was slow, and the fix

| # | hot spot (profiled with py-spy / cProfile) | fix | where |
|---|---|---|---|
| 1 | `Portal.borrow` rebuilt the PIT S&P 500 membership (~500 security keys via `_member_key` -> `_is_plain` -> `_is_etf`, a string scan of the Nasdaq directory) on EVERY call. The engine calls it for each short at every session end and at short submission: 0.5-4 s per call | `_is_etf` / `_is_plain` memoized per symbol; SP500 `universe()` memoized per session date (it depends on the date only; copies are handed out); borrow skips the membership lookup when `default_fee` ignores it (ETF / core / leveraged); member sets cached per session date | `data/portal.py` |
| 2 | `_identity_by_date` rebuilt `DatetimeIndex` columns of the ticker timeline per call | timeline column arrays memoized per timeline object; single-date fast path | `data/portal.py` |
| 3 | `Portal.bars(adjusted)`: pandas deep-copies `attrs` (including the identity `actions` DataFrame) on every intermediate op; `to_naive_date` (+ freq inference) over the whole history each call; `CorporateAction` objects rebuilt from `to_dict` each call; label lookup for `src_ticker` | selection on an attrs-free twin + ONE deep copy of the identity attrs at the end (same content); naive dates memoized per cached frame; parsed actions memoized per frame (objects are frozen, built lazily in row order); boolean take instead of label lookup when the index is unique | `data/portal.py` |
| 4 | `Portal.quotes` re-read the parquet file for every fill window (`use_quotes='auto'` symbols) | file frames memoized per (path, mtime, size); identical selection logic to `ParquetStore.read` | `data/portal.py` |
| 5 | `risk_free_rate_info` materialised the filtered rate frame each session | last row located on cached columns | `data/portal.py` |
| 6 | `ctx.history` fast path: `BarArrays.frame` rebuilt tz-aware indexes and session labels per call | built once per symbol and sliced | `engine/market.py` |
| 7 | swing `BarCache` incremental update built a DataFrame per symbol per day; full refetch deep-copied attrs 6x | `ctx.history_arrays` (same bars, same fallback); attrs of the local frame dropped before the column reads | `strategies/swing_common.py` |
| 8 | intraday `Tape.update` built a DataFrame per symbol per step | `ctx.history_arrays` (frame path kept for minimal test contexts) | `strategies/intraday_base.py` |
| 9 | lineage of a CIK recomputed per identity lookup | memoized per (cik, symbol, hint) | `data/portal.py` |

Every memo lives in `Portal._cache`, so `clear_cache()` drops it. The memos assume what the existing `_read`
cache already assumes: the store does not change under a running portal. The quote memo also re-reads a file
whose mtime or size changes.

PIT semantics are untouched. Every strategy-visible query still passes through `PITPortal` (as_of pinned,
returned rows re-checked). The engine cache keeps its `MarketData.check` guard. `history_arrays`
serves the same completed bars as `history`. The data-access diagnostics counters are unchanged (they are
part of the golden digest).

## 3. Results

CPU seconds per run (`time.process_time`). The machine was shared with the production batches (load 6-9 on
4 cores), so wall clock is noisy: base runs were recorded concurrently and are pessimistic in wall time.
CPU time is the fair measure. All cases are IDENTICAL (every digest equal).

| case | window | base (f9702e1) | now | speedup | fills |
|---|---|---|---|---|---|
| COINFLIP_spy_2015 (short SPY control) | 2015 | 203.3 cpu-s | 1.0 | ~200x | 69 |
| COINFLIP_spy_2015h1 | 2015 H1 | 102.0 | 3.3 | 31x | 35 |
| S046_etf_2015_2018 (19 ETFs, long/short) | 4 y | 4657.7 | 35.9 | 130x | 1194 |
| S046_etf_short_2015h1 | 2015 H1 | 530.5 | 9.6 | 55x | 148 |
| S068_etf_2015_2018 (long only) | 4 y | 49.3 | 20.6 | 2.4x | 547 |
| S085_sp500_2019q1 (12-2 momentum) | 2019 Q1 | 2320.9 | 51.9 | 45x | 228 |
| S074_sp500_2019q1 (weekly reversal L/S) | 2019 Q1 | 3804.4 | 93.9 | 41x | 1895 |
| S106_sp500_2019 (pairs) | 2019 | 135.4 wall | 60.8 wall / 49.9 | 2.2x | 0 |
| S013_1m_yahoo_2026_10 (20 symbols, 1m) | 19 sessions | 112.9 wall | 23.3 wall / 17.6 | 4.8x | 72 |
| S001_1m_yahoo_2026_10 | 19 sessions | 27.5 wall | 20.5 wall / 15.2 | 1.3x | 30 |
| S103_spy_2010_2018 | 9 y | 51.8 | 22.9 | 2.3x | 97 |
| S105_2015_2018 | 4 y | 9.3 | 6.2 | 1.5x | 94 |
| S126_spy_opt_2020h1 | 2020 H1 | 4.4 | 2.4 | 1.8x | 0 |
| S134_opt_2020 (real SPY chains) | 2020 | 108.3 | 105.5 | 1.0x | 44 |
| S129_spx_2020h1 (synthetic SPX) | 2020 H1 | 19.1 wall | 15.9 wall / 11.9 | 1.2x | 11 |
| S138_spx_2020h1 | 2020 H1 | 185.2 wall | 186.0 wall / 113.0 | 1.0x | 0 |

Every short-holding or S&P 500 strategy was dominated by hot spot #1. At f9702e1, one simulated year of
S085/S074 cost about 2.5-4.2 CPU-hours (the user-reported ">30 min/yr" predates the identity layer). It is
now about 3.5-6.5 CPU-min per year, including ~40 s of one-time data loading. For long-only daily strategies
the remaining cost is startup (actions classification, imports: 10-20 s per process) plus strategy-side
pandas indicators. Options are dominated by `options/marking.py` (section 4).

## 4. Not done / recommendations (outside this agent's edit scope)

* **OPTIONS (`options/marking.py`)**: `mark_contract` re-runs `prepare_chain` + `infer_forwards`
  (de-Americanisation, Bjerksund-Stensland IV inversion of the whole chain) for EVERY contract. The
  strategy's `leg_mark` and the engine's mark path through `Portal.option_quotes` ->
  `synthetic_option_quotes` both do it, about 2 x legs times per session on the same snapshot. That is
  50-60 % of an options run (S134). Memoizing the chain-level part per (snapshot, now, r, root, settlement,
  exercise) would cut options runs by roughly 2x. `Instrument.option` construction in `synthetic.chain` costs
  another ~15 %.
* **DATA (`actions_classify.classify`)**: about 1 s per symbol at the first `has()` / `bars()` of an ETF.
  This is pandas row-by-row work and is paid again in every forked grid task (~10-20 s per 19-ETF task).
  Vectorize it or persist the classified actions.
* **DATA (`Portal.security_master`)**: about 0.3 s per call. S085 / S074 / S106 call it each session or each
  refresh. It is as_of-dependent, so it can only be memoized on exact as_of. A per-date static part (rows
  from SEC/current constituents) could be cached.
* Strategy findings (not perf): S106 forms no pairs in 2019 (0 orders all year; sector groups look
  empty). S126 and S138 place no orders in 2020-2021 windows. Their goldens still pin equity, financing and
  the hot paths.

## 5. Separable borrow fix

`results/_worktrees/perf_borrow_fix_A.patch` has 4 hunks in `atlasbt/data/portal.py`: the `_is_etf` memo, the
SP500 `universe` per-date memo (2 hunks) and the `borrow` ETF shortcut with per-date member sets. It applies to
HEAD on its own. It was verified against the base on its own: `borrow()` on a 396-query grid of symbols,
security keys and dates (identical; 826 s -> 46 s), and CoinFlip SPY 2015 H1 plus S046 2015 H1 digests
(identical).

## Golden provenance (integrator)
* 13 cases were re-recorded on the UNOPTIMIZED committed code at dd5e315 (after the fidelity-tier change) and the
  optimized tree was checked IDENTICAL against them (independent integrator run).
* The 3 heavy cases (S046_etf_2015_2018, S074_sp500_2019q1, S085_sp500_2019q1) were proven IDENTICAL against the
  unoptimized f9702e1 baseline before the tier change; because the tier change (execution/fills.py) and the
  optimizations touch disjoint code, their goldens were refreshed with the optimized code at dd5e315+perf.
