# Handover: full state of the project

This is a factual record of what exists, where it is and what state it is in, as of 2026-11-03.

## 1. The request

The owner's original request: build the highest-fidelity backtest engine possible, on free data, for the 172
strategies in `docs/Trading_Strategy_Atlas_172_Systems.xlsx`.

* Find every usable data source.
* Make results point-in-time and true to real-life execution.
* Have separate agents build each part, red-team it and verify it.

Later instructions from the owner:

* Full network access granted.
* "Push anything to that repo" — standing permission to push to `rishabhsinha17/atlas-backtest`, which is
  public, MIT-licensed and was created by the owner.
* For survivorship (closing the gap left by delisted stocks), "don't care about anything before 2011".
* "Close all closeable gaps in parallel fast". Flag anything free that needs the owner's help.
* The owner supplied keys for Alpaca (paper account), Tiingo (free) and Databento (free credits), and approved
  using their name and email as the SEC contact string.
* The owner asked for plain-English, concise updates.

## 2. Constraints in force

* **The repository is public.** It contains no keys and no personal data. Commits use the author
  `Rishabh Sinha <rishabhsinha17@users.noreply.github.com>`.
* **Keys are read only from environment variables**: `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `TIINGO_API_KEY`,
  `DATABENTO_API_KEY` and `SEC_USER_AGENT`.
  * On the old cloud machine they were stored in `~/.atlasbt/secrets.env` (chmod 600).
  * The keys were pasted into chat once, and the owner was told to regenerate them.
* **Network rules** (`docs/ARCHITECTURE.md` §12):
  * User agent `atlasbt-research/0.1`; no browser spoofing; no bot-check bypassing; robots.txt respected.
  * Rate limits: Alpaca ≤ 180 requests/min; Tiingo ≤ 40 symbols/hour, 400/day and 400/month (free tier is
    50/h, 1,000 requests/day and 500 symbols/month), logged in `cache/logs/tiingo_requests.jsonl`; SEC ≤ 5
    requests/s.
* **Databento**:
  * Hard cap of $90 total.
  * Every request is priced with `metadata.get_cost` and logged in `$ATLASBT_CACHE/raw/databento_ledger.csv`.
  * Spent: **$18.42** over 3,605 requests (ledger at the time of the restart).
* **Licences**: Alpaca, Tiingo and Databento data are not committed to the repository.
  * `atlasbt/execution/calibration/alpaca_nbbo_2016_2026.csv.gz` (on the WIP branch, 455 KB, 21,155 rows) holds
    derived per-symbol-window half-spread statistics, not raw quotes.
* **Process used for every subsystem**: a builder agent, then an independent red-team agent (whose failing tests
  stay in `tests/redteam/`), then an independent verifier, then fixes.
* **No concurrent atlas runs.** Two concurrent `final` runs previously caused out-of-memory kills (S038, S061).

## 3. Locations

| Item | Location |
|---|---|
| Verified code | `main`, HEAD `3264fd5`. Code commit `5bd53a0`; handover doc commits after it. |
| Unfinished, untested work | branch `wip/gap-closing`: snapshots of the working tree on top of `main`, force-pushed |
| Old original repo (superseded) | old cloud machine `/home/user/atlas_bt` (HEAD `c56dde6`) |
| Data cache, 9.6 GB | old cloud machine `/home/user/atlas_bt/cache` (`ATLASBT_CACHE`); not backed up |
| Raw scout downloads, 1.2 GB (gitignored except 10 files) | old cloud machine `/home/user/atlas_bt/data_raw` |
| Earlier partial atlas results (11 strategies, 134 MB, superseded) | old cloud machine `/home/user/atlas_bt/results` |
| Agent scratch files and logs | old cloud machine `/tmp/claude-0/-home-user-atlas-backtest/2ba86eec-459b-5ac7-a2c7-f25f9304994f/scratchpad/` (`alp/`, `surv/`, `db/`) |

Cache store layout (`$ATLASBT_CACHE/store/<kind>/<source>/<symbol>[/<year>].parquet`). Kinds:

* `actions`, `bars_1d`, `bars_1m`, `bars_5m`, `bars_60m`;
* `events`, `factors`, `fundamentals`;
* `identity` (series, timeline, actions);
* `option_chain`, `option_quotes`, `quotes`;
* `rates`, `reference`, `trades`.

Raw HTTP responses are under `$ATLASBT_CACHE/raw/<host>/`.

## 4. The engine (on `main`)

### 4.1 Package layout (`atlasbt/`)

| Package | Contents |
|---|---|
| `core/` | `types.py` (contracts), `protocols.py` (DataPortal protocol), `clock.py` (XNYS calendar, ET/UTC), `schemas.py` |
| `engine/` | `backtester.py`, `config.py`, `context.py`, `market.py`, `orders.py`, `pit.py` (the `LookaheadError` guard), `results.py` |
| `execution/` | `fills.py`, `spread.py`, `spread_calibrated.py` plus `calibration/`, `impact.py`, `fees.py`, `rates.py`, `constraints.py` |
| `portfolio/` | `account.py`, `ledger.py`, `margin.py`, `financing.py`, `corporate_actions.py` |
| `risk/` | `sizing.py` |
| `data/` | `portal.py`, `identity.py`, `universes.py`, `fetch.py`, `store.py`, `adjust.py`, `survivorship.py`, `qa.py`, `ca_crosscheck.py`, `providers/*` |
| `options/` | `chain.py`, `synthetic.py`, `lifecycle.py`, `margin.py`, `marking.py`, `listing.py`, `parity.py`, `structures.py`, `calibrate.py` |
| `pricing/` | BSM, Black-76, CRR, Bjerksund-Stensland 2002, IV solver, SVI |
| `indicators/`, `validation/`, `reporting/` | indicators, statistics, reports |
| `strategies/` | `registry.py` plus family modules. All 172 strategies are implemented. |
| `run/` | `plan.py`, `pipeline.py`, `atlas.py`, `audit.py`, `retier.py`, `variants.py`, `fastportal.py`, `worktree.py`, `scheduler.py` |

### 4.2 Design facts

* **Bar timing**: bars are labelled by their start and become known at their end. Fills happen only at the next
  eligible bar or quote, with the adverse intrabar path taken first.
* **Fidelity tiers**: T0 (quotes), T1 (real bars plus a modelled spread), T2 (proxy), T3 (modelled data), and
  NA, which means `NOT_TESTABLE`. Tiers come from a reason-to-tier map (`retier_fill`).
* **Identity**: securities are keyed `TICKER~CIK##########`. Identity is resolved per bar date and fails closed.
* **Universe coverage**: the threshold is `min_universe_coverage=0.95`, evaluated with a windowed `has()`.
* **Spread model**: calibrated to NYSE Daily TAQ NBBO (2026-07-01), with era tick regimes, a pessimistic floor
  for 2010–2025, venue tick schedules and a liquid-ETF intraday profile. The default is
  `spread_model="calibrated"`.
* **Costs**: square-root impact with k=0.1. IBKR fees. Dated SEC Section 31, FINRA TAF and OCC fees.
* **Shorting and margin**: Reg SHO Rule 201, locates and borrow, Reg-T margin, and Cboe strategy margin for
  options.
* **Corporate actions**: dividends, splits, value-based spin-offs and delistings.
* **Options lifecycle**: OCC rules; early assignment only from same-session marks.
* **Option quote fallback**: a real quote first, then a same-day surface mark (T2), then a synthetic chain (T3).
* **Validation**:
  * Walk-forward windows: 3 years in-sample / 1 year out-of-sample, or 6 months / 2 months for intraday
    strategies, plus a 20% untouched holdout.
  * Cost-stress scenarios A–E, a break-even cost solver and a stationary bootstrap.
  * Multiple-testing statistics: deflated Sharpe with N = 1450, PBO/CSCV, Hansen SPA and BH-FDR.
* **Runner**:
  * Pinned clean worktrees (`launch --commit X --no-overlay`), resumable, offline (`ATLASBT_OFFLINE=1`).
  * Audits flag ZERO_ACTIVITY, LOW_ACTIVITY and LEADING_GAP.
  * Proxy variants (`S031@proxy`, `S034–S037@proxy`, `S015@mid_proxy`) are reported in their own T2 table and
    excluded from the primary trial count and from SPA.

### 4.3 Red-team record

From `STATUS.md`:

| Review | Findings → outcome |
|---|---|
| Analytics | 15 → 14 fixed, 1 invalid |
| Engine: money | 23 → 22 fixed, 1 invalid |
| Engine: time and execution | 13 → 13 fixed |
| Options | 10 → 10 fixed |
| Data | 15 → fixed |
| Identity re-audit | led to the redesign to per-bar-date `TICKER~CIK` keys |
| Spreads | 9 → fixed |
| Conformance vs the sheet, per family | 5 / 9 / 10 / 11 / 11 / 10 → fixed |
| Runner | 12 → fixed |

### 4.4 Replications

* SPY buy-and-hold vs adjusted close: −0.02%/yr.
* 60/40 vs pandas: correlation 1.000.
* Cboe PUT index: correlation 0.93; 6.0% vs 7.45%/yr; tracking error 4.1%.
* 12-2 momentum vs Fama-French UMD: correlation 0.89.
* Book/market vs HML: correlation 0.82.
* Coin flip: gross −1.0 bp (t −0.32).
* Perfect foresight: blocked.

### 4.5 Tests and performance

* About 1,900 offline tests. Golden digests are in `tests/perf/` (`golden.py`, `test_golden.py`). Goldens run
  only with `ATLASBT_SLOW=1`.
* Performance commit `8b99f13`: 41–200× faster, bit-identical.

### 4.6 Commits since the engine was finished

* `5bd53a0`:
  * Pattern-day-trader rule (FINRA 4210, `pdt_rule=True` by default; binds only when prior-close equity is
    under $25k; engine flatten and margin-call orders are exempt).
  * Opt-in `IBKR_PRO_TIERED_HISTORICAL` / `IBKR_PRO_FIXED_HISTORICAL` commissions. The change date is set to
    2019-09-26 and flagged approximate.
  * Opt-in `FinancingConfig(schedule="ibkr_pro")`: Fed funds benchmark, tiered spreads, cash credit at the
    benchmark − 0.5% above $10k.
  * `settlement_days()` lookup (T+3/T+2/T+1; not used by the engine).
  * Documented MOC cutoff history (not applied).
  * `atlasbt/data/ca_crosscheck.py`: across 657 symbols, splits disagree in 43 of 171 comparable events and
    dividends in 1,439 of 29,300.
  * `cboe_index_history` falls back to the store's `bars_1d/cboe`, because pinned worktrees lack the gitignored
    `VXN`/`RVX`/`VXAPL` CSVs.
  * Verified: 16 of 16 golden cases bit-identical to the previous commit on the same cache.
* `3264fd5`: this handover document.

## 5. Testability (`docs/plan.csv` on `main`; not yet regenerated)

* 126 of 172 testable: 5 at T0, 81 at T1, 38 at T2, 2 at T3.
* 46 not testable:
  * S001–S014, S016–S022, S027, S030, S039, S040: no multi-year 1-minute/5-minute stock bars.
  * S031–S037: 15:45/15:50 decision times.
  * S015: quotes.
  * S158–S160, S164, S171: intraday option quotes.
  * S161–S163, S172: SPX minute data.
  * S041, S045, S110: no point-in-time source.
  * S108: Vanguard ETFs, since fetched in commit `21cd6f3`.

## 6. Work on `wip/gap-closing` (2026-10-30)

Five agents ran in parallel, each with assigned files. The cloud container restarted on 2026-10-30 at 08:54 UTC.
The Alpaca, survivorship and Databento agents and all their background jobs were stopped at that point. None of
the work below is on `main`.

### 6.1 Events and borrow (agent finished)

* **New files**:
  * `providers/buybacks.py`, `providers/dividends_decl.py`, `providers/estimates.py`, `providers/iborrowdesk.py`.
  * Tests: `tests/data/test_{buyback_provider,dividend_decl,estimates,iborrow_provider}.py` and
    `tests/strategies/calendar_allocation/test_s042_dividend_decl.py`.
  * Fixtures: `tests/data/fixtures/{aapl_acceptance_2019_2020.csv,efts_aapl_dividend_2019_2020.json}`.
* **Edited**:
  * `providers/sec.py`: full-text search, acceptance times, document fetch gated on `SEC_USER_AGENT`.
  * `providers/borrow.py`.
  * `strategies/calendar_events.py`.
  * `portal.py`: event kinds `DIVIDEND_DECL` and `BUYBACK`; a real-borrow path in `borrow()`, memoised and
    unchanged when no data exists; fundamentals fields `eps_consensus_ntm` and `eps_num_analysts_ntm`.
  * `fetch.py`: tasks `dividend_decl` and `buybacks`.
* **S045 (buybacks)**:
  * 11,307 candidate documents → 695 events, 294 issuers, 2009–2026; 373 meet the ≥ 5% non-earnings rule.
  * Precision 27/30 on a hand-read sample.
  * 2019 smoke run: 14 trades.
  * Tier T2.
* **S042 (dividend declarations)**:
  * 8,059 events, 466 issuers.
  * Recall about 25% of S&P cash dividends 2018+. Issuers that don't use 8-Ks are missing (e.g. MSFT, JPM, JNJ),
    and there are none for ETFs.
  * AAPL: 54/54 dates match. Text check: 29/30.
  * Trades declared dividends only; tier T2.
  * The engine refuses the full universe: 1-minute coverage of `SP500_TOP50_DV` was 44% at the time.
* **S041 (analyst revisions)**:
  * DoltHub weekly consensus: history from the Hugging Face copy (2017-10-26 → 2026-01-25) plus 39 snapshots
    from the DoltHub API (to 2026-10-25). 667,532 rows.
  * Each snapshot is known at the close of the first session after its date.
  * Next-12-month EPS = FY1/FY2 blend weighted by time left in FY1; analyst count = min of the two; snapshots
    older than 21 days are ignored.
  * 5,890 snapshots with unknown identity were dropped.
  * Coverage: 497/506 members at mid-2018, 502/503 at 2026-10.
  * Spot checks: NVDA FY1 4.52 → 7.46 after 2023-05-24; META cut after 2022-02-02; AAPL 13.65 → 12.16 in
    Mar–Apr 2020.
  * 2018–2019 smoke run: 1,331 trades. `tests/data` + `tests/strategies`: 1,084 passed, 0 failed.
  * Window 2017-11 → 2026-10; tier T2.
* **Borrow**:
  * 0 rows stored. iBorrowDesk's free endpoints block scripted clients; its history (from 2015-07-13) is only in
    a paid keyed API (`IBORROWDESK_API_KEY`, $29/month plus unit packs).
  * IBKR's public short-stock FTP is a current snapshot only and did not resolve from the old machine.
  * The default borrow schedule (T2) remains in use.
* **S110 (iNAV)**: no free source found.
* **Nasdaq's dividend API**: disallowed by robots.txt; not used.

### 6.2 Alpaca intraday (agent stopped mid-way)

* **Files**:
  * `providers/alpaca.py` (live-verified 2026-10-30), `providers/_keyed.py`, new `providers/alpaca_eod.py`.
  * Intraday tasks in `fetch.py`.
  * `execution/spread_calibrated.py`, `execution/calibration/spread_calibration.json` (now version 4),
    `spread_calibration_v3.json`, `validation_report.json`, `alpaca_validate.py`, `tools/alpaca_nbbo_sample.py`,
    `alpaca_nbbo_2016_2026.csv.gz`.
  * Tests: `tests/data/test_alpaca.py`, `tests/execution/test_alpaca_calib.py`,
    `tests/execution/test_spread_calibrated.py`.
  * Fixtures: `tests/data/fixtures/alpaca_*.json`.
* **Fetch tasks**: `alpaca_universe`, `alpaca_bars_etf`, `alpaca_bars_top`, `alpaca_quotes`,
  `alpaca_bars_sp500_5m`. They are grouped as `--set intraday` and skip symbol-months already stored.
* **State at the stop**:

  | Item | State |
  |---|---|
  | `alpaca_bars_top` | finished (10,434 requests) |
  | `alpaca_bars_sp500_5m` | stopped at month 2025-07 |
  | `alpaca_quotes` | stopped mid-run (last logged window `close_m36` 2021-02) |
  | `bars_1m/alpaca` | 305 symbols, 4.2 GB |
  | `bars_5m/alpaca` | 693 symbols, 0.6 GB |
  | `quotes/alpaca` | 289 symbols, 0.5 GB |

* **Spread model vs real Alpaca NBBO**:
  * Sample: 19,066 rows; 40 days 2016-01-13 → 2025-10-08; 632 symbols; second Wednesday of each month, seven
    one-minute windows.
  * v3: mean absolute error 3.27 bp, median 1.18 bp, 64.9% within ×2, predicted below observed in 18% of rows.
  * v4, out-of-fold: mean absolute error 1.76 bp, median 0.49 bp, 83.5% within ×2, predicted below observed in
    33%.
  * v4 is in the working tree. It was not red-teamed, and goldens were not re-run against it.
* **Not done**:
  * Finishing the 5-minute and quote fetches.
  * The NBBO rebuild in decision windows.
  * The final `DATA.md`/`SPREADS.md` text.
  * A report of which of S001–S040 became testable.

### 6.3 Survivorship 2011+ (agent stopped mid-way)

* **Files**:
  * `providers/{dolthub,dolthub_bulk,sp500,tiingo}.py`, new `providers/alpaca_eod.py`.
  * `data/identity.py`, `data/universes.py`, `run/plan.py`.
  * Tests: `tests/data/test_survivorship_patch.py`, `tests/run/test_plan_star.py`, `tests/run/test_plan_tiering.py`.
  * `fetch.py` tasks `alpaca_eod` and `tiingo_eod`, each followed by `--only identity`.
* **Findings**:
  * DoltHub `stocks.ohlcv` (18,893 symbols) has no symbol whose last bar is before 2018-01-01, so it cannot patch
    names delisted before 2018.
  * Alpaca daily (2016+): 106 tickers fetched. Validation passed 100% (94 comparisons; median |log close ratio|
    1e-8; 26/26 splits reproduced).
  * Tiingo: 101 symbols stored. Many delisted names start only 2016-01-04. Names delisted before about 2015 are
    mostly absent. A reused ticker serves its current holder. Rows are used only under the CIK-ownership and
    price-agreement rules in `DATA.md` §4.19.
* **Identity fixes in the working tree**:
  * Share-class sibling evidence: VIAB → VIA, DISCK → DISCA, CMCSK → CMCSA.
  * `pit_rebase` of fja intervals: AABA → YHOO, VIAV → JDSU, ATGE → DV, FNMA → FNM, ABKFQ → ABK, IAC → IACI,
    KATE → LIZ.
  * Dropped contradicted splits: GE 2024-04-02 1.253 and SSP 2025-12-08 1.33.
* **S&P 500 point-in-time price coverage**, measured by the agent (`surv/before.json` vs
  `surv/after_final2.json`):

  | Month | Before | After |
  |---|---|---|
  | 2011-01 | 74.7% | 85.3% |
  | 2012-07 | 78.3% | 88.5% |
  | 2013-07 | 80.3% | 89.7% |
  | 2014-07 | 81.9% | 91.2% |
  | 2015-07 | 83.8% | 91.8% |
  | 2015-12 | — | 92.6% |
  | 2016-01 | 86.1% | 99.2% |
  | 2017-07 | 92.3% | 99.6% |
  | 2018-07 | 96.6% | 99.8% |
  | 2023-06 → 2026-10 | — | 100% |

  * Coverage is ≥ 95% from 2016-01. It is not ≥ 95% for 2011–2015: about 73 member names are missing in 2011.
  * `DATA.md` §8.2 still shows the 2026-10-29 table.
* **Not done**:
  * `plan.py` changes:
    * `symbols=("*",)` coverage check;
    * tiering from non-critical requirements: S042 `DIVIDEND_DECL`, S045 `BUYBACK`;
    * S041 fields `eps_consensus_ntm` must start 2017-11-30 and take their tier from `fidelity_of` (T2), not the
      2009 XBRL start and a fixed T1;
    * the planner must mark untestable what the engine refuses on coverage.
  * Moving the primary-window start.
  * Regenerating `docs/plan.csv` and updating the `STATUS.md` counts.
  * The final test run.

### 6.4 Databento options (agent stopped mid-way)

* **Files**:
  * `providers/databento.py` (live-verified 2026-10-30), new `options/validate_real.py`.
  * `strategies/options_events_common.py`, `strategies/options_events_earnings.py`.
  * Test: `tests/data/test_databento_opra.py`.
* **Method**: narrow `cbbo-1m` requests on `OPRA.PILLAR` by `raw_symbol`, covering only the contracts and
  minutes a strategy touches. For scale, the whole SPY chain for one minute costs $0.0107.
* **Jobs and last logged state**:

  | Job | Last logged state |
  |---|---|
  | `spx0dte` | plan of 900 jobs, 2023-03-28 → 2026-10-29 |
  | `spxeod` | reached 2025-04-08 |
  | `s160snap` | reached 2025-02-28 |
  | `s160puts` | reached 2026-07-09 |
  | `s171` | reached 2024-04-24 |
  | `earnings` | in progress |

* **Stored**: `option_quotes/databento`, 0.8 GB; `bars_1m/opra_parity` (4 files);
  `reference/databento_validation` (24 files).
* **Not done**:
  * Finishing the jobs.
  * Serving the quotes as T0 through `portal.option_quotes()` with SPX vs SPXW root and settlement identity.
  * A `databento_options` fetch task test.
  * The synthetic-surface bias measurement against real quotes.
  * Docs.

## 7. Known issues

* **Goldens depend on the cache.** With the new Alpaca quotes, 9 of 16 golden cases differ on unmodified `main`,
  because the engine uses real quotes when they exist. The differing cases are S001, S013, S046 ×2, S068, S074,
  S085, S103 and S105.
  * The broker change was verified by A/B: `main` vs candidate on the same cache, digests compared.
  * Goldens have not been re-recorded.
* **The full offline test suite** exceeded 40 minutes on the shared 4-core machine and was killed.
  * Last full-suite numbers, from the broker agent, mid-work: 1,923 passed, 212 skipped, 4 failed. The four
    failures were three synthetic-chain tests, since fixed in `5bd53a0`, and the VIAB security-master test,
    caused by the in-progress `identity.py` edits.
* **No pytest-xdist** is installed, so `-n` is unavailable.
* **Imports outside the repo**: scripts that live outside the repository need `PYTHONPATH=<repo>` to import
  `atlasbt`.
* **Unreachable sources**: web.archive.org and Stooq were unreachable through the old proxy.
* **The full atlas run has never completed.** It needs about 1–2 days on 4 cores. Container recycling killed
  earlier attempts, after about 11 strategies.
* **Data volume**: about 11 GB now (cache plus `data_raw`), about 20–25 GB expected after the fetches finish,
  plus about 1 GB of results.

## 8. Commands

```bash
pip install -e .                                           # Python >= 3.11
export ATLASBT_CACHE=<cache path>                          # plus the key variables in §2
ATLASBT_OFFLINE=1 python -m pytest -q tests/<area>         # offline tests
ATLASBT_SLOW=1 ATLASBT_OFFLINE=1 python -m pytest -q tests/perf/test_golden.py
python -m atlasbt.run plan
python -m atlasbt.run replicate
python -m atlasbt.run launch --commit <sha> --no-overlay final --workers 4 \
    --etf-budget -1 --options-budget 45 --stock-budget 60 --budget 30 --label final-<sha>
python -m atlasbt.data.fetch --set core                    # free sources (~3 GB)
python -m atlasbt.data.fetch --only alpaca_eod             # WIP branch
python -m atlasbt.data.fetch --only tiingo_eod             # WIP branch
python -m atlasbt.data.fetch --only identity
python -m atlasbt.data.fetch --only dividend_decl          # WIP branch, needs SEC_USER_AGENT
python -m atlasbt.data.fetch --only buybacks               # WIP branch, needs SEC_USER_AGENT
python -m atlasbt.data.fetch --set intraday                # WIP branch, Alpaca
python -m atlasbt.data.fetch --only databento_options      # WIP branch, budget-guarded
python -m atlasbt.data.ca_crosscheck                       # report in cache/qa
```

A/B golden digest script, used for the broker verification. Run it in each worktree, then compare the JSON files:

```python
import sys, json; sys.path.insert(0, 'tests/perf'); import golden
out = {c: golden.digest_result(golden.run_case(c)[0]) for c in sorted(golden.CASES)}
json.dump(out, open(sys.argv[1], 'w'), default=str)
```

## 9. Owner-facing discussion so far

* **Production setup**: the owner asked how to make everything persist and said they are willing to pay.
  * Options discussed: an always-on server (e.g. Hetzner, about €15–30/month), with backups of data and
    results to Cloudflare R2 or Backblaze B2 (about 25 GB).
  * Also: keys stored as environment settings, the atlas run as an auto-restarting service, and Claude Code on
    the server via `claude remote-control`.
  * No decision has been recorded.
* **Pending owner answer**: whether to relaunch the stopped Alpaca, survivorship and Databento work.
