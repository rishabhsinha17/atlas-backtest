# Handover: state of the work and how to continue

Written 2026-11-03 for the next agent. Read this first, then `README.md`, `docs/STATUS.md` and
`docs/ARCHITECTURE.md`.

## 1. The goal

The owner asked for the highest-fidelity backtest engine possible on free (now also cheap paid) data for the
172 strategies in `docs/Trading_Strategy_Atlas_172_Systems.xlsx`. Every result must be point-in-time, honest
about its fidelity tier, and independently red-teamed. The deliverable is:

1. the engine (done, on `main`);
2. a full run of every testable strategy with one ranked report (`results/ATLAS_REPORT.md`) — **not done yet**.

The owner prefers short, plain-English updates. They want work done in parallel and fast, and they have
authorised pushing anything to this repository (`rishabhsinha17/atlas-backtest`).

## 2. Rules that must not be broken

* **This repository is public.** Never commit API keys, secrets or personal data. Before every push, grep the
  tree for the key values and for the owner's email domain, and abort on a match.
* **Commit author**: `Rishabh Sinha <rishabhsinha17@users.noreply.github.com>` (noreply address only).
* **Keys are read from the environment only**: `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `TIINGO_API_KEY`,
  `DATABENTO_API_KEY`, `SEC_USER_AGENT` (the owner approved a contact string for SEC; never hardcode or log it).
  On the old cloud machine they were in `~/.atlasbt/secrets.env` (chmod 600, `source` it). The keys were pasted
  in chat once, so the owner should regenerate them; ask for the new ones to be put in environment settings,
  never in chat.
* **Network etiquette** (`docs/ARCHITECTURE.md` §12): user agent `atlasbt-research/0.1`, no browser spoofing,
  no bot-check bypassing, respect robots.txt and rate limits. Alpaca ≤ 180 req/min; Tiingo free tier
  ≈ 50 new symbols/hour; SEC ≤ 5 req/s.
* **Databento**: free credits only. Hard cap $90 total. Every request is priced first with
  `metadata.get_cost` and logged to `$ATLASBT_CACHE/raw/databento_ledger.csv`; refuse anything that would cross
  the cap. Spent so far: $18.42 (3,605 requests).
* **Data licences**: Alpaca, Tiingo and Databento data must not be republished. Keep the cache out of git and
  out of this public repo. (Check: `atlasbt/execution/calibration/alpaca_nbbo_2016_2026.csv.gz` on the WIP
  branch holds derived spread statistics per symbol-window, not raw quotes; confirm that is acceptable before
  merging it, or move it to private storage.)
* **Honesty**: never fabricate data or results. Missing data → `NOT_TESTABLE` with a reason, or a lower tier.
* **Point-in-time**: every record has `known_at`; bars are labelled by start and known at end; fills only at
  the next eligible bar or quote.
* **One atlas process at a time.** Two concurrent `final` runs caused out-of-memory kills before.

## 3. Where things are

| What | Where |
|---|---|
| Verified code | `main` (HEAD `5bd53a0`) |
| Unfinished, untested work from stopped agents | branch `wip/gap-closing` (force-pushed snapshots on top of `main`) |
| Data cache (~9.6 GB) | old cloud machine `/home/user/atlas_bt/cache` (`ATLASBT_CACHE`). **Not backed up anywhere.** A new machine must rebuild it (§7). |
| Raw scout downloads (~1.2 GB, gitignored) | old machine `/home/user/atlas_bt/data_raw`. Only 10 files are in git. |
| Earlier partial results (11 strategies, superseded) | old machine `/home/user/atlas_bt/results` |

`main` commits since the engine was finished:

* `5bd53a0`: pattern-day-trader rule (FINRA 4210, on by default, binds only under $25k equity); opt-in dated
  IBKR commission and financing schedules; corporate-action cross-check (`python -m atlasbt.data.ca_crosscheck`);
  Cboe vol-index loader falls back to the cache store (`bars_1d/cboe`) because pinned worktrees lack the
  gitignored CSVs — before this, NDX/RUT/single-stock synthetic chains failed in pinned runs. Verified: all
  16 golden cases bit-identical to the previous commit on the same cache.
* `bccbee1` and earlier: engine, data layer, options, spreads, 172 strategies, runner, docs (see `STATUS.md`).

## 4. What is on `wip/gap-closing` and its status

Five agents worked in parallel on the remaining fidelity gaps. Two finished; three were stopped mid-way when
the cloud machine restarted (2026-10-30 08:54 UTC).

### Finished, tested, not yet merged (shares files with unfinished work)

**Events / borrow** (owner files: `providers/{buybacks,dividends_decl,estimates,iborrowdesk}.py`, `sec.py`,
`borrow.py`, `strategies/calendar_events.py`, plus additive `portal.py`/`fetch.py` edits):

* **S045 buybacks**: SEC EDGAR full-text search → 695 events, 294 issuers, 2009–2026 (373 meet the ≥5%
  non-earnings rule). Precision 27/30 on a hand-checked sample. Testable at T2.
* **S042 dividend declarations**: 8,059 events from 8-K filings, about 25% recall of S&P cash dividends
  (issuers that don't announce via 8-K are missing). 29/30 correct. S042 trades only declared dividends, T2.
  Still gated by 1-minute bar coverage of `SP500_TOP50_DV` (44% at the time).
* **S041 analyst revisions**: DoltHub weekly consensus EPS (FY1/FY2 blend → NTM, analyst counts), 667,532 rows,
  2017-11 → 2026-10, served as `portal.fundamentals` fields `eps_consensus_ntm`, `eps_num_analysts_ntm`. Each
  snapshot is known at the close of the next session. Testable at T2; 2018–2019 smoke run OK (1,331 trades).
* **Borrow history**: not closeable for free. iBorrowDesk history needs its paid API ($29/month). The provider
  is built and tested offline. The default borrow schedule (T2) stays.
* **S110 iNAV**: no free source. Stays `NOT_TESTABLE`.

### Unfinished (agents stopped; resume them)

**Alpaca** (owner files: `providers/alpaca.py`, `_keyed.py`, `alpaca_eod.py`, intraday tasks in `fetch.py`,
`execution/spread_calibrated.py`, `execution/calibration/**`):

* Provider live-verified against the real API.
* Stored so far: 1-minute bars for 305 symbols (4.2 GB), 5-minute bars for 693 symbols (0.6 GB), quotes for
  289 symbols (0.5 GB).
* Fetch tasks exist: `python -m atlasbt.data.fetch --set intraday` (= `alpaca_universe`, `alpaca_bars_etf`,
  `alpaca_bars_top`, `alpaca_quotes`, `alpaca_bars_sp500_5m`). They skip already-stored symbol-months.
* Still to do: finish the fetches; rebuild the NBBO from the quote stream in decision windows; validate and,
  if needed, refit the spread model against real quotes (a v3 calibration file and `alpaca_validate.py` exist,
  unverified); update `DATA.md` and `SPREADS.md`; report which of S001–S040 become testable.

**Survivorship 2011+** (owner files: `providers/{dolthub,dolthub_bulk,sp500,tiingo}.py`, `data/identity.py`,
`data/universes.py`, `run/plan.py`):

* The owner decided to ignore everything before 2011. Target: ≥ 95% point-in-time S&P 500 price coverage from
  2011-01-01, so primary stock windows start in 2011 (they start in 2018 today).
* Before: 2010 65.7%, 2012 78.3%, 2015 84.0%, 2018 96.6%. The after-numbers were never reported.
* Sources wired: DoltHub, Alpaca daily (2016+, task `alpaca_eod`), Tiingo daily (task `tiingo_eod`, 101 symbols
  stored; beware reused tickers such as BBT — map via CIK and fail closed). Run `--only identity` afterwards.
* Still to do, in `plan.py`:
  * Check `symbols=("*",)` bars requirements with the same universe-coverage rule the engine uses (a test file
    `tests/run/test_plan_star.py` exists).
  * Tier from non-critical requirements: S042 `DIVIDEND_DECL`, S045 `BUYBACK`.
  * S041 fundamentals fields `eps_consensus_ntm` start 2017-11-30 with their tier from `fidelity_of` (T2).
    In general, take per-field start and tier from the portal, not from constants.
  * Planner must mark untestable what the engine refuses on coverage.
* Still to do, data fixes:
  * GE 2024-04-02 is stored as both a spin-off and a 1.253 split (multiplies GE shares).
  * SSP 2025-12-08 has a 1.33 split with no price gap.
  * The security-master test fails on VIAB, after this agent's `identity.py` edits.
  * Splits disagree in 25% of comparable events between DoltHub and the other source, mostly splits that
    DoltHub omits.
* Then regenerate `docs/plan.csv` and update the testability counts in `STATUS.md`.

**Databento options** (owner files: `providers/databento.py`, `options/**` incl. new `validate_real.py`,
`strategies/options_events_{common,earnings}.py`):

* Targets: S158–S160, S164, S171 (intraday option quotes) and S161–S163, S172 (SPX). Data is bought as narrow
  `cbbo-1m` requests on OPRA.PILLAR (only the contracts and minutes a strategy touches; the whole SPY chain is
  ≈ $0.01/minute).
* Stored so far: 0.8 GB in `option_quotes/databento` (SPXW 0DTE, SPX/SPXW end-of-day, S160, S171, earnings).
* Still to do: finish within the cap; serve the quotes through `portal.option_quotes()` as T0 (OCC root and
  settlement identity, SPX vs SPXW, AM/PM); add the `databento_options` fetch task, tests and docs; measure the
  synthetic (T3) surface bias against real quotes (known: ≈ 1.5%/yr rich vs the Cboe PUT index).

## 5. How to verify before merging to `main`

1. Make an isolated worktree from `main` and copy in only the files of the piece being merged, so unfinished
   work doesn't leak in.
2. Run the affected tests offline:
   `ATLASBT_OFFLINE=1 ATLASBT_CACHE=<cache> python -m pytest -q tests/<areas> tests/redteam/<areas>`.
   The full suite takes over 40 minutes on a busy 4-core machine; run subsets.
3. Golden tests are skipped unless `ATLASBT_SLOW=1`. **Goldens depend on the cache**: the new Alpaca quotes
   change fills (the engine uses real quotes when they exist), so 9 of 16 goldens differ even on unmodified
   `main`. That is expected. Judge a code change by A/B: run every golden case on `main` and on the candidate
   against the same cache and compare digests. A script for that is in §9. Re-record goldens
   (`tests/perf/golden.py`) only once the cache is final, and say so in the commit.
4. Process used throughout: builder agent → independent red-team agent (failing tests into
   `tests/redteam/...`) → independent verifier → fix. The new data work has not been red-teamed yet. Do that
   before the final run.

## 6. Running the atlas

```bash
python -m atlasbt.run plan                       # offline; writes the plan (copy to docs/plan.csv)
python -m atlasbt.run launch --commit <sha> --no-overlay final --workers 4 \
    --etf-budget -1 --options-budget 45 --stock-budget 60 --budget 30 --label final-<sha>
```

* The run is resumable and pinned to a clean worktree of `<sha>`, with `ATLASBT_CACHE` pointing at the main
  cache.
* It takes about 1–2 days on 4 cores.
* Outputs: `results/ATLAS_REPORT.md`, `leaderboard.csv`, `AUDIT.md`, per-strategy reports.
* Proxy variants (S031@proxy etc.) are reported in their own T2 table.
* Deflated Sharpe uses N = 1450.
* Before launching, make sure no other atlas process is running.

## 7. Rebuilding the data on a new machine

```bash
pip install -e .
export ATLASBT_CACHE=<path>   # plus the keys in §2
python -m atlasbt.data.fetch --set core                    # free sources, ~3 GB
python -m atlasbt.data.fetch --only alpaca_eod tiingo_eod  # survivorship patches, then:
python -m atlasbt.data.fetch --only identity
python -m atlasbt.data.fetch --only dividend_decl buybacks # SEC (needs SEC_USER_AGENT)
python -m atlasbt.data.fetch --set intraday                # Alpaca bars + quotes (~10+ GB)
python -m atlasbt.data.fetch --only databento_options      # budget-guarded; check the ledger first
```

Check the exact flags with `python -m atlasbt.data.fetch -h`. Some task names exist only on the WIP branch.
The estimates (S041) loader is in `providers/estimates.py`; check how its fetch is wired. The gitignored
`data_raw/` scout files are not in git. The code falls back to the store where it can (Cboe indices).
Otherwise re-download them via `data_raw/SOURCES.md`.

## 8. Production setup the owner is considering

The cloud sandbox resets: restarts kill running jobs, and a new session starts without the cache. The owner is
willing to pay a little. The recommended setup:

* **Server**: a small always-on server (e.g. Hetzner, Ubuntu, 8 vCPU / 16–32 GB RAM / ≥ 160 GB disk,
  roughly €15–30/month).
  * Run Claude Code there with `claude remote-control`.
  * Run the atlas as a systemd service that restarts and resumes after a crash.
* **Backups**: data and results backed up nightly to Cloudflare R2 or Backblaze B2 (about 25 GB total,
  well under $1/month), with a one-command restore.
* **Keys**: in a chmod-600 env file on the server.
* **Results**: committed per strategy.

Storage needed: about 11 GB now, about 20–25 GB once the downloads finish, plus about 1 GB of results.

## 9. Gotchas learned the hard way

* The cloud container recycles when the session is idle; long jobs die. Use `nohup`, resumable tasks and
  periodic check-ins, or run on the always-on server.
* `pytest -n` is not available (no xdist).
* Scripts outside the repo need `PYTHONPATH=<repo>` to import `atlasbt`.
* `data_raw/*` is gitignored; pinned worktrees only see the 10 tracked files.
* Nasdaq's API is disallowed by robots.txt. iBorrowDesk blocks scripted clients on free pages. web.archive.org
  was unreachable through the old proxy.
* A/B golden script (run once per worktree, then compare the JSON digests):

  ```python
  import sys, json; sys.path.insert(0, 'tests/perf'); import golden
  out = {c: golden.digest_result(golden.run_case(c)[0]) for c in sorted(golden.CASES)}
  json.dump(out, open(sys.argv[1], 'w'), default=str)
  ```

## 10. Next steps, in order

1. Resume the Alpaca, survivorship and Databento work from the WIP branch, with the data already fetched.
2. Merge each piece to `main` after A/B goldens and tests pass. Merge the events work together with the shared
   `portal.py`/`fetch.py` changes.
3. Red-team the new data work (identity with Tiingo/Alpaca EOD, NBBO rebuild, OPRA identity, event `known_at`).
4. Re-plan, update `STATUS.md` and `plan.csv`, and re-record goldens on the final cache.
5. Launch the full atlas once, keep it alive, then audit and retier, and give the owner a short plain-English
   report plus `ATLAS_REPORT.md`.
