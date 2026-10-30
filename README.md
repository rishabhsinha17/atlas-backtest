# atlas-backtest

A point-in-time backtest engine for the 172 strategies of the *Trading Strategy Atlas*
(`docs/Trading_Strategy_Atlas_172_Systems.xlsx`): US stocks, ETFs and options, built to be as faithful to
real-life execution as free data allows.

* **No lookahead.** Every data read is pinned to the current time and raises `LookaheadError` otherwise.
  Signals use completed bars only; orders fill at the next eligible bar or quote, never at a past price.
* **Realistic execution.** Bid-ask spreads come from a model calibrated to NYSE TAQ consolidated quotes, with
  era-aware tick regimes (1/8, 1/16, decimal). On top of that: square-root market impact, gap-aware stops,
  participation caps, and historical SEC/FINRA/OCC fee schedules. The engine also enforces Reg SHO Rule 201,
  locates and borrow fees, and Reg-T and Cboe option margin. It handles dividends, splits, spin-offs and
  delistings, plus option exercise, assignment and AM/PM settlement.
* **Survivorship-aware data.** Security identity is resolved by SEC CIK per bar date, so a reused ticker never
  splices two companies. When identity can't be verified the data fails closed, and S&P 500 coverage is
  reported, not hidden.
* **Honest fidelity.** Every fill is tagged:
  * **T0:** executable quotes.
  * **T1:** real bars with a modelled spread.
  * **T2:** proxy timing or inputs.
  * **T3:** modelled data, e.g. synthetic option chains.

  Strategies whose required data doesn't exist for free return `NOT_TESTABLE` with a reason, never a fake
  result.
* **Statistics.** Walk-forward parameter selection with an untouched holdout, cost-stress scenarios A–E, a
  break-even cost solver and stationary bootstrap. Multiple-testing checks: deflated Sharpe with an atlas-wide
  trial count, PBO (CSCV), White/Hansen SPA and Benjamini–Hochberg FDR.

Every layer was attacked by independent red-team agents. Each finding was adjudicated by a separate verifier
before being fixed. See [`docs/STATUS.md`](docs/STATUS.md).

## Quick start

```bash
pip install -e .                                   # Python >= 3.11
python -m atlasbt.data.fetch --set core            # build the local data cache from free sources (~3 GB)
ATLASBT_OFFLINE=1 python -m pytest -q              # test suite (offline; data-dependent tests skip without cache)
python -m atlasbt.run plan                         # which strategies are testable, windows, expected fidelity
python -m atlasbt.run replicate                    # end-to-end known-answer checks
python -m atlasbt.run launch --commit HEAD --no-overlay final --workers 4   # full atlas, resumable
```

Results go to `results/`: per-strategy reports, `leaderboard.csv`, `ATLAS_REPORT.md`, `AUDIT.md` and
`REPLICATION.md`. A full run takes roughly 1–2 days on 4 cores. Interrupted runs resume where they stopped.

## Keys

None are required: no API keys and no LLM tokens. Optional free keys raise fidelity:

| Variable | Effect |
|---|---|
| `ALPACA_API_KEY`, `ALPACA_API_SECRET` | consolidated 1-minute bars and NBBO quotes since 2016; unlocks the intraday strategies |
| `DATABENTO_API_KEY` | intraday option quotes (OPRA) for 0DTE, earnings, FOMC and parity strategies |
| `POLYGON_API_KEY` | additional minute bars |
| `SEC_USER_AGENT` | contact string; only needed when rebuilding SEC data |

The key-gated providers are written from official API docs but were not live-tested (no keys were available).

## Documentation

| File | Contents |
|---|---|
| [`docs/STATUS.md`](docs/STATUS.md) | what is verified, red-team record, testability, known limitations |
| [`docs/REPLICATION.md`](docs/REPLICATION.md) | end-to-end known-answer results |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | contracts, engine loop, fidelity tiers, rules |
| [`docs/ENGINE.md`](docs/ENGINE.md) | fills, costs, accounting, margin, sizing; every assumption with its source |
| [`docs/DATA.md`](docs/DATA.md) | sources, point-in-time conventions, identity, coverage |
| [`docs/OPTIONS.md`](docs/OPTIONS.md) | pricing, synthetic chains, lifecycle, margin |
| [`docs/SPREADS.md`](docs/SPREADS.md) | calibrated spread model and its validation |
| [`docs/ANALYTICS.md`](docs/ANALYTICS.md) | indicators, validation statistics, reporting |
| [`docs/STRATEGY_GUIDE.md`](docs/STRATEGY_GUIDE.md) | how sheet rules were interpreted |
| [`docs/RUNNING.md`](docs/RUNNING.md) | runner CLI and pinned-commit runs |
| [`docs/PERF.md`](docs/PERF.md) | performance work and bit-for-bit golden tests |

This is a research tool. Nothing here is evidence that any strategy makes money.
