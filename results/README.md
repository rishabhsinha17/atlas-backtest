# results/

Outputs of `python -m atlasbt.run` (see docs/RUNNING.md). Everything here except this README is
git-ignored and reproducible from the cache + code (provenance in `runs/<SID>/config.json`).

* `plan.csv`, `plan_cache.json` - requirement/window plan for all 172 strategies
* `runs/<SID>/` - baseline result, grid returns, grid plan, final summary (resumable state)
* `reports/<SID>/` - per-strategy report.md/json + charts
* `ATLAS_REPORT.md`, `leaderboard.{csv,md}`, `not_testable.csv`, `atlas_stats.json`
* `AUDIT.md`, `audit.json` - activity safeguard audit
* `REPLICATION.md`, `replication.json` - end-to-end known-answer validations
* `logs/` - run.log (scheduler), <SID>.log (engine/strategy output, tracebacks)
