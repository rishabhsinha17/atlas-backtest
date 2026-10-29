"""Active integrator notices that qualify every runner output (printed at the top of ATLAS_REPORT.md,
leaderboard.md, REPLICATION.md and each strategy report). Remove an entry when its fix has landed and
the affected results were re-run with --force."""

ACTIVE_NOTICES = [
    ("CALIBRATED-SPREADS", "Runs labelled calibrated-v1 use the engine's calibrated spread model (default since e3d3d4d; "
     "T2 'spread modelled (calibrated)'): SPY coin-flip round trip 1.0 bp vs 31 bp under the legacy Corwin-Schultz "
     "model, per-fill slippage 0.27 bp vs 0.31 bp quoted (IEX 2019-12-24). Earlier Corwin-Schultz results are "
     "superseded. Results remain pre-final until the pending engine fixes (quote mode per session, combo legs on "
     "master instruments, stale-mark early assignment) land."),
    ("QUOTES-NEVER", "Runs use BacktestConfig(use_quotes='never'): the engine's 'auto' mode switched SPY/QQQ/AAPL to "
     "quote-only fills because single sample days of quotes are cached (fix dispatched to ENGINE)."),
    ("STOCK-HOLD", "Single-stock strategies are on HOLD again (integrator 2026-10-29 07:10): the engine path resolves "
     "identity at as_of=range_end, so reused tickers show the wrong company (BBT 2019 at Berkshire Hills prices). "
     "All stock-universe outputs (incl. the UMD/HML replications) are STALE until DATA's identity-qualified keys land. "
     "When lifted: primary stock runs use the >=95% coverage window (2018-04+) with --universe-coverage filter; "
     "EXTENDED_SURVIVORSHIP_BIASED runs (>=60%) live in results/extended and are never ranked."),
    ("SYNTHETIC-FILLS-T3", "Since 2026-10-29 Portal.option_quotes falls back to synthetic session-close snapshots (T3), "
     "so synthetic-chain options strategies fill; their results are T3_MODELED (OPTIONS.md §7.4: synthetic "
     "put-writes ~+1.5%/yr rich vs the Cboe PUT index). Options runs made before the fallback are invalid."),
]


def banner_md() -> str:
    return "\n".join(["> **Active notices**"] + [f"> * **{k}**: {v}" for k, v in ACTIVE_NOTICES]) + "\n"
