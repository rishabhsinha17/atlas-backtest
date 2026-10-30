"""Active integrator notices that qualify every runner output (printed at the top of ATLAS_REPORT.md,
leaderboard.md, REPLICATION.md and each strategy report). Remove an entry when its fix has landed and
the affected results were re-run with --force."""

ACTIVE_NOTICES = [
    ("CALIBRATED-SPREADS", "Taker costs use the engine's calibrated spread model (v3 + 2010-2025 pessimistic floor, "
     "T2 'spread modelled (calibrated)'): SPY coin-flip round trip ~1.0 bp vs 31 bp under the legacy Corwin-Schultz "
     "model; per-fill slippage 0.26 bp vs 0.31 bp quoted (IEX 2019-12-24)."),
    ("STOCK-COVERAGE", "Stock-universe (S&P 500 PIT) primary results start at the first month with >= 95% PIT price "
     "coverage; members without verified data are excluded by the engine (coverage recorded per result, flagged T2). "
     "EXTENDED_SURVIVORSHIP_BIASED secondary runs (>= 60% coverage, results/extended) are never ranked: delisted and "
     "failed names are under-represented, so they are likely optimistic."),
    ("SYNTHETIC-FILLS-T3", "Options filled from synthetic session-close snapshots (no real chain: SPX/QQQ/IWM, SPY "
     "before 2019/2020) are T3_MODELED (OPTIONS.md §7.4: synthetic put-writes ~+1.5%/yr rich vs the Cboe PUT index); "
     "the fidelity scorecard shows the T2/T3 split."),
    ("QUOTES-AUTO", "use_quotes='auto': quote fills only where cached quotes cover the bar window (IEX 2019-12-24, "
     "Dukascopy 2026-10-27 sample days); bars elsewhere."),
]


def banner_md() -> str:
    return "\n".join(["> **Active notices**"] + [f"> * **{k}**: {v}" for k, v in ACTIVE_NOTICES]) + "\n"
