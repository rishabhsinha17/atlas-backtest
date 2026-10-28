"""Metrics, fidelity scorecards, per-strategy reports and the atlas leaderboard."""
from .fidelity import grade, normalize_scorecard, render_scorecard_md, scorecard_table
from .leaderboard import build_leaderboard, render_leaderboard_md, summarize_result, write_leaderboard
from .metrics import compute_metrics
from .report import strategy_validation, write_strategy_report
from .schema import normalize_fills, normalize_ledger, normalize_positions, normalize_trades

__all__ = [n for n in dir() if not n.startswith("_")]
