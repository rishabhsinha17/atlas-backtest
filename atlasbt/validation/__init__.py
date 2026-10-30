"""Out-of-sample validation and statistical adjustment for atlasbt results.

See docs/ANALYTICS.md for definitions and references.
"""
from .bootstrap import (BootstrapCI, bootstrap_ci, bootstrap_indices, bootstrap_metrics, circular_indices,
                        optimal_block_length, stationary_indices)
from .cost_stress import (SCENARIOS, all_in_round_trip_bp, apply_cost_stress, break_even_cost, cost_stress_table,
                          measured_costs_bp)
from .cv import PurgedKFold
from .multiple_testing import benjamini_hochberg, hansen_spa, white_reality_check
from .pbo import PBOResult, pbo_cscv
from .regimes import all_regimes, decade, rate_regime, regime_table, trend_regime, vix_terciles
from .sharpe_tests import (atlas_trials, dsr, dsr_adjusted, dsr_from_returns, dsr_variance, effective_trials, expected_max_sharpe,
                           min_track_record_length, parse_grid_size, psr, psr_from_returns, total_atlas_trials)
from .walkforward import (WalkForwardResult, expand_grid, param_key, walk_forward_matrix, walk_forward_run,
                          walk_forward_splits)

__all__ = [n for n in dir() if not n.startswith("_")]
