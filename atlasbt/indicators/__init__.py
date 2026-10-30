"""Causal technical / statistical indicators for atlasbt.

Contract: the value at position t depends only on inputs at positions <= t
(verified by perturbation tests for every public function). Warm-up values
are NaN; nothing is back-filled. See docs/ANALYTICS.md for definitions,
conventions and references.
"""
from .moving import ema, sma, wilder
from .oscillators import cci, macd, rsi, stochastic, williams_r
from .volatility import (atr, bandwidth_percentile, bollinger, donchian, keltner, overnight_variance_share,
                         percentile_rank, realized_vol, returns, rolling_std, squeeze_on, true_range,
                         vol_scaled_to_time, zscore)
from .trend import adx, dmi, high_ratio, ichimoku, linreg_slope, parabolic_sar, prior_all_time_high, supertrend
from .patterns import (bullish_engulfing, darvas_box, double_bottom, down_streak, flag_pattern, head_and_shoulders,
                       ibs, inside_bar, nr_n, pivots, volatility_contraction)
from .intraday import (adv, dollar_volume, minutes_since_open, opening_range, premarket, relative_volume,
                       session_extremes, session_vwap, time_of_day_profile, vwap_bands, window_stats)
from .statarb import (adf, engle_granger, kalman_hedge, pair_spread_z, pca_residuals, rolling_beta, rolling_ols,
                      select_pairs, ssd_pairs)
from . import cross_section, streaming

__all__ = [n for n in dir() if not n.startswith("_")]
