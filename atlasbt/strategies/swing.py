"""Swing family S046-S084 (daily): trend/breakouts S046-S067 and mean reversion S068-S084.

Importing this module registers every class with ``strategies.registry``.
Implementation lives in ``swing_common`` (shared position engine, universes,
point-in-time bar cache), ``swing_trend`` and ``swing_mr``.
"""
from __future__ import annotations

from .swing_common import COMMON_DEVIATIONS, SWING_ETF, BarCache, CrossSectionBase, Plan, SwingBase  # noqa: F401
from .swing_mr import (S068, S069, S070, S071, S072, S073, S074, S075, S076, S077, S078, S079, S080,  # noqa: F401
                       S081, S082, S083, S084)
from .swing_trend import (S046, S047, S048, S049, S050, S051, S052, S053, S054, S055, S056, S057,  # noqa: F401
                          S058, S059, S060, S061, S062, S063, S064, S065, S066, S067)

IDS = [f"S{i:03d}" for i in range(46, 85)]
