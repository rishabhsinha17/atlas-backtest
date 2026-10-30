"""Intraday family (sheet families "Intraday breakout / momentum",
"Intraday fades / reversals" and S116 "Broad-market ETF opening gap fade").

S001-S020: :mod:`atlasbt.strategies.intraday_breakout`
S021-S030, S116: :mod:`atlasbt.strategies.intraday_fade`
Shared machinery (tapes, streaming indicators across sessions, time-of-day
profiles, sizing/order helpers, module-wide deviations):
:mod:`atlasbt.strategies.intraday_base`.
"""
from __future__ import annotations

from . import intraday_breakout, intraday_fade  # noqa: F401  (registers the classes)
from .intraday_base import ETF_CORE, MEGACAP_UNIVERSE, MODULE_DEVIATIONS, IntradayStrategy  # noqa: F401

IDS = tuple([f"S{i:03d}" for i in range(1, 31)] + ["S116"])
