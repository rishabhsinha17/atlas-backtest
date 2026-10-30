"""Options events / skew / short-dated strategies S158-S172 (Trading Strategy Atlas).

Implementation split across helper modules (all registered on import):
  * ``options_events_common``   - shared conventions & machinery (read its docstring first)
  * ``options_events_earnings`` - S158, S159, S160
  * ``options_events_0dte``     - S161, S162, S163, S164
  * ``options_events_vol``      - S165, S166, S167, S168, S169, S170
  * ``options_events_arb``      - S171, S172

Every strategy is implemented fully against the DataPortal protocol so that it
runs at the best fidelity the data allows (T0/T1 with a keyed intraday option
quote provider such as Databento OPRA).  Critical requirements make the run
NOT_TESTABLE when the needed real data is absent (docs/OPTIONS.md §13: S158,
S159, S161-S164, S171, S172 need intraday or SPX quotes that the free data does
not have).  ``allow_synthetic=True`` (opt-in, never default) runs the mechanics
on synthetic / low-fidelity chains, flagged T3_MODELED.
"""
from __future__ import annotations

from .options_events_0dte import S161, S162, S163, S164
from .options_events_arb import S171, S172
from .options_events_earnings import S158, S159, S160
from .options_events_vol import S165, S166, S167, S168, S169, S170

__all__ = ["S158", "S159", "S160", "S161", "S162", "S163", "S164", "S165", "S166", "S167", "S168", "S169",
           "S170", "S171", "S172"]
