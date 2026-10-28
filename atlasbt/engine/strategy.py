"""Strategy base class and data requirements.

Example::

    class S001(Strategy):
        id = "S001"; bar_freq = "1m"; sizing = "DAY_RISK"; asset = "STK/ETF"
        universe = ["SPY"]
        params = {"or_minutes": 5}
        grid = {"or_minutes": [5, 15, 30]}
        requires = [Requirement("bars", "1m", sessions=("RTH",))]
        deviations = ["RVOL computed vs 20d same-minute median instead of full history"]

        def on_bar(self, ctx): ...

Rules (ARCHITECTURE §9): signals only from completed bars (the Context
enforces it), orders only via ``ctx``, sizing via ``atlasbt.risk``, and every
deviation from the sheet text declared in ``deviations``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional, Sequence

from ..core.types import Fidelity, Fill


@dataclass(frozen=True)
class Requirement:
    """A data dependency. ``critical`` requirements that are unavailable
    (portal.has False or fidelity NA) make the run NOT_TESTABLE."""

    kind: str                                  # 'bars' | 'quotes' | 'options' | 'actions' | 'borrow' | 'events' | ...
    freq: Optional[str] = None
    sessions: tuple[str, ...] = ("RTH",)
    symbols: Optional[tuple[str, ...]] = None  # None -> strategy universe
    critical: bool = True
    min_fidelity: Fidelity = Fidelity.T3_MODELED
    note: str = ""


#: engine-enforced session rules implied by sizing profiles (sheet "Sizing Profiles")
PROFILE_SESSION_RULES = {
    "DAY_RISK": {"no_new_entries_after": "15:15", "flat_by": "15:55"},
    "DAY_NOTIONAL": {"no_new_entries_after": None, "flat_by": "15:55"},
}


class Strategy:
    id: ClassVar[str] = "UNNAMED"
    name: ClassVar[str] = ""
    bar_freq: ClassVar[str] = "1d"
    sizing: ClassVar[Optional[str]] = None
    asset: ClassVar[str] = "STK"
    params: dict = {}
    grid: ClassVar[dict] = {}
    requires: ClassVar[Sequence[Requirement]] = ()
    deviations: ClassVar[Sequence[str]] = ()
    universe: ClassVar[Sequence[str]] = ()
    #: universe name resolved through portal.universe(name, as_of) each session (optional)
    universe_name: ClassVar[Optional[str]] = None
    #: engine-enforced intraday rules (ET "HH:MM"); default from the sizing profile
    no_new_entries_after: ClassVar[Optional[str]] = None
    flat_by: ClassVar[Optional[str]] = None
    #: timeline sessions ('RTH' default; add 'PRE'/'POST' to step through extended hours)
    sessions: ClassVar[tuple[str, ...]] = ("RTH",)

    def __init__(self, **params: Any):
        base = dict(type(self).params)
        unknown = set(params) - set(base)
        if unknown and base:
            raise ValueError(f"{self.id}: unknown params {sorted(unknown)}")
        base.update(params)
        self.params = base

    # --- engine-facing helpers
    def session_rules(self) -> dict:
        rules = dict(PROFILE_SESSION_RULES.get(self.sizing or "", {}))
        if type(self).no_new_entries_after is not None:
            rules["no_new_entries_after"] = type(self).no_new_entries_after
        if type(self).flat_by is not None:
            rules["flat_by"] = type(self).flat_by
        return rules

    def symbols(self, ctx) -> list[str]:
        if self.universe_name:
            return list(ctx.portal().universe(self.universe_name))
        return list(self.universe)

    # --- callbacks (override as needed)
    def init(self, ctx) -> None:
        pass

    def on_session_start(self, ctx) -> None:
        pass

    def on_bar(self, ctx) -> None:
        pass

    def on_fill(self, ctx, fill: Fill) -> None:
        pass

    def on_session_end(self, ctx) -> None:
        pass
