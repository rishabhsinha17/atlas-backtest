"""Strategy registry: S001..S172 -> Strategy class.

Family modules register their classes with ``@register``. Modules are imported
lazily so a broken family module cannot break the others.
"""
from __future__ import annotations

import importlib
import itertools
from typing import Iterable

from ..engine.strategy import Strategy

_REGISTRY: dict[str, type[Strategy]] = {}

FAMILY_MODULES = (
    "atlasbt.strategies.intraday",        # S001-S030, S116
    "atlasbt.strategies.calendar_events", # S031-S045
    "atlasbt.strategies.swing",           # S046-S084
    "atlasbt.strategies.factors",         # S085-S100
    "atlasbt.strategies.allocation",      # S089, S090, S101-S105, S117-S122, S124, S125
    "atlasbt.strategies.statarb",         # S106-S115, S123
    "atlasbt.strategies.options_income",  # S126-S157
    "atlasbt.strategies.options_events",  # S158-S172
)

_import_errors: dict[str, str] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    sid = cls.id
    if sid in _REGISTRY and _REGISTRY[sid] is not cls:
        raise ValueError(f"duplicate strategy id {sid}: {_REGISTRY[sid]} vs {cls}")
    _REGISTRY[sid] = cls
    return cls


def load_all() -> dict[str, str]:
    """Import every family module; returns {module: error} for failures."""
    for mod in FAMILY_MODULES:
        if mod in _import_errors:
            continue
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as e:
            if e.name == mod:
                _import_errors[mod] = "not implemented yet"
            else:
                _import_errors[mod] = repr(e)
        except Exception as e:  # pragma: no cover - surfaced in reports
            _import_errors[mod] = repr(e)
    return dict(_import_errors)


def get(sid: str) -> type[Strategy]:
    if sid not in _REGISTRY:
        load_all()
    return _REGISTRY[sid]


def all_ids() -> list[str]:
    load_all()
    return sorted(_REGISTRY)


def expand_grid(grid: dict) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def n_trials(ids: Iterable[str] | None = None) -> int:
    ids = list(ids) if ids is not None else all_ids()
    return sum(len(expand_grid(get(s).grid)) for s in ids)
