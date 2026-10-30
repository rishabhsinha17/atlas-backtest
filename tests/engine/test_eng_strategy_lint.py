"""Static lint: strategy code must not reach around the point-in-time Context.

Scans atlasbt/strategies/*.py and fails on
* private attribute access (``x._name``) on anything other than ``self`` / ``cls`` /
  ``super()`` (e.g. ``ctx._bt``, ``ctx.portal()._p``), and on introspection dunders
  (``__dict__``, ``__closure__``, ``__globals__``, ``__self__`` ...);
* ``getattr``/``setattr``/``hasattr``/``delattr``/``vars`` with a string starting with "_";
* importing ``inspect``, ``gc`` or ``ctypes``; touching ``sys.modules`` or ``sys._getframe``;
* importing private names (``_x``) from atlasbt, or engine internals
  (``atlasbt.engine.backtester/context/market/pit``) that expose the raw portal or data cache.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STRATEGY_DIR = ROOT / "atlasbt" / "strategies"
FORBIDDEN_MODULES = {"inspect", "gc", "ctypes"}
ENGINE_INTERNALS = {"backtester", "context", "market", "pit"}
ESCAPE_DUNDERS = {"__dict__", "__closure__", "__globals__", "__code__", "__self__", "__func__",
                  "__wrapped__", "__getattribute__", "__reduce__", "__reduce_ex__", "__subclasses__",
                  "__builtins__", "__class__"}


def _is_self_like(node: ast.AST) -> bool:
    if isinstance(node, ast.Name) and node.id in ("self", "cls"):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "super":
        return True
    return False


def violations_in(source: str, filename: str = "<src>") -> list[str]:
    tree = ast.parse(source, filename)
    out = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Attribute):
            a = node.attr
            if a in ESCAPE_DUNDERS:
                out.append(f"{filename}:{line}: introspection attribute .{a}")
            elif a.startswith("_") and not (a.startswith("__") and a.endswith("__")) and not _is_self_like(node.value):
                out.append(f"{filename}:{line}: private attribute access .{a}")
            if a in ("modules", "_getframe") and isinstance(node.value, ast.Name) and node.value.id == "sys":
                out.append(f"{filename}:{line}: sys.{a}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
                node.func.id in ("getattr", "setattr", "hasattr", "delattr", "vars"):
            target_self = bool(node.args) and _is_self_like(node.args[0])
            for arg in node.args[1:2]:
                if not target_self and isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("_"):
                    out.append(f"{filename}:{line}: {node.func.id}(..., {arg.value!r})")
            if node.func.id == "vars":
                out.append(f"{filename}:{line}: vars()")
        elif isinstance(node, ast.Import):
            for al in node.names:
                if al.name.split(".")[0] in FORBIDDEN_MODULES:
                    out.append(f"{filename}:{line}: import {al.name}")
                parts = al.name.split(".")
                if len(parts) >= 3 and parts[0] == "atlasbt" and parts[1] == "engine" and parts[2] in ENGINE_INTERNALS:
                    out.append(f"{filename}:{line}: import of engine internals {al.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.split(".")[0] in FORBIDDEN_MODULES:
                out.append(f"{filename}:{line}: from {mod} import ...")
            parts = mod.split(".")
            if "engine" in parts:
                k = parts.index("engine")
                if k + 1 < len(parts) and parts[k + 1] in ENGINE_INTERNALS:
                    out.append(f"{filename}:{line}: import from engine internals {mod}")
                elif k + 1 == len(parts):
                    for al in node.names:
                        if al.name in ENGINE_INTERNALS:
                            out.append(f"{filename}:{line}: import of engine internals {al.name}")
            for al in node.names:
                if al.name.startswith("_") and (mod.startswith("atlasbt") or node.level > 0):
                    out.append(f"{filename}:{line}: private import {al.name} from {mod or '.'}")
                if mod == "sys" and al.name in ("modules", "_getframe"):
                    out.append(f"{filename}:{line}: from sys import {al.name}")
    return out


def test_lint_detects_escape_hatches():
    bad = '''
import inspect
from atlasbt.engine.context import engine_of
def on_bar(self, ctx):
    raw = ctx.portal()._p
    bt = ctx._bt
    x = getattr(ctx, "_EngineContext__bt")
    import sys; m = sys.modules
    f = ctx.history.__self__
'''
    v = violations_in(bad)
    joined = "\n".join(v)
    for needle in ("import inspect", "engine internals", "._p", "._bt", "_EngineContext__bt", "sys.modules", "__self__"):
        assert needle in joined, (needle, v)
    ok = '''
class S:
    def on_bar(self, ctx):
        self._cache = {}
        super()._evaluate(ctx)
        n = type(self).__name__
        d = getattr(self, "_day", {})
        return ctx.history("X", 5)
'''
    assert violations_in(ok) == []


@pytest.mark.parametrize("path", sorted(STRATEGY_DIR.glob("*.py")), ids=lambda p: p.name)
def test_strategies_do_not_reach_around_the_context(path):
    v = violations_in(path.read_text(), path.name)
    assert not v, "\n".join(v)
