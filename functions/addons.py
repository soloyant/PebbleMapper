"""User-defined per-clast equations loaded from JSON.

Each addon declares ``name`` (output column), ``inputs`` (columns read),
``expression`` (evaluated per row), ``units``, ``description`` and an
optional display ``factor``. Per-project ``addons.json`` overrides the
repo-level ``user_addons.json``.

Expressions are parsed with ``ast.parse`` and evaluated by a whitelist walker
(no exec/eval, no attribute access, only a small approved function table), so
a malicious project file cannot execute arbitrary code.
"""
from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from functions._logging import get_logger

_log = get_logger(__name__)


# Whitelisted AST node types; ``Call`` is allowed only against _SAFE_FUNCTIONS.
_ALLOWED_NODE_TYPES: tuple = (
    ast.Expression,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.IfExp,
    ast.Name, ast.Load,
    ast.Constant,
    ast.Num, ast.Str,        # tolerated on older parsers
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
    ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
    ast.And, ast.Or, ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Call,
    ast.Tuple, ast.List,
)

_SAFE_FUNCTIONS: dict[str, Callable[..., Any]] = {}


def _register_safe_functions() -> None:
    """Populate the safe-function table lazily."""
    if _SAFE_FUNCTIONS:
        return
    for fname in (
        "sqrt", "log", "log2", "log10", "exp",
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh",
        "floor", "ceil", "fabs",
    ):
        if hasattr(math, fname):
            _SAFE_FUNCTIONS[fname] = getattr(math, fname)
    _SAFE_FUNCTIONS["abs"] = abs
    _SAFE_FUNCTIONS["min"] = min
    _SAFE_FUNCTIONS["max"] = max
    _SAFE_FUNCTIONS["round"] = round
    # Exponent cap: arbitrary-precision pow(2, pow(2, 30)) allocates ~128 MB.
    import builtins as _builtins
    def _safe_pow(base, exp):
        try:
            if float(exp) > 64:
                raise ValueError(
                    f"pow() exponent {exp!r} exceeds the safe limit of 64. "
                    "Pre-compute the constant or use a smaller expression.")
        except (TypeError, ValueError) as exc:
            if "exceeds" in str(exc):
                raise
            raise ValueError(
                f"pow() exponent must be numeric, got {type(exp).__name__}")
        return _builtins.pow(base, exp)
    _SAFE_FUNCTIONS["pow"] = _safe_pow


@dataclass
class Addon:
    """One registered custom-equation field."""
    name: str
    inputs: list[str] = field(default_factory=list)
    expression: str = ""
    units: str = ""
    description: str = ""
    factor: float = 1.0
    _ast: Optional[ast.Expression] = None
    _error: str = ""

    def validate(self) -> bool:
        """Parse and walk the expression. Sets ``self._ast`` on
        success, ``self._error`` on failure. Returns True iff valid."""
        try:
            tree = ast.parse(self.expression, mode="eval")
        except SyntaxError as ex:
            self._error = f"syntax error: {ex}"
            return False
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODE_TYPES):
                self._error = (
                    f"disallowed expression element "
                    f"{type(node).__name__!r}; only arithmetic, "
                    f"comparison, ternary, and a small approved "
                    f"function set are permitted.")
                return False
            if isinstance(node, ast.Name):
                if node.id.startswith("__") and node.id.endswith("__"):
                    self._error = (
                        f"dunder name {node.id!r} is not permitted "
                        "in addon expressions.")
                    return False
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    self._error = (
                        "function calls must be simple names "
                        "(e.g. sqrt(D), not module.fn(D)).")
                    return False
                _register_safe_functions()
                if node.func.id not in _SAFE_FUNCTIONS:
                    self._error = (
                        f"function {node.func.id!r} is not in the "
                        f"approved list: "
                        f"{sorted(_SAFE_FUNCTIONS)}")
                    return False
        self._ast = tree
        self._error = ""
        return True

    def evaluate(self, row: dict) -> float:
        """Evaluate the expression with ``row`` (dict or Series) providing
        input values. Unknown names raise NameError; Inf/NaN propagate."""
        if self._ast is None:
            if not self.validate():
                raise ValueError(
                    f"addon {self.name!r}: {self._error}")
        _register_safe_functions()
        # Cast row values to float where possible so numpy scalars play
        # well with math.* functions.
        env: dict[str, Any] = dict(_SAFE_FUNCTIONS)
        for k in self.inputs:
            if k in row:
                try:
                    env[k] = float(row[k])
                except (TypeError, ValueError):
                    env[k] = row[k]
        return self._evaluate_node(self._ast.body, env)

    def _evaluate_node(self, node: ast.AST, env: dict) -> Any:
        """Walk the validated AST; env is the only namespace."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Num):           # legacy parsers
            return node.n
        if isinstance(node, ast.Str):
            return node.s
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise NameError(
                    f"unknown name {node.id!r} in expression. "
                    f"Did you list it in 'inputs'?")
            return env[node.id]
        if isinstance(node, ast.BinOp):
            l = self._evaluate_node(node.left, env)
            r = self._evaluate_node(node.right, env)
            op = node.op
            if isinstance(op, ast.Add):       return l + r
            if isinstance(op, ast.Sub):       return l - r
            if isinstance(op, ast.Mult):      return l * r
            if isinstance(op, ast.Div):       return l / r
            if isinstance(op, ast.FloorDiv):  return l // r
            if isinstance(op, ast.Mod):       return l % r
            if isinstance(op, ast.Pow):       return l ** r
            raise ValueError(f"unsupported operator: {type(op).__name__}")
        if isinstance(node, ast.UnaryOp):
            v = self._evaluate_node(node.operand, env)
            if isinstance(node.op, ast.USub): return -v
            if isinstance(node.op, ast.UAdd): return +v
            if isinstance(node.op, ast.Not):  return not v
            raise ValueError(
                f"unsupported unary op: {type(node.op).__name__}")
        if isinstance(node, ast.BoolOp):
            vals = [self._evaluate_node(v, env) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(vals)
            if isinstance(node.op, ast.Or):
                return any(vals)
            raise ValueError(
                f"unsupported bool op: {type(node.op).__name__}")
        if isinstance(node, ast.Compare):
            left = self._evaluate_node(node.left, env)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._evaluate_node(comparator, env)
                if isinstance(op, ast.Eq):    ok = (left == right)
                elif isinstance(op, ast.NotEq): ok = (left != right)
                elif isinstance(op, ast.Lt):    ok = (left < right)
                elif isinstance(op, ast.LtE):   ok = (left <= right)
                elif isinstance(op, ast.Gt):    ok = (left > right)
                elif isinstance(op, ast.GtE):   ok = (left >= right)
                else:
                    raise ValueError(
                        f"unsupported comparison: "
                        f"{type(op).__name__}")
                if not ok:
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            cond = self._evaluate_node(node.test, env)
            if cond:
                return self._evaluate_node(node.body, env)
            return self._evaluate_node(node.orelse, env)
        if isinstance(node, ast.Call):
            fn = env.get(node.func.id)
            if fn is None:
                raise NameError(
                    f"function {node.func.id!r} not available.")
            args = [self._evaluate_node(a, env) for a in node.args]
            return fn(*args)
        if isinstance(node, (ast.Tuple, ast.List)):
            return [self._evaluate_node(el, env) for el in node.elts]
        raise ValueError(
            f"unsupported node {type(node).__name__}")

    def to_dict(self) -> dict:
        """Serialise back to a JSON-friendly dict."""
        return {
            "name": self.name,
            "inputs": list(self.inputs),
            "expression": self.expression,
            "units": self.units,
            "description": self.description,
            "factor": self.factor,
        }


def load_addons(*paths: str | Path) -> list[Addon]:
    """Load and validate addons from JSON files; invalid entries are skipped
    with a warning. De-duplicated by ``name``, later paths override."""
    merged: dict[str, Addon] = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as ex:
            _log.warning("Could not read %s: %s. Skipped.", p, ex)
            continue
        if not isinstance(data, list):
            data = [data]
        for entry in data:
            if not isinstance(entry, dict) or "name" not in entry:
                continue
            ad = Addon(
                name=str(entry["name"]),
                inputs=[str(x) for x in entry.get("inputs", [])],
                expression=str(entry.get("expression", "")),
                units=str(entry.get("units", "")),
                description=str(entry.get("description", "")),
                factor=float(entry.get("factor", 1.0) or 1.0),
            )
            if ad.validate():
                merged[ad.name] = ad
                try:
                    from functions.units import register_field_unit
                    register_field_unit(ad.name, ad.units, ad.factor)
                except Exception:
                    pass
            else:
                _log.warning(
                    "Addon %r in %s failed validation: %s. Skipped.",
                    ad.name, p, ad._error)
    return list(merged.values())


def apply_addons_to_dataframe(df, addons: list[Addon]):
    """Add one column per addon, computed row-wise (mutates and returns
    ``df``). A row where the addon throws gets NaN."""
    if not addons:
        return df
    for ad in addons:
        if not ad.validate():
            continue
        missing = [x for x in ad.inputs if x not in df.columns]
        if missing:
            _log.warning(
                "Addon %r: missing input column(s) %s. Skipped.",
                ad.name, missing)
            continue
        vals = []
        for _, row in df.iterrows():
            try:
                vals.append(float(ad.evaluate(row.to_dict())))
            except Exception:
                vals.append(float("nan"))
        df[ad.name] = vals
    return df


__all__ = [
    "Addon",
    "load_addons",
    "apply_addons_to_dataframe",
]
