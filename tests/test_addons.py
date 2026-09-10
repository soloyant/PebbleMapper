"""Tests for the addon sandbox (functions/addons.py).

Covers:
  - Sandbox block tests: disallowed AST node types must be rejected by validate()
  - Allowed expression correctness: safe functions + arithmetic evaluate correctly
  - DoS guard: pow() exponent > 64 raises ValueError at evaluate time
  - NaN propagation through apply_addons_to_dataframe
  - load_addons contract: invalid JSON → empty list, per-project override
"""
import math
import numpy as np
import pandas as pd
import json

import pytest

from functions.addons import Addon, apply_addons_to_dataframe, load_addons


# ---------------------------------------------------------------------------
# Sandbox: blocked expressions
# ---------------------------------------------------------------------------

BLOCKED = [
    "__import__('os')",
    "[x for x in range(10)]",
    "(x for x in range(10))",
    "{x: x for x in range(10)}",
    "lambda x: x",
    "a.__class__",
    "type(a)",
    "__builtins__",
    "{1: 2}",
    "{1, 2}",
]


@pytest.mark.parametrize("expr", BLOCKED)
def test_blocked_expressions(expr):
    ad = Addon(name="t", inputs=["a"], expression=expr)
    assert ad.validate() is False, f"Should have been blocked: {expr!r}"


# ---------------------------------------------------------------------------
# Allowed expressions: correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expr,env,expected", [
    ("sqrt(a)",           {"a": 4.0},           2.0),
    ("a + b * 2",         {"a": 1.0, "b": 3.0}, 7.0),
    ("a if a > 0 else b", {"a": -1.0, "b": 5.0}, 5.0),
    ("max(a, b)",         {"a": 3.0, "b": 7.0}, 7.0),
    ("abs(-a)",           {"a": 4.0},            4.0),
    ("round(a, 2)",       {"a": 3.14159},        3.14),
    ("a ** 2",            {"a": 3.0},            9.0),
    ("log(a) / log(2)",   {"a": 8.0},            3.0),
    ("(a + b) / 2",       {"a": 10.0, "b": 6.0}, 8.0),
    ("min(a, b, 0.0)",    {"a": 1.0, "b": 2.0}, 0.0),
])
def test_allowed_expressions(expr, env, expected):
    ad = Addon(name="t", inputs=list(env.keys()), expression=expr)
    assert ad.validate() is True, f"Should have been allowed: {expr!r}"
    result = ad.evaluate(env)
    assert abs(result - expected) < 1e-9, f"{expr!r}: got {result}, expected {expected}"


# ---------------------------------------------------------------------------
# DoS guard
# ---------------------------------------------------------------------------

def test_pow_dos_guard_65():
    """Exponent 65 > 64 must raise ValueError at evaluate time."""
    ad = Addon(name="dos", inputs=[], expression="pow(2, 65)")
    assert ad.validate() is True   # syntactically valid
    with pytest.raises(ValueError, match="exceeds the safe limit"):
        ad.evaluate({})


def test_pow_dos_guard_nested():
    """pow(2, pow(2, 30)): inner returns 1073741824 > 64 → blocked."""
    ad = Addon(name="dos2", inputs=[], expression="pow(2, pow(2, 30))")
    assert ad.validate() is True
    with pytest.raises(ValueError, match="exceeds the safe limit"):
        ad.evaluate({})


def test_pow_within_limit():
    """pow(2, 64) must still work (2^64 = 18446744073709551616)."""
    ad = Addon(name="ok", inputs=[], expression="pow(2, 64)")
    assert ad.validate() is True
    result = ad.evaluate({})
    assert result == 2**64


def test_pow_small_exponent_fine():
    """pow(2, 4) = 16, well within limit."""
    ad = Addon(name="ok2", inputs=[], expression="pow(2, 4)")
    assert ad.validate() is True
    assert ad.evaluate({}) == 16


# ---------------------------------------------------------------------------
# NaN propagation through apply_addons_to_dataframe
# ---------------------------------------------------------------------------

def test_division_by_zero_becomes_nan():
    ad = Addon(name="ratio", inputs=["a", "b"], expression="a / b")
    ad.validate()
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [0.0, 2.0]})
    result = apply_addons_to_dataframe(df, [ad])
    assert np.isnan(result["ratio"].iloc[0])
    assert abs(result["ratio"].iloc[1] - 1.0) < 1e-12


def test_missing_column_skips_addon():
    """If a required input column is absent, the addon is skipped entirely
    (the output column is not added to the DataFrame at all)."""
    ad = Addon(name="t", inputs=["missing_col"], expression="missing_col * 2")
    ad.validate()
    df = pd.DataFrame({"other": [1.0, 2.0]})
    result = apply_addons_to_dataframe(df, [ad])
    # The addon column must be absent, not NaN-filled.
    assert "t" not in result.columns


# ---------------------------------------------------------------------------
# load_addons contract
# ---------------------------------------------------------------------------

def test_load_addons_invalid_json(tmp_path):
    p = tmp_path / "addons.json"
    p.write_text("not valid json", encoding="utf-8")
    result = load_addons(str(p))
    assert result == []


def test_load_addons_nonexistent_path(tmp_path):
    result = load_addons(str(tmp_path / "does_not_exist.json"))
    assert result == []


def test_load_addons_per_project_overrides_repo(tmp_path):
    """Later path (per-project) overrides same-named addon from earlier (repo-level)."""
    repo = tmp_path / "repo.json"
    proj = tmp_path / "proj.json"
    repo.write_text(
        '[{"name":"X","inputs":[],"expression":"1.0","units":""}]',
        encoding="utf-8",
    )
    proj.write_text(
        '[{"name":"X","inputs":[],"expression":"2.0","units":""}]',
        encoding="utf-8",
    )
    result = load_addons(str(repo), str(proj))
    assert len(result) == 1
    assert result[0].evaluate({}) == 2.0


def test_load_addons_invalid_expression_skipped(tmp_path):
    """An addon whose expression fails validate() is silently excluded."""
    p = tmp_path / "addons.json"
    p.write_text(
        '[{"name":"bad","inputs":[],"expression":"lambda x:x","units":""},'
        ' {"name":"ok","inputs":[],"expression":"1.0","units":""}]',
        encoding="utf-8",
    )
    result = load_addons(str(p))
    names = [a.name for a in result]
    assert "bad" not in names
    assert "ok" in names


def test_loading_an_addon_registers_its_display_unit(tmp_path):
    from functions import units
    from functions.addons import load_addons
    f = tmp_path / "addons.json"
    f.write_text(json.dumps([
        {"name": "My_thrust", "inputs": ["Equivalent_diameter"],
         "expression": "9.81 * Equivalent_diameter", "units": "N/m^2", "factor": 1.0},
        {"name": "My_ratio", "inputs": ["Clast_width", "Clast_length"],
         "expression": "Clast_width / Clast_length", "units": "dimensionless"},
        {"name": "Half_length", "inputs": ["Clast_length"],
         "expression": "Clast_length / 2", "units": "mm", "factor": 1000.0},
    ]), encoding="utf-8")
    assert len(load_addons(f)) == 3
    assert units.field_unit_and_factor("My_thrust") == ("N/m^2", 1.0)
    assert units.field_unit_and_factor("my_ratio") == ("", 1.0)
    assert units.field_unit_and_factor("Half_length") == ("mm", 1000.0)
    # a built-in field is not shadowed by the registry
    assert units.field_unit_and_factor("Clast_length") == ("mm", 1000.0)
