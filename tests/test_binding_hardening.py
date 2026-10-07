"""The NiceGUI binding-propagation hardening.

``_propagate`` resets the ``propagation_visited``
cycle guard on reentrant calls from ``BindableProperty.__set__``, so a
two-way-bound value with ``x != x`` (NaN) recurses to native stack death.
This test reproduces the recursion at a safe Python recursion limit, then
proves ``crashsafe.harden_binding_propagation()`` terminates the same cycle
and names it once.
"""
import math
import sys

import pytest

from nicegui import binding

import functions.crashsafe as crashsafe


class _Box:
    value = binding.BindableProperty()

    def __init__(self):
        self.value = 0.0


@pytest.fixture()
def cycle_pair():
    a, b = _Box(), _Box()
    binding.bind_to(a, "value", b, "value")
    binding.bind_to(b, "value", a, "value")
    yield a, b
    binding.remove([a, b])


def test_nan_in_a_two_way_binding_recursed_before_the_fix(cycle_pair):
    a, _b = cycle_pair
    if getattr(binding, "_pm_hardened", False):
        pytest.skip("hardening already installed in this process; the "
                    "unpatched behaviour is proven by the fresh-process run")
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(220)   # fail as RecursionError, not a C-stack death
    try:
        with pytest.raises(RecursionError):
            a.value = float("nan")
    finally:
        sys.setrecursionlimit(limit)
        # Leave the pair in a sane state for teardown.
        object.__setattr__(a, "___value", 0.0)
        object.__setattr__(_b, "___value", 0.0)


def test_hardening_terminates_the_cycle_and_names_it(cycle_pair, capsys):
    a, b = cycle_pair
    assert crashsafe.harden_binding_propagation() is True
    crashsafe._cycle_warned.clear()
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(220)
    try:
        a.value = float("nan")           # must terminate, not recurse
    finally:
        sys.setrecursionlimit(limit)
    assert math.isnan(a.value) and math.isnan(b.value)  # value still propagated
    out = capsys.readouterr().out
    assert "binding cycle dampened at _Box.value" in out
    # Once per (type, attr) — a second poisoned write stays quiet.
    a.value = float("nan")
    assert "dampened" not in capsys.readouterr().out


def test_ordinary_values_still_propagate_after_hardening(cycle_pair):
    a, b = cycle_pair
    assert crashsafe.harden_binding_propagation() is True
    a.value = 42.5
    assert b.value == 42.5
    b.value = 7.0
    assert a.value == 7.0
