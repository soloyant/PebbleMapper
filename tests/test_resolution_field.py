"""Digitize's Resolution field across photographs."""
from __future__ import annotations

from functions.gsd import resolution_after_open


def test_a_photograph_with_a_gsd_fills_the_field_as_automatic():
    assert resolution_after_open(0.000567, 0.001, 0.001, False) == (0.000567, 0.000567, False)


def test_a_plain_photograph_empties_an_automatic_value():
    # the boot default …
    assert resolution_after_open(None, 0.001, 0.001, False) == (None, None, False)
    # … and another photograph's GSD
    assert resolution_after_open(None, 0.000567, 0.000567, False) == (None, None, False)


def test_a_plain_photograph_keeps_a_typed_value():
    assert resolution_after_open(None, 0.0007, None, True) == (0.0007, None, True)


def test_a_gsd_replaces_a_typed_value_and_it_is_automatic_again():
    assert resolution_after_open(0.001, 0.0007, None, True) == (0.001, 0.001, False)
