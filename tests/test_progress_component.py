"""The shared progress bar.

`gui/components/progress.py`. Every tab that showed progress had inlined its
own bar, and the copies diverged -- two different ways of hiding it, and an ETA
that existed only in Detection because it was parsed out of that tab's own log
lines. The Georeference survey check had none at all: a three-hour run over
twenty quadrats showed a spinner and nothing else.

The widget itself needs a browser; the arithmetic does not, and the arithmetic
is where a progress bar lies to you.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("nicegui")

from gui.components.progress import Progress  # noqa: E402


class FakeBar:
    def __init__(self):
        self.value = 0.0
        self.visible = None

    def set_visibility(self, v):
        self.visible = v


class FakeLabel(FakeBar):
    def __init__(self):
        super().__init__()
        self.text = ""

    def set_text(self, t):
        self.text = t


def make(show_eta=True):
    bar, label = FakeBar(), FakeLabel()
    return Progress(bar, label, show_eta=show_eta), bar, label


def test_it_starts_hidden():
    _, bar, label = make()
    assert bar.visible is False and label.visible is False


def test_start_shows_it_and_zeroes_it():
    p, bar, label = make()
    p.start(20, "Checking 20 quadrat(s)…")
    assert bar.visible is True and label.visible is True
    assert bar.value == 0.0
    assert "20" in label.text


def test_the_fraction_tracks_the_count():
    p, bar, _ = make()
    p.start(20)
    for done, expect in ((0, 0.0), (5, 0.25), (10, 0.5), (20, 1.0)):
        p.update(done, 20)
        assert bar.value == pytest.approx(expect)


def test_the_label_names_the_current_file():
    """'12 of 20' alone does not tell you whether it is stuck."""
    p, _, label = make()
    p.start(20)
    p.update(12, 20, "located DJI_0912.png")
    assert "12 of 20" in label.text
    assert "DJI_0912.png" in label.text


def test_a_count_past_the_total_does_not_overflow_the_bar():
    p, bar, _ = make()
    p.start(5)
    p.update(9, 5)
    assert bar.value == 1.0


def test_a_zero_total_does_not_divide_by_zero():
    p, bar, _ = make()
    p.start(0)
    p.update(0, 0)
    assert 0.0 <= bar.value <= 1.0


# --------------------------------------------------------------------------- #
#  The estimate                                                                #
# --------------------------------------------------------------------------- #
def test_no_estimate_is_offered_from_a_single_sample():
    """One finished item gives a number that swings wildly and reads as
    authoritative. Better to say nothing."""
    p, _, label = make()
    p.start(20)
    p.update(1, 20, "one done")
    assert "left" not in label.text


def test_an_estimate_appears_once_there_is_something_to_estimate_from():
    p, _, label = make()
    p.start(20)
    p.update(1, 20)
    time.sleep(0.05)
    p.update(2, 20)
    assert "left" in label.text


def test_no_estimate_when_it_is_finished():
    p, _, label = make()
    p.start(4)
    p.update(1, 4)
    p.update(4, 4)
    assert "left" not in label.text


def test_the_estimate_can_be_turned_off():
    p, _, label = make()
    p2, _, label2 = make(show_eta=False)
    for prog, lab in ((p, label), (p2, label2)):
        prog.start(10)
        prog.update(1, 10)
        time.sleep(0.02)
        prog.update(2, 10)
    assert "left" in label.text
    assert "left" not in label2.text


def test_finish_with_a_message_keeps_it_on_screen():
    """A survey that stopped early has to say so; hiding the bar would erase
    the only statement that it did not finish."""
    p, bar, label = make()
    p.start(20)
    p.finish("12 of 20 checked — stopped early")
    assert bar.value == 1.0
    assert label.visible is not False
    assert "stopped early" in label.text


def test_finish_with_no_message_puts_it_away():
    p, bar, label = make()
    p.start(20)
    p.finish()
    assert bar.visible is False and label.visible is False
