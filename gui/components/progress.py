"""Shared builder for the per-tab progress bars.

Must not import ``gui.app`` (circular import).
"""
import time

from nicegui import ui


class Progress:
    """A bar, a label and an estimate, driven by ``(done, total, message)``.

    The estimate is deliberately not shown until a second item has finished:
    one sample gives a number that swings wildly and reads as authoritative.
    """

    def __init__(self, bar, label, *, show_eta=True):
        self._bar = bar
        self._label = label
        self._show_eta = show_eta
        self._t0 = None
        self._first_done = None
        self.hide()

    def start(self, total, message=""):
        self._t0 = time.time()
        self._first_done = None
        self._bar.value = 0.0
        self._bar.set_visibility(True)
        self._label.set_visibility(True)
        self._label.set_text(message or f"Starting — {total} to do…")

    def hide(self):
        self._bar.set_visibility(False)
        self._label.set_visibility(False)

    def finish(self, message=""):
        self._bar.value = 1.0
        if message:
            self._label.set_text(message)
        else:
            self.hide()

    def update(self, done, total, message=""):
        total = max(1, int(total))
        done = max(0, min(int(done), total))
        self._bar.value = done / total
        self._label.set_text(
            f"{done} of {total}" + (f" — {message}" if message else "")
            + self._eta(done, total))

    def _eta(self, done, total):
        """' — about 12 min left', once there is enough to say so."""
        if not self._show_eta or self._t0 is None or done <= 0:
            return ""
        if self._first_done is None:
            self._first_done = time.time()
            return ""                      # one sample is not an estimate
        if done >= total:
            return ""
        per = (time.time() - self._t0) / done
        left = per * (total - done)
        if left < 90:
            return f" — about {left:.0f} s left"
        if left < 5400:
            return f" — about {left / 60:.0f} min left"
        return f" — about {left / 3600:.1f} h left"


def build_progress(*, classes="w-full mt-2", show_eta=True):
    """Create a progress bar + label and return a :class:`Progress`.

    Hidden until ``start()``. ``update(done, total, message)`` matches the
    ``progress_fn`` signature used in ``functions/``, so a worker can be handed
    ``progress.update`` directly.
    """
    bar = ui.linear_progress(value=0.0, show_value=False).classes(classes)
    label = ui.label("").classes("text-sm text-grey-7")
    return Progress(bar, label, show_eta=show_eta)
