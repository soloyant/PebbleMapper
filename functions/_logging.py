"""Shared logging infrastructure for the csm package.

Usage in each module::

    from functions._logging import get_logger
    _log = get_logger(__name__)   # → csm.clasts_detection, etc.

Three sinks are provided:

* **stderr** (WARNING+, set up automatically at import time) — visible in
  the terminal during development and in CI.
* **file handler** (DEBUG+, job-scoped) — attach with
  :func:`attach_file_handler` when a long-running job starts and detach
  with :func:`detach_file_handler` when it ends.  Each run gets its own
  log file under ``output_results/logs/``.
* **NiceGUI widget** (INFO+, GUI-scoped) — attach with
  :func:`attach_gui_handler` and pass a :class:`NiceGUILogHandler`
  wrapping the GUI log widget.

Backward compatibility: modules that still have explicit ``print()`` calls
continue to work — the logging layer is additive and non-breaking.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT_LOGGER_NAME = "csm"


def get_logger(module_name: str) -> logging.Logger:
    """Return the csm child logger for *module_name*.

    The full Python module name is shortened to the csm hierarchy::

        "functions.clasts_detection"  →  logging.getLogger("csm.clasts_detection")
        "gui.app"                     →  logging.getLogger("csm.app")
    """
    short = module_name.replace("functions.", "").replace("gui.", "")
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{short}")


def _ensure_stderr_handler() -> None:
    """Add a WARNING-level stderr handler to the root csm logger if none exists.

    Called once at import time so any module that does
    ``from functions._logging import get_logger`` automatically gets a
    stderr sink for WARNING+ messages without any further configuration.
    """
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    if not root.handlers:
        root.setLevel(logging.DEBUG)
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.WARNING)
        sh.setFormatter(
            logging.Formatter("%(levelname)-8s %(name)s — %(message)s")
        )
        root.addHandler(sh)


# Initialise immediately so the stderr sink is always in place.
_ensure_stderr_handler()


# ---------------------------------------------------------------------------
# File handler (job-scoped)
# ---------------------------------------------------------------------------

def attach_file_handler(log_path: Path) -> logging.FileHandler:
    """Add a DEBUG-level :class:`logging.FileHandler` to the root csm logger.

    The parent directory is created if it does not exist.  Returns the
    handler so the caller can remove it when the job finishes::

        fh = attach_file_handler(log_path)
        try:
            run_job(...)
        finally:
            detach_file_handler(fh)
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logging.getLogger(_ROOT_LOGGER_NAME).addHandler(fh)
    return fh


def detach_file_handler(fh: logging.FileHandler) -> None:
    """Remove and close a previously-attached file handler."""
    logging.getLogger(_ROOT_LOGGER_NAME).removeHandler(fh)
    fh.close()


# ---------------------------------------------------------------------------
# NiceGUI widget handler (GUI-scoped)
# ---------------------------------------------------------------------------

class NiceGUILogHandler(logging.Handler):
    """Forward INFO+ log records to a NiceGUI log element.

    Usage::

        handler = NiceGUILogHandler(log_element)
        attach_gui_handler(handler)
        try:
            run_job(...)
        finally:
            detach_gui_handler(handler)

    NiceGUI's ``ui.log`` widget's ``push()`` method is thread-safe in
    NiceGUI ≥ 1.x, so no additional synchronisation is needed.
    """

    def __init__(self, widget) -> None:
        super().__init__()
        self._widget = widget
        self.setLevel(logging.INFO)
        self.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            self._widget.push(line)
        except Exception:
            self.handleError(record)


def attach_gui_handler(handler: NiceGUILogHandler) -> None:
    """Add *handler* to the root csm logger."""
    logging.getLogger(_ROOT_LOGGER_NAME).addHandler(handler)


def detach_gui_handler(handler: NiceGUILogHandler) -> None:
    """Remove *handler* from the root csm logger and close it."""
    logging.getLogger(_ROOT_LOGGER_NAME).removeHandler(handler)
    handler.close()


__all__ = [
    "get_logger",
    "attach_file_handler",
    "detach_file_handler",
    "NiceGUILogHandler",
    "attach_gui_handler",
    "detach_gui_handler",
]
