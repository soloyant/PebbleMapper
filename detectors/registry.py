"""Registry of detection backends + availability discovery.

The GUI lists ``available_backends()`` (those whose weights / environment are
actually installed) and defaults to Mask R-CNN. Nothing else in the app needs
to know a backend's name.

Bringing your own model
-----------------------
Mask R-CNN is the model this tool was built for and the only one shipped, but
the detection interface is deliberately model-agnostic: implement
:class:`detectors.base.DetectorBackend` and PebbleMapper will drive your model
exactly as it drives Mask R-CNN. You do not need to modify this file.

Declare your backend in a JSON file, looked up in this order:

1. the path in the ``PEBBLEMAPPER_DETECTORS`` environment variable
2. ``<repo>/user_detectors.json``

Each entry names an importable module and a zero-argument factory returning a
``DetectorBackend``; an optional ``path`` is a directory put on ``sys.path``
before the import (relative to the JSON file's folder), so the package need
not be installed or reachable through ``PYTHONPATH``::

    [
      {"module": "my_package.my_backend", "factory": "make_backend",
       "path": "../my_backends"}
    ]

A backend that fails to import, build or validate is skipped with a warning —
a third-party model can never stop PebbleMapper starting. One that registers
but is not available here (``is_available()`` false: weights or environment
missing) is kept out of the selector and its ``install_hint`` is logged once
at startup. Your model runs in whatever Python environment suits it: see
``detectors/stub_backend.py`` and ``detectors/subprocess_runner.py`` for the
cross-environment pattern, which is how a backend with dependencies that
conflict with this env should be wired.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

from detectors.base import DetectorBackend
from detectors.maskrcnn import MaskRCNNBackend
from detectors.stub_backend import StubBackend

_log = logging.getLogger("csm.detectors")

_DEFAULT = "maskrcnn"

# Constructing a backend loads no model; the stub stays hidden in the GUI
# unless its dedicated conda env exists.
_BACKENDS = {b.info.name: b for b in (MaskRCNNBackend(), StubBackend())}


def all_backends() -> List[DetectorBackend]:
    """Every registered backend, regardless of availability."""
    return list(_BACKENDS.values())


def available_backends() -> List[DetectorBackend]:
    """Only backends that can actually run here (weights / env present)."""
    out = []
    for b in _BACKENDS.values():
        try:
            if b.is_available():
                out.append(b)
        except Exception:
            pass
    return out


def get_backend(name: str) -> Optional[DetectorBackend]:
    return _BACKENDS.get(name)


DEV_BACKENDS_ENV = "PEBBLEMAPPER_SHOW_DEV_BACKENDS"


def show_dev_backends() -> bool:
    """True when ``PEBBLEMAPPER_SHOW_DEV_BACKENDS=1``: developer/proof
    backends (``info.dev_only``, the stub) are offered as models too."""
    return os.environ.get(DEV_BACKENDS_ENV, "").strip() == "1"


def selector_options() -> dict:
    """``{name: display_name}`` of the models the app offers: the available
    backends minus the ``dev_only`` ones (unless
    ``PEBBLEMAPPER_SHOW_DEV_BACKENDS=1``), the default first. Never empty:
    Mask R-CNN is listed even when its weights are missing, so the choice
    stays valid and the run itself reports the problem."""
    show_dev = show_dev_backends()
    opts = {}
    for b in available_backends():
        if getattr(b.info, "dev_only", False) and not show_dev:
            continue
        opts[b.info.name] = b.info.display_name
    if _DEFAULT not in opts:
        d = _BACKENDS.get(_DEFAULT)
        opts = {_DEFAULT: d.info.display_name if d else "Mask R-CNN", **opts}
    else:
        opts = {_DEFAULT: opts.pop(_DEFAULT), **opts}
    return opts


def normalise_model_name(name) -> str:
    """``name`` when it is one of :func:`selector_options`, else the default."""
    return name if name in selector_options() else _DEFAULT


def default_backend_name() -> str:
    return _DEFAULT


def register(backend: DetectorBackend) -> None:
    """Add a backend to the registry.

    Third-party backends arrive here via :func:`load_user_backends`; call it
    directly when embedding PebbleMapper as a library.
    """
    if not isinstance(backend, DetectorBackend):
        raise TypeError(
            f"backend must implement detectors.base.DetectorBackend, "
            f"got {type(backend).__name__}")
    name = getattr(getattr(backend, "info", None), "name", "")
    if not name:
        raise ValueError("backend.info.name must be a non-empty string")
    _BACKENDS[name] = backend


def user_detectors_file() -> Optional[Path]:
    """The declarations file, or None when the user has not written one."""
    env = os.environ.get("PEBBLEMAPPER_DETECTORS", "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    p = Path(__file__).resolve().parents[1] / "user_detectors.json"
    return p if p.is_file() else None


def load_user_backends(path=None) -> List[str]:
    """Import and register the backends declared by the user.

    Returns the names registered. A declaration that cannot be imported,
    built or validated is logged and skipped: a third-party model must never
    stop PebbleMapper from starting, and a half-registered backend would be
    worse than an absent one.
    """
    import importlib

    src = Path(path) if path is not None else user_detectors_file()
    if src is None or not Path(src).is_file():
        return []
    try:
        entries = json.loads(Path(src).read_text(encoding="utf-8"))
    except (OSError, ValueError) as ex:
        _log.warning("Could not read %s: %s", src, ex)
        return []
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        _log.warning("%s must hold a list of entries, got %s",
                     src, type(entries).__name__)
        return []

    added: List[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            _log.warning("Skipping non-object entry in %s: %r", src, entry)
            continue
        mod_name = str(entry.get("module", "")).strip()
        factory_name = str(entry.get("factory", "") or "make_backend").strip()
        if not mod_name:
            _log.warning("Skipping entry with no 'module' in %s: %r", src, entry)
            continue
        extra_path = str(entry.get("path", "") or "").strip()
        if extra_path:
            _add_import_path(extra_path, Path(src).resolve().parent)
        try:
            mod = importlib.import_module(mod_name)
            factory = getattr(mod, factory_name)
            backend = factory()
            register(backend)
        except Exception as ex:                       # third-party code
            _log.warning("Detector backend %s:%s could not be loaded: %s: %s",
                         mod_name, factory_name, type(ex).__name__, ex)
            continue
        added.append(backend.info.name)
        _log.info("Registered detector backend %r from %s", backend.info.name,
                  mod_name)
        _report_unavailable(backend)
    return added


def _add_import_path(path: str, base: Path) -> Path:
    """Put a declared backend directory on ``sys.path`` (front, once).

    A relative ``path`` is resolved against the JSON file's own folder, so a
    declaration can travel with the backend it names.
    """
    import sys
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = base / p
    p = p.resolve()
    if not p.is_dir():
        _log.warning("Detector path %s does not exist; the import will "
                     "probably fail", p)
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
    return p


def _report_unavailable(backend: DetectorBackend) -> None:
    """One console line saying why a declared backend is not selectable."""
    try:
        available = bool(backend.is_available())
    except Exception as ex:                           # third-party code
        _log.warning("Detector backend %r: is_available() raised %s: %s; "
                     "hidden from the model selector",
                     backend.info.name, type(ex).__name__, ex)
        return
    if available:
        return
    hint = (getattr(backend.info, "install_hint", "") or "").strip()
    _log.warning("Detector backend %r (%s) is declared but not available "
                 "here, so it is hidden from the model selector. %s",
                 backend.info.name, backend.info.display_name,
                 hint or "It gave no install hint.")


# Discover third-party backends at import time so every entry point sees the same registry.
try:
    load_user_backends()
except Exception as _ex:                              # never block startup
    _log.warning("Detector discovery failed: %s: %s", type(_ex).__name__, _ex)
