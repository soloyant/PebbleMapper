"""The app-owned record of installed models, and loading them in isolation.

``~/.pebblemapper/models/installed.json`` lists every model the app installed
(030 FR-014). ``user_detectors.json`` stays the user's: nothing here writes it.

Each installed release is loaded under its own module name,
``pm_ext.<id>_<release>_<variant>`` (FR-016), so two releases of one plug-in
never share a module in the app's process and neither touches ``sys.path``.
A plug-in's own package must therefore import itself relatively; an absolute
self-import is refused with a message saying so (Principle I).
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import re
import sys
import types
from pathlib import Path
from typing import List, Optional


from installer.paths import installed_file, write_json_atomic

_log = logging.getLogger("csm.installer")

FORMAT = 1
_NAMESPACE = "pm_ext"


class LoadError(Exception):
    """An installed model could not be loaded; the message says why."""


# ---------------------------------------------------------------- the file
def read_installed(path: Optional[Path] = None) -> List[dict]:
    """The registered models; ``[]`` when none. An unreadable file is logged
    and treated as empty rather than stopping the app from starting."""
    path = path or installed_file()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as ex:
        _log.warning("Could not read %s (%s); no installed model is loaded.",
                     path, ex)
        return []
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        _log.warning("%s has an unknown format; no installed model is loaded.",
                     path)
        return []
    return [m for m in data.get("models", []) if isinstance(m, dict)]


def _write(models: List[dict], path: Optional[Path] = None) -> None:
    write_json_atomic(path or installed_file(), {"format": FORMAT,
                                                 "models": models})


def _same(entry: dict, model_id: str, release: str, variant: str) -> bool:
    return (entry.get("id"), entry.get("release"), entry.get("variant")) == (
        model_id, release, variant)


def register(entry: dict, path: Optional[Path] = None) -> None:
    """Add or replace one installed release (keyed by id, release, variant)."""
    for key in ("id", "release", "variant", "root", "env", "code", "weights",
                "backends"):
        if key not in entry:
            raise ValueError(f"registration entry lacks {key!r}")
    models = [m for m in read_installed(path)
              if not _same(m, entry["id"], entry["release"], entry["variant"])]
    models.append(entry)
    _write(models, path)


def unregister(model_id: str, release: str, variant: str,
               path: Optional[Path] = None) -> bool:
    """Remove one installed release; True if it was registered."""
    models = read_installed(path)
    kept = [m for m in models if not _same(m, model_id, release, variant)]
    if len(kept) == len(models):
        return False
    _write(kept, path)
    return True


def set_state(model_id: str, release: str, variant: str, state: str,
              reason: str = "", path: Optional[Path] = None) -> None:
    """Record ``ok`` / ``needs_repair`` (with its reason) for one release."""
    models = read_installed(path)
    for m in models:
        if _same(m, model_id, release, variant):
            m["state"] = state
            m["reason"] = reason
    _write(models, path)


# ------------------------------------------------------------ loading
def context_for(entry: dict):
    from detectors.base import InstallContext      # pulls TensorFlow: only when loading
    return InstallContext(env_prefix=Path(entry["env"]),
                          weights_dir=Path(entry["weights"]),
                          code_dir=Path(entry["code"]),
                          release=str(entry["release"]),
                          variant=str(entry.get("variant", "gpu")))


def registry_key(backend_name: str, entry: dict) -> str:
    """The registry key of one backend of an installed release:
    ``<name>@<release>``, plus ``-cpu`` for a processor install."""
    key = f"{backend_name}@{entry['release']}"
    return key + "-cpu" if entry.get("variant") == "cpu" else key


def _alias(entry: dict) -> str:
    raw = f"{entry['id']}_{entry['release']}_{entry.get('variant', 'gpu')}"
    return f"{_NAMESPACE}.{re.sub(r'[^0-9A-Za-z_]', '_', raw)}"


def _ensure_namespace() -> None:
    if _NAMESPACE not in sys.modules:
        ns = types.ModuleType(_NAMESPACE)
        ns.__path__ = []                       # a package with no files
        sys.modules[_NAMESPACE] = ns


def load_isolated(code_dir: Path, package: str, module: str, factory: str,
                  ctx, alias: str):
    """Import ``<code_dir>/<package>`` as ``alias`` and return
    ``<alias>.<module>.<factory>(ctx)``.

    Raises :class:`LoadError` with a sentence a maintainer can act on.
    """
    sub = import_isolated(code_dir, package, module, alias)
    return call_factory(sub, package, module, factory, ctx)


def import_isolated(code_dir: Path, package: str, module: str, alias: str):
    """Import ``<code_dir>/<package>`` as ``alias``; return its ``module``
    submodule (or the package itself when ``module`` is empty)."""
    pkg_dir = Path(code_dir) / package
    init = pkg_dir / "__init__.py"
    if not init.is_file():
        raise LoadError(f"{pkg_dir} is not a Python package (no __init__.py)")
    _ensure_namespace()
    for name in [n for n in sys.modules if n == alias or n.startswith(alias + ".")]:
        del sys.modules[name]                  # a reload replaces, never mixes
    spec = importlib.util.spec_from_file_location(
        alias, init, submodule_search_locations=[str(pkg_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    try:
        spec.loader.exec_module(mod)
        sub = importlib.import_module(f"{alias}.{module}") if module else mod
    except ModuleNotFoundError as ex:
        sys.modules.pop(alias, None)
        if ex.name and ex.name.split(".")[0] == package:
            raise LoadError(
                f"{package} imports itself by its own name ({ex.name}); a model "
                f"installed from the list must use relative imports") from ex
        raise LoadError(f"{package}.{module}: {ex}") from ex
    except Exception as ex:                    # third-party code
        sys.modules.pop(alias, None)
        raise LoadError(f"{package}.{module}: {type(ex).__name__}: {ex}") from ex
    return sub


def call_factory(sub, package: str, module: str, factory: str,
                 ctx):
    fn = getattr(sub, factory, None)
    if fn is None:
        raise LoadError(f"{package}.{module} has no factory {factory!r}")
    try:
        return fn(ctx)
    except Exception as ex:
        raise LoadError(f"{package}.{module}.{factory}(ctx) raised "
                        f"{type(ex).__name__}: {ex}") from ex


def load_entry(entry: dict) -> List[tuple]:
    """``[(registry_key, backend), ...]`` for every backend one installed
    release provides (ImageGrains provides two)."""
    ctx = context_for(entry)
    alias = _alias(entry)
    imported = {}                      # each module once, however many factories
    out = []
    for b in entry["backends"]:
        package, module = b["package"], b.get("module", "backend")
        if (package, module) not in imported:
            imported[(package, module)] = import_isolated(
                Path(entry["code"]), package, module, f"{alias}__{package}")
        backend = call_factory(imported[(package, module)], package, module,
                               b.get("factory", "make_backend"), ctx)
        out.append((registry_key(b["name"], entry), backend))
    return out
