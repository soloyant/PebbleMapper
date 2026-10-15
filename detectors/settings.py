"""Model settings (spec 032): what each model lets the user change, and what
a project has changed.

A model release describes its settings in ``pm_settings.json`` beside its
package's ``__init__.py`` (Mask R-CNN's ships with the application); the
format is ``specs/032-model-options/contracts/settings-description.md``. The
description is read as data — nothing is imported to read it — and checked
in full: a description with one fault is refused as a whole, and the model is
then treated as one that does not describe its settings (spec US6).

A project keeps the values the user changed in ``<project>/.model_settings.json``
(``~/.pebblemapper/model_settings.json`` without a project), per model name and
mode. Only changed values are stored and sent to a run, so a model works out
every other value itself, exactly as before 032 (FR-015, SC-005). A value the
current release does not describe, or no longer accepts, is moved to ``kept``:
never sent, never deleted (FR-020).
"""
from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

FORMAT = 1
DESCRIPTION_FILE = "pm_settings.json"
STORE_FILE = ".model_settings.json"
SHARED_CONTROLS = ("tile_size", "overlap", "device", "device_number", "confidence", "dedup")
KINDS = ("number", "choice", "bool")
UNITS = ("px", "px²", "mm", "m", "%", "fraction", "count", "")
MODES = ("quadrat", "ortho")
_KEY_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_")


class DescriptionError(ValueError):
    """A settings description that breaks the format (FR-003)."""


# ----------------------------------------------------------- description
def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _check_value(s: dict, value, where: str) -> None:
    """Raise DescriptionError when ``value`` is not a valid value of ``s``."""
    try:
        validate_value(s, value)
    except ValueError as ex:
        raise DescriptionError(f"{where}: {ex}") from None


def validate_description(data) -> dict:
    """Check a description in full; return it with optional fields filled.
    Raise :class:`DescriptionError` naming the first fault."""
    if not isinstance(data, dict):
        raise DescriptionError("the description is not an object")
    if data.get("format") != FORMAT:
        raise DescriptionError(f"format {data.get('format')!r} is not {FORMAT}")
    if not isinstance(data.get("model"), str) or not data["model"]:
        raise DescriptionError("'model' is missing")
    shared = data.get("shared_controls", list(SHARED_CONTROLS))
    if not isinstance(shared, list) or any(c not in SHARED_CONTROLS for c in shared):
        raise DescriptionError(f"'shared_controls' must be a subset of {SHARED_CONTROLS}")
    not_used = data.get("not_used", {})
    if not isinstance(not_used, dict) or any(
            not isinstance(v, str) or not v.strip() for v in not_used.values()):
        raise DescriptionError("'not_used' must give a reason for each control")
    for c in SHARED_CONTROLS:
        if c not in shared and c not in not_used:
            raise DescriptionError(f"shared control {c!r} is not used but no reason is given")
    settings = data.get("settings", [])
    if not isinstance(settings, list):
        raise DescriptionError("'settings' must be a list")
    keys = set()
    out = []
    for i, s in enumerate(settings):
        where = f"setting {i + 1}"
        if not isinstance(s, dict):
            raise DescriptionError(f"{where} is not an object")
        key = s.get("key")
        if not isinstance(key, str) or not key or not key[0].isalpha() \
                or not set(key) <= _KEY_CHARS:
            raise DescriptionError(f"{where}: key {key!r} is not a lowercase name")
        where = f"setting {key!r}"
        if key in keys:
            raise DescriptionError(f"{where} appears twice")
        keys.add(key)
        for field in ("label", "help"):
            if not isinstance(s.get(field), str) or not s[field].strip():
                raise DescriptionError(f"{where}: '{field}' is missing")
        if s.get("unit", "") not in UNITS:
            raise DescriptionError(f"{where}: unit {s.get('unit')!r} is not one of {UNITS}")
        kind = s.get("kind")
        if kind not in KINDS:
            raise DescriptionError(f"{where}: kind {kind!r} is not one of {KINDS}")
        if kind == "number":
            for b in ("min", "max"):
                if not _is_number(s.get(b)):
                    raise DescriptionError(f"{where}: '{b}' must be a number")
            if s["min"] > s["max"]:
                raise DescriptionError(f"{where}: min is above max")
            if "step" in s and (not _is_number(s["step"]) or s["step"] <= 0):
                raise DescriptionError(f"{where}: 'step' must be a positive number")
        if kind == "choice":
            ch = s.get("choices")
            if not isinstance(ch, list) or not ch or any(
                    not isinstance(c, dict) or "value" not in c or not c.get("label") for c in ch):
                raise DescriptionError(f"{where}: 'choices' must list {{value, label}}")
            if len({json.dumps(c["value"]) for c in ch}) != len(ch):
                raise DescriptionError(f"{where}: two choices share a value")
        modes = s.get("modes", list(MODES))
        if not isinstance(modes, list) or not modes or any(m not in MODES for m in modes):
            raise DescriptionError(f"{where}: 'modes' must be a subset of {MODES}")
        by_mode = s.get("default_by_mode")
        if by_mode is None:
            if "default" not in s:
                raise DescriptionError(f"{where}: no default")
            defaults = {m: s["default"] for m in modes}
        else:
            if not isinstance(by_mode, dict) or set(by_mode) != set(modes):
                raise DescriptionError(f"{where}: 'default_by_mode' must name every mode it applies to")
            defaults = dict(by_mode)
        for m, d in defaults.items():
            if isinstance(d, dict):
                if not isinstance(d.get("derived"), str) or not d["derived"].strip():
                    raise DescriptionError(f"{where}: a worked-out default needs its 'derived' sentence")
            else:
                _check_value(s, d, f"{where} default for {m}")
        authors = s.get("authors")
        if authors is not None:
            if not isinstance(authors, dict) or "value" not in authors \
                    or not isinstance(authors.get("source"), str) or not authors["source"].strip():
                raise DescriptionError(f"{where}: 'authors' needs a value and its source")
            if authors["value"] is not None:
                _check_value(s, authors["value"], f"{where} authors' value")
            fixed_defaults = [d for d in defaults.values() if not isinstance(d, dict)]
            # A value of None is no recommendation: nothing to differ from.
            if authors["value"] is not None \
                    and any(d != authors["value"] for d in fixed_defaults) and not str(
                    s.get("differs_because", "")).strip():
                raise DescriptionError(f"{where}: differs from the authors without 'differs_because'")
        req = s.get("requires")
        if req is not None and (not isinstance(req, dict) or "key" not in req or "value" not in req):
            raise DescriptionError(f"{where}: 'requires' needs a key and a value")
        out.append(dict(s, modes=modes, defaults=defaults, main=bool(s.get("main", False)),
                        unit=s.get("unit", "")))
    for s in out:
        req = s.get("requires")
        if req and req["key"] not in keys:
            raise DescriptionError(f"setting {s['key']!r} requires an unknown setting {req['key']!r}")
    fixed = data.get("fixed", [])
    if not isinstance(fixed, list) or any(
            not isinstance(f, dict) or not f.get("label") or not f.get("reason") for f in fixed):
        raise DescriptionError("'fixed' entries need a label and a reason")
    return dict(data, shared_controls=list(shared), not_used=dict(not_used),
                settings=out, fixed=list(fixed))


def number_text(v) -> str:
    """20.0 → 20, 0.85 → 0.85."""
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def unit_text(s: dict) -> str:
    """The unit as shown after a value ("" for a fraction or a count)."""
    unit = s.get("unit", "")
    return "" if unit in ("", "fraction", "count") else unit


def range_text(s: dict) -> str:
    """A number setting's allowed range, as the panel and the refusal say it."""
    unit = unit_text(s)
    sep = "" if unit == "%" else " "
    return (f"{number_text(s.get('min'))} to {number_text(s.get('max'))}"
            + (f"{sep}{unit}" if unit else ""))


def validate_value(s: dict, value):
    """``value`` as the setting accepts it, or ValueError saying what is
    allowed (FR-009)."""
    kind = s.get("kind")
    if kind == "bool":
        if isinstance(value, bool):
            return value
        raise ValueError("Allowed: yes or no")
    if kind == "choice":
        for c in s.get("choices", []):
            if c["value"] == value and type(c["value"]) is type(value):
                return value
        raise ValueError("Allowed: " + ", ".join(str(c["label"]) for c in s.get("choices", [])))
    if kind == "number":
        lo, hi = s.get("min"), s.get("max")
        allowed = f"Allowed: {range_text(s)}"
        if not _is_number(value):
            raise ValueError(allowed)
        if s.get("integer"):
            if float(value) != int(value):
                raise ValueError(allowed + ", whole numbers")
            value = int(value)
        if not lo <= value <= hi:
            raise ValueError(allowed)
        return value
    raise ValueError(f"unknown kind {kind!r}")


def default_for(s: dict, mode: str):
    """PebbleMapper's value of ``s`` in ``mode``: a value, or
    ``{"derived": sentence}`` when the model works it out; None when the
    setting does not apply to ``mode``."""
    return s.get("defaults", {}).get(mode)


def applies(s: dict, mode: str, values: Dict[str, Any], description: dict) -> Tuple[bool, str]:
    """Whether ``s`` is used in ``mode`` given the other values, and if not, why."""
    if mode not in s.get("modes", MODES):
        only = " and ".join(m for m in s.get("modes", ()))
        return False, f"used for {only} only"
    req = s.get("requires")
    if req:
        other = next((o for o in description["settings"] if o["key"] == req["key"]), None)
        if other is not None:
            v = values.get(other["key"], default_for(other, mode))
            if v != req["value"]:
                want = "on" if req["value"] is True else "off" if req["value"] is False \
                    else str(req["value"])
                return False, f"used only when '{other['label']}' is {want}"
    return True, ""


# ------------------------------------------------- finding a description
_CACHE: Dict[str, Tuple[float, Optional[dict], str]] = {}


def description_path(backend) -> Optional[Path]:
    """Where ``backend``'s description lives: its class's ``settings_file``,
    else ``pm_settings.json`` beside the module that defines its class."""
    explicit = getattr(backend, "settings_file", None)
    if explicit:
        return Path(explicit)
    mod = sys.modules.get(type(backend).__module__)
    f = getattr(mod, "__file__", None)
    return Path(f).parent / DESCRIPTION_FILE if f else None


def load_description(backend) -> Optional[dict]:
    """The checked description of ``backend``, or None (no file, or a file
    that breaks the format — the reason is logged once and kept for
    :func:`description_problem`)."""
    p = description_path(backend)
    if p is None or not p.is_file():
        return None
    key = str(p)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None
    hit = _CACHE.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        desc = validate_description(json.loads(p.read_text(encoding="utf-8")))
        problem = ""
        name = getattr(getattr(backend, "info", None), "name", None)
        if name and desc["model"] != name:
            raise DescriptionError(f"it describes {desc['model']!r}, not {name!r}")
    except (OSError, ValueError) as ex:
        desc, problem = None, str(ex)
        log.warning("model settings: %s is not used: %s", p, problem)
        print(f"[models] the settings description {p} is not used: {problem}", flush=True)
    _CACHE[key] = (mtime, desc, problem)
    return desc


def description_problem(backend) -> str:
    """Why a present description was refused ('' when none or fine)."""
    p = description_path(backend)
    hit = _CACHE.get(str(p)) if p else None
    return hit[2] if hit else ""


# ------------------------------------------------- results in a project
def recorded_value_sets(folder, model: str) -> List[Dict[str, Any]]:
    """The distinct sets of changed values the results of ``model`` in
    ``folder`` were made with, read from their records; a record from before
    032 counts as PebbleMapper's values (``{}``). Never raises."""
    import os
    name = model_name(model)
    seen, out = set(), []
    if not folder or not os.path.isdir(folder):
        return out
    for root, _dirs, files in os.walk(folder):
        for fn in files:
            if not fn.endswith(".manifest.json"):
                continue
            try:
                with open(os.path.join(root, fn), encoding="utf-8") as fh:
                    man = json.load(fh)
            except (OSError, ValueError):
                continue
            if model_name(str(man.get("model", ""))) != name:
                continue
            values = (man.get("params") or {}).get("settings_changed") or {}
            if not isinstance(values, dict):
                continue
            k = json.dumps(values, sort_keys=True)
            if k not in seen:
                seen.add(k)
                out.append(values)
    return out


def describe_values(values: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(values.items())) \
        if values else "PebbleMapper's values"


# ------------------------------------------------- environment variables
# Variables the plug-ins read that change a model's settings or device
# (spec 032 FR-019, research R6). Location variables (``PM_*_WEIGHTS``,
# ``PM_*_ENV``, ``PM_ORTHOSAM_WORK``) stay: hand-installed and development
# plug-ins find their environment and weights through them.
SETTING_ENV = ("PM_ORTHOSAM_MODEL", "PM_SEG_SAM_CKPT", "PM_SEG_SAM_CFG",
               "PM_SEG_DEVICE", "PM_IG_DEVICE")
IGNORED_ENV: List[str] = []


def strip_setting_env() -> List[str]:
    """Remove the setting and device variables before any model is loaded,
    so neither a backend nor its subprocess sees them; say once which were
    set. Returns the names removed."""
    import os
    import re
    names = [n for n in list(os.environ)
             if n.upper() in SETTING_ENV or re.fullmatch(r"PM_[A-Z0-9]+_OPTIONS", n.upper())]
    for n in names:
        value = os.environ.pop(n)
        IGNORED_ENV.append(f"{n}={value}")
    if names:
        print("[models] these environment variables changed model settings and are "
              "ignored: set the values in the model's Settings instead: "
              + ", ".join(IGNORED_ENV[-len(names):]), flush=True)
    return names


# ------------------------------------------------------------ run log
_GPU_NAMES: Optional[List[str]] = None


def gpu_names() -> List[str]:
    """The NVIDIA cards' names, read once per session (nvidia-smi)."""
    global _GPU_NAMES
    if _GPU_NAMES is None:
        try:
            from installer.sysinfo import gpus
            _GPU_NAMES = [g.name for g in gpus()]
        except Exception:
            _GPU_NAMES = []
    return _GPU_NAMES


def device_text(devicemode, devicenumber) -> str:
    if str(devicemode or "").upper() != "GPU":
        return "processor"
    n = int(devicenumber or 0)
    names = gpu_names()
    return f"graphics card {n}" + (f" ({names[n]})" if n < len(names) else "")


def run_header(backend, mode: str, changed: Dict[str, Any], devicemode,
               devicenumber) -> List[str]:
    """The first lines of every run's log (spec 032 FR-012): the model and
    its release, the device, and the settings that are not PebbleMapper's."""
    info = getattr(backend, "info", None)
    name = getattr(info, "display_name", None) or getattr(info, "name", "model")
    key = getattr(backend, "registry_key", "") or ""
    release = f", release {key.split('@', 1)[1]}" if "@" in key else ""
    variant = getattr(backend, "variant", "gpu")
    dev = "processor (processor version of the model)" if variant == "cpu" \
        else device_text(devicemode, devicenumber)
    lines = [f"[settings] model: {name}{release}; device: {dev}"]
    # Only what will be sent: described by this release, used in this mode,
    # valid (the backend drops and names anything else).
    desc = load_description(backend) if changed else None
    if changed and desc is None and getattr(info, "name", None) != "maskrcnn":
        lines.append("[settings] settings: this release's built-in values (it does not "
                     "describe its settings)")
        return lines
    if desc is not None:
        by_key = {s["key"]: s for s in desc["settings"]}
        kept = {}
        for k, v in changed.items():
            s = by_key.get(k)
            if s is None or mode not in s["modes"]:
                continue
            try:
                kept[k] = validate_value(s, v)
            except ValueError:
                pass
        changed = kept
    if changed:
        lines.append("[settings] settings changed: "
                     + ", ".join(f"{k}={v}" for k, v in changed.items()))
    else:
        lines.append("[settings] settings: PebbleMapper's values")
    return lines


# ------------------------------------------------------- project values
def model_name(key: str) -> str:
    """The name settings are stored under: the backend name, whatever the
    release or the processor variant (``seg@1.1.0-cpu`` → ``seg``)."""
    name = str(key).split("@", 1)[0]
    return name[:-4] if name.endswith("-cpu") else name


def store_path(project_dir: Optional[Path]) -> Path:
    if project_dir:
        return Path(project_dir) / STORE_FILE
    from functions.layout import app_home
    return app_home() / "model_settings.json"


class ProjectSettings:
    """The values a project changed, per model name and mode."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.models: Dict[str, dict] = {}
        self.problem = ""
        self.load()

    # -- file
    def load(self) -> None:
        self.models, self.problem = {}, ""
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("format") != FORMAT or not isinstance(data.get("models"), dict):
                raise ValueError(f"format {data.get('format')!r}")
            self.models = data["models"]
        except (OSError, ValueError, AttributeError) as ex:
            # Never overwritten silently: the next save keeps it beside.
            self.problem = (f"{self.path.name} could not be read ({ex}); PebbleMapper's "
                            "values are used until a setting is changed")
            print(f"[models] {self.problem}", flush=True)

    def save(self) -> None:
        from installer.paths import write_json_atomic
        if self.problem and self.path.is_file():
            bad = self.path.with_name(self.path.name + ".unreadable")
            if not bad.exists():
                self.path.replace(bad)
        write_json_atomic(self.path, {"format": FORMAT, "models": self.models,
                                      "saved_at": datetime.now().isoformat(timespec="seconds")})
        self.problem = ""

    # -- values
    def _slot(self, model: str, part: str, mode: str) -> dict:
        m = self.models.setdefault(model_name(model), {})
        return m.setdefault(part, {}).setdefault(mode, {})

    def values(self, model: str, mode: str) -> Dict[str, Any]:
        """The values changed for ``model`` in ``mode`` (never ``kept`` ones)."""
        return dict(self.models.get(model_name(model), {}).get("values", {}).get(mode, {}))

    def kept(self, model: str, mode: str) -> Dict[str, Any]:
        return dict(self.models.get(model_name(model), {}).get("kept", {}).get(mode, {}))

    def set(self, model: str, mode: str, description: dict, key: str, value) -> Any:
        """Check and store one value; storing PebbleMapper's own value removes
        the change. Returns the value stored. ValueError when not allowed."""
        s = next((x for x in description["settings"] if x["key"] == key), None)
        if s is None:
            raise ValueError(f"{key!r} is not a setting of this model")
        if mode not in s["modes"]:
            raise ValueError(f"{s['label']} is not used in {mode}")
        value = validate_value(s, value)
        slot = self._slot(model, "values", mode)
        if default_for(s, mode) == value:
            slot.pop(key, None)
        else:
            slot[key] = value
        self.save()
        return value

    def reset(self, model: str, mode: str, key: Optional[str] = None) -> None:
        slot = self._slot(model, "values", mode)
        if key is None:
            slot.clear()
        else:
            slot.pop(key, None)
        self.save()

    def check_against(self, model: str, description: Optional[dict]) -> List[str]:
        """Move every stored value the description does not accept to
        ``kept``; return one sentence per value moved (FR-009, FR-020).
        Kept values that became valid again are brought back."""
        notices = []
        if description is None:
            return notices
        by_key = {s["key"]: s for s in description["settings"]}
        entry = self.models.get(model_name(model))
        if not entry:
            return notices
        changed = False
        for mode in MODES:
            vals = entry.get("values", {}).get(mode, {})
            kept = entry.get("kept", {}).get(mode, {})
            for key, v in list(kept.items()):
                s = by_key.get(key)
                if s and mode in s["modes"] and key not in vals:
                    try:
                        vals[key] = validate_value(s, v)
                        del kept[key]
                        entry.setdefault("values", {})[mode] = vals
                        changed = True
                    except ValueError:
                        pass
            for key, v in list(vals.items()):
                s = by_key.get(key)
                reason = None
                if s is None:
                    reason = "this release does not have it"
                elif mode not in s["modes"]:
                    reason = f"it is not used in {mode}"
                else:
                    try:
                        validate_value(s, v)
                    except ValueError as ex:
                        reason = f"{v!r} is not accepted any more ({ex})"
                if reason:
                    entry.setdefault("kept", {}).setdefault(mode, {})[key] = v
                    del vals[key]
                    changed = True
                    label = s["label"] if s else key
                    notices.append(f"{label} ({mode}): PebbleMapper's value is used, "
                                   f"because {reason}; your value is kept in the project.")
        if changed:
            self.save()
        return notices

    def changed(self, model: str, mode: str, description: Optional[dict]) -> Dict[str, Any]:
        """The values that differ from PebbleMapper's and will be sent."""
        if description is None:
            return {}
        by_key = {s["key"]: s for s in description["settings"]}
        out = {}
        for key, v in self.values(model, mode).items():
            s = by_key.get(key)
            if s and mode in s["modes"] and default_for(s, mode) != v:
                out[key] = v
        return out
