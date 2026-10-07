"""Pluggable detection backends.

Any grain-detection model can be selected as the pipeline's detector; every
backend produces the same canonical per-clast table. Mask R-CNN is the shipped
backend. See ``detectors/base.py``.
"""
from detectors.base import (  # noqa: F401
    BackendInfo,
    CANONICAL_COLUMNS,
    DetectionManifest,
    DetectorBackend,
    TOOL_VERSION,
    output_csv_name,
    run_detect_jobs,
)
from detectors.instance_backend import InstanceSubprocessBackend  # noqa: F401
from detectors.registry import (  # noqa: F401
    all_backends,
    available_backends,
    default_backend_name,
    get_backend,
    normalise_model_name,
    register,
    selector_options,
)

__all__ = [
    "DetectorBackend", "BackendInfo", "DetectionManifest", "CANONICAL_COLUMNS",
    "TOOL_VERSION", "output_csv_name", "run_detect_jobs", "all_backends",
    "available_backends", "get_backend", "default_backend_name", "register",
    "selector_options", "normalise_model_name", "InstanceSubprocessBackend",
]
