"""030 FR-037, FR-028, SC-008: every failure says what to do, in plain words."""
import re

import pytest

from installer.errors import CLASSES, InstallError, human_size

_FACTS = {"drive": "C:", "missing": "2.1 GB", "host": "github.com",
          "file": "weights.pth", "reason": "it needs an NVIDIA graphics card",
          "folder": "C:\\Models", "step": "Building", "path": "x.dll",
          "detail": "boom"}
_JARGON = re.compile(r"\b(variant|lock|prefix|environment)s?\b", re.I)


@pytest.mark.parametrize("cls", CLASSES)
def test_every_class_names_a_failure_and_an_action(cls):
    err = InstallError(cls, **_FACTS)
    d = err.as_dict()
    assert d["message"] and d["action"]
    assert d["message"].endswith(".") and d["action"].endswith(".")
    assert "?" not in d["message"] + d["action"], "a fact was left unfilled"
    assert not _JARGON.search(d["message"] + " " + d["action"])
    assert "terminal" not in d["action"].lower()


def test_the_classes_of_the_data_models():
    # 030 data-model §6 (twelve) + 031 data-model: starter_too_old.
    assert len(CLASSES) == 13 and "starter_too_old" in CLASSES


def test_an_unknown_class_is_a_programming_error():
    with pytest.raises(ValueError):
        InstallError("nope")


def test_human_size():
    assert human_size(1_800_000_000) == "1.8 GB"
    assert human_size(340_000_000) == "340 MB"
