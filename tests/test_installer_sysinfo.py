"""030 FR-017–FR-022, FR-029: what this computer can do."""
import os
import sys
import time
from pathlib import Path

import pytest

from installer import power, sysinfo

GPU_REL = {"variants": {"gpu": {"min_driver": "527.41", "compute_cap": [5.0, 9.0],
                                "min_vram_mb": 3000},
                        "cpu": {"speed_vs_gpu": 20}}}
P600 = sysinfo.Gpu("Quadro P600", "582.42", 6.1, 4096)


def test_smi_parsing_and_the_name_table():
    got = sysinfo.parse_smi("Quadro P600, 582.42, 6.1, 4096\n")
    assert got == [P600]
    old = sysinfo.parse_smi("NVIDIA GeForce GTX 1050, 472.12, Quadro x, 2048")
    assert old[0].compute_cap == 6.1


@pytest.mark.parametrize("cards,force,variant,words", [
    ([P600], False, "gpu", "graphics card"),
    ([], False, "cpu", "about 20 times slower"),
    ([sysinfo.Gpu("Quadro P600", "472.12", 6.1, 4096)], False, "cpu", "processor"),
    ([sysinfo.Gpu("Tiny", "582.42", 6.1, 2048)], False, "cpu", "processor"),
    ([sysinfo.Gpu("Ancient", "582.42", 3.0, 8192)], False, "cpu", "processor"),
    ([P600], True, "cpu", "processor"),
])
def test_verdicts(cards, force, variant, words):
    v = sysinfo.verdict(GPU_REL, cards, force_cpu=force)
    assert v.variant == variant and words in v.text


def test_gpu_only_model_without_a_card_cannot_run_and_says_why():
    rel = {"variants": {"gpu": GPU_REL["variants"]["gpu"]}}
    v = sysinfo.verdict(rel, [])
    assert v.variant is None
    assert v.text == "Cannot run on this computer: it needs an NVIDIA graphics card"
    v = sysinfo.verdict(rel, [sysinfo.Gpu("Quadro P600", "472.12", 6.1, 4096)])
    assert "update it" in v.text


def test_disk_shortfall_names_the_drive(tmp_path):
    assert sysinfo.disk_shortfall(tmp_path, {"peak_bytes": 1}) == {}
    short = sysinfo.disk_shortfall(tmp_path, {"peak_bytes": 10 ** 18})
    [(drive, missing)] = short.items()
    assert drive and missing > 0


def test_metered_parsing():
    assert sysinfo.parse_cost("Fixed") is True
    assert sysinfo.parse_cost("Variable\r\n") is True
    assert sysinfo.parse_cost("Unrestricted") is False
    assert sysinfo.parse_cost("") is None


def test_cloud_folders_are_refused(tmp_path, monkeypatch):
    od = tmp_path / "OneDrive"
    (od / "x").mkdir(parents=True)
    monkeypatch.setattr(sysinfo, "_onedrive_roots", lambda: [od])
    monkeypatch.setattr(sysinfo, "_dropbox_roots", lambda: [])
    monkeypatch.setattr(sysinfo, "_gdrive_roots", lambda: [])
    r = sysinfo.check_models_folder(od / "x" / "PebbleMapper-models")
    assert not r.ok and "OneDrive" in r.reason
    assert sysinfo.check_models_folder(tmp_path / "plain").ok


def test_the_path_budget(tmp_path, monkeypatch):
    for fn in ("_onedrive_roots", "_dropbox_roots", "_gdrive_roots"):
        monkeypatch.setattr(sysinfo, fn, lambda: [])
    deep = Path("D:/") / ("x" * 150)
    r = sysinfo.check_models_folder(deep, longest_relative=139)
    assert not r.ok and "too long" in r.reason
    assert sysinfo.check_models_folder(Path("D:/PebbleMapper-models"), 139).ok


@pytest.mark.skipif(sys.platform != "win32", reason="Windows probes")
def test_the_probes_run_on_this_machine():
    t0 = time.perf_counter()
    cards = sysinfo.gpus()
    p = sysinfo.power()
    ms = sysinfo.modern_standby()
    d = sysinfo.default_models_dir()
    assert isinstance(cards, list) and isinstance(ms, bool)
    assert d.name == "PebbleMapper-models"
    assert p.on_battery in (True, False, None)
    assert time.perf_counter() - t0 < 10


@pytest.mark.skipif(sys.platform != "win32", reason="Windows power request")
def test_keep_awake_is_taken_and_released():
    with power.KeepAwake("test") as k:
        assert k.active
    assert not k.active


def test_usb_drives_are_never_the_default_and_warn(monkeypatch):
    monkeypatch.setattr(sysinfo, "_USB_CACHE", {"E", "D"})
    assert sysinfo.default_models_dir().drive.upper() not in ("E:", "D:")
    for fn in ("_onedrive_roots", "_dropbox_roots", "_gdrive_roots"):
        monkeypatch.setattr(sysinfo, fn, lambda: [])
    r = sysinfo.check_models_folder(Path("E:/PebbleMapper-models"))
    assert r.ok and "USB" in r.warning
