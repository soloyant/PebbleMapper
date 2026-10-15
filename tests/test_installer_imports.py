"""The install worker runs without the interface, and its download path does
not pay for TensorFlow (030 research R10)."""
import subprocess
import sys

import pytest

LIGHT = ["installer.paths", "installer.errors", "installer.fetch",
         "installer.archive", "installer.envbuild", "installer.jobs",
         "installer.catalog", "installer.sysinfo", "installer.power",
         "installer.registration"]


@pytest.mark.parametrize("mod", LIGHT)
def test_light_modules_import_neither_gui_nor_tensorflow(mod):
    code = (f"import sys, importlib; importlib.import_module({mod!r}); "
            "bad = [m for m in ('nicegui', 'gui', 'tensorflow') if m in sys.modules]; "
            "print(','.join(bad))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "", f"{mod} imported {out.stdout.strip()}"


def test_the_worker_never_imports_the_interface():
    out = subprocess.run([sys.executable, "-c",
                          "import sys, installer.worker; "
                          "print(','.join(m for m in ('nicegui', 'gui') if m in sys.modules))"],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == ""
