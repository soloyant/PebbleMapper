"""functions/io_robust.py — long-path reads, unreadable != empty."""
import os
import sys
from pathlib import Path

import pytest

from functions.io_robust import (UnreadableFileError, exists_robust,
                                 extended_path, open_robust,
                                 read_text_robust)


def test_normal_reads_pass_through(tmp_path):
    p = tmp_path / "plain.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    assert read_text_robust(p) == "a,b\n1,2\n"


def test_missing_file_stays_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_robust(tmp_path / "nope.csv")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path semantics")
def test_deep_path_beyond_max_path_reads(tmp_path):
    """A >260-char path written via the extended prefix must be readable."""
    deep = tmp_path
    while len(str(deep)) < 250:
        deep = deep / ("d" * 40)
    target = deep / ("f" * 60 + ".csv")
    assert len(str(target)) > 260
    os.makedirs(extended_path(deep), exist_ok=True)
    with open(extended_path(target), "w", encoding="utf-8") as fh:
        fh.write("transect_id,distance\nT1,0.0\n")
    assert exists_robust(target)
    text = read_text_robust(target)
    assert "T1" in text


def test_unreadable_vs_empty_distinction(tmp_path):
    """An empty file reads as empty; an existing-but-unopenable target (here
    a directory where a file is expected) raises UnreadableFileError with the
    real reason — never a look-alike of the empty case."""
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert read_text_robust(empty) == ""

    blocked = tmp_path / "iam_a_directory.csv"
    blocked.mkdir()
    with pytest.raises(UnreadableFileError) as exc:
        read_text_robust(blocked)
    assert "iam_a_directory.csv" in str(exc.value)


def test_unreadable_error_names_path_and_reason():
    err = UnreadableFileError("C:\\x\\y.csv", "path is longer than allowed")
    assert "C:\\x\\y.csv" in str(err)
    assert "longer" in str(err)
