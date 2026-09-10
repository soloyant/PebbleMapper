"""The per-clast schema contract.

`_CLAST_COLUMNS` is declared twice on purpose, in `functions/clasts_detection.py`
and `functions/clasts_merge.py`, so that merging need not import TensorFlow.
These tests keep the two copies identical, keep the reference page
("The clast table" section of `docs/user-manual.md`) true to the code, and keep old CSVs readable.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from functions import clasts_detection as _detection
from functions import clasts_merge as _merge

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DOC = REPO_ROOT / "docs" / "user-manual.md"
SCHEMA_HEADING = "## The clast table"
SCHEMA_ANCHOR = "the-clast-table"
SCHEMA_LINK = f"../user-manual.md#{SCHEMA_ANCHOR}"


def test_the_two_declarations_are_identical():
    """Content and order: a set comparison would let a column drift to the end."""
    detection = list(_detection._CLAST_COLUMNS)
    merge = list(_merge._CLAST_COLUMNS)
    assert detection == merge, (
        "_CLAST_COLUMNS has drifted between its two declarations.\n"
        f"  functions/clasts_detection.py : {detection}\n"
        f"  functions/clasts_merge.py     : {merge}\n"
        "The duplication is deliberate (merge must not import TensorFlow), so "
        "both must be edited together."
    )


def test_streaming_columns_is_the_canonical_list_plus_the_tile_marker():
    """The one legitimate difference, and only that one."""
    expected = list(_detection._CLAST_COLUMNS) + ["_tile_idx"]
    assert list(_detection._STREAMING_COLUMNS) == expected, (
        "_STREAMING_COLUMNS must be _CLAST_COLUMNS + ['_tile_idx'] exactly.\n"
        f"  got:      {list(_detection._STREAMING_COLUMNS)}\n"
        f"  expected: {expected}"
    )


def test_merge_does_not_import_tensorflow():
    """The reason the duplication exists. Run in a subprocess, because another
    test has already imported TensorFlow into this process."""
    probe = (
        "import sys; sys.path.insert(0, r'%s'); "
        "import functions.clasts_merge; "
        "print('tensorflow' in sys.modules)" % REPO_ROOT
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"probe failed to import functions.clasts_merge:\n{result.stderr[-2000:]}"
    )
    assert result.stdout.strip() == "False", (
        "functions/clasts_merge.py now pulls in TensorFlow; that is the whole "
        "reason _CLAST_COLUMNS is declared twice rather than imported."
    )


def _schema_section():
    """The clast-table appendix of the user manual, up to the next section."""
    assert SCHEMA_DOC.is_file(), f"user manual missing: {SCHEMA_DOC}"
    manual = SCHEMA_DOC.read_text(encoding="utf-8")
    assert SCHEMA_HEADING in manual, f"user manual has no {SCHEMA_HEADING!r}"
    return manual.split(SCHEMA_HEADING, 1)[1].split("\n## ", 1)[0]


def _contract_rows():
    """Parse the column table out of the manual's clast-table appendix."""
    rows = []
    for line in _schema_section().splitlines():
        parts = [p.strip() for p in line.split("|")]
        # "| 4 | `Clast_length` | label | mm | m | 1000.0 |" -> 8 parts with blanks
        if len(parts) == 8 and parts[1].isdigit() and parts[2].startswith("`"):
            rows.append({
                "column": parts[2].strip("`"),
                "unit": parts[4],
                "factor": parts[6],
            })
    return rows


def test_contract_document_matches_the_registries():
    """Every column, in order, with the unit functions/units.py reports."""
    from functions import units as _units

    rows = _contract_rows()
    assert [r["column"] for r in rows] == list(_detection._CLAST_COLUMNS), (
        "The manual's clast-table appendix lists different columns, or a different order, "
        "from _CLAST_COLUMNS.\n"
        f"  document: {[r['column'] for r in rows]}\n"
        f"  code:     {list(_detection._CLAST_COLUMNS)}"
    )
    for row in rows:
        unit, factor = _units.field_unit_and_factor(row["column"])
        expected_unit = unit if unit else "(dimensionless)"
        assert row["unit"] == expected_unit, (
            f"{row['column']}: document says unit {row['unit']!r}, "
            f"functions/units.py says {expected_unit!r}"
        )
        assert float(row["factor"]) == float(factor), (
            f"{row['column']}: document says factor {row['factor']}, "
            f"functions/units.py says {factor}"
        )


def test_the_backend_guide_routes_to_the_contract():
    """One description, referenced from the guide, never copied into it."""
    guide = REPO_ROOT / "docs" / "developer" / "adding-a-detection-model.md"
    assert guide.is_file(), f"developer guide missing: {guide}"
    text = guide.read_text(encoding="utf-8")
    assert SCHEMA_LINK in text, (
        f"docs/developer/adding-a-detection-model.md must link to {SCHEMA_LINK}"
    )
    for line in text.splitlines():
        if "Ellipse_major_axis" in line and SCHEMA_ANCHOR not in line:
            assert line.strip().startswith(">"), (
                "docs/developer/adding-a-detection-model.md has regrown a "
                f"per-clast column list:\n  {line.strip()}"
            )


def test_a_csv_predating_the_schema_expansion_still_reads(tmp_path):
    """The contract describes what the tool produces; CSVs written before the
    Perimeter / Eccentricity / Solidity / Mean_intensity columns existed must
    still merge."""
    import pandas as pd
    from functions.clasts_merge import merge_csvs

    old = ["clast_ID", "x", "y", "Clast_length", "Clast_width",
           "Ellipse_major_axis", "Ellipse_minor_axis", "Surface_area",
           "Equivalent_diameter", "Score", "Orientation"]
    assert set(old) < set(_detection._CLAST_COLUMNS)

    def _write(path, x0):
        # Real geometry: zero-size ellipses make degenerate polygons that the
        # IoU dedup discards, and an empty result would prove nothing.
        rows = [{
            "clast_ID": i, "x": x0 + i * 10.0, "y": 0.0,
            "Clast_length": 0.050, "Clast_width": 0.030,
            "Ellipse_major_axis": 0.050, "Ellipse_minor_axis": 0.030,
            "Surface_area": 0.00118, "Equivalent_diameter": 0.0388,
            "Score": 0.99, "Orientation": 12.0,
        } for i in range(3)]
        pd.DataFrame(rows, columns=old).to_csv(path, index=False)

    small, large, out = (tmp_path / "s.csv", tmp_path / "l.csv",
                         tmp_path / "merged.csv")
    _write(small, 0.0)
    _write(large, 100.0)

    merge_csvs(str(small), str(large), str(out))

    assert out.is_file(), "merge refused a CSV predating the schema expansion"
    merged = pd.read_csv(out)
    assert len(merged) == 6
    for column in old:
        assert column in merged.columns, f"{column} lost by merge"
