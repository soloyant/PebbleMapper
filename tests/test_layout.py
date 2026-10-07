"""Tests for functions/layout.py — project directory layout helpers.

Covers:
  - project_path: canonical path contains expected subfolder names
  - list_projects: known project names present
  - ensure_project_layout: creates correct dirs, is idempotent
  - Path-traversal guard: '../../evil', '/absolute/path', and similar names
    must be rejected by ensure_project_layout (and project_path via
    DATASETS_ROOT / project resolution)
  - default_starting_dir: returns a string path that exists or is creatable
"""
from pathlib import Path

import pytest

from functions.layout import (
    DATASETS_ROOT,
    PATH_KINDS,
    default_starting_dir,
    ensure_project_layout,
    list_projects,
    project_path,
)


# ---------------------------------------------------------------------------
# project_path
# ---------------------------------------------------------------------------

def test_project_path_vectors_contains_expected_segment():
    """project_path(…, 'vectors') must include either 'output_results' or 'results'."""
    p = project_path("Marshbeach_2023", "vectors")
    assert "output_results" in str(p) or "results" in str(p)


def test_project_path_empty_project_returns_datasets_root():
    assert project_path("") == DATASETS_ROOT


def test_project_path_empty_kind_returns_project_root():
    p = project_path("Marshbeach_2023", "")
    assert p == DATASETS_ROOT / "Marshbeach_2023"


def test_project_path_unknown_kind_returns_project_root():
    """Unknown 'kind' values must fall back to the project root, not raise."""
    p = project_path("Marshbeach_2023", "nonexistent_kind_xyz")
    assert p == DATASETS_ROOT / "Marshbeach_2023"


def test_project_path_all_kinds_return_path_objects():
    """Every registered kind must return a Path without raising."""
    for kind in PATH_KINDS:
        result = project_path("Marshbeach_2023", kind)
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# list_projects
# ---------------------------------------------------------------------------

def test_list_projects_returns_list():
    projects = list_projects()
    assert isinstance(projects, list)


def test_list_projects_includes_known(monkeypatch, tmp_path):
    """A project directory under DATASETS_ROOT must be listed.

    Must NOT assert on the real datasets/: that tree is
    gitignored field data, so any such assertion fails on a fresh clone.
    """
    monkeypatch.setattr("functions.layout.DATASETS_ROOT", tmp_path)
    (tmp_path / "Bio_Station").mkdir()
    assert "Bio_Station" in list_projects()


def test_list_projects_sorted():
    projects = list_projects()
    assert projects == sorted(projects)


def test_list_projects_empty_datasets_root(monkeypatch, tmp_path):
    """When DATASETS_ROOT doesn't exist, list_projects must return []."""
    monkeypatch.setattr("functions.layout.DATASETS_ROOT", tmp_path / "no_such_dir")
    assert list_projects() == []


# ---------------------------------------------------------------------------
# ensure_project_layout
# ---------------------------------------------------------------------------

def test_ensure_project_layout_creates_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr("functions.layout.DATASETS_ROOT", tmp_path)
    ensure_project_layout("test_project")
    proj = tmp_path / "test_project"
    # Inputs grouped under input_data/.
    assert (proj / "input_data" / "images").is_dir()
    assert (proj / "input_data" / "dem").is_dir()
    assert (proj / "input_data" / "geometries").is_dir()
    # Outputs.
    assert (proj / "output_results" / "vectors").is_dir()
    assert (proj / "output_results" / "rasters").is_dir()
    # Validation (structured inputs + results).
    assert (proj / "validation" / "raw").is_dir()
    # The folder Orthorectify writes (and Detect / Digitize read first).
    assert (proj / "validation" / "orthorectified").is_dir()
    assert (proj / "validation" / "georectified").is_dir()
    assert (proj / "validation" / "results").is_dir()
    # A README explaining where inputs go is dropped in the project root.
    assert (proj / "README.txt").exists()


def test_ensure_project_layout_idempotent(monkeypatch, tmp_path):
    """Calling ensure_project_layout twice must not raise."""
    monkeypatch.setattr("functions.layout.DATASETS_ROOT", tmp_path)
    ensure_project_layout("test_project")
    ensure_project_layout("test_project")  # second call must be a no-op


def test_ensure_project_layout_extra_kinds(monkeypatch, tmp_path):
    """extra_kinds='zonal' must create the zonal subfolder."""
    monkeypatch.setattr("functions.layout.DATASETS_ROOT", tmp_path)
    ensure_project_layout("test_project", extra_kinds=("zonal",))
    assert (tmp_path / "test_project" / "output_results" / "zonal").is_dir()


# ---------------------------------------------------------------------------
# Path-traversal guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_name", [
    "../../evil",
    "../sibling",
    "/absolute/path",
    "foo/bar",
    "foo\\bar",
    "",
])
def test_project_name_traversal_rejected(monkeypatch, tmp_path, bad_name):
    monkeypatch.setattr("functions.layout.DATASETS_ROOT", tmp_path)
    with pytest.raises((ValueError, OSError)):
        ensure_project_layout(bad_name)


def test_project_name_with_null_byte_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr("functions.layout.DATASETS_ROOT", tmp_path)
    with pytest.raises(ValueError):
        ensure_project_layout("evil\x00name")


def test_valid_project_names_accepted(monkeypatch, tmp_path):
    """Typical valid project names must not be rejected."""
    monkeypatch.setattr("functions.layout.DATASETS_ROOT", tmp_path)
    for name in ["MyProject", "beach_2023", "run-01", "data.v2"]:
        ensure_project_layout(name)  # must not raise


# ---------------------------------------------------------------------------
# default_starting_dir
# ---------------------------------------------------------------------------

def test_default_starting_dir_returns_string(monkeypatch, tmp_path):
    """default_starting_dir creates the directory it returns, so it MUST be
    run against a temporary root — un-patched it writes into the real
    datasets/ field-data tree and leaves a phantom project behind."""
    monkeypatch.setattr("functions.layout.DATASETS_ROOT", tmp_path)
    result = default_starting_dir("vectors", "Marshbeach_2023")
    assert isinstance(result, str)


def test_default_starting_dir_no_project_returns_datasets_root_or_repo():
    result = default_starting_dir("vectors", "")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# CSM_DATASETS_ROOT environment variable override
# ---------------------------------------------------------------------------

def test_datasets_root_env_override(tmp_path, monkeypatch):
    """Setting CSM_DATASETS_ROOT must redirect DATASETS_ROOT on re-import."""
    import importlib
    import functions.layout as layout_mod

    monkeypatch.setenv("CSM_DATASETS_ROOT", str(tmp_path))
    try:
        importlib.reload(layout_mod)
        assert layout_mod.DATASETS_ROOT == tmp_path.resolve()
    finally:
        # Restore module to its default state regardless of test outcome.
        monkeypatch.delenv("CSM_DATASETS_ROOT", raising=False)
        importlib.reload(layout_mod)


def test_datasets_root_default_when_env_unset(monkeypatch, tmp_path):
    """Without an env override or a persisted config, the default
    (<repo>/datasets) is preserved."""
    import importlib
    import functions.layout as layout_mod

    monkeypatch.delenv("CSM_DATASETS_ROOT", raising=False)
    monkeypatch.delenv("PEBBLEMAPPER_DATASETS_ROOT", raising=False)
    # Point HOME at an empty dir so a real ~/.pebblemapper/config.json (written
    # by the GUI picker) can't make this test machine-dependent.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    importlib.reload(layout_mod)
    try:
        assert layout_mod.DATASETS_ROOT == layout_mod.REPO_ROOT / "datasets"
    finally:
        importlib.reload(layout_mod)


# ---------------------------------------------------------------------------
# get_datasets_root / set_datasets_root (runtime-mutable root)
# ---------------------------------------------------------------------------

def test_set_datasets_root_applies_live(tmp_path):
    """Switching the root must immediately retarget project_path/list_projects."""
    import functions.layout as layout_mod
    original = layout_mod.get_datasets_root()
    try:
        (tmp_path / "ProjA").mkdir()
        (tmp_path / "ProjB").mkdir()
        layout_mod.set_datasets_root(tmp_path, persist=False)
        assert layout_mod.get_datasets_root() == tmp_path.resolve()
        assert layout_mod.project_path("ProjA") == tmp_path.resolve() / "ProjA"
        assert layout_mod.list_projects() == ["ProjA", "ProjB"]
    finally:
        layout_mod.set_datasets_root(original, persist=False)
    assert layout_mod.get_datasets_root() == Path(original).resolve()


def test_set_datasets_root_persists_to_config(tmp_path, monkeypatch):
    """With persist=True the chosen root is written to the config file."""
    import functions.layout as layout_mod
    original = layout_mod.get_datasets_root()
    monkeypatch.setattr(layout_mod, "_CONFIG_DIR", tmp_path)
    try:
        root = tmp_path / "store"
        root.mkdir()
        layout_mod.set_datasets_root(root, persist=True)
        assert (tmp_path / "config.json").exists()
        assert layout_mod._load_config().get("datasets_root") == str(root.resolve())
    finally:
        layout_mod.set_datasets_root(original, persist=False)


def test_app_home_env_override(tmp_path, monkeypatch):
    """PEBBLEMAPPER_HOME redirects the app-state directory at call time.

    Feature 023 stores queues and the crash breadcrumb under app_home();
    tests and portable deployments rely on this override, so it must win
    even when set after import and stop winning when unset again.
    """
    import functions.layout as layout_mod
    monkeypatch.setenv("PEBBLEMAPPER_HOME", str(tmp_path / "home"))
    assert layout_mod.app_home() == tmp_path / "home"
    # Config persistence follows the override (dir auto-created).
    layout_mod._save_config({"probe": "x"})
    assert (tmp_path / "home" / "config.json").exists()
    assert layout_mod._load_config().get("probe") == "x"
    monkeypatch.delenv("PEBBLEMAPPER_HOME")
    assert layout_mod.app_home() == layout_mod._CONFIG_DIR


# ---------------------------------------------------------------------------
# Optional per-project date level (multi-temporal)
# ---------------------------------------------------------------------------

def test_ensure_and_list_dates(monkeypatch, tmp_path):
    import functions.layout as layout_mod
    monkeypatch.setattr(layout_mod, "DATASETS_ROOT", tmp_path)
    layout_mod.ensure_project_layout("Multi", date="2023-08-15")
    layout_mod.ensure_project_layout("Multi", date="2024-06-01")
    assert (tmp_path / "Multi" / "2023-08-15" / "output_results" / "vectors").is_dir()
    assert layout_mod.list_dates("Multi") == ["2023-08-15", "2024-06-01"]
    assert layout_mod.is_dated_project("Multi") is True
    # A flat project has no dates and is not "dated".
    layout_mod.ensure_project_layout("Flat")
    assert layout_mod.list_dates("Flat") == []
    assert layout_mod.is_dated_project("Flat") is False


def test_active_date_scoped_to_project(monkeypatch, tmp_path):
    import functions.layout as layout_mod
    monkeypatch.setattr(layout_mod, "DATASETS_ROOT", tmp_path)
    try:
        layout_mod.set_active_date("2024-06-01", "Multi")
        assert (layout_mod.project_path("Multi", "vectors")
                == tmp_path / "Multi" / "2024-06-01" / "output_results" / "vectors")
        # A different project is unaffected by Multi's active date.
        assert (layout_mod.project_path("Flat", "vectors")
                == tmp_path / "Flat" / "output_results" / "vectors")
        # An explicit date arg overrides the active date.
        assert (layout_mod.project_path("Multi", "rasters", date="2023-08-15")
                == tmp_path / "Multi" / "2023-08-15" / "output_results" / "rasters")
        # Clearing returns flat resolution.
        layout_mod.set_active_date(None, None)
        assert layout_mod.project_path("Multi") == tmp_path / "Multi"
    finally:
        layout_mod.set_active_date(None, None)


def test_invalid_date_rejected(monkeypatch, tmp_path):
    import functions.layout as layout_mod
    monkeypatch.setattr(layout_mod, "DATASETS_ROOT", tmp_path)
    for bad in ["../evil", "2023/08", "notadate"]:
        with pytest.raises(ValueError):
            layout_mod.ensure_project_layout("P", date=bad)


# ---------------------------------------------------------------------------
# Validation structure migration (best-effort)
# ---------------------------------------------------------------------------

def test_migrate_validation_moves_only_georectified_tifs(monkeypatch, tmp_path):
    from osgeo import gdal, osr
    import functions.layout as layout_mod
    monkeypatch.setattr(layout_mod, "DATASETS_ROOT", tmp_path)
    layout_mod.ensure_project_layout("P")

    # The legacy flat validation bucket isn't created for new projects; make it
    # to simulate an older project whose files need reorganising.
    images = tmp_path / "P" / "validation" / "images"
    images.mkdir(parents=True, exist_ok=True)
    # A georeferenced GeoTIFF -> should migrate to validation/georectified.
    geotif = images / "survey.tif"
    ds = gdal.GetDriverByName("GTiff").Create(str(geotif), 8, 8, 1)
    ds.SetGeoTransform([100.0, 0.5, 0, 110.0, 0, -0.5])
    srs = osr.SpatialReference(); srs.ImportFromEPSG(32630)
    ds.SetProjection(srs.ExportToWkt())
    ds = None
    # A plain photo -> ambiguous, must be left in place.
    photo = images / "DSC_0001.jpg"
    photo.write_bytes(b"\xff\xd8\xff\xe0not-a-real-jpeg")

    # Dry-run reports the move but changes nothing.
    actions = layout_mod.migrate_validation_structure("P", dry_run=True)
    assert any("survey.tif" in a and "would move" in a for a in actions)
    assert geotif.exists()

    # Real run moves only the GeoTIFF.
    layout_mod.migrate_validation_structure("P", dry_run=False)
    assert not geotif.exists()
    assert (tmp_path / "P" / "validation" / "georectified" / "survey.tif").exists()
    assert photo.exists()  # plain photo left for manual sorting


# ---------------------------------------------------------------------------
# Inputs grouped under input_data/ — resolution fallback + migration
# ---------------------------------------------------------------------------

def test_images_kind_resolves_with_fallback(monkeypatch, tmp_path):
    """project_path('images'/'dem') returns the first historical location that
    exists, so older projects keep working; new projects get the canonical."""
    import functions.layout as layout_mod
    monkeypatch.setattr(layout_mod, "DATASETS_ROOT", tmp_path)
    # New project → canonical input_data/images.
    layout_mod.ensure_project_layout("New")
    assert (layout_mod.project_path("New", "images")
            == tmp_path / "New" / "input_data" / "images")
    # Old-canonical: imagery sat directly in input_data/ (no images subfolder).
    (tmp_path / "OldCanon" / "input_data").mkdir(parents=True)
    assert (layout_mod.project_path("OldCanon", "images")
            == tmp_path / "OldCanon" / "input_data")
    # Oldest-legacy: an images/ folder.
    (tmp_path / "Legacy" / "images").mkdir(parents=True)
    assert (layout_mod.project_path("Legacy", "images")
            == tmp_path / "Legacy" / "images")
    # DEM: a project created with the previous top-level dem/.
    (tmp_path / "OldDem" / "dem").mkdir(parents=True)
    assert (layout_mod.project_path("OldDem", "dem")
            == tmp_path / "OldDem" / "dem")


def test_migrate_input_structure(monkeypatch, tmp_path):
    import functions.layout as layout_mod
    monkeypatch.setattr(layout_mod, "DATASETS_ROOT", tmp_path)
    layout_mod.ensure_project_layout("P")
    in_dir = tmp_path / "P" / "input_data"
    # Loose imagery + an ROI sidecar directly in input_data/.
    (in_dir / "ortho.tif").write_bytes(b"x")
    (in_dir / "ortho_roi.geojson").write_text("{}", encoding="utf-8")
    # A previous top-level dem/ folder.
    (tmp_path / "P" / "dem").mkdir(parents=True, exist_ok=True)
    (tmp_path / "P" / "dem" / "elev.tif").write_bytes(b"x")

    # Dry-run reports moves, changes nothing.
    actions = layout_mod.migrate_input_structure("P", dry_run=True)
    assert any("ortho.tif" in a for a in actions)
    assert (in_dir / "ortho.tif").exists()

    layout_mod.migrate_input_structure("P", dry_run=False)
    assert (in_dir / "images" / "ortho.tif").exists()
    assert (in_dir / "geometries" / "ortho_roi.geojson").exists()
    assert (in_dir / "dem" / "elev.tif").exists()
    assert not (in_dir / "ortho.tif").exists()
    assert (tmp_path / "P" / "README.txt").exists()  # README untouched


# --------------------------------------------------------------------------- #
#  The project-tree contract                                                   #
# --------------------------------------------------------------------------- #
#  The golden. Captures how every kind resolves, for a project in the
#  canonical layout and a project in the legacy layout, so any resolver change
#  can be checked against it.
#
#  Every test below redirects DATASETS_ROOT into tmp_path. None may touch the
#  real datasets/: it is gitignored field data, absent on a fresh clone, and
#  writing to it is how a phantom project was once created.
# --------------------------------------------------------------------------- #
from functions import layout as _layout_018


def _canonical_project(tmp_path):
    """A project on disk in the canonical layout, all kinds present."""
    root = tmp_path / "canonical_proj"
    for sub in set(_layout_018._KIND_SUBPATHS.values()):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def _legacy_project(tmp_path):
    """A project on disk in the legacy layout, all kinds present."""
    root = tmp_path / "legacy_proj"
    for sub in set(_layout_018._LEGACY_KIND_SUBPATHS.values()):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


@pytest.mark.parametrize("kind", sorted(_layout_018._KIND_SUBPATHS))
def test_golden_canonical_layout_resolution_is_unchanged(kind, tmp_path):
    """Every kind resolves to its canonical subpath when that exists."""
    root = _canonical_project(tmp_path)
    got = _layout_018.resolve_project_subfolder(root, kind)
    assert got == root / _layout_018._KIND_SUBPATHS[kind]


@pytest.mark.parametrize("kind", sorted(_layout_018._LEGACY_KIND_SUBPATHS))
def test_golden_legacy_layout_resolution_is_unchanged(kind, tmp_path):
    """A project still in the legacy layout keeps resolving there.

    This is the compatibility guarantee — users have projects on
    disk — and it is the behaviour most at risk from the resolver change.
    """
    root = _legacy_project(tmp_path)
    got = _layout_018.resolve_project_subfolder(root, kind)
    assert got == root / _layout_018._LEGACY_KIND_SUBPATHS[kind]


def test_golden_ten_kinds_have_no_fallback_by_design(tmp_path):
    """10 of the 20 kinds have identical canonical and legacy
    paths, so there is nothing to fall back *to*. Resolution must not read
    that as a failed lookup."""
    identical = [k for k, v in _layout_018._KIND_SUBPATHS.items()
                 if _layout_018._LEGACY_KIND_SUBPATHS.get(k) == v]
    assert len(identical) == 10, identical


# --- An unrecognised directory says so -------------------------------- #

def test_unmigrated_tree_resolves_as_unrecognised(tmp_path):
    """An unmigrated tree is reported as unrecognised, not misresolved.

    A project whose data sits in output_vectors/ (the Marshbeach 2023 shape,
    in neither registry) used to resolve to output_results/vectors, a path
    that does not exist, pointing away from the data. The caller then read
    nothing and reported success.
    """
    root = tmp_path / "unmigrated"
    (root / "output_vectors").mkdir(parents=True)
    (root / "output_vectors" / "clasts.csv").write_text("clast_ID,x,y\n", encoding="utf-8")

    res = _layout_018.resolve_project_subfolder_checked(root, "vectors")
    assert res.unrecognised
    assert not res.found
    # The data is still where it always was; we simply no longer pretend.
    assert (root / "output_vectors" / "clasts.csv").exists()

    msg = _layout_018.describe_unrecognised_layout(root)
    assert str(root) in msg            # names the directory
    assert "Migrate" in msg or "migrate" in msg   # says what to do next


def test_write_resolution_still_returns_the_canonical_target(tmp_path):
    """Write resolution still returns the canonical target.

    Fails if the read fix was implemented by making resolution raise.
    """
    root = tmp_path / "brand_new"
    root.mkdir()
    got = _layout_018.resolve_project_subfolder(root, "vectors")
    assert got == root / "output_results" / "vectors"

    res = _layout_018.resolve_project_subfolder_checked(root, "vectors")
    assert res.path == root / "output_results" / "vectors"
    assert not res.found          # honest: it was guessed
    assert not res.unrecognised   # an empty project is new, not broken


def test_unknown_kind_is_not_an_unrecognised_project(tmp_path):
    """Different faults, different remedies."""
    root = _canonical_project(tmp_path)
    res = _layout_018.resolve_project_subfolder_checked(root, "no_such_kind")
    assert res.problem == "unknown-kind"
    assert not res.unrecognised


def test_recognised_project_with_no_data_is_not_unrecognised(tmp_path):
    """An empty but well-formed project is fine."""
    root = tmp_path / "empty_but_canonical"
    (root / "input_data" / "images").mkdir(parents=True)
    res = _layout_018.resolve_project_subfolder_checked(root, "vectors")
    assert not res.unrecognised
    assert not res.found


# --- Nothing invents a project ---------------------------------------- #

def test_starting_dir_does_not_invent_a_project(monkeypatch, tmp_path):
    """The phantom, reproduced then closed.

    This call once created
    datasets/__nonexistent_project__/output_results/vectors inside the real
    field-data tree. DATASETS_ROOT is redirected into tmp_path here: no test
    may touch the real datasets/ (gitignored, absent on a fresh clone).
    """
    monkeypatch.setattr("functions.layout.DATASETS_ROOT", tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())

    got = default_starting_dir("vectors", "__nonexistent_project__")

    after = sorted(p.name for p in tmp_path.iterdir())
    assert before == after, f"created {set(after) - set(before)}"
    assert not (tmp_path / "__nonexistent_project__").exists()
    # Still returns somewhere usable for a file dialog to open at.
    assert Path(got).exists()


def test_starting_dir_still_creates_for_a_real_project(monkeypatch, tmp_path):
    """The convenience that was explicitly kept.

    A project that exists still gets its missing kind folder created; only
    creation for a project that does *not* exist was the defect.
    """
    monkeypatch.setattr("functions.layout.DATASETS_ROOT", tmp_path)
    (tmp_path / "RealProject").mkdir()

    got = default_starting_dir("vectors", "RealProject")

    assert Path(got).is_dir()
    assert Path(got) == tmp_path / "RealProject" / "output_results" / "vectors"


# --- The written contract must not drift from the registries ---------- #

def test_contract_document_matches_the_live_registries():
    """The contract is generated, so it can drift.

    The resolver's own docstring named two callers that had not called it for
    some time; nobody noticed because nothing checked. This is that check for
    the user manual's "Project folders" appendix: every kind, both
    subpaths, verified against the registries themselves.
    """
    doc = Path(__file__).resolve().parents[1] / "docs" / "user-manual.md"
    assert doc.is_file(), f"user manual missing: {doc}"
    manual = doc.read_text(encoding="utf-8")
    heading = "## Project folders"
    assert heading in manual, f"user manual has no {heading!r} section"
    text = manual.split(heading, 1)[1].split("\n## ", 1)[0]

    for kind, canonical in _layout_018._KIND_SUBPATHS.items():
        legacy = _layout_018._LEGACY_KIND_SUBPATHS[kind]
        row = f"| `{kind}` | `{canonical}` | `{legacy}` |"
        assert row in text, f"contract row wrong or missing for {kind!r}: {row}"

    # All three mechanisms named.
    for mechanism in ("_KIND_SUBPATHS", "_LEGACY_KIND_SUBPATHS",
                      "_INPUT_KIND_CANDIDATES"):
        assert mechanism in text, f"contract does not name {mechanism}"
