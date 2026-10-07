"""Report build smoke + figure-structure assertions.

Builds a real PDF from the ``synthetic_project`` fixture via
``functions.report.build_pdf`` and asserts the build's *evidence*:

* the output file exists and is non-trivially large;
* the build log reports a sane inventory — clast total > 0;
* the PDF page count is >= a floor, when a PDF reader is importable
  (skips gracefully otherwise);
* the List of Figures is non-empty and every embedded ``Figure N`` caption
  carries actual caption text (tolerant of the figure COUNT, which may
  change).

Everything heavy (matplotlib / reportlab / a PDF reader) is gated so a
minimal env skips rather than errors. The fixture is self-contained: no
``backups/``, no network, no real datasets.
"""
from __future__ import annotations

import re

import pytest

# build_pdf pulls reportlab + matplotlib; skip the whole module cleanly if the
# report stack is not installed in this env.
pytest.importorskip("reportlab")
pytest.importorskip("matplotlib")

from functions.report import build_pdf  # noqa: E402


# Floor for "non-empty" — a reference dummy report was ~233 KB and
# 15 pages; a few KB is a conservative "something real was written" bar.
_MIN_PDF_BYTES = 4_096
_MIN_PAGES = 5

# Options mirror the reference dummy run:
# spatial_maps + validation OFF (GDAL rasters / validation JSON not in the
# minimal fixture); everything else ON.
_REPORT_OPTIONS = {
    "author": "Test harness",
    "affiliation": "Test suite",
    "description": "Synthetic project built by the test harness to prove the "
                   "report pipeline emits its expected figures and tables.",
    "include_cover": True,
    "include_overview": True,
    "include_methodology": True,
    "include_detection": True,
    "include_data_quality": True,
    "include_spatial_maps": False,
    "include_zonal_statistics": True,
    "include_publication_figures": True,
    "include_validation": False,
    "include_appendix_logs": True,
    "include_appendix_samples": True,
    "include_appendix_field_reference": True,  # auto-skips with no rasters
}


@pytest.fixture
def built_report(synthetic_project):
    """Build the PDF once; yield (out_path, manifest, captured log lines)."""
    manifest = synthetic_project
    out_path = manifest["reports_dir"] / "smoke_report.pdf"
    log_lines: list[str] = []
    build_pdf(manifest["root"], out_path, options=_REPORT_OPTIONS,
              log_fn=log_lines.append)
    return out_path, manifest, log_lines


def _read_pdf_text_and_pages(out_path):
    """Return (full_text, page_count) using whatever PDF reader is importable.

    Tries fitz (PyMuPDF) then
    pdfminer.six. Returns (None, None) if neither is available so callers can
    skip the page-count / text assertions gracefully.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(out_path))
        text = "\n".join(doc[i].get_text() or "" for i in range(doc.page_count))
        return text, doc.page_count
    except Exception:
        pass
    try:
        from pdfminer.high_level import extract_text
        from pdfminer.pdfpage import PDFPage
        text = extract_text(str(out_path))
        with open(str(out_path), "rb") as fh:
            n_pages = sum(1 for _ in PDFPage.get_pages(fh))
        return text, n_pages
    except Exception:
        return None, None


# --------------------------------------------------------------------------- #
#  Requirement 2 — file exists / non-empty / sane inventory / page floor       #
# --------------------------------------------------------------------------- #
def test_report_file_exists_and_nonempty(built_report):
    out_path, _manifest, _log = built_report
    assert out_path.exists(), f"PDF was not written to {out_path}"
    size = out_path.stat().st_size
    assert size > _MIN_PDF_BYTES, (
        f"PDF {out_path} is only {size} bytes (< {_MIN_PDF_BYTES}); "
        "build likely produced an empty/placeholder file")


def test_report_log_reports_sane_inventory(built_report):
    _out_path, manifest, log = built_report
    joined = "\n".join(log)
    # build_pdf logs "  total clasts: <N>". Parse it and
    # assert it is > 0 and matches the fixture's known clast total.
    m = re.search(r"total clasts:\s*([\d,]+)", joined)
    assert m, ("build log did not report a clast total; log was:\n" + joined)
    n_logged = int(m.group(1).replace(",", ""))
    assert n_logged > 0, "report inventory found zero clasts"
    assert n_logged == manifest["n_clasts_total"], (
        f"report counted {n_logged} clasts; fixture wrote "
        f"{manifest['n_clasts_total']}")


def test_report_page_count_floor(built_report):
    out_path, _manifest, _log = built_report
    _text, n_pages = _read_pdf_text_and_pages(out_path)
    if n_pages is None:
        pytest.skip("no PDF reader (fitz / pdfminer) importable in this env")
    assert n_pages >= _MIN_PAGES, (
        f"PDF has {n_pages} pages; expected >= {_MIN_PAGES}")


# --------------------------------------------------------------------------- #
#  Requirement 4 — List of Figures non-empty; every Figure N has a caption     #
# --------------------------------------------------------------------------- #
def test_report_figures_are_numbered_and_captioned(built_report):
    out_path, _manifest, _log = built_report
    text, n_pages = _read_pdf_text_and_pages(out_path)
    if text is None:
        pytest.skip("no PDF reader (fitz / pdfminer) importable in this env")

    # Every embedded figure caption is rendered as "Figure N. <caption text>"
    # (functions/report.py:_fig_caption). Capture the caption text after the
    # number; assert it is non-empty. Structure, not an exact count — C may
    # legitimately change the figure COUNT.
    captions = re.findall(r"Figure\s+(\d+)\.\s*(.*)", text)
    assert captions, (
        "List of Figures is empty — no 'Figure N.' caption found in the PDF; "
        "expected at least one numbered, captioned figure")

    for num, cap in captions:
        assert cap.strip(), (
            f"Figure {num} has an empty caption (guards this regression "
            "where embedded images carry only a raw filename / no caption)")


# --------------------------------------------------------------------------- #
#  One malformed zonal file, a user's cover image                              #
# --------------------------------------------------------------------------- #
def test_a_malformed_zonal_file_costs_itself_not_the_section(synthetic_project, tmp_path):
    """A polygons CSV without polygon_id/count columns brought the whole
    Zonal statistics section down (a table of no columns); it is skipped
    with a note and the other files still render."""
    manifest = synthetic_project
    root = manifest["root"]
    zon = root / "output_results" / "zonal"
    zon.mkdir(parents=True, exist_ok=True)
    (zon / "bad_zones=x_field=y.polygons.csv").write_text("this,is,not\na,zonal,file\n1,2\n", encoding="utf-8")
    log = []
    out_path = tmp_path / "junk_zonal.pdf"
    build_pdf(root, out_path, options=_REPORT_OPTIONS, log_fn=log.append)
    joined = "\n".join(log)
    assert "bad_zones=x_field=y.polygons.csv: no polygon_id or count column" in joined
    assert "Zonal statistics could not be rendered" not in joined
    text, _n = _read_pdf_text_and_pages(out_path)
    if text is None:
        pytest.skip("no PDF reader importable in this env")
    assert "not a polygon statistics table" in text
    assert "Zonal statistics" in text


def test_a_user_cover_image_is_named_not_captioned_as_the_footprint(synthetic_project, tmp_path):
    """The footprint caption belongs to the report's own overview map; a
    cover the user set showed it too."""
    from PIL import Image
    manifest = synthetic_project
    cover = tmp_path / "my_cover.png"
    Image.new("RGB", (80, 60), (120, 130, 140)).save(cover)
    log = []
    out_path = tmp_path / "cover.pdf"
    build_pdf(manifest["root"], out_path, options=dict(_REPORT_OPTIONS, cover_image=str(cover)),
              log_fn=log.append)
    text, _n = _read_pdf_text_and_pages(out_path)
    if text is None:
        pytest.skip("no PDF reader importable in this env")
    first_page = text.split("Summary")[0]
    assert "my_cover.png" in first_page
    assert "Footprint of the" not in first_page
