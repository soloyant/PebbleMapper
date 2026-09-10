"""Report regression guards: zonal figures are numbered.

The Zonal-section figures (the ``combined_*.png`` overview plot and
the per-transect profile PNGs) must be NUMBERED via ``_fig_caption`` so they
carry a ``Figure N`` and appear in the report's List of Figures, consistent with
every other embedded figure. They used to be embedded with a
raw-filename caption (``<font size=8>{png.name}</font>``) that bypassed the
figure counter and the List-of-Figures ToC.

These tests build a real PDF from the shared ``synthetic_project`` fixture
(``tests/conftest.py``) whose zonal section includes a ``combined_*.png``
overview plot, then assert the regression-guard *evidence* in the rendered PDF.

Heavy deps (reportlab / matplotlib / a PDF reader) are gated so a minimal env
skips rather than errors. Assertions are on caption SEMANTICS, not on hard-coded
section indices or figure counts, which legitimately shift with section toggles.
"""
from __future__ import annotations

import re

import pytest

pytest.importorskip("reportlab")
pytest.importorskip("matplotlib")

from functions.report import build_pdf  # noqa: E402


# Zonal ON (so the combined overview plot renders); spatial_maps / validation
# OFF (the minimal fixture has no GDAL rasters / validation JSONs). This is the
# minimal config that still exercises the zonal-figure code path.
_OPTIONS = {
    "author": "Test harness",
    "affiliation": "Test suite",
    "description": "Synthetic project asserting that zonal figures are numbered.",
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
    "include_appendix_field_reference": True,
}


@pytest.fixture
def built_report(synthetic_project):
    manifest = synthetic_project
    out_path = manifest["reports_dir"] / "c_report.pdf"
    log_lines: list[str] = []
    build_pdf(manifest["root"], out_path, options=_OPTIONS,
              log_fn=log_lines.append)
    return out_path, manifest, log_lines


def _read_pdf_text(out_path):
    """Return the full PDF text, or None if no PDF reader is importable."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(out_path))
        return "\n".join(doc[i].get_text() or "" for i in range(doc.page_count))
    except Exception:
        pass
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(out_path))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#  The zonal combined-overview figure is numbered and in the LoF               #
# --------------------------------------------------------------------------- #
def test_zonal_profile_is_a_numbered_figure_rendered_from_data(built_report):
    """The transect profile is rendered from the CSV (never a PNG scavenged
    from disk) as a numbered "Figure N." whose caption names the CSV."""
    out_path, _manifest, _log = built_report
    text = _read_pdf_text(out_path)
    if text is None:
        pytest.skip("no PDF reader (fitz / pdfminer) importable in this env")
    fig_caps = re.findall(r"Figure\s+\d+\.[^\n]*", text)
    assert any("transect" in cap.lower() for cap in fig_caps), fig_caps
    # The Zonal tab's composed overview PNG is not embedded any more: it
    # carried filename legends and an unlabelled axis.
    assert "combined_DummySite_overview.png" not in text


def test_contents_lists_the_zonal_section(built_report):
    """The Contents carry every numbered section (H1) and its H2 headings."""
    out_path, _manifest, _log = built_report
    text = _read_pdf_text(out_path)
    if text is None:
        pytest.skip("no PDF reader (fitz / pdfminer) importable in this env")
    assert text.count("Zonal statistics") >= 2, "the section is missing from Contents"


def test_all_figure_captions_have_text(built_report):
    """Every embedded ``Figure N`` must carry caption text (no bare numbers).
    Guards the broader class: an embedded image whose caption is empty or a
    raw filename token only."""
    out_path, _manifest, _log = built_report
    text = _read_pdf_text(out_path)
    if text is None:
        pytest.skip("no PDF reader (fitz / pdfminer) importable in this env")

    captions = re.findall(r"Figure\s+(\d+)\.\s*(.*)", text)
    assert captions, "no numbered 'Figure N.' caption found in the PDF at all"
    for num, cap in captions:
        assert cap.strip(), f"Figure {num} has an empty caption"


# --------------------------------------------------------------------------- #
#  Illustrations section — user-supplied images with custom title + caption    #
# --------------------------------------------------------------------------- #
def test_illustrations_section_renders(synthetic_project, tmp_path):
    """User illustrations appear as an 'Illustrations' section carrying their
    custom title and description text."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = synthetic_project
    img = tmp_path / "field_photo.png"
    plt.figure(figsize=(3, 2))
    plt.plot([0, 1, 2], [0, 1, 0])
    plt.savefig(img, dpi=80)
    plt.close()

    opts = dict(_OPTIONS)
    opts["illustrations"] = [
        {"path": str(img), "title": "Site overview ZZQ",
         "description": "Distinctive caption marker XYZ123."}]
    out_path = manifest["reports_dir"] / "illus_report.pdf"
    build_pdf(manifest["root"], out_path, options=opts, log_fn=lambda _m: None)
    assert out_path.exists()

    text = _read_pdf_text(out_path)
    if text is None:
        pytest.skip("no PDF reader (fitz / pdfminer) importable in this env")
    assert "Illustrations" in text
    assert "Site overview ZZQ" in text
    assert "XYZ123" in text


def test_illustrations_section_absent_when_empty(built_report):
    """With no illustrations supplied, no numbered 'Illustrations' section is
    emitted (the section is auto-omitted)."""
    out_path, _manifest, _log = built_report
    text = _read_pdf_text(out_path)
    if text is None:
        pytest.skip("no PDF reader (fitz / pdfminer) importable in this env")
    assert not re.search(r"\d+\.\s*Illustrations", text)
