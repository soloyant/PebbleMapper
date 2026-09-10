"""The PDF outline's bookmarks open their own pages."""
from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")
pytest.importorskip("reportlab")


def test_outline_entries_resolve_to_their_pages(tmp_path):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import PageBreak, Paragraph
    from functions import report as R
    out = tmp_path / "r.pdf"
    doc = R._PDFDocTemplate.build(out, A4, "P")
    ss = R._styles()
    story = []
    for i, title in enumerate(("Summary", "1. Data", "2. Results", "Appendix A")):
        if i:
            story.append(PageBreak())
        story.append(Paragraph(title, ss["H1Custom"]))
        story.append(Paragraph("Sub", ss["H2Custom"]))
        story.append(Paragraph("body " * 20, ss["Normal"]))
    doc.multiBuild(story, canvasmaker=R._make_numbered_canvas("P"))
    pdf = fitz.open(str(out))
    assert pdf.page_count == 4
    toc = pdf.get_toc(simple=False)
    h1 = [(t, pg, d.get("kind")) for lvl, t, pg, d in toc if lvl == 1]
    assert [pg for _, pg, _ in h1] == [1, 2, 3, 4], h1
    assert all(pg > 0 for _, pg, _ in h1), "destinations must resolve"
    h2 = [pg for lvl, t, pg, d in toc if lvl == 2]
    assert h2 == [1, 2, 3, 4]
    # the footer still counts the pages
    assert "Page 4 of 4" in pdf[3].get_text()
