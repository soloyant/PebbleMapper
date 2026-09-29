"""The PDF report's sections, in reading order.

Every builder appends ReportLab flowables to ``story`` and returns nothing.
They read the project through ``facts`` (functions.report_facts) and format
through the helpers of functions.report (tables, captions, figure embedding),
imported lazily so this module can be loaded without ReportLab.

Section order (a story: what was measured, how, how well, what it means):

    Cover · Summary · Contents
    1 Data and processing   2 Detection results   3 Spatial statistics
    4 Zonal statistics      5 Validation          6 Interpretation and limitations
    References · A Run register · B Data dictionary · C Glossary
"""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape as _xml_escape

import numpy as np

from functions import modes, naming
from functions import units as _u
from functions import report_facts as F

# --- Formatting ---
MINUS = "−"


def _R():
    from functions import report as R
    return R


def _dp_for_gsd(gsd_m: Optional[float]) -> int:
    """Decimals for a length in mm: whole millimetres on coarse imagery,
    a tenth on millimetre-per-pixel photographs."""
    if gsd_m is None:
        return 0
    return 1 if gsd_m < 0.002 else 0


def fmt_mm(v, dp: int = 0, unit: str = "mm") -> str:
    if v is None or not np.isfinite(v):
        return "—"
    s = f"{v:,.{dp}f}".replace("-", MINUS)
    return f"{s} {unit}" if unit else s


def fmt_num(v, dp: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    r = round(float(v), dp)
    if r == 0:
        r = 0.0            # no signed zero
    return f"{r:,.{dp}f}".replace("-", MINUS)


def fmt_signed(v, dp: int = 2, suffix: str = "") -> str:
    if v is None or not np.isfinite(v):
        return "—"
    r = round(float(v), dp)
    if r == 0:
        return f"0.{'0' * dp}{suffix}" if dp else f"0{suffix}"
    sign = "+" if r > 0 else MINUS
    return f"{sign}{abs(r):,.{dp}f}{suffix}"


def fmt_int(v) -> str:
    try:
        return f"{int(round(float(v))):,}"
    except (TypeError, ValueError):
        return "—"


def fmt_pct(frac, dp: int = 0) -> str:
    if frac is None or not np.isfinite(frac):
        return "—"
    return f"{100 * frac:.{dp}f} %"


def fmt_area(m2) -> str:
    if not m2:
        return "—"
    if m2 >= 1000:
        return f"{m2:,.0f} m²"
    if m2 >= 10:
        return f"{m2:.0f} m²"
    return f"{m2:.2f} m²"


def fmt_gsd(gsd_m) -> str:
    if not gsd_m:
        return "—"
    return f"{gsd_m * 1000:.2f} mm/px"


def _elapsed(s) -> str:
    return _R()._format_elapsed(s) if s else "—"


def _plural(n: int, one: str, many: Optional[str] = None) -> str:
    return one if n == 1 else (many or one + "s")


def _para(story, ss, text, style="Body"):
    from reportlab.platypus import Paragraph
    story.append(Paragraph(text, ss[style]))


def _spacer(story, h=6):
    from reportlab.platypus import Spacer
    story.append(Spacer(1, h))


def _keep(story, flowables):
    from reportlab.platypus import KeepTogether
    story.append(KeepTogether(flowables))


def _fig(fig, dpi, quality, max_w=6.5, max_h=8.0):
    return _R()._figure_image(fig, dpi=dpi, jpeg_quality=quality,
                              max_width_in=max_w, max_height_in=max_h)


def _close(fig):
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass


def _figure_block(story, ss, fig, caption, *, dpi, quality, max_w=6.5,
                  max_h=8.0, source="", heading=None):
    """Figure and its numbered caption (and the heading that introduces it),
    kept on one page."""
    from reportlab.platypus import Paragraph
    try:
        img = _fig(fig, dpi, quality, max_w, max_h)
    finally:
        _close(fig)
    parts = [Paragraph(heading, ss["H2Custom"])] if heading else []
    _keep(story, parts + [img, _R()._fig_caption(caption, ss, source=source)])


def _note(story, ss, text):
    _para(story, ss, f"<i>{text}</i>")


def _short_stem(stem: str) -> str:
    """A photograph or ortho name without the tool's tokens."""
    import re
    s = naming.strip_origin(str(stem or ""))
    s = re.sub(r"_(?:height|width)=[0-9.]+m", "", s)
    s = re.sub(r"_GSD=[0-9.]+m(?:_per_px)?", "", s)
    for marker in ("_of_UAV_ortho_image", "_ortho_image", "_rectified",
                   "_corrected", "_georeferenced"):
        s = s.replace(marker, "")
    return s.strip("_")


def _image_label(img: dict) -> str:
    """A short name for an image in prose and tables."""
    return _short_stem(img["stem"]) or img["stem"]


# --- Cover and summary ---
def cover(story, ss, inv, facts, options, log):
    from reportlab.platypus import Paragraph, Spacer, PageBreak, Image as RImage
    from functions import branding as _brand
    R = _R()
    story.append(Spacer(1, 60))
    story.append(Paragraph(f"{_brand.APP_NAME}<br/>Project report", ss["CoverTitle"]))
    story.append(Spacer(1, 14))
    site_line = facts["site"] + (f" — survey of {facts['date']}" if facts["date"] else "")
    story.append(Paragraph(f"<b>{site_line}</b>", ss["CoverMeta"]))
    story.append(Paragraph(f"Project folder: {inv['project_name']}", ss["CoverMeta"]))
    author = options.get("author") or ""
    aff = options.get("affiliation") or ""
    if author:
        story.append(Paragraph(author + (f", {aff}" if aff else ""), ss["CoverMeta"]))
    story.append(Paragraph(datetime.now().strftime("%d %B %Y"), ss["CoverMeta"]))
    story.append(Spacer(1, 18))

    cover_path = None
    cover_is_overview = False
    cover_image = options.get("cover_image")
    if cover_image and Path(cover_image).exists():
        cover_path = Path(cover_image)
    elif facts["n_ortho"]:
        try:
            from functions.layout import resolve_project_subfolder as _rsf
            target = _rsf(Path(inv["project_root"]), "figures") / "_project_overview_map.jpg"
            built = R._build_project_overview_map(inv, target)
            if built:
                cover_path = Path(built)
                cover_is_overview = True
        except Exception as ex:
            log(f"[cover] overview map skipped: {type(ex).__name__}: {ex}")
    elif facts["images"]:
        cover_path = facts["images"][0]["path"]
    if cover_path and cover_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        try:
            from reportlab.platypus import Table, TableStyle
            img = RImage(str(cover_path), width=6.5 * 72, height=4.6 * 72, kind="proportional")
            t = Table([[img]], colWidths=[7.4 * 72])
            t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
            story.append(t)
            # The footprint caption belongs to the overview map; a cover the
            # user chose is named instead.
            if cover_is_overview and facts["n_ortho"]:
                story.append(Paragraph(
                    f"<font size=8 color='#555'>Footprint of the "
                    f"{facts['n_ortho']} ortho-{_plural(facts['n_ortho'], 'image')}"
                    f"{' · ' + facts['images'][0]['crs'] if facts['images'][0].get('crs') else ''}."
                    f"</font>", ss["CoverMeta"]))
            elif not cover_is_overview:
                story.append(Paragraph(
                    f"<font size=8 color='#555'>{_xml_escape(cover_path.name)}</font>",
                    ss["CoverMeta"]))
        except Exception as ex:
            log(f"[cover] image not embedded: {type(ex).__name__}: {ex}")
    desc = options.get("description") or ""
    if desc:
        story.append(Spacer(1, 10))
        story.append(Paragraph(desc, ss["Body"]))
    # Build identity, small, at the foot.
    try:
        from functions import __version__ as _ver
    except Exception:
        _ver = "dev"
    preset = R.REPORT_PRESETS.get(str(options.get("preset", "full")).lower(),
                                  R.REPORT_PRESETS["full"])["label"]
    story.append(Spacer(1, 24))
    story.append(Paragraph(
        f"<font size=7 color='#7a8694'>Built with {_brand.APP_NAME} v{_ver} · "
        f"preset {preset} · {datetime.now():%Y-%m-%d %H:%M}</font>", ss["CoverMeta"]))
    story.append(PageBreak())




def _summary_paragraph(facts) -> str:
    kind = F.headline_kind(facts)
    imgs = [i for i in facts["images"] if i["kind"] == kind and i["n_clasts"]]
    if not imgs:
        return ("No detection output was found in this project. The sections "
                "below describe the inputs that are present; run the Detection "
                "tab and rebuild the report.")
    n_total = sum(i["n_clasts"] for i in imgs)
    area = sum(i["footprint_m2"] or 0 for i in imgs)
    what = ("UAV ortho-image" if kind == modes.ORTHO else "scaled quadrat photograph")
    parts = []
    parts.append(
        f"{fmt_int(n_total)} clasts were detected and measured on "
        f"{len(imgs)} {_plural(len(imgs), what)}"
        + (f" covering {fmt_area(area)}" if area else "") + ".")
    per = []
    for i in imgs:
        L = i["length_mm"]
        if not L:
            continue
        dp = _dp_for_gsd(i["gsd_m"])
        per.append(f"{_image_label(i)}: D50 {fmt_mm(L.get('D50'), dp)}, "
                   f"D84 {fmt_mm(L.get('D84'), dp)}, n = {fmt_int(L['n'])}")
    if per:
        parts.append("Median and 84th-percentile apparent a-axis length — " + "; ".join(per) + ".")
    fws = [i["folk_ward"] for i in imgs if i["folk_ward"]]
    if fws:
        s = [f["sorting_phi"] for f in fws]
        verbal = sorted({f["sorting_verbal"] for f in fws})
        parts.append(
            f"Folk and Ward sorting of the resolved fraction is "
            f"{fmt_num(min(s))}–{fmt_num(max(s))} φ ({', '.join(verbal)})."
            if len(fws) > 1 else
            f"Folk and Ward sorting of the resolved fraction is "
            f"{fmt_num(s[0])} φ ({verbal[0]}).")
    dmins = [i["dmin_m"] for i in imgs if i["dmin_m"]]
    if not dmins and any(i.get("image_missing") for i in imgs):
        parts.append("The imagery is not in the project, so its resolution and "
                     "detection limit are unknown; every percentile describes the "
                     "population the detector resolved, by number, not by weight.")
    if dmins:
        lo, hi = min(dmins) * 1000, max(dmins) * 1000
        rng = (f"{lo:.0f} mm" if abs(hi - lo) < 0.5 else f"{lo:.0f}–{hi:.0f} mm")
        parts.append(
            f"Clasts shorter than about {rng} ({F.DETECTION_LIMIT_PX} pixels) are "
            f"below the detection limit of this imagery, so every percentile and "
            f"moment describes the population above that size, by number of "
            f"clasts, not by weight.")
    small = [i for i in imgs if i["length_mm"] and i["length_mm"]["n"] < 100]
    if small:
        parts.append(
            f"With {' and '.join(fmt_int(i['length_mm']['n']) for i in small)} clasts, "
            f"the percentiles carry wide intervals (section 2) and D95 rests on a "
            f"handful of clasts.")
    usable = [v for v in facts["validations"] if v["usable"]]
    flagged = [v for v in facts["validations"] if not v["usable"]]
    if usable:
        n_pairs = sum(v["n_pairs"] for v in usable)
        kinds = sorted({v["detect_kind"] for v in usable})
        parts.append(
            f"{len(usable)} ground-truth {_plural(len(usable), 'comparison')} "
            f"({fmt_int(n_pairs)} paired clasts) "
            f"{'validates' if len(usable) == 1 else 'validate'} the "
            f"{' and '.join(kinds)} detections; see section 5.")
        if kind == modes.ORTHO and modes.ORTHO not in kinds:
            parts.append("No ground truth exists for the ortho-images "
                         "themselves: their absolute values are unvalidated.")
    elif flagged:
        parts.append(
            f"No usable validation: the {len(flagged)} saved "
            f"{_plural(len(flagged), 'comparison')} "
            f"{'is' if len(flagged) == 1 else 'are'} degenerate "
            f"({'; '.join(sorted({f for v in flagged for f in v['flags']}))}). "
            f"The detections in this project are unvalidated.")
    else:
        parts.append("No ground truth is present: the detections in this "
                     "project are unvalidated.")
    return " ".join(parts)


def summary(story, ss, inv, stats, facts, options, log, *, dpi, quality):
    from reportlab.platypus import PageBreak
    R = _R()
    _para(story, ss, "Summary", "H1Custom")
    _para(story, ss, _summary_paragraph(facts))
    _spacer(story, 8)

    kind = F.headline_kind(facts)
    imgs = [i for i in facts["images"] if i["kind"] == kind]
    rows = [["Item", "Value"]]
    n_img = len(facts["images"])
    kinds_txt = ", ".join(
        t for t in (
            f"{facts['n_ortho']} UAV ortho-{_plural(facts['n_ortho'], 'image')}" if facts["n_ortho"] else "",
            f"{facts['n_quadrat']} quadrat {_plural(facts['n_quadrat'], 'photograph')}"
            if facts["n_quadrat"] else "") if t)
    rows.append(["Images", kinds_txt or "none"])
    areas = [i["footprint_m2"] for i in imgs if i["footprint_m2"]]
    if areas:
        rows.append(["Imaged area", fmt_area(sum(areas)) + (
            f" ({', '.join(fmt_area(a) for a in areas)})" if len(areas) > 1 else "")])
    gsds = [i["gsd_m"] for i in imgs if i["gsd_m"]]
    if gsds:
        rows.append(["Ground sample distance",
                     " / ".join(fmt_gsd(g) for g in gsds)])
        rows.append([f"Detection limit ({F.DETECTION_LIMIT_PX} px)",
                     " / ".join(f"{F.DETECTION_LIMIT_PX * g * 1000:.0f} mm" for g in gsds)])
    counted = [i for i in imgs if i["n_clasts"]]
    if counted:
        rows.append(["Clasts measured", fmt_int(sum(i["n_clasts"] for i in counted)) + (
            " (" + ", ".join(f"{_image_label(i)} {fmt_int(i['n_clasts'])}" for i in counted) + ")"
            if len(counted) > 1 else "")])
        for i in counted:
            L = i["length_mm"]
            if not L:
                continue
            dp = _dp_for_gsd(i["gsd_m"])
            ci = i.get("d50_ci_mm")
            ci_txt = f" (95 % CI {fmt_mm(ci[0], dp, '')}–{fmt_mm(ci[1], dp)})" if ci else ""
            label = "D50 / D84 / D95" + (f" — {_image_label(i)}" if len(counted) > 1 else "")
            rows.append([label, f"{fmt_mm(L['D50'], dp)}{ci_txt} / "
                                f"{fmt_mm(L['D84'], dp)} / {fmt_mm(L['D95'], dp)}"])
        for i in counted:
            fw = i["folk_ward"]
            if not fw:
                continue
            label = "Sorting σφ, skewness Skφ, kurtosis K<sub>G</sub>" + (
                f" — {_image_label(i)}" if len(counted) > 1 else "")
            rows.append([label, f"{fmt_num(fw['sorting_phi'])} φ ({fw['sorting_verbal']}), "
                                f"{fmt_signed(fw['skewness_phi'])} ({fw['skewness_verbal']}), "
                                f"{fmt_num(fw['kurtosis_phi'])} ({fw['kurtosis_verbal']})"])
        dens = [i for i in counted if i["density_per_m2"]]
        if dens:
            rows.append(["Clast density", " / ".join(
                f"{i['density_per_m2']:.1f} per m²" for i in dens)])
        cov = [i for i in counted if i["areal_cover"] is not None]
        if cov:
            rows.append(["Detected areal cover", " / ".join(
                fmt_pct(i["areal_cover"]) for i in cov)])
    usable = [v for v in facts["validations"] if v["usable"]]
    if usable:
        pairs = sum(v["n_pairs"] for v in usable)
        rows.append(["Validation", f"{len(usable)} usable "
                     f"{_plural(len(usable), 'comparison')}, {fmt_int(pairs)} paired "
                     f"clasts (section 5)"])
    elif facts["validations"]:
        rows.append(["Validation", "none usable — see section 5"])
    else:
        rows.append(["Validation", "none (no ground truth in the project)"])
    from reportlab.platypus import Paragraph
    rows = [[Paragraph(str(a), ss["Body"]), Paragraph(str(b), ss["Body"])]
            if k else [a, b] for k, (a, b) in enumerate(rows)]
    story.append(R._tbl_caption("Headline values. Lengths are apparent a-axes "
                                "(mask major-axis chords) of the population above "
                                "the detection limit, by number.", ss))
    story.append(R._table(rows, col_widths=[2.4 * 72, 4.3 * 72], font_size=9))
    _spacer(story, 10)

    L = F.pooled_lengths(facts, kind)
    if L.size >= 5:
        try:
            from functions import report_figures as RF
            dmin = None
            dmins = [i["dmin_m"] for i in imgs if i["dmin_m"]]
            if dmins:
                dmin = max(dmins) * 1000.0
            fig = RF.histogram_adaptive(L, dmin_mm=dmin)
            meta = getattr(fig, "pm_meta", {}) or {}
            fit = ""
            if meta.get("mu_ln") is not None:
                fit = (f" Log-normal fit: μ<sub>ln</sub> = {meta['mu_ln']:.2f}, "
                       f"σ<sub>ln</sub> = {meta['sigma_ln']:.2f}.")
            q = F._quantiles_mm(L)
            _figure_block(
                story, ss, fig,
                f"Apparent a-axis length of the {fmt_int(q['n'])} "
                f"{'ortho' if kind == modes.ORTHO else 'quadrat'} clasts (log axis, "
                f"φ scale on top). Histogram with kernel density and "
                f"log-normal fit; D50 = {fmt_mm(q['D50'])}, D84 = {fmt_mm(q['D84'])}, "
                f"D95 = {fmt_mm(q['D95'])}.{fit}"
                + (" The grey band marks sizes below the detection limit, "
                   "where the histogram is a property of the imagery, not of "
                   "the beach." if dmin else ""),
                dpi=dpi, quality=quality, max_h=3.3)
        except Exception as ex:
            log(f"[summary] distribution figure skipped: {type(ex).__name__}: {ex}")
    story.append(PageBreak())


# --- 1  Data and processing ---
def data_and_processing(story, ss, inv, facts, options, log, section):
    from reportlab.platypus import Paragraph, PageBreak
    R = _R()
    _para(story, ss, f"{section}. Data and processing", "H1Custom")

    # 1.1 Inputs
    _para(story, ss, f"{section}.1 Inputs", "H2Custom")
    rows = [["Image", "Kind", "Size (px)", "GSD", "Footprint", "CRS", "Detection limit"]]
    for i in facts["images"]:
        rows.append([
            Paragraph(_image_label(i) + (" (image not in the project)"
                                         if i.get("image_missing") else ""), ss["Body"]),
            "UAV ortho-image" if i["kind"] == modes.ORTHO else "quadrat photograph",
            f"{fmt_int(i['width_px'])} × {fmt_int(i['height_px'])}" if i["width_px"] else "—",
            fmt_gsd(i["gsd_m"]),
            (f"{i['width_m']:.1f} × {i['height_m']:.1f} m ({fmt_area(i['footprint_m2'])}"
             f"{', clast bounding box' if i.get('footprint_kind') == 'bounding box' else ''})"
             if i["footprint_m2"] else "—"),
            Paragraph(i["crs"] or ("image pixels" if i["kind"] == modes.QUADRAT else "—"),
                      ss["Body"]),
            f"{i['dmin_m'] * 1000:.0f} mm" if i["dmin_m"] else "—",
        ])
    if len(rows) > 1:
        story.append(R._tbl_caption(
            "Input imagery. The detection limit is "
            f"{F.DETECTION_LIMIT_PX} pixels × GSD, the shortest clast the "
            "detector segments reliably (Soloy et al. 2020).", ss))
        story.append(R._table(rows, font_size=7.5,
                              col_widths=[1.2 * 72, 1.0 * 72, 0.9 * 72, 0.8 * 72,
                                          1.4 * 72, 0.9 * 72, 0.8 * 72]))
    else:
        _note(story, ss, "No images were found in the project's input folder.")
    if facts["date"] is None:
        _para(story, ss, "<font size=8 color='#555'>Acquisition date, platform, "
                         "flight height and georeferencing accuracy are not "
                         "recorded in the project; add them to the report "
                         "description if known.</font>")
    _spacer(story, 8)

    # 1.2 Method
    _para(story, ss, f"{section}.2 Method", "H2Custom")
    p = facts["params"]
    windows = ", ".join(f"{w:g} m" for w in p["windows_m"]) or "—"
    overlap = ", ".join(p["overlap"]) or "—"
    dedup = ", ".join(p["dedup"]) or "—"
    conf = ", ".join(p["min_confidence"]) or "—"
    texts = []
    if facts["n_ortho"]:
        texts.append(
            f"Each ortho-image was tiled at {len(p['windows_m']) or 'one'} window "
            f"{_plural(len(p['windows_m']) or 1, 'size')} ({windows}; overlap {overlap}) "
            f"and every tile was processed by the Mask R-CNN instance segmentation "
            f"network (He et al. 2017) trained on pebble-beach imagery (Soloy et al. "
            f"2020). Detections from overlapping tiles and windows were pooled and "
            f"duplicates removed where masks overlap by more than the IoU threshold "
            f"({dedup}); the result is one merged clast list per image. A larger "
            f"window is resampled to the network input, so it resolves larger "
            f"clasts at a coarser effective GSD; the merge keeps the largest "
            f"population.")
    if facts["n_quadrat"]:
        texts.append(
            "Quadrat photographs were rectified to a known frame size and "
            "processed as one image each; coordinates are image pixels scaled by "
            "the GSD recorded at rectification.")
    texts.append(
        "For each mask an ellipse is fitted to the outline to give the "
        "orientation; <b>clast length</b> and <b>clast width</b> are the chords "
        "of the mask through its centroid along the ellipse's major and minor "
        "axes, i.e. the apparent a- and b-axes in plan view. Area is the mask "
        "pixel count × GSD², the equivalent diameter is 2√(A/π), circularity is "
        "the equivalent diameter divided by the length and elongation the width "
        "divided by the length. Partly buried, overlapping or shadowed clasts are "
        "under-measured; the population is an areal-by-number sample of what "
        "the detector segments, not a grid-by-number (Wolman) or sieve-by-weight "
        "sample, and the two are not interchangeable without conversion "
        "(Kellerhals and Bray 1971).")
    if p["min_confidence"] and p["min_confidence"] != ["—"]:
        texts.append(f"Detections below a confidence of {conf} were discarded.")
    for t in texts:
        _para(story, ss, t)
    if facts["rasters_by_image"]:
        _para(story, ss,
              "Per-cell surfaces were computed on a regular grid from the merged "
              "clast list: percentiles and mean of the length distribution, its "
              "standard deviation and Folk and Ward moments where enough clasts "
              "fall in a cell, and the fraction of the cell covered by detected "
              "clasts of each size class. Cells are assigned by clast centroid.")
    _spacer(story, 8)

    # 1.3 Software
    _para(story, ss, f"{section}.3 Software", "H2Custom")
    import sys
    rows = [["Item", "Value"], ["Python", sys.version.split()[0]]]
    for mod, label, attr in (("tensorflow", "TensorFlow", "__version__"),
                             ("numpy", "NumPy", "__version__"),
                             ("pandas", "pandas", "__version__"),
                             ("scipy", "SciPy", "__version__"),
                             ("matplotlib", "matplotlib", "__version__"),
                             ("reportlab", "ReportLab", "Version")):
        try:
            import importlib
            rows.append([label, str(getattr(importlib.import_module(mod), attr))])
        except Exception:
            pass
    try:
        from osgeo import gdal as _gdal
        rows.append(["GDAL", _gdal.__version__])
    except Exception:
        pass
    w = facts["weights"]
    if w.get("path"):
        rows.append(["Detector weights", Path(w["path"]).name])
        try:
            import hashlib
            sha = hashlib.sha256(Path(w["path"]).read_bytes()).hexdigest()
            rows.append(["Weights SHA-256", Paragraph(f"<font size=7>{sha}</font>", ss["Body"])])
        except Exception:
            rows.append(["Weights SHA-256 (prefix)", w.get("hash") or "—"])
        rows.append(["Weights size", R._format_bytes(w.get("size_bytes") or 0)])
    try:
        from functions import __version__ as _ver
    except Exception:
        _ver = "dev"
    rows.append(["PebbleMapper", str(_ver)])
    rows.append(["Report generated", datetime.now().isoformat(timespec="minutes")])
    story.append(R._tbl_caption("Software versions and detector weights used for "
                                "the numbers in this report.", ss))
    story.append(R._table(rows, col_widths=[2.2 * 72, 4.5 * 72]))
    story.append(PageBreak())


# --- 2  Detection results ---
def detection_results(story, ss, inv, facts, options, log, section, *, dpi, quality):
    from reportlab.platypus import Paragraph, PageBreak
    R = _R()
    imgs = [i for i in facts["images"] if i["runs"] or i["n_clasts"]]
    _para(story, ss, f"{section}. Detection results", "H1Custom")
    if not imgs:
        _note(story, ss, "No detection CSV was found in the project.")
        story.append(PageBreak())
        return

    # 2.1 Run register
    _para(story, ss, f"{section}.1 Runs", "H2Custom")
    rows = [["Image", "Run", "Overlap", "Tiles", "Clasts", "Status", "Elapsed"]]
    notes = []
    for i in imgs:
        for r in i["runs"]:
            if r["merged"]:
                run = "merged"
            elif r["window_m"] is not None:
                run = f"{r['window_m']:g} m window"
            else:
                run = "full image"
            status = "counted" if r["counted"] else (
                "superseded by merge" if not r["merged"] and r["window_m"] is not None
                and i["canonical"] and i["canonical"] != r["path"]
                else "duplicate" if r["merged"] else "—")
            el = _elapsed(r["elapsed_s"]) if r["elapsed_s"] else "—"
            if r["n_aborted"]:
                el += f" (+{r['n_aborted']} aborted)"
            rows.append([Paragraph(_image_label(i), ss["Body"]), run,
                         r["overlap"] or "—", fmt_int(r["tiles"]) if r["tiles"] else "—",
                         fmt_int(r["n_clasts"]), status, el])
        for w, ns in i["repeatability"]:
            lo, hi = min(ns), max(ns)
            if lo:
                notes.append(f"{_image_label(i)} at {w:g} m ran {len(ns)} times "
                             f"({', '.join(fmt_int(n) for n in ns)} clasts): run-to-run "
                             f"repeatability ±{100 * (hi - lo) / (2 * lo):.1f} %.")
        mv = i["merged_variants"]
        if len(mv) > 1:
            counts = sorted({m["n_clasts"] for m in mv})
            if len(counts) > 1:
                notes.append(f"{_image_label(i)} carries {len(mv)} merged files that "
                             f"differ by {max(counts) - min(counts)} clasts; the "
                             f"{fmt_int(next(m['n_clasts'] for m in mv if m['counted']))}"
                             f"-clast file is the one counted throughout.")
    story.append(R._tbl_caption(
        "Detection runs. A counted run feeds every number in this report; a "
        "window run superseded by a merge and a duplicate merged export are "
        "listed for completeness. Elapsed is the completed run only.", ss))
    story.append(R._table(rows, font_size=7.5,
                          col_widths=[1.4 * 72, 1.1 * 72, 0.7 * 72, 0.6 * 72,
                                      0.8 * 72, 1.3 * 72, 1.0 * 72]))
    if notes:
        _para(story, ss, "<font size=8>" + " ".join(notes) + "</font>")
    _spacer(story, 8)

    # 2.2 What a detection looks like
    sub = 2
    overlays = 0
    try:
        from functions import report_figures as RF
    except Exception as ex:
        RF = None
        log(f"[detection] figure module unavailable: {ex}")
    if RF is not None:
        for i in imgs:
            df = i.get("clasts")
            if df is None or not len(df) or not (i.get("detect_image") or i.get("path")):
                continue
            try:
                fig = RF.detection_overlay(
                    i.get("detect_image") or i["path"], df, gsd_m=i["gsd_m"],
                    transform=i.get("transform"),
                    window_m=2.0 if i["kind"] == modes.ORTHO else None,
                    csv_path=i.get("canonical"))
            except Exception as ex:
                log(f"[detection] overlay for {i['name']} skipped: {type(ex).__name__}: {ex}")
                continue
            meta = getattr(fig, "pm_meta", {}) or {}
            _figure_block(
                story, ss, fig,
                f"Detected clasts on {_image_label(i)}"
                + (f" ({meta.get('window_m'):g} m window at the densest location)"
                   if meta.get("window_m") else "")
                + f": {fmt_int(meta.get('n_shown', 0))} detected clasts as contours "
                  f"with their major and minor axes"
                + ("" if meta.get("n_outlines") else
                   " (this CSV has no contour file, so only the axes are drawn)")
                + ". Clasts that are partly buried, touching or in shadow are "
                  "the ones typically missed.",
                dpi=dpi, quality=quality, max_h=4.6,
                heading=(f"{section}.{sub} Detections on the imagery"
                         if overlays == 0 else None))
            overlays += 1
        if overlays:
            sub += 1

    # 2.3 Size distributions
    counted = [i for i in imgs if i["length_mm"]]
    if counted:
        _para(story, ss, f"{section}.{sub} Size distributions", "H2Custom")
        header = ["Statistic"] + [_image_label(i) for i in counted]
        rows = [header]

        def _row(label, fn):
            rows.append([label] + [fn(i) for i in counted])

        _row("Clasts (n)", lambda i: fmt_int(i["length_mm"]["n"]))
        _row("Below the detection limit (kept, flagged)",
             lambda i: (f"{fmt_int(i.get('n_below_dmin', 0))} "
                        f"(< {i['dmin_m'] * 1000:.0f} mm)" if i["dmin_m"] else "—"))
        for q in ("D5", "D16", "D50", "D84", "D95"):
            def _q(i, q=q):
                dp = _dp_for_gsd(i["gsd_m"])
                v = i["length_mm"].get(q)
                s = fmt_mm(v, dp)
                if i["dmin_m"] and v is not None and v < 1.5 * i["dmin_m"] * 1000:
                    s += " †"
                return s
            _row(q + (" (median)" if q == "D50" else ""), _q)
        _row("D50 95 % interval (bootstrap)",
             lambda i: (f"{fmt_mm(i['d50_ci_mm'][0], _dp_for_gsd(i['gsd_m']), '')}–"
                        f"{fmt_mm(i['d50_ci_mm'][1], _dp_for_gsd(i['gsd_m']))}"
                        if i["d50_ci_mm"] else "—"))
        _row("Mean ± SD", lambda i: f"{fmt_mm(i['length_mm']['mean'], _dp_for_gsd(i['gsd_m']), '')} ± "
                                    f"{fmt_mm(i['length_mm']['std'], _dp_for_gsd(i['gsd_m']))}")
        _row("Width D50 (apparent b-axis)",
             lambda i: fmt_mm(i["width_mm"].get("D50"), _dp_for_gsd(i["gsd_m"])) if i["width_mm"] else "—")
        _row("Sorting σφ", lambda i: (f"{fmt_num(i['folk_ward']['sorting_phi'])} φ<br/>"
                                      f"{i['folk_ward']['sorting_verbal']}") if i["folk_ward"] else "—")
        _row("Skewness Skφ", lambda i: (f"{fmt_signed(i['folk_ward']['skewness_phi'])}<br/>"
                                        f"{i['folk_ward']['skewness_verbal']}") if i["folk_ward"] else "—")
        _row("Kurtosis K<sub>G</sub>", lambda i: (f"{fmt_num(i['folk_ward']['kurtosis_phi'])}<br/>"
                                                  f"{i['folk_ward']['kurtosis_verbal']}") if i["folk_ward"] else "—")
        _row("Clast density", lambda i: f"{i['density_per_m2']:.1f} per m²" if i["density_per_m2"] else "—")
        _row("Detected areal cover", lambda i: fmt_pct(i["areal_cover"]) if i["areal_cover"] is not None else "—")
        ori = [i for i in counted if i["orientation"] and np.isfinite(i["orientation"].get("R", float("nan")))]
        if ori:
            _row("Orientation: axial mean, resultant length <i>R</i>",
                 lambda i: (f"{i['orientation']['mean_deg']:.0f}°, "
                            f"<i>R</i> = {i['orientation']['R']:.2f}") if i["orientation"] else "—")
        rows = [[Paragraph(str(c), ss["Body"]) if k == 0 else c for k, c in enumerate(r)]
                for r in rows]
        # One column per image is narrow: every value wraps (a verbal Folk and
        # Ward class does not fit on one line), and an image name, which has
        # no spaces, may break anywhere.
        from reportlab.lib.styles import ParagraphStyle as _PS
        _hdr = _PS("_StatsHeader", parent=ss["Body"], fontSize=7, leading=9,
                   wordWrap="CJK", spaceBefore=0, spaceAfter=0)
        rows[0] = [rows[0][0]] + [Paragraph(str(c), _hdr) for c in rows[0][1:]]
        story.append(R._tbl_caption(
            "Apparent a-axis statistics per image, by number of clasts. "
            "† within 1.5 × the detection limit: censored by the imagery, not a "
            "property of the deposit. Folk and Ward moments are graphic "
            "measures on the φ scale (φ = −log<sub>2</sub> D<sub>mm</sub>) of the "
            "resolved fraction; σφ is a lower bound on the true sorting. "
            "Detected areal cover is the fraction of the footprint covered by "
            "detected masks: a measure of areal recall.", ss))
        story.append(R._table(rows, font_size=7, ss=ss, wrap_threshold=0,
                              col_widths=[1.9 * 72] + [(4.8 * 72) / len(counted)] * len(counted)))
        _spacer(story, 8)
        if RF is not None and len(counted) >= 1:
            try:
                series = {}
                dmins = {}
                for i in counted:
                    series[_image_label(i)] = i["clasts"]["Clast_length"].to_numpy(dtype=float)
                    if i["dmin_m"]:
                        dmins[_image_label(i)] = i["dmin_m"] * 1000.0
                fig = RF.cdf_overlay(series, dmin_mm=dmins or None)
                _figure_block(
                    story, ss, fig,
                    ("Cumulative distributions of apparent a-axis length per image, "
                     "with D50 and D84 marked. " if len(counted) > 1 else
                     "Cumulative distribution of apparent a-axis length, with D50 and D84 marked. ")
                    + "Shaded: below the detection limit.",
                    dpi=dpi, quality=quality, max_h=4.0)
            except Exception as ex:
                log(f"[detection] CDF overlay skipped: {type(ex).__name__}: {ex}")
        # Per-window overlay, when a merge was built from several windows.
        if RF is not None:
            for i in counted:
                wins = [r for r in i["runs"] if r["window_m"] is not None and not r["merged"]]
                if len({r["window_m"] for r in wins}) < 2:
                    continue
                series = {}
                seen = set()
                for r in sorted(wins, key=lambda r: r["window_m"]):
                    if r["window_m"] in seen:
                        continue
                    df = F._read_clasts(r["path"])
                    if df is None or "Clast_length" not in df or not len(df):
                        continue
                    seen.add(r["window_m"])
                    series[f"{r['window_m']:g} m window"] = df["Clast_length"].to_numpy(dtype=float)
                if i["clasts"] is not None and len(series) >= 2:
                    series["merged"] = i["clasts"]["Clast_length"].to_numpy(dtype=float)
                    try:
                        fig = RF.cdf_overlay(series)
                        _figure_block(
                            story, ss, fig,
                            f"{_image_label(i)}: the window runs that fed the merge. "
                            f"Larger windows see fewer, larger clasts; the merge keeps "
                            f"each clast once.", dpi=dpi, quality=quality, max_h=3.8)
                    except Exception as ex:
                        log(f"[detection] window overlay skipped: {type(ex).__name__}: {ex}")
        # Reading paragraph.
        reading = []
        for i in counted:
            L = i["length_mm"]
            fw = i["folk_ward"]
            dp = _dp_for_gsd(i["gsd_m"])
            s = (f"<b>{_image_label(i)}</b>: median {fmt_mm(L['D50'], dp)}, "
                 f"D84 {fmt_mm(L['D84'], dp)}")
            if fw:
                s += f", {fw['sorting_verbal']} ({fmt_num(fw['sorting_phi'])} φ)"
            if i["density_per_m2"]:
                s += f", {i['density_per_m2']:.0f} clasts per m²"
            if i["areal_cover"] is not None:
                s += (f"; detected masks cover {fmt_pct(i['areal_cover'])} of the "
                      f"footprint, so the sample is the detector's selection of the "
                      f"surface, not the whole surface")
            reading.append(s + ".")
        if len(counted) > 1:
            d50s = [(i["length_mm"]["D50"], _image_label(i)) for i in counted]
            lo, hi = min(d50s), max(d50s)
            if lo[0] > 0:
                reading.append(
                    f"The median differs by {100 * (hi[0] - lo[0]) / lo[0]:.0f} % between "
                    f"{lo[1]} and {hi[1]}"
                    + (", within the run-to-run repeatability quoted above."
                       if any(i["repeatability"] for i in counted)
                       and 100 * (hi[0] - lo[0]) / lo[0] < 3 else ".")
                )
        _para(story, ss, " ".join(reading))
    story.append(PageBreak())


# --- 3  Spatial statistics ---
_SHOWN_CATEGORIES = ("size", "shape", "density")
_TRANSPORT_NOTE = (
    "Transport-threshold surfaces (Hjulström-curve velocities, Shields and "
    "Soulsby parameters) are functions of the size surface alone, computed "
    "with fixed constants and outside the validity of the empirical curves "
    "for cobbles; they are listed in the register and not mapped. Appendix C "
    "gives the formulas.")


def _read_band(path: Path, band: int = 1):
    from osgeo import gdal
    ds = gdal.Open(str(path))
    if ds is None:
        return None, None, (None, 0)
    b = ds.GetRasterBand(band)
    arr = b.ReadAsArray().astype(float)
    nd = b.GetNoDataValue()
    if nd is not None:
        arr[arr == nd] = np.nan
    gt = ds.GetGeoTransform()
    extent = None
    if gt and gt != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        x0, x1 = gt[0], gt[0] + ds.RasterXSize * gt[1]
        y1, y0 = gt[3], gt[3] + ds.RasterYSize * gt[5]
        extent = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
    desc = b.GetDescription()
    n_bands = ds.RasterCount
    ds = None
    return arr, extent, (desc, n_bands)


def _panel(ax, arr, extent, cmap, vmin, vmax, unit, label, cyclic=False):
    import matplotlib.ticker as mt
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent,
                   origin="upper", interpolation="nearest")
    if extent:
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(mt.MaxNLocator(4))
            f = mt.ScalarFormatter(useOffset=False)
            f.set_scientific(False)
            axis.set_major_formatter(f)
        ax.tick_params(labelsize=6)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30)
            lbl.set_ha("right")
        ax.set_xlabel("Easting (m)", fontsize=7)
        ax.set_ylabel("Northing (m)", fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if label:
        ax.set_title(label, fontsize=8)
    return im


def _raster_figure(r: dict, *, shared=None, u_disp=None):
    """One figure for a raster: a single panel, or the non-empty bands of a
    multi-band stack sharing one colour range. Returns (fig, stats) or
    (None, reason)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt
    path = r["path"]
    arr0, extent, (desc, n_bands) = _read_band(path, 1)
    if arr0 is None:
        return None, "GDAL could not open the file"
    factor, unit, cmap = r["factor"], r["unit"], r["cmap"]
    cyclic = "orientation" in (r.get("field") or "").lower()
    if cyclic:
        cmap = "twilight"
    bands = []
    for b in range(1, n_bands + 1):
        arr, _e, (d, _n) = _read_band(path, b) if b > 1 else (arr0, extent, (desc, n_bands))
        v = arr * factor
        valid = v[np.isfinite(v)]
        if valid.size == 0:
            continue
        bands.append((b, d or f"Band {b}", v, valid))
    if not bands:
        return None, "no valid cell in any band"
    allv = np.concatenate([b[3] for b in bands])
    if cyclic:
        vmin, vmax = 0.0, 180.0
    elif shared is not None:
        vmin, vmax = shared
    elif r["diverging"]:
        m = float(np.nanmax(np.abs(np.percentile(allv, (2, 98)))))
        vmin, vmax = -m, m
    else:
        vmin, vmax = (float(x) for x in np.percentile(allv, (2, 98)))
        if vmin == vmax:
            vmin, vmax = float(allv.min()), float(allv.max())
    n = len(bands)
    cols = min(3, n)
    rows_ = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_, cols, figsize=(6.8, 2.6 * rows_ + 0.4),
                             squeeze=False, constrained_layout=True)
    im = None
    for k, (b, d, v, valid) in enumerate(bands):
        ax = axes[k // cols][k % cols]
        im = _panel(ax, v, extent, cmap, vmin, vmax, unit,
                    d if n > 1 else None, cyclic=cyclic)
    for k in range(n, rows_ * cols):
        axes[k // cols][k % cols].axis("off")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
    label = ("Fraction of cell area" if r.get("category") == "cover"
             else f"{r['display']}" + (f" ({unit})" if unit else ""))
    cbar.set_label(label, fontsize=8)
    if cyclic:
        cbar.set_ticks([0, 45, 90, 135, 180])
    elif u_disp is not None:
        from functions import precision as prec
        dp = prec.decimals_for(u_disp, unit)
        cbar.ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda x, _p, d=dp: f"{x:.{d}f}"))
    cbar.ax.tick_params(labelsize=7)
    try:
        if extent and n == 1:
            from functions.map_export import _add_scale_bar, _add_north_arrow
            _add_scale_bar(axes[0][0], units="m")
            _add_north_arrow(axes[0][0])
    except Exception:
        pass
    stats = {
        "n_bands_shown": n, "n_bands": n_bands,
        "min": float(allv.min()), "median": float(np.median(allv)), "max": float(allv.max()),
        "valid_frac": float(bands[0][3].size / bands[0][2].size) if n == 1 else None,
        "range": (vmin, vmax),
    }
    return fig, stats


def spatial_statistics(story, ss, inv, facts, options, log, section, *, dpi, quality):
    from reportlab.platypus import Paragraph, PageBreak
    R = _R()
    by_img = facts["rasters_by_image"]
    if not by_img:
        return False
    _para(story, ss, f"{section}. Spatial statistics", "H1Custom")
    _para(story, ss,
          "Per-cell surfaces of the merged clast list. Percentile and mean "
          "panels of one field share a colour range per image; dispersion "
          "panels have their own; ranges are the 2nd–98th percentiles of the "
          "valid cells, so a single extreme cell cannot compress the map. "
          "Cells without a detection are white. The register lists every "
          "surface on disk; the figures show those that carry information.")
    try:
        from functions import report_figures as RF
    except Exception:
        RF = None
    imgs_by_stem = {i["stem"]: i for i in facts["images"]}
    sub = 0
    for src, rasters in sorted(by_img.items()):
        img = imgs_by_stem.get(src) or next(
            (i for s, i in imgs_by_stem.items() if naming.same_image(s, src)), None)
        label = _image_label(img) if img else src
        sub += 1
        multi = len(by_img) > 1
        if multi:
            _para(story, ss, f"{section}.{sub} {label}", "H2Custom")
        # Register.
        rows = [["Field", "Statistic", "Cell", "Unit", "Bands", "Shown"]]
        seen_cover = False
        plan = []
        for r in sorted(rasters, key=lambda r: ((r["category"] != "size"), r["category"],
                                                str(r.get("field")), str(r.get("parameter")))):
            cat = r["category"]
            show = False
            why = ""
            if r["empty"]:
                why = "no data"
            elif cat == "cover":
                if seen_cover:
                    why = "duplicate of the first"
                else:
                    show, seen_cover = True, True
            elif cat == "transport":
                why = "size-derived; not mapped"
            elif cat == "orientation":
                p = (r.get("parameter") or "").lower()
                if p in ("average", "mean"):
                    show = True
                else:
                    why = "axial data: not defined"
            elif cat in _SHOWN_CATEGORIES:
                show = True
            else:
                show = True
            fld, _fu, _ff = _u.field_display(r.get("field"))
            stat = _u.parameter_display(r.get("param_key"))
            stat = stat[0] if stat else (r.get("param_key") or "—")
            if stat == "quantile":
                stat = "quantile (unrecorded)"
            if cat == "cover":
                fld, stat = "Detected areal cover", "by size class"
            rows.append([Paragraph(fld or "—", ss["Body"]), stat,
                         f"{r['cellsize']:g} m" if r.get("cellsize") else "—",
                         r["unit"] or "—", str(r.get("n_bands") or 1),
                         "yes" if show else why])
            if show:
                plan.append(r)
        story.append(R._tbl_caption(f"Surfaces computed for {label}.", ss))
        story.append(R._table(rows, font_size=7.5,
                              col_widths=[2.0 * 72, 1.5 * 72, 0.6 * 72, 0.7 * 72,
                                          0.6 * 72, 1.4 * 72]))
        _spacer(story, 6)
        # Cell support.
        if RF is not None and img is not None and img.get("clasts") is not None and img["kind"] == modes.ORTHO:
            cell = next((r["cellsize"] for r in rasters if r.get("cellsize")), 1.0)
            try:
                fig = RF.cell_count_map(img["clasts"], cellsize_m=cell,
                                        crs_label=img.get("crs"))
                meta = getattr(fig, "pm_meta", {}) or {}
                _figure_block(
                    story, ss, fig,
                    f"Clasts per {cell:g} m cell on {label}: median "
                    f"{fmt_int(meta.get('n_median', 0))} (range "
                    f"{fmt_int(meta.get('n_min', 0))}–{fmt_int(meta.get('n_max', 0))}); "
                    f"{fmt_pct(meta.get('frac_cells_lt30'))} of cells hold fewer than "
                    f"{F.MIN_CELL_N_PERCENTILE} clasts (hatched), where a percentile is "
                    f"unreliable and Folk and Ward moments are noise.",
                    dpi=dpi, quality=quality, max_h=4.2)
            except Exception as ex:
                log(f"[spatial] cell-count map skipped: {type(ex).__name__}: {ex}")
        # Shared ranges for percentile/average of one field.
        fam = {}
        for r in plan:
            if (r.get("param_key") in _u.PERCENTILE_FAMILY and r["category"] == "size"):
                arr, _e, _m = _read_band(r["path"], 1)
                if arr is None:
                    continue
                v = arr * r["factor"]
                v = v[np.isfinite(v)]
                if v.size:
                    lo, hi = np.percentile(v, (2, 98))
                    key = (r.get("field"), r.get("cellsize"))
                    cur = fam.get(key)
                    fam[key] = (min(lo, cur[0]), max(hi, cur[1])) if cur else (float(lo), float(hi))
        # Figures.
        unc = facts["uncertainty"]
        for r in plan:
            shared = fam.get((r.get("field"), r.get("cellsize"))) \
                if r.get("param_key") in _u.PERCENTILE_FAMILY else None
            u_disp = None
            try:
                if unc and r["unit"] == _u.field_unit_and_factor(r.get("field"))[0]:
                    u_disp = R._uncertainty_display(unc, r.get("field"), r["factor"])[0]
            except Exception:
                pass
            try:
                fig, st = _raster_figure(r, shared=shared, u_disp=u_disp)
            except Exception as ex:
                log(f"[spatial] {r['path'].name}: {type(ex).__name__}: {ex}")
                continue
            if fig is None:
                continue
            unit = r["unit"]
            dp = 0 if unit == "mm" else (2 if unit in ("", "φ") else 1)
            if r["category"] == "cover":
                cap = (f"Detected areal cover by size class on {label}: the fraction "
                       f"of each cell covered by detected masks whose length falls in "
                       f"the band. Empty classes are not shown. Summed over classes "
                       f"it is the detected cover — a lower bound on the true clast "
                       f"cover and a direct measure of areal recall.")
            elif r["category"] == "orientation":
                ori = img["orientation"] if img else {}
                cap = (f"Axial mean orientation per cell on {label} (doubled-angle "
                       f"circular mean, bearing clockwise from grid north, cyclic colour"
                       f"scale)."
                       + (f" Project-wide mean {ori['mean_deg']:.0f}° with mean resultant "
                          f"length <i>R</i> = {ori['R']:.2f}"
                          + (" — no preferred orientation." if ori["R"] < 0.3
                             else " — a weak fabric." if ori["R"] < 0.5 else " — a clear fabric.")
                          if ori else ""))
            else:
                cap = (f"{r['display']} on {label}, {r['cellsize']:g} m cells: "
                       f"{fmt_num(st['min'], dp)}–{fmt_num(st['max'], dp)} {unit} "
                       f"(median {fmt_num(st['median'], dp)} {unit})"
                       + (f", {fmt_pct(st['valid_frac'])} of cells valid" if st["valid_frac"] else "")
                       + (f"; colour range {fmt_num(st['range'][0], dp)}–{fmt_num(st['range'][1], dp)} "
                          f"{unit} shared with the other percentile panels of this field."
                          if shared else "."))
            _figure_block(story, ss, fig, cap, dpi=dpi, quality=quality,
                          max_h=4.4 if st["n_bands_shown"] == 1 else 6.5,
                          source=r["path"].name)
        if RF is not None and img is not None and img.get("clasts") is not None \
                and "Orientation" in img["clasts"]:
            try:
                fig = RF.orientation_rose(img["clasts"]["Orientation"].to_numpy(dtype=float))
                meta = getattr(fig, "pm_meta", {}) or {}
                _figure_block(
                    story, ss, fig,
                    f"Orientation of the major axes on {label} (axial rose): mean "
                    f"{meta.get('mean_deg', float('nan')):.0f}°, <i>R</i> = "
                    f"{meta.get('R', float('nan')):.2f}, n = {fmt_int(meta.get('n', 0))}.",
                    dpi=dpi, quality=quality, max_h=3.4)
            except Exception as ex:
                log(f"[spatial] rose skipped: {type(ex).__name__}: {ex}")
        # Reading.
        size_d50 = [r for r in plan if r["category"] == "size"
                    and (r.get("param_key") or "").upper() == "D50"
                    and "length" in (r.get("field") or "").lower()]
        if size_d50 and img and img["length_mm"]:
            _para(story, ss,
                  f"Read the median surface against the project median of "
                  f"{fmt_mm(img['length_mm']['D50'], _dp_for_gsd(img['gsd_m']))}: cells "
                  f"above it are coarser patches, below it finer. With a median cell "
                  f"population in the tens of clasts, a difference smaller than about "
                  f"10 % between neighbouring cells is within sampling noise.")
        if any(r["category"] == "transport" for r in rasters):
            _para(story, ss, f"<font size=8>{_TRANSPORT_NOTE}</font>")
        _spacer(story, 6)
    story.append(PageBreak())
    return True


# --- 4  Zonal statistics ---
def _zonal_title(entry: dict) -> str:
    R = _R()
    mode = entry.get("mode")
    what = {"polygons": "Polygon statistics",
            "transects": "Transect profiles"}.get(mode, "Zonal output")
    fld = _u.field_display(entry.get("field"))[0] if entry.get("field") else ""
    src = entry.get("source_image")
    img = _short_stem(str(src or ""))
    zone = _short_stem(R._pretty_set_name(entry.get("vector") or "").replace(" ", "_"))
    bits = [what]
    if fld:
        bits.append(f"of {fld.lower()}")
    if img:
        bits.append(f"on {img}")
    if zone and zone.lower() not in (img.lower(), "zones", "zone", "transects"):
        bits.append(f"(zones: {zone})")
    return " ".join(bits)


def _profile_figure_is_empty(entry: dict) -> bool:
    """A composed profile whose summary table holds no sample."""
    import csv
    summary = entry.get("summary")
    if not summary:
        return False
    try:
        with open(summary, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
    except OSError:
        return False
    if len(rows) < 2 or "n_samples" not in rows[0]:
        return False
    k = rows[0].index("n_samples")
    total = 0.0
    for r in rows[1:]:
        try:
            total += float(r[k])
        except (ValueError, IndexError):
            pass
    return total == 0


def zonal_statistics(story, ss, inv, facts, options, log, section, *, dpi, quality):
    from reportlab.platypus import Paragraph, PageBreak, Image as RImage
    import csv
    R = _R()
    entries = facts["zonal"]
    profiles = facts["profile_figures"]
    if not entries and not profiles:
        return False
    _para(story, ss, f"{section}. Zonal statistics", "H1Custom")
    _para(story, ss,
          "Statistics of the merged clast list inside user-drawn polygons and "
          "along transects sampled from the surfaces of the previous section.")
    empties = []
    shown = 0
    # Location map, when the zone sets are on disk.
    try:
        from functions import report_figures as RF
        from functions.layout import resolve_project_subfolder as _rsf
        geoms = _rsf(Path(inv["project_root"]), "geometries")
        sets = sorted({e.get("vector") for e in entries if e.get("vector")})
        found = []
        if geoms.is_dir():
            for s_ in sets:
                for cand in (geoms / f"{s_}.geojson", geoms / f"zones_{s_}.geojson"):
                    if cand.exists():
                        found.append(cand)
                        break
        img = next((i for i in facts["images"] if i["kind"] == modes.ORTHO), None)
        if found and img is not None:
            fig = RF.zone_location_map(img["path"], found, transform=img.get("transform"))
            _figure_block(story, ss, fig,
                          f"Zones and transects on {_image_label(img)}: "
                          f"{', '.join(R._pretty_set_name(p.stem) for p in found)}.",
                          dpi=dpi, quality=quality, max_h=4.6)
            shown += 1
    except Exception as ex:
        log(f"[zonal] location map skipped: {type(ex).__name__}: {ex}")

    for entry in profiles:
        png = entry["path"]
        if _profile_figure_is_empty(entry):
            empties.append("a composed profile figure with no sample")
            continue
        try:
            _keep(story, [RImage(str(png), width=6.2 * 72, height=3.6 * 72,
                                 kind="proportional"),
                          R._fig_caption("Transect profile composed in the Zonal tab.",
                                         ss, source=png.name)])
            shown += 1
        except Exception:
            empties.append(png.name)
    for entry in entries:
        p = entry["path"]
        mode = entry.get("mode")
        title = _zonal_title(entry)
        if mode == "polygons":
            try:
                with open(p, newline="", encoding="utf-8") as fh:
                    rows = list(csv.reader(fh))
            except OSError:
                continue
            if len(rows) < 2:
                empties.append(f"{title[0].lower() + title[1:]}: empty file")
                continue
            header, body = rows[0], rows[1:]
            if "polygon_id" not in header and "count" not in header:
                # Not a polygon statistics table at all; one such file used
                # to bring the whole section down with a table of no
                # columns.
                log(f"[zonal] {p.name}: no polygon_id or count column "
                    f"(found: {', '.join(header[:5])}); skipped.")
                empties.append(f"{p.name}: not a polygon statistics table "
                               f"(no polygon_id or count column), skipped")
                continue
            counts = []
            if "count" in header:
                ci = header.index("count")
                for r in body:
                    try:
                        counts.append(float(r[ci]))
                    except (ValueError, IndexError):
                        pass
            if counts and max(counts) == 0:
                empties.append(f"{title[0].lower() + title[1:]}: no clast inside any polygon")
                continue
            fld = entry.get("field")
            unit, factor = entry.get("unit"), 1.0
            if fld and unit is None:
                unit, factor = _u.field_unit_and_factor(fld)
            vcols = set(R._poly_value_cols(header)) if fld else set()
            udisp = None
            if fld:
                try:
                    udisp = R._uncertainty_display(facts["uncertainty"], fld,
                                                   _u.field_unit_and_factor(fld)[1])[0]
                except Exception:
                    pass
            wanted = ["polygon_id", "count", "density"]
            if fld:
                wanted += ["mean", "std", "median", "D50", "P50",
                           "sigma_phi", "sk_phi", "kg_phi"]
            keep = [c for c in wanted if c in header]
            idx = [header.index(c) for c in keep]
            out = [[c + (f" ({unit})" if (header.index(c) in vcols and unit) else
                         " (per m²)" if c == "density" else "") for c in keep]]
            for r in body[:30]:
                row = []
                for i_ in idx:
                    v = r[i_] if i_ < len(r) else "—"
                    if i_ in vcols:
                        try:
                            v = R._fmt_measured(float(v) * factor, unit, u_display=udisp)
                        except (TypeError, ValueError):
                            pass
                    elif header[i_] in ("sigma_phi", "sk_phi", "kg_phi", "density"):
                        try:
                            v = fmt_num(float(v), 2)
                        except (TypeError, ValueError):
                            pass
                    elif header[i_] == "count":
                        v = fmt_int(v)
                    row.append(v)
                out.append(row)
            _para(story, ss, title, "H2Custom")
            cap = (f"{title} ({fmt_int(len(body))} {_plural(len(body), 'polygon')}). Count "
                   f"is the number of clasts inside the polygon; percentiles from fewer "
                   f"than {F.MIN_CELL_N_PERCENTILE} clasts are unreliable.")
            if not fld:
                cap += (" The file does not record which quantity it summarises, so "
                        "only counts are shown; re-running the Zonal tab records it.")
            story.append(R._tbl_caption(cap, ss, source=p.name))
            story.append(R._table(out, font_size=7))
            if len(body) > 30:
                _note(story, ss, f"{len(body) - 30} more polygons in the CSV.")
            _spacer(story, 6)
            shown += 1
        elif mode == "transects":
            try:
                from functions import zonal_stats as zs
                series, skipped = zs.load_profile_series([p])
            except Exception as ex:
                log(f"[zonal] {p.name}: {ex}")
                continue
            fld = entry.get("field")
            if series and not any(s_.field for s_ in series) and fld:
                for s_ in series:
                    s_.field = fld
            kept = [s_ for s_ in series
                    if np.isfinite(np.asarray(s_.value, dtype=float)).any()]
            if not kept:
                empties.append(f"{title[0].lower() + title[1:]}: transects do not intersect the surface")
                continue
            import tempfile
            try:
                with tempfile.TemporaryDirectory() as td:
                    png = zs.plot_profile_figure(
                        kept, Path(td) / "profile.png",
                        title=f"{_u.field_display(fld)[0] if fld else 'Value'} along "
                              f"{R._pretty_set_name(entry.get('vector') or p.stem)}")
                    buf = io.BytesIO(Path(png).read_bytes())
                _keep(story, [Paragraph(title, ss["H2Custom"]),
                              RImage(buf, width=6.2 * 72, height=3.6 * 72,
                                     kind="proportional"),
                              R._fig_caption(
                                  f"{len(kept)} {_plural(len(kept), 'transect')} sampled "
                                  f"from the {_u.field_display(fld)[0].lower() if fld else ''} "
                                  f"surface; values are the underlying cell values, so "
                                  f"variation shorter than a cell is not resolved.",
                                  ss, source=p.name)])
                shown += 1
            except Exception as ex:
                log(f"[zonal] profile for {p.name}: {type(ex).__name__}: {ex}")
    if empties:
        _para(story, ss, "<font size=8>Not shown — " + "; ".join(empties) + ".</font>")
    if not shown and not empties:
        _note(story, ss, "No zonal output could be rendered.")
    story.append(PageBreak())
    return True


# --- 5  Validation ---
def _paired_arrays(v: dict):
    """Re-pair truth and detection from the CSVs (the JSON keeps no
    per-pair arrays). Returns a DataFrame with truth/detect columns (m)."""
    import pandas as pd
    from functions import validation as V
    t = pd.read_csv(v["truth"])
    d = pd.read_csv(v["detect"])
    field = v["field"]
    tol = (v["payload"].get("tolerance_m") or None)
    res = V.pair_csvs(t, d, tolerance=tol)
    pairs = res["matched_pairs"]
    if not pairs:
        return None
    return pd.DataFrame({
        "truth": [float(t.iloc[i][field]) for i, _ in pairs],
        "detect": [float(d.iloc[j][field]) for _, j in pairs],
    })


def validation(story, ss, inv, stats, facts, options, log, section, *, dpi, quality):
    from reportlab.platypus import Paragraph, PageBreak
    R = _R()
    vals = facts["validations"]
    truths = facts["truth_files"]
    if not vals and not truths:
        return False
    _para(story, ss, f"{section}. Validation", "H1Custom")
    usable = [v for v in vals if v["usable"]]
    flagged = [v for v in vals if not v["usable"]]
    if not usable:
        _para(story, ss,
              "<b>No usable comparison.</b> "
              + (f"{len(flagged)} saved {_plural(len(flagged), 'comparison')} "
                 f"{'is' if len(flagged) == 1 else 'are'} degenerate: "
                 + "; ".join(f"{naming.image_stem(v['truth'].name)} vs "
                             f"{naming.image_stem(v['detect'].name)} — "
                             + ", ".join(v["flags"]) for v in flagged) + "."
                 if flagged else "Ground-truth files are present but no comparison has been run.")
              + " The detections in this project are unvalidated; the method's published "
                "accuracy (Soloy et al. 2020) is the only reference: on quadrat "
                "photographs at 1 mm/px, detected lengths agreed with hand measurements "
                "to within a few millimetres, and on UAV imagery at 5 mm/px the smallest "
                "reliable detection was about 4 cm.")
        if truths:
            rows = [["Truth file", "Clasts"]]
            for t in truths:
                rows.append([Paragraph(t["path"].name, ss["Body"]),
                             fmt_int(t["rows"]) + ("" if t["rows"] >= F.MIN_VALIDATION_PAIRS
                                                   else " — too few to validate anything")])
            story.append(R._tbl_caption("Ground-truth files in the project.", ss))
            story.append(R._table(rows, col_widths=[4.7 * 72, 2.0 * 72]))
        story.append(PageBreak())
        return True

    _para(story, ss,
          "Each comparison pairs hand-digitised clasts with detections by mutual "
          "nearest neighbour, then compares the two size distributions. A "
          "paired detection is a true positive. Mask R-CNN, the built-in "
          "model, is tuned for precision over recall, so with it the informative "
          "questions are whether the detected sample recovers the right "
          "distribution (ΔD50, ΔD84, the K-S statistic) and how large the "
          "per-clast error is (RMSE, bias), more than how many clasts it found.")
    if flagged:
        _para(story, ss, f"<font size=8>Excluded as degenerate: "
              + "; ".join(f"{v['truth'].name} vs {v['detect'].name} ({', '.join(v['flags'])})"
                          for v in flagged) + ".</font>")
    try:
        from functions import report_figures as RF
    except Exception:
        RF = None
    for k, v in enumerate(usable, start=1):
        pl = v["payload"]
        m = pl.get("metrics") or {}
        d = pl.get("distribution") or {}
        pr = pl.get("paired") or {}
        tr = d.get("truncated") if isinstance(d.get("truncated"), dict) else None
        field = v["field"]
        unit, factor = _u.field_unit_and_factor(field)
        _para(story, ss, f"{section}.{k} {_u.field_display(field)[0]}: "
                         f"{naming.image_stem(v['truth'].name)} vs "
                         f"{naming.image_stem(v['detect'].name)}", "H2Custom")
        dmin_txt = (f"{v['dmin_m'] * 1000:.0f} mm ({F.DETECTION_LIMIT_PX} px × "
                    f"{fmt_gsd(v['detect_gsd_m'])})" if v["dmin_m"] else "—")

        def _pct(x):
            return fmt_signed(100 * x, 1, " %") if x is not None else "—"

        def _val(x):
            return fmt_mm(x * factor, 1 if unit == "mm" else 2, unit) if x is not None else "—"

        rows = [["Metric", "All clasts", f"D ≥ D<sub>min</sub>", "Reading"]]
        rows = [[Paragraph(c, ss["Body"]) for c in rows[0]]]
        rows.append(["True positives (n)", fmt_int(m.get("n_matched")),
                     fmt_int((tr or {}).get("n_matched")) if tr and tr.get("n_matched") else "—",
                     f"of {fmt_int(m.get('n_truth'))} truth and {fmt_int(m.get('n_detect'))} detected"])
        rows.append(["Recall / precision / F1",
                     f"{fmt_num(m.get('recall'), 2)} / {fmt_num(m.get('precision'), 2)} / "
                     f"{fmt_num(m.get('f1'), 2)}", "—", "TP / truth, TP / detected, harmonic mean"])
        rows.append(["ΔD50", _pct(m.get("D50_relerr")),
                     _pct(((tr.get("detect_d50") - tr.get("truth_d50")) / tr.get("truth_d50"))
                          if tr and tr.get("truth_d50") else None),
                     "+ = detection coarser than truth"])
        rows.append(["ΔD84", _pct(m.get("D84_relerr")),
                     _pct(((tr.get("detect_d84") - tr.get("truth_d84")) / tr.get("truth_d84"))
                          if tr and tr.get("truth_d84") else None),
                     ""])
        rows.append(["K-S D (p)", f"{fmt_num(d.get('ks_statistic'), 2)} "
                                  f"({fmt_num(d.get('ks_p_value'), 2)})",
                     (f"{fmt_num(tr.get('ks_statistic'), 2)} ({fmt_num(tr.get('ks_p_value'), 2)})"
                      if tr else "—"),
                     "small p: the distributions differ"])
        rows.append(["RMSE (paired)", _val(pr.get("rmse")), "—", "per-clast error"])
        rows.append(["Bias (paired)", _val(pr.get("bias")), "—", "+ = detection over-estimates"])
        rows.append(["R² (paired)", fmt_num(pr.get("r2"), 2), "—", ""])
        rows.append(["Detection limit D<sub>min</sub>", "—", dmin_txt,
                     f"truth clasts below it: {fmt_int((tr or {}).get('truth_n_below_dmin', 0))}"])
        rows = [[Paragraph(str(c), ss["Body"]) if j in (0, 3) else c for j, c in enumerate(r)]
                if i_ else r for i_, r in enumerate(rows)]
        story.append(R._tbl_caption(
            f"Validation of {_u.field_display(field)[0].lower()} ({unit}), truth "
            f"{v['truth'].name} against detection {v['detect'].name}. The second "
            f"column restricts both sides to clasts the imagery can resolve.", ss))
        story.append(R._table(rows, font_size=8,
                              col_widths=[1.7 * 72, 1.4 * 72, 1.2 * 72, 2.4 * 72]))
        _spacer(story, 6)
        if RF is not None and v["truth"].exists() and v["detect"].exists():
            try:
                paired = _paired_arrays(v)
                if paired is not None:
                    fig = RF.validation_diagnostics(paired, field=field, unit=unit, scale=factor)
                    _figure_block(
                        story, ss, fig,
                        f"True positives ({fmt_int(len(paired))}): detection against "
                        f"truth with the 1:1 line; Bland–Altman difference against mean "
                        f"with bias and 95 % limits of agreement; cumulative "
                        f"distributions with D50 and D84.",
                        dpi=dpi, quality=quality, max_h=3.4)
            except Exception as ex:
                log(f"[validation] diagnostics skipped: {type(ex).__name__}: {ex}")
    kinds = sorted({v["detect_kind"] for v in usable})
    if facts["n_ortho"] and modes.ORTHO not in kinds:
        _para(story, ss,
              "<b>Scope.</b> The comparisons above validate quadrat "
              "detections at millimetre GSD. No ground truth exists for the "
              "ortho-images, whose GSD is several times coarser; their absolute "
              "values rest on the method's published accuracy, not on this project.")
    story.append(PageBreak())
    return True


# --- 6  Interpretation and limitations ---
def interpretation(story, ss, inv, facts, options, log, section):
    from reportlab.platypus import PageBreak
    _para(story, ss, f"{section}. Interpretation and limitations", "H1Custom")
    kind = F.headline_kind(facts)
    imgs = [i for i in facts["images"] if i["kind"] == kind and i["length_mm"]]
    paras = []
    if imgs:
        s = []
        for i in imgs:
            L, fw = i["length_mm"], i["folk_ward"]
            dp = _dp_for_gsd(i["gsd_m"])
            klass = ("pebble" if L["D50"] < 56 else
                     "pebble-to-cobble" if L["D50"] < 72 else "cobble")
            t = (f"{_image_label(i)} is a {fw['sorting_verbal'] if fw else ''} "
                 f"{klass} deposit at the surface "
                 f"(D50 {fmt_mm(L['D50'], dp)}, D84 {fmt_mm(L['D84'], dp)}")
            if fw:
                t += f", {fw['skewness_verbal']}"
            t += ")"
            s.append(t)
        paras.append("<b>What the numbers say.</b> " + "; ".join(s) + ". "
                     "Wentworth classes: pebble 4–64 mm, cobble 64–256 mm.")
    lim = []
    dmins = [i["dmin_m"] for i in imgs if i["dmin_m"]]
    if dmins:
        lo, hi = min(dmins) * 1000, max(dmins) * 1000
        rng = f"{lo:.0f} mm" if abs(hi - lo) < 0.5 else f"{lo:.0f}–{hi:.0f} mm"
        censored = any(i["length_mm"].get("D5", 0) < 1.5 * i["dmin_m"] * 1000
                       for i in imgs if i["dmin_m"] and i["length_mm"])
        if censored:
            lim.append(
                f"<b>Censoring.</b> Nothing shorter than {rng} is detected, so D5 and "
                f"D16 sit at the detection limit and σφ, Skφ and K<sub>G</sub> describe "
                f"the resolved fraction only. A finer matrix, if present, is invisible "
                f"here; the deposit's true sorting is poorer than stated.")
        else:
            lim.append(
                f"<b>Detection limit.</b> Nothing shorter than {rng} is detected; the "
                f"population lies well above it, so the percentiles are not censored, "
                f"but a finer fraction below {rng} would be invisible.")
    covs = [i["areal_cover"] for i in imgs if i["areal_cover"] is not None]
    if covs:
        lim.append(
            f"<b>Recall.</b> Detected masks cover {fmt_pct(min(covs))}"
            f"{'–' + fmt_pct(max(covs)) if len(covs) > 1 else ''} of the footprint. The "
            f"detector favours isolated, fully exposed clasts; touching clasts of similar "
            f"size can merge and partly buried ones are shortened. Percentiles are those "
            f"of the detected subset.")
    lim.append(
        "<b>Measure.</b> Lengths are apparent plan-view a-axes by number of clasts; "
        "they are not sieve-by-weight or Wolman grid-by-number percentiles and read "
        "coarser than a b-axis count of the same deposit by roughly the a/b ratio "
        + (f" ({fmt_num(imgs[0]['width_mm']['D50'] / imgs[0]['length_mm']['D50'], 2)} here)"
           if imgs and imgs[0]["width_mm"] and imgs[0]["length_mm"] else "") + ".")
    if facts["rasters_by_image"]:
        lim.append(
            f"<b>Cells.</b> Per-cell values rest on tens of clasts; a percentile from "
            f"fewer than {F.MIN_CELL_N_PERCENTILE} and a Folk and Ward moment from "
            f"fewer than {F.MIN_CELL_N_MOMENTS} clasts are unreliable, and partial "
            f"edge cells are biased. Neighbouring cells within about 10 % of each "
            f"other are not distinguishable.")
    usable = [v for v in facts["validations"] if v["usable"]]
    if not usable:
        lim.append("<b>Validation.</b> None usable for this project: absolute sizes "
                   "rest on the method's published accuracy.")
    elif facts["n_ortho"] and modes.ORTHO not in {v["detect_kind"] for v in usable}:
        lim.append("<b>Validation.</b> Only the quadrat detections are validated "
                   "here; the ortho-image results inherit the method's published "
                   "accuracy at their coarser GSD.")
    if any(r["category"] == "transport" for rs in facts["rasters_by_image"].values() for r in rs):
        lim.append("<b>Transport thresholds.</b> The Hjulström-curve velocities on disk "
                   "are empirical fits for steady unidirectional flow, evaluated beyond "
                   "their stated validity for cobbles; on a wave-worked beach they are "
                   "indicative at best and are not mapped in this report.")
    for p in paras + lim:
        _para(story, ss, p)
    story.append(PageBreak())


def references(story, ss):
    _para(story, ss, "References", "H1Custom")
    refs = [
        "Folk, R. L. and Ward, W. C. (1957). Brazos River bar: a study in the "
        "significance of grain size parameters. <i>Journal of Sedimentary "
        "Petrology</i> 27, 3–26.",
        "He, K., Gkioxari, G., Dollár, P. and Girshick, R. (2017). Mask R-CNN. "
        "<i>Proceedings of the IEEE International Conference on Computer Vision</i>, "
        "2961–2969.",
        "Kellerhals, R. and Bray, D. I. (1971). Sampling procedures for coarse "
        "fluvial sediments. <i>Journal of the Hydraulics Division</i> 97, 1165–1180.",
        "Soloy, A., Turki, I., Fournier, M., Costa, S., Peuziat, B. and Lecoq, N. "
        "(2020). A deep learning-based method for quantifying and mapping the grain "
        "size on pebble beaches. <i>Remote Sensing</i> 12, 3659.",
        "Wentworth, C. K. (1922). A scale of grade and class terms for clastic "
        "sediments. <i>Journal of Geology</i> 30, 377–392.",
    ]
    for r in refs:
        _para(story, ss, r)


# --- Appendices ---
def appendix_run_register(story, ss, inv, facts):
    from reportlab.platypus import Paragraph, PageBreak
    R = _R()
    logs = inv.get("logs") or []
    if not logs:
        return
    story.append(PageBreak())
    _para(story, ss, "Appendix A. Run register", "H1Custom")
    _para(story, ss, "Every detection run recorded in the project's logs, "
                     "including aborted and repeated ones. The logs themselves stay "
                     "in the project's vectors folder.")
    rows = [["Log", "Run", "Outcome", "Window", "Overlap", "Tiles", "Elapsed"]]
    for lg in logs:
        p = lg.get("parameters", {}) or {}
        for run in lg.get("runs") or []:
            rows.append([Paragraph(Path(lg["path"]).name.replace("_detection_log.txt", ""),
                                   ss["Body"]),
                         (run.get("start_ts") or run.get("ts") or "")[:16],
                         run.get("outcome", "—"), str(p.get("metric_cropsize") or "—"),
                         str(p.get("overlap") or "—"),
                         fmt_int(F._first_int(p.get("tiles to process") or p.get("tile grid")))
                         if p else "—",
                         _elapsed(run.get("elapsed_s"))])
    story.append(R._tbl_caption("Detection runs from the logs.", ss))
    story.append(R._table(rows, font_size=7,
                          col_widths=[2.2 * 72, 1.1 * 72, 0.9 * 72, 0.6 * 72,
                                      0.6 * 72, 0.5 * 72, 0.8 * 72]))


_DICTIONARY = [
    ("clast_ID", "—", "Row identifier within one CSV."),
    ("x, y", "m or px", "Centroid of the mask: projected coordinates for ortho-images, image pixels for photographs."),
    ("Clast_length", "m", "Chord of the mask through its centroid along the fitted ellipse's major axis: the apparent a-axis."),
    ("Clast_width", "m", "Chord along the minor axis: the apparent b-axis."),
    ("Ellipse_major_axis, Ellipse_minor_axis", "m", "Axes of the ellipse fitted to the outline; used for orientation."),
    ("Surface_area", "m²", "Mask pixel count × GSD²."),
    ("Perimeter", "m", "Mask outline length."),
    ("Equivalent_diameter", "m", "2·√(Surface_area/π)."),
    ("Eccentricity, Solidity", "—", "Shape descriptors of the fitted ellipse and of the mask against its convex hull."),
    ("Mean_intensity", "—", "Mean image intensity inside the mask (0–255)."),
    ("Score", "—", "Detector confidence."),
    ("Orientation", "°", "Axial bearing of the major axis, 0–180°, clockwise from image-up or grid north (90° = east–west)."),
]


def appendix_data_dictionary(story, ss, facts):
    from reportlab.platypus import Paragraph, PageBreak
    R = _R()
    story.append(PageBreak())
    _para(story, ss, "Appendix B. Data dictionary", "H1Custom")
    _para(story, ss, "Columns of every detection CSV. Lengths are stored in "
                     "metres and shown in millimetres in this report.")
    rows = [["Column", "Unit", "Meaning"]]
    for c, u, m in _DICTIONARY:
        rows.append([Paragraph(c, ss["Body"]), u, Paragraph(m, ss["Body"])])
    story.append(R._table(rows, font_size=8, col_widths=[2.0 * 72, 0.6 * 72, 4.1 * 72]))
    _spacer(story, 8)
    _para(story, ss, "Output names", "H2Custom")
    _para(story, ss,
          "Every analysis output starts with the site, the survey date (dated "
          "projects) and the image it came from, then the stage: "
          "<i>&lt;site&gt;[_&lt;date&gt;]__&lt;image&gt;_ws&lt;window&gt;m.csv</i> for a "
          "window run, <i>_merged.csv</i> for the merge, "
          "<i>_&lt;field&gt;_&lt;statistic&gt;_cellsize=&lt;c&gt;m.tif</i> for a "
          "surface, <i>_zones=&lt;set&gt;.polygons.csv</i> / <i>.transects.csv</i> "
          "for zonal results.")


def appendix_glossary(story, ss, facts):
    from reportlab.platypus import Paragraph, PageBreak
    R = _R()
    story.append(PageBreak())
    _para(story, ss, "Appendix C. Glossary", "H1Custom")
    rows = [["Term", "Meaning"]]
    items = [
        ("GSD", "Ground sample distance: the ground size of one pixel."),
        ("Detection limit D<sub>min</sub>", f"{F.DETECTION_LIMIT_PX} pixels × GSD, the shortest clast segmented reliably (Soloy et al. 2020)."),
        ("D5 … D95", "Percentiles of clast length by number: 5 % of clasts are shorter than D5."),
        ("φ", "Krumbein scale, φ = −log<sub>2</sub>(D / 1 mm); coarser is more negative."),
        ("σφ, Skφ, K<sub>G</sub>", "Folk and Ward (1957) graphic sorting, skewness and kurtosis, from the φ percentiles: sorting < 0.35 very well, 0.35–0.50 well, 0.50–0.70 moderately well, 0.70–1.00 moderately, 1–2 poorly, 2–4 very poorly, > 4 extremely poorly sorted."),
        ("Merged list", "One clast list per image after the window runs are pooled and duplicates removed at the IoU threshold."),
        ("Detected areal cover", "Fraction of a cell or footprint covered by detected masks; by size class in the cover surfaces."),
        ("Axial mean, <i>R</i>", "Circular mean direction of orientations doubled to 0–360° and halved back; <i>R</i> in [0, 1] is the mean resultant length: 0 no preferred orientation, 1 all parallel."),
    ]
    if any(r["category"] == "transport" for rs in facts["rasters_by_image"].values() for r in rs):
        items.append(("Hjulström-curve velocities",
                      "Empirical fits to the Hjulström (1935) envelope for steady uniform flow about 1 m deep: deposition U = 77·D/(1 + 24·D) (Sundborg 1956) and an erosion fit valid for 0.1–10 mm grains, both of the equivalent diameter D in metres, with ρ<sub>s</sub> = 2650 kg/m³, ρ<sub>w</sub> = 1025 kg/m³. They are monotone in D and carry no information beyond the size surface."))
    for t, m in items:
        rows.append([Paragraph(t, ss["Body"]), Paragraph(m, ss["Body"])])
    story.append(R._table(rows, font_size=8, col_widths=[1.8 * 72, 4.9 * 72]))


# --- Orchestration ---
def build_story(story, ss, inv, stats, facts, options, log, *, dpi, quality):
    """Append the whole document to ``story`` in reading order."""
    from reportlab.platypus import Paragraph, PageBreak
    from reportlab.platypus.tableofcontents import TableOfContents
    R = _R()
    if options.get("include_cover", True):
        cover(story, ss, inv, facts, options, log)
        summary(story, ss, inv, stats, facts, options, log, dpi=dpi, quality=quality)
    if options.get("include_toc", True):
        toc = TableOfContents()
        toc.levelStyles = [ss["TOCH1"], ss["TOCH2"]]
        story.append(Paragraph("Contents", ss["TOCTitle"]))
        story.append(toc)
        story.append(PageBreak())
    counters = R._ensure_counters(ss)

    def guarded(label, fn, *args, **kw):
        """One malformed artefact costs its section, not the document."""
        mark, fig_mark, tbl_mark = len(story), counters.fig, counters.tbl
        try:
            return fn(*args, **kw)
        except Exception as ex:  # noqa: BLE001
            import traceback
            del story[mark:]
            counters.fig, counters.tbl = fig_mark, tbl_mark
            log(f"  [WARN] {label} could not be rendered: {type(ex).__name__}: {ex}")
            log(traceback.format_exc())
            story.append(Paragraph(
                f"<b>{label} could not be rendered</b> "
                f"<font color='#a00'>({type(ex).__name__}: {ex})</font>. The other "
                f"sections were built normally.", ss["Body"]))
            story.append(PageBreak())
            return False

    n = 1
    guarded("Data and processing", data_and_processing, story, ss, inv, facts, options, log, n); n += 1
    guarded("Detection results", detection_results, story, ss, inv, facts, options, log, n,
            dpi=dpi, quality=quality); n += 1
    if options.get("include_spatial_maps", True):
        if guarded("Spatial statistics", spatial_statistics, story, ss, inv, facts, options,
                   log, n, dpi=dpi, quality=quality):
            n += 1
    if options.get("include_zonal_statistics", True):
        if guarded("Zonal statistics", zonal_statistics, story, ss, inv, facts, options,
                   log, n, dpi=dpi, quality=quality):
            n += 1
    if options.get("include_validation", True):
        if guarded("Validation", validation, story, ss, inv, stats, facts, options, log, n,
                   dpi=dpi, quality=quality):
            n += 1
    if options.get("illustrations"):
        guarded("Illustrations", R._illustrations, story, ss, inv, options, section_num=n); n += 1
    guarded("Interpretation", interpretation, story, ss, inv, facts, options, log, n); n += 1
    references(story, ss)
    if options.get("include_appendix_logs", True):
        guarded("Run register", appendix_run_register, story, ss, inv, facts)
    if options.get("include_appendix_samples", True):
        guarded("Data dictionary", appendix_data_dictionary, story, ss, facts)
    if options.get("include_appendix_field_reference", True):
        guarded("Glossary", appendix_glossary, story, ss, facts)
