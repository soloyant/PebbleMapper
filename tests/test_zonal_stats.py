"""Regression tests for functions.zonal_stats Folk-Ward parity."""
import inspect
import numpy as np
import pytest


def test_folk_sorting_phi_differs_from_arithmetic_std_on_gravel():
    """Sanity: on a bimodal (poorly sorted) sand+gravel mixture the percentile-
    based Folk-Ward estimator differs from the arithmetic std by >=5%. This is
    the precondition that makes the sigma_phi test below meaningful."""
    from functions.zonal_stats import _phi, _folk_sorting_phi
    rng = np.random.default_rng(42)
    # Bimodal mixture: 60% fine sand around 0.5 mm, 40% cobbles around 80 mm.
    # The arithmetic std reaches further into the tails than the Folk-Ward
    # (φ95-φ5)/6.6 + (φ84-φ16)/4 estimator, so the two estimators diverge.
    fine = rng.lognormal(mean=np.log(0.0005), sigma=0.3, size=1200)
    coarse = rng.lognormal(mean=np.log(0.080), sigma=0.3, size=800)
    sizes_m = np.concatenate([fine, coarse])
    sizes_m = sizes_m[sizes_m > 0]

    phi = _phi(sizes_m)
    folk_ward = float(_folk_sorting_phi(phi))
    arithmetic = float(np.std(phi, ddof=1))

    rel_diff = abs(folk_ward - arithmetic) / max(abs(folk_ward), abs(arithmetic))
    assert rel_diff > 0.05, (
        f"distribution must distinguish estimators (got rel_diff={rel_diff:.4f}); "
        f"folk_ward={folk_ward:.4f} arithmetic={arithmetic:.4f}; "
        "adjust the test fixture")


def test_polygon_sigma_phi_uses_folk_ward_not_std():
    """The production code path that writes sigma_phi to the polygon-zonal
    CSV must call _folk_sorting_phi, not std(phi, ddof=1)."""
    import functions.zonal_stats as zs
    src = inspect.getsource(zs)
    # Locate the polygon stats sigma_phi assignment. Be permissive about formatting.
    # The bug shape: `sigma_phi=(float(_phi(vals).std(ddof=1)) ...`
    # The fix shape: `sigma_phi=(float(_folk_sorting_phi(_phi(vals))) ...`
    # Strategy: find every line containing 'sigma_phi' and a '=' assignment;
    # require at least one such line to reference _folk_sorting_phi, and
    # require that NO sigma_phi assignment line still calls .std(ddof=
    sigma_lines = [ln for ln in src.splitlines()
                   if "sigma_phi" in ln and "=" in ln and "_folk" not in ln.lower() and "phi" in ln]
    # Filter to lines that actually compute (call _phi or std or similar), excluding
    # dataclass field declarations and docstrings.
    computing = [ln for ln in sigma_lines if (".std(" in ln) or ("_folk_sorting_phi" in ln)]
    assert not any(".std(" in ln and "_folk_sorting_phi" not in ln for ln in computing), (
        f"sigma_phi must be computed via _folk_sorting_phi; "
        f"found std-based site(s): {computing!r}")
    assert "_folk_sorting_phi(_phi(" in src or "_folk_sorting_phi(\n" in src, (
        "_folk_sorting_phi should appear at the polygon sigma_phi assignment")


def test_plot_zonal_map_renders_raster_with_zone_and_transect(tmp_path):
    """plot_zonal_map writes a PNG combining the product raster, a named zone
    polygon, and a named transect line."""
    import json
    from osgeo import gdal, osr
    from functions import zonal_stats as zs

    nx = ny = 30
    px = 0.5
    ox, oy = 100.0, 110.0
    ras = tmp_path / "Clast_length_q50.tif"
    ds = gdal.GetDriverByName("GTiff").Create(str(ras), nx, ny, 1,
                                              gdal.GDT_Float32)
    ds.SetGeoTransform([ox, px, 0, oy, 0, -px])
    srs = osr.SpatialReference(); srs.ImportFromEPSG(32630)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.WriteArray(np.tile(np.linspace(0.02, 0.12, nx), (ny, 1)).astype("float32"))
    band.SetNoDataValue(-9999.0)
    ds = None

    xmin, xmax = ox, ox + nx * px
    ymin, ymax = oy - ny * px, oy
    vec = tmp_path / "features.geojson"
    vec.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"name": "ZoneA"},
             "geometry": {"type": "Polygon", "coordinates": [[
                 [xmin + 2, ymin + 2], [xmin + 6, ymin + 2],
                 [xmin + 6, ymin + 6], [xmin + 2, ymin + 6],
                 [xmin + 2, ymin + 2]]]}},
            {"type": "Feature", "properties": {"name": "T1"},
             "geometry": {"type": "LineString", "coordinates": [
                 [xmin + 1, (ymin + ymax) / 2],
                 [xmax - 1, (ymin + ymax) / 2]]}},
        ]}), encoding="utf-8")

    out_png = tmp_path / "zonal_map.png"
    res = zs.plot_zonal_map(ras, vec, out_png, field_name="Clast_length",
                            polygon_id_field="name", transect_id_field="name")
    from pathlib import Path as _P
    assert _P(res).exists() and _P(res).stat().st_size > 5000


# ---------------------------------------------------------------------------
# Provenance columns: the polygon CSV must record WHAT it summarises
# ---------------------------------------------------------------------------

def test_polygon_csv_records_field_and_unit(tmp_path):
    """Without these columns a zonal CSV is unreadable after the fact: 'mean'
    and 'D50' alone cannot distinguish a Clast_length run from a Clast_width
    one, and neither states a unit."""
    import csv
    import json
    import functions.zonal_stats as zs

    csv_in = tmp_path / "clasts.csv"
    with open(csv_in, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "Clast_length", "Clast_width"])
        for i in range(20):
            w.writerow([10 + i * 0.1, 10 + i * 0.1, 0.05 + i * 0.001,
                        0.03 + i * 0.001])

    vec = tmp_path / "zones.geojson"
    vec.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"name": "ZoneA"},
             "geometry": {"type": "Polygon", "coordinates": [[
                 [9, 9], [13, 9], [13, 13], [9, 13], [9, 9]]]}},
        ]}), encoding="utf-8")

    out = tmp_path / "out.polygons.csv"
    rows = zs.zonal_polygon_stats_from_csv(
        str(csv_in), str(vec), str(out),
        field_name="Clast_length", id_field="name")

    assert rows and rows[0].count > 0, "fixture must put clasts inside the zone"
    with open(out, newline="", encoding="utf-8") as fh:
        data = list(csv.DictReader(fh))
    assert data[0]["field"] == "Clast_length"
    assert data[0]["unit"], "unit column must not be empty"

    # A second field must be self-describing too — this is the pair that used
    # to be indistinguishable once written to disk.
    out2 = tmp_path / "out2.polygons.csv"
    zs.zonal_polygon_stats_from_csv(
        str(csv_in), str(vec), str(out2),
        field_name="Clast_width", id_field="name")
    with open(out2, newline="", encoding="utf-8") as fh:
        data2 = list(csv.DictReader(fh))
    assert data2[0]["field"] == "Clast_width"


def test_polygon_csv_keeps_polygon_id_first(tmp_path):
    """polygon_id must stay column 0 — the GUI table uses it as the row key
    and the provenance columns were appended specifically to avoid shifting it.
    """
    import csv
    import json
    import functions.zonal_stats as zs

    csv_in = tmp_path / "clasts.csv"
    with open(csv_in, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "Clast_length"])
        for i in range(10):
            w.writerow([10 + i * 0.1, 10 + i * 0.1, 0.05 + i * 0.001])

    vec = tmp_path / "zones.geojson"
    vec.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"name": "ZoneA"},
             "geometry": {"type": "Polygon", "coordinates": [[
                 [9, 9], [13, 9], [13, 13], [9, 13], [9, 9]]]}},
        ]}), encoding="utf-8")

    out = tmp_path / "out.polygons.csv"
    zs.zonal_polygon_stats_from_csv(
        str(csv_in), str(vec), str(out),
        field_name="Clast_length", id_field="name")
    with open(out, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert header[0] == "polygon_id"
    assert header[-2:] == ["field", "unit"]


# ---------------------------------------------------------------------------
# Zonal auto-filenames must encode the field
# ---------------------------------------------------------------------------

def test_zonal_auto_name_distinguishes_fields():
    """Two runs differing only in field must not collide. Polygon mode derives
    its stem from the clast CSV, which is identical across fields, so before
    the field token the second run silently overwrote the first."""
    from gui.app import zonal_auto_out_name
    a = zonal_auto_out_name("clasts", "zones", "Clast_length", "polygons")
    b = zonal_auto_out_name("clasts", "zones", "Clast_width", "polygons")
    assert a != b
    # Naming grammar: explicit key=value tokens.
    assert a == "clasts_zones=zones_field=Clast_length_id=index.polygons.csv"
    assert b == "clasts_zones=zones_field=Clast_width_id=index.polygons.csv"
    # ... and so must two runs differing only in how the zones are labelled.
    assert zonal_auto_out_name("clasts", "zones", "Clast_length", "polygons", "name") != a


def test_zonal_auto_name_sanitises_and_tolerates_empty_field():
    from gui.app import zonal_auto_out_name
    # Characters that are illegal / awkward in a filename get folded.
    n = zonal_auto_out_name("c", "z", "D50 (mm)", "polygons")
    assert "/" not in n and " " not in n and "(" not in n
    assert n.endswith(".polygons.csv")
    # No field → no field token, so nothing breaks when field is unset.
    assert zonal_auto_out_name("c", "z", "", "transects") == "c_zones=z.transects.csv"


def test_a_non_size_field_labels_its_50th_percentile_P50():
    """The moment block already writes a 'median' column; calling the 50th
    percentile 'median' too gave a CSV with two identical headers, which
    pandas silently renames on the way back in."""
    from functions.zonal_stats import _percentile_label
    assert _percentile_label("Score", 50) == "P50"
    assert _percentile_label("Score", 84) == "P84"
    # A size field keeps the D grammar.
    assert _percentile_label("Clast_length", 50) == "D50"


def test_report_recovers_vector_name_despite_field_token(tmp_path):
    """report.py splits the stem on '__' to label the vector layer. Appending
    '__<field>' must not leak into that label."""
    import csv
    from functions import report as rp

    zonal = tmp_path / "output_results" / "zonal"
    zonal.mkdir(parents=True)
    out = zonal / "clasts__myzones__Clast_length.polygons.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["polygon_id", "count", "mean", "field", "unit"])
        w.writerow(["ZoneA", "12", "0.05", "Clast_length", "m"])

    inv = rp.collect_project_inventory(tmp_path)
    entries = [e for e in inv["zonal"] if e["path"].name == out.name]
    assert entries, "zonal CSV must be inventoried"
    e = entries[0]
    assert e["vector"] == "myzones", f"vector label leaked the field: {e['vector']!r}"
    assert e["field"] == "Clast_length"
    assert e["unit"] == "m"


# ---------------------------------------------------------------------------
# Save guard: staged geometry must belong to the image the file is named after
# ---------------------------------------------------------------------------

def _feat(pts):
    return {"pts": [{"x": x, "y": y} for x, y in pts]}


def test_features_world_bbox_and_raster_bbox():
    from gui.app import features_world_bbox, raster_world_bbox
    assert features_world_bbox([]) is None
    assert features_world_bbox([{"pts": [{"px": 1, "py": 2}]}]) is None
    assert features_world_bbox([_feat([(1, 5), (3, 9)])]) == (1, 3, 5, 9)

    # n1 ortho: origin top-left, negative y pixel size.
    gt = (558250.108684739, 0.0044018, 0.0, 6981595.600082761, 0.0, -0.0044018)
    bbox = raster_world_bbox(gt, 5874, 3989)
    assert bbox is not None
    x0, x1, y0, y1 = bbox
    assert round(x0, 1) == 558250.1 and round(x1, 1) == 558276.0
    assert round(y1, 1) == 6981595.6 and y0 < y1
    assert raster_world_bbox(None, 10, 10) is None
    assert raster_world_bbox(gt, 0, 0) is None


def test_features_match_image_detects_the_real_mismatch():
    """The exact condition behind example_03_UAV's mislabelled zone file:
    zones drawn on n2, saved under n1's name."""
    from gui.app import features_match_image, raster_world_bbox

    gt1 = (558250.108684739, 0.0044018, 0.0, 6981595.600082761, 0.0, -0.0044018)
    n1 = raster_world_bbox(gt1, 5874, 3989)
    gt2 = (558293.620, 0.0044018, 0.0, 6981605.176, 0.0, -0.0044018)
    n2 = raster_world_bbox(gt2, 3688, 2235)

    zones_on_n2 = [_feat([(558295.503, 6981596.966), (558309.039, 6981604.347)])]
    from gui.app import features_world_bbox
    zb = features_world_bbox(zones_on_n2)

    assert features_match_image(zb, n2) is True, "must accept the right image"
    assert features_match_image(zb, n1) is False, "must reject the wrong image"


def test_features_match_image_is_permissive_when_unknown():
    """Must never block a save just because there is nothing to compare —
    a non-georeferenced image or an empty canvas has to keep working."""
    from gui.app import features_match_image
    assert features_match_image(None, (0, 1, 0, 1)) is True
    assert features_match_image((0, 1, 0, 1), None) is True
    assert features_match_image(None, None) is True
    # Touching boxes count as overlapping, not as a mismatch.
    assert features_match_image((1, 2, 1, 2), (2, 3, 2, 3)) is True


# ---------------------------------------------------------------------------
# Units: the CSV must be in display units and say so
# ---------------------------------------------------------------------------

def _clast_csv(tmp_path, n=20):
    import csv
    p = tmp_path / "clasts.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "Clast_length"])
        for i in range(n):
            w.writerow([10 + i * 0.1, 10 + i * 0.1, 0.05 + i * 0.001])
    return p


def _zones_geojson(tmp_path, rings, name="zones.geojson"):
    import json
    feats = []
    for i, ring in enumerate(rings):
        feats.append({
            "type": "Feature",
            "properties": {"name": f"Zone{i}"},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
    p = tmp_path / name
    p.write_text(json.dumps({"type": "FeatureCollection", "features": feats}),
                 encoding="utf-8")
    return p


def test_polygon_csv_is_written_in_display_units(tmp_path):
    """Lengths are stored in metres and displayed in millimetres everywhere
    else; the CSV must agree with the report and maps, not contradict them."""
    import csv
    import functions.zonal_stats as zs

    csv_in = _clast_csv(tmp_path)
    vec = _zones_geojson(tmp_path, [[[9, 9], [13, 9], [13, 13], [9, 13], [9, 9]]])
    out = tmp_path / "out.polygons.csv"
    rows = zs.zonal_polygon_stats_from_csv(
        str(csv_in), str(vec), str(out),
        field_name="Clast_length", id_field="name")

    with open(out, newline="", encoding="utf-8") as fh:
        rec = list(csv.DictReader(fh))[0]

    assert rec["unit"] == "mm"
    # Stored mean is ~0.0595 m; the file must say ~59.5, not 0.0595.
    assert 10.0 < float(rec["mean"]) < 100.0
    assert abs(float(rec["mean"]) - rows[0].mean * 1000.0) < 1e-6
    for col in ("std", "median", "iqr", "min", "max", "range", "D50"):
        assert float(rec[col]) > 1.0, f"{col} looks unconverted: {rec[col]}"


def test_polygon_csv_leaves_non_field_columns_unscaled(tmp_path):
    """Converting the field's statistics must not touch quantities that are
    not in the field's unit."""
    import csv
    import functions.zonal_stats as zs

    csv_in = _clast_csv(tmp_path)
    vec = _zones_geojson(tmp_path, [[[9, 9], [13, 9], [13, 13], [9, 13], [9, 9]]])
    out = tmp_path / "out.polygons.csv"
    rows = zs.zonal_polygon_stats_from_csv(
        str(csv_in), str(vec), str(out),
        field_name="Clast_length", id_field="name")
    r = rows[0]
    with open(out, newline="", encoding="utf-8") as fh:
        rec = list(csv.DictReader(fh))[0]

    assert float(rec["area_m2"]) == pytest.approx(r.area_m2, rel=1e-5)
    assert float(rec["density"]) == pytest.approx(r.density, rel=1e-5)
    assert float(rec["cv"]) == pytest.approx(r.cv, rel=1e-5)
    assert float(rec["skewness"]) == pytest.approx(r.skewness, rel=1e-5)
    assert float(rec["kurtosis"]) == pytest.approx(r.kurtosis, rel=1e-5)
    assert float(rec["sigma_phi"]) == pytest.approx(r.sigma_phi, rel=1e-5)


def test_dimensionless_field_passes_through_unconverted(tmp_path):
    """A field whose multiplier is 1.0 must not be scaled."""
    import csv
    import functions.zonal_stats as zs

    p = tmp_path / "clasts.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "Orientation"])
        for i in range(12):
            w.writerow([10 + i * 0.1, 10 + i * 0.1, 30 + i])
    vec = _zones_geojson(tmp_path, [[[9, 9], [13, 9], [13, 13], [9, 13], [9, 9]]])
    out = tmp_path / "orient.polygons.csv"
    rows = zs.zonal_polygon_stats_from_csv(
        str(p), str(vec), str(out), field_name="Orientation", id_field="name")
    with open(out, newline="", encoding="utf-8") as fh:
        rec = list(csv.DictReader(fh))[0]
    assert float(rec["mean"]) == pytest.approx(rows[0].mean, rel=1e-6)


# ---------------------------------------------------------------------------
# Coverage: a measured zero is not the same as missing data
# ---------------------------------------------------------------------------

def test_coverage_separates_measured_zero_from_outside_data(tmp_path):
    """A zone inside the surveyed area with no clasts is an observation; a
    zone outside it is missing data. The CSV must tell them apart."""
    import csv
    import functions.zonal_stats as zs

    # Clasts lie on the diagonal x == y, from 10.0 to 11.9. So a box in the
    # top-left corner of that bbox is inside the surveyed area but off the
    # diagonal, and therefore genuinely empty.
    csv_in = _clast_csv(tmp_path)
    vec = _zones_geojson(tmp_path, [
        [[9, 9], [13, 9], [13, 13], [9, 13], [9, 9]],        # contains clasts
        [[10.0, 11.5], [10.5, 11.5], [10.5, 11.9],
         [10.0, 11.9], [10.0, 11.5]],                        # inside bbox, empty
        [[500, 500], [510, 500], [510, 510], [500, 510], [500, 500]],  # far away
    ])
    out = tmp_path / "cov.polygons.csv"
    zs.zonal_polygon_stats_from_csv(
        str(csv_in), str(vec), str(out),
        field_name="Clast_length", id_field="name")

    with open(out, newline="", encoding="utf-8") as fh:
        recs = {r["polygon_id"]: r for r in csv.DictReader(fh)}

    assert len(recs) == 3, "every zone must keep its row"
    assert int(recs["Zone0"]["count"]) > 0
    assert recs["Zone0"]["coverage"] == "inside"
    assert int(recs["Zone1"]["count"]) == 0
    assert recs["Zone1"]["coverage"] == "inside", "empty but surveyed"
    assert int(recs["Zone2"]["count"]) == 0
    assert recs["Zone2"]["coverage"] == "outside", "not surveyed at all"


# ---------------------------------------------------------------------------
# Transect runs must record what they profiled
# ---------------------------------------------------------------------------

def test_transect_csv_records_field(tmp_path):
    import csv
    import json
    import numpy as _np
    from osgeo import gdal, osr
    from functions import zonal_stats as zs

    nx = ny = 20
    px = 0.5
    ox, oy = 100.0, 110.0
    ras = tmp_path / "D50.tif"
    ds = gdal.GetDriverByName("GTiff").Create(str(ras), nx, ny, 1,
                                              gdal.GDT_Float32)
    ds.SetGeoTransform([ox, px, 0, oy, 0, -px])
    srs = osr.SpatialReference(); srs.ImportFromEPSG(32630)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(
        _np.full((ny, nx), 0.05, dtype="float32"))
    ds = None

    vec = tmp_path / "line.geojson"
    vec.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"name": "T1"},
                      "geometry": {"type": "LineString", "coordinates": [
                          [ox + 1, oy - 5], [ox + 8, oy - 5]]}}]}),
        encoding="utf-8")

    out = tmp_path / "t.transects.csv"
    zs.zonal_transect_profile(str(ras), str(vec), str(out),
                              step_m=0.5, id_field="name",
                              field_name="Clast_length")
    with open(out, newline="", encoding="utf-8") as fh:
        recs = list(csv.DictReader(fh))
    assert recs, "transect must produce samples"
    assert recs[0]["field"] == "Clast_length"

    # Omitting the field keeps the previous schema, so old readers still work.
    out2 = tmp_path / "t2.transects.csv"
    zs.zonal_transect_profile(str(ras), str(vec), str(out2), step_m=0.5,
                              id_field="name")
    with open(out2, newline="", encoding="utf-8") as fh:
        assert "field" not in next(csv.reader(fh))


# ---------------------------------------------------------------------------
# Zone-set naming: identity independent of any image, and path-safe
# ---------------------------------------------------------------------------

def test_zone_set_name_rejects_path_escapes():
    """The name becomes a filename joined onto the project directory, so a
    separator or parent reference must never survive."""
    from gui.app import sanitise_zone_set_name
    for bad in ("../evil", "..", "a/b", r"a\b", "/abs", r"C:\x",
                r"..\..\etc", "", "   ", "\x00null"):
        assert sanitise_zone_set_name(bad) == "", f"must reject {bad!r}"


def test_zone_set_name_accepts_ordinary_names():
    from gui.app import sanitise_zone_set_name
    for good in ("upper_beach_zones", "Zones 2023", "transects.v2",
                 "site-A_profiles"):
        assert sanitise_zone_set_name(good) == good


def test_field_from_raster_prefers_sidecar(tmp_path):
    """The rasterize step records the field in a sidecar; that is
    authoritative and must win over filename guessing."""
    import json
    from gui.app import field_from_raster

    ras = tmp_path / "img_merged_Clast_length_D50_cellsize=1.0m.tif"
    ras.write_bytes(b"")
    assert field_from_raster(ras) == "Clast_length"     # from the filename

    side = tmp_path / (ras.name + ".json")
    side.write_text(json.dumps({"field": "Equivalent_diameter"}),
                    encoding="utf-8")
    assert field_from_raster(ras) == "Equivalent_diameter"   # sidecar wins


def test_field_from_raster_never_returns_empty(tmp_path):
    from gui.app import field_from_raster
    ras = tmp_path / "totally_unrecognisable.tif"
    ras.write_bytes(b"")
    assert field_from_raster(ras)


# ---------------------------------------------------------------------------
# CRS mismatch must be reported, not silently mis-located
# ---------------------------------------------------------------------------

def test_describe_crs_mismatch(tmp_path):
    import json
    import numpy as _np
    from osgeo import gdal, osr
    from functions import zonal_stats as zs

    def _raster(name, epsg):
        p = tmp_path / name
        ds = gdal.GetDriverByName("GTiff").Create(str(p), 5, 5, 1,
                                                  gdal.GDT_Float32)
        ds.SetGeoTransform([0, 1, 0, 10, 0, -1])
        srs = osr.SpatialReference(); srs.ImportFromEPSG(epsg)
        ds.SetProjection(srs.ExportToWkt())
        ds.GetRasterBand(1).WriteArray(_np.zeros((5, 5), dtype="float32"))
        ds = None
        return p

    def _vec(name, epsg):
        p = tmp_path / name
        crs = {"type": "name",
               "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}}
        p.write_text(json.dumps({
            "type": "FeatureCollection", "crs": crs,
            "features": [{"type": "Feature", "properties": {},
                          "geometry": {"type": "Polygon", "coordinates": [
                              [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}}]}),
            encoding="utf-8")
        return p

    same = zs.describe_crs_mismatch(_vec("a.geojson", 32630),
                                    _raster("a.tif", 32630))
    assert same is None, "matching CRSs must not warn"

    diff = zs.describe_crs_mismatch(_vec("b.geojson", 32631),
                                    _raster("b.tif", 32630))
    assert diff and "vector layer" in diff.lower()

    # Unanswerable cases must stay silent rather than cry mismatch.
    assert zs.describe_crs_mismatch(tmp_path / "nope.geojson",
                                    _raster("c.tif", 32630)) is None


# ---------------------------------------------------------------------------
# Back-compat: a CSV with no unit column is in stored units, unchanged
# ---------------------------------------------------------------------------

def test_report_leaves_pre_units_csv_untouched(tmp_path):
    """Files written before the units work have no `unit` column. They must
    keep rendering exactly as before and must never be relabelled as mm."""
    import csv
    from functions import report as rp

    zonal = tmp_path / "output_results" / "zonal"
    zonal.mkdir(parents=True)
    old = zonal / "clasts__myzones.polygons.csv"
    with open(old, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["polygon_id", "count", "mean", "density"])
        w.writerow(["ZoneA", "12", "0.0523", "44.1"])

    inv = rp.collect_project_inventory(tmp_path)
    entry = [e for e in inv["zonal"] if e["path"].name == old.name][0]
    assert entry["field"] is None
    assert entry["unit"] is None
    assert entry["vector"] == "myzones"

    # The value on disk is untouched by inventory collection.
    with open(old, newline="", encoding="utf-8") as fh:
        assert list(csv.DictReader(fh))[0]["mean"] == "0.0523"


def test_crs_check_does_not_fire_on_pebblemapper_zone_files(tmp_path):
    """A GeoJSON with no `crs` member is WGS 84 by definition, and that is
    exactly what this tool writes — while its coordinates are projected
    metres. Warning there would fire on every ordinary run, so the check must
    stay silent whenever the coordinates land inside the raster anyway."""
    import json
    import numpy as _np
    from osgeo import gdal, osr
    from functions import zonal_stats as zs

    ras = tmp_path / "r.tif"
    ds = gdal.GetDriverByName("GTiff").Create(str(ras), 40, 40, 1,
                                              gdal.GDT_Float32)
    ds.SetGeoTransform([9.0, 0.1, 0, 13.0, 0, -0.1])
    srs = osr.SpatialReference(); srs.ImportFromEPSG(32630)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(_np.zeros((40, 40), dtype="float32"))
    ds = None

    # No "crs" member, coordinates inside the raster extent.
    vec = tmp_path / "zones_no_crs.geojson"
    vec.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"name": "Z"},
                      "geometry": {"type": "Polygon", "coordinates": [[
                          [9.5, 11.0], [12.5, 11.0], [12.5, 12.0],
                          [9.5, 12.0], [9.5, 11.0]]]}}]}), encoding="utf-8")

    assert zs.describe_crs_mismatch(str(vec), str(ras)) is None

    # Genuinely elsewhere → still reported.
    far = tmp_path / "far.geojson"
    far.write_text(json.dumps({
        "type": "FeatureCollection",
        "crs": {"type": "name",
                "properties": {"name": "urn:ogc:def:crs:EPSG::32631"}},
        "features": [{"type": "Feature", "properties": {"name": "Z"},
                      "geometry": {"type": "Polygon", "coordinates": [[
                          [500000, 5000000], [500100, 5000000],
                          [500100, 5000100], [500000, 5000100],
                          [500000, 5000000]]]}}]}), encoding="utf-8")
    assert zs.describe_crs_mismatch(str(far), str(ras))


# ---------------------------------------------------------------------------
# Mode control: the rule the Analysis-card toggle drives (review fix 1)
# ---------------------------------------------------------------------------

def test_zonal_shape_for_mode():
    """Transect mode can only draw transects, and leaving it must drop the
    stale shape or the next commit is rejected for having < 3 points."""
    from gui.app import zonal_shape_for_mode
    assert zonal_shape_for_mode("transects", "polygon") == "transect"
    assert zonal_shape_for_mode("transects", "circle") == "transect"
    assert zonal_shape_for_mode("transects", "transect") == "transect"
    # Back to polygons: a stale transect becomes a polygon...
    assert zonal_shape_for_mode("polygons", "transect") == "polygon"
    # ...but a real polygon shape is preserved.
    for shape in ("polygon", "rectangle", "circle"):
        assert zonal_shape_for_mode("polygons", shape) == shape


def test_a_corrupted_shape_never_crashes_the_commit():
    """Regression.

    The migrated Zonal canvas wrote the mode-derived "transect" into the field
    its polygon-only toggle is bound to. Quasar, handed a value it has no
    option for, emitted null back and set the shape to None — and the canvas's
    ``min_pts`` lookup raised ``KeyError: None``, so Finalise failed every time
    and a transect could never be committed: the clicks just piled up as loose
    vertices. Normalising is what keeps a bad value from reaching that lookup.
    """
    from gui.app import normalise_draw_shape
    for bad in (None, "", "nonsense", 0):
        assert normalise_draw_shape(bad) == "polygon"
    for good in ("polygon", "rectangle", "circle", "transect"):
        assert normalise_draw_shape(good) == good


def test_zonal_never_pushes_a_mode_shape_into_the_bound_toggle():
    """The corruption above is only impossible while the mode-derived shape is
    NOT assigned into the toggle-bound ctx field. The analysis mode must reach
    the canvas through effective_shape_getter instead."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "gui" / "app.py").read_text(
        encoding="utf-8")
    assert "shape = zonal_shape_for_mode(" not in src, (
        "assigning zonal_shape_for_mode() into the toggle-bound shape field "
        "re-introduces the Quasar null-corruption; pass it through "
        "effective_shape_getter instead")
    assert "effective_shape_getter=_zonal_effective_shape" in src, (
        "the Zonal canvas must inject its mode-authoritative shape getter")


def test_the_mode_is_chosen_once_and_still_drives_the_handler():
    """The mode is chosen once, in the sub-tab.

    This used to require a mode toggle in BOTH the source canvas and the
    analysis card, and asserted they stayed in step. They did — both bound
    `zonal_mode`, so they could never diverge — but the user could not see
    that the two controls were one control: the second appeared a screen
    below the first with no sign it was the same switch. The choice is now
    the Zonal sub-tab, made once, and the sub-tab still drives the same
    handler so the draw-shape follows the mode.
    """
    import inspect
    import gui.app as app
    src = inspect.getsource(app)
    assert src.count('.bind_value(state, "zonal_mode")') == 0, (
        "the sub-tab is the mode chooser; no toggle should bind zonal_mode")
    assert '_zonal_subtabs.bind_value(state, "zonal_view")' in src, (
        "the sub-tab bar must be the single mode chooser")
    assert "_on_zonal_mode_change" in src, (
        "switching sub-tabs must still sync the draw-shape to the mode")


# ---------------------------------------------------------------------------
# Zone-set save path: all three branches, headless (review fix 2)
# ---------------------------------------------------------------------------

def test_zonal_set_save_path_uses_project_geometries(tmp_path, monkeypatch):
    """A named set belongs in the project's geometries folder — the branch the
    GUI could not demonstrate because the project picker needs a browser."""
    import functions.layout as layout_mod
    from gui.app import zonal_set_save_path
    monkeypatch.setattr(layout_mod, "DATASETS_ROOT", tmp_path)

    p = zonal_set_save_path("", "upper_beach_zones", "S", "")
    assert p == tmp_path / "S" / "input_data" / "geometries" / "upper_beach_zones.geojson"


def test_zonal_set_save_path_explicit_override_wins(tmp_path):
    from pathlib import Path
    from gui.app import zonal_set_save_path
    explicit = str(tmp_path / "somewhere" / "custom.geojson")
    assert zonal_set_save_path(explicit, "ignored", "S", "") == Path(explicit)


def test_zonal_set_save_path_falls_back_to_image_folder(tmp_path):
    from gui.app import zonal_set_save_path
    img = tmp_path / "imgs" / "A.tif"
    p = zonal_set_save_path("", "upper_beach_zones", "", str(img))
    assert p == tmp_path / "imgs" / "upper_beach_zones.geojson"


def test_zonal_set_save_path_returns_none_without_a_usable_name(tmp_path):
    """None is what lets the callers prompt instead of inventing a name from
    whichever image happens to be loaded."""
    from gui.app import zonal_set_save_path
    assert zonal_set_save_path("", "", "S", str(tmp_path / "A.tif")) is None
    assert zonal_set_save_path("", "   ", "S", "") is None
    assert zonal_set_save_path("", "../escape", "S", "") is None
    assert zonal_set_save_path("", "a/b", "S", "") is None


# ---------------------------------------------------------------------------
# Quick-look figure must not mix units (review fix 3)
# ---------------------------------------------------------------------------

def test_quick_look_violin_values_match_the_summary_csv_units(tmp_path):
    """The scatter is drawn from the summary CSV (display units) and the
    violin from the raw per-clast CSV. Both must be in the same unit, or one
    figure shows the same quantity 1000x apart."""
    import csv
    import functions.zonal_stats as zs
    from functions.units import field_unit_and_factor

    csv_in = _clast_csv(tmp_path)                 # Clast_length 0.05..0.069 m
    vec = _zones_geojson(tmp_path, [[[9, 9], [13, 9], [13, 13], [9, 13], [9, 9]]])
    out = tmp_path / "q.polygons.csv"
    zs.zonal_polygon_stats_from_csv(str(csv_in), str(vec), str(out),
                                    field_name="Clast_length", id_field="name")
    summary_mean = float(list(csv.DictReader(
        open(out, newline="", encoding="utf-8")))[0]["mean"])

    # What the violin panel now plots: raw values scaled by the same factor.
    factor = field_unit_and_factor("Clast_length")[1]
    import pandas as pd
    raw = pd.read_csv(csv_in)["Clast_length"].to_numpy() * factor

    assert factor == 1000.0
    assert raw.min() <= summary_mean <= raw.max(), (
        f"violin range {raw.min()}..{raw.max()} must bracket the summary mean "
        f"{summary_mean} — if it does not, the two panels are in different units")


def test_zonal_column_unit_labels_only_field_derived_columns():
    from gui.app import zonal_column_unit
    for col in ("mean", "std", "median", "iqr", "min", "max", "range",
                "D5", "D50", "D95", "P25"):
        assert zonal_column_unit(col, "mm") == "mm", col
    for col in ("count", "area_m2", "density", "cv", "skewness", "kurtosis",
                "mean_phi", "sigma_phi", "dem_mean", "coverage", "polygon_id"):
        assert zonal_column_unit(col, "mm") == "", col


def test_zonal_axis_label_leaves_pre_units_csvs_bare():
    """A CSV written before the unit column exists supplies '', and its axes
    must stay unlabelled rather than claim a unit they are not in."""
    from gui.app import zonal_axis_label, zonal_column_unit
    assert zonal_column_unit("mean", "") == ""
    assert zonal_axis_label("mean", "") == "mean"
    assert zonal_axis_label("mean", "mm") == "mean [mm]"


def test_per_polygon_values_scales_into_display_units():
    """Pin the actual code path: _per_polygon_values is a closure and cannot be
    imported, so assert on its source that the clipped values are scaled by the
    field's display factor before they reach the violin."""
    import inspect
    import gui.app as app
    src = inspect.getsource(app)
    i = src.find("def _per_polygon_values(")
    assert i > 0, "_per_polygon_values not found"
    body = src[i:i + 2500]
    assert "field_unit_and_factor" in body, (
        "_per_polygon_values must look up the display factor")
    assert "_factor" in body and "vals[inside] * _factor" in body, (
        "clipped per-clast values must be scaled into display units before "
        "being plotted beside the summary CSV")


# ---------------------------------------------------------------------------
# A dimensionless field must not grow a fabricated unit (review pass 2)
# ---------------------------------------------------------------------------

def test_dimensionless_field_axis_label_has_no_unit(tmp_path):
    """Score has no unit, so its CSV cell is empty — and pandas reads an
    all-empty column as NaN. The axis must read "mean", never "mean [nan]"."""
    import csv
    import pandas as pd
    import functions.zonal_stats as zs
    from functions.units import field_unit_and_factor
    from gui.app import zonal_axis_label, zonal_column_unit

    assert field_unit_and_factor("Score")[0] == "", "fixture assumes no unit"

    src = tmp_path / "c.csv"
    with open(src, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "Score"])
        for i in range(12):
            w.writerow([10 + i * 0.1, 10 + i * 0.1, 0.80 + i * 0.01])
    vec = _zones_geojson(tmp_path, [[[9, 9], [13, 9], [13, 13], [9, 13], [9, 9]]])
    out = tmp_path / "score.polygons.csv"
    zs.zonal_polygon_stats_from_csv(str(src), str(vec), str(out),
                                    field_name="Score", id_field="name")

    df = pd.read_csv(out)
    # Read the unit exactly as the quick-look does.
    raw = df["unit"].iloc[0]
    csv_unit = str(raw) if pd.notna(raw) else ""
    assert zonal_column_unit("mean", csv_unit) == ""
    assert zonal_axis_label("mean", zonal_column_unit("mean", csv_unit)) == "mean"

    # Even if the literal string "nan" reaches the helper, it is not a unit.
    for bogus in ("nan", "NaN", "None", "  ", None):
        assert zonal_column_unit("mean", bogus) == "", repr(bogus)

    # A real unit still survives.
    assert zonal_column_unit("mean", "mm") == "mm"


def test_provenance_columns_are_not_offered_as_plot_axes(tmp_path):
    """`unit` is all-NaN for a dimensionless field, so pandas types it float64
    and it would otherwise appear in the scatter-axis choices."""
    import csv
    import pandas as pd
    import functions.zonal_stats as zs
    from gui.app import ZONAL_PROVENANCE_COLUMNS

    src = tmp_path / "c.csv"
    with open(src, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "Score"])
        for i in range(12):
            w.writerow([10 + i * 0.1, 10 + i * 0.1, 0.80 + i * 0.01])
    vec = _zones_geojson(tmp_path, [[[9, 9], [13, 9], [13, 13], [9, 13], [9, 9]]])
    out = tmp_path / "score.polygons.csv"
    zs.zonal_polygon_stats_from_csv(str(src), str(vec), str(out),
                                    field_name="Score", id_field="name")

    df = pd.read_csv(out)
    assert "unit" in list(df.select_dtypes(include="number").columns), (
        "fixture must reproduce the all-NaN unit column")
    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if c not in ZONAL_PROVENANCE_COLUMNS]
    for col in ZONAL_PROVENANCE_COLUMNS:
        assert col not in numeric_cols
    assert "mean" in numeric_cols, "real statistics must remain selectable"

    # Pin the app's own axis list, not just this test's copy of the rule.
    import inspect
    import gui.app as app
    src = inspect.getsource(app)
    i = src.find('select_dtypes(include="number")')
    assert i > 0
    assert "ZONAL_PROVENANCE_COLUMNS" in src[max(0, i - 300):i + 300], (
        "the scatter-axis list must exclude the provenance columns")


def test_a_zonal_map_labels_sizes_like_the_map_tab():
    """The zonal publication map labelled a size raster in metres while the Map
    tab labelled the same file in millimetres."""
    from functions import map_export as mx
    stem = "Project__ortho_merged_Clast_length_D50_cellsize=1.0m"
    parameter = mx._parameter_from_stem(stem)
    unit, factor = mx._resolve_unit_for_field(stem, [0.02, 0.05], "metric", "auto",
                                              parameter=parameter)
    assert (unit, factor) == ("mm", 1000.0)
    assert mx._raster_label(stem, parameter, unit) == "Clast length [mm]"


def test_a_transect_profile_is_labelled_by_its_statistic_in_display_units(tmp_path):
    """The profile of a D50 raster was drawn in metres and labelled "Clast
    length" with no unit."""
    import matplotlib
    matplotlib.use("Agg")
    from unittest import mock
    from functions.zonal_stats import TransectSample, plot_transect_profile
    s = TransectSample("t1", np.array([0.0, 1.0, 2.0]), np.array([0.040, 0.045, 0.050]))
    captured = {}
    import matplotlib.axes
    real_plot = matplotlib.axes.Axes.plot

    def spy(self, *a, **k):
        captured.setdefault("y", a[1])
        return real_plot(self, *a, **k)
    with mock.patch.object(matplotlib.axes.Axes, "plot", spy), \
         mock.patch("matplotlib.figure.Figure.savefig"):
        with mock.patch("matplotlib.pyplot.close") as close:
            plot_transect_profile(s, tmp_path / "p.png", field_name="Clast_length", parameter="D50")
            fig = close.call_args[0][0]
    assert np.allclose(captured["y"], [40.0, 45.0, 50.0])
    assert fig.axes[0].get_ylabel() == "Clast length — median [mm]"
