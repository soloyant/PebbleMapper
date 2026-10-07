"""Filename convention: short build round-trips, legacy still parses."""
from functions import naming


# ---- build (short form) ----
def test_detection_csv_name_short():
    assert naming.detection_csv_name("Site_AB", 2.5) == "Site_AB_ws2.5m.csv"
    assert naming.detection_csv_name("Site_AB", 1.0) == "Site_AB_ws1m.csv"
    assert naming.detection_csv_name("Site_AB", 2.5, run=True) == "Site_AB_ws2.5m.run.csv"
    assert naming.detection_csv_name("Site_AB", 2.5, express=True) == "Site_AB_xprs_ws2.5m.csv"

def test_merged_csv_name_short():
    assert naming.merged_csv_name("Site_AB") == "Site_AB_merged.csv"
    assert naming.merged_csv_name("Site_AB", express=True) == "Site_AB_xprs_merged.csv"


# ---- round-trip parse of the SHORT names ----
def test_round_trip_short():
    n = naming.detection_csv_name("Marshbeach_2023_section_AB_georef", 2.5)
    assert naming.parse_window_size(n) == 2.5
    assert naming.is_merged(n) is False
    assert naming.image_stem(n) == "Marshbeach_2023_section_AB_georef"
    x = naming.detection_csv_name("Marshbeach_2023_section_AB_georef", 1.0, express=True)
    assert naming.is_express(x) is True
    assert naming.image_stem(x) == "Marshbeach_2023_section_AB_georef"
    m = naming.merged_csv_name("Marshbeach_2023_section_AB_georef")
    assert naming.is_merged(m) is True
    assert naming.image_stem(m) == "Marshbeach_2023_section_AB_georef"


# ---- backward-compat: the EXISTING field-data filenames must still parse ----
def test_legacy_window_csv_parses():
    legacy = "Marshbeach_2023_section_AB_georef_window_size=1m_individual_clast_values.csv"
    assert naming.parse_window_size(legacy) == 1.0
    assert naming.is_merged(legacy) is False
    assert naming.image_stem(legacy) == "Marshbeach_2023_section_AB_georef"

def test_legacy_window_csv_decimal():
    legacy = "Swimbeach_2023_window_size=2.5m_individual_clast_values.csv"
    assert naming.parse_window_size(legacy) == 2.5
    assert naming.image_stem(legacy) == "Swimbeach_2023"

def test_legacy_merged_csv_parses():
    legacy = "Marshbeach_2023_section_AB_georef_merged_individual_clast_values.csv"
    assert naming.is_merged(legacy) is True
    assert naming.parse_window_size(legacy) is None
    assert naming.image_stem(legacy) == "Marshbeach_2023_section_AB_georef"


# ---- the origin (site, date, image) and the zonal grammar ----
def test_origin_carries_site_and_date_unless_the_image_already_does():
    assert naming.origin_stem("IMG_0977", "Normandy_Etretat", "2020-06-10") == \
        "Normandy_Etretat_20200610__IMG_0977"
    # The stem carries the date: only the site is added.
    assert naming.origin_stem("02_etretat_20200610_alpha", "Normandy_Etretat", "2020-06-10") == \
        "Normandy_Etretat__02_etretat_20200610_alpha"
    # Undated project.
    assert naming.origin_stem("site_ortho", "Montana_Sommers", None) == "Montana_Sommers__site_ortho"
    # The stem carries the project too: no prefix, no separator.
    assert naming.origin_stem("Montana_Sommers_ortho", "Montana_Sommers", None) == "Montana_Sommers_ortho"
    # Idempotent.
    o = naming.origin_stem("IMG_0977", "Normandy_Etretat", "2020-06-10")
    assert naming.origin_stem(o, "Normandy_Etretat", "2020-06-10") == o


def test_image_stem_strips_the_origin_and_run_stem_keeps_it():
    o = naming.origin_stem("02_etretat_20200610_alpha", "Normandy_Etretat", "2020-06-10")
    for name in (naming.detection_csv_name(o, 1.0),
                 naming.detection_csv_name(o, 2.5, express=True, run=True),
                 naming.merged_csv_name(o),
                 naming.quadrat_csv_name(o),
                 naming.raster_name(o + "_merged", "Clast_length", "D50", 1.0),
                 naming.map_name(naming.raster_name(o + "_merged", "Clast_length", "D50", 1.0)[:-4]),
                 o + "_merged_Clast_length_D50_cellsize=1.0m_detection_log.txt"):
        assert naming.image_stem(name) == "02_etretat_20200610_alpha", name
        assert naming.run_stem(name) == o, name
    assert naming.same_image(naming.merged_csv_name(o), "02_etretat_20200610_alpha.tif")
    # Legacy names have no origin: unchanged behaviour.
    assert naming.image_stem("Site_AB_window_size=2.5m_individual_clast_values.csv") == "Site_AB"
    assert naming.run_stem("Site_AB_window_size=2.5m_individual_clast_values.csv") == "Site_AB"
    assert naming.image_stem("IMG_0977_individual_clasts.csv") == "IMG_0977"


def test_zonal_names_are_explicit_and_parse_both_grammars():
    raster = "Normandy_Etretat__02_alpha_merged_Clast_length_D50_cellsize=1.0m"
    n = naming.zonal_out_name(raster, "zones_upper beach", "Clast_length", "transects")
    # The raster already names the field: no field token; spaces become hyphens.
    assert n == raster + "_zones=upper-beach.transects.csv"
    p = naming.parse_zonal_name(n)
    assert p == {"mode": "transects", "base_stem": raster,
                 "zone_set": "upper-beach", "field_tag": None, "id_tag": None}
    csv = "Normandy_Etretat__02_alpha_merged"
    n2 = naming.zonal_out_name(csv, "zones_upper_beach", "Clast width", "polygons", "name")
    assert n2 == csv + "_zones=upper_beach_field=Clast_width_id=name.polygons.csv"
    p2 = naming.parse_zonal_name(n2)
    assert p2 == {"mode": "polygons", "base_stem": csv,
                  "zone_set": "upper_beach", "field_tag": "Clast_width",
                  "id_tag": "name"}
    # How the zones are labelled is part of the name: labelling them by the
    # feature index used to overwrite the run labelled by the layer's names.
    n3 = naming.zonal_out_name(csv, "zones_upper_beach", "Clast width", "polygons")
    assert n3.endswith("_id=index.polygons.csv") and n3 != n2
    assert naming.parse_zonal_name(n3)["id_tag"] == "index"
    # Legacy grammar still parses.
    legacy = "img_merged_Clast_length_D50_cellsize=1.0m__zones_img__Clast_length.transects.csv"
    assert naming.parse_zonal_name(legacy) == {
        "mode": "transects", "base_stem": "img_merged_Clast_length_D50_cellsize=1.0m",
        "zone_set": "zones_img", "field_tag": "Clast_length", "id_tag": None}
    assert naming.parse_zonal_name("img_merged__zones_img.polygons.csv") == {
        "mode": "polygons", "base_stem": "img_merged", "zone_set": "zones_img",
        "field_tag": None, "id_tag": None}
    assert naming.parse_zonal_name("not_a_zonal_file.csv") is None
    assert naming.zone_token("") == "all"
    assert len(n) < 100


def test_a_raster_name_carries_what_changes_the_raster():
    """Two rasterize runs that differ only by their size bins, or only by
    their minimum cell density, wrote the same file and the second silently
    replaced the first."""
    stem = "Site__ortho_merged"
    plain = naming.raster_name(stem, "Clast_length", "D50", 1.0)
    assert plain == stem + "_Clast_length_D50_cellsize=1.0m.tif"
    binned = naming.raster_name(stem, "(any)", "packing_index", 1.0,
                                bin_edges=[16, 64, 256])
    assert binned == stem + "_(any)_packing_index_cellsize=1.0m_bins=16-64-256.tif"
    assert binned != naming.raster_name(stem, "(any)", "packing_index", 1.0)
    phi = naming.raster_name(stem, "(any)", "packing_index", 1.0,
                             bin_edges=[-4, -6], bin_mode="phi")
    assert phi.endswith("_bins=phi-4--6.tif") and phi != binned
    dense = naming.raster_name(stem, "Clast_length", "D50", 1.0, min_density=30)
    assert dense == plain[:-4] + "_mind=30.tif" and dense != plain
    # 0 is "keep every cell", the default: no token, no churn in old names.
    assert naming.raster_name(stem, "Clast_length", "D50", 1.0, min_density=0) == plain
    # The cellsize token still reads, whatever follows it.
    import re
    for n in (plain, binned, phi, dense):
        assert re.search(r"_cellsize=([0-9.]+)m", n).group(1) == "1.0"
        assert naming.run_stem(n) == "Site__ortho"   # the origin, without _merged
    # A density raster has no field token.
    assert naming.raster_name("s", "density", "", 2.5) == "s_density_cellsize=2.5m.tif"


def test_terrestrial_csv_name_is_an_alias_of_quadrat_csv_name():
    # Kept for one release so older scripts keep working; same file name on disk.
    assert naming.terrestrial_csv_name is naming.quadrat_csv_name
    assert naming.quadrat_csv_name("Site_AB") == "Site_AB_individual_clasts.csv"
    assert naming.terrestrial_csv_name("Site_AB") == "Site_AB_individual_clasts.csv"
    assert "quadrat_csv_name" in naming.__all__ and "terrestrial_csv_name" in naming.__all__


def test_a_label_set_is_named_after_its_photograph_and_model():
    from functions import naming as N
    assert N.label_set_csv_name("IMG_0955_rectified_GSD=0.000567m", "seg") == \
        "IMG_0955_rectified_GSD=0.000567m_labels=seg.csv"
    assert N.parse_label_set_name("IMG_0955_rectified_GSD=0.000567m_labels=seg.csv") == \
        ("IMG_0955_rectified_GSD=0.000567m", "seg")
    assert N.label_set_id("Mask R-CNN (v1)") == "Mask-R-CNN-v1-"
    assert N.parse_label_set_name("IMG_0955_truth.csv") is None


def test_a_second_model_of_one_photograph_writes_a_second_file():
    from functions import naming as N
    origin = "P__IMG_0955_rectified_GSD=0.000567m"
    assert N.with_model(origin, "maskrcnn") == origin, "Mask R-CNN keeps the plain name"
    seg = N.with_model(origin, "seg")
    assert seg == origin + "_model=seg"
    assert N.with_model(seg, "imagegrains") == origin + "_model=imagegrains", "replaced, not stacked"
    assert N.with_model(seg, "maskrcnn") == origin
    q = N.quadrat_csv_name(seg)
    o = N.detection_csv_name(seg, 1.0)
    # Every model's output still names its photograph ...
    assert N.image_stem(q) == N.image_stem(o) == "IMG_0955_rectified_GSD=0.000567m"
    # ... says which model made it ...
    assert N.model_of(q) == N.model_of(o) == "seg"
    assert N.model_of(N.quadrat_csv_name(origin)) == "maskrcnn"
    # ... and keeps the token in what downstream outputs are named after.
    assert N.run_stem(o) == seg
    assert N.parse_window_size(o) == 1.0
