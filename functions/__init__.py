"""PebbleMapper core function library.

Modules
-------
clasts_detection
    Mask R-CNN detection of clasts in UAV ortho-images and quadrat
    photographs.
clasts_rasterize
    Convert per-clast CSVs into raster grain-size statistic maps.
clasts_merge
    Merge results from multiple ``metric_cropsize`` runs of the same
    UAV ortho into a single canonical CSV.
map_export
    Publication-quality map figures (Map-tab backend).
orthorectify
    Homography-based image rectification from a known-dimension quadrat.
validation
    Pairwise comparison of a detection CSV against a ground-truth CSV.
digitize
    Manual clast outlining from rasterised user clicks.
layout
    Canonical per-project directory registry used by every other module
    when resolving where data lives on disk.
zonal_stats
    Per-polygon distribution summaries and per-transect longitudinal
    profiles over the rasterised statistic surfaces.
quadrat_validation
    Quadrat-aware validation engine (three quadrat types — geotiff-
    aligned, centroid + dimensions, point + corner buffer) with GSD
    auto-detection from file metadata or filename token.
truncation
    Detection-limit-aware distribution comparison (D_min estimation,
    CCDF, logistic-fitted detection function).
addons
    Sandboxed user-defined per-clast equations loaded from JSON.
report
    PDF report builder consuming every output above.
"""

# Bump whenever the report layout or numerical pipeline changes deliverable
# output; the report cover footer carries it.
__version__ = "1.0.0"


# --- GDAL error model ---
# This package relies on ``gdal.Open(bad_path)`` returning None rather than
# raising; pin DontUseExceptions() once at package import so a future GDAL
# default flip cannot change that. Guarded so a GDAL-less environment works.
try:  # pragma: no cover - depends on the runtime GDAL build
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", FutureWarning)
        from osgeo import gdal as _gdal, ogr as _ogr, osr as _osr
    for _mod in (_gdal, _ogr, _osr):
        try:
            _mod.DontUseExceptions()
        except Exception:
            pass
    del _gdal, _ogr, _osr, _mod, _warnings
except Exception:
    pass
