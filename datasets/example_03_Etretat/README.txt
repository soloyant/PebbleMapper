example_03_Etretat -- a PebbleMapper example project
=====================================================

Site      Etretat, Normandy, France: a flint-shingle beach.
Survey    10 June 2020. UAV orthomosaic and DEM; quadrat photographs (iPhone 11)
          in a 0.84 m frame with 2 cm bars; RTK GNSS positions (FIX, ~5 mm).
CRS       EPSG:2154 (RGF93 / Lambert-93), metres.
Layout    the canonical PebbleMapper project layout (see docs/user-manual.md 2.2).

This is a small subset of the survey, chosen so that every tab of the app has
something to work on: the four quadrats closest together, and the 20.9 x 20.7 m
window of the ortho that holds them.

input_data/images/etretat_20200610_ortho_crop.tif
    Orthomosaic crop, 6110 x 6058 px at 3.41 mm/px, RGB + mask band (JPEG-in-TIFF).
    Detect (Ortho mode) -> Merge -> Rasterize -> Map -> Zonal -> Report.
input_data/dem/etretat_20200610_dem_crop.tif
    DEM of the same window, 1527 x 1515 px at 13.7 mm/px (Rasterize, Zonal).
input_data/geometries/zones_quadrats.geojson
    The four quadrat footprints (0.84 m frames) as placed in the ortho by
    Georeference, seeded by the RTK points (3 to 4 mm residual): position and
    orientation as laid on the beach (Zonal, Validate). Named after the RTK point
    and the photograph: Q25_IMG_0955, Q26_IMG_0957, Q41_IMG_0973, Q42_IMG_0974.
input_data/geometries/transect_1.geojson
    A 23.1 m cross-shore line through the centre of the crop, landward end
    first, along the DEM's steepest descent (bearing 328.5 degrees, towards the
    sea), perpendicular to the sediment bands (Zonal's transect sampling).
validation/raw/IMG_0955.JPG, IMG_0957.JPG
    Two of the quadrat photographs as shot (4032 x 3024, GPS position kept;
    re-encoded at JPEG quality 88). Orthorectify: the *_corners.txt files hold
    the four frame corners picked for them, *_segments.txt the frame's sides
    (0.84 m) and bar thickness (0.02 m).
validation/orthorectified/IMG_09xx_rectified_GSD=0.000xxxm.jpg (+ .json)
    The four photographs rectified by Orthorectify: ~1500 px at 0.55-0.57 mm/px,
    the frame bars along the edges; the sidecar carries the GSD, the corners,
    the frame thickness and the photograph's own GPS fix. Detect (Quadrat mode,
    the frame band is left out automatically), Digitize, Georeference.
validation/georectified/IMG_0955_rectified_GSD=0.000567m.tif (+ .georef.json)
    IMG_0955 placed in the ortho by Georeference (77 inliers, 3.1 mm residual).
    Redo it, or place the other three, with the Georeference tab. Its .tif.json
    sidecar records the frame bars (1.8 cm) for Detect and Digitize.
validation/IMG_0955_rectified_GSD=0.000567m_truth.csv
    (+ .contours.json, .provenance.json)
    Ground truth for IMG_0955: 1,362 clasts digitised in the Digitize tab,
    every outline checked by eye. It fits the rectified .jpg and the
    georeferenced .tif alike: they share one pixel grid. It started from Segment Every
    Grain's proposals (1,334 kept, 66 of them reshaped by hand), with 28 clasts
    drawn by hand; the provenance file says which is which. Left out: large
    pebbles whose edge is buried out of sight, and the frame band. Because it
    grew from one model's proposals, it is not an independent test of that
    model. Validate it against the Mask R-CNN detections of the same
    photograph (output_results/vectors/...IMG_0955..._individual_clasts.csv):
    320 clasts paired, precision 0.99, recall 0.23; on the pairs, Clast_length
    R2 0.98, RMSE 1.6 mm, bias -0.8 mm. The recall rises with size, from 0.12
    for clasts of 10-15 mm to 0.48 above 30 mm, so the detections' D84 reads
    34.2 mm against 26.5 mm in the truth.
validation/gps/etretat_20200610_quadrats_rtk.csv
    The RTK points of the four quadrats, one per row, with the photograph each
    one belongs to. The antenna stood at a corner of the frame (0.55 m from
    the fitted centre of every photograph).

Licence: Creative Commons Attribution 4.0 International (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/ . Please cite the PebbleMapper
software (CITATION.cff at the root of the repository) and credit the data as
"Etretat 2020-06-10 quadrat and UAV survey subset, A. Soloy, CC BY 4.0".

Photograph metadata: only the orientation and the GPS position (latitude,
longitude, altitude and its horizontal error) are kept; the position is what
the quadrat-placement seeding reads. Camera, lens, exposure and capture date
have been removed.
