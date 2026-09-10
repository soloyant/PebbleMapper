example_06_paper_validation -- a PebbleMapper example project
=============================================================

What      The validation set of Soloy et al. (2020), section 2.3: 105 pebbles
          measured with a caliper, each with its number written on it, laid
          out in a 0.84 m quadrat frame on tarmac and photographed from above.
          Eleven photographs: the 105 spread out (IMG_0806), then the 27
          largest in ten arrangements, from spread out to heaped. Each
          photograph comes rectified and paired with a Validate truth file, so
          the paper's check can be run again with any detection model.
Where     Pebbles sampled on the beach of Hautot-sur-Mer (Pourville), Normandy,
          France. Photographed 28 May 2020 (IMG_0784 to IMG_0796) and 1 June
          2020 (IMG_0806), Apple iPhone 11, 4032 x 3024 px.
Frame     0.84 x 0.84 m, bars 2 cm wide.
Layout    the canonical PebbleMapper project layout (see docs/user-manual.md 2.2).

The photographs
    IMG_0806         the 105 pebbles, spread out           0.502 mm/px
    IMG_0784 - 0787  the 27 largest, spread out            0.361 - 0.398 mm/px
                     (IMG_0784: a shadow across half the frame)
    IMG_0788         the 27 largest, close together        0.338 mm/px
    IMG_0790         the 27 largest, packed, touching      0.356 mm/px
    IMG_0791, 0792   the 27 largest, heaped                0.348, 0.325 mm/px
    IMG_0795, 0796   the 27 largest, heaped higher         0.285, 0.272 mm/px

validation/results/pebble_caliper_measurements.csv
    The ground truth: one row per pebble, the number written on it, the three
    caliper axes (major, minor and z, in metres) and, for the 27 largest, the
    mass (g) and water-displacement volume (mL).
validation/raw/IMG_0806.JPG
    The one photograph shipped as shot (re-encoded at JPEG quality 88; the GPS
    fix, the maker-note blob and the device identifiers removed). The other ten
    are shipped rectified only, to keep the example small.
validation/raw/IMG_nnnn_corners.txt, IMG_0806_segments.txt
    The frame corners on each raw photograph (found by registering the 2020
    rectification onto it), and the frame's side lengths (0.84 m) and bar
    width (2 cm), as Orthorectify reads them.
validation/orthorectified/IMG_nnnn_rectified_GSD=...m.jpg (+ .json)
    Each photograph rectified by PebbleMapper's Orthorectify at its automatic
    GSD, which keeps the frame's native pixel count (0.27 to 0.50 mm/px),
    re-encoded at JPEG quality 90. The sidecar records the GSD, the corners and
    the 2 cm frame band, which Detect leaves out automatically.
validation/IMG_nnnn_rectified_GSD=...m_truth.csv
    The caliper measurements as a Validate truth file, one per photograph:
    clast_ID is the pebble's number, x/y its position on the rectified
    photograph (pixels, y counted from the bottom as in every PebbleMapper
    CSV), Clast_length and Clast_width the caliper major and minor axes, plus
    z_axis, mass_g and volume_mL. The positions are the centres of the 2020
    detections, joined to their caliper rows by the 2020 analysis and checked
    against all three axes, carried onto these rectifications by image
    registration; the pebbles the 2020 model missed were placed by hand.
    On the heaped photographs only the pebbles fully visible from above are in
    the truth: IMG_0791 and IMG_0792 hold 13 of the 27 (left out: 2 3 6 7 8 9
    12 13 14 17 18 19 20 23), IMG_0795 holds 10, IMG_0796 12. A detection of a
    partly hidden pebble therefore counts as unpaired, and precision on those
    four photographs understates the model.
output_results/vectors/example_06_paper_validation__IMG_nnnn_..._individual_clasts.csv
    The app's Mask R-CNN detections, frame band left out, with their outlines
    (.contours.json). Validate: pick a truth file and the matching CSV,
    "Centroid + dimensions" footprint, and tick "Reject size mismatches > 50 %".
    Any other model run on these photographs writes a second file beside
    these, named with its _model=<id> token.

Measured on these files (Validate, Mask R-CNN)
    IMG_0806, the paper's photograph: 88 of the 105 pebbles paired, none of the
    detections unpaired (recall 0.84, precision 1.00; the paper reports 80 to
    90 % detected). Clast_length against the caliper major axis: R2 0.98, RMSE
    4.9 mm, bias -3.6 mm (the paper: R2 0.98, RMSE 3.9 mm); Clast_width against
    the minor axis: R2 0.96, RMSE 4.9 mm, bias -3.3 mm. The pebbles missed are
    the small ones: none of the 5 under 20 mm, 8 of the 13 of 20-30 mm, 35 of
    the 40 of 30-50 mm and 45 of the 47 larger ones are found.

    photograph   paired      precision   length R2   RMSE mm   bias mm
    IMG_0806     88 of 105   1.00        0.98         4.9      -3.6
    IMG_0784     27 of 27    0.96        0.88         9.4      +4.3
    IMG_0785     27 of 27    1.00        0.96         5.5      +2.9
    IMG_0786     27 of 27    1.00        0.91        10.4      +7.3
    IMG_0787     27 of 27    1.00        0.99         2.6      -0.4
    IMG_0788     24 of 27    1.00        0.98         4.4      -2.5
    IMG_0790     12 of 27    1.00        0.97         6.1      -4.5
    IMG_0791      9 of 13    1.00        0.92         7.1      -3.2
    IMG_0792      9 of 13    1.00        0.92         6.8      -2.9
    IMG_0795      6 of 10    0.86        0.81        15.9      -1.5
    IMG_0796     11 of 12    1.00        0.81        14.4      +7.2

    Spread out, the model finds nearly every pebble; once the pebbles touch it
    finds fewer (12 of 27 packed), but what it finds it still measures well.
    On the close-up photographs (0.27 to 0.40 mm/px) the length runs long, up
    to +7 mm, and on IMG_0806, the farthest shot, short (-3.6 mm): a pebble's
    top stands above the frame plane the rectification measures on, and the
    nearer the camera the more its outline is magnified. That reading fits the
    pattern but has not been tested.

Licence: Creative Commons Attribution 4.0 International (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/ . Please cite Soloy et al. (2020)
and the PebbleMapper software (CITATION.cff at the root of the repository), and
credit the data as "Soloy et al. (2020) caliper validation set, 2020-05-28 and
2020-06-01, A. Soloy, CC BY 4.0".
