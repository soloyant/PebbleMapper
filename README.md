<p align="center">
  <img src="gui/icon.svg" alt="PebbleMapper logo" width="96"/>
</p>

<h1 align="center">PebbleMapper</h1>

<p align="center">
  Detection, measurement and mapping of individual rounded clasts in photographs and drone ortho-images.
</p>

<p align="center">
  <a href="https://github.com/soloyant/pebblemapper/actions/workflows/tests.yml"><img src="https://github.com/soloyant/pebblemapper/actions/workflows/tests.yml/badge.svg" alt="tests"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/licence-MIT-blue.svg" alt="MIT licence"/></a>
  <img src="https://img.shields.io/badge/python-3.9-blue.svg" alt="Python 3.9"/>
  <img src="https://img.shields.io/badge/platform-Windows%2010%20%7C%2011-lightgrey.svg" alt="Windows 10 and 11"/>
</p>

PebbleMapper detects individual rounded clasts (pebbles, cobbles) in images of known
scale, measures each one, and turns the measurements into grain-size distributions, maps,
zonal statistics and a PDF report. It works clast by clast, so every statistic can be traced
back to the outlines it was computed from. Detection uses the Mask R-CNN model of
[Soloy et al. (2020)](https://doi.org/10.3390/rs12213659); other detection models can be
added as plug-ins. It is a desktop application with a browser interface and runs entirely
on your computer.

<p align="center">
  <img src="docs/figures/quadrat-before-after.jpg" alt="A quadrat photograph as taken, and rectified with every detected clast outlined and coloured by size class" width="100%"/>
</p>
<p align="center"><em>A quadrat photograph from the Étretat example as taken (left) and after
rectification and detection (right): each clast coloured by its half-phi size class, with its
length (solid) and width (dotted) chords. The inset shows the two measured chords of one clast.</em></p>

## Scope

The examples below come from the projects shipped in `datasets/` and can be reproduced from
them.

**Ortho-images of any size.** A georeferenced ortho-image is processed tile by tile at one or
several window sizes; detections in overlapping tiles are merged, and each clast is written
once, in the image's coordinate system, with its length and width (apparent a- and b-axes),
area, perimeter, orientation, shape indices and detection score. On the Étretat example
(21 × 21 m at 3.4 mm/px) this gives 12,292 clasts.

<p align="center">
  <img src="docs/figures/ortho-detection.jpg" alt="Detected clasts on a 2 m window of a drone ortho-image" width="60%"/>
</p>
<p align="center"><em>A 2 m window of the Étretat ortho-image: each detected clast by size
class, with its length and width chords.</em></p>

**Grain-size maps.** Any per-clast quantity (length, width, area, a custom formula) is
aggregated on a regular grid: percentiles, mean, Folk and Ward sorting, skewness, kurtosis,
density. Rasters are GeoTIFFs; the Map tab composes them into figures with scale bar, grid,
north arrow and basemap.

<p align="center">
  <img src="docs/figures/map-d50.jpg" alt="D50 per 1 m cell over the Étretat ortho-image" width="70%"/>
</p>
<p align="center"><em>D50 of clast length per 1 m cell over the Étretat ortho-image.</em></p>

**Zonal statistics and transects.** Statistics are extracted per polygon (quadrats, habitat
zones, survey units) and along transects, from GeoJSON drawn in the app or imported.

<p align="center">
  <img src="docs/figures/zones-map.jpg" alt="Quadrat footprints and a cross-shore transect over the ortho-image" width="48%"/>
  <img src="docs/figures/transect-profile.png" alt="D50 and elevation along the cross-shore transect" width="48%"/>
</p>
<p align="center"><em>Left: the four quadrats of the Étretat example, placed in the ortho-image
from their photographs, and a cross-shore transect. Right: D50 and elevation along the
transect, from the landward end: the median clast length rises from about 43 to 50 mm
across the break in slope.</em></p>

**Scale from an object.** A photograph taken without a quadrat or a known ground sample
distance is scaled from an object in the picture. With the object's size unknown, clasts are
measured in multiples of it; with its size known, in millimetres.

<p align="center">
  <img src="docs/figures/scale-from-object.jpg" alt="Clasts measured in boot widths on a phone photograph" width="55%"/>
</p>
<p align="center"><em>149 clasts measured in boot widths on a phone photograph (example_04;
photograph by Dr Mark Lorang).</em></p>

**Validation.** Detections are compared with clasts outlined by hand in the Digitize tab or
measured with a caliper: paired clast by clast, then as distributions (Bland–Altman
agreement, cumulative curves, Kolmogorov–Smirnov test, truncation at the detection limit).

<p align="center">
  <img src="docs/figures/validation.png" alt="Detected against caliper length, Bland–Altman and cumulative distributions" width="90%"/>
</p>
<p align="center"><em>The built-in model against the caliper on the 105 pebbles of
Soloy et al. (2020) (example_06): 88 pairs, R² 0.98, bias −3.6 mm.</em></p>

**Reports.** One PDF per project states the inputs, the method, the statistics per image,
the maps, the validation and their limits, with a data dictionary and references.

<p align="center">
  <img src="docs/figures/report-pages.jpg" alt="Four pages of the Étretat project report" width="100%"/>
</p>
<p align="center"><em>Pages of the report generated for the Étretat example.</em></p>

## Installation

PebbleMapper runs on Windows 10 and 11 (64-bit). You need about 10 GB of free disk space
and an internet connection during the installation. An NVIDIA graphics card makes
detection faster but is not required.

1. On this page, click the green **Code** button, then **Download ZIP**.
2. Create a folder with a short path that is not synchronised by OneDrive or Dropbox, for
   example `C:\PebbleMapper`. In your Downloads folder, right-click `pebblemapper-main.zip`,
   choose **Extract All…**, select that folder and click **Extract**. PebbleMapper runs from
   it, so keep it.
3. In the extracted folder, open `pebblemapper-main` until you see the file
   **Install PebbleMapper**, and double-click it. If Windows asks for confirmation, click
   **Run**, or **More info** then **Run anyway**.
4. A black window shows the progress. The installation takes 20 to 60 minutes. It
   installs Python (through Miniforge) if your computer has none, downloads the model (once it is public, see below),
   and ends with *PebbleMapper is installed*. Press a key to close the window.
5. Double-click the **PebbleMapper** icon on your desktop. The application opens in your
   web browser. Keep the black window open while you work; closing it stops PebbleMapper.

> **Model weights.** The trained Mask R-CNN weights are not public yet: their licensing is
> being cleared, which should be settled in the coming weeks. The installation completes
> without them and every tab except Detect works. In the meantime the author sends the
> weights privately on request: [open an issue](https://github.com/soloyant/pebblemapper/issues).
> Place the file you receive at `model_weights/mask_rcnn_clasts.h5` in the PebbleMapper
> folder.

If the installation stops with an error, double-click **Install PebbleMapper** again. It
continues where it stopped. To uninstall, delete the `pebblemapper-main` folder, the
`miniforge3` folder in your user folder (only if the installer created it) and the desktop
icon.

<details>
<summary>Installation from the command line (developers, Linux, macOS)</summary>

```bash
git clone https://github.com/soloyant/pebblemapper
cd pebblemapper
conda env create -f environment.yml    # on a machine without an NVIDIA GPU, remove the cudatoolkit and cudnn lines first
conda activate maskrcnn
python -m detectors.download_weights   # 367 MB, from Zenodo
python tools/check_install.py
python -m gui.app
```

On Windows, `.\install.ps1` does the same (`-Cpu`, `-SkipWeights`, `-Force`, `-CondaRoot <folder>`,
`-NoShortcut`). Linux and macOS are not tested; `gui/launch_gui.sh` starts the application
there.

</details>

## Getting started

The application opens on the **Overview** tab. Choose the folder that holds your projects,
then pick one of the example projects below or create your own. The tabs follow the order
of the work: Orthorectify, Detect, Digitize and Validate for photographs; Detect, Merge,
Rasterize, Map and Zonal for drone ortho-images; Report for both. **Express** runs the
whole drone-ortho chain in one step.

Example projects, in the `datasets` folder:

| Project | Content |
|---|---|
| `example_01_orthorectification` | A handheld quadrat photograph to rectify. |
| `example_02_quadrat_detection` | Rectified quadrat photographs to detect and digitise. |
| `example_03_Etretat` | A survey of the shingle beach at Étretat (Normandy, 10 June 2020): a 21 × 21 m crop of the drone ortho-image and its elevation model, four quadrat photographs, one of them with 1,362 clasts outlined by hand, RTK positions, zones and a transect. Every tab can be tried on it. |
| `example_04_gauge_scale` | A phone photograph with a boot for scale, for sizing clasts without a quadrat frame. |
| `example_06_paper_validation` | The caliper validation set of Soloy et al. (2020): 105 pebbles measured with a caliper, photographed spread out and, for the 27 largest, in ten arrangements up to heaped. A truth file per photograph. |

The [user manual](docs/user-manual.md) describes every tab, and the installation step by step
(section 2.1).

## Accuracy and limits

On IMG_0806 of `example_06_paper_validation` (105 pebbles spread out on sand,
0.50 mm/px), the built-in model finds 88 pebbles and makes no false detection. Detected
length against the caliper a-axis: R² 0.98, RMSE 4.9 mm, bias −3.6 mm. The 17 pebbles it
misses are the smallest: none of the five under 20 mm is found.

On a densely packed beach surface the model finds a smaller share of the clasts. On
IMG_0955 of `example_03_Etretat` (0.57 mm/px, 1,362 surface clasts outlined by hand) it
finds 320 (24 %), mostly the larger and paler ones, with almost no false detections
(99 %). The clasts it finds are measured accurately (length RMSE 1.6 mm), but the
distribution is biased towards coarse sizes: D50 20.2 mm against 17.7 mm by hand, D84
34.2 mm against 26.5 mm. Clasts shorter than 8 pixels are below the detection limit.
Where a complete count matters, compare with a plug-in model (below) or check against
clasts outlined by hand in the Digitize tab.

## Other detection models

Three plug-ins let PebbleMapper run other published models. Each is a separate repository
with its own environment and licence; they are not public yet, as their authors are being
consulted. PebbleMapper measures the outlines every model returns in the same way, so the
results are comparable.

The same photograph (IMG_0955, 1,362 clasts outlined by hand) through each model:

| Model | Detections | Hand-outlined clasts found | Detections that match one | Length RMSE | D50 | D84 | Time per photograph |
|---|---|---|---|---|---|---|---|
| Hand outlines | 1,362 | | | | 17.7 mm | 26.5 mm | |
| Mask R-CNN (built in) | 324 | 320 (24 %) | 99 % | 1.6 mm | 20.2 mm | 34.2 mm | 41 s (GPU) |
| [Segment Every Grain](https://github.com/zsylvester/segmenteverygrain) | 1,822 | 1,350 (99 %) | 74 % | 1.0 mm | 17.7 mm | 26.3 mm | 259 s (GPU) |
| [ImageGrains](https://github.com/dmair1989/imagegrains) | 2,034 | 1,316 (97 %) | 65 % | 1.7 mm | 17.2 mm | 26.0 mm | 51 s (CPU) |
| [PebbleCountsAuto](https://github.com/UP-RS-ESP/PebbleCounts) | 605 | 491 (36 %) | 81 % | 4.1 mm | 21.0 mm | 35.3 mm | 17 s (CPU) |

Detections are paired with hand outlines by position and size, as the Validate tab does.
The hand outlines started from Segment Every Grain's detections, corrected and completed by
hand, which favours that model here. Most unmatched outlines of Segment Every Grain and
ImageGrains are clasts lying partly buried between the surface clasts, which the hand
outlines leave out; their share of matching detections is therefore a lower bound.
Times are for one photograph once the model is loaded; loading adds 20 to 70 s once per run. They were measured on a 2018 laptop (Intel Core i7-8850H, NVIDIA Quadro P600 with 4 GB). ImageGrains' environment installs the processor build of PyTorch, and PebbleCountsAuto has no GPU code.

<p align="center">
  <img src="docs/figures/backends-overlays.jpg" alt="The same quadrat photograph through the hand outlines and the four models" width="100%"/>
</p>
<p align="center"><em>A 40 cm crop of the same photograph: the hand outlines and each model's
detections, coloured by the same size classes.</em></p>

## Documentation

- [User manual](docs/user-manual.md): every tab, the clast table, the project folders and
  custom equations.
- For developers: the [Python API](docs/developer/api.md) and
  [adding a detection model](docs/developer/adding-a-detection-model.md).

Questions, bug reports and suggestions: [open an issue](https://github.com/soloyant/pebblemapper/issues).
The **Feedback** section of the application's side menu builds a diagnostic file to
attach; it contains versions and logs, no images or measurements.

## Citation

Please cite the method paper when publishing results obtained with PebbleMapper:

> Soloy, A.; Turki, I.; Fournier, M.; Costa, S.; Peuziat, B.; Lecoq, N. (2020). *A Deep
> Learning-Based Method for Quantifying and Mapping the Grain Size on Pebble Beaches.*
> Remote Sensing, 12(21), 3659. [doi:10.3390/rs12213659](https://doi.org/10.3390/rs12213659)

and the trained model: Soloy, A.; Turki, I.; Fournier, M.; Costa, S.; Peuziat, B.; Lecoq, N.
(2026). *PebbleMapper: trained Mask R-CNN weights for clast detection.* Zenodo. [doi:10.5281/zenodo.20779877](https://doi.org/10.5281/zenodo.20779877) (publication pending)

`CITATION.cff` gives both in machine-readable form (GitHub's *Cite this repository*).

## Licence

The code is released under the MIT licence ([`LICENSE`](LICENSE)). `mrcnn/` contains the
[Matterport implementation of Mask R-CNN](https://github.com/matterport/Mask_RCNN) (MIT,
[`mrcnn/LICENSE`](mrcnn/LICENSE)). The trained weights and the example data are released
under Creative Commons Attribution 4.0 (CC BY 4.0), except the photograph of
`example_04_gauge_scale`, which remains Dr Mark Lorang's (see its `README.txt`). Maps drawn over a basemap must credit
the tile provider (Esri World Imagery, OpenStreetMap).

## Acknowledgements

The method and the trained model were developed during Antoine Soloy's doctoral research,
funded by the Région Normandie. Dr Mark Lorang provided the photograph of
`example_04_gauge_scale` used to develop and illustrate scaling from an object. PebbleMapper replaces the earlier
[Clast Size Mapping](https://github.com/soloyant/clast-size-mapping) code of Soloy et al.
(2020). It was developed by Antoine Soloy with Claude (Anthropic) as a programming
assistant for much of the code, tests and documentation; the method, the decisions and the
validation are the author's.
