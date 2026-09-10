example_02_quadrat_detection -- a PebbleMapper example project
=============================================================

What      Two quadrat photographs already rectified to 840 x 840 px at 1 mm/px
          (0.84 x 0.84 m frame), to try Detect in Quadrat mode and Digitize
          without rectifying first. The frame size and the GSD are in the file
          names, where Detect reads them.

images/example_02a_quadrat_height=0.84m_width=0.84m_GSD=0.001m_per_px.jpg
images/example_02b_quadrat_height=0.84m_width=0.84m_GSD=0.001m_per_px.jpg
    Detect: Quadrat mode, image directory images/, Add to queue, Run all
    queued. One clast table per photograph is written to
    output_results/vectors/.
images/example_02a_..._corners.txt
    The frame corners of example_02a in its own pixels, as Orthorectify
    writes them.

Licence: Creative Commons Attribution 4.0 International (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/ .
