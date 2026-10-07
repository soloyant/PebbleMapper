example_04_gauge_scale -- a PebbleMapper example project
========================================================

What      One phone photograph of a gravel bar with a boot in the frame, for the
          scale-from-an-object option of the Digitize tab: no ground sample
          distance, no quadrat, no georeference. Sizes come out in the object's
          own unit ("boot width") unless the object's length is known.
Layout    the canonical PebbleMapper project layout (see docs/user-manual.md 2.2),
          reduced to what this example needs.

images/IMG_4228.heic
    The photograph as shot (iPhone HEIC, 2142 x 2856 px). Digitize: pick it in
    the Image directory, open "No GSD? Scale from an object", draw a segment
    across the boot, and every clast is then measured in boot widths. Detect's
    Object scale mode reads the same segment file.
scaling_objects.json
    The project's library of scaling objects: one entry, "boot width", with its
    length unknown. Give it a length in millimetres and the same photograph is
    measured in millimetres instead.

Photograph: Dr Mark Lorang. Please credit him for any reuse of this image; it
is his work, not part of the MIT licence that covers the software.

Photograph metadata: the GPS fix, the device and lens names and the capture
date have been removed. The photograph was lent for this example and where it
was taken is not part of it; the scale comes from the object in the frame,
never from a location.
