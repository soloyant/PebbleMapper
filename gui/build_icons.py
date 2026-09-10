"""Rebuild gui/icon.ico and gui/icon.icns from gui/icon.svg.

Usage from the repo root:

    pip install cairosvg pillow      # one-time
    python gui/build_icons.py        # regenerates gui/icon.ico and gui/icon.icns

The SVG is what the app uses at runtime (favicon, in-page logo); the .ico and
.icns are only needed for desktop shortcuts (Windows) and app wrappers (macOS).
"""
from __future__ import annotations

import argparse
import io
import os
import struct
import sys
from pathlib import Path


# Microsoft's recommended ICO size set.
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]

# PNG-based ICNS type codes (Big Sur+): (4-byte type code, square pixel size).
ICNS_ENTRIES = [
    (b"ic07", 128),
    (b"ic08", 256),
    (b"ic09", 512),
    (b"ic10", 1024),   # also serves as 512@2x
]


def _check_deps():
    """Print a clear error and exit if cairosvg or PIL is missing."""
    try:
        import cairosvg   # noqa: F401
    except ImportError:
        sys.exit(
            "ERROR: cairosvg is not installed. Run:\n"
            "    pip install cairosvg pillow"
        )
    try:
        from PIL import Image   # noqa: F401
    except ImportError:
        sys.exit("ERROR: Pillow (PIL) is not installed. Run: pip install pillow")


def _render_png(svg_path: str, size: int) -> bytes:
    """Rasterize an SVG to a PNG byte string at the given square size."""
    import cairosvg
    return cairosvg.svg2png(
        url=svg_path, output_width=size, output_height=size,
    )


def build_ico(svg_path: str, out_path: str) -> None:
    """Build a multi-resolution Windows ICO from an SVG."""
    from PIL import Image

    print(f"Building {out_path}")
    images = []
    for size in ICO_SIZES:
        png = _render_png(svg_path, size)
        images.append(Image.open(io.BytesIO(png)).convert("RGBA"))
    # PIL embeds every size in one ICO when sizes= is given.
    images[-1].save(
        out_path,
        format="ICO",
        sizes=[(s, s) for s in ICO_SIZES],
    )
    n_bytes = os.path.getsize(out_path)
    print(f"  Wrote {n_bytes} bytes, {len(ICO_SIZES)} resolutions: "
          f"{ICO_SIZES}")


def build_icns(svg_path: str, out_path: str) -> None:
    """Build a macOS ICNS from an SVG.

    The ICNS format is a TLV container:
      Header: b'icns' + 4-byte big-endian total file size
      Each entry: 4-byte type code + 4-byte big-endian total entry size + payload
    For modern macOS, PNG payloads at the ic07-ic10 type codes cover all
    Finder / Dock / @2x rendering needs.
    """
    print(f"Building {out_path}")
    body = b""
    for type_code, size in ICNS_ENTRIES:
        png = _render_png(svg_path, size)
        entry_size = 8 + len(png)   # header (8) + payload
        body += type_code + struct.pack(">I", entry_size) + png
        print(f"  {type_code.decode()} ({size}x{size}): {len(png):,} bytes")
    total = 8 + len(body)
    header = b"icns" + struct.pack(">I", total)
    with open(out_path, "wb") as f:
        f.write(header + body)
    print(f"  Wrote {total:,} bytes total, {len(ICNS_ENTRIES)} resolutions")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate icon.ico and icon.icns from icon.svg.")
    parser.add_argument(
        "--svg", default="gui/icon.svg",
        help="Path to the source SVG (default: gui/icon.svg)")
    parser.add_argument(
        "--ico", default="gui/icon.ico",
        help="Output ICO path (default: gui/icon.ico)")
    parser.add_argument(
        "--icns", default="gui/icon.icns",
        help="Output ICNS path (default: gui/icon.icns)")
    parser.add_argument(
        "--no-ico", action="store_true",
        help="Skip ICO generation")
    parser.add_argument(
        "--no-icns", action="store_true",
        help="Skip ICNS generation")
    args = parser.parse_args()

    _check_deps()

    svg = Path(args.svg)
    if not svg.exists():
        sys.exit(f"ERROR: SVG not found at {svg}")

    if not args.no_ico:
        Path(args.ico).parent.mkdir(parents=True, exist_ok=True)
        build_ico(str(svg), args.ico)
    if not args.no_icns:
        Path(args.icns).parent.mkdir(parents=True, exist_ok=True)
        build_icns(str(svg), args.icns)

    print("\nDone. See DESKTOP_LAUNCHER.md for how to use these on Windows / macOS.")


if __name__ == "__main__":
    main()
