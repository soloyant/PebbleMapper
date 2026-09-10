"""EXIF GPS seeding for quadrat placement, and the rectification record.

A quadrat photograph's EXIF fix is typically good to a couple of metres,
which is a better seed than a handheld field GPS. Rectification drops the
EXIF block, so this module reads the fix from the raw photograph and writes
it forward onto the rectified output. It never overrides a user-supplied
seed (pin, seed-list row, typed coordinate) and never writes to a raw
photograph.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

__all__ = [
    "gps_from_image",
    "raw_sibling",
    "rectification_record",
    "write_placement_metadata",
    "looks_like_raw_photograph",
    "same_file",
    "HPE_LIMIT_M",
    "EXIF_SEED_RADIUS_M",
]

# A reported horizontal error worse than this means the fix is not used.
# Absolute, not tied to the search radius: the EXIF tag overstates the true
# error about sixfold, so it is a coarse filter, not a scale.
HPE_LIMIT_M = 40.0

# Search radius when the seed came from a photograph rather than a person.
EXIF_SEED_RADIUS_M = 10.0

_GPS_IFD = 34853

from functions import images as _images  # noqa: F401  (registers the HEIF opener)
from functions.naming import TOOL_SUFFIXES as _TOOL_SUFFIXES


def _dms_to_deg(value, ref: str, negative_ref: str) -> Optional[float]:
    """Degrees/minutes/seconds plus a hemisphere letter, as signed degrees."""
    try:
        d, m, s = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError, IndexError):
        return None
    deg = d + m / 60.0 + s / 3600.0
    if not math.isfinite(deg):
        return None
    return -deg if str(ref or "").strip().upper() == negative_ref else deg


def gps_from_image(path) -> Optional[dict]:
    """The position a photograph carries, or ``None``.

    Returns ``{"lon", "lat", "alt", "hpe_m", "source"}`` in EPSG:4326,
    longitude first; ``alt`` and ``hpe_m`` are ``None`` when absent. Never
    raises: unreadable, no EXIF and no GPS all come back as ``None``.
    """
    p = Path(str(path or ""))
    if not p.is_file():
        return None
    try:
        from PIL import Image, ExifTags
        with Image.open(str(p)) as im:
            exif = im.getexif()
            if not exif:
                return None
            gps_raw = exif.get_ifd(_GPS_IFD) if _GPS_IFD in exif else None
            if not gps_raw:
                return None
            gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
    except Exception:
        return None

    lat = _dms_to_deg(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef", "N"), "S")
    lon = _dms_to_deg(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef", "E"), "W")
    if lat is None or lon is None:
        return None
    # Null Island is a missing fix, not a position in the Gulf of Guinea.
    if abs(lat) < 1e-9 and abs(lon) < 1e-9:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None

    alt = None
    try:
        if gps.get("GPSAltitude") is not None:
            alt = float(gps["GPSAltitude"])
            if str(gps.get("GPSAltitudeRef", 0)) in ("1", "b'\\x01'"):
                alt = -alt
    except (TypeError, ValueError):
        alt = None
    hpe = None
    try:
        if gps.get("GPSHPositioningError") is not None:
            hpe = float(gps["GPSHPositioningError"])
    except (TypeError, ValueError):
        hpe = None

    return {"lon": float(lon), "lat": float(lat), "alt": alt,
            "hpe_m": hpe, "source": str(p)}


# Conventional names for the folder holding the originals. A rectified image
# sits in validation/orthorectified/ and its source in validation/raw/, a sibling
# of the folder the user selects.
_RAW_DIR_NAMES = ("raw", "raws", "original", "originals", "source", "sources",
                  "input_data", "photos", "photographs")


def _sibling_raw_dirs(base: Path) -> list:
    """Directories beside ``base`` that conventionally hold the originals."""
    out = []
    try:
        parent = base.parent
        if not parent or not parent.is_dir():
            return out
        for child in sorted(parent.iterdir()):
            if child.is_dir() and child.name.lower() in _RAW_DIR_NAMES \
                    and child != base:
                out.append(child)
    except OSError:
        pass
    return out


def raw_sibling(path, search_dirs=()) -> Optional[Path]:
    """The raw photograph a rectified image came from.

    Matches by stripping this tool's own suffixes and comparing stems (a raw
    ``01.png`` becomes ``01_rectified.png`` or ``01_corrected_width=1m_...jpg``).
    Looks in ``search_dirs``, then the file's own parent, then any sibling
    directory named for originals.
    """
    p = Path(str(path or ""))
    stem = p.stem
    for suf in _TOOL_SUFFIXES:
        i = stem.find(suf)
        if i > 0:
            stem = stem[:i]
            break
    dirs = [Path(d) for d in search_dirs if d] + [p.parent]
    for base in list(dirs):
        for extra in _sibling_raw_dirs(base):
            if extra not in dirs:
                dirs.append(extra)
    for d in dirs:
        if not d.is_dir():
            continue
        for cand in sorted(d.iterdir()):
            if cand.is_file() and cand != p and cand.stem == stem:
                return cand
    return None


def rectification_record(**kw) -> dict:
    """Metadata record of a rectification: that it happened, when, from what,
    which corners, and the resulting scale and size."""
    import datetime as _dt
    rec = {"PM_RECTIFIED": "yes",
           "PM_RECTIFIED_ON": _dt.datetime.now().isoformat(timespec="seconds")}
    for key, val in kw.items():
        if val is None:
            continue
        rec[f"PM_{key.upper()}"] = (
            # "|" between corners: ";" is the field separator in ImageDescription.
            "|".join(f"{a:.4f},{b:.4f}" for a, b in val)
            if key == "corners" else str(val))
    return rec


def write_placement_metadata(path, fields: dict) -> bool:
    """Attach EXIF metadata to an image the tool itself produced.

    Not GDAL: for PNG and JPEG it writes a detachable ``.aux.xml`` sidecar
    instead of embedding. The pixels are never re-encoded: the block is
    spliced into the PNG or JPEG byte stream (a Pillow re-save truncates an
    LZW ``.tif`` on the GPS IFD and re-compresses a ``.jpg``). Other formats
    fall back to a Pillow re-save that replaces the original only if the
    pixels read back identical.

    No geotransform is ever written: recording the GSD as one would make
    ``georef.describe_georeferencing`` report the output as already placed.
    """
    p = Path(str(path or ""))
    if not p.is_file():
        return False
    try:
        exif_bytes = _exif_block(fields)
        raw = p.read_bytes()
        if raw[:8] == b"\x89PNG\r\n\x1a\n":
            new = _png_with_exif(raw, exif_bytes)
        elif raw[:2] == b"\xff\xd8":
            new = _jpeg_with_exif(raw, exif_bytes)
        else:
            new = None
        if new is not None:
            _replace_atomically(p, new)
        elif not _pillow_resave_if_lossless(p, exif_bytes):
            return False
        # A format with no EXIF container (BMP, PPM) accepts the save and
        # drops the block; do not report success for it.
        if not read_rectification_record(p):
            return False
    except Exception:
        return False
    finally:
        # A stale PAM sidecar would describe the previous quadrat whatever
        # happened above, since the caller has already written the image.
        _drop_sidecar(p)
    return True


def _drop_sidecar(p: Path) -> None:
    aux = p.with_name(p.name + ".aux.xml")
    try:
        if aux.is_file():
            aux.unlink()
    except OSError:
        pass


def _exif_block(fields: dict) -> bytes:
    from PIL import Image
    exif = Image.Exif()
    # ImageDescription carries the record as a key=value list; every reader shows it.
    exif[270] = "; ".join(f"{k}={v}" for k, v in fields.items())
    if "lon" in fields and "lat" in fields:
        gps = _gps_ifd(float(fields["lat"]), float(fields["lon"]))
        if gps:
            exif[_GPS_IFD] = gps
    return exif.tobytes()


def _replace_atomically(p: Path, data: bytes) -> None:
    """Write beside the target, then rename over it, so a failure never
    leaves a part-written image."""
    import os
    tmp = p.with_name(p.name + ".pm-tmp")
    tmp.write_bytes(data)
    os.replace(str(tmp), str(p))


def _png_with_exif(raw: bytes, exif_bytes: bytes) -> bytes:
    """A PNG with an ``eXIf`` chunk, pixels untouched. The chunk holds a bare
    TIFF block with no ``Exif\\0\\0`` prefix (PNG 1.5 extensions, 4.3)."""
    import struct, zlib
    body = exif_bytes[6:] if exif_bytes[:6] == b"Exif\x00\x00" else exif_bytes
    chunk = (struct.pack(">I", len(body)) + b"eXIf" + body
             + struct.pack(">I", zlib.crc32(b"eXIf" + body) & 0xFFFFFFFF))
    out, i = bytearray(raw[:8]), 8
    while i + 8 <= len(raw):
        ln = struct.unpack(">I", raw[i:i + 4])[0]
        typ = raw[i + 4:i + 8]
        end = i + 12 + ln
        if typ == b"eXIf":                      # replace, never duplicate
            i = end
            continue
        if typ in (b"IDAT", b"IEND"):           # must precede the image data
            out += chunk + raw[i:]
            return bytes(out)
        out += raw[i:end]
        i = end
    return bytes(out + chunk)


def _jpeg_with_exif(raw: bytes, exif_bytes: bytes) -> bytes:
    """A JPEG with an APP1/Exif segment, pixels untouched."""
    import struct
    body = (exif_bytes if exif_bytes[:6] == b"Exif\x00\x00"
            else b"Exif\x00\x00" + exif_bytes)
    seg = b"\xff\xe1" + struct.pack(">H", len(body) + 2) + body
    i = 2                                        # past SOI
    out = bytearray(raw[:2])
    while i + 4 <= len(raw):
        if raw[i] != 0xFF:
            break
        marker = raw[i + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            break
        ln = struct.unpack(">H", raw[i + 2:i + 4])[0]
        if marker == 0xE1 and raw[i + 4:i + 10] == b"Exif\x00\x00":
            i += 2 + ln                          # replace, never duplicate
            continue
        if marker in (0xE0, 0xE1):               # after JFIF, before the rest
            out += raw[i:i + 2 + ln]
            i += 2 + ln
            continue
        break
    return bytes(out + seg + raw[i:])


def _pillow_resave_if_lossless(p: Path, exif_bytes: bytes) -> bool:
    """Last resort for formats with no container splice (TIFF, BMP, WebP):
    re-save beside the original and keep it only if the pixels read back
    identical; otherwise leave the original untouched and report failure."""
    import numpy as np
    from PIL import Image
    tmp = p.with_name(p.name + ".pm-tmp" + p.suffix)
    try:
        with Image.open(str(p)) as im:
            im.load()
            before = np.asarray(im)
            im.save(str(tmp), exif=exif_bytes)
        with Image.open(str(tmp)) as chk:
            chk.load()
            if np.asarray(chk).shape != before.shape or \
                    not np.array_equal(np.asarray(chk), before):
                return False
        _replace_atomically(p, tmp.read_bytes())
        return True
    except Exception:
        return False
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def same_file(a, b) -> bool:
    """Whether two paths name the same file on disk, by device and inode:
    ``Path.resolve()`` keeps a ``\\\\?\\`` or UNC prefix verbatim, so a string
    comparison can say two names differ while the OS opens one file."""
    import os
    try:
        sa, sb = os.stat(str(a)), os.stat(str(b))
    except OSError:
        try:
            return Path(str(a)).resolve() == Path(str(b)).resolve()
        except OSError:
            return False
    return (sa.st_dev, sa.st_ino) == (sb.st_dev, sb.st_ino)


def looks_like_raw_photograph(path) -> bool:
    """Whether a file looks like field data (a GPS fix and no rectification
    record) rather than something this tool made. Used to refuse overwriting
    a raw photograph."""
    p = Path(str(path or ""))
    if not p.is_file():
        return False
    if read_rectification_record(p).get("PM_RECTIFIED") == "yes":
        return False
    return gps_from_image(p) is not None


def _gps_ifd(lat: float, lon: float) -> dict:
    from PIL.TiffImagePlugin import IFDRational

    def dms(v):
        v = abs(float(v))
        d = int(v)
        m = int((v - d) * 60)
        s = (v - d - m / 60.0) * 3600.0
        return (IFDRational(d, 1), IFDRational(m, 1),
                IFDRational(int(round(s * 10000)), 10000))
    return {1: "N" if lat >= 0 else "S", 2: dms(lat),
            3: "E" if lon >= 0 else "W", 4: dms(lon)}


def read_rectification_record(path) -> dict:
    """Whatever :func:`write_placement_metadata` put on an image."""
    p = Path(str(path or ""))
    if not p.is_file():
        return {}
    try:
        from PIL import Image
        with Image.open(str(p)) as im:
            exif = im.getexif()
            raw = exif.get(270) if exif else None
    except Exception:
        return {}
    if not raw:
        return {}
    out = {}
    for part in str(raw).split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def seed_from_photo(path, search_dirs=(), bounds=None,
                    crs_transform=None) -> Tuple[Optional[dict], str]:
    """A usable seed from a photograph, or a reason it is not usable.

    ``bounds`` is the ortho's ``(xmin, ymin, xmax, ymax)`` in its own CRS and
    ``crs_transform`` a callable taking (lon, lat) to that CRS. When both are
    given the fix is checked against the ortho's footprint.
    """
    fix = gps_from_image(path)
    used = str(path)
    if fix is None:
        sib = raw_sibling(path, search_dirs)
        if sib is not None:
            fix = gps_from_image(sib)
            used = str(sib)
    if fix is None:
        return None, ("No position in this photograph"
                      + ("" if used == str(path)
                         else f" or in {Path(used).name}") + ".")

    # Bounds before HPE: a fix outside the ortho cannot be right whatever the
    # camera claims, and "check the photograph belongs to this survey" is the
    # more actionable message when both gates trip.
    if bounds is not None and crs_transform is not None:
        try:
            x, y = crs_transform(fix["lon"], fix["lat"])
        except Exception:
            x = None
        if x is not None:
            x0, y0, x1, y1 = bounds
            dx = max(min(x0, x1) - x, 0.0, x - max(x0, x1))
            dy = max(min(y0, y1) - y, 0.0, y - max(y0, y1))
            away = math.hypot(dx, dy)
            if away > EXIF_SEED_RADIUS_M:
                return None, (
                    f"The photograph's position falls {away:.0f} m outside "
                    "this ortho-image, so it cannot be where the quadrat was. "
                    "Check the photograph belongs to this survey.")
    if fix["hpe_m"] is not None and fix["hpe_m"] > HPE_LIMIT_M:
        return None, (f"The photograph reports its own position as accurate to "
                      f"only {fix['hpe_m']:.0f} m, past the {HPE_LIMIT_M:.0f} m "
                      "this will use. Give a coordinate or a pin instead.")
    return fix, ""
