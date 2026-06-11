"""Reading Zeiss CZI files for section detection.

A whole-slide CZI (the target ``tard_carbon_coat_001.czi`` is 75 945 x 78 229 px,
Gray16, 5 712 tiles, 13 GB) must never be decoded at full resolution into RAM.
We read a downscaled level from the stored pyramid via ``pylibCZIrw``'s
``read(..., zoom=...)`` and keep a :class:`CziGeometry` so detections on the
downscaled overview map back to full-resolution pixels (what ZEN annotations
use) and, optionally, to physical stage microns.

A dependency-free metadata parser (:func:`parse_czi_metadata_raw`) is also
provided: it follows the CZI ``FileHeader`` pointer straight to the
``ZISRAWMETADATA`` segment, so scaling / dimensions can be read without
``pylibCZIrw`` installed.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Dependency-free metadata (FileHeader -> MetadataPosition -> XML)
# --------------------------------------------------------------------------- #
def parse_czi_metadata_raw(path: str) -> dict:
    """Extract key CZI metadata without any CZI library.

    Returns a dict with ``size_x``, ``size_y``, ``size_c``, ``size_s``,
    ``size_m``, ``scale_x``, ``scale_y`` (meters/pixel), ``pixel_type`` and the
    raw ``xml`` string. Robust to absent fields (value is ``None``).
    """
    with open(path, "rb") as f:
        magic = f.read(16).split(b"\x00")[0].decode(errors="replace")
        if magic != "ZISRAWFILE":
            raise ValueError(f"Not a CZI file (magic={magic!r})")
        # FileHeaderSegmentData starts at byte 32; MetadataPosition is int64 @ 92.
        f.seek(92)
        metadata_position = struct.unpack("<q", f.read(8))[0]
        f.seek(metadata_position + 32)  # skip 32-byte segment header
        xml_size, _att = struct.unpack("<ii", f.read(8))
        f.seek(metadata_position + 32 + 256)  # skip fixed metadata header
        xml = f.read(xml_size).decode("utf-8", errors="replace")

    def _first(tag):
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.S)
        return m.group(1).strip() if m else None

    def _int(tag):
        v = _first(tag)
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    def _scaling(axis):
        m = re.search(
            rf'<Distance Id="{axis}">(.*?)</Distance>', xml, re.S
        )
        if not m:
            return None
        v = re.search(r"<Value>(.*?)</Value>", m.group(1))
        try:
            return float(v.group(1))
        except (TypeError, ValueError, AttributeError):
            return None

    return {
        "size_x": _int("SizeX"),
        "size_y": _int("SizeY"),
        "size_c": _int("SizeC"),
        "size_s": _int("SizeS"),
        "size_m": _int("SizeM"),
        "scale_x": _scaling("X"),
        "scale_y": _scaling("Y"),
        "pixel_type": _first("PixelType"),
        "xml": xml,
    }


# --------------------------------------------------------------------------- #
# Geometry: downscaled-overview pixels <-> full-res pixels <-> stage microns
# --------------------------------------------------------------------------- #
@dataclass
class CziGeometry:
    """Maps coordinates between the three frames STiM cares about.

    ``zoom`` is the factor the overview was read at (e.g. 0.0524). ``origin_x/y``
    is the full-resolution pixel origin of the read region (the CZI total/scene
    bounding-rectangle top-left, which can be non-zero). ``scale_x/y`` are
    meters/pixel. ``stage_center_um`` / ``center_px_full`` anchor the optional
    stage transform.
    """

    zoom: float
    origin_x: float = 0.0
    origin_y: float = 0.0
    scale_x: float | None = None  # meters per full-res pixel
    scale_y: float | None = None
    stage_center_um: tuple[float, float] | None = None
    center_px_full: tuple[float, float] | None = None
    y_direction: int = 1  # CZI Distance Id="Y" Direction (often -1)

    def ds_to_full(self, x_ds, y_ds):
        """Overview-pixel -> full-resolution-pixel (ZEN annotation frame)."""
        x = self.origin_x + np.asarray(x_ds, dtype=float) / self.zoom
        y = self.origin_y + np.asarray(y_ds, dtype=float) / self.zoom
        return x, y

    def full_to_ds(self, x_full, y_full):
        """Full-resolution-pixel -> overview-pixel (inverse of ds_to_full)."""
        x = (np.asarray(x_full, dtype=float) - self.origin_x) * self.zoom
        y = (np.asarray(y_full, dtype=float) - self.origin_y) * self.zoom
        return x, y

    def full_to_stage_um(self, x_full, y_full):
        """Full-res-pixel -> stage microns. Best-effort; needs-verification.

        Requires ``scale_*`` and a stage anchor; otherwise returns ``None``.
        """
        if self.scale_x is None or self.scale_y is None:
            return None
        sx_um = self.scale_x * 1e6
        sy_um = self.scale_y * 1e6
        if self.stage_center_um and self.center_px_full:
            cx_um, cy_um = self.stage_center_um
            cx_px, cy_px = self.center_px_full
        else:
            cx_um = cy_um = cx_px = cy_px = 0.0
        x_um = cx_um + (np.asarray(x_full, dtype=float) - cx_px) * sx_um
        y_um = cy_um + (np.asarray(y_full, dtype=float) - cy_px) * sy_um * self.y_direction
        return x_um, y_um


# --------------------------------------------------------------------------- #
# Reading + 16-bit -> 8-bit RGB
# --------------------------------------------------------------------------- #
def _pick_zoom(width: int, height: int, target_long_side: int) -> float:
    long_side = max(width, height)
    if long_side <= target_long_side:
        return 1.0
    return float(target_long_side) / float(long_side)


def read_czi_overview(path: str, target_long_side: int = 4096, channel: int = 0,
                      scene: int | None = None):
    """Read a downscaled overview of a CZI as a numpy array + :class:`CziGeometry`.

    Returns ``(array, geometry, meta)`` where ``array`` is the raw read (Gray16
    -> uint16 HxW, or HxWx3) and ``geometry`` lets you map detections back.
    Never decodes the full-resolution image.
    """
    from pylibCZIrw import czi as pyczi  # local import: optional dependency

    raw_meta = parse_czi_metadata_raw(path)

    with pyczi.open_czi(path) as cz:
        x0, y0, w, h = cz.total_bounding_rectangle
        zoom = _pick_zoom(w, h, target_long_side)
        plane = {"C": channel}
        kwargs = dict(roi=(x0, y0, w, h), plane=plane, zoom=zoom)
        if scene is not None:
            kwargs["scene"] = scene
        arr = cz.read(**kwargs)

    arr = np.squeeze(arr)
    geom = CziGeometry(
        zoom=zoom,
        origin_x=float(x0),
        origin_y=float(y0),
        scale_x=raw_meta.get("scale_x"),
        scale_y=raw_meta.get("scale_y"),
        center_px_full=(x0 + w / 2.0, y0 + h / 2.0),
    )
    meta = {**raw_meta, "zoom": zoom, "read_width": w, "read_height": h,
            "origin": (x0, y0)}
    return arr, geom, meta


def read_czi_region(path: str, x_full: int, y_full: int, w_full: int, h_full: int,
                    channel: int = 0, scene: int | None = None,
                    as_rgb8: bool = True, zoom: float = 1.0):
    """Read an arbitrary region of a CZI at full (``zoom=1.0``) or reduced zoom.

    ``x_full, y_full, w_full, h_full`` are FULL-resolution pixel coordinates in
    the CZI total bounding-rectangle frame -- the same frame
    :meth:`CziGeometry.ds_to_full` maps overview pixels into. ``zoom`` < 1 reads
    that region downscaled (used to build pyramid levels). Only decodes the
    sub-blocks intersecting the ROI, so it stays memory-bounded even on a
    wafer-scale montage. Returns a contrast-stretched uint8 RGB crop (via
    :func:`to_rgb8`) unless ``as_rgb8=False`` (then the raw Gray16/array).
    """
    from pylibCZIrw import czi as pyczi  # local import: optional dependency

    with pyczi.open_czi(path) as cz:
        bx, by, bw, bh = cz.total_bounding_rectangle
        # Clamp the ROI to the image so an edge view can't request out-of-bounds
        x0 = int(max(bx, min(x_full, bx + bw - 1)))
        y0 = int(max(by, min(y_full, by + bh - 1)))
        w = int(max(1, min(w_full, bx + bw - x0)))
        h = int(max(1, min(h_full, by + bh - y0)))
        kwargs = dict(roi=(x0, y0, w, h), plane={"C": channel}, zoom=float(zoom))
        if scene is not None:
            kwargs["scene"] = scene
        arr = cz.read(**kwargs)

    arr = np.squeeze(arr)
    return to_rgb8(arr) if as_rgb8 else arr


def percentile_lo_hi(arr: np.ndarray, low_pct: float = 1.0,
                     high_pct: float = 99.5) -> tuple[float, float]:
    """Compute the (lo, hi) contrast-stretch cuts used by :func:`to_rgb8`.

    Exposed so a multiscale pyramid can compute ONE global mapping from the
    overview and reuse it for every tile (per-tile percentiles would make
    adjacent tiles inconsistent -> brightness seams).
    """
    g = np.asarray(arr, dtype=np.float32)
    if g.ndim == 3:
        g = g[..., 0]
    lo, hi = np.percentile(g, [low_pct, high_pct])
    gmax = float(g.max())
    if hi >= gmax:
        sub = g[g < gmax]
        if sub.size:
            hi = float(np.percentile(sub, high_pct))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def to_rgb8(arr: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.5,
            clahe: bool = True, gamma: float | None = None,
            lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Convert a Gray16/Gray8/float image to a contrast-stretched uint8 RGB.

    SAM wants ``HxWx3`` uint8. Brightfield carbon-coated sections are low
    contrast, so we percentile-stretch and (by default) apply CLAHE before
    replicating to 3 channels. Pass explicit ``lo``/``hi`` (e.g. from
    :func:`percentile_lo_hi` on the overview) to apply a FIXED global mapping --
    used for pyramid tiles so they don't show per-tile brightness seams; in that
    mode ``clahe`` should be ``False`` (CLAHE is local and would also seam).
    """
    import cv2

    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[2] == 3:
        # already colour — just ensure uint8
        gray = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY) if a.dtype != np.uint8 \
            else cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    else:
        gray = a if a.ndim == 2 else a[..., 0]

    g = gray.astype(np.float32)
    if lo is None or hi is None:
        lo, hi = percentile_lo_hi(g, low_pct, high_pct)
    if hi <= lo:
        hi = lo + 1.0
    g = np.clip((g - lo) / (hi - lo), 0.0, 1.0)
    if gamma:
        g = np.power(g, gamma)
    g8 = (g * 255.0).astype(np.uint8)

    if clahe:
        clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g8 = clahe_op.apply(g8)

    return np.repeat(g8[:, :, None], 3, axis=2)


def build_czi_dask_pyramid(path: str, channel: int = 0, max_levels: int = 7,
                           tile: int = 3072, min_level_px: int = 1500,
                           contrast_long_side: int = 2048):
    """Lazily-loaded multiscale pyramid (list of dask arrays) for a CZI.

    Level 0 is full resolution (the total bounding rectangle); each subsequent
    level halves it, down to ~``min_level_px`` long side (or ``max_levels``).
    Each dask block lazily reads its region from the CZI at the level's zoom and
    maps it to uint8 RGB with a SINGLE global contrast mapping computed once from
    a small overview -- so only the tiles napari currently shows are decoded
    (full-res browsing without loading the ~18 GB image), and tiles are seamless.

    Returns ``(levels, geom)``: ``levels`` is a list of dask arrays ``(H,W,3)``
    uint8 (level 0 = full res); ``geom`` is a :class:`CziGeometry` with
    ``zoom=1.0`` mapping level-0 (bbox) pixels to full-res ZEN coords.
    """
    import dask
    import dask.array as da
    from pylibCZIrw import czi as pyczi

    raw_meta = parse_czi_metadata_raw(path)
    with pyczi.open_czi(path) as cz:
        bx, by, bw, bh = cz.total_bounding_rectangle

    # ONE global contrast mapping (from a small overview) reused by every tile.
    ov_arr, _, _ = read_czi_overview(path, target_long_side=contrast_long_side,
                                     channel=channel)
    lo, hi = percentile_lo_hi(ov_arr)

    geom = CziGeometry(zoom=1.0, origin_x=float(bx), origin_y=float(by),
                       scale_x=raw_meta.get("scale_x"), scale_y=raw_meta.get("scale_y"),
                       center_px_full=(bx + bw / 2.0, by + bh / 2.0))

    def _read_block(fx, fy, fw, fh, z, out_h, out_w):
        raw = read_czi_region(path, fx, fy, fw, fh, channel=channel,
                              zoom=z, as_rgb8=False)
        rgb = to_rgb8(raw, clahe=False, lo=lo, hi=hi)
        out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        hh, ww = min(out_h, rgb.shape[0]), min(out_w, rgb.shape[1])
        out[:hh, :ww] = rgb[:hh, :ww]
        return out

    levels = []
    for L in range(max_levels):
        z = 0.5 ** L
        lw = max(1, int(round(bw * z)))
        lh = max(1, int(round(bh * z)))
        rows = []
        for r0 in range(0, lh, tile):
            th = min(tile, lh - r0)
            cols = []
            for c0 in range(0, lw, tile):
                tw = min(tile, lw - c0)
                fx = int(bx + round(c0 / z))
                fy = int(by + round(r0 / z))
                fw = int(round(tw / z))
                fh = int(round(th / z))
                d = dask.delayed(_read_block)(fx, fy, fw, fh, z, th, tw)
                cols.append(da.from_delayed(d, shape=(th, tw, 3), dtype=np.uint8))
            rows.append(da.concatenate(cols, axis=1))   # along width
        levels.append(da.concatenate(rows, axis=0))      # along height
        if max(lw, lh) <= min_level_px:
            break
    return levels, geom


def is_czi(path: str) -> bool:
    return str(path).lower().endswith(".czi")


# --------------------------------------------------------------------------- #
# ZEN "Shuttle & Find" correlative fiducials (read-only, dependency-free)
# --------------------------------------------------------------------------- #
def parse_shuttle_and_find(xml_str: str) -> dict:
    """Parse ZEN's correlative-calibration fiducials out of CZI metadata XML.

    ZEN stores the LM↔SEM "Shuttle & Find" calibration markers (typically 3) at
    ``ImageDocument/Metadata/.../ShuttleAndFindData/Calibration/CorrelativeSession``
    in a ``<Markers>`` block — NOT as image-pixel annotations. Each ``<Marker>``
    carries ``StageXPosition``/``StageYPosition``/``FocusPosition`` in STAGE
    MICROMETERS (the motorised-stage frame), plus a ``<StageOrientation X= Y=>``
    (often ``-1,-1``) and ``<MicroscopeType>`` (e.g. ``LM``).

    Returns ``{"markers": [{"id","stage_x_um","stage_y_um","focus_um"}, ...],
    "stage_orientation": (x,y) | None, "holder": str | None,
    "microscope": str | None}``. ``markers`` is empty when the CZI has no
    Shuttle & Find calibration.
    """
    import xml.etree.ElementTree as ET

    out = {"markers": [], "stage_orientation": None,
           "holder": None, "microscope": None}
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return out
    cal = root.find(".//ShuttleAndFindData/Calibration")
    if cal is None:
        return out
    out["holder"] = cal.findtext("Holder") or None
    out["microscope"] = cal.findtext("MicroscopeType") or None
    so = cal.find("StageOrientation")
    if so is not None:
        try:
            out["stage_orientation"] = (int(float(so.get("X", "1"))),
                                        int(float(so.get("Y", "1"))))
        except (TypeError, ValueError):
            pass
    for m in cal.findall(".//Markers/Marker"):
        # Stage X/Y are the required fields — a marker without them is unusable.
        try:
            sx = float(m.get("StageXPosition"))
            sy = float(m.get("StageYPosition"))
        except (TypeError, ValueError):
            continue
        # Focus is optional: a present-but-garbage value must NOT discard the
        # (valid) marker, just null the focus.
        fx = m.get("FocusPosition")
        try:
            focus = float(fx) if fx not in (None, "") else None
        except (TypeError, ValueError):
            focus = None
        out["markers"].append({
            "id": m.get("Id"), "stage_x_um": sx, "stage_y_um": sy,
            "focus_um": focus,
        })
    return out


def read_shuttle_and_find_markers(path: str) -> dict:
    """Read ZEN Shuttle & Find correlative fiducials from a CZI (no CZI library).

    Thin wrapper over :func:`parse_shuttle_and_find` that pulls the metadata XML
    via :func:`parse_czi_metadata_raw` (so it works on a read-only drive and
    without ``pylibCZIrw``). See that function for the returned schema and the
    stage-micrometer caveat.
    """
    return parse_shuttle_and_find(parse_czi_metadata_raw(path)["xml"])
