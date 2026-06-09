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


def to_rgb8(arr: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.5,
            clahe: bool = True, gamma: float | None = None) -> np.ndarray:
    """Convert a Gray16/Gray8/float image to a contrast-stretched uint8 RGB.

    SAM wants ``HxWx3`` uint8. Brightfield carbon-coated sections are low
    contrast, so we percentile-stretch and (by default) apply CLAHE before
    replicating to 3 channels.
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
    lo, hi = np.percentile(g, [low_pct, high_pct])
    # Brightfield montages often contain saturated specular spikes (e.g. uint16
    # 65535) that hijack the high percentile and crush all real contrast. If the
    # high cut equals the global max, recompute it on the non-saturated bulk.
    gmax = float(g.max())
    if hi >= gmax:
        sub = g[g < gmax]
        if sub.size:
            hi = float(np.percentile(sub, high_pct))
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


def is_czi(path: str) -> bool:
    return str(path).lower().endswith(".czi")
