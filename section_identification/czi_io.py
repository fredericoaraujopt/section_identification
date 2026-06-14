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

import json
import os
import re
import struct
import xml.etree.ElementTree as ET
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

    # Stage<->pixel anchor for the Shuttle & Find correlative frame, parsed with
    # SCOPED ElementTree (a document-wide regex would happily match an unrelated
    # <Shape>/<TileRegion> <CenterPosition>, or miss a Y-before-X attribute order):
    #  * the scene's <CenterPosition>x,y</CenterPosition> is the stage position
    #    (µm) at the image centre,
    #  * <StageOrientation X= Y=> (in the S&F calibration) is the sign relating
    #    stage axes to pixel axes (commonly -1,-1).
    def _scene_anchor():
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            return None, None
        # explicit is-None: a <CenterPosition> has text but no children, so it is
        # falsy — `find() or find()` would fall through to the TileRegion one.
        cp = root.find(".//Scenes/Scene/CenterPosition")
        if cp is None:
            cp = root.find(".//TileRegions/TileRegion/CenterPosition")
        center = None
        if cp is not None and cp.text and "," in cp.text:
            try:
                a, b = cp.text.split(",")[:2]
                center = (float(a), float(b))
            except ValueError:
                center = None
        so = root.find(".//StageOrientation")
        orient = None
        if so is not None:
            try:
                orient = (int(float(so.get("X", "1"))), int(float(so.get("Y", "1"))))
            except (TypeError, ValueError):
                orient = None
        return center, orient

    size_s = _int("SizeS")
    scene_center, stage_orient = _scene_anchor()
    # The stage anchor maps the scene centre to the TOTAL bounding-rectangle
    # centre downstream; that identity only holds for a single scene. Don't
    # expose an anchor for multi-scene montages (callers then degrade to "no
    # stage anchor" rather than silently applying a constant offset).
    if size_s not in (None, 1):
        scene_center = None

    return {
        "size_x": _int("SizeX"),
        "size_y": _int("SizeY"),
        "size_c": _int("SizeC"),
        "size_s": size_s,
        "size_m": _int("SizeM"),
        "scale_x": _scaling("X"),
        "scale_y": _scaling("Y"),
        "pixel_type": _first("PixelType"),
        "scene_center_um": scene_center,
        "stage_orientation": stage_orient,
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
    x_direction: int = 1   # sign relating a stage axis to a pixel axis (±1)
    y_direction: int = 1
    swap_xy: bool = False  # image axes transposed vs stage (camera rotated 90°)

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
        """Full-res-pixel -> stage microns (ZEN Shuttle & Find frame).

        Anchors the scene centre (``stage_center_um``) to the image centre
        (``center_px_full``), then applies the scale, axis signs, and (if
        ``swap_xy``) a 90° transpose between image and stage axes. Requires the
        stage anchor + scale; returns ``None`` otherwise so callers degrade
        gracefully. Inverse of :meth:`stage_um_to_full`.
        """
        if (self.scale_x is None or self.scale_y is None
                or self.stage_center_um is None or self.center_px_full is None):
            return None
        sx_um, sy_um = self.scale_x * 1e6, self.scale_y * 1e6
        cx_um, cy_um = self.stage_center_um
        cx_px, cy_px = self.center_px_full
        fx = np.asarray(x_full, dtype=float) - cx_px
        fy = np.asarray(y_full, dtype=float) - cy_px
        if self.swap_xy:                       # pixel-x ↔ stage-y, pixel-y ↔ stage-x
            x_um = cx_um + fy * sy_um * self.y_direction
            y_um = cy_um + fx * sx_um * self.x_direction
        else:
            x_um = cx_um + fx * sx_um * self.x_direction
            y_um = cy_um + fy * sy_um * self.y_direction
        return x_um, y_um

    def stage_um_to_full(self, x_um, y_um):
        """Stage microns -> full-res pixel (inverse of :meth:`full_to_stage_um`).

        Returns ``None`` when the stage anchor / scale are unknown.
        """
        if (self.scale_x is None or self.scale_y is None
                or self.stage_center_um is None or self.center_px_full is None):
            return None
        sx_um, sy_um = self.scale_x * 1e6, self.scale_y * 1e6
        cx_um, cy_um = self.stage_center_um
        cx_px, cy_px = self.center_px_full
        dx = np.asarray(x_um, dtype=float) - cx_um
        dy = np.asarray(y_um, dtype=float) - cy_um
        if self.swap_xy:                       # stage-y -> pixel-x, stage-x -> pixel-y
            x_full = cx_px + (dy / sx_um) * self.x_direction
            y_full = cy_px + (dx / sy_um) * self.y_direction
        else:
            x_full = cx_px + (dx / sx_um) * self.x_direction
            y_full = cy_px + (dy / sy_um) * self.y_direction
        return x_full, y_full


# --------------------------------------------------------------------------- #
# Reading + 16-bit -> 8-bit RGB
# --------------------------------------------------------------------------- #
def _stage_pixel_transform(stage_orientation) -> tuple[bool, int, int]:
    """How the CZI's stage axes map to image pixel axes — ``(swap_xy, x_dir, y_dir)``.

    Determined empirically against a real wafer (M411 'Axio ImagerVario', whose 3
    Shuttle & Find marks sit at the top-right / bottom-right / bottom-left wafer
    corners): the camera image is **transposed** relative to the stage (rotated
    90°) — image-x tracks stage-y and image-y tracks stage-x — with positive
    signs. ZEN's ``<StageOrientation>`` (here -1,-1) does NOT predict this on its
    own, so the transform is fixed here rather than derived. This single helper is
    the one place to revisit if a different instrument's imported fiducials land
    on the wrong corners.
    """
    return True, 1, 1


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
    swap, xd, yd = _stage_pixel_transform(raw_meta.get("stage_orientation"))
    geom = CziGeometry(
        zoom=zoom,
        origin_x=float(x0),
        origin_y=float(y0),
        scale_x=raw_meta.get("scale_x"),
        scale_y=raw_meta.get("scale_y"),
        stage_center_um=raw_meta.get("scene_center_um"),
        center_px_full=(x0 + w / 2.0, y0 + h / 2.0),
        x_direction=xd, y_direction=yd, swap_xy=swap,
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

    swap, xd, yd = _stage_pixel_transform(raw_meta.get("stage_orientation"))
    geom = CziGeometry(zoom=1.0, origin_x=float(bx), origin_y=float(by),
                       scale_x=raw_meta.get("scale_x"), scale_y=raw_meta.get("scale_y"),
                       stage_center_um=raw_meta.get("scene_center_um"),
                       center_px_full=(bx + bw / 2.0, by + bh / 2.0),
                       x_direction=xd, y_direction=yd, swap_xy=swap)

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


# --------------------------------------------------------------------------- #
# Persisted (on-disk) Zarr pyramid cache
# --------------------------------------------------------------------------- #
# `build_czi_dask_pyramid` is instant, but every later zoom/pan re-decodes the
# visible region straight from the CZI via pylibCZIrw -- that decode+decompress
# on the interaction path is what makes fast zooming feel laggy. Persisting the
# pyramid once to a chunked Zarr (Blosc) next to the image turns every later
# session's zoom into a fast block read instead of a CZI decode. Inspired by the
# mVis viewer, which builds its pyramid once and caches it beside the source.

ZARR_CHUNK = 1024  # storage chunk (px). Smaller than the 3072 build tile so a
                   # pan reads less per step and the block cache is finer-grained.


def zarr_pyramid_path(out_dir: str, image_path: str) -> str:
    """Path of the cached Zarr pyramid for ``image_path`` inside ``out_dir``."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(out_dir, f"{stem}_display_pyramid.zarr")


def _zarr_meta_path(zpath: str) -> str:
    return os.path.join(zpath, "_stim_pyramid.json")


def _read_zarr_meta(zpath: str):
    try:
        with open(_zarr_meta_path(zpath)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def zarr_pyramid_exists(zpath: str, image_path: str | None = None) -> bool:
    """True only if a COMPLETE, current Zarr pyramid is present at ``zpath``.

    A build that crashed mid-write leaves level dirs but no ``complete`` flag, so
    it is treated as absent (and rebuilt). When ``image_path`` is given, a cache
    is also invalidated if the source CZI's size/mtime no longer match.
    """
    meta = _read_zarr_meta(zpath)
    if not meta or not meta.get("complete"):
        return False
    for i in range(meta.get("n_levels", 0)):
        if not os.path.isdir(os.path.join(zpath, str(i))):
            return False
    if image_path is not None:
        src = meta.get("source") or {}
        try:
            st = os.stat(image_path)
        except OSError:
            return False
        if src.get("size") != st.st_size or int(src.get("mtime", -1)) != int(st.st_mtime):
            return False
    return True


def open_czi_zarr_pyramid(zpath: str):
    """Open a cached Zarr pyramid as a list of dask arrays (level 0 = full res)."""
    import dask.array as da
    meta = _read_zarr_meta(zpath)
    if not meta:
        raise FileNotFoundError(f"no Zarr pyramid metadata at {zpath!r}")
    return [da.from_zarr(os.path.join(zpath, str(i)))
            for i in range(meta["n_levels"])]


def write_czi_zarr_pyramid(image_path: str, zpath: str, channel: int = 0,
                           progress=None, **build_kwargs):
    """Build the CZI display pyramid once and persist it to a chunked Zarr.

    Decodes the whole CZI exactly once with bounded memory (Dask streams the
    pyramid block by block). Writes to a ``.part`` dir and renames atomically so
    a crash never leaves a half-written cache that passes :func:`zarr_pyramid_exists`.
    Returns the freshly-written pyramid reopened as dask-from-Zarr arrays, so the
    caller can switch the display to the fast on-disk copy immediately.
    ``progress(level_done, level_total)`` is called per level if provided.
    """
    import shutil
    import dask
    import dask.array as da

    levels, _ = build_czi_dask_pyramid(image_path, channel=channel, **build_kwargs)
    tmp = zpath + ".part"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    n = len(levels)
    shapes = []
    # Force the SINGLE-THREADED scheduler for the decode. Each block opens the CZI
    # via pylibCZIrw, which is not safe under concurrent reads — the default threaded
    # scheduler races (a worker dies and surfaces as ``KeyError: '_read_block-…'``)
    # and runs many full-res blocks at once, breaking the bounded-memory guarantee.
    # Synchronous = one block at a time; this runs in a background thread already.
    with dask.config.set(scheduler="single-threaded"):
        for i, lvl in enumerate(levels):
            ch = (min(ZARR_CHUNK, lvl.shape[0]), min(ZARR_CHUNK, lvl.shape[1]), 3)
            da.to_zarr(lvl.rechunk(ch), os.path.join(tmp, str(i)), overwrite=True)
            shapes.append([int(s) for s in lvl.shape])
            if progress is not None:
                progress(i + 1, n)
    try:
        st = os.stat(image_path)
        src = {"size": st.st_size, "mtime": int(st.st_mtime)}
    except OSError:
        src = {}
    with open(os.path.join(tmp, "_stim_pyramid.json"), "w") as f:
        json.dump({"complete": True, "n_levels": n, "shapes": shapes,
                   "chunk": ZARR_CHUNK, "source": src}, f)
    if os.path.isdir(zpath):
        shutil.rmtree(zpath, ignore_errors=True)
    os.replace(tmp, zpath)
    return open_czi_zarr_pyramid(zpath)


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
