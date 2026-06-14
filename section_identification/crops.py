"""Per-section image-crop reader, single-sourced for the GUI and the workers.

QC and SIFT operate on a crop of each section. CZI crops come from
``czi_io.read_czi_region`` (full-res for SIFT — blood vessels must resolve — or
downscaled for QC); PNG/montage crops are sliced from the overview array. Both
return a grayscale ``uint8`` crop plus a boolean section mask in the crop frame,
so wafer background never contaminates a measurement.

Pure enough to unit-test on a synthetic PNG; imported by qc_worker/reorder_worker
(which have no GUI) and by app_core.
"""

from __future__ import annotations

import numpy as np

from . import czi_io

try:
    import cv2
except Exception:                        # pragma: no cover
    cv2 = None


def full_bbox(polygon_overview, geom):
    """Integer full-resolution bbox ``(x0, y0, x1, y1)`` of an overview polygon."""
    p = np.asarray(polygon_overview, float).reshape(-1, 2)
    if geom is not None:
        fx, fy = geom.ds_to_full(p[:, 0], p[:, 1])
        x, y = np.ravel(fx), np.ravel(fy)
    else:
        x, y = p[:, 0], p[:, 1]
    return (int(np.floor(x.min())), int(np.floor(y.min())),
            int(np.ceil(x.max())), int(np.ceil(y.max())))


def _rasterize(polygon_full, x0, y0, scale, shape):
    """Boolean mask of the polygon in the crop frame (full px -> crop px)."""
    p = np.asarray(polygon_full, float).reshape(-1, 2)
    local = np.column_stack([(p[:, 0] - x0) * scale, (p[:, 1] - y0) * scale])
    mask = np.zeros(shape[:2], np.uint8)
    if cv2 is not None and len(local) >= 3:
        cv2.fillPoly(mask, [np.round(local).astype(np.int32)], 1)
    return mask.astype(bool)


def read_section_crop(image_path, geom, polygon_overview, overview=None,
                      full_res: bool = False, target_long_side: int = 768):
    """Return ``(gray_uint8, mask_bool, (x0, y0, scale))`` for a section.

    ``full_res=True`` reads at native resolution (SIFT); otherwise downscales so
    the crop's long side ≈ ``target_long_side`` (QC). ``scale`` is crop-px per
    full-px. ``overview`` (the GUI's in-memory PNG array) avoids a re-read for
    non-CZI images; workers pass None and the PNG is loaded here.
    """
    x0, y0, x1, y1 = full_bbox(polygon_overview, geom)
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    poly_full = _poly_full(polygon_overview, geom)

    if czi_io.is_czi(image_path) and geom is not None:
        zoom = 1.0 if full_res else min(1.0, float(target_long_side) / max(w, h))
        rgb = czi_io.read_czi_region(image_path, x0, y0, w, h, zoom=zoom, as_rgb8=True)
        gray = _to_gray(rgb)
        scale = gray.shape[1] / float(w)            # actual achieved scale
    else:
        if overview is None:
            from PIL import Image
            overview = np.array(Image.open(image_path).convert("RGB"))
        cx0, cy0 = max(0, x0), max(0, y0)
        crop = overview[cy0:y1, cx0:x1]
        scale = 1.0
        if not full_res and max(w, h) > target_long_side and crop.size:
            scale = float(target_long_side) / max(w, h)
            if cv2 is not None:
                crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)),
                                         max(1, int(crop.shape[0] * scale))))
        gray = _to_gray(crop)
        x0, y0 = cx0, cy0

    mask = _rasterize(poly_full, x0, y0, scale, gray.shape)
    return gray, mask, (x0, y0, scale)


def _poly_full(polygon_overview, geom):
    p = np.asarray(polygon_overview, float).reshape(-1, 2)
    if geom is None:
        return p
    fx, fy = geom.ds_to_full(p[:, 0], p[:, 1])
    return np.column_stack([np.ravel(fx), np.ravel(fy)])


def _to_gray(arr):
    a = np.asarray(arr)
    if a.ndim == 3:
        if cv2 is not None:
            return cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return a.mean(axis=2).astype(np.uint8)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a
