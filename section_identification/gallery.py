"""Aligned section gallery — a montage of pose-rectified section thumbnails.

A powerful proofreading visual: every section rotated to its canonical (upright)
pose and laid out in a grid, so the user can scan hundreds of sections for
mis-detections, wrong orientations, or odd shapes at a glance. The montage
builder is pure (numpy/opencv) and headless-testable; the GUI shows it in a
popup.

Reading a crop per section is the cost, so the gallery caps how many it renders
and logs the cap (no silent truncation).
"""

from __future__ import annotations

import math

import numpy as np

try:
    import cv2
except Exception:                        # pragma: no cover
    cv2 = None


def upright_thumb(gray, angle_deg: float, size: int = 96) -> np.ndarray:
    """Rotate ``gray`` by ``-angle_deg`` (to canonical upright) and fit into a
    ``size×size`` uint8 thumbnail (letterboxed)."""
    g = np.asarray(gray)
    if g.ndim == 3:
        g = g.mean(axis=2)
    g = g.astype(np.float32)
    h, w = g.shape[:2]
    if cv2 is not None:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -float(angle_deg), 1.0)
        g = cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    else:                                # pragma: no cover
        from scipy.ndimage import rotate
        g = rotate(g, angle_deg, reshape=False, order=1)
    return _fit_square(g, size)


def _fit_square(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = float(size) / max(h, w, 1)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    if cv2 is not None:
        small = cv2.resize(img, (nw, nh))
    else:                                # pragma: no cover
        small = img[:: max(1, h // size), :: max(1, w // size)]
        nh, nw = small.shape[:2]
    out = np.zeros((size, size), np.float32)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small[:size, :size]
    return np.clip(out, 0, 255).astype(np.uint8)


def build_montage(thumbs, cols: int = 8, pad: int = 2, bg: int = 30) -> np.ndarray:
    """Tile equal-size thumbnails into a single grayscale montage image."""
    thumbs = [np.asarray(t) for t in thumbs]
    if not thumbs:
        return np.zeros((1, 1), np.uint8)
    size = thumbs[0].shape[0]
    n = len(thumbs)
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)
    cell = size + pad
    mont = np.full((rows * cell + pad, cols * cell + pad), bg, np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        y, x = r * cell + pad, c * cell + pad
        mont[y:y + size, x:x + size] = t[:size, :size]
    return mont


def build_gallery(app, max_sections: int = 64, thumb: int = 96, cols: int = 8):
    """Build the aligned montage for the current project. Returns
    ``(montage, n_rendered, n_total)``. Caps at ``max_sections`` (logged)."""
    from . import crops
    app.ensure_poses()
    secs = app.project.sections
    n_total = len(secs)
    use = secs[:max_sections]
    if n_total > max_sections:
        app.log("gallery", f"rendering first {max_sections} of {n_total} sections.")
    thumbs = []
    for s in use:
        try:
            gray, _mask, _ = crops.read_section_crop(
                app.image_path, app.geom, s.polygon, overview=app.overview,
                full_res=False, target_long_side=thumb * 3)
            thumbs.append(upright_thumb(gray, s.pose.angle_deg, thumb))
        except Exception:
            thumbs.append(np.zeros((thumb, thumb), np.uint8))
    return build_montage(thumbs, cols=cols), len(thumbs), n_total
