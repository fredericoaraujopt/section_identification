"""Relative-field-of-view navigator math (pure, headless-testable).

Supports the workflow's key inspection gesture: when the user is zoomed onto a
feature in section *i* (say a blood vessel) and steps to section *j*, the camera
should land on the **same relative position within section j at the same
magnification** — so the corresponding feature is in view without re-navigating.

The trick is to express the camera target in each section's *local* frame
(translate to the section centre, rotate to its canonical/upright pose), carry
that relative offset across sections, and map it back to world coordinates using
the next section's pose. Magnification (napari ``camera.zoom``) is preserved.

All functions are frame-agnostic: pass camera positions, section centres, and
bboxes in **one consistent coordinate frame** (the GUI uses napari *world*
coordinates, with section poses converted to world first). Points are ``(x, y)``.
The thin napari wiring (reading/writing ``viewer.camera``, swapping the napari
``(y, x)`` order, converting overview↔world) lives in the GUI controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


def _rot(angle_deg: float) -> np.ndarray:
    t = math.radians(angle_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]], dtype=float)


def world_to_local(pt, center, angle_deg: float = 0.0, flip: bool = False) -> np.ndarray:
    """Express world point ``pt`` in a section's upright, centred local frame.

    Inverse of :func:`local_to_world`. ``flip`` is the 180° pose ambiguity.
    """
    v = np.asarray(pt, float).reshape(2) - np.asarray(center, float).reshape(2)
    local = _rot(-angle_deg) @ v
    if flip:
        local = -local
    return local


def local_to_world(local, center, angle_deg: float = 0.0, flip: bool = False) -> np.ndarray:
    """Map a local-frame offset back to world coordinates for a section."""
    local = np.asarray(local, float).reshape(2).copy()
    if flip:
        local = -local
    return np.asarray(center, float).reshape(2) + _rot(angle_deg) @ local


@dataclass
class _SecPose:
    center: tuple
    angle_deg: float = 0.0
    flip: bool = False


def relative_offset(cam_center, pose_from) -> np.ndarray:
    """The camera target as a local offset within ``pose_from`` (a section pose
    with ``.center``/``.angle_deg``/``.flip`` — e.g. wafer_model.Pose)."""
    return world_to_local(cam_center, pose_from.center,
                           getattr(pose_from, "angle_deg", 0.0),
                           getattr(pose_from, "flip", False))


def snapped_center(rel_offset, pose_to) -> np.ndarray:
    """Where the camera should centre to keep ``rel_offset`` within ``pose_to``."""
    return local_to_world(rel_offset, pose_to.center,
                          getattr(pose_to, "angle_deg", 0.0),
                          getattr(pose_to, "flip", False))


def snap_between(cam_center, pose_from, pose_to) -> np.ndarray:
    """Convenience: map the current camera centre (within ``pose_from``) to the
    equivalent relative position within ``pose_to``. Magnification is unchanged
    (the caller keeps ``viewer.camera.zoom``)."""
    return snapped_center(relative_offset(cam_center, pose_from), pose_to)


def fit_center_zoom(bbox, canvas_px, margin: float = 0.15) -> tuple[tuple, float]:
    """Camera ``(center_xy, zoom)`` that fits ``bbox=(x0,y0,x1,y1)`` into a
    ``canvas_px=(w_px, h_px)`` viewport with a fractional ``margin`` of padding.

    napari ``zoom`` is canvas pixels per world unit, so to fit a world width
    ``w`` into ``W`` pixels with margin we use ``zoom = W / (w * (1 + 2*margin))``
    and take the limiting axis. Returns ``((cx, cy), zoom)``.
    """
    x0, y0, x1, y1 = (float(v) for v in bbox)
    w = max(x1 - x0, 1e-6)
    h = max(y1 - y0, 1e-6)
    cw, ch = float(canvas_px[0]), float(canvas_px[1])
    pad = 1.0 + 2.0 * max(margin, 0.0)
    zoom = min(cw / (w * pad), ch / (h * pad))
    center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    return center, float(zoom)


def fit_pose(section, canvas_px, margin: float = 0.15) -> tuple[tuple, float]:
    """``fit_center_zoom`` for a wafer_model.Section's bbox (overview frame)."""
    return fit_center_zoom(section.bbox(), canvas_px, margin)
