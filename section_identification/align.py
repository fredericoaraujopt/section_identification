"""Per-section orientation/pose recovery from polygon shape (pure, numpy-only).

Sections land on the wafer at arbitrary in-plane rotations. To proofread them in
a common orientation and to propagate an ROI from a reference section onto all
others, we need each section's 2-D pose: its centre and the rotation that brings
it to a canonical (upright) frame.

Method (shape-only, no image content):
  * centre = polygon area-centroid (shoelace),
  * principal axis = the major eigenvector of the covariance of the perimeter
    (resampled densely so the estimate is robust to vertex count),
  * the 180° ambiguity of the principal axis is resolved by the **skewness** of
    the vertex projection onto that axis — the canonical major axis points
    toward the section's "heavy"/wider end, giving a repeatable full-range angle.

This yields the same canonical local frame for two copies of one section at
different rotations (the invariant the ROI-propagation stage relies on). A true
mirror/reflection (section collected face-down) is *not* inferred here — the
``flip`` field is reserved for that and left False. Content-based refinement
(SIFT registration to a reference) is layered on top in the reorder stage.

Returns plain ``(center_xy, angle_deg, flip)`` so callers can build a
``wafer_model.Pose`` without this module importing the model.
"""

from __future__ import annotations

import math

import numpy as np


def _area_centroid(poly: np.ndarray) -> np.ndarray:
    """Shoelace area-centroid of a polygon; falls back to vertex mean if
    degenerate (near-zero area)."""
    x, y = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-9:
        return poly.mean(axis=0)
    cx = ((x + x1) * cross).sum() / (6.0 * a)
    cy = ((y + y1) * cross).sum() / (6.0 * a)
    return np.array([cx, cy], dtype=float)


def _resample_perimeter(poly: np.ndarray, n: int = 256) -> np.ndarray:
    """Resample the closed polygon's perimeter to ``n`` evenly spaced points so
    PCA isn't biased by uneven vertex spacing."""
    closed = np.vstack([poly, poly[:1]])
    seg = np.diff(closed, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    total = seglen.sum()
    if total < 1e-9:
        return poly
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    targets = np.linspace(0.0, total, n, endpoint=False)
    out = np.empty((n, 2), dtype=float)
    j = 0
    for i, t in enumerate(targets):
        while j < len(seglen) - 1 and cum[j + 1] < t:
            j += 1
        f = 0.0 if seglen[j] < 1e-9 else (t - cum[j]) / seglen[j]
        out[i] = closed[j] + f * seg[j]
    return out


def estimate_pose(polygon, resample: int = 256) -> tuple[tuple[float, float], float, bool]:
    """Estimate ``((cx, cy), angle_deg, flip)`` for a section polygon.

    ``angle_deg`` is the orientation (in world, ``(x, y)``, degrees, full range)
    of the section's canonical major axis; rotating world points by ``-angle_deg``
    about the centre brings the section upright. ``flip`` is reserved (False).
    """
    poly = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if len(poly) < 3:
        c = poly.mean(axis=0) if len(poly) else np.zeros(2)
        return (float(c[0]), float(c[1])), 0.0, False

    center = _area_centroid(poly)
    pts = _resample_perimeter(poly, resample) - center
    cov = np.cov(pts.T)
    if not np.all(np.isfinite(cov)):
        return (float(center[0]), float(center[1])), 0.0, False
    vals, vecs = np.linalg.eigh(cov)
    major = vecs[:, int(np.argmax(vals))]            # eigenvector of larger eigenvalue

    # Resolve the 180° ambiguity: point the major axis toward the heavy end
    # (positive third moment of the projection).
    t = pts @ major
    if float(np.mean(t ** 3)) < 0:
        major = -major

    angle_deg = math.degrees(math.atan2(float(major[1]), float(major[0])))
    return (float(center[0]), float(center[1])), float(angle_deg), False


def pose_for_section(section, resample: int = 256):
    """Convenience: estimate a pose and stamp it onto a wafer_model.Section's
    ``pose`` (imported lazily to keep this module model-free)."""
    (cx, cy), angle, flip = estimate_pose(section.polygon, resample)
    section.pose.center = (cx, cy)
    section.pose.angle_deg = angle
    section.pose.flip = flip
    return section.pose
