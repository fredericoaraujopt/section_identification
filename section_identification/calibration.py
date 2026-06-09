"""Calibrate the detector from a few user-drawn example sections.

You can't retrain SAM from a handful of examples, but you *can* set the
parameters that actually decide recall and trash-rejection for the wafer at hand:
the section size band (for the area/DBSCAN filter and the SAM minimum area) and
the working tile size (so a typical section is a good pixel size for SAM). This
turns "guess the parameters" into "draw 2-5 sections and go", and it implements
the user's spec directly: the minimum kept area is ~half the median section.

All inputs are polygons in the working-image (overview) frame; outputs are in
the same frame (areas in overview px), with full-resolution equivalents reported
via the optional :class:`CziGeometry`.
"""

from __future__ import annotations

import numpy as np


def _polygon_area(poly):
    import cv2
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    return float(abs(cv2.contourArea(p)))


def calibrate(example_polygons, geom=None, target_sam_px=100,
              min_area_frac=0.5, max_area_mult=2.0):
    """Derive detector settings from example section polygons (overview coords).

    Returns a dict:
      * ``median_area`` / ``section_px``  — typical section size (overview px).
      * ``min_area`` = ``min_area_frac`` x median  (smallest kept; user's spec).
      * ``max_area`` = ``max_area_mult`` x median  (drops 2x+ clumps/debris).
      * ``tile_px``  — SAM tile size so a section is ~``target_sam_px`` px after
        SAM resizes the tile to 1024 (smaller tile -> bigger section to SAM).
      * ``n_examples`` and, if ``geom`` given, ``section_px_full`` /
        ``median_area_full`` for reporting in full-resolution units.
    """
    if not example_polygons:
        raise ValueError("Need at least one example polygon to calibrate.")
    areas = np.array([_polygon_area(p) for p in example_polygons], dtype=float)
    areas = areas[areas > 0]
    if areas.size == 0:
        raise ValueError("Example polygons have zero area.")
    median_area = float(np.median(areas))
    section_px = float(np.sqrt(median_area))

    # tile_px so section_px * (1024/tile_px) ~= target_sam_px, clamped sensibly.
    tile_px = int(round(1024.0 * section_px / float(target_sam_px)))
    tile_px = int(np.clip(tile_px, 256, 1024))

    cal = {
        "n_examples": int(areas.size),
        "median_area": median_area,
        "section_px": section_px,
        "min_area": float(min_area_frac * median_area),
        "max_area": float(max_area_mult * median_area),
        "tile_px": tile_px,
        "target_sam_px": float(target_sam_px),
    }
    if geom is not None and getattr(geom, "zoom", None):
        z = geom.zoom
        cal["section_px_full"] = section_px / z
        cal["median_area_full"] = median_area / (z * z)
    return cal


def summary(cal):
    """One-line human-readable calibration summary for the GUI log."""
    s = (f"calibrated from {cal['n_examples']} examples: section ~{cal['section_px']:.0f}px "
         f"(area {cal['median_area']:.0f}); keep area {cal['min_area']:.0f}-{cal['max_area']:.0f}; "
         f"tile_px={cal['tile_px']}")
    if "section_px_full" in cal:
        s += f"  [full-res section ~{cal['section_px_full']:.0f}px]"
    return s
