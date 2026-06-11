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


def _thresholds_for(d_sam_px):
    """SAM confidence/stability thresholds by object size in the 1024 space
    (smaller objects → looser thresholds; SAM is less confident on them)."""
    if d_sam_px >= 20:
        return 0.85, 0.96
    if d_sam_px >= 10:
        return 0.80, 0.95
    return 0.75, 0.92


def calibrate(example_polygons, geom=None, target_sam_px=64,
              min_area_frac=0.5, max_area_mult=2.0, overview_long_side=None,
              target_working_px=64, max_overview_long_side=12000, profile=None):
    """Derive the FULL SAM parameter set + a detection strategy from example
    section polygons (overview coords), following SAM best practices.

    Every parameter is keyed off the typical section size and a single
    ``target_sam_px`` dial (how big a section should appear to SAM, which always
    works at 1024 px). When a ``profile`` (host_profile.HostProfile) is given it
    is treated as a budget: tile/overview/batch/model are clamped so the plan is
    feasible on the host (feasibility before fidelity), recorded in
    ``plan_summary``. Returns a dict with the legacy keys plus ``crop_n_layers``,
    ``crop_overlap_ratio``, ``crop_n_points_downscale_factor``, ``pred_iou_thresh``,
    ``stability_score_thresh``, ``stability_score_offset``, ``min_mask_region_area``,
    ``points_per_batch``, ``model_variant``, ``resolution_ok`` and ``plan_summary``.
    """
    if not example_polygons:
        raise ValueError("Need at least one example polygon to calibrate.")
    areas = np.array([_polygon_area(p) for p in example_polygons], dtype=float)
    areas = areas[areas > 0]
    if areas.size == 0:
        raise ValueError("Example polygons have zero area.")
    median_area = float(np.median(areas))
    section_px = float(np.sqrt(median_area))            # overview px (square-equiv)

    z = geom.zoom if (geom is not None and getattr(geom, "zoom", None)) else 1.0
    section_px_full = section_px / z

    # Tile size (overview px). If the WHOLE image already shows a section at
    # ≥ target_sam_px to SAM (SAM resizes the input to 1024), don't tile for
    # resolution — only the host memory cap can then force tiling. Otherwise
    # pick a tile small enough that a section is ~target_sam_px to SAM.
    ov_long = int(overview_long_side or 0) or max(int(round(section_px * 40)), 2048)
    tile_cap = int(getattr(profile, "tile_cap_px", 4096) or 4096)
    whole_to_sam = section_px * 1024.0 / max(ov_long, 1.0)
    ideal_tile = ov_long if whole_to_sam >= target_sam_px \
        else int(round(1024.0 * section_px / float(target_sam_px)))
    tile_px = int(np.clip(ideal_tile, 256, min(ov_long, tile_cap)))

    # How big a section is to SAM inside that tile, and whether tiling is needed.
    d_sam = section_px * 1024.0 / max(tile_px, 1.0)
    tiling_recommended = bool(tile_px < ov_long)

    # Grid: ≥ ~2.5 query points across a section within the tile.
    pps = int(np.clip(np.ceil(2.5 * tile_px / max(section_px, 1.0)), 16, 128))
    overlap = float(np.clip(1.5 * section_px / max(tile_px, 1.0), 0.15, 0.5))

    # SAM's built-in cropping for sections still small within a tile.
    if d_sam >= 16:
        crop_n_layers, crop_downscale = 0, 1
    else:
        crop_n_layers, crop_downscale = 1, 2
    # Crop overlap scaled to section size: a section straddling a sub-crop edge
    # must sit whole inside a neighbour. SAM's overlap ≈ ratio × tile_px, so set
    # ratio so that overlap exceeds ~1.5× a section.
    crop_overlap_ratio = float(np.clip(1.5 * section_px / max(tile_px, 1.0), 0.2, 0.6))

    pred_iou, stab = _thresholds_for(d_sam)
    pps_batch = int(getattr(profile, "points_per_batch", 16) or 16)
    model_variant = getattr(profile, "model_variant", None) or "base_plus"

    cal = {
        "n_examples": int(areas.size),
        "median_area": median_area,
        "section_px": section_px,
        "section_px_full": section_px_full,
        "median_area_full": median_area / (z * z),
        # area band for the DBSCAN/area post-filter (user's method)
        "min_area": float(min_area_frac * median_area),
        "max_area": float(max_area_mult * median_area),
        # strategy
        "tile_px": tile_px,
        "overlap": overlap,
        "tiling_recommended": tiling_recommended,
        "target_sam_px": float(target_sam_px),
        "section_to_sam_px": float(d_sam),
        # SAM AMG params
        "points_per_side": pps,
        "points_per_batch": pps_batch,
        "pred_iou_thresh": float(pred_iou),
        "stability_score_thresh": float(stab),
        "stability_score_offset": 1.0,
        "box_nms_thresh": 0.7,
        "crop_n_layers": int(crop_n_layers),
        "crop_overlap_ratio": crop_overlap_ratio,
        "crop_n_points_downscale_factor": int(crop_downscale),
        # SAM cleanup floor (processed px) — separate from the DBSCAN band
        "min_mask_region_area": int(round(0.05 * median_area)),
        "model_variant": model_variant,
    }

    if overview_long_side:
        full_long = overview_long_side / z
        whole_sam_px = section_px_full * 1024.0 / max(full_long, 1.0)
        cal["whole_image_section_px"] = float(whole_sam_px)
        rec = full_long * (float(target_working_px) / max(section_px_full, 1.0))
        cap = int(getattr(profile, "overview_cap_px", max_overview_long_side)
                  or max_overview_long_side)
        cal["recommended_overview_long_side"] = int(
            np.clip(rec, overview_long_side, min(max_overview_long_side, cap)))
        # Real detail is OK if a section already has enough REAL overview px.
        cal["resolution_ok"] = bool(section_px >= 0.85 * target_working_px)

    cal["plan_summary"] = _plan_summary(cal, profile)
    return cal


def _plan_summary(cal, profile=None):
    """Plain-language detection plan for the GUI."""
    strat = ("tiles" if cal.get("tiling_recommended") else "whole image")
    parts = [
        f"Sections ≈ {cal.get('section_px_full', cal['section_px']):.0f} px.",
        f"SAM will see them at ~{cal['section_to_sam_px']:.0f} px "
        f"({strat}, tile {cal['tile_px']}px, grid {cal['points_per_side']}/tile"
        + (f", crop_n_layers={cal['crop_n_layers']}" if cal['crop_n_layers'] else "")
        + ").",
    ]
    if profile is not None:
        parts.append(f"Host: {getattr(profile, 'device_label', profile.device)}, "
                     f"hiera_{cal.get('model_variant', '?')}, batch≤{cal['points_per_batch']}.")
    if "resolution_ok" in cal and not cal["resolution_ok"]:
        parts.append(f"⤴ Detail is resolution-limited — consider overview "
                     f"≈ {cal.get('recommended_overview_long_side', '?')} px for sharper masks.")
    elif "resolution_ok" in cal:
        parts.append("Resolution: good.")
    return " ".join(parts)


def summary(cal):
    """One-line human-readable calibration summary for the GUI log."""
    s = (f"calibrated from {cal['n_examples']} examples: section ~{cal['section_px']:.0f}px "
         f"(area {cal['median_area']:.0f}); keep area {cal['min_area']:.0f}-{cal['max_area']:.0f}; "
         f"tile_px={cal['tile_px']}, points_per_side={cal.get('points_per_side', '?')}, "
         f"crop_n_layers={cal.get('crop_n_layers', 0)}, pred_iou={cal.get('pred_iou_thresh', 0):.2f}")
    if "section_px_full" in cal:
        s += f"  [full-res section ~{cal['section_px_full']:.0f}px]"
    return s
