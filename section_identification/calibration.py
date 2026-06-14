"""Calibrate the detector from a few user-drawn example sections.

You can't retrain SAM from a handful of examples, but you *can* set the two
parameters that actually decide whether SAM finds the wafer's sections —
``tile_px`` and ``points_per_side`` — directly from the calibrated example's
geometry. The overview stays at the encoder's native 1024 px frame (reading a
finer overview buys no apparent size on a whole-image pass), so the method is:

1. **Resolve, then tile only if needed.** A single whole-image tile shows a
   section to SAM at ``minor_axis * 1024 / overview_px``. If that clears SAM's
   resolvable floor (``resolve_px``) keep one tile; only a sub-floor section is
   split into tiles, sized to magnify its thin axis back up to the floor.
2. **Grid for ≥2 points per section.** The seed-point count on a section follows
   ``points ≈ area * (pps / tile_px)²`` (measured 1.47 vs 1.45 predicted at grid
   32 on M411). Solve it for the target count (default 2).

The section size band (area/DBSCAN filter, SAM min area) is still set from the
median area. All inputs are polygons in the working-image (overview) frame;
outputs are in the same frame (areas in overview px), with full-resolution
equivalents reported via the optional :class:`CziGeometry`.
"""

from __future__ import annotations

import numpy as np


def _polygon_area(poly):
    import cv2
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    return float(abs(cv2.contourArea(p)))


def _minor_axis(poly):
    """Thin-axis length (px) of a polygon's min-area bounding box. This is the
    dimension that decides whether SAM can *resolve* the section — a long thin
    strip is hard to segment however long it is."""
    import cv2
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    (_, _), (w, h), _ = cv2.minAreaRect(p)
    return float(min(w, h))


def _thresholds_for(d_sam_px):
    """SAM confidence/stability thresholds by object size in the 1024 space
    (smaller objects → looser thresholds; SAM is less confident on them)."""
    if d_sam_px >= 20:
        return 0.85, 0.96
    if d_sam_px >= 10:
        return 0.80, 0.95
    return 0.75, 0.92


def calibrate(example_polygons, geom=None,
              min_area_frac=0.5, max_area_mult=2.0, overview_long_side=None,
              profile=None, resolve_px=24.0, min_points=2):
    """Derive the FULL SAM parameter set + a detection strategy from example
    section polygons (overview coords).

    Two knobs carry the plan, both keyed off the calibrated example:

    * ``tile_px`` — single whole-image tile UNLESS the section's thin (minor)
      axis falls below ``resolve_px`` in SAM's 1024 frame, in which case the
      image is split into tiles that magnify the thin axis back up to the floor.
    * ``points_per_side`` — solved from ``points ≈ area * (pps/tile_px)²`` so a
      typical section catches at least ``min_points`` seed points.

    The overview is left at its native frame (no resolution recommendation — the
    encoder works at 1024 px and tiling, not a finer overview, is what magnifies
    a section). When a ``profile`` (host_profile.HostProfile) is given the tile
    is clamped to the host's memory cap. Returns a dict with the legacy keys plus
    ``minor_px``, ``points_on_section``, ``resolve_px``, ``min_points``,
    ``crop_n_layers``, the quality thresholds, ``model_variant`` and
    ``plan_summary``.
    """
    if not example_polygons:
        raise ValueError("Need at least one example polygon to calibrate.")
    areas = np.array([_polygon_area(p) for p in example_polygons], dtype=float)
    minors = np.array([_minor_axis(p) for p in example_polygons], dtype=float)
    keep = areas > 0
    areas, minors = areas[keep], minors[keep]
    if areas.size == 0:
        raise ValueError("Example polygons have zero area.")
    median_area = float(np.median(areas))
    section_px = float(np.sqrt(median_area))            # overview px (square-equiv) → point COUNT
    minor_px = float(np.median(minors))                 # overview px (thin axis)   → RESOLVABILITY

    z = geom.zoom if (geom is not None and getattr(geom, "zoom", None)) else 1.0
    section_px_full = section_px / z
    minor_px_full = minor_px / z

    ov_long = int(overview_long_side or 0) or max(int(round(section_px * 40)), 2048)
    tile_cap = int(getattr(profile, "tile_cap_px", 4096) or 4096)

    # --- 1. Resolution gate. SAM's tile is pinned to 1024 px (1:1 into the encoder,
    #     no upscaling), so a 1024-px tile shows a section's thin axis to SAM at exactly
    #     minor_px (its size in THIS overview frame). 1024-px tiles therefore resolve the
    #     section iff minor_px >= resolve_px; the finer overview (N·1024) that makes that
    #     true is recommended below. apparent_single is the thin axis on a whole 1024-px
    #     image (invariant to the loaded overview) and drives that recommendation.
    apparent_single = minor_px * 1024.0 / max(ov_long, 1.0)
    if minor_px >= resolve_px:
        tile_px = 1024 if ov_long > 1024 else ov_long   # 1024-native tiles; whole if ov==1024
    else:
        # Not yet at a fine-enough overview and still sub-floor: upscale within a
        # sub-1024 tile as a fallback (real fix is the recommended overview below).
        tile_px = int(np.clip(round(minor_px * 1024.0 / float(resolve_px)), 256, ov_long))
    tile_px = int(min(tile_px, tile_cap))               # host memory may force smaller tiles
    tiling_recommended = bool(tile_px < ov_long)

    # --- 2. Grid: maintain ≥ min_points seed points on a section, in the tile
    #     frame. Validated model: points ≈ area * (pps/tile_px)². Solve for pps.
    pps = int(np.clip(np.ceil(tile_px * np.sqrt(float(min_points) / max(median_area, 1.0))),
                      16, 128))
    points_on_section = float(median_area * (pps / float(max(tile_px, 1))) ** 2)

    # Apparent size of a typical (area-equiv) section to SAM — for the quality gates.
    d_sam = section_px * 1024.0 / max(tile_px, 1.0)
    overlap = float(np.clip(1.5 * section_px / max(tile_px, 1.0), 0.15, 0.5))

    # SAM's built-in sub-cropping only if a section is STILL tiny inside the tile.
    if d_sam >= 16:
        crop_n_layers, crop_downscale = 0, 1
    else:
        crop_n_layers, crop_downscale = 1, 2
    crop_overlap_ratio = float(np.clip(1.5 * section_px / max(tile_px, 1.0), 0.2, 0.6))

    pred_iou, stab = _thresholds_for(d_sam)
    pps_batch = int(getattr(profile, "points_per_batch", 16) or 16)
    model_variant = getattr(profile, "model_variant", None) or "base_plus"

    cal = {
        "n_examples": int(areas.size),
        "median_area": median_area,
        "section_px": section_px,
        "section_px_full": section_px_full,
        "minor_px": minor_px,
        "minor_px_full": minor_px_full,
        "median_area_full": median_area / (z * z),
        # area band for the DBSCAN/area post-filter
        "min_area": float(min_area_frac * median_area),
        "max_area": float(max_area_mult * median_area),
        # strategy
        "tile_px": tile_px,
        "overlap": overlap,
        "tiling_recommended": tiling_recommended,
        "resolve_px": float(resolve_px),
        "min_points": int(min_points),
        "points_on_section": points_on_section,
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
        cal["whole_image_section_px"] = float(apparent_single)
        # Encoder-native plan: SAM's tile should be 1024 px (1:1 into the encoder, no
        # upscaling). To make a section's thin axis clear resolve_px inside a 1024
        # tile, read the overview at N·1024, where N tiles per axis each magnify the
        # section by N. apparent_single is the thin axis on a whole 1024-px image and
        # is invariant to the loaded overview, so N is too:
        #     N = ceil(resolve_px / apparent_single)
        # N=1 → overview 1024, whole image. N>1 → overview N·1024, N×N tiles of 1024.
        if z and z > 0:
            full_long = overview_long_side / z          # source's true long side (px)
            n_need = max(1, int(np.ceil(resolve_px / max(apparent_single, 1e-6))))
            n_src = max(1, int(full_long // 1024))       # can't read finer than the source
            ov_cap = int(getattr(profile, "overview_cap_px", 0) or 0)
            n_host = (max(1, ov_cap // 1024) if ov_cap else n_need)   # host memory budget
            n = int(np.clip(n_need, 1, min(n_src, n_host)))
            cal["recommended_overview_long_side"] = int(n * 1024)
        else:
            # No pyramid to re-read at a finer level (e.g. PNG): keep the overview.
            cal["recommended_overview_long_side"] = int(overview_long_side)
        cal["resolution_ok"] = True

    cal["plan_summary"] = _plan_summary(cal, profile)
    return cal


def _plan_summary(cal, profile=None):
    """Plain-language detection plan for the GUI. Leads with the number that
    actually decides recall — how many of SAM's sampling points land on each
    section — then explains the tiling/grid choice that achieves it."""
    pps = int(cal.get("points_per_side", 0))
    pts = cal.get("points_on_section", 0.0)
    tgt = cal.get("min_points", 2)
    grid = f"{pps}×{pps} grid of sampling points"

    # Headline: the recall guarantee, in the user's terms.
    parts = [f"About {pts:.1f} of SAM's sampling points will land on each section "
             f"(aiming for {tgt} so none are missed)."]

    if cal.get("tiling_recommended"):
        ov = int(cal.get("recommended_overview_long_side", 0) or 0)
        tile = int(cal.get("tile_px", 1) or 1)
        n = max(2, round(ov / tile)) if ov else 2
        resolve = cal.get("resolve_px", 24.0)
        small = cal.get("whole_image_section_px", 0.0)     # thin axis to SAM if NOT tiled
        big = small * (ov / tile) if tile else small       # thin axis to SAM after tiling
        if small < resolve:
            # Resolution-driven: sections too small to outline on the whole image.
            parts.append(
                f"SAM shrinks whatever it processes to a 1024-px square; on the whole "
                f"image each section would be only ~{small:.0f} px across there — too "
                f"small to outline reliably. So the image is split into ~{n}×{n} tiles, "
                f"enlarging each section to ~{big:.0f} px. Every tile is sampled on a "
                f"{grid}.")
        else:
            # Memory-driven: sections resolve fine, but the image is too big to process
            # in one pass on this machine.
            parts.append(
                f"Each section is already big enough to outline (~{small:.0f} px to SAM), "
                f"but the whole image is too large to process at once on this machine, so "
                f"it's split into ~{n}×{n} tiles. Every tile is sampled on a {grid}.")
    else:
        parts.append(f"The whole image is processed in one pass, sampled on a {grid}.")

    if profile is not None:
        parts.append(f"Running on {getattr(profile, 'device_label', profile.device)} "
                     f"with the hiera-{cal.get('model_variant', '?')} model.")
    return " ".join(parts)


def summary(cal):
    """One-line human-readable calibration summary for the GUI log."""
    s = (f"calibrated from {cal['n_examples']} examples: section ~{cal['section_px']:.0f}px "
         f"(thin ~{cal.get('minor_px', 0):.0f}px, area {cal['median_area']:.0f}); "
         f"keep area {cal['min_area']:.0f}-{cal['max_area']:.0f}; "
         f"{'tiles' if cal.get('tiling_recommended') else 'single tile'} "
         f"tile_px={cal['tile_px']}, points_per_side={cal.get('points_per_side', '?')} "
         f"(~{cal.get('points_on_section', 0):.1f} pts/section), "
         f"pred_iou={cal.get('pred_iou_thresh', 0):.2f}")
    if "section_px_full" in cal:
        s += f"  [full-res section ~{cal['section_px_full']:.0f}px]"
    return s
