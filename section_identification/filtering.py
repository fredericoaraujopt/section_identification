"""Filter SAM masks down to real tissue sections.

Two stages:
  1. **Shape gating** (new) — drop masks that cannot be sections regardless of
     area: the whole-field background blob, slivers, and very non-convex /
     non-compact shapes. Uses solidity, extent and aspect ratio.
  2. **Area DBSCAN** (original) — sections on a wafer are size-consistent, so we
     cluster surviving masks by area and keep the dominant cluster.

Shape gating makes the area clustering far more robust: it removes the outliers
(background, debris, support film, grid bars) that previously skewed DBSCAN.
"""

import numpy as np
from sklearn.cluster import DBSCAN


# --------------------------------------------------------------------------- #
# Stage 1: shape gating
# --------------------------------------------------------------------------- #
def mask_shape_stats(mask):
    """Compute area, extent, solidity, aspect ratio and border-touch for a mask."""
    import cv2

    seg = np.squeeze(mask["segmentation"]).astype(np.uint8)
    h, w = seg.shape
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area <= 0:
        return None
    x, y, bw, bh = cv2.boundingRect(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull)) or area
    extent = area / float(bw * bh) if bw * bh else 0.0
    solidity = area / hull_area if hull_area else 0.0
    aspect = max(bw, bh) / float(min(bw, bh)) if min(bw, bh) else 999.0
    touches_border = (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1)
    return {
        "area": area, "extent": extent, "solidity": solidity,
        "aspect": aspect, "touches_border": touches_border,
        "area_frac": area / float(h * w),
    }


def shape_prefilter(masks, min_solidity=0.85, min_extent=0.30, max_aspect=6.0,
                    max_area_frac=0.05, drop_border=False):
    """Keep only masks whose shape is plausible for a section.

    Drops: the near-full-frame background mask (``area_frac > max_area_frac``),
    slivers (low extent / high aspect), very concave blobs (low solidity), and
    optionally any mask touching the image border.
    """
    kept = []
    for m in masks:
        s = mask_shape_stats(m)
        if s is None:
            continue
        if s["area_frac"] > max_area_frac:
            continue
        if s["solidity"] < min_solidity:
            continue
        if s["extent"] < min_extent:
            continue
        if s["aspect"] > max_aspect:
            continue
        if drop_border and s["touches_border"]:
            continue
        kept.append(m)
    return kept


# --------------------------------------------------------------------------- #
# Stage 2: area DBSCAN (original behaviour, run on shape-gated masks)
# --------------------------------------------------------------------------- #
def _area_dbscan(sorted_masks, eps_values, min_samples_values):
    mask_areas = np.array([mask["area"] for mask in sorted_masks]).reshape(-1, 1)
    cluster_frequency = {}
    for eps in eps_values:
        for min_samples in min_samples_values:
            clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(mask_areas)
            largest_label, largest_size = None, -1
            for c in np.unique(clusters):
                if c == -1:
                    continue
                size_c = int(np.sum(clusters == c))
                if size_c > largest_size:
                    largest_size, largest_label = size_c, c
            if largest_label is None:
                continue
            key = frozenset(i for i, cl in enumerate(clusters) if cl == largest_label)
            entry = cluster_frequency.setdefault(key, {"count": 0, "params": []})
            entry["count"] += 1
            entry["params"].append((eps, min_samples))

    if not cluster_frequency:
        return [], (None, None)

    best_key = max(cluster_frequency, key=lambda k: cluster_frequency[k]["count"])
    chosen_params = cluster_frequency[best_key]["params"][0]
    return [sorted_masks[i] for i in best_key], chosen_params


def area_band_filter(masks, lo_ratio=0.2, hi_ratio=5.0):
    """Keep masks whose area is within ``[lo_ratio, hi_ratio] x median area``.

    After shape gating, the majority of masks are real sections, so the median
    area is a good estimate of section size. Large debris (and merged/garbage
    blobs) sit well above ``hi_ratio x median`` and are removed; this is the
    main debris killer and is robust to the DBSCAN parameters.
    """
    if not masks:
        return masks
    areas = np.array([m["area"] for m in masks], dtype=float)
    med = float(np.median(areas))
    lo, hi = med * lo_ratio, med * hi_ratio
    kept = [m for m, a in zip(masks, areas) if lo <= a <= hi]
    print(f"Area band [{lo:.0f}, {hi:.0f}] (median {med:.0f}): "
          f"kept {len(kept)}/{len(masks)} masks.")
    return kept or masks


def filtering(sorted_masks, eps_values, min_samples_values, apply_shape_gate=True,
              shape_kwargs=None, area_band=(0.2, 5.0)):
    """Filter masks to the dominant population of section-shaped, section-sized masks.

    Stages: shape gate -> adaptive area band (debris killer) -> area DBSCAN.
    Backward compatible with the original ``filtering(masks, eps, min_samples)``.
    """
    masks = sorted_masks
    if apply_shape_gate:
        before = len(masks)
        masks = shape_prefilter(masks, **(shape_kwargs or {}))
        print(f"Shape gate: kept {len(masks)}/{before} masks.")
    if not masks:
        return [], (None, None)

    if area_band:
        masks = area_band_filter(masks, *area_band)

    chosen, chosen_params = _area_dbscan(masks, eps_values, min_samples_values)
    print(f"Area DBSCAN: kept {len(chosen)}/{len(masks)} masks "
          f"(eps={chosen_params[0]}, min_samples={chosen_params[1]}).")
    return chosen, chosen_params
