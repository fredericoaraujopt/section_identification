"""Tiled, streaming SAM detection over an in-memory working image.

Running SAM once on a downscaled overview makes 200-400 px sections
sub-resolution. Instead we read the working image **once** (the overview, at a
resolution where sections are a few tens of px) and walk it in tiles:

* **Effective zoom knob** — SAM resizes every tile to 1024 px, so a smaller
  ``tile_px`` upscales the tile and makes sections appear bigger to SAM
  (``effective_upscale = 1024 / tile_px``). This is a cleaner recall knob than
  ``crop_n_layers``.
* **Bounded memory** — only one tile is segmented at a time; tiles are numpy
  slices of an image already in RAM (no repeated CZI reads, which are slow).
* **Sparse-wafer speed** — a cheap dark-object gate skips empty tiles, so SAM
  only runs where there is something to segment.
* **Streaming** — a per-tile callback drives the GUI's live view / progress.

All coordinates are in the **input image (overview) frame**; the caller maps to
full-resolution via its :class:`CziGeometry` on export, exactly like the rest of
the GUI.
"""

from __future__ import annotations

import numpy as np


def tile_boxes(width, height, tile_px, overlap=0.12):
    """``(x0, y0, w, h)`` tiles covering a WxH image with fractional overlap."""
    step = max(1, int(tile_px * (1.0 - overlap)))
    boxes = []
    y = 0
    while y < height:
        x = 0
        while x < width:
            w = min(tile_px, width - x)
            h = min(tile_px, height - y)
            boxes.append((x, y, w, h))
            if x + tile_px >= width:
                break
            x += step
        if y + tile_px >= height:
            break
        y += step
    return boxes


def tile_has_objects(gray, min_area_px, max_area_frac=0.9):
    """Cheap gate: is there a dark object in [min_area_px, max_area_frac*tile]?"""
    import cv2

    if gray.size == 0:
        return False
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    nlab, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    tile_area = gray.shape[0] * gray.shape[1]
    for i in range(1, nlab):
        a = stats[i, cv2.CC_STAT_AREA]
        if min_area_px <= a <= max_area_frac * tile_area:
            return True
    return False


def _centroid(poly):
    p = np.asarray(poly, dtype=float).reshape(-1, 2)
    return p[:, 0].mean(), p[:, 1].mean()


def plan_tiles(width, height, tile_px, overlap=0.12):
    """Return the tile boxes the detector WILL use — for GUI preview overlays."""
    return tile_boxes(width, height, tile_px, overlap)


def tiled_detect(image, checkpoint, *, model_cfg=None, device=None,
                 tile_px=512, overlap=0.12, min_area=200.0, max_area=1e12,
                 points_per_side=24, points_per_batch=16, pred_iou_thresh=0.8,
                 darkness_ratio=0.85, on_tile=None, dedup_dist_frac=0.5):
    """Detect sections by tiling an in-memory RGB image; return section dicts.

    Each result: ``{"polygon": Nx2 (x,y) in image coords, "area": float,
    "bbox": (x0,y0,x1,y1), "tile": idx}``. Areas/coords are in the input image
    (overview) frame.

    ``on_tile(tile_idx, n_tiles, box, new_sections)`` is called after every tile
    (``box`` = the tile's ``(x0,y0,w,h)``) so a GUI can show the current tile and
    stream freshly-confirmed sections.
    """
    from section_identification.section_detector import build_mask_generator
    from section_identification.device import get_device, autocast_ctx
    from section_identification.export import mask_to_polygon, decode_segmentation

    device = get_device(device)
    params = {
        "points_per_side": points_per_side, "points_per_batch": points_per_batch,
        "pred_iou_thresh": pred_iou_thresh, "stability_score_thresh": 0.92,
        "box_nms_thresh": 0.7, "crop_n_layers": 0,
        "min_mask_region_area": int(max(4, min_area)), "output_mode": "binary_mask",
    }
    amg = build_mask_generator(checkpoint, model_cfg, device, params)

    H, W = image.shape[:2]
    boxes = tile_boxes(W, H, tile_px, overlap)
    n = len(boxes)
    results = []
    for idx, (tx, ty, tw, th) in enumerate(boxes):
        tile = image[ty:ty + th, tx:tx + tw]
        gray = tile[:, :, 0] if tile.ndim == 3 else tile
        new_sections = []
        if tile_has_objects(gray, max(4.0, min_area)):
            with autocast_ctx(device):
                masks = amg.generate(np.ascontiguousarray(tile))
            # Largest first so centroid-dedup keeps the whole section, not a
            # sub-part nested inside it.
            masks = sorted(masks, key=lambda mm: -float(mm["area"]))
            tile_mean = float(gray.mean())
            for m in masks:
                area = float(m["area"])
                if area < min_area or area > max_area:
                    continue
                dec = decode_segmentation(m["segmentation"])
                inside = gray[dec > 0]
                # Sections are dark on a bright background; reject masks whose
                # interior isn't clearly darker than the tile (e.g. the montage
                # stitching-grid cells, which match the background brightness).
                if inside.size == 0 or inside.mean() >= darkness_ratio * tile_mean:
                    continue
                poly = mask_to_polygon(dec)
                if poly is None or len(poly) < 3:
                    continue
                pf = np.asarray(poly, dtype=float)
                pf[:, 0] += tx
                pf[:, 1] += ty
                cx, cy = _centroid(pf)
                r = dedup_dist_frac * float(np.sqrt(max(area, 1.0)))
                if any(abs(cx - ex) < r and abs(cy - ey) < r
                       for ex, ey in (_centroid(s["polygon"]) for s in results)):
                    continue
                sec = {"polygon": pf, "area": area, "tile": idx,
                       "bbox": (float(pf[:, 0].min()), float(pf[:, 1].min()),
                                float(pf[:, 0].max()), float(pf[:, 1].max()))}
                results.append(sec)
                new_sections.append(sec)
        if on_tile is not None:
            on_tile(idx + 1, n, (tx, ty, tw, th), new_sections)
    return results
