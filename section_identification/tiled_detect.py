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
* **Bounded work** — SAM runs once per tile; results are filtered by the section
  size band + de-duplicated across tile overlaps.
* **Streaming** — a per-tile callback drives the GUI's live view / progress.

All coordinates are in the **input image (overview) frame**; the caller maps to
full-resolution via its :class:`CziGeometry` on export, exactly like the rest of
the GUI.
"""

from __future__ import annotations

import numpy as np


def _release_torch_cache(device):
    """Free the framework's cached GPU memory between tiles.

    SAM2's AMG frees its own Python references after each tile, but PyTorch's
    caching allocator keeps the freed blocks resident — so on Apple MPS (unified
    memory) the process RSS climbs tile-by-tile and the machine starts swapping
    even though each individual tile fits. Emptying the cache returns it to the
    OS. No effect on results; a few ms per tile.
    """
    import gc
    gc.collect()
    try:
        import torch
        dev = str(getattr(device, "type", device))
        if dev == "cuda":
            torch.cuda.empty_cache()
        elif dev == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except Exception:
        pass


def tile_boxes(width, height, tile_px, overlap=0.12):
    """``(x0, y0, w, h)`` tiles that split the image into EQUAL cells (no sliver
    edge tiles), each grown by ``overlap`` so a section straddling a boundary
    still sits whole inside a neighbour. The number of cells per axis is the
    count of ~``tile_px`` cells nearest to the axis length."""
    tile_px = max(1, int(tile_px))

    def _spans(length):
        n = max(1, int(round(length / tile_px)))
        base = length / n
        margin = overlap * base
        spans = []
        for i in range(n):
            a = max(0.0, i * base - margin / 2.0)
            b = min(float(length), (i + 1) * base + margin / 2.0)
            spans.append((a, b - a))
        return spans

    xs, ys = _spans(width), _spans(height)
    return [(int(round(x0)), int(round(y0)), int(round(w)), int(round(h)))
            for (y0, h) in ys for (x0, w) in xs]


def _centroid(poly):
    p = np.asarray(poly, dtype=float).reshape(-1, 2)
    return p[:, 0].mean(), p[:, 1].mean()


def plan_tiles(width, height, tile_px, overlap=0.12):
    """Return the tile boxes the detector WILL use — for GUI preview overlays."""
    return tile_boxes(width, height, tile_px, overlap)


def tiled_detect(image, checkpoint, *, model_cfg=None, device=None,
                 tile_px=512, overlap=0.12, min_area=200.0, max_area=1e12,
                 points_per_side=24, points_per_batch=16, pred_iou_thresh=0.8,
                 stability_score_thresh=0.92, stability_score_offset=1.0,
                 box_nms_thresh=0.7, crop_n_layers=0, crop_nms_thresh=0.7,
                 crop_overlap_ratio=512 / 1500, crop_n_points_downscale_factor=1,
                 min_mask_region_area=None, use_m2m=False, multimask_output=True,
                 on_tile=None, on_tile_start=None, dedup_dist_frac=0.5):
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
    from section_identification import host_profile

    device = get_device(device)
    # Cap points_per_batch to a memory-safe value for this tile size + host (no
    # effect on results; prevents OOM/thrash on weak machines). multimask_output
    # decides whether SAM emits 3 masks/point (more memory) or 1, so the budget
    # math must know which.
    masks_per_point = 3 if multimask_output else 1
    budget = host_profile.detect_profile(str(device)).mem_budget_bytes
    # tile_boxes grows each interior tile by the overlap margin, so the real tile
    # SAM upsamples is up to ~tile_px*(1+overlap) per side (clamped to the image).
    # Budget against that larger size, not tile_px, so the cap can't run generous.
    import math
    H0, W0 = image.shape[:2]
    eff = int(math.ceil(tile_px * (1.0 + max(0.0, overlap))))
    eff_h, eff_w = min(H0, eff), min(W0, eff)
    safe_ppb = host_profile.safe_points_per_batch(
        budget, eff_h, eff_w, points_per_batch, masks_per_point=masks_per_point)
    mmra = int(min_mask_region_area) if min_mask_region_area is not None \
        else int(max(4, min_area))
    params = {
        "points_per_side": points_per_side, "points_per_batch": safe_ppb,
        "pred_iou_thresh": pred_iou_thresh,
        "stability_score_thresh": stability_score_thresh,
        "stability_score_offset": stability_score_offset,
        "box_nms_thresh": box_nms_thresh, "crop_n_layers": crop_n_layers,
        "crop_nms_thresh": crop_nms_thresh, "crop_overlap_ratio": crop_overlap_ratio,
        "crop_n_points_downscale_factor": crop_n_points_downscale_factor,
        "use_m2m": use_m2m, "multimask_output": multimask_output,
        # coco_rle keeps per-tile mask memory tiny (KB/mask) so even a single
        # whole-image tile can't recreate the multi-GB binary-mask cache.
        "min_mask_region_area": mmra, "output_mode": "coco_rle",
    }
    print(f"STIM_PROGRESS: SAM points_per_batch={safe_ppb} (host budget) on {device}", flush=True)
    amg = build_mask_generator(checkpoint, model_cfg, device, params)

    H, W = image.shape[:2]
    boxes = tile_boxes(W, H, tile_px, overlap)
    n = len(boxes)
    results = []
    edge = 2  # px tolerance for "touches the tile edge"
    for idx, (tx, ty, tw, th) in enumerate(boxes):
        if on_tile_start is not None:
            on_tile_start(idx + 1, n, (tx, ty, tw, th))   # show the tile BEFORE work
        tile = image[ty:ty + th, tx:tx + tw]
        new_sections = []
        with autocast_ctx(device):
            masks = amg.generate(np.ascontiguousarray(tile))
        # Largest first so centroid-dedup keeps the whole section, not a
        # sub-part nested inside it.
        masks = sorted(masks, key=lambda mm: -float(mm["area"]))
        for m in masks:
            area = float(m["area"])
            # NB: no size band here — `results` is the RAW, unfiltered SAM output
            # (small + large debris included, for downstream wafer QC). The size
            # band + DBSCAN are applied later (GUI _finalize_tiled) to produce the
            # 'Sections' layer. Only tiling bookkeeping (edge-reject + dedup) runs.
            bx, by, bw, bh = m["bbox"]  # tile-local x,y,w,h
            # Own each section by the tile that fully contains it: drop masks
            # touching a tile edge that is NOT an image border (with enough
            # overlap every section is fully inside some tile). Prevents sections
            # being cut/duplicated at tile boundaries.
            if ((bx <= edge and tx > 0) or (by <= edge and ty > 0) or
                    (bx + bw >= tw - edge and tx + tw < W) or
                    (by + bh >= th - edge and ty + th < H)):
                continue
            dec = decode_segmentation(m["segmentation"])
            poly = mask_to_polygon(dec)
            if poly is None or len(poly) < 3:
                continue
            pf = np.asarray(poly, dtype=float)
            pf[:, 0] += tx
            pf[:, 1] += ty
            cx, cy = _centroid(pf)
            r = dedup_dist_frac * float(np.sqrt(max(area, 1.0)))
            if any(abs(cx - ex) < r and abs(cy - ey) < r           # cross-tile dedup
                   for ex, ey in (_centroid(s["polygon"]) for s in results)):
                continue
            sec = {"polygon": pf, "area": area, "tile": idx,
                   "bbox": (float(pf[:, 0].min()), float(pf[:, 1].min()),
                            float(pf[:, 0].max()), float(pf[:, 1].max()))}
            results.append(sec)
            new_sections.append(sec)
        if on_tile is not None:
            on_tile(idx + 1, n, (tx, ty, tw, th), new_sections)
        del masks
        _release_torch_cache(device)
    return results
