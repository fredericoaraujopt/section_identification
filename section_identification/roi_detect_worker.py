"""Headless per-section automatic ROI worker.

Spawned by the GUI as
``python -m section_identification.roi_detect_worker --spec spec.json``
(the same QProcess + STIM_* streaming pattern as detect_worker / qc_worker).

For a wafer whose ROI is visually distinct from the resin, SAM finds it inside
each section: the automatic mask generator runs on a per-section RGB crop
(``points_per_side`` grid, memory-guarded ``points_per_batch``, the same quality
gates as the section detector), the resulting masks are mapped back to overview
pixels and scored against the drawn template (area + shape + containment), and
the single best template-compatible mask is streamed as that section's ROI.

Emits: ``STIM_ROISTART {k,n,id,bbox}`` before each section, ``STIM_ROI
{k,n,id,polygon|null,score}`` after, ``STIM_PROGRESS`` throughout, and a final
``STIM_ROI_DONE {n,hit}``. No napari/Qt — runnable and testable standalone.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import (crops, czi_io, host_profile, roi as roi_mod,
               worker_protocol as wp)


def _emit(tag, payload=None):
    print(wp.emit(tag, payload), flush=True)


def _load_overview(image_path, target):
    if czi_io.is_czi(image_path):
        arr, geom, _ = czi_io.read_czi_overview(image_path, target_long_side=target)
        return czi_io.to_rgb8(arr), geom
    from PIL import Image
    return np.array(Image.open(image_path).convert("RGB")), None


def _mask_to_overview_polygon(seg, origin_scale):
    """Decode one AMG mask → largest external contour → overview xy via
    (ox0,oy0,scale). Uses raw ``findContours`` (CHAIN_APPROX_SIMPLE, no
    Douglas-Peucker) so SAM's contour coordinates are preserved exactly."""
    import cv2
    from .export import decode_segmentation
    m = decode_segmentation(seg)
    if m is None:
        return None
    m = np.asarray(m).astype(np.uint8)
    if m.ndim != 2 or m.sum() == 0:
        return None
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)     # crop px
    if len(c) < 3:
        return None
    ox0, oy0, scale = origin_scale
    scale = scale or 1.0
    return [[float(ox0 + x / scale), float(oy0 + y / scale)] for x, y in c]


def run(spec):
    image = spec["image"]
    target = int(spec.get("target", 4096))
    checkpoint = spec["checkpoint"]
    device_pref = spec.get("device") or None
    params = spec["params"]
    tmpl = np.asarray(spec["template"], float).reshape(-1, 2)
    sections = spec["sections"]                     # [{"id":.., "polygon":[[x,y],..]}]
    contour_source = params.get("contour_source", "mask")

    from .device import autocast_ctx, get_device
    from .section_detector import build_mask_generator

    device = get_device(device_pref)
    overview, geom = _load_overview(image, target)

    roi_area = roi_mod._shoelace_area(tmpl)
    band = (roi_area * float(params.get("min_area_frac", 0.5)),
            roi_area * float(params.get("max_area_mult", 2.0)))
    floor = float(params.get("score_floor", 0.35))
    crop_long = int(params.get("crop_long", 1024))
    pps = int(params.get("points_per_side", 9))

    prof = host_profile.detect_profile(str(device))
    ppb = host_profile.safe_points_per_batch(prof.mem_budget_bytes, crop_long,
                                             crop_long, pps * pps, masks_per_point=3)
    amg_params = {
        "points_per_side": pps,
        "points_per_batch": int(ppb),
        "pred_iou_thresh": float(params.get("pred_iou_thresh", 0.8)),
        "stability_score_thresh": float(params.get("stability_score_thresh", 0.9)),
        "stability_score_offset": 1.0,
        "box_nms_thresh": 0.7,
        "crop_n_layers": 0,
        "min_mask_region_area": int(max(4, params.get("min_mask_region_area",
                                                      round(0.05 * roi_area)))),
        "use_m2m": False,
        "multimask_output": True,
        "output_mode": "coco_rle",
    }
    _emit(wp.PROGRESS, {"done": 0, "total": len(sections),
                        "phase": f"SAM points/side={pps}, points/batch={ppb} on {device}"})
    amg = build_mask_generator(checkpoint, None, device, amg_params)

    n, hit = len(sections), 0
    for k, s in enumerate(sections, start=1):
        sid, poly = s.get("id"), s.get("polygon")
        p = np.asarray(poly, float).reshape(-1, 2)
        bbox = [float(p[:, 0].min()), float(p[:, 1].min()),
                float(p[:, 0].max()), float(p[:, 1].max())]
        _emit("ROISTART", {"k": k, "n": n, "id": sid, "bbox": bbox})
        best_poly, best_score = None, 0.0
        try:
            got = crops.read_section_rgb_crop(
                image, geom, poly, overview=overview, target_long_side=crop_long,
                margin_frac=float(params.get("crop_margin", 0.15)))
            if got is not None:
                rgb, origin_scale = got
                with autocast_ctx(device):
                    masks = amg.generate(np.ascontiguousarray(rgb))
                cands = []
                for m in masks:
                    mp = _mask_to_overview_polygon(m.get("segmentation"), origin_scale)
                    if mp is not None:
                        cands.append((mp, float(m.get("predicted_iou", 0.0))))
                best_poly, best_score = roi_mod.choose_best_roi(
                    cands, tmpl, poly, band, floor=floor)
                if best_poly is not None and contour_source == "template":
                    best_poly = roi_mod.fit_template_to_mask(tmpl, best_poly)
        except Exception as e:
            _emit(wp.PROGRESS, {"done": k, "total": n, "phase": f"section {sid}: {e}"})
        if best_poly is not None:
            hit += 1
        _emit("ROI", {"k": k, "n": n, "id": sid,
                      "polygon": best_poly, "score": round(float(best_score), 3)})
        _emit(wp.PROGRESS, {"done": k, "total": n, "phase": "roi"})
    _emit("ROI_DONE", {"n": n, "hit": hit})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="path to a JSON spec file")
    a = ap.parse_args(argv)
    try:
        with open(a.spec) as f:
            spec = json.load(f)
        run(spec)
    except Exception as e:
        import traceback
        _emit(wp.ERROR, {"error": str(e), "trace": traceback.format_exc()})
        sys.exit(1)


if __name__ == "__main__":
    main()
