"""Headless detection worker, spawned by the GUI as a separate process.

Running SAM in a child process keeps the napari window responsive, lets the run
be cancelled by killing the process, and releases all of SAM's memory on exit.

Two modes:
  * ``whole``  — the classic single-pass SAM automatic generator (writes the RLE
    mask cache; the GUI loads it back).
  * ``tiled``  — the tiled streaming detector: emits the tile grid up front
    (``STIM_TILES``) and, after each tile, the freshly-confirmed sections
    (``STIM_TILE``) as JSON in the input-image (overview) frame, so the GUI can
    draw the current tile and pop sections up live.

Progress/markers on stdout: ``STIM_TILES``, ``STIM_TILE``, ``STIM_PROGRESS``,
``STIM_DONE``, ``STIM_ERROR``.
"""

import argparse
import json
import sys
import time


def _run_whole(a):
    print("STIM_PROGRESS: loading model and encoding image…", flush=True)
    from section_identification.section_detector import automatic_identification
    masks = automatic_identification(
        a.image, checkpoint=a.checkpoint, apply_filtering=False,
        points_per_side=a.points_per_side, points_per_batch=a.points_per_batch,
        pred_iou_thresh=a.pred_iou_thresh, crop_n_layers=a.crop_n_layers,
        min_mask_region_area=int(a.min_area), target_long_side=a.target_long_side)
    print(f"STIM_DONE: {len(masks)} masks cached", flush=True)


def _run_tiled(a):
    import numpy as np
    from section_identification import czi_io
    from section_identification.tiled_detect import tiled_detect, plan_tiles

    print("STIM_PROGRESS: reading working image…", flush=True)
    if czi_io.is_czi(a.image):
        arr, _geom, _meta = czi_io.read_czi_overview(a.image, a.target_long_side)
        work = czi_io.to_rgb8(arr)
    else:
        from PIL import Image
        work = np.array(Image.open(a.image).convert("RGB"))

    boxes = plan_tiles(work.shape[1], work.shape[0], a.tile_px, a.overlap)
    print("STIM_TILES " + json.dumps([list(map(int, b)) for b in boxes]), flush=True)

    def on_tile(k, n, box, new):
        secs = [{"poly": np.asarray(s["polygon"]).round(1).tolist(),
                 "area": round(float(s["area"]), 1)} for s in new]
        print("STIM_TILE " + json.dumps(
            {"k": k, "n": n, "box": list(map(int, box)), "sections": secs}),
            flush=True)

    res = tiled_detect(
        work, a.checkpoint, tile_px=a.tile_px, overlap=a.overlap,
        min_area=a.min_area, max_area=a.max_area,
        points_per_side=a.points_per_side, points_per_batch=a.points_per_batch,
        pred_iou_thresh=a.pred_iou_thresh, darkness_ratio=a.darkness_ratio,
        on_tile=on_tile)
    print(f"STIM_DONE: {len(res)} sections", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", default="whole", choices=["whole", "tiled"])
    ap.add_argument("--target-long-side", type=int, default=3072)
    ap.add_argument("--points-per-side", type=int, default=32)
    ap.add_argument("--points-per-batch", type=int, default=16)
    ap.add_argument("--pred-iou-thresh", type=float, default=0.8)
    ap.add_argument("--crop-n-layers", type=int, default=1)
    ap.add_argument("--min-area", type=float, default=20.0)
    # tiled-mode extras
    ap.add_argument("--tile-px", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.12)
    ap.add_argument("--max-area", type=float, default=1e12)
    ap.add_argument("--darkness-ratio", type=float, default=0.85)
    a = ap.parse_args()

    t0 = time.time()
    try:
        (_run_tiled if a.mode == "tiled" else _run_whole)(a)
        print(f"STIM_PROGRESS: finished in {time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        import traceback
        print("STIM_ERROR: " + repr(e), flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
