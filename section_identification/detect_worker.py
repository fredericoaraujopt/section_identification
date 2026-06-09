"""Headless detection worker, spawned by the GUI as a separate process.

Running SAM in a child process (rather than the GUI's own thread) means the
napari window never freezes, the run can be cancelled cleanly by killing the
process, and all of SAM's memory is released when it exits — important on a
24 GB machine. The worker just runs ``automatic_identification`` which writes the
(small, RLE) mask cache; the GUI then loads that cache instantly.

Progress is coarse (SAM's automatic generator is opaque), so we emit milestone
lines on stdout that the GUI surfaces in its Log.
"""

import argparse
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--target-long-side", type=int, default=3072)
    ap.add_argument("--points-per-side", type=int, default=32)
    ap.add_argument("--points-per-batch", type=int, default=16)
    ap.add_argument("--pred-iou-thresh", type=float, default=0.8)
    ap.add_argument("--crop-n-layers", type=int, default=1)
    ap.add_argument("--min-area", type=int, default=20)
    a = ap.parse_args()

    t0 = time.time()
    print("STIM_PROGRESS: loading model and encoding image…", flush=True)
    from section_identification.section_detector import automatic_identification
    try:
        masks = automatic_identification(
            a.image, checkpoint=a.checkpoint, apply_filtering=False,
            points_per_side=a.points_per_side, points_per_batch=a.points_per_batch,
            pred_iou_thresh=a.pred_iou_thresh, crop_n_layers=a.crop_n_layers,
            min_mask_region_area=a.min_area, target_long_side=a.target_long_side)
        print(f"STIM_DONE: {len(masks)} masks cached in {time.time() - t0:.0f}s",
              flush=True)
    except Exception as e:
        import traceback
        print("STIM_ERROR: " + repr(e), flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
