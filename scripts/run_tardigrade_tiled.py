#!/usr/bin/env python
"""Overnight tiled detection on the tardigrade wafer.

Reads the local STiM.czi at a higher overview resolution (real detail), runs the
tiled streaming detector across the whole wafer (empty tiles skipped), and writes:
  * an overlay PNG for quick visual inspection,
  * a STiM project JSON (full-res coords) so opening the CZI in the GUI restores
    the detections for inspection / correction / export.
"""

import json
import os
import time

import numpy as np
import cv2

from section_identification import czi_io
from section_identification.tiled_detect import tiled_detect

CZI = os.environ.get("STIM_CZI", "/Users/fredericoaraujo/Documents/tard_carbon_coat_001_STiM.czi")
CKPT = "checkpoint/sam2.1_hiera_base_plus.pt"
LONG_SIDE = int(os.environ.get("STIM_LONG", "6000"))
TILE_PX = int(os.environ.get("STIM_TILE", "512"))


def main():
    t0 = time.time()
    print(f"Reading overview at long side {LONG_SIDE}…", flush=True)
    arr, geom, meta = czi_io.read_czi_overview(CZI, target_long_side=LONG_SIDE)
    work = czi_io.to_rgb8(arr)
    print(f"Overview {work.shape} | zoom {meta['zoom']:.4g} | read {time.time()-t0:.0f}s",
          flush=True)

    def on_tile(k, n, box, new):
        if k % 10 == 0 or new:
            el = time.time() - t0
            print(f"tile {k}/{n} · +{len(new)} · {el:.0f}s elapsed", flush=True)

    res = tiled_detect(work, CKPT, tile_px=TILE_PX, overlap=0.25,
                       min_area=200.0, max_area=3500.0, points_per_side=24,
                       points_per_batch=12, pred_iou_thresh=0.78, darkness_ratio=0.85,
                       on_tile=on_tile)
    print(f"DETECTED {len(res)} sections in {time.time()-t0:.0f}s", flush=True)

    # overlay (downscaled view)
    fdir = f"{os.path.splitext(CZI)[0]}_files"
    os.makedirs(fdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(CZI))[0]
    vis = work.copy()
    for s in res:
        cv2.polylines(vis, [np.asarray(s["polygon"]).astype(np.int32)], True, (255, 0, 0), 2)
    cv2.imwrite(os.path.join(fdir, f"{base}_tiled_overlay.png"),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # project JSON (full-res coords) -> GUI restores on open
    def to_full(poly):
        p = np.asarray(poly, dtype=float)
        fx, fy = geom.ds_to_full(p[:, 0], p[:, 1])
        return [[float(a), float(b)] for a, b in zip(fx, fy)]

    proj = {"image": CZI, "sections": [to_full(s["polygon"]) for s in res], "fiducials": []}
    with open(os.path.join(fdir, f"{base}_stim_project.json"), "w") as f:
        json.dump(proj, f)
    print(f"Wrote overlay + project ({len(res)} sections) to {fdir}", flush=True)
    print(f"TOTAL {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
