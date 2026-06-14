"""Headless reorder worker: full-res SIFT per section, then serial-order recovery.

Spawned by the GUI as ``python -m section_identification.reorder_worker``. Reads
each section's **full-resolution** masked crop (blood vessels must resolve),
extracts SIFT descriptors (streaming SIFT progress), builds the inlier similarity
matrix (streaming pair progress), recovers the order, optionally caches the
similarity matrix to .npz, and emits ``STIM_REORDER_DONE`` with order + edges.

The full matrix is NOT streamed (too large); only the order, per-position
confidence, and edge list go on stdout. No napari/Qt here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import crops, czi_io, reorder, worker_protocol as wp


def _load_image(image_path, target):
    if czi_io.is_czi(image_path):
        arr, geom, _ = czi_io.read_czi_overview(image_path, target_long_side=target)
        return czi_io.to_rgb8(arr), geom
    from PIL import Image
    return np.array(Image.open(image_path).convert("RGB")), None


def _emit(tag, payload=None):
    print(wp.emit(tag, payload), flush=True)


def run(args):
    with open(args.sections) as f:
        sections = json.load(f)
    overview, geom = _load_image(args.image, args.target)
    ids = [s["id"] for s in sections]

    # 1) SIFT features per section (full-res) — the expensive, cacheable step.
    n = len(sections)
    features = []
    for k, s in enumerate(sections, start=1):
        try:
            gray, mask, _ = crops.read_section_crop(
                args.image, geom, s["polygon"], overview=overview, full_res=True)
            features.append(reorder.sift_features(gray, mask, nfeatures=args.nfeatures))
        except Exception as e:
            features.append((np.empty((0, 2), np.float32), None))
            _emit(wp.ERROR, {"where": f"sift:{s.get('id')}", "error": str(e)})
        _emit(wp.PROGRESS, {"done": k, "total": n, "phase": "sift"})

    # 2) pairwise similarity (streaming pair progress) + 3) order recovery.
    def prog(done, total):
        _emit(wp.REORDER_PROGRESS, {"done": done, "total": total, "phase": "match"})

    result = reorder.reorder(features, ids=ids, ratio=args.ratio,
                             method=args.method, progress=prog)
    sim = result.pop("similarity", None)
    if args.cache and sim is not None:
        try:
            os.makedirs(os.path.dirname(args.cache), exist_ok=True)
            np.savez_compressed(args.cache, similarity=sim, ids=np.array(ids, dtype=object))
            result["similarity_path"] = args.cache
        except Exception:
            pass
    _emit(wp.REORDER_DONE, result)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--sections", required=True, help="JSON: [{id, polygon(overview xy)}]")
    ap.add_argument("--target", type=int, default=4096)
    ap.add_argument("--nfeatures", type=int, default=0, help="0 = unlimited SIFT keypoints")
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--method", default="spectral+2opt")
    ap.add_argument("--cache", default=None, help="path to write similarity .npz")
    args = ap.parse_args(argv)
    try:
        run(args)
    except Exception as e:
        _emit(wp.ERROR, {"where": "reorder_worker", "error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
