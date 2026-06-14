"""Headless QC worker: score each section and stream STIM_QC results.

Spawned by the GUI as ``python -m section_identification.qc_worker --image … --sections …``
(same QProcess+stdout-streaming pattern as detect_worker). Reads the image once
to recover geometry, then for each section reads a (downscaled) masked crop and
runs the wafer_qc detectors, emitting one ``STIM_QC`` line per section so the GUI
colours the wafer live, and a final ``STIM_QC_DONE``.

No napari/Qt here — runnable and testable standalone.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import crops, czi_io, wafer_qc, worker_protocol as wp


def _load_image(image_path, target):
    """Return (overview_rgb_or_None, geom). For CZI we recover geom by reading
    the overview at the same long side the GUI used; for PNG geom is None and
    the overview array is loaded for slicing."""
    if czi_io.is_czi(image_path):
        arr, geom, _ = czi_io.read_czi_overview(image_path, target_long_side=target)
        return czi_io.to_rgb8(arr), geom
    from PIL import Image
    import numpy as np
    return np.array(Image.open(image_path).convert("RGB")), None


def _emit(tag, payload=None):
    print(wp.emit(tag, payload), flush=True)


def run(args):
    with open(args.sections) as f:
        sections = json.load(f)                     # [{"id":.., "polygon":[[x,y],..]}, ...]
    refs = json.loads(args.refs) if args.refs else None
    overview, geom = _load_image(args.image, args.target)

    n = len(sections)
    _emit(wp.PROGRESS, {"done": 0, "total": n, "phase": "qc"})
    for k, s in enumerate(sections, start=1):
        try:
            gray, mask, _ = crops.read_section_crop(
                args.image, geom, s["polygon"], overview=overview,
                full_res=False, target_long_side=args.long_side)
            res = wafer_qc.score_section(gray, mask, refs)
            _emit(wp.QC, {"section_id": s["id"], **res})
        except Exception as e:                       # never let one section kill the run
            _emit(wp.QC, {"section_id": s.get("id"), "error": str(e)})
        _emit(wp.PROGRESS, {"done": k, "total": n, "phase": "qc"})
    _emit(wp.QC_DONE, {"n": n})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--sections", required=True, help="JSON: [{id, polygon(overview xy)}]")
    ap.add_argument("--target", type=int, default=4096, help="overview long side used by the GUI")
    ap.add_argument("--long-side", type=int, default=640, help="QC working crop long side")
    ap.add_argument("--refs", default=None, help="JSON of QC reference overrides")
    args = ap.parse_args(argv)
    try:
        run(args)
    except Exception as e:
        _emit(wp.ERROR, {"where": "qc_worker", "error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
