"""Headless detection worker, spawned by the GUI as a separate process.

Running SAM in a child process keeps the napari window responsive, lets the run
be cancelled by killing the process, and releases all of SAM's memory on exit.

ONE streaming engine: the working image is walked in tiles (often a single tile
= whole image, when sections are big enough), emitting the tile grid up front
(``STIM_TILES``) and, after each tile, the freshly-confirmed sections
(``STIM_TILE``) as JSON in the input-image (overview) frame — so the GUI shows
the current tile and pops sections up live in *every* run (SAM's whole-image
generator is a black box and can't stream, hence we always tile).

Progress/markers on stdout: ``STIM_TILES``, ``STIM_TILE``, ``STIM_PROGRESS``,
``STIM_DONE``, ``STIM_ERROR``.
"""

import argparse
import json
import sys
import threading
import time


class _ResourceMonitor(threading.Thread):
    """Sample this worker's memory/CPU load while SAM runs and print it.

    The automatic detector is the heaviest thing STiM does to the user's
    machine; this makes that cost visible. A daemon thread samples every
    ``interval`` seconds (so it can't block detection or outlive it) and prints
    ``STIM_PROGRESS`` lines — process RSS, its share of system RAM, and CPU% —
    plus, on CUDA, the torch peak VRAM. A final summary reports the peak RSS.

    Degrades gracefully: if ``psutil`` isn't importable we say so once and stop,
    rather than failing the run (psutil is optional across STiM).
    """

    def __init__(self, device="", interval=3.0):
        super().__init__(daemon=True)
        self.device = (device or "").lower()
        self.interval = float(interval)
        self._stop = threading.Event()
        self.peak_rss = 0
        try:
            import psutil
            self._psutil = psutil
            self._proc = psutil.Process()
            self._proc.cpu_percent(None)          # prime the % baseline
            self._cores = psutil.cpu_count() or 1
        except Exception:
            self._psutil = None

    def _cuda_peak_gb(self):
        if "cuda" not in self.device:
            return None
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated() / (1024 ** 3)
        except Exception:
            pass
        return None

    def _sample(self, final=False):
        GB = 1024 ** 3
        try:
            rss = self._proc.memory_info().rss
            self.peak_rss = max(self.peak_rss, rss)
            vm = self._psutil.virtual_memory()
            cpu = self._proc.cpu_percent(None)     # sum across cores; may be >100
            msg = (f"RSS {rss / GB:.2f} GB "
                   f"({rss / vm.total * 100:.0f}% of {vm.total / GB:.0f} GB) · "
                   f"sys RAM {vm.percent:.0f}% used, {vm.available / GB:.1f} GB free · "
                   f"CPU {cpu:.0f}% (of {self._cores * 100}%)")
            vram = self._cuda_peak_gb()
            if vram is not None:
                msg += f" · CUDA peak {vram:.2f} GB"
            tag = "peak load" if final else "load"
            print(f"STIM_PROGRESS: [{tag}] {msg}", flush=True)
        except Exception:
            pass

    def run(self):
        if self._psutil is None:
            print("STIM_PROGRESS: [load] resource monitor unavailable "
                  "(install psutil to see RAM/CPU load).", flush=True)
            return
        while not self._stop.wait(self.interval):
            self._sample()

    def stop_and_report(self):
        self._stop.set()
        if self._psutil is None:
            return
        GB = 1024 ** 3
        # one last live sample, then the run's peak RSS
        self._sample()
        try:
            print(f"STIM_PROGRESS: [peak load] worker peak RSS "
                  f"{self.peak_rss / GB:.2f} GB.", flush=True)
        except Exception:
            pass


def _run_stream(a):
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
    # The working resolution (overview px) is what actually drives cost/detail —
    # echo it so the run log reflects the knob the user changed.
    print(f"STIM_PROGRESS: working image {work.shape[1]}x{work.shape[0]}px "
          f"(overview long {max(work.shape[:2])}px).", flush=True)

    # tile_px <= 0 (or >= the image) means "one tile = whole image".
    tile_px = a.tile_px if a.tile_px and a.tile_px > 0 else max(work.shape[:2])
    boxes = plan_tiles(work.shape[1], work.shape[0], tile_px, a.overlap)
    # A tile can't exceed the image, so report the EFFECTIVE tile size (and thus the
    # true ×upscale SAM applies), not the requested cap.
    eff_tile = min(int(tile_px), max(work.shape[:2]))
    print("STIM_TILES " + json.dumps([list(map(int, b)) for b in boxes]), flush=True)
    print(f"STIM_PROGRESS: {len(boxes)} tile(s) of {eff_tile}px "
          f"(SAM sees a section ×{1024.0/eff_tile:.2f}).", flush=True)

    def on_tile_start(k, n, box):
        print("STIM_TILESTART " + json.dumps(
            {"k": k, "n": n, "box": list(map(int, box))}), flush=True)

    def on_tile(k, n, box, new):
        secs = [{"poly": np.asarray(s["polygon"]).round(1).tolist(),
                 "area": round(float(s["area"]), 1)} for s in new]
        print("STIM_TILE " + json.dumps(
            {"k": k, "n": n, "box": list(map(int, box)), "sections": secs}),
            flush=True)

    res = tiled_detect(
        work, a.checkpoint, device=a.device or None, tile_px=tile_px, overlap=a.overlap,
        min_area=a.min_area, max_area=a.max_area,
        points_per_side=a.points_per_side, points_per_batch=a.points_per_batch,
        pred_iou_thresh=a.pred_iou_thresh,
        stability_score_thresh=a.stability_score_thresh,
        stability_score_offset=a.stability_score_offset,
        box_nms_thresh=a.box_nms_thresh, crop_n_layers=a.crop_n_layers,
        crop_nms_thresh=a.crop_nms_thresh, crop_overlap_ratio=a.crop_overlap_ratio,
        crop_n_points_downscale_factor=a.crop_n_points_downscale_factor,
        min_mask_region_area=(a.min_mask_region_area if a.min_mask_region_area >= 0 else None),
        use_m2m=bool(a.use_m2m), multimask_output=bool(a.multimask),
        cap_ppb=not bool(a.no_ppb_cap),
        on_tile=on_tile, on_tile_start=on_tile_start)
    print(f"STIM_DONE: {len(res)} sections", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="")  # "" = auto; else cpu/cuda/mps
    ap.add_argument("--target-long-side", type=int, default=3072)
    ap.add_argument("--points-per-side", type=int, default=32)
    ap.add_argument("--points-per-batch", type=int, default=16)
    ap.add_argument("--pred-iou-thresh", type=float, default=0.8)
    ap.add_argument("--stability-score-thresh", type=float, default=0.92)
    ap.add_argument("--stability-score-offset", type=float, default=1.0)
    ap.add_argument("--box-nms-thresh", type=float, default=0.7)
    ap.add_argument("--crop-n-layers", type=int, default=0)
    ap.add_argument("--crop-nms-thresh", type=float, default=0.7)
    ap.add_argument("--crop-overlap-ratio", type=float, default=512 / 1500)
    ap.add_argument("--crop-n-points-downscale-factor", type=int, default=1)
    ap.add_argument("--min-mask-region-area", type=int, default=-1)  # <0 => from min-area
    ap.add_argument("--use-m2m", type=int, default=0)
    # 1 = SAM default (3 masks/point, best recall); 0 = low-memory (1 mask/point)
    ap.add_argument("--multimask", type=int, default=1)
    # 1 = ignore the host RAM budget and run points-per-batch exactly as requested
    # (higher throughput, no OOM/thrash guard); 0 = clamp to a memory-safe value.
    ap.add_argument("--no-ppb-cap", type=int, default=0)
    ap.add_argument("--min-area", type=float, default=20.0)
    # 0 / negative => single whole-image tile
    ap.add_argument("--tile-px", type=int, default=0)
    ap.add_argument("--overlap", type=float, default=0.12)
    ap.add_argument("--max-area", type=float, default=1e12)
    a = ap.parse_args()

    t0 = time.time()
    monitor = _ResourceMonitor(device=a.device or "")
    monitor.start()
    try:
        _run_stream(a)
        print(f"STIM_PROGRESS: finished in {time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        import traceback
        print("STIM_ERROR: " + repr(e), flush=True)
        traceback.print_exc()
        sys.exit(1)
    finally:
        monitor.stop_and_report()


if __name__ == "__main__":
    main()
