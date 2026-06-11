import os
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from section_identification.device import get_device, device_str, autocast_ctx
from section_identification.filtering import filtering
from section_identification import czi_io


# --------------------------------------------------------------------------- #
# Model building (SAM 2.1 primary, original SAM 1 fallback)
# --------------------------------------------------------------------------- #
def _infer_sam2_cfg(checkpoint: str) -> str:
    """Map a SAM 2.1 checkpoint filename to its hydra config name."""
    name = os.path.basename(str(checkpoint)).lower()
    if "tiny" in name:
        return "configs/sam2.1/sam2.1_hiera_t.yaml"
    if "small" in name:
        return "configs/sam2.1/sam2.1_hiera_s.yaml"
    if "base_plus" in name or "b+" in name or "bplus" in name:
        return "configs/sam2.1/sam2.1_hiera_b+.yaml"
    if "large" in name:
        return "configs/sam2.1/sam2.1_hiera_l.yaml"
    return "configs/sam2.1/sam2.1_hiera_b+.yaml"


def _is_sam2_checkpoint(checkpoint: str) -> bool:
    name = os.path.basename(str(checkpoint)).lower()
    return name.endswith(".pt") or "sam2" in name


def build_image_predictor(checkpoint, model_cfg, device):
    """Build a SAM 2.1 image predictor for interactive click-to-segment.

    Used by the GUI to correct false negatives in real time: ``set_image`` once,
    then ``predict(point_coords=...)`` per click. SAM 1 (.pth) is unsupported
    here — interactive correction uses SAM 2.1 to match the detector.
    """
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    cfg = model_cfg or _infer_sam2_cfg(checkpoint)
    sam2 = build_sam2(cfg, str(checkpoint), device=str(device))
    return SAM2ImagePredictor(sam2)


def build_mask_generator(checkpoint, model_cfg, device, params):
    """Return an automatic mask generator (SAM 2.1 if possible, else SAM 1)."""
    if _is_sam2_checkpoint(checkpoint):
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        cfg = model_cfg or _infer_sam2_cfg(checkpoint)
        print(f"STIM_PROGRESS: loading SAM model (cfg={os.path.basename(cfg)}) on {device}…", flush=True)
        sam2 = build_sam2(cfg, str(checkpoint), device=str(device),
                          apply_postprocessing=False)
        # Full AMG breadth; params.get keeps callers that pass only the basics
        # working (defaults match SAM2's own).
        return SAM2AutomaticMaskGenerator(
            sam2,
            points_per_side=params["points_per_side"],
            points_per_batch=params["points_per_batch"],
            pred_iou_thresh=params["pred_iou_thresh"],
            stability_score_thresh=params["stability_score_thresh"],
            stability_score_offset=params.get("stability_score_offset", 1.0),
            mask_threshold=params.get("mask_threshold", 0.0),
            box_nms_thresh=params["box_nms_thresh"],
            crop_n_layers=params["crop_n_layers"],
            crop_nms_thresh=params.get("crop_nms_thresh", 0.7),
            crop_overlap_ratio=params.get("crop_overlap_ratio", 512 / 1500),
            crop_n_points_downscale_factor=params.get("crop_n_points_downscale_factor", 1),
            min_mask_region_area=params["min_mask_region_area"],
            use_m2m=params.get("use_m2m", False),
            # multimask_output=True (SAM default) emits 3 candidate masks per
            # prompt point and keeps the best — better recall, but 3× the
            # per-batch upsample memory. False is the low-memory path.
            multimask_output=params.get("multimask_output", True),
            output_mode=params["output_mode"],
        )

    # ---- legacy SAM 1 fallback (vit_h/.pth) ----
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    model_type = params.get("model_type", "vit_h")
    print(f"STIM_PROGRESS: loading original SAM ({model_type}) on {device}…", flush=True)
    sam = sam_model_registry[model_type](str(checkpoint))
    sam.to(device=str(device))
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=params["points_per_side"],
        pred_iou_thresh=params["pred_iou_thresh"],
        stability_score_thresh=params["stability_score_thresh"],
        min_mask_region_area=params["min_mask_region_area"],
        output_mode=params["output_mode"],
    )


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #
def load_image_for_detection(image_path, target_long_side=4096):
    """Load an image as uint8 RGB for SAM, plus a geometry handle.

    CZI files are read from the pyramid at a downscaled level (never full res)
    and contrast-stretched; ordinary images go through PIL. Returns
    ``(rgb8, geom_or_None)``.
    """
    if czi_io.is_czi(image_path):
        arr, geom, meta = czi_io.read_czi_overview(image_path, target_long_side)
        print(f"CZI overview: full={meta['size_x']}x{meta['size_y']} "
              f"zoom={meta['zoom']:.4g} read={arr.shape} "
              f"scale={meta['scale_x']} m/px")
        rgb = czi_io.to_rgb8(arr)
        return rgb, geom
    image = np.array(Image.open(image_path).convert("RGB"))
    return image, None


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def automatic_identification(image_path, checkpoint, image=None, model_cfg=None,
                             compress=False, apply_filtering=False,
                             visualize=False, eps_values=None,
                             min_samples_values=None, target_long_side=4096,
                             return_meta=False, **kwargs):
    """Automatic section detection with SAM 2.1 (drop-in for the old SAM 1 call).

    Args:
        image_path: image file (``.czi`` reads a downscaled pyramid overview;
            png/jpg/tif via PIL). Used for cache naming even when ``image`` is given.
        checkpoint: SAM 2.1 ``.pt`` checkpoint (``.pth`` falls back to SAM 1).
        image: optional pre-loaded uint8 RGB array (skips loading).
        model_cfg: optional SAM 2.1 hydra config; inferred from the checkpoint name.
        apply_filtering: cluster masks to keep the real sections (area + shape).
        return_meta: if True, return ``(masks, info)`` where ``info`` holds the
            ``geom`` and the ``image`` used.
        **kwargs: SAM AMG params — points_per_side, points_per_batch,
            pred_iou_thresh, stability_score_thresh, box_nms_thresh,
            crop_n_layers, min_mask_region_area, output_mode, device.

    Returns:
        masks sorted by area (or ``(masks, info)`` if ``return_meta``).
    """
    start_time = time.time()
    default_params = {
        "points_per_side": 32,
        "points_per_batch": 64,
        "pred_iou_thresh": 0.8,
        "stability_score_thresh": 0.92,
        "box_nms_thresh": 0.7,
        "crop_n_layers": 0,
        "min_mask_region_area": 100,
        # RLE keeps the mask cache tiny (KB/mask) instead of a full binary array
        # (~9 MB/mask at a 3072 overview -> multi-GB caches that trigger swap).
        "output_mode": "coco_rle",
        "device": None,  # auto-select
    }
    params = {**default_params, **kwargs}
    device = get_device(params["device"])

    if compress:
        print("[note] The lossy 'compress' path was retired; CZI files use "
              "pyramid downscaling and other formats load at full size.")

    # Ensure per-image working directory exists.
    file_directory = f"{os.path.splitext(image_path)[0]}_files"
    os.makedirs(file_directory, exist_ok=True)

    # Load image (unless one was supplied).
    geom = None
    if image is None:
        image, geom = load_image_for_detection(image_path, target_long_side)
    print("Image shape:", image.shape, "| device:", device_str(params["device"]))

    # SAM 2's automatic generator upsamples every mask in a batch back to the
    # full input size; on a large image that intermediate tensor
    # (points_per_batch x 3 x H x W x 4B) can blow past GPU memory. Cap
    # points_per_batch to keep it under a safe budget (no effect on results).
    H, W = image.shape[:2]
    from section_identification import host_profile
    budget = host_profile.detect_profile(params["device"]).mem_budget_bytes
    safe_ppb = host_profile.safe_points_per_batch(budget, H, W, params["points_per_batch"])
    if safe_ppb < params["points_per_batch"]:
        print(f"[mem] capping points_per_batch "
              f"{params['points_per_batch']} -> {safe_ppb} for {W}x{H} image", flush=True)
        params["points_per_batch"] = safe_ppb

    if eps_values is None:
        eps_values = np.linspace(100, 1200, 11)
    if min_samples_values is None:
        min_samples_values = range(1, 5)

    # Cache key includes model + resolution + key detection params so different
    # settings don't collide.
    model_tag = os.path.splitext(os.path.basename(str(checkpoint)))[0]
    param_tag = (f"pps{params['points_per_side']}_cnl{params['crop_n_layers']}"
                 f"_a{params['min_mask_region_area']}_iou{params['pred_iou_thresh']}")
    cache_file = (f"{file_directory}/"
                  f"{os.path.basename(os.path.splitext(image_path)[0])}"
                  f"_{model_tag}_{max(image.shape[:2])}_{param_tag}_masks.pkl")

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            sorted_masks = pickle.load(f)
        print(f"Loaded {len(sorted_masks)} cached masks.")
    else:
        mask_generator = build_mask_generator(checkpoint, model_cfg, device, params)
        print("STIM_PROGRESS: generating masks…", flush=True)
        with autocast_ctx(device):
            generated_masks = mask_generator.generate(image)
        print(f"Generated {len(generated_masks)} masks.", flush=True)
        sorted_masks = sorted(generated_masks, key=lambda x: x["area"], reverse=True)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(sorted_masks, f)
            print("Cached masks.")
        except OSError as e:
            # Don't lose a long detection run if the cache dir is gone (e.g. an
            # external drive unmounted mid-run).
            print(f"[warn] could not write mask cache ({e}); continuing without cache.")

    if apply_filtering:
        print("Filtering masks…")
        sorted_masks, chosen_params = filtering(sorted_masks, eps_values,
                                                min_samples_values)
        print(f"Filtering kept {len(sorted_masks)} masks "
              f"(params: {chosen_params}).")

    if visualize:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(image)
        label = os.path.splitext(os.path.basename(image_path))[0]
        ax.set_title(f"{label} - {'Filtered' if apply_filtering else 'Identified'} Masks")
        for i, mask in enumerate(sorted_masks):
            x, y, w, h = mask["bbox"]
            ax.add_patch(Rectangle((x, y), w, h, linewidth=1, edgecolor="blue",
                                   facecolor="none"))
            ax.text(x + w / 2, y + h / 2, str(i), color="yellow", ha="center",
                    va="center", fontsize=8,
                    bbox=dict(facecolor="black", alpha=0.5))
        plt.show()

    print(f"automatic_identification({os.path.basename(image_path)}): "
          f"{time.time() - start_time:.1f}s, {len(sorted_masks)} masks")

    if return_meta:
        return sorted_masks, {"geom": geom, "image": image}
    return sorted_masks
