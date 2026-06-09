#!/usr/bin/env python
"""Run STiM section detection on a CZI and write a ZEN-annotated CZI.

Example:
    python scripts/run_tardigrade.py \
        --czi /Volumes/JK/tard_carbon_coat_001.czi \
        --checkpoint checkpoint/sam2.1_hiera_base_plus.pt

Reads a downscaled pyramid overview (never the full 13 GB), runs SAM 2.1 on the
Apple GPU (MPS), filters to section-shaped masks, simplifies + scales polygons
back to full-resolution pixels, and produces:
  * <czi>_files/<name>_mask_coordinates.csv
  * <czi>_files/<name>_sections.geojson
  * <name>_STiM.czi   (copy of the source with <Layers> section polygons)
"""

import argparse
import os
import time

from section_identification import czi_io, czi_export, ordering
from section_identification.export import masks_to_polygons, export_mask_coordinates
from section_identification.section_detector import automatic_identification
from section_identification.device import describe as describe_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--czi", default="/Volumes/JK/tard_carbon_coat_001.czi")
    ap.add_argument("--checkpoint", default="checkpoint/sam2.1_hiera_base_plus.pt")
    ap.add_argument("--target-long-side", type=int, default=4096)
    ap.add_argument("--points-per-side", type=int, default=32)
    ap.add_argument("--points-per-batch", type=int, default=32)
    ap.add_argument("--min-area", type=int, default=100)
    ap.add_argument("--crop-n-layers", type=int, default=0,
                    help="Set 1 to catch small sections (slower).")
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--dst", default=None,
                    help="Output CZI path (default: <name>_STiM.czi next to source).")
    ap.add_argument("--skip-czi", action="store_true",
                    help="Skip the (slow, ~13 GB) CZI copy+annotate step.")
    args = ap.parse_args()

    print(f"Device: {describe_device()}")
    t0 = time.time()

    # 1) overview + geometry
    arr, geom, meta = czi_io.read_czi_overview(args.czi, args.target_long_side)
    overview = czi_io.to_rgb8(arr)
    print(f"Overview {overview.shape} | full {meta['size_x']}x{meta['size_y']} "
          f"| zoom {meta['zoom']:.4g} | scale {meta['scale_x']} m/px "
          f"| read {time.time() - t0:.1f}s")

    # 2) detection
    masks = automatic_identification(
        args.czi, checkpoint=args.checkpoint, image=overview,
        apply_filtering=not args.no_filter,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        crop_n_layers=args.crop_n_layers, min_mask_region_area=args.min_area)
    print(f"Detected {len(masks)} sections.")

    if not masks:
        print("No sections detected — try --no-filter or a larger --target-long-side.")
        return

    # 3) recover serial order (cross-correlation)
    bboxes = ordering.masks_to_bboxes(masks)
    if len(masks) >= 2:
        order, _ = ordering.order_sections(overview, bboxes, method="spectral")
        masks = [masks[i] for i in order]
        print(f"Cross-correlation order applied: {list(order)}")

    # 3b) overlay preview (overview coords) for quick QC
    try:
        import cv2
        from section_identification.export import mask_to_polygon
        vis = overview.copy()
        for i, m in enumerate(masks, start=1):
            p = mask_to_polygon(m["segmentation"])
            if p is not None and len(p) >= 3:
                cv2.drawContours(vis, [p.astype("int32")], -1, (255, 0, 0), 2)
        fdir = f"{os.path.splitext(args.czi)[0]}_files"
        os.makedirs(fdir, exist_ok=True)
        base0 = os.path.splitext(os.path.basename(args.czi))[0]
        cv2.imwrite(os.path.join(fdir, f"{base0}_overlay.png"),
                    cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"Wrote overlay: {fdir}/{base0}_overlay.png")
    except Exception as e:
        print(f"[warn] overlay failed: {e}")

    # 4) CSV + GeoJSON (full-resolution coords) — fast, local
    section_ids = [f"section_{i}" for i in range(1, len(masks) + 1)]
    export_mask_coordinates(args.czi, [], masks, [], geom=geom,
                            section_ids=section_ids, write_czi=False)

    # 5) annotated CZI (copy + metadata edit) — the ZEN deliverable
    if not args.skip_czi:
        polys_full = masks_to_polygons(masks, geom=geom)
        base = os.path.splitext(os.path.basename(args.czi))[0]
        dst = args.dst or os.path.join(os.path.dirname(args.czi), f"{base}_STiM.czi")
        print(f"Writing annotated CZI -> {dst} (copying ~{meta['size_x']*meta['size_y']*2/1e9:.1f} GB)…")
        report = czi_export.write_annotated_czi(
            args.czi, dst, [p.tolist() for p in polys_full], [],
            section_ids=section_ids)
        print(f"Annotated CZI: {report['dst']} | polygons={report['n_polygons']} "
              f"| round-trip ok={report['roundtrip_ok']}")

    print(f"Total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
