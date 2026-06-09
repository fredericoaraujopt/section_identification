#!/usr/bin/env python
"""Cheap end-to-end verification of the STiM upgrade.

Runs each risky path in isolation and prints PASS/FAIL, so we validate the
SAM 2.1 + CZI-read + CZI-write plumbing WITHOUT touching the 13 GB file or
needing ZEN. Safe to run repeatedly.
"""

import os
import sys
import time
import traceback

CZI = os.environ.get("STIM_CZI", "/Volumes/JK/tard_carbon_coat_001.czi")
CKPT = os.environ.get("STIM_CKPT", "checkpoint/sam2.1_hiera_base_plus.pt")
EXAMPLE = "images/example4.png"

results = []


def check(name, fn):
    t0 = time.time()
    try:
        info = fn()
        results.append((name, True, f"{info}  ({time.time()-t0:.1f}s)"))
        print(f"PASS  {name}: {info}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"FAIL  {name}: {e}")
        traceback.print_exc()


def t_imports():
    import section_identification.device  # noqa
    import section_identification.czi_io  # noqa
    import section_identification.czi_export  # noqa
    import section_identification.ordering  # noqa
    import section_identification.filtering  # noqa
    import section_identification.export  # noqa
    import section_identification.section_detector  # noqa
    from section_identification.device import describe
    return f"all modules import; device={describe()}"


def t_raw_metadata():
    from section_identification.czi_io import parse_czi_metadata_raw
    if not os.path.exists(CZI):
        return "skipped (CZI not mounted)"
    m = parse_czi_metadata_raw(CZI)
    assert m["size_x"] and m["size_y"], "missing dims"
    return f"{m['size_x']}x{m['size_y']} {m['pixel_type']} scale={m['scale_x']}"


def t_czi_read():
    from section_identification.czi_io import read_czi_overview, to_rgb8
    if not os.path.exists(CZI):
        return "skipped (CZI not mounted)"
    arr, geom, meta = read_czi_overview(CZI, target_long_side=1024)
    rgb = to_rgb8(arr)
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    # round-trip a coordinate
    fx, fy = geom.ds_to_full(0, 0)
    return f"overview {rgb.shape} zoom={meta['zoom']:.4g} origin_full=({fx},{fy})"


def t_czi_write_roundtrip():
    """Create a tiny CZI, inject Layers, re-read — validates the edit_czi API."""
    import numpy as np
    from pylibCZIrw import czi as pyczi
    from section_identification import czi_export

    tmp = "/tmp/stim_tiny.czi"
    tmp_out = "/tmp/stim_tiny_STiM.czi"
    for p in (tmp, tmp_out):
        if os.path.exists(p):
            os.remove(p)
    data = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    with pyczi.create_czi(tmp, exist_ok=True) as c:
        c.write(data, plane={"C": 0})
        c.write_metadata(scale_x=1.38e-6, scale_y=1.38e-6)
    poly = [[10, 10], [200, 10], [200, 200], [10, 200]]
    fids = [(20, 20), (230, 20), (20, 230)]
    report = czi_export.write_annotated_czi(tmp, tmp_out, [poly], fids)
    assert report["roundtrip_ok"], "polygons not found on re-read"
    return f"injected {report['n_polygons']} polys + {report['n_fiducials']} fids; roundtrip ok"


def t_sam2_detect():
    from section_identification.section_detector import automatic_identification
    if not os.path.exists(CKPT):
        return "skipped (checkpoint not present)"
    if not os.path.exists(EXAMPLE):
        return "skipped (example image not present)"
    masks = automatic_identification(EXAMPLE, checkpoint=CKPT,
                                     apply_filtering=False, points_per_side=16)
    assert len(masks) > 0, "no masks"
    return f"{len(masks)} masks on {EXAMPLE}"


def t_ordering():
    import numpy as np
    from section_identification import ordering
    img = (np.random.rand(400, 400) * 255).astype(np.uint8)
    bboxes = [(10, 10, 60, 60), (100, 10, 150, 60), (200, 10, 250, 60)]
    order, sim = ordering.order_sections(img, bboxes)
    assert len(order) == 3
    return f"order={list(order)} sim_shape={sim.shape}"


if __name__ == "__main__":
    check("imports", t_imports)
    check("raw_metadata", t_raw_metadata)
    check("czi_read", t_czi_read)
    check("czi_write_roundtrip", t_czi_write_roundtrip)
    check("ordering", t_ordering)
    check("sam2_detect", t_sam2_detect)
    print("\n=== SUMMARY ===")
    for name, ok, info in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {info}")
    n_fail = sum(1 for _, ok, _ in results if not ok)
    sys.exit(1 if n_fail else 0)
