"""Headless tests for the relative-FOV navigator math.

Run:  python -m section_identification.tests.test_fov_nav
"""

from __future__ import annotations

import numpy as np

from section_identification import fov_nav
from section_identification.fov_nav import _SecPose


def test_local_world_roundtrip():
    pose = _SecPose(center=(100.0, 50.0), angle_deg=37.0, flip=True)
    pt = (123.0, 71.0)
    local = fov_nav.world_to_local(pt, pose.center, pose.angle_deg, pose.flip)
    back = fov_nav.local_to_world(local, pose.center, pose.angle_deg, pose.flip)
    assert np.allclose(back, pt)


def test_snap_between_rotated_sections():
    # A upright at (100,100); B rotated 90° at (200,300).
    A = _SecPose(center=(100.0, 100.0), angle_deg=0.0)
    B = _SecPose(center=(200.0, 300.0), angle_deg=90.0)
    # camera 5 units along A's local +x  ->  world (105,100)
    cam = (105.0, 100.0)
    new = fov_nav.snap_between(cam, A, B)
    # in B (rotated 90°) the same local (+5,0) maps to (0,+5) -> (200,305)
    assert np.allclose(new, (200.0, 305.0))


def test_snap_identity_same_pose():
    A = _SecPose(center=(10.0, 20.0), angle_deg=15.0, flip=False)
    cam = (33.0, 5.0)
    assert np.allclose(fov_nav.snap_between(cam, A, A), cam)


def test_fit_center_zoom():
    # bbox 100x50 world units into a 800x600 canvas, 0 margin
    center, zoom = fov_nav.fit_center_zoom((0, 0, 100, 50), (800, 600), margin=0.0)
    assert center == (50.0, 25.0)
    # limiting axis is width: 800/100 = 8 (height would give 600/50=12)
    assert abs(zoom - 8.0) < 1e-9
    # margin shrinks zoom
    _, zoom_m = fov_nav.fit_center_zoom((0, 0, 100, 50), (800, 600), margin=0.25)
    assert zoom_m < zoom


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} fov_nav tests passed.")


if __name__ == "__main__":
    _run_all()
