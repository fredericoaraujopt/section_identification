"""Headless tests for shape-based pose recovery (align.py).

The key property for ROI propagation: two copies of the same section at
different rotations/positions must map to the SAME canonical local frame.

Run:  python -m section_identification.tests.test_align
"""

from __future__ import annotations

import math

import numpy as np

from section_identification import align, fov_nav


# An ASYMMETRIC trapezoid (wide bottom, narrow top) — has a well-defined heavy
# end, so the 180° disambiguation is stable.
TRAP = np.array([[-10.0, -5.0], [10.0, -5.0], [5.0, 5.0], [-5.0, 5.0]])


def _transform(poly, angle_deg, tx, ty):
    t = math.radians(angle_deg)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return (poly @ R.T) + np.array([tx, ty])


def test_recovers_rotation():
    p = _transform(TRAP, 30.0, 100.0, 200.0)
    (cx, cy), angle, flip = align.estimate_pose(p)
    # centroid near the transformed centroid of TRAP
    c0 = _transform(TRAP, 30.0, 100.0, 200.0).mean(axis=0)
    assert abs(cx - c0[0]) < 2.0 and abs(cy - c0[1]) < 2.0
    assert flip is False


def test_rotation_invariant_local_frame():
    # Same shape at two different rotations+translations -> same local polygon.
    p1 = _transform(TRAP, 20.0, 50.0, -30.0)
    p2 = _transform(TRAP, 200.0, 999.0, 12.0)

    (c1, a1, f1) = align.estimate_pose(p1)
    (c2, a2, f2) = align.estimate_pose(p2)

    local1 = np.array([fov_nav.world_to_local(v, c1, a1, f1) for v in p1])
    local2 = np.array([fov_nav.world_to_local(v, c2, a2, f2) for v in p2])

    # vertices keep order under rotation, so compare element-wise
    assert np.allclose(local1, local2, atol=1e-6), (local1, local2)


def test_degenerate_polygon():
    (cx, cy), angle, flip = align.estimate_pose([[0, 0], [1, 1]])  # < 3 pts
    assert angle == 0.0 and flip is False


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} align tests passed.")


if __name__ == "__main__":
    _run_all()
