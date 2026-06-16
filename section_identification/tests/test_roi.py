"""Headless tests for ROI template extraction, propagation, and fitting.

Run:  python -m section_identification.tests.test_roi
"""

from __future__ import annotations

import math

import numpy as np

from section_identification import align, fov_nav, roi
from section_identification.wafer_model import Pose, RoiTemplate, Section


TRAP = np.array([[-10.0, -5.0], [10.0, -5.0], [5.0, 5.0], [-5.0, 5.0]])
ROI_SQ = np.array([[2.0, 2.0], [6.0, 2.0], [6.0, 6.0], [2.0, 6.0]])  # small ROI on the ref


def _transform(poly, angle_deg, tx, ty):
    t = math.radians(angle_deg)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return (poly @ R.T) + np.array([tx, ty])


def _section(poly):
    s = Section(id="s", polygon=[[float(x), float(y)] for x, y in np.asarray(poly)])
    align.pose_for_section(s)
    return s


def test_template_propagation_is_pose_consistent():
    # define ROI on reference (upright) section, propagate to a rotated copy
    ref = _section(TRAP)
    tmpl = roi.template_from_polygon(ref.pose, ROI_SQ, ref_section_id=ref.id, fit_mode="template")

    tgt = _section(_transform(TRAP, 73.0, 500.0, -200.0))
    world = np.array(roi.propagate_to_pose(tmpl, tgt.pose))

    # mapping the propagated ROI back into the target's local frame == template
    back = np.array([fov_nav.world_to_local(p, tgt.pose.center, tgt.pose.angle_deg, tgt.pose.flip)
                     for p in world])
    assert np.allclose(back, np.asarray(tmpl.polygon_local), atol=1e-6)


def test_fit_full_fits_section_bbox():
    sec = TRAP
    big_roi = ROI_SQ * 10.0  # way bigger than the section
    fitted = np.asarray(roi.fit_roi(big_roi, sec, mode="full"))
    fx0, fy0, fx1, fy1 = fitted[:, 0].min(), fitted[:, 1].min(), fitted[:, 0].max(), fitted[:, 1].max()
    sx0, sy0, sx1, sy1 = sec[:, 0].min(), sec[:, 1].min(), sec[:, 0].max(), sec[:, 1].max()
    # fitted ROI sits within the section bbox (+eps) and touches the limiting axis
    assert fx0 >= sx0 - 1e-6 and fx1 <= sx1 + 1e-6
    assert fy0 >= sy0 - 1e-6 and fy1 <= sy1 + 1e-6


def test_fit_percent_scales_down():
    full = np.asarray(roi.fit_roi(ROI_SQ * 10, TRAP, mode="full"))
    half = np.asarray(roi.fit_roi(ROI_SQ * 10, TRAP, mode="percent", percent=50.0))
    def area(p):
        x, y = p[:, 0], p[:, 1]
        return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2
    assert area(half) < area(full)
    assert abs(area(half) / area(full) - 0.25) < 0.05  # 50% linear -> 25% area


def test_fit_clip_intersects_section():
    # ROI overhanging the section -> clipped polygon area <= section area
    roi_poly = np.array([[0, 0], [40, 0], [40, 40], [0, 40]], dtype=float)
    clipped = np.asarray(roi.fit_roi(roi_poly, TRAP, mode="clip"))
    def area(p):
        x, y = p[:, 0], p[:, 1]
        return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2
    assert area(clipped) <= area(TRAP) + 1e-6
    assert area(clipped) > 0


def test_propagate_all_sets_rois():
    secs = [_section(_transform(TRAP, a, 100 * i, 0)) for i, a in enumerate((0, 45, 120))]
    ref = secs[0]
    tmpl = roi.template_from_polygon(ref.pose, ROI_SQ, ref_section_id=ref.id, fit_mode="template")
    roi.propagate_all(tmpl, secs)
    assert all(s.roi is not None and len(s.roi.polygon) == len(ROI_SQ) for s in secs)


def test_focus_propagation_pose_consistent():
    ref = _section(TRAP)
    focus_pts = [[0.0, 0.0], [3.0, -2.0]]                 # world pts on the reference
    tmpl = RoiTemplate(ref_section_id=ref.id)
    tmpl.focus_local = roi.focus_template_from_points(ref.pose, focus_pts)
    tgt = _section(_transform(TRAP, 60.0, 120.0, 40.0))
    roi.propagate_all(tmpl, [tgt])                         # focus-only template
    assert tgt.roi is None                                 # no polygon -> no ROI created
    assert len(tgt.focus_overview) == 2
    back = np.array([fov_nav.world_to_local(p, tgt.pose.center, tgt.pose.angle_deg,
                                            tgt.pose.flip) for p in tgt.focus_overview])
    assert np.allclose(back, np.asarray(tmpl.focus_local), atol=1e-6)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} roi tests passed.")


if __name__ == "__main__":
    _run_all()
