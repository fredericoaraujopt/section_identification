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


def test_propagate_center_keeps_dimensions_and_centres():
    # sections of wildly different sizes; centered mode must keep ROI size fixed
    ref = _section(TRAP)
    small = _section(_transform(TRAP * 0.2, 30.0, 300.0, 100.0))
    big = _section(_transform(TRAP * 5.0, 100.0, -400.0, 50.0))
    # center mode stores the raw drawn polygon (as the ROI-stage button does)
    tmpl = RoiTemplate(polygon_local=[[float(x), float(y)] for x, y in ROI_SQ],
                       ref_section_id=ref.id, fit_mode="center")
    roi.propagate_all(tmpl, [ref, small, big])

    def dims(p):
        p = np.asarray(p, float)
        return (p[:, 0].max() - p[:, 0].min(), p[:, 1].max() - p[:, 1].min())
    roi_w, roi_h = dims(ROI_SQ)
    for s in (ref, small, big):
        assert s.roi is not None and s.roi.fit_mode == "center"
        w, h = dims(s.roi.polygon)
        assert abs(w - roi_w) < 1e-6 and abs(h - roi_h) < 1e-6   # unchanged dimensions
        c = np.asarray(s.roi.polygon, float).mean(axis=0)        # ROI centroid ...
        assert np.allclose(c, np.asarray(s.centroid()), atol=1e-6)  # ... on section centroid


def test_focus_only_on_roi_sections():
    # Focus points are placed ONLY on sections that have an ROI (an un-imaged
    # region needs no focus). A section-anchored point propagates pose-relative
    # to the section, so it maps consistently across rotations.
    ref = _section(TRAP)
    tmpl = roi.template_from_polygon(ref.pose, ROI_SQ, ref_section_id=ref.id, fit_mode="template")
    with_roi = _section(_transform(TRAP, 60.0, 120.0, 40.0))
    no_roi = _section(_transform(TRAP, 10.0, -200.0, 50.0))
    roi.propagate_all(tmpl, [ref, with_roi, no_roi])       # ROI on all 3 (template mode)
    no_roi.roi = None                                       # this one has no ROI

    # focus-only template (no polygon_local → doesn't recreate ROIs): a
    # section-anchored focus point outside the ROI but inside the section.
    ftmpl = RoiTemplate(ref_section_id=ref.id)
    ftmpl.focus_anchors = roi.focus_anchors_from_points(ref, [[9.0, -4.0]])
    ftmpl.focus_mode = "pose"
    assert ftmpl.focus_anchors[0]["anchor"] == "section"
    roi.propagate_all(ftmpl, [ref, with_roi, no_roi])
    assert len(with_roi.focus_overview) == 1               # has ROI → gets focus
    assert no_roi.focus_overview == []                     # no ROI → no focus (req. #1)


def test_focus_pose_anchoring_roi_vs_section():
    # Pose mode: an ROI-anchored point rides each section's ROI (rotation-aware);
    # a section-anchored point rides the section. Verify both land at the same
    # pose-normalised offset from their respective anchor on a rotated section.
    ref = _section(TRAP)
    tgt = _section(_transform(TRAP, 73.0, 400.0, -150.0))
    tmpl = roi.template_from_polygon(ref.pose, ROI_SQ, ref_section_id=ref.id, fit_mode="template")
    roi.propagate_all(tmpl, [ref, tgt])
    roi_c_ref = np.asarray(ref.roi.polygon, float).mean(axis=0)
    p_in = [float(roi_c_ref[0] + 1.0), float(roi_c_ref[1] + 1.0)]   # inside ROI
    p_out = [9.0, -4.0]                                             # in section, outside ROI
    tmpl.focus_anchors = roi.focus_anchors_from_points(ref, [p_in, p_out])
    tmpl.focus_mode = "pose"
    assert [a["anchor"] for a in tmpl.focus_anchors] == ["roi", "section"]
    roi.propagate_all(tmpl, [ref, tgt])
    fp = np.asarray(tgt.focus_overview, float)
    roi_c_t = np.asarray(tgt.roi.polygon, float).mean(axis=0)
    # ROI-anchored point, mapped back into the target's local frame about the ROI
    # centroid, equals the stored local offset (rotation-aware, same relative spot)
    back_roi = fov_nav.world_to_local(fp[0], (roi_c_t[0], roi_c_t[1]),
                                      tgt.pose.angle_deg, tgt.pose.flip)
    assert np.allclose(back_roi, tmpl.focus_anchors[0]["local"], atol=1e-6)
    back_sec = fov_nav.world_to_local(fp[1], tgt.centroid(),
                                      tgt.pose.angle_deg, tgt.pose.flip)
    assert np.allclose(back_sec, tmpl.focus_anchors[1]["local"], atol=1e-6)


def test_focus_center_anchoring():
    # Centre mode: an ROI-anchored point lands exactly on each section's ROI
    # centroid; a section-anchored point lands on the section centroid — wherever
    # in the anchor it was drawn (no offset, no pose).
    ref = _section(TRAP)
    small = _section(_transform(TRAP * 0.3, 20.0, 300.0, 100.0))
    big = _section(_transform(TRAP * 4.0, 90.0, -400.0, 50.0))
    secs = [ref, small, big]
    tmpl = RoiTemplate(polygon_local=[[float(x), float(y)] for x, y in ROI_SQ],
                       ref_section_id=ref.id, fit_mode="center")
    roi.propagate_all(tmpl, secs)                 # every section gets a centered ROI

    ref_roi_c = np.asarray(ref.roi.polygon, float).mean(axis=0)
    p_in = [float(ref_roi_c[0] + 0.5), float(ref_roi_c[1] + 0.5)]   # off-centre inside ROI
    p_out = [9.0, -4.0]                                             # in section, outside ROI
    tmpl.focus_anchors = roi.focus_anchors_from_points(ref, [p_in, p_out])
    tmpl.focus_mode = "center"
    assert [a["anchor"] for a in tmpl.focus_anchors] == ["roi", "section"]

    roi.propagate_focus_only(tmpl, secs)
    for s in secs:
        fp = np.asarray(s.focus_overview, float)
        roi_c = np.asarray(s.roi.polygon, float).mean(axis=0)
        assert np.allclose(fp[0], roi_c, atol=1e-6)                # ROI-anchored → ROI centre
        assert np.allclose(fp[1], np.asarray(s.centroid()), atol=1e-6)  # section centre


def test_focus_anchors_serialize_roundtrip():
    # focus_anchors (anchor + local) must survive save/reload, else propagation
    # silently reverts to the centroid on the next session.
    t = RoiTemplate(focus_mode="pose",
                    focus_anchors=[{"anchor": "roi", "local": [1.5, -2.5]},
                                   {"anchor": "section", "local": [3.0, 4.0]}])
    t2 = RoiTemplate.from_dict(t.to_dict())
    assert t2.focus_mode == "pose"
    assert [a["anchor"] for a in t2.focus_anchors] == ["roi", "section"]
    assert t2.focus_anchors[0]["local"] == [1.5, -2.5]
    assert t2.focus_anchors[1]["local"] == [3.0, 4.0]


def test_section_point_grid_inside_and_scales():
    sec = [[0, 0], [100, 0], [100, 100], [0, 100]]
    g8 = np.asarray(roi.section_point_grid(sec, 8))
    g4 = np.asarray(roi.section_point_grid(sec, 4))
    assert len(g8) > 0 and len(g4) < len(g8)                 # denser grid → more points
    assert g8[:, 0].min() >= 0 and g8[:, 0].max() <= 100     # all inside the section bbox
    # a triangle: grid points outside the polygon are dropped
    tri = [[0, 0], [100, 0], [0, 100]]
    gt = np.asarray(roi.section_point_grid(tri, 8))
    assert all(x + y <= 100 + 1e-6 for x, y in gt)


def test_score_and_choose_prefer_template_match():
    sec = [[0, 0], [100, 0], [100, 100], [0, 100]]
    tmpl = [[40, 40], [60, 40], [60, 60], [40, 60]]          # area 400
    band = (200.0, 800.0)
    good = [[41, 41], [59, 41], [59, 59], [41, 59]]          # template-like, contained
    tiny = [[48, 48], [52, 48], [52, 52], [48, 52]]          # area 16 → outside band → 0
    leak = [[90, 90], [130, 90], [130, 130], [90, 130]]      # partly outside section
    assert roi.score_mask(tiny, tmpl, sec, 0.95, band) == 0.0
    assert roi.score_mask(good, tmpl, sec, 0.95, band) > roi.score_mask(leak, tmpl, sec, 0.9, band)
    best, sc = roi.choose_best_roi([(good, 0.95), (tiny, 0.95), (leak, 0.9)], tmpl, sec, band, floor=0.35)
    assert best is not None and abs(np.asarray(best).mean(0)[0] - 50) < 5
    none, _ = roi.choose_best_roi([(tiny, 0.95)], tmpl, sec, band, floor=0.35)
    assert none is None                                       # nothing clears the floor


def test_fit_template_to_mask_matches_area_and_centroid():
    tmpl = [[0, 0], [10, 0], [10, 10], [0, 10]]               # area 100 at origin
    mask = [[100, 100], [120, 100], [120, 120], [100, 120]]   # area 400, centred (110,110)
    fit = np.asarray(roi.fit_template_to_mask(tmpl, mask))
    assert np.allclose(fit.mean(0), [110, 110], atol=1e-6)    # recentred on the mask
    fa = roi._shoelace_area(fit)
    assert abs(fa - 400.0) < 1e-3                             # scaled to the mask area


def test_calibrate_roi_params_from_template():
    try:
        import cv2  # noqa: F401
    except Exception:
        print("  skip test_calibrate_roi_params_from_template (cv2 unavailable)")
        return
    sec = [[0, 0], [100, 0], [100, 100], [0, 100]]            # 100×100 sections
    tmpl = [[40, 40], [60, 40], [60, 60], [40, 60]]           # 20×20 ROI
    p = roi.calibrate_roi_params(tmpl, [sec, sec, sec], min_points=3)
    assert 6 <= p["points_per_side"] <= 48
    assert p["points_on_roi"] >= 2.0                          # grid lands a few pts on the ROI
    assert 0.7 <= p["pred_iou_thresh"] <= 0.9
    assert p["roi_area"] > 0 and p["min_area_frac"] < 1.0 < p["max_area_mult"]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} roi tests passed.")


if __name__ == "__main__":
    _run_all()
