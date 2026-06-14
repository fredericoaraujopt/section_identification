"""Headless smoke tests for the wafer data model (no Qt/napari needed).

Run directly:  python -m section_identification.tests.test_wafer_model
Or via pytest:  pytest section_identification/tests/test_wafer_model.py
"""

from __future__ import annotations

import numpy as np

from section_identification.wafer_model import (
    SCHEMA_VERSION, FocusPoint, MatchEdge, MatchGraph, QCResult, Roi,
    RoiTemplate, Section, WaferProject,
)


# A square section: easy to reason about centroid/area/bbox.
SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10]]


class _FakeGeom:
    """Minimal duck-typed geometry: full = overview*2; stage_um = full + 1000."""

    def ds_to_full(self, x, y):
        return np.asarray(x) * 2.0, np.asarray(y) * 2.0

    def full_to_ds(self, x, y):
        return np.asarray(x) / 2.0, np.asarray(y) / 2.0

    def full_to_stage_um(self, x, y):
        return np.asarray(x) + 1000.0, np.asarray(y) + 1000.0


def test_section_geometry():
    s = Section(id="section_1", polygon=SQUARE)
    assert s.centroid() == (5.0, 5.0)
    assert s.area() == 100.0
    assert s.bbox() == (0.0, 0.0, 10.0, 10.0)
    # geom=None -> identity
    assert s.polygon_full(None) == [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    assert s.centroid_stage_um(None) is None
    # with a geom: full doubles, stage offsets +1000 on the doubled full coords
    g = _FakeGeom()
    assert s.polygon_full(g)[2] == [20.0, 20.0]
    assert s.centroid_stage_um(g) == (1010.0, 1010.0)


def test_section_roundtrip_with_substate():
    s = Section(
        id="section_7", polygon=SQUARE,
        roi=Roi(polygon=[[1, 1], [4, 1], [4, 4], [1, 4]], fit_mode="percent", fit_percent=80.0),
        focus_points=[FocusPoint(10.0, 20.0, 30.0), FocusPoint(11.0, 21.0, 31.0)],
        qc=QCResult(scores={"fold": 0.7, "overall": 0.7}, flags={"fold": True, "any": True},
                    features={"fold_max_ridge_len_px": 42.0}),
        serial_index=3, imaging_index=12, accepted=False, sift_ref="cache/section_7.npz",
    )
    s.pose.center = (5.0, 5.0); s.pose.angle_deg = 33.0; s.pose.flip = True
    s2 = Section.from_dict(s.to_dict())
    assert s2.id == "section_7"
    assert s2.roi.fit_mode == "percent" and s2.roi.fit_percent == 80.0
    assert [fp.z_um for fp in s2.focus_points] == [30.0, 31.0]
    assert s2.qc.flags["any"] is True and s2.qc.scores["fold"] == 0.7
    assert s2.serial_index == 3 and s2.imaging_index == 12 and s2.accepted is False
    assert s2.pose.center == (5.0, 5.0) and s2.pose.flip is True


def test_project_ids_and_ordering():
    p = WaferProject(image_path="/tmp/wafer.czi")
    p.set_sections_from_polygons([SQUARE, SQUARE, SQUARE])
    assert [s.id for s in p.sections] == ["section_1", "section_2", "section_3"]
    # next id skips used
    p.sections.pop(1)  # remove section_2
    assert p.new_id() == "section_2"
    # serial / imaging ordering
    p.sections[0].serial_index = 2; p.sections[1].serial_index = 1
    assert [s.id for s in p.in_serial_order()] == ["section_3", "section_1"]
    p.match_graph.order = ["section_3", "section_1"]
    assert [s.id for s in p.in_serial_order()] == ["section_3", "section_1"]
    p.sections[0].imaging_index = 5; p.sections[1].imaging_index = 0
    assert [s.id for s in p.in_imaging_order()] == ["section_3", "section_1"]


def test_project_roundtrip():
    p = WaferProject(image_path="/tmp/wafer.czi")
    p.set_sections_from_polygons([SQUARE, SQUARE])
    p.fiducials = [(1.0, 2.0), (3.0, 4.0)]
    p.raw_sections = [SQUARE]
    p.calibration_examples = [SQUARE]
    p.roi_templates = [RoiTemplate(polygon_local=[[0, 0], [2, 0], [2, 2]], ref_section_id="section_1",
                                   tile_um=(15.0, 16.0), focus_cols=2, focus_rows=2)]
    p.match_graph = MatchGraph(edges=[MatchEdge("section_1", "section_2", 42, 0.9)],
                               order=["section_1", "section_2"], method="spectral+2opt")
    d = p.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    p2 = WaferProject.from_dict(d)
    assert len(p2.sections) == 2
    assert p2.fiducials == [(1.0, 2.0), (3.0, 4.0)]
    assert p2.roi_templates[0].tile_um == (15.0, 16.0)
    assert p2.match_graph.edges[0].inliers == 42
    assert p2.match_graph.order == ["section_1", "section_2"]


def test_from_legacy():
    # legacy file: full-res polygons, no schema_version. to_overview halves coords.
    def to_overview(pts):
        return np.asarray(pts, dtype=float).reshape(-1, 2) / 2.0

    legacy = {
        "image": "/tmp/old.czi",
        "sections": [[[0, 0], [20, 0], [20, 20], [0, 20]]],   # full-res
        "fiducials": [[100, 200]],
        "raw_sections": [[[0, 0], [20, 0], [20, 20]]],
        "calibration_examples": [],
    }
    p = WaferProject.from_legacy(legacy, to_overview)
    assert p.image_path == "/tmp/old.czi"
    assert p.sections[0].id == "section_1"
    # 20 full -> 10 overview
    assert p.sections[0].bbox() == (0.0, 0.0, 10.0, 10.0)
    assert p.fiducials == [(50.0, 100.0)]
    assert len(p.raw_sections) == 1


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} wafer_model tests passed.")


if __name__ == "__main__":
    _run_all()
