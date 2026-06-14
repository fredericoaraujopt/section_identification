"""Headless tests for project_io save/load + legacy migration (no Qt/napari).

Run:  python -m section_identification.tests.test_project_io
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np

from section_identification import project_io
from section_identification.wafer_model import (QCResult, Roi, RoiTemplate,
                                                WaferProject)


class _FakeGeom:
    """full = overview * 2 (so disk coords double; load halves back)."""

    def ds_to_full(self, x, y):
        return np.asarray(x, float) * 2.0, np.asarray(y, float) * 2.0

    def full_to_ds(self, x, y):
        return np.asarray(x, float) / 2.0, np.asarray(y, float) / 2.0


SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10]]


def _project():
    p = WaferProject(image_path="/tmp/does_not_matter.czi")
    p.set_sections_from_polygons([SQUARE])
    p.sections[0].pose.center = (5.0, 5.0)
    p.sections[0].roi = Roi(polygon=[[2, 2], [4, 2], [4, 4], [2, 4]], fit_mode="full")
    p.sections[0].serial_index = 1
    p.fiducials = [(1.0, 2.0)]
    p.raw_sections = [SQUARE]
    p.roi_templates = [RoiTemplate(polygon_local=[[-1, -1], [1, -1], [1, 1]], ref_section_id="section_1")]
    return p


def test_disk_is_fullres_and_roundtrips_with_geom():
    g = _FakeGeom()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "proj.json")
        project_io.save(_project(), g, path=path)
        disk = json.load(open(path))
        assert disk["schema_version"] >= 2
        # overview (10,10) -> full (20,20) on disk
        assert disk["sections"][0]["polygon"][2] == [20.0, 20.0]
        assert disk["sections"][0]["pose"]["center"] == [10.0, 10.0]
        assert disk["sections"][0]["roi"]["polygon"][2] == [8.0, 8.0]
        assert disk["fiducials"][0] == [2.0, 4.0]
        # ROI template (pose-normalised) is NOT geom-converted
        assert disk["roi_templates"][0]["polygon_local"][1] == [1.0, -1.0]

        # load by reading the same file is exercised in test_load_versioned_via_project_path;
        # here verify the inverse conversion directly:
        d2 = project_io._convert_geometry(disk, project_io._poly_full_to_ds, g)
        proj2 = WaferProject.from_dict(d2)
        assert proj2.sections[0].bbox() == (0.0, 0.0, 10.0, 10.0)  # back to overview
        assert proj2.sections[0].pose.center == (5.0, 5.0)
        assert proj2.fiducials == [(1.0, 2.0)]
        assert proj2.roi_templates[0].polygon_local[1] == [1.0, -1.0]


def test_roundtrip_geom_none_identity():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "proj.json")
        project_io.save(_project(), None, path=path)
        disk = json.load(open(path))
        assert disk["sections"][0]["polygon"][2] == [10.0, 10.0]  # unchanged
        proj2 = WaferProject.from_dict(
            project_io._convert_geometry(disk, project_io._poly_full_to_ds, None))
        assert proj2.sections[0].bbox() == (0.0, 0.0, 10.0, 10.0)


def test_load_legacy_via_project_path(tmp_image=None):
    g = _FakeGeom()
    with tempfile.TemporaryDirectory() as td:
        img = os.path.join(td, "wafer.czi")
        ppath = project_io.project_path(img)
        os.makedirs(os.path.dirname(ppath), exist_ok=True)
        # legacy: full-res polygons, NO schema_version
        legacy = {"image": img,
                  "sections": [[[0, 0], [20, 0], [20, 20], [0, 20]]],
                  "fiducials": [[100, 200]], "raw_sections": [], "calibration_examples": []}
        with open(ppath, "w") as f:
            json.dump(legacy, f)
        proj = project_io.load(img, g)
        assert proj is not None
        assert proj.sections[0].id == "section_1"
        assert proj.sections[0].bbox() == (0.0, 0.0, 10.0, 10.0)  # 20 full -> 10 overview
        assert proj.fiducials == [(50.0, 100.0)]


def test_load_versioned_via_project_path():
    g = _FakeGeom()
    with tempfile.TemporaryDirectory() as td:
        img = os.path.join(td, "wafer.czi")
        project_io.save(_project_with_image(img), g)  # writes to project_path(img)
        proj = project_io.load(img, g)
        assert proj is not None and proj.image_path == img
        assert proj.sections[0].bbox() == (0.0, 0.0, 10.0, 10.0)
        assert proj.sections[0].serial_index == 1


def _project_with_image(img):
    p = _project()
    p.image_path = img
    return p


def test_apply_results_merge():
    src = WaferProject()
    src.set_sections_from_polygons([SQUARE, SQUARE])
    src.sections[0].qc = QCResult(scores={"overall": 0.9}, flags={"any": True})
    src.sections[0].serial_index = 0
    src.sections[1].serial_index = 1
    src.sections[0].imaging_index = 1
    src.sections[1].imaging_index = 0
    src.match_graph.order = ["section_1", "section_2"]

    tgt = WaferProject()
    tgt.set_sections_from_polygons([SQUARE, SQUARE])   # same ids, no results
    tgt.apply_results(src)
    assert tgt.sections[0].qc.flags["any"] is True
    assert tgt.sections[0].serial_index == 0 and tgt.sections[1].imaging_index == 0
    assert tgt.match_graph.order == ["section_1", "section_2"]


def test_workflow_sidecar_roundtrip():
    g = _FakeGeom()
    with tempfile.TemporaryDirectory() as td:
        img = os.path.join(td, "wafer.czi")
        p = _project_with_image(img)                    # has roi + serial_index=1
        p.sections[0].qc = QCResult(scores={"overall": 0.7}, flags={"fold": True, "any": True})
        project_io.save(p, g, path=project_io.workflow_path(img))
        assert os.path.isfile(project_io.workflow_path(img))

        fresh = WaferProject(image_path=img)
        fresh.set_sections_from_polygons([SQUARE])      # same single section, no results
        src = project_io.load(img, g, path=project_io.workflow_path(img))
        fresh.apply_results(src)
        assert fresh.sections[0].qc.flags["any"] is True
        assert fresh.sections[0].serial_index == 1
        assert fresh.sections[0].roi is not None


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} project_io tests passed.")


if __name__ == "__main__":
    _run_all()
