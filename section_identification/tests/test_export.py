"""Headless tests for CZI TileRegion XML I/O + wafer export adapters.

Run:  python -m section_identification.tests.test_export
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from section_identification import czi_export, wafer_export
from section_identification.wafer_model import FocusPoint, Roi, Section, WaferProject


# --- TileRegions XML round-trip -------------------------------------------- #
MINIMAL_CZI_XML = ("<ImageDocument><Metadata>"
                   "<Experiment><ExperimentBlocks><AcquisitionBlock>"
                   "<SubDimensionSetups><RegionsSetup><SampleHolder>"
                   "</SampleHolder></RegionsSetup></SubDimensionSetups>"
                   "</AcquisitionBlock></ExperimentBlocks></Experiment>"
                   "</Metadata></ImageDocument>")


def test_inject_and_read_tile_regions():
    regions = [
        {"center_um": (100.0, 200.0), "contour_um": (50.0, 40.0), "columns": 3, "rows": 2,
         "z_um": 500.0, "support_points": [(90.0, 190.0, 500.0), (110.0, 210.0, 501.0)]},
        {"center_um": (300.0, 400.0), "contour_um": (60.0, 60.0), "columns": 4, "rows": 4,
         "z_um": 505.0, "support_points": []},
    ]
    xml = czi_export.inject_tile_regions(MINIMAL_CZI_XML, regions)
    read = czi_export.read_tile_regions(xml)
    assert len(read) == 2
    assert read[0]["center_um"] == (100.0, 200.0)
    assert read[0]["contour_um"] == (50.0, 40.0)
    assert read[0]["columns"] == 3 and read[0]["rows"] == 2 and read[0]["z_um"] == 500.0
    assert len(read[0]["support_points"]) == 2
    assert read[0]["support_points"][1] == (110.0, 210.0, 501.0)
    assert read[0]["name"].startswith(czi_export.TILE_REGION_PREFIX)


def test_inject_tile_regions_idempotent():
    regions = [{"center_um": (1.0, 2.0), "contour_um": (3.0, 4.0), "columns": 1, "rows": 1}]
    xml = czi_export.inject_tile_regions(MINIMAL_CZI_XML, regions)
    xml = czi_export.inject_tile_regions(xml, regions)   # re-inject
    assert len(czi_export.read_tile_regions(xml)) == 1   # not duplicated


def test_creates_chain_when_absent():
    xml = czi_export.inject_tile_regions("<ImageDocument><Metadata/></ImageDocument>",
                                         [{"center_um": (0.0, 0.0), "columns": 1, "rows": 1}])
    assert len(czi_export.read_tile_regions(xml)) == 1


# --- export adapters ------------------------------------------------------- #
class _FakeGeom:
    zoom = 0.5
    scale_x = 8e-9   # 8 nm/px
    scale_y = 8e-9

    def ds_to_full(self, x, y):
        return np.asarray(x, float) / self.zoom, np.asarray(y, float) / self.zoom

    def full_to_stage_um(self, x, y):
        return np.asarray(x, float) * 0.001, np.asarray(y, float) * 0.001


def _project():
    p = WaferProject(image_path="/tmp/waferX.czi")
    p.set_sections_from_polygons([[[0, 0], [10, 0], [10, 10], [0, 10]],
                                  [[20, 0], [30, 0], [30, 10], [20, 10]],
                                  [[40, 0], [50, 0], [50, 10], [40, 10]]])
    # serial order recovered: section_2, section_1, section_3
    p.match_graph.order = ["section_2", "section_1", "section_3"]
    for s, si in zip(p.sections, (1, 0, 2)):     # serial_index
        s.serial_index = si
    # imaging (TSP) order: section_3 first, then section_1, then section_2
    p.sections[2].imaging_index = 0
    p.sections[0].imaging_index = 1
    p.sections[1].imaging_index = 2
    p.sections[0].roi = Roi(polygon=[[2, 2], [6, 2], [6, 6], [2, 6]])
    p.sections[0].focus_points = [FocusPoint(1.0, 2.0, 3.0)]
    return p


def test_manifest_and_adapters():
    g = _FakeGeom()
    manifest = wafer_export.build_manifest(_project(), g, mfov_counts={"section_1": 6})
    assert manifest["schema"] == "stim.wafer/1"
    assert manifest["units"]["pixel_size_um"][0] == 8e-9 * 1e6      # 0.008 µm/px
    # imaging order in the manifest = section_3, section_1, section_2
    assert manifest["ordering"]["imaging"] == ["section_3", "section_1", "section_2"]

    with tempfile.TemporaryDirectory() as td:
        paths = wafer_export.write_all(manifest, td,
                                       adapters=["json_manifest", "csv_table", "mvis_lmb", "magc"])
        for name in ("json_manifest", "csv_table", "mvis_lmb", "magc"):
            assert os.path.isfile(paths[name]), name
        # region_names.csv: id in imaging order, name from serial index
        with open(os.path.join(td, "region_names.csv")) as f:
            rows = [ln.strip() for ln in f if ln.strip()]
        # acquisition order = imaging order; name = S<serial_index+1>
        # 001 = section_3 (imaging 0, serial 2 -> S3), default 1 mfov
        assert rows[0] == "001; S3; 1", rows
        # 002 = section_1 (imaging 1, serial 1 -> S2), mfov_counts gave it 6
        assert rows[1] == "002; S2; 6", rows
        # 003 = section_2 (imaging 2, serial 0 -> S1)
        assert rows[2] == "003; S1; 1", rows


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} export tests passed.")


if __name__ == "__main__":
    _run_all()
