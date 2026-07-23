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


# --- ZEN CAT layers (sections + ROIs + focus, pixel frame) ----------------- #
def test_cat_layers_roundtrip_and_node_target():
    import xml.etree.ElementTree as ET
    skel = ("<ImageDocument><Metadata><Information><Image>"
            "<SizeX>8000</SizeX><SizeY>8000</SizeY></Image></Information>"
            "<MetadataNodes><MetadataNode><Layers/></MetadataNode></MetadataNodes>"
            "</Metadata></ImageDocument>")
    secs = [[(1000, 1000), (2000, 1000), (2000, 2000), (1000, 2000)],
            [(5000, 5000), (6000, 5000), (6000, 6000), (5000, 6000)]]
    rois = [[(1200, 1200), (1500, 1200), (1500, 1500), (1200, 1500)]]
    focus = [(1300, 1300), (5300, 5300)]
    xml = czi_export.inject_cat_layers(skel, secs, rois=rois, focus=focus,
                                       section_ids=["S1", "S2"])
    # CAT layers land in the MetadataNode Layers node (where ZEN reads them)
    root = ET.fromstring(xml)
    names = [L.get("Name")
             for L in root.findall(".//MetadataNodes/MetadataNode/Layers/Layer")]
    assert czi_export.CAT_SECTION_LAYER in names and czi_export.CAT_ROI_LAYER in names
    # a section polygon is tagged UniqueName=Section (the CAT type marker)
    assert root.find(".//Layer[@Name='CAT_Section']//UniqueName").text == "Section"
    # read back, sections/ROIs/focus separated (never conflated)
    ann = czi_export.read_cat_annotations(xml)
    assert len(ann["sections"]) == 2
    assert len(ann["rois"]) == 1
    assert len(ann["focus"]) == 2


def test_cat_layers_idempotent():
    skel = ("<ImageDocument><Metadata>"
            "<MetadataNodes><MetadataNode><Layers/></MetadataNode></MetadataNodes>"
            "</Metadata></ImageDocument>")
    secs = [[(0, 0), (10, 0), (10, 10), (0, 10)]]
    xml = czi_export.inject_cat_layers(skel, secs)
    xml = czi_export.inject_cat_layers(xml, secs)   # re-inject
    assert len(czi_export.read_cat_annotations(xml)["sections"]) == 1   # not duplicated


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
    p.sections[0].focus_points = [FocusPoint(1.0, 2.0, 3.0)]       # CZI SupportPoint
    p.sections[0].focus_overview = [(4.0, 4.0)]                    # user-drawn (overview px)
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


def test_manifest_carries_editable_focus():
    """The manifest must export the focus points the user drew/edited in STiM
    (``section.focus_overview``, overview px) — not only the CZI-origin
    ``focus_points`` (stage µm). Regression: these were silently dropped."""
    g = _FakeGeom()
    manifest = wafer_export.build_manifest(_project(), g)
    s1 = next(s for s in manifest["sections"] if s["id"] == "section_1")
    # overview (4,4) -> /zoom(0.5) -> full (8,8) -> *0.001 -> stage µm (0.008, 0.008)
    assert s1["focus_full_px"] == [[8.0, 8.0]]
    assert s1["focus_stage_um"] == [[0.008, 0.008]]
    # the CZI-origin SupportPoint is still carried separately
    assert s1["focus_points_stage_um"] == [{"x_um": 1.0, "y_um": 2.0, "z_um": 3.0}]
    # sections without drawn focus report None (not an empty-list surprise)
    s3 = next(s for s in manifest["sections"] if s["id"] == "section_3")
    assert s3["focus_full_px"] is None and s3["focus_stage_um"] is None


def test_geojson_includes_rois():
    """GeoJSON export must emit ROI polygon features when ROIs are selected
    (the FORMAT_DATA contract claims geojson carries 'rois'). Regression: the
    writer previously emitted only section + fiducial features."""
    g = _FakeGeom()
    manifest = wafer_export.build_manifest(_project(), g)
    fc = wafer_export.build_geojson(manifest, {"sections", "rois", "fiducials"})
    kinds = [f["properties"]["kind"] for f in fc["features"]]
    assert "roi" in kinds, "GeoJSON must carry ROI features"
    rois = [f for f in fc["features"] if f["properties"]["kind"] == "roi"]
    assert len(rois) == 1 and rois[0]["properties"]["id"] == "section_1"
    # ROI ring is closed and in full-res px: overview [[2,2]..] -> /zoom(0.5) -> [[4,4]..]
    ring = rois[0]["geometry"]["coordinates"][0]
    assert ring[0] == [4.0, 4.0] and ring[0] == ring[-1]
    # selecting only sections must NOT emit ROI features
    only_secs = wafer_export.build_geojson(manifest, {"sections"})
    assert all(f["properties"]["kind"] == "section" for f in only_secs["features"])


def test_zen_contour_roi_else_outline():
    g = _FakeGeom()
    manifest = wafer_export.build_manifest(_project(), g)
    # build_manifest now carries per-vertex stage µm for the section outline.
    s3 = next(s for s in manifest["sections"] if s["id"] == "section_3")
    # outline [[40,0],[50,0],[50,10],[40,10]] -> /zoom(0.5) -> *0.001 µm
    assert s3["polygon_stage_um"][0] == [0.08, 0.0]
    assert s3["polygon_stage_um"][2] == [0.1, 0.02]

    with tempfile.TemporaryDirectory() as td:
        path = wafer_export.write_zen_contour(manifest, td)
        assert path.endswith(".contour")
        with open(path, "rb") as f:
            raw = f.read()
        assert b"\r\n" in raw and b"\n\r" not in raw          # CRLF line endings
        lines = raw.decode().strip().split("\r\n")
        # rows in imaging/TSP order: section_3 (S3), section_1 (S2, has ROI), section_2 (S1)
        assert [ln.split(";", 1)[0] for ln in lines] == ["S3", "S2", "S1"]

        # S3 has no ROI -> outline; ring closed (first vertex repeated)
        s3_pts = [tuple(map(float, p.split("|"))) for p in lines[0].split(";")[1:]]
        assert len(s3_pts) == 5 and s3_pts[0] == s3_pts[-1] == (0.08, 0.0)

        # S2 == section_1, which HAS an ROI [[2,2],[6,2],[6,6],[2,6]]
        # -> /0.5 -> *0.001 -> first vertex (0.004, 0.004), closed
        s2_pts = [tuple(map(float, p.split("|"))) for p in lines[1].split(";")[1:]]
        assert s2_pts[0] == (0.004, 0.004) and s2_pts[0] == s2_pts[-1]

    # 'outline' policy ignores the ROI -> S2 uses section_1's outline instead
    with tempfile.TemporaryDirectory() as td:
        path = wafer_export.write_zen_contour(manifest, td, geometry="outline")
        lines = open(path, newline="").read().strip().split("\r\n")
        s2_pts = [tuple(map(float, p.split("|"))) for p in lines[1].split(";")[1:]]
        assert s2_pts[0] == (0.0, 0.0)        # section_1 outline starts at (0,0)


def test_zen_contour_no_geom_returns_none():
    manifest = wafer_export.build_manifest(_project(), None)   # no stage transform
    with tempfile.TemporaryDirectory() as td:
        assert wafer_export.write_zen_contour(manifest, td) is None
        assert not any(fn.endswith(".contour") for fn in os.listdir(td))


def test_fiducial_affine_geom():
    # ground-truth map: µm = px * 0.5 + (1000, -2000), with a 90° rotation mixed in
    px = [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [50.0, 50.0]]
    def truth(x, y):
        return (1000.0 + 0.5 * (-y), -2000.0 + 0.5 * x)     # rotate+scale+translate
    um = [list(truth(x, y)) for x, y in px]
    g = wafer_export.fiducial_affine_geom(px, um)
    assert g.rms_um < 1e-6                                    # exact fit (overdetermined, consistent)
    xs = np.array([10.0, 80.0]); ys = np.array([20.0, 5.0])
    ux, uy = g.full_to_stage_um(xs, ys)
    for i in range(2):
        tx, ty = truth(xs[i], ys[i])
        assert abs(ux[i] - tx) < 1e-6 and abs(uy[i] - ty) < 1e-6
    # plugs into the manifest path for a PNG source (geom=None otherwise)
    proj = _project(); proj.fiducials = px
    manifest = wafer_export.build_manifest(proj, g)
    assert manifest["sections"][0]["polygon_stage_um"] is not None


def test_read_acquisition_on_real_czi():
    """Guarded: only runs if a local CZI with TileRegions is present."""
    import os
    p = "images_local/mSEM706/M411_before_HVC.czi"
    if not os.path.exists(p):
        print("    (skip: real CZI not present)")
        return
    from section_identification import czi_io
    _, geom, _ = czi_io.read_czi_overview(p, target_long_side=2048)
    data = czi_export.read_acquisition_overview(p, geom)
    assert len(data["focus_points"]) >= 1
    assert len(data["regions"]) >= 1
    # overview coords are finite numbers
    assert all(np.isfinite(v) for v in data["focus_points"][0])


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} export tests passed.")


if __name__ == "__main__":
    _run_all()
