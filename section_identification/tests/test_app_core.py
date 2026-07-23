"""Headless tests for StimApp's ROI capture + section-less-ROI promotion.

These use a tiny fake GUI/viewer (no Qt/napari) to exercise the one flow that
must stay exactly in lockstep: promoting a section-less ROI adds a section to BOTH
the model and the GUI 'Sections' layer, so the next ``sync_sections`` (which
rebuilds the model from that layer) preserves every section's state instead of
wiping it on a count mismatch.

Run:  python -m section_identification.tests.test_app_core
"""

from __future__ import annotations

import numpy as np

from section_identification import layer_sync
from section_identification.app_core import StimApp


class _FakeShapes:
    """A napari-ish Shapes layer: ``.data`` is a list of (y, x) vertex arrays."""

    def __init__(self, polys_yx=None):
        self.data = [np.asarray(p, float) for p in (polys_yx or [])]


class _FakeViewer:
    def __init__(self):
        self.layers = {}


def _xy_to_yx(poly_xy):
    p = np.asarray(poly_xy, float).reshape(-1, 2)
    return p[:, ::-1]


def _yx_to_xy(poly_yx):
    p = np.asarray(poly_yx, float).reshape(-1, 2)
    return [[float(x), float(y)] for y, x in p]


class _FakeGui:
    """Minimal stand-in for SectionIdentificationGUI: owns the 'Sections' layer
    and the point→xy plumbing StimApp reads, plus ``add_section_polygons``."""

    def __init__(self, section_polys_xy):
        self.viewer = _FakeViewer()
        self.geom = None
        self.image_path = "/tmp/wafer.czi"
        self.overview = np.zeros((16, 16), dtype=np.uint8)
        self.shapes_layer = _FakeShapes([_xy_to_yx(p) for p in section_polys_xy])
        self.viewer.layers["Sections"] = self.shapes_layer
        self.logs = []

    def current_polygons_xy(self):
        return [_yx_to_xy(d) for d in self.shapes_layer.data if len(np.asarray(d)) >= 3]

    def current_fiducials_xy(self):
        return []

    def add_section_polygons(self, polys_xy):
        polys = [p for p in polys_xy if len(p) >= 3]
        self.shapes_layer.data = list(self.shapes_layer.data) + [_xy_to_yx(p) for p in polys]
        return len(polys)

    def log_msg(self, line):
        self.logs.append(line)


S0 = [[0, 0], [100, 0], [100, 100], [0, 100]]
S1 = [[200, 0], [300, 0], [300, 100], [200, 100]]
INSIDE0 = [[40, 40], [60, 40], [60, 60], [40, 60]]
ORPHAN = [[500, 500], [560, 500], [560, 560], [500, 560]]


def _app_with_rois(roi_polys_xy):
    gui = _FakeGui([S0, S1])
    app = StimApp(gui)
    app.sync_sections()                                   # 2 real sections
    gui.viewer.layers[layer_sync.ROI_LAYER] = _FakeShapes([_xy_to_yx(p) for p in roi_polys_xy])
    return app, gui


def test_capture_promotes_sectionless_roi_and_stays_in_lockstep():
    # An ROI inside s0 and an orphan far outside every section.
    app, gui = _app_with_rois([INSIDE0, ORPHAN])
    app.capture_annotations()

    secs = app.project.sections
    assert len(secs) == 3                                 # a synthetic section was added
    assert secs[0].roi is not None and secs[1].roi is None
    assert secs[2].synthetic is True
    # the orphan is preserved as the synthetic section's ROI, and sits inside it
    assert secs[2].roi is not None
    cx, cy = np.asarray(ORPHAN, float).mean(axis=0)
    x0, y0, x1, y1 = secs[2].bbox()
    assert x0 < cx < x1 and y0 < cy < y1
    # model and GUI layer moved in lockstep (3 == 3)
    assert len(gui.current_polygons_xy()) == 3

    # The crucial guarantee: the next sync_sections must NOT rebuild-and-wipe.
    app.sync_sections()
    secs = app.project.sections
    assert len(secs) == 3
    assert secs[0].roi is not None                        # real ROI survived
    assert secs[2].synthetic is True and secs[2].roi is not None  # promoted one too


def test_capture_is_idempotent_no_duplicate_sections():
    # Re-capturing the same layers must not keep spawning new synthetic sections:
    # the promoted ROI now lives inside its synthetic section, so it re-homes there.
    app, gui = _app_with_rois([INSIDE0, ORPHAN])
    app.capture_annotations()
    n_after_first = len(app.project.sections)
    # rebuild the ROI layer from the model (what layer_sync.show_rois would do)
    roi_polys = [s.roi.polygon for s in app.project.sections if s.roi]
    gui.viewer.layers[layer_sync.ROI_LAYER] = _FakeShapes([_xy_to_yx(p) for p in roi_polys])
    app.capture_annotations()
    assert len(app.project.sections) == n_after_first     # no new sections
    assert sum(1 for s in app.project.sections if s.synthetic) == 1


def test_extra_roi_sharing_a_section_is_promoted():
    # Two ROIs inside one section: the first keeps the section, the second is
    # preserved as its own synthetic section (never dropped).
    extra0 = [[10, 10], [30, 10], [30, 30], [10, 30]]
    app, _ = _app_with_rois([INSIDE0, extra0])
    app.capture_annotations()
    secs = app.project.sections
    assert len(secs) == 3                                 # s0, s1, + one synthetic
    assert sum(1 for s in secs if s.roi) == 2             # both ROIs kept (s1 had none)
    assert sum(1 for s in secs if s.synthetic) == 1
    # the two ROIs live on different sections (one ROI per section)
    owners = [i for i, s in enumerate(secs) if s.roi]
    assert len(owners) == 2 and owners[0] != owners[1]


def test_synthetic_sections_are_coloured_gold():
    # After promotion, the Sections layer's per-shape edge colour marks which
    # sections were auto-added (gold) vs detected (red).
    app, gui = _app_with_rois([INSIDE0, ORPHAN])
    app.capture_annotations()
    ec = np.asarray(gui.shapes_layer.edge_color, float)
    assert ec.shape == (3, 4)
    assert np.allclose(ec[0], layer_sync.DETECTED_EDGE)    # s0 detected → red
    assert np.allclose(ec[1], layer_sync.DETECTED_EDGE)    # s1 detected → red
    assert np.allclose(ec[2], layer_sync.SYNTHETIC_EDGE)   # promoted → gold


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} app_core tests passed.")


if __name__ == "__main__":
    _run_all()
