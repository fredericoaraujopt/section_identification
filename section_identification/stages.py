"""Workflow stage controllers (QC / ROIs / Reorder) + the 4-tab shell.

These are additive napari dock panels that consume the StimApp facade; the
existing SectionIdentificationGUI is untouched and becomes the "Sections" tab.
Each stage shows its computation on the wafer (layer_sync overlays), narrates to
the shared log, and runs heavy work in a streaming subprocess (worker_harness).

``attach_workflow(viewer, gui)`` builds everything and docks it; it is called
from interface.main() inside a try/except so any failure here can never stop the
core app from launching.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
from qtpy.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
                            QProgressBar, QPushButton, QScrollArea, QSpinBox,
                            QTabWidget, QVBoxLayout, QWidget)

from . import (compute_broker, czi_export, imaging_path, layer_sync, roi as roi_mod,
               wafer_export)
from .app_core import StimApp
from .nav import FovNavigator
from .section_table import SectionTableDock
from .wafer_model import QCResult
from .worker_harness import StreamWorker


def _napari_to_xy(poly_yx):
    p = np.asarray(poly_yx, float).reshape(-1, 2)
    return p[:, ::-1]


class _StageBase(QWidget):
    def __init__(self, app, nav, table):
        super().__init__()
        self.app = app
        self.nav = nav
        self.table = table
        self.worker = StreamWorker(self)
        self.col = QVBoxLayout(self)
        self.col.setContentsMargins(8, 8, 8, 8)
        self.col.setSpacing(6)
        self.progress = QProgressBar()
        self.progress.setVisible(False)

    def _need_image(self) -> bool:
        if not self.app.has_image():
            self.app.log(self.STAGE, "load an image and detect sections first.")
            return False
        return True

    def _set_progress(self, done, total):
        if total:
            self.progress.setVisible(True)
            self.progress.setRange(0, int(total))
            self.progress.setValue(int(done))


# --------------------------------------------------------------------------- #
# QC stage
# --------------------------------------------------------------------------- #
class StageQC(_StageBase):
    STAGE = "qc"

    def __init__(self, app, nav, table):
        super().__init__(app, nav, table)
        self.col.addWidget(QLabel("<b>Quality control</b> — debris · folds · "
                                  "shredding · chattering"))
        row = QHBoxLayout()
        self.btn_run = QPushButton("Run QC")
        self.btn_run.clicked.connect(self.run_qc)
        row.addWidget(self.btn_run)
        self.cb_color = QComboBox()
        self.cb_color.addItems(["colour by score", "colour by status"])
        self.cb_color.currentIndexChanged.connect(self._recolor)
        row.addWidget(self.cb_color)
        self.col.addLayout(row)
        self.lbl = QLabel("Scores each detected section on a downscaled crop and "
                          "colours the wafer by severity. Tune thresholds, then "
                          "re-colour instantly (raw features are cached).")
        self.lbl.setWordWrap(True)
        self.col.addWidget(self.lbl)
        self.col.addWidget(self.progress)
        self.col.addStretch(1)

    def run_qc(self):
        if not self._need_image() or self.worker.is_running():
            return
        proj = self.app.sync_sections()
        if not proj.sections:
            self.app.log(self.STAGE, "no sections to score.")
            return
        plan = compute_broker.qc_plan(compute_broker.get_profile())
        self.app.log(self.STAGE, f"scoring {len(proj.sections)} sections "
                                 f"@ {plan['working_long_side']}px …")
        spath = self.app.write_sections_tempfile()
        args = ["--image", self.app.image_path, "--sections", spath,
                "--target", self.app.target_long_side(),
                "--long-side", plan["working_long_side"]]
        self.worker.start("qc_worker", args, handlers={
            "PROGRESS": lambda p: self._set_progress(p.get("done", 0), p.get("total", 0)),
            "QC": self._on_qc,
            "QC_DONE": self._on_done,
            "ERROR": lambda p: self.app.log(self.STAGE, f"error: {p}"),
        }, on_log=lambda m: self.app.log(self.STAGE, m))

    def _on_qc(self, payload):
        if not payload or "error" in payload:
            return
        s = self.app.project.get(payload.get("section_id"))
        if s is not None:
            s.qc = QCResult(scores=payload.get("scores", {}), flags=payload.get("flags", {}),
                            features=payload.get("features", {}),
                            params_used=payload.get("params_used", {}))

    def _on_done(self, payload):
        self.progress.setVisible(False)
        flagged = sum(1 for s in self.app.project.sections if s.qc and s.qc.flags.get("any"))
        self.app.log(self.STAGE, f"done — {flagged} section(s) flagged.")
        self._recolor()
        self.table.refresh()

    def _recolor(self):
        by = "qc_status" if self.cb_color.currentIndex() == 1 else "qc_score"
        layer_sync.apply_qc_colors(self.app, by=by)


# --------------------------------------------------------------------------- #
# ROIs stage
# --------------------------------------------------------------------------- #
class StageROIs(_StageBase):
    STAGE = "rois"
    DRAFT = "ROI draft"

    def __init__(self, app, nav, table):
        super().__init__(app, nav, table)
        self.col.addWidget(QLabel("<b>ROIs</b> — define once, propagate to every "
                                  "section, write mFOVs for ZEN"))
        b1 = QPushButton("① New ROI draft layer (draw one polygon)")
        b1.clicked.connect(self._new_draft)
        self.col.addWidget(b1)

        fit_row = QHBoxLayout()
        fit_row.addWidget(QLabel("fit:"))
        self.cb_fit = QComboBox()
        self.cb_fit.addItems(["template", "full", "percent", "clip"])
        fit_row.addWidget(self.cb_fit)
        self.sp_pct = QDoubleSpinBox()
        self.sp_pct.setRange(1, 100)
        self.sp_pct.setValue(80)
        self.sp_pct.setSuffix(" %")
        fit_row.addWidget(self.sp_pct)
        self.col.addLayout(fit_row)

        b2 = QPushButton("② Define + propagate to all sections")
        b2.clicked.connect(self._propagate)
        self.col.addWidget(b2)

        grid = QHBoxLayout()
        grid.addWidget(QLabel("tile µm:"))
        self.sp_tile = QDoubleSpinBox()
        self.sp_tile.setRange(1, 5000)
        self.sp_tile.setValue(50)
        grid.addWidget(self.sp_tile)
        grid.addWidget(QLabel("focus grid:"))
        self.sp_fc = QSpinBox()
        self.sp_fc.setRange(1, 8)
        self.sp_fc.setValue(2)
        grid.addWidget(self.sp_fc)
        self.sp_fr = QSpinBox()
        self.sp_fr.setRange(1, 8)
        self.sp_fr.setValue(2)
        grid.addWidget(self.sp_fr)
        grid.addWidget(QLabel("Z µm:"))
        self.sp_z = QDoubleSpinBox()
        self.sp_z.setRange(-1e6, 1e6)
        self.sp_z.setValue(0)
        grid.addWidget(self.sp_z)
        self.col.addLayout(grid)

        b3 = QPushButton("③ Write sections + ROI mFOVs into CZI (for ZEN)")
        b3.clicked.connect(self._write_czi)
        self.col.addWidget(b3)

        self.lbl = QLabel("Draw an ROI inside one section, propagate it (each "
                          "section's pose places it correctly), fit coming-in "
                          "sections, then write TileRegions + focus SupportPoints "
                          "into the CZI for ZEN acquisition.")
        self.lbl.setWordWrap(True)
        self.col.addWidget(self.lbl)
        self.col.addStretch(1)

    def _new_draft(self):
        v = self.app.viewer
        if v is None:
            return
        if self.DRAFT in v.layers:
            v.layers.remove(self.DRAFT)
        try:
            v.add_shapes(name=self.DRAFT, shape_type="polygon", edge_color="magenta",
                         face_color="transparent", edge_width=2,
                         scale=self.app.layer_scale())
            self.app.log(self.STAGE, "draw one ROI polygon on the 'ROI draft' layer.")
        except Exception as e:
            self.app.log(self.STAGE, f"draft layer error: {e}")

    def _draft_polygon_xy(self):
        v = self.app.viewer
        if v is None or self.DRAFT not in v.layers:
            return None
        data = v.layers[self.DRAFT].data
        if not data:
            return None
        return _napari_to_xy(data[0])

    def _reference_section(self, roi_xy):
        cx, cy = np.asarray(roi_xy, float).mean(axis=0)
        try:
            from shapely.geometry import Point, Polygon
            pt = Point(cx, cy)
            for s in self.app.project.sections:
                if len(s.polygon) >= 3 and Polygon(s.polygon).buffer(0).contains(pt):
                    return s
        except Exception:
            pass
        # fallback: nearest centroid
        best, bd = None, 1e30
        for s in self.app.project.sections:
            sx, sy = s.centroid()
            d = (sx - cx) ** 2 + (sy - cy) ** 2
            if d < bd:
                best, bd = s, d
        return best

    def _propagate(self):
        if not self._need_image():
            return
        self.app.sync_sections()
        self.app.ensure_poses()
        roi_xy = self._draft_polygon_xy()
        if roi_xy is None:
            self.app.log(self.STAGE, "draw an ROI on the 'ROI draft' layer first.")
            return
        ref = self._reference_section(roi_xy)
        if ref is None:
            self.app.log(self.STAGE, "no reference section found under the ROI.")
            return
        fit = self.cb_fit.currentText()
        tmpl = roi_mod.template_from_polygon(ref.pose, roi_xy, ref_section_id=ref.id,
                                             fit_mode=fit, fit_percent=self.sp_pct.value(),
                                             focus_cols=self.sp_fc.value(),
                                             focus_rows=self.sp_fr.value())
        self.app.project.roi_templates = [tmpl]
        roi_mod.propagate_all(tmpl, self.app.project.sections)
        layer_sync.show_rois(self.app)
        self.app.log(self.STAGE, f"propagated ROI (ref {ref.id}, fit={fit}) to "
                                 f"{sum(1 for s in self.app.project.sections if s.roi)} sections.")

    # ---- CZI write ----
    def _tile_region_specs(self):
        geom = self.app.geom
        if geom is None:
            self.app.log(self.STAGE, "CZI geometry required to write stage-µm mFOVs.")
            return []
        tile = float(self.sp_tile.value())
        fc, fr, z = self.sp_fc.value(), self.sp_fr.value(), float(self.sp_z.value())
        specs = []
        for s in self.app.project.sections:
            if not s.roi or len(s.roi.polygon) < 3:
                continue
            ra = np.asarray(s.roi.polygon, float).reshape(-1, 2)
            fx, fy = geom.ds_to_full(ra[:, 0], ra[:, 1])
            su = geom.full_to_stage_um(np.ravel(fx), np.ravel(fy))
            if su is None:
                continue
            sx, sy = np.ravel(su[0]), np.ravel(su[1])
            x0, y0, x1, y1 = sx.min(), sy.min(), sx.max(), sy.max()
            w, h = max(x1 - x0, tile), max(y1 - y0, tile)
            cols = max(1, math.ceil(w / tile))
            rows = max(1, math.ceil(h / tile))
            sps = []
            for j in range(fr):
                for i in range(fc):
                    fxp = x0 + (i + 0.5) / fc * w
                    fyp = y0 + (j + 0.5) / fr * h
                    sps.append((fxp, fyp, z))
            specs.append({"center_um": ((x0 + x1) / 2, (y0 + y1) / 2),
                          "contour_um": (w, h), "columns": cols, "rows": rows,
                          "z_um": z, "support_points": sps,
                          "name": f"{czi_export.TILE_REGION_PREFIX}{s.id}"})
        return specs

    def _write_czi(self):
        from . import czi_io
        if not self._need_image():
            return
        if not czi_io.is_czi(self.app.image_path):
            self.app.log(self.STAGE, "ROI→CZI export needs a CZI source image.")
            return
        self.app.sync_sections()
        self.app.ensure_poses()
        specs = self._tile_region_specs()
        polys_full = [s.polygon_full(self.app.geom) for s in self.app.project.sections]
        fids_full = []  # fiducials handled by the existing exporter in the Sections tab
        dst = os.path.splitext(self.app.image_path)[0] + "_STiM_acq.czi"
        self.app.log(self.STAGE, f"writing {len(polys_full)} sections + "
                                 f"{len(specs)} mFOV regions → {os.path.basename(dst)} …")
        try:
            report = czi_export.write_annotated_czi(
                self.app.image_path, dst, polys_full, fids_full,
                section_ids=[s.id for s in self.app.project.sections],
                tile_regions=specs)
            self.app.log(self.STAGE, f"CZI written: {report}")
        except Exception as e:
            self.app.log(self.STAGE, f"CZI write failed: {e}")


# --------------------------------------------------------------------------- #
# Reorder + TSP stage
# --------------------------------------------------------------------------- #
class StageReorder(_StageBase):
    STAGE = "reorder"

    def __init__(self, app, nav, table):
        super().__init__(app, nav, table)
        self.col.addWidget(QLabel("<b>Reorder</b> — recover serial order (SIFT) · "
                                  "imaging route (TSP)"))
        self.btn_sift = QPushButton("① Compute serial order (full-res SIFT)")
        self.btn_sift.clicked.connect(self.run_sift)
        self.col.addWidget(self.btn_sift)
        self.btn_tsp = QPushButton("② Compute imaging route (TSP, min travel)")
        self.btn_tsp.clicked.connect(self.run_tsp)
        self.col.addWidget(self.btn_tsp)
        self.lbl = QLabel("SIFT matches every section pair (rotation-invariant) "
                          "and recovers the slice series; TSP then orders imaging "
                          "to minimise stage travel. On CZI export, IDs are "
                          "renumbered to the TSP order (ZEN images by ID).")
        self.lbl.setWordWrap(True)
        self.col.addWidget(self.lbl)
        self.lbl_travel = QLabel("")
        self.col.addWidget(self.lbl_travel)
        self.col.addWidget(self.progress)
        self.col.addStretch(1)

    def run_sift(self):
        if not self._need_image() or self.worker.is_running():
            return
        proj = self.app.sync_sections()
        if len(proj.sections) < 3:
            self.app.log(self.STAGE, "need ≥3 sections to reorder.")
            return
        plan = compute_broker.sift_plan(compute_broker.get_profile(), len(proj.sections))
        self.app.log(self.STAGE, f"SIFT over {len(proj.sections)} sections "
                                 f"({plan['n_pairs']} pairs) …")
        spath = self.app.write_sections_tempfile()
        cache = os.path.join(os.path.dirname(spath), "stim_similarity.npz")
        args = ["--image", self.app.image_path, "--sections", spath,
                "--target", self.app.target_long_side(),
                "--nfeatures", plan["nfeatures"], "--cache", cache]
        self.worker.start("reorder_worker", args, handlers={
            "PROGRESS": lambda p: self._set_progress(p.get("done", 0), p.get("total", 0)),
            "REORDER_PROGRESS": lambda p: self._set_progress(p.get("done", 0), p.get("total", 0)),
            "REORDER_DONE": self._on_reorder,
            "ERROR": lambda p: self.app.log(self.STAGE, f"error: {p}"),
        }, on_log=lambda m: self.app.log(self.STAGE, m))

    def _on_reorder(self, payload):
        self.progress.setVisible(False)
        if not payload:
            return
        order = payload.get("order", [])
        mg = self.app.project.match_graph
        mg.order = list(order)
        mg.method = payload.get("method")
        from .wafer_model import MatchEdge
        mg.edges = [MatchEdge(**e) for e in payload.get("edges", [])]
        for k, sid in enumerate(order):
            s = self.app.project.get(sid)
            if s is not None:
                s.serial_index = k
        layer_sync.show_matches(self.app)
        layer_sync.show_serial_chain(self.app)
        self.table.refresh()
        self.app.log(self.STAGE, f"serial order recovered ({len(order)} sections).")

    def run_tsp(self):
        if not self._need_image():
            return
        proj = self.app.sync_sections()
        geom = self.app.geom
        coords, secs = [], []
        for s in proj.sections:
            c = s.centroid_stage_um(geom) if geom is not None else s.centroid()
            coords.append(c)
            secs.append(s)
        if len(coords) < 2:
            self.app.log(self.STAGE, "need ≥2 sections for a route.")
            return
        order, total = imaging_path.order_by_travel(np.asarray(coords, float))
        for visit, idx in enumerate(order):
            secs[idx].imaging_index = visit
        layer_sync.show_route(self.app)
        self.table.refresh()
        unit = "µm" if geom is not None else "px"
        self.lbl_travel.setText(f"Route: {len(order)} stops, total travel "
                                f"{total:,.0f} {unit}")
        self.app.log(self.STAGE, f"TSP route computed: {total:,.0f} {unit} travel.")


# --------------------------------------------------------------------------- #
# tab shell
# --------------------------------------------------------------------------- #
def attach_workflow(viewer, gui):
    """Build the 4-tab shell (Sections=existing GUI) + bottom section table and
    dock them. Returns the StimApp. Raises on failure (caller guards)."""
    app = StimApp(gui)
    nav = FovNavigator(app)
    table = SectionTableDock(app, nav)

    tabs = QTabWidget()
    gui_scroll = QScrollArea()
    gui_scroll.setWidgetResizable(True)
    gui_scroll.setWidget(gui)
    tabs.addTab(gui_scroll, "① Sections")
    rois = StageROIs(app, nav, table)
    qc = StageQC(app, nav, table)
    reorder = StageReorder(app, nav, table)
    tabs.addTab(_wrap(rois), "② ROIs")
    tabs.addTab(_wrap(qc), "③ QC")
    tabs.addTab(_wrap(reorder), "④ Reorder")

    def _on_tab(_i):
        try:
            app.sync_sections()
            table.refresh()
        except Exception:
            pass
    tabs.currentChanged.connect(_on_tab)

    viewer.window.add_dock_widget(tabs, name="STiM", area="right")
    viewer.window.add_dock_widget(table, name="Sections", area="bottom")
    return app


def _wrap(widget):
    sc = QScrollArea()
    sc.setWidgetResizable(True)
    sc.setWidget(widget)
    return sc
