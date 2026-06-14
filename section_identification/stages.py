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
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                            QFormLayout, QHBoxLayout, QLabel, QProgressBar,
                            QPushButton, QScrollArea, QSpinBox, QTabWidget,
                            QTextEdit, QVBoxLayout, QWidget)

from . import (compute_broker, czi_export, export as legacy_export, imaging_path,
               layer_sync, roi as roi_mod, stage_help, wafer_export)
from .app_core import StimApp
from .nav import FovNavigator
from .section_table import SectionTableDock
from .wafer_model import QCResult
from .worker_harness import StreamWorker


def _napari_to_xy(poly_yx):
    p = np.asarray(poly_yx, float).reshape(-1, 2)
    return p[:, ::-1]


def build_tile_region_specs(project, geom, tile_um, fc, fr, z):
    """ZEN TileRegion specs (stage µm) for every section's ROI, ORDERED BY
    imaging (TSP) order so the region Id sequence == the acquisition route (ZEN
    images by Id). Returns ``(specs, mfov_counts_by_id)``. Empty without geom."""
    if geom is None:
        return [], {}
    tile = float(tile_um) or 50.0
    specs, counts = [], {}
    for i, s in enumerate(project.in_imaging_order(), start=1):
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
        cols, rows = max(1, math.ceil(w / tile)), max(1, math.ceil(h / tile))
        sps = [(x0 + (a + 0.5) / fc * w, y0 + (b + 0.5) / fr * h, z)
               for b in range(fr) for a in range(fc)]
        specs.append({
            "center_um": ((x0 + x1) / 2, (y0 + y1) / 2), "contour_um": (w, h),
            "columns": cols, "rows": rows, "z_um": z, "support_points": sps,
            "id": i,
            "name": f"{czi_export.TILE_REGION_PREFIX}{i:03d}_{wafer_export.serial_name(s)}",
        })
        counts[s.id] = cols * rows
    return specs, counts


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
        self._help_dlg = None

    def _header(self, title_html: str):
        """A stage title row with a '❔' help button (opens stage_help)."""
        row = QHBoxLayout()
        row.addWidget(QLabel(title_html))
        row.addStretch(1)
        btn = QPushButton("❔")
        btn.setFixedWidth(28)
        btn.setToolTip("What this stage does and how to use it.")
        btn.clicked.connect(self._show_help)
        row.addWidget(btn)
        self.col.addLayout(row)

    def _show_help(self):
        self._help_dlg = stage_help.show_help(self, self.STAGE)

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
        self._header("<b>Quality control</b> — debris · folds · shredding · chattering")
        row = QHBoxLayout()
        self.btn_run = QPushButton("Run QC")
        self.btn_run.clicked.connect(self.run_qc)
        row.addWidget(self.btn_run)
        self.cb_color = QComboBox()
        self.cb_color.addItems(["colour by score", "colour by status"])
        self.cb_color.currentIndexChanged.connect(self._recolor)
        row.addWidget(self.cb_color)
        self.col.addLayout(row)

        # live thresholds: re-flag instantly from cached features (no recompute)
        self._building = True
        from .wafer_qc import qc_defaults
        d = qc_defaults()
        form = QFormLayout()
        self.sp_debris = self._ref_spin(0, 1, 0.005, 3, d["debris_ref"])
        self.sp_fold = self._ref_spin(0, 5, 0.05, 2, d["fold_ref"])
        self.sp_shred = self._ref_spin(0, 5, 0.05, 2, d["shred_ref"])
        self.sp_chatter = self._ref_spin(0, 50, 0.5, 1, d["chatter_ref"])
        self.sp_flag = self._ref_spin(0, 1, 0.05, 2, d["flag"])
        form.addRow("debris ref", self.sp_debris)
        form.addRow("fold ref", self.sp_fold)
        form.addRow("shred ref", self.sp_shred)
        form.addRow("chatter ref", self.sp_chatter)
        form.addRow("flag ≥", self.sp_flag)
        self.col.addLayout(form)
        btn_cal = QPushButton("Calibrate thresholds from population")
        btn_cal.clicked.connect(self._calibrate)
        self.col.addWidget(btn_cal)
        self._building = False
        self.chk_diag = QCheckBox("Show diagnostic overlay for the selected section")
        self.chk_diag.setToolTip("On click in the section table, overlay the feature "
                                 "map that produced its dominant flag (ridges/blobs/"
                                 "components) so you see WHY it was flagged.")
        self.chk_diag.toggled.connect(self._toggle_diag)
        self.col.addWidget(self.chk_diag)
        self.table.add_select_listener(self._on_select)
        self.lbl = QLabel("Scores each detected section on a downscaled crop and "
                          "colours the wafer by severity. Tune thresholds, then "
                          "re-colour instantly (raw features are cached).")
        self.lbl.setWordWrap(True)
        self.col.addWidget(self.lbl)
        self.col.addWidget(self.progress)
        self.col.addStretch(1)

    def _ref_spin(self, lo, hi, step, dec, val):
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setDecimals(dec)
        sp.setValue(val)
        sp.valueChanged.connect(self._on_ref_change)
        return sp

    def _refs(self) -> dict:
        return {"debris_ref": self.sp_debris.value(), "fold_ref": self.sp_fold.value(),
                "shred_ref": self.sp_shred.value(), "chatter_ref": self.sp_chatter.value(),
                "flag": self.sp_flag.value()}

    def _on_ref_change(self, *_):
        if not self._building:
            self._rethreshold_all()

    def _rethreshold_all(self):
        from .wafer_qc import rethreshold
        refs = self._refs()
        n = 0
        for s in self.app.project.sections:
            if s.qc is not None:
                s.qc = QCResult(**rethreshold(s.qc.to_dict(), refs))
                n += 1
        if n:
            self._recolor()
            self.table.refresh()
            self.app.save_workflow()

    def _calibrate(self):
        from .wafer_qc import calibrate_qc
        qcs = [s.qc.to_dict() for s in self.app.project.sections if s.qc]
        if not qcs:
            self.app.log(self.STAGE, "run QC first, then calibrate from the results.")
            return
        refs = calibrate_qc(qcs)
        self._building = True
        self.sp_debris.setValue(refs["debris_ref"])
        self.sp_fold.setValue(refs["fold_ref"])
        self.sp_shred.setValue(refs["shred_ref"])
        self.sp_chatter.setValue(refs["chatter_ref"])
        self._building = False
        self._rethreshold_all()
        self.app.log(self.STAGE, f"calibrated thresholds from {len(qcs)} sections.")

    def _on_select(self, section):
        if self.chk_diag.isChecked():
            layer_sync.show_qc_diagnostic(self.app, section)

    def _toggle_diag(self, on):
        if not on:
            layer_sync.clear_qc_diagnostic(self.app)

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
                "--long-side", plan["working_long_side"],
                "--refs", json.dumps(self._refs())]
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
        self.app.save_workflow()

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
        self._header("<b>ROIs</b> — define once, propagate to every section, "
                     "write mFOVs for ZEN")
        self.lbl_fid = QLabel("")
        self.lbl_fid.setWordWrap(True)
        self.col.addWidget(self.lbl_fid)
        b0 = QPushButton("Read existing mFOVs + focus points from CZI")
        b0.clicked.connect(self._read_existing)
        self.col.addWidget(b0)

        b1 = QPushButton("① New ROI draft layer (draw one polygon)")
        b1.clicked.connect(self._new_draft)
        self.col.addWidget(b1)

        sam_row = QHBoxLayout()
        self.btn_sam = QPushButton("①ᴮ SAM-assist: trace ROI")
        self.btn_sam.setToolTip("Use the SAM editor to one-click an ROI (e.g. a "
                                "resin-bounded region); the mask is captured onto "
                                "the ROI draft layer instead of Sections.")
        self.btn_sam.clicked.connect(self._sam_assist)
        sam_row.addWidget(self.btn_sam)
        self.btn_sam_done = QPushButton("Finish SAM-assist")
        self.btn_sam_done.clicked.connect(self._finish_sam)
        sam_row.addWidget(self.btn_sam_done)
        self.col.addLayout(sam_row)

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
        self.refresh_fiducials()

    def refresh_fiducials(self):
        """Recommend marking fiducials when none are on file (anchors stage-µm
        export). Reads the existing GUI's Fiducials layer."""
        n = 0
        try:
            fl = getattr(self.app.gui, "fid_layer", None)
            n = 0 if fl is None else len(fl.data)
        except Exception:
            n = 0
        if n == 0:
            self.lbl_fid.setText("⚠ No fiducials on file — recommended before export. "
                                 "Import them in the Sections tab (CZI Shuttle & Find) "
                                 "or place with the 'm' key in the manual editor.")
            self.lbl_fid.setStyleSheet("QLabel{background:#3a2a16;padding:6px;border-radius:4px;}")
        else:
            self.lbl_fid.setText(f"Fiducials on file: {n}.")
            self.lbl_fid.setStyleSheet("QLabel{background:#16241a;padding:6px;border-radius:4px;}")

    def _read_existing(self):
        from . import czi_io
        if not self._need_image():
            return
        if not czi_io.is_czi(self.app.image_path) or self.app.geom is None:
            self.app.log(self.STAGE, "existing mFOV/focus read needs a CZI with geometry.")
            return
        try:
            data = czi_export.read_acquisition_overview(self.app.image_path, self.app.geom)
            layer_sync.show_existing_acquisition(self.app, data)
            self.app.log(self.STAGE, f"read {len(data['focus_points'])} focus points, "
                                     f"{len(data['regions'])} mFOV region(s) from CZI.")
        except Exception as e:
            self.app.log(self.STAGE, f"read existing acquisition failed: {e}")

    def _editor(self):
        gui = self.app.gui
        ed = getattr(gui, "_napari_editor", None)
        if ed is None:
            try:
                from .napari_sam_editor import NapariSamEditor
                ed = NapariSamEditor(gui)
                gui._napari_editor = ed
            except Exception as e:
                self.app.log(self.STAGE, f"SAM editor unavailable: {e}")
                return None
        return ed

    def _sam_assist(self):
        if not self._need_image():
            return
        ed = self._editor()
        if ed is None:
            return
        self._new_draft()                     # ensure the ROI draft layer exists
        try:
            ed.deactivate()                   # clean slate, then route commits to us
        except Exception:
            pass
        ed.activate(commit_target=self._capture_roi)
        self.app.log(self.STAGE, "SAM-assist ON — hover inside the ROI, SPACE to "
                                 "capture it onto the ROI draft, then 'Finish "
                                 "SAM-assist' and 'Define + propagate'.")

    def _finish_sam(self):
        ed = getattr(self.app.gui, "_napari_editor", None)
        if ed is not None:
            try:
                ed.deactivate()
            except Exception:
                pass
        self.app.log(self.STAGE, "SAM-assist OFF.")

    def _capture_roi(self, poly_yx):
        v = self.app.viewer
        if v is None:
            return
        if self.DRAFT not in v.layers:
            self._new_draft()
        lyr = v.layers[self.DRAFT]
        lyr.data = list(lyr.data) + [np.asarray(poly_yx, float)]
        self.app.log(self.STAGE, f"ROI captured ({len(lyr.data)} on draft).")

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
                                             focus_rows=self.sp_fr.value(),
                                             tile_um=(self.sp_tile.value(), self.sp_tile.value()))
        self.app.project.roi_templates = [tmpl]
        roi_mod.propagate_all(tmpl, self.app.project.sections)
        layer_sync.show_rois(self.app)
        self.app.log(self.STAGE, f"propagated ROI (ref {ref.id}, fit={fit}) to "
                                 f"{sum(1 for s in self.app.project.sections if s.roi)} sections.")
        self.app.save_workflow()

    # ---- CZI write ----
    def _tile_region_specs(self):
        geom = self.app.geom
        if geom is None:
            self.app.log(self.STAGE, "CZI geometry required to write stage-µm mFOVs.")
            return []
        specs, _ = build_tile_region_specs(
            self.app.project, geom, self.sp_tile.value(),
            self.sp_fc.value(), self.sp_fr.value(), float(self.sp_z.value()))
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
        self._header("<b>Reorder</b> — recover serial order (SIFT) · "
                     "imaging route (TSP)")
        self.btn_sift = QPushButton("① Compute serial order (full-res SIFT)")
        self.btn_sift.clicked.connect(self.run_sift)
        self.col.addWidget(self.btn_sift)
        self.btn_tsp = QPushButton("② Compute imaging route (TSP, min travel)")
        self.btn_tsp.clicked.connect(self.run_tsp)
        self.col.addWidget(self.btn_tsp)
        self.btn_inspect = QPushButton("Inspect match: pick 2 sections")
        self.btn_inspect.setToolTip("Then click two rows in the table — the SIFT "
                                    "inlier correspondences are drawn between them.")
        self.btn_inspect.clicked.connect(lambda: self._arm("inspect"))
        self.col.addWidget(self.btn_inspect)
        self.btn_swap = QPushButton("Swap serial order: pick 2 sections")
        self.btn_swap.setToolTip("Click two rows to swap their position in the "
                                 "recovered serial order.")
        self.btn_swap.clicked.connect(lambda: self._arm("swap"))
        self.col.addWidget(self.btn_swap)
        rrow = QHBoxLayout()
        rrow.addWidget(QLabel("route:"))
        for label, op in (("◀ earlier", "earlier"), ("later ▶", "later"),
                          ("drop", "drop"), ("reverse", "reverse")):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, o=op: self._route_op(o))
            rrow.addWidget(b)
        self.col.addLayout(rrow)
        self._pick_mode = None
        self._picks = []
        self.table.add_select_listener(self._on_pick)
        self.btn_heat = QPushButton("Show similarity heatmap")
        self.btn_heat.setToolTip("The SIFT inlier matrix, permuted by the recovered "
                                 "serial order — a correct ordering looks banded.")
        self.btn_heat.clicked.connect(self._show_heatmap)
        self.col.addWidget(self.btn_heat)
        self._heat_dlg = None
        self.btn_export = QPushButton("③ Export wafer manifest + mVis (region_names.csv)")
        self.btn_export.clicked.connect(self.export_wafer)
        self.col.addWidget(self.btn_export)
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
        mg.similarity_path = payload.get("similarity_path")
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
        self.app.save_workflow()

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
        self.app.save_workflow()

    def _arm(self, mode):
        if not self._need_image():
            return
        self._pick_mode = mode
        self._picks = []
        self.app.log(self.STAGE, f"{mode}: click two sections in the table.")

    def _on_pick(self, section):
        if self._pick_mode is None or section is None:
            return
        self._picks.append(section)
        if len(self._picks) < 2:
            return
        a, b = self._picks[0], self._picks[1]
        mode = self._pick_mode
        self._pick_mode = None
        self._picks = []
        if mode == "inspect":
            layer_sync.show_pair_matches(self.app, a, b)
        elif mode == "swap":
            if self.app.project.swap_serial(a.id, b.id):
                layer_sync.show_serial_chain(self.app)
                self.table.refresh()
                self.app.save_workflow()
                self.app.log(self.STAGE, f"swapped serial order: {a.id} ↔ {b.id}.")

    def _route_op(self, op):
        sel = getattr(self.table, "selected_section", None)
        if op == "reverse":
            self.app.project.reverse_imaging()
        elif sel is None:
            self.app.log(self.STAGE, "select a section row in the table first.")
            return
        elif op == "earlier":
            self.app.project.move_imaging(sel.id, -1)
        elif op == "later":
            self.app.project.move_imaging(sel.id, +1)
        elif op == "drop":
            self.app.project.drop_from_imaging(sel.id)
        layer_sync.show_route(self.app)
        self.table.refresh()
        self.app.save_workflow()

    def _show_heatmap(self):
        from . import reorder as reorder_mod
        path = self.app.project.match_graph.similarity_path
        if not path or not os.path.isfile(path):
            self.app.log(self.STAGE, "run SIFT first (no cached similarity matrix).")
            return
        try:
            data = np.load(path, allow_pickle=True)
            sim = data["similarity"]
            ids = list(data["ids"])
            order = self.app.project.match_graph.order
            idx = [ids.index(i) for i in order if i in ids] if order else None
            img = reorder_mod.heatmap_image(sim, idx if idx and len(idx) == len(sim) else None)
            h, w = img.shape
            img = np.ascontiguousarray(img)
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
            dlg = QDialog(self)
            dlg.setWindowTitle("SIFT similarity (serial-ordered)")
            lay = QVBoxLayout(dlg)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            label = QLabel()
            label.setPixmap(QPixmap.fromImage(qimg).scaled(max(w, 400), max(h, 400)))
            scroll.setWidget(label)
            lay.addWidget(scroll)
            dlg.resize(520, 540)
            self._heat_dlg = dlg
            dlg.show()
        except Exception as e:
            self.app.log(self.STAGE, f"heatmap error: {e}")

    def export_wafer(self):
        if not self._need_image():
            return
        proj = self.app.sync_sections()
        geom = self.app.geom
        counts = {}
        tmpl = proj.roi_templates[0] if proj.roi_templates else None
        if tmpl is not None and tmpl.tile_um and geom is not None:
            _, counts = build_tile_region_specs(proj, geom, tmpl.tile_um[0],
                                                tmpl.focus_cols, tmpl.focus_rows, 0.0)
        manifest = wafer_export.build_manifest(proj, geom, mfov_counts=counts)
        try:
            out_dir = legacy_export.resolve_export_dir(self.app.image_path, None)
        except Exception:
            out_dir = os.path.dirname(self.app.image_path or ".")
        paths = wafer_export.write_all(
            manifest, out_dir, adapters=["json_manifest", "csv_table", "mvis_lmb"])
        self.app.log(self.STAGE, f"wafer exported (IDs in imaging order): "
                                 f"{[os.path.basename(p) for p in paths.values()]}")


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

    loaded = {"for": None}

    def _on_tab(_i):
        try:
            app.sync_sections()
            if app.has_image() and loaded["for"] != app.image_path:
                if app.load_workflow():
                    app.log("io", "restored saved workflow results.")
                loaded["for"] = app.image_path
            table.refresh()
            rois.refresh_fiducials()
        except Exception:
            pass
    tabs.currentChanged.connect(_on_tab)

    # shared footer log so messages mirror across all tabs (not just Sections)
    log_widget = QTextEdit()
    log_widget.setReadOnly(True)
    log_widget.setMinimumHeight(70)
    app.add_log_sink(lambda line: log_widget.append(line))

    viewer.window.add_dock_widget(tabs, name="STiM", area="right")
    viewer.window.add_dock_widget(table, name="Sections", area="bottom")
    viewer.window.add_dock_widget(log_widget, name="Workflow log", area="bottom")
    return app


def _wrap(widget):
    sc = QScrollArea()
    sc.setWidgetResizable(True)
    sc.setWidget(widget)
    return sc
