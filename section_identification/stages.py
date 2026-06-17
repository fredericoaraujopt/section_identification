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
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                            QFormLayout, QFrame, QHBoxLayout, QLabel,
                            QProgressBar, QPushButton, QScrollArea, QSpinBox,
                            QTabWidget, QTextEdit, QVBoxLayout, QWidget)

from . import (compute_broker, czi_export, export as legacy_export, imaging_path,
               layer_sync, roi as roi_mod, stage_help, wafer_export)
from .app_core import StimApp
from .nav import FovNavigator
from .section_table import SectionTableDock
from .wafer_model import QCResult, RoiTemplate
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
        # user-defined focus points (overview px) if present, else an fc×fr grid
        sps = None
        if s.focus_overview:
            fa = np.asarray(s.focus_overview, float).reshape(-1, 2)
            ffx, ffy = geom.ds_to_full(fa[:, 0], fa[:, 1])
            fsu = geom.full_to_stage_um(np.ravel(ffx), np.ravel(ffy))
            if fsu is not None:
                sps = [(float(px), float(py), z)
                       for px, py in zip(np.ravel(fsu[0]), np.ravel(fsu[1]))]
        if not sps:
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


def run_export(app, write_contour: bool = True):
    """File-level wafer export (manifest + CSV + mVis region_names.csv, IDs in
    imaging/TSP order; optional ZEN .contour). Driven from the shell header so it
    isn't tied to any single stage panel."""
    if not app.has_image():
        app.log("export", "load an image and detect sections first.")
        return
    proj = app.sync_sections()
    geom = app.geom
    counts = {}
    tmpl = proj.roi_templates[0] if proj.roi_templates else None
    if tmpl is not None and tmpl.tile_um and geom is not None:
        _, counts = build_tile_region_specs(proj, geom, tmpl.tile_um[0],
                                            tmpl.focus_cols, tmpl.focus_rows, 0.0)
    manifest = wafer_export.build_manifest(proj, geom, mfov_counts=counts)
    try:
        out_dir = legacy_export.resolve_export_dir(app.image_path, None)
    except Exception:
        out_dir = os.path.dirname(app.image_path or ".")
    paths = wafer_export.write_all(
        manifest, out_dir, adapters=["json_manifest", "csv_table", "mvis_lmb"])
    if write_contour:
        try:
            cpath = wafer_export.write_zen_contour(manifest, out_dir)
        except Exception as e:
            cpath = None
            app.log("export", f"⚠️ ZEN .contour failed: {e}")
        if cpath:
            paths["zen_contour"] = cpath
        elif geom is None:
            app.log("export", "⚠️ ZEN .contour skipped — no stage-µm transform "
                    "(CZI source, or calibrate fiducials to stage µm for a PNG/LM image).")
    app.log("export", f"wrote to {out_dir}: {[os.path.basename(p) for p in paths.values()]}")


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
        self.btn_run.setToolTip("Score every detected section for debris, folds, "
                                "shredding and chattering on a downscaled crop, then "
                                "colour the wafer by severity.")
        self.btn_run.clicked.connect(self.run_qc)
        row.addWidget(self.btn_run)
        self.cb_color = QComboBox()
        self.cb_color.addItems(["colour by score", "colour by status"])
        self.cb_color.setToolTip("Colour sections by aggregate QC score, or by "
                                 "accept / review / reject status.")
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
        btn_cal.setToolTip("Set each detector's reference from the section "
                           "population (a high percentile) so only outliers flag. "
                           "Run QC first.")
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
        sp.setMaximumWidth(90)            # keep the QC form compact (no h-scroll)
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
        self._header("<b>ROIs</b> — define a region, propagate it to every section")
        self.lbl_fid = QLabel("")
        self.lbl_fid.setWordWrap(True)
        self.col.addWidget(self.lbl_fid)

        # --- Region of interest: draft (draw or SAM) -> define+propagate ---
        self.col.addWidget(QLabel("<b>Region of interest</b>"))
        self.btn_roi_draft = QPushButton("Draw ROI (draft)")
        self.btn_roi_draft.setToolTip("Add a layer and draw ONE polygon inside a "
                                      "section to use as the ROI template.")
        self.btn_roi_draft.clicked.connect(self._new_draft)
        self.col.addWidget(self.btn_roi_draft)
        self.btn_sam = QPushButton("SAM-assist: trace ROI (off)")
        self.btn_sam.setCheckable(True)
        self.btn_sam.setToolTip("Toggle the SAM editor to one-click an ROI inside a "
                                "section; the mask lands on the ROI draft (not "
                                "Sections). Click again to stop — like the manual "
                                "detector toggle.")
        self.btn_sam.toggled.connect(self._toggle_sam)
        self.col.addWidget(self.btn_sam)
        fit_row = QHBoxLayout()
        fit_row.addWidget(QLabel("fit:"))
        self.cb_fit = QComboBox()
        self.cb_fit.addItems(["template", "full", "percent", "clip"])
        self.cb_fit.setToolTip("How the ROI is fit to each (esp. smaller, coming-in) "
                               "section: keep template / scale to full extent / to a "
                               "percentage / clip to the section.")
        fit_row.addWidget(self.cb_fit, 1)
        self.sp_pct = QDoubleSpinBox()
        self.sp_pct.setRange(1, 100); self.sp_pct.setValue(80); self.sp_pct.setSuffix(" %")
        self.sp_pct.setMaximumWidth(72)
        fit_row.addWidget(self.sp_pct)
        self.col.addLayout(fit_row)
        self.btn_roi_def = QPushButton("Define + propagate ROI")
        self.btn_roi_def.setToolTip("Use the drafted ROI as a template and place it on "
                                    "every section via that section's pose, fit per the "
                                    "mode above.")
        self.btn_roi_def.clicked.connect(self._propagate)
        self.col.addWidget(self.btn_roi_def)

        # --- Focus points: same draft -> define+propagate pattern ---
        self.col.addWidget(QLabel("<b>Focus points</b>"))
        self.btn_focus_draft = QPushButton("Place focus points (draft)")
        self.btn_focus_draft.setToolTip("Add a Points layer; drop autofocus support "
                                        "points inside ONE reference section.")
        self.btn_focus_draft.clicked.connect(self._new_focus_draft)
        self.col.addWidget(self.btn_focus_draft)
        self.btn_focus_def = QPushButton("Define + propagate focus")
        self.btn_focus_def.setToolTip("Place the drafted focus points on every section "
                                      "via its pose; fine-tune by editing the 'Focus "
                                      "points' layer in napari. If none are placed, an "
                                      "auto grid (below) is used.")
        self.btn_focus_def.clicked.connect(self._define_focus)
        self.col.addWidget(self.btn_focus_def)

        # --- mFOV tiling preview ---
        self.col.addWidget(QLabel("<b>mFOV tiling</b>"))
        mrow = QHBoxLayout()
        self.chk_mfov = QCheckBox("Show mFOV grid")
        self.chk_mfov.setToolTip("Preview the tile grid ZEN will image inside each ROI "
                                 "for the tile size below. Single-beam: one mFOV per ROI "
                                 "when the tile is ≥ the ROI.")
        self.chk_mfov.toggled.connect(self._toggle_mfov)
        mrow.addWidget(self.chk_mfov)
        mrow.addWidget(QLabel("tile µm:"))
        self.sp_tile = QDoubleSpinBox()
        self.sp_tile.setRange(1, 5000); self.sp_tile.setValue(50); self.sp_tile.setMaximumWidth(80)
        self.sp_tile.setToolTip("mFOV/tile footprint in stage µm. For single-beam, set "
                                "≥ the ROI for one tile per ROI.")
        self.sp_tile.valueChanged.connect(lambda *_: self._refresh_mfov())
        mrow.addWidget(self.sp_tile)
        mrow.addStretch(1)
        self.col.addLayout(mrow)
        frow = QHBoxLayout()
        frow.addWidget(QLabel("auto-focus grid:"))
        self.sp_fc = QSpinBox(); self.sp_fc.setRange(1, 8); self.sp_fc.setValue(2)
        self.sp_fc.setMaximumWidth(56)
        self.sp_fr = QSpinBox(); self.sp_fr.setRange(1, 8); self.sp_fr.setValue(2)
        self.sp_fr.setMaximumWidth(56)
        self.sp_z = QDoubleSpinBox(); self.sp_z.setRange(-1e6, 1e6); self.sp_z.setValue(0)
        self.sp_z.setMaximumWidth(80)
        for w in (self.sp_fc, self.sp_fr):
            w.setToolTip("Autofocus support-point grid used only when no manual focus "
                         "points are placed.")
        frow.addWidget(self.sp_fc); frow.addWidget(QLabel("×")); frow.addWidget(self.sp_fr)
        frow.addWidget(QLabel("Z µm:")); frow.addWidget(self.sp_z)
        frow.addStretch(1)
        self.col.addLayout(frow)

        b0 = QPushButton("Read existing mFOVs + focus from CZI")
        b0.setToolTip("Read ZEN TileRegions + focus SupportPoints already in the CZI "
                      "and display them on the wafer.")
        b0.clicked.connect(self._read_existing)
        self.col.addWidget(b0)

        self.lbl = QLabel("Define an ROI (and optional focus points) on one section — "
                          "both propagate to all sections by pose. Write them into a "
                          "CZI / .contour from <b>File → Export</b>.")
        self.lbl.setWordWrap(True)
        self.col.addWidget(self.lbl)
        self.col.addStretch(1)
        self.refresh_fiducials()

    def _refresh_mfov(self):
        if getattr(self, "chk_mfov", None) is not None and self.chk_mfov.isChecked():
            self._toggle_mfov(True)

    def _toggle_mfov(self, on):
        if not on:
            layer_sync.clear_mfov_grid(self.app)
            return
        self.app.sync_sections()
        if not any(s.roi and s.roi.polygon for s in self.app.project.sections):
            self.app.log(self.STAGE, "define + propagate an ROI first.")
            self.chk_mfov.blockSignals(True)
            self.chk_mfov.setChecked(False)
            self.chk_mfov.blockSignals(False)
            return
        layer_sync.show_mfov_grid(self.app, self.sp_tile.value())

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
            # Prefer the CZI's CAT ROI/focus annotations (the format we export);
            # attach them to the sections so they're editable + re-exportable.
            n_roi, n_focus = self.app.restore_annotations_from_czi()
            if n_roi or n_focus:
                layer_sync.show_rois(self.app)
                layer_sync.show_focus_points(self.app)
                self.app.save_workflow()
                self.app.log(self.STAGE, f"loaded {n_roi} ROIs + {n_focus} focus "
                                         "points from CZI CAT annotations.")
                return
            # Legacy fallback: stage-µm TileRegions from older STiM exports.
            data = czi_export.read_acquisition_overview(self.app.image_path, self.app.geom)
            layer_sync.show_existing_acquisition(self.app, data)
            self.app.log(self.STAGE, f"read {len(data['focus_points'])} focus points, "
                                     f"{len(data['regions'])} mFOV region(s) from CZI "
                                     "(legacy TileRegions).")
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

    def _toggle_sam(self, on):
        ed = self._editor()
        if ed is None:
            self.btn_sam.setChecked(False)
            return
        if on:
            if not self._need_image():
                self.btn_sam.setChecked(False)
                return
            self._new_draft()                 # ensure the ROI draft layer exists
            try:
                ed.deactivate()               # clean slate, then route commits to us
            except Exception:
                pass
            ed.activate(commit_target=self._capture_roi)
            self.btn_sam.setText("SAM-assist: tracing… (click to stop)")
            self.app.log(self.STAGE, "SAM-assist ON — hover inside the ROI, SPACE to "
                                     "capture it onto the ROI draft, then 'Define + propagate'.")
        else:
            try:
                ed.deactivate()
            except Exception:
                pass
            self.btn_sam.setText("SAM-assist: trace ROI (off)")
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

    FOCUS_DRAFT = "Focus draft"

    def _new_focus_draft(self):
        v = self.app.viewer
        if v is None:
            return
        if self.FOCUS_DRAFT in v.layers:
            v.layers.remove(self.FOCUS_DRAFT)
        try:
            v.add_points(np.empty((0, 2)), name=self.FOCUS_DRAFT, size=8,
                         face_color="orange", border_color="orange",
                         scale=self.app.layer_scale())
            self.app.log(self.STAGE, "drop focus points inside one reference section, "
                                     "then 'Focus pts: define + propagate'.")
        except Exception as e:
            self.app.log(self.STAGE, f"focus draft error: {e}")

    def _define_focus(self):
        if not self._need_image():
            return
        self.app.sync_sections()
        self.app.ensure_poses()
        v = self.app.viewer
        if (v is None or self.FOCUS_DRAFT not in v.layers
                or len(v.layers[self.FOCUS_DRAFT].data) == 0):
            self.app.log(self.STAGE, "place focus points on the 'Focus draft' layer first.")
            return
        pts_xy = _napari_to_xy(np.asarray(v.layers[self.FOCUS_DRAFT].data))
        ref = self._reference_section(pts_xy)
        if ref is None:
            self.app.log(self.STAGE, "no reference section under the focus points.")
            return
        proj = self.app.project
        tmpl = proj.roi_templates[0] if proj.roi_templates else RoiTemplate(ref_section_id=ref.id)
        tmpl.focus_local = roi_mod.focus_template_from_points(ref.pose, pts_xy)
        if not proj.roi_templates:
            proj.roi_templates = [tmpl]
        roi_mod.propagate_all(tmpl, proj.sections)
        layer_sync.show_focus_points(self.app)
        self.app.log(self.STAGE, f"focus points propagated ({len(pts_xy)} per section, "
                                 f"ref {ref.id}). Edit the 'Focus points' layer to fine-tune.")
        self.app.save_workflow()

    def _new_draft(self):
        v = self.app.viewer
        if v is None:
            return
        if self.DRAFT in v.layers:
            v.layers.remove(self.DRAFT)
        try:
            v.add_shapes(name=self.DRAFT, shape_type="polygon", edge_color="magenta",
                         face_color="transparent", edge_width=layer_sync._overlay_width(self.app),
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
        self.btn_sift.setToolTip("Match every section pair with full-resolution SIFT "
                                 "(rotation-invariant) and recover the slice series; "
                                 "draws match lines + the recovered chain.")
        self.btn_sift.clicked.connect(self.run_sift)
        self.col.addWidget(self.btn_sift)
        self.btn_tsp = QPushButton("② Compute imaging route (TSP, min travel)")
        self.btn_tsp.setToolTip("Order imaging to minimise total stage travel "
                                "(open-path TSP) over section centroids; draws the route.")
        self.btn_tsp.clicked.connect(self.run_tsp)
        self.col.addWidget(self.btn_tsp)
        num_row = QHBoxLayout()
        self.btn_serial_num = QPushButton("Show serial #")
        self.btn_serial_num.setToolTip("Label each section with its serial-order number "
                                       "on the wafer (hides the outlines so numbers read "
                                       "cleanly).")
        self.btn_serial_num.clicked.connect(self._show_serial_numbers)
        num_row.addWidget(self.btn_serial_num)
        self.chk_outlines = QCheckBox("outlines")
        self.chk_outlines.setChecked(True)
        self.chk_outlines.setToolTip("Show/hide the section outlines (turn off to read the "
                                     "order numbers without clutter).")
        self.chk_outlines.toggled.connect(
            lambda on: layer_sync.set_sections_visible(self.app, on))
        num_row.addWidget(self.chk_outlines)
        self.col.addLayout(num_row)
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
        _route_tips = {
            "earlier": "Move the selected section one step earlier in the imaging route.",
            "later": "Move the selected section one step later in the imaging route.",
            "drop": "Remove the selected section from the imaging route.",
            "reverse": "Reverse the whole imaging route order.",
        }
        for label, op in (("◀ earlier", "earlier"), ("later ▶", "later"),
                          ("drop", "drop"), ("reverse", "reverse")):
            b = QPushButton(label)
            b.setToolTip(_route_tips[op])
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
        self.lbl = QLabel("SIFT matches every section pair (rotation-invariant) "
                          "and recovers the slice series; TSP then orders imaging "
                          "to minimise stage travel. Export (top bar) renumbers "
                          "IDs to the TSP order (ZEN images by ID).")
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
        self._reflect_outlines_off()
        self.table.refresh()
        unit = "µm" if geom is not None else "px"
        self.lbl_travel.setText(f"Route: {len(order)} stops, total travel "
                                f"{total:,.0f} {unit}")
        self.app.log(self.STAGE, f"TSP route computed: {total:,.0f} {unit} travel.")
        self.app.save_workflow()

    def _show_serial_numbers(self):
        self.app.sync_sections()
        layer_sync.show_serial_numbers(self.app)
        self._reflect_outlines_off()

    def _reflect_outlines_off(self):
        self.chk_outlines.blockSignals(True)
        self.chk_outlines.setChecked(False)
        self.chk_outlines.blockSignals(False)

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

# --------------------------------------------------------------------------- #
# tab shell
# --------------------------------------------------------------------------- #
def attach_workflow(viewer, gui):
    """Build the 4-tab shell (Sections=existing GUI) + a file-level header
    (import/export) + bottom section table + a single shared log. Returns the
    StimApp. Raises on failure (caller guards)."""
    app = StimApp(gui)
    nav = FovNavigator(app)
    table = SectionTableDock(app, nav)

    tabs = QTabWidget()
    gui_scroll = QScrollArea()
    gui_scroll.setWidgetResizable(True)
    gui_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    gui_scroll.setWidget(gui)
    tabs.addTab(gui_scroll, "① Sections")
    rois = StageROIs(app, nav, table)
    qc = StageQC(app, nav, table)
    reorder = StageReorder(app, nav, table)
    tabs.addTab(_wrap(rois), "② ROIs")
    tabs.addTab(_wrap(qc), "③ QC")
    tabs.addTab(_wrap(reorder), "④ Reorder")

    # ---- the ONE log: mirror the GUI's log (which also captures detector
    # stdout via log_msg->print->write->append) into the footer; hide the
    # Sections-tab copy + its "Log" label + the legacy per-stage Export section. ----
    log_widget = QTextEdit()
    log_widget.setReadOnly(True)
    log_widget.setMinimumHeight(120)
    try:
        if getattr(gui, "log", None) is not None:
            _orig_append = gui.log.append

            def _mirror(text, _o=_orig_append):
                _o(text)
                try:
                    log_widget.append(str(text).rstrip())
                except Exception:
                    pass
            gui.log.append = _mirror
        for attr in ("log", "_log_label", "_export_toggle", "_export_body"):
            w = getattr(gui, attr, None)
            if w is not None:
                w.setVisible(False)
    except Exception:
        pass

    container = QWidget()
    cl = QVBoxLayout(container)
    cl.setContentsMargins(4, 4, 4, 4)
    cl.setSpacing(6)
    cl.addWidget(tabs, 1)

    # ---- file-level actions, clearly separated at the bottom (not crammed under
    # the dock title): a single global Import / Export for the whole wafer. ----
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setStyleSheet("color:#444;")
    cl.addWidget(sep)
    filebar = QHBoxLayout()
    filebar.addWidget(QLabel("File:"))
    btn_open = QPushButton("Open image…")
    btn_open.setToolTip("Open a wafer image (CZI whole-slide, or PNG/montage).")
    btn_open.clicked.connect(lambda: getattr(gui, "select_image", lambda: None)())
    btn_export = QPushButton("Export…")
    btn_export.setToolTip("Global export: choose which data (sections, ROIs, focus, "
                          "order, QC…) and which file formats to write.")
    state = {"dlg": None}

    def _open_export():
        from .export_dialog import ExportDialog
        state["dlg"] = ExportDialog(app, parent=container)
        state["dlg"].show()
    btn_export.clicked.connect(_open_export)
    filebar.addWidget(btn_open)
    filebar.addWidget(btn_export)
    filebar.addStretch(1)
    cl.addLayout(filebar)

    loaded = {"for": None}

    def _on_tab(_i):
        try:
            app.sync_sections()
            if app.has_image() and loaded["for"] != app.image_path:
                if app.load_workflow():
                    app.log("io", "restored saved workflow results.")
                else:
                    # No sidecar: fall back to the CZI's own CAT ROI/focus
                    # annotations so an annotated CZI reloads ROIs + focus.
                    n_roi, n_focus = app.restore_annotations_from_czi()
                    if n_roi or n_focus:
                        layer_sync.show_rois(app)
                        layer_sync.show_focus_points(app)
                        app.log("io", f"restored {n_roi} ROIs + {n_focus} focus "
                                      "points from CZI annotations.")
                loaded["for"] = app.image_path
            table.refresh()
            rois.refresh_fiducials()
        except Exception:
            pass
    tabs.currentChanged.connect(_on_tab)

    dock = viewer.window.add_dock_widget(container, name="STiM", area="right")
    viewer.window.add_dock_widget(table, name="Sections", area="bottom")
    viewer.window.add_dock_widget(log_widget, name="Workflow log", area="bottom")
    # open narrower (still user-draggable afterwards)
    try:
        dock.setMaximumWidth(300)
        QTimer.singleShot(400, lambda: dock.setMaximumWidth(16777215))
    except Exception:
        pass
    return app


def _wrap(widget):
    sc = QScrollArea()
    sc.setWidgetResizable(True)
    sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)   # reflow to width; no h-scroll
    sc.setWidget(widget)
    return sc
