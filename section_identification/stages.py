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
from qtpy.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                            QDoubleSpinBox, QFormLayout, QFrame, QHBoxLayout,
                            QLabel, QProgressBar, QPushButton, QScrollArea,
                            QSpinBox, QTabWidget, QTextEdit, QVBoxLayout, QWidget)

from . import (compute_broker, czi_export, export as legacy_export, imaging_path,
               layer_sync, roi as roi_mod, stage_help, wafer_export)
from .app_core import StimApp
from .nav import FovNavigator
from .section_table import SectionTableDock
from .wafer_model import QCResult, Roi, RoiTemplate
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
    app.capture_annotations()   # preserve section-less ROIs as synthetic sections
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


class _CollapsibleSection(QWidget):
    """A titled header button that shows/hides a content widget — the accordion
    idiom used by the Sections tab (Calibrate / Automatic detector / Manual
    editor). Multiple can be open at once."""

    def __init__(self, title, content, open=True):
        super().__init__()
        self._title = title
        self.content = content
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.toggle = QPushButton(self._label(open))
        self.toggle.setCheckable(True)
        self.toggle.setChecked(open)
        self.toggle.setStyleSheet("QPushButton{text-align:left; font-weight:bold; "
                                  "padding:5px; border:none;}")
        self.toggle.toggled.connect(self._on_toggled)
        self.content.setVisible(open)
        v.addWidget(self.toggle)
        v.addWidget(self.content)

    def _label(self, on):
        return ("▼  " if on else "▶  ") + self._title

    def _on_toggled(self, on):
        self.content.setVisible(on)
        self.toggle.setText(self._label(on))


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

        # Three collapsible sub-stages (accordion), mirroring the Sections tab:
        # ROI placement / focus placement / mFOV visualisation.
        self.sec_roi = _CollapsibleSection("ROI", self._build_roi_tab(), open=True)
        self.sec_focus = _CollapsibleSection("Focus", self._build_focus_tab(), open=False)
        self.sec_mfov = _CollapsibleSection("mFOV", self._build_mfov_tab(), open=False)
        self.sec_roi.toggle.toggled.connect(self._on_roi_collapsed)
        for s in (self.sec_roi, self.sec_focus, self.sec_mfov):
            self.col.addWidget(s)
        self.col.addStretch(1)
        self._auto_running = False
        try:
            self.worker.finished.connect(self._on_roi_worker_finished)
        except Exception:
            pass
        self.refresh_fiducials()

    def _on_roi_collapsed(self, on):
        """Clear the auto-detect grid preview when the ROI section is collapsed."""
        if not on and getattr(self, "chk_auto_preview", None) is not None \
                and self.chk_auto_preview.isChecked():
            self.chk_auto_preview.setChecked(False)

    # ---- sub-tab builders -------------------------------------------------- #
    def _build_roi_tab(self):
        w = QWidget(); c = QVBoxLayout(w)
        c.addWidget(QLabel("<b>Region of interest</b>"))
        self.btn_roi_draft = QPushButton("Draw ROI (draft)")
        self.btn_roi_draft.setToolTip("Add a layer and draw ONE polygon inside a "
                                      "section to use as the ROI template.")
        self.btn_roi_draft.clicked.connect(self._new_draft)
        c.addWidget(self.btn_roi_draft)
        self.btn_sam = QPushButton("SAM-assist: trace ROI (off)")
        self.btn_sam.setCheckable(True)
        self.btn_sam.setToolTip("Toggle the SAM editor to one-click an ROI inside a "
                                "section; the mask lands on the ROI draft (not "
                                "Sections). Click again to stop — like the manual "
                                "detector toggle.")
        self.btn_sam.toggled.connect(self._toggle_sam)
        c.addWidget(self.btn_sam)
        self.btn_manual_roi = QPushButton("Manual ROI per section (SAM): OFF")
        self.btn_manual_roi.setCheckable(True)
        self.btn_manual_roi.setToolTip("Define a personalised ROI directly on each "
                                       "section (no template): pan / double-click table "
                                       "rows to a section, hover inside it and press "
                                       "SPACE to SAM-trace its ROI — written straight to "
                                       "that section (replacing any existing ROI). You "
                                       "can also hand-draw polygons on the '② ROIs' "
                                       "layer with napari's tools; both are saved.")
        self.btn_manual_roi.toggled.connect(self._toggle_manual_roi)
        c.addWidget(self.btn_manual_roi)

        c.addWidget(QLabel("<b>Propagate a template</b>"))
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
        c.addLayout(fit_row)
        self.btn_roi_def = QPushButton("Define + propagate ROI")
        self.btn_roi_def.setToolTip("Use the drafted ROI as a template and place it on "
                                    "every section via that section's pose, fit per the "
                                    "mode above.")
        self.btn_roi_def.clicked.connect(self._propagate)
        c.addWidget(self.btn_roi_def)
        self.btn_roi_center = QPushButton("Propagate ROI to section centers")
        self.btn_roi_center.setToolTip("Place a copy of the drafted ROI — with its "
                                       "drawn dimensions and orientation unchanged — at "
                                       "the centroid of every section. Ignores pose and "
                                       "the fit mode above; use when section polygons "
                                       "vary widely in size and pose-based fitting "
                                       "misplaces the ROI.")
        self.btn_roi_center.clicked.connect(self._propagate_center)
        c.addWidget(self.btn_roi_center)

        self._build_auto_detect_group(c)          # automatic SAM ROI detection
        c.addStretch(1)
        return w

    def _build_focus_tab(self):
        w = QWidget(); c = QVBoxLayout(w)
        c.addWidget(QLabel("<b>Focus points</b>"))
        self.btn_focus_draft = QPushButton("Place focus points (draft)")
        self.btn_focus_draft.setToolTip("Add a Points layer; drop autofocus support "
                                        "points inside ONE reference section.")
        self.btn_focus_draft.clicked.connect(self._new_focus_draft)
        c.addWidget(self.btn_focus_draft)
        self.btn_focus_def = QPushButton("Define + propagate focus")
        self.btn_focus_def.setToolTip("Place the drafted focus points on every section "
                                      "via its pose; fine-tune by editing the 'Focus "
                                      "points' layer in napari. If none are placed, an "
                                      "auto grid (below) is used.")
        self.btn_focus_def.clicked.connect(self._define_focus)
        c.addWidget(self.btn_focus_def)
        self.btn_focus_center = QPushButton("Propagate focus to centers")
        self.btn_focus_center.setToolTip("Place the drafted focus points at each "
                                         "section's centroid unchanged. Each point is "
                                         "anchored to the ROI if it was drawn inside the "
                                         "ROI (so it rides the propagated ROI), else to "
                                         "the section. Use when sections vary widely in "
                                         "size, like the matching ROI option.")
        self.btn_focus_center.clicked.connect(self._define_focus_center)
        c.addWidget(self.btn_focus_center)
        c.addStretch(1)
        return w

    def _build_mfov_tab(self):
        w = QWidget(); c = QVBoxLayout(w)
        c.addWidget(QLabel("<b>mFOV tiling</b>"))
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
        c.addLayout(mrow)
        frow = QHBoxLayout()
        frow.addWidget(QLabel("auto-focus grid:"))
        self.sp_fc = QSpinBox(); self.sp_fc.setRange(1, 8); self.sp_fc.setValue(2)
        self.sp_fc.setMaximumWidth(56)
        self.sp_fr = QSpinBox(); self.sp_fr.setRange(1, 8); self.sp_fr.setValue(2)
        self.sp_fr.setMaximumWidth(56)
        self.sp_z = QDoubleSpinBox(); self.sp_z.setRange(-1e6, 1e6); self.sp_z.setValue(0)
        self.sp_z.setMaximumWidth(80)
        for wd in (self.sp_fc, self.sp_fr):
            wd.setToolTip("Autofocus support-point grid used only when no manual focus "
                          "points are placed.")
        frow.addWidget(self.sp_fc); frow.addWidget(QLabel("×")); frow.addWidget(self.sp_fr)
        frow.addWidget(QLabel("Z µm:")); frow.addWidget(self.sp_z)
        frow.addStretch(1)
        c.addLayout(frow)

        b0 = QPushButton("Read existing mFOVs + focus from CZI")
        b0.setToolTip("Read ZEN TileRegions + focus SupportPoints already in the CZI "
                      "and display them on the wafer.")
        b0.clicked.connect(self._read_existing)
        c.addWidget(b0)

        self.lbl = QLabel("Define an ROI (and optional focus points) on one section — "
                          "both propagate to all sections by pose. Write them into a "
                          "CZI / .contour from <b>File → Export</b>.")
        self.lbl.setWordWrap(True)
        c.addWidget(self.lbl)
        c.addStretch(1)
        return w

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

    def _toggle_manual_roi(self, on):
        """Per-section manual ROI mode: route SAM commits straight to the section
        under the cursor (not the template draft), and keep an editable '② ROIs'
        layer around for hand-drawing. Mutually exclusive with SAM-assist draft."""
        ed = self._editor()
        if ed is None:
            self.btn_manual_roi.setChecked(False)
            return
        if on:
            if not self._need_image():
                self.btn_manual_roi.setChecked(False)
                return
            if self.btn_sam.isChecked():          # don't fight the draft SAM toggle
                self.btn_sam.setChecked(False)
            self.app.sync_sections()
            self.app.ensure_poses()
            layer_sync.ensure_roi_layer(self.app)
            try:
                ed.deactivate()
            except Exception:
                pass
            ed.activate(commit_target=self._capture_manual_roi)
            self.btn_manual_roi.setText("● Manual ROI per section (SAM): ON — click to stop")
            self.app.log(self.STAGE, "Manual ROI ON — hover inside a section, SPACE to "
                                     "SAM-trace its ROI. Double-click table rows / pan to "
                                     "move between sections.")
        else:
            try:
                ed.deactivate()
            except Exception:
                pass
            self.btn_manual_roi.setText("Manual ROI per section (SAM): OFF")
            self.app.log(self.STAGE, "Manual ROI OFF.")

    def _capture_manual_roi(self, poly_yx):
        """Commit a SAM-traced polygon as the ROI of the section it overlaps. A
        trace made where no section was detected is preserved by promoting it to
        its own margined section, rather than overwriting the nearest section's
        ROI. Appends the one shape to the overlay (O(1)) instead of rebuilding it,
        so defining many ROIs one-by-one stays fast."""
        poly_xy = np.asarray(_napari_to_xy(np.asarray(poly_yx, float)), float).reshape(-1, 2)
        self.app.sync_sections()
        sec, promoted = self.app.assign_or_promote_roi(
            [[float(x), float(y)] for x, y in poly_xy])
        if sec is None:
            self.app.log(self.STAGE, "couldn't place that ROI.")
            return
        layer_sync.append_roi(self.app, poly_xy)       # incremental — no full rebuild
        self.app.save_workflow()
        total = sum(1 for s in self.app.project.sections if s.roi)
        if promoted:
            self.app.log(self.STAGE, f"ROI was outside every section — created section "
                                     f"{sec.id} around it ({total} total).")
        else:
            self.app.log(self.STAGE, f"ROI set on section {sec.id} ({total} total).")

    FOCUS_DRAFT = "Focus draft"

    def _new_focus_draft(self):
        v = self.app.viewer
        if v is None:
            return
        if self.FOCUS_DRAFT in v.layers:
            v.layers.remove(self.FOCUS_DRAFT)
        try:
            v.add_points(np.empty((0, 2)), name=self.FOCUS_DRAFT, size=2,
                         face_color="orange", border_color="orange",
                         scale=self.app.layer_scale(),
                         metadata={"stim_screen_pts": True})
            sync = getattr(self.app.gui, "_sync_outline_widths", None)
            if sync is not None:
                try:
                    sync()
                except Exception:
                    pass
            self.app.log(self.STAGE, "drop focus points inside one reference section, "
                                     "then 'Focus pts: define + propagate'.")
        except Exception as e:
            self.app.log(self.STAGE, f"focus draft error: {e}")

    def _define_focus(self):
        self._propagate_focus(mode="pose")

    def _define_focus_center(self):
        self._propagate_focus(mode="center")

    def _propagate_focus(self, mode: str):
        """Propagate drafted focus points to every section **that has an ROI**. Each
        point is anchored to the ROI (if drawn inside it) or the section (if
        outside): ``pose`` maps its relative offset through the section's pose
        (rotation-aware — an ROI's top-left → every ROI's top-left); ``center``
        drops the drawn offset at the ROI / section centroid unchanged (robust when
        sections vary widely in size)."""
        if not self._need_image():
            return
        self.app.sync_sections()
        self.app.ensure_poses()
        v = self.app.viewer
        if (v is None or self.FOCUS_DRAFT not in v.layers
                or len(v.layers[self.FOCUS_DRAFT].data) == 0):
            self.app.log(self.STAGE, "place focus points on the 'Focus draft' layer first.")
            return
        proj = self.app.project
        if not any(getattr(s, "roi", None) and s.roi.polygon for s in proj.sections):
            self.app.log(self.STAGE, "define + propagate an ROI first — focus points are "
                                     "only placed on sections that have an ROI.")
            return
        pts_xy = _napari_to_xy(np.asarray(v.layers[self.FOCUS_DRAFT].data))
        ref = self._reference_section(pts_xy)
        if ref is None:
            self.app.log(self.STAGE, "no reference section under the focus points.")
            return
        tmpl = proj.roi_templates[0] if proj.roi_templates else RoiTemplate(ref_section_id=ref.id)
        tmpl.focus_local = roi_mod.focus_template_from_points(ref.pose, pts_xy)
        tmpl.focus_mode = mode
        tmpl.focus_anchors = roi_mod.focus_anchors_from_points(ref, pts_xy)
        n_roi = sum(1 for a in tmpl.focus_anchors if a.get("anchor") == "roi")
        if not proj.roi_templates:
            proj.roi_templates = [tmpl]
        # Focus-only: never re-propagate / overwrite the per-section ROIs.
        roi_mod.propagate_focus_only(tmpl, proj.sections)
        layer_sync.show_focus_points(self.app)
        n_sec = sum(1 for s in proj.sections if s.focus_overview)
        self.app.log(self.STAGE, f"focus points propagated ({mode}, {len(pts_xy)}/section "
                                 f"to {n_sec} ROI sections; {n_roi} anchored to ROI, "
                                 f"{len(pts_xy) - n_roi} to section). Edit the 'Focus "
                                 "points' layer to fine-tune.")
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

    def _propagate_center(self):
        """Propagate the drafted ROI to every section's centroid unchanged (no pose,
        no size fitting) — robust when section polygons vary widely in size."""
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
        tmpl = roi_mod.template_from_polygon(ref.pose, roi_xy, ref_section_id=ref.id,
                                             fit_mode="center", fit_percent=self.sp_pct.value(),
                                             focus_cols=self.sp_fc.value(),
                                             focus_rows=self.sp_fr.value(),
                                             tile_um=(self.sp_tile.value(), self.sp_tile.value()))
        # Center mode keeps the ROI exactly as drawn (no pose rotation), so store the
        # raw drawn polygon rather than the pose-normalised one; propagation only
        # translates it onto each section centroid.
        tmpl.polygon_local = [[float(x), float(y)] for x, y in
                              np.asarray(roi_xy, float).reshape(-1, 2)]
        self.app.project.roi_templates = [tmpl]
        roi_mod.propagate_all(tmpl, self.app.project.sections)
        layer_sync.show_rois(self.app)
        self.app.log(self.STAGE, f"propagated ROI (ref {ref.id}, centered, dimensions "
                                 f"unchanged) to "
                                 f"{sum(1 for s in self.app.project.sections if s.roi)} sections.")
        self.app.save_workflow()

    # ----- automatic per-section SAM ROI detection ----- #
    def _build_auto_detect_group(self, c):
        """Automatic ROI detection: SAM runs inside each section, prompted by a grid
        of points, and the mask most compatible with the template becomes the ROI.
        Parameters mirror the section detector and are populated from the template."""
        c.addWidget(QLabel("<b>Automatic ROI detection (SAM)</b>"))
        self.btn_auto_cal = QPushButton("Calibrate from template")
        self.btn_auto_cal.setToolTip("Populate the parameters below from the drawn ROI "
                                     "template's size (grid density, quality gates, area "
                                     "band) — the ROI analogue of the section detector's "
                                     "'Calibrate'. Draw/propagate a template first.")
        self.btn_auto_cal.clicked.connect(self._calibrate_auto)
        c.addWidget(self.btn_auto_cal)

        ck_row = QHBoxLayout()
        self.lbl_auto_ckpt = QLabel("SAM checkpoint:")
        self.btn_auto_ckpt = QPushButton("Select checkpoint…")
        self.btn_auto_ckpt.setToolTip("Choose the SAM model checkpoint used for detection "
                                      "(shared with the section detector).")
        self.btn_auto_ckpt.clicked.connect(self._select_auto_checkpoint)
        ck_row.addWidget(self.lbl_auto_ckpt); ck_row.addWidget(self.btn_auto_ckpt, 1)
        c.addLayout(ck_row)

        # Parameters live behind an "Advanced" fold, like the section detector.
        adv = QWidget(); ac = QVBoxLayout(adv)
        ac.setContentsMargins(0, 0, 0, 0)

        def _drow(label, w, tip):
            r = QHBoxLayout(); r.addWidget(QLabel(label)); w.setToolTip(tip)
            r.addWidget(w, 1); ac.addLayout(r); return w

        self.sp_auto_pps = QSpinBox(); self.sp_auto_pps.setRange(2, 48); self.sp_auto_pps.setValue(9)
        _drow("points / side:", self.sp_auto_pps,
              "Prompt-point grid density across each section (points_per_side, like the "
              "section detector). More points = SAM samples more locations to find the ROI.")
        self.sp_auto_iou = QDoubleSpinBox(); self.sp_auto_iou.setRange(0.0, 1.0)
        self.sp_auto_iou.setSingleStep(0.05); self.sp_auto_iou.setValue(0.80)
        _drow("pred IoU ≥:", self.sp_auto_iou,
              "Drop masks whose SAM predicted IoU is below this (pred_iou_thresh).")
        self.sp_auto_stab = QDoubleSpinBox(); self.sp_auto_stab.setRange(0.0, 1.0)
        self.sp_auto_stab.setSingleStep(0.02); self.sp_auto_stab.setValue(0.90)
        _drow("stability ≥:", self.sp_auto_stab,
              "Drop unstable masks (stability_score_thresh).")
        self.sp_auto_amin = QDoubleSpinBox(); self.sp_auto_amin.setRange(0.05, 1.0)
        self.sp_auto_amin.setSingleStep(0.05); self.sp_auto_amin.setValue(0.5)
        _drow("min area × tmpl:", self.sp_auto_amin,
              "Reject masks smaller than this fraction of the template ROI area.")
        self.sp_auto_amax = QDoubleSpinBox(); self.sp_auto_amax.setRange(1.0, 5.0)
        self.sp_auto_amax.setSingleStep(0.25); self.sp_auto_amax.setValue(2.0)
        _drow("max area × tmpl:", self.sp_auto_amax,
              "Reject masks larger than this multiple of the template ROI area.")
        self.sp_auto_floor = QDoubleSpinBox(); self.sp_auto_floor.setRange(0.0, 1.0)
        self.sp_auto_floor.setSingleStep(0.05); self.sp_auto_floor.setValue(0.35)
        _drow("score floor:", self.sp_auto_floor,
              "Minimum combined match score to accept SAM's ROI; below it the section "
              "keeps its propagated-template ROI (fallback).")
        self.sp_auto_margin = QDoubleSpinBox(); self.sp_auto_margin.setRange(0.0, 1.0)
        self.sp_auto_margin.setSingleStep(0.05); self.sp_auto_margin.setValue(0.15)
        _drow("crop margin:", self.sp_auto_margin,
              "Extra context around the section bbox when embedding it for SAM.")
        self.sp_auto_inset = QDoubleSpinBox(); self.sp_auto_inset.setRange(0.0, 0.4)
        self.sp_auto_inset.setSingleStep(0.05); self.sp_auto_inset.setValue(0.0)
        _drow("grid inset:", self.sp_auto_inset,
              "Shrink the grid inward from the section edge (fraction) to avoid edge points.")
        self.cb_auto_contour = QComboBox()
        self.cb_auto_contour.addItems(["SAM mask contour", "template fitted to mask"])
        _drow("contour:", self.cb_auto_contour,
              "Use SAM's mask outline directly, or keep the template shape scaled/placed "
              "to best match the SAM mask (uniform ROI shape across sections).")
        c.addWidget(_CollapsibleSection("Advanced parameters", adv, open=False))

        self.chk_auto_preview = QCheckBox("👁 Preview grid on sections (live)")
        self.chk_auto_preview.setToolTip("Overlay the SAM prompt-point grid + expected-ROI "
                                         "box inside every section. Double-click a table "
                                         "row / zoom in to inspect one section.")
        self.chk_auto_preview.toggled.connect(self._toggle_auto_preview)
        c.addWidget(self.chk_auto_preview)
        for sp in (self.sp_auto_pps, self.sp_auto_inset):
            sp.valueChanged.connect(lambda *_: self._refresh_auto_preview())

        runrow = QHBoxLayout()
        self.btn_auto_run = QPushButton("Run automatic ROI detection")
        self.btn_auto_run.setToolTip("Embed each section, prompt SAM on the grid, and set "
                                     "each section's ROI to the best template-matching "
                                     "mask. Runs section-by-section (Stop to cancel).")
        self.btn_auto_run.clicked.connect(self._run_auto_roi)
        self.btn_auto_stop = QPushButton("Stop")
        self.btn_auto_stop.setVisible(False)
        self.btn_auto_stop.clicked.connect(self._stop_auto_roi)
        runrow.addWidget(self.btn_auto_run, 1); runrow.addWidget(self.btn_auto_stop)
        c.addLayout(runrow)
        c.addWidget(self.progress)
        self._refresh_ckpt_label()

    def _auto_template_xy(self):
        """The reference ROI polygon (overview xy) used as the detection template:
        the current draft if any, else the reference/any section's propagated ROI."""
        xy = self._draft_polygon_xy()
        if xy is not None:
            return np.asarray(xy, float).reshape(-1, 2)
        proj = self.app.project
        rid = proj.roi_templates[0].ref_section_id if proj.roi_templates else None
        for want in (rid, None):
            for s in proj.sections:
                if (want is None or s.id == want) and s.roi and len(s.roi.polygon) >= 3:
                    return np.asarray(s.roi.polygon, float).reshape(-1, 2)
        return None

    def _auto_params_from_ui(self):
        return {
            "points_per_side": int(self.sp_auto_pps.value()),
            "pred_iou_thresh": float(self.sp_auto_iou.value()),
            "stability_score_thresh": float(self.sp_auto_stab.value()),
            "min_area_frac": float(self.sp_auto_amin.value()),
            "max_area_mult": float(self.sp_auto_amax.value()),
            "score_floor": float(self.sp_auto_floor.value()),
            "crop_margin": float(self.sp_auto_margin.value()),
            "inset": float(self.sp_auto_inset.value()),
            "contour_source": "template" if self.cb_auto_contour.currentIndex() == 1 else "mask",
        }

    def _calibrate_auto(self):
        if not self._need_image():
            return
        self.app.sync_sections(); self.app.ensure_poses()
        tmpl_xy = self._auto_template_xy()
        if tmpl_xy is None:
            self.app.log(self.STAGE, "draw or propagate a template ROI first.")
            return
        try:
            prof = self.app.gui._current_profile()
        except Exception:
            prof = None
        secs = [s.polygon for s in self.app.project.sections if len(s.polygon) >= 3]
        p = roi_mod.calibrate_roi_params(tmpl_xy, secs, profile=prof)
        self.sp_auto_pps.setValue(int(p["points_per_side"]))
        self.sp_auto_iou.setValue(float(p["pred_iou_thresh"]))
        self.sp_auto_stab.setValue(float(p["stability_score_thresh"]))
        self.sp_auto_amin.setValue(float(p["min_area_frac"]))
        self.sp_auto_amax.setValue(float(p["max_area_mult"]))
        self.sp_auto_floor.setValue(float(p["score_floor"]))
        if self.app.project.roi_templates:
            self.app.project.roi_templates[0].auto_params = {**self._auto_params_from_ui(),
                                                             "roi_area": p["roi_area"]}
        self.app.log(self.STAGE, f"calibrated auto-ROI from template: points/side="
                                 f"{p['points_per_side']} (~{p['points_on_roi']:.1f} on the "
                                 f"ROI), pred IoU≥{p['pred_iou_thresh']}, "
                                 f"stability≥{p['stability_score_thresh']}.")
        self._refresh_auto_preview()

    def _toggle_auto_preview(self, on):
        if not on:
            layer_sync.clear_roi_search_preview(self.app)
            return
        self._refresh_auto_preview()

    def _refresh_auto_preview(self):
        if getattr(self, "chk_auto_preview", None) is None or not self.chk_auto_preview.isChecked():
            return
        if not self.app.has_image():
            return
        self.app.sync_sections()
        tmpl_xy = self._auto_template_xy()
        if tmpl_xy is None:
            self.app.log(self.STAGE, "draw or propagate a template ROI to preview the grid.")
            self.chk_auto_preview.setChecked(False)
            return
        layer_sync.show_roi_search_preview(self.app, tmpl_xy, int(self.sp_auto_pps.value()),
                                           inset=float(self.sp_auto_inset.value()))

    def _refresh_ckpt_label(self):
        import os
        try:
            prof = self.app.gui._current_profile()
            ck = self.app.gui._checkpoint_for_model("Auto", prof)
            name = os.path.basename(str(ck)) if ck else "(none selected)"
        except Exception:
            name = "(shared with detector)"
        self.lbl_auto_ckpt.setText(f"SAM checkpoint: {name}")

    def _select_auto_checkpoint(self):
        try:
            self.app.gui.select_checkpoint()
        except Exception as e:
            self.app.log(self.STAGE, f"checkpoint select failed: {e}")
        self._refresh_ckpt_label()

    def _stop_auto_roi(self):
        if self.worker.is_running():
            self.worker.stop()
            self.app.log(self.STAGE, "stopping automatic ROI detection…")

    def _run_auto_roi(self):
        import os
        if not self._need_image() or self.worker.is_running():
            return
        self.app.sync_sections(); self.app.ensure_poses()
        tmpl_xy = self._auto_template_xy()
        if tmpl_xy is None:
            self.app.log(self.STAGE, "draw or propagate a template ROI first, then "
                                     "'Calibrate from template'.")
            return
        gui = self.app.gui
        try:
            prof = gui._current_profile()
            ckpt = gui._checkpoint_for_model("Auto", prof)
        except Exception:
            ckpt = None
        if not ckpt or not os.path.exists(str(ckpt)):
            self.app.log(self.STAGE, "select a SAM checkpoint first.")
            try:
                gui.select_checkpoint()
            except Exception:
                pass
            self._refresh_ckpt_label()
            return

        proj = self.app.project
        sections = [s for s in proj.sections if len(s.polygon) >= 3]
        if not sections:
            return
        # Fallback: seed every section with a centered template ROI when nothing
        # has been propagated, so sections where SAM finds nothing keep an ROI.
        if not any(getattr(s, "roi", None) and s.roi.polygon for s in sections):
            ref = self._reference_section(tmpl_xy)
            if ref is not None:
                seed = roi_mod.template_from_polygon(ref.pose, tmpl_xy, ref_section_id=ref.id,
                                                     fit_mode="center")
                seed.polygon_local = [[float(x), float(y)] for x, y in tmpl_xy]
                proj.roi_templates = [seed]
                roi_mod.propagate_all(seed, sections)
                layer_sync.show_rois(self.app)

        params = self._auto_params_from_ui()
        self._auto_contour = params["contour_source"]
        self._auto_hit = 0
        self._roi_refresh_ct = 0
        spec = {
            "image": self.app.image_path,
            "checkpoint": str(ckpt),
            "device": getattr(gui, "_device_prefer", "") or "",
            "target": self.app.target_long_side(),
            "template": [[float(x), float(y)] for x, y in np.asarray(tmpl_xy, float).reshape(-1, 2)],
            "sections": [{"id": s.id, "polygon": s.polygon} for s in sections],
            "params": {**params, "crop_long": 1024},
        }
        import json as _json
        import tempfile
        fd, spath = tempfile.mkstemp(suffix="_stim_roi_spec.json")
        with os.fdopen(fd, "w") as f:
            _json.dump(spec, f)

        if self.btn_sam.isChecked():
            self.btn_sam.setChecked(False)
        if self.btn_manual_roi.isChecked():
            self.btn_manual_roi.setChecked(False)
        self.btn_auto_run.setEnabled(False); self.btn_auto_stop.setVisible(True)
        self._auto_running = True
        self.app.log(self.STAGE, f"automatic ROI detection on {len(sections)} sections "
                                 "(background) — the view stays put; pan freely.")
        started = self.worker.start("roi_detect_worker", ["--spec", spath], handlers={
            "PROGRESS": lambda p: self._set_progress(p.get("done", 0), p.get("total", 0)),
            "ROISTART": self._on_roi_start,
            "ROI": self._on_roi_result,
            "ROI_DONE": self._on_roi_done,
            "ERROR": lambda p: self.app.log(self.STAGE, f"error: {p.get('error', p)}"),
        }, on_log=lambda m: self.app.log(self.STAGE, m))
        if not started:
            self._auto_running = False
            self.btn_auto_run.setEnabled(True); self.btn_auto_stop.setVisible(False)

    def _on_roi_start(self, payload):
        s = self.app.project.get(payload.get("id"))
        if s is not None:
            layer_sync.highlight_current_section(self.app, s)   # cyan, no camera move

    def _on_roi_result(self, payload):
        poly = payload.get("polygon")
        s = self.app.project.get(payload.get("id"))
        if s is not None and poly and len(poly) >= 3:
            s.roi = Roi(polygon=[[float(x), float(y)] for x, y in poly],
                        fit_mode=("auto" if self._auto_contour == "mask" else "auto_template"))
            self._auto_hit += 1
            self._roi_refresh_ct += 1
            if self._roi_refresh_ct % 3 == 0:       # throttle overlay rebuilds
                layer_sync.show_rois(self.app)

    def _on_roi_done(self, payload):
        self._auto_running = False
        layer_sync.clear_current_section(self.app)
        layer_sync.show_rois(self.app)
        self.progress.setVisible(False)
        self.btn_auto_run.setEnabled(True); self.btn_auto_stop.setVisible(False)
        self.app.save_workflow()
        n = payload.get("n", 0); hit = payload.get("hit", self._auto_hit)
        self.app.log(self.STAGE, f"automatic ROI done — SAM set {hit}/{n} sections; the "
                                 f"rest kept the propagated template. "
                                 f"{sum(1 for s in self.app.project.sections if s.roi)} ROIs total.")

    def _on_roi_worker_finished(self, _code):
        # safety net if the worker died without a ROI_DONE (crash / Stop)
        if not self._auto_running:
            return
        self._auto_running = False
        layer_sync.clear_current_section(self.app)
        self.btn_auto_run.setEnabled(True); self.btn_auto_stop.setVisible(False)
        self.progress.setVisible(False)
        try:
            layer_sync.show_rois(self.app)
            self.app.save_workflow()
        except Exception:
            pass
        self.app.log(self.STAGE, "automatic ROI detection stopped.")

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
    gui.app = app          # let the GUI's debounced autosave reach the full save
    nav = FovNavigator(app)
    table = SectionTableDock(app, nav)

    # Up/Down arrows (while the image canvas is focused) step to the prev/next
    # section at the current zoom — quick section-to-section proofreading.
    try:
        viewer.bind_key("Up", lambda v: table._step(-1), overwrite=True)
        viewer.bind_key("Down", lambda v: table._step(1), overwrite=True)
    except Exception:
        pass

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

    btn_save = QPushButton("💾 Save now")
    btn_save.setToolTip("Capture the current sections, ROIs and focus points "
                        "(including native napari edits) and save the working "
                        "session to disk. The session also autosaves ~1.5 s after "
                        "each edit and on quit — this is a manual, immediate save.")

    def _save_now():
        if not app.has_image():
            app.log("io", "load an image before saving.")
            return
        ok = app.save_all()
        app.log("io", "session saved." if ok else "save failed (see console).")
    btn_save.clicked.connect(_save_now)

    filebar.addWidget(btn_open)
    filebar.addWidget(btn_export)
    filebar.addWidget(btn_save)
    filebar.addStretch(1)
    filebar.addWidget(QLabel("outline px:"))
    sp_outline = QDoubleSpinBox()
    sp_outline.setRange(0.25, 8.0); sp_outline.setSingleStep(0.25)
    sp_outline.setDecimals(2); sp_outline.setValue(1.0)
    sp_outline.setMaximumWidth(64)
    sp_outline.setToolTip("On-screen thickness of every polygon outline (sections, "
                          "ROIs, calibration…), in canvas pixels. Stays constant as "
                          "you zoom, so outlines on small polygons don't get in the "
                          "way — lower this for finer placement.")
    sp_outline.valueChanged.connect(
        lambda v: getattr(gui, "set_outline_screen_px", lambda *_: None)(v))
    filebar.addWidget(sp_outline)
    cl.addLayout(filebar)

    loaded = {"done": None, "target": None, "tries": 0}

    def _restore_for_current_image():
        """Restore the saved workflow sidecar for the current image: merge the
        ROI/focus/order/QC data, redraw their overlays, and reapply the saved
        display settings. Runs once per image; waits a few ticks for the legacy
        GUI to finish restoring the section masks first (apply_results matches by
        section id, so the masks must be in place before we merge)."""
        if not app.has_image():
            return
        path = app.image_path
        if loaded["done"] == path:
            return
        if loaded["target"] != path:                 # a new image -> reset the wait
            loaded["target"] = path
            loaded["tries"] = 0
        proj = app.sync_sections()
        loaded["tries"] += 1
        if not proj.sections and loaded["tries"] < 15:
            return                                    # masks not restored yet; retry
        if app.load_workflow():
            layer_sync.restore_overlays(app)
            n = layer_sync.apply_display(app)
            app.log("io", "restored saved ROIs / focus / order"
                          + (f" + display settings ({n} layers)" if n else ""))
        else:
            # No sidecar: fall back to the CZI's own CAT ROI/focus annotations
            # so an annotated CZI reloads ROIs + focus.
            n_roi, n_focus = app.restore_annotations_from_czi()
            if n_roi or n_focus:
                layer_sync.show_rois(app)
                layer_sync.show_focus_points(app)
                app.log("io", f"restored {n_roi} ROIs + {n_focus} focus "
                              "points from CZI annotations.")
        loaded["done"] = path
        table.refresh()
        rois.refresh_fiducials()

    def _on_tab(_i):
        try:
            app.sync_sections()
            _restore_for_current_image()
            table.refresh()
            rois.refresh_fiducials()
        except Exception:
            pass
    tabs.currentChanged.connect(_on_tab)

    # Restore promptly after an image is opened, not only when a tab is clicked.
    # Cheap no-op once done; keeps watching so reopening a different wafer also
    # restores. Parented to ``container`` so Qt keeps it alive.
    def _poll_restore():
        try:
            _restore_for_current_image()
        except Exception:
            pass
    _restore_poll = QTimer(container)
    _restore_poll.setInterval(700)
    _restore_poll.timeout.connect(_poll_restore)
    _restore_poll.start()

    # Persist on quit so display-only tweaks (e.g. dragging an outline/arrow
    # width slider, which don't trigger a data save) still carry over.
    _qapp = QApplication.instance()
    if _qapp is not None:
        def _save_on_quit():
            try:
                app.save_all()
            except Exception:
                pass
        _qapp.aboutToQuit.connect(_save_on_quit)

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
