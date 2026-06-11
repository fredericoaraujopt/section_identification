"""STiM napari GUI.

One window: load an image (incl. whole-slide ``.czi`` from the pyramid), run SAM
2.1 detection (whole-image by default; an optional tiled mode for tiny-section
wafers), edit sections/fiducials natively in napari, and export CSV / GeoJSON /
a ZEN-annotated CZI. The working session autosaves and is restored on reopen.

Coordinate convention: napari layer data is ``(row, col)`` = ``(y, x)``; our
detection/export code uses ``(x, y)``. Helpers convert at the boundary.
"""

import ast
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, QTimer, QProcess, QProcessEnvironment
from qtpy.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

import napari

from section_identification.section_detector import automatic_identification
from section_identification.export import export_polygons, mask_to_polygon
from section_identification import czi_io
from section_identification import host_profile
from section_identification.device import describe as describe_device, device_str


def xy_to_napari(poly_xy):
    p = np.asarray(poly_xy, dtype=float).reshape(-1, 2)
    return p[:, ::-1]


def napari_to_xy(poly_yx):
    p = np.asarray(poly_yx, dtype=float).reshape(-1, 2)
    return p[:, ::-1]


class SectionIdentificationGUI(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.image_path = None
        self.overview = None
        self.geom = None
        self.calibration = None
        self._param_viz = None
        self.image_layer = None
        self.shapes_layer = None
        self.fid_layer = None
        self.calib_layer = None
        self.tiles_layer = None
        self.current_tile_layer = None
        self.raw_layer = None
        # autosave (debounced) + detection process + streaming state
        self._autosave_timer = QTimer(self); self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self.save_project)
        self.proc = None
        self._det_params = None
        self._det_t0 = 0.0
        self._stream_mode = False
        self._proc_buf = ""
        self._raw_sections = []
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)

        layout = QVBoxLayout(); layout.setContentsMargins(6, 6, 6, 6); layout.setSpacing(4)
        self.setLayout(layout)

        # checkpoint paths (set before the Advanced section uses them)
        pkg = Path(os.path.abspath(__file__)); ckpt_dir = pkg.parents[1] / "checkpoint"
        self.checkpoint = str(ckpt_dir / "sam2.1_hiera_base_plus.pt")
        self.sam1_checkpoint = str(ckpt_dir / "sam_vit_b_01ec64.pth")
        self.sam1_model_type = "vit_b"

        # collapsible section helper → returns the section's body layout
        def section(title, open=True):
            btn = QPushButton(("▾ " if open else "▸ ") + title)
            btn.setCheckable(True); btn.setChecked(open)
            btn.setStyleSheet("QPushButton{text-align:left;font-weight:bold;padding:6px;"
                              "border:none;border-radius:4px;background:#333;}")
            body = QWidget(); body.setVisible(open)
            bl = QVBoxLayout(body); bl.setContentsMargins(10, 4, 4, 8); bl.setSpacing(4)
            btn.toggled.connect(lambda on, b=body, bt=btn, t=title:
                                (b.setVisible(on), bt.setText(("▾ " if on else "▸ ") + t)))
            layout.addWidget(btn); layout.addWidget(body)
            return bl

        # ---- top (always visible): image picker ----
        self.btn_select = QPushButton("Select Image / CZI…")
        self.lbl_path = QLabel("No image selected"); self.lbl_path.setWordWrap(True)
        layout.addWidget(self.btn_select); layout.addWidget(self.lbl_path)

        # ===== 1 · Calibrate (optional, recommended) =====
        cal = section("1 · Calibrate  (optional, recommended)", open=True)
        self.btn_calibrate = QPushButton("Calibrate from examples")
        cal.addWidget(self.btn_calibrate)
        self.lbl_calib = QLabel("Draw 1–3 example sections in the 'Calibration examples' "
                                "layer, then Calibrate — it sets every SAM parameter from "
                                "the section size.")
        self.lbl_calib.setWordWrap(True); cal.addWidget(self.lbl_calib)
        self.lbl_plan = QLabel("Detection plan: calibrate to compute it.")
        self.lbl_plan.setWordWrap(True)
        self.lbl_plan.setStyleSheet("QLabel{background:#1e1e1e;padding:6px;border-radius:4px;}")
        cal.addWidget(self.lbl_plan)

        # ===== 2 · Automatic detector =====
        det = section("2 · Automatic detector", open=True)
        host_row = QHBoxLayout()
        host_row.addWidget(QLabel("Run on:"))
        self.cb_device = QComboBox(); self.cb_device.addItems(["Auto", "CPU", "CUDA", "MPS"])
        self.cb_device.setToolTip("Where SAM runs. Auto picks CUDA > Apple MPS > CPU. "
                                  "Device sets the memory/speed regime; CPU is slowest.")
        host_row.addWidget(self.cb_device, 1)
        det.addLayout(host_row)
        self.lbl_host = QLabel(f"Host: {describe_device()}")
        self.lbl_host.setWordWrap(True); det.addWidget(self.lbl_host)

        # ---- Advanced (nested fold): full SAM parameter set; each row has a
        #      tooltip + a "?" that opens that parameter's section in the guide ----
        self.btn_adv = QPushButton("▸ Advanced parameters"); self.btn_adv.setCheckable(True)
        det.addWidget(self.btn_adv)
        adv = QWidget(); adv.setVisible(False)
        advcol = QVBoxLayout(adv); advcol.setContentsMargins(8, 4, 4, 4)
        self.btn_guide = QPushButton("📖 Open parameter guide")
        advcol.addWidget(self.btn_guide)
        self.chk_viz = QCheckBox("👁 Preview parameters on the image (live)")
        self.chk_viz.setToolTip("Overlay SAM's query-point grid, the tile grid, its "
                                "sub-crops and a min-area disc — they update as you "
                                "change the values, so you see how SAM will behave.")
        advcol.addWidget(self.chk_viz)
        advf = QFormLayout(); advf.setContentsMargins(0, 0, 0, 0)
        advf.setRowWrapPolicy(QFormLayout.WrapLongRows)
        advf.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        advcol.addLayout(advf)
        self.btn_adv.toggled.connect(
            lambda on: (adv.setVisible(on),
                        self.btn_adv.setText(("▾ " if on else "▸ ") + "Advanced parameters")))

        def _row(widget, label, anchor, tip):
            widget.setToolTip(tip)
            q = QPushButton("?"); q.setFixedWidth(22)
            q.setToolTip("What does this do? (opens the guide)")
            q.clicked.connect(lambda _=False, a=anchor: self._open_guide(a))
            cont = QWidget(); h = QHBoxLayout(cont); h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(widget, 1); h.addWidget(q)
            advf.addRow(label, cont)
            return widget

        self.sp_pps = _row(QSpinBox(), "points / side", "points_per_side",
            "Density of SAM's query-point grid per tile. ~2–3 points across a section. "
            "More = finds smaller/closer objects, slower.")
        self.sp_pps.setRange(4, 192); self.sp_pps.setValue(32)
        self.sp_iou = _row(QDoubleSpinBox(), "pred IoU", "pred_iou_thresh",
            "SAM's confidence floor. Lower to recover faint sections; raise to drop weak ones.")
        self.sp_iou.setRange(0.0, 1.0); self.sp_iou.setSingleStep(0.05); self.sp_iou.setValue(0.80)
        self.sp_stab = _row(QDoubleSpinBox(), "stability", "stability_score_thresh",
            "Mask edge-stability floor. Lower for noisy/small sections; raise for clean ones.")
        self.sp_stab.setRange(0.0, 1.0); self.sp_stab.setSingleStep(0.01); self.sp_stab.setValue(0.92)
        self.sp_staboff = _row(QDoubleSpinBox(), "stability offset", "stability_score_offset",
            "Nudge used to measure stability. Usually leave at 1.0.")
        self.sp_staboff.setRange(0.1, 5.0); self.sp_staboff.setSingleStep(0.1); self.sp_staboff.setValue(1.0)
        self.sp_boxnms = _row(QDoubleSpinBox(), "box NMS", "box_nms_thresh",
            "Merge masks overlapping more than this. Lower = more dedup; raise to keep "
            "touching sections separate.")
        self.sp_boxnms.setRange(0.1, 1.0); self.sp_boxnms.setSingleStep(0.05); self.sp_boxnms.setValue(0.70)
        self.sp_crop = _row(QSpinBox(), "crop layers", "crop_n_layers",
            "SAM's built-in re-cropping for tiny objects. 0=off, 1=2×2 sub-crops (~5× slower). "
            "Calibrate sets it from section size.")
        self.sp_crop.setRange(0, 3); self.sp_crop.setValue(0)
        self.sp_cropov = _row(QDoubleSpinBox(), "crop overlap", "crop_overlap_ratio",
            "Overlap between SAM's sub-crops so edge sections aren't split.")
        self.sp_cropov.setRange(0.0, 0.8); self.sp_cropov.setSingleStep(0.02); self.sp_cropov.setValue(512 / 1500)
        self.sp_cropds = _row(QSpinBox(), "crop grid ÷", "crop_n_points_downscale_factor",
            "Thins the point grid on deeper crop layers. 2 is typical when crop layers ≥ 1.")
        self.sp_cropds.setRange(1, 4); self.sp_cropds.setValue(1)
        self.sp_minmask = _row(QSpinBox(), "min mask area", "min_mask_region_area",
            "SAM drops regions/holes smaller than this (specks filter inside SAM). "
            "~5% of a section's area.")
        self.sp_minmask.setRange(0, 10_000_000); self.sp_minmask.setValue(100)
        self.sp_minarea = _row(QSpinBox(), "min section area", "min_section_area",
            "Detections smaller than this are dropped + anchors the area-DBSCAN band. "
            "Calibrate sets it to ~½ the median section.")
        self.sp_minarea.setRange(0, 10_000_000); self.sp_minarea.setValue(50)
        self.chk_m2m = _row(QCheckBox(), "refine (use_m2m)", "use_m2m",
            "Extra mask-to-mask refinement: cleaner edges, ~2× slower.")
        self.chk_lowmem = _row(QCheckBox(), "low-memory (1 mask/pt)", "low_memory",
            "Memory-saver: SAM emits 1 mask per point instead of 3 → ~3× less "
            "peak mask memory (eases pressure on Macs / weak machines, lets the "
            "batch run bigger). May slightly lower recall on ambiguous sections. "
            "Off = SAM default (3 masks/point, best recall).")
        self.sp_tile = _row(QSpinBox(), "tile px (0=whole)", "tile_px",
            "Tile size; SAM upscales each tile to 1024 (smaller tile → bigger section to "
            "SAM). 0 = whole image. Calibrate sets it; host cap may shrink it.")
        self.sp_tile.setRange(0, 16384); self.sp_tile.setSingleStep(128); self.sp_tile.setValue(0)
        self.sp_overlap = _row(QDoubleSpinBox(), "tile overlap", "overlap",
            "Overlap between tiles so each section fits whole in ≥1 tile.")
        self.sp_overlap.setRange(0.0, 0.6); self.sp_overlap.setSingleStep(0.05); self.sp_overlap.setValue(0.2)
        self.sp_targetsam = _row(QSpinBox(), "target → SAM", "target_sam_px",
            "How big a section should look to SAM. The main quality↔speed dial; "
            "higher = sharper but more tiles. ~64 typical, ~40 on slow machines.")
        self.sp_targetsam.setRange(24, 256); self.sp_targetsam.setValue(64)
        self.sp_target = _row(QSpinBox(), "overview px", "overview_long_side",
            "Read resolution (real detail). Bigger = sharper, more memory/time; host-capped. "
            "Needs an image reload to take effect.")
        self.sp_target.setRange(1024, 16384); self.sp_target.setSingleStep(512); self.sp_target.setValue(8192)
        self.sp_ppb = _row(QSpinBox(), "points / batch", "points_per_batch",
            "Query points SAM runs at once. Memory/speed only — NO effect on results. "
            "Lower to avoid crashing/thrashing; raise with spare GPU. Auto-capped to host.")
        self.sp_ppb.setRange(1, 256); self.sp_ppb.setValue(16)
        self.cb_model = _row(QComboBox(), "model", "model",
            "Heavier = better but slower/more memory. Auto picks by host (tiny/small on "
            "CPU/weak, base_plus/large on GPU).")
        self.cb_model.addItems(["Auto", "tiny", "small", "base_plus", "large"])
        self.chk_filter = _row(QCheckBox(), "area DBSCAN", "dbscan",
            "Keep the dominant section-sized area cluster (drops debris/clumps). "
            "Leave on for wafers.")
        self.chk_filter.setChecked(True)
        # checkpoint selector lives inside Advanced (rarely changed; Auto model picks it)
        self.lbl_ckpt = QLabel(f"Checkpoint: …/{os.path.basename(self.checkpoint)}")
        self.lbl_ckpt.setWordWrap(True); advcol.addWidget(self.lbl_ckpt)
        self.btn_ckpt = QPushButton("Select checkpoint (.pt)")
        advcol.addWidget(self.btn_ckpt)
        det.addWidget(adv)                          # the Advanced fold sits in the detector section

        det_row = QHBoxLayout()
        self.btn_auto = QPushButton("Run Automatic Detection")
        self.btn_stop = QPushButton("Stop"); self.btn_stop.setVisible(False)
        det_row.addWidget(self.btn_auto); det_row.addWidget(self.btn_stop)
        det.addLayout(det_row)
        self.lbl_elapsed = QLabel(""); det.addWidget(self.lbl_elapsed)

        # ===== 3 · Manual detector =====
        man = section("3 · Manual detector", open=False)
        self.btn_manual = QPushButton("Manual detector (OpenCV)")
        man.addWidget(self.btn_manual)
        self.btn_manual_napari = QPushButton("Manual editor (napari)")
        self.btn_manual_napari.setCheckable(True)
        man.addWidget(self.btn_manual_napari)
        _ml = QLabel("<i>OpenCV: separate window (hover preview, click add, 'r' remove, "
                     "'m' fiducials, Esc finish). napari: in-viewer — zoom in, 'e' to "
                     "embed the view at full-res, hover/click to add, 'r' remove, "
                     "'m' fiducial.</i>")
        _ml.setWordWrap(True); man.addWidget(_ml)
        _edit = QLabel("<i>Edit results directly: select the 'Sections' layer, then use "
                       "napari's polygon tool to add a section or the select tool to "
                       "delete one — changes auto-save and export.</i>")
        _edit.setWordWrap(True); man.addWidget(_edit)

        # ===== 4 · Export =====
        ex = section("4 · Export", open=False)
        exp_row = QHBoxLayout()
        self.chk_exp_csv = QCheckBox("CSV"); self.chk_exp_csv.setChecked(True)
        self.chk_exp_geojson = QCheckBox("GeoJSON"); self.chk_exp_geojson.setChecked(True)
        self.chk_exp_png = QCheckBox("PNG"); self.chk_exp_png.setChecked(True)
        self.chk_exp_czi = QCheckBox("CZI"); self.chk_exp_czi.setChecked(False)
        self.chk_exp_czi.setToolTip("Annotated CZI for ZEN — copies the whole file "
                                    "(can be many GB); off by default.")
        for c in (self.chk_exp_csv, self.chk_exp_geojson, self.chk_exp_png, self.chk_exp_czi):
            exp_row.addWidget(c)
        ex.addLayout(exp_row)
        self.btn_export = QPushButton("Export selected")
        ex.addWidget(self.btn_export)

        # ---- bottom (always visible): log ----
        layout.addWidget(QLabel("<b>Log</b>"))
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(160)
        layout.addWidget(self.log, stretch=1)
        self.progress = QProgressBar(); self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self._old_stdout = sys.stdout
        sys.stdout = self

        self.btn_select.clicked.connect(self.select_image)
        self.btn_auto.clicked.connect(self.run_auto)
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_export.clicked.connect(self.export_coordinates)
        self.btn_ckpt.clicked.connect(self.select_checkpoint)
        self.btn_manual.clicked.connect(self.run_manual)
        self.btn_manual_napari.clicked.connect(self.toggle_manual_napari)
        self.btn_calibrate.clicked.connect(self.calibrate_from_examples)
        self.btn_guide.clicked.connect(lambda: self._open_guide())
        self.cb_device.currentTextChanged.connect(self._on_device_changed)
        self._device_prefer = ""
        self._refresh_host()

        # Live parameter previews: toggle + redraw whenever a geometric knob moves.
        self.chk_viz.toggled.connect(self._toggle_param_viz)
        for w in (self.sp_pps, self.sp_tile, self.sp_overlap, self.sp_crop,
                  self.sp_cropov, self.sp_cropds, self.sp_minarea, self.sp_minmask,
                  self.sp_targetsam):
            w.valueChanged.connect(self._param_viz_refresh)
        # NB: intentionally NOT refreshing on camera move — the previews are
        # fixed in image space (a central representative tile), so they don't
        # jitter as you pan/zoom.

    def _open_guide(self, anchor=None):
        try:
            from section_identification.param_guide import open_param_guide
            open_param_guide(self, anchor)
        except Exception:
            self.log_msg("parameter guide unavailable:\n" + traceback.format_exc())

    def _toggle_param_viz(self, on):
        try:
            if self._param_viz is None:
                from section_identification.param_viz import ParamVisualizer
                self._param_viz = ParamVisualizer(self)
            self._param_viz.set_active(bool(on))
        except Exception:
            self.log_msg("parameter preview unavailable:\n" + traceback.format_exc())

    def _param_viz_refresh(self, *a):
        if self._param_viz is not None:
            self._param_viz.refresh_if_active()

    # ----- host profile -----
    def _current_profile(self):
        return host_profile.detect_profile(getattr(self, "_device_prefer", "") or None)

    def _on_device_changed(self, text):
        self._device_prefer = "" if text == "Auto" else text.lower()
        self._refresh_host()

    def _refresh_host(self):
        try:
            prof = self._current_profile()
            self.lbl_host.setText("Host: " + prof.summary())
            if not getattr(self, "calibration", None):
                self.sp_ppb.setValue(int(prof.points_per_batch))
        except Exception:
            pass

    def _checkpoint_for_model(self, model_pref, prof):
        """Resolve the checkpoint path for the chosen/auto model variant, falling
        back to the loaded checkpoint when the lighter variant isn't downloaded."""
        variant = prof.model_variant if model_pref in ("Auto", "") else model_pref
        d = os.path.dirname(self.checkpoint)
        cand = os.path.join(d, f"sam2.1_hiera_{variant}.pt")
        if os.path.isfile(cand):
            return cand
        if variant not in os.path.basename(self.checkpoint):
            self.log_msg(f"[host] hiera_{variant} checkpoint not found; using "
                         f"{os.path.basename(self.checkpoint)} (download "
                         f"sam2.1_hiera_{variant}.pt for the lighter/faster model).")
        return self.checkpoint

    # ----- logging -----
    def write(self, text):
        self._old_stdout.write(text); self._old_stdout.flush()
        if text.strip():
            self.log.append(text.rstrip()); QApplication.processEvents()

    def flush(self):
        pass

    def log_msg(self, text):
        print(text)

    def select_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select SAM 2.1 checkpoint", "",
                                              "Checkpoints (*.pt *.pth)")
        if path:
            self.checkpoint = path
            self.lbl_ckpt.setText(f"Checkpoint: …/{os.path.basename(path)}")

    # ----- load image + restore session -----
    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select an image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.czi)")
        if not path:
            return
        self.image_path = path
        self.lbl_path.setText(f"Selected: {os.path.basename(path)}")
        self.lbl_path.setToolTip(path)
        self.log_msg(f"Loading {os.path.basename(path)}…")
        self._display_scale = 1.0
        self._display_levels = None
        try:
            if czi_io.is_czi(path):
                arr, geom, meta = czi_io.read_czi_overview(
                    path, target_long_side=self.sp_target.value())
                self.geom = geom
                self.overview = czi_io.to_rgb8(arr)
                self.log_msg(f"CZI {meta['size_x']}x{meta['size_y']}, overview "
                             f"{self.overview.shape}, zoom {meta['zoom']:.4g}")
                # CZI-backed lazy multiscale for DISPLAY: napari shows a low-res
                # overview and pulls full-resolution tiles only for the visible
                # region as you zoom in. Annotation DATA stays in overview pixels
                # (so detection/save/load/export are unchanged); the Shapes layers
                # are aligned to the full-res image via a per-layer `scale`.
                try:
                    levels, _ = czi_io.build_czi_dask_pyramid(path)
                    self._display_levels = levels
                    self._display_scale = 1.0 / geom.zoom  # overview px -> full-res world
                    self.log_msg(f"Full-res lazy multiscale: {len(levels)} levels "
                                 f"(L0 {levels[0].shape[1]}x{levels[0].shape[0]} px).")
                except Exception:
                    self.log_msg("[warn] lazy multiscale unavailable; using overview:\n"
                                 + traceback.format_exc())
            else:
                from PIL import Image
                self.overview = np.array(Image.open(path).convert("RGB"))
                self.geom = None
        except Exception:
            self.log_msg("❌ load failed:\n" + traceback.format_exc()); return

        self._reset_layers()
        if self._display_levels is not None:
            # Full-resolution lazy pyramid (level 0 = full res); shape layers are
            # scaled by 1/zoom so overview-pixel annotations overlay correctly.
            self.image_layer = self.viewer.add_image(
                self._display_levels, name="Wafer (full-res)", multiscale=True,
                rgb=True)
        elif max(self.overview.shape[:2]) > 4096:
            pyr = [self.overview, self.overview[::2, ::2], self.overview[::4, ::4]]
            self.image_layer = self.viewer.add_image(pyr, name="Overview", multiscale=True)
        else:
            self.image_layer = self.viewer.add_image(self.overview, name="Overview")
        # Clear any parameter-preview overlays from a previously-loaded image so
        # they can't render with the old image's scale/shape on the new one.
        if getattr(self, "_param_viz", None) is not None:
            try:
                self._param_viz.set_active(False)
            except Exception:
                pass
            try:
                self.chk_viz.setChecked(False)
            except Exception:
                pass
        self._restored_raw_xy = []
        self._restored_calib_xy = []
        polys_xy, fids_xy = self._restore_session()
        try:
            self._ensure_edit_layers(polys_xy)
            if fids_xy and self.fid_layer is not None:
                self.fid_layer.data = np.asarray(fids_xy, dtype=float)[:, ::-1]
            # restore the unfiltered detector output as a (hidden) reference layer
            if self._restored_raw_xy:
                lyr = self._set_shapes(
                    "raw_layer", "Raw detections",
                    [xy_to_napari(p) for p in self._restored_raw_xy],
                    edge="orange", face=(1, 0.55, 0, 0.12), width=2)
                lyr.visible = False
                self.log_msg(f"Restored {len(self._restored_raw_xy)} raw (unfiltered) "
                             "detections (layer hidden; toggle to view).")
        except Exception:
            self.log_msg("[warn] building layers failed:\n" + traceback.format_exc())
            self._ensure_edit_layers([])
        self._raw_sections = []
        self._ensure_calib_layer()

    def _restore_session(self):
        """Return (polys_overview, fids_overview) from project JSON, else CZI
        annotations, else the legacy mask_coordinates.csv."""
        polys, fids = self.load_project()
        if polys:
            self.log_msg(f"Restored {len(polys)} sections + {len(fids)} fiducials "
                         "from autosaved project.")
            return polys, fids
        if czi_io.is_czi(self.image_path) and self.geom is not None:
            try:
                from section_identification.czi_export import read_annotations
                pf, ff = read_annotations(self.image_path)
                polys = [self._to_overview(p) for p in pf]
                fids = [tuple(self._to_overview([f])[0]) for f in ff]
                if polys or fids:
                    self.log_msg(f"Loaded {len(polys)} polygons + {len(fids)} "
                                 "fiducials from CZI annotations.")
                    return polys, fids
            except Exception:
                self.log_msg("[warn] CZI annotation read failed:\n" + traceback.format_exc())
        polys, fids = self._load_legacy_csv()
        if polys:
            self.log_msg(f"Restored {len(polys)} sections from legacy "
                         "mask_coordinates.csv.")
        return polys, fids

    def _reset_layers(self):
        for lyr in list(self.viewer.layers):
            try:
                self.viewer.layers.remove(lyr)
            except Exception:
                pass
        self.image_layer = self.shapes_layer = self.fid_layer = None
        self.calib_layer = self.tiles_layer = None
        self.current_tile_layer = self.raw_layer = None

    def _layer_scale(self):
        """Per-layer scale so overview-pixel annotation DATA overlays the
        full-res multiscale image (1/geom.zoom). (1,1) when not in full-res
        display mode, so ordinary images are unaffected."""
        s = getattr(self, "_display_scale", 1.0)
        return (s, s)

    def _ensure_edit_layers(self, polygons_xy):
        if self.shapes_layer is not None and self.shapes_layer in self.viewer.layers:
            self.viewer.layers.remove(self.shapes_layer)
        data = [xy_to_napari(p) for p in polygons_xy] if polygons_xy else []
        self.shapes_layer = self.viewer.add_shapes(
            data, shape_type="polygon", name="Sections",
            face_color=[1, 0, 0, 0.18], edge_width=4, scale=self._layer_scale())
        try:
            self.shapes_layer.edge_color = "red"
        except Exception:
            pass
        if self.fid_layer is None or self.fid_layer not in self.viewer.layers:
            self.fid_layer = self.viewer.add_points(np.empty((0, 2)),
                                                    name="Fiducials", size=24,
                                                    scale=self._layer_scale())
            for attr, val in (("face_color", "cyan"), ("border_color", "blue"),
                              ("edge_color", "blue")):
                try:
                    setattr(self.fid_layer, attr, val)
                except Exception:
                    pass
        for lyr in (self.shapes_layer, self.fid_layer):
            try:
                lyr.events.data.connect(self._schedule_autosave)
            except Exception:
                pass

    def _ensure_calib_layer(self):
        if self.calib_layer is None or self.calib_layer not in self.viewer.layers:
            data = [xy_to_napari(p) for p in getattr(self, "_restored_calib_xy", [])]
            self.calib_layer = self.viewer.add_shapes(
                data, shape_type="polygon", name="Calibration examples",
                face_color=[0, 1, 0, 0.25], edge_width=4, scale=self._layer_scale())
            try:
                self.calib_layer.edge_color = "lime"
            except Exception:
                pass
            try:                                   # persist drawn examples too
                self.calib_layer.events.data.connect(self._schedule_autosave)
            except Exception:
                pass
        return self.calib_layer

    def current_calib_xy(self):
        lyr = getattr(self, "calib_layer", None)
        if lyr is None or lyr not in self.viewer.layers:
            return []
        return [napari_to_xy(d) for d in lyr.data if len(np.asarray(d)) >= 3]

    # ----- project autosave / restore -----
    def _schedule_autosave(self, *a):
        try:
            self._autosave_timer.start(1500)
        except Exception:
            pass

    def _project_path(self):
        base = os.path.splitext(os.path.basename(self.image_path))[0]
        return os.path.join(f"{os.path.splitext(self.image_path)[0]}_files",
                            f"{base}_stim_project.json")

    def _to_full(self, pts):
        p = np.asarray(pts, dtype=float).reshape(-1, 2)
        if self.geom is None:
            return [[float(x), float(y)] for x, y in p]
        fx, fy = self.geom.ds_to_full(p[:, 0], p[:, 1])
        return [[float(a), float(b)] for a, b in zip(fx, fy)]

    def _to_overview(self, pts):
        p = np.asarray(pts, dtype=float).reshape(-1, 2)
        if self.geom is None:
            return p
        x, y = self.geom.full_to_ds(p[:, 0], p[:, 1])
        return np.column_stack([x, y])

    def current_polygons_xy(self):
        if self.shapes_layer is None:
            return []
        return [napari_to_xy(d) for d in self.shapes_layer.data
                if len(np.asarray(d)) >= 3]

    def current_fiducials_xy(self):
        if self.fid_layer is None or len(self.fid_layer.data) == 0:
            return []
        return [tuple(map(float, napari_to_xy(p).ravel())) for p in self.fid_layer.data]

    def current_raw_xy(self):
        """All UNFILTERED detections (the 'Raw detections' layer), overview px."""
        lyr = getattr(self, "raw_layer", None)
        if lyr is None or lyr not in self.viewer.layers:
            return []
        return [napari_to_xy(d) for d in lyr.data if len(np.asarray(d)) >= 3]

    def save_project(self):
        if self.image_path is None:
            return
        try:
            data = {"image": self.image_path,
                    "sections": [self._to_full(p) for p in self.current_polygons_xy()],
                    "fiducials": [self._to_full([f])[0] for f in self.current_fiducials_xy()],
                    # full unfiltered detector output, kept for re-filtering / QC
                    "raw_sections": [self._to_full(p) for p in self.current_raw_xy()],
                    "calibration_examples": [self._to_full(p) for p in self.current_calib_xy()]}
            path = self._project_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def load_project(self):
        if self.image_path is None:
            return [], []
        path = self._project_path()
        if not os.path.isfile(path):
            return [], []
        try:
            data = json.load(open(path))
        except Exception:
            return [], []
        polys = [self._to_overview(s) for s in data.get("sections", [])]
        fids = [tuple(self._to_overview([f])[0]) for f in data.get("fiducials", [])]
        self._restored_raw_xy = [self._to_overview(s) for s in data.get("raw_sections", [])]
        self._restored_calib_xy = [self._to_overview(s) for s in data.get("calibration_examples", [])]
        return polys, fids

    def _load_legacy_csv(self):
        """Load sections from the original ``*_mask_coordinates.csv`` (overview
        coords assumed = image coords, i.e. for non-CZI images)."""
        base = os.path.splitext(os.path.basename(self.image_path))[0]
        path = os.path.join(f"{os.path.splitext(self.image_path)[0]}_files",
                            f"{base}_mask_coordinates.csv")
        if not os.path.isfile(path):
            return [], []
        polys, fids = [], []
        try:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    rtype = (row.get("type") or "").lower()
                    coords = row.get("contour_coordinates") or ""
                    if rtype == "fiducials":
                        try:
                            fids = [tuple(map(float, p)) for p in ast.literal_eval(coords)]
                        except Exception:
                            pass
                        continue
                    try:
                        contours = ast.literal_eval(coords)  # [[ [x,y],... ], ...]
                        cnt = max(contours, key=len)
                        poly = np.asarray(cnt, dtype=float).reshape(-1, 2)
                        if len(poly) >= 3:
                            polys.append(poly)
                    except Exception:
                        continue
        except Exception:
            return [], []
        return polys, fids

    # ----- detection (separate process) -----
    def run_auto(self):
        if self.overview is None:
            self.log_msg("⚠️ Select an image first."); return
        if not os.path.isfile(self.checkpoint):
            QMessageBox.information(self, "Missing checkpoint",
                                    f"SAM 2.1 checkpoint not found:\n{self.checkpoint}")
            self.select_checkpoint()
            if not os.path.isfile(self.checkpoint):
                return
        if self.proc is not None and self.proc.state() != QProcess.NotRunning:
            self.log_msg("Detection already running — press Stop first."); return

        # One streaming engine (SAM's whole-image generator can't stream, so we
        # always tile — often a single whole-image tile). Every SAM parameter
        # comes from the (calibrated) Advanced fields; the host profile picks the
        # model + caps the tile so the run stays feasible on this machine.
        prof = self._current_profile()
        cal = self.calibration or {}
        tile_px = int(self.sp_tile.value())
        if tile_px <= 0:
            tile_px = max(self.overview.shape[:2])          # whole image
        tile_px = int(min(tile_px, prof.tile_cap_px))       # memory cap may force tiling
        min_area = float(cal.get("min_area", self.sp_minarea.value() or 50))
        max_area = float(cal.get("max_area", 1e12))
        ckpt = self._checkpoint_for_model(self.cb_model.currentText(), prof)
        args = ["-m", "section_identification.detect_worker",
                "--image", self.image_path, "--checkpoint", ckpt,
                "--device", getattr(self, "_device_prefer", "") or "",
                "--target-long-side", str(self.sp_target.value()),
                "--points-per-side", str(self.sp_pps.value()),
                "--points-per-batch", str(self.sp_ppb.value()),
                "--pred-iou-thresh", str(self.sp_iou.value()),
                "--stability-score-thresh", str(self.sp_stab.value()),
                "--stability-score-offset", str(self.sp_staboff.value()),
                "--box-nms-thresh", str(self.sp_boxnms.value()),
                "--crop-n-layers", str(self.sp_crop.value()),
                "--crop-overlap-ratio", str(self.sp_cropov.value()),
                "--crop-n-points-downscale-factor", str(self.sp_cropds.value()),
                "--min-mask-region-area", str(self.sp_minmask.value()),
                "--use-m2m", "1" if self.chk_m2m.isChecked() else "0",
                "--multimask", "0" if self.chk_lowmem.isChecked() else "1",
                "--tile-px", str(tile_px), "--overlap", str(self.sp_overlap.value()),
                "--min-area", str(min_area), "--max-area", str(max_area)]
        self._stream_mode = True
        self._reset_stream_layers(); self._raw_sections = []; self._det_params = None
        whole = tile_px >= max(self.overview.shape[:2])
        self.log_msg(f"▶ Detection on {prof.device} ({os.path.basename(ckpt)}): "
                     f"{'whole image' if whole else 'tiles'}, tile_px={tile_px}, grid "
                     f"{self.sp_pps.value()}, area {min_area:.0f}–{max_area:.0f}.")
        self._proc_buf = ""

        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self.proc.setProcessEnvironment(env)
        # Run the worker from a neutral cwd: the in-repo sam2/ dir would otherwise
        # shadow the installed `sam2` package when cwd == repo root.
        self.proc.setWorkingDirectory(os.path.expanduser("~"))
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_proc_output)
        self.proc.finished.connect(self._on_proc_finished)
        self.proc.errorOccurred.connect(lambda e: self.log_msg(f"❌ process error: {e}"))
        self.btn_auto.setEnabled(False); self.btn_stop.setVisible(True)
        self.progress.setRange(0, 0); self.progress.setVisible(True)
        self._det_t0 = time.time(); self._elapsed_timer.start(1000)
        self.log_msg("(running in a background process — GUI stays responsive; Stop to cancel)")
        self.proc.start(sys.executable, args)

    def _tick_elapsed(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.lbl_elapsed.setText(f"⏱ {int(time.time() - self._det_t0)} s elapsed")

    def _on_proc_output(self):
        try:
            text = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        except Exception:
            return
        self._proc_buf += text
        *lines, self._proc_buf = self._proc_buf.split("\n")
        for line in lines:
            line = line.rstrip()
            if not line:
                continue
            if (line.startswith("STIM_TILES ") or line.startswith("STIM_TILE ")
                    or line.startswith("STIM_TILESTART ")):
                try:
                    self._handle_stim_line(line)
                except Exception:
                    pass
            else:
                self.log_msg(line)

    def _on_proc_finished(self, code, status):
        self._elapsed_timer.stop(); self.lbl_elapsed.setText("")
        self.btn_auto.setEnabled(True); self.btn_stop.setVisible(False)
        self.progress.setVisible(False)
        self.proc = None
        if code != 0:
            self.log_msg(f"⏹ detection stopped/failed (exit {code}); partials kept.")
            self._finalize_tiled()
            self._clear_detector_overlays()
            return
        try:
            self._finalize_tiled()
        except Exception:
            self.log_msg("❌ loading results failed:\n" + traceback.format_exc())
        self._clear_detector_overlays()

    def _clear_detector_overlays(self):
        """Once a run ends, remove the detector's transient overlays — the tile
        grid, current tile, and the live parameter previews (grid/crops/min-area).
        Keep 'Raw detections' (the result) and 'Sections'."""
        for attr in ("tiles_layer", "current_tile_layer"):
            lyr = getattr(self, attr, None)
            try:
                if lyr is not None and lyr in self.viewer.layers:
                    self.viewer.layers.remove(lyr)
            except Exception:
                pass
            setattr(self, attr, None)
        # keep Raw detections (the result) but hide it by default — the kept
        # 'Sections' layer is what the user works with.
        if getattr(self, "raw_layer", None) is not None:
            try:
                self.raw_layer.visible = False
            except Exception:
                pass
        if getattr(self, "_param_viz", None) is not None:
            try:
                self._param_viz.set_active(False)
            except Exception:
                pass
        try:
            self.chk_viz.setChecked(False)
        except Exception:
            pass

    def stop_detection(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.log_msg("■ Stopping…"); self.proc.kill()

    # ----- tiled streaming display -----
    def _set_shapes(self, attr, name, data, edge="white", face=(0, 0, 0, 0), width=3):
        lyr = getattr(self, attr, None)
        if lyr is not None and lyr in self.viewer.layers:
            self.viewer.layers.remove(lyr)
        lyr = self.viewer.add_shapes(data, shape_type="polygon", name=name,
                                     face_color=list(face), edge_width=width,
                                     scale=self._layer_scale())
        try:
            lyr.edge_color = edge
        except Exception:
            pass
        setattr(self, attr, lyr)
        return lyr

    @staticmethod
    def _box_rect(box):
        x, y, w, h = box
        return np.array([[y, x], [y, x + w], [y + h, x + w], [y + h, x]], dtype=float)

    def _reset_stream_layers(self):
        for attr in ("tiles_layer", "current_tile_layer", "raw_layer"):
            lyr = getattr(self, attr, None)
            if lyr is not None and lyr in self.viewer.layers:
                try:
                    self.viewer.layers.remove(lyr)
                except Exception:
                    pass
            setattr(self, attr, None)

    def _handle_stim_line(self, line):
        if line.startswith("STIM_TILESTART "):
            d = json.loads(line[len("STIM_TILESTART "):])
            # Show the tile being worked on NOW (before SAM runs on it).
            self._set_shapes("current_tile_layer", "Current tile",
                             [self._box_rect(d["box"])],
                             edge="cyan", face=(0, 1, 1, 0.12), width=4)
            self.log_msg(f"  tile {d['k']}/{d['n']} — segmenting…")
            return
        if line.startswith("STIM_TILES "):
            boxes = json.loads(line[len("STIM_TILES "):])
            self._set_shapes("tiles_layer", "Tiles", [self._box_rect(b) for b in boxes],
                             edge="yellow", face=(1, 1, 0, 0.06), width=2)
            self.log_msg(f"Tiling into {len(boxes)} tiles…")
            return
        d = json.loads(line[len("STIM_TILE "):])
        k, n = d["k"], d["n"]
        # "Current tile" was already drawn by STIM_TILESTART; here we just stream
        # the sections this tile confirmed.
        new = [xy_to_napari(np.asarray(s["poly"], dtype=float)) for s in d["sections"]
               if len(s["poly"]) >= 3]
        for s in d["sections"]:
            if len(s["poly"]) >= 3:
                self._raw_sections.append(s)
        if new:
            if self.raw_layer is None or self.raw_layer not in self.viewer.layers:
                self._set_shapes("raw_layer", "Raw detections", new,
                                 edge="orange", face=(1, 0.55, 0, 0.25), width=3)
            else:
                try:
                    self.raw_layer.add(new, shape_type="polygon")
                except Exception:
                    self.raw_layer.data = list(self.raw_layer.data) + new
        elapsed = max(1e-3, time.time() - self._det_t0)
        eta = elapsed / k * (n - k)
        self.log_msg(f"  tile {k}/{n} · +{len(d['sections'])} (total "
                     f"{len(self._raw_sections)}) · ~{int(eta)}s left")

    def _finalize_tiled(self):
        # `raw` = ALL SAM output (debris included) — kept in the Raw layer for QC.
        # Sections = raw → size band → DBSCAN.
        raw = self._raw_sections
        cal = self.calibration or {}
        lo = float(cal.get("min_area", 0.0))
        hi = float(cal.get("max_area", float("inf")))
        band = [s for s in raw if lo <= float(s["area"]) <= hi]
        kept = band
        if self.chk_filter.isChecked() and len(band) >= 3:
            try:
                from section_identification.filtering import filtering
                ml = [{"area": float(s["area"])} for s in band]
                alo = max(1.0, min(m["area"] for m in ml)); ahi = max(m["area"] for m in ml) + 1
                chosen, _ = filtering(ml, np.linspace(alo, ahi, 12), range(2, 5))
                ids = {id(m) for m in chosen}
                kept = [s for s, m in zip(band, ml) if id(m) in ids]
            except Exception:
                kept = band
        self._ensure_edit_layers([np.asarray(s["poly"], dtype=float) for s in kept])
        self.log_msg(f"✔️ {len(raw)} raw (all SAM, incl. debris) → {len(band)} in "
                     f"size band [{lo:.0f}–{hi:.0f}] → {len(kept)} kept → 'Sections'. "
                     "Raw layer keeps everything (hidden; toggle for QC).")
        self.save_project()

    # ----- calibration + preview -----
    def calibrate_from_examples(self):
        if self.overview is None:
            self.log_msg("⚠️ Load an image first."); return
        lyr = self._ensure_calib_layer()
        polys = [napari_to_xy(d) for d in lyr.data if len(np.asarray(d)) >= 3]
        if not polys:
            self.log_msg("Draw 2–5 example sections in 'Calibration examples' first.")
            try:
                self.viewer.layers.selection.active = lyr
            except Exception:
                pass
            return
        from section_identification.calibration import calibrate, summary
        try:
            prof = self._current_profile()
            self.calibration = calibrate(
                polys, geom=self.geom, overview_long_side=max(self.overview.shape[:2]),
                target_sam_px=self.sp_targetsam.value(), profile=prof)
        except Exception:
            self.log_msg("❌ calibration failed:\n" + traceback.format_exc()); return
        self._apply_calibration_to_ui(self.calibration, prof)
        self.log_msg("✔️ " + summary(self.calibration))
        self.log_msg("→ " + self.calibration.get("plan_summary", ""))
        # SAM runs on the overview; recommend a finer one if real detail is poor.
        rec = self.calibration.get("recommended_overview_long_side")
        cur = max(self.overview.shape[:2])
        if rec and rec > cur * 1.3:
            self.sp_target.setValue(int(rec))
            self.log_msg(f"⚠️ Sections are only ~{self.calibration['section_px']:.0f}px in "
                         f"this {cur}px overview. Overview long side set to {rec}; "
                         "RELOAD the image for real detail, then re-calibrate before running.")

    def _apply_calibration_to_ui(self, cal, prof=None):
        """Write the calibrated SAM parameters into the Advanced fields + plan."""
        def setv(widget, key, cast):
            if key in cal:
                try:
                    widget.setValue(cast(cal[key]))
                except Exception:
                    pass
        setv(self.sp_pps, "points_per_side", int)
        setv(self.sp_iou, "pred_iou_thresh", float)
        setv(self.sp_stab, "stability_score_thresh", float)
        setv(self.sp_staboff, "stability_score_offset", float)
        setv(self.sp_boxnms, "box_nms_thresh", float)
        setv(self.sp_crop, "crop_n_layers", int)
        setv(self.sp_cropov, "crop_overlap_ratio", float)
        setv(self.sp_cropds, "crop_n_points_downscale_factor", int)
        setv(self.sp_minmask, "min_mask_region_area", int)
        setv(self.sp_minarea, "min_area", int)          # DBSCAN area floor
        setv(self.sp_overlap, "overlap", float)
        # tile_px: 0 (whole image) unless tiling is recommended
        self.sp_tile.setValue(int(cal.get("tile_px", 0)) if cal.get("tiling_recommended") else 0)
        if prof is not None:
            self.sp_ppb.setValue(int(prof.points_per_batch))
            self.lbl_host.setText("Host: " + prof.summary())
        self.lbl_plan.setText("Plan: " + cal.get("plan_summary", ""))

    def preview_tiling(self):
        if self.overview is None:
            self.log_msg("⚠️ Load an image first."); return
        from section_identification.tiled_detect import plan_tiles
        H, W = self.overview.shape[:2]
        tile_px = int(self.calibration["tile_px"]) if self.calibration else 768
        overlap = float(self.calibration.get("overlap", 0.25)) if self.calibration else 0.25
        boxes = plan_tiles(W, H, tile_px, overlap)
        self._set_shapes("tiles_layer", "Tiles (preview)",
                         [self._box_rect(b) for b in boxes], edge="yellow",
                         face=(1, 1, 0, 0.10), width=2)
        self.log_msg(f"Preview: {len(boxes)} tiles of {tile_px}px (overlap {overlap}); "
                     f"SAM upscale ×{1024.0 / tile_px:.1f}.")

    # ----- manual cv2 editor -----
    def _image_file_for_interactive(self):
        if not czi_io.is_czi(self.image_path):
            return self.image_path
        import cv2
        base = os.path.splitext(self.image_path)[0]
        out_dir = f"{base}_files"; os.makedirs(out_dir, exist_ok=True)
        png = os.path.join(out_dir, os.path.basename(base) + "_overview.png")
        cv2.imwrite(png, cv2.cvtColor(self.overview, cv2.COLOR_RGB2BGR))
        return png

    def run_manual(self):
        if self.overview is None:
            self.log_msg("⚠️ Load an image first."); return
        if not os.path.isfile(self.sam1_checkpoint):
            QMessageBox.information(self, "Missing SAM 1 checkpoint",
                                    f"Manual editor needs:\n{self.sam1_checkpoint}")
            p, _ = QFileDialog.getOpenFileName(self, "Select SAM 1 checkpoint", "",
                                               "Checkpoints (*.pth *.pt)")
            if not p:
                return
            self.sam1_checkpoint = p
            self.sam1_model_type = ("vit_h" if "vit_h" in p else
                                    "vit_l" if "vit_l" in p else "vit_b")
        try:
            from section_identification.interactive import run_sam_interactive
            from section_identification.interactive_helpers import display_help
        except Exception:
            QMessageBox.warning(self, "Manual editor unavailable",
                                "Needs onnxruntime + segment-anything installed.")
            return
        img_path = self._image_file_for_interactive()
        self.progress.setRange(0, 0); self.progress.setVisible(True); QApplication.processEvents()
        try:
            display_help()
            self.log_msg("▶ Launching manual editor (separate window; Esc to finish)…")
            # For a CZI, hand the editor the source path + geometry so its 'e'
            # key can read the current view at full resolution, and the existing
            # sections as reference outlines.
            czi_p = self.image_path if czi_io.is_czi(self.image_path) else None
            ref = self.current_polygons_xy() if czi_p else None
            new_masks, stored_masks, fiducials = run_sam_interactive(
                img_path, checkpoint=self.sam1_checkpoint, stored_masks=[],
                model_type=self.sam1_model_type, device=device_str(),
                czi_path=czi_p, geom=self.geom, ref_polygons=ref)
            self.log_msg(f"✔️ Manual: {len(new_masks)} new, {len(fiducials)} fiducials.")
            new_polys = []
            for m in list(stored_masks) + list(new_masks):
                # Full-res masks carry their polygon already in overview coords.
                po = m.get("poly_overview") if isinstance(m, dict) else None
                p = np.asarray(po, dtype=float) if po is not None \
                    else mask_to_polygon(m["segmentation"])
                if p is not None and len(p) >= 3:
                    new_polys.append(p)
            # `ref` was mutated in place by the editor: detections the user
            # deleted with 'r' are gone, so rebuild from the survivors (+ new).
            survivors = ref if ref is not None else self.current_polygons_xy()
            self._ensure_edit_layers(list(survivors) + new_polys)
            if fiducials and self.fid_layer is not None:
                self.fid_layer.data = np.asarray(fiducials, dtype=float)[:, ::-1]
            self.save_project()
        except Exception:
            self.log_msg("❌ manual editor error:\n" + traceback.format_exc())
        finally:
            self.progress.setVisible(False)

    def toggle_manual_napari(self):
        """Activate/deactivate the in-viewer (napari) SAM editor."""
        if self.overview is None:
            self.log_msg("⚠️ Load an image first.")
            self.btn_manual_napari.setChecked(False)
            return
        try:
            if getattr(self, "_napari_editor", None) is None:
                from section_identification.napari_sam_editor import NapariSamEditor
                self._napari_editor = NapariSamEditor(self)
            active = self._napari_editor.toggle()
            self.btn_manual_napari.setChecked(bool(active))
        except Exception:
            self.btn_manual_napari.setChecked(False)
            self.log_msg("❌ napari editor error:\n" + traceback.format_exc())

    # ----- export -----
    def export_coordinates(self):
        if self.image_path is None:
            self.log_msg("⚠️ Nothing to export."); return
        polys = self.current_polygons_xy()
        if not polys:
            self.log_msg("⚠️ No section polygons to export."); return
        section_ids = [f"section_{k}" for k in range(1, len(polys) + 1)]
        fids = self.current_fiducials_xy()
        fmts = dict(write_csv=self.chk_exp_csv.isChecked(),
                    write_geojson=self.chk_exp_geojson.isChecked(),
                    write_png=self.chk_exp_png.isChecked(),
                    write_czi=self.chk_exp_czi.isChecked())
        if not any(fmts.values()):
            self.log_msg("⚠️ Select at least one export format (CSV/GeoJSON/PNG/CZI)."); return
        chosen = ", ".join(k[6:].upper() for k, v in fmts.items() if v)
        self.log_msg(f"▶ Exporting {len(polys)} sections, {len(fids)} fiducials → {chosen}…")
        try:
            outputs = export_polygons(self.image_path, polys, fids, geom=self.geom,
                                      section_ids=section_ids, **fmts)
            out_dir = outputs.get("dir")
            default_dir = f"{os.path.splitext(self.image_path)[0]}_files"
            if out_dir and os.path.abspath(out_dir) != os.path.abspath(default_dir):
                self.log_msg(f"ℹ️ Source folder isn't writable (read-only drive?) — "
                             f"exported to {out_dir} instead.")
            files = {k: v for k, v in outputs.items() if k != "dir"}
            self.log_msg("✔️ Exported: " + ", ".join(f"{k}={v}" for k, v in files.items()))
        except Exception:
            self.log_msg("❌ export error:\n" + traceback.format_exc())


def main():
    # Avoid the in-repo sam2/ dir shadowing the installed `sam2` package: never
    # run from the repo root. All app paths are absolute, so this is safe.
    try:
        os.chdir(os.path.expanduser("~"))
    except Exception:
        pass
    viewer = napari.Viewer()
    gui = SectionIdentificationGUI(viewer)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    # Scroll vertically only — content reflows to the panel width instead of
    # forcing a horizontal scrollbar (keeps the right dock narrow & readable).
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setWidget(gui)
    scroll.setMinimumWidth(230)        # allow a narrow dock; long text wraps
    dock = viewer.window.add_dock_widget(scroll, name="STiM", area="right")
    # Open the dock narrow, then release the cap so the user can still drag it.
    try:
        dock.setMaximumWidth(320)
        QTimer.singleShot(300, lambda: dock.setMaximumWidth(16777215))
    except Exception:
        pass
    napari.run()


if __name__ == "__main__":
    main()
