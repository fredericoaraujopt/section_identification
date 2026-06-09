"""STiM unified napari GUI.

One window for the whole workflow: load an image (including whole-slide ``.czi``,
read from the pyramid), run SAM 2.1 automatic detection, **edit the section
polygons and fiducials natively in napari** (a Shapes layer + a Points layer —
no more separate OpenCV window), recover serial order by cross-correlation via a
reorderable filmstrip, and export CSV / GeoJSON / a ZEN-annotated CZI.

Coordinate convention: napari layer data is ``(row, col)`` = ``(y, x)``; our
detection/export code uses ``(x, y)``. The helpers below convert at the boundary.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, QSize, QTimer, QProcess, QProcessEnvironment
from qtpy.QtGui import QIcon, QImage, QPixmap
from qtpy.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QProgressBar, QPushButton, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

import napari

from section_identification.section_detector import automatic_identification
from section_identification.export import export_polygons, mask_to_polygon
from section_identification import czi_io, ordering
from section_identification.device import describe as describe_device, device_str
# NB: run_sam_interactive / display_help are imported lazily inside run_manual,
# because interactive.py imports onnxruntime at module load — keeping it lazy
# means the GUI still starts even if the manual-editor deps aren't installed.


# --------------------------------------------------------------------------- #
# Coordinate / image helpers
# --------------------------------------------------------------------------- #
def xy_to_napari(poly_xy):
    """(x, y) Nx2 -> napari (y, x) Nx2."""
    p = np.asarray(poly_xy, dtype=float).reshape(-1, 2)
    return p[:, ::-1]


def napari_to_xy(poly_yx):
    """napari (y, x) Nx2 -> (x, y) Nx2."""
    p = np.asarray(poly_yx, dtype=float).reshape(-1, 2)
    return p[:, ::-1]


def numpy_to_qicon(arr, size=72):
    """uint8 RGB crop -> QIcon thumbnail."""
    import cv2

    a = arr
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    a = np.ascontiguousarray(cv2.resize(a.astype(np.uint8), (size, size)))
    h, w, _ = a.shape
    img = QImage(a.data, w, h, 3 * w, QImage.Format_RGB888)
    return QIcon(QPixmap.fromImage(img.copy()))


class SectionIdentificationGUI(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.image_path = None
        self.overview = None       # uint8 RGB overview
        self.geom = None           # CziGeometry (None for non-CZI)
        self.masks = []            # detected mask dicts (overview coords)
        self.image_layer = None
        self.shapes_layer = None
        self.fid_layer = None
        # Debounced autosave: edits schedule a save 1.5 s later (so dragging a
        # vertex doesn't write the project file on every mouse move).
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self.save_project)
        # Detection runs in a separate process (no GUI freeze, clean Stop).
        self.proc = None
        self._det_params = None
        self._det_t0 = 0.0
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)
        # Calibration + tiled-streaming state.
        self.calibration = None
        self.calib_layer = None          # 'Calibration examples' Shapes layer
        self.tiles_layer = None          # tile-grid preview
        self.current_tile_layer = None   # highlighted current tile
        self.raw_layer = None            # live raw detections (tiled streaming)
        self._raw_sections = []          # accumulated {poly(overview), area}
        self._stream_mode = False
        self._proc_buf = ""              # stdout line buffer for the worker

        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(QLabel(f"<b>Device:</b> {describe_device()}"))

        # --- File selection ---
        self.btn_select = QPushButton("Select Image / CZI…")
        self.lbl_path = QLabel("No image selected")
        self.lbl_path.setWordWrap(True)
        layout.addWidget(self.btn_select)
        layout.addWidget(self.lbl_path)

        # --- SAM parameters ---
        layout.addWidget(QLabel("<b>SAM 2.1 parameters</b>"))
        form = QFormLayout()
        self.sp_pps = QSpinBox(); self.sp_pps.setRange(4, 128); self.sp_pps.setValue(32)
        self.sp_ppb = QSpinBox(); self.sp_ppb.setRange(8, 256); self.sp_ppb.setValue(64)
        self.sp_iou = QDoubleSpinBox(); self.sp_iou.setRange(0.0, 1.0)
        self.sp_iou.setSingleStep(0.05); self.sp_iou.setValue(0.80)
        self.sp_minarea = QSpinBox(); self.sp_minarea.setRange(0, 100000)
        self.sp_minarea.setValue(20)
        self.sp_crop = QSpinBox(); self.sp_crop.setRange(0, 3); self.sp_crop.setValue(1)
        self.sp_target = QSpinBox(); self.sp_target.setRange(1024, 16384)
        self.sp_target.setSingleStep(512); self.sp_target.setValue(3072)
        form.addRow("points_per_side", self.sp_pps)
        form.addRow("points_per_batch (mem)", self.sp_ppb)
        form.addRow("pred_iou_thresh", self.sp_iou)
        form.addRow("min_mask_region_area", self.sp_minarea)
        form.addRow("crop_n_layers (small-section recall)", self.sp_crop)
        form.addRow("overview long side (px)", self.sp_target)
        layout.addLayout(form)

        self.chk_filter = QCheckBox("Filter for sections (area DBSCAN)")
        self.chk_filter.setChecked(True)
        layout.addWidget(self.chk_filter)
        self.chk_tiled = QCheckBox(
            "Tiled streaming detector (challenging wafers: bounded memory, live view)")
        layout.addWidget(self.chk_tiled)

        # --- Calibration from drawn examples ---
        cal_row = QHBoxLayout()
        self.btn_calibrate = QPushButton("Calibrate from examples")
        self.btn_preview = QPushButton("Preview tiling + points")
        cal_row.addWidget(self.btn_calibrate)
        cal_row.addWidget(self.btn_preview)
        layout.addLayout(cal_row)
        layout.addWidget(QLabel(
            "<i>Draw 2-5 example sections in the 'Calibration examples' layer, then "
            "Calibrate to auto-set section size, area filter and tile size. Preview "
            "shows how SAM will tile/sample the wafer before running.</i>"))
        self.lbl_calib = QLabel("Not calibrated.")
        self.lbl_calib.setWordWrap(True)
        layout.addWidget(self.lbl_calib)

        det_row = QHBoxLayout()
        self.btn_auto = QPushButton("Run Automatic Detection")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setVisible(False)
        det_row.addWidget(self.btn_auto)
        det_row.addWidget(self.btn_stop)
        layout.addLayout(det_row)
        self.lbl_elapsed = QLabel("")
        layout.addWidget(self.lbl_elapsed)

        # --- Manual detector (original SAM-assisted ONNX editor) ---
        layout.addWidget(QLabel("<b>Manual Detector</b>"))
        self.btn_manual = QPushButton("Launch Manual Detector")
        layout.addWidget(self.btn_manual)
        layout.addWidget(QLabel(
            "<i>Opens the interactive SAM editor: hover to preview a mask, click "
            "to add; 'r' to remove; 'm' for fiducials; Esc to finish. Results load "
            "back into the 'Sections' layer.</i>"))

        # --- Ordering filmstrip ---
        layout.addWidget(QLabel("<b>Serial order (drag to reorder)</b>"))
        self.btn_order = QPushButton("Auto-order by cross-correlation")
        layout.addWidget(self.btn_order)
        self.filmstrip = QListWidget()
        self.filmstrip.setViewMode(QListWidget.IconMode)
        self.filmstrip.setFlow(QListWidget.LeftToRight)
        self.filmstrip.setWrapping(True)
        self.filmstrip.setIconSize(QSize(72, 72))
        self.filmstrip.setDragDropMode(QListWidget.InternalMove)
        self.filmstrip.setFixedHeight(120)
        layout.addWidget(self.filmstrip)
        self.btn_refresh_strip = QPushButton("Refresh filmstrip from polygons")
        layout.addWidget(self.btn_refresh_strip)

        # --- Export ---
        self.btn_export = QPushButton("Export (CSV + GeoJSON + annotated CZI)")
        layout.addWidget(self.btn_export)

        # --- Checkpoints ---
        pkg = Path(os.path.abspath(__file__))
        ckpt_dir = pkg.parents[1] / "checkpoint"
        self.checkpoint = str(ckpt_dir / "sam2.1_hiera_base_plus.pt")  # SAM 2.1 (auto)
        # SAM 1 checkpoint used by the manual ONNX editor (vit_b by default).
        self.sam1_checkpoint = str(ckpt_dir / "sam_vit_b_01ec64.pth")
        self.sam1_model_type = "vit_b"
        self.lbl_ckpt = QLabel(f"Checkpoint: {self.checkpoint}")
        self.lbl_ckpt.setWordWrap(True)
        layout.addWidget(self.lbl_ckpt)
        self.btn_ckpt = QPushButton("Select SAM 2.1 Checkpoint (.pt)")
        layout.addWidget(self.btn_ckpt)

        # --- Log ---
        layout.addWidget(QLabel("<b>Log</b>"))
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(160)
        layout.addWidget(self.log)
        self.progress = QProgressBar(); self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # stdout -> log
        self._old_stdout = sys.stdout
        sys.stdout = self

        # signals
        self.btn_select.clicked.connect(self.select_image)
        self.btn_auto.clicked.connect(self.run_auto)
        self.btn_order.clicked.connect(self.auto_order)
        self.btn_refresh_strip.clicked.connect(self.rebuild_filmstrip)
        self.btn_export.clicked.connect(self.export_coordinates)
        self.btn_ckpt.clicked.connect(self.select_checkpoint)
        self.btn_manual.clicked.connect(self.run_manual)
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_calibrate.clicked.connect(self.calibrate_from_examples)
        self.btn_preview.clicked.connect(self.preview_tiling)

    # ----- logging plumbing -----
    def write(self, text):
        self._old_stdout.write(text); self._old_stdout.flush()
        if text.strip():
            self.log.append(text.rstrip()); QApplication.processEvents()

    def flush(self):
        pass

    def log_msg(self, text):
        print(text)

    # ----- checkpoint -----
    def select_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SAM 2.1 checkpoint", "",
            "Checkpoints (*.pt *.pth)")
        if path:
            self.checkpoint = path
            self.lbl_ckpt.setText(f"Checkpoint: {path}")

    # ----- load image -----
    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select an image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.czi)")
        if not path:
            return
        self.image_path = path
        self.lbl_path.setText(f"Selected: {path}")
        self.log_msg(f"Loading {os.path.basename(path)}…")
        try:
            if czi_io.is_czi(path):
                arr, geom, meta = czi_io.read_czi_overview(
                    path, target_long_side=self.sp_target.value())
                self.geom = geom
                self.overview = czi_io.to_rgb8(arr)
                self.log_msg(f"CZI full {meta['size_x']}x{meta['size_y']}, "
                             f"overview {self.overview.shape}, zoom {meta['zoom']:.4g}")
            else:
                from PIL import Image
                self.overview = np.array(Image.open(path).convert("RGB"))
                self.geom = None
        except Exception:
            self.log_msg("❌ load failed:\n" + traceback.format_exc())
            return

        self._reset_layers()
        self.image_layer = self.viewer.add_image(self.overview, name="Overview")

        # Restore the working session. Prefer the autosaved project (the most
        # recent state); fall back to annotations baked into a CZI. Everything
        # here is best-effort: a failure must NOT prevent the image from opening.
        polys_xy, fids_xy = [], []
        proj_polys, proj_fids = self.load_project()
        if proj_polys is not None and (proj_polys or proj_fids):
            polys_xy, fids_xy = proj_polys, proj_fids
            self.log_msg(f"Restored {len(polys_xy)} sections + {len(fids_xy)} "
                         f"fiducials from autosaved project.")
        elif self.geom is not None and czi_io.is_czi(path):
            try:
                from section_identification.czi_export import read_annotations
                polys_full, fids_full = read_annotations(path)
                for pf in polys_full:
                    p = np.asarray(pf, dtype=float)
                    xd, yd = self.geom.full_to_ds(p[:, 0], p[:, 1])
                    polys_xy.append(np.column_stack([xd, yd]))
                for (xf, yf) in fids_full:
                    xd, yd = self.geom.full_to_ds(xf, yf)
                    fids_xy.append((float(xd), float(yd)))
                if polys_xy or fids_xy:
                    self.log_msg(f"Loaded {len(polys_xy)} polygons + "
                                 f"{len(fids_xy)} fiducials from CZI annotations.")
            except Exception:
                self.log_msg("[warn] could not read CZI annotations:\n"
                             + traceback.format_exc())
                polys_xy, fids_xy = [], []

        try:
            self._ensure_edit_layers(polys_xy)
            if fids_xy and self.fid_layer is not None:
                self.fid_layer.data = np.asarray(fids_xy, dtype=float)[:, ::-1]
        except Exception:
            self.log_msg("[warn] building annotation layers failed; starting "
                         "empty:\n" + traceback.format_exc())
            self._ensure_edit_layers([])
        self.masks = []
        self._raw_sections = []
        self.filmstrip.clear()
        self._ensure_calib_layer()  # ready for the user to draw examples
        try:
            if polys_xy:
                self.rebuild_filmstrip()
        except Exception:
            self.log_msg("[warn] filmstrip build failed (annotations still loaded).")

    def _reset_layers(self):
        for lyr in list(self.viewer.layers):
            self.viewer.layers.remove(lyr)
        self.image_layer = self.shapes_layer = self.fid_layer = None
        self.calib_layer = self.tiles_layer = None
        self.current_tile_layer = self.raw_layer = None

    def _ensure_edit_layers(self, polygons_xy):
        """(Re)create the editable Sections (Shapes) and Fiducials (Points) layers.

        napari renamed ``edge_color`` -> ``border_color`` on Points across
        versions, so we create the layers minimally and set styling afterwards,
        tolerating whichever name this napari build uses.
        """
        if self.shapes_layer is not None and self.shapes_layer in self.viewer.layers:
            self.viewer.layers.remove(self.shapes_layer)
        data = [xy_to_napari(p) for p in polygons_xy] if polygons_xy else []
        self.shapes_layer = self.viewer.add_shapes(
            data, shape_type="polygon", name="Sections",
            face_color=[1, 0, 0, 0.12], edge_width=3)
        try:
            self.shapes_layer.edge_color = "red"
        except Exception:
            pass
        if self.fid_layer is None or self.fid_layer not in self.viewer.layers:
            self.fid_layer = self.viewer.add_points(
                np.empty((0, 2)), name="Fiducials", size=20)
            for attr, val in (("face_color", "cyan"), ("border_color", "blue"),
                              ("edge_color", "blue")):
                try:
                    setattr(self.fid_layer, attr, val)
                except Exception:
                    pass
        # Autosave the working session whenever the user edits polygons/fiducials.
        for layer in (self.shapes_layer, self.fid_layer):
            try:
                layer.events.data.connect(self._schedule_autosave)
            except Exception:
                pass

    def _schedule_autosave(self, *args):
        try:
            self._autosave_timer.start(1500)
        except Exception:
            pass

    # ----- project autosave / restore -----
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

    def save_project(self):
        """Persist the current sections + fiducials (full-res coords) to JSON.

        Best-effort and silent: autosave must never crash or interrupt the GUI.
        """
        if self.image_path is None:
            return
        try:
            data = {
                "image": self.image_path,
                "sections": [self._to_full(p) for p in self.current_polygons_xy()],
                "fiducials": [self._to_full([f])[0] for f in self.current_fiducials_xy()],
            }
            path = self._project_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def load_project(self):
        """Return (polygons_overview, fiducials_overview) from the autosaved
        project file, or (None, None) if there isn't one."""
        if self.image_path is None:
            return None, None
        path = self._project_path()
        if not os.path.isfile(path):
            return None, None
        try:
            data = json.load(open(path))
        except Exception:
            return None, None
        polys = [self._to_overview(s) for s in data.get("sections", [])]
        fids = [tuple(self._to_overview([f])[0]) for f in data.get("fiducials", [])]
        return polys, fids

    # ----- detection (runs in a separate process; GUI stays responsive) -----
    def run_auto(self):
        if self.overview is None:
            self.log_msg("⚠️ Select an image first."); return
        if not os.path.isfile(self.checkpoint):
            QMessageBox.information(
                self, "Missing checkpoint",
                f"SAM 2.1 checkpoint not found:\n{self.checkpoint}\n\nDownload "
                "sam2.1_hiera_base_plus.pt or select one.")
            self.select_checkpoint()
            if not os.path.isfile(self.checkpoint):
                return
        if self.proc is not None and self.proc.state() != QProcess.NotRunning:
            self.log_msg("Detection already running — press Stop first."); return

        common = ["-m", "section_identification.detect_worker",
                  "--image", self.image_path, "--checkpoint", self.checkpoint,
                  "--target-long-side", str(self.sp_target.value()),
                  "--points-per-side", str(self.sp_pps.value()),
                  "--points-per-batch", str(self.sp_ppb.value()),
                  "--pred-iou-thresh", str(self.sp_iou.value())]
        self._stream_mode = self.chk_tiled.isChecked()
        if self._stream_mode:
            cal = self.calibration or {}
            tile_px = int(cal.get("tile_px", 512))
            min_area = float(cal.get("min_area", self.sp_minarea.value() or 200))
            max_area = float(cal.get("max_area", 1e12))
            args = common + ["--mode", "tiled", "--tile-px", str(tile_px),
                             "--min-area", str(min_area), "--max-area", str(max_area)]
            self._reset_stream_layers()
            self._raw_sections = []
            self._det_params = None
        else:
            self._det_params = dict(
                points_per_side=self.sp_pps.value(),
                points_per_batch=self.sp_ppb.value(),
                pred_iou_thresh=self.sp_iou.value(),
                crop_n_layers=self.sp_crop.value(),
                min_mask_region_area=self.sp_minarea.value())
            args = common + ["--mode", "whole",
                             "--crop-n-layers", str(self._det_params["crop_n_layers"]),
                             "--min-area", str(self._det_params["min_mask_region_area"])]
        self._proc_buf = ""

        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_proc_output)
        self.proc.finished.connect(self._on_proc_finished)
        self.proc.errorOccurred.connect(
            lambda e: self.log_msg(f"❌ detector process error: {e}"))
        self.btn_auto.setEnabled(False)
        self.btn_stop.setVisible(True)
        self.progress.setRange(0, 0); self.progress.setVisible(True)
        self._det_t0 = time.time(); self._elapsed_timer.start(1000)
        self.log_msg("▶ Detecting in a background process — the GUI stays "
                     "responsive; press Stop to cancel.")
        self.proc.start(sys.executable, args)

    def _tick_elapsed(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.lbl_elapsed.setText(f"⏱ detector running… {int(time.time() - self._det_t0)} s")

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
            if line.startswith("STIM_TILES ") or line.startswith("STIM_TILE "):
                try:
                    self._handle_stim_line(line)
                except Exception:
                    pass  # never let a stream-parse error kill the GUI
            else:
                self.log_msg(line)

    def _on_proc_finished(self, code, status):
        self._elapsed_timer.stop(); self.lbl_elapsed.setText("")
        self.btn_auto.setEnabled(True); self.btn_stop.setVisible(False)
        self.progress.setVisible(False)
        self.proc = None
        if code != 0:
            self.log_msg(f"⏹ detection stopped/failed (exit {code}). "
                         "Partial results (if any) are kept.")
            if self._stream_mode:
                self._finalize_tiled()
            return
        try:
            if self._stream_mode:
                self._finalize_tiled()
            else:
                masks = automatic_identification(
                    self.image_path, checkpoint=self.checkpoint, image=self.overview,
                    apply_filtering=self.chk_filter.isChecked(),
                    target_long_side=self.sp_target.value(), **self._det_params)
                self.masks = masks
                polys_xy = [mask_to_polygon(m["segmentation"]) for m in masks]
                polys_xy = [p for p in polys_xy if p is not None and len(p) >= 3]
                self._ensure_edit_layers(polys_xy)
                self.log_msg(f"✔️ {len(polys_xy)} sections → 'Sections' layer.")
                self.rebuild_filmstrip()
                self.save_project()
        except Exception:
            self.log_msg("❌ loading detection results failed:\n" + traceback.format_exc())

    # ----- tiled streaming: live tile/section display -----
    def _set_shapes(self, attr, name, data, edge="white", face=(0, 0, 0, 0), width=2):
        lyr = getattr(self, attr, None)
        if lyr is not None and lyr in self.viewer.layers:
            self.viewer.layers.remove(lyr)
        lyr = self.viewer.add_shapes(data, shape_type="polygon", name=name,
                                     face_color=list(face), edge_width=width)
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
        if line.startswith("STIM_TILES "):
            boxes = json.loads(line[len("STIM_TILES "):])
            self._set_shapes("tiles_layer", "Tiles", [self._box_rect(b) for b in boxes],
                             edge="yellow", width=1)
            self.log_msg(f"Tiling into {len(boxes)} tiles…")
            return
        # STIM_TILE
        d = json.loads(line[len("STIM_TILE "):])
        k, n, box = d["k"], d["n"], d["box"]
        # highlight current tile
        self._set_shapes("current_tile_layer", "Current tile", [self._box_rect(box)],
                         edge="cyan", width=3)
        # append new sections to the live Raw layer
        new = [xy_to_napari(np.asarray(s["poly"], dtype=float)) for s in d["sections"]
               if len(s["poly"]) >= 3]
        for s in d["sections"]:
            if len(s["poly"]) >= 3:
                self._raw_sections.append(s)
        if new:
            if self.raw_layer is None or self.raw_layer not in self.viewer.layers:
                self._set_shapes("raw_layer", "Raw detections", new,
                                 edge="orange", face=(1, 0.5, 0, 0.15), width=2)
            else:
                try:
                    self.raw_layer.add(new, shape_type="polygon")
                except Exception:
                    self.raw_layer.data = list(self.raw_layer.data) + new
        # progress + ETA
        elapsed = max(1e-3, time.time() - self._det_t0)
        eta = elapsed / k * (n - k)
        self.lbl_elapsed.setText(
            f"⏱ tile {k}/{n} · {len(self._raw_sections)} raw sections · "
            f"{int(elapsed)}s elapsed · ~{int(eta)}s left")

    def _finalize_tiled(self):
        """Apply the area filter to streamed raw sections and populate 'Sections'."""
        raw = self._raw_sections
        kept = raw
        if self.chk_filter.isChecked() and len(raw) >= 3:
            try:
                from section_identification.filtering import filtering
                masks_like = [{"area": float(s["area"])} for s in raw]
                lo = max(50.0, min(m["area"] for m in masks_like))
                hi = max(m["area"] for m in masks_like) + 1.0
                chosen, _params = filtering(masks_like, np.linspace(lo, hi, 12), range(2, 5))
                chosen_ids = {id(m) for m in chosen}
                kept = [s for s, m in zip(raw, masks_like) if id(m) in chosen_ids]
            except Exception:
                kept = raw
        polys_xy = [np.asarray(s["poly"], dtype=float) for s in kept]
        self._ensure_edit_layers(polys_xy)
        self.log_msg(f"✔️ {len(raw)} raw → {len(polys_xy)} kept sections "
                     f"(orange = raw, red = kept). Edit in 'Sections'.")
        self.rebuild_filmstrip()
        self.save_project()

    def stop_detection(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.log_msg("■ Stopping detection…")
            self.proc.kill()

    # ----- calibration from drawn examples + tiling preview -----
    def _ensure_calib_layer(self):
        if self.calib_layer is None or self.calib_layer not in self.viewer.layers:
            self.calib_layer = self.viewer.add_shapes(
                [], shape_type="polygon", name="Calibration examples",
                face_color=[0, 1, 0, 0.2], edge_width=3)
            try:
                self.calib_layer.edge_color = "lime"
            except Exception:
                pass
        return self.calib_layer

    def calibrate_from_examples(self):
        if self.overview is None:
            self.log_msg("⚠️ Load an image first."); return
        lyr = self._ensure_calib_layer()
        polys = [napari_to_xy(d) for d in lyr.data if len(np.asarray(d)) >= 3]
        if not polys:
            self.log_msg("Draw 2-5 example sections in the 'Calibration examples' "
                         "layer (polygon tool), then click Calibrate.")
            try:
                self.viewer.layers.selection.active = lyr
            except Exception:
                pass
            return
        from section_identification.calibration import calibrate, summary
        try:
            self.calibration = calibrate(polys, geom=self.geom)
        except Exception:
            self.log_msg("❌ calibration failed:\n" + traceback.format_exc()); return
        self.sp_minarea.setValue(int(self.calibration["min_area"]))
        self.lbl_calib.setText(summary(self.calibration))
        self.log_msg("✔️ " + summary(self.calibration))
        self.chk_tiled.setChecked(True)
        self.preview_tiling()

    def preview_tiling(self):
        """Overlay the actual tile grid + a sample SAM point grid (no detection)."""
        if self.overview is None:
            self.log_msg("⚠️ Load an image first."); return
        from section_identification.tiled_detect import plan_tiles
        H, W = self.overview.shape[:2]
        tile_px = int(self.calibration["tile_px"]) if self.calibration else 512
        boxes = plan_tiles(W, H, tile_px)
        self._set_shapes("tiles_layer", "Tiles (preview)",
                         [self._box_rect(b) for b in boxes], edge="yellow", width=1)
        # sample point grid for the first tile only (representative density)
        pps = self.sp_pps.value()
        if boxes:
            x, y, w, h = boxes[0]
            xs = np.linspace(x, x + w, pps + 2)[1:-1]
            ys = np.linspace(y, y + h, pps + 2)[1:-1]
            pts = np.array([[yy, xx] for yy in ys for xx in xs], dtype=float)
            lyr = getattr(self, "_preview_pts", None)
            if lyr is not None and lyr in self.viewer.layers:
                self.viewer.layers.remove(lyr)
            self._preview_pts = self.viewer.add_points(
                pts, name="SAM point grid (1 tile)", size=max(2, tile_px // 80))
        eff = 1024.0 / tile_px
        msg = (f"Preview: {len(boxes)} tiles of {tile_px}px, points_per_side={pps} "
               f"→ {len(boxes) * pps * pps} prompt points; SAM upscale ×{eff:.1f}")
        if self.calibration:
            secpx = self.calibration["section_px"]
            msg += f"; section ~{secpx:.0f}px → ~{secpx * eff:.0f}px to SAM"
        self.log_msg(msg)

    # ----- polygons from the Shapes layer -----
    def current_polygons_xy(self):
        if self.shapes_layer is None:
            return []
        return [napari_to_xy(d) for d in self.shapes_layer.data
                if len(np.asarray(d)) >= 3]

    def current_fiducials_xy(self):
        if self.fid_layer is None or len(self.fid_layer.data) == 0:
            return []
        return [tuple(map(float, napari_to_xy(p).ravel()))
                for p in self.fid_layer.data]

    # ----- filmstrip / ordering -----
    def rebuild_filmstrip(self, order=None):
        self.filmstrip.clear()
        polys = self.current_polygons_xy()
        if not polys:
            return
        idx_order = order if order is not None else list(range(len(polys)))
        for rank, i in enumerate(idx_order, start=1):
            p = np.asarray(polys[i]).reshape(-1, 2)
            x0, y0 = p[:, 0].min(), p[:, 1].min()
            x1, y1 = p[:, 0].max(), p[:, 1].max()
            crop = self.overview[int(y0):int(y1) + 1, int(x0):int(x1) + 1]
            item = QListWidgetItem(f"{rank}")
            if crop.size:
                item.setIcon(numpy_to_qicon(crop))
            item.setData(Qt.UserRole, i)  # original polygon index
            self.filmstrip.addItem(item)

    def auto_order(self):
        polys = self.current_polygons_xy()
        if len(polys) < 2:
            self.log_msg("Need ≥2 sections to order."); return
        bboxes = ordering.polygons_to_bboxes(polys)
        order, _ = ordering.order_sections(self.overview, bboxes, method="spectral")
        self.log_msg(f"Cross-correlation order: {list(order)}")
        self.rebuild_filmstrip(order=list(order))

    def _export_order(self, n):
        """Return polygon-index order from the filmstrip (identity if mismatched)."""
        if self.filmstrip.count() == n:
            order = [self.filmstrip.item(k).data(Qt.UserRole)
                     for k in range(self.filmstrip.count())]
            if sorted(order) == list(range(n)):
                return order
        return list(range(n))

    # ----- export -----
    def export_coordinates(self):
        if self.image_path is None:
            self.log_msg("⚠️ Nothing to export."); return
        polys = self.current_polygons_xy()
        if not polys:
            self.log_msg("⚠️ No section polygons to export."); return
        order = self._export_order(len(polys))
        polys_ordered = [polys[i] for i in order]
        section_ids = [f"section_{k}" for k in range(1, len(polys_ordered) + 1)]
        fids = self.current_fiducials_xy()
        self.log_msg(f"▶ Exporting {len(polys_ordered)} sections, "
                     f"{len(fids)} fiducials…")
        try:
            outputs = export_polygons(
                self.image_path, polys_ordered, fids, geom=self.geom,
                section_ids=section_ids)
            self.log_msg("✔️ Exported: " + ", ".join(
                f"{k}={v}" for k, v in outputs.items()))
        except Exception:
            self.log_msg("❌ export error:\n" + traceback.format_exc())

    # ----- manual detector (original SAM-assisted ONNX editor) -----
    def _image_file_for_interactive(self):
        """Return an image-file path for the interactive editor.

        The interactive editor reads an image file with ``cv2.imread``; for a CZI
        we write the current overview to a PNG (the editor then works in overview
        pixel space, which export maps back to full resolution via ``geom``).
        """
        if not czi_io.is_czi(self.image_path):
            return self.image_path
        import cv2
        base = os.path.splitext(self.image_path)[0]
        out_dir = f"{base}_files"
        try:
            os.makedirs(out_dir, exist_ok=True)
            png = os.path.join(out_dir, os.path.basename(base) + "_overview.png")
            cv2.imwrite(png, cv2.cvtColor(self.overview, cv2.COLOR_RGB2BGR))
            return png
        except Exception:
            self.log_msg("❌ could not write overview PNG for the manual editor:\n"
                         + traceback.format_exc())
            return None

    def run_manual(self):
        if self.overview is None:
            self.log_msg("⚠️ Load an image first."); return
        if not os.path.isfile(self.sam1_checkpoint):
            QMessageBox.information(
                self, "Missing SAM 1 checkpoint",
                f"The manual editor uses a SAM 1 checkpoint:\n{self.sam1_checkpoint}"
                "\n\nDownload sam_vit_b_01ec64.pth (or sam_vit_h_4b8939.pth) or "
                "select one now.")
            path, _ = QFileDialog.getOpenFileName(
                self, "Select SAM 1 checkpoint", "", "Checkpoints (*.pth *.pt)")
            if not path:
                return
            self.sam1_checkpoint = path
            if "vit_h" in os.path.basename(path):
                self.sam1_model_type = "vit_h"
            elif "vit_l" in os.path.basename(path):
                self.sam1_model_type = "vit_l"
            else:
                self.sam1_model_type = "vit_b"

        try:
            from section_identification.interactive import run_sam_interactive
            from section_identification.interactive_helpers import display_help
        except Exception:
            QMessageBox.warning(
                self, "Manual editor unavailable",
                "The manual editor needs 'onnxruntime' and 'segment-anything' "
                "installed in this environment:\n\n"
                "  pip install onnxruntime segment-anything")
            return

        img_path = self._image_file_for_interactive()
        if img_path is None:
            return

        # The OpenCV editor overlays stored masks as full-frame binary arrays.
        # Masks are stored as RLE now, so decode them — but only if there aren't
        # too many (decoding hundreds of full-res masks would blow up memory; for
        # dense wafers use the napari zoom corrector instead).
        from section_identification.export import decode_segmentation
        stored = []
        OVERLAY_CAP = 60
        if self.masks and len(self.masks) <= OVERLAY_CAP:
            for m in self.masks:
                mm = dict(m)
                mm["segmentation"] = decode_segmentation(m["segmentation"])
                stored.append(mm)
        elif self.masks:
            self.log_msg(f"({len(self.masks)} sections is too many to overlay in the "
                         "OpenCV editor — launching without overlay; new clicks are "
                         "appended to the existing sections.)")

        self.progress.setRange(0, 0); self.progress.setVisible(True)
        QApplication.processEvents()
        try:
            display_help()
            self.log_msg("▶ Launching manual interactive editor (separate window; "
                         "the GUI is busy until you press Esc)…")
            new_masks, stored_masks, fiducials = run_sam_interactive(
                img_path, checkpoint=self.sam1_checkpoint, stored_masks=stored,
                model_type=self.sam1_model_type, device=device_str())
            self.log_msg(f"✔️ Manual session: {len(new_masks)} new, "
                         f"{len(stored_masks)} stored, {len(fiducials)} fiducials.")
            new_polys = [mask_to_polygon(m["segmentation"])
                         for m in list(stored_masks) + list(new_masks)]
            new_polys = [p for p in new_polys if p is not None and len(p) >= 3]
            if stored:
                # Full set was overlaid/edited -> it is authoritative.
                polys_xy = new_polys
            else:
                # Dense wafer: keep existing sections and append the new clicks.
                polys_xy = self.current_polygons_xy() + new_polys
            self._ensure_edit_layers(polys_xy)
            if fiducials and self.fid_layer is not None:
                self.fid_layer.data = np.asarray(fiducials, dtype=float)[:, ::-1]
            self.rebuild_filmstrip()
            self.save_project()
        except Exception:
            self.log_msg("❌ manual editor error:\n" + traceback.format_exc())
        finally:
            self.progress.setVisible(False)


def main():
    viewer = napari.Viewer()
    gui = SectionIdentificationGUI(viewer)
    viewer.window.add_dock_widget(gui, name="STiM", area="right")
    napari.run()


if __name__ == "__main__":
    main()
