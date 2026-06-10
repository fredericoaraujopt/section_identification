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
from qtpy.QtCore import QTimer, QProcess, QProcessEnvironment
from qtpy.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

import napari

from section_identification.section_detector import automatic_identification
from section_identification.export import export_polygons, mask_to_polygon
from section_identification import czi_io
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

        layout = QVBoxLayout(); layout.setContentsMargins(6, 6, 6, 6)
        self.setLayout(layout)
        layout.addWidget(QLabel(f"<b>Device:</b> {describe_device()}"))

        self.btn_select = QPushButton("Select Image / CZI…")
        self.lbl_path = QLabel("No image selected"); self.lbl_path.setWordWrap(True)
        layout.addWidget(self.btn_select); layout.addWidget(self.lbl_path)

        layout.addWidget(QLabel("<b>SAM 2.1 parameters</b>"))
        form = QFormLayout()
        self.sp_pps = QSpinBox(); self.sp_pps.setRange(4, 128); self.sp_pps.setValue(32)
        self.sp_ppb = QSpinBox(); self.sp_ppb.setRange(8, 256); self.sp_ppb.setValue(64)
        self.sp_iou = QDoubleSpinBox(); self.sp_iou.setRange(0.0, 1.0)
        self.sp_iou.setSingleStep(0.05); self.sp_iou.setValue(0.80)
        self.sp_minarea = QSpinBox(); self.sp_minarea.setRange(0, 1000000); self.sp_minarea.setValue(50)
        self.sp_target = QSpinBox(); self.sp_target.setRange(1024, 16384)
        self.sp_target.setSingleStep(512); self.sp_target.setValue(3072)
        form.addRow("points_per_side", self.sp_pps)
        form.addRow("points_per_batch (mem)", self.sp_ppb)
        form.addRow("pred_iou_thresh", self.sp_iou)
        form.addRow("min_mask_region_area", self.sp_minarea)
        form.addRow("overview long side (px)", self.sp_target)
        layout.addLayout(form)

        self.chk_filter = QCheckBox("Filter for sections (area DBSCAN)")
        self.chk_filter.setChecked(True)
        layout.addWidget(self.chk_filter)
        self.chk_tiled = QCheckBox("Tiled detector (advanced: tiny-section wafers)")
        layout.addWidget(self.chk_tiled)

        cal_row = QHBoxLayout()
        self.btn_calibrate = QPushButton("Calibrate from examples")
        self.btn_preview = QPushButton("Preview tiling")
        cal_row.addWidget(self.btn_calibrate); cal_row.addWidget(self.btn_preview)
        layout.addLayout(cal_row)
        self.lbl_calib = QLabel("Draw 2–5 example sections in 'Calibration examples', "
                                "then Calibrate (sets size/area/tile + point density).")
        self.lbl_calib.setWordWrap(True); layout.addWidget(self.lbl_calib)

        det_row = QHBoxLayout()
        self.btn_auto = QPushButton("Run Automatic Detection")
        self.btn_stop = QPushButton("Stop"); self.btn_stop.setVisible(False)
        det_row.addWidget(self.btn_auto); det_row.addWidget(self.btn_stop)
        layout.addLayout(det_row)
        self.lbl_elapsed = QLabel(""); layout.addWidget(self.lbl_elapsed)

        layout.addWidget(QLabel("<b>Manual Detector</b>"))
        self.btn_manual = QPushButton("Launch Manual Detector (scroll = zoom)")
        layout.addWidget(self.btn_manual)
        layout.addWidget(QLabel(
            "<i>Separate window: scroll to zoom, hover to preview, click to add, "
            "'r' remove, 'm' fiducials, Esc finish.</i>"))

        self.btn_export = QPushButton("Export (CSV + GeoJSON + annotated CZI)")
        layout.addWidget(self.btn_export)

        pkg = Path(os.path.abspath(__file__)); ckpt_dir = pkg.parents[1] / "checkpoint"
        self.checkpoint = str(ckpt_dir / "sam2.1_hiera_base_plus.pt")
        self.sam1_checkpoint = str(ckpt_dir / "sam_vit_b_01ec64.pth")
        self.sam1_model_type = "vit_b"
        self.lbl_ckpt = QLabel(f"Checkpoint: …/{os.path.basename(self.checkpoint)}")
        self.lbl_ckpt.setWordWrap(True); layout.addWidget(self.lbl_ckpt)
        self.btn_ckpt = QPushButton("Select SAM 2.1 Checkpoint (.pt)")
        layout.addWidget(self.btn_ckpt)

        layout.addWidget(QLabel("<b>Log</b>"))
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(180)
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
        self.btn_calibrate.clicked.connect(self.calibrate_from_examples)
        self.btn_preview.clicked.connect(self.preview_tiling)

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
        self.lbl_path.setText(f"Selected: {path}")
        self.log_msg(f"Loading {os.path.basename(path)}…")
        try:
            if czi_io.is_czi(path):
                arr, geom, meta = czi_io.read_czi_overview(
                    path, target_long_side=self.sp_target.value())
                self.geom = geom
                self.overview = czi_io.to_rgb8(arr)
                self.log_msg(f"CZI {meta['size_x']}x{meta['size_y']}, overview "
                             f"{self.overview.shape}, zoom {meta['zoom']:.4g}")
            else:
                from PIL import Image
                self.overview = np.array(Image.open(path).convert("RGB"))
                self.geom = None
        except Exception:
            self.log_msg("❌ load failed:\n" + traceback.format_exc()); return

        self._reset_layers()
        self.image_layer = self.viewer.add_image(self.overview, name="Overview")
        polys_xy, fids_xy = self._restore_session()
        try:
            self._ensure_edit_layers(polys_xy)
            if fids_xy and self.fid_layer is not None:
                self.fid_layer.data = np.asarray(fids_xy, dtype=float)[:, ::-1]
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

    def _ensure_edit_layers(self, polygons_xy):
        if self.shapes_layer is not None and self.shapes_layer in self.viewer.layers:
            self.viewer.layers.remove(self.shapes_layer)
        data = [xy_to_napari(p) for p in polygons_xy] if polygons_xy else []
        self.shapes_layer = self.viewer.add_shapes(
            data, shape_type="polygon", name="Sections",
            face_color=[1, 0, 0, 0.18], edge_width=4)
        try:
            self.shapes_layer.edge_color = "red"
        except Exception:
            pass
        if self.fid_layer is None or self.fid_layer not in self.viewer.layers:
            self.fid_layer = self.viewer.add_points(np.empty((0, 2)),
                                                    name="Fiducials", size=24)
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
            self.calib_layer = self.viewer.add_shapes(
                [], shape_type="polygon", name="Calibration examples",
                face_color=[0, 1, 0, 0.25], edge_width=4)
            try:
                self.calib_layer.edge_color = "lime"
            except Exception:
                pass
        return self.calib_layer

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

    def save_project(self):
        if self.image_path is None:
            return
        try:
            data = {"image": self.image_path,
                    "sections": [self._to_full(p) for p in self.current_polygons_xy()],
                    "fiducials": [self._to_full([f])[0] for f in self.current_fiducials_xy()]}
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

        common = ["-m", "section_identification.detect_worker",
                  "--image", self.image_path, "--checkpoint", self.checkpoint,
                  "--target-long-side", str(self.sp_target.value()),
                  "--points-per-side", str(self.sp_pps.value()),
                  "--points-per-batch", str(self.sp_ppb.value()),
                  "--pred-iou-thresh", str(self.sp_iou.value())]
        self._stream_mode = self.chk_tiled.isChecked()
        if self._stream_mode:
            cal = self.calibration or {}
            tile_px = int(cal.get("tile_px", 768))
            min_area = float(cal.get("min_area", self.sp_minarea.value() or 200))
            max_area = float(cal.get("max_area", 1e12))
            overlap = float(cal.get("overlap", 0.25))
            args = common + ["--mode", "tiled", "--tile-px", str(tile_px),
                             "--overlap", str(overlap),
                             "--min-area", str(min_area), "--max-area", str(max_area)]
            self._reset_stream_layers(); self._raw_sections = []; self._det_params = None
            self.log_msg(f"▶ Tiled detection: tile_px={tile_px}, overlap={overlap}, "
                         f"area {min_area:.0f}-{max_area:.0f}.")
        else:
            self._det_params = dict(
                points_per_side=self.sp_pps.value(), points_per_batch=self.sp_ppb.value(),
                pred_iou_thresh=self.sp_iou.value(), crop_n_layers=0,
                min_mask_region_area=self.sp_minarea.value())
            args = common + ["--mode", "whole", "--crop-n-layers", "0",
                             "--min-area", str(self._det_params["min_mask_region_area"])]
            self.log_msg("▶ Whole-image detection (no tiling).")
        self._proc_buf = ""

        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self.proc.setProcessEnvironment(env)
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
            if line.startswith("STIM_TILES ") or line.startswith("STIM_TILE "):
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
                polys = [mask_to_polygon(m["segmentation"]) for m in masks]
                polys = [p for p in polys if p is not None and len(p) >= 3]
                self._ensure_edit_layers(polys)
                self.log_msg(f"✔️ {len(polys)} sections → 'Sections' layer.")
                self.save_project()
        except Exception:
            self.log_msg("❌ loading results failed:\n" + traceback.format_exc())

    def stop_detection(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.log_msg("■ Stopping…"); self.proc.kill()

    # ----- tiled streaming display -----
    def _set_shapes(self, attr, name, data, edge="white", face=(0, 0, 0, 0), width=3):
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
                             edge="yellow", face=(1, 1, 0, 0.06), width=2)
            self.log_msg(f"Tiling into {len(boxes)} tiles…")
            return
        d = json.loads(line[len("STIM_TILE "):])
        k, n, box = d["k"], d["n"], d["box"]
        self._set_shapes("current_tile_layer", "Current tile", [self._box_rect(box)],
                         edge="cyan", face=(0, 1, 1, 0.12), width=4)
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
        raw = self._raw_sections
        kept = raw
        if self.chk_filter.isChecked() and len(raw) >= 3:
            try:
                from section_identification.filtering import filtering
                ml = [{"area": float(s["area"])} for s in raw]
                lo = max(50.0, min(m["area"] for m in ml)); hi = max(m["area"] for m in ml) + 1
                chosen, _ = filtering(ml, np.linspace(lo, hi, 12), range(2, 5))
                ids = {id(m) for m in chosen}
                kept = [s for s, m in zip(raw, ml) if id(m) in ids]
            except Exception:
                kept = raw
        self._ensure_edit_layers([np.asarray(s["poly"], dtype=float) for s in kept])
        self.log_msg(f"✔️ {len(raw)} raw → {len(kept)} kept (orange=raw, red=kept).")
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
            self.calibration = calibrate(
                polys, geom=self.geom, overview_long_side=max(self.overview.shape[:2]))
        except Exception:
            self.log_msg("❌ calibration failed:\n" + traceback.format_exc()); return
        self.sp_minarea.setValue(int(self.calibration["min_area"]))
        if self.calibration.get("points_per_side"):
            self.sp_pps.setValue(int(self.calibration["points_per_side"]))
        self.lbl_calib.setText(summary(self.calibration))
        self.log_msg("✔️ " + summary(self.calibration))
        # SAM runs on the downsampled overview; recommend a finer one if needed.
        rec = self.calibration.get("recommended_overview_long_side")
        cur = max(self.overview.shape[:2])
        if rec and rec > cur * 1.3:
            self.sp_target.setValue(int(rec))
            self.log_msg(f"⚠️ Sections are only ~{self.calibration['section_px']:.0f}px in "
                         f"this {cur}px overview. Set overview long side to {rec} and "
                         "RELOAD the image for real detail, then re-calibrate.")
        else:
            self.log_msg("(tick 'Tiled detector' to use these tiles, or run whole-image.)")

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
            new_masks, stored_masks, fiducials = run_sam_interactive(
                img_path, checkpoint=self.sam1_checkpoint, stored_masks=[],
                model_type=self.sam1_model_type, device=device_str())
            self.log_msg(f"✔️ Manual: {len(new_masks)} new, {len(fiducials)} fiducials.")
            new_polys = [mask_to_polygon(m["segmentation"])
                         for m in list(stored_masks) + list(new_masks)]
            new_polys = [p for p in new_polys if p is not None and len(p) >= 3]
            self._ensure_edit_layers(self.current_polygons_xy() + new_polys)
            if fiducials and self.fid_layer is not None:
                self.fid_layer.data = np.asarray(fiducials, dtype=float)[:, ::-1]
            self.save_project()
        except Exception:
            self.log_msg("❌ manual editor error:\n" + traceback.format_exc())
        finally:
            self.progress.setVisible(False)

    # ----- export -----
    def export_coordinates(self):
        if self.image_path is None:
            self.log_msg("⚠️ Nothing to export."); return
        polys = self.current_polygons_xy()
        if not polys:
            self.log_msg("⚠️ No section polygons to export."); return
        section_ids = [f"section_{k}" for k in range(1, len(polys) + 1)]
        fids = self.current_fiducials_xy()
        self.log_msg(f"▶ Exporting {len(polys)} sections, {len(fids)} fiducials…")
        try:
            outputs = export_polygons(self.image_path, polys, fids, geom=self.geom,
                                      section_ids=section_ids)
            self.log_msg("✔️ Exported: " + ", ".join(f"{k}={v}" for k, v in outputs.items()))
        except Exception:
            self.log_msg("❌ export error:\n" + traceback.format_exc())


def main():
    viewer = napari.Viewer()
    gui = SectionIdentificationGUI(viewer)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(gui)
    scroll.setMinimumWidth(300)
    viewer.window.add_dock_widget(scroll, name="STiM", area="right")
    napari.run()


if __name__ == "__main__":
    main()
