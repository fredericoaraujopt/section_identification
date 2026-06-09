"""STiM unified napari GUI.

One window for the whole workflow: load an image (including whole-slide ``.czi``,
read from the pyramid), run SAM 2.1 automatic detection, **edit the section
polygons and fiducials natively in napari** (a Shapes layer + a Points layer —
no more separate OpenCV window), recover serial order by cross-correlation via a
reorderable filmstrip, and export CSV / GeoJSON / a ZEN-annotated CZI.

Coordinate convention: napari layer data is ``(row, col)`` = ``(y, x)``; our
detection/export code uses ``(x, y)``. The helpers below convert at the boundary.
"""

import os
import sys
import traceback
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, QSize
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
from section_identification.device import describe as describe_device


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
        self.predictor = None          # SAM 2.1 image predictor (click-to-add)
        self.sam_click_enabled = False
        self._device = None

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

        self.chk_filter = QCheckBox("Filter for sections (shape + area)")
        self.chk_filter.setChecked(True)
        layout.addWidget(self.chk_filter)
        self.btn_auto = QPushButton("Run Automatic Detection")
        layout.addWidget(self.btn_auto)

        # --- SAM-assisted real-time correction ---
        layout.addWidget(QLabel("<b>Manual correction</b>"))
        self.btn_samclick = QPushButton("SAM 2.1 click-to-add: OFF")
        self.btn_samclick.setCheckable(True)
        layout.addWidget(self.btn_samclick)
        layout.addWidget(QLabel(
            "<i>Fix false negatives: toggle ON, then click a missed section — "
            "SAM 2.1 segments it and adds it to 'Sections'. Fix false positives: "
            "select a polygon in 'Sections' and press Delete. Drop registration "
            "points in the 'Fiducials' layer.</i>"))

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

        # --- Checkpoint ---
        pkg = Path(os.path.abspath(__file__))
        default_ckpt = pkg.parents[1] / "checkpoint" / "sam2.1_hiera_base_plus.pt"
        self.checkpoint = str(default_ckpt)
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
        self.btn_samclick.clicked.connect(self.toggle_sam_click)
        # Viewer-level click handler for SAM click-to-add (fires regardless of
        # which layer is active; gated by self.sam_click_enabled).
        self.viewer.mouse_drag_callbacks.append(self._on_viewer_click)

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
        # New image -> stale predictor; re-encode on next SAM click.
        self.predictor = None
        self.sam_click_enabled = False
        self.btn_samclick.setChecked(False)
        self.btn_samclick.setText("SAM 2.1 click-to-add: OFF")
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

        # Load any existing STiM annotations stored inside the CZI so reopening
        # an annotated CZI shows the saved polygons/fiducials. Everything here is
        # best-effort: a failure must NOT prevent the image from opening.
        polys_xy, fids_xy = [], []
        if self.geom is not None and czi_io.is_czi(path):
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
        self.filmstrip.clear()
        try:
            if polys_xy:
                self.rebuild_filmstrip()
        except Exception:
            self.log_msg("[warn] filmstrip build failed (annotations still loaded).")

    def _reset_layers(self):
        for lyr in list(self.viewer.layers):
            self.viewer.layers.remove(lyr)
        self.image_layer = self.shapes_layer = self.fid_layer = None

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

    # ----- detection -----
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
        self.progress.setRange(0, 0); self.progress.setVisible(True)
        QApplication.processEvents()
        try:
            self.log_msg("▶ Running SAM 2.1 automatic detection…")
            masks = automatic_identification(
                self.image_path, checkpoint=self.checkpoint, image=self.overview,
                apply_filtering=self.chk_filter.isChecked(),
                points_per_side=self.sp_pps.value(),
                points_per_batch=self.sp_ppb.value(),
                pred_iou_thresh=self.sp_iou.value(),
                crop_n_layers=self.sp_crop.value(),
                min_mask_region_area=self.sp_minarea.value(),
                target_long_side=self.sp_target.value())
            self.masks = masks
            polys_xy = [mask_to_polygon(m["segmentation"]) for m in masks]
            polys_xy = [p for p in polys_xy if p is not None and len(p) >= 3]
            self._ensure_edit_layers(polys_xy)
            self.log_msg(f"✔️ {len(polys_xy)} sections → editable 'Sections' layer.")
            self.rebuild_filmstrip()
        except Exception:
            self.log_msg("❌ detection error:\n" + traceback.format_exc())
        finally:
            self.progress.setVisible(False)

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

    # ----- SAM 2.1 click-to-add (real-time correction of false negatives) -----
    def toggle_sam_click(self):
        if self.btn_samclick.isChecked():
            if self.overview is None:
                self.log_msg("⚠️ Load an image first.")
                self.btn_samclick.setChecked(False)
                return
            try:
                self._ensure_predictor()
            except Exception:
                self.log_msg("❌ SAM predictor init failed:\n" + traceback.format_exc())
                self.btn_samclick.setChecked(False)
                return
            self.sam_click_enabled = True
            self.btn_samclick.setText("SAM 2.1 click-to-add: ON — click missed sections")
        else:
            self.sam_click_enabled = False
            self.btn_samclick.setText("SAM 2.1 click-to-add: OFF")

    def _ensure_predictor(self):
        """Build the SAM 2.1 image predictor and encode the current overview once."""
        if self.predictor is not None:
            return
        from section_identification.section_detector import build_image_predictor
        from section_identification.device import get_device, autocast_ctx
        self._device = get_device()
        self.progress.setRange(0, 0); self.progress.setVisible(True)
        QApplication.processEvents()
        self.log_msg("Initialising SAM 2.1 predictor (encoding overview)…")
        try:
            self.predictor = build_image_predictor(self.checkpoint, None, self._device)
            with autocast_ctx(self._device):
                self.predictor.set_image(self.overview)
            self.log_msg("✔️ SAM click-to-add ready.")
        finally:
            self.progress.setVisible(False)

    def _on_viewer_click(self, viewer, event):
        if not self.sam_click_enabled or self.image_layer is None:
            return
        if getattr(event, "button", 1) != 1:  # left button only
            return
        try:
            pos = self.image_layer.world_to_data(event.position)
        except Exception:
            return
        y, x = float(pos[0]), float(pos[1])
        h, w = self.overview.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            self._sam_add_at(x, y)

    def _sam_add_at(self, x, y):
        from section_identification.device import autocast_ctx
        try:
            with autocast_ctx(self._device):
                masks, scores, _ = self.predictor.predict(
                    point_coords=np.array([[x, y]], dtype=float),
                    point_labels=np.array([1], dtype=int),
                    multimask_output=True)
            # Sections are small/local: prefer the highest-scoring mask that
            # isn't a near-whole-image blob; fall back to the smallest mask.
            masks = np.asarray(masks)
            areas = masks.reshape(len(masks), -1).sum(axis=1)
            img_area = float(masks[0].size)
            compact = [i for i in range(len(masks)) if areas[i] < 0.3 * img_area]
            idx = (max(compact, key=lambda k: scores[k]) if compact
                   else int(np.argmin(areas)))
            best = masks[idx]
            poly = mask_to_polygon((best > 0).astype(np.uint8))
            if poly is None or len(poly) < 3:
                self.log_msg("No mask at that point.")
                return
            napari_poly = xy_to_napari(poly)
            try:
                self.shapes_layer.add(napari_poly, shape_type="polygon")
            except Exception:
                self.shapes_layer.data = list(self.shapes_layer.data) + [napari_poly]
            self.log_msg(f"➕ Added section at ({x:.0f},{y:.0f}); "
                         f"{len(self.shapes_layer.data)} total.")
        except Exception:
            self.log_msg("❌ SAM click failed:\n" + traceback.format_exc())


def main():
    viewer = napari.Viewer()
    gui = SectionIdentificationGUI(viewer)
    viewer.window.add_dock_widget(gui, name="STiM", area="right")
    napari.run()


if __name__ == "__main__":
    main()
