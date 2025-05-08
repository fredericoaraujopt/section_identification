# interface.py
import os
from pathlib import Path
import sys
import pickle
import traceback

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QCheckBox,
    QFileDialog, QTextEdit, QInputDialog, QProgressBar,
    QMessageBox, QApplication
)
from qtpy.QtCore import Qt

import napari
import numpy as np
from PIL import Image

from section_identification.section_detector import automatic_identification
from section_identification.interactive import run_sam_interactive
from section_identification.export import export_mask_coordinates
from section_identification.interactive_helpers import display_help

class SectionIdentificationGUI(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.image_path = None
        self.image_layer = None
        self.mask_layer = None
        self.box_layer = None
        self.id_points_layer = None
        self.latest_mode = None
        self.latest_masks = None
        self.latest_new_masks = None
        self.latest_stored_masks = None
        self.latest_fiducials = None
        self.box_layer = None
        self.id_points_layer = None

        # Layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # --- File selection ---
        self.btn_select = QPushButton("Select Image…")
        self.lbl_path = QLabel("No image selected")
        self.lbl_path.setWordWrap(True)
        layout.addWidget(self.btn_select)
        layout.addWidget(self.lbl_path)

        # Cache info
        self.lbl_cache = QLabel("")
        layout.addWidget(self.lbl_cache)

        # --- Automatic detector ---
        layout.addWidget(QLabel("<b>Automatic Detector</b>"))
        self.chk_compress = QCheckBox("Compress image before detection (currently not working)")
        self.chk_filter   = QCheckBox("Filter masks after detection")
        self.btn_auto     = QPushButton("Launch Automatic Detector")
        layout.addWidget(self.chk_compress)
        layout.addWidget(self.chk_filter)
        layout.addWidget(self.btn_auto)

        # --- Manual detector ---
        layout.addWidget(QLabel("<b>Manual Detector</b>"))
        self.btn_manual = QPushButton("Launch Manual Detector")
        layout.addWidget(self.btn_manual)

        # Export button
        self.btn_export = QPushButton("Export Coordinates")
        layout.addWidget(self.btn_export)

        # --- Log window ---
        layout.addWidget(QLabel("<b>Log</b>"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(200)
        layout.addWidget(self.log)

        # Authorship statement
        self.lbl_authorship = QLabel(
            "Developed by Frederico Araujo. Reach out to fredericoaraujo@college.harvard.edu"
        )
        self.lbl_authorship.setAlignment(Qt.AlignCenter)
        self.lbl_authorship.setStyleSheet("font-size: 12px; color: gray;")
        layout.addWidget(self.lbl_authorship)

        # Progress bar for long-running operations
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Default SAM checkpoint path (two levels up from package, in 'checkpoint' folder)
        package_path = Path(os.path.abspath(__file__))
        default_ckpt = package_path.parents[2] / "checkpoint" / "sam_vit_h_4b8939.pth"
        self.checkpoint = str(default_ckpt)
        self.lbl_ckpt = QLabel(f"Checkpoint: {self.checkpoint}")
        self.lbl_ckpt.setWordWrap(True)
        layout.addWidget(self.lbl_ckpt)

        # Button to select custom checkpoint
        self.btn_ckpt = QPushButton("Select SAM Checkpoint")
        layout.addWidget(self.btn_ckpt)
        self.btn_ckpt.clicked.connect(self.select_checkpoint)

        # Redirect stdout/stderr to log
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

        # Signal connections
        self.btn_select.clicked.connect(self.select_image)
        self.btn_auto.clicked.connect(self.run_auto)
        self.btn_manual.clicked.connect(self.run_manual)
        self.btn_export.clicked.connect(self.export_coordinates)
    def select_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SAM checkpoint (.pth file)", "", "PyTorch Checkpoint (*.pth)"
        )
        if path:
            self.checkpoint = path
            self.lbl_ckpt.setText(f"Checkpoint: {self.checkpoint}")

    def append_log(self, text):
        """Append a line to the log window and also write to the original stdout."""
        self.log.append(text)
        QApplication.processEvents()
        self._old_stdout.write(text + "\n")
        self._old_stdout.flush()

    def write(self, text):
        # Forward text to original stdout, and append to log if non-empty.
        self._old_stdout.write(text)
        self._old_stdout.flush()
        if text.strip():
            # Avoid duplicating newlines
            self.log.append(text.rstrip())
            QApplication.processEvents()
    def flush(self):
        pass

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select an image file", "", "Images (*.png *.jpg *.tif *.bmp)"
        )
        if not path:
            return

        self.image_path = path
        self.lbl_path.setText(f"Selected: {path}")

        # Show in napari
        img = np.array(Image.open(path).convert("RGB"))
        if self.image_layer and self.image_layer in self.viewer.layers:
            self.viewer.layers.remove(self.image_layer)
        self.image_layer = self.viewer.add_image(img, name="Base Image")

        if self.mask_layer and self.mask_layer in self.viewer.layers:
            self.viewer.layers.remove(self.mask_layer)
        self.mask_layer = None

        # Check cache
        folder = f"{os.path.splitext(path)[0]}_files"
        if os.path.isdir(folder):
            contents = os.listdir(folder)
            self.lbl_cache.setText(f"Cache folder found: {os.path.basename(folder)} contains {len(contents)} files")
        else:
            self.lbl_cache.setText("No cache folder found")

        # Reset latest trackers
        self.latest_mode = None
        self.latest_masks = None
        self.latest_new_masks = None
        self.latest_stored_masks = None
        self.latest_fiducials = None

    def run_auto(self):
        if not self.image_path:
            self.append_log("⚠️ Please select an image first.")
            return
        # Verify checkpoint exists
        if not os.path.isfile(self.checkpoint):
            QMessageBox.information(
                self,
                "Missing checkpoint",
                f"SAM checkpoint not found at:\n{self.checkpoint}\n\nPlease download from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth or select via the button."
            )
            self.select_checkpoint()
            if not os.path.isfile(self.checkpoint):
                return

        # Show busy indicator
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)
        QApplication.processEvents()

        self.append_log("▶ Running automatic_identification…")
        compress = self.chk_compress.isChecked()
        apply_filtering = self.chk_filter.isChecked()

        # Check for first-run warning
        dir_name = os.path.dirname(self.image_path)
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        folder = os.path.join(dir_name, f"{base_name}_files")
        if not os.path.isdir(folder):
            QMessageBox.information(
                self,
                "First run",
                "No intermediate data file found; first run may take several minutes."
            )

        try:
            masks = automatic_identification(
                self.image_path,
                checkpoint=self.checkpoint,
                compress=compress,
                apply_filtering=apply_filtering
            )
            self.handle_auto_result(masks)
        except Exception as e:
            self.handle_auto_error(traceback.format_exc())
        finally:
            # Hide busy indicator
            self.progress_bar.setVisible(False)
            self.btn_auto.setEnabled(True)

    def handle_auto_result(self, masks):
        """Handle results from the automatic detector thread."""
        self.append_log(f"✔️ automatic_identification returned {len(masks)} masks")
        img = np.array(Image.open(self.image_path).convert("RGB"))
        label_img = np.zeros(img.shape[:2], dtype=int)
        for i, mask in enumerate(masks, start=1):
            label_img[mask['segmentation'] > 0] = i

        # Remove previous mask layer if present
        if self.mask_layer and self.mask_layer in self.viewer.layers:
            self.viewer.layers.remove(self.mask_layer)
        self.mask_layer = self.viewer.add_labels(label_img, name="Auto Masks")

        # Remove previous box and ID layers
        if self.box_layer and self.box_layer in self.viewer.layers:
            self.viewer.layers.remove(self.box_layer)
        if self.id_points_layer and self.id_points_layer in self.viewer.layers:
            self.viewer.layers.remove(self.id_points_layer)

        # Compute bounding boxes and centroids for each mask
        bboxes = []
        centroids = []
        ids = []
        for i, mask in enumerate(masks, start=1):
            seg = mask['segmentation']
            binary = (seg > 0)
            coords = np.column_stack(np.where(binary))
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            bboxes.append([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])
            centroids.append([(y0 + y1) / 2, (x0 + x1) / 2])
            ids.append(i)

        # Add bounding box layer (transparent fill)
        self.box_layer = self.viewer.add_shapes(
            bboxes,
            shape_type='polygon',
            edge_color='red',
            face_color=[0, 0, 0, 0],
            edge_width=6,
            name='Mask Boxes'
        )
        self.box_layer.visible = False

        # Add ID points layer with small text
        self.id_points_layer = self.viewer.add_points(
            np.array(centroids),
            properties={'id': ids},
            text='id',
            name='Mask IDs',
            size=5
        )
        self.id_points_layer.text_size = 5
        self.id_points_layer.visible = False

        # Store latest state
        self.latest_mode = 'auto'
        self.latest_masks = masks

    def handle_auto_error(self, error_str):
        """Log an error from the automatic detector thread."""
        self.append_log("❌ Error in automatic_identification:")
        self.append_log(error_str)


    def run_manual(self):
        if not self.image_path:
            self.append_log("⚠️ Please select an image first.")
            return
        # Verify checkpoint exists
        if not os.path.isfile(self.checkpoint):
            QMessageBox.information(
                self,
                "Missing checkpoint",
                f"SAM checkpoint not found at:\n{self.checkpoint}\n\nPlease download from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth or select via the button."
            )
            self.select_checkpoint()
            if not os.path.isfile(self.checkpoint):
                return

        # Determine manual cache state file path
        dir_name = os.path.dirname(self.image_path)
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        folder = os.path.join(dir_name, f"{base_name}_files")
        state_fn = os.path.join(folder, f"{base_name}_interactive_state.pkl")
        cache_exists = os.path.isfile(state_fn)
        if not cache_exists:
            QMessageBox.information(
                self,
                "First run",
                "No intermediate data file found. Manual detector interface might take longer to load on first run."
            )

        # Show busy indicator
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)
        QApplication.processEvents()

        display_help()

        self.append_log("▶ Launching manual interface…")
        try:
            import cv2
            new_masks, stored_masks, fiducials = run_sam_interactive(
                self.image_path,
                checkpoint=self.checkpoint,
                stored_masks=self.latest_masks if self.latest_mode == 'auto' else [],
                model_type="vit_h",
                device="cpu"
            )

            self.append_log(f"✔️ Manual session done. {len(new_masks)} new masks, {len(stored_masks)} stored masks, {len(fiducials)} fiducials.")

            img = np.array(Image.open(self.image_path).convert("RGB"))
            label_img = np.zeros(img.shape[:2], dtype=int)
            # stored_masks are initial, new_masks added separately
            for i, mask in enumerate(stored_masks, start=1):
                label_img[(mask['segmentation']>0)] = i
            offset = len(stored_masks)
            for j, mask in enumerate(new_masks, start=1):
                label_img[(mask['segmentation']>0)] = offset + j

            if self.mask_layer:
                self.viewer.layers.remove(self.mask_layer)
            self.mask_layer = self.viewer.add_labels(label_img, name="Manual Masks")

            # Add box_layer and id_points_layer with visibility False
            if self.box_layer and self.box_layer in self.viewer.layers:
                self.viewer.layers.remove(self.box_layer)
            self.box_layer = self.viewer.add_shapes([], name="Boxes")
            self.box_layer.visible = False

            if self.id_points_layer and self.id_points_layer in self.viewer.layers:
                self.viewer.layers.remove(self.id_points_layer)
            self.id_points_layer = self.viewer.add_points([], name="ID Points")
            self.id_points_layer.visible = False

            self.latest_mode = 'manual'
            self.latest_new_masks = new_masks
            self.latest_stored_masks = stored_masks
            self.latest_fiducials = fiducials

            # Remove old box/points layers
            if self.box_layer and self.box_layer in self.viewer.layers:
                self.viewer.layers.remove(self.box_layer)
            if self.id_points_layer and self.id_points_layer in self.viewer.layers:
                self.viewer.layers.remove(self.id_points_layer)

            # Combine masks for bounding boxes
            combined = stored_masks + new_masks
            bboxes = []
            centroids = []
            ids = []
            for i, mask in enumerate(combined, start=1):
                seg = mask['segmentation']
                binary = (seg > 0)
                coords = np.column_stack(np.where(binary))
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0)
                bboxes.append([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])
                centroids.append([(y0 + y1)/2, (x0 + x1)/2])
                ids.append(i)

            self.box_layer = self.viewer.add_shapes(
                bboxes,
                shape_type='polygon',
                edge_color='red',
                face_color=[0, 0, 0, 0],    # transparent fill
                edge_width=6,
                name='Mask Boxes'
            )
            self.id_points_layer = self.viewer.add_points(
                np.array(centroids),
                properties={'id': ids},
                text='id',
                name='Mask IDs',
                size=5
            )
            self.id_points_layer.text_size = 5

        except Exception:
            self.append_log("❌ Error in manual interface:")
            self.append_log(traceback.format_exc())
        finally:
            # Hide busy indicator
            self.progress_bar.setVisible(False)

    def export_coordinates(self):
        if not self.image_path or self.latest_mode is None:
            self.append_log("⚠️ No masks to export. Run detection first.")
            return
        self.append_log("▶ Exporting coordinates…")
        if self.latest_mode == 'auto':
            masks = self.latest_masks
            export_mask_coordinates(self.image_path, [], masks, [])
        else:
            export_mask_coordinates(self.image_path, self.latest_new_masks, self.latest_stored_masks, self.latest_fiducials)
        self.append_log("✔️ Coordinates exported.")

def main():
    viewer = napari.Viewer()
    gui = SectionIdentificationGUI(viewer)
    viewer.window.add_dock_widget(gui, name="Section Identification", area="right")
    napari.run()

if __name__ == "__main__":
    main()