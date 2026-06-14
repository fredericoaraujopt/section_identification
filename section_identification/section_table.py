"""Per-section table dock — the proofreading/inspection list shared by stages.

A wide bottom-dock table of every section with its QC scores, recovered serial
order, and imaging order. Clicking a row drives the FOV navigator (zoom to that
section, preserving magnification across rows). Re-columns itself per stage and
refreshes from the WaferProject on demand.

Plain QTableWidget (robust, no model wiring) — adequate for a few hundred rows.
"""

from __future__ import annotations

from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (QAbstractItemView, QDialog, QHeaderView, QHBoxLayout,
                            QLabel, QPushButton, QScrollArea, QTableWidget,
                            QTableWidgetItem, QVBoxLayout, QWidget)

COLUMNS = ["#", "id", "QC", "debris", "fold", "shred", "chatter",
           "serial", "imaging", "status"]


class SectionTableDock(QWidget):
    def __init__(self, app, nav, parent=None):
        super().__init__(parent)
        self.app = app
        self.nav = nav
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        bar = QHBoxLayout()
        self.btn_gallery = QPushButton("Aligned gallery")
        self.btn_gallery.setToolTip("Montage of every section rotated to its "
                                    "canonical pose — scan for mis-detections.")
        self.btn_gallery.clicked.connect(self._show_gallery)
        bar.addWidget(self.btn_gallery)
        bar.addStretch(1)
        lay.addLayout(bar)
        self._gallery_dlg = None
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        try:
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        except Exception:
            pass
        self.table.cellClicked.connect(self._on_click)
        lay.addWidget(self.table)
        self._listeners = []

    def add_select_listener(self, fn):
        """Register ``fn(section)`` called when a row is selected (stages use this
        to show per-section diagnostics)."""
        self._listeners.append(fn)

    def _on_click(self, row, _col):
        secs = self.app.project.sections
        section = secs[row] if 0 <= row < len(secs) else None
        try:
            self.nav.go_to_index(row, keep_fov=True)
        except Exception:
            pass
        for fn in self._listeners:
            try:
                fn(section)
            except Exception:
                pass

    def _show_gallery(self):
        if not self.app.has_image():
            self.app.log("gallery", "load an image and detect sections first.")
            return
        try:
            self.app.sync_sections()
            from . import gallery
            mont, n, total = gallery.build_gallery(self.app)
            if mont.size <= 1:
                self.app.log("gallery", "no sections to show.")
                return
            h, w = mont.shape
            mont = mont.copy()                     # contiguous for QImage
            img = QImage(mont.data, w, h, w, QImage.Format_Grayscale8)
            dlg = QDialog(self)
            dlg.setWindowTitle(f"Aligned section gallery ({n} of {total})")
            dl = QVBoxLayout(dlg)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            label = QLabel()
            label.setPixmap(QPixmap.fromImage(img))
            scroll.setWidget(label)
            dl.addWidget(scroll)
            dlg.resize(min(1000, w + 40), min(800, h + 40))
            self._gallery_dlg = dlg                # keep a reference
            dlg.show()
        except Exception as e:
            self.app.log("gallery", f"gallery error: {e}")

    def _fmt(self, v, nd=2):
        return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    def refresh(self):
        secs = self.app.project.sections
        self.table.setRowCount(len(secs))
        for r, s in enumerate(secs):
            sc = s.qc.scores if s.qc else {}
            status = "reject" if (s.qc and s.qc.flags.get("any")) else "accept"
            vals = [r + 1, s.id, self._fmt(sc.get("overall")),
                    self._fmt(sc.get("debris")), self._fmt(sc.get("fold")),
                    self._fmt(sc.get("shred")), self._fmt(sc.get("chatter")),
                    self._fmt(s.serial_index, 0), self._fmt(s.imaging_index, 0),
                    status]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(str(v)))
