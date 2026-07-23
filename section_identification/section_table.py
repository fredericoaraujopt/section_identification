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

COLUMNS = ["#", "id", "area", "QC", "debris", "fold", "shred", "chatter",
           "serial", "imaging", "status", "del"]


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
        self.btn_orient = QPushButton("Orientation")
        self.btn_orient.setToolTip("Overlay each section's recovered 'up' axis as an "
                                   "arrow on the wafer — the visual interpretation of "
                                   "unifying section orientations.")
        self.btn_orient.clicked.connect(self._show_orientation)
        bar.addWidget(self.btn_orient)
        self.btn_prev = QPushButton("▲ prev")
        self.btn_prev.setToolTip("Go to the previous section, keeping the current zoom "
                                 "(keyboard: Up arrow while the image is focused).")
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next = QPushButton("▼ next")
        self.btn_next.setToolTip("Go to the next section, keeping the current zoom "
                                 "(keyboard: Down arrow while the image is focused).")
        self.btn_next.clicked.connect(lambda: self._step(1))
        bar.addWidget(self.btn_prev)
        bar.addWidget(self.btn_next)
        bar.addStretch(1)
        self.btn_accept = QPushButton("✓ accept")
        self.btn_accept.setToolTip("Mark the selected section accepted (proofread).")
        self.btn_accept.clicked.connect(lambda: self._set_accept(True))
        self.btn_reject = QPushButton("✗ reject")
        self.btn_reject.setToolTip("Mark the selected section rejected.")
        self.btn_reject.clicked.connect(lambda: self._set_accept(False))
        bar.addWidget(self.btn_accept)
        bar.addWidget(self.btn_reject)
        lay.addLayout(bar)
        self._gallery_dlg = None
        self.selected_section = None
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
        self.table.cellDoubleClicked.connect(self._on_double)
        # Arrow-key row changes (table focused) also drive the viewer.
        self.table.currentCellChanged.connect(
            lambda cur, _c, _pr, _pc: self._activate_row(cur))
        lay.addWidget(self.table)
        self._listeners = []
        self._rows = []          # sections in displayed (area-sorted) order

    def add_select_listener(self, fn):
        """Register ``fn(section)`` called when a row is selected (stages use this
        to show per-section diagnostics)."""
        self._listeners.append(fn)

    def _set_accept(self, ok):
        if self.selected_section is None:
            self.app.log("proofread", "select a section row first.")
            return
        self.selected_section.accepted = bool(ok)
        self.refresh()
        try:
            self.app.save_workflow()
        except Exception:
            pass

    def _row_section(self, row):
        return self._rows[row] if 0 <= row < len(self._rows) else None

    def _activate_row(self, row):
        """Select the row + recenter on its section, keeping the current zoom
        (pan without changing magnification). Shared by click, arrow keys, and the
        prev/next buttons."""
        section = self._row_section(row)
        if section is None:
            return
        self.selected_section = section
        try:
            self.nav.center_on(section)
        except Exception:
            pass
        for fn in self._listeners:
            try:
                fn(section)
            except Exception:
                pass

    def _on_click(self, row, _col):
        """Single click: select + recenter on the section (keeps magnification)."""
        self._activate_row(row)

    def _step(self, delta):
        """Move the selection up/down the table by ``delta`` rows and pan to that
        section at the current zoom — the prev/next buttons and Up/Down arrows.
        Wraps the selection to the ends so it never gets stuck."""
        n = len(self._rows)
        if n == 0:
            return
        cur = self.table.currentRow()
        nxt = 0 if cur < 0 else (cur + delta) % n        # wrap around
        moved = nxt != self.table.currentRow()
        self.table.setCurrentCell(nxt, 0)                # → currentCellChanged → activate
        try:
            self.table.scrollToItem(self.table.item(nxt, 0))
        except Exception:
            pass
        if not moved:                                    # single row: still recenter
            self._activate_row(nxt)

    def _on_double(self, row, _col):
        """Double click: snap to the project-wide consistent magnification."""
        section = self._row_section(row)
        self.selected_section = section
        try:
            self.nav.fit_consistent(section)
        except Exception:
            pass

    def _delete_section(self, section):
        """Remove a section from the wafer (Shapes layer + model) entirely."""
        proj = self.app.project
        try:
            idx = proj.sections.index(section)
        except ValueError:
            return
        layer = getattr(self.app.gui, "shapes_layer", None)
        if layer is not None:
            try:
                data = list(layer.data)
                if 0 <= idx < len(data):
                    del data[idx]
                    layer.data = data
            except Exception:
                pass
        proj.sections.remove(section)
        if self.selected_section is section:
            self.selected_section = None
        self.refresh()
        for save in (getattr(self.app.gui, "save_project", None), self.app.save_workflow):
            try:
                save and save()
            except Exception:
                pass
        self.app.log("proofread", f"deleted {section.id}.")

    def _show_orientation(self):
        if not self.app.has_image():
            self.app.log("proofread", "load an image and detect sections first.")
            return
        try:
            self.app.sync_sections()
            from . import layer_sync
            layer_sync.show_orientation(self.app)
        except Exception as e:
            self.app.log("proofread", f"orientation error: {e}")

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
        # display sorted by area (smallest first) — surfaces coming-in / spurious
        # detections at the top; click maps via the displayed order.
        self._rows = sorted(self.app.project.sections, key=lambda s: s.area())
        self.table.setRowCount(len(self._rows))
        for r, s in enumerate(self._rows):
            sc = s.qc.scores if s.qc else {}
            if not s.accepted:
                status = "rejected"
            elif s.qc and s.qc.flags.get("any"):
                status = "review"
            else:
                status = "ok"
            vals = [r + 1, s.id, self._fmt(s.area(), 0), self._fmt(sc.get("overall")),
                    self._fmt(sc.get("debris")), self._fmt(sc.get("fold")),
                    self._fmt(sc.get("shred")), self._fmt(sc.get("chatter")),
                    self._fmt(s.serial_index, 0), self._fmt(s.imaging_index, 0),
                    status]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(str(v)))
            trash = QPushButton("🗑")
            trash.setToolTip("Delete this section from the wafer.")
            trash.setFixedWidth(34)
            trash.clicked.connect(lambda _=False, sec=s: self._delete_section(sec))
            self.table.setCellWidget(r, len(COLUMNS) - 1, trash)
