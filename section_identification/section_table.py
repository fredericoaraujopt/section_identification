"""Per-section table dock — the proofreading/inspection list shared by stages.

A wide bottom-dock table of every section with its QC scores, recovered serial
order, and imaging order. Clicking a row drives the FOV navigator (zoom to that
section, preserving magnification across rows). Re-columns itself per stage and
refreshes from the WaferProject on demand.

Plain QTableWidget (robust, no model wiring) — adequate for a few hundred rows.
"""

from __future__ import annotations

from qtpy.QtWidgets import (QAbstractItemView, QHeaderView, QTableWidget,
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

    def _on_click(self, row, _col):
        try:
            self.nav.go_to_index(row, keep_fov=True)
        except Exception:
            pass

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
