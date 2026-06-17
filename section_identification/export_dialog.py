"""Global wafer Export dialog — one place to export everything.

Data toggles (sections / fiducials / ROIs / focus points / serial order /
imaging order / QC) × format toggles (JSON manifest / CSV / GeoJSON / mVis
region_names.csv / ZEN .contour / annotated CZI). Data toggles auto-disable when
that data hasn't been produced yet; a live note says which selected data a format
can't carry (e.g. QC isn't written into a CZI). Replaces the per-stage exports.
"""

from __future__ import annotations

import json
import os

from qtpy.QtWidgets import (QCheckBox, QDialog, QGroupBox, QHBoxLayout, QLabel,
                            QPushButton, QVBoxLayout)

from . import czi_export, czi_io, wafer_export

# data each format can actually carry
FORMAT_DATA = {
    "manifest": {"sections", "fiducials", "rois", "focus", "reorder", "imaging", "qc"},
    "csv": {"sections", "qc", "reorder", "imaging"},
    "geojson": {"sections", "fiducials", "rois"},
    "mvis": {"reorder", "imaging"},
    "contour": {"sections", "rois"},
    "czi": {"sections", "fiducials", "rois", "focus"},
}
DATA = [("sections", "Sections"), ("fiducials", "Fiducials"), ("rois", "ROIs"),
        ("focus", "Focus points"), ("reorder", "Serial order"),
        ("imaging", "Imaging order"), ("qc", "QC")]
FORMATS = [("manifest", "Wafer JSON manifest"), ("csv", "Per-section CSV"),
           ("geojson", "GeoJSON"), ("mvis", "mVis region_names.csv"),
           ("contour", "ZEN .contour"), ("czi", "Annotated CZI"),
           ("atlas", "Atlas")]
_NEEDS_STAGE_UM = {"mvis", "contour", "czi"}


def availability(app) -> dict:
    proj = app.project
    return {
        "sections": bool(proj.sections),
        "fiducials": bool(proj.fiducials),
        "rois": any(s.roi and s.roi.polygon for s in proj.sections),
        "focus": any(s.focus_overview for s in proj.sections),
        "reorder": bool(proj.match_graph.order),
        "imaging": any(s.imaging_index is not None for s in proj.sections),
        "qc": any(s.qc for s in proj.sections),
    }


class ExportDialog(QDialog):
    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.setWindowTitle("Export wafer")
        self.resize(440, 380)
        app.sync_sections()
        avail = availability(app)
        self._has_geom = app.geom is not None

        lay = QVBoxLayout(self)
        row = QHBoxLayout()

        # --- data ---
        dbox = QGroupBox("Data")
        dl = QVBoxLayout(dbox)
        self.data_cb = {}
        for key, label in DATA:
            cb = QCheckBox(label)
            on = bool(avail.get(key))
            cb.setChecked(on)
            cb.setEnabled(on)
            if not on:
                cb.setToolTip("Not produced yet in this session.")
            cb.toggled.connect(self._refresh_note)
            dl.addWidget(cb)
            self.data_cb[key] = cb
        row.addWidget(dbox)

        # --- formats ---
        fbox = QGroupBox("Formats")
        fl = QVBoxLayout(fbox)
        self.fmt_cb = {}
        for key, label in FORMATS:
            cb = QCheckBox(label)
            if key == "atlas":
                cb.setEnabled(False)
                cb.setToolTip("Atlas export not implemented — tell us the target "
                              "atlas format and we'll add it.")
            elif key in _NEEDS_STAGE_UM and not self._has_geom:
                cb.setEnabled(False)
                cb.setToolTip("Needs stage-µm coordinates: a CZI source, or a "
                              "PNG/LM image with fiducials calibrated to stage µm.")
            cb.setChecked(key in ("manifest", "csv"))
            cb.toggled.connect(self._refresh_note)
            fl.addWidget(cb)
            self.fmt_cb[key] = cb
        row.addWidget(fbox)
        lay.addLayout(row)

        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("QLabel{background:#2a2a16;padding:6px;border-radius:4px;}")
        lay.addWidget(self.note)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_export = QPushButton("Export")
        self.btn_export.clicked.connect(self._do_export)
        btn_row.addWidget(self.btn_export)
        lay.addLayout(btn_row)
        self._refresh_note()

    def _selected(self, cbs):
        return {k for k, cb in cbs.items() if cb.isEnabled() and cb.isChecked()}

    def _refresh_note(self, *_):
        data = self._selected(self.data_cb)
        fmts = self._selected(self.fmt_cb)
        msgs = []
        for f in fmts:
            dropped = data - FORMAT_DATA.get(f, set())
            if dropped:
                names = ", ".join(sorted(dropped))
                msgs.append(f"• {dict(FORMATS)[f]} won't include: {names}")
        self.note.setText("\n".join(msgs) if msgs
                          else "All selected data is carried by the selected formats.")

    # ---- write ----
    def _do_export(self):
        app = self.app
        if not app.has_image():
            app.log("export", "load an image first.")
            return
        data = self._selected(self.data_cb)
        fmts = self._selected(self.fmt_cb)
        if not fmts:
            app.log("export", "select at least one format.")
            return
        proj = app.sync_sections()
        app.ensure_poses()
        geom = app.geom
        from .stages import build_tile_region_specs

        counts = {}
        tmpl = proj.roi_templates[0] if proj.roi_templates else None
        if tmpl is not None and getattr(tmpl, "tile_um", None) and geom is not None:
            _, counts = build_tile_region_specs(proj, geom, tmpl.tile_um[0],
                                                tmpl.focus_cols, tmpl.focus_rows, 0.0)
        manifest = wafer_export.build_manifest(proj, geom, mfov_counts=counts)
        from . import export as legacy_export
        try:
            out_dir = legacy_export.resolve_export_dir(app.image_path, None)
        except Exception:
            out_dir = os.path.dirname(app.image_path or ".")

        written = []
        if "manifest" in fmts:
            written.append(wafer_export.write_json_manifest(manifest, out_dir))
        if "csv" in fmts:
            written.append(wafer_export.write_csv_table(manifest, out_dir))
        if "geojson" in fmts:
            written.append(self._write_geojson(manifest, out_dir, data))
        if "mvis" in fmts:
            written.append(wafer_export.write_mvis_lmb(manifest, out_dir))
        if "contour" in fmts:
            try:
                p = wafer_export.write_zen_contour(manifest, out_dir)
                if p:
                    written.append(p)
            except Exception as e:
                app.log("export", f"⚠️ .contour failed: {e}")
        if "czi" in fmts:
            p = self._write_czi(proj, geom, data)
            if p:
                written.append(p)

        app.log("export", f"wrote to {out_dir}: "
                          f"{[os.path.basename(p) for p in written if p]}")
        self.accept()

    def _write_geojson(self, manifest, out_dir, data):
        feats = []
        if "sections" in data:
            for s in manifest["sections"]:
                poly = s.get("polygon_full_px") or []
                if len(poly) >= 3:
                    ring = [[float(x), float(y)] for x, y in poly]
                    ring.append(ring[0])
                    feats.append({"type": "Feature",
                                  "geometry": {"type": "Polygon", "coordinates": [ring]},
                                  "properties": {"id": s["id"], "serial": s.get("serial_index"),
                                                 "imaging": s.get("imaging_index")}})
        if "fiducials" in data:
            for i, f in enumerate(manifest.get("fiducials", []), 1):
                if f.get("full_px"):
                    feats.append({"type": "Feature",
                                  "geometry": {"type": "Point", "coordinates": f["full_px"]},
                                  "properties": {"id": f"fiducial_{i}"}})
        path = os.path.join(out_dir, f"{manifest['wafer_id']}_sections.geojson")
        with open(path, "w") as fh:
            json.dump({"type": "FeatureCollection", "features": feats}, fh, indent=2)
        return path

    def _write_czi(self, proj, geom, data):
        """Write the annotated CZI in ZEN's CAT format: sections -> CAT_Section,
        ROIs -> CAT_ROI, focus -> a focus marker layer (all full-res PIXEL
        polygons, the frame ZEN's CAT workflow reads), plus the fiducials as the
        Shuttle & Find correlative calibration (stage µm). ZEN derives the
        acquisition TileRegions from the CAT_ROI polygons via that calibration, so
        we do NOT write stage-µm TileRegions here."""
        if not czi_io.is_czi(self.app.image_path):
            self.app.log("export", "annotated CZI needs a CZI source image.")
            return None
        import numpy as np

        def _to_full(poly_overview):
            a = np.asarray(poly_overview, float).reshape(-1, 2)
            fx, fy = geom.ds_to_full(a[:, 0], a[:, 1])
            return [[float(x), float(y)] for x, y in zip(np.ravel(fx), np.ravel(fy))]

        polys_full = [s.polygon_full(geom) for s in proj.sections] if "sections" in data else []

        rois_full = []
        if "rois" in data and geom is not None:
            for s in proj.sections:
                if s.roi and len(s.roi.polygon) >= 3:
                    rois_full.append(_to_full(s.roi.polygon))

        focus_full = []
        if "focus" in data and geom is not None:
            for s in proj.sections:
                if s.focus_overview:
                    focus_full.extend(_to_full(s.focus_overview))

        fids_full, sf = [], None
        if "fiducials" in data:
            for (fx, fy) in proj.fiducials:
                gx, gy = geom.ds_to_full(np.asarray([fx]), np.asarray([fy]))
                fids_full.append([float(np.ravel(gx)[0]), float(np.ravel(gy)[0])])
            man = wafer_export.build_manifest(proj, geom)
            sf = [f["stage_um"] for f in man.get("fiducials", []) if f.get("stage_um")] or None

        dst = os.path.splitext(self.app.image_path)[0] + "_STiM_acq.czi"
        try:
            report = czi_export.write_annotated_czi(
                self.app.image_path, dst, polys_full, fids_full,
                section_ids=[s.id for s in proj.sections],
                rois=rois_full or None, focus_full=focus_full or None,
                sf_markers_stage_um=sf)
            self.app.log("export", f"CZI: {report}")
            return dst
        except Exception as e:
            self.app.log("export", f"⚠️ CZI write failed: {e}")
            return None
