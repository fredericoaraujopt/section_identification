"""Global wafer Export dialog — one place to export everything.

Data toggles (sections / fiducials / ROIs / focus points / serial order /
imaging order / QC) × format toggles (JSON manifest / CSV / GeoJSON / mVis
region_names.csv / ZEN .contour / annotated CZI). Data toggles auto-disable when
that data hasn't been produced yet; a live note says which selected data a format
can't carry (e.g. QC isn't written into a CZI). Replaces the per-stage exports.
"""

from __future__ import annotations

import os

from qtpy.QtWidgets import (QCheckBox, QDialog, QGroupBox, QHBoxLayout, QLabel,
                            QPushButton, QVBoxLayout)

from . import czi_export, czi_io, wafer_export

# data each format can actually carry. The viewport PNG is a raster snapshot of
# whatever layers are currently visible, so it "carries" every visual data type
# (sections / fiducials / ROIs / focus / order / QC are all drawn on the canvas).
FORMAT_DATA = {
    "manifest": {"sections", "fiducials", "rois", "focus", "reorder", "imaging", "qc"},
    "csv": {"sections", "qc", "reorder", "imaging"},
    "geojson": {"sections", "fiducials", "rois"},
    "mvis": {"reorder", "imaging"},
    "contour": {"sections", "rois"},
    "czi": {"sections", "fiducials", "rois", "focus"},
    "png": {"sections", "fiducials", "rois", "focus", "reorder", "imaging", "qc"},
}
DATA = [("sections", "Sections"), ("fiducials", "Fiducials"), ("rois", "ROIs"),
        ("focus", "Focus points"), ("reorder", "Serial order"),
        ("imaging", "Imaging order"), ("qc", "QC")]
FORMATS = [("manifest", "Wafer JSON manifest"), ("csv", "Per-section CSV"),
           ("geojson", "GeoJSON"), ("mvis", "mVis region_names.csv"),
           ("contour", "ZEN .contour"), ("czi", "Annotated CZI"),
           ("png", "Viewport PNG"), ("atlas", "Atlas")]
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
        if "png" in fmts:
            msgs.append("• Viewport PNG renders the whole wafer with all visible "
                        "overlays at the imported image's resolution (no frame, "
                        "300 DPI) — the data toggles don't apply to it.")
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
        # Fold in any hand-edited ROIs and, crucially, preserve section-less ones:
        # capture promotes each ROI that has no section of its own into a margined
        # synthetic section, so every exported ROI is the sole ROI of some section
        # (what ZEN's CAT pairing expects) rather than being silently dropped.
        app.capture_annotations()
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
            p = self._write_czi(proj, geom, data, out_dir)
            if p:
                written.append(p)
        if "png" in fmts:
            p = self._write_viewport_png(out_dir, manifest["wafer_id"])
            if p:
                written.append(p)

        app.log("export", f"wrote to {out_dir}: "
                          f"{[os.path.basename(p) for p in written if p]}")
        self.accept()

    # cap the rendered long side so a native ~76k-px wafer (~18 GB) doesn't blow
    # up memory / exceed the GPU's max texture size; high enough to stay crisp.
    _PNG_MAX_LONG = 16384

    def _image_long_side(self):
        """Long side, in data pixels, of the imported wafer image layer (level-0
        for a multiscale CZI). Used to pick a memory-safe render scale."""
        import numpy as np
        from napari.layers import Image
        best = 0
        for lyr in self.app.viewer.layers:
            if not isinstance(lyr, Image):
                continue
            try:
                data = lyr.data
                shp = data[0].shape if isinstance(data, (list, tuple)) else data.shape
                spatial = shp[:2] if (getattr(lyr, "rgb", False)
                                      or (len(shp) >= 3 and shp[-1] in (3, 4))) else shp[:2]
                best = max(best, int(max(spatial)))
            except Exception:
                continue
        return best or None

    def _write_viewport_png(self, out_dir, wafer_id):
        """Render a PNG of the whole wafer with every visible overlay (sections,
        ROIs, focus points, order-number masks, colours), cropped tight to the
        image — no canvas frame — at the imported image's pixel resolution and
        tagged 300 DPI. Uses napari's ``export_figure`` (which fits the data
        extent and restores the view afterwards). Very large wafers are rendered
        at a capped long side (logged) to stay within memory / GPU limits."""
        import numpy as np
        viewer = self.app.viewer
        if viewer is None:
            self.app.log("export", "no napari viewer to capture.")
            return None
        path = os.path.join(out_dir, f"{wafer_id}_viewport.png")
        try:
            native = self._image_long_side()
            scale = 1.0
            if native and native > self._PNG_MAX_LONG:
                scale = self._PNG_MAX_LONG / float(native)
            arr = viewer.export_figure(scale_factor=scale, flash=False)
            arr = np.asarray(arr)
            if arr.ndim == 3 and arr.shape[2] == 4:      # drop alpha (figure is opaque)
                arr = arr[..., :3]
            from PIL import Image as PILImage
            PILImage.fromarray(arr).save(path, dpi=(300, 300))
            h, w = arr.shape[:2]
            note = f"{w}x{h}px @300dpi"
            if scale < 1.0:
                note += f" (native {native}px capped to ≤{self._PNG_MAX_LONG}px)"
            self.app.log("export", f"viewport PNG: {note}")
            return path
        except Exception as e:
            self.app.log("export", f"⚠️ viewport PNG failed: {e}")
            return None

    def _write_geojson(self, manifest, out_dir, data):
        # Delegates to wafer_export so section outlines, ROIs, and fiducials are
        # built + tested in one place (the dialog only picks the data toggles).
        return wafer_export.write_geojson(manifest, out_dir, data)

    def _write_czi(self, proj, geom, data, out_dir):
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

        # Focus points -> ZEN autofocus SupportPoints inside per-ROI TileRegions
        # (stage µm), the node ZEN actually reads as focus points. This is what
        # makes a focus point marked in STiM land as a ZEN focus support point.
        # build_tile_region_specs turns each section's focus_overview into the
        # region's support points (falling back to an fc×fr grid if none).
        tile_regions = None
        if "focus" in data and geom is not None:
            from .stages import build_tile_region_specs
            tmpl = proj.roi_templates[0] if proj.roi_templates else None
            tile_um = (tmpl.tile_um[0] if (tmpl and getattr(tmpl, "tile_um", None))
                       else 50.0)
            fc = getattr(tmpl, "focus_cols", 1) or 1
            fr = getattr(tmpl, "focus_rows", 1) or 1
            specs, _ = build_tile_region_specs(proj, geom, tile_um, fc, fr, 0.0)
            tile_regions = specs or None
            n_fp = sum(len(s.get("support_points") or []) for s in (specs or []))
            self.app.log("export", f"focus → {n_fp} support points in "
                                    f"{len(specs or [])} TileRegions (stage µm)")

        fids_full, sf = [], None
        if "fiducials" in data:
            for (fx, fy) in proj.fiducials:
                gx, gy = geom.ds_to_full(np.asarray([fx]), np.asarray([fy]))
                fids_full.append([float(np.ravel(gx)[0]), float(np.ravel(gy)[0])])
            man = wafer_export.build_manifest(proj, geom)
            sf = [f["stage_um"] for f in man.get("fiducials", []) if f.get("stage_um")] or None

        base = os.path.splitext(os.path.basename(self.app.image_path))[0]
        dst = os.path.join(out_dir, f"{base}_STiM_acq.czi")
        try:
            report = czi_export.write_annotated_czi(
                self.app.image_path, dst, polys_full, fids_full,
                section_ids=[s.id for s in proj.sections],
                rois=rois_full or None, focus_full=focus_full or None,
                tile_regions=tile_regions,
                sf_markers_stage_um=sf)
            self.app.log("export", f"CZI: {report}")
            return dst
        except Exception as e:
            self.app.log("export", f"⚠️ CZI write failed: {e}")
            return None
