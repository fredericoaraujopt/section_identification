"""StimApp — a thin facade over the existing SectionIdentificationGUI.

The new workflow tabs (ROIs / QC / Reorder) need shared access to the viewer,
geometry, the current sections, logging, and a :class:`WaferProject` to hang
per-stage results on. Rather than rewrite the 1600-line GUI, StimApp reads from
it (the GUI stays the source of truth for the wafer image + section geometry)
and maintains a WaferProject synced from the live "Sections" Shapes layer.

This keeps the existing detect/calibrate/manual/export UX untouched; the new
tabs are additive consumers of this facade.
"""

from __future__ import annotations

import json
import os
import tempfile

from . import align
from .wafer_model import WaferProject


class StimApp:
    def __init__(self, gui):
        self.gui = gui
        self.project = WaferProject()
        self._log_sinks = []        # extra log targets (e.g. the shared footer log)

    # -- passthrough state --
    @property
    def viewer(self):
        return getattr(self.gui, "viewer", None)

    @property
    def geom(self):
        return getattr(self.gui, "geom", None)

    @property
    def image_path(self):
        return getattr(self.gui, "image_path", None)

    @property
    def overview(self):
        return getattr(self.gui, "overview", None)

    def layer_scale(self):
        try:
            return tuple(self.gui._layer_scale())
        except Exception:
            return (1.0, 1.0)

    def target_long_side(self) -> int:
        """The overview long side the GUI loaded at — workers re-read the image
        at this to recover the same geometry."""
        ov = self.overview
        try:
            return int(max(ov.shape[0], ov.shape[1]))
        except Exception:
            return 4096

    def add_log_sink(self, fn):
        """Register an extra ``fn(line)`` target (the shared footer log) so log
        messages mirror across all tabs, not just the Sections-tab log."""
        self._log_sinks.append(fn)

    def log(self, stage: str, msg: str):
        line = f"[{stage}] {msg}"
        try:
            self.gui.log_msg(line)
        except Exception:
            print(line)
        for fn in self._log_sinks:
            try:
                fn(line)
            except Exception:
                pass

    # -- sections <-> project --
    def section_polygons(self):
        try:
            return list(self.gui.current_polygons_xy())
        except Exception:
            return []

    def fiducials(self):
        try:
            return list(self.gui.current_fiducials_xy())
        except Exception:
            return []

    def sync_sections(self) -> WaferProject:
        """Refresh the project from the live Sections layer. Per-section results
        (pose/qc/roi/order) are preserved when the section count is unchanged
        (the common case: geometry is fixed after detection); otherwise sections
        are rebuilt with fresh stable ids."""
        polys = self.section_polygons()
        self.project.image_path = self.image_path
        if len(polys) == len(self.project.sections) and self.project.sections:
            for s, poly in zip(self.project.sections, polys):
                s.polygon = [[float(x), float(y)] for x, y in poly]
        else:
            self.project.set_sections_from_polygons(polys)
        self.project.fiducials = [(float(x), float(y)) for x, y in self.fiducials()]
        return self.project

    def ensure_poses(self):
        """Estimate a shape pose for any section that lacks one."""
        for s in self.project.sections:
            if s.pose.center is None and len(s.polygon) >= 3:
                align.pose_for_section(s)

    # -- worker plumbing --
    def write_sections_tempfile(self) -> str:
        """Write ``[{id, polygon}]`` for the current sections to a temp JSON the
        QC/reorder workers read. Returns the path."""
        payload = [{"id": s.id, "polygon": s.polygon} for s in self.project.sections]
        fd, path = tempfile.mkstemp(suffix="_stim_sections.json")
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        return path

    def has_image(self) -> bool:
        return self.image_path is not None and self.overview is not None

    # -- workflow-result persistence (sidecar, separate from the GUI autosave) --
    def save_workflow(self):
        """Persist QC/order/ROI/pose/match-graph to the workflow sidecar."""
        if not self.has_image():
            return
        from . import layer_sync, project_io
        self.sync_sections()
        try:
            self.project.display_settings = layer_sync.capture_display(self)
        except Exception:
            pass
        project_io.save(self.project, self.geom,
                        path=project_io.workflow_path(self.image_path))

    def load_workflow(self) -> bool:
        """Restore saved workflow results onto the current sections (matched by
        id). Returns True if a sidecar was found and merged."""
        if not self.has_image():
            return False
        from . import project_io
        src = project_io.load(self.image_path, self.geom,
                              path=project_io.workflow_path(self.image_path))
        if src is None:
            return False
        self.sync_sections()
        self.project.apply_results(src)
        return True

    def restore_annotations_from_czi(self) -> tuple:
        """Read ROIs + focus points from the CZI's CAT annotations and attach them
        to the (already-synced) sections by geometric containment. Returns
        ``(n_rois, n_focus)``. The section polygons are restored by the GUI's
        section layer; this fills in the per-section ROI + focus state the GUI
        layer can't carry (so an annotated CZI reloads ROIs/focus, not just
        sections). No-op without a CZI source + geometry, or when the CZI carries
        no CAT ROI/focus annotations."""
        from . import czi_export, czi_io
        from .wafer_model import Roi
        import numpy as np

        if not (self.image_path and czi_io.is_czi(self.image_path)
                and self.geom is not None):
            return (0, 0)
        try:
            ann = czi_export.read_cat_annotations(self.image_path)
        except Exception:
            return (0, 0)
        rois_full = ann.get("rois", [])
        # Focus points live in ZEN's TileRegion autofocus SupportPoints (stage µm),
        # NOT as pixel annotations. Read them from there (already mapped to overview
        # px) and fall back to the legacy STiM_FocusPoints pixel layer for older
        # exports that predate this.
        focus_overview = []
        try:
            acq = czi_export.read_acquisition_overview(self.image_path, self.geom)
            focus_overview = [(float(p[0]), float(p[1]))
                              for p in acq.get("focus_points", [])]
        except Exception:
            focus_overview = []
        focus_full = ann.get("focus", [])      # legacy pixel-layer fallback
        if not rois_full and not focus_full and not focus_overview:
            return (0, 0)

        proj = self.sync_sections()
        geom = self.geom

        def _to_overview(pts_full):
            a = np.asarray(pts_full, float).reshape(-1, 2)
            ox, oy = geom.full_to_ds(a[:, 0], a[:, 1])
            return [(float(x), float(y)) for x, y in zip(np.ravel(ox), np.ravel(oy))]

        # Clear what we're about to repopulate so a reload doesn't accumulate.
        for s in proj.sections:
            s.roi = None
            s.focus_overview = []

        n_rois = 0
        for poly_full in rois_full:
            ov = _to_overview(poly_full)
            sec = self._section_containing(proj, ov)
            if sec is not None:
                sec.roi = Roi(polygon=[[x, y] for x, y in ov], fit_mode="manual")
                n_rois += 1

        # Prefer the TileRegion support points; only fall back to the legacy pixel
        # layer when no support points were found.
        if not focus_overview and focus_full:
            focus_overview = [_to_overview([(fx, fy)])[0] for (fx, fy) in focus_full]
        n_focus = 0
        for (ox, oy) in focus_overview:
            sec = self._section_containing(proj, [(ox, oy)])
            if sec is not None:
                sec.focus_overview.append((ox, oy))
                n_focus += 1
        return (n_rois, n_focus)

    @staticmethod
    def _section_containing(proj, pts_overview):
        """The section whose polygon contains the centroid of ``pts_overview``
        (overview px), else the nearest section by centroid. Mirrors the ROI
        stage's reference-section lookup so loaded ROIs land on the same section
        the user drew them on."""
        import numpy as np
        cx, cy = np.asarray(pts_overview, float).reshape(-1, 2).mean(axis=0)
        try:
            from shapely.geometry import Point, Polygon
            pt = Point(cx, cy)
            for s in proj.sections:
                if len(s.polygon) >= 3 and Polygon(s.polygon).buffer(0).contains(pt):
                    return s
        except Exception:
            pass
        best, bd = None, 1e30
        for s in proj.sections:
            sx, sy = s.centroid()
            d = (sx - cx) ** 2 + (sy - cy) ** 2
            if d < bd:
                best, bd = s, d
        return best

    @staticmethod
    def _section_locator(proj):
        """Build a fast point→section lookup ONCE (shapely polygons + STRtree),
        returning ``locate(cx, cy) -> section``. Reused across every shape in a
        capture pass, so containment is O(shapes·log·sections) instead of
        rebuilding a Polygon per section per shape (which made autosave freeze the
        UI on wafers with many ROIs). Falls back to nearest-centroid."""
        import numpy as np
        polys, valid = [], []
        Point = None
        try:
            from shapely.geometry import Point, Polygon
            for s in proj.sections:
                if len(s.polygon) >= 3:
                    try:
                        g = Polygon(s.polygon).buffer(0)
                    except Exception:
                        g = None
                    if g is not None and not g.is_empty:
                        polys.append(g); valid.append(s)
        except Exception:
            polys, valid = [], []
        tree = None
        if polys:
            try:
                from shapely import STRtree
                tree = STRtree(polys)
            except Exception:
                tree = None
        cents = np.array([s.centroid() for s in valid], float) if valid else None

        def locate(cx, cy):
            if Point is not None and polys:
                pt = Point(cx, cy)
                if tree is not None:
                    for i in np.atleast_1d(tree.query(pt)):
                        i = int(i)
                        if 0 <= i < len(polys) and polys[i].contains(pt):
                            return valid[i]
                else:
                    for g, s in zip(polys, valid):
                        if g.contains(pt):
                            return s
            if cents is not None and len(cents):
                d = (cents[:, 0] - cx) ** 2 + (cents[:, 1] - cy) ** 2
                return valid[int(d.argmin())]
            return None
        return locate

    # -- live capture of hand-edits made to the ROI / focus overlays ----------
    def capture_annotations(self) -> bool:
        """Read native napari edits (move / add / delete) from the live "② ROIs"
        and "Focus points" overlays back into the per-section model, so they
        survive save + reload. Each shape / point is assigned to the section
        whose polygon contains its centroid. Returns True if either overlay was
        present (and therefore captured — including having been emptied).

        This is what makes the overlays authoritative: without it, edits live
        only in the napari layers, which are regenerated from the model on the
        next refresh and never persisted."""
        import numpy as np
        from . import layer_sync
        from .wafer_model import Roi

        viewer = self.viewer
        if viewer is None or not self.project.sections:
            return False
        proj = self.project
        touched = False

        def _area(poly):
            p = np.asarray(poly, float).reshape(-1, 2)
            x, y = p[:, 0], p[:, 1]
            return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0

        has_roi = layer_sync.ROI_LAYER in viewer.layers
        has_focus = layer_sync.FOCUS_LAYER in viewer.layers
        if not (has_roi or has_focus):
            return False
        locate = self._section_locator(proj)         # built once, reused below

        # ROIs: Shapes layer stored as napari (y, x); model keeps (x, y). The
        # model holds one ROI per section, so on a collision keep the largest.
        if has_roi:
            for s in proj.sections:
                s.roi = None
            dropped = 0
            for shp in list(viewer.layers[layer_sync.ROI_LAYER].data):
                arr = np.asarray(shp, float).reshape(-1, 2)
                if len(arr) < 3:
                    continue
                xy = [[float(x), float(y)] for y, x in arr]
                cx, cy = arr[:, 1].mean(), arr[:, 0].mean()      # centroid (x, y)
                sec = locate(cx, cy)
                if sec is None:
                    continue
                if sec.roi is not None:
                    dropped += 1
                    if _area(xy) <= _area(sec.roi.polygon):
                        continue
                sec.roi = Roi(polygon=xy, fit_mode="manual")
            if dropped:
                self.log("rois", f"{dropped} ROI shape(s) shared a section with "
                                 "another; kept the largest (one ROI per section).")
            touched = True

        # Focus points: Points layer stored as napari (y, x).
        if has_focus:
            for s in proj.sections:
                s.focus_overview = []
            for p in np.asarray(viewer.layers[layer_sync.FOCUS_LAYER].data,
                                float).reshape(-1, 2):
                ox, oy = float(p[1]), float(p[0])
                sec = locate(ox, oy)
                if sec is not None:
                    sec.focus_overview.append((ox, oy))
            touched = True

        return touched

    def save_all(self) -> bool:
        """Capture every editable overlay into the model and persist everything:
        the legacy project JSON (sections / fiducials / calibration) and the
        workflow sidecar (ROIs / focus / poses / order / display). Safe to call
        often; used by the debounced autosave, the manual Save button, and on
        quit. Returns True on success."""
        if not self.has_image():
            return False
        try:
            self.sync_sections()
            self.capture_annotations()
        except Exception:
            pass
        ok = False
        try:
            save_project = getattr(self.gui, "save_project", None)
            if callable(save_project):
                save_project()
        except Exception:
            pass
        try:
            self.save_workflow()
            ok = True
        except Exception:
            pass
        return ok
