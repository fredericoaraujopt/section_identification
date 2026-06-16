"""Relative-FOV navigator controller (napari camera <-> fov_nav math).

Drives the inspection gesture: stepping section→section keeps the same
magnification and the same *relative* position within the section (so a feature
you're zoomed on stays in view across sections). First visit to a section fits
it; subsequent navigations preserve the relative view captured from wherever the
user currently is.

napari camera works in world (y, x); section poses are in overview (x, y). This
converts between them via the Sections layer scale, then delegates the geometry
to fov_nav (frame-agnostic).
"""

from __future__ import annotations

from . import fov_nav
from .fov_nav import _SecPose


class FovNavigator:
    def __init__(self, app):
        self.app = app
        self._current = None        # current Section being inspected

    # -- frame helpers --
    def _scale(self):
        s = self.app.layer_scale()
        return (float(s[0]), float(s[1]))      # (sy, sx)

    def _world_pose(self, section):
        sy, sx = self._scale()
        cx, cy = section.pose.center if section.pose.center else section.centroid()
        return _SecPose(center=(cx * sx, cy * sy),
                        angle_deg=section.pose.angle_deg, flip=section.pose.flip)

    def _world_bbox(self, section):
        sy, sx = self._scale()
        x0, y0, x1, y1 = section.bbox()
        return (x0 * sx, y0 * sy, x1 * sx, y1 * sy)

    def _canvas_px(self):
        # napari 0.7 moved the qt viewer to a private attr on some builds; try both.
        for getter in (lambda: self.app.viewer.window._qt_viewer.canvas.size,
                       lambda: self.app.viewer.window.qt_viewer.canvas.size):
            try:
                sz = getter()
                if sz and sz[0] and sz[1]:
                    return (float(sz[0]), float(sz[1]))
            except Exception:
                continue
        return (900.0, 700.0)

    def _consistent_zoom(self):
        """A single zoom used for ALL sections so they appear at the same
        magnification — the median section spans ~60% of the canvas."""
        secs = self.app.project.sections
        sy, sx = self._scale()
        sizes = []
        for s in secs:
            x0, y0, x1, y1 = s.bbox()
            sizes.append(max((x1 - x0) * sx, (y1 - y0) * sy))
        sizes = [v for v in sizes if v > 0]
        if not sizes:
            return None
        med = sorted(sizes)[len(sizes) // 2]
        cw, ch = self._canvas_px()
        return float(min(cw, ch)) / (med * 1.6)

    def _cam_world_xy(self):
        c = list(self.app.viewer.camera.center)
        return (float(c[-1]), float(c[-2]))    # (x, y)

    def _set_cam(self, world_x, world_y, zoom=None):
        cam = self.app.viewer.camera
        c = list(cam.center)
        c[-1], c[-2] = float(world_x), float(world_y)
        cam.center = tuple(c)
        if zoom is not None:
            cam.zoom = float(zoom)

    # -- navigation --
    def _center_world(self, section):
        sy, sx = self._scale()
        cx, cy = section.centroid()
        return (cx * sx, cy * sy)

    def center_on(self, section):
        """Recenter on a section, KEEPING the current zoom (single-click)."""
        if section is None:
            return
        wx, wy = self._center_world(section)
        self._set_cam(wx, wy)
        self._current = section

    def fit_consistent(self, section):
        """Center on a section at the project-wide CONSISTENT magnification
        (double-click) — every section snaps to the same zoom."""
        if section is None:
            return
        z = self._consistent_zoom()
        if z is None:
            return self.fit(section)
        wx, wy = self._center_world(section)
        self._set_cam(wx, wy, z)
        self._current = section

    def fit(self, section):
        (cx, cy), zoom = fov_nav.fit_center_zoom(self._world_bbox(section),
                                                 self._canvas_px(), margin=0.2)
        self._set_cam(cx, cy, zoom)
        self._current = section

    def go_to(self, section, keep_fov: bool = True):
        """Navigate to ``section``. If we were inspecting another section and
        ``keep_fov``, preserve the current relative view + magnification;
        otherwise fit the section."""
        if section is None:
            return
        if keep_fov and self._current is not None and self._current is not section:
            try:
                cam_xy = self._cam_world_xy()
                rel = fov_nav.relative_offset(cam_xy, self._world_pose(self._current))
                nx, ny = fov_nav.snapped_center(rel, self._world_pose(section))
                self._set_cam(nx, ny)        # keep current zoom
                self._current = section
                return
            except Exception:
                pass
        self.fit(section)

    def go_to_index(self, idx: int, keep_fov: bool = True):
        secs = self.app.project.sections
        if 0 <= idx < len(secs):
            self.go_to(secs[idx], keep_fov=keep_fov)

    def step(self, delta: int, keep_fov: bool = True):
        secs = self.app.project.sections
        if not secs:
            return
        cur = self._current
        try:
            i = secs.index(cur) if cur in secs else -1
        except Exception:
            i = -1
        self.go_to_index(max(0, min(len(secs) - 1, i + delta)), keep_fov=keep_fov)
