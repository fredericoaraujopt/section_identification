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
        try:
            sz = self.app.viewer.window.qt_viewer.canvas.size
            return (float(sz[0]), float(sz[1]))
        except Exception:
            return (900.0, 700.0)

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
