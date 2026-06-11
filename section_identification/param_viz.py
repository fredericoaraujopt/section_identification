"""Live, on-image visualizations of the detection parameters.

Toggling a parameter should let the user *see*, by inspection, how SAM will
behave — grounded in what SAM actually does. This draws, as napari overlay
layers that update the instant a slider moves:

  * ≈ tiles      — how the image is split into tiles (SAM upscales each to 1024).
  * ≈ grid       — SAM's real query-point grid (points_per_side) in a tile under
                   the current view: this *is* what SAM samples.
  * ≈ crops      — SAM's built-in sub-crops (crop_n_layers / crop_overlap_ratio).
  * ≈ min area   — a reference disc of the section-area floor, so its size is
                   obvious next to real sections.

All overlays live in OVERVIEW-pixel coordinates with the GUI's layer scale, so
they sit exactly on the full-resolution display (same frame as the Sections
layer). Coordinates handed to napari are (y, x).
"""

from __future__ import annotations

import numpy as np

MAX_GRID_POINTS = 6000   # safety cap on drawn query points


def _grid_points(box, pps):
    """SAM's query-point grid inside a tile box (x0,y0,w,h) → Nx2 (x,y) overview.

    Mirrors SAM's build_point_grid: points at (i+0.5)/pps across the tile."""
    x0, y0, w, h = box
    pps = max(1, int(pps))
    offs = (np.arange(pps) + 0.5) / pps
    xs = x0 + offs * w
    ys = y0 + offs * h
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    if len(pts) > MAX_GRID_POINTS:           # subsample for display only
        step = int(np.ceil(len(pts) / MAX_GRID_POINTS))
        pts = pts[::step]
    return pts


def _crop_boxes(box, n_layers, overlap_ratio):
    """SAM's overlapping sub-crops inside a tile (layer 1 = 2x2, layer 2 = 4x4…)."""
    x0, y0, w, h = box
    boxes = []
    for layer in range(1, int(n_layers) + 1):
        n = 2 ** layer
        ov = overlap_ratio * min(w, h) * (2.0 / n)
        cw = (w + (n - 1) * ov) / n
        ch = (h + (n - 1) * ov) / n
        for i in range(n):
            for j in range(n):
                cx0 = x0 + i * (cw - ov)
                cy0 = y0 + j * (ch - ov)
                boxes.append((cx0, cy0, cw, ch))
    return boxes


def _rect_yx(box):
    x0, y0, w, h = box
    return np.array([[y0, x0], [y0, x0 + w], [y0 + h, x0 + w], [y0 + h, x0]], float)


class ParamVisualizer:
    L_TILES, L_GRID, L_CROPS, L_MIN = "≈ tiles", "≈ grid", "≈ crops", "≈ min area"

    def __init__(self, gui):
        self.gui = gui
        self.viewer = gui.viewer
        self._active = False
        self._layers = {}

    # ---- geometry helpers (overview pixels) ----
    def _wh(self):
        return self.gui.overview.shape[1], self.gui.overview.shape[0]

    def _scale(self):
        return self.gui._layer_scale()

    def _representative_tile(self):
        """The central tile (x0,y0,w,h) in overview px — FIXED (does not follow
        the camera), so the grid/crops/min-area don't jitter as you pan/zoom."""
        from section_identification.tiled_detect import tile_boxes
        W, H = self._wh()
        tile = int(self.gui.sp_tile.value())
        if tile <= 0 or tile >= max(W, H):
            return (0, 0, W, H), True                     # whole image = one tile
        boxes = tile_boxes(W, H, tile, float(self.gui.sp_overlap.value()))
        cx, cy = W / 2.0, H / 2.0
        b = min(boxes, key=lambda t: (t[0] + t[2] / 2 - cx) ** 2 + (t[1] + t[3] / 2 - cy) ** 2)
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3])), False

    # ---- layer plumbing ----
    def _set(self, name, kind, data, **kw):
        lyr = self._layers.get(name)
        if lyr is not None and lyr not in self.viewer.layers:
            lyr = None
        if lyr is None:
            adder = self.viewer.add_points if kind == "points" else self.viewer.add_shapes
            lyr = adder(data, name=name, scale=self._scale(), **kw)
            self._layers[name] = lyr
        else:
            lyr.data = data
        return lyr

    def refresh(self):
        if not self._active or self.gui.overview is None:
            return
        try:
            self._refresh()
        except Exception:
            pass

    def _refresh(self):
        from section_identification.tiled_detect import tile_boxes
        W, H = self._wh()
        tile = int(self.gui.sp_tile.value())
        overlap = float(self.gui.sp_overlap.value())
        rep, whole = self._representative_tile()

        # ≈ tiles — the full tile grid (yellow), faint
        tboxes = ([(0, 0, W, H)] if whole
                  else tile_boxes(W, H, max(64, tile), overlap))
        self._set(self.L_TILES, "shapes", [_rect_yx(b) for b in tboxes],
                  shape_type="rectangle", edge_color="yellow", face_color=[0, 0, 0, 0],
                  edge_width=max(1, int(0.004 * max(W, H))))

        # ≈ grid — SAM's query points inside the representative tile (magenta)
        pts = _grid_points(rep, self.gui.sp_pps.value())
        self._set(self.L_GRID, "points", pts[:, ::-1],  # (x,y)→(y,x)
                  face_color="magenta", border_color="magenta",
                  size=max(3, int(0.004 * max(W, H))))

        # ≈ crops — SAM's sub-crops inside the representative tile (orange).
        # Empty data when there are no crops (a placeholder rect would render as
        # a stray degenerate shape).
        cboxes = _crop_boxes(rep, self.gui.sp_crop.value(), self.gui.sp_cropov.value())
        self._set(self.L_CROPS, "shapes", [_rect_yx(b) for b in cboxes],
                  shape_type="rectangle", edge_color="orange", face_color=[0, 0, 0, 0],
                  edge_width=max(1, int(0.003 * max(W, H))))

        # ≈ min area — reference disc of the section-area floor at the tile corner.
        # napari 'ellipse' wants the 4 bounding-box corners (a 2-point bbox is
        # mis-rendered), so build the full rectangle.
        area = float(self.gui.sp_minarea.value())
        r = max(1.0, np.sqrt(max(area, 1.0) / np.pi))
        cx0, cy0 = rep[0] + 1.5 * r, rep[1] + 1.5 * r
        disc = np.array([[cy0 - r, cx0 - r], [cy0 - r, cx0 + r],
                         [cy0 + r, cx0 + r], [cy0 + r, cx0 - r]], dtype=float)
        self._set(self.L_MIN, "shapes", [disc], shape_type="ellipse",
                  edge_color="lime", face_color=[0, 1, 0, 0.25],
                  edge_width=max(1, int(0.003 * max(W, H))))

    # ---- activation ----
    def set_active(self, on):
        self._active = bool(on)
        if on:
            self._refresh()
        else:
            for name in list(self._layers):
                lyr = self._layers.pop(name)
                try:
                    if lyr in self.viewer.layers:
                        self.viewer.layers.remove(lyr)
                except Exception:
                    pass

    def refresh_if_active(self, *a):
        if self._active:
            self.refresh()
