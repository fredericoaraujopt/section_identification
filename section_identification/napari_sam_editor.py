"""Napari-native interactive SAM editor — an in-viewer counterpart to the
OpenCV manual detector, running SAM on a FULL-RESOLUTION crop of the current
napari view so masks are sharp on individual sections.

It rides on the GUI's lazy full-res multiscale display: the image's WORLD
coordinates are full-resolution pixels, while the "Sections"/"Fiducials" layers
store OVERVIEW-pixel data shown via a per-layer ``scale`` of 1/geom.zoom. This
editor works in world (full-res) coords for SAM and converts committed masks
back to overview-pixel data (× geom.zoom), so detection / save / load / export
stay unchanged.

Design notes (why it is structured this way):
  * The SAM encoder runs SYNCHRONOUSLY on the main thread — PyTorch MPS is not
    thread-safe, so a worker thread risks crashes. The embed is a brief explicit
    pause (like the OpenCV editor).
  * Hover prediction and the click-commit run OFF the mouse-event callback (via
    short QTimers / deferred call). Doing SAM work inside the event re-enters
    napari's drag generator ("generator already executing").
  * The IMAGE layer is kept active so napari's Shapes-layer key map (r=rectangle,
    p=polygon, …) does not shadow our 'r'/'m'/'d'/'e' bindings.

Controls while active:
  (auto)  zoom into a region → the view is embedded at full-res automatically
  e       force re-embed the current view
  hover   live yellow mask preview under the cursor
  click   commit the previewed mask to the Sections layer
  r       select the section under the cursor (turns magenta); 'r' again removes it
  m       drop a fiducial at the cursor
  d       toggle the preview on/off
"""
import threading

import numpy as np
import cv2

from section_identification.device import device_str

# SAM resizes every input to 1024 px, so we read the current view at a zoom that
# keeps the crop ≈ this size: full resolution when zoomed in, downscaled when
# zoomed out. This makes the editor work at ANY zoom and bounds the encode cost
# (constant ~1024-px encode) so it can't blow up memory/runtime on weak machines.
ENCODE_PX = 1024
HOVER_DEBOUNCE_MS = 60       # mouse settle before running a hover prediction
CAMERA_DEBOUNCE_MS = 400     # camera settle before auto-embedding the new view


def _xy_to_yx(p):
    p = np.asarray(p, dtype=float).reshape(-1, 2)
    return p[:, ::-1]


class NapariSamEditor:
    PREVIEW_NAME = "SAM preview"
    SELECT_NAME = "SAM selection"

    def __init__(self, gui):
        self.gui = gui
        self.viewer = gui.viewer
        self._active = False
        self.predictor = None
        self.preview_layer = None
        self.sel_layer = None
        self._embed_ready = False
        self._embedding = False
        self._busy = False
        self._show_preview = True
        self._crop_w = self._crop_h = 0
        self._view_world_origin = (0.0, 0.0)
        self._embed_rect = None          # (wx0, wy0, wx1, wy1) world of current embed
        self._read_zoom = 1.0            # zoom the current crop was read at (≤1)
        self._cursor_world = None        # (x, y) world
        self._sel_idx = None             # index of the section selected for removal
        self.frame_layer = None          # red "manual mode" border
        self._lock = threading.Lock()
        self._keybinds = []          # list of (target, key) we bound
        self._hover_timer = None
        self._cam_timer = None
        self._cam_connected = False
        self._sel_connected = False
        self._reasserting = False    # guard against re-entrant active-layer resets
        self._dragging = False       # mouse held (panning) → defer embedding
        self._drag_cb = None
        # When set (by the ROIs stage), a committed mask is routed here as an
        # overview-(y,x) polygon instead of being appended to the Sections layer,
        # so the same editor can outline ROIs. Reset on deactivate.
        self.commit_target = None

    # ---------------- helpers ----------------
    def _log(self, msg):
        try:
            self.gui.log_msg(msg)
        except Exception:
            print(msg)

    def _zoom(self):
        """geom.zoom: world(full-res px) * zoom = overview-pixel (layer data)."""
        g = self.gui.geom
        return g.zoom if g is not None else 1.0

    def _image_wh(self):
        """Full image size in WORLD px — CZI multiscale level 0, else the loaded
        overview (PNG/other have no pyramid; world == overview px)."""
        lv = getattr(self.gui, "_display_levels", None)
        if lv:
            return lv[0].shape[1], lv[0].shape[0]
        ov = self.gui.overview
        return ov.shape[1], ov.shape[0]

    def _view_extent_world(self):
        """(wx0, wy0, wx1, wy1) of the visible canvas in full-res world px."""
        cam = self.viewer.camera
        try:
            cx = float(cam.center[-1]); cy = float(cam.center[-2])   # (z,y,x)
            zoom = float(cam.zoom)
        except Exception:
            return None
        if zoom <= 0:
            return None
        cw = ch = None
        for getter in (
            lambda: self.viewer.window._qt_viewer.canvas.size,
            lambda: self.viewer.window.qt_viewer.canvas.size,
        ):
            try:
                cw, ch = getter()
                break
            except Exception:
                continue
        if not cw or not ch:
            cw, ch = 1000, 800                                       # safe default
        hw, hh = cw / (2.0 * zoom), ch / (2.0 * zoom)
        return (cx - hw, cy - hh, cx + hw, cy + hh)

    @staticmethod
    def _contains(rect, ext, margin=0.0):
        """True if `rect` (with optional shrink `margin` frac) covers `ext`."""
        if rect is None or ext is None:
            return False
        mx = (rect[2] - rect[0]) * margin
        my = (rect[3] - rect[1]) * margin
        return (rect[0] - mx <= ext[0] and rect[1] - my <= ext[1]
                and rect[2] + mx >= ext[2] and rect[3] + my >= ext[3])

    # ---------------- embedding (synchronous, main thread) ----------------
    def _read_view_crop(self):
        """(crop_rgb, (wx0,wy0) world origin, read_zoom) for the current view, or
        None. Reads at a zoom that keeps the crop ≈ENCODE_PX (full-res when zoomed
        in, downscaled when zoomed out → works at ANY zoom). Supports CZI
        (read_czi_region) AND ordinary images (slices gui.overview)."""
        from section_identification import czi_io
        ext = self._view_extent_world()
        if ext is None:
            return None
        W, H = self._image_wh()
        wx0 = max(0.0, ext[0]); wy0 = max(0.0, ext[1])
        wx1 = min(float(W), ext[2]); wy1 = min(float(H), ext[3])
        ww, wh = wx1 - wx0, wy1 - wy0
        if min(ww, wh) < 4:
            return None
        rz = min(1.0, float(ENCODE_PX) / max(ww, wh))
        geom = self.gui.geom
        if geom is not None and czi_io.is_czi(self.gui.image_path):
            crop = czi_io.read_czi_region(
                self.gui.image_path, int(round(geom.origin_x + wx0)),
                int(round(geom.origin_y + wy0)), int(round(ww)), int(round(wh)), zoom=rz)
        else:                                       # PNG / other: slice the loaded image
            ov = self.gui.overview
            sub = ov[int(wy0):int(round(wy1)), int(wx0):int(round(wx1))]
            if sub.ndim == 2:
                sub = np.repeat(sub[:, :, None], 3, axis=2)
            sub = np.ascontiguousarray(sub[:, :, :3])
            if rz < 0.999 and sub.size:
                sub = cv2.resize(sub, (max(1, int(round(sub.shape[1] * rz))),
                                       max(1, int(round(sub.shape[0] * rz)))),
                                 interpolation=cv2.INTER_AREA)
            crop = sub
        if crop is None or getattr(crop, "size", 0) == 0:
            return None
        return np.ascontiguousarray(crop), (wx0, wy0), rz

    def _build_predictor(self):
        """SAM2 image predictor with the host-adaptive model (tiny/small on weak
        machines → much faster interactive encode)."""
        from section_identification.section_detector import build_image_predictor
        prof = self.gui._current_profile()
        ckpt = self.gui._checkpoint_for_model("Auto", prof)
        dev = device_str(getattr(self.gui, "_device_prefer", "") or None)
        return build_image_predictor(ckpt, None, dev)

    def _embed_view(self, force=False):
        if self._embedding or self.gui.overview is None:
            return
        ext = self._view_extent_world()
        if ext is None:
            return
        rz_now = min(1.0, float(ENCODE_PX) / max(ext[2] - ext[0], ext[3] - ext[1], 1.0))
        # skip if the current embedding still covers the view at a similar zoom
        if (not force and self._embed_ready and self._contains(self._embed_rect, ext)
                and abs(rz_now - self._read_zoom) <= 0.25 * max(rz_now, 1e-6)):
            return
        got = self._read_view_crop()
        if got is None:
            return
        crop, origin, rz = got
        self._embedding = True
        self._embed_ready = False
        self._log(f"napari editor: embedding view ({crop.shape[1]}×{crop.shape[0]} px)…")
        try:
            from qtpy.QtWidgets import QApplication
            QApplication.processEvents()
        except Exception:
            pass
        try:
            if self.predictor is None:
                self.predictor = self._build_predictor()
            with self._lock:
                self.predictor.set_image(crop)
            self._view_world_origin = origin
            self._read_zoom = rz
            self._crop_w, self._crop_h = crop.shape[1], crop.shape[0]
            self._embed_rect = (ext[0], ext[1], ext[2], ext[3])
            self._embed_ready = True
            self._log("napari editor: ready — hover to preview, Space to add.")
        except Exception as e:
            self._embed_ready = False
            self._log(f"napari editor: embed failed: {e}")
        finally:
            self._embedding = False
        self._do_hover()

    def _schedule_autoembed(self, *a):
        # While the mouse is held (panning) we never embed — restarting the timer
        # would just thrash. The drag-release handler restarts it once settled.
        if self._active and self._cam_timer is not None and not self._dragging:
            self._cam_timer.start(CAMERA_DEBOUNCE_MS)

    def _autoembed(self):
        # Skip if a drag is still in progress (the user is still moving); the
        # release handler will reschedule once they've actually stopped.
        if self._active and not self._embedding and not self._dragging:
            self._embed_view(force=False)

    def _on_drag(self, viewer, event):
        """Track left-drag (pan) so the encoder doesn't fire mid-move. Generator:
        runs at press, yields through the drag, resumes on release."""
        if not self._active:
            return
        self._dragging = True
        if self._cam_timer is not None:
            self._cam_timer.stop()                 # cancel any pending embed
        yield
        while event.type == "mouse_move":
            yield
        self._dragging = False                     # released → embed once settled
        if self._active and self._cam_timer is not None:
            self._cam_timer.start(CAMERA_DEBOUNCE_MS)

    # ---------------- prediction ----------------
    def _predict_overview_polygon(self, world_x, world_y):
        rz = self._read_zoom or 1.0
        cx = (world_x - self._view_world_origin[0]) * rz     # world → crop px
        cy = (world_y - self._view_world_origin[1]) * rz
        if not (0 <= cx < self._crop_w and 0 <= cy < self._crop_h):
            return None
        with self._lock:
            masks, _scores, _ = self.predictor.predict(
                point_coords=np.array([[cx, cy]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.float32),
                multimask_output=False)
        m = np.squeeze(np.asarray(masks)).astype(np.uint8)
        if m.ndim != 2 or m.sum() == 0:
            return None
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)   # crop px
        z = self._zoom()
        wx = self._view_world_origin[0] + c[:, 0] / rz       # crop px → world
        wy = self._view_world_origin[1] + c[:, 1] / rz
        return np.column_stack([wx * z, wy * z])                          # overview xy

    # ---------------- hover (debounced, off the event) ----------------
    def _on_move(self, viewer, event):
        # keep this trivial — heavy work here re-enters napari's drag generator
        if not self._active:
            return
        self._cursor_world = (float(event.position[-1]), float(event.position[-2]))
        if self._hover_timer is not None:
            self._hover_timer.start(HOVER_DEBOUNCE_MS)

    def _do_hover(self):
        if not self._active or not self._show_preview or self._busy:
            return
        if self._embedding or not self._embed_ready or self.preview_layer is None:
            return
        if self._cursor_world is None:
            return
        self._busy = True
        try:
            poly = self._predict_overview_polygon(*self._cursor_world)
            if poly is not None and len(poly) >= 3:
                self.preview_layer.data = [_xy_to_yx(poly)]
            else:
                self.preview_layer.data = []
        except Exception:
            pass
        finally:
            self._busy = False

    # ---------------- commit (Space) ----------------
    def _commit_preview(self):
        if not self._active or self.preview_layer is None:
            return
        data = list(self.preview_layer.data)
        if not data:
            self._log("napari editor: nothing to add (hover a section first).")
            return
        poly = np.asarray(data[-1], dtype=float)        # overview (y, x)
        self.preview_layer.data = []
        if self.commit_target is not None:               # ROI mode: route elsewhere
            try:
                self.commit_target(poly)
            except Exception as e:
                self._log(f"napari editor: ROI capture error: {e}")
            return
        self.gui.shapes_layer.data = list(self.gui.shapes_layer.data) + [poly]
        try:
            self.gui.save_project()
        except Exception:
            pass
        self._log(f"napari editor: section added "
                  f"({len(self.gui.shapes_layer.data)} total).")

    # ---------------- r: select (highlight) then remove ----------------
    def _section_under_cursor(self):
        if self.gui.shapes_layer is None:
            return None
        pos = self.viewer.cursor.position
        z = self._zoom()
        ov_x, ov_y = float(pos[-1]) * z, float(pos[-2]) * z
        best, best_area = None, None
        for i, poly_yx in enumerate(self.gui.shapes_layer.data):
            xy = np.asarray(poly_yx)[:, ::-1].astype(np.float32)
            if len(xy) < 3:
                continue
            cnt = xy.reshape(-1, 1, 2)
            if cv2.pointPolygonTest(cnt, (ov_x, ov_y), False) >= 0:
                a = cv2.contourArea(cnt)
                if best is None or a < best_area:
                    best, best_area = i, a
        return best

    def _highlight(self, idx):
        data = self.gui.shapes_layer.data
        if self.sel_layer is not None and 0 <= idx < len(data):
            self.sel_layer.data = [np.asarray(data[idx], dtype=float)]

    def _clear_selection(self):
        self._sel_idx = None
        if self.sel_layer is not None:
            self.sel_layer.data = []

    def _on_r(self):
        idx = self._section_under_cursor()
        if idx is None:
            self._clear_selection()
            self._log("napari editor: no section under the cursor.")
            return
        if idx == self._sel_idx:
            data = list(self.gui.shapes_layer.data)
            if 0 <= idx < len(data):
                del data[idx]
                self.gui.shapes_layer.data = data
                try:
                    self.gui.save_project()
                except Exception:
                    pass
                self._log(f"napari editor: removed section ({len(data)} remain).")
            self._clear_selection()
        else:
            self._sel_idx = idx
            self._highlight(idx)
            self._log("napari editor: section selected (magenta) — press 'r' again "
                      "to remove it, or 'r' elsewhere to pick another.")

    # ---------------- m / d ----------------
    def _on_m(self):
        fl = self.gui.fid_layer
        if fl is None:
            return
        pos = self.viewer.cursor.position
        z = self._zoom()
        pt = np.array([[float(pos[-2]) * z, float(pos[-1]) * z]])      # overview yx
        d = np.asarray(fl.data)
        fl.data = np.vstack([d, pt]) if len(d) else pt
        try:
            self.gui.save_project()
        except Exception:
            pass
        self._log(f"napari editor: fiducial added ({len(fl.data)} total).")

    def _toggle_preview(self):
        self._show_preview = not self._show_preview
        if self.preview_layer is not None:
            self.preview_layer.visible = self._show_preview
            if not self._show_preview:
                self.preview_layer.data = []
        self._log(f"napari editor: preview {'ON' if self._show_preview else 'OFF'}.")

    # ---------------- activate / deactivate ----------------
    def toggle(self):
        self.deactivate() if self._active else self.activate()
        return self._active

    def activate(self, commit_target=None):
        if self._active:
            return
        if self.gui.overview is None or self.gui.shapes_layer is None:
            self._log("napari editor: load an image first.")
            return
        self.commit_target = commit_target
        sc = self.gui._layer_scale()
        # thick red border around the image → unmistakable "manual editor" mode
        H, W = self.gui.overview.shape[:2]
        frame = np.array([[0, 0], [0, W], [H, W], [H, 0]], dtype=float)
        self.frame_layer = self.viewer.add_shapes(
            [frame], shape_type="rectangle", name="● Manual editor (Space=add)",
            edge_color="red", face_color=[0, 0, 0, 0],
            edge_width=max(4.0, max(W, H) * 0.004), scale=sc)
        self.sel_layer = self.viewer.add_shapes(
            [], shape_type="polygon", name=self.SELECT_NAME,
            face_color=[1, 0, 1, 0.25], edge_color="magenta", edge_width=6, scale=sc)
        self.preview_layer = self.viewer.add_shapes(
            [], shape_type="polygon", name=self.PREVIEW_NAME,
            face_color=[1, 1, 0, 0.30], edge_color="yellow", edge_width=4, scale=sc)
        for lyr in (self.frame_layer, self.preview_layer, self.sel_layer, self.gui.shapes_layer):
            try:
                lyr.mode = "pan_zoom"
            except Exception:
                pass
        # keep the IMAGE layer active so the Shapes key map (r/p/l/…) can't shadow ours
        try:
            self.viewer.layers.selection.active = self.gui.image_layer
        except Exception:
            pass

        from qtpy.QtCore import QTimer
        self._hover_timer = QTimer(); self._hover_timer.setSingleShot(True)
        self._hover_timer.timeout.connect(self._do_hover)
        self._cam_timer = QTimer(); self._cam_timer.setSingleShot(True)
        self._cam_timer.timeout.connect(self._autoembed)

        self.viewer.mouse_move_callbacks.append(self._on_move)
        self._drag_cb = self._on_drag
        self.viewer.mouse_drag_callbacks.append(self._drag_cb)   # defer embed while panning
        # No click handler: left-drag pans the image (napari default). The user
        # adds the previewed mask with Space instead, so panning is safe.
        try:
            self.viewer.camera.events.zoom.connect(self._schedule_autoembed)
            self.viewer.camera.events.center.connect(self._schedule_autoembed)
            self._cam_connected = True
        except Exception:
            self._cam_connected = False

        # Bind on the viewer AND the image layer. napari 0.7 registers r/e/d/p/l
        # (and m on Labels) as mode shortcuts on whatever layer is ACTIVE; the
        # active layer's keymap is resolved before the viewer's. We keep the
        # (non-Shapes) image layer active (see _keep_image_active) so those mode
        # shortcuts are never in the chain, and we also put our handlers on the
        # image layer itself so they sit in that highest-priority slot. (Space
        # alone worked before precisely because it isn't a layer mode shortcut.)
        self._keybinds = []
        targets = [self.viewer]
        if self.gui.image_layer is not None:
            targets.append(self.gui.image_layer)
        for k, fn in (("Space", lambda v: self._commit_preview()),
                      ("e", lambda v: self._embed_view(force=True)),
                      ("d", lambda v: self._toggle_preview()),
                      ("r", lambda v: self._on_r()),
                      ("m", lambda v: self._on_m())):
            for tgt in targets:
                try:
                    tgt.bind_key(k, fn, overwrite=True)
                    self._keybinds.append((tgt, k))
                except Exception:
                    pass

        # Keep the image layer active for the editor's lifetime so the Shapes
        # mode shortcuts can't reclaim r/m/d/e if the selection drifts.
        try:
            self.viewer.layers.selection.events.active.connect(self._keep_image_active)
            self._sel_connected = True
        except Exception:
            self._sel_connected = False

        self._active = True
        self._show_preview = True
        self._log("napari editor ON (red border): hover = preview, SPACE = add the "
                  "previewed section, click/drag = pan. r = select/remove section "
                  "under cursor, m = fiducial, d = toggle preview, e = re-embed view. "
                  "Works at any zoom. Click the button again to stop.")
        self._autoembed()

    def _keep_image_active(self, *a):
        """Snap the active layer back to the image while the editor is on.

        napari binds the section-editing letters (r/e/d, m on Labels) as mode
        shortcuts on the ACTIVE layer; if a Shapes layer (our preview/selection,
        or 'Sections') becomes active — by a layer-list click or a napari
        internal — it would shadow the editor's r/m/d/e. Forcing the image layer
        to stay active keeps those shortcuts out of the keymap chain. The guard
        stops the re-selection from recursing through this same event."""
        if not self._active or self._reasserting or self.gui.image_layer is None:
            return
        try:
            if self.viewer.layers.selection.active is not self.gui.image_layer:
                self._reasserting = True
                self.viewer.layers.selection.active = self.gui.image_layer
        except Exception:
            pass
        finally:
            self._reasserting = False

    def deactivate(self):
        if not self._active:
            return
        self._active = False
        for t in (self._hover_timer, self._cam_timer):
            try:
                t.stop()
            except Exception:
                pass
        if self._cam_connected:
            for ev, fn in ((self.viewer.camera.events.zoom, self._schedule_autoembed),
                           (self.viewer.camera.events.center, self._schedule_autoembed)):
                try:
                    ev.disconnect(fn)
                except Exception:
                    pass
            self._cam_connected = False
        if self._sel_connected:
            try:
                self.viewer.layers.selection.events.active.disconnect(self._keep_image_active)
            except Exception:
                pass
            self._sel_connected = False
        try:
            self.viewer.mouse_move_callbacks.remove(self._on_move)
        except Exception:
            pass
        if self._drag_cb is not None:
            try:
                self.viewer.mouse_drag_callbacks.remove(self._drag_cb)
            except Exception:
                pass
            self._drag_cb = None
        self._dragging = False
        for tgt, k in self._keybinds:
            try:
                tgt.bind_key(k, None, overwrite=True)
            except Exception:
                pass
        self._keybinds = []
        for attr in ("preview_layer", "sel_layer", "frame_layer"):
            lyr = getattr(self, attr, None)
            try:
                if lyr is not None and lyr in self.viewer.layers:
                    self.viewer.layers.remove(lyr)
            except Exception:
                pass
            setattr(self, attr, None)
        self._embed_ready = False
        self._sel_idx = None
        self.commit_target = None        # back to Sections-commit for the next activation
        self._log("napari editor OFF.")
