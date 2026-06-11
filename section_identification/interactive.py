import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path


# Import your helper functions from package.
from section_identification.interactive_helpers import overlay_stored_masks
from section_identification.interactive_helpers import process_overlay
from section_identification.interactive_helpers import fiducials
from section_identification.interactive_helpers import exclude_mask
from section_identification.interactive_helpers import display_masks
from section_identification.interactive_helpers import load_interactive_state, save_interactive_state, recompose_overlay

from section_identification.onnx_export import install_and_export_sam_onnx
from section_identification.create_embedding import create_embedding_if_needed

# Constants
LONG_SIDE_LENGTH = 1024         # SAM requires the longest side to be 1024
THROTTLE_TIME = 0.05            # Minimum time (in seconds) between model inferences

# Global variables for interactive state
latest_click = None             # (x, y, clickType) updated from mouse events
last_processed_click = None     # Last processed click (to throttle)
last_event_time = 0             # Timestamp of last accepted mouse event
current_overlay = None          # The current overlay image to display

# --- Zoom / pan view state -------------------------------------------------
# The window always displays a rendered "view": a crop of the full-resolution
# overlay (top-left = (view_ox, view_oy) in image px, size = img/zoom) resized
# back up to the original image size. Mouse coords reported by OpenCV are in
# this rendered view's space and are mapped back to image space before they
# ever reach SAM, so all downstream code keeps receiving image-space coords.
view_zoom = 1.0                 # >= 1.0 ; 1.0 = whole image visible
view_ox = 0.0                   # crop top-left x in image px
view_oy = 0.0                   # crop top-left y in image px
_img_w = 1                      # current image width  (set in run_sam_interactive)
_img_h = 1                      # current image height
latest_mouse_win = (0, 0)       # last cursor position in window/view coords
_panning = False                # right-drag / ctrl-drag pan in progress
_pan_anchor = None              # (win_x, win_y, view_ox, view_oy) at pan start
MAX_ZOOM = 40.0
MAX_FULLRES_CROP = 8000         # largest full-res crop (px) we re-embed on 'e'
                                # (smaller / more zoomed-in = sharper to SAM)

# Arrow-key panning. cv2.waitKeyEx returns platform-specific extended codes for
# the arrows (and & 0xFF throws them away), so map every known code set.
_ARROW_DIR = {
    63232: "up", 63233: "down", 63234: "left", 63235: "right",          # macOS
    65362: "up", 65364: "down", 65361: "left", 65363: "right",          # GTK
    2490368: "up", 2621440: "down", 2424832: "left", 2555904: "right",  # Windows
}


def _clamp_view():
    """Keep zoom in range and the crop fully inside the image."""
    global view_zoom, view_ox, view_oy
    view_zoom = max(1.0, min(view_zoom, MAX_ZOOM))
    crop_w = _img_w / view_zoom
    crop_h = _img_h / view_zoom
    view_ox = max(0.0, min(view_ox, _img_w - crop_w))
    view_oy = max(0.0, min(view_oy, _img_h - crop_h))


def _win_to_img(mx, my):
    """Map a window/view coordinate to full-resolution image coordinates."""
    ix = int(round(view_ox + mx / view_zoom))
    iy = int(round(view_oy + my / view_zoom))
    ix = max(0, min(ix, _img_w - 1))
    iy = max(0, min(iy, _img_h - 1))
    return ix, iy


def _zoom_at(win_x, win_y, factor):
    """Zoom by `factor` while keeping the image point under (win_x, win_y) fixed."""
    global view_zoom, view_ox, view_oy
    # image point currently under the cursor
    ix = view_ox + win_x / view_zoom
    iy = view_oy + win_y / view_zoom
    view_zoom = max(1.0, min(view_zoom * factor, MAX_ZOOM))
    # solve so that the same image point stays under the cursor after zoom
    view_ox = ix - win_x / view_zoom
    view_oy = iy - win_y / view_zoom
    _clamp_view()


def _reset_view():
    global view_zoom, view_ox, view_oy
    view_zoom, view_ox, view_oy = 1.0, 0.0, 0.0


def _wheel_delta(flags):
    """Signed scroll-wheel delta. cv2.getMouseWheelDelta is missing in some
    builds (e.g. 4.13), so fall back to the high signed word of `flags`."""
    if hasattr(cv2, "getMouseWheelDelta"):
        return cv2.getMouseWheelDelta(flags)
    d = (flags >> 16) & 0xFFFF
    return d - 0x10000 if d >= 0x8000 else d


# --- Full-resolution re-embedding (used by the 'e' = embed-current-view key) -
# The overview embedding is computed once on the downsampled overview, so SAM
# only ever "sees" a section at overview scale (a few px). To segment an
# individual section we read the current view as a FULL-resolution crop from the
# CZI and recompute the embedding on it, then feed that to the same ONNX
# decoder (which is image-size-agnostic via orig_im_size).
def _build_predictor(checkpoint, model_type, device):
    """Load a SAM image encoder once and keep it for repeated re-embedding."""
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    return SamPredictor(sam)


def _embed_image_rgb(predictor, image_rgb):
    """Compute a (1, 256, 64, 64) image embedding for an RGB uint8 crop."""
    predictor.set_image(image_rgb)
    return predictor.get_image_embedding().cpu().numpy().astype(np.float32)


def render_view(overlay):
    """Crop the overlay to the current view and resize it back to full size."""
    if view_zoom <= 1.0 and view_ox == 0.0 and view_oy == 0.0:
        return overlay
    h, w = overlay.shape[:2]
    crop_w = max(1, int(round(w / view_zoom)))
    crop_h = max(1, int(round(h / view_zoom)))
    ox = max(0, min(int(round(view_ox)), w - crop_w))
    oy = max(0, min(int(round(view_oy)), h - crop_h))
    crop = overlay[oy:oy + crop_h, ox:ox + crop_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)

def load_image(image_path):
    """Load an image using OpenCV."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image

def compute_sam_scale(image, long_side=LONG_SIDE_LENGTH):
    """Compute scaling factor so that the longest side equals long_side.
       Returns (samScale, height, width)."""
    h, w = image.shape[:2]
    samScale = long_side / max(h, w)
    return samScale, h, w

def prepare_inputs(embedding, click, samScale, orig_size):
    """
    Prepare ONNX model inputs.
      - embedding: pre-computed image embedding (np.array)
      - click: (x, y, clickType)
      - samScale: scaling factor computed from image dimensions
      - orig_size: (height, width) of original image
    Returns a dictionary of inputs.
    """
    x, y, clickType = click
    point_coords = np.array([[[x * samScale, y * samScale]]], dtype=np.float32)
    point_labels = np.array([[clickType]], dtype=np.float32)
    mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    has_mask_input = np.zeros((1,), dtype=np.float32)
    orig_im_size = np.array([orig_size[0], orig_size[1]], dtype=np.float32)
    inputs = {
        "image_embeddings": embedding.astype(np.float32),
        "point_coords": point_coords,
        "point_labels": point_labels,
        "mask_input": mask_input,
        "has_mask_input": has_mask_input,
        "orig_im_size": orig_im_size
    }
    return inputs

def run_model(session, inputs):
    """Run the ONNX model and return the predicted mask (first output)."""
    outputs = session.run(None, inputs)
    # For debugging, you might uncomment the prints below:
    """
    print("Model output names:")
    for i, out in enumerate(outputs):
        print(f"Output[{i}]: shape = {out.shape}, dtype = {out.dtype}")
    """
    return outputs[0]

def overlay_mask(image, mask, alpha=0.4, threshold=0.0, color=(0, 114, 189)):
    """
    Overlay the predicted mask on the image.
      - image: original BGR image (numpy array)
      - mask: predicted mask from ONNX model (expected shape: [1, 1, H, W])
      - threshold: threshold value (default 0.0, as recommended)
    Returns the blended overlay.
    """
    # Remove batch and channel dimensions
    mask = np.squeeze(mask)  # now shape (H, W)
    binary_mask = (mask > threshold).astype(np.uint8)
    overlay = image.copy()
    overlay[binary_mask == 1] = color
    output = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return output

def mouse_callback(event, x, y, flags, param):
    """OpenCV mouse callback.

    Coordinates arrive in the rendered view's space; we map them back to image
    space via the zoom/pan state so SAM always receives image-space coords.
      - mouse wheel (or ctrl+wheel) -> zoom centred on the cursor
      - right-drag, or ctrl+left-drag -> pan
      - left click -> add a permanent mask
      - mouse move -> ephemeral hover preview (throttled)
    """
    global latest_hover, latest_click, last_event_time, latest_mouse_win
    global _panning, _pan_anchor, view_ox, view_oy
    current_time = time.time()
    latest_mouse_win = (x, y)
    ctrl = bool(flags & cv2.EVENT_FLAG_CTRLKEY)

    # --- Zoom on the scroll wheel (works with or without ctrl held) ---
    if event == cv2.EVENT_MOUSEWHEEL:
        delta = _wheel_delta(flags)
        if delta != 0:
            _zoom_at(x, y, 1.25 if delta > 0 else 1 / 1.25)
        return

    # --- Start a pan: right button, or ctrl+left button ---
    if event == cv2.EVENT_RBUTTONDOWN or (event == cv2.EVENT_LBUTTONDOWN and ctrl):
        _panning = True
        _pan_anchor = (x, y, view_ox, view_oy)
        return
    if event in (cv2.EVENT_RBUTTONUP, cv2.EVENT_LBUTTONUP):
        if _panning:
            _panning = False
            _pan_anchor = None
            return

    # --- Pan in progress: translate the view, no hover/segmentation ---
    if _panning and event == cv2.EVENT_MOUSEMOVE and _pan_anchor is not None:
        ax, ay, aox, aoy = _pan_anchor
        view_ox = aox - (x - ax) / view_zoom
        view_oy = aoy - (y - ay) / view_zoom
        _clamp_view()
        return

    if event == cv2.EVENT_MOUSEMOVE:
        if current_time - last_event_time > THROTTLE_TIME:
            ix, iy = _win_to_img(x, y)
            latest_hover = (ix, iy, 1)  # all moves treated as positive prompts
            last_event_time = current_time
    elif event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = _win_to_img(x, y)
        latest_click = (ix, iy, 1)  # image-space click for a permanent mask
    

def run_sam_interactive(image_path, checkpoint, stored_masks, model_type="vit_h", device="cpu",
                        czi_path=None, geom=None, ref_polygons=None):
    """
    Run interactive SAM segmentation for a given image.
      - image_path: path to the input image (the overview PNG for a CZI).
      - checkpoint: path to the SAM model checkpoint (.pth)
      - stored_masks: initially stored masks.
      - model_type: SAM model type, default "vit_h"
      - device: device to run the embedding creation ("cpu" or "cuda")
      - czi_path / geom: when both are given, the 'e' key reads the current view
        as a FULL-resolution crop from the CZI and re-embeds on it, so clicks
        segment individual sections at real resolution; 'b' returns to overview.
      - ref_polygons: existing section polygons (overview px) drawn as reference
        outlines so you can see what was already detected while adding more.
    The function exports a quantized ONNX model for this image, creates the image embedding
    (if needed), and launches an interactive OpenCV window. The loop stops when ESC is pressed.
    """
    from section_identification.interactive_helpers import process_new_mask
    global latest_click, latest_hover, last_processed_click, current_overlay
    global _img_w, _img_h, view_ox, view_oy, view_zoom

    full_res_available = bool(czi_path) and (geom is not None)
    # Normalize the reference outlines IN PLACE so 'r' deletions propagate back
    # to the caller: run_manual reads the surviving list after we return (the
    # return arity stays 3-tuple, which demo.ipynb relies on).
    if ref_polygons is None:
        ref_polygons = []
    for _i in range(len(ref_polygons)):
        ref_polygons[_i] = np.asarray(ref_polygons[_i], dtype=np.float32).reshape(-1, 2)
    ref_polys = ref_polygons        # SAME list object as the caller's
    pending_ref = None              # index of a reference outline awaiting delete-confirm
    mode = "overview"               # or "crop" (full-resolution region)
    predictor = None                # lazily built SAM encoder for re-embedding
    crop_masks = []                 # masks added in full-res mode (carry poly_overview)
    crop_to_ov = ov_to_crop = None  # coord transforms, set when entering crop mode
    ov_view = None                  # (zoom, ox, oy) of the overview view saved on 'e'

    # Step 1: Export and quantize the ONNX model for this image.
    image_path = Path(image_path)

    # Try to load prior interactive state. Discard it if it was saved for a
    # different-size image (its cached overlays won't match -> cv2 size error).
    cache_loaded = False
    new_masks = []
    markers = []
    state = load_interactive_state(image_path)
    if state is not None:
        try:
            cand = state["stored_masks"] + state["new_masks"]
            base_overlay = recompose_overlay(load_image(image_path), cand, alpha=0.5)
            stored_masks = state["stored_masks"]
            new_masks = state["new_masks"]
            markers = state["fiducials"]
            cache_loaded = True
        except Exception as e:
            print(f"[warn] ignoring stale interactive cache ({e}); starting fresh.")
            cache_loaded = False
            new_masks, markers = [], []

    final_model_path = install_and_export_sam_onnx(
        image_path=image_path,
        checkpoint=checkpoint,
        model_type=model_type,
        )

    # Step 2: Create or load the embedding for the image.
    embedding_file = create_embedding_if_needed(
        image_path=str(image_path),
        checkpoint=checkpoint,
        model_type=model_type,
        device=device
    )
    embedding = np.load(embedding_file)

    # Step 3: Load the image and compute scaling. Load stored masks.
    image = load_image(image_path)
    samScale, h, w = compute_sam_scale(image)
    orig_size = (h, w)
    _img_w, _img_h = w, h
    _reset_view()
    print(f"Image size: {w}x{h} (width x height), samScale: {samScale:.3f}")

    if not cache_loaded:
        # original overlay + cache overlays for first run
        base_overlay = overlay_stored_masks(image, stored_masks)
        for mask_details in stored_masks:
            segmentation = mask_details['segmentation']
            binary_mask = (segmentation > 0).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_color_overlay = np.zeros_like(image, dtype=np.float32)
            color = np.array([0.8, 0.9, 1])
            for c in range(3):
                mask_color_overlay[:, :, c] = binary_mask * color[c]
            mask_color_overlay = (mask_color_overlay * 255).astype(np.uint8)
            cv2.drawContours(mask_color_overlay, contours, -1, (255, 0, 0), 15)
            mask_details["overlay"] = mask_color_overlay
        # After caching, recompose overlay from cache
        base_overlay = recompose_overlay(image, stored_masks, alpha=0.5)
        new_masks = []
        markers = []

    # Snapshot the overview working state so 'b' can restore it after a
    # full-resolution crop session.
    ov_image, ov_embedding = image, embedding
    ov_samScale, ov_orig_size = samScale, orig_size

    def _to_disp(p):
        """Map an overview-coord polygon into the current display's coords."""
        if mode == "crop" and ov_to_crop is not None:
            qx, qy = ov_to_crop(p[:, 0], p[:, 1])
            return np.column_stack([qx, qy])
        return p

    def _overlay_refs(disp):
        """Draw outlines on a display copy: yellow = an existing detection
        (red+thick = selected for deletion via 'r'); green = a section you've
        added at full resolution (so it stays visible back in the overview)."""
        for i, p in enumerate(ref_polys):
            q = _to_disp(p)
            if mode == "overview" and i == pending_ref:
                cv2.polylines(disp, [q.astype(np.int32)], True, (0, 0, 255), 4)
            else:
                cv2.polylines(disp, [q.astype(np.int32)], True, (0, 255, 255), 2)
        for m in crop_masks:
            po = m.get("poly_overview")
            if po is None:
                continue
            q = _to_disp(np.asarray(po, dtype=np.float32))
            cv2.polylines(disp, [q.astype(np.int32)], True, (0, 200, 0), 2)
        return disp

    def _hit_ref(hover):
        """Index of the smallest reference outline under the cursor, else None.
        Uses point-in-polygon so no full-size masks are needed for the 842."""
        if hover is None or not ref_polys:
            return None
        pt = (float(hover[0]), float(hover[1]))
        best, best_area = None, None
        for i, p in enumerate(ref_polys):
            cnt = p.reshape(-1, 1, 2).astype(np.float32)
            if cv2.pointPolygonTest(cnt, pt, False) >= 0:
                a = cv2.contourArea(cnt)
                if best is None or a < best_area:
                    best, best_area = i, a
        return best

    # Step 4: Create an ONNX runtime session.
    try:
        session = ort.InferenceSession(str(final_model_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model from {final_model_path}: {e}")

    # Step 5: Set up the interactive OpenCV window.
    window_name = "SAM Interactive (Press ESC to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    print("[Info] Starting interactive segmentation. Move the mouse over the image to update mask.")
    print("[Controls] scroll/+/- = zoom at cursor | arrow keys (or right-drag) = pan | "
          "0 = reset view | hover = preview | click = add | "
          "r = remove (hover a detection, r then r) | d = toggle masks | "
          "m = fiducials | ESC = exit")
    if full_res_available:
        print("[Full-res] Zoom/pan to a region, then press 'e' to read it at FULL "
              "resolution from the CZI and re-embed (clicks then segment individual "
              "sections); press 'b' to return to the overview.")
    elif czi_path or geom is not None:
        print("[Full-res] Unavailable (need both the CZI path and its geometry).")

    # Reset global mouse event trackers
    latest_click = None
    latest_hover = None
    last_processed_click = None
    display_on = True

    try:
        while True:
            keyx = cv2.waitKeyEx(1)   # full code (keeps arrow keys)
            key = keyx & 0xFF         # ASCII byte for the ordinary-key handlers
            if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            if key == ord('m'):
                # Suspend segmentation and launch fiducials mode.
                markers = fiducials(image)
                print("Fiducial markers collected:", markers)
                # After fiducials mode, segmentation resumes with its previous state.

            if key == ord('r') and latest_hover is not None:
                if mode == "crop":
                    base_overlay, crop_masks, _ = exclude_mask(
                        image, [], crop_masks, latest_hover, base_overlay)
                else:
                    # Prefer deleting a loaded detection outline under the cursor;
                    # first 'r' selects it (red), second 'r' (still hovering it)
                    # deletes. If none is under the cursor, fall back to removing
                    # a mask the user added by clicking.
                    idx = _hit_ref(latest_hover)
                    if idx is not None and idx == pending_ref:
                        del ref_polys[idx]
                        pending_ref = None
                        print(f"[refs] deleted detection outline; {len(ref_polys)} remain.")
                    elif idx is not None:
                        pending_ref = idx
                        print("[refs] detection selected — press 'r' again "
                              "(while still hovering it) to delete.")
                    else:
                        pending_ref = None
                        base_overlay, new_masks, stored_masks = exclude_mask(
                            image, stored_masks, new_masks, latest_hover, base_overlay)

            if key == ord('d'):
                display_on = not display_on

            # --- Keyboard zoom (reliable when the trackpad wheel event is not
            #     delivered by the macOS HighGUI backend) ---
            if key in (ord('+'), ord('=')):
                mx, my = latest_mouse_win
                _zoom_at(mx, my, 1.25)
            elif key in (ord('-'), ord('_')):
                mx, my = latest_mouse_win
                _zoom_at(mx, my, 1 / 1.25)
            elif key == ord('0'):
                _reset_view()

            # --- Pan with the arrow keys (preferred), or w/a/s/z as a fallback
            #     for builds that don't deliver arrow-key codes. ---
            pan_dir = _ARROW_DIR.get(keyx)
            if pan_dir is None:
                pan_dir = {ord('w'): "up", ord('s'): "down",
                           ord('a'): "left", ord('z'): "right"}.get(key)
            if pan_dir is not None:
                step_x = (_img_w / view_zoom) * 0.15  # ~15% of the visible crop
                step_y = (_img_h / view_zoom) * 0.15
                if pan_dir == "up":
                    view_oy -= step_y
                elif pan_dir == "down":
                    view_oy += step_y
                elif pan_dir == "left":
                    view_ox -= step_x
                elif pan_dir == "right":
                    view_ox += step_x
                _clamp_view()

            # --- 'e': read the current view as a FULL-resolution crop from the
            #     CZI and re-embed, so clicks segment individual sections. ---
            if key == ord('e') and mode == "overview" and full_res_available:
                vw, vh = _img_w / view_zoom, _img_h / view_zoom
                fx0, fy0 = geom.ds_to_full(view_ox, view_oy)
                fx0, fy0 = float(np.asarray(fx0)), float(np.asarray(fy0))
                fw, fh = vw / geom.zoom, vh / geom.zoom
                if max(fw, fh) > MAX_FULLRES_CROP:
                    print(f"[Full-res] Region {int(fw)}x{int(fh)} px too large to "
                          f"re-embed; zoom in more (max {MAX_FULLRES_CROP}).")
                else:
                    try:
                        from section_identification import czi_io
                        print(f"[Full-res] Reading {int(fw)}x{int(fh)} px crop @ full "
                              "resolution and re-embedding…")
                        crop_rgb = czi_io.read_czi_region(
                            czi_path, int(round(fx0)), int(round(fy0)),
                            int(round(fw)), int(round(fh)))
                        if predictor is None:
                            predictor = _build_predictor(checkpoint, model_type, device)
                        embedding = _embed_image_rgb(predictor, crop_rgb)
                        image = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                        _img_w, _img_h = image.shape[1], image.shape[0]
                        samScale, _hc, _wc = compute_sam_scale(image)
                        orig_size = (_hc, _wc)
                        base_overlay = image.copy()
                        z, oxg, oyg = geom.zoom, geom.origin_x, geom.origin_y
                        rx0, ry0 = fx0, fy0

                        def crop_to_ov(cx, cy, rx0=rx0, ry0=ry0, z=z, oxg=oxg, oyg=oyg):
                            return ((rx0 + np.asarray(cx, float) - oxg) * z,
                                    (ry0 + np.asarray(cy, float) - oyg) * z)

                        def ov_to_crop(ox, oy, rx0=rx0, ry0=ry0, z=z, oxg=oxg, oyg=oyg):
                            return (oxg + np.asarray(ox, float) / z - rx0,
                                    oyg + np.asarray(oy, float) / z - ry0)

                        ov_view = (view_zoom, view_ox, view_oy)  # restore on 'b'
                        mode = "crop"
                        pending_ref = None
                        _reset_view()
                        latest_hover = latest_click = last_processed_click = None
                        print(f"[Full-res] Full-resolution view ({_wc}x{_hc}); samScale "
                              f"{samScale:.3f}. Click sections to add; 'b' = overview.")
                    except Exception as e:
                        print(f"[Full-res] failed: {e}")
            elif key == ord('e') and mode == "crop":
                print("[Full-res] Already in full-resolution view; press 'b' first.")

            # --- 'b': return from a full-resolution crop to the overview. ---
            if key == ord('b') and mode == "crop":
                image, embedding = ov_image, ov_embedding
                samScale, orig_size = ov_samScale, ov_orig_size
                _img_w, _img_h = image.shape[1], image.shape[0]
                base_overlay = recompose_overlay(image, stored_masks + new_masks, alpha=0.5)
                crop_to_ov = ov_to_crop = None
                mode = "overview"
                pending_ref = None
                # Stay at the zoom/pan we had before 'e' so you can nudge to the
                # next section and press 'e' again, instead of zooming out fully.
                if ov_view is not None:
                    view_zoom, view_ox, view_oy = ov_view
                    _clamp_view()
                else:
                    _reset_view()
                latest_hover = latest_click = last_processed_click = None
                print(f"[Full-res] Back to overview ({len(crop_masks)} full-res "
                      "section(s) added so far).")

            # Set overlay_to_display based on display_on flag
            if display_on:
                overlay_to_display = base_overlay.copy()
            else:
                overlay_to_display = image.copy()

            # --- Dynamic Hover: Create an ephemeral mask using latest_hover.
            if display_on and latest_hover is not None:
                overlay_to_display = process_overlay(overlay_to_display, embedding, latest_hover, samScale, orig_size, session)

            # --- Permanent Click: Process click events only if new.
            if latest_click is not None and latest_click != last_processed_click:
                if mode == "crop":
                    base_overlay, crop_masks = process_new_mask(
                        base_overlay, embedding, latest_click, samScale, orig_size,
                        session, crop_masks)
                    # Tag the new mask with its polygon in OVERVIEW coords so the
                    # GUI can place it correctly without a crop-sized mask.
                    if crop_masks and crop_to_ov is not None:
                        seg = np.squeeze(crop_masks[-1]["segmentation"]).astype(np.uint8)
                        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts:
                            c = max(cnts, key=cv2.contourArea).reshape(-1, 2)
                            ovx, ovy = crop_to_ov(c[:, 0], c[:, 1])
                            crop_masks[-1]["poly_overview"] = np.column_stack([ovx, ovy]).tolist()
                else:
                    base_overlay, new_masks = process_new_mask(
                        base_overlay, embedding, latest_click, samScale, orig_size,
                        session, new_masks)
                last_processed_click = latest_click
                # Refresh the display overlay with the new permanent mask.
                if display_on:
                    overlay_to_display = base_overlay.copy()
                else:
                    overlay_to_display = image.copy()

            overlay_to_display = _overlay_refs(overlay_to_display)
            cv2.imshow(window_name, render_view(overlay_to_display))

    except Exception as e:
        print("An error occurred during the interactive loop:", e)
    finally:
        # Persist interactive edits. Only the overview-space masks are pickled
        # (crop_masks carry crop-sized arrays that would not match the overview
        # on reload); the full-res additions are still returned to the GUI,
        # which stores their polygons in the project file.
        save_interactive_state(image_path, new_masks, stored_masks, markers)
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)
        print(f"[Info] Exiting interactive segmentation "
              f"({len(new_masks)} overview + {len(crop_masks)} full-res added).")
        return (new_masks + crop_masks), stored_masks, markers