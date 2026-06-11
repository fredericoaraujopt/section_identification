# Legacy OpenCV Manual Section-Detector — Permanent Reference

> **Status: RETIRED.** This document is the durable record of the OpenCV-window
> SAM-assisted manual editor, written *before* it is deleted from the codebase.
> It is being superseded by the napari in-viewer editor
> (`section_identification/napari_sam_editor.py`, class `NapariSamEditor`).
> Everything needed to fully understand the old tool and to restore it from git
> history is captured here. Code is cited by function name and (at time of
> writing) line number in the source files. Repo state when documented:
> branch `stim-sam2-czi-zen`, files modified working-tree, last commit touching
> the interactive modules: `2ad3bd2`.

---

## 1. Purpose & what it did

The legacy tool was an **OpenCV-window, SAM-1-assisted manual section editor**.
It opened from the GUI button labelled **"Manual detector (OpenCV)"** (Section
"3 · Manual detector" of the control panel). It launched a *separate*
`cv2.namedWindow` (not embedded in napari) titled
`"SAM Interactive (Press ESC to exit)"` in which the user could:

- **hover** the mouse to see a live, ephemeral SAM mask preview under the cursor;
- **left-click** to commit that mask as a permanent section;
- **`r`** to remove a section (either an existing detection outline or a
  user-added mask);
- **`m`** to enter a separate zoomed "fiducials" window and drop calibration
  markers;
- **`e`** (CZI only) to read the current view as a *full-resolution* crop from
  the CZI and re-embed SAM on it, so clicks segment individual sections at real
  resolution; **`b`** to return to the overview;
- zoom/pan the view (scroll wheel, `+`/`-`, `0` reset, arrow keys / right-drag);
- **`d`** to toggle the mask overlay on/off;
- **Esc** to finish, at which point edits are returned to the GUI and persisted.

Internally it ran **SAM 1** via a **quantized ONNX decoder** plus a pre-computed
**image embedding** for fast interactive hover/click. The `e`/`b` full-res path
additionally built a live `SamPredictor` (the torch SAM image encoder) to
re-embed crops on demand.

The implementation lived in two modules:

- `section_identification/interactive.py` (~616 lines) — the editor loop,
  zoom/pan view state, ONNX inference, the `run_sam_interactive` entry point.
- `section_identification/interactive_helpers.py` (~527 lines) — overlay
  composition, hover/click mask processing, fiducials mode, mask deletion,
  state save/load, and the Qt help dialog.

---

## 2. Entry points

### 2.1 `run_manual()` — `section_identification/interface.py` (line ~992)

This is the GUI handler wired to the "Manual detector (OpenCV)" button. Flow:

1. **Guard:** returns with a log warning if no image is loaded
   (`self.overview is None`).
2. **Checkpoint check:** requires a SAM-1 `.pth` checkpoint at
   `self.sam1_checkpoint`. Defaults set in `__init__` (interface.py ~82–83):
   - `self.sam1_checkpoint = str(ckpt_dir / "sam_vit_b_01ec64.pth")`
     (where `ckpt_dir = <repo>/checkpoint`)
   - `self.sam1_model_type = "vit_b"`

   If the file is missing it pops a `QMessageBox`, then a `QFileDialog` to pick a
   `*.pth`/`*.pt`. The picked path sets `sam1_checkpoint` and infers the model
   type from the filename: `vit_h` if `"vit_h" in p`, else `vit_l`, else `vit_b`.
3. **Imports (deferred, in a try/except):**
   ```python
   from section_identification.interactive import run_sam_interactive
   from section_identification.interactive_helpers import display_help
   ```
   On `ImportError` it warns "Manual editor unavailable — Needs onnxruntime +
   segment-anything installed." and returns. (These two imports are exactly the
   wiring that must be removed/restored — see §9.)
4. **Image file for the editor** via `self._image_file_for_interactive()`
   (interface.py ~982): for a **PNG/non-CZI** it returns `self.image_path`
   unchanged; for a **CZI** it writes the in-memory `self.overview` (RGB) to
   `<base>_files/<base>_overview.png` (BGR via `cv2.cvtColor`) and returns that
   PNG path. (SAM in this editor always runs on this overview PNG.)
5. **CZI extras:**
   - `czi_p = self.image_path if czi_io.is_czi(self.image_path) else None`
   - `ref = self.current_polygons_xy() if czi_p else None` — the already-detected
     section polygons (overview px, xy), passed as reference outlines.
   - `geom = self.geom` — the calibration/geometry object that maps overview
     (downsampled) coords ⇄ full-resolution CZI coords (`geom.zoom`,
     `geom.origin_x/origin_y`, `geom.ds_to_full(...)`).
6. **Call:**
   ```python
   new_masks, stored_masks, fiducials = run_sam_interactive(
       img_path, checkpoint=self.sam1_checkpoint, stored_masks=[],
       model_type=self.sam1_model_type, device=device_str(),
       czi_path=czi_p, geom=self.geom, ref_polygons=ref)
   ```
   Note `stored_masks=[]` is passed empty: the GUI never seeds the editor with
   masks; existing detections are passed as `ref_polygons` (reference outlines)
   instead. `device_str()` comes from `section_identification.device`.
7. **Result handling (re-entry into napari layers):**
   - Build `new_polys` by iterating `list(stored_masks) + list(new_masks)`.
     For each mask dict, prefer its `"poly_overview"` key (full-res additions
     carry their polygon already in overview coords); otherwise convert the
     mask's `"segmentation"` raster to a polygon via
     `mask_to_polygon(m["segmentation"])` (from
     `section_identification.export`). Keep polygons with `len(p) >= 3`.
   - `ref` was **mutated in place** by the editor (deletions via `r` removed
     entries), so the surviving detections are read back from it:
     `survivors = ref if ref is not None else self.current_polygons_xy()`.
   - Rebuild the editable layers:
     `self._ensure_edit_layers(list(survivors) + new_polys)`.
   - If `fiducials` were returned and a `fid_layer` exists, set
     `self.fid_layer.data = np.asarray(fiducials, dtype=float)[:, ::-1]`
     (xy → yx for napari).
   - `self.save_project()`.
   - A progress bar is shown (`setRange(0, 0)`) for the duration and hidden in
     `finally`. Errors are logged via `traceback.format_exc()`.

### 2.2 `run_sam_interactive(...)` — `section_identification/interactive.py` (line 253)

**Signature:**
```python
def run_sam_interactive(image_path, checkpoint, stored_masks, model_type="vit_h",
                        device="cpu", czi_path=None, geom=None, ref_polygons=None):
```

**Parameters:**
- `image_path` — path to the input image (the overview PNG for a CZI).
- `checkpoint` — path to the SAM-1 checkpoint (`.pth`).
- `stored_masks` — initially stored masks (the GUI passes `[]`).
- `model_type` — SAM model type; default `"vit_h"` (the GUI passes `"vit_b"`).
- `device` — `"cpu"` or `"cuda"`, used only for embedding/predictor creation.
- `czi_path` / `geom` — when **both** are provided, the `e` key reads the current
  view as a full-resolution crop from the CZI and re-embeds; `b` returns to the
  overview. `full_res_available = bool(czi_path) and (geom is not None)`.
- `ref_polygons` — existing section polygons (overview px) drawn as reference
  outlines. **Normalized in place**: each entry is reshaped to a float32 `(-1,2)`
  array; `ref_polys` aliases the *same list object* the caller passed, so `r`
  deletions propagate back to `run_manual()`'s `ref`.

**Return value (always a 3-tuple):**
```python
return (new_masks + crop_masks), stored_masks, markers
```
- element 0 — overview-space masks added by clicking (`new_masks`) **plus**
  full-resolution additions (`crop_masks`, each carrying a `"poly_overview"`);
- element 1 — `stored_masks` (the surviving seeded masks; empty from the GUI);
- element 2 — `markers`, the fiducial coordinate list `[(x, y), ...]`.

The 3-tuple arity is load-bearing: `demo.ipynb` and `test.py` unpack it as
`new_masks, stored_masks, fiducials = run_sam_interactive(...)`.

### 2.3 The other caller — `section_identification/test.py` (line ~29)

```python
from section_identification.interactive import run_sam_interactive
new_masks, stored_masks, fiducials = run_sam_interactive(
    image2, checkpoint, masks, model_type="vit_h", device="cpu")
```
Here `image2` is a plain PNG, `checkpoint` is `sam_vit_h_4b8939.pth`, and `masks`
is the output of `automatic_identification(...)` — i.e. this caller *does* seed
`stored_masks` with previously detected masks, and uses no CZI/geom/ref_polygons.
This is the simplest non-GUI usage and the canonical example for restore.

---

## 3. Complete control scheme

Read from the main loop in `run_sam_interactive` (interactive.py ~427–602) and
`fiducials()` (interactive_helpers.py ~190).

Key handling uses `cv2.waitKeyEx(1)` → `keyx` (full extended code, keeps arrows)
and `key = keyx & 0xFF` (ASCII byte). Window-close or **Esc (27)** breaks the
loop.

| Input | Effect |
|---|---|
| **mouse move (hover)** | Throttled (`THROTTLE_TIME = 0.05 s`); sets `latest_hover = (ix, iy, 1)` (image-space, positive prompt) → ephemeral SAM preview mask drawn each frame. |
| **left-click** | Sets `latest_click = (ix, iy, 1)` → commits a permanent mask via `process_new_mask`. In `crop` mode the mask is tagged with `poly_overview`. |
| **`r`** (with a live hover) | **Remove.** In `crop` mode: deletes a `crop_masks` entry under the cursor (`exclude_mask`). In `overview` mode: first prefers a reference-detection outline under the cursor — first `r` selects it (drawn red/thick, `pending_ref`), second `r` (still hovering) deletes it from `ref_polys`; if no outline is under the cursor it falls back to `exclude_mask` on `stored_masks`/`new_masks` (which itself opens a "Confirm Exclusion" window awaiting another `r`). |
| **`m`** | Toggle **fiducials mode** → `markers = fiducials(image)`; blocks until `m` is pressed again inside that window. |
| **`d`** | Toggle the mask overlay (`display_on`); when off, the raw image (no overlays/hover) is shown. |
| **scroll wheel** (or ctrl+wheel) | Zoom centred on the cursor (`_zoom_at`, factor 1.25 / 0.8). |
| **`+` / `=`** | Zoom in at last cursor position. |
| **`-` / `_`** | Zoom out at last cursor position. |
| **`0`** | Reset view (`_reset_view`): zoom 1.0, origin (0,0). |
| **arrow keys** | Pan ~15% of the visible crop. Extended codes mapped per-platform in `_ARROW_DIR` (macOS / GTK / Windows). |
| **`w` / `s` / `a` / `z`** | Fallback pan (up/down/left/right) for builds that don't deliver arrow codes. |
| **right-drag**, or **ctrl+left-drag** | Pan the view. |
| **`e`** (CZI + geom only, `overview` mode) | Read the current view as a full-res CZI crop, re-embed, enter `crop` mode. Refuses if crop `max(w,h) > MAX_FULLRES_CROP` (8000 px). |
| **`b`** (`crop` mode) | Return to the overview, restoring the pre-`e` zoom/pan. |
| **Esc (27)** / window close | Finish: save state, destroy window, return the 3-tuple. |

**Fiducials window** (`fiducials()`): a dedicated `cv2.namedWindow`
("Fiducials Mode (Press 'm' to exit)") showing the full image plus a 16×
zoomed inset (`zoom_factor = 16`, `zoom_window_size = 1500`,
`patch_size = 1500//16`). A green cross follows the cursor; **left-click** saves
a marker `(x, y)` in original-image coords (red persistent circle, brief blue
flash); **`m`** exits and returns the `markers` list.

---

## 4. Full data flow

### Coordinate spaces and the zoom/pan view
The OpenCV window always renders a **view**: a crop of the full overlay
(top-left `(view_ox, view_oy)` in image px, size `img/zoom`) resized back to the
original image size (`render_view`). Mouse coordinates from OpenCV are in this
rendered view space and are mapped back to image space by `_win_to_img(mx,my)`
*before* they reach SAM, so all downstream code always receives **image-space**
coords. Module globals hold this state: `view_zoom`, `view_ox`, `view_oy`,
`_img_w`, `_img_h`, plus mouse trackers `latest_click`, `latest_hover`,
`last_event_time`, `latest_mouse_win`, and pan state `_panning`/`_pan_anchor`.
`_clamp_view()` keeps zoom in `[1.0, MAX_ZOOM=40]` and the crop inside the image.

### Mask representation
A mask is a `dict`. Keys produced by this editor:
- `"segmentation"` — a 2-D binary `uint8` raster (image/overview px, or crop px
  in full-res mode). Produced by thresholding the ONNX/predictor output `> 0`
  and squeezing.
- `"color"` — `[0.8, 0.9, 1]` (soft blue) for clicked masks.
- `"overlay"` — a precomputed BGR `uint8` overlay layer (the colored fill plus a
  thick blue contour, BGR `(255,0,0)` width 20) used by `recompose_overlay` to
  redraw quickly without re-running SAM.
- `"poly_overview"` (full-res additions only) — the mask's outer contour mapped
  into **overview** coords as a list of `[x, y]` pairs. This lets the GUI place
  the section without ever needing a full-res-sized raster.

### Overview ↔ full-resolution mapping (the `e`/`b` path)
On `e` (interactive.py ~498–541):
- the view rectangle in overview px is converted to full-res via
  `geom.ds_to_full(view_ox, view_oy)` and `fw, fh = vw/geom.zoom, vh/geom.zoom`;
- the crop is read with `czi_io.read_czi_region(czi_path, fx0, fy0, fw, fh)`;
- a `SamPredictor` is built lazily (`_build_predictor`) and the crop embedded
  (`_embed_image_rgb` → `(1,256,64,64)` float32) — replacing the active
  `embedding`/`image`/`samScale`/`orig_size`;
- two closures are created:
  - `crop_to_ov(cx, cy) = ((rx0 + cx - origin_x) * zoom, (ry0 + cy - origin_y) * zoom)`
  - `ov_to_crop(ox, oy) = (origin_x + ox/zoom - rx0, origin_y + oy/zoom - ry0)`
  where `rx0,ry0 = fx0,fy0` and `zoom,origin_x,origin_y = geom.{zoom,origin_x,origin_y}`.
- `mode = "crop"`. Clicks in crop mode produce a `crop_masks` entry; its contour
  is mapped through `crop_to_ov` into `poly_overview`.

On `b` (~546–563): restores the snapshotted overview state
(`ov_image, ov_embedding, ov_samScale, ov_orig_size`), rebuilds `base_overlay`
from `stored_masks + new_masks`, returns to `mode="overview"` and the saved
`ov_view` zoom/pan.

`_overlay_refs` draws, on a display copy each frame: existing detections in
**yellow** (BGR `(0,255,255)`, width 2), the delete-pending one in **red**
(width 4), and full-res additions in **green** (BGR `(0,200,0)`, width 2,
mapped via `_to_disp` so they stay visible in either mode). `_hit_ref` uses
`cv2.pointPolygonTest` to find the smallest reference outline under the cursor.

### SAM inference per frame
- **Hover:** `process_overlay(overlay, embedding, hover, samScale, orig_size, session)`
  → `prepare_inputs` builds the ONNX feed dict, `run_model` runs the decoder,
  `overlay_mask` blends the predicted mask (alpha 0.4, BGR `(0,114,189)`).
- **Click:** `process_new_mask(base_overlay, embedding, click, samScale, orig_size, session, masks)`
  → same inference, but the result is thresholded, contoured, blended
  permanently into `base_overlay`, and appended to the masks list as a dict.

### Re-entry into the GUI
After Esc, `run_manual` converts the returned masks to polygons (preferring
`poly_overview`, else `mask_to_polygon`), reads the survivors back out of the
mutated `ref` list, and calls `_ensure_edit_layers(survivors + new_polys)` so the
sections appear in the napari "Sections" layer; fiducials go to `fid_layer`
(xy→yx). Everything is then `save_project()`-ed and re-exportable like any other
detection. Detection / save / load / export code paths are untouched by the
editor.

---

## 5. The `interactive_state.pkl` cache format

Written/read by `save_interactive_state` / `load_interactive_state` in
`interactive_helpers.py` (lines 4–23).

**Location:**
```
<image_dir>/<base>_files/<base>_interactive_state.pkl
```
where `<base> = os.path.splitext(image_path)[0]`'s basename. For a CZI this sits
alongside the generated `<base>_overview.png`, the ONNX models
(`<base>_onnx.onnx`, `<base>_onnx_quantized.onnx`), and the embedding
(`<base>_embedding.npy`), all under `<base>_files/`.

**Format:** a pickled `dict` with exactly three keys:
```python
{
    "stored_masks": stored_masks,   # list of mask dicts (the seeded masks)
    "new_masks":    new_masks,      # list of mask dicts (clicked, overview-space)
    "fiducials":    fiducials,      # list of (x, y) marker tuples
}
```

**Save** happens in the `finally` block of `run_sam_interactive` (~611):
`save_interactive_state(image_path, new_masks, stored_masks, markers)`.
Note: **only overview-space masks are pickled** — `crop_masks` (full-res
additions) are *not* saved to the pkl because their crop-sized rasters wouldn't
match the overview on reload; they are still *returned* to the GUI, which stores
their polygons in the project file.

**Load** happens early in `run_sam_interactive` (~298–310): if a pkl exists, it
recomposes `base_overlay` from `stored_masks + new_masks` via
`recompose_overlay`. If that raises (e.g. cached overlays were built for a
different-size image), the cache is discarded with a `[warn] ignoring stale
interactive cache` message and the session starts fresh. This stale-cache guard
was the subject of commit `7e5dd9a` ("Fix manual-editor crash on stale
interactive-state cache").

---

## 6. Every function — one-line descriptions

### `section_identification/interactive.py`
| Function | Description |
|---|---|
| `_clamp_view()` | Keep `view_zoom` in `[1, MAX_ZOOM]` and the crop fully inside the image. |
| `_win_to_img(mx, my)` | Map a window/view coordinate to full-resolution image coords. |
| `_zoom_at(win_x, win_y, factor)` | Zoom by `factor` keeping the image point under the cursor fixed. |
| `_reset_view()` | Reset zoom to 1.0 and pan origin to (0,0). |
| `_wheel_delta(flags)` | Signed scroll-wheel delta, with a fallback for cv2 builds lacking `getMouseWheelDelta`. |
| `_build_predictor(checkpoint, model_type, device)` | Load a torch SAM image encoder once for repeated re-embedding (`SamPredictor`). |
| `_embed_image_rgb(predictor, image_rgb)` | Compute a `(1,256,64,64)` float32 image embedding for an RGB uint8 crop. |
| `render_view(overlay)` | Crop the overlay to the current view and resize back to full size (nearest). |
| `load_image(image_path)` | `cv2.imread` an image; raise `FileNotFoundError` if missing. |
| `compute_sam_scale(image, long_side=1024)` | Return `(samScale, h, w)` so the longest side maps to 1024. |
| `prepare_inputs(embedding, click, samScale, orig_size)` | Build the ONNX feed dict (point_coords/labels, empty mask_input, orig_im_size). |
| `run_model(session, inputs)` | Run the ONNX decoder and return the first output (the mask logits). |
| `overlay_mask(image, mask, alpha=0.4, threshold=0.0, color=(0,114,189))` | Blend a thresholded predicted mask over the image. |
| `mouse_callback(event, x, y, flags, param)` | OpenCV mouse handler: wheel=zoom, right/ctrl-drag=pan, move=hover, l-click=add; maps view→image coords. |
| `run_sam_interactive(...)` | **Main entry point** — exports/quantizes ONNX, builds/loads the embedding, runs the interactive loop, persists state, returns `(new+crop_masks, stored_masks, markers)`. |

Module constants: `LONG_SIDE_LENGTH=1024`, `THROTTLE_TIME=0.05`, `MAX_ZOOM=40.0`,
`MAX_FULLRES_CROP=8000`, `_ARROW_DIR` (per-platform arrow-key code map). Module
globals: `latest_click`, `last_processed_click`, `last_event_time`,
`current_overlay`, `view_zoom`, `view_ox`, `view_oy`, `_img_w`, `_img_h`,
`latest_mouse_win`, `_panning`, `_pan_anchor`.

### `section_identification/interactive_helpers.py`
| Function | Description |
|---|---|
| `save_interactive_state(image_path, new_masks, stored_masks, fiducials)` | Pickle `{stored_masks, new_masks, fiducials}` to `<base>_files/<base>_interactive_state.pkl`. |
| `load_interactive_state(image_path)` | Load that pkl if it exists, else `None`. |
| `recompose_overlay(image, masks, alpha=0.5)` | Blend each mask's cached `"overlay"` over the base image, skipping size-mismatched overlays. |
| `display_masks(image, stored_masks, new_masks, show=True, alpha=0.5)` | Convenience: return overlaid image if `show`, else a copy of the original. |
| `overlay_stored_masks(image, masks, alpha=0.5, random_color=True)` | Build the initial overlay: pastel fill + thick blue contour per mask. |
| `process_overlay(base_overlay, embedding, hover, samScale, orig_size, session)` | Run SAM on a hover prompt and blend the **ephemeral** preview mask. |
| `process_new_mask(base_image, embedding, click, samScale, orig_size, session, new_masks, alpha=1)` | Run SAM on a click, blend a **permanent** mask, append its dict (with `"overlay"`). |
| `fiducials(image)` | Modal fiducials window with a 16× zoom inset; returns saved `(x,y)` markers; exits on `m`. |
| `exclude_mask(image, stored_masks, new_masks, current_mouse, base_overlay)` | Find the mask under the cursor, highlight it red in a "Confirm Exclusion" window, delete on a confirming `r`; returns `(overlay, new_masks, stored_masks)`. |
| `display_help()` | Show a Qt `QTextBrowser` help dialog (controls + tutorial/GitHub links), parented to the napari window; caches the dialog on the function. |
| `process_new_mask_SamPredictor(base_image, click, predictor, new_masks)` | **Unused alternate** path: commit a mask via a torch `SamPredictor` (`multimask_output=True`) instead of ONNX; returns SAM-style mask dict. |

> Note: `interactive.py` imports a name `display_masks` from the helpers (line
> 14) but does not actually use it in the loop; `recompose_overlay`,
> `overlay_stored_masks`, `process_overlay`, `process_new_mask`, `fiducials`,
> `exclude_mask`, `save_/load_interactive_state` are the actively used helpers.

---

## 7. Dependencies & the SAM-1 ONNX/predictor path

**Python packages** (top of `interactive.py` / `interactive_helpers.py`):
- `cv2` (OpenCV) — windows, drawing, contours, image I/O, resize.
- `numpy`.
- `onnxruntime` (`import onnxruntime as ort`) — runs the quantized SAM decoder
  in the interactive loop.
- `segment_anything` (Meta SAM 1) — `sam_model_registry`, `SamPredictor`
  (imported lazily in `_build_predictor` and in `create_embedding.py`).
- `torch` (transitively, for the encoder/embedding).
- `qtpy` + `napari` (only `display_help`, for the Qt help dialog parented to the
  napari window).
- `pycocotools` (imported in helpers; used elsewhere for RLE).

**Companion modules used:**
- `section_identification.onnx_export.install_and_export_sam_onnx(image_path, checkpoint, model_type)`
  — exports + (by default) quantizes a per-image SAM ONNX decoder. Writes
  `<base>_files/<base>_onnx.onnx` and `<base>_files/<base>_onnx_quantized.onnx`;
  returns the quantized path if present (cached on subsequent runs).
- `section_identification.create_embedding.create_embedding_if_needed(image_path, checkpoint, model_type, device)`
  — builds (or reuses) the image embedding via SAM's encoder; writes
  `<base>_files/<base>_embedding.npy` and returns its path. `interactive.py`
  then `np.load`s it.
- `section_identification.czi_io.read_czi_region(...)` — the full-res crop reader
  used by the `e` key.
- `section_identification.device.device_str()` — chooses `cpu`/`cuda` (called by
  the GUI to pass `device`).

**SAM-1 inference path used in the loop (ONNX):**
1. `install_and_export_sam_onnx` → quantized ONNX decoder for this image.
2. `create_embedding_if_needed` → `(1,256,64,64)` encoder embedding (`.npy`),
   `np.load`-ed into `embedding`.
3. `ort.InferenceSession(final_model_path)` → `session`.
4. Per hover/click: `prepare_inputs(embedding, point, samScale, orig_size)` →
   `session.run(None, inputs)` → mask logits → threshold/contour/overlay.

The ONNX input dict keys (from `prepare_inputs`): `image_embeddings`,
`point_coords` (scaled by `samScale`), `point_labels`, `mask_input`
(`zeros (1,1,256,256)`), `has_mask_input` (`zeros (1,)`), `orig_im_size`
(`[H, W]`). `samScale = 1024 / max(H, W)`.

**Full-res `e`/`b` path (torch predictor):** lazily builds a `SamPredictor`
from the same `.pth` checkpoint/`model_type` (`_build_predictor`) and re-embeds
crops (`_embed_image_rgb`). The same ONNX `session` then decodes against the new
embedding (the decoder is image-size-agnostic via `orig_im_size`).

**Default GUI checkpoint:** `checkpoint/sam_vit_b_01ec64.pth`, model type
`vit_b` (interface.py ~82–83). `test.py` uses `sam_vit_h_4b8939.pth` / `vit_h`.

---

## 8. Why it's being retired

It is **superseded by `section_identification/napari_sam_editor.py`
(class `NapariSamEditor`)**, an *in-viewer* editor that runs **SAM 2.1**
(host-adaptive tiny/small/base+ via `build_image_predictor`) directly inside the
napari canvas — no separate OpenCV window.

Key advantages of the napari editor:
- Works **in-viewer** on the GUI's lazy full-res multiscale display: it embeds a
  full-res crop of the *current napari view* automatically when you zoom/pan
  (debounced), so masks are sharp on individual sections at **any zoom** without
  the explicit `e`/`b` overview↔crop dance.
- Bounded encode cost: reads the view at a zoom that keeps the crop ≈
  `ENCODE_PX = 1024` px.
- Edits write straight to the GUI's existing **"Sections"** / **"Fiducials"**
  layers (overview-px data via per-layer `scale = 1/geom.zoom`), so detection,
  save, load, and export are unchanged.
- Handles **both CZI** (`read_czi_region`) **and ordinary images** (slices
  `gui.overview`) — see the PNG+CZI compatibility requirement.

**Napari editor's equivalent controls** (from its module docstring and
`activate()` key binds, napari_sam_editor.py ~442–467):

| Napari editor | Legacy OpenCV equivalent |
|---|---|
| (auto) zoom/pan → auto re-embed view; **`e`** forces re-embed | manual **`e`** crop embed / **`b`** back |
| **hover** → yellow live preview | hover preview |
| **Space** → commit previewed section (left-drag stays as pan) | **left-click** to add |
| **`r`** → select section under cursor (magenta); `r` again removes | **`r`** remove |
| **`m`** → drop a fiducial at cursor | **`m`** fiducials window |
| **`d`** → toggle preview | **`d`** toggle masks |
| toggle button OFF | **Esc** |

It is wired to the **"Manual editor (napari)"** checkable button via
`toggle_manual_napari()` (interface.py ~1047), which lazily constructs
`self._napari_editor = NapariSamEditor(self)` and calls `.toggle()`.

> Doc-vs-label note for future readers: the panel label (Section 3) and an old
> log line describe the napari "add" gesture as a click, but the actual binding
> is **`Space`** (`_commit_preview`); left-click/drag remains pan. If you adjust
> the label, match the real binding.

---

## 9. How to restore the legacy OpenCV editor

When the tool is retired, the following are removed. To reinstate, recover the
two deleted modules from git history and re-add the GUI wiring.

### What gets removed
1. **The two modules themselves are deleted:**
   - `section_identification/interactive.py`
   - `section_identification/interactive_helpers.py`
2. **In `section_identification/interface.py`:**
   - The **`run_manual(self)`** method (~992–1045).
   - The **`_image_file_for_interactive(self)`** helper (~982–990) — only used by
     `run_manual`; the napari editor reads the view itself.
   - The **`btn_manual` button** and its label fragment in Section "3 · Manual
     detector": `self.btn_manual = QPushButton("Manual detector (OpenCV)")`
     (~244) and `man.addWidget(self.btn_manual)` (~245).
   - The **signal connection** `self.btn_manual.clicked.connect(self.run_manual)`
     (~292).
   - The **deferred imports** inside `run_manual`:
     `from section_identification.interactive import run_sam_interactive` and
     `from section_identification.interactive_helpers import display_help` (~1006–1007).
   - (Keep `btn_manual_napari`, `toggle_manual_napari`, and the
     `napari_sam_editor` import — those are the replacement.)
3. **Non-GUI callers** that import `run_sam_interactive` will break if the module
   is gone: `section_identification/test.py` (line ~29) and `demo.ipynb`. These
   are examples/notebooks, not part of the GUI, but note them.

### What stays usable as-is
- `onnx_export.py`, `create_embedding.py`, and the on-disk artifacts
  (`<base>_files/*_onnx*.onnx`, `*_embedding.npy`, `*_interactive_state.pkl`) are
  independent and remain valid for a restored editor.
- The mask→polygon path (`section_identification.export.mask_to_polygon`),
  `czi_io.read_czi_region`, `geom`, and `device_str()` are all still present.

### Restore procedure
1. **Recover the deleted modules** from the commit before deletion. The last
   commit that touched both modules at documentation time was `2ad3bd2`; restore
   from whatever commit is immediately prior to their deletion, e.g.:
   ```bash
   # find the deletion commit
   git log --oneline -- section_identification/interactive.py
   # restore both files from the commit just BEFORE deletion (<sha>):
   git checkout <sha> -- section_identification/interactive.py \
                          section_identification/interactive_helpers.py
   ```
   (Or `git show <sha>:section_identification/interactive.py > ...` to inspect
   first.)
2. **Re-add the GUI wiring** in `interface.py`:
   - Re-add `_image_file_for_interactive` and `run_manual` (copy from history or
     from §2.1 above).
   - In the Section-3 UI builder, re-add:
     ```python
     self.btn_manual = QPushButton("Manual detector (OpenCV)")
     man.addWidget(self.btn_manual)
     ```
   - In the signal-wiring block, re-add:
     ```python
     self.btn_manual.clicked.connect(self.run_manual)
     ```
   - The deferred imports of `run_sam_interactive` and `display_help` live inside
     `run_manual` itself, so they come back with that method.
3. **Verify deps** are installed: `onnxruntime`, `segment-anything`, plus a
   SAM-1 `.pth` checkpoint at `checkpoint/sam_vit_b_01ec64.pth` (or pick another
   via the dialog).
4. **Smoke test** with the non-GUI example in `test.py` (§2.3) on a PNG, then via
   the GUI button on a CZI to exercise the `e`/`b` full-res path.

---

## Appendix — exact line references (at documentation time)

- `interactive.py`: `run_sam_interactive` def @ **253**; key loop @ **427–602**;
  `e` handler @ **498–544**; `b` handler @ **546–563**; save in `finally` @
  **611**; return @ **616**. Constants @ **21–54**.
- `interactive_helpers.py`: `save_interactive_state` @ **4**; `load` @ **16**;
  `recompose_overlay` @ **24**; `overlay_stored_masks` @ **62**;
  `process_overlay` @ **118**; `process_new_mask` @ **139**; `fiducials` @
  **190**; `exclude_mask` @ **308**; `display_help` @ **405**;
  `process_new_mask_SamPredictor` @ **460**.
- `interface.py`: `btn_manual` @ **244–245**; `btn_manual.clicked.connect` @
  **292**; `_image_file_for_interactive` @ **982**; `run_manual` @ **992**;
  `run_sam_interactive(...)` call @ **1022–1025**; `toggle_manual_napari` @
  **1047**; SAM-1 checkpoint defaults @ **82–83**.
- `test.py`: `run_sam_interactive` import/call @ **29–31**.
- `napari_sam_editor.py`: `NapariSamEditor` class @ **52**; controls docstring @
  **22–30**; key binds @ **442–446**; activate banner @ **464–467**.
