import os
import time
import threading
from pathlib import Path

import numpy as np
import cv2
import ipycanvas
import ipyevents
import ipywidgets as widgets
from IPython.display import display

import torch
from segment_anything import sam_model_registry, SamPredictor

##############################################################################
# Helper: create_embedding_if_needed
##############################################################################

def create_embedding_if_needed(image_path, checkpoint, model_type="vit_h", device="cuda"):
    """
    If a .npy embedding doesn't already exist next to image_path, create one using SAM's
    image encoder (the heavy part). We'll store the result as <image_path>_embedding.npy
    so we can re-run prompts quickly.
    """
    embedding_path = f"{os.path.splitext(str(image_path))[0]}_embedding.npy"
    if os.path.exists(embedding_path):
        print(f"[Info] Embedding already exists: {embedding_path}")
        return embedding_path

    print(f"[Info] Creating embedding for {image_path} with checkpoint {checkpoint}...")
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"[Error] Could not read image from path: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)
    embedding = predictor.get_image_embedding().cpu().numpy()
    np.save(embedding_path, embedding)
    print(f"[Info] Created embedding file: {embedding_path}")
    return embedding_path

##############################################################################
# Helper: run_sam_point_prompt
##############################################################################

def run_sam_point_prompt(
    image_shape,
    embedding_path,
    point_x,
    point_y,
    device="cuda",
    model_type="vit_h",
    checkpoint=None,
    multimask_output=True,
):
    """
    Given a single point (x,y) in image coordinates, run the prompt-based SAM
    inference using a pre-computed embedding. Returns the best mask (or first mask
    if only one) as a binary array, plus an area/bbox for convenience.
    """
    # Load the embedding
    if not os.path.isfile(embedding_path):
        print("[Error] Missing embedding file:", embedding_path)
        return None

    embedding = np.load(embedding_path)
    # Re-init the prompt encoder + mask decoder
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    # We set the image embedding directly, skipping the heavy image encoder
    # The embedding is (1, embed_dim, H/patch, W/patch), e.g. (1, 256, 64, 64)
    # original_image_size is the actual (H, W) of the original image
    predictor.set_torch_image_embedding(
        torch_embedding=torch.tensor(embedding, device=device),
        original_image_size=image_shape[:2]
    )

    # Transform the point to the scaled coordinates used by SAM
    input_point = np.array([[point_x, point_y]], dtype=np.float32)
    input_label = np.array([1], dtype=np.int32)  # 1 => positive

    # We must transform the point from original image coords => encoder coords
    transformed_point = predictor.transform.apply_coords(
        coords=input_point,
        shape=image_shape[:2]
    )

    # Predict up to 3 masks, or 1 if multimask_output=False
    masks, scores, logits = predictor.predict(
        point_coords=transformed_point,
        point_labels=input_label,
        box=None,
        multimask_output=multimask_output
    )
    # Pick the highest-scoring mask
    best_idx = np.argmax(scores)
    chosen_mask = masks[best_idx]

    # Compute bounding box + area
    y_idx, x_idx = np.where(chosen_mask)
    if len(y_idx) == 0:
        return None
    x_min, x_max = x_idx.min(), x_idx.max()
    y_min, y_max = y_idx.min(), y_idx.max()
    bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
    area = float(chosen_mask.sum())

    return {
        "bbox": bbox,
        "area": area,
        "binary_mask": chosen_mask
    }

##############################################################################
# Main: manual_correction
##############################################################################

def manual_correction(
    image_path,
    generated_masks,
    checkpoint=None,
    model_type="vit_h",
    device="cuda",
    overlay_opacity=0.5,
    wait_time_ms=200
):
    """
    A Python notebook function that partially replicates the 'move your cursor to see a mask'
    approach from the SAM minimal web demo. We'll open an ipycanvas showing the image, track
    mouse moves, run a point-based prompt for each mouse position after the user 'pauses'
    for wait_time_ms, then overlay the predicted mask on the canvas. If the user clicks,
    we finalize that mask into an 'accepted' list. The user can also press keys to remove
    or re-accept the last mask.

    Args:
        image_path (str or Path): Path to the image file.
        generated_masks (list): List of pre-existing masks from your
                                `automatic_identification(...)`.
        checkpoint (str): Path to a SAM .pth checkpoint. If provided, we can re-run
                          point-based prompts using the embedding.
        model_type (str): E.g. 'vit_h'. Must match the checkpoint.
        device (str): 'cuda' or 'cpu'.
        overlay_opacity (float): Opacity of the mask overlay, e.g. 0.5 = 50%.
        wait_time_ms (int): How long to wait (in ms) after the last mouse move
                            before running the prompt-based inference.

    Returns:
        finalize(): a function that returns the final list of accepted masks (both
                    existing + newly added).
    """
    # -----------------------------------------------------
    # 0) Ensure embedding if checkpoint is provided
    # -----------------------------------------------------
    embedding_path = None
    if checkpoint is not None:
        embedding_path = create_embedding_if_needed(image_path, checkpoint, model_type, device)
    else:
        # If no checkpoint, we can't re-run prompts, but we can still let the user see existing masks
        # and accept/reject them in a simpler interface
        possible_path = f"{os.path.splitext(str(image_path))[0]}_embedding.npy"
        if os.path.exists(possible_path):
            embedding_path = possible_path
        else:
            print("[Warning] No checkpoint or embedding file found => cannot add new masks from pointer.")

    # -----------------------------------------------------
    # 1) Load the image
    # -----------------------------------------------------
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"[Error] Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    (img_h, img_w) = image_rgb.shape[:2]

    # Convert to ipycanvas-friendly RGBA buffer
    # ipycanvas expects a (height, width, 4) uint8 array in top-to-bottom orientation
    canvas_bg = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    canvas_bg[:, :, :3] = image_rgb[..., ::1]  # BGR->RGB fixed above, so just copy
    canvas_bg[:, :, 3] = 255  # fully opaque

    # Keep track of the existing masks
    sorted_masks = sorted(generated_masks, key=lambda m: m.get("area", 0), reverse=True)
    accepted_flags = [True]*len(sorted_masks)

    # We'll also store newly accepted masks from pointer events
    new_masks = []
    new_flags = []

    # Current "preview" mask from pointer, to be displayed on canvas but not yet accepted
    preview_mask_data = None

    # -----------------------------------------------------
    # 2) Create ipycanvas for drawing
    # -----------------------------------------------------
    canvas = ipycanvas.Canvas(width=img_w, height=img_h)
    # We'll maintain a separate overlay for the mask
    # (You could do it on the same canvas, but layering is easier with multiple canvases.)
    mask_canvas = ipycanvas.Canvas(width=img_w, height=img_h, sync_image_data=True)

    # Initialize them with the background image
    canvas.put_image_data(canvas_bg, 0, 0)

    # We'll create an ipyevents to track mouse movement/click/keyboard
    mouse_event = ipyevents.Event(source=canvas, watched_events=["mousemove", "click", "keydown"])
    # We'll keep time of last movement
    last_move_ts = time.time()
    last_mouse_pos = (0, 0)

    # We'll define a function to re-draw everything
    def redraw_all():
        # Clear canvas first
        canvas.clear()
        canvas.put_image_data(canvas_bg, 0, 0)
        # Re-draw accepted existing masks
        for i, mask_data in enumerate(sorted_masks):
            if accepted_flags[i]:
                _draw_mask(canvas, mask_data, overlay_opacity)
        # Re-draw newly accepted masks
        for i, mask_data in enumerate(new_masks):
            if new_flags[i]:
                _draw_mask(canvas, mask_data, overlay_opacity)
        # Also draw the current preview mask on top
        if preview_mask_data:
            _draw_mask(canvas, preview_mask_data, overlay_opacity)

    # We'll keep a background thread that checks if we've paused movement
    stop_thread = False
    def check_mouse_pause():
        nonlocal last_move_ts, last_mouse_pos, preview_mask_data
        while not stop_thread:
            time.sleep(0.05)  # check every ~50ms
            now = time.time()
            elapsed_ms = (now - last_move_ts)*1000
            if elapsed_ms > wait_time_ms:
                # time to run the prompt
                if checkpoint is not None and embedding_path is not None:
                    # run the point prompt at last_mouse_pos
                    px, py = last_mouse_pos
                    new_mask = run_sam_point_prompt(
                        image_rgb.shape,
                        embedding_path,
                        px, py,
                        device=device,
                        model_type=model_type,
                        checkpoint=checkpoint
                    )
                    preview_mask_data = new_mask
                    # Then re-draw
                    redraw_all()
                last_move_ts = now  # reset, so we don't keep re-calling

    # Start the background thread
    thread_obj = threading.Thread(target=check_mouse_pause, daemon=True)
    thread_obj.start()

    @mouse_event.on_dom_event
    def handle_event(event):
        nonlocal last_move_ts, last_mouse_pos, preview_mask_data
        etype = event["type"]

        if etype == "mousemove":
            # update last_mouse_pos
            x = event["relativeX"]
            y = event["relativeY"]
            # clamp
            if x < 0: x = 0
            if y < 0: y = 0
            if x >= img_w: x = img_w-1
            if y >= img_h: y = img_h-1
            last_mouse_pos = (x, y)
            last_move_ts = time.time()

        elif etype == "click":
            # If we have a preview mask, accept it
            if preview_mask_data is not None:
                new_masks.append(preview_mask_data)
                new_flags.append(True)
                preview_mask_data = None
                redraw_all()

        elif etype == "keydown":
            # If user pressed 'r' => reject last new mask?
            # If user pressed 'a' => accept last new mask?
            key_val = event.get("key", "")
            if key_val == "r":
                # revert acceptance of the last new mask if any
                if len(new_masks)>0:
                    idx = len(new_masks)-1
                    new_flags[idx] = False
                    redraw_all()
            elif key_val == "a":
                if len(new_masks)>0:
                    idx = len(new_masks)-1
                    new_flags[idx] = True
                    redraw_all()
            elif key_val == "ArrowLeft":
                # e.g., reject the last existing mask
                if len(sorted_masks)>0:
                    accepted_flags[-1] = False
                    redraw_all()
            elif key_val == "ArrowRight":
                if len(sorted_masks)>0:
                    accepted_flags[-1] = True
                    redraw_all()

    # We'll define a finalize function that merges everything
    def finalize():
        # shut down the background thread
        nonlocal stop_thread
        stop_thread = True
        # return all accepted masks
        final_list = []
        for i, m in enumerate(sorted_masks):
            if accepted_flags[i]:
                final_list.append(m)
        for i, m in enumerate(new_masks):
            if new_flags[i]:
                final_list.append(m)
        return final_list

    # We'll layout everything
    instructions = widgets.HTML(
        value=(
            "<b>Instructions:</b><br>"
            "• Hover your mouse => the model updates a preview mask after you stop moving.<br>"
            "• Click => accept that preview mask.<br>"
            "• Press 'r' => reject the last newly accepted mask.<br>"
            "• Press 'a' => re-accept the last newly accepted mask.<br>"
            "• (Optional) Press ArrowLeft => reject last old mask; ArrowRight => re-accept it.<br>"
            f"Waiting {wait_time_ms}ms after movement before running prompt."
        )
    )
    display(instructions)
    # Display the canvas
    container = widgets.VBox([canvas])
    display(container)

    # initially draw all existing accepted masks
    redraw_all()

    return finalize


##############################################################################
# Private function to draw a mask overlay
##############################################################################

def _draw_mask(canvas, mask_data, alpha=0.5):
    """Draw the bounding box + translucent overlay for a single mask_data with 'binary_mask'."""
    bin_mask = mask_data.get("binary_mask")
    if bin_mask is None:
        return  # can't draw
    h, w = bin_mask.shape
    # Use put_image_data to overlay a translucent color
    # We'll do a quick approach: color the mask region in e.g. RGBA = (30,144,255, alpha*255)
    color = (30, 144, 255, int(alpha*255))
    # We can do pixel-by-pixel drawing or a simpler polygon approach
    # ipycanvas supports .fill_style and .fill_polygon, but let's do a quick approach.

    # Minimal approach: build an RGBA array, then use Canvas.put_image_data to blend
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[bin_mask > 0] = color
    # we can do a cheap alpha compositing with the existing canvas pixels in pure python,
    # but that's quite slow. We'll do a direct put_image_data for demonstration:

    # We can do alpha compositing in python:
    with canvas.hold_sync():
        # read existing data
        old_data = canvas.get_image_data(0,0, w,h)  # shape (h, w, 4)
        # alpha blend
        # newColor = alpha*overlay + (1-alpha)*old_data
        blend_a = overlay[..., 3:]/255.0
        new_data = old_data.copy()
        new_data[..., :3] = (old_data[..., :3]*(1-blend_a) + overlay[..., :3]*blend_a).astype(np.uint8)
        # The new alpha can remain 255 or do a more advanced blend
        # We'll just set it to 255 for a stable display
        new_data[..., 3] = 255

        # put it back
        canvas.put_image_data(new_data, 0, 0)

    # Optionally, draw the bounding box
    bbox = mask_data.get("bbox")
    if bbox is not None:
        x, y, w_box, h_box = bbox
        canvas.stroke_style = "lime"
        canvas.line_width = 2
        canvas.stroke_rect(x, y, w_box, h_box)