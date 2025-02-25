import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

from manual_detector_helpers import overlay_stored_masks
from manual_detector_helpers import fiducials

# Import your helper functions from your package.
# (Assumes these are defined in modules onnx_export and embedding_helper.)
from onnx_export import install_and_export_sam_onnx
from create_embedding import create_embedding_if_needed

# Constants
LONG_SIDE_LENGTH = 1024         # SAM requires the longest side to be 1024
THROTTLE_TIME = 0.05            # Minimum time (in seconds) between model inferences

# Global variables for interactive state
latest_click = None             # (x, y, clickType) updated from mouse events
last_processed_click = None     # Last processed click (to throttle)
last_event_time = 0             # Timestamp of last accepted mouse event
current_overlay = None          # The current overlay image to display

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
    """OpenCV mouse callback. On mouse move, update latest_click if throttle interval passed."""
    global latest_hover, latest_click, last_event_time
    current_time = time.time()
    if event == cv2.EVENT_MOUSEMOVE:
        if current_time - last_event_time > THROTTLE_TIME:
            latest_hover = (x, y, 1)  # treating all moves as positive prompts, update hover variable
            last_event_time = current_time
    elif event == cv2.EVENT_LBUTTONDOWN:
        latest_click = (x, y, 1) # update click variable
    

def run_sam_interactive(image_path, checkpoint, stored_masks, model_type="vit_h", device="cpu"):
    """
    Run interactive SAM segmentation for a given image.
      - image_path: path to the input image.
      - checkpoint: path to the SAM model checkpoint (.pth)
      - stored_masks: initially stored masks.
      - model_type: SAM model type, default "vit_h"
      - device: device to run the embedding creation ("cpu" or "cuda")
    The function exports a quantized ONNX model for this image, creates the image embedding
    (if needed), and launches an interactive OpenCV window. The loop stops when ESC is pressed.
    """
    from manual_detector_helpers import process_new_mask
    global latest_click, latest_hover, last_processed_click, current_overlay

    # Step 1: Export and quantize the ONNX model for this image.
    image_path = Path(image_path)
    final_model_path = install_and_export_sam_onnx(
        image_path=image_path,
        checkpoint=checkpoint,
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
    print(f"Image size: {w}x{h} (width x height), samScale: {samScale:.3f}")

    # Overlay initally stored masks
    base_overlay = overlay_stored_masks(image, stored_masks)
    new_masks = [] # Initialize array for new masks

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

    # Reset global mouse event trackers
    latest_click = None
    latest_hover = None
    last_processed_click = None

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break 

            if key == ord('m'):
                # Suspend segmentation and launch fiducials mode.
                markers = fiducials(image)
                print("Fiducial markers collected:", markers)
                # After fiducials mode, segmentation resumes with its previous state.

            # Start with the permanent overlay.
            overlay_to_display = base_overlay.copy()

            # --- Dynamic Hover: Create an ephemeral mask using latest_hover.
            if latest_hover is not None:
                inputs_hover = prepare_inputs(embedding, latest_hover, samScale, orig_size)
                mask_hover = run_model(session, inputs_hover)
                overlay_to_display = overlay_mask(overlay_to_display, mask_hover)

            # --- Permanent Click: Process click events only if new.
            if latest_click is not None and latest_click != last_processed_click:
                base_overlay,new_masks = process_new_mask(base_overlay, embedding, latest_click, samScale, orig_size, session, new_masks)
                last_processed_click = latest_click
                # Refresh the display overlay with the new permanent mask.
                overlay_to_display = base_overlay.copy()

            cv2.imshow(window_name, overlay_to_display)

    except Exception as e:
        print("An error occurred during the interactive loop:", e)
    finally:
        cv2.destroyAllWindows()
        print("[Info] Exiting interactive segmentation.")
        return new_masks, stored_masks, markers