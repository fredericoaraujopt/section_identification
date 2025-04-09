import numpy as np
import cv2

import numpy as np
import cv2

import numpy as np
import cv2
from pycocotools import mask as mask_utils

def overlay_stored_masks(image, masks, alpha=0.5, random_color=True): # initial overlay
    """
    Overlay stored masks on the original image using detailed mask information.
    - image: Original image on which to overlay masks.
    - masks: List of dictionaries, each containing mask data including a segmentation mask 
             (either as a binary array or in COCO RLE format).
    - alpha: Transparency factor for the masks.
    - random_color: Whether to apply random colors to each mask.
    
    Returns:
        An image with the masks overlaid, each mask having a thick contour.
    """
    overlay = image.copy()

    # Pre-compute random colors if necessary.
    if random_color:
        color_choices = np.array([
            [1, 0.8, 0.8],  # Pastel Red
            [0.8, 1, 0.8],  # Pastel Green
            [0.8, 0.8, 1],  # Pastel Blue
            [1, 1, 0.8],    # Pastel Yellow
            [1, 0.8, 1],    # Pastel Magenta
            [0.8, 1, 1]     # Pastel Cyan
        ])
        colors = color_choices[np.random.choice(len(color_choices), size=len(masks))]
    else:
        colors = [np.array([0.8, 0.9, 1]) for _ in range(len(masks))]

    # Process each mask individually.
    for i, mask_details in enumerate(masks):
        print(f"Processing mask {i+1}/{len(masks)}")
        segmentation = mask_details['segmentation']

        # Convert segmentation to binary mask
        binary_mask = (segmentation > 0).astype(np.uint8) 
        
        # Find contours in the binary mask.
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a color mask overlay for this mask.
        mask_color_overlay = np.zeros_like(image, dtype=np.float32)
        color = colors[i]
        for c in range(3):  # For each channel (assumes image is in BGR)
            mask_color_overlay[:, :, c] = binary_mask * color[c]
        
        # Convert the float overlay to uint8.
        mask_color_overlay = (mask_color_overlay * 255).astype(np.uint8)
        
        # Draw contours with a specified thickness (e.g., 2 pixels) and color (blue in BGR).
        cv2.drawContours(mask_color_overlay, contours, -1, (255, 0, 0), 15)
        
        # Blend the current mask overlay with the overall overlay image.
        overlay = cv2.addWeighted(overlay, 1, mask_color_overlay, alpha, 0)
    
    return overlay

def process_overlay(base_overlay, embedding, hover, samScale, orig_size, session): # hover
    """
    Process hover input to create an ephemeral mask and overlay it on the given image.
    
    Parameters:
    - base_overlay: The current display overlay.
    - embedding: The pre-computed image embedding.
    - hover: Coordinates and type of hover event.
    - samScale: Scale factor for the SAM model.
    - orig_size: Original dimensions of the image.
    - session: ONNX session to run the model.

    Returns:
    - Updated overlay with the ephemeral mask applied.
    """
    import interactive
    inputs_hover = interactive.prepare_inputs(embedding, hover, samScale, orig_size)
    mask_hover = interactive.run_model(session, inputs_hover)
    updated_overlay = interactive.overlay_mask(base_overlay, mask_hover)
    return updated_overlay

def process_new_mask(base_image, embedding, click, samScale, orig_size, session, new_masks, alpha=1): # click to display
    """
    Process a user click to generate and permanently overlay a new mask on the image.
      - base_image: The current image with all permanent overlays.
      - embedding: The pre-computed image embedding.
      - click: Tuple (x, y, clickType) for the user click.
      - samScale: Scale factor for the SAM model.
      - orig_size: Original dimensions of the image.
      - session: ONNX session to run the model.
      - new_masks: List to store details of new masks.
      - alpha: Transparency factor for the masks.
    Returns the updated image with the new mask overlayed and the updated new_masks list.
    """
    import cv2
    import numpy as np
    import interactive

    # Prepare inputs and run the model to obtain the raw mask prediction.
    inputs = interactive.prepare_inputs(embedding, click, samScale, orig_size)
    mask_pred = interactive.run_model(session, inputs)

    # Use the same color and contour style as overlay_stored_masks.
    # overlay_stored_masks uses a deep ocean blue with an alpha of 0.6.
    color = np.array([0.8, 0.9, 1])  # Default to a soft blue
    # Threshold the model output to obtain a binary mask.
    binary_mask = (mask_pred > 0).astype(np.uint8)
    # Squeeze the extra dimensions so that binary_mask has shape (H, W)
    binary_mask = np.squeeze(binary_mask)
    # Find external contours.
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an overlay image for the new mask.
    mask_color_overlay = np.zeros_like(base_image, dtype=np.float32)
    for c in range(3):  # Only use RGB channels for the overlay.
        mask_color_overlay[:, :, c] = binary_mask * color[c]
    mask_color_overlay = (mask_color_overlay * 255).astype(np.uint8)
    
    # Draw contours with thick borders.
    cv2.drawContours(mask_color_overlay, contours, -1, (255, 0, 0), 20)

    # Blend the new mask overlay onto the base image.
    updated_overlay = cv2.addWeighted(base_image, 1, mask_color_overlay, alpha, 0)

    # Store the new mask details for later reference.
    new_mask_details = {"segmentation": binary_mask, "color": color[:3].tolist()}
    new_masks.append(new_mask_details)
    print(f"New mask added at position {click}")

    return updated_overlay, new_masks

import time
def fiducials(image):
    """
    Launch fiducials mode: displays an overlay of a zoomed live view (100x100 pixels, zoom factor 4)
    based on the current mouse position. A dynamic green cross follows the hover on the zoomed view.
    On mouse click, the marker (displayed with a temporary blue indicator) is saved (with coordinates 
    relative to the original image) and printed. Press 'm' to exit fiducials mode.
    
    Parameters:
        image (np.ndarray): The original image (BGR) on which to operate.
    
    Returns:
        markers (list): List of saved marker coordinates [(x1, y1), (x2, y2), ...]
    """
    zoom_factor = 16
    zoom_window_size = 1500      # size (in pixels) for the zoomed view window
    patch_size = zoom_window_size // zoom_factor  # region size in original image (e.g. 25x25)

    markers = []                # List to hold saved marker coordinates
    current_mouse_pos = (0, 0)  # Latest mouse position (in original image coordinates)
    temp_marker = None          # Temporary marker info: (x, y, timestamp)

    # Create a dedicated window for fiducials mode.
    fiducials_window = "Fiducials Mode (Press 'm' to exit)"
    cv2.namedWindow(fiducials_window, cv2.WINDOW_NORMAL)

    def fiducials_mouse_callback(event, x, y, flags, param):
        nonlocal current_mouse_pos, markers, temp_marker
        current_mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            # Save marker (coordinates are relative to the original image)
            markers.append((x, y))
            print(f"Fiducial saved at coordinates {(x, y)}")
            temp_marker = (x, y, time.time())

    cv2.setMouseCallback(fiducials_window, fiducials_mouse_callback)

    def get_zoom_view():
        """
        Extract a patch centered on the current mouse position from the original image,
        resize it (zoom factor 4) to create a zoomed view, and draw the dynamic marker and any
        saved markers.
        """
        x, y = current_mouse_pos
        h_img, w_img = image.shape[:2]
        half_patch = patch_size // 2

        # Determine patch boundaries (handling borders)
        x1 = max(x - half_patch, 0)
        y1 = max(y - half_patch, 0)
        x2 = min(x1 + patch_size, w_img)
        y2 = min(y1 + patch_size, h_img)
        if (x2 - x1) < patch_size:
            x1 = max(x2 - patch_size, 0)
        if (y2 - y1) < patch_size:
            y1 = max(y2 - patch_size, 0)
        
        patch = image[y1:y2, x1:x2].copy()
        zoom_view = cv2.resize(patch, (zoom_window_size, zoom_window_size), interpolation=cv2.INTER_NEAREST)

        # Draw a dynamic green cross at the relative mouse position.
        rel_x = int((x - x1) * zoom_factor)
        rel_y = int((y - y1) * zoom_factor)
        cv2.drawMarker(zoom_view, (rel_x, rel_y), (0, 255, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=50, thickness=5)

        # Draw any saved markers (as red circles) that fall within the patch.
        for mx, my in markers:
            if x1 <= mx < x2 and y1 <= my < y2:
                marker_rel_x = int((mx - x1) * zoom_factor)
                marker_rel_y = int((my - y1) * zoom_factor)
                cv2.circle(zoom_view, (marker_rel_x, marker_rel_y), 5, (0, 0, 255), thickness=5)

        # Draw temporary marker (blue circle) if recently clicked.
        if temp_marker is not None:
            tx, ty, t_stamp = temp_marker
            if time.time() - t_stamp < 0.5:
                rel_tx = int((tx - x1) * zoom_factor)
                rel_ty = int((ty - y1) * zoom_factor)
                cv2.circle(zoom_view, (rel_tx, rel_ty), 10, (255, 0, 0), thickness=5)
        return zoom_view

    # Fiducials mode event loop.
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('m'):
            # Exit fiducials mode when 'm' is pressed again.
            break

        display_img = image.copy()

        # Draw persistent markers (red circles) on the display image.
        for mx, my in markers:
            cv2.circle(display_img, (mx, my), 10, (0, 0, 255), thickness=5)

        # Draw a temporary marker (blue circle) if applicable.
        if temp_marker is not None:
            tx, ty, t_stamp = temp_marker
            if time.time() - t_stamp < 0.5:
                cv2.circle(display_img, (tx, ty), 10, (255, 0, 0), thickness=5)
            else:
                temp_marker = None

        # Get the zoom view based on the current mouse position.
        zoom_view = get_zoom_view()

        # Overlay the zoom view on the top-right corner.
        h_img, w_img = display_img.shape[:2]
        h_zoom, w_zoom = zoom_view.shape[:2]
        x_offset = max(w_img - w_zoom - 10, 0)
        y_offset = 10
        display_img[y_offset:y_offset+h_zoom, x_offset:x_offset+w_zoom] = zoom_view
        cv2.rectangle(display_img, (x_offset, y_offset), (x_offset+w_zoom, y_offset+h_zoom), (0, 255, 255), thickness=2)

        cv2.imshow(fiducials_window, display_img)

    cv2.destroyWindow(fiducials_window)
    return markers

def exclude_mask(image, stored_masks, new_masks, current_mouse, base_overlay):
    """
    Exclude a mask from stored_masks or new_masks based on the current mouse coordinates.
    
    Parameters:
      image         : Original BGR image (numpy array).
      stored_masks  : List of initially stored masks (each with a 'segmentation' field).
      new_masks     : List of masks added during the session (each with a 'segmentation' field).
      current_mouse : Tuple (x, y, clickType) or (x, y) representing the current mouse coordinates.
      base_overlay  : The current permanent overlay image.
      
    Returns:
      updated_base_overlay, new_masks, stored_masks.
    
    Workflow:
      1. Check at the given coordinates whether any mask (from new_masks or stored_masks) exists.
      2. If found, create a temporary overlay that highlights that mask’s contour (in red) and display it.
      3. Wait for the user to press “r” again to confirm deletion.
      4. If confirmed, remove that mask from its list and update base_overlay accordingly.
      5. If not confirmed, leave the masks unchanged.
    """
    import cv2
    import numpy as np
    import interactive  # for overlay_stored_masks
    
    # Unpack mouse coordinates (assume first two elements are x, y)
    x, y = current_mouse[:2]
    mask_found = None
    mask_list = None  # either 'new' or 'stored'
    mask_index = None

    # Helper: Given a segmentation field, return a binary mask (2D).
    def get_binary(segmentation):
        bin_mask = np.squeeze(segmentation)
        if bin_mask.dtype != np.uint8:
            bin_mask = (bin_mask > 0).astype(np.uint8)
        return bin_mask

    # Search new_masks first.
    for i, mask in enumerate(new_masks):
        bin_mask = get_binary(mask["segmentation"])
        # Ensure the coordinate is within the mask shape.
        if y < bin_mask.shape[0] and x < bin_mask.shape[1]:
            if bin_mask[y, x] == 1:
                mask_found = mask
                mask_list = 'new'
                mask_index = i
                break

    # If not found, search stored_masks.
    if mask_found is None:
        for i, mask in enumerate(stored_masks):
            bin_mask = get_binary(mask["segmentation"])
            if y < bin_mask.shape[0] and x < bin_mask.shape[1]:
                if bin_mask[y, x] == 1:
                    mask_found = mask
                    mask_list = 'stored'
                    mask_index = i
                    break

    if mask_found is None:
        print("No mask found at this location.")
        return base_overlay, new_masks, stored_masks

    # If a mask is found, extract its contours for highlighting.
    seg = get_binary(mask_found["segmentation"])
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a temporary overlay: copy the current base overlay.
    temp_overlay = base_overlay.copy()
    # Highlight the found mask by drawing its contours in red (BGR: (0, 0, 255)) with a thick line.
    cv2.drawContours(temp_overlay, contours, -1, (0, 0, 255), 4)
    cv2.imshow("Confirm Exclusion", temp_overlay)
    print("Mask found at location ({}, {}).".format(x, y))
    print("Press 'r' again to confirm deletion of the highlighted mask, or any other key to cancel.")
    
    # Wait indefinitely for a key press.
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow("Confirm Exclusion")
    
    if key == ord('r'):
        # Delete the mask.
        if mask_list == 'new':
            del new_masks[mask_index]
            print("Deleted mask from new_masks.")
        else:
            del stored_masks[mask_index]
            print("Deleted mask from stored_masks.")
        # Recompute the base overlay by combining the remaining masks.
        combined_masks = stored_masks + new_masks
        base_overlay = interactive.overlay_stored_masks(image, combined_masks)
    else:
        print("Exclusion cancelled.")
    
    return base_overlay, new_masks, stored_masks

# -----------------------------------------------
# HELPER that uses SamPredictor instead of ONNX
# -----------------------------------------------

def process_new_mask_SamPredictor(base_image, click, predictor, new_masks):
    """
    Process a user click to generate and permanently overlay a new mask on the image using SamPredictor.
      - base_image: The current image with permanent overlays.
      - click: Tuple (x, y) representing the user click coordinates.
      - predictor: A SamPredictor instance that has already been set with the current image.
      - new_masks: List to store details of new masks.
    Returns:
      - updated_overlay: The image with the new mask overlaid.
      - new_masks: The updated list of mask details.
    """
    import numpy as np
    import cv2
    import interactive  # Assuming interactive provides overlay_stored_masks

    # Unpack the click coordinates.
    x, y = click[:2]
    
    # Prepare the point prompt: one positive point.
    input_point = np.array([[x, y]], dtype=np.float32)
    input_label = np.array([1], dtype=np.float32)  # 1 indicates a positive prompt

    # Predict masks using SamPredictor.
    # Here we use multimask_output=True to obtain several candidate masks.
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )
    
    # Select the best mask (with the highest score).
    best_index = int(np.argmax(scores))
    best_mask = masks[best_index]  # Shape: (H, W), binary mask

    # Compute additional mask details to match standard SAM output structure.
    # Calculate the area (number of mask pixels).
    area = int(np.sum(best_mask))
    
    # Compute bounding box from the largest contour.
    binary_mask = best_mask.astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x_box, y_box, w_box, h_box = cv2.boundingRect(largest_contour)
        bbox = [x_box, y_box, w_box, h_box]
    else:
        bbox = [0, 0, 0, 0]
    
    # Build the new mask details dictionary.
    new_mask_details = {
        "segmentation": best_mask,
        "area": area,
        "bbox": bbox,
        "predicted_iou": float(scores[best_index]),  # Use the predictor's score as a proxy
        "point_coords": [[x, y]],
        "stability_score": 0.0,  # Placeholder (can be updated if more info is available)
        "crop_box": [0, 0, base_image.shape[1], base_image.shape[0]],
        "color": (0, 114, 189)  # Standard blue color for consistency
    }
    
    # Append the new mask details to the list.
    new_masks.append(new_mask_details)
    print(f"New mask added: {new_mask_details} at position {(x, y)}")
    
    # Update the overlay using overlay_stored_masks for consistent appearance.
    updated_overlay = interactive.overlay_stored_masks(base_image, new_masks)
    
    return updated_overlay, new_masks