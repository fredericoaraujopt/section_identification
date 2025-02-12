import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

import onnxruntime
from segment_anything import sam_model_registry, SamPredictor
import pycocotools.mask as mask_util

##############################################################################
# manual_correction: Real-time hover to get masks from SAM ONNX
##############################################################################

import os
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

def create_embedding_if_needed(image_path, checkpoint, model_type="vit_h", device="cpu"):
    """
    Creates a SAM embedding file for the given image_path using the specified checkpoint/model_type,
    unless the embedding file already exists.  The file will be saved to the same directory 
    as image_path, named "<original_basename>_embedding.npy".

    Args:
        image_path (str): Path to the image file.
        checkpoint (str or Path): Path to the SAM checkpoint (e.g. 'sam_vit_h_4b8939.pth').
        model_type (str): Type of the SAM model, default 'vit_h'.
        device (str): 'cpu' or e.g. 'cuda:0'.

    Returns:
        str: The path to the embedding file.
    """
    # 1) Create the embedding file name in the same directory
    embedding_file = f"{os.path.splitext(image_path)[0]}_embedding.npy"

    # 2) Check if it already exists
    if os.path.exists(embedding_file):
        print(f"[Info] Embedding file already exists for '{image_path}': {embedding_file}")
        return embedding_file

    # 3) Otherwise, load the image and create the embedding
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"[Error] Could not read image from path: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)

    image_embedding = predictor.get_image_embedding().cpu().numpy()
    np.save(embedding_file, image_embedding)
    print(f"[Info] Created embedding file: {embedding_file}")

    return embedding_file

# NEW VERSION

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ipywidgets import (
    interact,
    IntSlider,
    Button,
    Text,
    Dropdown,
    HBox,
    VBox,
    Label,
    BoundedIntText
)

# If you have your own create_embedding_if_needed, import it:
# from some_module import create_embedding_if_needed
from segment_anything import sam_model_registry, SamPredictor


def manual_correction_v2(
    image_path,
    generated_masks,
    checkpoint=None,
    model_type="vit_h",
    device="cuda",
    display_mask_overlay=True
):
    """
    A partial Python-based approach to manually correct (add or remove) masks
    that were automatically identified by Segment Anything.

    Features:
      1) Ensures the SAM image embedding is created if you provide a checkpoint.
      2) Displays the existing masks from 'generated_masks'.
      3) Lets you accept/reject masks.
      4) Lets you optionally add new masks via bounding box or point prompts.
      5) Returns a finalize() function to collect the final accepted set.

    Args:
        image_path (str or Path): Path to the original image file.
        generated_masks (list): A list of mask dicts from 'automatic_identification'.
            Each dict should have at least:
              - 'bbox': [x_min, y_min, width, height]
              - 'area': (float)
              - 'binary_mask': (2D boolean np.ndarray) - optional, if you want overlay
        checkpoint (str, optional): Path to SAM .pth checkpoint. If provided, we
                                    can re-run prompts to add new masks.
        model_type (str): Which SAM model to use. e.g. "vit_h", "vit_l", "vit_b".
        device (str): "cuda" or "cpu".
        display_mask_overlay (bool): If True, overlays the binary mask on the image.

    Returns:
        A function finalize() which, when called, returns the final accepted list of masks.
    """
    # ---------------------------------------------------------------
    # 0) Ensure embedding is created if checkpoint is provided
    # ---------------------------------------------------------------
    if checkpoint is not None:
        embedding_path = create_embedding_if_needed(
            image_path,
            checkpoint=checkpoint,
            model_type=model_type,
            device=device
        )
    else:
        embedding_path = f"{os.path.splitext(str(image_path))[0]}_embedding.npy"
        if not os.path.exists(embedding_path):
            print("[Warning] No checkpoint given, and no existing embedding found. "
                  "Cannot add new masks from prompts. Displaying only existing masks...")

    # ---------------------------------------------------------------
    # 1) Load image
    # ---------------------------------------------------------------
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"[Error] Could not read image at path: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_h, image_w = image_rgb.shape[:2]

    # ---------------------------------------------------------------
    # 2) Sort the existing masks by area (descending or ascending)
    # ---------------------------------------------------------------
    sorted_masks = sorted(generated_masks, key=lambda x: x["area"], reverse=True)
    accepted_flags = [True] * len(sorted_masks)  # track accept/reject

    # We'll store any new masks the user adds from prompts
    new_prompt_masks = []
    new_accepted_flags = []  # parallel to new_prompt_masks

    # ---------------------------------------------------------------
    # 3) Helper to show a single mask
    # ---------------------------------------------------------------
    def _show_mask(img, mask_index, from_existing=True):
        """Display the mask with bounding box (and optional overlay)."""
        if from_existing:
            mask_data = sorted_masks[mask_index]
            is_accepted = accepted_flags[mask_index]
        else:
            mask_data = new_prompt_masks[mask_index]
            is_accepted = new_accepted_flags[mask_index]

        area = mask_data.get("area", -1)
        bbox = mask_data["bbox"]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        rect = Rectangle(
            (bbox[0], bbox[1]),
            bbox[2],
            bbox[3],
            linewidth=2,
            edgecolor="lime",
            facecolor="none"
        )
        ax.add_patch(rect)

        # If we have 'binary_mask' and user wants overlay
        if display_mask_overlay and "binary_mask" in mask_data:
            bin_mask = mask_data["binary_mask"]
            if bin_mask.shape[:2] == (image_h, image_w):
                overlay = np.zeros((image_h, image_w, 4), dtype=np.float32)
                color_rgba = np.array([30/255, 144/255, 255/255, 0.5])
                overlay[bin_mask > 0] = color_rgba
                ax.imshow(overlay)

        status = "ACCEPTED" if is_accepted else "REJECTED"
        ax.set_title(f"Mask area={area:.0f}, {status}")
        plt.show()

    # ---------------------------------------------------------------
    # 4) Interactive acceptance for existing masks
    # ---------------------------------------------------------------
    slider_existing = IntSlider(
        min=0,
        max=len(sorted_masks) - 1 if len(sorted_masks) > 0 else 0,
        step=1,
        value=0,
        description="Mask # (Existing)"
    )
    info_label_exist = Label(value="Accept or reject the current mask.")
    btn_accept_exist = Button(description="Accept", button_style="success")
    btn_reject_exist = Button(description="Reject", button_style="danger")

    def _update_display_existing(idx):
        if len(sorted_masks) == 0:
            print("[Info] No existing masks to display.")
            return
        _show_mask(image_rgb, idx, from_existing=True)

    def _accept_exist(_):
        idx = slider_existing.value
        accepted_flags[idx] = True
        info_label_exist.value = f"Mask #{idx} => ACCEPTED"
        _show_mask(image_rgb, idx, from_existing=True)

    def _reject_exist(_):
        idx = slider_existing.value
        accepted_flags[idx] = False
        info_label_exist.value = f"Mask #{idx} => REJECTED"
        _show_mask(image_rgb, idx, from_existing=True)

    btn_accept_exist.on_click(_accept_exist)
    btn_reject_exist.on_click(_reject_exist)

    from ipywidgets import Output, interactive_output
    w_out_exist = Output()

    controls_exist = HBox([btn_accept_exist, btn_reject_exist])
    ui_exist = VBox([slider_existing, controls_exist, info_label_exist])
    display(ui_exist, w_out_exist)

    def _observe_exist(change):
        with w_out_exist:
            w_out_exist.clear_output(wait=True)
            _update_display_existing(change["new"])

    slider_existing.observe(_observe_exist, names="value")
    # Trigger initial display if we have existing masks
    if len(sorted_masks) > 0:
        _update_display_existing(slider_existing.value)

    # ---------------------------------------------------------------
    # 5) Optional: Add new masks from bounding box or point prompts
    # ---------------------------------------------------------------
    # This is only possible if we have a valid checkpoint & embedding:
    can_add_prompts = (checkpoint is not None and os.path.exists(embedding_path))

    label_add_prompt = Label(value="(Optional) Add a new mask via a bounding box or point prompt.")
    dropdown_prompt = Dropdown(options=["None", "BoundingBox", "Point"], value="None",
                               description="Prompt Type:")

    # For bounding box
    bbox_xmin = BoundedIntText(value=50, min=0, max=image_w, step=1, description="x1:")
    bbox_ymin = BoundedIntText(value=50, min=0, max=image_h, step=1, description="y1:")
    bbox_xmax = BoundedIntText(value=150, min=0, max=image_w, step=1, description="x2:")
    bbox_ymax = BoundedIntText(value=150, min=0, max=image_h, step=1, description="y2:")

    btn_add_bbox = Button(description="Add Mask (BBox Prompt)", button_style="info")

    # For point prompt
    point_x = BoundedIntText(value=100, min=0, max=image_w, step=1, description="px:")
    point_y = BoundedIntText(value=100, min=0, max=image_h, step=1, description="py:")
    point_label = Dropdown(options=[("Positive", 1), ("Negative", 0)], value=1, description="Type:")
    btn_add_point = Button(description="Add Mask (Point Prompt)", button_style="info")

    # We also show a slider for newly added masks (to accept/reject them)
    slider_new = IntSlider(
        min=0,
        max=0,
        step=1,
        value=0,
        description="Mask # (New)"
    )
    info_label_new = Label(value="No new masks yet.")
    btn_accept_new = Button(description="Accept new", button_style="success")
    btn_reject_new = Button(description="Reject new", button_style="danger")

    def _add_bbox_prompt(_):
        if not can_add_prompts:
            print("[Warning] Cannot add new masks without a valid checkpoint & embedding.")
            return
        # Gather bounding box
        x1 = bbox_xmin.value
        y1 = bbox_ymin.value
        x2 = bbox_xmax.value
        y2 = bbox_ymax.value
        if x2 < x1 or y2 < y1:
            print("[Error] x2 < x1 or y2 < y1 => invalid box. Please fix.")
            return

        # Run SAM with box prompt
        new_mask = _run_sam_prompt(
            image_rgb,
            embedding_path,
            prompt_type="box",
            box=(x1, y1, x2, y2),
            device=device,
            model_type=model_type,
            checkpoint=checkpoint
        )
        if new_mask is None:
            print("[Error] Could not create a new mask from that bounding box.")
            return

        new_prompt_masks.append(new_mask)
        new_accepted_flags.append(True)
        slider_new.max = len(new_prompt_masks) - 1
        info_label_new.value = f"Added 1 new mask. Total new: {len(new_prompt_masks)}"

    def _add_point_prompt(_):
        if not can_add_prompts:
            print("[Warning] Cannot add new masks without a valid checkpoint & embedding.")
            return
        px = point_x.value
        py = point_y.value
        plabel = point_label.value  # 1=positive, 0=negative

        new_mask = _run_sam_prompt(
            image_rgb,
            embedding_path,
            prompt_type="point",
            point=(px, py, plabel),
            device=device,
            model_type=model_type,
            checkpoint=checkpoint
        )
        if new_mask is None:
            print("[Error] Could not create a new mask from that point prompt.")
            return
        new_prompt_masks.append(new_mask)
        new_accepted_flags.append(True)
        slider_new.max = len(new_prompt_masks) - 1
        info_label_new.value = f"Added 1 new mask. Total new: {len(new_prompt_masks)}"

    btn_add_bbox.on_click(_add_bbox_prompt)
    btn_add_point.on_click(_add_point_prompt)

    def _update_display_new(idx):
        if len(new_prompt_masks) == 0:
            print("[Info] No new masks to show yet.")
            return
        _show_mask(image_rgb, idx, from_existing=False)

    def _accept_new(_):
        idx = slider_new.value
        new_accepted_flags[idx] = True
        info_label_new.value = f"New mask #{idx} => ACCEPTED"
        _show_mask(image_rgb, idx, from_existing=False)

    def _reject_new(_):
        idx = slider_new.value
        new_accepted_flags[idx] = False
        info_label_new.value = f"New mask #{idx} => REJECTED"
        _show_mask(image_rgb, idx, from_existing=False)

    btn_accept_new.on_click(_accept_new)
    btn_reject_new.on_click(_reject_new)

    w_out_new = Output()

    controls_new_sliders = HBox([slider_new, btn_accept_new, btn_reject_new])
    controls_bbox = HBox([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, btn_add_bbox])
    controls_point = HBox([point_x, point_y, point_label, btn_add_point])

    # For clarity in the notebook layout:
    label_new_section = Label(value="=== Add/Review Newly Created Masks ===")
    ui_new = VBox([
        label_add_prompt, 
        controls_bbox,
        controls_point,
        label_new_section,
        controls_new_sliders,
        info_label_new
    ])

    display(ui_new, w_out_new)

    def _observe_new(change):
        with w_out_new:
            w_out_new.clear_output(wait=True)
            _update_display_new(change["new"])

    slider_new.observe(_observe_new, names="value")

    # ----------------------------------------------------------------
    # 6) finalize() function
    # ----------------------------------------------------------------
    def finalize():
        """
        Returns the final list of accepted masks:
        1) accepted from the original 'generated_masks'
        2) newly created masks from bounding box/point prompts
        """
        # combine old + new
        final_list = []
        for idx, m in enumerate(sorted_masks):
            if accepted_flags[idx]:
                final_list.append(m)
        for idx, m in enumerate(new_prompt_masks):
            if new_accepted_flags[idx]:
                final_list.append(m)
        return final_list

    return finalize


def create_embedding_if_needed(
    image_path,
    checkpoint,
    model_type="vit_h",
    device="cuda"
):
    """
    Example helper that ensures an embedding .npy file exists, as in the official SAM notebook.
    """
    import cv2
    from segment_anything import sam_model_registry, SamPredictor

    embedding_path = f"{os.path.splitext(str(image_path))[0]}_embedding.npy"
    if os.path.exists(embedding_path):
        print(f"[Info] Embedding already exists: {embedding_path}")
        return embedding_path

    # Load image
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"[Error] Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    embedding = predictor.get_image_embedding().cpu().numpy()
    np.save(embedding_path, embedding)
    print(f"[Info] Created embedding file: {embedding_path}")
    return embedding_path


def _run_sam_prompt(
    image_rgb,
    embedding_path,
    prompt_type,
    box=None,
    point=None,
    device="cuda",
    model_type="vit_h",
    checkpoint=None
):
    """
    A helper to run SAM on an existing image embedding with a single bounding box or point prompt.
    Returns a dictionary: { 'bbox', 'area', 'binary_mask' } for the largest predicted mask.
    """

    if not os.path.exists(embedding_path):
        print(f"[Error] No embedding found at {embedding_path}!")
        return None

    # Load the embedding
    image_embedding = np.load(embedding_path)

    # Load the model again (lightweight prompt encoder & mask decoder)
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    # Instead of predictor.set_image(image_rgb), we manually override the embedding
    predictor.set_torch_image_embedding(
        torch_embedding=torch.tensor(image_embedding, device=device),
        original_image_size=image_rgb.shape[:2]
    )

    # Prepare inputs for the onnx-like style:
    # But in PyTorch, we can just call predictor.predict()
    # We'll do the simpler approach with `predictor`:
    if prompt_type == "box" and box is not None:
        input_box = np.array(box, dtype=np.float32).reshape(2, 2)
        transformed_box = predictor.transform.apply_boxes(input_box, image_rgb.shape[:2])
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=transformed_box,
            multimask_output=True
        )
        # Take the best mask
        best_idx = np.argmax(scores)
        chosen_mask = masks[best_idx]
        # Convert bounding box
        y_indices, x_indices = np.where(chosen_mask)
        if len(y_indices) == 0:
            print("[Warning] Box prompt produced an empty mask.")
            return None
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        bounding_box = [x_min, y_min, x_max - x_min, y_max - y_min]
        area = float(np.sum(chosen_mask))

        return {
            "bbox": bounding_box,
            "area": area,
            "binary_mask": chosen_mask
        }

    elif prompt_type == "point" and point is not None:
        px, py, plabel = point  # e.g. (x, y, 1)
        input_point = np.array([[px, py]], dtype=np.float32)
        input_label = np.array([plabel], dtype=np.int32)
        transformed_point = predictor.transform.apply_coords(
            coords=input_point,
            shape=image_rgb.shape[:2]
        )

        masks, scores, logits = predictor.predict(
            point_coords=transformed_point,
            point_labels=input_label,
            multimask_output=True
        )
        best_idx = np.argmax(scores)
        chosen_mask = masks[best_idx]

        y_indices, x_indices = np.where(chosen_mask)
        if len(y_indices) == 0:
            print("[Warning] Point prompt produced an empty mask.")
            return None
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        bounding_box = [x_min, y_min, x_max - x_min, y_max - y_min]
        area = float(np.sum(chosen_mask))

        return {
            "bbox": bounding_box,
            "area": area,
            "binary_mask": chosen_mask
        }

    else:
        print(f"[Error] Unknown prompt type '{prompt_type}' or missing data.")
        return None
    
# NEW

def manual_correction(
    sorted_masks,
    image_path,
    onnx_model_path,
    checkpoint=None,
    model_type = "vit_h",
    device="cpu"
):
    """
    Mimic the Segment Anything web demo with real-time 'hover' to preview a mask,
    using an ONNX model + precomputed embedding. Allows inclusion/exclusion
    of masks in a final list.
    ...

    Args:
        sorted_masks (list): The output from automatic_identification(...).
        image_path (str): Path to the same image used for identification.
        onnx_model_path (str): Path to the quantized ONNX model file.
        checkpoint (str): Path to SAM checkpoint, only needed if embedding not yet created.
        model_type (str): Name of the SAM architecture, e.g. "vit_h" (default).
        device (str): (Unused for onnxruntime GPU) but used for embedding creation if needed.

    Returns:
        final_masks (list): The updated list of masks.
    """

    # -------------------------------------------------------------------------
    # 0) Ensure the embedding is created if it does not exist
    # -------------------------------------------------------------------------
    if checkpoint is not None:
        embedding_path = create_embedding_if_needed(
            image_path,
            checkpoint=checkpoint,
            model_type=model_type,
            device=device
        )
    else:
        # If user already has an embedding path, or pass None if we strictly need it:
        # Raise an error if the embedding is not found.
        embedding_path = f"{os.path.splitext(image_path)[0]}_embedding.npy"
        if not os.path.exists(embedding_path):
            print(f"[Warning] No checkpoint provided and embedding not found: {embedding_path}")

    # -----------------------------
    # 1) Load image + embedding
    # -----------------------------
    if not os.path.exists(image_path):
        print(f"[Error] image_path does not exist: {image_path}")
        return sorted_masks
    img_pil = Image.open(image_path).convert("RGB")
    img = np.array(img_pil)
    H, W = img.shape[:2]

    if not os.path.exists(embedding_path):
        print(f"[Error] embedding_path does not exist: {embedding_path}")
        return sorted_masks
    image_embedding = np.load(embedding_path).astype(np.float32)

    if not os.path.exists(onnx_model_path):
        print(f"[Error] onnx_model_path does not exist: {onnx_model_path}")
        return sorted_masks

    # Create ONNX session
    session = onnxruntime.InferenceSession(
        onnx_model_path, 
        providers=["CPUExecutionProvider"]
    )
    output_names = [o.name for o in session.get_outputs()]  # e.g. ["masks","iou_predictions","low_res_masks"]

    # For thresholding the predicted logits
    # (By default, Segment Anything uses 0.0 as mask_threshold.)
    mask_threshold = 0.0

    # -----------------------------
    # 2) Keep track of final masks
    #    Start with a copy of sorted_masks (all "included" by default).
    # -----------------------------
    final_masks = list(sorted_masks)

    # We'll store the "hover preview" as a dictionary so we can add it if the user
    # presses "i" to include.
    current_preview_mask = None  # store a dict with {bool_mask, iou, etc.}

    # -----------------------------
    # 3) Matplotlib Figure Setup
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Manual Correction Demo (Hover -> Preview Mask)")

    # Show the base image
    ax.imshow(img)

    # Create a second overlay for the dynamic “preview” mask
    preview_layer = ax.imshow(np.zeros_like(img), alpha=0.4, cmap='jet')
    preview_layer.set_visible(False)  # hidden at first

    # Create a bounding-box overlay for the “included” masks
    # We'll re-draw them each time something changes.
    def draw_included_masks():
        # Clear old bounding boxes
        for patch in reversed(ax.patches):
            patch.remove()

        # Draw bounding box for each included mask
        for i, m in enumerate(final_masks):
            bbox = m['bbox']  # [x, y, w, h]
            rect = Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                linewidth=1.5, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                bbox[0] + bbox[2]/2,
                bbox[1] + bbox[3]/2,
                f"{i}",
                color='yellow', ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.5, pad=1),
                fontsize=8
            )

    draw_included_masks()

    fig.canvas.draw_idle()

    # -----------------------------
    # 4) Helper: Run Onnx with single point
    # -----------------------------
    def run_onnx_inference(px, py):
        """
        px, py: pixel coordinates in the original image.
        Returns a bool_mask plus iou_pred
        """
        # naive scaling for the downsample factor
        # if image_embedding is shape (1,256,H//4,W//4) (ViT-H),
        # then scale_x = (W//4)/W
        _, _, eH, eW = image_embedding.shape
        scale_x = eW / W
        scale_y = eH / H

        coords = np.array([[[px*scale_x, py*scale_y]]], dtype=np.float32)  # shape (1,1,2)
        labels = np.array([[1]], dtype=np.float32)  # 1=positive prompt

        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)  # no prior mask
        has_mask = np.array([0.0], dtype=np.float32)
        orig_im_size = np.array([H, W], dtype=np.float32)

        onnx_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": coords,
            "point_labels": labels,
            "mask_input": mask_input,
            "has_mask_input": has_mask,
            "orig_im_size": orig_im_size
        }

        outputs = session.run(output_names, onnx_inputs)
        masks = outputs[0]  # shape (1,1,H,W)
        iou_predictions = outputs[1]  # shape (1,1)

        mask_2d = masks[0,0] > mask_threshold
        iou = float(iou_predictions[0,0])

        return mask_2d, iou

    # -----------------------------
    # 5) Mouse Move Callback
    #    Like the web demo: each time the user releases the mouse, we run the model 
    #    for that location and update the preview mask.
    # -----------------------------
    
    def on_mouse_release(event):
        nonlocal current_preview_mask
        if not event.inaxes:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        x_int, y_int = int(round(x)), int(round(y))
        # Ensure in bounds
        if x_int < 0 or y_int < 0 or x_int >= W or y_int >= H:
            return

        # Run ONNX for single point
        bool_mask, iou_pred = run_onnx_inference(x_int, y_int)
        if np.any(bool_mask):
            # Update preview image
            # Create an RGBA overlay
            preview_img = np.zeros_like(img)
            preview_img[bool_mask] = [255, 0, 0]  # red
            preview_layer.set_data(preview_img)
            preview_layer.set_visible(True)
            fig.canvas.draw_idle()

            # Store in current_preview_mask
            current_preview_mask = {
                'segmentation': bool_mask,
                'iou': iou_pred
            }
        else:
            # hide if empty
            preview_layer.set_visible(False)
            current_preview_mask = None
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_mouse_release) # 'button_release_event'

    # -----------------------------
    # 6) Convert boolean mask -> SAM style dict
    # -----------------------------
    def convert_bool_mask_to_sam_dict(bool_mask, iou):
        """
        Creates a dict with the same fields as SamAutomaticMaskGenerator output,
        e.g. 'segmentation', 'area', 'bbox', ...
        """
        # Use pycocotools to encode RLE
        rle = mask_util.encode(np.asfortranarray(bool_mask.astype(np.uint8)))
        area = int(bool_mask.sum())
        rows, cols = np.where(bool_mask)
        if rows.size == 0 or cols.size == 0:
            bbox = [0,0,0,0]
        else:
            y_min, y_max = rows.min(), rows.max()
            x_min, x_max = cols.min(), cols.max()
            w = x_max - x_min + 1
            h = y_max - y_min + 1
            bbox = [float(x_min), float(y_min), float(w), float(h)]

        return {
            'segmentation': rle,
            'area': area,
            'bbox': bbox,
            'predicted_iou': iou,
            'stability_score': iou,  # or some other measure
            'crop_box': [0,0,W,H]
        }

    # -----------------------------
    # 7) Key Presses to Include/Exclude
    #    Press 'i' to include the current preview as a new mask.
    #    Press 'e' to exclude the last included mask (or the user can specify logic).
    # -----------------------------
    def on_key_press(event):
        nonlocal final_masks, current_preview_mask
        if event.key == 'i':
            # Include
            if current_preview_mask is None:
                print("[Info] No preview mask to include.")
                return
            # Convert to SAM dict
            bool_mask = current_preview_mask['segmentation']
            iou = current_preview_mask['iou']
            sam_dict = convert_bool_mask_to_sam_dict(bool_mask, iou)

            if sam_dict['area'] == 0:
                print("[Info] Preview mask is empty, skipping.")
                return

            final_masks.append(sam_dict)
            print(f"[Add] New mask included (area={sam_dict['area']}, iou={iou:.3f}).")

            # Redraw
            draw_included_masks()
            fig.canvas.draw_idle()

        elif event.key == 'e':
            # Exclude last mask (simple approach)
            if len(final_masks) > 0:
                removed = final_masks.pop()
                print(f"[Remove] Last mask removed, area={removed['area']}.")
                draw_included_masks()
                fig.canvas.draw_idle()
            else:
                print("[Info] No masks to remove.")

        elif event.key == 'escape':
            # ESC pressed: we can end the session
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # -----------------------------
    # 8) Start the interactive loop
    # -----------------------------
    print("[Instructions]")
    print(" - Hover mouse to preview a mask in real-time.")
    print(" - Press 'i' to INCLUDE the previewed mask.")
    print(" - Press 'e' to EXCLUDE the last included mask.")
    print(" - Press ESC or close the window to finish.\n")

    plt.show()

    # After the window is closed:
    print(f"[Done] Final number of masks = {len(final_masks)}")
    return final_masks