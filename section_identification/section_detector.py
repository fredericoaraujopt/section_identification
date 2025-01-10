import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ipywidgets import interact, IntSlider
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from .preprocess import preprocess_image
from .filtering import filtering

def automatic_identification(image_path, compress=False, apply_filtering=False, eps_values=None, min_samples_values=None, **kwargs):
    """
    Perform automatic mask identification on an input image using the SAM model.

    Args:
        image_path (str): Path to the input image file.
        compress (bool): Whether to compress the image before processing.
        filtering (bool): Whether to filter masks using the filtering function.
        eps_values (list or array-like): Values of eps to try in DBSCAN for filtering.
        min_samples_values (list or array-like): Values of min_samples to try in DBSCAN for filtering.
        **kwargs: Optional parameters to configure the SAM model. Supported keys include:
            - points_per_side (int): Number of points per side for mask generation.
            - pred_iou_thresh (float): IOU threshold for mask prediction.
            - stability_score_thresh (float): Stability score threshold for mask filtering.
            - min_mask_region_area (int): Minimum area of a mask to be considered.
            - output_mode (str): Output mode for the masks (e.g., 'binary_mask').
            - device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        list: A list of generated masks sorted by area, with each mask containing its properties like bbox and area.
    """
    # Default parameters
    default_params = {
        'points_per_side': 32,
        'pred_iou_thresh': 0.9,
        'stability_score_thresh': 0.95,
        'min_mask_region_area': 500,
        'output_mode': 'binary_mask',
        'device': 'cpu'
    }

    # Default filtering parameters
    if eps_values is None:
        eps_values = np.linspace(100, 1200, 11)
    if min_samples_values is None:
        min_samples_values = range(1, 5)
    
    # Override default parameters with user-provided ones
    params = {**default_params, **kwargs}

    # Load and optionally preprocess the image
    if compress:
        print("Compressing the image...")
        image = preprocess_image(image_path)
        print("Image is compressed.")
    else:
        image = np.array(Image.open(image_path))

    # Cache file for storing generated masks
    cache_file = f"{os.path.splitext(image_path)[0]}_masks.pkl"

    # Check if masks have already been generated for this image
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            sorted_masks = pickle.load(f)
        print("Loaded cached masks.")
    else:
        # Initialize the SAM model
        print("Initializing SAM model...")
        model_type = "vit_h"
        checkpoint = "../models/sam_vit_h_4b8939.pth"  # Update with your checkpoint path
        sam = sam_model_registry[model_type](checkpoint)
        sam.to(device=params['device'])
        print("SAM model was initialized successfully.")

        # Initialize mask generator with SAM
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=params['points_per_side'],
            pred_iou_thresh=params['pred_iou_thresh'],
            stability_score_thresh=params['stability_score_thresh'],
            min_mask_region_area=params['min_mask_region_area'],
            output_mode=params['output_mode']
        )

        # Generate masks
        print("Generating masks...")
        generated_masks = mask_generator.generate(image)
        print("Masks were generated.")

        # Sort masks by area
        sorted_masks = sorted(generated_masks, key=lambda x: x['area'], reverse=True)
        
        # Save masks to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(sorted_masks, f)
        print("Generated and cached masks.")

    # Optionally filter masks
    if apply_filtering:
        print("Filtering masks...")
        sorted_masks, chosen_params = filtering(sorted_masks, eps_values, min_samples_values)
        print(f"Filtering completed with chosen parameters: {chosen_params}")

    sorted_mask_ids = {id(mask): idx for idx, mask in enumerate(sorted_masks)}

    def plot_masks(ax, masks, edge_color):
        for mask in masks:
            bbox = mask['bbox']  # Assuming bbox is [x_min, y_min, width, height]
            rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor=edge_color, facecolor='none')
            ax.add_patch(rect)
            mask_id = sorted_mask_ids[id(mask)]
            ax.text(bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, f"{mask_id}",
                    color='yellow', ha='center', va='center', fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.5))

    def display_mask(index):
        sorted_by_area = sorted(sorted_masks, key=lambda x: x['area'])
        mask = sorted_by_area[index]
        bbox = mask['bbox']

        # Create a subplot for the mask
        fig, ax = plt.subplots(figsize=(5, 5))

        # Display the image
        ax.imshow(image, cmap='gray')
        ax.set_title(f'Mask ID: {sorted_mask_ids[id(mask)]}\nArea: {mask["area"]} pixels')

        # Draw the bounding box for the mask
        rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        # Zoom into the mask area
        ax.set_xlim([bbox[0], bbox[0] + bbox[2]])
        ax.set_ylim([bbox[1] + bbox[3], bbox[1]])

        plt.show()

    # Plot the first figure: image with masks
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(image, cmap='gray')
    ax.set_title('Image with Identified Masks' if not filtering else 'Image with Filtered Masks')
    plot_masks(ax, sorted_masks, 'blue')
    plt.show()

    # Interactive slider for browsing masks
    interact(display_mask, index=IntSlider(min=0, max=len(sorted_masks) - 1, step=1, value=0))

    return sorted_masks