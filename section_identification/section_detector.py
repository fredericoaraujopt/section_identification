import numpy as np

def section_detector(image: np.array) -> np.array:
    """Generate instance segmentation of an optical image of a wafer.

    Args:
        image (np.array): Optical image of wafer of shape (H, W).

    Returns:
        np.array: Instance segmentation of shape (H,W) where value of 0 means background, and nonzero values indicate the instance ID of the section.
    """
    # Add segmentation code here
    segmentation = np.zeros(image.shape, dtype=np.uint16)

    return segmentation