import cv2
import math
import numpy as np
from PIL import Image
import os

def run_ruler(image_path, microscope_magnification, original_length=1):
    """
    Ruler function to measure objects in an image.
    
    Parameters:
      image_path: Path to the input image (PNG).
      microscope_magnification: A scaling factor from the microscope (e.g. 0.75 for 0.75x).
      original_length: A reference length in millimeters.
    
    How it works:
      1. Reads the PNG metadata (DPI) via PIL to determine mm per pixel.
      2. Computes a conversion factor as: (25.4 / dpi) * microscope_magnification.
      3. Opens an interactive OpenCV window where:
         - The first left-click sets the start point.
         - As you move the mouse, a dynamic green line is drawn from the start point to the current mouse position.
         - A second left-click fixes the measurement, changes the line color to red, and displays the measured
           distance (in mm) and the ratio of this measurement to the original length.
      4. Supports multiple measurements in one session.
    
    Returns:
      A list of measurements. Each measurement is a tuple:
      (start_point, end_point, measured_distance_mm, compression_ratio)
    """
    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Calibrated using Fiji measurements of the microscope reticle
    # Calibrated using Fiji measurements of the microscope reticle (at 1.0x magnification)
    pixels_per_mm_calibrated = 125.28
    effective_pixels_per_mm = pixels_per_mm_calibrated * microscope_magnification
    conversion_factor = 1.0 / effective_pixels_per_mm
    print(f"[Debug] Using calibrated scale: effective_pixels_per_mm = {effective_pixels_per_mm:.2f}, conversion_factor = {conversion_factor:.6f} mm/pixel")
    
    # Variables to hold measurement data
    measurements = []  # List of finalized measurements.
    current_start = None  # Starting point of a measurement.
    dynamic_point = None  # Current mouse position for the in-progress measurement.
    
    zoom_factor = 1.0
    roi_x, roi_y = 0, 0
    anchor = None
    last_cursor = None

    # Mouse callback function to capture clicks and mouse movement.
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_start, dynamic_point, measurements, zoom_factor, roi_x, roi_y, anchor, last_cursor
        # Convert display coordinates to original image coordinates using the ROI offset and zoom factor
        orig_x = int(x / zoom_factor + roi_x)
        orig_y = int(y / zoom_factor + roi_y)
        # Update the last known cursor position
        last_cursor = (orig_x, orig_y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if current_start is None:
                # Start a new measurement.
                current_start = (orig_x, orig_y)
            else:
                # Complete the measurement.
                end_point = (orig_x, orig_y)
                # Calculate Euclidean distance (in pixels) between the two points.
                pixel_distance = math.hypot(end_point[0] - current_start[0],
                                            end_point[1] - current_start[1])
                measured_mm = pixel_distance * conversion_factor
                compression_ratio = measured_mm / original_length
                measurements.append((current_start, end_point, measured_mm, compression_ratio))
                # Reset the current measurement.
                current_start = None
                dynamic_point = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if current_start is not None:
                dynamic_point = (orig_x, orig_y)
    
    window_name = "Ruler Measurement (Press ESC to exit)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Main interactive loop.
    while True:
        height, width = image.shape[:2]
        # Use the anchor (cursor position) if available; otherwise, default to image center
        if anchor is None:
            anchor = (width // 2, height // 2)
        anchor_x, anchor_y = anchor
        roi_width = int(width / zoom_factor)
        roi_height = int(height / zoom_factor)
        roi_x = anchor_x - roi_width // 2
        roi_y = anchor_y - roi_height // 2
        # Clamp the ROI to image boundaries
        if roi_x < 0:
            roi_x = 0
        if roi_y < 0:
            roi_y = 0
        if roi_x + roi_width > width:
            roi_x = width - roi_width
        if roi_y + roi_height > height:
            roi_y = height - roi_height
        roi = image[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
        display_img = cv2.resize(roi, (width, height), interpolation=cv2.INTER_LINEAR)

        # Draw all fixed (finalized) measurements.
        for (p1, p2, measured_mm, compression_ratio) in measurements:
            p1_zoom = (int((p1[0] - roi_x) * zoom_factor), int((p1[1] - roi_y) * zoom_factor))
            p2_zoom = (int((p2[0] - roi_x) * zoom_factor), int((p2[1] - roi_y) * zoom_factor))
            # Draw finalized measurement arrowed lines in red (arrowheads at both endpoints).
            cv2.arrowedLine(display_img, p1_zoom, p2_zoom, (0, 0, 255), 2, tipLength=0.2)
            cv2.arrowedLine(display_img, p2_zoom, p1_zoom, (0, 0, 255), 2, tipLength=0.2)
            # Compute the midpoint for displaying text.
            mid_point_zoom = ((p1_zoom[0] + p2_zoom[0]) // 2, (p1_zoom[1] + p2_zoom[1]) // 2)
            text = f"{measured_mm:.2f} mm, {compression_ratio:.2f}x"
            cv2.putText(display_img, text, mid_point_zoom, cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 2)
        # Draw the dynamic line (in-progress measurement) in green.
        if current_start is not None and dynamic_point is not None:
            current_start_zoom = (int((current_start[0] - roi_x) * zoom_factor), int((current_start[1] - roi_y) * zoom_factor))
            dynamic_point_zoom = (int((dynamic_point[0] - roi_x) * zoom_factor), int((dynamic_point[1] - roi_y) * zoom_factor))
            cv2.arrowedLine(display_img, current_start_zoom, dynamic_point_zoom, (0, 255, 0), 2, tipLength=0.2)
            cv2.arrowedLine(display_img, dynamic_point_zoom, current_start_zoom, (0, 255, 0), 2, tipLength=0.2)
        
        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit.
            break
        elif key == ord('='):
            # Set the zoom anchor to the last cursor position if available
            if last_cursor is not None:
                anchor = last_cursor
            zoom_factor = min(zoom_factor * 1.2, 10.0)  # Zoom in, capped at 10x.
        elif key == ord('-'):
            # Set the zoom anchor to the last cursor position if available
            if last_cursor is not None:
                anchor = last_cursor
            zoom_factor = max(zoom_factor / 1.2, 0.1)   # Zoom out, capped at 0.1x.
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print("[Info] Exiting ruler interface.")
    return measurements