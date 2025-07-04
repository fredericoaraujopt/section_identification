def export_mask_coordinates(image_path, new_masks, stored_masks, fiducials, visualize=False, sample_points=None):
    """
    Export the contour coordinates of new_masks and stored_masks and fiducial information to a CSV file.
    
    For each mask (from new_masks and stored_masks):
      - Uses the 'segmentation' field to compute the entire contour coordinates.
      - Assigns a unique ID (e.g. "new_1", "stored_1", etc.)
      
    For fiducials:
      - Accepts any number of fiducial coordinate pairs.
      - Computes all pairwise Euclidean distances and stores them.
    
    The CSV file is saved in a directory:
         file_directory = f"{os.path.splitext(image_path)[0]}_files"
    with a filename such as:
         {os.path.basename(os.path.splitext(image_path)[0])}.csv
         
    If visualize=True, the function will display:
      - The image with overlaid contours (drawn with thick borders).
      - Markers at each fiducial location.

    If sample_points is not None, the function will store only the specified number of equally spaced points per contour.
      - This is useful to reduce the number of points in the CSV even though at the cost of precise shape of the polygon.
    
    Returns the export file path.
    """
    import os
    import csv
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # Create the output directory.
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    file_directory = f"{os.path.splitext(image_path)[0]}_files"
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)
    export_file = os.path.join(file_directory, f"{base_name}_mask_coordinates.csv")
    
    def get_contours(segmentation):
        """
        Given a segmentation output (from ONNX or SAM), squeeze and threshold
        it (if needed) and return a list of contours.
        Each contour is represented as a list of (x,y) points.
        """
        seg = np.squeeze(segmentation)
        # If not already a binary mask, threshold it.
        if seg.dtype != np.uint8:
            seg = (seg > 0).astype(np.uint8)
        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_list = []
        for cnt in contours:
            if sample_points is not None and len(cnt) > sample_points:
                # Reducing number of points to sample_points equally spaced points.
                total_points = len(cnt)
                indices = np.round(np.linspace(0, total_points - 1, sample_points)).astype(int)
                sampled_cnt = cnt[indices]
                points = sampled_cnt.reshape(-1, 2).tolist()
            else:
                points = cnt.reshape(-1, 2).tolist()
            contour_list.append(points)
        return contour_list

    def compute_pairwise_distances(fiducials):
        """
        Given a list of fiducial coordinate pairs [(x,y), ...], compute all pairwise Euclidean distances.
        Returns a dictionary with keys in the format "fiducial_i-fiducial_j" and the corresponding distance.
        """
        distances = {}
        n = len(fiducials)
        for i in range(n):
            for j in range(i+1, n):
                pt1 = np.array(fiducials[i], dtype=float)
                pt2 = np.array(fiducials[j], dtype=float)
                d = np.linalg.norm(pt1 - pt2)
                distances[f"fiducial_{i+1}-fiducial_{j+1}"] = d
        return distances

    rows = []
    global_idx = 1

   # Process new masks.
    for idx, mask in enumerate(new_masks, start=1):
        contours = get_contours(mask["segmentation"])
        row = {
            "id": f"section_{global_idx}",
            "type": "new_mask",
            "contour_coordinates": str(contours),
            "distance": ""  # Not applicable for masks.
        }
        rows.append(row)
        global_idx += 1

    # Process stored masks.
    for idx, mask in enumerate(stored_masks, start=1):
        contours = get_contours(mask["segmentation"])
        row = {
            "id": f"section_{global_idx}",
            "type": "stored_mask",
            "contour_coordinates": str(contours),
            "distance": ""
        }
        rows.append(row)
        global_idx += 1

    # Process fiducials.
    fiducials_row = {
        "id": "fiducials",
        "type": "fiducials",
        "contour_coordinates": str(fiducials)
    }
    # Compute pairwise distances if there is at least 2 fiducials.
    if isinstance(fiducials, (list, tuple)) and len(fiducials) >= 2:
        distances = compute_pairwise_distances(fiducials)
        fiducials_row["distance"] = str(distances)
    else:
        fiducials_row["distance"] = ""
    rows.append(fiducials_row)

    # Write rows to CSV.
    fieldnames = ["id", "type", "contour_coordinates", "distance"]
    with open(export_file, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Exported mask coordinates to {export_file}")

    # Optional visualization step.
    if visualize:
        # Load image using cv2 (BGR) and convert to RGB.
        img = cv2.imread(image_path)
        if img is None:
            print("Could not load image for visualization.")
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vis_img = img_rgb.copy()
            # Draw contours from both new and stored masks.
            for mask in new_masks + stored_masks:
                contours = get_contours(mask["segmentation"])
                # Draw each contour with a thick border.
                for cnt in contours:
                    cnt_np = np.array(cnt, dtype=np.int32)
                    cv2.drawContours(vis_img, [cnt_np], -1, (0, 0, 255), 5)
            # Draw fiducials.
            if isinstance(fiducials, (list, tuple)):
                for pt in fiducials:
                    pt = (int(pt[0]), int(pt[1]))
                    cv2.circle(vis_img, pt, 10, (255, 0, 0), -1)  # blue filled circle
            # Optionally, draw lines between fiducials to visualize distances.
            if isinstance(fiducials, (list, tuple)) and len(fiducials) >= 2:
                for i in range(len(fiducials)):
                    for j in range(i+1, len(fiducials)):
                        pt1 = (int(fiducials[i][0]), int(fiducials[i][1]))
                        pt2 = (int(fiducials[j][0]), int(fiducials[j][1]))
                        cv2.line(vis_img, pt1, pt2, (255, 0, 255), 2)  # magenta line
            # Display the visualization using matplotlib.
            plt.figure(figsize=(10, 10))
            plt.imshow(vis_img)
            plt.title("Exported Masks and Fiducials")
            plt.axis("off")
            plt.show()

    return export_file