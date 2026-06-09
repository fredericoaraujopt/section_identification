"""Export detected section polygons + fiducials.

Outputs (all keyed off ``image_path``'s ``*_files`` directory):
  * CSV   — the original ``id,type,contour_coordinates,distance`` table.
  * GeoJSON — polygons/points in full-resolution pixels (+ stage µm if known).
  * CZI   — a copy of the source CZI with ZEN ``<Layers>`` annotations
            (only when the source is a ``.czi``; see :mod:`czi_export`).

Polygons are simplified with Douglas–Peucker (``cv2.approxPolyDP``) so ZEN and
downstream tools receive clean, low-vertex shapes, and are mapped from the
downscaled detection overview back to full-resolution pixels via the
:class:`~section_identification.czi_io.CziGeometry` handle.
"""

import csv
import os

import numpy as np


# --------------------------------------------------------------------------- #
# Mask -> polygon
# --------------------------------------------------------------------------- #
def contours_from_mask(segmentation, simplify_eps=1.5, sample_points=None):
    """Binary mask -> list of contours, each an ``Nx2`` int array (x, y).

    ``simplify_eps`` is the Douglas–Peucker tolerance in pixels (0 disables).
    """
    import cv2

    seg = np.squeeze(segmentation)
    if seg.dtype != np.uint8:
        seg = (seg > 0).astype(np.uint8)
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        if simplify_eps and simplify_eps > 0:
            cnt = cv2.approxPolyDP(cnt, simplify_eps, True)
        pts = cnt.reshape(-1, 2)
        if sample_points is not None and len(pts) > sample_points:
            idx = np.round(np.linspace(0, len(pts) - 1, sample_points)).astype(int)
            pts = pts[idx]
        out.append(pts)
    return out


def mask_to_polygon(segmentation, simplify_eps=1.5, sample_points=None):
    """Return the single largest external contour of a mask as ``Nx2`` (x, y)."""
    contours = contours_from_mask(segmentation, simplify_eps, sample_points)
    if not contours:
        return None
    return max(contours, key=lambda c: len(c))


def scale_polygon(poly, geom):
    """Map an overview-pixel polygon to full-resolution pixels via ``geom``."""
    if geom is None or poly is None:
        return None if poly is None else np.asarray(poly, dtype=float)
    p = np.asarray(poly, dtype=float).reshape(-1, 2)
    fx, fy = geom.ds_to_full(p[:, 0], p[:, 1])
    return np.column_stack([fx, fy])


def masks_to_polygons(masks, geom=None, simplify_eps=1.5, sample_points=None):
    """Detected masks -> list of full-resolution polygons (one per mask)."""
    polys = []
    for m in masks:
        poly = mask_to_polygon(m["segmentation"], simplify_eps, sample_points)
        if poly is None or len(poly) < 3:
            continue
        polys.append(scale_polygon(poly, geom) if geom is not None
                     else np.asarray(poly, dtype=float))
    return polys


def scale_points(points, geom):
    """Map overview-pixel points (fiducials) to full-resolution pixels."""
    if not points:
        return []
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if geom is None:
        return [tuple(map(float, xy)) for xy in p]
    fx, fy = geom.ds_to_full(p[:, 0], p[:, 1])
    return [(float(a), float(b)) for a, b in zip(fx, fy)]


def compute_pairwise_distances(fiducials):
    """All pairwise Euclidean distances between fiducial points."""
    distances = {}
    n = len(fiducials)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(np.subtract(fiducials[i], fiducials[j])))
            distances[f"fiducial_{i + 1}-fiducial_{j + 1}"] = d
    return distances


# --------------------------------------------------------------------------- #
# Top-level export
# --------------------------------------------------------------------------- #
def export_mask_coordinates(image_path, new_masks, stored_masks, fiducials,
                            geom=None, visualize=False, sample_points=None,
                            simplify_eps=1.5, write_czi=True, section_ids=None):
    """Export polygons + fiducials (from masks) to CSV + GeoJSON (+ annotated CZI).

    Coordinates are written in full-resolution pixels when ``geom`` is provided.
    Returns a dict of output file paths.
    """
    all_masks = list(stored_masks or []) + list(new_masks or [])
    polygons = masks_to_polygons(all_masks, geom=geom, simplify_eps=simplify_eps,
                                 sample_points=sample_points)
    fiducials_full = scale_points(fiducials, geom)
    return _write_outputs(image_path, polygons, fiducials_full,
                          section_ids=section_ids, visualize=visualize,
                          write_czi=write_czi, geom=geom)


def export_polygons(image_path, polygons, fiducials, geom=None, visualize=False,
                    write_czi=True, section_ids=None):
    """Export already-extracted polygons (e.g. user-edited in the GUI).

    ``polygons`` / ``fiducials`` are in overview-pixel coords; they are scaled to
    full-resolution via ``geom`` before writing.
    """
    polys_full = [scale_polygon(p, geom) for p in polygons if p is not None
                  and len(np.asarray(p).reshape(-1, 2)) >= 3]
    fiducials_full = scale_points(fiducials, geom)
    return _write_outputs(image_path, polys_full, fiducials_full,
                          section_ids=section_ids, visualize=visualize,
                          write_czi=write_czi, geom=geom)


def _write_outputs(image_path, polygons, fiducials_full, section_ids=None,
                   visualize=False, write_czi=True, geom=None):
    """Shared writer: polygons + fiducials already in full-res pixels -> files."""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    file_directory = f"{os.path.splitext(image_path)[0]}_files"
    os.makedirs(file_directory, exist_ok=True)

    polygons = [np.asarray(p, dtype=float) for p in polygons]
    if section_ids is None:
        section_ids = [f"section_{i}" for i in range(1, len(polygons) + 1)]

    # ---- CSV (kept for backward compatibility) ----
    csv_path = os.path.join(file_directory, f"{base_name}_mask_coordinates.csv")
    rows = []
    for sid, poly in zip(section_ids, polygons):
        rows.append({"id": sid, "type": "section",
                     "contour_coordinates": str([poly.tolist()]), "distance": ""})
    fid_row = {"id": "fiducials", "type": "fiducials",
               "contour_coordinates": str([list(p) for p in fiducials_full])}
    fid_row["distance"] = (str(compute_pairwise_distances(fiducials_full))
                           if len(fiducials_full) >= 2 else "")
    rows.append(fid_row)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "type",
                                               "contour_coordinates", "distance"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV: {csv_path}")

    # ---- GeoJSON ----
    from section_identification import czi_export
    geojson_path = os.path.join(file_directory, f"{base_name}_sections.geojson")
    czi_export.write_geojson(geojson_path, polygons, fiducials_full, geom=geom)
    print(f"Wrote GeoJSON: {geojson_path}")

    outputs = {"csv": csv_path, "geojson": geojson_path}

    # ---- Annotated CZI ----
    from section_identification import czi_io
    if write_czi and czi_io.is_czi(image_path):
        dst = os.path.join(os.path.dirname(image_path), f"{base_name}_STiM.czi")
        try:
            report = czi_export.write_annotated_czi(
                image_path, dst, [p.tolist() for p in polygons], fiducials_full,
                section_ids=section_ids)
            outputs["czi"] = report["dst"]
            print(f"Wrote annotated CZI: {report['dst']} "
                  f"(round-trip ok: {report['roundtrip_ok']})")
        except Exception as e:  # don't lose CSV/GeoJSON if CZI write fails
            print(f"[warn] annotated-CZI write failed: {e}")

    if visualize:
        _visualize(image_path, polygons, fiducials_full)

    return outputs


def _visualize(image_path, polygons, fiducials):
    import cv2
    import matplotlib.pyplot as plt
    from section_identification import czi_io

    if czi_io.is_czi(image_path):
        # Visualising the full CZI overview is handled by the GUI / run script.
        return
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    for poly in polygons:
        cv2.drawContours(img, [np.asarray(poly, dtype=np.int32)], -1, (0, 0, 255), 3)
    for x, y in fiducials:
        cv2.circle(img, (int(x), int(y)), 10, (255, 0, 0), -1)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title("Exported sections and fiducials")
    plt.axis("off")
    plt.show()
