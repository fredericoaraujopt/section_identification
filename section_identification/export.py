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
import hashlib
import os
import tempfile

import numpy as np

from . import atomicio


# --------------------------------------------------------------------------- #
# Output directory (read-only-drive safe)
# --------------------------------------------------------------------------- #
def resolve_export_dir(image_path, out_dir=None):
    """Return a WRITABLE directory for the run's outputs.

    Preference order:
      1. an explicit ``out_dir`` (the GUI may pass one),
      2. ``<image>_files`` next to the source (the historical location),
      3. ``~/STiM_exports/<image-stem>`` — used when the source sits on a
         read-only volume (macOS mounts NTFS drives read-only, which is exactly
         what killed the M411 export), so a long run's results are never lost,
      4. a temp dir, as a last resort.

    Each candidate is probed by actually writing a tiny file (``os.access`` lies
    on some network/NTFS mounts), and the first that succeeds is returned. The
    chosen dir is created.
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    # The fallback dirs are keyed on the basename only, so two sources with the
    # same name in different folders (or two drives) would resolve to the SAME
    # fallback and silently overwrite each other — and on a read-only source the
    # fallback is the only place results land. Disambiguate with a short, stable
    # hash of the absolute source path (same source → same dir, so re-exports are
    # still idempotent).
    tag = hashlib.sha1(os.path.abspath(image_path).encode("utf-8", "replace")).hexdigest()[:8]
    base_tagged = f"{base}_{tag}"
    candidates = []
    if out_dir:
        candidates.append(out_dir)
    candidates.append(f"{os.path.splitext(image_path)[0]}_files")
    candidates.append(os.path.join(os.path.expanduser("~"), "STiM_exports", base_tagged))
    candidates.append(os.path.join(tempfile.gettempdir(), "STiM_exports", base_tagged))

    last_err = None
    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, ".stim_write_test")
            with open(probe, "w"):
                pass
        except OSError as e:
            last_err = e
            continue
        # The write succeeded → this dir is usable for the real outputs. Removing
        # the probe is best-effort: a mount that allows create but denies unlink
        # must not make us abandon a perfectly writable folder.
        try:
            os.remove(probe)
        except OSError:
            pass
        return d
    # Every candidate failed (extremely unlikely); surface the real reason.
    raise OSError(f"No writable export directory found for {image_path!r}: {last_err}")


# --------------------------------------------------------------------------- #
# Mask -> polygon
# --------------------------------------------------------------------------- #
def decode_segmentation(segmentation):
    """Return a 2D uint8 mask from either a binary array or an RLE dict.

    SAM masks may be stored as full binary arrays (``output_mode='binary_mask'``)
    or, to keep the cache small, as run-length-encoded dicts
    (``'coco_rle'``/``'uncompressed_rle'``). This decodes either to an ``HxW``
    array, decoding only one mask at a time (bounded memory).
    """
    if isinstance(segmentation, dict):
        from pycocotools import mask as mask_utils
        seg = dict(segmentation)
        counts = seg.get("counts")
        if isinstance(counts, str):  # coco_rle stores counts as a str for JSON
            seg["counts"] = counts.encode("utf-8")
        return mask_utils.decode(seg)
    return np.squeeze(np.asarray(segmentation))


def contours_from_mask(segmentation, simplify_eps=1.5, sample_points=None):
    """Binary/RLE mask -> list of contours, each an ``Nx2`` int array (x, y).

    ``simplify_eps`` is the Douglas–Peucker tolerance in pixels (0 disables).
    """
    import cv2

    seg = decode_segmentation(segmentation)
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
                            simplify_eps=1.5, write_czi=True, section_ids=None,
                            out_dir=None):
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
                          write_czi=write_czi, geom=geom, out_dir=out_dir)


def export_polygons(image_path, polygons, fiducials, geom=None, visualize=False,
                    write_czi=True, section_ids=None, write_csv=True,
                    write_geojson=True, write_png=True, out_dir=None,
                    write_sf=False):
    """Export already-extracted polygons (e.g. user-edited in the GUI).

    ``polygons`` / ``fiducials`` are in overview-pixel coords; they are scaled to
    full-resolution via ``geom`` before writing. Each ``write_*`` flag selects
    which outputs to produce. ``out_dir`` overrides where files land (falls back
    to a writable location automatically — see :func:`resolve_export_dir`).
    ``write_sf`` also writes the fiducials into the annotated CZI's ZEN Shuttle &
    Find calibration markers (implies ``write_czi``; needs a CZI source + ``geom``).
    """
    polys_full = [scale_polygon(p, geom) for p in polygons if p is not None
                  and len(np.asarray(p).reshape(-1, 2)) >= 3]
    fiducials_full = scale_points(fiducials, geom)
    return _write_outputs(image_path, polys_full, fiducials_full,
                          section_ids=section_ids, visualize=visualize,
                          write_czi=write_czi, geom=geom, write_csv=write_csv,
                          write_geojson=write_geojson, write_png=write_png,
                          out_dir=out_dir, write_sf=write_sf)


def _write_outputs(image_path, polygons, fiducials_full, section_ids=None,
                   visualize=False, write_czi=True, geom=None,
                   write_png=True, png_long_side=16384,
                   write_csv=True, write_geojson=True, out_dir=None,
                   write_sf=False):
    """Shared writer: polygons + fiducials already in full-res pixels -> files.
    Each output is gated by its ``write_*`` flag (customizable export)."""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    file_directory = resolve_export_dir(image_path, out_dir)

    polygons = [np.asarray(p, dtype=float) for p in polygons]
    if section_ids is None:
        section_ids = [f"section_{i}" for i in range(1, len(polygons) + 1)]
    outputs = {"dir": file_directory}

    # ---- CSV ----
    if write_csv:
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
        def _write_csv(tmp):
            with open(tmp, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "type",
                                                       "contour_coordinates", "distance"])
                writer.writeheader()
                writer.writerows(rows)
        atomicio.atomic_write(csv_path, _write_csv)
        print(f"Wrote CSV: {csv_path}")
        outputs["csv"] = csv_path

    # ---- GeoJSON ----
    if write_geojson:
        from section_identification import czi_export
        geojson_path = os.path.join(file_directory, f"{base_name}_sections.geojson")
        czi_export.write_geojson(geojson_path, polygons, fiducials_full, geom=geom)
        print(f"Wrote GeoJSON: {geojson_path}")
        outputs["geojson"] = geojson_path

    # ---- High-resolution overlay PNG (polygons + fiducial crosses) ----
    if write_png:
        png_path = os.path.join(file_directory, f"{base_name}_overlay.png")
        try:
            render_overlay_png(image_path, polygons, fiducials_full, png_path,
                               target_long_side=png_long_side)
            outputs["png"] = png_path
            print(f"Wrote overlay PNG: {png_path}")
        except Exception as e:  # don't lose other outputs if the PNG fails
            print(f"[warn] overlay PNG failed: {e}")

    # ---- Annotated CZI ----
    from section_identification import czi_io
    if (write_czi or write_sf) and czi_io.is_czi(image_path):
        # Write the (full-size) annotated copy into the resolved output dir, not
        # next to the source — the source may be on a read-only drive, and that
        # is also where a 16 GB copy would otherwise land.
        dst = os.path.join(file_directory, f"{base_name}_STiM.czi")
        # When requested, also map the fiducials into stage µm for the ZEN
        # Shuttle & Find calibration markers (only the COPY is edited).
        sf_markers = sf_orient = None
        if write_sf:
            if geom is None or not fiducials_full:
                print("[warn] S&F markers requested but there are no fiducials / no "
                      "geometry; CZI written without S&F markers.")
            else:
                sm = geom.full_to_stage_um(
                    np.asarray([p[0] for p in fiducials_full], dtype=float),
                    np.asarray([p[1] for p in fiducials_full], dtype=float))
                if sm is not None:
                    sf_markers = list(zip(np.atleast_1d(sm[0]).tolist(),
                                          np.atleast_1d(sm[1]).tolist()))
                    # Markers are absolute stage µm (physically correct via the
                    # verified transform); leave the CZI's own <StageOrientation>
                    # untouched (sf_orient=None) rather than writing STiM's
                    # internal corrected signs into ZEN's calibration metadata.
                    sf_orient = None
                else:
                    print("[warn] S&F markers requested but the CZI lacks a stage "
                          "anchor (scene CenterPosition / multi-scene); skipping S&F write.")
        try:
            report = czi_export.write_annotated_czi(
                image_path, dst, [p.tolist() for p in polygons], fiducials_full,
                section_ids=section_ids, sf_markers_stage_um=sf_markers,
                sf_orientation=sf_orient)
            outputs["czi"] = report["dst"]
            print(f"Wrote annotated CZI: {report['dst']} "
                  f"(round-trip ok: {report['roundtrip_ok']}"
                  + (f", S&F markers: {report['n_sf_markers']}" if sf_markers else "")
                  + ")")
        except Exception as e:  # don't lose CSV/GeoJSON if CZI write fails
            print(f"[warn] annotated-CZI write failed: {e}")

    if visualize:
        _visualize(image_path, polygons, fiducials_full)

    return outputs


def render_overlay_png(image_path, polygons_full, fiducials_full, out_path,
                       target_long_side=16384, section_color=(0, 0, 255),
                       fiducial_color=(0, 255, 255)):
    """Render a high-resolution RGB PNG of the wafer with section polygons and
    fiducial CROSS markers overlaid.

    ``polygons_full`` / ``fiducials_full`` are FULL-resolution pixel coords. The
    wafer is read at ``target_long_side`` (a true-native 76k-px PNG would be
    ~18 GB in RAM, so we render at a high but memory-safe size; the annotated CZI
    carries the genuine full-resolution geometry). Colors are BGR (cv2).
    Returns ``out_path``.
    """
    import cv2
    from section_identification import czi_io

    if czi_io.is_czi(image_path):
        arr, geom_r, _ = czi_io.read_czi_overview(image_path, target_long_side=target_long_side)
        img = czi_io.to_rgb8(arr)              # HxWx3 uint8 (gray replicated)
        to_render = geom_r.full_to_ds          # full-res px -> this render's px
    else:
        from PIL import Image
        img = np.array(Image.open(image_path).convert("RGB"))
        to_render = lambda xs, ys: (np.asarray(xs, float), np.asarray(ys, float))

    img = np.ascontiguousarray(img)
    H, W = img.shape[:2]
    th = max(1, int(round(max(H, W) / 9000)))           # fine polygon line thickness
    arm = max(18, int(round(max(H, W) / 110)))          # fiducial cross half-length

    for poly in polygons_full:
        p = np.asarray(poly, dtype=float).reshape(-1, 2)
        xr, yr = to_render(p[:, 0], p[:, 1])
        pts = np.column_stack([xr, yr]).round().astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [pts], True, section_color, th, lineType=cv2.LINE_AA)

    for (fx, fy) in (fiducials_full or []):
        xr, yr = to_render(fx, fy)
        cx, cy = int(round(float(np.asarray(xr)))), int(round(float(np.asarray(yr))))
        cv2.line(img, (cx - arm, cy), (cx + arm, cy), fiducial_color, th + 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy - arm), (cx, cy + arm), fiducial_color, th + 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), arm, fiducial_color, max(2, th), cv2.LINE_AA)

    # Atomic: encode to a temp with the real extension (cv2 infers format from it),
    # then rename in — a killed render can't leave a truncated PNG at out_path.
    tmp = out_path + ".part.png"
    try:
        if not cv2.imwrite(tmp, img):
            raise IOError(f"cv2.imwrite failed for {tmp}")
        os.replace(tmp, out_path)
    except BaseException:
        try:
            if os.path.lexists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise
    return out_path


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
