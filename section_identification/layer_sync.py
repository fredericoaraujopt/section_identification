"""Render WaferProject results onto napari layers (the visually-guided overlays).

Each stage shows its actual computation on the wafer: QC colours the Sections
polygons by severity; ROIs draws the propagated regions; reorder draws SIFT match
lines (coloured by confidence) and the recovered serial chain; imaging draws the
TSP route as directed vectors plus per-section order numbers. Overlay layers use
the SAME scale as the Sections layer so they register with the full-res wafer,
and are glyph-prefixed (①②③④) so they group in the layer list.

All functions are defensive (remove-and-re-add by name; swallow per-layer errors)
so a visualisation hiccup can't take down the GUI.
"""

from __future__ import annotations

import math

import numpy as np

QC_LAYER = None  # QC recolours the existing Sections layer in place
ROI_LAYER = "② ROIs"
MATCH_LAYER = "③ SIFT matches"
CHAIN_LAYER = "③ Serial order"
ROUTE_LAYER = "④ Route"
ORDER_LAYER = "④ Imaging order"


def _xy_to_yx(poly):
    p = np.asarray(poly, float).reshape(-1, 2)
    return p[:, ::-1]


def _remove(viewer, name):
    if viewer is not None and name in viewer.layers:
        try:
            viewer.layers.remove(name)
        except Exception:
            pass


def _centroid_yx(section):
    cx, cy = section.centroid()
    return (cy, cx)


def _overlay_width(app, frac: float = 0.012, lo: float = 0.5, hi: float = 8.0):
    """A thin, consistent overlay edge width (data units) scaled to the median
    section size, so outlines stay ~1 px on screen regardless of overview px."""
    sizes = []
    for s in app.project.sections:
        x0, y0, x1, y1 = s.bbox()
        sizes.append(max(x1 - x0, y1 - y0))
    if not sizes:
        return 1.0
    med = sorted(sizes)[len(sizes) // 2]
    return float(min(max(med * frac, lo), hi))


ORIENT_LAYER = "Orientation"


def show_orientation(app):
    """Draw, per section, an arrow from its centre along the recovered canonical
    ('up') axis — the visual interpretation of unifying section orientations."""
    viewer = app.viewer
    _remove(viewer, ORIENT_LAYER)
    app.ensure_poses()
    vecs = []
    for s in app.project.sections:
        if s.pose.center is None:
            continue
        cx, cy = s.pose.center
        ang = np.radians(s.pose.angle_deg)
        x0, y0, x1, y1 = s.bbox()
        L = 0.45 * max(x1 - x0, y1 - y0)
        vecs.append([[cy, cx], [float(np.sin(ang) * L), float(np.cos(ang) * L)]])
    if not vecs:
        app.log("proofread", "no sections to orient.")
        return
    try:
        viewer.add_vectors(np.asarray(vecs, float), name=ORIENT_LAYER,
                           edge_color="yellow", vector_style="arrow",
                           edge_width=_overlay_width(app), scale=app.layer_scale())
        app.log("proofread", f"orientation arrows for {len(vecs)} sections "
                             "(each points along its canonical 'up' axis).")
    except Exception as e:
        app.log("proofread", f"orientation overlay error: {e}")


def clear_orientation(app):
    _remove(app.viewer, ORIENT_LAYER)


# --------------------------------------------------------------------------- #
# QC: recolour the existing Sections layer by severity
# --------------------------------------------------------------------------- #
def apply_qc_colors(app, by: str = "qc_score"):
    layer = getattr(app.gui, "shapes_layer", None)
    if layer is None:
        return
    secs = app.project.sections
    n = len(layer.data)
    if n != len(secs):                # ordering must align; bail if it doesn't
        app.log("qc", f"colour skipped (layer has {n} shapes, model {len(secs)})")
        return
    scores, status, sids = [], [], []
    for s in secs:
        sc = s.qc.scores.get("overall", 0.0) if s.qc else 0.0
        scores.append(float(sc))
        status.append("reject" if (s.qc and s.qc.flags.get("any")) else "accept")
        sids.append(s.id)
    try:
        layer.features = {"qc_score": scores, "qc_status": status, "section_id": sids}
        if by == "qc_status":
            layer.face_color = "qc_status"
            layer.face_color_cycle = ["#33cc55", "#ff5533"]
        else:
            layer.face_color = "qc_score"
            layer.face_colormap = "magma"
        layer.opacity = 0.55
    except Exception as e:
        app.log("qc", f"colour error: {e}")


QC_DIAG_LAYER = "③ QC diagnostic"


def show_qc_diagnostic(app, section):
    """Overlay the feature map that produced a section's dominant QC flag (Frangi
    ridges for folds, bright-outlier mask for debris, components for shredding),
    placed over that section so the user *sees why* it was flagged. Best-effort.
    """
    viewer = app.viewer
    _remove(viewer, QC_DIAG_LAYER)
    if section is None or section.qc is None or not app.has_image():
        return
    try:
        from . import crops, wafer_qc
        flag = wafer_qc.dominant_flag(section.qc.to_dict())
        gray, mask, (x0, y0, cs) = crops.read_section_crop(
            app.image_path, app.geom, section.polygon, overview=app.overview,
            full_res=False, target_long_side=640)
        fm = wafer_qc.feature_maps(gray, mask)
        if flag == "debris":
            mp, cmap = fm["bright"].astype(float), "red"
        elif flag == "shred":
            mp, cmap = (fm["labels"] > 0).astype(float), "cyan"
        else:                                   # fold (default) / chatter -> ridges
            mp, cmap = fm["ridges"].astype(float), "red"
        if mp.max() <= 0:
            return
        # world == full-res px; 1 crop px = 1/cs world; crop origin at (y0, x0).
        viewer.add_image(mp, name=QC_DIAG_LAYER, colormap=cmap, blending="additive",
                         opacity=0.8, scale=(1.0 / cs, 1.0 / cs), translate=(y0, x0))
        app.log("qc", f"{section.id}: dominant flag = {flag}")
    except Exception as e:
        app.log("qc", f"diagnostic overlay error: {e}")


def clear_qc_diagnostic(app):
    _remove(app.viewer, QC_DIAG_LAYER)


def clear_qc_colors(app):
    layer = getattr(app.gui, "shapes_layer", None)
    if layer is None:
        return
    try:
        layer.face_color = "transparent"
        layer.opacity = 1.0
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# ROIs
# --------------------------------------------------------------------------- #
FOCUS_LAYER = "Focus points"


def show_focus_points(app):
    """Show propagated focus support points as an editable Points layer (the user
    can nudge/add/remove them per section natively in napari)."""
    viewer = app.viewer
    _remove(viewer, FOCUS_LAYER)
    pts = [[y, x] for s in app.project.sections for (x, y) in s.focus_overview]
    if not pts:
        return
    try:
        viewer.add_points(np.asarray(pts, float), name=FOCUS_LAYER, size=8,
                          face_color="orange", border_color="orange",
                          scale=app.layer_scale())
    except Exception as e:
        app.log("rois", f"focus overlay error: {e}")


MFOV_LAYER = "mFOV grid"


def show_mfov_grid(app, tile_um: float):
    """Preview the tile grid ZEN will acquire inside each ROI: subdivide each ROI
    bbox into Columns×Rows tiles (from the ROI's stage-µm size / ``tile_um``).
    Single-beam: when the tile ≥ the ROI, that's a single tile (the whole ROI)."""
    viewer = app.viewer
    _remove(viewer, MFOV_LAYER)
    geom = app.geom
    tile = float(tile_um) or 50.0
    rects = []
    for s in app.project.sections:
        if not s.roi or len(s.roi.polygon) < 3:
            continue
        ra = np.asarray(s.roi.polygon, float).reshape(-1, 2)
        x0, y0, x1, y1 = ra[:, 0].min(), ra[:, 1].min(), ra[:, 0].max(), ra[:, 1].max()
        cols = rows = 1
        if geom is not None:
            fx, fy = geom.ds_to_full(ra[:, 0], ra[:, 1])
            su = geom.full_to_stage_um(np.ravel(fx), np.ravel(fy))
            if su is not None:
                sx, sy = np.ravel(su[0]), np.ravel(su[1])
                cols = max(1, math.ceil((sx.max() - sx.min()) / tile))
                rows = max(1, math.ceil((sy.max() - sy.min()) / tile))
        for c in range(cols):
            for r in range(rows):
                gx0 = x0 + (x1 - x0) * c / cols
                gx1 = x0 + (x1 - x0) * (c + 1) / cols
                gy0 = y0 + (y1 - y0) * r / rows
                gy1 = y0 + (y1 - y0) * (r + 1) / rows
                rects.append(np.array([[gy0, gx0], [gy0, gx1], [gy1, gx1], [gy1, gx0]]))
    if not rects:
        app.log("rois", "no ROIs to tile — define + propagate an ROI first.")
        return
    try:
        viewer.add_shapes(rects, shape_type="rectangle", name=MFOV_LAYER,
                          edge_color="yellow", face_color="transparent",
                          edge_width=_overlay_width(app, frac=0.006),
                          scale=app.layer_scale())
        app.log("rois", f"mFOV grid: {len(rects)} tiles across "
                        f"{sum(1 for s in app.project.sections if s.roi)} ROIs.")
    except Exception as e:
        app.log("rois", f"mFOV grid error: {e}")


def clear_mfov_grid(app):
    _remove(app.viewer, MFOV_LAYER)


def show_rois(app):
    viewer = app.viewer
    _remove(viewer, ROI_LAYER)
    polys = [_xy_to_yx(s.roi.polygon) for s in app.project.sections
             if s.roi and len(s.roi.polygon) >= 3]
    if not polys:
        return
    try:
        viewer.add_shapes(polys, shape_type="polygon", name=ROI_LAYER,
                          edge_color="cyan", face_color="transparent",
                          edge_width=_overlay_width(app), scale=app.layer_scale())
    except Exception as e:
        app.log("rois", f"overlay error: {e}")


EXIST_FOCUS = "Focus support points (CZI)"
EXIST_MFOV = "Existing mFOVs (CZI)"


def show_existing_acquisition(app, data):
    """Display focus support points + existing mFOV regions read from a CZI
    (already converted to overview px by czi_export.read_acquisition_overview)."""
    viewer = app.viewer
    _remove(viewer, EXIST_FOCUS)
    _remove(viewer, EXIST_MFOV)
    fp = data.get("focus_points", [])
    if fp:
        pts = np.array([[y, x] for (x, y, _z) in fp])
        zs = [round(float(z), 1) for (_x, _y, z) in fp]
        try:
            viewer.add_points(pts, name=EXIST_FOCUS, size=5, face_color="yellow",
                              scale=app.layer_scale(), features={"z": zs},
                              text={"string": "{z}", "size": 7, "color": "yellow",
                                    "anchor": "upper_left"})
        except Exception as e:
            app.log("rois", f"focus-point overlay error: {e}")
    regs = [_xy_to_yx(r["polygon_overview"]) for r in data.get("regions", [])
            if len(r.get("polygon_overview", [])) >= 3]
    if regs:
        try:
            viewer.add_shapes(regs, shape_type="polygon", name=EXIST_MFOV,
                              edge_color="lime", face_color="transparent",
                              edge_width=_overlay_width(app), scale=app.layer_scale())
        except Exception as e:
            app.log("rois", f"mFOV overlay error: {e}")


# --------------------------------------------------------------------------- #
# Reorder: SIFT match lines + recovered serial chain
# --------------------------------------------------------------------------- #
def show_matches(app):
    viewer = app.viewer
    _remove(viewer, MATCH_LAYER)
    by_id = {s.id: s for s in app.project.sections}
    lines, conf = [], []
    for e in app.project.match_graph.edges:
        a, b = by_id.get(e.a), by_id.get(e.b)
        if a is None or b is None:
            continue
        lines.append(np.array([_centroid_yx(a), _centroid_yx(b)]))
        conf.append(float(e.confidence))
    if not lines:
        return
    try:
        viewer.add_shapes(lines, shape_type="line", name=MATCH_LAYER,
                          edge_width=1.5, scale=app.layer_scale(),
                          features={"confidence": conf},
                          edge_color="confidence", edge_colormap="viridis")
    except Exception as e:
        app.log("reorder", f"match overlay error: {e}")


MATCH_INSPECT_LAYER = "③ Match inspect"


def show_pair_matches(app, sec_a, sec_b, ratio: float = 0.75):
    """Draw the SIFT inlier correspondences between two sections as lines on the
    wafer (full-res crops, recomputed for just this pair). The richest reorder
    diagnostic: see exactly which features tie two sections together."""
    viewer = app.viewer
    _remove(viewer, MATCH_INSPECT_LAYER)
    if sec_a is None or sec_b is None or sec_a is sec_b or not app.has_image():
        return
    try:
        from . import crops, reorder
        geom = app.geom
        ga, _, (ax0, ay0, asc) = crops.read_section_crop(
            app.image_path, geom, sec_a.polygon, overview=app.overview, full_res=True)
        gb, _, (bx0, by0, bsc) = crops.read_section_crop(
            app.image_path, geom, sec_b.polygon, overview=app.overview, full_res=True)
        kpa, da = reorder.sift_features(ga)
        kpb, db = reorder.sift_features(gb)
        pa, pb = reorder.matched_points(kpa, da, kpb, db, ratio)
        if len(pa) == 0:
            app.log("reorder", f"no inlier matches between {sec_a.id} and {sec_b.id}.")
            return

        def _to_overview(pts, x0, y0, sc):
            fx = x0 + pts[:, 0] / sc
            fy = y0 + pts[:, 1] / sc
            if geom is not None:
                ox, oy = geom.full_to_ds(fx, fy)
                return np.column_stack([np.ravel(ox), np.ravel(oy)])
            return np.column_stack([fx, fy])

        a_ov = _to_overview(pa, ax0, ay0, asc)
        b_ov = _to_overview(pb, bx0, by0, bsc)
        lines = [np.array([[a_ov[i, 1], a_ov[i, 0]], [b_ov[i, 1], b_ov[i, 0]]])
                 for i in range(len(a_ov))]
        viewer.add_shapes(lines, shape_type="line", name=MATCH_INSPECT_LAYER,
                          edge_color="magenta", edge_width=1.0, scale=app.layer_scale())
        app.log("reorder", f"{len(lines)} inlier matches: {sec_a.id} ↔ {sec_b.id}.")
    except Exception as e:
        app.log("reorder", f"match-inspect error: {e}")


def show_serial_chain(app):
    viewer = app.viewer
    _remove(viewer, CHAIN_LAYER)
    order = app.project.match_graph.order
    by_id = {s.id: s for s in app.project.sections}
    seq = [by_id[i] for i in order if i in by_id]
    if len(seq) < 2:
        return
    vectors = []
    for k in range(len(seq) - 1):
        y0, x0 = _centroid_yx(seq[k])
        y1, x1 = _centroid_yx(seq[k + 1])
        vectors.append([[y0, x0], [y1 - y0, x1 - x0]])
    try:
        viewer.add_vectors(np.array(vectors), name=CHAIN_LAYER,
                           edge_width=_overlay_width(app), vector_style="arrow",
                           edge_color="orange", scale=app.layer_scale())
    except Exception as e:
        app.log("reorder", f"chain overlay error: {e}")


# --------------------------------------------------------------------------- #
# Imaging order: TSP route + order numbers
# --------------------------------------------------------------------------- #
SERIAL_NUM_LAYER = "③ Serial order #"


def set_sections_visible(app, on: bool):
    """Show/hide the Sections outlines (so order numbers don't clutter them)."""
    lyr = getattr(app.gui, "shapes_layer", None)
    if lyr is not None:
        try:
            lyr.visible = bool(on)
        except Exception:
            pass


def show_serial_numbers(app, hide_sections: bool = True):
    """Label each section with its recovered serial-order number on the wafer
    (a number mask). Hides the section outlines so the numbers read cleanly."""
    viewer = app.viewer
    _remove(viewer, SERIAL_NUM_LAYER)
    pts, nums = [], []
    for s in app.project.sections:
        if s.serial_index is None:
            continue
        cx, cy = s.centroid()
        pts.append([cy, cx])
        nums.append(int(s.serial_index) + 1)
    if not pts:
        app.log("reorder", "no serial order yet — compute the SIFT order first.")
        return
    try:
        viewer.add_points(np.asarray(pts, float), name=SERIAL_NUM_LAYER, size=1,
                          face_color="transparent", border_color="transparent",
                          scale=app.layer_scale(), features={"serial": nums},
                          text={"string": "{serial}", "size": 12, "color": "lime",
                                "anchor": "center"})
        if hide_sections:
            set_sections_visible(app, False)
    except Exception as e:
        app.log("reorder", f"serial-number overlay error: {e}")


def show_route(app, hide_sections: bool = True):
    viewer = app.viewer
    _remove(viewer, ROUTE_LAYER)
    _remove(viewer, ORDER_LAYER)
    seq = app.project.in_imaging_order()
    if len(seq) < 1:
        return
    pts = np.array([_centroid_yx(s) for s in seq])
    order_idx = [int(s.imaging_index) if s.imaging_index is not None else k
                 for k, s in enumerate(seq)]
    try:
        viewer.add_points(pts, name=ORDER_LAYER, size=1, face_color="transparent",
                          border_color="transparent", scale=app.layer_scale(),
                          features={"imaging_order": order_idx},
                          text={"string": "{imaging_order}", "size": 11,
                                "color": "cyan", "anchor": "center"})
        if hide_sections:
            set_sections_visible(app, False)
    except Exception as e:
        app.log("imaging", f"order-number overlay error: {e}")
    if len(seq) >= 2:
        vectors = []
        for k in range(len(seq) - 1):
            y0, x0 = pts[k]
            y1, x1 = pts[k + 1]
            vectors.append([[y0, x0], [y1 - y0, x1 - x0]])
        try:
            viewer.add_vectors(np.array(vectors), name=ROUTE_LAYER,
                               edge_width=_overlay_width(app), vector_style="arrow",
                               edge_color="yellow", scale=app.layer_scale())
        except Exception as e:
            app.log("imaging", f"route overlay error: {e}")


# --------------------------------------------------------------------------- #
# Session restore: redraw overlays for restored data, then reapply the user's
# saved display settings (widths / sizes / colours / opacity / visibility).
# --------------------------------------------------------------------------- #
def restore_overlays(app):
    """Redraw the overlays for whatever workflow data was just restored (ROIs,
    focus points, serial-order chain, imaging/TSP route + numbers). Section
    visibility is left to :func:`apply_display` so we never force-hide here."""
    proj = app.project
    for cond, fn in (
        (lambda: any(getattr(s, "roi", None) and s.roi.polygon for s in proj.sections),
         lambda: show_rois(app)),
        (lambda: any(getattr(s, "focus_overview", None) for s in proj.sections),
         lambda: show_focus_points(app)),
        (lambda: bool(proj.match_graph.order)
                 or any(s.serial_index is not None for s in proj.sections),
         lambda: show_serial_chain(app)),
        (lambda: any(s.imaging_index is not None for s in proj.sections),
         lambda: show_route(app, hide_sections=False)),
    ):
        try:
            if cond():
                fn()
        except Exception as e:
            app.log("io", f"overlay restore skipped: {e}")


def _scalar(arr, current=None):
    """A representative scalar from a napari per-element prop (edge_width / size).
    Prefer the actual per-element array (what's rendered); the ``current_*`` value
    only sets the width/size for *new* elements, so it lags what the user sees."""
    for src in (arr, current):
        try:
            a = np.ravel(np.asarray(src, dtype=float))
            if a.size:
                return float(a[0])
        except Exception:
            continue
    return None


def _color_to_json(c):
    """Normalise a napari colour (named string, hex, or RGBA array) to something
    JSON-serialisable that the matching setter accepts back."""
    try:
        if c is None:
            return None
        if isinstance(c, str):
            return c
        a = np.ravel(np.asarray(c))
        if a.dtype.kind in "US":          # array of colour strings
            return str(a[0])
        return [float(x) for x in a[:4]]
    except Exception:
        return None


def capture_display(app) -> dict:
    """Snapshot the visual properties of every STiM layer (everything except the
    base wafer image, for which only visibility/opacity are kept) into a
    JSON-serialisable dict keyed by layer name, so they persist across sessions."""
    from napari.layers import Image, Points, Shapes, Vectors
    viewer = getattr(app, "viewer", None)
    if viewer is None:
        return {}
    out = {}
    for lyr in list(viewer.layers):
        try:
            d = {"visible": bool(lyr.visible), "opacity": float(lyr.opacity)}
            if isinstance(lyr, Image):
                out[lyr.name] = d
                continue
            if isinstance(lyr, Shapes):
                d["edge_width"] = _scalar(lyr.edge_width, getattr(lyr, "current_edge_width", None))
                d["edge_color"] = _color_to_json(getattr(lyr, "current_edge_color", None))
                d["face_color"] = _color_to_json(getattr(lyr, "current_face_color", None))
            elif isinstance(lyr, Points):
                d["size"] = _scalar(lyr.size, getattr(lyr, "current_size", None))
                d["face_color"] = _color_to_json(getattr(lyr, "current_face_color", None))
                d["border_color"] = _color_to_json(getattr(lyr, "current_border_color", None))
                if getattr(lyr, "text", None) is not None and lyr.text.string is not None:
                    try:
                        d["text_size"] = float(lyr.text.size)
                    except Exception:
                        pass
            elif isinstance(lyr, Vectors):
                try:
                    d["edge_width"] = float(lyr.edge_width)
                except Exception:
                    pass
                d["edge_color"] = _color_to_json(getattr(lyr, "edge_color", None))
            out[lyr.name] = {k: v for k, v in d.items() if v is not None}
        except Exception:
            continue
    return out


def _set_attr(lyr, attr, val):
    if val is None:
        return
    try:
        setattr(lyr, attr, val)
    except Exception:
        pass


def apply_display(app) -> int:
    """Reapply saved visual properties to any matching layers currently present.
    Best-effort per property; returns the number of layers touched. The Sections
    layer's colours are left alone (they're driven by detection / QC)."""
    from napari.layers import Points, Shapes, Vectors
    viewer = getattr(app, "viewer", None)
    settings = getattr(app.project, "display_settings", None) or {}
    if viewer is None or not settings:
        return 0
    n = 0
    for lyr in list(viewer.layers):
        d = settings.get(lyr.name)
        if not d:
            continue
        keep_colors = lyr.name != "Sections"      # don't flatten per-section QC colours
        try:
            _set_attr(lyr, "visible", bool(d["visible"]) if "visible" in d else None)
            _set_attr(lyr, "opacity", float(d["opacity"]) if "opacity" in d else None)
            if isinstance(lyr, Shapes):
                if d.get("edge_width") is not None:
                    _set_attr(lyr, "edge_width", float(d["edge_width"]))
                    _set_attr(lyr, "current_edge_width", float(d["edge_width"]))
                if keep_colors:
                    _set_attr(lyr, "edge_color", d.get("edge_color"))
                    _set_attr(lyr, "face_color", d.get("face_color"))
            elif isinstance(lyr, Points):
                if d.get("size") is not None:
                    _set_attr(lyr, "size", float(d["size"]))
                    _set_attr(lyr, "current_size", float(d["size"]))
                _set_attr(lyr, "face_color", d.get("face_color"))
                _set_attr(lyr, "border_color", d.get("border_color"))
                if d.get("text_size") is not None and getattr(lyr, "text", None) is not None:
                    try:
                        lyr.text.size = float(d["text_size"])
                    except Exception:
                        pass
            elif isinstance(lyr, Vectors):
                if d.get("edge_width") is not None:
                    _set_attr(lyr, "edge_width", float(d["edge_width"]))
                _set_attr(lyr, "edge_color", d.get("edge_color"))
            n += 1
        except Exception:
            continue
    return n
