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
                          edge_width=2, scale=app.layer_scale())
    except Exception as e:
        app.log("rois", f"overlay error: {e}")


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
        viewer.add_vectors(np.array(vectors), name=CHAIN_LAYER, edge_width=2,
                           vector_style="arrow", edge_color="orange",
                           scale=app.layer_scale())
    except Exception as e:
        app.log("reorder", f"chain overlay error: {e}")


# --------------------------------------------------------------------------- #
# Imaging order: TSP route + order numbers
# --------------------------------------------------------------------------- #
def show_route(app):
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
        viewer.add_points(pts, name=ORDER_LAYER, size=6, face_color="cyan",
                          scale=app.layer_scale(),
                          features={"imaging_order": order_idx},
                          text={"string": "{imaging_order}", "size": 9,
                                "color": "white", "anchor": "center"})
    except Exception as e:
        app.log("imaging", f"order-number overlay error: {e}")
    if len(seq) >= 2:
        vectors = []
        for k in range(len(seq) - 1):
            y0, x0 = pts[k]
            y1, x1 = pts[k + 1]
            vectors.append([[y0, x0], [y1 - y0, x1 - x0]])
        try:
            viewer.add_vectors(np.array(vectors), name=ROUTE_LAYER, edge_width=2,
                               vector_style="arrow", edge_color="yellow",
                               scale=app.layer_scale())
        except Exception as e:
            app.log("imaging", f"route overlay error: {e}")
