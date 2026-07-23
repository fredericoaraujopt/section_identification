"""ROI definition, propagation, and coming-in fitting (pure, headless-testable).

Stage "ROIs": the user defines one region of interest on a reference section
(via SAM or by hand); it is stored in that section's *pose-normalised* (upright,
centred) local frame as a :class:`wafer_model.RoiTemplate`, then propagated onto
every section by mapping the local polygon through each section's pose. Because
the template lives in the canonical frame, propagation automatically follows each
section's rotation — the ROI lands on the same anatomical region everywhere.

"Coming-in" sections (the first few in a series) are smaller/partial, so a raw
propagated ROI can overhang. ``fit_roi`` offers: ``full`` (scale the ROI to the
section's extent), ``percent`` (scale to a fraction of that), ``clip`` (intersect
with the section polygon), or ``template``/``manual`` (leave as-is).

All coordinates are ``(x, y)`` in the working overview-pixel frame, consistent
with the rest of the model. The mFOV tile-grid / focus-point geometry for ZEN is
built in the ROI stage controller from these polygons + stage-µm conversion.
"""

from __future__ import annotations

import numpy as np

from . import fov_nav
from .wafer_model import Roi, RoiTemplate


def _bbox(poly: np.ndarray):
    return (poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max())


def template_from_polygon(pose, roi_polygon, ref_section_id=None, **kw) -> RoiTemplate:
    """Build a :class:`RoiTemplate` from an ROI drawn in world/overview coords on
    a reference section with the given ``pose`` (wafer_model.Pose)."""
    local = np.array([fov_nav.world_to_local(p, pose.center, pose.angle_deg, pose.flip)
                      for p in np.asarray(roi_polygon, float).reshape(-1, 2)])
    return RoiTemplate(polygon_local=[[float(x), float(y)] for x, y in local],
                       ref_section_id=ref_section_id, **kw)


def propagate_to_pose(template: RoiTemplate, pose) -> list[list[float]]:
    """Map a template's local polygon into a section's world frame via its pose."""
    return [[float(x), float(y)] for x, y in
            (fov_nav.local_to_world(p, pose.center, pose.angle_deg, pose.flip)
             for p in np.asarray(template.polygon_local, float).reshape(-1, 2))]


def focus_template_from_points(pose, points_xy) -> list[list[float]]:
    """Express focus points drawn on a reference section in its pose-normalised
    local frame (so they propagate to every section like an ROI)."""
    pts = np.asarray(points_xy, float).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in
            (fov_nav.world_to_local(p, pose.center, pose.angle_deg, pose.flip)
             for p in pts)]


def propagate_centered_to_section(template: RoiTemplate, section) -> list[list[float]] | None:
    """Place the template's ROI at a section's centroid with **no** pose, scale or
    shape adjustment — the polygon keeps its drawn dimensions and orientation and
    is simply translated so its centroid lands on the section centroid.

    Unlike :func:`propagate_to_pose` + :func:`fit_roi`, this ignores the section's
    rotation and size. It is the robust choice when section polygons vary widely in
    size across a wafer, where size-based fitting misplaces the ROI."""
    local = np.asarray(template.polygon_local, float).reshape(-1, 2)
    if len(local) < 3 or not section.polygon:
        return None
    local = local - local.mean(axis=0)      # centre the ROI on the origin
    cx, cy = section.centroid()
    return [[float(x + cx), float(y + cy)] for x, y in local]


def propagate_focus_to_pose(template: RoiTemplate, pose) -> list[list[float]]:
    """Map a template's local focus points into a section's world frame."""
    if not template.focus_local:
        return []
    return [[float(x), float(y)] for x, y in
            (fov_nav.local_to_world(p, pose.center, pose.angle_deg, pose.flip)
             for p in np.asarray(template.focus_local, float).reshape(-1, 2))]


def _point_in_polygon(pt, poly) -> bool:
    poly = np.asarray(poly, float).reshape(-1, 2)
    if len(poly) < 3:
        return False
    try:
        from shapely.geometry import Point, Polygon
        return bool(Polygon(poly).buffer(0).contains(Point(float(pt[0]), float(pt[1]))))
    except Exception:
        # ray casting fallback
        x, y = float(pt[0]), float(pt[1])
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if ((y1 > y) != (y2 > y)) and \
               (x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1):
                inside = not inside
        return inside


def focus_anchors_from_points(ref_section, points_xy) -> list[dict]:
    """Classify each drafted focus point (overview px) against a reference section
    and record how to reproduce it on every section. A point inside the section's
    ROI anchors to the **ROI**; otherwise to the **section**. Each anchor stores:

    * ``local`` — the offset from the anchor centroid in the reference section's
      pose-normalised (upright) frame → used by pose/relative propagation so it
      lands at the same *relative* spot (e.g. an ROI's top-left → every ROI's
      top-left), rotation-aware.
    * ``drawn`` — the raw world offset from the anchor centroid → used by
      centre propagation (no pose, no scaling)."""
    pts = np.asarray(points_xy, float).reshape(-1, 2)
    sec_c = np.asarray(ref_section.centroid(), float)
    a_ref = ref_section.pose.angle_deg
    flip = ref_section.pose.flip
    roi_poly = None
    roi_c = None
    if getattr(ref_section, "roi", None) and len(ref_section.roi.polygon) >= 3:
        roi_poly = np.asarray(ref_section.roi.polygon, float).reshape(-1, 2)
        roi_c = roi_poly.mean(axis=0)
    anchors = []
    for p in pts:
        if roi_poly is not None and _point_in_polygon(p, roi_poly):
            anchor, ac = "roi", roi_c
        else:
            anchor, ac = "section", sec_c
        local = fov_nav.world_to_local(p, (float(ac[0]), float(ac[1])), a_ref, flip)
        anchors.append({"anchor": anchor,
                        "local": [float(local[0]), float(local[1])]})
    return anchors


def _anchor_centroids(section):
    """(roi_centroid, section_centroid) for a section, or (None, section_centroid)
    when it has no ROI."""
    sec_c = np.asarray(section.centroid(), float)
    roi_c = None
    if getattr(section, "roi", None) and len(section.roi.polygon) >= 3:
        roi_c = np.asarray(section.roi.polygon, float).reshape(-1, 2).mean(axis=0)
    return roi_c, sec_c


def propagate_focus_anchored(template: RoiTemplate, section, mode: str) -> list[list[float]]:
    """Place the template's anchored focus points on a section. Only sections that
    have an ROI receive focus points (an ROI-less region isn't imaged, so it needs
    no focus). ``mode='pose'`` reproduces the relative position through the
    section's pose (rotation-aware — e.g. an ROI's top-left → each ROI's top-left);
    ``mode='center'`` places every point at the ROI / section **centroid**."""
    if not template.focus_anchors:
        return []
    roi_c, sec_c = _anchor_centroids(section)
    if roi_c is None:                       # no ROI → no focus points (req. #1)
        return []
    out = []
    for a in template.focus_anchors:
        base = roi_c if a.get("anchor") == "roi" else sec_c
        if mode == "center":
            out.append([float(base[0]), float(base[1])])       # at the centroid
        else:
            w = fov_nav.local_to_world(a.get("local", [0.0, 0.0]),
                                       (float(base[0]), float(base[1])),
                                       section.pose.angle_deg, section.pose.flip)
            out.append([float(w[0]), float(w[1])])
    return out


def propagate_focus_only(template: RoiTemplate, sections) -> None:
    """Set ``section.focus_overview`` for every section **without touching the
    ROIs**. Used by the focus buttons so propagating focus never re-propagates /
    overwrites the per-section ROIs (that clobbering also gave ROI-less sections a
    focus point). Focus lands only on sections that have an ROI."""
    for s in sections:
        if template.focus_anchors:
            s.focus_overview = propagate_focus_anchored(template, s, getattr(template, "focus_mode", "pose"))
        elif template.focus_local and s.pose.center is not None \
                and getattr(s, "roi", None) and len(s.roi.polygon) >= 3:
            s.focus_overview = propagate_focus_to_pose(template, s.pose)   # legacy
        else:
            s.focus_overview = []


def _scale_about(poly: np.ndarray, center, s: float) -> np.ndarray:
    center = np.asarray(center, float).reshape(2)
    return (poly - center) * s + center


def fit_roi(roi_polygon, section_polygon, mode: str = "template",
            percent: float = 100.0) -> list[list[float]]:
    """Fit a propagated ROI to a section. Returns the fitted polygon (overview px).

    * ``template`` / ``manual`` — unchanged.
    * ``full`` — scale (about the ROI centroid) to the largest size fitting the
      section's bounding box, then recentre on the section centroid.
    * ``percent`` — like ``full`` but scaled to ``percent`` % of it.
    * ``clip`` — intersect the ROI with the section polygon (shapely).
    """
    roi = np.asarray(roi_polygon, float).reshape(-1, 2)
    sec = np.asarray(section_polygon, float).reshape(-1, 2)
    if mode in ("template", "manual") or len(roi) < 3 or len(sec) < 3:
        return [[float(x), float(y)] for x, y in roi]

    if mode == "clip":
        try:
            from shapely.geometry import Polygon
            inter = Polygon(roi).buffer(0).intersection(Polygon(sec).buffer(0))
            if inter.is_empty:
                return [[float(x), float(y)] for x, y in roi]
            geom = max(inter.geoms, key=lambda g: g.area) if inter.geom_type == "MultiPolygon" else inter
            return [[float(x), float(y)] for x, y in np.asarray(geom.exterior.coords)[:-1]]
        except Exception:
            return [[float(x), float(y)] for x, y in roi]

    # full / percent: scale about ROI centroid to fit the section bbox
    rx0, ry0, rx1, ry1 = _bbox(roi)
    sx0, sy0, sx1, sy1 = _bbox(sec)
    rw, rh = max(rx1 - rx0, 1e-9), max(ry1 - ry0, 1e-9)
    sw, sh = max(sx1 - sx0, 1e-9), max(sy1 - sy0, 1e-9)
    s = min(sw / rw, sh / rh)
    if mode == "percent":
        s *= max(percent, 0.0) / 100.0
    roi_c = roi.mean(axis=0)
    scaled = _scale_about(roi, roi_c, s)
    sec_c = sec.mean(axis=0)
    scaled = scaled - scaled.mean(axis=0) + sec_c     # recentre on section
    return [[float(x), float(y)] for x, y in scaled]


def propagate_all(template: RoiTemplate, sections) -> None:
    """Set ``section.roi`` (and ``focus_overview`` if the template has focus
    points) for every section by propagating + fitting the template through each
    section's pose.

    ``fit_mode == "center"`` is special: the ROI is placed at each section's
    centroid unchanged (see :func:`propagate_centered_to_section`), independent of
    pose — for wafers whose section polygons vary widely in size."""
    for s in sections:
        if template.polygon_local:
            if template.fit_mode == "center":
                fitted = propagate_centered_to_section(template, s)
                if fitted is not None:
                    s.roi = Roi(polygon=fitted, fit_mode="center",
                                fit_percent=template.fit_percent)
            elif s.pose.center is not None:
                world = propagate_to_pose(template, s.pose)
                fitted = fit_roi(world, s.polygon, template.fit_mode, template.fit_percent)
                s.roi = Roi(polygon=fitted, fit_mode=template.fit_mode,
                            fit_percent=template.fit_percent)
        # Focus points — only on sections that have an ROI (an un-imaged region
        # needs no focus). Each point rides its anchor (ROI if drawn inside it,
        # else the section): "pose" keeps the relative spot rotation-aware,
        # "center" drops the drawn offset at the ROI/section centroid unchanged.
        mode = getattr(template, "focus_mode", "pose")
        if template.focus_anchors:
            s.focus_overview = propagate_focus_anchored(template, s, mode)
        elif template.focus_local and s.pose.center is not None \
                and getattr(s, "roi", None) and len(s.roi.polygon) >= 3:
            s.focus_overview = propagate_focus_to_pose(template, s.pose)   # legacy
        else:
            s.focus_overview = []


# --------------------------------------------------------------------------- #
# Automatic per-section ROI detection (template-guided)
#
# For wafers where the ROI is clearly visible against the resin but the sections
# vary too much for geometric propagation, SAM runs inside each section: one
# encoder embedding per section, then a grid of point prompts across the section
# interior (the same idiom SAM's automatic mask generator uses), each producing
# candidate masks that are scored against the template (area + shape) and SAM's
# own confidence. The single best-scoring mask becomes that section's ROI. These
# helpers are the pure, headless-testable core; the GUI (StageROIs) drives the
# embedding/prediction loop and the live preview.
# --------------------------------------------------------------------------- #

def _shoelace_area(poly) -> float:
    p = np.asarray(poly, float).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def build_point_grid(n: int) -> np.ndarray:
    """``n × n`` points in the unit square at cell centres — SAM's own grid
    convention (``(2i+1)/(2n)``). Mirrors ``sam2`` / ``param_viz`` so the preview
    and the run use identical sampling."""
    n = max(int(n), 1)
    offs = (2.0 * np.arange(n) + 1.0) / (2.0 * n)
    xs, ys = np.meshgrid(offs, offs)
    return np.column_stack([xs.ravel(), ys.ravel()])


def section_point_grid(section_polygon, points_per_side: int,
                       inset: float = 0.0) -> list[list[float]]:
    """Grid of prompt points (overview xy) spanning a section's bounding box and
    clipped to the section polygon, so SAM is only prompted inside the section.
    ``inset`` shrinks the bbox by a fraction on each side to avoid edge points."""
    sec = np.asarray(section_polygon, float).reshape(-1, 2)
    if len(sec) < 3:
        return []
    x0, y0, x1, y1 = _bbox(sec)
    w, h = (x1 - x0), (y1 - y0)
    x0 += w * inset; x1 -= w * inset
    y0 += h * inset; y1 -= h * inset
    g = build_point_grid(points_per_side)
    pts = np.column_stack([x0 + g[:, 0] * (x1 - x0), y0 + g[:, 1] * (y1 - y0)])
    inside = [p for p in pts if _point_in_polygon(p, sec)]
    return [[float(x), float(y)] for x, y in inside]


def _iou(poly_a, poly_b) -> float:
    try:
        from shapely.geometry import Polygon
        A = Polygon(np.asarray(poly_a, float).reshape(-1, 2)).buffer(0)
        B = Polygon(np.asarray(poly_b, float).reshape(-1, 2)).buffer(0)
        if A.is_empty or B.is_empty:
            return 0.0
        inter = A.intersection(B).area
        union = A.union(B).area
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        return 0.0


def _containment(poly, container) -> float:
    """Fraction of ``poly``'s area that lies inside ``container`` (0..1)."""
    try:
        from shapely.geometry import Polygon
        P = Polygon(np.asarray(poly, float).reshape(-1, 2)).buffer(0)
        C = Polygon(np.asarray(container, float).reshape(-1, 2)).buffer(0)
        if P.is_empty or P.area <= 0:
            return 0.0
        return float(P.intersection(C).area / P.area)
    except Exception:
        return 1.0


def fit_template_to_mask(template_polygon, mask_polygon) -> list[list[float]]:
    """Return the template's shape scaled (about its centroid) to match the mask's
    area and recentred on the mask centroid — a uniform ROI shape whose size and
    placement follow SAM. Used when the user picks "template fitted to mask"."""
    tmpl = np.asarray(template_polygon, float).reshape(-1, 2)
    mask = np.asarray(mask_polygon, float).reshape(-1, 2)
    if len(tmpl) < 3 or len(mask) < 3:
        return [[float(x), float(y)] for x, y in tmpl]
    ta, ma = _shoelace_area(tmpl), _shoelace_area(mask)
    s = float(np.sqrt(ma / ta)) if ta > 0 else 1.0
    scaled = _scale_about(tmpl, tmpl.mean(axis=0), s)
    scaled = scaled - scaled.mean(axis=0) + mask.mean(axis=0)
    return [[float(x), float(y)] for x, y in scaled]


# default scoring weights: SAM confidence, area match, shape match, containment
_ROI_SCORE_WEIGHTS = {"iou": 0.35, "area": 0.25, "shape": 0.25, "contain": 0.15}


def score_mask(mask_polygon, template_polygon, section_polygon, sam_iou: float,
               area_band, weights=None) -> float:
    """Score a candidate SAM mask for how well it matches the template ROI and how
    confident/clean it is. Returns 0 for masks outside the area band; otherwise a
    weighted blend of SAM confidence, area closeness to the template, shape IoU
    (scale-normalised, centroid-aligned) and containment within the section."""
    w = dict(_ROI_SCORE_WEIGHTS)
    if weights:
        w.update(weights)
    mask = np.asarray(mask_polygon, float).reshape(-1, 2)
    if len(mask) < 3:
        return 0.0
    area_m = _shoelace_area(mask)
    amin, amax = float(area_band[0]), float(area_band[1])
    if area_m <= 0 or area_m < amin or area_m > amax:
        return 0.0
    target = np.sqrt(amin * amax) if amin > 0 else amax        # geo-mean of band
    area_score = float(max(0.0, 1.0 - abs(np.log(area_m / target)) if target > 0 else 0.0))
    # scale-normalise the template to the mask area, align centroids → shape IoU
    tmpl_fit = fit_template_to_mask(template_polygon, mask)
    shape_score = _iou(tmpl_fit, mask)
    contain = _containment(mask, section_polygon)
    return float(w["iou"] * max(0.0, min(1.0, sam_iou))
                 + w["area"] * area_score
                 + w["shape"] * shape_score
                 + w["contain"] * contain)


def choose_best_roi(candidates, template_polygon, section_polygon, area_band,
                    floor: float = 0.35, weights=None):
    """From ``candidates`` (list of ``(mask_polygon, sam_iou)``) pick the single
    best-scoring mask compatible with the template. Returns ``(polygon, score)``,
    or ``(None, best_score)`` if nothing clears ``floor`` (caller then falls back
    to the propagated template)."""
    best_poly, best_score = None, 0.0
    for mask_poly, sam_iou in candidates:
        sc = score_mask(mask_poly, template_polygon, section_polygon,
                        float(sam_iou), area_band, weights)
        if sc > best_score:
            best_poly, best_score = mask_poly, sc
    if best_poly is None or best_score < floor:
        return None, best_score
    return [[float(x), float(y)] for x, y in np.asarray(best_poly, float).reshape(-1, 2)], best_score


def calibrate_roi_params(template_polygon, section_polygons, profile=None,
                         min_points: float = 3.0, min_area_frac: float = 0.5,
                         max_area_mult: float = 2.0) -> dict:
    """Derive the automatic-ROI SAM parameters from the drawn template's geometry —
    the ROI analogue of :func:`calibration.calibrate`, seeded by the template ROI
    instead of example sections.

    Each section is embedded at ~1024 px (long side), so the ROI appears to SAM at
    ``roi_minor * 1024 / section_long``. ``points_per_side`` is solved so the grid
    lands ``>= min_points`` points on the ROI; quality gates follow the ROI's
    apparent size. Returns a dict of populated parameter values."""
    from .calibration import _minor_axis, _polygon_area, _thresholds_for

    roi_area = _polygon_area(template_polygon)
    roi_minor = _minor_axis(template_polygon)
    if roi_area <= 0:
        raise ValueError("Template ROI has zero area.")

    # typical section bounding-box long side (the crop that gets embedded ~1024 px)
    longs = []
    for poly in section_polygons or []:
        p = np.asarray(poly, float).reshape(-1, 2)
        if len(p) >= 3:
            x0, y0, x1, y1 = _bbox(p)
            longs.append(max(x1 - x0, y1 - y0))
    sec_long = float(np.median(longs)) if longs else float(np.sqrt(roi_area) * 4.0)
    sec_bbox_area = sec_long * sec_long

    # points on the ROI ≈ roi_area * (pps / sec_long)²  →  solve for pps
    pps = int(np.clip(np.ceil(sec_long * np.sqrt(float(min_points) / max(roi_area, 1.0))),
                      6, 48))
    points_on_roi = float(roi_area * (pps / max(sec_long, 1.0)) ** 2)

    rz = min(1.0, 1024.0 / max(sec_long, 1.0))       # section crop → ~1024 px
    d_sam = roi_minor * rz                            # apparent ROI thin axis to SAM
    pred_iou, stab = _thresholds_for(d_sam)
    pps_batch = int(getattr(profile, "points_per_batch", 16) or 16)
    model_variant = getattr(profile, "model_variant", None) or "base_plus"

    return {
        "points_per_side": pps,
        "points_on_roi": points_on_roi,
        "pred_iou_thresh": pred_iou,
        "stability_score_thresh": stab,
        "min_area_frac": float(min_area_frac),
        "max_area_mult": float(max_area_mult),
        "roi_area": float(roi_area),
        "roi_minor": float(roi_minor),
        "section_long_px": sec_long,
        "apparent_roi_px": float(d_sam),
        "points_per_batch": pps_batch,
        "model_variant": model_variant,
        "score_floor": 0.35,
    }


def area_band_for_section(section, params) -> tuple:
    """The [min, max] area band a detected mask must fall in for a given section:
    the section's *propagated* template area × (min_frac, max_mult). Falls back to
    the calibrated reference ``roi_area`` when the section has no propagated ROI."""
    ref = float(params.get("roi_area", 0.0))
    if getattr(section, "roi", None) and len(section.roi.polygon) >= 3:
        ref = _shoelace_area(section.roi.polygon)
    lo = ref * float(params.get("min_area_frac", 0.5))
    hi = ref * float(params.get("max_area_mult", 2.0))
    return (lo, hi)
