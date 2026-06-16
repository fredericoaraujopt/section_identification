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


def propagate_focus_to_pose(template: RoiTemplate, pose) -> list[list[float]]:
    """Map a template's local focus points into a section's world frame."""
    if not template.focus_local:
        return []
    return [[float(x), float(y)] for x, y in
            (fov_nav.local_to_world(p, pose.center, pose.angle_deg, pose.flip)
             for p in np.asarray(template.focus_local, float).reshape(-1, 2))]


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
    section's pose."""
    for s in sections:
        if s.pose.center is None:
            continue
        if template.polygon_local:
            world = propagate_to_pose(template, s.pose)
            fitted = fit_roi(world, s.polygon, template.fit_mode, template.fit_percent)
            s.roi = Roi(polygon=fitted, fit_mode=template.fit_mode,
                        fit_percent=template.fit_percent)
        if template.focus_local:
            s.focus_overview = propagate_focus_to_pose(template, s.pose)
