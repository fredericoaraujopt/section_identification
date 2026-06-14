"""Versioned, backward-compatible persistence for a :class:`WaferProject`.

On disk we store **full-resolution** pixel coordinates (frame-independent across
overview-resolution changes — the same property the legacy
``*_stim_project.json`` relied on). The in-memory model holds **overview-px**
coordinates, so this module is the single place that converts geometry at the
disk boundary (the model itself is frame-agnostic).

Schema:
  * ``schema_version >= 2`` — rich per-section records (this module's format).
  * no ``schema_version`` — a legacy file (bare full-res polygon lists); loaded
    via :meth:`WaferProject.from_legacy` so existing projects keep working.

Geometry conversion uses a ``geom`` exposing ``ds_to_full``/``full_to_ds``
(:class:`section_identification.czi_io.CziGeometry`); ``geom=None`` (non-CZI /
PNG) means overview == full, i.e. identity.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .wafer_model import SCHEMA_VERSION, WaferProject


# --------------------------------------------------------------------------- #
# paths (mirrors the legacy GUI layout so the same file is reused)
# --------------------------------------------------------------------------- #
def project_path(image_path: str) -> str:
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(f"{os.path.splitext(image_path)[0]}_files",
                        f"{base}_stim_project.json")


# --------------------------------------------------------------------------- #
# overview <-> full-res converters (identity when geom is None)
# --------------------------------------------------------------------------- #
def _poly_ds_to_full(poly, geom):
    if geom is None or not poly:
        return [[float(x), float(y)] for x, y in np.asarray(poly, float).reshape(-1, 2)]
    p = np.asarray(poly, float).reshape(-1, 2)
    fx, fy = geom.ds_to_full(p[:, 0], p[:, 1])
    return [[float(a), float(b)] for a, b in zip(np.ravel(fx), np.ravel(fy))]


def _poly_full_to_ds(poly, geom):
    if geom is None or not poly:
        return [[float(x), float(y)] for x, y in np.asarray(poly, float).reshape(-1, 2)]
    p = np.asarray(poly, float).reshape(-1, 2)
    x, y = geom.full_to_ds(p[:, 0], p[:, 1])
    return [[float(a), float(b)] for a, b in zip(np.ravel(x), np.ravel(y))]


def _pt_conv(pt, conv_poly, geom):
    if pt is None:
        return None
    return conv_poly([pt], geom)[0]


def _convert_geometry(d: dict, conv_poly, geom) -> dict:
    """Return a copy of a v2 project dict with all wafer-frame geometry mapped
    through ``conv_poly`` (overview<->full). Frame-independent fields (stage-µm
    focus points, qc, indices, match graph) and the pose-normalised ROI template
    polygons are left untouched."""
    out = dict(d)
    secs = []
    for s in d.get("sections", []):
        s2 = dict(s)
        s2["polygon"] = conv_poly(s.get("polygon", []), geom)
        if s.get("roi"):
            roi = dict(s["roi"])
            roi["polygon"] = conv_poly(roi.get("polygon", []), geom)
            s2["roi"] = roi
        pose = dict(s.get("pose") or {})
        if pose.get("center") is not None:
            pose["center"] = _pt_conv(pose["center"], conv_poly, geom)
            s2["pose"] = pose
        secs.append(s2)
    out["sections"] = secs
    out["fiducials"] = [_pt_conv(p, conv_poly, geom) for p in d.get("fiducials", [])]
    out["raw_sections"] = [conv_poly(p, geom) for p in d.get("raw_sections", [])]
    out["calibration_examples"] = [conv_poly(p, geom)
                                   for p in d.get("calibration_examples", [])]
    return out


# --------------------------------------------------------------------------- #
# save / load
# --------------------------------------------------------------------------- #
def save(project: WaferProject, geom, path: str | None = None) -> str | None:
    """Serialise ``project`` (overview px) to disk as full-res JSON. Returns the
    path written, or None on failure (best-effort, like the legacy autosave)."""
    img = project.image_path
    if img is None:
        return None
    try:
        path = path or project_path(img)
        disk = _convert_geometry(project.to_dict(), _poly_ds_to_full, geom)
        disk["schema_version"] = SCHEMA_VERSION
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        with open(tmp, "w") as f:
            json.dump(disk, f)
        os.replace(tmp, path)
        return path
    except Exception:
        return None


def load(image_path: str, geom) -> WaferProject | None:
    """Load a project for ``image_path``. Returns a :class:`WaferProject` (with
    overview-px geometry) or None if no project file exists / parse fails.
    Legacy files (no ``schema_version``) are migrated via ``from_legacy``."""
    if image_path is None:
        return None
    path = project_path(image_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None

    if "schema_version" not in d:
        # legacy: full-res polygon lists; map each to overview via geom.
        proj = WaferProject.from_legacy(
            d, to_overview=lambda pts: _poly_full_to_ds(pts, geom))
        proj.image_path = image_path
        return proj

    overview = _convert_geometry(d, _poly_full_to_ds, geom)
    proj = WaferProject.from_dict(overview)
    proj.image_path = image_path
    return proj
