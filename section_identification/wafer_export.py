"""Pluggable wafer/section/order export adapters.

Builds a canonical :func:`build_manifest` from a :class:`WaferProject` (+ geom)
and writes it through a registry of adapters. This is additive to the legacy
CSV/GeoJSON/PNG/annotated-CZI export (export.py / czi_export.py) — those stay the
default; the manifest + adapters are extra outputs.

Adapters (``write(manifest, out_dir) -> path``):
  * ``json_manifest`` — the canonical interchange (always useful).
  * ``csv_table``     — one spreadsheet row per section (ids, order, qc, stage µm).
  * ``mvis_lmb``      — ``region_names.csv`` (``id; name; n_mfovs``) for the
                        mVis_LMB viewer: ``id`` = TSP/imaging order, ``name`` =
                        serial-section name (e.g. ``S15``). See the
                        ``mvis-lmb-integration`` project memory.
  * ``magc``          — partial projection toward the connectomics .magc wafer
                        format (JSON manifest is the source of truth; flagged).
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def _stage_um(geom, x_full, y_full):
    if geom is None:
        return None
    s = geom.full_to_stage_um(np.asarray([x_full]), np.asarray([y_full]))
    if s is None:
        return None
    return [float(np.ravel(s[0])[0]), float(np.ravel(s[1])[0])]


def _pixel_size_um(geom):
    if geom is None:
        return None
    sx = getattr(geom, "scale_x", None)
    sy = getattr(geom, "scale_y", None)
    if sx is None:
        return None
    return [float(sx) * 1e6, float((sy or sx)) * 1e6]   # meters/px -> µm/px


def serial_name(section) -> str:
    """mVis-style region name from the recovered serial order (``S<n>``),
    falling back to the section id when unordered."""
    if section.serial_index is not None:
        return f"S{int(section.serial_index) + 1}"
    return section.id


def build_manifest(project, geom, wafer_id: str | None = None,
                   mfov_counts: dict | None = None) -> dict:
    """Assemble the canonical wafer manifest from the project (overview px) +
    geom. Polygons are emitted in full-res px and (when geom) stage µm."""
    mfov_counts = mfov_counts or {}
    img = project.image_path
    sections = []
    for s in project.sections:
        cfull = None
        cstage = None
        cx, cy = s.centroid()
        full = s.polygon_full(geom)
        if full:
            arr = np.asarray(full, float)
            cfull = [float(arr[:, 0].mean()), float(arr[:, 1].mean())]
            cstage = _stage_um(geom, cfull[0], cfull[1])
        roi_full = None
        roi_stage = None
        if s.roi and s.roi.polygon:
            ra = np.asarray(s.roi.polygon, float).reshape(-1, 2)
            if geom is not None:
                fx, fy = geom.ds_to_full(ra[:, 0], ra[:, 1])
                roi_full = [[float(a), float(b)] for a, b in zip(np.ravel(fx), np.ravel(fy))]
                roi_stage = [_stage_um(geom, a, b) for a, b in roi_full]
            else:
                roi_full = [[float(a), float(b)] for a, b in ra]
        sections.append({
            "id": s.id,
            "serial_index": s.serial_index,
            "imaging_index": s.imaging_index,
            "serial_name": serial_name(s),
            "accepted": bool(s.accepted),
            "polygon_full_px": full,
            "centroid_full_px": cfull,
            "centroid_stage_um": cstage,
            "area_full_px": s.area() / (getattr(geom, "zoom", 1.0) ** 2 if geom else 1.0),
            "roi_full_px": roi_full,
            "roi_stage_um": roi_stage,
            "focus_points_stage_um": [fp.to_dict() for fp in s.focus_points],
            "qc": s.qc.to_dict() if s.qc else None,
            "mfovs": int(mfov_counts.get(s.id, 1)),
        })
    fiducials = []
    for (fx, fy) in project.fiducials:
        ff = None
        fs = None
        if geom is not None:
            gx, gy = geom.ds_to_full(np.asarray([fx]), np.asarray([fy]))
            ff = [float(np.ravel(gx)[0]), float(np.ravel(gy)[0])]
            fs = _stage_um(geom, ff[0], ff[1])
        else:
            ff = [float(fx), float(fy)]
        fiducials.append({"full_px": ff, "stage_um": fs})

    return {
        "schema": "stim.wafer/1",
        "wafer_id": wafer_id or (os.path.splitext(os.path.basename(img))[0] if img else "wafer"),
        "source_image": img,
        "units": {"pixel_size_um": _pixel_size_um(geom)},
        "fiducials": fiducials,
        "ordering": {
            "serial": list(project.match_graph.order),
            "imaging": [s["id"] for s in sorted(
                sections, key=lambda d: (d["imaging_index"] is None, d["imaging_index"] or 0))],
        },
        "sections": sections,
    }


# --------------------------------------------------------------------------- #
# adapters
# --------------------------------------------------------------------------- #
def _imaging_sorted(manifest):
    """Sections sorted by imaging (TSP) order; unordered go last, stably."""
    return sorted(manifest["sections"],
                  key=lambda d: (d["imaging_index"] is None, d["imaging_index"] or 0))


def write_json_manifest(manifest: dict, out_dir: str) -> str:
    path = os.path.join(out_dir, f"{manifest['wafer_id']}_wafer.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def write_csv_table(manifest: dict, out_dir: str) -> str:
    path = os.path.join(out_dir, f"{manifest['wafer_id']}_sections.csv")
    cols = ["section_id", "serial_index", "imaging_index", "serial_name", "accepted",
            "centroid_x_um", "centroid_y_um", "area_full_px",
            "qc_overall", "qc_debris", "qc_fold", "qc_shred", "qc_chatter", "flag_any"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for s in manifest["sections"]:
            c = s.get("centroid_stage_um") or [None, None]
            sc = (s.get("qc") or {}).get("scores", {})
            fl = (s.get("qc") or {}).get("flags", {})
            w.writerow([s["id"], s["serial_index"], s["imaging_index"], s["serial_name"],
                        s["accepted"], c[0], c[1], s.get("area_full_px"),
                        sc.get("overall"), sc.get("debris"), sc.get("fold"),
                        sc.get("shred"), sc.get("chatter"), (fl or {}).get("any")])
    return path


def write_mvis_lmb(manifest: dict, out_dir: str) -> str:
    """region_names.csv ('id; name; n_mfovs') in TSP/imaging order — the mVis_LMB
    contract. id = acquisition order (ZEN images by Id), name = serial section."""
    path = os.path.join(out_dir, "region_names.csv")
    ordered = _imaging_sorted(manifest)
    with open(path, "w", newline="") as f:
        for i, s in enumerate(ordered, start=1):
            n_mfovs = max(1, int(s.get("mfovs", 1)))
            f.write(f"{i:03d}; {s['serial_name']}; {n_mfovs}\n")
    # sidecar with stage coords for a future spatial wafer overview
    side = os.path.join(out_dir, "region_positions.csv")
    with open(side, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "section_id", "stage_x_um", "stage_y_um", "n_mfovs"])
        for i, s in enumerate(ordered, start=1):
            c = s.get("centroid_stage_um") or [None, None]
            w.writerow([f"{i:03d}", s["serial_name"], s["id"], c[0], c[1],
                        max(1, int(s.get("mfovs", 1)))])
    return path


def write_magc(manifest: dict, out_dir: str) -> str:
    """Partial projection toward the connectomics .magc wafer format. The JSON
    manifest remains the source of truth; this is a best-effort mapping (section
    polygons + centers + serial order) flagged for validation against a real
    .magc sample before relying on it downstream."""
    path = os.path.join(out_dir, f"{manifest['wafer_id']}.magc")
    lines = ["[sections]"]
    for s in manifest["sections"]:
        poly = s.get("polygon_full_px") or []
        flat = ".".join(f"{x:.1f},{y:.1f}" for x, y in poly)
        lines.append(f"{s['id']} = {flat}")
    lines.append("")
    lines.append("[serialorder]")
    lines.append("serialorder = " + ".".join(str(i) for i in manifest["ordering"]["serial"]))
    lines.append("")
    lines.append("# NOTE: partial .magc projection — validate field names against a")
    lines.append("# real MagFinder/SBEMimage .magc before downstream use.")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


ADAPTERS = {
    "json_manifest": write_json_manifest,
    "csv_table": write_csv_table,
    "mvis_lmb": write_mvis_lmb,
    "magc": write_magc,
}


def write_all(manifest: dict, out_dir: str, adapters=None) -> dict:
    """Run the named adapters (default: json_manifest + csv_table + mvis_lmb).
    Returns ``{adapter_name: path}``."""
    os.makedirs(out_dir, exist_ok=True)
    names = adapters or ["json_manifest", "csv_table", "mvis_lmb"]
    out = {}
    for name in names:
        fn = ADAPTERS.get(name)
        if fn is not None:
            out[name] = fn(manifest, out_dir)
    return out
