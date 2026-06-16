"""Unified in-memory data model for a wafer and its sections.

This is the spine that carries a section through every workflow stage
(detect → proofread → ROIs → QC → reorder → imaging-order). It is **pure
Python** (numpy only) — no Qt, no napari, no pylibCZIrw — so it is unit-testable
headless and safe to import from workers.

Coordinate convention (matches the GUI's ``current_polygons_xy`` /
``napari_to_xy``): every geometry field stored here is ``(x, y)`` in
**overview pixels** — the working frame the napari layers display. Conversion to
full-resolution pixels or stage microns is done on demand via a ``geom`` object
(any object exposing ``ds_to_full`` / ``full_to_ds`` / ``full_to_stage_um`` —
i.e. :class:`section_identification.czi_io.CziGeometry`). Persistence to disk
stores the frame-independent full-resolution coordinates; that conversion lives
in :mod:`section_identification.project_io`, not here.

The model is the source of truth; napari layers are a *view* of it
(see ``layer_sync.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# Bumped when the on-disk project schema changes. A project file with no
# ``schema_version`` is a legacy (pre-wafer-model) file and loads via
# ``WaferProject.from_legacy``.
SCHEMA_VERSION = 2


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _poly_xy(points) -> list[list[float]]:
    """Normalise any polygon-ish input to ``[[x, y], ...]`` floats."""
    a = np.asarray(points, dtype=float).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in a]


def _identity_or(geom, attr: str):
    """Return ``getattr(geom, attr)`` or ``None`` when there is no geometry."""
    return None if geom is None else getattr(geom, attr, None)


# --------------------------------------------------------------------------- #
# per-section sub-records
# --------------------------------------------------------------------------- #
@dataclass
class Pose:
    """A section's canonical 2-D pose, recovered from polygon shape (align.py).

    ``center`` is ``(x, y)`` overview px; ``angle_deg`` is the rotation that
    brings the section's principal axis to canonical (upright); ``flip`` records
    the 180°/mirror disambiguation. Rotation/flip are scale-invariant, so they
    are identical in the overview and full-res frames; only ``center`` is frame
    dependent (converted at the disk boundary).
    """

    center: Optional[tuple[float, float]] = None
    angle_deg: float = 0.0
    flip: bool = False

    def to_dict(self) -> dict:
        return {"center": list(self.center) if self.center is not None else None,
                "angle_deg": float(self.angle_deg), "flip": bool(self.flip)}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Pose":
        if not d:
            return cls()
        c = d.get("center")
        return cls(center=(float(c[0]), float(c[1])) if c else None,
                   angle_deg=float(d.get("angle_deg", 0.0)),
                   flip=bool(d.get("flip", False)))


@dataclass
class FocusPoint:
    """An autofocus support point in **stage microns** (ZEN SupportPoint)."""

    x_um: float
    y_um: float
    z_um: float

    def to_dict(self) -> dict:
        return {"x_um": float(self.x_um), "y_um": float(self.y_um),
                "z_um": float(self.z_um)}

    @classmethod
    def from_dict(cls, d: dict) -> "FocusPoint":
        return cls(float(d["x_um"]), float(d["y_um"]), float(d["z_um"]))


@dataclass
class QCResult:
    """Quality-control outcome for one section (see wafer_qc.py).

    ``scores`` / ``flags`` are keyed by detector (debris/fold/shred/chatter/
    overall). ``features`` holds the raw, unit-bearing measurements so the GUI
    can re-threshold instantly without recomputing.
    """

    scores: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    params_used: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"scores": self.scores, "flags": self.flags,
                "features": self.features, "params_used": self.params_used}

    @classmethod
    def from_dict(cls, d: dict | None) -> Optional["QCResult"]:
        if not d:
            return None
        return cls(scores=dict(d.get("scores", {})), flags=dict(d.get("flags", {})),
                   features=dict(d.get("features", {})),
                   params_used=dict(d.get("params_used", {})))


@dataclass
class Roi:
    """A per-section region-of-interest polygon (overview px) for imaging.

    Produced by propagating a :class:`RoiTemplate`; ``fit_mode`` records how it
    was fit to this section (``template`` = raw propagation, ``full`` = scaled to
    the section extent, ``percent`` = scaled to ``fit_percent`` %, ``clip`` =
    intersected with the section polygon, ``manual`` = user-edited).
    """

    polygon: list = field(default_factory=list)
    fit_mode: str = "template"
    fit_percent: float = 100.0

    def to_dict(self) -> dict:
        return {"polygon": _poly_xy(self.polygon), "fit_mode": self.fit_mode,
                "fit_percent": float(self.fit_percent)}

    @classmethod
    def from_dict(cls, d: dict | None) -> Optional["Roi"]:
        if not d:
            return None
        return cls(polygon=_poly_xy(d.get("polygon", [])),
                   fit_mode=d.get("fit_mode", "template"),
                   fit_percent=float(d.get("fit_percent", 100.0)))


@dataclass
class Section:
    """One detected tissue section, carrying all per-stage state."""

    id: str
    polygon: list                                   # (x,y) overview px, canonical
    pose: Pose = field(default_factory=Pose)
    roi: Optional[Roi] = None
    focus_overview: list = field(default_factory=list)  # (x,y) overview px, editable
    focus_points: list = field(default_factory=list)  # FocusPoint (stage µm, from CZI)
    qc: Optional[QCResult] = None
    serial_index: Optional[int] = None              # stage 4 reorder
    imaging_index: Optional[int] = None             # stage 4 TSP
    accepted: bool = True                            # proofread / QC accept-reject
    sift_ref: Optional[str] = None                   # cached-descriptor path

    # -- geometry helpers (overview frame) --
    def polygon_array(self) -> np.ndarray:
        return np.asarray(self.polygon, dtype=float).reshape(-1, 2)

    def centroid(self) -> tuple[float, float]:
        p = self.polygon_array()
        return (float(p[:, 0].mean()), float(p[:, 1].mean()))

    def area(self) -> float:
        """Shoelace polygon area in overview px²."""
        p = self.polygon_array()
        if len(p) < 3:
            return 0.0
        x, y = p[:, 0], p[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)

    def bbox(self) -> tuple[float, float, float, float]:
        """(x0, y0, x1, y1) overview-px bounding box."""
        p = self.polygon_array()
        return (float(p[:, 0].min()), float(p[:, 1].min()),
                float(p[:, 0].max()), float(p[:, 1].max()))

    # -- frame conversions (need a geom; identity when geom is None) --
    def polygon_full(self, geom) -> list[list[float]]:
        if geom is None:
            return _poly_xy(self.polygon)
        p = self.polygon_array()
        fx, fy = geom.ds_to_full(p[:, 0], p[:, 1])
        return [[float(a), float(b)] for a, b in zip(np.ravel(fx), np.ravel(fy))]

    def centroid_stage_um(self, geom) -> Optional[tuple[float, float]]:
        if geom is None:
            return None
        cx, cy = self.centroid()
        fx, fy = geom.ds_to_full(np.asarray([cx]), np.asarray([cy]))
        s = geom.full_to_stage_um(np.ravel(fx), np.ravel(fy))
        if s is None:
            return None
        return (float(np.ravel(s[0])[0]), float(np.ravel(s[1])[0]))

    def to_dict(self) -> dict:
        return {"id": self.id, "polygon": _poly_xy(self.polygon),
                "pose": self.pose.to_dict(),
                "roi": self.roi.to_dict() if self.roi else None,
                "focus_overview": _poly_xy(self.focus_overview),
                "focus_points": [fp.to_dict() for fp in self.focus_points],
                "qc": self.qc.to_dict() if self.qc else None,
                "serial_index": self.serial_index,
                "imaging_index": self.imaging_index,
                "accepted": bool(self.accepted), "sift_ref": self.sift_ref}

    @classmethod
    def from_dict(cls, d: dict) -> "Section":
        return cls(
            id=str(d["id"]), polygon=_poly_xy(d.get("polygon", [])),
            pose=Pose.from_dict(d.get("pose")),
            roi=Roi.from_dict(d.get("roi")),
            focus_overview=_poly_xy(d.get("focus_overview", [])),
            focus_points=[FocusPoint.from_dict(fp) for fp in d.get("focus_points", [])],
            qc=QCResult.from_dict(d.get("qc")),
            serial_index=d.get("serial_index"), imaging_index=d.get("imaging_index"),
            accepted=bool(d.get("accepted", True)), sift_ref=d.get("sift_ref"))


# --------------------------------------------------------------------------- #
# wafer-level records
# --------------------------------------------------------------------------- #
@dataclass
class RoiTemplate:
    """A reusable ROI defined in a reference section's pose-normalised frame.

    ``polygon_local`` is the ROI in the reference section's canonical (upright,
    centered) frame; propagation maps it onto each section via that section's
    pose. The remaining fields are the ZEN mFOV/autofocus acquisition knobs.
    """

    polygon_local: list = field(default_factory=list)
    focus_local: list = field(default_factory=list)        # (x,y) focus pts, local frame
    ref_section_id: Optional[str] = None
    fit_mode: str = "full"
    fit_percent: float = 100.0
    tile_um: Optional[tuple[float, float]] = None     # mFOV/tile footprint (µm)
    overlap: float = 0.1                               # tile overlap fraction
    target_px_nm: Optional[float] = None               # target pixel size
    focus_cols: int = 2                                # SupportPoint grid
    focus_rows: int = 2

    def to_dict(self) -> dict:
        return {"polygon_local": _poly_xy(self.polygon_local),
                "focus_local": _poly_xy(self.focus_local),
                "ref_section_id": self.ref_section_id, "fit_mode": self.fit_mode,
                "fit_percent": float(self.fit_percent),
                "tile_um": list(self.tile_um) if self.tile_um else None,
                "overlap": float(self.overlap), "target_px_nm": self.target_px_nm,
                "focus_cols": int(self.focus_cols), "focus_rows": int(self.focus_rows)}

    @classmethod
    def from_dict(cls, d: dict) -> "RoiTemplate":
        t = d.get("tile_um")
        return cls(polygon_local=_poly_xy(d.get("polygon_local", [])),
                   focus_local=_poly_xy(d.get("focus_local", [])),
                   ref_section_id=d.get("ref_section_id"),
                   fit_mode=d.get("fit_mode", "full"),
                   fit_percent=float(d.get("fit_percent", 100.0)),
                   tile_um=(float(t[0]), float(t[1])) if t else None,
                   overlap=float(d.get("overlap", 0.1)),
                   target_px_nm=d.get("target_px_nm"),
                   focus_cols=int(d.get("focus_cols", 2)),
                   focus_rows=int(d.get("focus_rows", 2)))


@dataclass
class MatchEdge:
    """One SIFT pairwise match between two sections (reorder stage)."""

    a: str
    b: str
    inliers: int = 0
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "inliers": int(self.inliers),
                "confidence": float(self.confidence)}

    @classmethod
    def from_dict(cls, d: dict) -> "MatchEdge":
        return cls(a=str(d["a"]), b=str(d["b"]), inliers=int(d.get("inliers", 0)),
                   confidence=float(d.get("confidence", 0.0)))


@dataclass
class MatchGraph:
    """The SIFT similarity graph + recovered serial order (reorder stage).

    Kept separate from the section list so it survives section add/delete; edges
    reference sections by ``id``.
    """

    edges: list = field(default_factory=list)          # MatchEdge
    order: list = field(default_factory=list)          # section ids, serial order
    method: Optional[str] = None
    similarity_path: Optional[str] = None              # cached .npz sidecar

    def to_dict(self) -> dict:
        return {"edges": [e.to_dict() for e in self.edges], "order": list(self.order),
                "method": self.method, "similarity_path": self.similarity_path}

    @classmethod
    def from_dict(cls, d: dict | None) -> "MatchGraph":
        if not d:
            return cls()
        return cls(edges=[MatchEdge.from_dict(e) for e in d.get("edges", [])],
                   order=list(d.get("order", [])), method=d.get("method"),
                   similarity_path=d.get("similarity_path"))


@dataclass
class WaferProject:
    """The whole project: sections + wafer-level state. Source of truth."""

    image_path: Optional[str] = None
    schema_version: int = SCHEMA_VERSION
    sections: list = field(default_factory=list)            # Section
    fiducials: list = field(default_factory=list)            # (x,y) overview px
    raw_sections: list = field(default_factory=list)         # polygons overview px
    calibration_examples: list = field(default_factory=list)  # polygons overview px
    roi_templates: list = field(default_factory=list)         # RoiTemplate
    match_graph: MatchGraph = field(default_factory=MatchGraph)
    qc_summary: dict = field(default_factory=dict)

    # -- section management --
    def get(self, sid: str) -> Optional[Section]:
        for s in self.sections:
            if s.id == sid:
                return s
        return None

    def new_id(self) -> str:
        """Next free ``section_N`` id (stable; never reuses)."""
        used = set()
        for s in self.sections:
            if s.id.startswith("section_"):
                try:
                    used.add(int(s.id.rsplit("_", 1)[1]))
                except ValueError:
                    pass
        n = 1
        while n in used:
            n += 1
        return f"section_{n}"

    def add_section(self, polygon, sid: Optional[str] = None) -> Section:
        s = Section(id=sid or self.new_id(), polygon=_poly_xy(polygon))
        self.sections.append(s)
        return s

    def set_sections_from_polygons(self, polygons) -> None:
        """Replace the section list from a list of overview-px polygons,
        assigning fresh stable ids in order."""
        self.sections = []
        for poly in polygons:
            self.add_section(poly)

    def in_serial_order(self) -> list[Section]:
        if self.match_graph.order:
            by_id = {s.id: s for s in self.sections}
            ordered = [by_id[i] for i in self.match_graph.order if i in by_id]
            if ordered:
                return ordered
        keyed = [s for s in self.sections if s.serial_index is not None]
        return sorted(keyed, key=lambda s: s.serial_index) or list(self.sections)

    def in_imaging_order(self) -> list[Section]:
        keyed = [s for s in self.sections if s.imaging_index is not None]
        return sorted(keyed, key=lambda s: s.imaging_index) or list(self.sections)

    # -- manual order / route editing (pure; the GUI calls these) --
    def swap_serial(self, id_a: str, id_b: str) -> bool:
        """Swap two sections' serial order (their serial_index and their slots in
        the recovered match-graph order). Returns True on success."""
        sa, sb = self.get(id_a), self.get(id_b)
        if sa is None or sb is None or sa is sb:
            return False
        sa.serial_index, sb.serial_index = sb.serial_index, sa.serial_index
        o = self.match_graph.order
        if id_a in o and id_b in o:
            ia, ib = o.index(id_a), o.index(id_b)
            o[ia], o[ib] = o[ib], o[ia]
        return True

    def drop_from_imaging(self, sid: str) -> bool:
        """Remove a section from the imaging route and compact the rest 0..n-1."""
        s = self.get(sid)
        if s is None or s.imaging_index is None:
            return False
        s.imaging_index = None
        remaining = sorted((x for x in self.sections if x.imaging_index is not None),
                           key=lambda x: x.imaging_index)
        for i, x in enumerate(remaining):
            x.imaging_index = i
        return True

    def move_imaging(self, sid: str, delta: int) -> bool:
        """Move a section earlier/later in the imaging route by ``delta`` steps."""
        ordered = [x for x in self.sections if x.imaging_index is not None]
        ordered.sort(key=lambda x: x.imaging_index)
        ids = [x.id for x in ordered]
        if sid not in ids:
            return False
        i = ids.index(sid)
        j = i + delta
        if not (0 <= j < len(ordered)):
            return False
        a, b = ordered[i], ordered[j]
        a.imaging_index, b.imaging_index = b.imaging_index, a.imaging_index
        return True

    def reverse_imaging(self) -> None:
        """Reverse the imaging route order."""
        ordered = sorted((x for x in self.sections if x.imaging_index is not None),
                         key=lambda x: x.imaging_index)
        n = len(ordered)
        for i, x in enumerate(ordered):
            x.imaging_index = n - 1 - i

    # -- serialisation (frame-agnostic: serialises whatever is in the fields) --
    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "image": self.image_path,
            "sections": [s.to_dict() for s in self.sections],
            "fiducials": [[float(x), float(y)] for x, y in self.fiducials],
            "raw_sections": [_poly_xy(p) for p in self.raw_sections],
            "calibration_examples": [_poly_xy(p) for p in self.calibration_examples],
            "roi_templates": [t.to_dict() for t in self.roi_templates],
            "match_graph": self.match_graph.to_dict(),
            "qc_summary": self.qc_summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WaferProject":
        return cls(
            image_path=d.get("image"),
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            sections=[Section.from_dict(s) for s in d.get("sections", [])],
            fiducials=[(float(p[0]), float(p[1])) for p in d.get("fiducials", [])],
            raw_sections=[_poly_xy(p) for p in d.get("raw_sections", [])],
            calibration_examples=[_poly_xy(p) for p in d.get("calibration_examples", [])],
            roi_templates=[RoiTemplate.from_dict(t) for t in d.get("roi_templates", [])],
            match_graph=MatchGraph.from_dict(d.get("match_graph")),
            qc_summary=dict(d.get("qc_summary", {})),
        )

    def apply_results(self, source: "WaferProject") -> "WaferProject":
        """Merge per-stage results from ``source`` into this project, matching
        sections by ``id`` (geometry stays this project's — from the live layer).
        Used to restore saved QC/order/ROI/pose state after a reload."""
        by_id = {s.id: s for s in source.sections}
        for t in self.sections:
            s = by_id.get(t.id)
            if s is None:
                continue
            t.qc = s.qc
            t.serial_index = s.serial_index
            t.imaging_index = s.imaging_index
            t.roi = s.roi
            t.focus_overview = s.focus_overview
            t.focus_points = s.focus_points
            t.accepted = s.accepted
            if s.pose.center is not None:
                t.pose = s.pose
        self.match_graph = source.match_graph
        self.roi_templates = source.roi_templates
        self.qc_summary = source.qc_summary
        return self

    @classmethod
    def from_legacy(cls, d: dict, to_overview) -> "WaferProject":
        """Build a project from a legacy (pre-wafer-model) project dict.

        Legacy files store ``sections``/``fiducials``/``raw_sections``/
        ``calibration_examples`` as **full-resolution** polygons and have no
        ``schema_version``. ``to_overview`` is a callable mapping a full-res
        polygon/point array to overview px (the GUI's ``_to_overview``; identity
        for non-CZI). Sections get fresh stable ids; no QC/order/ROI state.
        """
        proj = cls(image_path=d.get("image"))
        for poly in d.get("sections", []):
            proj.add_section(to_overview(poly))
        for f in d.get("fiducials", []):
            ov = np.asarray(to_overview([f]), dtype=float).reshape(-1, 2)[0]
            proj.fiducials.append((float(ov[0]), float(ov[1])))
        proj.raw_sections = [_poly_xy(to_overview(p)) for p in d.get("raw_sections", [])]
        proj.calibration_examples = [_poly_xy(to_overview(p))
                                     for p in d.get("calibration_examples", [])]
        return proj
