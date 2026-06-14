"""Write section polygons + fiducials into a CZI for ZEN (Shuttle & Find).

ZEN stores graphic annotations in the CZI metadata XML, under a
``…/Metadata/[MetadataNodes/MetadataNode/]Layers/Layer/Elements`` tree. Each
section becomes a closed ``<Polygon>`` whose ``<Geometry><Points>`` is a
space-separated list of ``x,y`` pixel pairs (full-resolution image frame — the
frame ZEN's annotations use). Element names follow Bio-Formats'
``ZeissCZIReader.translateLayers`` (the most complete public reference) and the
ZISRAW spec's ``<Bezier>``/``<Polygon>``/``<Points>`` sample.

We do NOT rewrite the 13 GB of pixels: we copy the source CZI and edit only the
metadata in place via ``pylibCZIrw``'s ``edit_czi`` / ``CziEditor`` (>= 6.0.0).

IMPORTANT (needs-verification): whether a given ZEN build renders externally
injected ``<Layers>``, and whether Shuttle & Find treats the fiducial markers as
calibration POIs, must be confirmed empirically. The 3-point LM<->SEM
calibration itself is an interactive per-instrument step with no documented
external file format. This module produces a best-effort ZEN-readable CZI and a
machine-readable GeoJSON sidecar; verify in ZEN and iterate.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np

SECTIONS_LAYER = "STiM_Sections"
FIDUCIALS_LAYER = "STiM_Fiducials"


# --------------------------------------------------------------------------- #
# XML construction (pure, unit-testable without pylibCZIrw)
# --------------------------------------------------------------------------- #
def format_points(points) -> str:
    """``[(x,y), ...]`` -> ``"x1,y1 x2,y2 ..."`` with 2-dp coordinates."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)


def _polygon_element(idx: int, points, name: str | None = None) -> ET.Element:
    poly = ET.Element("Polygon", attrib={"Id": str(idx)})
    geom = ET.SubElement(poly, "Geometry")
    ET.SubElement(geom, "Points").text = format_points(points)
    if name:
        attrs = ET.SubElement(poly, "Attributes")
        ET.SubElement(attrs, "Name").text = name
        texts = ET.SubElement(poly, "TextElements")
        ET.SubElement(texts, "TextElement").text = name
    return poly


def _marker_element(idx: int, x: float, y: float, radius: float,
                    name: str | None = None) -> ET.Element:
    """A fiducial as a small <Ellipse> (Bio-Formats: CenterX/Y, RadiusX/Y)."""
    el = ET.Element("Ellipse", attrib={"Id": str(idx)})
    geom = ET.SubElement(el, "Geometry")
    ET.SubElement(geom, "CenterX").text = f"{x:.2f}"
    ET.SubElement(geom, "CenterY").text = f"{y:.2f}"
    ET.SubElement(geom, "RadiusX").text = f"{radius:.2f}"
    ET.SubElement(geom, "RadiusY").text = f"{radius:.2f}"
    if name:
        attrs = ET.SubElement(el, "Attributes")
        ET.SubElement(attrs, "Name").text = name
    return el


def _make_layer(name: str, elements: list[ET.Element]) -> ET.Element:
    layer = ET.Element("Layer", attrib={"Name": name})
    ET.SubElement(layer, "Usage").text = "Annotation"
    ET.SubElement(layer, "IsProtected").text = "false"
    container = ET.SubElement(layer, "Elements")
    for e in elements:
        container.append(e)
    return layer


def inject_layers(xml_str: str, polygons, fiducials,
                  fiducial_radius_px: float = 50.0,
                  section_ids: list | None = None) -> str:
    """Return CZI metadata XML with STiM section/fiducial Layers added.

    ``polygons`` is a list of polygons, each ``[(x,y), ...]`` in full-res pixels.
    ``fiducials`` is a list of ``(x,y)`` in full-res pixels. Existing ``<Layers>``
    are reused (our Layers are appended); otherwise a ``<Layers>`` is created
    under ``<Metadata>``.
    """
    root = ET.fromstring(xml_str)

    # Find an existing <Layers> anywhere, else create one under <Metadata>.
    layers = root.find(".//Layers")
    if layers is None:
        metadata = root.find("Metadata")
        if metadata is None:  # extremely defensive
            metadata = root
        layers = ET.SubElement(metadata, "Layers")

    # Idempotent re-export: drop any STiM layers a previous run wrote so we
    # don't accumulate stale duplicates (e.g. an old, smaller section count).
    for layer in list(layers.findall("Layer")):
        if layer.get("Name") in (SECTIONS_LAYER, FIDUCIALS_LAYER):
            layers.remove(layer)

    if polygons:
        poly_elems = []
        for i, poly in enumerate(polygons, start=1):
            label = None
            if section_ids is not None and i - 1 < len(section_ids):
                label = str(section_ids[i - 1])
            else:
                label = f"section_{i}"
            poly_elems.append(_polygon_element(i, poly, name=label))
        layers.append(_make_layer(SECTIONS_LAYER, poly_elems))

    if fiducials:
        fid_elems = [
            _marker_element(i, float(x), float(y), fiducial_radius_px,
                            name=f"fiducial_{i}")
            for i, (x, y) in enumerate(fiducials, start=1)
        ]
        layers.append(_make_layer(FIDUCIALS_LAYER, fid_elems))

    return ET.tostring(root, encoding="unicode")


# --------------------------------------------------------------------------- #
# Shuttle & Find correlative markers (stage µm, in the calibration node)
# --------------------------------------------------------------------------- #
def _find_child(parent, *paths):
    """First node matching any path, using explicit ``is not None`` (an
    ElementTree element with no children is falsy, so a ``find() or find()``
    chain would skip a present-but-empty element)."""
    for p in paths:
        el = parent.find(p)
        if el is not None:
            return el
    return None


def _max_marker_index(markers) -> int:
    """Largest trailing integer across existing ``Marker`` ``Id``s (0 if none)."""
    mx = 0
    for m in markers:
        mm = re.search(r"(\d+)\s*$", m.get("Id") or "")
        if mm:
            mx = max(mx, int(mm.group(1)))
    return mx


def inject_shuttle_and_find(xml_str: str, markers_stage_um,
                            orientation: tuple | None = None,
                            holder: str | None = None,
                            microscope: str | None = None,
                            session_id: str | None = None,
                            replace: bool = True) -> str:
    """Return CZI metadata XML with the S&F correlative ``<Markers>`` set.

    ``markers_stage_um`` is a list of ``(stage_x_um, stage_y_um)`` or
    ``(x, y, focus_um)`` tuples — the frame ZEN's Shuttle & Find calibration
    uses (see :func:`czi_io.read_shuttle_and_find_markers`). If a
    ``ShuttleAndFindData/Calibration`` node already exists its ``<Markers>`` are
    replaced (``replace=True``) while ``CorrelativeSession``/``Holder``/
    ``MicroscopeType``/``StageOrientation`` are preserved (any passed explicitly
    are overwritten); otherwise the node is created under ``<Image>``. New markers
    without a focus inherit the mean focus of the markers being replaced (keeping
    the calibration Z-plane) when one is available. With ``replace=False`` new
    ``Id``s continue past the largest existing index (no duplicate ``Id``s).
    """
    root = ET.fromstring(xml_str)
    cal = root.find(".//ShuttleAndFindData/Calibration")
    if cal is None:
        parent = _find_child(root, ".//Metadata/Information/Image",
                             ".//Information/Image", ".//Image", "Metadata")
        if parent is None:           # NB: `_find_child(...) or root` would re-trip
            parent = root            # the falsy-empty-element trap _find_child avoids
        saf = ET.SubElement(parent, "ShuttleAndFindData")
        cal = ET.SubElement(saf, "Calibration")
        ET.SubElement(cal, "CorrelativeSession",
                      {"CorrelativeSessionId": session_id or "STiM"})
        ET.SubElement(cal, "Holder").text = holder or "STiM"
        ET.SubElement(cal, "Markers")
        ET.SubElement(cal, "MicroscopeType").text = microscope or "LM"
        ox, oy = orientation or (1, 1)
        ET.SubElement(cal, "StageOrientation",
                      {"X": str(int(ox)), "Y": str(int(oy))})

    # Update preserved fields only when explicitly overridden (explicit is-None
    # checks — a childless element is falsy, so `find() or SubElement()` dupes).
    if orientation is not None:
        so = cal.find("StageOrientation")
        if so is None:
            so = ET.SubElement(cal, "StageOrientation")
        so.set("X", str(int(orientation[0]))); so.set("Y", str(int(orientation[1])))
    if holder is not None:
        h = cal.find("Holder")
        if h is None:
            h = ET.SubElement(cal, "Holder")
        h.text = holder
    if microscope is not None:
        mt = cal.find("MicroscopeType")
        if mt is None:
            mt = ET.SubElement(cal, "MicroscopeType")
        mt.text = microscope

    # Find the markers container with the DESCENDANT axis (ZEN may nest it under
    # CorrelativeSession; our parser reads it with `.//Markers/Marker`).
    markers_el = cal.find(".//Markers")
    if markers_el is None:
        markers_el = ET.SubElement(cal, "Markers")

    # Capture the existing markers' focus so replacements keep the Z-plane.
    prev_focus = []
    for m in markers_el.findall("Marker"):
        try:
            prev_focus.append(float(m.get("FocusPosition")))
        except (TypeError, ValueError):
            pass
    default_focus = sum(prev_focus) / len(prev_focus) if prev_focus else None

    if replace:
        for m in list(markers_el.findall("Marker")):
            markers_el.remove(m)
    start = _max_marker_index(markers_el.findall("Marker"))
    for i, mk in enumerate(markers_stage_um, start=start + 1):
        sx, sy = float(mk[0]), float(mk[1])
        attrib = {"Id": f"Marker:{i}",
                  "StageXPosition": f"{sx:.6f}", "StageYPosition": f"{sy:.6f}"}
        focus = mk[2] if (len(mk) >= 3 and mk[2] is not None) else default_focus
        if focus is not None:
            attrib["FocusPosition"] = f"{float(focus):.6f}"
        ET.SubElement(markers_el, "Marker", attrib)
    return ET.tostring(root, encoding="unicode")


# --------------------------------------------------------------------------- #
# Tile regions + focus support points (ZEN mFOV acquisition, stage microns)
# --------------------------------------------------------------------------- #
# ZEN places mFOVs from TileRegion nodes under
#   …/Experiment/.../RegionsSetup/SampleHolder/TileRegions/TileRegion
# CenterPosition/ContourSize/Z are stage microns; Columns/Rows are tile counts;
# SupportPoints are the autofocus plane (stage µm X/Y/Z). Verified against real
# CZIs. ZEN-build acceptance of externally injected regions must be confirmed
# empirically (same caveat as the annotation Layers above).
TILE_REGION_PREFIX = "STiM_TR_"


def _support_point_element(idx: int, x: float, y: float, z: float) -> ET.Element:
    sp = ET.Element("SupportPoint", attrib={"Name": "SP", "Id": f"STiM_SP_{idx}"})
    ET.SubElement(sp, "X").text = f"{float(x):.3f}"
    ET.SubElement(sp, "Y").text = f"{float(y):.3f}"
    ET.SubElement(sp, "Z").text = f"{float(z):.3f}"
    ET.SubElement(sp, "AdditionalValues")
    return sp


def _tile_region_element(idx: int, spec: dict) -> ET.Element:
    """Build a <TileRegion> from a spec dict: ``center_um=(x,y)``,
    ``contour_um=(w,h)``, ``columns``, ``rows``, ``z_um``,
    ``support_points=[(x,y,z), ...]`` (stage µm), optional ``name``/``contour``."""
    name = spec.get("name") or f"{TILE_REGION_PREFIX}{idx}"
    tr = ET.Element("TileRegion", attrib={"Name": name, "Id": str(spec.get("id", idx))})
    cx, cy = spec["center_um"]
    cw, ch = spec.get("contour_um", (0.0, 0.0))
    ET.SubElement(tr, "CenterPosition").text = f"{float(cx):.3f},{float(cy):.3f}"
    ET.SubElement(tr, "ContourSize").text = f"{float(cw):.3f},{float(ch):.3f}"
    ET.SubElement(tr, "Columns").text = str(int(spec.get("columns", 1)))
    ET.SubElement(tr, "Rows").text = str(int(spec.get("rows", 1)))
    ET.SubElement(tr, "Z").text = f"{float(spec.get('z_um', 0.0)):.3f}"
    ET.SubElement(tr, "Contour", attrib={"Type": spec.get("contour", "Rectangle")})
    sps = spec.get("support_points") or []
    if sps:
        container = ET.SubElement(tr, "SupportPoints")
        for k, (sx, sy, sz) in enumerate(sps, start=1):
            container.append(_support_point_element(idx * 100 + k, sx, sy, sz))
    ET.SubElement(tr, "AdditionalValues")
    return tr


def _ensure_tile_regions(root: ET.Element) -> ET.Element:
    """Return the <TileRegions> container, reusing an existing one or creating
    the RegionsSetup/SampleHolder/TileRegions chain (best-effort) if absent."""
    tr = root.find(".//TileRegions")
    if tr is not None:
        return tr
    sh = root.find(".//SampleHolder")
    if sh is None:
        rs = root.find(".//RegionsSetup")
        if rs is None:
            md = root.find("Metadata")
            if md is None:
                md = root
            node = md
            for tag in ("Experiment", "ExperimentBlocks", "AcquisitionBlock",
                        "SubDimensionSetups", "RegionsSetup", "SampleHolder"):
                child = node.find(tag)
                if child is None:
                    child = ET.SubElement(node, tag)
                node = child
            sh = node
        else:
            sh = ET.SubElement(rs, "SampleHolder")
    return ET.SubElement(sh, "TileRegions")


def inject_tile_regions(xml_str: str, regions, replace: bool = True) -> str:
    """Return CZI metadata XML with STiM TileRegions (one per ROI) added.

    ``regions`` is a list of spec dicts (see :func:`_tile_region_element`).
    Previous STiM TileRegions (``Name`` starting ``STiM_TR_``) are removed first
    when ``replace`` so re-export doesn't accumulate duplicates.
    """
    root = ET.fromstring(xml_str)
    container = _ensure_tile_regions(root)
    if replace:
        for tr in list(container.findall("TileRegion")):
            if (tr.get("Name") or "").startswith(TILE_REGION_PREFIX):
                container.remove(tr)
    for i, spec in enumerate(regions, start=1):
        container.append(_tile_region_element(i, spec))
    return ET.tostring(root, encoding="unicode")


def read_tile_regions(xml_str: str) -> list[dict]:
    """Read TileRegions back from CZI metadata XML. Returns a list of dicts with
    ``center_um``/``contour_um``/``columns``/``rows``/``z_um``/``support_points``
    (stage µm) and ``name`` — used to display existing mFOVs/focus points on load.
    """
    root = ET.fromstring(xml_str)
    out = []

    def _pair(text):
        try:
            a, b = (float(v) for v in text.split(","))
            return (a, b)
        except Exception:
            return None

    for tr in root.findall(".//TileRegion"):
        cp = tr.findtext("CenterPosition")
        cs = tr.findtext("ContourSize")
        sps = []
        for sp in tr.findall(".//SupportPoint"):
            try:
                sps.append((float(sp.findtext("X")), float(sp.findtext("Y")),
                            float(sp.findtext("Z"))))
            except (TypeError, ValueError):
                continue
        out.append({
            "name": tr.get("Name"),
            "center_um": _pair(cp) if cp else None,
            "contour_um": _pair(cs) if cs else None,
            "columns": int(tr.findtext("Columns") or 0),
            "rows": int(tr.findtext("Rows") or 0),
            "z_um": float(tr.findtext("Z") or 0.0),
            "support_points": sps,
        })
    return out


# --------------------------------------------------------------------------- #
# GeoJSON sidecar (pixel + optional stage microns)
# --------------------------------------------------------------------------- #
def write_geojson(path: str, polygons, fiducials, geom=None) -> str:
    """Write a GeoJSON FeatureCollection (pixel coords; stage-um if available)."""
    features = []

    def _stage(xs, ys):
        if geom is None:
            return None
        s = geom.full_to_stage_um(np.asarray(xs), np.asarray(ys))
        if s is None:
            return None
        return [[float(a), float(b)] for a, b in zip(s[0].ravel(), s[1].ravel())]

    for i, poly in enumerate(polygons, start=1):
        ring = [[float(x), float(y)] for x, y in np.asarray(poly).reshape(-1, 2)]
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])  # GeoJSON polygons are closed
        props = {"id": f"section_{i}", "type": "section"}
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        st = _stage(xs, ys)
        if st is not None:
            props["stage_um"] = st
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": props,
        })

    for i, (x, y) in enumerate(fiducials or [], start=1):
        props = {"id": f"fiducial_{i}", "type": "fiducial"}
        st = _stage([x], [y])
        if st is not None:
            props["stage_um"] = st[0]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(x), float(y)]},
            "properties": props,
        })

    fc = {"type": "FeatureCollection", "features": features}
    with open(path, "w") as f:
        json.dump(fc, f, indent=2)
    return path


# --------------------------------------------------------------------------- #
# CZI writing (copy + metadata-only edit) + round-trip check
# --------------------------------------------------------------------------- #
def _read_metadata_xml(path: str) -> str:
    from pylibCZIrw import czi as pyczi
    with pyczi.open_czi(path) as cz:
        return cz.raw_metadata


def write_annotated_czi(src_czi: str, dst_czi: str, polygons, fiducials,
                        copy: bool = True, fiducial_radius_px: float = 50.0,
                        section_ids: list | None = None,
                        sf_markers_stage_um=None, sf_orientation=None,
                        tile_regions=None) -> dict:
    """Produce a new CZI carrying STiM section polygons + fiducial markers.

    Copies ``src_czi`` -> ``dst_czi`` (preserving full-res pixels), injects a
    ``<Layers>`` block into the metadata via ``pylibCZIrw.edit_czi``, then
    re-reads the result to confirm the polygons survived (round-trip).

    When ``sf_markers_stage_um`` is given (a list of ``(x_um, y_um[, focus])``),
    those are ALSO written into the ZEN Shuttle & Find calibration ``<Markers>``
    node — i.e. the fiducials become correlative POIs ZEN can use. This edits
    only the destination COPY, never ``src_czi``, so the source's calibration is
    untouched. ``sf_orientation`` overrides the ``StageOrientation`` sign.

    Returns a report dict (``dst``, ``n_polygons``, ``n_fiducials``,
    ``n_sf_markers``, ``roundtrip_ok``).
    """
    if copy:
        if os.path.abspath(src_czi) != os.path.abspath(dst_czi):
            shutil.copy2(src_czi, dst_czi)
    elif not os.path.exists(dst_czi):
        raise FileNotFoundError(dst_czi)

    # Build the new metadata XML.
    old_xml = _read_metadata_xml(dst_czi)
    new_xml = inject_layers(old_xml, polygons, fiducials,
                            fiducial_radius_px=fiducial_radius_px,
                            section_ids=section_ids)
    if sf_markers_stage_um:
        new_xml = inject_shuttle_and_find(new_xml, sf_markers_stage_um,
                                          orientation=sf_orientation)
    if tile_regions:
        new_xml = inject_tile_regions(new_xml, tile_regions)

    # Commit metadata-only via the CziEditor (pylibCZIrw >= 6.0.0).
    _commit_metadata(dst_czi, new_xml)

    report = {
        "n_tile_regions": len(tile_regions or []),
        "dst": dst_czi,
        "n_polygons": len(polygons),
        "n_fiducials": len(fiducials or []),
        "n_sf_markers": len(sf_markers_stage_um or []),
        "roundtrip_ok": roundtrip_check(dst_czi, expect_polygons=len(polygons)),
    }
    return report


def _commit_metadata(dst_czi: str, new_xml: str) -> None:
    """Write ``new_xml`` as the CZI's metadata in place.

    Tries the documented ``edit_czi`` / ``create_metadata_builder`` API and a
    couple of plausible variants, so small naming differences across pylibCZIrw
    point releases don't break us. Raises with a helpful message otherwise.
    """
    from pylibCZIrw import czi as pyczi

    edit_czi = getattr(pyczi, "edit_czi", None)
    if edit_czi is None:
        raise RuntimeError(
            "pylibCZIrw.edit_czi not found — need pylibCZIrw>=6.0.0 for in-place "
            "metadata editing. Available czi attrs: "
            + ", ".join(a for a in dir(pyczi) if not a.startswith("_"))
        )

    with edit_czi(dst_czi) as editor:
        builder = editor.create_metadata_builder()
        # set the full metadata XML
        set_xml = getattr(builder, "set_xml", None)
        if set_xml is None:
            raise RuntimeError(
                "metadata builder has no set_xml; attrs: "
                + ", ".join(a for a in dir(builder) if not a.startswith("_"))
            )
        set_xml(new_xml)
        can_commit = getattr(builder, "can_commit", None)
        if can_commit is not None and not can_commit():
            raise RuntimeError("CziEditor reports metadata is not committable.")
        builder.commit()


def read_annotations(czi_path: str):
    """Read STiM annotations back from a CZI's ``<Layers>``.

    Returns ``(polygons, fiducials)`` in full-resolution pixel coords:
    ``polygons`` is a list of ``[(x,y), ...]`` (from ``<Polygon><Points>``) and
    ``fiducials`` a list of ``(x,y)`` (from ``<Ellipse>`` centres). Lets the GUI
    reopen an annotated CZI and show/edit what was saved.
    """
    xml = _read_metadata_xml(czi_path)
    root = ET.fromstring(xml)
    polygons = []
    for p in root.findall(".//Polygon"):
        pts_el = p.find("Geometry/Points")
        if pts_el is None or not pts_el.text:
            continue
        pts = []
        for token in pts_el.text.split():
            if "," in token:
                x, y = token.split(",")
                pts.append((float(x), float(y)))
        if len(pts) >= 3:
            polygons.append(pts)
    fiducials = []
    for e in root.findall(".//Ellipse"):
        g = e.find("Geometry")
        if g is None:
            continue
        cx, cy = g.findtext("CenterX"), g.findtext("CenterY")
        if cx is not None and cy is not None:
            fiducials.append((float(cx), float(cy)))
    return polygons, fiducials


def roundtrip_check(dst_czi: str, expect_polygons: int | None = None) -> bool:
    """Re-read the CZI metadata and confirm STiM polygons are present."""
    try:
        xml = _read_metadata_xml(dst_czi)
    except Exception:
        return False
    root = ET.fromstring(xml)
    polys = root.findall(".//Polygon")
    if expect_polygons is not None:
        return len(polys) >= expect_polygons and expect_polygons > 0
    return len(polys) > 0
