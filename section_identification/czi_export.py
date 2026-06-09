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
                        section_ids: list | None = None) -> dict:
    """Produce a new CZI carrying STiM section polygons + fiducial markers.

    Copies ``src_czi`` -> ``dst_czi`` (preserving full-res pixels), injects a
    ``<Layers>`` block into the metadata via ``pylibCZIrw.edit_czi``, then
    re-reads the result to confirm the polygons survived (round-trip).

    Returns a report dict (``dst``, ``n_polygons``, ``n_fiducials``,
    ``roundtrip_ok``).
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

    # Commit metadata-only via the CziEditor (pylibCZIrw >= 6.0.0).
    _commit_metadata(dst_czi, new_xml)

    report = {
        "dst": dst_czi,
        "n_polygons": len(polygons),
        "n_fiducials": len(fiducials or []),
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
