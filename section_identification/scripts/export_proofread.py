"""Export the proofread tardigrade wafer from the autosaved project JSON.

The GUI autosaves section polygons + fiducials (full-resolution coords) to
``*_stim_project.json``. This produces the deliverables from that saved state:

  fast : CSV + GeoJSON + high-resolution overlay PNG (polygons + fiducial crosses)
  czi  : annotated CZI copy for ZEN Shuttle & Find (sections + fiducial markers)
  all  : both

Usage:  python -m section_identification.scripts.export_proofread [fast|czi|all] [png_long_side]
"""
import os
import sys
import json

import numpy as np

from section_identification import czi_io, czi_export
from section_identification import export as exp

PROJ = ("/Users/fredericoaraujo/Documents/tard_carbon_coat_001_STiM_files/"
        "tard_carbon_coat_001_STiM_stim_project.json")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "fast"
    png_long_side = int(sys.argv[2]) if len(sys.argv) > 2 else 24000

    d = json.load(open(PROJ))
    czi = d["image"]
    polys_full = [np.asarray(p, dtype=float) for p in d.get("sections", [])]
    fids_full = [tuple(map(float, f)) for f in d.get("fiducials", [])]
    section_ids = [f"section_{i}" for i in range(1, len(polys_full) + 1)]
    print(f"[export] {len(polys_full)} sections, {len(fids_full)} fiducials "
          f"from {os.path.basename(czi)}  (mode={mode})")

    if mode in ("fast", "all"):
        # small read just to obtain geometry for the GeoJSON stage-µm coords
        _, geom, _ = czi_io.read_czi_overview(czi, target_long_side=2048)
        out = exp._write_outputs(
            czi, polys_full, fids_full, section_ids=section_ids, geom=geom,
            write_czi=False, write_png=True, png_long_side=png_long_side)
        print("[export] fast outputs:")
        for k, v in out.items():
            print(f"    {k}: {v}")

    if mode in ("czi", "all"):
        dst = os.path.join(os.path.dirname(czi),
                           "tard_carbon_coat_001_annotated_STiM.czi")
        print(f"[export] writing annotated CZI copy -> {dst} (12 GB copy, please wait)…")
        rep = czi_export.write_annotated_czi(
            czi, dst, [p.tolist() for p in polys_full], fids_full,
            section_ids=section_ids)
        print(f"[export] CZI: {rep}")


if __name__ == "__main__":
    main()
