"""Run the whole headless test suite (no Qt/napari needed).

    python -m section_identification.tests.run_all

These cover the pure logic layer of the connectomics-workflow expansion:
data model, persistence, FOV-nav math, pose alignment, ROI propagation, QC
detectors, SIFT reordering, TSP, export adapters, and the worker protocol.
The napari GUI is intentionally NOT exercised here (it needs a display).
"""

from __future__ import annotations

import importlib
import warnings

MODULES = [
    "wafer_model", "project_io", "fov_nav", "align", "imaging_path",
    "roi", "wafer_qc", "reorder", "export", "worker_protocol",
]


def main():
    warnings.simplefilter("error", FutureWarning)   # fail on skimage/np deprecations
    total = 0
    failed = []
    for name in MODULES:
        mod = importlib.import_module(f"section_identification.tests.test_{name}")
        try:
            print(f"[{name}]")
            mod._run_all()
            # _run_all prints "N ... tests passed."; count via the test_ functions
            total += sum(1 for k in dir(mod) if k.startswith("test_"))
        except Exception as e:                       # pragma: no cover
            failed.append((name, e))
            print(f"  FAILED: {e}")
    print("=" * 50)
    if failed:
        print(f"{len(failed)} module(s) FAILED: {[f[0] for f in failed]}")
        raise SystemExit(1)
    print(f"ALL GREEN — {total} tests across {len(MODULES)} modules.")


if __name__ == "__main__":
    main()
