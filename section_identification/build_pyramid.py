"""Standalone CLI to build the CZI display Zarr pyramid in its own process.

The build is parallelised across a process pool (see
:func:`section_identification.czi_io.write_czi_zarr_pyramid`). Running it here --
rather than from inside the napari GUI process -- means the pool's spawned workers
re-import THIS lightweight module (``czi_io`` + ``zarr`` + ``numpy``) and never
napari/Qt, so worker startup is fast and there is no chance of the GUI relaunching
itself. The GUI invokes this via ``subprocess`` and parses the ``PROGRESS`` lines.

Usage:
    python -m section_identification.build_pyramid IMAGE ZPATH [--channel N] [--workers K]

stdout protocol (one token stream, line-buffered):
    PROGRESS <level_done> <level_total>   -- emitted per completed pyramid level
    DONE                                  -- emitted once the cache is in place
Any exception is written to stderr and the process exits non-zero.
"""
from __future__ import annotations

import argparse
import sys

from section_identification import czi_io


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image_path")
    ap.add_argument("zpath")
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None,
                    help="process-pool size (default: min(8, cpus-2))")
    a = ap.parse_args(argv)

    def progress(done, total):
        print(f"PROGRESS {done} {total}", flush=True)

    czi_io.write_czi_zarr_pyramid(a.image_path, a.zpath, channel=a.channel,
                                  max_workers=a.workers, progress=progress)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
