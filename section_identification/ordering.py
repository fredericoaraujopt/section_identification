"""Recover serial-section order by image similarity (cross-correlation).

Sections rarely sit on the wafer/tape in acquisition order. For serial-section
connectomics the *sequence* matters, so we estimate it from appearance:
consecutive physical sections look almost identical, so ordering sections to
maximise adjacent similarity recovers the series.

Pipeline: crop a normalised thumbnail per section -> pairwise normalised
cross-correlation -> seriation (spectral Fiedler ordering, with a greedy
nearest-neighbour fallback). Returns a permutation the GUI/export can apply to
section IDs.
"""

from __future__ import annotations

import numpy as np


def masks_to_bboxes(masks) -> list[tuple[int, int, int, int]]:
    """SAM masks (``bbox=[x,y,w,h]``) -> ``(x0,y0,x1,y1)`` boxes."""
    boxes = []
    for m in masks:
        x, y, w, h = m["bbox"]
        boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


def polygons_to_bboxes(polygons) -> list[tuple[int, int, int, int]]:
    boxes = []
    for poly in polygons:
        p = np.asarray(poly, dtype=float).reshape(-1, 2)
        boxes.append((int(p[:, 0].min()), int(p[:, 1].min()),
                      int(p[:, 0].max()), int(p[:, 1].max())))
    return boxes


def extract_thumbnails(image: np.ndarray, bboxes, size: int = 64) -> np.ndarray:
    """Crop each bbox and resize to ``size x size``, zero-mean/unit-std."""
    import cv2

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thumbs = []
    for (x0, y0, x1, y1) in bboxes:
        crop = gray[max(0, y0):y1 + 1, max(0, x0):x1 + 1]
        if crop.size == 0:
            crop = np.zeros((size, size), dtype=gray.dtype)
        t = cv2.resize(crop.astype(np.float32), (size, size))
        t -= t.mean()
        # Normalise to unit L2 so the inner product is a bounded NCC in [-1, 1].
        # Constant crops (norm 0) stay all-zero rather than blowing up.
        n = float(np.linalg.norm(t))
        if n > 1e-6:
            t /= n
        thumbs.append(np.nan_to_num(t))
    return np.asarray(thumbs)


def similarity_matrix(thumbs: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation between every pair of thumbnails.

    Thumbnails are already zero-mean/unit-std, so the inner product over pixels
    is the normalised cross-correlation coefficient in ``[-1, 1]``.
    """
    n = len(thumbs)
    if n == 0:
        return np.zeros((0, 0))
    flat = np.nan_to_num(thumbs.reshape(n, -1).astype(np.float64))
    sim = flat @ flat.T  # thumbnails are unit-L2 -> entries are NCC in [-1, 1]
    sim = np.clip(np.nan_to_num(sim), -1.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    return sim


def spectral_order(sim: np.ndarray) -> np.ndarray:
    """Seriation via the Fiedler vector of the similarity-graph Laplacian."""
    n = len(sim)
    if n <= 2:
        return np.arange(n)
    W = np.clip(sim, 0.0, None)
    d = W.sum(axis=1)
    L = np.diag(d) - W
    vals, vecs = np.linalg.eigh(L)
    fiedler = vecs[:, 1]  # second-smallest eigenvector
    return np.argsort(fiedler)


def greedy_order(sim: np.ndarray) -> np.ndarray:
    """Nearest-neighbour chain from the most isolated section (an endpoint)."""
    n = len(sim)
    if n <= 1:
        return np.arange(n)
    start = int(np.argmin(sim.sum(axis=1)))
    visited = [start]
    remaining = set(range(n)) - {start}
    while remaining:
        last = visited[-1]
        nxt = max(remaining, key=lambda j: sim[last, j])
        visited.append(nxt)
        remaining.discard(nxt)
    return np.asarray(visited)


def order_sections(image: np.ndarray, bboxes, method: str = "spectral",
                   size: int = 64):
    """Return ``(order, similarity)``: a permutation of section indices.

    ``order[k]`` is the original index of the k-th section in the recovered
    series.
    """
    thumbs = extract_thumbnails(image, bboxes, size)
    sim = similarity_matrix(thumbs)
    if method == "greedy":
        order = greedy_order(sim)
    else:
        order = spectral_order(sim)
    return order, sim
