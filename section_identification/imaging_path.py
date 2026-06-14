"""Imaging-order assignment via open-path TSP (pure, headless-testable).

Stage 4 of the workflow: given section centroids (in **stage microns**, the
frame the microscope moves in), compute an acquisition order that minimises total
stage travel. This is a free travel-minimisation (per the project decision) — an
*open* Hamiltonian path (imaging may start and end anywhere), not a closed tour.

Approach (zero new dependencies): nearest-neighbour construction + 2-opt local
search. n≈300 solves in well under a second. ``python-tsp``/OR-Tools are
deliberately avoided. Travel is Euclidean in the supplied coordinate frame.

The result feeds the export stage, where section/ROI Ids are renumbered to this
order so ZEN (which acquires by Id order) follows the optimised route.
"""

from __future__ import annotations

import numpy as np

try:                                    # scipy is present (skimage/sklearn dep)
    from scipy.spatial.distance import cdist as _cdist
except Exception:                        # pragma: no cover - fallback
    _cdist = None


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    if _cdist is not None:
        return _cdist(coords, coords)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def path_length(D: np.ndarray, order, closed: bool = False) -> float:
    order = list(order)
    if len(order) < 2:
        return 0.0
    total = sum(float(D[order[k], order[k + 1]]) for k in range(len(order) - 1))
    if closed:
        total += float(D[order[-1], order[0]])
    return total


def nearest_neighbor_path(D: np.ndarray, start: int = 0) -> list[int]:
    n = len(D)
    if n == 0:
        return []
    unvisited = set(range(n))
    order = [start]
    unvisited.discard(start)
    while unvisited:
        last = order[-1]
        nxt = min(unvisited, key=lambda j: D[last, j])
        order.append(nxt)
        unvisited.discard(nxt)
    return order


def two_opt(D: np.ndarray, order, max_passes: int = 50) -> list[int]:
    """2-opt improvement of an OPEN path (no closing edge). Reverses segments
    while any reversal shortens the path; converges quickly for n in the low
    hundreds."""
    order = list(order)
    n = len(order)
    if n < 4:
        return order
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(0, n - 1):
            a = order[i - 1] if i > 0 else None       # edge (i-1, i); None at the open start
            b = order[i]
            for j in range(i + 1, n):
                c = order[j]
                d = order[j + 1] if j + 1 < n else None  # edge (j, j+1); None at the open end
                # current cost of the two boundary edges
                before = 0.0
                after = 0.0
                if a is not None:
                    before += D[a, b]; after += D[a, c]
                if d is not None:
                    before += D[c, d]; after += D[b, d]
                if after + 1e-12 < before:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True
                    b = order[i]
    return order


def order_by_travel(coords, start: int | None = None,
                    max_passes: int = 50) -> tuple[list[int], float]:
    """Return ``(order, total_travel)`` minimising open-path stage travel over
    ``coords`` (Nx2). ``order[k]`` is the index of the k-th section to image.

    When ``start`` is None the nearest-neighbour seed is begun from the most
    extreme point (smallest x+y) for determinism, then 2-opt is free to reorder.
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    n = len(coords)
    if n <= 1:
        return list(range(n)), 0.0
    D = distance_matrix(coords)
    if start is None:
        start = int(np.argmin(coords[:, 0] + coords[:, 1]))
    order = nearest_neighbor_path(D, start)
    order = two_opt(D, order, max_passes=max_passes)
    return order, path_length(D, order, closed=False)
