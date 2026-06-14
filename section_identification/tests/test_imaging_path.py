"""Headless tests for the open-path TSP imaging-order solver.

Run:  python -m section_identification.tests.test_imaging_path
"""

from __future__ import annotations

import numpy as np

from section_identification import imaging_path as ip


def test_collinear_recovers_monotonic():
    # points on a line, scrambled; optimal open path = the span (sorted order)
    xs = np.array([3.0, 0.0, 4.0, 1.0, 2.0])
    coords = np.column_stack([xs, np.zeros_like(xs)])
    order, total = ip.order_by_travel(coords)
    visited_x = xs[order]
    # strictly monotonic (either direction) and total == span (4.0)
    assert np.all(np.diff(visited_x) > 0) or np.all(np.diff(visited_x) < 0)
    assert abs(total - 4.0) < 1e-9


def test_square_open_path_is_three_sides():
    # 4 corners of a unit square; min open Hamiltonian path = 3 (three edges)
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    order, total = ip.order_by_travel(coords)
    assert sorted(order) == [0, 1, 2, 3]
    assert abs(total - 3.0) < 1e-9


def test_two_opt_never_worsens():
    rng = np.random.RandomState(0)
    coords = rng.rand(40, 2) * 100.0
    D = ip.distance_matrix(coords)
    nn = ip.nearest_neighbor_path(D, 0)
    opt = ip.two_opt(D, nn)
    assert ip.path_length(D, opt) <= ip.path_length(D, nn) + 1e-9
    # and beats the naive identity order
    assert ip.path_length(D, opt) <= ip.path_length(D, list(range(40))) + 1e-9


def test_trivial_sizes():
    assert ip.order_by_travel(np.zeros((0, 2))) == ([], 0.0)
    assert ip.order_by_travel(np.array([[1.0, 2.0]])) == ([0], 0.0)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} imaging_path tests passed.")


if __name__ == "__main__":
    _run_all()
