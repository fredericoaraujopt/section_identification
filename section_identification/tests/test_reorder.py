"""Headless tests for SIFT reordering core.

Two parts: (1) seriation math on a synthetic banded similarity matrix recovers
the true order; (2) the SIFT pipeline gives high inliers for a rotated copy of a
patch and low inliers for an unrelated patch.

Run:  python -m section_identification.tests.test_reorder
"""

from __future__ import annotations

import numpy as np

from section_identification import reorder

try:
    import cv2
except Exception:
    cv2 = None


def test_seriation_recovers_banded_order():
    # banded similarity for a known serial order, then shuffle the indices
    n = 12
    base = np.array([[np.exp(-abs(i - j) / 2.0) for j in range(n)] for i in range(n)])
    np.fill_diagonal(base, 0.0)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    shuffled = base[np.ix_(perm, perm)]

    order = reorder.recover_order(shuffled, method="spectral+2opt")
    # recovered order maps back to a monotonic sequence in the original index
    recovered_original = [perm[i] for i in order]
    fwd = recovered_original == sorted(recovered_original)
    rev = recovered_original == sorted(recovered_original, reverse=True)
    assert fwd or rev, recovered_original


def test_order_confidence_shape():
    n = 6
    sim = np.ones((n, n)) * 5.0
    np.fill_diagonal(sim, 0.0)
    order = list(range(n))
    conf = reorder.order_confidence(sim, order)
    assert len(conf) == n and all(0.0 <= c <= 1.0 for c in conf)


def _texture(seed, size=200):
    """A richly textured patch (many SIFT-able corners)."""
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size), np.uint8)
    for _ in range(60):
        cx, cy = rng.randint(20, size - 20, 2)
        r = rng.randint(4, 14)
        val = int(rng.randint(80, 255))
        yy, xx = np.ogrid[:size, :size]
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = val
    return img


def test_sift_matches_rotated_copy_not_unrelated():
    if cv2 is None:
        print("  skip test_sift (cv2 missing)")
        return
    a = _texture(1)
    M = cv2.getRotationMatrix2D((100, 100), 35.0, 1.0)
    a_rot = cv2.warpAffine(a, M, (200, 200))
    b = _texture(99)  # unrelated

    fa = reorder.sift_features(a)
    fa_rot = reorder.sift_features(a_rot)
    fb = reorder.sift_features(b)

    same = reorder.pairwise_inliers(*fa, *fa_rot)
    diff = reorder.pairwise_inliers(*fa, *fb)
    assert same > diff, (same, diff)
    assert same >= 8, same


def test_matched_points_pairs():
    if cv2 is None:
        print("    (skip: cv2 missing)")
        return
    a = _texture(1)
    M = cv2.getRotationMatrix2D((100, 100), 25.0, 1.0)
    a_rot = cv2.warpAffine(a, M, (200, 200))
    fa = reorder.sift_features(a)
    fb = reorder.sift_features(a_rot)
    ptsA, ptsB = reorder.matched_points(*fa, *fb)
    assert len(ptsA) == len(ptsB) and len(ptsA) >= 6
    assert ptsA.shape[1] == 2 and ptsB.shape[1] == 2
    # an unrelated pair yields few/no inliers
    pc = reorder.sift_features(_texture(77))
    pA, pB = reorder.matched_points(*fa, *pc)
    assert len(pA) < len(ptsA)


def test_heatmap_image():
    sim = np.array([[0, 5, 1], [5, 0, 2], [1, 2, 0]], float)
    img = reorder.heatmap_image(sim)
    assert img.shape == (3, 3) and img.dtype == np.uint8
    assert img.max() == 255 and img.min() == 0
    # reordering permutes rows+cols
    permuted = reorder.heatmap_image(sim, order_indices=[2, 0, 1])
    expected = (sim[np.ix_([2, 0, 1], [2, 0, 1])])
    assert np.argmax(permuted) == np.argmax((expected - expected.min()))


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} reorder tests passed.")


if __name__ == "__main__":
    _run_all()
