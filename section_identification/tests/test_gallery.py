"""Headless tests for the aligned section gallery montage.

Run:  python -m section_identification.tests.test_gallery
"""

from __future__ import annotations

import numpy as np

from section_identification import gallery

try:
    import cv2
except Exception:
    cv2 = None


def _asym_patch(size=120):
    img = np.zeros((size, size), np.uint8)
    img[20:size - 20, size // 2 - 6:size // 2 - 2] = 220     # off-centre bright bar
    img[30:50, 30:50] = 150                                   # corner mark (breaks symmetry)
    return img


def test_upright_thumb_rectifies_rotation():
    if cv2 is None:
        print("    (skip: cv2 missing)")
        return
    base = _asym_patch()
    M = cv2.getRotationMatrix2D((60, 60), 40.0, 1.0)
    rotated = cv2.warpAffine(base, M, (120, 120))
    t0 = gallery.upright_thumb(base, 0.0, size=64).astype(float)
    tr = gallery.upright_thumb(rotated, 40.0, size=64).astype(float)   # un-rotate by 40
    corr = np.corrcoef(t0.ravel(), tr.ravel())[0, 1]
    assert corr > 0.7, corr


def test_build_montage_shape():
    thumbs = [np.full((32, 32), i * 10, np.uint8) for i in range(5)]
    mont = gallery.build_montage(thumbs, cols=3, pad=2)
    cell = 32 + 2
    assert mont.shape == (2 * cell + 2, 3 * cell + 2)
    assert gallery.build_montage([]).shape == (1, 1)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} gallery tests passed.")


if __name__ == "__main__":
    _run_all()
