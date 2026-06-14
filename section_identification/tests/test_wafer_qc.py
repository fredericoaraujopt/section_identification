"""Headless tests for QC detectors on synthetic defects (no real images needed).

Each test builds a clean section crop + a single injected defect and asserts the
targeted detector fires (flag True, score above the clean baseline).

Run:  python -m section_identification.tests.test_wafer_qc
"""

from __future__ import annotations

import numpy as np

from section_identification import wafer_qc as qc


SIZE = 160
R = 60
_rng = np.random.RandomState(1)


def _disk(size=SIZE, r=R):
    yy, xx = np.ogrid[:size, :size]
    c = size // 2
    mask = (yy - c) ** 2 + (xx - c) ** 2 <= r ** 2
    return mask


def _clean(base=120.0):
    mask = _disk()
    gray = np.zeros((SIZE, SIZE), float)
    gray[mask] = base
    gray = gray + _rng.normal(0, 2.0, gray.shape)
    return gray, mask


def test_clean_section_is_quiet():
    gray, mask = _clean()
    res = qc.score_section(gray, mask)
    assert res["flags"]["any"] is False, res["scores"]


def test_fold_fires():
    gray, mask = _clean()
    c = SIZE // 2
    gray[c - 1:c + 2, :][mask[c - 1:c + 2, :]] = 30.0   # dark elongated ridge
    res = qc.score_section(gray, mask)
    base = qc.score_section(*_clean())["scores"]["fold"]
    assert res["scores"]["fold"] > base
    assert res["flags"]["fold"] is True, res["scores"]


def test_chatter_fires():
    gray, mask = _clean()
    x = np.arange(SIZE)
    stripes = 40.0 * np.sin(2 * np.pi * x / 6.0)        # periodic ripples, period 6px
    gray = gray + stripes[None, :]
    gray[~mask] = 0.0
    res = qc.score_section(gray, mask)
    base = qc.score_section(*_clean())["scores"]["chatter"]
    assert res["scores"]["chatter"] > base
    assert res["flags"]["chatter"] is True, res["scores"]


def test_debris_fires():
    gray, mask = _clean()
    yy, xx = np.ogrid[:SIZE, :SIZE]
    for (cy, cx) in [(70, 70), (90, 95), (60, 100)]:
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= 16     # bright specks
        gray[blob & mask] = 255.0
    res = qc.score_section(gray, mask)
    base = qc.score_section(*_clean())["scores"]["debris"]
    assert res["scores"]["debris"] > base
    assert res["flags"]["debris"] is True, res["scores"]


def test_shred_fires():
    gray, mask = _clean()
    # remove a wedge (concavity -> low solidity) and add a detached fragment
    yy, xx = np.ogrid[:SIZE, :SIZE]
    c = SIZE // 2
    wedge = (xx > c) & (yy > c)
    mask = mask & ~wedge
    frag = (yy - 20) ** 2 + (xx - 20) ** 2 <= 36
    mask = mask | frag
    gray = np.zeros((SIZE, SIZE), float)
    gray[mask] = 120.0
    res = qc.score_section(gray, mask)
    assert res["features"]["n_components"] >= 2 or res["features"]["solidity"] < 0.85
    assert res["flags"]["shred"] is True, (res["scores"], res["features"])


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} wafer_qc tests passed.")


if __name__ == "__main__":
    _run_all()
