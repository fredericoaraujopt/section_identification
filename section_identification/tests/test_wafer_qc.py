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


def test_feature_maps_shapes_and_content():
    gray, mask = _clean()
    c = SIZE // 2
    gray[c - 1:c + 2, :][mask[c - 1:c + 2, :]] = 30.0      # inject a fold
    fm = qc.feature_maps(gray, mask)
    assert fm["ridges"].shape == gray.shape
    assert fm["ridges"].max() > 0                           # fold lit up the ridge map
    assert fm["bright"].shape == mask.shape
    assert fm["labels"].shape == mask.shape
    assert fm["spectrum"].ndim == 2
    assert fm["blobs"].ndim == 2 and fm["blobs"].shape[1] == 3


def test_rethreshold_uses_cached_severity():
    gray, mask = _clean()
    c = SIZE // 2
    gray[c - 1:c + 2, :][mask[c - 1:c + 2, :]] = 30.0   # fold
    res = qc.score_section(gray, mask)
    assert "fold_severity" in res["features"]
    sev = res["features"]["fold_severity"]
    # tighten the fold reference -> higher score, no recompute
    tight = qc.rethreshold(res, {"fold_ref": sev / 4.0})
    assert tight["scores"]["fold"] >= res["scores"]["fold"]
    assert tight["flags"]["fold"] is True
    # loosen it -> lower score / unflag
    loose = qc.rethreshold(res, {"fold_ref": sev * 100.0})
    assert loose["scores"]["fold"] < tight["scores"]["fold"]


def test_calibrate_qc_from_population():
    # population with one clear fold outlier
    pop = []
    for s in (0.1, 0.1, 0.1, 0.1, 2.0):
        pop.append({"features": {"debris_severity": 0.0, "fold_severity": s,
                                 "shred_severity": 0.0, "chatter_severity": 0.0}})
    refs = qc.calibrate_qc(pop, percentile=80.0)
    assert "fold_ref" in refs
    # 80th percentile of folds sits between the cluster (0.1) and the outlier (2.0)
    assert 0.1 < refs["fold_ref"] < 2.0


def test_dominant_flag():
    res = {"scores": {"debris": 0.1, "fold": 0.8, "shred": 0.0, "chatter": 0.2, "overall": 0.8},
           "flags": {"debris": False, "fold": True, "shred": False, "chatter": False, "any": True}}
    assert qc.dominant_flag(res) == "fold"
    assert qc.dominant_flag({}) is None


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} wafer_qc tests passed.")


if __name__ == "__main__":
    _run_all()
