"""Serial-section reordering via SIFT feature matching (headless-testable core).

Stage "Reorder": sections sit on the wafer in arbitrary order and rotation. We
recover the serial order from appearance — consecutive physical sections share
fine structure (vasculature, cell layout). SIFT is rotation/scale invariant, so
arbitrary on-wafer rotation needs no pre-alignment; the number of RANSAC inliers
between two sections is a robust similarity.

Pipeline (compute lives in reorder_worker.py; this is the reusable core):
  1. ``sift_features`` per section on a FULL-RESOLUTION masked crop (blood
     vessels must be resolved — SIFT is the one stage that does NOT downsample).
  2. ``pairwise_inliers`` — Lowe-ratio kNN matches + ``estimateAffinePartial2D``
     RANSAC inlier count -> a symmetric similarity matrix.
  3. ``recover_order`` — spectral seriation (reused from ordering.py) refined by
     open-path 2-opt (reused from imaging_path.py), oriented so the most
     isolated end is first.

The descriptor extraction (1) is the expensive step and is cached to .npz by the
worker; the similarity matrix (2) is cached too so re-running ordering is instant.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:                        # pragma: no cover
    cv2 = None

from . import imaging_path
from .ordering import greedy_order, spectral_order


# --------------------------------------------------------------------------- #
# SIFT features + pairwise similarity
# --------------------------------------------------------------------------- #
def sift_features(gray, mask=None, nfeatures: int = 0):
    """Return ``(keypoints_xy: Nx2 float32, descriptors: NxD float32)`` for a
    grayscale (full-res) crop. ``mask`` confines keypoints to the section."""
    if cv2 is None:
        raise RuntimeError("opencv (cv2) is required for SIFT features")
    g = np.asarray(gray)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8) if g.max() > 1.0 else (g * 255).astype(np.uint8)
    m = None
    if mask is not None:
        m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    sift = cv2.SIFT_create(nfeatures=int(nfeatures))
    kp, desc = sift.detectAndCompute(g, m)
    if not kp or desc is None:
        return np.empty((0, 2), np.float32), None
    xy = np.array([k.pt for k in kp], dtype=np.float32)
    return xy, desc.astype(np.float32)


def _matcher():
    # BFMatcher(L2) is deterministic and robust for SIFT; the worker may switch
    # to FLANN for speed at scale (compute-adaptive).
    return cv2.BFMatcher(cv2.NORM_L2)


def pairwise_inliers(kp_a, desc_a, kp_b, desc_b, ratio: float = 0.75,
                     min_matches: int = 4) -> int:
    """RANSAC-verified inlier count between two sections' SIFT features."""
    if cv2 is None or desc_a is None or desc_b is None:
        return 0
    if len(desc_a) < 2 or len(desc_b) < 2:
        return 0
    matches = _matcher().knnMatch(desc_a, desc_b, k=2)
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) < min_matches:
        return 0
    pts_a = np.float32([kp_a[m.queryIdx] for m in good])
    pts_b = np.float32([kp_b[m.trainIdx] for m in good])
    _, inliers = cv2.estimateAffinePartial2D(pts_a, pts_b, method=cv2.RANSAC,
                                             ransacReprojThreshold=5.0)
    return int(inliers.sum()) if inliers is not None else 0


def matched_points(kp_a, desc_a, kp_b, desc_b, ratio: float = 0.75,
                   min_matches: int = 4):
    """Return ``(ptsA, ptsB)`` — the RANSAC-verified inlier keypoint pairs (crop
    coords) between two sections, for drawing correspondences. Empty arrays if
    too few matches."""
    empty = (np.empty((0, 2), np.float32), np.empty((0, 2), np.float32))
    if cv2 is None or desc_a is None or desc_b is None:
        return empty
    if len(desc_a) < 2 or len(desc_b) < 2:
        return empty
    good = []
    for pair in _matcher().knnMatch(desc_a, desc_b, k=2):
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) < min_matches:
        return empty
    pa = np.float32([kp_a[m.queryIdx] for m in good])
    pb = np.float32([kp_b[m.trainIdx] for m in good])
    _, inliers = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC,
                                             ransacReprojThreshold=5.0)
    if inliers is None:
        return empty
    mask = inliers.ravel().astype(bool)
    return pa[mask], pb[mask]


def similarity_matrix(features, ratio: float = 0.75, progress=None) -> np.ndarray:
    """Full symmetric inlier-count matrix for ``features`` (list of
    ``(kp_xy, desc)``). ``progress(done, total)`` is called per pair if given."""
    n = len(features)
    S = np.zeros((n, n), dtype=float)
    total = n * (n - 1) // 2
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            kp_a, d_a = features[i]
            kp_b, d_b = features[j]
            v = pairwise_inliers(kp_a, d_a, kp_b, d_b, ratio)
            S[i, j] = S[j, i] = v
            done += 1
            if progress is not None:
                progress(done, total)
    return S


# --------------------------------------------------------------------------- #
# order recovery (seriation + 2-opt)
# --------------------------------------------------------------------------- #
def recover_order(sim: np.ndarray, method: str = "spectral+2opt") -> list[int]:
    """Recover the serial order (list of section indices) from a similarity
    matrix. Spectral seriation initialises; open-path 2-opt refines it by
    maximising adjacent similarity (= minimising adjacent dissimilarity)."""
    sim = np.asarray(sim, dtype=float)
    n = len(sim)
    if n <= 2:
        return list(range(n))

    if method == "greedy":
        order = list(np.asarray(greedy_order(sim)))
    else:
        order = list(np.asarray(spectral_order(sim)))
        # refine: maximise summed adjacent similarity via 2-opt on dissimilarity
        dissim = sim.max() - sim
        np.fill_diagonal(dissim, 0.0)
        order = imaging_path.two_opt(dissim, order)

    # orient so the most isolated section (lowest total similarity) is first
    rowsum = sim.sum(axis=1)
    if rowsum[order[-1]] < rowsum[order[0]]:
        order = order[::-1]
    return [int(i) for i in order]


def order_confidence(sim: np.ndarray, order) -> list[float]:
    """Per-position confidence = adjacent-edge inliers normalised by the matrix
    max (flags weak joins for review)."""
    sim = np.asarray(sim, float)
    mx = sim.max() or 1.0
    conf = [1.0]
    for k in range(1, len(order)):
        conf.append(float(sim[order[k - 1], order[k]] / mx))
    return conf


def heatmap_image(sim, order_indices=None) -> np.ndarray:
    """Normalise a similarity matrix to a uint8 image for display, optionally
    permuted by ``order_indices`` (the recovered serial order) so the banded /
    diagonal-dominant structure of a correct ordering becomes visible."""
    S = np.asarray(sim, dtype=float)
    if order_indices is not None and len(order_indices) == len(S):
        idx = np.asarray(order_indices, dtype=int)
        S = S[np.ix_(idx, idx)]
    mn, mx = float(S.min()), float(S.max())
    return ((S - mn) / ((mx - mn) or 1.0) * 255.0).astype(np.uint8)


def reorder(features, ids=None, ratio: float = 0.75, method: str = "spectral+2opt",
            progress=None) -> dict:
    """End-to-end: features -> similarity -> order. Returns a dict with the
    recovered ``order`` (ids if given, else indices), per-position confidence,
    and the edge list for the UI match graph."""
    S = similarity_matrix(features, ratio, progress)
    idx_order = recover_order(S, method)
    ids = list(ids) if ids is not None else list(range(len(features)))
    conf = order_confidence(S, idx_order)
    order_ids = [ids[i] for i in idx_order]
    edges = []
    for k in range(1, len(idx_order)):
        a, b = idx_order[k - 1], idx_order[k]
        edges.append({"a": ids[a], "b": ids[b], "inliers": int(S[a, b]),
                      "confidence": float(conf[k])})
    return {"order": order_ids, "method": method,
            "confidence_per_position": conf, "edges": edges,
            "similarity": S}
