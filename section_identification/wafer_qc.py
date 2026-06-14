"""Per-section wafer quality-control detectors (pure, headless-testable).

Stage "QC": classical computer-vision detectors for the four defect classes that
matter on a serial-section wafer. Each operates on a per-section **grayscale
crop** (float or uint8) plus a boolean **mask** (the section interior, so wafer
background doesn't contaminate the measurement), and returns a continuous
``severity`` plus the raw, unit-bearing ``features``. Severities are mapped to a
``[0, 1]`` score via a saturating reference (``score = clip(severity/ref, 0, 1)``)
and flagged at a per-detector threshold — so the GUI can re-threshold instantly
from the stored raw features without recomputing.

Detectors:
  * debris  — bright high-contrast specks (contamination): MAD intensity outliers
              + Laplacian-of-Gaussian blob count.
  * fold    — dark elongated ridges (folded/doubled material): Frangi vesselness
              + elongated-component length.
  * shred   — tearing/fragmentation: connected-component count, solidity, holes,
              area vs the wafer-median section area.
  * chatter — periodic knife-vibration ripples: off-DC peak in the windowed 2-D
              power spectrum.

Design mirrors calibration.py: defaults are population-derived references the
user tunes, not opaque cutoffs. Everything here is numpy/scipy/skimage only.
"""

from __future__ import annotations

import numpy as np

from skimage.feature import blob_log
from skimage.filters import frangi, window
from skimage.measure import label, regionprops

try:
    from scipy.signal import find_peaks
except Exception:                        # pragma: no cover
    find_peaks = None


def _remove_small(binary, min_area):
    """Drop connected components with area < ``min_area`` (version-stable
    replacement for skimage.morphology.remove_small_objects)."""
    binary = np.asarray(binary, bool)
    lab = label(binary)
    if lab.max() == 0:
        return binary
    counts = np.bincount(lab.ravel())
    keep = counts >= int(min_area)
    keep[0] = False
    return keep[lab]


def _axis_len(region, which: str) -> float:
    """``which`` in {'minor','major'}; tolerant of skimage's axis_* rename.

    Accesses the new ``axis_*_length`` name first and only falls back to the
    deprecated ``*_axis_length`` if the new one is genuinely absent, so we never
    trip skimage's deprecation warning on a current install.
    """
    for name in (f"axis_{which}_length", f"{which}_axis_length"):
        try:
            v = getattr(region, name)
        except Exception:
            continue
        if v is not None:
            return float(v)
    return 0.0


# --------------------------------------------------------------------------- #
# defaults (population-tunable references + flag thresholds)
# --------------------------------------------------------------------------- #
def qc_defaults() -> dict:
    return {
        "debris_ref": 0.01,        # outlier-area fraction reference (1% of section)
        "fold_ref": 0.5,           # longest ridge / section minor axis
        "shred_ref": 1.0,          # combined shred severity reference
        "chatter_ref": 4.0,        # spectral peak / band-median ratio
        "flag": 0.5,               # score >= flag -> boolean True (per detector)
        "median_area": None,       # wafer-median section area (px²); set by calibrate
        "mad_k": 6.0,              # bright-outlier threshold (median + k*MAD)
        "min_blob_area": 4,        # px, drop specks smaller than this
    }


def _as_gray_float(gray) -> np.ndarray:
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim == 3:
        g = g.mean(axis=2)
    return g


def _norm(severity: float, ref: float) -> float:
    if ref <= 0:
        return 0.0
    return float(min(max(severity / ref, 0.0), 1.0))


# --------------------------------------------------------------------------- #
# individual detectors -> (severity, features dict)
# --------------------------------------------------------------------------- #
def detect_debris(gray, mask, mad_k=6.0, min_blob_area=4):
    g = _as_gray_float(gray)
    m = np.asarray(mask, bool)
    inside = g[m]
    if inside.size < 16:
        return 0.0, {"debris_area_frac": 0.0, "debris_n_blobs": 0}
    med = np.median(inside)
    mad = np.median(np.abs(inside - med)) or 1.0
    bright = (g > med + mad_k * mad) & m            # contamination = bright specks
    bright = _remove_small(bright, int(min_blob_area))
    area_frac = float(bright.sum()) / float(m.sum() or 1)
    n_blobs = 0
    try:
        norm = (g - g.min()) / (np.ptp(g) or 1.0)
        blobs = blob_log(norm * m, max_sigma=8, num_sigma=5, threshold=0.12)
        n_blobs = int(len(blobs))
    except Exception:
        pass
    return area_frac, {"debris_area_frac": area_frac, "debris_n_blobs": n_blobs}


def detect_folds(gray, mask):
    g = _as_gray_float(gray)
    m = np.asarray(mask, bool)
    if m.sum() < 64:
        return 0.0, {"fold_max_ridge_len_px": 0.0, "fold_total_ridge_len_px": 0.0}
    g = (g - g.min()) / (np.ptp(g) or 1.0)
    vez = frangi(g, black_ridges=True)              # dark ridges = folds
    vez = vez * m
    if vez.max() <= 0:
        return 0.0, {"fold_max_ridge_len_px": 0.0, "fold_total_ridge_len_px": 0.0}
    thr = vez[m].mean() + 2.0 * vez[m].std()
    ridges = vez > max(thr, 1e-6)
    ridges = _remove_small(ridges, 8)
    max_len = 0.0
    total_len = 0.0
    for r in regionprops(label(ridges)):
        if r.eccentricity > 0.9:                    # elongated only
            ln = _axis_len(r, "major")
            max_len = max(max_len, ln)
            total_len += ln
    # section minor axis (scale reference)
    minor = 1.0
    sec_props = regionprops(label(m.astype(np.uint8)))
    if sec_props:
        minor = max(1.0, _axis_len(sec_props[0], "minor"))
    severity = max_len / max(minor, 1.0)
    return severity, {"fold_max_ridge_len_px": float(max_len),
                      "fold_total_ridge_len_px": float(total_len)}


def detect_shredding(gray, mask, median_area=None):
    m = np.asarray(mask, bool)
    if m.sum() < 16:
        return 0.0, {"solidity": 1.0, "n_components": 0, "hole_frac": 0.0, "area_ratio": 1.0}
    lab = label(m.astype(np.uint8))
    props = sorted(regionprops(lab), key=lambda r: r.area, reverse=True)
    n_components = int(sum(1 for r in props if r.area >= 0.02 * m.sum()))
    main = props[0]
    solidity = float(main.solidity) if main.solidity else 1.0
    filled = float(main.area_filled if hasattr(main, "area_filled") else main.filled_area)
    hole_frac = float(max(filled - main.area, 0.0)) / max(filled, 1.0)
    total_area = float(m.sum())
    area_ratio = 1.0 if not median_area else total_area / float(median_area)
    sev = max((1.0 - solidity) / 0.15,              # solidity_ref ~0.85
              (n_components - 1) / 1.0,
              hole_frac / 0.02,
              max(0.0, 1.0 - area_ratio) / 0.4)
    return float(sev), {"solidity": solidity, "n_components": n_components,
                        "hole_frac": float(hole_frac), "area_ratio": float(area_ratio)}


def detect_chatter(gray, mask):
    g = _as_gray_float(gray)
    m = np.asarray(mask, bool)
    if m.sum() < 256:
        return 0.0, {"chatter_peak_ratio": 0.0, "chatter_freq_cyc_per_px": 0.0}
    # work on the bounding box of the mask, mean-filled outside, Hann-windowed
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = g[y0:y1, x0:x1].copy()
    cm = m[y0:y1, x0:x1]
    crop[~cm] = crop[cm].mean() if cm.any() else 0.0
    crop = crop - crop.mean()
    h, w = crop.shape
    if min(h, w) < 16:
        return 0.0, {"chatter_peak_ratio": 0.0, "chatter_freq_cyc_per_px": 0.0}
    win = window("hann", crop.shape)
    P = np.abs(np.fft.fftshift(np.fft.fft2(crop * win))) ** 2
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rad = np.hypot(yy - cy, xx - cx).astype(int)
    rmax = min(cy, cx)
    # skip the low-frequency core AND the lowest radii (few samples there -> the
    # bin mean is noisy and would fake a peak on a clean section).
    r_min = max(6, rmax // 8)
    if rmax - r_min < 4:
        return 0.0, {"chatter_peak_ratio": 0.0, "chatter_freq_cyc_per_px": 0.0}
    prof = np.array([P[rad == r].mean() if np.any(rad == r) else 0.0
                     for r in range(r_min, rmax)])
    med = float(np.median(prof)) or 1e-9
    # severity = how far the dominant peak rises ABOVE the local baseline,
    # measured as a prominence (sharp ripple peak) relative to the band median.
    if find_peaks is not None:
        peaks, props = find_peaks(prof, prominence=med)
        if len(peaks) == 0:
            return 0.0, {"chatter_peak_ratio": 0.0, "chatter_freq_cyc_per_px": 0.0}
        k = int(np.argmax(props["prominences"]))
        peak_ratio = float(props["prominences"][k] / med)
        peak_idx = int(peaks[k]) + r_min
    else:                                            # pragma: no cover
        peak_ratio = float(prof.max() / med)
        peak_idx = int(np.argmax(prof)) + r_min
    freq = peak_idx / float(max(h, w))               # cycles per px (approx)
    return peak_ratio, {"chatter_peak_ratio": peak_ratio,
                        "chatter_freq_cyc_per_px": float(freq)}


# --------------------------------------------------------------------------- #
# combined per-section scoring
# --------------------------------------------------------------------------- #
def feature_maps(gray, mask) -> dict:
    """Return the intermediate maps each detector produces, for on-wafer
    visualisation (the visually-guided principle): the Frangi ridge map (folds),
    the bright-outlier mask + LoG blobs (debris), the connected-component label
    map (shred), and the windowed log power spectrum (chatter). All crop-sized
    except ``spectrum`` (bbox-sized) and ``blobs`` (``Nx3`` y,x,r). Pure arrays —
    the GUI overlays them; this is unit-testable on a synthetic crop.
    """
    g = _as_gray_float(gray)
    m = np.asarray(mask, bool)
    gn = (g - g.min()) / (np.ptp(g) or 1.0)

    ridges = frangi(gn, black_ridges=True) * m

    if m.sum() >= 16:
        inside = g[m]
        med = np.median(inside)
        mad = np.median(np.abs(inside - med)) or 1.0
        bright = (g > med + qc_defaults()["mad_k"] * mad) & m
    else:
        bright = np.zeros_like(m)
    try:
        blobs = blob_log(gn * m, max_sigma=8, num_sigma=5, threshold=0.12)
    except Exception:
        blobs = np.empty((0, 3))

    labels = label((m).astype(np.uint8))

    spectrum = np.zeros((1, 1))
    if m.sum() >= 256:
        ys, xs = np.where(m)
        crop = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(float)
        if min(crop.shape) >= 16:
            crop = crop - crop.mean()
            P = np.abs(np.fft.fftshift(np.fft.fft2(crop * window("hann", crop.shape)))) ** 2
            spectrum = np.log1p(P)

    return {"ridges": ridges, "bright": bright, "blobs": blobs,
            "labels": labels, "spectrum": spectrum}


def dominant_flag(qc_result: dict) -> str | None:
    """The flag with the highest score (for choosing which diagnostic to show)."""
    if not qc_result:
        return None
    scores = {k: v for k, v in qc_result.get("scores", {}).items() if k != "overall"}
    flags = qc_result.get("flags", {})
    flagged = {k: scores.get(k, 0.0) for k, on in flags.items() if on and k != "any"}
    pool = flagged or scores
    return max(pool, key=pool.get) if pool else None


def score_section(gray, mask, refs: dict | None = None) -> dict:
    """Run all detectors and return a QC result dict (scores/flags/features)
    matching :class:`wafer_model.QCResult`'s fields."""
    r = dict(qc_defaults())
    if refs:
        r.update(refs)

    deb_sev, deb_f = detect_debris(gray, mask, r["mad_k"], r["min_blob_area"])
    fold_sev, fold_f = detect_folds(gray, mask)
    shred_sev, shred_f = detect_shredding(gray, mask, r["median_area"])
    chat_sev, chat_f = detect_chatter(gray, mask)

    scores = {
        "debris": _norm(deb_sev, r["debris_ref"]),
        "fold": _norm(fold_sev, r["fold_ref"]),
        "shred": _norm(shred_sev, r["shred_ref"]),
        "chatter": _norm(chat_sev, r["chatter_ref"]),
    }
    scores["overall"] = float(max(scores.values()))
    flags = {k: bool(v >= r["flag"]) for k, v in scores.items() if k != "overall"}
    flags["any"] = bool(any(flags.values()))
    features = {**deb_f, **fold_f, **shred_f, **chat_f}
    return {"scores": scores, "flags": flags, "features": features,
            "params_used": {k: r[k] for k in
                            ("debris_ref", "fold_ref", "shred_ref", "chatter_ref",
                             "flag", "mad_k", "median_area")}}
