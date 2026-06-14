"""Workflow-wide compute broker: turn the probed host profile into per-stage
knobs so QC / SIFT / TSP stay responsive on weak hardware.

`host_profile.detect_profile` already adapts SAM (model variant, points-per-batch,
resolution caps). This extends the same idea to the new stages — choosing QC
working resolution + worker count, SIFT feature budget + matcher + pre-gating +
pool size, and the TSP refinement budget — from device/RAM/cores. SIFT stays
full-resolution per the project decision; responsiveness comes from caps + pool
size + pre-gating, never from downsampling SIFT input.

Defensive about the HostProfile shape (getattr fallbacks) so it can't crash the
GUI if a field is missing on some host.
"""

from __future__ import annotations

from . import host_profile


def get_profile(prefer: str | None = None):
    try:
        return host_profile.detect_profile(prefer)
    except Exception:
        return None


def _cores(profile, default=4):
    return max(1, int(getattr(profile, "cores", default) or default))


def _avail_gb(profile, default=8.0):
    return float(getattr(profile, "avail_gb", None) or getattr(profile, "ram_gb", default) or default)


def workers(profile) -> int:
    """Parallel CPU workers for embarrassingly-parallel stages (leave 2 free)."""
    return max(1, _cores(profile) - 2)


def qc_plan(profile) -> dict:
    """QC working resolution + worker count. Lower resolution on tight RAM keeps
    300 sections fast; the detectors are scale-tolerant."""
    avail = _avail_gb(profile)
    long_side = 512 if avail < 6 else (640 if avail < 12 else 768)
    return {"working_long_side": long_side, "workers": workers(profile)}


def sift_plan(profile, n_sections: int) -> dict:
    """SIFT feature budget, matcher, pool size, and pre-gate.

    Full-resolution always; cap ``nfeatures`` and prefer FLANN + a coarse
    pre-gate as the section count (and so the O(n²) pair load) grows.
    """
    avail = _avail_gb(profile)
    nfeatures = 0 if avail >= 12 else (2000 if avail >= 6 else 1000)   # 0 = unlimited
    n_pairs = n_sections * (n_sections - 1) // 2
    return {
        "nfeatures": nfeatures,
        "matcher": "flann" if n_sections > 80 else "bf",
        "pregate": n_sections > 150,          # skip hopeless pairs with a cheap gate
        "pool": workers(profile),
        "n_pairs": n_pairs,
    }


def tsp_plan(profile, n_sections: int) -> dict:
    """2-opt refinement budget — bounded passes so even large wafers stay snappy."""
    return {"max_passes": 50 if n_sections <= 400 else 20}


def summary(profile) -> str:
    if profile is None:
        return "compute: host profile unavailable; using safe defaults."
    try:
        return (f"compute: {getattr(profile, 'device_label', '?')}, "
                f"{_avail_gb(profile):.0f} GB free, {_cores(profile)} cores → "
                f"{workers(profile)} workers")
    except Exception:
        return "compute: host profiled."
