"""Host-adaptive performance profile.

STiM is deployed on varied hardware — slow/old lab machines, some CPU-only;
Linux boxes have CUDA but no MPS, Apple Silicon has MPS. A run that is fine on a
24 GB M4 Pro will OOM/thrash a 8 GB CPU laptop. This module **probes the host**
(device, RAM/VRAM, cores) and derives a feasible profile: which SAM model to
use, a memory-safe ``points_per_batch``, and caps on the working/tile resolution.
The auto-tuner (:mod:`calibration`) treats this as a budget — *feasibility before
fidelity* — and the GUI shows it (and lets the user override).

No hard dependency on ``psutil``: it's used if importable, else we fall back to
``os.sysconf`` and finally a conservative constant.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict

GB = 1024 ** 3


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #
def _total_ram_bytes() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        return 8 * GB  # conservative default for an unknown host


def _avail_ram_bytes() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        return _total_ram_bytes() // 2


def _vram_bytes(device: str) -> int:
    if device == "cuda":
        try:
            import torch
            _free, total = torch.cuda.mem_get_info()
            return int(total)
        except Exception:
            return 0
    return 0


def _cpu_cores() -> int:
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


# --------------------------------------------------------------------------- #
# Profile
# --------------------------------------------------------------------------- #
@dataclass
class HostProfile:
    device: str                 # "cuda" | "mps" | "cpu"
    device_label: str           # human-readable
    ram_gb: float               # total system RAM
    avail_gb: float             # currently available RAM
    vram_gb: float              # GPU VRAM (CUDA only; 0 otherwise)
    cores: int
    model_variant: str          # "tiny" | "small" | "base_plus" | "large"
    points_per_batch: int       # memory-safe default batch
    tile_cap_px: int            # max tile long-side a single SAM pass should use
    overview_cap_px: int        # max overview long-side to read into RAM
    mem_budget_bytes: float     # budget for the per-batch upsample tensor
    threads: int                # torch CPU threads

    def as_dict(self):
        return asdict(self)

    def summary(self) -> str:
        return (f"{self.device_label} · {self.ram_gb:.0f} GB RAM"
                + (f" · {self.vram_gb:.0f} GB VRAM" if self.vram_gb else "")
                + f" → SAM hiera_{self.model_variant}, points_per_batch≤{self.points_per_batch}, "
                f"tiles≤{self.tile_cap_px}px, overview≤{self.overview_cap_px}px")


def _pick_model(device: str, ram_gb: float, vram_gb: float) -> str:
    """Heaviest SAM2 variant the host can comfortably run."""
    if device == "cuda":
        if vram_gb >= 10:
            return "large"
        if vram_gb >= 6:
            return "base_plus"
        return "small"
    if device == "mps":
        return "base_plus" if ram_gb >= 16 else "small"
    # CPU: base_plus is painfully slow on old machines → stay light.
    if ram_gb >= 24:
        return "small"
    return "tiny"


def detect_profile(prefer: str | None = None) -> HostProfile:
    """Probe the host (honouring an explicit device ``prefer``) → HostProfile."""
    from section_identification.device import get_device, describe
    device = get_device(prefer).type
    label = describe() if prefer in (None, "", "auto") else device.upper()
    ram = _total_ram_bytes()
    avail = _avail_ram_bytes()
    vram = _vram_bytes(device)
    cores = _cpu_cores()
    ram_gb, avail_gb, vram_gb = ram / GB, avail / GB, vram / GB

    model = _pick_model(device, ram_gb, vram_gb)

    # Memory budget for SAM's per-batch upsample tensor (the OOM risk): a slice
    # of the GPU VRAM on CUDA, else a slice of *available* RAM (leave headroom).
    if device == "cuda" and vram:
        budget = 0.5 * vram
    elif device == "mps":
        # Apple unified memory: this budget competes with the model weights, the
        # napari multiscale display, and the OS — all in the SAME pool — and the
        # MPS allocator caches aggressively. A 0.35×avail slice (the CPU default)
        # is too generous and is what put M411 under memory stress. Cap by both
        # available AND total RAM so a transiently-high "available" can't lead to
        # a batch that then thrashes once napari/OS reclaim memory.
        budget = max(0.5 * GB, min(0.20 * avail, 0.15 * ram))
    else:
        budget = max(0.6 * GB, 0.35 * avail)

    # Device-default batch, later clamped per-tile by safe_points_per_batch().
    ppb_default = {"cuda": 64, "mps": 24, "cpu": 12}.get(device, 12)

    # Resolution caps so one tile read + one SAM pass fit the machine.
    if ram_gb >= 24:
        overview_cap, tile_cap = 12000, 4096
    elif ram_gb >= 12:
        overview_cap, tile_cap = 9000, 3072
    elif ram_gb >= 8:
        overview_cap, tile_cap = 6000, 2048
    else:
        overview_cap, tile_cap = 4096, 1536
    if device == "cpu":
        tile_cap = min(tile_cap, 2048)  # CPU SAM is slow; keep tiles modest

    threads = max(1, min(cores, 8))

    return HostProfile(
        device=device, device_label=label, ram_gb=ram_gb, avail_gb=avail_gb,
        vram_gb=vram_gb, cores=cores, model_variant=model,
        points_per_batch=int(ppb_default), tile_cap_px=int(tile_cap),
        overview_cap_px=int(overview_cap), mem_budget_bytes=float(budget),
        threads=int(threads))


# --------------------------------------------------------------------------- #
# Memory-safe batch + cost estimate
# --------------------------------------------------------------------------- #
def safe_points_per_batch(mem_budget_bytes: float, tile_h: int, tile_w: int,
                          requested: int, masks_per_point: int = 3) -> int:
    """Largest ``points_per_batch`` whose upsample tensor fits the budget.

    SAM2's AMG upsamples each batch's masks back to the (working) tile size:
    ``points_per_batch × masks_per_point × H × W × 4 bytes`` (float32), plus a
    threshold/stability copy. ``masks_per_point`` is 3 when ``multimask_output``
    is on (SAM emits 3 candidate masks per prompt) and 1 when it is off — passing
    it makes the estimate track the REAL tensor, so the multimask=off memory win
    is actually granted as a higher feasible batch. ~1.2× slack covers the bool
    threshold + stability-score intermediates. Independent of results — purely a
    memory/throughput knob.
    """
    per_point = max(1.0, max(1, masks_per_point) * tile_h * tile_w * 4 * 1.2)
    cap = max(1, int(mem_budget_bytes / per_point))
    return max(1, min(int(requested), cap))


# Rough per-SAM-pass seconds at 1024 px, by model and device. Order-of-magnitude
# only — used to warn the user, not to be precise.
_SEC_PER_PASS = {
    "cuda": {"tiny": 0.3, "small": 0.5, "base_plus": 0.9, "large": 1.6},
    "mps":  {"tiny": 1.5, "small": 2.5, "base_plus": 4.0, "large": 7.0},
    "cpu":  {"tiny": 12.0, "small": 22.0, "base_plus": 45.0, "large": 90.0},
}


def estimate_run(profile: HostProfile, n_tiles: int, points_per_side: int,
                 model_variant: str | None = None) -> dict:
    """Rough peak-memory + runtime estimate for a planned run (for the GUI)."""
    model = model_variant or profile.model_variant
    base = _SEC_PER_PASS.get(profile.device, _SEC_PER_PASS["cpu"]).get(model, 20.0)
    # more grid points → more decode work (sub-linear); normalise to a 32 grid
    grid_factor = max(0.4, (max(points_per_side, 1) / 32.0) ** 1.3)
    seconds = base * grid_factor * max(1, n_tiles)
    peak_gb = profile.mem_budget_bytes / GB + 0.5  # batch tensor + model/overhead
    warn = ""
    if profile.device == "cpu" and seconds > 180:
        warn = "CPU host: this will be slow — fewer/larger sections or a GPU help."
    elif peak_gb > 0.8 * profile.avail_gb:
        warn = "Close to available memory — lower points_per_batch if it stalls."
    return {"seconds": seconds, "minutes": seconds / 60.0,
            "peak_gb": peak_gb, "model": model, "n_tiles": int(n_tiles),
            "warning": warn}
