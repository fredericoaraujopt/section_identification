"""Device selection and autocast helpers.

Centralises the CUDA / Apple-MPS / CPU decision so the rest of STiM never
hard-codes a device again. Importing this module sets
``PYTORCH_ENABLE_MPS_FALLBACK=1`` *before* torch is first used, which lets the
handful of ops SAM 2 does not yet implement on Metal fall back to the CPU
instead of raising. SAM 2's MPS support is officially "preliminary", so we keep
everything in fp32 on MPS (bf16 autocast is CUDA-only — it is documented to
misbehave on Metal).
"""

import os

# Must be set before the first torch MPS op. setdefault so an explicit user
# choice in the environment still wins.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import contextlib

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch device.

    Order: explicit ``prefer`` > CUDA > Apple MPS > CPU.
    """
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def device_str(prefer: str | None = None) -> str:
    """``get_device`` as a plain string (``"cuda"``/``"mps"``/``"cpu"``)."""
    return get_device(prefer).type


@contextlib.contextmanager
def autocast_ctx(device):
    """Mixed-precision context that is a no-op anywhere except CUDA.

    On CUDA we use bf16 autocast + TF32 matmuls (the SAM 2 fast path). On MPS
    and CPU we run in fp32 — autocast on MPS is unreliable for SAM 2.
    """
    dev = device.type if isinstance(device, torch.device) else str(device)
    if dev == "cuda":
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def describe() -> str:
    """Human-readable one-liner for logs/GUI."""
    d = get_device()
    if d.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    if d.type == "mps":
        return "Apple MPS (Metal GPU)"
    return "CPU"
