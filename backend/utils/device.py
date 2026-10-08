"""GPU / device detection and memory guards.

Everything here degrades gracefully on machines without CUDA (macOS dev laptops, CI,
CPU-only servers). The functions never raise on import and never require torch to be
installed — they just report what is actually available, so the UI can show honest
status instead of pretending a model is ready.
"""

from __future__ import annotations

import platform
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DeviceInfo:
    """Snapshot of the compute environment shown on the Model Status page."""

    torch_available: bool = False
    torch_version: Optional[str] = None
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    device_count: int = 0
    device_names: List[str] = field(default_factory=list)
    total_memory_gb: List[float] = field(default_factory=list)
    free_memory_gb: List[float] = field(default_factory=list)
    compute_capability: List[str] = field(default_factory=list)
    supports_fp16: bool = False
    supports_bf16: bool = False
    mps_available: bool = False
    resolved_device: str = "cpu"
    python_version: str = ""
    platform: str = ""
    cpu_count: int = 0
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "torch_available": self.torch_available,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "device_count": self.device_count,
            "device_names": self.device_names,
            "vram_total_gb": self.total_memory_gb,
            "vram_free_gb": self.free_memory_gb,
            "compute_capability": self.compute_capability,
            "supports_fp16": self.supports_fp16,
            "supports_bf16": self.supports_bf16,
            "mps_available": self.mps_available,
            "resolved_device": self.resolved_device,
            "resolved_device_label": self.device_label(),
            "python_version": self.python_version,
            "platform": self.platform,
            "cpu_count": self.cpu_count,
            "notes": self.notes,
        }

    def device_label(self) -> str:
        if self.resolved_device == "cuda" and self.device_names:
            return f"CUDA · {self.device_names[0]}"
        if self.resolved_device == "mps":
            return "Apple Silicon (MPS)"
        if self.torch_available:
            return "CPU (no GPU acceleration)"
        return "CPU (ML stack not installed)"


@lru_cache(maxsize=1)
def detect_devices() -> DeviceInfo:
    """Inspect the host once and cache the result."""
    info = DeviceInfo(
        python_version=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        cpu_count=os.cpu_count() or 1,
    )
    try:
        import torch
    except Exception as exc:
        info.notes.append(f"PyTorch is not installed ({exc.__class__.__name__}). Training and diffusion inference are disabled.")
        info.resolved_device = "cpu"
        return info

    info.torch_available = True
    info.torch_version = getattr(torch, "__version__", "unknown")

    try:
        info.cuda_available = bool(torch.cuda.is_available())
    except Exception:
        info.cuda_available = False

    if info.cuda_available:
        try:
            info.cuda_version = getattr(torch.version, "cuda", None)
            info.device_count = torch.cuda.device_count()
            for index in range(info.device_count):
                props = torch.cuda.get_device_properties(index)
                info.device_names.append(props.name)
                info.total_memory_gb.append(round(props.total_memory / 1024 ** 3, 2))
                try:
                    free, _total = torch.cuda.mem_get_info(index)
                    info.free_memory_gb.append(round(free / 1024 ** 3, 2))
                except Exception:
                    info.free_memory_gb.append(float("nan"))
                capability = f"{props.major}.{props.minor}"
                info.compute_capability.append(capability)
            major = int(float(info.compute_capability[0].split(".")[0])) if info.compute_capability else 0
            info.supports_fp16 = major >= 7
            info.supports_bf16 = major >= 8
        except Exception as exc:  # pragma: no cover - driver weirdness
            info.notes.append(f"CUDA detected but properties could not be read: {exc}")
    else:
        info.notes.append(
            "No CUDA GPU detected. Real-time tracking still runs; diffusion try-on will use CPU "
            "(very slow) or be disabled."
        )

    try:
        mps = getattr(torch.backends, "mps", None)
        info.mps_available = bool(mps is not None and mps.is_available())
    except Exception:
        info.mps_available = False

    info.resolved_device = "cuda" if info.cuda_available else ("mps" if info.mps_available else "cpu")
    return info


def resolve_torch_device(preference: str = "auto", force_cpu: bool = False):
    """Return a ``torch.device`` honouring the user's preference when possible."""
    import torch

    available = detect_devices()
    if force_cpu or preference == "cpu":
        return torch.device("cpu")
    if preference in {"auto", "cuda"} and available.cuda_available:
        return torch.device("cuda")
    if preference in {"auto", "mps"} and available.mps_available:
        return torch.device("mps")
    if preference == "cuda" and not available.cuda_available:
        logger.warning("CUDA was requested but is unavailable — falling back to CPU.")
    return torch.device("cpu")


def pick_mixed_precision(requested: str = "auto") -> str:
    """Choose ``fp16`` / ``bf16`` / ``no`` based on the detected GPU capabilities."""
    info = detect_devices()
    requested = (requested or "auto").lower()
    if requested == "no":
        return "no"
    if requested == "bf16":
        return "bf16" if info.supports_bf16 else ("fp16" if info.supports_fp16 else "no")
    if requested == "fp16":
        return "fp16" if info.supports_fp16 else "no"
    if info.supports_bf16:
        return "bf16"
    if info.supports_fp16:
        return "fp16"
    return "no"


def vram_warning(required_gb: float, resolution: int) -> Optional[str]:
    """Return a human-readable warning when the GPU cannot hold a workload."""
    info = detect_devices()
    if not info.cuda_available:
        return (
            f"No CUDA GPU: {resolution}px diffusion try-on is not feasible on this machine. "
            "Use Fast mode (lightweight pipeline), or run the model on a cloud GPU and point "
            "VESTIAI_VTON_API at it."
        )
    free = min([v for v in info.free_memory_gb if v == v] or [0.0]) if info.free_memory_gb else 0.0
    total = max(info.total_memory_gb or [0.0])
    usable = free if free > 0 else total
    if usable and usable < required_gb:
        return (
            f"Only ~{usable:.1f} GB VRAM available but ~{required_gb:.1f} GB is recommended for "
            f"{resolution}px. Lower the resolution, enable gradient checkpointing / attention "
            "slicing, or reduce batch size."
        )
    if total < 6.0:
        return f"{total:.1f} GB VRAM detected — 512px+ diffusion try-on will be tight. Expect OOM warnings."
    return None


def empty_cuda_cache() -> None:
    """Free cached CUDA blocks (called after heavy inference / OOM recovery)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # pragma: no cover
        pass


def gpu_utilization() -> Dict[str, Any]:
    """Best-effort VRAM usage via nvidia-smi (returns ``{}`` when unavailable)."""
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        query = "utilization.gpu,memory.used,memory.total,temperature.gpu"
        out = subprocess.run(  # noqa: S603 - fixed argv
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return {}
        rows = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                rows.append({
                    "gpu_util_pct": float(parts[0]) if parts[0].replace(".", "").isdigit() else None,
                    "vram_used_mb": float(parts[1]) if parts[1].replace(".", "").isdigit() else None,
                    "vram_total_mb": float(parts[2]) if parts[2].replace(".", "").isdigit() else None,
                    "temperature_c": float(parts[3]) if len(parts) > 3 and parts[3].replace(".", "").isdigit() else None,
                })
        return {"gpus": rows}
    except Exception:
        return {}
