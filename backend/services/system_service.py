"""System/model status aggregation for the Model Status + Settings pages.

Everything reported here is *measured*, never assumed: if torch is missing, the report says
so; if a checkpoint is corrupt, the report says which file and why; if the GPU cannot hold the
configured resolution, the report contains the exact warning string shown in the UI.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.models.checkpointing import CheckpointManager
from backend.models.vton_controlnet import environment_report
from backend.utils.device import detect_devices, gpu_utilization, pick_mixed_precision, vram_warning
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SystemService:
    """Read-only views over the environment."""

    def __init__(self, settings, checkpoint_manager: Optional[CheckpointManager] = None) -> None:
        self.settings = settings
        self.checkpoints = checkpoint_manager or CheckpointManager(settings.checkpoints_dir)
        self.started_at = time.time()

    # ---------------------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        """Cheap liveness probe (no model loading)."""
        return {
            "ok": True,
            "app": self.settings.app_name,
            "version": self.settings.app_version,
            "uptime_s": round(time.time() - self.started_at, 1),
            "timestamp": time.time(),
        }

    def model_status(self, adapter=None, closet_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Aggregate status used by the Model Status page and the top-bar indicator."""
        devices = detect_devices()
        checkpoints = self.checkpoints.status()
        ml = environment_report()
        adapter_status = adapter.status() if adapter is not None else None
        ready = bool(adapter_status and adapter_status.get("ready"))
        return {
            "devices": devices.as_dict(),
            "gpu_live": gpu_utilization(),
            "ml_stack": {
                "ready": ml.get("ml_stack_ready"),
                "torch_version": ml.get("torch_version"),
                "diffusers_version": ml.get("diffusers_version"),
                "transformers_version": ml.get("transformers_version"),
                "hint": ml.get("ml_stack_hint"),
            },
            "checkpoints": checkpoints,
            "adapter": adapter_status,
            "resolution": self.settings.resolution,
            "recommended_precision": pick_mixed_precision(self.settings.mixed_precision),
            "vram_warning": vram_warning(8.0, self.settings.resolution),
            "closet": closet_stats or {},
            "summary": {
                "ai_ready": ready,
                "mode": (adapter_status or {}).get("name", "unknown"),
                "message": (adapter_status or {}).get("detail", "No adapter initialised."),
            },
        }

    def device_status(self) -> Dict[str, Any]:
        devices = detect_devices()
        return {
            "device": devices.as_dict(),
            "gpu_live": gpu_utilization(),
            "torch_threads": self.settings.torch_num_threads or "auto",
            "force_cpu": self.settings.force_cpu,
            "resolved_device": devices.resolved_device,
            "label": devices.device_label(),
            "vram_warning": vram_warning(8.0, self.settings.resolution),
        }

    def settings_view(self) -> Dict[str, Any]:
        """Settings exposed to the UI (a curated subset — not the whole dataclass)."""
        s = self.settings
        return {
            "app": {"name": s.app_name, "version": s.app_version, "debug": s.debug},
            "paths": {
                "uploads": str(s.uploads_dir), "garments": str(s.garments_dir),
                "datasets": str(s.datasets_dir), "checkpoints": str(s.checkpoints_dir),
                "results": str(s.results_dir), "captures": str(s.captures_dir),
                "logs": str(s.log_dir),
            },
            "device": {
                "preference": s.device, "force_cpu": s.force_cpu, "allow_tf32": s.allow_tf32,
                "min_gpu_memory_gb": s.min_gpu_memory_gb,
            },
            "realtime": {
                "segmentation": s.segmentation_enabled,
                "smoothing_alpha": s.smoothing_alpha,
                "pose_complexity": s.pose_model_complexity,
                "tracking_lost_grace_frames": s.tracking_lost_grace_frames,
            },
            "tryon": {
                "backend": s.tryon_backend,
                "base_model": s.vton_base_pipeline,
                "resolution": s.resolution,
                "num_inference_steps": s.num_inference_steps,
                "guidance_scale": s.guidance_scale,
                "seed": s.seed,
                "live_ai_interval_ms": s.live_ai_interval_ms,
                "enable_cpu_diffusion": s.enable_cpu_diffusion,
            },
            "training": {
                "mode": s.training_mode, "resolution": s.resolution, "batch_size": s.batch_size,
                "learning_rate": s.learning_rate, "num_epochs": s.num_epochs,
                "mixed_precision": s.mixed_precision, "gradient_accumulation": s.gradient_accumulation,
                "gradient_checkpointing": s.gradient_checkpointing,
                "optimizer": s.optimizer, "scheduler": s.scheduler, "tracker": s.tracker,
            },
            "garment": {
                "canvas": s.target_garment_canvas,
                "background_removal": s.background_removal,
                "min_resolution": s.garment_min_resolution,
            },
            "runtime": {
                "python": sys.version.split()[0],
                "platform": f"{platform.system()} {platform.release()}",
                "open_cv": _module_version("cv2"),
                "numpy": _module_version("numpy"),
                "pillow": _module_version("PIL"),
                "mediapipe": _module_version("mediapipe"),
                "torch": _module_version("torch"),
                "diffusers": _module_version("diffusers"),
                "rembg": _module_version("rembg"),
                "onnxruntime": _module_version("onnxruntime"),
            },
        }

    def disk_usage(self) -> Dict[str, Any]:
        """Report how much space each runtime folder uses (useful before training)."""
        out: Dict[str, Any] = {}
        for name in ("uploads_dir", "garments_dir", "datasets_dir", "checkpoints_dir", "results_dir", "captures_dir", "models_cache_dir", "log_dir"):
            path = Path(getattr(self.settings, name))
            out[name] = {
                "path": str(path),
                "exists": path.exists(),
                "size_mb": round(_dir_size(path) / 1024 ** 2, 2),
                "files": _count_files(path),
            }
        try:
            usage = shutil.disk_usage(str(self.settings.project_root))
            out["volume"] = {
                "total_gb": round(usage.total / 1024 ** 3, 1),
                "used_gb": round(usage.used / 1024 ** 3, 1),
                "free_gb": round(usage.free / 1024 ** 3, 1),
            }
        except OSError:  # pragma: no cover
            pass
        return out

    def components(self) -> List[Dict[str, Any]]:
        """Dependency checklist rendered on the Model Status page."""
        def entry(name: str, module: str, purpose: str, required: bool, install: str) -> Dict[str, Any]:
            version = _module_version(module)
            return {
                "name": name, "module": module, "installed": version is not None, "version": version,
                "purpose": purpose, "required": required, "install": install,
            }

        return [
            entry("FastAPI", "fastapi", "HTTP API", True, "pip install fastapi"),
            entry("NumPy", "numpy", "tensor maths everywhere", True, "pip install numpy"),
            entry("Pillow", "PIL", "image I/O", True, "pip install pillow"),
            entry("OpenCV", "cv2", "warping, segmentation, video", True, "pip install opencv-python"),
            entry("MediaPipe", "mediapipe", "server-side pose + hand tracking", False, "pip install mediapipe"),
            entry("PyTorch", "torch", "model training + inference", False, "pip install torch --index-url https://download.pytorch.org/whl/cu121"),
            entry("Diffusers", "diffusers", "diffusion try-on model", False, "pip install diffusers transformers accelerate"),
            entry("Transformers", "transformers", "CLIP text/image encoders", False, "pip install transformers"),
            entry("rembg", "rembg", "best-quality background removal", False, "pip install rembg onnxruntime"),
            entry("scikit-learn", "sklearn", "classifier head training", False, "pip install scikit-learn"),
            entry("TensorBoard", "tensorboard", "training metrics UI", False, "pip install tensorboard"),
            entry("wanDB", "wandb", "optional experiment tracking", False, "pip install wandb"),
        ]

    def about(self) -> Dict[str, Any]:
        """Content for the About page (kept server-side so docs and UI agree)."""
        return {
            "app": self.settings.app_name,
            "version": self.settings.app_version,
            "tagline": "Two-pipeline virtual try-on: real-time geometric tracking + fine-tuned diffusion synthesis.",
            "pipelines": [
                {
                    "key": "lightweight",
                    "name": "Real-time pose-aware tracking",
                    "latency": "5-25 ms per frame on CPU",
                    "details": "MediaPipe BlazePose landmarks -> smoothed anchors -> affine/perspective warp -> "
                               "occlusion handling -> alpha composite. Runs in the browser and on the server.",
                },
                {
                    "key": "diffusion",
                    "name": "Fine-tuned diffusion VTON",
                    "latency": "1-6 s per image on a modern GPU (many seconds-minutes on CPU)",
                    "details": "ControlNet conditioned on the garment + inpainting UNet conditioned on the "
                               "agnostic person, fine-tuned on VITON-HD / DressCode-compatible data.",
                },
            ],
            "honesty": [
                "The lightweight pipeline is an AR overlay: it preserves the garment texture and your identity, "
                "but it does not synthesise new fabric folds or lighting.",
                "Photorealistic results require a trained checkpoint on a GPU; without one the app says so and "
                "disables that button rather than pretending.",
                "Product photos from shopping sites are copyrighted — bring your own images or use the generated "
                "sample garments.",
            ],
        }


def _module_version(name: str) -> Optional[str]:
    try:
        import importlib

        module = importlib.import_module(name)
        return getattr(module, "__version__", "installed")
    except Exception:
        return None


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:  # pragma: no cover
            continue
    return total


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())
