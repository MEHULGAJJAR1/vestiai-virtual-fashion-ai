"""Central configuration for VestiAI.

Configuration is resolved with the following precedence (highest wins):

1. Explicit keyword arguments passed to :func:`get_settings`.
2. Environment variables (``VESTIAI_*``).
3. A YAML config file (``--config`` / ``VESTIAI_CONFIG`` / ``configs/default.yaml``).
4. Built-in defaults declared below.

Only the standard library plus ``PyYAML`` is required, so the settings module can be
imported before the heavy ML dependencies (torch / diffusers / mediapipe) are installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

try:  # PyYAML is a hard requirement of the project, but keep the import defensive.
    import yaml
except Exception:  # pragma: no cover - only hit in a bare environment
    yaml = None  # type: ignore[assignment]

# --------------------------------------------------------------------------------------
# Repository layout
# --------------------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

ENV_PREFIX = "VESTIAI_"


def _env_key(name: str) -> str:
    return f"{ENV_PREFIX}{name.upper()}"


def _coerce(raw: str, target_type: Any) -> Any:
    """Coerce an environment-variable string into the type of the default value."""
    if target_type is bool:
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    if target_type is int:
        return int(float(raw))
    if target_type is float:
        return float(raw)
    if target_type is Path or (isinstance(target_type, type) and issubclass(target_type, Path)):
        return Path(raw).expanduser()
    if target_type is str:
        return raw
    return raw


@dataclass
class Settings:
    """Strongly typed application settings.

    Attributes are grouped by concern. Every attribute can be overridden through the
    ``VESTIAI_<UPPER_SNAKE_CASE_NAME>`` environment variable, e.g.
    ``VESTIAI_FORCE_CPU=true`` or ``VESTIAI_DEFAULT_RESOLUTION=768``.
    """

    # ----- general -------------------------------------------------------------------
    app_name: str = "VestiAI"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs"
    cors_origins: str = "*"

    # ----- storage -------------------------------------------------------------------
    project_root: Path = PROJECT_ROOT
    uploads_dir: Path = PROJECT_ROOT / "uploads"
    garments_dir: Path = PROJECT_ROOT / "garments"
    datasets_dir: Path = PROJECT_ROOT / "datasets"
    checkpoints_dir: Path = PROJECT_ROOT / "checkpoints"
    results_dir: Path = PROJECT_ROOT / "results"
    captures_dir: Path = PROJECT_ROOT / "captures"
    models_cache_dir: Path = PROJECT_ROOT / "models_cache"
    closet_db: Path = PROJECT_ROOT / "garments" / "closet.json"
    samples_dir: Path = PROJECT_ROOT / "datasets" / "samples"

    # ----- device / runtime ----------------------------------------------------------
    device: str = "auto"              # auto | cuda | mps | cpu
    force_cpu: bool = False
    allow_tf32: bool = True
    cuda_memory_fraction: float = 0.9
    min_gpu_memory_gb: float = 6.0    # below this the diffusion path warns / disables
    torch_num_threads: int = 0        # 0 => leave to torch default

    # ----- garment preprocessing -----------------------------------------------------
    garment_max_side: int = 1024
    garment_min_side: int = 256
    garment_min_resolution: int = 160
    background_removal: str = "auto"   # auto | rembg | grabcut | none
    target_garment_canvas: int = 768   # square canvas the garment is normalized onto

    # ----- pose / realtime CV --------------------------------------------------------
    pose_model_complexity: int = 1
    pose_min_detection_confidence: float = 0.5
    pose_min_tracking_confidence: float = 0.5
    segmentation_enabled: bool = True
    smoothing_alpha: float = 0.35      # EMA factor for landmark smoothing (0=raw, 1=frozen)
    tracking_lost_grace_frames: int = 8

    # ----- VTON inference ------------------------------------------------------------
    tryon_backend: str = "auto"        # auto | diffusion | lightweight | disabled
    vton_pretrained: str = "yisol/IDM-VTON"
    vton_base_pipeline: str = "stabilityai/stable-diffusion-2-inpainting"
    resolution: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 2.0
    seed: int = 42
    live_ai_interval_ms: int = 700     # minimum ms between heavy VTON calls in live mode
    live_ai_max_side: int = 512
    enable_cpu_diffusion: bool = False  # diffusion on CPU is minutes/image -> opt-in

    # ----- checkpoint discovery ------------------------------------------------------
    best_checkpoint: Path = PROJECT_ROOT / "checkpoints" / "best_model"
    latest_checkpoint: Path = PROJECT_ROOT / "checkpoints" / "latest_model"

    # ----- training ------------------------------------------------------------------
    training_mode: str = "QUICK_DEMO"   # QUICK_DEMO | FINE_TUNE | FULL_TRAINING
    config_file: Optional[Path] = None
    mixed_precision: str = "fp16"       # no | fp16 | bf16
    gradient_accumulation: int = 4
    gradient_checkpointing: bool = True
    batch_size: int = 2
    learning_rate: float = 1e-5
    num_epochs: int = 10
    num_workers: int = 2
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_steps: int = 100
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    checkpoint_every_n_epochs: int = 1
    validate_every_n_epochs: int = 1
    save_top_k: int = 3
    early_stopping_patience: int = 5
    resume: bool = True
    log_every_n_steps: int = 10
    sample_data_mode: bool = True
    tracker: str = "tensorboard"        # tensorboard | wandb | none

    # ---------------------------------------------------------------------------------
    extra: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------------------------
    def __post_init__(self) -> None:
        for f in fields(self):
            if f.name == "extra":
                continue
            current = getattr(self, f.name)
            if isinstance(current, Path) or f.type is Path:
                if not isinstance(current, Path):
                    setattr(self, f.name, Path(str(current)).expanduser())

    # -- helpers ----------------------------------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = str(value) if isinstance(value, Path) else value
        return out

    def ensure_dirs(self) -> None:
        """Create every runtime directory. Safe to call repeatedly."""
        for name in (
            "uploads_dir", "garments_dir", "datasets_dir", "checkpoints_dir",
            "results_dir", "captures_dir", "models_cache_dir", "log_dir", "samples_dir",
        ):
            Path(getattr(self, name)).mkdir(parents=True, exist_ok=True)
        for sub in ("best_model", "latest_model", "epochs"):
            (Path(self.checkpoints_dir) / sub).mkdir(parents=True, exist_ok=True)

    def resolve_device(self) -> str:
        """Return the effective torch device string without importing torch at module load."""
        if self.force_cpu:
            return "cpu"
        if self.device != "auto":
            return self.device
        try:  # pragma: no cover - depends on host hardware
            import torch

            if torch.cuda.is_available():
                return "cuda"
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def _load_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if not path or yaml is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at the top level.")
    return data


def _apply_env(settings: Settings) -> Settings:
    valid = {f.name: f for f in fields(Settings)}
    for name in list(valid):
        if name == "extra":
            continue
        raw = os.environ.get(_env_key(name))
        if raw is None:
            continue
        setattr(settings, name, _coerce(raw, type(getattr(settings, name))))
    return settings


def load_settings(
    config_file: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Settings:
    """Build a :class:`Settings` instance from YAML, environment and explicit overrides."""
    path = Path(config_file) if config_file else None
    if path is None:
        env_path = os.environ.get(_env_key("config"))
        if env_path:
            path = Path(env_path)
        elif (PROJECT_ROOT / "configs" / "default.yaml").exists():
            path = PROJECT_ROOT / "configs" / "default.yaml"

    data = _load_yaml(path)
    data.setdefault("config_file", str(path) if path else None)

    known = {f.name for f in fields(Settings)}
    clean = {k: v for k, v in data.items() if k in known}
    extra = {k: v for k, v in data.items() if k not in known}

    settings = Settings(**clean)
    if extra:
        settings.extra.update(extra)
    if overrides:
        for key, value in overrides.items():
            if key in known:
                setattr(settings, key, value)
            else:
                settings.extra[key] = value
    for key, value in kwargs.items():
        if value is None:
            continue
        if key in known:
            setattr(settings, key, value)
        else:
            settings.extra[key] = value

    _apply_env(settings)
    settings.ensure_dirs()
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton used by API routes and services."""
    return load_settings()


def reload_settings(config_file: Optional[str | Path] = None, **kwargs: Any) -> Settings:
    """Invalidate the cached singleton (used by tests and the Settings page)."""
    get_settings.cache_clear()  # type: ignore[attr-defined]
    settings = load_settings(config_file, **kwargs)
    # repopulate cache
    get_settings.__wrapped__.__dict__  # noqa: B018 - documentation only
    return settings
