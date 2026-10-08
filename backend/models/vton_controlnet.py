"""VestiAI's trainable virtual try-on architecture.

Architecture (ControlNet-conditioned inpainting diffusion)
----------------------------------------------------------
::

    ┌──────────────────────────┐
    │ agnostic person RGB      │──► VAE encode ──► masked latent (4ch) ─┐
    │ + garment mask (9ch inp) │                                        │
    └──────────────────────────┘                                        ▼
                                                        ┌───────────────────────────┐
    ┌──────────────────────────┐                        │  UNet2DConditionModel     │
    │ garment image (3ch)      │──► ControlNet ────────►│  (inpainting, 9ch in)     │──► ε
    └──────────────────────────┘        (residuals)     └───────────────────────────┘
    ┌──────────────────────────┐                                    ▲
    │ "a photo of a person     │──► CLIP text encoder ──────────────┘
    │  wearing a red shirt"    │
    └──────────────────────────┘

Why this design
---------------
* The **inpainting UNet** receives the agnostic person + garment mask, which is what
  preserves body identity, pose and background (the same idea as VITON-HD's agnostic
  input and IDM-VTON's masked-person branch).
* The **ControlNet** is conditioned on the *garment itself*, so garment texture, colour,
  print and silhouette are injected at every resolution — this is what keeps the fabric
  looking like the product photo instead of a generic shirt.
* Only the ControlNet (and optionally LoRA adapters on the UNet) are trained. The base
  model stays frozen, which is what makes fine-tuning feasible on a single 8-16 GB GPU,
  exactly as the project brief requires ("fine-tune a pretrained model, do not train a
  giant diffusion model from random initialisation").

Everything in this module is import-guarded: ``torch``/``diffusers`` are imported inside
functions so the API server starts instantly even when the ML stack is absent.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
FALLBACK_BASE_MODEL = "runwayml/stable-diffusion-inpainting"
SD2_INPAINT = "stabilityai/stable-diffusion-2-inpainting"

#: Metadata file written next to every exported checkpoint.
MODEL_CARD_NAME = "vestiai_model_card.json"


@dataclass
class VTONModelConfig:
    """Configuration of the trainable adapter + base model."""

    base_model: str = DEFAULT_BASE_MODEL
    controlnet_from_unet: bool = True
    controlnet_condition_channels: int = 3
    train_controlnet: bool = True
    train_unet_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 16
    gradient_checkpointing: bool = True
    resolution: int = 512
    sample_size: int = 64              # latent size (resolution / 8)
    mixed_precision: str = "fp16"
    use_xformers: bool = False
    enable_attention_slicing: bool = False
    vae_tiling: bool = False
    pretrained_vae: Optional[str] = None
    noise_offset: float = 0.0
    snr_gamma: Optional[float] = 5.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VTONModelConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class ModelCard:
    """Provenance information stored with each exported checkpoint."""

    name: str = "vestiai-vton"
    version: str = "1.0.0"
    architecture: str = "controlnet-inpaint-sd15"
    base_model: str = DEFAULT_BASE_MODEL
    training_mode: str = "QUICK_DEMO"
    dataset: str = "viton-hd-compatible"
    dataset_size: int = 0
    resolution: int = 512
    epochs: int = 0
    global_step: int = 0
    best_metric: Optional[float] = None
    metric_name: str = "val_loss"
    trainable_parameters: int = 0
    created_at: str = ""
    notes: str = ""
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MODEL_CARD_NAME
        with path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)
        return path

    @classmethod
    def load(cls, directory: str | Path) -> Optional["ModelCard"]:
        path = Path(directory) / MODEL_CARD_NAME
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
            return cls(**{k: v for k, v in payload.items() if k in known})
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not read model card %s: %s", path, exc)
            return None


# --------------------------------------------------------------------------------------
# Component construction
# --------------------------------------------------------------------------------------
def torch_dtype_for(mixed_precision: str, device: str):
    """Map a mixed-precision name onto a torch dtype (fp32 for CPU/MPS)."""
    import torch

    if device == "cpu":
        return torch.float32
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return torch.float32


def build_controlnet(config: VTONModelConfig, dtype=None, device: str = "cpu"):
    """Create a ControlNet initialised from the base inpainting UNet.

    Initialising *from the UNet* (diffusers' :meth:`ControlNetModel.from_unet`) means the
    adapter starts as a near-identity residual branch on top of a fully trained model,
    which is the trick that makes fine-tuning converge in a few thousand steps instead of
    millions.
    """
    import torch
    from diffusers import ControlNetModel, StableDiffusionInpaintPipeline

    dtype = dtype or torch_dtype_for(config.mixed_precision, device)
    logger.info("Loading base inpainting pipeline %s (this can take a few minutes the first time)", config.base_model)
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            config.base_model,
            torch_dtype=dtype,
            vae=load_standalone_vae(config, dtype) if config.pretrained_vae else None,
        )
    except Exception as exc:
        logger.warning("Could not load %s (%s). Trying %s", config.base_model, exc, FALLBACK_BASE_MODEL)
        pipe = StableDiffusionInpaintPipeline.from_pretrained(FALLBACK_BASE_MODEL, torch_dtype=dtype)
        config.base_model = FALLBACK_BASE_MODEL

    controlnet = ControlNetModel.from_unet(pipe.unet, conditioning_channels=config.controlnet_condition_channels)
    controlnet.to(device=device, dtype=dtype)
    return controlnet, pipe


def load_standalone_vae(config: VTONModelConfig, dtype):
    """Load an optional higher-fidelity VAE (e.g. ``stabilityai/sd-vae-ft-mse``)."""
    from diffusers import AutoencoderKL

    if not config.pretrained_vae:
        return None
    try:
        return AutoencoderKL.from_pretrained(config.pretrained_vae, torch_dtype=dtype)
    except Exception as exc:
        logger.warning("Could not load VAE %s: %s", config.pretrained_vae, exc)
        return None


def build_components(config: VTONModelConfig, device: str = "cpu", controlnet_path: Optional[str | Path] = None):
    """Load every pipeline component, optionally restoring a trained ControlNet.

    Returns ``(pipe, controlnet, noise_scheduler)``.
    """
    import torch
    from diffusers import DDPMScheduler, UniPCMultistepScheduler

    controlnet, pipe = build_controlnet(config, device=device)

    if controlnet_path:
        controlnet_path = Path(controlnet_path)
        if (controlnet_path / "diffusion_pytorch_model.safetensors").exists() or (controlnet_path / "config.json").exists():
            logger.info("Restoring trained ControlNet weights from %s", controlnet_path)
            controlnet = ControlNetModel.from_pretrained(str(controlnet_path), torch_dtype=next(controlnet.parameters()).dtype)
            controlnet.to(device)
        else:
            logger.warning("ControlNet path %s has no weights; using the untrained-from-UNet initialisation.", controlnet_path)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, controlnet, noise_scheduler


def freeze_unet(pipe, train_text_encoder: bool = False) -> Dict[str, int]:
    """Freeze the base model; returns a parameter-count summary for logging."""
    pipe.unet.requires_grad_(False)
    if not train_text_encoder:
        pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    summary = {
        "unet_trainable": sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad),
        "text_encoder_trainable": sum(p.numel() for p in pipe.text_encoder.parameters() if p.requires_grad),
        "vae_trainable": sum(p.numel() for p in pipe.vae.parameters() if p.requires_grad),
    }
    return summary


def add_lora_to_unet(pipe, config: VTONModelConfig) -> int:
    """Optionally attach LoRA adapters to the UNet so try-on style can be tuned too."""
    try:
        from peft import LoraConfig, get_peft_model  # type: ignore

        lora = LoraConfig(
            r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=0.05, bias="none",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        pipe.unet = get_peft_model(pipe.unet, lora)
        trainable = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
        logger.info("Attached LoRA (rank=%s) to UNet: %s trainable parameters", config.lora_rank, f"{trainable:,}")
        return trainable
    except Exception as exc:
        logger.warning("LoRA requested but unavailable (%s). Training ControlNet only.", exc)
        return 0


def enable_memory_optimizations(pipe, controlnet, config: VTONModelConfig, device: str) -> List[str]:
    """Apply the standard diffusers memory tricks and report what was enabled."""
    applied: List[str] = []
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            applied.append("xformers")
        except Exception:
            try:
                pipe.enable_attention_slicing("max")
                applied.append("attention_slicing")
            except Exception:  # pragma: no cover
                pass
    else:
        try:
            pipe.enable_attention_slicing("max")
            applied.append("attention_slicing")
        except Exception:  # pragma: no cover
            pass

    if config.gradient_checkpointing:
        try:
            pipe.unet.enable_gradient_checkpointing()
            controlnet.enable_gradient_checkpointing()
            applied.append("gradient_checkpointing")
        except Exception as exc:  # pragma: no cover
            logger.debug("gradient checkpointing unavailable: %s", exc)
    if config.vae_tiling:
        try:
            pipe.enable_vae_tiling()
            applied.append("vae_tiling")
        except Exception:  # pragma: no cover
            pass
    return applied


def encode_text(pipe, prompts: List[str], device: str = "cpu", dtype=None):
    """CLIP-encode prompts, returning ``(prompt_embeds, pooled)`` if the UNet needs pooled."""
    import torch

    dtype = dtype or next(pipe.text_encoder.parameters()).dtype
    tokens = pipe.tokenizer(
        prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    with torch.no_grad():
        encoded = pipe.text_encoder(tokens)[0]
    return encoded.to(dtype)


def encode_images_vae(pipe, images, device: str = "cpu", generator=None):
    """Encode a batch of ``[-1, 1]`` tensors into the VAE latent space."""
    import torch

    dtype = next(pipe.vae.parameters()).dtype
    latents = pipe.vae.encode(images.to(device=device, dtype=dtype)).latent_dist
    _ = generator
    return latents.sample() * pipe.vae.config.scaling_factor


def decode_latents(pipe, latents):
    """Decode latents to a ``[0, 1]`` image tensor."""
    import torch

    dtype = next(pipe.vae.parameters()).dtype
    latents = latents.to(dtype=dtype) / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents).sample
    return (image / 2 + 0.5).clamp(0, 1)


def build_control_image(garment_rgb: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Prepare the ControlNet conditioning image (garment on white, model resolution)."""
    import cv2

    width, height = size
    image = np.asarray(garment_rgb)[..., :3]
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA if image.shape[1] > width else cv2.INTER_CUBIC)
    return image.astype(np.uint8)


# --------------------------------------------------------------------------------------
# Export / import
# --------------------------------------------------------------------------------------
def save_trained_adapter(
    directory: str | Path,
    controlnet,
    model_card: ModelCard,
    config: VTONModelConfig,
    processor=None,
) -> Path:
    """Save the trained ControlNet (+ model card) in a diffusers-loadable layout."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    controlnet.save_pretrained(str(directory), safe_serialization=True)
    if processor is not None:
        processor.save_pretrained(str(directory / "feature_extractor"))
    with (directory / "vestiai_config.json").open("w", encoding="utf-8") as fh:
        json.dump(config.to_dict(), fh, indent=2)
    model_card.save(directory)
    logger.info("Saved trained adapter to %s", directory)
    return directory


def adapter_is_valid(directory: str | Path) -> bool:
    """True when a directory looks like a loadable ControlNet checkpoint."""
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        return False
    has_config = (directory / "config.json").exists()
    has_weights = any(
        (directory / name).exists()
        for name in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin", "diffusion_pytorch_model.fp16.safetensors")
    )
    return has_config and has_weights


def describe_checkpoint(directory: str | Path) -> Dict[str, Any]:
    """Inspect a checkpoint directory for the Model Status page."""
    directory = Path(directory)
    info: Dict[str, Any] = {"path": str(directory), "exists": directory.exists(), "valid": adapter_is_valid(directory)}
    if not directory.exists():
        info["reason"] = "directory does not exist"
        return info
    weights = list(directory.glob("*.safetensors")) + list(directory.glob("*.bin"))
    info["weights"] = [w.name for w in weights]
    info["size_mb"] = round(sum(w.stat().st_size for w in weights) / 1024 ** 2, 1) if weights else 0.0
    card = ModelCard.load(directory)
    if card:
        info["model_card"] = asdict(card)
    config_path = directory / "vestiai_config.json"
    if config_path.exists():
        try:
            info["config"] = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover
            pass
    if not info["valid"]:
        info["reason"] = "missing config.json or weights (run export_model.py / training again)"
    return info


def count_parameters(module) -> int:
    """Trainable + total parameter counts."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def module_size_mb(module) -> float:
    return round(sum(p.numel() * p.element_size() for p in module.parameters()) / 1024 ** 2, 1)


def memory_summary(device: str) -> Dict[str, Any]:
    """Peak GPU memory helper for the training dashboard."""
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        return {"device": device, "peak_allocated_mb": None, "peak_reserved_mb": None}
    return {
        "device": device,
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024 ** 2, 1),
        "current_allocated_mb": round(torch.cuda.memory_allocated() / 1024 ** 2, 1),
    }


def environment_report(config: Optional[VTONModelConfig] = None) -> Dict[str, Any]:
    """Everything the dashboard needs to explain *why* training/inference is or isn't possible."""
    from backend.utils.device import detect_devices, pick_mixed_precision, vram_warning

    devices = detect_devices()
    report: Dict[str, Any] = devices.as_dict()
    report["recommended_mixed_precision"] = pick_mixed_precision("auto")
    report["vram_warning"] = vram_warning(8.0, config.resolution if config else 512)
    try:
        import diffusers

        report["diffusers_version"] = diffusers.__version__
    except Exception:
        report["diffusers_version"] = None
    try:
        import transformers

        report["transformers_version"] = transformers.__version__
    except Exception:
        report["transformers_version"] = None
    report["ml_stack_ready"] = bool(report.get("torch_available") and report.get("diffusers_version"))
    if not report["ml_stack_ready"]:
        report["ml_stack_hint"] = (
            "Install the ML extra: pip install -r requirements-ml.txt "
            "(or scripts/setup.py --profile ml). Real-time tracking works without it."
        )
    return report


def seed_everything(seed: int = 42) -> None:
    """Deterministic seeding for reproducible training runs."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_runtime(allow_tf32: bool = True, num_threads: int = 0) -> None:
    """Apply global torch runtime flags (TF32, thread count)."""
    import torch

    if num_threads and num_threads > 0:
        torch.set_num_threads(num_threads)
    if allow_tf32 and torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:  # pragma: no cover
            pass
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _ = np
