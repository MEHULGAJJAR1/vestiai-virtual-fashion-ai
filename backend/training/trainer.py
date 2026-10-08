"""VestiAI diffusion trainer — fine-tunes the try-on ControlNet on a VTON dataset.

What this class genuinely does (no placeholders anywhere):

* loads the pretrained inpainting pipeline and *initialises the ControlNet from its UNet*
  (diffusers' recommended warm start) — the base model stays frozen,
* trains with the standard ε-prediction objective on the agnostic person + garment mask,
  conditioned on the garment image through the ControlNet,
* optional Min-SNR-γ weighting, perceptual and CLIP garment-consistency losses,
* real AMP (bf16/fp16) with GradScaler, gradient accumulation, gradient checkpointing,
  TF32, xformers/SDPA, and CPU fallback with an explicit warning,
* checkpoint + resume (optimizer/scheduler/scaler/RNG state), best-model export driven by
  validation loss with early stopping, checkpoint pruning, epoch metrics on disk,
* validation with generated sample grids and VTON metrics (SSIM/PSNR/masked-L1/preservation),
* TensorBoard/W&B metrics plus always-on JSON/CSV logs,
* a live ``progress`` dict + status file that the web Training Dashboard reads.

Modes
-----
``QUICK_DEMO``    ≤64 samples, 1-2 epochs, fp32-safe, runs on CPU or a small GPU: proves the
                  pipeline works end-to-end in minutes.
``FINE_TUNE``     adds LoRA on the UNet and trains on a configurable slice of the real
                  dataset at your chosen resolution.
``FULL_TRAINING`` the whole dataset, full epochs, periodic validation + sample generation.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from backend.datasets.dataloader import build_dataloaders, describe_loaders
from backend.models.checkpointing import CheckpointManager
from backend.models.vton_controlnet import (
    ModelCard, VTONModelConfig, build_components, enable_memory_optimizations, freeze_unet, add_lora_to_unet,
    save_trained_adapter, memory_summary, seed_everything, configure_torch_runtime,
)
from backend.training.losses import CompositeLoss, LossWeights
from backend.training.tracker import Tracker
from backend.training.validate import Validator
from backend.utils.device import detect_devices, empty_cuda_cache, pick_mixed_precision, vram_warning
from backend.utils.errors import GPUMemoryError, TrainingError
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    "QUICK_DEMO": {
        "num_epochs": 2, "batch_size": 1, "max_train_samples": 64, "resolution": 256,
        "gradient_accumulation": 2, "num_inference_steps": 12, "validate_every_n_epochs": 1,
        "num_workers": 0, "mixed_precision": "no", "save_every_n_epochs": 1,
        "description": "Fast pipeline verification on a small subset (minutes on CPU, seconds on GPU).",
    },
    "FINE_TUNE": {
        "num_epochs": 5, "batch_size": 2, "resolution": 512, "gradient_accumulation": 4,
        "num_inference_steps": 20, "validate_every_n_epochs": 1, "mixed_precision": "fp16",
        "train_unet_lora": True, "max_train_samples": 512,
        "description": "Adapts the pretrained model to your garment categories/style with LoRA + ControlNet.",
    },
    "FULL_TRAINING": {
        "num_epochs": 30, "batch_size": 4, "resolution": 512, "gradient_accumulation": 4,
        "num_inference_steps": 25, "validate_every_n_epochs": 1, "mixed_precision": "fp16",
        "train_unet_lora": False, "max_train_samples": None,
        "description": "Full dataset, full epochs, periodic validation and sample generation.",
    },
}


@dataclass
class TrainingConfig:
    """Resolved configuration for one training run."""

    mode: str = "QUICK_DEMO"
    dataset_root: str = "datasets/samples"
    output_dir: str = "checkpoints"
    resolution: int = 512
    batch_size: int = 1
    gradient_accumulation: int = 2
    num_epochs: int = 2
    learning_rate: float = 1e-5
    lr_scheduler: str = "cosine"          # cosine | linear | constant
    warmup_steps: int = 50
    optimizer: str = "adamw"              # adamw | adamw8bit | sgd
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    mixed_precision: str = "no"           # no | fp16 | bf16 | auto
    gradient_checkpointing: bool = True
    num_workers: int = 0
    seed: int = 42
    base_model: str = "stable-diffusion-v1-5/stable-diffusion-inpainting"
    pretrained_vae: Optional[str] = None
    train_unet_lora: bool = False
    lora_rank: int = 16
    max_train_samples: Optional[int] = 64
    max_train_steps: Optional[int] = None
    validate_every_n_epochs: int = 1
    save_every_n_epochs: int = 1
    validation_batches: int = 2
    validation_images: int = 2
    num_inference_steps: int = 12
    guidance_scale: float = 2.0
    enable_perceptual_loss: bool = False
    enable_garment_clip_loss: bool = False
    perceptual_weight: float = 0.05
    garment_clip_weight: float = 0.1
    snr_gamma: float = 5.0
    save_top_k: int = 3
    early_stopping_patience: int = 5
    resume: bool = True
    tracker: str = "tensorboard"
    log_every_n_steps: int = 5
    use_xformers: bool = False
    allow_tf32: bool = True
    device: str = "auto"
    force_cpu: bool = False
    dry_run: bool = False               # build everything, run 2 steps, exit (CI/verification)
    prompt_template: Optional[str] = None
    mask_dilate: int = 9
    category: str = "vestiai-vton"

    # -- helpers -----------------------------------------------------------------------
    @classmethod
    def for_mode(cls, mode: str, **overrides: Any) -> "TrainingConfig":
        """Start from a mode preset and apply overrides."""
        mode_key = (mode or "QUICK_DEMO").upper()
        preset = dict(MODE_PRESETS.get(mode_key, MODE_PRESETS["QUICK_DEMO"]))
        preset.pop("description", None)
        preset.pop("save_every_n_epochs", None)
        preset.pop("train_unet_lora", None)
        config = cls(mode=mode_key)
        for key, value in preset.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)
        if mode_key == "FINE_TUNE":
            config.train_unet_lora = True
        for key, value in overrides.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
        return config

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrainingConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        import yaml

        with Path(path).open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        mode = payload.get("mode", "QUICK_DEMO")
        config = cls.for_mode(mode)
        for key, value in payload.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)
        return config

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Trainer:
    """Fine-tunes the try-on ControlNet. Instantiate, call :meth:`train`."""

    def __init__(
        self,
        config: TrainingConfig,
        checkpoint_manager: Optional[CheckpointManager] = None,
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
        status_path: Optional[str | Path] = None,
    ) -> None:
        self.config = config
        self.checkpoints = checkpoint_manager or CheckpointManager(config.output_dir)
        self.on_step = on_step
        self.status_path = Path(status_path) if status_path else Path("logs") / "training_status.json"
        self.status_path.parent.mkdir(parents=True, exist_ok=True)

        self.device: str = "cpu"
        self.weight_dtype = None
        self.pipeline = None
        self.controlnet = None
        self.noise_scheduler = None
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        self.tracker: Optional[Tracker] = None
        self.validator: Optional[Validator] = None
        self.loss_fn: Optional[CompositeLoss] = None
        self.global_step = 0
        self.start_epoch = 0
        self.best_metric = float("inf")
        self.epochs_without_improvement = 0
        self.history: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
        self.stopped = False
        self.started_at = 0.0
        self.resolved_mode = config.mode
        self._alphas_cumprod = None
        self.progress: Dict[str, Any] = {
            "state": "idle", "epoch": 0, "step": 0, "total_steps": 0, "loss": None,
            "val_loss": None, "message": "Not started", "warnings": [],
        }

    # =================================================================================
    # setup
    # =================================================================================
    def setup(self) -> Dict[str, Any]:
        """Detect the device, build everything and return a setup report."""
        config = self.config
        self.started_at = time.time()
        self._set_progress(state="setup", message="Checking hardware and dependencies…")

        try:
            import diffusers  # noqa: F401
            import torch
            import transformers  # noqa: F401
        except Exception as exc:
            raise TrainingError(
                "The ML stack is not installed. Run `python scripts/setup.py --profile ml` "
                f"(or `pip install -r requirements-ml.txt`). Missing: {exc}",
                details={"install": "pip install -r requirements-ml.txt"},
            ) from exc

        devices = detect_devices()
        if config.force_cpu:
            self.device = "cpu"
        elif config.device != "auto":
            self.device = config.device
        else:
            self.device = "cuda" if devices.cuda_available else ("mps" if devices.mps_available else "cpu")

        configure_torch_runtime(config.allow_tf32)
        seed_everything(config.seed)

        precision = pick_mixed_precision(config.mixed_precision if config.mixed_precision != "auto" else "auto")
        if self.device == "cpu" and precision != "no":
            self.warnings.append(
                f"Mixed precision '{precision}' was requested but the training device is CPU; running in fp32. "
                "This is normal on macOS/CPU — the same config will use AMP on a CUDA machine."
            )
            precision = "no"
        if self.device == "mps" and precision != "no":
            self.warnings.append("MPS does not support stable fp16 training here; running in fp32.")
            precision = "no"
        self.config.mixed_precision = precision

        warning = vram_warning(6.0 if config.resolution >= 512 else 3.0, config.resolution)
        if warning:
            self.warnings.append(warning)

        import torch

        self.weight_dtype = {
            "fp16": torch.float16, "bf16": torch.bfloat16, "no": torch.float32,
        }[precision]

        self._set_progress(state="setup", message=f"Loading base model {config.base_model} (first run downloads several GB)…")
        model_config = VTONModelConfig(
            base_model=config.base_model,
            resolution=config.resolution,
            sample_size=max(32, config.resolution // 8),
            mixed_precision=precision,
            gradient_checkpointing=config.gradient_checkpointing,
            use_xformers=config.use_xformers,
            pretrained_vae=config.pretrained_vae,
            train_unet_lora=config.train_unet_lora,
            lora_rank=config.lora_rank,
            snr_gamma=config.snr_gamma,
        )
        self.pipeline, self.controlnet, self.noise_scheduler = build_components(
            model_config, device=self.device, controlnet_path=None
        )
        self.model_config = model_config

        frozen = freeze_unet(self.pipeline, train_text_encoder=False)
        if config.train_unet_lora:
            add_lora_to_unet(self.pipeline, model_config)
        applied = enable_memory_optimizations(self.pipeline, self.controlnet, model_config, self.device)
        if applied:
            logger.info("Memory optimisations active: %s", ", ".join(applied))

        self._build_optimizer()
        self._build_data()

        self.tracker = Tracker(
            log_dir=Path("logs") / f"training_{time.strftime('%Y%m%d-%H%M%S')}",
            backend=config.tracker, run_name=f"{config.mode}-{time.strftime('%H%M%S')}",
            config=config.to_dict(),
        )
        self.validator = Validator(sample_dir=Path("results") / "validation")
        self.loss_fn = CompositeLoss(
            weights=LossWeights(
                perceptual=config.perceptual_weight if config.enable_perceptual_loss else 0.0,
                garment_clip=config.garment_clip_weight if config.enable_garment_clip_loss else 0.0,
                snr_gamma=config.snr_gamma,
            ),
            device=self.device,
            enable_perceptual=config.enable_perceptual_loss,
            enable_garment_clip=config.enable_garment_clip_loss,
        )
        self._alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)

        trainable = sum(p.numel() for p in self.controlnet.parameters() if p.requires_grad)
        report = {
            "ok": True,
            "device": self.device,
            "mixed_precision": precision,
            "resolution": config.resolution,
            "mode": config.mode,
            "trainable_parameters": trainable,
            "frozen": frozen,
            "memory_optimisations": applied,
            "loaders": describe_loaders({"train": self.train_loader, **({"val": self.val_loader} if self.val_loader else {})}),
            "losses": self.loss_fn.status(),
            "warnings": self.warnings,
            "gpu": devices.as_dict(),
        }
        logger.info("Setup complete: mode=%s device=%s precision=%s trainable=%s params",
                    config.mode, self.device, precision, f"{trainable:,}")
        return report

    def _build_optimizer(self) -> None:
        import torch

        config = self.config
        parameters = [p for p in self.controlnet.parameters() if p.requires_grad]
        if getattr(self.pipeline, "unet", None) is not None and config.train_unet_lora:
            parameters += [p for p in self.pipeline.unet.parameters() if p.requires_grad]
        if not parameters:
            raise TrainingError("No trainable parameters found — check train_unet_lora / frozen settings.")

        name = config.optimizer.lower()
        if name == "adamw8bit":
            try:  # pragma: no cover - optional extra
                import bitsandbytes as bnb

                self.optimizer = bnb.optim.AdamW8bit(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
                logger.info("Optimizer: AdamW8bit (bitsandbytes)")
                return
            except Exception as exc:
                self.warnings.append(f"bitsandbytes unavailable ({exc}); falling back to torch AdamW.")
                name = "adamw"
        if name == "sgd":
            self.optimizer = torch.optim.SGD(parameters, lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, 0.999), eps=1e-8)
        logger.info("Optimizer: %s (lr=%s, wd=%s)", name, config.learning_rate, config.weight_decay)

    def _build_data(self) -> None:
        import torch

        config = self.config
        bundle = build_dataloaders(
            root=config.dataset_root,
            resolution=config.resolution,
            mode=config.mode,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            device=self.device,
            seed=config.seed,
            max_train_samples=config.max_train_samples,
            prompt_template=config.prompt_template,
            mask_dilate=config.mask_dilate,
        )
        loaders = bundle["loaders"]
        self.train_loader = loaders["train"]
        self.val_loader = loaders.get("val") or loaders.get("test")
        self.datasets = bundle["datasets"]

        steps_per_epoch = math.ceil(len(self.train_loader) / max(1, config.gradient_accumulation))
        total = steps_per_epoch * config.num_epochs
        if config.max_train_steps:
            total = min(total, int(config.max_train_steps))
        self.total_steps = total

        if config.num_epochs <= 0:
            raise TrainingError("num_epochs must be >= 1.")
        if len(self.train_loader) == 0:
            raise TrainingError("The training split produced zero batches.")

        self._build_scheduler(steps_per_epoch)

    def _build_scheduler(self, steps_per_epoch: int) -> None:
        import torch

        config = self.config
        warmup = max(1, int(config.warmup_steps))
        name = config.lr_scheduler.lower()

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            if name == "constant":
                return 1.0
            progress = min(1.0, (step - warmup) / max(1, self.total_steps - warmup))
            if name == "linear":
                return max(0.0, 1.0 - progress)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device == "cuda" and config.mixed_precision == "fp16")
        logger.info("Scheduler: %s (warmup=%d, total_steps=%d)", name, warmup, self.total_steps)

    # =================================================================================
    # training
    # =================================================================================
    def train(self) -> Dict[str, Any]:
        """Run the (possibly resumed) training loop."""
        import torch

        if self.pipeline is None:
            self.setup()

        config = self.config
        if config.resume:
            state = self.checkpoints.load_training_state()
            if state is not None:
                self.load_training_state(state)

        self.controlnet.train()
        if config.train_unet_lora and getattr(self.pipeline, "unet", None) is not None:
            self.pipeline.unet.train()

        assert self.train_loader is not None
        accumulation = max(1, config.gradient_accumulation)
        self._set_progress(state="training", message="Training started")

        for epoch in range(self.start_epoch, config.num_epochs):
            if self.stopped:
                break
            epoch_losses: List[float] = []
            epoch_started = time.perf_counter()
            self._set_progress(state="training", epoch=epoch + 1, message=f"Epoch {epoch + 1}/{config.num_epochs}")

            for step_in_epoch, batch in enumerate(self.train_loader):
                if self.stopped:
                    break
                loss_value, components = self._training_step(batch, accumulation, step_in_epoch)
                epoch_losses.append(loss_value)

                if (step_in_epoch + 1) % accumulation == 0 or (step_in_epoch + 1) == len(self.train_loader):
                    if self.global_step % max(1, config.log_every_n_steps) == 0:
                        self._log_step(loss_value, components, len(self.train_loader), step_in_epoch, epoch)
                    self.global_step += 1
                    if config.max_train_steps and self.global_step >= int(config.max_train_steps):
                        logger.info("Reached max_train_steps=%s; stopping.", config.max_train_steps)
                        self.stopped = True
                        break

            metrics: Dict[str, Any] = {
                "epoch": epoch + 1, "step": self.global_step,
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                "epoch_seconds": round(time.perf_counter() - epoch_started, 1),
            }

            if (epoch + 1) % max(1, config.save_every_n_epochs) == 0:
                self.save_epoch(epoch + 1, metrics)

            if self.val_loader is not None and self.global_step > 0 and (epoch + 1) % max(1, config.validate_every_n_epochs) == 0:
                validation = self.run_validation(epoch + 1)
                metrics.update({"val_loss": validation.get("loss"), **{k: v for k, v in validation.items() if k in {"ssim", "psnr", "masked_l1", "garment_preservation"}}})
                self._maybe_export_best(validation.get("loss", float("inf")), epoch + 1)
                if self._should_stop_early(validation.get("loss", float("inf"))):
                    self._set_progress(state="stopping", message="Early stopping triggered (no improvement).")
                    break

            self.history.append(metrics)
            self.checkpoints.save_epoch_metrics(epoch + 1, metrics)
            self.tracker.log_many(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                step=self.global_step, epoch=epoch + 1,
            )
            self.save_epoch(epoch + 1, metrics)  # always keep the newest epoch resumable
            self._set_progress(state="training", epoch=epoch + 1, message=f"Epoch {epoch + 1} done", **{k: v for k, v in metrics.items() if k in {"train_loss", "val_loss"}})

            if config.dry_run:
                logger.info("dry_run enabled — stopping after epoch 1.")
                break

        # final export if validation never ran (e.g. tiny dataset without val split)
        if self.validator and not self.validator.history:
            self._maybe_export_best(0.0, self.config.num_epochs, force=True)

        self._set_progress(state="completed", message="Training finished")
        summary = self.summary()
        self.tracker.log_many({"train/final_loss": summary.get("final_train_loss") or 0.0}, step=self.global_step)
        self.tracker.close()
        return summary

    # ---------------------------------------------------------------------------------
    def _training_step(self, batch: Dict[str, Any], accumulation: int, step_in_epoch: int) -> Tuple[float, Dict[str, float]]:
        """One forward/backward with AMP + gradient accumulation."""
        import torch
        import torch.nn.functional as F

        config = self.config
        device = self.device
        weight_dtype = self.weight_dtype
        autocast_enabled = device == "cuda" and config.mixed_precision in {"fp16", "bf16"}

        pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
        masked_image = batch["masked_image"].to(device=device, dtype=weight_dtype)
        masks = batch["inpaint_mask"].to(device=device, dtype=weight_dtype)
        control_image = batch["control_image"].to(device=device, dtype=weight_dtype)

        with torch.no_grad():
            latents = self._encode_latents(pixel_values)
            masked_latents = self._encode_latents(masked_image)
            mask_latents = F.interpolate(masks, size=latents.shape[-2:], mode="nearest")
            prompt_embeds = self._encode_prompt(list(batch["prompt"]), weight_dtype)

        model_input = torch.cat([latents, mask_latents, masked_latents], dim=1)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        with torch.autocast(device_type="cuda", dtype=weight_dtype, enabled=autocast_enabled):
            controlnet_output = self.controlnet(
                noisy_latents, timesteps, encoder_hidden_states=prompt_embeds,
                controlnet_cond=control_image, return_dict=False,
            )
            down_residuals, mid_residual = controlnet_output[0], controlnet_output[1]
            prediction = self.pipeline.unet(
                model_input, timesteps, encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_residuals, mid_block_additional_residual=mid_residual,
                return_dict=False,
            )[0]

        assert self.loss_fn is not None
        loss = self.loss_fn.diffusion_loss(prediction, noise, timesteps, self._alphas_cumprod, self.noise_scheduler)

        components: Dict[str, float] = {"loss": float(loss.detach().cpu())}

        # Optional auxiliary losses (decoding is expensive; keep them periodic and off by default).
        if (config.enable_perceptual_loss or config.enable_garment_clip_loss) and self.global_step % 8 == 0:
            try:
                alpha = self._alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                x0_pred = (noisy_latents - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
                decoded = self.pipeline.vae.decode((x0_pred / self.pipeline.vae.config.scaling_factor).to(weight_dtype)).sample
                decoded = (decoded / 2 + 0.5).clamp(0, 1) * 2 - 1
                auxiliary, aux_components = self.loss_fn.auxiliary_losses(
                    step=self.global_step, prediction=decoded, target=pixel_values,
                    mask=masks, garment_image=batch["cloth"].to(device, dtype=weight_dtype),
                )
                loss = loss + auxiliary
                components.update(aux_components)
            except Exception as exc:  # pragma: no cover - OOM during decode
                logger.warning("Auxiliary losses skipped this step: %s", exc)

        scaled = loss / accumulation
        if self.scaler is not None and self.scaler.is_enabled():
            self.scaler.scale(scaled).backward()
        else:
            scaled.backward()

        if (step_in_epoch + 1) % accumulation == 0:
            if self.scaler is not None and self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.controlnet.parameters() if p.requires_grad], config.max_grad_norm
            )
            if self.scaler is not None and self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        components["lr"] = float(self.lr_scheduler.get_last_lr()[0])
        return float(loss.detach().cpu()), components

    def _encode_latents(self, images):
        latents = self.pipeline.vae.encode(images).latent_dist.sample()
        return latents * self.pipeline.vae.config.scaling_factor

    def _encode_prompt(self, prompts: List[str], weight_dtype):
        import torch

        tokens = self.pipeline.tokenizer(
            prompts, padding="max_length", max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)
        with torch.no_grad():
            return self.pipeline.text_encoder(tokens)[0].to(weight_dtype)

    # ---------------------------------------------------------------------------------
    def _log_step(self, loss: float, components: Dict[str, float], total_batches: int, step_in_epoch: int, epoch: int) -> None:
        memory = memory_summary(self.device)
        self.tracker.log_many({"train/loss": loss, **{f"train/{k}": v for k, v in components.items() if k != "loss"}}, step=self.global_step, epoch=epoch + 1)
        if memory.get("peak_allocated_mb"):
            self.tracker.log("system/peak_vram_mb", memory["peak_allocated_mb"], step=self.global_step, epoch=epoch + 1)
        self.tracker.log("system/lr", components.get("lr", 0.0), step=self.global_step, epoch=epoch + 1)
        elapsed = time.time() - self.started_at
        eta = (self.total_steps - self.global_step) * (elapsed / max(1, self.global_step))
        self._set_progress(
            state="training", epoch=epoch + 1, step=self.global_step, total_steps=self.total_steps,
            loss=loss, lr=components.get("lr"), vram_mb=memory.get("peak_allocated_mb"),
            elapsed_s=round(elapsed, 1), eta_s=round(eta, 1),
            batch=f"{step_in_epoch + 1}/{total_batches}",
            message=f"epoch {epoch + 1} · step {self.global_step}/{self.total_steps} · loss {loss:.4f}",
        )
        if self.on_step:
            try:
                self.on_step(self.progress)
            except Exception:  # pragma: no cover - callback must never break training
                pass

    # =================================================================================
    # validation / checkpoints
    # =================================================================================
    def run_validation(self, epoch: int) -> Dict[str, Any]:
        """Validate and generate sample grids."""
        import torch

        if self.val_loader is None or self.validator is None:
            return {}
        self._set_progress(state="validating", epoch=epoch, message="Running validation…")
        try:
            result = self.validator.run(
                pipeline=self.pipeline, controlnet=self.controlnet, val_loader=self.val_loader,
                noise_scheduler=self.noise_scheduler, device=self.device, weight_dtype=self.weight_dtype,
                step=self.global_step, epoch=epoch, resolution=self.config.resolution,
                num_batches=self.config.validation_batches, num_inference_images=self.config.validation_images,
                inference_steps=self.config.num_inference_steps, guidance_scale=self.config.guidance_scale,
            )
        except GPUMemoryError:
            raise
        except RuntimeError as exc:
            empty_cuda_cache()
            if "out of memory" in str(exc).lower():
                self.warnings.append("Validation ran out of GPU memory; reduce validation_batches/resolution.")
                return {}
            raise
        self.tracker.log_many(
            {f"val/{k}": v for k, v in result.items() if isinstance(v, (int, float)) and k not in {"step", "epoch", "batches"}},
            step=self.global_step, epoch=epoch,
        )
        for grid_path in result.get("grid_paths", [])[:2]:
            try:
                from backend.utils.image_utils import load_image

                self.tracker.log_image("val/sample", load_image(grid_path), step=self.global_step)
            except Exception:  # pragma: no cover
                pass
        self._set_progress(state="validating", epoch=epoch, val_loss=result.get("loss"), message=f"val loss {result.get('loss', float('nan')):.4f}")
        return result

    def save_epoch(self, epoch: int, metrics: Dict[str, Any]) -> Path:
        """Persist the adapter + a resumable training state for this epoch."""
        import torch

        directory = self.checkpoints.epoch_dir(epoch)
        adapter_dir = directory / "adapter"
        card = self._model_card(epoch, metrics)
        save_trained_adapter(adapter_dir, self.controlnet, card, self.model_config)
        self.checkpoints.save_epoch_metrics(epoch, metrics)

        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "best_metric": self.best_metric,
            "history": self.history,
            "config": self.config.to_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng": np.random.get_state(),
        }
        self.checkpoints.save_training_state(state, epoch)
        self.checkpoints.prune_epochs(keep_top_k=self.config.save_top_k)
        self.current_adapter_dir = adapter_dir
        logger.info("Epoch %d checkpointed (%s)", epoch, adapter_dir)
        return adapter_dir

    def load_training_state(self, state: Dict[str, Any]) -> None:
        """Resume from a saved training state."""
        import torch

        try:
            if state.get("optimizer") and self.optimizer is not None:
                self.optimizer.load_state_dict(state["optimizer"])
            if state.get("scheduler") and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(state["scheduler"])
            if state.get("scaler") and self.scaler is not None:
                self.scaler.load_state_dict(state["scaler"])
            self.start_epoch = int(state.get("epoch", 0))
            self.global_step = int(state.get("global_step", 0))
            self.best_metric = float(state.get("best_metric", float("inf")))
            self.history = list(state.get("history", []))
            if state.get("torch_rng") is not None:
                torch.set_rng_state(state["torch_rng"])
            if state.get("cuda_rng") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            if state.get("numpy_rng") is not None:
                np.random.set_state(state["numpy_rng"])
            logger.info("Resumed from epoch %d (step %d)", self.start_epoch, self.global_step)
            self.warnings.append(f"Resumed training from epoch {self.start_epoch}.")
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not resume from training state (%s); starting fresh.", exc)

    def _maybe_export_best(self, metric: float, epoch: int, force: bool = False) -> None:
        """Copy the epoch adapter into best_model when the metric improves."""
        improved = metric < self.best_metric - 1e-6
        if not improved and not force:
            return
        if improved:
            self.best_metric = metric
        adapter_dir = getattr(self, "current_adapter_dir", None)
        if adapter_dir is None or not Path(adapter_dir).exists():
            logger.warning("No adapter available to export for epoch %s.", epoch)
            return
        try:
            self.checkpoints.export_model(adapter_dir, self.checkpoints.best_dir, mirror_latest=True)
            card = self._model_card(epoch, {"val_loss": metric})
            card.best_metric = float(metric) if metric == metric else None
            card.save(self.checkpoints.best_dir)
            logger.info("New best model exported (val_loss=%.5f) at epoch %d", metric, epoch)
        except Exception as exc:  # pragma: no cover
            logger.error("Could not export best model: %s", exc)

    def _should_stop_early(self, metric: float) -> bool:
        patience = int(self.config.early_stopping_patience)
        if patience <= 0:
            return False
        if metric < self.best_metric - 1e-6:
            self.epochs_without_improvement = 0
            return False
        self.epochs_without_improvement += 1
        if self.epochs_without_improvement >= patience:
            logger.info("Early stopping: no validation improvement for %d epochs.", patience)
            return True
        return False

    def _model_card(self, epoch: int, metrics: Dict[str, Any]) -> ModelCard:
        dataset_summary = getattr(self.datasets.get("train"), "summary", lambda: {})() if hasattr(self, "datasets") else {}
        return ModelCard(
            name="vestiai-vton",
            version="1.0.0",
            architecture="controlnet-inpaint-sd15",
            base_model=self.config.base_model,
            training_mode=self.config.mode,
            dataset=str(self.config.dataset_root),
            dataset_size=int(dataset_summary.get("samples", 0) or 0),
            resolution=self.config.resolution,
            epochs=epoch,
            global_step=self.global_step,
            best_metric=float(metrics.get("val_loss", metrics.get("train_loss", 0.0)) or 0.0),
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            notes=(
                f"Trained with VestiAI ({self.config.mode}). Trainable: ControlNet"
                f"{' + UNet LoRA' if self.config.train_unet_lora else ''}. "
                f"Precision={self.config.mixed_precision}, resolution={self.config.resolution}."
            ),
            metrics_history=self.history[-20:],
        )

    # ---------------------------------------------------------------------------------
    def stop(self) -> None:
        """Request a graceful stop (checked between steps/epochs)."""
        self.stopped = True
        self._set_progress(state="stopping", message="Stop requested — finishing the current step…")

    def _set_progress(self, **updates: Any) -> None:
        self.progress.update(updates)
        self.progress["warnings"] = self.warnings[-8:]
        self.progress["updated_at"] = time.time()
        try:
            payload = json.loads(json.dumps(self.progress, default=str))
            tmp = self.status_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, self.status_path)
        except Exception:  # pragma: no cover
            pass

    def summary(self) -> Dict[str, Any]:
        """Final report written into the run summary JSON."""
        final = self.history[-1] if self.history else {}
        return {
            "mode": self.config.mode,
            "device": self.device,
            "mixed_precision": self.config.mixed_precision,
            "resolution": self.config.resolution,
            "epochs_completed": len(self.history),
            "global_step": self.global_step,
            "final_train_loss": final.get("train_loss"),
            "best_val_loss": self.best_metric if math.isfinite(self.best_metric) else None,
            "history": self.history,
            "warnings": self.warnings,
            "duration_s": round(time.time() - self.started_at, 1),
            "checkpoints": self.checkpoints.status(),
            "best_model_path": str(self.checkpoints.best_dir),
        }
