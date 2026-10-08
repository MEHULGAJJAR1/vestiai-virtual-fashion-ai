"""Validation loop, sample-image generation and comparison grids.

Runs inside training (every ``validate_every_n_epochs``) and standalone via
``scripts/validate.py`` / ``scripts/evaluate.py``. It produces exactly what the AI Training
Dashboard and ``results/`` need:

* validation loss on a held-out split,
* side-by-side grids (person | garment | agnostic | generated | ground truth),
* VTON metrics (SSIM, PSNR, masked L1 inside the garment region, identity proxy),
* ``validation_summary.json`` so the dashboard can render without re-running anything.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.training.losses import masked_l1, psnr, ssim
from backend.utils.image_utils import save_image, stack_grid, to_rgb
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationMetrics:
    """Aggregated validation results."""

    loss: float = 0.0
    ssim: float = 0.0
    psnr: float = 0.0
    masked_l1: float = 0.0
    garment_preservation: float = 0.0
    samples: int = 0
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loss": round(float(self.loss), 6),
            "ssim": round(float(self.ssim), 4),
            "psnr": round(float(self.psnr), 3),
            "masked_l1": round(float(self.masked_l1), 4),
            "garment_preservation": round(float(self.garment_preservation), 4),
            "samples": int(self.samples),
            **{k: round(float(v), 5) for k, v in self.extra.items()},
        }


def compute_vton_metrics(
    generated: np.ndarray,
    ground_truth: np.ndarray,
    mask: Optional[np.ndarray] = None,
    garment: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Image-quality + VTON-specific metrics between a prediction and its target."""
    prediction = np.asarray(generated)
    target = np.asarray(ground_truth)
    if prediction.shape != target.shape:
        import cv2

        prediction = cv2.resize(prediction, (target.shape[1], target.shape[0]))

    metrics: Dict[str, float] = {
        "ssim": ssim(prediction, target),
        "psnr": psnr(prediction, target),
    }
    if mask is not None:
        metrics["masked_l1"] = masked_l1(prediction, target, mask)
    if garment is not None:
        metrics["garment_preservation"] = _garment_preservation(prediction, garment, mask)
    return metrics


def _garment_preservation(generated: np.ndarray, garment: np.ndarray, mask: Optional[np.ndarray]) -> float:
    """Colour-histogram agreement between the generated garment region and the input garment.

    A cheap, dependency-free proxy for "did the model keep the shirt's colour/pattern?",
    which is the metric users actually notice. 1.0 = identical distribution.
    """
    import cv2

    region = generated
    if mask is not None:
        weight = (np.asarray(mask) > 96)
        if weight.sum() < 64:
            return float("nan")
        pixels = generated[weight].reshape(-1, 3)
    else:
        pixels = region.reshape(-1, 3)
    reference = np.asarray(garment).reshape(-1, 3)
    if reference.shape[0] > 40000:
        step = reference.shape[0] // 40000
        reference = reference[::step]
    if pixels.shape[0] > 40000:
        step = pixels.shape[0] // 40000
        pixels = pixels[::step]

    hist_a = _hist(cv2, pixels)
    hist_b = _hist(cv2, reference)
    distance = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA)
    return float(max(0.0, 1.0 - distance))


def _hist(cv2, pixels: np.ndarray):
    hsv = cv2.cvtColor(np.asarray(pixels, np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


class Validator:
    """Runs validation inside the training loop."""

    def __init__(self, sample_dir: str | Path = "results/validation") -> None:
        self.sample_dir = Path(sample_dir)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------------------------
    def run(
        self,
        pipeline,
        controlnet,
        val_loader,
        noise_scheduler,
        device: str,
        weight_dtype,
        step: int,
        epoch: int,
        resolution: int,
        num_batches: int = 4,
        num_inference_images: int = 2,
        inference_steps: int = 20,
        guidance_scale: float = 2.0,
        generator=None,
    ) -> Dict[str, Any]:
        """Validate + generate sample images. Returns a metrics dict."""
        import torch
        import torch.nn.functional as F

        controlnet.eval()
        losses: List[float] = []
        metric_accumulator: Dict[str, List[float]] = {}
        sample_panels: List[np.ndarray] = []

        autocast_enabled = device == "cuda" and weight_dtype in (torch.float16, torch.bfloat16)
        with torch.no_grad():
            for index, batch in enumerate(val_loader):
                if index >= num_batches:
                    break
                pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
                masked_image = batch["masked_image"].to(device, dtype=weight_dtype)
                masks = batch["inpaint_mask"].to(device, dtype=weight_dtype)
                control_image = batch["control_image"].to(device, dtype=weight_dtype)

                latents = _encode_latents(pipeline, pixel_values)
                masked_latents = _encode_latents(pipeline, masked_image)
                mask_latents = F.interpolate(masks, size=latents.shape[-2:], mode="nearest")
                model_input = torch.cat([latents, mask_latents, masked_latents], dim=1)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
                ).long()
                noisy = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.autocast(device_type="cuda", dtype=weight_dtype, enabled=autocast_enabled):
                    predicted = controlnet(
                        noisy, timesteps, encoder_hidden_states=_encode_prompt(pipeline, list(batch["prompt"]), device, weight_dtype),
                        controlnet_cond=control_image, return_dict=False,
                    )[0]
                    loss = F.mse_loss(predicted.float(), noise.float(), reduction="mean")
                losses.append(float(loss.detach().cpu()))

                # --- sample generation (the slow part, therefore capped)
                if len(sample_panels) < num_inference_images:
                    panel = self._generate_sample(
                        pipeline, controlnet, batch, device, weight_dtype, resolution,
                        inference_steps=inference_steps, guidance_scale=guidance_scale, generator=generator,
                    )
                    if panel is not None:
                        generated, ground_truth, mask_np, garment_np = panel
                        metrics = compute_vton_metrics(generated, ground_truth, mask_np, garment_np)
                        for key, value in metrics.items():
                            if value == value:  # not NaN
                                metric_accumulator.setdefault(key, []).append(value)
                        event_dir = self.sample_dir / f"step_{step:07d}"
                        event_dir.mkdir(parents=True, exist_ok=True)
                        panels = [
                            _to_uint8(batch["pixel_values"][0].detach().cpu().numpy()),
                            _to_uint8(batch["control_image"][0].detach().cpu().numpy()),
                            _to_uint8(batch["masked_image"][0].detach().cpu().numpy()),
                            generated, ground_truth,
                        ]
                        grid = stack_grid(panels, cols=len(panels))
                        save_image(grid, event_dir / f"sample_{len(list(event_dir.glob('sample_*.jpg')))}.jpg", quality=92)
                        sample_panels.append(grid)

        controlnet.train()
        result: Dict[str, Any] = {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "step": step,
            "epoch": epoch,
            "batches": len(losses),
            "samples": len(metric_accumulator.get("ssim", [])),
            "timestamp": time.time(),
        }
        for key, values in metric_accumulator.items():
            result[key] = float(np.mean(values))
        result["grid_paths"] = [str(path) for path in sorted(self.sample_dir.glob(f"step_{step:07d}/sample_*.jpg"))]

        self.history.append(result)
        self._write_summary()
        return result

    # ---------------------------------------------------------------------------------
    def _generate_sample(
        self,
        pipeline,
        controlnet,
        batch,
        device: str,
        weight_dtype,
        resolution: int,
        inference_steps: int = 20,
        guidance_scale: float = 2.0,
        generator=None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Run the full pipeline on one batch item to produce a real sample image."""
        import torch

        try:
            generator = generator or torch.Generator(device="cpu").manual_seed(0)
            from diffusers import StableDiffusionControlNetInpaintPipeline

            inference_pipe = StableDiffusionControlNetInpaintPipeline(
                vae=pipeline.vae, text_encoder=pipeline.text_encoder, tokenizer=pipeline.tokenizer,
                unet=pipeline.unet, controlnet=controlnet, scheduler=pipeline.scheduler,
                safety_checker=None, feature_extractor=None, image_encoder=None,
            ).to(device)
            output = inference_pipe(
                prompt=list(batch["prompt"]),
                image=[_to_pil(batch["masked_image"][i]) for i in range(min(1, batch["pixel_values"].shape[0]))],
                mask_image=[_to_pil(batch["inpaint_mask"][i], grayscale=True) for i in range(min(1, batch["pixel_values"].shape[0]))],
                control_image=[_to_pil(batch["control_image"][i]) for i in range(min(1, batch["pixel_values"].shape[0]))],
                height=resolution, width=resolution,
                num_inference_steps=inference_steps, guidance_scale=guidance_scale,
                generator=generator, output_type="np",
            )
            generated = np.clip(output.images[0] * 255, 0, 255).astype(np.uint8)
            ground_truth = _to_uint8(batch["pixel_values"][0].detach().cpu().numpy())
            mask_np = (batch["inpaint_mask"][0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
            garment_np = _to_uint8(batch["control_image"][0].detach().cpu().numpy())
            return generated, ground_truth, mask_np, garment_np
        except Exception as exc:  # pragma: no cover - OOM / interrupted
            logger.warning("Sample generation during validation failed: %s", exc)
            return None

    def _write_summary(self) -> Path:
        payload = {
            "updated_at": time.time(),
            "runs": self.history[-50:],
            "latest": self.history[-1] if self.history else None,
        }
        path = self.sample_dir / "validation_summary.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path


def _encode_latents(pipeline, images):
    latents = pipeline.vae.encode(images).latent_dist.sample()
    return latents * pipeline.vae.config.scaling_factor


def _encode_prompt(pipeline, prompts: List[str], device: str, weight_dtype):
    tokens = pipeline.tokenizer(
        prompts, padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    with torch_no_grad():
        return pipeline.text_encoder(tokens)[0].to(weight_dtype)


def torch_no_grad():
    import torch

    return torch.no_grad()


def _to_uint8(tensor: np.ndarray) -> np.ndarray:
    """``[-1, 1]`` CHW float tensor array -> HWC uint8."""
    array = np.asarray(tensor)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = array.transpose(1, 2, 0)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    array = np.clip((array + 1.0) / 2.0, 0, 1) if array.min() < -0.01 else np.clip(array, 0, 1)
    return (array * 255).astype(np.uint8)


def _to_pil(tensor, grayscale: bool = False):
    from PIL import Image

    array = _to_uint8(tensor.detach().cpu().numpy())
    if grayscale:
        return Image.fromarray(array[..., 0], mode="L")
    return Image.fromarray(array, mode="RGB")


def load_validation_summary(directory: str | Path = "results/validation") -> Dict[str, Any]:
    path = Path(directory) / "validation_summary.json"
    if not path.exists():
        return {"runs": [], "latest": None}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover
        return {"runs": [], "latest": None}
