"""Training losses for the VTON diffusion adapter.

Implemented losses (all real, all used by the trainer):

===============  ============================================================
Loss             Purpose
===============  ============================================================
``noise_mse``    Standard ε-prediction objective (the backbone of fine-tuning).
``snr_weighted`` Min-SNR-γ weighting (Hang et al., 2023) which balances the noise
                 levels during training and speeds up convergence noticeably.
``perceptual``   VGG/LPIPS feature distance between the decoded prediction and the
                 ground-truth region — sharpens garment texture (optional extra).
``garment_clip`` CLIP image-embedding cosine distance between the *generated region*
                 and the *garment input* — directly optimises "did the shirt keep
                 its print?" (optional extra).
``masked_*``     All image-space losses are computed inside the garment mask so the
                 person's identity and background are never penalised/changed.
===============  ============================================================

The module imports torch lazily so the API server keeps working without the ML stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------------------
# Core diffusion objective
# --------------------------------------------------------------------------------------
def snr_weighting(alphas_cumprod, timesteps, gamma: float = 5.0):
    """Min-SNR-γ loss weight per sample (returns a detached weight vector)."""
    import torch

    snr = alphas_cumprod[timesteps] / (1.0 - alphas_cumprod[timesteps])
    weight = torch.clamp(snr, max=gamma) / (snr + 1.0)
    return weight


def noise_mse(
    predicted: "torch.Tensor",
    target: "torch.Tensor",
    weighting=None,
    reduce: str = "mean",
):
    """Weighted MSE between predicted and target noise."""
    import torch

    differences = (predicted.float() - target.float()) ** 2
    differences = differences.mean(dim=tuple(range(1, differences.ndim)))
    if weighting is not None:
        differences = differences * weighting.to(differences.device).float()
    if reduce == "mean":
        return differences.mean()
    if reduce == "sum":
        return differences.sum()
    return differences


def masked_mse(predicted, target, mask, eps: float = 1e-6):
    """MSE restricted to ``mask > 0`` (used for decoded-image supervision)."""
    weight = (mask > 0.5).float()
    if weight.sum() < eps:
        return torch_zero_like(predicted)
    diff = (predicted.float() - target.float()) ** 2
    diff = diff * weight
    return diff.sum() / (weight.sum() * diff.shape[1] + eps)


def torch_zero_like(tensor):
    import torch

    return torch.zeros((), device=tensor.device, dtype=tensor.dtype)


# --------------------------------------------------------------------------------------
# Optional perceptual + CLIP losses
# --------------------------------------------------------------------------------------
class PerceptualLoss:
    """LPIPS if available, otherwise a VGG16 feature-matching loss.

    Both are optional: if neither can be built (no weights downloaded / torchvision absent)
    the loss reports ``None`` and the trainer simply skips it, logging why once.
    """

    def __init__(self, device: str = "cpu", use_lpips: bool = True, net: str = "vgg") -> None:
        self.device = device
        self.net = None
        self.kind = "disabled"
        self.reason: Optional[str] = None
        if use_lpips:
            try:  # pragma: no cover - optional extra
                import lpips

                self.net = lpips.LPIPS(net=net, verbose=False).to(device).eval()
                for parameter in self.net.parameters():
                    parameter.requires_grad_(False)
                self.kind = f"lpips-{net}"
                return
            except Exception as exc:
                self.reason = f"lpips unavailable ({exc})"
        try:  # pragma: no cover - optional extra
            import torch
            import torchvision

            weights = torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES
            model = torchvision.models.vgg16(weights=weights).features[:16].to(device).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.net = model
            self.kind = "vgg16-features"
        except Exception as exc:
            self.reason = (self.reason + "; " if self.reason else "") + f"vgg16 unavailable ({exc})"
            self.net = None
            self.kind = "disabled"

    def available(self) -> bool:
        return self.net is not None

    def __call__(self, prediction, target, mask=None):
        if self.net is None:
            return None
        import torch
        import torch.nn.functional as F

        mean = torch.tensor([0.485, 0.456, 0.406], device=prediction.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=prediction.device).view(1, 3, 1, 1)
        pred_norm = (prediction - mean) / std
        target_norm = (target - mean) / std
        if self.kind.startswith("lpips"):
            value = self.net(pred_norm * 2 - 1, target_norm * 2 - 1).mean()
        else:
            value = F.l1_loss(self.net(pred_norm), self.net(target_norm))
        if mask is not None and float(mask.mean()) > 0 and value is not None:
            weight = mask.float().mean().clamp(min=0.02)
            value = value * weight
        return value


class GarmentConsistencyLoss:
    """CLIP image-image cosine distance between generated region and the garment input.

    This is the loss that pushes the model to preserve prints, logos and fabric patterns.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu") -> None:
        self.model = None
        self.processor = None
        self.device = device
        self.reason: Optional[str] = None
        try:  # pragma: no cover - optional extra
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

            self.processor = CLIPImageProcessor.from_pretrained(model_name)
            self.model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(device).eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        except Exception as exc:
            self.reason = f"CLIP consistency loss disabled ({exc.__class__.__name__}: {exc})"
            logger.info(self.reason)

    def available(self) -> bool:
        return self.model is not None

    def __call__(self, generated_region, garment_image):
        """Cosine distance in CLIP image-embedding space (0 = identical look)."""
        if self.model is None:
            return None
        import torch

        with torch.no_grad():
            gen = self.processor(images=_to_pil_batch(generated_region), return_tensors="pt").pixel_values.to(self.device, dtype=self.model.dtype)
            cloth = self.processor(images=_to_pil_batch(garment_image), return_tensors="pt").pixel_values.to(self.device, dtype=self.model.dtype)
            gen_embed = self.model(gen).image_embeds
            cloth_embed = self.model(cloth).image_embeds
        gen_embed = gen_embed / gen_embed.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        cloth_embed = cloth_embed / cloth_embed.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return (1.0 - (gen_embed * cloth_embed).sum(dim=-1)).mean()


def _to_pil_batch(tensor):
    """Convert a ``[-1, 1]`` (B, 3, H, W) tensor into a list of PIL images."""
    from PIL import Image

    images = []
    array = tensor.detach().float().cpu().numpy()
    for item in array:
        item = np.clip((item.transpose(1, 2, 0) + 1.0) / 2.0, 0, 1)
        images.append(Image.fromarray((item * 255).astype(np.uint8)))
    return images


@dataclass
class LossWeights:
    """Weights + configuration for the composite loss."""

    diffusion: float = 1.0
    perceptual: float = 0.0        # enabled by config (adds a VAE decode per step)
    garment_clip: float = 0.0      # enabled by config
    perceptual_every_n_steps: int = 4
    clip_every_n_steps: int = 8
    snr_gamma: float = 5.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "diffusion": self.diffusion, "perceptual": self.perceptual, "garment_clip": self.garment_clip,
            "perceptual_every_n_steps": self.perceptual_every_n_steps,
            "clip_every_n_steps": self.clip_every_n_steps, "snr_gamma": self.snr_gamma,
        }


class CompositeLoss:
    """Aggregates the diffusion objective with the optional auxiliary losses."""

    def __init__(
        self,
        weights: Optional[LossWeights] = None,
        device: str = "cpu",
        enable_perceptual: bool = False,
        enable_garment_clip: bool = False,
        perceptual_net: str = "vgg",
    ) -> None:
        self.weights = weights or LossWeights()
        self.device = device
        self.perceptual = PerceptualLoss(device=device, use_lpips=enable_perceptual, net=perceptual_net) if enable_perceptual else PerceptualLoss.__new__(PerceptualLoss)
        if not enable_perceptual:
            self.perceptual.net = None
            self.perceptual.kind = "disabled"
            self.perceptual.reason = "disabled by config"
            self.perceptual.device = device
        self.garment = GarmentConsistencyLoss(device=device) if enable_garment_clip else None
        self.last_components: Dict[str, float] = {}

    def status(self) -> Dict[str, Any]:
        return {
            "weights": self.weights.to_dict(),
            "perceptual": {"kind": self.perceptual.kind, "available": self.perceptual.available(), "reason": self.perceptual.reason},
            "garment_clip": {"available": bool(self.garment and self.garment.available()), "reason": getattr(self.garment, "reason", "disabled by config")},
        }

    # ---------------------------------------------------------------------------------
    def diffusion_loss(self, predicted, target, timesteps, alphas_cumprod, noise_scheduler=None):
        """SNR-weighted ε-prediction MSE."""
        weighting = None
        if self.weights.snr_gamma and self.weights.snr_gamma > 0 and alphas_cumprod is not None:
            weighting = snr_weighting(alphas_cumprod, timesteps, gamma=self.weights.snr_gamma)
        _ = noise_scheduler
        return noise_mse(predicted, target, weighting=weighting)

    def auxiliary_losses(self, step: int, prediction=None, target=None, mask=None, garment_image=None, vae=None) -> Tuple[Any, Dict[str, float]]:
        """Compute the periodic auxiliary terms. Returns ``(total, components)``."""
        import torch

        components: Dict[str, float] = {}
        total = torch.zeros((), device=self.device)

        if self.perceptual.available() and self.weights.perceptual > 0 and prediction is not None:
            if step % max(1, self.weights.perceptual_every_n_steps) == 0:
                value = self.perceptual(prediction, target, mask)
                if value is not None:
                    total = total + value * self.weights.perceptual
                    components["perceptual"] = float(value.detach().cpu())

        if self.garment is not None and self.garment.available() and self.weights.garment_clip > 0 and prediction is not None:
            if step % max(1, self.weights.clip_every_n_steps) == 0:
                value = self.garment(prediction, garment_image)
                if value is not None:
                    total = total + value * self.weights.garment_clip
                    components["garment_clip"] = float(value.detach().cpu())

        _ = vae
        self.last_components = components
        return total, components


def psnr(prediction: np.ndarray, target: np.ndarray, max_value: float = 255.0) -> float:
    """Peak signal-to-noise ratio (higher is better) for evaluation grids."""
    prediction = np.asarray(prediction, np.float32)
    target = np.asarray(target, np.float32)
    mse = float(np.mean((prediction - target) ** 2))
    if mse <= 1e-10:
        return 99.0
    return float(10.0 * np.log10((max_value ** 2) / mse))


def ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    """Structural similarity, implemented here to avoid an scikit-image dependency."""
    prediction = np.asarray(prediction, np.float32)
    target = np.asarray(target, np.float32)
    if prediction.ndim == 3:
        prediction = prediction.mean(axis=-1)
        target = target.mean(axis=-1)
    mu_x = prediction.mean()
    mu_y = target.mean()
    sigma_x = prediction.var()
    sigma_y = target.var()
    sigma_xy = float(((prediction - mu_x) * (target - mu_y)).mean())
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return float(numerator / denominator) if denominator else 0.0


def masked_l1(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """L1 error inside a mask — the classic VTON "preservation" metric."""
    weight = (np.asarray(mask) > 96)
    if weight.sum() == 0:
        return float("nan")
    diff = np.abs(np.asarray(prediction, np.float32) - np.asarray(target, np.float32))
    return float(diff[weight].mean())
