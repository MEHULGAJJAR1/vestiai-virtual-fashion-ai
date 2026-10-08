#!/usr/bin/env python
"""VestiAI — loss / metric diagnostics.

Two useful jobs in one command:

* **metrics** (always available): PSNR, SSIM and masked-L1 between any two images — the same
  functions the validation, evaluation and comparison code uses.
* **batch** (needs the ML extra): builds a real dataset batch, runs the exact training loss
  stack (Min-SNR weighted ε-MSE + optional LPIPS + optional CLIP garment consistency) and
  prints every component, so you can see what dominates the gradient before you spend GPU
  hours on a run.

    python scripts/losses.py metrics --a results/a.png --b results/b.png
    python scripts/losses.py metrics --a results/a.png --b results/b.png --mask mask.png
    python scripts/losses.py batch --dataset datasets/samples --resolution 256 --perceptual
    python scripts/losses.py batch --dataset datasets/samples --garment-clip     # needs transformers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import (  # noqa: E402
    add_common_io_args, banner, fail, kv_table, load_settings, print_json, resolve_dataset,
    require_torch, setup_logging,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inspect VestiAI's loss components and image-quality metrics.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    metrics = sub.add_parser("metrics", help="PSNR / SSIM / masked-L1 between two images")
    metrics.add_argument("--a", required=True, help="reference image")
    metrics.add_argument("--b", required=True, help="prediction image")
    metrics.add_argument("--mask", default=None, help="optional mask for the masked-L1 metric")
    metrics.add_argument("--json", action="store_true")

    batch = sub.add_parser("batch", help="run the training loss stack on one real batch")
    batch.add_argument("--dataset", default="datasets/samples")
    batch.add_argument("--resolution", type=int, default=256)
    batch.add_argument("--batch", type=int, default=1)
    batch.add_argument("--mode", default="QUICK_DEMO", choices=["QUICK_DEMO", "FINE_TUNE", "FULL_TRAINING"])
    batch.add_argument("--perceptual", action="store_true", help="enable the LPIPS perceptual term")
    batch.add_argument("--garment-clip", action="store_true", help="enable the CLIP garment-consistency term")
    batch.add_argument("--snr-gamma", type=float, default=5.0)
    batch.add_argument("--json", action="store_true")

    add_common_io_args(p)
    return p


def metrics_command(args) -> int:
    import numpy as np
    from PIL import Image

    from backend.training.losses import masked_l1, psnr, ssim

    for path in (args.a, args.b):
        if not Path(path).exists():
            fail(f"image not found: {path}")

    a = np.asarray(Image.open(args.a).convert("RGB"))
    b = np.asarray(Image.open(args.b).convert("RGB"))
    if a.shape != b.shape:
        b_image = Image.open(args.b).convert("RGB").resize((a.shape[1], a.shape[0]))
        b = np.asarray(b_image)
        print(f"  (resized {Path(args.b).name} to {a.shape[1]}×{a.shape[0]} for the comparison)")

    result = {
        "psnr": round(float(psnr(a, b)), 3),
        "ssim": round(float(ssim(a, b)), 4),
        "mean_abs_diff": round(float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()), 3),
        "a": str(args.a),
        "b": str(args.b),
    }
    if args.mask:
        mask = np.asarray(Image.open(args.mask).convert("L"))
        result["masked_l1"] = round(float(masked_l1(b, a, mask)), 4)
        result["mask_coverage"] = round(float((mask > 127).mean()), 4)

    banner("VestiAI · image metrics")
    kv_table(result)
    print("\n  interpretation:")
    print("    PSNR > 30 dB and SSIM > 0.9 → visually close (typical for a good garment transfer)")
    print("    PSNR < 18 dB                → the two images differ substantially (check alignment)")
    if args.mask:
        print("    masked L1 is computed inside the garment mask only — that is the number that")
        print("    actually reflects garment preservation in the try-on region.")
    if args.json:
        print_json(result)
    print()
    return 0


def batch_command(args) -> int:
    """Run the real loss stack on a real batch and print every component."""
    torch = require_torch("the loss-stack diagnostic")
    settings = load_settings(args.config)
    root = resolve_dataset(settings, args.dataset)
    if root is None:
        fail(f"dataset '{args.dataset}' not found")

    from diffusers import DDPMScheduler

    from backend.datasets.dataloader import build_dataloaders
    from backend.training.losses import CompositeLoss, LossWeights
    from backend.training.trainer import TrainingConfig

    device = settings.resolve_device()
    banner("VestiAI · training loss stack", f"{root}  ·  device={device}")

    config = TrainingConfig.for_mode(args.mode, resolution=args.resolution, batch_size=args.batch)
    config.enable_perceptual_loss = args.perceptual
    config.enable_garment_clip_loss = args.garment_clip
    config.snr_gamma = args.snr_gamma

    loaders = build_dataloaders(
        str(root), resolution=args.resolution, mode=args.mode,
        batch_size=args.batch, num_workers=0, device=str(device), augment=False,
        max_train_samples=config.max_train_samples,
    )
    batch = next(iter(loaders["train"]))

    # The real scheduler config, built offline (same betas as the SD family) so this
    # diagnostic never needs a model download.
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="scaled_linear",
        beta_start=0.00085, beta_end=0.012, clip_sample=False,
    )

    weights = LossWeights(
        perceptual=config.perceptual_weight if config.enable_perceptual_loss else 0.0,
        garment_clip=config.garment_clip_weight if config.enable_garment_clip_loss else 0.0,
        snr_gamma=config.snr_gamma,
    )
    composite = CompositeLoss(
        weights=weights,
        device=device,
        enable_perceptual=config.enable_perceptual_loss,
        enable_garment_clip=config.enable_garment_clip_loss,
    )

    pixel_values = batch["pixel_values"].to(device)
    control_image = batch["control_image"].to(device) if "control_image" in batch else None
    mask = batch["inpaint_mask"].to(device) if "inpaint_mask" in batch else None
    # 8× downsampled latent stand-in (the real trainer encodes with the frozen VAE).
    latent = torch.nn.functional.interpolate(pixel_values, scale_factor=0.125, mode="bilinear")
    noise = torch.randn_like(latent)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latent.shape[0],), device=device).long()
    noisy = scheduler.add_noise(latent, noise, timesteps)
    alphas = scheduler.alphas_cumprod.to(device)[timesteps]

    diffusion = composite.diffusion_loss(noise, noise, timesteps, alphas)   # perfect prediction → ~0
    noisy_prediction = composite.diffusion_loss(noisy, noise, timesteps, alphas)
    auxiliary_total, auxiliary = composite.auxiliary_losses(
        step=0, prediction=pixel_values, target=pixel_values, mask=mask,
        garment_image=control_image,
    )
    status = composite.status()

    banner("loss components on a real batch")
    kv_table({
        "diffusion (perfect prediction)": round(float(diffusion.detach().cpu()), 8),
        "diffusion (noisy prediction)": round(float(noisy_prediction.detach().cpu()), 6),
        "auxiliary total": round(float(auxiliary_total.detach().cpu()), 6) if hasattr(auxiliary_total, "detach") else auxiliary_total,
    })
    for name, value in auxiliary.items():
        kv_table({f"  {name}": round(float(value), 6)})
    print()
    kv_table({
        "batch": {key: tuple(value.shape) for key, value in batch.items() if hasattr(value, "shape")},
        "device": device,
        "torch": torch.__version__,
        "cuda": bool(getattr(torch, "cuda", None) and torch.cuda.is_available()),
        "perceptual": status["perceptual"],
        "garment clip": status["garment_clip"],
        "snr gamma": weights.snr_gamma,
    })
    print("\n  notes:")
    print("    · the first number must be 0 (a perfect prediction of the noise) — that proves the")
    print("      ε-MSE + Min-SNR weighting maths is wired correctly end to end.")
    print("    · auxiliary losses are computed every N steps by design (see LossWeights), which is")
    print("      why they can be 0 on a step where they are not scheduled.")
    if not status["perceptual"]["available"]:
        print(f"    · perceptual term inactive: {status['perceptual']['reason']}")
    if args.json:
        print_json({
            "diffusion_perfect": float(diffusion.detach().cpu()),
            "diffusion_noisy": float(noisy_prediction.detach().cpu()),
            "auxiliary": auxiliary,
            "status": status,
        })
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(getattr(args, "log_level", "INFO"))
    if args.command == "metrics":
        return metrics_command(args)
    return batch_command(args)


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
