#!/usr/bin/env python
"""VestiAI — run validation on a checkpoint.

Generates try-on samples for the validation split and writes
``results/validation/step_<N>/sample_*.jpg`` grids (person | garment | agnostic | generated |
ground-truth) plus the metrics the training dashboard plots: SSIM, PSNR and masked-L1.

Use it after a training run, or mid-run to sanity check a checkpoint:

    python scripts/validate.py --checkpoint checkpoints/best_model
    python scripts/validate.py --checkpoint checkpoints/epochs/epoch_0004 --batches 8 --images 4
    python scripts/validate.py --checkpoint checkpoints/best_model --dataset datasets/samples --split test

Requires the ML extra (PyTorch + Diffusers). Without it the script exits with instructions
instead of producing fake images.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import (  # noqa: E402
    PROJECT_ROOT, add_common_io_args, banner, kv_table, load_settings, print_json,
    require_checkpoint, require_torch, resolve_dataset, setup_logging,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate validation grids + metrics for a trained checkpoint.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", default=None, help="checkpoint folder (default: auto — best, latest, newest epoch)")
    p.add_argument("--dataset", default=None, help="dataset name/path (default: the one recorded in the checkpoint)")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--batches", type=int, default=2, help="validation batches to run")
    p.add_argument("--images", type=int, default=2, help="images per batch to render")
    p.add_argument("--steps", type=int, default=20, help="diffusion steps used for each sample")
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="results/validation", help="where the grids are written")
    p.add_argument("--json", action="store_true")
    add_common_io_args(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    torch = require_torch("validation")
    settings = load_settings(args.config)

    checkpoint = require_checkpoint(args.checkpoint, "Validation")
    banner("VestiAI · validation", str(checkpoint))

    from backend.datasets.dataloader import build_dataloaders
    from backend.models.vton_controlnet import VTONModelConfig, build_components, describe_checkpoint
    from backend.training.validate import Validator

    info = describe_checkpoint(checkpoint)
    kv_table({
        "checkpoint": info.get("path"),
        "valid": info.get("valid"),
        "weights": info.get("weights"),
        "size": f"{info.get('size_mb', 0)} MB",
        "model card": (info.get("model_card") or {}).get("category", "—"),
    })
    card = info.get("model_card") or {}
    dataset_requested = args.dataset or card.get("dataset") or "datasets/samples"
    dataset = resolve_dataset(settings, dataset_requested)
    if dataset is None:
        print(f"\n✗ dataset '{dataset_requested}' not found — pass --dataset explicitly.\n")
        return 4
    kv_table({"dataset": str(dataset), "split": args.split, "resolution": args.resolution})
    print()

    device = settings.resolve_device()
    vton_config = VTONModelConfig(
        base_model=card.get("base_model") or settings.vton_base_pipeline,
        resolution=args.resolution,
        mixed_precision=settings.mixed_precision,
        gradient_checkpointing=False,
        sample_size=max(8, args.resolution // 8),
    )
    components = build_components(vton_config, device=device, controlnet_path=checkpoint)
    pipeline = components.get("pipe") if isinstance(components, dict) else components[0]
    controlnet = components.get("controlnet") if isinstance(components, dict) else components[1]

    loaders = build_dataloaders(
        str(dataset), resolution=args.resolution, mode="FINE_TUNE", batch_size=1,
        num_workers=0, device=str(device), augment=False,
    )
    loader = loaders.get(args.split) or loaders.get("train")
    if loader is None:
        print(f"\n✗ no '{args.split}' split available in {dataset}\n")
        return 4

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    validator = Validator(sample_dir=Path(args.output))
    report = validator.run(
        pipeline=pipeline,
        controlnet=controlnet,
        val_loader=loader,
        noise_scheduler=pipeline.scheduler,
        device=device,
        weight_dtype=getattr(pipeline, "dtype", torch.float32),
        step=0,
        epoch=0,
        resolution=args.resolution,
        num_batches=args.batches,
        num_inference_images=args.images,
        inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    )

    banner("validation results")
    kv_table({k: v for k, v in (report or {}).items() if not isinstance(v, (list, dict))})
    grids = (report or {}).get("grids") or []
    if grids:
        print("\n  grids:")
        for grid in grids[:8]:
            print(f"    {grid}")
    metrics = (report or {}).get("metrics") or {}
    if metrics:
        print()
        kv_table({"ssim": metrics.get("ssim"), "psnr": metrics.get("psnr"), "masked_l1": metrics.get("masked_l1")})
        print("\n  interpretation: SSIM/PSNR/L1 above are computed *inside the garment mask*,")
        print("  which is the region the model is actually responsible for.")
    if args.json:
        print_json(report)
    print(f"\n  open the grids: {Path(args.output).resolve()}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
