#!/usr/bin/env python
"""VestiAI — train / fine-tune the virtual try-on model.

This is the entry point you would use on a GPU box. It builds the ControlNet-conditioned
inpainting pipeline, loads a real dataset through ``build_dataloaders`` and runs the
trainer: AMP, gradient accumulation, gradient checkpointing, periodic validation grids,
TensorBoard/W&B metrics, best/latest/epoch checkpoints, early stopping and resume.

Modes (presets live in backend/training/trainer.py)

    QUICK_DEMO    2 epochs, 256 px, 64 samples, no AMP — verify the pipeline in minutes on CPU
    FINE_TUNE     5 epochs, 512 px, LoRA — adapt a pretrained VTON to your garment categories
    FULL_TRAINING 30 epochs, 512 px — complete dataset on a real GPU

Examples

    python scripts/train.py --mode QUICK_DEMO                     # verify end to end
    python scripts/train.py --mode QUICK_DEMO --dry-run           # build everything, 2 steps, exit
    python scripts/train.py --mode FINE_TUNE --config configs/training_finetune.yaml \\
        --dataset datasets/viton_hd --batch-size 2 --epochs 5
    python scripts/train.py --mode FULL_TRAINING --resume --no-resume-if-missing \\
        --device cuda --precision bf16

Every run writes:
    logs/training_status.json          live dashboard state (what the UI polls)
    logs/training_<timestamp>/         metrics.jsonl + metrics.csv + TensorBoard events
    checkpoints/epochs/epoch_XXXX/     adapter weights + training_state.pt + metrics.json
    checkpoints/best_model/            best validation loss so far
    checkpoints/latest_model/          newest weights (for resume)
    results/validation/step_*/         generated validation grids
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from _common import (  # noqa: E402
    PROJECT_ROOT, banner, fail, human_time, kv_table, load_settings, log_level_arg,
    print_device_banner, print_json, require_torch, resolve_dataset, setup_logging,
)

MODES = ("QUICK_DEMO", "FINE_TUNE", "FULL_TRAINING")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train or fine-tune the VestiAI diffusion try-on model.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", default="QUICK_DEMO", choices=MODES, help="training preset")
    p.add_argument("--config", default=None, help="training YAML (configs/training_*.yaml)")
    p.add_argument("--dataset", default=None, help="dataset name under datasets/ or a path")
    p.add_argument("--output", default=None, help="checkpoint root (default: checkpoints/)")

    p.add_argument("--resolution", type=int, default=None, choices=[256, 384, 512, 768, 1024])
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="learning rate")
    p.add_argument("--accum", type=int, default=None, help="gradient accumulation steps")
    p.add_argument("--workers", type=int, default=None, help="dataloader workers")
    p.add_argument("--precision", default=None, choices=["auto", "fp16", "bf16", "no"])
    p.add_argument("--optimizer", default=None, choices=["adamw", "adamw8bit", "sgd"])
    p.add_argument("--scheduler", default=None, choices=["cosine", "linear", "constant"])
    p.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps")
    p.add_argument("--max-samples", type=int, default=None, help="cap the training set size")
    p.add_argument("--base-model", default=None, help="base inpainting pipeline (HF id or path)")
    p.add_argument("--lora", action="store_true", help="train LoRA adapters on the UNet as well")
    p.add_argument("--no-lora", action="store_true", help="disable LoRA even if the preset enables it")
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--perceptual", action="store_true", help="enable the LPIPS perceptual loss")
    p.add_argument("--garment-clip", action="store_true", help="enable the CLIP garment-consistency loss")
    p.add_argument("--xformers", action="store_true", help="enable xformers attention")
    p.add_argument("--tracker", default=None, choices=["tensorboard", "wandb", "none"])
    p.add_argument("--no-resume", action="store_true", help="ignore checkpoints/latest_model")
    p.add_argument("--dry-run", action="store_true", help="build everything, run 2 steps and exit (verification)")

    p.add_argument("--json", action="store_true", help="print the final summary as JSON")
    log_level_arg(p)
    return p


def resolve_config(args):
    """Training config = mode preset → --config YAML → explicit CLI flags (last one wins)."""
    from backend.training.trainer import TrainingConfig

    if args.config:
        path = Path(args.config)
        if not path.exists():
            fail(f"training config not found: {path}")
        config = TrainingConfig.from_yaml(path)
        print(f"  training config: {path}")
    else:
        config = TrainingConfig.for_mode(args.mode)
    config.mode = args.mode

    # Runtime settings (paths, cache dirs) come from configs/default.yaml — `--config` is the
    # *training* YAML, so the two never collide.
    settings = load_settings()
    dataset = resolve_dataset(settings, args.dataset, create=False) if args.dataset else None
    if dataset is not None:
        config.dataset_root = str(dataset)
    elif args.dataset:
        fail(f"dataset '{args.dataset}' not found (looked under {settings.datasets_dir})")

    overrides = {
        "resolution": args.resolution,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "gradient_accumulation": args.accum,
        "num_workers": args.workers,
        "mixed_precision": args.precision,
        "optimizer": args.optimizer,
        "lr_scheduler": args.scheduler,
        "device": args.device,
        "seed": args.seed,
        "max_train_steps": args.max_steps,
        "max_train_samples": args.max_samples,
        "base_model": args.base_model,
        "lora_rank": args.lora_rank,
        "tracker": args.tracker,
        "output_dir": args.output,
    }
    for key, value in overrides.items():
        if value is None:            # flag not passed → keep whatever the YAML/preset said
            continue
        if not hasattr(config, key):
            print(f"  ⚠ ignoring --{key.replace('_', '-')}: not a TrainingConfig field")
            continue
        setattr(config, key, value)

    if args.lora:
        config.train_unet_lora = True
    if args.no_lora:
        config.train_unet_lora = False
    if args.perceptual:
        config.enable_perceptual_loss = True
    if args.garment_clip:
        config.enable_garment_clip_loss = True
    if args.xformers:
        config.use_xformers = True
    if args.no_resume:
        config.resume = False
    config.dry_run = args.dry_run
    return config, dataset


def preflight(config, dataset) -> None:
    """Fail loudly and early rather than after an hour of loading."""
    from backend.utils.device import detect_devices, gpu_utilization

    devices = detect_devices()
    problems: list[str] = []
    if dataset is None:
        problems.append(
            f"dataset '{config.dataset_root}' does not exist — run "
            "`python scripts/prepare_dataset.py --use-samples` first"
        )
    else:
        from backend.datasets.preprocessing import validate_layout

        layout = validate_layout(dataset)
        if not layout.get("ok"):
            problems.append(f"dataset is not trainable: {', '.join(layout.get('issues') or ['unknown problem'])}")
        else:
            counts = {
                name: (entry.get("files", {}).get("image", 0), entry.get("pairs", 0))
                for name, entry in (layout.get("splits") or {}).items() if entry.get("exists")
            }
            rendered = ", ".join(f"{name}: {images} image(s)/{pairs} pair(s)" for name, (images, pairs) in counts.items())
            print(f"  dataset: {dataset} · {rendered or 'no splits found'}")
            for warning in (layout.get("warnings") or [])[:3]:
                print(f"    ⚠ {warning}")

    if not devices.torch_available:
        problems.append("PyTorch is not installed — run `python scripts/setup.py --profile ml`")
    elif config.mode != "QUICK_DEMO" and not devices.cuda_available:
        print("\n  ⚠ no CUDA GPU detected.")
        print("    QUICK_DEMO will still run (minutes on CPU). FINE_TUNE/FULL_TRAINING belong on a")
        print("    CUDA machine or in the cloud — see configs/training_colab.yaml and docs/TRAINING.md.\n")
    if config.mode == "FULL_TRAINING" and devices.cuda_available and devices.vram_total_gb:
        vram = devices.vram_total_gb[0]
        needed = {256: 8, 384: 12, 512: 16, 768: 24, 1024: 40}.get(config.resolution, 16)
        if vram < needed:
            problems.append(
                f"FULL_TRAINING at {config.resolution}px wants ≈{needed} GB VRAM but this GPU has {vram} GB — "
                f"lower --resolution, reduce --batch-size, or keep gradient_checkpointing on"
            )
    if config.optimizer == "adamw8bit" and config.device in {"cpu"}:
        problems.append("adamw8bit needs a CUDA GPU (or bitsandbytes CPU build) — use optimizer: adamw")
    if problems:
        print("\n✗ pre-flight checks failed:")
        for item in problems:
            print(f"    · {item}")
        print()
        raise SystemExit(4)
    if devices.cuda_available:
        live = gpu_utilization()
        if live.get("gpus"):
            gpu = live["gpus"][0]
            print(f"  gpu live: {gpu.get('gpu_util_pct', 0)}% util · "
                  f"{int(gpu.get('vram_used_mb') or 0)}/{int(gpu.get('vram_total_mb') or 0)} MB used")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    require_torch("training")

    config, dataset = resolve_config(args)

    banner("VestiAI · training", f"mode={config.mode}  resolution={config.resolution}px")
    print_device_banner()
    preflight(config, dataset)

    from backend.training.trainer import Trainer

    status_path = PROJECT_ROOT / "logs" / "training_status.json"
    session_started = time.time()
    trainer = Trainer(config=config, status_path=status_path)

    print("\n  plan:")
    kv_table({
        "dataset": config.dataset_root,
        "resolution": f"{config.resolution}px",
        "epochs": config.num_epochs,
        "batch size": config.batch_size,
        "gradient accumulation": config.gradient_accumulation,
        "effective batch": config.batch_size * config.gradient_accumulation,
        "learning rate": config.learning_rate,
        "optimizer": config.optimizer,
        "scheduler": config.lr_scheduler,
        "mixed precision": config.mixed_precision,
        "lora": config.train_unet_lora,
        "perceptual loss": config.enable_perceptual_loss,
        "garment clip loss": config.enable_garment_clip_loss,
        "resume": config.resume,
        "output": config.output_dir,
    })
    print()

    try:
        summary = trainer.train()
    except KeyboardInterrupt:
        print("\n  interrupted — the newest checkpoint in checkpoints/epochs/ is still usable,")
        print("  and `--resume` will continue from it.\n")
        return 130
    except Exception as exc:
        print(f"\n✗ training failed: {exc.__class__.__name__}: {exc}")
        print("  logs/training_status.json holds the last known state; the traceback above shows where.\n")
        raise

    elapsed = time.time() - session_started
    banner("training finished")
    printable = {k: v for k, v in (summary or {}).items() if not isinstance(v, (list, dict)) or k in {"epochs"}}
    kv_table(printable)
    print(f"  wall clock : {human_time(elapsed)}")

    best = Path(config.output_dir) / "best_model"
    if best.exists():
        from backend.models.vton_controlnet import describe_checkpoint

        info = describe_checkpoint(best)
        print("\n  best checkpoint:")
        kv_table(info, indent="    ")

    print("\n  next steps:")
    print(f"    python scripts/validate.py --checkpoint {best}")
    print(f"    python scripts/evaluate.py --checkpoint {best} --dataset {config.dataset_root}")
    print("    python scripts/inference.py --person you.jpg --garment tee.jpg --output result.png")
    print("    then reload the app (Model Status → Reload checkpoint) to use it in the browser.\n")

    if args.json:
        print_json({"summary": summary, "elapsed_s": round(elapsed, 1), "config": config.to_dict()})
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
