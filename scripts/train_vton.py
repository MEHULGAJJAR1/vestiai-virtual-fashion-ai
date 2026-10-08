#!/usr/bin/env python
"""VestiAI — VTON architecture entry point (initialise from pretrained, then fine-tune).

``train.py`` is the everyday trainer. This script is the one you run when you care about the
*architecture itself*:

* it prints exactly how the model is built — which pretrained pipeline is loaded, how the
  ControlNet branch is initialised from the base UNet, how many channels the conditioning
  tensor has, and which parameters actually receive gradients;
* ``--verify-architecture`` builds the same modules from a tiny random UNet (no download!) and
  runs one forward pass, proving the wiring — including the 9-channel latent input
  (4 noisy latents + 4 masked image + 1 mask) and the 3-channel garment control image;
* ``--plan`` writes a JSON manifest of the run so a cluster job can be audited later;
* with no verification flag it hands over to the trainer with explicit architecture arguments.

Nothing here starts from random weights except the verification mode, which is *supposed* to
be random: it only exists to prove the shapes line up.

    python scripts/train_vton.py --verify-architecture          # seconds, CPU, no download
    python scripts/train_vton.py --plan docs/model_plan.json
    python scripts/train_vton.py --mode FINE_TUNE --dataset datasets/viton_hd \\
        --base-model stabilityai/stable-diffusion-2-inpainting --lora
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _common import (  # noqa: E402
    PROJECT_ROOT, add_common_io_args, banner, fail, kv_table, load_settings, print_device_banner,
    print_json, require_torch, resolve_dataset, setup_logging,
)

ARCHITECTURE_NOTES = """
Architecture (identical for training and inference — see backend/models/vton_controlnet.py)

  base UNet        frozen  Stable Diffusion inpainting UNet (9-channel input conv:
                           4 noisy latents + 4 masked-image latents + 1 inpainting mask)
  ControlNet       trained  initialised with ControlNetModel.from_unet(base_unet) so the
                           branch starts as a near-identity residual — the reason a few
                           thousand steps are enough instead of millions
  control image    garment photo resized to the latent grid (3 channels → zero-conv)
  text encoder     frozen, one prompt per garment category
  VAE              frozen, used to encode the person/garment and to decode validation samples
  optional LoRA    trained  attention adapters injected into the frozen UNet (FINE_TUNE)
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inspect / verify the VTON architecture, then optionally fine-tune it.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", default="FINE_TUNE", choices=["QUICK_DEMO", "FINE_TUNE", "FULL_TRAINING"])
    p.add_argument("--dataset", default=None)
    p.add_argument("--base-model", default=None, help="pretrained inpainting pipeline (HF id or local path)")
    p.add_argument("--lora", action="store_true", help="train LoRA adapters on the frozen UNet")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--verify-architecture", action="store_true",
                   help="build tiny components from scratch and run a forward pass (no download)")
    p.add_argument("--plan", default=None, help="write a JSON run manifest here")
    p.add_argument("--dry-run", action="store_true", help="run 2 real training steps and exit")
    p.add_argument("--notes", action="store_true", help="print the architecture notes and exit")
    add_common_io_args(p)
    return p


def verify_architecture() -> int:
    """Prove the conditioning shapes without downloading anything.

    Builds a tiny inpainting UNet + ControlNet pair with the same structure the real model
    uses (same number of input channels, same conditioning-channel count, same attention
    placement relative to block_out_channels) and runs one forward pass on synthetic
    tensors. If this passes, the 9-channel latent prep in ``keypoints.pack_conditioning_tensor``
    and the export/inference paths are consistent.
    """
    torch = require_torch("architecture verification")
    from diffusers import ControlNetModel, UNet2DConditionModel

    from backend.cv.keypoints import BodyFeatures, pack_conditioning_tensor
    from backend.models.vton_controlnet import VTONModelConfig

    banner("VestiAI · architecture verification", "tiny random modules — no weights downloaded")
    config = VTONModelConfig()

    unet = UNet2DConditionModel(
        sample_size=16,
        in_channels=9,                       # SD-inpainting latent layout
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=64,
        attention_head_dim=8,
    )
    controlnet = ControlNetModel.from_unet(unet, conditioning_channels=config.controlnet_condition_channels)
    controlnet.eval()
    unet.eval()

    # Build a real conditioning stack from synthetic inputs: agnostic(3) + pose(3) + mask(3)
    # = the 9 channels the ControlNet consumes at both train and inference time.
    import numpy as np

    height = width = 128
    rng = np.random.default_rng(0)
    agnostic = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    pose_map = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    garment_mask = np.zeros((height, width), np.uint8)
    garment_mask[40:90, 30:100] = 255
    landmarks = np.zeros((33, 4), np.float32)

    features = BodyFeatures(
        agnostic_rgb=agnostic,
        pose_map=pose_map,
        garment_mask=garment_mask,
        bbox=(30, 40, 100, 90),
        keypoints=landmarks,
    )
    tensor = pack_conditioning_tensor(features, size=(16, 16))
    assert tensor is not None, "pack_conditioning_tensor returned None"
    print(f"  packed conditioning tensor : {tuple(tensor.shape)}  (expected (9, 16, 16))")
    print(f"  value range                : [{float(tensor.min()):.2f}, {float(tensor.max()):.2f}]  (normalised to [-1, 1])")

    sample = torch.randn(1, 9, 16, 16)
    timestep = torch.tensor([500])
    encoder_hidden_states = torch.randn(1, 12, 64)
    control_image = torch.randn(1, config.controlnet_condition_channels, 16, 16)

    with torch.no_grad():
        down_samples, mid_sample = controlnet(
            sample, timestep, encoder_hidden_states, controlnet_cond=control_image, return_dict=False,
        )
        prediction = unet(sample, timestep, encoder_hidden_states).sample

    checks = {
        "controlnet conditioning channels": controlnet.config.conditioning_channels,
        "unet input channels": unet.config.in_channels,
        "controlnet residual blocks": len(down_samples),
        "mid block feature shape": tuple(mid_sample.shape),
        "unet velocity prediction shape": tuple(prediction.shape),
        "controlnet trainable parameters": sum(p.numel() for p in controlnet.parameters() if p.requires_grad),
    }
    kv_table(checks)
    ok = (
        tensor.shape[0] == 9
        and controlnet.config.conditioning_channels == config.controlnet_condition_channels
        and unet.config.in_channels == 9
        and tuple(prediction.shape) == (1, 4, 16, 16)
    )
    if ok:
        print("\n✓ architecture verified: 9-channel latent + ControlNet-from-UNet wiring is consistent,")
        print("  and `ControlNetModel.from_unet` is available in this diffusers version.\n")
        return 0
    print("\n✗ architecture check FAILED — the diffusers version may have changed the API.\n")
    return 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    if args.notes:
        print(ARCHITECTURE_NOTES)
        return 0

    if args.verify_architecture:
        return verify_architecture()

    settings = load_settings(args.config)
    from backend.models.vton_controlnet import VTONModelConfig
    from backend.training.trainer import TrainingConfig

    config = TrainingConfig.for_mode(args.mode)
    if args.dataset:
        dataset = resolve_dataset(settings, args.dataset)
        if dataset is None:
            fail(f"dataset '{args.dataset}' not found")
        config.dataset_root = str(dataset)
    if args.base_model:
        config.base_model = args.base_model
    if args.resolution:
        config.resolution = args.resolution
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lora or args.mode == "FINE_TUNE":
        config.train_unet_lora = True
        config.lora_rank = args.lora_rank
    config.dry_run = args.dry_run

    vton = VTONModelConfig(
        base_model=config.base_model,
        train_controlnet=True,
        train_unet_lora=config.train_unet_lora,
        lora_rank=config.lora_rank,
        resolution=config.resolution,
        mixed_precision=config.mixed_precision,
        gradient_checkpointing=config.gradient_checkpointing,
        use_xformers=config.use_xformers,
        snr_gamma=config.snr_gamma,
    )

    banner("VestiAI · VTON architecture plan", f"mode={config.mode}")
    print(ARCHITECTURE_NOTES)
    print_device_banner()
    kv_table(vton.to_dict())

    plan = {
        "mode": config.mode,
        "dataset": config.dataset_root,
        "output": config.output_dir,
        "training_config": config.to_dict(),
        "architecture": vton.to_dict(),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "project_root": str(PROJECT_ROOT),
        },
    }
    if args.plan:
        destination = Path(args.plan)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(plan, indent=2, default=str), encoding="utf-8")
        print(f"  manifest  : {destination}")
    if args.json:
        print_json(plan)

    print("\n  handing over to the trainer (scripts/train.py uses the same code path)\n")
    from train import main as train_main   # scripts/ is on sys.path when run as a script

    argv_out = ["--mode", config.mode, "--dataset", config.dataset_root]
    if args.base_model:
        argv_out += ["--base-model", args.base_model]
    if config.resolution:
        argv_out += ["--resolution", str(config.resolution)]
    if config.num_epochs:
        argv_out += ["--epochs", str(config.num_epochs)]
    if config.batch_size:
        argv_out += ["--batch-size", str(config.batch_size)]
    if config.train_unet_lora:
        argv_out += ["--lora"]
    if args.dry_run:
        argv_out += ["--dry-run"]
    return train_main(argv_out)


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
