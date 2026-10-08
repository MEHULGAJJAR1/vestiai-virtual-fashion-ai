#!/usr/bin/env python
"""VestiAI — dataloader smoke test + throughput benchmark.

Builds the real ``build_dataloaders(...)`` stack (dataset → collate → worker processes) and
measures what the trainer will actually see: tensor shapes, dtypes, prompt strings and pairs
per second, optionally including a host→GPU transfer. Run this before a long training job —
it catches broken pairs, wrong resolutions and mis-sized batches in seconds instead of after
an hour of GPU time.

    python scripts/dataloader.py --dataset datasets/samples
    python scripts/dataloader.py --dataset datasets/viton_hd --resolution 512 --batch 4 --batches 30
    python scripts/dataloader.py --dataset datasets/samples --workers 4 --device cuda --transfer
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from _common import (  # noqa: E402
    add_common_io_args, banner, fail, kv_table, load_settings, print_json, resolve_dataset,
    setup_logging,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark the VestiAI dataloader on a real dataset.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True, help="dataset name under datasets/ or an absolute path")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--batch", type=int, default=2, help="batch size")
    p.add_argument("--workers", type=int, default=2, help="DataLoader worker processes")
    p.add_argument("--batches", type=int, default=10, help="how many batches to time")
    p.add_argument("--mode", default="QUICK_DEMO", choices=["QUICK_DEMO", "FINE_TUNE", "FULL_TRAINING"])
    p.add_argument("--device", default="cpu", help="device used for the optional transfer test")
    p.add_argument("--transfer", action="store_true", help="also move each batch to --device and time it")
    p.add_argument("--no-augment", action="store_true", help="disable augmentation to measure raw IO")
    p.add_argument("--json", action="store_true")
    add_common_io_args(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    settings = load_settings(args.config)

    root = resolve_dataset(settings, args.dataset)
    if root is None:
        fail(f"dataset '{args.dataset}' not found (looked in {settings.datasets_dir})")

    banner("VestiAI · dataloader benchmark", str(root))
    kv_table({
        "resolution": args.resolution,
        "batch size": args.batch,
        "workers": args.workers,
        "augmentation": "off" if args.no_augment else "on",
        "mode": args.mode,
    })
    print()

    from backend.datasets.dataloader import build_dataloaders, describe_loaders

    started = time.time()
    loaders = build_dataloaders(
        str(root),
        resolution=args.resolution,
        mode=args.mode,
        batch_size=args.batch,
        num_workers=args.workers,
        device=args.device,
        augment=not args.no_augment,
    )
    description = describe_loaders(loaders)
    kv_table({"loader sizes": {k: (v.get("batches") if isinstance(v, dict) else v) for k, v in description.items()}})

    train_loader = loaders.get("train") if isinstance(loaders, dict) else loaders
    if train_loader is None or not hasattr(train_loader, "__iter__"):
        fail("the train loader could not be built — is the dataset empty?")

    batches = 0
    samples = 0
    first_shapes: dict = {}
    transfer_seconds = 0.0
    device = None
    if args.transfer:
        import torch

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print(f"  ! --transfer requested {args.device} but CUDA is unavailable — timing CPU copies instead")
            device = torch.device("cpu")

    started = time.time()
    for batch in train_loader:
        if not first_shapes:
            first_shapes = {
                key: (tuple(value.shape), str(value.dtype))
                for key, value in batch.items()
                if hasattr(value, "shape")
            }
            sample_prompts = batch.get("prompt") if isinstance(batch.get("prompt"), (list, tuple)) else None
        batches += 1
        samples += int(batch["pixel_values"].shape[0]) if hasattr(batch.get("pixel_values"), "shape") else args.batch
        if device is not None:
            tick = time.perf_counter()
            for value in batch.values():
                if hasattr(value, "to"):
                    value.to(device)
            transfer_seconds += time.perf_counter() - tick
        if batches >= args.batches:
            break

    elapsed = time.time() - started
    decode_seconds = max(1e-6, elapsed - transfer_seconds)
    result = {
        "batches": batches,
        "samples": samples,
        "seconds": round(elapsed, 3),
        "samples_per_second": round(samples / elapsed, 2) if elapsed else 0,
        "seconds_per_batch": round(elapsed / max(1, batches), 4),
        "tensor_summary": first_shapes,
        "device_transfer_s": round(transfer_seconds, 3),
        "estimated_epoch_seconds": None,
    }
    train_batches = description.get("train", {}).get("batches") if isinstance(description.get("train"), dict) else None
    if train_batches:
        result["estimated_epoch_seconds"] = round(train_batches * (elapsed / max(1, batches)), 1)

    print()
    banner("results")
    kv_table({k: v for k, v in result.items() if k != "tensor_summary"})
    print("\n  tensors per batch:")
    for key, value in first_shapes.items():
        print(f"    {key:<16} {value[0]}  {value[1]}")
    if sample_prompts:
        print("\n  prompts:")
        for prompt in list(sample_prompts)[:3]:
            print(f"    “{prompt}”")
    print(f"\n  decode throughput: {decode_seconds:.2f}s for {samples} samples "
          f"({samples / decode_seconds:.2f} samples/s)")
    if result["estimated_epoch_seconds"]:
        print(f"  estimated epoch  : {result['estimated_epoch_seconds']}s (data loading only)")

    if args.json:
        print_json(result)
    print()
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
