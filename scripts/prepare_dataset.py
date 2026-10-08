#!/usr/bin/env python
"""VestiAI — dataset preparation.

Three real modes, no downloads unless you ask for them:

1. **Sample data** (``--use-samples``) generates a procedurally drawn VITON-style dataset so
   the whole pipeline is runnable with zero downloads and zero copyright risk.
2. **Convert** (``--source /path/to/viton_hd --name viton_hd``) converts a *legitimately
   obtained* VITON-HD / DressCode folder into VestiAI's layout, deriving the agnostic image,
   the cloth mask and the pose conditioning from the source files.
3. **Split** (``--split-only``) re-partitions an existing VestiAI dataset into train/val/test.

Output layout (compatible with VITON-HD):

    <name>/train/{image,cloth,cloth-mask,agnostic,pose}/ + pairs.txt
    <name>/val/...   <name>/test/...   <name>/metadata.json

Nothing here scrapes shopping sites: bring your own images or use the generator.

Examples
--------
    python scripts/prepare_dataset.py --use-samples
    python scripts/prepare_dataset.py --use-samples --train 96 --val 16 --test 16 --size 512
    python scripts/prepare_dataset.py --source ~/data/viton_hd --name viton_hd --resolution 512
    python scripts/prepare_dataset.py --split-only --dataset datasets/viton_hd --val-fraction 0.1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from _common import (  # noqa: E402
    PROJECT_ROOT, add_common_io_args, banner, fail, human_time, kv_table, load_settings,
    print_json, resolve_dataset, setup_logging,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare a VestiAI training dataset (sample / convert / split).",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--use-samples", action="store_true", help="generate the procedural sample dataset")
    mode.add_argument("--source", default=None, help="source folder (VITON-HD / DressCode compatible)")
    mode.add_argument("--split-only", action="store_true", help="re-split an existing VestiAI dataset")

    p.add_argument("--name", default=None, help="dataset name under datasets/ (default: samples | source folder name)")
    p.add_argument("--dataset", default=None, help="existing dataset (with --split-only)")
    p.add_argument("--resolution", type=int, default=512, choices=[256, 512, 768, 1024])
    p.add_argument("--train", type=int, default=48, help="sample mode: number of training pairs")
    p.add_argument("--val", type=int, default=12, help="sample mode: number of validation pairs")
    p.add_argument("--test", type=int, default=12, help="sample mode: number of test pairs")
    p.add_argument("--size", type=int, default=512, help="sample mode: generated image size")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--overwrite", action="store_true", help="regenerate an existing sample dataset")
    p.add_argument("--no-metadata", action="store_true", help="skip metadata.json (rarely useful)")
    p.add_argument("--json", action="store_true", help="print the raw service report")
    add_common_io_args(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    settings = load_settings(args.config)

    from backend.services.dataset_service import DatasetService

    service = DatasetService(settings.datasets_dir)
    started = time.time()

    # ------------------------------------------------------------------ sample mode
    if args.use_samples:
        banner("VestiAI · generating the sample dataset", "procedural, no downloads, no third-party images")
        counts = {"train": args.train, "val": args.val, "test": args.test}
        print(f"  splits    : {counts}")
        print(f"  image size: {args.size}px (stored at {args.size}, resized to {args.resolution} while training)")
        print()
        report = service.generate_samples(count=counts, size=args.size, seed=args.seed, overwrite=args.overwrite)
        if args.json:
            print_json(report)
        root = Path(report.get("root") or report.get("path") or (Path(settings.datasets_dir) / "samples"))
        print()
        banner("dataset ready", str(root))
        kv_table({k: v for k, v in report.items() if k in {"name", "root", "counts", "samples", "seconds", "size"}})
        print(f"  elapsed   : {human_time(time.time() - started)}")
        print("\n  next: python scripts/validate_dataset.py --dataset " + (args.name or "samples"))
        print("        python scripts/train.py --mode QUICK_DEMO\n")
        return 0

    # ------------------------------------------------------------------ convert mode
    if args.source:
        source = Path(args.source).expanduser()
        if not source.exists():
            fail(f"source folder not found: {source}")
        name = args.name or source.name.lower().replace(" ", "_")
        banner("VestiAI · converting a VITON-HD / DressCode folder", f"{source}  →  datasets/{name}")
        print("  Only converts files you already have on disk. VestiAI never downloads or scrapes")
        print("  shopping sites — see docs/TRAINING.md for the legitimate dataset sources.\n")

        from backend.datasets.preprocessing import DatasetConfig

        config = DatasetConfig(
            root=str(Path(settings.datasets_dir) / name),
            resolution=args.resolution,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        report = service.convert_viton_folder(source, name=name, config=config)
        if args.json:
            print_json(report)
        banner("conversion finished", str(report.get("root", "")))
        kv_table({k: v for k, v in report.items() if k != "root"})
        print(f"  elapsed   : {human_time(time.time() - started)}")
        if report.get("warnings"):
            print("\n  warnings:")
            for warning in report["warnings"][:8]:
                print(f"    · {warning}")
        print("\n  next: python scripts/validate_dataset.py --dataset " + name)
        print("        python scripts/train.py --mode FINE_TUNE --dataset datasets/" + name + "\n")
        return 0

    # ------------------------------------------------------------------ split mode
    target = resolve_dataset(settings, args.dataset or args.name)
    if target is None:
        fail("--split-only needs --dataset <name-or-path> pointing at an existing dataset")
    banner("VestiAI · re-splitting a dataset", str(target))
    report = service.split_dataset(str(target), val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed)
    if args.json:
        print_json(report)
    kv_table(report)
    print(f"\n  elapsed : {human_time(time.time() - started)}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
