#!/usr/bin/env python
"""VestiAI — build dataset samples from your own person + garment photos.

This is the bridge between "I have photos" and "I have a trainable dataset". It runs the
same functions the preparation pipeline uses:

    person photo ─pose─┐
                       ├─→ worn-garment mask → agnostic image → pose map → sample
    garment photo ─mask┘

    python scripts/preprocessing.py --person me.jpg --garment tee.jpg --output datasets/mine/train
    python scripts/preprocessing.py --person me.jpg --garment tee.jpg --category kurta --kind upper
    python scripts/preprocessing.py --pairs pairs.csv --output datasets/mine/train --resolution 512

`pairs.csv` is three columns: ``person_path,garment_path,category`` (a header is optional).
Every written sample gets ``image/cloth/cloth-mask/agnostic/pose`` files plus a ``pairs.txt``
row, so the folder is immediately usable by ``scripts/train.py``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from _common import (  # noqa: E402
    add_common_io_args, banner, fail, kv_table, load_settings, print_json, setup_logging,
)

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create VTON training samples from person + garment photo pairs.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--person", default=None, help="person photo (single-pair mode)")
    p.add_argument("--garment", default=None, help="garment photo (single-pair mode)")
    p.add_argument("--pairs", default=None, help="CSV of person,garment[,category] pairs")
    p.add_argument("--output", required=True, help="split folder, e.g. datasets/mine/train")
    p.add_argument("--category", default="t-shirt", help="garment category for the prompt (single-pair mode)")
    p.add_argument("--kind", default="upper", choices=["upper", "lower", "overall"])
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--agnostic-mode", default="grey", choices=["grey", "blur", "white", "black"])
    p.add_argument("--mask-dilate", type=int, default=9)
    p.add_argument("--augment", type=int, default=0, help="write N augmented copies of each sample")
    p.add_argument("--json", action="store_true")
    add_common_io_args(p)
    return p


def collect_pairs(args) -> list[dict]:
    if args.pairs:
        rows: list[dict] = []
        with Path(args.pairs).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for index, row in enumerate(reader):
                if not row or row[0].strip().lower() in {"person", "person_path", "#"}:
                    continue
                if len(row) < 2:
                    continue
                rows.append({
                    "person": row[0].strip(),
                    "garment": row[1].strip(),
                    "category": (row[2].strip() if len(row) > 2 and row[2].strip() else args.category),
                })
        if not rows:
            fail(f"no usable rows in {args.pairs}")
        return rows
    if args.person and args.garment:
        return [{"person": args.person, "garment": args.garment, "category": args.category}]
    fail("provide either --pairs CSV or both --person and --garment")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    load_settings(args.config)          # validates the runtime config early

    import numpy as np
    from PIL import Image

    from backend.datasets.preprocessing import build_sample, write_sample

    pairs = collect_pairs(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    banner("VestiAI · building dataset samples", f"{len(pairs)} pair(s) → {output}")
    kv_table({
        "resolution": args.resolution,
        "kind": args.kind,
        "agnostic mode": args.agnostic_mode,
        "augmented copies": args.augment,
    })
    print()

    written: list[dict] = []
    failures: list[dict] = []
    for index, pair in enumerate(pairs, start=1):
        person_path = Path(pair["person"]).expanduser()
        garment_path = Path(pair["garment"]).expanduser()
        label = f"  [{index}/{len(pairs)}] {person_path.name} + {garment_path.name}"
        if not person_path.exists() or not garment_path.exists():
            failures.append({"pair": pair, "reason": "file not found"})
            print(f"{label} ✗ file not found")
            continue
        if person_path.suffix.lower() not in SUPPORTED or garment_path.suffix.lower() not in SUPPORTED:
            failures.append({"pair": pair, "reason": f"unsupported suffix ({person_path.suffix}/{garment_path.suffix})"})
            print(f"{label} ✗ unsupported image type")
            continue
        try:
            person = np.asarray(Image.open(person_path).convert("RGB"))
            garment = np.asarray(Image.open(garment_path).convert("RGB"))
            sample = build_sample(
                person, garment,
                kind=args.kind,
                agnostic_mode=args.agnostic_mode,
                resolution=args.resolution,
            )
        except Exception as exc:
            failures.append({"pair": pair, "reason": str(exc)})
            print(f"{label} ✗ {exc}")
            continue

        identifier = f"{person_path.stem}__{garment_path.stem}"
        name = write_sample(sample, output, identifier, cloth_name=garment_path.name)
        with (output / "pairs.txt").open("a", encoding="utf-8") as handle:
            handle.write(f"{person_path.name} {garment_path.name} {pair['category']} {args.kind}\n")

        if args.augment:
            from backend.datasets.preprocessing import augment_sample, AugmentConfig
            import random

            rng = random.Random(index)
            config = AugmentConfig()
            for copy_index in range(args.augment):
                variant = augment_sample(sample, config, rng)
                write_sample(variant, output, f"{identifier}_aug{copy_index + 1}", cloth_name=garment_path.name)

        written.append({"identifier": identifier, "files": name, "category": pair["category"]})
        print(f"{label} ✓ {identifier}")

    print()
    banner("summary")
    kv_table({"written": len(written), "failed": len(failures), "output": str(output)})
    if failures:
        print("\n  failures:")
        for item in failures[:10]:
            print(f"    ✗ {Path(item['pair']['person']).name}: {item['reason'][:100]}")
    if args.json:
        print_json({"written": written, "failures": failures})

    print("\n  next:")
    print(f"    python scripts/dataset.py --dataset {output} --preview results/dataset_preview.png")
    print(f"    python scripts/validate_dataset.py --dataset {output}")
    print(f"    python scripts/train.py --mode QUICK_DEMO --dataset {output}\n")
    return 0 if written else 1


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
