#!/usr/bin/env python
"""VestiAI — inspect a training dataset (counts, sizes, categories, preview grid).

    python scripts/dataset.py --dataset datasets/samples
    python scripts/dataset.py --dataset datasets/viton_hd --preview results/dataset_preview.png
    python scripts/dataset.py --dataset datasets/viton_hd --json

`--preview` writes a contact sheet with one row per split showing
person | cloth | cloth-mask | agnostic | pose so you can eyeball the data before spending
GPU hours on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from _common import (  # noqa: E402
    add_common_io_args, banner, fail, kv_table, load_settings, print_json, resolve_dataset,
    setup_logging,
)

SPLITS = ("train", "val", "test")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Report the contents of a VestiAI training dataset.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True, help="dataset name under datasets/ or an absolute path")
    p.add_argument("--split", default="all", choices=["all", *SPLITS])
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--preview", default=None, help="write a contact sheet PNG here")
    p.add_argument("--rows", type=int, default=3, help="rows per split in the preview sheet")
    p.add_argument("--json", action="store_true")
    add_common_io_args(p)
    return p


def read_pairs(split_dir: Path) -> list[dict]:
    pairs_file = split_dir / "pairs.txt"
    if not pairs_file.exists():
        return []
    rows = []
    for line in pairs_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        rows.append({
            "person": parts[0],
            "cloth": parts[1],
            "category": parts[2] if len(parts) > 2 else "unknown",
            "kind": parts[3] if len(parts) > 3 else "upper",
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    settings = load_settings(args.config)

    root = resolve_dataset(settings, args.dataset)
    if root is None:
        fail(f"dataset '{args.dataset}' not found (looked in {settings.datasets_dir})")

    from backend.datasets.dataset import dataset_report

    banner("VestiAI · dataset inspector", str(root))
    report = dataset_report(root, resolution=args.resolution)
    if args.json:
        print_json(report)

    splits = SPLITS if args.split == "all" else (args.split,)
    total = 0
    categories: Counter = Counter()
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"  {split:<6} : missing")
            continue
        pairs = read_pairs(split_dir)
        total += len(pairs)
        for row in pairs:
            categories[row["category"]] += 1
        missing = [field for field in ("image", "cloth", "cloth-mask", "agnostic", "pose")
                   if not (split_dir / field).exists() or not any((split_dir / field).glob("*"))]
        kv_table({
            "pairs": len(pairs),
            "image": _count(split_dir / "image"),
            "cloth": _count(split_dir / "cloth"),
            "cloth-mask": _count(split_dir / "cloth-mask"),
            "agnostic": _count(split_dir / "agnostic"),
            "pose": _count(split_dir / "pose"),
            "missing": ", ".join(missing) or "none",
        })
        print()

    kv_table({"total pairs": total, "categories": dict(categories.most_common())})
    metadata = root / "metadata.json"
    origin = None
    if metadata.exists():
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            origin = payload.get("origin")
            kv_table({
                "metadata": {k: payload.get(k) for k in ("name", "origin", "generator", "size", "layout")
                             if payload.get(k) is not None},
            })
            if payload.get("counts"):
                print(f"    recorded counts : {payload['counts']}")
            if payload.get("note"):
                print(f"    note            : {payload['note']}")
        except Exception as exc:
            print(f"  ⚠ metadata.json is unreadable: {exc}")
    else:
        print("  ⚠ metadata.json is missing (prepare_dataset.py writes it)")

    if args.preview:
        out = write_preview(root, Path(args.preview), rows=args.rows, splits=splits)
        print(f"\n  preview  : {out}")

    mode = "QUICK_DEMO" if origin == "generated" else "FINE_TUNE"
    print(f"\n  train it : python scripts/train.py --mode {mode} --dataset {root}")
    if origin == "generated":
        print("             (sample data verifies the pipeline; download VITON-HD/DressCode for")
        print("              quality training — see docs/TRAINING.md and prepare_dataset.py --source)")
    print()
    return 0


def _count(folder: Path) -> int:
    return sum(1 for _ in folder.glob("*")) if folder.exists() else 0


def write_preview(root: Path, destination: Path, rows: int = 3, splits: tuple[str, ...] = SPLITS) -> Path:
    """Contact sheet: one row per split, columns person | cloth | mask | agnostic | pose."""
    import numpy as np
    from PIL import Image

    fields = ("image", "cloth", "cloth-mask", "agnostic", "pose")
    tile = 200
    sheet = np.full((tile * max(1, len(splits) * rows), tile * len(fields), 3), 18, np.uint8)
    row_index = 0
    for split in splits:
        split_dir = root / split
        pairs = read_pairs(split_dir)
        if not pairs:
            continue
        for pair in pairs[:rows]:
            stem = Path(pair["person"]).stem
            names = {
                "image": pair["person"],
                "cloth": pair["cloth"],
                "cloth-mask": f"{Path(pair['cloth']).stem}.png",
                "agnostic": f"{stem}.jpg",
                "pose": f"{stem}.png",
            }
            for column, field in enumerate(fields):
                candidate = (split_dir / field) / names[field]
                if not candidate.exists():          # tolerate .jpg/.png variations
                    candidates = list((split_dir / field).glob(f"{Path(names[field]).stem}.*"))
                    candidate = candidates[0] if candidates else None
                if candidate is None:
                    continue
                with Image.open(candidate) as image:
                    array = np.asarray(image.convert("RGB").resize((tile, tile)))
                sheet[row_index * tile:(row_index + 1) * tile, column * tile:(column + 1) * tile] = array
            row_index += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(destination)
    return destination


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
