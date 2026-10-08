#!/usr/bin/env python
"""VestiAI — dataset validation + statistics.

Checks the on-disk layout, then opens real samples and verifies the things a trainer would
trip over: image sizes, mask coverage, agnostic-image presence, pose files, pairs.txt
consistency and duplicate identifiers. Exits non-zero when the dataset cannot be trained on
so it is usable as a CI gate.

    python scripts/validate_dataset.py --dataset datasets/samples
    python scripts/validate_dataset.py --dataset datasets/viton_hd --strict --limit 40
    python scripts/validate_dataset.py --dataset datasets/viton_hd --fix-missing-agnostic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import (  # noqa: E402
    add_common_io_args, banner, fail, kv_table, load_settings, print_json, resolve_dataset,
    setup_logging,
)

SPLITS = ("train", "val", "test")
FIELD_NAMES = ("image", "cloth", "cloth-mask", "agnostic", "pose")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate a VestiAI / VITON-HD dataset and print its statistics.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True, help="dataset name under datasets/ or an absolute path")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--limit", type=int, default=25, help="how many samples to open and check (0 = layout only)")
    p.add_argument("--strict", action="store_true", help="treat warnings as errors")
    p.add_argument("--fix-missing-agnostic", action="store_true", help="regenerate missing agnostic images")
    p.add_argument("--json", action="store_true")
    add_common_io_args(p)
    return p


def audit_samples(root: Path, limit: int) -> dict:
    """Open real files and check shapes/masks — catches corruption the layout check can't."""
    import numpy as np
    from PIL import Image

    problems: list[str] = []
    checked = 0
    sizes: dict[str, int] = {}
    coverage: list[float] = []
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for field in FIELD_NAMES:
            folder = split_dir / field
            if not folder.exists():
                problems.append(f"{split}/{field} is missing")
                continue
            files = sorted(folder.glob("*"))
            if not files:
                problems.append(f"{split}/{field} is empty")
                continue
            for file in files:
                if checked >= max(0, limit):
                    break
                try:
                    with Image.open(file) as image:
                        image.load()
                        sizes[f"{split}/{field}"] = max(sizes.get(f"{split}/{field}", 0), image.size[0])
                        if field == "cloth-mask":
                            array = np.asarray(image.convert("L"))
                            coverage.append(float((array > 127).mean()))
                except Exception as exc:
                    problems.append(f"unreadable file {file.relative_to(root)}: {exc}")
                checked += 1
        if checked >= max(0, limit):
            break
    stats: dict = {"files_checked": checked, "max_side": sizes, "problems": problems}
    if coverage:
        stats["cloth_mask_coverage"] = {
            "min": round(min(coverage), 4),
            "max": round(max(coverage), 4),
            "mean": round(sum(coverage) / len(coverage), 4),
        }
        # A garment mask that covers essentially the whole canvas is a sign the mask failed.
        if max(coverage) > 0.98:
            problems.append("at least one cloth-mask covers >98 % of the canvas (mask extraction likely failed)")
        if min(coverage) < 0.01:
            problems.append("at least one cloth-mask is essentially empty")
    return stats


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    settings = load_settings(args.config)

    root = resolve_dataset(settings, args.dataset)
    if root is None:
        fail(f"dataset '{args.dataset}' not found (looked in {settings.datasets_dir} and the working tree)")

    from backend.datasets.preprocessing import prepare_metadata, validate_layout

    banner("VestiAI · dataset validation", str(root))
    layout = validate_layout(root, strict=args.strict)
    kv_table({
        "ok": layout.get("ok"),
        "splits": ", ".join(
            f"{name}:{entry.get('files', {}).get('image', 0)} image(s)/{entry.get('pairs', 0)} pair(s)"
            for name, entry in (layout.get("splits") or {}).items() if entry.get("exists")
        ) or "none",
        "split folders": [name for name, entry in (layout.get("splits") or {}).items() if entry.get("exists")],
        "metadata": "present" if (root / "metadata.json").exists() else "missing",
    })

    audit = audit_samples(root, args.limit)
    if audit["files_checked"]:
        print()
        kv_table({"files checked": audit["files_checked"], "max side": audit["max_side"]})
        if audit.get("cloth_mask_coverage"):
            kv_table({"cloth mask coverage": audit["cloth_mask_coverage"]})

    issues = list(layout.get("issues") or []) + list(audit["problems"])
    warnings = list(layout.get("warnings") or [])

    if args.fix_missing_agnostic:
        fixed = regenerate_agnostic(root)
        print(f"\n  regenerated {fixed} agnostic image(s)")

    if not (root / "metadata.json").exists():
        payload = prepare_metadata(root, extra={"resolution": args.resolution, "validated_at": __import__("time").time()})
        print(f"  wrote metadata.json ({payload.get('total_pairs', 0)} pairs)")

    if args.json:
        print_json({"layout": layout, "audit": audit, "issues": issues, "warnings": warnings})
    else:
        if warnings:
            print("\n  warnings:")
            for item in warnings[:12]:
                print(f"    · {item}")
        if issues:
            print("\n  ✗ issues:")
            for item in issues[:20]:
                print(f"    · {item}")
        else:
            print("\n  ✓ no blocking issues found")

    if issues:
        print(f"\n✗ {len(issues)} issue(s) — this dataset needs fixing before training.\n")
        return 1
    print("\n✓ dataset is trainable.\n")
    return 0


def regenerate_agnostic(root: Path) -> int:
    """Recreate missing ``agnostic`` images with the real preprocessing pipeline.

    The agnostic image is the person photo with the clothing region erased (and the pose
    skeleton re-drawn on top) — exactly what ``build_sample`` produces during preparation,
    so the file stays consistent with the rest of the dataset.
    """
    import numpy as np
    from PIL import Image

    from backend.datasets.preprocessing import build_sample, cloth_mask_from_product_image

    fixed = 0
    for split in SPLITS:
        split_dir = root / split
        pairs = split_dir / "pairs.txt"
        if not pairs.exists():
            continue
        agnostic_dir = split_dir / "agnostic"
        agnostic_dir.mkdir(parents=True, exist_ok=True)
        pose_dir = split_dir / "pose"
        pose_dir.mkdir(parents=True, exist_ok=True)
        for line in pairs.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            person_name, cloth_name = parts[0], parts[1]
            category = parts[2] if len(parts) > 2 else "t-shirt"
            stem = Path(person_name).stem
            target = agnostic_dir / f"{stem}.jpg"
            image_path = split_dir / "image" / person_name
            cloth_path = split_dir / "cloth" / cloth_name
            if target.exists() or not image_path.exists() or not cloth_path.exists():
                continue
            person = np.asarray(Image.open(image_path).convert("RGB"))
            garment = np.asarray(Image.open(cloth_path).convert("RGB"))
            mask_path = split_dir / "cloth-mask" / f"{Path(cloth_name).stem}.png"
            cloth_mask = None
            if mask_path.exists():
                cloth_mask = np.asarray(Image.open(mask_path).convert("L"))
            else:
                cloth_mask = cloth_mask_from_product_image(garment)
            kind = "lower" if category in {"bottom"} else "upper"
            try:
                sample = build_sample(person, garment, cloth_mask=cloth_mask, kind=kind, agnostic_mode="grey")
            except Exception as exc:  # pragma: no cover - corrupt input
                print(f"    · skipped {stem}: {exc}")
                continue
            Image.fromarray(sample["agnostic"]).save(target, quality=95)
            if not (pose_dir / f"{stem}.png").exists():
                Image.fromarray(sample["pose"]).save(pose_dir / f"{stem}.png")
            fixed += 1
    return fixed


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
