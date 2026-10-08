#!/usr/bin/env python
"""VestiAI — batch garment preprocessing from the command line.

Runs the *exact* pipeline the API uses on a folder of garment photos, so a catalogue can be
prepared without opening the browser:

    decode → background removal → alpha refinement → quality gate → category
    classification → square canvas + mask → palette → closet record

    python scripts/preprocess_garments.py --input ~/photos/tees --category t-shirt
    python scripts/preprocess_garments.py --input ./catalogue --output datasets/tees
    python scripts/preprocess_garments.py --input ./catalogue --force --report rejects.json

Every accepted image is added to **My Closet** (that is what the ingest pipeline does — it is
the same code path as the upload button) and, with ``--output``, also copied into
``<output>/<key>/{garment.png,mask.png,cutout.png,preview.jpg,meta.json}`` so the folder can be
used as a garment source for dataset preparation.

Rejected images are reported with the reason the quality gate gave (resolution, skin ratio,
occlusion) — nothing is silently skipped. ``--force`` bypasses the gate for images you have
already checked by hand.

Only use images you are allowed to use: VestiAI never downloads anything from shopping sites.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from _common import (  # noqa: E402
    add_common_io_args, banner, fail, human_time, kv_table, load_settings, print_json, setup_logging,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
EXPORT_FILES = {
    "image_path": "garment.png",
    "mask_path": "mask.png",
    "cutout_path": "cutout.png",
    "preview_path": "preview.jpg",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preprocess a folder of garment images (background removal, mask, category).",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, help="folder containing garment product photos")
    p.add_argument("--output", default=None, help="folder to write processed garments into")
    p.add_argument("--category", default=None, help="force a category instead of auto-detecting it")
    p.add_argument("--force", action="store_true", help="skip the quality gate (accept everything decodable)")
    p.add_argument("--limit", type=int, default=0, help="stop after N images (0 = all)")
    p.add_argument("--report", default=None, help="write a JSON report of accepted/rejected images here")
    p.add_argument("--json", action="store_true")
    add_common_io_args(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    settings = load_settings(args.config)

    source = Path(args.input).expanduser()
    if not source.exists():
        fail(f"input folder not found: {source}")
    files = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        fail(f"no images ({', '.join(sorted(IMAGE_SUFFIXES))}) found under {source}")
    if args.limit:
        files = files[: args.limit]

    output = Path(args.output).expanduser() if args.output else None
    if output:
        output.mkdir(parents=True, exist_ok=True)

    from backend.services.garment_service import ClosetStore, GarmentService

    closet = ClosetStore(settings.garments_dir, settings.closet_db)
    service = GarmentService(
        closet,
        uploads_dir=settings.uploads_dir,
        canvas=settings.target_garment_canvas,
        min_resolution=settings.garment_min_resolution,
        max_side=settings.garment_max_side,
        background_removal=settings.background_removal,
        enable_zero_shot=bool(settings.extra.get("enable_zero_shot_classifier", False)),
        device=settings.device,
        classifier_head=settings.extra.get("classifier_head") or (settings.models_cache_dir / "garment_classifier.joblib"),
    )

    banner("VestiAI · garment preprocessing", f"{len(files)} image(s) from {source}")
    kv_table({
        "background removal": settings.background_removal,
        "canvas": f"{settings.target_garment_canvas}px",
        "quality gate": "disabled (--force)" if args.force else "enabled",
        "closet library": f"{settings.garments_dir} (each accepted image is added)",
        "dataset export": str(output) if output else "(not written — pass --output)",
    })
    print()

    accepted: list[dict] = []
    rejected: list[dict] = []
    started = time.time()

    for index, path in enumerate(files, start=1):
        prefix = f"  [{index:>3}/{len(files)}] {path.name[:42]:<42}"
        try:
            record = service.ingest(
                path.read_bytes(),
                filename=path.name,
                category_hint=args.category,
                skip_quality_gate=args.force,
                source={"kind": "cli", "folder": str(source), "script": "preprocess_garments.py"},
            )
        except Exception as exc:
            rejected.append({
                "file": str(path),
                "error": exc.__class__.__name__,
                "reason": getattr(exc, "user_message", None) or str(exc),
                "details": getattr(exc, "details", {}) if hasattr(exc, "details") else {},
            })
            print(f"{prefix} ✗ {getattr(exc, 'user_message', None) or exc}")
            continue

        if output:
            export_one(output, path, record)

        classification = record.get("classification") or {}
        quality = record.get("quality") or {}
        accepted.append({
            "file": str(path),
            "key": record.get("key"),
            "label": record.get("label"),
            "category": record.get("category"),
            "category_label": record.get("category_label"),
            "garment_type": record.get("garment_type"),
            "confidence": round(float(classification.get("confidence") or 0.0), 3),
            "method": classification.get("method"),
            "quality_score": quality.get("score"),
            "background_removal": quality.get("background_removal"),
            "milliseconds": record.get("ingest_ms"),
        })
        confidence = float(classification.get("confidence") or 0.0)
        print(f"{prefix} ✓ {str(record.get('category')):<12} {confidence:.2f}  {record.get('key')}")

    elapsed = time.time() - started
    summary = {
        "input": str(source),
        "output": str(output) if output else None,
        "total": len(files),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "elapsed_s": round(elapsed, 2),
        "per_image_ms": round(elapsed * 1000 / max(1, len(files)), 1),
    }

    print()
    banner("summary")
    kv_table({**summary, "elapsed": human_time(elapsed)})
    if accepted:
        by_category: dict[str, int] = {}
        for item in accepted:
            by_category[item["category"] or "unknown"] = by_category.get(item["category"] or "unknown", 0) + 1
        kv_table({"by category": by_category})
        print("\n  accepted garments are in My Closet — open the app, or list them with:")
        print("    python scripts/dataset.py --list        # dataset-side view\n")
    if rejected:
        print("  rejected (and why):")
        for item in rejected[:15]:
            print(f"    ✗ {Path(item['file']).name}: {item['reason'][:110]}")
        print("    tip: --force accepts them anyway, or fix the source image.\n")

    report = {"summary": summary, "accepted": accepted, "rejected": rejected}
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"  report written to {args.report}")
    if args.json:
        print_json(report)

    print(f"\n  elapsed: {human_time(elapsed)}\n")
    return 0 if accepted else 1


def export_one(output: Path, source_path: Path, record: dict) -> None:
    """Copy the processed files for one garment into the output folder (dataset-style)."""
    folder = Path(output) / str(record.get("key") or source_path.stem)
    folder.mkdir(parents=True, exist_ok=True)
    files = record.get("files") or {}
    for field, name in EXPORT_FILES.items():
        source_file = files.get(field)
        if source_file and Path(source_file).exists():
            shutil.copy2(source_file, folder / name)
    # Drop the convenience base64 blobs: the real files are copied right next to meta.json.
    payload = {k: v for k, v in record.items() if k != "thumbnail" and not k.endswith("_data_uri")}
    (folder / "meta.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
