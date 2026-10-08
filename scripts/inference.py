#!/usr/bin/env python
"""VestiAI — command-line try-on inference.

Person photo + garment photo (or a garment already in the closet) → a try-on image, plus a
side-by-side comparison sheet. Works with either pipeline:

* ``--backend lightweight`` — geometric warp: 10–40 ms, no GPU, preserves the real garment
  texture, does not synthesise folds/lighting.
* ``--backend diffusion``   — the trained model: needs a checkpoint and the ML extra.
* ``--backend auto``        — diffusion when a checkpoint is installed, otherwise lightweight
  (and the CLI tells you which one answered).

    python scripts/inference.py --person me.jpg --garment tee.jpg --output results/me_tee.png
    python scripts/inference.py --person me.jpg --garment-key sample-tee-1-t-shirt-ab12cd
    python scripts/inference.py --person me.jpg --garment tee.jpg --backend diffusion --steps 30
    python scripts/inference.py --batch pairs.csv --output-dir results/batch
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from _common import (  # noqa: E402
    add_common_io_args, banner, fail, human_time, kv_table, load_settings, print_json,
    setup_logging,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run virtual try-on from the command line.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = p.add_mutually_exclusive_group()
    source.add_argument("--garment", help="garment image file")
    source.add_argument("--garment-key", help="key of a garment already in the closet")
    p.add_argument("--person", required=True, help="person photo")
    p.add_argument("--output", default=None, help="write the try-on image here (default: results/tryon.png)")

    p.add_argument("--batch", default=None, help="CSV of person,garment[,category] rows")
    p.add_argument("--output-dir", default="results/batch", help="folder used with --batch")

    p.add_argument("--backend", default="auto", choices=["auto", "diffusion", "lightweight"])
    p.add_argument("--checkpoint", default=None, help="checkpoint folder (diffusion backend)")
    p.add_argument("--category", default=None, help="garment category (default: auto-detect or the closet record)")
    p.add_argument("--resolution", type=int, default=512, choices=[256, 512, 768, 1024])
    p.add_argument("--steps", type=int, default=30, help="diffusion steps")
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask", default=None, help="optional garment mask PNG (alpha or L)")
    p.add_argument("--no-comparison", action="store_true", help="skip the comparison sheet")
    p.add_argument("--json", action="store_true")
    add_common_io_args(p)
    return p


def resolve_garment(args, settings):
    """Return ``(rgb, alpha, category, label, key)`` for the requested garment."""
    import numpy as np

    from backend.utils.image_utils import load_image

    if args.garment_key:
        from backend.services.garment_service import ClosetStore

        closet = ClosetStore(settings.garments_dir, settings.closet_db)
        record = closet.maybe_get(args.garment_key)
        if record is None:
            fail(f"garment '{args.garment_key}' is not in the closet (see GET /api/garments)")
        rgb = load_image(record.image_path)
        alpha = load_image(record.mask_path, mode="L") if Path(record.mask_path).exists() else None
        return np.asarray(rgb), (np.asarray(alpha) if alpha is not None else None), record.category, record.label, record.key

    if not args.garment:
        fail("provide --garment <image> or --garment-key <key>")
    path = Path(args.garment).expanduser()
    if not path.exists():
        fail(f"garment image not found: {path}")
    rgb = np.asarray(load_image(path))
    alpha = None
    if args.mask:
        if not Path(args.mask).exists():
            fail(f"mask not found: {args.mask}")
        alpha = np.asarray(load_image(args.mask, mode="L"))
    return rgb, alpha, args.category or "", path.stem, None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    settings = load_settings(args.config)

    if args.batch:
        return run_batch(args, settings)

    from backend.training.inference import InferenceConfig, VTONInference

    garment_rgb, garment_alpha, category, label, key = resolve_garment(args, settings)
    config = InferenceConfig(
        backend=args.backend,
        checkpoint=args.checkpoint,
        resolution=args.resolution,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        category=category or args.category or "t-shirt",
        device=settings.resolve_device(),
    )

    banner("VestiAI · try-on inference", f"{Path(args.person).name} + {label}")
    kv_table({
        "backend requested": args.backend,
        "category": config.category,
        "resolution": config.resolution,
        "steps": config.steps if hasattr(config, "steps") else args.steps,
        "device": config.device,
    })
    print()

    started = time.time()
    engine = VTONInference(config)
    output = Path(args.output) if args.output else Path(settings.results_dir) / "tryon.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    result = engine.try_on(
        person=args.person,
        garment=garment_rgb,
        garment_mask=garment_alpha,
        output_path=output,
        save_comparison=not args.no_comparison,
    )
    elapsed = time.time() - started

    if not result.get("ok"):
        print(f"✗ inference failed: {result.get('reason') or result.get('error')}")
        print("  the real-time (lightweight) pipeline is unaffected — try --backend lightweight.\n")
        return 1

    banner("result")
    kv_table({
        "backend": result.get("backend"),
        "output": result.get("output_path", str(output)),
        "comparison": result.get("comparison_path", "—"),
        "inference ms": round(result.get("latency_ms", elapsed * 1000), 1),
        "wall clock": human_time(elapsed),
        "category": result.get("category", config.category),
    })
    for warning in result.get("warnings") or []:
        print(f"  ⚠ {warning}")
    if args.json:
        print_json(result)
    print()
    return 0


def run_batch(args, settings) -> int:
    """Try every row of a CSV through the same engine (one model load, many images)."""
    from backend.training.inference import InferenceConfig, VTONInference
    from backend.utils.image_utils import load_image

    csv_path = Path(args.batch).expanduser()
    if not csv_path.exists():
        fail(f"batch CSV not found: {csv_path}")

    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0].strip().lower() in {"person", "person_path", "#"}:
                continue
            if len(row) >= 2:
                rows.append({
                    "person": row[0].strip(),
                    "garment": row[1].strip(),
                    "category": row[2].strip() if len(row) > 2 and row[2].strip() else "t-shirt",
                })
    if not rows:
        fail(f"no usable rows in {csv_path}")

    config = InferenceConfig(
        backend=args.backend, checkpoint=args.checkpoint, resolution=args.resolution,
        num_inference_steps=args.steps, guidance_scale=args.guidance, seed=args.seed,
        device=settings.resolve_device(),
    )
    engine = VTONInference(config)
    banner("VestiAI · batch try-on", f"{len(rows)} pair(s) → {args.output_dir}")

    pairs = []
    for row in rows:
        person, garment = Path(row["person"]).expanduser(), Path(row["garment"]).expanduser()
        if not person.exists() or not garment.exists():
            print(f"  ✗ missing file: {row['person']} / {row['garment']}")
            continue
        pairs.append({
            "person": load_image(person),
            "garment": load_image(garment),
            "category": row["category"],
            "name": f"{person.stem}__{garment.stem}",
        })

    started = time.time()
    results = engine.batch(pairs, output_dir=args.output_dir)
    elapsed = time.time() - started

    ok = [item for item in results if item.get("ok")]
    banner("summary")
    kv_table({
        "requested": len(pairs),
        "succeeded": len(ok),
        "failed": len(results) - len(ok),
        "wall clock": human_time(elapsed),
        "mean ms": round(sum(item.get("latency_ms", 0) for item in ok) / max(1, len(ok)), 1),
        "output dir": str(args.output_dir),
    })
    if args.json:
        print_json(results)
    print()
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
