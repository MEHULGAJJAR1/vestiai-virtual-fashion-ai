#!/usr/bin/env python
"""VestiAI — evaluate a checkpoint on a test split.

Produces, in ``results/evaluation/<timestamp>/``:

* per-sample side-by-side images (person | garment | agnostic | generated | ground truth)
* ``report.json`` — VTON metrics (SSIM, PSNR, masked-L1, identity similarity, garment
  colour/texture preservation where applicable) plus the inference settings used
* ``report.md`` — the same numbers as a human-readable table you can paste into a PR

and refreshes ``results/training_curves.png`` from the recorded history.

    python scripts/evaluate.py --checkpoint checkpoints/best_model
    python scripts/evaluate.py --checkpoint checkpoints/best_model --dataset datasets/viton_hd --split test --limit 32
    python scripts/evaluate.py --checkpoint checkpoints/best_model --backend lightweight   # no GPU, no diffusion

The ``lightweight`` backend lets you evaluate the geometric pipeline on the same data so the
two pipelines can be compared honestly in the same report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from _common import (  # noqa: E402
    banner, fail, human_time, kv_table, load_settings, print_json, resolve_dataset, setup_logging,
    add_common_io_args,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a VestiAI checkpoint (or the lightweight pipeline) on a dataset split.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", default=None, help="checkpoint folder (default: auto)")
    p.add_argument("--dataset", default=None, help="dataset name/path (default: the checkpoint's training set)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=8, help="number of samples to evaluate")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--steps", type=int, default=25, help="diffusion steps per sample")
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", default="diffusion", choices=["diffusion", "lightweight"],
                   help="which pipeline to evaluate")
    p.add_argument("--output", default="results/evaluation")
    p.add_argument("--curves", action="store_true", help="also refresh results/training_curves.png")
    p.add_argument("--json", action="store_true")
    add_common_io_args(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    settings = load_settings(args.config)

    if args.backend == "lightweight":
        return evaluate_lightweight(args, settings)

    from _common import require_checkpoint, require_torch

    torch = require_torch("diffusion evaluation")           # noqa: F841
    checkpoint = require_checkpoint(args.checkpoint, "Evaluation")

    from backend.models.vton_controlnet import describe_checkpoint
    from backend.training.evaluate import evaluate_checkpoint

    info = describe_checkpoint(checkpoint)
    card = info.get("model_card") or {}
    dataset_requested = args.dataset or card.get("dataset") or "datasets/samples"
    dataset = resolve_dataset(settings, dataset_requested)
    if dataset is None:
        fail(f"dataset '{dataset_requested}' not found — pass --dataset explicitly")

    banner("VestiAI · evaluation", f"{checkpoint}  ·  {dataset} [{args.split}]")
    kv_table({
        "samples": args.limit,
        "resolution": args.resolution,
        "steps": args.steps,
        "guidance": args.guidance,
        "base model": card.get("base_model") or settings.vton_base_pipeline,
        "backend": "diffusion",
    })
    print()

    started = time.time()
    report = evaluate_checkpoint(
        checkpoint_dir=checkpoint,
        dataset_root=dataset,
        split=args.split,
        resolution=args.resolution,
        limit=args.limit,
        device=settings.resolve_device(),
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        output_dir=args.output,
        seed=args.seed,
        base_model=card.get("base_model") or settings.vton_base_pipeline,
    )
    elapsed = time.time() - started

    payload = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    metrics = payload.get("metrics") or {}
    banner("metrics")
    kv_table({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()})
    kv_table({
        "samples": payload.get("samples"),
        "output dir": payload.get("output_dir"),
        "elapsed": human_time(elapsed),
        "seconds per sample": round(elapsed / max(1, args.limit), 2),
    })

    write_markdown(Path(args.output), payload, checkpoint, dataset, args)

    if args.curves:
        from backend.training.evaluate import plot_training_curves
        from backend.training.tracker import load_history

        epochs = load_epoch_history(settings.log_dir)
        if epochs:
            path = plot_training_curves(epochs, Path(settings.results_dir) / "training_curves.png")
            print(f"\n  refreshed curves: {path}")
        else:
            print("\n  (no training history found — run a training job first)")

    if args.json:
        print_json(payload)
    print(f"\n  report: {Path(args.output) / 'report.md'}\n")
    return 0


def evaluate_lightweight(args, settings) -> int:
    """Evaluate the geometric pipeline on the same split so both can be compared."""
    import numpy as np
    from PIL import Image

    from backend.ai.lightweight import LightweightTryOnAdapter
    from backend.datasets.preprocessing import find_dataset_root
    from backend.training.validate import compute_vton_metrics
    from backend.utils.image_utils import load_image

    root = resolve_dataset(settings, args.dataset) or find_dataset_root()
    if root is None:
        fail("no dataset found — pass --dataset")
    split_dir = Path(root) / args.split
    if not split_dir.exists():
        fail(f"split folder not found: {split_dir}")

    banner("VestiAI · evaluation (lightweight pipeline)", f"{split_dir}")
    adapter = LightweightTryOnAdapter()
    from backend.ai.adapter import TryOnRequest

    pairs = []
    pairs_file = split_dir / "pairs.txt"
    if pairs_file.exists():
        for line in pairs_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                pairs.append(parts)
    if not pairs:
        fail(f"no pairs.txt in {split_dir}")

    out_dir = Path(args.output) / f"lightweight-{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for index, parts in enumerate(pairs[: args.limit], start=1):
        person_name, cloth_name = parts[0], parts[1]
        category = parts[2] if len(parts) > 2 else "t-shirt"
        person_path = split_dir / "image" / person_name
        cloth_path = split_dir / "cloth" / cloth_name
        if not person_path.exists() or not cloth_path.exists():
            continue
        person = load_image(person_path)
        garment = load_image(cloth_path)
        try:
            result = adapter.try_on(TryOnRequest(person_image=person, garment_image=garment, category=category))
        except Exception as exc:
            print(f"  [{index}] {person_name} ✗ {exc}")
            continue
        if not result.ok:
            print(f"  [{index}] {person_name} ✗ {result.reason}")
            continue
        comparison = np.concatenate([person, np.asarray(Image.fromarray(garment).resize((person.shape[1], person.shape[0])))], axis=1)
        Image.fromarray(comparison).save(out_dir / f"{Path(person_name).stem}_lightweight.jpg", quality=88)
        results.append({
            "person": person_name, "cloth": cloth_name, "category": category,
            "latency_ms": round(result.latency_ms, 1),
            "coverage": (result.debug or {}).get("coverage"),
        })
        print(f"  [{index}/{min(args.limit, len(pairs))}] {person_name} ✓ {result.latency_ms:.0f} ms")

    summary = {
        "backend": "lightweight",
        "samples": len(results),
        "mean_latency_ms": round(sum(r["latency_ms"] for r in results) / max(1, len(results)), 1),
        "output_dir": str(out_dir),
        "note": "The lightweight pipeline warps the real garment texture onto the person; it does not "
                "synthesise fabric or lighting, so pixel metrics against a ground-truth photo are not "
                "meaningful — compare it visually and by latency, not by SSIM.",
    }
    banner("summary")
    kv_table(summary)
    (out_dir / "report.json").write_text(json.dumps({"summary": summary, "samples": results}, indent=2), encoding="utf-8")
    if args.json:
        print_json(summary)
    print()
    return 0


def load_epoch_history(log_dir: Path) -> list[dict]:
    """Per-epoch metric history, from logs/training_status.json or the newest metrics.jsonl.

    ``plot_training_curves`` expects one dict per epoch with the metric names as keys — the
    same structure the training service keeps for the dashboard.
    """
    import json as _json

    status_file = Path(log_dir) / "training_status.json"
    if status_file.exists():
        try:
            payload = _json.loads(status_file.read_text(encoding="utf-8"))
            epochs = (payload.get("history") or {}).get("epochs")
            if epochs:
                return list(epochs)
        except Exception:
            pass

    candidates = sorted(Path(log_dir).glob("training_*/metrics.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        return []
    try:
        from backend.training.tracker import load_history

        curves = load_history(candidates[0])
    except Exception:
        return []
    # Fold the per-step curves into one row per epoch (last value wins).
    by_epoch: dict[int, dict] = {}
    for name, points in curves.items():
        for point in points:
            epoch = int(point.get("epoch") or 0)
            by_epoch.setdefault(epoch, {"epoch": epoch})[name] = float(point.get("value", 0.0))
    return [by_epoch[key] for key in sorted(by_epoch)]



def write_markdown(output_root: Path, payload: dict, checkpoint: Path, dataset: Path, args) -> None:
    reports = sorted(Path(output_root).glob("*/report.json"))
    target_dir = Path(payload.get("output_dir") or (reports[-1].parent if reports else output_root))
    target_dir.mkdir(parents=True, exist_ok=True)
    metrics = payload.get("metrics") or {}
    lines = [
        "# VestiAI evaluation report",
        "",
        f"- **checkpoint**: `{checkpoint}`",
        f"- **dataset**: `{dataset}` (split: `{args.split}`)",
        f"- **samples**: {payload.get('samples', args.limit)}",
        f"- **settings**: {args.resolution}px · {args.steps} steps · guidance {args.guidance} · seed {args.seed}",
        "",
        "| metric | value | what it tells you |",
        "| --- | --- | --- |",
    ]
    explanations = {
        "ssim": "structural similarity inside the garment mask — overall fidelity",
        "psnr": "peak signal-to-noise ratio inside the mask — pixel accuracy",
        "masked_l1": "mean absolute error inside the mask — lower is better",
        "identity_similarity": "person identity preserved outside the garment region",
        "garment_preservation": "colour/texture agreement with the product photo",
    }
    for key, value in metrics.items():
        rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"| `{key}` | {rendered} | {explanations.get(key, '')} |")
    lines += [
        "",
        "> Metrics are computed **inside the garment mask** unless stated otherwise: that is the",
        "> region the model is responsible for, and it avoids inflating scores with unchanged",
        "> background pixels.",
        "",
        f"Side-by-side samples: `{target_dir}`",
    ]
    (target_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
