"""Evaluation pipeline: metrics, comparison grids and training curves.

``scripts/evaluate.py`` calls this to answer "is the trained model actually any good?":

* runs the full trained pipeline on N held-out (person, garment) pairs,
* computes SSIM / PSNR / masked-L1 inside the garment region / colour-preservation score,
* writes side-by-side grids ``person | garment | agnostic | generated | ground truth``,
* aggregates a JSON summary (mean/median/std per metric) for the dashboard,
* renders training curves as SVG/PNG **without requiring matplotlib** (a tiny SVG writer is
  used when matplotlib is missing, so the artefact always exists).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.training.validate import compute_vton_metrics
from backend.utils.image_utils import load_image, save_image, stack_grid
from backend.utils.logging_utils import get_logger
from backend.utils.timing import Timer

logger = get_logger(__name__)


@dataclass
class EvaluationReport:
    """Aggregated evaluation results."""

    checkpoint: str
    split: str
    samples: int
    metrics: Dict[str, float] = field(default_factory=dict)
    per_sample: List[Dict[str, Any]] = field(default_factory=list)
    grid_paths: List[str] = field(default_factory=list)
    duration_s: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "split": self.split,
            "samples": self.samples,
            "metrics": {k: round(float(v), 5) for k, v in self.metrics.items()},
            "per_sample": self.per_sample,
            "grid_paths": self.grid_paths,
            "duration_s": round(self.duration_s, 1),
            "notes": self.notes,
            "created_at": time.time(),
            "created_at_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


def evaluate_checkpoint(
    checkpoint_dir: str | Path,
    dataset_root: str | Path = "datasets/samples",
    split: str = "test",
    resolution: int = 512,
    limit: int = 8,
    device: str = "auto",
    num_inference_steps: int = 25,
    guidance_scale: float = 2.0,
    output_dir: str | Path = "results/evaluation",
    seed: int = 0,
    base_model: str = "stable-diffusion-v1-5/stable-diffusion-inpainting",
) -> EvaluationReport:
    """Evaluate a trained checkpoint against a dataset split."""
    import torch
    from PIL import Image
    from diffusers import StableDiffusionControlNetInpaintPipeline

    from backend.datasets.dataset import VTONPairDataset
    from backend.models.vton_controlnet import VTONModelConfig, build_components, torch_dtype_for
    from backend.utils.device import detect_devices, empty_cuda_cache

    started = time.perf_counter()
    checkpoint_dir = Path(checkpoint_dir)
    output_root = Path(output_dir) / time.strftime("%Y%m%d-%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)

    devices = detect_devices()
    resolved_device = "cuda" if devices.cuda_available and device != "cpu" else "cpu"
    if resolved_device == "cpu":
        logger.warning("Evaluating on CPU — diffusion sampling will be slow (%s images).", limit)

    config = VTONModelConfig(base_model=base_model, resolution=resolution, mixed_precision="fp16")
    pipeline, controlnet, _scheduler = build_components(config, device=resolved_device, controlnet_path=checkpoint_dir)
    dtype = torch_dtype_for("fp16", resolved_device)
    inference_pipe = StableDiffusionControlNetInpaintPipeline(
        vae=pipeline.vae, text_encoder=pipeline.text_encoder, tokenizer=pipeline.tokenizer,
        unet=pipeline.unet, controlnet=controlnet, scheduler=pipeline.scheduler,
        safety_checker=None, feature_extractor=None, image_encoder=None,
    ).to(resolved_device)
    inference_pipe.set_progress_bar_config(disable=True)

    dataset = VTONPairDataset(dataset_root, split=split, resolution=resolution, augment=False, verbose=False)
    if len(dataset) == 0:
        raise FileNotFoundError(f"No samples found in {Path(dataset_root) / split}.")

    report = EvaluationReport(checkpoint=str(checkpoint_dir), split=split, samples=0)
    metric_values: Dict[str, List[float]] = {}
    grids: List[str] = []
    generator = torch.Generator(device="cpu").manual_seed(seed)

    for index in range(min(limit, len(dataset))):
        sample = dataset[index]
        try:
            with Timer() as timer:
                with torch.inference_mode():
                    output = inference_pipe(
                        prompt=str(sample["prompt"]),
                        image=_to_pil(sample["masked_image"]), mask_image=_to_pil(sample["inpaint_mask"][0], grayscale=True),
                        control_image=_to_pil(sample["control_image"]),
                        height=resolution, width=resolution,
                        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                        generator=generator, output_type="np",
                    )
            generated = np.clip(output.images[0] * 255, 0, 255).astype(np.uint8)
        except RuntimeError as exc:
            empty_cuda_cache()
            report.notes.append(f"Sample {index} failed: {exc}")
            logger.error("Inference failed for sample %s: %s", index, exc)
            continue

        ground_truth = _tensor_to_uint8(sample["pixel_values"])
        mask_np = (sample["inpaint_mask"][0].numpy() * 255).astype(np.uint8)
        garment_np = _tensor_to_uint8(sample["control_image"])
        metrics = compute_vton_metrics(generated, ground_truth, mask_np, garment_np)
        for key, value in metrics.items():
            if value == value:
                metric_values.setdefault(key, []).append(float(value))
        report.per_sample.append({
            "index": index,
            "identifier": sample.get("identifier", str(index)),
            "category": sample.get("category", "unknown"),
            "metrics": {k: round(float(v), 5) for k, v in metrics.items() if v == v},
            "latency_ms": round(timer.ms, 1),
        })

        panels = [
            _tensor_to_uint8(sample["pixel_values"]),
            garment_np,
            _tensor_to_uint8(sample["masked_image"]),
            generated,
            ground_truth,
        ]
        grid = stack_grid(panels, cols=5)
        grid_path = output_root / f"grid_{index:03d}.jpg"
        save_image(grid, grid_path, quality=92)
        grids.append(str(grid_path))
        report.samples += 1

    report.metrics = {key: float(np.mean(values)) for key, values in metric_values.items()}
    report.metrics.update({f"{key}_std": float(np.std(values)) for key, values in metric_values.items()})
    report.grid_paths = grids
    report.duration_s = time.perf_counter() - started

    if grids:
        summary_grid = stack_grid([load_image(path) for path in grids[:4]], cols=1)
        save_image(summary_grid, output_root / "summary_grid.jpg", quality=90)
        report.grid_paths.append(str(output_root / "summary_grid.jpg"))

    (output_root / "evaluation.json").write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    logger.info("Evaluation complete: %s samples, metrics=%s", report.samples, report.metrics)
    return report


def _to_pil(tensor, grayscale: bool = False):
    from PIL import Image

    array = _tensor_to_uint8(tensor)
    if grayscale:
        return Image.fromarray(array[..., 0], mode="L")
    return Image.fromarray(array, mode="RGB")


def _tensor_to_uint8(tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = array.transpose(1, 2, 0)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.min() < -0.01:
        array = (array + 1.0) / 2.0
    return (np.clip(array, 0, 1) * 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
# Training curves
# --------------------------------------------------------------------------------------
def plot_training_curves(
    history: Sequence[Dict[str, Any]],
    output_path: str | Path = "results/training_curves.png",
    title: str = "VestiAI · training curves",
) -> Path:
    """Render loss/learning-rate curves.

    Uses matplotlib when installed; otherwise writes a self-contained SVG so the artefact
    always exists (the dashboard can display either).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(item.get("epoch", index + 1)) for index, item in enumerate(history)]
    series = {
        "train_loss": [float(item.get("train_loss")) for item in history if item.get("train_loss") is not None],
        "val_loss": [float(item["val_loss"]) for item in history if item.get("val_loss") is not None],
        "ssim": [float(item["ssim"]) for item in history if item.get("ssim") is not None],
        "psnr": [float(item["psnr"]) for item in history if item.get("psnr") is not None],
    }
    if not any(series.values()):
        logger.warning("No metrics to plot.")
        return output_path

    try:  # pragma: no cover - optional dependency
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(epochs[: len(series["train_loss"])], series["train_loss"], marker="o", label="train loss")
        if series["val_loss"]:
            axes[0].plot(epochs[: len(series["val_loss"])], series["val_loss"], marker="s", label="val loss")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].set_title("Loss")
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        if series["ssim"]:
            axes[1].plot(epochs[: len(series["ssim"])], series["ssim"], marker="o", color="#2a9d8f", label="SSIM")
        if series["psnr"]:
            twin = axes[1].twinx()
            twin.plot(epochs[: len(series["psnr"])], series["psnr"], marker="^", color="#e76f51", label="PSNR")
            twin.set_ylabel("PSNR (dB)")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("SSIM")
        axes[1].set_title("Validation quality")
        axes[1].grid(alpha=0.3)
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(output_path.with_suffix(".png"), dpi=140)
        plt.close(figure)
        return output_path.with_suffix(".png")
    except Exception as exc:
        logger.info("matplotlib unavailable (%s); writing an SVG chart instead.", exc)

    svg_path = output_path.with_suffix(".svg")
    svg_path.write_text(_svg_curves(epochs, series, title), encoding="utf-8")
    return svg_path


def _svg_curves(epochs: List[int], series: Dict[str, List[float]], title: str, width: int = 820, height: int = 320) -> str:
    """Minimal dependency-free SVG line chart with two panels."""
    def panel(x: int, y: int, w: int, h: int, keys: List[str], colors: Dict[str, str], caption: str) -> str:
        all_values = [v for key in keys for v in series.get(key, [])]
        if not all_values:
            return ""
        low, high = min(all_values), max(all_values)
        span = (high - low) or 1.0
        parts = [f'<text x="{x + w / 2}" y="{y - 8}" text-anchor="middle" fill="#e6e6e6" font-size="13" font-family="system-ui">{caption}</text>']
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="#3a3a44"/>')
        for key in keys:
            values = series.get(key, [])
            if len(values) < 1:
                continue
            points = []
            for index, value in enumerate(values):
                px = x + (index / max(1, len(values) - 1)) * w
                py = y + h - ((value - low) / span) * (h - 12) - 6
                points.append(f"{px:.1f},{py:.1f}")
            parts.append(f'<polyline fill="none" stroke="{colors.get(key, "#7aa2f7")}" stroke-width="2" points="{" ".join(points)}"/>')
        for index, key in enumerate(keys):
            if key in series and series[key]:
                colour = colors.get(key, "#7aa2f7")
                parts.append(f'<rect x="{x + 8}" y="{y + 10 + index * 18}" width="10" height="10" fill="{colour}"/>')
                parts.append(f'<text x="{x + 24}" y="{y + 19 + index * 18}" fill="#c9c9d1" font-size="11" font-family="system-ui">{key}</text>')
        return "".join(parts)

    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#12121a"/>',
        f'<text x="{width / 2}" y="24" text-anchor="middle" fill="#ffffff" font-size="15" font-family="system-ui">{title}</text>',
        panel(40, 60, 340, 200, ["train_loss", "val_loss"], {"train_loss": "#7aa2f7", "val_loss": "#f7768e"}, "Loss"),
        panel(440, 60, 340, 200, ["ssim", "psnr"], {"ssim": "#9ece6a", "psnr": "#e0af68"}, "Validation quality"),
        "</svg>",
    ]
    return "".join(body)


def load_evaluations(directory: str | Path = "results/evaluation") -> List[Dict[str, Any]]:
    """Read all evaluation summaries (newest first)."""
    root = Path(directory)
    if not root.exists():
        return []
    reports: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*/evaluation.json")):
        try:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:  # pragma: no cover
            continue
    reports.sort(key=lambda item: item.get("created_at", 0), reverse=True)
    return reports
