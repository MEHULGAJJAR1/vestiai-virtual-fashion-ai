"""Inference helper: load a trained checkpoint and run person + garment -> try-on image.

Used by ``scripts/inference.py`` for batch/CLI generation and by the API service for
single-image requests (through the adapter interface). Keeping this module free of FastAPI
imports means it can also be used from a notebook or a cloud worker.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from backend.ai.adapter import TryOnRequest, TryOnResult
from backend.ai.registry import build_adapter
from backend.models.checkpointing import CheckpointManager
from backend.utils.image_utils import load_image, save_image, stack_grid
from backend.utils.logging_utils import get_logger
from backend.utils.timing import Timer

logger = get_logger(__name__)

CATEGORY_PROMPTS = {
    "t-shirt": "a photo of a person wearing a t-shirt",
    "shirt": "a photo of a person wearing a button-up shirt",
    "jacket": "a photo of a person wearing a jacket",
    "kurta": "a photo of a person wearing a kurta",
    "dress": "a photo of a person wearing a dress",
    "top": "a photo of a person wearing a top",
    "bottom": "a photo of a person wearing trousers",
}


@dataclass
class InferenceConfig:
    """Options for a single inference call."""

    backend: str = "auto"                # auto | diffusion | lightweight
    checkpoint: Optional[str] = None     # explicit checkpoint dir (overrides auto-discovery)
    resolution: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 2.0
    seed: int = 42
    category: str = "t-shirt"
    prompt: Optional[str] = None
    device: str = "auto"


class VTONInference:
    """Thin, reusable wrapper around the adapter registry."""

    def __init__(self, config: Optional[InferenceConfig] = None) -> None:
        self.config = config or InferenceConfig()
        checkpoints = CheckpointManager(Path("checkpoints"))
        if self.config.checkpoint:
            resolved = checkpoints.resolve_for_inference(self.config.checkpoint)
            self.checkpoint_path = resolved or Path(self.config.checkpoint)
        else:
            self.checkpoint_path = checkpoints.resolve_for_inference("auto")
        self.adapter = build_adapter(
            self.config.backend,
            settings=None,
            checkpoint_manager=checkpoints,
        )
        logger.info(
            "Inference backend: %s (checkpoint: %s)", self.adapter.name,
            self.checkpoint_path or "none — geometric pipeline",
        )

    def try_on(
        self,
        person: Union[str, Path, np.ndarray],
        garment: Union[str, Path, np.ndarray],
        garment_mask: Optional[Union[str, Path, np.ndarray]] = None,
        output_path: Optional[str | Path] = None,
        save_comparison: bool = True,
    ) -> Dict[str, Any]:
        """Generate a try-on image and (optionally) save it with a comparison grid."""
        person_image = load_image(person, mode="RGB") if isinstance(person, (str, Path)) else np.asarray(person)
        garment_image = load_image(garment, mode="RGB") if isinstance(garment, (str, Path)) else np.asarray(garment)
        if garment_mask is None:
            mask_array = None
        elif isinstance(garment_mask, (str, Path)):
            mask_array = load_image(garment_mask, mode="L")
        else:
            mask_array = np.asarray(garment_mask)

        request = TryOnRequest(
            person_image=person_image,
            garment_image=garment_image,
            garment_mask=mask_array,
            category=self.config.category,
            prompt=self.config.prompt or CATEGORY_PROMPTS.get(self.config.category),
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            seed=self.config.seed,
            resolution=self.config.resolution,
        )
        with Timer() as timer:
            result: TryOnResult = self.adapter.try_on(request)

        payload: Dict[str, Any] = {
            "ok": result.ok,
            "backend": result.backend,
            "latency_ms": round(timer.ms, 1),
            "reason": result.reason,
            "warnings": result.warnings,
            "debug": result.debug,
        }
        if not result.ok or result.image is None:
            return payload

        if output_path is not None:
            output_path = Path(output_path)
            save_image(result.image, output_path)
            payload["output_path"] = str(output_path)
            if save_comparison:
                panels = [person_image, garment_image, result.image]
                comparison = stack_grid(panels, cols=3)
                comparison_path = output_path.with_name(output_path.stem + "_comparison.jpg")
                save_image(comparison, comparison_path, quality=92)
                payload["comparison_path"] = str(comparison_path)
            metadata_path = output_path.with_suffix(".json")
            metadata_path.write_text(json.dumps({
                "backend": result.backend, "latency_ms": payload["latency_ms"],
                "category": self.config.category, "resolution": self.config.resolution,
                "steps": self.config.num_inference_steps, "guidance": self.config.guidance_scale,
                "seed": self.config.seed, "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
                "debug": result.debug, "warnings": result.warnings, "created_at": time.time(),
            }, indent=2, default=str), encoding="utf-8")
            payload["metadata_path"] = str(metadata_path)
        else:
            payload["image"] = result.image
        return payload

    def batch(
        self,
        pairs: List[Dict[str, Any]],
        output_dir: str | Path = "results/batch",
    ) -> List[Dict[str, Any]]:
        """Run a list of ``{"person": path, "garment": path, "category": str, "name": str}``."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results: List[Dict[str, Any]] = []
        for item in pairs:
            name = item.get("name") or Path(str(item["person"])).stem
            try:
                payload = self._batch_one(item, output_dir, name)
                results.append(payload)
            except Exception as exc:
                logger.error("Batch item %s failed: %s", name, exc)
                results.append({"ok": False, "name": name, "reason": str(exc)})
        return results

    def _batch_one(self, item: Dict[str, Any], output_dir: Path, name: str) -> Dict[str, Any]:
        if item.get("category"):
            self.config.category = str(item["category"])
        return self.try_on(
            person=item["person"], garment=item["garment"],
            garment_mask=item.get("garment_mask"), output_path=output_dir / f"{name}_tryon.png",
        )

    def status(self) -> Dict[str, Any]:
        return {
            "adapter": self.adapter.status(),
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
            "config": self.config.__dict__,
        }
