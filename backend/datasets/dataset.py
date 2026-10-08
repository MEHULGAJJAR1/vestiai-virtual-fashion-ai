"""Training dataset for the VTON diffusion adapter.

Returns, per sample, everything the diffusion trainer needs:

===================  ==========================================================
Key                  Shape / dtype          Meaning
===================  ==========================================================
``pixel_values``     (3, R, R) float32 [-1,1]  ground-truth person (target latent source)
``masked_image``     (3, R, R) float32 [-1,1]  agnostic person (garment region erased)
``inpaint_mask``     (1, R, R) float32 [0,1]   region the model must synthesise (dilated)
``control_image``    (3, R, R) float32 [-1,1]  the garment (ControlNet condition)
``cloth``            (3, 256, 256) [-1,1]      garment crop for the CLIP consistency loss
``prompt``           str                       text prompt built from the category
``original_size``    tuple                     (w, h) for VAE round-trip bookkeeping
===================  ==========================================================

The loader is robust: it indexes ``pairs.txt`` when present, otherwise it pairs files by
stem, skips broken samples with a warning (never crashing a multi-hour run on one bad JPEG),
and can synthesise the agnostic image on the fly when a dataset ships without it.
"""

from __future__ import annotations

import json
import random
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from backend.cv import keypoints as kp
from backend.cv.pose import PoseEstimator
from backend.datasets.preprocessing import (
    AugmentConfig, augment_sample, garment_mask_from_alpha, person_garment_mask,
)
from backend.utils.image_utils import load_image
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

CATEGORY_PROMPTS = {
    "t-shirt": "a photo of a person wearing a t-shirt",
    "shirt": "a photo of a person wearing a button-up shirt",
    "jacket": "a photo of a person wearing a jacket",
    "kurta": "a photo of a person wearing a kurta",
    "dress": "a photo of a person wearing a dress",
    "top": "a photo of a person wearing a top",
    "sweater": "a photo of a person wearing a sweater",
    "traditional": "a photo of a person wearing traditional clothing",
    "formal": "a photo of a person wearing formal clothing",
    "bottom": "a photo of a person wearing trousers",
    "upper": "a photo of a person wearing an upper-body garment",
    "lower": "a photo of a person wearing lower-body clothing",
    "unknown": "a photo of a person wearing a garment",
}


@dataclass
class SampleRecord:
    """One indexed training sample."""

    identifier: str
    person_path: Path
    cloth_path: Path
    cloth_mask_path: Optional[Path]
    agnostic_path: Optional[Path]
    pose_path: Optional[Path]
    category: str = "upper"


class VTONPairDataset:
    """PyTorch dataset over the canonical VestiAI / VITON-HD layout.

    Parameters
    ----------
    root, split:
        Dataset root and which split folder to read.
    resolution:
        Square training resolution (256/512/768/1024). Must be divisible by 8.
    augment:
        Enable augmentation (train only).
    max_samples:
        Truncate the split — this is what ``QUICK_DEMO`` mode uses.
    build_missing:
        Compute agnostic/pose/mask on the fly when the dataset does not ship them.
    """

    def __init__(
        self,
        root: str | Path = "datasets/samples",
        split: str = "train",
        resolution: int = 512,
        augment: bool = True,
        max_samples: Optional[int] = None,
        seed: int = 42,
        augment_config: Optional[AugmentConfig] = None,
        build_missing: bool = True,
        prompt_template: Optional[str] = None,
        mask_dilate: int = 9,
        skip_broken: bool = True,
        verbose: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.resolution = int(resolution) - (int(resolution) % 8)
        if self.resolution != int(resolution):
            logger.warning("Resolution %s is not divisible by 8; using %s.", resolution, self.resolution)
        self.augment = augment
        self.max_samples = max_samples
        self.seed = seed
        self.augment_config = augment_config or AugmentConfig()
        self.build_missing = build_missing
        self.prompt_template = prompt_template
        self.mask_dilate = int(mask_dilate)
        self.skip_broken = skip_broken

        self.split_dir = self.root / split
        self.records: List[SampleRecord] = []
        self._broken: List[str] = []
        self._pose_estimator: Optional[PoseEstimator] = None
        self._rng = random.Random(seed)
        self._index()
        if verbose:
            logger.info(
                "Dataset %s/%s: %d samples at %dpx (augment=%s)", self.root.name, split, len(self.records), self.resolution, augment,
            )

    # ---------------------------------------------------------------------------------
    # indexing
    # ---------------------------------------------------------------------------------
    def _index(self) -> None:
        if not self.split_dir.exists():
            logger.warning("Split folder %s does not exist — dataset is empty.", self.split_dir)
            return
        pairs_path = self.split_dir / "pairs.txt"
        image_dir = self.split_dir / "image"
        cloth_dir = self.split_dir / "cloth"
        mask_dir = self.split_dir / "cloth-mask"
        agnostic_dir = self.split_dir / "agnostic"
        pose_dir = self.split_dir / "pose"

        if pairs_path.exists():
            for line in pairs_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 2:
                    continue
                person_name, cloth_name = parts[0], parts[1]
                category = parts[2] if len(parts) > 2 else "upper"
                record = self._make_record(person_name, cloth_name, category, image_dir, cloth_dir, mask_dir, agnostic_dir, pose_dir)
                if record:
                    self.records.append(record)
        else:
            cloth_lookup = {path.stem: path for path in sorted(cloth_dir.glob("*.*"))} if cloth_dir.exists() else {}
            for person_path in sorted(image_dir.glob("*.*")) if image_dir.exists() else []:
                stem = person_path.stem
                cloth_path = cloth_lookup.get(stem) or (next(iter(cloth_lookup.values())) if cloth_lookup else None)
                if cloth_path is None:
                    continue
                record = self._make_record(person_path.name, cloth_path.name, "upper", image_dir, cloth_dir, mask_dir, agnostic_dir, pose_dir)
                if record:
                    self.records.append(record)

        if self.max_samples is not None and self.max_samples > 0 and len(self.records) > self.max_samples:
            rng = random.Random(self.seed)
            self.records = rng.sample(self.records, self.max_samples)
            logger.info("Truncated split '%s' to %d samples (max_samples).", self.split, len(self.records))

    def _make_record(
        self,
        person_name: str,
        cloth_name: str,
        category: str,
        image_dir: Path,
        cloth_dir: Path,
        mask_dir: Path,
        agnostic_dir: Path,
        pose_dir: Path,
    ) -> Optional[SampleRecord]:
        person_path = image_dir / person_name
        cloth_path = cloth_dir / cloth_name
        if not person_path.exists() or not cloth_path.exists():
            self._broken.append(f"missing files: {person_name} / {cloth_name}")
            return None
        mask_path = mask_dir / cloth_name
        agnostic_path = agnostic_dir / person_name
        pose_path = pose_dir / (Path(person_name).stem + ".png")
        return SampleRecord(
            identifier=Path(person_name).stem,
            person_path=person_path,
            cloth_path=cloth_path,
            cloth_mask_path=mask_path if mask_path.exists() else None,
            agnostic_path=agnostic_path if agnostic_path.exists() else None,
            pose_path=pose_path if pose_path.exists() else None,
            category=category,
        )

    # ---------------------------------------------------------------------------------
    # torch Dataset protocol
    # ---------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        sample = self._load(record)
        if sample is None:
            # Return a neighbouring valid sample instead of crashing the epoch.
            for offset in range(1, len(self.records)):
                alternative = self.records[(index + offset) % len(self.records)]
                sample = self._load(alternative)
                if sample is not None:
                    record = alternative
                    break
            if sample is None:  # pragma: no cover - entire split unreadable
                raise RuntimeError("No readable samples in this split — check the dataset folder.")

        if self.augment:
            sample = augment_sample(sample, self.augment_config, self._rng)

        return self._to_tensors(sample, record)

    def _load(self, record: SampleRecord) -> Optional[Dict[str, np.ndarray]]:
        try:
            person = load_image(record.person_path, mode="RGB")
            cloth = load_image(record.cloth_path, mode="RGB")
            if record.cloth_mask_path is not None:
                cloth_mask = load_image(record.cloth_mask_path, mode="L")
            else:
                from backend.cv.segmentation import BackgroundRemover

                _rgba, alpha = BackgroundRemover("grabcut").remove(cloth)
                cloth_mask = garment_mask_from_alpha(alpha)

            if record.agnostic_path is not None and record.pose_path is not None:
                agnostic = load_image(record.agnostic_path, mode="RGB")
                pose = load_image(record.pose_path, mode="RGB")
                worn_mask = np.zeros(person.shape[:2], np.uint8)
            else:
                if not self.build_missing:
                    raise FileNotFoundError("agnostic/pose missing and build_missing=False")
                worn_mask, landmarks = person_garment_mask(person, None, kind=self._kind_for(record.category))
                features = kp.build_body_features(person, landmarks, kind=self._kind_for(record.category), garment_mask=worn_mask)
                if features is None:  # pragma: no cover
                    raise ValueError("could not build body features")
                agnostic = features.agnostic_rgb
                pose = features.pose_map
                worn_mask = features.garment_mask
                _ = landmarks

            return {
                "person": person, "cloth": cloth, "cloth_mask": cloth_mask,
                "agnostic": agnostic, "pose": pose, "worn_mask": worn_mask,
                "category": record.category,
            }
        except Exception as exc:
            self._broken.append(f"{record.identifier}: {exc}")
            if not self.skip_broken:
                raise
            logger.warning("Skipping unreadable sample %s (%s)", record.identifier, exc)
            traceback.print_exc(limit=1)
            return None

    def _kind_for(self, category: str) -> str:
        if category in {"dress", "kurta", "traditional"}:
            return "dress"
        if category in {"bottom", "lower"}:
            return "lower"
        return "upper"

    def _to_tensors(self, sample: Dict[str, Any], record: SampleRecord) -> Dict[str, Any]:
        import torch

        resolution = self.resolution
        person = _resize(sample["person"], resolution)
        agnostic = _resize(sample["agnostic"], resolution)
        pose = _resize(sample["pose"], resolution)
        cloth = _resize(sample["cloth"], resolution)
        cloth_mask = _resize(sample["cloth_mask"], resolution, nearest=True)
        worn_mask = _resize(sample["worn_mask"], resolution, nearest=True)

        inpaint_mask = worn_mask.copy()
        if self.mask_dilate > 0:
            kernel = np.ones((self.mask_dilate, self.mask_dilate), np.uint8)
            inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=1)

        def to_tensor(image: np.ndarray, channels_first: bool = True) -> "torch.Tensor":
            array = np.asarray(image).astype(np.float32)
            if array.ndim == 2:
                array = array[..., None]
            array = array / 127.5 - 1.0
            tensor = torch.from_numpy(array)
            return tensor.permute(2, 0, 1) if channels_first else tensor

        mask_tensor = torch.from_numpy((np.asarray(inpaint_mask, np.float32) / 255.0)[None, ...])
        prompt = self._build_prompt(record.category)
        return {
            "pixel_values": to_tensor(person),
            "masked_image": to_tensor(agnostic),
            "inpaint_mask": mask_tensor,
            "control_image": to_tensor(cloth),
            "cloth": to_tensor(_resize(cloth, 256)),
            "cloth_mask": torch.from_numpy((np.asarray(cloth_mask, np.float32) / 255.0)[None, ...]),
            "worn_mask": torch.from_numpy((np.asarray(worn_mask, np.float32) / 255.0)[None, ...]),
            "prompt": prompt,
            "category": record.category,
            "identifier": record.identifier,
            "original_size": torch.tensor([resolution, resolution]),
        }

    def _build_prompt(self, category: str) -> str:
        if self.prompt_template:
            return self.prompt_template.format(category=category)
        return CATEGORY_PROMPTS.get(category, CATEGORY_PROMPTS["unknown"])

    # ---------------------------------------------------------------------------------
    # introspection
    # ---------------------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        categories: Dict[str, int] = {}
        for record in self.records:
            categories[record.category] = categories.get(record.category, 0) + 1
        return {
            "root": str(self.root),
            "split": self.split,
            "samples": len(self.records),
            "resolution": self.resolution,
            "augment": self.augment,
            "categories": categories,
            "broken": len(self._broken),
            "has_agnostic": sum(1 for r in self.records if r.agnostic_path is not None),
            "has_pose": sum(1 for r in self.records if r.pose_path is not None),
            "has_cloth_mask": sum(1 for r in self.records if r.cloth_mask_path is not None),
        }

    def sample_batch(self, count: int = 4) -> Dict[str, Any]:
        """Fetch a small batch without a DataLoader (used by validation previews)."""
        indices = list(range(min(count, len(self))))
        items = [self[i] for i in indices]
        return collate_fn(items)


def _resize(image: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
    array = np.asarray(image)
    if array.shape[0] == size and array.shape[1] == size:
        return array
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(array, (size, size), interpolation=interpolation)


def collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-length fields (prompt/category are kept as lists)."""
    import torch

    keys = batch[0].keys()
    out: Dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            try:
                out[key] = torch.stack(values)
            except RuntimeError:  # pragma: no cover - mismatched shapes
                out[key] = values
        else:
            out[key] = list(values)
    return out


def build_dataset(
    root: str | Path,
    split: str = "train",
    resolution: int = 512,
    mode: str = "QUICK_DEMO",
    max_samples: Optional[int] = None,
    augment: Optional[bool] = None,
    seed: int = 42,
    **kwargs: Any,
) -> VTONPairDataset:
    """Convenience factory used by scripts and the trainer.

    ``QUICK_DEMO`` caps the split at 64 samples so a full epoch finishes in minutes.
    """
    if max_samples is None:
        if mode.upper() == "QUICK_DEMO":
            max_samples = 64
        elif mode.upper() == "FINE_TUNE":
            max_samples = kwargs.pop("fine_tune_samples", None)
    if augment is None:
        augment = split == "train"
    return VTONPairDataset(
        root=root, split=split, resolution=resolution, augment=augment,
        max_samples=max_samples, seed=seed, **kwargs,
    )


def dataset_report(root: str | Path, resolution: int = 512) -> Dict[str, Any]:
    """Summarise every split of a dataset (used by the dashboard and CLI)."""
    report: Dict[str, Any] = {"root": str(root), "splits": {}}
    for split in ("train", "val", "test"):
        try:
            dataset = VTONPairDataset(root, split=split, resolution=resolution, augment=False, verbose=False)
            report["splits"][split] = dataset.summary()
        except Exception as exc:  # pragma: no cover
            report["splits"][split] = {"error": str(exc)}
    metadata_path = Path(root) / "metadata.json"
    if metadata_path.exists():
        try:
            report["metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover
            pass
    return report
