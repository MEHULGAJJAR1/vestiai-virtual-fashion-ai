"""Dataset preprocessing: build agnostic images, cloth masks, pose maps and metadata.

This is the module shared by:

* ``scripts/prepare_dataset.py`` — converts VITON-HD / DressCode (or the built-in sample
  generator) into VestiAI's canonical layout,
* the training ``Dataset`` — every augmentation maps through the same helpers,
* ``scripts/preprocess_garments.py`` — bulk-prepares a folder of product photos.

Canonical layout (VITON-HD compatible, so real datasets drop straight in)::

    <root>/<split>/image/00001_00.jpg          person wearing a garment
    <root>/<split>/cloth/00001_00.jpg          the garment on a plain background
    <root>/<split>/cloth-mask/00001_00.jpg     255 = garment
    <root>/<split>/agnostic/00001_00.jpg       person with the garment region erased
    <root>/<split>/pose/00001_00.png           posemap (stick figure + torso fill)
    <root>/<split>/pairs.txt                   "<person> <cloth> <category>" per line
    <root>/metadata.json                       counts, licence, provenance
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from backend.cv import keypoints as kp
from backend.utils.image_utils import load_image, save_image, to_rgb
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

SPLITS = ("train", "val", "test")
SUBFOLDERS = ("image", "cloth", "cloth-mask", "agnostic", "pose")


@dataclass
class DatasetConfig:
    """Everything the preprocessing/prepare scripts need."""

    root: str = "datasets/viton_hd"
    resolution: int = 512
    splits: Tuple[str, ...] = SPLITS
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 42
    mask_dilate: int = 9
    agnostic_mode: str = "grey"       # grey | blur | white
    overwrite: bool = False
    max_samples: Optional[int] = None
    categories: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {**self.__dict__, "splits": list(self.splits)}


# --------------------------------------------------------------------------------------
# Mask utilities
# --------------------------------------------------------------------------------------
def dilate_mask(mask: np.ndarray, size: int = 9, iterations: int = 1) -> np.ndarray:
    """Dilate a binary mask (the inpainting region is always slightly larger than the cloth)."""
    if size <= 0:
        return mask
    kernel = np.ones((int(size), int(size)), np.uint8)
    return cv2.dilate(np.asarray(mask, np.uint8), kernel, iterations=iterations)


def garment_mask_from_alpha(alpha: np.ndarray, threshold: int = 40) -> np.ndarray:
    """Binary garment mask from a cut-out alpha channel."""
    array = np.asarray(alpha)
    if array.dtype != np.uint8:
        array = (np.clip(array.astype(np.float32), 0, 1) * 255).astype(np.uint8)
    return ((array > threshold).astype(np.uint8)) * 255


def cloth_mask_from_product_image(rgb: np.ndarray, background_removal: str = "auto") -> np.ndarray:
    """Segment a garment product photo into a cloth mask (GrabCut / rembg)."""
    from backend.cv.segmentation import BackgroundRemover

    remover = BackgroundRemover(background_removal)
    _rgba, alpha = remover.remove(rgb)
    return garment_mask_from_alpha(alpha)


def person_garment_mask(
    person_rgb: np.ndarray,
    landmarks: Optional[np.ndarray] = None,
    kind: str = "upper",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Estimate the worn-garment mask on a person image.

    Uses pose landmarks when available (accurate) and falls back to GrabCut over the torso
    box. Real VITON-HD ships these masks; for arbitrary photos this is how they are built.
    """
    if landmarks is None:
        from backend.cv.pose import PoseEstimator

        estimator = PoseEstimator(enable_segmentation=False, allow_download=True)
        result = estimator.estimate(person_rgb, include_mask=False)
        estimator.close()
        landmarks = result.landmarks if result.detected else None

    if landmarks is None:
        # Last resort: central rectangle
        height, width = person_rgb.shape[:2]
        box = (int(width * 0.2), int(height * 0.2), int(width * 0.8), int(height * 0.75))
        mask = np.zeros((height, width), np.uint8)
        mask[box[1]:box[3], box[0]:box[2]] = 255
        return mask, None

    bbox = kp.garment_region_bbox(landmarks, person_rgb.shape[:2], kind=kind)
    if bbox is None:
        height, width = person_rgb.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        mask[int(height * 0.2):int(height * 0.7), int(width * 0.2):int(width * 0.8)] = 255
        return mask, landmarks

    from backend.cv.segmentation import segment_garment_from_person

    mask = segment_garment_from_person(person_rgb, bbox, iterations=4)
    return mask, landmarks


# --------------------------------------------------------------------------------------
# Sample building
# --------------------------------------------------------------------------------------
def build_sample(
    person_image: np.ndarray,
    garment_image: np.ndarray,
    cloth_mask: Optional[np.ndarray] = None,
    landmarks: Optional[np.ndarray] = None,
    kind: str = "upper",
    agnostic_mode: str = "grey",
    resolution: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Assemble a single training sample from a person + garment pair."""
    person = to_rgb(person_image)
    garment = to_rgb(garment_image)

    if cloth_mask is None:
        cloth_mask = cloth_mask_from_product_image(garment_image)

    worn_mask, landmarks = person_garment_mask(person, landmarks=landmarks, kind=kind)
    features = kp.build_body_features(person, landmarks, kind=kind, garment_mask=worn_mask, erase_mode=agnostic_mode)
    if features is None:  # pragma: no cover - degenerate input
        height, width = person.shape[:2]
        agnostic = person.copy()
        pose_map = np.zeros((height, width, 3), np.uint8)
        worn_mask = np.zeros((height, width), np.uint8)
    else:
        agnostic = features.agnostic_rgb
        pose_map = features.pose_map
        worn_mask = features.garment_mask

    sample = {
        "person": person,
        "cloth": garment,
        "cloth_mask": garment_mask_from_alpha(cloth_mask) if cloth_mask.dtype != np.uint8 else cloth_mask,
        "agnostic": agnostic,
        "pose": pose_map,
        "worn_mask": worn_mask,
        "landmarks": landmarks if landmarks is not None else np.zeros((33, 4), np.float32),
    }
    if resolution:
        sample = {key: (resize_sample_field(value, resolution) if key != "landmarks" else value) for key, value in sample.items()}
    return sample


def resize_sample_field(array: np.ndarray, resolution: int) -> np.ndarray:
    """Square-resize a sample field, keeping masks nearest-neighbour."""
    interpolation = cv2.INTER_NEAREST if array.dtype == np.uint8 and array.ndim == 2 else cv2.INTER_AREA
    return cv2.resize(np.asarray(array), (resolution, resolution), interpolation=interpolation)


def write_sample(sample: Dict[str, np.ndarray], split_dir: Path, identifier: str, cloth_name: Optional[str] = None) -> str:
    """Write one sample into the split folder; returns the ``pairs.txt`` line."""
    cloth_name = cloth_name or f"{identifier}.jpg"
    person_name = f"{identifier}.jpg"
    (split_dir / "image").mkdir(parents=True, exist_ok=True)
    save_image(sample["person"], split_dir / "image" / person_name, quality=95)
    save_image(sample["cloth"], split_dir / "cloth" / cloth_name, quality=95)
    save_image(sample["cloth_mask"], split_dir / "cloth-mask" / cloth_name, quality=95)
    save_image(sample["agnostic"], split_dir / "agnostic" / person_name, quality=95)
    save_image(sample["pose"], split_dir / "pose" / person_name.replace(".jpg", ".png"))
    return f"{person_name} {cloth_name} {sample.get('category', 'unknown') if isinstance(sample.get('category'), str) else 'upper'}\n"


# --------------------------------------------------------------------------------------
# Augmentation (shared by training & validation loaders)
# --------------------------------------------------------------------------------------
@dataclass
class AugmentConfig:
    """Augmentation knobs; all defaults are safe (no geometry-breaking transforms)."""

    horizontal_flip: bool = True
    flip_probability: float = 0.5
    brightness: float = 0.12
    contrast: float = 0.12
    saturation: float = 0.12
    hue: float = 0.02
    rotate_degrees: float = 3.0
    scale_jitter: float = 0.05
    translate_fraction: float = 0.03
    blur_probability: float = 0.05
    noise_probability: float = 0.08


def augment_sample(sample: Dict[str, np.ndarray], config: AugmentConfig, rng: random.Random) -> Dict[str, np.ndarray]:
    """Apply geometry-consistent augmentation to a sample dict.

    Geometry (flip / rotate / scale / translate) is applied to person, agnostic, pose and
    worn mask *together*; photometric jitter is applied to the person and the garment
    *independently* (they are different photographs — matching their colour would teach the
    model to copy the person's lighting onto the garment).
    """
    out = {key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in sample.items()}
    height, width = out["person"].shape[:2]

    if config.horizontal_flip and rng.random() < config.flip_probability:
        for key in ("person", "agnostic", "pose", "worn_mask", "cloth", "cloth_mask"):
            if key in out:
                out[key] = np.ascontiguousarray(out[key][:, ::-1])
        if "landmarks" in out and out["landmarks"] is not None and len(out["landmarks"]):
            out["landmarks"] = kp._pixel_landmarks(out["landmarks"], width, height).copy()
            out["landmarks"][:, 0] = width - out["landmarks"][:, 0]

    angle = rng.uniform(-config.rotate_degrees, config.rotate_degrees)
    scale = 1.0 + rng.uniform(-config.scale_jitter, config.scale_jitter)
    tx = rng.uniform(-config.translate_fraction, config.translate_fraction) * width
    ty = rng.uniform(-config.translate_fraction, config.translate_fraction) * height
    if abs(angle) > 1e-3 or abs(scale - 1.0) > 1e-3 or abs(tx) > 0.5 or abs(ty) > 0.5:
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
        matrix[0, 2] += tx
        matrix[1, 2] += ty
        for key in ("person", "agnostic", "pose", "worn_mask"):
            if key in out:
                flags = cv2.INTER_NEAREST if out[key].dtype == np.uint8 and out[key].ndim == 2 else cv2.INTER_LINEAR
                out[key] = cv2.warpAffine(out[key], matrix, (width, height), flags=flags, borderMode=cv2.BORDER_REPLICATE)
        if "landmarks" in out and out["landmarks"] is not None and len(out["landmarks"]):
            landmarks = out["landmarks"].copy()
            ones = np.ones((landmarks.shape[0], 1), np.float32)
            landmarks[:, :2] = (np.hstack([landmarks[:, :2], ones]) @ matrix.T)
            out["landmarks"] = landmarks

    if config.brightness or config.contrast or config.saturation:
        out["person"] = colour_jitter(out["person"], config, rng)
        out["cloth"] = colour_jitter(out["cloth"], config, rng)

    if config.blur_probability and rng.random() < config.blur_probability:
        sigma = rng.uniform(0.4, 1.0)
        out["person"] = cv2.GaussianBlur(out["person"], (0, 0), sigma)
    if config.noise_probability and rng.random() < config.noise_probability:
        noise = np.random.default_rng(rng.randint(0, 10 ** 6)).normal(0, rng.uniform(2, 7), out["person"].shape)
        out["person"] = np.clip(out["person"].astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out


def colour_jitter(image: np.ndarray, config: AugmentConfig, rng: random.Random) -> np.ndarray:
    """Random brightness/contrast/saturation/hue adjustment (PIL-free)."""
    array = np.asarray(image, np.float32)
    if config.brightness:
        array = array * (1.0 + rng.uniform(-config.brightness, config.brightness))
    if config.contrast:
        mean = array.reshape(-1, 3).mean(axis=0)
        factor = 1.0 + rng.uniform(-config.contrast, config.contrast)
        array = (array - mean) * factor + mean
    array = np.clip(array, 0, 255).astype(np.uint8)
    if config.saturation or config.hue:
        hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV).astype(np.float32)
        if config.saturation:
            hsv[..., 1] *= 1.0 + rng.uniform(-config.saturation, config.saturation)
        if config.hue:
            hsv[..., 0] = (hsv[..., 0] + rng.uniform(-config.hue, config.hue) * 180.0) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        array = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return array


# --------------------------------------------------------------------------------------
# Dataset-level preparation
# --------------------------------------------------------------------------------------
def prepare_metadata(root: str | Path, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Write ``metadata.json`` describing the dataset (counts, categories, licence)."""
    root = Path(root)
    counts: Dict[str, int] = {}
    categories: Dict[str, int] = {}
    for split in SPLITS:
        pairs_path = root / split / "pairs.txt"
        if not pairs_path.exists():
            counts[split] = 0
            continue
        lines = [line.strip() for line in pairs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        counts[split] = len(lines)
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                categories[parts[2]] = categories.get(parts[2], 0) + 1
    payload = {
        "root": str(root),
        "counts": counts,
        "total": sum(counts.values()),
        "categories": categories,
        "layout": list(SUBFOLDERS),
        "generated_at": __import__("time").time(),
    }
    if extra:
        payload.update(extra)
    (root / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def validate_layout(root: str | Path, strict: bool = False) -> Dict[str, Any]:
    """Check that a dataset folder has the expected structure + matching file counts."""
    root = Path(root)
    report: Dict[str, Any] = {"root": str(root), "ok": True, "splits": {}, "issues": [], "warnings": []}
    if not root.exists():
        report["ok"] = False
        report["issues"].append(f"Dataset root {root} does not exist.")
        return report

    for split in SPLITS:
        split_dir = root / split
        if not split_dir.exists():
            report["warnings"].append(f"Split '{split}' is missing.")
            report["splits"][split] = {"exists": False}
            continue
        entry: Dict[str, Any] = {"exists": True, "files": {}, "missing": []}
        for sub in SUBFOLDERS:
            sub_dir = split_dir / sub
            count = len([p for p in sub_dir.glob("*") if p.is_file()]) if sub_dir.exists() else 0
            entry["files"][sub] = count
            if sub == "agnostic":
                # agnostic images are optional: the trainer can build them on the fly.
                continue
            if count == 0:
                entry["missing"].append(sub)
                if strict and sub in {"image", "cloth", "cloth-mask"}:
                    report["ok"] = False
                    report["issues"].append(f"{split}/{sub} is empty (required).")
        pairs = split_dir / "pairs.txt"
        entry["pairs"] = _count_lines(pairs)
        if entry["pairs"] == 0:
            entry["missing"].append("pairs.txt")
            report["warnings"].append(f"{split}/pairs.txt is missing; the loader will index files directly.")
        if len(set(entry["files"].get(sub, 0) for sub in ("image", "cloth", "cloth-mask"))) > 1:
            report["warnings"].append(f"{split}: image/cloth/cloth-mask file counts differ — some samples may be skipped.")
        report["splits"][split] = entry

    metadata = root / "metadata.json"
    if metadata.exists():
        try:
            report["metadata"] = json.loads(metadata.read_text(encoding="utf-8"))
        except Exception as exc:
            report["warnings"].append(f"metadata.json is unreadable: {exc}")
    else:
        report["warnings"].append("metadata.json is missing (run prepare_dataset.py to generate it).")
    return report


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def find_dataset_root(candidates: Optional[Sequence[str | Path]] = None) -> Optional[Path]:
    """Locate a dataset: explicit candidates, then the conventional locations."""
    search: List[Path] = [Path(item) for item in (candidates or [])]
    search += [
        Path("datasets/viton_hd"), Path("datasets/viton-hd"), Path("datasets/dresscode"),
        Path("datasets/samples"), Path("datasets"),
    ]
    for candidate in search:
        if candidate.exists() and ((candidate / "train").exists() or (candidate / "pairs.txt").exists()):
            return candidate
    return None
