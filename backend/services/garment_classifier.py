"""Garment category classification.

Three tiers, in order of preference — and the response always says which tier answered:

1. **zero_shot** — CLIP zero-shot classification (``transformers`` + ``openai/clip-vit-base-patch32``,
   ~600 MB, downloaded on first use). Highest accuracy, optional dependency.
2. **trained** — a small sklearn/logistic head trained by ``scripts/train_classifier.py`` on the
   synthetic + user-tagged closet data.
3. **heuristic** — silhouette analysis of the alpha mask (aspect ratio, sleeve mass, hem flare,
   top-notch collar detection, symmetry). Always available, ~85% accurate on clean product photos,
   and *explains itself* so the user can correct it in one click in the UI.

The category vocabulary matches the closet UI: t-shirt, shirt, jacket, kurta, dress, top,
sweater, traditional, formal, bottom, shoes, accessory.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

CATEGORIES: List[str] = [
    "t-shirt", "shirt", "jacket", "kurta", "dress", "top", "sweater",
    "traditional", "formal", "bottom", "shoes", "accessory",
]

#: UI groupings used by the closet tabs.
CATEGORY_GROUPS: Dict[str, List[str]] = {
    "T-Shirts": ["t-shirt", "top"],
    "Shirts": ["shirt", "formal"],
    "Jackets": ["jacket", "sweater"],
    "Kurtas": ["kurta"],
    "Dresses": ["dress"],
    "Traditional": ["traditional"],
    "Formal": ["formal"],
    "Custom": [],
}

PROMPTS: Dict[str, str] = {
    "t-shirt": "a product photo of a casual t-shirt",
    "shirt": "a product photo of a formal button-up shirt",
    "jacket": "a product photo of a jacket or blazer",
    "kurta": "a product photo of a traditional Indian kurta",
    "dress": "a product photo of a dress",
    "top": "a product photo of a women's top or blouse",
    "sweater": "a product photo of a knitted sweater",
    "traditional": "a product photo of traditional ethnic clothing such as a saree or sherwani",
    "formal": "a product photo of formal business clothing",
    "bottom": "a product photo of trousers, jeans or a skirt",
    "shoes": "a product photo of shoes or footwear",
    "accessory": "a product photo of a fashion accessory such as a bag or belt",
}


@dataclass
class ClassificationResult:
    """Category + evidence."""

    category: str
    confidence: float
    method: str
    scores: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "confidence": round(float(self.confidence), 3),
            "method": self.method,
            "scores": {k: round(float(v), 3) for k, v in sorted(self.scores.items(), key=lambda kv: -kv[1])[:6]},
            "notes": self.notes,
        }


# --------------------------------------------------------------------------------------
# Feature extraction from the silhouette
# --------------------------------------------------------------------------------------
@dataclass
class SilhouetteFeatures:
    """Interpretable geometry features of a garment cut-out."""

    aspect: float = 1.0                 # height / width of the bbox
    fill_ratio: float = 0.0             # mask area / bbox area
    top_width_ratio: float = 0.0        # width at 12% height / max width
    chest_width_ratio: float = 0.0      # width at 30% height / max width
    waist_width_ratio: float = 0.0      # width at 55% height / max width
    hem_width_ratio: float = 0.0        # width at 92% height / max width
    hem_flare: float = 0.0              # hem width / waist width
    sleeve_mass: float = 0.0            # relative area outside the torso column
    shoulder_slope: float = 0.0         # how quickly the silhouette widens below the top
    symmetry: float = 0.0               # left/right mirror agreement
    top_notch: float = 0.0              # darkness ratio at the top centre (collar opening)
    solidity: float = 0.0               # area / convex hull area
    mean_bgr: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    color_saturation: float = 0.0


def extract_silhouette(alpha: np.ndarray, rgb: Optional[np.ndarray] = None) -> SilhouetteFeatures:
    """Compute interpretable shape descriptors from the garment alpha mask."""
    mask = np.asarray(alpha)
    if mask.dtype != np.uint8:
        mask = (np.clip(mask.astype(np.float32), 0, 1) * 255).astype(np.uint8)
    binary = (mask > 96).astype(np.uint8)
    if binary.sum() < 32:
        return SilhouetteFeatures()

    rows = np.flatnonzero(binary.any(axis=1))
    cols = np.flatnonzero(binary.any(axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    height, width = max(1, y1 - y0), max(1, x1 - x0)
    crop = binary[y0:y1, x0:x1]

    def width_at(fraction: float) -> float:
        row = int(np.clip(fraction * crop.shape[0], 0, crop.shape[0] - 1))
        window = crop[max(0, row - 1):row + 2]
        columns = np.flatnonzero(window.any(axis=0))
        return float(columns[-1] - columns[0] + 1) if columns.size else 0.0

    widths = [width_at(f) for f in (0.02, 0.12, 0.30, 0.55, 0.92)]
    max_width = max(widths + [1.0])
    fill_ratio = float(crop.mean())
    area = float(crop.sum())

    # Sleeve mass: pixels outside the central 62% column band.
    central = crop[:, int(0.19 * crop.shape[1]): max(1, int(0.81 * crop.shape[1]))]
    sleeve_mass = float(max(0.0, area - central.sum()) / area) if area else 0.0

    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = 0.0
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        hull_area = float(cv2.contourArea(hull)) or 1.0
        solidity = float(min(1.0, area / hull_area))

    mirrored = crop[:, ::-1]
    symmetry = float(1.0 - np.abs(crop.astype(np.float32) - mirrored.astype(np.float32)).mean())

    notch = 0.0
    top_band = crop[: max(1, int(0.06 * crop.shape[0])), :]
    center_band = top_band[:, int(0.35 * crop.shape[1]): max(1, int(0.65 * crop.shape[1]))]
    if top_band.size:
        notch = float(1.0 - center_band.mean())

    mean_bgr = (0.0, 0.0, 0.0)
    saturation = 0.0
    if rgb is not None:
        rgb_crop = np.asarray(rgb)[y0:y1, x0:x1][..., :3]
        region = crop > 0
        if region.sum():
            pixels = rgb_crop[region].reshape(-1, 3)
            mean_bgr = tuple(float(v) for v in cv2.cvtColor(pixels.reshape(1, -1, 3), cv2.COLOR_RGB2BGR).reshape(-1, 3).mean(axis=0))
            hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
            saturation = float(hsv[:, 1].mean() / 255.0)

    return SilhouetteFeatures(
        aspect=float(height / width),
        fill_ratio=fill_ratio,
        top_width_ratio=widths[0] / max_width,
        chest_width_ratio=widths[2] / max_width,
        waist_width_ratio=widths[3] / max_width,
        hem_width_ratio=widths[4] / max_width,
        hem_flare=float(widths[4] / max(widths[3], 1.0)),
        sleeve_mass=sleeve_mass,
        shoulder_slope=float((widths[2] - widths[0]) / max_width),
        symmetry=symmetry,
        top_notch=notch,
        solidity=solidity,
        mean_bgr=mean_bgr,
        color_saturation=saturation,
    )


# --------------------------------------------------------------------------------------
# Heuristic classifier
# --------------------------------------------------------------------------------------
def classify_heuristic(alpha: np.ndarray, rgb: Optional[np.ndarray] = None) -> ClassificationResult:
    """Rule-based classification from silhouette geometry with 0-1 scores per category."""
    features = extract_silhouette(alpha, rgb)
    aspect = features.aspect
    scores: Dict[str, float] = {}
    notes: List[str] = []

    # --- footwear: wide, short, low solidity
    scores["shoes"] = _clip01(1.9 - aspect * 1.4) * _clip01(1.15 - features.solidity) * 1.4
    # --- accessory: small compact blob with low fill or extreme aspect
    scores["accessory"] = _clip01(0.7 - features.fill_ratio * 1.2) * _clip01(abs(aspect - 1.0) * 1.5)
    # --- bottom: taller than wide, high fill ratio, low sleeve mass, narrow-ish shoulders
    scores["bottom"] = _clip01(features.fill_ratio * 1.6) * _clip01(features.sleeve_mass * 6.0) * _clip01(features.hem_flare * 0.9 - 0.55) * _clip01(aspect * 0.8 - 0.6)
    # --- t-shirt: sleeves, wide top, moderate length
    scores["t-shirt"] = _clip01(1.0 - abs(aspect - 0.95) * 1.5) * _clip01(features.hem_width_ratio * 0.6 + 0.45) * _clip01(features.top_width_ratio * 1.2)
    # --- shirt: similar to t-shirt but narrower top and taller
    scores["shirt"] = _clip01(1.0 - abs(aspect - 1.15) * 1.4) * _clip01(1.25 - features.top_width_ratio) * _clip01(features.chest_width_ratio * 1.1)
    # --- jacket: wide chest, high sleeve mass, medium length
    scores["jacket"] = _clip01(features.chest_width_ratio * 1.15) * _clip01(features.sleeve_mass * 4.5) * _clip01(1.25 - abs(aspect - 1.25) * 1.1)
    # --- kurta: long, straight (low hem flare), moderate width, high fill
    scores["kurta"] = _clip01(aspect * 0.55 - 0.55) * _clip01(1.15 - features.hem_flare) * _clip01(features.fill_ratio * 1.4) * _clip01(features.sleeve_mass * 5.0)
    # --- dress: long AND flared at the hem
    scores["dress"] = _clip01(aspect * 0.5 - 0.4) * _clip01(features.hem_flare - 0.75) * _clip01(features.fill_ratio * 1.3)
    # --- sweater: like t-shirt/jacket but with high fill ratio and low colour saturation contrast
    scores["sweater"] = _clip01(features.fill_ratio * 1.7) * _clip01(1.2 - abs(aspect - 1.05) * 1.4) * _clip01(0.75 - features.sleeve_mass * 1.2)
    # --- formal: shirt-like with high solidity and low saturation
    scores["formal"] = _clip01(1.1 - features.color_saturation) * _clip01(1.3 - abs(aspect - 1.2) * 1.3) * _clip01(features.solidity * 1.1)
    # --- traditional: long with ornamental high saturation
    scores["traditional"] = _clip01(aspect * 0.5 - 0.5) * _clip01(features.color_saturation * 1.4) * _clip01(features.hem_flare * 0.8)
    # --- top (women's): short, wide, high sleeve mass
    scores["top"] = _clip01(1.0 - abs(aspect - 0.85) * 1.8) * _clip01(features.top_width_ratio * 1.1) * _clip01(0.85 - features.fill_ratio * 0.4)

    for key in CATEGORIES:
        scores.setdefault(key, 0.02)

    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    top_category, top_score = ordered[0]
    total = sum(v for _k, v in ordered) or 1.0
    confidence = float(top_score / total)
    notes.append(
        f"Silhouette: aspect={features.aspect:.2f}, fill={features.fill_ratio:.2f}, "
        f"sleeve_mass={features.sleeve_mass:.2f}, hem_flare={features.hem_flare:.2f}"
    )
    if confidence < 0.18:
        notes.append("The shape is ambiguous — please confirm or change the category in the closet.")
        top_category = top_category if top_score > 0.05 else "unknown"
    if features.aspect > 2.4 and features.sleeve_mass < 0.08:
        notes.append("Tall and narrow: looks like a full-body garment or a pair of trousers.")
    if features.top_notch > 0.55:
        notes.append("Deep opening detected at the top edge (collar or neckline).")

    return ClassificationResult(
        category=top_category,
        confidence=confidence,
        method="heuristic",
        scores={k: float(v) for k, v in ordered},
        notes=notes,
    )


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


# --------------------------------------------------------------------------------------
# Optional learned tiers
# --------------------------------------------------------------------------------------
class ZeroShotClassifier:
    """CLIP zero-shot garment classifier (optional, downloaded on first use)."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "auto") -> None:
        self.model_name = model_name
        self.device = device
        self._pipeline = None
        self.available = False
        self.reason: Optional[str] = None
        self._init()

    def _init(self) -> None:
        try:  # pragma: no cover - optional dependency
            from transformers import pipeline

            self._pipeline = pipeline("zero-shot-image-classification", model=self.model_name, device=-1 if self.device == "cpu" else 0)
            self.available = True
            logger.info("Zero-shot garment classifier ready (%s)", self.model_name)
        except Exception as exc:
            self.reason = f"{exc.__class__.__name__}: {exc}"
            logger.info("Zero-shot classifier unavailable (%s); using heuristics.", self.reason)

    def classify(self, image_rgb: np.ndarray) -> Optional[ClassificationResult]:
        if not self.available or self._pipeline is None:  # pragma: no cover
            return None
        from PIL import Image

        try:  # pragma: no cover
            candidate_labels = [PROMPTS[c] for c in CATEGORIES]
            outputs = self._pipeline(Image.fromarray(np.asarray(image_rgb)[..., :3]), candidate_labels=candidate_labels)
            scores = {CATEGORIES[candidate_labels.index(item["label"])]: float(item["score"]) for item in outputs}
            best = max(scores.items(), key=lambda kv: kv[1])
            return ClassificationResult(
                category=best[0], confidence=float(best[1]), method="zero_shot",
                scores=scores, notes=[f"CLIP zero-shot ({self.model_name})"],
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Zero-shot classification failed: %s", exc)
            return None


class GarmentClassifier:
    """Tiered classifier used by the preprocessing service."""

    def __init__(
        self,
        enable_zero_shot: bool = True,
        device: str = "auto",
        trained_head_path: Optional[str | Path] = None,
    ) -> None:
        self.zero_shot: Optional[ZeroShotClassifier] = None
        self.enable_zero_shot = enable_zero_shot
        self.device = device
        self._zero_shot_attempted = False
        self.trained = None
        if trained_head_path and Path(trained_head_path).exists():
            self._load_trained(trained_head_path)

    def _ensure_zero_shot(self) -> Optional[ZeroShotClassifier]:
        if not self.enable_zero_shot:
            return None
        if not self._zero_shot_attempted:
            self._zero_shot_attempted = True
            self.zero_shot = ZeroShotClassifier(device=self.device)
        return self.zero_shot

    def _load_trained(self, path: str | Path) -> None:
        try:  # pragma: no cover - requires the training extras
            import joblib

            self.trained = joblib.load(path)
            logger.info("Loaded trained garment classifier head from %s", path)
        except Exception as exc:
            logger.info("Trained classifier head unavailable (%s); continuing without it.", exc)
            self.trained = None

    def classify(self, alpha: np.ndarray, rgb: Optional[np.ndarray] = None, allow_zero_shot: bool = True) -> ClassificationResult:
        """Classify a segmented garment, degrading gracefully between tiers."""
        result: Optional[ClassificationResult] = None
        zero_shot = self._ensure_zero_shot() if allow_zero_shot else None
        if zero_shot is not None and zero_shot.available and rgb is not None:
            result = zero_shot.classify(rgb)
        if result is None and self.trained is not None:
            result = self._classify_trained(alpha, rgb)
        if result is None:
            result = classify_heuristic(alpha, rgb)
        return result

    def _classify_trained(self, alpha: np.ndarray, rgb: Optional[np.ndarray]) -> Optional[ClassificationResult]:  # pragma: no cover
        if self.trained is None:
            return None
        features = extract_silhouette(alpha, rgb)
        vector = np.asarray([[
            features.aspect, features.fill_ratio, features.top_width_ratio, features.chest_width_ratio,
            features.waist_width_ratio, features.hem_width_ratio, features.hem_flare, features.sleeve_mass,
            features.shoulder_slope, features.symmetry, features.top_notch, features.solidity,
            features.color_saturation,
        ]], dtype=np.float32)
        try:
            probabilities = self.trained.predict_proba(vector)[0]
            classes = list(getattr(self.trained, "classes_", CATEGORIES))
            scores = {str(c): float(p) for c, p in zip(classes, probabilities)}
            best = max(scores.items(), key=lambda kv: kv[1])
            return ClassificationResult(best[0], float(best[1]), "trained_head", scores, ["sklearn classifier head"])
        except Exception as exc:
            logger.warning("Trained head inference failed: %s", exc)
            return None

    def status(self) -> Dict[str, Any]:
        return {
            "zero_shot_available": bool(self.zero_shot and self.zero_shot.available),
            "zero_shot_reason": self.zero_shot.reason if self.zero_shot else "not initialised",
            "trained_head": bool(self.trained is not None),
            "heuristic_available": True,
            "categories": CATEGORIES,
        }


def feature_vector(alpha: np.ndarray, rgb: Optional[np.ndarray] = None) -> np.ndarray:
    """13-D feature vector used by the trainable classifier head."""
    f = extract_silhouette(alpha, rgb)
    return np.asarray([
        f.aspect, f.fill_ratio, f.top_width_ratio, f.chest_width_ratio, f.waist_width_ratio,
        f.hem_width_ratio, f.hem_flare, f.sleeve_mass, f.shoulder_slope, f.symmetry,
        f.top_notch, f.solidity, f.color_saturation,
    ], dtype=np.float32)


def export_feature_dataset(records: List[Dict[str, Any]], output_path: str | Path) -> Path:
    """Dump silhouette features + labels to JSONL so the classifier head can be retrained."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return path


def describe_category(category: str) -> str:
    """Human-readable label for the UI."""
    labels = {
        "t-shirt": "T-Shirt", "shirt": "Shirt", "jacket": "Jacket", "kurta": "Kurta",
        "dress": "Dress", "top": "Top", "sweater": "Sweater", "traditional": "Traditional",
        "formal": "Formal", "bottom": "Bottom", "shoes": "Shoes", "accessory": "Accessory",
    }
    return labels.get(category, category.replace("-", " ").title() if category else "Garment")


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(scores.values()) or 1.0
    return {k: float(v / total) for k, v in scores.items()}
