"""Sample-data mode: synthesise realistic-enough garments and training pairs locally.

Why synthesise instead of shipping photos? Because product photography from Amazon /
Flipkart / Myntra is copyrighted, and the brief explicitly forbids scraping those sites.
So VestiAI ships a **generator** that draws garments procedurally (correct silhouettes,
fabric texture, shading, prints) and a pair generator that composites those garments onto a
procedural mannequin with pose landmarks. That gives:

* a closet the user can play with immediately (no downloads),
* a tiny but real dataset (person / garment / cloth-mask / agnostic / pose / pairs) so
  ``QUICK_DEMO`` training genuinely exercises the whole ML pipeline end-to-end,
* and a documented upgrade path to real data: ``scripts/prepare_dataset.py`` converts
  VITON-HD or DressCode (downloaded by *you* under their licences) into the same layout.

Nothing here is faked output — these are real image tensors produced by real code.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from backend.utils.image_utils import ensure_rgba, save_image, to_rgb
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

GARMENT_TEMPLATES: Dict[str, Dict[str, object]] = {
    "t-shirt": {"aspect": 1.15, "sleeve": "short", "length": 0.62, "hem_flare": 1.05, "collar": "crew"},
    "shirt": {"aspect": 1.35, "sleeve": "long", "length": 0.70, "hem_flare": 1.02, "collar": "point"},
    "jacket": {"aspect": 1.25, "sleeve": "long", "length": 0.78, "hem_flare": 1.10, "collar": "lapel"},
    "kurta": {"aspect": 1.85, "sleeve": "long", "length": 1.00, "hem_flare": 1.08, "collar": "mandarin"},
    "dress": {"aspect": 2.10, "sleeve": "sleeveless", "length": 1.00, "hem_flare": 1.55, "collar": "v"},
    "top": {"aspect": 0.95, "sleeve": "short", "length": 0.55, "hem_flare": 1.0, "collar": "v"},
    "bottom": {"aspect": 1.90, "sleeve": "none", "length": 1.0, "hem_flare": 1.0, "collar": "waist"},
}

COLOURS = [
    (33, 63, 122), (176, 42, 55), (28, 92, 60), (222, 184, 92), (18, 26, 38),
    (150, 90, 190), (240, 240, 240), (200, 120, 60), (90, 130, 130), (250, 200, 210),
]


@dataclass
class SyntheticGarment:
    """A generated garment image + metadata."""

    image: np.ndarray      # RGBA
    category: str
    label: str
    colour: Tuple[int, int, int]


def _fabric_texture(size: Tuple[int, int], colour: Tuple[int, int, int], style: str, rng: random.Random) -> np.ndarray:
    """Procedural fabric: base colour + weave noise + optional stripes/check/print."""
    height, width = size
    base = np.zeros((height, width, 3), np.float32)
    base[:] = np.asarray(colour, np.float32) / 255.0

    # weave noise
    noise = np.random.default_rng(rng.randint(0, 10 ** 6)).normal(0, 0.022, (height, width, 1)).astype(np.float32)
    base = np.clip(base + noise, 0, 1)

    if style == "stripes":
        period = rng.choice([10, 14, 18])
        stripe = (np.arange(width) // period) % 2
        base *= (0.86 + 0.14 * stripe)[None, :, None]
    elif style == "check":
        period = rng.choice([14, 22])
        rows = ((np.arange(height) // period) % 2)[:, None]
        cols = ((np.arange(width) // period) % 2)[None, :]
        base *= (0.9 + 0.1 * np.logical_xor(rows.astype(bool), cols.astype(bool)))[..., None]
    elif style == "print":
        rng_local = np.random.default_rng(rng.randint(0, 10 ** 6))
        for _ in range(rng_local.integers(12, 26)):
            cx, cy = rng_local.integers(0, width), rng_local.integers(0, height)
            radius = rng_local.integers(5, 16)
            tone = rng_local.uniform(0.65, 1.25)
            cv2.circle(base, (int(cx), int(cy)), int(radius), tuple(float(v * tone) for v in colour[::-1]), -1, cv2.LINE_AA)
        base = cv2.GaussianBlur(base, (0, 0), 1.2)

    # soft vertical shading for volume
    gradient = np.linspace(0.94, 1.06, width, dtype=np.float32)[None, :, None]
    base = np.clip(base * gradient, 0, 1)
    return (base * 255).astype(np.uint8)


def generate_garment(
    category: str = "t-shirt",
    colour: Optional[Tuple[int, int, int]] = None,
    style: Optional[str] = None,
    size: int = 640,
    seed: Optional[int] = None,
) -> SyntheticGarment:
    """Draw a single garment as RGBA with a transparent background."""
    rng = random.Random(seed)
    template = GARMENT_TEMPLATES.get(category, GARMENT_TEMPLATES["t-shirt"])
    colour = colour or rng.choice(COLOURS)
    style = style or rng.choice(["plain", "plain", "stripes", "check", "print"])

    aspect = float(template["aspect"])  # type: ignore[arg-type]
    length = float(template["length"])  # type: ignore[arg-type]
    flare = float(template["hem_flare"])  # type: ignore[arg-type]
    sleeve = str(template["sleeve"])
    collar = str(template["collar"])

    canvas_h = int(size * min(1.6, max(0.9, aspect)))
    canvas_w = size
    rgb = _fabric_texture((canvas_h, canvas_w), colour, style, rng)
    alpha = np.zeros((canvas_h, canvas_w), np.uint8)

    cx = canvas_w // 2
    shoulder_y = int(canvas_h * 0.10)
    hem_y = int(canvas_h * (0.30 + 0.62 * length))
    shoulder_w = int(canvas_w * 0.44)
    waist_w = int(shoulder_w * (0.92 + 0.06 * flare))
    hem_w = int(shoulder_w * (1.02 * flare))

    top_left = (cx - shoulder_w // 2, shoulder_y)
    top_right = (cx + shoulder_w // 2, shoulder_y)
    hem_left = (cx - hem_w // 2, hem_y)
    hem_right = (cx + hem_w // 2, hem_y)
    waist_left = (cx - waist_w // 2, int(shoulder_y + (hem_y - shoulder_y) * 0.55))
    waist_right = (cx + waist_w // 2, int(shoulder_y + (hem_y - shoulder_y) * 0.55))

    body = np.array([top_left, top_right, waist_right, hem_right, hem_left, waist_left], np.int32)

    # --- collar
    collar_h = int(shoulder_w * 0.20)
    if collar == "v":
        neck = np.array([
            [cx - shoulder_w // 6, shoulder_y], [cx + shoulder_w // 6, shoulder_y],
            [cx, shoulder_y + collar_h],
        ], np.int32)
    elif collar == "mandarin":
        neck = np.array([
            [cx - shoulder_w // 7, shoulder_y - int(collar_h * 0.5)], [cx + shoulder_w // 7, shoulder_y - int(collar_h * 0.5)],
            [cx + shoulder_w // 7, shoulder_y + collar_h // 2], [cx - shoulder_w // 7, shoulder_y + collar_h // 2],
        ], np.int32)
    elif collar == "lapel":
        neck = np.array([
            [cx - shoulder_w // 4, shoulder_y], [cx, shoulder_y + collar_h],
            [cx + shoulder_w // 4, shoulder_y], [cx, shoulder_y + collar_h // 3],
        ], np.int32)
    else:  # crew / point
        neck = np.array([
            [cx - shoulder_w // 7, shoulder_y - int(collar_h * 0.3)],
            [cx + shoulder_w // 7, shoulder_y - int(collar_h * 0.3)],
            [cx + shoulder_w // 9, shoulder_y + collar_h],
            [cx - shoulder_w // 9, shoulder_y + collar_h],
        ], np.int32)

    cv2.fillPoly(alpha, [body], 255)
    cv2.fillPoly(alpha, [neck], 0)

    # --- sleeves
    sleeve_len = {"long": int(canvas_h * 0.34), "short": int(canvas_h * 0.16), "sleeveless": 0, "none": 0}[sleeve]
    if sleeve_len > 0:
        for side in (-1, 1):
            x_outer = cx + side * shoulder_w // 2
            x_tip = cx + side * int(shoulder_w * (0.62 if sleeve == "long" else 0.74))
            sleeve_poly = np.array([
                [x_outer, shoulder_y + int(canvas_h * 0.01)],
                [x_tip, shoulder_y + sleeve_len],
                [x_tip - side * int(shoulder_w * 0.16), shoulder_y + int(sleeve_len * 1.12)],
                [x_outer - side * int(shoulder_w * 0.14), shoulder_y + int(canvas_h * 0.09)],
            ], np.int32)
            cv2.fillPoly(alpha, [sleeve_poly], 255)
            cv2.polylines(alpha, [sleeve_poly], True, 255, 3)

    # --- trouser split for bottoms
    if category == "bottom":
        crotch_y = int(shoulder_y + (hem_y - shoulder_y) * 0.45)
        gap_w = int(shoulder_w * 0.10)
        cv2.rectangle(alpha, (cx - gap_w, crotch_y), (cx + gap_w, hem_y + 4), 0, -1)

    # --- waistband / hem detail
    cv2.rectangle(alpha, (top_left[0], shoulder_y - 3), (top_right[0], shoulder_y + 3), 255, -1)

    # shade edges so the garment reads as cloth rather than a flat cut-out
    kernel = np.ones((5, 5), np.uint8)
    inner = cv2.erode(alpha, kernel, iterations=2)
    rim = cv2.subtract(alpha, inner).astype(np.float32) / 255.0
    rgb = np.clip(rgb.astype(np.float32) * (1.0 - 0.22 * rim[..., None]), 0, 255).astype(np.uint8)
    rgb = cv2.GaussianBlur(rgb, (0, 0), 0.8)

    rgba = np.dstack([rgb, alpha])
    return SyntheticGarment(image=rgba, category=category, label=f"{category}-{style}", colour=colour)


# --------------------------------------------------------------------------------------
# Procedural mannequin (training person images)
# --------------------------------------------------------------------------------------
MANNEQUIN_SKINS = [(238, 200, 172), (206, 160, 120), (150, 108, 78), (92, 62, 45), (245, 222, 200)]
MANNEQUIN_BOTTOMS = [(40, 45, 60), (70, 70, 78), (30, 60, 90), (110, 80, 60)]


def generate_person(
    size: int = 512,
    seed: Optional[int] = None,
    pose: str = "front",
    background: Optional[Tuple[int, int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw a stylised human body with a garment-ready torso.

    Returns ``(rgb, torso_mask)`` where ``torso_mask`` marks the region a garment should
    cover — exactly what the dataset preparation needs to build cloth masks without a
    person-parsing network.
    """
    rng = random.Random(seed)
    height = width = size
    background = background or (rng.randint(200, 250), rng.randint(200, 250), rng.randint(205, 250))
    canvas = np.zeros((height, width, 3), np.uint8)
    canvas[:] = background
    canvas = cv2.GaussianBlur(canvas, (0, 0), 0.6)

    skin = rng.choice(MANNEQUIN_SKINS)
    bottom = rng.choice(MANNEQUIN_BOTTOMS)

    cx = width // 2 + rng.randint(-int(0.03 * width), int(0.03 * width))
    shoulder_y = int(height * rng.uniform(0.20, 0.24))
    hip_y = int(height * rng.uniform(0.52, 0.57))
    shoulder_w = int(width * rng.uniform(0.24, 0.30))
    head_r = int(width * 0.085)
    left_arm_lift = rng.uniform(-0.22, 0.22) if pose != "arms_up" else -0.55
    right_arm_lift = rng.uniform(-0.22, 0.22) if pose != "arms_up" else -0.55
    lean = rng.uniform(-0.06, 0.06)

    # legs / trousers
    for side in (-1, 1):
        x_hip = int(cx + side * shoulder_w * 0.26)
        x_ankle = int(cx + side * shoulder_w * 0.30 + lean * 40)
        cv2.line(canvas, (x_hip, hip_y), (x_ankle, int(height * 0.97)),
                 tuple(int(v) for v in bottom[::-1]), max(8, int(width * 0.055)), cv2.LINE_AA)

    # arms
    for side, lift in ((-1, left_arm_lift), (1, right_arm_lift)):
        x_shoulder = int(cx + side * shoulder_w // 2)
        elbow = (int(x_shoulder + side * width * 0.10), int(shoulder_y + (hip_y - shoulder_y) * 0.55 + lift * height * 0.2))
        wrist = (int(elbow[0] + side * width * 0.04), int(elbow[1] + (hip_y - shoulder_y) * 0.55 + lift * height * 0.25))
        cv2.line(canvas, (x_shoulder, shoulder_y), elbow, tuple(int(v) for v in skin[::-1]), max(6, int(width * 0.032)), cv2.LINE_AA)
        cv2.line(canvas, elbow, wrist, tuple(int(v) for v in skin[::-1]), max(5, int(width * 0.028)), cv2.LINE_AA)

    # torso (person wears a plain base layer; the try-on garment replaces it)
    torso = np.array([
        [cx - shoulder_w // 2, shoulder_y],
        [cx + shoulder_w // 2, shoulder_y],
        [cx + int(shoulder_w * 0.42), hip_y],
        [cx - int(shoulder_w * 0.42), hip_y],
    ], np.int32)
    base_layer = tuple(int(v * 0.92) for v in skin[::-1])
    cv2.fillPoly(canvas, [torso], base_layer)

    # neck + head
    cv2.rectangle(canvas, (cx - int(width * 0.035), shoulder_y - int(height * 0.05)),
                  (cx + int(width * 0.035), shoulder_y + 4), tuple(int(v) for v in skin[::-1]), -1)
    cv2.circle(canvas, (cx, shoulder_y - int(height * 0.075)), head_r, tuple(int(v) for v in skin[::-1]), -1, cv2.LINE_AA)

    # subtle noise + vignette so images are not too synthetic for the VAE
    noise = np.random.default_rng(rng.randint(0, 10 ** 6)).normal(0, 3.0, canvas.shape).astype(np.float32)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    torso_mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(torso_mask, [torso], 255)

    # --- landmarks (BlazePose layout) for the generated skeleton
    landmarks = np.zeros((33, 4), np.float32)
    lw = int(shoulder_w * 0.5)
    lm = {
        0: (cx, shoulder_y - int(height * 0.075)),                       # nose
        11: (cx - lw, shoulder_y), 12: (cx + lw, shoulder_y),            # shoulders
        13: (cx - lw - int(width * 0.10), int(shoulder_y + (hip_y - shoulder_y) * 0.55 + left_arm_lift * height * 0.2)),
        14: (cx + lw + int(width * 0.10), int(shoulder_y + (hip_y - shoulder_y) * 0.55 + right_arm_lift * height * 0.2)),
        15: (cx - lw - int(width * 0.14), int(shoulder_y + (hip_y - shoulder_y) * 1.1 + left_arm_lift * height * 0.45)),
        16: (cx + lw + int(width * 0.14), int(shoulder_y + (hip_y - shoulder_y) * 1.1 + right_arm_lift * height * 0.45)),
        23: (cx - int(shoulder_w * 0.30), hip_y), 24: (cx + int(shoulder_w * 0.30), hip_y),
        25: (cx - int(shoulder_w * 0.30), int(height * 0.78)), 26: (cx + int(shoulder_w * 0.30), int(height * 0.78)),
        27: (cx - int(shoulder_w * 0.34), int(height * 0.97)), 28: (cx + int(shoulder_w * 0.34), int(height * 0.97)),
    }
    for index, (x, y) in lm.items():
        landmarks[index] = (float(x), float(y), 0.0, 0.95)
    return canvas, torso_mask


def garment_and_person_pair(size: int = 512, seed: Optional[int] = None, category: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Build one training quadruple: person, garment, cloth mask, agnostic person."""
    from backend.cv import keypoints as kp
    from backend.cv.garment_align import compute_fit_transform, warp_garment

    rng = random.Random(seed)
    category = category or rng.choice(list(GARMENT_TEMPLATES.keys()))
    person, torso_mask = generate_person(size=size, seed=rng.randint(0, 10 ** 6))
    garment = generate_garment(category=category, size=size, seed=rng.randint(0, 10 ** 6))

    landmarks = _extract_landmarks_for_person(person, size)
    cloth_mask = np.zeros(person.shape[:2], np.uint8)
    solved = compute_fit_transform(garment.image[..., 3], landmarks, person.shape[:2], category)
    if solved is not None:
        matrix, _mode, _pose_anchors, _garment_anchors = solved
        warped = warp_garment(garment.image[..., :3], garment.image[..., 3], matrix, (size, size))
        cloth_mask = warped.alpha
    else:  # pragma: no cover - fall back to the torso mask
        cloth_mask = torso_mask

    features = kp.build_body_features(person, landmarks, kind="upper", garment_mask=cloth_mask)
    agnostic = features.agnostic_rgb if features is not None else person.copy()
    pose_map = features.pose_map if features is not None else np.zeros_like(person)
    return {
        "person": person,
        "garment": garment.image,
        "cloth_mask": cloth_mask,
        "agnostic": agnostic,
        "pose_map": pose_map,
        "category": np.asarray([category]),
    }


def _extract_landmarks_for_person(person: np.ndarray, size: int) -> np.ndarray:
    """Landmarks for a generated person.

    If a real pose estimator is installed we use it (keeps the synthetic set aligned with
    real inference); otherwise we detect the torso directly from the known geometry, and the
    result is stored explicitly as ``generated`` provenance in the dataset metadata.
    """
    try:
        from backend.cv.pose import PoseEstimator

        estimator = PoseEstimator(enable_segmentation=False, allow_download=False, prefer_tasks=True)
        result = estimator.estimate(person, include_mask=False)
        estimator.close()
        if result.detected and result.backend.startswith("mediapipe"):
            return result.landmarks
    except Exception as exc:  # pragma: no cover
        logger.debug("Pose estimator unavailable for synthetic person: %s", exc)

    # Fallback: reproduce the generator's geometry (deterministic for the same seed).
    height = width = size
    landmarks = np.zeros((33, 4), np.float32)
    # locate the torso from the known base-layer colour band
    gray = cv2.cvtColor(person, cv2.COLOR_RGB2GRAY)
    mask = (gray > 40) & (gray < 250)
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size:
        top, bottom = int(rows[0]), int(rows[-1])
    else:  # pragma: no cover
        top, bottom = int(0.2 * height), int(0.9 * height)
    shoulder_y = top + int(0.08 * (bottom - top))
    hip_y = top + int(0.55 * (bottom - top))
    shoulder_w = int(0.27 * width)
    cx = width // 2
    for index, (x, y, vis) in {
        11: (cx - shoulder_w // 2, shoulder_y, 0.7), 12: (cx + shoulder_w // 2, shoulder_y, 0.7),
        23: (cx - int(shoulder_w * 0.30), hip_y, 0.6), 24: (cx + int(shoulder_w * 0.30), hip_y, 0.6),
        0: (cx, shoulder_y - int(0.06 * height), 0.6),
    }.items():
        landmarks[index] = (float(x), float(y), 0.0, vis)
    return landmarks


def populate_sample_closet(garment_service, count: int = 10, seed: int = 7) -> List[Dict[str, object]]:
    """Generate + ingest sample garments through the real preprocessing pipeline."""
    import io
    from PIL import Image

    rng = random.Random(seed)
    created: List[Dict[str, object]] = []
    categories = list(GARMENT_TEMPLATES.keys())
    for index in range(count):
        category = categories[index % len(categories)]
        garment = generate_garment(category=category, size=640, seed=rng.randint(0, 10 ** 6))
        # Save as a JPEG on white so the pipeline runs its real background-removal path.
        rgba = garment.image
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        composited = (rgba[..., :3].astype(np.float32) * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(composited).save(buffer, format="PNG")
        try:
            record = garment_service.ingest(
                buffer.getvalue(),
                filename=f"sample_{category}_{index}.png",
                label=f"Sample {category.replace('-', ' ').title()} {index + 1}",
                source={"kind": "generated", "generator": "vestiai.sample_data", "license": "CC0 (procedurally generated)"},
                skip_quality_gate=True,
                tags=["sample-data"],
            )
            created.append(record)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not ingest sample garment %s: %s", category, exc)
    logger.info("Sample closet populated with %d garments", len(created))
    return created


def write_sample_dataset(root: str | Path, train: int = 24, val: int = 6, test: int = 6, size: int = 512, seed: int = 11) -> Dict[str, int]:
    """Write a complete VITON-HD-layout dataset of synthetic pairs.

    Layout produced (identical to the real dataset layout the training scripts expect)::

        root/
          train/{image,cloth,cloth-mask,agnostic,pose}/*.png
          train/pairs.txt
          val/...  test/...
          metadata.json
          LICENSE.txt      (CC0 note for generated data)
    """
    root = Path(root)
    counts = {"train": train, "val": val, "test": test}
    rng = random.Random(seed)
    summary: Dict[str, int] = {}

    for split, count in counts.items():
        for sub in ("image", "cloth", "cloth-mask", "agnostic", "pose"):
            (root / split / sub).mkdir(parents=True, exist_ok=True)
        pairs: List[str] = []
        base = {"train": 0, "val": 1000, "test": 2000}[split]
        for index in range(count):
            identifier = f"{base + index:05d}"
            sample = garment_and_person_pair(size=size, seed=rng.randint(0, 10 ** 6))
            save_image(sample["person"], root / split / "image" / f"{identifier}_00.jpg")
            save_image(sample["garment"][..., :3], root / split / "cloth" / f"{identifier}_00.jpg")
            save_image(sample["cloth_mask"], root / split / "cloth-mask" / f"{identifier}_00.jpg")
            save_image(sample["agnostic"], root / split / "agnostic" / f"{identifier}_00.jpg")
            save_image(sample["pose_map"], root / split / "pose" / f"{identifier}_00.png")
            pairs.append(f"{identifier}_00.jpg {identifier}_00.jpg {str(sample['category'][0])}\n")
        (root / split / "pairs.txt").write_text("".join(pairs), encoding="utf-8")
        summary[split] = count

    (root / "metadata.json").write_text(
        __import__("json").dumps({
            "name": "vestiai-sample",
            "origin": "generated",
            "generator": "vestiai.backend.services.sample_data",
            "size": size,
            "counts": summary,
            "layout": "VITON-HD compatible (image, cloth, cloth-mask, agnostic, pose, pairs.txt)",
            "note": "Procedurally generated for pipeline verification. Replace with real VITON-HD / DressCode "
                    "data (which you download yourself under its licence) for meaningful quality training.",
        }, indent=2),
        encoding="utf-8",
    )
    (root / "LICENSE.txt").write_text(
        "VestiAI sample dataset\n======================\n\n"
        "Every image in this folder was generated procedurally by VestiAI's own code "
        "(backend/services/sample_data.py). No third-party or scraped imagery is included, "
        "so the dataset is free to use for testing the training pipeline (treat as CC0).\n\n"
        "It is intended for smoke-testing ONLY: the images are cartoon-like stylised figures, "
        "so a model trained on them will not produce photorealistic results. Download VITON-HD "
        "or DressCode (see docs/TRAINING.md) for real training.\n",
        encoding="utf-8",
    )
    logger.info("Wrote sample dataset to %s: %s", root, summary)
    return summary


def sample_dataset_available(root: str | Path = "datasets/samples") -> bool:
    return (Path(root) / "train" / "pairs.txt").exists()
