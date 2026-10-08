"""Body and pose feature extraction shared by training and inference.

The diffusion training pipeline conditions on a compact "agnostic" representation of the
body: the person with the garment region erased, plus a posemap. This module builds those
representations from raw landmarks so the dataset preparation script, the training data
loader and the inference adapter all use *identical* code — a common source of train/test
skew in VTON projects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from backend.cv import pose as pose_mod
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

#: Region of interest for the garment on the body, as fractions of the shoulder->hip span.
GARMENT_ROI = {
    "upper": {"top": -0.28, "bottom": 1.35, "width": 1.55},
    "dress": {"top": -0.28, "bottom": 2.30, "width": 1.70},
    "lower": {"top": 0.85, "bottom": 2.60, "width": 1.20},
}


@dataclass
class BodyFeatures:
    """Intermediate body representation used by both training and inference."""

    agnostic_rgb: np.ndarray          # person with garment region blurred/erased
    pose_map: np.ndarray              # (H, W, 3) uint8 posemap
    garment_mask: np.ndarray          # (H, W) uint8, 255 where the garment should go
    bbox: Tuple[int, int, int, int]   # garment ROI box in image coords
    keypoints: np.ndarray             # (33, 4) landmarks in pixels


def _pixel_landmarks(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """Accept both normalised and pixel landmarks; always return pixel coordinates."""
    array = np.asarray(landmarks, dtype=np.float32).copy()
    if array.size == 0:
        return np.zeros((pose_mod.NUM_LANDMARKS, 4), dtype=np.float32)
    finite = array[:, :2][np.isfinite(array[:, :2])]
    if finite.size and finite.max() <= 2.0:  # looks normalised
        array[:, 0] *= width
        array[:, 1] *= height
    return array


def build_pose_map(
    landmarks: np.ndarray,
    size: Tuple[int, int],
    include_torso_fill: bool = True,
) -> np.ndarray:
    """Render a BlazePose-style posemap (stick figure + torso fill) as an RGB uint8 image."""
    height, width = size
    canvas = np.zeros((height, width, 3), np.uint8)
    pixel = _pixel_landmarks(landmarks, width, height)
    if pixel.shape[0] < 25:
        return canvas
    scale = max(1, int(min(width, height) * 0.012))

    for a, b in pose_mod.POSE_CONNECTIONS:
        if pixel[a, 3] < 0.2 or pixel[b, 3] < 0.2:
            continue
        pa = (int(np.clip(pixel[a, 0], 0, width - 1)), int(np.clip(pixel[a, 1], 0, height - 1)))
        pb = (int(np.clip(pixel[b, 0], 0, width - 1)), int(np.clip(pixel[b, 1], 0, height - 1)))
        cv2.line(canvas, pa, pb, (255, 255, 255), scale, cv2.LINE_AA)

    if include_torso_fill:
        quad_indices = [pose_mod.LEFT_SHOULDER, pose_mod.RIGHT_SHOULDER, pose_mod.RIGHT_HIP, pose_mod.LEFT_HIP]
        if all(pixel[i, 3] > 0.2 for i in quad_indices):
            points = np.array([[int(pixel[i, 0]), int(pixel[i, 1])] for i in quad_indices], np.int32)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [points], (60, 60, 140))
            canvas = cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0)

    for index in range(len(pixel)):
        if pixel[index, 3] < 0.2:
            continue
        x, y = int(np.clip(pixel[index, 0], 0, width - 1)), int(np.clip(pixel[index, 1], 0, height - 1))
        cv2.circle(canvas, (x, y), max(1, scale // 2), (255, 255, 255), -1, cv2.LINE_AA)

    # Heat-blur so the network gets smooth gradients rather than 1px lines.
    canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=max(1.0, min(width, height) * 0.006))
    return canvas


def garment_region_bbox(landmarks: np.ndarray, size: Tuple[int, int], kind: str = "upper", pad: float = 0.06) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box of the region a garment should occupy, from pose landmarks."""
    height, width = size
    pixel = _pixel_landmarks(landmarks, width, height)
    left_shoulder = pixel[pose_mod.LEFT_SHOULDER]
    right_shoulder = pixel[pose_mod.RIGHT_SHOULDER]
    left_hip = pixel[pose_mod.LEFT_HIP]
    right_hip = pixel[pose_mod.RIGHT_HIP]
    if left_shoulder[3] < 0.2 or right_shoulder[3] < 0.2:
        return None
    if left_hip[3] < 0.2 or right_hip[3] < 0.2:
        shoulder_span = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))
        hip_y = float((left_shoulder[1] + right_shoulder[1]) / 2.0 + shoulder_span * 1.15)
        center_x = float((left_shoulder[0] + right_shoulder[0]) / 2.0)
        hip_half = shoulder_span * 0.42
        left_hip = np.array([center_x - hip_half, hip_y, 0, 0.3], np.float32)
        right_hip = np.array([center_x + hip_half, hip_y, 0, 0.3], np.float32)

    profile = GARMENT_ROI.get(kind, GARMENT_ROI["upper"])
    shoulder_span = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))
    if shoulder_span < 4:
        return None
    center = (left_shoulder[:2] + right_shoulder[:2] + left_hip[:2] + right_hip[:2]) / 4.0
    top = float(min(left_shoulder[1], right_shoulder[1]) + profile["top"] * shoulder_span)
    bottom = float(max(left_hip[1], right_hip[1]) + (profile["bottom"] - 1.0) * shoulder_span)
    half_w = profile["width"] * shoulder_span * (0.5 + pad)
    x0 = int(max(0, center[0] - half_w))
    x1 = int(min(width, center[0] + half_w))
    y0 = int(max(0, top))
    y1 = int(min(height, bottom))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def garment_mask_from_bbox(shape: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Soft garment mask (255 inside the ROI, feathered at the edge)."""
    height, width = shape[:2]
    mask = np.zeros((height, width), np.uint8)
    x0, y0, x1, y1 = bbox
    cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, (x1 - x0) * 0.02))
    return mask


def agnostic_image(rgb: np.ndarray, mask: np.ndarray, mode: str = "grey") -> np.ndarray:
    """Erase the garment region to build the model's "agnostic" input.

    ``grey`` replaces the region with the mean body colour (matches VITON-HD's agnostic
    convention closely enough for fine-tuning), ``blur`` blurs, ``white`` fills white.
    """
    image = np.asarray(rgb, np.uint8).copy()
    binary = (np.asarray(mask) > 96).astype(np.uint8)
    if binary.sum() == 0:
        return image
    if mode == "white":
        image[binary > 0] = 255
        return image
    if mode == "blur":
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=24)
        image[binary > 0] = blurred[binary > 0]
        return image
    # grey
    region = binary > 0
    mean = image[region].reshape(-1, 3).mean(axis=0) if region.sum() else np.array([128, 128, 128])
    image[region] = mean.astype(np.uint8)
    return image


def build_body_features(
    rgb: np.ndarray,
    landmarks: np.ndarray,
    kind: str = "upper",
    garment_mask: Optional[np.ndarray] = None,
    erase_mode: str = "grey",
) -> Optional[BodyFeatures]:
    """Assemble the full conditioning package for a person image."""
    rgb = np.asarray(rgb, np.uint8)
    height, width = rgb.shape[:2]
    pixel = _pixel_landmarks(landmarks, width, height)
    bbox = garment_region_bbox(pixel, (height, width), kind=kind)
    if bbox is None:
        return None
    if garment_mask is None:
        garment_mask = garment_mask_from_bbox((height, width), bbox)
    elif garment_mask.shape[:2] != (height, width):
        garment_mask = cv2.resize(garment_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    agnostic = agnostic_image(rgb, garment_mask, mode=erase_mode)
    pose_map = build_pose_map(pixel, (height, width))
    return BodyFeatures(
        agnostic_rgb=agnostic,
        pose_map=pose_map,
        garment_mask=(garment_mask > 96).astype(np.uint8) * 255,
        bbox=bbox,
        keypoints=pixel,
    )


def resize_conditioning(features: BodyFeatures, size: Tuple[int, int]) -> BodyFeatures:
    """Resize all conditioning images to the model resolution (w, h)."""
    width, height = size
    return BodyFeatures(
        agnostic_rgb=cv2.resize(features.agnostic_rgb, (width, height), interpolation=cv2.INTER_AREA if width < features.agnostic_rgb.shape[1] else cv2.INTER_CUBIC),
        pose_map=cv2.resize(features.pose_map, (width, height), interpolation=cv2.INTER_AREA),
        garment_mask=cv2.resize(features.garment_mask, (width, height), interpolation=cv2.INTER_NEAREST),
        bbox=features.bbox,
        keypoints=features.keypoints,
    )


def pack_conditioning_tensor(features: BodyFeatures, size: Tuple[int, int], normalize: bool = True) -> np.ndarray:
    """Stack agnostic image + posemap + garment mask into a single ``(9, H, W)`` array.

    Layout (channels): 0-2 agnostic RGB, 3-5 posemap RGB, 6-8 garment mask replicated —
    the same packing the training scripts use, so inference and training agree bit-for-bit.
    """
    resized = resize_conditioning(features, size)
    agnostic = resized.agnostic_rgb.astype(np.float32)
    pose = resized.pose_map.astype(np.float32)
    mask = resized.garment_mask.astype(np.float32)
    if normalize:
        agnostic = agnostic / 127.5 - 1.0
        pose = pose / 127.5 - 1.0
        mask = mask / 127.5 - 1.0
    mask = np.repeat(mask[..., None], 3, axis=-1)
    return np.concatenate([agnostic, pose, mask], axis=-1).transpose(2, 0, 1)


def summarize_features(features: BodyFeatures) -> Dict[str, object]:
    """Small JSON-serialisable summary for logging."""
    x0, y0, x1, y1 = features.bbox
    return {
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
        "garment_mask_coverage": round(float((features.garment_mask > 127).mean()), 4),
        "pose_visible_landmarks": int((features.keypoints[:, 3] > 0.2).sum()),
    }


def landmarks_from_json(payload: Sequence[Dict[str, float]]) -> np.ndarray:
    """Convert the frontend's landmark JSON list into a ``(33, 4)`` array."""
    array = np.zeros((pose_mod.NUM_LANDMARKS, 4), dtype=np.float32)
    for item in payload:
        index = int(item.get("index", -1))
        if 0 <= index < pose_mod.NUM_LANDMARKS:
            array[index] = (
                float(item.get("x", 0.0)), float(item.get("y", 0.0)),
                float(item.get("z", 0.0)), float(item.get("visibility", 0.0)),
            )
    return array
