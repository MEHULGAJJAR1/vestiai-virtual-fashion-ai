"""Occlusion handling for the real-time warp pipeline.

When the wearer's forearm/hand is *in front of* the torso, a naive overlay would draw the
garment on top of the arm — the classic "cardboard" artefact. Real VTON systems solve this
with a person-agnostic parsing mask. Live, we approximate it well enough to look right:

* :func:`arm_occlusion_boxes` builds capsules around limbs whose BlazePose ``z`` says they
  are closer to the camera than the shoulder plane.
* :func:`apply_occlusion` erases (or feathers) those regions from the garment alpha so the
  arm shows through.
* :func:`hair_and_head_regions` protects the face/neck so collars never cover the chin.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from backend.cv import pose as pose_mod
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

Box = Tuple[int, int, int, int]


def _limb_visibility(landmarks: np.ndarray, indices: Sequence[int], minimum: float = 0.35) -> bool:
    return all(landmarks[i, 3] >= minimum for i in indices if i < len(landmarks))


def arm_occlusion_boxes(
    landmarks: np.ndarray,
    frame_shape: Tuple[int, int],
    depth_margin: float = 0.15,
) -> List[Box]:
    """Return capsules around limb segments that sit **in front of** the torso plane.

    Depth comes from the BlazePose ``z`` channel (smaller = closer to the camera, normalised
    by shoulder width). Rather than boxing the whole arm — which would erase perfectly
    visible sleeves — the arm is split into two segments and each is judged independently:

    * ``shoulder -> elbow`` is treated as occluding only when the **elbow** is in front
      (e.g. folded arms),
    * ``elbow -> wrist`` only when the **wrist** is in front (e.g. a hand resting on the hip),
    * plus a small capsule around the hand itself.

    A person standing with arms at their sides therefore keeps their sleeves while a hand
    crossing the body correctly punches through the garment.
    """
    height, width = frame_shape[:2]
    boxes: List[Box] = []
    if landmarks is None or len(landmarks) < 17:
        return boxes

    left_shoulder = landmarks[pose_mod.LEFT_SHOULDER]
    right_shoulder = landmarks[pose_mod.RIGHT_SHOULDER]
    if left_shoulder[3] < 0.3 or right_shoulder[3] < 0.3:
        return boxes
    torso_z = float((left_shoulder[2] + right_shoulder[2]) / 2.0)
    shoulder_span = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])) + 1e-6

    def depth_delta(index: int) -> float:
        return (torso_z - float(landmarks[index][2])) / shoulder_span

    def make_box(points: Sequence[Tuple[float, float]], pad: float) -> Box:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (
            int(max(0, min(xs) - pad)), int(max(0, min(ys) - pad)),
            int(min(width, max(xs) + pad)), int(min(height, max(ys) + pad)),
        )

    chains = (
        (pose_mod.LEFT_SHOULDER, pose_mod.LEFT_ELBOW, pose_mod.LEFT_WRIST),
        (pose_mod.RIGHT_SHOULDER, pose_mod.RIGHT_ELBOW, pose_mod.RIGHT_WRIST),
    )
    for shoulder, elbow, wrist in chains:
        visible = {index: float(landmarks[index][3]) for index in (shoulder, elbow, wrist)}
        segment_pad = shoulder_span * 0.09

        # upper arm: only when the elbow is genuinely in front of the torso
        if visible[elbow] > 0.3 and depth_delta(elbow) > depth_margin:
            boxes.append(make_box(
                [(float(landmarks[shoulder][0]), float(landmarks[shoulder][1])),
                 (float(landmarks[elbow][0]), float(landmarks[elbow][1]))],
                segment_pad,
            ))

        # forearm: only when the wrist is in front
        if visible[wrist] > 0.25 and depth_delta(wrist) > depth_margin:
            if visible[elbow] > 0.25:
                boxes.append(make_box(
                    [(float(landmarks[elbow][0]), float(landmarks[elbow][1])),
                     (float(landmarks[wrist][0]), float(landmarks[wrist][1]))],
                    segment_pad,
                ))
            # the hand itself
            hand_pad = shoulder_span * 0.16
            boxes.append(make_box(
                [(float(landmarks[wrist][0]), float(landmarks[wrist][1]))],
                hand_pad,
            ))
    return boxes


def capsules_from_boxes(boxes: Iterable[Box], frame_shape: Tuple[int, int], softness: int = 9) -> np.ndarray:
    """Rasterise boxes as feathered ellipses into a 0..1 occlusion weight map."""
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), np.float32)
    for (x0, y0, x1, y1) in boxes:
        center = (int((x0 + x1) / 2), int((y0 + y1) / 2))
        axes = (max(2, int((x1 - x0) / 2)), max(2, int((y1 - y0) / 2)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, thickness=-1, lineType=cv2.LINE_AA)
    if softness > 0:
        ksize = int(softness) * 2 + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    return np.clip(mask, 0.0, 1.0)


def head_region_mask(landmarks: np.ndarray, frame_shape: Tuple[int, int], padding: float = 0.25) -> Optional[np.ndarray]:
    """Feathered ellipse covering the head/neck so collars never eat the chin."""
    height, width = frame_shape[:2]
    nose = landmarks[pose_mod.NOSE] if landmarks is not None and len(landmarks) > pose_mod.NOSE else None
    left_ear = landmarks[pose_mod.LEFT_EAR] if landmarks is not None else None
    right_ear = landmarks[pose_mod.RIGHT_EAR] if landmarks is not None else None
    if nose is None or nose[3] < 0.3:
        return None

    if left_ear is not None and right_ear is not None and left_ear[3] > 0.25 and right_ear[3] > 0.25:
        span = float(np.linalg.norm(left_ear[:2] - right_ear[:2]))
        center = ((left_ear[0] + right_ear[0]) / 2.0, (left_ear[1] + right_ear[1]) / 2.0)
    else:
        span = float(nose[2]) if float(nose[2]) > 1.0 else 60.0
        center = (float(nose[0]), float(nose[1]))

    radius = max(12.0, span * (0.75 + padding))
    mask = np.zeros((height, width), np.float32)
    cv2.ellipse(mask, (int(center[0]), int(center[1])), (int(radius), int(radius * 1.15)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2.0, radius * 0.18))
    return np.clip(mask, 0.0, 1.0)


def apply_occlusion(
    alpha: np.ndarray,
    landmarks: np.ndarray,
    frame_shape: Tuple[int, int],
    protect_head: bool = True,
    depth_margin: float = 0.12,
    strength: float = 1.0,
) -> np.ndarray:
    """Return the garment alpha with occluding limbs (and the head) cut out."""
    result = np.asarray(alpha, dtype=np.float32).copy()
    if result.max() <= 1.5:
        result *= 255.0

    boxes = arm_occlusion_boxes(landmarks, frame_shape, depth_margin=depth_margin)
    if boxes:
        occlusion = capsules_from_boxes(boxes, frame_shape)
        result = result * (1.0 - np.clip(occlusion * strength, 0.0, 1.0))

    if protect_head:
        head = head_region_mask(landmarks, frame_shape)
        if head is not None:
            # Only remove the part of the garment that is *above* the collar area.
            result = result * (1.0 - np.clip(head * 0.85, 0.0, 1.0))

    return np.clip(result, 0, 255).astype(np.uint8)


def occluded_fraction(before: np.ndarray, after: np.ndarray) -> float:
    """How much of the garment was hidden by occlusion handling (for diagnostics)."""
    b = (np.asarray(before) > 32).sum()
    a = (np.asarray(after) > 32).sum()
    return float(max(0.0, (b - a) / b)) if b else 0.0


def occlusion_report(landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> dict:
    """Summarise occlusion state for the live status panel."""
    boxes = arm_occlusion_boxes(landmarks, frame_shape)
    return {
        "arm_occlusions": len(boxes),
        "boxes": [[int(v) for v in box] for box in boxes],
        "head_protected": head_region_mask(landmarks, frame_shape) is not None,
    }
