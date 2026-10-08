"""Garment alignment: garment anchors -> pose landmarks -> affine/perspective warp.

This is the geometric core of the **real-time lightweight pipeline**. The maths is plain
2-D projective geometry (the same family of transforms used by classical VITON warping
modules such as the second-order / TPS warps in CP-VTON), implemented with OpenCV so it
runs at hundreds of FPS on a CPU:

1. :class:`GarmentAnchors` extracts stable anchor points from the garment alpha channel
   (shoulder corners, armpit line, hem corners, centre line).
2. :class:`PoseAnchors` extracts the corresponding body anchors from BlazePose landmarks.
3. :func:`estimate_transform` solves a partial-affine (4-DoF) or full perspective (8-DoF)
   transform between them, with an aspect-preserving similarity fallback when too few
   correspondences are visible.
4. :func:`warp_garment` rasterises the garment with bilinear/cubic interpolation plus a
   feathered alpha edge.

Every function is pure (no globals, no I/O) which keeps them unit-testable — see
``tests/test_pose_transforms.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from backend.cv import pose as pose_mod
from backend.utils.errors import UnsupportedGarmentError
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

Point = Tuple[float, float]


# --------------------------------------------------------------------------------------
# Category geometry profiles
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GarmentProfile:
    """How a garment category scales and anchors onto a body.

    Values are multiples of the shoulder width / torso length measured from the pose.
    They were tuned against upper-body garments in the sample dataset and are exposed so
    the Settings page can fine-tune fit per category.
    """

    key: str
    label: str
    width_factor: float = 1.28       # garment width / shoulder width
    length_factor: float = 1.45      # garment length / (shoulder->hip) distance
    shoulder_offset: float = 0.06    # where the garment's neckline sits, relative to torso
    hem_anchor: str = "hip"          # hip | mid_thigh | knee | ankle
    covers_arms: bool = True
    supports_bottom: bool = False


PROFILES: Dict[str, GarmentProfile] = {
    "t-shirt": GarmentProfile("t-shirt", "T-Shirt", 1.30, 1.30, 0.05, "hip", True),
    "shirt": GarmentProfile("shirt", "Shirt", 1.26, 1.40, 0.06, "hip", True),
    "jacket": GarmentProfile("jacket", "Jacket", 1.42, 1.60, 0.04, "mid_thigh", True),
    "kurta": GarmentProfile("kurta", "Kurta", 1.32, 2.05, 0.05, "mid_thigh", True),
    "dress": GarmentProfile("dress", "Dress", 1.34, 2.60, 0.05, "knee", True),
    "traditional": GarmentProfile("traditional", "Traditional", 1.45, 2.30, 0.04, "mid_thigh", True),
    "formal": GarmentProfile("formal", "Formal", 1.28, 1.42, 0.06, "hip", True),
    "sweater": GarmentProfile("sweater", "Sweater", 1.34, 1.35, 0.04, "hip", True),
    "top": GarmentProfile("top", "Top", 1.22, 1.15, 0.07, "hip", True),
    "bottom": GarmentProfile("bottom", "Bottom", 1.10, 2.10, 0.00, "ankle", False, True),
    "shoes": GarmentProfile("shoes", "Shoes", 0.45, 0.20, 0.00, "ankle", False, True),
    "accessory": GarmentProfile("accessory", "Accessory", 0.60, 0.35, 0.02, "shoulder", False, False),
    "unknown": GarmentProfile("unknown", "Garment", 1.28, 1.40, 0.05, "hip", True),
}

HEM_FACTOR = {"hip": 1.0, "mid_thigh": 1.45, "knee": 2.05, "ankle": 2.75, "shoulder": 0.0}


def get_profile(category: str) -> GarmentProfile:
    """Look up a profile by category key, defaulting to ``unknown``."""
    key = (category or "unknown").strip().lower().replace(" ", "-")
    return PROFILES.get(key, PROFILES["unknown"])


# --------------------------------------------------------------------------------------
# Anchors
# --------------------------------------------------------------------------------------
@dataclass
class GarmentAnchors:
    """Anchor points expressed in garment-canvas pixel coordinates."""

    top_left: Point
    top_right: Point
    neck: Point
    bottom_left: Point
    bottom_right: Point
    center: Point
    width: float
    height: float
    bbox: Tuple[int, int, int, int]

    def as_source_points(self, use_perspective: bool = True) -> np.ndarray:
        """Source points for the transform, ordered to match :meth:`PoseAnchors.dest_points`."""
        if use_perspective:
            return np.asarray([self.top_left, self.top_right, self.bottom_right, self.bottom_left], dtype=np.float32)
        return np.asarray([self.top_left, self.top_right, self.bottom_right], dtype=np.float32)

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "top_left": list(self.top_left), "top_right": list(self.top_right), "neck": list(self.neck),
            "bottom_left": list(self.bottom_left), "bottom_right": list(self.bottom_right),
            "center": list(self.center), "width": [self.width], "height": [self.height],
        }


@dataclass
class PoseAnchors:
    """Anchor points expressed in frame pixel coordinates."""

    top_left: Point
    top_right: Point
    neck: Point
    bottom_left: Point
    bottom_right: Point
    center: Point
    shoulder_width: float
    torso_length: float
    confidence: float = 1.0

    def dest_points(self, use_perspective: bool = True) -> np.ndarray:
        if use_perspective:
            return np.asarray([self.top_left, self.top_right, self.bottom_right, self.bottom_left], dtype=np.float32)
        return np.asarray([self.top_left, self.top_right, self.bottom_right], dtype=np.float32)


def extract_garment_anchors(alpha: np.ndarray, width_bias: float = 0.5) -> GarmentAnchors:
    """Derive garment anchors from an alpha mask (0..255 or 0..1).

    The top edge of the silhouette gives the shoulder line; the widest row in the upper
    third is used as a robust shoulder estimate, which handles garments photographed on a
    hanger (narrow top, wide body) sensibly.
    """
    mask = np.asarray(alpha)
    if mask.dtype != np.uint8:
        mask = (np.clip(mask.astype(np.float32), 0, 1) * 255).astype(np.uint8)
    binary = (mask > 96).astype(np.uint8)
    if binary.sum() == 0:
        raise UnsupportedGarmentError("The garment mask is empty — nothing to align.")

    from backend.utils.image_utils import bbox_from_mask

    # NOTE: pass the 0/255 mask (not the 0/1 binary) because ``bbox_from_mask`` scales its
    # threshold for uint8 input.
    bbox = bbox_from_mask(binary * 255)
    if bbox is None:
        raise UnsupportedGarmentError("The garment mask is empty — nothing to align.")
    x0, y0, x1, y1 = bbox
    height = float(y1 - y0)

    # Widest row within the upper 35% gives the shoulder span (ignores sleeves hanging down).
    search_end = int(y0 + max(4.0, height * 0.35))
    widths: List[Tuple[float, float, float]] = []
    for row in range(y0, min(search_end, y1)):
        columns = np.flatnonzero(binary[row])
        if columns.size:
            widths.append((float(columns[-1] - columns[0]), float(columns[0]), float(columns[-1])))
    if widths:
        span, left_edge, right_edge = max(widths, key=lambda item: item[0])
    else:  # pragma: no cover - degenerate
        span, left_edge, right_edge = float(x1 - x0), float(x0), float(x1)

    top_row = y0
    bottom_row = y1
    # NOTE: reduce the band to a 1-D column profile first — calling flatnonzero on a 2-D
    # slice would return row-major flattened indices, not column indices.
    hem_band = binary[max(y0, y1 - max(2, int(0.03 * height))) : y1]
    bottom_columns = np.flatnonzero(hem_band.any(axis=0))
    if bottom_columns.size:
        hem_left, hem_right = float(bottom_columns.min()), float(bottom_columns.max())
    else:  # pragma: no cover
        hem_left, hem_right = float(x0), float(x1)

    # Keep every anchor inside the silhouette bbox: a single stray pixel must not be able to
    # skew the perspective solve.
    hem_left = float(min(max(hem_left, x0), x1))
    hem_right = float(min(max(hem_right, x0), x1))
    left_edge = float(min(max(left_edge, x0), x1))
    right_edge = float(min(max(right_edge, x0), x1))
    if right_edge - left_edge < 4:  # pragma: no cover - degenerate mask
        left_edge, right_edge = float(x0), float(x1)

    neck_y = y0 + 0.04 * height
    neck = (float((x0 + x1) / 2.0), float(neck_y))

    return GarmentAnchors(
        top_left=(left_edge, float(top_row)),
        top_right=(right_edge, float(top_row)),
        neck=(float((x0 + x1) / 2.0), float(neck_y)),
        bottom_left=(hem_left, float(bottom_row)),
        bottom_right=(hem_right, float(bottom_row)),
        center=(float((x0 + x1) / 2.0), float((y0 + y1) / 2.0)),
        width=float(span),
        height=height,
        bbox=(int(x0), int(y0), int(x1), int(y1)),
    )


def _landmark_xy(landmarks: np.ndarray, index: int, min_visibility: float = 0.25) -> Optional[np.ndarray]:
    if landmarks is None or index >= len(landmarks):
        return None
    x, y, _z, visibility = landmarks[index]
    if visibility < min_visibility:
        return None
    return np.asarray([x, y], dtype=np.float32)


def extract_pose_anchors(
    landmarks: np.ndarray,
    profile: GarmentProfile,
    frame_shape: Tuple[int, int],
    allow_heuristic: bool = True,
) -> Optional[PoseAnchors]:
    """Map BlazePose landmarks onto body anchors for a given garment profile.

    Uses the image-space landmarks; ``z`` is used only for occlusion reasoning elsewhere.
    Returns ``None`` when the torso is not visible enough to place a garment credibly.
    """
    height, width = frame_shape[:2]
    left_shoulder = _landmark_xy(landmarks, pose_mod.LEFT_SHOULDER)
    right_shoulder = _landmark_xy(landmarks, pose_mod.RIGHT_SHOULDER)
    if left_shoulder is None or right_shoulder is None:
        return None

    center_x = float((left_shoulder[0] + right_shoulder[0]) / 2.0)
    shoulder_y = float((left_shoulder[1] + right_shoulder[1]) / 2.0)
    shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))
    if shoulder_width < 8.0:
        return None

    left_hip = _landmark_xy(landmarks, pose_mod.LEFT_HIP)
    right_hip = _landmark_xy(landmarks, pose_mod.RIGHT_HIP)
    if left_hip is not None and right_hip is not None:
        hip_y = float((left_hip[1] + right_hip[1]) / 2.0)
        hip_x = float((left_hip[0] + right_hip[0]) / 2.0)
        torso_length = abs(hip_y - shoulder_y)
        # Slight lean: rotate the garment with the torso.
        torso_angle = math.atan2(hip_x - center_x, max(1e-3, hip_y - shoulder_y))
        lean = math.degrees(torso_angle) * 0.35
    else:
        torso_length = shoulder_width * 1.15
        hip_y = shoulder_y + torso_length
        lean = 0.0

    torso_length = max(torso_length, shoulder_width * 0.85)

    # Garment top slightly above the shoulder line so the neckline covers the collar.
    top_y = shoulder_y - profile.shoulder_offset * shoulder_width * 0.9
    half_width = shoulder_width * profile.width_factor / 2.0

    hem_extra = HEM_FACTOR.get(profile.hem_anchor, 1.0)
    hem_y = top_y + shoulder_width * profile.length_factor * 0.75 * (1.0 + 0.35 * (hem_extra - 1.0)) + torso_length * 0.25
    if profile.hem_anchor == "shoulder":
        hem_y = top_y + shoulder_width * 0.5

    # Rotate corner offsets by the torso lean so a tilted body tilts the garment too.
    theta = math.radians(lean)

    def rotate(dx: float, dy: float) -> Point:
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        return (center_x + dx * cos_t - dy * sin_t, top_y + dx * sin_t + dy * cos_t)

    top_left = rotate(-half_width, 0.0)
    top_right = rotate(half_width, 0.0)
    bottom_left = rotate(-half_width * 1.02, hem_y - top_y)
    bottom_right = rotate(half_width * 1.02, hem_y - top_y)
    neck = rotate(0.0, 0.0)

    confidence = float(np.mean(landmarks[[pose_mod.LEFT_SHOULDER, pose_mod.RIGHT_SHOULDER, pose_mod.LEFT_HIP, pose_mod.RIGHT_HIP], 3]))

    anchors = PoseAnchors(
        top_left=top_left,
        top_right=top_right,
        neck=neck,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        center=(center_x, (top_y + hem_y) / 2.0),
        shoulder_width=shoulder_width,
        torso_length=torso_length,
        confidence=confidence,
    )

    if allow_heuristic:
        # Keep anchors inside the frame with a small margin so the warp never clips entirely.
        if not (-width <= anchors.center[0] <= 2 * width and -height <= anchors.center[1] <= 2 * height):
            return None
    return anchors


# --------------------------------------------------------------------------------------
# Transform estimation
# --------------------------------------------------------------------------------------
@dataclass
class WarpResult:
    """Output of :func:`warp_garment`."""

    rgba: np.ndarray                 # warped RGBA at frame resolution
    alpha: np.ndarray                # warped alpha (uint8)
    matrix: np.ndarray               # 2x3 (affine) or 3x3 (perspective)
    mode: str                        # "affine" | "perspective" | "similarity"
    coverage: float = 0.0


def estimate_transform(
    source: np.ndarray,
    target: np.ndarray,
    allow_perspective: bool = True,
) -> Tuple[np.ndarray, str]:
    """Estimate the garment->body transform.

    Uses a full 8-DoF perspective transform when 4 correspondences are supplied,
    otherwise a 4-DoF partial affine (rotation+scale+translation) which is more stable
    under landmark noise.
    """
    source = np.asarray(source, dtype=np.float32).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float32).reshape(-1, 2)
    if source.shape != target.shape:
        raise ValueError("Source and target point sets must have identical shape.")

    if allow_perspective and len(source) >= 4:
        matrix = cv2.getPerspectiveTransform(source[:4], target[:4])
        if np.all(np.isfinite(matrix)):
            return matrix, "perspective"

    if len(source) >= 2:
        matrix, _inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)
        if matrix is not None and np.all(np.isfinite(matrix)):
            return matrix.astype(np.float32), "affine"

    raise UnsupportedGarmentError("Not enough visible body anchors to align the garment.")


def warp_garment(
    garment_rgb: np.ndarray,
    garment_alpha: np.ndarray,
    matrix: np.ndarray,
    output_size: Tuple[int, int],
    interpolation: int = cv2.INTER_CUBIC,
) -> WarpResult:
    """Rasterise the garment onto a frame-sized canvas with the given transform."""
    out_w, out_h = output_size
    rgb = np.asarray(garment_rgb)[..., :3]
    alpha = np.asarray(garment_alpha)
    if alpha.max() <= 1.5:
        alpha = alpha * 255.0
    alpha = alpha.astype(np.float32)

    if matrix.shape == (3, 3):
        warped_rgb = cv2.warpPerspective(rgb, matrix, (out_w, out_h), flags=interpolation, borderMode=cv2.BORDER_CONSTANT)
        warped_alpha = cv2.warpPerspective(alpha, matrix, (out_w, out_h), flags=interpolation, borderMode=cv2.BORDER_CONSTANT)
        mode = "perspective"
    else:
        warped_rgb = cv2.warpAffine(rgb, matrix, (out_w, out_h), flags=interpolation, borderMode=cv2.BORDER_CONSTANT)
        warped_alpha = cv2.warpAffine(alpha, matrix, (out_w, out_h), flags=interpolation, borderMode=cv2.BORDER_CONSTANT)
        mode = "affine"

    warped_alpha_u8 = np.clip(warped_alpha, 0, 255).astype(np.uint8)
    coverage = float((warped_alpha_u8 > 32).mean())
    return WarpResult(
        rgba=np.dstack([np.clip(warped_rgb, 0, 255).astype(np.uint8), warped_alpha_u8]),
        alpha=warped_alpha_u8,
        matrix=np.asarray(matrix, dtype=np.float32),
        mode=mode,
        coverage=coverage,
    )


def compute_fit_transform(
    garment_alpha: np.ndarray,
    landmarks: np.ndarray,
    frame_shape: Tuple[int, int],
    category: str = "t-shirt",
    allow_perspective: bool = True,
    category_overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> Optional[Tuple[np.ndarray, str, PoseAnchors, GarmentAnchors]]:
    """Full pipeline: garment alpha + pose landmarks -> transform.

    Returns ``None`` when the body is not visible enough (the caller then hides the garment).
    """
    profile = get_profile(category)
    if category_overrides and category in category_overrides:
        override = category_overrides[category]
        profile = GarmentProfile(
            key=profile.key,
            label=profile.label,
            width_factor=float(override.get("width_factor", profile.width_factor)),
            length_factor=float(override.get("length_factor", profile.length_factor)),
            shoulder_offset=float(override.get("shoulder_offset", profile.shoulder_offset)),
            hem_anchor=str(override.get("hem_anchor", profile.hem_anchor)),
            covers_arms=profile.covers_arms,
            supports_bottom=profile.supports_bottom,
        )

    pose_anchors = extract_pose_anchors(landmarks, profile, frame_shape)
    if pose_anchors is None:
        return None
    garment_anchors = extract_garment_anchors(garment_alpha)
    matrix, mode = estimate_transform(
        garment_anchors.as_source_points(allow_perspective),
        pose_anchors.dest_points(allow_perspective),
        allow_perspective=allow_perspective,
    )
    return matrix, mode, pose_anchors, garment_anchors


def solve_scale_for_anchor(
    frame_shape: Tuple[int, int],
    shoulder_width: float,
    profile: GarmentProfile,
    garment_bbox_height: float,
) -> float:
    """Utility: uniform scale that would fit a garment onto a body (used for previews)."""
    _ = frame_shape
    target_height = shoulder_width * profile.length_factor
    return float(target_height / max(1.0, garment_bbox_height))


def blend_rgba_over_frame(frame_rgb: np.ndarray, warped: np.ndarray, alpha_override: Optional[np.ndarray] = None, feather: float = 1.2) -> np.ndarray:
    """Alpha-composite a warped garment over the camera frame with a soft edge."""
    frame = np.asarray(frame_rgb, dtype=np.float32)
    overlay_rgb = np.asarray(warped[..., :3], dtype=np.float32)
    alpha = warped[..., 3] if warped.shape[-1] == 4 else np.asarray(alpha_override)
    alpha = np.asarray(alpha, dtype=np.float32) / 255.0
    if feather > 0:
        # OpenCV requires an odd, positive kernel size — never feed it a rounded even value.
        ksize = max(1, int(round(feather * 2)) | 1)
        if ksize > 1:
            alpha = cv2.GaussianBlur(alpha, (ksize, ksize), 0)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    blended = frame * (1.0 - alpha) + overlay_rgb * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def transform_points(points: Sequence[Point], matrix: np.ndarray) -> np.ndarray:
    """Apply a 2x3 or 3x3 transform to 2-D points (handy for tests & debug overlays)."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if np.asarray(matrix).shape == (3, 3):
        out = cv2.perspectiveTransform(pts, np.asarray(matrix, dtype=np.float32))
    else:
        out = cv2.transform(pts, np.asarray(matrix, dtype=np.float32))
    return out.reshape(-1, 2)


def coverage_ratio(alpha: np.ndarray) -> float:
    """Fraction of the frame covered by the garment alpha (sanity metric for live mode)."""
    array = np.asarray(alpha)
    return float((array > 32).mean()) if array.size else 0.0


@dataclass
class AlignmentState:
    """Book-keeping the live service keeps between frames."""

    last_matrix: Optional[np.ndarray] = None
    last_anchors: Optional[PoseAnchors] = None
    lost_frames: int = 0
    history: List[float] = field(default_factory=list)
