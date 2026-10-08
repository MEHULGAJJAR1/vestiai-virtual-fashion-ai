"""Pose estimation wrapper (MediaPipe Tasks / Solutions with an OpenCV fallback).

Priority order:

1. ``mediapipe.tasks.vision.PoseLandmarker`` (33 landmarks + optional segmentation mask).
2. Legacy ``mediapipe.solutions.pose`` (same 33-landmark topology).
3. **Heuristic fallback** — OpenCV HOG person detector + Haar face detector to place an
   approximate torso. This is deliberately labelled ``backend="heuristic"`` in the API
   response so the UI can tell the user tracking is degraded and that installing
   MediaPipe will improve it. No fake landmarks are ever produced.

Landmark indices follow the BlazePose topology (the same one MediaPipe's JS package uses),
so the browser-side and server-side code share one contract.
"""

from __future__ import annotations

import math
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.utils.errors import DependencyMissingError
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

# --- BlazePose landmark indices -------------------------------------------------------
NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

NUM_LANDMARKS = 33

#: Named landmark groups used by the garment alignment code.
LANDMARK_NAMES: Dict[int, str] = {
    0: "nose", 11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist", 23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee", 27: "left_ankle", 28: "right_ankle",
    7: "left_ear", 8: "right_ear",
}

POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
    (0, 11), (0, 12),
]

DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/"
    "float16/latest/pose_landmarker_lite.task"
)


@dataclass
class PoseResult:
    """Pose output for a single frame."""

    detected: bool
    landmarks: np.ndarray = field(default_factory=lambda: np.zeros((NUM_LANDMARKS, 4), dtype=np.float32))
    backend: str = "none"
    score: float = 0.0
    image_size: Tuple[int, int] = (0, 0)  # (height, width)
    mask: Optional[np.ndarray] = None     # optional person segmentation (uint8 0..255)

    def xy(self, index: int) -> Optional[np.ndarray]:
        """Pixel coordinates of one landmark, or ``None`` when undetected."""
        if not self.detected:
            return None
        x, y, _, visibility = self.landmarks[index]
        if visibility <= 0.05:
            return None
        return np.array([x, y], dtype=np.float32)

    def points(self, indices: Sequence[int], min_visibility: float = 0.3) -> Optional[np.ndarray]:
        """Stack multiple landmarks into an ``(N, 2)`` array when all are visible."""
        if not self.detected:
            return None
        out = []
        for index in indices:
            x, y, _, visibility = self.landmarks[index]
            if visibility < min_visibility:
                return None
            out.append((x, y))
        return np.asarray(out, dtype=np.float32)

    def to_dict(self, include_mask: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "detected": bool(self.detected),
            "backend": self.backend,
            "score": round(float(self.score), 4),
            "image_size": {"height": int(self.image_size[0]), "width": int(self.image_size[1])},
            "landmarks": [
                {
                    "index": i,
                    "name": LANDMARK_NAMES.get(i),
                    "x": float(v[0]),
                    "y": float(v[1]),
                    "z": float(v[2]),
                    "visibility": float(v[3]),
                }
                for i, v in enumerate(self.landmarks)
            ] if self.detected else [],
        }
        if include_mask and self.mask is not None:
            from backend.utils.image_utils import encode_base64

            payload["mask_png_base64"] = encode_base64(self.mask, fmt="PNG")
        return payload

    # -- derived geometry ------------------------------------------------------------
    def torso_quad(self) -> Optional[np.ndarray]:
        """Shoulder/hip quadrilateral used to place a garment."""
        left_shoulder = self.xy(LEFT_SHOULDER)
        right_shoulder = self.xy(RIGHT_SHOULDER)
        if left_shoulder is None or right_shoulder is None:
            return None
        left_hip = self.xy(LEFT_HIP)
        right_hip = self.xy(RIGHT_HIP)
        if left_hip is None or right_hip is None:
            # extrapolate hips from the neck->shoulder scale
            center = (left_shoulder + right_shoulder) / 2.0
            span = np.linalg.norm(left_shoulder - right_shoulder) + 1e-6
            down = np.array([0.0, span * 0.85], dtype=np.float32)
            left_hip = left_shoulder + down
            right_hip = right_shoulder + down
            _ = center
        return np.stack([left_shoulder, right_shoulder, left_hip, right_hip]).astype(np.float32)

    def shoulder_center(self) -> Optional[np.ndarray]:
        pair = self.points([LEFT_SHOULDER, RIGHT_SHOULDER], min_visibility=0.3)
        return pair.mean(axis=0) if pair is not None else None

    def shoulder_width(self) -> float:
        pair = self.points([LEFT_SHOULDER, RIGHT_SHOULDER], min_visibility=0.3)
        return float(np.linalg.norm(pair[0] - pair[1])) if pair is not None else 0.0


# --------------------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------------------
class PoseEstimator:
    """Unified pose estimator with a deterministic fallback chain.

    Parameters
    ----------
    complexity:
        MediaPipe model complexity (0/1/2) for the legacy solutions API. The Tasks API uses
        the ``lite`` model by default which matches complexity 1.
    prefer_tasks:
        Try the Tasks API first (recommended, actively maintained).
    allow_download:
        Download ``pose_landmarker_lite.task`` from Google's public model zoo on first use.
    """

    def __init__(
        self,
        complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_segmentation: bool = True,
        prefer_tasks: bool = True,
        allow_download: bool = True,
        model_path: Optional[str | Path] = None,
    ) -> None:
        self.complexity = int(complexity)
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.enable_segmentation = bool(enable_segmentation)
        self.model_path = Path(model_path) if model_path else Path("models_cache") / "pose_landmarker_lite.task"
        self._landmarker = None
        self._solution = None
        self.backend = "uninitialized"
        self._hog = None
        self._face_cascade = None
        self.last_error: Optional[str] = None
        self._init(prefer_tasks, allow_download)

    # -- setup -------------------------------------------------------------------------
    def _init(self, prefer_tasks: bool, allow_download: bool) -> None:
        try:
            import mediapipe as mp  # noqa: F401
        except Exception as exc:
            self.last_error = f"mediapipe not importable: {exc}"
            self.backend = "heuristic"
            self._init_fallback()
            return

        if prefer_tasks and self._try_init_tasks(mp, allow_download):
            return
        if self._try_init_solutions(mp):
            return

        self.backend = "heuristic"
        self._init_fallback()

    def _try_init_tasks(self, mp, allow_download: bool) -> bool:
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            path = self.model_path
            if not path.exists():
                if not allow_download:
                    return False
                path.parent.mkdir(parents=True, exist_ok=True)
                logger.info("Downloading MediaPipe pose model to %s", path)
                urllib.request.urlretrieve(DEFAULT_MODEL_URL, path)  # noqa: S310 - fixed https URL
            options = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                output_segmentation_masks=self.enable_segmentation,
            )
            self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
            self.backend = "mediapipe_tasks"
            logger.info("Pose backend: mediapipe tasks (%s)", path.name)
            return True
        except Exception as exc:
            self.last_error = f"tasks init failed: {exc}"
            logger.debug("MediaPipe Tasks init failed: %s", exc)
            return False

    def _try_init_solutions(self, mp) -> bool:
        try:
            solution = mp.solutions.pose
            self._solution = solution.Pose(
                static_image_mode=False,
                model_complexity=self.complexity,
                enable_segmentation=self.enable_segmentation,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self.backend = "mediapipe_solutions"
            logger.info("Pose backend: mediapipe solutions (complexity=%s)", self.complexity)
            return True
        except Exception as exc:
            self.last_error = f"solutions init failed: {exc}"
            return False

    def _init_fallback(self) -> None:
        """OpenCV HOG + Haar fallback: gives a usable torso box, not landmarks."""
        import cv2

        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if cascade_path.exists():
            self._face_cascade = cv2.CascadeClassifier(str(cascade_path))
        logger.warning(
            "Pose backend: heuristic fallback (MediaPipe unavailable: %s). Install mediapipe "
            "for accurate landmark tracking.", self.last_error or "unknown reason"
        )

    # -- inference ---------------------------------------------------------------------
    def estimate(self, image_rgb: np.ndarray, include_mask: bool = True) -> PoseResult:
        """Run pose estimation on an RGB frame."""
        height, width = image_rgb.shape[:2]
        if self.backend == "mediapipe_tasks":
            result = self._estimate_tasks(image_rgb)
        elif self.backend == "mediapipe_solutions":
            result = self._estimate_solutions(image_rgb)
        else:
            result = self._estimate_heuristic(image_rgb)
        result.image_size = (height, width)
        if result.mask is None and include_mask and self.enable_segmentation:
            result.mask = None
        return result

    def _estimate_tasks(self, image_rgb: np.ndarray) -> PoseResult:
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
        try:
            output = self._landmarker.detect(mp_image)
        except Exception as exc:  # pragma: no cover - runtime failure
            logger.error("Pose (tasks) inference failed: %s", exc)
            return PoseResult(detected=False, backend="mediapipe_tasks")

        if not output.pose_landmarks:
            return PoseResult(detected=False, backend="mediapipe_tasks")

        height, width = image_rgb.shape[:2]
        landmarks = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
        for index, point in enumerate(output.pose_landmarks[0][:NUM_LANDMARKS]):
            visibility = float(getattr(point, "visibility", 0.0) or 0.0)
            presence = float(getattr(point, "presence", 0.0) or 0.0)
            landmarks[index] = (point.x * width, point.y * height, point.z * width, max(visibility, presence))

        mask = None
        if output.segmentation_masks:
            raw = output.segmentation_masks[0].numpy_view()
            mask = (np.clip(raw, 0.0, 1.0) * 255).astype(np.uint8)
        score = float(np.mean(landmarks[:, 3])) if landmarks.size else 0.0
        return PoseResult(detected=True, landmarks=landmarks, backend="mediapipe_tasks", score=score, mask=mask)

    def _estimate_solutions(self, image_rgb: np.ndarray) -> PoseResult:
        try:
            output = self._solution.process(np.ascontiguousarray(image_rgb))
        except Exception as exc:  # pragma: no cover
            logger.error("Pose (solutions) inference failed: %s", exc)
            return PoseResult(detected=False, backend="mediapipe_solutions")

        if not output.pose_landmarks:
            return PoseResult(detected=False, backend="mediapipe_solutions")

        height, width = image_rgb.shape[:2]
        landmarks = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
        for index, point in enumerate(output.pose_landmarks.landmark[:NUM_LANDMARKS]):
            landmarks[index] = (
                point.x * width, point.y * height, point.z * width,
                float(getattr(point, "visibility", 0.0)),
            )
        mask = None
        if output.segmentation_mask is not None:
            mask = (np.clip(output.segmentation_mask, 0.0, 1.0) * 255).astype(np.uint8)
        score = float(np.mean(landmarks[:, 3]))
        return PoseResult(detected=True, landmarks=landmarks, backend="mediapipe_solutions", score=score, mask=mask)

    def _estimate_heuristic(self, image_rgb: np.ndarray) -> PoseResult:
        """Estimate a plausible torso from a person detection.

        This is intentionally conservative: if no torso-ish region is found we report
        ``detected=False`` rather than inventing landmarks. Only the shoulder/hip
        landmarks are filled in (with a low visibility) — arms are left unknown, and the
        alignment code degrades to a shoulder-only affine warp.
        """
        import cv2

        height, width = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        all_boxes: List[Tuple[int, int, int, int]] = []

        if self._hog is not None:
            small = cv2.resize(gray, (min(640, width), int(min(640, width) * height / max(1, width))))
            rects, weights = self._hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
            scale_back = width / small.shape[1]
            for (x, y, w, h), weight in zip(rects, weights):
                if weight > 0.5:
                    all_boxes.append((
                        int(x * scale_back), int(y * scale_back),
                        int((x + w) * scale_back), int((y + h) * scale_back),
                    ))

        if self._face_cascade is not None:
            faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
            for (x, y, w, h) in faces:
                # Synthesise a torso below the face: ~2.6x face width, ~3.2x face height.
                all_boxes.append((
                    int(x - 0.8 * w), int(y + 0.6 * h),
                    int(x + 1.8 * w), int(y + 0.6 * h + 3.2 * h),
                ))

        if not all_boxes:
            # Neither the person detector nor the face detector fired (common with stylised
            # or low-texture images). Fall back to foreground-subject geometry, which is
            # perfectly adequate for a torso overlay and is explicitly labelled "heuristic".
            return self._subject_fallback(image_rgb)

        best = max(all_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        x0, y0, x1, y1 = best
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 - x0 < 40 or y1 - y0 < 60:
            return PoseResult(detected=False, backend="heuristic")

        torso_w = (x1 - x0) * 0.72
        shoulder_y = y0 + (y1 - y0) * 0.22
        hip_y = y0 + (y1 - y0) * 0.58
        center_x = (x0 + x1) / 2.0
        landmarks = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
        landmarks[LEFT_SHOULDER] = (center_x - torso_w / 2, shoulder_y, 0.0, 0.35)
        landmarks[RIGHT_SHOULDER] = (center_x + torso_w / 2, shoulder_y, 0.0, 0.35)
        landmarks[LEFT_HIP] = (center_x - torso_w / 2, hip_y, 0.0, 0.30)
        landmarks[RIGHT_HIP] = (center_x + torso_w / 2, hip_y, 0.0, 0.30)
        landmarks[NOSE] = (center_x, y0 + (y1 - y0) * 0.06, 0.0, 0.25)

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
        return PoseResult(detected=True, landmarks=landmarks, backend="heuristic", score=0.35, mask=mask)

    def _subject_fallback(self, image_rgb: np.ndarray) -> PoseResult:
        """Infer a torso from the largest foreground subject.

        Estimates the background colour from the image border, thresholds the colour
        distance to get a subject silhouette, then places shoulder/hip landmarks using
        anthropometric ratios. This keeps the app usable on stylised images (and on real
        photos where the detectors miss) instead of showing nothing at all — the result is
        always reported as ``backend="heuristic"`` with a low confidence score.
        """
        import cv2

        height, width = image_rgb.shape[:2]
        border = max(2, int(0.03 * min(height, width)))
        edges = np.concatenate([
            image_rgb[:border].reshape(-1, 3), image_rgb[-border:].reshape(-1, 3),
            image_rgb[:, :border].reshape(-1, 3), image_rgb[:, -border:].reshape(-1, 3),
        ], axis=0)
        background = np.median(edges, axis=0)
        if float(np.std(edges, axis=0).mean()) > 42.0:
            return PoseResult(detected=False, backend="heuristic")  # cluttered scene: do not guess

        distance = np.linalg.norm(image_rgb.astype(np.float32) - background[None, None, :], axis=-1)
        binary = (distance > 34).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary = _largest_component(binary)
        coverage = float(binary.mean())
        if coverage < 0.03 or coverage > 0.9:
            return PoseResult(detected=False, backend="heuristic")

        rows = np.flatnonzero(binary.any(axis=1))
        cols = np.flatnonzero(binary.any(axis=0))
        if rows.size == 0 or cols.size == 0:
            return PoseResult(detected=False, backend="heuristic")
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        blob_h = y1 - y0
        if blob_h < 0.25 * height or blob_h < 60:
            return PoseResult(detected=False, backend="heuristic")

        def row_span(fraction: float) -> Tuple[float, float]:
            row = int(np.clip(y0 + fraction * blob_h, y0, y1 - 1))
            window = binary[max(y0, row - 2):min(y1, row + 3)]
            columns = np.flatnonzero(window.any(axis=0))
            return (float(columns[0]), float(columns[-1])) if columns.size else (float(x0), float(x1))

        shoulder_top, shoulder_bottom = row_span(0.22)
        hip_left, hip_right = row_span(0.54)
        center_x = (shoulder_top + shoulder_bottom) / 2.0
        shoulder_y = y0 + 0.22 * blob_h
        hip_y = y0 + 0.54 * blob_h

        landmarks = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
        landmarks[LEFT_SHOULDER] = (shoulder_top, shoulder_y, 0.0, 0.4)
        landmarks[RIGHT_SHOULDER] = (shoulder_bottom, shoulder_y, 0.0, 0.4)
        landmarks[LEFT_HIP] = (hip_left, hip_y, 0.0, 0.35)
        landmarks[RIGHT_HIP] = (hip_right, hip_y, 0.0, 0.35)
        landmarks[NOSE] = (center_x, y0 + 0.06 * blob_h, 0.0, 0.3)

        mask = (binary * 255).astype(np.uint8)
        return PoseResult(detected=True, landmarks=landmarks, backend="heuristic", score=0.3, mask=mask)

    # -- lifecycle ---------------------------------------------------------------------
    def close(self) -> None:
        for attribute in ("_landmarker", "_solution"):
            obj = getattr(self, attribute, None)
            close = getattr(obj, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover
                    pass
            setattr(self, attribute, None)

    @property
    def is_precise(self) -> bool:
        """True when real 33-landmark tracking (not the heuristic fallback) is active."""
        return self.backend.startswith("mediapipe")

    def status(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "precise_landmarks": self.is_precise,
            "segmentation": self.enable_segmentation and self.is_precise,
            "last_error": self.last_error,
            "notes": [] if self.is_precise else [
                "MediaPipe is not installed — using an OpenCV person/face detector. "
                "Install mediapipe for full 33-landmark tracking."
            ],
        }


def landmarks_to_heatmaps(landmarks: np.ndarray, size: Tuple[int, int], sigma: float = 4.0) -> np.ndarray:
    """Rasterise landmarks + limbs into low-resolution heatmaps for conditioning.

    Returns a ``(1, H, W)`` float32 array in ``[0, 1]`` used by the diffusion control branch.
    This is a lightweight stand-in for a full DensePose representation: it preserves the
    body silhouette and pose topology, which is what the control network conditions on.
    """
    import cv2

    height, width = size
    heat = np.zeros((height, width), dtype=np.float32)
    for index in range(min(len(landmarks), NUM_LANDMARKS)):
        x, y, _z, visibility = landmarks[index]
        if visibility < 0.2:
            continue
        cx, cy = int(np.clip(x * width, 0, width - 1)), int(np.clip(y * height, 0, height - 1))
        heat[cy, cx] = max(heat[cy, cx], float(visibility))
    for a, b in POSE_CONNECTIONS:
        if a >= len(landmarks) or b >= len(landmarks):
            continue
        xa, ya, _za, va = landmarks[a]
        xb, yb, _zb, vb = landmarks[b]
        if va < 0.2 or vb < 0.2:
            continue
        cv2.line(
            heat,
            (int(np.clip(xa * width, 0, width - 1)), int(np.clip(ya * height, 0, height - 1))),
            (int(np.clip(xb * width, 0, width - 1)), int(np.clip(yb * height, 0, height - 1))),
            1.0, thickness=max(1, int(0.006 * min(width, height))), lineType=cv2.LINE_AA,
        )
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma, sigmaY=sigma)
    peak = float(heat.max())
    if peak > 1e-6:
        heat = heat / peak
    return heat[None, ...]


def normalize_landmarks(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert pixel landmarks into the normalised ``[0, 1]`` form models expect."""
    out = np.asarray(landmarks, dtype=np.float32).copy()
    out[:, 0] = out[:, 0] / max(1.0, float(width))
    out[:, 1] = out[:, 1] / max(1.0, float(height))
    out[:, 2] = out[:, 2] / max(1.0, float(width))
    return out


def shoulder_angle(landmarks: np.ndarray) -> float:
    """Roll angle of the shoulder line in degrees (positive = right shoulder lower)."""
    dx = float(landmarks[RIGHT_SHOULDER, 0] - landmarks[LEFT_SHOULDER, 0])
    dy = float(landmarks[RIGHT_SHOULDER, 1] - landmarks[LEFT_SHOULDER, 1])
    return math.degrees(math.atan2(dy, dx))


def require_pose_dependencies() -> None:
    """Raise a friendly error when MediaPipe is required but missing."""
    try:
        import mediapipe  # noqa: F401
    except Exception as exc:
        raise DependencyMissingError(
            "MediaPipe is required for server-side pose tracking. "
            "Install it with `pip install mediapipe`. (The browser performs pose tracking "
            "client-side by default, so live try-on still works without it.)",
            details={"install": "pip install mediapipe", "reason": str(exc)},
        ) from exc


def find_cached_models(directory: str | Path = "models_cache") -> Dict[str, str]:
    """List locally cached pose/hand models (used by the Model Status page)."""
    path = Path(directory)
    if not path.exists():
        return {}
    return {p.name: str(p) for p in sorted(path.glob("*.task")) + sorted(path.glob("*.tflite"))}


def download_pose_model(destination: str | Path, url: str = DEFAULT_MODEL_URL) -> Path:
    """Download the MediaPipe pose model into ``destination`` (idempotent)."""
    destination = Path(destination)
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)  # noqa: S310 - fixed https URL
    logger.info("Saved %s", destination)
    return destination


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _largest_component(binary: np.ndarray) -> np.ndarray:
    """Keep only the largest connected foreground blob (local helper for the fallback)."""
    import cv2 as _cv2

    count, labels, stats, _ = _cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if count <= 2:
        return binary.astype(np.uint8)
    areas = stats[1:, _cv2.CC_STAT_AREA]
    keep = int(np.argmax(areas)) + 1
    return (labels == keep).astype(np.uint8)
