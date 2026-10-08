"""Lightweight (geometric) try-on adapter — always available, CPU-only, ~10 ms/image.

This is the *fallback* that makes VestiAI genuinely usable before a diffusion checkpoint
exists, and it is also the engine behind live tracking. It performs:

1. pose anchor extraction (MediaPipe landmarks, or the OpenCV heuristic),
2. garment -> body transform estimation (perspective/affine),
3. bilinear warp + occlusion-aware alpha composite,
4. optional Poisson/seam-light blending for a cleaner join.

It is deliberately honest: results look like a well-tracked AR overlay, not a photoshop-grade
synthesis, and the adapter reports that in ``warnings``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np

from backend.ai.adapter import TryOnAdapter, TryOnRequest, TryOnResult
from backend.cv import garment_align, occlusion, pose as pose_mod
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LightweightTryOnAdapter(TryOnAdapter):
    """Pose-driven affine/perspective garment warp."""

    name = "lightweight"
    display_name = "Real-time geometric warp"

    def __init__(
        self,
        estimator: Optional[pose_mod.PoseEstimator] = None,
        category_overrides: Optional[Dict[str, Dict[str, float]]] = None,
        feather: float = 1.6,
    ) -> None:
        super().__init__(feather=feather)
        self.estimator = estimator
        self.category_overrides = category_overrides or {}
        self.feather = feather

    # -- lifecycle ---------------------------------------------------------------------
    def is_ready(self) -> bool:
        return True

    def status(self) -> Dict[str, Any]:
        payload = self.base_status()
        payload.update({
            "detail": "Classical CV pipeline: pose landmarks -> affine/perspective warp -> occlusion-aware composite.",
            "quality": "fast / non-photorealistic",
            "device": "cpu",
            "requires_checkpoint": False,
        })
        if self.estimator is not None:
            payload["pose"] = self.estimator.status()
        return payload

    def warmup(self) -> bool:
        return True

    # -- inference ---------------------------------------------------------------------
    def try_on(self, request: TryOnRequest) -> TryOnResult:
        return self._timed(self._try_on_impl, request)

    def _try_on_impl(self, request: TryOnRequest) -> TryOnResult:
        person = np.asarray(request.person_image)[..., :3]
        garment_rgb = np.asarray(request.garment_image)[..., :3]
        garment_alpha = request.garment_mask
        if garment_alpha is None:
            # Treat a fully opaque garment canvas as its own mask when no alpha is supplied.
            garment_alpha = np.full(garment_rgb.shape[:2], 255, np.uint8)

        landmarks = request.landmarks
        if landmarks is None:
            if self.estimator is None:
                self.estimator = pose_mod.PoseEstimator()
            pose_result = self.estimator.estimate(person, include_mask=False)
            if not pose_result.detected:
                return TryOnResult(
                    ok=False, backend=self.name,
                    reason="No person detected in the frame. Step into view and try again.",
                )
            landmarks = pose_result.landmarks

        solved = garment_align.compute_fit_transform(
            garment_alpha, landmarks, person.shape[:2], request.category,
            category_overrides=self.category_overrides,
        )
        if solved is None:
            return TryOnResult(
                ok=False, backend=self.name,
                reason="Could not locate the shoulders/hips well enough to place the garment. "
                       "Face the camera with your upper body visible.",
            )
        matrix, mode, _pose_anchors, _garment_anchors = solved

        warped = garment_align.warp_garment(garment_rgb, garment_alpha, matrix, (person.shape[1], person.shape[0]))
        alpha = occlusion.apply_occlusion(warped.alpha, landmarks, person.shape[:2])
        if garment_align.coverage_ratio(alpha) < 0.002:
            return TryOnResult(
                ok=False, backend=self.name,
                reason="The garment would be invisible at this framing. Move back from the camera.",
            )

        composed = garment_align.blend_rgba_over_frame(person, np.dstack([warped.rgba[..., :3], alpha]), feather=self.feather)
        alpha_ratio = float((alpha > 32).mean())
        return TryOnResult(
            ok=True,
            image=composed,
            raw_generation=warped.rgba,
            composited_mask=alpha,
            backend=self.name,
            warnings=[
                "Geometric warp: garment texture and transparency are preserved, but lighting "
                "and fabric folds are not synthesised. Train a checkpoint for photorealistic output."
            ],
            debug={
                "transform_mode": mode,
                "coverage": round(alpha_ratio, 4),
                "matrix": np.asarray(matrix).round(3).tolist(),
                "occlusion": occlusion.occlusion_report(landmarks, person.shape[:2]),
            },
        )


def estimate_warp_preview(
    person_image: np.ndarray,
    garment_rgb: np.ndarray,
    garment_alpha: np.ndarray,
    landmarks: np.ndarray,
    category: str = "t-shirt",
) -> Optional[np.ndarray]:
    """Convenience helper used by the closet page to show how a garment would fit."""
    solved = garment_align.compute_fit_transform(garment_alpha, landmarks, person_image.shape[:2], category)
    if solved is None:
        return None
    matrix, _mode, _p, _g = solved
    warped = garment_align.warp_garment(garment_rgb, garment_alpha, matrix, (person_image.shape[1], person_image.shape[0]))
    alpha = occlusion.apply_occlusion(warped.alpha, landmarks, person_image.shape[:2])
    return garment_align.blend_rgba_over_frame(person_image, np.dstack([warped.rgba[..., :3], alpha]))


def soft_blend_seam(base: np.ndarray, overlay: np.ndarray, mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Optional seam smoothing using a guided filter (kept for callers that want it)."""
    alpha = (np.asarray(mask, np.float32) / 255.0)
    if radius > 0:
        alpha = cv2.GaussianBlur(alpha, (radius * 2 + 1, radius * 2 + 1), 0)
    alpha = alpha[..., None]
    return np.clip(np.asarray(base, np.float32) * (1 - alpha) + np.asarray(overlay, np.float32) * alpha, 0, 255).astype(np.uint8)
