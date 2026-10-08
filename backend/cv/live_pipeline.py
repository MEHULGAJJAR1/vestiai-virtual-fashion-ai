"""Server-side live try-on pipeline.

The browser normally runs pose tracking itself (MediaPipe Tasks Vision) for maximum
smoothness, but the same pipeline is implemented server-side so that:

* headless/browser-less smoke tests can exercise the full path,
* thin clients (a phone, a Raspberry Pi, a kiosk) can stream frames and receive an overlay,
* the WebSocket endpoint can refine frames with the heavy VTON model when the client asks.

Flow per frame::

    decode -> pose -> temporal smoothing -> garment transform -> warp
           -> occlusion handling -> alpha composite -> [optional] AI refinement

Design notes
------------
* All state is instance-level and guarded by a lock: the pipeline is safe to call from a
  FastAPI threadpool or the WebSocket loop.
* When the body is lost the garment is dropped **immediately** (no ghost overlay) and the
  last transform is kept for a short grace window so tracking resumes without a pop.
* The heavy diffusion model is invoked at a controlled frequency (``ai_interval_ms``) and
  its result is cached and reused, which is what keeps live mode usable without flicker.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from backend.cv import garment_align, occlusion, pose as pose_mod
from backend.cv.smoothing import ScalarSmoother, SmootherConfig, TemporalStabilizer
from backend.utils.image_utils import decode_base64, encode_base64, to_rgb
from backend.utils.logging_utils import get_logger
from backend.utils.timing import FPSMeter, Timer

logger = get_logger(__name__)


@dataclass
class GarmentSpec:
    """A garment ready for the lightweight pipeline (normalised square canvas)."""

    key: str
    label: str
    category: str
    rgb: np.ndarray
    alpha: np.ndarray
    mask_url: Optional[str] = None
    preview_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def rgba(self) -> np.ndarray:
        return np.dstack([self.rgb[..., :3], self.alpha])


@dataclass
class LiveFrameResult:
    """Result of processing a single frame."""

    ok: bool
    person_detected: bool
    garment_visible: bool
    overlay_png_base64: Optional[str]
    status: Dict[str, Any]
    error: Optional[str] = None


class LiveTryOnPipeline:
    """Stateful real-time try-on engine."""

    def __init__(
        self,
        estimator: Optional[pose_mod.PoseEstimator] = None,
        smoothing: Optional[SmootherConfig] = None,
        ai_adapter: Optional[Any] = None,
        ai_interval_ms: int = 700,
        enable_ai: bool = False,
        category_overrides: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self.estimator = estimator or pose_mod.PoseEstimator()
        self.stabilizer = TemporalStabilizer(smoothing or SmootherConfig())
        self.ai_adapter = ai_adapter
        self.ai_interval_ms = int(ai_interval_ms)
        self.enable_ai = bool(enable_ai)
        self.category_overrides = category_overrides or {}

        self.fps = FPSMeter(window=24)
        self.alpha_smoother = ScalarSmoother(alpha=0.4)
        self._lock = threading.Lock()
        self._last_ai_at = 0.0
        self._last_ai_frame: Optional[np.ndarray] = None
        self._last_ai_stats: Dict[str, Any] = {}
        self._lost_frames = 0
        self._frames = 0
        self._last_transform: Optional[np.ndarray] = None
        self._last_warp_mode = "none"
        self.last_pose: Optional[pose_mod.PoseResult] = None

    # ---------------------------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------------------------
    def reset(self) -> None:
        with self._lock:
            self.stabilizer.reset()
            self.alpha_smoother.reset()
            self.fps.reset()
            self._lost_frames = 0
            self._frames = 0
            self._last_transform = None
            self._last_ai_frame = None
            self.last_pose = None

    def process(
        self,
        frame_rgb: np.ndarray,
        garment: Optional[GarmentSpec],
        *,
        force_ai: bool = False,
        mirror: bool = False,
    ) -> LiveFrameResult:
        """Run one frame through the pipeline."""
        with self._lock:
            started = time.perf_counter()
            self._frames += 1
            fps = self.fps.tick()
            frame = to_rgb(frame_rgb)
            if mirror:
                frame = np.ascontiguousarray(frame[:, ::-1])

            pose_result = self._estimate(frame)
            smoothed = self.stabilizer.landmarks.update(pose_result.landmarks, pose_result.detected)

            person_visible = bool(pose_result.detected and smoothed is not None)
            garment_visible = False
            composed = frame.copy()
            transform_mode = "none"
            occlusion_info: Dict[str, Any] = {}
            warp_ms = 0.0

            if person_visible and garment is not None:
                with Timer() as warp_timer:
                    transform = None
                    try:
                        solved = garment_align.compute_fit_transform(
                            garment.alpha, smoothed, frame.shape[:2], garment.category,
                            category_overrides=self.category_overrides,
                        )
                        if solved is not None:
                            transform, mode, _pose_anchors, _garment_anchors = solved
                            transform_mode = mode
                            self._last_transform = transform
                            self._last_warp_mode = mode
                    except Exception as exc:
                        logger.debug("Transform estimation failed: %s", exc)
                        transform = None

                    if transform is not None:
                        warped = garment_align.warp_garment(
                            garment.rgb, garment.alpha, transform, (frame.shape[1], frame.shape[0])
                        )
                        alpha = occlusion.apply_occlusion(
                            warped.alpha, smoothed, frame.shape[:2], protect_head=True
                        )
                        if alpha.max() > 0 and garment_align.coverage_ratio(alpha) > 0.001:
                            weight = self.alpha_smoother.update(1.0)
                            _ = weight
                            composed = garment_align.blend_rgba_over_frame(
                                composed, np.dstack([warped.rgba[..., :3], alpha])
                            )
                            garment_visible = True
                            occlusion_info = occlusion.occlusion_report(smoothed, frame.shape[:2])
                        else:
                            garment_visible = False
                    elif self._last_transform is not None and self._lost_frames < self.stabilizer.config.max_hold_frames:
                        # Hold the previous transform briefly to avoid flicker on a dropped frame.
                        warped = garment_align.warp_garment(
                            garment.rgb, garment.alpha, self._last_transform, (frame.shape[1], frame.shape[0])
                        )
                        alpha = occlusion.apply_occlusion(warped.alpha, smoothed, frame.shape[:2])
                        composed = garment_align.blend_rgba_over_frame(
                            composed, np.dstack([warped.rgba[..., :3], alpha])
                        )
                        garment_visible = True
                        transform_mode = f"{self._last_warp_mode}(held)"
                warp_ms = warp_timer.ms

            if person_visible:
                self._lost_frames = 0
                if not garment_visible:
                    self._last_transform = None
            else:
                self._lost_frames += 1
                if self._lost_frames > self.stabilizer.config.max_hold_frames:
                    self._last_transform = None
                    self.alpha_smoother.reset()

            ai_info = self._maybe_refine(composed, garment, frame, smoothed, force_ai)
            if ai_info.get("applied") and ai_info.get("image") is not None:
                composed = ai_info["image"]

            latency = (time.perf_counter() - started) * 1000.0
            self.last_pose = pose_result
            status = {
                "frame_index": self._frames,
                "fps": fps,
                "frame_latency_ms": round(latency, 2),
                "warp_latency_ms": round(warp_ms, 2),
                "pose_backend": pose_result.backend,
                "pose_score": round(float(pose_result.score), 3),
                "pose_detected": person_visible,
                "segmentation": bool(pose_result.mask is not None),
                "tracking": self.stabilizer.landmarks.tracking,
                "garment_visible": garment_visible,
                "garment_key": garment.key if garment else None,
                "transform_mode": transform_mode,
                "occlusion": occlusion_info,
                "ai": {
                    "enabled": self.enable_ai,
                    "available": bool(self.ai_adapter and getattr(self.ai_adapter, "is_ready", lambda: False)()),
                    "applied": bool(ai_info.get("applied")),
                    "reason": ai_info.get("reason"),
                    "latency_ms": ai_info.get("latency_ms"),
                    "interval_ms": self.ai_interval_ms,
                },
            }
            overlay = encode_base64(composed, fmt="JPEG") if garment is not None else encode_base64(composed, fmt="JPEG")
            return LiveFrameResult(
                ok=True,
                person_detected=person_visible,
                garment_visible=garment_visible,
                overlay_png_base64=overlay,
                status=status,
            )

    def process_base64(self, frame_base64: str, garment: Optional[GarmentSpec], **kwargs: Any) -> LiveFrameResult:
        """Convenience wrapper for the HTTP/WebSocket APIs."""
        frame = decode_base64(frame_base64)
        return self.process(frame, garment, **kwargs)

    # ---------------------------------------------------------------------------------
    # internals
    # ---------------------------------------------------------------------------------
    def _estimate(self, frame: np.ndarray) -> pose_mod.PoseResult:
        try:
            return self.estimator.estimate(frame, include_mask=self.estimator.enable_segmentation)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Pose estimation crashed: %s", exc, exc_info=True)
            return pose_mod.PoseResult(detected=False, backend="error")

    def _maybe_refine(
        self,
        composed: np.ndarray,
        garment: Optional[GarmentSpec],
        raw_frame: np.ndarray,
        landmarks: Optional[np.ndarray],
        force: bool,
    ) -> Dict[str, Any]:
        """Run the heavy model on a throttled schedule, reusing the cached result."""
        if garment is None or landmarks is None:
            return {"applied": False, "reason": "no garment or body"}
        if not self.enable_ai or self.ai_adapter is None:
            return {"applied": False, "reason": "ai adapter disabled"}
        if not getattr(self.ai_adapter, "is_ready", lambda: False)():
            return {"applied": False, "reason": getattr(self.ai_adapter, "status", lambda: {})().get("detail", "model not ready")}

        now = time.time() * 1000.0
        if not force and (now - self._last_ai_at) < self.ai_interval_ms:
            if self._last_ai_frame is not None:
                return {"applied": True, "reason": "cached", "image": self._last_ai_frame, "latency_ms": 0.0}
            return {"applied": False, "reason": "throttled"}

        self._last_ai_at = now
        try:
            with Timer() as timer:
                refined = self.ai_adapter.try_on(
                    person_image=raw_frame,
                    garment_image=garment.rgb,
                    garment_mask=garment.alpha,
                    landmarks=landmarks,
                )
            image = refined.get("image") if isinstance(refined, dict) else None
            if image is None:
                return {"applied": False, "reason": refined.get("reason", "no output") if isinstance(refined, dict) else "no output"}
            self._last_ai_frame = image
            self._last_ai_stats = {"latency_ms": round(timer.ms, 1)}
            return {"applied": True, "reason": "fresh", "image": image, "latency_ms": round(timer.ms, 1)}
        except Exception as exc:
            logger.warning("Live AI refinement failed: %s", exc)
            return {"applied": False, "reason": f"error: {exc.__class__.__name__}"}

    def status(self) -> Dict[str, Any]:
        return {
            "frames_processed": self._frames,
            "fps": self.fps.fps,
            "pose": self.estimator.status(),
            "smoothing": self.stabilizer.status(),
            "ai": {
                "enabled": self.enable_ai,
                "interval_ms": self.ai_interval_ms,
                "adapter": self.ai_adapter.status() if self.ai_adapter is not None else None,
                "last_stats": self._last_ai_stats,
            },
            "lost_frames": self._lost_frames,
        }

    def close(self) -> None:
        try:
            self.estimator.close()
        except Exception:  # pragma: no cover
            pass


def decode_hidden_garment(payload: Dict[str, Any]) -> Optional[GarmentSpec]:
    """Build a :class:`GarmentSpec` from a closet record loaded from disk."""
    from backend.utils.image_utils import load_image

    try:
        rgb = load_image(payload["image_path"], mode="RGB")
        alpha = load_image(payload["mask_path"], mode="L")
    except Exception as exc:
        logger.warning("Could not load garment %s: %s", payload.get("key"), exc)
        return None
    return GarmentSpec(
        key=str(payload.get("key", "garment")),
        label=str(payload.get("label", "Garment")),
        category=str(payload.get("category", "unknown")),
        rgb=rgb,
        alpha=alpha,
        preview_url=payload.get("preview_url"),
        mask_url=payload.get("mask_url"),
        metadata={k: v for k, v in payload.items() if k not in {"rgb", "alpha"}},
    )
