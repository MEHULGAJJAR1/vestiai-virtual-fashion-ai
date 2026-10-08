"""Try-on orchestration: single-image synthesis, live sessions, result archiving.

The service owns:

* the active :class:`VTONAdapter` (built through the registry, swappable at runtime),
* the :class:`LiveTryOnPipeline` sessions used by the live endpoints / WebSocket,
* saving of results (composite, comparison grid, masks, metadata) into ``results/``.

Adapters are swapped via :meth:`TryOnService.set_backend`, which means the frontend never
needs to know whether the answer came from the diffusion model or the geometric fallback —
it just displays ``backend`` in the status bar.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.ai.adapter import TryOnRequest, TryOnResult, VTONAdapter
from backend.ai.registry import available_adapters, build_adapter
from backend.cv.live_pipeline import GarmentSpec, LiveFrameResult, LiveTryOnPipeline
from backend.cv.pose import PoseEstimator
from backend.cv.smoothing import SmootherConfig
from backend.models.checkpointing import CheckpointManager
from backend.services.garment_service import ClosetStore, GarmentRecord
from backend.utils.errors import GarmentNotFoundError, InferenceError, ValidationError
from backend.utils.image_utils import decode_base64, load_image, save_image, stack_grid, to_data_uri, to_rgb
from backend.utils.logging_utils import get_logger
from backend.utils.timing import Timer

logger = get_logger(__name__)


@dataclass
class SavedResult:
    """A stored try-on result."""

    result_id: str
    directory: str
    created_at: float
    backend: str
    garment_key: Optional[str]
    latency_ms: float
    metadata: Dict[str, Any]

    def to_dict(self, include_images: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "result_id": self.result_id,
            "created_at": self.created_at,
            "created_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created_at)),
            "backend": self.backend,
            "garment_key": self.garment_key,
            "latency_ms": round(self.latency_ms, 1),
            "metadata": self.metadata,
            "files": {
                "output": f"/api/results/{self.result_id}/file/output",
                "comparison": f"/api/results/{self.result_id}/file/comparison",
                "person": f"/api/results/{self.result_id}/file/person",
                "garment": f"/api/results/{self.result_id}/file/garment",
            },
        }
        if include_images:
            payload["urls"] = payload["files"]
        return payload


    def to_storage_dict(self) -> Dict[str, Any]:
        """Exact constructor payload — what ``results/results.json`` persists."""
        return {
            "result_id": self.result_id,
            "directory": self.directory,
            "created_at": self.created_at,
            "backend": self.backend,
            "garment_key": self.garment_key,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_storage_dict(cls, payload: Dict[str, Any]) -> "SavedResult":
        """Rebuild a result from disk, tolerating older index files."""
        data = dict(payload)
        files = data.pop("files", None) or {}
        directory = data.get("directory")
        if not directory:
            # Older indexes only stored URL templates; recover the folder from results_dir.
            data.pop("urls", None)
            directory = data.pop("directory", None)
        return cls(
            result_id=str(data["result_id"]),
            directory=str(directory) if directory else "",
            created_at=float(data.get("created_at", 0.0)),
            backend=str(data.get("backend", "unknown")),
            garment_key=data.get("garment_key"),
            latency_ms=float(data.get("latency_ms", 0.0)),
            metadata=data.get("metadata", {}) or {},
        )


class TryOnService:
    """High-level try-on API used by the FastAPI routes."""

    def __init__(
        self,
        settings,
        closet: ClosetStore,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ) -> None:
        self.settings = settings
        self.closet = closet
        self.checkpoints = checkpoint_manager or CheckpointManager(settings.checkpoints_dir)
        self.results_dir = Path(settings.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._estimator: Optional[PoseEstimator] = None
        self._adapter: VTONAdapter = build_adapter("auto", settings=settings, checkpoint_manager=self.checkpoints)
        self._sessions: Dict[str, LiveTryOnPipeline] = {}
        self._results: Dict[str, SavedResult] = {}
        self._load_results_index()

    # ---------------------------------------------------------------------------------
    # adapter management
    # ---------------------------------------------------------------------------------
    @property
    def adapter(self) -> VTONAdapter:
        return self._adapter

    def set_backend(self, name: str, reload_model: bool = True) -> Dict[str, Any]:
        """Switch the active adapter (``auto`` | ``diffusion`` | ``lightweight`` | ``disabled``)."""
        with self._lock:
            previous = self._adapter
            if reload_model and hasattr(previous, "unload"):
                previous.unload()
            self._adapter = build_adapter("auto" if name == "auto" else name, settings=self.settings, checkpoint_manager=self.checkpoints, estimator=self.estimator)
            # Refresh live sessions so they use the new adapter.
            for session in self._sessions.values():
                session.ai_adapter = self._adapter
            logger.info("Try-on backend switched to %s", self._adapter.name)
            return self.adapter_status()

    def reload_model(self) -> Dict[str, Any]:
        """Re-scan ``checkpoints/`` and rebuild the adapter (Model Status page action)."""
        with self._lock:
            if hasattr(self._adapter, "reload"):
                self._adapter.reload()  # type: ignore[attr-defined]
            else:
                self._adapter = build_adapter("auto", settings=self.settings, checkpoint_manager=self.checkpoints, estimator=self.estimator)
            return self.adapter_status()

    def adapter_status(self) -> Dict[str, Any]:
        payload = self._adapter.status()
        payload["available_adapters"] = available_adapters()
        payload["checkpoints"] = self.checkpoints.status()
        return payload

    @property
    def estimator(self) -> PoseEstimator:
        """Shared pose estimator (created lazily, reused by every session)."""
        if self._estimator is None:
            self._estimator = PoseEstimator(
                complexity=self.settings.pose_model_complexity,
                min_detection_confidence=self.settings.pose_min_detection_confidence,
                min_tracking_confidence=self.settings.pose_min_tracking_confidence,
                enable_segmentation=self.settings.segmentation_enabled,
            )
        return self._estimator

    # ---------------------------------------------------------------------------------
    # single-image try-on
    # ---------------------------------------------------------------------------------
    def run(
        self,
        person_image: np.ndarray,
        garment: GarmentRecord,
        *,
        prompt: Optional[str] = None,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        resolution: Optional[int] = None,
        seed: Optional[int] = None,
        save: bool = True,
        landmarks: Optional[np.ndarray] = None,
        backend: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Person + garment -> try-on image (and a saved comparison grid)."""
        if person_image is None or person_image.size == 0:
            raise ValidationError("No person image supplied.")

        adapter = self._adapter
        if backend and backend != adapter.name:
            self.set_backend(backend)
            adapter = self._adapter

        person = to_rgb(person_image)
        garment_rgb = load_image(garment.image_path, mode="RGB")
        garment_alpha = load_image(garment.mask_path, mode="L")
        if garment_rgb.shape[:2] != garment_alpha.shape[:2]:
            from backend.utils.image_utils import ensure_size

            garment_alpha = ensure_size(garment_alpha, (garment_rgb.shape[1], garment_rgb.shape[0]))

        request = TryOnRequest(
            person_image=person,
            garment_image=garment_rgb,
            garment_mask=garment_alpha,
            landmarks=landmarks if landmarks is not None else self._pose_for(person),
            category=garment.category,
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            resolution=resolution,
            seed=seed,
        )
        with Timer() as timer:
            result: TryOnResult = adapter.try_on(request)

        if not result.ok or result.image is None:
            raise InferenceError(
                result.reason or "Try-on could not be generated.",
                details={"backend": adapter.name, "warnings": result.warnings},
            )

        payload = result.to_dict(include_images=False)
        payload["latency_ms"] = round(timer.ms, 1)
        if save:
            saved = self.save_result(
                person=person, garment=result, garment_record=garment, latency_ms=timer.ms,
            )
            payload["result"] = saved.to_dict(include_images=False)
            payload["images"] = {
                "output": to_data_uri(result.image, "PNG"),
                "comparison": to_data_uri(load_image(Path(saved.directory) / "comparison.jpg"), "JPEG"),
            }
        else:
            payload["images"] = {"output": to_data_uri(result.image, "PNG")}
        return payload

    def _pose_for(self, person: np.ndarray):
        try:
            pose = self.estimator.estimate(person, include_mask=False)
            return pose.landmarks if pose.detected else None
        except Exception as exc:  # pragma: no cover
            logger.debug("Pose estimation skipped: %s", exc)
            return None

    def run_from_base64(self, person_base64: str, garment_key: str, **kwargs: Any) -> Dict[str, Any]:
        garment = self.closet.get(garment_key)
        return self.run(decode_base64(person_base64), garment, **kwargs)

    # ---------------------------------------------------------------------------------
    # results
    # ---------------------------------------------------------------------------------
    def save_result(
        self,
        person: np.ndarray,
        garment: TryOnResult,
        garment_record: Optional[GarmentRecord],
        latency_ms: float,
    ) -> SavedResult:
        """Persist a try-on result plus a person|garment|output comparison grid."""
        result_id = f"tryon-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        directory = self.results_dir / result_id
        directory.mkdir(parents=True, exist_ok=True)
        assert garment.image is not None  # checked by caller

        save_image(person, directory / "person.jpg", quality=92)
        save_image(garment.image, directory / "output.png")
        if garment_record is not None:
            try:
                save_image(load_image(garment_record.preview_path), directory / "garment.jpg", quality=90)
            except Exception as exc:  # pragma: no cover
                logger.debug("Could not copy garment preview: %s", exc)
        if garment.composited_mask is not None:
            save_image(garment.composited_mask, directory / "mask.png")

        panels = [person]
        if garment_record is not None and (directory / "garment.jpg").exists():
            panels.append(load_image(directory / "garment.jpg"))
        panels.append(garment.image)
        comparison = stack_grid(panels, cols=len(panels))
        save_image(comparison, directory / "comparison.jpg", quality=90)

        metadata = {
            "backend": garment.backend,
            "warnings": garment.warnings,
            "debug": garment.debug,
            "latency_ms": round(latency_ms, 1),
            "garment_key": garment_record.key if garment_record else None,
            "garment_category": garment_record.category if garment_record else None,
            "adapter": self._adapter.name,
            "device": getattr(self.settings, "device", "auto"),
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

        record = SavedResult(
            result_id=result_id, directory=str(directory), created_at=time.time(),
            backend=garment.backend, garment_key=garment_record.key if garment_record else None,
            latency_ms=latency_ms, metadata=metadata,
        )
        self._results[result_id] = record
        self._save_results_index()
        return record

    def list_results(self, limit: int = 50) -> List[Dict[str, Any]]:
        records = sorted(self._results.values(), key=lambda r: r.created_at, reverse=True)[:limit]
        return [record.to_dict() for record in records]

    def get_result(self, result_id: str) -> SavedResult:
        record = self._results.get(result_id)
        if record is None:
            raise GarmentNotFoundError(f"No saved result '{result_id}'.", details={"result_id": result_id})
        return record

    def delete_result(self, result_id: str) -> bool:
        record = self._results.pop(result_id, None)
        if record is None:
            return False
        import shutil

        shutil.rmtree(record.directory, ignore_errors=True)
        self._save_results_index()
        return True

    def _save_results_index(self) -> None:
        index = self.results_dir / "results.json"
        index.write_text(json.dumps({"results": [r.to_storage_dict() for r in self._results.values()]}, indent=2), encoding="utf-8")

    def _load_results_index(self) -> None:
        index = self.results_dir / "results.json"
        if not index.exists():
            return
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
        except Exception:
            return
        for item in payload.get("results", []):
            try:
                record = SavedResult.from_storage_dict(item)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping unreadable result index entry: %s", exc)
                continue
            if not record.directory:
                record.directory = str(self.results_dir / record.result_id)
            if not Path(record.directory).exists():
                logger.warning("Skipping result %s — folder %s is gone.", record.result_id, record.directory)
                continue
            self._results[record.result_id] = record

    # ---------------------------------------------------------------------------------
    # live sessions
    # ---------------------------------------------------------------------------------
    def create_session(
        self,
        session_id: Optional[str] = None,
        garment_key: Optional[str] = None,
        enable_ai: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Create (or replace) a live tracking session."""
        session_id = session_id or f"live-{uuid.uuid4().hex[:8]}"
        config = SmootherConfig(alpha=self.settings.smoothing_alpha)
        pipeline = LiveTryOnPipeline(
            estimator=self.estimator,
            smoothing=config,
            ai_adapter=self._adapter,
            ai_interval_ms=self.settings.live_ai_interval_ms,
            enable_ai=bool(self.settings.live_ai_enabled if hasattr(self.settings, "live_ai_enabled") else False) if enable_ai is None else enable_ai,
            category_overrides=getattr(self.settings, "extra", {}).get("category_overrides"),
        )
        with self._lock:
            self._sessions[session_id] = pipeline
        return {
            "session_id": session_id,
            "garment_key": garment_key,
            "ai_enabled": pipeline.enable_ai,
            "status": pipeline.status(),
        }

    def get_session(self, session_id: str) -> LiveTryOnPipeline:
        pipeline = self._sessions.get(session_id)
        if pipeline is None:
            raise ValidationError(f"Unknown live session '{session_id}'. Call /api/live/session first.")
        return pipeline

    def process_frame(self, session_id: str, frame_base64: str, garment_key: Optional[str], **kwargs: Any) -> Dict[str, Any]:
        pipeline = self.get_session(session_id)
        garment = self._load_spec(garment_key)
        result: LiveFrameResult = pipeline.process_base64(frame_base64, garment, **kwargs)
        return {
            "ok": result.ok,
            "person_detected": result.person_detected,
            "garment_visible": result.garment_visible,
            "overlay": result.overlay_png_base64,
            "status": result.status,
            "error": result.error,
        }

    def close_session(self, session_id: str) -> bool:
        with self._lock:
            pipeline = self._sessions.pop(session_id, None)
        if pipeline is None:
            return False
        # Pose estimator is shared, so only reset the session state.
        pipeline.reset()
        return True

    def live_config(self, **updates: Any) -> Dict[str, Any]:
        """Update live inference parameters (AI interval, enable/disable, smoothing)."""
        applied: Dict[str, Any] = {}
        if "ai_interval_ms" in updates and updates["ai_interval_ms"] is not None:
            value = max(50, int(updates["ai_interval_ms"]))
            self.settings.live_ai_interval_ms = value
            applied["ai_interval_ms"] = value
        if "smoothing_alpha" in updates and updates["smoothing_alpha"] is not None:
            value = float(np.clip(float(updates["smoothing_alpha"]), 0.0, 1.0))
            self.settings.smoothing_alpha = value
            applied["smoothing_alpha"] = value
        if "ai_enabled" in updates and updates["ai_enabled"] is not None:
            applied["ai_enabled"] = bool(updates["ai_enabled"])
        if "resolution" in updates and updates["resolution"] is not None:
            value = max(256, min(1024, int(updates["resolution"])))
            self.settings.live_ai_max_side = value
            applied["resolution"] = value
        if "num_inference_steps" in updates and updates["num_inference_steps"] is not None:
            value = max(4, min(100, int(updates["num_inference_steps"])))
            self.settings.num_inference_steps = value
            applied["num_inference_steps"] = value

        for pipeline in self._sessions.values():
            if "ai_interval_ms" in applied:
                pipeline.ai_interval_ms = applied["ai_interval_ms"]
            if "ai_enabled" in applied:
                pipeline.enable_ai = applied["ai_enabled"]
            if "smoothing_alpha" in applied:
                pipeline.stabilizer.config.alpha = applied["smoothing_alpha"]
        return {"ok": True, "applied": applied, "active_sessions": len(self._sessions)}

    def _load_spec(self, garment_key: Optional[str]) -> Optional[GarmentSpec]:
        if not garment_key:
            return None
        record = self.closet.get(garment_key)
        try:
            rgb = load_image(record.image_path, mode="RGB")
            alpha = load_image(record.mask_path, mode="L")
        except Exception as exc:
            raise GarmentNotFoundError(f"Garment files for '{garment_key}' could not be read: {exc}") from exc
        return GarmentSpec(
            key=record.key, label=record.label, category=record.category, rgb=rgb, alpha=alpha,
            preview_url=record.preview_url, mask_url=record.mask_url, metadata={"palette": record.palette},
        )

    def sessions_status(self) -> List[Dict[str, Any]]:
        return [{"session_id": key, **pipeline.status()} for key, pipeline in self._sessions.items()]

    def shutdown(self) -> None:
        with self._lock:
            self._sessions.clear()
        if self._estimator is not None:
            self._estimator.close()
            self._estimator = None
        if hasattr(self._adapter, "unload"):
            self._adapter.unload()
