"""Typed exceptions + FastAPI exception handlers.

Every user-facing failure path in VestiAI raises one of these so the API can return a
machine-readable ``error_code`` that the frontend turns into a friendly message.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class VestiAIError(Exception):
    """Base class for all application errors."""

    status_code: int = 500
    error_code: str = "internal_error"
    user_message: str = "Something went wrong inside VestiAI."

    def __init__(self, message: Optional[str] = None, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(message or self.user_message)
        self.message = message or self.user_message
        self.details = details or {}

    def to_payload(self) -> Dict[str, Any]:
        return {
            "ok": False,
            "error_code": self.error_code,
            "message": self.message,
            "user_message": self.user_message,
            "details": self.details,
        }


# -- 400 family ------------------------------------------------------------------------
class ValidationError(VestiAIError):
    status_code = 422
    error_code = "validation_error"
    user_message = "The request was not valid. Please check the submitted fields."


class InvalidImageError(VestiAIError):
    status_code = 400
    error_code = "invalid_image"
    user_message = "That file could not be read as an image. Please upload a JPG or PNG."


class UnsupportedGarmentError(VestiAIError):
    status_code = 400
    error_code = "unsupported_garment"
    user_message = (
        "This garment image is not supported. Upload a flat-lay or product photo of a single "
        "garment (no full-body model shots, no heavy occlusion, at least 160 px on the short side)."
    )


class LowResolutionError(UnsupportedGarmentError):
    error_code = "low_resolution"
    user_message = "The garment image is too small to work with. Please upload at least 160x160 px."


class GarmentNotFoundError(VestiAIError):
    status_code = 404
    error_code = "garment_not_found"
    user_message = "That garment is not in your closet any more."


class DatasetError(VestiAIError):
    status_code = 400
    error_code = "dataset_error"
    user_message = "The dataset could not be prepared. Check the dataset path and structure."


# -- 409 / 503 family ------------------------------------------------------------------
class ModelNotAvailableError(VestiAIError):
    status_code = 503
    error_code = "model_unavailable"
    user_message = (
        "The AI try-on model is not installed yet. Train or download a checkpoint, place it in "
        "checkpoints/best_model and restart. The real-time tracking mode still works."
    )


class CheckpointCorruptError(VestiAIError):
    status_code = 500
    error_code = "checkpoint_corrupt"
    user_message = "The model file looks corrupted. Re-export or retrain the checkpoint."


class GPUMemoryError(VestiAIError):
    status_code = 507
    error_code = "cuda_oom"
    user_message = (
        "The GPU ran out of memory. Lower the resolution, reduce batch size / steps, or switch to "
        "Fast mode (lightweight pipeline)."
    )


class GPUNotAvailableError(VestiAIError):
    status_code = 503
    error_code = "gpu_unavailable"
    user_message = (
        "No CUDA GPU was detected. Diffusion inference is disabled; enable CPU diffusion in "
        "Settings only if you accept ~1 minute per image."
    )


class TrainingError(VestiAIError):
    status_code = 409
    error_code = "training_error"
    user_message = "The training run could not start or was interrupted. See the training log."


class TrainingBusyError(VestiAIError):
    status_code = 409
    error_code = "training_busy"
    user_message = "A training run is already active. Stop it before starting another."


class NoCameraError(VestiAIError):
    status_code = 400
    error_code = "no_camera"
    user_message = "No camera was found. Check that a webcam is connected and not in use."


class InferenceError(VestiAIError):
    status_code = 500
    error_code = "inference_failed"
    user_message = "Try-on inference failed. The real-time mode is unaffected."


class DependencyMissingError(VestiAIError):
    status_code = 503
    error_code = "dependency_missing"
    user_message = "An optional component is not installed. Run scripts/setup.py to install extras."


HTTP_STATUS_BY_ERROR: Dict[str, int] = {
    cls.error_code: cls.status_code
    for cls in (
        ValidationError, InvalidImageError, UnsupportedGarmentError, GarmentNotFoundError,
        DatasetError, ModelNotAvailableError, CheckpointCorruptError, GPUMemoryError,
        GPUNotAvailableError, TrainingError, TrainingBusyError, NoCameraError,
        InferenceError, DependencyMissingError,
    )
}
