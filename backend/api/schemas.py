"""Pydantic request/response models for the VestiAI API.

Keeping them in one module makes the OpenAPI schema coherent and lets the frontend be typed
against ``/openapi.json`` if desired.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------------------
# generic
# --------------------------------------------------------------------------------------
class OkResponse(BaseModel):
    """Generic success envelope."""

    ok: bool = True
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error envelope emitted by the exception handlers."""

    ok: bool = False
    error_code: str
    message: str
    user_message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    ok: bool
    app: str
    version: str
    uptime_s: float
    timestamp: float


# --------------------------------------------------------------------------------------
# garments
# --------------------------------------------------------------------------------------
class GarmentUpdateRequest(BaseModel):
    """PATCH body for editing a closet entry."""

    label: Optional[str] = None
    category: Optional[str] = None
    favourite: Optional[bool] = None
    tags: Optional[List[str]] = None
    garment_type: Optional[str] = None


class GarmentListResponse(BaseModel):
    ok: bool = True
    count: int
    garments: List[Dict[str, Any]]
    stats: Dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------
# try-on
# --------------------------------------------------------------------------------------
class TryOnRequestModel(BaseModel):
    """Single-image try-on request.

    ``person_base64`` may be a raw base64 string or a ``data:image/...;base64,`` URI (that is
    what the browser sends after a canvas capture).
    """

    garment_key: str = Field(..., description="Closet key returned by /api/garments/upload")
    person_base64: Optional[str] = Field(None, description="Person photo as base64/PNG")
    landmarks: Optional[List[Dict[str, float]]] = Field(None, description="Optional BlazePose landmarks from the client")
    prompt: Optional[str] = None
    category: Optional[str] = None
    num_inference_steps: Optional[int] = Field(None, ge=1, le=100)
    guidance_scale: Optional[float] = Field(None, ge=0.0, le=20.0)
    resolution: Optional[int] = Field(None, ge=256, le=1024)
    seed: Optional[int] = None
    backend: Optional[str] = Field(None, description="auto | diffusion | lightweight | disabled")

    @field_validator("resolution")
    @classmethod
    def _multiple_of_8(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return value
        return max(256, min(1024, (int(value) // 8) * 8))


class LiveConfigRequest(BaseModel):
    """Live inference configuration."""

    ai_enabled: Optional[bool] = None
    ai_interval_ms: Optional[int] = Field(None, ge=50, le=5000)
    smoothing_alpha: Optional[float] = Field(None, ge=0.0, le=1.0)
    resolution: Optional[int] = Field(None, ge=256, le=1024)
    num_inference_steps: Optional[int] = Field(None, ge=4, le=100)


class SessionCreateRequest(BaseModel):
    session_id: Optional[str] = None
    garment_key: Optional[str] = None
    enable_ai: Optional[bool] = False


class FrameRequest(BaseModel):
    """One camera frame for server-side compositing."""

    session_id: str
    frame_base64: str
    garment_key: Optional[str] = None
    force_ai: bool = False
    mirror: bool = False


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------
class TrainingStartRequest(BaseModel):
    """Start a training run."""

    mode: str = Field("QUICK_DEMO", description="QUICK_DEMO | FINE_TUNE | FULL_TRAINING")
    dataset_root: Optional[str] = None
    dry_run: bool = False
    overrides: Dict[str, Any] = Field(default_factory=dict, description="Any TrainingConfig field, e.g. {num_epochs: 3}")

    @field_validator("mode")
    @classmethod
    def _known_mode(cls, value: str) -> str:
        allowed = {"QUICK_DEMO", "FINE_TUNE", "FULL_TRAINING"}
        upper = (value or "").upper()
        if upper not in allowed:
            raise ValueError(f"mode must be one of {sorted(allowed)}")
        return upper


class TrainingStopRequest(BaseModel):
    job_id: Optional[str] = None


# --------------------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------------------
class SettingsUpdateRequest(BaseModel):
    """Runtime-tunable settings (a curated subset of :class:`backend.config.Settings`)."""

    resolution: Optional[int] = Field(None, ge=256, le=1024)
    num_inference_steps: Optional[int] = Field(None, ge=1, le=100)
    guidance_scale: Optional[float] = Field(None, ge=0.0, le=20.0)
    live_ai_interval_ms: Optional[int] = Field(None, ge=50, le=5000)
    smoothing_alpha: Optional[float] = Field(None, ge=0.0, le=1.0)
    segmentation_enabled: Optional[bool] = None
    force_cpu: Optional[bool] = None
    enable_cpu_diffusion: Optional[bool] = None
    background_removal: Optional[str] = None
    tryon_backend: Optional[str] = None
    seed: Optional[int] = None
    tracker: Optional[str] = None
    batch_size: Optional[int] = Field(None, ge=1, le=64)
    learning_rate: Optional[float] = Field(None, gt=0, le=1.0)
    num_epochs: Optional[int] = Field(None, ge=1, le=1000)
    mixed_precision: Optional[str] = None
    gradient_accumulation: Optional[int] = Field(None, ge=1, le=64)
    category_overrides: Optional[Dict[str, Dict[str, float]]] = None


# --------------------------------------------------------------------------------------
# captureresults
# --------------------------------------------------------------------------------------
class CapturePhotoRequest(BaseModel):
    image_base64: str
    garment_key: Optional[str] = None
    backend: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecommendationRequest(BaseModel):
    style: str = "casual"
    limit: int = Field(6, ge=1, le=50)
    seed: Optional[int] = None


class OutfitRequest(BaseModel):
    style: str = "casual"
    seed: Optional[int] = None


class SampleClosetRequest(BaseModel):
    count: int = Field(10, ge=1, le=48)
    seed: int = 7


class DatasetGenerateRequest(BaseModel):
    train: int = Field(24, ge=4, le=500)
    val: int = Field(6, ge=1, le=200)
    test: int = Field(6, ge=1, le=200)
    size: int = Field(512, ge=128, le=1024)


class DatasetConvertRequest(BaseModel):
    source: str
    name: str = "viton_hd"
    val_fraction: float = Field(0.1, ge=0.0, le=0.5)
    test_fraction: float = Field(0.1, ge=0.0, le=0.5)
    max_samples: Optional[int] = None


class MessageResponse(BaseModel):
    ok: bool = True
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)


class StatusResponse(BaseModel):
    ok: bool = True
    data: Dict[str, Any] = Field(default_factory=dict)
