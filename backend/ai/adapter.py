"""Try-on adapter interface.

The frontend talks to *this* interface, never to a specific model. Swapping the diffusion
backend for another architecture (IDM-VTON, OOTDiffusion, a custom model) means writing one
new adapter class and registering it — no UI changes, no API changes.

Contract
--------
* ``is_ready()`` — can this adapter produce output right now?
* ``status()`` — human-readable + machine-readable explanation (used by Model Status page).
* ``try_on(...)`` — person image + garment image + mask -> composited RGB image.
* ``warmup()`` — optional eager load so the first user click is not a multi-second stall.
* ``unload()`` — free VRAM.

Adapters must never raise for "not installed"; they return a structured failure so the UI
can show a friendly message and keep the lightweight pipeline running.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TryOnRequest:
    """Everything an adapter needs for one synthesis."""

    person_image: np.ndarray                       # RGB, full frame or crop
    garment_image: np.ndarray                      # RGB, normalised garment canvas
    garment_mask: Optional[np.ndarray] = None      # uint8 alpha for the garment
    landmarks: Optional[np.ndarray] = None         # (33, 4) pixel landmarks when known
    category: str = "t-shirt"
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    resolution: Optional[int] = None
    mask_dilate: int = 12
    return_intermediate: bool = False


@dataclass
class TryOnResult:
    """Adapter output."""

    ok: bool
    image: Optional[np.ndarray] = None                       # composited RGB (person + garment)
    raw_generation: Optional[np.ndarray] = None              # unblended model output
    composited_mask: Optional[np.ndarray] = None
    backend: str = "unknown"
    latency_ms: float = 0.0
    reason: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_images: bool = False) -> Dict[str, Any]:
        from backend.utils.image_utils import to_data_uri

        payload: Dict[str, Any] = {
            "ok": self.ok,
            "backend": self.backend,
            "latency_ms": round(self.latency_ms, 1),
            "reason": self.reason,
            "warnings": self.warnings,
            "debug": self.debug,
        }
        if include_images:
            if self.image is not None:
                payload["image_data_uri"] = to_data_uri(self.image, "PNG")
            if self.raw_generation is not None:
                payload["raw_data_uri"] = to_data_uri(self.raw_generation, "PNG")
            if self.composited_mask is not None:
                payload["mask_data_uri"] = to_data_uri(np.dstack([self.composited_mask] * 3), "PNG")
        return payload


class VTONAdapter(abc.ABC):
    """Base class for virtual try-on backends."""

    name: str = "base"
    display_name: str = "Base adapter"

    def __init__(self, **kwargs: Any) -> None:
        self.loaded_at: float = 0.0
        self.last_error: Optional[str] = None
        self.call_count: int = 0
        self.total_latency_ms: float = 0.0
        self.extra: Dict[str, Any] = dict(kwargs)

    # -- lifecycle ---------------------------------------------------------------------
    @abc.abstractmethod
    def is_ready(self) -> bool:
        """True when ``try_on`` can run without further setup."""

    @abc.abstractmethod
    def status(self) -> Dict[str, Any]:
        """Detailed status for the dashboard."""

    @abc.abstractmethod
    def try_on(self, request: TryOnRequest) -> TryOnResult:
        """Synthesise a try-on image."""

    def warmup(self) -> bool:
        """Eagerly load weights. Returns ``True`` on success."""
        return self.is_ready()

    def unload(self) -> None:
        """Free memory. Default is a no-op."""
        logger.info("%s: unload() called (no-op)", self.name)

    # -- helpers -----------------------------------------------------------------------
    def _timed(self, fn, *args: Any, **kwargs: Any) -> TryOnResult:
        started = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            self.last_error = f"{exc.__class__.__name__}: {exc}"
            logger.error("%s inference failed: %s", self.name, exc, exc_info=True)
            return TryOnResult(ok=False, backend=self.name, reason=self.last_error,
                               latency_ms=(time.perf_counter() - started) * 1000.0)
        result.latency_ms = (time.perf_counter() - started) * 1000.0
        self.call_count += 1
        self.total_latency_ms += result.latency_ms
        return result

    def average_latency_ms(self) -> float:
        return round(self.total_latency_ms / self.call_count, 1) if self.call_count else 0.0

    def base_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "ready": self.is_ready(),
            "calls": self.call_count,
            "avg_latency_ms": self.average_latency_ms(),
            "last_error": self.last_error,
            "extra": self.extra,
        }


#: Backwards/forwards-compatible alias — the interface is referred to as ``TryOnAdapter``
#: in the docs and as ``VTONAdapter`` in the registry code.
TryOnAdapter = VTONAdapter


class NullAdapter(VTONAdapter):
    """Adapter used when no model is available.

    It never claims success. ``try_on`` returns a clear reason so the UI can explain what
    the user must do (train a checkpoint / install the ML stack), and the lightweight
    pipeline keeps providing the live overlay.
    """

    name = "disabled"
    display_name = "AI try-on disabled"

    def __init__(self, reason: str = "No AI model is available on this machine.") -> None:
        super().__init__()
        self.reason = reason

    def is_ready(self) -> bool:
        return False

    def status(self) -> Dict[str, Any]:
        payload = self.base_status()
        payload.update({"detail": self.reason, "ready": False})
        return payload

    def try_on(self, request: TryOnRequest) -> TryOnResult:
        return TryOnResult(
            ok=False,
            backend=self.name,
            reason=self.reason,
            warnings=["Live tracking is unaffected — only the photorealistic AI synthesis needs the model."],
        )

    def warmup(self) -> bool:
        return False
