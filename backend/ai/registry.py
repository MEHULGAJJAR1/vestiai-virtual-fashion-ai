"""Adapter registry — the single place that decides which try-on backend is active.

``build_adapter(name)`` returns an object satisfying :class:`~backend.ai.adapter.VTONAdapter`.
The frontend only ever talks to the service layer, which uses this registry, so swapping in
a new architecture (IDM-VTON, OOTDiffusion, a cloud endpoint) is a one-line registration.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from backend.ai.adapter import NullAdapter, VTONAdapter
from backend.models.checkpointing import CheckpointManager
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

_FACTORIES: Dict[str, Callable[..., VTONAdapter]] = {}


def register(name: str, factory: Callable[..., VTONAdapter]) -> None:
    """Register an adapter factory under ``name``."""
    _FACTORIES[name] = factory
    logger.debug("Registered VTON adapter factory: %s", name)


def available_adapters() -> Dict[str, str]:
    """Map of adapter name -> docstring first line (used by the About/Status pages)."""
    out: Dict[str, str] = {}
    for name, factory in _FACTORIES.items():
        doc = (factory.__doc__ or "").strip().splitlines()
        out[name] = doc[0] if doc else ""
    return out


def build_adapter(
    name: str = "auto",
    *,
    settings: Optional[Any] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    estimator: Optional[Any] = None,
) -> VTONAdapter:
    """Instantiate the requested adapter.

    ``auto`` prefers the fine-tuned diffusion model and falls back to the geometric warp —
    and always returns *some* adapter so the API never 500s.
    """
    name = (name or "auto").lower()
    checkpoint_manager = checkpoint_manager or CheckpointManager()

    if name == "auto":
        candidate = _FACTORIES["diffusion"](
            checkpoint_manager=checkpoint_manager, settings=settings, estimator=estimator, lazy=True
        )
        if candidate.is_ready():
            logger.info("Active try-on adapter: %s", candidate.display_name)
            return candidate
        base_fallback = bool(getattr(settings, "vton_allow_base_fallback", False)) if settings else False
        if base_fallback and candidate.status().get("mode") == "base_fallback":
            return candidate
        logger.info(
            "Diffusion adapter unavailable (%s). Falling back to the lightweight geometric pipeline.",
            candidate.status().get("detail", "unknown reason"),
        )
        lightweight = _FACTORIES["lightweight"](settings=settings, estimator=estimator)
        lightweight.extra["diffusion_unavailable_reason"] = candidate.status().get("detail")
        return lightweight

    if name == "diffusion":
        return _FACTORIES["diffusion"](checkpoint_manager=checkpoint_manager, settings=settings, estimator=estimator, lazy=True)
    if name == "lightweight":
        return _FACTORIES["lightweight"](settings=settings, estimator=estimator)
    if name == "disabled":
        return NullAdapter("AI try-on was disabled in Settings.")

    logger.warning("Unknown adapter '%s'; using auto.", name)
    return build_adapter("auto", settings=settings, checkpoint_manager=checkpoint_manager, estimator=estimator)


# --------------------------------------------------------------------------------------
# registrations
# --------------------------------------------------------------------------------------
def _make_lightweight(settings: Optional[Any] = None, estimator: Optional[Any] = None, **_kwargs: Any) -> VTONAdapter:
    """CPU-only pose-driven warp — always available, ~10 ms per image."""
    from backend.ai.lightweight import LightweightTryOnAdapter

    overrides = getattr(settings, "extra", {}).get("category_overrides") if settings else None
    return LightweightTryOnAdapter(estimator=estimator, category_overrides=overrides or {})


def _make_diffusion(
    settings: Optional[Any] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    estimator: Optional[Any] = None,
    lazy: bool = True,
    **_kwargs: Any,
) -> VTONAdapter:
    """Fine-tuned Stable-Diffusion ControlNet inpainting try-on (needs the ML extra + a checkpoint)."""
    from backend.ai.diffusion_adapter import DiffusionTryOnAdapter

    _ = estimator  # unused: the diffusion adapter creates its own estimator on demand
    if settings is None:
        return DiffusionTryOnAdapter(checkpoint_manager=checkpoint_manager, lazy=lazy)
    return DiffusionTryOnAdapter(
        checkpoint_manager=checkpoint_manager,
        base_model=getattr(settings, "vton_base_pipeline", None) or getattr(settings, "vton_pretrained", "stabilityai/stable-diffusion-2-inpainting"),
        device=getattr(settings, "device", "auto"),
        resolution=getattr(settings, "resolution", 512),
        num_inference_steps=getattr(settings, "num_inference_steps", 30),
        guidance_scale=getattr(settings, "guidance_scale", 2.0),
        seed=getattr(settings, "seed", 42),
        allow_base_fallback=bool(getattr(settings, "vton_allow_base_fallback", False)),
        enable_cpu=bool(getattr(settings, "enable_cpu_diffusion", False)),
        mixed_precision=getattr(settings, "mixed_precision", "auto"),
        lazy=lazy,
    )


register("lightweight", _make_lightweight)
register("diffusion", _make_diffusion)
