"""FastAPI application factory for VestiAI.

Responsibilities
----------------
* build the service container (settings, closet, garment service, try-on service, training
  service, system service, datasets, captures, recommendations) once at startup,
* wire exception handlers so every failure returns a machine-readable ``error_code`` plus a
  friendly message,
* mount the routers and the static frontend (which is what makes the app a single-process
  «open the browser and it works» experience),
* expose ``/health`` and ``/openapi.json`` (FastAPI provides the latter automatically).
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.api.routes import router as api_router, ws_router
from backend.config import Settings, get_settings
from backend.models.checkpointing import CheckpointManager
from backend.services.capture_service import CaptureService
from backend.services.dataset_service import DatasetService
from backend.services.garment_service import ClosetStore, GarmentService
from backend.services.recommendation_service import RecommendationService
from backend.services.system_service import SystemService
from backend.services.training_service import TrainingService
from backend.services.tryon_service import TryOnService
from backend.utils.errors import VestiAIError
from backend.utils.logging_utils import get_logger, setup_logging

logger = get_logger("api")


class ServiceContainer:
    """Owns every long-lived service. Attached to ``app.state.container``."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.closet = ClosetStore(settings.garments_dir, settings.closet_db)
        self.garments = GarmentService(
            closet=self.closet,
            uploads_dir=settings.uploads_dir,
            canvas=settings.target_garment_canvas,
            min_resolution=settings.garment_min_resolution,
            max_side=settings.garment_max_side,
            background_removal=settings.background_removal,
            enable_zero_shot=bool(settings.extra.get("enable_zero_shot_classifier", True)),
            device=settings.resolve_device(),
            classifier_head=settings.extra.get("classifier_head"),
        )
        self.checkpoints = CheckpointManager(settings.checkpoints_dir)
        self.tryon = TryOnService(settings, self.closet, self.checkpoints)
        self.training = TrainingService(settings, self.checkpoints)
        self.system = SystemService(settings, self.checkpoints)
        self.datasets = DatasetService(settings.datasets_dir)
        self.captures = CaptureService(settings.captures_dir)
        self.recommendations = RecommendationService(self.closet)
        self.started_at = time.time()

    # -- lifecycle ---------------------------------------------------------------------
    def warmup(self) -> Dict[str, Any]:
        """Optionally preload the AI model (only when a checkpoint and ML stack exist)."""
        report: Dict[str, Any] = {"warmed": False}
        if self.settings.extra.get("preload_model", False):
            try:
                report["warmed"] = self.tryon.adapter.warmup()
                report["detail"] = self.tryon.adapter.status().get("detail")
            except Exception as exc:  # pragma: no cover
                logger.warning("Model preload failed: %s", exc)
                report["detail"] = str(exc)
        return report

    def shutdown(self) -> None:
        try:
            self.training.shutdown()
        finally:
            try:
                self.tryon.shutdown()
            finally:
                logger.info("VestiAI services shut down.")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Build the FastAPI application."""
    settings = settings or get_settings()
    setup_logging(settings.log_level, settings.log_dir)
    settings.ensure_dirs()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = ServiceContainer(settings)
        app.state.container = container
        logger.info("%s %s starting — device=%s, closet=%d garments, checkpoints=%s",
                    settings.app_name, settings.app_version, container.settings.resolve_device(),
                    len(container.closet.all()), container.checkpoints.resolve_for_inference("auto") or "none")
        container.warmup()
        try:
            yield
        finally:
            container.shutdown()

    app = FastAPI(
        title=f"{settings.app_name} API",
        version=settings.app_version,
        description=(
            "VestiAI — dual-pipeline virtual try-on.\n\n"
            "* **Real-time pipeline**: BlazePose landmarks → affinity/perspective warp → occlusion handling.\n"
            "* **AI pipeline**: fine-tuned Stable-Diffusion ControlNet inpainting for photorealistic synthesis.\n\n"
            "Training endpoints run real fine-tuning on a VITON-HD/DressCode-compatible dataset; "
            "when no checkpoint exists the AI endpoints explain exactly what to run and the geometric "
            "pipeline keeps working."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in str(settings.cors_origins).split(",")] or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----- exception handlers ---------------------------------------------------------
    @app.exception_handler(VestiAIError)
    async def vestiai_error_handler(request: Request, exc: VestiAIError) -> JSONResponse:
        log = logger.error if exc.status_code >= 500 else logger.warning
        log("%s %s -> %s (%s)", request.method, request.url.path, exc.error_code, exc.message)
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "ok": False, "error_code": "validation_error",
                "message": "Request validation failed.",
                "user_message": "Some fields were invalid — please check the form and try again.",
                "details": {"errors": exc.errors()[:8], "path": str(request.url.path)},
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code == 404 and not str(request.url.path).startswith(("/api", "/ws")):
            index = Path(settings.project_root) / "frontend" / "index.html"
            if index.exists():
                return FileResponse(str(index), media_type="text/html")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "ok": False, "error_code": "http_error", "message": str(exc.detail),
                "user_message": str(exc.detail), "details": {"status": exc.status_code},
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "ok": False, "error_code": "internal_error",
                "message": f"{exc.__class__.__name__}: {exc}",
                "user_message": "Something went wrong inside VestiAI. The details are in logs/vestiai.log.",
                "details": {},
            },
        )

    # ----- routes ---------------------------------------------------------------------
    app.include_router(api_router)
    app.include_router(ws_router)

    @app.get("/health", tags=["system"], summary="Health check (alias of /api/health)")
    async def health_alias(request: Request) -> Dict[str, Any]:
        return request.app.state.container.system.health()

    # ----- static frontend ------------------------------------------------------------
    frontend = Path(settings.project_root) / "frontend"
    if frontend.exists():
        app.mount("/static", StaticFiles(directory=str(frontend)), name="static")
        assets = frontend / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(str(frontend / "index.html"))

        @app.get("/favicon.ico", include_in_schema=False)
        async def favicon():
            icon = frontend / "assets" / "favicon.svg"
            if icon.exists():
                return FileResponse(str(icon), media_type="image/svg+xml")
            return JSONResponse(status_code=204, content=None)
    else:  # pragma: no cover - API-only deployment
        @app.get("/", include_in_schema=False)
        async def root_info() -> Dict[str, Any]:
            return {
                "app": settings.app_name, "version": settings.app_version,
                "docs": "/docs", "note": "frontend/ is not present in this deployment.",
            }

    return app


app = create_app()
