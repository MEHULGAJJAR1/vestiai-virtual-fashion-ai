"""FastAPI routers — every endpoint the UI uses.

Endpoint map (see ``docs/API.md`` for the full reference with examples)::

    GET  /api/health                     liveness
    GET  /api/status                     model + system status
    GET  /api/system/devices             GPU/CPU detail
    GET  /api/system/components          dependency checklist
    GET  /api/system/disk                folder sizes
    GET  /api/settings, PATCH /api/settings
    POST /api/system/sample-closet       generate sample garments
    POST /api/system/cache/clear         free GPU cache

    POST /api/garments/upload            upload + full preprocessing pipeline
    POST /api/garments/preview           dry-run preprocessing (nothing stored)
    GET  /api/garments                   list + stats
    GET  /api/garments/{key}             one garment
    GET  /api/garments/{key}/file/{kind} serve image/mask/cutout/preview
    PATCH /api/garments/{key}            rename/reclassify/favourite
    DELETE /api/garments/{key}           delete
    POST /api/garments/{key}/reprocess   re-run preprocessing on the original

    POST /api/tryon                      single-image try-on (trained model or fallback)
    GET  /api/tryon/backends             list/select adapters
    POST /api/tryon/backend              switch adapter / reload checkpoint

    POST /api/live/session               create a live session
    POST /api/live/frame                 process one frame server-side
    POST /api/live/config                tune live AI frequency/resolution
    DELETE /api/live/session/{id}        close a session
    WS   /ws/live/{session_id}           streaming live try-on

    POST /api/training/start             start QUICK_DEMO / FINE_TUNE / FULL_TRAINING
    GET  /api/training/status            dashboard payload
    POST /api/training/stop              graceful stop
    GET  /api/training/checkpoints       checkpoint list
    POST /api/training/checkpoints/{name}/delete | /promote
    GET  /api/training/metrics           metric curves
    GET  /api/training/validation        validation runs + sample grids
    GET  /api/training/curves/image      training curve plot
    POST /api/training/dataset/generate | /convert
    GET  /api/training/dataset/list | /{name}/validate | /{name}/stats

    GET  /api/results, GET /api/results/{id}, GET /api/results/{id}/file/{kind}
    POST /api/captures/photo, POST /api/captures/video, GET /api/captures

    GET  /api/recommendations/styles | /recommend | /outfit
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

from backend.api.schemas import (
    CapturePhotoRequest, DatasetConvertRequest, DatasetGenerateRequest, FrameRequest,
    GarmentListResponse, GarmentUpdateRequest, LiveConfigRequest, MessageResponse,
    OutfitRequest, RecommendationRequest, SampleClosetRequest, SessionCreateRequest,
    SettingsUpdateRequest, StatusResponse, TrainingStartRequest, TrainingStopRequest,
    TryOnRequestModel,
)
from backend.utils.errors import VestiAIError, ValidationError
from backend.utils.image_utils import decode_base64, load_image
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api")
ws_router = APIRouter()


# --------------------------------------------------------------------------------------
# dependency accessors (services live on app.state, created in the lifespan)
# --------------------------------------------------------------------------------------
def get_container(request: Request) -> Any:
    """Return the service container attached in ``create_app``."""
    container = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - startup ordering issue
        raise HTTPException(status_code=503, detail="Services are still starting up.")
    return container


# ======================================================================================
# health / status
# ======================================================================================
@router.get("/health", tags=["system"], summary="Liveness probe")
def health(container: Any = Depends(get_container)) -> Dict[str, Any]:
    return container.system.health()


@router.get("/status", tags=["system"], summary="Aggregated model + system status")
def status(container: Any = Depends(get_container)) -> Dict[str, Any]:
    return container.system.model_status(
        adapter=container.tryon.adapter,
        closet_stats=container.closet.stats(),
    )


@router.get("/system/devices", tags=["system"], summary="Device / GPU detail")
def devices(container: Any = Depends(get_container)) -> Dict[str, Any]:
    return container.system.device_status()


@router.get("/system/components", tags=["system"], summary="Dependency checklist")
def components(container: Any = Depends(get_container)) -> Dict[str, Any]:
    items = container.system.components()
    return {"ok": True, "components": items, "required_missing": [c["module"] for c in items if c["required"] and not c["installed"]]}


@router.get("/system/disk", tags=["system"], summary="Runtime folder sizes")
def disk(container: Any = Depends(get_container)) -> Dict[str, Any]:
    return container.system.disk_usage()


@router.get("/system/about", tags=["system"], summary="About-page content")
def about(container: Any = Depends(get_container)) -> Dict[str, Any]:
    return container.system.about()


@router.post("/system/cache/clear", tags=["system"], summary="Free CUDA cache")
def clear_cache(container: Any = Depends(get_container)) -> Dict[str, Any]:
    from backend.utils.device import empty_cuda_cache

    empty_cuda_cache()
    return {"ok": True, "message": "CUDA cache cleared."}


@router.post("/system/sample-closet", tags=["system"], summary="Generate sample garments (no downloads)")
def sample_closet(payload: SampleClosetRequest, container: Any = Depends(get_container)) -> Dict[str, Any]:
    from backend.services.sample_data import populate_sample_closet

    created = populate_sample_closet(container.garments, count=payload.count, seed=payload.seed)
    return {"ok": True, "created": len(created), "garments": created, "stats": container.closet.stats()}


# ======================================================================================
# settings
# ======================================================================================
@router.get("/settings", tags=["settings"], summary="Current settings (curated view)")
def get_settings_view(container: Any = Depends(get_container)) -> Dict[str, Any]:
    return container.system.settings_view()


@router.patch("/settings", tags=["settings"], summary="Update runtime settings")
def patch_settings(payload: SettingsUpdateRequest, container: Any = Depends(get_container)) -> Dict[str, Any]:
    settings = container.settings
    applied: Dict[str, Any] = {}
    updates = payload.model_dump(exclude_none=True)
    category_overrides = updates.pop("category_overrides", None)
    for key, value in updates.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
            applied[key] = value
    if category_overrides is not None:
        settings.extra["category_overrides"] = category_overrides
        applied["category_overrides"] = category_overrides

    # Push live-affecting changes into the running sessions/adapters.
    if any(key in applied for key in ("live_ai_interval_ms", "smoothing_alpha", "resolution", "num_inference_steps")):
        container.tryon.live_config(
            ai_interval_ms=applied.get("live_ai_interval_ms"),
            smoothing_alpha=applied.get("smoothing_alpha"),
            resolution=applied.get("resolution"),
            num_inference_steps=applied.get("num_inference_steps"),
        )
    if applied.get("tryon_backend"):
        container.tryon.set_backend(applied["tryon_backend"])
    logger.info("Settings updated: %s", list(applied))
    return {"ok": True, "applied": applied, "settings": container.system.settings_view()}


# ======================================================================================
# garments
# ======================================================================================
@router.post("/garments/upload", tags=["garments"], summary="Upload + preprocess a garment")
async def upload_garment(
    file: UploadFile = File(..., description="Garment/product image (jpg, png, webp, bmp)"),
    category: Optional[str] = Form(None),
    label: Optional[str] = Form(None),
    skip_quality_gate: bool = Form(False),
    container: Any = Depends(get_container),
) -> Dict[str, Any]:
    data = await file.read()
    if not data:
        raise ValidationError("The uploaded file was empty.")
    record = container.garments.ingest(
        data,
        filename=file.filename or "upload.png",
        category_hint=category,
        label=label,
        source={"kind": "upload", "filename": file.filename, "content_type": file.content_type},
        skip_quality_gate=skip_quality_gate,
    )
    return {"ok": True, "garment": record, "stats": container.closet.stats()}


@router.get("/garments", response_model=GarmentListResponse, tags=["garments"], summary="List the closet")
def list_garments(
    category: Optional[str] = Query(None),
    garment_type: Optional[str] = Query(None),
    favourite: Optional[bool] = Query(None),
    include_preview: bool = Query(False, description="Embed base64 previews (slower, handy for offline UI)"),
    container: Any = Depends(get_container),
) -> Dict[str, Any]:
    records = container.closet.filter(category=category, garment_type=garment_type, favourite=favourite)
    return {
        "ok": True,
        "count": len(records),
        "garments": [record.to_dict(include_data_uri=include_preview) for record in records],
        "stats": container.closet.stats(),
    }


@router.get("/garments/stats", tags=["garments"], summary="Closet statistics")
def garment_stats(container: Any = Depends(get_container)) -> Dict[str, Any]:
    return {"ok": True, **container.closet.stats(), "classifier": container.garments.classifier_status()}


@router.get("/garments/{key}", tags=["garments"], summary="One garment")
def get_garment(key: str, include_preview: bool = Query(True), container: Any = Depends(get_container)) -> Dict[str, Any]:
    record = container.closet.get(key)
    return {"ok": True, "garment": record.to_dict(include_data_uri=include_preview)}


@router.get("/garments/{key}/file/{kind}", tags=["garments"], summary="Serve a garment file")
def garment_file(key: str, kind: str, container: Any = Depends(get_container)) -> FileResponse:
    record = container.closet.get(key)
    mapping = {
        "image": record.image_path,
        "mask": record.mask_path,
        "cutout": record.cutout_path,
        "preview": record.preview_path,
    }
    path = mapping.get(kind)
    if path is None:
        raise ValidationError(f"Unknown file kind '{kind}'. Use one of {sorted(mapping)}.")
    if not Path(path).exists():
        raise VestiAIError("That garment file is missing from disk.")
    media = "image/png" if Path(path).suffix == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media)


@router.patch("/garments/{key}", tags=["garments"], summary="Update a garment")
def update_garment(key: str, payload: GarmentUpdateRequest, container: Any = Depends(get_container)) -> Dict[str, Any]:
    record = container.closet.update(key, **payload.model_dump(exclude_none=True))
    return {"ok": True, "garment": record.to_dict()}


@router.delete("/garments/{key}", tags=["garments"], summary="Delete a garment")
def delete_garment(key: str, container: Any = Depends(get_container)) -> Dict[str, Any]:
    deleted = container.closet.delete(key)
    if not deleted:
        raise VestiAIError("Garment not found.", details={"key": key})
    return {"ok": True, "message": f"Deleted {key}.", "stats": container.closet.stats()}


@router.post("/garments/{key}/reprocess", tags=["garments"], summary="Re-run preprocessing")
def reprocess_garment(key: str, container: Any = Depends(get_container)) -> Dict[str, Any]:
    record = container.garments.reprocess(key)
    return {"ok": True, "garment": record, "stats": container.closet.stats()}


@router.post("/garments/preview", tags=["garments"], summary="Dry-run preprocessing (nothing stored)")
async def preview_garment(
    file: UploadFile = File(...),
    category: Optional[str] = Form(None),
    container: Any = Depends(get_container),
) -> Dict[str, Any]:
    """Validate + segment an upload without touching the closet — used by the upload dialog."""
    from backend.services.garment_service import dominant_colours
    from backend.cv.segmentation import extract_garment_region, garment_quality_report, refine_alpha
    from backend.utils.image_utils import to_data_uri, to_rgb

    data = await file.read()
    if not data:
        raise ValidationError("The uploaded file was empty.")
    try:
        from backend.utils.image_utils import load_bytes

        rgba = load_bytes(data, mode="RGBA")
    except ValueError as exc:
        raise ValidationError(str(exc), details={"filename": file.filename}) from exc

    rgb = to_rgb(rgba)
    service = container.garments
    cutout, alpha = service.remover.remove(rgb)
    alpha = refine_alpha(rgb, alpha)[..., 3]
    quality = garment_quality_report(alpha, rgb, min_resolution=service.min_resolution)
    classification = service.classifier.classify(alpha, rgb)
    canvas_rgb, canvas_alpha = extract_garment_region(rgb, alpha, target_canvas=min(512, service.canvas))
    return {
        "ok": True,
        "quality": quality,
        "classification": classification.to_dict() if hasattr(classification, "to_dict") else classification,
        "palette": dominant_colours(rgb, alpha),
        "preview_data_uri": to_data_uri(cutout, "PNG"),
        "canvas_data_uri": to_data_uri(canvas_rgb, "JPEG"),
        "mask_data_uri": to_data_uri(canvas_alpha, "PNG"),
        "would_accept": quality.get("ok", True),
    }


# ======================================================================================
# try-on
# ======================================================================================
@router.post("/tryon", tags=["try-on"], summary="Single-image virtual try-on")
def run_tryon(payload: TryOnRequestModel, container: Any = Depends(get_container)) -> Dict[str, Any]:
    if not payload.person_base64:
        raise ValidationError("person_base64 is required (send a photo or a canvas capture).")
    person = decode_base64(payload.person_base64)
    garment = container.closet.get(payload.garment_key)
    if payload.category:
        container.closet.update(garment.key, category=payload.category)
        garment = container.closet.get(garment.key)
    result = container.tryon.run(
        person_image=person,
        garment=garment,
        prompt=payload.prompt,
        steps=payload.num_inference_steps,
        guidance=payload.guidance_scale,
        resolution=payload.resolution,
        seed=payload.seed,
        backend=payload.backend,
    )
    return {"ok": True, **result}


@router.get("/tryon/backends", tags=["try-on"], summary="Available try-on backends")
def tryon_backends(container: Any = Depends(get_container)) -> Dict[str, Any]:
    return {"ok": True, **container.tryon.adapter_status()}


@router.post("/tryon/backend", tags=["try-on"], summary="Switch backend / reload checkpoint")
def set_tryon_backend(
    name: str = Query("auto", description="auto | diffusion | lightweight | disabled"),
    reload: bool = Query(True),
    container: Any = Depends(get_container),
) -> Dict[str, Any]:
    return {"ok": True, **container.tryon.set_backend(name, reload_model=reload)}


@router.post("/tryon/reload", tags=["try-on"], summary="Re-scan checkpoints and reload the model")
def reload_tryon_model(container: Any = Depends(get_container)) -> Dict[str, Any]:
    return {"ok": True, **container.tryon.reload_model()}


# ======================================================================================
# live
# ======================================================================================
@router.post("/live/session", tags=["live"], summary="Create a live try-on session")
def create_live_session(payload: SessionCreateRequest, container: Any = Depends(get_container)) -> Dict[str, Any]:
    return {"ok": True, **container.tryon.create_session(payload.session_id, payload.garment_key, payload.enable_ai)}


@router.get("/live/sessions", tags=["live"], summary="List live sessions")
def list_live_sessions(container: Any = Depends(get_container)) -> Dict[str, Any]:
    sessions = container.tryon.sessions_status()
    return {"ok": True, "count": len(sessions), "sessions": sessions}


@router.delete("/live/session/{session_id}", tags=["live"], summary="Close a live session")
def close_live_session(session_id: str, container: Any = Depends(get_container)) -> Dict[str, Any]:
    closed = container.tryon.close_session(session_id)
    return {"ok": closed, "message": "Session closed." if closed else "Session not found."}


@router.post("/live/frame", tags=["live"], summary="Process one frame server-side")
def process_frame(payload: FrameRequest, container: Any = Depends(get_container)) -> Dict[str, Any]:
    return container.tryon.process_frame(
        payload.session_id, payload.frame_base64, payload.garment_key,
        force_ai=payload.force_ai, mirror=payload.mirror,
    )


@router.post("/live/config", tags=["live"], summary="Tune live inference")
def live_config(payload: LiveConfigRequest, container: Any = Depends(get_container)) -> Dict[str, Any]:
    return container.tryon.live_config(
        ai_enabled=payload.ai_enabled, ai_interval_ms=payload.ai_interval_ms,
        smoothing_alpha=payload.smoothing_alpha, resolution=payload.resolution,
        num_inference_steps=payload.num_inference_steps,
    )


@router.get("/live/config", tags=["live"], summary="Current live inference configuration")
def live_config_view(container: Any = Depends(get_container)) -> Dict[str, Any]:
    settings = container.settings
    return {
        "ok": True,
        "config": {
            "ai_interval_ms": settings.live_ai_interval_ms,
            "live_ai_max_side": settings.live_ai_max_side,
            "smoothing_alpha": settings.smoothing_alpha,
            "segmentation": settings.segmentation_enabled,
            "pose_complexity": settings.pose_model_complexity,
            "resolution": settings.resolution,
            "num_inference_steps": settings.num_inference_steps,
            "client_side_pose": True,
        },
        "adapter": container.tryon.adapter.status(),
        "sessions": container.tryon.sessions_status(),
    }


@ws_router.websocket("/ws/live/{session_id}")
async def live_websocket(websocket: WebSocket, session_id: str) -> None:
    """Streaming live try-on.

    Client -> server: ``{"type":"frame","frame":"<base64 jpeg>","garment_key":"...","mirror":true}``
    Server -> client: ``{"type":"result","overlay":"<base64 jpeg>","status":{...}}``

    The heavy diffusion model is only invoked when the client sets ``"ai": true`` and the
    configured interval has elapsed; the geometric warp answers every frame.
    """
    await websocket.accept()
    container = getattr(websocket.app.state, "container", None)
    if container is None:  # pragma: no cover
        await websocket.close(code=1013)
        return

    garment_key: Optional[str] = None
    ai_enabled = False
    frames = 0
    logger.info("WebSocket live session %s connected", session_id)
    try:
        while True:
            message = await websocket.receive_json()
            kind = message.get("type", "frame")
            if kind == "ping":
                await websocket.send_json({"type": "pong", "t": time.time()})
                continue
            if kind == "config":
                ai_enabled = bool(message.get("ai", ai_enabled))
                garment_key = message.get("garment_key", garment_key)
                await websocket.send_json({"type": "ack", "config": {"ai": ai_enabled, "garment_key": garment_key}})
                continue
            if kind == "close":
                break
            if kind != "frame":
                await websocket.send_json({"type": "error", "message": f"Unknown message type '{kind}'"})
                continue

            garment_key = message.get("garment_key", garment_key)
            frame = message.get("frame")
            if not frame:
                await websocket.send_json({"type": "error", "message": "Frame data missing."})
                continue
            try:
                container.tryon.get_session(session_id)
            except Exception:
                container.tryon.create_session(session_id, garment_key, enable_ai=False)
            result = container.tryon.process_frame(
                session_id, frame, garment_key,
                force_ai=bool(message.get("ai")) and ai_enabled,
                mirror=bool(message.get("mirror", False)),
            )
            frames += 1
            await websocket.send_json({"type": "result", **result})
    except WebSocketDisconnect:
        logger.info("WebSocket live session %s disconnected after %d frames", session_id, frames)
    except Exception as exc:  # pragma: no cover - transport errors
        logger.warning("WebSocket session %s error: %s", session_id, exc)
    finally:
        container.tryon.close_session(session_id)


# ======================================================================================
# training
# ======================================================================================
@router.post("/training/start", tags=["training"], summary="Start a training run")
def start_training(payload: TrainingStartRequest, container: Any = Depends(get_container)) -> Dict[str, Any]:
    job = container.training.start(
        mode=payload.mode, config_overrides=payload.overrides,
        dataset_root=payload.dataset_root, dry_run=payload.dry_run,
    )
    return {"ok": True, "job": job}


@router.get("/training/status", tags=["training"], summary="Training dashboard payload")
def training_status(container: Any = Depends(get_container)) -> Dict[str, Any]:
    return {"ok": True, **container.training.status()}


@router.post("/training/stop", tags=["training"], summary="Stop the active run")
def stop_training(payload: TrainingStopRequest, container: Any = Depends(get_container)) -> Dict[str, Any]:
    job = container.training.stop(payload.job_id)
    return {"ok": True, "job": job}


@router.get("/training/logs", tags=["training"], summary="Tail of the training log")
def training_logs(lines: int = Query(120, ge=10, le=2000), container: Any = Depends(get_container)) -> Dict[str, Any]:
    return {"ok": True, "lines": container.training.log_tail(lines)}


@router.get("/training/checkpoints", tags=["training"], summary="List checkpoints")
def training_checkpoints(container: Any = Depends(get_container)) -> Dict[str, Any]:
    return {"ok": True, **container.training.checkpoints_view()}


@router.delete("/training/checkpoints/{name}", tags=["training"], summary="Delete an epoch checkpoint")
def delete_checkpoint(name: str, container: Any = Depends(get_container)) -> Dict[str, Any]:
    return {"ok": True, **container.training.delete_checkpoint(name)}


@router.post("/training/checkpoints/{name}/promote", tags=["training"], summary="Promote a checkpoint to best_model")
def promote_checkpoint(name: str, container: Any = Depends(get_container)) -> Dict[str, Any]:
    return {"ok": True, **container.training.promote_checkpoint(name)}


@router.get("/training/metrics", tags=["training"], summary="Metric curves")
def training_metrics(container: Any = Depends(get_container)) -> Dict[str, Any]:
    status = container.training.status()
    return {
        "ok": True,
        "metrics": status.get("metrics", {}),
        "history": status.get("history", {}),
        "trainer": status.get("trainer", {}),
    }


@router.get("/training/validation", tags=["training"], summary="Validation runs + samples")
def training_validation(container: Any = Depends(get_container)) -> Dict[str, Any]:
    from backend.training.validate import load_validation_summary

    summary = load_validation_summary(Path(container.settings.results_dir) / "validation")
    grids = sorted((Path(container.settings.results_dir) / "validation").glob("step_*/sample_*.jpg"))
    return {
        "ok": True,
        **summary,
        "grids": [f"/api/training/validation/image/{path.parent.name}/{path.name}" for path in grids[-24:]],
    }


@router.get("/training/validation/image/{step_name}/{filename}", tags=["training"], summary="Serve a validation grid")
def validation_image(step_name: str, filename: str, container: Any = Depends(get_container)) -> FileResponse:
    path = Path(container.settings.results_dir) / "validation" / step_name / filename
    if not path.exists() or ".." in step_name or ".." in filename:
        raise VestiAIError("Validation image not found.")
    return FileResponse(str(path), media_type="image/jpeg")


@router.get("/training/curves", tags=["training"], summary="Training curves data + plot")
def training_curves(container: Any = Depends(get_container)) -> Dict[str, Any]:
    return container.training.curves()


@router.get("/training/curves/image", tags=["training"], summary="Training curve plot (PNG or SVG)")
def training_curves_image(container: Any = Depends(get_container)) -> Response:
    from backend.training.evaluate import plot_training_curves

    history = container.training.status().get("history", {}).get("epochs", [])
    path = plot_training_curves(history, Path(container.settings.results_dir) / "training_curves.png")
    if path.suffix == ".svg":
        return Response(path.read_text(encoding="utf-8"), media_type="image/svg+xml")
    return FileResponse(str(path), media_type="image/png")


@router.get("/training/evaluations", tags=["training"], summary="Evaluation reports")
def training_evaluations(container: Any = Depends(get_container)) -> Dict[str, Any]:
    from backend.training.evaluate import load_evaluations

    reports = load_evaluations(Path(container.settings.results_dir) / "evaluation")
    return {"ok": True, "reports": reports, "count": len(reports)}


# ---- datasets ------------------------------------------------------------------------
@router.get("/training/dataset/list", tags=["training"], summary="Datasets on disk")
def dataset_list(container: Any = Depends(get_container)) -> Dict[str, Any]:
    return {"ok": True, "datasets": container.datasets.list_datasets(), "active": container.training._resolve_dataset_root()}


@router.post("/training/dataset/generate", tags=["training"], summary="Generate the sample dataset")
def dataset_generate(payload: DatasetGenerateRequest, container: Any = Depends(get_container)) -> Dict[str, Any]:
    result = container.datasets.generate_samples(
        count={"train": payload.train, "val": payload.val, "test": payload.test}, size=payload.size,
    )
    return {"ok": True, **result}


@router.post("/training/dataset/convert", tags=["training"], summary="Convert a VITON-HD/DressCode folder")
def dataset_convert(payload: DatasetConvertRequest, container: Any = Depends(get_container)) -> Dict[str, Any]:
    from backend.datasets.preprocessing import DatasetConfig

    config = DatasetConfig(
        root=str(Path(container.settings.datasets_dir) / payload.name),
        val_fraction=payload.val_fraction, test_fraction=payload.test_fraction, max_samples=payload.max_samples,
    )
    result = container.datasets.convert_viton_folder(payload.source, name=payload.name, config=config)
    return {"ok": True, **result}


@router.get("/training/dataset/{name}/validate", tags=["training"], summary="Validate a dataset layout")
def dataset_validate(name: str, strict: bool = Query(False), container: Any = Depends(get_container)) -> Dict[str, Any]:
    return {"ok": True, "report": container.datasets.validate(name, strict=strict)}


@router.get("/training/dataset/{name}/stats", tags=["training"], summary="Dataset statistics")
def dataset_stats(name: str, container: Any = Depends(get_container)) -> Dict[str, Any]:
    return {"ok": True, "report": container.datasets.statistics(name)}


# ======================================================================================
# results / captures
# ======================================================================================
@router.get("/results", tags=["results"], summary="Saved try-on results")
def list_results(limit: int = Query(50, ge=1, le=500), container: Any = Depends(get_container)) -> Dict[str, Any]:
    results = container.tryon.list_results(limit)
    return {"ok": True, "count": len(results), "results": results}


@router.get("/results/{result_id}", tags=["results"], summary="One result")
def get_result(result_id: str, container: Any = Depends(get_container)) -> Dict[str, Any]:
    record = container.tryon.get_result(result_id)
    return {"ok": True, "result": record.to_dict()}


@router.get("/results/{result_id}/file/{kind}", tags=["results"], summary="Serve a result file")
def result_file(result_id: str, kind: str, container: Any = Depends(get_container)) -> FileResponse:
    record = container.tryon.get_result(result_id)
    mapping = {
        "output": "output.png", "comparison": "comparison.jpg",
        "person": "person.jpg", "garment": "garment.jpg", "mask": "mask.png",
    }
    filename = mapping.get(kind)
    if filename is None:
        raise ValidationError(f"Unknown file kind '{kind}'. Use one of {sorted(mapping)}.")
    path = Path(record.directory) / filename
    if not path.exists():
        raise VestiAIError(f"'{kind}' is not available for this result.")
    return FileResponse(str(path), media_type="image/png" if path.suffix == ".png" else "image/jpeg")


@router.delete("/results/{result_id}", tags=["results"], summary="Delete a result")
def delete_result(result_id: str, container: Any = Depends(get_container)) -> Dict[str, Any]:
    deleted = container.tryon.delete_result(result_id)
    return {"ok": deleted, "message": "Result deleted." if deleted else "Result not found."}


@router.post("/captures/photo", tags=["captures"], summary="Save a captured photo")
def capture_photo(payload: CapturePhotoRequest, container: Any = Depends(get_container)) -> Dict[str, Any]:
    capture = container.captures.save_photo_base64(
        payload.image_base64, garment_key=payload.garment_key, backend=payload.backend, metadata=payload.metadata,
    )
    return {"ok": True, "capture": capture.to_dict(), "stats": container.captures.stats()}


@router.post("/captures/video", tags=["captures"], summary="Save a recorded clip")
async def capture_video(
    file: UploadFile = File(..., description="WebM/MP4 recording from MediaRecorder"),
    garment_key: Optional[str] = Form(None),
    backend: Optional[str] = Form(None),
    duration_s: Optional[float] = Form(None),
    container: Any = Depends(get_container),
) -> Dict[str, Any]:
    data = await file.read()
    capture = container.captures.save_video(
        data, filename=file.filename or "clip.webm", garment_key=garment_key, backend=backend,
        metadata={"duration_s": duration_s, "content_type": file.content_type},
    )
    return {"ok": True, "capture": capture.to_dict(), "stats": container.captures.stats()}


@router.get("/captures", tags=["captures"], summary="List captures")
def list_captures(kind: Optional[str] = Query(None, pattern="^(photo|video)$"), limit: int = Query(100, ge=1, le=500), container: Any = Depends(get_container)) -> Dict[str, Any]:
    return {"ok": True, "captures": container.captures.list(kind, limit), "stats": container.captures.stats()}


@router.get("/captures/{capture_id}/file", tags=["captures"], summary="Serve a capture")
def capture_file(capture_id: str, container: Any = Depends(get_container)) -> FileResponse:
    capture = container.captures.get(capture_id)
    path = Path(capture.path)
    if not path.exists():
        raise VestiAIError("Capture file is missing from disk.")
    media = "video/webm" if capture.kind == "video" else ("image/png" if path.suffix == ".png" else "image/jpeg")
    return FileResponse(str(path), media_type=media, filename=path.name)


@router.delete("/captures/{capture_id}", tags=["captures"], summary="Delete a capture")
def delete_capture(capture_id: str, container: Any = Depends(get_container)) -> Dict[str, Any]:
    deleted = container.captures.delete(capture_id)
    return {"ok": deleted, "message": "Capture deleted." if deleted else "Capture not found."}


# ======================================================================================
# recommendations / outfits
# ======================================================================================
@router.get("/recommendations/styles", tags=["recommendations"], summary="Available styles")
def styles(container: Any = Depends(get_container)) -> Dict[str, Any]:
    return {"ok": True, "styles": container.recommendations.styles()}


@router.post("/recommendations/recommend", tags=["recommendations"], summary="Style recommendations")
def recommend(payload: RecommendationRequest, container: Any = Depends(get_container)) -> Dict[str, Any]:
    return container.recommendations.recommend(payload.style, payload.limit, payload.seed)


@router.post("/recommendations/outfit", tags=["recommendations"], summary="Build an outfit")
def build_outfit(payload: OutfitRequest, container: Any = Depends(get_container)) -> Dict[str, Any]:
    return container.recommendations.build_outfit(payload.style, payload.seed)


@router.get("/recommendations/pairing/{garment_key}", tags=["recommendations"], summary="Colour pairings")
def colour_pairing(garment_key: str, container: Any = Depends(get_container)) -> Dict[str, Any]:
    return {"ok": True, **container.recommendations.colour_pairing(garment_key)}


@router.get("/openapi/lite", tags=["system"], summary="Endpoint list for the About page")
def endpoint_list() -> Dict[str, Any]:
    lines = [line.strip() for line in (__doc__ or "").splitlines() if line.strip().startswith(("GET", "POST", "PATCH", "DELETE", "WS"))]
    return {"ok": True, "endpoints": lines}
