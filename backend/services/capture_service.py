"""Capture + recording storage.

The browser captures stills from the canvas and records clips with ``MediaRecorder``. Both
are uploaded here so they live with the project (not just in the browser's memory) and can be
listed, downloaded or deleted from the UI.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from backend.utils.errors import ValidationError
from backend.utils.image_utils import decode_base64, load_bytes, save_image
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

ALLOWED_VIDEO_SUFFIXES = {".webm", ".mp4", ".mkv", ".mov"}


@dataclass
class Capture:
    """A saved photo or clip."""

    capture_id: str
    kind: str                  # photo | video
    path: str
    created_at: float
    garment_key: Optional[str]
    backend: Optional[str]
    size_bytes: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capture_id": self.capture_id,
            "kind": self.kind,
            "url": f"/api/captures/{self.capture_id}/file",
            "thumbnail_url": f"/api/captures/{self.capture_id}/thumbnail" if self.kind == "photo" else None,
            "created_at": self.created_at,
            "created_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created_at)),
            "garment_key": self.garment_key,
            "backend": self.backend,
            "size_bytes": self.size_bytes,
            "size_kb": round(self.size_bytes / 1024, 1),
            "metadata": self.metadata,
        }


class CaptureService:
    """Manages ``captures/`` + ``captures/index.json``."""

    def __init__(self, captures_dir: str | Path = "captures") -> None:
        self.root = Path(captures_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self._items: Dict[str, Capture] = {}
        self._load()

    # -- persistence -------------------------------------------------------------------
    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Capture index unreadable (%s); starting empty.", exc)
            return
        for item in payload.get("captures", []):
            path = Path(item.get("path", ""))
            if not path.exists():
                continue
            try:
                self._items[item["capture_id"]] = Capture(
                    capture_id=item["capture_id"], kind=item.get("kind", "photo"), path=str(path),
                    created_at=float(item.get("created_at", 0.0)), garment_key=item.get("garment_key"),
                    backend=item.get("backend"), size_bytes=int(item.get("size_bytes", path.stat().st_size)),
                    metadata=item.get("metadata", {}),
                )
            except Exception:
                continue

    def _save(self) -> None:
        payload = {"count": len(self._items), "captures": [item.to_dict() | {"path": item.path} for item in self._items.values()]}
        self.index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -- writes ------------------------------------------------------------------------
    def save_photo(
        self,
        image: np.ndarray,
        garment_key: Optional[str] = None,
        backend: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Capture:
        capture_id = f"photo-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:5]}"
        path = save_image(image, self.root / f"{capture_id}.png")
        record = Capture(
            capture_id=capture_id, kind="photo", path=str(path), created_at=time.time(),
            garment_key=garment_key, backend=backend, size_bytes=path.stat().st_size,
            metadata=metadata or {},
        )
        self._items[capture_id] = record
        self._save()
        logger.info("Saved capture %s", capture_id)
        return record

    def save_photo_base64(self, payload: str, **kwargs: Any) -> Capture:
        try:
            image = decode_base64(payload)
        except Exception as exc:
            raise ValidationError(f"Could not decode the captured image: {exc}") from exc
        return self.save_photo(image, **kwargs)

    def save_video(
        self,
        data: bytes,
        filename: str = "clip.webm",
        garment_key: Optional[str] = None,
        backend: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Capture:
        if not data:
            raise ValidationError("The uploaded recording was empty.")
        suffix = Path(filename).suffix.lower() or ".webm"
        if suffix not in ALLOWED_VIDEO_SUFFIXES:
            raise ValidationError(f"Unsupported recording type '{suffix}'. Allowed: {sorted(ALLOWED_VIDEO_SUFFIXES)}")
        capture_id = f"video-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:5]}"
        path = self.root / f"{capture_id}{suffix}"
        path.write_bytes(data)
        record = Capture(
            capture_id=capture_id, kind="video", path=str(path), created_at=time.time(),
            garment_key=garment_key, backend=backend, size_bytes=path.stat().st_size,
            metadata=metadata or {"duration_s": None},
        )
        self._items[capture_id] = record
        self._save()
        logger.info("Saved recording %s (%.1f MB)", capture_id, record.size_bytes / 1024 ** 2)
        return record

    # -- reads -------------------------------------------------------------------------
    def list(self, kind: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        items = sorted(self._items.values(), key=lambda c: c.created_at, reverse=True)
        if kind:
            items = [item for item in items if item.kind == kind]
        return [item.to_dict() for item in items[:limit]]

    def get(self, capture_id: str) -> Capture:
        capture = self._items.get(capture_id)
        if capture is None:
            raise ValidationError(f"No capture '{capture_id}'.")
        return capture

    def delete(self, capture_id: str) -> bool:
        capture = self._items.pop(capture_id, None)
        if capture is None:
            return False
        Path(capture.path).unlink(missing_ok=True)
        self._save()
        return True

    def stats(self) -> Dict[str, Any]:
        photos = [item for item in self._items.values() if item.kind == "photo"]
        videos = [item for item in self._items.values() if item.kind == "video"]
        return {
            "total": len(self._items),
            "photos": len(photos),
            "videos": len(videos),
            "total_mb": round(sum(item.size_bytes for item in self._items.values()) / 1024 ** 2, 2),
            "latest": (sorted(self._items.values(), key=lambda c: c.created_at, reverse=True)[0].to_dict() if self._items else None),
        }
