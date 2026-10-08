"""Garment intake: upload -> validate -> segment -> classify -> normalise -> closet.

Pipeline (all steps real, no placeholders)::

    raw upload (any product photo from Amazon/Flipkart/your camera)
      ├─ 1. decode + EXIF fix + size guard                 -> InvalidImageError / LowResolutionError
      ├─ 2. downscale to garment_max_side                  -> speed + memory
      ├─ 3. background removal (rembg if installed, else GrabCut)
      ├─ 4. alpha refinement + largest-component keep      -> clean cut-out
      ├─ 5. quality report (skin ratio, aspect, coverage)  -> UnsupportedGarmentError
      ├─ 6. category classification (CLIP -> head -> rules)
      ├─ 7. crop to bbox + letterbox onto a square canvas  -> canonical garment tensor
      └─ 8. persist PNG cut-out + mask + previews + JSON record in the closet index

The closet is a JSON index with per-garment folders so it is trivially inspectable,
backup-able and usable without a database.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from backend.cv.segmentation import BackgroundRemover, extract_garment_region, garment_quality_report, refine_alpha
from backend.services.garment_classifier import CATEGORY_GROUPS, GarmentClassifier, describe_category
from backend.utils.errors import GarmentNotFoundError, InvalidImageError, LowResolutionError, UnsupportedGarmentError
from backend.utils.image_utils import (
    load_bytes, load_image, resize_keep_aspect, save_image, to_data_uri, to_rgb,
)
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class GarmentRecord:
    """One garment in the closet."""

    key: str
    label: str
    category: str
    created_at: float
    original_name: str
    image_path: str            # normalised RGB canvas (PNG, square)
    mask_path: str             # alpha mask (PNG, square, mode L)
    cutout_path: str           # RGBA cut-out at natural aspect (PNG)
    preview_path: str          # 320px JPEG preview for the UI
    width: int
    height: int
    canvas: int
    classification: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    source: Dict[str, Any] = field(default_factory=dict)
    palette: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    favourite: bool = False
    garment_type: str = "top"   # top | bottom | shoes | accessory (drives outfit slots)

    # -- derived -----------------------------------------------------------------------
    @property
    def preview_url(self) -> str:
        return f"/api/garments/{self.key}/file/preview"

    @property
    def image_url(self) -> str:
        return f"/api/garments/{self.key}/file/image"

    @property
    def mask_url(self) -> str:
        return f"/api/garments/{self.key}/file/mask"

    @property
    def cutout_url(self) -> str:
        return f"/api/garments/{self.key}/file/cutout"

    # -- serialisation -----------------------------------------------------------------
    def to_storage_dict(self) -> Dict[str, Any]:
        """Exact constructor payload — guarantees ``closet.json`` can be read back.

        ``to_dict()`` is the *API* shape (flattened, with URLs); this one is the *storage*
        shape. Keeping them separate is what stops a saved closet from being silently
        dropped on the next start-up.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_storage_dict(cls, payload: Mapping[str, Any]) -> "GarmentRecord":
        """Rebuild a record from disk, also accepting older ``to_dict()`` payloads."""
        data: Dict[str, Any] = dict(payload)
        files = data.pop("files", None) or {}
        for name in ("image_path", "mask_path", "cutout_path", "preview_path"):
            if not data.get(name) and files.get(name):
                data[name] = files[name]
        size = data.pop("size", None) or {}
        if isinstance(size, Mapping):
            if not data.get("width") and size.get("width"):
                data["width"] = int(size["width"])
            if not data.get("height") and size.get("height"):
                data["height"] = int(size["height"])
        data.pop("category_label", None)
        data.pop("created_at_iso", None)
        data.pop("urls", None)
        data.pop("preview_data_uri", None)
        data.pop("cutout_data_uri", None)
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            logger.debug("Ignoring unknown closet fields %s for %s", unknown, data.get("key"))
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def to_dict(self, include_data_uri: bool = False) -> Dict[str, Any]:
        payload = {
            "key": self.key,
            "label": self.label,
            "category": self.category,
            "category_label": describe_category(self.category),
            "garment_type": self.garment_type,
            "created_at": self.created_at,
            "created_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created_at)),
            "original_name": self.original_name,
            "canvas": self.canvas,
            "size": {"width": self.width, "height": self.height},
            "classification": self.classification,
            "quality": self.quality,
            "source": self.source,
            "palette": self.palette,
            "tags": self.tags,
            "favourite": self.favourite,
            "urls": {
                "preview": self.preview_url,
                "image": self.image_url,
                "mask": self.mask_url,
                "cutout": self.cutout_url,
            },
            "files": {
                "image_path": self.image_path,
                "mask_path": self.mask_path,
                "cutout_path": self.cutout_path,
                "preview_path": self.preview_path,
            },
        }
        if include_data_uri:
            try:
                payload["preview_data_uri"] = to_data_uri(load_image(self.preview_path), "JPEG")
                payload["cutout_data_uri"] = to_data_uri(load_image(self.cutout_path, mode="RGBA"), "PNG")
            except Exception as exc:  # pragma: no cover
                logger.debug("Could not embed data URI for %s: %s", self.key, exc)
        return payload


class ClosetStore:
    """JSON-backed garment library."""

    INDEX_NAME = "closet.json"

    def __init__(self, garments_dir: str | Path, index_path: Optional[str | Path] = None) -> None:
        self.root = Path(garments_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = Path(index_path) if index_path else self.root / self.INDEX_NAME
        self._records: Dict[str, GarmentRecord] = {}
        self.load()

    # -- persistence -------------------------------------------------------------------
    def load(self) -> Dict[str, GarmentRecord]:
        """Read the index, dropping entries whose files disappeared."""
        self._records = {}
        if not self.index_path.exists():
            return self._records
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Closet index %s is unreadable (%s); starting empty.", self.index_path, exc)
            return self._records
        for item in payload.get("garments", []):
            try:
                record = GarmentRecord.from_storage_dict(item)
            except (TypeError, KeyError, ValueError) as exc:
                logger.warning("Dropping unreadable closet entry %r: %s", item.get("key"), exc)
                continue
            if Path(record.image_path).exists() and Path(record.mask_path).exists():
                self._records[record.key] = record
            else:
                logger.warning("Dropping closet entry %s — files missing.", record.key)
        return self._records

    def save(self) -> Path:
        """Atomically write the index."""
        payload = {
            "version": 1,
            "updated_at": time.time(),
            "count": len(self._records),
            "garments": [record.to_storage_dict() for record in self._records.values()],
        }
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.index_path)
        return self.index_path

    # -- CRUD --------------------------------------------------------------------------
    def all(self) -> List[GarmentRecord]:
        return sorted(self._records.values(), key=lambda r: r.created_at, reverse=True)

    def get(self, key: str) -> GarmentRecord:
        record = self._records.get(key)
        if record is None:
            raise GarmentNotFoundError(f"No garment with key '{key}' in the closet.", details={"key": key})
        return record

    def maybe_get(self, key: str) -> Optional[GarmentRecord]:
        return self._records.get(key)

    def add(self, record: GarmentRecord) -> GarmentRecord:
        self._records[record.key] = record
        self.save()
        return record

    def delete(self, key: str, remove_files: bool = True) -> bool:
        record = self._records.pop(key, None)
        if record is None:
            return False
        if remove_files:
            directory = Path(record.image_path).parent
            if directory.exists() and directory.is_relative_to(self.root):
                shutil.rmtree(directory, ignore_errors=True)
        self.save()
        return True

    def update(self, key: str, **fields: Any) -> GarmentRecord:
        record = self.get(key)
        allowed = {"label", "category", "favourite", "tags", "garment_type"}
        for name, value in fields.items():
            if name in allowed and value is not None:
                setattr(record, name, value)
        if fields.get("category"):
            record.category = str(fields["category"])
        self.save()
        return record

    def clear(self) -> int:
        count = len(self._records)
        for record in list(self._records.values()):
            self.delete(record.key)
        return count

    # -- queries -----------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        records = self.all()
        by_category: Dict[str, int] = {}
        for record in records:
            by_category[record.category] = by_category.get(record.category, 0) + 1
        return {
            "total": len(records),
            "by_category": by_category,
            "groups": {group: [r.key for r in records if r.category in members] for group, members in CATEGORY_GROUPS.items()},
            "favourites": [r.key for r in records if r.favourite],
            "last_added": records[0].to_dict() if records else None,
            "index_path": str(self.index_path),
        }

    def filter(self, category: Optional[str] = None, garment_type: Optional[str] = None, favourite: Optional[bool] = None) -> List[GarmentRecord]:
        records = self.all()
        if category:
            wanted = category.strip().lower()
            records = [r for r in records if r.category == wanted or r.key == category]
        if garment_type:
            records = [r for r in records if r.garment_type == garment_type]
        if favourite is not None:
            records = [r for r in records if r.favourite == favourite]
        return records

    def next_in_cycle(self, current_key: Optional[str], category: Optional[str] = None) -> Optional[GarmentRecord]:
        """Next garment in the closet (used by 'Change Outfit' and the open-palm gesture)."""
        records = self.filter(category=category)
        if not records:
            return None
        if current_key is None:
            return records[0]
        keys = [r.key for r in records]
        try:
            index = (keys.index(current_key) + 1) % len(keys)
        except ValueError:
            index = 0
        return records[index]

    def previous_in_cycle(self, current_key: Optional[str], category: Optional[str] = None) -> Optional[GarmentRecord]:
        records = self.filter(category=category)
        if not records:
            return None
        if current_key is None:
            return records[-1]
        keys = [r.key for r in records]
        try:
            index = (keys.index(current_key) - 1) % len(keys)
        except ValueError:
            index = 0
        return records[index]


class GarmentService:
    """Upload + preprocessing service (used by the API and the CLI scripts)."""

    def __init__(
        self,
        closet: ClosetStore,
        uploads_dir: str | Path = "uploads",
        canvas: int = 768,
        min_resolution: int = 160,
        max_side: int = 1024,
        background_removal: str = "auto",
        enable_zero_shot: bool = True,
        device: str = "auto",
        classifier_head: Optional[str | Path] = None,
    ) -> None:
        self.closet = closet
        self.uploads_dir = Path(uploads_dir)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.canvas = int(canvas)
        self.min_resolution = int(min_resolution)
        self.max_side = int(max_side)
        self.remover = BackgroundRemover(background_removal)
        self.classifier = GarmentClassifier(enable_zero_shot=enable_zero_shot, device=device, trained_head_path=classifier_head)

    # ---------------------------------------------------------------------------------
    # main entry point
    # ---------------------------------------------------------------------------------
    def ingest(
        self,
        data: bytes,
        filename: str = "upload.png",
        category_hint: Optional[str] = None,
        label: Optional[str] = None,
        source: Optional[Dict[str, Any]] = None,
        skip_quality_gate: bool = False,
        tags: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Run the full intake pipeline and return the stored garment record."""
        started = time.perf_counter()
        suffix = Path(filename).suffix.lower()
        if suffix and suffix not in ALLOWED_SUFFIXES:
            raise InvalidImageError(
                f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}.",
                details={"filename": filename},
            )
        try:
            image = load_bytes(data, mode="RGBA")
        except ValueError as exc:
            raise InvalidImageError(str(exc), details={"filename": filename}) from exc

        rgb = to_rgb(image)
        height, width = rgb.shape[:2]
        if min(height, width) < self.min_resolution:
            raise LowResolutionError(
                f"Image is {width}x{height}; at least {self.min_resolution}px on the short side is required.",
                details={"width": width, "height": height, "minimum": self.min_resolution},
            )

        rgb = resize_keep_aspect(rgb, self.max_side)

        # Prefer an embedded alpha channel (already-cut PNGs) over re-segmentation.
        embedded_alpha = image[..., 3] if image.ndim == 3 and image.shape[-1] == 4 else None
        if embedded_alpha is not None and embedded_alpha.size and embedded_alpha.min() < 250 and float((embedded_alpha > 20).mean()) < 0.985:
            alpha = embedded_alpha
            cutout = np.dstack([rgb, alpha])
            logger.info("Using the uploaded image's own alpha channel (already transparent).")
            backend_used = "embedded_alpha"
        else:
            cutout, alpha = self.remover.remove(rgb)
            backend_used = self.remover.active

        alpha = refine_alpha(rgb, alpha)[..., 3]

        quality = garment_quality_report(alpha, rgb, min_resolution=self.min_resolution)
        quality["background_removal"] = backend_used
        if not quality["ok"] and not skip_quality_gate:
            raise UnsupportedGarmentError(details={"quality_report": quality})

        # Normalise the classifier output to a plain dict immediately: the service, the API
        # and the closet store all speak JSON, so there is exactly one representation.
        raw_classification = self.classifier.classify(alpha, rgb)
        classification: Dict[str, Any] = (
            raw_classification.to_dict() if hasattr(raw_classification, "to_dict") else dict(raw_classification)
        )
        if category_hint:
            hint = category_hint.strip().lower()
            if hint and hint != classification.get("category"):
                classification = {
                    "category": hint,
                    "confidence": 1.0,
                    "method": "user_override",
                    "scores": classification.get("scores", {}),
                    "notes": [f"Category set by the user (auto-detected: {classification.get('category')})."],
                }

        canvas_rgb, canvas_alpha = extract_garment_region(rgb, alpha, target_canvas=self.canvas)
        palette = dominant_colours(rgb, alpha)

        key = self._make_key(label or Path(filename).stem, classification.get("category", "garment"))
        directory = self.closet.root / key
        directory.mkdir(parents=True, exist_ok=True)

        image_path = save_image(canvas_rgb, directory / "garment.png")
        mask_path = save_image(canvas_alpha, directory / "mask.png")
        cutout_path = save_image(cutout, directory / "cutout.png")
        preview = _make_preview(cutout)
        preview_path = save_image(preview, directory / "preview.jpg", quality=88)
        (directory / f"original{suffix or '.png'}").write_bytes(data)
        (directory / "meta.json").write_text(json.dumps({
            "key": key, "filename": filename, "quality": quality, "classification": classification,
            "palette": palette, "created_at": time.time(), "source": source or {},
        }, indent=2), encoding="utf-8")

        record = GarmentRecord(
            key=key,
            label=label or _prettify(Path(filename).stem),
            category=str(classification.get("category", "unknown")),
            created_at=time.time(),
            original_name=filename,
            image_path=str(image_path),
            mask_path=str(mask_path),
            cutout_path=str(cutout_path),
            preview_path=str(preview_path),
            width=int(canvas_rgb.shape[1]),
            height=int(canvas_rgb.shape[0]),
            canvas=self.canvas,
            classification=classification if isinstance(classification, dict) else classification.to_dict(),
            quality=quality,
            source=source or {},
            palette=palette,
            tags=list(tags or []),
            garment_type=infer_garment_type(str(classification.get("category", "unknown"))),
        )
        self.closet.add(record)
        elapsed = (time.perf_counter() - started) * 1000.0
        logger.info("Ingested garment '%s' (%s) via %s in %.0f ms", record.label, record.category, backend_used, elapsed)
        payload = record.to_dict(include_data_uri=True)
        payload["ingest_ms"] = round(elapsed, 1)
        payload["warnings"] = quality.get("warnings", [])
        return payload

    def ingest_path(self, path: str | Path, **kwargs: Any) -> Dict[str, Any]:
        path = Path(path)
        return self.ingest(path.read_bytes(), filename=path.name, **kwargs)

    def reclassify(self, key: str, category: str) -> GarmentRecord:
        return self.closet.update(key, category=category)

    def reprocess(self, key: str) -> Dict[str, Any]:
        """Re-run preprocessing on the stored original (e.g. after changing the remover)."""
        record = self.closet.get(key)
        directory = Path(record.image_path).parent
        originals = list(directory.glob("original.*"))
        if not originals:
            raise GarmentNotFoundError(f"Original file for '{key}' is missing; re-upload the garment.")
        self.closet.delete(key, remove_files=False)
        shutil.rmtree(directory, ignore_errors=True)
        return self.ingest_path(originals[0], category_hint=record.category, label=record.label)

    # ---------------------------------------------------------------------------------
    def _make_key(self, label: str, category: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", f"{label}".lower()).strip("-")[:40] or "garment"
        return f"{slug}-{category}-{uuid.uuid4().hex[:6]}"

    def classifier_status(self) -> Dict[str, Any]:
        return self.classifier.status()


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def infer_garment_type(category: str) -> str:
    """Map a category onto an outfit slot."""
    if category in {"bottom"}:
        return "bottom"
    if category in {"shoes"}:
        return "shoes"
    if category in {"accessory"}:
        return "accessory"
    return "top"


def dominant_colours(rgb: np.ndarray, alpha: np.ndarray, k: int = 4) -> List[str]:
    """Extract the dominant colours of the garment as hex strings (used for UI chips)."""
    import cv2

    binary = (np.asarray(alpha) > 96)
    if binary.sum() < 32:
        return []
    pixels = np.asarray(rgb)[..., :3][binary].reshape(-1, 3).astype(np.float32)
    if pixels.shape[0] > 20000:
        step = max(1, pixels.shape[0] // 20000)
        pixels = pixels[::step]
    k = max(1, min(k, len(pixels)))
    try:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
        _compact, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.flatten(), minlength=len(centers))
        order = np.argsort(-counts)
        return [f"#{int(r):02x}{int(g):02x}{int(b):02x}" for r, g, b in centers[order]]
    except Exception as exc:  # pragma: no cover
        logger.debug("k-means palette failed: %s", exc)
        mean = pixels.mean(axis=0)
        return [f"#{int(mean[0]):02x}{int(mean[1]):02x}{int(mean[2]):02x}"]


def _make_preview(cutout: np.ndarray, size: int = 320) -> np.ndarray:
    """Small JPEG-friendly preview on a white background."""
    from PIL import Image

    rgba = np.asarray(cutout)
    if rgba.ndim == 2:
        rgba = np.dstack([np.repeat(rgba[..., None], 3, -1), np.full(rgba.shape, 255, np.uint8)])
    if rgba.shape[-1] == 3:
        rgba = np.dstack([rgba, np.full(rgba.shape[:2], 255, np.uint8)])
    height, width = rgba.shape[:2]
    scale = min(size / max(1, width), size / max(1, height))
    with Image.fromarray(rgba, mode="RGBA") as img:
        resized = img.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS)
        canvas = Image.new("RGB", (max(1, resized.width + 16), max(1, resized.height + 16)), (255, 255, 255))
        canvas.paste(resized, (8, 8), resized)
        return np.asarray(canvas)


def _prettify(name: str) -> str:
    text = re.sub(r"[_\-]+", " ", name).strip()
    text = re.sub(r"\s{2,}", " ", text)
    return text.title()[:60] or "Garment"
