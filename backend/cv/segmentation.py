"""Body / garment segmentation utilities.

Two segmentation problems are solved here:

1. **Body** segmentation (which pixels belong to the person) — from MediaPipe's mask when
   available, otherwise a foreground/background split via GrabCut seeded by the pose box
   or a simple frame-difference heuristic.
2. **Garment** segmentation (isolating a single garment in a product photo) — GrabCut
   with automatic seeding plus alpha-matting style edge refinement, used both as a
   fallback and as the refinement step after a learned background remover such as
   ``rembg`` / ``transparent-background``.

Both are classical CV and always available, which keeps the "no heavy checkpoint
installed" experience fully functional.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from backend.utils.errors import UnsupportedGarmentError
from backend.utils.image_utils import ensure_rgba, mask_bbox_ratio, to_rgb
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------------------
# Learned background removal (optional extras)
# --------------------------------------------------------------------------------------
class BackgroundRemover:
    """Wrap an optional learned matting backend with a GrabCut fallback.

    ``backend`` accepts ``auto``, ``rembg``, ``grabcut`` or ``none``. When ``auto`` the
    class prefers ``rembg`` (if installed) and silently falls back to GrabCut otherwise,
    reporting which one was actually used so the UI can be honest about quality.
    """

    def __init__(self, backend: str = "auto", model_name: str = "u2net") -> None:
        self.requested = backend
        self.model_name = model_name
        self.active = "grabcut"
        self._session = None
        self._init_backend()

    def _init_backend(self) -> None:
        if self.requested == "none":
            self.active = "none"
            return
        if self.requested in {"auto", "rembg"}:
            try:  # pragma: no cover - depends on optional dependency
                from rembg import new_session  # type: ignore

                self._session = new_session(self.model_name)
                self.active = "rembg"
                logger.info("Background removal backend: rembg (%s)", self.model_name)
                return
            except Exception as exc:
                if self.requested == "rembg":
                    logger.warning("rembg requested but unavailable (%s); using GrabCut.", exc)
        self.active = "grabcut"
        logger.info("Background removal backend: GrabCut (classical CV)")

    def remove(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(rgba, alpha_uint8)`` for the input RGB image."""
        rgba = self.grabcut_alpha(image_rgb) if self.active in {"grabcut", "none"} or self._session is None else self._rembg_alpha(image_rgb)
        alpha = rgba[..., 3]
        return rgba, alpha

    def _rembg_alpha(self, image_rgb: np.ndarray) -> np.ndarray:  # pragma: no cover - optional dep
        from rembg import remove  # type: ignore

        out = remove(image_rgb, session=self._session)
        out = np.asarray(out)
        if out.ndim == 2:
            out = np.dstack([np.stack([out] * 3, -1), np.full(out.shape, 255, np.uint8)])
        if out.shape[-1] == 3:
            out = np.dstack([out, np.full(out.shape[:2], 255, np.uint8)])
        return refine_alpha(out[..., :3], out[..., 3])

    # -- classical path ---------------------------------------------------------------
    def grabcut_alpha(self, image_rgb: np.ndarray, iterations: int = 5) -> np.ndarray:
        """Segment the foreground product with GrabCut seeded from the border."""
        image = to_rgb(image_rgb)
        height, width = image.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        border = max(2, int(0.02 * min(height, width)))
        rect = (border, border, width - 2 * border, height - 2 * border)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
        except cv2.error as exc:  # pragma: no cover - tiny images
            logger.warning("GrabCut failed (%s); using centre-box alpha.", exc)
            alpha = np.zeros((height, width), np.uint8)
            alpha[border:height - border, border:width - border] = 255
            return ensure_rgba(np.dstack([image, alpha]))

        binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        if binary.mean() < 0.005 or binary.mean() > 0.98:
            # Degenerate result (e.g. a textureless product shot) — keep a soft centre box.
            binary = np.zeros((height, width), np.uint8)
            binary[border:height - border, border:width - border] = 1

        binary = _keep_largest_component(binary)
        alpha = (binary * 255).astype(np.uint8)
        alpha = morphology_cleanup(alpha)
        rgba = ensure_rgba(np.dstack([image, alpha]))
        return rgba


def _keep_largest_component(binary: np.ndarray) -> np.ndarray:
    """Keep only the largest connected foreground blob."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if count <= 2:
        return binary
    # index 0 is background
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = int(np.argmax(areas)) + 1
    return (labels == keep).astype(np.uint8)


def morphology_cleanup(alpha: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Close pinholes and remove speckles from a binary alpha channel."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned


def refine_alpha(rgb: np.ndarray, alpha: np.ndarray, feather: int = 3) -> np.ndarray:
    """Feather edges and suppress low-alpha noise (alpha-matting style refinement)."""
    alpha = alpha.astype(np.float32)
    if alpha.max() > 1.5:
        alpha /= 255.0
    alpha[alpha < 0.12] = 0.0
    alpha[alpha > 0.92] = 1.0
    if feather > 0:
        blur = feather * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (blur, blur), 0)
    stem = np.clip(alpha, 0.0, 1.0)
    # Un-premultiply the fringe so semi-transparent edges do not darken.
    weight = np.clip(stem, 0.15, 1.0)[..., None]
    corrected = np.clip(rgb.astype(np.float32) / weight, 0, 255)
    return np.dstack([corrected.astype(np.uint8), (stem * 255).astype(np.uint8)])


# --------------------------------------------------------------------------------------
# Body segmentation helpers
# --------------------------------------------------------------------------------------
def body_mask_from_pose(pose_mask: Optional[np.ndarray], frame_shape: Tuple[int, int], fallback_box: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
    """Return a body mask: prefer the MediaPipe mask, else an ellipse over the torso box."""
    height, width = frame_shape[:2]
    if pose_mask is not None:
        mask = np.asarray(pose_mask)
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return ((mask > 127).astype(np.uint8) * 255)
    if fallback_box is None:
        return None
    x0, y0, x1, y1 = fallback_box
    mask = np.zeros((height, width), np.uint8)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    cv2.ellipse(mask, (cx, cy), (max(4, (x1 - x0) // 2), max(4, (y1 - y0) // 2)), 0, 0, 360, 255, -1)
    return mask


def foreground_mask_frame_difference(current: np.ndarray, reference: np.ndarray, threshold: int = 26) -> np.ndarray:
    """Cheap static-camera person mask: absolute difference against a background plate."""
    diff = cv2.absdiff(to_rgb(current), to_rgb(reference))
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(binary, np.ones((7, 7), np.uint8), iterations=1)


def validate_person_mask(mask: Optional[np.ndarray], min_ratio: float = 0.02, max_ratio: float = 0.9) -> Tuple[bool, str]:
    """Sanity-check that a reasoned person mask is plausible for a webcam frame."""
    if mask is None:
        return False, "no segmentation mask available"
    ratio = mask_bbox_ratio(mask)
    if ratio < min_ratio:
        return False, f"person occupies only {ratio * 100:.1f}% of the frame — move closer"
    if ratio > max_ratio:
        return False, f"person fills {ratio * 100:.1f}% of the frame — move back a little"
    return True, "ok"


# --------------------------------------------------------------------------------------
# Garment validation
# --------------------------------------------------------------------------------------
def garment_quality_report(alpha: np.ndarray, rgb: np.ndarray, min_resolution: int = 160) -> Dict[str, Any]:
    """Score a segmented garment and explain why an image may be rejected.

    Guards against the most common bad uploads: full-body model photos, watermarks-only
    images, heavily occluded items and low-resolution thumbnails.
    """
    height, width = rgb.shape[:2]
    report: Dict[str, Any] = {
        "width": int(width),
        "height": int(height),
        "coverage": round(float(mask_bbox_ratio(alpha)), 4),
        "issues": [],
        "warnings": [],
    }
    if min(height, width) < min_resolution:
        report["issues"].append(
            f"Resolution {width}x{height} is below the {min_resolution}px minimum: garment details will be lost."
        )

    x0, y0, x1, y1 = _bbox(alpha)
    box_w, box_h = x1 - x0, y1 - y0
    aspect = box_h / max(1, box_w)
    report["bbox"] = [int(x0), int(y0), int(x1), int(y1)]
    report["aspect_ratio"] = round(float(aspect), 3)
    report["fill_ratio"] = round(box_w * box_h / float(max(1, (mask_bbox_ratio(alpha) * width * height))) / 100.0, 4) if mask_bbox_ratio(alpha) > 0 else 0.0

    if report["coverage"] < 0.04:
        report["issues"].append("Almost nothing was detected — the image may be blank, a logo, or a very small thumbnail.")
    if aspect > 3.2:
        report["issues"].append("Very tall narrow shape detected — this looks like a full-body model photo, not a garment product shot.")
    if aspect < 0.35:
        report["warnings"].append("Very wide shape detected — check that the garment is not a pair of shoes or an accessory crop.")

    # Skin/face heuristic: a product flat-lay has almost no skin pixels.
    skin_ratio = _skin_ratio(rgb)
    report["skin_ratio"] = round(float(skin_ratio), 4)
    if skin_ratio > 0.18:
        report["issues"].append(
            f"About {skin_ratio * 100:.0f}% skin tones detected — this is probably a photo of a person wearing the garment, not the garment itself."
        )

    # Hole ratio: transparent inner regions suggest occlusion or a ring/hanger.
    holes = _hole_ratio(alpha)
    report["hole_ratio"] = round(float(holes), 4)
    if holes > 0.35:
        report["warnings"].append("Large transparent regions inside the garment outline — the item may be partially occluded.")

    report["ok"] = not report["issues"]
    return report


def _bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    from backend.utils.image_utils import bbox_from_mask

    box = bbox_from_mask(mask)
    if box is None:
        height, width = mask.shape[:2]
        return 0, 0, width, height
    return box


def _skin_ratio(rgb: np.ndarray) -> float:
    """Fraction of pixels in a broad YCrCb skin-tone band."""
    ycrcb = cv2.cvtColor(to_rgb(rgb), cv2.COLOR_RGB2YCrCb)
    mask = cv2.inRange(ycrcb, np.array([0, 133, 77], np.uint8), np.array([255, 173, 127], np.uint8))
    return float((mask > 0).mean())


def _hole_ratio(alpha: np.ndarray) -> float:
    binary = (np.asarray(alpha) > 127).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or len(contours) < 2:
        return 0.0
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    area = float(filled.sum())
    if area <= 0:
        return 0.0
    inner = int(binary.sum())
    return float(max(0.0, area - inner) / area)


def require_garment_quality(report: Dict[str, Any]) -> None:
    """Raise :class:`UnsupportedGarmentError` when a report contains blocking issues."""
    if not report.get("ok", True):
        raise UnsupportedGarmentError(details={"quality_report": report})


def segment_garment_from_person(
    rgb: np.ndarray, upper_garment_box: Tuple[int, int, int, int], iterations: int = 4
) -> np.ndarray:
    """Segment an upper garment worn by a person (used to build training pairs)."""
    mask = np.zeros(rgb.shape[:2], np.uint8)
    x0, y0, x1, y1 = upper_garment_box
    rect = (max(0, x0), max(0, y0), max(1, x1 - x0), max(1, y1 - y0))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(to_rgb(rgb), mask, rect, bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    binary = _keep_largest_component(binary)
    return (binary * 255).astype(np.uint8)


def extract_garment_region(rgb: np.ndarray, alpha: np.ndarray, target_canvas: int = 768, pad_ratio: float = 0.06) -> Tuple[np.ndarray, np.ndarray]:
    """Crop the garment to its bbox and letterbox it onto a square canvas.

    Returns ``(rgb_canvas, alpha_canvas)`` — the normalised representation stored in the
    closet and consumed by both the warp pipeline and the diffusion adapter.
    """
    from backend.utils.image_utils import bbox_from_mask, load_image  # noqa: F401 - keeps helpers together

    rgb = to_rgb(rgb)
    box = bbox_from_mask(alpha, pad=int(pad_ratio * max(rgb.shape[:2])))
    if box is None:
        raise UnsupportedGarmentError("Could not find the garment silhouette in the image.")
    x0, y0, x1, y1 = box
    crop_rgb = rgb[y0:y1, x0:x1]
    crop_alpha = np.asarray(alpha)[y0:y1, x0:x1]

    height, width = crop_rgb.shape[:2]
    scale = min(target_canvas / max(1, width), target_canvas / max(1, height))
    new_w, new_h = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    resized_rgb = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    resized_alpha = cv2.resize(crop_alpha, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas_rgb = np.full((target_canvas, target_canvas, 3), 255, np.uint8)
    canvas_alpha = np.zeros((target_canvas, target_canvas), np.uint8)
    off_x, off_y = (target_canvas - new_w) // 2, (target_canvas - new_h) // 2
    canvas_rgb[off_y:off_y + new_h, off_x:off_x + new_w] = resized_rgb
    canvas_alpha[off_y:off_y + new_h, off_x:off_x + new_w] = resized_alpha
    return canvas_rgb, canvas_alpha


def save_alpha_preview(rgb: np.ndarray, alpha: np.ndarray, path: str | Path) -> Path:
    """Write an RGBA PNG preview of a garment cut-out."""
    from backend.utils.image_utils import save_image

    return save_image(np.dstack([to_rgb(rgb), alpha]), path)
