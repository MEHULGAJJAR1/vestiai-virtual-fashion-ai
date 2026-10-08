"""Image I/O and geometry helpers shared by the CV, AI and service layers.

All functions accept and return NumPy arrays in **RGB** order unless the name says
otherwise (``*_bgr`` helpers exist for OpenCV hand-offs). Alpha channels are handled
explicitly: ``to_rgb`` drops alpha, ``ensure_rgba`` guarantees one.
"""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

ArrayLike = np.ndarray
BBox = Tuple[int, int, int, int]  # x0, y0, x1, y1

ALLOWED_UPLOAD_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


# --------------------------------------------------------------------------------------
# Loading / saving
# --------------------------------------------------------------------------------------
def load_image(path: str | Path, mode: str = "RGB") -> np.ndarray:
    """Load an image from disk, honouring EXIF orientation."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if mode != "A":
            img = img.convert(mode)
        return np.asarray(img)


def load_bytes(data: bytes, mode: str = "RGB") -> np.ndarray:
    """Decode raw bytes into an array. Raises ``ValueError`` on invalid data."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img = ImageOps.exif_transpose(img)
            if mode != "A":
                img = img.convert(mode)
            return np.asarray(img)
    except Exception as exc:  # PIL raises a zoo of exceptions
        raise ValueError(f"Could not decode image bytes: {exc}") from exc


def save_image(image: np.ndarray, path: str | Path, quality: int = 95) -> Path:
    """Save an RGB/RGBA/L array to disk, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8) if array.max() <= 1.0 else array.astype(np.uint8)
    with Image.fromarray(array) as img:
        fmt = "PNG" if path.suffix.lower() == ".png" else "JPEG"
        if fmt == "JPEG" and img.mode == "RGBA":
            img = img.convert("RGB")
        img.save(path, format=fmt, quality=quality)
    return path


def encode_base64(image: np.ndarray, fmt: str = "PNG") -> str:
    """Encode an array as a base64 data-URI friendly string (no prefix)."""
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    with Image.fromarray(array) as img:
        buffer = io.BytesIO()
        img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_base64(payload: str) -> np.ndarray:
    """Decode a base64 string or data-URI into an RGB array."""
    if "," in payload[:64] and payload.strip().startswith("data:"):
        payload = payload.split(",", 1)[1]
    return load_bytes(base64.b64decode(payload))


def to_data_uri(image: np.ndarray, fmt: str = "PNG") -> str:
    """Return a ``data:image/...;base64,...`` string usable directly in the browser."""
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encode_base64(image, fmt)}"


# --------------------------------------------------------------------------------------
# Colour / channel helpers
# --------------------------------------------------------------------------------------
def to_rgb(image: np.ndarray) -> np.ndarray:
    """Drop alpha if present, guaranteeing an HxWx3 uint8 RGB array."""
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 4:
        alpha = array[..., 3:4].astype(np.float32) / 255.0
        rgb = array[..., :3].astype(np.float32)
        # Composite over white so downstream encoders never see black halos.
        array = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return array[..., :3]


def ensure_rgba(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] == 4:
        return array
    rgb = to_rgb(array)
    alpha = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=-1)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(image)[..., ::-1])


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return bgr_to_rgb(image)


def to_float01(image: np.ndarray) -> np.ndarray:
    return np.asarray(image).astype(np.float32) / 255.0


def to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array
    if array.max(initial=0.0) <= 1.0001:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------
def resize_keep_aspect(image: np.ndarray, max_side: int, min_side: int = 0) -> np.ndarray:
    """Resize so the longest side equals ``max_side`` (never upscales beyond min_side)."""
    height, width = image.shape[:2]
    longest = max(height, width)
    shortest = min(height, width)
    scale = 1.0
    if longest > max_side:
        scale = max_side / float(longest)
    if min_side and shortest * scale < min_side:
        scale = max(scale, min_side / float(shortest))
    if abs(scale - 1.0) < 1e-6:
        return image
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    mode = "RGBA" if image.ndim == 3 and image.shape[-1] == 4 else "RGB"
    with Image.fromarray(image, mode=mode) as img:
        resized = img.resize(new_size, Image.LANCZOS)
        return np.asarray(resized)


def letterbox(image: np.ndarray, size: Tuple[int, int], fill: int | Sequence[int] = 0) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize+pad to ``size=(w, h)`` keeping aspect ratio.

    Returns ``(canvas, scale, (pad_x, pad_y))`` so coordinates can be mapped back.
    """
    target_w, target_h = size
    height, width = image.shape[:2]
    scale = min(target_w / width, target_h / height)
    new_w, new_h = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    mode = "RGBA" if image.ndim == 3 and image.shape[-1] == 4 else "RGB"
    with Image.fromarray(image, mode=mode) as img:
        resized = np.asarray(img.resize((new_w, new_h), Image.LANCZOS))
    canvas_shape = (target_h, target_w) + ((4,) if mode == "RGBA" else (3,))
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    if isinstance(fill, (int, float)):
        canvas[...] = int(fill)
    else:
        canvas[...] = np.asarray(fill, dtype=np.uint8)
    pad_x, pad_y = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


def center_crop(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    height, width = image.shape[:2]
    top = max(0, (height - target_h) // 2)
    left = max(0, (width - target_w) // 2)
    cropped = image[top:top + target_h, left:left + target_w]
    if cropped.shape[0] != target_h or cropped.shape[1] != target_w:
        mode = "RGBA" if image.ndim == 3 and image.shape[-1] == 4 else "RGB"
        canvas = np.zeros((target_h, target_w) + image.shape[2:], dtype=image.dtype)
        canvas[: cropped.shape[0], : cropped.shape[1]] = cropped
        return canvas
    return cropped


def bbox_from_mask(mask: np.ndarray, threshold: float = 0.5, pad: int = 0) -> Optional[BBox]:
    """Return the tight bounding box of a binary mask, or ``None`` when empty."""
    binary = np.asarray(mask) > threshold * (255 if np.asarray(mask).dtype == np.uint8 else 1)
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if not rows.any() or not cols.any():
        return None
    y0, y1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    if pad:
        height, width = binary.shape[:2]
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(width, x1 + pad), min(height, y1 + pad)
    return x0, y0, x1, y1


def crop_to_bbox(image: np.ndarray, bbox: BBox, pad: int = 0) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    height, width = image.shape[:2]
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(width, x1 + pad), min(height, y1 + pad)
    return image[y0:y1, x0:x1]


def apply_alpha_mask(rgb: np.ndarray, mask: np.ndarray, feather: int = 2) -> np.ndarray:
    """Return an RGBA image where ``mask`` (0..1 float or 0..255 uint8) becomes alpha."""
    import cv2  # local import keeps this module importable without OpenCV

    rgb = to_rgb(rgb)
    alpha = mask.astype(np.float32)
    if alpha.max() > 1.5:
        alpha = alpha / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)
    if feather > 0:
        ksize = int(feather) * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (ksize, ksize), 0)
    return np.dstack([rgb, (alpha * 255.0).astype(np.uint8)])


def mask_bbox_ratio(mask: np.ndarray) -> float:
    """Fraction of the frame covered by the mask — used for sanity checks."""
    binary = np.asarray(mask) > 0
    return float(binary.mean()) if binary.size else 0.0


def rotate_scale_landmarks(points: np.ndarray, angle_deg: float, scale: float, center: Sequence[float]) -> np.ndarray:
    """Rotate + scale 2-D points around ``center`` (used by the warp pipeline & tests)."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    theta = math.radians(angle_deg)
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float32)
    centered = (points - np.asarray(center, dtype=np.float32)) * scale
    return (centered @ rotation.T) + np.asarray(center, dtype=np.float32)


def ensure_size(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Hard-resize (w, h) ignoring aspect ratio — used right before model input."""
    with Image.fromarray(_as_pil_mode(image)) as img:
        return np.asarray(img.resize(size, Image.BICUBIC))


def _as_pil_mode(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = to_uint8(array)
    return array


def stack_grid(images: Iterable[np.ndarray], cols: int = 3, pad: int = 8, background: int = 24) -> np.ndarray:
    """Tile images into a single grid — used for validation sample sheets."""
    items = [to_rgb(img) if img.ndim == 3 and img.shape[-1] in (3, 4) else to_rgb(img) for img in images]
    items = [np.asarray(img) for img in items]
    if not items:
        return np.full((64, 64, 3), background, dtype=np.uint8)
    cols = max(1, min(cols, len(items)))
    rows = math.ceil(len(items) / cols)
    cell_h = max(img.shape[0] for img in items)
    cell_w = max(img.shape[1] for img in items)
    grid = np.full((rows * cell_h + pad * (rows + 1), cols * cell_w + pad * (cols + 1), 3), background, dtype=np.uint8)
    for index, img in enumerate(items):
        row, col = divmod(index, cols)
        y = pad + row * (cell_h + pad)
        x = pad + col * (cell_w + pad)
        grid[y:y + img.shape[0], x:x + img.shape[1]] = img
    return grid
