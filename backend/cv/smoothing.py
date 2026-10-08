"""Temporal smoothing for live tracking.

Raw per-frame landmarks jitter, which makes a warped garment shimmy. This module provides:

* :class:`LandmarkSmoother` — exponential moving average over the 33 landmarks with
  adaptive damping (fast motion is trusted more than micro-noise) plus detection dropout
  handling so the garment freezes for a few frames instead of snapping to a default pose.
* :class:`ScalarSmoother` / :class:`FPSCounter` — small helpers used by the live service.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import numpy as np


@dataclass
class SmootherConfig:
    """Tuning knobs for :class:`LandmarkSmoother`."""

    alpha: float = 0.45          # base EMA weight for new observations
    motion_alpha: float = 0.8    # weight used when motion exceeds ``motion_scale``
    motion_scale: float = 12.0   # px of mean landmark displacement deemed "fast motion"
    visibility_floor: float = 0.3
    max_hold_frames: int = 8
    velocity_decay: float = 0.7


class LandmarkSmoother:
    """EMA + velocity prediction smoother for landmark arrays shaped ``(33, 4)``."""

    def __init__(self, config: Optional[SmootherConfig] = None) -> None:
        self.config = config or SmootherConfig()
        self.state: Optional[np.ndarray] = None
        self.velocity: Optional[np.ndarray] = None
        self.miss_count: int = 0
        self.last_valid: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.state = None
        self.velocity = None
        self.miss_count = 0
        self.last_valid = None

    def update(self, landmarks: Optional[np.ndarray], detected: bool) -> Optional[np.ndarray]:
        """Feed one frame of landmarks; returns the smoothed array or ``None`` when lost."""
        if not detected or landmarks is None:
            return self._handle_miss()

        current = np.asarray(landmarks, dtype=np.float32).copy()
        if self.state is None or self.state.shape != current.shape:
            self.state = current
            self.velocity = np.zeros_like(current)
            self.miss_count = 0
            self.last_valid = current
            return current.copy()

        delta = current - self.state
        motion = float(np.mean(np.linalg.norm(delta[:, :2], axis=1)))
        alpha = self.config.motion_alpha if motion > self.config.motion_scale else self.config.alpha

        # Blend position/visibility only where the new observation is trustworthy.
        trustworthy = current[:, 3] >= self.config.visibility_floor
        weight = np.where(trustworthy, alpha, 0.0)[:, None]

        new_state = self.state * (1.0 - weight) + current * weight
        self.velocity = self.velocity * self.config.velocity_decay + delta * (1.0 - self.config.velocity_decay)
        self.state = new_state
        self.miss_count = 0
        self.last_valid = new_state.copy()
        return new_state.copy()

    def _handle_miss(self) -> Optional[np.ndarray]:
        """Extrapolate briefly, then give up so the garment can be hidden."""
        self.miss_count += 1
        if self.state is None:
            return None
        if self.miss_count > self.config.max_hold_frames:
            return None
        self.state = self.state + self.velocity * 0.5
        return self.state.copy()

    @property
    def tracking(self) -> bool:
        return self.state is not None and self.miss_count <= self.config.max_hold_frames

    def status(self) -> Dict[str, float]:
        return {
            "tracking": self.tracking,
            "miss_frames": self.miss_count,
            "mean_velocity_px": float(np.mean(np.linalg.norm(self.velocity[:, :2], axis=1))) if self.velocity is not None else 0.0,
        }


class ScalarSmoother:
    """Single-value EMA used for alpha blending and transform matrices."""

    def __init__(self, alpha: float = 0.35, initial: Optional[float] = None) -> None:
        self.alpha = float(alpha)
        self.value: Optional[float] = initial

    def update(self, value: float) -> float:
        if self.value is None:
            self.value = float(value)
        else:
            self.value = self.value * (1.0 - self.alpha) + float(value) * self.alpha
        return self.value

    def reset(self) -> None:
        self.value = None


class MatrixSmoother:
    """EMA over 2x3 affine matrices, keeping the matrix well-formed.

    Blending affine matrices element-wise is only an approximation, but for the small
    per-frame deltas of a webcam stream it is stable and far cheaper than decomposing;
    the linear part is re-orthonormalised afterwards to avoid shear drift.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = float(alpha)
        self.matrix: Optional[np.ndarray] = None

    def update(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32).reshape(2, 3)
        if self.matrix is None:
            self.matrix = matrix.copy()
            return self.matrix.copy()
        blended = self.matrix * (1.0 - self.alpha) + matrix * self.alpha
        self.matrix = _orthonormalize(blended)
        return self.matrix.copy()

    def reset(self) -> None:
        self.matrix = None


def _orthonormalize(matrix: np.ndarray) -> np.ndarray:
    """Remove shear/scale drift from the linear part of a 2x3 affine matrix."""
    out = np.asarray(matrix, dtype=np.float32).reshape(2, 3).copy()
    linear = out[:, :2]
    try:
        u, _s, vh = np.linalg.svd(linear)
        scale = float(np.sqrt(abs(np.linalg.det(linear)))) or 1.0
        out[:, :2] = (u @ vh) * scale
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate matrix
        pass
    return out


class RingBuffer:
    """Fixed-size history buffer used for frame reuse / caching in live mode."""

    def __init__(self, capacity: int = 8) -> None:
        self.capacity = max(1, capacity)
        self.items: Deque[object] = deque(maxlen=self.capacity)

    def push(self, item: object) -> None:
        self.items.append(item)

    def latest(self) -> Optional[object]:
        return self.items[-1] if self.items else None

    def __len__(self) -> int:
        return len(self.items)

    def clear(self) -> None:
        self.items.clear()


@dataclass
class TemporalStabilizer:
    """Combine several smoothers into one object the live service can hold."""

    config: SmootherConfig = field(default_factory=SmootherConfig)
    landmarks: LandmarkSmoother = field(init=False)
    matrix: MatrixSmoother = field(init=False)

    def __post_init__(self) -> None:
        self.landmarks = LandmarkSmoother(self.config)
        self.matrix = MatrixSmoother(alpha=min(0.6, self.config.alpha))

    def reset(self) -> None:
        self.landmarks.reset()
        self.matrix.reset()

    def status(self) -> Dict[str, float]:
        return self.landmarks.status()
