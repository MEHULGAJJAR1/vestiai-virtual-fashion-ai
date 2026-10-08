"""Gesture recognition from hand landmarks.

The browser performs the actual hand tracking (MediaPipe Tasks Vision) and posts the 21
landmarks here, or gesture classification runs client-side using the mirrored logic in
``frontend/js/gestures.js``. Having the canonical classifier in Python keeps it unit
testable and available to the server-side live pipeline as well.

Gesture vocabulary
------------------
============== =========================================================
Gesture        Action
============== =========================================================
open_palm      next outfit
fist           previous outfit
thumbs_up      capture photo
victory        toggle AI / fast quality mode
pinch          drag / reposition garment vertically
swipe_left     previous outfit (alternate)
swipe_right    next outfit (alternate)
============== =========================================================

The classifier is rule-based on joint angles and is deliberately conservative: it returns
``unknown`` rather than guessing, and ``GestureController`` debounces repeated triggers.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_CHAINS: Dict[str, Tuple[int, int, int]] = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP),
}

GESTURES = ("open_palm", "fist", "thumbs_up", "victory", "pinch", "swipe_left", "swipe_right", "unknown")

#: Gesture -> frontend action id (kept in sync with frontend/js/gestures.js)
GESTURE_ACTIONS: Dict[str, str] = {
    "open_palm": "next_outfit",
    "fist": "previous_outfit",
    "thumbs_up": "capture",
    "victory": "toggle_quality",
    "pinch": "move_garment",
    "swipe_left": "previous_outfit",
    "swipe_right": "next_outfit",
    "unknown": "none",
}


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at ``b`` in degrees for the polyline a-b-c (2-D)."""
    ba = np.asarray(a[:2], np.float32) - np.asarray(b[:2], np.float32)
    bc = np.asarray(c[:2], np.float32) - np.asarray(b[:2], np.float32)
    norm = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if norm < 1e-6:
        return 180.0
    cosine = float(np.clip(np.dot(ba, bc) / norm, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def finger_extended(landmarks: np.ndarray, finger: str, straight_threshold: float = 155.0) -> bool:
    """A finger counts as extended when its PIP joint is nearly straight *and* the tip is
    farther from the wrist than the PIP joint (this handles rotated hands)."""
    mcp, pip, tip = FINGER_CHAINS[finger]
    if _angle(landmarks[mcp], landmarks[pip], landmarks[tip]) < straight_threshold:
        return False
    wrist = landmarks[WRIST]
    return float(np.linalg.norm(landmarks[tip][:2] - wrist[:2])) > float(np.linalg.norm(landmarks[pip][:2] - wrist[:2])) * 1.05


def thumb_extended(landmarks: np.ndarray, straight_threshold: float = 150.0) -> bool:
    """Thumb extension using the CMC-MCP-IP chain plus a lateral offset test."""
    if _angle(landmarks[THUMB_CMC], landmarks[THUMB_MCP], landmarks[THUMB_IP]) < straight_threshold:
        return False
    palm_center = landmarks[[INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]].mean(axis=0)
    return float(np.linalg.norm(landmarks[THUMB_TIP][:2] - palm_center[:2])) > float(
        np.linalg.norm(landmarks[THUMB_IP][:2] - palm_center[:2])
    )


def hand_scale(landmarks: np.ndarray) -> float:
    """Rough hand size in pixels (wrist -> middle MCP), used for scale-invariant tests."""
    return float(np.linalg.norm(landmarks[MIDDLE_MCP][:2] - landmarks[WRIST][:2])) + 1e-6


def classify_hand(landmarks: Sequence[Sequence[float]]) -> Dict[str, object]:
    """Classify a single hand into a gesture name with per-finger detail."""
    array = np.asarray(landmarks, dtype=np.float32).reshape(-1, 3)
    if array.shape[0] < 21:
        return {"gesture": "unknown", "action": "none", "confidence": 0.0, "fingers": {}}

    extended = {name: finger_extended(array, name) for name in FINGER_CHAINS}
    thumb = thumb_extended(array)
    extended_count = sum(extended.values())
    scale = hand_scale(array)

    thumb_tip = array[THUMB_TIP]
    index_tip = array[INDEX_TIP]
    pinch_distance = float(np.linalg.norm(thumb_tip[:2] - index_tip[:2])) / scale

    gesture = "unknown"
    confidence = 0.55

    if pinch_distance < 0.42 and extended["middle"] and extended["ring"] and extended["pinky"] and not extended["index"]:
        gesture, confidence = "pinch", 0.7
    elif thumb and extended_count == 0:
        # Thumb pointing upward = thumbs-up (tip clearly above the wrist in image coords).
        gesture, confidence = ("thumbs_up", 0.8) if thumb_tip[1] < array[WRIST][1] else ("unknown", 0.4)
    elif extended_count == 4 and thumb:
        gesture, confidence = "open_palm", 0.85
    elif extended_count == 0 and not thumb:
        gesture, confidence = "fist", 0.85
    elif extended["index"] and extended["middle"] and not extended["ring"] and not extended["pinky"]:
        gesture, confidence = "victory", 0.8
    elif extended_count == 4 and not thumb:
        gesture, confidence = "open_palm", 0.65

    return {
        "gesture": gesture,
        "action": GESTURE_ACTIONS[gesture],
        "confidence": confidence,
        "fingers": {**{k: bool(v) for k, v in extended.items()}, "thumb": bool(thumb)},
    }


def detect_swipe(history: Deque[float], velocity_threshold: float = 0.35, window: int = 6) -> Optional[str]:
    """Detect a horizontal swipe from a deque of normalised wrist x positions."""
    if len(history) < window:
        return None
    recent = list(history)[-window:]
    delta = recent[-1] - recent[0]
    if abs(delta) < velocity_threshold:
        return None
    return "swipe_right" if delta > 0 else "swipe_left"


@dataclass
class GestureController:
    """Debounced gesture -> action dispatcher.

    ``cooldown_s`` prevents a held pose from firing the action 30 times per second, and
    :meth:`filter_gesture` requires ``confirm_frames`` consecutive identical classifications
    before triggering (kills flicker).
    """

    cooldown_s: float = 1.2
    confirm_frames: int = 3
    swipe_velocity: float = 0.35
    _history: Deque[float] = field(default_factory=lambda: deque(maxlen=24))
    _candidates: Deque[str] = field(default_factory=lambda: deque(maxlen=8))
    _last_action: str = "none"
    _last_trigger: float = 0.0
    last_gesture: str = "unknown"
    enabled: bool = True

    def reset(self) -> None:
        self._history.clear()
        self._candidates.clear()
        self._last_action = "none"
        self._last_trigger = 0.0
        self.last_gesture = "unknown"

    def update(self, hand_landmarks: Optional[Sequence[Sequence[float]]], now: Optional[float] = None) -> Dict[str, object]:
        """Feed one frame of hand landmarks (or ``None``) and get the debounced action."""
        now = time.time() if now is None else now
        if not self.enabled:
            return {"gesture": "unknown", "action": "none", "triggered": False, "enabled": False}

        if hand_landmarks is None:
            self._candidates.clear()
            self._history.clear()
            return {"gesture": "unknown", "action": "none", "triggered": False, "enabled": True}

        array = np.asarray(hand_landmarks, dtype=np.float32).reshape(-1, 3)
        result = classify_hand(array)
        gesture = str(result["gesture"])

        # Swipe takes priority when it is detected with a confident pose.
        self._history.append(float(array[WRIST][0]))
        swipe = detect_swipe(self._history, velocity_threshold=self.swipe_velocity)
        if swipe and gesture in {"open_palm", "unknown"}:
            gesture = swipe
            result = {"gesture": swipe, "action": GESTURE_ACTIONS[swipe], "confidence": 0.6, "fingers": result.get("fingers", {})}

        self._candidates.append(gesture)
        triggered = False
        action = "none"
        if len(self._candidates) >= self.confirm_frames and len(set(list(self._candidates)[-self.confirm_frames:])) == 1:
            if gesture not in {"unknown", "none"} and (now - self._last_trigger) > self.cooldown_s:
                action = GESTURE_ACTIONS.get(gesture, "none")
                self._last_trigger = now
                triggered = action != "none"

        self.last_action = action if triggered else self._last_action
        self.last_gesture = gesture
        return {
            "gesture": gesture,
            "action": GESTURE_ACTIONS.get(gesture, "none"),
            "triggered": triggered,
            "executed_action": self.last_action,
            "confidence": float(result.get("confidence", 0.0)),
            "fingers": result.get("fingers", {}),
            "enabled": True,
        }

    def status(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "last_gesture": self.last_gesture,
            "last_action": self.last_action,
            "supported": list(GESTURES),
        }


def landmarks_to_normalized(landmarks: Sequence[Sequence[float]], width: int, height: int) -> List[List[float]]:
    """Convert pixel hand landmarks to normalised ``[0, 1]`` coordinates for the UI overlay."""
    out: List[List[float]] = []
    for point in landmarks:
        x, y = point[0], point[1]
        z = point[2] if len(point) > 2 else 0.0
        out.append([round(float(x) / max(1, width), 5), round(float(y) / max(1, height), 5), round(float(z), 5)])
    return out


def require_hand_tracking() -> None:
    """Fetch the MediaPipe hands solution or raise a friendly error."""
    try:
        import mediapipe as mp

        return mp.solutions.hands
    except Exception as exc:
        from backend.utils.errors import DependencyMissingError

        raise DependencyMissingError(
            "Server-side hand tracking needs MediaPipe. The browser does hand tracking "
            "client-side via MediaPipe Tasks Vision, so Gesture Mode works without this.",
            details={"install": "pip install mediapipe", "reason": str(exc)},
        ) from exc
