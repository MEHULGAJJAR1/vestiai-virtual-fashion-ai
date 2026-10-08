"""Small timing/throughput helpers used for latency reporting and FPS counters."""

from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from typing import Deque, Iterator, Optional


class Timer:
    """Context manager that measures wall-clock duration in milliseconds.

    >>> with Timer() as t:
    ...     pass
    >>> t.ms >= 0
    True
    """

    def __init__(self) -> None:
        self.started_at: float = 0.0
        self.finished_at: float = 0.0
        self.ms: float = 0.0

    def __enter__(self) -> "Timer":
        self.started_at = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> None:
        self.finished_at = time.perf_counter()
        self.ms = (self.finished_at - self.started_at) * 1000.0


class FPSMeter:
    """Rolling FPS estimate over a fixed window (used by the live pipeline)."""

    def __init__(self, window: int = 30) -> None:
        self._times: Deque[float] = deque(maxlen=max(2, window))
        self._last: Optional[float] = None

    def tick(self) -> float:
        """Register a frame and return the current FPS."""
        now = time.perf_counter()
        if self._last is not None:
            self._times.append(now - self._last)
        self._last = now
        return self.fps

    @property
    def fps(self) -> float:
        if not self._times:
            return 0.0
        mean = sum(self._times) / len(self._times)
        return round(1.0 / mean, 1) if mean > 1e-6 else 0.0

    @property
    def latency_ms(self) -> float:
        if not self._times:
            return 0.0
        return round(1000.0 * sum(self._times) / len(self._times), 2)

    def reset(self) -> None:
        self._times.clear()
        self._last = None


@contextmanager
def timed(label: str = "block") -> Iterator[dict]:
    """Yield a dict that receives the elapsed milliseconds::

    >>> with timed("warp") as info:
    ...     pass
    >>> "ms" in info
    True
    """
    info: dict = {"label": label, "ms": 0.0}
    start = time.perf_counter()
    try:
        yield info
    finally:
        info["ms"] = round((time.perf_counter() - start) * 1000.0, 3)
