"""Training metrics tracking: TensorBoard, Weights & Biases and always-on JSON/CSV.

Every run writes machine-readable metrics (``metrics.jsonl``) plus a CSV, so the dashboard can
plot curves even when TensorBoard is not installed. TensorBoard and W&B are used when
available and silently skipped (with a clear log line) when they are not.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """A single logged step/epoch."""

    step: int
    epoch: float
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"step": self.step, "epoch": self.epoch, "name": self.name, "value": self.value, "timestamp": self.timestamp}


class Tracker:
    """Fan-out metrics sink."""

    def __init__(
        self,
        log_dir: str | Path,
        backend: str = "tensorboard",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        project: str = "vestiai-vton",
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.backend = (backend or "tensorboard").lower()
        self.run_name = run_name or time.strftime("%Y%m%d-%H%M%S")
        self.project = project
        self.points: List[MetricPoint] = []
        self._tb = None
        self._wandb = None
        self._csv_path = self.log_dir / "metrics.csv"
        self._jsonl_path = self.log_dir / "metrics.jsonl"
        self._csv_initialised = False
        self._init_backends(config or {})

    # ---------------------------------------------------------------------------------
    def _init_backends(self, config: Dict[str, Any]) -> None:
        if self.backend in {"tensorboard", "both"}:
            try:  # pragma: no cover - optional dependency
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
                logger.info("TensorBoard logging -> %s", self.log_dir / "tensorboard")
            except Exception as exc:
                logger.info("TensorBoard unavailable (%s); JSON/CSV metrics still recorded.", exc)
        if self.backend in {"wandb", "both"}:
            try:  # pragma: no cover - optional dependency / requires login
                import wandb

                self._wandb = wandb.init(project=self.project, name=self.run_name, config=config, reinit=True)
                logger.info("Weights & Biases run: %s", getattr(self._wandb, "url", "started"))
            except Exception as exc:
                logger.info("Weights & Biases unavailable (%s); continuing without it.", exc)

    # ---------------------------------------------------------------------------------
    def log(self, name: str, value: float, step: int = 0, epoch: float = 0.0, log_tb: bool = True) -> None:
        """Record one scalar."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        point = MetricPoint(step=int(step), epoch=float(epoch), name=str(name), value=value)
        self.points.append(point)
        if self._tb is not None and log_tb:
            self._tb.add_scalar(name, value, int(step))
        if self._wandb is not None:
            self._wandb.log({name: value, "step": int(step), "epoch": float(epoch)})
        self._append_jsonl(point)
        self._append_csv(point)

    def log_many(self, values: Dict[str, float], step: int = 0, epoch: float = 0.0, log_tb: bool = True) -> None:
        for name, value in values.items():
            if value is None:
                continue
            self.log(name, value, step=step, epoch=epoch, log_tb=log_tb)

    def log_image(self, name: str, image, step: int = 0) -> None:
        """Log a sample image (TensorBoard/W&B only — never blocks the run)."""
        try:
            if self._tb is not None:
                self._tb.add_image(name, image, int(step), dataformats="HWC")
            if self._wandb is not None:
                import wandb

                self._wandb.log({name: wandb.Image(image)}, step=int(step))
        except Exception as exc:  # pragma: no cover
            logger.debug("image logging skipped: %s", exc)

    def log_text(self, name: str, text: str, step: int = 0) -> None:
        try:
            if self._tb is not None:
                self._tb.add_text(name, text, int(step))
        except Exception:  # pragma: no cover
            pass

    def log_config(self, config: Dict[str, Any]) -> None:
        try:
            if self._tb is not None:
                self._tb.add_text("config", "```json\n" + json.dumps(config, indent=2, default=str) + "\n```", 0)
        except Exception:  # pragma: no cover
            pass

    # ---------------------------------------------------------------------------------
    def _append_jsonl(self, point: MetricPoint) -> None:
        try:
            with self._jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(point.to_dict()) + "\n")
        except OSError:  # pragma: no cover
            pass

    def _append_csv(self, point: MetricPoint) -> None:
        try:
            exists = self._csv_path.exists()
            with self._csv_path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["step", "epoch", "name", "value", "timestamp"])
                if not exists and not self._csv_initialised:
                    writer.writeheader()
                    self._csv_initialised = True
                writer.writerow(point.to_dict())
        except OSError:  # pragma: no cover
            pass

    # ---------------------------------------------------------------------------------
    def curves(self) -> Dict[str, List[Dict[str, float]]]:
        """Group recorded points into per-metric series for the dashboard charts."""
        series: Dict[str, List[Dict[str, float]]] = {}
        for point in self.points:
            series.setdefault(point.name, []).append({"step": point.step, "epoch": point.epoch, "value": point.value})
        return series

    def latest(self) -> Dict[str, float]:
        """Most recent value per metric."""
        out: Dict[str, float] = {}
        for point in self.points:
            out[point.name] = point.value
        return out

    def close(self) -> None:
        try:
            if self._tb is not None:
                self._tb.flush()
                self._tb.close()
        except Exception:  # pragma: no cover
            pass
        try:
            if self._wandb is not None:
                self._wandb.finish()
        except Exception:  # pragma: no cover
            pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "log_dir": str(self.log_dir),
            "tensorboard": bool(self._tb is not None),
            "wandb": bool(self._wandb is not None),
            "points": len(self.points),
            "latest": self.latest(),
        }


def load_history(metrics_path: str | Path) -> Dict[str, List[Dict[str, float]]]:
    """Read ``metrics.jsonl`` back into per-metric series (used by the dashboard after restart)."""
    path = Path(metrics_path)
    series: Dict[str, List[Dict[str, float]]] = {}
    if not path.exists():
        return series
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            series.setdefault(str(record.get("name", "unknown")), []).append({
                "step": float(record.get("step", 0)), "epoch": float(record.get("epoch", 0.0)),
                "value": float(record.get("value", 0.0)),
            })
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not read %s: %s", path, exc)
    return series
