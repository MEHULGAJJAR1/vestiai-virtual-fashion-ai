"""Checkpoint management: atomic saves, best/latest/epoch layout, resume metadata.

Layout created under ``checkpoints/``::

    checkpoints/
      best_model/          <- exported ControlNet + model card (used by inference)
      latest_model/        <- mirror of the most recent export (used by the web app)
      epochs/
        epoch_0002/
          adapter/         <- ControlNet weights
          training_state.pt<- optimizer / scheduler / scaler / RNG state (resume)
          metrics.json
        epoch_0004/ ...
      training_state_latest.pt

``best_model`` is only overwritten when the validation metric improves, so a crashed or
diverging run can never destroy your good checkpoint. All writes are atomic
(``tmp`` + ``os.replace``) so an interrupted save cannot leave a half-written file.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.models.vton_controlnet import MODEL_CARD_NAME, ModelCard, adapter_is_valid, describe_checkpoint
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

TRAINING_STATE_NAME = "training_state.pt"
METRICS_NAME = "metrics.json"


@dataclass
class CheckpointRecord:
    """One checkpoint on disk."""

    name: str
    path: str
    kind: str                 # best | latest | epoch
    epoch: Optional[int] = None
    step: Optional[int] = None
    metric: Optional[float] = None
    metric_name: str = "val_loss"
    created_at: float = 0.0
    size_mb: float = 0.0
    valid: bool = False
    has_optimizer_state: bool = False
    model_card: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "path": self.path, "kind": self.kind, "epoch": self.epoch,
            "step": self.step, "metric": self.metric, "metric_name": self.metric_name,
            "created_at": self.created_at,
            "created_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created_at)) if self.created_at else None,
            "size_mb": self.size_mb, "valid": self.valid, "has_optimizer_state": self.has_optimizer_state,
            "model_card": self.model_card,
        }


class CheckpointManager:
    """Owns the ``checkpoints/`` tree."""

    def __init__(self, root: str | Path = "checkpoints") -> None:
        self.root = Path(root)
        self.epochs_dir = self.root / "epochs"
        self.best_dir = self.root / "best_model"
        self.latest_dir = self.root / "latest_model"
        for path in (self.root, self.epochs_dir, self.best_dir, self.latest_dir):
            path.mkdir(parents=True, exist_ok=True)

    # -- paths -------------------------------------------------------------------------
    def epoch_dir(self, epoch: int) -> Path:
        return self.epochs_dir / f"epoch_{epoch:04d}"

    def training_state_path(self, epoch: Optional[int] = None) -> Path:
        if epoch is None:
            return self.root / "training_state_latest.pt"
        return self.epoch_dir(epoch) / TRAINING_STATE_NAME

    # -- saving ------------------------------------------------------------------------
    def save_training_state(self, state: Dict[str, Any], epoch: int) -> Path:
        """Atomically persist a resumable training state (optimizer/scheduler/RNG)."""
        import torch

        target = self.training_state_path(epoch)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, target)
        # mirror as "latest" for easy resume
        latest = self.training_state_path(None)
        tmp_latest = latest.with_suffix(".pt.tmp")
        torch.save(state, tmp_latest)
        os.replace(tmp_latest, latest)
        logger.info("Saved training state -> %s (%.1f MB)", target, target.stat().st_size / 1024 ** 2)
        return target

    def load_training_state(self, path: Optional[str | Path] = None):
        """Load a resumable training state, or ``None`` when absent."""
        import torch

        path = Path(path) if path else self.training_state_path(None)
        if not path.exists():
            return None
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            logger.error("Could not load training state %s: %s", path, exc)
            return None

    def save_epoch_metrics(self, epoch: int, metrics: Dict[str, Any]) -> Path:
        directory = self.epoch_dir(epoch)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / METRICS_NAME
        _atomic_json(path, metrics)
        return path

    def export_model(
        self,
        adapter_directory: str | Path,
        target: str | Path,
        mirror_latest: bool = True,
    ) -> Path:
        """Copy an exported adapter directory into ``best_model`` (atomic swap)."""
        source = Path(adapter_directory)
        target = Path(target)
        if not source.exists():
            raise FileNotFoundError(f"Adapter directory {source} does not exist.")
        staging = target.parent / f".{target.name}.staging"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(source, staging)
        if target.exists():
            shutil.rmtree(target)
        os.replace(staging, target)
        logger.info("Exported model -> %s", target)
        if mirror_latest and target != self.latest_dir:
            self.export_model(target, self.latest_dir, mirror_latest=False)
        return target

    # -- reading -----------------------------------------------------------------------
    def list_checkpoints(self) -> List[CheckpointRecord]:
        """Enumerate all checkpoints, newest first, with validity + metrics."""
        records: List[CheckpointRecord] = []
        for name, path, kind in (
            ("best_model", self.best_dir, "best"),
            ("latest_model", self.latest_dir, "latest"),
        ):
            if path.exists():
                records.append(self._record(name, path, kind))
        if self.epochs_dir.exists():
            for child in sorted(self.epochs_dir.iterdir()):
                if not child.is_dir():
                    continue
                epoch = _epoch_from_name(child.name)
                adapter = child / "adapter" if (child / "adapter").exists() else child
                records.append(self._record(child.name, adapter, "epoch", epoch=epoch, state_dir=child))
        records.sort(key=lambda r: (r.created_at or 0), reverse=True)
        return records

    def _record(self, name: str, path: Path, kind: str, epoch: Optional[int] = None, state_dir: Optional[Path] = None) -> CheckpointRecord:
        info = describe_checkpoint(path)
        card = info.get("model_card") or {}
        metrics = self.read_metrics(state_dir or path)
        return CheckpointRecord(
            name=name,
            path=str(path),
            kind=kind,
            epoch=epoch if epoch is not None else _first_int(card.get("epochs"), metrics.get("epoch")),
            step=_first_int(card.get("global_step"), metrics.get("global_step")),
            metric=_first_float(metrics.get("val_loss"), card.get("best_metric")),
            metric_name=str(metrics.get("metric_name", card.get("metric_name", "val_loss"))),
            created_at=_newest_mtime(path),
            size_mb=float(info.get("size_mb") or 0.0),
            valid=bool(info.get("valid")),
            has_optimizer_state=(state_dir / TRAINING_STATE_NAME).exists() if state_dir else (path / TRAINING_STATE_NAME).exists(),
            model_card=card or None,
        )

    def read_metrics(self, directory: str | Path) -> Dict[str, Any]:
        path = Path(directory) / METRICS_NAME
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover
            return {}

    def read_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Aggregate per-epoch metrics from all epoch folders into training curves."""
        history: List[Dict[str, Any]] = []
        if self.epochs_dir.exists():
            for child in sorted(self.epochs_dir.iterdir()):
                metrics = self.read_metrics(child)
                if metrics:
                    metrics.setdefault("epoch", _epoch_from_name(child.name))
                    history.append(metrics)
        history.sort(key=lambda m: (m.get("epoch") or 0))
        best = self.read_metrics(self.best_dir)
        return {"epochs": history, "best": best, "count": len(history)}

    def prune_epochs(self, keep_top_k: int = 3, metric_name: str = "val_loss", higher_is_better: bool = False) -> List[str]:
        """Delete older epoch folders, keeping the ``keep_top_k`` best."""
        records = [r for r in self.list_checkpoints() if r.kind == "epoch" and r.metric is not None]
        if len(records) <= keep_top_k:
            return []
        ordered = sorted(records, key=lambda r: (r.metric if r.metric is not None else float("inf")), reverse=higher_is_better)
        keep = {r.name for r in ordered[:keep_top_k]}
        removed: List[str] = []
        for record in records:
            if record.name in keep:
                continue
            directory = self.epochs_dir / record.name
            if directory.exists():
                shutil.rmtree(directory, ignore_errors=True)
                removed.append(record.name)
        if removed:
            logger.info("Pruned epoch checkpoints: %s", ", ".join(removed))
        return removed

    def resolve_for_inference(self, preferred: str = "auto") -> Optional[Path]:
        """Pick the checkpoint the web app should serve.

        ``auto`` prefers ``best_model`` → ``latest_model`` → newest valid epoch folder.
        """
        if preferred and preferred not in {"auto", "best", "latest"}:
            candidate = Path(preferred)
            return candidate if adapter_is_valid(candidate) else None
        order = {
            "best": [self.best_dir, self.latest_dir],
            "latest": [self.latest_dir, self.best_dir],
            "auto": [self.best_dir, self.latest_dir],
        }[preferred if preferred in {"best", "latest"} else "auto"]
        for directory in order:
            if adapter_is_valid(directory):
                return directory
        for record in self.list_checkpoints():
            if record.kind == "epoch" and record.valid:
                return Path(record.path)
        return None

    def status(self) -> Dict[str, Any]:
        """Compact status object for the API/UI."""
        records = self.list_checkpoints()
        best = self.resolve_for_inference("auto")
        return {
            "root": str(self.root),
            "count": len(records),
            "best": next((r.to_dict() for r in records if r.kind == "best"), None),
            "latest": next((r.to_dict() for r in records if r.kind == "latest"), None),
            "epochs": [r.to_dict() for r in records if r.kind == "epoch"],
            "inference_path": str(best) if best else None,
            "ready_for_inference": best is not None,
            "model_card": (ModelCard.load(best).__dict__ if best and ModelCard.load(best) else None),
        }


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, path)


def _epoch_from_name(name: str) -> Optional[int]:
    try:
        return int(name.split("_")[-1])
    except (ValueError, IndexError):
        return None


def _newest_mtime(path: Path) -> float:
    if not path.exists():
        return 0.0
    try:
        if path.is_file():
            return path.stat().st_mtime
        times = [p.stat().st_mtime for p in path.rglob("*") if p.is_file()]
        return max(times) if times else path.stat().st_mtime
    except OSError:  # pragma: no cover
        return 0.0


def _first_int(*values) -> Optional[int]:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_float(*values) -> Optional[float]:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def model_card_path(directory: str | Path) -> Path:
    return Path(directory) / MODEL_CARD_NAME
