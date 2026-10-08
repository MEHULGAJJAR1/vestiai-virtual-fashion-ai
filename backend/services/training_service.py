"""Training job manager: start / monitor / stop runs from the web dashboard or the API.

Design
------
* Training runs in a **subprocess** (``python -m scripts.train_vton --config ...``) rather than
  in the API process. That keeps the web server responsive, guarantees CUDA memory is released
  when the run ends, and means a crashed training run can never take the API down with it.
* A single active job is allowed at a time (``GPU_JOB`` semantics). Concurrent requests get a
  clear 409 instead of two processes fighting over VRAM.
* Live status is read from the status JSON the trainer writes (``logs/training_status.json``)
  plus the metrics stream (``logs/**/metrics.jsonl``), so the dashboard shows real numbers with
  sub-second latency and the numbers survive a server restart.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.models.checkpointing import CheckpointManager
from backend.training.evaluate import load_evaluations, plot_training_curves
from backend.training.tracker import load_history
from backend.training.trainer import MODE_PRESETS, TrainingConfig
from backend.training.validate import load_validation_summary
from backend.utils.device import detect_devices, gpu_utilization, vram_warning
from backend.utils.errors import TrainingBusyError, TrainingError
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingJob:
    """A tracked training process."""

    job_id: str
    mode: str
    config: Dict[str, Any]
    started_at: float
    pid: Optional[int] = None
    process: Optional[subprocess.Popen] = None
    status_path: str = ""
    log_path: str = ""
    state: str = "starting"
    stopped_at: Optional[float] = None
    exit_code: Optional[int] = None

    def to_dict(self, include_process: bool = False) -> Dict[str, Any]:
        payload = {
            "job_id": self.job_id,
            "mode": self.mode,
            "state": self.state,
            "started_at": self.started_at,
            "started_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started_at)),
            "finished_at": self.stopped_at,
            "exit_code": self.exit_code,
            "pid": self.pid,
            "config": self.config,
            "status_path": self.status_path,
            "log_path": self.log_path,
            "duration_s": round((self.stopped_at or time.time()) - self.started_at, 1),
        }
        if include_process:
            payload["process"] = bool(self.process)
        return payload

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None


class TrainingService:
    """Owns the training subprocess and exposes dashboard data."""

    def __init__(self, settings, checkpoint_manager: Optional[CheckpointManager] = None, project_root: Optional[str | Path] = None) -> None:
        self.settings = settings
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2])
        self.checkpoints = checkpoint_manager or CheckpointManager(settings.checkpoints_dir)
        self.logs_dir = Path(settings.log_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.logs_dir / "training_status.json"
        self.jobs: Dict[str, TrainingJob] = {}
        self.active_job_id: Optional[str] = None
        self._lock = threading.Lock()

    # =================================================================================
    # job control
    # =================================================================================
    def start(
        self,
        mode: str = "QUICK_DEMO",
        config_overrides: Optional[Dict[str, Any]] = None,
        dataset_root: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Launch a training run as a subprocess."""
        with self._lock:
            if self.active_job_id and self.jobs.get(self.active_job_id, TrainingJob("", "", {}, 0)).alive:
                raise TrainingBusyError(
                    "A training run is already active.",
                    details={"job_id": self.active_job_id, "status": self.get_active_status()},
                )

            mode_key = (mode or self.settings.training_mode or "QUICK_DEMO").upper()
            if mode_key not in MODE_PRESETS:
                raise TrainingError(f"Unknown training mode '{mode}'. Choose from {list(MODE_PRESETS)}.")

            config = self._build_config(mode_key, config_overrides or {}, dataset_root)
            job_id = f"train-{time.strftime('%Y%m%d-%H%M%S')}"
            log_path = self.logs_dir / f"{job_id}.log"
            python = sys.executable or "python"

            argv = [
                python, "-m", "scripts.train_vton",
                "--mode", mode_key,
                "--dataset", config.dataset_root,
                "--output", str(config.output_dir),
                "--resolution", str(config.resolution),
                "--batch-size", str(config.batch_size),
                "--epochs", str(config.num_epochs),
                "--lr", str(config.learning_rate),
                "--status-file", str(self.status_path),
                "--mode-config", json.dumps(config.to_dict()),
            ]
            if dry_run or config.dry_run:
                argv.append("--dry-run")

            env = dict(os.environ)
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("PYTHONPATH", str(self.project_root))

            log_handle = log_path.open("ab")
            try:
                process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                    argv, cwd=str(self.project_root), stdout=log_handle, stderr=subprocess.STDOUT,
                    env=env, start_new_session=True,
                )
            except FileNotFoundError as exc:
                log_handle.close()
                raise TrainingError(
                    "Could not start the training process — is Python available in PATH?",
                    details={"argv": argv, "error": str(exc)},
                ) from exc

            job = TrainingJob(
                job_id=job_id, mode=mode_key, config=config.to_dict(), started_at=time.time(),
                pid=process.pid, process=process, status_path=str(self.status_path), log_path=str(log_path),
                state="running",
            )
            self.jobs[job_id] = job
            self.active_job_id = job_id
            logger.info("Training job %s started (mode=%s pid=%s)", job_id, mode_key, process.pid)
            self._write_active_job(job)
            return job.to_dict()

    def _build_config(self, mode: str, overrides: Dict[str, Any], dataset_root: Optional[str]) -> TrainingConfig:
        settings = self.settings
        config = TrainingConfig.for_mode(mode)
        config.dataset_root = str(dataset_root or self._resolve_dataset_root())
        config.output_dir = str(settings.checkpoints_dir)
        config.device = settings.device
        config.force_cpu = settings.force_cpu
        config.allow_tf32 = settings.allow_tf32
        config.tracker = settings.tracker
        if mode == "QUICK_DEMO":
            config.resolution = min(config.resolution, 256)
        else:
            config.resolution = settings.resolution
            config.learning_rate = settings.learning_rate
            config.num_epochs = settings.num_epochs
            config.batch_size = settings.batch_size
            config.gradient_accumulation = settings.gradient_accumulation
            config.mixed_precision = settings.mixed_precision
            config.optimizer = settings.optimizer
            config.lr_scheduler = settings.scheduler
            config.save_top_k = settings.save_top_k
            config.early_stopping_patience = settings.early_stopping_patience
        for key, value in overrides.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
        return config

    def _resolve_dataset_root(self) -> str:
        """Pick the dataset: the prepared one if present, otherwise the sample set."""
        candidates = [
            Path(self.settings.datasets_dir) / "viton_hd",
            Path(self.settings.datasets_dir) / "viton-hd",
            Path(self.settings.datasets_dir) / "dresscode",
            Path(self.settings.samples_dir),
        ]
        for candidate in candidates:
            if (candidate / "train" / "pairs.txt").exists() or (candidate / "train" / "image").exists():
                return str(candidate)
        return str(Path(self.settings.samples_dir))

    def stop(self, job_id: Optional[str] = None, timeout: float = 20.0) -> Dict[str, Any]:
        """Gracefully stop a run (SIGINT first so the trainer can checkpoint, then SIGTERM)."""
        job = self._resolve_job(job_id)
        if job is None:
            raise TrainingError("No training job to stop.")
        if not job.alive:
            job.state = "stopped"
            return job.to_dict()

        try:
            os.killpg(os.getpgid(job.pid or 0), signal.SIGINT)
        except Exception as exc:  # pragma: no cover - platform differences
            logger.warning("SIGINT to training process failed (%s); trying terminate().", exc)
            try:
                job.process.terminate()  # type: ignore[union-attr]
            except Exception:
                pass

        deadline = time.time() + timeout
        while time.time() < deadline and job.alive:
            time.sleep(0.4)
        if job.alive:  # pragma: no cover - stubborn process
            try:
                os.killpg(os.getpgid(job.pid or 0), signal.SIGKILL)
            except Exception:
                try:
                    job.process.kill()  # type: ignore[union-attr]
                except Exception:
                    pass
        job.state = "stopped"
        job.stopped_at = time.time()
        job.exit_code = job.process.poll() if job.process else None
        logger.info("Training job %s stopped (exit=%s)", job.job_id, job.exit_code)
        self._write_active_job(None)
        return job.to_dict()

    # =================================================================================
    # status / dashboard
    # =================================================================================
    def get_active_status(self) -> Optional[Dict[str, Any]]:
        job = self.jobs.get(self.active_job_id) if self.active_job_id else None
        if job is None:
            return None
        if not job.alive and job.state == "running":
            job.state = "finished" if (job.process.poll() == 0 if job.process else True) else "failed"
            job.stopped_at = time.time()
            job.exit_code = job.process.poll() if job.process else None
        return job.to_dict()

    def status(self) -> Dict[str, Any]:
        """Everything the AI Training Dashboard needs in one call."""
        active = self.get_active_status()
        trainer_status = self._read_status_file()
        history = self.checkpoints.read_history()
        metrics = self._read_latest_metrics()
        devices = detect_devices()
        return {
            "active_job": active,
            "trainer": trainer_status,
            "modes": {key: {k: v for k, v in value.items()} for key, value in MODE_PRESETS.items()},
            "dataset": self.dataset_summary(),
            "checkpoints": self.checkpoints.status(),
            "history": history,
            "metrics": metrics,
            "validation": load_validation_summary(Path(self.settings.results_dir) / "validation"),
            "evaluations": load_evaluations(Path(self.settings.results_dir) / "evaluation")[:3],
            "device": devices.as_dict(),
            "gpu_live": gpu_utilization(),
            "vram_warning": vram_warning(6.0, self.settings.resolution),
            "log_tail": self.log_tail(60),
        }

    def dataset_summary(self) -> Dict[str, Any]:
        """Sizes and layout of the datasets currently available."""
        out: Dict[str, Any] = {"available": [], "active": None}
        for name in ("viton_hd", "viton-hd", "dresscode", "samples"):
            path = Path(self.settings.datasets_dir) / name
            if not path.exists():
                continue
            counts = {}
            for split in ("train", "val", "test"):
                pairs = path / split / "pairs.txt"
                images = path / split / "image"
                counts[split] = (sum(1 for _ in pairs.open("r", encoding="utf-8")) if pairs.exists()
                                 else (len(list(images.glob("*"))) if images.exists() else 0))
            metadata = {}
            meta_path = path / "metadata.json"
            if meta_path.exists():
                try:
                    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:  # pragma: no cover
                    metadata = {}
            entry = {
                "name": name, "path": str(path), "counts": counts, "total": sum(counts.values()),
                "metadata": metadata,
            }
            out["available"].append(entry)
        samples_exists = (Path(self.settings.samples_dir) / "train").exists()
        out["sample_mode_ready"] = samples_exists
        out["hint"] = (
            "Sample dataset ready (procedurally generated). Run `python scripts/prepare_dataset.py "
            "--use-samples` if you have not generated it yet, or point --dataset at your "
            "VITON-HD/DressCode conversion."
        )
        active = self._resolve_dataset_root()
        out["active"] = active
        return out

    def _read_status_file(self) -> Dict[str, Any]:
        if not self.status_path.exists():
            return {}
        try:
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - partially written file
            return {}

    def _read_latest_metrics(self) -> Dict[str, Any]:
        candidates = sorted(self.logs_dir.glob("training_*/metrics.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            return {}
        curves = load_history(candidates[0])
        return {
            "path": str(candidates[0]),
            "curves": {name: values[-200:] for name, values in curves.items()},
            "latest": {name: values[-1]["value"] for name, values in curves.items() if values},
        }

    def log_tail(self, lines: int = 60) -> List[str]:
        job = self.jobs.get(self.active_job_id) if self.active_job_id else None
        path = Path(job.log_path) if job else None
        if path is None or not path.exists():
            candidates = sorted(self.logs_dir.glob("train-*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            path = candidates[0] if candidates else None
        if path is None or not path.exists():
            return []
        try:
            content = path.read_text(encoding="utf-8", errors="replace").splitlines()
            return content[-lines:]
        except OSError:  # pragma: no cover
            return []

    def checkpoints_view(self) -> Dict[str, Any]:
        return self.checkpoints.status()

    def delete_checkpoint(self, name: str) -> Dict[str, Any]:
        """Delete one epoch checkpoint folder (never ``best_model``/``latest_model`` by accident)."""
        import shutil

        target = self.checkpoints.epochs_dir / name
        if not target.exists():
            raise TrainingError(f"Checkpoint '{name}' not found under {self.checkpoints.epochs_dir}.")
        shutil.rmtree(target, ignore_errors=True)
        logger.info("Deleted checkpoint %s", name)
        return self.checkpoints.status()

    def promote_checkpoint(self, name: str) -> Dict[str, Any]:
        """Promote an epoch checkpoint to ``best_model`` (used after a manual comparison)."""
        source = self.checkpoints.epochs_dir / name / "adapter"
        if not source.exists():
            source = self.checkpoints.epochs_dir / name
        if not source.exists():
            raise TrainingError(f"Checkpoint '{name}' has no adapter folder.")
        self.checkpoints.export_model(source, self.checkpoints.best_dir, mirror_latest=True)
        return self.checkpoints.status()

    def curves(self, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Return training curves and (re)generate the plot artefact."""
        data = history or self.checkpoints.read_history().get("epochs", [])
        if not data:
            return {"ok": False, "reason": "No training history yet.", "epochs": []}
        plot_path = plot_training_curves(data, Path(self.settings.results_dir) / "training_curves.png")
        return {"ok": True, "epochs": data, "plot": str(plot_path), "plot_url": f"/api/training/curves/image"}

    # ---------------------------------------------------------------------------------
    def _write_active_job(self, job: Optional[TrainingJob]) -> None:
        path = self.logs_dir / "active_job.json"
        payload = job.to_dict() if job else {"state": "none"}
        try:
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except OSError:  # pragma: no cover
            pass

    def _resolve_job(self, job_id: Optional[str]) -> Optional[TrainingJob]:
        if job_id:
            return self.jobs.get(job_id)
        return self.jobs.get(self.active_job_id) if self.active_job_id else None

    def shutdown(self) -> None:
        """Stop any child process at server shutdown so GPUs are not left occupied."""
        job = self._resolve_job(None)
        if job is not None and job.alive:
            logger.warning("Server shutting down with an active training job; stopping it.")
            try:
                self.stop(job.job_id, timeout=10)
            except Exception:  # pragma: no cover
                pass
