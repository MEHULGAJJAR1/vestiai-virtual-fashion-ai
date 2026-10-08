"""Shared helpers for every VestiAI CLI script.

Running ``python scripts/<name>.py`` puts ``scripts/`` on ``sys.path``, so the other scripts
import this as ``from _common import ...``. It keeps the CLI layer thin and consistent:
same banner, same project-root resolution, same "is the ML stack installed?" message.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import-time convenience
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------------------
# presentation
# ---------------------------------------------------------------------------------------
def banner(title: str, subtitle: str = "") -> None:
    """Print a section banner (kept ASCII so it renders on every terminal)."""
    width = max(60, len(title) + 6)
    print("=" * width)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * width)


def kv_table(payload: Dict[str, Any], indent: str = "  ") -> None:
    """Print a flat dictionary as aligned ``key : value`` rows."""
    if not payload:
        return
    width = max(len(str(k)) for k in payload)
    for key, value in payload.items():
        rendered = json.dumps(value, default=str) if isinstance(value, (dict, list, tuple)) else str(value)
        print(f"{indent}{str(key).ljust(width)} : {rendered}")


def human_time(seconds: float) -> str:
    """``754.2`` → ``12m 34s``."""
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def human_bytes(count: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(count) < 1024 or unit == "TB":
            return f"{count:.1f} {unit}"
        count /= 1024
    return f"{count:.1f} TB"


# ---------------------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------------------
def log_level_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="console verbosity (default: INFO)",
    )


def setup_logging(level: str = "INFO") -> None:
    """Send the backend logger to stdout *and* logs/vestiai.log for the CLI tools."""
    from backend.utils.logging_utils import setup_logging as _setup

    _setup(level=level, log_dir=PROJECT_ROOT / "logs")


def load_settings(config: Optional[str] = None):
    """Load runtime settings (``configs/default.yaml`` + env + CLI config)."""
    from backend.config import load_settings as _load

    return _load(config_file=config)


def require_torch(purpose: str = "this command") -> Any:
    """Import torch or exit with an actionable message (never a bare ImportError)."""
    try:
        import torch  # noqa: WPS433 (deliberate late import)

        return torch
    except Exception as exc:  # pragma: no cover - depends on the machine
        print(f"\n✗ {purpose} needs PyTorch, which is not installed here ({exc}).")
        print("  Install the ML extra first:")
        print("      python scripts/setup.py --profile ml")
        print("  Live try-on and the API work without it (lightweight pipeline).\n")
        raise SystemExit(2)


def ml_stack_report() -> Dict[str, Any]:
    """Versions of the optional stack — used by several scripts' preflight blocks."""
    from backend.models.vton_controlnet import environment_report

    return environment_report()


def print_device_banner() -> None:
    """Show the resolved device + torch status before a long-running command."""
    from backend.config import resolve_device
    from backend.utils.device import detect_devices

    devices = detect_devices()
    resolved = resolve_device(None)
    print(f"  device   : {resolved}  ({devices.device_label()})")
    print(f"  torch    : {devices.torch_version or 'not installed'}")
    if devices.cuda_available and devices.vram_total_gb:
        print(f"  VRAM     : {devices.vram_total_gb[0]} GB")
    elif not devices.cuda_available:
        print("  CUDA     : not available — QUICK_DEMO works on CPU; use a GPU node for the rest")
    print()


# ---------------------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------------------
def resolve_dataset(settings: Any, requested: Optional[str], *, create: bool = False) -> Optional[Path]:
    """Resolve a dataset name or path to an absolute directory.

    ``settings.datasets_dir`` is searched first, so ``--dataset viton_hd`` finds
    ``datasets/viton_hd`` even when the command runs from another directory.
    """
    if not requested:
        return None
    candidate = Path(requested)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for option in (Path.cwd() / candidate, Path(settings.datasets_dir) / candidate, PROJECT_ROOT / candidate):
        if option.exists():
            return option.resolve()
    if create:
        target = Path(settings.datasets_dir) / candidate
        target.mkdir(parents=True, exist_ok=True)
        return target
    return None


def dataset_summary(root: Optional[Path], resolution: int = 512) -> Dict[str, Any]:
    """Pair counts + problems for a dataset directory (safe on an empty folder)."""
    if root is None or not Path(root).exists():
        return {"ok": False, "reason": "dataset directory not found"}
    from backend.datasets.preprocessing import validate_layout

    report = validate_layout(root)
    report["root"] = str(root)
    report.setdefault("resolution", resolution)
    return report


def progress_printer(label: str, every: float = 2.0):
    """Return a callback factory printer suitable for long loops (used by converters)."""
    started = time.time()

    def _print(step: int, total: int, message: str = "") -> None:
        if total and step % max(1, int(total / 12) or 1) and step != total:
            return
        elapsed = time.time() - started
        rate = step / elapsed if elapsed > 0 else 0
        tail = f" — {message}" if message else ""
        print(f"  {label}: {step}/{total or '?'} ({rate:.1f}/s){tail}")

    return _print


# ---------------------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------------------
def resolve_checkpoint(requested: Optional[str] = None) -> Optional[Path]:
    """Resolve a checkpoint directory: explicit path → best_model → latest_model → newest epoch."""
    from backend.config import load_settings as _load
    from backend.models.checkpointing import CheckpointManager

    settings = _load()
    manager = CheckpointManager(settings.checkpoints_dir)
    if requested:
        candidate = Path(requested)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate)
            if not candidate.exists():
                candidate = PROJECT_ROOT / requested
        resolved = manager.resolve_for_inference(str(candidate))
        return Path(resolved) if resolved else None
    resolved = manager.resolve_for_inference("auto")
    return Path(resolved) if resolved else None


def require_checkpoint(requested: Optional[str], purpose: str) -> Path:
    resolved = resolve_checkpoint(requested)
    if resolved is None:
        print(f"\n✗ {purpose} needs a trained checkpoint and none was found.")
        print("  Looked in : checkpoints/best_model, checkpoints/latest_model, checkpoints/epochs/*")
        print("  Train one : python scripts/train.py --mode QUICK_DEMO")
        print("  Or import : copy a checkpoint folder into checkpoints/best_model\n")
        raise SystemExit(3)
    return resolved


# ---------------------------------------------------------------------------------------
# argparse helpers
# ---------------------------------------------------------------------------------------
class _Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Shows defaults and keeps the epilog's line breaks."""


def parser(description: str, epilog: str = "") -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=description, epilog=epilog, formatter_class=_Formatter)


def add_common_io_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="runtime YAML (default: configs/default.yaml)")
    log_level_arg(p)


def fail(message: str, code: int = 1) -> None:
    """Print an error the way the rest of the project reports problems and exit."""
    print(f"\n✗ {message}\n")
    raise SystemExit(code)


def print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def parse_list(value: Optional[str]) -> Optional[Sequence[str]]:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]
