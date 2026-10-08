#!/usr/bin/env python
"""VestiAI — environment setup helper.

Installs the right dependency profile for the machine you are on, detects the accelerator
(Apple MPS / NVIDIA CUDA / CPU) and prints exactly what to do next. It never silently
installs a GPU wheel on a CPU-only box, never requires a local GPU for training, and never
touches your system Python — it operates on the interpreter it is run with.

    python scripts/setup.py --profile core     # API + live try-on (fast, no torch)
    python scripts/setup.py --profile ml       # + PyTorch/Diffusers extras, CUDA-aware
    python scripts/setup.py --profile dev      # + pytest and friends
    python scripts/setup.py --profile all      # everything
    python scripts/setup.py --check            # report only, install nothing
    python scripts/setup.py --torch cuda121    # force a specific torch wheel

Exit codes: 0 ok · 1 install failed · 2 unusable environment
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from _common import PROJECT_ROOT, banner, kv_table, print_json  # noqa: E402

PROFILES = {
    "core": ["requirements.txt"],
    "ml": ["requirements.txt", "requirements-ml.txt"],
    "dev": ["requirements.txt", "requirements-dev.txt"],
    "all": ["requirements.txt", "requirements-ml.txt", "requirements-dev.txt"],
}

TORCH_INDEXES = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cuda118": "https://download.pytorch.org/whl/cu118",
    "cuda121": "https://download.pytorch.org/whl/cu121",
    "cuda124": "https://download.pytorch.org/whl/cu124",
    "cuda126": "https://download.pytorch.org/whl/cu126",
    "rocm62": "https://download.pytorch.org/whl/rocm6.2",
}

CHECK_MODULES = {
    "fastapi": "FastAPI web layer",
    "uvicorn": "ASGI server",
    "cv2": "OpenCV (imaging + warping)",
    "numpy": "numeric core",
    "PIL": "Pillow image IO",
    "mediapipe": "pose + hand landmarks",
    "sklearn": "CPU classifier head",
    "torch": "PyTorch (optional — training/diffusion)",
    "diffusers": "Diffusers (optional — VTON pipeline)",
    "transformers": "Transformers (optional — CLIP tiers)",
    "rembg": "rembg (optional — better garment cut-outs)",
}


# ---------------------------------------------------------------------------------------
def python_report() -> dict:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "pip": shutil.which("pip") or shutil.which("pip3") or "not found",
    }


def detect_accelerator() -> dict:
    """Best-effort accelerator detection that works before torch is installed."""
    report = {"accelerator": "cpu", "reason": "no GPU runtime detected", "cuda_driver": None}
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        report.update(accelerator="mps", reason="Apple Silicon — MPS backend available to PyTorch")
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15, check=False,
            )
            first = (out.stdout or "").strip().splitlines()
            if first:
                parts = [p.strip() for p in first[0].split(",")]
                report.update(
                    accelerator="cuda",
                    reason=f"NVIDIA GPU: {parts[0]} ({parts[1] if len(parts) > 1 else '?'} VRAM)",
                    cuda_driver=parts[2] if len(parts) > 2 else None,
                    gpu_count=len(first),
                )
        except Exception as exc:  # pragma: no cover - driver quirks
            report["cuda_driver"] = f"nvidia-smi failed: {exc}"
    return report


def guess_torch_flavour(accelerator: str, forced: str | None) -> str:
    if forced:
        if forced not in TORCH_INDEXES:
            raise SystemExit(f"unknown --torch value '{forced}' (choose from {', '.join(TORCH_INDEXES)})")
        return forced
    if accelerator == "cuda":
        # Pick the newest CUDA wheel <= the installed driver's capability, defaulting to 12.4.
        return "cuda124"
    if accelerator == "mps":
        return "cpu"          # the default macOS wheel already contains MPS support
    return "cpu"


def pip_install(args: list[str], dry: bool = False) -> bool:
    command = [sys.executable, "-m", "pip", "install"] + args
    print("  $ " + " ".join(command))
    if dry:
        return True
    return subprocess.run(command, check=False).returncode == 0


def check_modules() -> dict:
    import importlib

    found: dict[str, str] = {}
    for module, label in CHECK_MODULES.items():
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, "__version__", "installed")
            found[label] = version
        except Exception:
            found[label] = "missing"
    return found


# ---------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Install the dependency profile that matches this machine.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--profile", default="core", choices=sorted(PROFILES))
    p.add_argument("--torch", default=None, choices=sorted(TORCH_INDEXES), help="force a torch wheel flavour")
    p.add_argument("--check", action="store_true", help="report the environment and exit")
    p.add_argument("--no-torch", action="store_true", help="install the ML extra without touching torch")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    p.add_argument("--upgrade", action="store_true", help="pass --upgrade to pip")
    p.add_argument("--json", action="store_true", help="machine-readable report")
    args = p.parse_args(argv)

    accelerator = detect_accelerator()
    report = {**python_report(), **accelerator}
    if args.json:
        print_json(report)
        if args.check:
            return 0

    if not args.json:
        banner("VestiAI · environment setup", f"profile: {args.profile}")
        kv_table(report)
        print()

    if args.check:
        banner("dependency check", "what is importable right now")
        for label, version in check_modules().items():
            mark = "✓" if version != "missing" else "·"
            print(f"  {mark} {label.ljust(34)} {version}")
        print("\n  torch / diffusers / transformers missing is fine: the app falls back to the")
        print("  geometric pipeline and the UI states that clearly.\n")
        return 0

    requirements = [PROJECT_ROOT / name for name in PROFILES[args.profile]]
    for path in requirements:
        if not path.exists():
            print(f"✗ required file missing: {path}")
            return 2

    ok = True
    flavour = guess_torch_flavour(accelerator["accelerator"], args.torch)
    if "requirements-ml.txt" in [path.name for path in requirements] and not args.no_torch:
        banner("step 1/2 · PyTorch", f"wheel flavour: {flavour}  ({TORCH_INDEXES[flavour]})")
        torch_args = ["torch", "torchvision"]
        if flavour in {"cuda118", "cuda121", "cuda124", "cuda126", "rocm62"}:
            torch_args = ["--index-url", TORCH_INDEXES[flavour]] + torch_args
        if args.upgrade:
            torch_args.append("--upgrade")
        ok = pip_install(torch_args, dry=args.dry_run) and ok

    banner(f"step {2 if 'requirements-ml.txt' in [p.name for p in requirements] else 1} · requirements")
    for path in requirements:
        print(f"  installing {path.name}")
        ok = pip_install(["-r", str(path)] + (["--upgrade"] if args.upgrade else []), dry=args.dry_run) and ok

    print()
    banner("next steps")
    print("  1. start the app            : python run.py")
    print("  2. open the UI              : http://localhost:8000")
    print("  3. verify the whole pipeline: python scripts/prepare_dataset.py --use-samples")
    print("                                python scripts/train.py --mode QUICK_DEMO --dry-run")
    if accelerator["accelerator"] == "cuda":
        print("  4. full fine-tune           : python scripts/train.py --mode FINE_TUNE \\")
        print("                                    --config configs/training_finetune.yaml --dataset datasets/viton_hd")
    else:
        print("  4. no CUDA GPU here → QUICK_DEMO works on CPU; run FINE_TUNE/FULL_TRAINING on")
        print("     Colab/RunPod (configs/training_colab.yaml) and copy back checkpoints/best_model.")
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
