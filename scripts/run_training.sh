#!/usr/bin/env bash
# ======================================================================================
# VestiAI — one-command training runner (macOS / Linux / WSL)
#
#   ./scripts/run_training.sh                      # QUICK_DEMO on the sample dataset
#   ./scripts/run_training.sh --mode FINE_TUNE --dataset datasets/viton_hd
#   ./scripts/run_training.sh --help
#
# Everything the script does is delegated to the Python CLI tools, so you can always run
# the exact same steps by hand:
#
#   python scripts/setup.py --profile ml          # PyTorch + Diffusers (once)
#   python scripts/prepare_dataset.py --use-samples
#   python scripts/validate_dataset.py --dataset datasets/samples
#   python scripts/train.py --mode QUICK_DEMO
#
# On Windows use scripts\run_training.bat instead.
# ======================================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODE="QUICK_DEMO"
DATASET=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)    MODE="${2:-}"; shift 2 ;;
    --dataset) DATASET="${2:-}"; shift 2 ;;
    -h|--help)
      sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

PY="${PYTHON:-python3}"
if ! command -v "${PY}" >/dev/null 2>&1; then
  echo "✗ ${PY} not found — install Python 3.10+ or set PYTHON=/path/to/python" >&2
  exit 2
fi

step() { printf '\n\033[1;35m==> %s\033[0m\n' "$*"; }

step "VestiAI training · mode=${MODE}"

step "1/5  environment"
if ! "${PY}" -c "import torch" >/dev/null 2>&1; then
  echo "    PyTorch is not installed. Installing the ML profile (this can take a few minutes)…"
  "${PY}" scripts/setup.py --profile ml || {
    echo "✗ ML profile install failed. Run it manually to see the full log:" >&2
    echo "    ${PY} scripts/setup.py --profile ml" >&2
    exit 3
  }
else
  "${PY}" scripts/setup.py --check --profile ml || true
fi

step "2/5  dataset"
if [[ -n "${DATASET}" ]]; then
  "${PY}" scripts/validate_dataset.py --dataset "${DATASET}" || {
    echo "✗ dataset '${DATASET}' did not pass validation." >&2
    exit 4
  }
elif [[ -f "datasets/samples/train/pairs.txt" ]]; then
  echo "    using datasets/samples (already prepared)"
  "${PY}" scripts/validate_dataset.py --dataset datasets/samples || true
else
  echo "    no dataset found — generating the sample dataset"
  "${PY}" scripts/prepare_dataset.py --use-samples
fi

step "3/5  GPU check"
if "${PY}" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "    CUDA available."
else
  echo "    No CUDA GPU — QUICK_DEMO still runs on CPU (a few minutes)."
  echo "    FINE_TUNE / FULL_TRAINING belong on a CUDA box; see configs/training_colab.yaml."
  if [[ "${MODE}" != "QUICK_DEMO" ]]; then
    echo "    Continuing anyway — pass --dry-run to verify the wiring first."
  fi
fi

step "4/5  training (mode=${MODE})"
TRAIN_ARGS=("--mode" "${MODE}")
if [[ -n "${DATASET}" ]]; then
  TRAIN_ARGS+=("--dataset" "${DATASET}")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  TRAIN_ARGS+=("${EXTRA_ARGS[@]}")
fi
"${PY}" scripts/train.py "${TRAIN_ARGS[@]}"

step "5/5  next steps"
"${PY}" - <<'PY'
from pathlib import Path
best = Path("checkpoints/best_model")
latest = Path("checkpoints/latest_model")
for label, path in (("best", best), ("latest", latest)):
    state = "present" if path.exists() else "missing"
    print(f"    {label:<6} checkpoint: {path} ({state})")
print("    validate : python scripts/validate.py --checkpoint checkpoints/best_model")
print("    evaluate : python scripts/evaluate.py --checkpoint checkpoints/best_model")
print("    serve    : python run.py            # Model Status → Reload checkpoint")
PY

echo
echo "✓ training run finished."
