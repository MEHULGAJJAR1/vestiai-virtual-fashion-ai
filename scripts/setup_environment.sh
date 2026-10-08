#!/usr/bin/env bash
# ======================================================================================
# VestiAI — one-command environment bootstrap (macOS / Linux / WSL)
#
#   ./scripts/setup_environment.sh              # core profile (API + live try-on)
#   ./scripts/setup_environment.sh --ml         # + PyTorch / Diffusers (CUDA-aware)
#   ./scripts/setup_environment.sh --all        # + dev/test tooling
#   ./scripts/setup_environment.sh --venv .venv --ml
#
# Creates a virtual environment, installs the right wheels for this machine, verifies the
# install and prints the next commands. Safe to re-run.
#
# Windows users: use scripts\run_training.bat for training, or run this file under WSL /
# Git-Bash. See docs/DEPLOYMENT.md for the native PowerShell equivalent.
# ======================================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PROFILE="core"
VENV=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ml)   PROFILE="ml"; shift ;;
    --all)  PROFILE="all"; shift ;;
    --dev)  PROFILE="dev"; shift ;;
    --core) PROFILE="core"; shift ;;
    --venv) VENV="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

echo "=============================================================="
echo "  VestiAI — environment bootstrap"
echo "  project : ${PROJECT_ROOT}"
echo "  profile : ${PROFILE}"
echo "  venv    : ${VENV}"
echo "=============================================================="

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "✗ ${PYTHON_BIN} not found. Install Python 3.10–3.12 and retry." >&2
  exit 2
fi

PY_VERSION="$(${PYTHON_BIN} -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "  python  : ${PY_VERSION}"
case "${PY_VERSION}" in
  3.10|3.11|3.12|3.13) : ;;
  *) echo "⚠ VestiAI targets Python 3.10–3.13 (found ${PY_VERSION}); continuing anyway." ;;
esac

# --- GPU hints -------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "  GPU     : $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n1)"
  TORCH_FLAVOUR="--torch cuda124"
elif [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  echo "  GPU     : Apple Silicon (MPS) — the default macOS torch wheel includes MPS"
  TORCH_FLAVOUR=""
else
  echo "  GPU     : none detected — CPU wheels only (QUICK_DEMO still works)"
  TORCH_FLAVOUR="--torch cpu"
fi

# --- virtualenv ------------------------------------------------------------------------
if [[ ! -d "${VENV}" ]]; then
  echo "→ creating virtual environment"
  "${PYTHON_BIN}" -m venv "${VENV}"
fi
# shellcheck disable=SC1090
source "${VENV}/bin/activate"
python -m pip install --quiet --upgrade pip wheel

echo "→ installing the '${PROFILE}' profile"
# shellcheck disable=SC2086
python scripts/setup.py --profile "${PROFILE}" ${TORCH_FLAVOUR}

# --- vendor the browser runtime (optional; the repo already contains it) ---------------
VENDOR="frontend/vendor/vision_bundle.mjs"
if [[ -f "${VENDOR}" ]]; then
  echo "→ browser pose runtime already vendored (frontend/vendor/)"
else
  echo "→ fetching the MediaPipe tasks-vision bundle for the browser"
  bash scripts/fetch_frontend_vendor.sh || echo "⚠ could not fetch the vendored runtime — Live mode will say so in the UI"
fi

echo "→ verifying the installation"
python scripts/setup.py --check || true

cat <<EOF

==============================================================
  Ready.
==============================================================
  activate : source ${VENV}/bin/activate
  run app  : python run.py                  → http://localhost:8000
  docs     : http://localhost:8000/docs
  samples  : python scripts/prepare_dataset.py --use-samples
  verify   : python scripts/train.py --mode QUICK_DEMO --dry-run
  tests    : python -m pytest tests -q

EOF
