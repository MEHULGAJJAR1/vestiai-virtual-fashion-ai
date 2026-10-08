#!/usr/bin/env bash
# ======================================================================================
# VestiAI — vendor the browser-side MediaPipe runtime into frontend/vendor/.
#
# The app must work offline once installed, so the tasks-vision bundle (WASM + JS) and the
# pose / hand / selfie-segmenter models are downloaded once and committed to the repo.
# Running this script again simply refreshes them to the pinned version.
#
#   ./scripts/fetch_frontend_vendor.sh
#   MEDIAPIPE_TASKS_VERSION=0.10.14 ./scripts/fetch_frontend_vendor.sh
#
# Everything lives under frontend/vendor/:
#   vision_bundle.mjs
#   wasm/vision_wasm_internal.{js,wasm}
#   wasm/vision_wasm_nosimd_internal.{js,wasm}
#   models/pose_landmarker_lite.task
#   models/hand_landmarker.task
#   models/selfie_segmenter.tflite
# ======================================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENDOR="${ROOT}/frontend/vendor"
VERSION="${MEDIAPIPE_TASKS_VERSION:-0.10.14}"
CDN="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}"
MODELS="https://storage.googleapis.com/mediapipe-models"

mkdir -p "${VENDOR}/wasm" "${VENDOR}/models"

fetch() {  # fetch <url> <destination>
  local url="$1" dest="$2"
  if [[ -s "${dest}" ]]; then
    echo "  · $(basename "${dest}") already present ($(du -h "${dest}" | cut -f1))"
    return 0
  fi
  echo "  → $(basename "${dest}")"
  curl -fsSL --retry 3 --retry-delay 2 "${url}" -o "${dest}"
}

echo "=============================================================="
echo "  Vendoring MediaPipe tasks-vision ${VERSION} → frontend/vendor/"
echo "=============================================================="

fetch "${CDN}/vision_bundle.mjs"                                   "${VENDOR}/vision_bundle.mjs"
fetch "${CDN}/wasm/vision_wasm_internal.js"                        "${VENDOR}/wasm/vision_wasm_internal.js"
fetch "${CDN}/wasm/vision_wasm_internal.wasm"                      "${VENDOR}/wasm/vision_wasm_internal.wasm"
fetch "${CDN}/wasm/vision_wasm_nosimd_internal.js"                 "${VENDOR}/wasm/vision_wasm_nosimd_internal.js"
fetch "${CDN}/wasm/vision_wasm_nosimd_internal.wasm"               "${VENDOR}/wasm/vision_wasm_nosimd_internal.wasm"

fetch "${MODELS}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" \
      "${VENDOR}/models/pose_landmarker_lite.task"
fetch "${MODELS}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" \
      "${VENDOR}/models/hand_landmarker.task"
fetch "${MODELS}/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite" \
      "${VENDOR}/models/selfie_segmenter.tflite"

echo
echo "  total: $(du -sh "${VENDOR}" | cut -f1)"
echo "  Live Try-On can now run with no internet access."
echo
