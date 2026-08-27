#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage: $(basename "$0") [VIMEO_ROOT] [NUM_VAL_PAIRS]

Download Vimeo-90K septuplet, extract sequences, and build LF-VSN list files.

Arguments:
  VIMEO_ROOT      Output directory for dataset assets. Default: ${CODE_DIR}/vimeo90k
  NUM_VAL_PAIRS   Number of cover sequences in test list. Default: 200

Examples:
  $(basename "$0")
  $(basename "$0") /data/vimeo90k 200
EOF
  exit 0
fi

VIMEO_ROOT="${1:-${CODE_DIR}/vimeo90k}"
NUM_VAL_PAIRS="${2:-200}"
DATA_URL="http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip"
ZIP_PATH="${VIMEO_ROOT}/downloads/vimeo_septuplet.zip"

mkdir -p "${VIMEO_ROOT}/downloads"

if [[ ! -f "${ZIP_PATH}" ]]; then
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 16 -s 16 -k 1M -d "${VIMEO_ROOT}/downloads" -o "vimeo_septuplet.zip" "${DATA_URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${ZIP_PATH}" "${DATA_URL}"
  else
    curl -L "${DATA_URL}" -o "${ZIP_PATH}"
  fi
fi

if [[ ! -d "${VIMEO_ROOT}/sequences" ]]; then
  unzip -q -n "${ZIP_PATH}" -d "${VIMEO_ROOT}/downloads"

  if [[ -d "${VIMEO_ROOT}/downloads/vimeo_septuplet/sequences" ]]; then
    mv "${VIMEO_ROOT}/downloads/vimeo_septuplet/sequences" "${VIMEO_ROOT}/sequences"
  elif [[ -d "${VIMEO_ROOT}/downloads/sequences" ]]; then
    mv "${VIMEO_ROOT}/downloads/sequences" "${VIMEO_ROOT}/sequences"
  else
    echo "Could not locate extracted sequences folder."
    exit 1
  fi
fi

python3 "${SCRIPT_DIR}/build_vimeo90k_lists.py" \
  --output-dir "${VIMEO_ROOT}" \
  --num-test-pairs "${NUM_VAL_PAIRS}" \
  --test-secrets 1 7 \
  --data-root "${VIMEO_ROOT}/sequences" \
  --min-frames 7

echo "Vimeo-90K septuplet setup complete."
echo "Data root: ${VIMEO_ROOT}/sequences"
echo "Train list: ${VIMEO_ROOT}/sep_trainlist.txt"
echo "Test list (1 secret): ${VIMEO_ROOT}/sep_testlist.txt"
echo "Test list (7 secret): ${VIMEO_ROOT}/sep_testlist_7video.txt"
