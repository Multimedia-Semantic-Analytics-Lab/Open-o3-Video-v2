#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${UV_PROJECT_ENVIRONMENT:-${ROOT}/.venv}"
python -m pip install --upgrade uv
uv venv --python 3.11 "${VENV}"
source "${VENV}/bin/activate"
uv pip install -e "${ROOT}/verl" \
  "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0" "torchdata==0.10.0" \
  "vllm==0.11.0" "transformers==4.57.1" "numpy==1.26.4" \
  "qwen-vl-utils==0.0.14" "decord==0.6.0" "ninja>=1.11"
MAX_JOBS="${MAX_JOBS:-8}" uv pip install "flash-attn==2.8.3" --no-build-isolation
echo "RL environment ready: ${VENV}"
