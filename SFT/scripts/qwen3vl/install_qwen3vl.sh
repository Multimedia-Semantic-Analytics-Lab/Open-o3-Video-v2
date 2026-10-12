#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"
export UV_LINK_MODE=copy
python -m pip install --upgrade uv

uv venv .venv --python 3.12 --allow-existing
source .venv/bin/activate

# flash-attn and Transformer Engine build against the installed PyTorch/CUDA stack.
uv pip install "torch==2.8.0" "torchvision==0.23.0" \
  --index https://download.pytorch.org/whl/cu126
uv sync --active

echo "Install complete. Virtual environment: ${SCRIPT_DIR}/.venv"
