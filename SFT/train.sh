#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

CONFIG="${1:-configs/qwen3vl/open-o3-video/qwen3_sft_mix_multi_image_135k.yaml}"
PREPARED_DIR="${PREPARED_DIR:-${ROOT}/prepared}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/openo3-video-v2-sft}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/scripts/qwen3vl/.venv/bin/python}"
: "${MODEL_PATH:?Set MODEL_PATH to a downloaded Qwen3-VL base model}"
export PREPARED_DIR OUTPUT_DIR MODEL_PATH

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 2
fi

TMP_CONFIG="$(mktemp --suffix=.yaml)"
trap 'rm -f "${TMP_CONFIG}"' EXIT

"${PYTHON_BIN}" - "${CONFIG}" "${TMP_CONFIG}" <<'PY'
import os
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
prepared = Path(os.environ["PREPARED_DIR"]).expanduser().resolve()
datasets = [prepared / f"{name}.jsonl" for name in config["dataset"]]
missing = [str(path) for path in datasets if not path.is_file()]
if missing:
    raise SystemExit("Prepare the SFT data first; missing:\n" + "\n".join(missing))
config["dataset"] = [str(path) for path in datasets]
config["model"] = os.environ["MODEL_PATH"]
config["output_dir"] = os.environ["OUTPUT_DIR"]
config.pop("custom_register_path", None)
config["report_to"] = "none" if os.environ.get("WANDB_MODE", "disabled") == "disabled" else "wandb"
Path(sys.argv[2]).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cat "${TMP_CONFIG}"
  exit 0
fi

"${PYTHON_BIN}" -m swift.cli._megatron.main sft --config "${TMP_CONFIG}"
