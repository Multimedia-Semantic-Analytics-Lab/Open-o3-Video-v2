#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/openo3_run_lib.sh"
openo3_prepare_env 8

run_model_name="openo3video_v2_4b"
model_name="openo3video_v2_4b"
export OPENO3VIDEO_V2_4B_MODEL_PATH="${OPENO3VIDEO_V2_4B_MODEL_PATH:-models/Open-o3-Video-v2-4B}"
checkpoint_label="${OPENO3VIDEO_V2_4B_MODEL_PATH}"

dataset_names=(
  "video_mme"
  "video_holmes"
  "lvbench"
  "tvgbench"
)

dataset_dirs=(
  "${VIDEO_MME_DATASET_DIR:-${DATA_ROOT:-data/benchmarks}/Video-MME/Video-MME.json}"
  "${VIDEO_HOLMES_DATASET_DIR:-${DATA_ROOT:-data/benchmarks}/Video-Holmes/test_Video-Holmes.json}"
  "${LVBENCH_DATASET_DIR:-${DATA_ROOT:-data/benchmarks}/LVBench/lvbench.json}"
  "${TVGBENCH_DATASET_DIR:-${DATA_ROOT:-data/benchmarks}/TVGBench/tvgbench.json}"
)

max_turn_list=(5 5 5 0)
frames_limit_list=(96 96 96 96)
max_new_tokens_list=(4096 4096 4096 4096)
videozerobench_mode_list=("" "" "" "")
dataset_env_list=(
  ""
  ""
  ""
  "TVGBENCH_VIDEO_ROOT=${TVGBENCH_VIDEO_ROOT:-${DATA_ROOT:-data/benchmarks}/TVGBench/videos}"
)

openo3_run "$@"
