#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/openo3_run_lib.sh"
openo3_prepare_env 8

run_model_name="openo3video_v2_4b"
model_name="openo3video_v2_4b"
export OPENO3VIDEO_V2_4B_MODEL_PATH="${OPENO3VIDEO_V2_4B_MODEL_PATH:-models/Open-o3-Video-v2-4B}"
checkpoint_label="${OPENO3VIDEO_V2_4B_MODEL_PATH}"

dataset_names=(
  "v_star"
  "timelens_bench"
  "video_mme"
  "video_mme_v2"
  "worldsense"
  "videommmu"
  "video_holmes"
  "lvbench"
  "tvgbench"
)

dataset_dirs=(
  "${V_STAR_DATASET_DIR:-${DATA_ROOT:-data/benchmarks}/V-STaR/V_STaR_test.json}"
  "${TIMELENS_BENCH_DATASET_DIR:-${DATA_ROOT:-data/benchmarks}/TimeLens-Bench/charades-timelens.json}"
  "${VIDEO_MME_DATASET_DIR:-${DATA_ROOT:-data/benchmarks}/Video-MME/Video-MME.json}"
  "${VIDEO_MME_V2_DATASET_DIR:-${DATA_ROOT:-data/benchmarks}/Video-MME-v2/Video-MME-v2_fullset.json}"
  "${WORLDSENSE_DATASET_DIR:-${DATA_ROOT:-data/benchmarks}/WorldSense/worldsense.json}"
  "${VIDEOMMMU_DATASET_DIR:-${DATA_ROOT:-data/benchmarks}/VideoMMMU/videommmu.json}"
  "${VIDEO_HOLMES_DATASET_DIR:-${DATA_ROOT:-data/benchmarks}/Video-Holmes/test_Video-Holmes.json}"
  "${LVBENCH_DATASET_DIR:-${DATA_ROOT:-data/benchmarks}/LVBench/lvbench.json}"
  "${TVGBENCH_DATASET_DIR:-${DATA_ROOT:-data/benchmarks}/TVGBench/tvgbench.json}"
)

max_turn_list=(5 0 5 5 5 5 5 5 0)
frames_limit_list=(96 96 96 96 96 96 96 96 96)
max_new_tokens_list=(4096 4096 4096 4096 4096 4096 4096 4096 4096)
dataset_env_list=(
  "V_STAR_VIDEO_DIR=${V_STAR_VIDEO_DIR:-${DATA_ROOT:-data/benchmarks}/V-STaR/videos}"
  ""
  ""
  ""
  ""
  ""
  ""
  ""
  "TVGBENCH_VIDEO_ROOT=${TVGBENCH_VIDEO_ROOT:-${DATA_ROOT:-data/benchmarks}/TVGBench/videos}"
)

openo3_run "$@"
