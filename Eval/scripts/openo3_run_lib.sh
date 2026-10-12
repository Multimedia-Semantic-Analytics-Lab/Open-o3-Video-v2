#!/usr/bin/env bash

openo3_is_truthy() {
  case "$(printf '%s' "${1:-0}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

openo3_prepare_env() {
  local default_nproc_per_node="${1:-8}"
  local script_dir
  local repo_root
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd "${script_dir}/.." && pwd)"
  cd "${repo_root}"

  export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${repo_root}/.venv}"
  if openo3_is_truthy "${ACTIVATE_VENV:-0}" && [[ -f "${UV_PROJECT_ENVIRONMENT}/bin/activate" ]]; then
    source "${UV_PROJECT_ENVIRONMENT}/bin/activate"
  fi
  if openo3_is_truthy "${ADD_NVIDIA_LIB_PATH:-0}"; then
    local python_version
    python_version="$("${UV_PROJECT_ENVIRONMENT}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    export LD_LIBRARY_PATH="${UV_PROJECT_ENVIRONMENT}/lib/python${python_version}/site-packages/nvidia/nvjitlink/lib:${UV_PROJECT_ENVIRONMENT}/lib/python${python_version}/site-packages/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"
  fi

  if ! openo3_is_truthy "${SKIP_UV_SYNC:-0}" && ! openo3_is_truthy "${DRY_RUN:-0}"; then
    uv sync
  fi

  if [[ -z "${MASTER_PORT:-}" ]]; then
    export MASTER_PORT=29500
  fi

  export OPENO3_DEBUG="${OPENO3_DEBUG:-1}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export NPROC_PER_NODE="${NPROC_PER_NODE:-${default_nproc_per_node}}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export OPENO3_EVAL_VSTAR_EVAL_CUDA_VISIBLE_DEVICES="${OPENO3_EVAL_VSTAR_EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
}

openo3_should_run_dataset() {
  local dataset_name="$1"
  shift
  if [[ "$#" -eq 0 ]]; then
    return 0
  fi
  local selected
  for selected in "$@"; do
    [[ "${dataset_name}" == "${selected}" ]] && return 0
  done
  return 1
}

openo3_apply_extra_env() {
  local extra_env="$1"
  [[ -n "${extra_env}" ]] || return 0
  local key="${extra_env%%=*}"
  local value="${extra_env#*=}"
  if [[ -z "${!key-}" ]]; then
    export "${key}=${value}"
  fi
}

openo3_run() {
  local selected_datasets=("$@")
  if [[ -n "${ONLY_DATASETS:-}" ]]; then
    read -r -a selected_datasets <<< "${ONLY_DATASETS//,/ }"
  fi

  local n="${#dataset_names[@]}"
  if [[ "${#dataset_dirs[@]}" -ne "${n}" \
    || "${#max_turn_list[@]}" -ne "${n}" \
    || "${#frames_limit_list[@]}" -ne "${n}" \
    || "${#max_new_tokens_list[@]}" -ne "${n}" \
    || "${#dataset_env_list[@]}" -ne "${n}" ]]; then
    echo "[Open-o3 Eval] dataset array length mismatch" >&2
    exit 1
  fi

  local result_stamp="${RESULT_STAMP:-$(date +"%Y%m%d-%H%M%S")}"
  local result_root="${RESULT_ROOT:-./openo3_eval_result/${run_model_name}}"
  local default_fps="${FPS:-1.0}"
  local default_frames_limit="${FRAMES_LIMIT:-96}"
  local default_max_new_tokens="${MAX_NEW_TOKENS:-${model_default_max_new_tokens:-4096}}"
  local default_temperature="${TEMPERATURE:-${model_default_temperature:-0.01}}"
  local default_top_p="${TOP_P:-${model_default_top_p:-0.001}}"
  local default_top_k="${TOP_K:-${model_default_top_k:-1}}"
  local default_repetition_penalty="${REPETITION_PENALTY:-${model_default_repetition_penalty:-1.0}}"

  echo "Evaluating model: ${run_model_name}"
  echo "Model config key: ${model_name}"
  if [[ -n "${checkpoint_label:-}" ]]; then
    echo "Checkpoint: ${checkpoint_label}"
  fi
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "RESULT_STAMP=${result_stamp}"

  local i
  for i in "${!dataset_names[@]}"; do
    local dataset_name="${dataset_names[$i]}"
    if ! openo3_should_run_dataset "${dataset_name}" "${selected_datasets[@]}"; then
      continue
    fi

    local dataset_dir="${dataset_dirs[$i]}"
    local max_turn="${MAX_TURN:-${max_turn_list[$i]}}"
    local frames_limit="${frames_limit_list[$i]:-${default_frames_limit}}"
    local max_new_tokens="${max_new_tokens_list[$i]:-${default_max_new_tokens}}"
    local extra_env="${dataset_env_list[$i]:-}"
    local use_tools="${USE_TOOLS:-true}"
    local log_name="${dataset_name}"
    local extra_args=()

    openo3_apply_extra_env "${extra_env}"

    if [[ -n "${VAL_DATASET_SAMPLE:-}" ]]; then
      extra_args+=(--val_dataset_sample "${VAL_DATASET_SAMPLE}")
    fi
    if [[ -n "${ANSWER_PARSER:-}" ]]; then
      extra_args+=(--answer_parser "${ANSWER_PARSER}")
    fi
    if openo3_is_truthy "${REUSE:-${model_default_reuse:-0}}"; then
      extra_args+=(--reuse)
    fi

    local result_path="${RESULT_PATH:-${result_root}/openo3_eval/${log_name}_fullset_${result_stamp}.jsonl}"
    local log_path="${LOG_PATH:-${result_root}/logs/eval_${run_model_name}_${log_name}_fullset_${result_stamp}.log}"

    echo "Running ${dataset_name}"
    echo "Dataset dir: ${dataset_dir}"
    echo "Result path: ${result_path}"
    echo "Log path: ${log_path}"
    echo "fps=${default_fps}, frames_limit=${frames_limit}, max_turn=${max_turn}, max_new_tokens=${max_new_tokens}"

    local cmd=(
      uv run --no-sync openo3-eval eval-openo3
      --model "${model_name}"
      --dataset_pp "${dataset_name}"
      --dataset_dir "${dataset_dir}"
      --result_path "${result_path}"
      --fps "${default_fps}"
      --frames_limit "${frames_limit}"
      --max_turn "${max_turn}"
      --max_new_tokens "${max_new_tokens}"
      --temperature "${default_temperature}"
      --top_p "${default_top_p}"
      --top_k "${default_top_k}"
      --repetition_penalty "${default_repetition_penalty}"
      --do_sample "${DO_SAMPLE:-false}"
      --use_tools "${use_tools}"
      "${extra_args[@]}"
    )

    if openo3_is_truthy "${DRY_RUN:-0}"; then
      printf 'Command:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
    else
      mkdir -p "$(dirname "${result_path}")" "$(dirname "${log_path}")"
      "${cmd[@]}" > "${log_path}" 2>&1
    fi
  done
}
