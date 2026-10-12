#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

PROJECT_NAME="${PROJECT_NAME:-$(basename "${repo_root}")}"
RAY_PORT=6379
RAY_DASHBOARD_HOST="0.0.0.0"
RAY_DASHBOARD_PORT=8265
RAY_OBJECT_STORE_MEMORY_BYTES=21474836480
RAY_MEMORY_USAGE_THRESHOLD="0.99"
NUM_GPUS=8
RUN_NNODES=2
RUN_NPROC_PER_NODE=8
INSTALL_ENV="${INSTALL_ENV:-0}"
RAY_LOG_DIR="${RAY_LOG_DIR:-${repo_root}/logs/ray}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${repo_root}/logs}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/start_openo3_train.sh}"

usage() {
    cat <<EOF
Usage:
  $0 head [--install]
  $0 worker <HEAD_IP[:PORT]> [--install]
  $0 train [TRAIN_SCRIPT]
  $0 status [HEAD_IP[:PORT]]
  $0 stop

Defaults:
  RAY_PORT=${RAY_PORT}
  RAY_OBJECT_STORE_MEMORY_BYTES=${RAY_OBJECT_STORE_MEMORY_BYTES}
  RAY_MEMORY_USAGE_THRESHOLD=${RAY_MEMORY_USAGE_THRESHOLD}
  NUM_GPUS=${NUM_GPUS}
  NNODES=${RUN_NNODES}
  NPROC_PER_NODE=${RUN_NPROC_PER_NODE}
  VENV_PATH=$(default_venv_path)

Typical flow:
  # run on both nodes first
  bash run_2nodes.sh stop

  # head node
  bash run_2nodes.sh head

  # worker node, copy the address printed by head
  bash run_2nodes.sh worker <HEAD_IP>:${RAY_PORT}

  # back on head node
  bash run_2nodes.sh status
  bash run_2nodes.sh train
EOF
}

first_ipv4() {
    hostname -I 2>/dev/null | tr ' ' '\n' | awk '/^[0-9]+\./ {print; exit}'
}

default_node_ip() {
    echo "${RAY_NODE_IP_ADDRESS:-${MY_HOST_IP:-$(first_ipv4)}}"
}

default_venv_path() {
    echo "${VENV_PATH:-${UV_PROJECT_ENVIRONMENT:-${repo_root}/.venv}}"
}

normalize_ray_address() {
    local address="$1"
    if [[ -z "${address}" ]]; then
        return
    fi
    if [[ "${address}" == *:* ]]; then
        echo "${address}"
    else
        echo "${address}:${RAY_PORT}"
    fi
}

default_ray_address() {
    if [[ -n "${RAY_ADDRESS:-}" ]]; then
        normalize_ray_address "${RAY_ADDRESS}"
    else
        normalize_ray_address "$(default_node_ip)"
    fi
}

activate_env() {
    local venv_path
    venv_path="$(default_venv_path)"
    if [[ "${INSTALL_ENV}" == "1" ]]; then
        UV_PROJECT_ENVIRONMENT="${venv_path}" bash setup.sh
    elif [[ ! -d "${venv_path}" ]]; then
        echo "Missing venv: ${venv_path}" >&2
        echo "Set INSTALL_ENV=1 or VENV_PATH/UV_PROJECT_ENVIRONMENT." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "${venv_path}/bin/activate"
}

load_train_env() {
    activate_env
    # shellcheck disable=SC1091
    source scripts/train/env.sh
    # shellcheck disable=SC1091
    source scripts/train/before_train.sh
    export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARN}"
    export VLLM_ALLOW_RUNTIME_LORA_UPDATING="${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-true}"
    export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
    export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
    export RAY_memory_usage_threshold="${RAY_MEMORY_USAGE_THRESHOLD}"
}

cleanup_ray_start_processes() {
    pkill -f "[r]ay start --head" || true
    pkill -f "[r]ay start --address" || true
}

wait_for_ray_head() {
    local address="$1"
    local ray_pid="$2"
    local log_file="$3"
    local timeout="${RAY_START_TIMEOUT:-60}"
    local deadline=$((SECONDS + timeout))

    echo "Waiting for Ray head at ${address} ..."
    while (( SECONDS < deadline )); do
        if ray status --address="${address}" >/dev/null 2>&1; then
            echo "Ray head is reachable at ${address}"
            return 0
        fi
        if ! kill -0 "${ray_pid}" 2>/dev/null; then
            echo "Ray head exited before becoming reachable. Recent log:" >&2
            strings "${log_file}" | tail -n 80 >&2
            return 1
        fi
        sleep 2
    done

    echo "Ray head did not become reachable within ${timeout}s. Recent log:" >&2
    strings "${log_file}" | tail -n 80 >&2
    return 1
}

write_ray_log_header() {
    local log_file="$1"
    shift
    {
        echo "== run_2nodes ray start =="
        echo "date=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        echo "cwd=${repo_root}"
        echo "node_ip=$(default_node_ip)"
        printf "command:"
        printf " %q" "$@"
        echo
        echo
    } > "${log_file}"
}

start_head() {
    load_train_env
    mkdir -p "${RAY_LOG_DIR}"

    local node_ip
    node_ip="$(default_node_ip)"
    if [[ -z "${node_ip}" ]]; then
        echo "Cannot determine head node IP. Set RAY_NODE_IP_ADDRESS." >&2
        exit 1
    fi

    ray stop --force || true
    cleanup_ray_start_processes

    local cmd=(
        ray start --head
        --node-ip-address="${node_ip}"
        --port="${RAY_PORT}"
        --dashboard-host="${RAY_DASHBOARD_HOST}"
        --dashboard-port="${RAY_DASHBOARD_PORT}"
        --object-store-memory="${RAY_OBJECT_STORE_MEMORY_BYTES}"
        --num-gpus="${NUM_GPUS}"
        --block
    )
    write_ray_log_header "${RAY_LOG_DIR}/ray_head.log" "${cmd[@]}"
    nohup "${cmd[@]}" >> "${RAY_LOG_DIR}/ray_head.log" 2>&1 &
    local ray_pid=$!
    wait_for_ray_head "${node_ip}:${RAY_PORT}" "${ray_pid}" "${RAY_LOG_DIR}/ray_head.log"

    echo "Ray head started on ${node_ip}:${RAY_PORT}, num_gpus=${NUM_GPUS}"
    echo "Log: ${RAY_LOG_DIR}/ray_head.log"
    echo "Recommended worker command: bash run_2nodes.sh worker ${node_ip}:${RAY_PORT}"
    echo "Local IPs: $(hostname -I 2>/dev/null || true)"
}

start_worker() {
    local address="${1:-}"
    if [[ -z "${address}" ]]; then
        echo "worker requires HEAD_IP[:PORT]" >&2
        usage
        exit 1
    fi
    address="$(normalize_ray_address "${address}")"

    load_train_env
    mkdir -p "${RAY_LOG_DIR}"

    local node_ip
    node_ip="$(default_node_ip)"
    if [[ -z "${node_ip}" ]]; then
        echo "Cannot determine worker node IP. Set RAY_NODE_IP_ADDRESS." >&2
        exit 1
    fi

    ray stop --force || true
    pkill -f "[r]ay start --address" || true

    local cmd=(
        ray start
        --address="${address}"
        --node-ip-address="${node_ip}"
        --object-store-memory="${RAY_OBJECT_STORE_MEMORY_BYTES}"
        --num-gpus="${NUM_GPUS}"
        --block
    )
    write_ray_log_header "${RAY_LOG_DIR}/ray_worker.log" "${cmd[@]}"
    nohup "${cmd[@]}" >> "${RAY_LOG_DIR}/ray_worker.log" 2>&1 &

    echo "Ray worker joining ${address}, node_ip=${node_ip}, num_gpus=${NUM_GPUS}"
    echo "Log: ${RAY_LOG_DIR}/ray_worker.log"
}

start_train() {
    local train_script="${1:-${TRAIN_SCRIPT}}"
    if [[ ! -f "${train_script}" ]]; then
        echo "Missing train script: ${train_script}" >&2
        exit 1
    fi

    load_train_env
    mkdir -p "${TRAIN_LOG_DIR}"

    export NNODES="${RUN_NNODES}"
    export NPROC_PER_NODE="${RUN_NPROC_PER_NODE}"
    export SKIP_RAY_STOP=1
    export RAY_ADDRESS="${RAY_ADDRESS:-$(default_ray_address)}"

    local log_file="${TRAIN_LOG_DIR}/run_2nodes_$(date +"%Y%m%d-%H%M%S").log"
    echo "Starting ${train_script}"
    echo "NNODES=${NNODES}, NPROC_PER_NODE=${NPROC_PER_NODE}, RAY_ADDRESS=${RAY_ADDRESS}"
    echo "Log: ${log_file}"
    nohup bash "${train_script}" > "${log_file}" 2>&1 &
}

status_ray() {
    local address="${1:-}"
    activate_env
    if [[ -n "${address}" ]]; then
        ray status --address="$(normalize_ray_address "${address}")"
    else
        ray status --address="$(default_ray_address)"
    fi
}

stop_ray() {
    activate_env
    ray stop --force || true
    cleanup_ray_start_processes
    local ray_tmp_dir
    ray_tmp_dir="${RAY_TMPDIR:-$(python -c 'import tempfile; print(tempfile.gettempdir())')}"
    rm -f "${ray_tmp_dir}/ray/ray_current_cluster" 2>/dev/null || true
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

cmd="$1"
shift

case "${cmd}" in
    head)
        [[ "${1:-}" == "--install" ]] && INSTALL_ENV=1
        start_head
        ;;
    worker)
        address="${1:-}"
        shift || true
        [[ "${1:-}" == "--install" ]] && INSTALL_ENV=1
        start_worker "${address}"
        ;;
    train)
        start_train "${1:-${TRAIN_SCRIPT}}"
        ;;
    status)
        status_ray "${1:-}"
        ;;
    stop)
        stop_ray
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown command: ${cmd}" >&2
        usage
        exit 1
        ;;
esac
