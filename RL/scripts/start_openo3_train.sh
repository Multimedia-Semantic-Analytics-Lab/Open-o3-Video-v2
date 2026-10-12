#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65535

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

source "${repo_root}/scripts/train/env.sh"
source "${repo_root}/scripts/train/before_train.sh"

maybe_stop_ray() {
    if [[ "${SKIP_RAY_STOP:-0}" == "1" ]]; then
        echo "SKIP_RAY_STOP=1, reuse existing Ray cluster."
        return
    fi
    ray stop "$@"
}

PROJECT_NAME="${PROJECT_NAME:-openo3-video-v2-agentloop}"
nnodes="${NNODES}"
n_gpus_per_node="${NPROC_PER_NODE}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-GRPO-openo3v2-NNODES${nnodes}-NGPU${n_gpus_per_node}}"

CKPT_SAVE_DIR="${OUT_CKPT_ROOT}/${EXPERIMENT_NAME}"
LOG_SAVE_DIR="${REPO_ROOT}/logs/${EXPERIMENT_NAME}/$(date +"%Y%m%d-%H")"

export PROJECT_NAME EXPERIMENT_NAME CKPT_SAVE_DIR LOG_SAVE_DIR

PROJECT_DIR="${repo_root}/verl"
CONFIG_PATH="${PROJECT_DIR}/examples/video_tools/config"
PYTHON_BIN="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
: "${MODEL_PATH:?Set MODEL_PATH to an Open-o3-Video-v2 SFT checkpoint}"
: "${DATA_ROOT:?Set DATA_ROOT to the downloaded ModelScope dataset root}"
: "${MEDIA_ROOT:?Set MEDIA_ROOT to the root of your downloaded videos}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing Python environment: ${PYTHON_BIN}" >&2
    exit 2
fi

OPENO3_TRAIN_JSON="${OPENO3_TRAIN_JSON:-${repo_root}/data/rl_train.json}"
VIDEOO3_TRAIN_JSON="${VIDEOO3_TRAIN_JSON:-${DATA_ROOT}/RL/video-o3_with_grounded_cot_7891.json}"
TIMELENS_TRAIN_JSON="${TIMELENS_TRAIN_JSON:-${DATA_ROOT}/RL/timelens_32k_stgr_format_temporal_qa_ge120s.json}"
OPENO3_VAL_JSON="${OPENO3_VAL_JSON:-${repo_root}/data/rl_val.json}"

OPENO3_VIDEO_ROOT="${OPENO3_VIDEO_ROOT:-${MEDIA_ROOT}}"
TIMELENS_VIDEO_ROOT="${TIMELENS_VIDEO_ROOT:-${MEDIA_ROOT}/timelens-100k}"
VIDEOO3_VIDEO_ROOT="${VIDEOO3_VIDEO_ROOT:-${MEDIA_ROOT}/seeker}"
CGBENCH_VIDEO_ROOT="${CGBENCH_VIDEO_ROOT:-${MEDIA_ROOT}/cgbench}"

CUSTOM_DATASET_PATH="${repo_root}/verl/custom_datasets/openo3_dataset.py"
CUSTOM_REWARD_PATH="${repo_root}/verl/custom_rewards/openo3_reward.py"
TOOL_CONFIG_PATH="${PROJECT_DIR}/examples/video_tools/config/openo3_video_tool_config.yaml"
AGENT_LOOP_CONFIG_PATH="${PROJECT_DIR}/examples/video_tools/config/openo3_loop.yaml"

OPENO3_RATIO=0.4
VIDEOO3_RATIO=0.5
TIMELENS_RATIO=0.1
CGBENCH_RATIO=0.0
MIX_RATIO_OVERSAMPLE=false

for input in "${OPENO3_TRAIN_JSON}" "${VIDEOO3_TRAIN_JSON}" "${TIMELENS_TRAIN_JSON}" "${OPENO3_VAL_JSON}"; do
    [[ -f "${input}" ]] || { echo "Missing training annotation: ${input}" >&2; exit 2; }
done

maybe_stop_ray

PYTHON_BIN_DIR="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)"
case ":${PATH}:" in
    *":${PYTHON_BIN_DIR}:"*) ;;
    *) PATH="${PYTHON_BIN_DIR}:${PATH}" ;;
esac
export PATH

MAX_PROMPT_LENGTH=28672
MAX_RESPONSE_LENGTH=8192
MAX_MODEL_LEN="36864"

ANS_REWARD_WEIGHT="${ANS_REWARD_WEIGHT:-0.8}"
FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.2}"
TOOL_REWARD_WEIGHT="${TOOL_REWARD_WEIGHT:-1.0}"
TOOL_NO_CALL_REWARD="${TOOL_NO_CALL_REWARD:-0.3}"
TOOL_CALL_PENALTY="${TOOL_CALL_PENALTY:-0.1}"

mkdir -p "${CKPT_SAVE_DIR}" "${LOG_SAVE_DIR}" "${WANDB_DIR}" "${WANDB_ARTIFACT_DIR}" "${TMPDIR}"

cd "${PROJECT_DIR}"

"${PYTHON_BIN}" -u -m verl.trainer.main_ppo \
    --config-path="${CONFIG_PATH}" \
    --config-name=openo3_multiturn_grpo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=true \
    data.train_files="['${OPENO3_TRAIN_JSON}','${VIDEOO3_TRAIN_JSON}','${TIMELENS_TRAIN_JSON}']" \
    data.val_files="['${OPENO3_VAL_JSON}']" \
    data.train_batch_size=32 \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.train_max_samples="${UNIFIED_TRAIN_MAX_SAMPLES}" \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    data.dataloader_num_workers=0 \
    data.custom_cls.path="${CUSTOM_DATASET_PATH}" \
    data.custom_cls.name=OpenO3VideoDataset \
    +data.openo3_files="['${OPENO3_TRAIN_JSON}']" \
    +data.videoo3_files="['${VIDEOO3_TRAIN_JSON}']" \
    +data.timelens_files="['${TIMELENS_TRAIN_JSON}']" \
    +data.cgbench_files="[]" \
    +data.val_openo3_files="['${OPENO3_VAL_JSON}']" \
    +data.val_videoo3_files="[]" \
    +data.val_timelens_files="[]" \
    +data.val_cgbench_files="[]" \
    +data.open_o3_video_root="${OPENO3_VIDEO_ROOT}" \
    +data.timelens_video_root="${TIMELENS_VIDEO_ROOT}" \
    +data.base_media_dir="${VIDEOO3_VIDEO_ROOT}" \
    +data.cgbench_video_root="${CGBENCH_VIDEO_ROOT}" \
    +data.target_fps=1.0 \
    +data.max_video_frames=32 \
    +data.frame_max_height=224 \
    +data.train_max_samples_random="${UNIFIED_TRAIN_MAX_SAMPLES_RANDOM}" \
    +data.train_max_samples_seed="${UNIFIED_TRAIN_MAX_SAMPLES_SEED}" \
    +data.openo3_max_samples=-1 \
    +data.videoo3_max_samples=-1 \
    +data.timelens_max_samples=-1 \
    +data.cgbench_max_samples=-1 \
    +data.openo3_mix_ratio="${OPENO3_RATIO}" \
    +data.videoo3_mix_ratio="${VIDEOO3_RATIO}" \
    +data.timelens_mix_ratio="${TIMELENS_RATIO}" \
    +data.cgbench_mix_ratio="${CGBENCH_RATIO}" \
    +data.mix_ratio_oversample="${MIX_RATIO_OVERSAMPLE}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.24 \
    actor_rollout_ref.actor.grad_clip=5.0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    +actor_rollout_ref.rollout.limit_images=384 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.agent.default_agent_loop=openo3_tool_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG_PATH}" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    reward_model.use_reward_loop=True \
    reward_model.reward_manager=naive \
    custom_reward_function.path="${CUSTOM_REWARD_PATH}" \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.extra_info.ans_reward_weight="${ANS_REWARD_WEIGHT}" \
    +custom_reward_function.reward_kwargs.extra_info.format_reward_weight="${FORMAT_REWARD_WEIGHT}" \
    +custom_reward_function.reward_kwargs.extra_info.tool_reward_weight="${TOOL_REWARD_WEIGHT}" \
    +custom_reward_function.reward_kwargs.extra_info.tool_no_call_reward="${TOOL_NO_CALL_REWARD}" \
    +custom_reward_function.reward_kwargs.extra_info.tool_call_penalty="${TOOL_CALL_PENALTY}" \
    trainer.logger="['console','wandb']" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    trainer.test_freq=10000 \
    trainer.save_freq=50 \
    trainer.default_local_dir="${CKPT_SAVE_DIR}" \
    2>&1 | tee "${LOG_SAVE_DIR}/train_log.txt"
