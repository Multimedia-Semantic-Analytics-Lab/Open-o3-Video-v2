# Reinforcement Learning

This directory contains the GRPO recipe, `crop_video` tool, multi-turn agent loop, and bundled verl source. See [Data](../DATA.md) for annotations and videos, and the [root README](../README.md) for an SFT checkpoint. Run these commands from `RL/`.

## Single-node run

```bash
bash setup.sh
python prepare_data.py --data-root ../data/annotations --output-dir data
export DATA_ROOT="$PWD/../data/annotations"
export MEDIA_ROOT="$PWD/../data/media"
export MODEL_PATH="$PWD/../models/openo3-video-v2-4b-sft"
bash scripts/start_openo3_train.sh
```

The setup uses Python 3.11 and requires CUDA, `nvcc`, and a C++ compiler. Data preparation selects 100 Video-o3 examples for validation with seed 42 and writes disjoint `data/videoo3_train.json` and `data/videoo3_val.json` files. The entrypoint reads Open-o3 and TimeLens directly from `DATA_ROOT/RL/`. Training samples from the three sources at a 0.4/0.5/0.1 ratio, with a maximum of 12,800 examples by default (approximately 13K in the paper). The released JSONs contain larger source pools. All media roots can be overridden with `OPENO3_VIDEO_ROOT`, `VIDEOO3_VIDEO_ROOT`, and `TIMELENS_VIDEO_ROOT`.

The default run uses eight GPUs, batch size 32, prompt length 28,672, response length 8,192, rollout count 8, and actor learning rate 1e-6. Training-example selection uses a fixed seed of 111 by default (`UNIFIED_TRAIN_MAX_SAMPLES_SEED`); dataloader order and stochastic rollouts are not fixed by that setting. Validation before training is disabled. The 10,000-step interval exceeds this one-epoch run, so the 100-example validation set is evaluated at the final step. Checkpoints are saved every 50 steps under `checkpoints/`. The training script holds the reward weights and remaining settings. Use `OUT_CKPT_ROOT` to change the checkpoint location. W&B is disabled by default. To enable it, run `wandb login` in the RL environment and set `WANDB_MODE=online`.

## Two-node run

`run_2nodes.sh` starts a Ray head and worker, then launches the same training entrypoint with eight GPUs per node. Install the environment and provide the same code, data, and model paths on both nodes.

```bash
# Head node
bash run_2nodes.sh head

# Worker node; use the address printed by the head
bash run_2nodes.sh worker HEAD_IP:6379

# Head node, after the worker joins
bash run_2nodes.sh status
bash run_2nodes.sh train
```

Set `RAY_NODE_IP_ADDRESS` if automatic IP detection is wrong. Run `bash run_2nodes.sh stop` on each node to stop Ray. The `train` action sets `NNODES=2`, `NPROC_PER_NODE=8`, and `SKIP_RAY_STOP=1`.

## Code map

- `verl/custom_datasets/openo3_dataset.py`: training data mixture.
- `verl/custom_agent_loops/openo3_tool_agent_loop.py`: multi-turn inspection.
- `verl/custom_tools/openo3_video_agent_tools.py`: `crop_video`.
- `verl/custom_rewards/openo3_reward.py`: reward calculation.
- `verl/examples/video_tools/config/`: GRPO and tool settings.

The bundled verl retains its [license](verl/LICENSE) and [notice](verl/Notice.txt).
