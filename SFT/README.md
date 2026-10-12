# Supervised Fine-Tuning

This directory contains the 4B and 8B MS-SWIFT Megatron recipes. See [Data](../DATA.md) for annotation sources and video layout, and the [root README](../README.md) for model downloads. Run the commands below from the repository root.

## Environment and data

```bash
bash SFT/scripts/qwen3vl/install_qwen3vl.sh
python SFT/prepare_data.py --data-root data/annotations --media-root data/media --check-only
python SFT/prepare_data.py --data-root data/annotations --media-root data/media --output-dir SFT/prepared
```

The installer creates `SFT/scripts/qwen3vl/.venv` with Python 3.12. It requires Linux, CUDA 12.6, `nvcc`, a C++ compiler, and FFmpeg. The converter writes eight JSONL files and a frame cache. It preserves turn order and key frames. Full videos use 1 fps; coarse, medium, and fine crops use 1, 2, and 4 fps. Each observation has at most 96 frames. Use `--media-manifest` if your video paths differ from the layout in [Data](../DATA.md).

## Train

```bash
export PREPARED_DIR="$PWD/SFT/prepared"
export MODEL_PATH="$PWD/models/Qwen3-VL-4B-Instruct"
export OUTPUT_DIR="$PWD/SFT/checkpoints/v2-4b-sft"
NPROC_PER_NODE=8 bash SFT/train.sh
```

For 8B, use the 8B base model and config:

```bash
export MODEL_PATH="$PWD/models/Qwen3-VL-8B-Instruct"
export OUTPUT_DIR="$PWD/SFT/checkpoints/v2-8b-sft"
NPROC_PER_NODE=8 bash SFT/train.sh configs/qwen3vl/open-o3-video/qwen3_sft_mix_multi_image_135k_8b.yaml
```

| Setting | 4B | 8B |
| --- | ---: | ---: |
| Micro batch | 1 | 1 |
| Global batch | 64 | 64 |
| Maximum sequence length | 64,000 | 64,000 |
| Learning rate | 5e-6 | 5e-6 |
| Epochs | 1 | 1 |

Both recipes save every 500 steps. `DRY_RUN=1` prints the resolved config. For multiple nodes, set `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT`. W&B is disabled by default. To enable it, run `wandb login` in the SFT environment and set `WANDB_MODE=online`.

Training produces a Megatron checkpoint. Export it to Transformers format before RL, following the [MS-SWIFT export guide](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Megatron-SWIFT/Quick-start.md).
