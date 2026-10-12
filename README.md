# Open-o3-Video-v2

**Seek, Ground, and Reason with Verifiable Spatio-Temporal Evidence**

Jiahao Meng<sup>*</sup>, Zhongyuan Bao<sup>*</sup>, Yue Tan, Haiquan Wen, Renrui Zhang, Anna Wang, Jason Li, Yanwei Fu, Yunhai Tong, and Ming-Hsuan Yang.

<sup>*</sup> Equal contribution.

[[Project Page](https://msalab-pku.github.io/projects/Open-o3-Video-v2)] · [[4B Model](https://modelscope.cn/models/FDUBay/Open-o3-Video-v2-4B)] · [[Training Annotations](https://modelscope.cn/datasets/FDUBay/Open-o3-Video-v2-data)]

Open-o3-Video-v2 is an agentic video reasoning system. It inspects visual evidence, grounds objects and events, and reasons across multiple turns. This repository provides supervised fine-tuning (SFT), reinforcement learning (RL), and evaluation code.

![Open-o3-Video-v2 overview](assets/teaser.png)

## Method

The training data includes **STGR-v2-CoT-135K** supervised examples and **STGR-v2-RL-13K** for policy optimization. The RL agent uses a `crop_video` tool and rewards for answers, format, grounding, and tool use. See the [project page](https://msalab-pku.github.io/projects/Open-o3-Video-v2) for the method and results.

![Data construction](assets/data.png)
![Training pipeline](assets/method.png)

## Quick Start

### 1. Environment

Use Linux with NVIDIA GPUs and a CUDA toolkit compatible with PyTorch 2.8. Install [uv](https://docs.astral.sh/uv/) and FFmpeg, including `ffprobe`. The stages use separate environments:

~~~bash
bash SFT/scripts/qwen3vl/install_qwen3vl.sh
bash RL/setup.sh
(cd Eval && uv sync)
~~~

See [SFT](SFT/README.md), [RL](RL/README.md), and [Eval](Eval/README.md) for stage-specific requirements.

No API key is stored in this repository. If a model hub requires authentication, sign in with its CLI on the host. W&B is disabled by default. To enable experiment tracking, run `wandb login` in the stage environment and set `WANDB_MODE=online`.

### 2. Annotations and videos

The [ModelScope release](https://modelscope.cn/datasets/FDUBay/Open-o3-Video-v2-data) contains JSON annotations. It does **not** contain source videos. Obtain each referenced video from the linked datasets or their underlying public corpora. Follow the source licenses and keep the paths recorded in JSON.

~~~bash
source SFT/scripts/qwen3vl/.venv/bin/activate
modelscope download --dataset FDUBay/Open-o3-Video-v2-data --local_dir ./data/annotations
export DATA_ROOT="$PWD/data/annotations"
export MEDIA_ROOT="$PWD/data/media"
~~~

The annotation snapshot has eight files under `SFT/json_data/` and three under `RL/`. The expected video layout is:

~~~text
data/
├── annotations/
│   ├── SFT/json_data/*.json
│   └── RL/*.json
└── media/
    ├── videor1/LLaVA-Video-178K/...     # Open-o3 Video records
    ├── timerft/...
    ├── videoespresso/videos/...
    ├── videoespresso/kfs/...
    ├── tvg_r1/...
    ├── gqa/...
    ├── treevgr/...
    ├── stgr/plm/videos/...
    ├── stgr/plm/kfs/...
    ├── stgr/temporal_grounding/videos/...
    ├── stgr/temporal_grounding/kfs/...
    ├── seeker/ytb_*.mp4                  # Video-o3 SFT
    ├── seeker/pnorm2/...                 # Video-o3 RL
    ├── timelens-100k/videos/...          # TimeLens-100K
    ├── cgbench/...                       # CG-Bench
    ├── longvt/...                        # LongVT
    ├── caprl/2_3_m_academic_v0_1/...    # CapRL
    └── omni100k/...                      # OmniVideo-100K
~~~

Keep each source's subdirectories and key-frame files. Do not flatten downloaded archives.

| Source | Annotation path | Place files under |
| --- | --- | --- |
| [Open-o3 Video](https://huggingface.co/datasets/marinero4972/Open-o3-Video) | `video_path`, `image_path`, `key_frames[].path` | `data/media/` with the source directories above |
| [Video-o3](https://github.com/MCG-NJU/Video-o3) | `videos[].url` (SFT), `video` (RL) | `data/media/seeker/` |
| [TimeLens-100K](https://huggingface.co/datasets/TencentARC/TimeLens-100K) | `timelens-100k/videos/...` (SFT), `videos/...` (RL) | `data/media/timelens-100k/` |
| [CG-Bench](https://github.com/CG-Bench/CG-Bench) | `videos[].url` | `data/media/cgbench/` |
| [LongVT](https://github.com/EvolvingLMMs-Lab/LongVT) | `videos[].url` | `data/media/longvt/` |
| [CapRL](https://huggingface.co/datasets/internlm/CapRL-Video-178K) | `video_path` | `data/media/caprl/` |
| [OmniVideo-100K](https://huggingface.co/datasets/MiG-NJU/OmniVideo-100K) | `video_path` | `data/media/omni100k/` |

The SFT converter checks every media path before extraction. For a different local layout, pass a JSON manifest. Each value is relative to the manifest file. For example, save this as `data/media_manifest.json`:

~~~json
{"ytb_example.mp4": "media/seeker/ytb_example.mp4"}
~~~

~~~bash
python SFT/prepare_data.py --data-root "$DATA_ROOT" --media-root "$MEDIA_ROOT" \
  --media-manifest data/media_manifest.json --check-only
python SFT/prepare_data.py --data-root "$DATA_ROOT" --media-root "$MEDIA_ROOT" \
  --media-manifest data/media_manifest.json --output-dir SFT/prepared
python RL/prepare_data.py --data-root "$DATA_ROOT" --output-dir RL/data
~~~

Omit `--media-manifest` when the videos follow the table. The converter reports unresolved paths with their source file and row index.

### 3. Supervised fine-tuning

Download the [Qwen3-VL-4B-Instruct](https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct) base model and run the retained 135K recipe:

~~~bash
modelscope download --model Qwen/Qwen3-VL-4B-Instruct --local_dir ./models/Qwen3-VL-4B-Instruct
MODEL_PATH="$PWD/models/Qwen3-VL-4B-Instruct" \
PREPARED_DIR="$PWD/SFT/prepared" \
OUTPUT_DIR="$PWD/SFT/checkpoints/openo3-video-v2-4b-sft" \
NPROC_PER_NODE=8 bash SFT/train.sh
~~~

The 8B recipe uses [Qwen3-VL-8B-Instruct](https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct) and `SFT/configs/qwen3vl/open-o3-video/qwen3_sft_mix_multi_image_135k_8b.yaml`. See [SFT/README.md](SFT/README.md).

### 4. Reinforcement learning

RL starts from an SFT model in Transformers format. Download the released [4B SFT model](https://modelscope.cn/models/FDUBay/Open-o3-Video-v2-4B-SFT), or export a checkpoint from step 3. The default recipe uses eight GPUs and reads the split created above:

~~~bash
modelscope download --model FDUBay/Open-o3-Video-v2-4B-SFT \
  --local_dir ./models/openo3-video-v2-4b-sft
MODEL_PATH="$PWD/models/openo3-video-v2-4b-sft" \
DATA_ROOT="$PWD/data/annotations" \
MEDIA_ROOT="$PWD/data/media" \
bash RL/scripts/start_openo3_train.sh
~~~

See [RL/README.md](RL/README.md) for the two-node Ray launcher and path overrides. The released GRPO reward weights and optimizer settings remain in the training script.

### 5. Evaluation

Download the [4B model](https://modelscope.cn/models/FDUBay/Open-o3-Video-v2-4B) locally. Obtain each benchmark's annotations and media from its original release.

~~~bash
cd Eval
modelscope download --model FDUBay/Open-o3-Video-v2-4B --local_dir ./models/Open-o3-Video-v2-4B
VIDEO_MME_DATASET_DIR="$PWD/data/benchmarks/Video-MME/Video-MME.json" \
VAL_DATASET_SAMPLE=1 bash scripts/eval_openo3video_v2_4b.sh video_mme
~~~

Remove `VAL_DATASET_SAMPLE` for a full run. See [Eval/README.md](Eval/README.md) for benchmark paths and tool settings.

## Models

| Release | Link |
| --- | --- |
| Open-o3-Video-v2-4B | [ModelScope](https://modelscope.cn/models/FDUBay/Open-o3-Video-v2-4B) |
| Open-o3-Video-v2-8B | [ModelScope](https://modelscope.cn/models/FDUBay/Open-o3-Video-v2-8B) |
| Open-o3-Video-v2-4B-SFT | [ModelScope](https://modelscope.cn/models/FDUBay/Open-o3-Video-v2-4B-SFT) |
| Open-o3-Video-v2-8B-SFT | [ModelScope](https://modelscope.cn/models/FDUBay/Open-o3-Video-v2-8B-SFT) |

## Results

The 4B model achieves **40.4% mAM and 56.3% mLGM on V-STaR**. The project page also reports 53.2 on LVBench, 48.0 on Video-Holmes, and 49.8 mIoU on Charades-TimeLens.

![V-STaR results](assets/vstar-results-table.png)
![General video benchmark results](assets/general-results-table.png)

## Demos

<table>
  <tr><td><img src="assets/case-03.gif" alt="Reasoning demo"></td><td><img src="assets/case-05.gif" alt="Reasoning demo"></td></tr>
  <tr><td><img src="assets/case-04.gif" alt="Reasoning demo"></td><td><img src="assets/case-06.gif" alt="Reasoning demo"></td></tr>
  <tr><td><img src="assets/case-01.gif" alt="Reasoning demo"></td><td><img src="assets/case-02.gif" alt="Reasoning demo"></td></tr>
</table>

## License and Citation

The project code is released under [Apache-2.0](LICENSE). The bundled verl code retains its [license](RL/verl/LICENSE) and [notice](RL/verl/Notice.txt). Source videos and benchmarks retain their original licenses.

~~~bibtex
@article{meng2026openo3v2,
  title={Open-o3-Video-v2: Seek, Ground, and Reason with Verifiable Spatio-Temporal Evidence},
  author={Meng, Jiahao and Bao, Zhongyuan and Tan, Yue and Wen, Haiquan and Zhang, Renrui and Wang, Anna and Li, Jason and Fu, Yanwei and Tong, Yunhai and Yang, Ming-Hsuan},
  journal={arXiv preprint arXiv:2610.xxxxx},
  year={2026}
}
~~~

## Acknowledgements

This work builds on [MS-SWIFT](https://github.com/modelscope/ms-swift), [verl](https://github.com/volcengine/verl), and the video datasets linked above.
