# Data

Run these commands from the repository root.

## JSON annotations

Download the [released annotations](https://modelscope.cn/datasets/FDUBay/Open-o3-Video-v2-data) from ModelScope. The CLI is included in the SFT environment.

```bash
modelscope download --dataset FDUBay/Open-o3-Video-v2-data --local_dir data/annotations
```

The download contains `data/annotations/SFT/json_data/*.json` and `data/annotations/RL/*.json`. It contains no video files.

The paper's **STGR-v2-RL-13K** denotes the approximately 13K examples sampled for RL training. The released RL JSON files contain larger source pools. The default recipe selects up to 12,800 examples from the Open-o3, Video-o3, and TimeLens pools at a 40/50/10 ratio.

## Videos and images

Download the source media separately. Keep the directories referenced by the JSON files.

| Source | Download | Directory under `data/media/` |
| --- | --- | --- |
| Open-o3 Video | [Dataset](https://huggingface.co/datasets/marinero4972/Open-o3-Video) | `videor1/`, `timerft/`, `videoespresso/`, `tvg_r1/`, `gqa/`, `treevgr/`, `stgr/` |
| Video-o3 | [Project](https://github.com/MCG-NJU/Video-o3) | `seeker/ytb_*.mp4`, `seeker/pnorm2/` |
| TimeLens-100K | [Dataset](https://huggingface.co/datasets/TencentARC/TimeLens-100K) | `timelens-100k/videos/` |
| CG-Bench | [Project](https://github.com/CG-Bench/CG-Bench) | `cgbench/` |
| LongVT | [Project](https://github.com/EvolvingLMMs-Lab/LongVT) | `longvt/` |
| CapRL | [Dataset](https://huggingface.co/datasets/internlm/CapRL-Video-178K) | `caprl/2_3_m_academic_v0_1/` |
| OmniVideo-100K | [Dataset](https://huggingface.co/datasets/MiG-NJU/OmniVideo-100K) | `omni100k/` |

For example, a Video-o3 RL path such as `pnorm2/clip.mp4` resolves to `data/media/seeker/pnorm2/clip.mp4`. Keep key-frame directories such as `videoespresso/kfs/` and `stgr/plm/kfs/`. For a different layout, use the SFT converter's `--media-manifest` option or the [RL media-root settings](RL/README.md).

## Prepare training inputs

```bash
python SFT/prepare_data.py --data-root data/annotations --media-root data/media --output-dir SFT/prepared
python RL/prepare_data.py --data-root data/annotations --output-dir RL/data
```

The SFT script resolves media paths, extracts timestamped frames, and writes eight multi-image JSONL files. Use `--check-only` to validate paths without extracting frames. The RL script holds out 100 examples for validation.
