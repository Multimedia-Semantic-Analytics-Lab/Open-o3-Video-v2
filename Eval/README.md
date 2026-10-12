# Evaluation

The local `video_o3` evaluator covers all nine benchmarks in the paper: V-STaR, VideoMME, VideoMME-v2, WorldSense, VideoMMMU, Video-Holmes, LVBench, Charades-TimeLens, and TVGBench. Run the commands below from `Eval/` on a Linux host with CUDA.

## Setup

```bash
python -m pip install uv
uv sync
uv pip install --python .venv/bin/python modelscope
.venv/bin/modelscope download --model FDUBay/Open-o3-Video-v2-4B --local_dir ./models/Open-o3-Video-v2-4B
```

The environment uses Python 3.12. The default model key is `openo3video_v2_4b`; the runner reads `models/Open-o3-Video-v2-4B`. Set `OPENO3VIDEO_V2_4B_MODEL_PATH` to use another local path. Other model keys and their path variables are defined in `openo3_eval/video_o3/config.py`.

## Benchmark inputs

Download each benchmark's annotations and videos from its release. Set the paths below to your local files. The default root is `data/benchmarks/` inside `Eval/`.

| Dataset key | Annotation variable | Media variable |
| --- | --- | --- |
| `v_star` | `V_STAR_DATASET_DIR` | `V_STAR_VIDEO_DIR` |
| `timelens_bench` (Charades-TimeLens) | `TIMELENS_BENCH_DATASET_DIR` | — |
| `video_mme` | `VIDEO_MME_DATASET_DIR` | — |
| `video_mme_v2` | `VIDEO_MME_V2_DATASET_DIR` | — |
| `worldsense` | `WORLDSENSE_DATASET_DIR` | — |
| `videommmu` | `VIDEOMMMU_DATASET_DIR` | `VIDEOMMMU_QUESTION_IMAGE_DIR` (if needed) |
| `video_holmes` | `VIDEO_HOLMES_DATASET_DIR` | — |
| `lvbench` | `LVBENCH_DATASET_DIR` | `LVBENCH_MEDIA_ROOT` |
| `tvgbench` | `TVGBENCH_DATASET_DIR` | `TVGBENCH_VIDEO_ROOT` |

For Charades-TimeLens, point `TIMELENS_BENCH_DATASET_DIR` to `charades-timelens.json` and place videos under `TimeLens-Bench/videos/charades/`. V-STaR expects `V_STaR_test.json` and a video directory. After inference, the included scorer uses Qwen2.5-72B-Instruct to judge answer similarity and computes temporal and spatial grounding metrics. Set `OPENO3_EVAL_VSTAR_EVAL_MODEL` to a local judge checkpoint.

Video-MME accepts TSV or JSON rows with a local `video_path`; relative paths resolve from the annotation directory. Video-Holmes uses `video/VIDEO_ID.mp4` beside its annotation file when `video_path` is absent. For TVGBench, place `tvgbench.json` under `data/benchmarks/TVGBench/` and videos under `data/benchmarks/TVGBench/videos/tvgbench_data/`.

## Run

```bash
export VIDEO_HOLMES_DATASET_DIR="$PWD/data/benchmarks/Video-Holmes/test_Video-Holmes.json"
DRY_RUN=1 bash scripts/eval_openo3video_v2_4b.sh video_holmes
VAL_DATASET_SAMPLE=1 bash scripts/eval_openo3video_v2_4b.sh video_holmes
```

Remove `VAL_DATASET_SAMPLE` for a full run. With no dataset argument, the script runs all nine paper benchmarks. Pass one or more dataset keys to run a subset. Set `CUDA_VISIBLE_DEVICES` and `NPROC_PER_NODE` for the available GPUs. The common settings are 1 fps, 96 frames, 4,096 new tokens, temperature 0.01, top-p 0.001, and top-k 1. General understanding and V-STaR allow five `crop_video` turns. Charades-TimeLens and TVGBench use `max_turn=0`; TVGBench also uses `use_tools=false`.

Results and logs are written under `openo3_eval_result/openo3video_v2_4b/`, or under `RESULT_ROOT` if set. For direct CLI options, run `uv run --no-sync openo3-eval eval-openo3 --help`. Run `uv run --no-sync pytest tests/` for parser and dataset tests.
