import os
from pathlib import Path

DEFAULT_VIDEOMMMU_QUESTION_IMAGE_DIR = Path(os.environ.get("VIDEOMMMU_IMAGE_DIR", "data/benchmarks/VideoMMMU/images"))

DEFAULT_LONGVIDEOBENCH_MEDIA_ROOT = Path(os.environ.get("LONGVIDEOBENCH_MEDIA_ROOT", "data/benchmarks/LongVideoBench"))

DEFAULT_LVBENCH_MEDIA_ROOT = Path(os.environ.get("LVBENCH_MEDIA_ROOT", "data/benchmarks/LVBench"))

DEFAULT_LONGVIDEOREASON_EVAL_VIDEO_ROOT = Path(os.environ.get("LONGVIDEOREASON_EVAL_VIDEO_ROOT", "data/benchmarks/LongVideo-Reason/videos"))
