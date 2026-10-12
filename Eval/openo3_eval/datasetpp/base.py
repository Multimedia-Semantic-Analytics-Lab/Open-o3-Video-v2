import json
import os
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Tuple

import openo3_eval.env as env
from openo3_eval.registry import ANSWER_PARSER_REGISTRY
from openo3_eval.runtime import get_logger
from openo3_eval.util import make_image_cache_param_key, print_metric_pretty, resolve_prompt

logger = get_logger()

RAW_VIDEO_INPUT_MODE = "video_path"
IMAGE_INPUT_MODE = "image"
INPUT_MODES = {RAW_VIDEO_INPUT_MODE, IMAGE_INPUT_MODE}


def normalize_input_mode(value: str, *, source: str) -> str:
    mode = str(value).strip().lower()
    if mode not in INPUT_MODES:
        supported = "', '".join(sorted(INPUT_MODES))
        raise ValueError(f"[Open-o3 Eval] Invalid {source}: {value!r}, must be one of '{supported}'")
    return mode


class BaseDatasetPP:
    """Base class for Open-o3 Eval dataset plugins.

    Subclasses must implement:
    - build_dataset: populate self.dataset with List[Dict[str, Any]]
    - calc_metric: compute evaluation metrics from inference results

    Subclasses can set `dataset_dir_env_key` to specify a per-dataset env var
    for the dataset directory (e.g. "TIMELENS_BENCH_DATASET_DIR").

    Subclasses can set `user_prompt_env_key` to allow
    overriding prompts via environment variables.

    For video/image datasets, subclasses can set:
    - ``input_mode_env_key``  → per-dataset override for input mode (video_path/image)
    - ``sample_fps_env_key``  → per-dataset override for frame sampling FPS
    - ``resize_env_key``      → per-dataset override for frame resize (WxH)
    """

    user_prompt_default: ClassVar[str] = "How are you?"

    dataset_name: ClassVar[str] = ""
    answer_parser: ClassVar[str] = ""

    dataset_dir_env_key: ClassVar[str] = ""
    user_prompt_env_key: ClassVar[str] = ""
    input_mode_env_key: ClassVar[str] = ""
    sample_fps_env_key: ClassVar[str] = ""
    resize_env_key: ClassVar[str] = ""

    def __init__(self, args: Any) -> None:
        self.args = args
        self.dataset: List[Dict[str, Any]] = []

        # print once
        self._user_prompt_logged = False
        self._answer_parser_logged = False

    def get_dataset_dir(self) -> Path:
        """Resolve dataset_dir: --dataset_dir > per-dataset env var."""
        dataset_dir = self.args.dataset_dir
        if not dataset_dir and self.dataset_dir_env_key:
            dataset_dir = os.environ.get(self.dataset_dir_env_key, "")
        if not dataset_dir:
            hint = f" Set --dataset_dir or {self.dataset_dir_env_key} in .env" if self.dataset_dir_env_key else ""
            raise ValueError(f"[Open-o3 Eval] dataset_dir not specified.{hint}")
        path = Path(dataset_dir)
        if not path.exists():
            raise FileNotFoundError(f"[Open-o3 Eval] dataset_dir not found: {path}")
        return path

    def get_user_prompt_template(self, default: str | Path | None = None) -> str:
        """Resolve user prompt template: env var > default > class default, and load file content if the value is an existing path."""
        prompt: str | Path = default if default is not None else self.user_prompt_default
        if self.user_prompt_env_key:
            prompt = os.environ.get(self.user_prompt_env_key, str(prompt))
        return resolve_prompt(str(prompt))

    def build_user_prompt(
        self,
        default: str | Path | None = None,
        prompt_replacements: Dict[str, str] | None = None,
        **prompt_kwargs: Any,
    ) -> str:
        """Build the final user prompt from the resolved template and format variables."""
        prompt = self.get_user_prompt_template(default).format(**prompt_kwargs)
        if prompt_replacements:
            for old, new in prompt_replacements.items():
                prompt = prompt.replace(old, new)
        if not self._user_prompt_logged:
            logger.info(f"[Open-o3 Eval] user prompt:\n{prompt}")
            self._user_prompt_logged = True
        return prompt

    # -- video / frame extraction helpers ----------------------------------

    def get_input_mode(self) -> str:
        """Per-dataset env → global ``env.INPUT_MODE``."""
        if self.input_mode_env_key:
            val = os.environ.get(self.input_mode_env_key)
            if val:
                return normalize_input_mode(val, source=self.input_mode_env_key)
        return normalize_input_mode(env.INPUT_MODE, source="OPENO3_EVAL_INPUT_MODE")

    @staticmethod
    def is_video_path_input_mode(input_mode: str) -> bool:
        return input_mode == RAW_VIDEO_INPUT_MODE

    def get_sample_fps(self) -> float:
        """Per-dataset env → global ``env.SAMPLE_FPS``."""
        if self.sample_fps_env_key:
            val = os.environ.get(self.sample_fps_env_key)
            if val:
                return float(val)
        return env.SAMPLE_FPS

    def get_resize(self) -> Tuple[int, int] | None:
        """Per-dataset env (``WxH``) → global ``env.RESIZE``."""
        if self.resize_env_key:
            val = os.environ.get(self.resize_env_key)
            if val:
                return env._parse_resize(val)
        return env.RESIZE

    def get_frame_cache_dir(self, dataset_name: str) -> str:
        """Return ``<CACHE_DIR>/<dataset_name>/frames``."""
        return os.path.join(env.CACHE_DIR, dataset_name, "frames")

    def get_dataset_cache_stem(self) -> str:
        """Return a cache stem for dataset jsonl keyed by build-affecting settings."""
        input_mode = self.get_input_mode()
        if self.is_video_path_input_mode(input_mode):
            return "dataset_video_path"

        return f"dataset_image_{make_image_cache_param_key(self.get_sample_fps(), self.get_resize(), env.JPEG_QUALITY)}"

    def parse_answer(self, text: str, **kwargs: Any) -> Any:
        answer_parser = getattr(self.args, "answer_parser", "") or self.answer_parser
        if not answer_parser:
            raise ValueError("[Open-o3 Eval] answer parser name is not specified.")

        if not self._answer_parser_logged:
            logger.info(f"[Open-o3 Eval] parsing answer with parser: {answer_parser}")
            self._answer_parser_logged = True

        return ANSWER_PARSER_REGISTRY.get(answer_parser)()(text, **kwargs)

    def build_dataset(self) -> List[Dict[str, Any]]:
        """Build dataset and store in self.dataset. Returns the list."""
        raise NotImplementedError

    def calc_metric(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute evaluation metrics from inference result data list."""
        raise NotImplementedError

    def build_sample_eval(self, item: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Return per-sample evaluation fields to be stored in the result row."""
        return {}

    def print_metric(self, metrics: Dict[str, Any], dataset_pp: str) -> None:
        """Print metrics in a human-readable format."""
        print_metric_pretty(metrics, dataset_pp)

    def save_dataset(self, path: str) -> None:
        """Write self.dataset to a jsonl file."""
        with open(path, "w") as f:
            for item in self.dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
