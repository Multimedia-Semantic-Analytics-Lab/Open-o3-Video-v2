from __future__ import annotations

import ast
import copy
import importlib.util
import json
import logging
import math
import os
import sys
from typing import Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

try:
    from omegaconf import DictConfig, ListConfig
except ModuleNotFoundError:  # pragma: no cover - lightweight import environments
    DictConfig = Any
    ListConfig = list

logger = logging.getLogger(__name__)


def _load_tool_format_module():
    module_name = "openo3_custom_tool_format"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = os.path.join(os.path.dirname(__file__), "tool_format.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load tool format helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_tool_format = _load_tool_format_module()


THINK_ANSWER_SYS_PROMPT = (
    "Output your reasoning process within the <think> and </think> tags.\n"
    "Your reasoning must be grounded in visual evidence from the video. Whenever you mention an object, person, "
    "or specific visual element that is relevant to the question, attach its visual grounding in the following "
    "format:\n"
    "<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s.\n"
    "During reasoning, adaptively decides whether additional visual observation is needed. If the currently "
    "observed evidence is insufficient, ambiguous, or requires a closer inspection of a specific video segment, "
    "you may request additional visual information by outputting:\n"
    '<tool_call>{"name": "crop_video", "arguments": {"start_time": ..., "end_time": ..., '
    '"sampling_strategy": "medium"}}</tool_call>\n'
    "Here, start_time and end_time are the start and end times in integer seconds within the full video. "
    'The value of sampling_strategy must be one of ["coarse", "medium", "fine"]. Choose the sampling strategy '
    "adaptively according to the difficulty of the segment: use \"coarse\" for broad context, \"medium\" for "
    "normal inspection, and \"fine\" for detailed or ambiguous visual evidence.\n"
    "Do not call the crop_video tool unnecessarily. If the available visual evidence is already sufficient to "
    "answer the question confidently, directly integrate the observed clues and provide the answer.\n"
    "Once you have gathered enough visual evidence and can determine the correct answer, output the correct answer "
    "within the <answer> and </answer> tags."
)


OPENO3_SYSTEM_PROMPT = {
    "visual QA": (
        "A conversation between user and assistant. The user provides an image and asks a question, and the "
        "Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then "
        "provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> "
        "and <answer> </answer> tags, respectively. When referring to particular objects in the reasoning process, "
        "the assistant MUST localize the object with bounding box coordinates between <box> and </box>. The answer "
        "must strictly follow the format `<obj>object_name</obj><box>bounding_box</box>`. You MUST strictly follow "
        "the format."
    ),
    "temporal-spatial free-form QA": (
        "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant "
        "solves it.\n" + THINK_ANSWER_SYS_PROMPT
    ),
    "temporal QA": (
        "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant "
        "determines the precise time period that answers the question. The assistant MUST first think about the "
        "reasoning process in the mind and then provide the user with the answer. The reasoning process and answer "
        "are enclosed within <think> </think> and <answer> </answer> tags, respectively. The final answer must "
        "strictly follow the following format: `<answer>From <t>start_time</t>s to <t>end_time</t>s</answer>`."
    ),
    "temporal QA (MCQ)": (
        "A conversation between user and assistant. The user provides a video and a multiple-choice question, and "
        "the Assistant determines the precise time period that answers the question and selects the correct option. "
        "The assistant MUST first think about the reasoning process in the mind and then provide the user with the "
        "answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, "
        "respectively. When mentioning time during the reasoning process, the assistant must use the format: "
        "`<t>time_in_seconds</t>s`. The answer must strictly follow the following format: "
        "`From <t>start_time</t>s to <t>end_time</t>s.\\nCorrect Option: [ONLY THE LETTER]`."
    ),
    "General video QA MCQ": (
        "A conversation between user and assistant. The user provides a video and asks a multiple-choice question, "
        "and the Assistant solves it.\n" + THINK_ANSWER_SYS_PROMPT + "Only output the correct option in the "
        "<answer> </answer> section."
    ),
    "General video QA Free-form": (
        "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant "
        "solves it.\n" + THINK_ANSWER_SYS_PROMPT
    ),
}


VIDEOO3_SYSTEM_PROMPT_MULTI_ROUND_MC = (
    "You are a helpful assistant. Answer the user's multiple-choice question based on the provided video.\n"
    + THINK_ANSWER_SYS_PROMPT
    + "The following is a example of the thinking process and the final answer:\n"
    " - When you want to examine a specific clue segment of a video more closely, produce exactly: "
    "'<think>your thinking process here</think>\n"
    '<tool_call>{"name": "crop_video", "arguments": {"start_time": START_TIMESTAMP, '
    '"end_time": END_TIMESTAMP, "sampling_strategy": "medium"}}</tool_call>\n'
    "'.\n"
    " - When you believe the current information is sufficient to give an answer, produce exactly: "
    "'<think>your thinking process here</think>\n<answer>Option Letter</answer>\n'."
)

VIDEOO3_QUESTION_TEMPLATE_MULTI_ROUND_MC = (
    "Here is the original full video (Observation 0):\n"
    "<video>\n"
    'Answer the following multiple-choice question according to the content of the video: "{question}".\n'
    "Options:\n"
    "{options}\n"
    "You are advised to first observe potential clue segments, use '<think>YOUR THINK</think>\n"
    "<tool_call>YOUR VIDEO CROP TOOL</tool_call>\n"
    "' to to specify the segment you want to observe in detail.\n"
    "If the evidence is visible in the original video for a long enough time and is clear enough to support a "
    "confident and correct answer, you may answer directly.\n"
)


OPENO3_TASKS = set(OPENO3_SYSTEM_PROMPT)
VIDEOO3_SOURCES = {"tool_reward_with_round_penalty", "tool_reward_without_round_penalty"}
CGBENCH_SOURCES = VIDEOO3_SOURCES
DEFAULT_TIMELENS_VIDEO_ROOT = os.environ.get("TIMELENS_VIDEO_ROOT", "data/videos/timelens")


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return [str(item) for item in list(value) if item not in (None, "")]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple)):
                    return [str(item) for item in parsed if item not in (None, "")]
            except (SyntaxError, ValueError):
                pass
        return [stripped]
    return [str(value)]


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _has_key(config: Any, key: str) -> bool:
    try:
        return key in config
    except TypeError:
        return False


def _read_rows(path: str) -> list[dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".json", ".jsonl"}:
        with open(path, "r", encoding="utf-8") as f:
            if ext == ".jsonl":
                return [json.loads(line) for line in f if line.strip()]
            payload = json.load(f)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "items", "examples"):
                if isinstance(payload.get(key), list):
                    return payload[key]
        raise ValueError(f"Unsupported JSON payload in {path}: {type(payload).__name__}")
    if ext in {".parquet", ".pq"}:
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Unsupported data file format: {path}")


def _sample_rows(rows: list[dict[str, Any]], limit: int, *, random_sample: bool, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    if random_sample:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(rows), size=limit, replace=False).tolist()
        return [rows[i] for i in indices]
    return rows[:limit]


def _sample_exact_rows(
    rows: list[dict[str, Any]],
    count: int,
    *,
    random_sample: bool,
    seed: int,
    replace: bool,
) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    if count <= len(rows):
        return _sample_rows(rows, count, random_sample=random_sample, seed=seed)
    if not replace:
        return rows
    if random_sample:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(rows), size=count, replace=True).tolist()
        return [copy.deepcopy(rows[i]) for i in indices]
    repeats, remainder = divmod(count, len(rows))
    return [copy.deepcopy(row) for _ in range(repeats) for row in rows] + [
        copy.deepcopy(row) for row in rows[:remainder]
    ]


def _get_float_config(config: Any, key: str, default: float) -> float:
    try:
        return float(config.get(key, default))
    except (TypeError, ValueError):
        return default


def _allocate_mix_counts(
    sizes: dict[str, int],
    ratios: dict[str, float],
    total_limit: int,
    *,
    oversample: bool,
) -> dict[str, int]:
    positive = [name for name, ratio in ratios.items() if ratio > 0]
    available = [name for name in positive if sizes.get(name, 0) > 0]
    counts = {name: 0 for name in sizes}
    if not available:
        return counts

    ratio_sum = sum(ratios[name] for name in available)
    normalized = {name: ratios[name] / ratio_sum for name in available}
    if oversample:
        target_total = total_limit if total_limit > 0 else sum(sizes[name] for name in available)
    else:
        feasible_total = int(min(sizes[name] / normalized[name] for name in available))
        target_total = feasible_total if total_limit <= 0 else min(total_limit, feasible_total)
    if target_total <= 0:
        return counts

    desired = {name: target_total * normalized[name] for name in available}
    remaining = target_total
    while remaining > 0:
        candidates = [
            name
            for name in available
            if oversample or counts[name] < sizes[name]
        ]
        if not candidates:
            break
        chosen = max(candidates, key=lambda name: desired[name] - counts[name])
        counts[chosen] += 1
        remaining -= 1
    return counts


def _mix_row_groups(
    groups: dict[str, list[dict[str, Any]]],
    ratios: dict[str, float],
    total_limit: int,
    *,
    random_sample: bool,
    seed: int,
    oversample: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not any(ratio > 0 for ratio in ratios.values()):
        rows = [row for name in groups for row in groups[name]]
        return _sample_rows(rows, total_limit, random_sample=random_sample, seed=seed), {
            name: len(groups[name]) for name in groups
        }

    sizes = {name: len(rows) for name, rows in groups.items()}
    counts = _allocate_mix_counts(sizes, ratios, total_limit, oversample=oversample)
    mixed: list[dict[str, Any]] = []
    for offset, name in enumerate(groups):
        mixed.extend(
            _sample_exact_rows(
                groups[name],
                counts.get(name, 0),
                random_sample=random_sample,
                seed=seed + 10 + offset,
                replace=oversample,
            )
        )
    if random_sample and len(mixed) > 1:
        rng = np.random.default_rng(seed + 100)
        indices = rng.permutation(len(mixed)).tolist()
        mixed = [mixed[i] for i in indices]
    return mixed, counts


def _openo3_roots(data_root: str, timelens_video_root: str | None = None) -> dict[str, str]:
    return {
        "videoespresso_video": os.path.join(data_root, "videoespresso/videos"),
        "videoespresso_kf": os.path.join(data_root, "videoespresso/kfs"),
        "timerft": os.path.join(data_root, "timerft"),
        "timelens": timelens_video_root or DEFAULT_TIMELENS_VIDEO_ROOT,
        "tvg": os.path.join(data_root, "tvg_r1"),
        "gqa": os.path.join(data_root, "gqa"),
        "stgr_tg": os.path.join(data_root, "stgr/temporal_grounding/videos"),
        "stgr_tg_kf": os.path.join(data_root, "stgr/temporal_grounding/kfs"),
        "stgr_plm": os.path.join(data_root, "stgr/plm/videos"),
        "stgr_plm_kf": os.path.join(data_root, "stgr/plm/kfs"),
        "videor1": os.path.join(data_root, "videor1"),
    }


def _join_under_root(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    normalized = path.replace("\\", "/")
    root_name = os.path.basename(os.path.normpath(root))
    if normalized == root_name:
        normalized = ""
    elif normalized.startswith(root_name + "/"):
        normalized = normalized[len(root_name) + 1 :]
    return os.path.join(root, normalized)


def _resolve_openo3_paths(
    row: dict[str, Any],
    data_root: str,
    timelens_video_root: str | None = None,
) -> dict[str, Optional[str]]:
    roots = _openo3_roots(data_root, timelens_video_root)
    source = row.get("source", "")
    out: dict[str, Optional[str]] = {"video_path": None, "image_path": None, "key_frame_root": None}
    if source == "videoespresso_train_video":
        out["video_path"] = os.path.join(roots["videoespresso_video"], row["video_path"])
        out["key_frame_root"] = roots["videoespresso_kf"]
    elif source == "timerft":
        out["video_path"] = os.path.join(roots["timerft"], row["video_path"])
    elif source == "timelens":
        out["video_path"] = _join_under_root(roots["timelens"], row["video_path"])
    elif source == "gqa":
        out["image_path"] = os.path.join(roots["gqa"], row["image_path"])
    elif "STR_plm" in source:
        out["video_path"] = os.path.join(roots["stgr_plm"], row["video_path"])
        out["key_frame_root"] = roots["stgr_plm_kf"]
    elif "STR" in source:
        out["video_path"] = os.path.join(roots["stgr_tg"], row["video_path"])
        out["key_frame_root"] = roots["stgr_tg_kf"]
    elif "TVG" in source:
        out["video_path"] = os.path.join(roots["tvg"], row["video_path"])
    elif "videor1" in source:
        out["video_path"] = os.path.join(roots["videor1"], row["video_path"])
    else:
        raise ValueError(f"Unknown Open-o3 source: {source}")
    for field in ("video_path", "image_path"):
        relative = row.get(field)
        if relative and not os.path.isabs(str(relative)):
            direct = os.path.join(data_root, str(relative))
            if os.path.isfile(direct):
                out[field] = direct
    return out


def _resolve_relative_video_path(row: dict[str, Any], base_media_dir: str, dataset_name: str) -> str:
    video_path = row.get("video") or row.get("video_path")
    if video_path is None:
        raise ValueError(f"{dataset_name} row missing video path; keys={list(row.keys())}")
    video_path = str(video_path)
    if os.path.isabs(video_path):
        return video_path

    path_with_ext = video_path if os.path.splitext(video_path)[1] else f"{video_path}.mp4"
    basename = os.path.basename(path_with_ext)
    candidates = [
        os.path.join(base_media_dir, path_with_ext),
        os.path.join(base_media_dir, basename),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[-1]


def _resolve_videoo3_path(row: dict[str, Any], base_media_dir: str) -> str:
    return _resolve_relative_video_path(row, base_media_dir, "Video-o3")


def _resolve_cgbench_path(row: dict[str, Any], video_root: str) -> str:
    return _resolve_relative_video_path(row, video_root, "CG-Bench")


def _resize_max_height(image: Image.Image, max_height: int) -> Image.Image:
    if max_height <= 0 or image.height <= max_height:
        return image
    width = max(1, int(round(image.width * (max_height / image.height))))
    return image.resize((width, max_height), Image.Resampling.BICUBIC)


def _sample_indices(frame_count: int, fps: float, target_fps: float, max_frames: int) -> list[int]:
    if frame_count <= 0:
        return []
    fps = fps if fps > 0 else 1.0
    target_fps = target_fps if target_fps > 0 else 1.0
    duration = frame_count / fps
    count = max(1, int(math.floor(duration * target_fps)))
    if max_frames > 0:
        count = min(count, max_frames)
    if count == 1:
        return [0]
    return [int(round(i * (frame_count - 1) / (count - 1))) for i in range(count)]


def _load_video_frames(
    video_path: str,
    *,
    target_fps: float,
    max_frames: int,
    max_height: int,
) -> tuple[list[Image.Image], list[float], float]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 1.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        indices = _sample_indices(frame_count, fps, target_fps, max_frames)
        frames: list[Image.Image] = []
        timestamps: list[float] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = _resize_max_height(Image.fromarray(frame), max_height)
            frames.append(image)
            timestamps.append(float(idx / fps))
        if not frames:
            raise RuntimeError(f"No frames decoded from {video_path}")
        duration = (frame_count / fps) if frame_count > 0 else max(timestamps)
        return frames, timestamps, float(duration)
    finally:
        cap.release()


def _merge_key_frames(
    frames: list[Image.Image],
    timestamps: list[float],
    key_frames: list[dict[str, Any]],
    key_frame_root: str,
    target_size: tuple[int, int],
) -> tuple[list[Image.Image], list[float]]:
    extras: list[tuple[float, Image.Image]] = []
    for key_frame in key_frames:
        path = os.path.join(key_frame_root, key_frame["path"])
        image = Image.open(path).convert("RGB").resize(target_size, Image.Resampling.BICUBIC)
        extras.append((float(round(key_frame["time"])), image))
    merged = [(float(t), image) for image, t in zip(frames, timestamps, strict=False)] + extras
    merged.sort(key=lambda item: item[0])
    return [image for _, image in merged], [t for t, _ in merged]


def _frame_content(frames: list[Image.Image], timestamps: list[float], total_seconds: float) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image, timestamp in zip(frames, timestamps, strict=False):
        content.append({"type": "text", "text": f"<{round(float(timestamp), 1)} seconds>"})
        content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": "\n"})
    content.append({"type": "text", "text": f"The video is in total {int(round(total_seconds))} seconds.\n"})
    return content


class OpenO3VideoDataset(Dataset):
    """Dataset for legacy Open-o3-Video multi-image timestamp training."""

    def __init__(
        self,
        data_files,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.max_samples = max_samples
        self.target_fps = float(config.get("target_fps", 1.0))
        self.max_video_frames = int(config.get("max_video_frames", os.getenv("UNIFIED_MAX_VIDEO_FRAMES", 32)))
        self.frame_max_height = int(config.get("frame_max_height", os.getenv("UNIFIED_FRAME_MAX_HEIGHT", 224)))
        self.base_media_dir = str(config.get("base_media_dir") or os.getenv("LLAVA_VIDEO_ROOT", "./datasets"))
        self.open_o3_video_root = str(config.get("open_o3_video_root") or os.getenv("OPEN_O3_VIDEO_ROOT", "./data"))
        self.timelens_video_root = str(
            config.get("timelens_video_root") or os.getenv("TIMELENS_VIDEO_ROOT", DEFAULT_TIMELENS_VIDEO_ROOT)
        )
        self.cgbench_video_root = str(
            config.get("cgbench_video_root") or os.getenv("CGBENCH_VIDEO_ROOT", self.base_media_dir)
        )
        self.train_max_samples_random = _as_bool(config.get("train_max_samples_random"), True)
        self.train_max_samples_seed = int(config.get("train_max_samples_seed", 42))
        self.tool_format = _tool_format.resolve_tool_format(config.get("tool_format"))
        self.openo3_max_samples = int(config.get("openo3_max_samples", -1))
        self.videoo3_max_samples = int(config.get("videoo3_max_samples", -1))
        self.timelens_max_samples = int(config.get("timelens_max_samples", -1))
        self.cgbench_max_samples = int(config.get("cgbench_max_samples", -1))
        self.mix_ratio_oversample = _as_bool(config.get("mix_ratio_oversample"), False)
        self.mix_ratios = {
            "openo3": _get_float_config(config, "openo3_mix_ratio", 0.0),
            "videoo3": _get_float_config(config, "videoo3_mix_ratio", 0.0),
            "timelens": _get_float_config(config, "timelens_mix_ratio", 0.0),
            "cgbench": _get_float_config(config, "cgbench_mix_ratio", 0.0),
        }

        data_files_list = _ensure_list(data_files)
        val_files = _ensure_list(config.get("val_files"))
        is_val = bool(data_files_list and val_files and set(data_files_list) == set(val_files))
        if is_val:
            openo3_files = (
                _ensure_list(config.get("val_openo3_files"))
                if _has_key(config, "val_openo3_files")
                else data_files_list
            )
            videoo3_files = (
                _ensure_list(config.get("val_videoo3_files")) if _has_key(config, "val_videoo3_files") else []
            )
            timelens_files = (
                _ensure_list(config.get("val_timelens_files")) if _has_key(config, "val_timelens_files") else []
            )
            cgbench_files = (
                _ensure_list(config.get("val_cgbench_files")) if _has_key(config, "val_cgbench_files") else []
            )
        else:
            openo3_files = (
                _ensure_list(config.get("openo3_files")) if _has_key(config, "openo3_files") else data_files_list
            )
            videoo3_files = _ensure_list(config.get("videoo3_files")) if _has_key(config, "videoo3_files") else []
            timelens_files = _ensure_list(config.get("timelens_files")) if _has_key(config, "timelens_files") else []
            cgbench_files = _ensure_list(config.get("cgbench_files")) if _has_key(config, "cgbench_files") else []

        self._rows = self._load_rows(openo3_files, videoo3_files, timelens_files, cgbench_files)

    def _load_rows(
        self,
        openo3_files: list[str],
        videoo3_files: list[str],
        timelens_files: list[str],
        cgbench_files: list[str],
    ) -> list[dict[str, Any]]:
        openo3_rows: list[dict[str, Any]] = []
        videoo3_rows: list[dict[str, Any]] = []
        timelens_rows: list[dict[str, Any]] = []
        cgbench_rows: list[dict[str, Any]] = []
        for path in openo3_files:
            for source_idx, row in enumerate(_read_rows(path)):
                row = dict(row)
                row["_family"] = "openo3"
                row["_source_file"] = path
                row["_source_index"] = source_idx
                row.setdefault("data_source", row.get("task", "default"))
                openo3_rows.append(row)
        for path in videoo3_files:
            for source_idx, row in enumerate(_read_rows(path)):
                row = dict(row)
                row["_family"] = "videoo3"
                row["_source_file"] = path
                row["_source_index"] = source_idx
                row.setdefault("data_source", row.get("source", row.get("data_source", "default")))
                videoo3_rows.append(row)
        for path in timelens_files:
            for source_idx, row in enumerate(_read_rows(path)):
                row = dict(row)
                row["_family"] = "timelens"
                row["_source_file"] = path
                row["_source_index"] = source_idx
                row.setdefault("data_source", row.get("task", "default"))
                timelens_rows.append(row)
        for path in cgbench_files:
            for source_idx, row in enumerate(_read_rows(path)):
                row = dict(row)
                row["_family"] = "cgbench"
                row["_source_file"] = path
                row["_source_index"] = source_idx
                row.setdefault("data_source", row.get("source", row.get("data_source", "default")))
                cgbench_rows.append(row)

        openo3_rows = _sample_rows(
            openo3_rows,
            self.openo3_max_samples,
            random_sample=self.train_max_samples_random,
            seed=self.train_max_samples_seed,
        )
        videoo3_rows = _sample_rows(
            videoo3_rows,
            self.videoo3_max_samples,
            random_sample=self.train_max_samples_random,
            seed=self.train_max_samples_seed + 1,
        )
        timelens_rows = _sample_rows(
            timelens_rows,
            self.timelens_max_samples,
            random_sample=self.train_max_samples_random,
            seed=self.train_max_samples_seed + 2,
        )
        cgbench_rows = _sample_rows(
            cgbench_rows,
            self.cgbench_max_samples,
            random_sample=self.train_max_samples_random,
            seed=self.train_max_samples_seed + 3,
        )
        rows, mix_counts = _mix_row_groups(
            {
                "openo3": openo3_rows,
                "videoo3": videoo3_rows,
                "timelens": timelens_rows,
                "cgbench": cgbench_rows,
            },
            self.mix_ratios,
            self.max_samples,
            random_sample=self.train_max_samples_random,
            seed=self.train_max_samples_seed,
            oversample=self.mix_ratio_oversample,
        )
        logger.warning(
            "OpenO3VideoDataset loaded %d rows: openo3=%d videoo3=%d timelens=%d cgbench=%d "
            "mix_counts=%s mix_ratios=%s oversample=%s",
            len(rows),
            len(openo3_rows),
            len(videoo3_rows),
            len(timelens_rows),
            len(cgbench_rows),
            mix_counts,
            self.mix_ratios,
            self.mix_ratio_oversample,
        )
        return rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = copy.deepcopy(self._rows[idx])
        if row["_family"] in {"openo3", "timelens"}:
            return self._build_openo3(row)
        if row["_family"] == "videoo3":
            return self._build_videoo3(row)
        if row["_family"] == "cgbench":
            return self._build_cgbench(row)
        raise ValueError(f"Unknown row family: {row.get('_family')}")

    def _build_openo3(self, row: dict[str, Any]) -> dict[str, Any]:
        task = row.get("task")
        if task not in OPENO3_TASKS:
            raise ValueError(f"Unknown Open-o3 task: {task}")
        paths = _resolve_openo3_paths(row, self.open_o3_video_root, self.timelens_video_root)
        system = OPENO3_SYSTEM_PROMPT[task]

        if task == "visual QA":
            image_path = paths["image_path"]
            if not image_path:
                raise ValueError(f"visual QA row missing image path: {row.get('id')}")
            image = Image.open(image_path).convert("RGB")
            user_content = [{"type": "image", "image": image}, {"type": "text", "text": row["question"]}]
            timestamps: list[float] = []
            duration = None
            video_path = None
            image_size = image.size
        else:
            video_path = paths["video_path"]
            if not video_path:
                raise ValueError(f"Open-o3 video row missing video path: {row.get('id')}")
            frames, timestamps, duration = _load_video_frames(
                video_path,
                target_fps=self.target_fps,
                max_frames=self.max_video_frames,
                max_height=self.frame_max_height,
            )
            image_size = frames[0].size
            if task == "temporal-spatial free-form QA" and row.get("key_frames") and paths.get("key_frame_root"):
                frames, timestamps = _merge_key_frames(
                    frames,
                    timestamps,
                    row["key_frames"],
                    paths["key_frame_root"],
                    image_size,
                )
            user_content = _frame_content(frames, timestamps, duration)
            user_content.append({"type": "text", "text": row["question"]})
            image_path = None

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        extra_info = self._extra_info(row, video_path, image_path, timestamps, duration, image_size)
        return self._item(
            row=row,
            messages=messages,
            data_source=task,
            ground_truth=row.get("answer"),
            extra_info=extra_info,
        )

    def _build_videoo3(self, row: dict[str, Any]) -> dict[str, Any]:
        data_source = row.get("data_source") or row.get("source")
        if data_source not in VIDEOO3_SOURCES:
            raise ValueError(f"Unknown Video-o3 data source: {data_source}")
        video_path = _resolve_videoo3_path(row, self.base_media_dir)
        return self._build_tool_mc_video(row, data_source, video_path)

    def _build_cgbench(self, row: dict[str, Any]) -> dict[str, Any]:
        data_source = row.get("data_source") or row.get("source")
        if data_source not in CGBENCH_SOURCES:
            raise ValueError(f"Unknown CG-Bench data source: {data_source}")
        video_path = _resolve_cgbench_path(row, self.cgbench_video_root)
        return self._build_tool_mc_video(row, data_source, video_path)

    def _build_tool_mc_video(self, row: dict[str, Any], data_source: str, video_path: str) -> dict[str, Any]:
        frames, timestamps, duration = _load_video_frames(
            video_path,
            target_fps=self.target_fps,
            max_frames=self.max_video_frames,
            max_height=self.frame_max_height,
        )
        question = VIDEOO3_QUESTION_TEMPLATE_MULTI_ROUND_MC.format(
            question=row["question"],
            options=str(row["options"]),
        )
        prefix, suffix = question.split("<video>", 1)
        user_content = [{"type": "text", "text": prefix}]
        user_content.extend(_frame_content(frames, timestamps, duration))
        user_content.append({"type": "text", "text": suffix})
        messages = [
            {"role": "system", "content": VIDEOO3_SYSTEM_PROMPT_MULTI_ROUND_MC},
            {"role": "user", "content": user_content},
        ]

        ground_truth = copy.deepcopy(row.get("solution"))
        if ground_truth is None:
            ground_truth = {"answer": row.get("answer", ""), "clue": row.get("clue", [])}
        elif not isinstance(ground_truth, dict):
            ground_truth = {"answer": ground_truth, "clue": row.get("clue", [])}
        ground_truth["key_objects"] = row.get("key_objects")

        extra_info = self._extra_info(row, video_path, None, timestamps, duration, frames[0].size)
        return self._item(
            row=row,
            messages=messages,
            data_source=data_source,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

    def _extra_info(
        self,
        row: dict[str, Any],
        video_path: Optional[str],
        image_path: Optional[str],
        timestamps: list[float],
        duration: Optional[float],
        image_size: Optional[tuple[int, int]],
    ) -> dict[str, Any]:
        extra_info = dict(row.get("extra_info", {}) or {})
        extra_info.update(
            {
                "index": row.get("uid", row.get("id", row.get("doc_id", row.get("_source_index", 0)))),
                "question": row.get("question", ""),
                "options": row.get("options"),
                "video_sample_fps": self.target_fps,
                "max_video_frames": self.max_video_frames,
                "frame_max_height": self.frame_max_height,
                "video_duration": duration,
                "image_size": image_size,
                "frame_timestamps": timestamps,
                "debug_source_file": row.get("_source_file"),
                "debug_source_index": row.get("_source_index"),
                "debug_family": row.get("_family"),
            }
        )
        if video_path is not None:
            extra_info["video_path"] = video_path
        if image_path is not None:
            extra_info["image_path"] = image_path
        if "key_frames" in row:
            extra_info["key_frames"] = row["key_frames"]
        if "key_items" in row:
            extra_info["key_items"] = row["key_items"]
        return extra_info

    def _item(
        self,
        *,
        row: dict[str, Any],
        messages: list[dict[str, Any]],
        data_source: str,
        ground_truth: Any,
        extra_info: dict[str, Any],
    ) -> dict[str, Any]:
        video_path = extra_info.get("video_path")
        messages = _tool_format.rewrite_messages_for_tool_format(messages, self.tool_format)
        tools_kwargs = {
            "crop_video": {
                "create_kwargs": {
                    "video_path": video_path,
                    "video_duration": extra_info.get("video_duration"),
                }
            }
        }
        return {
            "raw_prompt": messages,
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "data_source": data_source,
            "reward_model": {"ground_truth": ground_truth},
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "tools_kwargs": tools_kwargs,
            "index": extra_info["index"],
            "uid": extra_info["index"],
            "agent_name": "openo3_tool_agent",
        }

    def resume_dataset_state(self) -> None:
        return None

    @classmethod
    async def process_vision_info(
        cls,
        messages: list[dict[str, Any]],
        image_patch_size,
        config: DictConfig,
    ) -> tuple[list[Image.Image], None]:
        images: list[Image.Image] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image":
                    continue
                image = part.get("image")
                if isinstance(image, Image.Image):
                    images.append(image.convert("RGB"))
                elif isinstance(image, str):
                    images.append(Image.open(image).convert("RGB"))
        return images, None
