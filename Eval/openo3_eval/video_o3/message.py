from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import openo3_eval.env as env

from .frames import frame_timestamps_from_meta, sample_full_video_frames
from .prompts import (
    MULTI_IMAGE_INITIAL_PROMPT,
    QUESTION_TEMPLATE_MC,
    QUESTION_TEMPLATE_TG,
    build_direct_answer_prompt,
    build_direct_temporal_grounding_prompt,
    build_think_answer_prompt,
    build_think_answer_temporal_grounding_prompt,
)

TEMPORAL_GROUNDING_DATASETS = {"charades", "timelens_bench", "tvgbench"}


def is_temporal_grounding_dataset(dataset_name: str) -> bool:
    return dataset_name in TEMPORAL_GROUNDING_DATASETS


def _get_video_o3_fields(item: Dict[str, Any], *, require_options: bool = True) -> Tuple[str, str, str]:
    pp_meta = item.get("pp_meta", {})
    videos = item.get("videos") or []
    if not videos:
        raise ValueError("[Open-o3 Eval] OpenO3 evaluation requires dataset items with `videos`.")

    video_path = str(videos[0])
    question = str(pp_meta.get("gt_question") or item.get("gt_question") or "").strip()
    candidates = pp_meta.get("gt_candidates") or item.get("gt_candidates") or []
    options_text = str(pp_meta.get("gt_options_text") or item.get("gt_options_text") or "").strip()
    if not options_text and candidates:
        options_text = "\n".join(map(str, candidates))
    if not question or (require_options and not options_text):
        raise ValueError("[Open-o3 Eval] OpenO3 evaluation requires gt_question and gt_options_text/gt_candidates.")
    return video_path, question, options_text


def _build_frame_timestamp_items(
    frame_paths: List[str],
    timestamps: List[float],
    *,
    image_kwargs: Dict[str, Any] | None = None,
    frame_prompt_template: str = "<{timestamp:.1f} seconds><image>\n",
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for frame_index, (frame_path, timestamp) in enumerate(zip(frame_paths, timestamps, strict=True), start=1):
        items.append(
            {
                "type": "text",
                "value": frame_prompt_template.format(frame_index=frame_index, timestamp=timestamp),
            }
        )
        image_item: Dict[str, Any] = {"type": "image", "value": frame_path}
        if image_kwargs:
            image_item.update(image_kwargs)
        items.append(image_item)
    return items


def _frame_image_kwargs(args: Any, frame_count: int) -> Dict[str, Any]:
    if frame_count <= 0:
        return {}

    from openo3_eval.video_o3.config import VIDEO_O3_MODEL_CONFIGS

    config = VIDEO_O3_MODEL_CONFIGS.get(str(getattr(args, "model", "")))
    if config is None:
        return {}

    image_kwargs = {
        "min_pixels": config.min_pixels,
        "max_pixels": config.max_pixels,
        "total_pixels": config.total_pixels,
    }
    dataset_name = str(getattr(args, "dataset_pp", ""))
    if config.pixel_overrides and dataset_name in config.pixel_overrides:
        image_kwargs.update(config.pixel_overrides[dataset_name])
    return image_kwargs


def _frame_prompt_template(args: Any) -> str:
    from openo3_eval.video_o3.config import VIDEO_O3_MODEL_CONFIGS

    config = VIDEO_O3_MODEL_CONFIGS.get(str(getattr(args, "model", "")))
    if config is not None and config.frame_prompt_template is not None:
        return config.frame_prompt_template
    return "<{timestamp:.1f} seconds><image>\n"


def build_video_o3_message(
    args: Any,
    item: Dict[str, Any],
    dataset_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    is_tg = is_temporal_grounding_dataset(dataset_name)
    video_path, question, options_text = _get_video_o3_fields(item, require_options=not is_tg)
    frames, frame_meta = sample_full_video_frames(
        video_path=video_path,
        cache_dir=os.path.join(env.CACHE_DIR, "openo3", dataset_name, "frames"),
        fps=args.fps,
        frames_limit=args.frames_limit,
    )

    frame_timestamps = frame_timestamps_from_meta(frame_meta, len(frames))
    frame_meta = {**frame_meta, "timestamps": frame_timestamps}
    video_intro = MULTI_IMAGE_INITIAL_PROMPT.format(
        sample_fps=float(frame_meta["sample_fps"]),
        total_frames=int(frame_meta["sample_n_frames"]),
        max_duration=float(frame_meta["duration"]),
    )

    if is_tg:
        prompt = QUESTION_TEMPLATE_TG.format(question=question)
    else:
        prompt = QUESTION_TEMPLATE_MC.format(question=question, options=options_text)

    message: List[Dict[str, Any]] = [
        *([{"type": "question_type", "value": "tg"}] if is_tg else []),
        {"type": "raw_video", "value": video_path},
    ]
    direct_prompt = (
        build_direct_temporal_grounding_prompt(question)
        if is_tg
        else build_direct_answer_prompt(question, options_text)
    )
    think_answer_prompt = (
        build_think_answer_temporal_grounding_prompt(question)
        if is_tg
        else build_think_answer_prompt(question, options_text)
    )
    message.append({"type": "direct_prompt", "value": direct_prompt})
    message.append({"type": "think_answer_prompt", "value": think_answer_prompt})
    message.extend(
        [
            *_build_frame_timestamp_items(
                frames,
                frame_timestamps,
                image_kwargs=_frame_image_kwargs(args, len(frames)),
                frame_prompt_template=_frame_prompt_template(args),
            ),
            {"type": "text", "value": video_intro + prompt},
        ]
    )
    return message, frame_meta


def build_video_o3_custom_message(
    args: Any,
    *,
    video_path: str,
    prompt_text: str,
    dataset_name: str,
    question_type: str = "oe",
    direct_prompt: str | None = None,
    think_answer_prompt: str | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    frames, frame_meta = sample_full_video_frames(
        video_path=video_path,
        cache_dir=os.path.join(env.CACHE_DIR, "openo3", dataset_name, "frames"),
        fps=args.fps,
        frames_limit=args.frames_limit,
    )

    frame_timestamps = frame_timestamps_from_meta(frame_meta, len(frames))
    frame_meta = {**frame_meta, "timestamps": frame_timestamps}
    video_intro = MULTI_IMAGE_INITIAL_PROMPT.format(
        sample_fps=float(frame_meta["sample_fps"]),
        total_frames=int(frame_meta["sample_n_frames"]),
        max_duration=float(frame_meta["duration"]),
    )

    message: List[Dict[str, Any]] = [
        {"type": "question_type", "value": question_type},
        {"type": "raw_video", "value": video_path},
    ]
    if direct_prompt is not None:
        message.append({"type": "direct_prompt", "value": direct_prompt})
    if think_answer_prompt is not None:
        message.append({"type": "think_answer_prompt", "value": think_answer_prompt})
    message.extend(
        [
            *_build_frame_timestamp_items(
                frames,
                frame_timestamps,
                image_kwargs=_frame_image_kwargs(args, len(frames)),
                frame_prompt_template=_frame_prompt_template(args),
            ),
            {"type": "text", "value": video_intro + prompt_text},
        ]
    )

    return message, frame_meta
