import ast
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openo3_eval.datasetpp.base import BaseDatasetPP
from openo3_eval.registry import DATASET_REGISTRY, AnswerParsers, DataSets
from openo3_eval.runtime import get_logger
from openo3_eval.vl_utils.extract_frames import batch_extract_frames

logger = get_logger()

LETTERS = ("A", "B", "C", "D", "E", "F")


def _parse_candidates(raw_value: Any) -> List[str]:
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value]
    if isinstance(raw_value, dict):
        return [f"{key}. {raw_value.get(key, '')}".replace("'", "") for key in LETTERS if key in raw_value]

    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return []
    try:
        parsed = ast.literal_eval(raw_text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        if isinstance(parsed, dict):
            return [f"{key}. {parsed.get(key, '')}".replace("'", "") for key in LETTERS if key in parsed]
    except Exception:
        pass

    return [line.strip() for line in raw_text.splitlines() if line.strip()]


def _rows_from_video_holmes_json(json_path: Path) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw_rows = json.load(f)

    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_rows):
        video = str(item.get("video ID") or item.get("video") or item.get("video_id") or "").strip()
        options = item.get("Options") or item.get("options") or {}
        rows.append(
            {
                "index": idx,
                "video": video,
                "video_path": item.get("video_path") or f"./video/{video}.mp4",
                "candidates": _parse_candidates(options),
                "question": item.get("Question") or item.get("question") or "",
                "answer": item.get("Answer") or item.get("answer") or "",
                "question_id": item.get("Question ID") or item.get("question_id") or "",
                "question_type": item.get("Question Type") or item.get("question_type") or "",
                "explanation": item.get("Explanation") or item.get("explanation") or "",
            }
        )
    return rows


def _load_video_holmes_rows(dataset_path: Path) -> Tuple[List[Dict[str, Any]], Path, Path]:
    if dataset_path.is_file():
        if dataset_path.suffix == ".json":
            return _rows_from_video_holmes_json(dataset_path), dataset_path, dataset_path.parent
        if dataset_path.suffix == ".tsv":
            with open(dataset_path, "r", encoding="utf-8") as f:
                return list(csv.DictReader(f, delimiter="\t")), dataset_path, dataset_path.parent
        raise ValueError(f"[Open-o3 Eval] Unsupported VideoHolmes annotation file: {dataset_path}")

    dataset_dir = dataset_path
    tsv_path = dataset_dir / "Video_Holmes.tsv"
    if tsv_path.exists():
        with open(tsv_path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter="\t")), tsv_path, dataset_dir

    json_path = dataset_dir / "test_Video-Holmes.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"[Open-o3 Eval] VideoHolmes annotation not found: expected {tsv_path} or {json_path}"
        )

    return _rows_from_video_holmes_json(json_path), json_path, dataset_dir


@DATASET_REGISTRY.register(name=DataSets.video_holmes)
class VideoHolmes(BaseDatasetPP):
    """Video-Holmes multiple-choice video benchmark.

    Per-dataset env vars:
      - VIDEO_HOLMES_DATASET_DIR: root directory (contains test_Video-Holmes.json or Video_Holmes.tsv)
      - VIDEO_HOLMES_USER_PROMPT: prompt txt path/template
      - VIDEO_HOLMES_INPUT_MODE: "video_path" | "image"
      - VIDEO_HOLMES_SAMPLE_FPS: frame extraction FPS
      - VIDEO_HOLMES_RESIZE: frame resize "WxH" or empty
    """

    user_prompt_default = (
        "<video>\n"
        "Based on the given video, reason and answer the single-choice question. "
        "Respond with only the option letter.\n"
        "Question: {question}\n"
        "Options:\n{choice}\n"
        "Answer:"
    )

    dataset_name = DataSets.video_holmes
    answer_parser = AnswerParsers.video_holmes_parser

    dataset_dir_env_key = "VIDEO_HOLMES_DATASET_DIR"
    user_prompt_env_key = "VIDEO_HOLMES_USER_PROMPT"
    input_mode_env_key = "VIDEO_HOLMES_INPUT_MODE"
    sample_fps_env_key = "VIDEO_HOLMES_SAMPLE_FPS"
    resize_env_key = "VIDEO_HOLMES_RESIZE"

    def build_dataset(self) -> List[Dict[str, Any]]:
        dataset_path = self.get_dataset_dir()
        input_mode = self.get_input_mode()
        sample_fps = self.get_sample_fps()
        resize = self.get_resize()
        frame_cache_dir = self.get_frame_cache_dir(self.dataset_name)

        self.dataset = []
        rows, anno_path, benchmark_root = _load_video_holmes_rows(dataset_path)

        all_items: List[Tuple[Dict[str, Any], str, List[str], str, str]] = []
        skipped_missing_video = 0
        for row in rows:
            video = str(row.get("video") or row.get("video_id") or row.get("video ID") or "").strip()
            video_path = str(row.get("video_path") or "").strip()
            if video_path and not os.path.isabs(video_path):
                video_path = str(benchmark_root / video_path)
            if not video_path:
                video_path = str(benchmark_root / "video" / f"{video}.mp4")
            if not os.path.exists(video_path):
                skipped_missing_video += 1
                continue

            candidates = _parse_candidates(row.get("candidates") or row.get("Options") or row.get("options") or [])
            if not candidates:
                logger.warning(f"[Open-o3 Eval] VideoHolmes has no candidates for video {video}; skipping.")
                continue

            choice = "\n".join(candidates)
            answer = str(row.get("answer") or row.get("Answer") or "").strip().upper()
            all_items.append((row, video_path, candidates, choice, answer))

        cache_results: Dict[str, Tuple[List[str], List[float]]] = {}
        if input_mode == "image":
            unique_videos = list({video_path for _, video_path, _, _, _ in all_items})
            cache_results = batch_extract_frames(
                video_paths=unique_videos,
                cache_dir=frame_cache_dir,
                sample_fps=sample_fps,
                resize=resize,
            )

        skipped_frames = 0
        for row, video_path, candidates, choice, answer in all_items:
            question = str(row.get("question") or row.get("Question") or "").strip()
            pp_meta = {
                "gt_answer": answer,
                "gt_candidates": candidates,
                "gt_question": question,
                "gt_options_text": choice,
                "gt_index": str(row.get("index") or "").strip(),
                "gt_video": str(row.get("video") or row.get("video ID") or "").strip(),
                "gt_question_id": str(row.get("question_id") or row.get("Question ID") or "").strip(),
                "gt_question_type": str(row.get("question_type") or row.get("Question Type") or "").strip(),
                "gt_explanation": str(row.get("explanation") or row.get("Explanation") or "").strip(),
            }
            build_user_prompt_kwargs = {"question": question, "choice": choice}

            if self.is_video_path_input_mode(input_mode):
                prompt = self.build_user_prompt(prompt_replacements=None, **build_user_prompt_kwargs)
                item = {
                    "messages": [{"role": "user", "content": prompt}],
                    "videos": [video_path],
                    "pp_meta": pp_meta,
                }
                self.dataset.append(item)
            else:
                cached = cache_results.get(video_path)
                if not cached:
                    skipped_frames += 1
                    continue

                frame_paths, frame_ts = cached
                image_token_content = "".join(f"<{ts:.1f} seconds><image>" for ts in frame_ts)
                prompt = self.build_user_prompt(
                    prompt_replacements={"<video>": image_token_content},
                    **build_user_prompt_kwargs,
                )
                item = {
                    "messages": [{"role": "user", "content": prompt}],
                    "images": frame_paths,
                    "pp_meta": pp_meta,
                }
                self.dataset.append(item)

        logger.info(
            f"[Open-o3 Eval] VideoHolmes loaded {len(self.dataset)} samples from {anno_path}; "
            f"skipped {skipped_missing_video} missing videos, {skipped_frames} frame failures "
            f"(mode={input_mode})"
        )
        return self.dataset

    def calc_metric(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        eval_groups: Dict[str, List[Dict[str, Any]]] = {"overall": data_list}
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for data in data_list:
            pp_meta = data.get("pp_meta", {})
            question_type = str(pp_meta.get("gt_question_type") or data.get("gt_question_type") or "unknown").strip()
            by_type.setdefault(question_type, []).append(data)
        for question_type, items in sorted(by_type.items()):
            eval_groups[f"question_type/{question_type}"] = items

        metrics: Dict[str, Any] = {}
        for group_name, items in eval_groups.items():
            total = len(items)
            valid = 0
            correct = 0
            for item in items:
                pp_meta = item.get("pp_meta", {})
                candidates = pp_meta.get("gt_candidates") or item.get("gt_candidates") or []
                pred = self.parse_answer(item.get("response", ""), candidates=candidates)
                gold = str(pp_meta.get("gt_answer") or item.get("gt_answer") or "").strip().upper()
                if pred:
                    valid += 1
                if pred and gold and pred == gold:
                    correct += 1

            metrics[group_name] = {
                "total": total,
                "valid": valid,
                "acc": round(correct * 100 / total, 2) if total > 0 else 0,
                "valid_acc": round(correct * 100 / valid, 2) if valid > 0 else 0,
            }

        return metrics

    def build_sample_eval(self, item: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        pp_meta = result.get("pp_meta", {})
        candidates = pp_meta.get("gt_candidates") or result.get("gt_candidates") or []
        pred = self.parse_answer(result.get("response", ""), candidates=candidates)
        gold = str(pp_meta.get("gt_answer") or result.get("gt_answer") or "").strip().upper()
        is_valid = bool(pred)
        is_correct = bool(pred and gold and pred == gold)
        return {
            "sample_eval": {
                "metric_type": "classification",
                "pred": pred,
                "gold": gold,
                "is_valid": is_valid,
                "is_correct": is_correct,
                "acc": float(is_correct),
            }
        }
