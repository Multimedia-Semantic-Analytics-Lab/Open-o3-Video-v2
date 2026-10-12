import ast
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openo3_eval.datasetpp.base import BaseDatasetPP
from openo3_eval.registry import DATASET_REGISTRY, AnswerParsers, DataSets
from openo3_eval.runtime import get_logger
from openo3_eval.vl_utils.extract_frames import batch_extract_frames

logger = get_logger()


def _load_and_format_srt(srt_path: str | Path, max_chars: int = 12000) -> str:
    """Read an SRT file using pysrt, clean up formatting, and return concatenated text."""
    if not os.path.exists(srt_path):
        return ""

    import pysrt

    try:
        subs = pysrt.open(srt_path, encoding="utf-8")
    except Exception:
        try:
            subs = pysrt.open(srt_path)  # Fallback to auto-detect encoding (often handled by chardet)
        except Exception as e:
            logger.warning(f"[Open-o3 Eval] Failed to parse SRT file {srt_path}: {e}")
            return ""

    text_blocks: List[str] = []
    for sub in subs:
        if not sub.text:
            continue

        # Clean HTML tags and specific subtitle markers like \N
        clean_text = re.sub(r"<[^>]+>", "", sub.text).replace("\\N", " ").replace("\n", " ").strip()
        if not clean_text:
            continue

        # Avoid perfectly duplicate sequential lines with the same text
        if text_blocks and text_blocks[-1] == clean_text:
            continue

        text_blocks.append(clean_text)

    text = "\n".join(text_blocks).strip()
    return text[:max_chars]


@DATASET_REGISTRY.register(name=DataSets.video_mme)
class VideoMME(BaseDatasetPP):
    """VideoMME evaluation dataset.

    Per-dataset env vars (override globals):
      - VIDEO_MME_DATASET_DIR: root directory (contains Video-MME.tsv, subtitle/)
      - VIDEO_MME_USER_PROMPT: prompt txt path (with or without {asr})
      - VIDEO_MME_INPUT_MODE: "video_path" | "image"
      - VIDEO_MME_SAMPLE_FPS: frame extraction FPS
      - VIDEO_MME_RESIZE: frame resize "WxH" or empty
      - VIDEO_MME_ASR_MAX_CHARS: ASR (subtitle) truncation length
    """

    dataset_name = DataSets.video_mme
    answer_parser = AnswerParsers.video_mme_parser

    dataset_dir_env_key = "VIDEO_MME_DATASET_DIR"
    user_prompt_env_key = "VIDEO_MME_USER_PROMPT"
    input_mode_env_key = "VIDEO_MME_INPUT_MODE"
    sample_fps_env_key = "VIDEO_MME_SAMPLE_FPS"
    resize_env_key = "VIDEO_MME_RESIZE"

    def _load_rows(self, dataset_path: Path) -> Tuple[List[Dict[str, Any]], Path, Path]:
        if dataset_path.is_file():
            benchmark_root = dataset_path.parent
            if dataset_path.suffix == ".json":
                with open(dataset_path, "r", encoding="utf-8") as f:
                    rows = json.load(f)
                if not isinstance(rows, list):
                    raise ValueError(f"[Open-o3 Eval] VideoMME json must contain a list of rows: {dataset_path}")
                return rows, dataset_path, benchmark_root
            if dataset_path.suffix == ".tsv":
                with open(dataset_path, "r", encoding="utf-8") as f:
                    return list(csv.DictReader(f, delimiter="\t")), dataset_path, benchmark_root
            raise ValueError(f"[Open-o3 Eval] Unsupported VideoMME annotation file: {dataset_path}")

        anno_path = dataset_path / "Video-MME.tsv"
        if not anno_path.exists():
            raise FileNotFoundError(f"[Open-o3 Eval] VideoMME data file not found: {anno_path}")
        with open(anno_path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter="\t")), anno_path, dataset_path

    def _parse_candidates(self, row: Dict[str, Any], video: str) -> List[str]:
        raw_candidates = row.get("candidates") or []
        if isinstance(raw_candidates, list):
            return [str(candidate) for candidate in raw_candidates]
        raw_text = str(raw_candidates).strip()
        try:
            candidates = ast.literal_eval(raw_text)
            if isinstance(candidates, list):
                return [str(candidate) for candidate in candidates]
        except Exception:
            pass
        logger.warning(f"[Open-o3 Eval] Invalid candidates field for video {video}: {raw_text!r}; skipping.")
        return []

    def build_dataset(self) -> List[Dict[str, Any]]:
        dataset_path = self.get_dataset_dir()
        input_mode = self.get_input_mode()
        sample_fps = self.get_sample_fps()
        resize = self.get_resize()
        frame_cache_dir = self.get_frame_cache_dir(self.dataset_name)
        asr_max_chars = int(os.environ.get("VIDEO_MME_ASR_MAX_CHARS", "12000"))

        # Determine if we even need ASR to save processing time
        prompt_template = self.get_user_prompt_template()
        need_asr = "{asr}" in prompt_template

        self.dataset = []

        rows, anno_path, benchmark_root = self._load_rows(dataset_path)

        # Collect valid items
        all_items: List[Tuple[Dict[str, Any], str, List[str], str, str, str]] = []
        skipped_missing_video = 0
        for row in rows:
            video = str(row.get("video") or row.get("videoID") or row.get("video_id") or "").strip()
            video_path = str(row.get("video_path") or "").strip()
            if video_path and not os.path.isabs(video_path):
                video_path = str(benchmark_root / video_path)
            if not video_path or not os.path.exists(video_path):
                skipped_missing_video += 1
                continue

            # Figure out subtitle path
            subtitle_path = str(row.get("subtitle_path") or "").strip()
            if subtitle_path:
                if not os.path.isabs(subtitle_path):
                    subtitle_path = str(benchmark_root / subtitle_path)
            else:
                subtitle_path = str(benchmark_root / "subtitle" / f"{video}.srt")

            asr_info = ""
            if need_asr:
                asr_info = _load_and_format_srt(subtitle_path, max_chars=asr_max_chars)

            if not asr_info:
                asr_info = "No ASR info available."

            candidates = self._parse_candidates(row, video)
            if not candidates:
                continue

            choice = "\n".join(map(str, candidates))
            answer = str(row.get("answer") or "").strip().upper()

            all_items.append((row, video_path, candidates, choice, answer, asr_info))

        cache_results: Dict[str, Tuple[List[str], List[float]]] = {}
        if input_mode == "image":
            unique_videos = list({video_path for _, video_path, _, _, _, _ in all_items})
            cache_results = batch_extract_frames(
                video_paths=unique_videos,
                cache_dir=frame_cache_dir,
                sample_fps=sample_fps,
                resize=resize,
            )

        skipped_frames = 0
        for row, video_path, candidates, choice, answer, asr_info in all_items:
            pp_meta = {
                "gt_answer": answer,
                "gt_candidates": candidates,
                "gt_question": str(row.get("question") or "").strip(),
                "gt_options_text": choice,
                "gt_index": str(row.get("index") or "").strip(),
                "gt_video": str(row.get("video") or "").strip(),
                "gt_duration": str(row.get("duration") or "").strip(),
                "gt_domain": str(row.get("domain") or "").strip(),
                "gt_sub_category": str(row.get("sub_category") or "").strip(),
                "gt_task_type": str(row.get("task_type") or "").strip(),
            }

            build_user_prompt_kwargs = {
                "question": str(row.get("question") or "").strip(),
                "choice": choice,
                "asr": asr_info,
            }

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
            f"[Open-o3 Eval] VideoMME loaded {len(self.dataset)} samples from {anno_path}; "
            f"skipped {skipped_missing_video} missing videos, {skipped_frames} frame failures "
            f"(mode={input_mode})"
        )
        return self.dataset

    def calc_metric(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        eval_groups: Dict[str, List[Dict[str, Any]]] = {"overall": data_list}

        dimensions = {
            "duration": "gt_duration",
            "domain": "gt_domain",
            "sub_category": "gt_sub_category",
            "task_type": "gt_task_type",
        }
        for dim_name, field in dimensions.items():
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for data in data_list:
                pp_meta = data.get("pp_meta", {})
                value = str(pp_meta.get(field) or data.get(field) or "unknown").strip()
                groups.setdefault(value, []).append(data)
            for value, items in sorted(groups.items()):
                eval_groups[f"{dim_name}/{value}"] = items

        for group_name, items in eval_groups.items():
            total = len(items)
            valid = 0
            correct = 0
            for item in items:
                pp_meta = item.get("pp_meta", {})
                candidates = pp_meta.get("gt_candidates") or item.get("gt_candidates") or []

                pred = self.parse_answer(item.get("response", ""), candidates=candidates)
                gold = str(pp_meta.get("gt_answer") or item.get("gt_answer") or "").strip()
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
