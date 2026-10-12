import ast
import csv
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openo3_eval.datasetpp.base import BaseDatasetPP
from openo3_eval.registry import DATASET_REGISTRY, AnswerParsers, DataSets
from openo3_eval.runtime import get_logger
from openo3_eval.vl_utils.extract_frames import batch_extract_frames

logger = get_logger()


def _load_and_format_srt(srt_path: str | Path, max_chars: int = 12000) -> str:
    if not srt_path or not os.path.exists(srt_path):
        return ""

    import pysrt

    try:
        subs = pysrt.open(str(srt_path), encoding="utf-8")
    except Exception:
        try:
            subs = pysrt.open(str(srt_path))
        except Exception as exc:
            logger.warning(f"[Open-o3 Eval] Failed to parse WorldSense subtitle {srt_path}: {exc}")
            return ""

    text_blocks: List[str] = []
    for sub in subs:
        clean_text = re.sub(r"<[^>]+>", "", str(sub.text or "")).replace("\\N", " ").replace("\n", " ").strip()
        if not clean_text:
            continue
        if text_blocks and text_blocks[-1] == clean_text:
            continue
        text_blocks.append(clean_text)
    return "\n".join(text_blocks)[:max_chars]


def _parse_candidates(raw_value: Any) -> List[str]:
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]

    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return []

    try:
        parsed = ast.literal_eval(raw_text)
    except Exception:
        parsed = None

    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]

    return [line.strip() for line in raw_text.splitlines() if line.strip()]


def _resolve_existing_path(benchmark_root: Path, *candidates: str) -> str:
    for candidate in candidates:
        candidate_text = str(candidate or "").strip()
        if not candidate_text:
            continue
        candidate_path = Path(candidate_text)
        if candidate_path.is_absolute() and candidate_path.exists():
            return str(candidate_path)
        joined = benchmark_root / candidate_text
        if joined.exists():
            return str(joined)
    return ""


def _build_flat_file_index(benchmark_root: Path) -> Dict[str, str]:
    try:
        output = subprocess.check_output(
            ["find", str(benchmark_root), "-maxdepth", "1", "-type", "f"],
            text=True,
        )
    except Exception:
        return {}

    index: Dict[str, str] = {}
    for line in output.splitlines():
        file_path = line.strip()
        if not file_path:
            continue
        index[Path(file_path).name] = file_path
    return index


def _load_worldsense_rows(dataset_path: Path) -> Tuple[List[Dict[str, Any]], Path, Path]:
    if dataset_path.is_file():
        benchmark_root = dataset_path.parent
        if dataset_path.suffix == ".json":
            with open(dataset_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            if not isinstance(rows, list):
                raise ValueError(f"[Open-o3 Eval] WorldSense json must contain a list of rows: {dataset_path}")
            return rows, dataset_path, benchmark_root
        if dataset_path.suffix == ".tsv":
            with open(dataset_path, "r", encoding="utf-8") as f:
                return list(csv.DictReader(f, delimiter="\t")), dataset_path, benchmark_root
        raise ValueError(f"[Open-o3 Eval] Unsupported WorldSense annotation file: {dataset_path}")

    candidates = [dataset_path / "WorldSense.json", dataset_path / "WorldSense.tsv"]
    anno_path = next((path for path in candidates if path.exists()), None)
    if anno_path is None:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"[Open-o3 Eval] WorldSense annotation not found. searched: {searched}")
    return _load_worldsense_rows(anno_path)


def _summarize_accuracy(items: List[Dict[str, Any]], dataset: "WorldSense") -> Dict[str, Any]:
    total = len(items)
    valid = 0
    correct = 0
    for item in items:
        pp_meta = item.get("pp_meta", {})
        candidates = pp_meta.get("gt_candidates") or []
        pred = dataset.parse_answer(item.get("response", ""), candidates=candidates)
        gold = str(pp_meta.get("gt_answer") or "").strip().upper()
        if pred:
            valid += 1
        if pred and gold and pred == gold:
            correct += 1
    return {
        "total": total,
        "valid": valid,
        "valid_pct": round(valid * 100 / total, 2) if total > 0 else 0.0,
        "correct": correct,
        "acc": round(correct * 100 / total, 2) if total > 0 else 0.0,
    }


@DATASET_REGISTRY.register(name=DataSets.worldsense)
class WorldSense(BaseDatasetPP):
    user_prompt_default = Path(__file__).resolve().parents[1] / "prompts" / "worldsense.txt"
    dataset_name = DataSets.worldsense
    answer_parser = AnswerParsers.video_mme_parser

    dataset_dir_env_key = "WORLDSENSE_DATASET_DIR"
    user_prompt_env_key = "WORLDSENSE_USER_PROMPT"
    input_mode_env_key = "WORLDSENSE_INPUT_MODE"
    sample_fps_env_key = "WORLDSENSE_SAMPLE_FPS"
    resize_env_key = "WORLDSENSE_RESIZE"

    def build_dataset(self) -> List[Dict[str, Any]]:
        dataset_path = self.get_dataset_dir()
        input_mode = self.get_input_mode()
        sample_fps = self.get_sample_fps()
        resize = self.get_resize()
        frame_cache_dir = self.get_frame_cache_dir(self.dataset_name)
        asr_max_chars = int(os.environ.get("WORLDSENSE_ASR_MAX_CHARS", "12000"))
        prompt_template = self.get_user_prompt_template()
        need_asr = "{asr}" in prompt_template

        self.dataset = []
        rows, anno_path, benchmark_root = _load_worldsense_rows(dataset_path)
        flat_file_index = _build_flat_file_index(benchmark_root) if benchmark_root.is_dir() else {}

        all_items: List[Tuple[Dict[str, Any], str, str, List[str], str, str]] = []
        skipped_missing_video = 0
        for row in rows:
            video_id = str(row.get("video") or row.get("video_id") or "").strip()
            video_path = str(row.get("video_path") or "").strip()
            if video_path and os.path.isabs(video_path) and os.path.exists(video_path):
                resolved_video_path = video_path
            else:
                resolved_video_path = flat_file_index.get(f"{video_id}.mp4", "")
                if not resolved_video_path:
                    resolved_video_path = _resolve_existing_path(
                        benchmark_root,
                        video_path,
                        f"{video_id}.mp4",
                        os.path.join("videos", f"{video_id}.mp4"),
                    )
            if not resolved_video_path:
                skipped_missing_video += 1
                continue

            subtitle_path = str(row.get("subtitle_path") or "").strip()
            if subtitle_path and os.path.isabs(subtitle_path) and os.path.exists(subtitle_path):
                resolved_subtitle_path = subtitle_path
            else:
                resolved_subtitle_path = flat_file_index.get(f"{video_id}.srt", "")
                if not resolved_subtitle_path:
                    resolved_subtitle_path = _resolve_existing_path(
                        benchmark_root,
                        subtitle_path,
                        f"{video_id}.srt",
                        os.path.join("subtitles", f"{video_id}.srt"),
                    )
            asr_info = _load_and_format_srt(resolved_subtitle_path, max_chars=asr_max_chars) if need_asr else ""
            if not asr_info:
                asr_info = "No ASR info available."

            candidates = _parse_candidates(row.get("candidates"))
            if not candidates:
                logger.warning(f"[Open-o3 Eval] WorldSense row has no candidates, skip video={video_id!r}")
                continue

            all_items.append(
                (
                    row,
                    resolved_video_path,
                    video_id,
                    candidates,
                    "\n".join(candidates),
                    asr_info,
                )
            )

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
        for row, video_path, video_id, candidates, choice, asr_info in all_items:
            question = str(row.get("question") or "").strip()
            pp_meta = {
                "gt_answer": str(row.get("answer") or "").strip().upper(),
                "gt_candidates": candidates,
                "gt_question": question,
                "gt_options_text": choice,
                "gt_index": str(row.get("index") or "").strip(),
                "gt_video": video_id,
                "gt_duration": str(row.get("duration") or "").strip(),
                "gt_domain": str(row.get("domain") or "").strip(),
                "gt_sub_category": str(row.get("sub_category") or "").strip(),
                "gt_task_domain": str(row.get("task_domain") or "").strip(),
                "gt_task_type": str(row.get("task_type") or "").strip(),
                "gt_video_caption": str(row.get("video_caption") or "").strip(),
            }

            prompt_kwargs = {"question": question, "choice": choice, "asr": asr_info}
            if self.is_video_path_input_mode(input_mode):
                prompt = self.build_user_prompt(**prompt_kwargs)
                item = {
                    "messages": [{"role": "user", "content": prompt}],
                    "videos": [video_path],
                    "pp_meta": pp_meta,
                }
                self.dataset.append(item)
                continue

            cached = cache_results.get(video_path)
            if not cached:
                skipped_frames += 1
                continue
            frame_paths, frame_ts = cached
            image_token_content = "".join(f"<{ts:.1f} seconds><image>" for ts in frame_ts)
            prompt = self.build_user_prompt(prompt_replacements={"<video>": image_token_content}, **prompt_kwargs)
            item = {
                "messages": [{"role": "user", "content": prompt}],
                "images": frame_paths,
                "pp_meta": pp_meta,
            }
            self.dataset.append(item)

        logger.info(
            f"[Open-o3 Eval] WorldSense loaded {len(self.dataset)} samples from {anno_path}; "
            f"skipped {skipped_missing_video} missing videos, {skipped_frames} frame failures (mode={input_mode})"
        )
        return self.dataset

    def calc_metric(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {"overall": _summarize_accuracy(data_list, self)}

        for dim_key, meta_key in (
            ("duration", "gt_duration"),
            ("domain", "gt_domain"),
            ("sub_category", "gt_sub_category"),
            ("task_domain", "gt_task_domain"),
            ("task_type", "gt_task_type"),
        ):
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for item in data_list:
                value = str(item.get("pp_meta", {}).get(meta_key) or "").strip()
                if value:
                    grouped.setdefault(value, []).append(item)
            for group_name, items in sorted(grouped.items()):
                metrics[f"{dim_key}/{group_name}"] = _summarize_accuracy(items, self)

        return metrics

    def build_sample_eval(self, item: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        pp_meta = result.get("pp_meta", {})
        candidates = pp_meta.get("gt_candidates") or []
        pred = self.parse_answer(result.get("response", ""), candidates=candidates)
        gold = str(pp_meta.get("gt_answer") or "").strip().upper()
        return {
            "sample_eval": {
                "metric_type": "multiple_choice_accuracy",
                "pred_answer": pred,
                "gold_answer": gold,
                "correct": bool(pred and gold and pred == gold),
            }
        }
