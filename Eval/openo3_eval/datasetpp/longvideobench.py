import ast
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from openo3_eval.datasetpp.base import BaseDatasetPP
from openo3_eval.datasetpp.paths import DEFAULT_LONGVIDEOBENCH_MEDIA_ROOT
from openo3_eval.registry import DATASET_REGISTRY, AnswerParsers, DataSets
from openo3_eval.runtime import get_logger
from openo3_eval.vl_utils.extract_frames import batch_extract_frames

logger = get_logger()

CHOICE_LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _load_longvideobench_rows(dataset_path: Path) -> Tuple[List[Dict[str, Any]], Path]:
    if dataset_path.is_file():
        if dataset_path.suffix == ".json":
            with open(dataset_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            if not isinstance(rows, list):
                raise ValueError(f"[Open-o3 Eval] LongVideoBench json must contain a list of rows: {dataset_path}")
            return rows, dataset_path
        if dataset_path.suffix == ".tsv":
            with open(dataset_path, "r", encoding="utf-8") as f:
                return list(csv.DictReader(f, delimiter="\t")), dataset_path
        if dataset_path.suffix == ".parquet":
            df = pd.read_parquet(dataset_path)
            return df.to_dict(orient="records"), dataset_path
        raise ValueError(f"[Open-o3 Eval] Unsupported LongVideoBench annotation file: {dataset_path}")

    candidates = [
        dataset_path / "LongVideoBench.tsv",
        dataset_path / "longvideobench.json",
        dataset_path / "lvb_val.json",
        dataset_path / "test-00000-of-00001.parquet",
    ]
    anno_path = next((path for path in candidates if path.exists()), None)
    if anno_path is None:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"[Open-o3 Eval] LongVideoBench annotation not found. searched: {searched}")
    return _load_longvideobench_rows(anno_path)


def _resolve_media_root(dataset_path: Path) -> Path:
    candidates: List[Path] = []
    env_root = os.environ.get("LONGVIDEOBENCH_MEDIA_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root))
    if dataset_path.is_dir():
        candidates.append(dataset_path)
    else:
        candidates.append(dataset_path.parent)
        candidates.append(dataset_path.parent.parent)
    candidates.append(DEFAULT_LONGVIDEOBENCH_MEDIA_ROOT)

    for candidate in candidates:
        if (candidate / "videos").exists():
            return candidate
    return candidates[0]


def _load_and_format_subtitle(subtitle_path: str | Path, max_chars: int = 12000) -> str:
    if not subtitle_path or not os.path.exists(subtitle_path):
        return ""

    with open(subtitle_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            logger.warning(f"[Open-o3 Eval] Failed to parse LongVideoBench subtitle {subtitle_path}: {exc}")
            return ""

    texts: List[str] = []
    for entry in data if isinstance(data, list) else []:
        text = str(entry.get("text") or entry.get("line") or "").strip()
        text = text.replace("&nbsp;", " ").replace("\n", " ").strip()
        if not text:
            continue
        if texts and texts[-1] == text:
            continue
        texts.append(re.sub(r"\s+", " ", text))
    return "\n".join(texts)[:max_chars]


def _parse_candidates(row: Dict[str, Any]) -> List[str]:
    raw_candidates = row.get("candidates")
    if isinstance(raw_candidates, list):
        return [str(candidate).strip() for candidate in raw_candidates if str(candidate).strip()]
    if raw_candidates is not None:
        raw_text = str(raw_candidates).strip()
        if raw_text:
            try:
                parsed = ast.literal_eval(raw_text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return [str(candidate).strip() for candidate in parsed if str(candidate).strip()]

    candidates: List[str] = []
    for idx in range(8):
        value = str(row.get(f"option{idx}") or "").strip()
        if not value or value.upper() == "N/A":
            continue
        candidates.append(value)
    return candidates


def _format_choices(candidates: Iterable[str]) -> List[str]:
    return [f"{CHOICE_LETTERS[idx]}. {str(candidate).strip()}" for idx, candidate in enumerate(candidates)]


def _extract_answer_letter(row: Dict[str, Any], num_candidates: int) -> str:
    for key in ("answer", "gt_answer"):
        raw_answer = row.get(key)
        if raw_answer is not None and str(raw_answer).strip():
            text = str(raw_answer).strip().upper()
            if text in CHOICE_LETTERS[:num_candidates]:
                return text

    raw_choice = row.get("correct_choice")
    if raw_choice is None or str(raw_choice).strip() == "":
        return ""

    try:
        choice_idx = int(raw_choice)
    except Exception:
        text = str(raw_choice).strip().upper()
        return text if text in CHOICE_LETTERS[:num_candidates] else ""

    if choice_idx < 0 or choice_idx >= num_candidates:
        return ""
    return CHOICE_LETTERS[choice_idx]


def _resolve_existing_path(root: Path, relative_path: str, *fallbacks: str) -> str:
    candidates = [relative_path, *fallbacks]
    for candidate in candidates:
        candidate_text = str(candidate or "").strip()
        if not candidate_text:
            continue
        candidate_path = Path(candidate_text)
        if candidate_path.is_absolute() and candidate_path.exists():
            return str(candidate_path)
        normalized = candidate_text[2:] if candidate_text.startswith("./") else candidate_text
        joined = root / normalized
        if joined.exists():
            return str(joined)
        if "/" not in normalized:
            for subdir in ("videos", "subtitles"):
                joined = root / subdir / normalized
                if joined.exists():
                    return str(joined)
    return ""


@DATASET_REGISTRY.register(name=DataSets.longvideobench)
class LongVideoBench(BaseDatasetPP):
    user_prompt_default = Path(__file__).resolve().parents[1] / "prompts" / "longvideobench.txt"
    dataset_name = DataSets.longvideobench
    answer_parser = AnswerParsers.video_mme_parser

    dataset_dir_env_key = "LONGVIDEOBENCH_DATASET_DIR"
    user_prompt_env_key = "LONGVIDEOBENCH_USER_PROMPT"
    input_mode_env_key = "LONGVIDEOBENCH_INPUT_MODE"
    sample_fps_env_key = "LONGVIDEOBENCH_SAMPLE_FPS"
    resize_env_key = "LONGVIDEOBENCH_RESIZE"

    def build_dataset(self) -> List[Dict[str, Any]]:
        dataset_path = self.get_dataset_dir()
        input_mode = self.get_input_mode()
        sample_fps = self.get_sample_fps()
        resize = self.get_resize()
        frame_cache_dir = self.get_frame_cache_dir(self.dataset_name)
        asr_max_chars = int(os.environ.get("LONGVIDEOBENCH_ASR_MAX_CHARS", "12000"))
        prompt_template = self.get_user_prompt_template()
        need_asr = "{asr}" in prompt_template

        rows, anno_path = _load_longvideobench_rows(dataset_path)
        media_root = _resolve_media_root(dataset_path)
        self.dataset = []

        all_items: List[Tuple[Dict[str, Any], str, List[str], str, str, str]] = []
        skipped_missing_video = 0
        skipped_missing_answer = 0
        for row in rows:
            candidates = _parse_candidates(row)
            if not candidates:
                continue
            answer = _extract_answer_letter(row, len(candidates))
            if not answer:
                skipped_missing_answer += 1
                continue

            video_id = str(row.get("video_id") or row.get("video") or "").strip()
            video_path = _resolve_existing_path(
                media_root,
                str(row.get("video_path") or "").strip(),
                f"videos/{video_id}.mp4",
            )
            if not video_path:
                skipped_missing_video += 1
                continue

            subtitle_path = _resolve_existing_path(
                media_root,
                str(row.get("subtitle_path") or "").strip(),
                f"subtitles/{video_id}_en.json",
            )
            asr_info = _load_and_format_subtitle(subtitle_path, max_chars=asr_max_chars) if need_asr else ""
            if not asr_info:
                asr_info = "No ASR info available."

            formatted_choices = _format_choices(candidates)
            all_items.append(
                (
                    row,
                    video_path,
                    candidates,
                    "\n".join(formatted_choices),
                    answer,
                    asr_info,
                )
            )

        if not all_items:
            raise ValueError(
                f"[Open-o3 Eval] LongVideoBench found no evaluable rows in {anno_path}. "
                "The public test parquet has no ground-truth answer; please use validation annotations."
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
        for row, video_path, candidates, choice_text, answer, asr_info in all_items:
            pp_meta = {
                "gt_answer": answer,
                "gt_candidates": candidates,
                "gt_question": str(row.get("question") or "").strip(),
                "gt_options_text": choice_text,
                "gt_id": str(row.get("id") or "").strip(),
                "gt_video_id": str(row.get("video_id") or row.get("video") or "").strip(),
                "gt_duration_group": str(row.get("duration_group") or "").strip(),
                "gt_question_category": str(row.get("question_category") or "").strip(),
                "gt_topic_category": str(row.get("topic_category") or "").strip(),
                "gt_level": str(row.get("level") or "").strip(),
                "gt_position": str(row.get("position") or "").strip(),
            }
            prompt_kwargs = {
                "question": str(row.get("question") or "").strip(),
                "choice": choice_text,
                "asr": asr_info,
            }

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
            prompt = self.build_user_prompt(
                prompt_replacements={"<video>": image_token_content},
                **prompt_kwargs,
            )
            item = {
                "messages": [{"role": "user", "content": prompt}],
                "images": frame_paths,
                "pp_meta": pp_meta,
            }
            self.dataset.append(item)

        logger.info(
            f"[Open-o3 Eval] LongVideoBench loaded {len(self.dataset)} samples from {anno_path}; "
            f"skipped {skipped_missing_video} missing videos, {skipped_missing_answer} rows without GT answer, "
            f"{skipped_frames} frame failures (mode={input_mode})"
        )
        return self.dataset

    def build_sample_eval(self, item: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        pp_meta = item.get("pp_meta", {})
        pred = self.parse_answer(result.get("response", ""), candidates=pp_meta.get("gt_candidates") or [])
        gold = str(pp_meta.get("gt_answer") or "").strip().upper()
        return {
            "longvideobench_pred": pred,
            "longvideobench_gold": gold,
            "longvideobench_correct": float(bool(pred and gold and pred == gold)),
        }

    def calc_metric(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        groups: Dict[str, List[Dict[str, Any]]] = {"overall": data_list}
        for dim_name, meta_key in (
            ("duration_group", "gt_duration_group"),
            ("question_category", "gt_question_category"),
            ("topic_category", "gt_topic_category"),
            ("level", "gt_level"),
        ):
            dim_groups: Dict[str, List[Dict[str, Any]]] = {}
            for item in data_list:
                value = str(item.get("pp_meta", {}).get(meta_key) or "").strip()
                if value:
                    dim_groups.setdefault(value, []).append(item)
            for value, items in sorted(dim_groups.items()):
                groups[f"{dim_name}/{value}"] = items

        for group_name, items in groups.items():
            total = len(items)
            valid = 0
            correct = 0
            for item in items:
                pp_meta = item.get("pp_meta", {})
                pred = self.parse_answer(item.get("response", ""), candidates=pp_meta.get("gt_candidates") or [])
                gold = str(pp_meta.get("gt_answer") or "").strip().upper()
                if pred:
                    valid += 1
                if pred and gold and pred == gold:
                    correct += 1
            metrics[group_name] = {
                "total": total,
                "valid": valid,
                "valid_pct": round(valid * 100 / total, 2) if total else 0.0,
                "correct": correct,
                "acc": round(correct * 100 / total, 2) if total else 0.0,
            }
        return metrics
