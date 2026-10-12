import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openo3_eval.datasetpp.base import BaseDatasetPP
from openo3_eval.datasetpp.paths import DEFAULT_LVBENCH_MEDIA_ROOT
from openo3_eval.registry import DATASET_REGISTRY, AnswerParsers, DataSets
from openo3_eval.runtime import get_logger
from openo3_eval.vl_utils.extract_frames import batch_extract_frames

logger = get_logger()

CHOICE_LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
QUESTION_OPTION_SPLIT_RE = re.compile(r"\n\s*\([A-Z]\)")
OPTION_RE = re.compile(r"\(([A-Z])\)\s*(.*?)(?=\n\s*\([A-Z]\)|$)", re.S)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_annotation_path() -> Path:
    return _repo_root() / "data" / "lvbench.json"


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _extract_question_and_candidates(raw_question: Any) -> Tuple[str, List[str]]:
    raw_text = str(raw_question or "").strip()
    if not raw_text:
        return "", []

    parts = QUESTION_OPTION_SPLIT_RE.split(raw_text, maxsplit=1)
    question = parts[0].strip()
    options = [
        _normalize_text(option_text)
        for _, option_text in OPTION_RE.findall(raw_text)
        if _normalize_text(option_text)
    ]
    return question, options


def _format_choices(candidates: List[str]) -> List[str]:
    return [f"{CHOICE_LETTERS[idx]}. {candidate}" for idx, candidate in enumerate(candidates)]


def _load_lvbench_rows(dataset_path: Path) -> Tuple[List[Dict[str, Any]], Path]:
    if dataset_path.is_file():
        if dataset_path.suffix != ".json":
            raise ValueError(f"[Open-o3 Eval] LVBench only supports json annotation files: {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"[Open-o3 Eval] LVBench json must contain a list of rows: {dataset_path}")
        return rows, dataset_path

    candidates = [
        dataset_path / "lvbench.json",
        _repo_annotation_path(),
    ]
    anno_path = next((path for path in candidates if path.exists()), None)
    if anno_path is None:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"[Open-o3 Eval] LVBench annotation not found. searched: {searched}")
    return _load_lvbench_rows(anno_path)


def _resolve_media_root(dataset_path: Path) -> Path:
    candidates: List[Path] = []
    env_root = os.environ.get("LVBENCH_MEDIA_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root))
    if dataset_path.is_dir():
        candidates.append(dataset_path)
    else:
        candidates.append(dataset_path.parent)
        candidates.append(dataset_path.parent.parent)
    candidates.append(DEFAULT_LVBENCH_MEDIA_ROOT)

    for candidate in candidates:
        if (candidate / "video_chunks").exists():
            return candidate
    return candidates[0]


@DATASET_REGISTRY.register(name=DataSets.lvbench)
class LVBench(BaseDatasetPP):
    user_prompt_default = Path(__file__).resolve().parents[1] / "prompts" / "lvbench.txt"
    dataset_name = DataSets.lvbench
    answer_parser = AnswerParsers.video_mme_parser

    dataset_dir_env_key = "LVBENCH_DATASET_DIR"
    user_prompt_env_key = "LVBENCH_USER_PROMPT"
    input_mode_env_key = "LVBENCH_INPUT_MODE"
    sample_fps_env_key = "LVBENCH_SAMPLE_FPS"
    resize_env_key = "LVBENCH_RESIZE"

    def build_dataset(self) -> List[Dict[str, Any]]:
        dataset_path = self.get_dataset_dir()
        input_mode = self.get_input_mode()
        sample_fps = self.get_sample_fps()
        resize = self.get_resize()
        frame_cache_dir = self.get_frame_cache_dir(self.dataset_name)

        rows, anno_path = _load_lvbench_rows(dataset_path)
        media_root = _resolve_media_root(dataset_path)
        self.dataset = []

        all_items: List[Tuple[Dict[str, Any], str, List[str], str, str]] = []
        skipped_missing_video = 0
        skipped_missing_answer = 0
        skipped_invalid_question = 0

        for row in rows:
            question = _normalize_text(row.get("question"))
            candidates = row.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                question, candidates = _extract_question_and_candidates(row.get("raw_question") or row.get("question"))
            candidates = [_normalize_text(candidate) for candidate in candidates if _normalize_text(candidate)]
            if not question or len(candidates) < 2:
                skipped_invalid_question += 1
                continue

            answer = str(row.get("answer") or "").strip().upper()
            if answer not in CHOICE_LETTERS[: len(candidates)]:
                skipped_missing_answer += 1
                continue

            video_rel_path = str(row.get("video_rel_path") or f"video_chunks/{row.get('video_id', '')}.mp4").strip()
            video_path = media_root / video_rel_path
            if not video_path.exists():
                skipped_missing_video += 1
                continue

            formatted_choices = _format_choices(candidates)
            all_items.append((row, str(video_path), candidates, "\n".join(formatted_choices), question))

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
        for row, video_path, candidates, choice_text, question in all_items:
            pp_meta = {
                "gt_answer": str(row.get("answer") or "").strip().upper(),
                "gt_candidates": candidates,
                "gt_question": question,
                "gt_options_text": choice_text,
                "gt_question_id": str(row.get("id") or "").strip(),
                "gt_video_id": str(row.get("video_id") or "").strip(),
                "gt_video_type": str(row.get("video_type") or "").strip(),
                "gt_time_reference": str(row.get("time_reference") or "").strip(),
                "gt_question_types": list(row.get("question_types") or []),
                "gt_raw_question": str(row.get("raw_question") or "").strip(),
            }
            prompt_kwargs = {"question": question, "choice": choice_text}

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
            f"[Open-o3 Eval] LVBench loaded {len(self.dataset)} samples from {anno_path}; "
            f"skipped {skipped_missing_video} missing videos, {skipped_missing_answer} rows without GT answer, "
            f"{skipped_invalid_question} malformed questions, {skipped_frames} frame failures (mode={input_mode})"
        )
        return self.dataset

    def build_sample_eval(self, item: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        pp_meta = item.get("pp_meta", {})
        pred = self.parse_answer(result.get("response", ""), candidates=pp_meta.get("gt_candidates") or [])
        gold = str(pp_meta.get("gt_answer") or "").strip().upper()
        return {
            "lvbench_pred": pred,
            "lvbench_gold": gold,
            "lvbench_correct": float(bool(pred and gold and pred == gold)),
        }

    def calc_metric(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        groups: Dict[str, List[Dict[str, Any]]] = {"overall": data_list}
        for row in data_list:
            pp_meta = row.get("pp_meta", {})
            video_type = str(pp_meta.get("gt_video_type") or "").strip()
            if video_type:
                groups.setdefault(f"video_type/{video_type}", []).append(row)
            for question_type in pp_meta.get("gt_question_types") or []:
                value = str(question_type).strip()
                if value:
                    groups.setdefault(f"question_type/{value}", []).append(row)

        metrics: Dict[str, Any] = {}
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
