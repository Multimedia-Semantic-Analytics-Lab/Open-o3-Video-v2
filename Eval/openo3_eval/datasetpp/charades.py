import ast
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from openo3_eval.datasetpp.base import BaseDatasetPP
from openo3_eval.registry import DATASET_REGISTRY, AnswerParsers, DataSets
from openo3_eval.runtime import get_logger

logger = get_logger()


def _parse_intervals(raw_value: Any) -> List[List[float]]:
    if isinstance(raw_value, list):
        parsed = raw_value
    else:
        raw_text = str(raw_value or "").strip()
        if not raw_text:
            return []
        try:
            parsed = json.loads(raw_text)
        except Exception:
            parsed = ast.literal_eval(raw_text)

    intervals: List[List[float]] = []
    for item in parsed:
        if isinstance(item, dict):
            item = item.get("timestamp", [])
        if isinstance(item, list) and len(item) == 2:
            intervals.append([float(item[0]), float(item[1])])
    return intervals


def _extract_intervals_from_json_item(item: Dict[str, Any]) -> List[List[float]]:
    solution = item.get("solution", {})
    clues = solution.get("clue", []) if isinstance(solution, dict) else []
    intervals = []
    for clue in clues:
        timestamp = clue.get("timestamp") if isinstance(clue, dict) else None
        if isinstance(timestamp, list) and len(timestamp) == 2:
            intervals.append([float(timestamp[0]), float(timestamp[1])])
    return intervals


def _resolve_charades_json(dataset_dir: Path) -> Path:
    for env_key in ("CHARADES_JSON_PATH", "Charades_json_PATH"):
        env_value = os.environ.get(env_key)
        if env_value:
            path = Path(env_value)
            if path.exists():
                return path
            raise FileNotFoundError(f"[Open-o3 Eval] {env_key} points to missing file: {path}")

    if dataset_dir.is_file():
        return dataset_dir

    candidates = [
        dataset_dir / "Charades.tsv",
        dataset_dir / "charades_test.json",
        dataset_dir / "Charades.json",
        dataset_dir / "data_info" / "charades_test.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"[Open-o3 Eval] Charades annotation not found under {dataset_dir}. "
        "Expected Charades.tsv, charades_test.json, or set CHARADES_JSON_PATH."
    )


def _resolve_video_path(dataset_dir: Path, video_value: str) -> str:
    video_path = Path(video_value)
    if video_path.is_absolute() and video_path.exists():
        return str(video_path)

    roots = []
    for env_key in ("CHARADES_VIDEO_ROOT", "Charades_PATH"):
        env_value = os.environ.get(env_key)
        if env_value:
            roots.append(Path(env_value))
    root = dataset_dir if dataset_dir.is_dir() else dataset_dir.parent
    roots.extend([root, root / "data", root / "video", root / "videos"])

    for base in roots:
        candidate = base / video_value
        if candidate.exists():
            return str(candidate)
    return str((root / "data" / video_value).resolve())


def _load_charades_rows(dataset_dir: Path) -> List[Dict[str, Any]]:
    anno_path = _resolve_charades_json(dataset_dir)
    if anno_path.suffix.lower() == ".tsv":
        with open(anno_path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    with open(anno_path, "r", encoding="utf-8") as f:
        raw_rows = json.load(f)

    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_rows):
        intervals = _extract_intervals_from_json_item(item)
        rows.append(
            {
                "index": idx,
                "doc_id": item.get("doc_id") or f"charades_{idx}",
                "video": item.get("video") or "",
                "duration": item.get("duration") or "",
                "question": item.get("problem") or item.get("question") or "",
                "answer": json.dumps(intervals),
                "clue_intervals": json.dumps(intervals),
                "task_mode": "miou",
                "data_source": item.get("data_source") or "",
            }
        )
    return rows


def _merge_intervals(intervals: List[List[float]]) -> List[List[float]]:
    if not intervals:
        return []
    sorted_intervals = sorted(([float(s), float(e)] for s, e in intervals), key=lambda item: item[0])
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return merged


def _interval_iou(gold: List[List[float]], pred: List[List[float]]) -> float:
    gold_merged = _merge_intervals(gold)
    pred_merged = _merge_intervals(pred)
    gold_len = sum(end - start for start, end in gold_merged)
    pred_len = sum(end - start for start, end in pred_merged)
    intersection = 0.0
    for gold_start, gold_end in gold_merged:
        for pred_start, pred_end in pred_merged:
            intersection += max(0.0, min(gold_end, pred_end) - max(gold_start, pred_start))
    union = gold_len + pred_len - intersection
    return intersection / union if union > 0 else 0.0


@DATASET_REGISTRY.register(name=DataSets.charades)
class Charades(BaseDatasetPP):
    """Charades-STA temporal grounding benchmark."""

    user_prompt_default = (
        "<video>\n"
        "Please analyze the provided video and locate the video segment according to the given query.\n"
        "Query: {question}"
    )

    dataset_name = DataSets.charades
    answer_parser = AnswerParsers.charades_parser

    dataset_dir_env_key = "CHARADES_DATASET_DIR"
    user_prompt_env_key = "CHARADES_USER_PROMPT"
    input_mode_env_key = "CHARADES_INPUT_MODE"
    sample_fps_env_key = "CHARADES_SAMPLE_FPS"
    resize_env_key = "CHARADES_RESIZE"

    def get_dataset_dir(self) -> Path:
        dataset_dir = self.args.dataset_dir
        if not dataset_dir:
            dataset_dir = os.environ.get("CHARADES_DATASET_DIR", "") or os.environ.get("Charades_PATH", "")
        if not dataset_dir:
            raise ValueError("[Open-o3 Eval] dataset_dir not specified. Set --dataset_dir, CHARADES_DATASET_DIR, or Charades_PATH.")
        path = Path(dataset_dir)
        if not path.exists():
            raise FileNotFoundError(f"[Open-o3 Eval] dataset_dir not found: {path}")
        return path

    def build_dataset(self) -> List[Dict[str, Any]]:
        dataset_dir = self.get_dataset_dir()
        self.dataset = []
        skipped_missing_video = 0
        skipped_no_interval = 0

        for row in _load_charades_rows(dataset_dir):
            question = str(row.get("question") or row.get("problem") or "").strip()
            video = str(row.get("video") or "").strip()
            if not question or not video:
                continue
            try:
                intervals = _parse_intervals(row.get("clue_intervals") or row.get("answer"))
            except Exception:
                skipped_no_interval += 1
                continue
            if not intervals:
                skipped_no_interval += 1
                continue

            video_path = _resolve_video_path(dataset_dir, video)
            if not os.path.exists(video_path):
                skipped_missing_video += 1
                continue

            duration_raw = row.get("duration") or ""
            try:
                duration = float(duration_raw) if str(duration_raw).strip() else 0.0
            except ValueError:
                duration = 0.0

            pp_meta = {
                "gt_answer": intervals,
                "gt_intervals": intervals,
                "gt_question": question,
                "gt_index": str(row.get("index") or "").strip(),
                "gt_doc_id": str(row.get("doc_id") or "").strip(),
                "gt_video": video,
                "gt_duration": duration,
                "gt_task_mode": str(row.get("task_mode") or "miou").strip(),
                "gt_data_source": str(row.get("data_source") or "").strip(),
            }
            prompt = self.build_user_prompt(prompt_replacements=None, question=question)
            self.dataset.append(
                {
                    "messages": [{"role": "user", "content": prompt}],
                    "videos": [video_path],
                    "pp_meta": pp_meta,
                }
            )

        logger.info(
            f"[Open-o3 Eval] Charades loaded {len(self.dataset)} samples from {dataset_dir}; "
            f"skipped {skipped_missing_video} missing videos, {skipped_no_interval} empty/invalid intervals"
        )
        return self.dataset

    def calc_metric(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        scores: List[float] = []
        total = len(data_list)
        valid = 0
        r1_counts = {0.3: 0, 0.5: 0, 0.7: 0}

        for item in data_list:
            pp_meta = item.get("pp_meta", {})
            gold = pp_meta.get("gt_intervals") or pp_meta.get("gt_answer") or []
            if isinstance(gold, str):
                gold = _parse_intervals(gold)
            duration = pp_meta.get("gt_duration") or 0
            try:
                duration_float = float(duration)
            except (TypeError, ValueError):
                duration_float = 0.0

            pred = self.parse_answer(item.get("response", ""), duration=duration_float)
            if pred is None:
                continue

            score = _interval_iou(gold, pred)
            valid += 1
            scores.append(score)
            for threshold in r1_counts:
                if score >= threshold:
                    r1_counts[threshold] += 1

        mean_iou = round(sum(scores) / valid, 4) if valid else 0
        return {
            "overall": {
                "total": total,
                "valid": valid,
                "miou": mean_iou,
                "R@1_IoU=0.3": round(r1_counts[0.3] / total * 100, 2) if total else 0,
                "R@1_IoU=0.5": round(r1_counts[0.5] / total * 100, 2) if total else 0,
                "R@1_IoU=0.7": round(r1_counts[0.7] / total * 100, 2) if total else 0,
            }
        }

    def build_sample_eval(self, item: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        pp_meta = result.get("pp_meta", {})
        gold = pp_meta.get("gt_intervals") or pp_meta.get("gt_answer") or []
        if isinstance(gold, str):
            gold = _parse_intervals(gold)
        duration = pp_meta.get("gt_duration") or 0
        try:
            duration_float = float(duration)
        except (TypeError, ValueError):
            duration_float = 0.0

        pred = self.parse_answer(result.get("response", ""), duration=duration_float)
        score = _interval_iou(gold, pred) if pred is not None else 0.0
        return {
            "sample_eval": {
                "metric_type": "temporal_grounding",
                "pred_intervals": pred,
                "gold_intervals": gold,
                "is_valid": pred is not None,
                "iou": round(float(score), 6),
                "acc@0.3": float(score >= 0.3),
                "acc@0.5": float(score >= 0.5),
                "acc@0.7": float(score >= 0.7),
            }
        }
