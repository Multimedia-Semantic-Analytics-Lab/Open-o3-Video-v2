import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openo3_eval.runtime import get_logger

from openo3_eval.datasetpp.base import BaseDatasetPP
from openo3_eval.registry import DATASET_REGISTRY, DataSets
from openo3_eval.video_o3.message import build_video_o3_custom_message

logger = get_logger()

LEVELS = ("level-1", "level-2", "level-3", "level-4", "level-5")
VIDEOZEROBENCH_MODES = {"full_levels", "level_3_only"}
BOX_MODE = "normalized 0-1000"


def parse_json_field(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except Exception:
            return default
    return default


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def strip_code_fence(text: Any) -> str:
    if text is None:
        return ""
    value = str(text)
    match = re.search(r"```(?:json|python|bash|text)?\s*\n(.*?)\n```", value, flags=re.IGNORECASE | re.DOTALL)
    if match:
        value = match.group(1)
    return value.strip()


def extract_answer_text(text: Any) -> str:
    value = strip_code_fence(text)
    if not value:
        return ""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", value, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return value


def norm_answer(text: Any) -> str:
    value = extract_answer_text(text).strip()
    return re.sub(r'^[\s"\'“”‘’]+|[\s"\'“”‘’\.\。]+$', "", value)


def is_correct(gt: Any, pred: Any) -> bool:
    gt_text = norm_answer(gt)
    pred_text = norm_answer(pred)
    if not gt_text or not pred_text:
        return False
    if re.fullmatch(r"\d+", gt_text):
        return pred_text == gt_text
    if re.search(r"[A-Za-z]", gt_text):
        return gt_text.lower() == pred_text.lower()
    if "\u8272" in gt_text:
        return pred_text in gt_text
    if gt_text == "\u8f66":
        return gt_text in pred_text
    return gt_text == pred_text


def extract_gt_windows(sample: Dict[str, Any]) -> List[Tuple[float, float]]:
    windows = parse_json_field(sample.get("evidence_windows"), [])
    out: List[Tuple[float, float]] = []
    for window in windows if isinstance(windows, list) else []:
        if not isinstance(window, dict):
            continue
        start = safe_float(window.get("start"))
        end = safe_float(window.get("end"))
        if start is None or end is None or end <= start:
            continue
        out.append((float(start), float(end)))
    return out


def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    merged = [intervals[0]]
    for start, end in sorted(intervals[1:], key=lambda item: (item[0], item[1])):
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def total_length(intervals: List[Tuple[float, float]]) -> float:
    return sum(max(0.0, end - start) for start, end in intervals)


def intersection_intervals(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            out.append((start, end))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return out


def tiou_multi(gt: List[Tuple[float, float]], pred: List[Tuple[float, float]]) -> float:
    gt_merged = merge_intervals(sorted(gt))
    pred_merged = merge_intervals(sorted(pred))
    inter = total_length(intersection_intervals(gt_merged, pred_merged))
    union = total_length(gt_merged) + total_length(pred_merged) - inter
    return float(inter / union) if union > 0 else 0.0


def parse_pred_windows(text: Any) -> Optional[List[Tuple[float, float]]]:
    value = extract_answer_text(text)
    if not value:
        return None
    value = value.replace("<t>", "<").replace("</t>", " seconds>")

    def parse_time_token(token: str) -> Optional[float]:
        token = token.strip().strip("<>").replace("seconds", "").replace("second", "").replace("\u79d2", "").strip()
        if not token:
            return None
        if ":" in token:
            match = re.fullmatch(r"(\d+)\s*:\s*(\d{2})(?:\.(\d+))?", token)
            if not match:
                return None
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            if seconds >= 60:
                return None
            result = minutes * 60 + seconds
            if match.group(3):
                result += float("0." + match.group(3))
            return float(result)
        if re.fullmatch(r"\d+(?:\.\d+)?", token) is None:
            return None
        return float(token)

    time_token = r"(?:<?\s*\d+:\d{2}(?:\.\d+)?(?:\s*(?:seconds?|\u79d2))?\s*>?|<?\s*\d+(?:\.\d+)?(?:\s*(?:seconds?|\u79d2))?\s*>?)"
    patterns = [
        re.compile(rf"(?is)\b(?:from\s+)?({time_token})\s*(?:[^\d:]+)?\s*to\s*({time_token})\b"),
        re.compile(rf"(?is)\b({time_token})\s*[-–—~]\s*({time_token})\b"),
    ]
    out: List[Tuple[float, float]] = []
    for pattern in patterns:
        for match in pattern.finditer(value):
            start = parse_time_token(match.group(1))
            end = parse_time_token(match.group(2))
            if start is None or end is None or end <= start:
                continue
            out.append((float(start), float(end)))
    return out or None


def extract_gt_boxes_by_time(sample: Dict[str, Any], time_round: int = 2) -> Dict[float, List[List[float]]]:
    boxes = parse_json_field(sample.get("evidence_boxes"), [])
    out: Dict[float, List[List[float]]] = {}
    for item in boxes if isinstance(boxes, list) else []:
        if not isinstance(item, dict):
            continue
        time_value = safe_float(item.get("time"))
        box = item.get("box")
        if time_value is None or not (isinstance(box, list) and len(box) == 4):
            continue
        try:
            parsed_box = [float(coord) for coord in box]
        except Exception:
            continue
        out.setdefault(round(float(time_value), time_round), []).append(parsed_box)
    return out


def sanitize_box(box: List[float]) -> Optional[Tuple[float, float, float, float]]:
    if not (isinstance(box, list) and len(box) == 4):
        return None
    x1, y1, x2, y2 = [max(0.0, min(1.0, float(value))) for value in box]
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def union_area_rects(rects: List[Tuple[float, float, float, float]]) -> float:
    if not rects:
        return 0.0
    xs = sorted({rect[0] for rect in rects} | {rect[2] for rect in rects})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        ys: List[Tuple[float, float]] = []
        for x1, y1, x2, y2 in rects:
            if x1 < right and x2 > left:
                ys.append((y1, y2))
        if not ys:
            continue
        ys.sort()
        merged = [ys[0]]
        for start, end in ys[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        area += (right - left) * sum(max(0.0, end - start) for start, end in merged)
    return float(area)


def intersection_rect(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> Optional[Tuple[float, float, float, float]]:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def viou_for_time(gt_boxes: List[List[float]], pred_boxes: List[List[float]]) -> float:
    gt_rects = [rect for rect in (sanitize_box(box) for box in gt_boxes) if rect is not None]
    pred_rects = [rect for rect in (sanitize_box(box) for box in pred_boxes) if rect is not None]
    if not gt_rects:
        return 1.0
    if not pred_rects:
        return 0.0
    inter_rects = [
        rect
        for gt_rect in gt_rects
        for pred_rect in pred_rects
        for rect in [intersection_rect(gt_rect, pred_rect)]
        if rect is not None
    ]
    inter = union_area_rects(inter_rects)
    union = union_area_rects(gt_rects) + union_area_rects(pred_rects) - inter
    return float(inter / union) if union > 0 else 0.0


def viou_avg(sample: Dict[str, Any], pred_map: Optional[Dict[float, List[List[float]]]]) -> float:
    gt_map = extract_gt_boxes_by_time(sample)
    if not gt_map:
        return 1.0
    if pred_map is None:
        return 0.0
    scores = [viou_for_time(gt_map.get(time_key, []), pred_map.get(time_key, [])) for time_key in sorted(gt_map)]
    return float(sum(scores) / len(scores)) if scores else 1.0


def parse_pred_spatial_json(text: Any, box_mode: str = BOX_MODE) -> Optional[Dict[float, List[List[float]]]]:
    value = extract_answer_text(text)
    if not value:
        return None
    if value.startswith("{") and value.endswith("}"):
        value = "[" + value + "]"
    try:
        data = json.loads(value)
    except Exception:
        return None
    if not isinstance(data, list):
        return None

    out: Dict[float, List[List[float]]] = {}
    for item in data:
        if not isinstance(item, dict):
            return None
        time_value = safe_float(item.get("time"))
        boxes = item.get("bbox_2d")
        if time_value is None or not isinstance(boxes, list) or not boxes:
            return None
        raw_boxes = boxes if isinstance(boxes[0], list) else [boxes]
        parsed_boxes: List[List[float]] = []
        for box in raw_boxes:
            if not (isinstance(box, list) and len(box) == 4):
                return None
            try:
                if box_mode == "normalized 0-1000":
                    parsed_boxes.append([float(coord) / 1000.0 for coord in box])
                elif box_mode == "normalized 0-1":
                    parsed_boxes.append([float(coord) for coord in box])
                else:
                    return None
            except Exception:
                return None
        out[round(float(time_value), 2)] = parsed_boxes
    return out


@DATASET_REGISTRY.register(name=DataSets.videozerobench)
class VideoZeroBench(BaseDatasetPP):
    dataset_name = DataSets.videozerobench
    dataset_dir_env_key = "VIDEOZEROBENCH_DATASET_DIR"
    input_mode_env_key = "VIDEOZEROBENCH_INPUT_MODE"

    def get_mode(self) -> str:
        mode = str(getattr(self.args, "videozerobench_mode", "") or os.environ.get("VIDEOZEROBENCH_MODE", "full_levels"))
        mode = mode.strip().lower()
        if mode not in VIDEOZEROBENCH_MODES:
            supported = ", ".join(sorted(VIDEOZEROBENCH_MODES))
            raise ValueError(f"[Open-o3 Eval] Invalid videozerobench mode: {mode!r}. Supported: {supported}")
        return mode

    def _load_rows(self, dataset_path: Path) -> Tuple[List[Dict[str, Any]], Path, Path]:
        if dataset_path.is_file():
            if dataset_path.suffix != ".json":
                raise ValueError(f"[Open-o3 Eval] videozerobench expects a json file, got: {dataset_path}")
            with open(dataset_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            if not isinstance(rows, list):
                raise ValueError(f"[Open-o3 Eval] videozerobench json must contain a list of rows: {dataset_path}")
            return rows, dataset_path, dataset_path.parent

        anno_path = dataset_path / "VideoZeroBench_500_v0.json"
        if not anno_path.exists():
            raise FileNotFoundError(f"[Open-o3 Eval] videozerobench annotation not found: {anno_path}")
        with open(anno_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"[Open-o3 Eval] videozerobench json must contain a list of rows: {anno_path}")
        return rows, anno_path, dataset_path

    def _resolve_video_root(self, benchmark_root: Path) -> Path:
        video_root = benchmark_root / "compressed"
        if not video_root.exists():
            raise FileNotFoundError(f"[Open-o3 Eval] videozerobench video dir not found: {video_root}")
        return video_root

    def build_dataset(self) -> List[Dict[str, Any]]:
        dataset_path = self.get_dataset_dir()
        rows, anno_path, benchmark_root = self._load_rows(dataset_path)
        video_root = self._resolve_video_root(benchmark_root)

        self.dataset = []
        skipped_missing_video = 0
        for row in rows:
            video_name = str(row.get("video") or "").strip()
            video_path = video_root / video_name
            if not video_name or not video_path.exists():
                skipped_missing_video += 1
                continue

            pp_meta = {
                "gt_question": str(row.get("question") or "").strip(),
                "gt_answer": str(row.get("answer") or "").strip(),
                "gt_question_id": row.get("question_id"),
                "gt_video": video_name,
                "gt_video_id": str(row.get("video_id") or "").strip(),
                "gt_category": str(row.get("category") or "").strip(),
                "gt_language": str(row.get("language") or "").strip(),
                "gt_duration": safe_float(row.get("duration")),
                "gt_evidence_span": str(row.get("evidence_span") or "").strip(),
                "gt_capabilities": list(row.get("annotation_capabilities") or []),
                "gt_evidence_windows": parse_json_field(row.get("evidence_windows"), []),
                "gt_evidence_boxes": parse_json_field(row.get("evidence_boxes"), []),
            }
            self.dataset.append({"videos": [str(video_path)], "pp_meta": pp_meta})

        logger.info(
            f"[Open-o3 Eval] videozerobench loaded {len(self.dataset)} samples from {anno_path}; "
            f"skipped {skipped_missing_video} missing videos (mode={self.get_mode()})"
        )
        return self.dataset

    @staticmethod
    def _sample_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
        pp_meta = item.get("pp_meta", {})
        return {
            "question": pp_meta.get("gt_question"),
            "answer": pp_meta.get("gt_answer"),
            "question_id": pp_meta.get("gt_question_id"),
            "video": pp_meta.get("gt_video"),
            "video_id": pp_meta.get("gt_video_id"),
            "category": pp_meta.get("gt_category"),
            "language": pp_meta.get("gt_language"),
            "duration": pp_meta.get("gt_duration"),
            "evidence_span": pp_meta.get("gt_evidence_span"),
            "annotation_capabilities": pp_meta.get("gt_capabilities") or [],
            "evidence_windows": pp_meta.get("gt_evidence_windows") or [],
            "evidence_boxes": pp_meta.get("gt_evidence_boxes") or [],
        }

    @staticmethod
    def _format_temporal_evidence(evidence_windows: Any) -> Optional[str]:
        windows = parse_json_field(evidence_windows, [])
        parts = []
        for window in windows if isinstance(windows, list) else []:
            if not isinstance(window, dict):
                continue
            start = safe_float(window.get("start"))
            end = safe_float(window.get("end"))
            if start is None or end is None:
                continue
            parts.append(f"From <{start:.2f} seconds> to <{end:.2f} seconds>")
        if not parts:
            return None
        return "The temporal evidence for answering the question is: " + "; ".join(parts) + "."

    @staticmethod
    def _format_spatial_evidence(evidence_boxes: Any) -> Optional[str]:
        boxes = parse_json_field(evidence_boxes, [])
        parts = []
        for item in boxes if isinstance(boxes, list) else []:
            if not isinstance(item, dict):
                continue
            time_value = safe_float(item.get("time"))
            box = item.get("box")
            if time_value is None or not (isinstance(box, list) and len(box) == 4):
                continue
            try:
                # box mode: normalized 0-1000
                x1, y1, x2, y2 = [int(round(1000 * float(value))) for value in box]
            except Exception:
                continue
            parts.append(f"Time=<{time_value:.2f} seconds>, Normalized Box=[{x1},{y1},{x2},{y2}]")
        if not parts:
            return None
        return "The spatial evidence for answering the question is: " + "; ".join(parts) + "."

    @staticmethod
    def _key_times(sample: Dict[str, Any]) -> List[float]:
        seen = set()
        out = []
        for item in parse_json_field(sample.get("evidence_boxes"), []):
            if not isinstance(item, dict):
                continue
            time_value = safe_float(item.get("time"))
            if time_value is None:
                continue
            rounded = round(float(time_value), 3)
            if rounded in seen:
                continue
            seen.add(rounded)
            out.append(float(time_value))
        return out

    def _build_qa_prompt(
        self,
        sample: Dict[str, Any],
        *,
        use_temporal_hint: bool,
        use_spatial_hint: bool,
    ) -> str:
        lines = [
            "Answer the following question according to the content of the video.",
            f"Question: {sample.get('question', '')}",
        ]
        if use_temporal_hint:
            temporal_hint = self._format_temporal_evidence(sample.get("evidence_windows"))
            if temporal_hint:
                lines.append(temporal_hint)
        if use_spatial_hint:
            spatial_hint = self._format_spatial_evidence(sample.get("evidence_boxes"))
            if spatial_hint:
                lines.append(spatial_hint)
        lines.append("Put the final answer inside <answer> and </answer>.")
        return "\n".join(lines)

    @staticmethod
    def _build_temporal_prompt(question: str) -> str:
        return (
            f"Question: {question}\n"
            "Task: Find one or more of the most important key time ranges in the video that provide sufficient "
            "evidence to answer the question.\n"
            "Output format:\n"
            "From <t>start_timestamp_1</t>s to <t>end_timestamp_1</t>s. "
            "From <t>start_timestamp_2</t>s to <t>end_timestamp_2</t>s.\n"
            "Rules:\n"
            "- Output only the time ranges in the specified sentence format.\n"
            "- Do not answer the original question.\n"
            "- Use absolute time in seconds, not MM:SS.\n"
            "- Put the final result inside <answer> and </answer>.\n"
        )

    @staticmethod
    def _build_spatial_prompt(question: str, key_times: List[float]) -> str:
        times_text = ", ".join(f"<{time_value:.2f} seconds>" for time_value in key_times)
        return (
            f"Question: {question}\n"
            f"Given key time points (absolute seconds): {times_text}\n"
            "Task: For each provided time point, output one or more 2D bounding boxes that are relevant evidence "
            "for answering the question.\n"
            "Output format: a JSON array of objects.\n"
            '[{"time": ..., "bbox_2d":[[...],[...],...]}, {"time": ..., "bbox_2d":[[...],...]}, ...]\n'
            "Rules:\n"
            "- Output only valid JSON inside <answer> and </answer>.\n"
            "- The number of objects should match the number of provided time points.\n"
            "- Each object's 'time' must be one of the provided time points.\n"
            "- Each box uses normalized coordinates in [0,1000]: [x_min, y_min, x_max, y_max].\n"
        )

    @staticmethod
    def _direct_answer_prompt(prompt_text: str, answer_format: str) -> str:
        return (
            f"\n{prompt_text}\n"
            "Do not think out loud. Do not request more observations. "
            f"Output only `<answer>{answer_format}</answer>`.\n"
        )

    @staticmethod
    def _think_answer_prompt(prompt_text: str, answer_format: str) -> str:
        return (
            f"\n{prompt_text}\n"
            "First reason inside `<think>` and `</think>`, then output only "
            f"`<answer>{answer_format}</answer>`.\n"
        )

    def build_video_o3_tasks(self, args: Any, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        sample = self._sample_from_item(item)
        video_path = str((item.get("videos") or [""])[0])
        levels = ["level-3"] if self.get_mode() == "level_3_only" else list(LEVELS)
        tasks: List[Dict[str, Any]] = []

        for level in levels:
            task_max_turn = None
            question_type = "oe"
            if level == "level-1":
                prompt_text = self._build_qa_prompt(sample, use_temporal_hint=True, use_spatial_hint=True)
                answer_format = "YOUR_FINAL_ANSWER"
            elif level == "level-2":
                prompt_text = self._build_qa_prompt(sample, use_temporal_hint=True, use_spatial_hint=False)
                answer_format = "YOUR_FINAL_ANSWER"
            elif level == "level-3":
                prompt_text = self._build_qa_prompt(sample, use_temporal_hint=False, use_spatial_hint=False)
                answer_format = "YOUR_FINAL_ANSWER"
            elif level == "level-4":
                if not extract_gt_windows(sample):
                    tasks.append({"task_name": level, "skip": True, "error": "missing evidence_windows"})
                    continue
                prompt_text = self._build_temporal_prompt(str(sample.get("question") or ""))
                answer_format = "From <t>start_time</t>s to <t>end_time</t>s"
                question_type = "tg"
                task_max_turn = 0
            elif level == "level-5":
                key_times = self._key_times(sample)
                if not key_times:
                    tasks.append({"task_name": level, "skip": True, "error": "missing evidence_boxes"})
                    continue
                prompt_text = self._build_spatial_prompt(str(sample.get("question") or ""), key_times)
                answer_format = '[{"time": ..., "bbox_2d":[[...],[...]]}]'
                task_max_turn = 0
            else:
                raise ValueError(f"[Open-o3 Eval] Unknown videozerobench level: {level}")

            message, frame_meta = build_video_o3_custom_message(
                args,
                video_path=video_path,
                prompt_text=prompt_text,
                dataset_name=self.dataset_name,
                question_type=question_type,
                direct_prompt=self._direct_answer_prompt(prompt_text, answer_format),
                think_answer_prompt=self._think_answer_prompt(prompt_text, answer_format),
            )
            task = {"task_name": level, "message": message, "frame_meta": frame_meta, "skip": False}
            if task_max_turn is not None:
                task["max_turn"] = task_max_turn
            tasks.append(task)

        return tasks

    def merge_video_o3_task_results(self, item: Dict[str, Any], task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        predictions: Dict[str, Any] = {}
        meta_by_level: Dict[str, Any] = {}
        for task in task_results:
            level = str(task["task_name"])
            predictions[level] = {
                "task": "qa" if level in {"level-1", "level-2", "level-3"} else ("temporal_grounding" if level == "level-4" else "spatial_grounding"),
                "model_answer": task.get("response"),
                "error": task.get("error"),
            }
            meta_by_level[level] = {
                "raw_response": task.get("raw_response"),
                "frame_meta": task.get("frame_meta"),
                "turn_count": task.get("turn_count"),
                "inference_time": task.get("inference_time"),
                "inference_time_generate": task.get("inference_time_generate"),
                "dialogue_turns": task.get("dialogue_turns") or [],
            }

        if self.get_mode() == "level_3_only":
            level3 = predictions.get("level-3", {})
            return {
                "response": level3.get("model_answer", ""),
                "dialogue_turns": meta_by_level.get("level-3", {}).get("dialogue_turns", []),
                "video_o3": meta_by_level.get("level-3", {}),
                "videozerobench_predictions": predictions,
            }

        return {
            "response": json.dumps(predictions, ensure_ascii=False),
            "dialogue_turns_by_level": {level: meta.get("dialogue_turns", []) for level, meta in meta_by_level.items()},
            "video_o3_levels": meta_by_level,
            "videozerobench_predictions": predictions,
        }

    def calc_metric(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        mode = self.get_mode()
        if mode == "level_3_only":
            total = len(data_list)
            correct = 0
            for item in data_list:
                pp_meta = item.get("pp_meta", {})
                gold = pp_meta.get("gt_answer") or ""
                pred = item.get("response") or ""
                correct += int(is_correct(gold, pred))
            return {
                "mode": mode,
                "Total_questions": total,
                "Level-3_acc": round(correct * 100 / total, 2) if total > 0 else 0.0,
            }

        total = len(data_list)
        score_l1 = 0.0
        score_l2 = 0.0
        score_l3 = 0.0
        score_l4 = 0.0
        score_l5 = 0.0
        sum_tiou = 0.0
        sum_viou = 0.0
        temporal_valid = 0
        spatial_valid = 0

        for item in data_list:
            pp_meta = item.get("pp_meta", {})
            sample = {
                "answer": pp_meta.get("gt_answer"),
                "evidence_windows": pp_meta.get("gt_evidence_windows"),
                "evidence_boxes": pp_meta.get("gt_evidence_boxes"),
            }
            preds = item.get("videozerobench_predictions") or {}
            l1_pred = preds.get("level-1", {})
            l2_pred = preds.get("level-2", {})
            l3_pred = preds.get("level-3", {})
            l4_pred = preds.get("level-4", {})
            l5_pred = preds.get("level-5", {})

            acc1 = 1.0 if is_correct(sample["answer"], l1_pred.get("model_answer")) else 0.0
            acc2 = 1.0 if is_correct(sample["answer"], l2_pred.get("model_answer")) else 0.0
            acc3 = 1.0 if is_correct(sample["answer"], l3_pred.get("model_answer")) else 0.0
            score_l1 += acc1
            score_l2 += acc2
            score_l3 += acc3

            gt_windows = extract_gt_windows(sample)
            tiou = 0.0
            if gt_windows:
                pred_windows = parse_pred_windows(l4_pred.get("model_answer"))
                if pred_windows is not None:
                    tiou = tiou_multi(gt_windows, pred_windows)
                temporal_valid += 1
                sum_tiou += tiou
            if acc3 > 0 and tiou > 0.3:
                score_l4 += 1.0

            gt_box_map = extract_gt_boxes_by_time(sample)
            viou = 0.0
            if gt_box_map:
                pred_map = parse_pred_spatial_json(l5_pred.get("model_answer"))
                if pred_map is not None:
                    viou = viou_avg(sample, pred_map)
                spatial_valid += 1
                sum_viou += viou
            if acc3 > 0 and tiou > 0.3 and viou > 0.3:
                score_l5 += 1.0

        denom = float(total) if total > 0 else 1.0
        return {
            "mode": mode,
            "Total_questions": total,
            "Level-1_acc": round(score_l1 / denom * 100, 2),
            "Level-2_acc": round(score_l2 / denom * 100, 2),
            "Level-3_acc": round(score_l3 / denom * 100, 2),
            "Level-4_mean_tIoU": round((sum_tiou / temporal_valid * 100) if temporal_valid > 0 else 0.0, 2),
            "Level-4_score": round(score_l4 / denom * 100, 2),
            "Level-5_mean_vIoU": round((sum_viou / spatial_valid * 100) if spatial_valid > 0 else 0.0, 2),
            "Level-5_score": round(score_l5 / denom * 100, 2),
        }

    def build_sample_eval(self, item: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        mode = self.get_mode()
        pp_meta = result.get("pp_meta", {})
        gold_answer = pp_meta.get("gt_answer") or ""

        if mode == "level_3_only":
            pred = result.get("response") or ""
            acc = float(is_correct(gold_answer, pred))
            return {
                "sample_eval": {
                    "metric_type": "videozerobench_level3",
                    "level-3_pred": pred,
                    "level-3_gold": gold_answer,
                    "level-3_acc": acc,
                    "acc": acc,
                }
            }

        preds = result.get("videozerobench_predictions") or {}
        l1_pred = preds.get("level-1", {})
        l2_pred = preds.get("level-2", {})
        l3_pred = preds.get("level-3", {})
        l4_pred = preds.get("level-4", {})
        l5_pred = preds.get("level-5", {})

        acc1 = float(is_correct(gold_answer, l1_pred.get("model_answer")))
        acc2 = float(is_correct(gold_answer, l2_pred.get("model_answer")))
        acc3 = float(is_correct(gold_answer, l3_pred.get("model_answer")))

        gt_windows = extract_gt_windows({"evidence_windows": pp_meta.get("gt_evidence_windows")})
        pred_windows = parse_pred_windows(l4_pred.get("model_answer"))
        tiou = tiou_multi(gt_windows, pred_windows) if gt_windows and pred_windows is not None else 0.0

        gt_box_map = extract_gt_boxes_by_time({"evidence_boxes": pp_meta.get("gt_evidence_boxes")})
        pred_map = parse_pred_spatial_json(l5_pred.get("model_answer"))
        viou = viou_avg({"evidence_boxes": pp_meta.get("gt_evidence_boxes")}, pred_map) if gt_box_map and pred_map is not None else 0.0

        return {
            "sample_eval": {
                "metric_type": "videozerobench_full_levels",
                "level-1_acc": acc1,
                "level-2_acc": acc2,
                "level-3_acc": acc3,
                "level-4_tiou": round(float(tiou), 6),
                "level-4_score": float(acc3 > 0.0 and tiou > 0.3),
                "level-5_viou": round(float(viou), 6),
                "level-5_score": float(acc3 > 0.0 and tiou > 0.3 and viou > 0.3),
            }
        }
