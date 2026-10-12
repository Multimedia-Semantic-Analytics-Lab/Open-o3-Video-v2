import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openo3_eval.runtime import get_logger

from openo3_eval.datasetpp.base import BaseDatasetPP
from openo3_eval.registry import DATASET_REGISTRY, DataSets
from openo3_eval.video_o3.message import build_video_o3_custom_message
from openo3_eval.video_o3.prompts import (
    SYSTEM_PROMPT_DIRECT_ANSWER_OE,
    SYSTEM_PROMPT_DIRECT_ANSWER_TG,
    build_direct_temporal_grounding_prompt,
)

logger = get_logger()

DEFAULT_VSTAR_EVAL_SCRIPT = str(Path(__file__).resolve().parents[1] / "external" / "eval_vstar.py")
DEFAULT_VSTAR_EVAL_CUDA_VISIBLE_DEVICES = "0,1,2,3"


def _extract_answer_text(text: Any) -> str:
    if text is None:
        return ""
    value = str(text).strip()
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", value, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return value


def _extract_timestamps(text: Any) -> List[float]:
    value = _extract_answer_text(text)
    if not value:
        return []

    def _replace_mmss(match: re.Match[str]) -> str:
        minutes, seconds = map(int, match.group(0).split(":"))
        return str(minutes * 60 + seconds)

    value = re.sub(r"\b\d+:\d{2}\b", _replace_mmss, value)
    matches = re.findall(r"\b\d+(?:\.\d+)?\b", value)
    return [float(matches[0]), float(matches[1])] if len(matches) >= 2 else []


def _fix_incomplete_json(json_str: str) -> str:
    open_square = json_str.count("[")
    close_square = json_str.count("]")
    open_curly = json_str.count("{")
    close_curly = json_str.count("}")
    if open_square > close_square:
        json_str += "]" * (open_square - close_square)
    elif close_square > open_square:
        json_str = "[" * (close_square - open_square) + json_str
    if open_curly > close_curly:
        json_str += "}" * (open_curly - close_curly)
    elif close_curly > open_curly:
        json_str = "{" * (close_curly - open_curly) + json_str
    return json_str


def _extract_json_candidates(value: str) -> List[str]:
    candidates: List[str] = []
    fenced_match = re.search(r"```json\s*(.*?)\s*(?:```|$)", value, re.DOTALL | re.IGNORECASE)
    search_spaces = [fenced_match.group(1) if fenced_match else value, value]

    for text in search_spaces:
        if not text:
            continue
        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            if start == -1:
                continue
            end = text.rfind(closer)
            if end >= start:
                candidates.append(text[start : end + 1].strip())
            candidates.append(text[start:].strip())

    deduped: List[str] = []
    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return deduped


def _normalize_box_qwen3(box: Iterable[float], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [
        int(x1 / width * 1000),
        int(y1 / height * 1000),
        int(x2 / width * 1000),
        int(y2 / height * 1000),
    ]


def _denormalize_box_qwen3(box: Any, width: int, height: int) -> Optional[List[int]]:
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return None
    return [
        int(x1 / 1000 * width),
        int(y1 / 1000 * height),
        int(x2 / 1000 * width),
        int(y2 / 1000 * height),
    ]


def _extract_bounding_boxes(answer_text: Any, width: int, height: int) -> Optional[Dict[str, List[int]]]:
    value = _extract_answer_text(answer_text)
    if not value:
        return None

    for payload in _extract_json_candidates(value):
        payload = payload.replace("'", '"')
        candidates = [payload, _fix_incomplete_json(payload)]
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                merged: Dict[str, Any] = {}
                for item in parsed:
                    merged.update(item)
                parsed = merged
            if isinstance(parsed, list):
                out: Dict[str, List[int]] = {}
                for item in parsed:
                    if not (isinstance(item, list) and len(item) == 2):
                        continue
                    key = str(item[0])
                    denorm = _denormalize_box_qwen3(item[1], width, height)
                    if denorm is not None:
                        out[key] = denorm
                if out:
                    return out
                continue
            if isinstance(parsed, dict):
                out = {}
                for key, box in parsed.items():
                    denorm = _denormalize_box_qwen3(box, width, height)
                    if denorm is not None:
                        out[str(key)] = denorm
                if out:
                    return out
                continue
    return None


def _build_video_index(video_root: Path) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for root, _, files in os.walk(video_root):
        for file_name in files:
            if not file_name.lower().endswith(".mp4"):
                continue
            stem = Path(file_name).stem
            index.setdefault(stem, str(Path(root) / file_name))
    return index


def _float_metrics(line: str) -> Dict[str, float]:
    return {
        match.group(1).strip(): float(match.group(2))
        for match in re.finditer(r"([A-Za-z0-9@&=\.\- ]+):\s*(-?\d+(?:\.\d+)?)", line)
    }


@DATASET_REGISTRY.register(name=DataSets.v_star)
class VStar(BaseDatasetPP):
    dataset_name = DataSets.v_star

    dataset_dir_env_key = "V_STAR_DATASET_DIR"
    input_mode_env_key = "V_STAR_INPUT_MODE"

    def _resolve_anno_and_video_root(self, dataset_path: Path) -> Tuple[Path, Path]:
        if dataset_path.is_file():
            anno_path = dataset_path
            benchmark_root = dataset_path.parent
        else:
            benchmark_root = dataset_path
            anno_path = benchmark_root / "V_STaR_test.json"
            if not anno_path.exists():
                raise FileNotFoundError(f"[Open-o3 Eval] V-STAR annotation not found: {anno_path}")
        video_root = Path(os.environ.get("V_STAR_VIDEO_DIR", str(benchmark_root / "videos")))
        if not video_root.exists():
            raise FileNotFoundError(f"[Open-o3 Eval] V-STAR video directory not found: {video_root}")
        return anno_path, video_root

    def build_dataset(self) -> List[Dict[str, Any]]:
        dataset_path = self.get_dataset_dir()
        anno_path, video_root = self._resolve_anno_and_video_root(dataset_path)
        with open(anno_path, "r", encoding="utf-8") as f:
            raw_rows = json.load(f)
        if not isinstance(raw_rows, list):
            raise ValueError(f"[Open-o3 Eval] V-STAR annotation must be a list: {anno_path}")

        video_index = _build_video_index(video_root)
        self.dataset = []
        missing_videos = 0
        for row in raw_rows:
            vid = str(row.get("vid") or "").strip()
            video_path = video_index.get(vid)
            if not vid or not video_path:
                missing_videos += 1
                continue
            question = str(row.get("question") or "").strip()
            item = {
                **row,
                "messages": [{"role": "user", "content": question}],
                "videos": [video_path],
                "pp_meta": {
                    "gt_question": question,
                    "gt_answer": str(row.get("answer") or "").strip(),
                    "gt_vid": vid,
                },
            }
            self.dataset.append(item)

        logger.info(
            f"[Open-o3 Eval] V-STAR loaded {len(self.dataset)} samples from {anno_path}; "
            f"skipped {missing_videos} missing videos"
        )
        return self.dataset

    def build_video_o3_tasks(self, args: Any, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        video_path = str((item.get("videos") or [""])[0])
        video_length = round(float(item.get("frame_count", 0)) / max(float(item.get("fps", 1) or 1), 1e-6), 2)
        width = int(item.get("width") or 1)
        height = int(item.get("height") or 1)
        task_max_turn = 0 if getattr(args, "max_turn", None) is None else int(getattr(args, "max_turn"))
        question = str(item.get("question") or "").strip()
        temporal_question = str(item.get("temporal_question") or "").strip()
        spatial_question = str(item.get("spatial_question") or "").strip()
        spatial_question_2 = str(item.get("spatial_question_2") or "").strip()
        boxes = [
            _normalize_box_qwen3(
                [box["xmin"], box["ymin"], box["xmax"], box["ymax"]],
                width,
                height,
            )
            for box in item.get("bboxes", [])
            if isinstance(box, dict) and all(key in box for key in ("xmin", "ymin", "xmax", "ymax"))
        ]
        st = math.ceil(float((item.get("timestamps") or [0, 0])[0]))
        et = math.floor(float((item.get("timestamps") or [0, 0])[1]))
        time_range = list(range(st, et + 1)) if et >= st else []

        task_specs = [
            (
                "answer_vqa",
                {
                    "prompt_text": (
                        f"Answer the question about the video in one sentence: {question}\n"
                        "(If the answer is a person, you do not need to identify the person.)\n"
                        "Output only <answer>YOUR_ANSWER</answer>."
                    ),
                    "direct_prompt": (
                        f"\nAnswer the question about the video in one sentence: {question}\n"
                        "(If the answer is a person, you do not need to identify the person.)\n"
                        "Do not think out loud. Do not request more observations. "
                        "Output only <answer>YOUR_ANSWER</answer>.\n"
                    ),
                },
                SYSTEM_PROMPT_DIRECT_ANSWER_OE,
                "oe",
            ),
            (
                "answer_temporal",
                {
                    "prompt_text": (
                        f"This video is {video_length} seconds long.\n"
                        f"Answer the question about the video: {temporal_question}\n"
                        "Directly output the start and end moment timestamps.\n"
                        "Output only <answer>From <t>start_time</t>s to <t>end_time</t>s</answer>."
                    ),
                    "direct_prompt": build_direct_temporal_grounding_prompt(temporal_question),
                },
                SYSTEM_PROMPT_DIRECT_ANSWER_TG,
                "tg",
            ),
            (
                "answer_spatial",
                {
                    "prompt_text": (
                        f"Please answer the question about the video: {spatial_question} "
                        "with a series of bounding boxes in [x1, y1, x2, y2] format.\n"
                        f"For each whole second within the time range {time_range} provided (inclusive of the boundaries), output a series of bounding boxes of the object in JSON format. "
                        "The keys should be the whole seconds (as strings), and the values should be the box in [x1, y1, x2, y2] format."
                        f"Example output: <answer>{{\"{time_range[0] if time_range else 0}\": [x1, y1, x2, y2], ...}}</answer>"
                    ),
                    "direct_prompt": (
                        f"Please answer the question about the video: {spatial_question} "
                        "with a series of bounding boxes in [x1, y1, x2, y2] format.\n"
                        f"For each whole second within the time range {time_range} provided (inclusive of the boundaries), output a series of bounding boxes of the object in JSON format. "
                        "The keys should be the whole seconds (as strings), and the values should be the box in [x1, y1, x2, y2] format."
                        f"Example output: <answer>{{\"{time_range[0] if time_range else 0}\": [x1, y1, x2, y2], ...}}</answer>"
                        "Do not output reasoning process. Do not request more observations. Output only the final JSON inside <answer>...</answer>.\n"
                    ),
                },
                SYSTEM_PROMPT_DIRECT_ANSWER_OE,
                "oe",
            ),
            (
                "answer_spatial_2",
                {
                    "prompt_text": (
                        f"Please answer the question about the video: {spatial_question_2} "
                        "with a series of bounding boxes in [x1, y1, x2, y2] format.\n"
                        "For each whole second that may related to the question, output a series of bounding boxes of the object in JSON format." 
                        f"You only need to output {len(boxes)} bbox(es). You need to determine which frame is related to the question, and you don't need to output the bbox for the frames not related to the question."
                        "The keys should be the whole seconds (as strings), and the values should be the bounding box in [x1, y1, x2, y2] format.\n"
                        f"Example output: <answer>{{\"0\": [x1, y1, x2, y2], \"1\":..., ..., \"{len(boxes)}\":...}}</answer> (if the frames at 0~{len(boxes)} second are related to the questions)"
                    ),
                    "direct_prompt": (
                        f"Please answer the question about the video: {spatial_question_2} "
                        "with a series of bounding boxes in [x1, y1, x2, y2] format.\n"
                        "For each whole second that may related to the question, output a series of bounding boxes of the object in JSON format." 
                        f"You only need to output {len(boxes)} bbox(es). You need to determine which frame is related to the question, and you don't need to output the bbox for the frames not related to the question."
                        "The keys should be the whole seconds (as strings), and the values should be the bounding box in [x1, y1, x2, y2] format.\n"
                        f"Example output: <answer>{{\"0\": [x1, y1, x2, y2], \"1\":..., ..., \"{len(boxes)}\":...}}</answer> (if the frames at 0~{len(boxes)} second are related to the questions)"
                        "Do not output reasoning process. Do not request more observations. Output only the final JSON inside <answer>...</answer>.\n"
                    ),
                },
                SYSTEM_PROMPT_DIRECT_ANSWER_OE,
                "oe",
            ),
            (
                "answer_temporal_2",
                {
                    "prompt_text": (
                        f"This video is {video_length} seconds long.\n"
                        f"Answer the question about the video: {temporal_question}\n"
                        f"There are {len(boxes)} bounding boxes of the key object related to the question without knowing the time: {boxes}.\n"
                        "Directly output the start and end moment timestamps.\n"
                        "Output only <answer>From <t>start_time</t>s to <t>end_time</t>s</answer>."
                    ),
                    "direct_prompt": (
                        "\n"
                        f"This video is {video_length} seconds long.\n"
                        f"Answer the question about the video: {temporal_question}\n"
                        f"There are {len(boxes)} bounding boxes of the key object related to the question without knowing the time: {boxes}.\n"
                        "Do not think out loud. Do not request more observations. "
                        "Output only <answer>From <t>start_time</t>s to <t>end_time</t>s</answer>.\n"
                    ),
                },
                SYSTEM_PROMPT_DIRECT_ANSWER_TG,
                "tg",
            ),
        ]

        tasks: List[Dict[str, Any]] = []
        for task_name, task_prompt, system_prompt, question_type in task_specs:
            think_answer_prompt = (
                task_prompt["prompt_text"]
                + "\nFirst reason inside <think> and </think>, then put the final answer inside <answer> and </answer>."
            )
            message, frame_meta = build_video_o3_custom_message(
                args,
                video_path=video_path,
                prompt_text=task_prompt["prompt_text"],
                dataset_name=self.dataset_name.value if hasattr(self.dataset_name, "value") else str(self.dataset_name),
                question_type=question_type,
                direct_prompt=task_prompt["direct_prompt"],
                think_answer_prompt=think_answer_prompt,
            )
            tasks.append(
                {
                    "task_name": task_name,
                    "message": message,
                    "frame_meta": frame_meta,
                    "use_tools": True,
                    "max_turn": task_max_turn,
                    "system_prompt": system_prompt,
                }
            )
        return tasks

    def merge_video_o3_task_results(
        self,
        item: Dict[str, Any],
        task_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        by_name = {task["task_name"]: task for task in task_results}
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)

        answer_vqa_raw = by_name.get("answer_vqa", {}).get("response")
        answer_temporal_raw = by_name.get("answer_temporal", {}).get("response")
        answer_spatial_raw = by_name.get("answer_spatial", {}).get("response")
        answer_spatial_2_raw = by_name.get("answer_spatial_2", {}).get("response")
        answer_temporal_2_raw = by_name.get("answer_temporal_2", {}).get("response")

        return {
            "answer_vqa_raw_output": answer_vqa_raw,
            "answer_vqa": _extract_answer_text(answer_vqa_raw),
            "answer_temporal_pre": answer_temporal_raw,
            "answer_temporal": _extract_timestamps(answer_temporal_raw),
            "answer_spatial_pre": answer_spatial_raw,
            "answer_spatial": _extract_bounding_boxes(answer_spatial_raw, width, height),
            "answer_spatial_pre_2": answer_spatial_2_raw,
            "answer_spatial_2": _extract_bounding_boxes(answer_spatial_2_raw, width, height),
            "answer_temporal_pre_2": answer_temporal_2_raw,
            "answer_temporal_2": _extract_timestamps(answer_temporal_2_raw),
            "dialogue_turns_by_task": {
                task["task_name"]: task.get("dialogue_turns") or []
                for task in task_results
            },
            "video_o3": {
                task["task_name"]: {
                    "turn_count": task.get("turn_count"),
                    "inference_time": task.get("inference_time"),
                    "inference_time_generate": task.get("inference_time_generate"),
                    "raw_response": task.get("raw_response"),
                    "frame_meta": task.get("frame_meta"),
                    "dialogue_turns": task.get("dialogue_turns") or [],
                }
                for task in task_results
            },
        }

    def _result_json_path(self, rows: List[Dict[str, Any]]) -> Tuple[str, bool]:
        result_path = getattr(self.args, "result_path", None)
        if result_path:
            if result_path.endswith(".jsonl"):
                return result_path[:-6] + ".vstar_eval.json", False
            return result_path + ".vstar_eval.json", False
        fd, temp_path = tempfile.mkstemp(prefix="v_star_eval_", suffix=".json")
        os.close(fd)
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        return temp_path, True

    def _write_result_json(self, rows: List[Dict[str, Any]]) -> Tuple[str, bool]:
        result_json_path, is_temp = self._result_json_path(rows)
        if not is_temp:
            result_dir = os.path.dirname(result_json_path)
            if result_dir:
                os.makedirs(result_dir, exist_ok=True)
            with open(result_json_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
        return result_json_path, is_temp

    def _run_external_eval(self, result_json_path: str) -> str:
        script_path = os.environ.get("OPENO3_EVAL_VSTAR_EVAL_SCRIPT", DEFAULT_VSTAR_EVAL_SCRIPT)
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"[Open-o3 Eval] V-STAR eval script not found: {script_path}")
        python_exe = os.environ.get("OPENO3_EVAL_VSTAR_EVAL_PYTHON", sys.executable)
        cmd = [python_exe, script_path, "--result_file", result_json_path]
        logger.info(f"[Open-o3 Eval] Running V-STAR external eval: {' '.join(cmd)}")
        env = os.environ.copy()
        eval_cuda_visible_devices = os.environ.get(
            "OPENO3_EVAL_VSTAR_EVAL_CUDA_VISIBLE_DEVICES",
            DEFAULT_VSTAR_EVAL_CUDA_VISIBLE_DEVICES,
        ).strip()
        env["CUDA_VISIBLE_DEVICES"] = eval_cuda_visible_devices
        completed = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if completed.returncode != 0:
            raise RuntimeError(
                "[Open-o3 Eval] V-STAR external eval failed:\n"
                f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
            )
        if completed.stderr.strip():
            logger.warning(f"[Open-o3 Eval] V-STAR external eval stderr:\n{completed.stderr}")
        logger.info(f"[Open-o3 Eval] V-STAR external eval stdout:\n{completed.stdout}")
        return completed.stdout

    def _parse_external_eval_stdout(self, stdout: str) -> Dict[str, Any]:
        metrics: Dict[str, Dict[str, Any]] = {}
        current_section: Optional[str] = None
        current_chain: Optional[str] = None

        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.endswith(":") and (
                line == "Overall Statistics:"
                or line.startswith("Video Length: ")
                or line.startswith("Domain: ")
            ):
                current_section = line[:-1]
                metrics.setdefault(current_section, {})
                current_chain = None
                continue
            if current_section is None:
                continue
            if line in {"Chain 1:", "Chain 2:"}:
                current_chain = "chain1" if line == "Chain 1:" else "chain2"
                continue
            if line.startswith("VQA: "):
                values = _float_metrics(line[len("VQA: ") :])
                metrics[current_section].update(
                    {
                        "vqa_avg_all_score": values.get("Avg All Score"),
                        "vqa_avg_valid_score": values.get("Avg Valid Score"),
                        "vqa_accuracy": values.get("Accuracy"),
                    }
                )
                continue
            if line.startswith("Temporal Answer: "):
                values = _float_metrics(line[len("Temporal Answer: ") :])
                prefix = f"{current_chain}_temporal" if current_chain else "temporal"
                metrics[current_section].update(
                    {
                        f"{prefix}_r1_iou_0.3": values.get("R1@IoU=0.3"),
                        f"{prefix}_r1_iou_0.5": values.get("R1@IoU=0.5"),
                        f"{prefix}_r1_iou_0.7": values.get("R1@IoU=0.7"),
                        f"{prefix}_miou": values.get("Mean IoU"),
                    }
                )
                continue
            if line.startswith("Spatial Answer: "):
                values = _float_metrics(line[len("Spatial Answer: ") :])
                prefix = f"{current_chain}_spatial" if current_chain else "spatial"
                metrics[current_section].update(
                    {
                        f"{prefix}_map_0.1": values.get("mAP@0.1"),
                        f"{prefix}_map_0.3": values.get("mAP@0.3"),
                        f"{prefix}_map_0.5": values.get("mAP@0.5"),
                        f"{prefix}_map_0.7": values.get("mAP@0.7"),
                        f"{prefix}_map_0.9": values.get("mAP@0.9"),
                        f"{prefix}_miou": values.get("Mean mIoU"),
                    }
                )
                continue
            if line.startswith("AM1:"):
                values = _float_metrics(line)
                metrics[current_section].update(
                    {
                        "am1": values.get("AM1"),
                        "am2": values.get("AM2"),
                        "mam": values.get("mAM"),
                    }
                )
                continue
            if line.startswith("LGM1:"):
                values = _float_metrics(line)
                metrics[current_section].update(
                    {
                        "lgm1": values.get("LGM1"),
                        "lgm2": values.get("LGM2"),
                        "mlgm": values.get("mLGM"),
                    }
                )
                continue
            if line.startswith("VQA & Temp:"):
                values = _float_metrics(line)
                metrics[current_section].update(
                    {
                        "vqa_temp_chain1": values.get("Chain 1"),
                        "vqa_temp_chain2": values.get("Chain 2"),
                    }
                )
                continue
            if line.startswith("VQA & Spat:"):
                values = _float_metrics(line)
                metrics[current_section].update(
                    {
                        "vqa_spat_chain1": values.get("Chain 1"),
                        "vqa_spat_chain2": values.get("Chain 2"),
                    }
                )
                continue
            if line.startswith("Temp & Spat:"):
                values = _float_metrics(line)
                metrics[current_section].update(
                    {
                        "temp_spat_chain1": values.get("Chain 1"),
                        "temp_spat_chain2": values.get("Chain 2"),
                    }
                )
                continue
            if line.startswith("VQA & Temp & Spat:"):
                values = _float_metrics(line)
                metrics[current_section].update(
                    {
                        "vqa_temp_spat_chain1": values.get("Chain 1"),
                        "vqa_temp_spat_chain2": values.get("Chain 2"),
                    }
                )
                continue
        return metrics

    def calc_metric(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        result_json_path, is_temp = self._write_result_json(data_list)
        try:
            stdout = self._run_external_eval(result_json_path)
            metrics = self._parse_external_eval_stdout(stdout)
            if not metrics:
                raise ValueError("[Open-o3 Eval] Failed to parse V-STAR metrics from external eval stdout.")
            return metrics
        finally:
            if is_temp and os.path.exists(result_json_path):
                os.remove(result_json_path)

    def build_sample_eval(self, item: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        gold = str((result.get("pp_meta") or {}).get("gt_answer") or result.get("gt_answer") or "").strip()
        pred = str(result.get("answer_vqa") or "").strip()
        vqa_exact_match = float(bool(pred and gold and pred.lower() == gold.lower()))
        return {
            "sample_eval": {
                "metric_type": "multi_task_video_reasoning",
                "vqa_pred": pred,
                "vqa_gold": gold,
                "vqa_exact_match": vqa_exact_match,
                "answer_temporal": result.get("answer_temporal"),
                "answer_temporal_2": result.get("answer_temporal_2"),
                "answer_spatial": result.get("answer_spatial"),
                "answer_spatial_2": result.get("answer_spatial_2"),
            }
        }
