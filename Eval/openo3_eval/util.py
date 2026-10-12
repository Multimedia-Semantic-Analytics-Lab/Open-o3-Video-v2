import base64
import json
import os
import re
from typing import Any, Dict, List, Tuple

from openo3_eval.runtime import get_logger
from tabulate import tabulate

logger = get_logger()

VIDEO_O3_DIALOGUE_TRACE_TAG = "VIDEO_O3_DIALOGUE_TRACE_B64"


def resolve_prompt(prompt: str) -> str:
    """Return *prompt* itself, or its file contents if *prompt* points to an existing file."""
    if not prompt:
        return prompt
    if os.path.isfile(prompt):
        with open(prompt, "r", encoding="utf-8") as f:
            return f.read()
    return prompt


def make_image_cache_param_key(sample_fps: float, resize: Tuple[int, int] | None, jpeg_quality: int) -> str:
    """Build a cache key encoding all cache-varying params, e.g. ``raw_fps2.0_q95``."""
    r = f"{resize[0]}x{resize[1]}" if resize else "raw"
    return f"{r}_fps{sample_fps}_q{jpeg_quality}"


def print_metric_pretty(metrics: Dict[str, Any], dataset_pp: str) -> None:
    """Pretty-print evaluation metric as an ASCII table.

    Supports two shapes:
    - Nested: {sub_dataset: {metric: value, ...}, ...}  → multi-row table
    - Flat:   {metric: value, ...}                       → single-row table
    """
    logger.info(f"[Open-o3 Eval] Eval metrics:\n{json.dumps(metrics, indent=2, ensure_ascii=False)}")

    if not metrics:
        return

    # Auto-detect: if any value is a dict, treat as nested (sub-dataset → metrics)
    is_nested = any(isinstance(v, dict) for v in metrics.values())

    if not is_nested:
        # Flat metrics: single row, metric names as columns
        headers = list(metrics.keys())
        rows = [list(metrics.values())]
    else:
        # Nested: each key is a sub-dataset row
        all_keys: List[str] = []
        for v in metrics.values():
            if isinstance(v, dict):
                for k in v:
                    if k not in all_keys:
                        all_keys.append(k)
        headers = ["Dataset", *all_keys]
        rows = []
        for ds_name, ds_metrics in metrics.items():
            if isinstance(ds_metrics, dict):
                rows.append([ds_name] + [ds_metrics.get(k, "-") for k in all_keys])
            else:
                rows.append([ds_name] + [ds_metrics] + ["-"] * (len(all_keys) - 1))

    table = tabulate(rows, headers=headers, tablefmt="simple_outline", numalign="right", stralign="left")
    logger.info(f"\n[Open-o3 Eval] Evaluation Metrics — {dataset_pp}\n{table}")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    result_dir = os.path.dirname(path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    result_dir = os.path.dirname(path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rank_result_path(result_path: str, rank: int, world_size: int) -> str:
    if world_size <= 1:
        return result_path
    return f"{result_path}.rank{rank}.jsonl"


def merge_rank_results(result_path: str, world_size: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rank in range(world_size):
        rows.extend(load_jsonl(rank_result_path(result_path, rank, world_size)))
    rows.sort(key=lambda row: int(row.get("_openo3_eval_sample_idx", 0)))
    write_jsonl(result_path, rows)
    return rows


def _encode_video_o3_metadata_payload(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_video_o3_metadata_payload(payload: str) -> Any:
    raw = base64.urlsafe_b64decode(payload.encode("ascii"))
    return json.loads(raw.decode("utf-8"))


def build_video_o3_dialogue_trace_suffix(dialogue_turns: List[Dict[str, Any]]) -> str:
    if not dialogue_turns:
        return ""
    encoded = _encode_video_o3_metadata_payload(dialogue_turns)
    return f"\n[{VIDEO_O3_DIALOGUE_TRACE_TAG}:{encoded}]"


def strip_video_o3_metadata(response: str) -> Tuple[str, Dict[str, Any]]:
    metadata: Dict[str, Any] = {
        "turn_count": None,
        "inference_time": None,
        "inference_time_generate": None,
        "dialogue_turns": [],
    }
    cleaned = response

    match = re.search(r"\[TURN_COUNT:\s*(\d+)\]", cleaned)
    if match:
        metadata["turn_count"] = int(match.group(1))
        cleaned = re.sub(r"\n?\[TURN_COUNT:\s*\d+\]", "", cleaned)

    match = re.search(r"\[INFERENCE_TIME:\s*([\d.]+)s?\]", cleaned)
    if match:
        metadata["inference_time"] = float(match.group(1))
        cleaned = re.sub(r"\n?\[INFERENCE_TIME:\s*[\d.]+s?\]", "", cleaned)

    match = re.search(r"\[INFERENCE_TIME_GENERATE:\s*([\d.]+)s?\]", cleaned)
    if match:
        metadata["inference_time_generate"] = float(match.group(1))
        cleaned = re.sub(r"\n?\[INFERENCE_TIME_GENERATE:\s*[\d.]+s?\]", "", cleaned)

    match = re.search(rf"\[{VIDEO_O3_DIALOGUE_TRACE_TAG}:([A-Za-z0-9_\-=]+)\]", cleaned)
    if match:
        try:
            dialogue_turns = _decode_video_o3_metadata_payload(match.group(1))
            if isinstance(dialogue_turns, list):
                metadata["dialogue_turns"] = dialogue_turns
        except Exception:
            logger.warning("[Open-o3 Eval] Failed to decode Video-O3 dialogue trace metadata.", exc_info=True)
        cleaned = re.sub(rf"\n?\[{VIDEO_O3_DIALOGUE_TRACE_TAG}:[A-Za-z0-9_\-=]+\]", "", cleaned)

    return cleaned.strip(), metadata
