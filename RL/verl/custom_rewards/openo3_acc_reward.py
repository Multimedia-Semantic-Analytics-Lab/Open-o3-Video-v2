from __future__ import annotations

import json
import re
from typing import Any

from custom_rewards.openo3_reward_common import (
    calculate_iou,
    clamp01,
    extract_answer,
    last_turn,
    mcq_match,
    parse_temporal_segment,
    resize_box_to_qwen3,
    rouge_f,
)


def _ans_viou(solution: Any, ground_truth: Any, extra_info: dict[str, Any]) -> float:
    output_ans = extract_answer(last_turn(solution))
    pattern = re.compile(r"<box>(\[.*?\])</box>")
    gt_match = pattern.search(f"<answer>{ground_truth}</answer>")
    pred_match = pattern.search(output_ans)
    if not gt_match or not pred_match:
        return 0.0
    try:
        gt_box = json.loads(gt_match.group(1))
        pred_box = json.loads(pred_match.group(1))
    except (json.JSONDecodeError, ValueError):
        return 0.0
    gt_box = resize_box_to_qwen3(gt_box, extra_info.get("image_size"))
    return calculate_iou(gt_box, pred_box)


def _ans_tiou(solution: Any, ground_truth: Any, data_source: str) -> float:
    if data_source == "temporal QA":
        gt_raw = ground_truth
    elif data_source == "temporal QA (MCQ)":
        try:
            gt_raw = str(ground_truth).split("\n")[1]
        except IndexError:
            return 0.0
    else:
        return 0.0
    gt_seg = parse_temporal_segment(gt_raw)
    if gt_seg is None:
        return 0.0
    output_ans = extract_answer(last_turn(solution))
    pred_seg = parse_temporal_segment(output_ans)
    if pred_seg is None:
        return 0.0
    start, end = pred_seg
    inter = max(0.0, min(end, gt_seg[1]) - max(start, gt_seg[0]))
    union = max(end, gt_seg[1]) - min(start, gt_seg[0])
    return float(inter / union) if union > 0 else 0.0


def compute_openo3_acc(solution: Any, ground_truth: Any, data_source: str, extra_info: dict[str, Any]) -> float:
    answer = extract_answer(last_turn(solution))
    if data_source == "visual QA":
        return clamp01(_ans_viou(solution, ground_truth, extra_info))
    if data_source == "temporal QA":
        return clamp01(_ans_tiou(solution, ground_truth, data_source))
    if data_source == "temporal QA (MCQ)":
        try:
            gt_letter = str(ground_truth).split("\n")[0].strip()
            choice = answer.split("Correct Option:")[1].strip()
        except (IndexError, AttributeError):
            choice = ""
        return (mcq_match(choice, gt_letter) + _ans_tiou(solution, ground_truth, data_source)) / 2
    if data_source == "General video QA MCQ":
        return mcq_match(answer, str(ground_truth).strip())
    gt_text = extract_answer(f"<answer>{ground_truth}</answer>") or str(ground_truth)
    return clamp01(rouge_f(gt_text, answer))


def compute_videoo3_acc(solution: Any, ground_truth: Any) -> float:
    answer = extract_answer(last_turn(solution))
    gt = ground_truth.get("answer", "") if isinstance(ground_truth, dict) else ground_truth
    return mcq_match(answer, str(gt).strip())
