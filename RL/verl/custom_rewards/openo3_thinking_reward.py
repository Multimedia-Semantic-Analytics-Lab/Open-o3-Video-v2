from __future__ import annotations

import json
from typing import Any

import numpy as np

from custom_rewards.openo3_reward_common import (
    VIDEOO3_TOOL_SOURCES,
    calculate_iou,
    clamp01,
    extract_all_think,
    get_float,
    normalize_object_name,
    parse_gt_intervals,
    parse_temporal_segment,
    parse_temporal_spatial_claims,
    parse_think_timestamps,
    resize_box_to_qwen3,
    resize_normalized_box_to_qwen3,
)


def _thinking_text(solution: Any, extra_info: dict[str, Any] | None) -> str:
    del extra_info
    return extract_all_think(solution)


def _thinking_temporal_segment_reward(
    solution: Any,
    ground_truth: Any,
    data_source: str,
    extra_info: dict[str, Any] | None,
) -> float:
    if data_source in ("visual QA", "temporal-spatial free-form QA") or "General video QA" in data_source:
        return 0.0
    think_content = _thinking_text(solution, extra_info)
    pred_times = parse_think_timestamps(think_content)
    gt_intervals = parse_gt_intervals(ground_truth)
    if not gt_intervals and not isinstance(ground_truth, dict):
        gt_raw = ground_truth
        if data_source == "temporal QA (MCQ)":
            try:
                gt_raw = str(ground_truth).split("\n")[1]
            except IndexError:
                gt_raw = None
        parsed_interval = parse_temporal_segment(gt_raw)
        if parsed_interval is not None:
            gt_intervals = [parsed_interval]
    if not pred_times or not gt_intervals:
        return 0.0

    tolerance = 0.1

    def hits(value: float) -> bool:
        return any((start - tolerance) <= value <= (end + tolerance) for start, end in gt_intervals)

    return sum(1.0 for value in pred_times if hits(value)) / len(pred_times)


def _thinking_temporal_point_reward(
    solution: Any,
    data_source: str,
    extra_info: dict[str, Any] | None,
) -> float:
    extra_info = extra_info or {}
    if data_source in ("visual QA", "temporal QA", "temporal QA (MCQ)") or "General video QA" in data_source:
        return 0.0
    think_content = _thinking_text(solution, extra_info)
    pred_times = parse_think_timestamps(think_content)
    key_frames = extra_info.get("key_frames") or []
    gt_times = [frame["time"] for frame in key_frames if isinstance(frame, dict) and "time" in frame]
    if not pred_times or not gt_times:
        return 0.0

    sigma = 1.0
    total = 0.0
    for value in pred_times:
        diff = min(abs(value - gt) for gt in gt_times)
        total += float(np.exp(-(diff**2) / (2 * sigma**2)))
    return total / len(pred_times)


def _thinking_spatial_reward(
    solution: Any,
    ground_truth: Any,
    data_source: str,
    extra_info: dict[str, Any] | None,
) -> float:
    extra_info = extra_info or {}
    think_content = _thinking_text(solution, extra_info)
    if not think_content:
        return 0.0

    if data_source == "visual QA":
        import re

        pattern = re.compile(r"<box>(\[.*?\])</box>")
        gt_match = pattern.search(f"<answer>{ground_truth}</answer>")
        if not gt_match:
            return 0.0
        try:
            gt_box = json.loads(gt_match.group(1))
        except (json.JSONDecodeError, ValueError):
            return 0.0
        gt_box = resize_box_to_qwen3(gt_box, extra_info.get("image_size"))
        pred_boxes = []
        for raw in pattern.findall(think_content):
            try:
                pred_boxes.append(json.loads(raw))
            except (json.JSONDecodeError, ValueError):
                continue
        return max((calculate_iou(gt_box, box) for box in pred_boxes), default=0.0)

    if data_source in ("temporal QA", "temporal QA (MCQ)") or "General video QA" in data_source:
        return 0.0

    claims = parse_temporal_spatial_claims(think_content)
    if not claims:
        return 0.0

    key_objects = None
    if isinstance(ground_truth, dict):
        key_objects = ground_truth.get("key_objects")
    if not isinstance(key_objects, dict):
        key_objects = extra_info.get("key_objects")
    if isinstance(key_objects, dict) and key_objects:
        ko_times: list[tuple[float, str]] = []
        for raw_time in key_objects.keys():
            try:
                ko_times.append((float(raw_time), raw_time))
            except (TypeError, ValueError):
                continue
        if not ko_times:
            return 0.0

        total = 0.0
        for claim in claims:
            pred_t = float(claim["timestamp"])
            closest_key = None
            min_diff = float("inf")
            for t_value, t_key in ko_times:
                diff = abs(t_value - pred_t)
                if diff < 1.0 and diff < min_diff:
                    min_diff = diff
                    closest_key = t_key
            if closest_key is None:
                continue
            objects = key_objects.get(closest_key, {}) or {}
            normalized_claim_name = normalize_object_name(claim["object_name"])
            matched_objects = []
            for object_name, payload in objects.items():
                normalized_gt_name = normalize_object_name(object_name)
                if normalized_claim_name and normalized_gt_name and (
                    normalized_claim_name in normalized_gt_name or normalized_gt_name in normalized_claim_name
                ):
                    matched_objects.append(payload)
            if not matched_objects:
                matched_objects = list(objects.values())

            claim_boxes = claim["bboxes"]
            if not isinstance(claim_boxes, list) or not claim_boxes:
                continue
            if not isinstance(claim_boxes[0], list):
                claim_boxes = [claim_boxes]

            max_iou = 0.0
            for obj in matched_objects:
                if not isinstance(obj, dict):
                    continue
                gt_box = obj.get("box")
                if not isinstance(gt_box, list) or len(gt_box) != 4:
                    continue
                gt_box = resize_normalized_box_to_qwen3(gt_box)
                per_gt = [calculate_iou(gt_box, claim_box) for claim_box in claim_boxes]
                if per_gt:
                    max_iou = max(max_iou, max(per_gt))
            total += max_iou
        return total / len(claims)

    key_items = extra_info.get("key_items") or {}
    key_frames = extra_info.get("key_frames") or []
    gt_times = [frame["time"] for frame in key_frames if isinstance(frame, dict) and "time" in frame]
    if not gt_times:
        return 0.0

    total = 0.0
    for claim in claims:
        pred_t = float(claim["timestamp"])
        closest_time = None
        min_diff = float("inf")
        for gt_time in gt_times:
            diff = abs(gt_time - pred_t)
            if diff < 1.0 and diff < min_diff:
                min_diff = diff
                closest_time = gt_time
        if closest_time is None:
            continue
        key_frame = next((frame for frame in key_frames if frame.get("time") == closest_time), None)
        if key_frame is None:
            continue
        objects = key_items.get(str(key_frame.get("idx")), {}) or {}
        normalized_claim_name = normalize_object_name(claim["object_name"])
        matched_objects = {}
        for object_name, object_boxes in objects.items():
            normalized_gt_name = normalize_object_name(object_name)
            if normalized_claim_name and normalized_gt_name and (
                normalized_claim_name in normalized_gt_name or normalized_gt_name in normalized_claim_name
            ):
                matched_objects[object_name] = object_boxes
        if not matched_objects:
            matched_objects = objects

        claim_boxes = claim["bboxes"]
        if not isinstance(claim_boxes, list) or not claim_boxes:
            continue
        if not isinstance(claim_boxes[0], list):
            claim_boxes = [claim_boxes]

        max_iou = 0.0
        for gt_boxes in matched_objects.values():
            ious = []
            for gt_box in gt_boxes:
                if not isinstance(gt_box, list) or len(gt_box) != 4:
                    continue
                gt_box = resize_normalized_box_to_qwen3(gt_box)
                per_gt = [calculate_iou(gt_box, claim_box) for claim_box in claim_boxes]
                if per_gt:
                    ious.append(max(per_gt))
            if ious:
                max_iou = max(max_iou, sum(ious) / len(ious))
        total += max_iou
    return total / len(claims)


def compute_thinking_reward(
    solution: Any,
    ground_truth: Any,
    data_source: str,
    extra_info: dict[str, Any] | None,
) -> float:
    if data_source == "visual QA":
        return clamp01(_thinking_spatial_reward(solution, ground_truth, data_source, extra_info))
    if data_source in {"temporal QA", "temporal QA (MCQ)"}:
        return clamp01(_thinking_temporal_segment_reward(solution, ground_truth, data_source, extra_info))
    if "General video QA" in str(data_source):
        return clamp01(get_float(extra_info, "general_thinking_reward_constant", 0.3))
    if data_source == "temporal-spatial free-form QA":
        return clamp01(
            (
                _thinking_temporal_point_reward(solution, data_source, extra_info)
                + _thinking_spatial_reward(solution, ground_truth, data_source, extra_info)
            )
            / 2
        )
    if data_source in VIDEOO3_TOOL_SOURCES:
        return clamp01(
            (
                _thinking_temporal_segment_reward(solution, ground_truth, data_source, extra_info)
                + _thinking_spatial_reward(solution, ground_truth, data_source, extra_info)
            )
            / 2
        )
    return 0.0
