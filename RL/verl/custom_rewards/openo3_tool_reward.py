from __future__ import annotations

from typing import Any

from custom_rewards.openo3_reward_common import (
    TOOL_CALL_RE,
    clamp01,
    get_float,
    join_trajectory,
    parse_gt_intervals,
    parse_tool_call_payload,
    with_implicit_initial_think,
)


def _extract_grounding_intervals(solution: Any) -> list[list[float]]:
    text = join_trajectory(solution)
    if not text:
        return []
    text = with_implicit_initial_think(text)
    intervals: list[list[float]] = []
    for match in TOOL_CALL_RE.finditer(text):
        prefix = text[: match.start()]
        if not (prefix.count("<think>") > 0 and prefix.count("<think>") == prefix.count("</think>")):
            continue
        payload = parse_tool_call_payload(match.group(1))
        if payload is None:
            continue
        if not isinstance(payload, dict) or payload.get("name") != "crop_video":
            continue
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            continue
        if arguments.get("sampling_strategy") not in {"coarse", "medium", "fine"}:
            continue
        try:
            start_time = float(arguments["start_time"])
            end_time = float(arguments["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if end_time <= start_time:
            continue
        intervals.append([min(start_time, end_time), max(start_time, end_time)])
    return intervals


def _tool_call_count(solution: Any) -> int:
    text = join_trajectory(solution)
    return len(TOOL_CALL_RE.findall(text)) if text else 0


def _temporal_iou(seg_a: list[float], seg_b: list[float]) -> float:
    start_a, end_a = float(seg_a[0]), float(seg_a[1])
    start_b, end_b = float(seg_b[0]), float(seg_b[1])
    inter = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    return float(inter / union) if union > 0 else 0.0


def _merge_intervals(intervals: list[list[float]]) -> list[list[float]]:
    cleaned: list[list[float]] = []
    for interval in intervals:
        if not isinstance(interval, list) or len(interval) != 2:
            continue
        try:
            start, end = float(interval[0]), float(interval[1])
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        cleaned.append([start, end])
    if not cleaned:
        return []
    cleaned.sort(key=lambda item: (item[0], item[1]))
    merged: list[list[float]] = [cleaned[0]]
    for start, end in cleaned[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return merged


def _interval_total_length(intervals: list[list[float]]) -> float:
    return sum(max(0.0, float(end) - float(start)) for start, end in intervals)


def _interval_intersection_length(intervals_a: list[list[float]], intervals_b: list[list[float]]) -> float:
    i, j = 0, 0
    total = 0.0
    while i < len(intervals_a) and j < len(intervals_b):
        start_a, end_a = intervals_a[i]
        start_b, end_b = intervals_b[j]
        total += max(0.0, min(end_a, end_b) - max(start_a, start_b))
        if end_a < end_b:
            i += 1
        else:
            j += 1
    return total


def _multi_segment_iou(pred_intervals: list[list[float]], gt_intervals: list[list[float]]) -> float:
    pred_merged = _merge_intervals(pred_intervals)
    gt_merged = _merge_intervals(gt_intervals)
    if not pred_merged or not gt_merged:
        return 0.0
    inter = _interval_intersection_length(pred_merged, gt_merged)
    union = _interval_total_length(pred_merged) + _interval_total_length(gt_merged) - inter
    return clamp01(float(inter / union) if union > 0 else 0.0)


def _prefix_utility_scores(pred_intervals: list[list[float]], ground_truth: Any) -> list[float]:
    gt_intervals = parse_gt_intervals(ground_truth)
    if not pred_intervals or not gt_intervals:
        return []
    return [_multi_segment_iou(pred_intervals[:idx], gt_intervals) for idx in range(1, len(pred_intervals) + 1)]


def _overlap_pair_count(pred_intervals: list[list[float]], threshold: float = 0.5) -> int:
    if len(pred_intervals) <= 1:
        return 0
    overlap_count = 0
    for i, interval_a in enumerate(pred_intervals):
        for interval_b in pred_intervals[i + 1 :]:
            if _temporal_iou(interval_a, interval_b) > threshold:
                overlap_count += 1
    return overlap_count


def _has_clue_intervals(ground_truth: Any) -> bool:
    return bool(parse_gt_intervals(ground_truth))


def _tool_call_excess_penalty(call_count: int, extra_info: dict[str, Any] | None) -> float:
    penalty_unit = get_float(extra_info, "tool_call_penalty", 0.1)
    free_calls = int(round(get_float(extra_info, "tool_free_call_limit", 3.0)))
    excess_calls = max(0, int(call_count) - max(0, free_calls))
    return max(0.0, penalty_unit * excess_calls)


def _tool_repeat_penalty(pred_intervals: list[list[float]], extra_info: dict[str, Any] | None) -> float:
    penalty_unit = get_float(extra_info, "tool_call_penalty", 0.1)
    repeat_threshold = get_float(extra_info, "tool_repeat_iou_threshold", 0.7)
    overlap_pairs = _overlap_pair_count(pred_intervals, threshold=repeat_threshold)
    return max(0.0, penalty_unit * overlap_pairs)


def _tool_fragmentation_penalty(pred_intervals: list[list[float]], extra_info: dict[str, Any] | None) -> float:
    penalty_unit = get_float(extra_info, "tool_fragmentation_penalty", 0.1)
    min_run_length = max(3, int(round(get_float(extra_info, "tool_fragmentation_min_run", 3.0))))
    adjacency_gap_tolerance = max(0.0, get_float(extra_info, "tool_fragmentation_gap_tolerance", 1e-6))
    overlap_tolerance = max(0.0, get_float(extra_info, "tool_fragmentation_overlap_tolerance", 1e-6))
    if len(pred_intervals) < min_run_length:
        return 0.0

    def _is_fragmented_pair(left: list[float], right: list[float]) -> bool:
        if len(left) != 2 or len(right) != 2:
            return False
        left_start, left_end = float(left[0]), float(left[1])
        right_start, right_end = float(right[0]), float(right[1])
        if left_end <= left_start or right_end <= right_start:
            return False
        gap = right_start - left_end
        if gap < -overlap_tolerance:
            return False
        return gap <= adjacency_gap_tolerance

    fragmented_run_count = 0
    run_length = 1
    for previous, current in zip(pred_intervals, pred_intervals[1:]):
        if _is_fragmented_pair(previous, current):
            run_length += 1
            continue
        if run_length >= min_run_length:
            fragmented_run_count += 1
        run_length = 1
    if run_length >= min_run_length:
        fragmented_run_count += 1
    return max(0.0, penalty_unit * fragmented_run_count)


def _tool_gain_bonus(pred_intervals: list[list[float]], ground_truth: Any, extra_info: dict[str, Any] | None) -> float:
    gain_bonus_unit = get_float(extra_info, "tool_gain_bonus", 0.1)
    improve_eps = get_float(extra_info, "tool_gain_min_delta", 0.1)
    prefix_scores = _prefix_utility_scores(pred_intervals, ground_truth)
    if len(prefix_scores) <= 1:
        return 0.0
    previous_score = prefix_scores[0]
    useful_extra_calls = 0
    for score in prefix_scores[1:]:
        if score - previous_score > improve_eps:
            useful_extra_calls += 1
        previous_score = score
    return max(0.0, gain_bonus_unit * useful_extra_calls)


def _tool_regression_penalty(
    pred_intervals: list[list[float]],
    ground_truth: Any,
    extra_info: dict[str, Any] | None,
) -> float:
    regression_drop_threshold = get_float(extra_info, "tool_regression_drop_threshold", 0.1)
    regression_penalty_unit = get_float(extra_info, "tool_regression_penalty", 0.05)
    prefix_scores = _prefix_utility_scores(pred_intervals, ground_truth)
    if len(prefix_scores) <= 1:
        return 0.0
    previous_score = prefix_scores[0]
    regression_count = 0
    for score in prefix_scores[1:]:
        if previous_score - score > regression_drop_threshold:
            regression_count += 1
        previous_score = score
    return max(0.0, regression_penalty_unit * regression_count)


def _tool_reward_without_clue(solution: Any, extra_info: dict[str, Any] | None) -> float:
    base_reward = get_float(extra_info, "tool_no_call_reward", 0.3)
    call_count = _tool_call_count(solution)
    pred_intervals = _extract_grounding_intervals(solution)
    penalty = (
        _tool_call_excess_penalty(call_count, extra_info)
        + _tool_repeat_penalty(pred_intervals, extra_info)
        + _tool_fragmentation_penalty(pred_intervals, extra_info)
    )
    return clamp01(base_reward - penalty)


def _tool_reward_with_clue(
    solution: Any,
    ground_truth: Any,
    extra_info: dict[str, Any] | None,
) -> float:
    pred_intervals = _extract_grounding_intervals(solution)
    if not pred_intervals:
        return 0.0
    call_count = _tool_call_count(solution)
    prefix_scores = _prefix_utility_scores(pred_intervals, ground_truth)
    iou_score = prefix_scores[-1] if prefix_scores else 0.0
    gain_bonus = _tool_gain_bonus(pred_intervals, ground_truth, extra_info)
    penalty = (
        _tool_call_excess_penalty(call_count, extra_info)
        + _tool_repeat_penalty(pred_intervals, extra_info)
        + _tool_fragmentation_penalty(pred_intervals, extra_info)
        + _tool_regression_penalty(pred_intervals, ground_truth, extra_info)
    )
    return clamp01(iou_score + gain_bonus - penalty)


def compute_tool_reward(
    solution: Any,
    ground_truth: Any,
    extra_info: dict[str, Any] | None,
    *,
    ans_score: float,
) -> float:
    del ans_score
    if _has_clue_intervals(ground_truth):
        return _tool_reward_with_clue(solution, ground_truth, extra_info)
    return _tool_reward_without_clue(solution, extra_info)
