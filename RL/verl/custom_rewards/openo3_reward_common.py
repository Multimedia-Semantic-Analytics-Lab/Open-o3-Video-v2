from __future__ import annotations

import ast
import json
import re
from typing import Any

import numpy as np

try:
    from rouge_score import rouge_scorer
except ModuleNotFoundError:  # pragma: no cover - depends on training env
    rouge_scorer = None


OPENO3_SOURCES = {
    "visual QA",
    "temporal QA",
    "temporal QA (MCQ)",
    "General video QA MCQ",
    "General video QA Free-form",
    "temporal-spatial free-form QA",
}
VIDEOO3_TOOL_SOURCES = {
    "tool_reward_without_round_penalty",
    "tool_reward_with_round_penalty",
}

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TIME_TAG_RE = re.compile(r"<t>\s*(.*?)\s*</t>\s*s?", re.DOTALL)
TG_PATTERN = re.compile(
    r"(?:From\s+)?<t>\s*([+-]?\d+(?:\.\d+)?)\s*</t>\s*s\s+to\s+"
    r"<t>\s*([+-]?\d+(?:\.\d+)?)\s*</t>\s*s"
)
TG_ANSWER_FORMAT_RE = re.compile(r"^From\s+<t>\d+(?:\.\d+)?</t>s\s+to\s+<t>\d+(?:\.\d+)?</t>s$")
OBJ_BOX_TIME_RE = re.compile(
    r"<obj>\s*(.*?)\s*</obj>\s*((?:<box>\s*\[.*?\]\s*</box>\s*)+)at\s*<t>\s*(.*?)\s*</t>\s*s?",
    re.DOTALL,
)
NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
QWEN35_FUNCTION_RE = re.compile(r"<function=\s*([^\s<>]+)\s*(.*?)\s*</function>", re.DOTALL)
QWEN35_PARAMETER_RE = re.compile(r"<parameter=(.*?)>(.*?)</parameter>", re.DOTALL)
MAJOR_TOKEN_RE = re.compile(
    "|".join(
        [
            r"(?P<think_open><think>)",
            r"(?P<think_close></think>)",
            r"(?P<tool_call_open><tool_call>)",
            r"(?P<tool_call_close></tool_call>)",
            r"(?P<answer_open><answer>)",
            r"(?P<answer_close></answer>)",
        ]
    )
)


def join_trajectory(solution: Any) -> str:
    if isinstance(solution, list):
        return "\n".join(item for item in solution if isinstance(item, str))
    if isinstance(solution, str):
        return solution
    return ""


def with_implicit_initial_think(text: str) -> str:
    match = MAJOR_TOKEN_RE.search(text)
    if match is not None and match.lastgroup == "think_close":
        return "<think>" + text
    return text


def last_turn(solution: Any) -> str:
    if isinstance(solution, list):
        for item in reversed(solution):
            if isinstance(item, str):
                return item
        return ""
    return solution if isinstance(solution, str) else ""


def extract_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""
    matches = ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else ""


def extract_all_think(solution: Any) -> str:
    turns = solution if isinstance(solution, list) else [solution]
    chunks: list[str] = []
    for turn in turns:
        if not isinstance(turn, str):
            continue
        chunks.extend(THINK_RE.findall(turn))
    return "\n".join(chunks)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_temporal_segment(start: Any, end: Any) -> list[float] | None:
    try:
        start_f, end_f = float(start), float(end)
    except (TypeError, ValueError):
        return None
    if end_f < start_f:
        return None
    return [start_f, end_f]


def parse_temporal_segment(value: Any) -> list[float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return normalize_temporal_segment(value[0], value[1])

    text = str(value).strip()
    if not text:
        return None
    answer_text = extract_answer(text) if ANSWER_RE.search(text) else text
    match = TG_PATTERN.search(answer_text)
    if match:
        return normalize_temporal_segment(match.group(1), match.group(2))

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
        return normalize_temporal_segment(parsed[0], parsed[1])
    return None


def get_float(extra_info: dict[str, Any] | None, key: str, default: float) -> float:
    try:
        return float((extra_info or {}).get(key, default))
    except (TypeError, ValueError):
        return default


def get_weight(extra_info: dict[str, Any] | None, key: str, default: float) -> float:
    aliases = {
        "ans_reward_weight": ("ans_reward_weight", "acc_reward_weight"),
        "format_reward_weight": ("format_reward_weight",),
        "thinking_reward_weight": ("thinking_reward_weight", "think_reward_weight"),
        "tool_reward_weight": ("tool_reward_weight", "use_tool_reward_weight"),
    }
    for alias in aliases.get(key, (key,)):
        if alias not in (extra_info or {}):
            continue
        return get_float(extra_info, alias, default)
    return default


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for token_a in a:
        prev = 0
        for j, token_b in enumerate(b, start=1):
            old = dp[j]
            if token_a == token_b:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = old
    return dp[-1]


def f1(overlap: int, ref_count: int, hyp_count: int) -> float:
    if overlap <= 0 or ref_count <= 0 or hyp_count <= 0:
        return 0.0
    precision = overlap / hyp_count
    recall = overlap / ref_count
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def fallback_rouge_f(reference: str, hypothesis: str) -> float:
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    if not ref or not hyp:
        return 0.0
    rouge1 = f1(len(set(ref) & set(hyp)), len(ref), len(hyp))
    ref_bi = list(zip(ref, ref[1:], strict=False))
    hyp_bi = list(zip(hyp, hyp[1:], strict=False))
    rouge2 = f1(len(set(ref_bi) & set(hyp_bi)), len(ref_bi), len(hyp_bi)) if ref_bi and hyp_bi else 0.0
    rouge_l = f1(lcs_len(ref, hyp), len(ref), len(hyp))
    return (rouge1 + rouge2 + rouge_l) / 3


def rouge_f(reference: str, hypothesis: str) -> float:
    if rouge_scorer is None:
        return fallback_rouge_f(reference, hypothesis)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return (scores["rouge1"].fmeasure + scores["rouge2"].fmeasure + scores["rougeL"].fmeasure) / 3


def mcq_match(prediction: str, ground_truth: str) -> float:
    gt = str(ground_truth).strip()
    pred = str(prediction).strip()
    candidates = {gt, gt + ".", f"({gt})", f"[{gt}]"}
    if pred in candidates:
        return 1.0
    match = re.match(r"^\(?([A-Z])\)?[.:]?", pred)
    return 1.0 if match and match.group(1) == gt else 0.0


def calculate_iou(box_a, box_b) -> float:
    try:
        a = np.asarray(box_a, dtype=float)
        b = np.asarray(box_b, dtype=float)
        if a.shape[0] != 4 or b.shape[0] != 4:
            return 0.0
    except Exception:
        return 0.0
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def resize_box_to_qwen3(bbox, image_size):
    if image_size is None:
        return list(bbox)
    width, height = float(image_size[0]), float(image_size[1])
    if width <= 0 or height <= 0:
        return list(bbox)
    x0, y0, x1, y1 = bbox
    return [x0 * 1000 / width, y0 * 1000 / height, x1 * 1000 / width, y1 * 1000 / height]


def resize_normalized_box_to_qwen3(bbox) -> list[float]:
    x0, y0, x1, y1 = bbox
    return [float(x0) * 1000, float(y0) * 1000, float(x1) * 1000, float(y1) * 1000]


def parse_single_timestamp_value(text: Any) -> float | None:
    if not isinstance(text, str):
        text = str(text)
    stripped = text.strip()
    if not stripped or NUMERIC_RE.fullmatch(stripped) is None:
        return None
    try:
        return float(stripped)
    except (TypeError, ValueError):
        return None


def parse_think_timestamps(think_content: str) -> list[float]:
    values: list[float] = []
    for content in TIME_TAG_RE.findall(think_content):
        value = parse_single_timestamp_value(content)
        if value is not None:
            values.append(value)
    return values


def normalize_object_name(text: str) -> str:
    lowered = str(text).strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def parse_temporal_spatial_claims(think_content: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for idx, match in enumerate(OBJ_BOX_TIME_RE.finditer(think_content or "")):
        object_name = match.group(1).strip()
        timestamp = parse_single_timestamp_value(match.group(3))
        if timestamp is None:
            continue
        try:
            box_strs = re.findall(r"\[.*?\]", match.group(2))
            bboxes = [json.loads(raw) for raw in box_strs]
        except (json.JSONDecodeError, ValueError):
            continue
        claims.append(
            {
                "id": idx,
                "object_name": object_name,
                "timestamp": float(timestamp),
                "bboxes": bboxes,
            }
        )
    return claims


def parse_qwen35_tool_payload(raw: str) -> dict[str, Any] | None:
    match = QWEN35_FUNCTION_RE.search(raw.strip())
    if not match:
        return None
    arguments: dict[str, Any] = {}
    for key, raw_value in QWEN35_PARAMETER_RE.findall(match.group(2).strip()):
        value = raw_value.strip()
        try:
            parsed_value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            parsed_value = value
        arguments[key.strip()] = parsed_value
    return {"name": match.group(1).strip(), "arguments": arguments}


def parse_tool_call_payload(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        payload = parse_qwen35_tool_payload(raw)
    return payload if isinstance(payload, dict) else None


def parse_gt_intervals(ground_truth: Any) -> list[list[float]]:
    intervals: list[list[float]] = []

    def push(segment: Any) -> None:
        if not isinstance(segment, list) or len(segment) != 2:
            return
        try:
            start, end = float(segment[0]), float(segment[1])
        except (TypeError, ValueError):
            return
        if start == end or start > end:
            return
        intervals.append([min(start, end), max(start, end)])

    if isinstance(ground_truth, dict):
        for item in ground_truth.get("clue", []) or []:
            if isinstance(item, dict) and "timestamp" in item:
                push(item["timestamp"])
            elif isinstance(item, list):
                push(item)
    elif isinstance(ground_truth, list):
        for item in ground_truth:
            if isinstance(item, dict) and "timestamp" in item:
                push(item["timestamp"])
            elif isinstance(item, list):
                push(item)
    return intervals
