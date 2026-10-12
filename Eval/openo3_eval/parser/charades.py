import json
import re
from typing import Any, List

from openo3_eval.parser.answer_parsing import AnswerParsing
from openo3_eval.registry import ANSWER_PARSER_REGISTRY, AnswerParsers
from openo3_eval.util import strip_video_o3_metadata


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if "```" not in stripped:
        return stripped
    first = stripped.find("```")
    last = stripped.rfind("```")
    if first == -1 or last == -1 or last <= first:
        return stripped
    fenced = stripped[first + 3 : last].strip()
    if fenced.lower().startswith("json"):
        fenced = fenced[4:].strip()
    return fenced


def _coerce_interval_list(value: Any) -> List[List[float]] | None:
    if isinstance(value, dict):
        if "result" in value:
            return _coerce_interval_list(value["result"])
        if "temporal_segment" in value:
            return _coerce_interval_list(value["temporal_segment"])
        return None

    if not isinstance(value, list):
        return None
    if len(value) == 2 and all(isinstance(item, (int, float, str)) for item in value):
        try:
            start, end = float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
        return [[start, end]] if end >= start else None

    intervals: List[List[float]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            return None
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            return None
        if end < start:
            return None
        intervals.append([start, end])
    return intervals or None


def _scale_normalized_intervals(intervals: List[List[float]], duration: float | None) -> List[List[float]]:
    if duration is None or duration <= 0:
        return intervals
    if all(start <= 1 and end <= 1 for start, end in intervals):
        return [[start * duration, end * duration] for start, end in intervals]
    return intervals


@ANSWER_PARSER_REGISTRY.register(name=AnswerParsers.charades_parser)
class CharadesAnswerParser:
    def __call__(self, text: str, duration: float | None = None, **_: Any) -> List[List[float]] | None:
        cleaned, _ = strip_video_o3_metadata(str(text or ""))
        parsed = AnswerParsing(cleaned).strip_think_blocks().extract_answer_block()
        text_to_parse = parsed.text.strip() if parsed.text.strip() else cleaned.strip()
        if not text_to_parse:
            return None

        text_to_parse = _strip_code_fence(text_to_parse)
        json_candidates = [text_to_parse]
        start_idx = text_to_parse.find("{")
        end_idx = text_to_parse.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_candidates.insert(0, text_to_parse[start_idx : end_idx + 1])
        start_idx = text_to_parse.find("[")
        end_idx = text_to_parse.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_candidates.append(text_to_parse[start_idx : end_idx + 1])

        for candidate in json_candidates:
            try:
                parsed_json = json.loads(candidate)
            except Exception:
                continue
            intervals = _coerce_interval_list(parsed_json)
            if intervals is not None:
                return _scale_normalized_intervals(intervals, duration)

        numbers = re.findall(r"-?\d+\.?\d*", text_to_parse)
        if len(numbers) < 2:
            return None
        if len(numbers) % 2:
            numbers = numbers[:-1]
        intervals = []
        for idx in range(0, len(numbers), 2):
            start, end = float(numbers[idx]), float(numbers[idx + 1])
            if end < start:
                return None
            intervals.append([start, end])
        return _scale_normalized_intervals(intervals, duration)
