import json
import re
from typing import Tuple

from openo3_eval.parser.answer_parsing import AnswerParsing
from openo3_eval.registry import ANSWER_PARSER_REGISTRY, AnswerParsers

_NUM_PATTERN = r"[-+]?\d+(?:\.\d+)?"
_LOOSE_RESULT_PATTERN = re.compile(
    rf'"result"\s*:\s*\[\s*\[?\s*({_NUM_PATTERN})\s*,\s*({_NUM_PATTERN})'
)
_LOOSE_TEMPORAL_PATTERN = re.compile(
    rf'"temporal_segment"\s*:\s*\[\s*({_NUM_PATTERN})\s*,\s*({_NUM_PATTERN})'
)


def _parse_json_window(text: str) -> Tuple[float, float] | None:
    candidates = [text.strip()]
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidates.insert(0, text[start_idx : end_idx + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict) and "temporal_segment" in obj:
            obj = obj["temporal_segment"]
        elif isinstance(obj, dict) and "result" in obj:
            obj = obj["result"]
            if isinstance(obj, list) and obj and isinstance(obj[0], list):
                obj = obj[0]
        if isinstance(obj, list) and len(obj) == 2:
            start, end = float(obj[0]), float(obj[1])
            if start < end:
                return start, end

    # Loose fallback: model often emits truncated/malformed JSON like
    # '{"result": [14, 20]' or '{"result": [[13, 16]' which json.loads cannot
    # decode. Recover the first [start, end] pair via regex so the metric
    # reflects the actual temporal grounding ability instead of returning
    # the (-100, -100) sentinel for the majority of samples.
    for pattern in (_LOOSE_RESULT_PATTERN, _LOOSE_TEMPORAL_PATTERN):
        m = pattern.search(text)
        if m:
            start, end = float(m.group(1)), float(m.group(2))
            if start < end:
                return start, end
    return None


@ANSWER_PARSER_REGISTRY.register(name=AnswerParsers.timelens_bench_parser)
class TimeLensBenchAnswerParser:
    def __call__(self, text: str) -> Tuple[Tuple[float, float], bool]:
        parsed = AnswerParsing(text).strip_think_blocks().extract_answer_block()

        json_window = _parse_json_window(parsed.text)
        if json_window is not None:
            return json_window, parsed.is_truncated

        timestamps = parsed.parse_time_ranges(allow_hyphen=True, allow_to=True, allow_angle_seconds=True, allow_t_tag=True, max_ranges=3)

        if len(timestamps) == 0:
            # Fallback: extract any two numbers from the text (loose matching)
            numbers = re.findall(r"-?\d+\.?\d*", parsed.text)
            if len(numbers) >= 2:
                start, end = float(numbers[0]), float(numbers[1])
                if start < end:
                    timestamps = [(start, end)]

        if len(timestamps) > 1:
            timestamps = timestamps[:1]
        elif len(timestamps) == 0:
            timestamps = [(-100.0, -100.0)]

        return timestamps[0], parsed.is_truncated
