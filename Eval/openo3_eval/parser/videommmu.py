import re
from typing import Any, Iterable, List

from openo3_eval.parser.answer_parsing import AnswerParsing
from openo3_eval.registry import ANSWER_PARSER_REGISTRY, AnswerParsers

LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _extract_numbers(text: str) -> List[str]:
    patterns = [
        r"-?\b\d{1,3}(?:,\d{3})+\b",
        r"-?\d+(?:\.\d+)?[eE][+-]?\d+",
        r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])",
    ]
    out: List[str] = []
    for pattern in patterns:
        out.extend(re.findall(pattern, text))
    return out


def _normalize_open_token(token: str) -> List[Any]:
    text = str(token or "").strip()
    if not text:
        return []
    try:
        return [round(float(text.replace(",", "")), 2)]
    except Exception:
        lowered = text.lower()
        if len(lowered) == 1:
            return [f" {lowered}", f"{lowered} "]
        return [lowered]


def _parse_open_response(text: str) -> List[Any]:
    cleaned = text.strip().strip(".")
    if not cleaned:
        return []

    indicators = [
        "could be ",
        "so ",
        "is ",
        "thus ",
        "therefore ",
        "final ",
        "answer ",
        "result ",
        "are ",
        "in total ",
        "total ",
        "identify ",
        "recognize ",
        "calculated as ",
        "counted as ",
        "measured as ",
        "observed as ",
        "concluded as ",
        "found to be ",
        "equals ",
        "determined to be ",
        "number of ",
        "value is ",
        "adds up to ",
        "have ",
        "has ",
        "=",
    ]

    key_responses: List[str] = []
    for chunk in re.split(r"\n+", cleaned):
        chunk_text = chunk.strip()
        if not chunk_text:
            continue
        best = ""
        for indicator in indicators:
            if indicator in chunk_text.lower():
                tail = chunk_text.lower().split(indicator)[-1].strip()
                if tail and (not best or len(tail) < len(best)):
                    best = tail
        key_responses.append(best or chunk_text.lower())

    pred_values: List[Any] = []
    for response in key_responses:
        pred_values.extend(_normalize_open_token(response))
        for number in _extract_numbers(response):
            pred_values.extend(_normalize_open_token(number))

    deduped: List[Any] = []
    for value in pred_values:
        if value not in deduped:
            deduped.append(value)
    return deduped


@ANSWER_PARSER_REGISTRY.register(name=AnswerParsers.videommmu_parser)
class VideoMMMUAnswerParser:
    def __call__(self, text: str, question_type: str = "multiple-choice", candidates: Iterable[str] | None = None) -> Any:
        parsed = AnswerParsing(text).strip_think_blocks().extract_answer_block()
        text_to_search = parsed.text.strip() if parsed.text.strip() else str(text or "").strip()
        if not text_to_search:
            return [] if question_type == "open" else ""

        if str(question_type).strip().lower() == "open":
            return _parse_open_response(text_to_search)

        if text_to_search.upper() in LETTERS:
            return text_to_search.upper()

        patterns = [
            r"(?i)(?:best\s+)?answer\s+(?:is\s*)?(?:[:*])?\s*\(?([A-Z])\)?",
            r"(?i)option\s+([A-Z])\b",
            r"(?i)choice\s+([A-Z])\b",
            r"^\s*\(?([A-Z])\)?\s*[\.\:]?$",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text_to_search)
            if matches:
                return matches[-1].upper()

        lines = [line.strip() for line in text_to_search.splitlines() if line.strip()]
        for line in reversed(lines[-3:]):
            matches = re.findall(r"\b([A-Z])\b", line)
            if len(matches) == 1:
                return matches[0].upper()
            match = re.match(r"^\s*\(?([A-Z])\)?\s*[\.\:\)]", line, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        normalized_text = _normalize_text(text_to_search)
        for idx, candidate in enumerate(list(candidates or [])[: len(LETTERS)]):
            candidate_text = _normalize_text(candidate)
            if candidate_text and (
                normalized_text == candidate_text
                or normalized_text in candidate_text
                or candidate_text in normalized_text
            ):
                return LETTERS[idx]

        matches = re.findall(r"\b([A-Z])\b", text_to_search)
        return matches[-1].upper() if matches else ""
