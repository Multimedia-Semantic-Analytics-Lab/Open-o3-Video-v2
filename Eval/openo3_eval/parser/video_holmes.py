import re
from typing import Any, Iterable

from openo3_eval.parser.answer_parsing import AnswerParsing
from openo3_eval.registry import ANSWER_PARSER_REGISTRY, AnswerParsers

LETTERS = ("A", "B", "C", "D", "E", "F")


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


@ANSWER_PARSER_REGISTRY.register(name=AnswerParsers.video_holmes_parser)
class VideoHolmesAnswerParser:
    def __call__(self, text: str, candidates: Iterable[str] | None = None) -> str:
        parsed_text = AnswerParsing(text).strip_think_blocks().extract_answer_block().text.strip()
        text_to_search = parsed_text if parsed_text else str(text).strip()
        if not text_to_search:
            return ""

        if text_to_search.upper() in LETTERS:
            return text_to_search.upper()

        candidates_to_search = [text_to_search]
        answer_matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text_to_search, flags=re.IGNORECASE | re.DOTALL)
        if answer_matches:
            candidates_to_search.insert(0, answer_matches[-1].strip())

        for candidate_text in candidates_to_search:
            patterns = [
                r"^\s*[\(\[\{]?\s*([A-F])\s*[\)\]\}]?\s*[\.\:\)]?\b",
                r"(?i)\b(?:answer|final|option|choice)\b[^A-F]*([A-F])\b",
                r"(?i)(?:best\s+)?answer\s+(?:is\s*)?(?:[:*])?\s*\(?([A-F])\)?",
            ]
            for pattern in patterns:
                matches = re.findall(pattern, candidate_text)
                if matches:
                    return matches[-1].upper()

        normalized_value = _normalize_text(text_to_search)
        for idx, candidate in enumerate(list(candidates or [])[: len(LETTERS)]):
            candidate_text = _normalize_text(candidate)
            if candidate_text and (
                normalized_value == candidate_text
                or normalized_value in candidate_text
                or candidate_text in normalized_value
            ):
                return LETTERS[idx]

        ultimate_matches = re.findall(r"\b([A-F])\b", text_to_search, flags=re.IGNORECASE)
        if ultimate_matches:
            return ultimate_matches[-1].upper()

        return ""
