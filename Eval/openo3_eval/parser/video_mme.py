import re
from typing import Any, Iterable

from openo3_eval.parser.answer_parsing import AnswerParsing
from openo3_eval.registry import ANSWER_PARSER_REGISTRY, AnswerParsers

LETTERS = tuple("ABCDEFGH")


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


@ANSWER_PARSER_REGISTRY.register(name=AnswerParsers.video_mme_parser)
class VideoMMEAnswerParser:
    def __call__(self, text: str, candidates: Iterable[str] | None = None) -> str:
        # Strip <think> blocks and whitespace for clean extraction
        parsed_text = AnswerParsing(text).strip_think_blocks().extract_answer_block().text.strip()

        # If parsing is empty (e.g. model output got truncated inside <think> without closing it)
        # we fallback to using the entire raw text to salvage an answer.
        text_to_search = parsed_text if parsed_text else text.strip()
        if not text_to_search:
            return ""

        # 1. Exact match for a single letter
        if text_to_search.upper() in LETTERS:
            return text_to_search.upper()

        # 2. Strong regex patterns to find the answer (case-insensitive)
        patterns = [
            r"(?i)(?:best\s+)?answer\s+(?:is\s*)?(?:[:*])?\s*\(?([A-H])\)?",
            r"(?i)option\s+([A-H])\b",
            r"(?i)choice\s+([A-H])\b",
            r"(?i)(?:^|\n)\s*(?:the\s+)?(?:best\s+)?answer\s+is\s*([A-H])\b",
            r"^\s*\(?([A-H])\)?\s*[\.\:]?$",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text_to_search)
            if matches:
                return matches[-1].upper()

        # 3. Fallback: Search the very last line (or last few lines) for a single A/B/C/D
        lines = [line.strip() for line in text_to_search.splitlines() if line.strip()]
        if lines:
            # Check the last 3 lines from bottom to top to be safe against trailing blank lines or weird artifacts
            for last_line in reversed(lines[-3:]):
                last_line_matches = re.findall(r"\b([A-H])\b", last_line)
                if len(last_line_matches) == 1:
                    return last_line_matches[0].upper()

                # Or if it starts with the letter like "A. Apples." or "A)"
                match = re.match(r"^\s*\(?([A-H])\)?\s*[\.\:\)]", last_line, re.IGNORECASE)
                if match:
                    return match.group(1).upper()

        # 4. Fallback 2: Content matching
        normalized_value = _normalize_text(text_to_search)
        for idx, candidate in enumerate(list(candidates or [])[: len(LETTERS)]):
            candidate_text = _normalize_text(candidate)
            if candidate_text and (
                normalized_value == candidate_text
                or normalized_value in candidate_text
                or candidate_text in normalized_value
            ):
                return LETTERS[idx]

        # 5. Ultimate fallback: grab the last standalone choice letter in the text.
        ultimate_matches = re.findall(r"\b([A-H])\b", text_to_search)
        if ultimate_matches:
            return ultimate_matches[-1].upper()

        return ""
