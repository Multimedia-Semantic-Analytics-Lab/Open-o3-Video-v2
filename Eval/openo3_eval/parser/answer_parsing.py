import re
from collections.abc import Iterator
from re import Match
from typing import ClassVar, List, Tuple


class AnswerParsing:
    time_token_pattern: ClassVar[str] = r"(?:\d+:\d+(?:\.\d+)?|\d+(?:\.\d+)?)"
    seconds_unit_pattern: ClassVar[str] = r"(?:seconds?|\u79d2)"

    def __init__(self, text: str, is_truncated: bool = False) -> None:
        self.text = text
        self.is_truncated = is_truncated

    def extract_answer_block(self) -> "AnswerParsing":
        answer_matches = re.findall(r"<answer>(.*?)</answer>", self.text, flags=re.IGNORECASE | re.DOTALL)
        if answer_matches:
            self.text = answer_matches[-1].strip()
        else:
            self.text = self.text.strip()
        return self

    def strip_think_blocks(self) -> "AnswerParsing":
        if "<think>" in self.text and "</think>" not in self.text:
            self.text = ""
            self.is_truncated = True
            return self

        cleaned = re.sub(r"<think>.*?</think>", "", self.text, flags=re.DOTALL)
        if "<think>" in cleaned or "</think>" in cleaned:
            self.text = ""
            self.is_truncated = True
            return self

        self.text = cleaned
        return self

    @staticmethod
    def parse_time_token(token: str) -> float:
        token = token.strip().removesuffix("\u79d2").strip()
        if ":" in token:
            parts = token.split(":")
            return float(parts[0]) * 60 + float(parts[1])
        return float(token)

    def parse_time_ranges(
        self,
        *,
        allow_hyphen: bool = True,
        allow_to: bool = True,
        allow_angle_seconds: bool = False,
        allow_t_tag: bool = False,
        max_ranges: int | None = None,
        require_ordered: bool = True,
    ) -> List[Tuple[float, float]]:
        results: List[Tuple[float, float]] = []

        def _append_matches(matches: Iterator[Match[str]]) -> None:
            for match in matches:
                start = self.parse_time_token(match.group(1))
                end = self.parse_time_token(match.group(2))
                if require_ordered and start >= end:
                    continue
                results.append((start, end))
                if max_ranges is not None and len(results) >= max_ranges:
                    break

        if allow_angle_seconds:
            angle_pattern = re.compile(
                rf"<\s*({self.time_token_pattern})\s*{self.seconds_unit_pattern}\s*>\s*-\s*<\s*({self.time_token_pattern})\s*{self.seconds_unit_pattern}\s*>",
                re.IGNORECASE,
            )
            _append_matches(angle_pattern.finditer(self.text))
            if max_ranges is not None and len(results) >= max_ranges:
                return results[:max_ranges]

        if allow_t_tag:
            # Match <t>start</t>s to <t>end</t>s format
            t_tag_pattern = re.compile(
                rf"<t>\s*({self.time_token_pattern})\s*</t>\s*s?\s*to\s*<t>\s*({self.time_token_pattern})\s*</t>\s*s?",
                re.IGNORECASE,
            )
            _append_matches(t_tag_pattern.finditer(self.text))
            if max_ranges is not None and len(results) >= max_ranges:
                return results[:max_ranges]

        separators = []
        if allow_hyphen:
            separators.append(r"-")
        if allow_to:
            separators.append(r"to")
        if not separators:
            return results

        separator_pattern = "|".join(separators)
        range_pattern = re.compile(
            rf"({self.time_token_pattern})(?:\s*{self.seconds_unit_pattern})?\s*(?:{separator_pattern})\s*({self.time_token_pattern})(?:\s*{self.seconds_unit_pattern})?",
            re.IGNORECASE,
        )
        _append_matches(range_pattern.finditer(self.text))

        if max_ranges is not None:
            return results[:max_ranges]
        return results

    def find_single_angle_second(self) -> float | None:
        matches = list(
            re.finditer(
                rf"<\s*({self.time_token_pattern})\s*{self.seconds_unit_pattern}\s*>",
                self.text,
                re.IGNORECASE,
            )
        )
        if not matches:
            return None
        return self.parse_time_token(matches[-1].group(1))
