from __future__ import annotations

import re
from typing import Any

from custom_rewards.openo3_reward_common import (
    TG_ANSWER_FORMAT_RE,
    TOOL_CALL_RE,
    clamp01,
    extract_answer,
    last_turn,
    parse_temporal_segment,
    parse_tool_call_payload,
    with_implicit_initial_think,
)


MAJOR_PAIRS = (("<think>", "</think>"), ("<tool_call>", "</tool_call>"), ("<answer>", "</answer>"))
MINOR_PAIRS = (("<obj>", "</obj>"), ("<t>", "</t>"), ("<box>", "</box>"))
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


def _balanced_pairs(text: str, pairs) -> bool:
    return all(text.count(open_tag) == text.count(close_tag) for open_tag, close_tag in pairs)


def _major_token_sequence(text: str):
    return [(match.lastgroup, match.start(), match.end()) for match in MAJOR_TOKEN_RE.finditer(text)]


def _structure_ok(seq) -> bool:
    names = [name for name, _, _ in seq]
    if len(names) < 4 or names[-4:] != ["think_open", "think_close", "answer_open", "answer_close"]:
        return False
    middle = names[:-4]
    if len(middle) % 4 != 0:
        return False
    expected = ["think_open", "think_close", "tool_call_open", "tool_call_close"]
    return all(middle[i : i + 4] == expected for i in range(0, len(middle), 4))


def _no_misnest(text: str, seq) -> bool:
    i = 0
    while i < len(seq):
        name, _, end = seq[i]
        if not name.endswith("_open") or i + 1 >= len(seq):
            return False
        kind = name[: -len("_open")]
        close_name, close_start, _ = seq[i + 1]
        if close_name != f"{kind}_close":
            return False
        inner = text[end:close_start]
        for other in ("think", "tool_call", "answer"):
            if other != kind and (f"<{other}>" in inner or f"</{other}>" in inner):
                return False
        i += 2
    return True


def _tool_call_payloads_ok(text: str) -> bool:
    for raw in TOOL_CALL_RE.findall(text):
        payload = parse_tool_call_payload(raw)
        if payload is None:
            return False
        if not isinstance(payload, dict) or payload.get("name") != "crop_video":
            return False
        args = payload.get("arguments")
        if not isinstance(args, dict):
            return False
        if args.get("sampling_strategy") not in {"coarse", "medium", "fine"}:
            return False
        if not isinstance(args.get("start_time"), (int, float)) or not isinstance(args.get("end_time"), (int, float)):
            return False
        if float(args["end_time"]) <= float(args["start_time"]):
            return False
    return True


def _temporal_qa_answer_format_ok(solution: Any) -> bool:
    answer = extract_answer(last_turn(solution))
    if not answer or not TG_ANSWER_FORMAT_RE.fullmatch(answer):
        return False
    segment = parse_temporal_segment(answer)
    return segment is not None and segment[1] >= segment[0]


def compute_format_reward(solution: Any, data_source: str | None = None) -> float:
    text = "\n".join(solution) if isinstance(solution, list) else solution
    if not isinstance(text, str) or not text:
        return 0.0
    text = with_implicit_initial_think(text)
    if not _balanced_pairs(text, MAJOR_PAIRS) or not _balanced_pairs(text, MINOR_PAIRS):
        return 0.0
    seq = _major_token_sequence(text)
    if not _structure_ok(seq):
        return 0.0
    if not _no_misnest(text, seq):
        return 0.0
    if not _tool_call_payloads_ok(text):
        return 0.0
    if data_source == "temporal QA" and not _temporal_qa_answer_format_ok(solution):
        return 0.0
    return clamp01(1.0)
