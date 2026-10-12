import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


TOOL_FORMAT_CONFIG_KEY = "tool_format"
TOOL_FORMAT_ENV_VAR = "TOOL_FORMAT"
DEFAULT_TOOL_FORMAT = "default"
QWEN3_5_TOOL_FORMAT = "qwen3_5"
SUPPORTED_TOOL_FORMATS = {DEFAULT_TOOL_FORMAT, QWEN3_5_TOOL_FORMAT}

_GROUNDING_BLOCK_RE = re.compile(r"<grounding>\s*(.*?)\s*</grounding>", re.DOTALL)
_TOOL_CALL_JSON_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_LEGACY_CROP_RE = re.compile(
    r"<tool_call>\s*"
    r"\{\s*\"name\"\s*:\s*\"crop_video\"\s*,\s*\"arguments\"\s*:\s*"
    r"\{\s*\"start_time\"\s*:\s*(?P<start>[^,\n}]+)\s*,\s*"
    r"\"end_time\"\s*:\s*(?P<end>[^,\n}]+)\s*,\s*"
    r"\"sampling_strategy\"\s*:\s*(?P<sampling>[^}\n]+?)\s*\}\s*\}\s*"
    r"</tool_call>",
    re.DOTALL,
)
_TOOL_CALL_QWEN_RE = re.compile(
    r"<tool_call>\s*<function=(?P<name>[^>\n]+)>\s*(?P<body>.*?)\s*</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"<parameter=(?P<name>[^>\n]+)>\s*(?P<value>.*?)\s*</parameter>",
    re.DOTALL,
)
_TOOL_CALL_PLACEHOLDER_RE = re.compile(
    r"<tool_call>\s*(YOUR VIDEO CROP TOOL|YOUR GROUNDING)\s*</tool_call>",
    re.DOTALL,
)
_TOOL_CALL_AND_RE = re.compile(r"<tool_call>\s*and\s*</tool_call>", re.DOTALL)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*.*?\s*</tool_call>", re.DOTALL)


def normalize_tool_format(tool_format: Optional[str]) -> str:
    if not isinstance(tool_format, str):
        return DEFAULT_TOOL_FORMAT
    normalized = tool_format.strip().lower()
    if normalized in SUPPORTED_TOOL_FORMATS:
        return normalized
    return DEFAULT_TOOL_FORMAT


def _resolve_config_path_from_argv() -> Optional[Path]:
    for index, arg in enumerate(sys.argv):
        if arg == "--config" and index + 1 < len(sys.argv):
            return Path(sys.argv[index + 1]).resolve()
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1]).resolve()
    return None


def _resolve_tool_format_from_config() -> Optional[str]:
    if yaml is None:
        return None

    config_path = _resolve_config_path_from_argv()
    if config_path is None or not config_path.is_file():
        return None

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        return None

    return config.get(TOOL_FORMAT_CONFIG_KEY)


def resolve_tool_format(tool_format: Optional[str] = None) -> str:
    if tool_format is not None:
        return normalize_tool_format(tool_format)

    env_value = os.environ.get(TOOL_FORMAT_ENV_VAR)
    if env_value:
        return normalize_tool_format(env_value)

    config_value = _resolve_tool_format_from_config()
    if config_value:
        return normalize_tool_format(config_value)

    return DEFAULT_TOOL_FORMAT


def _dump_parameter_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def build_tool_call_block(
    *,
    function_name: str,
    arguments: dict[str, Any],
    tool_format: Optional[str] = None,
) -> str:
    tool_format = resolve_tool_format(tool_format)
    if tool_format == QWEN3_5_TOOL_FORMAT:
        lines = ["<tool_call>", f"<function={function_name}>"]
        for key, value in arguments.items():
            lines.extend(
                [
                    f"<parameter={key}>",
                    _dump_parameter_value(value),
                    "</parameter>",
                ]
            )
        lines.extend(["</function>", "</tool_call>"])
        return "\n".join(lines)

    payload = {"name": function_name, "arguments": arguments}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"


def build_crop_video_tool_call_example(
    *,
    start_time: Any,
    end_time: Any,
    sampling_strategy: Any,
    tool_format: Optional[str] = None,
) -> str:
    return build_tool_call_block(
        function_name="crop_video",
        arguments={
            "start_time": start_time,
            "end_time": end_time,
            "sampling_strategy": sampling_strategy,
        },
        tool_format=tool_format,
    )


def _parse_scalar_value(raw_value: str) -> Any:
    value = raw_value.strip()
    if value == "":
        return ""
    try:
        return json.loads(value)
    except Exception:
        return value


def _strip_wrapping_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _parse_default_tool_call_payload(payload: str) -> Optional[tuple[str, dict[str, Any]]]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None
    name = data.get("name")
    arguments = data.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return name, arguments


def _parse_qwen_tool_call_payload(block: str) -> Optional[tuple[str, dict[str, Any]]]:
    match = _TOOL_CALL_QWEN_RE.fullmatch(block.strip())
    if match is None:
        return None

    name = match.group("name").strip()
    arguments: dict[str, Any] = {}
    for param_match in _PARAMETER_RE.finditer(match.group("body")):
        arguments[param_match.group("name").strip()] = _parse_scalar_value(param_match.group("value"))
    return name, arguments


def parse_tool_call_block(block: str) -> Optional[tuple[str, dict[str, Any]]]:
    if not isinstance(block, str):
        return None

    stripped_block = block.strip()
    parsed = _parse_qwen_tool_call_payload(stripped_block)
    if parsed is not None:
        return parsed

    legacy_match = _TOOL_CALL_LEGACY_CROP_RE.fullmatch(stripped_block)
    if legacy_match is not None:
        return (
            "crop_video",
            {
                "start_time": _strip_wrapping_quotes(legacy_match.group("start")),
                "end_time": _strip_wrapping_quotes(legacy_match.group("end")),
                "sampling_strategy": _strip_wrapping_quotes(legacy_match.group("sampling")),
            },
        )

    json_match = _TOOL_CALL_JSON_RE.fullmatch(stripped_block)
    if json_match is None:
        return None
    return _parse_default_tool_call_payload(json_match.group(1))


def parse_last_tool_call(content: str) -> Optional[tuple[str, dict[str, Any]]]:
    if not isinstance(content, str):
        return None

    matches = list(_TOOL_CALL_BLOCK_RE.finditer(content))
    if not matches:
        return None
    return parse_tool_call_block(matches[-1].group(0))


def _convert_grounding_payload(payload: str, tool_format: str) -> str:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return f"<tool_call>{payload}</tool_call>"

    if not isinstance(data, dict):
        return f"<tool_call>{payload}</tool_call>"

    segment = data.get("temporal_segment")
    if isinstance(segment, list) and len(segment) == 2:
        return build_tool_call_block(
            function_name="crop_video",
            arguments={
                "start_time": segment[0],
                "end_time": segment[1],
                "sampling_strategy": data.get("sampling_strategy", "medium"),
            },
            tool_format=tool_format,
        )

    if isinstance(data.get("name"), str) and isinstance(data.get("arguments"), dict):
        return build_tool_call_block(
            function_name=data["name"],
            arguments=data["arguments"],
            tool_format=tool_format,
        )

    return f"<tool_call>{json.dumps(data, ensure_ascii=False)}</tool_call>"


def _convert_default_tool_call_blocks(content: str, tool_format: str) -> str:
    def replacer(match: re.Match[str]) -> str:
        parsed = _parse_default_tool_call_payload(match.group(1))
        if parsed is None:
            return match.group(0)
        function_name, arguments = parsed
        return build_tool_call_block(
            function_name=function_name,
            arguments=arguments,
            tool_format=tool_format,
        )

    return _TOOL_CALL_JSON_RE.sub(replacer, content)


def _convert_legacy_crop_tool_call_blocks(content: str, tool_format: str) -> str:
    def replacer(match: re.Match[str]) -> str:
        return build_tool_call_block(
            function_name="crop_video",
            arguments={
                "start_time": _strip_wrapping_quotes(match.group("start")),
                "end_time": _strip_wrapping_quotes(match.group("end")),
                "sampling_strategy": _strip_wrapping_quotes(match.group("sampling")),
            },
            tool_format=tool_format,
        )

    return _TOOL_CALL_LEGACY_CROP_RE.sub(replacer, content)


def _convert_qwen_tool_call_blocks(content: str, tool_format: str) -> str:
    def replacer(match: re.Match[str]) -> str:
        parsed = _parse_qwen_tool_call_payload(match.group(0))
        if parsed is None:
            return match.group(0)
        function_name, arguments = parsed
        return build_tool_call_block(
            function_name=function_name,
            arguments=arguments,
            tool_format=tool_format,
        )

    return _TOOL_CALL_QWEN_RE.sub(replacer, content)


def rewrite_message_content_for_tool_format(
    content: str,
    tool_format: Optional[str] = None,
) -> str:
    if not isinstance(content, str):
        return content

    tool_format = resolve_tool_format(tool_format)
    content = _GROUNDING_BLOCK_RE.sub(
        lambda match: _convert_grounding_payload(match.group(1).strip(), tool_format),
        content,
    )
    content = content.replace("<grounding>", "<tool_call>")
    content = content.replace("</grounding>", "</tool_call>")
    content = content.replace("YOUR GROUNDING", "YOUR VIDEO CROP TOOL")
    content = _TOOL_CALL_PLACEHOLDER_RE.sub(
        build_crop_video_tool_call_example(
            start_time="START_TIMESTAMP",
            end_time="END_TIMESTAMP",
            sampling_strategy="medium",
            tool_format=tool_format,
        ),
        content,
    )
    content = _TOOL_CALL_AND_RE.sub("the same tool-call block format", content)
    content = _convert_legacy_crop_tool_call_blocks(content, tool_format)
    content = _convert_default_tool_call_blocks(content, tool_format)
    content = _convert_qwen_tool_call_blocks(content, tool_format)
    return content


def rewrite_messages_for_tool_format(
    messages: list[dict[str, Any]],
    tool_format: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> list[dict[str, Any]]:
    tool_format = resolve_tool_format(tool_format)
    for msg in messages:
        if msg.get("role") == "system" and system_prompt is not None:
            msg["content"] = system_prompt
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = rewrite_message_content_for_tool_format(content, tool_format)
            continue
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("text"), str):
                    item["text"] = rewrite_message_content_for_tool_format(item["text"], tool_format)
                if item.get("type") == "text" and isinstance(item.get("value"), str):
                    item["value"] = rewrite_message_content_for_tool_format(item["value"], tool_format)
    return messages
