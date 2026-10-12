from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import random
import socket
import sys
import threading
import time
from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text, build_qwen35_tool_response_text
from verl.utils.profiler import simple_timer


VISION_SPECIAL_BAD_WORDS = ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>")
_DEBUG_WRITE_LOCK = threading.Lock()
_DEBUG_WRITE_ERROR_REPORTED = False
_DEBUG_TRAJECTORY_KEY = "__debug_trajectory__"


def _load_tool_format_module():
    module_name = "openo3_custom_tool_format"
    if module_name in sys.modules:
        return sys.modules[module_name]
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    module_path = os.path.join(repo_root, "verl", "custom_datasets", "tool_format.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load tool format helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_tool_format = _load_tool_format_module()


class OpenO3Qwen35ToolParser:
    """Qwen3.5 parser aligned with the custom prompt tool-format rewrite."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)
        content, parsed_calls = _tool_format.parse_tool_call_blocks(text)
        function_calls = [
            FunctionCall(name=item["name"], arguments=json.dumps(item["arguments"], ensure_ascii=False))
            for item in parsed_calls
        ]
        return content, function_calls


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truncate(text: Any, max_chars: int) -> str:
    text = "" if text is None else str(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max(1, max_chars // 2)
    return text[:half] + "\n...(truncated)...\n" + text[-half:]


def _safe_json_value(value: Any, max_chars: int = 2000) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate(value, max_chars)
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item, max_chars=max_chars) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_json_value(val, max_chars=max_chars) for key, val in value.items()}
    size = getattr(value, "size", None)
    if size is not None and value.__class__.__name__.endswith("Image"):
        return {"type": "image", "size": list(size)}
    return _truncate(repr(value), max_chars)


def _summarize_messages(messages: Any, max_chars: int) -> list[dict[str, Any]]:
    summarized = []
    for message in list(messages or []):
        if not isinstance(message, dict):
            summarized.append({"value": _safe_json_value(message, max_chars=max_chars)})
            continue
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    parts.append({"value": _safe_json_value(part, max_chars=max_chars)})
                    continue
                part_type = part.get("type")
                if part_type == "text":
                    parts.append({"type": "text", "text": _truncate(part.get("text", ""), max_chars)})
                elif part_type == "image" or "image" in part:
                    image = part.get("image")
                    parts.append({"type": "image", "size": list(getattr(image, "size", []) or [])})
                else:
                    parts.append(_safe_json_value(part, max_chars=max_chars))
            content = parts
        else:
            content = _safe_json_value(content, max_chars=max_chars)
        summarized.append({"role": message.get("role"), "content": content})
    return summarized


def _tool_call_to_dict(tool_call: Any) -> dict[str, Any]:
    arguments = getattr(tool_call, "arguments", "")
    try:
        parsed_arguments = json.loads(arguments)
    except (TypeError, json.JSONDecodeError):
        parsed_arguments = arguments
    return {"name": getattr(tool_call, "name", None), "arguments": parsed_arguments}


def _debug_base_dir() -> str:
    explicit_dir = os.getenv("OPENO3_DEBUG_DUMP_DIR") or os.getenv("UNIFIED_DEBUG_DUMP_DIR")
    if explicit_dir:
        return explicit_dir
    log_save_dir = os.getenv("LOG_SAVE_DIR")
    if log_save_dir:
        return os.path.join(log_save_dir, "debug")
    return os.path.join(os.getcwd(), "debug")


def _debug_file_path() -> str:
    host = socket.gethostname().replace("/", "_")
    return os.path.join(_debug_base_dir(), "rollout", f"{host}-{os.getpid()}.jsonl")


@register("openo3_tool_agent")
class OpenO3ToolAgentLoop(ToolAgentLoop):
    """Tool agent loop with legacy Open-o3 stop and multi-image observations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.tool_parser_name == "qwen3_5":
            self.tool_parser = OpenO3Qwen35ToolParser(self.tokenizer)

    async def _encode_qwen35_tool_response(
        self,
        tool_response_text: str,
        images: list[Any] | None,
    ) -> list[int]:
        if images:
            if not getattr(self.processor, "image_processor", None):
                raise ValueError("Qwen3.5 tool image responses require a multimodal processor.")

            return await self.loop.run_in_executor(
                None,
                lambda: self.processor(
                    text=[tool_response_text],
                    images=images,
                    return_tensors="pt",
                )["input_ids"].squeeze(0).tolist(),
            )

        return await self.loop.run_in_executor(
            None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
        )

    async def _handle_pending_state(self, agent_data, sampling_params: dict[str, Any]) -> AgentState:
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=None,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        params = dict(sampling_params)
        params.setdefault("stop", ["</tool_call>"])
        params.setdefault("include_stop_str_in_output", True)
        bad_words = list(params.get("bad_words") or [])
        for word in VISION_SPECIAL_BAD_WORDS:
            if word not in bad_words:
                bad_words.append(word)
        params["bad_words"] = bad_words

        kwargs = dict(kwargs)
        tools_kwargs = dict(kwargs.get("tools_kwargs", {}) or {})
        debug_enabled = self._rollout_debug_enabled()
        sample_rate = _as_float(os.getenv("OPENO3_DEBUG_SAMPLE_RATE"), 1.0)
        debug_record = debug_enabled and random.random() < max(0.0, min(1.0, sample_rate))
        request_id = tools_kwargs.get("__debug_request_id__") or (
            f"{socket.gethostname()}-{os.getpid()}-{time.time_ns()}"
        )
        tools_kwargs["__debug_request_id__"] = request_id
        tools_kwargs["__debug_record__"] = debug_record
        kwargs["tools_kwargs"] = tools_kwargs

        if debug_record:
            extra_info = kwargs.get("extra_info", {}) or {}
            tools_kwargs[_DEBUG_TRAJECTORY_KEY] = {
                "event": "rollout_trajectory",
                "schema_version": 2,
                "request_id": request_id,
                "data_source": kwargs.get("data_source"),
                "index": kwargs.get("index"),
                "uid": kwargs.get("uid"),
                "start_time": time.time(),
                "extra_info": self._sample_extra_info(extra_info),
                "raw_prompt": _summarize_messages(
                    kwargs.get("raw_prompt"),
                    max_chars=self._debug_max_text_chars(),
                ),
                "tool_names": list(self.tools.keys()),
                "sampling_params": _safe_json_value(params),
                "turns": [],
            }

        try:
            output = await super().run(params, **kwargs)
        except Exception as exc:
            if debug_record:
                trajectory = self._debug_trajectory_from_tools_kwargs(tools_kwargs)
                if trajectory is not None:
                    trajectory["status"] = "error"
                    trajectory["end_time"] = time.time()
                    trajectory["error"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    self._write_rollout_debug(trajectory)
            raise

        if debug_record:
            trajectory = self._debug_trajectory_from_tools_kwargs(tools_kwargs)
            if trajectory is not None:
                trajectory["status"] = "ok"
                trajectory["end_time"] = time.time()
                trajectory["summary"] = {
                    "num_turns": output.num_turns,
                    "response_token_count": len(output.response_ids),
                    "response_mask_token_count": len(output.response_mask),
                    "generated_token_count": int(sum(output.response_mask)),
                    "tool_observation_token_count": int(len(output.response_mask) - sum(output.response_mask)),
                    "reward_score": output.reward_score,
                    "tool_rewards": output.extra_fields.get("tool_rewards"),
                    "turn_scores": output.extra_fields.get("turn_scores"),
                    "metrics": output.metrics.model_dump() if hasattr(output.metrics, "model_dump") else output.metrics,
                }
                self._write_rollout_debug(trajectory)
        return output

    async def _handle_generating_state(
        self, agent_data, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        self._log_generation_turn(agent_data, output)

        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            agent_data.messages.append({"role": "assistant", "content": assistant_message})

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        if self.interaction_config_file:
            return AgentState.INTERACTING
        return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data) -> AgentState:
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []

        tasks = []
        tool_call_names = []
        executed_tool_calls = agent_data.tool_calls[: self.max_parallel_calls]
        for tool_call in executed_tool_calls:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        for call_index, (tool_call, response_tuple) in enumerate(zip(executed_tool_calls, responses, strict=False)):
            tool_response, tool_reward, tool_metrics = response_tuple
            images = list(tool_response.image or [])
            timestamps = list((tool_metrics or {}).get("timestamps") or [])

            if images:
                content: list[dict[str, Any]] = []
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text + "\n"})
                for image_index, image in enumerate(images):
                    if image_index < len(timestamps):
                        content.append(
                            {"type": "text", "text": f"<{round(float(timestamps[image_index]), 1)} seconds>"}
                        )
                    content.append({"type": "image"})
                    content.append({"type": "text", "text": "\n"})
                    new_images_this_turn.append(image)
                message = {"role": "tool", "content": content}
            elif tool_response.video:
                raise NotImplementedError("Open-o3 tool observations should return images, not videos.")
            else:
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)
            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)
            self._log_tool_call_result(
                agent_data,
                tool_call=tool_call,
                call_index=call_index,
                tool_response=tool_response,
                tool_reward=tool_reward,
                tool_metrics=tool_metrics,
            )

        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        elif self.tool_parser_name == "qwen3_5":
            tool_response_text = build_qwen35_tool_response_text(add_messages)
            response_ids = await self._encode_qwen35_tool_response(tool_response_text, new_images_this_turn)
        else:
            response_ids = await self.apply_chat_template(
                add_messages,
                images=new_images_this_turn if new_images_this_turn else None,
                videos=None,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            agent_data.image_data.extend(new_images_this_turn)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    def _rollout_debug_enabled(self) -> bool:
        return _as_bool(os.getenv("OPENO3_ROLLOUT_DEBUG"), False) or _as_bool(
            os.getenv("UNIFIED_DEBUG_MODE"), False
        )

    def _debug_max_text_chars(self) -> int:
        try:
            return int(os.getenv("OPENO3_DEBUG_MAX_TEXT_CHARS", "20000"))
        except ValueError:
            return 20000

    def _debug_request_id(self, agent_data) -> str:
        return str(agent_data.tools_kwargs.get("__debug_request_id__") or agent_data.request_id)

    def _should_debug_record(self, agent_data) -> bool:
        return bool(agent_data.tools_kwargs.get("__debug_record__"))

    def _debug_trajectory_from_tools_kwargs(self, tools_kwargs: dict[str, Any]) -> dict[str, Any] | None:
        trajectory = tools_kwargs.get(_DEBUG_TRAJECTORY_KEY)
        return trajectory if isinstance(trajectory, dict) else None

    def _debug_trajectory(self, agent_data) -> dict[str, Any] | None:
        return self._debug_trajectory_from_tools_kwargs(agent_data.tools_kwargs)

    def _sample_extra_info(self, extra_info: dict[str, Any]) -> dict[str, Any]:
        keep_keys = (
            "index",
            "question",
            "options",
            "video_path",
            "image_path",
            "video_duration",
            "frame_timestamps",
            "debug_source_file",
            "debug_source_index",
            "debug_family",
        )
        return {key: _safe_json_value(extra_info.get(key), max_chars=4000) for key in keep_keys if key in extra_info}

    def _write_rollout_debug(self, payload: dict[str, Any]) -> None:
        global _DEBUG_WRITE_ERROR_REPORTED
        payload = {
            "time": time.time(),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            **payload,
        }
        try:
            path = _debug_file_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            line = json.dumps(_safe_json_value(payload, max_chars=self._debug_max_text_chars()), ensure_ascii=False)
            with _DEBUG_WRITE_LOCK:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as exc:  # noqa: BLE001
            if not _DEBUG_WRITE_ERROR_REPORTED:
                _DEBUG_WRITE_ERROR_REPORTED = True
                print(f"[OpenO3ToolAgentLoop] failed to write rollout debug log: {exc}", flush=True)

    def _log_generation_turn(self, agent_data, output) -> None:
        if not self._should_debug_record(agent_data):
            return
        trajectory = self._debug_trajectory(agent_data)
        if trajectory is None:
            return
        response_text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        trajectory.setdefault("turns", []).append(
            {
                "assistant_turn": agent_data.assistant_turns,
                "user_turn": agent_data.user_turns,
                "time": time.time(),
                "response_token_count": len(agent_data.response_ids),
                "response_text": response_text,
                "tool_call_count": len(agent_data.tool_calls),
                "tool_calls": [_tool_call_to_dict(tool_call) for tool_call in agent_data.tool_calls],
                "tool_results": [],
                "finish_reason": getattr(output, "finish_reason", None),
            }
        )

    def _log_tool_call_result(
        self,
        agent_data,
        *,
        tool_call,
        call_index: int,
        tool_response,
        tool_reward,
        tool_metrics,
    ) -> None:
        if not self._should_debug_record(agent_data):
            return
        trajectory = self._debug_trajectory(agent_data)
        if trajectory is None:
            return
        images = list(tool_response.image or [])
        videos = list(tool_response.video or [])
        result = {
            "time": time.time(),
            "assistant_turn": agent_data.assistant_turns,
            "user_turn": agent_data.user_turns,
            "call_index": call_index,
            "tool_call": _tool_call_to_dict(tool_call),
            "tool_response_text": tool_response.text or "",
            "tool_reward": tool_reward,
            "image_count": len(images),
            "image_sizes": [list(getattr(image, "size", []) or []) for image in images],
            "video_count": len(videos),
            "tool_metrics": _safe_json_value(tool_metrics),
            "ok": not (tool_metrics or {}).get("error"),
        }
        turns = trajectory.setdefault("turns", [])
        if turns and turns[-1].get("assistant_turn") == agent_data.assistant_turns:
            turns[-1].setdefault("tool_results", []).append(result)
        else:
            trajectory.setdefault("orphan_tool_results", []).append(result)
