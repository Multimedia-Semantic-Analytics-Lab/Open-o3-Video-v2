from __future__ import annotations

import logging
import math
import os
from typing import Any, Optional
from uuid import uuid4

import cv2
from PIL import Image

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


CROP_VIDEO_SCHEMA = OpenAIFunctionToolSchema.model_validate(
    {
        "type": "function",
        "function": {
            "name": "crop_video",
            "description": "Crop a video segment for closer visual inspection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "number",
                        "description": "Start time in seconds within the full video.",
                    },
                    "end_time": {
                        "type": "number",
                        "description": "End time in seconds within the full video.",
                    },
                    "sampling_strategy": {
                        "type": "string",
                        "enum": ["coarse", "medium", "fine"],
                        "description": "Sampling density for the selected segment.",
                    },
                },
                "required": ["start_time", "end_time", "sampling_strategy"],
            },
        },
    }
)


def _resize_max_height(image: Image.Image, max_height: int) -> Image.Image:
    if max_height <= 0 or image.height <= max_height:
        return image
    width = max(1, int(round(image.width * (max_height / image.height))))
    return image.resize((width, max_height), Image.Resampling.BICUBIC)


def _sample_indices(start_frame: int, end_frame: int, max_frames: int) -> list[int]:
    total = max(0, end_frame - start_frame + 1)
    if total <= 0:
        return []
    count = total if max_frames <= 0 else min(total, max_frames)
    if count == 1:
        return [start_frame]
    return [int(round(start_frame + i * (total - 1) / (count - 1))) for i in range(count)]


def _open_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 1.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count > 0 else 0.0
    return cap, fps, frame_count, duration


def _normalise_segment(start_time: Any, end_time: Any, duration: float) -> tuple[float, float]:
    start = float(start_time)
    end = float(end_time)
    if not (math.isfinite(start) and math.isfinite(end)):
        raise ValueError("start_time and end_time must be finite numbers")
    if start > end:
        start, end = end, start
    duration = max(float(duration or 0.0), 0.0)
    if duration <= 0.0:
        raise ValueError("video duration is not positive")
    start = min(max(start, 0.0), duration)
    end = min(max(end, 0.0), duration)
    if end - start < 1.0:
        if end >= duration:
            start = max(0.0, duration - 1.0)
            end = duration
        else:
            end = min(duration, start + 1.0)
            if end - start < 1.0:
                start = max(0.0, end - 1.0)
    if end <= start:
        raise ValueError(f"Invalid time range after normalization: {start} to {end}")
    return start, end


def _read_segment_frames(
    video_path: str,
    start_time: float,
    end_time: float,
    *,
    max_frames: int,
    frame_max_height: int,
) -> tuple[list[Image.Image], list[float], float]:
    cap, fps, frame_count, duration = _open_video(video_path)
    try:
        start, end = _normalise_segment(start_time, end_time, duration)
        start_frame = max(0, int(math.floor(start * fps)))
        end_frame = min(max(frame_count - 1, 0), int(math.ceil(end * fps)))
        indices = _sample_indices(start_frame, end_frame, max_frames=max_frames)
        frames: list[Image.Image] = []
        timestamps: list[float] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            frames.append(_resize_max_height(image, frame_max_height))
            timestamps.append(float(idx / fps))
        if not frames:
            raise RuntimeError(f"No frames decoded for segment {start} to {end}: {video_path}")
        return frames, timestamps, duration
    finally:
        cap.release()


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class CropVideoTool(BaseTool):
    """Legacy-shape crop_video tool for Open-o3 agent-loop training."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None = None):
        super().__init__(config, tool_schema or CROP_VIDEO_SCHEMA)
        self._instance_dict: dict[str, dict[str, Any]] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {}) or {}
        self._instance_dict[instance_id] = {
            "tool_call_count": 0,
            "video_path": create_kwargs.get("video_path"),
            "video_duration": create_kwargs.get("video_duration"),
        }
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        try:
            state = self._instance_dict[instance_id]
            state["tool_call_count"] += 1

            video_path = state.get("video_path")
            if not video_path:
                agent_data = kwargs.get("agent_data")
                extra_info = getattr(agent_data, "extra_fields", {}).get("extra_info", {}) if agent_data else {}
                video_path = extra_info.get("video_path")
            if not video_path:
                return ToolResponse(text="Error: video_path is unavailable for this sample."), 0.0, {"error": "no_video_path"}
            if not os.path.exists(str(video_path)):
                return (
                    ToolResponse(text=f"Error: Video file not found: {video_path}"),
                    0.0,
                    {"error": f"file_not_found:{video_path}"},
                )

            strategy = str(parameters.get("sampling_strategy", "medium")).lower()
            if strategy not in {"coarse", "medium", "fine"}:
                strategy = "medium"
            max_frames_cfg = self.config.get("max_frames", {}) if isinstance(self.config, dict) else {}
            if isinstance(max_frames_cfg, dict):
                max_frames = _as_int(max_frames_cfg.get(strategy), {"coarse": 8, "medium": 16, "fine": 32}[strategy])
            else:
                max_frames = _as_int(max_frames_cfg, {"coarse": 8, "medium": 16, "fine": 32}[strategy])
            frame_max_height = _as_int(self.config.get("frame_max_height", 336), 336)

            frames, timestamps, duration = _read_segment_frames(
                str(video_path),
                parameters.get("start_time"),
                parameters.get("end_time"),
                max_frames=max_frames,
                frame_max_height=frame_max_height,
            )
            text = (
                "The tool executed successfully. "
                f"Observed {len(frames)} frames from {round(timestamps[0], 1)}s to {round(timestamps[-1], 1)}s."
            )
            return (
                ToolResponse(image=frames, text=text),
                0.0,
                {"video_duration": duration, "frames": len(frames), "timestamps": timestamps},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CropVideoTool] failed: %s", exc, exc_info=True)
            return ToolResponse(text=f"Tool execution failed: {exc}"), 0.0, {"error": str(exc)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return float(self._instance_dict.get(instance_id, {}).get("tool_call_count", 0))

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
