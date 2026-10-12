from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from typing import Any, ClassVar

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import StoppingCriteriaList

from openo3_eval.util import build_video_o3_dialogue_trace_suffix
from openo3_eval.video_o3.tool_format import parse_last_tool_call, rewrite_messages_for_tool_format

from .base_model import BaseModel, KeywordsStoppingCriteria, get_gpu_memory
from .prompts import (
    ERROR_INFO_MULTI_TURN_PROMPT,
    FORCE_FINAL_ANSWER_PROMPT_ERROR_INFO,
    FORCE_FINAL_ANSWER_PROMPT_FRAMES,
    SYSTEM_PROMPT_DIRECT_ANSWER_MC,
    SYSTEM_PROMPT_DIRECT_ANSWER_OE,
    SYSTEM_PROMPT_DIRECT_ANSWER_TG,
    SYSTEM_PROMPT_MULTI_ROUND_MC_V03,
    SYSTEM_PROMPT_MULTI_ROUND_OE,
    SYSTEM_PROMPT_MULTI_ROUND_TG_OPENO3_VERSION,
    SYSTEM_PROMPT_THINK_ANSWER_MC,
    SYSTEM_PROMPT_THINK_ANSWER_OE,
    SYSTEM_PROMPT_THINK_ANSWER_TG,
    TOOL_CALL_CROP_FRAMES_PROMPT,
)

CHAT_TEMPLATE_MULTI_IMAGE = """
{% set image_count = namespace(value=0) %}
{% for message in messages %}
    {% set has_image_placeholder = namespace(value=False) %}
    {% if message['content'] is not string %}
        {% for content in message['content'] %}
            {% if 'text' in content and '<image>' in content['text'] %}
                {% set has_image_placeholder.value = True %}
            {% endif %}
        {% endfor %}
    {% endif %}
    <|im_start|>{{ message['role'] }}
    {% if message['content'] is string %}
        {{ message['content'] | replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>') }}<|im_end|>
    {% else %}
        {% for content in message['content'] %}
            {% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}
                {% if not has_image_placeholder.value %}
                    {% set image_count.value = image_count.value + 1 %}
                    {% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}
                    <|vision_start|><|image_pad|><|vision_end|>
                {% endif %}
            {% elif 'text' in content %}
                {{ content['text'] | replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>') }}
            {% endif %}
        {% endfor %}
        <|im_end|>
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}
"""


def ensure_image_path(image: str) -> str:
    prefixes = ["file://", "http://", "https://", "data:image"]
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return image
    raise ValueError(f"Invalid image: {image}")


def ensure_video_path(video: str) -> str:
    if video.startswith("file://"):
        return video[7:]
    if os.path.exists(video):
        return video
    raise ValueError(f"Invalid video: {video}")


def load_local_or_url_image(image: str) -> Image.Image:
    if image.startswith("file://"):
        image = image[7:]
    with Image.open(image) as pil_image:
        return pil_image.convert("RGB")


def _disable_transformers_flash_attn_generic() -> None:
    def _false(*args, **kwargs):
        return False

    import transformers.utils as transformers_utils
    import transformers.utils.import_utils as import_utils

    transformers_utils.is_flash_attn_2_available = _false
    import_utils.is_flash_attn_2_available = _false

    for module_name in list(sys.modules):
        if module_name == "flash_attn" or module_name.startswith("flash_attn."):
            sys.modules.pop(module_name, None)


def _get_model_type_from_config(model_path: str) -> str | None:
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f).get("model_type")
    except Exception:
        logging.warning("Failed to read model_type from %s", config_path, exc_info=True)
        return None


def _resolve_image_text_model_cls(model_path: str):
    from transformers import AutoModelForImageTextToText

    model_type = _get_model_type_from_config(model_path)
    if model_type == "qwen3_vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration

            return Qwen3VLForConditionalGeneration
        except Exception as err:
            raise RuntimeError(
                f"Model at {model_path} is a Qwen3-VL checkpoint, but the current transformers runtime "
                "does not provide Qwen3VLForConditionalGeneration. Please upgrade transformers to a "
                "version that supports Qwen3-VL."
            ) from err
    if model_type == "qwen3_5":
        try:
            from transformers import Qwen3_5ForConditionalGeneration

            return Qwen3_5ForConditionalGeneration
        except Exception as err:
            raise RuntimeError(
                f"Model at {model_path} is a Qwen3.5-VL checkpoint, but the current transformers runtime "
                "does not provide Qwen3_5ForConditionalGeneration. Please upgrade transformers to a "
                "version that supports Qwen3.5-VL."
            ) from err
    return AutoModelForImageTextToText


def _resolve_torch_dtype(torch_dtype: Any) -> Any:
    if not isinstance(torch_dtype, str):
        return torch_dtype
    normalized = torch_dtype.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    return torch_dtype


def _load_auto_processor(model_path: str):
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except ValueError as err:
        if _get_model_type_from_config(model_path) != "qwen2_5_vl" or "Unrecognized image processor" not in str(err):
            raise

        # transformers 5.x removed the legacy Qwen2_5_VLImageProcessor name used
        # by older checkpoints, while retaining compatible Qwen2-VL processors.
        from transformers import (
            AutoTokenizer,
            Qwen2_5_VLProcessor,
            Qwen2VLImageProcessor,
            Qwen2VLVideoProcessor,
        )

        logging.warning("Loading legacy Qwen2.5-VL processor config from %s with Qwen2-VL processors", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_path)
        video_processor = Qwen2VLVideoProcessor.from_pretrained(model_path)
        return Qwen2_5_VLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=tokenizer.chat_template,
        )


def _load_auto_processor_and_model(
    model_path: str,
    *,
    single_gpu_mode: bool,
    attn_implementation: str | None = None,
    torch_dtype: Any = "auto",
):
    model_cls = _resolve_image_text_model_cls(model_path)
    processor = _load_auto_processor(model_path)
    base_kwargs = {
        "torch_dtype": _resolve_torch_dtype(torch_dtype),
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if not single_gpu_mode:
        base_kwargs["device_map"] = "auto"

    target_device = "cuda:0" if single_gpu_mode else None
    attn_impls = [attn_implementation] if attn_implementation else ["flash_attention_2", "sdpa"]
    for attn_impl in attn_impls:
        try:
            model = model_cls.from_pretrained(
                model_path,
                attn_implementation=attn_impl,
                **base_kwargs,
            )
            if single_gpu_mode:
                torch.cuda.set_device(target_device)
                model.to(target_device)
            logging.info("Loaded %s with attn_implementation=%s", model_path, attn_impl)
            return processor, model, attn_impl
        except Exception:
            error_text = traceback.format_exc()
            if attn_impl == "flash_attention_2" and any(
                token in error_text for token in ("flash_attn", "undefined symbol", "flash_attention")
            ):
                logging.warning(
                    "Failed to load %s with flash_attention_2, retrying with SDPA.\n%s", model_path, error_text
                )
                _disable_transformers_flash_attn_generic()
                continue
            raise
    raise RuntimeError(f"Failed to load model: {model_path}")


class VideoO3MultiImageChat(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True
    allowed_meta_types: ClassVar[list[str]] = [*BaseModel.allowed_meta_types, "direct_prompt", "think_answer_prompt"]

    def _debug_enabled(self):
        return os.environ.get("OPENO3_DEBUG", "0") == "1"

    def _debug(self, message):
        if self._debug_enabled():
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[OPENO3_MULTIIMG_DEBUG {now}] {message}", flush=True)

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens=4096,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        system_prompt: str | None = None,
        post_process: bool = False,
        verbose: bool = False,
        use_tools: bool = True,
        max_turn: int = 5,
        use_dynamic_quota: bool = False,
        do_sample: bool = False,
        fallback_mode: str = "direct_answer",  # "direct_answer" or "think_then_answer"
        **kwargs,
    ):
        super().__init__()
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        self.post_process = post_process
        self.verbose = verbose
        self.use_tools = use_tools
        self.use_dynamic_quota = use_dynamic_quota
        self.do_sample = do_sample
        self.fps = kwargs.pop("fps", 2)
        self.nframe = kwargs.pop("nframe", 128)
        self.max_turn = max_turn
        self.model_path = model_path
        self.requested_attn_implementation = kwargs.pop("attn_implementation", None)
        self.torch_dtype = kwargs.pop("torch_dtype", "auto")
        self.tg_system_prompt = kwargs.pop("tg_system_prompt", None)
        self.single_turn_system_prompt = kwargs.pop("single_turn_system_prompt", None)
        self.tool_format = kwargs.pop("tool_format", None)
        self.frame_prompt_template = kwargs.pop(
            "frame_prompt_template",
            "<{timestamp:.1f} seconds><image>\n",
        )
        self._temp_dirs: list[str] = []
        self.fallback_mode = fallback_mode

        self.generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "return_dict_in_generate": True,
            "do_sample": do_sample,
            "use_cache": True,
        }
        self.system_prompt = system_prompt or SYSTEM_PROMPT_MULTI_ROUND_MC_V03

        explicit_single_gpu_mode = kwargs.pop("single_gpu_mode", None)
        if explicit_single_gpu_mode is None:
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            if visible_devices:
                self.single_gpu_mode = len([x for x in visible_devices.split(",") if x.strip()]) == 1
            else:
                self.single_gpu_mode = torch.cuda.device_count() == 1
        else:
            self.single_gpu_mode = explicit_single_gpu_mode

        print("**************************************************")
        print("multi-image model init parameters:")
        print(f"model_path: {model_path}")
        print(f"use_tools: {use_tools}")
        print(f"use_dynamic_quota: {use_dynamic_quota}")
        print(f"do_sample: {do_sample}")
        print(f"max_turn: {max_turn}")
        print(f"min_pixels: {min_pixels}")
        print(f"max_pixels: {max_pixels}")
        print(f"total_pixels: {total_pixels}")
        print(f"max_new_tokens: {max_new_tokens}")
        print(f"top_p: {top_p}")
        print(f"top_k: {top_k}")
        print(f"temperature: {temperature}")
        print(f"repetition_penalty: {repetition_penalty}")
        print(f"torch_dtype: {self.torch_dtype}")
        print(f"single_gpu_mode: {self.single_gpu_mode}")
        print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
        print("**************************************************")

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems else -1
        assert max_gpu_mem > 0
        assert not kwargs.get("use_lmdeploy", False), (
            "VideoO3MultiImageChat currently only supports transformers inference."
        )
        if self.single_gpu_mode and not torch.cuda.is_available():
            raise RuntimeError("single_gpu_mode=True but CUDA is unavailable.")

        self.processor, self.model, self.attn_implementation = _load_auto_processor_and_model(
            model_path,
            single_gpu_mode=self.single_gpu_mode,
            attn_implementation=self.requested_attn_implementation,
            torch_dtype=self.torch_dtype,
        )
        self.model.eval()
        try:
            self.model.generation_config.use_cache = True
        except Exception:
            pass
        torch.cuda.empty_cache()

    def _get_generation_device(self) -> torch.device:
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            return next(self.model.parameters()).device

    def _move_inputs_to_model_device(self, inputs):
        target_device = self._get_generation_device()
        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(target_device)
        return inputs

    def _prepare_content(self, inputs: list[dict[str, Any]], dataset: str | None = None) -> list[dict[str, Any]]:
        content = []
        for item_in in inputs:
            if item_in["type"] in {"raw_video", "question_type", "direct_prompt", "think_answer_prompt"}:
                continue
            if item_in["type"] == "image":
                item = {"type": "image", "image": ensure_image_path(item_in["value"])}
                if self.min_pixels is not None:
                    item["min_pixels"] = self.min_pixels
                if self.max_pixels is not None:
                    item["max_pixels"] = self.max_pixels
                if self.total_pixels is not None:
                    item["total_pixels"] = self.total_pixels
                for key in ["min_pixels", "max_pixels", "total_pixels", "resized_height", "resized_width"]:
                    if key in item_in and item_in[key] is not None:
                        item[key] = item_in[key]
            elif item_in["type"] == "text":
                item = {"type": "text", "text": item_in["value"]}
            else:
                raise ValueError(f"Invalid message type: {item_in['type']}, {item_in}")
            content.append(item)
        return content

    def _build_frame_timestamp_items(
        self,
        frame_paths: list[str],
        timestamps: list[float],
    ) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for frame_index, (frame_path, timestamp) in enumerate(zip(frame_paths, timestamps, strict=True), start=1):
            items.append(
                {
                    "type": "text",
                    "value": self.frame_prompt_template.format(frame_index=frame_index, timestamp=timestamp),
                }
            )
            items.append({"type": "image", "value": frame_path})
        return items

    def extract_tool_call_info(self, response):
        parsed_tool_call = parse_last_tool_call(response)
        if parsed_tool_call is None:
            raise ValueError(f"No tool call content found in response: {response}")
        function_name, arguments = parsed_tool_call
        if function_name != "crop_video":
            raise ValueError(f"Unsupported tool call name: {function_name}")
        if not isinstance(arguments, dict):
            raise ValueError(f"Invalid tool call arguments: {arguments}")
        if "start_time" not in arguments or "end_time" not in arguments or "sampling_strategy" not in arguments:
            raise ValueError(f"Missing required fields in tool call arguments: {arguments}")
        start_timestamp = float(arguments["start_time"])
        end_timestamp = float(arguments["end_time"])
        sampling_strategy = str(arguments["sampling_strategy"])
        return start_timestamp, end_timestamp, sampling_strategy

    def _messages_for_generation(self, messages):
        if self.tool_format is None:
            return messages
        return rewrite_messages_for_tool_format(copy.deepcopy(messages), tool_format=self.tool_format)

    def _extract_raw_video_question_type_and_direct_prompt(self, message):
        raw_video = None
        question_type = None
        direct_prompt = None
        think_answer_prompt = None
        cleaned_message = []
        for item in message:
            if item.get("type") == "raw_video" and raw_video is None:
                raw_video = item["value"]
                continue
            if item.get("type") == "question_type" and question_type is None:
                question_type = item["value"]
                continue
            if item.get("type") == "direct_prompt" and direct_prompt is None:
                direct_prompt = item["value"]
                continue
            if item.get("type") == "think_answer_prompt" and think_answer_prompt is None:
                think_answer_prompt = item["value"]
                continue
            cleaned_message.append(item)
        return raw_video, question_type, direct_prompt, think_answer_prompt, cleaned_message

    def _image_processor_pixel_kwargs(self, image_items: list[dict[str, Any]]) -> dict[str, int]:
        if not image_items:
            return {}

        item_min_pixels = [int(item["min_pixels"]) for item in image_items if item.get("min_pixels") is not None]
        min_pixels = min(item_min_pixels) if item_min_pixels else self.min_pixels

        item_max_pixels = [int(item["max_pixels"]) for item in image_items if item.get("max_pixels") is not None]
        configured_max_pixels = min(item_max_pixels) if item_max_pixels else self.max_pixels

        item_total_pixels = [int(item["total_pixels"]) for item in image_items if item.get("total_pixels") is not None]
        total_pixels = min(item_total_pixels) if item_total_pixels else self.total_pixels

        max_pixels = configured_max_pixels
        if self.use_dynamic_quota and total_pixels is not None:
            frame_factor = 2
            nframes = int(math.ceil(len(image_items) / frame_factor) * frame_factor)
            dynamic_max_pixels = int(total_pixels / max(nframes, 1) * frame_factor)
            if configured_max_pixels is not None:
                dynamic_max_pixels = min(configured_max_pixels, dynamic_max_pixels)
            if min_pixels is not None:
                dynamic_max_pixels = max(dynamic_max_pixels, int(min_pixels * 1.05))
            max_pixels = dynamic_max_pixels

        if max_pixels is not None and min_pixels is not None:
            min_pixels = min(min_pixels, max_pixels)

        processor_kwargs: dict[str, int] = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = int(min_pixels)
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = int(max_pixels)
        return processor_kwargs

    def _collect_images_and_processor_kwargs(self, messages):
        images = []
        image_items = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                continue
            for item in content:
                if item.get("type") == "image" or "image" in item:
                    images.append(load_local_or_url_image(item["image"]))
                    image_items.append(item)
        return images, self._image_processor_pixel_kwargs(image_items)

    def _trim_stopped_response(self, response: str, stop_tags: list[str]) -> str:
        stop_pos = None
        for stop_tag in stop_tags:
            pos = response.find(stop_tag)
            if pos != -1:
                end_pos = pos + len(stop_tag)
                stop_pos = end_pos if stop_pos is None else min(stop_pos, end_pos)
        if stop_pos is None:
            return response
        return response[:stop_pos]

    def _post_process_response(self, response: str) -> str:
        if not self.post_process:
            return response
        resp = response.split("\\boxed{")[-1]
        counter, end = 1, None
        for idx, char in enumerate(resp):
            if char == "{":
                counter += 1
            elif char == "}":
                counter -= 1
            if counter == 0:
                end = idx
                break
            if idx == len(resp) - 1:
                end = len(resp)
        return resp[:end] if end is not None else response

    def _run_model_once(self, messages, stop_tags: list[str]):
        messages = self._messages_for_generation(messages)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=CHAT_TEMPLATE_MULTI_IMAGE,
        )
        images, image_processor_kwargs = self._collect_images_and_processor_kwargs(messages)
        inputs = self.processor(
            text=text,
            images=images if images else None,
            padding=True,
            return_tensors="pt",
            **image_processor_kwargs,
        )
        inputs = self._move_inputs_to_model_device(inputs)
        stopping_criteria = StoppingCriteriaList(
            [
                KeywordsStoppingCriteria(
                    keywords=stop_tags,
                    tokenizer=self.processor.tokenizer,
                    input_ids=inputs.input_ids,
                )
            ]
        )
        generate_kwargs = dict(self.generate_kwargs)
        generate_kwargs["stopping_criteria"] = stopping_criteria

        start_time = time.time()
        outputs = self.model.generate(**inputs, **generate_kwargs)
        generate_time = time.time() - start_time
        generated_ids = outputs.sequences if hasattr(outputs, "sequences") else outputs
        generated_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        response = self.processor.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        response = self._trim_stopped_response(response, stop_tags)
        response = self._post_process_response(response)
        return response, text, generate_time

    def _extract_answer_content(self, response: str) -> str | None:
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if not match:
            return None
        return match.group(1).strip()

    def _sampling_strategy_to_num_frames(self, sampling_strategy: str) -> int:
        mapping = {
            "coarse": 8,
            "medium": 16,
            "fine": 24,
        }
        return mapping.get(sampling_strategy, 16)

    def _register_temp_dir(self, temp_dir: str) -> None:
        self._temp_dirs.append(temp_dir)

    def _cleanup_temp_dirs(self) -> None:
        for temp_dir in self._temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)
        self._temp_dirs.clear()

    def _sample_video_segment_as_frames(
        self,
        video_path: str,
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        sampling_strategy: str = "medium",
    ):
        local_video_path = ensure_video_path(video_path)
        cap = cv2.VideoCapture(local_video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {local_video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or total_frame_count <= 0:
            cap.release()
            raise ValueError(
                f"Invalid video metadata for {local_video_path}: fps={fps}, frame_count={total_frame_count}"
            )

        duration = total_frame_count / fps
        clip_start = 0.0 if start_time is None else max(0.0, min(float(start_time), duration))
        clip_end = duration if end_time is None else max(clip_start, min(float(end_time), duration))
        if clip_end <= clip_start:
            cap.release()
            raise ValueError(f"Invalid segment range: [{clip_start}, {clip_end}] for duration={duration}")

        start_frame = min(total_frame_count - 1, max(0, math.floor(clip_start * fps)))
        end_frame = min(total_frame_count - 1, max(start_frame, math.ceil(clip_end * fps) - 1))
        num_target_frames = self._sampling_strategy_to_num_frames(sampling_strategy)
        frame_indices = np.linspace(start_frame, end_frame, num=num_target_frames)
        frame_indices = np.unique(np.clip(np.round(frame_indices).astype(int), start_frame, end_frame))
        if frame_indices.size == 0:
            frame_indices = np.array([start_frame], dtype=int)

        temp_dir = tempfile.mkdtemp(prefix="video_o3_multiimg_")
        self._register_temp_dir(temp_dir)
        frame_paths = []
        timestamps = []
        for frame_no, frame_idx in enumerate(frame_indices.tolist(), start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if not success:
                continue
            frame_path = os.path.join(temp_dir, f"frame_{frame_no:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            timestamps.append(frame_idx / fps)
        cap.release()

        if not frame_paths:
            raise ValueError(f"Failed to decode sampled frames from {local_video_path}")

        sampled_duration = max(clip_end - clip_start, 1e-6)
        sample_fps = len(frame_paths) / sampled_duration
        return (
            frame_paths,
            timestamps,
            {
                "sample_fps": sample_fps,
                "sample_n_frames": len(frame_paths),
                "start_time": clip_start,
                "end_time": clip_end,
            },
        )

    def _build_grounded_user_content(self, frame_paths, timestamps, prompt_text):
        content = []
        content.extend(self._prepare_content([{"type": "text", "value": prompt_text}]))
        content.extend(self._prepare_content(self._build_frame_timestamp_items(frame_paths, timestamps)))
        return content

    def _build_direct_answer_fallback_content(self, cleaned_message, direct_prompt):
        original_frame_message = []
        pending_frame_text = None
        pending_aux_image_text = None
        for item in cleaned_message:
            if item.get("type") == "text":
                text = item.get("value", "")
                if re.match(r"^(<[\d.]+ seconds><image>|Frame \d+ at [\d.]+s: <image>)\s*$", text):
                    pending_frame_text = item
                    pending_aux_image_text = None
                elif "<image>" in text:
                    pending_aux_image_text = item
                    pending_frame_text = None
                else:
                    pending_frame_text = None
                    pending_aux_image_text = None
                continue

            if item.get("type") == "image" and pending_frame_text is not None:
                original_frame_message.extend([pending_frame_text, item])
                pending_frame_text = None
                continue

            if item.get("type") == "image" and pending_aux_image_text is not None:
                original_frame_message.extend([pending_aux_image_text, item])
                pending_aux_image_text = None

        fallback_content = self._prepare_content(original_frame_message)
        if direct_prompt is not None:
            fallback_content.append({"type": "text", "text": direct_prompt})
        return fallback_content

    def _fallback_system_prompt_for_question_type(self, question_type):
        if self.single_turn_system_prompt is not None:
            return self.single_turn_system_prompt
        if self.fallback_mode == "direct_answer":
            if question_type == "oe":
                return SYSTEM_PROMPT_DIRECT_ANSWER_OE
            elif question_type == "tg":
                return SYSTEM_PROMPT_DIRECT_ANSWER_TG
            return SYSTEM_PROMPT_DIRECT_ANSWER_MC
        if self.fallback_mode == "think_then_answer":
            if question_type == "oe":
                return SYSTEM_PROMPT_THINK_ANSWER_OE
            elif question_type == "tg":
                return SYSTEM_PROMPT_THINK_ANSWER_TG
            return SYSTEM_PROMPT_THINK_ANSWER_MC
        raise ValueError(f"Invalid fallback_mode: {self.fallback_mode}")

    def _run_direct_answer_fallback(self, fallback_content, question_type):
        direct_system_prompt = self._fallback_system_prompt_for_question_type(question_type)
        fallback_messages = [
            {"role": "system", "content": direct_system_prompt},
            {"role": "user", "content": copy.deepcopy(fallback_content)},
        ]
        fallback_response, fallback_prompt, fallback_generate_time = self._run_model_once(
            fallback_messages, ["</answer>"]
        )
        answer = self._extract_answer_content(fallback_response)
        self._debug(f"Fallback messages: {fallback_messages}")
        self._debug(f"Fallback direct-answer response: {fallback_response}")
        return {
            "answer": answer if answer is not None else fallback_response.strip(),
            "prompt": fallback_prompt,
            "response": fallback_response,
            "generate_time": fallback_generate_time,
        }

    def _uses_spatial_system_prompt(self) -> bool:
        return "visual grounding" in (self.system_prompt or "")

    def _system_prompt_for_question_type(self, question_type: str) -> str | None:
        if question_type == "oe":
            return SYSTEM_PROMPT_MULTI_ROUND_OE
        if question_type == "tg":
            tg_system_prompt = getattr(self, "tg_system_prompt", None)
            if tg_system_prompt is not None:
                return tg_system_prompt
            return SYSTEM_PROMPT_MULTI_ROUND_TG_OPENO3_VERSION
        return self.system_prompt

    def generate_inner_transformers_tools(self, message, dataset=None):
        self._temp_dirs = []
        raw_video, question_type, direct_prompt, think_answer_prompt, cleaned_message = (
            self._extract_raw_video_question_type_and_direct_prompt(message)
        )
        if raw_video is None:
            raise ValueError("raw_video is required for VideoO3MultiImageChat tools mode.")
        question_type = question_type or "mc"

        system_prompt = self._system_prompt_for_question_type(question_type)
        initial_user_content = self._prepare_content(cleaned_message, dataset=dataset)
        if self.fallback_mode == "think_then_answer":
            fallback_user_prompt = think_answer_prompt if think_answer_prompt is not None else direct_prompt
        else:
            fallback_user_prompt = direct_prompt
        direct_answer_fallback_content = self._build_direct_answer_fallback_content(
            cleaned_message, fallback_user_prompt
        )
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": initial_user_content})

        turn_count = 0
        total_generate_time = 0.0
        inference_start_time = time.time()
        dialogue_turns = []

        try:
            while True:
                torch.cuda.empty_cache()
                force_answer = turn_count >= self.max_turn
                if force_answer:
                    fallback_result = self._run_direct_answer_fallback(direct_answer_fallback_content, question_type)
                    total_generate_time += float(fallback_result["generate_time"])
                    dialogue_turns.append(
                        {
                            "turn_index": turn_count + 1,
                            "kind": "fallback",
                            "prompt": fallback_result["prompt"],
                            "response": fallback_result["response"],
                            "generate_time": round(float(fallback_result["generate_time"]), 4),
                        }
                    )
                    total_inference_time = time.time() - inference_start_time
                    return (
                        f"{fallback_result['answer']}\n[TURN_COUNT: {turn_count + 1}]\n"
                        f"[INFERENCE_TIME: {total_inference_time:.2f}s]\n"
                        f"[INFERENCE_TIME_GENERATE: {total_generate_time:.2f}s]"
                        f"{build_video_o3_dialogue_trace_suffix(dialogue_turns)}"
                    )

                response, text, generate_time = self._run_model_once(messages, ["</tool_call>", "</answer>"])
                total_generate_time += generate_time
                trace_entry = {
                    "turn_index": turn_count + 1,
                    "kind": "main",
                    "prompt": text,
                    "response": response,
                    "generate_time": round(float(generate_time), 4),
                }

                answer = self._extract_answer_content(response)
                if answer is not None:
                    dialogue_turns.append(trace_entry)
                    total_inference_time = time.time() - inference_start_time
                    return (
                        f"{answer}\n[TURN_COUNT: {turn_count + 1}]\n"
                        f"[INFERENCE_TIME: {total_inference_time:.2f}s]\n"
                        f"[INFERENCE_TIME_GENERATE: {total_generate_time:.2f}s]"
                        f"{build_video_o3_dialogue_trace_suffix(dialogue_turns)}"
                    )

                turn_count += 1
                messages.append({"role": "assistant", "content": response})

                try:
                    start_timestamp, end_timestamp, sampling_strategy = self.extract_tool_call_info(response)
                    frame_paths, timestamps, clip_meta = self._sample_video_segment_as_frames(
                        raw_video,
                        start_time=start_timestamp,
                        end_time=end_timestamp,
                        sampling_strategy=sampling_strategy,
                    )
                    next_prompt = TOOL_CALL_CROP_FRAMES_PROMPT.format(
                        action_turn=turn_count,
                        observation_turn=turn_count,
                        start_time=clip_meta["start_time"],
                        end_time=clip_meta["end_time"],
                        total_frames=clip_meta["sample_n_frames"],
                        sample_fps=clip_meta["sample_fps"],
                    )
                    trace_entry["tool_call"] = {
                        "start_time": start_timestamp,
                        "end_time": end_timestamp,
                        "sampling_strategy": sampling_strategy,
                    }
                    trace_entry["tool_result"] = {
                        "start_time": clip_meta["start_time"],
                        "end_time": clip_meta["end_time"],
                        "sample_n_frames": clip_meta["sample_n_frames"],
                        "sample_fps": round(float(clip_meta["sample_fps"]), 4),
                        "timestamps": [round(float(timestamp), 4) for timestamp in timestamps],
                    }
                    if turn_count >= self.max_turn:
                        next_prompt += FORCE_FINAL_ANSWER_PROMPT_FRAMES

                    new_user_content = self._build_grounded_user_content(frame_paths, timestamps, next_prompt)
                    trace_entry["next_user_prompt"] = next_prompt
                    dialogue_turns.append(trace_entry)
                    messages.append({"role": "user", "content": new_user_content})
                except Exception as err:
                    logging.error("Failed to extract/process tool call information: %s", err)
                    trace_entry["tool_error"] = str(err)
                    if turn_count >= self.max_turn:
                        error_message = FORCE_FINAL_ANSWER_PROMPT_ERROR_INFO.format(error_info=str(err))
                    else:
                        error_message = ERROR_INFO_MULTI_TURN_PROMPT.format(error_info=str(err))
                    trace_entry["next_user_prompt"] = error_message
                    dialogue_turns.append(trace_entry)
                    messages.append({"role": "user", "content": error_message})
        finally:
            self._cleanup_temp_dirs()

    def generate_inner_transformers(self, message, dataset=None):
        initial_content = self._prepare_content(message, dataset=dataset)
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": initial_content})
        response, prompt, generate_time = self._run_model_once(messages, ["</answer>"])
        answer = self._extract_answer_content(response) or response
        dialogue_turns = [
            {
                "turn_index": 1,
                "kind": "single_turn",
                "prompt": prompt,
                "response": response,
                "generate_time": round(float(generate_time), 4),
            }
        ]
        return f"{answer}{build_video_o3_dialogue_trace_suffix(dialogue_turns)}"

    def generate_inner(self, message, dataset=None):
        if self.use_tools:
            return self.generate_inner_transformers_tools(message, dataset=dataset)
        return self.generate_inner_transformers(message, dataset=dataset)
