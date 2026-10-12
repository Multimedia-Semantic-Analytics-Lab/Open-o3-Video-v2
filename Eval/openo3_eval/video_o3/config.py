import os
from dataclasses import dataclass
from typing import Any, Dict

from openo3_eval.video_o3.prompts import (
    SYSTEM_PROMPT_MULTI_ROUND_MC_V03,
    SYSTEM_PROMPT_MULTI_ROUND_TG_OPENO3_VERSION,
    SYSTEM_PROMPT_OPENO3_SINGLE_TURN,
)


@dataclass(frozen=True)
class VideoO3ModelConfig:
    env_key: str
    model_path: str
    min_pixels: int
    max_pixels: int
    total_pixels: int
    use_tools: bool = True
    use_dynamic_quota: bool = True
    attn_implementation: str | None = None
    system_prompt: str | None = None
    tg_system_prompt: str | None = None
    single_turn_system_prompt: str | None = None
    default_max_turn: int | None = None
    fallback_mode: str = "direct_answer"  # "direct_answer" or "think_then_answer"
    tool_format: str | None = None
    frame_prompt_template: str | None = None
    pixel_overrides: Dict[str, Dict[str, int]] | None = None
    runtime_kwargs: Dict[str, Any] | None = None
    env_fallback_keys: tuple[str, ...] = ()

    def resolve_model_path(self) -> str:
        for env_key in (self.env_key, *self.env_fallback_keys):
            value = os.environ.get(env_key)
            if value:
                return value
        if not self.model_path:
            raise ValueError(f"Set {self.env_key} to a local model checkpoint")
        return self.model_path

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model_path": self.resolve_model_path(),
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "total_pixels": self.total_pixels,
            "use_tools": self.use_tools,
            "use_dynamic_quota": self.use_dynamic_quota,
        }
        if self.attn_implementation is not None:
            kwargs["attn_implementation"] = self.attn_implementation
        if self.system_prompt is not None:
            kwargs["system_prompt"] = self.system_prompt
        if self.tg_system_prompt is not None:
            kwargs["tg_system_prompt"] = self.tg_system_prompt
        if self.single_turn_system_prompt is not None:
            kwargs["single_turn_system_prompt"] = self.single_turn_system_prompt
        if self.fallback_mode is not None:
            kwargs["fallback_mode"] = self.fallback_mode
        if self.tool_format is not None:
            kwargs["tool_format"] = self.tool_format
        if self.frame_prompt_template is not None:
            kwargs["frame_prompt_template"] = self.frame_prompt_template
        if self.runtime_kwargs:
            kwargs.update(self.runtime_kwargs)
        return kwargs


VIDEO_O3_MODEL_CONFIGS: Dict[str, VideoO3ModelConfig] = {
    "openo3video_v2_4b": VideoO3ModelConfig(
        env_key="OPENO3VIDEO_V2_4B_MODEL_PATH",
        model_path="models/Open-o3-Video-v2-4B",
        min_pixels=20 * 32 * 32,
        max_pixels=768 * 32 * 32,
        total_pixels=16384 * 32 * 32,
        use_tools=True,
        use_dynamic_quota=True,
        attn_implementation="sdpa",
        system_prompt=SYSTEM_PROMPT_MULTI_ROUND_MC_V03,
        tg_system_prompt=SYSTEM_PROMPT_MULTI_ROUND_TG_OPENO3_VERSION,
        default_max_turn=5,
    ),
    "openo3videov2_qwen3vl_8b_rl": VideoO3ModelConfig(
        env_key="OPENO3VIDEOV2_QWEN3VL_8B_RL_MODEL_PATH",
        model_path="models/Open-o3-Video-v2-8B",
        min_pixels=20 * 32 * 32,
        max_pixels=768 * 32 * 32,
        total_pixels=16384 * 32 * 32,
        use_tools=True,
        use_dynamic_quota=True,
        attn_implementation="sdpa",
        system_prompt=SYSTEM_PROMPT_MULTI_ROUND_MC_V03,
        tg_system_prompt=SYSTEM_PROMPT_MULTI_ROUND_TG_OPENO3_VERSION,
        default_max_turn=5,
        runtime_kwargs={"torch_dtype": "bfloat16"},
    ),
    "openo3videov2_sft_135k": VideoO3ModelConfig(
        env_key="OPENO3VIDEOV2_SFT_135K_NEWFORMAT_MODEL_PATH",
        model_path="models/Open-o3-Video-v2-4B-SFT",
        min_pixels=20 * 32 * 32,
        max_pixels=768 * 32 * 32,
        total_pixels=16384 * 32 * 32,
        use_tools=True,
        use_dynamic_quota=True,
        attn_implementation="sdpa",
        system_prompt=SYSTEM_PROMPT_MULTI_ROUND_MC_V03,
        tg_system_prompt=SYSTEM_PROMPT_MULTI_ROUND_TG_OPENO3_VERSION,
        default_max_turn=5,
    ),
    "openo3videov2_sft_135k_8b": VideoO3ModelConfig(
        env_key="OPENO3VIDEOV2_SFT_135K_8B_MODEL_PATH",
        model_path="models/Open-o3-Video-v2-8B-SFT",
        min_pixels=20 * 32 * 32,
        max_pixels=768 * 32 * 32,
        total_pixels=16384 * 32 * 32,
        use_tools=True,
        use_dynamic_quota=True,
        attn_implementation="sdpa",
        system_prompt=SYSTEM_PROMPT_MULTI_ROUND_MC_V03,
        tg_system_prompt=SYSTEM_PROMPT_MULTI_ROUND_TG_OPENO3_VERSION,
        default_max_turn=5,
    ),
    "openo3video_qwen25vl_7b": VideoO3ModelConfig(
        env_key="OPENO3VIDEO_QWEN25VL_7B_MODEL_PATH",
        model_path="models/openo3video_qwen25vl_7b",
        min_pixels=4 * 28 * 28,
        max_pixels=512 * 512,
        total_pixels=64 * 512 * 512,
        use_tools=True,
        use_dynamic_quota=True,
        system_prompt=SYSTEM_PROMPT_OPENO3_SINGLE_TURN,
        single_turn_system_prompt=SYSTEM_PROMPT_OPENO3_SINGLE_TURN,
        default_max_turn=0,
        fallback_mode="think_then_answer",
        frame_prompt_template="Frame {frame_index} at {timestamp:.1f}s: <image>\n",
        pixel_overrides={"v_star": {"max_pixels": 2 * 1024 * 1024}},
        runtime_kwargs={"torch_dtype": "bfloat16"},
        env_fallback_keys=("OPENO3VIDEO_QWEN25VL_8B_MODEL_PATH",),
    ),
    "openo3video_qwen25vl_8b": VideoO3ModelConfig(
        env_key="OPENO3VIDEO_QWEN25VL_8B_MODEL_PATH",
        model_path="models/openo3video_qwen25vl_8b",
        min_pixels=4 * 28 * 28,
        max_pixels=512 * 512,
        total_pixels=64 * 512 * 512,
        use_tools=True,
        use_dynamic_quota=True,
        system_prompt=SYSTEM_PROMPT_OPENO3_SINGLE_TURN,
        single_turn_system_prompt=SYSTEM_PROMPT_OPENO3_SINGLE_TURN,
        default_max_turn=0,
        fallback_mode="think_then_answer",
        frame_prompt_template="Frame {frame_index} at {timestamp:.1f}s: <image>\n",
        pixel_overrides={"v_star": {"max_pixels": 2 * 1024 * 1024}},
        runtime_kwargs={"torch_dtype": "bfloat16"},
        env_fallback_keys=("OPENO3VIDEO_QWEN25VL_7B_MODEL_PATH",),
    ),
    "openo3video_v1_sft": VideoO3ModelConfig(
        env_key="OPENO3VIDEO_V1_SFT_MODEL_PATH",
        model_path="models/openo3video_v1_sft",
        min_pixels=4 * 28 * 28,
        max_pixels=512 * 512,
        total_pixels=64 * 512 * 512,
        use_tools=True,
        use_dynamic_quota=True,
        system_prompt=SYSTEM_PROMPT_OPENO3_SINGLE_TURN,
        single_turn_system_prompt=SYSTEM_PROMPT_OPENO3_SINGLE_TURN,
        default_max_turn=0,
        fallback_mode="think_then_answer",
        frame_prompt_template="Frame {frame_index} at {timestamp:.1f}s: <image>\n",
        pixel_overrides={"v_star": {"max_pixels": 2 * 1024 * 1024}},
        runtime_kwargs={"torch_dtype": "bfloat16"},
    ),
    "qwen3vl": VideoO3ModelConfig(
        env_key="QWEN3VL_PATH",
        model_path="models/qwen3vl",
        min_pixels=20 * 32 * 32,
        max_pixels=768 * 32 * 32,
        total_pixels=16384 * 32 * 32,
        use_tools=True,
        use_dynamic_quota=True,
        default_max_turn=0,
    ),
    "qwen3vl-8b": VideoO3ModelConfig(
        env_key="QWEN3VL_8B_PATH",
        model_path="models/qwen3vl-8b",
        min_pixels=20 * 32 * 32,
        max_pixels=768 * 32 * 32,
        total_pixels=16384 * 32 * 32,
        use_tools=True,
        use_dynamic_quota=True,
        default_max_turn=0,
    ),
    "qwen3vl-cot": VideoO3ModelConfig(
        env_key="QWEN3VL_PATH",
        model_path="models/qwen3vl-cot",
        min_pixels=20 * 32 * 32,
        max_pixels=768 * 32 * 32,
        total_pixels=16384 * 32 * 32,
        use_tools=True,
        use_dynamic_quota=True,
        default_max_turn=0,
        fallback_mode="think_then_answer",
    ),
    "qwen3_5-cot": VideoO3ModelConfig(
        env_key="QWEN3_5_PATH",
        model_path="models/qwen3_5-cot",
        min_pixels=20 * 32 * 32,
        max_pixels=768 * 32 * 32,
        total_pixels=16384 * 32 * 32,
        use_tools=True,
        use_dynamic_quota=True,
        default_max_turn=0,
        fallback_mode="think_then_answer",
    ),
    "qwen3.5-35b-a3b": VideoO3ModelConfig(
        env_key="QWEN3_5_35B_A3B_PATH",
        model_path="models/qwen3.5-35b-a3b",
        min_pixels=20 * 32 * 32,
        max_pixels=768 * 32 * 32,
        total_pixels=16384 * 32 * 32,
        use_tools=True,
        use_dynamic_quota=True,
        default_max_turn=0,
        fallback_mode="think_then_answer",
        tool_format="qwen3_5",
        runtime_kwargs={"max_num_seqs": 1},
    ),
    "qwen3.5-35b-a3b-sft-62k": VideoO3ModelConfig(
        env_key="QWEN3_5_35B_A3B_62K_SFT_PATH",
        model_path="models/qwen3.5-35b-a3b-sft-62k",
        min_pixels=20 * 32 * 32,
        max_pixels=768 * 32 * 32,
        total_pixels=16384 * 32 * 32,
        use_tools=True,
        use_dynamic_quota=True,
        system_prompt=SYSTEM_PROMPT_MULTI_ROUND_MC_V03,
        tg_system_prompt=SYSTEM_PROMPT_MULTI_ROUND_TG_OPENO3_VERSION,
        default_max_turn=0,
        fallback_mode="think_then_answer",
        tool_format="qwen3_5",
        runtime_kwargs={"max_num_seqs": 1},
    ),
    "qwen3.5-35b-a3b-sft-135k": VideoO3ModelConfig(
        env_key="QWEN3_5_35B_A3B_135K_SFT_PATH",
        model_path="models/qwen3.5-35b-a3b-sft-135k",
        min_pixels=20 * 32 * 32,
        max_pixels=768 * 32 * 32,
        total_pixels=16384 * 32 * 32,
        use_tools=True,
        use_dynamic_quota=True,
        system_prompt=SYSTEM_PROMPT_MULTI_ROUND_MC_V03,
        tg_system_prompt=SYSTEM_PROMPT_MULTI_ROUND_TG_OPENO3_VERSION,
        default_max_turn=0,
        fallback_mode="think_then_answer",
        tool_format="qwen3_5",
        runtime_kwargs={"max_num_seqs": 1, "torch_dtype": "bfloat16"},
    ),
    "openo3videov2_qwen35_35b_rl": VideoO3ModelConfig(
        env_key="OPENO3VIDEOV2_QWEN35_35B_RL_MODEL_PATH",
        model_path="models/openo3videov2_qwen35_35b_rl",
        min_pixels=20 * 32 * 32,
        max_pixels=768 * 32 * 32,
        total_pixels=16384 * 32 * 32,
        use_tools=True,
        use_dynamic_quota=True,
        system_prompt=SYSTEM_PROMPT_MULTI_ROUND_MC_V03,
        tg_system_prompt=SYSTEM_PROMPT_MULTI_ROUND_TG_OPENO3_VERSION,
        default_max_turn=0,
        fallback_mode="think_then_answer",
        tool_format="qwen3_5",
        runtime_kwargs={"max_num_seqs": 1, "torch_dtype": "bfloat16"},
    ),
}
