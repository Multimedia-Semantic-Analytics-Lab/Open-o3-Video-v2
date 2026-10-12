from argparse import Namespace

from openo3_eval.pipelines.video_o3_eval import _build_model
from openo3_eval.video_o3.config import VIDEO_O3_MODEL_CONFIGS
from openo3_eval.video_o3.message import build_video_o3_message
from openo3_eval.video_o3.model_multi_image import VideoO3MultiImageChat
from openo3_eval.video_o3.prompts import SYSTEM_PROMPT_OPENO3_SINGLE_TURN


def test_open_o3_video_qwen25vl_config_matches_reference() -> None:
    config = VIDEO_O3_MODEL_CONFIGS["openo3video_qwen25vl_7b"]

    assert config.model_path == "models/openo3video_qwen25vl_7b"
    assert config.env_key == "OPENO3VIDEO_QWEN25VL_7B_MODEL_PATH"
    assert config.default_max_turn == 0
    assert config.fallback_mode == "think_then_answer"
    assert config.single_turn_system_prompt == SYSTEM_PROMPT_OPENO3_SINGLE_TURN
    assert config.frame_prompt_template == "Frame {frame_index} at {timestamp:.1f}s: <image>\n"
    assert config.runtime_kwargs == {"torch_dtype": "bfloat16"}
    assert config.pixel_overrides == {"v_star": {"max_pixels": 2 * 1024 * 1024}}


def test_open_o3_video_uses_reference_single_turn_system_prompt() -> None:
    model = object.__new__(VideoO3MultiImageChat)
    model.single_turn_system_prompt = SYSTEM_PROMPT_OPENO3_SINGLE_TURN
    model.fallback_mode = "think_then_answer"

    assert model._fallback_system_prompt_for_question_type("mc") == SYSTEM_PROMPT_OPENO3_SINGLE_TURN
    assert model._fallback_system_prompt_for_question_type("oe") == SYSTEM_PROMPT_OPENO3_SINGLE_TURN
    assert model._fallback_system_prompt_for_question_type("tg") == SYSTEM_PROMPT_OPENO3_SINGLE_TURN


def test_open_o3_video_uses_reference_frame_labels(monkeypatch) -> None:
    monkeypatch.setattr(
        "openo3_eval.video_o3.message.sample_full_video_frames",
        lambda **kwargs: (["fixtures/frame.jpg"], {"sample_fps": 1.0, "sample_n_frames": 1, "duration": 4.0}),
    )
    args = Namespace(
        model="openo3video_qwen25vl_7b",
        dataset_pp="videommmu",
        fps=1.0,
        frames_limit=64,
    )
    item = {
        "videos": ["fixtures/video.mp4"],
        "pp_meta": {
            "gt_question": "What happens?",
            "gt_options_text": "A. One\nB. Two",
        },
    }

    message, _ = build_video_o3_message(args, item, "videommmu")

    frame_labels = [entry["value"] for entry in message if entry.get("type") == "text" and "<image>" in entry["value"]]
    assert frame_labels == ["Frame 1 at 0.0s: <image>\n"]


def test_open_o3_video_config_default_keeps_zero_max_turn(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("openo3_eval.pipelines.video_o3_eval.VideoO3MultiImageChat", fake_model)
    monkeypatch.setenv("OPENO3VIDEO_QWEN25VL_7B_MODEL_PATH", str(tmp_path))
    args = Namespace(
        model="openo3video_qwen25vl_7b",
        max_turn=None,
        fps=1.0,
        max_new_tokens=2048,
        temperature=0.0,
        top_p=0.001,
        top_k=1,
        repetition_penalty=1.05,
        use_tools=True,
        do_sample=False,
    )

    _build_model(args)

    assert captured["max_turn"] == 0


def test_open_o3_video_qwen25vl_legacy_8b_alias_uses_7b_defaults(monkeypatch) -> None:
    monkeypatch.delenv("OPENO3VIDEO_QWEN25VL_7B_MODEL_PATH", raising=False)
    monkeypatch.setenv("OPENO3VIDEO_QWEN25VL_8B_MODEL_PATH", "fixtures/legacy-qwen25vl-path")

    config = VIDEO_O3_MODEL_CONFIGS["openo3video_qwen25vl_8b"]

    assert config.default_max_turn == 0
    assert config.single_turn_system_prompt == SYSTEM_PROMPT_OPENO3_SINGLE_TURN
    assert VIDEO_O3_MODEL_CONFIGS["openo3video_qwen25vl_7b"].to_kwargs()["model_path"] == "fixtures/legacy-qwen25vl-path"


def test_open_o3_video_v1_sft_config_uses_qwen25vl_7b_single_turn_defaults() -> None:
    config = VIDEO_O3_MODEL_CONFIGS["openo3video_v1_sft"]

    assert config.env_key == "OPENO3VIDEO_V1_SFT_MODEL_PATH"
    assert config.model_path == "models/openo3video_v1_sft"
    assert config.default_max_turn == 0
    assert config.fallback_mode == "think_then_answer"
    assert config.single_turn_system_prompt == SYSTEM_PROMPT_OPENO3_SINGLE_TURN
