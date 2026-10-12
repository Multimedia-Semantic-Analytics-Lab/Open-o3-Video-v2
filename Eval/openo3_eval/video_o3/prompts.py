from __future__ import annotations

# ---------------------------------------------------------------------------
# Main multi-turn system prompts
# These are the default tool-using prompts for the active multi-image evaluator.
# `config.py` selects them per model, and `model_multi_image.py` uses them as
# runtime fallbacks when the task type switches between MC / OE / TG.
# ---------------------------------------------------------------------------

_THINK_ANSWER_SYS_PROMPT_V03 = (
    "Output your reasoning process within the <think> and </think> tags.\n"
    "Your reasoning must be grounded in visual evidence from the video. Whenever you mention an object, person, or specific visual element that is relevant to the question, attach its visual grounding in the following format:\n"
    "<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s.\n"
    "During reasoning, adaptively decides whether additional visual observation is needed. If the currently observed evidence is insufficient, ambiguous, or requires a closer inspection of a specific video segment, you may request additional visual information by outputting:\n"
    "<tool_call>{\"name\": \"crop_video\", \"arguments\": {\"start_time\": ..., \"end_time\": ..., \"sampling_strategy\": \"medium\"}}</tool_call>\n"
    "Here, start_time and end_time are in integer seconds within the full video. The value of sampling_strategy must be one of [\"coarse\", \"medium\", \"fine\"]. Choose the sampling strategy adaptively according to the difficulty of the segment: use \"coarse\" for broad context, \"medium\" for normal inspection, and \"fine\" for detailed or ambiguous visual evidence.\n"
    "Do not call the crop_video tool unnecessarily. If the available visual evidence is already sufficient to answer the question confidently, directly integrate the observed clues and provide the answer.\n"
    "Once you have gathered enough visual evidence and can determine the correct answer, output the correct answer within the <answer> and </answer> tags."
)

SYSTEM_PROMPT_MULTI_ROUND_OE = (
    "You are a helpful assistant. Answer the user's question based on the provided video.\n"
    + _THINK_ANSWER_SYS_PROMPT_V03
    + "The following is a example of the thinking process and the final answer:\n"
    " - When you want to examine a specific clue segment of a video more closely, produce exactly: '<think>your thinking process here</think>\n"
    "<tool_call>{\"name\": \"crop_video\", \"arguments\": {\"start_time\": START_TIMESTAMP, \"end_time\": END_TIMESTAMP, \"sampling_strategy\": \"medium\"}}</tool_call>\n"
    "'.\n"
    " - When you believe the current information is sufficient to give an answer, produce exactly: '<think>your thinking process here</think>\n"
    "<answer>Your Answer</answer>\n"
    "'."
)

SYSTEM_PROMPT_MULTI_ROUND_MC_V03 = (
    "You are a helpful assistant. Answer the user's multiple-choice question based on the provided video.\n"
    + _THINK_ANSWER_SYS_PROMPT_V03
    + "The following is a example of the thinking process and the final answer:\n"
    " - When you want to examine a specific clue segment of a video more closely, produce exactly: '<think>your thinking process here</think>\n"
    "<tool_call>{\"name\": \"crop_video\", \"arguments\": {\"start_time\": START_TIMESTAMP, \"end_time\": END_TIMESTAMP, \"sampling_strategy\": \"medium\"}}</tool_call>\n"
    "'.\n"
    " - When you believe the current information is sufficient to give an answer, produce exactly: '<think>your thinking process here</think>\n"
    "<answer>Option Letter</answer>\n"
    "'."
)

SYSTEM_PROMPT_MULTI_ROUND_TG_OPENO3_VERSION = (
    "A conversation between user and assistant. The user provides a video and asks a question, and the "
    "Assistant determines the precise time period that answers the question. The assistant MUST first "
    "think about the reasoning process in the mind and then provide the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. "
    "The answer must strictly follow the following format: "
    "`From <t>start_time</t>s to <t>end_time</t>s`."
)


# ---------------------------------------------------------------------------
# Fallback system prompts
# These are only used when the tool loop must stop and the model needs to
# answer directly from the frames it has already seen.
# `fallback_mode="direct_answer"` uses the DIRECT prompts.
# `fallback_mode="think_then_answer"` uses the THINK prompts.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_DIRECT_ANSWER_MC = """You are a helpful assistant. The user provides uniformly sampled frames from a video and a multiple-choice question.
Answer immediately and only output `<answer>OPTION_LETTER</answer>`.
"""

SYSTEM_PROMPT_DIRECT_ANSWER_OE = """You are a helpful assistant. The user provides uniformly sampled frames from a video and a question.
Answer immediately and only output `<answer>YOUR_ANSWER</answer>`.
"""

SYSTEM_PROMPT_DIRECT_ANSWER_TG = """You are a helpful assistant. The user provides uniformly sampled frames from a video and a temporal grounding query.
Answer immediately and only output `<answer>From <t>start_time</t>s to <t>end_time</t>s</answer>`.
"""

SYSTEM_PROMPT_THINK_ANSWER_MC = """You are a helpful assistant. The user provides uniformly sampled frames from a video and a multiple-choice question.
First reason within `<think>` and `</think>`, then answer with only `<answer>OPTION_LETTER</answer>`.
"""

SYSTEM_PROMPT_THINK_ANSWER_OE = """You are a helpful assistant. The user provides uniformly sampled frames from a video and a question.
First reason within `<think>` and `</think>`, then answer with only `<answer>YOUR_ANSWER</answer>`.
"""

SYSTEM_PROMPT_THINK_ANSWER_TG = """You are a helpful assistant. The user provides uniformly sampled frames from a video and a temporal grounding query.
First reason within `<think>` and `</think>`, then answer with only `<answer>From <t>start_time</t>s to <t>end_time</t>s</answer>`.
"""


SYSTEM_PROMPT_OPENO3_SINGLE_TURN = (
    "A conversation between user and assistant. The user provides a video and asks a question, and the "
    "Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then "
    "provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> "
    "and <answer> </answer> tags, respectively. All reasoning must be grounded in visual evidence from the "
    "video. When you mention any related object, person, or specific visual element in the reasoning process, "
    "you must strictly follow the following format: "
    "`<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. The answer part only requires "
    "a text response; tags like <obj>, <box>, <t> are not needed."
)


# ---------------------------------------------------------------------------
# Observation-turn prompts
# These are appended during the multi-turn tool loop to describe the initial
# frame sample, follow-up cropped observations, and forced-final-answer states.
# ---------------------------------------------------------------------------

MULTI_IMAGE_INITIAL_PROMPT = (
    "Here are uniformly sampled frames from the original full video (Observation 0). "
    "The video spans 0.0s to {max_duration:.1f}s and the current frame set contains "
    "{total_frames} frames sampled at approximately {sample_fps:.2f} fps.\n"
)

TOOL_CALL_CROP_FRAMES_PROMPT = (
    "After the above Action {action_turn}, here is the refined observation (Observation {observation_turn}). "
    "These frames correspond to the original video segment from {start_time:.1f}s to {end_time:.1f}s, "
    "with approximately {total_frames} sampled frames at {sample_fps:.2f} fps.\n"
    "Continue your reasoning process inside <think> and </think>. "
    "If needed, you can keep selecting temporal segments from the original video by outputting <tool_call> "
    "and </tool_call> as before. Once you are ready to provide the final answer, put it inside <answer> "
    "and </answer>.\n"
)

FORCE_FINAL_ANSWER_PROMPT_FRAMES = (
    "You have reached the final turn. Based on all previous observations, you MUST provide the final answer now. "
    "Continue your reasoning only inside <think> and </think>, and then output the final answer inside "
    "<answer> and </answer>. Do NOT request another temporal segment and do NOT output <tool_call>.\n"
)

ERROR_INFO_MULTI_TURN_PROMPT = (
    "ERROR occurs during tool calling. Error Information: {error_info}. Please analyze the error information obtained "
    "from the function tool and adjust your response. Continue your reasoning process inside <think> and </think> "
    "and follow the instructions strictly."
)

FORCE_FINAL_ANSWER_PROMPT_ERROR_INFO = (
    "ERROR occurs during tool calling. Error Information: {error_info}. Continue your reasoning process inside "
    "<think> and </think> and follow the instructions below strictly.\n"
    "You have reached the final turn and MUST provide the final answer based on all the previous conversation and "
    "observations. Think carefully about all the information you have gathered inside <think> and </think> and give "
    "your best answer inside <answer> and </answer>. Do NOT request another temporal segment and do NOT output <tool_call>."
)


# ---------------------------------------------------------------------------
# First-turn user prompt templates
# `message.py` uses these to build the initial user turn that accompanies the
# uniformly sampled frame set.
# ---------------------------------------------------------------------------

QUESTION_TEMPLATE_MC = """
Answer the following multiple-choice question according to the content of the video:
{question}
Options:
{options}
You are advised to first observe potential clue segments, use '<think>YOUR THINK</think>\n<tool_call>{{"name": "crop_video", "arguments": {{"start_time": START_TIMESTAMP, "end_time": END_TIMESTAMP, "sampling_strategy": "medium"}}}}</tool_call>\n' to specify the segment you want to observe in detail.
If the evidence is visible in the original video for a long enough time and is clear enough to support a confident and correct answer, you may answer directly.
"""

QUESTION_TEMPLATE_OE = """
Answer the following question according to the content of the video:
{question}
You are advised to first observe potential clue segments, use '<think>YOUR THINK</think>\n<tool_call>{{"name": "crop_video", "arguments": {{"start_time": START_TIMESTAMP, "end_time": END_TIMESTAMP, "sampling_strategy": "medium"}}}}</tool_call>\n' to specify the segment you want to observe in detail.
If the evidence is visible in the original video for a long enough time and is clear enough to support a confident and correct answer, you may answer directly.
"""

QUESTION_TEMPLATE_TG = """
Please analyze the provided video and locate the video segment according to the given query.
Query: {question}
"""


# ---------------------------------------------------------------------------
# Single-turn user prompt builders
# These produce the explicit user prompt variants stored in message metadata:
# - `direct_prompt`: answer immediately with no further tool use
# - `think_answer_prompt`: think first, then answer without tool use
# `datasetpp/v_star.py` also reuses the direct variants for rescoring tasks.
# ---------------------------------------------------------------------------

def build_direct_answer_prompt(question: str, options_text: str) -> str:
    return (
        "\nAnswer the following multiple-choice question according to the content of the video:\n"
        f"{question}\n"
        "Options:\n"
        f"{options_text}\n"
        "Do not think out loud. Do not request more observations. Output only `<answer>OPTION_LETTER</answer>`.\n"
    )


def build_direct_answer_prompt_oe(question: str) -> str:
    return (
        "\nAnswer the following question according to the content of the video:\n"
        f"{question}\n"
        "Do not think out loud. Do not request more observations. Output only `<answer>YOUR_ANSWER</answer>`.\n"
    )


def build_direct_temporal_grounding_prompt(question: str) -> str:
    return (
        "\nPlease analyze the provided video frames and locate the video segment according to the given query.\n"
        f"Query: {question}\n"
        "Do not think out loud. Do not request more observations. "
        'Output only `<answer>From <t>start_time</t>s to <t>end_time</t>s</answer>`.\n'
    )


def build_think_answer_prompt(question: str, options_text: str) -> str:
    return (
        "\nAnswer the following multiple-choice question according to the content of the video:\n"
        f"{question}\n"
        "Options:\n"
        f"{options_text}\n"
        "First reason inside `<think>` and `</think>`, then output the final answer as `<answer>OPTION_LETTER</answer>`."
    )


def build_think_answer_prompt_oe(question: str) -> str:
    return (
        "\nAnswer the following question according to the content of the video:\n"
        f"{question}\n"
        "First reason inside `<think>` and `</think>`, then output the final answer as `<answer>YOUR_ANSWER</answer>`."
    )


def build_think_answer_temporal_grounding_prompt(question: str) -> str:
    return (
        "\nPlease analyze the provided video frames and locate the video segment according to the given query.\n"
        f"Query: {question}\n"
        "First reason inside `<think>` and `</think>`, then output the final answer as "
        '`<answer>From <t>start_time</t>s to <t>end_time</t>s</answer>`.'
    )
