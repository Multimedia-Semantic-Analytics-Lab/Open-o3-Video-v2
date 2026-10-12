from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# This module is often loaded from a file path via `spec_from_file_location`,
# so the project root and `verl/` package root may be missing from `sys.path`.
_CUSTOM_REWARDS_DIR = Path(__file__).resolve().parent
_VERL_ROOT = _CUSTOM_REWARDS_DIR.parent
_RL_ROOT = _VERL_ROOT.parent
for _path in (_VERL_ROOT, _RL_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from custom_rewards.openo3_acc_reward import compute_openo3_acc, compute_videoo3_acc
from custom_rewards.openo3_format_reward import compute_format_reward
from custom_rewards.openo3_reward_common import (
    OPENO3_SOURCES,
    VIDEOO3_TOOL_SOURCES,
    clamp01,
    get_weight,
)
from custom_rewards.openo3_thinking_reward import compute_thinking_reward
from custom_rewards.openo3_tool_reward import compute_tool_reward


def _maybe_patch_metric_utils_for_thinking_reward() -> None:
    try:
        from verl.trainer.ppo import metric_utils as _metric_utils
    except Exception:
        return
    if getattr(_metric_utils, "_openo3_thinking_reward_patched", False):
        return

    original = _metric_utils.compute_reward_extra_metrics

    def _wrapped(batch):
        metrics = original(batch)
        if "thinking_reward_score" in batch.non_tensor_batch:
            try:
                arr = np.array(batch.non_tensor_batch["thinking_reward_score"], dtype=np.float32)
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    metrics["reward/thinking_reward/mean"] = float(arr.mean())
                    metrics["reward/thinking_reward/max"] = float(arr.max())
                    metrics["reward/thinking_reward/min"] = float(arr.min())
            except Exception:
                pass
        return metrics

    _metric_utils.compute_reward_extra_metrics = _wrapped
    _metric_utils._openo3_thinking_reward_patched = True


_maybe_patch_metric_utils_for_thinking_reward()


def compute_score(
    data_source: str,
    solution_str: Any,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> dict[str, Any]:
    del kwargs
    extra_info = extra_info or {}
    if data_source in OPENO3_SOURCES:
        acc_score = compute_openo3_acc(solution_str, ground_truth, data_source, extra_info)
    elif data_source in VIDEOO3_TOOL_SOURCES:
        acc_score = compute_videoo3_acc(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"openo3_reward has no handler for data_source={data_source!r}")

    format_reward_score = compute_format_reward(solution_str, data_source)
    thinking_reward_score = compute_thinking_reward(solution_str, ground_truth, data_source, extra_info)
    tool_reward_score = compute_tool_reward(solution_str, ground_truth, extra_info, ans_score=acc_score)

    ans_weight = get_weight(extra_info, "ans_reward_weight", 0.8)
    format_weight = get_weight(extra_info, "format_reward_weight", 0.2)
    thinking_weight = get_weight(extra_info, "thinking_reward_weight", 0.7)
    tool_weight = get_weight(extra_info, "tool_reward_weight", 1.0)

    score = (
        ans_weight * clamp01(acc_score)
        + format_weight * clamp01(format_reward_score)
        + thinking_weight * clamp01(thinking_reward_score)
        + tool_weight * clamp01(tool_reward_score)
    )
    return {
        "score": score,
        "acc_score": clamp01(acc_score),
        "format_reward_score": clamp01(format_reward_score),
        "thinking_reward_score": clamp01(thinking_reward_score),
        "tool_reward_score": clamp01(tool_reward_score),
        "predict_str": solution_str,
        "ground_truth": ground_truth,
    }


__all__ = [
    "OPENO3_SOURCES",
    "VIDEOO3_TOOL_SOURCES",
    "compute_score",
]
