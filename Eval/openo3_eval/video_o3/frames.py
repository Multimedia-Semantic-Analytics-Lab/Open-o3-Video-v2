from __future__ import annotations

from typing import Any, Dict, List, Tuple


def sample_full_video_frames(
    *,
    video_path: str,
    cache_dir: str,
    fps: float = 1.0,
    frames_limit: int = 768,
) -> Tuple[List[str], Dict[str, Any]]:
    """Uniformly sample frames from a full video using OpenO3's policy."""
    from openo3_eval.vl_utils.video_o3_vision import sample_video_frames_for_video_o3

    return sample_video_frames_for_video_o3(
        video_path=video_path,
        cache_dir=cache_dir,
        fps=fps,
        frames_limit=frames_limit,
    )


def frame_timestamps_from_meta(frame_meta: Dict[str, Any], frame_count: int) -> List[float]:
    """Recover per-frame timestamps (in seconds) from cache metadata."""
    raw_fps = float(frame_meta.get("raw_fps") or 0)
    indices = frame_meta.get("indices") or []
    if raw_fps > 0 and len(indices) == frame_count:
        return [float(index) / raw_fps for index in indices]

    sample_fps = float(frame_meta.get("sample_fps") or 0)
    if sample_fps > 0:
        timestamps = [idx / sample_fps for idx in range(frame_count)]
        duration = frame_meta.get("duration")
        if duration is not None:
            duration = float(duration)
            return [min(timestamp, duration) for timestamp in timestamps]
        return timestamps

    return [0.0 for _ in range(frame_count)]
