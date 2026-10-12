import pytest

from openo3_eval.video_o3.frames import frame_timestamps_from_meta
from openo3_eval.vl_utils.video_o3_vision import _sample_full_video_indices_for_video_o3


def test_video_o3_capped_sampling_matches_reference_eval_policy():
    indices, sample_fps, duration = _sample_full_video_indices_for_video_o3(
        n_frames=67612,
        raw_fps=30.0,
        fps=1.0,
        frames_limit=96,
    )

    assert len(indices) == 96
    assert indices[:3] == [697, 1394, 2091]
    assert indices[-3:] == [65520, 66217, 66914]
    assert sample_fps == 96 / duration


def test_video_o3_short_video_sampling_uses_first_frame():
    indices, _, _ = _sample_full_video_indices_for_video_o3(
        n_frames=12,
        raw_fps=30.0,
        fps=1.0,
        frames_limit=96,
    )

    assert indices == [0]


def test_video_o3_frame_timestamps_use_real_frame_indices_when_available():
    timestamps = frame_timestamps_from_meta(
        {
            "raw_fps": 30.0,
            "sample_fps": 96 / (67612 / 30.0),
            "indices": [697, 1394, 66914],
            "duration": 67612 / 30.0,
        },
        frame_count=3,
    )

    assert timestamps == pytest.approx([697 / 30.0, 1394 / 30.0, 66914 / 30.0])
