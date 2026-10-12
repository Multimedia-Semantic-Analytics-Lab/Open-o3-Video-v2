from typing import Any

import numpy as np
import torch

from openo3_eval.vl_utils.vision_process import _read_video_torchvision


def test_read_video_torchvision_returns_numpy_hwc(monkeypatch: Any) -> None:
    video = torch.arange(4 * 3 * 2 * 2, dtype=torch.uint8).reshape(4, 3, 2, 2)

    def fake_read_video(*args: Any, **kwargs: Any) -> Any:
        return video, None, {"video_fps": 2.0}

    monkeypatch.setattr("openo3_eval.vl_utils.vision_process.io.read_video", fake_read_video)

    frames, metadata, fps = _read_video_torchvision({"video": "demo.mp4", "fps": 1.0})

    assert isinstance(frames, np.ndarray)
    assert frames.shape == (2, 2, 2, 3)
    assert metadata == {
        "fps": 2.0,
        "frames_indices": [0, 2],
        "total_num_frames": 4,
        "video_backend": "torchvision",
    }
    assert fps == 1.0
    np.testing.assert_array_equal(frames[0], video[0].permute(1, 2, 0).numpy())
    np.testing.assert_array_equal(frames[1], video[2].permute(1, 2, 0).numpy())
