from types import SimpleNamespace

from openo3_eval.datasetpp.videozerobench import VideoZeroBench


def _args(mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_dir="",
        videozerobench_mode=mode,
        model="stub-model",
        fps=1.0,
        frames_limit=32,
    )


def _item(with_boxes: bool = True):
    return {
        "videos": ["fixtures/fake.mp4"],
        "pp_meta": {
            "gt_question": "How many people are in the room?",
            "gt_answer": "8",
            "gt_question_id": 1,
            "gt_video": "fake.mp4",
            "gt_video_id": "fake",
            "gt_category": "Daily Vlogs",
            "gt_language": "en",
            "gt_duration": 10.0,
            "gt_evidence_span": "short-term",
            "gt_capabilities": ["counting"],
            "gt_evidence_windows": [{"start": 1.0, "end": 3.0}],
            "gt_evidence_boxes": ([{"time": 2.0, "box": [0.1, 0.1, 0.4, 0.4]}] if with_boxes else []),
        },
    }


def test_videozerobench_task_building_full_levels(monkeypatch):
    captured_kwargs = []

    def _stub(*args, **kwargs):
        captured_kwargs.append(kwargs)
        return [{"type": "text", "value": "stub"}], {"duration": 10.0}

    monkeypatch.setattr(
        "openo3_eval.datasetpp.videozerobench.build_video_o3_custom_message",
        _stub,
    )
    dataset = VideoZeroBench(_args("full_levels"))

    tasks = dataset.build_video_o3_tasks(_args("full_levels"), _item())

    assert [task["task_name"] for task in tasks] == ["level-1", "level-2", "level-3", "level-4", "level-5"]
    assert all(not task["skip"] for task in tasks)
    assert [task.get("max_turn") for task in tasks] == [None, None, None, 0, 0]
    level4_kwargs = captured_kwargs[3]
    assert level4_kwargs["question_type"] == "tg"
    assert "<t>start_time</t>s to <t>end_time</t>s" in level4_kwargs["direct_prompt"]
    assert "<t>start_timestamp_1</t>s to <t>end_timestamp_1</t>s" in level4_kwargs["prompt_text"]


def test_videozerobench_task_building_level3_only(monkeypatch):
    monkeypatch.setattr(
        "openo3_eval.datasetpp.videozerobench.build_video_o3_custom_message",
        lambda *args, **kwargs: ([{"type": "text", "value": "stub"}], {"duration": 10.0}),
    )
    dataset = VideoZeroBench(_args("level_3_only"))

    tasks = dataset.build_video_o3_tasks(_args("level_3_only"), _item())

    assert len(tasks) == 1
    assert tasks[0]["task_name"] == "level-3"
    assert not tasks[0]["skip"]
    assert "max_turn" not in tasks[0]


def test_videozerobench_full_level_metrics():
    dataset = VideoZeroBench(_args("full_levels"))
    rows = [
        {
            "pp_meta": _item()["pp_meta"],
            "videozerobench_predictions": {
                "level-1": {"model_answer": "<answer>8</answer>"},
                "level-2": {"model_answer": "<answer>8</answer>"},
                "level-3": {"model_answer": "<answer>8</answer>"},
                "level-4": {"model_answer": "<answer>From <t>1.0</t>s to <t>3.0</t>s</answer>"},
                "level-5": {"model_answer": '<answer>[{"time": 2.0, "bbox_2d": [[100, 100, 400, 400]]}]</answer>'},
            },
        }
    ]

    metrics = dataset.calc_metric(rows)

    assert metrics["Level-1_acc"] == 100.0
    assert metrics["Level-2_acc"] == 100.0
    assert metrics["Level-3_acc"] == 100.0
    assert metrics["Level-4_mean_tIoU"] == 100.0
    assert metrics["Level-4_score"] == 100.0
    assert metrics["Level-5_mean_vIoU"] == 100.0
    assert metrics["Level-5_score"] == 100.0


def test_videozerobench_level3_only_metrics():
    dataset = VideoZeroBench(_args("level_3_only"))
    rows = [{"pp_meta": _item()["pp_meta"], "response": "<answer>8</answer>"}]

    metrics = dataset.calc_metric(rows)

    assert metrics["Level-3_acc"] == 100.0
