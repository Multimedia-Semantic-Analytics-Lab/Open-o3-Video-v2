from openo3_eval.datasetpp.base import BaseDatasetPP
from openo3_eval.pipelines.video_o3_eval import _merge_eval_task_results, _run_task_with_model_overrides
from openo3_eval.util import build_video_o3_dialogue_trace_suffix, strip_video_o3_metadata


class _DummyModel:
    def __init__(self):
        self.max_turn = 5
        self.use_tools = True
        self.calls = []

    def generate(self, message, dataset=None):
        self.calls.append(
            {
                "message": message,
                "dataset": dataset,
                "max_turn": self.max_turn,
                "use_tools": self.use_tools,
            }
        )
        return "ok"


class _DummyDatasetPP(BaseDatasetPP):
    def build_dataset(self):
        return []

    def calc_metric(self, data_list):
        return {}


def test_run_task_with_model_overrides_applies_and_restores_model_state():
    model = _DummyModel()
    task = {"message": [{"type": "text", "value": "stub"}], "max_turn": 0, "use_tools": True}

    response = _run_task_with_model_overrides(model, task, dataset_name="videozerobench")

    assert response == "ok"
    assert model.calls == [
        {
            "message": [{"type": "text", "value": "stub"}],
            "dataset": "videozerobench",
            "max_turn": 0,
            "use_tools": True,
        }
    ]
    assert model.max_turn == 5
    assert model.use_tools is True


def test_strip_video_o3_metadata_extracts_dialogue_turns():
    dialogue_turns = [{"turn_index": 1, "prompt": "prompt", "response": "response"}]
    raw = (
        "final answer\n"
        "[TURN_COUNT: 2]\n"
        "[INFERENCE_TIME: 1.25s]\n"
        "[INFERENCE_TIME_GENERATE: 0.75s]"
        f"{build_video_o3_dialogue_trace_suffix(dialogue_turns)}"
    )

    cleaned, metadata = strip_video_o3_metadata(raw)

    assert cleaned == "final answer"
    assert metadata["turn_count"] == 2
    assert metadata["inference_time"] == 1.25
    assert metadata["inference_time_generate"] == 0.75
    assert metadata["dialogue_turns"] == dialogue_turns


def test_merge_eval_task_results_keeps_dialogue_turns_for_default_datasets():
    dataset_pp = _DummyDatasetPP(args=None)
    merged = _merge_eval_task_results(
        dataset_pp,
        item={},
        task_results=[
            {
                "task_name": "default",
                "response": "answer",
                "raw_response": "raw",
                "frame_meta": {"n_frames": 8},
                "turn_count": 2,
                "inference_time": 1.0,
                "inference_time_generate": 0.8,
                "dialogue_turns": [{"turn_index": 1, "prompt": "p", "response": "r"}],
            }
        ],
    )

    assert merged["response"] == "answer"
    assert merged["dialogue_turns"] == [{"turn_index": 1, "prompt": "p", "response": "r"}]
    assert merged["video_o3"]["dialogue_turns"] == [{"turn_index": 1, "prompt": "p", "response": "r"}]
