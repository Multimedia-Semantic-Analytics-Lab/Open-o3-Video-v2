import math
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from openo3_eval.external import eval_vstar
from openo3_eval.datasetpp.v_star import VStar, _extract_bounding_boxes


def test_v_star_build_dataset_loads_nested_video_tree(tmp_path: Path) -> None:
    root = tmp_path / "V-STAR"
    videos = root / "videos" / "subdir"
    videos.mkdir(parents=True)
    (videos / "123.mp4").write_bytes(b"")
    anno = root / "V_STaR_test.json"
    anno.write_text(
        '[{"vid":"123","question":"q","answer":"a","fps":24,"frame_count":48,"width":640,"height":360,'
        '"temporal_question":"tq","timestamps":[0,1],"spatial_question":"sq","spatial_question_2":"sq2","bboxes":[]}]',
        encoding="utf-8",
    )

    bench = VStar(Namespace(dataset_dir=str(root), result_path=None))
    dataset = bench.build_dataset()

    assert len(dataset) == 1
    assert dataset[0]["videos"] == [str(videos / "123.mp4")]
    assert dataset[0]["pp_meta"]["gt_question"] == "q"


def test_v_star_build_video_o3_tasks_force_direct_answer(monkeypatch) -> None:
    bench = VStar(Namespace(dataset_dir="", result_path=None))
    item = {
        "videos": ["fixtures/fake.mp4"],
        "question": "what",
        "temporal_question": "when",
        "spatial_question": "where",
        "spatial_question_2": "where2",
        "timestamps": [1.2, 3.8],
        "width": 640,
        "height": 360,
        "fps": 24,
        "frame_count": 240,
        "bboxes": [{"xmin": 64, "ymin": 36, "xmax": 128, "ymax": 72}],
    }

    captured = []
    monkeypatch.setattr(
        "openo3_eval.datasetpp.v_star.build_video_o3_custom_message",
        lambda *args, **kwargs: (captured.append(kwargs) or [{"type": "text", "value": kwargs["prompt_text"]}], {"stub": True}),
    )

    tasks = bench.build_video_o3_tasks(SimpleNamespace(max_turn=0), item)

    assert [task["task_name"] for task in tasks] == [
        "answer_vqa",
        "answer_temporal",
        "answer_spatial",
        "answer_spatial_2",
        "answer_temporal_2",
    ]
    assert all(task["use_tools"] is True for task in tasks)
    assert all(task["max_turn"] == 0 for task in tasks)
    assert all("direct_prompt" in kwargs and kwargs["direct_prompt"] for kwargs in captured)


def test_v_star_merge_video_o3_task_results_parses_outputs() -> None:
    bench = VStar(Namespace(dataset_dir="", result_path=None))
    item = {"width": 640, "height": 360}
    task_results = [
        {"task_name": "answer_vqa", "response": "<answer>a phone</answer>"},
        {"task_name": "answer_temporal", "response": "<answer>From <t>1.5</t>s to <t>3.0</t>s</answer>"},
        {"task_name": "answer_spatial", "response": '<answer>{"2":[100,200,300,400]}</answer>'},
        {"task_name": "answer_spatial_2", "response": '<answer>{"3":[100,200,300,400]}</answer>'},
        {"task_name": "answer_temporal_2", "response": "<answer>From <t>4</t>s to <t>6</t>s</answer>"},
    ]

    merged = bench.merge_video_o3_task_results(item, task_results)

    assert merged["answer_vqa"] == "a phone"
    assert merged["answer_temporal"] == [1.5, 3.0]
    assert merged["answer_spatial"] == {"2": [64, 72, 192, 144]}
    assert merged["answer_spatial_2"] == {"3": [64, 72, 192, 144]}
    assert merged["answer_temporal_2"] == [4.0, 6.0]


def test_v_star_extract_bounding_boxes_recovers_missing_closing_brace() -> None:
    parsed = _extract_bounding_boxes('{"2":[100,200,300,400]', 640, 360)

    assert parsed == {"2": [64, 72, 192, 144]}


def test_v_star_calc_metric_runs_external_eval_and_parses_stdout(monkeypatch, tmp_path: Path) -> None:
    result_path = tmp_path / "result.jsonl"
    bench = VStar(Namespace(dataset_dir="", result_path=str(result_path)))
    sample_stdout = """
Overall Statistics:
VQA: Avg All Score: 2.5000, Avg Valid Score: 2.6000, Accuracy: 0.8000
Chain 1:
Temporal Answer: R1@IoU=0.3: 0.7000, R1@IoU=0.5: 0.6000, R1@IoU=0.7: 0.5000, Mean IoU: 0.4000
Spatial Answer: mAP@0.1: 0.3000, mAP@0.3: 0.2000, mAP@0.5: 0.1000, mAP@0.7: 0.0500, mAP@0.9: 0.0100, Mean mIoU: 0.2500

Chain 2:
Temporal Answer: R1@IoU=0.3: 0.7100, R1@IoU=0.5: 0.6100, R1@IoU=0.7: 0.5100, Mean IoU: 0.4100
Spatial Answer: mAP@0.1: 0.3100, mAP@0.3: 0.2100, mAP@0.5: 0.1100, mAP@0.7: 0.0600, mAP@0.9: 0.0200, Mean mIoU: 0.2600
AM1:0.4833, AM2:0.4900, mAM:0.4867
LGM1:0.5000, LGM2:0.5100, mLGM:0.5050
"""

    class _Completed:
        returncode = 0
        stdout = sample_stdout
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: _Completed())
    monkeypatch.setattr("os.path.exists", lambda path: True if str(path).endswith("eval_vstar.py") else Path(path).exists())

    metrics = bench.calc_metric([{"vid": "1", "answer_vqa": "x"}])

    assert metrics["Overall Statistics"]["vqa_accuracy"] == 0.8
    assert metrics["Overall Statistics"]["chain1_temporal_miou"] == 0.4
    assert metrics["Overall Statistics"]["chain2_spatial_miou"] == 0.26
    assert metrics["Overall Statistics"]["mam"] == 0.4867
    assert result_path.with_suffix(".vstar_eval.json").exists()


def test_v_star_external_eval_defaults_to_four_gpus(monkeypatch, tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text("[]", encoding="utf-8")
    bench = VStar(Namespace(dataset_dir="", result_path=None))
    captured = {}

    class _Completed:
        returncode = 0
        stdout = "Overall Statistics:\nVQA: Avg All Score: 0.0000, Avg Valid Score: 0.0000, Accuracy: 0.0000\n"
        stderr = ""

    def _run(cmd, capture_output, text, env):
        captured["env"] = env
        return _Completed()

    monkeypatch.setattr("subprocess.run", _run)
    monkeypatch.setattr("os.path.exists", lambda path: True if str(path).endswith("eval_vstar.py") else Path(path).exists())

    bench._run_external_eval(str(result_path))

    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_v_star_external_eval_lgm_handles_perfect_scores() -> None:
    lgm = eval_vstar._compute_lgm(1.0, 1.0, 1.0)

    assert math.isfinite(lgm)
    assert lgm > 0


def test_v_star_external_eval_repairs_spatial_predictions_from_pre_fields() -> None:
    repaired = eval_vstar._repair_spatial_prediction('{"2":[100,200,300,400]', 640, 360)

    assert repaired == {"2": [64, 72, 192, 144]}
